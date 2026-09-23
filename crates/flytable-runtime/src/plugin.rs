//! Local certified plugin seat [`LocalPluginSeatAgent`].
//!
//! Wraps `flytable-inference-host::SubprocessHost`: sends the current events and the
//! rule-enumerated legal actions to the translator subprocess (`flya-inference-v2`,
//! with frozen v1), receives an index into the legal list, and maps it back to a
//! FlyTable action.
//!
//! No observation encoding or action masks here (those stay in the model package or
//! translator); this seat only carries the protocol. The host checks the returned
//! index (digest echo and exact legal list match); out-of-range, abstain or errors
//! fall back to the first legal discard or Pass and are counted in the status.

use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use flytable_seat::contract::{InferenceDecision, RuleLine};
use flytable_table::{ReactionAction, TurnAction};

use flytable_inference_host::host::{DecisionPhase, EngineProcessConfig, SubprocessHost};
use flytable_inference_host::registry::CertifiedPluginRuntime;

use crate::agent::{
    inference_error_code, AgentChoice, ReactionRequest, SeatAgent, SeatAgentKind, SeatAgentStatus,
    TurnRequest,
};
use crate::match_host::{fallback_turn, MatchKind};
use crate::variant::Variant;

/// Builds the `match_context` sent to the translator (same shape in the hello
/// handshake and every infer call; see `flytable-inference-host::host`).
///
/// `length` (single / east / half) is a match-level setting that affects a model's
/// endgame strategy and cannot be inferred from events, so it must be sent
/// explicitly (hard-coding `"east"` would mislead a model in a hanchan). Dynamic state
/// such as round wind, honba, sticks and scores already comes with the events, so
/// only `rule_line` and `length` are sent here, and no observations or tensors.
fn match_context_json(rule_line: &str, length: MatchKind) -> serde_json::Value {
    serde_json::json!({
        "rule_line": rule_line,
        "length": length.as_str(),
    })
}

/// Local plugin seat. One instance owns one subprocess of a certified plugin runtime
/// (one `(session, model, seat, rule_line)`).
pub struct LocalPluginSeatAgent {
    model_id: String,
    host: SubprocessHost,
    calls: u64,
    fallbacks: u64,
    errors: u64,
    last_latency_ms: u64,
}

impl LocalPluginSeatAgent {
    /// Starts a plugin seat from a runtime certified by the registry (the recommended
    /// path: get `runtime` from `certified_plugin_runtimes` / [`crate::catalog`]).
    pub fn from_runtime(
        seat: u8,
        runtime: &CertifiedPluginRuntime,
        session_id: impl Into<String>,
        length: MatchKind,
    ) -> Result<Self> {
        let rule_line = parse_rule_line(&runtime.rule_line)?;
        let mut config =
            EngineProcessConfig::new(runtime.launch_cmd.clone(), seat, rule_line, session_id);
        config.args = runtime.launch_args.clone();
        config.cwd = Some(runtime.plugin_dir.clone());
        config.protocol_versions = vec![runtime.protocol.clone().into()];
        // Use the real match length rather than a hard-coded `"east"`.
        config.match_context = match_context_json(&runtime.rule_line, length);
        Self::from_config(runtime.model_id.clone(), config)
    }

    /// Variant-aware constructor for runtime consumers. This keeps the public
    /// MatchHost path from accidentally seating a 3p plugin in a 4p game, or
    /// vice versa. The lower-level [`Self::from_runtime`] remains available for
    /// tools that already validated the rule line.
    pub fn from_runtime_for_variant<V: Variant>(
        seat: u8,
        runtime: &CertifiedPluginRuntime,
        session_id: impl Into<String>,
        length: MatchKind,
    ) -> Result<Self> {
        let expected = V::RULE_LINE.wire_value();
        if runtime.rule_line != expected {
            return Err(anyhow!(
                "plugin {} rule_line {} does not match match rule_line {}",
                runtime.model_id,
                runtime.rule_line,
                expected
            ));
        }
        Self::from_runtime(seat, runtime, session_id, length)
    }

    /// Starts directly from an [`EngineProcessConfig`] (without registry certification).
    pub fn from_config(model_id: String, config: EngineProcessConfig) -> Result<Self> {
        let host = SubprocessHost::start(config)
            .map_err(|e| anyhow!("start plugin host {model_id}: {:?}: {}", e.kind, e.detail))?;
        Ok(Self {
            model_id,
            host,
            calls: 0,
            fallbacks: 0,
            errors: 0,
            last_latency_ms: 0,
        })
    }
}

impl<V: Variant> SeatAgent<V> for LocalPluginSeatAgent {
    fn decide_turn(&mut self, req: &TurnRequest<V>) -> AgentChoice<TurnAction> {
        self.calls += 1;
        // Fully qualified legal actions (same as the host's certification smoke test).
        let legal_actions: Vec<V::LegalAction> = req
            .legal
            .iter()
            .map(|a| V::turn_action_to_legal(req.view, a))
            .collect();
        let mut events = Vec::with_capacity(req.events.len() + 1);
        events.push(V::start_game_event());
        events.extend(req.events.iter().cloned());

        let started = Instant::now();
        let result = V::host_infer(
            &mut self.host,
            req.decision_id.to_string(),
            DecisionPhase::Discard,
            &events,
            &legal_actions,
            req.wall_remaining,
        );
        let latency = started.elapsed().as_millis() as u64;
        self.last_latency_ms = latency;

        match result {
            Ok(InferenceDecision::Select { index, .. }) if index < req.legal.len() => {
                AgentChoice::ok(req.legal[index].clone(), latency)
            }
            Ok(InferenceDecision::Select { index, .. }) => {
                self.fallbacks += 1;
                AgentChoice::fallback(
                    fallback_turn(req.legal),
                    latency,
                    Some(format!("index:{index}")),
                    "MODEL_ACTION_ID_OUT_OF_RANGE",
                    format!("out_of_range_index:{index}"),
                )
            }
            Ok(InferenceDecision::Abstain) => {
                self.fallbacks += 1;
                AgentChoice::fallback(
                    fallback_turn(req.legal),
                    latency,
                    Some("abstain".to_string()),
                    "MODEL_ABSTAIN",
                    "abstain",
                )
            }
            Err(err) => {
                self.fallbacks += 1;
                self.errors += 1;
                AgentChoice::fallback(
                    fallback_turn(req.legal),
                    latency,
                    None,
                    inference_error_code(err.kind),
                    format!("{:?}", err.kind),
                )
            }
        }
    }

    fn decide_reaction(&mut self, req: &ReactionRequest<V>) -> AgentChoice<ReactionAction> {
        self.calls += 1;
        // `choices[0]` is Pass (pass_all); the rest correspond to `legal_actions` one to one.
        let mut choices: Vec<ReactionAction> = Vec::with_capacity(req.legal.len() + 1);
        let mut legal_actions: Vec<V::LegalAction> = Vec::with_capacity(req.legal.len() + 1);
        let has_ron = req.legal.iter().any(|a| matches!(a, ReactionAction::Ron));
        let has_call = req
            .legal
            .iter()
            .any(|a| !matches!(a, ReactionAction::Ron | ReactionAction::Pass));
        choices.push(ReactionAction::Pass);
        legal_actions.push(V::pass_all(has_ron, has_call));
        for action in req.legal {
            if let Some(legal) = V::reaction_to_legal(req.view, action, req.discarder, req.tile) {
                choices.push(action.clone());
                legal_actions.push(legal);
            }
        }

        let mut events = Vec::with_capacity(req.events.len() + 1);
        events.push(V::start_game_event());
        events.extend(req.events.iter().cloned());

        let started = Instant::now();
        let result = V::host_infer(
            &mut self.host,
            req.decision_id.to_string(),
            DecisionPhase::Response,
            &events,
            &legal_actions,
            req.wall_remaining,
        );
        let latency = started.elapsed().as_millis() as u64;
        self.last_latency_ms = latency;

        match result {
            Ok(InferenceDecision::Select { index, .. }) if index < choices.len() => {
                AgentChoice::ok(choices[index].clone(), latency)
            }
            Ok(InferenceDecision::Select { index, .. }) => {
                // Out of range or abstain: pass safely (never invent actions outside the rules).
                self.fallbacks += 1;
                AgentChoice::fallback(
                    ReactionAction::Pass,
                    latency,
                    Some(format!("index:{index}")),
                    "MODEL_ACTION_ID_OUT_OF_RANGE",
                    "out_of_range_pass",
                )
            }
            Ok(InferenceDecision::Abstain) => {
                self.fallbacks += 1;
                AgentChoice::fallback(
                    ReactionAction::Pass,
                    latency,
                    Some("abstain".to_string()),
                    "MODEL_ABSTAIN",
                    "abstain_pass",
                )
            }
            Err(err) => {
                self.fallbacks += 1;
                self.errors += 1;
                AgentChoice::fallback(
                    ReactionAction::Pass,
                    latency,
                    None,
                    inference_error_code(err.kind),
                    format!("{:?}", err.kind),
                )
            }
        }
    }

    fn kind(&self) -> SeatAgentKind {
        SeatAgentKind::LocalPlugin
    }

    fn model_id(&self) -> String {
        self.model_id.clone()
    }

    fn status(&self) -> SeatAgentStatus {
        SeatAgentStatus {
            kind: SeatAgentKind::LocalPlugin.as_str().to_string(),
            model_id: self.model_id.clone(),
            calls: self.calls,
            fallbacks: self.fallbacks,
            errors: self.errors,
            last_latency_ms: Some(self.last_latency_ms),
        }
    }
}

fn parse_rule_line(value: &str) -> Result<RuleLine> {
    match value {
        "riichi4p" => Ok(RuleLine::Riichi4p),
        "riichi3p" => Ok(RuleLine::Riichi3p),
        other => {
            Err(anyhow::anyhow!("unknown rule_line {other:?}")).context("plugin runtime rule_line")
        }
    }
}
