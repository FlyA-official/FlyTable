//! The remaining seat kinds (remote_model / observed_external / observer) and manual
//! seats: shapes and placeholder implementations.
//!
//! There are six seat kinds. algorithm and local_plugin are fully implemented in their
//! own modules; this module provides implementations or placeholders for the other
//! four that do not block orchestration:
//! - [`ManualSeat`] - human seat. It never fakes random or algorithmic actions; when
//!   asked it returns an explicit "not ready" marker, and the product injects the real
//!   human action. Headless orchestration records each move as
//!   `fallback=manual_not_ready`.
//! - [`RemoteModelConfig`] / [`RemoteModelError`] / [`RemoteModelSeat`]: remote model.
//!   Networking is delegated to `flytable-inference-host::RemoteHttpHost`; the runtime
//!   only handles seat semantics and safe fallback.
//! - [`ObservedInput`] / [`ObservedSourceStatus`]: input and status shapes for
//!   externally observed seats. Shapes only; the actions that can actually be taken
//!   come from the platform's `possible_actions` (the runtime does not override them).
//! - [`ObserverConfig`] - delegation config for recommend-only seats (recommendation
//!   panels and the like).

use std::time::Instant;

use flytable_seat::contract::InferenceDecision;
use flytable_table::{ReactionAction, TurnAction};

use flytable_inference_host::host::{DecisionPhase, RemoteHttpConfig, RemoteHttpHost};

use crate::agent::{
    inference_error_code, AgentChoice, ReactionRequest, SeatAgent, SeatAgentKind, SeatAgentStatus,
    TurnRequest,
};
use crate::match_host::fallback_turn;
use crate::variant::Variant;

/// Human seat placeholder: not ready (an explicit marker until the product injects a real action).
pub const MANUAL_NOT_READY: &str =
    "manual_seat_not_ready: human action injection is a product-layer responsibility";
/// Remote model config missing: without an endpoint only a safe fallback is possible.
pub const REMOTE_NOT_IMPLEMENTED: &str =
    "remote_model_not_configured: endpoint/token routing was not supplied by product layer";

/// Human seat placeholder. When asked it does not make a random or algorithmic
/// decision; it returns an explicit "not ready" marker plus a safe fallback action
/// (first legal discard or Pass), so headless orchestration can continue with clear
/// attribution (`trace.fallback` = the marker). Real human actions come from the
/// product (GUI/CLI) through the injection interface.
#[derive(Debug, Default)]
pub struct ManualSeat {
    calls: u64,
    fallbacks: u64,
}

impl ManualSeat {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<V: Variant> SeatAgent<V> for ManualSeat {
    fn decide_turn(&mut self, req: &TurnRequest<V>) -> AgentChoice<TurnAction> {
        self.calls += 1;
        self.fallbacks += 1;
        AgentChoice::fallback(
            fallback_turn(req.legal),
            0,
            None,
            "MANUAL_ACTION_NOT_READY",
            MANUAL_NOT_READY,
        )
    }

    fn decide_reaction(&mut self, _req: &ReactionRequest<V>) -> AgentChoice<ReactionAction> {
        self.calls += 1;
        self.fallbacks += 1;
        AgentChoice::fallback(
            ReactionAction::Pass,
            0,
            None,
            "MANUAL_ACTION_NOT_READY",
            MANUAL_NOT_READY,
        )
    }

    fn kind(&self) -> SeatAgentKind {
        SeatAgentKind::Manual
    }

    fn model_id(&self) -> String {
        "manual".to_string()
    }

    fn status(&self) -> SeatAgentStatus {
        SeatAgentStatus {
            kind: SeatAgentKind::Manual.as_str().to_string(),
            model_id: "manual".to_string(),
            calls: self.calls,
            fallbacks: self.fallbacks,
            errors: 0,
            last_latency_ms: Some(0),
        }
    }
}

/// Remote model seat config. `bearer_token` is a short-lived, scoped, expiring seat
/// credential; the runtime keeps it only in memory and drops it when the match ends.
/// It is never written to disk or the catalog.
///
/// Does not derive `Debug`: `bearer_token` would leak if formatted with `{:?}` into
/// anyhow context, traces or logs. The manual `Debug` only shows
/// `has_bearer_token: bool`, matching `RemoteHttpHost`.
#[derive(Clone)]
pub struct RemoteModelConfig {
    pub seat: u8,
    pub model_id: String,
    pub rule_line: String,
    pub credential_ref: Option<String>,
    pub endpoint: Option<String>,
    pub bearer_token: Option<String>,
    pub match_context: serde_json::Value,
}

impl std::fmt::Debug for RemoteModelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteModelConfig")
            .field("seat", &self.seat)
            .field("model_id", &self.model_id)
            .field("rule_line", &self.rule_line)
            .field("credential_ref", &self.credential_ref)
            .field("endpoint", &self.endpoint)
            .field("has_bearer_token", &self.bearer_token.is_some())
            .field("match_context", &self.match_context)
            .finish()
    }
}

/// Remote call error kinds (match `last_error.kind`; defined but not produced yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteModelError {
    Unauthorized,
    NoQuota,
    Unavailable,
    Timeout,
    /// Not implemented yet.
    NotImplemented,
}

/// Remote model seat. With a complete config it calls the configured HTTP inference
/// endpoint; with a missing config or a remote error it takes the safe fallback and
/// records the reason in `fallback` / `errors`.
#[derive(Debug)]
pub struct RemoteModelSeat {
    config: RemoteModelConfig,
    host: Option<RemoteHttpHost>,
    calls: u64,
    fallbacks: u64,
    errors: u64,
    last_latency_ms: u64,
}

impl RemoteModelSeat {
    pub fn new(config: RemoteModelConfig) -> Self {
        let host = remote_http_host_from_config(&config).ok();
        Self {
            config,
            host,
            calls: 0,
            fallbacks: 0,
            errors: 0,
            last_latency_ms: 0,
        }
    }

    pub fn config(&self) -> &RemoteModelConfig {
        &self.config
    }
}

impl<V: Variant> SeatAgent<V> for RemoteModelSeat {
    fn decide_turn(&mut self, req: &TurnRequest<V>) -> AgentChoice<TurnAction> {
        self.calls += 1;
        let Some(host) = self.host.as_ref() else {
            self.fallbacks += 1;
            return AgentChoice::fallback(
                fallback_turn(req.legal),
                0,
                None,
                "MODEL_NOT_CONFIGURED",
                REMOTE_NOT_IMPLEMENTED,
            );
        };
        let legal_actions = req
            .legal
            .iter()
            .map(|action| V::turn_action_to_legal(req.view, action))
            .collect::<Vec<_>>();
        let mut events = Vec::with_capacity(req.events.len() + 1);
        events.push(V::start_game_event());
        events.extend(req.events.iter().cloned());
        let started = Instant::now();
        let result = V::remote_infer(
            host,
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
                    format!("remote_out_of_range_index:{index}"),
                )
            }
            Ok(InferenceDecision::Abstain) => {
                self.fallbacks += 1;
                AgentChoice::fallback(
                    fallback_turn(req.legal),
                    latency,
                    Some("abstain".to_string()),
                    "MODEL_ABSTAIN",
                    "remote_abstain",
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
                    format!("remote_{:?}", err.kind),
                )
            }
        }
    }

    fn decide_reaction(&mut self, req: &ReactionRequest<V>) -> AgentChoice<ReactionAction> {
        self.calls += 1;
        let Some(host) = self.host.as_ref() else {
            self.fallbacks += 1;
            return AgentChoice::fallback(
                ReactionAction::Pass,
                0,
                None,
                "MODEL_NOT_CONFIGURED",
                REMOTE_NOT_IMPLEMENTED,
            );
        };
        let mut choices = Vec::with_capacity(req.legal.len() + 1);
        let mut legal_actions = Vec::with_capacity(req.legal.len() + 1);
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
        let result = V::remote_infer(
            host,
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
                self.fallbacks += 1;
                AgentChoice::fallback(
                    ReactionAction::Pass,
                    latency,
                    Some(format!("index:{index}")),
                    "MODEL_ACTION_ID_OUT_OF_RANGE",
                    format!("remote_out_of_range_index:{index}"),
                )
            }
            Ok(InferenceDecision::Abstain) => {
                self.fallbacks += 1;
                AgentChoice::fallback(
                    ReactionAction::Pass,
                    latency,
                    Some("abstain".to_string()),
                    "MODEL_ABSTAIN",
                    "remote_abstain",
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
                    format!("remote_{:?}", err.kind),
                )
            }
        }
    }

    fn kind(&self) -> SeatAgentKind {
        SeatAgentKind::RemoteModel
    }

    fn model_id(&self) -> String {
        self.config.model_id.clone()
    }

    fn status(&self) -> SeatAgentStatus {
        SeatAgentStatus {
            kind: SeatAgentKind::RemoteModel.as_str().to_string(),
            model_id: self.config.model_id.clone(),
            calls: self.calls,
            fallbacks: self.fallbacks,
            errors: self.errors,
            last_latency_ms: Some(self.last_latency_ms),
        }
    }
}

fn remote_http_host_from_config(config: &RemoteModelConfig) -> Result<RemoteHttpHost, ()> {
    let endpoint = config
        .endpoint
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or(())?;
    let rule_line = match config.rule_line.as_str() {
        "riichi4p" => flytable_seat::contract::RuleLine::Riichi4p,
        "riichi3p" => flytable_seat::contract::RuleLine::Riichi3p,
        _ => return Err(()),
    };
    let mut http = RemoteHttpConfig::new(
        endpoint,
        config.seat,
        rule_line,
        config
            .credential_ref
            .clone()
            .unwrap_or_else(|| config.model_id.clone()),
    );
    http.bearer_token = config.bearer_token.clone();
    http.match_context = config.match_context.clone();
    RemoteHttpHost::new(http).map_err(|_| ())
}

/// Input shape of an externally observed seat (minimal placeholder). `source_mode` is
/// always observed; available actions come from the platform's `possible_actions`
/// (the runtime does not re-enumerate or override them). Shape only for now.
#[derive(Debug, Clone)]
pub struct ObservedInput {
    pub session_id: String,
    pub rule_line: String,
    pub seat: u8,
    pub decision_id: String,
    /// The platform's authoritative action table (opaque JSON kept as is; the runtime only chooses from it).
    pub possible_actions: serde_json::Value,
    pub source_epoch: u64,
    pub source_status: ObservedSourceStatus,
}

/// Observation source status (reconnect / re-mirror semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservedSourceStatus {
    Live,
    Resync,
    Reconstructable,
}

/// Recommend-only observer config (recommendation panels and the like): delegates to
/// an algorithm, plugin or remote model for recommendations.
#[derive(Debug, Clone)]
pub struct ObserverConfig {
    /// For example `algorithm:tsumogiri` / `local_plugin:<model_id>` / `remote_model:<model_id>`.
    pub recommend_via: String,
}
