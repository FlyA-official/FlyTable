//! Seat bus interface [`SeatAgent`]: how the host asks the current seat for an action.
//!
//! A superset of [`flytable_seat::SeatDecider`]: besides turning a view into an
//! action, it reports the decision source ([`SeatAgentKind`]), model id, latency and
//! fallback state for match logs and status displays.
//!
//! Shared by 4-player and 3-player. The differences (event stream shape, legal action
//! wire) are in [`crate::variant::Variant`], so one agent implementation serves both
//! (algorithm seats do not care about seat count; plugin seats bridge to the right
//! host call through `Variant`).

use flytable_table::{ReactionAction, SeatView, TurnAction};

use crate::variant::Variant;

/// Stable trace error code for an inference boundary error.
pub const fn inference_error_code(
    kind: flytable_seat::contract::InferenceErrorKind,
) -> &'static str {
    use flytable_seat::contract::InferenceErrorKind;
    match kind {
        InferenceErrorKind::InvalidAction => "MODEL_INVALID_ACTION",
        InferenceErrorKind::StaleDecision => "MODEL_STALE_DECISION",
        InferenceErrorKind::StateSyncRequired => "MODEL_STATE_DIGEST_MISMATCH",
        InferenceErrorKind::Timeout => "MODEL_TIMEOUT",
        InferenceErrorKind::Protocol => "MODEL_PROTOCOL_ERROR",
        InferenceErrorKind::Internal => "MODEL_INTERNAL_ERROR",
        InferenceErrorKind::PreSendUnavailable => "MODEL_PRE_SEND_UNAVAILABLE",
        InferenceErrorKind::EngineUnavailable => "MODEL_ENGINE_UNAVAILABLE",
    }
}

/// Source of seat decisions. `ObservedExternal` / `Observer` are placeholders for now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeatAgentKind {
    /// Human seat (actions injected by a GUI or CLI). Not implemented in the CLI yet.
    Manual,
    /// In-process rule-based seat (built-in tsumogiri).
    Algorithm,
    /// Local certified plugin seat calling a `flya-inference-v2` (with frozen v1) translator through `SubprocessHost`.
    LocalPlugin,
    /// Remote model seat (calls a remote inference endpoint with a short-lived credential).
    RemoteModel,
    /// Proxy seat for an external third-party client (observed external input).
    ObservedExternal,
    /// Recommend-only (recommendation panels and the like).
    Observer,
}

impl SeatAgentKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            SeatAgentKind::Manual => "manual",
            SeatAgentKind::Algorithm => "algorithm",
            SeatAgentKind::LocalPlugin => "local_plugin",
            SeatAgentKind::RemoteModel => "remote_model",
            SeatAgentKind::ObservedExternal => "observed_external",
            SeatAgentKind::Observer => "observer",
        }
    }
}

/// Seat status summary (health, latency, latest fallback, call counts).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SeatAgentStatus {
    pub kind: String,
    pub model_id: String,
    /// Number of decisions (turn + reaction).
    pub calls: u64,
    /// Number of fallbacks (abstain, error or illegal).
    pub fallbacks: u64,
    /// Number of engine errors (included in `fallbacks`).
    pub errors: u64,
    pub last_latency_ms: Option<u64>,
}

/// Result of one decision: action, latency and optional fallback summary (`None` means the agent answered normally).
#[derive(Debug, Clone)]
pub struct AgentChoice<A> {
    /// Summary of what the agent actually returned. `None` means it was `action` itself;
    /// anything that cannot be a typed action (out-of-range index, abstain, protocol error)
    /// must be kept as a raw summary.
    pub submitted_action: Option<String>,
    pub action: A,
    pub latency_ms: u64,
    /// `Some` means a fallback was used (plugin abstain, error, timeout); the string is the redacted reason.
    pub fallback: Option<String>,
    /// Stable, machine-readable error code; `None` for a normal choice.
    pub error_code: Option<&'static str>,
}

impl<A> AgentChoice<A> {
    pub fn ok(action: A, latency_ms: u64) -> Self {
        Self {
            submitted_action: None,
            action,
            latency_ms,
            fallback: None,
            error_code: None,
        }
    }

    pub fn fallback(
        action: A,
        latency_ms: u64,
        submitted_action: Option<String>,
        error_code: &'static str,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            submitted_action,
            action,
            latency_ms,
            fallback: Some(reason.into()),
            error_code: Some(error_code),
        }
    }
}

/// Decision request for the seat's own turn. `view` / `legal` do not depend on seat count.
///
/// Visibility boundary: the authoritative full event stream (including other players'
/// hands and draws) is only stored `pub(crate)`, so [`SeatAgent`] implementations
/// outside this crate cannot reach it; third-party agents only get the per-seat
/// projection through [`TurnRequest::visible_events`]. Forwarding agents inside the
/// runtime (plugin and placeholder seats) pass the raw stream to the inference host,
/// which projects it again for the seat bound at spawn time, so both paths expose
/// exactly the same information.
pub struct TurnRequest<'a, V: Variant> {
    /// Globally monotonic id the host generates and binds to this window; agents must not replace it.
    pub decision_id: &'a str,
    /// Digest the host computes from the visible state and the authoritative legal set.
    pub state_digest: &'a str,
    pub seat: u8,
    pub view: &'a SeatView,
    /// Authoritative event stream since the start of the hand (before projection). For
    /// internal forwarding; external code should use [`TurnRequest::visible_events`].
    pub(crate) events: &'a [V::Event],
    /// Legal turn actions enumerated by the rules (the only authoritative action table).
    pub legal: &'a [TurnAction],
    pub wall_remaining: u32,
}

impl<'a, V: Variant> TurnRequest<'a, V> {
    /// Builds a request for unit tests of agents outside this crate (at runtime the host always builds it).
    ///
    /// It still takes authoritative events: the test author already holds that data, so
    /// nothing leaks. The real boundary is that the host never hands the authoritative
    /// stream to external agents, which field visibility guarantees.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn for_agent_test(
        decision_id: &'a str,
        state_digest: &'a str,
        seat: u8,
        view: &'a SeatView,
        events: &'a [V::Event],
        legal: &'a [TurnAction],
        wall_remaining: u32,
    ) -> Self {
        Self {
            decision_id,
            state_digest,
            seat,
            view,
            events,
            legal,
            wall_remaining,
        }
    }

    /// Event stream since the start of the hand, projected for this seat: other players'
    /// hands and draws are hidden placeholders.
    ///
    /// Projected on demand, so seats that never call it pay nothing.
    #[must_use]
    pub fn visible_events(&self) -> Vec<V::VisibleEvent> {
        self.events
            .iter()
            .map(|e| V::project_event(e, self.seat))
            .collect()
    }
}

/// Decision request for a response window after another player's discard. `legal`
/// excludes Pass (always implicitly allowed). Same visibility boundary as [`TurnRequest`].
pub struct ReactionRequest<'a, V: Variant> {
    pub decision_id: &'a str,
    pub state_digest: &'a str,
    pub seat: u8,
    pub view: &'a SeatView,
    /// Authoritative event stream since the start of the hand (before projection). For
    /// internal forwarding; external code should use [`ReactionRequest::visible_events`].
    pub(crate) events: &'a [V::Event],
    /// Legal responses of this seat (pon / chi / open kan / ron), excluding Pass.
    pub legal: &'a [ReactionAction],
    pub discarder: u8,
    pub tile: flytable_core::tile::Tile,
    pub wall_remaining: u32,
}

impl<V: Variant> ReactionRequest<'_, V> {
    /// Event stream projected for this seat (see [`TurnRequest::visible_events`]).
    #[must_use]
    pub fn visible_events(&self) -> Vec<V::VisibleEvent> {
        self.events
            .iter()
            .map(|e| V::project_event(e, self.seat))
            .collect()
    }
}

/// Common seat decision interface. The host treats algorithm, plugin, human, remote and observed seats the same.
pub trait SeatAgent<V: Variant>: Send {
    /// Action on the seat's own turn. It should be in `req.legal`; the host checks again,
    /// falls back if it is not, and marks it in [`DecisionTrace`].
    fn decide_turn(&mut self, req: &TurnRequest<V>) -> AgentChoice<TurnAction>;

    /// Picks a response from the legal set, or [`ReactionAction::Pass`].
    fn decide_reaction(&mut self, req: &ReactionRequest<V>) -> AgentChoice<ReactionAction>;

    /// Decision source (for logs and status).
    fn kind(&self) -> SeatAgentKind;

    /// Model id (`tsumogiri` for the built-in algorithm, the registry-normalized `flya-plugin:...` for plugins).
    fn model_id(&self) -> String;

    /// Health, latency and latest fallback summary.
    fn status(&self) -> SeatAgentStatus;
}

/// Recordable trace of one decision: match and hand, turn/seat, legal_count,
/// selected, agent kind/model_id, latency or fallback. Serializes to JSONL with a
/// stable schema that products and replay comparisons rely on.
///
/// Every field except `latency_ms` (wall clock, varies for plugin seats) is
/// reproducible for the same seed and seat config. Algorithm seats always report
/// `latency_ms = 0`, so algorithm-only traces are fully deterministic. Golden
/// comparisons should ignore `latency_ms`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DecisionErrorTrace {
    /// Stable error code, e.g. `MODEL_ACTION_NOT_LEGAL` / `MODEL_TIMEOUT`.
    pub code: String,
    pub category: String,
    /// Human-readable summary of the authoritative legal actions; contains no hidden information.
    pub legal_actions: Vec<String>,
    /// Redacted details. Must not contain tokens, cookies, credentials or full raw responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DecisionTrace {
    /// Decision id generated and bound by the host.
    pub decision_id: String,
    /// Digest of the host-visible state and the authoritative legal actions.
    pub state_digest: String,
    /// Index of this hand within the match (0-based; always 0 for single-hand matches).
    pub kyoku_index: u32,
    /// Round wind: 0 = East, 1 = South, 2 = West, 3 = North.
    pub bakaze: u8,
    /// Hand number (1-based: East 1 = 1, ...).
    pub kyoku: u8,
    pub honba: u8,
    /// Riichi sticks on the table.
    pub kyotaku: u8,
    /// Turn decision number within the hand (reactions reuse the number of their turn); monotonic across hands.
    pub turn_index: u32,
    pub seat: u8,
    /// `"turn"` or `"reaction"`.
    pub phase: String,
    /// Number of legal actions.
    pub legal_count: usize,
    /// Selected action (human-readable). Legacy field, same as `final_action`.
    pub selected: String,
    /// What the model or agent submitted; protocol errors, out-of-range indices and abstains must still leave an explanation.
    pub submitted_action: String,
    /// The action actually applied after authoritative validation and independent fallback.
    pub final_action: String,
    pub agent_kind: String,
    pub model_id: String,
    pub latency_ms: u64,
    /// `Some` means a fallback was used (reason summary).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback: Option<String>,
    /// Whether the selected action was in the host's legal set (`false` means the host replaced it).
    pub selected_in_legal: bool,
    /// Structured error; `None` for a normal decision.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<DecisionErrorTrace>,
}
