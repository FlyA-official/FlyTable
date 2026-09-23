//! In-process rule-based seat [`AlgorithmSeatAgent`].
//!
//! Wraps a `flytable-seat` [`SeatDecider`] as a [`SeatAgent`] with a model_id, call
//! counts and (near zero) latency. Built in:
//! - `tsumogiri`: [`TsumogiriDecider`] (tsumogiri only, no calls, no wins), the
//!   disconnect autoplay behavior.
//!
//! Stronger seats plug in as external engine plugins (see `flytable-inference-host`).
//! No observations, networking or processes.

use flytable_seat::{SeatDecider, TsumogiriDecider};
use flytable_table::{ReactionAction, TurnAction};

use crate::agent::{
    AgentChoice, ReactionRequest, SeatAgent, SeatAgentKind, SeatAgentStatus, TurnRequest,
};
use crate::variant::Variant;

/// Built-in algorithm kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmKind {
    /// Tsumogiri seat ([`TsumogiriDecider`]): disconnect autoplay.
    Tsumogiri,
}

impl AlgorithmKind {
    pub const fn model_id(self) -> &'static str {
        match self {
            AlgorithmKind::Tsumogiri => "tsumogiri",
        }
    }

    /// Parses a CLI spec string.
    pub fn parse(spec: &str) -> Option<Self> {
        match spec {
            "tsumogiri" => Some(AlgorithmKind::Tsumogiri),
            _ => None,
        }
    }
}

/// Algorithm seat wrapping a [`SeatDecider`]; works for 4-player and 3-player.
pub struct AlgorithmSeatAgent {
    kind: AlgorithmKind,
    decider: Box<dyn SeatDecider + Send>,
    calls: u64,
    last_latency_ms: u64,
}

impl AlgorithmSeatAgent {
    pub fn new(_seat: u8, kind: AlgorithmKind) -> Self {
        let decider: Box<dyn SeatDecider + Send> = match kind {
            AlgorithmKind::Tsumogiri => Box::new(TsumogiriDecider),
        };
        Self {
            kind,
            decider,
            calls: 0,
            last_latency_ms: 0,
        }
    }
}

impl<V: Variant> SeatAgent<V> for AlgorithmSeatAgent {
    fn decide_turn(&mut self, req: &TurnRequest<V>) -> AgentChoice<TurnAction> {
        self.calls += 1;
        self.last_latency_ms = 0;
        // Algorithms only need the view, not events. Pass the host's legal set so the
        // decider does not enumerate again.
        AgentChoice::ok(self.decider.decide_turn_with_legal(req.view, req.legal), 0)
    }

    fn decide_reaction(&mut self, req: &ReactionRequest<V>) -> AgentChoice<ReactionAction> {
        self.calls += 1;
        self.last_latency_ms = 0;
        AgentChoice::ok(self.decider.decide_reaction(req.view, req.legal), 0)
    }

    fn kind(&self) -> SeatAgentKind {
        SeatAgentKind::Algorithm
    }

    fn model_id(&self) -> String {
        self.kind.model_id().to_string()
    }

    fn status(&self) -> SeatAgentStatus {
        SeatAgentStatus {
            kind: SeatAgentKind::Algorithm.as_str().to_string(),
            model_id: self.kind.model_id().to_string(),
            calls: self.calls,
            fallbacks: 0,
            errors: 0,
            last_latency_ms: Some(self.last_latency_ms),
        }
    }
}
