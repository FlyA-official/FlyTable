//! [`RuntimeStatus`]: seat to agent/model, current state, last error, fallback and
//! latency, so products can pass runtime status through instead of rebuilding it.
//!
//! Currently derived from a finished [`FullMatchReport`] (post-match snapshot). Field
//! meanings are stable; redacted text reuses the trace's fallback summary (no tokens
//! or keys).

use crate::agent::SeatAgentStatus;
use crate::match_host::{FullMatchReport, MatchFailure, TRACE_SCHEMA_VERSION};
use flytable_core::rules::RedFiveCounts;

/// Status of one seat.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SeatRuntimeStatus {
    pub seat: u8,
    pub agent_kind: String,
    pub model_id: String,
    /// Summary: `healthy` (no fallbacks or errors) / `degraded` (fallbacks or abstains) / `error` (engine errors).
    pub state: &'static str,
    pub calls: u64,
    pub fallbacks: u64,
    pub errors: u64,
    pub last_latency_ms: Option<u64>,
    /// Latest fallback reason (redacted; `None` if there was none).
    pub last_fallback: Option<String>,
}

/// Runtime status snapshot of a match (seat to agent/model plus health).
#[derive(Debug, Clone, serde::Serialize)]
pub struct RuntimeStatus {
    pub schema_version: u32,
    pub variant: String,
    pub rule_profile: String,
    pub red_fives: RedFiveCounts,
    pub match_id: String,
    pub completion: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<MatchFailure>,
    pub seats: Vec<SeatRuntimeStatus>,
}

impl RuntimeStatus {
    /// Derived from a full report, merging each seat's [`SeatAgentStatus`] with its latest fallback in the traces.
    pub fn from_full_report(report: &FullMatchReport) -> Self {
        let last_fallbacks = last_fallback_per_seat(report, report.players as usize);
        let seats = report
            .agents
            .iter()
            .enumerate()
            .map(|(seat, status)| {
                let last_fallback = last_fallbacks.get(seat).cloned().flatten();
                SeatRuntimeStatus {
                    seat: seat as u8,
                    agent_kind: status.kind.to_string(),
                    model_id: status.model_id.clone(),
                    state: seat_state(status),
                    calls: status.calls,
                    fallbacks: status.fallbacks,
                    errors: status.errors,
                    last_latency_ms: status.last_latency_ms,
                    last_fallback,
                }
            })
            .collect();
        Self {
            schema_version: TRACE_SCHEMA_VERSION,
            variant: report.variant.clone(),
            rule_profile: report.rule_profile.clone(),
            red_fives: report.red_fives,
            match_id: report.match_id.clone(),
            completion: report.completion.clone(),
            failure: report.failure.clone(),
            seats,
        }
    }
}

fn seat_state(status: &SeatAgentStatus) -> &'static str {
    if status.errors > 0 {
        "error"
    } else if status.fallbacks > 0 {
        "degraded"
    } else {
        "healthy"
    }
}

/// Latest non-empty fallback summary per seat across all hands.
fn last_fallback_per_seat(report: &FullMatchReport, seats: usize) -> Vec<Option<String>> {
    let mut last = vec![None; seats];
    for kyoku in &report.kyokus {
        for trace in &kyoku.traces {
            if let Some(fb) = &trace.fallback {
                if let Some(slot) = last.get_mut(trace.seat as usize) {
                    *slot = Some(fb.clone());
                }
            }
        }
    }
    if let Some(attempt) = &report.current_attempt {
        for trace in &attempt.traces {
            if let Some(fallback) = &trace.fallback {
                if let Some(slot) = last.get_mut(trace.seat as usize) {
                    *slot = Some(fallback.clone());
                }
            }
        }
    }
    last
}
