//! Match host [`MatchHost`]: generalizes the self-play loop into asking agents per
//! seat, with a full match lifecycle.
//!
//! One hand: draw, ask the current seat's [`SeatAgent`], adjudicate by the rules
//! (`Board::apply_turn`), poll the other seats' response windows after a discard
//! (pon / chi / open kan / ron), advance, until a win or exhaustive draw. After a call,
//! kan or nukidora the same seat decides again without drawing, as in the smoke test
//! loop of `flytable-inference-host::registry`.
//!
//! Across hands, [`run_full_match_4p`] / [`run_full_match_3p`] add the match lifecycle:
//! honba, riichi sticks, renchan, winds and score carry-over are all delegated to
//! `flytable-table::progress`, so the runtime has no second copy of the rules. Each
//! hand has its own seed and wall, and the whole match is reproducible from one seed.
//!
//! Algorithm and local plugin seats can be mixed freely, and agent instances are
//! reused across hands (plugin subprocesses are not restarted per hand). Progression
//! is driven by `Board4p` / `Board3p`; this module only orchestrates.

use anyhow::{anyhow, Result};
use flytable_core::rules::{RedFiveCounts, RiichiRuleProfile};
use flytable_event::{Event3p, Event4p};
use flytable_table::progress::{rankings, MatchLength};
use flytable_table::{Board3p, Board4p, KyokuOutcome, KyokuOutcome3p, ReactionAction, TurnAction};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

use flytable_inference_host::registry::CertifiedPluginRuntime;

use crate::agent::{DecisionErrorTrace, DecisionTrace, SeatAgent, SeatAgentStatus};
use crate::algorithm::{AlgorithmKind, AlgorithmSeatAgent};
use crate::placeholders::{RemoteModelConfig, RemoteModelSeat};
use crate::plugin::LocalPluginSeatAgent;
use crate::variant::{Variant, Variant3p, Variant4p};

/// Trace / report schema version. Products and replay comparisons rely on it; bump on breaking changes.
///
/// v4 adds `current_attempt` on top of the explicit failure states of v3, and in-progress
/// snapshots become `kyoku_in_progress` / `match_snapshot`. `final_scores` /
/// `rankings` are only set on natural completion.
pub const TRACE_SCHEMA_VERSION: u32 = 4;

/// Low word of each hand's wall seed (matches `Board4p::start((seed, 0x5eed))`, so hand 0 of a match equals a single hand).
pub(crate) const KYOKU_SEED_LO: u64 = 0x5eed;
/// Seed mixing multiplier so hands do not share walls.
pub(crate) const KYOKU_SEED_MIX: u64 = 0x9e37_79b9_7f4a_7c15;

/// Who sits in a seat (produced by the CLI or caller).
pub enum SeatSpec {
    /// Built-in algorithm seat.
    Algorithm(AlgorithmKind),
    /// Local certified plugin seat (runtime obtained from the registry).
    Plugin(CertifiedPluginRuntime),
    /// Remote online model seat.
    RemoteModel(RemoteModelConfig),
}

/// Match length. `Single` plays one hand (legacy run-match behavior); `East` / `Half` play to the end of a tonpuusen / hanchan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchKind {
    Single,
    East,
    Half,
}

impl MatchKind {
    /// Parses a CLI string.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "single" | "one" | "single-kyoku" => Some(Self::Single),
            "east" | "tonpuu" => Some(Self::East),
            "half" | "hanchan" => Some(Self::Half),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::East => "east",
            Self::Half => "half",
        }
    }

    /// Match length passed to progress. `Single` uses East semantics only to compute the
    /// hand's deltas (not for progression or game end).
    pub(crate) fn progress_length(self) -> MatchLength {
        match self {
            Self::Half => MatchLength::Half,
            _ => MatchLength::East,
        }
    }
}

/// Match configuration.
#[derive(Debug, Clone, Copy)]
pub struct MatchConfig {
    pub seed: u64,
    pub kind: MatchKind,
    /// Platform rules; the seat-count variant does not replace them.
    pub rule_profile: RiichiRuleProfile,
    /// Starting score per seat; `None` uses the profile's default for the seat count.
    pub start_score: Option<i32>,
}

impl MatchConfig {
    /// A single hand (legacy run-match form).
    pub fn single(seed: u64) -> Self {
        Self {
            seed,
            kind: MatchKind::Single,
            rule_profile: RiichiRuleProfile::default(),
            start_score: None,
        }
    }

    /// A match of the given length.
    pub fn new(seed: u64, kind: MatchKind) -> Self {
        Self {
            seed,
            kind,
            rule_profile: RiichiRuleProfile::default(),
            start_score: None,
        }
    }

    pub fn with_rule_profile(mut self, rule_profile: RiichiRuleProfile) -> Self {
        self.rule_profile = rule_profile;
        self
    }
}

/// Hand context (stamped on every trace for location and display).
#[derive(Debug, Clone, Copy)]
pub(crate) struct KyokuContext {
    pub(crate) kyoku_index: u32,
    pub(crate) bakaze: u8,
    pub(crate) kyoku: u8,
    pub(crate) honba: u8,
    pub(crate) kyotaku: u8,
}

/// Stable, redacted structure for a host failure.
///
/// This is a state machine terminal state, not a mahjong result: it must not enter a
/// [`KyokuRecord`] or produce settlement, rankings or `match_end`. `detail` may only
/// hold a stable redacted summary, never raw plugin, protocol or internal error text.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MatchFailure {
    pub code: String,
    pub stage: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seat: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision_id: Option<String>,
    pub detail: String,
}

impl std::fmt::Display for MatchFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} stage={}", self.code, self.stage)?;
        if let Some(seat) = self.seat {
            write!(f, " seat={seat}")?;
        }
        if let Some(decision_id) = &self.decision_id {
            write!(f, " decision_id={decision_id}")?;
        }
        write!(f, " detail={}", self.detail)
    }
}

/// Full report of one hand (see [`FullMatchReport`] for matches).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MatchReport {
    pub schema_version: u32,
    /// Deterministic match id (identical for the same variant, length, platform and seed; usable as a replay key).
    pub match_id: String,
    pub rule_line: String,
    /// Platform rules; `rule_line` only says 4p/3p and does not replace this.
    pub rule_profile: String,
    /// Red five counts of this hand's physical wall.
    pub red_fives: RedFiveCounts,
    pub players: u8,
    pub seed: u64,
    /// Result summary (hora / ryukyoku / abortive).
    pub outcome: String,
    pub events_total: usize,
    pub turns: u32,
    /// Scores after settlement (including this hand's deltas).
    pub final_scores: Vec<i32>,
    pub traces: Vec<DecisionTrace>,
    pub agents: Vec<SeatAgentStatus>,
}

impl MatchReport {
    /// Projects a single-hand [`FullMatchReport`] into the legacy single-hand report.
    fn from_single(full: FullMatchReport) -> Self {
        let final_scores = if full.final_scores.is_empty() {
            full.current_scores.clone()
        } else {
            full.final_scores.clone()
        };
        let kyoku = full
            .kyokus
            .into_iter()
            .next()
            .expect("single match has exactly one kyoku");
        Self {
            schema_version: full.schema_version,
            match_id: full.match_id,
            rule_line: full.variant,
            rule_profile: full.rule_profile,
            red_fives: full.red_fives,
            players: full.players,
            seed: full.seed,
            outcome: kyoku.outcome,
            events_total: kyoku.events_total,
            turns: kyoku.turns,
            final_scores,
            traces: kyoku.traces,
            agents: full.agents,
        }
    }
}

/// Stable record of one hand: state, per-move traces and settlement.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KyokuRecord {
    pub kyoku_index: u32,
    pub bakaze: u8,
    pub kyoku: u8,
    /// Honba at the start of the hand.
    pub honba: u8,
    /// Riichi sticks carried into the hand.
    pub kyotaku: u8,
    pub oya: u8,
    /// Wall seed of the hand (reproducible: the match seed determines every hand's seed).
    pub seed_hi: u64,
    pub seed_lo: u64,
    /// Scores at the start of the hand (before riichi sticks were taken).
    pub start_scores: Vec<i32>,
    /// Whether each seat declared riichi this hand (source of sticks).
    pub riichi_declared: Vec<bool>,
    /// Result summary.
    pub outcome: String,
    /// Per-seat deltas for the hand (including honba, sticks and penalties; from progress).
    pub deltas: Vec<i32>,
    /// Scores after settlement (start_scores after riichi sticks + deltas); the next hand's start_scores.
    pub scores_after: Vec<i32>,
    /// Next hand's honba, sticks and dealer (the final state if the match ended).
    pub next_honba: u8,
    pub next_kyotaku: u8,
    pub next_oya: u8,
    pub turns: u32,
    pub events_total: usize,
    pub traces: Vec<DecisionTrace>,
}

/// A hand attempt that has not been settled. It can be in progress or failed, but never
/// has outcome, deltas, scores_after or other settlement fields.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CurrentKyokuAttempt {
    pub kyoku_index: u32,
    pub bakaze: u8,
    pub kyoku: u8,
    pub honba: u8,
    pub kyotaku: u8,
    pub oya: u8,
    pub seed_hi: u64,
    pub seed_lo: u64,
    pub start_scores: Vec<i32>,
    pub turns: u32,
    pub events_total: usize,
    pub traces: Vec<DecisionTrace>,
}

/// Full report of a match (stable schema for products and replay).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FullMatchReport {
    pub schema_version: u32,
    pub match_id: String,
    pub variant: String,
    /// `tenhou` / `mahjong-soul` / `riichi-city`.
    pub rule_profile: String,
    /// Red five counts of the physical wall; room-level profiles cannot be described by platform name alone.
    pub red_fives: RedFiveCounts,
    pub players: u8,
    pub seed: u64,
    pub length: String,
    pub start_scores: Vec<i32>,
    pub kyokus: Vec<KyokuRecord>,
    /// Unfinished hand. Settled hands only appear in `kyokus`, never here as well.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_attempt: Option<CurrentKyokuAttempt>,
    /// Current authoritative scores, present in all three states.
    pub current_scores: Vec<i32>,
    /// Only set on natural completion; empty and omitted from JSON while in progress or failed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub final_scores: Vec<i32>,
    /// Rankings (seats by descending score).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rankings: Vec<u8>,
    /// Whether the match reached its natural end (match length reached or a player busted).
    pub ended: bool,
    /// `completed` / `in_progress` / `failed`.
    pub completion: String,
    /// Host failure state; `None` for a normal match.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<MatchFailure>,
    pub agents: Vec<SeatAgentStatus>,
}

impl FullMatchReport {
    /// Strictly checks the v4 three-state invariants. Persisted reports must pass this
    /// before use; unknown versions fail closed.
    pub fn validate_schema(&self) -> Result<(), String> {
        if self.schema_version != TRACE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported trace schema version {} (expected {})",
                self.schema_version, TRACE_SCHEMA_VERSION
            ));
        }
        let players = self.players as usize;
        if !matches!(players, 3 | 4) {
            return Err(format!("report player count must be 3 or 4, got {players}"));
        }
        if self.start_scores.len() != players || self.current_scores.len() != players {
            return Err("report score width does not match player count".into());
        }
        if !self.final_scores.is_empty() && self.final_scores.len() != players {
            return Err("final score width does not match player count".into());
        }
        match self.completion.as_str() {
            "completed" => {
                if !self.ended
                    || self.failure.is_some()
                    || self.current_attempt.is_some()
                    || self.final_scores.len() != players
                    || self.rankings.len() != players
                {
                    return Err("completed report has contradictory terminal fields".into());
                }
                if self.current_scores != self.final_scores {
                    return Err("completed report current/final score snapshots disagree".into());
                }
                if self.rankings != rankings(&self.final_scores) {
                    return Err("completed report rankings disagree with final scores".into());
                }
            }
            "failed" => {
                if self.ended
                    || self.failure.is_none()
                    || self.current_attempt.is_none()
                    || !self.final_scores.is_empty()
                    || !self.rankings.is_empty()
                {
                    return Err("failed report has contradictory terminal fields".into());
                }
            }
            "in_progress" => {
                if self.ended
                    || self.failure.is_some()
                    || !self.final_scores.is_empty()
                    || !self.rankings.is_empty()
                {
                    return Err("in-progress report has contradictory terminal fields".into());
                }
            }
            other => return Err(format!("unknown report completion state {other:?}")),
        }

        if self.completion == "completed" && self.kyokus.is_empty() {
            return Err("completed report must contain at least one settled kyoku".into());
        }

        let mut decision_ids = HashSet::new();
        let mut last_trace_turn = None;
        let mut last_turn_decision = None;
        let mut previous_kyoku: Option<&KyokuRecord> = None;
        for (index, kyoku) in self.kyokus.iter().enumerate() {
            if kyoku.kyoku_index != index as u32 {
                return Err(format!(
                    "kyoku[{index}] index {} is not contiguous",
                    kyoku.kyoku_index
                ));
            }
            validate_round_context(
                kyoku.kyoku_index,
                kyoku.bakaze,
                kyoku.kyoku,
                kyoku.honba,
                kyoku.kyotaku,
                kyoku.oya,
                players,
                "settled kyoku",
            )?;
            if kyoku.next_oya as usize >= players {
                return Err(format!(
                    "kyoku[{index}] next dealer {} is outside the table",
                    kyoku.next_oya
                ));
            }
            let rotated_oya = (kyoku.oya + 1) % self.players;
            if !matches!(kyoku.next_oya, next if next == kyoku.oya || next == rotated_oya) {
                return Err(format!(
                    "kyoku[{index}] next dealer {} is neither renchan nor the next seat",
                    kyoku.next_oya
                ));
            }
            if kyoku.start_scores.len() != players
                || kyoku.riichi_declared.len() != players
                || kyoku.deltas.len() != players
                || kyoku.scores_after.len() != players
            {
                return Err(format!(
                    "kyoku[{index}] score/riichi/delta width does not match player count"
                ));
            }
            if !matches!(
                kyoku.outcome.as_str(),
                outcome
                    if outcome.starts_with("hora ")
                        || outcome.starts_with("multi_ron ")
                        || outcome.starts_with("ryukyoku ")
                        || outcome.starts_with("nagashi_mangan ")
                        || outcome.starts_with("abortive_ryukyoku ")
            ) {
                return Err(format!(
                    "kyoku[{index}] has unknown outcome summary {:?}",
                    kyoku.outcome
                ));
            }

            let expected_start = previous_kyoku
                .map(|previous| previous.scores_after.as_slice())
                .unwrap_or(self.start_scores.as_slice());
            if kyoku.start_scores != expected_start {
                return Err(format!(
                    "kyoku[{index}] start scores break the match score carry chain"
                ));
            }
            if let Some(previous) = previous_kyoku {
                if kyoku.honba != previous.next_honba
                    || kyoku.kyotaku != previous.next_kyotaku
                    || kyoku.oya != previous.next_oya
                {
                    return Err(format!(
                        "kyoku[{index}] round context disagrees with the previous next-state"
                    ));
                }
                let (expected_bakaze, expected_kyoku) = next_round_label(previous, self.players)?;
                if (kyoku.bakaze, kyoku.kyoku) != (expected_bakaze, expected_kyoku) {
                    return Err(format!(
                        "kyoku[{index}] round label disagrees with the previous dealer transition"
                    ));
                }
            }

            for seat in 0..players {
                let riichi_cost = if kyoku.riichi_declared[seat] {
                    1_000
                } else {
                    0
                };
                let expected = kyoku.start_scores[seat]
                    .checked_sub(riichi_cost)
                    .and_then(|score| score.checked_add(kyoku.deltas[seat]))
                    .ok_or_else(|| {
                        format!("kyoku[{index}] score arithmetic overflows at seat {seat}")
                    })?;
                if kyoku.scores_after[seat] != expected {
                    return Err(format!(
                        "kyoku[{index}] score delta disagrees with start score/riichi at seat {seat}"
                    ));
                }
            }
            let bank_before = score_total(&kyoku.start_scores) + i64::from(kyoku.kyotaku) * 1_000;
            let bank_after =
                score_total(&kyoku.scores_after) + i64::from(kyoku.next_kyotaku) * 1_000;
            if bank_before != bank_after {
                return Err(format!(
                    "kyoku[{index}] score plus kyotaku is not conserved"
                ));
            }

            validate_trace_block(
                &kyoku.traces,
                kyoku.kyoku_index,
                kyoku.bakaze,
                kyoku.kyoku,
                kyoku.honba,
                kyoku.kyotaku,
                kyoku.turns,
                players,
                &mut decision_ids,
                &mut last_trace_turn,
                &mut last_turn_decision,
                "settled kyoku",
            )?;
            previous_kyoku = Some(kyoku);
        }

        if let Some(attempt) = &self.current_attempt {
            if attempt.kyoku_index != self.kyokus.len() as u32 {
                return Err("current attempt context disagrees with completed kyokus".into());
            }
            validate_round_context(
                attempt.kyoku_index,
                attempt.bakaze,
                attempt.kyoku,
                attempt.honba,
                attempt.kyotaku,
                attempt.oya,
                players,
                "current attempt",
            )?;
            if attempt.start_scores.len() != players {
                return Err("current attempt score width does not match player count".into());
            }
            let expected_start = previous_kyoku
                .map(|previous| previous.scores_after.as_slice())
                .unwrap_or(self.start_scores.as_slice());
            if attempt.start_scores != expected_start {
                return Err("current attempt breaks the match score carry chain".into());
            }
            if let Some(previous) = previous_kyoku {
                if attempt.honba != previous.next_honba
                    || attempt.kyotaku != previous.next_kyotaku
                    || attempt.oya != previous.next_oya
                {
                    return Err(
                        "current attempt context disagrees with the previous next-state".into(),
                    );
                }
                let (expected_bakaze, expected_kyoku) = next_round_label(previous, self.players)?;
                if (attempt.bakaze, attempt.kyoku) != (expected_bakaze, expected_kyoku) {
                    return Err(
                        "current attempt round label disagrees with the previous transition".into(),
                    );
                }
            }
            validate_trace_block(
                &attempt.traces,
                attempt.kyoku_index,
                attempt.bakaze,
                attempt.kyoku,
                attempt.honba,
                attempt.kyotaku,
                attempt.turns,
                players,
                &mut decision_ids,
                &mut last_trace_turn,
                &mut last_turn_decision,
                "current attempt",
            )?;
        } else {
            let expected_current = previous_kyoku
                .map(|previous| previous.scores_after.as_slice())
                .unwrap_or(self.start_scores.as_slice());
            if self.current_scores != expected_current {
                return Err(
                    "report without a current attempt has an unexplained current score snapshot"
                        .into(),
                );
            }
        }

        if self.completion == "completed"
            && self
                .kyokus
                .last()
                .is_some_and(|kyoku| kyoku.scores_after != self.final_scores)
        {
            return Err("completed report final scores disagree with the final kyoku".into());
        }
        Ok(())
    }

    /// Reads from JSON and runs the fail-closed version and state checks.
    pub fn from_json(input: &str) -> Result<Self, String> {
        let report: Self =
            serde_json::from_str(input).map_err(|error| format!("invalid report JSON: {error}"))?;
        report.validate_schema()?;
        Ok(report)
    }

    /// JSONL stream for products. Completed hands are
    /// `kyoku_start -> decision* -> kyoku_end`; unfinished attempts are
    /// `kyoku_start -> decision* -> kyoku_in_progress|kyoku_failed`.
    pub fn jsonl_lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        out.push(
            serde_json::json!({
                "type": "match",
                "schema_version": self.schema_version,
                "match_id": self.match_id,
                "variant": self.variant,
                "rule_profile": self.rule_profile,
                "red_fives": self.red_fives,
                "players": self.players,
                "seed": self.seed,
                "length": self.length,
                "start_scores": self.start_scores,
            })
            .to_string(),
        );
        for kyoku in &self.kyokus {
            out.push(
                serde_json::json!({
                    "type": "kyoku_start",
                    "kyoku_index": kyoku.kyoku_index,
                    "bakaze": kyoku.bakaze,
                    "kyoku": kyoku.kyoku,
                    "honba": kyoku.honba,
                    "kyotaku": kyoku.kyotaku,
                    "oya": kyoku.oya,
                    "seed_hi": kyoku.seed_hi,
                    "seed_lo": kyoku.seed_lo,
                    "start_scores": kyoku.start_scores,
                })
                .to_string(),
            );
            for trace in &kyoku.traces {
                let mut value = serde_json::to_value(trace).expect("trace serializes");
                value["type"] = serde_json::json!("decision");
                out.push(value.to_string());
            }
            out.push(
                serde_json::json!({
                    "type": "kyoku_end",
                    "kyoku_index": kyoku.kyoku_index,
                    "outcome": kyoku.outcome,
                    "riichi_declared": kyoku.riichi_declared,
                    "deltas": kyoku.deltas,
                    "scores_after": kyoku.scores_after,
                    "next_honba": kyoku.next_honba,
                    "next_kyotaku": kyoku.next_kyotaku,
                    "next_oya": kyoku.next_oya,
                })
                .to_string(),
            );
        }
        if let Some(attempt) = &self.current_attempt {
            out.push(
                serde_json::json!({
                    "type": "kyoku_start",
                    "kyoku_index": attempt.kyoku_index,
                    "bakaze": attempt.bakaze,
                    "kyoku": attempt.kyoku,
                    "honba": attempt.honba,
                    "kyotaku": attempt.kyotaku,
                    "oya": attempt.oya,
                    "seed_hi": attempt.seed_hi,
                    "seed_lo": attempt.seed_lo,
                    "start_scores": attempt.start_scores,
                })
                .to_string(),
            );
            for trace in &attempt.traces {
                let mut value = serde_json::to_value(trace).expect("trace serializes");
                value["type"] = serde_json::json!("decision");
                out.push(value.to_string());
            }
            let attempt_terminal = if self.completion == "failed" {
                serde_json::json!({
                    "type": "kyoku_failed",
                    "kyoku_index": attempt.kyoku_index,
                    "turns": attempt.turns,
                    "events_total": attempt.events_total,
                    "current_scores": self.current_scores,
                    "failure": self.failure,
                })
            } else {
                serde_json::json!({
                    "type": "kyoku_in_progress",
                    "kyoku_index": attempt.kyoku_index,
                    "turns": attempt.turns,
                    "events_total": attempt.events_total,
                    "current_scores": self.current_scores,
                })
            };
            out.push(attempt_terminal.to_string());
        }
        match self.completion.as_str() {
            "completed" => out.push(
                serde_json::json!({
                    "type": "match_end",
                    "final_scores": self.final_scores,
                    "rankings": self.rankings,
                    "ended": true,
                    "completion": "completed",
                })
                .to_string(),
            ),
            "failed" => out.push(
                serde_json::json!({
                    "type": "match_failed",
                    "current_scores": self.current_scores,
                    "ended": false,
                    "completion": "failed",
                    "failure": self.failure,
                })
                .to_string(),
            ),
            "in_progress" => out.push(
                serde_json::json!({
                    "type": "match_snapshot",
                    "current_scores": self.current_scores,
                    "ended": false,
                    "completion": "in_progress",
                })
                .to_string(),
            ),
            _ => {}
        }
        out
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_trace_block(
    traces: &[DecisionTrace],
    kyoku_index: u32,
    bakaze: u8,
    kyoku: u8,
    honba: u8,
    kyotaku: u8,
    turns: u32,
    players: usize,
    decision_ids: &mut HashSet<String>,
    last_trace_turn: &mut Option<u32>,
    last_turn_decision: &mut Option<u32>,
    owner: &str,
) -> Result<(), String> {
    let mut counted_turns = 0u32;
    for (index, trace) in traces.iter().enumerate() {
        if trace.kyoku_index != kyoku_index
            || trace.bakaze != bakaze
            || trace.kyoku != kyoku
            || trace.honba != honba
            || trace.kyotaku != kyotaku
        {
            return Err(format!(
                "{owner} trace[{index}] context disagrees with its owning kyoku"
            ));
        }
        if trace.seat as usize >= players {
            return Err(format!(
                "{owner} trace[{index}] seat {} is outside the table",
                trace.seat
            ));
        }
        if trace.decision_id.is_empty() || !decision_ids.insert(trace.decision_id.clone()) {
            return Err(format!(
                "{owner} trace[{index}] has an empty or duplicate decision_id"
            ));
        }
        if trace.state_digest.is_empty() {
            return Err(format!("{owner} trace[{index}] has an empty state_digest"));
        }
        if trace.legal_count == 0 {
            return Err(format!(
                "{owner} trace[{index}] has an empty legal action set"
            ));
        }
        if trace.selected != trace.final_action {
            return Err(format!(
                "{owner} trace[{index}] compatibility selected field disagrees with final_action"
            ));
        }
        if last_trace_turn.is_some_and(|previous| trace.turn_index < previous) {
            return Err(format!("{owner} trace[{index}] turn_index moves backwards"));
        }
        match trace.phase.as_str() {
            "turn" => {
                if let Some(previous) = *last_turn_decision {
                    let expected = previous
                        .checked_add(1)
                        .ok_or_else(|| format!("{owner} trace[{index}] turn_index overflows"))?;
                    if trace.turn_index != expected {
                        return Err(format!(
                            "{owner} trace[{index}] turn decision index is not contiguous"
                        ));
                    }
                }
                *last_turn_decision = Some(trace.turn_index);
                counted_turns = counted_turns
                    .checked_add(1)
                    .ok_or_else(|| format!("{owner} turn count overflows"))?;
            }
            "reaction" => {
                let Some(previous) = *last_turn_decision else {
                    return Err(format!(
                        "{owner} trace[{index}] reaction appears before any turn decision"
                    ));
                };
                let expected = previous.checked_add(1).ok_or_else(|| {
                    format!("{owner} trace[{index}] reaction turn_index overflows")
                })?;
                if !matches!(
                    trace.turn_index,
                    candidate if candidate == previous || candidate == expected
                ) {
                    return Err(format!(
                        "{owner} trace[{index}] reaction is not attached to the preceding turn"
                    ));
                }
            }
            phase => {
                return Err(format!(
                    "{owner} trace[{index}] has unknown decision phase {phase:?}"
                ));
            }
        }
        *last_trace_turn = Some(trace.turn_index);
    }
    if counted_turns != turns {
        return Err(format!(
            "{owner} turns={turns} disagrees with {counted_turns} turn traces"
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_round_context(
    kyoku_index: u32,
    bakaze: u8,
    kyoku: u8,
    _honba: u8,
    _kyotaku: u8,
    oya: u8,
    players: usize,
    owner: &str,
) -> Result<(), String> {
    if oya as usize >= players {
        return Err(format!(
            "{owner}[{kyoku_index}] dealer is outside the table"
        ));
    }
    if kyoku == 0 || kyoku as usize > players {
        return Err(format!(
            "{owner}[{kyoku_index}] round number {kyoku} is outside 1..={players}"
        ));
    }
    if bakaze > 3 {
        return Err(format!(
            "{owner}[{kyoku_index}] round wind index {bakaze} is outside 0..=3"
        ));
    }
    Ok(())
}

fn next_round_label(previous: &KyokuRecord, players: u8) -> Result<(u8, u8), String> {
    if previous.next_oya == previous.oya {
        return Ok((previous.bakaze, previous.kyoku));
    }
    if previous.next_oya != (previous.oya + 1) % players {
        return Err(format!(
            "kyoku[{}] next dealer is not a valid rotation",
            previous.kyoku_index
        ));
    }
    if previous.next_oya == 0 {
        Ok((previous.bakaze.saturating_add(1), 1))
    } else {
        Ok((previous.bakaze, previous.kyoku.saturating_add(1)))
    }
}

fn score_total(scores: &[i32]) -> i64 {
    scores.iter().map(|score| i64::from(*score)).sum()
}

/// Deterministic match id. The three standard profiles keep a short form; room-level
/// variants append a SHA-256 prefix of the full profile, so replays with the same
/// platform and seed but different rules do not collide.
pub(crate) fn match_id(
    variant: &str,
    kind: MatchKind,
    seed: u64,
    rule_profile: RiichiRuleProfile,
) -> String {
    let standard = RiichiRuleProfile::from_platform(rule_profile.platform());
    if rule_profile == RiichiRuleProfile::tenhou() {
        format!("ft-{variant}-{}-{seed:016x}", kind.as_str())
    } else if rule_profile == standard {
        format!(
            "ft-{variant}-{}-{}-{seed:016x}",
            kind.as_str(),
            rule_profile.platform().as_str()
        )
    } else {
        let digest = Sha256::digest(format!("{rule_profile:?}").as_bytes());
        let fingerprint = digest[..8]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        format!(
            "ft-{variant}-{}-{}-{fingerprint}-{seed:016x}",
            kind.as_str(),
            rule_profile.platform().as_str()
        )
    }
}

/// Derives each hand's wall seed (determined by the match seed, distinct per hand).
pub(crate) fn kyoku_seed(base: (u64, u64), kyoku_serial: u64) -> (u64, u64) {
    let mix = kyoku_serial.wrapping_mul(KYOKU_SEED_MIX);
    (base.0 ^ mix, base.1.wrapping_add(mix))
}

/// Prefix policy for the `session_id` sent to plugin subprocesses.
///
/// Model packages and plugins may use `session_id` as a cache key, so routing the
/// public headless API through the live core must not change it. Two explicit policies:
/// - [`SessionIdPolicy::LegacyHeadless`] -> `run-match-{p}p-{seed}-seat{seat}`: used by
///   the public headless paths ([`run_full_match_4p`] / [`run_full_match_3p`] /
///   [`run_match_4p`] / [`run_match_3p`] / CLI `run-match`).
/// - [`SessionIdPolicy::Live`] -> `live-match-{p}p-{seed}-seat{seat}`: used by the
///   product live API ([`crate::live::LiveMatchSession::new_4p`] / `new_3p`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionIdPolicy {
    /// Product live API: `live-match-*`.
    Live,
    /// Legacy public headless API: `run-match-*` (kept stable for plugin session keys).
    LegacyHeadless,
}

impl SessionIdPolicy {
    /// Plugin subprocess `session_id` for a seat under this policy (`players` is 3 or 4).
    pub fn plugin_session_id(self, players: u8, seed: u64, seat: u8) -> String {
        let prefix = match self {
            SessionIdPolicy::Live => "live-match",
            SessionIdPolicy::LegacyHeadless => "run-match",
        };
        format!("{prefix}-{players}p-{seed}-seat{seat}")
    }
}

/// Plays one full 4p hand (legacy single-hand API). `specs` must have length 4.
pub fn run_match_4p(seed: u64, specs: Vec<SeatSpec>) -> Result<MatchReport> {
    let full = run_full_match_4p(MatchConfig::single(seed), specs)?;
    Ok(MatchReport::from_single(full))
}

/// Plays a single hand, tonpuusen or hanchan in 4p (full lifecycle). `specs` must have length 4.
pub fn run_full_match_4p(config: MatchConfig, specs: Vec<SeatSpec>) -> Result<FullMatchReport> {
    if specs.len() != 4 {
        return Err(anyhow!("4p run-match needs exactly 4 seat specs"));
    }
    let mut agents: Vec<Box<dyn SeatAgent<Variant4p>>> = Vec::with_capacity(4);
    for (seat, spec) in specs.into_iter().enumerate() {
        agents.push(build_agent_4p(seat as u8, spec, config.seed, config.kind)?);
    }
    run_full_match_4p_with_agents(config, agents)
}

/// Plays a 4p match with agents supplied by the caller (for custom seats). `agents` must have length 4.
///
/// Fully automatic headless semantics: driven through the boxed-agent constructor of
/// [`LiveMatchSession`](crate::live::LiveMatchSession), sharing the one match host core
/// with the product live API. Manual pending is not supported; manual seats that must
/// stop at external decision points use [`crate::live::LiveSeat::Manual`] through `new_4p`.
pub fn run_full_match_4p_with_agents(
    config: MatchConfig,
    agents: Vec<Box<dyn SeatAgent<Variant4p>>>,
) -> Result<FullMatchReport> {
    if agents.len() != 4 {
        return Err(anyhow!("4p run-match needs exactly 4 seat agents"));
    }
    let mut session = crate::live::LiveMatchSession::new_4p_with_agents(
        crate::live::live_config(config),
        agents,
    )?;
    session.run_to_match_end()?;
    Ok(session.report())
}

#[allow(clippy::too_many_arguments)]
/// Head bump (atama-hane) helper: with multiple ron, pick the winner closest to the
/// discarder in turn order (starting from the next seat). Only used for the final
/// ruling when the room profile selects head bump; other profiles keep multiple ron.
///
/// The distance `(seat + players - discarder) % players` is unique within
/// `1..=players-1`, so there are no ties, and the result does not depend on iterating
/// seats in absolute order.
pub(crate) fn select_ron_winner(candidates: &[u8], discarder: u8, players: u8) -> Option<u8> {
    candidates
        .iter()
        .copied()
        .min_by_key(|&seat| (seat + players - discarder) % players)
}

/// Plays one full 3p hand (legacy single-hand API). `specs` must have length 3.
pub fn run_match_3p(seed: u64, specs: Vec<SeatSpec>) -> Result<MatchReport> {
    let full = run_full_match_3p(MatchConfig::single(seed), specs)?;
    Ok(MatchReport::from_single(full))
}

/// Plays a single hand, tonpuusen or hanchan in 3p (full lifecycle). `specs` must have length 3.
pub fn run_full_match_3p(config: MatchConfig, specs: Vec<SeatSpec>) -> Result<FullMatchReport> {
    if specs.len() != 3 {
        return Err(anyhow!("3p run-match needs exactly 3 seat specs"));
    }
    let mut agents: Vec<Box<dyn SeatAgent<Variant3p>>> = Vec::with_capacity(3);
    for (seat, spec) in specs.into_iter().enumerate() {
        agents.push(build_agent_3p(seat as u8, spec, config.seed, config.kind)?);
    }
    run_full_match_3p_with_agents(config, agents)
}

/// Plays a 3p match with agents supplied by the caller (for custom seats). `agents` must have length 3.
///
/// Fully automatic headless semantics, as [`run_full_match_4p_with_agents`]: driven
/// through the boxed-agent entry of
/// [`LiveMatchSession`](crate::live::LiveMatchSession), with no manual pending.
pub fn run_full_match_3p_with_agents(
    config: MatchConfig,
    agents: Vec<Box<dyn SeatAgent<Variant3p>>>,
) -> Result<FullMatchReport> {
    if agents.len() != 3 {
        return Err(anyhow!("3p run-match needs exactly 3 seat agents"));
    }
    let mut session = crate::live::LiveMatchSession::new_3p_with_agents(
        crate::live::live_config(config),
        agents,
    )?;
    session.run_to_match_end()?;
    Ok(session.report())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decision_trace(
    ctx: KyokuContext,
    turn_index: u32,
    decision_id: String,
    state_digest: String,
    seat: u8,
    phase: &'static str,
    legal_count: usize,
    legal_actions: Vec<String>,
    submitted_action: String,
    final_action: String,
    agent_kind: &'static str,
    model_id: String,
    latency_ms: u64,
    fallback: Option<String>,
    selected_in_legal: bool,
    error_code: Option<&'static str>,
) -> DecisionTrace {
    let error = error_code.map(|code| DecisionErrorTrace {
        code: code.to_string(),
        category: decision_error_category(code).to_string(),
        legal_actions,
        detail: fallback.clone(),
    });
    DecisionTrace {
        decision_id,
        state_digest,
        kyoku_index: ctx.kyoku_index,
        bakaze: ctx.bakaze,
        kyoku: ctx.kyoku,
        honba: ctx.honba,
        kyotaku: ctx.kyotaku,
        turn_index,
        seat,
        phase: phase.to_string(),
        legal_count,
        selected: final_action.clone(),
        submitted_action,
        final_action,
        agent_kind: agent_kind.to_string(),
        model_id,
        latency_ms,
        fallback,
        selected_in_legal,
        error,
    }
}

fn decision_error_category(code: &str) -> &'static str {
    if code.starts_with("MODEL_") || code.starts_with("MANUAL_") {
        "model"
    } else if code.starts_with("PLATFORM_") {
        "platform_adapter"
    } else if code.starts_with("RULE_") {
        "rule_engine"
    } else {
        "internal"
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_report(
    variant: &'static str,
    players: u8,
    config: MatchConfig,
    start_score: i32,
    kyokus: Vec<KyokuRecord>,
    current_attempt: Option<CurrentKyokuAttempt>,
    match_ended: bool,
    current_scores: Option<Vec<i32>>,
    failure: Option<MatchFailure>,
    agents: Vec<SeatAgentStatus>,
) -> FullMatchReport {
    let current_scores = current_scores.unwrap_or_else(|| {
        kyokus
            .last()
            .map(|k| k.scores_after.clone())
            .unwrap_or_else(|| vec![start_score; players as usize])
    });
    let failed = failure.is_some();
    let ended = match_ended && !failed;
    let completion = if failed {
        "failed".to_string()
    } else if ended {
        "completed".to_string()
    } else {
        "in_progress".to_string()
    };
    let final_scores = if ended {
        current_scores.clone()
    } else {
        Vec::new()
    };
    let ranks = if ended {
        rankings(&final_scores)
    } else {
        Vec::new()
    };
    FullMatchReport {
        schema_version: TRACE_SCHEMA_VERSION,
        match_id: match_id(variant, config.kind, config.seed, config.rule_profile),
        variant: variant.to_string(),
        rule_profile: config.rule_profile.platform().as_str().to_string(),
        red_fives: config.rule_profile.red_fives(players as usize),
        players,
        seed: config.seed,
        length: config.kind.as_str().to_string(),
        start_scores: vec![start_score; players as usize],
        kyokus,
        current_attempt: if ended { None } else { current_attempt },
        current_scores,
        final_scores,
        rankings: ranks,
        ended,
        completion,
        failure,
        agents,
    }
}

fn build_agent_4p(
    seat: u8,
    spec: SeatSpec,
    seed: u64,
    kind: MatchKind,
) -> Result<Box<dyn SeatAgent<Variant4p>>> {
    match spec {
        SeatSpec::Algorithm(algo) => Ok(Box::new(AlgorithmSeatAgent::new(seat, algo))),
        SeatSpec::Plugin(rt) => {
            let session = SessionIdPolicy::LegacyHeadless.plugin_session_id(4, seed, seat);
            Ok(Box::new(LocalPluginSeatAgent::from_runtime_for_variant::<
                Variant4p,
            >(seat, &rt, session, kind)?))
        }
        SeatSpec::RemoteModel(config) => Ok(Box::new(RemoteModelSeat::new(config))),
    }
}

fn build_agent_3p(
    seat: u8,
    spec: SeatSpec,
    seed: u64,
    kind: MatchKind,
) -> Result<Box<dyn SeatAgent<Variant3p>>> {
    match spec {
        SeatSpec::Algorithm(algo) => Ok(Box::new(AlgorithmSeatAgent::new(seat, algo))),
        SeatSpec::Plugin(rt) => {
            let session = SessionIdPolicy::LegacyHeadless.plugin_session_id(3, seed, seat);
            Ok(Box::new(LocalPluginSeatAgent::from_runtime_for_variant::<
                Variant3p,
            >(seat, &rt, session, kind)?))
        }
        SeatSpec::RemoteModel(config) => Ok(Box::new(RemoteModelSeat::new(config))),
    }
}

pub(crate) fn update_call(
    call: &mut Option<(u8, u8, ReactionAction)>,
    priority: u8,
    seat: u8,
    action: ReactionAction,
    discarder: u8,
    seats: u8,
) {
    let better = match call {
        None => true,
        Some((p, current, _)) => {
            priority > *p
                || (priority == *p
                    && call_distance(discarder, seat, seats)
                        < call_distance(discarder, *current, seats))
        }
    };
    if better {
        *call = Some((priority, seat, action));
    }
}

fn call_distance(discarder: u8, caller: u8, seats: u8) -> u8 {
    (caller + seats - discarder) % seats
}

pub(crate) fn events_4p(board: &Board4p) -> Vec<Event4p> {
    let mut events = Vec::with_capacity(board.log.len() + 1);
    events.push(Variant4p::start_game_event());
    events.extend(board.log.iter().cloned());
    events
}

pub(crate) fn events_3p(board: &Board3p) -> Vec<Event3p> {
    let mut events = Vec::with_capacity(board.log.len() + 1);
    events.push(Variant3p::start_game_event());
    events.extend(board.log.iter().cloned());
    events
}

pub(crate) fn fallback_turn(legal: &[TurnAction]) -> TurnAction {
    legal
        .iter()
        .find(|action| action.is_plain_discard())
        .or_else(|| legal.first())
        .cloned()
        .expect("legal turn actions are non-empty")
}

pub(crate) fn reaction_in_legal(action: &ReactionAction, legal: &[ReactionAction]) -> bool {
    // Pass is always an implicitly legal choice.
    matches!(action, ReactionAction::Pass)
        || legal.iter().any(|candidate| candidate.equivalent(action))
}

pub(crate) fn describe_turn(action: &TurnAction) -> String {
    match action {
        TurnAction::Discard { tile, tsumogiri } => {
            format!(
                "dahai {tile}{}",
                if *tsumogiri { " (tsumogiri)" } else { "" }
            )
        }
        TurnAction::DealerOpeningDiscard { tile } => {
            format!("dealer_opening_dahai {tile}")
        }
        TurnAction::Riichi { tile, .. } => format!("riichi+dahai {tile}"),
        TurnAction::DealerOpeningRiichi { tile } => {
            format!("dealer_opening_riichi+dahai {tile}")
        }
        TurnAction::Ankan { tile } => format!("ankan {tile}"),
        TurnAction::Kakan { tile } => format!("kakan {tile}"),
        TurnAction::Tsumo => "tsumo".to_string(),
        TurnAction::Nukidora => "nukidora".to_string(),
        TurnAction::KyuushuKyuuhai => "kyushukyuhai".to_string(),
    }
}

pub(crate) fn describe_reaction(action: &ReactionAction) -> String {
    match action {
        ReactionAction::Pass => "pass".to_string(),
        ReactionAction::Pon { .. } => "pon".to_string(),
        ReactionAction::Chi { .. } => "chi".to_string(),
        ReactionAction::Daiminkan => "daiminkan".to_string(),
        ReactionAction::Ron => "ron".to_string(),
    }
}

pub(crate) fn summarize_4p(outcome: &KyokuOutcome) -> String {
    match outcome {
        KyokuOutcome::Hora {
            winner,
            from,
            score,
        } => summarize_hora(*winner, *from, score),
        KyokuOutcome::MultiHora { first, additional } => {
            let mut winners = vec![first.winner];
            winners.extend(additional.iter().map(|win| win.winner));
            format!("multi_ron winners={winners:?} from={:?}", first.from)
        }
        KyokuOutcome::Ryukyoku { tenpai } => format!("ryukyoku tenpai={tenpai:?}"),
        KyokuOutcome::NagashiMangan { winners, tenpai } => {
            format!("nagashi_mangan winners={winners:?} tenpai={tenpai:?}")
        }
        KyokuOutcome::AbortiveRyukyoku { reason } => format!("abortive_ryukyoku {reason:?}"),
    }
}

pub(crate) fn summarize_3p(outcome: &KyokuOutcome3p) -> String {
    match outcome {
        KyokuOutcome3p::Hora {
            winner,
            from,
            score,
        } => summarize_hora(*winner, *from, score),
        KyokuOutcome3p::MultiHora { first, additional } => {
            let mut winners = vec![first.winner];
            winners.extend(additional.iter().map(|win| win.winner));
            format!("multi_ron winners={winners:?} from={:?}", first.from)
        }
        KyokuOutcome3p::Ryukyoku { tenpai } => format!("ryukyoku tenpai={tenpai:?}"),
        KyokuOutcome3p::NagashiMangan { winners, tenpai } => {
            format!("nagashi_mangan winners={winners:?} tenpai={tenpai:?}")
        }
        KyokuOutcome3p::AbortiveRyukyoku { reason } => format!("abortive_ryukyoku {reason:?}"),
    }
}

fn summarize_hora(winner: u8, from: Option<u8>, score: &flytable_core::score::FullScore) -> String {
    let how = match from {
        Some(f) => format!("ron(from={f})"),
        None => "tsumo".to_string(),
    };
    let yaku: Vec<String> = score
        .yaku
        .yaku
        .iter()
        .map(|(y, h)| format!("{}({h})", y.name()))
        .collect();
    if score.score.yakuman > 0 {
        format!(
            "hora winner={winner} {how} yakuman x{} yaku=[{}]",
            score.score.yakuman,
            yaku.join(",")
        )
    } else {
        format!(
            "hora winner={winner} {how} {}han {}fu (dora {}) yaku=[{}]",
            score.yaku.han + score.dora_han,
            score.fu,
            score.dora_han,
            yaku.join(",")
        )
    }
}
