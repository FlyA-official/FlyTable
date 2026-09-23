//! Product-level live match runtime [`LiveMatchSession`], the base for GUI-driven live sessions.
//!
//! The headless `run_full_match_*` in [`crate::match_host`] runs fully automatic seats
//! to the end in one go. The live runtime adds the ability to stop at external
//! decision points: when a manual seat is to discard, or a manual seat has legal
//! responses, the runtime pauses and exposes a [`PendingWindow`]. The caller injects
//! the real action with [`LiveMatchSession::submit_action`] /
//! [`LiveMatchSession::submit_reaction`] and resumes with
//! [`LiveMatchSession::continue_after_submit`]. Algorithm and local plugin seats are
//! still asked and advanced automatically.
//!
//! - A manual seat's turn produces a turn [`PendingWindow`]; `ManualSeat::decide_turn`
//!   is never called to fake progress.
//! - A response window with a manual responder produces a reaction [`PendingWindow`].
//!   After every seat has submitted, head bump / multiple ron / triple ron draw are
//!   resolved by the profile; ron beats pon/kan, which beats chi.
//! - Algorithm and plugin seats are asked automatically. Illegal actions, abstains
//!   and errors use the existing host fallback and trace semantics and never panic.
//! - [`LiveStepResult`] reports hand end and match end; the next hand is started by the
//!   same session ([`LiveMatchSession::start_next_kyoku`]) without rebuilding state.
//!
//! This module only orchestrates; progression is driven by `Board4p` / `Board3p`.
//! There are no observations, masks or tensors. Information isolation is guaranteed
//! by the `SeatView` types ([`LiveMatchSession::snapshot`] goes through `view_for`,
//! which cannot hold other seats' hands). Trace and next-hand seed derivation reuse
//! `match_host`.

use anyhow::{anyhow, bail, Result};
use sha2::{Digest, Sha256};

use flytable_core::meld::Meld;
use flytable_core::rules::{RedFiveCounts, RiichiRuleProfile, RonResolution};
use flytable_core::tile::Tile;
use flytable_event::matchlog::WindowPhase;
use flytable_event::{Event3p, Event4p};
use flytable_seat::contract::{project_3p, project_4p, VisibleEvent3p, VisibleEvent4p};
use flytable_table::legal::legal_turn_actions;
use flytable_table::progress::{self, rankings, MatchLength, RoundState, Settlement};
use flytable_table::tileset::{
    tileset_3p, tileset_3p_with_red_fives, tileset_4p, tileset_4p_with_red_fives,
};
use flytable_table::wall::Wall;
use flytable_table::{
    Board3p, Board4p, KyokuOutcome, KyokuOutcome3p, PendingRobbery, ReactionAction, SeatView,
    TurnAction,
};

use crate::agent::{DecisionTrace, ReactionRequest, SeatAgent, SeatAgentStatus, TurnRequest};
use crate::algorithm::{AlgorithmKind, AlgorithmSeatAgent};
use crate::decision_window::{reaction_window, turn_window, HostWindow, WindowFault};
use crate::match_host::{
    decision_trace, describe_reaction, describe_turn, events_3p, events_4p, finish_report,
    kyoku_seed, match_id, reaction_in_legal, select_ron_winner, summarize_3p, summarize_4p,
    update_call, CurrentKyokuAttempt, FullMatchReport, KyokuContext, MatchConfig, MatchFailure,
    MatchKind, SeatSpec, SessionIdPolicy, KYOKU_SEED_LO, TRACE_SCHEMA_VERSION,
};
use crate::placeholders::{ManualSeat, RemoteModelConfig, RemoteModelSeat};
use crate::plugin::LocalPluginSeatAgent;
use crate::variant::{Variant, Variant3p, Variant4p};

use flytable_inference_host::registry::CertifiedPluginRuntime;

const MAX_STEPS: u64 = 100_000;
/// Safety valve: an abnormally long hand ends in a draw instead of looping forever.
const MAX_LOG_EVENTS: usize = 12_000;

/// Live match configuration (same meaning as [`MatchConfig`], a separate type for the live API).
#[derive(Debug, Clone, Copy)]
pub struct LiveMatchConfig {
    pub seed: u64,
    pub kind: MatchKind,
    /// Platform rules; Tenhou by default.
    pub rule_profile: RiichiRuleProfile,
    /// Starting score per seat; `None` uses the profile's default for the seat count.
    pub start_score: Option<i32>,
    /// Whether to record L3 decision windows, the raw material for game review and training samples. Off by default.
    ///
    /// It is an observability option, not a rule: each decision builds an extra canonical
    /// candidate set, which only deployments producing match logs need. When off, the
    /// whole path costs nothing (not even per-seat views are built).
    pub record_decision_windows: bool,
}

impl LiveMatchConfig {
    pub fn new(seed: u64, kind: MatchKind) -> Self {
        Self {
            seed,
            kind,
            rule_profile: RiichiRuleProfile::default(),
            start_score: None,
            record_decision_windows: false,
        }
    }

    pub fn single(seed: u64) -> Self {
        Self {
            seed,
            kind: MatchKind::Single,
            rule_profile: RiichiRuleProfile::default(),
            start_score: None,
            record_decision_windows: false,
        }
    }

    pub fn with_rule_profile(mut self, rule_profile: RiichiRuleProfile) -> Self {
        self.rule_profile = rule_profile;
        self
    }

    /// Turns L3 decision window recording on or off (see [`Self::record_decision_windows`]).
    #[must_use]
    pub fn with_decision_windows(mut self, record: bool) -> Self {
        self.record_decision_windows = record;
        self
    }

    fn as_match_config(&self) -> MatchConfig {
        MatchConfig {
            seed: self.seed,
            kind: self.kind,
            rule_profile: self.rule_profile,
            start_score: self.start_score,
        }
    }
}

/// Who sits in a seat (from the product's point of view). `Manual` waits for an
/// injected action (a human seat in a GUI or CLI, or filled in by an upstream
/// product); `Algorithm` / `Plugin` are driven by the runtime.
///
/// `RemoteModel` is driven through an HTTP inference endpoint; networking stays in
/// `flytable-inference-host`, and this module only handles the seat lifecycle and
/// fallback.
pub enum LiveSeat {
    Manual,
    Algorithm(AlgorithmKind),
    Plugin(CertifiedPluginRuntime),
    RemoteModel(RemoteModelConfig),
}

/// Kind of pending window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PendingKind {
    /// A manual seat is to discard.
    Turn,
    /// Some manual seats have legal responses to a discard.
    Reaction,
    /// Some manual seats can rob a kan or nukidora declaration.
    Robbery,
}

/// A manual seat awaiting a response and its legal responses (excluding Pass, which is always implicitly legal).
#[derive(Debug, Clone)]
pub struct PendingReactionSeat {
    pub seat: u8,
    pub legal: Vec<ReactionAction>,
}

/// The window currently waiting for an injected action.
#[derive(Debug, Clone)]
pub struct PendingWindow {
    /// Globally monotonic decision window id (checked on submit to reject actions for the wrong window).
    pub decision_id: String,
    /// SHA-256 digest binding the window's authoritative state and legal set.
    pub state_digest: String,
    pub kind: PendingKind,
    /// Turn: the waiting seat. Reaction: the discarder.
    pub actor: u8,
    /// Reaction: the discarded tile and its discarder.
    pub discarder: Option<u8>,
    pub tile: Option<Tile>,
    /// Legal turn actions of a turn window.
    pub turn_actions: Vec<TurnAction>,
    /// Manual seats in a reaction window that have not submitted yet (emptied as they submit).
    pub reaction_seats: Vec<PendingReactionSeat>,
}

impl PendingWindow {
    /// Seats waiting for an action (one seat for a turn; the remaining manual seats for a reaction).
    pub fn waiting_seats(&self) -> Vec<u8> {
        match self.kind {
            PendingKind::Turn => vec![self.actor],
            PendingKind::Reaction | PendingKind::Robbery => {
                self.reaction_seats.iter().map(|r| r.seat).collect()
            }
        }
    }
}

/// State after a step or submission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiveStepResult {
    /// Stopped at a manual seat's turn.
    PendingTurn { seat: u8 },
    /// Stopped at a response window; lists the manual seats yet to submit.
    PendingReaction { seats: Vec<u8> },
    /// An action was applied but no stable stopping point has been reached (call `continue_after_submit`).
    InProgress,
    /// The hand ended (`start_next_kyoku` continues).
    KyokuEnded,
    /// The match ended.
    MatchEnded,
    /// Host failure; not a mahjong result. The state is frozen.
    Failed { failure: MatchFailure },
}

/// Product-facing rendering of a meld.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MeldSnapshot {
    /// `chi` / `pon` / `daiminkan` / `kakan` / `ankan` / `nukidora`.
    pub kind: &'static str,
    pub tiles: Vec<String>,
}

/// What is visible about a seat in a snapshot (from the viewer's perspective: only the viewer's own hand is included).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SeatSnapshot {
    pub seat: u8,
    /// The viewer's own hand (red fives included); `None` for other seats, whose hands are never sent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hand: Option<Vec<String>>,
    /// Number of tiles in the closed hand (public).
    pub hand_count: u8,
    pub melds: Vec<MeldSnapshot>,
    pub discards: Vec<String>,
    pub riichi: bool,
    /// The viewer's own temporary and riichi furiten; `None` for other seats (furiten is private).
    ///
    /// Discard furiten is not included: it follows from the discards, which consumers
    /// already compare against the waits. These two cannot be derived from public
    /// information, especially temporary furiten from a passive missed ron without a
    /// yaku: ron is not in the legal responses then, so there is no pass to observe and
    /// no other way to know the seat is furiten.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub furiten: Option<bool>,
    /// 3-player nukidora count (`None` in 4-player).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nuki: Option<u8>,
}

/// Summary of the pending window in a snapshot.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PendingSummary {
    pub decision_id: String,
    pub state_digest: String,
    pub kind: PendingKind,
    /// Seats waiting for an action.
    pub seats: Vec<u8>,
    /// Number of legal actions (turn: the seat's legal actions; reaction: the sum over waiting seats).
    pub legal_action_count: usize,
}

/// Product-facing snapshot of a live match (rule results, from the viewer's perspective).
#[derive(Debug, Clone, serde::Serialize)]
pub struct LiveMatchSnapshot {
    pub schema_version: u32,
    /// `riichi4p` / `riichi3p`.
    pub variant: &'static str,
    /// Platform rules; not derivable from the seat-count variant.
    pub rule_profile: &'static str,
    /// Red five counts of the current wall.
    pub red_fives: RedFiveCounts,
    pub players: u8,
    /// `awaiting_turn` / `awaiting_reaction` / `in_progress` / `kyoku_end` /
    /// `match_end` / `failed`.
    pub phase: &'static str,
    pub kyoku_index: u32,
    /// Round wind: 0 = East, 1 = South, 2 = West, 3 = North.
    pub bakaze: u8,
    pub kyoku: u8,
    pub honba: u8,
    pub kyotaku: u8,
    pub scores: Vec<i32>,
    /// Dealer seat.
    pub dealer: u8,
    /// Seat whose turn it is.
    pub current_turn: u8,
    /// Seat waiting for a decision (the turn window's seat; `None` without a pending window).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actor: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pending: Option<PendingSummary>,
    pub dora_indicators: Vec<String>,
    pub wall_remaining: u32,
    /// Per-seat visible information from the viewer's perspective.
    pub seats: Vec<SeatSnapshot>,
    pub ended: bool,
    /// A host failure is not `ended`; callers must handle this field separately.
    pub failed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<MatchFailure>,
    /// Result summary of the hand (only after it ends).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    /// Rankings when the match has ended (seats by descending score), otherwise empty.
    pub rankings: Vec<u8>,
    pub agents: Vec<SeatAgentStatus>,
}

/// Status of one seat (health, latency, counts, and whether it is manual).
#[derive(Debug, Clone, serde::Serialize)]
pub struct LiveSeatStatus {
    pub seat: u8,
    pub agent_kind: String,
    pub model_id: String,
    /// `manual` / `healthy` / `degraded` / `error`.
    pub state: &'static str,
    pub manual: bool,
    pub calls: u64,
    pub fallbacks: u64,
    pub errors: u64,
    pub last_latency_ms: Option<u64>,
}

/// Runtime status snapshot of a live match.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LiveMatchStatus {
    pub schema_version: u32,
    pub variant: &'static str,
    pub rule_profile: &'static str,
    pub red_fives: RedFiveCounts,
    pub match_id: String,
    pub phase: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<MatchFailure>,
    pub seats: Vec<LiveSeatStatus>,
}

/// Table abstraction that lets the [`LiveMatchSession`] core be written once for 4-player and 3-player.
///
/// Only covers what the live core needs from a table: projection, progression,
/// adjudication, settlement and result summaries. The rules stay in `Board4p` /
/// `Board3p`; this trait only unifies their common shape.
pub trait LiveBoard {
    /// Variant (event stream, legal action wire, host calls, per-seat projection).
    type V: VisibleVariant;
    /// Hand result type (`KyokuOutcome` / `KyokuOutcome3p`).
    type Outcome;

    const SEATS: u8;
    const VARIANT: &'static str;

    /// Starts a hand from a seed and match state (`StartKyoku` already in the log).
    fn start_kyoku(seed: (u64, u64), state: &RoundState, rule_profile: RiichiRuleProfile) -> Self;

    fn turn(&self) -> u8;
    fn last_discard(&self) -> Option<(u8, Tile)>;
    fn clear_last_discard(&mut self) -> Option<Self::Outcome>;
    fn hand_len(&self, seat: u8) -> usize;
    fn wall_remaining(&self) -> u32;
    fn dora_indicators(&self) -> Vec<Tile>;
    fn scores(&self) -> Vec<i32>;
    fn oya(&self) -> u8;
    fn kyoku(&self) -> u8;
    fn honba(&self) -> u8;
    fn kyotaku(&self) -> u8;
    fn log_len(&self) -> usize;
    /// Nukidora count of a seat (`None` in 4-player).
    fn nuki(&self, seat: u8) -> Option<u8>;
    /// Closed hand of a seat, for snapshot projection.
    fn seat_hand(&self, seat: u8) -> Vec<Tile>;
    fn seat_melds(&self, seat: u8) -> Vec<Meld>;
    fn seat_discards(&self, seat: u8) -> Vec<Tile>;
    fn seat_riichi(&self, seat: u8) -> bool;

    /// Authoritative event log slice (without the `StartGame` synthesized for the wire).
    ///
    /// Same stream and indices as [`Self::log_len`]; L3 window anchors count in it.
    fn log_slice(&self) -> &[<Self::V as Variant>::Event];
    /// Event stream for agents (with the opening `StartGame`, as in `match_host`).
    fn events_with_start_game(&self) -> Vec<<Self::V as Variant>::Event>;
    /// Seats with accepted riichi (`ReachAccepted`) in the log, for stick accounting.
    fn reach_accepted_seats(&self) -> Vec<u8>;
    /// Visible events of a seat since the start of the hand (other seats' deals and draws masked).
    fn visible_events_for(&self, seat: u8) -> Vec<VisibleEventOf<Self>>;

    fn view_for(&self, seat: u8) -> SeatView;
    fn view_for_reaction(&self, seat: u8) -> SeatView;
    /// Authoritative legal turn actions. Defaults to the shared maintainer enumeration; implementations may override.
    fn legal_turn_actions(&self, seat: u8) -> Vec<TurnAction> {
        legal_turn_actions(&self.view_for(seat))
    }
    fn legal_reactions(&self, seat: u8) -> Vec<ReactionAction>;
    fn pending_robbery(&self) -> Option<PendingRobbery>;
    fn ron_resolution(&self) -> RonResolution;
    /// Records that the seat passed on a legal ron.
    fn note_missed_ron(&mut self, seat: u8) -> Result<(), String>;

    fn draw_for_turn(&mut self) -> Option<Tile>;
    fn ryukyoku(&mut self) -> Self::Outcome;
    fn apply_turn(&mut self, action: TurnAction) -> Result<Option<Self::Outcome>, String>;
    fn apply_ron(&mut self, seat: u8, discarder: u8, tile: Tile) -> Result<Self::Outcome, String>;
    fn apply_rons(
        &mut self,
        seats: &[u8],
        discarder: u8,
        tile: Tile,
    ) -> Result<Self::Outcome, String>;
    fn apply_robbery_rons(&mut self, seats: &[u8]) -> Result<Self::Outcome, String>;
    fn resolve_pending_robbery_passes(&mut self) -> Result<Option<Self::Outcome>, String>;
    fn abort_sanchaho(&mut self) -> Result<Self::Outcome, String>;
    /// Applies pon / chi / open kan (chi is illegal in 3-player).
    fn apply_call(
        &mut self,
        seat: u8,
        action: &ReactionAction,
    ) -> Result<Option<Self::Outcome>, String>;
    fn advance_turn(&mut self);

    fn settle(
        &self,
        state: &RoundState,
        length: MatchLength,
        outcome: &Self::Outcome,
    ) -> Settlement;
    fn summarize(outcome: &Self::Outcome) -> String;
}

/// Per-seat visible event type of a `LiveBoard`.
pub type VisibleEventOf<B> = <<B as LiveBoard>::V as VisibleVariant>::Visible;

/// Bridge from a variant to its per-seat visible events (4-player `VisibleEvent4p`, 3-player `VisibleEvent3p`).
pub trait VisibleVariant: Variant {
    type Visible;
}
impl VisibleVariant for Variant4p {
    type Visible = VisibleEvent4p;
}
impl VisibleVariant for Variant3p {
    type Visible = VisibleEvent3p;
}

pub use flytable_table::progress::hora_payment_with_pao;

impl LiveBoard for Board4p {
    type V = Variant4p;
    type Outcome = KyokuOutcome;
    const SEATS: u8 = 4;
    const VARIANT: &'static str = "riichi4p";

    fn start_kyoku(seed: (u64, u64), state: &RoundState, rule_profile: RiichiRuleProfile) -> Self {
        let tiles = tileset_4p_with_red_fives(rule_profile.red_fives(4))
            .expect("RiichiRuleProfile only exposes validated 4p red-five counts");
        let wall = Wall::shuffled(tiles, 4, seed);
        let scores: [i32; 4] = std::array::from_fn(|i| state.scores[i]);
        Board4p::from_wall_with_state(
            wall,
            scores,
            state.bakaze,
            state.kyoku,
            state.oya,
            state.honba,
            state.kyotaku,
        )
        .with_rule_profile(rule_profile)
        .expect("wall was constructed from the same validated rule profile")
    }

    fn turn(&self) -> u8 {
        self.turn
    }
    fn last_discard(&self) -> Option<(u8, Tile)> {
        self.last_discard
    }
    fn clear_last_discard(&mut self) -> Option<KyokuOutcome> {
        Board4p::clear_last_discard(self)
    }
    fn hand_len(&self, seat: u8) -> usize {
        self.players[seat as usize].hand.len()
    }
    fn wall_remaining(&self) -> u32 {
        self.wall.live_remaining() as u32
    }
    fn dora_indicators(&self) -> Vec<Tile> {
        self.wall.dora_indicators()
    }
    fn scores(&self) -> Vec<i32> {
        self.scores.to_vec()
    }
    fn oya(&self) -> u8 {
        self.oya
    }
    fn kyoku(&self) -> u8 {
        self.kyoku
    }
    fn honba(&self) -> u8 {
        self.honba
    }
    fn kyotaku(&self) -> u8 {
        self.kyotaku
    }
    fn log_len(&self) -> usize {
        self.log.len()
    }
    fn nuki(&self, _seat: u8) -> Option<u8> {
        None
    }
    fn seat_hand(&self, seat: u8) -> Vec<Tile> {
        self.players[seat as usize].hand.clone()
    }
    fn seat_melds(&self, seat: u8) -> Vec<Meld> {
        self.players[seat as usize].melds.clone()
    }
    fn seat_discards(&self, seat: u8) -> Vec<Tile> {
        self.players[seat as usize].discards.clone()
    }
    fn seat_riichi(&self, seat: u8) -> bool {
        self.players[seat as usize].riichi
    }

    fn log_slice(&self) -> &[Event4p] {
        &self.log
    }
    fn events_with_start_game(&self) -> Vec<Event4p> {
        events_4p(self)
    }
    fn reach_accepted_seats(&self) -> Vec<u8> {
        self.log
            .iter()
            .filter_map(|e| match e {
                Event4p::ReachAccepted { actor } => Some(*actor),
                _ => None,
            })
            .collect()
    }
    fn visible_events_for(&self, seat: u8) -> Vec<VisibleEvent4p> {
        let mut out = Vec::with_capacity(self.log.len() + 1);
        out.push(project_4p(&Variant4p::start_game_event(), seat));
        out.extend(self.log.iter().map(|e| project_4p(e, seat)));
        out
    }

    fn view_for(&self, seat: u8) -> SeatView {
        Board4p::view_for(self, seat)
    }
    fn view_for_reaction(&self, seat: u8) -> SeatView {
        Board4p::view_for_reaction(self, seat)
    }
    fn legal_reactions(&self, seat: u8) -> Vec<ReactionAction> {
        Board4p::legal_reactions(self, seat)
    }
    fn pending_robbery(&self) -> Option<PendingRobbery> {
        Board4p::pending_robbery(self)
    }
    fn ron_resolution(&self) -> RonResolution {
        self.rule_profile.ron_resolution()
    }
    fn note_missed_ron(&mut self, seat: u8) -> Result<(), String> {
        Board4p::note_missed_ron(self, seat)
    }

    fn draw_for_turn(&mut self) -> Option<Tile> {
        Board4p::draw_for_turn(self)
    }
    fn ryukyoku(&mut self) -> KyokuOutcome {
        Board4p::ryukyoku(self)
    }
    fn apply_turn(&mut self, action: TurnAction) -> Result<Option<KyokuOutcome>, String> {
        Board4p::apply_turn(self, action)
    }
    fn apply_ron(&mut self, seat: u8, discarder: u8, tile: Tile) -> Result<KyokuOutcome, String> {
        Board4p::apply_ron(self, seat, discarder, tile)
    }
    fn apply_rons(
        &mut self,
        seats: &[u8],
        discarder: u8,
        tile: Tile,
    ) -> Result<KyokuOutcome, String> {
        Board4p::apply_rons(self, seats, discarder, tile)
    }
    fn apply_robbery_rons(&mut self, seats: &[u8]) -> Result<KyokuOutcome, String> {
        Board4p::apply_robbery_rons(self, seats)
    }
    fn resolve_pending_robbery_passes(&mut self) -> Result<Option<KyokuOutcome>, String> {
        Board4p::resolve_pending_robbery_passes(self)
    }
    fn abort_sanchaho(&mut self) -> Result<KyokuOutcome, String> {
        Board4p::abort_sanchaho(self)
    }
    fn apply_call(
        &mut self,
        seat: u8,
        action: &ReactionAction,
    ) -> Result<Option<KyokuOutcome>, String> {
        match action {
            ReactionAction::Pon { consumed } => {
                Board4p::apply_pon(self, seat, *consumed).map(|()| None)
            }
            ReactionAction::Chi { consumed } => {
                Board4p::apply_chi(self, seat, *consumed).map(|()| None)
            }
            ReactionAction::Daiminkan => Board4p::apply_daiminkan(self, seat),
            ReactionAction::Ron | ReactionAction::Pass => Ok(None),
        }
    }
    fn advance_turn(&mut self) {
        Board4p::advance_turn(self)
    }

    fn settle(
        &self,
        state: &RoundState,
        length: MatchLength,
        outcome: &KyokuOutcome,
    ) -> Settlement {
        match outcome {
            KyokuOutcome::Hora {
                winner,
                from,
                score,
            } => progress::settle_hora_with_profile(
                state,
                length,
                hora_payment_with_pao(
                    *winner,
                    *from,
                    score,
                    self.pao_liability(*winner),
                    Self::SEATS as usize,
                ),
                self.rule_profile,
            ),
            KyokuOutcome::MultiHora { first, additional } => {
                let first_payment = hora_payment_with_pao(
                    first.winner,
                    first.from,
                    &first.score,
                    self.pao_liability(first.winner),
                    Self::SEATS as usize,
                );
                let additional_payments: Vec<_> = additional
                    .iter()
                    .map(|win| {
                        hora_payment_with_pao(
                            win.winner,
                            win.from,
                            &win.score,
                            self.pao_liability(win.winner),
                            Self::SEATS as usize,
                        )
                    })
                    .collect();
                progress::settle_multi_hora_with_profile(
                    state,
                    length,
                    first_payment,
                    &additional_payments,
                    self.rule_profile,
                )
            }
            KyokuOutcome::Ryukyoku { tenpai } => {
                progress::settle_ryukyoku_with_profile(state, length, tenpai, self.rule_profile)
            }
            KyokuOutcome::NagashiMangan { winners, tenpai } => {
                progress::settle_nagashi_mangan_with_profile(
                    state,
                    length,
                    winners,
                    tenpai,
                    self.rule_profile,
                )
            }
            KyokuOutcome::AbortiveRyukyoku { .. } => {
                progress::settle_abortive_ryukyoku_with_profile(state, length, self.rule_profile)
            }
        }
    }
    fn summarize(outcome: &KyokuOutcome) -> String {
        summarize_4p(outcome)
    }
}

impl LiveBoard for Board3p {
    type V = Variant3p;
    type Outcome = KyokuOutcome3p;
    const SEATS: u8 = 3;
    const VARIANT: &'static str = "riichi3p";

    fn start_kyoku(seed: (u64, u64), state: &RoundState, rule_profile: RiichiRuleProfile) -> Self {
        let tiles = tileset_3p_with_red_fives(rule_profile.red_fives(3))
            .expect("RiichiRuleProfile only exposes validated 3p red-five counts");
        let wall = Wall::shuffled(tiles, 3, seed);
        let scores: [i32; 3] = std::array::from_fn(|i| state.scores[i]);
        Board3p::from_wall_with_state(
            wall,
            scores,
            state.bakaze,
            state.kyoku,
            state.oya,
            state.honba,
            state.kyotaku,
        )
        .with_rule_profile(rule_profile)
        .expect("wall was constructed from the same validated rule profile")
    }

    fn turn(&self) -> u8 {
        self.turn
    }
    fn last_discard(&self) -> Option<(u8, Tile)> {
        self.last_discard
    }
    fn clear_last_discard(&mut self) -> Option<KyokuOutcome3p> {
        Board3p::clear_last_discard(self)
    }
    fn hand_len(&self, seat: u8) -> usize {
        self.players[seat as usize].hand.len()
    }
    fn wall_remaining(&self) -> u32 {
        self.wall.live_remaining() as u32
    }
    fn dora_indicators(&self) -> Vec<Tile> {
        self.wall.dora_indicators()
    }
    fn scores(&self) -> Vec<i32> {
        self.scores.to_vec()
    }
    fn oya(&self) -> u8 {
        self.oya
    }
    fn kyoku(&self) -> u8 {
        self.kyoku
    }
    fn honba(&self) -> u8 {
        self.honba
    }
    fn kyotaku(&self) -> u8 {
        self.kyotaku
    }
    fn log_len(&self) -> usize {
        self.log.len()
    }
    fn nuki(&self, seat: u8) -> Option<u8> {
        Some(self.nuki[seat as usize])
    }
    fn seat_hand(&self, seat: u8) -> Vec<Tile> {
        self.players[seat as usize].hand.clone()
    }
    fn seat_melds(&self, seat: u8) -> Vec<Meld> {
        self.players[seat as usize].melds.clone()
    }
    fn seat_discards(&self, seat: u8) -> Vec<Tile> {
        self.players[seat as usize].discards.clone()
    }
    fn seat_riichi(&self, seat: u8) -> bool {
        self.players[seat as usize].riichi
    }

    fn log_slice(&self) -> &[Event3p] {
        &self.log
    }
    fn events_with_start_game(&self) -> Vec<Event3p> {
        events_3p(self)
    }
    fn reach_accepted_seats(&self) -> Vec<u8> {
        self.log
            .iter()
            .filter_map(|e| match e {
                Event3p::ReachAccepted { actor } => Some(*actor),
                _ => None,
            })
            .collect()
    }
    fn visible_events_for(&self, seat: u8) -> Vec<VisibleEvent3p> {
        let mut out = Vec::with_capacity(self.log.len() + 1);
        out.push(project_3p(&Variant3p::start_game_event(), seat));
        out.extend(self.log.iter().map(|e| project_3p(e, seat)));
        out
    }

    fn view_for(&self, seat: u8) -> SeatView {
        Board3p::view_for(self, seat)
    }
    fn view_for_reaction(&self, seat: u8) -> SeatView {
        Board3p::view_for_reaction(self, seat)
    }
    fn legal_reactions(&self, seat: u8) -> Vec<ReactionAction> {
        Board3p::legal_reactions(self, seat)
    }
    fn pending_robbery(&self) -> Option<PendingRobbery> {
        Board3p::pending_robbery(self)
    }
    fn ron_resolution(&self) -> RonResolution {
        self.rule_profile.ron_resolution()
    }
    fn note_missed_ron(&mut self, seat: u8) -> Result<(), String> {
        Board3p::note_missed_ron(self, seat)
    }

    fn draw_for_turn(&mut self) -> Option<Tile> {
        Board3p::draw_for_turn(self)
    }
    fn ryukyoku(&mut self) -> KyokuOutcome3p {
        Board3p::ryukyoku(self)
    }
    fn apply_turn(&mut self, action: TurnAction) -> Result<Option<KyokuOutcome3p>, String> {
        Board3p::apply_turn(self, action)
    }
    fn apply_ron(&mut self, seat: u8, discarder: u8, tile: Tile) -> Result<KyokuOutcome3p, String> {
        Board3p::apply_ron(self, seat, discarder, tile)
    }
    fn apply_rons(
        &mut self,
        seats: &[u8],
        discarder: u8,
        tile: Tile,
    ) -> Result<KyokuOutcome3p, String> {
        Board3p::apply_rons(self, seats, discarder, tile)
    }
    fn apply_robbery_rons(&mut self, seats: &[u8]) -> Result<KyokuOutcome3p, String> {
        Board3p::apply_robbery_rons(self, seats)
    }
    fn resolve_pending_robbery_passes(&mut self) -> Result<Option<KyokuOutcome3p>, String> {
        Board3p::resolve_pending_robbery_passes(self)
    }
    fn abort_sanchaho(&mut self) -> Result<KyokuOutcome3p, String> {
        Err("triple ron draws do not exist in 3-player".to_string())
    }
    fn apply_call(
        &mut self,
        seat: u8,
        action: &ReactionAction,
    ) -> Result<Option<KyokuOutcome3p>, String> {
        match action {
            ReactionAction::Pon { consumed } => {
                Board3p::apply_pon(self, seat, *consumed).map(|()| None)
            }
            ReactionAction::Daiminkan => Board3p::apply_daiminkan(self, seat),
            ReactionAction::Chi { .. } => Err("3p does not allow chi".to_string()),
            ReactionAction::Ron | ReactionAction::Pass => Ok(None),
        }
    }
    fn advance_turn(&mut self) {
        Board3p::advance_turn(self)
    }

    fn settle(
        &self,
        state: &RoundState,
        length: MatchLength,
        outcome: &KyokuOutcome3p,
    ) -> Settlement {
        match outcome {
            KyokuOutcome3p::Hora {
                winner,
                from,
                score,
            } => progress::settle_hora_with_profile(
                state,
                length,
                hora_payment_with_pao(
                    *winner,
                    *from,
                    score,
                    self.pao_liability(*winner),
                    Self::SEATS as usize,
                ),
                self.rule_profile,
            ),
            KyokuOutcome3p::MultiHora { first, additional } => {
                let first_payment = hora_payment_with_pao(
                    first.winner,
                    first.from,
                    &first.score,
                    self.pao_liability(first.winner),
                    Self::SEATS as usize,
                );
                let additional_payments: Vec<_> = additional
                    .iter()
                    .map(|win| {
                        hora_payment_with_pao(
                            win.winner,
                            win.from,
                            &win.score,
                            self.pao_liability(win.winner),
                            Self::SEATS as usize,
                        )
                    })
                    .collect();
                progress::settle_multi_hora_with_profile(
                    state,
                    length,
                    first_payment,
                    &additional_payments,
                    self.rule_profile,
                )
            }
            KyokuOutcome3p::Ryukyoku { tenpai } => {
                progress::settle_ryukyoku_with_profile(state, length, tenpai, self.rule_profile)
            }
            KyokuOutcome3p::NagashiMangan { winners, tenpai } => {
                progress::settle_nagashi_mangan_with_profile(
                    state,
                    length,
                    winners,
                    tenpai,
                    self.rule_profile,
                )
            }
            KyokuOutcome3p::AbortiveRyukyoku { .. } => {
                progress::settle_abortive_ryukyoku_with_profile(state, length, self.rule_profile)
            }
        }
    }
    fn summarize(outcome: &KyokuOutcome3p) -> String {
        summarize_3p(outcome)
    }
}

/// Internal flow signal of drive and submit.
enum DriveSignal {
    /// The state advanced; keep driving.
    Continue,
    /// A pending window is set; wait for an external action.
    Paused,
    /// The hand ended (outcome set).
    KyokuEnded,
    /// Host failure (failure set).
    Failed,
}

/// Response window state while waiting for manual seats: responses already computed
/// for automatic seats plus the manual responses received so far.
struct ReactionState {
    discarder: u8,
    tile: Tile,
    /// `true` for a kan or nukidora robbing window (Ron/Pass only), `false` for a regular discard.
    robbery: bool,
    /// Automatic seats that declared ron.
    auto_ron: Vec<u8>,
    /// Best call from an automatic seat `(priority, seat, action)`: 2 = pon/open kan, 1 = chi.
    auto_call: Option<(u8, u8, ReactionAction)>,
    /// Manual seats that have not submitted.
    pending_seats: Vec<u8>,
    /// Manual responses received `(seat, action)`.
    manual_responses: Vec<(u8, ReactionAction)>,
}

/// Product-level live match session (one core for 4-player and 3-player). See the [module docs](self).
///
/// The aliases [`LiveMatchSession4p`] / [`LiveMatchSession3p`] give consumers stable names.
pub struct LiveMatchSession<B: LiveBoard> {
    // Match level (constant or carried over across hands).
    config: LiveMatchConfig,
    players: u8,
    start_score: i32,
    length: MatchLength,
    base_seed: (u64, u64),
    kyoku_serial: u64,
    state: RoundState,
    /// Decision agent per seat. Manual seats hold a `ManualSeat` for status only and are never asked for decisions.
    agents: Vec<Box<dyn SeatAgent<B::V>>>,
    /// Whether each seat is manual (`true` stops at a pending window instead of advancing).
    manual: Vec<bool>,

    // Hand level.
    board: B,
    ctx: KyokuContext,
    seed: (u64, u64),
    kyoku_start_scores: Vec<i32>,
    traces: Vec<DecisionTrace>,
    /// L3 decision windows emitted in this hand (always empty when `record_decision_windows` is off).
    windows: Vec<HostWindow>,
    /// Failures to build a window. Unreachable in normal play (the applied action is
    /// always in the authoritative legal set), but the recorder must not interrupt the
    /// match because of its own bugs, so failures become inspectable fault entries.
    window_faults: Vec<WindowFault>,
    /// Window number, monotonic within a hand (failures take a number too, leaving a visible gap).
    window_serial: u64,
    turn_index: u32,
    steps: u64,
    outcome: Option<B::Outcome>,
    outcome_summary: Option<String>,
    settlement: Option<Settlement>,
    pending: Option<PendingWindow>,
    reaction_state: Option<ReactionState>,
    decision_seq: u64,
    /// Host failure state. Once set, the session is frozen and only returns this structure.
    failure: Option<MatchFailure>,

    // Match accumulation.
    kyokus: Vec<crate::match_host::KyokuRecord>,
    match_ended: bool,
}

/// 4-player live session.
pub type LiveMatchSession4p = LiveMatchSession<Board4p>;
/// 3-player live session.
pub type LiveMatchSession3p = LiveMatchSession<Board3p>;

impl LiveMatchSession<Board4p> {
    /// Structured result and settlement of the hand that just ended.
    ///
    /// Only available until the next hand starts. Products should consume it on
    /// `KyokuEnded` rather than parse rule results from the diagnostic strings in
    /// [`FullMatchReport`](crate::FullMatchReport).
    pub fn last_settled_round(&self) -> Option<(&KyokuOutcome, &Settlement)> {
        self.outcome.as_ref().zip(self.settlement.as_ref())
    }

    /// Starts a 4-player live match. `seats` must have length 4. Nothing advances yet;
    /// call [`Self::drive_until_wait_or_end`] to reach the first stopping point.
    pub fn new_4p(config: LiveMatchConfig, seats: Vec<LiveSeat>) -> Result<Self> {
        if seats.len() != 4 {
            bail!("4p live match needs exactly 4 seat bindings");
        }
        let start_score = config
            .start_score
            .unwrap_or_else(|| config.rule_profile.start_score(4));
        let mut agents: Vec<Box<dyn SeatAgent<Variant4p>>> = Vec::with_capacity(4);
        let mut manual = Vec::with_capacity(4);
        for (seat, spec) in seats.into_iter().enumerate() {
            let (agent, is_manual) =
                build_live_agent_4p(seat as u8, spec, config.seed, config.kind)?;
            agents.push(agent);
            manual.push(is_manual);
        }
        Ok(Self::from_parts(config, 4, start_score, agents, manual))
    }

    /// Starts a 4-player live match whose first hand uses the caller's ordered wall.
    ///
    /// For dev replay and scripted wall acceptance in products. Later hands derive their
    /// walls from the same seed lifecycle; the caller does not rebuild the host loop.
    pub fn new_4p_with_ordered_wall(
        config: LiveMatchConfig,
        seats: Vec<LiveSeat>,
        tiles: Vec<Tile>,
    ) -> Result<Self> {
        if tiles.len() != tileset_4p().len() {
            bail!(
                "4p scripted wall must contain {} tiles, got {}",
                tileset_4p().len(),
                tiles.len()
            );
        }
        if seats.len() != 4 {
            bail!("4p live match needs exactly 4 seat bindings");
        }
        let start_score = config
            .start_score
            .unwrap_or_else(|| config.rule_profile.start_score(4));
        let mut agents: Vec<Box<dyn SeatAgent<Variant4p>>> = Vec::with_capacity(4);
        let mut manual = Vec::with_capacity(4);
        for (seat, spec) in seats.into_iter().enumerate() {
            let (agent, is_manual) =
                build_live_agent_4p(seat as u8, spec, config.seed, config.kind)?;
            agents.push(agent);
            manual.push(is_manual);
        }
        let state = RoundState::new_match(4, start_score);
        let scores: [i32; 4] = std::array::from_fn(|i| state.scores[i]);
        let wall = Wall::from_ordered_for_profile(tiles, 4, config.rule_profile)
            .map_err(anyhow::Error::msg)?;
        let board = Board4p::from_wall_with_state(
            wall,
            scores,
            state.bakaze,
            state.kyoku,
            state.oya,
            state.honba,
            state.kyotaku,
        )
        .with_rule_profile(config.rule_profile)
        .map_err(anyhow::Error::msg)?;
        Ok(Self::from_parts_with_initial_board(
            config,
            4,
            start_score,
            agents,
            manual,
            state,
            board,
        ))
    }

    /// Starts a fully automatic headless 4-player live match with boxed agents supplied
    /// by the caller (`agents` must have length 4).
    ///
    /// The constructor through which the public headless API
    /// ([`crate::match_host::run_full_match_4p_with_agents`]) reuses the live core. Every
    /// seat is non-manual, so [`Self::run_to_match_end`] runs to the end without manual
    /// pending. Manual seats that must stop at external decision points use
    /// [`LiveSeat::Manual`] through [`Self::new_4p`]. Plugin `session_id`s are decided by
    /// the caller when building agents (headless keeps `run-match-*` through
    /// [`crate::match_host::SessionIdPolicy::LegacyHeadless`]).
    pub fn new_4p_with_agents(
        config: LiveMatchConfig,
        agents: Vec<Box<dyn SeatAgent<Variant4p>>>,
    ) -> Result<Self> {
        if agents.len() != 4 {
            bail!("4p live match needs exactly 4 seat agents");
        }
        let start_score = config
            .start_score
            .unwrap_or_else(|| config.rule_profile.start_score(4));
        let manual = vec![false; 4];
        Ok(Self::from_parts(config, 4, start_score, agents, manual))
    }
}

impl LiveMatchSession<Board3p> {
    /// Structured result and settlement of the hand that just ended; see
    /// [`LiveMatchSession::<Board4p>::last_settled_round`].
    pub fn last_settled_round(&self) -> Option<(&KyokuOutcome3p, &Settlement)> {
        self.outcome.as_ref().zip(self.settlement.as_ref())
    }

    /// Starts a 3-player live match. `seats` must have length 3.
    pub fn new_3p(config: LiveMatchConfig, seats: Vec<LiveSeat>) -> Result<Self> {
        if seats.len() != 3 {
            bail!("3p live match needs exactly 3 seat bindings");
        }
        let start_score = config
            .start_score
            .unwrap_or_else(|| config.rule_profile.start_score(3));
        let mut agents: Vec<Box<dyn SeatAgent<Variant3p>>> = Vec::with_capacity(3);
        let mut manual = Vec::with_capacity(3);
        for (seat, spec) in seats.into_iter().enumerate() {
            let (agent, is_manual) =
                build_live_agent_3p(seat as u8, spec, config.seed, config.kind)?;
            agents.push(agent);
            manual.push(is_manual);
        }
        Ok(Self::from_parts(config, 3, start_score, agents, manual))
    }

    /// Starts a 3-player live match whose first hand uses the caller's ordered wall.
    pub fn new_3p_with_ordered_wall(
        config: LiveMatchConfig,
        seats: Vec<LiveSeat>,
        tiles: Vec<Tile>,
    ) -> Result<Self> {
        if tiles.len() != tileset_3p().len() {
            bail!(
                "3p scripted wall must contain {} tiles, got {}",
                tileset_3p().len(),
                tiles.len()
            );
        }
        if seats.len() != 3 {
            bail!("3p live match needs exactly 3 seat bindings");
        }
        let start_score = config
            .start_score
            .unwrap_or_else(|| config.rule_profile.start_score(3));
        let mut agents: Vec<Box<dyn SeatAgent<Variant3p>>> = Vec::with_capacity(3);
        let mut manual = Vec::with_capacity(3);
        for (seat, spec) in seats.into_iter().enumerate() {
            let (agent, is_manual) =
                build_live_agent_3p(seat as u8, spec, config.seed, config.kind)?;
            agents.push(agent);
            manual.push(is_manual);
        }
        let state = RoundState::new_match(3, start_score);
        let scores: [i32; 3] = std::array::from_fn(|i| state.scores[i]);
        let wall = Wall::from_ordered_for_profile(tiles, 3, config.rule_profile)
            .map_err(anyhow::Error::msg)?;
        let board = Board3p::from_wall_with_state(
            wall,
            scores,
            state.bakaze,
            state.kyoku,
            state.oya,
            state.honba,
            state.kyotaku,
        )
        .with_rule_profile(config.rule_profile)
        .map_err(anyhow::Error::msg)?;
        Ok(Self::from_parts_with_initial_board(
            config,
            3,
            start_score,
            agents,
            manual,
            state,
            board,
        ))
    }

    /// Starts a fully automatic headless 3-player live match with boxed agents (`agents`
    /// must have length 3). Same as [`Self::new_4p_with_agents`].
    pub fn new_3p_with_agents(
        config: LiveMatchConfig,
        agents: Vec<Box<dyn SeatAgent<Variant3p>>>,
    ) -> Result<Self> {
        if agents.len() != 3 {
            bail!("3p live match needs exactly 3 seat agents");
        }
        let start_score = config
            .start_score
            .unwrap_or_else(|| config.rule_profile.start_score(3));
        let manual = vec![false; 3];
        Ok(Self::from_parts(config, 3, start_score, agents, manual))
    }
}

impl<B: LiveBoard> LiveMatchSession<B> {
    fn from_parts(
        config: LiveMatchConfig,
        players: u8,
        start_score: i32,
        agents: Vec<Box<dyn SeatAgent<B::V>>>,
        manual: Vec<bool>,
    ) -> Self {
        let state = RoundState::new_match(players as usize, start_score);
        let base_seed = (config.seed, KYOKU_SEED_LO);
        let seed = kyoku_seed(base_seed, 0);
        let board = B::start_kyoku(seed, &state, config.rule_profile);
        Self::from_parts_with_initial_board(
            config,
            players,
            start_score,
            agents,
            manual,
            state,
            board,
        )
    }

    fn from_parts_with_initial_board(
        config: LiveMatchConfig,
        players: u8,
        start_score: i32,
        agents: Vec<Box<dyn SeatAgent<B::V>>>,
        manual: Vec<bool>,
        state: RoundState,
        board: B,
    ) -> Self {
        let length = config.kind.progress_length();
        let base_seed = (config.seed, KYOKU_SEED_LO);
        let seed = kyoku_seed(base_seed, 0);
        let ctx = KyokuContext {
            kyoku_index: 0,
            bakaze: state.bakaze,
            kyoku: state.kyoku,
            honba: state.honba,
            kyotaku: state.kyotaku,
        };
        let kyoku_start_scores = state.scores.clone();
        Self {
            config,
            players,
            start_score,
            length,
            base_seed,
            kyoku_serial: 0,
            state,
            agents,
            manual,
            board,
            ctx,
            seed,
            kyoku_start_scores,
            traces: Vec::new(),
            windows: Vec::new(),
            window_faults: Vec::new(),
            window_serial: 0,
            turn_index: 0,
            steps: 0,
            outcome: None,
            outcome_summary: None,
            settlement: None,
            pending: None,
            reaction_state: None,
            decision_seq: 0,
            failure: None,
            kyokus: Vec::new(),
            match_ended: false,
        }
    }

    pub fn variant(&self) -> &'static str {
        B::VARIANT
    }

    pub fn players(&self) -> u8 {
        self.players
    }

    /// The match's rule profile, authoritative for replay. Archiving inlines it into
    /// `start_match`, and replays use it rather than whatever configuration was in effect.
    pub fn rule_profile(&self) -> RiichiRuleProfile {
        self.config.rule_profile
    }

    /// Whether the current hand has ended.
    pub fn is_kyoku_terminal(&self) -> bool {
        self.outcome.is_some()
    }

    /// Whether the match has ended (last hand settled).
    pub fn is_match_ended(&self) -> bool {
        self.match_ended
    }

    /// Whether the session is frozen in a host failure.
    pub fn is_failed(&self) -> bool {
        self.failure.is_some()
    }

    /// Stable host failure structure; `None` for a normal session.
    pub fn failure(&self) -> Option<&MatchFailure> {
        self.failure.as_ref()
    }

    /// Current pending window (`None` while advancing automatically or after the hand ended).
    pub fn pending_window(&self) -> Option<&PendingWindow> {
        self.pending.as_ref()
    }

    /// Legal action summary of the current pending window (`None` without one).
    pub fn legal_actions_for_pending(&self) -> Option<PendingSummary> {
        self.pending.as_ref().map(pending_summary)
    }

    /// Per-move traces of the current hand (cleared on the next hand; see [`Self::report`] for the match).
    pub fn trace(&self) -> &[DecisionTrace] {
        &self.traces
    }

    /// L3 decision windows of the current hand (requires `record_decision_windows`).
    ///
    /// Like [`Self::last_settled_round`], cleared on the next hand; products should take
    /// them on `KyokuEnded` and merge them into their match log. Windows are deliberately
    /// not in [`FullMatchReport`](crate::FullMatchReport): that is a diagnostic trace
    /// versioned by `TRACE_SCHEMA_VERSION`, while windows belong to the archived match
    /// log with its own schema. Keeping the version axes separate means adding a review
    /// field never invalidates historical reports.
    pub fn decision_windows(&self) -> &[HostWindow] {
        &self.windows
    }

    /// Faults while recording L3 windows (always empty in normal play).
    ///
    /// The recorder does not interrupt the match for its own bugs, but it does not
    /// swallow them either. Faults are structured (window number, seat, phase, table
    /// anchor), since a gap in the numbering is not enough: a failure on the last window
    /// leaves no visible gap. A match log builder that sees faults must mark the hand's
    /// L3 as incomplete rather than produce training or statistics data that only looks
    /// complete.
    pub fn decision_window_faults(&self) -> &[WindowFault] {
        &self.window_faults
    }

    /// Status summary of every seat.
    pub fn agent_statuses(&self) -> Vec<SeatAgentStatus> {
        self.agents.iter().map(|a| a.status()).collect()
    }

    /// Authoritative event log of the current hand (without the `StartGame` synthesized for the wire).
    ///
    /// A host-side interface: it contains other players' hands and draws. Products need
    /// it to assemble archived match logs, since L3 window anchors count in this stream.
    /// Never pass it to a seat agent; agents only get [`Self::visible_events_for_seat`].
    pub fn authoritative_events(&self) -> &[<B::V as Variant>::Event] {
        self.board.log_slice()
    }

    /// The table itself (host side, with all hidden state). The archive builder reads the
    /// winner's hand, melds and ura dora from its final state, which the event stream
    /// does not have.
    pub(crate) fn board(&self) -> &B {
        &self.board
    }

    /// Match state of this hand (so the settlement split reuses the same settle closure).
    pub(crate) fn round_state(&self) -> &RoundState {
        &self.state
    }

    /// Hanchan / tonpuusen (needed by settlement).
    pub(crate) fn match_length(&self) -> MatchLength {
        self.length
    }

    /// Projected visible events of a seat since the start of the hand.
    pub fn visible_events_for_seat(&self, seat: u8) -> Vec<VisibleEventOf<B>> {
        self.board.visible_events_for(seat)
    }

    /// Advances until a manual seat's decision point (turn or reaction) or the end of the
    /// hand or match. Algorithm and plugin seats advance automatically.
    pub fn drive_until_wait_or_end(&mut self) -> LiveStepResult {
        loop {
            if self.failure.is_some() || self.outcome.is_some() || self.pending.is_some() {
                break;
            }
            self.steps += 1;
            if self.steps > MAX_STEPS {
                self.fail(
                    "HOST_STEP_LIMIT_EXCEEDED",
                    "drive_guard",
                    None,
                    None,
                    "deterministic step safety limit exceeded",
                );
                break;
            }
            if self.board.log_len() > MAX_LOG_EVENTS {
                self.fail(
                    "HOST_LOG_LIMIT_EXCEEDED",
                    "drive_guard",
                    None,
                    None,
                    "deterministic event-log safety limit exceeded",
                );
                break;
            }

            // If the current seat holds 3n+2 tiles and no discard awaits responses, decide
            // directly (after a call, kan or tsumogiri); otherwise draw first. An empty live wall
            // means an exhaustive draw.
            if !self.current_actor_needs_turn_decision() {
                let wall_before = self.board.wall_remaining();
                if self.board.draw_for_turn().is_none() {
                    if wall_before == 0 {
                        let out = self.board.ryukyoku();
                        self.finish_round(out);
                        continue;
                    }
                    self.fail(
                        "RULE_DRAW_APPLY_REJECTED",
                        "draw",
                        Some(self.board.turn()),
                        None,
                        "authoritative draw failed while live wall was non-empty",
                    );
                    break;
                }
            }

            let seat = self.board.turn();
            if self.manual[seat as usize] {
                match self.set_turn_window(seat) {
                    DriveSignal::Paused => break,
                    DriveSignal::Failed => break,
                    DriveSignal::Continue | DriveSignal::KyokuEnded => continue,
                }
            }
            match self.auto_turn(seat) {
                DriveSignal::Continue | DriveSignal::KyokuEnded => continue,
                DriveSignal::Paused | DriveSignal::Failed => break,
            }
        }
        self.stable_result()
    }

    /// Continues after a submission or a new hand (same as [`Self::drive_until_wait_or_end`];
    /// safe in any state: returns as is when pending or ended).
    pub fn continue_after_submit(&mut self) -> LiveStepResult {
        self.drive_until_wait_or_end()
    }

    /// Advances to the next hand, applying the settled match state to a new wall in place.
    /// `Ok(true)` means a new hand started; `Ok(false)` means the match is over.
    pub fn start_next_kyoku(&mut self) -> Result<bool> {
        if let Some(failure) = &self.failure {
            bail!("{failure}");
        }
        if self.match_ended {
            return Ok(false);
        }
        let Some(settlement) = self.settlement.take() else {
            bail!("current kyoku not settled yet; cannot advance");
        };
        if settlement.ended {
            self.match_ended = true;
            return Ok(false);
        }
        let next = settlement.next;
        self.kyoku_serial = self.kyoku_serial.wrapping_add(1);
        let seed = kyoku_seed(self.base_seed, self.kyoku_serial);
        self.board = B::start_kyoku(seed, &next, self.config.rule_profile);
        self.state = next;
        self.seed = seed;
        self.kyoku_start_scores = self.state.scores.clone();
        self.ctx = KyokuContext {
            kyoku_index: self.kyokus.len() as u32,
            bakaze: self.state.bakaze,
            kyoku: self.state.kyoku,
            honba: self.state.honba,
            kyotaku: self.state.kyotaku,
        };
        self.traces.clear();
        // Window numbers are monotonic within a hand, unlike turn_index, so they reset per hand.
        self.windows.clear();
        self.window_faults.clear();
        self.window_serial = 0;
        // turn_index is not reset: it is monotonic across hands (see `DecisionTrace::turn_index`), as in headless `run_full_match_*`.
        self.steps = 0;
        self.outcome = None;
        self.outcome_summary = None;
        self.settlement = None;
        self.pending = None;
        self.reaction_state = None;
        Ok(true)
    }

    /// For fully automatic seats only: runs to the end of the match (errors on a manual
    /// pending window). The driver headless `run_full_match_*` uses on the live core.
    pub fn run_to_match_end(&mut self) -> Result<()> {
        loop {
            match self.drive_until_wait_or_end() {
                LiveStepResult::MatchEnded => break,
                LiveStepResult::Failed { failure } => bail!("{failure}"),
                LiveStepResult::KyokuEnded => {
                    if !self.start_next_kyoku()? {
                        break;
                    }
                }
                LiveStepResult::PendingTurn { seat } => {
                    bail!("run_to_match_end requires all-auto seats; manual turn pending at seat {seat}")
                }
                LiveStepResult::PendingReaction { seats } => {
                    bail!("run_to_match_end requires all-auto seats; manual reaction pending at {seats:?}")
                }
                LiveStepResult::InProgress => {}
            }
        }
        Ok(())
    }

    /// Match report (stable schema for products and replay, same structure as `match_host`).
    pub fn report(&self) -> FullMatchReport {
        let current_attempt = if !self.match_ended && self.outcome.is_none() {
            Some(CurrentKyokuAttempt {
                kyoku_index: self.ctx.kyoku_index,
                bakaze: self.ctx.bakaze,
                kyoku: self.ctx.kyoku,
                honba: self.ctx.honba,
                kyotaku: self.ctx.kyotaku,
                oya: self.board.oya(),
                seed_hi: self.seed.0,
                seed_lo: self.seed.1,
                start_scores: self.kyoku_start_scores.clone(),
                turns: self
                    .traces
                    .iter()
                    .filter(|trace| trace.phase == "turn")
                    .count() as u32,
                events_total: self.board.log_len(),
                traces: self.traces.clone(),
            })
        } else {
            None
        };
        let current_scores = if self.outcome.is_some() {
            self.kyokus
                .last()
                .map(|kyoku| kyoku.scores_after.clone())
                .unwrap_or_else(|| self.board.scores())
        } else {
            self.board.scores()
        };
        finish_report(
            B::VARIANT,
            self.players,
            self.config.as_match_config(),
            self.start_score,
            self.kyokus.clone(),
            current_attempt,
            self.match_ended,
            Some(current_scores),
            self.failure.clone(),
            self.agent_statuses(),
        )
    }

    /// Compatibility entry point that submits with the current window's id and digest.
    /// Products and network boundaries should use [`Self::submit_action_bound`] and send
    /// the bound values back so stale responses are rejected.
    pub fn submit_action(&mut self, seat: u8, action: TurnAction) -> Result<LiveStepResult> {
        if self.failure.is_some() {
            return Ok(self.stable_result());
        }
        let (decision_id, state_digest) = self
            .pending
            .as_ref()
            .map(|window| (window.decision_id.clone(), window.state_digest.clone()))
            .ok_or_else(|| anyhow!("no pending decision window"))?;
        self.submit_action_bound(&decision_id, &state_digest, seat, action)
    }

    /// Submits a manual turn action bound to a decision id and state digest.
    pub fn submit_action_bound(
        &mut self,
        decision_id: &str,
        state_digest: &str,
        seat: u8,
        action: TurnAction,
    ) -> Result<LiveStepResult> {
        if self.failure.is_some() {
            return Ok(self.stable_result());
        }
        let window = self
            .pending
            .as_ref()
            .ok_or_else(|| anyhow!("no pending decision window"))?;
        if window.decision_id != decision_id {
            bail!("STALE_DECISION: decision_id does not match current window");
        }
        if window.state_digest != state_digest {
            bail!("STATE_DIGEST_MISMATCH: state_digest does not match current window");
        }
        if window.kind != PendingKind::Turn {
            bail!("pending window is a reaction window, not a turn");
        }
        if window.actor != seat {
            bail!("turn pending is for seat {}, not {seat}", window.actor);
        }
        if !window.turn_actions.contains(&action) {
            bail!("submitted action is not in the legal turn set");
        }
        let legal_actions: Vec<String> = window.turn_actions.iter().map(describe_turn).collect();
        // The window is about to be cleared; keep the legal set (not cloned when recording is off).
        let window_turn_actions = if self.config.record_decision_windows {
            window.turn_actions.clone()
        } else {
            Vec::new()
        };
        let trace_decision_id = window.decision_id.clone();
        let trace_state_digest = window.state_digest.clone();
        let kind = self.agents[seat as usize].kind().as_str();
        let model_id = self.agents[seat as usize].model_id();
        self.pending = None;

        if self.config.record_decision_windows {
            let view = self.board.view_for(seat);
            self.record_turn_window(seat, &view, &window_turn_actions, &action);
        }

        let before = self.board.last_discard();
        let applied = self.board.apply_turn(action.clone());
        // Manual seats get no fake fallback: an illegal action returns `Err` (the window is cleared; the caller retries).
        let signal = match applied {
            Ok(maybe_outcome) => {
                self.push_trace(
                    trace_decision_id,
                    trace_state_digest,
                    "turn",
                    seat,
                    legal_actions,
                    describe_turn(&action),
                    describe_turn(&action),
                    kind,
                    model_id,
                    0,
                    None,
                    true,
                    None,
                );
                self.turn_index += 1;
                match maybe_outcome {
                    Some(out) => {
                        self.finish_round(out);
                        DriveSignal::KyokuEnded
                    }
                    None => self.after_turn_applied(before),
                }
            }
            Err(e) => {
                // The action was enumerated as legal from the same authoritative state, so a
                // rejection by apply is an inconsistency in the rules core. It is not the caller's
                // fault, and the window is not reopened or a fallback faked.
                let _ = e;
                self.push_trace(
                    trace_decision_id.clone(),
                    trace_state_digest,
                    "turn",
                    seat,
                    legal_actions,
                    describe_turn(&action),
                    "not_applied".to_string(),
                    kind,
                    model_id,
                    0,
                    Some("authoritative apply rejected an enumerated legal action".to_string()),
                    true,
                    Some("RULE_LEGAL_ACTION_APPLY_REJECTED"),
                );
                self.fail(
                    "RULE_LEGAL_ACTION_APPLY_REJECTED",
                    "turn_apply",
                    Some(seat),
                    Some(trace_decision_id),
                    "authoritative apply rejected an enumerated legal action",
                );
                DriveSignal::Failed
            }
        };
        Ok(self.signal_to_result(signal))
    }

    /// Compatibility entry point; products should use [`Self::submit_reaction_bound`].
    pub fn submit_reaction(&mut self, seat: u8, action: ReactionAction) -> Result<LiveStepResult> {
        if self.failure.is_some() {
            return Ok(self.stable_result());
        }
        let (decision_id, state_digest) = self
            .pending
            .as_ref()
            .map(|window| (window.decision_id.clone(), window.state_digest.clone()))
            .ok_or_else(|| anyhow!("no pending decision window"))?;
        self.submit_reaction_bound(&decision_id, &state_digest, seat, action)
    }

    /// Submits a manual response bound to the current window; once all are in, resolves
    /// by profile and call priority.
    pub fn submit_reaction_bound(
        &mut self,
        decision_id: &str,
        state_digest: &str,
        seat: u8,
        action: ReactionAction,
    ) -> Result<LiveStepResult> {
        if self.failure.is_some() {
            return Ok(self.stable_result());
        }
        let window = self
            .pending
            .as_ref()
            .ok_or_else(|| anyhow!("no pending decision window"))?;
        if window.decision_id != decision_id {
            bail!("STALE_DECISION: decision_id does not match current window");
        }
        if window.state_digest != state_digest {
            bail!("STATE_DIGEST_MISMATCH: state_digest does not match current window");
        }
        if !matches!(window.kind, PendingKind::Reaction | PendingKind::Robbery) {
            bail!("pending window is a turn window, not a reaction");
        }
        let legal = window
            .reaction_seats
            .iter()
            .find(|r| r.seat == seat)
            .ok_or_else(|| anyhow!("seat {seat} has no pending reaction in this window"))?
            .legal
            .clone();
        let trace_decision_id = window.decision_id.clone();
        let trace_state_digest = window.state_digest.clone();
        // Pass is always legal; anything else must be in the seat's legal responses.
        if !reaction_in_legal(&action, &legal) {
            bail!("submitted reaction is not legal for seat {seat}");
        }
        // Record before note_missed_ron, which changes the seat's furiten; the window needs the state at opening.
        if self.config.record_decision_windows {
            if let Some((discarder, tile, robbery)) = self
                .reaction_state
                .as_ref()
                .map(|rs| (rs.discarder, rs.tile, rs.robbery))
            {
                let view = self.board.view_for_reaction(seat);
                self.record_reaction_window(seat, &view, discarder, tile, &legal, &action, robbery);
            }
        }
        if legal.contains(&ReactionAction::Ron)
            && !matches!(action, ReactionAction::Ron)
            && self.board.note_missed_ron(seat).is_err()
        {
            self.fail(
                "RULE_MISSED_RON_APPLY_REJECTED",
                "reaction_furiten",
                Some(seat),
                Some(trace_decision_id),
                "authoritative missed-ron state update failed",
            );
            return Ok(self.stable_result());
        }
        if self.reaction_state.is_none() {
            self.fail(
                "HOST_REACTION_STATE_MISSING",
                "reaction_submit",
                Some(seat),
                Some(trace_decision_id),
                "pending window had no matching reaction accumulator",
            );
            return Ok(self.stable_result());
        }

        let kind = self.agents[seat as usize].kind().as_str();
        let model_id = self.agents[seat as usize].model_id();
        self.push_trace(
            trace_decision_id,
            trace_state_digest,
            "reaction",
            seat,
            reaction_legal_descriptions(&legal),
            describe_reaction(&action),
            describe_reaction(&action),
            kind,
            model_id,
            0,
            None,
            true,
            None,
        );

        // Record the response and remove the seat from the pending set.
        {
            let rs = self
                .reaction_state
                .as_mut()
                .expect("reaction state was checked immediately above");
            rs.manual_responses.push((seat, action));
            rs.pending_seats.retain(|s| *s != seat);
        }
        if let Some(w) = self.pending.as_mut() {
            w.reaction_seats.retain(|r| r.seat != seat);
        }

        let all_in = self
            .reaction_state
            .as_ref()
            .map(|rs| rs.pending_seats.is_empty())
            .unwrap_or(true);
        if all_in {
            let signal = self.resolve_reaction_window();
            Ok(self.signal_to_result(signal))
        } else {
            Ok(self.stable_result())
        }
    }

    fn current_actor_needs_turn_decision(&self) -> bool {
        self.board.last_discard().is_none()
            && self.board.pending_robbery().is_none()
            && self.board.hand_len(self.board.turn()) % 3 == 2
    }

    fn set_turn_window(&mut self, seat: u8) -> DriveSignal {
        let view = self.board.view_for(seat);
        view.assert_no_hidden_truth();
        let actions = self.board.legal_turn_actions(seat);
        if actions.is_empty() {
            self.fail(
                "RULE_TURN_LEGAL_SET_EMPTY",
                "turn_legal",
                Some(seat),
                None,
                "authoritative turn legal-action set was empty",
            );
            return DriveSignal::Failed;
        }
        let decision_id = self.next_decision_id();
        let state_digest =
            self.decision_state_digest("turn", seat, &format!("{view:?}|{actions:?}"));
        self.pending = Some(PendingWindow {
            decision_id,
            state_digest,
            kind: PendingKind::Turn,
            actor: seat,
            discarder: None,
            tile: None,
            turn_actions: actions,
            reaction_seats: Vec::new(),
        });
        DriveSignal::Paused
    }

    /// Automatic turn: decide, adjudicate (with the match_host fallback and trace
    /// semantics for illegal actions and apply errors), then poll responses.
    fn auto_turn(&mut self, seat: u8) -> DriveSignal {
        let view = self.board.view_for(seat);
        view.assert_no_hidden_truth();
        let legal = self.board.legal_turn_actions(seat);
        if legal.is_empty() {
            self.fail(
                "RULE_TURN_LEGAL_SET_EMPTY",
                "turn_legal",
                Some(seat),
                None,
                "authoritative turn legal-action set was empty",
            );
            return DriveSignal::Failed;
        }
        let events = self.board.events_with_start_game();
        let wall = self.board.wall_remaining();
        let decision_id = self.next_decision_id();
        let state_digest = self.decision_state_digest("turn", seat, &format!("{view:?}|{legal:?}"));
        let choice = {
            let req = TurnRequest {
                decision_id: &decision_id,
                state_digest: &state_digest,
                seat,
                view: &view,
                events: &events,
                legal: &legal,
                wall_remaining: wall,
            };
            self.agents[seat as usize].decide_turn(&req)
        };

        let in_legal = choice.error_code.is_none() && legal.contains(&choice.action);
        let submitted_action = choice
            .submitted_action
            .clone()
            .unwrap_or_else(|| describe_turn(&choice.action));
        let applied_action = if in_legal {
            choice.action.clone()
        } else {
            crate::match_host::fallback_turn(&legal)
        };
        let mut fallback_note = if in_legal {
            choice.fallback.clone()
        } else {
            Some(
                choice
                    .fallback
                    .clone()
                    .unwrap_or_else(|| "illegal_selection".to_string()),
            )
        };
        let mut error_code = if in_legal {
            choice.error_code
        } else {
            Some("MODEL_ACTION_NOT_LEGAL")
        };

        // Record the window before applying: afterwards `log_len` already includes this
        // move's own events. The recorded action is the one actually applied (the fallback
        // when the model was out of range); the match log wants what really happened, and
        // the model's raw submission stays in the trace.
        self.record_turn_window(seat, &view, &legal, &applied_action);

        let before = self.board.last_discard();
        let applied = self.board.apply_turn(applied_action.clone());
        let apply_failure_code = if applied.is_err() {
            let (code, note) = if in_legal {
                (
                    "RULE_LEGAL_ACTION_APPLY_REJECTED",
                    "authoritative apply rejected an enumerated legal action",
                )
            } else {
                (
                    "RULE_FALLBACK_APPLY_REJECTED",
                    "authoritative apply rejected the deterministic legal fallback",
                )
            };
            fallback_note = Some(note.to_string());
            error_code = Some(code);
            Some((code, note))
        } else {
            None
        };

        let kind = self.agents[seat as usize].kind().as_str();
        let model_id = self.agents[seat as usize].model_id();
        self.push_trace(
            decision_id,
            state_digest,
            "turn",
            seat,
            legal.iter().map(describe_turn).collect(),
            submitted_action,
            if apply_failure_code.is_some() {
                "not_applied".to_string()
            } else {
                describe_turn(&applied_action)
            },
            kind,
            model_id,
            choice.latency_ms,
            fallback_note,
            in_legal,
            error_code,
        );

        if let Some((code, detail)) = apply_failure_code {
            self.fail(
                code,
                "turn_apply",
                Some(seat),
                Some(
                    self.traces
                        .last()
                        .expect("failure trace was just appended")
                        .decision_id
                        .clone(),
                ),
                detail,
            );
            return DriveSignal::Failed;
        }
        self.turn_index += 1;

        match applied {
            Ok(Some(out)) => {
                self.finish_round(out);
                DriveSignal::KyokuEnded
            }
            Ok(None) => self.after_turn_applied(before),
            Err(_) => unreachable!("apply error was converted into a frozen failure above"),
        }
    }

    /// After a turn action: poll responses if there is a new discard; otherwise (kan or nukidora) continue in place.
    fn after_turn_applied(&mut self, before: Option<(u8, Tile)>) -> DriveSignal {
        if let Some(pending) = self.board.pending_robbery() {
            return self.poll_robbery_reactions(pending);
        }
        match self.board.last_discard() {
            Some((discarder, tile)) if self.board.last_discard() != before => {
                self.poll_reactions_after_discard(discarder, tile)
            }
            _ => DriveSignal::Continue,
        }
    }

    /// Ron polling after a kan or nukidora declaration. Only Ron/Pass; the declared tile cannot be called.
    fn poll_robbery_reactions(&mut self, pending: PendingRobbery) -> DriveSignal {
        let mut auto_ron = Vec::new();
        let mut manual_pending = Vec::new();
        // Hand history for this robbing window, built lazily (as in poll_reactions_after_discard).
        let mut window_events: Option<Vec<<B::V as Variant>::Event>> = None;
        for seat in 0..B::SEATS {
            if seat == pending.actor {
                continue;
            }
            let legal = self.board.legal_reactions(seat);
            if legal.is_empty() {
                continue;
            }
            if !legal
                .iter()
                .all(|action| matches!(action, ReactionAction::Ron))
            {
                self.fail(
                    "RULE_ROBBERY_LEGAL_SET_INVALID",
                    "robbery_legal",
                    Some(seat),
                    None,
                    "robbery window exposed a non-ron reaction",
                );
                return DriveSignal::Failed;
            }
            if self.manual[seat as usize] {
                manual_pending.push(PendingReactionSeat { seat, legal });
                continue;
            }
            let view = self.board.view_for_reaction(seat);
            // The board does not change inside a response window, so the history is the same for
            // every seat: build it once and reuse it. Laziness also means windows with only
            // manual or only algorithm seats never build it.
            if window_events.is_none() {
                window_events = Some(self.board.events_with_start_game());
            }
            let events = window_events.as_deref().unwrap_or(&[]);
            let wall = self.board.wall_remaining();
            let decision_id = self.next_decision_id();
            let state_digest = self.decision_state_digest(
                "robbery",
                seat,
                &format!("{view:?}|{legal:?}|{pending:?}"),
            );
            let choice = {
                let req = ReactionRequest {
                    decision_id: &decision_id,
                    state_digest: &state_digest,
                    seat,
                    view: &view,
                    events,
                    legal: &legal,
                    discarder: pending.actor,
                    tile: pending.tile,
                    wall_remaining: wall,
                };
                self.agents[seat as usize].decide_reaction(&req)
            };
            let selected_in_legal =
                choice.error_code.is_none() && reaction_in_legal(&choice.action, &legal);
            let submitted_action = choice
                .submitted_action
                .clone()
                .unwrap_or_else(|| format!("{:?}", choice.action));
            let (selected_action, fallback_note) = if selected_in_legal {
                (choice.action.clone(), choice.fallback.clone())
            } else {
                (
                    ReactionAction::Pass,
                    Some(
                        choice
                            .fallback
                            .clone()
                            .unwrap_or_else(|| "illegal_robbery_selection".to_string()),
                    ),
                )
            };
            let error_code = if selected_in_legal {
                choice.error_code
            } else {
                Some("MODEL_ACTION_NOT_LEGAL")
            };
            self.record_reaction_window(
                seat,
                &view,
                pending.actor,
                pending.tile,
                &legal,
                &selected_action,
                true,
            );
            let kind = self.agents[seat as usize].kind().as_str();
            let model_id = self.agents[seat as usize].model_id();
            let failure_decision_id = decision_id.clone();
            self.push_trace(
                decision_id,
                state_digest,
                "robbery",
                seat,
                reaction_legal_descriptions(&legal),
                submitted_action,
                describe_reaction(&selected_action),
                kind,
                model_id,
                choice.latency_ms,
                fallback_note,
                selected_in_legal,
                error_code,
            );
            if legal.contains(&ReactionAction::Ron)
                && !matches!(selected_action, ReactionAction::Ron)
                && self.board.note_missed_ron(seat).is_err()
            {
                self.fail(
                    "RULE_MISSED_RON_APPLY_REJECTED",
                    "robbery_furiten",
                    Some(seat),
                    Some(failure_decision_id),
                    "authoritative missed-ron state update failed",
                );
                return DriveSignal::Failed;
            }
            if matches!(selected_action, ReactionAction::Ron) {
                auto_ron.push(seat);
            }
        }

        if manual_pending.is_empty() {
            return self.resolve_robbery_reactions(pending.actor, pending.tile, auto_ron);
        }
        let pending_seats = manual_pending.iter().map(|entry| entry.seat).collect();
        let decision_id = self.next_decision_id();
        let state_digest = self.decision_state_digest(
            "robbery",
            pending.actor,
            &format!("{manual_pending:?}|{pending:?}"),
        );
        self.reaction_state = Some(ReactionState {
            discarder: pending.actor,
            tile: pending.tile,
            robbery: true,
            auto_ron,
            auto_call: None,
            pending_seats,
            manual_responses: Vec::new(),
        });
        self.pending = Some(PendingWindow {
            decision_id,
            state_digest,
            kind: PendingKind::Robbery,
            actor: pending.actor,
            discarder: Some(pending.actor),
            tile: Some(pending.tile),
            turn_actions: Vec::new(),
            reaction_seats: manual_pending,
        });
        DriveSignal::Paused
    }

    /// Response polling after a discard: automatic seats are asked immediately, manual
    /// seats become pending. Resolves immediately when no manual seat is pending.
    fn poll_reactions_after_discard(&mut self, discarder: u8, tile: Tile) -> DriveSignal {
        let seats = B::SEATS;
        let mut auto_ron: Vec<u8> = Vec::new();
        let mut auto_call: Option<(u8, u8, ReactionAction)> = None;
        let mut manual_pending: Vec<PendingReactionSeat> = Vec::new();
        // Hand history for this response window; the board does not change, so seats share one copy.
        let mut window_events: Option<Vec<<B::V as Variant>::Event>> = None;

        for s in 0..seats {
            if s == discarder {
                continue;
            }
            let legal = self.board.legal_reactions(s);
            if legal.is_empty() {
                continue;
            }
            if self.manual[s as usize] {
                manual_pending.push(PendingReactionSeat { seat: s, legal });
                continue;
            }
            // Automatic seat: ask the agent now (illegal -> Pass fallback and trace semantics as in match_host).
            let view = self.board.view_for_reaction(s);
            // Build the history once per window (saves up to two full copies in 4P).
            if window_events.is_none() {
                window_events = Some(self.board.events_with_start_game());
            }
            let events = window_events.as_deref().unwrap_or(&[]);
            let wall = self.board.wall_remaining();
            let decision_id = self.next_decision_id();
            let state_digest = self.decision_state_digest(
                "reaction",
                s,
                &format!("{view:?}|{legal:?}|{discarder}|{tile}"),
            );
            let choice = {
                let req = ReactionRequest {
                    decision_id: &decision_id,
                    state_digest: &state_digest,
                    seat: s,
                    view: &view,
                    events,
                    legal: &legal,
                    discarder,
                    tile,
                    wall_remaining: wall,
                };
                self.agents[s as usize].decide_reaction(&req)
            };
            let selected_in_legal =
                choice.error_code.is_none() && reaction_in_legal(&choice.action, &legal);
            let submitted_action = choice
                .submitted_action
                .clone()
                .unwrap_or_else(|| format!("{:?}", choice.action));
            let (selected_action, fallback_note) = if selected_in_legal {
                (choice.action.clone(), choice.fallback.clone())
            } else {
                (
                    ReactionAction::Pass,
                    Some(
                        choice
                            .fallback
                            .clone()
                            .unwrap_or_else(|| "illegal_reaction_selection".to_string()),
                    ),
                )
            };
            let error_code = if selected_in_legal {
                choice.error_code
            } else {
                Some("MODEL_ACTION_NOT_LEGAL")
            };
            self.record_reaction_window(s, &view, discarder, tile, &legal, &selected_action, false);
            let kind = self.agents[s as usize].kind().as_str();
            let model_id = self.agents[s as usize].model_id();
            let failure_decision_id = decision_id.clone();
            self.push_trace(
                decision_id,
                state_digest,
                "reaction",
                s,
                reaction_legal_descriptions(&legal),
                submitted_action,
                describe_reaction(&selected_action),
                kind,
                model_id,
                choice.latency_ms,
                fallback_note,
                selected_in_legal,
                error_code,
            );
            if legal.contains(&ReactionAction::Ron)
                && !matches!(selected_action, ReactionAction::Ron)
                && self.board.note_missed_ron(s).is_err()
            {
                self.fail(
                    "RULE_MISSED_RON_APPLY_REJECTED",
                    "reaction_furiten",
                    Some(s),
                    Some(failure_decision_id),
                    "authoritative missed-ron state update failed",
                );
                return DriveSignal::Failed;
            }
            match &selected_action {
                ReactionAction::Ron => auto_ron.push(s),
                ReactionAction::Pon { .. } | ReactionAction::Daiminkan => {
                    update_call(
                        &mut auto_call,
                        2,
                        s,
                        choice.action.clone(),
                        discarder,
                        seats,
                    );
                }
                ReactionAction::Chi { .. } => {
                    update_call(
                        &mut auto_call,
                        1,
                        s,
                        choice.action.clone(),
                        discarder,
                        seats,
                    );
                }
                ReactionAction::Pass => {}
            }
        }

        if manual_pending.is_empty() {
            return self.resolve_reactions(discarder, tile, auto_ron, auto_call);
        }

        // A manual seat can respond: suspend the window until each submits.
        let pending_seats: Vec<u8> = manual_pending.iter().map(|r| r.seat).collect();
        let decision_id = self.next_decision_id();
        let state_digest = self.decision_state_digest(
            "reaction",
            discarder,
            &format!("{manual_pending:?}|{discarder}|{tile}"),
        );
        self.reaction_state = Some(ReactionState {
            discarder,
            tile,
            robbery: false,
            auto_ron,
            auto_call,
            pending_seats,
            manual_responses: Vec::new(),
        });
        self.pending = Some(PendingWindow {
            decision_id,
            state_digest,
            kind: PendingKind::Reaction,
            actor: discarder,
            discarder: Some(discarder),
            tile: Some(tile),
            turn_actions: Vec::new(),
            reaction_seats: manual_pending,
        });
        DriveSignal::Paused
    }

    /// Once all manual responses are in, merge automatic and manual declarations and
    /// resolve by head bump and pon/kan over chi.
    fn resolve_reaction_window(&mut self) -> DriveSignal {
        let rs = match self.reaction_state.take() {
            Some(rs) => rs,
            None => {
                self.fail(
                    "HOST_REACTION_STATE_MISSING",
                    "reaction_resolve",
                    None,
                    self.latest_decision_id(),
                    "reaction resolution had no accumulator",
                );
                return DriveSignal::Failed;
            }
        };
        let mut ron_candidates = rs.auto_ron;
        let mut call = rs.auto_call;
        for (seat, action) in rs.manual_responses {
            match &action {
                ReactionAction::Ron => ron_candidates.push(seat),
                ReactionAction::Pon { .. } | ReactionAction::Daiminkan => {
                    update_call(&mut call, 2, seat, action, rs.discarder, B::SEATS);
                }
                ReactionAction::Chi { .. } => {
                    update_call(&mut call, 1, seat, action, rs.discarder, B::SEATS);
                }
                ReactionAction::Pass => {}
            }
        }
        self.pending = None;
        if rs.robbery {
            return self.resolve_robbery_reactions(rs.discarder, rs.tile, ron_candidates);
        }
        self.resolve_reactions(rs.discarder, rs.tile, ron_candidates, call)
    }

    /// Resolves a kan or nukidora robbing window: multiple ron as for discards; the
    /// replacement draw only happens when nobody wins.
    fn resolve_robbery_reactions(
        &mut self,
        actor: u8,
        _tile: Tile,
        mut ron_candidates: Vec<u8>,
    ) -> DriveSignal {
        ron_candidates.sort_by_key(|&seat| (seat + B::SEATS - actor) % B::SEATS);
        ron_candidates.dedup();
        if ron_candidates.is_empty() {
            return match self.board.resolve_pending_robbery_passes() {
                Ok(Some(out)) => {
                    self.finish_round(out);
                    DriveSignal::KyokuEnded
                }
                Ok(None) => DriveSignal::Continue,
                Err(_) => {
                    self.fail(
                        "RULE_ROBBERY_PASS_RESOLUTION_FAILED",
                        "robbery_pass_resolve",
                        Some(actor),
                        self.latest_decision_id(),
                        "authoritative robbery-pass resolution failed",
                    );
                    DriveSignal::Failed
                }
            };
        }
        let resolution = self.board.ron_resolution();
        if resolution == RonResolution::TripleRonAbortive
            && B::SEATS == 4
            && ron_candidates.len() == 3
        {
            return match self.board.abort_sanchaho() {
                Ok(out) => {
                    self.finish_round(out);
                    DriveSignal::KyokuEnded
                }
                Err(_) => {
                    self.fail(
                        "RULE_SANCHAHO_APPLY_REJECTED",
                        "robbery_sanchaho",
                        None,
                        self.latest_decision_id(),
                        "authoritative triple-ron abortive draw application failed",
                    );
                    DriveSignal::Failed
                }
            };
        }
        if resolution == RonResolution::AtamaHane {
            let Some(winner) = select_ron_winner(&ron_candidates, actor, B::SEATS) else {
                self.fail(
                    "HOST_RON_RESOLUTION_EMPTY",
                    "robbery_ron_resolve",
                    None,
                    self.latest_decision_id(),
                    "non-empty ron candidates produced no atama-hane winner",
                );
                return DriveSignal::Failed;
            };
            ron_candidates.clear();
            ron_candidates.push(winner);
        }
        match self.board.apply_robbery_rons(&ron_candidates) {
            Ok(out) => {
                self.finish_round(out);
                DriveSignal::KyokuEnded
            }
            Err(_) => {
                self.fail(
                    "RULE_ROBBERY_RON_APPLY_REJECTED",
                    "robbery_ron_apply",
                    ron_candidates.first().copied(),
                    self.latest_decision_id(),
                    "authoritative robbery ron application failed",
                );
                DriveSignal::Failed
            }
        }
    }

    /// Common resolution: head bump / multiple ron / triple ron draw by profile; ron always beats calls.
    fn resolve_reactions(
        &mut self,
        discarder: u8,
        tile: Tile,
        mut ron_candidates: Vec<u8>,
        call: Option<(u8, u8, ReactionAction)>,
    ) -> DriveSignal {
        ron_candidates.sort_by_key(|&seat| (seat + B::SEATS - discarder) % B::SEATS);
        ron_candidates.dedup();
        if !ron_candidates.is_empty() {
            let resolution = self.board.ron_resolution();
            if resolution == RonResolution::TripleRonAbortive
                && B::SEATS == 4
                && ron_candidates.len() == 3
            {
                return match self.board.abort_sanchaho() {
                    Ok(out) => {
                        self.finish_round(out);
                        DriveSignal::KyokuEnded
                    }
                    Err(_) => {
                        self.fail(
                            "RULE_SANCHAHO_APPLY_REJECTED",
                            "sanchaho",
                            None,
                            self.latest_decision_id(),
                            "authoritative triple-ron abortive draw application failed",
                        );
                        DriveSignal::Failed
                    }
                };
            }
            if resolution == RonResolution::AtamaHane {
                let Some(winner) = select_ron_winner(&ron_candidates, discarder, B::SEATS) else {
                    self.fail(
                        "HOST_RON_RESOLUTION_EMPTY",
                        "ron_resolve",
                        None,
                        self.latest_decision_id(),
                        "non-empty ron candidates produced no atama-hane winner",
                    );
                    return DriveSignal::Failed;
                };
                ron_candidates.clear();
                ron_candidates.push(winner);
            }
            return match self.board.apply_rons(&ron_candidates, discarder, tile) {
                Ok(out) => {
                    self.finish_round(out);
                    DriveSignal::KyokuEnded
                }
                Err(_) => {
                    self.fail(
                        "RULE_RON_APPLY_REJECTED",
                        "ron_apply",
                        ron_candidates.first().copied(),
                        self.latest_decision_id(),
                        "authoritative normal ron application failed",
                    );
                    DriveSignal::Failed
                }
            };
        }
        if let Some((_, s, action)) = call {
            return match self.board.apply_call(s, &action) {
                Ok(Some(out)) => {
                    self.finish_round(out);
                    DriveSignal::KyokuEnded
                }
                Ok(None) => {
                    // Call succeeded: the core cleared last_discard and the caller decides next.
                    DriveSignal::Continue
                }
                Err(_) => {
                    // A mismatch between the legal set and the apply layer is an internal error; fail
                    // closed instead of treating the declared call as Pass.
                    self.fail(
                        "RULE_CALL_APPLY_REJECTED",
                        "call_apply",
                        Some(s),
                        self.latest_decision_id(),
                        "authoritative call application failed",
                    );
                    DriveSignal::Failed
                }
            };
        }
        // Nobody responded: clear the discard and move to the next seat (which draws normally).
        if let Some(out) = self.board.clear_last_discard() {
            self.finish_round(out);
            return DriveSignal::KyokuEnded;
        }
        self.board.advance_turn();
        DriveSignal::Continue
    }

    /// Records the hand result: sync the board's two-stage riichi accounting, settle with progress, record the final state.
    fn finish_round(&mut self, outcome: B::Outcome) {
        let declared = self.sync_riichi_state();
        let settlement = self.board.settle(&self.state, self.length, &outcome);
        let summary = B::summarize(&outcome);
        let turns = self.traces.iter().filter(|t| t.phase == "turn").count() as u32;
        self.kyokus.push(crate::match_host::KyokuRecord {
            kyoku_index: self.ctx.kyoku_index,
            bakaze: self.ctx.bakaze,
            kyoku: self.ctx.kyoku,
            honba: self.ctx.honba,
            kyotaku: self.ctx.kyotaku,
            oya: self.state.oya,
            seed_hi: self.seed.0,
            seed_lo: self.seed.1,
            start_scores: self.kyoku_start_scores.clone(),
            riichi_declared: declared,
            outcome: summary.clone(),
            deltas: settlement.deltas.clone(),
            scores_after: settlement.scores.clone(),
            next_honba: settlement.next.honba,
            next_kyotaku: settlement.next.kyotaku,
            next_oya: settlement.next.oya,
            turns,
            events_total: self.board.log_len(),
            traces: self.traces.clone(),
        });
        if self.config.kind == MatchKind::Single || settlement.ended {
            self.match_ended = true;
        }
        self.outcome = Some(outcome);
        self.outcome_summary = Some(summary);
        self.settlement = Some(settlement);
        self.pending = None;
        self.reaction_state = None;
    }

    /// Enters the irreversible host failure state. The first failure wins; later drive/submit calls only see that structure.
    fn fail(
        &mut self,
        code: &str,
        stage: &str,
        seat: Option<u8>,
        decision_id: Option<String>,
        detail: &str,
    ) {
        if self.failure.is_some() {
            return;
        }
        self.failure = Some(MatchFailure {
            code: code.to_string(),
            stage: stage.to_string(),
            seat,
            decision_id,
            detail: detail.to_string(),
        });
        self.pending = None;
        self.reaction_state = None;
    }

    fn latest_decision_id(&self) -> Option<String> {
        self.traces.last().map(|trace| trace.decision_id.clone())
    }

    /// The board takes the riichi stick when the declaration's response window closes; the match layer only syncs its public state.
    fn sync_riichi_state(&mut self) -> Vec<bool> {
        let seats = self.state.scores.len();
        let mut declared = vec![false; seats];
        for actor in self.board.reach_accepted_seats() {
            let s = actor as usize;
            if s < seats && !declared[s] {
                declared[s] = true;
            }
        }
        self.state.scores = self.board.scores();
        self.state.kyotaku = self.board.kyotaku();
        declared
    }

    fn next_decision_id(&mut self) -> String {
        self.decision_seq += 1;
        format!("ft-live-{}-d{}", B::VARIANT, self.decision_seq)
    }

    fn decision_state_digest(&self, phase: &str, seat: u8, payload: &str) -> String {
        let state = format!(
            "flytable-decision-v2|{}|{}|{}|{}|{}|{}|{}|{}|{:?}|{}",
            B::VARIANT,
            self.config.rule_profile.platform(),
            self.ctx.kyoku_index,
            self.ctx.bakaze,
            self.ctx.kyoku,
            self.board.honba(),
            self.board.kyotaku(),
            self.board.log_len(),
            self.board.scores(),
            format_args!("{phase}|{seat}|{}|{payload}", self.board.wall_remaining()),
        );
        let mut hasher = Sha256::new();
        hasher.update(state.as_bytes());
        format!("sha256:{:x}", hasher.finalize())
    }

    /// Window number, monotonic within the hand. Failures take a number too, so gaps in the log point at fault entries.
    fn next_window_id(&mut self) -> u64 {
        let id = self.window_serial;
        self.window_serial += 1;
        id
    }

    /// Records a turn window. No-op when `record_decision_windows` is off.
    ///
    /// The anchor is `log_len()`, the cursor into the authoritative table event log (not
    /// the match log; the streams are not 1:1, see `HostWindow::rebase`), excluding the
    /// `StartGame` that `events_with_start_game` prepends for the inference wire. Must be
    /// called before `apply_*`, or the anchor would point past this move's own events.
    fn record_turn_window(
        &mut self,
        seat: u8,
        view: &SeatView,
        legal: &[TurnAction],
        chosen: &TurnAction,
    ) {
        if !self.config.record_decision_windows {
            return;
        }
        let window_id = self.next_window_id();
        let anchor = self.board.log_len() as u64;
        match turn_window::<B::V>(window_id, anchor, seat, view, legal, chosen) {
            Ok(window) => self.windows.push(window),
            Err(reason) => self.window_faults.push(WindowFault {
                window_id,
                seat,
                phase: WindowPhase::Turn,
                anchor_board_seq: anchor,
                reason,
            }),
        }
    }

    /// Records a response or robbing window. No-op when `record_decision_windows` is off.
    ///
    /// Passes the action itself rather than an index: response candidates are not a
    /// one-to-one mapping of the legal set (`Pass` and 3-player `Chi` are excluded), and
    /// the index conversion is done in one place, `decision_window`.
    #[allow(clippy::too_many_arguments)]
    fn record_reaction_window(
        &mut self,
        seat: u8,
        view: &SeatView,
        discarder: u8,
        tile: Tile,
        legal: &[ReactionAction],
        chosen: &ReactionAction,
        robbery: bool,
    ) {
        if !self.config.record_decision_windows {
            return;
        }
        let window_id = self.next_window_id();
        let anchor = self.board.log_len() as u64;
        match reaction_window::<B::V>(
            window_id, anchor, seat, view, discarder, tile, legal, chosen, robbery,
        ) {
            Ok(window) => self.windows.push(window),
            Err(reason) => self.window_faults.push(WindowFault {
                window_id,
                seat,
                phase: if robbery {
                    WindowPhase::Robbery
                } else {
                    WindowPhase::Reaction
                },
                anchor_board_seq: anchor,
                reason,
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_trace(
        &mut self,
        decision_id: String,
        state_digest: String,
        phase: &'static str,
        seat: u8,
        legal_actions: Vec<String>,
        submitted_action: String,
        final_action: String,
        agent_kind: &'static str,
        model_id: String,
        latency_ms: u64,
        fallback: Option<String>,
        selected_in_legal: bool,
        error_code: Option<&'static str>,
    ) {
        let legal_count = legal_actions.len();
        self.traces.push(decision_trace(
            self.ctx,
            self.turn_index,
            decision_id,
            state_digest,
            seat,
            phase,
            legal_count,
            legal_actions,
            submitted_action,
            final_action,
            agent_kind,
            model_id,
            latency_ms,
            fallback,
            selected_in_legal,
            error_code,
        ));
    }

    fn signal_to_result(&self, signal: DriveSignal) -> LiveStepResult {
        match signal {
            DriveSignal::Paused | DriveSignal::KyokuEnded | DriveSignal::Failed => {
                self.stable_result()
            }
            DriveSignal::Continue => {
                if self.pending.is_some() || self.outcome.is_some() {
                    self.stable_result()
                } else {
                    LiveStepResult::InProgress
                }
            }
        }
    }

    fn stable_result(&self) -> LiveStepResult {
        if let Some(failure) = &self.failure {
            LiveStepResult::Failed {
                failure: failure.clone(),
            }
        } else if let Some(p) = &self.pending {
            match p.kind {
                PendingKind::Turn => LiveStepResult::PendingTurn { seat: p.actor },
                PendingKind::Reaction | PendingKind::Robbery => LiveStepResult::PendingReaction {
                    seats: p.reaction_seats.iter().map(|r| r.seat).collect(),
                },
            }
        } else if self.match_ended {
            LiveStepResult::MatchEnded
        } else if self.outcome.is_some() {
            LiveStepResult::KyokuEnded
        } else {
            LiveStepResult::InProgress
        }
    }

    /// Product-facing live snapshot from `viewer`'s perspective (through `view_for`, so visibility is enforced by the types).
    pub fn snapshot(&self, viewer: u8) -> LiveMatchSnapshot {
        let players = self.players;
        let ended = self.outcome.is_some() && self.failure.is_none();
        let (scores, honba, kyotaku) = if let Some(settlement) = &self.settlement {
            (
                settlement.scores.clone(),
                settlement.next.honba,
                settlement.next.kyotaku,
            )
        } else {
            (
                self.board.scores(),
                self.board.honba(),
                self.board.kyotaku(),
            )
        };
        let seats: Vec<SeatSnapshot> = (0..players)
            .map(|seat| self.seat_snapshot(seat, viewer))
            .collect();
        let rankings = if self.match_ended {
            rankings(&scores)
        } else {
            Vec::new()
        };
        LiveMatchSnapshot {
            schema_version: TRACE_SCHEMA_VERSION,
            variant: B::VARIANT,
            rule_profile: self.config.rule_profile.platform().as_str(),
            red_fives: self.config.rule_profile.red_fives(self.players as usize),
            players,
            phase: self.phase_str(),
            kyoku_index: self.ctx.kyoku_index,
            bakaze: self.ctx.bakaze,
            kyoku: self.board.kyoku(),
            honba,
            kyotaku,
            scores,
            dealer: self.board.oya(),
            current_turn: self.board.turn(),
            actor: self.pending.as_ref().and_then(|p| match p.kind {
                PendingKind::Turn => Some(p.actor),
                PendingKind::Reaction | PendingKind::Robbery => None,
            }),
            pending: self.pending.as_ref().map(pending_summary),
            dora_indicators: self
                .board
                .dora_indicators()
                .iter()
                .map(|t| t.to_string())
                .collect(),
            wall_remaining: self.board.wall_remaining(),
            seats,
            ended,
            failed: self.failure.is_some(),
            failure: self.failure.clone(),
            outcome: self.outcome_summary.clone(),
            rankings,
            agents: self.agent_statuses(),
        }
    }

    fn seat_snapshot(&self, seat: u8, viewer: u8) -> SeatSnapshot {
        let hand_tiles = self.board.seat_hand(seat);
        let hand = if seat == viewer {
            Some(hand_tiles.iter().map(|t| t.to_string()).collect())
        } else {
            None
        };
        SeatSnapshot {
            seat,
            hand,
            hand_count: hand_tiles.len() as u8,
            melds: self
                .board
                .seat_melds(seat)
                .iter()
                .map(render_meld)
                .collect(),
            discards: self
                .board
                .seat_discards(seat)
                .iter()
                .map(|t| t.to_string())
                .collect(),
            riichi: self.board.seat_riichi(seat),
            // Only for the viewer; `view_for` builds a whole SeatView, so skip it for other seats.
            furiten: (seat == viewer).then(|| {
                let view = self.board.view_for(seat);
                view.me.temporary_furiten || view.me.riichi_furiten
            }),
            nuki: self.board.nuki(seat),
        }
    }

    /// Runtime status snapshot (seat to agent/model, health, and whether it is manual).
    pub fn status(&self) -> LiveMatchStatus {
        let seats = self
            .agents
            .iter()
            .enumerate()
            .map(|(seat, agent)| {
                let st = agent.status();
                let manual = self.manual[seat];
                let state = if manual {
                    "manual"
                } else if st.errors > 0 {
                    "error"
                } else if st.fallbacks > 0 {
                    "degraded"
                } else {
                    "healthy"
                };
                LiveSeatStatus {
                    seat: seat as u8,
                    agent_kind: st.kind.to_string(),
                    model_id: st.model_id,
                    state,
                    manual,
                    calls: st.calls,
                    fallbacks: st.fallbacks,
                    errors: st.errors,
                    last_latency_ms: st.last_latency_ms,
                }
            })
            .collect();
        LiveMatchStatus {
            schema_version: TRACE_SCHEMA_VERSION,
            variant: B::VARIANT,
            rule_profile: self.config.rule_profile.platform().as_str(),
            red_fives: self.config.rule_profile.red_fives(self.players as usize),
            match_id: match_id(
                B::VARIANT,
                self.config.kind,
                self.config.seed,
                self.config.rule_profile,
            ),
            phase: self.phase_str(),
            failure: self.failure.clone(),
            seats,
        }
    }

    fn phase_str(&self) -> &'static str {
        if self.failure.is_some() {
            "failed"
        } else if self.match_ended {
            "match_end"
        } else if self.outcome.is_some() {
            "kyoku_end"
        } else if let Some(p) = &self.pending {
            match p.kind {
                PendingKind::Turn => "awaiting_turn",
                PendingKind::Reaction => "awaiting_reaction",
                PendingKind::Robbery => "awaiting_robbery",
            }
        } else {
            "in_progress"
        }
    }
}

/// Headless 4p with product live semantics: runs the whole match automatically on the
/// live core and returns a [`FullMatchReport`]. `specs` must have length 4.
///
/// The only difference from the public headless
/// [`crate::match_host::run_full_match_4p`] is that this builds seats through
/// [`LiveSeat`], so plugin `session_id`s use the product policy `live-match-*`
/// ([`crate::match_host::SessionIdPolicy::Live`]) while the public headless API keeps
/// `run-match-*`. Both use the same [`LiveMatchSession`] core; this is a thin wrapper
/// with a different session id policy.
pub fn run_full_match_4p_via_live(
    config: MatchConfig,
    specs: Vec<SeatSpec>,
) -> Result<FullMatchReport> {
    let seats: Vec<LiveSeat> = specs.into_iter().map(seat_spec_to_live).collect();
    let mut session = LiveMatchSession::new_4p(live_config(config), seats)?;
    session.run_to_match_end()?;
    Ok(session.report())
}

/// Headless 3p with product live semantics; see above. `specs` must have length 3.
pub fn run_full_match_3p_via_live(
    config: MatchConfig,
    specs: Vec<SeatSpec>,
) -> Result<FullMatchReport> {
    let seats: Vec<LiveSeat> = specs.into_iter().map(seat_spec_to_live).collect();
    let mut session = LiveMatchSession::new_3p(live_config(config), seats)?;
    session.run_to_match_end()?;
    Ok(session.report())
}

/// `MatchConfig` to `LiveMatchConfig` (type conversion only), so the public headless API can reuse the live core.
pub(crate) fn live_config(config: MatchConfig) -> LiveMatchConfig {
    LiveMatchConfig {
        seed: config.seed,
        kind: config.kind,
        rule_profile: config.rule_profile,
        start_score: config.start_score,
        // Headless batch self-play produces no match logs; keep the hot path unchanged.
        record_decision_windows: false,
    }
}

fn seat_spec_to_live(spec: SeatSpec) -> LiveSeat {
    match spec {
        SeatSpec::Algorithm(kind) => LiveSeat::Algorithm(kind),
        SeatSpec::Plugin(rt) => LiveSeat::Plugin(rt),
        SeatSpec::RemoteModel(config) => LiveSeat::RemoteModel(config),
    }
}

fn build_live_agent_4p(
    seat: u8,
    spec: LiveSeat,
    seed: u64,
    kind: MatchKind,
) -> Result<(Box<dyn SeatAgent<Variant4p>>, bool)> {
    match spec {
        LiveSeat::Manual => Ok((
            Box::new(ManualSeat::new()) as Box<dyn SeatAgent<Variant4p>>,
            true,
        )),
        LiveSeat::Algorithm(algo) => Ok((
            Box::new(AlgorithmSeatAgent::new(seat, algo)) as Box<dyn SeatAgent<Variant4p>>,
            false,
        )),
        LiveSeat::Plugin(rt) => {
            let session = SessionIdPolicy::Live.plugin_session_id(4, seed, seat);
            let agent = LocalPluginSeatAgent::from_runtime_for_variant::<Variant4p>(
                seat, &rt, session, kind,
            )?;
            Ok((Box::new(agent) as Box<dyn SeatAgent<Variant4p>>, false))
        }
        LiveSeat::RemoteModel(config) => Ok((
            Box::new(RemoteModelSeat::new(config)) as Box<dyn SeatAgent<Variant4p>>,
            false,
        )),
    }
}

fn build_live_agent_3p(
    seat: u8,
    spec: LiveSeat,
    seed: u64,
    kind: MatchKind,
) -> Result<(Box<dyn SeatAgent<Variant3p>>, bool)> {
    match spec {
        LiveSeat::Manual => Ok((
            Box::new(ManualSeat::new()) as Box<dyn SeatAgent<Variant3p>>,
            true,
        )),
        LiveSeat::Algorithm(algo) => Ok((
            Box::new(AlgorithmSeatAgent::new(seat, algo)) as Box<dyn SeatAgent<Variant3p>>,
            false,
        )),
        LiveSeat::Plugin(rt) => {
            let session = SessionIdPolicy::Live.plugin_session_id(3, seed, seat);
            let agent = LocalPluginSeatAgent::from_runtime_for_variant::<Variant3p>(
                seat, &rt, session, kind,
            )?;
            Ok((Box::new(agent) as Box<dyn SeatAgent<Variant3p>>, false))
        }
        LiveSeat::RemoteModel(config) => Ok((
            Box::new(RemoteModelSeat::new(config)) as Box<dyn SeatAgent<Variant3p>>,
            false,
        )),
    }
}

fn pending_summary(p: &PendingWindow) -> PendingSummary {
    let legal_action_count = match p.kind {
        PendingKind::Turn => p.turn_actions.len(),
        PendingKind::Reaction | PendingKind::Robbery => {
            p.reaction_seats.iter().map(|r| r.legal.len()).sum()
        }
    };
    PendingSummary {
        decision_id: p.decision_id.clone(),
        state_digest: p.state_digest.clone(),
        kind: p.kind,
        seats: p.waiting_seats(),
        legal_action_count,
    }
}

fn reaction_legal_descriptions(legal: &[ReactionAction]) -> Vec<String> {
    let mut descriptions = vec![describe_reaction(&ReactionAction::Pass)];
    descriptions.extend(legal.iter().map(describe_reaction));
    descriptions
}

fn render_meld(meld: &Meld) -> MeldSnapshot {
    let kind = match meld {
        Meld::Chi { .. } => "chi",
        Meld::Pon { .. } => "pon",
        Meld::Daiminkan { .. } => "daiminkan",
        Meld::Kakan { .. } => "kakan",
        Meld::Ankan { .. } => "ankan",
        Meld::Nukidora { .. } => "nukidora",
    };
    MeldSnapshot {
        kind,
        tiles: meld.tiles().iter().map(|t| t.to_string()).collect(),
    }
}
