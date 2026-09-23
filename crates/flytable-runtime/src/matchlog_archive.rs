//! Assembles a hand into a complete, archivable match log.
//!
//! Of the four match log layers, L0 (facts), L1 (ruling timing) and L3 (decision
//! windows) have their own producers. This module does L2 (settlement): it builds
//! `MatchlogEvent::Hora` / `Ryukyoku` and connects a live match session to a complete
//! match log.
//!
//! It lives in the runtime because assembly needs three things at once: the
//! authoritative event stream (projected into L0/L1), the decision windows (L3,
//! rebased from table to match log coordinates), and the settlement plus final table
//! state (the winner's hand, melds, winning tile and ura dora for L2). Only
//! `LiveMatchSession` has all three.
//!
//! Archiving is refused rather than producing a log that only looks complete:
//!
//! 1. `validate_window` is not automatic. serde only checks shape; window invariants
//!    are checked explicitly.
//! 2. A non-empty `decision_window_faults()` means L3 is incomplete and archiving must
//!    be refused, rather than producing training or statistics data with silently
//!    missing windows.
//!
//! Both are enforced in [`archive_kyoku_4p`]; failures return [`ArchiveError`] instead
//! of degrading.

use flytable_event::matchlog::{
    profile_fingerprint, validate, DecisionWindow, MatchlogEvent, RuleEra, RyukyokuKind,
    TileIdentity, MATCHLOG_SCHEMA,
};
use flytable_protocol::matchlog_adapters::{
    matchlog_from_engine_3p_indexed, matchlog_from_engine_4p_indexed,
};
use flytable_seat::contract::CanonicalLegalAction;
use flytable_table::progress::{
    split_multi_hora, split_settlement, MatchLength, RoundState, Settlement, SettlementSplit,
};
use flytable_table::{
    AbortiveRyukyokuReason3p, AbortiveRyukyokuReason4p, Board3p, Board4p, KyokuOutcome,
    KyokuOutcome3p,
};

use flytable_table::matchlog_settlement::{
    hora_body, ron_tile_3p, ron_tile_4p, ryukyoku_event, tr, trs,
};

use crate::decision_window::validate_window;
use crate::live::{hora_payment_with_pao, LiveBoard, LiveMatchSession};

/// Assembly failure. None of these can be ignored; each means some layer of the log is incomplete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchiveError {
    /// The hand has not ended; there is no L2 to write.
    KyokuNotFinished,
    /// Recording L3 failed, so the decision window layer is incomplete (rule 2).
    IncompleteDecisionWindows(Vec<String>),
    /// A window is inconsistent (rule 1).
    MalformedWindow { window_id: u64, reason: String },
    /// A window's table anchor could not be converted to match log coordinates.
    AnchorRebaseFailed(String),
    /// A required fact for L2 was missing.
    MissingSettlementFact(String),
    /// The produced event stream failed the match log's own invariant checks.
    InvalidEventStream(String),
}

impl std::fmt::Display for ArchiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KyokuNotFinished => write!(f, "the hand has not ended and cannot be archived"),
            Self::IncompleteDecisionWindows(v) => {
                write!(
                    f,
                    "L3 decision windows incomplete ({} faults): {v:?}",
                    v.len()
                )
            }
            Self::MalformedWindow { window_id, reason } => {
                write!(f, "window {window_id} is inconsistent: {reason}")
            }
            Self::AnchorRebaseFailed(e) => write!(f, "anchor conversion failed: {e}"),
            Self::MissingSettlementFact(e) => {
                write!(f, "settlement layer is missing a required fact: {e}")
            }
            Self::InvalidEventStream(e) => write!(f, "event stream failed validation: {e}"),
        }
    }
}
impl std::error::Error for ArchiveError {}

/// Archive of one hand.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KyokuArchive {
    /// L0 + L1 + L2 in order.
    pub events: Vec<MatchlogEvent>,
    /// L3, with anchors converted to the coordinates of `events`.
    ///
    /// Empty when `record_decision_windows` is off; the log is still complete (L0 to L2),
    /// just without decision windows. Hence `default` rather than required.
    #[serde(default)]
    pub windows: Vec<DecisionWindow<CanonicalLegalAction>>,
}

/// Archive of a whole match: header plus hands. This is the unit written to disk.
///
/// The rule profile snapshot in `StartMatch` is authoritative for replay and exists
/// once per match. Storing hands separately would copy it into every hand, and
/// diverging copies would leave no way to tell which one counts.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchArchive {
    /// Header (`StartMatch`), once per match.
    pub start_match: MatchlogEvent,
    pub kyokus: Vec<KyokuArchive>,
}

impl MatchArchive {
    /// Flattens the match into a single match log event stream.
    ///
    /// This is the shape `flytable_protocol::matchlog_replay::ReplayCursor::new` expects:
    /// it splits hands at `StartKyoku`, and every [`KyokuArchive`] starts with one. Save,
    /// read back, flatten, and the cursor can seek.
    #[must_use]
    pub fn flatten(&self) -> Vec<MatchlogEvent> {
        let mut out =
            Vec::with_capacity(1 + self.kyokus.iter().map(|k| k.events.len()).sum::<usize>());
        out.push(self.start_match.clone());
        for k in &self.kyokus {
            out.extend(k.events.iter().cloned());
        }
        out
    }

    /// Invariant check of the whole match (the same `validate` as when archiving each hand).
    ///
    /// # Errors
    ///
    /// Returns a readable summary of the violations. Logs read back from disk should be
    /// checked again, since the file may have been edited or written by another version.
    pub fn validate_all(&self) -> Result<(), String> {
        let v = validate(&self.flatten());
        if v.is_empty() {
            return Ok(());
        }
        Err(format!(
            "{} violations, first: {:?}",
            v.len(),
            v.first().expect("checked non-empty above")
        ))
    }
}

/// Assembles the 4P archive of one hand.
///
/// # Errors
///
/// See [`ArchiveError`]. None of them is let through.
pub fn archive_kyoku_4p(session: &LiveMatchSession<Board4p>) -> Result<KyokuArchive, ArchiveError> {
    let (outcome, settlement) = session
        .last_settled_round()
        .ok_or(ArchiveError::KyokuNotFinished)?;

    // Rule 2: a recording fault means L3 is incomplete; refuse to archive.
    let faults = session.decision_window_faults();
    if !faults.is_empty() {
        return Err(ArchiveError::IncompleteDecisionWindows(
            faults.iter().map(ToString::to_string).collect(),
        ));
    }

    let board = session.board();
    let raw = session.authoritative_events();
    let (body, map) = matchlog_from_engine_4p_indexed(raw);

    // The projector only produces in-hand L0/L1 events, without the header and the
    // start-of-hand facts. They are added here rather than in the projector, which is
    // already validated and unrelated to these two events.
    let mut events = vec![start_kyoku_event_4p(raw)?];
    let head_len = events.len();
    events.extend(body);

    // L3: convert to match log coordinates and check each window (rule 1).
    let mut windows = Vec::with_capacity(session.decision_windows().len());
    for w in session.decision_windows() {
        validate_window(w).map_err(|reason| ArchiveError::MalformedWindow {
            window_id: w.window_id,
            reason,
        })?;
        let mut rebased = w
            .clone()
            .rebase(&map)
            .map_err(ArchiveError::AnchorRebaseFailed)?;
        // The prepended events shift the stream, so anchors shift by the same amount.
        rebased.anchor_seq += head_len as u64;
        windows.push(rebased);
    }

    // L2: the core of this module.
    let split = split_settlement(session.round_state(), |st| {
        board.settle(st, session.match_length(), outcome)
    });
    events.extend(settlement_events_4p(
        board,
        outcome,
        settlement,
        &split,
        session.round_state(),
        session.match_length(),
    )?);

    // Validate the complete stream with the header (`validate` first checks for start_match).
    let mut full = vec![start_match_event(board.rule_profile, 4, None)];
    full.extend(events.iter().cloned());
    let violations = validate(&full);
    if !violations.is_empty() {
        return Err(ArchiveError::InvalidEventStream(format!(
            "{} violations, first: {:?}",
            violations.len(),
            violations[0]
        )));
    }

    Ok(KyokuArchive { events, windows })
}

/// Multiple ron: build the three-part split for each winner.
///
/// Honba and riichi sticks go only to the first winner (`first`); the others only get
/// their base points. This follows `settle_multi_hora_with_profile`, which only adds
/// honba for `index == 0`.
fn per_winner_splits(
    state: &RoundState,
    length: MatchLength,
    profile: flytable_core::rules::RiichiRuleProfile,
    seats: usize,
    first: flytable_table::progress::HoraPayment,
    additional: &[flytable_table::progress::HoraPayment],
) -> Vec<SettlementSplit> {
    let multi = split_multi_hora(state, length, first, additional, profile);
    let zeros = vec![0i32; seats];
    (0..=additional.len())
        .map(|i| SettlementSplit {
            base: multi.per_winner_base[i].clone(),
            honba: if i == 0 {
                multi.honba.clone()
            } else {
                zeros.clone()
            },
            kyotaku: if i == 0 {
                multi.kyotaku.clone()
            } else {
                zeros.clone()
            },
        })
        .collect()
}

/// Settlement layer. Returns a `Vec` because multiple ron emits one win event per
/// winner; one event only holds one winner's yaku, hand and points.
fn settlement_events_4p(
    board: &Board4p,
    outcome: &KyokuOutcome,
    settlement: &Settlement,
    split: &SettlementSplit,
    state: &RoundState,
    length: MatchLength,
) -> Result<Vec<MatchlogEvent>, ArchiveError> {
    match outcome {
        KyokuOutcome::Hora {
            winner,
            from,
            score,
        } => Ok(vec![MatchlogEvent::Hora(Box::new(
            hora_body(
                &board.players,
                &board.wall.ura_indicators(),
                ron_tile_4p(board.last_discard, &board.log),
                board.bakaze,
                board.seat_wind(*winner),
                board.pao_payer(*winner),
                *winner,
                *from,
                score,
                split,
            )
            .map_err(ArchiveError::MissingSettlementFact)?,
        ))]),
        KyokuOutcome::MultiHora { first, additional } => {
            let pay = |h: &flytable_table::HoraResult| {
                hora_payment_with_pao(
                    h.winner,
                    h.from,
                    &h.score,
                    board.pao_liability(h.winner),
                    board.players.len(),
                )
            };
            let extra: Vec<_> = additional.iter().map(pay).collect();
            let splits = per_winner_splits(
                state,
                length,
                board.rule_profile,
                board.players.len(),
                pay(first),
                &extra,
            );
            let mut out = Vec::with_capacity(additional.len() + 1);
            for (h, per) in std::iter::once(first).chain(additional.iter()).zip(&splits) {
                out.push(MatchlogEvent::Hora(Box::new(
                    hora_body(
                        &board.players,
                        &board.wall.ura_indicators(),
                        ron_tile_4p(board.last_discard, &board.log),
                        board.bakaze,
                        board.seat_wind(h.winner),
                        board.pao_payer(h.winner),
                        h.winner,
                        h.from,
                        &h.score,
                        per,
                    )
                    .map_err(ArchiveError::MissingSettlementFact)?,
                )));
            }
            Ok(out)
        }
        KyokuOutcome::Ryukyoku { tenpai } => Ok(vec![ryukyoku_event(
            &board.players,
            RyukyokuKind::Exhaustive,
            tenpai,
            settlement,
            split,
        )]),
        KyokuOutcome::NagashiMangan { tenpai, .. } => Ok(vec![ryukyoku_event(
            &board.players,
            RyukyokuKind::NagashiMangan,
            tenpai,
            settlement,
            split,
        )]),
        KyokuOutcome::AbortiveRyukyoku { reason } => {
            let kind = match reason {
                AbortiveRyukyokuReason4p::KyuushuKyuuhai => RyukyokuKind::KyuushuKyuuhai,
                AbortiveRyukyokuReason4p::Suukaikan => RyukyokuKind::Suukaikan,
                AbortiveRyukyokuReason4p::SuufonRenda => RyukyokuKind::SuufonRenda,
                AbortiveRyukyokuReason4p::SuuchaRiichi => RyukyokuKind::SuuchaRiichi,
                AbortiveRyukyokuReason4p::Sanchaho => RyukyokuKind::SanchaHora,
            };
            // Abortive draws reveal no tenpai and have no noten penalties.
            Ok(vec![ryukyoku_event(
                &board.players,
                kind,
                &[],
                settlement,
                split,
            )])
        }
    }
}

/// Header. The inline rule profile snapshot is authoritative for replay; nothing is
/// looked up from configuration at replay time.
///
/// Public because the match archive uses it as [`MatchArchive::start_match`]; each
/// hand's archive builds its own copy for validation, but that copy is not output.
///
/// `settlement` is filled in by whoever sets up the table; the engine does not guess.
/// Rank points and oka vary by room on some platforms. They are a ranking policy
/// rather than a game rule, so `RiichiRuleProfile` deliberately does not contain them,
/// and only the table creator knows whether a table is ranked and how.
///
/// `None` means the log carries no ranking settlement (self-play, debugging), which is
/// a valid form. Tenhou's two presets are `MatchSettlementProfile::tenhou_4p()` /
/// `tenhou_3p()`.
#[must_use]
pub fn start_match_event(
    profile: flytable_core::rules::RiichiRuleProfile,
    seats: u8,
    settlement: Option<flytable_event::matchlog::MatchSettlementProfile>,
) -> MatchlogEvent {
    MatchlogEvent::StartMatch {
        schema: MATCHLOG_SCHEMA.to_string(),
        seats,
        rule_profile: profile,
        rule_era: RuleEra("tenhou/flytable".to_string()),
        profile_fingerprint: profile_fingerprint(&profile),
        settlement_profile: settlement,
        // The engine works at kind level; the event stream has no physical tile ids.
        tile_identity: TileIdentity::None,
        names: None,
        seed: None,
    }
}

/// Converts the first `StartKyoku` of the table log into the match log's start-of-hand facts.
fn start_kyoku_event_4p(raw: &[flytable_event::Event4p]) -> Result<MatchlogEvent, ArchiveError> {
    let Some(flytable_event::Event4p::StartKyoku {
        bakaze,
        dora_marker,
        kyoku,
        honba,
        kyotaku,
        oya,
        scores,
        tehais,
    }) = raw.first()
    else {
        return Err(ArchiveError::MissingSettlementFact(
            "the table log does not start with StartKyoku".into(),
        ));
    };
    Ok(MatchlogEvent::StartKyoku {
        bakaze: *bakaze,
        kyoku: *kyoku,
        honba: *honba,
        kyotaku: *kyotaku,
        oya: *oya,
        scores: scores.to_vec(),
        haipai: tehais.iter().map(|h| trs(h)).collect(),
        dora_marker: tr(*dora_marker),
    })
}

/// Assembles the 3P archive of one hand.
///
/// Differs from 4P in only three ways: a different projector, `KyokuOutcome3p` is a
/// different type, and the only abortive draws are kyuushu kyuuhai and four kans
/// (four riichi, four winds and triple ron cannot happen). Nukidora uses the same meld
/// conversion as `MeldKind::Kita`, with its han in `nuki_han`.
///
/// # Errors
///
/// See [`ArchiveError`].
pub fn archive_kyoku_3p(session: &LiveMatchSession<Board3p>) -> Result<KyokuArchive, ArchiveError> {
    let (outcome, settlement) = session
        .last_settled_round()
        .ok_or(ArchiveError::KyokuNotFinished)?;

    let faults = session.decision_window_faults();
    if !faults.is_empty() {
        return Err(ArchiveError::IncompleteDecisionWindows(
            faults.iter().map(ToString::to_string).collect(),
        ));
    }

    let board = session.board();
    let raw = session.authoritative_events();
    let (body, map) = matchlog_from_engine_3p_indexed(raw);

    let mut events = vec![start_kyoku_event_3p(raw)?];
    let head_len = events.len();
    events.extend(body);

    let mut windows = Vec::with_capacity(session.decision_windows().len());
    for w in session.decision_windows() {
        validate_window(w).map_err(|reason| ArchiveError::MalformedWindow {
            window_id: w.window_id,
            reason,
        })?;
        let mut rebased = w
            .clone()
            .rebase(&map)
            .map_err(ArchiveError::AnchorRebaseFailed)?;
        rebased.anchor_seq += head_len as u64;
        windows.push(rebased);
    }

    let split = split_settlement(session.round_state(), |st| {
        board.settle(st, session.match_length(), outcome)
    });
    events.extend(settlement_events_3p(
        board,
        outcome,
        settlement,
        &split,
        session.round_state(),
        session.match_length(),
    )?);

    let mut full = vec![start_match_event(board.rule_profile, 3, None)];
    full.extend(events.iter().cloned());
    let violations = validate(&full);
    if !violations.is_empty() {
        return Err(ArchiveError::InvalidEventStream(format!(
            "{} violations, first: {:?}",
            violations.len(),
            violations[0]
        )));
    }

    Ok(KyokuArchive { events, windows })
}

fn settlement_events_3p(
    board: &Board3p,
    outcome: &KyokuOutcome3p,
    settlement: &Settlement,
    split: &SettlementSplit,
    state: &RoundState,
    length: MatchLength,
) -> Result<Vec<MatchlogEvent>, ArchiveError> {
    match outcome {
        KyokuOutcome3p::Hora {
            winner,
            from,
            score,
        } => Ok(vec![MatchlogEvent::Hora(Box::new(
            hora_body(
                &board.players,
                &board.wall.ura_indicators(),
                ron_tile_3p(board.last_discard, &board.log),
                board.bakaze,
                board.seat_wind(*winner),
                board.pao_payer(*winner),
                *winner,
                *from,
                score,
                split,
            )
            .map_err(ArchiveError::MissingSettlementFact)?,
        ))]),
        KyokuOutcome3p::MultiHora { first, additional } => {
            let pay = |h: &flytable_table::HoraResult| {
                hora_payment_with_pao(
                    h.winner,
                    h.from,
                    &h.score,
                    board.pao_liability(h.winner),
                    board.players.len(),
                )
            };
            let extra: Vec<_> = additional.iter().map(pay).collect();
            let splits = per_winner_splits(
                state,
                length,
                board.rule_profile,
                board.players.len(),
                pay(first),
                &extra,
            );
            let mut out = Vec::with_capacity(additional.len() + 1);
            for (h, per) in std::iter::once(first).chain(additional.iter()).zip(&splits) {
                out.push(MatchlogEvent::Hora(Box::new(
                    hora_body(
                        &board.players,
                        &board.wall.ura_indicators(),
                        ron_tile_3p(board.last_discard, &board.log),
                        board.bakaze,
                        board.seat_wind(h.winner),
                        board.pao_payer(h.winner),
                        h.winner,
                        h.from,
                        &h.score,
                        per,
                    )
                    .map_err(ArchiveError::MissingSettlementFact)?,
                )));
            }
            Ok(out)
        }
        KyokuOutcome3p::Ryukyoku { tenpai } => Ok(vec![ryukyoku_event(
            &board.players,
            RyukyokuKind::Exhaustive,
            tenpai,
            settlement,
            split,
        )]),
        KyokuOutcome3p::NagashiMangan { tenpai, .. } => Ok(vec![ryukyoku_event(
            &board.players,
            RyukyokuKind::NagashiMangan,
            tenpai,
            settlement,
            split,
        )]),
        KyokuOutcome3p::AbortiveRyukyoku { reason } => {
            let kind = match reason {
                AbortiveRyukyokuReason3p::KyuushuKyuuhai => RyukyokuKind::KyuushuKyuuhai,
                AbortiveRyukyokuReason3p::Suukaikan => RyukyokuKind::Suukaikan,
            };
            Ok(vec![ryukyoku_event(
                &board.players,
                kind,
                &[],
                settlement,
                split,
            )])
        }
    }
}

fn start_kyoku_event_3p(raw: &[flytable_event::Event3p]) -> Result<MatchlogEvent, ArchiveError> {
    let Some(flytable_event::Event3p::StartKyoku {
        bakaze,
        dora_marker,
        kyoku,
        honba,
        kyotaku,
        oya,
        scores,
        tehais,
    }) = raw.first()
    else {
        return Err(ArchiveError::MissingSettlementFact(
            "the table log does not start with StartKyoku".into(),
        ));
    };
    Ok(MatchlogEvent::StartKyoku {
        bakaze: *bakaze,
        kyoku: *kyoku,
        honba: *honba,
        kyotaku: *kyotaku,
        oya: *oya,
        scores: scores.to_vec(),
        haipai: tehais.iter().map(|h| trs(h)).collect(),
        dora_marker: tr(*dora_marker),
    })
}
