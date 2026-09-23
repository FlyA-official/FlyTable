//! Emits L3 decision windows from the host.
//!
//! Only the host holds all three parts of a window at once: the authoritative legal
//! set (enumerated before calling the agent, for fallback and validation), the actual
//! choice, and the event anchor. `flytable-protocol` is a pure contract crate (only
//! core and event); making it depend on the table would create a cycle
//! (table -> seat -> protocol).
//!
//! Rules:
//!
//! - The caller passes the legal set; this module never enumerates it again. The host
//!   already computed `legal_turn_actions`, and a second computation could disagree.
//! - Pass the chosen action, not an index. The candidates of a response window are not
//!   a one-to-one mapping of the legal set: `Pass` and 3-player `Chi` are filtered out
//!   by [`Variant::reaction_to_legal`], so legal set indices and window indices differ.
//!   Leaving the conversion to callers would invite silent misalignment at every call
//!   site. Matching uses `ReactionAction::equivalent` rather than `==`, the same rule as
//!   the host's `match_host::reaction_in_legal`: chi/pon consumed tiles are an
//!   unordered multiset.
//! - Passing is an explicit action. Response and robbery windows must offer `PassAll`
//!   and `chosen` points at it; `chosen = None` never means pass. Turn windows never
//!   offer `PassAll`.

use flytable_core::tile::Tile;
use flytable_event::matchlog::{DecisionWindow, WindowPhase};
use flytable_seat::contract::{CanonicalAction, CanonicalLegalAction, Declines};
use flytable_table::{ReactionAction, SeatView, TurnAction};

use crate::variant::Variant;

/// A window emitted by the host.
///
/// Not a [`DecisionWindow`]: the anchor is in a different coordinate system. The host
/// only has the table event log, while the match log's `anchor_seq` is a match log
/// cursor. The streams are not 1:1 (`Reach` folds into the next `Dahai`; `Kakan` /
/// `Ankan` expand into `Call` plus `RobberyWindow`), so the conversion belongs to the
/// projection layer; see [`HostWindow::rebase`].
///
/// Two types rather than one field with two meanings, so a table index can never be
/// stored where a match log index is expected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostWindow {
    pub window_id: u64,
    pub seat: u8,
    pub phase: WindowPhase,
    /// Cursor into the authoritative table event log (`Board::log_len()`, the number of
    /// table events so far). Excludes the `StartGame` synthesized for the inference wire.
    pub anchor_board_seq: u64,
    pub offers: Vec<CanonicalLegalAction>,
    pub chosen: usize,
}

impl HostWindow {
    /// Converts into match log coordinates, giving an archivable [`DecisionWindow`].
    ///
    /// `map` is the prefix cursor map from the projector (the second return value of
    /// `flytable_protocol::matchlog_adapters::matchlog_from_engine_4p_indexed` /
    /// `..._3p_indexed`) and must come from the same hand and the same table log.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the anchor is outside `map`, which means the window and the log
    /// do not match (usually the map of another hand). An error is better than a
    /// plausible-looking index.
    pub fn rebase(self, map: &[usize]) -> Result<DecisionWindow<CanonicalLegalAction>, String> {
        let anchor = *map.get(self.anchor_board_seq as usize).ok_or_else(|| {
            format!(
                "window {} has table anchor {} outside the projection map ({} entries); does the map come from a different event log?",
                self.window_id,
                self.anchor_board_seq,
                map.len()
            )
        })?;
        Ok(DecisionWindow {
            window_id: self.window_id,
            seat: self.seat,
            phase: self.phase,
            anchor_seq: anchor as u64,
            offers: self.offers,
            chosen: self.chosen,
        })
    }
}

/// A fault while recording an L3 window.
///
/// Structured so the match log builder can mark exactly which window is missing; a
/// gap in the sequence is not enough, since a failure on the last window leaves no gap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowFault {
    pub window_id: u64,
    pub seat: u8,
    pub phase: WindowPhase,
    pub anchor_board_seq: u64,
    pub reason: String,
}

impl std::fmt::Display for WindowFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "window {} (seat {}, {:?}, table anchor {}): {}",
            self.window_id, self.seat, self.phase, self.anchor_board_seq, self.reason
        )
    }
}

fn numbered(actions: Vec<CanonicalAction>) -> Vec<CanonicalLegalAction> {
    actions
        .into_iter()
        .enumerate()
        .map(|(action_id, action)| CanonicalLegalAction { action_id, action })
        .collect()
}

/// Packs a turn window.
///
/// `chosen` is the action the host actually applied (the fallback when the model was
/// out of range, not the model's raw submission); the window records what really
/// happened at the table.
///
/// # Errors
///
/// Returns `Err` if `chosen` is not in `legal`. That is an internal inconsistency of
/// the host and unreachable in normal play (it is checked with `legal.contains`
/// before applying). The caller decides what to do; this module does not substitute
/// another action, since recording a wrong move is worse than not recording it.
pub fn turn_window<V: Variant>(
    window_id: u64,
    anchor_board_seq: u64,
    seat: u8,
    view: &SeatView,
    legal: &[TurnAction],
    chosen: &TurnAction,
) -> Result<HostWindow, String> {
    let chosen_idx = legal.iter().position(|a| a == chosen).ok_or_else(|| {
        format!("turn window: applied action {chosen:?} is not in the authoritative legal set")
    })?;
    // Turn actions map one to one, so candidate indices equal legal set indices.
    let offers = numbered(
        legal
            .iter()
            .map(|a| V::legal_to_canonical(V::turn_action_to_legal(view, a)))
            .collect(),
    );
    Ok(HostWindow {
        window_id,
        seat,
        phase: WindowPhase::Turn,
        anchor_board_seq,
        offers,
        chosen: chosen_idx,
    })
}

/// Packs a response window (`robbery = false`) or a robbery window (`true`).
///
/// Passing `ReactionAction::Pass` as `chosen` means pass and resolves to the index of `PassAll`.
///
/// # Errors
///
/// Returns `Err` if `chosen` is neither `Pass` nor equivalent to a candidate (as in
/// [`turn_window`], an internal inconsistency; it is not turned into `Pass`).
#[allow(clippy::too_many_arguments)]
pub fn reaction_window<V: Variant>(
    window_id: u64,
    anchor_board_seq: u64,
    seat: u8,
    view: &SeatView,
    discarder: u8,
    tile: Tile,
    legal: &[ReactionAction],
    chosen: &ReactionAction,
    robbery: bool,
) -> Result<HostWindow, String> {
    let declines = Declines {
        ron: legal.iter().any(|a| matches!(a, ReactionAction::Ron)),
        call: legal
            .iter()
            .any(|a| !matches!(a, ReactionAction::Ron | ReactionAction::Pass)),
    };
    // Resolve the index while building the candidates: they are not a one-to-one mapping
    // of the legal set (Pass and 3-player Chi are filtered out), so both index spaces
    // must line up in the same pass.
    let mut canon: Vec<CanonicalAction> = Vec::with_capacity(legal.len() + 1);
    let mut chosen_idx = None;
    for action in legal {
        let Some(legal_action) = V::reaction_to_legal(view, action, discarder, tile) else {
            continue;
        };
        if chosen_idx.is_none() && action.equivalent(chosen) {
            chosen_idx = Some(canon.len());
        }
        canon.push(V::legal_to_canonical(legal_action));
    }
    let pass_idx = canon.len();
    canon.push(V::legal_to_canonical(V::pass_all(
        declines.ron,
        declines.call,
    )));
    let chosen = match chosen {
        ReactionAction::Pass => pass_idx,
        other => chosen_idx.ok_or_else(|| {
            format!("response window: applied action {other:?} is not among the window's offers")
        })?,
    };
    Ok(HostWindow {
        window_id,
        seat,
        phase: if robbery {
            WindowPhase::Robbery
        } else {
            WindowPhase::Reaction
        },
        anchor_board_seq,
        offers: numbered(canon),
        chosen,
    })
}

/// Window self-consistency check (for conformance gates and debugging).
///
/// # Errors
///
/// Three malformations: `chosen` out of range, `action_id` not equal to its index, or
/// `PassAll` present or absent against the phase.
pub fn validate_window(w: &HostWindow) -> Result<(), String> {
    if w.chosen >= w.offers.len() {
        return Err(format!(
            "chosen {} out of range ({} offers)",
            w.chosen,
            w.offers.len()
        ));
    }
    for (i, o) in w.offers.iter().enumerate() {
        if o.action_id != i {
            return Err(format!(
                "offer {i} has action_id {}, which must equal its index (downstream matches ids back to indices)",
                o.action_id
            ));
        }
    }
    let has_pass = w
        .offers
        .iter()
        .any(|o| matches!(o.action, CanonicalAction::PassAll { .. }));
    match (w.phase.requires_pass_option(), has_pass) {
        (true, false) => Err("response and robbery windows must offer PassAll".into()),
        (false, true) => Err("turn windows must not offer PassAll".into()),
        _ => Ok(()),
    }
}
