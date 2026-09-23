//! L3 decision windows of the match log.
//!
//! Review and training tools both need the same thing: what a player could do at a
//! given moment and what they did. Without it they would have to replay the game and
//! enumerate legal actions themselves, which is a second rules implementation outside
//! the engine.
//!
//! The layer is thin: actions reuse `CanonicalLegalAction` from `flytable-protocol`,
//! and this layer only adds the window (seat, phase, anchor, choice).
//! `CanonicalLegalAction` alone has only `action_id` and `action`, which is not
//! enough to describe a window.
//!
//! Passing is always an explicit `PassAll` offer; `None` never means pass. Reaction
//! and robbery windows must offer `PassAll`, and choosing it means passing. `Turn`
//! windows never offer `PassAll`, since a turn must end in a discard or declaration.

use serde::{Deserialize, Serialize};

/// Phase of a decision window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowPhase {
    /// Own turn after drawing: discard, or declare tsumo / kyuushu kyuuhai / kan /
    /// nukidora / riichi. No `PassAll`.
    Turn,
    /// Response window after another player's discard (chi / pon / open kan / ron / pass).
    Reaction,
    /// Robbing window after a kan or nukidora (chankan / robbing North / pass).
    Robbery,
}

impl WindowPhase {
    /// Whether `offers` must contain `PassAll` in this phase.
    #[must_use]
    pub const fn requires_pass_option(self) -> bool {
        matches!(self, Self::Reaction | Self::Robbery)
    }
}

/// A decision window.
///
/// `A` is the action type. This crate does not depend on `flytable-protocol`, so the
/// caller supplies `CanonicalLegalAction`, reusing its vocabulary without a
/// dependency cycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionWindow<A> {
    /// Monotonic within a hand. `u64`, matching the wire `seq`.
    pub window_id: u64,
    pub seat: u8,
    pub phase: WindowPhase,
    /// Match log event cursor when the window opens: the number of match log events
    /// so far (half-open, same convention as the envelope's
    /// `to_seq = from_seq + events.len()`; well defined on an empty stream).
    ///
    /// This is in match log space, not table event log space. The two are not 1:1:
    /// `Reach` folds into the following `Dahai`, and `Kakan` / `Ankan` expand into
    /// `Call` plus `RobberyWindow`. The host only knows the table cursor; the
    /// projection layer converts it (see `HostWindow::rebase` in `flytable-runtime`).
    pub anchor_seq: u64,
    /// Offered actions.
    pub offers: Vec<A>,
    /// Required. Equals the `action_id` of one entry in `offers`; passing is expressed by
    /// pointing at `PassAll`. `usize`, matching `CanonicalLegalAction::action_id`.
    pub chosen: usize,
}

impl<A> DecisionWindow<A> {
    /// Whether `chosen` is within the index range of `offers`.
    ///
    /// Only a range check. Checking that `chosen` equals an entry's `action_id` requires
    /// the action type and is done by `flytable-protocol`.
    #[must_use]
    pub fn chosen_in_range(&self) -> bool {
        self.chosen < self.offers.len()
    }
}
