//! Read-only derived phase of a board, and typed transition errors.
//!
//! What a board can do next used to be inferred from five fields (`terminal`,
//! `last_discard`, `pending_robbery`, `drawn_tile`, hand size mod 3), and
//! `draw_for_turn()` returned the same `None` for five different cases:
//!
//! * live wall exhausted (a normal exhaustive draw; the caller should end the hand);
//! * hand already over (a caller bug);
//! * wrong phase (not discarded yet, or already drawn);
//! * a pending discard or kan declaration awaiting responses;
//! * an abnormal hand size for the current seat.
//!
//! Callers had to check `wall_remaining()` to tell a draw from a bug.
//!
//! This module is additive: it adds the derived [`BoardPhase`], [`TransitionError`]
//! and a typed entry point (`try_draw_for_turn`). The existing entry points delegate
//! to it with unchanged semantics, so consumers can migrate at their own pace.

use flytable_core::tile::Tile;

/// Current phase of a board, derived from existing fields so it cannot drift from the actual state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoardPhase {
    /// The hand is over; no further progression.
    Terminal,
    /// `seat` is to draw.
    AwaitingDraw { seat: u8 },
    /// `seat` has drawn (or just called) and must discard.
    AwaitingDiscard { seat: u8 },
    /// `tile` discarded by `discarder` is awaiting responses (chi / pon / kan / ron).
    ReactionWindow { discarder: u8, tile: Tile },
    /// A kan or nukidora by `actor` is awaiting robbing responses.
    RobberyWindow { actor: u8 },
}

impl BoardPhase {
    /// Stable phase name for error messages and logs.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Terminal => "terminal",
            Self::AwaitingDraw { .. } => "awaiting_draw",
            Self::AwaitingDiscard { .. } => "awaiting_discard",
            Self::ReactionWindow { .. } => "reaction_window",
            Self::RobberyWindow { .. } => "robbery_window",
        }
    }

    /// Seat to act now (`None` in a response window, which has no single actor).
    #[must_use]
    pub const fn actor(self) -> Option<u8> {
        match self {
            Self::AwaitingDraw { seat } | Self::AwaitingDiscard { seat } => Some(seat),
            Self::RobberyWindow { actor } => Some(actor),
            Self::Terminal | Self::ReactionWindow { .. } => None,
        }
    }
}

/// Why a transition failed.
///
/// [`Self::WallExhausted`] is a normal situation (end the hand with an exhaustive
/// draw); the others are API misuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TransitionError {
    /// Live wall exhausted. Not an error: signals an exhaustive draw.
    #[error("live wall exhausted (proceed to exhaustive draw settlement)")]
    WallExhausted,
    #[error("the hand has ended")]
    RoundFinished,
    #[error("current phase is {actual}, operation requires {expected}")]
    WrongPhase {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("seat {seat} out of range ({seats} seats)")]
    SeatOutOfRange { seat: u8, seats: u8 },
}

impl TransitionError {
    /// Stable error code. Branch on this rather than on error messages.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::WallExhausted => "WALL_EXHAUSTED",
            Self::RoundFinished => "ROUND_FINISHED",
            Self::WrongPhase { .. } => "WRONG_PHASE",
            Self::SeatOutOfRange { .. } => "SEAT_OUT_OF_RANGE",
        }
    }

    /// Whether this is a normal situation rather than a caller error.
    #[must_use]
    pub const fn is_normal_outcome(self) -> bool {
        matches!(self, Self::WallExhausted)
    }
}
