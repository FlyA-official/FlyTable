//! `flytable-table`: the FlyTable game core.
//!
//! Holds the true game state (wall, private hands, progression and adjudication) and
//! projects an imperfect-information view for any seat (see [`view`]).
//!
//! 4-player and 3-player each have their own board ([`board4p`] / `board3p`) and share
//! wall geometry ([`wall`]), tile sets ([`tileset`]), seat state ([`player`]) and the
//! settlement bridge ([`scoring`]).

pub mod board3p;
pub mod board4p;
pub mod progress;
pub mod settle;
pub mod wall;

/// Authoritative result for one winner. Multiple ron consists of independent results.
#[derive(Debug, Clone)]
pub struct HoraResult {
    pub winner: u8,
    pub from: Option<u8>,
    pub score: flytable_core::score::FullScore,
}

// The table-state modules live in `flytable-maintainer` so clients can depend on it
// alone. They are re-exported here to keep the existing paths.
//
// Items are listed explicitly rather than with a glob, so new public items in
// `flytable-maintainer` do not silently become part of this crate's API.
pub use flytable_maintainer::{
    ActionReconciliation, ActionSetRelation, DecisionKind, Mirror3p, Mirror4p, MirrorCenter,
    MirrorIssue, MirrorQuality, OpponentView, PendingRobbery, PlayerState, ReactionAction,
    RobberyKind, SeatView, SeatViewError, SelfView, TurnAction, legal_turn_actions,
    permanent_furiten, reconcile_action_sets,
};
pub use flytable_maintainer::{action, calc, legal, player, product, scoring, tileset, view};

pub mod matchlog_settlement;
pub mod phase;
pub mod timing;

pub use board3p::{AbortiveRyukyokuReason3p, Board3p, KyokuOutcome3p};
pub use board4p::{AbortiveRyukyokuReason4p, Board4p, KyokuOutcome, Prompt};
pub use phase::{BoardPhase, TransitionError};
pub use progress::{
    HoraPayment, MatchLength, RoundState, Settlement, SettlementSplit, split_settlement,
};
