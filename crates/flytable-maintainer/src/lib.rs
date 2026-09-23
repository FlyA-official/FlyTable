//! `flytable-maintainer`: table state maintenance for FlyTable.
//!
//! Per-seat views ([`view`]), read-only calculators ([`calc`]: shanten, safe tiles,
//! yaku outlook), legal actions ([`legal`]), the table mirror ([`product`]), and
//! seat state, tile set, action and scoring primitives.
//!
//! No game progression: wall, dealing, turn flow, adjudication and settlement live
//! in `flytable-table`. This crate only reads, computes and mirrors, so clients can
//! depend on it alone.

pub mod action;
pub mod calc;
pub mod legal;
pub mod player;
pub mod product;
pub mod scoring;
pub mod tileset;
pub mod view;

pub use action::{DecisionKind, PendingRobbery, ReactionAction, RobberyKind, TurnAction};
pub use legal::{legal_turn_actions, permanent_furiten};
pub use player::PlayerState;
pub use product::{
    ActionReconciliation, ActionSetRelation, Mirror3p, Mirror4p, MirrorCenter, MirrorIssue,
    MirrorQuality, reconcile_action_sets,
};
pub use view::{OpponentView, SeatView, SeatViewError, SelfView};
