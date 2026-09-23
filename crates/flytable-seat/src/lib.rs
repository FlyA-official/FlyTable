//! `flytable-seat`: seat decision interface, inference request gate (stub) and the
//! built-in tsumogiri seat.
//!
//! - [`SeatDecider`] - a seat that turns a view into an action. Humans, scripts, LLMs
//!   and model seats all plug in through it, and so does CLI self-play.
//! - [`InferenceGate`] - the gate that sends a standard event stream and receives an
//!   action index when a model decision is needed. [`StubGate`] is the default.
//! - [`TsumogiriDecider`] - built-in tsumogiri seat (disconnect autoplay: only
//!   tsumogiri, no calls, no wins).
//!
//! No observation encoding here. `InferenceGate` only passes the standard event
//! stream and an action index; encoding events into tensors is always on the model
//! side.

use flytable_event::{Event3p, Event4p};
use flytable_table::{ReactionAction, SeatView, TurnAction};

pub mod tsumogiri;

/// The `flya-inference-v2` contract types (with frozen v1) live in `flytable-protocol`
/// and are re-exported to keep the `flytable_seat::*` and `flytable_seat::contract::*` paths.
pub use flytable_protocol as contract;
pub use flytable_protocol::*;
pub use tsumogiri::TsumogiriDecider;

/// Seat decision interface: take a view, return an action.
///
/// Shared by 4-player and 3-player: [`SeatView`] / [`TurnAction`] / [`ReactionAction`]
/// do not depend on seat count (`others` holds the opponents; `Chi` is 4-player only
/// and `Nukidora` 3-player only, enforced by the table core).
///
/// Deciders only get an imperfect-information view and never see other hands. For
/// responses the table core passes the legal set `legal`, and the decider picks from
/// it (or passes).
pub trait SeatDecider {
    /// Action on the seat's own turn after drawing.
    fn decide_turn(&mut self, view: &SeatView) -> TurnAction;

    /// Same, but the host also passes the authoritative legal set it has already computed.
    ///
    /// The host always enumerates legal actions before calling the agent (for fallback
    /// and validation). Enumerating again here would compute `legal_turn_actions` twice
    /// per turn, the most expensive step on the hot path (shanten for every candidate
    /// discard).
    ///
    /// The default forwards to [`SeatDecider::decide_turn`], so existing implementations
    /// are unaffected; override this to skip the second enumeration.
    fn decide_turn_with_legal(&mut self, view: &SeatView, legal: &[TurnAction]) -> TurnAction {
        let _ = legal;
        self.decide_turn(view)
    }

    /// Picks a response after another player's discard (defaults to Pass).
    fn decide_reaction(&mut self, view: &SeatView, legal: &[ReactionAction]) -> ReactionAction {
        let _ = (view, legal);
        ReactionAction::Pass
    }
}

/// Inference request gate: send the standard event stream, get an action index back.
///
/// The shape deliberately exposes only "events out, index back", the two contracts
/// of the event protocol. 4-player and 3-player each have an entry point, since
/// their event streams differ (chi in 4-player, nukidora in 3-player).
///
/// The index points into the legal action list FlyTable enumerates from the rules,
/// not into a model's fixed action space. Enumerating legal actions is a rules
/// computation. When a model is connected, a boundary adapter maps the model's
/// action space onto this list, so action space knowledge stays in the adapter or
/// model and never enters the FlyTable core.
///
/// `None` means no opinion (the caller falls back to rule-based decisions).
pub trait InferenceGate {
    /// 4-player: index into the legal action list for `seat`, given the events so far.
    fn infer_4p(&mut self, events: &[Event4p], seat: u8) -> Option<usize>;

    /// 3-player: same, for the 3-player event stream.
    fn infer_3p(&mut self, events: &[Event3p], seat: u8) -> Option<usize>;
}

/// Stub gate that always has no opinion, so the gate works without any model or network dependency.
#[derive(Debug, Default, Clone)]
pub struct StubGate;

impl InferenceGate for StubGate {
    fn infer_4p(&mut self, _events: &[Event4p], _seat: u8) -> Option<usize> {
        None
    }
    fn infer_3p(&mut self, _events: &[Event3p], _seat: u8) -> Option<usize> {
        None
    }
}
