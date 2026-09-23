//! Built-in tsumogiri seat [`TsumogiriDecider`]: the autoplay behavior of a
//! disconnected player on online platforms.

use flytable_table::{ReactionAction, SeatView, TurnAction};

use crate::SeatDecider;

/// Tsumogiri seat: always discards the drawn tile; never riichi, tsumo, kan or
/// nukidora; passes on every discard (no ron, chi, pon or kan).
///
/// Hands end by exhaustive draw or another player's win; a tile is discarded every
/// turn, so play never stalls.
#[derive(Debug, Clone, Copy, Default)]
pub struct TsumogiriDecider;

impl SeatDecider for TsumogiriDecider {
    fn decide_turn(&mut self, view: &SeatView) -> TurnAction {
        if view.me.dealer_opening {
            return TurnAction::DealerOpeningDiscard {
                tile: *view.me.hand.last().expect("non-empty hand on turn"),
            };
        }
        match view.me.drawn_tile {
            Some(t) => TurnAction::Discard {
                tile: t,
                tsumogiri: true,
            },
            // This seat never calls, so this is unreachable in normal play; discard the last tile to be safe.
            None => TurnAction::Discard {
                tile: *view.me.hand.last().expect("non-empty hand on turn"),
                tsumogiri: false,
            },
        }
    }

    fn decide_reaction(&mut self, _view: &SeatView, _legal: &[ReactionAction]) -> ReactionAction {
        ReactionAction::Pass
    }
}
