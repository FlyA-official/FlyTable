//! [`Variant`]: puts the 4-player / 3-player differences behind one trait so
//! [`crate::SeatAgent`] and [`crate::MatchHost`] are written once.
//!
//! There are only two kinds of difference: the concrete event and legal action types
//! (`Event4p` vs `Event3p`, `LegalAction4p` vs `LegalAction3p`), and whether the host
//! is called through `infer_4p` or `infer_3p`. Progression is still driven explicitly
//! by `Board4p` / `Board3p` in [`crate::match_host`].
//!
//! The `TurnAction` / `ReactionAction` to `LegalAction*` conversion matches the
//! certification smoke test in `flytable-inference-host` (`registry.rs`): the host
//! matches fully qualified actions back to indices exactly.

use flytable_core::meld::Meld;
use flytable_core::tile::Tile;
use flytable_event::{Event3p, Event4p};
use flytable_seat::contract::{
    project_3p, project_4p, InferenceDecision, InferenceError, KanKind, LegalAction3p,
    LegalAction4p, ResponseOpportunities, RuleLine, VisibleEvent3p, VisibleEvent4p,
};
use flytable_table::{ReactionAction, SeatView, TurnAction};

use flytable_inference_host::host::{
    DecisionPhase, InferenceInput3p, InferenceInput4p, RemoteHttpHost, SourceInfo, SubprocessHost,
};

/// Every variant difference the shared interfaces need.
pub trait Variant {
    /// Event type (`Event4p` / `Event3p`). Authoritative full events, including other
    /// players' hands and draws; only used inside the runtime (see the visibility
    /// boundary in [`crate::agent::TurnRequest`]).
    type Event: Clone;
    /// Per-seat projected event type (`VisibleEvent4p` / `VisibleEvent3p`), the only form seat agents receive.
    type VisibleEvent: Clone;
    /// Legal action wire type (`LegalAction4p` / `LegalAction3p`).
    type LegalAction: Clone + PartialEq + std::fmt::Debug;

    /// Number of seats.
    const SEATS: u8;
    /// Rule line (for host hello and digests).
    const RULE_LINE: RuleLine;

    /// The `StartGame` event. `Board::log` does not include it, so it is prepended before
    /// calling the host (as in `registry::events_with_start_game_*`).
    fn start_game_event() -> Self::Event;

    /// Projects an authoritative event for a seat (other hands and draws become hidden placeholders).
    fn project_event(event: &Self::Event, seat: u8) -> Self::VisibleEvent;

    /// Turn action to a fully qualified legal action (for exact index matching by the host).
    fn turn_action_to_legal(view: &SeatView, action: &TurnAction) -> Self::LegalAction;

    /// Response to a fully qualified legal action. `Pass` returns `None` (handled by [`Variant::pass_all`]).
    fn reaction_to_legal(
        view: &SeatView,
        action: &ReactionAction,
        discarder: u8,
        tile: Tile,
    ) -> Option<Self::LegalAction>;

    /// The pass action for this response window (`pass_all`). `ron` / `call` record which opportunities were declined (metadata only).
    fn pass_all(ron: bool, call: bool) -> Self::LegalAction;

    /// Legal action to canonical action (for L3 decision windows; conversion differs between 3P and 4P).
    fn legal_to_canonical(action: Self::LegalAction) -> flytable_seat::contract::CanonicalAction;

    /// Calls the host's `infer_*`.
    fn host_infer(
        host: &mut SubprocessHost,
        decision_id: String,
        phase: DecisionPhase,
        events: &[Self::Event],
        legal_actions: &[Self::LegalAction],
        wall_remaining: u32,
    ) -> Result<InferenceDecision, InferenceError>;

    /// Calls the remote HTTP host's `infer_*`.
    fn remote_infer(
        host: &RemoteHttpHost,
        decision_id: String,
        phase: DecisionPhase,
        events: &[Self::Event],
        legal_actions: &[Self::LegalAction],
        wall_remaining: u32,
    ) -> Result<InferenceDecision, InferenceError>;
}

/// 4-player riichi mahjong.
pub struct Variant4p;
/// 3-player riichi mahjong (with nukidora).
pub struct Variant3p;

impl Variant for Variant4p {
    type Event = Event4p;
    type VisibleEvent = VisibleEvent4p;
    type LegalAction = LegalAction4p;
    const SEATS: u8 = 4;
    const RULE_LINE: RuleLine = RuleLine::Riichi4p;

    fn start_game_event() -> Event4p {
        Event4p::StartGame {
            names: ["a", "b", "c", "d"].map(str::to_owned),
            seed: None,
        }
    }

    fn project_event(event: &Event4p, seat: u8) -> VisibleEvent4p {
        project_4p(event, seat)
    }

    fn turn_action_to_legal(view: &SeatView, action: &TurnAction) -> LegalAction4p {
        match action {
            TurnAction::Discard { tile, tsumogiri } => LegalAction4p::Discard {
                pai: *tile,
                tsumogiri: *tsumogiri,
                riichi: false,
            },
            TurnAction::DealerOpeningDiscard { tile } => LegalAction4p::DealerOpeningDiscard {
                pai: *tile,
                riichi: false,
            },
            TurnAction::Riichi { tile, tsumogiri } => LegalAction4p::Discard {
                pai: *tile,
                tsumogiri: *tsumogiri,
                riichi: true,
            },
            TurnAction::DealerOpeningRiichi { tile } => LegalAction4p::DealerOpeningDiscard {
                pai: *tile,
                riichi: true,
            },
            TurnAction::Ankan { tile } => LegalAction4p::Kan {
                pai: *tile,
                kind: KanKind::Ankan,
                consumed: same_kind_tiles(&view.me.hand, *tile, 4),
            },
            TurnAction::Kakan { tile } => LegalAction4p::Kan {
                pai: *tile,
                kind: KanKind::Kakan,
                consumed: kakan_consumed(&view.me.melds, *tile),
            },
            TurnAction::Tsumo if view.me.dealer_opening => LegalAction4p::DealerOpeningTsumo,
            TurnAction::Tsumo => LegalAction4p::Tsumo {
                pai: view.me.drawn_tile.expect("tsumo requires a drawn tile"),
            },
            TurnAction::KyuushuKyuuhai => LegalAction4p::Kyushukyuhai,
            TurnAction::Nukidora => unreachable!("4p has no nukidora"),
        }
    }

    fn reaction_to_legal(
        view: &SeatView,
        action: &ReactionAction,
        discarder: u8,
        tile: Tile,
    ) -> Option<LegalAction4p> {
        match action {
            ReactionAction::Pass => None,
            ReactionAction::Pon { consumed } => Some(LegalAction4p::Pon {
                pai: tile,
                consumed: *consumed,
            }),
            ReactionAction::Chi { consumed } => Some(LegalAction4p::Chi {
                pai: tile,
                consumed: *consumed,
            }),
            ReactionAction::Daiminkan => Some(LegalAction4p::Kan {
                pai: tile,
                kind: KanKind::Daiminkan,
                consumed: same_kind_tiles(&view.me.hand, tile, 3),
            }),
            ReactionAction::Ron => Some(LegalAction4p::Ron {
                pai: tile,
                target: discarder,
            }),
        }
    }

    fn pass_all(ron: bool, call: bool) -> LegalAction4p {
        LegalAction4p::PassAll {
            declines: ResponseOpportunities { ron, call },
        }
    }

    fn legal_to_canonical(action: LegalAction4p) -> flytable_seat::contract::CanonicalAction {
        flytable_seat::contract::CanonicalAction::from_legal_4p(&action)
    }

    fn host_infer(
        host: &mut SubprocessHost,
        decision_id: String,
        phase: DecisionPhase,
        events: &[Event4p],
        legal_actions: &[LegalAction4p],
        wall_remaining: u32,
    ) -> Result<InferenceDecision, InferenceError> {
        host.infer_4p(InferenceInput4p {
            decision_id,
            phase,
            events,
            legal_actions,
            wall_remaining,
            source: SourceInfo::authoritative(),
            remote_auth: None,
            seat: None,
        })
    }

    fn remote_infer(
        host: &RemoteHttpHost,
        decision_id: String,
        phase: DecisionPhase,
        events: &[Event4p],
        legal_actions: &[LegalAction4p],
        wall_remaining: u32,
    ) -> Result<InferenceDecision, InferenceError> {
        host.infer_4p(InferenceInput4p {
            decision_id,
            phase,
            events,
            legal_actions,
            wall_remaining,
            source: SourceInfo::authoritative(),
            remote_auth: None,
            seat: None,
        })
    }
}

impl Variant for Variant3p {
    type Event = Event3p;
    type VisibleEvent = VisibleEvent3p;
    type LegalAction = LegalAction3p;
    const SEATS: u8 = 3;
    const RULE_LINE: RuleLine = RuleLine::Riichi3p;

    fn start_game_event() -> Event3p {
        Event3p::StartGame {
            names: ["a", "b", "c"].map(str::to_owned),
            seed: None,
        }
    }

    fn project_event(event: &Event3p, seat: u8) -> VisibleEvent3p {
        project_3p(event, seat)
    }

    fn turn_action_to_legal(view: &SeatView, action: &TurnAction) -> LegalAction3p {
        match action {
            TurnAction::Discard { tile, tsumogiri } => LegalAction3p::Discard {
                pai: *tile,
                tsumogiri: *tsumogiri,
                riichi: false,
            },
            TurnAction::DealerOpeningDiscard { tile } => LegalAction3p::DealerOpeningDiscard {
                pai: *tile,
                riichi: false,
            },
            TurnAction::Riichi { tile, tsumogiri } => LegalAction3p::Discard {
                pai: *tile,
                tsumogiri: *tsumogiri,
                riichi: true,
            },
            TurnAction::DealerOpeningRiichi { tile } => LegalAction3p::DealerOpeningDiscard {
                pai: *tile,
                riichi: true,
            },
            TurnAction::Ankan { tile } => LegalAction3p::Kan {
                pai: *tile,
                kind: KanKind::Ankan,
                consumed: same_kind_tiles(&view.me.hand, *tile, 4),
            },
            TurnAction::Kakan { tile } => LegalAction3p::Kan {
                pai: *tile,
                kind: KanKind::Kakan,
                consumed: kakan_consumed(&view.me.melds, *tile),
            },
            TurnAction::Nukidora => LegalAction3p::Nukidora,
            TurnAction::Tsumo if view.me.dealer_opening => LegalAction3p::DealerOpeningTsumo,
            TurnAction::Tsumo => LegalAction3p::Tsumo {
                pai: view.me.drawn_tile.expect("tsumo requires a drawn tile"),
            },
            TurnAction::KyuushuKyuuhai => LegalAction3p::Kyushukyuhai,
        }
    }

    fn reaction_to_legal(
        view: &SeatView,
        action: &ReactionAction,
        discarder: u8,
        tile: Tile,
    ) -> Option<LegalAction3p> {
        match action {
            ReactionAction::Pass => None,
            ReactionAction::Pon { consumed } => Some(LegalAction3p::Pon {
                pai: tile,
                consumed: *consumed,
            }),
            // No chi in 3-player.
            ReactionAction::Chi { .. } => None,
            ReactionAction::Daiminkan => Some(LegalAction3p::Kan {
                pai: tile,
                kind: KanKind::Daiminkan,
                consumed: same_kind_tiles(&view.me.hand, tile, 3),
            }),
            ReactionAction::Ron => Some(LegalAction3p::Ron {
                pai: tile,
                target: discarder,
            }),
        }
    }

    fn pass_all(ron: bool, call: bool) -> LegalAction3p {
        LegalAction3p::PassAll {
            declines: ResponseOpportunities { ron, call },
        }
    }

    fn legal_to_canonical(action: LegalAction3p) -> flytable_seat::contract::CanonicalAction {
        flytable_seat::contract::CanonicalAction::from_legal_3p(&action)
    }

    fn host_infer(
        host: &mut SubprocessHost,
        decision_id: String,
        phase: DecisionPhase,
        events: &[Event3p],
        legal_actions: &[LegalAction3p],
        wall_remaining: u32,
    ) -> Result<InferenceDecision, InferenceError> {
        host.infer_3p(InferenceInput3p {
            decision_id,
            phase,
            events,
            legal_actions,
            wall_remaining,
            source: SourceInfo::authoritative(),
            remote_auth: None,
            seat: None,
            match_context: None,
        })
    }

    fn remote_infer(
        host: &RemoteHttpHost,
        decision_id: String,
        phase: DecisionPhase,
        events: &[Event3p],
        legal_actions: &[LegalAction3p],
        wall_remaining: u32,
    ) -> Result<InferenceDecision, InferenceError> {
        host.infer_3p(InferenceInput3p {
            decision_id,
            phase,
            events,
            legal_actions,
            wall_remaining,
            source: SourceInfo::authoritative(),
            remote_auth: None,
            seat: None,
            match_context: None,
        })
    }
}

/// The first `n` tiles of the kind in hand (4 for a closed kan, 3 for an open kan).
fn same_kind_tiles(hand: &[Tile], tile: Tile, n: usize) -> Vec<Tile> {
    hand.iter()
        .copied()
        .filter(|t| t.kind() == tile.kind())
        .take(n)
        .collect()
}

/// Consumed tiles of an added kan: the two matching tiles of the pon plus the called tile (as in the registry).
fn kakan_consumed(melds: &[Meld], tile: Tile) -> Vec<Tile> {
    melds
        .iter()
        .find_map(|meld| match meld {
            Meld::Pon {
                tile: base,
                called,
                consumed,
                ..
            } if base.kind() == tile.kind() => Some(vec![consumed[0], consumed[1], *called]),
            _ => None,
        })
        .unwrap_or_default()
}
