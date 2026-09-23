//! 4-player event stream (mjai style).
//!
//! Seat arrays are `[T; 4]`, actors `0..=3`, and chi exists. 3-player events live in
//! [`crate::event3p`] with a different shape. Part of the versioned protocol shared
//! with external engines.

use flytable_core::tile::Tile;
use serde::{Deserialize, Serialize};

use crate::scoring::HoraScoring;

/// 4-player seat index (`0..=3`).
pub type Actor4 = u8;

/// 4-player event.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event4p {
    #[default]
    None,

    StartGame {
        #[serde(default)]
        names: [String; 4],
        /// `(nonce, key)` seed for a reproducible wall.
        seed: Option<(u64, u64)>,
    },
    StartKyoku {
        /// Round wind.
        bakaze: Tile,
        /// First dora indicator.
        dora_marker: Tile,
        /// Hand number, 1-based (East 1 = 1).
        kyoku: u8,
        honba: u8,
        kyotaku: u8,
        /// Dealer seat.
        oya: Actor4,
        scores: [i32; 4],
        /// Starting hands, 13 tiles each (hidden seats show unknown tiles in views).
        tehais: [[Tile; 13]; 4],
    },

    Tsumo {
        actor: Actor4,
        pai: Tile,
    },
    /// The dealer's 14th starting tile (Mahjong Soul). It joins the concealed hand but
    /// does not become `drawn_tile`, since the platform has no tsumogiri/tedashi
    /// distinction before the first discard.
    DealerOpening {
        actor: Actor4,
        pai: Tile,
    },
    Dahai {
        actor: Actor4,
        pai: Tile,
        /// Tsumogiri (`true`) or from hand (`false`).
        tsumogiri: bool,
    },
    /// The dealer's first discard (Mahjong Soul); the tile has no drawn/hand origin.
    DealerOpeningDahai {
        actor: Actor4,
        pai: Tile,
    },

    /// The seat enters forced autoplay (disconnect semantics).
    ///
    /// This is a behavioral guarantee, not a cause. From this event until
    /// [`Event4p::SeatResumed`], the seat:
    ///
    /// - always discards the tile it just drew;
    /// - makes no calls (chi, pon, kan);
    /// - does not ron;
    /// - does not score nagashi mangan.
    ///
    /// Only platforms that guarantee all of these may emit it. When the behavior is
    /// unknown (for example an auto-play mode with undisclosed logic), do not emit it;
    /// models would otherwise learn from a guarantee that does not hold.
    ///
    /// The match log records the platform fact separately as
    /// `MatchlogEvent::PlatformDisconnect`; this event is the behavioral constraint
    /// derived from it.
    SeatForcedAutoplay {
        actor: Actor4,
    },
    /// The seat leaves forced autoplay and resumes normal decisions.
    SeatResumed {
        actor: Actor4,
    },

    Chi {
        actor: Actor4,
        target: Actor4,
        pai: Tile,
        consumed: [Tile; 2],
    },
    Pon {
        actor: Actor4,
        target: Actor4,
        pai: Tile,
        consumed: [Tile; 2],
    },
    Daiminkan {
        actor: Actor4,
        target: Actor4,
        pai: Tile,
        consumed: [Tile; 3],
    },
    Kakan {
        actor: Actor4,
        pai: Tile,
        consumed: [Tile; 3],
    },
    Ankan {
        actor: Actor4,
        consumed: [Tile; 4],
    },
    Dora {
        dora_marker: Tile,
    },

    Reach {
        actor: Actor4,
    },
    ReachAccepted {
        actor: Actor4,
    },

    Hora {
        actor: Actor4,
        target: Actor4,
        deltas: Option<[i32; 4]>,
        ura_markers: Option<Vec<Tile>>,
        scoring: Option<HoraScoring>,
    },
    Ryukyoku {
        deltas: Option<[i32; 4]>,
    },

    EndKyoku,
    EndGame,
}

impl Event4p {
    /// Seat that performed the event, if any.
    pub fn actor(&self) -> Option<Actor4> {
        match *self {
            Event4p::Tsumo { actor, .. }
            | Event4p::DealerOpening { actor, .. }
            | Event4p::Dahai { actor, .. }
            | Event4p::DealerOpeningDahai { actor, .. }
            | Event4p::Chi { actor, .. }
            | Event4p::Pon { actor, .. }
            | Event4p::Daiminkan { actor, .. }
            | Event4p::Kakan { actor, .. }
            | Event4p::Ankan { actor, .. }
            | Event4p::Reach { actor }
            | Event4p::ReachAccepted { actor }
            | Event4p::Hora { actor, .. } => Some(actor),
            _ => None,
        }
    }
}
