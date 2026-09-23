//! 3-player event stream.
//!
//! - Seat arrays are `[T; 3]`, actors `0..=2`.
//! - No chi.
//! - Nukidora: a North set aside as dora.
//! - The 108-tile wall has only 1m and 9m in manzu. This is enforced by the wall and
//!   rule layers, not by the event types.
//!
//! Scores are `[i32; 3]`. When the mjai 4-entry form is needed, [`crate::compat3p`]
//! converts at the boundary.

use flytable_core::tile::Tile;
use serde::{Deserialize, Serialize};

use crate::scoring::HoraScoring;

/// 3-player seat index (`0..=2`).
pub type Actor3 = u8;

/// 3-player event.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event3p {
    #[default]
    None,

    StartGame {
        #[serde(default)]
        names: [String; 3],
        seed: Option<(u64, u64)>,
    },
    StartKyoku {
        bakaze: Tile,
        dora_marker: Tile,
        /// Hand number, 1-based (East 1 = 1).
        kyoku: u8,
        honba: u8,
        kyotaku: u8,
        oya: Actor3,
        scores: [i32; 3],
        tehais: [[Tile; 13]; 3],
    },

    Tsumo {
        actor: Actor3,
        pai: Tile,
    },
    /// The dealer's 14th starting tile (Mahjong Soul). Joins the concealed hand without
    /// setting `drawn_tile`.
    DealerOpening {
        actor: Actor3,
        pai: Tile,
    },
    Dahai {
        actor: Actor3,
        pai: Tile,
        tsumogiri: bool,
    },
    /// The dealer's first discard (Mahjong Soul); the tile has no drawn/hand origin.
    DealerOpeningDahai {
        actor: Actor3,
        pai: Tile,
    },

    /// The seat enters forced autoplay (disconnect semantics).
    ///
    /// Same guarantees as 4-player [`crate::event4p::Event4p::SeatForcedAutoplay`]:
    /// forced tsumogiri, no calls, no ron, no nagashi mangan. Only platforms that
    /// guarantee all of these may emit it.
    SeatForcedAutoplay {
        actor: Actor3,
    },
    /// The seat leaves forced autoplay and resumes normal decisions.
    SeatResumed {
        actor: Actor3,
    },

    // No chi in 3-player.
    Pon {
        actor: Actor3,
        target: Actor3,
        pai: Tile,
        consumed: [Tile; 2],
    },
    Daiminkan {
        actor: Actor3,
        target: Actor3,
        pai: Tile,
        consumed: [Tile; 3],
    },
    Kakan {
        actor: Actor3,
        pai: Tile,
        consumed: [Tile; 3],
    },
    Ankan {
        actor: Actor3,
        consumed: [Tile; 4],
    },
    /// Nukidora: set aside a North from hand as dora. 3-player only.
    Nukidora {
        actor: Actor3,
        pai: Tile,
    },
    Dora {
        dora_marker: Tile,
    },

    Reach {
        actor: Actor3,
    },
    ReachAccepted {
        actor: Actor3,
    },

    Hora {
        actor: Actor3,
        target: Actor3,
        deltas: Option<[i32; 3]>,
        ura_markers: Option<Vec<Tile>>,
        scoring: Option<HoraScoring>,
    },
    Ryukyoku {
        deltas: Option<[i32; 3]>,
    },

    EndKyoku,
    EndGame,
}

impl Event3p {
    /// Seat that performed the event, if any.
    pub fn actor(&self) -> Option<Actor3> {
        match *self {
            Event3p::Tsumo { actor, .. }
            | Event3p::DealerOpening { actor, .. }
            | Event3p::Dahai { actor, .. }
            | Event3p::DealerOpeningDahai { actor, .. }
            | Event3p::Pon { actor, .. }
            | Event3p::Daiminkan { actor, .. }
            | Event3p::Kakan { actor, .. }
            | Event3p::Ankan { actor, .. }
            | Event3p::Nukidora { actor, .. }
            | Event3p::Reach { actor }
            | Event3p::ReachAccepted { actor }
            | Event3p::Hora { actor, .. } => Some(actor),
            _ => None,
        }
    }
}
