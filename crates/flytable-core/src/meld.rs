//! Meld (call) representation.
//!
//! The enum lists every meld kind, but which ones are legal depends on the variant:
//! `Chi` only exists in 4-player, `Nukidora` only in 3-player.

use crate::tile::Tile;

/// Absolute seat index. Relative positions are resolved by the table layer.
pub type SeatId = u8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Meld {
    /// Chi: a sequence. `tiles` holds all three tiles including the called one; `from`
    /// is the player to the left. 4-player only.
    Chi {
        tiles: [Tile; 3],
        called: Tile,
        from: SeatId,
    },
    /// Pon: a triplet. `consumed` are the two tiles taken from hand.
    Pon {
        tile: Tile,
        called: Tile,
        consumed: [Tile; 2],
        from: SeatId,
    },
    /// Open kan: a discarded tile plus three from hand.
    Daiminkan {
        tile: Tile,
        called: Tile,
        from: SeatId,
    },
    /// Added kan: the fourth tile added to an existing pon.
    Kakan { tile: Tile, added: Tile },
    /// Closed kan: four tiles from hand.
    Ankan { tile: Tile },
    /// Nukidora: a North set aside as dora. 3-player only; not a meld group.
    Nukidora { tile: Tile },
}

impl Meld {
    /// Representative kind of the meld (red fives folded). Nukidora returns North.
    pub fn kind_tile(&self) -> Tile {
        match self {
            Meld::Chi { tiles, .. } => tiles[0].deaka(),
            Meld::Pon { tile, .. }
            | Meld::Daiminkan { tile, .. }
            | Meld::Kakan { tile, .. }
            | Meld::Ankan { tile }
            | Meld::Nukidora { tile } => tile.deaka(),
        }
    }

    /// Only closed kans count as concealed.
    pub fn is_concealed(&self) -> bool {
        matches!(self, Meld::Ankan { .. })
    }

    pub fn is_kan(&self) -> bool {
        matches!(
            self,
            Meld::Daiminkan { .. } | Meld::Kakan { .. } | Meld::Ankan { .. }
        )
    }

    /// Whether the meld breaks a closed hand. Closed kans and nukidora do not.
    pub fn breaks_menzen(&self) -> bool {
        !matches!(self, Meld::Ankan { .. } | Meld::Nukidora { .. })
    }

    /// Number of red fives inside a kan.
    ///
    /// A kan always uses all four copies of its kind, so for a five the count depends
    /// only on the kind and the rule profile. The kan variants store a single
    /// representative tile, so [`Meld::tiles()`] cannot report red fives for them; use
    /// this instead. Returns 0 for non-kan melds (`Chi` and `Pon` keep their physical
    /// tiles).
    pub fn kan_aka_count(&self, reds: crate::rules::RedFiveCounts) -> u8 {
        if !self.is_kan() {
            return 0;
        }
        let t = self.kind_tile();
        if t.rank() != Some(5) {
            return 0;
        }
        match t.suit() {
            crate::tile::Suit::Man => reds.man,
            crate::tile::Suit::Pin => reds.pin,
            crate::tile::Suit::Sou => reds.sou,
            crate::tile::Suit::Honor => 0,
        }
    }

    /// Tiles that make up the meld.
    ///
    /// Kan variants only store a representative tile, so red fives are not reflected
    /// here; use [`Meld::kan_aka_count`] for red dora.
    pub fn tiles(&self) -> Vec<Tile> {
        match self {
            Meld::Chi { tiles, .. } => tiles.to_vec(),
            Meld::Pon {
                called, consumed, ..
            } => vec![consumed[0], consumed[1], *called],
            Meld::Daiminkan { tile, called, .. } => vec![*tile, *tile, *tile, *called],
            Meld::Kakan { tile, added } => vec![*tile, *tile, *tile, *added],
            Meld::Ankan { tile } => vec![*tile, *tile, *tile, *tile],
            Meld::Nukidora { tile } => vec![*tile],
        }
    }
}
