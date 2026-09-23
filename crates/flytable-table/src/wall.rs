//! The wall.
//!
//! The wall structure does not depend on seat count: it is a tile list plus dealing
//! geometry. The variant supplies which tiles and how many (136 with red fives for
//! 4-player, 108 without 2m..8m for 3-player).
//!
//! Geometry follows standard riichi practice: after shuffling, hands are dealt from
//! the front, followed by the live wall; the last 14 tiles are the dead wall with 4
//! replacement tiles and the dora and ura dora indicators.

use flytable_core::rules::{RedFiveCounts, RiichiRuleProfile};
use flytable_core::tile::{Suit, Tile};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;

/// Dead wall size.
pub const DEAD_WALL_LEN: usize = 14;
/// Replacement tiles for kans.
pub const RINSHAN_LEN: usize = 4;
/// At most 4 nukidora in 3-player; together with up to 4 kans this makes the 8
/// replacement tiles of Tenhou's published rules.
pub const SANMA_NUKI_SUPPLEMENT_LEN: usize = 4;
/// At most 5 dora indicators (1 at the start plus 1 per kan).
pub const MAX_DORA_INDICATORS: usize = crate::tileset::MAX_DORA_INDICATORS;

/// A shuffled wall.
#[derive(Debug, Clone)]
pub struct Wall {
    /// All tiles, shuffled. `[0..haipai_total)` is the deal, then the live wall, then the 14-tile dead wall.
    tiles: Vec<Tile>,
    /// Total tiles dealt (seats x 13). Kept to describe the geometry.
    #[allow(dead_code)]
    haipai_total: usize,
    /// Index of the next live wall tile.
    live_cursor: usize,
    /// End of the live wall (start of the dead wall).
    live_end: usize,
    /// Kans so far (selects the next replacement tile and indicator).
    kans_drawn: usize,
    /// Kan dora actually revealed. Closed kans reveal at once; the board confirms open and added kans later.
    dora_revealed: usize,
    /// Tiles used for 3-player nukidora replacement draws.
    supplements_drawn: usize,
}

impl Wall {
    /// Builds a wall by shuffling the given tile set with a seed.
    pub fn shuffled(mut tile_set: Vec<Tile>, seats: usize, seed: (u64, u64)) -> Self {
        validate_physical_tileset(&tile_set, seats)
            .expect("tile set must form a valid physical wall for the seat count");
        let mut rng = seed_rng(seed);
        tile_set.shuffle(&mut rng);
        Self::build(tile_set, seats).expect("a validated tile set must form a valid wall")
    }

    /// Builds a wall from the default standard tile set in a fixed order (no shuffle).
    ///
    /// Explicit rule profiles, including rooms with no or extra red fives, must use
    /// [`Self::from_ordered_for_profile`] instead.
    pub fn from_ordered(tiles: Vec<Tile>, seats: usize) -> Result<Self, String> {
        let expected = match seats {
            4 => crate::tileset::tileset_4p(),
            3 => crate::tileset::tileset_3p(),
            _ => return Err(format!("the wall supports 3 or 4 seats, got {seats}")),
        };
        Self::from_ordered_against(tiles, seats, expected)
    }

    /// Builds a wall from an ordered tile set that matches the rule profile exactly.
    pub fn from_ordered_for_profile(
        tiles: Vec<Tile>,
        seats: usize,
        profile: RiichiRuleProfile,
    ) -> Result<Self, String> {
        profile.validate_for_players(seats)?;
        let expected = match seats {
            4 => crate::tileset::tileset_4p_with_red_fives(profile.red_fives(4))?,
            3 => crate::tileset::tileset_3p_with_red_fives(profile.red_fives(3))?,
            _ => return Err(format!("the wall supports 3 or 4 seats, got {seats}")),
        };
        Self::from_ordered_against(tiles, seats, expected)
    }

    fn from_ordered_against(
        tiles: Vec<Tile>,
        seats: usize,
        mut expected: Vec<Tile>,
    ) -> Result<Self, String> {
        if tiles.len() != expected.len() {
            return Err(format!(
                "{seats}p wall must have {} tiles, got {}",
                expected.len(),
                tiles.len()
            ));
        }
        let mut actual = tiles.clone();
        actual.sort_by_key(|tile| tile.id());
        expected.sort_by_key(|tile| tile.id());
        if actual != expected {
            return Err(format!("{seats}p wall has an invalid tile multiset"));
        }
        Self::build(tiles, seats)
    }

    fn build(tiles: Vec<Tile>, seats: usize) -> Result<Self, String> {
        if !matches!(seats, 3 | 4) {
            return Err(format!("the wall supports 3 or 4 seats, got {seats}"));
        }
        let haipai_total = seats * 13;
        let total = tiles.len();
        if total < haipai_total + DEAD_WALL_LEN {
            return Err(format!(
                "{seats}p wall is too small for the deal and dead wall"
            ));
        }
        let live_end = total - DEAD_WALL_LEN;
        Ok(Self {
            tiles,
            haipai_total,
            live_cursor: haipai_total,
            live_end,
            kans_drawn: 0,
            dora_revealed: 0,
            supplements_drawn: 0,
        })
    }

    /// Red five counts per suit in this wall.
    pub fn red_five_counts(&self) -> RedFiveCounts {
        let mut counts = RedFiveCounts {
            man: 0,
            pin: 0,
            sou: 0,
        };
        for tile in self.tiles.iter().copied().filter(|tile| tile.is_aka()) {
            match tile.suit() {
                Suit::Man => counts.man += 1,
                Suit::Pin => counts.pin += 1,
                Suit::Sou => counts.sou += 1,
                Suit::Honor => unreachable!("honors have no red tiles"),
            }
        }
        counts
    }

    /// The 13 dealt tiles of a seat (`0..seats`).
    pub fn haipai(&self, seat: usize) -> [Tile; 13] {
        let start = seat * 13;
        let mut h = [Tile::default(); 13];
        h.copy_from_slice(&self.tiles[start..start + 13]);
        h
    }

    /// Tiles left in the live wall.
    pub fn live_remaining(&self) -> usize {
        self.live_end.saturating_sub(self.live_cursor)
    }

    /// Replacement tiles left.
    pub fn rinshan_remaining(&self) -> usize {
        RINSHAN_LEN.saturating_sub(self.kans_drawn)
    }

    /// Nukidora replacement tiles left in 3-player; up to 8 replacement tiles together with the 4 kan tiles.
    pub fn supplement_remaining(&self) -> usize {
        SANMA_NUKI_SUPPLEMENT_LEN.saturating_sub(self.supplements_drawn)
    }

    /// Draws from the live wall (`None` means an exhaustive draw).
    pub fn draw(&mut self) -> Option<Tile> {
        if self.live_cursor >= self.live_end {
            return None;
        }
        let t = self.tiles[self.live_cursor];
        self.live_cursor += 1;
        Some(t)
    }

    /// Replacement draw after a kan, counted from the end of the dead wall. Each kan
    /// shortens the live wall by one.
    pub fn draw_rinshan(&mut self) -> Option<Tile> {
        if self.kans_drawn >= RINSHAN_LEN {
            return None;
        }
        // Replacement tiles are the last 4 of the dead wall; the live wall end moves forward by one.
        let idx = self.tiles.len() - 1 - self.kans_drawn;
        self.kans_drawn += 1;
        if self.live_end > self.live_cursor {
            self.live_end -= 1;
        }
        Some(self.tiles[idx])
    }

    /// Reveals a kan dora indicator that a kan has created but not yet revealed.
    pub fn reveal_kan_dora(&mut self) -> Option<Tile> {
        if self.dora_revealed >= self.kans_drawn || self.dora_revealed + 1 >= MAX_DORA_INDICATORS {
            return None;
        }
        self.dora_revealed += 1;
        self.dora_indicators().last().copied()
    }

    /// Supplementary draw (3-player nukidora). Does not use the 4 kan replacement tiles
    /// or reveal a new indicator.
    pub fn draw_supplement(&mut self) -> Option<Tile> {
        if self.live_cursor >= self.live_end || self.supplements_drawn >= SANMA_NUKI_SUPPLEMENT_LEN
        {
            return None;
        }
        self.live_end -= 1;
        self.supplements_drawn += 1;
        Some(self.tiles[self.live_end])
    }

    /// Visible dora indicators (1 at the start plus 1 per kan).
    pub fn dora_indicators(&self) -> Vec<Tile> {
        // Indicators sit in the dead wall just before the replacement tiles, at fixed indices.
        let base = self.tiles.len() - RINSHAN_LEN - 1;
        let n = (1 + self.dora_revealed).min(MAX_DORA_INDICATORS);
        (0..n).map(|i| self.tiles[base - i]).collect()
    }

    /// Ura dora indicators (revealed on a riichi win), as many as the dora indicators.
    pub fn ura_indicators(&self) -> Vec<Tile> {
        let base = self.tiles.len() - RINSHAN_LEN - 1 - MAX_DORA_INDICATORS;
        let n = (1 + self.dora_revealed).min(MAX_DORA_INDICATORS);
        (0..n).map(|i| self.tiles[base - i]).collect()
    }
}

fn validate_physical_tileset(tiles: &[Tile], seats: usize) -> Result<(), String> {
    let standard = match seats {
        4 => crate::tileset::tileset_4p(),
        3 => crate::tileset::tileset_3p(),
        _ => return Err(format!("the wall supports 3 or 4 seats, got {seats}")),
    };
    if tiles.len() != standard.len() {
        return Err(format!(
            "{seats}p wall must have {} tiles, got {}",
            standard.len(),
            tiles.len()
        ));
    }
    let mut actual_kinds: Vec<_> = tiles.iter().map(|tile| tile.kind()).collect();
    let mut expected_kinds: Vec<_> = standard.iter().map(|tile| tile.kind()).collect();
    actual_kinds.sort_unstable();
    expected_kinds.sort_unstable();
    if actual_kinds != expected_kinds {
        return Err(format!("{seats}p wall has an invalid base tile multiset"));
    }
    Ok(())
}

/// Reproducible RNG from `(nonce, key)`.
fn seed_rng(seed: (u64, u64)) -> ChaCha12Rng {
    let mut bytes = [0u8; 32];
    bytes[0..8].copy_from_slice(&seed.0.to_le_bytes());
    bytes[8..16].copy_from_slice(&seed.1.to_le_bytes());
    ChaCha12Rng::from_seed(bytes)
}
