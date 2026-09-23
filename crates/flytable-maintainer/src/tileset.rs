//! Tile sets per variant.
//!
//! The 4-player and 3-player sets are spelled out explicitly rather than deriving
//! 3-player by subtracting from 4-player.
//!
//! Red fives replace plain fives per [`RedFiveCounts`] and do not change the tile
//! count. The default builders use one red five per suit (none in manzu for
//! 3-player); the explicit profile builders can build walls without red fives or
//! with more.

use flytable_core::rules::RedFiveCounts;
use flytable_core::tile::{Suit, Tile};

/// Maximum number of dora / ura indicators revealed in a hand (1 at the start plus up to 4 kans).
///
/// Shared by the wall and the passive mirror so a canonical event stream with a
/// sixth indicator cannot still be reported as `Exact`.
pub const MAX_DORA_INDICATORS: usize = 5;

/// 4-player wall: 136 tiles, 4 of each of the 34 kinds, with one red 5m/5p/5s.
pub fn tileset_4p() -> Vec<Tile> {
    tileset_4p_with_red_fives(RedFiveCounts {
        man: 1,
        pin: 1,
        sou: 1,
    })
    .expect("standard 4p red five counts are valid")
}

/// Builds a 4-player wall with the given red five counts.
pub fn tileset_4p_with_red_fives(red_fives: RedFiveCounts) -> Result<Vec<Tile>, String> {
    red_fives.validate_for_players(4)?;
    let mut v = Vec::with_capacity(136);
    for (suit, red_count) in [
        (Suit::Man, red_fives.man),
        (Suit::Pin, red_fives.pin),
        (Suit::Sou, red_fives.sou),
    ] {
        for n in 1..=9u8 {
            push_with_aka(&mut v, Tile::number(suit, n).unwrap(), 4, red_count);
        }
    }
    for idx in 0..7u8 {
        let t = Tile::honor(idx).unwrap();
        for _ in 0..4 {
            v.push(t);
        }
    }
    debug_assert_eq!(v.len(), 136);
    Ok(v)
}

/// 3-player wall: 108 tiles. Manzu has only 1m and 9m (4 each); pinzu, souzu and
/// honors are complete. Red fives are 5pr and 5sr.
pub fn tileset_3p() -> Vec<Tile> {
    tileset_3p_with_red_fives(RedFiveCounts {
        man: 0,
        pin: 1,
        sou: 1,
    })
    .expect("standard 3p red five counts are valid")
}

/// Builds a 3-player wall with the given red five counts. `man` must be 0; 2m..8m are never added.
pub fn tileset_3p_with_red_fives(red_fives: RedFiveCounts) -> Result<Vec<Tile>, String> {
    red_fives.validate_for_players(3)?;
    let mut v = Vec::with_capacity(108);
    for n in [1u8, 9u8] {
        for _ in 0..4 {
            v.push(Tile::number(Suit::Man, n).unwrap());
        }
    }
    for (suit, red_count) in [(Suit::Pin, red_fives.pin), (Suit::Sou, red_fives.sou)] {
        for n in 1..=9u8 {
            push_with_aka(&mut v, Tile::number(suit, n).unwrap(), 4, red_count);
        }
    }
    for idx in 0..7u8 {
        let t = Tile::honor(idx).unwrap();
        for _ in 0..4 {
            v.push(t);
        }
    }
    debug_assert_eq!(v.len(), 108);
    Ok(v)
}

/// Pushes `count` copies of a tile; for a five, the first `red_count` are red.
fn push_with_aka(v: &mut Vec<Tile>, tile: Tile, count: u8, red_count: u8) {
    let is_five = tile.rank() == Some(5) && !tile.is_honor();
    for i in 0..count {
        if is_five && i < red_count {
            v.push(tile.akaize());
        } else {
            v.push(tile);
        }
    }
}

/// Every pair of hand tiles that forms a sequence with the called tile `called`
/// (suit and ranks compared after folding red fives). Returns the actual hand tiles,
/// red fives included, sorted by kind. Shared by the maintainer and the table core.
pub fn chi_combos(hand: &[Tile], called: Tile) -> Vec<[Tile; 2]> {
    let mut out: Vec<[Tile; 2]> = Vec::new();
    let Some(r) = called.rank() else {
        return out;
    };
    let suit = called.suit();
    let matching = |n: u8| -> Vec<Tile> {
        if !(1..=9).contains(&n) {
            return Vec::new();
        }
        hand.iter()
            .copied()
            .filter(|t| t.suit() == suit && t.rank() == Some(n))
            .collect()
    };
    // `called` as the low, middle or high tile: [n+1, n+2], [n-1, n+1], [n-2, n-1].
    let patterns: [(i8, i8); 3] = [(1, 2), (-1, 1), (-2, -1)];
    for (a, b) in patterns {
        let na = r as i8 + a;
        let nb = r as i8 + b;
        if !(1..=9).contains(&na) || !(1..=9).contains(&nb) {
            continue;
        }
        let left = matching(na as u8);
        let right = matching(nb as u8);
        for &ta in &left {
            for &tb in &right {
                let mut combo = [ta, tb];
                combo.sort_by_key(|t| (t.kind(), t.id()));
                if !out.iter().any(|c| c == &combo) {
                    out.push(combo);
                }
            }
        }
    }
    out
}

/// Every pair of same-kind tiles usable for pon or open kan, including red five
/// choices. Shared by the maintainer and the table core.
pub fn pon_pair_combos(same_kind_tiles: &[Tile]) -> Vec<[Tile; 2]> {
    let mut out = Vec::new();
    for i in 0..same_kind_tiles.len() {
        for j in i + 1..same_kind_tiles.len() {
            let mut combo = [same_kind_tiles[i], same_kind_tiles[j]];
            combo.sort_by_key(|t| t.id());
            if !out.iter().any(|c| c == &combo) {
                out.push(combo);
            }
        }
    }
    out
}

/// Whether at least one tile can still be discarded after a call.
///
/// With kuikae forbidden, chi and pon forbid some tile kinds. If every remaining
/// tile is forbidden, the seat has no legal discard and the game would be stuck.
/// Tenhou does not offer the call in that situation, and common rules treat the call
/// itself as invalid.
///
/// `hand` is the closed hand before the call, `consumed` are the tiles used by the
/// call, and `forbidden` are the kinds forbidden afterwards
/// ([`kuikae_forbidden_after_chi`] for chi, the called kind for pon).
pub fn call_leaves_legal_discard(hand: &[Tile], consumed: &[Tile], forbidden: &[usize]) -> bool {
    if forbidden.is_empty() {
        return true;
    }
    let mut rest: Vec<Tile> = hand.to_vec();
    for c in consumed {
        if let Some(pos) = rest.iter().position(|t| t == c) {
            rest.remove(pos);
        }
    }
    rest.iter().any(|t| !forbidden.contains(&t.kind()))
}

/// Kinds that may not be discarded right after a chi: those that would form the same
/// sequence with the two consumed tiles.
///
/// Covers both same-tile and suji kuikae. The maintainer mirror and the
/// authoritative table must both call this so their legal action sets cannot drift.
pub fn kuikae_forbidden_after_chi(called: Tile, consumed: [Tile; 2]) -> Vec<usize> {
    let mut forbidden = Vec::new();
    for kind in 0..27usize {
        let candidate = unsafe { Tile::from_id_unchecked(kind as u8) };
        if candidate.suit() != consumed[0].suit() || candidate.suit() != consumed[1].suit() {
            continue;
        }
        let mut kinds = [candidate.kind(), consumed[0].kind(), consumed[1].kind()];
        kinds.sort_unstable();
        if kinds[1] == kinds[0] + 1 && kinds[2] == kinds[1] + 1 {
            forbidden.push(kind);
        }
    }
    if !forbidden.contains(&called.kind()) {
        forbidden.push(called.kind());
    }
    forbidden
}
