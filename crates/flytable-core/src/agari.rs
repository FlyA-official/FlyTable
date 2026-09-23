//! Win detection and ukeire helpers. Yaku and scoring live in `yaku` and `score`.

use crate::hand::TileCounts;
use crate::shanten::{AGARI, shanten, shanten_3p};

/// Whether the hand (including the winning tile) is a complete shape, ignoring yaku.
pub fn is_agari(concealed: &TileCounts, melds: u8) -> bool {
    shanten(concealed, melds) == AGARI
}

/// 3-player variant of [`is_agari`].
pub fn is_agari_3p(concealed: &TileCounts, melds: u8) -> bool {
    shanten_3p(concealed, melds) == AGARI
}

/// Tile kinds that would reduce shanten by one (ukeire), as indices into the 34 kinds.
pub fn ukeire(concealed: &TileCounts, melds: u8) -> Vec<usize> {
    ukeire_for_rule_line(concealed, melds, false)
}

/// 3-player ukeire; never returns 2m..8m.
pub fn ukeire_3p(concealed: &TileCounts, melds: u8) -> Vec<usize> {
    ukeire_for_rule_line(concealed, melds, true)
}

fn ukeire_for_rule_line(concealed: &TileCounts, melds: u8, is_sanma: bool) -> Vec<usize> {
    let shanten_fn = if is_sanma { shanten_3p } else { shanten };
    let cur = shanten_fn(concealed, melds);
    let mut out = Vec::new();
    let mut c = *concealed;
    for k in 0..34usize {
        if is_sanma && (1..=7).contains(&k) {
            continue;
        }
        // Only existence matters here, so a kind already held four times is skipped.
        if c.count(k) >= 4 {
            continue;
        }
        // Safety: k < 34
        let t = unsafe { crate::tile::Tile::from_id_unchecked(k as u8) };
        c.add(t, 1);
        if shanten_fn(&c, melds) < cur {
            out.push(k);
        }
        c.sub(t, 1);
    }
    out
}

/// Winning tiles when tenpai (the ukeire that completes the hand).
pub fn winning_tiles(concealed: &TileCounts, melds: u8) -> Vec<usize> {
    winning_tiles_for_rule_line(concealed, melds, false)
}

/// 3-player winning tiles; never returns 2m..8m.
pub fn winning_tiles_3p(concealed: &TileCounts, melds: u8) -> Vec<usize> {
    winning_tiles_for_rule_line(concealed, melds, true)
}

fn winning_tiles_for_rule_line(concealed: &TileCounts, melds: u8, is_sanma: bool) -> Vec<usize> {
    let shanten_fn = if is_sanma { shanten_3p } else { shanten };
    if shanten_fn(concealed, melds) != crate::shanten::TENPAI {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut c = *concealed;
    for k in 0..34usize {
        if is_sanma && (1..=7).contains(&k) {
            continue;
        }
        if c.count(k) >= 4 {
            continue;
        }
        let t = unsafe { crate::tile::Tile::from_id_unchecked(k as u8) };
        c.add(t, 1);
        if shanten_fn(&c, melds) == AGARI {
            out.push(k);
        }
        c.sub(t, 1);
    }
    out
}
