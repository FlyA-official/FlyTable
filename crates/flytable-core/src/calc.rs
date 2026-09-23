//! Calculators over known information.
//!
//! Only facts that follow from the rules and visible tiles are computed here.
//! Anything that involves judgment or probability belongs to the decision maker.
//!
//! - [`shanten_calc`] - shanten.
//! - [`genbutsu_safe`] - genbutsu, tiles guaranteed safe by the furiten rule.
//! - [`certain_yaku`] - yaku present for every winning tile.
//! - [`possible_yaku`] - yaku present for at least one winning tile.
//!
//! Suji, kabe and other danger estimates are judgments and are out of scope.

use crate::agari;
use crate::hand::TileCounts;
use crate::tile::Tile;
use crate::yaku::{Yaku, YakuContext, evaluate};

/// Shanten from concealed counts plus the number of called melds.
pub fn shanten_calc(concealed: &TileCounts, open_melds: u8) -> i8 {
    crate::shanten::shanten(concealed, open_melds)
}

/// 3-player shanten: 1m/9m never form sequences and 2m..8m are never ukeire.
pub fn shanten_calc_3p(concealed: &TileCounts, open_melds: u8) -> i8 {
    crate::shanten::shanten_3p(concealed, open_melds)
}

/// Genbutsu for one opponent: tile kinds in that player's own discards.
///
/// A player is furiten on any tile in their own discards and cannot ron it, so these
/// tiles are always safe against them. Returns base kinds. Suji and kabe are not
/// included.
pub fn genbutsu_safe(opponent_discards: &[Tile]) -> Vec<usize> {
    let mut set = [false; 34];
    for t in opponent_discards {
        set[t.kind()] = true;
    }
    (0..34).filter(|&k| set[k]).collect()
}

/// Tile kinds that are genbutsu against every given opponent (intersection).
pub fn genbutsu_safe_against_all(opponents_discards: &[&[Tile]]) -> Vec<usize> {
    if opponents_discards.is_empty() {
        return Vec::new();
    }
    let mut acc = [true; 34];
    for &disc in opponents_discards {
        let mut here = [false; 34];
        for t in disc {
            here[t.kind()] = true;
        }
        for k in 0..34 {
            acc[k] &= here[k];
        }
    }
    (0..34).filter(|&k| acc[k]).collect()
}

/// Yaku that hold whichever winning tile completes the hand: the intersection of
/// the yaku sets over all winning tiles. Empty when not tenpai.
pub fn certain_yaku(ctx_template: &YakuContext) -> Vec<Yaku> {
    certain_yaku_for_rule_line(ctx_template, false)
}

pub fn certain_yaku_3p(ctx_template: &YakuContext) -> Vec<Yaku> {
    certain_yaku_for_rule_line(ctx_template, true)
}

fn certain_yaku_for_rule_line(ctx_template: &YakuContext, is_sanma: bool) -> Vec<Yaku> {
    let branches = yaku_per_win(ctx_template, is_sanma);
    if branches.is_empty() {
        return Vec::new();
    }
    let mut iter = branches.into_iter();
    let mut acc: Vec<Yaku> = iter.next().unwrap();
    for set in iter {
        acc.retain(|y| set.contains(y));
    }
    acc
}

/// Union of yaku sets over all winning tiles.
pub fn possible_yaku(ctx_template: &YakuContext) -> Vec<Yaku> {
    possible_yaku_for_rule_line(ctx_template, false)
}

pub fn possible_yaku_3p(ctx_template: &YakuContext) -> Vec<Yaku> {
    possible_yaku_for_rule_line(ctx_template, true)
}

fn possible_yaku_for_rule_line(ctx_template: &YakuContext, is_sanma: bool) -> Vec<Yaku> {
    let branches = yaku_per_win(ctx_template, is_sanma);
    let mut acc: Vec<Yaku> = Vec::new();
    for set in branches {
        for y in set {
            if !acc.contains(&y) {
                acc.push(y);
            }
        }
    }
    acc
}

/// Yaku set for each winning tile. `ctx_template.concealed` is the tenpai hand
/// (13 - 3 * melds tiles); `win_tile` is replaced per winning tile.
fn yaku_per_win(ctx_template: &YakuContext, is_sanma: bool) -> Vec<Vec<Yaku>> {
    let counts = TileCounts::from_raw(ctx_template.concealed);
    let open = ctx_template.melds_set_count_pub();
    let waits = if is_sanma {
        agari::winning_tiles_3p(&counts, open)
    } else {
        agari::winning_tiles(&counts, open)
    };
    let mut out = Vec::new();
    for w in waits {
        let win = unsafe { Tile::from_id_unchecked(w as u8) };
        let mut c = counts;
        c.add(win, 1);
        let mut ctx = ctx_template.clone();
        ctx.concealed = *c.raw();
        ctx.win_tile = win;
        let r = evaluate(&ctx);
        let ys: Vec<Yaku> = r.yaku.iter().map(|(y, _)| *y).collect();
        out.push(ys);
    }
    out
}
