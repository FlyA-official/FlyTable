//! Shanten calculation.
//!
//! Tenpai is 0 and a complete hand is -1. The result is the minimum over the
//! standard form, chiitoitsu and kokushi.
//!
//! `melds` is the number of called melds (chi, pon and kan count as one each;
//! nukidora does not count).

use crate::hand::TileCounts;
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

pub const AGARI: i8 = -1;
pub const TENPAI: i8 = 0;

/// Shanten of a hand. `concealed` excludes melds, and must hold `13 - 3 * melds`
/// or `14 - 3 * melds` tiles.
pub fn shanten(concealed: &TileCounts, melds: u8) -> i8 {
    shanten_for_rule_line(concealed, melds, false)
}

/// 3-player shanten: 2m..8m do not exist and 1m/9m cannot form sequences.
pub fn shanten_3p(concealed: &TileCounts, melds: u8) -> i8 {
    shanten_for_rule_line(concealed, melds, true)
}

fn shanten_for_rule_line(concealed: &TileCounts, melds: u8, is_sanma: bool) -> i8 {
    let std = shanten_standard_for_rule_line(concealed, melds, is_sanma);
    if melds == 0 {
        let chi = shanten_chiitoitsu(concealed);
        let kok = shanten_kokushi(concealed);
        std.min(chi).min(kok)
    } else {
        std
    }
}

/// Standard-form shanten (four groups and a pair).
pub fn shanten_standard(concealed: &TileCounts, melds: u8) -> i8 {
    shanten_standard_for_rule_line(concealed, melds, false)
}

/// 3-player standard-form shanten; 1m/9m only form triplets or pairs.
pub fn shanten_standard_3p(concealed: &TileCounts, melds: u8) -> i8 {
    shanten_standard_for_rule_line(concealed, melds, true)
}

fn shanten_standard_for_rule_line(concealed: &TileCounts, melds: u8, is_sanma: bool) -> i8 {
    shanten_standard_exact(concealed, melds, is_sanma)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LocalPattern {
    melds: u8,
    pair: bool,
    counts: [u8; 9],
}

const LOCAL_INF: u8 = u8::MAX;
type LocalCosts = [[u8; 2]; 5];

static SUIT_PATTERNS: OnceLock<Vec<LocalPattern>> = OnceLock::new();
static HONOR_PATTERNS: OnceLock<Vec<LocalPattern>> = OnceLock::new();
static SANMA_MAN_PATTERNS: OnceLock<Vec<LocalPattern>> = OnceLock::new();
static LOCAL_COST_CACHE: OnceLock<Mutex<HashMap<(u8, u32), LocalCosts>>> = OnceLock::new();

/// Exact standard-form distance for 4-player and 3-player.
///
/// The common recursive formula treats the leftover of a four-of-a-kind after
/// taking a triplet as a single that can still pair, which implicitly assumes a
/// fifth copy. That undercounts by one when a kind is fully used, and at the 1m/9m
/// boundary in 3-player. Instead this enumerates valid target shapes per suit and
/// runs a DP over the four suits, so target counts never exceed four and 3-player
/// never produces 2m..8m.
fn shanten_standard_exact(concealed: &TileCounts, melds: u8, is_sanma: bool) -> i8 {
    let need_melds = 4usize.saturating_sub(melds as usize);
    let raw = concealed.raw();
    let groups = [
        local_costs(if is_sanma { 2 } else { 0 }, &raw[..9]),
        local_costs(0, &raw[9..18]),
        local_costs(0, &raw[18..27]),
        local_costs(1, &raw[27..34]),
    ];
    let mut dp = [[LOCAL_INF; 2]; 5];
    dp[0][0] = 0;
    for group in groups {
        let mut next = [[LOCAL_INF; 2]; 5];
        for used_melds in 0..=need_melds {
            for used_pair in 0..=1 {
                let base = dp[used_melds][used_pair];
                if base == LOCAL_INF {
                    continue;
                }
                for local_melds in 0..=need_melds - used_melds {
                    for local_pair in 0..=1 - used_pair {
                        let cost = group[local_melds][local_pair];
                        if cost == LOCAL_INF {
                            continue;
                        }
                        let slot = &mut next[used_melds + local_melds][used_pair + local_pair];
                        *slot = (*slot).min(base.saturating_add(cost));
                    }
                }
            }
        }
        dp = next;
    }
    dp[need_melds][1] as i8 - 1
}

fn local_costs(kind: u8, counts: &[u8]) -> LocalCosts {
    let mut encoded = 0u32;
    for &count in counts {
        encoded = encoded * 5 + count as u32;
    }
    let cache = LOCAL_COST_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(costs) = cache
        .lock()
        .expect("local shanten cache poisoned")
        .get(&(kind, encoded))
        .copied()
    {
        return costs;
    }

    let patterns = match kind {
        0 => SUIT_PATTERNS.get_or_init(|| build_local_patterns(0)),
        1 => HONOR_PATTERNS.get_or_init(|| build_local_patterns(1)),
        2 => SANMA_MAN_PATTERNS.get_or_init(|| build_local_patterns(2)),
        _ => unreachable!("unknown local shanten pattern kind"),
    };
    let mut costs = [[LOCAL_INF; 2]; 5];
    for pattern in patterns {
        let additions = pattern
            .counts
            .iter()
            .zip(counts.iter().copied().chain(std::iter::repeat(0)))
            .map(|(&target, current)| target.saturating_sub(current))
            .sum::<u8>();
        let slot = &mut costs[pattern.melds as usize][usize::from(pattern.pair)];
        *slot = (*slot).min(additions);
    }
    cache
        .lock()
        .expect("local shanten cache poisoned")
        .insert((kind, encoded), costs);
    costs
}

fn build_local_patterns(kind: u8) -> Vec<LocalPattern> {
    let valid: Vec<usize> = match kind {
        0 => (0..9).collect(),
        1 => (0..7).collect(),
        2 => vec![0, 8],
        _ => unreachable!("unknown local shanten pattern kind"),
    };
    let mut meld_shapes: Vec<[usize; 3]> = valid.iter().map(|&rank| [rank, rank, rank]).collect();
    if kind == 0 {
        meld_shapes.extend((0..7).map(|rank| [rank, rank + 1, rank + 2]));
    }

    let mut unique = HashSet::new();
    for pair in std::iter::once(None).chain(valid.iter().copied().map(Some)) {
        let mut counts = [0u8; 9];
        if let Some(rank) = pair {
            counts[rank] = 2;
        }
        build_local_patterns_from(&meld_shapes, 0, 0, pair.is_some(), &mut counts, &mut unique);
    }
    let mut patterns: Vec<_> = unique.into_iter().collect();
    patterns.sort_by_key(|pattern| (pattern.melds, pattern.pair, pattern.counts));
    patterns
}

fn build_local_patterns_from(
    meld_shapes: &[[usize; 3]],
    start: usize,
    melds: u8,
    pair: bool,
    counts: &mut [u8; 9],
    out: &mut HashSet<LocalPattern>,
) {
    out.insert(LocalPattern {
        melds,
        pair,
        counts: *counts,
    });
    if melds == 4 {
        return;
    }
    for index in start..meld_shapes.len() {
        let shape = meld_shapes[index];
        for rank in shape {
            counts[rank] += 1;
        }
        if counts.iter().all(|&count| count <= 4) {
            build_local_patterns_from(meld_shapes, index, melds + 1, pair, counts, out);
        }
        for rank in shape {
            counts[rank] -= 1;
        }
    }
}

/// Chiitoitsu shanten: 6 - pairs + max(0, 7 - distinct kinds).
pub fn shanten_chiitoitsu(c: &TileCounts) -> i8 {
    let mut pairs = 0i8;
    let mut kinds = 0i8;
    for &n in c.raw() {
        if n >= 1 {
            kinds += 1;
        }
        if n >= 2 {
            pairs += 1;
        }
    }
    let lack = (7 - kinds).max(0);
    6 - pairs + lack
}

/// Kokushi shanten: 13 - distinct terminals/honors - (1 if any of them is paired).
pub fn shanten_kokushi(c: &TileCounts) -> i8 {
    const YAOCHUU: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    let mut kinds = 0i8;
    let mut has_pair = false;
    for &k in &YAOCHUU {
        let n = c.count(k);
        if n >= 1 {
            kinds += 1;
        }
        if n >= 2 {
            has_pair = true;
        }
    }
    13 - kinds - i8::from(has_pair)
}
