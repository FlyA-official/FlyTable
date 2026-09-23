//! Decomposition of complete hands.
//!
//! Enumerates every way to split a concealed 3n+2 hand into a pair and groups.
//! Yaku evaluation takes the best over all of them, since one hand can have
//! several readings (`111222333m` is three triplets or three sequences).
//!
//! Chiitoitsu and kokushi are marked separately. Called melds are passed in by the
//! caller as fixed groups and are not decomposed here.

use crate::tile::Tile;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MeldShape {
    /// Sequence, identified by its lowest kind (1m means 1m2m3m).
    Shuntsu(u8),
    /// Triplet of a kind.
    Kotsu(u8),
}

impl MeldShape {
    /// Representative kind (lowest tile for sequences).
    pub fn tile_kind(self) -> u8 {
        match self {
            MeldShape::Shuntsu(k) | MeldShape::Kotsu(k) => k,
        }
    }
    pub fn is_shuntsu(self) -> bool {
        matches!(self, MeldShape::Shuntsu(_))
    }
    pub fn is_kotsu(self) -> bool {
        matches!(self, MeldShape::Kotsu(_))
    }
}

/// One standard-form reading of the concealed tiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StandardParse {
    pub pair: u8,
    /// Groups from the concealed tiles (called melds excluded).
    pub melds: Vec<MeldShape>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decomposition {
    /// All concealed readings of a standard hand (four groups and a pair, called melds included).
    Standard(Vec<StandardParse>),
    Chiitoitsu,
    Kokushi,
}

/// Enumerates every concealed reading of a standard hand, including called melds.
///
/// Does not handle chiitoitsu or kokushi; callers can use it to compare scores when
/// a hand is both chiitoitsu and standard.
pub fn standard_parses(concealed: &[u8; 34], open_melds: u8) -> Vec<StandardParse> {
    if open_melds > 4 {
        return Vec::new();
    }
    let need_melds = 4 - open_melds as usize;
    let mut parses = Vec::new();
    for pair in 0..34u8 {
        if concealed[pair as usize] >= 2 {
            let mut counts = *concealed;
            counts[pair as usize] -= 2;
            let mut melds = Vec::with_capacity(need_melds);
            let mut found_here = Vec::new();
            collect_melds(&mut counts, 0, need_melds, &mut melds, &mut found_here);
            for m in found_here {
                parses.push(StandardParse { pair, melds: m });
            }
        }
    }
    // Different extraction orders can produce the same group set.
    parses.sort_by(|a, b| a.melds.cmp(&b.melds).then(a.pair.cmp(&b.pair)));
    parses.dedup();
    parses
}

/// Decomposes the concealed counts. Returns `Standard(vec![])` if the hand is not complete.
pub fn decompose(concealed: &[u8; 34], open_melds: u8) -> Decomposition {
    // The special forms require a closed hand.
    if open_melds == 0 {
        if is_chiitoitsu(concealed) {
            return Decomposition::Chiitoitsu;
        }
        if is_kokushi(concealed) {
            return Decomposition::Kokushi;
        }
    }
    Decomposition::Standard(standard_parses(concealed, open_melds))
}

/// Backtracks from `start`, collecting every way to form `need` groups that uses all tiles.
fn collect_melds(
    counts: &mut [u8; 34],
    start: usize,
    need: usize,
    acc: &mut Vec<MeldShape>,
    out: &mut Vec<Vec<MeldShape>>,
) {
    if need == 0 {
        if counts.iter().all(|&c| c == 0) {
            out.push(acc.clone());
        }
        return;
    }
    let mut k = start;
    while k < 34 && counts[k] == 0 {
        k += 1;
    }
    if k == 34 {
        return;
    }

    if counts[k] >= 3 {
        counts[k] -= 3;
        acc.push(MeldShape::Kotsu(k as u8));
        collect_melds(counts, k, need - 1, acc, out);
        acc.pop();
        counts[k] += 3;
    }
    // Number tiles only, without crossing suit boundaries.
    if is_number(k) && k % 9 <= 6 && counts[k + 1] > 0 && counts[k + 2] > 0 {
        counts[k] -= 1;
        counts[k + 1] -= 1;
        counts[k + 2] -= 1;
        acc.push(MeldShape::Shuntsu(k as u8));
        collect_melds(counts, k, need - 1, acc, out);
        acc.pop();
        counts[k] += 1;
        counts[k + 1] += 1;
        counts[k + 2] += 1;
    }
}

#[inline]
fn is_number(k: usize) -> bool {
    k < 27
}

pub fn is_chiitoitsu(c: &[u8; 34]) -> bool {
    let mut pairs = 0;
    for &n in c {
        match n {
            0 => {}
            2 => pairs += 1,
            _ => return false, // 1, 3 or 4 copies; four of a kind is not two pairs
        }
    }
    pairs == 7
}

pub fn is_kokushi(c: &[u8; 34]) -> bool {
    const YAOCHUU: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    let mut total = 0u8;
    let mut has_pair = false;
    for (k, &count) in c.iter().enumerate() {
        if YAOCHUU.contains(&k) {
            if count == 0 {
                return false;
            }
            if count >= 2 {
                has_pair = true;
            }
            total += count;
        } else if count != 0 {
            return false;
        }
    }
    has_pair && total == 14
}

/// Where the winning tile sits in a reading; determines fu and pinfu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitShape {
    /// Two-sided wait (23 waiting on 1/4).
    Ryanmen,
    /// Closed wait (13 waiting on 2).
    Kanchan,
    /// Edge wait (12 waiting on 3, 89 waiting on 7).
    Penchan,
    /// Pair wait.
    Tanki,
    /// Dual pon wait.
    Shanpon,
}

/// Every valid interpretation of `win` within `parse`.
///
/// The winning tile can belong to the pair, a sequence end or a triplet. Fu takes
/// the highest-fu interpretation (closed, edge and pair waits add 2), while pinfu
/// requires the two-sided one. Because the preferences differ, this returns all of
/// them; [`wait_shape`] gives the pinfu-favorable choice.
pub fn wait_shapes(parse: &StandardParse, win: Tile) -> Vec<WaitShape> {
    fn push(out: &mut Vec<WaitShape>, cand: WaitShape) {
        if !out.contains(&cand) {
            out.push(cand);
        }
    }

    let wk = win.kind() as u8;
    let mut out = Vec::new();

    if parse.pair == wk {
        push(&mut out, WaitShape::Tanki);
    }
    for m in &parse.melds {
        match *m {
            MeldShape::Kotsu(k) if k == wk => push(&mut out, WaitShape::Shanpon),
            MeldShape::Shuntsu(s) => {
                let (a, b, c) = (s, s + 1, s + 2);
                if wk == a || wk == c {
                    // Edge: 12 waiting on 3, or 89 waiting on 7.
                    let rank = a % 9;
                    if (wk == c && rank == 0) || (wk == a && rank == 6) {
                        push(&mut out, WaitShape::Penchan);
                    } else {
                        push(&mut out, WaitShape::Ryanmen);
                    }
                } else if wk == b {
                    push(&mut out, WaitShape::Kanchan);
                }
            }
            _ => {}
        }
    }
    out
}

/// The wait most favorable for pinfu: two-sided > closed/edge > dual pon/pair.
/// Dual pon also turns a concealed triplet open, so two-sided is preferred there
/// too. Do not use this for fu; see [`wait_shapes`].
pub fn wait_shape(parse: &StandardParse, win: Tile) -> Option<WaitShape> {
    wait_shapes(parse, win).into_iter().reduce(better_wait)
}

fn better_wait(a: WaitShape, b: WaitShape) -> WaitShape {
    fn rank(w: WaitShape) -> u8 {
        match w {
            WaitShape::Ryanmen => 3,
            WaitShape::Kanchan | WaitShape::Penchan => 2,
            WaitShape::Shanpon | WaitShape::Tanki => 1,
        }
    }
    if rank(b) > rank(a) { b } else { a }
}
