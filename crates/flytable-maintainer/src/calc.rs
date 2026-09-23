//! Calculators over a seat view (for table-state services and clients).
//!
//! Thin wrappers that connect the known-information calculators in `flytable-core`
//! to seat views. A view only contains what the seat can see, so these only use
//! known information.
//!
//! Same boundary as `core::calc`: only certain facts (shanten, genbutsu, certain and
//! possible yaku); no probabilities or judgment.

use flytable_core::calc;
use flytable_core::hand::TileCounts;
use flytable_core::tile::Tile;
use flytable_core::yaku::{Yaku, YakuContext};

use crate::view::SeatView;

/// Own shanten.
pub fn my_shanten(view: &SeatView) -> i8 {
    let counts = TileCounts::from_tiles(view.me.hand.iter().copied());
    let open = view
        .me
        .melds
        .iter()
        .filter(|m| !matches!(m, flytable_core::meld::Meld::Nukidora { .. }))
        .count() as u8;
    if view.others.len() + 1 == 3 {
        calc::shanten_calc_3p(&counts, open)
    } else {
        calc::shanten_calc(&counts, open)
    }
}

/// Shanten for given closed counts and meld count (used by legal action enumeration
/// to check tenpai after a discard).
pub fn shanten_after(counts: &TileCounts, open_melds: u8) -> i8 {
    calc::shanten_calc(counts, open_melds)
}

/// Shanten after a discard for an explicit seat-count variant.
pub fn shanten_after_for(counts: &TileCounts, open_melds: u8, is_sanma: bool) -> i8 {
    if is_sanma {
        calc::shanten_calc_3p(counts, open_melds)
    } else {
        calc::shanten_calc(counts, open_melds)
    }
}

/// Whether the seat is tenpai.
pub fn my_tenpai(view: &SeatView) -> bool {
    my_shanten(view) == flytable_core::shanten::TENPAI
}

/// Genbutsu against one opponent.
pub fn genbutsu_against(view: &SeatView, opponent_seat: u8) -> Vec<usize> {
    view.others
        .iter()
        .find(|o| o.seat == opponent_seat)
        .map(|o| calc::genbutsu_safe(&o.discards))
        .unwrap_or_default()
}

/// Genbutsu against every opponent in riichi. Opponents not in riichi are excluded,
/// since whether they are tenpai is a judgment.
pub fn genbutsu_against_riichi(view: &SeatView) -> Vec<usize> {
    let discs: Vec<&[Tile]> = view
        .others
        .iter()
        .filter(|o| o.riichi)
        .map(|o| o.discards.as_slice())
        .collect();
    calc::genbutsu_safe_against_all(&discs)
}

/// Certain and possible yaku of the own hand when tenpai: `(certain, possible)`.
pub fn my_yaku_outlook(view: &SeatView) -> (Vec<Yaku>, Vec<Yaku>) {
    if !my_tenpai(view) {
        return (Vec::new(), Vec::new());
    }
    let counts = TileCounts::from_tiles(view.me.hand.iter().copied());
    let offset = (view.me.seat + 4 - view.oya) % 4;
    let jikaze = unsafe { Tile::from_id_unchecked(27 + offset.min(3)) };
    let ctx = YakuContext {
        rule_profile: view.rule_profile,
        concealed: *counts.raw(),
        melds: view.me.melds.clone(),
        win_tile: Tile::default(),
        is_tsumo: false,
        menzen: view.me.melds.iter().all(|m| !m.breaks_menzen()),
        bakaze: view.bakaze,
        jikaze,
        riichi: view.me.riichi,
        double_riichi: false,
        ippatsu: false,
        haitei: false,
        houtei: false,
        rinshan: false,
        chankan: false,
        tenhou: false,
        chiihou: false,
        nuki_count: 0,
        pei_is_yakuhai: false,
    };
    if view.others.len() + 1 == 3 {
        (calc::certain_yaku_3p(&ctx), calc::possible_yaku_3p(&ctx))
    } else {
        (calc::certain_yaku(&ctx), calc::possible_yaku(&ctx))
    }
}

// Safe tile evaluation
//
// Genbutsu, four visible copies, kabe and suji, using only known information
// (discards, melds, own hand, dora indicators, riichi state). Whether an opponent
// is tenpai is left to the caller or model. `targets` are chosen by the caller
// (`riichi_targets` for rule-based threats).

/// Safety level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeLevel {
    /// Certain: genbutsu against every target, or all four copies visible.
    Absolute,
    /// Inferred: genbutsu against some targets, kabe or suji. Not guaranteed.
    Inferred,
}

/// Reason a tile is safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeReason {
    Genbutsu,
    DeadWall,
    Kabe,
    DoubleSuji,
    Suji,
}

impl SafeLevel {
    pub const fn wire_value(self) -> &'static str {
        match self {
            SafeLevel::Absolute => "absolute",
            SafeLevel::Inferred => "inferred",
        }
    }
}

impl SafeReason {
    pub const fn wire_value(self) -> &'static str {
        match self {
            SafeReason::Genbutsu => "genbutsu",
            SafeReason::DeadWall => "dead_wall",
            SafeReason::Kabe => "kabe",
            SafeReason::DoubleSuji => "double_suji",
            SafeReason::Suji => "suji",
        }
    }
}

/// Safety of one tile against the given targets (strongest reason).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafeTile {
    /// Tile kind `0..34`.
    pub tile: usize,
    pub level: SafeLevel,
    pub reason: SafeReason,
    /// Number of targets covered (genbutsu/suji: how many targets; four visible/kabe: all targets).
    pub coverage: u8,
}

/// Seats of opponents in riichi.
pub fn riichi_targets(view: &SeatView) -> Vec<u8> {
    view.others
        .iter()
        .filter(|o| o.riichi)
        .map(|o| o.seat)
        .collect()
}

/// Evaluates the safety of `candidates` (kinds `0..34`) against `targets`.
/// Returns the strongest reason per tile: genbutsu (all) / four visible > genbutsu
/// (some) > kabe > double suji > suji. Tiles with no reason are omitted.
pub fn evaluate_safe_tiles(view: &SeatView, targets: &[u8], candidates: &[usize]) -> Vec<SafeTile> {
    if targets.is_empty() {
        return Vec::new();
    }
    let visible = visible_counts(view);
    let target_discards: Vec<[bool; 34]> = targets
        .iter()
        .map(|&seat| {
            let mut set = [false; 34];
            if let Some(o) = view.others.iter().find(|o| o.seat == seat) {
                for d in &o.discards {
                    if d.kind() < 34 {
                        set[d.kind()] = true;
                    }
                }
            }
            set
        })
        .collect();

    evaluate_safe_tiles_raw(&visible, &target_discards, candidates)
}

/// Raw entry point: visible counts `visible[34]`, per-target genbutsu sets
/// `target_discards` (`[bool; 34]` each) and candidate kinds, with the same
/// classification. For callers that already have these counts and no `SeatView`.
pub fn evaluate_safe_tiles_raw(
    visible: &[u8; 34],
    target_discards: &[[bool; 34]],
    candidates: &[usize],
) -> Vec<SafeTile> {
    if target_discards.is_empty() {
        return Vec::new();
    }
    candidates
        .iter()
        .filter(|&&t| t < 34)
        .filter_map(|&t| classify_safe_tile(t, visible, target_discards))
        .collect()
}

fn classify_safe_tile(
    t: usize,
    visible: &[u8; 34],
    target_discards: &[[bool; 34]],
) -> Option<SafeTile> {
    let total = target_discards.len() as u8;
    let genbutsu_hits = target_discards.iter().filter(|d| d[t]).count() as u8;

    if genbutsu_hits == total {
        return Some(SafeTile {
            tile: t,
            level: SafeLevel::Absolute,
            reason: SafeReason::Genbutsu,
            coverage: total,
        });
    }
    if visible[t] >= 4 {
        return Some(SafeTile {
            tile: t,
            level: SafeLevel::Absolute,
            reason: SafeReason::DeadWall,
            coverage: total,
        });
    }
    if genbutsu_hits > 0 {
        return Some(SafeTile {
            tile: t,
            level: SafeLevel::Inferred,
            reason: SafeReason::Genbutsu,
            coverage: genbutsu_hits,
        });
    }
    if kabe_safe(t, visible) {
        return Some(SafeTile {
            tile: t,
            level: SafeLevel::Inferred,
            reason: SafeReason::Kabe,
            coverage: total,
        });
    }
    let mut double = 0u8;
    let mut single = 0u8;
    for d in target_discards {
        match suji_matches(t, d) {
            2.. => double += 1,
            1 => single += 1,
            _ => {}
        }
    }
    if double > 0 {
        return Some(SafeTile {
            tile: t,
            level: SafeLevel::Inferred,
            reason: SafeReason::DoubleSuji,
            coverage: double,
        });
    }
    if single > 0 {
        return Some(SafeTile {
            tile: t,
            level: SafeLevel::Inferred,
            reason: SafeReason::Suji,
            coverage: single,
        });
    }
    None
}

/// Visible copies per kind: own hand, all discards, all melds and dora indicators.
fn visible_counts(view: &SeatView) -> [u8; 34] {
    let mut counts = [0u8; 34];
    let mut incr = |t: &Tile| {
        let k = t.kind();
        if k < 34 {
            counts[k] = counts[k].saturating_add(1);
        }
    };
    for t in &view.me.hand {
        incr(t);
    }
    for t in &view.me.discards {
        incr(t);
    }
    for m in &view.me.melds {
        for t in m.tiles() {
            incr(&t);
        }
    }
    for o in &view.others {
        for t in &o.discards {
            incr(t);
        }
        for m in &o.melds {
            for t in m.tiles() {
                incr(&t);
            }
        }
    }
    for t in &view.dora_indicators {
        incr(t);
    }
    counts
}

/// Number tile kinds (< 27) to `(suit 0/1/2, rank 1..9)`; honors to `None`.
fn suit_rank(kind: usize) -> Option<(usize, usize)> {
    if kind < 27 {
        Some((kind / 9, kind % 9 + 1))
    } else {
        None
    }
}

fn kind_of(suit: usize, rank: usize) -> usize {
    suit * 9 + (rank - 1)
}

/// Number of suji partners of `t` in a target's genbutsu (0, 1 or 2).
fn suji_matches(t: usize, discards: &[bool; 34]) -> usize {
    let Some((suit, rank)) = suit_rank(t) else {
        return 0;
    };
    let partners: &[usize] = match rank {
        1 => &[4],
        2 => &[5],
        3 => &[6],
        4 => &[1, 7],
        5 => &[2, 8],
        6 => &[3, 9],
        7 => &[4],
        8 => &[5],
        9 => &[6],
        _ => &[],
    };
    partners
        .iter()
        .filter(|&&p| discards[kind_of(suit, p)])
        .count()
}

/// Kabe: the neighbours that could form a wait around `t` are all four visible.
fn kabe_safe(t: usize, visible: &[u8; 34]) -> bool {
    let Some((suit, rank)) = suit_rank(t) else {
        return false;
    };
    let blocked = |ranks: &[usize]| -> bool {
        ranks.is_empty() || ranks.iter().any(|&r| visible[kind_of(suit, r)] >= 4)
    };
    let lower: Vec<usize> = [rank.checked_sub(2), rank.checked_sub(1)]
        .into_iter()
        .flatten()
        .filter(|&r| r >= 1)
        .collect();
    let upper: Vec<usize> = [rank + 1, rank + 2]
        .into_iter()
        .filter(|&r| r <= 9)
        .collect();
    blocked(&lower) && blocked(&upper)
}
