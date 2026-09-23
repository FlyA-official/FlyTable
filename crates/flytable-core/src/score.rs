//! Fu and points.
//!
//! Han and fu to points, including limit hands and yakuman multipliers, for dealer
//! and non-dealer, tsumo and ron. 3-player payment splits (tsumo loss) are applied
//! by the caller; this module computes the payment unit.

use crate::decompose::{MeldShape, StandardParse, WaitShape, wait_shape, wait_shapes};
use crate::meld::Meld;
use crate::rules::{KazoeLimit, RiichiRuleProfile};
use crate::tile::Tile;
use crate::yaku::{YakuContext, YakuResult};

/// Authoritative limit tier of a win.
///
/// Display code must not derive this from han and fu again; profile differences such
/// as kiriage and the counted-yakuman cap are already applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreLimit {
    Normal,
    Mangan,
    Haneman,
    Baiman,
    Sanbaiman,
    Yakuman(u8),
}

impl ScoreResult {
    /// Total a tsumo winner receives (in 3-player only two players pay).
    ///
    /// This is also what the liable player pays alone under pao on a tsumo: the full
    /// tsumo total, not the ron value (e.g. a 3-player dealer daisangen tsumo pays
    /// 32000 = 16000 x 2, not 48000). Zero for ron results.
    pub const fn tsumo_total(&self, seats: usize) -> i32 {
        let payers = seats as i32 - 1;
        if self.tsumo_oya == 0 {
            // Dealer tsumo: each non-dealer pays tsumo_ko.
            self.tsumo_ko * payers
        } else {
            // Non-dealer tsumo: dealer pays tsumo_oya, other non-dealers pay tsumo_ko.
            self.tsumo_oya + self.tsumo_ko * (payers - 1)
        }
    }
}

impl ScoreLimit {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Mangan => "mangan",
            Self::Haneman => "haneman",
            Self::Baiman => "baiman",
            Self::Sanbaiman => "sanbaiman",
            Self::Yakuman(_) => "yakuman",
        }
    }

    pub const fn yakuman_multiplier(self) -> u8 {
        match self {
            Self::Yakuman(multiplier) => multiplier,
            _ => 0,
        }
    }
}

/// Scoring result of a single win.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoreResult {
    pub han: u8,
    pub fu: u8,
    pub limit: ScoreLimit,
    /// Explicit yakuman multiplier; counted yakuman stay 0. Use `limit.yakuman_multiplier()` for display.
    pub yakuman: u8,
    /// Ron: amount paid by the discarder.
    pub ron: i32,
    /// Tsumo: amount paid by each non-dealer.
    pub tsumo_ko: i32,
    /// Tsumo: amount paid by the dealer (0 on a dealer tsumo; use tsumo_ko).
    pub tsumo_oya: i32,
}

/// Full evaluation: yaku, best decomposition by han and fu, points and dora.
///
/// `dora_han` is the total from dora, red fives, ura dora and nukidora, counted by
/// the table layer. Returns `None` when there is no yaku.
pub fn evaluate_full(ctx: &YakuContext, dora_han: u8, is_oya: bool) -> Option<FullScore> {
    let decomp = crate::decompose::decompose(&ctx.concealed, ctx.melds_set_count_pub());

    // Yakuman take a separate path; fu is irrelevant.
    let yk = crate::yaku::evaluate(ctx);
    if yk.yakuman > 0 {
        let s = score(&yk, 0, is_oya, ctx.is_tsumo);
        return Some(FullScore {
            yaku: yk,
            fu: 0,
            dora_han: 0,
            regular_dora_han: 0,
            red_dora_han: 0,
            ura_dora_han: 0,
            nuki_dora_han: 0,
            score: s,
        });
    }

    // Regular yaku: score every decomposition and keep the best.
    let (is_chiitoi, parses): (bool, Vec<StandardParse>) = match &decomp {
        crate::decompose::Decomposition::Chiitoitsu => (
            true,
            crate::decompose::standard_parses(&ctx.concealed, ctx.melds_set_count_pub()),
        ),
        crate::decompose::Decomposition::Standard(ps) => (false, ps.clone()),
        crate::decompose::Decomposition::Kokushi => return None, // handled above as yakuman
    };

    let mut best: Option<FullScore> = None;

    if is_chiitoi {
        let yk = crate::yaku::eval_chiitoitsu(ctx);
        if yk.han == 0 {
            return None;
        }
        let total_han = yk.han + dora_han;
        let fu = calc_fu(ctx, &dummy_parse(), false, true);
        let s = score_with_han(total_han, fu, is_oya, ctx.is_tsumo, ctx.rule_profile);
        best = Some(FullScore {
            yaku: yk,
            fu,
            dora_han,
            regular_dora_han: dora_han,
            red_dora_han: 0,
            ura_dora_han: 0,
            nuki_dora_han: 0,
            score: s,
        });
    }

    for parse in &parses {
        let yk = crate::yaku::eval_standard_pub(ctx, parse);
        if yk.han == 0 {
            continue; // no yaku in this decomposition
        }
        let is_pinfu = yk
            .yaku
            .iter()
            .any(|(y, _)| matches!(y, crate::yaku::Yaku::Pinfu));
        let fu = calc_fu(ctx, parse, is_pinfu, false);
        let total_han = yk.han + dora_han;
        let s = score_with_han(total_han, fu, is_oya, ctx.is_tsumo, ctx.rule_profile);
        let cand = FullScore {
            yaku: yk,
            fu,
            dora_han,
            regular_dora_han: dora_han,
            red_dora_han: 0,
            ura_dora_han: 0,
            nuki_dora_han: 0,
            score: s,
        };
        best = Some(match best {
            None => cand,
            Some(cur) => better_score(cand, cur),
        });
    }
    best
}

#[derive(Debug, Clone)]
pub struct FullScore {
    pub yaku: YakuResult,
    pub fu: u8,
    /// Total dora han.
    pub dora_han: u8,
    /// Regular dora han, including indicators pointing at set-aside Norths.
    pub regular_dora_han: u8,
    pub red_dora_han: u8,
    pub ura_dora_han: u8,
    /// Han from nukidora themselves, excluding indicator hits.
    pub nuki_dora_han: u8,
    pub score: ScoreResult,
}

fn payout(s: &ScoreResult) -> i32 {
    s.ron + s.tsumo_oya + s.tsumo_ko * 2
}

/// Picks between decompositions: higher points, then higher han, then higher fu.
///
/// Comparing points alone keeps whichever candidate came first on a tie, which can
/// show different han/fu from Tenhou. Example: 444m 555m 666m 345s 66s, ron on 6s:
/// sanankou 5 han 50 fu and iipeikou 4 han 40 fu are both mangan; the former wins.
fn better_score(cand: FullScore, cur: FullScore) -> FullScore {
    let key = |f: &FullScore| (payout(&f.score), f.score.han, f.fu);
    if key(&cand) > key(&cur) { cand } else { cur }
}

fn dummy_parse() -> StandardParse {
    StandardParse {
        pair: 0,
        melds: vec![],
    }
}

/// Points for a given total han (yaku plus dora) and fu.
fn score_with_han(
    han: u8,
    fu: u8,
    is_oya: bool,
    is_tsumo: bool,
    profile: RiichiRuleProfile,
) -> ScoreResult {
    let bp = base_points_with_profile(han, fu, profile);
    score_from_base(bp, han, fu, 0, is_oya, is_tsumo)
}

/// Base points from han and fu; kiriage and counted yakuman come from the profile.
fn base_points_with_profile(han: u8, fu: u8, profile: RiichiRuleProfile) -> i32 {
    // Yakuman are handled elsewhere.
    match han {
        0 => 0,
        1..=4 => {
            let bp = fu as i32 * (1 << (2 + han as i32));
            if profile.kiriage_mangan() && matches!((han, fu), (4, 30) | (3, 60)) {
                2000
            } else {
                bp.min(2000)
            }
        }
        5 => 2000,       // mangan
        6..=7 => 3000,   // haneman
        8..=10 => 4000,  // baiman
        11..=12 => 6000, // sanbaiman
        _ => match profile.kazoe_limit() {
            KazoeLimit::Yakuman => 8000,
            KazoeLimit::Sanbaiman => 6000,
        },
    }
}

/// Rounds up to the next 100.
fn ceil100(x: i32) -> i32 {
    ((x + 99) / 100) * 100
}

/// Points for a win. `is_oya` is whether the winner is the dealer. Honba and riichi
/// sticks are added by the caller during settlement.
pub fn score(result: &YakuResult, fu: u8, is_oya: bool, is_tsumo: bool) -> ScoreResult {
    score_with_profile(result, fu, is_oya, is_tsumo, RiichiRuleProfile::tenhou())
}

/// Points using an explicit platform or room profile.
pub fn score_with_profile(
    result: &YakuResult,
    fu: u8,
    is_oya: bool,
    is_tsumo: bool,
    profile: RiichiRuleProfile,
) -> ScoreResult {
    if result.yakuman > 0 {
        return score_yakuman(result.yakuman, is_oya, is_tsumo);
    }
    let bp = base_points_with_profile(result.han, fu, profile);
    score_from_base(bp, result.han, fu, 0, is_oya, is_tsumo)
}

fn score_yakuman(mult: u8, is_oya: bool, is_tsumo: bool) -> ScoreResult {
    let bp = 8000 * mult as i32;
    ScoreResult {
        yakuman: mult,
        ..score_from_base(bp, 0, 0, mult, is_oya, is_tsumo)
    }
}

fn score_from_base(
    bp: i32,
    han: u8,
    fu: u8,
    yakuman: u8,
    is_oya: bool,
    is_tsumo: bool,
) -> ScoreResult {
    let limit = if yakuman > 0 {
        ScoreLimit::Yakuman(yakuman)
    } else {
        match bp {
            2000 => ScoreLimit::Mangan,
            3000 => ScoreLimit::Haneman,
            4000 => ScoreLimit::Baiman,
            6000 => ScoreLimit::Sanbaiman,
            8000 => ScoreLimit::Yakuman(1),
            _ => ScoreLimit::Normal,
        }
    };
    let mut r = ScoreResult {
        han,
        fu,
        limit,
        yakuman,
        ron: 0,
        tsumo_ko: 0,
        tsumo_oya: 0,
    };
    if is_oya {
        if is_tsumo {
            r.tsumo_ko = ceil100(bp * 2);
        } else {
            r.ron = ceil100(bp * 6);
        }
    } else {
        if is_tsumo {
            r.tsumo_oya = ceil100(bp * 2);
            r.tsumo_ko = ceil100(bp);
        } else {
            r.ron = ceil100(bp * 4);
        }
    }
    r
}

/// Fu for a given context and decomposition, rounded up to 10.
///
/// Chiitoitsu (25 fu) and pinfu (20 tsumo / 30 ron) are handled by the caller. This
/// adds base, group, pair, wait and tsumo / closed-ron fu.
pub fn calc_fu(
    ctx: &YakuContext,
    parse: &StandardParse,
    is_pinfu: bool,
    is_chiitoitsu: bool,
) -> u8 {
    if is_chiitoitsu {
        return 25;
    }
    if is_pinfu {
        return if ctx.is_tsumo { 20 } else { 30 };
    }

    let mut fu: u8 = 20;

    if ctx.menzen && !ctx.is_tsumo {
        fu += 10;
    }
    if ctx.is_tsumo {
        fu += 2;
    }

    let melds = all_melds_with_concealed(ctx, parse);
    let kan_kinds: Vec<u8> = ctx
        .melds
        .iter()
        .filter(|m| m.is_kan())
        .map(|m| m.kind_tile().kind() as u8)
        .collect();
    for (m, concealed) in &melds {
        if let MeldShape::Kotsu(k) = m {
            if kan_kinds.contains(k) {
                continue;
            }
            let t = unsafe { Tile::from_id_unchecked(*k) };
            let base = if t.is_yaochuu() { 8 } else { 4 };
            // Concealed triplets score the base, open triplets half.
            fu += if *concealed { base } else { base / 2 };
        }
    }
    // Kans: open kan doubles the triplet base, closed kan doubles it again.
    for meld in &ctx.melds {
        match meld {
            Meld::Ankan { tile } => {
                fu += if tile.is_yaochuu() { 32 } else { 16 };
            }
            Meld::Daiminkan { tile, .. } | Meld::Kakan { tile, .. } => {
                fu += if tile.is_yaochuu() { 16 } else { 8 };
            }
            _ => {}
        }
    }

    // Pair fu: +2 each for dragon, round wind and seat wind (a double-wind pair gets 4).
    let pair = unsafe { Tile::from_id_unchecked(parse.pair) };
    if pair.is_dragon() {
        fu += 2;
    }
    if parse.pair == ctx.bakaze.kind() as u8 {
        fu += 2;
    }
    if parse.pair == ctx.jikaze.kind() as u8 {
        fu += 2;
    }

    // Wait fu: +2 for closed, edge or pair waits. When the winning tile can be read
    // several ways (44456m winning on 4m is a 4m pair wait or a 44m pair with a 456m
    // two-sided wait), the highest fu is used. Pinfu uses `wait_shape` instead, which
    // prefers the two-sided reading.
    if wait_shapes(parse, ctx.win_tile).iter().any(|w| {
        matches!(
            w,
            WaitShape::Kanchan | WaitShape::Penchan | WaitShape::Tanki
        )
    }) {
        fu += 2;
    }

    // Round up to 10. An open hand with only the 20 base fu scores 30.
    let rounded = fu.div_ceil(10) * 10;
    if rounded == 20 { 30 } else { rounded }
}

/// Expands groups with a concealed flag, as in the yaku module. Only a triplet completed by ron on a dual pon wait is open.
fn all_melds_with_concealed(ctx: &YakuContext, parse: &StandardParse) -> Vec<(MeldShape, bool)> {
    let mut v: Vec<(MeldShape, bool)> = parse.melds.iter().map(|&m| (m, true)).collect();
    if !ctx.is_tsumo
        && wait_shape(parse, ctx.win_tile) == Some(WaitShape::Shanpon)
        && let Some(slot) = v.iter_mut().find(|(m, c)| {
            *c && matches!(m, MeldShape::Kotsu(k) if *k == ctx.win_tile.kind() as u8)
        })
    {
        slot.1 = false;
    }
    for m in &ctx.melds {
        match m {
            Meld::Chi { tiles, .. } => {
                let mut ks: Vec<u8> = tiles.iter().map(|t| t.kind() as u8).collect();
                ks.sort_unstable();
                v.push((MeldShape::Shuntsu(ks[0]), false));
            }
            Meld::Pon { tile, .. } => v.push((MeldShape::Kotsu(tile.kind() as u8), false)),
            Meld::Daiminkan { tile, .. } | Meld::Kakan { tile, .. } => {
                v.push((MeldShape::Kotsu(tile.kind() as u8), false))
            }
            Meld::Ankan { tile } => v.push((MeldShape::Kotsu(tile.kind() as u8), true)),
            Meld::Nukidora { .. } => {}
        }
    }
    v
}
