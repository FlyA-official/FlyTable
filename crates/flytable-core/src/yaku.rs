//! Yaku detection and han counting.
//!
//! Input: the winning hand (concealed counts plus melds) and its context (round and
//! seat wind, tsumo or ron, situational flags). Output: the yaku that apply and the
//! total han or yakuman multiplier, taking the highest-scoring decomposition.
//!
//! Shared by 4-player and 3-player. Variant differences (no chi in 3-player,
//! nukidora han) come in through [`YakuContext`] and the melds passed by the caller.

use crate::decompose::{Decomposition, MeldShape, StandardParse, WaitShape, wait_shape};
use crate::meld::Meld;
use crate::rules::RiichiRuleProfile;
use crate::tile::Tile;

/// A single yaku. Han values for open hands are reduced at evaluation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Yaku {
    // 1 han
    Riichi,
    Ippatsu,
    MenzenTsumo,
    Pinfu,
    Tanyao,
    Iipeikou,
    Yakuhai(YakuhaiKind),
    Haitei,
    Houtei,
    Rinshan,
    Chankan,
    // 2 han
    DoubleRiichi,
    Chiitoitsu,
    Sanshoku, // closed 2 / open 1
    SanshokuDoukou,
    Ittsuu, // closed 2 / open 1
    Chanta, // closed 2 / open 1
    Toitoi,
    Sanankou,
    Sankantsu,
    Honroutou,
    Shousangen,
    // 3 han
    Honitsu, // closed 3 / open 2
    Junchan, // closed 3 / open 2
    Ryanpeikou,
    // 6 han
    Chinitsu, // closed 6 / open 5
    Yakuman(YakumanKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YakuhaiKind {
    Haku,
    Hatsu,
    Chun,
    Bakaze,
    Jikaze,
    /// North as yakuhai (optional 3-player rule; never passed in 4-player).
    Pei,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YakumanKind {
    KokushiMusou,
    KokushiMusouJuusanmen,
    Suuankou,
    SuuankouTanki,
    Daisangen,
    Shousuushi,
    Daisuushi,
    Tsuuiisou,
    Chinroutou,
    Ryuuiisou,
    ChuurenPoutou,
    JunseiChuuren,
    Suukantsu,
    Tenhou,
    Chiihou,
}

impl Yaku {
    /// Romanized yaku name for CLI and report output. Not part of any protocol field.
    pub fn name(self) -> &'static str {
        match self {
            Yaku::Riichi => "Riichi",
            Yaku::Ippatsu => "Ippatsu",
            Yaku::MenzenTsumo => "Menzen Tsumo",
            Yaku::Pinfu => "Pinfu",
            Yaku::Tanyao => "Tanyao",
            Yaku::Iipeikou => "Iipeikou",
            Yaku::Yakuhai(k) => k.name(),
            Yaku::Haitei => "Haitei",
            Yaku::Houtei => "Houtei",
            Yaku::Rinshan => "Rinshan Kaihou",
            Yaku::Chankan => "Chankan",
            Yaku::DoubleRiichi => "Double Riichi",
            Yaku::Chiitoitsu => "Chiitoitsu",
            Yaku::Sanshoku => "Sanshoku Doujun",
            Yaku::SanshokuDoukou => "Sanshoku Doukou",
            Yaku::Ittsuu => "Ittsuu",
            Yaku::Chanta => "Chanta",
            Yaku::Toitoi => "Toitoi",
            Yaku::Sanankou => "Sanankou",
            Yaku::Sankantsu => "Sankantsu",
            Yaku::Honroutou => "Honroutou",
            Yaku::Shousangen => "Shousangen",
            Yaku::Honitsu => "Honitsu",
            Yaku::Junchan => "Junchan",
            Yaku::Ryanpeikou => "Ryanpeikou",
            Yaku::Chinitsu => "Chinitsu",
            Yaku::Yakuman(k) => k.name(),
        }
    }
}

impl YakuhaiKind {
    pub fn name(self) -> &'static str {
        match self {
            YakuhaiKind::Haku => "Yakuhai Haku",
            YakuhaiKind::Hatsu => "Yakuhai Hatsu",
            YakuhaiKind::Chun => "Yakuhai Chun",
            YakuhaiKind::Bakaze => "Yakuhai Round Wind",
            YakuhaiKind::Jikaze => "Yakuhai Seat Wind",
            YakuhaiKind::Pei => "Yakuhai North",
        }
    }
}

impl YakumanKind {
    pub fn name(self) -> &'static str {
        match self {
            YakumanKind::KokushiMusou => "Kokushi Musou",
            YakumanKind::KokushiMusouJuusanmen => "Kokushi Musou 13-sided",
            YakumanKind::Suuankou => "Suuankou",
            YakumanKind::SuuankouTanki => "Suuankou Tanki",
            YakumanKind::Daisangen => "Daisangen",
            YakumanKind::Shousuushi => "Shousuushi",
            YakumanKind::Daisuushi => "Daisuushi",
            YakumanKind::Tsuuiisou => "Tsuuiisou",
            YakumanKind::Chinroutou => "Chinroutou",
            YakumanKind::Ryuuiisou => "Ryuuiisou",
            YakumanKind::ChuurenPoutou => "Chuuren Poutou",
            YakumanKind::JunseiChuuren => "Junsei Chuuren Poutou",
            YakumanKind::Suukantsu => "Suukantsu",
            YakumanKind::Tenhou => "Tenhou",
            YakumanKind::Chiihou => "Chiihou",
        }
    }
}

/// Win context: situational flags and table state.
#[derive(Debug, Clone)]
pub struct YakuContext {
    /// Platform rules; the seat-count variant alone does not determine them.
    pub rule_profile: RiichiRuleProfile,
    /// Concealed counts including the winning tile, red fives folded.
    pub concealed: [u8; 34],
    /// Called melds and kans. Nukidora are counted separately in `nuki_count`.
    pub melds: Vec<Meld>,
    /// Winning tile, red five folded.
    pub win_tile: Tile,
    pub is_tsumo: bool,
    /// Closed hand (closed kans still count as closed).
    pub menzen: bool,
    /// Round wind tile (e.g. East = 27).
    pub bakaze: Tile,
    pub jikaze: Tile,
    // Situational yaku flags.
    pub riichi: bool,
    pub double_riichi: bool,
    pub ippatsu: bool,
    pub haitei: bool,
    pub houtei: bool,
    pub rinshan: bool,
    pub chankan: bool,
    pub tenhou: bool,
    pub chiihou: bool,
    /// Number of nukidora (3-player; always 0 in 4-player). Adds dora han, not a yaku.
    pub nuki_count: u8,
    /// North counts as yakuhai (optional 3-player rule).
    pub pei_is_yakuhai: bool,
}

impl YakuContext {
    /// Every group (called and concealed) as `(shape, concealed)` pairs for uniform
    /// yaku checks. Only meaningful for standard hands.
    fn all_melds(&self, parse: &StandardParse) -> Vec<(MeldShape, bool)> {
        let mut v: Vec<(MeldShape, bool)> = parse
            .melds
            .iter()
            .map(|&m| (m, true)) // concealed groups start out closed
            .collect();
        // On ron, a triplet becomes open only when this reading completes it with the
        // winning tile as a dual pon wait. If the best reading puts the winning tile in a
        // sequence, a same-kind concealed triplet stays concealed (affects sanankou,
        // suuankou and fu).
        if !self.is_tsumo
            && wait_shape(parse, self.win_tile) == Some(WaitShape::Shanpon)
            && let Some(slot) = v.iter_mut().find(|(m, c)| {
                *c && matches!(m, MeldShape::Kotsu(k) if *k == self.win_tile.kind() as u8)
            })
        {
            slot.1 = false;
        }
        for m in &self.melds {
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

    /// Number of kans of any type.
    fn kan_count(&self) -> usize {
        self.melds.iter().filter(|m| m.is_kan()).count()
    }
}

/// Yaku result of a single win.
#[derive(Debug, Clone, PartialEq)]
pub struct YakuResult {
    pub yaku: Vec<(Yaku, u8)>,
    pub han: u8,
    /// Yakuman multiplier (0 = not yakuman). With yakuman, `han` ignores regular yaku.
    pub yakuman: u8,
}

impl YakuResult {
    fn empty() -> Self {
        Self {
            yaku: Vec::new(),
            han: 0,
            yakuman: 0,
        }
    }
}

/// Evaluates the yaku under `ctx`, taking the highest-han decomposition.
pub fn evaluate(ctx: &YakuContext) -> YakuResult {
    let decomp = crate::decompose::decompose(&ctx.concealed, ctx.melds_set_count());

    // Yakuman take precedence over regular yaku.
    if let Some(ym) = eval_yakuman(ctx, &decomp) {
        return ym;
    }

    match &decomp {
        Decomposition::Chiitoitsu => {
            let mut best = eval_chiitoitsu(ctx);
            for parse in crate::decompose::standard_parses(&ctx.concealed, ctx.melds_set_count()) {
                let r = eval_standard(ctx, &parse);
                if r.han > best.han {
                    best = r;
                }
            }
            best
        }
        Decomposition::Kokushi => YakuResult::empty(), // yakuman, handled above
        Decomposition::Standard(parses) => {
            let mut best = YakuResult::empty();
            for parse in parses {
                let r = eval_standard(ctx, parse);
                if r.han > best.han {
                    best = r;
                }
            }
            // A hand may be both chiitoitsu and standard; take the higher.
            best
        }
    }
}

impl YakuContext {
    /// Number of melds that count as groups (nukidora excluded).
    fn melds_set_count(&self) -> u8 {
        self.melds
            .iter()
            .filter(|m| !matches!(m, Meld::Nukidora { .. }))
            .count() as u8
    }

    /// Public wrapper used by the score module.
    pub fn melds_set_count_pub(&self) -> u8 {
        self.melds_set_count()
    }
}

/// Standard-form yaku evaluation for the score module, per decomposition.
pub fn eval_standard_pub(ctx: &YakuContext, parse: &StandardParse) -> YakuResult {
    eval_standard(ctx, parse)
}

/// Yakuman check. `Some` means the hand is a yakuman.
fn eval_yakuman(ctx: &YakuContext, decomp: &Decomposition) -> Option<YakuResult> {
    // The hand must be complete first.
    //
    // Tenhou / chiihou and tsuuiisou / chinroutou / ryuuiisou do not depend on the
    // decomposition (they look at flags or the whole hand). Without this check an
    // incomplete hand like `z1111 z333 z555 z6666` would count as tsuuiisou.
    // `decompose` returns `Standard(vec![])` for incomplete hands.
    let is_agari_shape = match decomp {
        Decomposition::Kokushi | Decomposition::Chiitoitsu => true,
        Decomposition::Standard(parses) => !parses.is_empty(),
    };
    if !is_agari_shape {
        return None;
    }

    let mut yakuman = Vec::new();

    if ctx.tenhou {
        yakuman.push(YakumanKind::Tenhou);
    }
    if ctx.chiihou {
        yakuman.push(YakumanKind::Chiihou);
    }

    // Kokushi; the 13-sided wait is kept as a separate kind for display and platform rules.
    if matches!(decomp, Decomposition::Kokushi) {
        yakuman.push(if is_kokushi_thirteen_wait(ctx) {
            YakumanKind::KokushiMusouJuusanmen
        } else {
            YakumanKind::KokushiMusou
        });
    }

    // Whole-hand yakuman, independent of decomposition.
    let all = all_tiles(ctx);
    if !all.is_empty() {
        if all.iter().all(|t| t.is_honor()) {
            yakuman.push(YakumanKind::Tsuuiisou);
        }
        if all.iter().all(|t| t.is_terminal()) {
            yakuman.push(YakumanKind::Chinroutou);
        }
        if all.iter().all(|t| is_green(*t)) {
            yakuman.push(YakumanKind::Ryuuiisou);
        }
    }

    // Triplet-based yakuman.
    if let Decomposition::Standard(parses) = decomp {
        for parse in parses {
            let melds = ctx.all_melds(parse);
            let kotsu_kinds: Vec<u8> = melds
                .iter()
                .filter(|(m, _)| m.is_kotsu())
                .map(|(m, _)| m.tile_kind())
                .collect();
            let dragons = kotsu_kinds
                .iter()
                .filter(|&&k| (31..=33).contains(&k))
                .count();
            if dragons == 3 {
                push_unique(&mut yakuman, YakumanKind::Daisangen);
            }
            let winds = kotsu_kinds
                .iter()
                .filter(|&&k| (27..=30).contains(&k))
                .count();
            let wind_pair = (27..=30).contains(&parse.pair);
            if winds == 4 {
                push_unique(&mut yakuman, YakumanKind::Daisuushi);
            } else if winds == 3 && wind_pair {
                push_unique(&mut yakuman, YakumanKind::Shousuushi);
            }
            let ankou = melds
                .iter()
                .filter(|(m, concealed)| m.is_kotsu() && *concealed)
                .count();
            if ankou >= 4 {
                // A pair wait upgrades to suuankou tanki.
                if parse.pair == ctx.win_tile.kind() as u8 {
                    push_unique(&mut yakuman, YakumanKind::SuuankouTanki);
                } else if ctx.is_tsumo {
                    push_unique(&mut yakuman, YakumanKind::Suuankou);
                }
                // A triplet completed by ron is open, so a non-tanki ron never makes suuankou (simplified: tsumo only).
            }
            if ctx.kan_count() == 4 {
                push_unique(&mut yakuman, YakumanKind::Suukantsu);
            }
        }
        // Chuuren poutou (closed, specific single-suit shape).
        if ctx.menzen
            && let Some(jun) = chuuren(ctx)
        {
            push_unique(&mut yakuman, jun);
        }
    }

    if yakuman.is_empty() {
        return None;
    }
    let mult = yakuman
        .iter()
        .map(|&kind| yakuman_weight(ctx.rule_profile, kind))
        .sum();
    Some(YakuResult {
        // The second tuple element is always regular han; the multiplier is only in `yakuman`.
        yaku: yakuman
            .iter()
            .map(|&kind| (Yaku::Yakuman(kind), 0))
            .collect(),
        han: 0,
        yakuman: mult,
    })
}

fn yakuman_weight(profile: RiichiRuleProfile, kind: YakumanKind) -> u8 {
    if !profile.double_special_yakuman() {
        return 1;
    }
    match kind {
        YakumanKind::SuuankouTanki
        | YakumanKind::JunseiChuuren
        | YakumanKind::Daisuushi
        | YakumanKind::KokushiMusouJuusanmen => 2,
        _ => 1,
    }
}

/// 13-sided kokushi: without the winning tile, exactly one of each terminal and honor.
fn is_kokushi_thirteen_wait(ctx: &YakuContext) -> bool {
    let win = ctx.win_tile.deaka();
    if !win.is_yaochuu() {
        return false;
    }
    let win_kind = win.kind();
    let mut concealed = ctx.concealed;
    if concealed[win_kind] == 0 {
        return false;
    }
    concealed[win_kind] -= 1;
    const YAOCHUU: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    YAOCHUU.iter().all(|&kind| concealed[kind] == 1)
        && (0..34).all(|kind| concealed[kind] == 0 || YAOCHUU.contains(&kind))
}

/// Standard-form yaku (four groups and a pair).
fn eval_standard(ctx: &YakuContext, parse: &StandardParse) -> YakuResult {
    let mut yaku: Vec<(Yaku, u8)> = Vec::new();
    let melds = ctx.all_melds(parse);
    let all = all_tiles(ctx);
    let menzen = ctx.menzen;

    // Situational yaku
    if ctx.double_riichi {
        yaku.push((Yaku::DoubleRiichi, 2));
    } else if ctx.riichi {
        yaku.push((Yaku::Riichi, 1));
    }
    if ctx.ippatsu && (ctx.riichi || ctx.double_riichi) {
        yaku.push((Yaku::Ippatsu, 1));
    }
    if ctx.is_tsumo && menzen {
        yaku.push((Yaku::MenzenTsumo, 1));
    }
    // Tenhou rules: haitei and rinshan do not combine; when the rinshan tile is the
    // last draw, only rinshan kaihou counts.
    if ctx.haitei && !ctx.rinshan {
        yaku.push((Yaku::Haitei, 1));
    }
    if ctx.houtei {
        yaku.push((Yaku::Houtei, 1));
    }
    if ctx.rinshan {
        yaku.push((Yaku::Rinshan, 1));
    }
    if ctx.chankan {
        yaku.push((Yaku::Chankan, 1));
    }

    // Yakuhai (dragons, round wind, seat wind)
    for (m, _) in &melds {
        if let MeldShape::Kotsu(k) = m {
            let t = unsafe { Tile::from_id_unchecked(*k) };
            if t.is_dragon() {
                let kind = match *k {
                    31 => YakuhaiKind::Haku,
                    32 => YakuhaiKind::Hatsu,
                    _ => YakuhaiKind::Chun,
                };
                yaku.push((Yaku::Yakuhai(kind), 1));
            } else if t.is_wind() {
                if *k == ctx.bakaze.kind() as u8 {
                    yaku.push((Yaku::Yakuhai(YakuhaiKind::Bakaze), 1));
                }
                if *k == ctx.jikaze.kind() as u8 {
                    yaku.push((Yaku::Yakuhai(YakuhaiKind::Jikaze), 1));
                }
                // 3-player: North as yakuhai (optional)
                if ctx.pei_is_yakuhai && *k == 30 {
                    yaku.push((Yaku::Yakuhai(YakuhaiKind::Pei), 1));
                }
            }
        }
    }

    // Pinfu: closed, all sequences, non-yakuhai pair, two-sided wait
    if menzen && melds.iter().all(|(m, _)| m.is_shuntsu()) {
        let pair_t = unsafe { Tile::from_id_unchecked(parse.pair) };
        let pair_is_yakuhai = pair_t.is_dragon()
            || parse.pair == ctx.bakaze.kind() as u8
            || parse.pair == ctx.jikaze.kind() as u8;
        if !pair_is_yakuhai && wait_shape(parse, ctx.win_tile) == Some(WaitShape::Ryanmen) {
            yaku.push((Yaku::Pinfu, 1));
        }
    }

    if all.iter().all(|t| !t.is_yaochuu()) && (menzen || ctx.rule_profile.allows_open_tanyao()) {
        yaku.push((Yaku::Tanyao, 1));
    }

    // Iipeikou / ryanpeikou (closed only)
    if menzen {
        let shuntsu: Vec<u8> = melds
            .iter()
            .filter_map(|(m, _)| match m {
                MeldShape::Shuntsu(k) => Some(*k),
                _ => None,
            })
            .collect();
        let dups = count_duplicate_pairs(&shuntsu);
        if dups == 2 {
            yaku.push((Yaku::Ryanpeikou, 3));
        } else if dups == 1 {
            yaku.push((Yaku::Iipeikou, 1));
        }
    }

    if has_sanshoku_shuntsu(&melds) {
        yaku.push((Yaku::Sanshoku, if menzen { 2 } else { 1 }));
    }
    if has_sanshoku_doukou(&melds) {
        yaku.push((Yaku::SanshokuDoukou, 2));
    }
    if has_ittsuu(&melds) {
        yaku.push((Yaku::Ittsuu, if menzen { 2 } else { 1 }));
    }

    if melds.iter().all(|(m, _)| m.is_kotsu()) {
        yaku.push((Yaku::Toitoi, 2));
    }
    let ankou = melds.iter().filter(|(m, c)| m.is_kotsu() && *c).count();
    if ankou == 3 {
        yaku.push((Yaku::Sanankou, 2));
    }
    if ctx.kan_count() == 3 {
        yaku.push((Yaku::Sankantsu, 2));
    }

    // Chanta / junchan / honroutou
    let all_groups_have_yaochuu =
        melds.iter().all(|(m, _)| meld_has_yaochuu(*m)) && parse_pair_yaochuu(parse);
    if all_groups_have_yaochuu {
        let has_honor = all.iter().any(|t| t.is_honor());
        let has_shuntsu = melds.iter().any(|(m, _)| m.is_shuntsu());
        if !has_shuntsu {
            // All terminal/honor triplets and pair: honroutou (2 han)
            yaku.push((Yaku::Honroutou, 2));
        } else if has_honor {
            yaku.push((Yaku::Chanta, if menzen { 2 } else { 1 }));
        } else {
            yaku.push((Yaku::Junchan, if menzen { 3 } else { 2 }));
        }
    }

    if is_shousangen(&melds, parse) {
        yaku.push((Yaku::Shousangen, 2));
    }

    // Honitsu / chinitsu
    match flush_kind(&all) {
        Some(true) => yaku.push((Yaku::Chinitsu, if menzen { 6 } else { 5 })),
        Some(false) => yaku.push((Yaku::Honitsu, if menzen { 3 } else { 2 })),
        None => {}
    }

    let han: u8 = yaku.iter().map(|(_, h)| h).sum();
    YakuResult {
        yaku,
        han,
        yakuman: 0,
    }
}

/// Chiitoitsu yaku evaluation.
pub(crate) fn eval_chiitoitsu(ctx: &YakuContext) -> YakuResult {
    let mut yaku: Vec<(Yaku, u8)> = vec![(Yaku::Chiitoitsu, 2)];
    if ctx.double_riichi {
        yaku.push((Yaku::DoubleRiichi, 2));
    } else if ctx.riichi {
        yaku.push((Yaku::Riichi, 1));
    }
    if ctx.ippatsu && (ctx.riichi || ctx.double_riichi) {
        yaku.push((Yaku::Ippatsu, 1));
    }
    if ctx.is_tsumo {
        yaku.push((Yaku::MenzenTsumo, 1));
    }
    // Tenhou rules: haitei and rinshan do not combine.
    if ctx.haitei && !ctx.rinshan {
        yaku.push((Yaku::Haitei, 1));
    }
    if ctx.houtei {
        yaku.push((Yaku::Houtei, 1));
    }
    if ctx.rinshan {
        yaku.push((Yaku::Rinshan, 1));
    }
    if ctx.chankan {
        yaku.push((Yaku::Chankan, 1));
    }
    let all = all_tiles(ctx);
    if all.iter().all(|t| !t.is_yaochuu()) {
        yaku.push((Yaku::Tanyao, 1));
    }
    // Honroutou combines with chiitoitsu (all terminal/honor pairs).
    if all.iter().all(|t| t.is_yaochuu()) {
        yaku.push((Yaku::Honroutou, 2));
    }
    match flush_kind(&all) {
        Some(true) => yaku.push((Yaku::Chinitsu, 6)),
        Some(false) => yaku.push((Yaku::Honitsu, 3)),
        None => {}
    }
    let han = yaku.iter().map(|(_, h)| h).sum();
    YakuResult {
        yaku,
        han,
        yakuman: 0,
    }
}

/// All tiles in the hand (concealed and melds), red fives folded.
fn all_tiles(ctx: &YakuContext) -> Vec<Tile> {
    let mut v: Vec<Tile> = Vec::new();
    for k in 0..34usize {
        for _ in 0..ctx.concealed[k] {
            v.push(unsafe { Tile::from_id_unchecked(k as u8) });
        }
    }
    for m in &ctx.melds {
        match m {
            Meld::Nukidora { .. } => {}
            _ => {
                for t in m.tiles() {
                    v.push(t.deaka());
                }
            }
        }
    }
    v
}

/// Tiles allowed in ryuuiisou: 2s 3s 4s 6s 8s and hatsu.
fn is_green(t: Tile) -> bool {
    matches!(t.kind(), 19 | 20 | 21 | 23 | 25 | 32)
}

fn push_unique(v: &mut Vec<YakumanKind>, k: YakumanKind) {
    if !v.contains(&k) {
        v.push(k);
    }
}

/// Chuuren poutou check (closed single-suit shape). Returns junsei or regular.
/// Nukidora do not break the closed hand and are not melds.
fn chuuren(ctx: &YakuContext) -> Option<YakumanKind> {
    if ctx
        .melds
        .iter()
        .any(|m| !matches!(m, Meld::Nukidora { .. }))
    {
        return None;
    }
    let all = all_tiles(ctx);
    let suit = all.first()?.suit();
    if suit == crate::tile::Suit::Honor {
        return None;
    }
    if !all.iter().all(|t| t.suit() == suit) {
        return None;
    }
    let base = match suit {
        crate::tile::Suit::Man => 0,
        crate::tile::Suit::Pin => 9,
        crate::tile::Suit::Sou => 18,
        _ => return None,
    };
    let mut cnt = [0u8; 9];
    for t in &all {
        cnt[t.kind() - base] += 1;
    }
    // 1112345678999 plus one extra tile
    let need = [3, 1, 1, 1, 1, 1, 1, 1, 3];
    let mut extra = None;
    for i in 0..9 {
        if cnt[i] < need[i] {
            return None;
        }
        if cnt[i] == need[i] + 1 {
            if extra.is_some() {
                return None;
            }
            extra = Some(i);
        } else if cnt[i] != need[i] {
            return None;
        }
    }
    extra?;
    // Junsei chuuren: removing the winning tile leaves exactly 1112345678999 (nine-sided wait).
    let win_idx = ctx.win_tile.kind().checked_sub(base)?;
    let mut without = cnt;
    without[win_idx] -= 1;
    if without == need {
        Some(YakumanKind::JunseiChuuren)
    } else {
        Some(YakumanKind::ChuurenPoutou)
    }
}

/// Number of duplicate sequence pairs (for iipeikou / ryanpeikou).
fn count_duplicate_pairs(shuntsu: &[u8]) -> u8 {
    let mut counts = std::collections::HashMap::new();
    for &s in shuntsu {
        *counts.entry(s).or_insert(0u8) += 1;
    }
    let mut pairs = 0;
    for &c in counts.values() {
        pairs += c / 2;
    }
    pairs
}

/// Sanshoku doujun: the same sequence in all three suits.
fn has_sanshoku_shuntsu(melds: &[(MeldShape, bool)]) -> bool {
    let shuntsu: Vec<u8> = melds
        .iter()
        .filter_map(|(m, _)| match m {
            MeldShape::Shuntsu(k) => Some(*k),
            _ => None,
        })
        .collect();
    for &s in &shuntsu {
        if s >= 18 {
            continue;
        }
        let r = s % 9;
        let m = r;
        let p = 9 + r;
        let so = 18 + r;
        if shuntsu.contains(&m) && shuntsu.contains(&p) && shuntsu.contains(&so) {
            return true;
        }
    }
    false
}

fn has_sanshoku_doukou(melds: &[(MeldShape, bool)]) -> bool {
    let kotsu: Vec<u8> = melds
        .iter()
        .filter_map(|(m, _)| match m {
            MeldShape::Kotsu(k) if *k < 27 => Some(*k),
            _ => None,
        })
        .collect();
    for &k in &kotsu {
        let r = k % 9;
        if kotsu.contains(&r) && kotsu.contains(&(9 + r)) && kotsu.contains(&(18 + r)) {
            return true;
        }
    }
    false
}

/// Ittsuu: 123, 456 and 789 in one suit.
fn has_ittsuu(melds: &[(MeldShape, bool)]) -> bool {
    let shuntsu: Vec<u8> = melds
        .iter()
        .filter_map(|(m, _)| match m {
            MeldShape::Shuntsu(k) => Some(*k),
            _ => None,
        })
        .collect();
    for base in [0u8, 9, 18] {
        if shuntsu.contains(&base) && shuntsu.contains(&(base + 3)) && shuntsu.contains(&(base + 6))
        {
            return true;
        }
    }
    false
}

/// Whether a group contains a terminal or honor.
fn meld_has_yaochuu(m: MeldShape) -> bool {
    match m {
        MeldShape::Kotsu(k) => {
            let t = unsafe { Tile::from_id_unchecked(k) };
            t.is_yaochuu()
        }
        MeldShape::Shuntsu(s) => s % 9 == 0 || s % 9 == 6, // 123 or 789
    }
}

fn parse_pair_yaochuu(parse: &StandardParse) -> bool {
    unsafe { Tile::from_id_unchecked(parse.pair) }.is_yaochuu()
}

/// Shousangen: two dragon triplets plus a dragon pair.
fn is_shousangen(melds: &[(MeldShape, bool)], parse: &StandardParse) -> bool {
    let dragon_kotsu = melds
        .iter()
        .filter(|(m, _)| matches!(m, MeldShape::Kotsu(k) if (31..=33).contains(k)))
        .count();
    let pair_dragon = (31..=33).contains(&parse.pair);
    dragon_kotsu == 2 && pair_dragon
}

/// Flush check: `Some(true)` chinitsu, `Some(false)` honitsu, `None` neither.
fn flush_kind(all: &[Tile]) -> Option<bool> {
    let mut suits = std::collections::HashSet::new();
    let mut has_honor = false;
    for t in all {
        if t.is_honor() {
            has_honor = true;
        } else {
            suits.insert(t.suit());
        }
    }
    if suits.len() == 1 {
        Some(!has_honor)
    } else {
        None
    }
}
