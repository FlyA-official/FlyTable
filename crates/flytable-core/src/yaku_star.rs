//! Lightweight yaku star map.
//!
//! Outputs only facts the rules determine: three shanten tracks, ukeire, a wait
//! bitmap, and for each of the 42 yaku whether it is structurally reachable and
//! whether it is confirmed on every wait. No per-yaku exact shanten, probabilities
//! or tensor encoding.

use crate::agari;
use crate::hand::TileCounts;
use crate::meld::Meld;
use crate::rules::RiichiRuleProfile;
use crate::shanten;
use crate::tile::{Suit, Tile};
use crate::yaku::{Yaku, YakuContext, YakuhaiKind, YakumanKind, evaluate};

pub const YAKU_STAR_COUNT: usize = 42;

/// The order of the 42 slots is part of the output contract; do not reorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum YakuStarId {
    MenzenTsumo = 0,
    Riichi = 1,
    Ippatsu = 2,
    DoubleRiichi = 3,
    Haitei = 4,
    Houtei = 5,
    Rinshan = 6,
    Chankan = 7,
    Tanyao = 8,
    Pinfu = 9,
    Iipeikou = 10,
    Ryanpeikou = 11,
    SanshokuDoujun = 12,
    Ittsuu = 13,
    Chanta = 14,
    Junchan = 15,
    Honitsu = 16,
    Chinitsu = 17,
    Chuuren = 18,
    YakuhaiRound = 19,
    YakuhaiSeat = 20,
    Haku = 21,
    Hatsu = 22,
    Chun = 23,
    SanshokuDoukou = 24,
    Toitoi = 25,
    Sanankou = 26,
    Suuankou = 27,
    Sankantsu = 28,
    Suukantsu = 29,
    Chiitoitsu = 30,
    Honroutou = 31,
    Chinroutou = 32,
    Kokushi = 33,
    Shousangen = 34,
    Daisangen = 35,
    Shousuushi = 36,
    Daisuushi = 37,
    Tsuuiisou = 38,
    Ryuuiisou = 39,
    Tenhou = 40,
    Chiihou = 41,
}

impl YakuStarId {
    pub const ALL: [Self; YAKU_STAR_COUNT] = [
        Self::MenzenTsumo,
        Self::Riichi,
        Self::Ippatsu,
        Self::DoubleRiichi,
        Self::Haitei,
        Self::Houtei,
        Self::Rinshan,
        Self::Chankan,
        Self::Tanyao,
        Self::Pinfu,
        Self::Iipeikou,
        Self::Ryanpeikou,
        Self::SanshokuDoujun,
        Self::Ittsuu,
        Self::Chanta,
        Self::Junchan,
        Self::Honitsu,
        Self::Chinitsu,
        Self::Chuuren,
        Self::YakuhaiRound,
        Self::YakuhaiSeat,
        Self::Haku,
        Self::Hatsu,
        Self::Chun,
        Self::SanshokuDoukou,
        Self::Toitoi,
        Self::Sanankou,
        Self::Suuankou,
        Self::Sankantsu,
        Self::Suukantsu,
        Self::Chiitoitsu,
        Self::Honroutou,
        Self::Chinroutou,
        Self::Kokushi,
        Self::Shousangen,
        Self::Daisangen,
        Self::Shousuushi,
        Self::Daisuushi,
        Self::Tsuuiisou,
        Self::Ryuuiisou,
        Self::Tenhou,
        Self::Chiihou,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Yaku status; precedence is `Blocked > Confirmed > Possible`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YakuStarStatus {
    Blocked,
    Possible,
    Confirmed,
    NotApplicable,
}

/// Hand-wide standard-form facts.
///
/// The chiitoitsu and kokushi tracks are [`i8::MAX`] when the hand is open or has
/// called groups, meaning the track is unavailable. Nukidora neither count as groups
/// nor open the hand. `wait_mask` is non-zero only when `shanten_min == 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StandardShapeFacts {
    pub shanten_normal: i8,
    pub shanten_chiitoi: i8,
    pub shanten_kokushi: i8,
    pub shanten_min: i8,
    pub ukeire_kinds: u8,
    pub ukeire_unseen: u16,
    pub wait_mask: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YakuStarRow {
    pub status: YakuStarStatus,
    pub structural_reachable: bool,
    pub static_identity_han: u8,
    pub ron_wait_mask: u64,
    pub tsumo_wait_mask: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YakuStarState {
    pub facts: StandardShapeFacts,
    pub rows: [YakuStarRow; YAKU_STAR_COUNT],
}

/// Table facts needed by the star map. 4-player callers set `nuki_count` to 0 and
/// `pei_is_yakuhai` to false.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YakuStarContext {
    pub rule_profile: RiichiRuleProfile,
    pub bakaze: Tile,
    pub jikaze: Tile,
    pub menzen: bool,
    pub total_declared_kans: u8,
    pub has_legal_kan_candidate: bool,
    pub riichi: bool,
    pub double_riichi: bool,
    pub ippatsu: bool,
    pub haitei: bool,
    pub houtei: bool,
    pub rinshan: bool,
    pub chankan: bool,
    /// Tenhou still possible (the dealer has not discarded yet and no call interrupted the first turn).
    pub tenhou: bool,
    /// Chiihou still possible (the player has not discarded yet and no call interrupted the first turn).
    pub chiihou: bool,
    pub nuki_count: u8,
    pub pei_is_yakuhai: bool,
}

/// Maps a yaku or yakuman to its star-map slot. The optional 3-player North yakuhai has no slot and returns `None`.
pub const fn yaku_star_id(yaku: Yaku) -> Option<YakuStarId> {
    Some(match yaku {
        Yaku::MenzenTsumo => YakuStarId::MenzenTsumo,
        Yaku::Riichi => YakuStarId::Riichi,
        Yaku::Ippatsu => YakuStarId::Ippatsu,
        Yaku::DoubleRiichi => YakuStarId::DoubleRiichi,
        Yaku::Haitei => YakuStarId::Haitei,
        Yaku::Houtei => YakuStarId::Houtei,
        Yaku::Rinshan => YakuStarId::Rinshan,
        Yaku::Chankan => YakuStarId::Chankan,
        Yaku::Tanyao => YakuStarId::Tanyao,
        Yaku::Pinfu => YakuStarId::Pinfu,
        Yaku::Iipeikou => YakuStarId::Iipeikou,
        Yaku::Ryanpeikou => YakuStarId::Ryanpeikou,
        Yaku::Sanshoku => YakuStarId::SanshokuDoujun,
        Yaku::Ittsuu => YakuStarId::Ittsuu,
        Yaku::Chanta => YakuStarId::Chanta,
        Yaku::Junchan => YakuStarId::Junchan,
        Yaku::Honitsu => YakuStarId::Honitsu,
        Yaku::Chinitsu => YakuStarId::Chinitsu,
        Yaku::Yakuhai(YakuhaiKind::Bakaze) => YakuStarId::YakuhaiRound,
        Yaku::Yakuhai(YakuhaiKind::Jikaze) => YakuStarId::YakuhaiSeat,
        Yaku::Yakuhai(YakuhaiKind::Haku) => YakuStarId::Haku,
        Yaku::Yakuhai(YakuhaiKind::Hatsu) => YakuStarId::Hatsu,
        Yaku::Yakuhai(YakuhaiKind::Chun) => YakuStarId::Chun,
        Yaku::Yakuhai(YakuhaiKind::Pei) => return None,
        Yaku::SanshokuDoukou => YakuStarId::SanshokuDoukou,
        Yaku::Toitoi => YakuStarId::Toitoi,
        Yaku::Sanankou => YakuStarId::Sanankou,
        Yaku::Sankantsu => YakuStarId::Sankantsu,
        Yaku::Chiitoitsu => YakuStarId::Chiitoitsu,
        Yaku::Honroutou => YakuStarId::Honroutou,
        Yaku::Shousangen => YakuStarId::Shousangen,
        Yaku::Yakuman(kind) => yakuman_star_id(kind),
    })
}

/// Double-yakuman variants share the slot of their base yakuman.
pub const fn yakuman_star_id(kind: YakumanKind) -> YakuStarId {
    match kind {
        YakumanKind::KokushiMusou | YakumanKind::KokushiMusouJuusanmen => YakuStarId::Kokushi,
        YakumanKind::Suuankou | YakumanKind::SuuankouTanki => YakuStarId::Suuankou,
        YakumanKind::Daisangen => YakuStarId::Daisangen,
        YakumanKind::Shousuushi => YakuStarId::Shousuushi,
        YakumanKind::Daisuushi => YakuStarId::Daisuushi,
        YakumanKind::Tsuuiisou => YakuStarId::Tsuuiisou,
        YakumanKind::Chinroutou => YakuStarId::Chinroutou,
        YakumanKind::Ryuuiisou => YakuStarId::Ryuuiisou,
        YakumanKind::ChuurenPoutou | YakumanKind::JunseiChuuren => YakuStarId::Chuuren,
        YakumanKind::Suukantsu => YakuStarId::Suukantsu,
        YakumanKind::Tenhou => YakuStarId::Tenhou,
        YakumanKind::Chiihou => YakuStarId::Chiihou,
    }
}

/// 3-player keeps all 42 rows; no slot is permanently `NotApplicable` at present.
pub const fn yaku_3p_not_applicable(_id: YakuStarId) -> bool {
    false
}

pub const fn static_identity_han(id: YakuStarId) -> u8 {
    match id {
        YakuStarId::DoubleRiichi
        | YakuStarId::Honitsu
        | YakuStarId::Toitoi
        | YakuStarId::Sanankou
        | YakuStarId::Sankantsu
        | YakuStarId::Chanta
        | YakuStarId::Shousangen
        | YakuStarId::Honroutou => 2,
        YakuStarId::Ryanpeikou => 3,
        YakuStarId::Chinitsu => 5,
        YakuStarId::Chuuren
        | YakuStarId::Suuankou
        | YakuStarId::Suukantsu
        | YakuStarId::Chinroutou
        | YakuStarId::Kokushi
        | YakuStarId::Daisangen
        | YakuStarId::Shousuushi
        | YakuStarId::Daisuushi
        | YakuStarId::Tsuuiisou
        | YakuStarId::Ryuuiisou
        | YakuStarId::Tenhou
        | YakuStarId::Chiihou => 13,
        _ => 1,
    }
}

/// 4-player yaku star map.
pub fn yaku_star(
    concealed: &TileCounts,
    tiles_seen: &TileCounts,
    melds: &[Meld],
    ctx: &YakuStarContext,
) -> YakuStarState {
    yaku_star_for_rule_line(concealed, tiles_seen, melds, ctx, false)
}

/// 3-player yaku star map; ukeire and waits exclude 2m..8m. Melds must not contain chi.
pub fn yaku_star_3p(
    concealed: &TileCounts,
    tiles_seen: &TileCounts,
    melds: &[Meld],
    ctx: &YakuStarContext,
) -> YakuStarState {
    yaku_star_for_rule_line(concealed, tiles_seen, melds, ctx, true)
}

fn yaku_star_for_rule_line(
    concealed: &TileCounts,
    tiles_seen: &TileCounts,
    melds: &[Meld],
    ctx: &YakuStarContext,
    is_sanma: bool,
) -> YakuStarState {
    let meld_facts = MeldFacts { melds };
    let fixed_count = meld_facts.fixed_count() as u8;
    let special_tracks_available = ctx.menzen && fixed_count == 0;
    let shanten_normal = if is_sanma {
        shanten::shanten_standard_3p(concealed, fixed_count)
    } else {
        shanten::shanten_standard(concealed, fixed_count)
    };
    let shanten_chiitoi = if special_tracks_available {
        shanten::shanten_chiitoitsu(concealed)
    } else {
        i8::MAX
    };
    let shanten_kokushi = if special_tracks_available {
        shanten::shanten_kokushi(concealed)
    } else {
        i8::MAX
    };
    let shanten_min = shanten_normal.min(shanten_chiitoi).min(shanten_kokushi);
    let ukeire = if is_sanma {
        agari::ukeire_3p(concealed, fixed_count)
    } else {
        agari::ukeire(concealed, fixed_count)
    };
    let ukeire_unseen = ukeire.iter().fold(0u16, |sum, &kind| {
        sum.saturating_add(u16::from(4u8.saturating_sub(tiles_seen.count(kind))))
    });
    let wait_mask = if shanten_min == shanten::TENPAI {
        ukeire.iter().fold(0u64, |mask, &kind| mask | 1 << kind)
    } else {
        0
    };
    let facts = StandardShapeFacts {
        shanten_normal,
        shanten_chiitoi,
        shanten_kokushi,
        shanten_min,
        ukeire_kinds: ukeire.len() as u8,
        ukeire_unseen,
        wait_mask,
    };

    let mut ron_masks = [0u64; YAKU_STAR_COUNT];
    let mut tsumo_masks = [0u64; YAKU_STAR_COUNT];
    if shanten_min == shanten::TENPAI {
        for kind in 0..34 {
            let bit = 1u64 << kind;
            if wait_mask & bit == 0 {
                continue;
            }
            mark_confirmed_slots(concealed, kind, melds, ctx, false, bit, &mut ron_masks);
            mark_confirmed_slots(concealed, kind, melds, ctx, true, bit, &mut tsumo_masks);
        }
    }

    let mut rows = [YakuStarRow {
        status: YakuStarStatus::Possible,
        structural_reachable: true,
        static_identity_han: 1,
        ron_wait_mask: 0,
        tsumo_wait_mask: 0,
    }; YAKU_STAR_COUNT];
    for id in YakuStarId::ALL {
        let row = &mut rows[id.index()];
        row.static_identity_han = static_identity_han(id);
        if is_sanma && yaku_3p_not_applicable(id) {
            row.status = YakuStarStatus::NotApplicable;
            row.structural_reachable = false;
        } else if structural_blocked(id, meld_facts, ctx) {
            row.status = YakuStarStatus::Blocked;
            row.structural_reachable = false;
        } else {
            row.ron_wait_mask = ron_masks[id.index()];
            row.tsumo_wait_mask = tsumo_masks[id.index()];
            if row.ron_wait_mask != 0 || row.tsumo_wait_mask != 0 {
                row.status = YakuStarStatus::Confirmed;
            }
        }
    }
    YakuStarState { facts, rows }
}

#[allow(clippy::too_many_arguments)]
fn mark_confirmed_slots(
    concealed: &TileCounts,
    kind: usize,
    melds: &[Meld],
    star: &YakuStarContext,
    is_tsumo: bool,
    bit: u64,
    masks: &mut [u64; YAKU_STAR_COUNT],
) {
    if concealed.count(kind) >= 4 {
        return;
    }
    let win_tile = unsafe { Tile::from_id_unchecked(kind as u8) };
    let mut agari = *concealed;
    agari.add(win_tile, 1);
    let result = evaluate(&YakuContext {
        rule_profile: star.rule_profile,
        concealed: *agari.raw(),
        melds: melds.to_vec(),
        win_tile,
        is_tsumo,
        menzen: star.menzen,
        bakaze: star.bakaze,
        jikaze: star.jikaze,
        riichi: star.riichi,
        double_riichi: star.double_riichi,
        ippatsu: star.ippatsu,
        haitei: is_tsumo && star.haitei,
        houtei: !is_tsumo && star.houtei,
        rinshan: is_tsumo && star.rinshan,
        chankan: !is_tsumo && star.chankan,
        tenhou: is_tsumo && star.tenhou,
        chiihou: is_tsumo && star.chiihou,
        nuki_count: star.nuki_count,
        pei_is_yakuhai: star.pei_is_yakuhai,
    });
    if result.han == 0 && result.yakuman == 0 {
        return;
    }
    for &(yaku, _) in &result.yaku {
        if let Some(id) = yaku_star_id(yaku) {
            masks[id.index()] |= bit;
        }
    }
}

#[derive(Clone, Copy)]
struct MeldFacts<'a> {
    melds: &'a [Meld],
}

impl MeldFacts<'_> {
    fn fixed_count(self) -> usize {
        self.melds
            .iter()
            .filter(|meld| !matches!(meld, Meld::Nukidora { .. }))
            .count()
    }

    fn own_kans(self) -> usize {
        self.melds.iter().filter(|meld| meld.is_kan()).count()
    }

    fn ankans(self) -> usize {
        self.melds
            .iter()
            .filter(|meld| matches!(meld, Meld::Ankan { .. }))
            .count()
    }

    fn pons(self) -> usize {
        self.melds
            .iter()
            .filter(|meld| matches!(meld, Meld::Pon { .. }))
            .count()
    }

    fn has_chi(self) -> bool {
        self.melds
            .iter()
            .any(|meld| matches!(meld, Meld::Chi { .. }))
    }

    fn has_open_triplet_or_kan(self) -> bool {
        self.melds.iter().any(|meld| {
            matches!(
                meld,
                Meld::Pon { .. } | Meld::Daiminkan { .. } | Meld::Kakan { .. }
            )
        })
    }

    fn has_fixed_triplet_or_kan(self) -> bool {
        self.melds
            .iter()
            .any(|meld| matches!(meld, Meld::Pon { .. }) || meld.is_kan())
    }

    fn chi_any(self, pred: impl Fn(usize) -> bool) -> bool {
        self.melds.iter().any(|meld| match meld {
            Meld::Chi { tiles, .. } => pred(tiles.iter().map(|tile| tile.kind()).min().unwrap()),
            _ => false,
        })
    }

    fn triplet_kan_any(self, pred: impl Fn(Tile) -> bool) -> bool {
        self.melds.iter().any(|meld| match meld {
            Meld::Pon { tile, .. }
            | Meld::Daiminkan { tile, .. }
            | Meld::Kakan { tile, .. }
            | Meld::Ankan { tile } => pred(tile.deaka()),
            _ => false,
        })
    }

    fn fixed_tiles_any(self, pred: impl Fn(Tile) -> bool + Copy) -> bool {
        self.melds.iter().any(|meld| match meld {
            Meld::Chi { tiles, .. } => tiles.iter().copied().any(pred),
            Meld::Pon { tile, .. }
            | Meld::Daiminkan { tile, .. }
            | Meld::Kakan { tile, .. }
            | Meld::Ankan { tile } => pred(tile.deaka()),
            Meld::Nukidora { .. } => false,
        })
    }

    fn triplet_contains(self, target: usize) -> bool {
        self.triplet_kan_any(|tile| tile.kind() == target)
    }

    fn numeric_suit_count(self) -> u8 {
        let mut mask = 0u8;
        for meld in self.melds {
            let tile = match meld {
                Meld::Chi { tiles, .. } => tiles[0],
                Meld::Pon { tile, .. }
                | Meld::Daiminkan { tile, .. }
                | Meld::Kakan { tile, .. }
                | Meld::Ankan { tile } => *tile,
                Meld::Nukidora { .. } => continue,
            };
            mask |= match tile.suit() {
                Suit::Man => 1,
                Suit::Pin => 2,
                Suit::Sou => 4,
                Suit::Honor => 0,
            };
        }
        mask.count_ones() as u8
    }
}

fn structural_blocked(id: YakuStarId, melds: MeldFacts<'_>, ctx: &YakuStarContext) -> bool {
    let fixed_count = melds.fixed_count();
    if fixed_count > 4 {
        return true;
    }
    match id {
        YakuStarId::Riichi
        | YakuStarId::DoubleRiichi
        | YakuStarId::Ippatsu
        | YakuStarId::MenzenTsumo => !ctx.menzen,
        YakuStarId::Pinfu => !ctx.menzen || melds.has_fixed_triplet_or_kan(),
        YakuStarId::Iipeikou => !ctx.menzen || fixed_count.saturating_sub(melds.ankans()) > 2,
        YakuStarId::Ryanpeikou => !ctx.menzen || melds.ankans() > 0 || fixed_count > 0,
        YakuStarId::Chiitoitsu | YakuStarId::Kokushi | YakuStarId::Chuuren => {
            fixed_count > 0 || !ctx.menzen
        }
        YakuStarId::Tanyao => melds.fixed_tiles_any(Tile::is_yaochuu),
        YakuStarId::Honitsu => melds.numeric_suit_count() > 1,
        YakuStarId::Chinitsu => {
            melds.fixed_tiles_any(Tile::is_honor) || melds.numeric_suit_count() > 1
        }
        YakuStarId::Chanta => {
            melds.chi_any(|start| !sequence_has_terminal(start))
                || melds.triplet_kan_any(|tile| !tile.is_yaochuu())
        }
        YakuStarId::Junchan => {
            melds.chi_any(|start| !sequence_has_terminal(start))
                || melds.triplet_kan_any(|tile| !tile.is_terminal())
        }
        YakuStarId::Honroutou => {
            melds.has_chi() || melds.fixed_tiles_any(|tile| !tile.is_yaochuu())
        }
        YakuStarId::Chinroutou => {
            melds.has_chi() || melds.fixed_tiles_any(|tile| !tile.is_terminal())
        }
        YakuStarId::Tsuuiisou => melds.fixed_tiles_any(|tile| !tile.is_honor()),
        YakuStarId::Ryuuiisou => melds.fixed_tiles_any(|tile| !is_green(tile)),
        YakuStarId::Toitoi => melds.has_chi(),
        YakuStarId::Suuankou => !ctx.menzen || melds.has_chi() || melds.has_open_triplet_or_kan(),
        YakuStarId::Sanankou => melds.ankans() + 4usize.saturating_sub(fixed_count) < 3,
        YakuStarId::Sankantsu => kan_yaku_blocked(melds, ctx, 3),
        YakuStarId::Suukantsu => kan_yaku_blocked(melds, ctx, 4),
        YakuStarId::SanshokuDoujun => sequence_pattern_blocked(melds),
        YakuStarId::Ittsuu => ittsuu_blocked(melds),
        YakuStarId::SanshokuDoukou => sanshoku_doukou_blocked(melds),
        YakuStarId::YakuhaiRound => yakuhai_triplet_blocked(melds, ctx.bakaze.kind()),
        YakuStarId::YakuhaiSeat => yakuhai_triplet_blocked(melds, ctx.jikaze.kind()),
        YakuStarId::Haku => yakuhai_triplet_blocked(melds, 31),
        YakuStarId::Hatsu => yakuhai_triplet_blocked(melds, 32),
        YakuStarId::Chun => yakuhai_triplet_blocked(melds, 33),
        YakuStarId::Daisangen => required_triplet_group_blocked(melds, &[31, 32, 33], 3),
        YakuStarId::Shousangen => required_triplet_group_blocked(melds, &[31, 32, 33], 2),
        YakuStarId::Daisuushi => required_triplet_group_blocked(melds, &[27, 28, 29, 30], 4),
        YakuStarId::Shousuushi => required_triplet_group_blocked(melds, &[27, 28, 29, 30], 3),
        YakuStarId::Tenhou => !ctx.tenhou,
        YakuStarId::Chiihou => !ctx.chiihou,
        _ => false,
    }
}

fn kan_yaku_blocked(melds: MeldFacts<'_>, ctx: &YakuStarContext, target: usize) -> bool {
    let own_kans = melds.own_kans();
    if own_kans >= target {
        return false;
    }
    if usize::from(ctx.total_declared_kans.max(own_kans as u8).min(4)) >= 4 {
        return true;
    }
    let max_kans = own_kans + melds.pons() + 4usize.saturating_sub(melds.fixed_count());
    max_kans < target && !(ctx.has_legal_kan_candidate && max_kans + 1 >= target)
}

fn yakuhai_triplet_blocked(melds: MeldFacts<'_>, target: usize) -> bool {
    !melds.triplet_contains(target) && melds.fixed_count() >= 4
}

fn required_triplet_group_blocked(
    melds: MeldFacts<'_>,
    targets: &[usize],
    required: usize,
) -> bool {
    let fixed = targets
        .iter()
        .filter(|&&target| melds.triplet_contains(target))
        .count();
    required.saturating_sub(fixed) > 4usize.saturating_sub(melds.fixed_count())
}

fn sequence_pattern_blocked(melds: MeldFacts<'_>) -> bool {
    for rank in 0..=6 {
        let have = (0..3)
            .filter(|&suit| {
                let start = suit * 9 + rank;
                melds.chi_any(|chi| chi == start)
            })
            .count();
        if 3usize.saturating_sub(have) <= 4usize.saturating_sub(melds.fixed_count()) {
            return false;
        }
    }
    true
}

fn ittsuu_blocked(melds: MeldFacts<'_>) -> bool {
    for suit in 0..3 {
        let base = suit * 9;
        let have = [base, base + 3, base + 6]
            .iter()
            .filter(|&&start| melds.chi_any(|chi| chi == start))
            .count();
        if 3usize.saturating_sub(have) <= 4usize.saturating_sub(melds.fixed_count()) {
            return false;
        }
    }
    true
}

fn sanshoku_doukou_blocked(melds: MeldFacts<'_>) -> bool {
    for rank in 0..9 {
        let have = (0..3)
            .filter(|&suit| melds.triplet_contains(suit * 9 + rank))
            .count();
        if 3usize.saturating_sub(have) <= 4usize.saturating_sub(melds.fixed_count()) {
            return false;
        }
    }
    true
}

fn sequence_has_terminal(start: usize) -> bool {
    start < 27 && matches!(start % 9, 0 | 6)
}

fn is_green(tile: Tile) -> bool {
    matches!(tile.kind(), 19 | 20 | 21 | 23 | 25 | 32)
}
