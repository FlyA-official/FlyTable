//! Mapping between Tenhou yaku IDs and FlyTable yaku, used by `HoraBody.normal` /
//! `yakuman` in the match log.
//!
//! The match log builder and replay validation share this single table so they
//! cannot drift apart.
//!
//! # Wind parameters
//!
//! Tenhou splits yakuhai winds into eight IDs (seat wind E/S/W/N = 10..13, round
//! wind E/S/W/N = 14..17), while [`Yaku::Yakuhai`] only records seat or round wind.
//! Yaku to ID therefore needs `jikaze` / `bakaze` from the caller; ID to yaku loses
//! the specific wind.
//!
//! # IDs not in the table
//!
//! - `36` nagashi mangan: recorded as `Ryukyoku.kind = NagashiMangan`, not a yaku.
//! - `52` / `53` / `54` dora / ura dora / red dora: go into `dora_han` / `ura_han` /
//!   `aka_han` of `HoraBody`, never `normal` (`validate` rejects it).
//! - 3-player North yakuhai ([`YakuhaiKind::Pei`]): no ID in the 4-player table;
//!   North dora goes into `nuki_han`.

use flytable_core::tile::Tile;
use flytable_core::yaku::{Yaku, YakuhaiKind, YakumanKind};

/// Wind offset in the yaku table (E=0 S=1 W=2 N=3).
fn wind_offset(tile: Tile) -> Option<u8> {
    match tile.to_string().as_str() {
        "E" => Some(0),
        "S" => Some(1),
        "W" => Some(2),
        "N" => Some(3),
        _ => None,
    }
}

/// FlyTable yaku to Tenhou yaku ID.
///
/// `jikaze` / `bakaze` are only used for yakuhai winds (see the module docs).
///
/// Returns `None` when there is no ID, currently only for 3-player North yakuhai
/// (`Yakuhai(Pei)`), which goes into `nuki_han`.
#[must_use]
pub fn tenhou_yaku_id(yaku: &Yaku, jikaze: Tile, bakaze: Tile) -> Option<u8> {
    Some(match yaku {
        Yaku::MenzenTsumo => 0,
        Yaku::Riichi => 1,
        Yaku::Ippatsu => 2,
        Yaku::Chankan => 3,
        Yaku::Rinshan => 4,
        Yaku::Haitei => 5,
        Yaku::Houtei => 6,
        Yaku::Pinfu => 7,
        Yaku::Tanyao => 8,
        Yaku::Iipeikou => 9,
        Yaku::Yakuhai(YakuhaiKind::Jikaze) => 10 + wind_offset(jikaze)?,
        Yaku::Yakuhai(YakuhaiKind::Bakaze) => 14 + wind_offset(bakaze)?,
        Yaku::Yakuhai(YakuhaiKind::Haku) => 18,
        Yaku::Yakuhai(YakuhaiKind::Hatsu) => 19,
        Yaku::Yakuhai(YakuhaiKind::Chun) => 20,
        Yaku::Yakuhai(YakuhaiKind::Pei) => return None,
        Yaku::DoubleRiichi => 21,
        Yaku::Chiitoitsu => 22,
        Yaku::Chanta => 23,
        Yaku::Ittsuu => 24,
        Yaku::Sanshoku => 25,
        Yaku::SanshokuDoukou => 26,
        Yaku::Sankantsu => 27,
        Yaku::Toitoi => 28,
        Yaku::Sanankou => 29,
        Yaku::Shousangen => 30,
        Yaku::Honroutou => 31,
        Yaku::Ryanpeikou => 32,
        Yaku::Junchan => 33,
        Yaku::Honitsu => 34,
        Yaku::Chinitsu => 35,
        Yaku::Yakuman(k) => match k {
            YakumanKind::Tenhou => 37,
            YakumanKind::Chiihou => 38,
            YakumanKind::Daisangen => 39,
            YakumanKind::Suuankou => 40,
            YakumanKind::SuuankouTanki => 41,
            YakumanKind::Tsuuiisou => 42,
            YakumanKind::Ryuuiisou => 43,
            YakumanKind::Chinroutou => 44,
            YakumanKind::ChuurenPoutou => 45,
            YakumanKind::JunseiChuuren => 46,
            YakumanKind::KokushiMusou => 47,
            YakumanKind::KokushiMusouJuusanmen => 48,
            YakumanKind::Daisuushi => 49,
            YakumanKind::Shousuushi => 50,
            YakumanKind::Suukantsu => 51,
        },
    })
}

/// Tenhou yaku ID to FlyTable yaku.
///
/// Wind yakuhai collapse to `Yakuhai(Jikaze)` / `Yakuhai(Bakaze)`; comparisons only
/// need seat versus round wind.
#[must_use]
pub fn tenhou_yaku_name(id: u8) -> Option<&'static str> {
    Some(match id {
        0 => "MenzenTsumo",
        1 => "Riichi",
        2 => "Ippatsu",
        3 => "Chankan",
        4 => "Rinshan",
        5 => "Haitei",
        6 => "Houtei",
        7 => "Pinfu",
        8 => "Tanyao",
        9 => "Iipeikou",
        10..=13 => "Yakuhai(Jikaze)",
        14..=17 => "Yakuhai(Bakaze)",
        18 => "Yakuhai(Haku)",
        19 => "Yakuhai(Hatsu)",
        20 => "Yakuhai(Chun)",
        21 => "DoubleRiichi",
        22 => "Chiitoitsu",
        23 => "Chanta",
        24 => "Ittsuu",
        25 => "Sanshoku",
        26 => "SanshokuDoukou",
        27 => "Sankantsu",
        28 => "Toitoi",
        29 => "Sanankou",
        30 => "Shousangen",
        31 => "Honroutou",
        32 => "Ryanpeikou",
        33 => "Junchan",
        34 => "Honitsu",
        35 => "Chinitsu",
        37 => "Yakuman(Tenhou)",
        38 => "Yakuman(Chiihou)",
        39 => "Yakuman(Daisangen)",
        40 => "Yakuman(Suuankou)",
        41 => "Yakuman(SuuankouTanki)",
        42 => "Yakuman(Tsuuiisou)",
        43 => "Yakuman(Ryuuiisou)",
        44 => "Yakuman(Chinroutou)",
        45 => "Yakuman(ChuurenPoutou)",
        46 => "Yakuman(JunseiChuuren)",
        47 => "Yakuman(KokushiMusou)",
        48 => "Yakuman(KokushiMusouJuusanmen)",
        49 => "Yakuman(Daisuushi)",
        50 => "Yakuman(Shousuushi)",
        51 => "Yakuman(Suukantsu)",
        _ => return None,
    })
}

/// Whether the ID is a dora component (52 dora / 53 ura / 54 red). These never go into `normal`.
#[must_use]
pub const fn is_dora_id(id: u8) -> bool {
    matches!(id, 52..=54)
}

/// FlyTable yaku to a name in the same scheme as `tenhou_yaku_name`, for comparison.
#[must_use]
pub fn yaku_name(yaku: &Yaku) -> String {
    match yaku {
        Yaku::Yakuhai(k) => format!(
            "Yakuhai({})",
            match k {
                YakuhaiKind::Haku => "Haku",
                YakuhaiKind::Hatsu => "Hatsu",
                YakuhaiKind::Chun => "Chun",
                YakuhaiKind::Bakaze => "Bakaze",
                YakuhaiKind::Jikaze => "Jikaze",
                YakuhaiKind::Pei => "Pei",
            }
        ),
        Yaku::Yakuman(k) => format!("Yakuman({k:?})"),
        other => format!("{other:?}"),
    }
}
