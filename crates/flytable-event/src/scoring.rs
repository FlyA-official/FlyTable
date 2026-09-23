//! Win settlement details carried by `Hora` events (shared by 4-player and 3-player).
//! Follows mjai conventions and is part of the versioned protocol.

use serde::{Deserialize, Serialize};

/// Han breakdown: regular (han and fu) or yakuman (multiplier).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HoraAgari {
    Normal { fu: u8, han: u8 },
    Yakuman { count: u8 },
}

/// Points for each payment case (ron / non-dealer tsumo / dealer tsumo).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HoraPoint {
    pub ron: i32,
    pub tsumo_ko: i32,
    pub tsumo_oya: i32,
}

/// Flags for yaku determined by game state rather than hand shape.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HoraYakuFlags {
    pub riichi: bool,
    pub double_riichi: bool,
    pub ippatsu: bool,
    pub menzen_tsumo: bool,
    pub haitei: bool,
    pub houtei: bool,
    pub rinshan: bool,
    pub chankan: bool,
    pub tenhou: bool,
    pub chiihou: bool,
}

/// Full settlement details of a win.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HoraScoring {
    pub agari: HoraAgari,
    pub point: HoraPoint,
    /// Han from yaku, excluding dora.
    pub additional_hans: u8,
    pub dora_han: u8,
    pub red_dora_han: u8,
    pub ura_dora_han: u8,
    /// Nukidora han (3-player only; always 0 in 4-player).
    pub nuki_dora_han: u8,
    pub yaku_flags: HoraYakuFlags,
}
