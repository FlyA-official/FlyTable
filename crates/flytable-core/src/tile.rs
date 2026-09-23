//! Tile representation.
//!
//! `Tile` covers the full tile space: 34 base kinds plus 3 red fives. Which tiles
//! exist in a given variant (3-player mahjong only has 1m and 9m in manzu) is a
//! property of the wall and rule variant, not of this type.
//!
//! IDs follow the mjai convention so events round-trip losslessly:
//! ```text
//!  0..=8   1m..9m       9..=17  1p..9p      18..=26 1s..9s
//! 27 E  28 S  29 W  30 N      31 P(haku) 32 F(hatsu) 33 C(chun)
//! 34 5mr   35 5pr   36 5sr    (red fives)
//! ```

use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Suit {
    Man,
    Pin,
    Sou,
    Honor,
}

/// A single tile, red fives distinguished. Internally an ID in `0..=36`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tile(u8);

pub const NUM_KINDS: usize = 34;
/// Number of tile IDs including red fives (excluding the unknown tile).
pub const NUM_IDS: usize = 37;
/// ID of the unknown tile, used for hidden tiles in seat views.
pub const UNKNOWN_ID: u8 = 37;

const STRINGS: [&str; NUM_IDS + 1] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", // 0..=8
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", // 9..=17
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", // 18..=26
    "E", "S", "W", "N", "P", "F", "C", // 27..=33
    "5mr", "5pr", "5sr", // 34..=36
    "?",   // 37 unknown
];

const AKA_5M: u8 = 34;
const AKA_5P: u8 = 35;
const AKA_5S: u8 = 36;

#[derive(Debug, thiserror::Error)]
pub enum InvalidTile {
    #[error("invalid tile id {0}")]
    Id(usize),
    #[error("invalid tile name {0:?}")]
    Name(String),
}

impl Tile {
    pub const fn unknown() -> Self {
        Self(UNKNOWN_ID)
    }

    pub const fn is_unknown(self) -> bool {
        self.0 == UNKNOWN_ID
    }

    /// Builds a tile from an ID in `0..=37` (37 = unknown).
    pub const fn from_id(id: u8) -> Result<Self, InvalidTile> {
        if (id as usize) <= NUM_IDS {
            Ok(Self(id))
        } else {
            Err(InvalidTile::Id(id as usize))
        }
    }

    /// # Safety
    /// The caller must ensure `id <= NUM_IDS`.
    pub const unsafe fn from_id_unchecked(id: u8) -> Self {
        Self(id)
    }

    /// Builds a number tile from a suit and a 1-based rank. Use [`Tile::honor`] for honors.
    pub fn number(suit: Suit, n: u8) -> Result<Self, InvalidTile> {
        match suit {
            Suit::Man if (1..=9).contains(&n) => Ok(Self(n - 1)),
            Suit::Pin if (1..=9).contains(&n) => Ok(Self(9 + n - 1)),
            Suit::Sou if (1..=9).contains(&n) => Ok(Self(18 + n - 1)),
            _ => Err(InvalidTile::Id((suit as usize) * 100 + n as usize)),
        }
    }

    /// Builds an honor tile from `0..=6` (E S W N P F C).
    pub fn honor(idx: u8) -> Result<Self, InvalidTile> {
        if idx < 7 {
            Ok(Self(27 + idx))
        } else {
            Err(InvalidTile::Id(27 + idx as usize))
        }
    }

    /// Raw ID, including red-five IDs `34..=36`.
    pub const fn id(self) -> u8 {
        self.0
    }

    /// Folds a red five back to the plain five. Rule computations work in the 34-kind space.
    pub const fn deaka(self) -> Self {
        match self.0 {
            AKA_5M => Self(4),
            AKA_5P => Self(13),
            AKA_5S => Self(22),
            _ => self,
        }
    }

    /// Turns a plain five into its red five; other tiles are returned unchanged.
    pub const fn akaize(self) -> Self {
        match self.0 {
            4 => Self(AKA_5M),
            13 => Self(AKA_5P),
            22 => Self(AKA_5S),
            _ => self,
        }
    }

    pub const fn is_aka(self) -> bool {
        matches!(self.0, AKA_5M | AKA_5P | AKA_5S)
    }

    /// Base kind index `0..=33` after folding red fives.
    pub const fn kind(self) -> usize {
        self.deaka().0 as usize
    }

    pub const fn suit(self) -> Suit {
        match self.deaka().0 {
            0..=8 => Suit::Man,
            9..=17 => Suit::Pin,
            18..=26 => Suit::Sou,
            _ => Suit::Honor,
        }
    }

    /// Rank `1..=9` for number tiles, `None` for honors.
    pub const fn rank(self) -> Option<u8> {
        let k = self.deaka().0;
        match k {
            0..=8 => Some(k + 1),
            9..=17 => Some(k - 9 + 1),
            18..=26 => Some(k - 18 + 1),
            _ => None,
        }
    }

    /// Honor index `0..=6` (E S W N P F C), `None` for number tiles.
    pub const fn honor_idx(self) -> Option<u8> {
        let k = self.deaka().0;
        if k >= 27 && k <= 33 {
            Some(k - 27)
        } else {
            None
        }
    }

    pub const fn is_honor(self) -> bool {
        matches!(self.deaka().0, 27..=33)
    }

    /// 3-player dora from indicator: 1m and 9m point to each other, everything else as in 4-player.
    pub const fn dora_from_indicator_3p(self) -> Self {
        let k = self.deaka().0;
        match k {
            0 => Self(8), // 1m → 9m
            8 => Self(0), // 9m → 1m
            _ => self.dora_from_indicator(),
        }
    }

    /// Dora from indicator: numbers wrap 9 -> 1 within a suit, winds wrap N -> E, dragons wrap C -> P.
    pub const fn dora_from_indicator(self) -> Self {
        let k = self.deaka().0;
        match k {
            0..=8 => Self(if k == 8 { 0 } else { k + 1 }),
            9..=17 => Self(if k == 17 { 9 } else { k + 1 }),
            18..=26 => Self(if k == 26 { 18 } else { k + 1 }),
            27..=30 => Self(if k == 30 { 27 } else { k + 1 }),
            31..=33 => Self(if k == 33 { 31 } else { k + 1 }),
            _ => self,
        }
    }

    pub const fn is_dragon(self) -> bool {
        matches!(self.deaka().0, 31..=33)
    }

    pub const fn is_wind(self) -> bool {
        matches!(self.deaka().0, 27..=30)
    }

    /// Terminal (number tile 1 or 9).
    pub const fn is_terminal(self) -> bool {
        matches!(self.deaka().0, 0 | 8 | 9 | 17 | 18 | 26)
    }

    /// Terminal or honor.
    pub const fn is_yaochuu(self) -> bool {
        self.is_terminal() || self.is_honor()
    }

    /// Next tile in the same suit. Does not wrap past 9; honors return `None`.
    pub const fn succ_in_suit(self) -> Option<Self> {
        let k = self.deaka().0;
        match k {
            0..=7 | 9..=16 | 18..=25 => Some(Self(k + 1)),
            _ => None,
        }
    }
}

impl Default for Tile {
    /// Hidden tiles in a view are unknown by default.
    fn default() -> Self {
        Self::unknown()
    }
}

impl fmt::Debug for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(STRINGS[self.0 as usize])
    }
}

impl FromStr for Tile {
    type Err = InvalidTile;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        STRINGS
            .iter()
            .position(|&t| t == s)
            .map(|i| Self(i as u8))
            .ok_or_else(|| InvalidTile::Name(s.to_owned()))
    }
}

impl serde::Serialize for Tile {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(STRINGS[self.0 as usize])
    }
}

impl<'de> serde::Deserialize<'de> for Tile {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = <String as serde::Deserialize>::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}
