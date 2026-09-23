//! Basic types for `flytable-matchlog-v1`: physical tile references, melds,
//! settlement profiles, fingerprints.

use flytable_core::rules::RiichiRuleProfile;
use flytable_core::tile::Tile;
use serde::{Deserialize, Serialize};

/// Match log schema identifier. Breaking changes bump the major version.
pub const MATCHLOG_SCHEMA: &str = "flytable-matchlog-v1";

/// Tile reference with an optional physical identity.
///
/// The engine works at the kind level (`Tile`, `0..=37`); rules never distinguish
/// the four copies of a kind, and red fives are already modelled in `Tile`. The
/// physical identity (`0..=135`) is only used by the match log for wall
/// reconstruction and copy-level validation. The engine core is unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileRef {
    pub tile: Tile,
    /// 136-tile ID (`kind * 4 + copy`; red fives are 16 / 52 / 88). Scoped to a single hand.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical_id: Option<u8>,
}

impl TileRef {
    #[must_use]
    pub const fn new(tile: Tile, physical_id: Option<u8>) -> Self {
        Self { tile, physical_id }
    }

    /// Reference without a physical identity (engine-generated logs, mjai sources).
    #[must_use]
    pub const fn opaque(tile: Tile) -> Self {
        Self {
            tile,
            physical_id: None,
        }
    }

    /// Whether the physical ID agrees with the tile kind and red flag.
    #[must_use]
    pub fn is_consistent(self) -> bool {
        let Some(id) = self.physical_id else {
            return true;
        };
        if id >= 136 {
            return false;
        }
        // `id / 4` is the kind; 16 / 52 / 88 are the red 5m / 5p / 5s.
        if usize::from(id / 4) != self.tile.kind() {
            return false;
        }
        matches!(id, 16 | 52 | 88) == self.tile.is_aka()
    }
}

/// How complete the physical identities in a match log are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TileIdentity {
    /// Every `physical_id` is `None`. Sporadic values are not allowed.
    #[default]
    None,
    /// Mixed. Given IDs must be consistent, but the ledger need not close.
    Partial,
    /// Every `TileRef` that appears has an ID, and the ledger closes over the observed set.
    ///
    /// Not all 136 / 108 tiles have to appear: unrevealed dead-wall tiles and the
    /// undrawn live wall never show up in events.
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeldKind {
    Chi,
    Pon,
    Daiminkan,
    Kakan,
    Ankan,
    /// 3-player nukidora. Does not create a dora indicator.
    Kita,
}

/// Meld type for the match log.
///
/// Separate from `flytable_core::meld::Meld`, which uses kind-level `Tile` and does
/// not keep the four physical tiles of a kan, while every tile field here must be a
/// `TileRef`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchlogMeld {
    pub kind: MeldKind,
    /// Source seat; `None` for `Ankan` and `Kita`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<u8>,
    /// The called tile. For `Kakan` the added fourth tile; `None` for `Ankan`; for `Kita` the North.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claimed: Option<TileRef>,
    /// Tiles of the meld other than `claimed`.
    ///
    /// Not necessarily from the player's own hand: the three tiles of a `Kakan` are the
    /// original pon, one of which came from another player's discard. Ownership is
    /// decided by the ledger.
    pub consumed: Vec<TileRef>,
}

impl MatchlogMeld {
    /// Per-kind shape invariants.
    ///
    /// Checks shape only (presence of `from` / `claimed`, number of `consumed`). Tile
    /// provenance needs the global ledger and is checked in `validate`.
    #[must_use]
    pub fn shape_is_valid(&self, seats: usize) -> bool {
        let n = self.consumed.len();
        match self.kind {
            MeldKind::Chi => seats == 4 && self.from.is_some() && self.claimed.is_some() && n == 2,
            MeldKind::Pon => self.from.is_some() && self.claimed.is_some() && n == 2,
            MeldKind::Daiminkan => self.from.is_some() && self.claimed.is_some() && n == 3,
            MeldKind::Kakan => self.from.is_some() && self.claimed.is_some() && n == 3,
            MeldKind::Ankan => self.from.is_none() && self.claimed.is_none() && n == 4,
            MeldKind::Kita => seats == 3 && self.from.is_none() && self.claimed.is_some() && n == 0,
        }
    }
}

/// Tie-break rule. v1 only has one; `Split` has no platform using it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TieBreak {
    /// Ties go to the player earlier in seating order from the first dealer (Tenhou 4P and 3P).
    #[default]
    SeatOrder,
}

/// Rank points, oka and tie-break.
///
/// Kept separate from `RiichiRuleProfile`: rank points vary by room on some
/// platforms, and they are a ranking policy rather than a game rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchSettlementProfile {
    /// In thousands of points. Tenhou 4P = `[20, 10, -10, -20]`, 3P = `[20, 0, -20]`.
    /// Length must equal `seats`.
    pub rank_points: Vec<i32>,
    /// Same unit. Tenhou 4P = 20, 3P = 15.
    pub oka: i32,
    pub tie_break: TieBreak,
}

impl MatchSettlementProfile {
    #[must_use]
    pub fn tenhou_4p() -> Self {
        Self {
            rank_points: vec![20, 10, -10, -20],
            oka: 20,
            tie_break: TieBreak::SeatOrder,
        }
    }

    #[must_use]
    pub fn tenhou_3p() -> Self {
        Self {
            rank_points: vec![20, 0, -20],
            oka: 15,
            tie_break: TieBreak::SeatOrder,
        }
    }

    /// Mahjong Soul 4P ranked settlement: return 25000 (equal to the start, no oka),
    /// uma `[+15, +5, -5, -15]`, ties broken by seat order.
    #[must_use]
    pub fn mahjong_soul_4p() -> Self {
        Self {
            rank_points: vec![15, 5, -5, -15],
            oka: 0,
            tie_break: TieBreak::SeatOrder,
        }
    }

    /// Mahjong Soul 3P ranked settlement: return 35000 (equal to the start, no oka), uma `[+15, 0, -15]`.
    #[must_use]
    pub fn mahjong_soul_3p() -> Self {
        Self {
            rank_points: vec![15, 0, -15],
            oka: 0,
            tie_break: TieBreak::SeatOrder,
        }
    }

    #[must_use]
    pub fn is_valid_for(&self, seats: usize) -> bool {
        self.rank_points.len() == seats
    }
}

/// Rule era. Labels the source only and does not drive replay.
///
/// Replay always uses the inline `RiichiRuleProfile` snapshot. Platform rules change
/// over time, but changing the era string alone has no effect; historical logs must
/// carry a historical profile snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RuleEra(pub String);

/// Rule profile fingerprint: a stable 16-hex-digit hash of the canonical serialization.
///
/// Used for change detection and coverage checks ("does the engine support the
/// profile this log declares"), not for security, so it uses dependency-free FNV-1a
/// 64 rather than SHA-256.
#[must_use]
pub fn profile_fingerprint(profile: &RiichiRuleProfile) -> String {
    // serde_json writes struct fields in declaration order, which is stable within a
    // build. Adding or removing fields should change the fingerprint.
    let canonical = serde_json::to_string(profile).unwrap_or_default();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in canonical.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}
