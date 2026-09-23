//! Event enum of `flytable-matchlog-v1`.
//!
//! One event type serves both 3P and 4P: every per-seat array is a `Vec`, with the
//! global invariant that its length equals `start_match.seats`. Fixed-width 3P/4P
//! types would split every event in two.

use flytable_core::rules::RiichiRuleProfile;
use flytable_core::tile::Tile;
use serde::{Deserialize, Serialize};

use super::types::{MatchSettlementProfile, MatchlogMeld, RuleEra, TileIdentity, TileRef};

/// Wall source of a draw. The three paths have different geometry: live wall
/// cursor, dead wall end, and live wall tail (3-player nukidora replacement).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrawSource {
    /// Regular draw from the live wall cursor.
    Live,
    /// Replacement draw after a kan, from the dead wall end (`Wall::draw_rinshan`).
    Rinshan,
    /// 3-player nukidora replacement from the live wall tail (`Wall::draw_supplement`).
    /// Does not use rinshan capacity or reveal dora.
    Supplement,
}

/// What created a dora indicator (not what released it).
///
/// Creation and release are different: nukidora never creates an indicator, so a
/// nukidora after consecutive kans means "created by the added kan, released by the
/// nukidora". Release timing follows from the rules and is not stored. The initial
/// indicator comes with `StartKyoku` and has no event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DoraCreatedBy {
    Ankan,
    Daiminkan,
    Kakan,
}

/// Kind of robbing window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RobberyKind {
    /// Robbing an added kan (always allowed).
    Kakan,
    /// Kokushi robbing a closed kan (`allows_kokushi_ankan_ron`; off on Tenhou, on on Mahjong Soul).
    Ankan,
    /// Robbing a nukidora (3-player, `allows_nukidora_ron`).
    Kita,
}

/// Opening and closing of a robbing window.
///
/// `Open` / `Close` are emitted even when nobody can rob, so every implementation
/// produces the same stream. A successful rob emits only `Open`: the window ends in
/// a win, not a normal close.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowEdge {
    Open,
    Close,
}

/// Kind of draw. Required, since `Event3p/4p::Ryukyoku` only carries deltas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RyukyokuKind {
    /// Exhaustive draw (wall exhausted).
    Exhaustive,
    /// Kyuushu kyuuhai (the only draw a player declares).
    KyuushuKyuuhai,
    /// Four kans.
    Suukaikan,
    /// Four riichi (4P only).
    SuuchaRiichi,
    /// Four identical wind discards (4P only).
    SuufonRenda,
    /// Triple ron (4P only, with `ron_resolution = TripleRonAbortive`).
    SanchaHora,
    /// Nagashi mangan.
    NagashiMangan,
}

impl RyukyokuKind {
    /// Whether this draw kind can occur with the given seat count.
    #[must_use]
    pub const fn possible_with(self, seats: usize) -> bool {
        match self {
            Self::SuuchaRiichi | Self::SuufonRenda | Self::SanchaHora => seats == 4,
            _ => true,
        }
    }
}

/// Scoring limit tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Limit {
    None,
    Mangan,
    Haneman,
    Baiman,
    Sanbaiman,
    /// N-times yakuman.
    Yakuman(u8),
}

/// Explicit platform override of an engine-derived value.
///
/// Some platform behavior (a disconnect combined with nagashi mangan, for example)
/// makes the recorded log contradict the engine, and the log is the truth. Without
/// this marker "contradicts the engine means invalid" and "platform behavior is not
/// a mismatch" would conflict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlatformOverride {
    /// Index of the overridden event in the stream.
    pub target_seq: u64,
    /// Dotted path of the overridden field (flat, not nested).
    pub field: String,
    /// Recorded value, authoritative for replay. The overridden event stores this value.
    pub recorded: serde_json::Value,
    /// Engine-derived value, kept as evidence only.
    pub engine: serde_json::Value,
    /// Stable reason code. Must be in the versioned allowlist, otherwise the stream counts as a rule mismatch.
    pub reason_code: String,
    /// Corpus sample ID.
    pub sample_id: String,
}

/// Win settlement.
///
/// `concealed`, `melds` and `machi` are disjoint and together form the whole hand.
/// `concealed` holds only the closed hand, without melds or the winning tile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HoraBody {
    pub winner: u8,
    /// `from == winner` means tsumo.
    pub from: u8,
    /// Winning tile, kept separate from `concealed`.
    pub machi: TileRef,
    pub concealed: Vec<TileRef>,
    pub melds: Vec<MatchlogMeld>,
    /// `(yaku id 0..=54, han)`. Mutually exclusive with `yakuman`.
    ///
    /// Counted yakuman go here together with `limit`: they are regular yaku adding up to
    /// 13 han and must not be moved into `yakuman`.
    pub normal: Vec<(u8, u8)>,
    /// `(yaku id, multiplier)`. When non-empty, `normal` must be empty.
    pub yakuman: Vec<(u8, u8)>,
    pub fu: u8,
    pub limit: Limit,
    pub dora_han: u8,
    pub ura_han: u8,
    pub aka_han: u8,
    pub nuki_han: u8,
    /// Non-empty only for riichi wins.
    pub ura_markers: Vec<TileRef>,
    /// Liable seat (daisangen / daisuushi).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pao: Option<u8>,
    /// Excluding honba and riichi sticks. Zero-sum.
    pub base_deltas: Vec<i32>,
    /// Honba part. Zero-sum.
    pub honba_deltas: Vec<i32>,
    /// Riichi stick part. Not zero-sum: sticks go from the table pool to the winner, so
    /// the sum equals the collected amount (sticks in the pool before settlement x 1000,
    /// including sticks accepted this hand).
    pub kyotaku_deltas: Vec<i32>,
}

/// Draw settlement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RyukyokuBody {
    pub kind: RyukyokuKind,
    pub tenpai_mask: Vec<bool>,
    /// Closed hands revealed by tenpai players; `None` if not revealed. Melds are not
    /// repeated here; they are already in the call events.
    pub concealed: Vec<Option<Vec<TileRef>>>,
    /// Noten penalty or nagashi mangan payment. Zero-sum.
    pub base_deltas: Vec<i32>,
    /// Zero-sum.
    pub honba_deltas: Vec<i32>,
    /// Riichi sticks carried to the next hand (not distributed).
    pub kyotaku_carry: u8,
}

/// Match log event. `tag = "type"` with `deny_unknown_fields`: core fields have a
/// single meaning; anything else goes into `ext.<namespace>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum MatchlogEvent {
    // Header (L0)
    StartMatch {
        schema: String,
        seats: u8,
        /// Inline rule profile snapshot, authoritative for replay.
        rule_profile: RiichiRuleProfile,
        /// Labels the source only; does not drive replay.
        rule_era: RuleEra,
        /// Snapshot fingerprint, used for coverage checks and era consistency.
        profile_fingerprint: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        settlement_profile: Option<MatchSettlementProfile>,
        tile_identity: TileIdentity,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        names: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seed: Option<(u64, u64)>,
    },

    // Facts (L0)
    StartKyoku {
        bakaze: Tile,
        /// Hand number, 1-based.
        kyoku: u8,
        honba: u8,
        kyotaku: u8,
        oya: u8,
        scores: Vec<i32>,
        /// 13 starting tiles per seat.
        haipai: Vec<Vec<TileRef>>,
        /// The initial indicator is only carried here; there is no separate `Dora` event.
        dora_marker: TileRef,
    },
    Tsumo {
        actor: u8,
        pai: TileRef,
        source: DrawSource,
    },
    /// The dealer's 14th starting tile (Mahjong Soul).
    DealerOpening {
        actor: u8,
        pai: TileRef,
    },
    Dahai {
        actor: u8,
        pai: TileRef,
        tsumogiri: bool,
        /// Whether this discard declares riichi (acceptance is `ReachAccepted`).
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        riichi_declare: bool,
    },
    DealerOpeningDahai {
        actor: u8,
        pai: TileRef,
    },
    Call {
        actor: u8,
        meld: MatchlogMeld,
    },
    /// Kyuushu kyuuhai declaration.
    KyuushuDeclare {
        actor: u8,
    },

    // Platform (typed core event rather than `ext`, since it affects L2 rulings)
    PlatformDisconnect {
        seat: u8,
    },
    PlatformReconnect {
        seat: u8,
    },

    // Rulings (L1)
    /// Riichi accepted (the declaration tile passed the ron window). Always before any `Ryukyoku`.
    ReachAccepted {
        actor: u8,
    },
    /// Dora reveal. The face is an L0 fact; the position is an L1 ruling.
    Dora {
        marker: TileRef,
        created_by: DoraCreatedBy,
    },
    RobberyWindow {
        kind: RobberyKind,
        edge: WindowEdge,
    },

    // Settlement (L2)
    Hora(Box<HoraBody>),
    Ryukyoku(Box<RyukyokuBody>),
    /// Transition to the next hand.
    Advance {
        next_kyoku: u8,
        next_bakaze: Tile,
        honba: u8,
        kyotaku: u8,
        renchan: bool,
        ended: bool,
    },
    MatchEnd {
        final_scores: Vec<i32>,
        /// Rank points. `None` while undetermined.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rank_points: Option<Vec<f64>>,
    },
}
