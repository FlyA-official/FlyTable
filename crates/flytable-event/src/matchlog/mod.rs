//! `flytable-matchlog-v1`: archival match log format.
//!
//! Relation to existing types:
//! * [`crate::Event3p`] / [`crate::Event4p`] (`PROTOCOL_VERSION = 2`) are the engine's
//!   internal events. The match log uses the same vocabulary; conversion is a
//!   mechanical projection.
//! * `CanonicalEvent` / `MahjongEventsEnvelope` in `flya-mahjong-events` are the live
//!   per-seat wire format. They stay separate and share no serde types, since a
//!   strict archive and a flattened wire format have conflicting needs.
//! * The match log covers what neither does: storing and exchanging complete games.
//!
//! Layers:
//! ```text
//! L0 facts        player actions, deal/draw payloads, wall reveal payloads, platform events
//! L1 rulings      timings derived by the engine: riichi accepted, dora reveal, robbing windows, draws
//! L2 settlement   per-yaku breakdown, fu, pao, split deltas, tenpai reveal, hand transitions, game end
//! L3 windows      decision windows: seat, phase, event anchor, legal actions, actual choice
//! ```
//! Wall reveal payloads are L0: indicator faces come from fixed dead-wall slots and
//! never pass through a draw event, so the engine only consumes them. The face is
//! an L0 fact; its timing is an L1 ruling.

pub mod event;
pub mod types;
pub mod validate;
pub mod window;
pub mod yaku_id;

pub use event::{
    DoraCreatedBy, DrawSource, HoraBody, Limit, MatchlogEvent, PlatformOverride, RobberyKind,
    RyukyokuBody, RyukyokuKind, WindowEdge,
};
pub use types::{
    MATCHLOG_SCHEMA, MatchSettlementProfile, MatchlogMeld, MeldKind, RuleEra, TieBreak,
    TileIdentity, TileRef, profile_fingerprint,
};
pub use validate::{Violation, validate};

pub use window::{DecisionWindow, WindowPhase};
pub use yaku_id::{is_dora_id, tenhou_yaku_id, tenhou_yaku_name, yaku_name};
