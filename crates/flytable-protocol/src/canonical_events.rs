//! Lossless canonical event stream and versioned envelope for `flya-mahjong-events-v2`
//! (with frozen v1 validation).
//!
//! `Event4p` / `Event3p` use `#[serde(tag = "type")]` without `deny_unknown_fields`, so
//! deserializing MJAI events into them silently drops the finer details of `hora`
//! (winning tile, per-yaku breakdown, final scores), `ryukyoku` (reason, tenpai,
//! hands) and `end_game` (final scores, ranks). Hence:
//! - [`CanonicalEvent`] is the lossless carrier: `type` plus every other field
//!   flattened (including unknown optional MJAI fields), so JSON round-trips field
//!   by field. Explicit adapters ([`CanonicalEvent::to_event_4p`] / [`to_event_3p`])
//!   downgrade it to `Event4p/3p` for the engine, dropping details the engine does not
//!   use.
//! - [`MahjongEventsEnvelope`] is the versioned per-seat envelope (schema major gate,
//!   rule line, visibility source, seq, canonical `legal_actions`, optional
//!   `legal_digest`). [`MahjongEventsEnvelope::validate`] enforces all invariants:
//!   rule line matches 4P/3P, seat ranges for viewer/actor/target, `from_seq` /
//!   `to_seq` versus event count, unique `action_id` equal to its index, no `chi` in
//!   3P and no `nukidora` in 4P, no `none` in fact streams, no seeds or other
//!   players' hands or draws in observed projections, and the optional digest.
//!
//! This lets `flya-decision-v1` embed and validate versioned canonical events
//! instead of accepting an arbitrary `Vec<Value>`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use flytable_core::tile::Tile;
use flytable_event::{Event3p, Event4p, HoraAgari, HoraPoint, HoraScoring, HoraYakuFlags};

use crate::wire::{CanonicalConvertError, CanonicalLegalAction, MahjongEventsSchema, RuleLineWire};
use crate::wire::{EventSource, SourceKind};
use crate::{LegalAction3p, LegalAction4p, VisibleBody3p, VisibleBody4p, project_3p, project_4p};

/// Canonical event types allowed in a 4P fact stream (`none` is a bot response, not a fact).
pub const EVENT_TYPES_4P: &[&str] = &[
    "start_game",
    "start_kyoku",
    "tsumo",
    "dealer_opening",
    "dahai",
    "dealer_opening_dahai",
    "seat_forced_autoplay",
    "seat_resumed",
    "chi",
    "pon",
    "daiminkan",
    "kakan",
    "ankan",
    "dora",
    "reach",
    "reach_accepted",
    "hora",
    "ryukyoku",
    "end_kyoku",
    "end_game",
];

/// Canonical event types allowed in a 3P fact stream (no `chi`, nukidora as `kita`, no `none`).
pub const EVENT_TYPES_3P: &[&str] = &[
    "start_game",
    "start_kyoku",
    "tsumo",
    "dealer_opening",
    "dahai",
    "dealer_opening_dahai",
    "seat_forced_autoplay",
    "seat_resumed",
    "pon",
    "daiminkan",
    "kakan",
    "ankan",
    "kita",
    "dora",
    "reach",
    "reach_accepted",
    "hora",
    "ryukyoku",
    "end_kyoku",
    "end_game",
];

/// v1 native base event fields: exactly the field names that serializing FlyTable's
/// `Event4p` / `Event3p` (including `HoraScoring`) can produce. This is the single
/// source. [`native_canonical_event_matrix`] builds every variant explicitly, so adding
/// a field fails to compile there until it is registered here.
pub const CANONICAL_BASE_EVENT_FIELDS: &[&str] = &[
    "actor",
    "bakaze",
    "consumed",
    "deltas",
    "dora_marker",
    "honba",
    "kyoku",
    "kyotaku",
    "names",
    "oya",
    "pai",
    "scores",
    "scoring",
    "seed",
    "target",
    "tehais",
    "tsumogiri",
    "ura_markers",
];

/// Whether a field name is a registered canonical event field: a base field, or an
/// MJAI dialect detail field accepted by at least one event schema (the settlement
/// and game-end details carried by `hora` / `ryukyoku` / `end_game` /
/// `reach_accepted` / `start_game`).
///
/// Detail fields are not listed separately in this file.
/// [`crate::public_event::all_registered_mjai_detail_fields`] walks the samples of
/// [`native_canonical_event_matrix`] and reads each event's public schema in
/// [`crate::public_event`], the same registry that
/// [`crate::public_event::validate_public_canonical_event`] uses. The two can
/// therefore never disagree about a field.
///
/// This is a coarse "could this be a valid field" check for diagnostics. The actual
/// security boundary for public requests is always
/// [`crate::public_event::validate_public_canonical_event`].
#[must_use]
pub fn is_canonical_event_field(field: &str) -> bool {
    CANONICAL_BASE_EVENT_FIELDS.contains(&field)
        || crate::public_event::all_registered_mjai_detail_fields().contains(field)
}

/// A maximal sample of every FlyTable fact event variant (every `Option` set, full
/// `HoraScoring`), serialized as lossless [`CanonicalEvent`]s (3P `nukidora`
/// normalized to `kita`). Used to check that the fields are a subset of
/// [`CANONICAL_BASE_EVENT_FIELDS`] and that the public parser accepts every event
/// FlyTable can produce.
///
/// Every variant is built explicitly without `..`, so adding a field to
/// `Event4p` / `Event3p` fails to compile here.
#[must_use]
pub fn native_canonical_event_matrix() -> Vec<(RuleLineWire, CanonicalEvent)> {
    let u = Tile::unknown();
    let scoring = HoraScoring {
        agari: HoraAgari::Normal { fu: 30, han: 4 },
        point: HoraPoint {
            ron: 8000,
            tsumo_ko: 2000,
            tsumo_oya: 4000,
        },
        additional_hans: 3,
        dora_han: 1,
        red_dora_han: 1,
        ura_dora_han: 0,
        nuki_dora_han: 0,
        yaku_flags: HoraYakuFlags {
            riichi: true,
            double_riichi: false,
            ippatsu: false,
            menzen_tsumo: true,
            haitei: false,
            houtei: false,
            rinshan: false,
            chankan: false,
            tenhou: false,
            chiihou: false,
        },
    };

    // Build every variant explicitly; do not use `..`.
    let events_4p = vec![
        Event4p::StartGame {
            names: ["a", "b", "c", "d"].map(String::from),
            seed: Some((1, 2)),
        },
        Event4p::StartKyoku {
            bakaze: u,
            dora_marker: u,
            kyoku: 1,
            honba: 0,
            kyotaku: 0,
            oya: 0,
            scores: [25000; 4],
            tehais: [[u; 13]; 4],
        },
        Event4p::Tsumo { actor: 0, pai: u },
        Event4p::DealerOpening { actor: 0, pai: u },
        Event4p::Dahai {
            actor: 0,
            pai: u,
            tsumogiri: false,
        },
        Event4p::DealerOpeningDahai { actor: 0, pai: u },
        Event4p::SeatForcedAutoplay { actor: 1 },
        Event4p::SeatResumed { actor: 1 },
        Event4p::Chi {
            actor: 1,
            target: 0,
            pai: u,
            consumed: [u; 2],
        },
        Event4p::Pon {
            actor: 1,
            target: 0,
            pai: u,
            consumed: [u; 2],
        },
        Event4p::Daiminkan {
            actor: 1,
            target: 0,
            pai: u,
            consumed: [u; 3],
        },
        Event4p::Kakan {
            actor: 0,
            pai: u,
            consumed: [u; 3],
        },
        Event4p::Ankan {
            actor: 0,
            consumed: [u; 4],
        },
        Event4p::Dora { dora_marker: u },
        Event4p::Reach { actor: 0 },
        Event4p::ReachAccepted { actor: 0 },
        Event4p::Hora {
            actor: 0,
            target: 2,
            deltas: Some([8000, -2000, -4000, -2000]),
            ura_markers: Some(vec![u]),
            scoring: Some(scoring),
        },
        Event4p::Ryukyoku {
            deltas: Some([3000, -1000, -1000, -1000]),
        },
        Event4p::EndKyoku,
        Event4p::EndGame,
    ];

    let events_3p = vec![
        Event3p::StartGame {
            names: ["a", "b", "c"].map(String::from),
            seed: Some((1, 2)),
        },
        Event3p::StartKyoku {
            bakaze: u,
            dora_marker: u,
            kyoku: 1,
            honba: 0,
            kyotaku: 0,
            oya: 0,
            scores: [35000; 3],
            tehais: [[u; 13]; 3],
        },
        Event3p::Tsumo { actor: 0, pai: u },
        Event3p::DealerOpening { actor: 0, pai: u },
        Event3p::Dahai {
            actor: 0,
            pai: u,
            tsumogiri: false,
        },
        Event3p::DealerOpeningDahai { actor: 0, pai: u },
        Event3p::SeatForcedAutoplay { actor: 1 },
        Event3p::SeatResumed { actor: 1 },
        Event3p::Pon {
            actor: 1,
            target: 0,
            pai: u,
            consumed: [u; 2],
        },
        Event3p::Daiminkan {
            actor: 1,
            target: 0,
            pai: u,
            consumed: [u; 3],
        },
        Event3p::Kakan {
            actor: 0,
            pai: u,
            consumed: [u; 3],
        },
        Event3p::Ankan {
            actor: 0,
            consumed: [u; 4],
        },
        Event3p::Nukidora { actor: 0, pai: u },
        Event3p::Dora { dora_marker: u },
        Event3p::Reach { actor: 0 },
        Event3p::ReachAccepted { actor: 0 },
        Event3p::Hora {
            actor: 0,
            target: 1,
            deltas: Some([4000, -2000, -2000]),
            ura_markers: Some(vec![u]),
            scoring: Some(scoring),
        },
        Event3p::Ryukyoku {
            deltas: Some([0, 1500, -1500]),
        },
        Event3p::EndKyoku,
        Event3p::EndGame,
    ];

    let mut matrix = Vec::with_capacity(events_4p.len() + events_3p.len());
    for ev in &events_4p {
        matrix.push((RuleLineWire::Riichi4p, canonical_from_native(ev)));
    }
    for ev in &events_3p {
        matrix.push((RuleLineWire::Riichi3p, canonical_from_native(ev)));
    }
    matrix
}

/// Serializes a native event as a lossless canonical event (via `from_mjai_value`,
/// 3P `nukidora` to `kita`).
pub(crate) fn canonical_from_native<E: Serialize>(ev: &E) -> CanonicalEvent {
    let value = serde_json::to_value(ev).expect("native FlyTable event serializes");
    CanonicalEvent::from_mjai_value(value).expect("native FlyTable event is canonical")
}

/// Structural reshaping of [`VisibleBody4p`](crate::VisibleBody4p) into [`Event4p`].
/// No privacy logic; the projection is done entirely in [`crate::project_4p`]. This
/// only reuses the serialization path of [`canonical_from_native`].
pub fn visible_body_to_event_4p(body: VisibleBody4p) -> Event4p {
    match body {
        VisibleBody4p::None => Event4p::None,
        VisibleBody4p::StartGame { names } => Event4p::StartGame { names, seed: None },
        VisibleBody4p::StartKyoku {
            bakaze,
            dora_marker,
            kyoku,
            honba,
            kyotaku,
            oya,
            scores,
            tehais,
        } => Event4p::StartKyoku {
            bakaze,
            dora_marker,
            kyoku,
            honba,
            kyotaku,
            oya,
            scores,
            tehais,
        },
        VisibleBody4p::Tsumo { actor, pai } => Event4p::Tsumo { actor, pai },
        VisibleBody4p::DealerOpening { actor, pai } => Event4p::DealerOpening { actor, pai },
        VisibleBody4p::Dahai {
            actor,
            pai,
            tsumogiri,
        } => Event4p::Dahai {
            actor,
            pai,
            tsumogiri,
        },
        VisibleBody4p::DealerOpeningDahai { actor, pai } => {
            Event4p::DealerOpeningDahai { actor, pai }
        }
        VisibleBody4p::SeatForcedAutoplay { actor } => Event4p::SeatForcedAutoplay { actor },
        VisibleBody4p::SeatResumed { actor } => Event4p::SeatResumed { actor },
        VisibleBody4p::Chi {
            actor,
            target,
            pai,
            consumed,
        } => Event4p::Chi {
            actor,
            target,
            pai,
            consumed,
        },
        VisibleBody4p::Pon {
            actor,
            target,
            pai,
            consumed,
        } => Event4p::Pon {
            actor,
            target,
            pai,
            consumed,
        },
        VisibleBody4p::Daiminkan {
            actor,
            target,
            pai,
            consumed,
        } => Event4p::Daiminkan {
            actor,
            target,
            pai,
            consumed,
        },
        VisibleBody4p::Kakan {
            actor,
            pai,
            consumed,
        } => Event4p::Kakan {
            actor,
            pai,
            consumed,
        },
        VisibleBody4p::Ankan { actor, consumed } => Event4p::Ankan { actor, consumed },
        VisibleBody4p::Dora { dora_marker } => Event4p::Dora { dora_marker },
        VisibleBody4p::Reach { actor } => Event4p::Reach { actor },
        VisibleBody4p::ReachAccepted { actor } => Event4p::ReachAccepted { actor },
        VisibleBody4p::Hora {
            actor,
            target,
            deltas,
            ura_markers,
            scoring,
        } => Event4p::Hora {
            actor,
            target,
            deltas,
            ura_markers,
            scoring,
        },
        VisibleBody4p::Ryukyoku { deltas } => Event4p::Ryukyoku { deltas },
        VisibleBody4p::EndKyoku => Event4p::EndKyoku,
        VisibleBody4p::EndGame => Event4p::EndGame,
    }
}

/// Structural reshaping of [`VisibleBody3p`](crate::VisibleBody3p) into [`Event3p`]
/// (see [`visible_body_to_event_4p`]).
pub fn visible_body_to_event_3p(body: VisibleBody3p) -> Event3p {
    match body {
        VisibleBody3p::None => Event3p::None,
        VisibleBody3p::StartGame { names } => Event3p::StartGame { names, seed: None },
        VisibleBody3p::StartKyoku {
            bakaze,
            dora_marker,
            kyoku,
            honba,
            kyotaku,
            oya,
            scores,
            tehais,
        } => Event3p::StartKyoku {
            bakaze,
            dora_marker,
            kyoku,
            honba,
            kyotaku,
            oya,
            scores,
            tehais,
        },
        VisibleBody3p::Tsumo { actor, pai } => Event3p::Tsumo { actor, pai },
        VisibleBody3p::DealerOpening { actor, pai } => Event3p::DealerOpening { actor, pai },
        VisibleBody3p::Dahai {
            actor,
            pai,
            tsumogiri,
        } => Event3p::Dahai {
            actor,
            pai,
            tsumogiri,
        },
        VisibleBody3p::DealerOpeningDahai { actor, pai } => {
            Event3p::DealerOpeningDahai { actor, pai }
        }
        VisibleBody3p::SeatForcedAutoplay { actor } => Event3p::SeatForcedAutoplay { actor },
        VisibleBody3p::SeatResumed { actor } => Event3p::SeatResumed { actor },
        VisibleBody3p::Pon {
            actor,
            target,
            pai,
            consumed,
        } => Event3p::Pon {
            actor,
            target,
            pai,
            consumed,
        },
        VisibleBody3p::Daiminkan {
            actor,
            target,
            pai,
            consumed,
        } => Event3p::Daiminkan {
            actor,
            target,
            pai,
            consumed,
        },
        VisibleBody3p::Kakan {
            actor,
            pai,
            consumed,
        } => Event3p::Kakan {
            actor,
            pai,
            consumed,
        },
        VisibleBody3p::Ankan { actor, consumed } => Event3p::Ankan { actor, consumed },
        VisibleBody3p::Nukidora { actor, pai } => Event3p::Nukidora { actor, pai },
        VisibleBody3p::Dora { dora_marker } => Event3p::Dora { dora_marker },
        VisibleBody3p::Reach { actor } => Event3p::Reach { actor },
        VisibleBody3p::ReachAccepted { actor } => Event3p::ReachAccepted { actor },
        VisibleBody3p::Hora {
            actor,
            target,
            deltas,
            ura_markers,
            scoring,
        } => Event3p::Hora {
            actor,
            target,
            deltas,
            ura_markers,
            scoring,
        },
        VisibleBody3p::Ryukyoku { deltas } => Event3p::Ryukyoku { deltas },
        VisibleBody3p::EndKyoku => Event3p::EndKyoku,
        VisibleBody3p::EndGame => Event3p::EndGame,
    }
}

/// Lossless canonical event: `type` plus every other field flattened.
///
/// Same semantics as `flytable-event`, but the wire form keeps unknown optional MJAI
/// fields, so MJAI JSON -> canonical -> MJAI JSON is lossless.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    /// All remaining fields (including MJAI `hora` / `ryukyoku` / `end_game` details), kept verbatim.
    #[serde(flatten)]
    pub fields: Map<String, Value>,
}

impl CanonicalEvent {
    /// Builds an event with only `type` (internal use).
    pub fn new(event_type: impl Into<String>) -> Self {
        Self {
            event_type: event_type.into(),
            fields: Map::new(),
        }
    }

    /// Full JSON object (`{"type": ..., ...fields}`).
    pub fn to_value(&self) -> Value {
        let mut obj = self.fields.clone();
        obj.insert("type".to_string(), Value::String(self.event_type.clone()));
        Value::Object(obj)
    }

    fn field_i64(&self, key: &str) -> Option<i64> {
        self.fields.get(key).and_then(Value::as_i64)
    }

    /// Adapter to an internal 4P engine event; MJAI details the engine does not use are dropped.
    pub fn to_event_4p(&self) -> Result<Event4p, CanonicalConvertError> {
        serde_json::from_value::<Event4p>(self.to_value()).map_err(|e| {
            CanonicalConvertError(format!("event {:?} -> Event4p: {e}", self.event_type))
        })
    }

    /// Adapter to an internal 3P engine event.
    pub fn to_event_3p(&self) -> Result<Event3p, CanonicalConvertError> {
        let mut value = self.to_value();
        // Internal MJAI types still use `nukidora`; the canonical name is `kita`.
        if self.event_type == "kita" {
            value["type"] = Value::String("nukidora".to_string());
        }
        serde_json::from_value::<Event3p>(value).map_err(|e| {
            CanonicalConvertError(format!("event {:?} -> Event3p: {e}", self.event_type))
        })
    }

    /// MJAI dialect input adapter. Some MJAI implementations call the 3P North event
    /// `nukidora`; it is normalized to `kita` here. Other fields are kept verbatim.
    pub fn from_mjai_value(value: Value) -> Result<Self, CanonicalConvertError> {
        let mut event: Self = serde_json::from_value(value)
            .map_err(|e| CanonicalConvertError(format!("invalid MJAI event: {e}")))?;
        if event.event_type == "nukidora" {
            event.event_type = "kita".to_string();
        }
        Ok(event)
    }

    /// Output adapter for existing MJAI 3P consumers; canonical form stays `kita`.
    pub fn to_mjai_value(&self, three_player: bool) -> Value {
        let mut value = self.to_value();
        if three_player && self.event_type == "kita" {
            value["type"] = Value::String("nukidora".to_string());
        }
        value
    }
}

/// `flya-mahjong-events-v2` versioned per-seat envelope (lossless events, reads frozen v1).
///
/// Unlike [`crate::wire::MahjongEvents`], which holds typed `Vec<Event4p/3p>` and
/// loses MJAI details, this envelope carries lossless [`CanonicalEvent`]s and is the
/// public wire of `flya-decision-v1`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MahjongEventsEnvelope {
    pub schema: MahjongEventsSchema,
    pub rule_line: RuleLineWire,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule_profile: Option<String>,
    pub viewer_seat: u8,
    pub source: EventSource,
    pub from_seq: u64,
    pub to_seq: u64,
    pub events: Vec<CanonicalEvent>,
    pub legal_actions: Vec<CanonicalLegalAction>,
    /// Digest of the full state (FNV-1a over the canonical events JSON). Required and
    /// checked on public requests.
    pub state_digest: String,
    /// Digest of the legal actions. Required and checked on public requests.
    pub legal_digest: String,
    /// Extension namespace (forward compatible).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ext: Option<Value>,
    /// New optional root fields from the same major are kept verbatim so parse -> serialize loses nothing.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Envelope validation error (stable code and redacted message).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvelopeError {
    pub code: &'static str,
    pub message: String,
}

impl EnvelopeError {
    pub(crate) fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}
impl std::error::Error for EnvelopeError {}

/// Stable digest of canonical `legal_actions` (dependency-free FNV-1a 64).
#[must_use]
pub fn compute_legal_digest(legal: &[CanonicalLegalAction]) -> String {
    stable_digest(legal)
}

/// Stable digest of canonical events.
#[must_use]
pub fn compute_state_digest(events: &[CanonicalEvent]) -> String {
    stable_digest(events)
}

fn stable_digest<T: Serialize + ?Sized>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{hash:016x}")
}

impl MahjongEventsEnvelope {
    /// Seat count (4P = 4, 3P = 3).
    #[must_use]
    pub fn seats(&self) -> u8 {
        match self.rule_line {
            RuleLineWire::Riichi4p => 4,
            RuleLineWire::Riichi3p => 3,
        }
    }

    /// Enforces every frozen invariant. Any violation returns `Err` (fail-closed).
    pub fn validate(&self) -> Result<(), EnvelopeError> {
        let seats = self.seats();
        let legacy_v1 =
            crate::wire::schema_major_matches(self.schema.as_str(), "flya-mahjong-events", 1);
        let allowed: &[&str] = match self.rule_line {
            RuleLineWire::Riichi4p => EVENT_TYPES_4P,
            RuleLineWire::Riichi3p => EVENT_TYPES_3P,
        };

        if self.viewer_seat >= seats {
            return Err(EnvelopeError::new(
                "viewer_seat_out_of_range",
                format!("viewer_seat {} >= seats {seats}", self.viewer_seat),
            ));
        }

        // Public inference requests must carry the seat's complete history from 0, not an arbitrary suffix.
        if self.from_seq != 0 {
            return Err(EnvelopeError::new(
                "partial_state_history",
                "canonical inference state must start at sequence 0",
            ));
        }
        // to_seq = from_seq + events.len().
        let want_to = self.from_seq.checked_add(self.events.len() as u64);
        if want_to != Some(self.to_seq) {
            return Err(EnvelopeError::new(
                "seq_count_mismatch",
                format!(
                    "to_seq {} != from_seq {} + events {} = {:?}",
                    self.to_seq,
                    self.from_seq,
                    self.events.len(),
                    want_to
                ),
            ));
        }

        let observed = self.source.kind == SourceKind::Observed;
        let unknown_tile = unknown_tile_value();

        for (i, ev) in self.events.iter().enumerate() {
            if legacy_v1
                && matches!(
                    ev.event_type.as_str(),
                    "dealer_opening"
                        | "dealer_opening_dahai"
                        | "seat_forced_autoplay"
                        | "seat_resumed"
                )
            {
                return Err(EnvelopeError::new(
                    "schema_feature_mismatch",
                    format!(
                        "events[{i}] type {:?} requires flya-mahjong-events-v2",
                        ev.event_type
                    ),
                ));
            }
            if ev.event_type == "none" {
                return Err(EnvelopeError::new(
                    "none_in_fact_stream",
                    format!("events[{i}] is `none`; `none` is a bot response, not a fact"),
                ));
            }
            if !allowed.contains(&ev.event_type.as_str()) {
                return Err(EnvelopeError::new(
                    "disallowed_event_type",
                    format!(
                        "events[{i}] type {:?} not allowed for {}",
                        ev.event_type,
                        rule_line_str(self.rule_line)
                    ),
                ));
            }
            // Run the typed adapters at the boundary so a missing actor/pai or a wrong array
            // shape fails here rather than at runtime.
            let converted = match self.rule_line {
                RuleLineWire::Riichi4p => ev.to_event_4p().map(|_| ()),
                RuleLineWire::Riichi3p => ev.to_event_3p().map(|_| ()),
            };
            converted.map_err(|e| {
                EnvelopeError::new("invalid_event_shape", format!("events[{i}]: {e}"))
            })?;

            for key in ["scores", "deltas", "tehais"] {
                if let Some(values) = ev.fields.get(key).and_then(Value::as_array)
                    && values.len() != usize::from(seats)
                {
                    return Err(EnvelopeError::new(
                        "seat_array_width_mismatch",
                        format!(
                            "events[{i}].{key} has {} entries, expected {seats}",
                            values.len()
                        ),
                    ));
                }
            }
            // Seat range of actor/target, when present.
            for key in ["actor", "target"] {
                if let Some(v) = ev.fields.get(key) {
                    let seat = v.as_i64().ok_or_else(|| {
                        EnvelopeError::new(
                            "bad_event_seat",
                            format!("events[{i}].{key} must be an integer"),
                        )
                    })?;
                    if seat < 0 || seat >= i64::from(seats) {
                        return Err(EnvelopeError::new(
                            "seat_out_of_range",
                            format!("events[{i}].{key}={seat} out of range [0,{seats})"),
                        ));
                    }
                }
            }

            if observed {
                validate_observed_privacy(i, ev, self.viewer_seat, &unknown_tile)?;
            }
        }

        // legal_actions: `action_id` equals the index, and each converts to an internal
        // legal action (which also checks no chi in 3P and no kita in 4P).
        for (i, la) in self.legal_actions.iter().enumerate() {
            if legacy_v1
                && matches!(
                    la.action,
                    crate::wire::CanonicalAction::DealerOpeningDahai { .. }
                        | crate::wire::CanonicalAction::DealerOpeningRiichiDahai { .. }
                        | crate::wire::CanonicalAction::Tsumo { pai: None }
                )
            {
                return Err(EnvelopeError::new(
                    "schema_feature_mismatch",
                    format!(
                        "legal_actions[{i}] type {:?} requires flya-actions-v2",
                        la.action.wire_type()
                    ),
                ));
            }
            if la.action_id != i {
                return Err(EnvelopeError::new(
                    "action_id_not_sequential",
                    format!("legal_actions[{i}].action_id={} != {i}", la.action_id),
                ));
            }
            let conv = match self.rule_line {
                RuleLineWire::Riichi4p => la.action.to_legal_4p().map(|_| ()),
                RuleLineWire::Riichi3p => la.action.to_legal_3p().map(|_| ()),
            };
            conv.map_err(|e| {
                EnvelopeError::new("illegal_legal_action", format!("legal_actions[{i}]: {e}"))
            })?;
            if let crate::wire::CanonicalAction::Ron { target, .. } = la.action
                && target >= seats
            {
                return Err(EnvelopeError::new(
                    "action_target_out_of_range",
                    format!("legal_actions[{i}].target={target} out of range [0,{seats})"),
                ));
            }
        }

        let actual_state = compute_state_digest(&self.events);
        if self.state_digest != actual_state {
            return Err(EnvelopeError::new(
                "state_digest_mismatch",
                "state_digest does not match canonical events",
            ));
        }
        let actual_legal = compute_legal_digest(&self.legal_actions);
        if self.legal_digest != actual_legal {
            return Err(EnvelopeError::new(
                "legal_digest_mismatch",
                "legal_digest does not match legal_actions",
            ));
        }
        if observed {
            if let Some(ext) = &self.ext {
                reject_sensitive_extension(ext, "ext")?;
            }
            for (key, value) in &self.extra {
                reject_sensitive_extension(value, key)?;
            }
        }

        Ok(())
    }

    /// Event-type-level validation and sanitization for the public entry point.
    ///
    /// Runs every envelope invariant from [`validate`](Self::validate), then passes each
    /// event through [`crate::public_event::validate_public_canonical_event`] for
    /// per-event field checks and rebuilds it. Returns a sanitized envelope with rebuilt
    /// events and a recomputed `state_digest`.
    ///
    /// Inference must only receive the sanitized envelope, never the raw flattened
    /// fields, so private inputs such as `obs/tensor/mask/features` hidden under valid
    /// field names are rejected here. For valid input (native events, `Hora` scoring
    /// present or absent, registered MJAI details) the sanitized envelope is identical and
    /// `state_digest` does not change.
    pub fn sanitize_public(&self) -> Result<MahjongEventsEnvelope, EnvelopeError> {
        self.validate()?;
        let mut sanitized_events = Vec::with_capacity(self.events.len());
        for (i, event) in self.events.iter().enumerate() {
            let clean = crate::public_event::validate_public_canonical_event(self.rule_line, event)
                .map_err(|e| EnvelopeError::new(e.code, format!("events[{i}]: {}", e.message)))?;
            sanitized_events.push(clean);
        }
        let state_digest = compute_state_digest(&sanitized_events);
        Ok(MahjongEventsEnvelope {
            schema: self.schema.clone(),
            rule_line: self.rule_line,
            rule_profile: self.rule_profile.clone(),
            viewer_seat: self.viewer_seat,
            source: self.source.clone(),
            from_seq: self.from_seq,
            to_seq: self.to_seq,
            events: sanitized_events,
            legal_actions: self.legal_actions.clone(),
            state_digest,
            legal_digest: self.legal_digest.clone(),
            ext: self.ext.clone(),
            extra: self.extra.clone(),
        })
    }

    /// Converts to internal 4P engine events (call `validate` first, or use a 4P rule line).
    pub fn to_events_4p(&self) -> Result<Vec<Event4p>, CanonicalConvertError> {
        self.events
            .iter()
            .map(CanonicalEvent::to_event_4p)
            .collect()
    }

    /// Converts to internal 3P engine events.
    pub fn to_events_3p(&self) -> Result<Vec<Event3p>, CanonicalConvertError> {
        self.events
            .iter()
            .map(CanonicalEvent::to_event_3p)
            .collect()
    }

    /// Builds an observed envelope (other players' hands and draws masked as unknown)
    /// from an authoritative 4P event stream and the seat's legal actions, with
    /// `from_seq = 0`, `to_seq = events.len()`, recomputed digests, and
    /// [`Self::validate`] already run.
    ///
    /// Steps: (1) project each event for `viewer_seat` with [`crate::project_4p`];
    /// (2) reshape into [`Event4p`] ([`visible_body_to_event_4p`]); (3) convert to lossless
    /// [`CanonicalEvent`]s through [`canonical_from_native`], the same path as
    /// [`native_canonical_event_matrix`]; (4) convert `legal_actions` 1:1 into
    /// [`CanonicalLegalAction`]s in input order (`action_id` = index, so callers can map
    /// `selected_index` back); (5) compute digests; (6) `validate()`.
    ///
    /// Remote seats and other consumers use this so projection and encoding have one
    /// implementation.
    pub fn observed_from_typed_4p(
        events: &[Event4p],
        viewer_seat: u8,
        legal: &[LegalAction4p],
    ) -> Result<Self, EnvelopeError> {
        let canonical_events: Vec<CanonicalEvent> = events
            .iter()
            .map(|ev| {
                let visible = project_4p(ev, viewer_seat);
                canonical_from_native(&visible_body_to_event_4p(visible.event))
            })
            .collect();
        let legal_actions = CanonicalLegalAction::table_from_legal_4p(legal);
        let state_digest = compute_state_digest(&canonical_events);
        let legal_digest = compute_legal_digest(&legal_actions);
        let to_seq = canonical_events.len() as u64;
        let envelope = MahjongEventsEnvelope {
            schema: MahjongEventsSchema::default(),
            rule_line: RuleLineWire::Riichi4p,
            rule_profile: None,
            viewer_seat,
            source: EventSource {
                kind: SourceKind::Observed,
                epoch: 0,
                status: None,
            },
            from_seq: 0,
            to_seq,
            events: canonical_events,
            legal_actions,
            state_digest,
            legal_digest,
            ext: None,
            extra: Map::new(),
        };
        envelope.validate()?;
        Ok(envelope)
    }

    /// 3P version of [`Self::observed_from_typed_4p`], using [`crate::project_3p`] /
    /// [`Event3p`] / [`LegalAction3p`].
    pub fn observed_from_typed_3p(
        events: &[Event3p],
        viewer_seat: u8,
        legal: &[LegalAction3p],
    ) -> Result<Self, EnvelopeError> {
        let canonical_events: Vec<CanonicalEvent> = events
            .iter()
            .map(|ev| {
                let visible = project_3p(ev, viewer_seat);
                canonical_from_native(&visible_body_to_event_3p(visible.event))
            })
            .collect();
        let legal_actions = CanonicalLegalAction::table_from_legal_3p(legal);
        let state_digest = compute_state_digest(&canonical_events);
        let legal_digest = compute_legal_digest(&legal_actions);
        let to_seq = canonical_events.len() as u64;
        let envelope = MahjongEventsEnvelope {
            schema: MahjongEventsSchema::default(),
            rule_line: RuleLineWire::Riichi3p,
            rule_profile: None,
            viewer_seat,
            source: EventSource {
                kind: SourceKind::Observed,
                epoch: 0,
                status: None,
            },
            from_seq: 0,
            to_seq,
            events: canonical_events,
            legal_actions,
            state_digest,
            legal_digest,
            ext: None,
            extra: Map::new(),
        };
        envelope.validate()?;
        Ok(envelope)
    }
}

fn rule_line_str(rl: RuleLineWire) -> &'static str {
    match rl {
        RuleLineWire::Riichi4p => "riichi4p",
        RuleLineWire::Riichi3p => "riichi3p",
    }
}

/// Canonical JSON value of the unknown tile (other players' hands and draws in observed projections).
fn unknown_tile_value() -> Value {
    serde_json::to_value(Tile::unknown()).unwrap_or(Value::String("?".to_string()))
}

/// Information boundary: observed projections must not carry seeds, other players'
/// hands or draws, or the wall.
fn validate_observed_privacy(
    i: usize,
    ev: &CanonicalEvent,
    viewer_seat: u8,
    unknown_tile: &Value,
) -> Result<(), EnvelopeError> {
    match ev.event_type.as_str() {
        "start_game" => {
            if ev.fields.get("seed").is_some_and(|v| !v.is_null()) {
                return Err(EnvelopeError::new(
                    "observed_leaks_seed",
                    format!("events[{i}] start_game carries seed in an observed stream"),
                ));
            }
        }
        "start_kyoku" => {
            if let Some(tehais) = ev.fields.get("tehais").and_then(Value::as_array) {
                for (seat, hand) in tehais.iter().enumerate() {
                    if seat == viewer_seat as usize {
                        continue;
                    }
                    let leaks = hand
                        .as_array()
                        .map(|tiles| tiles.iter().any(|t| t != unknown_tile))
                        .unwrap_or(false);
                    if leaks {
                        return Err(EnvelopeError::new(
                            "observed_leaks_hidden_tiles",
                            format!(
                                "events[{i}] start_kyoku leaks seat {seat} concealed hand in observed stream"
                            ),
                        ));
                    }
                }
            }
        }
        "tsumo" | "dealer_opening" => {
            let actor = ev.field_i64("actor").unwrap_or(-1);
            let leaks = actor != i64::from(viewer_seat)
                && ev.fields.get("pai").is_some_and(|pai| pai != unknown_tile);
            if leaks {
                return Err(EnvelopeError::new(
                    "observed_leaks_hidden_tiles",
                    format!(
                        "events[{i}] {} by seat {actor} leaks concealed tile in observed stream",
                        ev.event_type
                    ),
                ));
            }
        }
        _ => {}
    }
    for (key, value) in &ev.fields {
        if key == "tehais" || (ev.event_type == "start_game" && key == "seed") {
            continue;
        }
        if sensitive_key(key) && !value.is_null() {
            return Err(EnvelopeError::new(
                "observed_leaks_hidden_state",
                format!("observed stream contains hidden field events[{i}].{key}"),
            ));
        }
        reject_sensitive_extension(value, &format!("events[{i}].{key}"))?;
    }
    Ok(())
}

fn sensitive_key(key: &str) -> bool {
    matches!(
        key,
        "seed" | "wall" | "wall_tiles" | "yama" | "dead_wall" | "hidden_hands" | "ura_dora_markers"
    )
}

fn reject_sensitive_extension(value: &Value, path: &str) -> Result<(), EnvelopeError> {
    const SENSITIVE: &[&str] = &[
        "seed",
        "wall",
        "wall_tiles",
        "yama",
        "dead_wall",
        "hidden_hands",
        "ura_dora_markers",
    ];
    match value {
        Value::Object(obj) => {
            for (key, child) in obj {
                let child_path = format!("{path}.{key}");
                if (SENSITIVE.contains(&key.as_str()) || sensitive_key(key)) && !child.is_null() {
                    return Err(EnvelopeError::new(
                        "observed_leaks_hidden_state",
                        format!("observed stream contains hidden field {child_path}"),
                    ));
                }
                reject_sensitive_extension(child, &child_path)?;
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                reject_sensitive_extension(child, &format!("{path}[{index}]"))?;
            }
        }
        _ => {}
    }
    Ok(())
}
