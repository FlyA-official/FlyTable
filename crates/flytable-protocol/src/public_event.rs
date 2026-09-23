//! Per-event-type structural validation and sanitization for the public entry point.
//!
//! Checking that a field name is in the global set ([`is_canonical_event_field`]) is
//! not enough; the field must also belong to the current event type and its nested
//! structure. Otherwise a request such as `dahai.scoring.features` or
//! `hora.scoring.observation_data` passes: the name is valid on some event,
//! `serde_json::from_value::<Event4p>` silently ignores extra fields, the envelope
//! check passes, and the raw flattened fields get forwarded with private payloads in
//! the inference request.
//!
//! This module provides the authoritative per-event validation and rebuild:
//! - [`public_schema_4p`] / [`public_schema_3p`] map every `Event4p` / `Event3p`
//!   variant, with an exhaustive `match` and no wildcard, to its allowed MJAI detail
//!   fields and their value validators. Adding a variant fails to compile until its
//!   schema is declared.
//! - [`validate_public_canonical_event`] (1) parses the authoritative native fields
//!   through `to_event_4p/3p` and serializes them canonically, (2) checks every raw
//!   field: native fields must equal the rebuilt values exactly (so nested extra data
//!   is rejected), detail fields must be in the event's schema and pass their
//!   validator, anything else is rejected, and (3) returns a freshly built sanitized
//!   event. Downstream only ever receives the sanitized event.
//!
//! Registered standard and dialect fields are preserved verbatim; arbitrary payloads
//! added by a caller are not.

use serde_json::{Map, Value};

use flytable_core::tile::Tile;
use flytable_event::{Event3p, Event4p};

use crate::canonical_events::{CanonicalEvent, EnvelopeError, canonical_from_native};
use crate::wire::RuleLineWire;

/// Value validator for a detail field. Every validator rejects objects and free-form
/// JSON, so private payloads cannot hide under a valid field name.
#[derive(Debug, Clone, Copy)]
enum DetailKind {
    /// Own seat: an integer in `0..seats` (MJAI `start_game.id`).
    SeatId,
    /// A single canonical tile string (e.g. `hora.pai`).
    Tile,
    /// Array of tile strings (ura indicators, winning hand, tenpai hand) with length `<= max`.
    TileList { max: usize },
    /// Per-seat tile grid: outer length == seats, each inner array has length
    /// `<= inner_max` (tenpai hands revealed in `ryukyoku.tehais`).
    TileGridSeat { inner_max: usize },
    /// Yaku array of `[name (<= 32 bytes), han]` pairs, length `<= max`.
    YakuList { max: usize },
    /// Integer in `[min, max]` (fu, han, points, nukidora count).
    IntBounded { min: i64, max: i64 },
    /// Per-seat integer array: length == seats, each in `[min, max]` (scores, deltas, ranks).
    IntSeatArray { min: i64, max: i64 },
    /// Per-seat boolean array: length == seats (tenpai flags).
    BoolSeatArray,
    /// Short string of at most `max` UTF-8 bytes (draw reason).
    ShortString { max: usize },
}

/// One detail field rule (name and validator).
#[derive(Debug, Clone, Copy)]
struct DetailRule {
    name: &'static str,
    kind: DetailKind,
}

const fn rule(name: &'static str, kind: DetailKind) -> DetailRule {
    DetailRule { name, kind }
}

/// Public schema of an event type: the MJAI detail fields allowed on it. Native fields
/// are not listed; they are determined by the authoritative rebuild.
#[derive(Debug, Clone, Copy)]
struct PublicEventSchema {
    detail: &'static [DetailRule],
}

impl PublicEventSchema {
    fn detail_rule(&self, name: &str) -> Option<&DetailRule> {
        self.detail.iter().find(|r| r.name == name)
    }
}

const EMPTY: PublicEventSchema = PublicEventSchema { detail: &[] };

// Detail fields per event
//
// Only `hora`, `ryukyoku`, `end_game`, `reach_accepted` and `start_game` carry MJAI
// details; every other event accepts only its native fields.

const START_GAME_DETAIL: &[DetailRule] = &[rule("id", DetailKind::SeatId)];

const REACH_ACCEPTED_DETAIL: &[DetailRule] = &[
    rule(
        "scores",
        DetailKind::IntSeatArray {
            min: -1_000_000,
            max: 1_000_000,
        },
    ),
    rule(
        "deltas",
        DetailKind::IntSeatArray {
            min: -1_000_000,
            max: 1_000_000,
        },
    ),
];

const HORA_DETAIL: &[DetailRule] = &[
    rule("pai", DetailKind::Tile),
    rule("uradora_markers", DetailKind::TileList { max: 8 }),
    rule("hora_tehais", DetailKind::TileList { max: 18 }),
    rule("yakus", DetailKind::YakuList { max: 24 }),
    rule("fu", DetailKind::IntBounded { min: 0, max: 255 }),
    rule("fan", DetailKind::IntBounded { min: 0, max: 255 }),
    rule(
        "hora_points",
        DetailKind::IntBounded {
            min: 0,
            max: 1_000_000,
        },
    ),
    rule(
        "scores",
        DetailKind::IntSeatArray {
            min: -1_000_000,
            max: 1_000_000,
        },
    ),
    rule("nuki_dora", DetailKind::IntBounded { min: 0, max: 255 }),
];

const RYUKYOKU_DETAIL: &[DetailRule] = &[
    rule("reason", DetailKind::ShortString { max: 32 }),
    rule("tenpais", DetailKind::BoolSeatArray),
    rule("tehais", DetailKind::TileGridSeat { inner_max: 18 }),
    rule(
        "scores",
        DetailKind::IntSeatArray {
            min: -1_000_000,
            max: 1_000_000,
        },
    ),
];

const END_GAME_DETAIL: &[DetailRule] = &[
    rule(
        "scores",
        DetailKind::IntSeatArray {
            min: -1_000_000,
            max: 1_000_000,
        },
    ),
    rule("rankings", DetailKind::IntSeatArray { min: 1, max: 4 }),
];

/// 4-player public schema per variant. Exhaustive `match` with no wildcard, so a new
/// `Event4p` variant fails to compile until its schema is declared.
fn public_schema_4p(event: &Event4p) -> PublicEventSchema {
    match event {
        // `none` is not a fact event (the envelope rejects it); an empty schema keeps the match exhaustive.
        Event4p::None => EMPTY,
        Event4p::StartGame { .. } => PublicEventSchema {
            detail: START_GAME_DETAIL,
        },
        Event4p::StartKyoku { .. } => EMPTY,
        Event4p::Tsumo { .. } => EMPTY,
        Event4p::DealerOpening { .. } => EMPTY,
        Event4p::Dahai { .. } => EMPTY,
        Event4p::DealerOpeningDahai { .. } => EMPTY,
        // Actor only, no details.
        Event4p::SeatForcedAutoplay { .. } => EMPTY,
        Event4p::SeatResumed { .. } => EMPTY,
        Event4p::Chi { .. } => EMPTY,
        Event4p::Pon { .. } => EMPTY,
        Event4p::Daiminkan { .. } => EMPTY,
        Event4p::Kakan { .. } => EMPTY,
        Event4p::Ankan { .. } => EMPTY,
        Event4p::Dora { .. } => EMPTY,
        Event4p::Reach { .. } => EMPTY,
        Event4p::ReachAccepted { .. } => PublicEventSchema {
            detail: REACH_ACCEPTED_DETAIL,
        },
        Event4p::Hora { .. } => PublicEventSchema {
            detail: HORA_DETAIL,
        },
        Event4p::Ryukyoku { .. } => PublicEventSchema {
            detail: RYUKYOKU_DETAIL,
        },
        Event4p::EndKyoku => EMPTY,
        Event4p::EndGame => PublicEventSchema {
            detail: END_GAME_DETAIL,
        },
    }
}

/// 3-player public schema per variant. Exhaustive `match` with no wildcard. No `chi`;
/// adds `nukidora` (canonical `kita`).
fn public_schema_3p(event: &Event3p) -> PublicEventSchema {
    match event {
        Event3p::None => EMPTY,
        Event3p::StartGame { .. } => PublicEventSchema {
            detail: START_GAME_DETAIL,
        },
        Event3p::StartKyoku { .. } => EMPTY,
        Event3p::Tsumo { .. } => EMPTY,
        Event3p::DealerOpening { .. } => EMPTY,
        Event3p::Dahai { .. } => EMPTY,
        Event3p::DealerOpeningDahai { .. } => EMPTY,
        // Actor only, no details.
        Event3p::SeatForcedAutoplay { .. } => EMPTY,
        Event3p::SeatResumed { .. } => EMPTY,
        Event3p::Pon { .. } => EMPTY,
        Event3p::Daiminkan { .. } => EMPTY,
        Event3p::Kakan { .. } => EMPTY,
        Event3p::Ankan { .. } => EMPTY,
        Event3p::Nukidora { .. } => EMPTY,
        Event3p::Dora { .. } => EMPTY,
        Event3p::Reach { .. } => EMPTY,
        Event3p::ReachAccepted { .. } => PublicEventSchema {
            detail: REACH_ACCEPTED_DETAIL,
        },
        Event3p::Hora { .. } => PublicEventSchema {
            detail: HORA_DETAIL,
        },
        Event3p::Ryukyoku { .. } => PublicEventSchema {
            detail: RYUKYOKU_DETAIL,
        },
        Event3p::EndKyoku => EMPTY,
        Event3p::EndGame => PublicEventSchema {
            detail: END_GAME_DETAIL,
        },
    }
}

/// Union of all registered MJAI detail field names, derived from
/// [`public_schema_4p`] / [`public_schema_3p`] rather than maintained as a second list.
/// Each sample from [`crate::canonical_events::native_canonical_event_matrix`] (one per
/// fact event variant) is converted back to a native event and its registered detail
/// fields are collected.
///
/// This is the only source for the dialect-detail part of
/// [`crate::canonical_events::is_canonical_event_field`], so that check and
/// [`validate_public_canonical_event`] always read the same registry.
#[must_use]
pub fn all_registered_mjai_detail_fields() -> std::collections::BTreeSet<&'static str> {
    let mut fields = std::collections::BTreeSet::new();
    for (rule_line, event) in crate::canonical_events::native_canonical_event_matrix() {
        let detail = match rule_line {
            RuleLineWire::Riichi4p => {
                let native = event
                    .to_event_4p()
                    .expect("native_canonical_event_matrix 4p sample adapts to Event4p");
                public_schema_4p(&native).detail
            }
            RuleLineWire::Riichi3p => {
                let native = event
                    .to_event_3p()
                    .expect("native_canonical_event_matrix 3p sample adapts to Event3p");
                public_schema_3p(&native).detail
            }
        };
        fields.extend(detail.iter().map(|rule| rule.name));
    }
    fields
}

const fn seats_of(rule_line: RuleLineWire) -> usize {
    match rule_line {
        RuleLineWire::Riichi4p => 4,
        RuleLineWire::Riichi3p => 3,
    }
}

fn err(code: &'static str, message: impl Into<String>) -> EnvelopeError {
    EnvelopeError::new(code, message)
}

/// Validates an event against its type and returns a freshly built sanitized event.
///
/// Downstream only receives the return value, never the raw flattened map. Any
/// violation returns `Err` (fail-closed):
/// - `disallowed_event_type` / `none_in_fact_stream`: type not allowed for the rule line, or `none`.
/// - `invalid_event_shape`: native fields missing or malformed.
/// - `public_event_extension_forbidden`: the event carries `ext`.
/// - `event_field_value_mismatch`: a native field carries nested extra data (e.g. `hora.scoring.features`).
/// - `field_not_allowed_on_event`: the field is not registered on this event.
/// - `detail_field_bad_shape`: a detail field fails its validator.
pub fn validate_public_canonical_event(
    rule_line: RuleLineWire,
    event: &CanonicalEvent,
) -> Result<CanonicalEvent, EnvelopeError> {
    if event.event_type == "none" {
        return Err(err(
            "none_in_fact_stream",
            "`none` is a bot response, not a fact event",
        ));
    }

    let seats = seats_of(rule_line);

    // 1. Parse the authoritative native representation. The adapter drops details the
    //    engine does not use; missing or malformed native fields surface here as
    //    invalid_event_shape, as do event types from the other rule line (no chi in 3P,
    //    no kita in 4P).
    let (schema, reconstructed) = match rule_line {
        RuleLineWire::Riichi4p => {
            let native = event
                .to_event_4p()
                .map_err(|e| err("invalid_event_shape", e.0))?;
            (public_schema_4p(&native), canonical_from_native(&native))
        }
        RuleLineWire::Riichi3p => {
            let native = event
                .to_event_3p()
                .map_err(|e| err("invalid_event_shape", e.0))?;
            (public_schema_3p(&native), canonical_from_native(&native))
        }
    };

    // 2. Check every raw field and build the sanitized event.
    let mut sanitized_fields: Map<String, Value> = Map::new();
    for (key, value) in &event.fields {
        if key == "ext" {
            return Err(err(
                "public_event_extension_forbidden",
                "public event extensions are not forwarded; use top-level namespaced extensions",
            ));
        }
        if let Some(canonical_value) = reconstructed.fields.get(key) {
            // Native fields must equal the rebuilt value exactly. Nested extra data (such as
            // features inside scoring) makes them differ and is rejected rather than silently
            // dropped.
            if value != canonical_value {
                return Err(err(
                    "event_field_value_mismatch",
                    format!(
                        "event {:?} native field {key:?} carries data outside its canonical schema",
                        event.event_type
                    ),
                ));
            }
            sanitized_fields.insert(key.clone(), value.clone());
        } else if let Some(detail) = schema.detail_rule(key) {
            validate_detail(detail.kind, value, seats).map_err(|reason| {
                err(
                    "detail_field_bad_shape",
                    format!(
                        "event {:?} detail field {key:?} is invalid: {reason}",
                        event.event_type
                    ),
                )
            })?;
            // Registered detail fields are kept verbatim.
            sanitized_fields.insert(key.clone(), value.clone());
        } else {
            return Err(err(
                "field_not_allowed_on_event",
                format!(
                    "field {key:?} is not allowed on canonical event {:?}",
                    event.event_type
                ),
            ));
        }
    }

    Ok(CanonicalEvent {
        event_type: reconstructed.event_type.clone(),
        fields: sanitized_fields,
    })
}

fn validate_detail(kind: DetailKind, value: &Value, seats: usize) -> Result<(), String> {
    match kind {
        DetailKind::SeatId => {
            let n = as_int(value)?;
            if n < 0 || n as usize >= seats {
                return Err(format!("seat id {n} out of range [0,{seats})"));
            }
        }
        DetailKind::Tile => {
            parse_tile(value)?;
        }
        DetailKind::TileList { max } => {
            let arr = as_array(value)?;
            if arr.len() > max {
                return Err(format!("tile list len {} exceeds {max}", arr.len()));
            }
            for t in arr {
                parse_tile(t)?;
            }
        }
        DetailKind::TileGridSeat { inner_max } => {
            let arr = as_array(value)?;
            if arr.len() != seats {
                return Err(format!("seat grid width {} != {seats}", arr.len()));
            }
            for hand in arr {
                let tiles = as_array(hand)?;
                if tiles.len() > inner_max {
                    return Err(format!("hand len {} exceeds {inner_max}", tiles.len()));
                }
                for t in tiles {
                    parse_tile(t)?;
                }
            }
        }
        DetailKind::YakuList { max } => {
            let arr = as_array(value)?;
            if arr.len() > max {
                return Err(format!("yaku list len {} exceeds {max}", arr.len()));
            }
            for pair in arr {
                let pair = as_array(pair)?;
                if pair.len() != 2 {
                    return Err("yaku entry must be [name, fan]".to_string());
                }
                let name = pair[0]
                    .as_str()
                    .ok_or_else(|| "yaku name must be a string".to_string())?;
                if name.len() > 32 {
                    return Err("yaku name exceeds 32 bytes".to_string());
                }
                let fan = as_int(&pair[1])?;
                if !(0..=255).contains(&fan) {
                    return Err(format!("yaku fan {fan} out of range"));
                }
            }
        }
        DetailKind::IntBounded { min, max } => {
            let n = as_int(value)?;
            if n < min || n > max {
                return Err(format!("value {n} out of range [{min},{max}]"));
            }
        }
        DetailKind::IntSeatArray { min, max } => {
            let arr = as_array(value)?;
            if arr.len() != seats {
                return Err(format!("seat array width {} != {seats}", arr.len()));
            }
            for v in arr {
                let n = as_int(v)?;
                if n < min || n > max {
                    return Err(format!("value {n} out of range [{min},{max}]"));
                }
            }
        }
        DetailKind::BoolSeatArray => {
            let arr = as_array(value)?;
            if arr.len() != seats {
                return Err(format!("seat array width {} != {seats}", arr.len()));
            }
            for v in arr {
                if !v.is_boolean() {
                    return Err("seat array entry must be a boolean".to_string());
                }
            }
        }
        DetailKind::ShortString { max } => {
            let s = value
                .as_str()
                .ok_or_else(|| "value must be a string".to_string())?;
            if s.len() > max {
                return Err(format!("string len {} exceeds {max}", s.len()));
            }
        }
    }
    Ok(())
}

fn as_int(value: &Value) -> Result<i64, String> {
    value
        .as_i64()
        .ok_or_else(|| "value must be an integer".to_string())
}

fn as_array(value: &Value) -> Result<&Vec<Value>, String> {
    value
        .as_array()
        .ok_or_else(|| "value must be an array".to_string())
}

fn parse_tile(value: &Value) -> Result<Tile, String> {
    let s = value
        .as_str()
        .ok_or_else(|| "tile must be a string".to_string())?;
    s.parse::<Tile>().map_err(|_| format!("invalid tile {s:?}"))
}
