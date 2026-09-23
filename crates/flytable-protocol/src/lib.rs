//! Contract types for the inference plugin protocol.
//!
//! Defines per-seat visible events, full-granularity legal actions, inference result
//! and capability types, and the privacy projection from `Event4p/3p` to
//! `VisibleEvent4p/3p`. No host, networking, subprocess, model or observation
//! encoding code.

use flytable_core::tile::Tile;
use flytable_event::{Actor3, Actor4, Event3p, Event4p, HoraScoring};

pub mod canonical_events;
pub mod matchlog_adapters;
pub mod matchlog_replay;
pub mod public_event;
pub mod wire;

pub use canonical_events::{
    CANONICAL_BASE_EVENT_FIELDS, CanonicalEvent, EVENT_TYPES_3P, EVENT_TYPES_4P, EnvelopeError,
    MahjongEventsEnvelope, compute_legal_digest, compute_state_digest, is_canonical_event_field,
    native_canonical_event_matrix, visible_body_to_event_3p, visible_body_to_event_4p,
};
pub use public_event::{all_registered_mjai_detail_fields, validate_public_canonical_event};
pub use wire::{
    CanonicalAction, CanonicalConvertError, CanonicalLegalAction, Declines, EventSource,
    FLYA_ACTIONS_V1, FLYA_ACTIONS_V2, FLYA_MAHJONG_EVENTS_V1, FLYA_MAHJONG_EVENTS_V2,
    MahjongEvents, MahjongEvents3p, MahjongEvents4p, MahjongEventsSchema, RuleLineWire, SourceKind,
    schema_major_matches,
};

/// Frozen inference protocol v1.
pub const FLYA_INFERENCE_PROTOCOL_V1: &str = "flya-inference-v1";
/// Current protocol. Adds the originless dealer opening events and actions; the
/// host projects them for v1 clients.
pub const FLYA_INFERENCE_PROTOCOL_V2: &str = "flya-inference-v2";

/// 4-player per-seat visible event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibleEvent4p {
    pub viewer_seat: Actor4,
    pub event: VisibleBody4p,
}

/// 4-player visible event body.
///
/// Same shape as [`Event4p`], except `StartGame.seed` is removed and
/// `StartKyoku.tehais` / `Tsumo.pai` are masked for other seats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisibleBody4p {
    None,
    StartGame {
        names: [String; 4],
    },
    StartKyoku {
        bakaze: Tile,
        dora_marker: Tile,
        kyoku: u8,
        honba: u8,
        kyotaku: u8,
        oya: Actor4,
        scores: [i32; 4],
        tehais: [[Tile; 13]; 4],
    },
    Tsumo {
        actor: Actor4,
        pai: Tile,
    },
    DealerOpening {
        actor: Actor4,
        pai: Tile,
    },
    Dahai {
        actor: Actor4,
        pai: Tile,
        tsumogiri: bool,
    },
    DealerOpeningDahai {
        actor: Actor4,
        pai: Tile,
    },
    /// Entering or leaving forced autoplay is public; not masked.
    SeatForcedAutoplay {
        actor: Actor4,
    },
    SeatResumed {
        actor: Actor4,
    },
    Chi {
        actor: Actor4,
        target: Actor4,
        pai: Tile,
        consumed: [Tile; 2],
    },
    Pon {
        actor: Actor4,
        target: Actor4,
        pai: Tile,
        consumed: [Tile; 2],
    },
    Daiminkan {
        actor: Actor4,
        target: Actor4,
        pai: Tile,
        consumed: [Tile; 3],
    },
    Kakan {
        actor: Actor4,
        pai: Tile,
        consumed: [Tile; 3],
    },
    Ankan {
        actor: Actor4,
        consumed: [Tile; 4],
    },
    Dora {
        dora_marker: Tile,
    },
    Reach {
        actor: Actor4,
    },
    ReachAccepted {
        actor: Actor4,
    },
    Hora {
        actor: Actor4,
        target: Actor4,
        deltas: Option<[i32; 4]>,
        ura_markers: Option<Vec<Tile>>,
        scoring: Option<HoraScoring>,
    },
    Ryukyoku {
        deltas: Option<[i32; 4]>,
    },
    EndKyoku,
    EndGame,
}

/// 3-player per-seat visible event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibleEvent3p {
    pub viewer_seat: Actor3,
    pub event: VisibleBody3p,
}

/// 3-player visible event body.
///
/// Same shape as [`Event3p`], except `StartGame.seed` is removed and
/// `StartKyoku.tehais` / `Tsumo.pai` are masked for other seats. `Nukidora` is public.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisibleBody3p {
    None,
    StartGame {
        names: [String; 3],
    },
    StartKyoku {
        bakaze: Tile,
        dora_marker: Tile,
        kyoku: u8,
        honba: u8,
        kyotaku: u8,
        oya: Actor3,
        scores: [i32; 3],
        tehais: [[Tile; 13]; 3],
    },
    Tsumo {
        actor: Actor3,
        pai: Tile,
    },
    DealerOpening {
        actor: Actor3,
        pai: Tile,
    },
    Dahai {
        actor: Actor3,
        pai: Tile,
        tsumogiri: bool,
    },
    DealerOpeningDahai {
        actor: Actor3,
        pai: Tile,
    },
    /// Entering or leaving forced autoplay is public; not masked.
    SeatForcedAutoplay {
        actor: Actor3,
    },
    SeatResumed {
        actor: Actor3,
    },
    Pon {
        actor: Actor3,
        target: Actor3,
        pai: Tile,
        consumed: [Tile; 2],
    },
    Daiminkan {
        actor: Actor3,
        target: Actor3,
        pai: Tile,
        consumed: [Tile; 3],
    },
    Kakan {
        actor: Actor3,
        pai: Tile,
        consumed: [Tile; 3],
    },
    Ankan {
        actor: Actor3,
        consumed: [Tile; 4],
    },
    Nukidora {
        actor: Actor3,
        pai: Tile,
    },
    Dora {
        dora_marker: Tile,
    },
    Reach {
        actor: Actor3,
    },
    ReachAccepted {
        actor: Actor3,
    },
    Hora {
        actor: Actor3,
        target: Actor3,
        deltas: Option<[i32; 3]>,
        ura_markers: Option<Vec<Tile>>,
        scoring: Option<HoraScoring>,
    },
    Ryukyoku {
        deltas: Option<[i32; 3]>,
    },
    EndKyoku,
    EndGame,
}

/// Full event that can be projected into a per-seat visible event.
pub trait ProjectVisible {
    type ViewerSeat;
    type VisibleEvent;

    fn project_visible(&self, viewer_seat: Self::ViewerSeat) -> Self::VisibleEvent;
}

/// Per-seat projection entry point.
///
/// Rust has no overloading, so a trait keeps the `project(&event, seat)` call shape;
/// the actual projection is the per-variant match in [`project_4p`] / [`project_3p`].
pub fn project<E>(event: &E, viewer_seat: E::ViewerSeat) -> E::VisibleEvent
where
    E: ProjectVisible,
{
    event.project_visible(viewer_seat)
}

impl ProjectVisible for Event4p {
    type ViewerSeat = Actor4;
    type VisibleEvent = VisibleEvent4p;

    fn project_visible(&self, viewer_seat: Self::ViewerSeat) -> Self::VisibleEvent {
        project_4p(self, viewer_seat)
    }
}

impl ProjectVisible for Event3p {
    type ViewerSeat = Actor3;
    type VisibleEvent = VisibleEvent3p;

    fn project_visible(&self, viewer_seat: Self::ViewerSeat) -> Self::VisibleEvent {
        project_3p(self, viewer_seat)
    }
}

/// Explicit per-variant projection from `Event4p` to `VisibleEvent4p`.
pub fn project_4p(event: &Event4p, viewer_seat: Actor4) -> VisibleEvent4p {
    let event = match event {
        Event4p::None => VisibleBody4p::None,
        Event4p::StartGame { names, seed: _ } => VisibleBody4p::StartGame {
            names: names.clone(),
        },
        Event4p::StartKyoku {
            bakaze,
            dora_marker,
            kyoku,
            honba,
            kyotaku,
            oya,
            scores,
            tehais,
        } => VisibleBody4p::StartKyoku {
            bakaze: *bakaze,
            dora_marker: *dora_marker,
            kyoku: *kyoku,
            honba: *honba,
            kyotaku: *kyotaku,
            oya: *oya,
            scores: *scores,
            tehais: project_tehais_4p(tehais, viewer_seat),
        },
        Event4p::Tsumo { actor, pai } => VisibleBody4p::Tsumo {
            actor: *actor,
            pai: visible_tsumo_pai(*actor, *pai, viewer_seat),
        },
        Event4p::DealerOpening { actor, pai } => VisibleBody4p::DealerOpening {
            actor: *actor,
            pai: visible_tsumo_pai(*actor, *pai, viewer_seat),
        },
        Event4p::Dahai {
            actor,
            pai,
            tsumogiri,
        } => VisibleBody4p::Dahai {
            actor: *actor,
            pai: *pai,
            tsumogiri: *tsumogiri,
        },
        Event4p::DealerOpeningDahai { actor, pai } => VisibleBody4p::DealerOpeningDahai {
            actor: *actor,
            pai: *pai,
        },
        // A disconnect is visible to everyone.
        Event4p::SeatForcedAutoplay { actor } => {
            VisibleBody4p::SeatForcedAutoplay { actor: *actor }
        }
        Event4p::SeatResumed { actor } => VisibleBody4p::SeatResumed { actor: *actor },
        Event4p::Chi {
            actor,
            target,
            pai,
            consumed,
        } => VisibleBody4p::Chi {
            actor: *actor,
            target: *target,
            pai: *pai,
            consumed: *consumed,
        },
        Event4p::Pon {
            actor,
            target,
            pai,
            consumed,
        } => VisibleBody4p::Pon {
            actor: *actor,
            target: *target,
            pai: *pai,
            consumed: *consumed,
        },
        Event4p::Daiminkan {
            actor,
            target,
            pai,
            consumed,
        } => VisibleBody4p::Daiminkan {
            actor: *actor,
            target: *target,
            pai: *pai,
            consumed: *consumed,
        },
        Event4p::Kakan {
            actor,
            pai,
            consumed,
        } => VisibleBody4p::Kakan {
            actor: *actor,
            pai: *pai,
            consumed: *consumed,
        },
        Event4p::Ankan { actor, consumed } => VisibleBody4p::Ankan {
            actor: *actor,
            consumed: *consumed,
        },
        Event4p::Dora { dora_marker } => VisibleBody4p::Dora {
            dora_marker: *dora_marker,
        },
        Event4p::Reach { actor } => VisibleBody4p::Reach { actor: *actor },
        Event4p::ReachAccepted { actor } => VisibleBody4p::ReachAccepted { actor: *actor },
        Event4p::Hora {
            actor,
            target,
            deltas,
            ura_markers,
            scoring,
        } => VisibleBody4p::Hora {
            actor: *actor,
            target: *target,
            deltas: *deltas,
            ura_markers: ura_markers.clone(),
            scoring: *scoring,
        },
        Event4p::Ryukyoku { deltas } => VisibleBody4p::Ryukyoku { deltas: *deltas },
        Event4p::EndKyoku => VisibleBody4p::EndKyoku,
        Event4p::EndGame => VisibleBody4p::EndGame,
    };

    VisibleEvent4p { viewer_seat, event }
}

/// Explicit per-variant projection from `Event3p` to `VisibleEvent3p`.
pub fn project_3p(event: &Event3p, viewer_seat: Actor3) -> VisibleEvent3p {
    let event = match event {
        Event3p::None => VisibleBody3p::None,
        Event3p::StartGame { names, seed: _ } => VisibleBody3p::StartGame {
            names: names.clone(),
        },
        Event3p::StartKyoku {
            bakaze,
            dora_marker,
            kyoku,
            honba,
            kyotaku,
            oya,
            scores,
            tehais,
        } => VisibleBody3p::StartKyoku {
            bakaze: *bakaze,
            dora_marker: *dora_marker,
            kyoku: *kyoku,
            honba: *honba,
            kyotaku: *kyotaku,
            oya: *oya,
            scores: *scores,
            tehais: project_tehais_3p(tehais, viewer_seat),
        },
        Event3p::Tsumo { actor, pai } => VisibleBody3p::Tsumo {
            actor: *actor,
            pai: visible_tsumo_pai(*actor, *pai, viewer_seat),
        },
        Event3p::DealerOpening { actor, pai } => VisibleBody3p::DealerOpening {
            actor: *actor,
            pai: visible_tsumo_pai(*actor, *pai, viewer_seat),
        },
        Event3p::Dahai {
            actor,
            pai,
            tsumogiri,
        } => VisibleBody3p::Dahai {
            actor: *actor,
            pai: *pai,
            tsumogiri: *tsumogiri,
        },
        Event3p::DealerOpeningDahai { actor, pai } => VisibleBody3p::DealerOpeningDahai {
            actor: *actor,
            pai: *pai,
        },
        // A disconnect is visible to everyone.
        Event3p::SeatForcedAutoplay { actor } => {
            VisibleBody3p::SeatForcedAutoplay { actor: *actor }
        }
        Event3p::SeatResumed { actor } => VisibleBody3p::SeatResumed { actor: *actor },
        Event3p::Pon {
            actor,
            target,
            pai,
            consumed,
        } => VisibleBody3p::Pon {
            actor: *actor,
            target: *target,
            pai: *pai,
            consumed: *consumed,
        },
        Event3p::Daiminkan {
            actor,
            target,
            pai,
            consumed,
        } => VisibleBody3p::Daiminkan {
            actor: *actor,
            target: *target,
            pai: *pai,
            consumed: *consumed,
        },
        Event3p::Kakan {
            actor,
            pai,
            consumed,
        } => VisibleBody3p::Kakan {
            actor: *actor,
            pai: *pai,
            consumed: *consumed,
        },
        Event3p::Ankan { actor, consumed } => VisibleBody3p::Ankan {
            actor: *actor,
            consumed: *consumed,
        },
        Event3p::Nukidora { actor, pai } => VisibleBody3p::Nukidora {
            actor: *actor,
            pai: *pai,
        },
        Event3p::Dora { dora_marker } => VisibleBody3p::Dora {
            dora_marker: *dora_marker,
        },
        Event3p::Reach { actor } => VisibleBody3p::Reach { actor: *actor },
        Event3p::ReachAccepted { actor } => VisibleBody3p::ReachAccepted { actor: *actor },
        Event3p::Hora {
            actor,
            target,
            deltas,
            ura_markers,
            scoring,
        } => VisibleBody3p::Hora {
            actor: *actor,
            target: *target,
            deltas: *deltas,
            ura_markers: ura_markers.clone(),
            scoring: *scoring,
        },
        Event3p::Ryukyoku { deltas } => VisibleBody3p::Ryukyoku { deltas: *deltas },
        Event3p::EndKyoku => VisibleBody3p::EndKyoku,
        Event3p::EndGame => VisibleBody3p::EndGame,
    };

    VisibleEvent3p { viewer_seat, event }
}

fn project_tehais_4p(tehais: &[[Tile; 13]; 4], viewer_seat: Actor4) -> [[Tile; 13]; 4] {
    let mut visible = [[Tile::unknown(); 13]; 4];
    if let Some(dst) = visible.get_mut(viewer_seat as usize) {
        *dst = tehais[viewer_seat as usize];
    }
    visible
}

fn project_tehais_3p(tehais: &[[Tile; 13]; 3], viewer_seat: Actor3) -> [[Tile; 13]; 3] {
    let mut visible = [[Tile::unknown(); 13]; 3];
    if let Some(dst) = visible.get_mut(viewer_seat as usize) {
        *dst = tehais[viewer_seat as usize];
    }
    visible
}

fn visible_tsumo_pai<A: PartialEq>(actor: A, pai: Tile, viewer_seat: A) -> Tile {
    if actor == viewer_seat {
        pai
    } else {
        Tile::unknown()
    }
}

/// Full-granularity 4-player legal action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LegalAction4p {
    Discard {
        pai: Tile,
        tsumogiri: bool,
        riichi: bool,
    },
    DealerOpeningDiscard {
        pai: Tile,
        riichi: bool,
    },
    Kan {
        pai: Tile,
        kind: KanKind,
        consumed: Vec<Tile>,
    },
    Tsumo {
        pai: Tile,
    },
    DealerOpeningTsumo,
    Kyushukyuhai,
    PassAll {
        declines: ResponseOpportunities,
    },
    Pon {
        pai: Tile,
        consumed: [Tile; 2],
    },
    Chi {
        pai: Tile,
        consumed: [Tile; 2],
    },
    Ron {
        pai: Tile,
        target: Actor4,
    },
}

impl LegalAction4p {
    /// Wire `type` string defined by the protocol.
    pub const fn wire_type(&self) -> &'static str {
        match self {
            LegalAction4p::Discard { .. } => "dahai",
            LegalAction4p::DealerOpeningDiscard { .. } => "dealer_opening_dahai",
            LegalAction4p::Kan { kind, .. } => kind.wire_type(),
            LegalAction4p::Tsumo { .. } => "tsumo",
            LegalAction4p::DealerOpeningTsumo => "tsumo",
            LegalAction4p::Kyushukyuhai => "kyushukyuhai",
            LegalAction4p::PassAll { .. } => "pass_all",
            LegalAction4p::Pon { .. } => "pon",
            LegalAction4p::Chi { .. } => "chi",
            LegalAction4p::Ron { .. } => "ron",
        }
    }
}

/// Full-granularity 3-player legal action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LegalAction3p {
    Discard {
        pai: Tile,
        tsumogiri: bool,
        riichi: bool,
    },
    DealerOpeningDiscard {
        pai: Tile,
        riichi: bool,
    },
    Kan {
        pai: Tile,
        kind: KanKind,
        consumed: Vec<Tile>,
    },
    Nukidora,
    Tsumo {
        pai: Tile,
    },
    DealerOpeningTsumo,
    Kyushukyuhai,
    PassAll {
        declines: ResponseOpportunities,
    },
    Pon {
        pai: Tile,
        consumed: [Tile; 2],
    },
    Ron {
        pai: Tile,
        target: Actor3,
    },
}

impl LegalAction3p {
    /// Wire `type` string defined by the protocol.
    pub const fn wire_type(&self) -> &'static str {
        match self {
            LegalAction3p::Discard { .. } => "dahai",
            LegalAction3p::DealerOpeningDiscard { .. } => "dealer_opening_dahai",
            LegalAction3p::Kan { kind, .. } => kind.wire_type(),
            LegalAction3p::Nukidora => "nukidora",
            LegalAction3p::Tsumo { .. } => "tsumo",
            LegalAction3p::DealerOpeningTsumo => "tsumo",
            LegalAction3p::Kyushukyuhai => "kyushukyuhai",
            LegalAction3p::PassAll { .. } => "pass_all",
            LegalAction3p::Pon { .. } => "pon",
            LegalAction3p::Ron { .. } => "ron",
        }
    }
}

/// Kan type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KanKind {
    Ankan,
    Kakan,
    Daiminkan,
}

impl KanKind {
    /// Wire `type` string for each kan type.
    pub const fn wire_type(self) -> &'static str {
        match self {
            KanKind::Ankan => "ankan",
            KanKind::Kakan => "kakan",
            KanKind::Daiminkan => "daiminkan",
        }
    }
}

/// Opportunities that a single pass in this response window gives up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResponseOpportunities {
    pub ron: bool,
    pub call: bool,
}

/// Opaque meta payload, kept as raw JSON text so this crate needs no serde.
///
/// On the wire it is bounded opaque JSON; this crate does not interpret it.
pub type InferenceMeta = Box<str>;

/// Inference decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceDecision {
    Select {
        index: usize,
        meta: Option<InferenceMeta>,
    },
    Abstain,
}

/// Inference error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceError {
    pub kind: InferenceErrorKind,
    pub detail: Box<str>,
}

/// Kind of inference error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferenceErrorKind {
    InvalidAction,
    StaleDecision,
    StateSyncRequired,
    Timeout,
    Protocol,
    Internal,
    /// Failed before the request was written or sent to the provider. Only this kind may
    /// be retried safely.
    PreSendUnavailable,
    EngineUnavailable,
}

/// Engine capability declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineCaps {
    pub caps_version: u32,
    pub protocol_versions: Vec<Box<str>>,
    pub rule_lines: Vec<RuleLine>,
    pub riichi_style: RiichiStyle,
    pub supports_incremental: bool,
    pub returns_ranked_actions: bool,
}

/// Supported rule lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuleLine {
    Riichi4p,
    Riichi3p,
}

impl RuleLine {
    pub const fn wire_value(self) -> &'static str {
        match self {
            RuleLine::Riichi4p => "riichi4p",
            RuleLine::Riichi3p => "riichi3p",
        }
    }
}

/// How the model handles riichi. Metadata only for the host; does not change the call path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RiichiStyle {
    ParallelDiscard,
    DeclareThenDiscard,
}

impl RiichiStyle {
    pub const fn wire_value(self) -> &'static str {
        match self {
            RiichiStyle::ParallelDiscard => "parallel_discard",
            RiichiStyle::DeclareThenDiscard => "declare_then_discard",
        }
    }
}

// Recommendation annotations
//
// The display channel for model output. Translators normalize whatever a model
// produces into annotations tagged with scope, value type and display style. The
// host only validates structure (valid type tags, `action_ref` within the legal
// actions, probabilities in [0, 1]) and never interprets meaning (`id` and `title`
// are opaque strings passed through to the frontend).
//
// Annotations are display-only and never affect play. The authoritative action is
// always the one chosen in [`InferenceDecision`]; the host may drop invalid
// annotations, whole or individually, without changing the game. The frontend
// renders by `type`, so one UI works with any model.
//
// This crate only defines the canonical data shape. JSON parsing, validation and
// normalization live in `flytable-inference-host`, the same split as
// [`LegalAction4p`] and the host's `legal_action_4p_to_wire`.

/// All recommendation annotations for one decision.
#[derive(Debug, Clone, PartialEq)]
pub struct Recommendation {
    /// Annotations bound to a specific legal action (primary metric plus attributes).
    pub actions: Vec<ActionAnnotation>,
    /// Global information not bound to an action (confidence, commentary, ...).
    pub global: Vec<Annotation>,
}

/// Annotations on one legal action.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionAnnotation {
    /// Index into the request's `legal_actions` (the host checks `0..legal_len`).
    pub action_ref: usize,
    /// Primary metric shown on the main recommendation (probability or rank).
    pub primary: Option<PrimaryMetric>,
    /// Optional text; the frontend can derive a default from the action.
    pub label: Option<String>,
    /// Per-action attributes (danger, intent, ...).
    pub attributes: Vec<Annotation>,
}

/// Primary metric of a legal action.
#[derive(Debug, Clone, PartialEq)]
pub enum PrimaryMetric {
    /// Probability in [0, 1].
    Probability(f64),
    /// Rank, starting at 1 (lower is better).
    Rank(u32),
}

impl PrimaryMetric {
    /// Wire `kind` string.
    pub const fn wire_kind(&self) -> &'static str {
        match self {
            PrimaryMetric::Probability(_) => "probability",
            PrimaryMetric::Rank(_) => "rank",
        }
    }
}

/// One piece of auxiliary information, per action or global.
#[derive(Debug, Clone, PartialEq)]
pub struct Annotation {
    /// Stable ID used by the frontend for grouping and color; opaque to the host.
    pub id: String,
    /// Optional display title.
    pub title: Option<String>,
    pub value: AnnotationValue,
    /// Display style (meaningful per action; global entries use the default).
    pub display: AnnotationDisplay,
}

/// Value type of an annotation.
#[derive(Debug, Clone, PartialEq)]
pub enum AnnotationValue {
    /// Number with a display format (raw or percent).
    Number { value: f64, format: NumberFormat },
    /// Semantic ID with translator-provided text; the frontend shows `label`, or `value` if there is none.
    SemanticId {
        value: String,
        label: Option<String>,
    },
    /// Free text (for example commentary from an LLM).
    Text { value: String },
}

impl AnnotationValue {
    /// Wire `type` string.
    pub const fn wire_type(&self) -> &'static str {
        match self {
            AnnotationValue::Number { .. } => "number",
            AnnotationValue::SemanticId { .. } => "semantic_id",
            AnnotationValue::Text { .. } => "text",
        }
    }
}

/// Number display format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberFormat {
    Raw,
    /// Percentage (`0.95` shows as 95%).
    Percent,
}

impl NumberFormat {
    pub const fn wire_value(self) -> &'static str {
        match self {
            NumberFormat::Raw => "raw",
            NumberFormat::Percent => "percent",
        }
    }
}

/// Display style token chosen by the translator. `title` is the heading,
/// `type` / `format` decide value formatting, `id` is only for grouping and color.
/// Unknown tokens pass through as `Other` and the frontend falls back to rendering
/// by `type`, so new styles need no host changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnnotationDisplay {
    /// Shown under the top-N primary recommendations (per action).
    Follow,
    /// Legacy alias of `Wall`.
    Tiled,
    /// One value per legal tile (per action).
    Wall,
    /// `Wall` colored by a 0..1 value (per action, number).
    Heat,
    /// Text card with title and body (global text).
    Note,
    /// Title and value (global number, optionally percent).
    Stat,
    /// Progress bar, 0..1 (global number).
    Meter,
    /// Badge with title and label (global semantic ID).
    Badge,
    /// Unknown style token, passed through unchanged.
    Other(String),
}

impl AnnotationDisplay {
    pub fn wire_value(&self) -> &str {
        match self {
            AnnotationDisplay::Follow => "follow",
            AnnotationDisplay::Tiled => "tiled",
            AnnotationDisplay::Wall => "wall",
            AnnotationDisplay::Heat => "heat",
            AnnotationDisplay::Note => "note",
            AnnotationDisplay::Stat => "stat",
            AnnotationDisplay::Meter => "meter",
            AnnotationDisplay::Badge => "badge",
            AnnotationDisplay::Other(token) => token.as_str(),
        }
    }

    /// Parses a style token: known tokens map to variants, unknown non-empty tokens to
    /// `Other`, empty to `Follow`.
    pub fn from_wire(token: &str) -> Self {
        match token {
            "follow" => AnnotationDisplay::Follow,
            "tiled" => AnnotationDisplay::Tiled,
            "wall" => AnnotationDisplay::Wall,
            "heat" => AnnotationDisplay::Heat,
            "note" => AnnotationDisplay::Note,
            "stat" => AnnotationDisplay::Stat,
            "meter" => AnnotationDisplay::Meter,
            "badge" => AnnotationDisplay::Badge,
            other if !other.is_empty() => AnnotationDisplay::Other(other.to_string()),
            _ => AnnotationDisplay::Follow,
        }
    }
}
