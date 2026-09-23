//! `flytable-runtime`: FlyTable's seat bus and match host.
//!
//! Brings the game loop (draw, decide, call window, adjudicate, continue) and the
//! question of who decides for each seat into one place.
//!
//! - [`SeatAgent`] - common seat decision interface (a superset of
//!   [`SeatDecider`](flytable_seat::SeatDecider)). The host asks the current seat for
//!   an action the same way regardless of agent type and adjudicates it by the rules.
//!   - [`AlgorithmSeatAgent`]: in-process rule-based seat (built-in tsumogiri).
//!   - [`LocalPluginSeatAgent`]: local certified plugin seat, calling a translator
//!     subprocess over `flya-inference-v2` (with frozen v1) through
//!     `flytable-inference-host::SubprocessHost`.
//!   - Remote / ObservedExternal / Observer have variants in [`SeatAgentKind`] (some
//!     still placeholders).
//! - [`MatchHost`] - generalizes the CLI self-play loop into asking agents per seat, runs
//!   full matches ([`run_match_4p`] / [`run_match_3p`]) and records a
//!   [`DecisionTrace`] for every move.
//!
//! Boundaries:
//! - No observation encoding, action masks, model tensors, torch or onnxruntime. Only
//!   the event protocol: events and legal actions out, a legal list index back.
//! - Rule truth stays in `flytable-core/event/table/seat`; this crate only orchestrates.
//! - Process side effects are confined to `flytable-inference-host`.

pub mod agent;
pub mod algorithm;
pub mod catalog;
pub mod decision_window;
pub mod live;
pub mod match_host;
pub mod matchlog_archive;
pub mod placeholders;
pub mod plugin;
pub mod status;
pub mod variant;

pub use agent::{
    AgentChoice, DecisionTrace, ReactionRequest, SeatAgent, SeatAgentKind, SeatAgentStatus,
    TurnRequest,
};
pub use algorithm::{AlgorithmKind, AlgorithmSeatAgent};
pub use catalog::{
    builtin_algorithms, discover_plugins, find_plugin, list_catalog, CatalogEntry, CatalogHealth,
    CatalogSource,
};
pub use decision_window::{reaction_window, turn_window, validate_window, HostWindow, WindowFault};
pub use live::{
    run_full_match_3p_via_live, run_full_match_4p_via_live, LiveBoard, LiveMatchConfig,
    LiveMatchSession, LiveMatchSession3p, LiveMatchSession4p, LiveMatchSnapshot, LiveMatchStatus,
    LiveSeat, LiveSeatStatus, LiveStepResult, MeldSnapshot, PendingKind, PendingReactionSeat,
    PendingSummary, PendingWindow, SeatSnapshot,
};
pub use match_host::{
    run_full_match_3p, run_full_match_3p_with_agents, run_full_match_4p,
    run_full_match_4p_with_agents, run_match_3p, run_match_4p, CurrentKyokuAttempt,
    FullMatchReport, KyokuRecord, MatchConfig, MatchFailure, MatchKind, MatchReport, SeatSpec,
    SessionIdPolicy, TRACE_SCHEMA_VERSION,
};
pub use matchlog_archive::{
    archive_kyoku_3p, archive_kyoku_4p, start_match_event, ArchiveError, KyokuArchive, MatchArchive,
};
pub use placeholders::{
    ManualSeat, ObservedInput, ObservedSourceStatus, ObserverConfig, RemoteModelConfig,
    RemoteModelError, RemoteModelSeat,
};
pub use plugin::LocalPluginSeatAgent;
pub use status::{RuntimeStatus, SeatRuntimeStatus};
pub use variant::{Variant, Variant3p, Variant4p};

/// Lets consumers (the CLI) name the plugin runtime type without depending on `flytable-inference-host`.
pub use flytable_inference_host::registry::CertifiedPluginRuntime;
