//! FlyTable inference host.
//!
//! Starts external model and translator subprocesses, talks to them over
//! `flya-inference-v2` (with frozen v1), and discovers and certifies third-party
//! plugins. All process and filesystem side effects live here, physically separate
//! from the rule crates (core/event/table/seat), which stay free of processes,
//! scanning, networking and models.
//!
//! Products can depend on this crate directly to reuse the same integration logic.
//!
//! - [`host`] - the `SubprocessHost` pipe (hello/infer/end, digest echo, exact match
//!   against the legal list).
//! - [`registry`] - `plugin.toml` scanning, `package_hash`, the `.flya-cert.json`
//!   certification cache, and smoke tests on a real board.

pub mod driver;
pub mod host;
pub mod registry;

pub use driver::{
    DriverHealth, DriverStage, InferenceDriver, LocalProcessDriver, ProviderMetrics,
    ProviderPolicy, ProviderScheduledDriver, RemoteProviderDriver, StageDecision,
};
