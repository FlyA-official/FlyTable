//! `flytable-event`: FlyTable's versioned event stream (mjai style).
//!
//! The shared language for game progression between FlyTable and external engines,
//! and between internal layers. Treat it as an interface contract.
//!
//! 4-player and 3-player have separate event enums:
//! - [`event4p::Event4p`] - `[T; 4]`, actors `0..=3`, has chi.
//! - [`event3p::Event3p`] - `[T; 3]`, actors `0..=2`, no chi, has nukidora.
//!
//! They are deliberately not unified behind generics. Only the settlement details in
//! [`scoring`] are shared. See [`PROTOCOL_VERSION`].

pub mod compat3p;
pub mod event3p;
pub mod event4p;
pub mod matchlog;
pub mod scoring;

pub use event3p::{Actor3, Event3p};
pub use event4p::{Actor4, Event4p};
pub use scoring::{HoraAgari, HoraPoint, HoraScoring, HoraYakuFlags};

/// Version of the internal event stream. Bump it whenever the shape of [`Event3p`] or
/// [`Event4p`] changes.
///
/// Independent of the [`matchlog`] schema version: crate semver, match log schema,
/// inference protocol and trace schema are versioned separately.
pub const PROTOCOL_VERSION: u32 = 2;
