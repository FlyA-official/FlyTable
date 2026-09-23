//! Projection from engine events to `flytable-matchlog-v1` (see [`engine_events`]).

pub mod engine_events;

pub use engine_events::{
    project_3p as matchlog_from_engine_3p, project_3p_indexed as matchlog_from_engine_3p_indexed,
    project_4p as matchlog_from_engine_4p, project_4p_indexed as matchlog_from_engine_4p_indexed,
};
