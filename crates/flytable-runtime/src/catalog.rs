//! Local model catalog [`list_catalog`]: a stable merged view of built-in algorithms
//! and local plugins for products to pass through.
//!
//! The runtime is the only authority for the local catalog (built-in algorithms plus
//! `flytable-inference-host::registry`). It does not go online, and remote models are
//! not in the local catalog. Scanning and certification results (including reasons
//! for failed and skipped entries) come from the registry.

use std::path::Path;

use flytable_inference_host::registry::{
    certified_plugin_runtimes, scan_models_root, CertifiedPluginRuntime, PluginScanStatus,
    SmokeSummary,
};

use crate::algorithm::AlgorithmKind;

/// Source of a catalog entry. `RemoteModel` is a placeholder (never produced locally).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogSource {
    BuiltinAlgorithm,
    LocalPlugin,
    /// Remote model (authoritative online). Placeholder: not listed locally, no networking.
    RemoteModel,
}

impl CatalogSource {
    /// Seat agent kind for this source (matches `SeatAgentKind::as_str`).
    pub const fn seat_agent_kind(self) -> &'static str {
        match self {
            CatalogSource::BuiltinAlgorithm => "algorithm",
            CatalogSource::LocalPlugin => "local_plugin",
            CatalogSource::RemoteModel => "remote_model",
        }
    }
}

/// Health / certification state of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogHealth {
    /// Built-in algorithm: always usable (rules only, no certification).
    Builtin,
    /// Local plugin: certified in this scan.
    Certified,
    /// Local plugin: certification cache hit (package_hash unchanged).
    CertifiedCached,
    /// Local plugin: certification failed (see `error_summary`).
    Failed,
    /// Local plugin: skipped (see `error_summary`).
    Skipped,
}

/// One entry of the merged catalog (stable schema for products).
#[derive(Debug, Clone, serde::Serialize)]
pub struct CatalogEntry {
    /// Model id: `tsumogiri` for the built-in algorithm, the registry-normalized
    /// `flya-plugin:...` for plugins. Failed or skipped plugins have no stable id and are
    /// `None` (identified by `plugin_dir` / `display_name`).
    pub model_id: Option<String>,
    pub display_name: String,
    pub source: CatalogSource,
    /// Seat agent kind (algorithm / local_plugin / remote_model).
    pub seat_agent_kind: &'static str,
    /// Supported rule lines (riichi4p / riichi3p).
    pub rule_lines: Vec<String>,
    /// Riichi style (parallel_discard / declare_then_discard); `None` for built-in algorithms.
    pub riichi_style: Option<String>,
    pub health: CatalogHealth,
    /// Whether it can take a seat directly (certified / certified_cached / builtin).
    pub usable: bool,
    /// Failure or skip reason (already redacted by the registry).
    pub error_summary: Option<String>,
    /// Plugin directory (local plugins only).
    pub plugin_dir: Option<String>,
    /// Package content fingerprint (local plugins only; transient files excluded, stable across scans).
    pub package_hash: Option<String>,
    /// Certification smoke summary (decisions, all legal, max latency).
    pub smoke: Option<SmokeSummary>,
}

/// Built-in algorithms (seat-count independent, support 4-player and 3-player).
pub fn builtin_algorithms() -> Vec<CatalogEntry> {
    [AlgorithmKind::Tsumogiri]
        .into_iter()
        .map(|kind| CatalogEntry {
            model_id: Some(kind.model_id().to_string()),
            display_name: kind.model_id().to_string(),
            source: CatalogSource::BuiltinAlgorithm,
            seat_agent_kind: CatalogSource::BuiltinAlgorithm.seat_agent_kind(),
            rule_lines: vec!["riichi4p".to_string(), "riichi3p".to_string()],
            riichi_style: None,
            health: CatalogHealth::Builtin,
            usable: true,
            error_summary: None,
            plugin_dir: None,
            package_hash: None,
            smoke: None,
        })
        .collect()
}

/// Scans and certifies local plugins under models_root (through the registry).
pub fn discover_plugins(models_root: impl AsRef<Path>) -> Vec<CertifiedPluginRuntime> {
    certified_plugin_runtimes(models_root)
}

/// Finds a certified plugin runtime by model_id (the registry-normalized
/// `flya-plugin:...`, not the directory name).
pub fn find_plugin(
    models_root: impl AsRef<Path>,
    model_id: &str,
) -> Option<CertifiedPluginRuntime> {
    discover_plugins(models_root)
        .into_iter()
        .find(|rt| rt.model_id == model_id)
}

/// Merged catalog (built-in algorithms plus local plugins, including failed and
/// skipped entries for display). With `models_root = None` only built-in algorithms
/// are listed.
pub fn list_catalog(models_root: Option<&Path>) -> Vec<CatalogEntry> {
    let mut out = builtin_algorithms();
    let Some(root) = models_root else {
        return out;
    };
    out.extend(local_plugin_catalog_entries(root));
    out
}

/// Local plugin catalog entries (certified, failed and skipped). Scanning and certification happen only here.
pub(crate) fn local_plugin_catalog_entries(root: &Path) -> Vec<CatalogEntry> {
    let mut out = Vec::new();
    // Certified runtimes (with model_id / package_hash / smoke / cached).
    let certified = certified_plugin_runtimes(root);
    for rt in &certified {
        out.push(CatalogEntry {
            model_id: Some(rt.model_id.clone()),
            display_name: rt.name.clone(),
            source: CatalogSource::LocalPlugin,
            seat_agent_kind: CatalogSource::LocalPlugin.seat_agent_kind(),
            rule_lines: vec![rt.rule_line.clone()],
            riichi_style: Some(rt.riichi_style.clone()),
            health: if rt.cached {
                CatalogHealth::CertifiedCached
            } else {
                CatalogHealth::Certified
            },
            usable: true,
            error_summary: None,
            plugin_dir: Some(rt.plugin_dir.display().to_string()),
            package_hash: Some(rt.package_hash.clone()),
            smoke: Some(rt.smoke.clone()),
        });
    }

    // Failed and skipped entries from the scan report are still listed with reasons, but cannot take a seat.
    let report = scan_models_root(root);
    for entry in report.plugins {
        let (health, error_summary) = match &entry.status {
            // Certified entries were collected above.
            PluginScanStatus::Certified { .. } => continue,
            PluginScanStatus::Failed { reason } => (CatalogHealth::Failed, Some(reason.clone())),
            PluginScanStatus::Skipped { reason } => (CatalogHealth::Skipped, Some(reason.clone())),
        };
        out.push(CatalogEntry {
            model_id: None,
            display_name: entry.name,
            source: CatalogSource::LocalPlugin,
            seat_agent_kind: CatalogSource::LocalPlugin.seat_agent_kind(),
            rule_lines: entry.rule_line.into_iter().collect(),
            riichi_style: entry.riichi_style,
            health,
            usable: false,
            error_summary,
            plugin_dir: Some(entry.plugin_dir.display().to_string()),
            package_hash: entry.package_hash,
            smoke: None,
        });
    }
    out
}
