use std::cmp;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use flytable_core::meld::Meld;
use flytable_core::tile::Tile;
use flytable_event::{Event3p, Event4p};
use flytable_seat::contract::{
    EngineCaps, InferenceDecision, KanKind, LegalAction3p, LegalAction4p, RiichiStyle, RuleLine,
    FLYA_INFERENCE_PROTOCOL_V1, FLYA_INFERENCE_PROTOCOL_V2,
};
use flytable_table::legal::legal_turn_actions;
use flytable_table::product::{Mirror3p, Mirror4p, MirrorCenter};
use flytable_table::{Board3p, Board4p, SeatView, TurnAction};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::host::{
    DecisionPhase, EngineProcessConfig, InferenceInput3p, InferenceInput4p, SourceInfo,
    SubprocessHost,
};

pub const CERT_FILE_NAME: &str = ".flya-cert.json";
pub const PACKAGE_MANIFEST_FILE_NAME: &str = "flya-package.toml";
pub const PACKAGE_CERT_DIR_NAME: &str = ".flya-certs";
pub const SMOKE_DECISIONS: usize = 8;
pub const PLUGIN_SECURITY_WARNING: &str = "Loading plugin packages executes arbitrary code; certification is a protocol smoke check, not a sandbox.";
pub const PLUGIN_CERTIFICATION_LIMIT: &str = "Certification checks launch, handshake, caps, digest echo, and sampled legal decisions only; runtime legal-action matching remains the safety boundary.";

#[derive(Debug, Clone, Deserialize)]
pub struct PluginManifest {
    pub schema: u32,
    pub name: String,
    pub protocol: String,
    pub rule_line: String,
    pub riichi_style: String,
    pub launch: LaunchManifest,
    pub caps: Option<ManifestCaps>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LaunchManifest {
    pub cmd: String,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ManifestCaps {
    #[serde(default)]
    pub supports_incremental: bool,
    #[serde(default)]
    pub returns_ranked_actions: bool,
}

#[derive(Debug, Clone)]
struct ValidatedManifest {
    name: String,
    protocol: String,
    rule_line: RuleLine,
    riichi_style: RiichiStyle,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PackageManifest {
    pub schema: u32,
    pub package_id: String,
    pub display_name: String,
    pub protocol: String,
    pub riichi_style: String,
    pub translator: PackageTranslatorManifest,
    pub caps: Option<ManifestCaps>,
    #[serde(default)]
    pub models: Vec<PackageModelManifest>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PackageTranslatorManifest {
    pub exe: Option<String>,
    pub py: Option<String>,
    pub python: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PackageModelManifest {
    pub id: String,
    pub display_name: String,
    pub rule_line: String,
    pub kind: String,
    pub path: Option<String>,
    pub remote_backend: Option<String>,
    pub weight_file: Option<String>,
    pub remote_model: Option<String>,
    pub engine: Option<PackageEngineManifest>,
    pub caps: Option<ManifestCaps>,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PackageEngineManifest {
    pub kind: String,
    pub location: String,
    pub protocol: String,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Debug, Clone)]
struct ValidatedPackage {
    schema: u32,
    package_id: String,
    protocol: String,
    riichi_style: RiichiStyle,
    translator: PackageTranslator,
}

#[derive(Debug, Clone)]
struct PackageTranslator {
    launch_cmd: PathBuf,
    launch_prefix_args: Vec<String>,
}

#[derive(Debug, Clone)]
struct ValidatedPackageModel {
    slug: String,
    rule_line: RuleLine,
    riichi_style: RiichiStyle,
    kind: PackageModelKind,
    launch_args: Vec<String>,
}

#[derive(Debug, Clone)]
enum PackageModelKind {
    Onnx {
        path: String,
    },
    Remote {
        backend: String,
    },
    RuntimeLocal {
        weight_file: String,
        engine: ValidatedPackageEngine,
    },
    RuntimeRemote {
        remote_model: String,
        engine: ValidatedPackageEngine,
    },
}

#[derive(Debug, Clone)]
struct ValidatedPackageEngine {
    kind: PackageEngineKind,
    location: String,
    protocol: String,
    args: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackageEngineKind {
    Local,
    Remote,
}

#[derive(Debug, Clone, Serialize)]
pub struct PluginScanReport {
    pub models_root: PathBuf,
    pub security_warning: &'static str,
    pub certification_limit: &'static str,
    pub scanned: usize,
    pub certified: usize,
    pub failed: usize,
    pub skipped: usize,
    pub plugins: Vec<PluginScanEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PluginScanEntry {
    pub name: String,
    pub plugin_dir: PathBuf,
    pub layout: String,
    pub package_id: Option<String>,
    pub model_slug: String,
    pub rule_line: Option<String>,
    pub riichi_style: Option<String>,
    pub package_hash: Option<String>,
    #[serde(flatten)]
    pub status: PluginScanStatus,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PluginScanStatus {
    Certified {
        cached: bool,
        smoke: SmokeSummary,
        cert_path: PathBuf,
    },
    Failed {
        reason: String,
    },
    Skipped {
        reason: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmokeSummary {
    pub decisions: usize,
    pub all_legal: bool,
    pub dealer_opening_checked: bool,
    pub max_infer_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginCertificate {
    pub schema: u32,
    pub package_hash: String,
    pub protocol: String,
    pub rule_line: String,
    pub caps: CertificateCaps,
    pub smoke: SmokeSummary,
    pub validated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateCaps {
    pub riichi_style: String,
    pub supports_incremental: bool,
    pub returns_ranked_actions: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CertifiedPluginRuntime {
    pub model_id: String,
    pub name: String,
    pub plugin_dir: PathBuf,
    pub rule_line: String,
    pub riichi_style: String,
    pub protocol: String,
    pub launch_cmd: PathBuf,
    pub launch_args: Vec<String>,
    pub package_hash: String,
    pub cached: bool,
    pub smoke: SmokeSummary,
}

/// Executor kind of a certified runtime. Comes from the model manifest covered by
/// `package_hash`, not from caller hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CertifiedRuntimeDriverKind {
    LocalProcess,
    RemoteProvider,
}

impl CertifiedRuntimeDriverKind {
    #[must_use]
    pub const fn wire_value(self) -> &'static str {
        match self {
            Self::LocalProcess => "local_process",
            Self::RemoteProvider => "remote_provider",
        }
    }
}

/// Reads the executor kind of a certified runtime from its manifest again and
/// re-verifies the model digest.
///
/// `CertifiedPluginRuntime` keeps its public shape for downstream code using struct
/// literals. The kind is still decided by `models.kind`, which is covered by the
/// certification and package hash, so requests cannot forge it. Legacy plugin
/// manifests have no remote semantics and are always `local_process`.
pub fn certified_runtime_driver_kind(
    runtime: &CertifiedPluginRuntime,
) -> Result<CertifiedRuntimeDriverKind> {
    let manifest_path = runtime.plugin_dir.join(PACKAGE_MANIFEST_FILE_NAME);
    if !manifest_path.is_file() {
        return Ok(CertifiedRuntimeDriverKind::LocalProcess);
    }
    let manifest = read_package_manifest(&runtime.plugin_dir)?;
    let package = validate_package_manifest(&runtime.plugin_dir, &manifest)
        .map_err(|error| anyhow!(error))?;
    let model = manifest
        .models
        .iter()
        .find(|model| model.id == runtime.name && model.rule_line == runtime.rule_line)
        .ok_or_else(|| anyhow!("certified runtime model is missing from package manifest"))?;
    let validated = validate_package_model(&runtime.plugin_dir, &package, model)
        .map_err(|error| anyhow!(error))?;
    let digest = package_model_hash(&runtime.plugin_dir, &manifest, model, &validated)?;
    if digest != runtime.package_hash {
        return Err(anyhow!(
            "certified runtime package digest changed while resolving driver kind"
        ));
    }
    Ok(driver_kind_of_package_model(&validated.kind))
}

fn driver_kind_of_package_model(kind: &PackageModelKind) -> CertifiedRuntimeDriverKind {
    match kind {
        PackageModelKind::Remote { .. } | PackageModelKind::RuntimeRemote { .. } => {
            CertifiedRuntimeDriverKind::RemoteProvider
        }
        PackageModelKind::Onnx { .. } | PackageModelKind::RuntimeLocal { .. } => {
            CertifiedRuntimeDriverKind::LocalProcess
        }
    }
}

pub fn scan_models_root(models_root: impl AsRef<Path>) -> PluginScanReport {
    let models_root = models_root.as_ref().to_path_buf();
    let mut plugins = Vec::new();
    for (folder, rule_line) in [("4p", RuleLine::Riichi4p), ("3p", RuleLine::Riichi3p)] {
        let rule_dir = models_root.join(folder);
        if !rule_dir.exists() {
            continue;
        }
        let Ok(entries) = fs::read_dir(&rule_dir) else {
            plugins.push(PluginScanEntry::skipped(
                rule_dir,
                format!("failed to read {}", folder),
            ));
            continue;
        };
        for entry in entries.flatten() {
            let plugin_dir = entry.path();
            if !plugin_dir.is_dir() {
                continue;
            }
            let manifest_path = plugin_dir.join("plugin.toml");
            if !manifest_path.exists() {
                continue;
            }
            plugins.push(scan_plugin_dir(&plugin_dir, rule_line));
        }
    }
    if let Ok(entries) = fs::read_dir(&models_root) {
        let mut package_dirs = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .filter(|path| {
                !matches!(
                    path.file_name().and_then(|name| name.to_str()),
                    Some("4p" | "3p")
                )
            })
            .filter(|path| path.join(PACKAGE_MANIFEST_FILE_NAME).exists())
            .collect::<Vec<_>>();
        package_dirs.sort();
        for package_dir in package_dirs {
            plugins.extend(scan_package_dir(&package_dir));
        }
    }
    let certified = plugins
        .iter()
        .filter(|entry| matches!(entry.status, PluginScanStatus::Certified { .. }))
        .count();
    let failed = plugins
        .iter()
        .filter(|entry| matches!(entry.status, PluginScanStatus::Failed { .. }))
        .count();
    let skipped = plugins
        .iter()
        .filter(|entry| matches!(entry.status, PluginScanStatus::Skipped { .. }))
        .count();
    PluginScanReport {
        models_root,
        security_warning: PLUGIN_SECURITY_WARNING,
        certification_limit: PLUGIN_CERTIFICATION_LIMIT,
        scanned: plugins.len(),
        certified,
        failed,
        skipped,
        plugins,
    }
}

pub fn rescan_models_root(models_root: impl AsRef<Path>) -> PluginScanReport {
    scan_models_root(models_root)
}

pub fn certified_plugin_runtimes(models_root: impl AsRef<Path>) -> Vec<CertifiedPluginRuntime> {
    certified_plugin_runtimes_from_report(scan_models_root(models_root))
}

/// Convert one scan report into the certified runtimes it already proved.
///
/// Callers that need to expose failed certification reasons can scan once, inspect the
/// report, then pass it here without launching every translator a second time.
pub fn certified_plugin_runtimes_from_report(
    report: PluginScanReport,
) -> Vec<CertifiedPluginRuntime> {
    report
        .plugins
        .into_iter()
        .filter_map(certified_runtime_from_scan_entry)
        .collect()
}

fn scan_plugin_dir(plugin_dir: &Path, expected_rule_line: RuleLine) -> PluginScanEntry {
    let manifest = match read_manifest(plugin_dir) {
        Ok(manifest) => manifest,
        Err(err) => {
            return PluginScanEntry::failed(plugin_dir.to_path_buf(), format!("{err:#}"));
        }
    };
    let name = manifest.name.clone();
    let rule_line = Some(manifest.rule_line.clone());
    let riichi_style = Some(manifest.riichi_style.clone());
    let validated = match validate_manifest(&manifest, expected_rule_line) {
        Ok(validated) => validated,
        Err(err) => {
            return PluginScanEntry {
                name,
                plugin_dir: plugin_dir.to_path_buf(),
                layout: "legacy".to_string(),
                package_id: None,
                model_slug: manifest.name.clone(),
                rule_line,
                riichi_style,
                package_hash: None,
                status: PluginScanStatus::Failed { reason: err },
            };
        }
    };
    let package_hash = match package_hash(plugin_dir) {
        Ok(hash) => hash,
        Err(err) => {
            return PluginScanEntry {
                name,
                plugin_dir: plugin_dir.to_path_buf(),
                layout: "legacy".to_string(),
                package_id: None,
                model_slug: manifest.name.clone(),
                rule_line,
                riichi_style,
                package_hash: None,
                status: PluginScanStatus::Failed {
                    reason: format!("failed to hash package: {err:#}"),
                },
            };
        }
    };
    let cert_path = plugin_dir.join(CERT_FILE_NAME);
    if let Ok(cert) = read_certificate(&cert_path) {
        if cert.schema == 2
            && cert.package_hash == package_hash
            && cert.protocol == validated.protocol
            && cert.rule_line == validated.rule_line.wire_value()
            && cert.smoke.dealer_opening_checked
        {
            return PluginScanEntry {
                name,
                plugin_dir: plugin_dir.to_path_buf(),
                layout: "legacy".to_string(),
                package_id: None,
                model_slug: validated.name.clone(),
                rule_line,
                riichi_style,
                package_hash: Some(package_hash),
                status: PluginScanStatus::Certified {
                    cached: true,
                    smoke: cert.smoke,
                    cert_path,
                },
            };
        }
    }
    match certify_plugin(plugin_dir, &manifest, &validated, &package_hash) {
        Ok(smoke) => PluginScanEntry {
            name,
            plugin_dir: plugin_dir.to_path_buf(),
            layout: "legacy".to_string(),
            package_id: None,
            model_slug: validated.name.clone(),
            rule_line,
            riichi_style,
            package_hash: Some(package_hash),
            status: PluginScanStatus::Certified {
                cached: false,
                smoke,
                cert_path,
            },
        },
        Err(err) => PluginScanEntry {
            name,
            plugin_dir: plugin_dir.to_path_buf(),
            layout: "legacy".to_string(),
            package_id: None,
            model_slug: validated.name.clone(),
            rule_line,
            riichi_style,
            package_hash: Some(package_hash),
            status: PluginScanStatus::Failed {
                reason: format!("{err:#}"),
            },
        },
    }
}

fn scan_package_dir(package_dir: &Path) -> Vec<PluginScanEntry> {
    let manifest = match read_package_manifest(package_dir) {
        Ok(manifest) => manifest,
        Err(err) => {
            return vec![PluginScanEntry::failed_with_layout(
                package_dir.to_path_buf(),
                "package",
                package_dir
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(str::to_owned),
                None,
                format!("{err:#}"),
            )];
        }
    };
    let package_id = manifest.package_id.clone();
    let validated_package = match validate_package_manifest(package_dir, &manifest) {
        Ok(validated) => validated,
        Err(err) => {
            return vec![PluginScanEntry::failed_with_layout(
                package_dir.to_path_buf(),
                "package",
                Some(package_id),
                None,
                err,
            )];
        }
    };
    if manifest.models.is_empty() {
        return vec![PluginScanEntry::failed_with_layout(
            package_dir.to_path_buf(),
            "package",
            Some(package_id),
            None,
            "at least one [[models]] entry is required".to_string(),
        )];
    }

    manifest
        .models
        .iter()
        .map(|model| scan_package_model(package_dir, &manifest, &validated_package, model))
        .collect()
}

fn scan_package_model(
    package_dir: &Path,
    manifest: &PackageManifest,
    package: &ValidatedPackage,
    model: &PackageModelManifest,
) -> PluginScanEntry {
    let model_slug = if model.id.trim().is_empty() {
        package.package_id.clone()
    } else {
        model.id.clone()
    };
    let name = model_slug.clone();
    let validated = match validate_package_model(package_dir, package, model) {
        Ok(validated) => validated,
        Err(err) => {
            return package_model_failed_entry(
                package_dir,
                &package.package_id,
                &model_slug,
                Some(model.rule_line.clone()),
                Some(manifest.riichi_style.clone()),
                None,
                err,
            );
        }
    };
    let rule_line = Some(validated.rule_line.wire_value().to_string());
    let riichi_style = Some(validated.riichi_style.wire_value().to_string());
    let package_hash = match package_model_hash(package_dir, manifest, model, &validated) {
        Ok(hash) => hash,
        Err(err) => {
            return package_model_failed_entry(
                package_dir,
                &package.package_id,
                &model_slug,
                rule_line,
                riichi_style,
                None,
                format!("failed to hash package model: {err:#}"),
            );
        }
    };
    let cert_path = package_cert_path(package_dir, &validated.rule_line, &validated.slug);
    if let Ok(cert) = read_certificate(&cert_path) {
        if cert.schema == 2
            && cert.package_hash == package_hash
            && cert.protocol == package.protocol
            && cert.rule_line == validated.rule_line.wire_value()
            && cert.smoke.dealer_opening_checked
        {
            return PluginScanEntry {
                name,
                plugin_dir: package_dir.to_path_buf(),
                layout: "package".to_string(),
                package_id: Some(package.package_id.clone()),
                model_slug,
                rule_line,
                riichi_style,
                package_hash: Some(package_hash),
                status: PluginScanStatus::Certified {
                    cached: true,
                    smoke: cert.smoke,
                    cert_path,
                },
            };
        }
    }

    match certify_plugin_runtime(
        package_dir,
        package.translator.launch_cmd.clone(),
        validated.launch_args.clone(),
        &ValidatedManifest {
            name: validated.slug.clone(),
            protocol: package.protocol.clone(),
            rule_line: validated.rule_line,
            riichi_style: validated.riichi_style,
        },
        &package_hash,
        &cert_path,
    ) {
        Ok(smoke) => PluginScanEntry {
            name,
            plugin_dir: package_dir.to_path_buf(),
            layout: "package".to_string(),
            package_id: Some(package.package_id.clone()),
            model_slug,
            rule_line,
            riichi_style,
            package_hash: Some(package_hash),
            status: PluginScanStatus::Certified {
                cached: false,
                smoke,
                cert_path,
            },
        },
        Err(err) => PluginScanEntry {
            name,
            plugin_dir: package_dir.to_path_buf(),
            layout: "package".to_string(),
            package_id: Some(package.package_id.clone()),
            model_slug,
            rule_line,
            riichi_style,
            package_hash: Some(package_hash),
            status: PluginScanStatus::Failed {
                reason: format!("{err:#}"),
            },
        },
    }
}

fn package_model_failed_entry(
    package_dir: &Path,
    package_id: &str,
    model_slug: &str,
    rule_line: Option<String>,
    riichi_style: Option<String>,
    package_hash: Option<String>,
    reason: String,
) -> PluginScanEntry {
    PluginScanEntry {
        name: model_slug.to_string(),
        plugin_dir: package_dir.to_path_buf(),
        layout: "package".to_string(),
        package_id: Some(package_id.to_string()),
        model_slug: model_slug.to_string(),
        rule_line,
        riichi_style,
        package_hash,
        status: PluginScanStatus::Failed { reason },
    }
}

fn certified_runtime_from_scan_entry(entry: PluginScanEntry) -> Option<CertifiedPluginRuntime> {
    let (cached, smoke) = match &entry.status {
        PluginScanStatus::Certified { cached, smoke, .. } => (*cached, smoke.clone()),
        PluginScanStatus::Failed { .. } | PluginScanStatus::Skipped { .. } => return None,
    };
    if entry.layout == "package" {
        return certified_package_runtime_from_scan_entry(entry, cached, smoke);
    }
    let expected_rule_line = entry
        .rule_line
        .as_deref()
        .and_then(|value| parse_rule_line(value).ok())?;
    let manifest = read_manifest(&entry.plugin_dir).ok()?;
    let validated = validate_manifest(&manifest, expected_rule_line).ok()?;
    let package_hash = entry.package_hash?;
    let model_id = plugin_model_id(&validated.name, &validated.rule_line, &package_hash);
    Some(CertifiedPluginRuntime {
        model_id,
        name: validated.name,
        plugin_dir: entry.plugin_dir.clone(),
        rule_line: validated.rule_line.wire_value().to_string(),
        riichi_style: validated.riichi_style.wire_value().to_string(),
        protocol: validated.protocol,
        launch_cmd: resolve_launch_cmd(&entry.plugin_dir, &manifest.launch.cmd),
        launch_args: manifest.launch.args,
        package_hash,
        cached,
        smoke,
    })
}

fn certified_package_runtime_from_scan_entry(
    entry: PluginScanEntry,
    cached: bool,
    smoke: SmokeSummary,
) -> Option<CertifiedPluginRuntime> {
    let manifest = read_package_manifest(&entry.plugin_dir).ok()?;
    let package = validate_package_manifest(&entry.plugin_dir, &manifest).ok()?;
    // A v2 package may list the same model.id in several [[models]] entries (for
    // example 4p and 3p of one model). Matching by id alone would always hit the first
    // entry, so a 3p entry would borrow the 4p rule_line, path and hash. The rule_line
    // carried by the scan entry (written by scan_package_model after checking that
    // entry) is the only reliable disambiguator and must match too.
    let expected_rule_line = entry.rule_line.as_deref()?;
    let model = manifest
        .models
        .iter()
        .find(|model| model.id == entry.model_slug && model.rule_line == expected_rule_line)?;
    let validated = validate_package_model(&entry.plugin_dir, &package, model).ok()?;
    let package_hash = entry.package_hash?;
    let model_id = plugin_model_id(&validated.slug, &validated.rule_line, &package_hash);
    Some(CertifiedPluginRuntime {
        model_id,
        name: validated.slug,
        plugin_dir: entry.plugin_dir.clone(),
        rule_line: validated.rule_line.wire_value().to_string(),
        riichi_style: validated.riichi_style.wire_value().to_string(),
        protocol: package.protocol,
        launch_cmd: package.translator.launch_cmd,
        launch_args: validated.launch_args,
        package_hash,
        cached,
        smoke,
    })
}

fn plugin_model_id(name: &str, rule_line: &RuleLine, package_hash: &str) -> String {
    let prefix = package_hash.chars().take(16).collect::<String>();
    format!(
        "flya-plugin:{}:{}:{}",
        rule_line.wire_value(),
        sanitize_model_id_component(name),
        prefix
    )
}

fn sanitize_model_id_component(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
            out.push(ch);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "plugin".to_string()
    } else {
        trimmed.to_string()
    }
}

fn read_manifest(plugin_dir: &Path) -> Result<PluginManifest> {
    let path = plugin_dir.join("plugin.toml");
    let raw =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("invalid TOML manifest {}", path.display()))
}

fn read_package_manifest(package_dir: &Path) -> Result<PackageManifest> {
    let path = package_dir.join(PACKAGE_MANIFEST_FILE_NAME);
    let raw =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut manifest: PackageManifest = toml::from_str(&raw)
        .with_context(|| format!("invalid TOML manifest {}", path.display()))?;
    let models_dir = package_dir.join("models.d");
    if models_dir.is_dir() {
        let mut fragments = fs::read_dir(&models_dir)
            .with_context(|| format!("failed to read {}", models_dir.display()))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        fragments.sort_by_key(|entry| entry.path());
        for entry in fragments {
            let fragment_path = entry.path();
            if !entry.file_type()?.is_file()
                || fragment_path.extension().and_then(|value| value.to_str()) != Some("toml")
            {
                continue;
            }
            let raw = fs::read_to_string(&fragment_path)
                .with_context(|| format!("failed to read {}", fragment_path.display()))?;
            let model: PackageModelManifest = toml::from_str(&raw)
                .with_context(|| format!("invalid model fragment {}", fragment_path.display()))?;
            if manifest
                .models
                .iter()
                .any(|existing| existing.id == model.id && existing.rule_line == model.rule_line)
            {
                return Err(anyhow!(
                    "duplicate package model {} for {}",
                    model.id,
                    model.rule_line
                ));
            }
            manifest.models.push(model);
        }
    }
    Ok(manifest)
}

fn validate_manifest(
    manifest: &PluginManifest,
    expected_rule_line: RuleLine,
) -> std::result::Result<ValidatedManifest, String> {
    if manifest.schema != 1 {
        return Err(format!("unsupported schema {}", manifest.schema));
    }
    if manifest.name.trim().is_empty() {
        return Err("name is required".to_string());
    }
    if !is_supported_inference_protocol(&manifest.protocol) {
        return Err(format!("unsupported protocol {}", manifest.protocol));
    }
    let rule_line = parse_rule_line(&manifest.rule_line)?;
    if rule_line != expected_rule_line {
        return Err(format!(
            "rule_line {} does not match parent directory {}",
            manifest.rule_line,
            expected_rule_line.wire_value()
        ));
    }
    let riichi_style = parse_riichi_style(&manifest.riichi_style)?;
    if manifest.launch.cmd.trim().is_empty() {
        return Err("launch.cmd is required".to_string());
    }
    Ok(ValidatedManifest {
        name: manifest.name.clone(),
        protocol: manifest.protocol.clone(),
        rule_line,
        riichi_style,
    })
}

fn validate_package_manifest(
    package_dir: &Path,
    manifest: &PackageManifest,
) -> std::result::Result<ValidatedPackage, String> {
    if !matches!(manifest.schema, 2 | 3) {
        return Err(format!("unsupported schema {}", manifest.schema));
    }
    if manifest.package_id.trim().is_empty() {
        return Err("package_id is required".to_string());
    }
    if manifest.display_name.trim().is_empty() {
        return Err("display_name is required".to_string());
    }
    if let Some(dir_name) = package_dir.file_name().and_then(|name| name.to_str()) {
        if dir_name != manifest.package_id {
            return Err(format!(
                "package_id {} does not match package directory {}",
                manifest.package_id, dir_name
            ));
        }
    }
    if !is_supported_inference_protocol(&manifest.protocol) {
        return Err(format!("unsupported protocol {}", manifest.protocol));
    }
    if manifest.schema == 3 && manifest.caps.is_none() {
        return Err("schema 3 package caps are required".to_string());
    }
    let riichi_style = parse_riichi_style(&manifest.riichi_style)?;
    let translator = validate_package_translator(package_dir, &manifest.translator)?;
    Ok(ValidatedPackage {
        schema: manifest.schema,
        package_id: manifest.package_id.clone(),
        protocol: manifest.protocol.clone(),
        riichi_style,
        translator,
    })
}

fn is_supported_inference_protocol(protocol: &str) -> bool {
    matches!(
        protocol,
        FLYA_INFERENCE_PROTOCOL_V1 | FLYA_INFERENCE_PROTOCOL_V2
    )
}

fn validate_package_translator(
    package_dir: &Path,
    manifest: &PackageTranslatorManifest,
) -> std::result::Result<PackageTranslator, String> {
    let exe_rel = manifest.exe.as_deref().unwrap_or("translator.exe");
    let py_rel = manifest.py.as_deref().unwrap_or("translator.py");
    let exe_path = package_dir.join(validate_relative_manifest_path("translator.exe", exe_rel)?);
    if exe_path.is_file() {
        return Ok(PackageTranslator {
            launch_cmd: exe_path,
            launch_prefix_args: manifest.args.clone(),
        });
    }
    let py_path = validate_relative_manifest_path("translator.py", py_rel)?;
    if package_dir.join(&py_path).is_file() {
        let python = manifest.python.as_deref().unwrap_or("python");
        let mut launch_prefix_args = Vec::with_capacity(1 + manifest.args.len());
        launch_prefix_args.push(relative_slash_path_from_path(&py_path));
        launch_prefix_args.extend(manifest.args.clone());
        return Ok(PackageTranslator {
            launch_cmd: resolve_launch_cmd(package_dir, python),
            launch_prefix_args,
        });
    }
    Err(format!(
        "package must contain translator executable {} or Python translator {}",
        exe_rel, py_rel
    ))
}

fn validate_package_model(
    package_dir: &Path,
    package: &ValidatedPackage,
    model: &PackageModelManifest,
) -> std::result::Result<ValidatedPackageModel, String> {
    if model.id.trim().is_empty() {
        return Err("models.id is required".to_string());
    }
    if model.display_name.trim().is_empty() {
        return Err(format!("display_name is required for model {}", model.id));
    }
    let rule_line = parse_rule_line(&model.rule_line)?;
    if package.schema == 3 {
        return validate_runtime_package_model(package_dir, package, model, rule_line);
    }
    let kind = match model.kind.as_str() {
        "onnx" => {
            let path = model
                .path
                .as_deref()
                .ok_or_else(|| format!("path is required for onnx model {}", model.id))?;
            let rel_path = validate_relative_manifest_path("models.path", path)?;
            validate_model_path_prefix(&rel_path, rule_line)?;
            let model_path = package_dir.join(&rel_path);
            if !model_path.is_file() {
                return Err(format!(
                    "model path {} does not exist",
                    relative_slash_path_from_path(&rel_path)
                ));
            }
            PackageModelKind::Onnx {
                path: relative_slash_path_from_path(&rel_path),
            }
        }
        "remote" => {
            if model.path.is_some() {
                return Err(format!("remote model {} must not set path", model.id));
            }
            let backend = model.remote_backend.as_deref().ok_or_else(|| {
                format!("remote_backend is required for remote model {}", model.id)
            })?;
            if backend.trim().is_empty() {
                return Err(format!(
                    "remote_backend is required for remote model {}",
                    model.id
                ));
            }
            PackageModelKind::Remote {
                backend: backend.to_string(),
            }
        }
        other => {
            return Err(format!("unsupported model kind {other:?} for {}", model.id));
        }
    };
    let mut launch_args = package.translator.launch_prefix_args.clone();
    for arg in &model.args {
        launch_args.push(expand_package_arg(arg, &kind, rule_line)?);
    }
    Ok(ValidatedPackageModel {
        slug: model.id.clone(),
        rule_line,
        riichi_style: package.riichi_style,
        kind,
        launch_args,
    })
}

fn validate_runtime_package_model(
    package_dir: &Path,
    package: &ValidatedPackage,
    model: &PackageModelManifest,
    rule_line: RuleLine,
) -> std::result::Result<ValidatedPackageModel, String> {
    if model.caps.is_none() {
        return Err(format!("schema 3 model {} caps are required", model.id));
    }
    if model.path.is_some() || model.remote_backend.is_some() {
        return Err(format!(
            "schema 3 model {} must use weight_file/remote_model and engine, not path/remote_backend",
            model.id
        ));
    }
    let engine = validate_runtime_engine(package_dir, model)?;
    let kind =
        match model.kind.as_str() {
            "local" => {
                if engine.kind != PackageEngineKind::Local {
                    return Err(format!(
                        "local model {} requires [models.engine].kind = \"local\"",
                        model.id
                    ));
                }
                let weight_file = model.weight_file.as_deref().ok_or_else(|| {
                    format!("weight_file is required for local model {}", model.id)
                })?;
                let rel_path = validate_relative_manifest_path("models.weight_file", weight_file)?;
                validate_model_path_prefix(&rel_path, rule_line)?;
                let weight_path = package_dir.join(&rel_path);
                if !weight_path.is_file() {
                    return Err(format!(
                        "weight_file {} does not exist",
                        relative_slash_path_from_path(&rel_path)
                    ));
                }
                if model
                    .remote_model
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty())
                {
                    return Err(format!(
                        "local model {} must not set remote_model",
                        model.id
                    ));
                }
                PackageModelKind::RuntimeLocal {
                    weight_file: relative_slash_path_from_path(&rel_path),
                    engine,
                }
            }
            "remote" => {
                if engine.kind != PackageEngineKind::Remote {
                    return Err(format!(
                        "remote model {} requires [models.engine].kind = \"remote\"",
                        model.id
                    ));
                }
                if model
                    .weight_file
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty())
                {
                    return Err(format!(
                        "remote model {} must not set local weight_file",
                        model.id
                    ));
                }
                let remote_model = model.remote_model.as_deref().ok_or_else(|| {
                    format!("remote_model is required for remote model {}", model.id)
                })?;
                if remote_model.trim().is_empty() {
                    return Err(format!(
                        "remote_model is required for remote model {}",
                        model.id
                    ));
                }
                PackageModelKind::RuntimeRemote {
                    remote_model: remote_model.to_string(),
                    engine,
                }
            }
            other => {
                return Err(format!(
                    "schema 3 model {} kind must be \"local\" or \"remote\", got {other:?}",
                    model.id
                ));
            }
        };
    let mut launch_args = Vec::new();
    for arg in &package.translator.launch_prefix_args {
        launch_args.push(expand_runtime_package_arg(
            arg,
            package_dir,
            model,
            &kind,
            rule_line,
        )?);
    }
    for arg in &model.args {
        launch_args.push(expand_runtime_package_arg(
            arg,
            package_dir,
            model,
            &kind,
            rule_line,
        )?);
    }
    Ok(ValidatedPackageModel {
        slug: model.id.clone(),
        rule_line,
        riichi_style: package.riichi_style,
        kind,
        launch_args,
    })
}

fn validate_runtime_engine(
    package_dir: &Path,
    model: &PackageModelManifest,
) -> std::result::Result<ValidatedPackageEngine, String> {
    let engine = model
        .engine
        .as_ref()
        .ok_or_else(|| format!("[[models]] {} requires [models.engine]", model.id))?;
    if engine.protocol.trim().is_empty() {
        return Err(format!(
            "engine.protocol is required for model {}",
            model.id
        ));
    }
    if engine.location.trim().is_empty() {
        return Err(format!(
            "engine.location is required for model {}",
            model.id
        ));
    }
    let kind = match engine.kind.as_str() {
        "local" => {
            let rel_path =
                validate_relative_manifest_path("models.engine.location", &engine.location)?;
            validate_engine_path_prefix(&rel_path)?;
            let engine_path = package_dir.join(&rel_path);
            if !engine_path.is_file() {
                return Err(format!(
                    "engine.location {} does not exist",
                    relative_slash_path_from_path(&rel_path)
                ));
            }
            PackageEngineKind::Local
        }
        "remote" => {
            if !(engine.location.starts_with("http://") || engine.location.starts_with("https://"))
            {
                return Err(format!(
                    "remote engine.location for model {} must be an http(s) URL",
                    model.id
                ));
            }
            PackageEngineKind::Remote
        }
        other => {
            return Err(format!(
                "engine.kind for model {} must be \"local\" or \"remote\", got {other:?}",
                model.id
            ));
        }
    };
    Ok(ValidatedPackageEngine {
        kind,
        location: engine.location.clone(),
        protocol: engine.protocol.clone(),
        args: engine.args.clone(),
    })
}

fn validate_engine_path_prefix(path: &Path) -> std::result::Result<(), String> {
    let actual = path
        .components()
        .find_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy().into_owned()),
            _ => None,
        })
        .ok_or_else(|| "models.engine.location is empty".to_string())?;
    if actual != "bin" {
        return Err("local engine.location must be under bin/".to_string());
    }
    Ok(())
}

fn validate_model_path_prefix(path: &Path, rule_line: RuleLine) -> std::result::Result<(), String> {
    let expected = match rule_line {
        RuleLine::Riichi4p => "4p",
        RuleLine::Riichi3p => "3p",
    };
    let components = path
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let actual = components
        .first()
        .ok_or_else(|| "models.path is empty".to_string())?;
    // `weights/<file>` is the shared directory for released weights. Only the layout is
    // checked here, not the content; integrity is ensured by the release process
    // (sha256 and size per file against the lock file).
    //
    // Do not add hash checks here: the expected hashes could only come from a lock file
    // in the same mount as the weights, so anyone able to replace the weights could
    // replace it too. That stops no tampering and would make the host parse release
    // manifests. Real tamper protection means content addressing (file name = hash).
    if actual == "weights" && components.len() == 2 {
        return Ok(());
    }
    if actual != expected {
        return Err(format!(
            "rule_line {} requires model path under {}/ or weights/<file>",
            rule_line.wire_value(),
            expected
        ));
    }
    Ok(())
}

fn expand_package_arg(
    arg: &str,
    kind: &PackageModelKind,
    rule_line: RuleLine,
) -> std::result::Result<String, String> {
    let model_path = match kind {
        PackageModelKind::Onnx { path } => Some(path.as_str()),
        PackageModelKind::Remote { .. }
        | PackageModelKind::RuntimeRemote { .. }
        | PackageModelKind::RuntimeLocal { .. } => None,
    };
    if arg.contains("{model_path}") && model_path.is_none() {
        return Err("{model_path} cannot be used by remote models".to_string());
    }
    Ok(arg
        .replace("{model_path}", model_path.unwrap_or(""))
        .replace("{rule_line}", rule_line.wire_value()))
}

fn expand_runtime_package_arg(
    arg: &str,
    package_dir: &Path,
    model: &PackageModelManifest,
    kind: &PackageModelKind,
    rule_line: RuleLine,
) -> std::result::Result<String, String> {
    let (engine_kind, engine_location, engine_protocol, weight_file, remote_model) = match kind {
        PackageModelKind::RuntimeLocal {
            weight_file,
            engine,
        } => (
            "local",
            engine.location.as_str(),
            engine.protocol.as_str(),
            weight_file.as_str(),
            "",
        ),
        PackageModelKind::RuntimeRemote {
            remote_model,
            engine,
        } => (
            "remote",
            engine.location.as_str(),
            engine.protocol.as_str(),
            "",
            remote_model.as_str(),
        ),
        PackageModelKind::Onnx { .. } | PackageModelKind::Remote { .. } => {
            return Err("runtime placeholder expansion requires a schema 3 model".to_string());
        }
    };
    if arg.contains("{model_path}") && weight_file.is_empty() {
        return Err("{model_path} cannot be used by remote models".to_string());
    }
    Ok(arg
        .replace("{manifest_path}", PACKAGE_MANIFEST_FILE_NAME)
        .replace("{package_dir}", &package_dir.display().to_string())
        .replace("{model_id}", &model.id)
        .replace("{model_display_name}", &model.display_name)
        .replace("{rule_line}", rule_line.wire_value())
        .replace("{engine_kind}", engine_kind)
        .replace("{engine_location}", engine_location)
        .replace("{engine_protocol}", engine_protocol)
        .replace("{weight_file}", weight_file)
        .replace("{model_path}", weight_file)
        .replace("{remote_model}", remote_model))
}

fn validate_relative_manifest_path(
    label: &str,
    value: &str,
) -> std::result::Result<PathBuf, String> {
    if value.trim().is_empty() {
        return Err(format!("{label} is required"));
    }
    let path = PathBuf::from(value);
    if path.is_absolute() {
        return Err(format!("{label} must be package-relative"));
    }
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(value) => out.push(value),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!("{label} must not escape the package directory"));
            }
        }
    }
    if out.as_os_str().is_empty() {
        return Err(format!("{label} is required"));
    }
    Ok(out)
}

pub fn package_hash(dir: impl AsRef<Path>) -> Result<String> {
    let dir = dir.as_ref();
    let mut files = Vec::new();
    collect_package_files(dir, dir, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut hasher = Sha256::new();
    for (relative, path) in files {
        hasher.update(relative.as_bytes());
        hasher.update([0]);
        let mut file =
            fs::File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
        let mut buf = [0u8; 8192];
        loop {
            let read = file
                .read(&mut buf)
                .with_context(|| format!("failed to read {}", path.display()))?;
            if read == 0 {
                break;
            }
            hasher.update(&buf[..read]);
        }
        hasher.update([0xff]);
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn package_model_hash(
    package_dir: &Path,
    manifest: &PackageManifest,
    model: &PackageModelManifest,
    validated: &ValidatedPackageModel,
) -> Result<String> {
    #[derive(Serialize)]
    struct PackageHashInput<'a> {
        schema: u32,
        package_id: &'a str,
        protocol: &'a str,
        riichi_style: &'a str,
        translator: &'a PackageTranslatorManifest,
        caps: &'a Option<ManifestCaps>,
        model: &'a PackageModelManifest,
    }

    let mut hasher = Sha256::new();
    hasher.update(b"flya-package-v2-model-runtime");
    hasher.update([0]);
    let canonical_manifest = serde_json::to_vec(&PackageHashInput {
        schema: manifest.schema,
        package_id: &manifest.package_id,
        protocol: &manifest.protocol,
        riichi_style: &manifest.riichi_style,
        translator: &manifest.translator,
        caps: &manifest.caps,
        model,
    })?;
    hasher.update(b"manifest");
    hasher.update([0]);
    hasher.update(canonical_manifest);
    hasher.update([0xff]);

    let mut shared_files = Vec::new();
    collect_package_runtime_files(package_dir, package_dir, &mut shared_files)?;
    shared_files.sort_by(|left, right| left.0.cmp(&right.0));
    for (relative, path) in shared_files {
        update_hash_with_file(&mut hasher, "shared", &relative, &path)?;
    }

    match &validated.kind {
        PackageModelKind::Onnx { path } => {
            let model_path = package_dir.join(path);
            update_hash_with_file(&mut hasher, "model", path, &model_path)?;
        }
        PackageModelKind::Remote { backend } => {
            hasher.update(b"remote");
            hasher.update([0]);
            hasher.update(backend.as_bytes());
            hasher.update([0xff]);
        }
        PackageModelKind::RuntimeLocal {
            weight_file,
            engine,
        } => {
            update_runtime_engine_hash(&mut hasher, package_dir, engine)?;
            let weight_path = package_dir.join(weight_file);
            update_hash_with_file(&mut hasher, "weight", weight_file, &weight_path)?;
        }
        PackageModelKind::RuntimeRemote {
            remote_model,
            engine,
        } => {
            update_runtime_engine_hash(&mut hasher, package_dir, engine)?;
            hasher.update(b"remote_model");
            hasher.update([0]);
            hasher.update(remote_model.as_bytes());
            hasher.update([0xff]);
        }
    }

    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn update_runtime_engine_hash(
    hasher: &mut Sha256,
    package_dir: &Path,
    engine: &ValidatedPackageEngine,
) -> Result<()> {
    match engine.kind {
        PackageEngineKind::Local => {
            let engine_path = package_dir.join(&engine.location);
            update_hash_with_file(hasher, "engine", &engine.location, &engine_path)?;
        }
        PackageEngineKind::Remote => {
            hasher.update(b"remote_engine");
            hasher.update([0]);
            hasher.update(engine.location.as_bytes());
            hasher.update([0xff]);
        }
    }
    hasher.update(b"engine_protocol");
    hasher.update([0]);
    hasher.update(engine.protocol.as_bytes());
    hasher.update([0xff]);
    let canonical_args = serde_json::to_vec(&engine.args)?;
    hasher.update(b"engine_args");
    hasher.update([0]);
    hasher.update(canonical_args);
    hasher.update([0xff]);
    Ok(())
}

fn update_hash_with_file(
    hasher: &mut Sha256,
    scope: &str,
    relative: &str,
    path: &Path,
) -> Result<()> {
    hasher.update(scope.as_bytes());
    hasher.update([0]);
    hasher.update(relative.as_bytes());
    hasher.update([0]);
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut buf = [0u8; 8192];
    loop {
        let read = file
            .read(&mut buf)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    hasher.update([0xff]);
    Ok(())
}

fn collect_package_files(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) -> Result<()> {
    let mut entries = fs::read_dir(dir)
        .with_context(|| format!("failed to read package directory {}", dir.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("failed to enumerate {}", dir.display()))?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        let metadata = entry
            .metadata()
            .with_context(|| format!("failed to stat {}", path.display()))?;
        if metadata.is_dir() {
            // Skip transient directories (translator run/build output), otherwise package_hash and model_id drift between scans.
            if is_transient_dir(&path) {
                continue;
            }
            collect_package_files(root, &path, out)?;
        } else if metadata.is_file() {
            // Skip transient files (bytecode, logs, OS clutter).
            if is_transient_file(&path) {
                continue;
            }
            let relative = relative_slash_path(root, &path)?;
            if relative == CERT_FILE_NAME {
                continue;
            }
            out.push((relative, path));
        }
    }
    Ok(())
}

fn collect_package_runtime_files(
    root: &Path,
    dir: &Path,
    out: &mut Vec<(String, PathBuf)>,
) -> Result<()> {
    let mut entries = fs::read_dir(dir)
        .with_context(|| format!("failed to read package directory {}", dir.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("failed to enumerate {}", dir.display()))?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        let metadata = entry
            .metadata()
            .with_context(|| format!("failed to stat {}", path.display()))?;
        let relative = relative_slash_path(root, &path)?;
        if metadata.is_dir() {
            if is_transient_dir(&path)
                || matches!(relative.as_str(), "4p" | "3p" | "models.d" | "weights")
            {
                continue;
            }
            collect_package_runtime_files(root, &path, out)?;
        } else if metadata.is_file() {
            if is_transient_file(&path)
                || relative == CERT_FILE_NAME
                || relative == PACKAGE_MANIFEST_FILE_NAME
                || relative.starts_with(&format!("{PACKAGE_CERT_DIR_NAME}/"))
            {
                continue;
            }
            out.push((relative, path));
        }
    }
    Ok(())
}

/// Transient directories: output that translators (such as native inference
/// engines) write while running or building, which does not affect model or
/// translator behavior.
///
/// Excluding them keeps `package_hash` (and so `model_id`) stable across scans (plugin
/// runtimes may write `__pycache__/*.pyc` or logs into their own directory).
/// Directories holding model or translator logic (source, config, weights) are never
/// excluded.
fn is_transient_dir(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(
            "__pycache__"
                | ".local"
                | ".git"
                | ".flya-certs"
                | ".mypy_cache"
                | ".pytest_cache"
                | ".ruff_cache"
                | ".ipynb_checkpoints"
        )
    )
}

/// Transient files: Python bytecode (`*.pyc` / `*.pyo`), logs (`*.log`) and OS clutter.
///
/// Deliberately not excluded: `*.pyd` (compiled extensions affect behavior), weights
/// (`*.pth` / `*.onnx`), manifests, configs and translator source. Changing them
/// should require recertification or change the model_id. Known limitation: logs not
/// named `*.log` (such as a translator's `debug.txt`) are not covered.
fn is_transient_file(path: &Path) -> bool {
    let name = match path.file_name().and_then(|name| name.to_str()) {
        Some(name) => name,
        None => return false,
    };
    if matches!(name, ".DS_Store" | "Thumbs.db") {
        return true;
    }
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("pyc" | "pyo" | "log")
    )
}

fn relative_slash_path(root: &Path, path: &Path) -> Result<String> {
    let rel = path
        .strip_prefix(root)
        .with_context(|| format!("{} is not under {}", path.display(), root.display()))?;
    let parts = rel
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>();
    Ok(parts.join("/"))
}

fn relative_slash_path_from_path(path: &Path) -> String {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn package_cert_path(package_dir: &Path, rule_line: &RuleLine, slug: &str) -> PathBuf {
    package_dir.join(PACKAGE_CERT_DIR_NAME).join(format!(
        "{}.{}.json",
        rule_line.wire_value(),
        sanitize_model_id_component(slug)
    ))
}

fn certify_plugin(
    plugin_dir: &Path,
    manifest: &PluginManifest,
    validated: &ValidatedManifest,
    package_hash: &str,
) -> Result<SmokeSummary> {
    certify_plugin_runtime(
        plugin_dir,
        resolve_launch_cmd(plugin_dir, &manifest.launch.cmd),
        manifest.launch.args.clone(),
        validated,
        package_hash,
        &plugin_dir.join(CERT_FILE_NAME),
    )
}

fn certify_plugin_runtime(
    plugin_dir: &Path,
    launch_cmd: PathBuf,
    launch_args: Vec<String>,
    validated: &ValidatedManifest,
    package_hash: &str,
    cert_path: &Path,
) -> Result<SmokeSummary> {
    // Certification is UX and cache validation only. It starts third-party code
    // and samples known decisions; the runtime host still rejects every action
    // that does not exactly match the FlyTable legal list.
    let mut config = EngineProcessConfig::new(
        launch_cmd,
        0,
        validated.rule_line,
        format!("cert-{}-{}", validated.name, std::process::id()),
    );
    config.args = launch_args;
    config.cwd = Some(plugin_dir.to_path_buf());
    config.timeout = Duration::from_secs(15);
    config.match_context = serde_json::json!({"length": "east", "certification": true});
    config.protocol_versions = vec![validated.protocol.clone().into()];
    let mut host = SubprocessHost::start(config).map_err(inference_err)?;
    let caps = host
        .caps()
        .cloned()
        .ok_or_else(|| anyhow!("engine hello returned no caps"))?;
    validate_caps(&caps, validated)?;
    let allow_dealer_opening_abstain = validated.protocol == FLYA_INFERENCE_PROTOCOL_V1;
    let smoke = match validated.rule_line {
        RuleLine::Riichi4p => smoke_4p(&mut host, SMOKE_DECISIONS, allow_dealer_opening_abstain)?,
        RuleLine::Riichi3p => smoke_3p(&mut host, SMOKE_DECISIONS, allow_dealer_opening_abstain)?,
    };
    let negotiated_protocol = host
        .negotiated_protocol()
        .ok_or_else(|| anyhow!("engine hello did not negotiate a protocol"))?;
    let cert = PluginCertificate {
        schema: 2,
        package_hash: package_hash.to_string(),
        protocol: negotiated_protocol.to_string(),
        rule_line: validated.rule_line.wire_value().to_string(),
        caps: CertificateCaps {
            riichi_style: caps.riichi_style.wire_value().to_string(),
            supports_incremental: caps.supports_incremental,
            returns_ranked_actions: caps.returns_ranked_actions,
        },
        smoke: smoke.clone(),
        validated_at: OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string()),
    };
    let raw = serde_json::to_string_pretty(&cert)?;
    if let Some(parent) = cert_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(cert_path, raw)
        .with_context(|| format!("failed to write {}", cert_path.display()))?;
    Ok(smoke)
}

fn resolve_launch_cmd(plugin_dir: &Path, cmd: &str) -> PathBuf {
    let raw = PathBuf::from(cmd);
    if raw.is_absolute() || !cmd.contains(['/', '\\']) {
        raw
    } else {
        plugin_dir.join(raw)
    }
}

fn validate_caps(caps: &EngineCaps, manifest: &ValidatedManifest) -> Result<()> {
    if !caps
        .protocol_versions
        .iter()
        .any(|version| version.as_ref() == manifest.protocol)
    {
        return Err(anyhow!(
            "caps protocol_versions does not include {}",
            manifest.protocol
        ));
    }
    if !caps.rule_lines.contains(&manifest.rule_line) {
        return Err(anyhow!(
            "caps.rule_lines does not include {}",
            manifest.rule_line.wire_value()
        ));
    }
    if caps.riichi_style != manifest.riichi_style {
        return Err(anyhow!(
            "manifest riichi_style {} disagrees with engine caps {}",
            manifest.riichi_style.wire_value(),
            caps.riichi_style.wire_value()
        ));
    }
    Ok(())
}

fn smoke_dealer_opening_4p(host: &mut SubprocessHost, allow_abstain: bool) -> Result<u64> {
    let hand = [
        "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
    ]
    .map(|tile| tile.parse::<Tile>().expect("certification tile literal"));
    let events = vec![
        Event4p::StartGame {
            names: ["a", "b", "c", "d"].map(str::to_owned),
            seed: None,
        },
        Event4p::StartKyoku {
            bakaze: "E".parse().expect("tile literal"),
            dora_marker: "N".parse().expect("tile literal"),
            kyoku: 1,
            honba: 0,
            kyotaku: 0,
            oya: 0,
            scores: [25000; 4],
            tehais: [
                hand,
                [Tile::unknown(); 13],
                [Tile::unknown(); 13],
                [Tile::unknown(); 13],
            ],
        },
        Event4p::DealerOpening {
            actor: 0,
            pai: "9s".parse().expect("tile literal"),
        },
    ];
    let mut mirror = Mirror4p::new(0);
    for event in &events {
        mirror
            .feed(event.clone())
            .map_err(|err| anyhow!("4p dealer-opening smoke mirror failed: {err}"))?;
    }
    let view = mirror.mirror_view(0);
    let actions = mirror.legal_turn(0);
    if !actions
        .iter()
        .any(|action| matches!(action, TurnAction::DealerOpeningDiscard { .. }))
    {
        return Err(anyhow!(
            "4p dealer-opening smoke produced no neutral discard"
        ));
    }
    let legal_actions = actions
        .iter()
        .map(|action| turn_action_to_legal_4p(action, &view))
        .collect::<Vec<_>>();
    let started = Instant::now();
    let decision = host
        .infer_4p(InferenceInput4p {
            decision_id: "cert-4p-dealer-opening".to_string(),
            phase: DecisionPhase::Discard,
            events: &events,
            legal_actions: &legal_actions,
            wall_remaining: view.tiles_left,
            source: SourceInfo::authoritative(),
            remote_auth: None,
            seat: None,
        })
        .map_err(inference_err)?;
    select_opening_index(decision, legal_actions.len(), allow_abstain)?;
    Ok(started.elapsed().as_millis() as u64)
}

fn smoke_dealer_opening_3p(host: &mut SubprocessHost, allow_abstain: bool) -> Result<u64> {
    let hand = [
        "1m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "E", "S",
    ]
    .map(|tile| tile.parse::<Tile>().expect("certification tile literal"));
    let events = vec![
        Event3p::StartGame {
            names: ["a", "b", "c"].map(str::to_owned),
            seed: None,
        },
        Event3p::StartKyoku {
            bakaze: "E".parse().expect("tile literal"),
            dora_marker: "W".parse().expect("tile literal"),
            kyoku: 1,
            honba: 0,
            kyotaku: 0,
            oya: 0,
            scores: [35000; 3],
            tehais: [hand, [Tile::unknown(); 13], [Tile::unknown(); 13]],
        },
        Event3p::DealerOpening {
            actor: 0,
            pai: "P".parse().expect("tile literal"),
        },
    ];
    let mut mirror = Mirror3p::new(0);
    for event in &events {
        mirror
            .feed(event.clone())
            .map_err(|err| anyhow!("3p dealer-opening smoke mirror failed: {err}"))?;
    }
    let view = mirror.mirror_view(0);
    let actions = mirror.legal_turn(0);
    if !actions
        .iter()
        .any(|action| matches!(action, TurnAction::DealerOpeningDiscard { .. }))
    {
        return Err(anyhow!(
            "3p dealer-opening smoke produced no neutral discard"
        ));
    }
    let legal_actions = actions
        .iter()
        .map(|action| turn_action_to_legal_3p(action, &view))
        .collect::<Vec<_>>();
    let started = Instant::now();
    let decision = host
        .infer_3p(InferenceInput3p {
            decision_id: "cert-3p-dealer-opening".to_string(),
            phase: DecisionPhase::Discard,
            events: &events,
            legal_actions: &legal_actions,
            wall_remaining: view.tiles_left,
            source: SourceInfo::authoritative(),
            remote_auth: None,
            seat: None,
            match_context: None,
        })
        .map_err(inference_err)?;
    select_opening_index(decision, legal_actions.len(), allow_abstain)?;
    Ok(started.elapsed().as_millis() as u64)
}

fn smoke_4p(
    host: &mut SubprocessHost,
    target: usize,
    allow_dealer_opening_abstain: bool,
) -> Result<SmokeSummary> {
    let mut board = Board4p::start((2026061505, 0xA11C_E001), [25000; 4]);
    let mut needs_draw = true;
    let mut steps = 0usize;
    let mut decisions = 0usize;
    let mut max_infer_ms = smoke_dealer_opening_4p(host, allow_dealer_opening_abstain)?;
    while decisions < target {
        if needs_draw && board.draw_for_turn().is_none() {
            board = Board4p::start((2026061505 + decisions as u64, 0xA11C_E001), [25000; 4]);
            needs_draw = true;
            continue;
        }
        let seat = board.turn;
        let actions = legal_turn_actions(&board.view_for(seat));
        if actions.is_empty() {
            return Err(anyhow!("4p smoke produced no legal turn actions"));
        }
        let index = if seat == 0 {
            let legal_actions = actions
                .iter()
                .map(|action| turn_action_to_legal_4p(action, &board.view_for(seat)))
                .collect::<Vec<_>>();
            let events = events_with_start_game_4p(&board);
            let started = Instant::now();
            let decision = host
                .infer_4p(InferenceInput4p {
                    decision_id: format!("cert-4p-{steps}"),
                    phase: DecisionPhase::Discard,
                    events: &events,
                    legal_actions: &legal_actions,
                    wall_remaining: board.wall.live_remaining() as u32,
                    source: SourceInfo::authoritative(),
                    remote_auth: None,
                    seat: None,
                })
                .map_err(inference_err)?;
            max_infer_ms = cmp::max(max_infer_ms, started.elapsed().as_millis() as u64);
            decisions += 1;
            select_index(decision, legal_actions.len())?
        } else {
            fallback_turn_index(&actions)
        };
        let before_discard = board.last_discard;
        if board
            .apply_turn(actions[index].clone())
            .map_err(|err| anyhow!("4p smoke apply_turn failed: {err}"))?
            .is_some()
        {
            board = Board4p::start((2026061505 + decisions as u64, 0xA11C_E002), [25000; 4]);
            needs_draw = true;
            continue;
        }
        if board.last_discard.is_some() && board.last_discard != before_discard {
            board.advance_turn();
            needs_draw = true;
        } else {
            needs_draw = false;
        }
        steps += 1;
        if steps > 5000 {
            return Err(anyhow!("4p smoke exceeded step budget"));
        }
    }
    Ok(SmokeSummary {
        decisions,
        all_legal: true,
        dealer_opening_checked: true,
        max_infer_ms,
    })
}

fn smoke_3p(
    host: &mut SubprocessHost,
    target: usize,
    allow_dealer_opening_abstain: bool,
) -> Result<SmokeSummary> {
    let mut board = Board3p::start((2026061506, 0xA11C_E301), [35000; 3]);
    let mut needs_draw = true;
    let mut steps = 0usize;
    let mut decisions = 0usize;
    let mut max_infer_ms = smoke_dealer_opening_3p(host, allow_dealer_opening_abstain)?;
    while decisions < target {
        if needs_draw && board.draw_for_turn().is_none() {
            board = Board3p::start((2026061506 + decisions as u64, 0xA11C_E301), [35000; 3]);
            needs_draw = true;
            continue;
        }
        let seat = board.turn;
        let actions = legal_turn_actions(&board.view_for(seat));
        if actions.is_empty() {
            return Err(anyhow!("3p smoke produced no legal turn actions"));
        }
        let index = if seat == 0 {
            let legal_actions = actions
                .iter()
                .map(|action| turn_action_to_legal_3p(action, &board.view_for(seat)))
                .collect::<Vec<_>>();
            let events = events_with_start_game_3p(&board);
            let started = Instant::now();
            let decision = host
                .infer_3p(InferenceInput3p {
                    decision_id: format!("cert-3p-{steps}"),
                    phase: DecisionPhase::Discard,
                    events: &events,
                    legal_actions: &legal_actions,
                    wall_remaining: board.wall.live_remaining() as u32,
                    source: SourceInfo::authoritative(),
                    remote_auth: None,
                    seat: None,
                    match_context: None,
                })
                .map_err(inference_err)?;
            max_infer_ms = cmp::max(max_infer_ms, started.elapsed().as_millis() as u64);
            decisions += 1;
            select_index(decision, legal_actions.len())?
        } else {
            fallback_turn_index(&actions)
        };
        let before_discard = board.last_discard;
        if board
            .apply_turn(actions[index].clone())
            .map_err(|err| anyhow!("3p smoke apply_turn failed: {err}"))?
            .is_some()
        {
            board = Board3p::start((2026061506 + decisions as u64, 0xA11C_E302), [35000; 3]);
            needs_draw = true;
            continue;
        }
        if board.last_discard.is_some() && board.last_discard != before_discard {
            board.advance_turn();
            needs_draw = true;
        } else {
            needs_draw = false;
        }
        steps += 1;
        if steps > 5000 {
            return Err(anyhow!("3p smoke exceeded step budget"));
        }
    }
    Ok(SmokeSummary {
        decisions,
        all_legal: true,
        dealer_opening_checked: true,
        max_infer_ms,
    })
}

fn events_with_start_game_4p(board: &Board4p) -> Vec<Event4p> {
    let mut events = vec![Event4p::StartGame {
        names: ["a", "b", "c", "d"].map(str::to_owned),
        seed: None,
    }];
    events.extend(board.log.iter().cloned());
    events
}

fn events_with_start_game_3p(board: &Board3p) -> Vec<Event3p> {
    let mut events = vec![Event3p::StartGame {
        names: ["a", "b", "c"].map(str::to_owned),
        seed: None,
    }];
    events.extend(board.log.iter().cloned());
    events
}

fn turn_action_to_legal_4p(action: &TurnAction, view: &SeatView) -> LegalAction4p {
    match action {
        TurnAction::DealerOpeningDiscard { tile } => LegalAction4p::DealerOpeningDiscard {
            pai: *tile,
            riichi: false,
        },
        TurnAction::DealerOpeningRiichi { tile } => LegalAction4p::DealerOpeningDiscard {
            pai: *tile,
            riichi: true,
        },
        TurnAction::Discard { tile, tsumogiri } => LegalAction4p::Discard {
            pai: *tile,
            tsumogiri: *tsumogiri,
            riichi: false,
        },
        TurnAction::Riichi { tile, tsumogiri } => LegalAction4p::Discard {
            pai: *tile,
            tsumogiri: *tsumogiri,
            riichi: true,
        },
        TurnAction::Ankan { tile } => LegalAction4p::Kan {
            pai: *tile,
            kind: KanKind::Ankan,
            consumed: view
                .me
                .hand
                .iter()
                .copied()
                .filter(|t| t.kind() == tile.kind())
                .take(4)
                .collect(),
        },
        TurnAction::Kakan { tile } => LegalAction4p::Kan {
            pai: *tile,
            kind: KanKind::Kakan,
            consumed: kakan_consumed(&view.me.melds, *tile),
        },
        TurnAction::Tsumo if view.me.dealer_opening => LegalAction4p::DealerOpeningTsumo,
        TurnAction::Tsumo => LegalAction4p::Tsumo {
            pai: *view.me.hand.last().expect("tsumo requires a drawn tile"),
        },
        TurnAction::KyuushuKyuuhai => LegalAction4p::Kyushukyuhai,
        TurnAction::Nukidora => panic!("4p must not expose nukidora"),
    }
}

fn turn_action_to_legal_3p(action: &TurnAction, view: &SeatView) -> LegalAction3p {
    match action {
        TurnAction::DealerOpeningDiscard { tile } => LegalAction3p::DealerOpeningDiscard {
            pai: *tile,
            riichi: false,
        },
        TurnAction::DealerOpeningRiichi { tile } => LegalAction3p::DealerOpeningDiscard {
            pai: *tile,
            riichi: true,
        },
        TurnAction::Discard { tile, tsumogiri } => LegalAction3p::Discard {
            pai: *tile,
            tsumogiri: *tsumogiri,
            riichi: false,
        },
        TurnAction::Riichi { tile, tsumogiri } => LegalAction3p::Discard {
            pai: *tile,
            tsumogiri: *tsumogiri,
            riichi: true,
        },
        TurnAction::Ankan { tile } => LegalAction3p::Kan {
            pai: *tile,
            kind: KanKind::Ankan,
            consumed: view
                .me
                .hand
                .iter()
                .copied()
                .filter(|t| t.kind() == tile.kind())
                .take(4)
                .collect(),
        },
        TurnAction::Kakan { tile } => LegalAction3p::Kan {
            pai: *tile,
            kind: KanKind::Kakan,
            consumed: kakan_consumed(&view.me.melds, *tile),
        },
        TurnAction::Nukidora => LegalAction3p::Nukidora,
        TurnAction::Tsumo if view.me.dealer_opening => LegalAction3p::DealerOpeningTsumo,
        TurnAction::Tsumo => LegalAction3p::Tsumo {
            pai: *view.me.hand.last().expect("tsumo requires a drawn tile"),
        },
        TurnAction::KyuushuKyuuhai => LegalAction3p::Kyushukyuhai,
    }
}

fn kakan_consumed(melds: &[Meld], tile: Tile) -> Vec<Tile> {
    melds
        .iter()
        .find_map(|meld| match meld {
            Meld::Pon {
                tile: base,
                called,
                consumed,
                ..
            } if base.kind() == tile.kind() => Some(vec![consumed[0], consumed[1], *called]),
            _ => None,
        })
        .unwrap_or_default()
}

fn fallback_turn_index(actions: &[TurnAction]) -> usize {
    actions
        .iter()
        .position(TurnAction::is_plain_discard)
        .unwrap_or(0)
}

fn select_index(decision: InferenceDecision, legal_len: usize) -> Result<usize> {
    match decision {
        InferenceDecision::Select { index, .. } if index < legal_len => Ok(index),
        InferenceDecision::Select { index, .. } => {
            Err(anyhow!("engine selected out-of-range legal index {index}"))
        }
        InferenceDecision::Abstain => Err(anyhow!("engine abstained during certification smoke")),
    }
}

fn select_opening_index(
    decision: InferenceDecision,
    legal_len: usize,
    allow_abstain: bool,
) -> Result<()> {
    if allow_abstain && matches!(decision, InferenceDecision::Abstain) {
        return Ok(());
    }
    select_index(decision, legal_len).map(|_| ())
}

fn read_certificate(path: &Path) -> Result<PluginCertificate> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("invalid certificate {}", path.display()))
}

fn parse_rule_line(value: &str) -> std::result::Result<RuleLine, String> {
    match value {
        "riichi4p" => Ok(RuleLine::Riichi4p),
        "riichi3p" => Ok(RuleLine::Riichi3p),
        other => Err(format!("unknown rule_line {other:?}")),
    }
}

fn parse_riichi_style(value: &str) -> std::result::Result<RiichiStyle, String> {
    match value {
        "parallel_discard" => Ok(RiichiStyle::ParallelDiscard),
        "declare_then_discard" => Ok(RiichiStyle::DeclareThenDiscard),
        other => Err(format!("unknown riichi_style {other:?}")),
    }
}

fn inference_err(err: flytable_seat::contract::InferenceError) -> anyhow::Error {
    anyhow!("{:?}: {}", err.kind, err.detail)
}

impl PluginScanEntry {
    fn failed(plugin_dir: PathBuf, reason: String) -> Self {
        Self::failed_with_layout(plugin_dir, "legacy", None, None, reason)
    }

    fn failed_with_layout(
        plugin_dir: PathBuf,
        layout: &str,
        package_id: Option<String>,
        model_slug: Option<String>,
        reason: String,
    ) -> Self {
        let fallback = plugin_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("<unknown>")
            .to_string();
        let model_slug = model_slug.unwrap_or_else(|| fallback.clone());
        Self {
            name: model_slug.clone(),
            plugin_dir,
            layout: layout.to_string(),
            package_id,
            model_slug,
            rule_line: None,
            riichi_style: None,
            package_hash: None,
            status: PluginScanStatus::Failed { reason },
        }
    }

    fn skipped(plugin_dir: PathBuf, reason: String) -> Self {
        Self {
            name: plugin_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<unknown>")
                .to_string(),
            model_slug: plugin_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<unknown>")
                .to_string(),
            plugin_dir,
            layout: "legacy".to_string(),
            package_id: None,
            rule_line: None,
            riichi_style: None,
            package_hash: None,
            status: PluginScanStatus::Skipped { reason },
        }
    }
}
