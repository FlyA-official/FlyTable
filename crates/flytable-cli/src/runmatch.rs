//! `run-match` subcommand: a minimal seat bus.
//!
//! The host (runtime `MatchHost`) asks each seat for an action; algorithm and
//! certified plugin seats answer through the common [`SeatAgent`] abstraction, and
//! FlyTable adjudicates until the match ends. Prints turn/seat, legal_count,
//! selected, agent kind/model_id and latency / fallback for every move.
//!
//! [`SeatAgent`]: flytable_runtime::SeatAgent

use std::path::PathBuf;

use flytable_core::rules::RiichiRuleProfile;
use flytable_runtime::{
    AlgorithmKind, CertifiedPluginRuntime, FullMatchReport, MatchConfig, MatchKind, RuntimeStatus,
    SeatSpec, discover_plugins, run_full_match_3p, run_full_match_4p,
};

/// Runs the match and prints the result. `Err(msg)` means an argument or startup error.
#[allow(clippy::too_many_arguments)]
pub fn run(
    players: u8,
    seed: u64,
    seat_specs: Vec<String>,
    models_root: Option<PathBuf>,
    jsonl: bool,
    quiet: bool,
    length: &str,
    platform: &str,
    status: bool,
) -> Result<(), String> {
    let n = match players {
        3 | 4 => players as usize,
        _ => return Err("only 3 or 4 seats are supported".to_string()),
    };
    let kind = MatchKind::parse(length)
        .ok_or_else(|| format!("unknown --length {length:?} (expected single / east / half)"))?;
    let rule_profile = platform.parse::<RiichiRuleProfile>()?;

    let mut specs: Vec<Option<SeatSpec>> = (0..n).map(|_| None).collect();
    let needs_plugins = seat_specs.iter().any(|entry| {
        entry.split_once('=').is_some_and(|(_, spec)| {
            spec.trim().starts_with("plugin:") || spec.trim().starts_with("plugin@")
        })
    });
    let plugin_cache = if needs_plugins {
        let root = models_root.as_deref().ok_or_else(plugin_needs_root)?;
        Some(discover_plugins(root))
    } else {
        None
    };
    for entry in &seat_specs {
        let (idx, spec) = entry
            .split_once('=')
            .ok_or_else(|| format!("--seat must be I=SPEC, got {entry:?}"))?;
        let i: usize = idx
            .trim()
            .parse()
            .map_err(|_| format!("invalid --seat index: {idx:?}"))?;
        if i >= n {
            return Err(format!("seat index {i} is outside 0..{n}"));
        }
        if specs[i].is_some() {
            return Err(format!("seat {i} specified more than once"));
        }
        specs[i] = Some(parse_spec(
            players,
            spec.trim(),
            models_root.as_deref(),
            plugin_cache.as_deref(),
        )?);
    }
    // Unspecified seats default to tsumogiri.
    let specs: Vec<SeatSpec> = specs
        .into_iter()
        .map(|o| o.unwrap_or(SeatSpec::Algorithm(AlgorithmKind::Tsumogiri)))
        .collect();

    let config = MatchConfig::new(seed, kind).with_rule_profile(rule_profile);
    let report = if n == 4 {
        run_full_match_4p(config, specs)
    } else {
        run_full_match_3p(config, specs)
    }
    .map_err(|e| format!("failed to start match: {e:#}"))?;

    print_report(&report, jsonl, quiet);
    if status {
        let runtime_status = RuntimeStatus::from_full_report(&report);
        println!("{}", serde_json::to_string(&runtime_status).unwrap());
    }
    Ok(())
}

fn parse_spec(
    players: u8,
    spec: &str,
    models_root: Option<&std::path::Path>,
    plugins: Option<&[CertifiedPluginRuntime]>,
) -> Result<SeatSpec, String> {
    let want = if players == 4 { "riichi4p" } else { "riichi3p" };
    // `plugin:<model_id>`: select by the exact model_id normalized by the registry (the
    // directory name is not assumed to be the id).
    if let Some(model_id) = spec.strip_prefix("plugin:") {
        let root = models_root.ok_or_else(plugin_needs_root)?;
        let plugins = plugins.ok_or_else(plugin_needs_root)?;
        let rt = plugins
            .iter()
            .find(|rt| rt.model_id == model_id)
            .cloned()
            .ok_or_else(|| not_found_msg(root, model_id, plugins))?;
        check_rule(&rt, want, players)?;
        Ok(SeatSpec::Plugin(rt))
    } else if let Some(name) = spec.strip_prefix("plugin@") {
        // `plugin@<manifest_name>`: select a certified runtime by the `name` in plugin.toml,
        // within the single authoritative scan of this run. Useful for plugins whose
        // package_hash changes between scans (for example ones that write .pyc files or
        // logs into their directory), since the id cannot drift between a discovery scan
        // and a use scan.
        let root = models_root.ok_or_else(plugin_needs_root)?;
        let plugins = plugins.ok_or_else(plugin_needs_root)?;
        let mut matches: Vec<_> = plugins
            .iter()
            .filter(|rt| rt.name == name && rt.rule_line == want)
            .cloned()
            .collect();
        match matches.len() {
            0 => Err(not_found_by_name(root, name, want, plugins)),
            1 => Ok(SeatSpec::Plugin(matches.remove(0))),
            n => Err(format!(
                "name={name:?} matches {n} plugins under {want}; use plugin:<model_id> instead"
            )),
        }
    } else if let Some(kind) = AlgorithmKind::parse(spec) {
        Ok(SeatSpec::Algorithm(kind))
    } else {
        Err(format!(
            "unknown seat spec {spec:?} (expected tsumogiri / plugin:<model_id> / plugin@<name>)"
        ))
    }
}

fn plugin_needs_root() -> String {
    "plugin seats need --models-root <root> (the registry scans <root>/{4p,3p}/<plugin>)"
        .to_string()
}

fn check_rule(
    rt: &flytable_runtime::CertifiedPluginRuntime,
    want: &str,
    players: u8,
) -> Result<(), String> {
    if rt.rule_line != want {
        return Err(format!(
            "plugin {} is for rule line {}, which does not match --players {players} ({want})",
            rt.model_id, rt.rule_line
        ));
    }
    Ok(())
}

fn not_found_msg(
    root: &std::path::Path,
    model_id: &str,
    plugins: &[CertifiedPluginRuntime],
) -> String {
    let available: Vec<String> = plugins
        .iter()
        .map(|rt| format!("{} (name={}) [{}]", rt.model_id, rt.name, rt.rule_line))
        .collect();
    if available.is_empty() {
        format!(
            "no certified plugins under models-root {} (model_id={model_id:?} not found)",
            root.display()
        )
    } else {
        format!(
            "model_id={model_id:?} not found. Certified plugins:\n  - {}",
            available.join("\n  - ")
        )
    }
}

fn not_found_by_name(
    root: &std::path::Path,
    name: &str,
    want: &str,
    plugins: &[CertifiedPluginRuntime],
) -> String {
    let available: Vec<String> = plugins
        .iter()
        .map(|rt| format!("name={} [{}] -> {}", rt.name, rt.rule_line, rt.model_id))
        .collect();
    if available.is_empty() {
        format!(
            "no certified plugins under models-root {} (name={name:?} {want})",
            root.display()
        )
    } else {
        format!(
            "name={name:?} not found under {want}. Certified plugins:\n  - {}",
            available.join("\n  - ")
        )
    }
}

fn print_report(report: &FullMatchReport, jsonl: bool, quiet: bool) {
    if jsonl {
        // Stable JSONL stream (schema v4): a complete run ends with match_end; live
        // snapshots and failures of the same report use match_snapshot / match_failed.
        for line in report.jsonl_lines() {
            println!("{line}");
        }
        return;
    }
    if !quiet {
        for kyoku in &report.kyokus {
            println!(
                "=== kyoku#{} wind {} hand {} honba {} sticks {} dealer {} (seed {:#x}/{:#x}) ===",
                kyoku.kyoku_index,
                kyoku.bakaze,
                kyoku.kyoku,
                kyoku.honba,
                kyoku.kyotaku,
                kyoku.oya,
                kyoku.seed_hi,
                kyoku.seed_lo,
            );
            for t in &kyoku.traces {
                let fb = t
                    .fallback
                    .as_ref()
                    .map(|f| format!("  (fallback: {f})"))
                    .unwrap_or_default();
                println!(
                    "  [t{:>3}] seat {} {:<8} legal={:<2} -> {:<22} via {}/{}  {}ms{}",
                    t.turn_index,
                    t.seat,
                    t.phase,
                    t.legal_count,
                    t.selected,
                    t.agent_kind,
                    t.model_id,
                    t.latency_ms,
                    fb,
                );
            }
            println!(
                "  result: {}  deltas={:?}  scores={:?}",
                kyoku.outcome, kyoku.deltas, kyoku.scores_after
            );
        }
    }
    println!(
        "--- run-match {} platform={} length={} seed={} match_id={} (schema v{}) ---",
        report.variant,
        report.rule_profile,
        report.length,
        report.seed,
        report.match_id,
        report.schema_version
    );
    println!(
        "hands: {}  completion={}  match over: {}",
        report.kyokus.len(),
        report.completion,
        report.ended
    );
    println!("current scores: {:?}", report.current_scores);
    if report.ended {
        println!(
            "final scores: {:?}  ranking (seats): {:?}",
            report.final_scores, report.rankings
        );
    }
    println!("seat status:");
    for (i, s) in report.agents.iter().enumerate() {
        println!(
            "  seat{} {}/{} calls={} fallbacks={} errors={} last_latency={}ms",
            i,
            s.kind,
            s.model_id,
            s.calls,
            s.fallbacks,
            s.errors,
            s.last_latency_ms.unwrap_or(0),
        );
    }
}
