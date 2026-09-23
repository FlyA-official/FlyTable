//! `catalog` subcommand: lists the local model catalog merged by the runtime
//! (built-in algorithms plus the inference-host registry).

use std::path::PathBuf;

use flytable_runtime::{CatalogEntry, list_catalog};

pub fn run(models_root: Option<PathBuf>, json: bool) -> Result<(), String> {
    let entries = list_catalog(models_root.as_deref());
    if json {
        for entry in &entries {
            println!("{}", serde_json::to_string(entry).unwrap());
        }
        return Ok(());
    }
    println!("model catalog ({} entries):", entries.len());
    for e in &entries {
        print_entry(e);
    }
    if models_root.is_none() {
        println!(
            "(no --models-root given; listing built-in algorithms only. Pass it to scan local plugins)"
        );
    }
    Ok(())
}

fn print_entry(e: &CatalogEntry) {
    let id = e.model_id.as_deref().unwrap_or("<uncertified>");
    let style = e.riichi_style.as_deref().unwrap_or("-");
    let smoke = e
        .smoke
        .as_ref()
        .map(|s| {
            format!(
                "  smoke(decisions {}/all legal {}/max {}ms)",
                s.decisions, s.all_legal, s.max_infer_ms
            )
        })
        .unwrap_or_default();
    let err = e
        .error_summary
        .as_ref()
        .map(|r| format!("  reason: {r}"))
        .unwrap_or_default();
    println!(
        "  [{:?}/{}] {} «{}» rule_line={:?} style={} health={:?} usable={}{}{}",
        e.source,
        e.seat_agent_kind,
        id,
        e.display_name,
        e.rule_lines,
        style,
        e.health,
        e.usable,
        smoke,
        err,
    );
}
