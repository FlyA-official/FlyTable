//! Match log export and inspection from the command line: play a match, save it,
//! read it back.

use std::path::Path;

use flytable_core::rules::RiichiRuleProfile;
use flytable_event::matchlog::{MatchSettlementProfile, MatchlogEvent};
use flytable_runtime::{
    AlgorithmKind, LiveMatchConfig, LiveMatchSession, LiveSeat, LiveStepResult, MatchArchive,
    MatchKind, archive_kyoku_3p, archive_kyoku_4p, start_match_event,
};

/// Parses seat specs; only `tsumogiri` exists.
fn seats(specs: &[String], n: usize) -> Result<Vec<LiveSeat>, String> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let spec = specs.get(i).map_or("tsumogiri", String::as_str);
        let kind = AlgorithmKind::parse(spec).ok_or_else(|| {
            format!("unknown algorithm `{spec}` for seat {i} (only tsumogiri is available)")
        })?;
        out.push(LiveSeat::Algorithm(kind));
    }
    Ok(out)
}

/// Runs a self-play match and writes the match log to `out`.
///
/// # Errors
///
/// Unknown seat spec, host failure, archive rejection (see `ArchiveError`), or a write error.
pub fn emit(
    players: u8,
    seed: u64,
    length: &str,
    platform: &str,
    seat_specs: &[String],
    out: &Path,
    pretty: bool,
) -> Result<(), String> {
    let profile = platform
        .parse::<RiichiRuleProfile>()
        .map_err(|e| e.to_string())?;
    let kind = match length {
        "single" => MatchKind::Single,
        "east" => MatchKind::East,
        "half" => MatchKind::Half,
        other => return Err(format!("unknown match length `{other}` (single/east/half)")),
    };
    // Record decision windows; an exported log without them is useless for review.
    let config = LiveMatchConfig::new(seed, kind)
        .with_rule_profile(profile)
        .with_decision_windows(true);

    let kyokus = match players {
        4 => {
            let mut s = LiveMatchSession::new_4p(config, seats(seat_specs, 4)?)
                .map_err(|e| e.to_string())?;
            drive(&mut s, archive_kyoku_4p)?
        }
        3 => {
            let mut s = LiveMatchSession::new_3p(config, seats(seat_specs, 3)?)
                .map_err(|e| e.to_string())?;
            drive(&mut s, archive_kyoku_3p)?
        }
        n => return Err(format!("only 3 or 4 seats are supported, got {n}")),
    };

    // Rank points and oka are filled in by whoever sets up the table. Self-play is not a
    // ranked match, but Tenhou has well-defined values, so they are included to make
    // the export closer to a real log.
    let settlement = match (platform, players) {
        ("tenhou", 4) => Some(MatchSettlementProfile::tenhou_4p()),
        ("tenhou", 3) => Some(MatchSettlementProfile::tenhou_3p()),
        // Other platforms vary by room; the engine does not guess.
        _ => None,
    };
    let archive = MatchArchive {
        start_match: start_match_event(profile, players, settlement),
        kyokus,
    };
    // Validate before writing; writing a log that fails its own checks is worse than writing nothing.
    archive.validate_all()?;

    let json = if pretty {
        serde_json::to_string_pretty(&archive)
    } else {
        serde_json::to_string(&archive)
    }
    .map_err(|e| e.to_string())?;
    std::fs::write(out, &json).map_err(|e| format!("failed to write {}: {e}", out.display()))?;

    let events: usize = archive.kyokus.iter().map(|k| k.events.len()).sum();
    let windows: usize = archive.kyokus.iter().map(|k| k.windows.len()).sum();
    println!(
        "wrote {}\n  hands {} · events {} · decision windows {} · {} bytes",
        out.display(),
        archive.kyokus.len(),
        events,
        windows,
        json.len()
    );
    Ok(())
}

/// Drives a fully automatic match and archives each hand.
fn drive<B, F>(session: &mut LiveMatchSession<B>, archive: F) -> Result<Vec<K>, String>
where
    B: flytable_runtime::live::LiveBoard,
    F: Fn(&LiveMatchSession<B>) -> Result<K, flytable_runtime::ArchiveError>,
{
    let mut out = Vec::new();
    loop {
        match session.drive_until_wait_or_end() {
            LiveStepResult::KyokuEnded => {
                out.push(archive(session).map_err(|e| e.to_string())?);
                if !session.start_next_kyoku().map_err(|e| e.to_string())? {
                    break;
                }
            }
            LiveStepResult::MatchEnded => break,
            LiveStepResult::InProgress => {}
            LiveStepResult::Failed { failure } => return Err(format!("host failed: {failure}")),
            other => {
                return Err(format!(
                    "automatic seats should never leave a window pending: {other:?}"
                ));
            }
        }
    }
    Ok(out)
}

type K = flytable_runtime::KyokuArchive;

/// Reads a match log back and prints a summary.
///
/// Checks always rerun on read: the file may have been edited or written by another
/// version.
///
/// # Errors
///
/// Read failure, JSON parse failure (including unknown fields), or failed invariant checks.
pub fn inspect(path: &Path, verbose: bool) -> Result<(), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let archive: MatchArchive =
        serde_json::from_str(&text).map_err(|e| format!("parse failed: {e}"))?;
    archive.validate_all()?;

    let MatchlogEvent::StartMatch {
        schema,
        seats,
        rule_era,
        profile_fingerprint,
        tile_identity,
        ..
    } = &archive.start_match
    else {
        return Err("the first event is not the header".into());
    };

    println!("match log {}", path.display());
    println!("  schema {schema} · {seats} seats · source {}", rule_era.0);
    println!("  rule profile fingerprint {profile_fingerprint}");
    println!("  physical tile identity {tile_identity:?}");
    println!("  hands {}", archive.kyokus.len());

    for (i, k) in archive.kyokus.iter().enumerate() {
        let outcome = match k.events.last() {
            Some(MatchlogEvent::Hora(b)) => {
                let how = if b.from == b.winner { "tsumo" } else { "ron" };
                format!("{how}, seat {} · {} fu", b.winner, b.fu)
            }
            Some(MatchlogEvent::Ryukyoku(b)) => format!("{:?}", b.kind),
            _ => "(no settlement layer)".into(),
        };
        println!(
            "  hand {}: events {} · windows {} · {}",
            i + 1,
            k.events.len(),
            k.windows.len(),
            outcome
        );
        if verbose {
            for w in &k.windows {
                println!(
                    "      window {} seat {} {:?} anchor {} · {} offers -> {}",
                    w.window_id,
                    w.seat,
                    w.phase,
                    w.anchor_seq,
                    w.offers.len(),
                    w.chosen
                );
            }
        }
    }
    println!("\nvalidation passed (rechecked after reading back).");
    Ok(())
}
