//! FlyTable CLI.
//!
//! A development and debugging tool (self-play, seat bus matches, translator
//! certification smoke tests), not an end-user SDK.
//!
//! - `selfplay`: offline self-play with built-in tsumogiri seats. No models,
//!   network or external processes.
//! - `run-match`: seat bus match. The host asks each seat for an action; algorithm
//!   and certified plugin seats answer through the common `SeatAgent` abstraction,
//!   and FlyTable adjudicates until the match ends. Plugin seats start certified
//!   translator subprocesses through `flytable-inference-host`; algorithm-only
//!   matches start no subprocess. Observation encoding and action masks always live
//!   in the model package.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use flytable_core::rules::RiichiRuleProfile;

mod catalog;
mod matchlog;
mod runmatch;
mod selfplay;
mod selfplay3p;

#[derive(Parser)]
#[command(name = "flytable", about = "FlyTable game core CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Runs a self-play match and writes the archived match log to a file (including decision windows).
    EmitLog {
        /// Number of seats (3 or 4).
        #[arg(long, default_value_t = 4)]
        players: u8,
        /// Random seed (reproducible).
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Match length: single / east / half.
        #[arg(long, default_value = "east")]
        length: String,
        /// Platform rules: tenhou / mahjong-soul / riichi-city.
        #[arg(long, default_value = "tenhou")]
        platform: String,
        /// Seat algorithm, repeatable. Only `tsumogiri` exists, which is also the default.
        #[arg(long = "seat", value_name = "SPEC")]
        seat: Vec<String>,
        /// Output file.
        #[arg(long, value_name = "PATH")]
        out: PathBuf,
        /// Pretty-print the output (compact by default).
        #[arg(long, default_value_t = false)]
        pretty: bool,
    },
    /// Reads an archived match log back, reruns the invariant checks and prints a summary.
    InspectLog {
        /// Match log file.
        #[arg(value_name = "PATH")]
        path: PathBuf,
        /// Print every decision window.
        #[arg(long, default_value_t = false)]
        verbose: bool,
    },
    /// Runs one offline self-play hand (rules only; no models, network or processes).
    Selfplay {
        /// Number of seats.
        #[arg(long, default_value_t = 4)]
        players: u8,
        /// Random seed (reproducible).
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Platform rules: tenhou / mahjong-soul / riichi-city.
        #[arg(long, default_value = "tenhou")]
        platform: String,
        /// Quiet: print only the settlement, not the event stream.
        #[arg(long, default_value_t = false)]
        quiet: bool,
    },
    /// Seat bus match: mixed seats (algorithms and certified plugins) play a full match.
    RunMatch {
        /// Number of seats (3 or 4).
        #[arg(long, default_value_t = 4)]
        players: u8,
        /// Random seed (reproducible).
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Seat configuration, repeatable: `--seat 0=tsumogiri --seat 2=plugin:<model_id>`.
        /// Unspecified seats default to tsumogiri.
        #[arg(long = "seat", value_name = "I=SPEC")]
        seat: Vec<String>,
        /// Match length: `single` (default, one hand) / `east` (tonpuusen) / `half` (hanchan).
        #[arg(long = "length", value_name = "LEN", default_value = "single")]
        length: String,
        /// Platform rules: tenhou / mahjong-soul / riichi-city.
        #[arg(long, default_value = "tenhou")]
        platform: String,
        /// Plugin models root (the registry scans `<root>/{4p,3p}/<plugin>`); required with plugin seats.
        #[arg(long = "models-root", value_name = "PATH")]
        models_root: Option<PathBuf>,
        /// Print the per-move decision trace as JSONL (human-readable table by default).
        #[arg(long, default_value_t = false)]
        jsonl: bool,
        /// Quiet: print only the settlement and seat status, not the per-move trace.
        #[arg(long, default_value_t = false)]
        quiet: bool,
        /// After the match, also print RuntimeStatus (seat to agent/model plus health) as one JSON line.
        #[arg(long, default_value_t = false)]
        status: bool,
    },
    /// Lists the local model catalog (built-in algorithms, certified plugins, failed or skipped entries).
    Catalog {
        /// Plugin models root (the registry scans `<root>/{4p,3p}/<plugin>`); without it only built-in algorithms are listed.
        #[arg(long = "models-root", value_name = "PATH")]
        models_root: Option<PathBuf>,
        /// Print as JSON (human-readable table by default).
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::EmitLog {
            players,
            seed,
            length,
            platform,
            seat,
            out,
            pretty,
        } => {
            if let Err(e) = matchlog::emit(players, seed, &length, &platform, &seat, &out, pretty) {
                eprintln!("failed to export match log: {e}");
                std::process::exit(1);
            }
        }
        Command::InspectLog { path, verbose } => {
            if let Err(e) = matchlog::inspect(&path, verbose) {
                eprintln!("failed to read match log: {e}");
                std::process::exit(1);
            }
        }
        Command::Selfplay {
            players,
            seed,
            platform,
            quiet,
        } => {
            let rule_profile = platform
                .parse::<RiichiRuleProfile>()
                .unwrap_or_else(|error| {
                    eprintln!("{error}");
                    std::process::exit(2);
                });
            let result = match players {
                4 => selfplay::run_4p(seed, !quiet, rule_profile),
                3 => selfplay3p::run_3p(seed, !quiet, rule_profile),
                _ => {
                    eprintln!("only 3 or 4 seats are supported.");
                    std::process::exit(2);
                }
            };
            if let Err(error) = result {
                eprintln!("host failed: {error}");
                std::process::exit(1);
            }
        }
        Command::RunMatch {
            players,
            seed,
            seat,
            models_root,
            jsonl,
            quiet,
            length,
            platform,
            status,
        } => {
            if let Err(e) = runmatch::run(
                players,
                seed,
                seat,
                models_root,
                jsonl,
                quiet,
                &length,
                &platform,
                status,
            ) {
                eprintln!("run-match failed: {e}");
                std::process::exit(2);
            }
        }
        Command::Catalog { models_root, json } => {
            if let Err(e) = catalog::run(models_root, json) {
                eprintln!("catalog failed: {e}");
                std::process::exit(2);
            }
        }
    }
}
