# FlyTable

FlyTable is a rules kernel and authoritative table engine for Japanese Riichi Mahjong, covering both four-player and three-player (sanma) games. It is written in Rust and developed by Nashout as the rules core of FlyAgent and other FlyA products.

This repository is a public edition of the kernel, published primarily for academic study and reference.

## About this edition

The public edition is a periodic snapshot of the kernel used in FlyA products. It contains the rules engine, the table engine, the protocol contracts, the engine host and the runtime. The following are not included:

- internal tooling and test suites;
- integrations with online services;
- built-in AI players. The only built-in seat is a tsumogiri bot (see [Seats](#seats)).

## Features

- **Rules core**: tiles, hands and melds; shanten; winning-hand decomposition; yaku and yakuman; fu, han and point calculation.
- **Four-player and three-player variants**: sanma is a first-class variant, including kita (north extraction) and its scoring.
- **Platform rule profiles**: presets for Tenhou, Mahjong Soul and Riichi City. Differences between platforms are modelled as named rule knobs. [`docs/rule-knobs.toml`](docs/rule-knobs.toml) lists each knob's value per platform and the evidence behind it.
- **Authoritative table engine**: wall building and dealing, turn flow, calls, robbing a kan, abortive and exhaustive draws, settlement, and match progression (single hand, East-only, half game).
- **Information isolation**: every decision maker receives only its own seat's view. Opponents' concealed tiles are never exposed.
- **Legal-action enumeration**: the table computes the complete set of legal actions for every decision. Seats choose from it, and FlyTable adjudicates the outcome.
- **Event protocol**: an mjai-style event stream for both variants, with canonical contracts for seat-visible events, legal actions and decisions.
- **Match logs**: `flytable-matchlog-v1`, a layered archival format (facts, adjudication, settlement, decision windows) that is re-validated whenever it is read back.
- **Engine host**: external engines run as separate processes behind a versioned protocol, with plugin discovery and certification.
- **Command-line tool**: matches, self-play, match-log export and inspection, and seat catalog listing.

## Repository layout

| Crate | Responsibility |
| --- | --- |
| `flytable-core` | Tiles, hands, melds, shanten, winning decomposition, yaku, fu/han, scoring, rule profiles |
| `flytable-event` | Four-player and three-player event types; the `flytable-matchlog-v1` archival format |
| `flytable-protocol` | Seat-visible events, legal actions, decisions and capabilities; match-log projection and replay |
| `flytable-maintainer` | Per-seat views, legal-action enumeration, read-only analysis, table mirroring |
| `flytable-table` | The authoritative table: wall, dealing, progression, adjudication, settlement |
| `flytable-seat` | The seat decision interface and the built-in tsumogiri bot |
| `flytable-inference-host` | External engine processes, the inference protocol, plugin discovery and certification |
| `flytable-runtime` | Seat bus and match host: mixed seats, full matches, live sessions, match-log archiving |
| `flytable-cli` | The `flytable` command-line tool |

The rules crates (`flytable-core`, `flytable-event`, `flytable-protocol`, `flytable-maintainer`, `flytable-table` and `flytable-seat`) perform no I/O: no networking, no subprocesses, no file-system access and no model inference. Side effects are confined to `flytable-inference-host`, `flytable-runtime` and the command-line tool.

## Getting started

FlyTable requires Rust 1.96 or later.

```sh
cargo build --release
```

This produces the `flytable` binary in `target/release/`. Some examples:

```sh
# A half game between four tsumogiri bots under Tenhou rules
flytable run-match --players 4 --length half --platform tenhou

# A three-player East-only game, printing only the summary
flytable run-match --players 3 --length east --quiet

# Export an archival match log, then read it back and re-validate it
flytable emit-log --players 4 --length east --out match.json --pretty
flytable inspect-log match.json

# List the seats available locally: built-ins and certified plugins
flytable catalog --models-root ./models
```

Run `flytable help` or `flytable <command> --help` for the full set of options.

## Seats

Each seat is filled by a decision maker. One is built in:

| Spec | Behaviour |
| --- | --- |
| `tsumogiri` | Discards the tile it has just drawn. It never calls, never declares riichi and never wins. This is the auto-play applied to a disconnected player, and it is the default for any seat left unassigned. |

Stronger players are supplied as external engines through the plugin interface:

```sh
flytable run-match --players 4 --models-root ./models --seat 0=plugin:<model_id>
```

`plugin@<name>` selects a certified plugin by the name declared in its manifest.

## Engine plugins

An engine runs as a separate process and exchanges JSON Lines with FlyTable over stdin and stdout, using the `flya-inference-v1` or `flya-inference-v2` protocol (`hello`, `infer`, `end`). For each decision, FlyTable sends the event history visible to that seat together with the authoritative list of legal actions. The engine replies with its choice from that list, and FlyTable checks the reply against the request digests before applying it. Engines never enumerate legal actions or adjudicate rules themselves. Through the same decision contract, the runtime can also seat engines that are reached over HTTP.

Plugins are discovered under a models root:

```text
<models-root>/
├── 4p/<plugin>/plugin.toml        a four-player engine
├── 3p/<plugin>/plugin.toml        a three-player engine
└── <package>/flya-package.toml    a package declaring one or more models
```

A plugin must be certified before it can take a seat. FlyTable launches it and requires legal answers to a fixed set of sampled decisions on synthetic tables. The result is cached in `.flya-cert.json` against a hash of the package contents and is invalidated whenever the package changes.

> **Security note:** loading a plugin executes arbitrary code. Certification checks protocol conformance; it is not a sandbox. Only run plugins you trust.

The wire format is defined in `flytable-protocol` and `flytable-inference-host`.

## Design principles

- **A single source of rule truth.** Only the rules crates decide what is legal and how a hand scores. Seats and engines choose among the options they are given.
- **Engines stay outside the kernel.** Observation encoding, action spaces and models belong to the engine. FlyTable exchanges events and choices only.
- **Isolation by construction.** Seat views are built per seat and carry no hidden information.
- **Explicit platform differences.** Every supported difference between platforms is a named rule knob with a documented value and evidence grade.
- **Reproducibility.** Walls are generated from explicit seeds, so the same seed and deterministic seats reproduce the same match.

## Development and contributions

FlyTable is developed in Nashout's internal repository, and this edition is updated from it periodically. Bug reports and questions are welcome as issues. Pull requests are reviewed, but accepted changes are applied internally and appear here with the next update.

The command-line tool is intended for development and inspection rather than as an end-user application.

## Trademarks

Tenhou, Mahjong Soul and Riichi City are trademarks of their respective owners. FlyTable is an independent project and is not affiliated with or endorsed by any of them, and it does not connect to any game service.

## Copyright and license

Copyright © 2026 Nashout. FlyA and FlyAgent are products of Nashout.

FlyTable is released under the [MIT License](LICENSE).
