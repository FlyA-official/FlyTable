//! Match log invariant checks: widths, meld shapes, disjoint and conserved tile
//! sets, seat-count constraints and yaku table exclusivity.
//!
//! These rules cannot be expressed in the schema and run after deserialization.
//! The engine core does not track physical tiles, so consistency is checked here.

use std::collections::BTreeMap;

use super::event::{HoraBody, MatchlogEvent, RyukyokuBody};
use super::types::{MatchlogMeld, TileRef};

/// An invariant violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    /// Index of the event in the stream (`None` for global violations).
    pub seq: Option<usize>,
    pub code: &'static str,
    pub detail: String,
}

impl Violation {
    fn at(seq: usize, code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            seq: Some(seq),
            code,
            detail: detail.into(),
        }
    }
}

fn width(seq: usize, name: &'static str, got: usize, seats: usize, out: &mut Vec<Violation>) {
    if got != seats {
        out.push(Violation::at(
            seq,
            "width_mismatch",
            format!("{name} length {got} != seats {seats}"),
        ));
    }
}

fn zero_sum(seq: usize, name: &'static str, deltas: &[i32], out: &mut Vec<Violation>) {
    let sum: i32 = deltas.iter().sum();
    if sum != 0 {
        out.push(Violation::at(
            seq,
            "not_zero_sum",
            format!("{name} sums to {sum}, expected 0"),
        ));
    }
}

fn check_tile_refs(seq: usize, name: &'static str, refs: &[TileRef], out: &mut Vec<Violation>) {
    for r in refs {
        if !r.is_consistent() {
            out.push(Violation::at(
                seq,
                "tile_id_inconsistent",
                format!(
                    "{name}: physical_id {:?} is inconsistent with tile {:?}",
                    r.physical_id, r.tile
                ),
            ));
        }
    }
}

fn meld_refs(meld: &MatchlogMeld) -> Vec<TileRef> {
    let mut v = meld.consumed.clone();
    if let Some(c) = meld.claimed {
        v.push(c);
    }
    v
}

fn check_hora(seq: usize, body: &HoraBody, seats: usize, out: &mut Vec<Violation>) {
    width(seq, "hora.base_deltas", body.base_deltas.len(), seats, out);
    width(
        seq,
        "hora.honba_deltas",
        body.honba_deltas.len(),
        seats,
        out,
    );
    width(
        seq,
        "hora.kyotaku_deltas",
        body.kyotaku_deltas.len(),
        seats,
        out,
    );

    // Only base and honba are zero-sum. Riichi sticks move from the table pool to the
    // winner, so sum(kyotaku_deltas) + pool change = 0, and the pool change is not part
    // of this event.
    zero_sum(seq, "hora.base_deltas", &body.base_deltas, out);
    zero_sum(seq, "hora.honba_deltas", &body.honba_deltas, out);
    if body.kyotaku_deltas.iter().sum::<i32>() < 0 {
        out.push(Violation::at(
            seq,
            "kyotaku_negative",
            "riichi sticks only flow from the pool to the winner; the sum must not be negative",
        ));
    }

    // If `yakuman` is non-empty, `normal` must be empty. The reverse is valid: counted
    // yakuman use `normal` plus `limit`.
    if !body.yakuman.is_empty() && !body.normal.is_empty() {
        out.push(Violation::at(
            seq,
            "yaku_tables_overlap",
            "normal must be empty when yakuman is non-empty",
        ));
    }
    for (id, _) in body.normal.iter().chain(body.yakuman.iter()) {
        if *id > 54 {
            out.push(Violation::at(
                seq,
                "yaku_id_out_of_range",
                format!("yaku id {id} > 54"),
            ));
        }
    }
    // Dora components (52 dora, 53 ura, 54 red) go into their own fields, never `normal`.
    for (id, _) in &body.normal {
        if matches!(id, 52..=54) {
            out.push(Violation::at(
                seq,
                "dora_in_yaku_table",
                format!("dora yaku id {id} belongs in dora_han and related fields"),
            ));
        }
    }

    for m in &body.melds {
        if !m.shape_is_valid(seats) {
            out.push(Violation::at(
                seq,
                "meld_shape_invalid",
                format!("{:?} violates the meld shape invariant", m.kind),
            ));
        }
        check_tile_refs(seq, "hora.melds", &meld_refs(m), out);
    }
    check_tile_refs(seq, "hora.concealed", &body.concealed, out);
    check_tile_refs(seq, "hora.machi", &[body.machi], out);
    check_tile_refs(seq, "hora.ura_markers", &body.ura_markers, out);

    // `concealed`, `melds` and `machi` must be disjoint, checked by physical ID
    // (skipped when there are no physical IDs).
    let mut seen: BTreeMap<u8, &'static str> = BTreeMap::new();
    let mut note = |r: &TileRef, where_: &'static str, out: &mut Vec<Violation>| {
        if let Some(id) = r.physical_id
            && let Some(prev) = seen.insert(id, where_)
        {
            out.push(Violation::at(
                seq,
                "tile_sets_overlap",
                format!("physical tile {id} appears in both {prev} and {where_}"),
            ));
        }
    };
    note(&body.machi, "machi", out);
    for r in &body.concealed {
        note(r, "concealed", out);
    }
    for m in &body.melds {
        for r in meld_refs(m) {
            note(&r, "melds", out);
        }
    }

    if body.winner as usize >= seats || body.from as usize >= seats {
        out.push(Violation::at(
            seq,
            "seat_out_of_range",
            "winner / from out of range",
        ));
    }
    if let Some(pao) = body.pao
        && pao as usize >= seats
    {
        out.push(Violation::at(seq, "seat_out_of_range", "pao out of range"));
    }
}

fn check_ryukyoku(seq: usize, body: &RyukyokuBody, seats: usize, out: &mut Vec<Violation>) {
    width(
        seq,
        "ryukyoku.tenpai_mask",
        body.tenpai_mask.len(),
        seats,
        out,
    );
    width(seq, "ryukyoku.concealed", body.concealed.len(), seats, out);
    width(
        seq,
        "ryukyoku.base_deltas",
        body.base_deltas.len(),
        seats,
        out,
    );
    width(
        seq,
        "ryukyoku.honba_deltas",
        body.honba_deltas.len(),
        seats,
        out,
    );
    zero_sum(seq, "ryukyoku.base_deltas", &body.base_deltas, out);
    zero_sum(seq, "ryukyoku.honba_deltas", &body.honba_deltas, out);

    if !body.kind.possible_with(seats) {
        out.push(Violation::at(
            seq,
            "ryukyoku_kind_impossible",
            format!("{:?} cannot occur with {seats} players", body.kind),
        ));
    }
    for hand in body.concealed.iter().flatten() {
        check_tile_refs(seq, "ryukyoku.concealed", hand, out);
    }
}

/// Checks all global invariants and returns every violation without stopping at the
/// first one.
///
/// `seats` comes from the `StartMatch` event; a missing header is fatal.
#[must_use]
pub fn validate(events: &[MatchlogEvent]) -> Vec<Violation> {
    let mut out = Vec::new();

    let Some(MatchlogEvent::StartMatch {
        seats,
        schema,
        profile_fingerprint: fp,
        rule_profile,
        settlement_profile,
        ..
    }) = events.first()
    else {
        out.push(Violation {
            seq: None,
            code: "missing_start_match",
            detail: "the first event must be start_match (header)".into(),
        });
        return out;
    };
    let seats = *seats as usize;

    if schema != super::types::MATCHLOG_SCHEMA {
        out.push(Violation::at(
            0,
            "schema_mismatch",
            format!("unknown schema {schema}"),
        ));
    }
    if !matches!(seats, 3 | 4) {
        out.push(Violation::at(
            0,
            "seats_out_of_range",
            format!("seats = {seats}"),
        ));
    }
    let want_fp = super::types::profile_fingerprint(rule_profile);
    if *fp != want_fp {
        out.push(Violation::at(
            0,
            "fingerprint_mismatch",
            format!("declared {fp}, computed {want_fp}"),
        ));
    }
    if let Some(sp) = settlement_profile
        && !sp.is_valid_for(seats)
    {
        out.push(Violation::at(
            0,
            "width_mismatch",
            format!(
                "settlement_profile.rank_points length {} != seats {seats}",
                sp.rank_points.len()
            ),
        ));
    }

    for (seq, ev) in events.iter().enumerate() {
        match ev {
            MatchlogEvent::StartMatch { .. } if seq != 0 => {
                out.push(Violation::at(
                    seq,
                    "duplicate_start_match",
                    "start_match may only appear at the start of the stream",
                ));
            }
            MatchlogEvent::StartKyoku {
                scores,
                haipai,
                dora_marker,
                ..
            } => {
                width(seq, "start_kyoku.scores", scores.len(), seats, &mut out);
                width(seq, "start_kyoku.haipai", haipai.len(), seats, &mut out);
                for hand in haipai {
                    if hand.len() != 13 {
                        out.push(Violation::at(
                            seq,
                            "haipai_size",
                            format!("{} starting tiles, expected 13", hand.len()),
                        ));
                    }
                    check_tile_refs(seq, "start_kyoku.haipai", hand, &mut out);
                }
                check_tile_refs(seq, "start_kyoku.dora_marker", &[*dora_marker], &mut out);
            }
            MatchlogEvent::Call { meld, .. } => {
                if !meld.shape_is_valid(seats) {
                    out.push(Violation::at(
                        seq,
                        "meld_shape_invalid",
                        format!("{:?} violates the meld shape invariant", meld.kind),
                    ));
                }
                check_tile_refs(seq, "call.meld", &meld_refs(meld), &mut out);
            }
            MatchlogEvent::Tsumo { pai, .. }
            | MatchlogEvent::Dahai { pai, .. }
            | MatchlogEvent::DealerOpening { pai, .. }
            | MatchlogEvent::DealerOpeningDahai { pai, .. } => {
                check_tile_refs(seq, "pai", &[*pai], &mut out);
            }
            MatchlogEvent::Dora { marker, .. } => {
                check_tile_refs(seq, "dora.marker", &[*marker], &mut out);
            }
            MatchlogEvent::Hora(body) => check_hora(seq, body, seats, &mut out),
            MatchlogEvent::Ryukyoku(body) => check_ryukyoku(seq, body, seats, &mut out),
            MatchlogEvent::MatchEnd {
                final_scores,
                rank_points,
            } => {
                width(
                    seq,
                    "match_end.final_scores",
                    final_scores.len(),
                    seats,
                    &mut out,
                );
                if let Some(rp) = rank_points {
                    width(seq, "match_end.rank_points", rp.len(), seats, &mut out);
                }
            }
            _ => {}
        }

        for seat in event_seats(ev) {
            if seat as usize >= seats {
                out.push(Violation::at(
                    seq,
                    "seat_out_of_range",
                    format!("seat {seat} out of range"),
                ));
            }
        }
    }
    out
}

fn event_seats(ev: &MatchlogEvent) -> Vec<u8> {
    match ev {
        MatchlogEvent::Tsumo { actor, .. }
        | MatchlogEvent::Dahai { actor, .. }
        | MatchlogEvent::DealerOpening { actor, .. }
        | MatchlogEvent::DealerOpeningDahai { actor, .. }
        | MatchlogEvent::KyuushuDeclare { actor }
        | MatchlogEvent::ReachAccepted { actor } => vec![*actor],
        MatchlogEvent::Call { actor, meld } => {
            let mut v = vec![*actor];
            v.extend(meld.from);
            v
        }
        MatchlogEvent::StartKyoku { oya, .. } => vec![*oya],
        MatchlogEvent::PlatformDisconnect { seat } | MatchlogEvent::PlatformReconnect { seat } => {
            vec![*seat]
        }
        _ => vec![],
    }
}
