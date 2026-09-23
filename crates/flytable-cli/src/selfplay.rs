//! 4-player offline self-play loop (with call and ron responses).
//!
//! Draw, ask the current seat, apply, poll the other seats on a discard (ron / pon),
//! advance, until someone wins or the wall runs out. No models involved.

use flytable_core::rules::{RiichiRuleProfile, RonResolution};
use flytable_seat::{SeatDecider, TsumogiriDecider};
use flytable_table::{Board4p, KyokuOutcome, ReactionAction};

pub fn run_4p(
    seed: u64,
    print_events: bool,
    rule_profile: RiichiRuleProfile,
) -> Result<(), String> {
    let mut board = Board4p::start_with_rule_profile((seed, 0x5eed), [25000; 4], rule_profile)
        .map_err(|error| format!("RULE_WALL_PROFILE_INVALID: {error}"))?;
    let mut deciders = vec![TsumogiriDecider; 4];

    let mut emitted = 0usize;
    let outcome = loop {
        let wall_before = board.wall.live_remaining();
        if board.draw_for_turn().is_none() {
            if wall_before > 0 {
                return Err(
                    "RULE_DRAW_APPLY_REJECTED: live wall was non-empty in selfplay".to_string(),
                );
            }
            break board.ryukyoku();
        }
        let seat = board.turn;
        let view = board.view_for(seat);
        view.assert_no_hidden_truth();

        let action = deciders[seat as usize].decide_turn(&view);
        match board.apply_turn(action) {
            Ok(Some(out)) => break out,
            Ok(None) => {}
            Err(_) => {
                return Err(format!(
                    "RULE_LEGAL_ACTION_APPLY_REJECTED: seat={seat} stage=selfplay_turn"
                ));
            }
        }
        flush_events(&board, &mut emitted, print_events);

        if board.pending_robbery().is_some() {
            if let Some(out) = poll_robbery(&mut board, &mut deciders)? {
                break out;
            }
            flush_events(&board, &mut emitted, print_events);
        }

        // Poll responses after a discard (ron before pon).
        if let Some((discarder, tile)) = board.last_discard {
            if let Some(out) = poll_reactions(&mut board, &mut deciders, discarder, tile)? {
                break out;
            }
            flush_events(&board, &mut emitted, print_events);
        }

        // If `last_discard` is still set (nobody called), move to the next seat; a pon changes `turn` itself.
        if board.last_discard.is_some() {
            board.advance_turn();
        }

        if board.log.len() > 10_000 {
            return Err("HOST_LOG_LIMIT_EXCEEDED: stage=selfplay_guard".to_string());
        }
    };

    flush_events(&board, &mut emitted, print_events);
    report(&board, &outcome);
    Ok(())
}

/// Polls the non-discarding seats. Ron first, then pon. `Some` means the hand ended (ron).
fn poll_reactions(
    board: &mut Board4p,
    deciders: &mut [TsumogiriDecider],
    discarder: u8,
    _tile: flytable_core::tile::Tile,
) -> Result<Option<KyokuOutcome>, String> {
    let mut pon: Option<(u8, [flytable_core::tile::Tile; 2])> = None;
    let mut rons = Vec::new();
    for s in 0..4u8 {
        if s == discarder {
            continue;
        }
        let legal = board.legal_reactions(s);
        if legal.is_empty() {
            continue;
        }
        let view = board.view_for_reaction(s);
        let action = deciders[s as usize].decide_reaction(&view, &legal);
        if legal.contains(&ReactionAction::Ron) && !matches!(action, ReactionAction::Ron) {
            board.note_missed_ron(s).map_err(|_| {
                format!("RULE_MISSED_RON_APPLY_REJECTED: seat={s} stage=selfplay_reaction")
            })?;
        }
        match action {
            ReactionAction::Ron => rons.push(s),
            ReactionAction::Pon { consumed } if pon.is_none() => {
                pon = Some((s, consumed));
            }
            _ => {}
        }
    }
    if !rons.is_empty() {
        let tile = board
            .last_discard
            .map(|(_, tile)| tile)
            .ok_or_else(|| "HOST_REACTION_STATE_MISSING: no last discard".to_string())?;
        return resolve_rons(board, discarder, tile, rons).map(Some);
    }
    if let Some((s, consumed)) = pon {
        board
            .apply_pon(s, consumed)
            .map_err(|_| format!("RULE_CALL_APPLY_REJECTED: seat={s} stage=selfplay_reaction"))?;
    }
    Ok(None)
}

fn poll_robbery(
    board: &mut Board4p,
    deciders: &mut [TsumogiriDecider],
) -> Result<Option<KyokuOutcome>, String> {
    let pending = board
        .pending_robbery()
        .ok_or_else(|| "HOST_ROBBERY_STATE_MISSING: no pending declaration".to_string())?;
    let mut rons = Vec::new();
    for s in 0..4u8 {
        if s == pending.actor {
            continue;
        }
        let legal = board.legal_robbery_reactions(s);
        if legal.is_empty() {
            continue;
        }
        let view = board.view_for_reaction(s);
        let action = deciders[s as usize].decide_reaction(&view, &legal);
        if legal.contains(&ReactionAction::Ron) && !matches!(action, ReactionAction::Ron) {
            board.note_missed_ron(s).map_err(|_| {
                format!("RULE_MISSED_RON_APPLY_REJECTED: seat={s} stage=selfplay_robbery")
            })?;
        }
        if matches!(action, ReactionAction::Ron) {
            rons.push(s);
        }
    }
    if rons.is_empty() {
        return board.resolve_pending_robbery_passes().map_err(|_| {
            "RULE_ROBBERY_PASS_RESOLUTION_FAILED: stage=selfplay_robbery".to_string()
        });
    }
    resolve_robbery_rons(board, pending.actor, rons).map(Some)
}

fn resolve_rons(
    board: &mut Board4p,
    discarder: u8,
    tile: flytable_core::tile::Tile,
    mut winners: Vec<u8>,
) -> Result<KyokuOutcome, String> {
    normalize_winners(&mut winners, discarder, board.rule_profile.ron_resolution());
    if board.rule_profile.ron_resolution() == RonResolution::TripleRonAbortive && winners.len() == 3
    {
        return board
            .abort_sanchaho()
            .map_err(|_| "RULE_SANCHAHO_APPLY_REJECTED: stage=selfplay_reaction".to_string());
    }
    board
        .apply_rons(&winners, discarder, tile)
        .map_err(|_| format!("RULE_RON_APPLY_REJECTED: seats={winners:?} stage=selfplay_reaction"))
}

fn resolve_robbery_rons(
    board: &mut Board4p,
    actor: u8,
    mut winners: Vec<u8>,
) -> Result<KyokuOutcome, String> {
    normalize_winners(&mut winners, actor, board.rule_profile.ron_resolution());
    if board.rule_profile.ron_resolution() == RonResolution::TripleRonAbortive && winners.len() == 3
    {
        return board
            .abort_sanchaho()
            .map_err(|_| "RULE_SANCHAHO_APPLY_REJECTED: stage=selfplay_robbery".to_string());
    }
    board.apply_robbery_rons(&winners).map_err(|_| {
        format!("RULE_ROBBERY_RON_APPLY_REJECTED: seats={winners:?} stage=selfplay_robbery")
    })
}

fn normalize_winners(winners: &mut Vec<u8>, from: u8, resolution: RonResolution) {
    winners.sort_by_key(|&seat| (seat + 4 - from) % 4);
    winners.dedup();
    if resolution == RonResolution::AtamaHane {
        winners.truncate(1);
    }
}

fn flush_events(board: &Board4p, emitted: &mut usize, print: bool) {
    if print {
        for ev in &board.log[*emitted..] {
            println!("{}", serde_json::to_string(ev).unwrap());
        }
    }
    *emitted = board.log.len();
}

fn report(board: &Board4p, outcome: &KyokuOutcome) {
    println!("--- result ---");
    match outcome {
        KyokuOutcome::Hora {
            winner,
            from,
            score,
        } => {
            let how = match from {
                Some(f) => format!("ron (from seat {f})"),
                None => "tsumo".to_string(),
            };
            let yaku_names: Vec<String> = score
                .yaku
                .yaku
                .iter()
                .map(|(y, h)| format!("{}({h})", y.name()))
                .collect();
            if score.score.yakuman > 0 {
                println!(
                    "seat {winner} wins: {how}, yakuman x{}",
                    score.score.yakuman
                );
            } else {
                let payout = if from.is_some() {
                    format!("{} points", score.score.ron)
                } else if *winner == board.oya {
                    format!("{} x3 points", score.score.tsumo_ko)
                } else {
                    format!(
                        "dealer {} / non-dealer {} points",
                        score.score.tsumo_oya, score.score.tsumo_ko
                    )
                };
                println!(
                    "seat {winner} wins: {how}, {} han {} fu (dora {}) -> {payout}",
                    score.yaku.han + score.dora_han,
                    score.fu,
                    score.dora_han
                );
            }
            println!("yaku: {}", yaku_names.join(", "));
        }
        KyokuOutcome::MultiHora { first, additional } => {
            let mut winners = vec![first.winner];
            winners.extend(additional.iter().map(|win| win.winner));
            println!(
                "multiple ron, winners: {winners:?}, from seat: {:?}",
                first.from
            );
        }
        KyokuOutcome::Ryukyoku { tenpai } => {
            println!("exhaustive draw, tenpai seats: {tenpai:?}");
        }
        KyokuOutcome::NagashiMangan { winners, tenpai } => {
            println!("nagashi mangan, winners: {winners:?}, tenpai seats: {tenpai:?}");
        }
        KyokuOutcome::AbortiveRyukyoku { reason } => {
            println!("abortive draw: {reason:?}");
        }
    }
    println!("total events: {}", board.log.len());
}
