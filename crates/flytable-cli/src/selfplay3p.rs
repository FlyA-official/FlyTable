//! 3-player offline self-play loop.
//!
//! Same as 4-player with three seats: draw (declaring nukidora first), decide, poll
//! responses (no chi), advance. No models involved. The loop handles nukidora at
//! draw time so the base deciders stay neutral.

use flytable_core::rules::{RiichiRuleProfile, RonResolution};
use flytable_seat::{SeatDecider, TsumogiriDecider};
use flytable_table::{Board3p, KyokuOutcome3p, ReactionAction, TurnAction};

pub fn run_3p(
    seed: u64,
    print_events: bool,
    rule_profile: RiichiRuleProfile,
) -> Result<(), String> {
    let mut board = Board3p::start_with_rule_profile((seed, 0x5eed), [35000; 3], rule_profile)
        .map_err(|error| format!("RULE_WALL_PROFILE_INVALID: {error}"))?;
    let mut deciders = vec![TsumogiriDecider; 3];
    let north_kind = "N".parse::<flytable_core::tile::Tile>().unwrap().kind();

    let mut emitted = 0usize;
    let outcome = 'game: loop {
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

        // Nukidora: declare any North in hand first (a simple fixed policy; North is almost only dora in 3-player).
        while board.players[seat as usize]
            .hand
            .iter()
            .any(|t| t.kind() == north_kind)
            && flytable_table::legal_turn_actions(&board.view_for(seat))
                .contains(&TurnAction::Nukidora)
        {
            if let Some(out) = board.apply_turn(TurnAction::Nukidora).map_err(|_| {
                format!("RULE_LEGAL_ACTION_APPLY_REJECTED: seat={seat} stage=selfplay_nukidora")
            })? {
                break 'game out;
            }
            flush(&board, &mut emitted, print_events);
            if board.pending_robbery().is_some() {
                if let Some(out) = poll_robbery(&mut board, &mut deciders)? {
                    break 'game out;
                }
                flush(&board, &mut emitted, print_events);
            }
        }

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
        flush(&board, &mut emitted, print_events);

        if board.pending_robbery().is_some() {
            if let Some(out) = poll_robbery(&mut board, &mut deciders)? {
                break out;
            }
            flush(&board, &mut emitted, print_events);
        }

        if let Some((discarder, tile)) = board.last_discard {
            if let Some(out) = poll(&mut board, &mut deciders, discarder, tile)? {
                break out;
            }
            flush(&board, &mut emitted, print_events);
        }
        if board.last_discard.is_some() {
            board.advance_turn();
        }
        if board.log.len() > 10_000 {
            return Err("HOST_LOG_LIMIT_EXCEEDED: stage=selfplay_guard".to_string());
        }
    };

    flush(&board, &mut emitted, print_events);
    report(&board, &outcome);
    Ok(())
}

fn poll(
    board: &mut Board3p,
    deciders: &mut [TsumogiriDecider],
    discarder: u8,
    _tile: flytable_core::tile::Tile,
) -> Result<Option<KyokuOutcome3p>, String> {
    let mut pon: Option<(u8, [flytable_core::tile::Tile; 2])> = None;
    let mut rons = Vec::new();
    for s in 0..3u8 {
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
            ReactionAction::Pon { consumed } if pon.is_none() => pon = Some((s, consumed)),
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
    board: &mut Board3p,
    deciders: &mut [TsumogiriDecider],
) -> Result<Option<KyokuOutcome3p>, String> {
    let pending = board
        .pending_robbery()
        .ok_or_else(|| "HOST_ROBBERY_STATE_MISSING: no pending declaration".to_string())?;
    let mut rons = Vec::new();
    for s in 0..3u8 {
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
    normalize_winners(
        &mut rons,
        pending.actor,
        board.rule_profile.ron_resolution(),
    );
    board.apply_robbery_rons(&rons).map(Some).map_err(|_| {
        format!("RULE_ROBBERY_RON_APPLY_REJECTED: seats={rons:?} stage=selfplay_robbery")
    })
}

fn resolve_rons(
    board: &mut Board3p,
    discarder: u8,
    tile: flytable_core::tile::Tile,
    mut winners: Vec<u8>,
) -> Result<KyokuOutcome3p, String> {
    normalize_winners(&mut winners, discarder, board.rule_profile.ron_resolution());
    board
        .apply_rons(&winners, discarder, tile)
        .map_err(|_| format!("RULE_RON_APPLY_REJECTED: seats={winners:?} stage=selfplay_reaction"))
}

fn normalize_winners(winners: &mut Vec<u8>, from: u8, resolution: RonResolution) {
    winners.sort_by_key(|&seat| (seat + 3 - from) % 3);
    winners.dedup();
    if resolution == RonResolution::AtamaHane {
        winners.truncate(1);
    }
}

fn flush(board: &Board3p, emitted: &mut usize, print: bool) {
    if print {
        for ev in &board.log[*emitted..] {
            println!("{}", serde_json::to_string(ev).unwrap());
        }
    }
    *emitted = board.log.len();
}

fn report(board: &Board3p, outcome: &KyokuOutcome3p) {
    println!("--- result (3p) ---");
    match outcome {
        KyokuOutcome3p::Hora {
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
                println!(
                    "seat {winner} wins: {how}, {} han {} fu (dora {}, including nukidora)",
                    score.yaku.han + score.dora_han,
                    score.fu,
                    score.dora_han
                );
            }
            println!("yaku: {}", yaku_names.join(", "));
        }
        KyokuOutcome3p::MultiHora { first, additional } => {
            let mut winners = vec![first.winner];
            winners.extend(additional.iter().map(|win| win.winner));
            println!(
                "double ron, winners: {winners:?}, from seat: {:?}",
                first.from
            );
        }
        KyokuOutcome3p::Ryukyoku { tenpai } => {
            println!("exhaustive draw, tenpai seats: {tenpai:?}");
        }
        KyokuOutcome3p::NagashiMangan { winners, tenpai } => {
            println!("nagashi mangan, winners: {winners:?}, tenpai seats: {tenpai:?}");
        }
        KyokuOutcome3p::AbortiveRyukyoku { reason } => {
            println!("abortive draw: {reason:?}");
        }
    }
    println!("total events: {}", board.log.len());
}
