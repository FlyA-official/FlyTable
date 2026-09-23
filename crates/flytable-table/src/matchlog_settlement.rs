//! Builds the L2 settlement layer of the match log.
//!
//! These functions only take plain data (seat state, ura indicators, seat and round
//! wind, split deltas) and no board types, so 3P and 4P share them and they can be
//! checked against recorded games without the runtime. That matters for `hora_body`:
//! on a tsumo the winning tile is already in hand and one copy must be removed, and
//! removing the wrong copy (a red five instead of a plain five) is invisible to a
//! points-only check.
//!
//! Settlement math already lives in [`crate::progress`].

use flytable_core::score::{FullScore, ScoreLimit};
use flytable_core::tile::Tile;
use flytable_event::matchlog::{
    HoraBody, Limit, MatchlogEvent, MatchlogMeld, MeldKind, RyukyokuBody, RyukyokuKind, TileRef,
    tenhou_yaku_id,
};
use flytable_event::{Event3p, Event4p};
use flytable_maintainer::PlayerState;

use crate::progress::{Settlement, SettlementSplit};

pub fn tr(t: Tile) -> TileRef {
    TileRef::new(t, None)
}

pub fn trs(ts: &[Tile]) -> Vec<TileRef> {
    ts.iter().copied().map(tr).collect()
}

pub fn limit_of(limit: ScoreLimit) -> Limit {
    match limit {
        ScoreLimit::Normal => Limit::None,
        ScoreLimit::Mangan => Limit::Mangan,
        ScoreLimit::Haneman => Limit::Haneman,
        ScoreLimit::Baiman => Limit::Baiman,
        ScoreLimit::Sanbaiman => Limit::Sanbaiman,
        ScoreLimit::Yakuman(n) => Limit::Yakuman(n),
    }
}

pub fn meld_of(m: &flytable_core::meld::Meld) -> MatchlogMeld {
    use flytable_core::meld::Meld;
    match m {
        Meld::Chi {
            tiles,
            called,
            from,
        } => MatchlogMeld {
            kind: MeldKind::Chi,
            from: Some(*from),
            claimed: Some(tr(*called)),
            consumed: trs(&tiles
                .iter()
                .copied()
                .filter(|t| t != called)
                .collect::<Vec<_>>()),
        },
        Meld::Pon {
            called,
            consumed,
            from,
            ..
        } => MatchlogMeld {
            kind: MeldKind::Pon,
            from: Some(*from),
            claimed: Some(tr(*called)),
            consumed: trs(consumed),
        },
        Meld::Daiminkan { tile, called, from } => MatchlogMeld {
            kind: MeldKind::Daiminkan,
            from: Some(*from),
            claimed: Some(tr(*called)),
            consumed: vec![tr(*tile); 3],
        },
        Meld::Kakan { tile, added } => MatchlogMeld {
            kind: MeldKind::Kakan,
            // The source seat is in the original pon; the final board no longer has it, so replay fills it in.
            from: None,
            claimed: Some(tr(*added)),
            consumed: vec![tr(*tile); 3],
        },
        Meld::Ankan { tile } => MatchlogMeld {
            kind: MeldKind::Ankan,
            from: None,
            claimed: None,
            consumed: vec![tr(*tile); 4],
        },
        Meld::Nukidora { tile } => MatchlogMeld {
            kind: MeldKind::Kita,
            from: None,
            claimed: Some(tr(*tile)),
            consumed: Vec::new(),
        },
    }
}
/// `(normal, yakuman)`, both as `(Tenhou yaku id, han or multiplier)`, mutually exclusive.
pub type YakuColumns = (Vec<(u8, u8)>, Vec<(u8, u8)>);

/// Splits the yaku of a `FullScore` into the match log's `normal` and `yakuman` columns.
///
/// Rules checked by `validate`:
/// - dora are not yaku and go into `dora_han` / `ura_han` / `aka_han` / `nuki_han`;
/// - when `yakuman` is non-empty, `normal` must be empty;
/// - counted yakuman use `normal` plus `limit` and are never moved into `yakuman`.
pub fn split_yaku(score: &FullScore, jikaze: Tile, bakaze: Tile) -> YakuColumns {
    use flytable_core::yaku::Yaku;
    let mut normal = Vec::new();
    let mut yakuman = Vec::new();
    for (y, han) in &score.yaku.yaku {
        if matches!(y, Yaku::Yakuman(_)) {
            if let Some(id) = tenhou_yaku_id(y, jikaze, bakaze) {
                yakuman.push((id, (*han).max(1)));
            }
        } else if let Some(id) = tenhou_yaku_id(y, jikaze, bakaze) {
            normal.push((id, *han));
        }
    }
    if yakuman.is_empty() {
        (normal, Vec::new())
    } else {
        // Explicit yakuman: regular yaku do not score, so the column is cleared.
        (Vec::new(), yakuman)
    }
}
/// Builds the win body from plain data; identical for 3P and 4P.
#[allow(clippy::too_many_arguments)]
pub fn hora_body(
    players: &[PlayerState],
    ura_indicators: &[Tile],
    ron_tile: Option<Tile>,
    bakaze: Tile,
    jikaze: Tile,
    pao: Option<u8>,
    winner: u8,
    from: Option<u8>,
    score: &FullScore,
    split: &SettlementSplit,
) -> Result<HoraBody, String> {
    let seat = winner as usize;
    let player = &players[seat];
    let tsumo = from.is_none();

    // Winning tile: the drawn tile on tsumo, the last discard on ron.
    let machi = if tsumo {
        player
            .drawn_tile
            .or_else(|| player.hand.last().copied())
            .ok_or_else(|| "missing winning tile for tsumo".to_string())?
    } else {
        ron_tile.ok_or_else(|| "missing winning tile for ron".to_string())?
    };

    // `concealed` excludes the winning tile. On tsumo it is already in hand and one copy
    // is removed; on ron it never entered the hand.
    let mut concealed = player.hand.clone();
    if let (true, Some(pos)) = (tsumo, concealed.iter().rposition(|t| *t == machi)) {
        concealed.remove(pos);
    }

    let (normal, yakuman) = split_yaku(score, jikaze, bakaze);

    Ok(HoraBody {
        winner,
        // The match log expresses tsumo as `from == winner`.
        from: from.unwrap_or(winner),
        machi: tr(machi),
        concealed: trs(&concealed),
        melds: player.melds.iter().map(meld_of).collect(),
        normal,
        yakuman,
        fu: score.fu,
        limit: limit_of(score.score.limit),
        dora_han: score.regular_dora_han,
        ura_han: score.ura_dora_han,
        aka_han: score.red_dora_han,
        nuki_han: score.nuki_dora_han,
        ura_markers: if player.riichi {
            trs(ura_indicators)
        } else {
            Vec::new()
        },
        pao,
        base_deltas: split.base.clone(),
        honba_deltas: split.honba.clone(),
        kyotaku_deltas: split.kyotaku.clone(),
    })
}

/// Builds the draw body from plain data.
pub fn ryukyoku_event(
    players: &[PlayerState],
    kind: RyukyokuKind,
    tenpai: &[u8],
    settlement: &Settlement,
    split: &SettlementSplit,
) -> MatchlogEvent {
    let seats = players.len();
    let tenpai_mask: Vec<bool> = (0..seats as u8).map(|s| tenpai.contains(&s)).collect();
    // Tenpai players reveal their hands; noten players are `None`, which differs from an empty vec.
    let concealed = (0..seats)
        .map(|s| tenpai_mask[s].then(|| trs(&players[s].hand)))
        .collect();
    MatchlogEvent::Ryukyoku(Box::new(RyukyokuBody {
        kind,
        tenpai_mask,
        concealed,
        base_deltas: split.base.clone(),
        honba_deltas: split.honba.clone(),
        // Riichi sticks are not distributed on a draw and carry over.
        kyotaku_carry: settlement.next.kyotaku,
    }))
}

/// Winning tile of a ron.
///
/// Normally the last discard. Robbing a kan or nukidora is different: the robbed tile
/// is the declared tile, and `pending_robbery` has already been cleared by settlement
/// time (at the end of `apply_robbery_rons`), so it is taken from the last kan or
/// nukidora declaration in the event stream.
#[must_use]
pub fn ron_tile_4p(last_discard: Option<(u8, Tile)>, log: &[Event4p]) -> Option<Tile> {
    if let Some((_, t)) = last_discard {
        return Some(t);
    }
    log.iter().rev().find_map(|e| match e {
        Event4p::Kakan { pai, .. } => Some(*pai),
        // Kokushi robbing a closed kan: all four copies are the same kind, so any works.
        Event4p::Ankan { consumed, .. } => consumed.first().copied(),
        _ => None,
    })
}

/// 3-player version, which can also rob a nukidora.
#[must_use]
pub fn ron_tile_3p(last_discard: Option<(u8, Tile)>, log: &[Event3p]) -> Option<Tile> {
    if let Some((_, t)) = last_discard {
        return Some(t);
    }
    log.iter().rev().find_map(|e| match e {
        Event3p::Kakan { pai, .. } => Some(*pai),
        Event3p::Ankan { consumed, .. } => consumed.first().copied(),
        Event3p::Nukidora { pai, .. } => Some(*pai),
        _ => None,
    })
}
