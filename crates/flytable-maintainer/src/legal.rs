//! Legal action enumeration: the action list behind the decision interface and inference output.
//!
//! A rules computation (the list must match the rules), not a model action space.
//! Decision makers and inference return an index into this list.
//!
//! Shared by 4-player and 3-player through [`crate::view::SeatView`]. Variant
//! differences (chi, nukidora) are enforced by each table core; this lists the
//! actions the rules allow the seat right now.

use crate::action::{ReactionAction, RobberyKind, TurnAction};
use crate::player::PlayerState;
use crate::scoring::ScoringInput;
use crate::view::SeatView;
use flytable_core::agari;
use flytable_core::hand::TileCounts;
use flytable_core::meld::Meld;
use flytable_core::tile::Tile;

fn declared_kan_count(view: &SeatView) -> usize {
    view.me
        .melds
        .iter()
        .chain(view.others.iter().flat_map(|other| other.melds.iter()))
        .filter(|meld| meld.is_kan())
        .count()
}

/// Turn actions the rules allow on the seat's own turn.
///
/// Includes every discardable tile (by tsumogiri/hand origin, red fives separate),
/// riichi when possible, tsumo on a complete hand, closed kan with four copies,
/// nukidora with a North (3-player) and kyuushu kyuuhai.
pub fn legal_turn_actions(view: &SeatView) -> Vec<TurnAction> {
    let mut out = Vec::new();
    if !view.is_valid() || view.round_terminal {
        return out;
    }
    let hand = &view.me.hand;
    if hand.is_empty()
        || view.turn != view.me.seat
        || view.last_discard.is_some()
        || view.pending_robbery.is_some()
    {
        return out;
    }
    let open = view
        .me
        .melds
        .iter()
        .filter(|m| !matches!(m, flytable_core::meld::Meld::Nukidora { .. }))
        .count() as u8;
    // Only after a draw, or when a discard is due after chi/pon, does the hand have
    // 3n+2 tiles. The dealt 13 tiles and the 3n+1 state after a discard must return
    // nothing; otherwise the dealer could discard before the first draw.
    if hand.len() + usize::from(open) * 3 != 14 {
        return out;
    }
    let counts = TileCounts::from_tiles(hand.iter().copied());

    // Declarations (tsumo, kyuushu kyuuhai, nukidora, closed/added kan, riichi) are only
    // legal right after a draw (including a replacement draw). After chi/pon
    // (`drawn_tile == None`, see [`crate::view::SelfView`]) only discards are allowed.
    let post_draw = view.me.drawn_tile.is_some() || view.me.dealer_opening;

    // Tsumo must match the settlement check in `Board::apply_turn`, so a complete hand
    // with no yaku is never listed.
    if post_draw && is_agari_for_view(view, &counts, open) && can_tsumo_agari(view) {
        out.push(TurnAction::Tsumo);
    }

    if post_draw && view.kyuushu_kyuuhai_window && yaochuu_kind_count(hand) >= 9 {
        out.push(TurnAction::KyuushuKyuuhai);
    }

    // Nukidora (3-player only) needs a North in hand and a tile left for the
    // replacement draw. In 4-player North is a normal honor. The variant is taken from
    // the seat count (others + self).
    let seat_count = view.others.len() + 1;
    let north = "N".parse::<flytable_core::tile::Tile>().unwrap();
    let public_nuki_count = view
        .me
        .melds
        .iter()
        .chain(view.others.iter().flat_map(|other| other.melds.iter()))
        .filter(|meld| matches!(meld, Meld::Nukidora { .. }))
        .count();
    if post_draw
        && seat_count == 3
        && view.tiles_left > 0
        // 3-player has four Norths: at most four nukidora, plus at most four kans, for eight replacement tiles.
        && public_nuki_count < 4
        && hand.iter().any(|t| t.kind() == north.kind())
        && (!view.me.riichi
            || view
                .me
                .drawn_tile
                .is_some_and(|drawn| drawn.kind() == north.kind()))
    {
        out.push(TurnAction::Nukidora);
    }

    // No closed or added kan once the live wall is empty or four kans have been declared.
    if post_draw && view.tiles_left > 0 && declared_kan_count(view) < 4 {
        for k in 0..34usize {
            if counts.count(k) == 4 {
                let tile = unsafe { flytable_core::tile::Tile::from_id_unchecked(k as u8) };
                if view.me.riichi && !can_riichi_ankan(view, &counts, open, tile) {
                    continue;
                }
                out.push(TurnAction::Ankan { tile });
            }
        }

        // Added kan: a pon of the kind plus the drawn fourth tile (same draw and wall conditions).
        for m in &view.me.melds {
            if let flytable_core::meld::Meld::Pon { tile, .. } = m {
                let k = tile.kind();
                if let Some(&added) = hand.iter().find(|t| t.kind() == k) {
                    out.push(TurnAction::Kakan { tile: added });
                }
            }
        }
    }

    let mut discard_candidates = discard_candidates(hand, view.me.drawn_tile);
    discard_candidates.retain(|(tile, _)| !view.me.kuikae_forbidden.contains(&tile.kind()));
    if view.me.riichi {
        discard_candidates.retain(|(_, tsumogiri)| *tsumogiri);
    }

    // Discardable tiles, deduplicated by exact tile and tsumogiri/hand origin.
    for &(tile, tsumogiri) in &discard_candidates {
        if view.me.dealer_opening {
            out.push(TurnAction::DealerOpeningDiscard { tile });
        } else {
            out.push(TurnAction::Discard { tile, tsumogiri });
        }
    }

    // Riichi: closed hand, tenpai after some discard, right after a draw.
    let has_riichi_stick = view
        .scores
        .get(view.me.seat as usize)
        .is_some_and(|score| *score >= 1_000);
    // Riichi requires a remaining draw after the declaration on both Tenhou and Mahjong
    // Soul. `no_draw_riichi = true` is only for custom rooms.
    let has_future_draw =
        view.rule_profile.allows_no_draw_riichi() || view.tiles_left >= seat_count as u32;
    if post_draw
        && view.me.melds.iter().all(|m| !m.breaks_menzen())
        && !view.me.riichi
        && has_riichi_stick
        && has_future_draw
    {
        for &(tile, tsumogiri) in &discard_candidates {
            let mut c = counts;
            c.sub(tile, 1);
            if crate::calc::shanten_after_for(&c, open, seat_count == 3)
                == flytable_core::shanten::TENPAI
            {
                if view.me.dealer_opening {
                    out.push(TurnAction::DealerOpeningRiichi { tile });
                } else {
                    out.push(TurnAction::Riichi { tile, tsumogiri });
                }
            }
        }
    }

    out
}

/// Discard furiten: any winning tile of the current wait is in the own discards.
pub fn discard_furiten(view: &SeatView) -> bool {
    if !view.is_valid() || view.round_terminal {
        return true;
    }
    let counts = TileCounts::from_tiles(view.me.hand.iter().copied());
    let open = view
        .me
        .melds
        .iter()
        .filter(|m| !matches!(m, Meld::Nukidora { .. }))
        .count() as u8;
    let waits = winning_tiles_for_view(view, &counts, open);
    if waits.is_empty() {
        return false;
    }
    view.me.discards.iter().any(|d| waits.contains(&d.kind()))
}

/// Alias of [`discard_furiten`] kept for existing callers.
pub fn permanent_furiten(view: &SeatView) -> bool {
    if !view.is_valid() || view.round_terminal {
        return true;
    }
    discard_furiten(view)
}

/// Any furiten (discard, temporary or riichi). Ron enumeration and adjudication both
/// go through this, preferring to omit a ron over listing an illegal one.
pub fn any_furiten(view: &SeatView) -> bool {
    if !view.is_valid() || view.round_terminal {
        return true;
    }
    view.me.temporary_furiten || view.me.riichi_furiten || discard_furiten(view)
}

/// Legal responses to `last_discard` (ron / pon / open kan / chi).
///
/// Same rules as the authoritative `Board4p::legal_reactions`, but on the observed
/// [`SeatView`], so a mirror can enumerate responses itself without relying on a
/// platform's list. Reuses `pon_pair_combos` / `chi_combos`, so red five choices
/// match the authoritative implementation.
///
/// Ron requires a complete hand, no furiten and a yaku; after riichi only ron remains.
pub fn legal_reactions(view: &SeatView) -> Vec<ReactionAction> {
    let mut out = Vec::new();
    if !view.is_valid() || view.round_terminal {
        return out;
    }
    if let Some(pending) = view.pending_robbery {
        if pending.actor != view.me.seat
            && robbery_ron_allowed(view, pending.kind)
            && can_ron_agari(
                view,
                pending.tile,
                matches!(pending.kind, RobberyKind::Kakan),
            )
        {
            out.push(ReactionAction::Ron);
        }
        return out;
    }
    let Some((from, tile)) = view.last_discard else {
        return out;
    };
    if from == view.me.seat {
        return out;
    }
    // Ron: complete hand, no furiten, and a yaku (checked through `scoring::settle`,
    // like `can_tsumo_agari`). After riichi, no chi, pon or open kan.
    if can_ron_agari(view, tile, false) {
        out.push(ReactionAction::Ron);
    }
    // No chi, pon or open kan on the houtei discard; ron is still possible.
    if view.tiles_left == 0 {
        return out;
    }
    // The same for the discard that completes a pending four-kan draw: ron only.
    if view.suukaikan_pending {
        return out;
    }
    if view.me.riichi {
        return out;
    }
    let k = tile.kind();
    let same: Vec<Tile> = view
        .me
        .hand
        .iter()
        .copied()
        .filter(|t| t.kind() == k)
        .collect();
    // With kuikae forbidden, a call must leave at least one discardable tile, otherwise
    // the call is invalid (see `tileset::call_leaves_legal_discard`).
    let kuikae_on = view.rule_profile.kuikae_forbidden();
    if same.len() >= 2 {
        for consumed in crate::tileset::pon_pair_combos(&same) {
            if kuikae_on
                && !crate::tileset::call_leaves_legal_discard(&view.me.hand, &consumed, &[k])
            {
                continue;
            }
            out.push(ReactionAction::Pon { consumed });
        }
        if same.len() >= 3 && declared_kan_count(view) < 4 {
            out.push(ReactionAction::Daiminkan);
        }
    }
    // Chi: 4-player only, from the player to the left (`(from + 1) % 4 == seat`), number tiles.
    let seat_count = view.others.len() + 1;
    if seat_count == 4 && (from + 1) % 4 == view.me.seat && tile.rank().is_some() {
        for combo in crate::tileset::chi_combos(&view.me.hand, tile) {
            if kuikae_on {
                let forbidden = crate::tileset::kuikae_forbidden_after_chi(tile, combo);
                if !crate::tileset::call_leaves_legal_discard(&view.me.hand, &combo, &forbidden) {
                    continue;
                }
            }
            out.push(ReactionAction::Chi { consumed: combo });
        }
    }
    out
}

fn discard_candidates(hand: &[Tile], drawn_tile: Option<Tile>) -> Vec<(Tile, bool)> {
    let mut out = Vec::new();
    if hand.is_empty() {
        return out;
    }

    let drawn_is_last = drawn_tile.is_some() && hand.last().copied() == drawn_tile;
    let hand_cut_end = if drawn_is_last {
        hand.len() - 1
    } else {
        hand.len()
    };
    let mut seen_hand_cut = [false; flytable_core::tile::NUM_IDS];
    for &tile in &hand[..hand_cut_end] {
        let id = tile.id() as usize;
        if id < seen_hand_cut.len() && !seen_hand_cut[id] {
            seen_hand_cut[id] = true;
            out.push((tile, false));
        }
    }
    if drawn_is_last {
        out.push((drawn_tile.expect("drawn_is_last implies Some"), true));
    }
    out
}

fn can_riichi_ankan(view: &SeatView, counts: &TileCounts, open: u8, tile: Tile) -> bool {
    match view.rule_profile.riichi_ankan_rule() {
        flytable_core::rules::RiichiAnkanRule::Forbidden => return false,
        flytable_core::rules::RiichiAnkanRule::DrawnFourthAndWaitsUnchanged => {}
    }
    let Some(last) = view.me.hand.last().copied() else {
        return false;
    };
    if last.kind() != tile.kind() {
        return false;
    }

    let mut before_draw = *counts;
    before_draw.sub(last, 1);
    if before_draw.count(tile.kind()) != 3 {
        return false;
    }

    let waits_before = winning_tiles_for_view(view, &before_draw, open);
    if waits_before.is_empty() {
        return false;
    }

    let mut after_kan = *counts;
    after_kan.sub(tile, 4);
    // Closed kan after riichi: drawn fourth tile and unchanged waits are enough; a
    // possible sequence reading does not forbid it (Tenhou and Mahjong Soul).
    winning_tiles_for_view(view, &after_kan, open + 1) == waits_before
}

fn yaochuu_kind_count(hand: &[Tile]) -> usize {
    let mut seen = [false; 34];
    for &tile in hand {
        if tile.is_yaochuu() {
            seen[tile.kind()] = true;
        }
    }
    seen.into_iter().filter(|seen| *seen).count()
}

fn can_tsumo_agari(view: &SeatView) -> bool {
    if view.me.dealer_opening {
        let mut seen = [false; flytable_core::tile::NUM_IDS];
        return view.me.hand.iter().copied().any(|tile| {
            let id = tile.id() as usize;
            id < seen.len()
                && !std::mem::replace(&mut seen[id], true)
                && can_tsumo_with(view, tile, true)
        });
    }
    view.me
        .hand
        .last()
        .copied()
        .is_some_and(|tile| can_tsumo_with(view, tile, false))
}

fn can_tsumo_with(view: &SeatView, win_tile: Tile, tenhou: bool) -> bool {
    let mut player = PlayerState::new(view.me.seat, [Tile::default(); 13]);
    player.hand = view.me.hand.clone();
    player.melds = view.me.melds.clone();
    player.discards = view.me.discards.clone();
    player.riichi = view.me.riichi;
    player.menzen = view.me.melds.iter().all(|m| !m.breaks_menzen());

    let seat_count = view.others.len() + 1;
    let input = ScoringInput {
        rule_profile: view.rule_profile,
        player: &player,
        win_tile,
        is_tsumo: true,
        bakaze: view.bakaze,
        jikaze: seat_wind(view.oya, view.me.seat, seat_count),
        is_oya: view.me.seat == view.oya,
        dora_indicators: &view.dora_indicators,
        ura_indicators: &[],
        riichi: player.riichi,
        double_riichi: false,
        ippatsu: view.me.ippatsu,
        haitei: view.tiles_left == 0,
        houtei: false,
        rinshan: view.last_draw_was_rinshan,
        chankan: false,
        tenhou,
        chiihou: false,
        nuki_count: view
            .me
            .melds
            .iter()
            .filter(|m| matches!(m, Meld::Nukidora { .. }))
            .count() as u8,
        is_sanma: seat_count == 3,
    };
    crate::scoring::settle(&input).is_some()
}

/// Whether the rules allow robbing this kind of declaration.
fn robbery_ron_allowed(view: &SeatView, kind: RobberyKind) -> bool {
    match kind {
        RobberyKind::Kakan => true,
        RobberyKind::Nukidora => view.rule_profile.allows_nukidora_ron(),
        RobberyKind::Ankan => view.rule_profile.allows_kokushi_ankan_ron(),
    }
}

/// Whether the seat can ron `win_tile`: no furiten, complete hand, and a yaku.
///
/// Uses the same check as [`can_tsumo_agari`] (through [`crate::scoring::settle`]).
/// Situational flags such as ippatsu are false in a passive view, so a ron that
/// relies only on them is not enumerated and is left to the platform.
fn can_ron_agari(view: &SeatView, win_tile: Tile, chankan: bool) -> bool {
    if any_furiten(view) {
        return false;
    }
    let open = view
        .me
        .melds
        .iter()
        .filter(|m| !matches!(m, Meld::Nukidora { .. }))
        .count() as u8;
    let mut with = TileCounts::from_tiles(view.me.hand.iter().copied());
    with.add(win_tile, 1);
    if !is_agari_for_view(view, &with, open) {
        return false;
    }
    if matches!(
        view.pending_robbery.map(|pending| pending.kind),
        Some(RobberyKind::Ankan)
    ) && !flytable_core::decompose::is_kokushi(with.raw())
    {
        return false;
    }

    let mut player = PlayerState::new(view.me.seat, [Tile::default(); 13]);
    player.hand = view.me.hand.clone();
    player.hand.push(win_tile);
    player.melds = view.me.melds.clone();
    player.discards = view.me.discards.clone();
    player.riichi = view.me.riichi;
    player.menzen = view.me.melds.iter().all(|m| !m.breaks_menzen());

    let seat_count = view.others.len() + 1;
    let input = ScoringInput {
        rule_profile: view.rule_profile,
        player: &player,
        win_tile,
        is_tsumo: false,
        bakaze: view.bakaze,
        jikaze: seat_wind(view.oya, view.me.seat, seat_count),
        is_oya: view.me.seat == view.oya,
        dora_indicators: &view.dora_indicators,
        ura_indicators: &[],
        riichi: player.riichi,
        double_riichi: false,
        ippatsu: view.me.ippatsu,
        haitei: false,
        houtei: view.tiles_left == 0,
        rinshan: false,
        chankan,
        tenhou: false,
        chiihou: false,
        nuki_count: view
            .me
            .melds
            .iter()
            .filter(|m| matches!(m, Meld::Nukidora { .. }))
            .count() as u8,
        is_sanma: seat_count == 3,
    };
    crate::scoring::settle(&input).is_some()
}

fn is_agari_for_view(view: &SeatView, counts: &TileCounts, open: u8) -> bool {
    if view.others.len() + 1 == 3 {
        agari::is_agari_3p(counts, open)
    } else {
        agari::is_agari(counts, open)
    }
}

fn winning_tiles_for_view(view: &SeatView, counts: &TileCounts, open: u8) -> Vec<usize> {
    if view.others.len() + 1 == 3 {
        agari::winning_tiles_3p(counts, open)
    } else {
        agari::winning_tiles(counts, open)
    }
}

fn seat_wind(oya: u8, seat: u8, seat_count: usize) -> Tile {
    let seats = seat_count as u8;
    let offset = (seat + seats - oya) % seats;
    unsafe { Tile::from_id_unchecked(27 + offset) }
}
