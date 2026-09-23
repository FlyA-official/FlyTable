//! Hand-to-hand progression and match-level settlement (shared by 4-player and 3-player).
//!
//! The boards ([`crate::board4p`] / [`crate::board3p`]) only play out a hand and decide
//! who wins or whether it is a draw. How scores change afterwards, whether the dealer
//! rotates and whether the match is over are match-level rules, and they all live
//! here.
//!
//! - Turns a [`KyokuOutcome`] or draw tenpai set plus riichi sticks into per-seat
//!   deltas, including honba, riichi sticks and noten penalties.
//! - Decides the next hand's round wind, hand number, dealer, honba and riichi sticks
//!   (rotation, renchan, extension, game end).
//! - Decides whether the match is over (tonpuusen / hanchan, dealer renchan in the
//!   last hand, busting below zero).
//!
//! No observation or action spaces, no external processes, no hidden seat state.

use flytable_core::rules::RiichiRuleProfile;

/// Match-level state at the start of a hand (round wind, hand number, dealer, honba,
/// riichi sticks, scores).
///
/// Round wind is encoded as East = 0, South = 1, West = 2, North = 3; `kyoku` starts
/// at 1 (East 1 = (0, 1)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundState {
    /// Round wind: 0 = East, 1 = South, 2 = West, 3 = North.
    pub bakaze: u8,
    /// Hand number (1-based): East 1, East 2, ...; South 1 is (1, 1).
    pub kyoku: u8,
    pub honba: u8,
    /// Riichi sticks on the table, 1000 points each, all taken by the winner.
    pub kyotaku: u8,
    /// Current dealer seat.
    pub oya: u8,
    /// Scores per seat (length = seat count).
    pub scores: Vec<i32>,
}

impl RoundState {
    /// Start of a match: East 1, dealer seat 0, no honba or riichi sticks.
    pub fn new_match(seats: usize, start_score: i32) -> Self {
        Self {
            bakaze: 0,
            kyoku: 1,
            honba: 0,
            kyotaku: 0,
            oya: 0,
            scores: vec![start_score; seats],
        }
    }

    fn seats(&self) -> usize {
        self.scores.len()
    }
}

/// Match length (decides the last hand).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchLength {
    /// Tonpuusen: the East round only (East 4 in 4-player, East 3 in 3-player).
    East,
    /// Hanchan: through the South round (South 4 in 4-player, South 3 in 3-player).
    Half,
}

/// Payment description of a win, independent of seat count (derived from the
/// boards' [`flytable_core::score::ScoreResult`]). Only who pays whom and how much.
#[derive(Debug, Clone, Copy)]
pub enum HoraPayment {
    /// Ron: `from` pays `ron` to `winner`.
    Ron { winner: u8, from: u8, ron: i32 },
    /// Ron with pao: `liable` and the discarder each pay half the base points; the liable
    /// player also pays all honba.
    PaoRon {
        winner: u8,
        from: u8,
        liable: u8,
        ron: i32,
    },
    /// Tsumo: the dealer pays `tsumo_oya`, other non-dealers `tsumo_ko` each.
    Tsumo {
        winner: u8,
        tsumo_oya: i32,
        tsumo_ko: i32,
    },
    /// Tsumo with pao: the liable player pays the full ron value and all honba.
    PaoTsumo { winner: u8, liable: u8, ron: i32 },
    /// Tsumo with pao on part of a composite yakuman (Mahjong Soul): the liable player
    /// alone pays the covered part and all honba; the uncovered part is split as a normal
    /// tsumo (the liable player pays their share too). Unreachable under Tenhou rules,
    /// where yakuman do not stack.
    PaoTsumoPartial {
        winner: u8,
        liable: u8,
        /// Tsumo total of the covered yakuman part (paid by the liable player alone).
        liable_part: i32,
        /// Uncovered part: amount paid by the dealer.
        tsumo_oya: i32,
        /// Uncovered part: amount paid by each non-dealer.
        tsumo_ko: i32,
    },
    /// Ron with pao on part of a composite yakuman, by analogy with the tsumo case: the
    /// liable player and the discarder split the covered part, the discarder pays the
    /// uncovered part in full, and honba go to the liable player.
    PaoRonPartial {
        winner: u8,
        from: u8,
        liable: u8,
        /// Paid by the liable player (half of the covered part).
        liable_pay: i32,
        /// Paid by the discarder (the other half of the covered part plus all of the uncovered part).
        from_pay: i32,
    },
}

/// Payment derived from a `Hora` (shared by 4-player and 3-player; the single source
/// for live play, match log archives and replay). `tsumo_oya` / `tsumo_ko` are passed
/// through since settlement decides who pays the dealer amount from `state.oya`.
pub fn hora_payment(
    winner: u8,
    from: Option<u8>,
    score: &flytable_core::score::FullScore,
) -> HoraPayment {
    match from {
        Some(f) => HoraPayment::Ron {
            winner,
            from: f,
            ron: score.score.ron,
        },
        None => HoraPayment::Tsumo {
            winner,
            tsumo_oya: score.score.tsumo_oya,
            tsumo_ko: score.score.tsumo_ko,
        },
    }
}

/// The only constructor of pao payments. `liability = (liable seat, covered yakuman
/// multiplier)`. Composite yakuman with partial pao are split per part (Mahjong
/// Soul); otherwise the liable player pays in full. Under Tenhou the total multiplier
/// is always 1, so the split branch is never reached.
pub fn hora_payment_with_pao(
    winner: u8,
    from: Option<u8>,
    score: &flytable_core::score::FullScore,
    liability: Option<(u8, u8)>,
    seats: usize,
) -> HoraPayment {
    let Some((liable, covered)) = liability else {
        return hora_payment(winner, from, score);
    };
    let n = i32::from(score.score.yakuman);
    let p = i32::from(covered);
    if n > p && p > 0 {
        return match from {
            None => {
                let total = score.score.tsumo_total(seats);
                debug_assert_eq!(
                    total % n,
                    0,
                    "yakuman tsumo total must divide by the multiplier"
                );
                HoraPayment::PaoTsumoPartial {
                    winner,
                    liable,
                    liable_part: total / n * p,
                    tsumo_oya: score.score.tsumo_oya / n * (n - p),
                    tsumo_ko: score.score.tsumo_ko / n * (n - p),
                }
            }
            Some(from) => {
                // Ron side, by analogy with the tsumo case.
                let ron = score.score.ron;
                let covered_part = ron / n * p;
                HoraPayment::PaoRonPartial {
                    winner,
                    from,
                    liable,
                    liable_pay: covered_part / 2,
                    from_pay: covered_part - covered_part / 2 + (ron - covered_part),
                }
            }
        };
    }
    match from {
        Some(from) => HoraPayment::PaoRon {
            winner,
            from,
            liable,
            ron: score.score.ron,
        },
        // `ScoreResult.ron` is 0 for a tsumo, so the pao tsumo total comes from the tsumo amounts.
        None => HoraPayment::PaoTsumo {
            winner,
            liable,
            ron: score.score.tsumo_total(seats),
        },
    }
}

/// Result of settling a hand: per-seat deltas, the next hand's state, and whether the match ended.
#[derive(Debug, Clone)]
pub struct Settlement {
    /// Per-seat deltas for this hand (including honba, riichi sticks and penalties).
    pub deltas: Vec<i32>,
    /// Scores after settlement, before the next hand.
    pub scores: Vec<i32>,
    /// State of the next hand (the final state if `ended`).
    pub next: RoundState,
    /// Whether the match is over.
    pub ended: bool,
}

/// Settlement split into base points, honba and riichi sticks.
///
/// The match log keeps the three parts separate because review and statistics tools
/// need different ones. Their sum always equals [`Settlement::deltas`].
///
/// `base` and `honba` are each zero-sum. `kyotaku` is not: sticks go from the table
/// pool to the winner, so its sum equals the amount collected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettlementSplit {
    pub base: Vec<i32>,
    pub honba: Vec<i32>,
    pub kyotaku: Vec<i32>,
}

impl SettlementSplit {
    /// The three parts should add up to the combined deltas (self-check).
    #[must_use]
    pub fn total(&self) -> Vec<i32> {
        (0..self.base.len())
            .map(|i| self.base[i] + self.honba[i] + self.kyotaku[i])
            .collect()
    }
}

/// Per-winner split of a multiple ron.
///
/// A single winner only needs [`split_settlement`]. With multiple ron the match log
/// emits one win event per winner, while [`Settlement`] only has the combined deltas.
///
/// No scoring logic is reimplemented:
/// - Per-winner base points: settle each winner as if they were the only one. Payments
///   in a multiple ron are independent (each collects from the discarder, or splits
///   with the liable player under pao), so the sum equals the combined base
///   (`debug_assert`).
/// - Honba and riichi sticks: only for the first winner (`first`), extracted with the
///   same difference method as `split_settlement`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiHoraSplit {
    /// Base points per winner, ordered `[first] + additional` (index 0 is the head bump winner). Each zero-sum.
    pub per_winner_base: Vec<Vec<i32>>,
    /// Honba, first winner only. Zero-sum.
    pub honba: Vec<i32>,
    /// Riichi sticks, first winner only. Not zero-sum (from the table pool to the winner).
    pub kyotaku: Vec<i32>,
}

/// Splits a multiple ron. See [`MultiHoraSplit`].
#[must_use]
pub fn split_multi_hora(
    state: &RoundState,
    length: MatchLength,
    first: HoraPayment,
    additional: &[HoraPayment],
    profile: RiichiRuleProfile,
) -> MultiHoraSplit {
    let bare = with_counters(state, 0, 0);
    let per_winner_base: Vec<Vec<i32>> = std::iter::once(&first)
        .chain(additional.iter())
        .map(|pay| settle_multi_hora_with_profile(&bare, length, *pay, &[], profile).deltas)
        .collect();

    let all_bare = settle_multi_hora_with_profile(&bare, length, first, additional, profile).deltas;
    let with_honba = settle_multi_hora_with_profile(
        &with_counters(state, state.honba, 0),
        length,
        first,
        additional,
        profile,
    )
    .deltas;
    let full = settle_multi_hora_with_profile(state, length, first, additional, profile).deltas;

    debug_assert!(
        {
            let mut sum = vec![0i32; all_bare.len()];
            for w in &per_winner_base {
                for (i, v) in w.iter().enumerate() {
                    sum[i] += v;
                }
            }
            sum == all_bare
        },
        "per-winner base points must add up to the combined base; payments are not independent"
    );

    MultiHoraSplit {
        per_winner_base,
        honba: diff(&with_honba, &all_bare),
        kyotaku: diff(&full, &with_honba),
    }
}

fn with_counters(state: &RoundState, honba: u8, kyotaku: u8) -> RoundState {
    RoundState {
        honba,
        kyotaku,
        ..state.clone()
    }
}

fn diff(a: &[i32], b: &[i32]) -> Vec<i32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// Splits any settlement into three parts.
///
/// No scoring logic is reimplemented: the same `settle` closure is called three times
/// (without honba or sticks, with honba only, and as is) and the parts are taken as
/// differences.
pub fn split_settlement<F>(state: &RoundState, settle: F) -> SettlementSplit
where
    F: Fn(&RoundState) -> Settlement,
{
    let bare = settle(&with_counters(state, 0, 0)).deltas;
    let with_honba = settle(&with_counters(state, state.honba, 0)).deltas;
    let full = settle(state).deltas;
    SettlementSplit {
        honba: diff(&with_honba, &bare),
        kyotaku: diff(&full, &with_honba),
        base: bare,
    }
}

/// Honba bonus on ron: 100 points per paying player, so 300 in 4-player and 200 in
/// 3-player (Tenhou: 4-player South 300, 3-player South 200). Using the 4-player
/// constant in 3-player would overcharge 100 per honba.
const fn honba_ron(seats: usize) -> i32 {
    HONBA_TSUMO_EACH * (seats as i32 - 1)
}
const HONBA_TSUMO_EACH: i32 = 100;
/// One riichi stick is 1000 points.
const RIICHI_STICK: i32 = 1000;
/// Noten penalty pool: 1000 per noten seat baseline, 3000 in 4-player and 2000 in
/// 3-player.
///
/// 3-player: one tenpai gets +2000 and each noten pays 1000; two tenpai get +1000
/// each and the noten pays 2000; zero or three tenpai means no payment.
const fn noten_pool(seats: usize) -> i32 {
    1000 * (seats as i32 - 1)
}

/// Settles a hand that ended in a win: deltas (including honba and riichi sticks)
/// and dealer rotation, renchan or game end.
///
/// `state` is the state at the start of the hand, `length` the match length, `pay`
/// the payment. Riichi sticks (including those accepted this hand, already in
/// `state.kyotaku`) all go to the winner.
pub fn settle_hora(state: &RoundState, length: MatchLength, pay: HoraPayment) -> Settlement {
    settle_hora_with_profile(state, length, pay, RiichiRuleProfile::tenhou())
}

/// Settles a win with an explicit platform or room profile.
pub fn settle_hora_with_profile(
    state: &RoundState,
    length: MatchLength,
    pay: HoraPayment,
    profile: RiichiRuleProfile,
) -> Settlement {
    settle_multi_hora_with_profile(state, length, pay, &[], profile)
}

/// Settles multiple wins on the same discard.
///
/// `first` must be the winner closest to the discarder; only they get honba and
/// riichi sticks. Each winner's base ron points are paid by the discarder. The dealer
/// keeps the seat if among the winners. The separate `first` parameter guarantees at
/// least one winner.
pub fn settle_multi_hora(
    state: &RoundState,
    length: MatchLength,
    first: HoraPayment,
    additional: &[HoraPayment],
) -> Settlement {
    settle_multi_hora_with_profile(
        state,
        length,
        first,
        additional,
        RiichiRuleProfile::tenhou(),
    )
}

/// Settles multiple wins with an explicit platform or room profile.
pub fn settle_multi_hora_with_profile(
    state: &RoundState,
    length: MatchLength,
    first: HoraPayment,
    additional: &[HoraPayment],
    profile: RiichiRuleProfile,
) -> Settlement {
    let seats = state.seats();
    let mut deltas = vec![0i32; seats];
    let honba = state.honba as i32;
    let mut winners = Vec::with_capacity(additional.len() + 1);
    for (index, pay) in std::iter::once(&first).chain(additional.iter()).enumerate() {
        let winner = match *pay {
            HoraPayment::Ron { winner, from, ron } => {
                let bonus = if index == 0 {
                    honba * honba_ron(seats)
                } else {
                    0
                };
                deltas[from as usize] -= ron + bonus;
                deltas[winner as usize] += ron + bonus;
                winner
            }
            HoraPayment::PaoRon {
                winner,
                from,
                liable,
                ron,
            } => {
                let bonus = if index == 0 {
                    honba * HONBA_TSUMO_EACH * (seats as i32 - 1)
                } else {
                    0
                };
                let half = ron / 2;
                deltas[liable as usize] -= half + bonus;
                deltas[from as usize] -= half;
                deltas[winner as usize] += ron + bonus;
                winner
            }
            HoraPayment::Tsumo {
                winner,
                tsumo_oya,
                tsumo_ko,
            } => {
                let bonus = if index == 0 {
                    honba * HONBA_TSUMO_EACH
                } else {
                    0
                };
                for s in 0..seats as u8 {
                    if s == winner {
                        continue;
                    }
                    let base = if s == state.oya { tsumo_oya } else { tsumo_ko };
                    let pay_amt = base + bonus;
                    deltas[s as usize] -= pay_amt;
                    deltas[winner as usize] += pay_amt;
                }
                winner
            }
            HoraPayment::PaoTsumo {
                winner,
                liable,
                ron,
            } => {
                let bonus = if index == 0 {
                    honba * HONBA_TSUMO_EACH * (seats as i32 - 1)
                } else {
                    0
                };
                deltas[liable as usize] -= ron + bonus;
                deltas[winner as usize] += ron + bonus;
                winner
            }
            HoraPayment::PaoTsumoPartial {
                winner,
                liable,
                liable_part,
                tsumo_oya,
                tsumo_ko,
            } => {
                let bonus = if index == 0 {
                    honba * HONBA_TSUMO_EACH * (seats as i32 - 1)
                } else {
                    0
                };
                deltas[liable as usize] -= liable_part + bonus;
                deltas[winner as usize] += liable_part + bonus;
                for s in 0..seats as u8 {
                    if s == winner {
                        continue;
                    }
                    let base = if s == state.oya { tsumo_oya } else { tsumo_ko };
                    deltas[s as usize] -= base;
                    deltas[winner as usize] += base;
                }
                winner
            }
            HoraPayment::PaoRonPartial {
                winner,
                from,
                liable,
                liable_pay,
                from_pay,
            } => {
                let bonus = if index == 0 {
                    honba * HONBA_TSUMO_EACH * (seats as i32 - 1)
                } else {
                    0
                };
                deltas[liable as usize] -= liable_pay + bonus;
                deltas[from as usize] -= from_pay;
                deltas[winner as usize] += liable_pay + from_pay + bonus;
                winner
            }
        };
        winners.push(winner);
    }
    // Riichi sticks go only to the first winner.
    deltas[winners[0] as usize] += state.kyotaku as i32 * RIICHI_STICK;

    let scores: Vec<i32> = (0..seats).map(|i| state.scores[i] + deltas[i]).collect();

    // Renchan if any winner is the dealer.
    let oya_won = winners.contains(&state.oya);
    let next = advance_after_kyoku(state, length, oya_won, &scores);
    let ended = is_match_end(
        state,
        length,
        oya_won,
        DealerKeepReason::Hora,
        &scores,
        profile,
    );

    Settlement {
        deltas,
        scores,
        next,
        ended,
    }
}

/// Settles an exhaustive draw: noten penalties, riichi sticks stay on the table,
/// honba + 1; renchan if the dealer is tenpai, otherwise rotation.
pub fn settle_ryukyoku(state: &RoundState, length: MatchLength, tenpai: &[u8]) -> Settlement {
    settle_ryukyoku_with_profile(state, length, tenpai, RiichiRuleProfile::tenhou())
}

/// Settles an exhaustive draw with an explicit platform or room profile.
pub fn settle_ryukyoku_with_profile(
    state: &RoundState,
    length: MatchLength,
    tenpai: &[u8],
    profile: RiichiRuleProfile,
) -> Settlement {
    let seats = state.seats();
    let mut deltas = vec![0i32; seats];
    let n_tenpai = tenpai.len();
    // No penalty when nobody or everybody is tenpai.
    if n_tenpai != 0 && n_tenpai != seats {
        let pool = noten_pool(seats);
        let per_tenpai = pool / n_tenpai as i32;
        let per_noten = pool / (seats - n_tenpai) as i32;
        for s in 0..seats as u8 {
            if tenpai.contains(&s) {
                deltas[s as usize] += per_tenpai;
            } else {
                deltas[s as usize] -= per_noten;
            }
        }
    }
    let mut scores: Vec<i32> = (0..seats).map(|i| state.scores[i] + deltas[i]).collect();

    // Renchan if the dealer is tenpai. Riichi sticks carry over.
    let oya_tenpai = tenpai.contains(&state.oya);
    let mut next = advance_after_ryukyoku(state, length, oya_tenpai, &scores);
    let ended = is_match_end(
        state,
        length,
        oya_tenpai,
        DealerKeepReason::Tenpai,
        &scores,
        profile,
    );
    if ended {
        award_terminal_kyotaku(state, &mut deltas, &mut scores, &mut next);
    }

    Settlement {
        deltas,
        scores,
        next,
        ended,
    }
}

/// Settles nagashi mangan. Each eligible player collects mangan tsumo base points
/// independently; no honba or riichi sticks.
///
/// Tenhou treats it as a replacement for the exhaustive draw payments: renchan still
/// depends on formal tenpai, riichi sticks stay, honba + 1.
pub fn settle_nagashi_mangan(
    state: &RoundState,
    length: MatchLength,
    winners: &[u8],
    tenpai: &[u8],
) -> Settlement {
    settle_nagashi_mangan_with_profile(state, length, winners, tenpai, RiichiRuleProfile::tenhou())
}

/// Settles nagashi mangan with an explicit platform or room profile.
pub fn settle_nagashi_mangan_with_profile(
    state: &RoundState,
    length: MatchLength,
    winners: &[u8],
    tenpai: &[u8],
    profile: RiichiRuleProfile,
) -> Settlement {
    let seats = state.seats();
    let mut deltas = vec![0i32; seats];
    for &winner in winners {
        if winner as usize >= seats {
            continue;
        }
        for payer in 0..seats as u8 {
            if payer == winner {
                continue;
            }
            let amount = if winner == state.oya || payer == state.oya {
                4_000
            } else {
                2_000
            };
            deltas[payer as usize] -= amount;
            deltas[winner as usize] += amount;
        }
    }
    let mut scores: Vec<i32> = (0..seats).map(|i| state.scores[i] + deltas[i]).collect();
    let oya_tenpai = tenpai.contains(&state.oya);
    let mut next = advance_after_ryukyoku(state, length, oya_tenpai, &scores);
    let ended = is_match_end(
        state,
        length,
        oya_tenpai,
        DealerKeepReason::Tenpai,
        &scores,
        profile,
    );
    if ended {
        award_terminal_kyotaku(state, &mut deltas, &mut scores, &mut next);
    }
    Settlement {
        deltas,
        scores,
        next,
        ended,
    }
}

/// Settles an abortive draw (kyuushu kyuuhai and similar): no payments, riichi sticks
/// stay, same dealer, honba + 1.
pub fn settle_abortive_ryukyoku(state: &RoundState, length: MatchLength) -> Settlement {
    settle_abortive_ryukyoku_with_profile(state, length, RiichiRuleProfile::tenhou())
}

/// Settles an abortive draw with an explicit platform or room profile.
pub fn settle_abortive_ryukyoku_with_profile(
    state: &RoundState,
    length: MatchLength,
    profile: RiichiRuleProfile,
) -> Settlement {
    let seats = state.seats();
    let mut deltas = vec![0i32; seats];
    let mut scores = state.scores.clone();
    let mut next = advance_after_ryukyoku(state, length, true, &scores);
    // Abortive draws are always renchan (Tenhou). They do not trigger agariyame or
    // tenpaiyame and are not capped by the extension limit (which only applies on
    // rotation, see `is_match_end`); only busting can end the match.
    let ended = anyone_busted(&scores, profile);
    if ended {
        award_terminal_kyotaku(state, &mut deltas, &mut scores, &mut next);
    }

    Settlement {
        deltas,
        scores,
        next,
        ended,
    }
}

/// When the match ends without a win, the riichi sticks on the table go to the
/// current leader. The end check itself does not count them.
fn award_terminal_kyotaku(
    state: &RoundState,
    deltas: &mut [i32],
    scores: &mut [i32],
    next: &mut RoundState,
) {
    if state.kyotaku == 0 || scores.is_empty() {
        return;
    }
    let winner = rankings(scores)[0] as usize;
    let award = state.kyotaku as i32 * RIICHI_STICK;
    deltas[winner] += award;
    scores[winner] += award;
    next.scores = scores.to_vec();
    next.kyotaku = 0;
}

/// Scheduled last hand (last East hand in tonpuusen, last South hand in hanchan).
fn is_scheduled_final_kyoku(state: &RoundState, length: MatchLength) -> bool {
    let seats = state.seats() as u8;
    state.bakaze == scheduled_last_bakaze(length) && state.kyoku == seats
}

fn scheduled_last_bakaze(length: MatchLength) -> u8 {
    match length {
        MatchLength::East => 0, // tonpuusen: end of East
        MatchLength::Half => 1, // hanchan: end of South
    }
}

/// With busting enabled, the match ends when a score is below 0 (exactly 0 continues).
fn anyone_busted(scores: &[i32], profile: RiichiRuleProfile) -> bool {
    profile.bust_below_zero() && scores.iter().any(|&s| s < 0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DealerKeepReason {
    Hora,
    Tenpai,
}

/// Match end check driven by the platform or room profile. `oya_kept` means the dealer
/// won, or was tenpai at a draw.
///
/// - Busting below zero (0 continues).
/// - From the scheduled last hand on: renchan first; the dealer stops only when at or
///   above the target and in first place.
/// - On rotation the match ends if anyone reached the target; otherwise tonpuusen
///   extends into South and hanchan into West, with sudden death each hand.
/// - The extension is at most one extra round (4-player East to South 4, South to
///   West 4; 3-player likewise to hand 3).
fn is_match_end(
    state: &RoundState,
    length: MatchLength,
    oya_kept: bool,
    keep_reason: DealerKeepReason,
    scores: &[i32],
    profile: RiichiRuleProfile,
) -> bool {
    if anyone_busted(scores, profile) {
        return true;
    }
    let scheduled = scheduled_last_bakaze(length);
    let seats = state.seats() as u8;
    let in_end_game = state.bakaze > scheduled || is_scheduled_final_kyoku(state, length);
    if !in_end_game {
        return false;
    }

    // The last extension hand only ends the match on rotation (there is no North round
    // after it); dealer renchan continues with more honba. On Tenhou, West 3 with a
    // dealer renchan and nobody at the target continues as West 3 honba 1, honba 2.
    if !oya_kept
        && state.bakaze >= scheduled.saturating_add(profile.max_extra_bakaze())
        && state.kyoku >= seats
    {
        return true;
    }

    let target = profile.target_score(state.seats());
    let top = rankings(scores).first().copied();
    if oya_kept {
        let stop_enabled = match keep_reason {
            DealerKeepReason::Hora => profile.allows_agariyame(),
            DealerKeepReason::Tenpai => profile.allows_tenpaiyame(),
        };
        stop_enabled && top == Some(state.oya) && scores[state.oya as usize] >= target
    } else {
        scores.iter().any(|score| *score >= target)
    }
}

/// State of the next hand after a win. `oya_won` means the dealer won (renchan).
fn advance_after_kyoku(
    state: &RoundState,
    length: MatchLength,
    oya_won: bool,
    scores: &[i32],
) -> RoundState {
    if oya_won {
        // Renchan: same dealer, honba + 1, riichi sticks cleared (taken by the winner), same hand and wind.
        RoundState {
            honba: state.honba.saturating_add(1),
            kyotaku: 0,
            scores: scores.to_vec(),
            ..clone_core(state)
        }
    } else {
        // Rotation: next dealer, honba 0, sticks cleared, hand and wind advance as needed.
        rotate_dealer(state, length, 0, 0, scores)
    }
}

/// State of the next hand after a draw. `oya_tenpai` means the dealer was tenpai (renchan). Riichi sticks stay.
fn advance_after_ryukyoku(
    state: &RoundState,
    length: MatchLength,
    oya_tenpai: bool,
    scores: &[i32],
) -> RoundState {
    if oya_tenpai {
        // Renchan: same dealer, honba + 1, sticks stay.
        RoundState {
            honba: state.honba.saturating_add(1),
            kyotaku: state.kyotaku,
            scores: scores.to_vec(),
            ..clone_core(state)
        }
    } else {
        // Rotation: next dealer, honba + 1 (accumulates over draws), sticks stay.
        rotate_dealer(
            state,
            length,
            state.honba.saturating_add(1),
            state.kyotaku,
            scores,
        )
    }
}

/// Rotation: the dealer moves one seat; after a full lap the hand number wraps and the round wind advances.
fn rotate_dealer(
    state: &RoundState,
    _length: MatchLength,
    honba: u8,
    kyotaku: u8,
    scores: &[i32],
) -> RoundState {
    let seats = state.seats() as u8;
    let next_oya = (state.oya + 1) % seats;
    // When the dealer returns to seat 0 a full round is complete and the wind advances.
    let wrapped = next_oya == 0;
    let (bakaze, kyoku) = if wrapped {
        (state.bakaze.saturating_add(1), 1)
    } else {
        (state.bakaze, state.kyoku.saturating_add(1))
    };
    RoundState {
        bakaze,
        kyoku,
        honba,
        kyotaku,
        oya: next_oya,
        scores: scores.to_vec(),
    }
}

/// Copies the fields that do not change (bakaze/kyoku/oya are decided by each branch; this keeps all for renchan).
fn clone_core(state: &RoundState) -> RoundState {
    RoundState {
        bakaze: state.bakaze,
        kyoku: state.kyoku,
        honba: state.honba,
        kyotaku: state.kyotaku,
        oya: state.oya,
        scores: state.scores.clone(),
    }
}

/// Rankings: seats by descending score, ties broken by seat order.
pub fn rankings(scores: &[i32]) -> Vec<u8> {
    let mut idx: Vec<u8> = (0..scores.len() as u8).collect();
    idx.sort_by(|&a, &b| scores[b as usize].cmp(&scores[a as usize]).then(a.cmp(&b)));
    idx
}
