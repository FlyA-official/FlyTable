//! 4-player game core.
//!
//! Holds perfect information (wall and four private seat states), deals, advances,
//! adjudicates and settles, and projects an imperfect-information view for any seat.
//!
//! 4-player only (4 seats, chi). 3-player is a separate board in [`crate::board3p`].
//!
//! Every external action is checked again here against the legal set of the current
//! [`SeatView`].

use flytable_core::rules::{KanDoraTiming, RiichiRuleProfile};
use flytable_core::tile::Tile;

use crate::action::{PendingRobbery, ReactionAction, RobberyKind, TurnAction};
use crate::player::PlayerState;
use crate::view::{OpponentView, SeatView, SeatViewError, SelfView};
use crate::wall::Wall;

const SEATS: usize = 4;

/// Round wind index to tile (0 = East, 1 = South, 2 = West, 3 = North). Out of range falls back to East.
fn bakaze_from_idx(idx: u8) -> Tile {
    match idx {
        1 => "S",
        2 => "W",
        3 => "N",
        _ => "E",
    }
    .parse()
    .unwrap()
}

/// How a hand ended.
#[derive(Debug, Clone)]
pub enum KyokuOutcome {
    /// A player won.
    Hora {
        winner: u8,
        from: Option<u8>,
        /// Full settlement (yaku, han, fu, points, dora).
        score: flytable_core::score::FullScore,
    },
    /// Multiple ron on one discard. `first` is the winner closest to the discarder, so the result is never empty.
    MultiHora {
        first: crate::HoraResult,
        additional: Vec<crate::HoraResult>,
    },
    /// Exhaustive draw; `tenpai` lists the tenpai seats.
    Ryukyoku { tenpai: Vec<u8> },
    /// Nagashi mangan. Renchan, honba and sticks follow the exhaustive draw; `tenpai`
    /// decides whether the dealer keeps the seat.
    NagashiMangan { winners: Vec<u8>, tenpai: Vec<u8> },
    /// Abortive draw.
    AbortiveRyukyoku { reason: AbortiveRyukyokuReason4p },
}

/// Reason for a 4-player abortive draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortiveRyukyokuReason4p {
    KyuushuKyuuhai,
    SuufonRenda,
    SuuchaRiichi,
    Suukaikan,
    Sanchaho,
}

/// Request sent to decision makers during play.
pub enum Prompt {
    /// `seat` acts after drawing.
    Turn { seat: u8 },
    /// `discarder` discarded `tile`; ask the seats that may respond.
    Reaction { discarder: u8, tile: Tile },
}

/// 4-player table.
pub struct Board4p {
    /// Platform rules. Defaults to Tenhou; integrations should call [`Self::with_rule_profile`].
    pub rule_profile: RiichiRuleProfile,
    pub wall: Wall,
    pub players: Vec<PlayerState>,
    pub bakaze: Tile,
    pub kyoku: u8,
    pub honba: u8,
    pub kyotaku: u8,
    pub oya: u8,
    pub scores: [i32; SEATS],
    /// Seat to draw or act.
    pub turn: u8,
    /// Latest discard `(seat, tile)`.
    pub last_discard: Option<(u8, Tile)>,
    /// Event log of this hand (standard event stream).
    pub log: Vec<flytable_event::Event4p>,
    /// Whether the latest draw was a replacement draw.
    last_draw_was_rinshan: bool,
    /// Whether the drawn tile can still score ippatsu tsumo; the public ippatsu flag expires once the player draws.
    last_draw_had_ippatsu: bool,
    /// Riichi declared with its discard, awaiting the response window. If the
    /// declaration tile deals in, no stick is taken and no `ReachAccepted` is emitted;
    /// otherwise it is confirmed when the window clears or on a call.
    pending_riichi: Option<u8>,
    /// Kan declaration written to the log, awaiting ron responses, before the replacement draw.
    pending_robbery: Option<PendingRobbery>,
    /// Dora from open or added kans still to be revealed (after the replacement draw's
    /// discard or before the next replacement draw).
    pending_kan_dora: u8,
    /// Liable player per seat for daisangen.
    pao_daisangen: [Option<u8>; SEATS],
    /// Liable player per seat for daisuushi.
    pao_daisuushi: [Option<u8>; SEATS],
    /// Whether the hand has ended in a win or draw. Irreversible.
    terminal: bool,
}

impl Board4p {
    /// Starts a hand from a seed (East 1, dealer seat 0).
    pub fn start(seed: (u64, u64), scores: [i32; SEATS]) -> Self {
        let wall = Wall::shuffled(crate::tileset::tileset_4p(), SEATS, seed);
        Self::from_wall(wall, scores, 0, 1)
    }

    /// Starts a hand with an explicit rule profile that decides the physical tile set before shuffling.
    pub fn start_with_rule_profile(
        seed: (u64, u64),
        scores: [i32; SEATS],
        rule_profile: RiichiRuleProfile,
    ) -> Result<Self, String> {
        rule_profile.validate_for_players(SEATS)?;
        let tiles = crate::tileset::tileset_4p_with_red_fives(rule_profile.red_fives(SEATS))?;
        let wall = Wall::shuffled(tiles, SEATS, seed);
        Self::from_wall(wall, scores, 0, 1).with_rule_profile(rule_profile)
    }

    /// Sets the platform rules and checks that the existing wall's red fives match the profile.
    ///
    /// Prefer [`Self::start_with_rule_profile`] for new shuffles; this is mainly for
    /// ordered wall fixtures and fails closed on a mismatch before any action is enumerated.
    pub fn with_rule_profile(mut self, rule_profile: RiichiRuleProfile) -> Result<Self, String> {
        rule_profile.validate_for_players(SEATS)?;
        let actual = self.wall.red_five_counts();
        let expected = rule_profile.red_fives(SEATS);
        if actual != expected {
            return Err(format!(
                "4p wall red fives do not match the rule profile: wall={actual:?}, profile={expected:?}"
            ));
        }
        self.rule_profile = rule_profile;
        Ok(self)
    }

    /// Starts a hand from a given wall (East round, honba 0, no riichi sticks).
    pub fn from_wall(wall: Wall, scores: [i32; SEATS], oya: u8, kyoku: u8) -> Self {
        Self::from_wall_with_state(wall, scores, 0, kyoku, oya, 0, 0)
    }

    /// Starts a hand from a given wall and full match state (for hand-to-hand progression).
    ///
    /// `bakaze`: 0 = East, 1 = South, 2 = West, 3 = North. `honba` / `kyotaku` carry into
    /// this hand (they affect `StartKyoku` and settlement). The 1000 points for a riichi
    /// stick are deducted by the match layer; this only records the stick count.
    pub fn from_wall_with_state(
        wall: Wall,
        scores: [i32; SEATS],
        bakaze_idx: u8,
        kyoku: u8,
        oya: u8,
        honba: u8,
        kyotaku: u8,
    ) -> Self {
        let players = (0..SEATS)
            .map(|s| PlayerState::new(s as u8, wall.haipai(s)))
            .collect();
        let bakaze = bakaze_from_idx(bakaze_idx);
        let dora = wall.dora_indicators();
        let mut board = Self {
            rule_profile: RiichiRuleProfile::default(),
            wall,
            players,
            bakaze,
            kyoku,
            honba,
            kyotaku,
            oya,
            scores,
            turn: oya,
            last_discard: None,
            log: Vec::new(),
            last_draw_was_rinshan: false,
            last_draw_had_ippatsu: false,
            pending_riichi: None,
            pending_robbery: None,
            pending_kan_dora: 0,
            pao_daisangen: [None; SEATS],
            pao_daisuushi: [None; SEATS],
            terminal: false,
        };
        // Emit the start-of-hand event.
        let tehais = std::array::from_fn(|s| board.players[s].hand.clone().try_into().unwrap());
        board.log.push(flytable_event::Event4p::StartKyoku {
            bakaze,
            dora_marker: dora.first().copied().unwrap_or_default(),
            kyoku,
            honba,
            kyotaku,
            oya,
            scores,
            tehais,
        });
        board
    }

    /// Imperfect-information view for a seat (other hands as counts only).
    pub fn view_for(&self, seat: u8) -> SeatView {
        if seat as usize >= SEATS {
            return SeatView::invalid(
                seat,
                SEATS as u8,
                self.rule_profile,
                SeatViewError::InvalidSeat {
                    requested: seat,
                    seats: SEATS as u8,
                },
            );
        }
        let me_ps = &self.players[seat as usize];
        let me = SelfView {
            seat,
            hand: me_ps.hand.clone(),
            drawn_tile: me_ps.drawn_tile,
            dealer_opening: false,
            melds: me_ps.melds.clone(),
            discards: me_ps.discards.clone(),
            riichi: me_ps.riichi,
            ippatsu: me_ps.ippatsu || (seat == self.turn && self.last_draw_had_ippatsu),
            kuikae_forbidden: me_ps.kuikae_forbidden.clone(),
            temporary_furiten: me_ps.temporary_furiten,
            riichi_furiten: me_ps.riichi_furiten,
        };
        let others = (0..SEATS as u8)
            .filter(|&s| s != seat)
            .map(|s| {
                let p = &self.players[s as usize];
                OpponentView {
                    seat: s,
                    hand_count: p.hand.len() as u8,
                    melds: p.melds.clone(),
                    discards: p.discards.clone(),
                    riichi: p.riichi,
                }
            })
            .collect();
        SeatView {
            state_error: None,
            round_terminal: self.terminal,
            rule_profile: self.rule_profile,
            bakaze: self.bakaze,
            kyoku: self.kyoku,
            honba: self.honba,
            kyotaku: self.kyotaku,
            oya: self.oya,
            scores: self.scores.to_vec(),
            dora_indicators: self.wall.dora_indicators(),
            tiles_left: self.wall.live_remaining() as u32,
            me,
            others,
            turn: self.turn,
            last_discard: self.last_discard,
            pending_robbery: self.pending_robbery,
            kyuushu_kyuuhai_window: self.is_kyuushu_kyuuhai_window(seat),
            suukaikan_pending: self.last_discard.is_some() && self.suukaikan_condition_met(),
            last_draw_was_rinshan: self.last_draw_was_rinshan,
        }
    }

    /// Current phase, derived read-only from existing fields so it cannot drift from the
    /// actual state. Use this instead of combining `terminal` / `last_discard` /
    /// `pending_robbery` / `drawn_tile` / hand size.
    #[must_use]
    pub fn phase(&self) -> crate::phase::BoardPhase {
        use crate::phase::BoardPhase;
        if self.terminal {
            return BoardPhase::Terminal;
        }
        if let Some(p) = self.pending_robbery {
            return BoardPhase::RobberyWindow { actor: p.actor };
        }
        if let Some((discarder, tile)) = self.last_discard {
            return BoardPhase::ReactionWindow { discarder, tile };
        }
        let seat = self.turn;
        let me = &self.players[seat as usize];
        // 3n+2 (just drew or called) means discard; 3n+1 means draw.
        if me.drawn_tile.is_some() || me.hand.len() % 3 == 2 {
            BoardPhase::AwaitingDiscard { seat }
        } else {
            BoardPhase::AwaitingDraw { seat }
        }
    }

    /// Draws with a typed failure reason.
    ///
    /// Same as [`Self::draw_for_turn`], but the ambiguous `None` is split:
    /// `WallExhausted` is the normal exhaustive draw signal, everything else is API misuse.
    pub fn try_draw_for_turn(&mut self) -> Result<Tile, crate::phase::TransitionError> {
        use crate::phase::{BoardPhase, TransitionError};
        let phase = self.phase();
        match phase {
            BoardPhase::Terminal => return Err(TransitionError::RoundFinished),
            BoardPhase::AwaitingDraw { .. } => {}
            other => {
                return Err(TransitionError::WrongPhase {
                    expected: "awaiting_draw",
                    actual: other.name(),
                });
            }
        }
        if self.turn as usize >= SEATS {
            return Err(TransitionError::SeatOutOfRange {
                seat: self.turn,
                seats: SEATS as u8,
            });
        }
        self.draw_for_turn().ok_or(TransitionError::WallExhausted)
    }

    /// Draws a tile for the current seat (`None` means the live wall is empty: exhaustive draw).
    pub fn draw_for_turn(&mut self) -> Option<Tile> {
        if self.terminal
            || self.turn as usize >= SEATS
            || self.last_discard.is_some()
            || self.pending_robbery.is_some()
            || self.players[self.turn as usize].drawn_tile.is_some()
            || self.players[self.turn as usize].hand.len() % 3 != 1
        {
            return None;
        }
        let t = self.wall.draw()?;
        self.last_draw_was_rinshan = false;
        self.last_draw_had_ippatsu = self.players[self.turn as usize].ippatsu;
        self.players[self.turn as usize].ippatsu = false;
        self.players[self.turn as usize].draw(t);
        self.log.push(flytable_event::Event4p::Tsumo {
            actor: self.turn,
            pai: t,
        });
        Some(t)
    }

    /// Applies the current seat's turn action. `Some(outcome)` means the hand ended.
    pub fn apply_turn(&mut self, action: TurnAction) -> Result<Option<KyokuOutcome>, String> {
        if self.terminal {
            return Err("the hand has ended".into());
        }
        if self.pending_robbery.is_some() {
            return Err("a kan declaration is still awaiting ron responses".into());
        }
        if self.turn as usize >= SEATS {
            return Err(format!("invalid acting seat {}", self.turn));
        }
        match &action {
            TurnAction::Discard { tile, tsumogiri } | TurnAction::Riichi { tile, tsumogiri } => {
                let mut probe = self.players[self.turn as usize].clone();
                probe.discard_with_source(*tile, *tsumogiri)?;
            }
            TurnAction::Ankan { tile }
                if self.players[self.turn as usize].riichi
                    && !self.ankan_is_legal_now(self.turn, *tile) =>
            {
                return Err("cannot make this closed kan after riichi".into());
            }
            TurnAction::KyuushuKyuuhai if !self.can_kyuushu_kyuuhai(self.turn) => {
                return Err("kyuushu kyuuhai condition not met".into());
            }
            _ => {}
        }
        let legal = crate::legal::legal_turn_actions(&self.view_for(self.turn));
        if !legal.contains(&action) {
            return Err(format!(
                "action is not in the current legal set: {action:?}"
            ));
        }
        match action {
            TurnAction::DealerOpeningDiscard { .. } | TurnAction::DealerOpeningRiichi { .. } => {
                unreachable!("authoritative board never enters a platform dealer-opening mirror")
            }
            TurnAction::Discard { tile, tsumogiri } => {
                // Four identical winds is not checked here. Like four riichi and four kans it is an
                // abortive draw that only happens once the tile was not won, so it is checked in
                // `clear_last_discard`. Checking here would miss the riichi declaration discard,
                // which goes through the `TurnAction::Riichi` branch.
                self.do_discard(self.turn, tile, tsumogiri)?;
                Ok(None)
            }
            TurnAction::Riichi { tile, tsumogiri } => {
                let seat = self.turn as usize;
                let double_riichi = self.players.iter().all(|player| player.melds.is_empty())
                    && self.players[seat].discards.is_empty();
                self.log
                    .push(flytable_event::Event4p::Reach { actor: self.turn });
                self.do_discard(self.turn, tile, tsumogiri)?;
                self.players[seat].riichi = true;
                self.players[seat].double_riichi = double_riichi;
                self.players[seat].ippatsu = true;
                self.players[seat].riichi_discard_idx =
                    Some(self.players[seat].discards.len().saturating_sub(1));
                self.pending_riichi = Some(self.turn);
                Ok(None)
            }
            TurnAction::Tsumo => {
                let seat = self.turn;
                let win = *self.players[seat as usize]
                    .hand
                    .last()
                    .ok_or("cannot tsumo with an empty hand")?;
                let score = self
                    .settle(seat, win, true)
                    .ok_or("no yaku; cannot win by tsumo")?;
                self.push_hora_event(seat, seat, &score);
                self.terminal = true;
                Ok(Some(KyokuOutcome::Hora {
                    winner: seat,
                    from: None,
                    score,
                }))
            }
            TurnAction::Ankan { tile } => {
                // Closed kan: remove four tiles, replacement draw, reveal a new indicator.
                self.do_ankan(self.turn, tile)
            }
            TurnAction::Kakan { tile } => {
                // Added kan onto an existing pon. Scripted chankan goes through `apply_chankan`;
                // the normal path draws the replacement tile immediately.
                self.do_kakan(self.turn, tile)
            }
            TurnAction::KyuushuKyuuhai => self.do_kyuushu_kyuuhai(),
            TurnAction::Nukidora => Err("nukidora does not exist in 4p".into()),
        }
    }

    fn do_kyuushu_kyuuhai(&mut self) -> Result<Option<KyokuOutcome>, String> {
        if !self.can_kyuushu_kyuuhai(self.turn) {
            return Err("kyuushu kyuuhai condition not met".into());
        }
        self.log.push(flytable_event::Event4p::Ryukyoku {
            deltas: Some([0; SEATS]),
        });
        self.terminal = true;
        Ok(Some(KyokuOutcome::AbortiveRyukyoku {
            reason: AbortiveRyukyokuReason4p::KyuushuKyuuhai,
        }))
    }

    fn do_discard(&mut self, seat: u8, tile: Tile, tsumogiri: bool) -> Result<(), String> {
        self.players[seat as usize].discard_with_source(tile, tsumogiri)?;
        self.last_draw_was_rinshan = false;
        self.last_draw_had_ippatsu = false;
        self.last_discard = Some((seat, tile));
        self.log.push(flytable_event::Event4p::Dahai {
            actor: seat,
            pai: tile,
            tsumogiri,
        });
        self.reveal_pending_kan_dora();
        self.note_yakuless_missed_ron(seat, tile);
        Ok(())
    }

    /// Passive missed ron without a yaku: record temporary furiten as soon as the tile is discarded.
    ///
    /// Temporary furiten applies whenever a waited tile is discarded and not won,
    /// regardless of yaku. But ron is only in the legal responses when there is a yaku,
    /// so a tenpai seat without a yaku is never asked, and the usual "Ron was legal but
    /// not chosen -> `note_missed_ron`" path never sees it.
    ///
    /// This covers that case: the tile completes the seat's hand, the seat is not already
    /// furiten, and ron is still illegal, which can only mean no yaku.
    ///
    /// When ron is legal the choice belongs to the seat, so it is skipped here; setting
    /// furiten early would block that very ron.
    fn note_yakuless_missed_ron(&mut self, from: u8, tile: Tile) {
        let missed: Vec<u8> = (0..SEATS as u8)
            .filter(|&seat| seat != from)
            .filter(|&seat| {
                let player = &self.players[seat as usize];
                !player.is_furiten_blocked()
                    && player.would_agari_with(tile)
                    && !self.can_ron_agari(seat, tile)
            })
            .collect();
        for seat in missed {
            self.players[seat as usize].note_missed_ron();
        }
    }

    fn reveal_pending_kan_dora(&mut self) {
        while self.pending_kan_dora > 0 {
            let Some(marker) = self.wall.reveal_kan_dora() else {
                break;
            };
            self.pending_kan_dora -= 1;
            self.log.push(flytable_event::Event4p::Dora {
                dora_marker: marker,
            });
        }
    }

    /// Mahjong Soul: on consecutive kans the pending indicator from the previous kan is
    /// revealed at this declaration, so a robbing player gets it. Tenhou (flag false)
    /// keeps revealing after the window and before the replacement draw.
    fn flush_pending_kan_dora_for_declaration(&mut self) {
        if self.rule_profile.kan_dora_pending_flush_at_declaration() {
            self.reveal_pending_kan_dora();
        }
    }

    /// The declaration tile did not deal in: take 1000 points, add a stick, emit the second-stage event.
    fn accept_pending_riichi(&mut self) {
        let Some(actor) = self.pending_riichi.take() else {
            return;
        };
        let score = &mut self.scores[actor as usize];
        debug_assert!(
            *score >= 1_000,
            "legal enumeration guarantees a riichi stick"
        );
        *score -= 1_000;
        self.kyotaku = self.kyotaku.saturating_add(1);
        self.log
            .push(flytable_event::Event4p::ReachAccepted { actor });
    }

    /// The declaration tile dealt in: undo the pending riichi (the declaration event stays, no second stage).
    fn cancel_pending_riichi(&mut self) {
        let Some(actor) = self.pending_riichi.take() else {
            return;
        };
        let player = &mut self.players[actor as usize];
        player.riichi = false;
        player.double_riichi = false;
        player.ippatsu = false;
        player.riichi_discard_idx = None;
    }

    /// Closes the current response window. With no ron, a pending riichi is accepted first.
    pub fn clear_last_discard(&mut self) -> Option<KyokuOutcome> {
        if self.terminal {
            return None;
        }
        self.accept_pending_riichi();
        self.last_discard = None;
        if self.rule_profile.allows_suucha_riichi()
            && self.players.iter().all(|player| player.riichi)
        {
            self.log.push(flytable_event::Event4p::Ryukyoku {
                deltas: Some([0; SEATS]),
            });
            self.terminal = true;
            return Some(KyokuOutcome::AbortiveRyukyoku {
                reason: AbortiveRyukyokuReason4p::SuuchaRiichi,
            });
        }
        // Four kans: nobody won the fourth kan player's discard, so the hand ends in a draw now.
        if self.suukaikan_condition_met() {
            return Some(self.abort_suukaikan());
        }
        // Four identical winds is checked alongside the other two draws, after
        // `accept_pending_riichi`.
        //
        // It cannot end the hand in the discard branch of `apply_turn`: a riichi declaration
        // discard goes through `TurnAction::Riichi` and would never reach that code, and the
        // hand would end before the ron window opens.
        //
        // Accepting riichi before the draw matches Tenhou: for four riichi and for four
        // kans on a declaration discard, the stick is recorded first and then the draw.
        if self.is_suufon_renda() {
            self.log.push(flytable_event::Event4p::Ryukyoku {
                deltas: Some([0; SEATS]),
            });
            self.terminal = true;
            return Some(KyokuOutcome::AbortiveRyukyoku {
                reason: AbortiveRyukyokuReason4p::SuufonRenda,
            });
        }
        None
    }

    /// Verify that the public response window is backed by the same, still
    /// uncalled tile at the end of the discarder river.
    ///
    /// `last_discard` is public for inspection/fixtures, so action entry
    /// points must not trust it in isolation.  Checking before any mutation
    /// also keeps malformed injected states atomic.
    fn validate_discard_window(&self, from: u8, tile: Tile) -> Result<(), String> {
        let player = self
            .players
            .get(from as usize)
            .ok_or_else(|| format!("invalid discarding seat {from}"))?;
        let index = player
            .discards
            .len()
            .checked_sub(1)
            .ok_or("no discard record for the current discard window")?;
        if player.discards[index] != tile {
            return Err("the current discard window does not match the last discard".into());
        }
        if player.called_discards.contains(&index) {
            return Err("the current discard has already been called".into());
        }
        Ok(())
    }

    fn ankan_is_legal_now(&self, seat: u8, tile: Tile) -> bool {
        crate::legal::legal_turn_actions(&self.view_for(seat))
            .iter()
            .any(
                |action| matches!(action, TurnAction::Ankan { tile: t } if t.kind() == tile.kind()),
            )
    }

    fn do_ankan(&mut self, seat: u8, tile: Tile) -> Result<Option<KyokuOutcome>, String> {
        if self.players[seat as usize].riichi && !self.ankan_is_legal_now(seat, tile) {
            return Err("cannot make this closed kan after riichi".into());
        }
        if self.wall.rinshan_remaining() == 0 {
            return Err("no replacement tiles left".into());
        }
        let p = &mut self.players[seat as usize];
        let k = tile.kind();
        let have = p.hand.iter().filter(|t| t.kind() == k).count();
        if have < 4 {
            return Err("closed kan requires four tiles".into());
        }
        let p = &mut self.players[seat as usize];
        let consumed: [Tile; 4] = remove_kind_tiles(&mut p.hand, k, 4)
            .expect("closed kan count already checked")
            .try_into()
            .expect("closed kan removes four tiles");
        p.drawn_tile = None;
        p.melds
            .push(flytable_core::meld::Meld::Ankan { tile: tile.deaka() });
        self.flush_pending_kan_dora_for_declaration();
        self.log.push(flytable_event::Event4p::Ankan {
            actor: seat,
            consumed,
        });
        // The previous delayed kan dora waits until this declaration's ron window passes (see resolve_pending_robbery_passes).
        self.pending_robbery = Some(PendingRobbery {
            actor: seat,
            tile: tile.deaka(),
            kind: RobberyKind::Ankan,
        });
        if !self.has_robbery_candidate() {
            return self.resolve_pending_robbery_passes();
        }
        Ok(None)
    }

    /// Added kan onto an existing pon: replacement draw and a new indicator.
    fn do_kakan(&mut self, seat: u8, tile: Tile) -> Result<Option<KyokuOutcome>, String> {
        if self.wall.rinshan_remaining() == 0 {
            return Err("no replacement tiles left".into());
        }
        let k = tile.kind();
        // Find the pon of this kind.
        let pon_idx = self.players[seat as usize]
            .melds
            .iter()
            .position(
                |m| matches!(m, flytable_core::meld::Meld::Pon { tile: t, .. } if t.kind() == k),
            )
            .ok_or("added kan requires a pon of the same kind")?;
        // Remove the added tile from hand (it may be red).
        let hpos = self.players[seat as usize]
            .hand
            .iter()
            .position(|t| t.kind() == k)
            .ok_or("the added kan tile is not in hand")?;
        let p = &mut self.players[seat as usize];
        let added = p.hand.remove(hpos);
        let (base, called, consumed, from) = match &p.melds[pon_idx] {
            flytable_core::meld::Meld::Pon {
                tile,
                called,
                consumed,
                from,
            } => (*tile, *called, *consumed, *from),
            _ => unreachable!(),
        };
        let kakan_consumed = [consumed[0], consumed[1], called];
        let _ = (called, from);
        p.drawn_tile = None;
        p.melds[pon_idx] = flytable_core::meld::Meld::Kakan { tile: base, added };
        self.flush_pending_kan_dora_for_declaration();
        self.log.push(flytable_event::Event4p::Kakan {
            actor: seat,
            pai: added,
            consumed: kakan_consumed,
        });
        // The previous delayed kan dora waits until this declaration's ron window passes (see resolve_pending_robbery_passes).
        self.pending_robbery = Some(PendingRobbery {
            actor: seat,
            tile: added,
            kind: RobberyKind::Kakan,
        });
        if !self.has_robbery_candidate() {
            return self.resolve_pending_robbery_passes();
        }
        Ok(None)
    }

    /// Current public kan declaration; while `Some`, no draw or normal turn action is allowed.
    pub const fn pending_robbery(&self) -> Option<PendingRobbery> {
        self.pending_robbery
    }

    /// Nobody robbed the kan: only now is ippatsu broken, the kan completed and the replacement drawn.
    pub fn resolve_pending_robbery_passes(&mut self) -> Result<Option<KyokuOutcome>, String> {
        if self.terminal {
            return Err("the hand has ended".into());
        }
        let pending = self
            .pending_robbery
            .take()
            .ok_or("no pending kan declaration")?;
        self.clear_all_ippatsu();
        // Four kans does not end the hand here: the kan player draws the replacement and
        // discards as usual (see `clear_last_discard`).
        //
        // The delayed dora of the previous open or added kan is due now. On Tenhou it is
        // revealed before the discard or the next replacement draw
        // (`kakan -> rinshan -> kita/kakan/ankan -> DORA -> replacement`), and not revealed
        // when the nukidora is robbed. So it appears after this declaration's ron window and
        // before the replacement draw, and a robbing player does not get it.
        self.reveal_pending_kan_dora();
        self.last_draw_was_rinshan = true;
        self.last_draw_had_ippatsu = false;
        let r = self
            .wall
            .draw_rinshan()
            .ok_or("no replacement tiles left")?;
        match pending.kind {
            RobberyKind::Ankan => {
                if let Some(dm) = self.wall.reveal_kan_dora() {
                    self.log
                        .push(flytable_event::Event4p::Dora { dora_marker: dm });
                }
            }
            RobberyKind::Kakan => match self.rule_profile.kan_dora_timing() {
                KanDoraTiming::AnkanImmediateOpenKanDelayed => {
                    self.pending_kan_dora = self.pending_kan_dora.saturating_add(1);
                }
                KanDoraTiming::Immediate => {
                    if let Some(dm) = self.wall.reveal_kan_dora() {
                        self.log
                            .push(flytable_event::Event4p::Dora { dora_marker: dm });
                    }
                }
            },
            RobberyKind::Nukidora => return Err("nukidora does not exist in 4p".into()),
        }
        self.players[pending.actor as usize].draw(r);
        self.log.push(flytable_event::Event4p::Tsumo {
            actor: pending.actor,
            pai: r,
        });
        Ok(None)
    }

    /// Four-kan draw condition: four kans held by at least two players (suukantsu
    /// excluded). Once true it stays true, and the fourth kan player's discard ends the
    /// hand when its window clears, so "condition met and a pending discard" identifies
    /// the ron-only window without extra state.
    fn suukaikan_condition_met(&self) -> bool {
        // Shared with 3P in `crate::timing`.
        crate::timing::suukaikan_condition_met(
            self.rule_profile.allows_suukaikan(),
            self.players
                .iter()
                .map(|player| player.melds.iter().filter(|meld| meld.is_kan()).count()),
            self.pending_kan_dora,
        )
    }

    fn abort_suukaikan(&mut self) -> KyokuOutcome {
        self.log.push(flytable_event::Event4p::Ryukyoku {
            deltas: Some([0; SEATS]),
        });
        self.terminal = true;
        KyokuOutcome::AbortiveRyukyoku {
            reason: AbortiveRyukyokuReason4p::Suukaikan,
        }
    }

    /// Moves to the next seat (after a discard nobody called or won).
    pub fn advance_turn(&mut self) {
        if self.terminal || self.pending_robbery.is_some() {
            return;
        }
        if self.clear_last_discard().is_some() {
            return;
        }
        self.turn = (self.turn + 1) % SEATS as u8;
    }

    /// Legal responses of a seat to the current `last_discard` (without Pass).
    pub fn legal_reactions(&self, seat: u8) -> Vec<crate::action::ReactionAction> {
        let mut out = Vec::new();
        if self.terminal || seat as usize >= SEATS {
            return out;
        }
        if self.pending_robbery.is_some() {
            return self.legal_robbery_reactions(seat);
        }
        let Some((from, tile)) = self.last_discard else {
            return out;
        };
        if self.validate_discard_window(from, tile).is_err() {
            return out;
        }
        if seat == from {
            return out;
        }
        let p = &self.players[seat as usize];
        let k = tile.kind();
        // Ron: complete hand, no furiten and a yaku (same as settlement in apply_ron). After
        // riichi only ron is possible.
        if self.can_ron_agari(seat, tile) {
            out.push(ReactionAction::Ron);
        }
        // Only ron on the discard that completes a pending four-kan draw.
        if self.suukaikan_condition_met() {
            return out;
        }
        if self.wall.live_remaining() == 0 {
            return out;
        }
        if p.riichi {
            return out;
        }
        // Pon: two tiles of the kind in hand. With kuikae forbidden, a call must leave a
        // discardable tile (see `tileset::call_leaves_legal_discard`), as in the
        // maintainer's `legal::legal_reactions`.
        let kuikae_on = self.rule_profile.kuikae_forbidden();
        let same: Vec<Tile> = p.hand.iter().copied().filter(|t| t.kind() == k).collect();
        if same.len() >= 2 {
            for consumed in crate::tileset::pon_pair_combos(&same) {
                if kuikae_on && !crate::tileset::call_leaves_legal_discard(&p.hand, &consumed, &[k])
                {
                    continue;
                }
                out.push(ReactionAction::Pon { consumed });
            }
            if same.len() >= 3 {
                out.push(ReactionAction::Daiminkan);
            }
        }
        // Chi: only from the player to the left (`(from + 1) % 4 == seat`) and only number
        // tiles. Lists every pair of hand tiles that forms a sequence with the called tile.
        if (from + 1) % SEATS as u8 == seat && tile.rank().is_some() {
            for combo in crate::tileset::chi_combos(&p.hand, tile) {
                if kuikae_on {
                    let forbidden = crate::tileset::kuikae_forbidden_after_chi(tile, combo);
                    if !crate::tileset::call_leaves_legal_discard(&p.hand, &combo, &forbidden) {
                        continue;
                    }
                }
                out.push(ReactionAction::Chi { consumed: combo });
            }
        }
        out
    }

    /// Legal responses to a kan declaration: ron only; Pass is always implicit.
    pub fn legal_robbery_reactions(&self, seat: u8) -> Vec<ReactionAction> {
        let Some(pending) = self.pending_robbery else {
            return Vec::new();
        };
        if self.terminal
            || seat as usize >= SEATS
            || seat == pending.actor
            || !self.can_robbery_ron(seat, pending)
        {
            Vec::new()
        } else {
            vec![ReactionAction::Ron]
        }
    }

    /// View for a response (includes who discarded what).
    pub fn view_for_reaction(&self, seat: u8) -> SeatView {
        self.view_for(seat)
    }

    /// The seat passed on a ron: riichi furiten in riichi, temporary furiten otherwise.
    pub fn note_missed_ron(&mut self, seat: u8) -> Result<(), String> {
        if self.terminal {
            return Err("the hand has ended".into());
        }
        let player = self
            .players
            .get_mut(seat as usize)
            .ok_or_else(|| format!("invalid seat {seat}"))?;
        player.note_missed_ron();
        Ok(())
    }

    /// Applies a ron from response polling. The winning tile is `last_discard`.
    pub fn apply_ron(&mut self, winner: u8, from: u8, tile: Tile) -> Result<KyokuOutcome, String> {
        self.apply_rons(&[winner], from, tile)
    }

    /// Atomically applies one or more rons on the same discard.
    ///
    /// Every winner is checked for a complete hand, furiten and yaku first; if any check
    /// fails, no event is added and the hand does not end. Results are ordered from
    /// closest to farthest from the discarder so settlement can assign honba and sticks.
    pub fn apply_rons(
        &mut self,
        winners: &[u8],
        from: u8,
        tile: Tile,
    ) -> Result<KyokuOutcome, String> {
        if self.terminal {
            return Err("the hand has ended".into());
        }
        if from as usize >= SEATS {
            return Err("invalid discarder seat".into());
        }
        if self.last_discard != Some((from, tile)) {
            return Err("ron source must match the current discard".into());
        }
        self.validate_discard_window(from, tile)?;
        if winners.is_empty() {
            return Err("ron needs at least one winner".into());
        }
        let mut ordered = winners.to_vec();
        ordered.sort_by_key(|&winner| (winner + SEATS as u8 - from) % SEATS as u8);
        let mut seen = [false; SEATS];
        let mut results = Vec::with_capacity(ordered.len());
        for winner in ordered {
            if winner as usize >= SEATS || winner == from || seen[winner as usize] {
                return Err("invalid or duplicate ron seat".into());
            }
            seen[winner as usize] = true;
            if !self.players[winner as usize].would_agari_with(tile) {
                return Err(format!(
                    "seat {winner} does not have a complete hand; cannot ron"
                ));
            }
            if self.players[winner as usize].is_furiten_blocked() {
                return Err(format!("seat {winner} is furiten; cannot ron"));
            }
            self.players[winner as usize].hand.push(tile);
            let score = self.settle(winner, tile, false);
            self.players[winner as usize].hand.pop();
            let score = score.ok_or_else(|| format!("seat {winner} has no yaku; cannot ron"))?;
            results.push(crate::HoraResult {
                winner,
                from: Some(from),
                score,
            });
        }
        if self.pending_riichi == Some(from) {
            self.cancel_pending_riichi();
        }
        for result in &results {
            self.push_hora_event(result.winner, from, &result.score);
        }
        self.terminal = true;
        let first = results.remove(0);
        if results.is_empty() {
            Ok(KyokuOutcome::Hora {
                winner: first.winner,
                from: first.from,
                score: first.score,
            })
        } else {
            Ok(KyokuOutcome::MultiHora {
                first,
                additional: results,
            })
        }
    }

    /// Atomically applies one or more rons on the current kan declaration.
    pub fn apply_robbery_rons(&mut self, winners: &[u8]) -> Result<KyokuOutcome, String> {
        if self.terminal {
            return Err("the hand has ended".into());
        }
        let pending = self.pending_robbery.ok_or("no pending kan declaration")?;
        if winners.is_empty() {
            return Err("ron needs at least one winner".into());
        }
        let mut ordered = winners.to_vec();
        ordered.sort_by_key(|&winner| (winner + SEATS as u8 - pending.actor) % SEATS as u8);
        let mut seen = [false; SEATS];
        let mut results = Vec::with_capacity(ordered.len());
        for winner in ordered {
            if winner as usize >= SEATS || winner == pending.actor || seen[winner as usize] {
                return Err("invalid or duplicate ron seat".into());
            }
            seen[winner as usize] = true;
            if !self.can_robbery_ron(winner, pending) {
                return Err(format!(
                    "seat {winner} cannot ron the current kan declaration"
                ));
            }
            let mut probe = self.players[winner as usize].clone();
            probe.hand.push(pending.tile);
            let score = self
                .settle_with_flags(
                    winner,
                    &probe,
                    pending.tile,
                    false,
                    false,
                    matches!(pending.kind, RobberyKind::Kakan),
                )
                .ok_or_else(|| {
                    format!("seat {winner} has no yaku; cannot ron the current kan declaration")
                })?;
            results.push(crate::HoraResult {
                winner,
                from: Some(pending.actor),
                score,
            });
        }
        for result in &results {
            self.push_hora_event(result.winner, pending.actor, &result.score);
        }
        self.pending_robbery = None;
        self.terminal = true;
        let first = results.remove(0);
        if results.is_empty() {
            Ok(KyokuOutcome::Hora {
                winner: first.winner,
                from: first.from,
                score: first.score,
            })
        } else {
            Ok(KyokuOutcome::MultiHora {
                first,
                additional: results,
            })
        }
    }

    /// Triple ron abortive draw. Only allowed in a response window for a discard or kan declaration.
    pub fn abort_sanchaho(&mut self) -> Result<KyokuOutcome, String> {
        if self.terminal {
            return Err("the hand has ended".into());
        }
        if self.last_discard.is_none() && self.pending_robbery.is_none() {
            return Err("triple ron must happen in a ron response window".into());
        }
        self.log.push(flytable_event::Event4p::Ryukyoku {
            deltas: Some([0; SEATS]),
        });
        self.terminal = true;
        Ok(KyokuOutcome::AbortiveRyukyoku {
            reason: AbortiveRyukyokuReason4p::Sanchaho,
        })
    }

    /// Scripted entry point: declare an added kan and have `winner` rob it immediately. Uses the same state machine as live play.
    pub fn apply_chankan(
        &mut self,
        winner: u8,
        actor: u8,
        tile: Tile,
    ) -> Result<KyokuOutcome, String> {
        if self.terminal {
            return Err("the hand has ended".into());
        }
        if actor as usize >= SEATS || winner as usize >= SEATS || winner == actor {
            return Err("invalid chankan seat".into());
        }
        let pending = PendingRobbery {
            actor,
            tile,
            kind: RobberyKind::Kakan,
        };
        if !self.can_robbery_ron(winner, pending) {
            return Err("the winner cannot rob this added kan".into());
        }
        self.do_kakan(actor, tile)?;
        self.apply_robbery_rons(&[winner])
    }

    /// Ron legality: complete hand, no furiten and a yaku, exactly as settled by
    /// `apply_ron`. The winning tile is added to a cloned hand, so state is not touched.
    fn can_ron_agari(&self, seat: u8, tile: Tile) -> bool {
        let p = &self.players[seat as usize];
        if p.is_furiten_blocked() || !p.would_agari_with(tile) {
            return false;
        }
        let mut probe = p.clone();
        probe.hand.push(tile);
        self.settle_with_flags(seat, &probe, tile, false, false, false)
            .is_some()
    }

    fn has_robbery_candidate(&self) -> bool {
        let Some(pending) = self.pending_robbery else {
            return false;
        };
        (0..SEATS as u8).any(|seat| self.can_robbery_ron(seat, pending))
    }

    fn can_robbery_ron(&self, seat: u8, pending: PendingRobbery) -> bool {
        if seat as usize >= SEATS || seat == pending.actor {
            return false;
        }
        match pending.kind {
            RobberyKind::Kakan => {}
            RobberyKind::Ankan if !self.rule_profile.allows_kokushi_ankan_ron() => {
                return false;
            }
            RobberyKind::Nukidora => return false,
            RobberyKind::Ankan => {}
        }
        let p = &self.players[seat as usize];
        if p.is_furiten_blocked() || !p.would_agari_with(pending.tile) {
            return false;
        }
        let mut probe = p.clone();
        probe.hand.push(pending.tile);
        if pending.kind == RobberyKind::Ankan
            && !flytable_core::decompose::is_kokushi(probe.hand_counts().raw())
        {
            return false;
        }
        self.settle_with_flags(
            seat,
            &probe,
            pending.tile,
            false,
            false,
            pending.kind == RobberyKind::Kakan,
        )
        .is_some()
    }

    /// Builds the settlement input and calls the rules core. `win` is the winning tile;
    /// the player's hand must already include it.
    fn settle(
        &self,
        winner: u8,
        win: Tile,
        is_tsumo: bool,
    ) -> Option<flytable_core::score::FullScore> {
        self.settle_with_flags(
            winner,
            &self.players[winner as usize],
            win,
            is_tsumo,
            is_tsumo && self.last_draw_was_rinshan,
            false,
        )
    }

    fn settle_with_flags(
        &self,
        winner: u8,
        player: &PlayerState,
        win: Tile,
        is_tsumo: bool,
        rinshan: bool,
        chankan: bool,
    ) -> Option<flytable_core::score::FullScore> {
        let p = player;
        let jikaze = self.seat_wind(winner);
        let ura_indicators = if p.riichi {
            self.wall.ura_indicators()
        } else {
            Vec::new()
        };
        let uninterrupted_first_draw = is_tsumo
            && self.players.iter().all(|player| player.melds.is_empty())
            && self
                .players
                .iter()
                .map(|player| player.discards.len())
                .sum::<usize>()
                == (winner + SEATS as u8 - self.oya) as usize % SEATS;
        let input = crate::scoring::ScoringInput {
            rule_profile: self.rule_profile,
            player: p,
            win_tile: win,
            is_tsumo,
            bakaze: self.bakaze,
            jikaze,
            is_oya: winner == self.oya,
            dora_indicators: &self.wall.dora_indicators(),
            ura_indicators: &ura_indicators,
            riichi: p.riichi,
            double_riichi: p.double_riichi,
            ippatsu: p.ippatsu || (is_tsumo && self.last_draw_had_ippatsu),
            haitei: is_tsumo && self.wall.live_remaining() == 0,
            houtei: !is_tsumo && self.wall.live_remaining() == 0,
            rinshan,
            chankan,
            tenhou: uninterrupted_first_draw && winner == self.oya,
            chiihou: uninterrupted_first_draw && winner != self.oya,
            nuki_count: 0,
            is_sanma: false,
        };
        crate::scoring::settle(&input)
    }

    /// Seat wind (dealer is East, counter-clockwise).
    ///
    /// Public because the match log needs the specific wind to write Tenhou yaku IDs
    /// (seat wind E/S/W/N are four different IDs), and that must use this same calculation.
    pub fn seat_wind(&self, seat: u8) -> Tile {
        let offset = (seat + SEATS as u8 - self.oya) % SEATS as u8;
        // 27=E 28=S 29=W 30=N
        unsafe { Tile::from_id_unchecked(27 + offset) }
    }

    /// Emits the Hora event with settlement details.
    fn push_hora_event(&mut self, winner: u8, from: u8, score: &flytable_core::score::FullScore) {
        use flytable_event::{HoraAgari, HoraScoring};
        let ura_markers = self.players[winner as usize]
            .riichi
            .then(|| self.wall.ura_indicators());
        let agari = if score.score.yakuman > 0 {
            HoraAgari::Yakuman {
                count: score.score.yakuman,
            }
        } else {
            HoraAgari::Normal {
                fu: score.fu,
                han: score.yaku.han + score.dora_han,
            }
        };
        let scoring = HoraScoring {
            agari,
            point: flytable_event::HoraPoint {
                ron: score.score.ron,
                tsumo_ko: score.score.tsumo_ko,
                tsumo_oya: score.score.tsumo_oya,
            },
            additional_hans: score.yaku.han,
            dora_han: score.regular_dora_han,
            red_dora_han: score.red_dora_han,
            ura_dora_han: score.ura_dora_han,
            nuki_dora_han: 0,
            yaku_flags: crate::scoring::yaku_flags_from_score(score),
        };
        self.log.push(flytable_event::Event4p::Hora {
            actor: winner,
            target: from,
            deltas: None,
            ura_markers,
            scoring: Some(scoring),
        });
    }

    /// Applies a pon (response to another player's discard).
    pub fn apply_pon(&mut self, actor: u8, consumed: [Tile; 2]) -> Result<(), String> {
        if self.terminal {
            return Err("the hand has ended".into());
        }
        if actor as usize >= SEATS {
            return Err(format!("invalid seat {actor}"));
        }
        let (from, tile) = self.last_discard.ok_or("no discard to pon")?;
        self.validate_discard_window(from, tile)?;
        if self.players[actor as usize].riichi {
            return Err("cannot pon after riichi".into());
        }
        let action = crate::action::ReactionAction::Pon { consumed };
        if !self
            .legal_reactions(actor)
            .iter()
            .any(|legal| legal.equivalent(&action))
        {
            return Err("pon is not in the current legal set".into());
        }
        self.accept_pending_riichi();
        {
            let p = &mut self.players[actor as usize];
            // Remove the two matching tiles.
            for c in consumed {
                let pos = p
                    .hand
                    .iter()
                    .position(|&t| t == c)
                    .expect("legal pon combination already validated");
                p.hand.remove(pos);
            }
            p.melds.push(flytable_core::meld::Meld::Pon {
                tile: tile.deaka(),
                called: tile,
                consumed,
                from,
            });
            p.drawn_tile = None;
            p.menzen = false;
            p.kuikae_forbidden = if self.rule_profile.kuikae_forbidden() {
                vec![tile.kind()]
            } else {
                Vec::new()
            };
        }
        self.players[from as usize].mark_latest_discard_called()?;
        self.update_pao_after_open_honor_call(actor, from, tile);
        self.clear_all_ippatsu();
        self.log.push(flytable_event::Event4p::Pon {
            actor,
            target: from,
            pai: tile,
            consumed,
        });
        // The pon player discards next.
        self.turn = actor;
        self.last_discard = None;
        Ok(())
    }

    /// Applies a chi (response to the left player's discard). `consumed` are the two tiles from hand.
    pub fn apply_chi(&mut self, actor: u8, consumed: [Tile; 2]) -> Result<(), String> {
        if self.terminal {
            return Err("the hand has ended".into());
        }
        if actor as usize >= SEATS {
            return Err(format!("invalid seat {actor}"));
        }
        let (from, tile) = self.last_discard.ok_or("no discard to chi")?;
        self.validate_discard_window(from, tile)?;
        if self.players[actor as usize].riichi {
            return Err("cannot chi after riichi".into());
        }
        let action = crate::action::ReactionAction::Chi { consumed };
        if !self
            .legal_reactions(actor)
            .iter()
            .any(|legal| legal.equivalent(&action))
        {
            return Err("chi is not in the current legal set".into());
        }
        self.accept_pending_riichi();
        {
            let p = &mut self.players[actor as usize];
            for c in consumed {
                let pos = p
                    .hand
                    .iter()
                    .position(|&t| t == c)
                    .expect("legal chi combination already validated");
                p.hand.remove(pos);
            }
            // Sequence tiles sorted by kind for display.
            let mut tiles = [tile, consumed[0], consumed[1]];
            tiles.sort_by_key(|t| t.kind());
            p.melds.push(flytable_core::meld::Meld::Chi {
                tiles,
                called: tile,
                from,
            });
            p.drawn_tile = None;
            p.menzen = false;
            p.kuikae_forbidden = if self.rule_profile.kuikae_forbidden() {
                crate::tileset::kuikae_forbidden_after_chi(tile, consumed)
            } else {
                Vec::new()
            };
        }
        self.players[from as usize].mark_latest_discard_called()?;
        self.clear_all_ippatsu();
        self.log.push(flytable_event::Event4p::Chi {
            actor,
            target: from,
            pai: tile,
            consumed,
        });
        // The chi player discards next.
        self.turn = actor;
        self.last_discard = None;
        Ok(())
    }

    /// Applies an open kan (three tiles from hand plus the discard): replacement draw and new indicator.
    pub fn apply_daiminkan(&mut self, actor: u8) -> Result<Option<KyokuOutcome>, String> {
        if self.terminal {
            return Err("the hand has ended".into());
        }
        if actor as usize >= SEATS {
            return Err(format!("invalid seat {actor}"));
        }
        if self.wall.rinshan_remaining() == 0 {
            return Err("no replacement tiles left".into());
        }
        let (from, tile) = self.last_discard.ok_or("no discard to kan")?;
        self.validate_discard_window(from, tile)?;
        if self.players[actor as usize].riichi {
            return Err("cannot make an open kan after riichi".into());
        }
        if !self
            .legal_reactions(actor)
            .contains(&crate::action::ReactionAction::Daiminkan)
        {
            return Err("open kan is not in the current legal set".into());
        }
        self.accept_pending_riichi();
        let k = tile.kind();
        let same: Vec<Tile> = self.players[actor as usize]
            .hand
            .iter()
            .copied()
            .filter(|t| t.kind() == k)
            .collect();
        if same.len() < 3 {
            return Err("open kan requires three tiles of the kind in hand".into());
        }
        self.clear_all_ippatsu();
        let p = &mut self.players[actor as usize];
        let consumed: [Tile; 3] = remove_kind_tiles(&mut p.hand, k, 3)
            .expect("open kan count already checked")
            .try_into()
            .expect("open kan removes three tiles");
        p.melds.push(flytable_core::meld::Meld::Daiminkan {
            tile: tile.deaka(),
            called: tile,
            from,
        });
        p.menzen = false;
        self.players[from as usize].mark_latest_discard_called()?;
        self.update_pao_after_open_honor_call(actor, from, tile);
        self.log.push(flytable_event::Event4p::Daiminkan {
            actor,
            target: from,
            pai: tile,
            consumed,
        });
        self.turn = actor;
        self.last_discard = None;
        // Open kan dora is delayed: draw the replacement first, reveal on the discard or before the next replacement draw.
        self.reveal_pending_kan_dora();
        if let Some(r) = self.wall.draw_rinshan() {
            match self.rule_profile.kan_dora_timing() {
                KanDoraTiming::AnkanImmediateOpenKanDelayed => {
                    self.pending_kan_dora = self.pending_kan_dora.saturating_add(1);
                }
                KanDoraTiming::Immediate => {
                    if let Some(dm) = self.wall.reveal_kan_dora() {
                        self.log
                            .push(flytable_event::Event4p::Dora { dora_marker: dm });
                    }
                }
            }
            self.last_draw_was_rinshan = true;
            self.last_draw_had_ippatsu = false;
            self.players[actor as usize].draw(r);
            self.log
                .push(flytable_event::Event4p::Tsumo { actor, pai: r });
            Ok(None)
        } else {
            Err("no replacement tiles left".into())
        }
    }

    /// The liable player if `winner` is under pao for daisangen or daisuushi.
    pub fn pao_payer(&self, winner: u8) -> Option<u8> {
        if !self.rule_profile.allows_pao() || winner as usize >= SEATS {
            return None;
        }
        self.pao_daisangen[winner as usize].or(self.pao_daisuushi[winner as usize])
    }

    /// Pao liability `(liable seat, covered yakuman multiplier)`. Daisangen is always 1;
    /// daisuushi is 1 or 2 depending on `double_special_yakuman`; both by the same player
    /// add up. Used to split composite yakuman (Mahjong Soul: the covered part is paid by
    /// the liable player alone, the rest is split normally). With different liable
    /// players for the two, the daisangen player is used.
    pub fn pao_liability(&self, winner: u8) -> Option<(u8, u8)> {
        if !self.rule_profile.allows_pao() || winner as usize >= SEATS {
            return None;
        }
        let w = winner as usize;
        let daisuushi_mult = if self.rule_profile.double_special_yakuman() {
            2
        } else {
            1
        };
        match (self.pao_daisangen[w], self.pao_daisuushi[w]) {
            (Some(a), Some(b)) if a == b => Some((a, 1 + daisuushi_mult)),
            (Some(a), Some(_)) => Some((a, 1)),
            (Some(a), None) => Some((a, 1)),
            (None, Some(b)) => Some((b, daisuushi_mult)),
            (None, None) => None,
        }
    }

    fn update_pao_after_open_honor_call(&mut self, actor: u8, from: u8, called: Tile) {
        if !self.rule_profile.allows_pao() || !called.is_honor() {
            return;
        }
        let mut honor_triplets = [false; 7];
        for meld in &self.players[actor as usize].melds {
            if matches!(
                meld,
                flytable_core::meld::Meld::Pon { .. }
                    | flytable_core::meld::Meld::Daiminkan { .. }
                    | flytable_core::meld::Meld::Kakan { .. }
                    | flytable_core::meld::Meld::Ankan { .. }
            ) {
                let kind = meld.kind_tile().kind();
                if (27..=33).contains(&kind) {
                    honor_triplets[kind - 27] = true;
                }
            }
        }
        let called_kind = called.kind();
        if (31..=33).contains(&called_kind) && honor_triplets[4..].iter().all(|&v| v) {
            self.pao_daisangen[actor as usize] = Some(from);
        }
        if (27..=30).contains(&called_kind) && honor_triplets[..4].iter().all(|&v| v) {
            self.pao_daisuushi[actor as usize] = Some(from);
        }
    }

    /// Exhaustive draw settlement (tenpai check for noten penalties only).
    pub fn ryukyoku(&mut self) -> KyokuOutcome {
        let tenpai = self
            .players
            .iter()
            .filter(|p| p.is_tenpai())
            .map(|p| p.seat)
            .collect();
        let winners: Vec<u8> =
            if self.rule_profile.allows_nagashi_mangan() && self.wall.live_remaining() == 0 {
                self.players
                    .iter()
                    .filter(|player| player.is_nagashi_mangan_eligible())
                    .map(|player| player.seat)
                    .collect()
            } else {
                Vec::new()
            };
        if !self.terminal {
            self.log.push(flytable_event::Event4p::Ryukyoku {
                deltas: Some([0; SEATS]),
            });
            self.terminal = true;
        }
        if winners.is_empty() {
            KyokuOutcome::Ryukyoku { tenpai }
        } else {
            KyokuOutcome::NagashiMangan { winners, tenpai }
        }
    }

    /// Whether the hand has ended. After that, public progression APIs reject or do nothing.
    pub const fn is_terminal(&self) -> bool {
        self.terminal
    }

    pub fn can_kyuushu_kyuuhai(&self, seat: u8) -> bool {
        !self.terminal
            && self.is_kyuushu_kyuuhai_window(seat)
            && yaochuu_kind_count(&self.players[seat as usize].hand) >= 9
    }

    fn is_kyuushu_kyuuhai_window(&self, seat: u8) -> bool {
        if seat != self.turn || self.last_discard.is_some() {
            return false;
        }
        if self.players[seat as usize].hand.len() % 3 != 2 {
            return false;
        }
        if self.players.iter().any(|p| !p.melds.is_empty()) {
            return false;
        }
        let discards = self.players.iter().map(|p| p.discards.len()).sum::<usize>();
        let first_turn_offset = (seat + SEATS as u8 - self.oya) % SEATS as u8;
        discards == first_turn_offset as usize
    }

    fn is_suufon_renda(&self) -> bool {
        if self.players.iter().any(|p| !p.melds.is_empty()) {
            return false;
        }
        let first_discards: Option<Vec<Tile>> = self
            .players
            .iter()
            .map(|p| match p.discards.as_slice() {
                [first] => Some(*first),
                _ => None,
            })
            .collect();
        let Some(first_discards) = first_discards else {
            return false;
        };
        let first = first_discards[0];
        first.is_wind()
            && first_discards
                .iter()
                .all(|tile| tile.kind() == first.kind())
    }

    fn clear_all_ippatsu(&mut self) {
        for player in &mut self.players {
            player.ippatsu = false;
        }
    }
}

fn remove_kind_tiles(hand: &mut Vec<Tile>, kind: usize, count: usize) -> Option<Vec<Tile>> {
    let mut removed = Vec::with_capacity(count);
    let mut i = 0;
    while i < hand.len() && removed.len() < count {
        if hand[i].kind() == kind {
            removed.push(hand.remove(i));
        } else {
            i += 1;
        }
    }
    if removed.len() == count {
        Some(removed)
    } else {
        hand.extend(removed);
        None
    }
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

impl crate::product::GameServer for Board4p {
    fn seat_count(&self) -> usize {
        SEATS
    }
    fn view_for(&self, seat: u8) -> SeatView {
        Board4p::view_for(self, seat)
    }
    fn current_turn(&self) -> u8 {
        self.turn
    }
    fn legal_reactions(&self, seat: u8) -> Vec<crate::action::ReactionAction> {
        Board4p::legal_reactions(self, seat)
    }
}
