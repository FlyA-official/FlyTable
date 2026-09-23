//! Match log replay cursor, the basis for game review.
//!
//! Stepping back is done by replaying from the start of the hand, not by reversing
//! events. Reversal would need undo logic for every event (restoring a hand after a
//! pon, taking back dora), which amounts to a second rules implementation. Seeking to
//! any position replays from the hand's first event; a hand has at most a few
//! hundred events, so this takes microseconds.
//!
//! [`ReplayState`] folds events into what a review UI renders: each seat's closed
//! hand, discards and melds, dora indicators, scores, honba and riichi sticks. It
//! does not check legality, score hands or detect tenpai; that is the job of the
//! engine and `flytable-maintainer`. This keeps the cursor from ever disagreeing
//! with the engine.

use std::collections::BTreeMap;

use flytable_event::matchlog::{MatchlogEvent, MatchlogMeld, MeldKind, TileRef};

/// Observable state of one seat.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SeatState {
    /// Closed hand without melds: 14 tiles after a draw, 13 after a discard.
    pub concealed: Vec<TileRef>,
    /// Discards in order.
    pub discards: Vec<TileRef>,
    /// Indices of discards that were called, drawn sideways or greyed out in a review UI.
    pub called_from_river: Vec<usize>,
    pub melds: Vec<MatchlogMeld>,
    /// Riichi declared and accepted.
    pub riichi: bool,
    /// Index of the riichi declaration tile in the discards.
    pub riichi_discard_idx: Option<usize>,
    pub score: i32,
}

/// Observable state of one hand.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReplayState {
    pub seats: usize,
    pub bakaze: Option<flytable_core::tile::Tile>,
    pub kyoku: u8,
    pub honba: u8,
    pub kyotaku: u8,
    pub oya: u8,
    pub seat: Vec<SeatState>,
    /// Revealed dora indicators, including the initial one.
    pub dora_indicators: Vec<TileRef>,
    /// Ura dora indicators (visible only after a win is revealed).
    pub ura_indicators: Vec<TileRef>,
    /// Inside a robbing window.
    pub robbery_open: bool,
    /// Whether the hand has ended.
    pub finished: bool,
    /// Seats currently disconnected.
    pub disconnected: Vec<bool>,
}

fn remove_one(hand: &mut Vec<TileRef>, t: &TileRef) {
    // Remove by physical identity when present, otherwise by kind.
    if let Some(pos) = t
        .physical_id
        .and_then(|id| hand.iter().position(|x| x.physical_id == Some(id)))
        .or_else(|| hand.iter().position(|x| x.tile == t.tile))
    {
        hand.remove(pos);
    }
}

impl ReplayState {
    /// Applies one event. Only folds observable state; no rulings.
    pub fn apply(&mut self, ev: &MatchlogEvent) {
        match ev {
            MatchlogEvent::StartMatch { seats, .. } => {
                self.seats = usize::from(*seats);
                self.seat = vec![SeatState::default(); self.seats];
                self.disconnected = vec![false; self.seats];
            }
            MatchlogEvent::StartKyoku {
                bakaze,
                kyoku,
                honba,
                kyotaku,
                oya,
                scores,
                haipai,
                dora_marker,
            } => {
                let seats = haipai.len();
                self.seats = seats;
                self.bakaze = Some(*bakaze);
                self.kyoku = *kyoku;
                self.honba = *honba;
                self.kyotaku = *kyotaku;
                self.oya = *oya;
                self.dora_indicators = vec![*dora_marker];
                self.ura_indicators.clear();
                self.robbery_open = false;
                self.finished = false;
                // Disconnects persist across hands until reconnect, so they are not reset.
                if self.disconnected.len() != seats {
                    self.disconnected = vec![false; seats];
                }
                self.seat = haipai
                    .iter()
                    .enumerate()
                    .map(|(i, h)| SeatState {
                        concealed: h.clone(),
                        score: scores.get(i).copied().unwrap_or(0),
                        ..SeatState::default()
                    })
                    .collect();
            }
            MatchlogEvent::Tsumo { actor, pai, .. }
            | MatchlogEvent::DealerOpening { actor, pai } => {
                if let Some(s) = self.seat.get_mut(usize::from(*actor)) {
                    s.concealed.push(*pai);
                }
            }
            MatchlogEvent::Dahai {
                actor,
                pai,
                riichi_declare,
                ..
            } => {
                let declare = *riichi_declare;
                if let Some(s) = self.seat.get_mut(usize::from(*actor)) {
                    remove_one(&mut s.concealed, pai);
                    s.discards.push(*pai);
                    if declare {
                        s.riichi_discard_idx = Some(s.discards.len() - 1);
                    }
                }
            }
            MatchlogEvent::DealerOpeningDahai { actor, pai } => {
                let declare = false;
                if let Some(s) = self.seat.get_mut(usize::from(*actor)) {
                    remove_one(&mut s.concealed, pai);
                    s.discards.push(*pai);
                    if declare {
                        s.riichi_discard_idx = Some(s.discards.len() - 1);
                    }
                }
            }
            MatchlogEvent::Call { actor, meld } => self.apply_call(*actor, meld),
            MatchlogEvent::Dora { marker, .. } => self.dora_indicators.push(*marker),
            MatchlogEvent::RobberyWindow { edge, .. } => {
                self.robbery_open = matches!(edge, flytable_event::matchlog::WindowEdge::Open);
            }
            MatchlogEvent::ReachAccepted { actor } => {
                if let Some(s) = self.seat.get_mut(usize::from(*actor)) {
                    s.riichi = true;
                    s.score -= 1000;
                }
                self.kyotaku += 1;
            }
            MatchlogEvent::PlatformDisconnect { seat } => {
                if let Some(d) = self.disconnected.get_mut(usize::from(*seat)) {
                    *d = true;
                }
            }
            MatchlogEvent::PlatformReconnect { seat } => {
                if let Some(d) = self.disconnected.get_mut(usize::from(*seat)) {
                    *d = false;
                }
            }
            MatchlogEvent::Hora(b) => {
                self.finished = true;
                self.ura_indicators = b.ura_markers.clone();
                for (i, s) in self.seat.iter_mut().enumerate() {
                    s.score += b.base_deltas.get(i).copied().unwrap_or(0)
                        + b.honba_deltas.get(i).copied().unwrap_or(0)
                        + b.kyotaku_deltas.get(i).copied().unwrap_or(0);
                }
                // Riichi sticks collected by the winner.
                if b.kyotaku_deltas.iter().any(|x| *x > 0) {
                    self.kyotaku = 0;
                }
            }
            MatchlogEvent::Ryukyoku(b) => {
                self.finished = true;
                for (i, s) in self.seat.iter_mut().enumerate() {
                    s.score += b.base_deltas.get(i).copied().unwrap_or(0)
                        + b.honba_deltas.get(i).copied().unwrap_or(0);
                }
                self.kyotaku = b.kyotaku_carry;
            }
            _ => {}
        }
    }

    fn apply_call(&mut self, actor: u8, meld: &MatchlogMeld) {
        // Mark the called tile in the source seat's discards.
        if let (Some(from), Some(_claimed)) = (meld.from, meld.claimed)
            && !matches!(meld.kind, MeldKind::Kakan)
            && let Some(src) = self.seat.get_mut(usize::from(from))
            && !src.discards.is_empty()
        {
            let idx = src.discards.len() - 1;
            src.called_from_river.push(idx);
        }
        let Some(s) = self.seat.get_mut(usize::from(actor)) else {
            return;
        };
        match meld.kind {
            MeldKind::Kakan => {
                // Added kan: remove the added tile from hand and upgrade the pon.
                if let Some(added) = meld.claimed {
                    remove_one(&mut s.concealed, &added);
                }
                if let Some(pos) = s.melds.iter().position(|m| {
                    m.kind == MeldKind::Pon
                        && m.claimed.map(|c| c.tile.deaka()) == meld.claimed.map(|c| c.tile.deaka())
                }) {
                    s.melds[pos] = meld.clone();
                } else {
                    s.melds.push(meld.clone());
                }
            }
            MeldKind::Kita => {
                // Nukidora: move the North out of hand into its own meld entry (counts for han).
                if let Some(n) = meld.claimed {
                    remove_one(&mut s.concealed, &n);
                }
                s.melds.push(meld.clone());
            }
            _ => {
                for c in &meld.consumed {
                    remove_one(&mut s.concealed, c);
                }
                s.melds.push(meld.clone());
            }
        }
    }
}

/// Range of one hand in the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KyokuSpan {
    /// Index of `StartKyoku`.
    pub start: usize,
    /// Index of the hand's last event (inclusive).
    pub end: usize,
}

/// Hand index over a stream plus a seekable cursor.
#[derive(Debug, Clone)]
pub struct ReplayCursor<'a> {
    stream: &'a [MatchlogEvent],
    spans: Vec<KyokuSpan>,
    /// Indices of header events (`StartMatch` etc.), replayed before any seek.
    prelude: Vec<usize>,
    kyoku_idx: usize,
    offset: usize,
    state: ReplayState,
}

impl<'a> ReplayCursor<'a> {
    /// Builds the index in one O(n) pass; seeking then only replays one hand.
    #[must_use]
    pub fn new(stream: &'a [MatchlogEvent]) -> Self {
        let mut spans: Vec<KyokuSpan> = Vec::new();
        let mut prelude = Vec::new();
        for (i, ev) in stream.iter().enumerate() {
            match ev {
                MatchlogEvent::StartKyoku { .. } => {
                    if let Some(last) = spans.last_mut() {
                        last.end = i - 1;
                    }
                    spans.push(KyokuSpan {
                        start: i,
                        end: stream.len().saturating_sub(1),
                    });
                }
                _ if spans.is_empty() => prelude.push(i),
                _ => {}
            }
        }
        let mut c = Self {
            stream,
            spans,
            prelude,
            kyoku_idx: 0,
            offset: 0,
            state: ReplayState::default(),
        };
        c.rebuild();
        c
    }

    #[must_use]
    pub fn kyoku_count(&self) -> usize {
        self.spans.len()
    }

    /// Number of events that can be stepped through in the current hand.
    ///
    /// `offset` counts events applied after `StartKyoku`, so position `(k, 0)` is the
    /// start of hand k with tiles dealt and no actions yet.
    #[must_use]
    pub fn kyoku_len(&self) -> usize {
        self.spans
            .get(self.kyoku_idx)
            .map_or(0, |s| s.end.saturating_sub(s.start))
    }

    #[must_use]
    pub fn position(&self) -> (usize, usize) {
        (self.kyoku_idx, self.offset)
    }

    #[must_use]
    pub fn state(&self) -> &ReplayState {
        &self.state
    }

    /// The `offset`-th event of the hand, counted after `StartKyoku`.
    #[must_use]
    pub fn peek_at(&self, offset: usize) -> Option<&'a MatchlogEvent> {
        let span = self.spans.get(self.kyoku_idx)?;
        self.stream.get(span.start + 1 + offset)
    }

    /// Event at the current position, i.e. the next event to be applied.
    #[must_use]
    pub fn peek(&self) -> Option<&'a MatchlogEvent> {
        self.peek_at(self.offset)
    }

    /// Seeks to just before event `offset` of hand `kyoku` by replaying from the start of the hand.
    pub fn seek(&mut self, kyoku: usize, offset: usize) {
        self.kyoku_idx = kyoku.min(self.spans.len().saturating_sub(1));
        self.offset = offset.min(self.kyoku_len());
        self.rebuild();
    }

    /// Steps forward. Stops at the end of the hand and returns `false` instead of moving
    /// into the next hand.
    ///
    /// If `step` rolled over into the next hand, n steps would no longer equal seeking to
    /// event n, and a review progress bar would jump across hands. Use
    /// [`ReplayCursor::next_kyoku`] to change hands.
    pub fn step_forward(&mut self) -> bool {
        if self.offset < self.kyoku_len() {
            if let Some(ev) = self.peek() {
                self.state.apply(ev);
            }
            self.offset += 1;
            true
        } else {
            false
        }
    }

    /// Moves to the start of the next hand. Returns `false` on the last hand.
    pub fn next_kyoku(&mut self) -> bool {
        if self.kyoku_idx + 1 < self.spans.len() {
            self.seek(self.kyoku_idx + 1, 0);
            true
        } else {
            false
        }
    }

    /// Moves to the start of the previous hand.
    pub fn prev_kyoku(&mut self) -> bool {
        if self.kyoku_idx > 0 {
            self.seek(self.kyoku_idx - 1, 0);
            true
        } else {
            false
        }
    }

    /// Steps back (replays to the previous position). Stops at the start of the hand,
    /// like `step_forward`.
    pub fn step_back(&mut self) -> bool {
        if self.offset > 0 {
            self.seek(self.kyoku_idx, self.offset - 1);
            true
        } else {
            false
        }
    }

    /// Branches from the current position: a copy of the state and the events so far,
    /// for exploring an alternative play. Does not affect the main line.
    #[must_use]
    pub fn fork(&self) -> (ReplayState, &'a [MatchlogEvent]) {
        let span = self.spans[self.kyoku_idx];
        (
            self.state.clone(),
            &self.stream[span.start + 1..span.start + 1 + self.offset],
        )
    }

    /// Indices of every decision point in the hand (discard, call, riichi, win, draw declaration).
    #[must_use]
    pub fn decision_offsets(&self) -> Vec<usize> {
        let span = self.spans[self.kyoku_idx];
        (0..span.end.saturating_sub(span.start))
            .filter(|o| {
                matches!(
                    self.stream.get(span.start + 1 + o),
                    Some(MatchlogEvent::Dahai { .. })
                        | Some(MatchlogEvent::Call { .. })
                        | Some(MatchlogEvent::KyuushuDeclare { .. })
                )
            })
            .collect()
    }

    fn rebuild(&mut self) {
        let mut st = ReplayState::default();
        for i in &self.prelude {
            st.apply(&self.stream[*i]);
        }
        // Disconnects carry over: replay platform events from earlier hands.
        for s in self.spans.iter().take(self.kyoku_idx) {
            for ev in &self.stream[s.start..=s.end] {
                if matches!(
                    ev,
                    MatchlogEvent::PlatformDisconnect { .. }
                        | MatchlogEvent::PlatformReconnect { .. }
                ) {
                    st.apply(ev);
                }
            }
        }
        if let Some(span) = self.spans.get(self.kyoku_idx) {
            // `StartKyoku` is always applied, so `(k, 0)` is the start of the hand with tiles dealt.
            st.apply(&self.stream[span.start]);
            for ev in &self.stream[span.start + 1..span.start + 1 + self.offset] {
                st.apply(ev);
            }
        }
        self.state = st;
    }
}

/// Per-seat summary of a hand (discards, calls, riichi).
#[must_use]
pub fn kyoku_summary(state: &ReplayState) -> BTreeMap<usize, (usize, usize, bool)> {
    state
        .seat
        .iter()
        .enumerate()
        .map(|(i, s)| (i, (s.discards.len(), s.melds.len(), s.riichi)))
        .collect()
}
