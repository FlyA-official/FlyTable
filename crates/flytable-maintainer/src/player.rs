//! Private state of one seat, part of the authoritative table state.
//!
//! Shared by 4-player and 3-player. Holds the true hand, melds, discards, riichi
//! state and so on. Seat views are projected from it by the table layer (own tiles
//! in full, other players as counts only), so hidden hands never leak.

use flytable_core::agari;
use flytable_core::hand::TileCounts;
use flytable_core::meld::Meld;
use flytable_core::shanten;
use flytable_core::tile::Tile;

/// Private state of a seat.
#[derive(Debug, Clone)]
pub struct PlayerState {
    pub seat: u8,
    /// 3-player: shanten and waits must not use 1m/9m as sequence ends.
    pub is_sanma: bool,
    /// Closed hand (actual tiles including red fives).
    pub hand: Vec<Tile>,
    /// Whether the last tile in hand was actually just drawn. `None` when discarding after chi/pon.
    pub drawn_tile: Option<Tile>,
    pub melds: Vec<Meld>,
    /// Discards in order.
    pub discards: Vec<Tile>,
    /// Indices of discards called by other players. They stay in the history, but void nagashi mangan.
    pub called_discards: Vec<usize>,
    pub riichi: bool,
    /// Double riichi (declared on the first turn with no calls before it).
    pub double_riichi: bool,
    /// Index of the riichi declaration in the discards.
    pub riichi_discard_idx: Option<usize>,
    /// Ippatsu still possible.
    pub ippatsu: bool,
    /// Kinds forbidden for the discard right after chi/pon (kuikae); cleared after that discard.
    pub kuikae_forbidden: Vec<usize>,
    /// Closed hand (no melds other than closed kans).
    pub menzen: bool,
    /// Discard furiten: a current wait is in the own discards.
    pub furiten: bool,
    /// Temporary furiten: passed on a ron, lasts until the seat's next discard.
    pub temporary_furiten: bool,
    /// Riichi furiten: passed on a ron after riichi; no ron for the rest of the hand (tsumo still allowed).
    pub riichi_furiten: bool,
}

impl PlayerState {
    pub fn new(seat: u8, haipai: [Tile; 13]) -> Self {
        Self {
            seat,
            is_sanma: false,
            hand: haipai.to_vec(),
            drawn_tile: None,
            melds: Vec::new(),
            discards: Vec::new(),
            called_discards: Vec::new(),
            riichi: false,
            double_riichi: false,
            riichi_discard_idx: None,
            ippatsu: false,
            kuikae_forbidden: Vec::new(),
            menzen: true,
            furiten: false,
            temporary_furiten: false,
            riichi_furiten: false,
        }
    }

    /// Builds a 3-player seat state.
    pub fn new_sanma(seat: u8, haipai: [Tile; 13]) -> Self {
        Self {
            is_sanma: true,
            ..Self::new(seat, haipai)
        }
    }

    /// Ron is blocked by any furiten (discard, temporary or riichi).
    pub fn is_furiten_blocked(&self) -> bool {
        self.furiten || self.temporary_furiten || self.riichi_furiten
    }

    /// Records a passed ron: riichi furiten when in riichi, otherwise temporary furiten.
    pub fn note_missed_ron(&mut self) {
        if self.riichi {
            self.riichi_furiten = true;
        } else {
            self.temporary_furiten = true;
        }
    }

    /// Number of melds (chi, pon, kan count one each; nukidora does not count).
    pub fn meld_set_count(&self) -> u8 {
        self.melds
            .iter()
            .filter(|m| !matches!(m, Meld::Nukidora { .. }))
            .count() as u8
    }

    /// Closed-hand counts, red fives folded.
    pub fn hand_counts(&self) -> TileCounts {
        TileCounts::from_tiles(self.hand.iter().copied())
    }

    pub fn shanten(&self) -> i8 {
        if self.is_sanma {
            shanten::shanten_3p(&self.hand_counts(), self.meld_set_count())
        } else {
            shanten::shanten(&self.hand_counts(), self.meld_set_count())
        }
    }

    pub fn is_tenpai(&self) -> bool {
        self.shanten() == shanten::TENPAI
    }

    /// Winning tile kinds when tenpai.
    pub fn waits(&self) -> Vec<usize> {
        if self.is_sanma {
            agari::winning_tiles_3p(&self.hand_counts(), self.meld_set_count())
        } else {
            agari::winning_tiles(&self.hand_counts(), self.meld_set_count())
        }
    }

    /// Draws a tile. Temporary furiten clears on the seat's next discard, not on the draw (Tenhou).
    pub fn draw(&mut self, tile: Tile) {
        self.hand.push(tile);
        self.drawn_tile = Some(tile);
    }

    /// Removes a discarded tile from hand. Returns whether it was present.
    pub fn discard(&mut self, tile: Tile) -> bool {
        if self.kuikae_forbidden.contains(&tile.kind()) {
            return false;
        }
        if let Some(pos) = self.hand.iter().position(|&t| t == tile) {
            self.hand.remove(pos);
            self.drawn_tile = None;
            self.discards.push(tile);
            self.temporary_furiten = false;
            self.kuikae_forbidden.clear();
            self.recompute_furiten();
            true
        } else {
            false
        }
    }

    /// Discards with an explicit tsumogiri/hand origin and rejects public events that
    /// contradict it.
    pub fn discard_with_source(&mut self, tile: Tile, tsumogiri: bool) -> Result<(), String> {
        if self.kuikae_forbidden.contains(&tile.kind()) {
            return Err(format!(
                "kuikae forbids discarding {tile} right after the call"
            ));
        }
        let Some(last) = self.hand.last().copied() else {
            return Err("cannot discard from an empty hand".into());
        };
        if tsumogiri {
            let Some(drawn) = self.drawn_tile else {
                return Err("no drawn tile to discard as tsumogiri".into());
            };
            if drawn != tile || last != tile {
                return Err(format!(
                    "tsumogiri must discard the drawn {drawn}, not {tile}"
                ));
            }
            self.hand.pop();
        } else {
            if self.drawn_tile == Some(last) {
                let last_idx = self.hand.len() - 1;
                if let Some(pos) = self.hand[..last_idx].iter().position(|&t| t == tile) {
                    self.hand.remove(pos);
                } else if last == tile {
                    return Err(format!("a hand discard cannot be the only drawn {tile}"));
                } else {
                    return Err(format!("{tile} is not in hand"));
                }
            } else if let Some(pos) = self.hand.iter().position(|&t| t == tile) {
                self.hand.remove(pos);
            } else {
                return Err(format!("{tile} is not in hand"));
            }
        }
        self.drawn_tile = None;
        self.discards.push(tile);
        self.temporary_furiten = false;
        self.kuikae_forbidden.clear();
        self.recompute_furiten();
        Ok(())
    }

    /// Discards from the dealer's 14-tile opening hand, without a tsumogiri/hand origin.
    pub fn discard_dealer_opening(&mut self, tile: Tile) -> Result<(), String> {
        if self.drawn_tile.is_some() {
            return Err("the dealer opening discard cannot have a drawn_tile".into());
        }
        let pos = self
            .hand
            .iter()
            .position(|candidate| *candidate == tile)
            .ok_or_else(|| format!("{tile} is not in hand"))?;
        self.hand.remove(pos);
        self.discards.push(tile);
        self.temporary_furiten = false;
        self.kuikae_forbidden.clear();
        self.recompute_furiten();
        Ok(())
    }

    /// Marks the latest discard as called (chi / pon / open kan). Idempotent.
    pub fn mark_latest_discard_called(&mut self) -> Result<(), String> {
        let index = self
            .discards
            .len()
            .checked_sub(1)
            .ok_or("no discard to mark as called")?;
        if !self.called_discards.contains(&index) {
            self.called_discards.push(index);
        }
        Ok(())
    }

    /// Nagashi mangan eligibility (Tenhou): at least one discard, all terminals or
    /// honors, and none called by another player.
    pub fn is_nagashi_mangan_eligible(&self) -> bool {
        !self.discards.is_empty()
            && self.discards.iter().all(|tile| tile.is_yaochuu())
            && self.called_discards.is_empty()
    }

    /// Whether adding `win_tile` completes the hand (yaku not checked).
    pub fn would_agari_with(&self, win_tile: Tile) -> bool {
        let mut c = self.hand_counts();
        c.add(win_tile, 1);
        if self.is_sanma {
            agari::is_agari_3p(&c, self.meld_set_count())
        } else {
            agari::is_agari(&c, self.meld_set_count())
        }
    }

    /// Recomputes discard furiten: whether any current wait is in the own discards.
    pub fn recompute_furiten(&mut self) {
        let waits = self.waits();
        self.furiten = self.discards.iter().any(|d| waits.contains(&d.kind()));
    }
}
