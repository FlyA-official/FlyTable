//! Integration interfaces for products.
//!
//! One trait per role, plus in-crate implementations showing each can be expressed;
//! products implement them as needed.
//!
//! - [`GameServer`] - holds perfect information, sends each seat its imperfect view,
//!   receives and adjudicates actions.
//! - [`GameClient`] - only gets a view and the legal actions, returns a choice, never
//!   sees hidden state.
//! - [`MirrorCenter`] - consumes a standard event stream, mirrors the table, runs rule
//!   calculations and decides when to request inference. Bridging external signals to
//!   standard events is outside FlyTable.
//!
//! All interfaces use seat-count-independent [`SeatView`] / [`TurnAction`]; the
//! implementation chooses whether to drive `board4p::Board4p` or `board3p::Board3p`.

use crate::action::{PendingRobbery, ReactionAction, RobberyKind, TurnAction};
use crate::view::SeatView;

/// Server interface: the sole holder of the true game state.
///
/// Implementers hold perfect information (wall and every hand), expose a seat's
/// imperfect view only through `view_for`, and receive, adjudicate and apply actions
/// through `apply_*`.
pub trait GameServer {
    /// Number of seats (4 or 3).
    fn seat_count(&self) -> usize;

    /// Imperfect-information view for a seat.
    fn view_for(&self, seat: u8) -> SeatView;

    /// Seat whose turn it is.
    fn current_turn(&self) -> u8;

    /// Legal responses of a seat to the current discard.
    fn legal_reactions(&self, seat: u8) -> Vec<ReactionAction>;
}

/// Client interface: plays from one seat's perspective.
///
/// Implementers never see hidden state. They receive a [`SeatView`] (opponent hands
/// as counts only) and the legal actions and return a choice. Humans and browser
/// shells both go through this.
pub trait GameClient {
    /// Chooses a turn action for the given view.
    fn choose_turn(&mut self, view: &SeatView, legal: &[TurnAction]) -> TurnAction;

    /// Chooses a response (pon / kan / ron / pass) for the given view and legal responses.
    fn choose_reaction(&mut self, view: &SeatView, legal: &[ReactionAction]) -> ReactionAction {
        let _ = (view, legal);
        ReactionAction::Pass
    }
}

/// Mirror interface: consumes a standard event stream, mirrors state and runs rule calculations.
///
/// `E` is the standard event type ([`flytable_event::Event4p`] or `Event3p`). An
/// external bridge translates third-party signals into it; FlyTable only accepts this
/// one input. Implementers advance the mirror from events and expose seat views and
/// rule results.
///
/// The rule calculations ([`legal_turn`](MirrorCenter::legal_turn) /
/// [`shanten`](MirrorCenter::shanten) / [`furiten`](MirrorCenter::furiten)) are
/// default methods over the [`SeatView`] from [`mirror_view`](MirrorCenter::mirror_view),
/// so both implementations share them.
pub trait MirrorCenter<E> {
    /// Feeds one standard event. Rejects invalid or inconsistent events.
    fn feed(&mut self, event: E) -> Result<(), String>;

    /// Current reconstruction quality. Only `Exact` allows normal inference on FlyTable's own legal actions.
    fn quality(&self) -> MirrorQuality;

    /// Reason for the latest downgrade or rejection.
    fn last_issue(&self) -> Option<&MirrorIssue>;

    /// Explicitly marks an event gap or missing information reported by the adapter.
    /// Normal inference fails closed until resync.
    fn mark_degraded(&mut self, code: &'static str, detail: impl Into<String>);

    /// Mirrored view for a seat (same shape as the server's).
    fn mirror_view(&self, seat: u8) -> SeatView;

    /// Whether the seat should request inference now (its turn, or it may respond).
    /// The decision itself belongs to the model; this is only the trigger.
    fn should_infer(&self, seat: u8) -> bool;

    /// Turn actions the rules allow the seat.
    ///
    /// Equivalent to [`legal_turn_actions`](crate::legal::legal_turn_actions)`(&mirror_view(seat))`.
    /// The own hand is fully visible, so turn actions can be computed from the mirror
    /// without an authoritative board.
    fn legal_turn(&self, seat: u8) -> Vec<TurnAction> {
        crate::legal::legal_turn_actions(&self.mirror_view(seat))
    }

    /// Shanten of the seat (red fives folded, counting melds). Tenpai is 0, complete is -1.
    fn shanten(&self, seat: u8) -> i8 {
        crate::calc::my_shanten(&self.mirror_view(seat))
    }

    /// Whether the seat is in discard furiten (own discards intersect current waits).
    /// Temporary and riichi furiten are not included (see
    /// [`permanent_furiten`](crate::legal::permanent_furiten)).
    fn furiten(&self, seat: u8) -> bool {
        crate::legal::permanent_furiten(&self.mirror_view(seat))
    }

    /// Legal responses of the seat to the latest discard (ron / pon / open kan / chi).
    ///
    /// Equivalent to [`legal_reactions`](crate::legal::legal_reactions)`(&mirror_view(seat))`,
    /// so a mirror can enumerate responses without relying on the platform's list. Ron
    /// includes shape, yaku and furiten checks.
    fn legal_reactions(&self, seat: u8) -> Vec<ReactionAction> {
        crate::legal::legal_reactions(&self.mirror_view(seat))
    }
}

use crate::player::PlayerState;
use crate::view::{OpponentView, SeatViewError, SelfView};
use flytable_core::meld::Meld;
use flytable_core::rules::{RiichiRuleProfile, RonResolution};
use flytable_core::tile::Tile;
use flytable_event::{Event3p, Event4p};

/// Quality of the passive reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorQuality {
    Exact,
    Degraded,
    Unsafe,
}

/// Relation between FlyTable's legal set and the actions a third-party platform currently offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionSetRelation {
    Exact,
    /// The platform offers fewer (common with UI or timing restrictions).
    PlatformNarrower,
    /// The platform offers actions FlyTable does not accept.
    PlatformWider,
    /// Each side has exclusive actions, but there is a safe intersection.
    Divergent,
    /// Both non-empty with no intersection.
    Disjoint,
    /// At least one side has no actions; no normal decision boundary.
    Missing,
}

/// Action boundary for mirror mode with two authorities.
///
/// `inference_actions` always come from the intersection, and normal model inference
/// is only allowed on an exact mirror. A degraded mirror stops model inference and
/// keeps only `fallback_actions` the platform has shown to be executable. An unsafe
/// mirror has neither and requires resync.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionReconciliation<A> {
    pub quality: MirrorQuality,
    pub relation: ActionSetRelation,
    pub common: Vec<A>,
    pub flytable_only: Vec<A>,
    pub platform_only: Vec<A>,
    pub inference_actions: Vec<A>,
    pub fallback_actions: Vec<A>,
}

impl<A> ActionReconciliation<A> {
    pub fn normal_inference_allowed(&self) -> bool {
        self.quality == MirrorQuality::Exact && !self.inference_actions.is_empty()
    }

    /// Deterministic safe fallback: the first action allowed at this quality level.
    pub fn deterministic_fallback(&self) -> Option<&A> {
        self.fallback_actions.first()
    }
}

/// Compares FlyTable's authoritative legal set with the platform's current actions.
///
/// Adapters convert platform actions into the same standard action type first. Raw
/// platform actions and failed conversions belong in adapter diagnostics; do not
/// fuzzy-match to force an intersection.
pub fn reconcile_action_sets<A: Clone + PartialEq>(
    quality: MirrorQuality,
    flytable: &[A],
    platform: &[A],
) -> ActionReconciliation<A> {
    let flytable = dedup_actions(flytable);
    let platform = dedup_actions(platform);
    let common: Vec<A> = flytable
        .iter()
        .filter(|action| platform.contains(action))
        .cloned()
        .collect();
    let flytable_only: Vec<A> = flytable
        .iter()
        .filter(|action| !platform.contains(action))
        .cloned()
        .collect();
    let platform_only: Vec<A> = platform
        .iter()
        .filter(|action| !flytable.contains(action))
        .cloned()
        .collect();
    let relation = if flytable.is_empty() || platform.is_empty() {
        ActionSetRelation::Missing
    } else if common.is_empty() {
        ActionSetRelation::Disjoint
    } else if flytable_only.is_empty() && platform_only.is_empty() {
        ActionSetRelation::Exact
    } else if platform_only.is_empty() {
        ActionSetRelation::PlatformNarrower
    } else if flytable_only.is_empty() {
        ActionSetRelation::PlatformWider
    } else {
        ActionSetRelation::Divergent
    };
    let inference_actions = if quality == MirrorQuality::Exact {
        common.clone()
    } else {
        Vec::new()
    };
    let fallback_actions = match quality {
        MirrorQuality::Exact => common.clone(),
        MirrorQuality::Degraded => platform.clone(),
        MirrorQuality::Unsafe => Vec::new(),
    };
    ActionReconciliation {
        quality,
        relation,
        common,
        flytable_only,
        platform_only,
        inference_actions,
        fallback_actions,
    }
}

fn dedup_actions<A: Clone + PartialEq>(actions: &[A]) -> Vec<A> {
    let mut unique = Vec::with_capacity(actions.len());
    for action in actions {
        if !unique.contains(action) {
            unique.push(action.clone());
        }
    }
    unique
}

/// Redacted, stable diagnostic for a mirror quality change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirrorIssue {
    pub event_index: u64,
    pub code: &'static str,
    pub detail: String,
}

/// Live wall at the start of a 4-player hand (136 - 4 x 13 dealt - 14 dead wall = 70).
/// Matches the usual `tiles_remaining` semantics of platforms.
const LIVE_WALL_4P: u32 = 70;
/// Live wall at the start of a 3-player hand (108 - 3 x 13 dealt - 14 dead wall = 55).
const LIVE_WALL_3P: u32 = 55;

/// Inventory of physical tiles the mirror has seen so far in the hand.
///
/// This records first reveals, not a scan of current zones: moving own tiles from
/// hand to discards or melds, or a called discard into a meld, is not counted twice;
/// another seat's hidden tile is added when first discarded or called. So even though
/// [`Meld`] only keeps a folded representative for kans, red five identities already
/// revealed in events are not lost.
#[derive(Debug, Clone, PartialEq, Eq)]
struct KnownTileInventory {
    normal: [u8; 34],
    red: [u8; 3],
}

impl Default for KnownTileInventory {
    fn default() -> Self {
        Self {
            normal: [0; 34],
            red: [0; 3],
        }
    }
}

impl KnownTileInventory {
    fn observe(
        &mut self,
        tile: Tile,
        seats: usize,
        profile: RiichiRuleProfile,
    ) -> Result<(), String> {
        if tile.is_unknown() {
            return Ok(());
        }
        let kind = tile.kind();
        if kind >= 34 {
            return Err(format!("known_tiles: invalid tile kind {kind}"));
        }
        if seats == 3 && kind < 9 && !matches!(kind, 0 | 8) {
            return Err(format!("known_tiles: invalid sanma tile {tile}"));
        }

        if tile.is_aka() {
            let suit = match kind {
                4 => 0,
                13 => 1,
                22 => 2,
                _ => return Err(format!("known_tiles: non-five tile marked red: {tile}")),
            };
            self.red[suit] = self.red[suit]
                .checked_add(1)
                .ok_or_else(|| format!("known_tiles: red count overflow for {tile}"))?;
        } else {
            self.normal[kind] = self.normal[kind]
                .checked_add(1)
                .ok_or_else(|| format!("known_tiles: normal count overflow for {tile}"))?;
        }
        self.validate_kind(kind, seats, profile)
    }

    fn observe_all(
        &mut self,
        tiles: impl IntoIterator<Item = Tile>,
        seats: usize,
        profile: RiichiRuleProfile,
    ) -> Result<(), String> {
        for tile in tiles {
            self.observe(tile, seats, profile)?;
        }
        Ok(())
    }

    fn validate_kind(
        &self,
        kind: usize,
        seats: usize,
        profile: RiichiRuleProfile,
    ) -> Result<(), String> {
        let red_fives = profile.red_fives(seats);
        let red_limit = match kind {
            4 => Some(red_fives.man),
            13 => Some(red_fives.pin),
            22 => Some(red_fives.sou),
            _ => None,
        };
        if let Some(red_limit) = red_limit {
            let suit = match kind {
                4 => 0,
                13 => 1,
                22 => 2,
                _ => unreachable!("red-five kind was matched above"),
            };
            let red = self.red[suit];
            let normal = self.normal[kind];
            if red > red_limit {
                return Err(format!(
                    "known_tiles: red five count {red} exceeds profile limit {red_limit} for kind {kind}"
                ));
            }
            let normal_limit = 4 - red_limit;
            if normal > normal_limit {
                return Err(format!(
                    "known_tiles: normal five count {normal} exceeds profile limit {normal_limit} for kind {kind}"
                ));
            }
            if red.saturating_add(normal) > 4 {
                return Err(format!(
                    "known_tiles: fifth physical copy of five kind {kind}"
                ));
            }
        } else if self.normal[kind] > 4 {
            return Err(format!("known_tiles: fifth physical copy of kind {kind}"));
        }
        Ok(())
    }
}

/// Accepted terminal result: a single tsumo, or one ron packet from one discard.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TerminalResultPacket {
    Tsumo {
        winner: u8,
        ura_markers: Vec<Tile>,
    },
    Ron {
        source: u8,
        robbery: Option<RobberyKind>,
        winners: Vec<u8>,
        max_winners: u8,
        /// Ron legality for the own seat, computed from the full hand while the response window is open.
        ///
        /// `None` means the mirror was not `Exact` at the time, so an incomplete state must
        /// not contradict the platform's authoritative result. The snapshot has to survive
        /// the first `Hora`: the first winner makes the mirror terminal, but later `Hora`s of
        /// a multi-ron packet still need the same window to judge the own seat.
        self_ron_legal_at_open: Option<bool>,
        ura_markers: Vec<Tile>,
    },
}

/// Removes one exact tile (red fives distinguished) from the hand. Returns whether it was present.
fn remove_tile(hand: &mut Vec<Tile>, tile: Tile) -> bool {
    if let Some(pos) = hand.iter().position(|&t| t == tile) {
        hand.remove(pos);
        return true;
    }
    false
}

/// Removes one unknown placeholder (when another seat reveals a tile).
fn remove_unknown(hand: &mut Vec<Tile>) -> bool {
    if let Some(pos) = hand.iter().position(|t| t.is_unknown()) {
        hand.remove(pos);
        return true;
    }
    false
}

#[derive(Debug, Clone)]
struct PendingReach {
    actor: u8,
    double_riichi: bool,
    /// Legal riichi discards frozen at `Reach` for the own seat; empty when the hand is unknown.
    legal_discards: Vec<TurnAction>,
}

/// Passive observation core shared by 4-player and 3-player.
///
/// Tracks public information for every seat and the true hand of the own seat (other
/// hands are [`Tile::unknown`] placeholders), and runs rule calculations on it. It
/// never holds the wall, only what events revealed, so it cannot contain other seats'
/// hidden state. Driven by [`Mirror4p`] / [`Mirror3p`]; all state changes are here.
#[derive(Clone)]
struct MirrorCore {
    seats: usize,
    rule_profile: RiichiRuleProfile,
    /// Own seat (true hand; other seats only follow public information).
    me: u8,
    bakaze: Tile,
    oya: u8,
    kyoku: u8,
    honba: u8,
    kyotaku: u8,
    scores: Vec<i32>,
    dora_indicators: Vec<Tile>,
    players: Vec<PlayerState>,
    turn: u8,
    last_discard: Option<(u8, Tile)>,
    /// Live wall remaining (start minus draws; replacement draws also arrive as `Tsumo`).
    tiles_left: u32,
    /// Whether any call, nukidora or kan has happened (closes the kyuushu kyuuhai window).
    any_call: bool,
    /// Whether the next `Tsumo` is a replacement draw (after a kan or nukidora).
    pending_rinshan_draw: bool,
    /// Whether the own seat's last draw was a replacement draw.
    last_draw_was_rinshan: bool,
    /// The dealer's 14-tile opening window (Mahjong Soul).
    dealer_opening: bool,
    /// The latest discard is in the own waits and the seat has not won yet; furiten is
    /// recorded when the response window closes.
    pending_missed_ron: bool,
    /// `Reach` received: legal discards at declaration time are bound, then `ReachAccepted` is awaited.
    pending_reach: Option<PendingReach>,
    /// Kan or nukidora declaration not yet closed by a replacement draw or a win.
    pending_robbery: Option<PendingRobbery>,
    /// Accepted terminal result: kind, source, robbed kind, winners and profile limits.
    terminal_packet: Option<TerminalResultPacket>,
    /// First-reveal inventory for the hand; enforces the room's red five configuration and the four-copy limit.
    known_tiles: KnownTileInventory,
    started: bool,
    terminal: bool,
    quality: MirrorQuality,
    issues: Vec<MirrorIssue>,
    event_index: u64,
}

impl MirrorCore {
    /// Four-kan draw condition (same as `Board4p` / `Board3p`): four kans by at least two
    /// players. Melds are public, so the mirror can compute it exactly.
    fn suukaikan_condition_met(&self) -> bool {
        if !self.rule_profile.allows_suukaikan() {
            return false;
        }
        let kan_counts: Vec<usize> = self
            .players
            .iter()
            .map(|player| player.melds.iter().filter(|meld| meld.is_kan()).count())
            .collect();
        let total: usize = kan_counts.iter().sum();
        let owners = kan_counts.iter().filter(|&&count| count > 0).count();
        total >= 4 && owners >= 2
    }

    fn new(seats: usize, me: u8, rule_profile: RiichiRuleProfile) -> Self {
        let mut core = Self {
            seats,
            rule_profile,
            me,
            bakaze: Tile::default(),
            oya: 0,
            kyoku: 1,
            honba: 0,
            kyotaku: 0,
            scores: vec![25000; seats],
            dora_indicators: Vec::new(),
            players: Vec::new(),
            turn: 0,
            last_discard: None,
            tiles_left: 0,
            any_call: false,
            pending_rinshan_draw: false,
            last_draw_was_rinshan: false,
            dealer_opening: false,
            pending_missed_ron: false,
            pending_reach: None,
            pending_robbery: None,
            terminal_packet: None,
            known_tiles: KnownTileInventory::default(),
            started: false,
            terminal: false,
            quality: MirrorQuality::Degraded,
            issues: Vec::new(),
            event_index: 0,
        };
        if me as usize >= seats {
            core.reject(
                "invalid_mirror_seat",
                format!("mirror seat {me} is outside 0..{seats}"),
            );
        }
        core
    }

    fn issue(&mut self, quality: MirrorQuality, code: &'static str, detail: impl Into<String>) {
        if self.quality != MirrorQuality::Unsafe || quality == MirrorQuality::Unsafe {
            self.quality = quality;
        }
        self.issues.push(MirrorIssue {
            event_index: self.event_index,
            code,
            detail: detail.into(),
        });
    }

    fn pending_reach_actor(&self) -> Option<u8> {
        self.pending_reach.as_ref().map(|pending| pending.actor)
    }

    fn awaiting_reach_discard(&self) -> bool {
        self.pending_reach.as_ref().is_some_and(
            |pending| !matches!(self.last_discard, Some((actor, _)) if actor == pending.actor),
        )
    }

    fn validate_reach_discard(&self, actor: u8, action: &TurnAction) -> Result<(), String> {
        let Some(pending) = &self.pending_reach else {
            return Ok(());
        };
        if pending.actor != actor {
            return Err(format!(
                "phase_error: waiting for seat {} riichi discard, got seat {actor}",
                pending.actor
            ));
        }
        if actor == self.me && !pending.legal_discards.contains(action) {
            return Err(format!(
                "illegal_reach_discard: declaration discard is not in the frozen legal set: {action:?}"
            ));
        }
        Ok(())
    }

    fn commit_reach_discard(&mut self, actor: u8) {
        let Some(pending) = self
            .pending_reach
            .as_ref()
            .filter(|pending| pending.actor == actor)
        else {
            return;
        };
        let player = &mut self.players[actor as usize];
        player.riichi = true;
        player.double_riichi = pending.double_riichi;
        player.ippatsu = true;
    }

    fn reject(&mut self, code: &'static str, detail: impl Into<String>) {
        self.issue(MirrorQuality::Unsafe, code, detail);
    }

    fn degrade(&mut self, code: &'static str, detail: impl Into<String>) {
        if self.quality != MirrorQuality::Unsafe {
            self.issue(MirrorQuality::Degraded, code, detail);
        }
    }

    fn ensure_actor(&self, actor: u8) -> Result<(), String> {
        if actor as usize >= self.seats {
            Err(format!(
                "invalid_actor: seat {actor} is outside 0..{}",
                self.seats
            ))
        } else {
            Ok(())
        }
    }

    fn ensure_active(&self) -> Result<(), String> {
        if !self.started {
            Err("phase_error: start_kyoku has not been observed".to_string())
        } else if self.terminal {
            Err("phase_error: kyoku is already terminal".to_string())
        } else {
            Ok(())
        }
    }

    fn validate_start_kyoku(
        &self,
        oya: u8,
        dora_marker: Tile,
        my_hand: &[Tile; 13],
        sanma: bool,
    ) -> Result<KnownTileInventory, String> {
        if self.me as usize >= self.seats {
            return Err(format!(
                "invalid_mirror_seat: seat {} is outside 0..{}",
                self.me, self.seats
            ));
        }
        if self.started && !self.terminal && self.quality != MirrorQuality::Unsafe {
            return Err("phase_error: start_kyoku received during an active kyoku".into());
        }
        self.ensure_actor(oya)?;
        if dora_marker.is_unknown() {
            return Err("start_kyoku: dora marker cannot be unknown".into());
        }
        if sanma != (self.seats == 3) {
            return Err("start_kyoku: variant/player count mismatch".into());
        }
        self.rule_profile.validate_for_players(self.seats)?;
        let mut known_tiles = KnownTileInventory::default();
        known_tiles.observe_all(my_hand.iter().copied(), self.seats, self.rule_profile)?;
        known_tiles.observe(dora_marker, self.seats, self.rule_profile)?;
        Ok(known_tiles)
    }

    fn max_ron_winners(&self) -> u8 {
        match self.rule_profile.ron_resolution() {
            RonResolution::AtamaHane => 1,
            RonResolution::Multiple => self.seats.saturating_sub(1) as u8,
            RonResolution::TripleRonAbortive => self.seats.saturating_sub(1).min(2) as u8,
        }
    }

    fn validate_robbery_result(&self, kind: RobberyKind) -> Result<(), String> {
        match kind {
            RobberyKind::Kakan => Ok(()),
            RobberyKind::Ankan if self.rule_profile.allows_kokushi_ankan_ron() => Ok(()),
            RobberyKind::Nukidora if self.rule_profile.allows_nukidora_ron() => Ok(()),
            RobberyKind::Ankan => {
                Err("hora: current profile does not allow robbing an ankan".into())
            }
            RobberyKind::Nukidora => {
                Err("hora: current profile does not allow robbing nukidora".into())
            }
        }
    }

    fn merge_ura_markers(
        &self,
        existing: &[Tile],
        incoming: Option<&[Tile]>,
    ) -> Result<(Vec<Tile>, KnownTileInventory), String> {
        let Some(incoming) = incoming.filter(|markers| !markers.is_empty()) else {
            return Ok((existing.to_vec(), self.known_tiles.clone()));
        };
        if !existing.is_empty() {
            if existing != incoming {
                return Err("hora: ura markers changed inside one result packet".into());
            }
            return Ok((existing.to_vec(), self.known_tiles.clone()));
        }
        let mut known_tiles = self.known_tiles.clone();
        known_tiles.observe_all(incoming.iter().copied(), self.seats, self.rule_profile)?;
        Ok((incoming.to_vec(), known_tiles))
    }

    fn hora(
        &mut self,
        actor: u8,
        target: u8,
        deltas: Option<&[i32]>,
        ura_markers: Option<&[Tile]>,
    ) -> Result<(), String> {
        if !self.started {
            return Err("phase_error: hora before start_kyoku".into());
        }
        self.ensure_actor(actor)?;
        self.ensure_actor(target)?;
        if let Some(deltas) = deltas
            && deltas.len() != self.seats
        {
            return Err("hora: delta width does not match seat count".into());
        }
        if let Some(markers) = ura_markers {
            if markers.is_empty() {
                return Err("hora: explicit ura marker list cannot be empty".into());
            }
            if !self.players[actor as usize].riichi {
                return Err("hora: explicit ura markers require a riichi winner".into());
            }
            if markers.len() != self.dora_indicators.len() {
                return Err(format!(
                    "hora: ura marker count {} does not match {} visible dora indicators",
                    markers.len(),
                    self.dora_indicators.len()
                ));
            }
        }

        let response = match (self.last_discard, self.pending_robbery) {
            (Some((source, _)), None) => Some((source, None)),
            (None, Some(pending)) => Some((pending.actor, Some(pending.kind))),
            (None, None) => None,
            (Some(_), Some(_)) => {
                return Err("hora: discard and robbery response windows overlap".into());
            }
        };
        let (next_packet, next_known_tiles, closes_response_window) = match self
            .terminal_packet
            .clone()
        {
            Some(TerminalResultPacket::Tsumo { .. }) => {
                return Err("hora: no result may follow a tsumo packet".into());
            }
            Some(TerminalResultPacket::Ron {
                source,
                robbery,
                mut winners,
                max_winners,
                self_ron_legal_at_open,
                ura_markers: existing_ura,
            }) => {
                if actor == target {
                    return Err("hora: tsumo cannot be mixed into a ron packet".into());
                }
                if target != source {
                    return Err("hora: target changed inside one ron packet".into());
                }
                if actor == source {
                    return Err("hora: ron source cannot also be a winner".into());
                }
                if winners.contains(&actor) {
                    return Err("hora: duplicate winner inside one ron packet".into());
                }
                if winners.len() >= max_winners as usize {
                    return Err(format!(
                        "hora: ron packet exceeds profile winner limit {max_winners}"
                    ));
                }
                let expected_limit = self.max_ron_winners();
                if max_winners != expected_limit {
                    return Err("hora: stored ron packet limit disagrees with profile".into());
                }
                if actor == self.me && self_ron_legal_at_open == Some(false) {
                    return Err(
                        "hora: exact mirror proves self ron was illegal at response-window open"
                            .into(),
                    );
                }
                let (merged_ura, known_tiles) =
                    self.merge_ura_markers(&existing_ura, ura_markers)?;
                winners.push(actor);
                (
                    TerminalResultPacket::Ron {
                        source,
                        robbery,
                        winners,
                        max_winners,
                        self_ron_legal_at_open,
                        ura_markers: merged_ura,
                    },
                    known_tiles,
                    false,
                )
            }
            None => {
                if self.terminal {
                    return Err("phase_error: additional hora outside a result packet".into());
                }
                let (merged_ura, known_tiles) = self.merge_ura_markers(&[], ura_markers)?;
                if actor == target {
                    if response.is_some() {
                        return Err("hora: tsumo cannot occur during a response window".into());
                    }
                    if actor != self.turn
                        || (self.players[actor as usize].drawn_tile.is_none()
                            && !(self.dealer_opening && actor == self.oya))
                    {
                        return Err("hora: tsumo requires a draw or dealer opening window".into());
                    }
                    if actor == self.me
                        && self.quality == MirrorQuality::Exact
                        && !crate::legal::legal_turn_actions(&self.mirror_view(actor))
                            .iter()
                            .any(|action| matches!(action, TurnAction::Tsumo))
                    {
                        return Err(
                            "hora: exact mirror proves self tsumo is not a legal action".into()
                        );
                    }
                    (
                        TerminalResultPacket::Tsumo {
                            winner: actor,
                            ura_markers: merged_ura,
                        },
                        known_tiles,
                        false,
                    )
                } else {
                    let Some((source, robbery)) = response else {
                        return Err("hora: ron has no pending discard or robbery source".into());
                    };
                    if target != source {
                        return Err(
                            "hora: target does not match the pending response source".into()
                        );
                    }
                    if actor == source {
                        return Err("hora: ron source cannot also be a winner".into());
                    }
                    if let Some(kind) = robbery {
                        self.validate_robbery_result(kind)?;
                    }
                    let self_ron_legal_at_open =
                        (self.quality == MirrorQuality::Exact).then(|| {
                            crate::legal::legal_reactions(&self.mirror_view(self.me))
                                .iter()
                                .any(|action| matches!(action, ReactionAction::Ron))
                        });
                    if actor == self.me && self_ron_legal_at_open == Some(false) {
                        return Err(
                            "hora: exact mirror proves self ron is not a legal reaction".into()
                        );
                    }
                    let max_winners = self.max_ron_winners();
                    (
                        TerminalResultPacket::Ron {
                            source,
                            robbery,
                            winners: vec![actor],
                            max_winners,
                            self_ron_legal_at_open,
                            ura_markers: merged_ura,
                        },
                        known_tiles,
                        true,
                    )
                }
            }
        };

        if self.pending_reach_actor() == Some(target) {
            let player = &mut self.players[target as usize];
            player.riichi = false;
            player.double_riichi = false;
            player.ippatsu = false;
            self.pending_reach = None;
        }
        if let Some(deltas) = deltas {
            for (score, delta) in self.scores.iter_mut().zip(deltas) {
                *score += *delta;
            }
        }
        self.pending_missed_ron = false;
        if closes_response_window {
            self.pending_robbery = None;
        }
        self.known_tiles = next_known_tiles;
        self.terminal_packet = Some(next_packet);
        self.terminal = true;
        Ok(())
    }

    fn ryukyoku(&mut self, deltas: Option<&[i32]>) -> Result<(), String> {
        self.ensure_active()?;
        if let Some(deltas) = deltas {
            if deltas.len() != self.seats {
                return Err("ryukyoku: delta width does not match seat count".into());
            }
            for (score, delta) in self.scores.iter_mut().zip(deltas) {
                *score += *delta;
            }
        }
        self.commit_missed_ron_if_pending();
        if let Some((from, _)) = self.last_discard.take() {
            self.turn = (from + 1) % self.seats as u8;
        }
        self.pending_robbery = None;
        self.terminal_packet = None;
        self.terminal = true;
        Ok(())
    }

    fn end_kyoku(&mut self) -> Result<(), String> {
        if !self.started || !self.terminal {
            return Err("phase_error: end_kyoku before a terminal result".into());
        }
        self.pending_robbery = None;
        self.last_discard = None;
        self.terminal_packet = None;
        self.started = false;
        Ok(())
    }

    /// Closes the previous ron window: records temporary or riichi furiten if the seat passed.
    fn commit_missed_ron_if_pending(&mut self) {
        if self.pending_missed_ron {
            self.players[self.me as usize].note_missed_ron();
            self.pending_missed_ron = false;
        }
    }

    /// After another player's discard, opens a missed-ron window if the tile is in the own
    /// waits and the seat is not furiten.
    ///
    /// Only the wait shape matters, not yaku. Temporary furiten applies when a waited
    /// tile is discarded and not won, even without a yaku; later tiles in the same turn
    /// cannot be won either (even with a yaku) until the seat's next discard.
    ///
    /// So this cannot use whether `legal_reactions` contains `Ron`: `can_ron_agari` ends
    /// with `scoring::settle().is_some()`, which would never open the window for a hand
    /// without a yaku.
    fn note_opponent_discard_for_furiten(&mut self, from: u8, tile: Tile) {
        if from == self.me {
            return;
        }
        self.commit_missed_ron_if_pending();
        let me = &self.players[self.me as usize];
        if me.is_furiten_blocked() {
            return;
        }
        if me.waits().contains(&tile.kind()) {
            self.pending_missed_ron = true;
        }
    }

    /// Opens the missed-ron window after a kan or nukidora declaration; the declared
    /// tile goes into nobody's discards.
    fn note_robbery_for_furiten(&mut self) {
        self.commit_missed_ron_if_pending();
        let Some(pending) = self.pending_robbery else {
            return;
        };
        if pending.actor != self.me
            && crate::legal::legal_reactions(&self.mirror_view(self.me))
                .iter()
                .any(|action| matches!(action, ReactionAction::Ron))
        {
            self.pending_missed_ron = true;
        }
    }

    fn clear_all_ippatsu(&mut self) {
        for player in &mut self.players {
            player.ippatsu = false;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn start_kyoku(
        &mut self,
        bakaze: Tile,
        dora_marker: Tile,
        kyoku: u8,
        honba: u8,
        kyotaku: u8,
        oya: u8,
        scores: &[i32],
        my_hand: [Tile; 13],
        initial_tiles_left: u32,
        known_tiles: KnownTileInventory,
    ) {
        self.bakaze = bakaze;
        self.oya = oya;
        self.kyoku = kyoku;
        self.honba = honba;
        self.kyotaku = kyotaku;
        self.scores = scores.to_vec();
        self.dora_indicators = vec![dora_marker];
        self.turn = oya;
        self.last_discard = None;
        self.tiles_left = initial_tiles_left;
        self.any_call = false;
        self.pending_rinshan_draw = false;
        self.last_draw_was_rinshan = false;
        self.dealer_opening = false;
        self.pending_missed_ron = false;
        self.pending_reach = None;
        self.pending_robbery = None;
        self.terminal_packet = None;
        self.known_tiles = known_tiles;
        self.started = true;
        self.terminal = false;
        self.quality = if my_hand.iter().any(|tile| tile.is_unknown()) {
            MirrorQuality::Degraded
        } else {
            MirrorQuality::Exact
        };
        self.issues.clear();
        let me = self.me;
        self.players = (0..self.seats as u8)
            .map(|s| {
                // Only the own seat gets its real starting hand; other seats get placeholders.
                let hand = if s == me {
                    my_hand
                } else {
                    [Tile::unknown(); 13]
                };
                if self.seats == 3 {
                    PlayerState::new_sanma(s, hand)
                } else {
                    PlayerState::new(s, hand)
                }
            })
            .collect();
    }

    fn tsumo(&mut self, actor: u8, pai: Tile) -> Result<(), String> {
        self.ensure_active()?;
        self.ensure_actor(actor)?;
        if self.dealer_opening {
            return Err("phase_error: dealer opening must be resolved before another draw".into());
        }
        let expected = self.pending_robbery.map_or_else(
            || {
                self.last_discard
                    .map_or(self.turn, |(from, _)| (from + 1) % self.seats as u8)
            },
            |pending| pending.actor,
        );
        if actor != expected {
            return Err(format!(
                "phase_error: expected seat {expected} to draw, got {actor}"
            ));
        }
        if self.players[actor as usize].drawn_tile.is_some() {
            return Err(format!(
                "duplicate_draw: seat {actor} already has a drawn tile"
            ));
        }
        if actor == self.me {
            let mut known_tiles = self.known_tiles.clone();
            known_tiles.observe(pai, self.seats, self.rule_profile)?;
            self.known_tiles = known_tiles;
        }
        // A draw closes the previous discard's response window (records furiten if passed).
        self.commit_missed_ron_if_pending();
        if self.pending_robbery.take().is_some() {
            self.clear_all_ippatsu();
        }
        // Own draws are real tiles; other seats only get one more unknown.
        let t = if actor == self.me {
            pai
        } else {
            Tile::unknown()
        };
        self.players[actor as usize].draw(t);
        self.turn = actor;
        // A new draw means the previous discard can no longer be answered.
        self.last_discard = None;
        // Drawing from an empty live wall is physically impossible. `saturating_sub` would
        // silently clamp to 0 and hide it, so it is reported.
        //
        // Reported rather than rejected: the mirror is on the observing side and its count
        // may be off by one if upstream dropped an event. Rejecting would void the whole
        // game; downgrading is enough, since a degraded mirror stops inference and only
        // uses actions the platform has shown to be executable.
        if self.tiles_left == 0 {
            self.degrade(
                "draw_from_empty_wall",
                format!("seat {actor} drew while the live wall is already empty"),
            );
        }
        self.tiles_left = self.tiles_left.saturating_sub(1);
        self.last_draw_was_rinshan = self.pending_rinshan_draw;
        self.pending_rinshan_draw = false;
        if actor == self.me && pai.is_unknown() {
            self.degrade(
                "unknown_self_draw",
                format!("seat {actor} draw is hidden from its own mirror"),
            );
        }
        Ok(())
    }

    fn dealer_opening(&mut self, actor: u8, pai: Tile) -> Result<(), String> {
        self.ensure_active()?;
        self.ensure_actor(actor)?;
        if actor != self.oya
            || actor != self.turn
            || self.last_discard.is_some()
            || self.pending_robbery.is_some()
            || self.dealer_opening
            || self.players.iter().any(|player| {
                !player.discards.is_empty()
                    || !player.melds.is_empty()
                    || player.drawn_tile.is_some()
            })
            || self.players[actor as usize].hand.len() != 13
        {
            return Err(
                "phase_error: dealer_opening is only valid before the dealer's first action".into(),
            );
        }
        if actor == self.me {
            let mut known_tiles = self.known_tiles.clone();
            known_tiles.observe(pai, self.seats, self.rule_profile)?;
            self.known_tiles = known_tiles;
        }
        self.players[actor as usize].hand.push(if actor == self.me {
            pai
        } else {
            Tile::unknown()
        });
        self.tiles_left = self.tiles_left.saturating_sub(1);
        self.dealer_opening = true;
        if actor == self.me && pai.is_unknown() {
            self.degrade(
                "unknown_self_dealer_opening",
                format!("seat {actor} dealer opening tile is hidden from its own mirror"),
            );
        }
        Ok(())
    }

    fn dahai(&mut self, actor: u8, pai: Tile, tsumogiri: bool) -> Result<(), String> {
        self.ensure_active()?;
        self.ensure_actor(actor)?;
        if self.dealer_opening {
            return Err("phase_error: dealer opening discard requires dealer_opening_dahai".into());
        }
        if actor != self.turn || self.last_discard.is_some() || self.pending_robbery.is_some() {
            return Err(format!(
                "phase_error: seat {actor} cannot discard in the current phase"
            ));
        }
        self.validate_reach_discard(
            actor,
            &TurnAction::Riichi {
                tile: pai,
                tsumogiri,
            },
        )?;
        if actor != self.me {
            let mut known_tiles = self.known_tiles.clone();
            known_tiles.observe(pai, self.seats, self.rule_profile)?;
            self.known_tiles = known_tiles;
        }
        let me = self.me;
        let declared_reach = self.pending_reach_actor() == Some(actor);
        let p = &mut self.players[actor as usize];
        if actor == me {
            p.discard_with_source(pai, tsumogiri)
                .map_err(|err| format!("dahai: {err}"))?;
        } else {
            if tsumogiri && p.drawn_tile.is_none() {
                return Err(format!(
                    "dahai: seat {actor} reports tsumogiri without a draw"
                ));
            }
            if !remove_unknown(&mut p.hand) {
                return Err(format!("dahai: seat {actor} has no unknown tile to remove"));
            }
            p.drawn_tile = None;
            p.discards.push(pai);
            p.temporary_furiten = false;
            p.kuikae_forbidden.clear();
        }
        if !declared_reach {
            p.ippatsu = false;
        }
        self.turn = actor;
        self.last_discard = Some((actor, pai));
        self.last_draw_was_rinshan = false;
        if actor == me {
            // Own discard: PlayerState recomputes discard furiten; clear the ron window.
            self.pending_missed_ron = false;
        } else {
            self.note_opponent_discard_for_furiten(actor, pai);
        }
        self.commit_reach_discard(actor);
        Ok(())
    }

    fn dealer_opening_dahai(&mut self, actor: u8, pai: Tile) -> Result<(), String> {
        self.ensure_active()?;
        self.ensure_actor(actor)?;
        if !self.dealer_opening
            || actor != self.oya
            || actor != self.turn
            || self.last_discard.is_some()
            || self.pending_robbery.is_some()
        {
            return Err("phase_error: dealer_opening_dahai outside dealer opening window".into());
        }
        self.validate_reach_discard(actor, &TurnAction::DealerOpeningRiichi { tile: pai })?;
        if actor != self.me {
            let mut known_tiles = self.known_tiles.clone();
            known_tiles.observe(pai, self.seats, self.rule_profile)?;
            self.known_tiles = known_tiles;
        }
        let me = self.me;
        let declared_reach = self.pending_reach_actor() == Some(actor);
        let player = &mut self.players[actor as usize];
        if actor == me {
            player
                .discard_dealer_opening(pai)
                .map_err(|err| format!("dealer_opening_dahai: {err}"))?;
        } else {
            if !remove_unknown(&mut player.hand) {
                return Err("dealer_opening_dahai: dealer has no concealed tile to remove".into());
            }
            player.discards.push(pai);
            player.temporary_furiten = false;
            player.kuikae_forbidden.clear();
        }
        if !declared_reach {
            player.ippatsu = false;
        }
        self.dealer_opening = false;
        self.turn = actor;
        self.last_discard = Some((actor, pai));
        self.last_draw_was_rinshan = false;
        if actor == me {
            self.pending_missed_ron = false;
        } else {
            self.note_opponent_discard_for_furiten(actor, pai);
        }
        self.commit_reach_discard(actor);
        Ok(())
    }

    /// Removes tiles consumed by a call (exact tiles for the own seat, `count` unknowns for others).
    fn consume(&mut self, actor: u8, tiles: &[Tile]) -> Result<(), String> {
        if actor as usize >= self.players.len() {
            return Err(format!("consume: invalid seat {actor}"));
        }
        let me = self.me;
        let p = &mut self.players[actor as usize];
        if actor == me {
            let mut next = p.hand.clone();
            for &t in tiles {
                if !remove_tile(&mut next, t) {
                    return Err(format!("consume: seat {actor} missing tile {t:?}"));
                }
            }
            p.hand = next;
        } else {
            let available = p.hand.iter().filter(|t| t.is_unknown()).count();
            if available < tiles.len() {
                return Err(format!(
                    "consume: seat {actor} missing unknown tile (need {})",
                    tiles.len()
                ));
            }
            for _ in 0..tiles.len() {
                let removed = remove_unknown(&mut p.hand);
                debug_assert!(removed);
            }
        }
        p.drawn_tile = None;
        Ok(())
    }

    fn validate_call(&self, actor: u8, target: u8, pai: Tile, chi: bool) -> Result<(), String> {
        self.ensure_active()?;
        self.ensure_actor(actor)?;
        self.ensure_actor(target)?;
        if actor == target || self.last_discard != Some((target, pai)) {
            return Err("call_window_mismatch: target/tile does not match last discard".into());
        }
        if chi && (target + 1) % self.seats as u8 != actor {
            return Err("illegal_chi_direction: only the next seat may chi".into());
        }
        Ok(())
    }

    fn chi(&mut self, actor: u8, target: u8, pai: Tile, consumed: [Tile; 2]) -> Result<(), String> {
        self.validate_call(actor, target, pai, true)?;
        let mut kinds = [consumed[0].kind(), consumed[1].kind(), pai.kind()];
        kinds.sort_unstable();
        let same_suit = kinds[0] < 27 && kinds[0] / 9 == kinds[2] / 9;
        if !same_suit || kinds[0] + 1 != kinds[1] || kinds[1] + 1 != kinds[2] {
            return Err("chi: consumed tiles do not form a sequence".into());
        }
        if actor != self.me {
            let mut known_tiles = self.known_tiles.clone();
            known_tiles.observe_all(consumed, self.seats, self.rule_profile)?;
            self.known_tiles = known_tiles;
        }
        self.commit_missed_ron_if_pending();
        self.consume(actor, &consumed)?;
        let mut tiles = [consumed[0], consumed[1], pai];
        tiles.sort_by_key(|t| t.kind());
        self.players[actor as usize].melds.push(Meld::Chi {
            tiles,
            called: pai,
            from: target,
        });
        self.players[actor as usize].kuikae_forbidden = if self.rule_profile.kuikae_forbidden() {
            crate::tileset::kuikae_forbidden_after_chi(pai, consumed)
        } else {
            Vec::new()
        };
        self.players[target as usize].mark_latest_discard_called()?;
        self.turn = actor;
        self.last_discard = None;
        self.any_call = true;
        self.last_draw_was_rinshan = false;
        self.clear_all_ippatsu();
        Ok(())
    }

    fn pon(&mut self, actor: u8, target: u8, pai: Tile, consumed: [Tile; 2]) -> Result<(), String> {
        self.validate_call(actor, target, pai, false)?;
        if consumed.iter().any(|tile| tile.kind() != pai.kind()) {
            return Err("pon: consumed tiles do not match called tile".into());
        }
        if actor != self.me {
            let mut known_tiles = self.known_tiles.clone();
            known_tiles.observe_all(consumed, self.seats, self.rule_profile)?;
            self.known_tiles = known_tiles;
        }
        self.commit_missed_ron_if_pending();
        self.consume(actor, &consumed)?;
        self.players[actor as usize].melds.push(Meld::Pon {
            tile: pai.deaka(),
            called: pai,
            consumed,
            from: target,
        });
        self.players[actor as usize].kuikae_forbidden = if self.rule_profile.kuikae_forbidden() {
            vec![pai.kind()]
        } else {
            Vec::new()
        };
        self.players[target as usize].mark_latest_discard_called()?;
        self.turn = actor;
        self.last_discard = None;
        self.any_call = true;
        self.last_draw_was_rinshan = false;
        self.clear_all_ippatsu();
        Ok(())
    }

    fn daiminkan(
        &mut self,
        actor: u8,
        target: u8,
        pai: Tile,
        consumed: [Tile; 3],
    ) -> Result<(), String> {
        self.validate_call(actor, target, pai, false)?;
        if consumed.iter().any(|tile| tile.kind() != pai.kind()) {
            return Err("daiminkan: consumed tiles do not match called tile".into());
        }
        if actor != self.me {
            let mut known_tiles = self.known_tiles.clone();
            known_tiles.observe_all(consumed, self.seats, self.rule_profile)?;
            self.known_tiles = known_tiles;
        }
        self.commit_missed_ron_if_pending();
        self.consume(actor, &consumed)?;
        self.players[actor as usize].melds.push(Meld::Daiminkan {
            tile: pai.deaka(),
            called: pai,
            from: target,
        });
        self.players[target as usize].mark_latest_discard_called()?;
        self.turn = actor;
        self.last_discard = None;
        self.any_call = true;
        self.pending_rinshan_draw = true;
        self.last_draw_was_rinshan = false;
        self.clear_all_ippatsu();
        // The replacement draw advances `tiles_left` through the following `Tsumo`.
        Ok(())
    }

    fn kakan(&mut self, actor: u8, pai: Tile, consumed: [Tile; 3]) -> Result<(), String> {
        self.ensure_active()?;
        self.ensure_actor(actor)?;
        if actor != self.turn || self.last_discard.is_some() || self.pending_robbery.is_some() {
            return Err("phase_error: kakan outside the actor turn".into());
        }
        let kind = pai.kind();
        let pon_idx = self
            .players
            .get(actor as usize)
            .and_then(|p| {
                p.melds
                    .iter()
                    .position(|m| matches!(m, Meld::Pon { tile, .. } if tile.kind() == kind))
            })
            .ok_or_else(|| format!("kakan: seat {actor} has no matching pon for {pai:?}"))?;
        let mut expected = match &self.players[actor as usize].melds[pon_idx] {
            Meld::Pon {
                called, consumed, ..
            } => [consumed[0], consumed[1], *called],
            _ => unreachable!("pon index was selected above"),
        };
        let mut reported = consumed;
        expected.sort_by_key(|tile| tile.id());
        reported.sort_by_key(|tile| tile.id());
        if expected != reported {
            return Err("kakan: consumed tiles do not match the original pon".into());
        }
        if actor != self.me {
            let mut known_tiles = self.known_tiles.clone();
            known_tiles.observe(pai, self.seats, self.rule_profile)?;
            self.known_tiles = known_tiles;
        }
        self.consume(actor, &[pai])?;
        let p = &mut self.players[actor as usize];
        if !matches!(p.melds.get(pon_idx), Some(Meld::Pon { .. })) {
            return Err(format!(
                "kakan: seat {actor} has no matching pon for {pai:?}"
            ));
        }
        p.melds[pon_idx] = Meld::Kakan {
            tile: pai.deaka(),
            added: pai,
        };
        self.turn = actor;
        self.last_discard = None;
        self.any_call = true;
        self.pending_rinshan_draw = true;
        self.last_draw_was_rinshan = false;
        self.pending_robbery = Some(PendingRobbery {
            actor,
            tile: pai,
            kind: RobberyKind::Kakan,
        });
        self.note_robbery_for_furiten();
        Ok(())
    }

    fn ankan(&mut self, actor: u8, consumed: [Tile; 4]) -> Result<(), String> {
        self.ensure_active()?;
        self.ensure_actor(actor)?;
        if actor != self.turn || self.last_discard.is_some() || self.pending_robbery.is_some() {
            return Err("phase_error: ankan outside the actor turn".into());
        }
        if consumed
            .iter()
            .any(|tile| tile.kind() != consumed[0].kind())
        {
            return Err("ankan: consumed tiles must have one kind".into());
        }
        if actor != self.me {
            let mut known_tiles = self.known_tiles.clone();
            known_tiles.observe_all(consumed, self.seats, self.rule_profile)?;
            self.known_tiles = known_tiles;
        }
        self.consume(actor, &consumed)?;
        self.players[actor as usize].melds.push(Meld::Ankan {
            tile: consumed[0].deaka(),
        });
        self.turn = actor;
        self.last_discard = None;
        self.any_call = true;
        self.dealer_opening = false;
        self.pending_rinshan_draw = true;
        self.last_draw_was_rinshan = false;
        self.pending_robbery = Some(PendingRobbery {
            actor,
            tile: consumed[0].deaka(),
            kind: RobberyKind::Ankan,
        });
        self.note_robbery_for_furiten();
        Ok(())
    }

    fn nukidora(&mut self, actor: u8, pai: Tile) -> Result<(), String> {
        self.ensure_active()?;
        self.ensure_actor(actor)?;
        if actor != self.turn || self.last_discard.is_some() || self.pending_robbery.is_some() {
            return Err("phase_error: nukidora outside the actor turn".into());
        }
        let north: Tile = "N".parse().expect("literal north");
        if pai.kind() != north.kind() {
            return Err("nukidora: only north may be extracted".into());
        }
        if actor != self.me {
            let mut known_tiles = self.known_tiles.clone();
            known_tiles.observe(pai, self.seats, self.rule_profile)?;
            self.known_tiles = known_tiles;
        }
        self.consume(actor, &[pai])?;
        self.players[actor as usize]
            .melds
            .push(Meld::Nukidora { tile: pai.deaka() });
        self.turn = actor;
        self.last_discard = None;
        self.any_call = true;
        self.dealer_opening = false;
        self.pending_rinshan_draw = true;
        self.last_draw_was_rinshan = false;
        self.pending_robbery = Some(PendingRobbery {
            actor,
            tile: pai,
            kind: RobberyKind::Nukidora,
        });
        self.note_robbery_for_furiten();
        // The nukidora replacement draw advances `tiles_left` through the following `Tsumo`.
        Ok(())
    }

    fn reach(&mut self, actor: u8) -> Result<(), String> {
        self.ensure_active()?;
        self.ensure_actor(actor)?;
        if actor != self.turn
            || self.last_discard.is_some()
            || self.pending_robbery.is_some()
            || self.pending_reach.is_some()
        {
            return Err("phase_error: reach outside a turn decision".into());
        }
        let legal_discards = if actor == self.me {
            let legal = crate::legal::legal_turn_actions(&self.mirror_view(actor))
                .into_iter()
                .filter(|action| {
                    matches!(
                        action,
                        TurnAction::Riichi { .. } | TurnAction::DealerOpeningRiichi { .. }
                    )
                })
                .collect::<Vec<_>>();
            if legal.is_empty() {
                return Err("illegal_reach: no legal riichi discard exists".into());
            }
            legal
        } else if (self.players[actor as usize].drawn_tile.is_none()
            && !(self.dealer_opening && actor == self.oya))
            || !self.players[actor as usize]
                .melds
                .iter()
                .all(|meld| !meld.breaks_menzen())
            || self.players[actor as usize].riichi
            || self.scores[actor as usize] < 1_000
        {
            return Err("illegal_reach: public prerequisites are not met".into());
        } else {
            Vec::new()
        };
        let double_riichi = self.players.iter().all(|player| player.melds.is_empty())
            && self.players[actor as usize].discards.is_empty();
        self.pending_reach = Some(PendingReach {
            actor,
            double_riichi,
            legal_discards,
        });
        Ok(())
    }

    fn reach_accepted(&mut self, actor: u8) -> Result<(), String> {
        self.ensure_active()?;
        self.ensure_actor(actor)?;
        if self.pending_reach_actor() != Some(actor)
            || !matches!(self.last_discard, Some((discarder, _)) if discarder == actor)
        {
            return Err("phase_error: reach_accepted without matching declaration discard".into());
        }
        if self.scores[actor as usize] < 1_000 {
            return Err("illegal_reach_acceptance: score below 1000".into());
        }
        // Riichi accepted: 1000 points to the table. `Reach` declares, the discard sets
        // riichi, `ReachAccepted` takes the stick.
        self.scores[actor as usize] -= 1000;
        self.kyotaku += 1;
        self.pending_reach = None;
        Ok(())
    }

    fn dora(&mut self, dora_marker: Tile) -> Result<(), String> {
        self.ensure_active()?;
        if dora_marker.is_unknown() {
            return Err("dora: public marker cannot be unknown".into());
        }
        if self.dora_indicators.len() >= crate::tileset::MAX_DORA_INDICATORS {
            return Err(format!(
                "dora: cannot reveal more than {} indicators",
                crate::tileset::MAX_DORA_INDICATORS
            ));
        }
        let mut known_tiles = self.known_tiles.clone();
        known_tiles.observe(dora_marker, self.seats, self.rule_profile)?;
        self.known_tiles = known_tiles;
        self.dora_indicators.push(dora_marker);
        Ok(())
    }

    fn mirror_view(&self, seat: u8) -> SeatView {
        if seat as usize >= self.seats {
            return SeatView::invalid(
                seat,
                self.seats as u8,
                self.rule_profile,
                SeatViewError::InvalidSeat {
                    requested: seat,
                    seats: self.seats as u8,
                },
            );
        }
        if self.players.is_empty() {
            return SeatView::invalid(
                seat,
                self.seats as u8,
                self.rule_profile,
                SeatViewError::MirrorNotStarted,
            );
        }
        if self.quality == MirrorQuality::Unsafe {
            return SeatView::invalid(
                seat,
                self.seats as u8,
                self.rule_profile,
                SeatViewError::MirrorUnsafe,
            );
        }
        if self.quality == MirrorQuality::Degraded {
            return SeatView::invalid(
                seat,
                self.seats as u8,
                self.rule_profile,
                SeatViewError::MirrorDegraded,
            );
        }
        let me_ps = &self.players[seat as usize];
        let others = (0..self.seats as u8)
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
        // Kyuushu kyuuhai window: the seat's first draw (a drawn tile, empty discards, no
        // melds) with no calls, nukidora or kans before it.
        let kyuushu_kyuuhai_window = !self.any_call
            && me_ps.discards.is_empty()
            && me_ps.melds.is_empty()
            && (me_ps.drawn_tile.is_some() || (self.dealer_opening && seat == self.oya));
        SeatView {
            state_error: None,
            round_terminal: self.terminal,
            rule_profile: self.rule_profile,
            bakaze: self.bakaze,
            kyoku: self.kyoku,
            honba: self.honba,
            kyotaku: self.kyotaku,
            oya: self.oya,
            scores: self.scores.clone(),
            dora_indicators: self.dora_indicators.clone(),
            tiles_left: self.tiles_left,
            me: SelfView {
                seat,
                hand: me_ps.hand.clone(),
                drawn_tile: me_ps.drawn_tile,
                dealer_opening: self.dealer_opening && seat == self.oya,
                melds: me_ps.melds.clone(),
                discards: me_ps.discards.clone(),
                riichi: me_ps.riichi,
                ippatsu: me_ps.ippatsu,
                kuikae_forbidden: me_ps.kuikae_forbidden.clone(),
                temporary_furiten: me_ps.temporary_furiten,
                riichi_furiten: me_ps.riichi_furiten,
            },
            others,
            turn: self.turn,
            last_discard: self.last_discard,
            pending_robbery: self.pending_robbery,
            kyuushu_kyuuhai_window,
            suukaikan_pending: self.last_discard.is_some() && self.suukaikan_condition_met(),
            last_draw_was_rinshan: self.last_draw_was_rinshan,
        }
    }

    fn should_infer(&self, seat: u8) -> bool {
        if self.quality != MirrorQuality::Exact
            || !self.started
            || self.terminal
            || seat as usize >= self.seats
        {
            return false;
        }
        if self.awaiting_reach_discard() {
            return false;
        }
        // Trigger only: the seat's own draw, or another player's discard it may answer.
        if self.pending_robbery.is_some() {
            return crate::legal::legal_reactions(&self.mirror_view(seat))
                .iter()
                .any(|action| matches!(action, ReactionAction::Ron));
        }
        if self.turn == seat && self.last_discard.is_none() {
            return true;
        }
        if let Some((from, _)) = self.last_discard {
            return from != seat; // another player just discarded; a response may be possible
        }
        false
    }
}

/// 4-player mirror: consumes [`Event4p`], rebuilds public state plus the own hand,
/// and runs rule calculations.
#[derive(Clone)]
pub struct Mirror4p {
    core: MirrorCore,
    last_event: Option<Event4p>,
}

impl Mirror4p {
    /// Creates an empty mirror for seat `me` (`0..=3`).
    pub fn new(me: u8) -> Self {
        Self::new_with_rule_profile(me, RiichiRuleProfile::default())
    }

    /// Creates an empty mirror with explicit platform rules.
    pub fn new_with_rule_profile(me: u8, rule_profile: RiichiRuleProfile) -> Self {
        Self {
            core: MirrorCore::new(4, me, rule_profile),
            last_event: None,
        }
    }
}

impl MirrorCenter<Event4p> for Mirror4p {
    fn feed(&mut self, event: Event4p) -> Result<(), String> {
        use Event4p as E;
        if self.core.quality != MirrorQuality::Unsafe
            && self.last_event.as_ref() == Some(&event)
            && !matches!(&event, E::None)
        {
            let detail = format!("duplicate event at index {}", self.core.event_index);
            self.core.reject("duplicate_event", detail.clone());
            return Err(detail);
        }
        if self.core.quality == MirrorQuality::Unsafe
            && !matches!(&event, E::StartKyoku { .. } | E::StartGame { .. })
        {
            return Err("mirror_unsafe: resynchronization required".into());
        }
        if self.core.awaiting_reach_discard() {
            let expected = self
                .core
                .pending_reach_actor()
                .expect("awaiting reach discard has an actor");
            let valid = match &event {
                E::Dahai { actor, .. } | E::DealerOpeningDahai { actor, .. } => *actor == expected,
                E::None => true,
                _ => false,
            };
            if !valid {
                let detail =
                    format!("phase_error: waiting for seat {expected} declaration discard");
                self.core.reject("event_rejected", detail.clone());
                return Err(detail);
            }
        }
        let accepted_event = event.clone();
        let mut next = self.core.clone();
        let result = match event {
            E::StartKyoku {
                bakaze,
                dora_marker,
                kyoku,
                honba,
                kyotaku,
                oya,
                scores,
                tehais,
            } => {
                if next.me as usize >= next.seats {
                    Err(format!(
                        "invalid_mirror_seat: seat {} is outside 0..{}",
                        next.me, next.seats
                    ))
                } else {
                    match next.validate_start_kyoku(
                        oya,
                        dora_marker,
                        &tehais[next.me as usize],
                        false,
                    ) {
                        Err(err) => Err(err),
                        Ok(known_tiles) => {
                            let my_hand = tehais[next.me as usize];
                            next.start_kyoku(
                                bakaze,
                                dora_marker,
                                kyoku,
                                honba,
                                kyotaku,
                                oya,
                                &scores,
                                my_hand,
                                LIVE_WALL_4P,
                                known_tiles,
                            );
                            Ok(())
                        }
                    }
                }
            }
            E::Tsumo { actor, pai } => next.tsumo(actor, pai),
            E::DealerOpening { actor, pai } => next.dealer_opening(actor, pai),
            E::Dahai {
                actor,
                pai,
                tsumogiri,
            } => next.dahai(actor, pai, tsumogiri),
            E::DealerOpeningDahai { actor, pai } => next.dealer_opening_dahai(actor, pai),
            E::Chi {
                actor,
                target,
                pai,
                consumed,
            } => next.chi(actor, target, pai, consumed),
            E::Pon {
                actor,
                target,
                pai,
                consumed,
            } => next.pon(actor, target, pai, consumed),
            E::Daiminkan {
                actor,
                target,
                pai,
                consumed,
            } => next.daiminkan(actor, target, pai, consumed),
            E::Kakan {
                actor,
                pai,
                consumed,
            } => next.kakan(actor, pai, consumed),
            E::Ankan { actor, consumed } => next.ankan(actor, consumed),
            E::Dora { dora_marker } => next.dora(dora_marker),
            E::Reach { actor } => next.reach(actor),
            E::ReachAccepted { actor } => next.reach_accepted(actor),
            E::Hora {
                actor,
                target,
                deltas,
                ura_markers,
                ..
            } => next.hora(
                actor,
                target,
                deltas.as_ref().map(|value| value.as_slice()),
                ura_markers.as_deref(),
            ),
            E::Ryukyoku { deltas } => next.ryukyoku(deltas.as_ref().map(|value| value.as_slice())),
            E::EndKyoku => next.end_kyoku(),
            E::EndGame => {
                if next.started && !next.terminal {
                    Err("phase_error: end_game during an active kyoku".into())
                } else {
                    next.started = false;
                    Ok(())
                }
            }
            E::StartGame { .. } => {
                if next.started && !next.terminal {
                    Err("phase_error: start_game during an active kyoku".into())
                } else {
                    Ok(())
                }
            }
            // Forced autoplay only describes opponent behavior and changes no tile state; out-of-range seats are still rejected.
            E::SeatForcedAutoplay { actor } | E::SeatResumed { actor } => {
                if usize::from(actor) >= next.seats {
                    Err(format!(
                        "invalid_actor: seat {actor} is outside 0..{}",
                        next.seats
                    ))
                } else {
                    Ok(())
                }
            }
            E::None => Ok(()),
        };
        if result.is_ok() {
            next.event_index = next.event_index.saturating_add(1);
            self.core = next;
            self.last_event = Some(accepted_event);
        } else if let Err(detail) = &result {
            self.core.reject("event_rejected", detail.clone());
        }
        result
    }

    fn quality(&self) -> MirrorQuality {
        self.core.quality
    }

    fn last_issue(&self) -> Option<&MirrorIssue> {
        self.core.issues.last()
    }

    fn mark_degraded(&mut self, code: &'static str, detail: impl Into<String>) {
        self.core.degrade(code, detail);
    }

    fn mirror_view(&self, seat: u8) -> SeatView {
        self.core.mirror_view(seat)
    }

    fn should_infer(&self, seat: u8) -> bool {
        self.core.should_infer(seat)
    }
}

/// 3-player mirror: consumes [`Event3p`] (no chi, nukidora, 3 seats), otherwise the
/// same as 4-player through [`MirrorCore`].
#[derive(Clone)]
pub struct Mirror3p {
    core: MirrorCore,
    last_event: Option<Event3p>,
}

impl Mirror3p {
    /// Creates an empty mirror for seat `me` (`0..=2`).
    pub fn new(me: u8) -> Self {
        Self::new_with_rule_profile(me, RiichiRuleProfile::default())
    }

    /// Creates an empty mirror with explicit platform rules.
    pub fn new_with_rule_profile(me: u8, rule_profile: RiichiRuleProfile) -> Self {
        Self {
            core: MirrorCore::new(3, me, rule_profile),
            last_event: None,
        }
    }
}

impl MirrorCenter<Event3p> for Mirror3p {
    fn feed(&mut self, event: Event3p) -> Result<(), String> {
        use Event3p as E;
        if self.core.quality != MirrorQuality::Unsafe
            && self.last_event.as_ref() == Some(&event)
            && !matches!(&event, E::None)
        {
            let detail = format!("duplicate event at index {}", self.core.event_index);
            self.core.reject("duplicate_event", detail.clone());
            return Err(detail);
        }
        if self.core.quality == MirrorQuality::Unsafe
            && !matches!(&event, E::StartKyoku { .. } | E::StartGame { .. })
        {
            return Err("mirror_unsafe: resynchronization required".into());
        }
        if self.core.awaiting_reach_discard() {
            let expected = self
                .core
                .pending_reach_actor()
                .expect("awaiting reach discard has an actor");
            let valid = match &event {
                E::Dahai { actor, .. } | E::DealerOpeningDahai { actor, .. } => *actor == expected,
                E::None => true,
                _ => false,
            };
            if !valid {
                let detail =
                    format!("phase_error: waiting for seat {expected} declaration discard");
                self.core.reject("event_rejected", detail.clone());
                return Err(detail);
            }
        }
        let accepted_event = event.clone();
        let mut next = self.core.clone();
        let result = match event {
            E::StartKyoku {
                bakaze,
                dora_marker,
                kyoku,
                honba,
                kyotaku,
                oya,
                scores,
                tehais,
            } => {
                if next.me as usize >= next.seats {
                    Err(format!(
                        "invalid_mirror_seat: seat {} is outside 0..{}",
                        next.me, next.seats
                    ))
                } else {
                    match next.validate_start_kyoku(
                        oya,
                        dora_marker,
                        &tehais[next.me as usize],
                        true,
                    ) {
                        Err(err) => Err(err),
                        Ok(known_tiles) => {
                            let my_hand = tehais[next.me as usize];
                            next.start_kyoku(
                                bakaze,
                                dora_marker,
                                kyoku,
                                honba,
                                kyotaku,
                                oya,
                                &scores,
                                my_hand,
                                LIVE_WALL_3P,
                                known_tiles,
                            );
                            Ok(())
                        }
                    }
                }
            }
            E::Tsumo { actor, pai } => next.tsumo(actor, pai),
            E::DealerOpening { actor, pai } => next.dealer_opening(actor, pai),
            E::Dahai {
                actor,
                pai,
                tsumogiri,
            } => next.dahai(actor, pai, tsumogiri),
            E::DealerOpeningDahai { actor, pai } => next.dealer_opening_dahai(actor, pai),
            E::Pon {
                actor,
                target,
                pai,
                consumed,
            } => next.pon(actor, target, pai, consumed),
            E::Daiminkan {
                actor,
                target,
                pai,
                consumed,
            } => next.daiminkan(actor, target, pai, consumed),
            E::Kakan {
                actor,
                pai,
                consumed,
            } => next.kakan(actor, pai, consumed),
            E::Ankan { actor, consumed } => next.ankan(actor, consumed),
            E::Nukidora { actor, pai } => next.nukidora(actor, pai),
            E::Dora { dora_marker } => next.dora(dora_marker),
            E::Reach { actor } => next.reach(actor),
            E::ReachAccepted { actor } => next.reach_accepted(actor),
            E::Hora {
                actor,
                target,
                deltas,
                ura_markers,
                ..
            } => next.hora(
                actor,
                target,
                deltas.as_ref().map(|value| value.as_slice()),
                ura_markers.as_deref(),
            ),
            E::Ryukyoku { deltas } => next.ryukyoku(deltas.as_ref().map(|value| value.as_slice())),
            E::EndKyoku => next.end_kyoku(),
            E::EndGame => {
                if next.started && !next.terminal {
                    Err("phase_error: end_game during an active kyoku".into())
                } else {
                    next.started = false;
                    Ok(())
                }
            }
            E::StartGame { .. } => {
                if next.started && !next.terminal {
                    Err("phase_error: start_game during an active kyoku".into())
                } else {
                    Ok(())
                }
            }
            // Forced autoplay only describes opponent behavior and changes no tile state; out-of-range seats are still rejected.
            E::SeatForcedAutoplay { actor } | E::SeatResumed { actor } => {
                if usize::from(actor) >= next.seats {
                    Err(format!(
                        "invalid_actor: seat {actor} is outside 0..{}",
                        next.seats
                    ))
                } else {
                    Ok(())
                }
            }
            E::None => Ok(()),
        };
        if result.is_ok() {
            next.event_index = next.event_index.saturating_add(1);
            self.core = next;
            self.last_event = Some(accepted_event);
        } else if let Err(detail) = &result {
            self.core.reject("event_rejected", detail.clone());
        }
        result
    }

    fn quality(&self) -> MirrorQuality {
        self.core.quality
    }

    fn last_issue(&self) -> Option<&MirrorIssue> {
        self.core.issues.last()
    }

    fn mark_degraded(&mut self, code: &'static str, detail: impl Into<String>) {
        self.core.degrade(code, detail);
    }

    fn mirror_view(&self, seat: u8) -> SeatView {
        self.core.mirror_view(seat)
    }

    fn should_infer(&self, seat: u8) -> bool {
        self.core.should_infer(seat)
    }
}
