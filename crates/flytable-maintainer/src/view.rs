//! Seat view: information isolation at the type level.
//!
//! The state sent to one seat has no place to hold hidden tiles. Other hands are
//! counts (`u8`), the wall is a remaining count, and ura dora only exist after they
//! are revealed. Hidden information is never filled in and filtered later, so
//! isolation holds in the core.
//!
//! Shared by 4-player and 3-player; the length of `others` is the number of opponents.

use flytable_core::meld::Meld;
use flytable_core::rules::RiichiRuleProfile;
use flytable_core::tile::Tile;

use crate::action::PendingRobbery;

/// Structured reason why a trustworthy seat view could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeatViewError {
    InvalidSeat { requested: u8, seats: u8 },
    MirrorNotStarted,
    MirrorDegraded,
    MirrorUnsafe,
}

/// Everything the seat can see about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfView {
    pub seat: u8,
    /// Own hand (actual tiles, including red fives).
    pub hand: Vec<Tile>,
    /// A tile just drawn and not yet discarded. `None` when discarding after chi/pon.
    pub drawn_tile: Option<Tile>,
    /// The dealer's 14-tile opening window (Mahjong Soul); `drawn_tile` must be `None`.
    pub dealer_opening: bool,
    pub melds: Vec<Meld>,
    /// Own discards in order.
    pub discards: Vec<Tile>,
    pub riichi: bool,
    /// Ippatsu still possible.
    pub ippatsu: bool,
    /// Kinds forbidden for the discard right after chi/pon (kuikae).
    pub kuikae_forbidden: Vec<usize>,
    /// Temporary furiten (passed on a ron, until the seat's next discard).
    pub temporary_furiten: bool,
    /// Riichi furiten (passed on a ron after riichi; no ron for the rest of the hand).
    pub riichi_furiten: bool,
}

/// What is visible about one opponent. There is no field for their hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpponentView {
    pub seat: u8,
    /// Number of tiles in the closed hand.
    pub hand_count: u8,
    /// Opponent melds (public).
    pub melds: Vec<Meld>,
    /// Opponent discards (public).
    pub discards: Vec<Tile>,
    pub riichi: bool,
}

/// View of the table sent to one seat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeatView {
    /// `Some` means the view must not be used for inference or action enumeration;
    /// legal action entry points must fail closed.
    pub state_error: Option<SeatViewError>,
    /// The hand has ended (win or draw). Still displayable, but no more actions.
    pub round_terminal: bool,
    /// Platform rules in effect. Legal actions and settlement must use this, not just the seat count.
    pub rule_profile: RiichiRuleProfile,
    /// Round wind.
    pub bakaze: Tile,
    /// Hand number (1-based), honba and riichi sticks.
    pub kyoku: u8,
    pub honba: u8,
    pub kyotaku: u8,
    /// Dealer seat.
    pub oya: u8,
    /// Scores in seat order (public).
    pub scores: Vec<i32>,
    /// Dora indicators (public). Ura dora are not in the view until revealed.
    pub dora_indicators: Vec<Tile>,
    /// Tiles left in the live wall.
    pub tiles_left: u32,
    pub me: SelfView,
    /// Opponents in seat order, excluding self.
    pub others: Vec<OpponentView>,
    /// Seat whose turn it is.
    pub turn: u8,
    /// The latest discard and its discarder, if any (for calls and ron).
    pub last_discard: Option<(u8, Tile)>,
    /// Added kan / closed kan / nukidora declared but not yet followed by the
    /// replacement draw. Only ron or pass is possible in this window.
    pub pending_robbery: Option<PendingRobbery>,
    /// Kyuushu kyuuhai window: after the first draw, with no calls, nukidora or kans before it.
    pub kyuushu_kyuuhai_window: bool,
    /// Four-kan draw pending: the fourth kan (held by two or more players) is complete
    /// and `last_discard` is the kan player's next discard. This response window only
    /// offers ron; if nobody wins the hand ends in an abortive draw.
    ///
    /// The draw happens after the kan player's replacement draw and discard (or
    /// nukidora), not at the kan declaration.
    pub suukaikan_pending: bool,
    /// Whether the last draw was a replacement draw (after a kan or 3-player nukidora).
    /// Lets legal tsumo enumeration recognize rinshan kaihou as the only yaku. Filled by
    /// the authoritative board and the mirror.
    pub last_draw_was_rinshan: bool,
}

impl SeatView {
    /// Builds a fail-closed view that carries no game state and produces no legal actions.
    pub fn invalid(
        requested: u8,
        seats: u8,
        rule_profile: RiichiRuleProfile,
        error: SeatViewError,
    ) -> Self {
        Self {
            state_error: Some(error),
            round_terminal: true,
            rule_profile,
            bakaze: Tile::default(),
            kyoku: 0,
            honba: 0,
            kyotaku: 0,
            oya: 0,
            scores: vec![0; seats as usize],
            dora_indicators: Vec::new(),
            tiles_left: 0,
            me: SelfView {
                seat: requested,
                hand: Vec::new(),
                drawn_tile: None,
                dealer_opening: false,
                melds: Vec::new(),
                discards: Vec::new(),
                riichi: false,
                ippatsu: false,
                kuikae_forbidden: Vec::new(),
                temporary_furiten: false,
                riichi_furiten: false,
            },
            others: Vec::new(),
            turn: 0,
            last_discard: None,
            pending_robbery: None,
            kyuushu_kyuuhai_window: false,
            suukaikan_pending: false,
            last_draw_was_rinshan: false,
        }
    }

    pub const fn is_valid(&self) -> bool {
        self.state_error.is_none()
    }

    /// Debug check that the view leaks no hidden information beyond `?` placeholders.
    /// Only covers fields we fill ourselves; the types already prevent opponent hand values.
    pub fn assert_no_hidden_truth(&self) {
        // Opponent views have no hand field, so there is nothing to check; the own hand may hold real tiles.
        debug_assert!(self.me.hand.len() <= 14);
    }
}
