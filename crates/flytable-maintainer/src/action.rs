//! Seat actions: what a decision maker returns after seeing its view.
//!
//! The enums list every possibility; the table core decides legality and rejects
//! actions that do not exist in the variant (`Chi` is 4-player only, `Nukidora`
//! 3-player only).

use flytable_core::tile::Tile;

/// Declarations that another player may rob with ron.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobberyKind {
    /// Added kan; robbing it scores chankan.
    Kakan,
    /// Closed kan; robbable only where the rules allow it, and only by kokushi.
    Ankan,
    /// 3-player nukidora; the North can be robbed, but it does not score chankan.
    Nukidora,
}

/// Kan or nukidora declaration that is public but not yet followed by its replacement draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingRobbery {
    pub actor: u8,
    pub tile: Tile,
    pub kind: RobberyKind,
}

/// Actions on one's own turn.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnAction {
    /// Discard a tile (`tsumogiri` marks a discard of the drawn tile).
    Discard {
        tile: Tile,
        tsumogiri: bool,
    },
    /// The dealer's first discard (Mahjong Soul); the 14 tiles have no drawn/hand origin.
    DealerOpeningDiscard {
        tile: Tile,
    },
    /// Declare riichi and discard.
    Riichi {
        tile: Tile,
        tsumogiri: bool,
    },
    /// The dealer's first discard with riichi (Mahjong Soul).
    DealerOpeningRiichi {
        tile: Tile,
    },
    Ankan {
        tile: Tile,
    },
    /// Added kan onto an existing pon.
    Kakan {
        tile: Tile,
    },
    Tsumo,
    /// Nukidora (3-player only).
    Nukidora,
    /// Kyuushu kyuuhai (first turn only, when the condition holds).
    KyuushuKyuuhai,
}

impl TurnAction {
    /// Non-riichi discard usable as a conservative fallback (tsumogiri, from hand, or the
    /// originless dealer opening discard).
    pub const fn is_plain_discard(&self) -> bool {
        matches!(
            self,
            Self::Discard { .. } | Self::DealerOpeningDiscard { .. }
        )
    }
}

/// Responses after another player's discard (every non-discarder is polled).
#[derive(Debug, Clone, PartialEq)]
pub enum ReactionAction {
    Pass,
    /// Chi (4-player only, from the player to the left). `consumed` are the two tiles from hand.
    Chi {
        consumed: [Tile; 2],
    },
    Pon {
        consumed: [Tile; 2],
    },
    /// Open kan.
    Daiminkan,
    Ron,
}

impl ReactionAction {
    /// Semantic equality. Chi/pon consumed tiles are an unordered multiset, so a
    /// different array order is not a different action.
    pub fn equivalent(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Chi { consumed: left } | Self::Pon { consumed: left },
                Self::Chi { consumed: right } | Self::Pon { consumed: right },
            ) if std::mem::discriminant(self) == std::mem::discriminant(other) => {
                let mut left = *left;
                let mut right = *right;
                left.sort_by_key(|tile| tile.id());
                right.sort_by_key(|tile| tile.id());
                left == right
            }
            _ => self == other,
        }
    }
}

/// When a decision maker is asked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecisionKind {
    /// After drawing on one's own turn.
    Turn,
    /// After another player's discard, kan or nukidora.
    Reaction,
}
