//! Pure rule computations shared by 4-player and 3-player mahjong: tiles, hands,
//! melds, shanten, win detection, yaku and scoring.
//!
//! Nothing here depends on game progression state, and there is no model-specific
//! encoding (observations, action spaces, tensors).

pub mod agari;
pub mod calc;
pub mod decompose;
pub mod hand;
pub mod meld;
pub mod rules;
pub mod score;
pub mod shanten;
pub mod tile;
pub mod yaku;
pub mod yaku_star;

pub use hand::TileCounts;
pub use meld::Meld;
pub use tile::{Suit, Tile};
