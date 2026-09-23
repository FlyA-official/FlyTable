//! Settlement bridge from the table layer to the rules core.
//!
//! Builds a core [`YakuContext`] from the table state (hand, melds, dora
//! indicators, situational flags), calls the yaku/han/fu/points core, and returns
//! the settlement details.
//!
//! Dora han is counted here (indicators, red fives, ura dora, nukidora); yaku
//! evaluation itself never sees dora.

use flytable_core::meld::Meld;
use flytable_core::rules::RiichiRuleProfile;
use flytable_core::score::{FullScore, evaluate_full};
use flytable_core::tile::Tile;
use flytable_core::yaku::{Yaku, YakuContext};

use crate::player::PlayerState;

/// Table snapshot needed for settlement.
pub struct ScoringInput<'a> {
    /// Platform rules in effect.
    pub rule_profile: RiichiRuleProfile,
    pub player: &'a PlayerState,
    pub win_tile: Tile,
    pub is_tsumo: bool,
    pub bakaze: Tile,
    pub jikaze: Tile,
    pub is_oya: bool,
    pub dora_indicators: &'a [Tile],
    /// Ura dora indicators revealed for riichi (empty without riichi).
    pub ura_indicators: &'a [Tile],
    pub riichi: bool,
    pub double_riichi: bool,
    pub ippatsu: bool,
    pub haitei: bool,
    pub houtei: bool,
    pub rinshan: bool,
    pub chankan: bool,
    pub tenhou: bool,
    pub chiihou: bool,
    /// Number of nukidora (3-player).
    pub nuki_count: u8,
    /// 3-player (affects the manzu indicator wrap).
    pub is_sanma: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct DoraBreakdown {
    regular: u8,
    red: u8,
    ura: u8,
    nuki: u8,
}

impl DoraBreakdown {
    fn total(self) -> u8 {
        self.regular + self.red + self.ura + self.nuki
    }
}

/// Counts dora han, keeping the categories the event protocol needs.
fn count_dora(input: &ScoringInput) -> DoraBreakdown {
    let mut all_tiles: Vec<Tile> = input.player.hand.clone();
    for m in &input.player.melds {
        if !matches!(m, Meld::Nukidora { .. }) {
            all_tiles.extend(m.tiles());
        }
    }

    // Dora and ura dora: each indicator points at a kind, +1 per tile of that kind in
    // the hand. In 3-player, set-aside Norths are not in the hand but still count once
    // per nukidora when an indicator points at North, independent of the nukidora han
    // itself (a West indicator makes one North worth 1 for nukidora plus 1 dora).
    // North = kind 30 (E=27..C=33).
    const NORTH_KIND: usize = 30;
    let count_indicators = |indicators: &[Tile]| -> u8 {
        let mut count = 0u8;
        for ind in indicators {
            let target = if input.is_sanma {
                ind.dora_from_indicator_3p()
            } else {
                ind.dora_from_indicator()
            };
            count += all_tiles
                .iter()
                .filter(|t| t.kind() == target.kind())
                .count() as u8;
            if input.is_sanma && target.kind() == NORTH_KIND && input.nuki_count > 0 {
                count += input.nuki_count;
            }
        }
        count
    };
    let regular = count_indicators(input.dora_indicators);
    let ura = count_indicators(input.ura_indicators);
    // Red fives: count physical tiles in the closed hand and chi/pon. Kan melds cannot
    // record which tile is red, so the count comes from the profile: a kan uses all four
    // copies, so all red fives of that suit are in it.
    let seats = if input.is_sanma { 3 } else { 4 };
    let reds = input.rule_profile.red_fives(seats);
    let mut red = input.player.hand.iter().filter(|t| t.is_aka()).count() as u8;
    for m in &input.player.melds {
        red += match m {
            Meld::Nukidora { .. } => 0,
            m if m.is_kan() => m.kan_aka_count(reds),
            m => m.tiles().iter().filter(|t| t.is_aka()).count() as u8,
        };
    }
    let nuki = input
        .nuki_count
        .saturating_mul(input.rule_profile.nuki_dora_han());
    DoraBreakdown {
        regular,
        red,
        ura,
        nuki,
    }
}

/// Runs settlement. Returns `None` when there is no yaku.
pub fn settle(input: &ScoringInput) -> Option<FullScore> {
    // Closed-hand counts, red fives folded, winning tile included (already in player.hand).
    let counts = input.player.hand_counts();
    let ctx = YakuContext {
        rule_profile: input.rule_profile,
        concealed: *counts.raw(),
        melds: input.player.melds.clone(),
        win_tile: input.win_tile.deaka(),
        is_tsumo: input.is_tsumo,
        menzen: input.player.menzen,
        bakaze: input.bakaze,
        jikaze: input.jikaze,
        riichi: input.riichi,
        double_riichi: input.double_riichi,
        ippatsu: input.ippatsu,
        haitei: input.haitei,
        houtei: input.houtei,
        rinshan: input.rinshan,
        chankan: input.chankan,
        tenhou: input.tenhou,
        chiihou: input.chiihou,
        nuki_count: input.nuki_count,
        pei_is_yakuhai: false,
    };
    let dora = count_dora(input);
    let mut score = evaluate_full(&ctx, dora.total(), input.is_oya)?;
    if score.score.yakuman == 0 {
        score.regular_dora_han = dora.regular;
        score.red_dora_han = dora.red;
        score.ura_dora_han = dora.ura;
        score.nuki_dora_han = dora.nuki;
    }
    Some(score)
}

pub fn yaku_flags_from_score(score: &FullScore) -> flytable_event::HoraYakuFlags {
    let mut flags = flytable_event::HoraYakuFlags::default();
    for (yaku, _) in &score.yaku.yaku {
        match yaku {
            Yaku::Riichi => flags.riichi = true,
            Yaku::DoubleRiichi => flags.double_riichi = true,
            Yaku::Ippatsu => flags.ippatsu = true,
            Yaku::MenzenTsumo => flags.menzen_tsumo = true,
            Yaku::Haitei => flags.haitei = true,
            Yaku::Houtei => flags.houtei = true,
            Yaku::Rinshan => flags.rinshan = true,
            Yaku::Chankan => flags.chankan = true,
            Yaku::Yakuman(flytable_core::yaku::YakumanKind::Tenhou) => flags.tenhou = true,
            Yaku::Yakuman(flytable_core::yaku::YakumanKind::Chiihou) => flags.chiihou = true,
            _ => {}
        }
    }
    flags
}
