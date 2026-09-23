//! Platform rule profiles.
//!
//! `riichi4p` / `riichi3p` only describe seat count and action shapes. Tenhou,
//! Mahjong Soul and Riichi City differ in details such as yakuman multipliers, so
//! every difference that affects adjudication is an explicit field here and is
//! passed down with the table, mirror and seat views. Never branch on a product
//! name at the call site.

use std::fmt;
use std::str::FromStr;

/// Platform a rule profile comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RulePlatform {
    Tenhou,
    MahjongSoul,
    RiichiCity,
}

impl RulePlatform {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tenhou => "tenhou",
            Self::MahjongSoul => "mahjong-soul",
            Self::RiichiCity => "riichi-city",
        }
    }
}

impl fmt::Display for RulePlatform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for RulePlatform {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "tenhou" => Ok(Self::Tenhou),
            "majsoul" | "mahjong-soul" | "mahjong_soul" => Ok(Self::MahjongSoul),
            "riichi-city" | "riichi_city" | "mahjong-ichibangai" => Ok(Self::RiichiCity),
            other => Err(format!(
                "unknown rule platform {other:?} (expected tenhou / mahjong-soul / riichi-city)"
            )),
        }
    }
}

/// How multiple valid ron calls on one discard are resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RonResolution {
    /// Only the closest player to the discarder wins (head bump).
    AtamaHane,
    /// Every valid call wins.
    Multiple,
    /// Double ron is allowed; a triple ron in 4-player is an abortive draw.
    TripleRonAbortive,
}

/// Cap for counted yakuman (13+ han from regular yaku and dora).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum KazoeLimit {
    /// 13+ han scores as a single yakuman.
    Yakuman,
    /// 13+ han is capped at sanbaiman.
    Sanbaiman,
}

/// Restriction on closed kan after riichi.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RiichiAnkanRule {
    Forbidden,
    /// Only with the drawn fourth tile, and only if the set of waits is unchanged.
    DrawnFourthAndWaitsUnchanged,
}

/// When kan dora indicators are revealed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum KanDoraTiming {
    /// Closed kan reveals before the replacement draw; open and added kans reveal after
    /// the replacement draw and before the next discard.
    AnkanImmediateOpenKanDelayed,
    /// Every kan reveals before the replacement draw.
    Immediate,
}

/// Red five count per suit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RedFiveCounts {
    pub man: u8,
    pub pin: u8,
    pub sou: u8,
}

impl RedFiveCounts {
    /// Checks that the counts can form a physical tile set for the given seat count.
    pub fn validate_for_players(self, seats: usize) -> Result<(), String> {
        validate_red_fives(seats, self)
    }
}

/// Copyable riichi rule profile for a platform.
///
/// Covers only the platform differences FlyTable implements and has verified; it
/// is not a complete specification of any platform. Add a field for each new
/// difference rather than branching on the platform elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RiichiRuleProfile {
    platform: RulePlatform,
    /// Open tanyao (kuitan).
    open_tanyao: bool,
    /// Red fives in a standard 4-player wall.
    red_fives_4p: RedFiveCounts,
    /// Red fives in a standard 3-player wall.
    red_fives_3p: RedFiveCounts,
    /// Starting score (4-player / 3-player).
    start_score_4p: i32,
    start_score_3p: i32,
    /// Target score (4-player / 3-player), used for extension and game-end checks.
    target_score_4p: i32,
    target_score_3p: i32,
    /// End the game when a score drops below zero (exactly 0 continues).
    bust_below_zero: bool,
    /// Maximum number of extra rounds after the scheduled last round.
    max_extra_bakaze: u8,
    /// Agariyame: in the final hand, a winning dealer in first place at or above the target may end the game.
    agariyame: bool,
    /// Tenpaiyame: same as `agariyame`, for a dealer tenpai at an exhaustive draw.
    tenpaiyame: bool,
    /// Forbid swap calling (same-tile and suji kuikae).
    kuikae_forbidden: bool,
    /// Round 4 han 30 fu and 3 han 60 fu up to mangan (kiriage).
    kiriage_mangan: bool,
    kazoe_limit: KazoeLimit,
    /// Whether 13-sided kokushi, suuankou tanki, junsei chuuren and daisuushi each score double yakuman.
    double_special_yakuman: bool,
    ron_resolution: RonResolution,
    /// Whether kokushi may rob a closed kan.
    kokushi_ankan_ron: bool,
    /// Whether a nukidora may be robbed with ron in 3-player.
    nukidora_ron: bool,
    nagashi_mangan: bool,
    /// Pao (liability payment) for daisangen and daisuushi.
    pao: bool,
    /// Abortive draw on four kans by different players.
    suukaikan: bool,
    /// Abortive draw on four riichi (4-player).
    suucha_riichi: bool,
    /// Whether riichi may be declared with no draws left (including declaring on the haitei discard).
    no_draw_riichi: bool,
    riichi_ankan: RiichiAnkanRule,
    kan_dora_timing: KanDoraTiming,
    /// On consecutive kans, whether the pending indicator from the previous kan is
    /// revealed at the moment of the next kan or nukidora declaration (Mahjong Soul).
    /// When `false`, it is revealed after the robbing window and before the replacement
    /// draw (Tenhou).
    kan_dora_pending_flush_at_declaration: bool,
    /// In 3-player, whether a nukidora declaration after the fourth kan triggers the
    /// four-kan abortive draw. Tenhou triggers it immediately without a replacement
    /// draw; Mahjong Soul does not, and a later rinshan tsumo still scores.
    suukaikan_on_kita_declaration: bool,
    /// Han each nukidora adds by itself, independent of indicators pointing at North.
    nuki_dora_han: u8,
}

impl Default for RiichiRuleProfile {
    /// Defaults to Tenhou; integrations should still call the platform constructor explicitly.
    fn default() -> Self {
        Self::tenhou()
    }
}

impl RiichiRuleProfile {
    pub const fn platform(self) -> RulePlatform {
        self.platform
    }

    pub const fn allows_open_tanyao(self) -> bool {
        self.open_tanyao
    }

    pub const fn red_fives(self, seats: usize) -> RedFiveCounts {
        if seats == 3 {
            self.red_fives_3p
        } else {
            self.red_fives_4p
        }
    }

    /// Overrides the red five counts for the given seat count.
    ///
    /// Red fives replace plain fives, so each suit allows `0..=4`. The 3-player wall has
    /// no 5m, so `man` must be 0. Returns `Result` so an impossible profile is rejected
    /// before shuffling.
    pub fn with_red_fives(mut self, seats: usize, counts: RedFiveCounts) -> Result<Self, String> {
        validate_red_fives(seats, counts)?;
        match seats {
            4 => self.red_fives_4p = counts,
            3 => self.red_fives_3p = counts,
            _ => unreachable!("validate_red_fives accepted unsupported seat count"),
        }
        Ok(self)
    }

    /// Checks that the profile can form a physical wall for the given seat count.
    pub fn validate_for_players(self, seats: usize) -> Result<(), String> {
        self.red_fives(seats).validate_for_players(seats)
    }

    pub const fn start_score(self, seats: usize) -> i32 {
        if seats == 3 {
            self.start_score_3p
        } else {
            self.start_score_4p
        }
    }

    pub const fn target_score(self, seats: usize) -> i32 {
        if seats == 3 {
            self.target_score_3p
        } else {
            self.target_score_4p
        }
    }

    pub const fn bust_below_zero(self) -> bool {
        self.bust_below_zero
    }

    pub const fn max_extra_bakaze(self) -> u8 {
        self.max_extra_bakaze
    }

    pub const fn allows_agariyame(self) -> bool {
        self.agariyame
    }

    pub const fn allows_tenpaiyame(self) -> bool {
        self.tenpaiyame
    }

    pub const fn kuikae_forbidden(self) -> bool {
        self.kuikae_forbidden
    }

    pub const fn kiriage_mangan(self) -> bool {
        self.kiriage_mangan
    }

    pub const fn kazoe_limit(self) -> KazoeLimit {
        self.kazoe_limit
    }

    pub const fn double_special_yakuman(self) -> bool {
        self.double_special_yakuman
    }

    pub const fn ron_resolution(self) -> RonResolution {
        self.ron_resolution
    }

    pub const fn allows_kokushi_ankan_ron(self) -> bool {
        self.kokushi_ankan_ron
    }

    pub const fn allows_nukidora_ron(self) -> bool {
        self.nukidora_ron
    }

    pub const fn allows_nagashi_mangan(self) -> bool {
        self.nagashi_mangan
    }

    pub const fn allows_pao(self) -> bool {
        self.pao
    }

    pub const fn allows_suukaikan(self) -> bool {
        self.suukaikan
    }

    pub const fn allows_no_draw_riichi(self) -> bool {
        self.no_draw_riichi
    }

    pub const fn allows_suucha_riichi(self) -> bool {
        self.suucha_riichi
    }

    pub const fn riichi_ankan_rule(self) -> RiichiAnkanRule {
        self.riichi_ankan
    }

    pub const fn kan_dora_timing(self) -> KanDoraTiming {
        self.kan_dora_timing
    }

    pub const fn kan_dora_pending_flush_at_declaration(self) -> bool {
        self.kan_dora_pending_flush_at_declaration
    }

    pub const fn suukaikan_on_kita_declaration(self) -> bool {
        self.suukaikan_on_kita_declaration
    }

    pub const fn nuki_dora_han(self) -> u8 {
        self.nuki_dora_han
    }

    /// Room-level variant for head bump / multiple ron.
    pub const fn with_ron_resolution(mut self, ron_resolution: RonResolution) -> Self {
        self.ron_resolution = ron_resolution;
        self
    }

    /// Room-level variant for riichi with no draws left.
    pub const fn with_no_draw_riichi(mut self, enabled: bool) -> Self {
        self.no_draw_riichi = enabled;
        self
    }

    pub const fn with_open_tanyao(mut self, enabled: bool) -> Self {
        self.open_tanyao = enabled;
        self
    }

    /// Room-level variant for starting and target score.
    pub const fn with_match_points(
        mut self,
        start_score_4p: i32,
        target_score_4p: i32,
        start_score_3p: i32,
        target_score_3p: i32,
    ) -> Self {
        self.start_score_4p = start_score_4p;
        self.target_score_4p = target_score_4p;
        self.start_score_3p = start_score_3p;
        self.target_score_3p = target_score_3p;
        self
    }

    /// Room-level variant for game-end rules.
    pub const fn with_end_rules(
        mut self,
        bust_below_zero: bool,
        max_extra_bakaze: u8,
        agariyame: bool,
        tenpaiyame: bool,
    ) -> Self {
        self.bust_below_zero = bust_below_zero;
        self.max_extra_bakaze = max_extra_bakaze;
        self.agariyame = agariyame;
        self.tenpaiyame = tenpaiyame;
        self
    }

    /// Room-level variant for kuikae.
    pub const fn with_kuikae_forbidden(mut self, forbidden: bool) -> Self {
        self.kuikae_forbidden = forbidden;
        self
    }

    /// Room-level variant for scoring caps.
    pub const fn with_scoring_limits(
        mut self,
        kiriage_mangan: bool,
        kazoe_limit: KazoeLimit,
    ) -> Self {
        self.kiriage_mangan = kiriage_mangan;
        self.kazoe_limit = kazoe_limit;
        self
    }

    /// Room-level variant for closed kan after riichi.
    pub const fn with_riichi_ankan_rule(mut self, rule: RiichiAnkanRule) -> Self {
        self.riichi_ankan = rule;
        self
    }

    /// Room-level variant for kan dora timing.
    pub const fn with_kan_dora_timing(mut self, timing: KanDoraTiming) -> Self {
        self.kan_dora_timing = timing;
        self
    }

    pub const fn with_kan_dora_pending_flush_at_declaration(mut self, enabled: bool) -> Self {
        self.kan_dora_pending_flush_at_declaration = enabled;
        self
    }

    pub const fn with_suukaikan_on_kita_declaration(mut self, enabled: bool) -> Self {
        self.suukaikan_on_kita_declaration = enabled;
        self
    }

    pub const fn tenhou() -> Self {
        Self {
            platform: RulePlatform::Tenhou,
            open_tanyao: true,
            red_fives_4p: RedFiveCounts {
                man: 1,
                pin: 1,
                sou: 1,
            },
            red_fives_3p: RedFiveCounts {
                man: 0,
                pin: 1,
                sou: 1,
            },
            start_score_4p: 25_000,
            start_score_3p: 35_000,
            target_score_4p: 30_000,
            target_score_3p: 40_000,
            bust_below_zero: true,
            max_extra_bakaze: 1,
            agariyame: true,
            tenpaiyame: true,
            kuikae_forbidden: true,
            kiriage_mangan: false,
            kazoe_limit: KazoeLimit::Yakuman,
            double_special_yakuman: false,
            ron_resolution: RonResolution::TripleRonAbortive,
            kokushi_ankan_ron: false,
            nukidora_ron: true,
            nagashi_mangan: true,
            pao: true,
            suukaikan: true,
            suucha_riichi: true,
            // Tenhou requires 1000+ points and a remaining draw to declare riichi. Match logs
            // do not record this option, so the documented rule is used.
            no_draw_riichi: false,
            riichi_ankan: RiichiAnkanRule::DrawnFourthAndWaitsUnchanged,
            kan_dora_timing: KanDoraTiming::AnkanImmediateOpenKanDelayed,
            // Tenhou reveals the pending indicator together with the next kan's after its
            // robbing window; if the kan is robbed it is not revealed.
            kan_dora_pending_flush_at_declaration: false,
            // Tenhou: passing the nukidora window triggers the four-kan draw, with no replacement draw.
            suukaikan_on_kita_declaration: true,
            nuki_dora_han: 1,
        }
    }

    pub const fn mahjong_soul() -> Self {
        Self {
            platform: RulePlatform::MahjongSoul,
            open_tanyao: true,
            red_fives_4p: RedFiveCounts {
                man: 1,
                pin: 1,
                sou: 1,
            },
            red_fives_3p: RedFiveCounts {
                man: 0,
                pin: 1,
                sou: 1,
            },
            start_score_4p: 25_000,
            start_score_3p: 35_000,
            target_score_4p: 30_000,
            target_score_3p: 40_000,
            bust_below_zero: true,
            max_extra_bakaze: 1,
            agariyame: true,
            tenpaiyame: true,
            kuikae_forbidden: true,
            // Not specified in Mahjong Soul's public FAQ; defaults to no kiriage.
            kiriage_mangan: false,
            kazoe_limit: KazoeLimit::Yakuman,
            double_special_yakuman: true,
            ron_resolution: RonResolution::Multiple,
            kokushi_ankan_ron: true,
            nukidora_ron: true,
            nagashi_mangan: true,
            pao: true,
            suukaikan: true,
            suucha_riichi: true,
            // Not specified in Mahjong Soul's public FAQ; defaults to forbidden.
            no_draw_riichi: false,
            riichi_ankan: RiichiAnkanRule::DrawnFourthAndWaitsUnchanged,
            kan_dora_timing: KanDoraTiming::AnkanImmediateOpenKanDelayed,
            // Mahjong Soul reveals the pending indicator when the next kan or nukidora is
            // declared, so a robbing player gets it.
            kan_dora_pending_flush_at_declaration: true,
            // Mahjong Soul: nukidora does not trigger the four-kan draw; the replacement draw
            // and a following rinshan tsumo proceed normally.
            suukaikan_on_kita_declaration: false,
            nuki_dora_han: 1,
        }
    }

    pub const fn riichi_city() -> Self {
        Self {
            platform: RulePlatform::RiichiCity,
            open_tanyao: true,
            red_fives_4p: RedFiveCounts {
                man: 1,
                pin: 1,
                sou: 1,
            },
            red_fives_3p: RedFiveCounts {
                man: 0,
                pin: 1,
                sou: 1,
            },
            start_score_4p: 25_000,
            start_score_3p: 35_000,
            target_score_4p: 30_000,
            target_score_3p: 40_000,
            bust_below_zero: true,
            max_extra_bakaze: 1,
            agariyame: true,
            tenpaiyame: true,
            kuikae_forbidden: true,
            // No citable official rule page for Riichi City; conservative default.
            kiriage_mangan: false,
            kazoe_limit: KazoeLimit::Yakuman,
            double_special_yakuman: true,
            ron_resolution: RonResolution::Multiple,
            kokushi_ankan_ron: true,
            nukidora_ron: true,
            nagashi_mangan: true,
            pao: true,
            suukaikan: true,
            suucha_riichi: true,
            // No citable official rule page for Riichi City; conservative default.
            no_draw_riichi: false,
            riichi_ankan: RiichiAnkanRule::DrawnFourthAndWaitsUnchanged,
            kan_dora_timing: KanDoraTiming::AnkanImmediateOpenKanDelayed,
            // No evidence for Riichi City; follows Tenhou.
            kan_dora_pending_flush_at_declaration: false,
            // No evidence for Riichi City; follows Tenhou.
            suukaikan_on_kita_declaration: true,
            nuki_dora_han: 1,
        }
    }

    pub const fn from_platform(platform: RulePlatform) -> Self {
        match platform {
            RulePlatform::Tenhou => Self::tenhou(),
            RulePlatform::MahjongSoul => Self::mahjong_soul(),
            RulePlatform::RiichiCity => Self::riichi_city(),
        }
    }
}

fn validate_red_fives(seats: usize, counts: RedFiveCounts) -> Result<(), String> {
    if !matches!(seats, 3 | 4) {
        return Err(format!("rule profile supports 3 or 4 seats, got {seats}"));
    }
    for (suit, count) in [
        ("man", counts.man),
        ("pin", counts.pin),
        ("sou", counts.sou),
    ] {
        if count > 4 {
            return Err(format!(
                "{seats}p {suit} red five count must be in 0..=4, got {count}"
            ));
        }
    }
    if seats == 3 && counts.man != 0 {
        return Err(format!(
            "3p wall has no 5m, so the man red five count must be 0, got {}",
            counts.man
        ));
    }
    Ok(())
}

impl fmt::Display for RiichiRuleProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.platform.fmt(f)
    }
}

impl FromStr for RiichiRuleProfile {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        value.parse().map(Self::from_platform)
    }
}
