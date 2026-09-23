//! Timing checks shared by 3P and 4P.
//!
//! `Board3p` and `Board4p` are separate on purpose (no chi, nukidora, different tile
//! sets and settlement), but a few timing checks are identical in both. Keeping two
//! copies means every fix has to be applied twice, so these pure checks live here.
//! They do not touch event types or board internals.

/// Condition for the four-kan abortive draw. All four must hold:
///
/// 1. the rule is enabled (`allows_suukaikan`);
/// 2. at least four kans on the table;
/// 3. kans held by at least two players (four kans by one player is suukantsu, a
///    yakuman, not a draw);
/// 4. the fourth kan has taken effect: a closed kan immediately, an open or added kan
///    once its delayed dora is revealed (`pending_kan_dora == 0`).
///
/// Condition 4 matters in 3-player on Tenhou:
/// * closed kan as the fourth kan, then nukidora: the draw happens as soon as the
///   window passes, with no replacement draw;
/// * added kan as the fourth kan: `kita` -> `DORA` -> `tsumo` -> `dahai` -> window
///   passes -> `ryukyoku`.
///
/// Without condition 4 the second sequence would end too early.
#[must_use]
pub fn suukaikan_condition_met(
    allows_suukaikan: bool,
    kan_counts_per_seat: impl IntoIterator<Item = usize>,
    pending_kan_dora: u8,
) -> bool {
    if !allows_suukaikan {
        return false;
    }
    let mut total = 0usize;
    let mut owners = 0usize;
    for count in kan_counts_per_seat {
        total += count;
        if count > 0 {
            owners += 1;
        }
    }
    total >= 4 && owners >= 2 && pending_kan_dora == 0
}
