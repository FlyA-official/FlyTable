//! Rough settlement helper (han and fu to points), kept for API compatibility.
//!
//! The boards do not use it; full yaku, han, fu and points come from
//! `flytable-core::score` through `flytable-maintainer::scoring`.

/// Rough uncapped ron points for the given han, fu and dealer flag. Use
/// `flytable_maintainer::scoring::settle` in new code.
pub fn placeholder_ron_points(han: u8, fu: u8, is_oya: bool) -> i32 {
    let base = (fu as i32) * (1 << (2 + han as i32));
    let mult = if is_oya { 6 } else { 4 };
    round_up_100(base * mult)
}

fn round_up_100(x: i32) -> i32 {
    ((x + 99) / 100) * 100
}
