//! Boundary conversion between 3-player events and mjai.
//!
//! Internally 3-player uses three seats (`[i32; 3]` scores, 3-seat events). The mjai
//! format pads 3-player to four entries; that conversion happens only here, at the
//! boundary, so the core never sees a fake fourth seat.

/// 3-seat scores to mjai's 4-entry form (fourth entry 0). Outbound boundary only.
pub fn scores_3_to_mjai4(s: [i32; 3]) -> [i32; 4] {
    [s[0], s[1], s[2], 0]
}

/// mjai 4-entry scores to 3 seats. Inbound boundary only. Returns `Err` if the fourth
/// entry is non-zero.
pub fn scores_mjai4_to_3(s: [i32; 4]) -> Result<[i32; 3], String> {
    if s[3] != 0 {
        return Err(format!(
            "3p mjai fourth seat placeholder must be 0, got {}",
            s[3]
        ));
    }
    Ok([s[0], s[1], s[2]])
}

/// 3-seat deltas to mjai's 4-entry form (fourth entry 0).
pub fn deltas_3_to_mjai4(d: [i32; 3]) -> [i32; 4] {
    [d[0], d[1], d[2], 0]
}
