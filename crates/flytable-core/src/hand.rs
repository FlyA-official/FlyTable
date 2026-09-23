//! Count-based hand representation: `[u8; 34]`, one slot per base kind, red fives
//! folded into plain fives.
//!
//! Shanten, win detection and scoring all work on this. Red fives only matter for
//! scoring and are tracked separately.

use crate::tile::{NUM_KINDS, Tile};

/// Tile counts over the 34 base kinds.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCounts([u8; NUM_KINDS]);

impl Default for TileCounts {
    fn default() -> Self {
        Self([0; NUM_KINDS])
    }
}

impl TileCounts {
    pub const fn new() -> Self {
        Self([0; NUM_KINDS])
    }

    pub const fn from_raw(raw: [u8; NUM_KINDS]) -> Self {
        Self(raw)
    }

    pub fn from_tiles<I: IntoIterator<Item = Tile>>(tiles: I) -> Self {
        let mut c = Self::new();
        for t in tiles {
            c.add(t, 1);
        }
        c
    }

    #[inline]
    pub fn count(&self, kind: usize) -> u8 {
        self.0[kind]
    }

    #[inline]
    pub fn count_of(&self, tile: Tile) -> u8 {
        self.0[tile.kind()]
    }

    #[inline]
    pub fn add(&mut self, tile: Tile, n: u8) {
        self.0[tile.kind()] += n;
    }

    #[inline]
    pub fn sub(&mut self, tile: Tile, n: u8) {
        self.0[tile.kind()] -= n;
    }

    pub fn total(&self) -> u32 {
        self.0.iter().map(|&x| x as u32).sum()
    }

    pub fn raw(&self) -> &[u8; NUM_KINDS] {
        &self.0
    }

    pub fn raw_mut(&mut self) -> &mut [u8; NUM_KINDS] {
        &mut self.0
    }

    /// Number of distinct kinds held.
    pub fn distinct(&self) -> u8 {
        self.0.iter().filter(|&&x| x > 0).count() as u8
    }

    /// Expands back into a tile list (plain fives, no red).
    pub fn to_tiles(&self) -> Vec<Tile> {
        let mut v = Vec::with_capacity(self.total() as usize);
        for (k, &n) in self.0.iter().enumerate() {
            // Safety: k < 34 < NUM_IDS
            let t = unsafe { Tile::from_id_unchecked(k as u8) };
            for _ in 0..n {
                v.push(t);
            }
        }
        v
    }
}

impl std::fmt::Debug for TileCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TileCounts[")?;
        let tiles = self.to_tiles();
        for (i, t) in tiles.iter().enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{t}")?;
        }
        f.write_str("]")
    }
}

/// Parses `"123m456p"`-style notation into `TileCounts`. Honors are `E S W N P F C`;
/// `0` in a suit is a red five (e.g. `05m`).
pub fn counts_from_str(s: &str) -> Result<TileCounts, String> {
    let mut c = TileCounts::new();
    let mut digits: Vec<char> = Vec::new();
    for ch in s.chars() {
        match ch {
            '0'..='9' => digits.push(ch),
            'm' | 'p' | 's' => {
                for d in digits.drain(..) {
                    let t = if d == '0' {
                        format!("5{ch}r")
                    } else {
                        format!("{d}{ch}")
                    };
                    let tile: Tile = t.parse().map_err(|_| format!("bad tile {t}"))?;
                    c.add(tile, 1);
                }
            }
            'E' | 'S' | 'W' | 'N' | 'P' | 'F' | 'C' => {
                let tile: Tile = ch.to_string().parse().map_err(|_| format!("bad {ch}"))?;
                c.add(tile, 1);
            }
            ' ' | ',' => {}
            _ => return Err(format!("unexpected char {ch}")),
        }
    }
    if !digits.is_empty() {
        return Err("trailing digits without suit".into());
    }
    Ok(c)
}
