//! L0 and L1 projection from the engine event log to `flytable-matchlog-v1`.
//!
//! The input is [`Event4p`] / [`Event3p`], output the engine has already
//! adjudicated. This module only re-encodes structure: it makes timing facts
//! implicit in the log explicit as ruling events. It reads no rules or profile and
//! works purely from event adjacency, so it cannot disagree with the engine.
//!
//! | match log event | derived from |
//! |---|---|
//! | `dora.created_by` | FIFO order of kan declarations: the n-th `Dora` belongs to the n-th unmatched kan |
//! | `robbery_window` | after a kan or nukidora: only `open` if immediately followed by `Hora` targeting the declarer (robbed), otherwise `open` + `close` |
//! | `tsumo.source` | right after a kan: `rinshan`; right after a nukidora: `supplement`; otherwise `live` |
//! | `dahai.riichi_declare` | a `Reach` event immediately before |
//!
//! L2 (split deltas, draw kinds) needs settlement context and is added by the caller
//! (for example the runtime's match log archive) after [`project_4p`] / [`project_3p`].

use flytable_event::matchlog::{
    DoraCreatedBy, DrawSource, MatchlogEvent, MatchlogMeld, MeldKind, RobberyKind, TileRef,
    WindowEdge,
};
use flytable_event::{Event3p, Event4p};

/// The last declaration that affects the source of the next draw.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pending {
    None,
    /// Kan declaration, with its creator kind for FIFO matching of `Dora`.
    Kan(DoraCreatedBy, RobberyKind),
    /// Nukidora declaration (creates no indicator).
    Kita,
}

/// Rolling projection state.
struct Proj {
    out: Vec<MatchlogEvent>,
    /// Kan creators declared but not yet matched to a `Dora` event, FIFO.
    creators: std::collections::VecDeque<DoraCreatedBy>,
    pending: Pending,
    /// Whether the previous event was `Reach` (marks the declaration discard).
    reach_declared: bool,
    /// `(pon player, kind) -> source seat`. The engine's `Kakan` event has no source seat,
    /// but the earlier `Pon` does, and an added kan keeps it.
    pon_from: std::collections::HashMap<(u8, usize), u8>,
}

impl Proj {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            creators: std::collections::VecDeque::new(),
            pending: Pending::None,
            reach_declared: false,
            pon_from: std::collections::HashMap::new(),
        }
    }

    /// Closes the window normally after a declaration unless the next event is a rob.
    ///
    /// A successful rob emits only `open`: the window ends in a win.
    fn close_window_if_open(&mut self, robbed: bool) {
        let kind = match self.pending {
            Pending::Kan(_, k) => k,
            Pending::Kita => RobberyKind::Kita,
            Pending::None => return,
        };
        if !robbed {
            self.out.push(MatchlogEvent::RobberyWindow {
                kind,
                edge: WindowEdge::Close,
            });
        }
    }

    fn open_window(&mut self, kind: RobberyKind) {
        self.out.push(MatchlogEvent::RobberyWindow {
            kind,
            edge: WindowEdge::Open,
        });
    }

    /// Source of the current draw, determined by the preceding declaration.
    fn draw_source(&mut self) -> DrawSource {
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::Kan(..) => DrawSource::Rinshan,
            Pending::Kita => DrawSource::Supplement,
            Pending::None => DrawSource::Live,
        }
    }
}

fn tr(t: flytable_core::tile::Tile) -> TileRef {
    // The engine works at kind level, so engine-generated logs have no physical identity.
    TileRef::opaque(t)
}

fn trs(ts: &[flytable_core::tile::Tile]) -> Vec<TileRef> {
    ts.iter().copied().map(tr).collect()
}

/// 4-player projection.
#[must_use]
pub fn project_4p(events: &[Event4p]) -> Vec<MatchlogEvent> {
    project_4p_indexed(events).0
}

/// 4-player projection plus a prefix cursor map from table index to match log index.
///
/// `map` has length `events.len() + 1`, and `map[i]` is the number of match log
/// events produced after the first `i` table events. Anything anchored at table
/// cursor `b` has match log cursor `map[b]`.
///
/// The streams are not 1:1: `Reach` folds into the next `Dahai` (1 -> 0 events),
/// `Kakan` / `Ankan` expand into `Call` plus `RobberyWindow` (1 -> 2 or 3), and some
/// table events are not projected at all. The host only knows the table cursor, so
/// the conversion belongs in the projection layer.
#[must_use]
pub fn project_4p_indexed(events: &[Event4p]) -> (Vec<MatchlogEvent>, Vec<usize>) {
    let mut p = Proj::new();
    let mut map = Vec::with_capacity(events.len() + 1);
    let mut i = 0;
    while i < events.len() {
        map.push(p.out.len());
        let ev = &events[i];
        let next = events.get(i + 1);
        match ev {
            Event4p::Tsumo { actor, pai } => {
                let source = p.draw_source();
                p.out.push(MatchlogEvent::Tsumo {
                    actor: *actor,
                    pai: tr(*pai),
                    source,
                });
            }
            Event4p::DealerOpening { actor, pai } => {
                p.out.push(MatchlogEvent::DealerOpening {
                    actor: *actor,
                    pai: tr(*pai),
                });
            }
            Event4p::Dahai {
                actor,
                pai,
                tsumogiri,
            } => {
                p.out.push(MatchlogEvent::Dahai {
                    actor: *actor,
                    pai: tr(*pai),
                    tsumogiri: *tsumogiri,
                    riichi_declare: std::mem::take(&mut p.reach_declared),
                });
            }
            Event4p::DealerOpeningDahai { actor, pai } => {
                p.out.push(MatchlogEvent::DealerOpeningDahai {
                    actor: *actor,
                    pai: tr(*pai),
                });
            }
            Event4p::Chi {
                actor,
                target,
                pai,
                consumed,
            } => {
                p.out.push(MatchlogEvent::Call {
                    actor: *actor,
                    meld: MatchlogMeld {
                        kind: MeldKind::Chi,
                        from: Some(*target),
                        claimed: Some(tr(*pai)),
                        consumed: trs(consumed),
                    },
                });
            }
            Event4p::Pon {
                actor,
                target,
                pai,
                consumed,
            } => {
                // Remember the source seat for a later added kan.
                p.pon_from.insert((*actor, pai.deaka().kind()), *target);
                p.out.push(MatchlogEvent::Call {
                    actor: *actor,
                    meld: MatchlogMeld {
                        kind: MeldKind::Pon,
                        from: Some(*target),
                        claimed: Some(tr(*pai)),
                        consumed: trs(consumed),
                    },
                });
            }
            Event4p::Daiminkan {
                actor,
                target,
                pai,
                consumed,
            } => {
                p.out.push(MatchlogEvent::Call {
                    actor: *actor,
                    meld: MatchlogMeld {
                        kind: MeldKind::Daiminkan,
                        from: Some(*target),
                        claimed: Some(tr(*pai)),
                        consumed: trs(consumed),
                    },
                });
                // An open kan has no robbing window, but it creates an indicator (possibly revealed later).
                p.creators.push_back(DoraCreatedBy::Daiminkan);
                // No robbing window, so no window event, but it still takes a `Dora` slot.
                p.pending = Pending::Kan(DoraCreatedBy::Daiminkan, RobberyKind::Kakan);
            }
            Event4p::Kakan {
                actor,
                pai,
                consumed,
            } => {
                let from = p.pon_from.get(&(*actor, pai.deaka().kind())).copied();
                p.out.push(MatchlogEvent::Call {
                    actor: *actor,
                    meld: MatchlogMeld {
                        kind: MeldKind::Kakan,
                        from,
                        claimed: Some(tr(*pai)),
                        consumed: trs(consumed),
                    },
                });
                p.creators.push_back(DoraCreatedBy::Kakan);
                p.pending = Pending::Kan(DoraCreatedBy::Kakan, RobberyKind::Kakan);
                p.open_window(RobberyKind::Kakan);
                p.close_window_if_open(matches!(next, Some(Event4p::Hora { .. })));
            }
            Event4p::Ankan { actor, consumed } => {
                p.out.push(MatchlogEvent::Call {
                    actor: *actor,
                    meld: MatchlogMeld {
                        kind: MeldKind::Ankan,
                        from: None,
                        claimed: None,
                        consumed: trs(consumed),
                    },
                });
                p.creators.push_back(DoraCreatedBy::Ankan);
                p.pending = Pending::Kan(DoraCreatedBy::Ankan, RobberyKind::Ankan);
                p.open_window(RobberyKind::Ankan);
                p.close_window_if_open(matches!(next, Some(Event4p::Hora { .. })));
            }
            Event4p::Dora { dora_marker } => {
                // FIFO: the n-th `Dora` belongs to the n-th unmatched kan.
                let created_by = p.creators.pop_front().unwrap_or(DoraCreatedBy::Kakan);
                p.out.push(MatchlogEvent::Dora {
                    marker: tr(*dora_marker),
                    created_by,
                });
            }
            Event4p::Reach { .. } => p.reach_declared = true,
            Event4p::ReachAccepted { actor, .. } => {
                p.out.push(MatchlogEvent::ReachAccepted { actor: *actor });
            }
            _ => {}
        }
        i += 1;
    }
    map.push(p.out.len());
    (p.out, map)
}

/// 3-player projection (no chi, has nukidora).
#[must_use]
pub fn project_3p(events: &[Event3p]) -> Vec<MatchlogEvent> {
    project_3p_indexed(events).0
}

/// 3-player projection plus the prefix cursor map (see [`project_4p_indexed`]).
#[must_use]
pub fn project_3p_indexed(events: &[Event3p]) -> (Vec<MatchlogEvent>, Vec<usize>) {
    let mut p = Proj::new();
    let mut map = Vec::with_capacity(events.len() + 1);
    let mut i = 0;
    while i < events.len() {
        map.push(p.out.len());
        let ev = &events[i];
        let next = events.get(i + 1);
        match ev {
            Event3p::Tsumo { actor, pai } => {
                let source = p.draw_source();
                p.out.push(MatchlogEvent::Tsumo {
                    actor: *actor,
                    pai: tr(*pai),
                    source,
                });
            }
            Event3p::DealerOpening { actor, pai } => {
                p.out.push(MatchlogEvent::DealerOpening {
                    actor: *actor,
                    pai: tr(*pai),
                });
            }
            Event3p::Dahai {
                actor,
                pai,
                tsumogiri,
            } => {
                p.out.push(MatchlogEvent::Dahai {
                    actor: *actor,
                    pai: tr(*pai),
                    tsumogiri: *tsumogiri,
                    riichi_declare: std::mem::take(&mut p.reach_declared),
                });
            }
            Event3p::DealerOpeningDahai { actor, pai } => {
                p.out.push(MatchlogEvent::DealerOpeningDahai {
                    actor: *actor,
                    pai: tr(*pai),
                });
            }
            Event3p::Pon {
                actor,
                target,
                pai,
                consumed,
            } => {
                // Remember the source seat for a later added kan.
                p.pon_from.insert((*actor, pai.deaka().kind()), *target);
                p.out.push(MatchlogEvent::Call {
                    actor: *actor,
                    meld: MatchlogMeld {
                        kind: MeldKind::Pon,
                        from: Some(*target),
                        claimed: Some(tr(*pai)),
                        consumed: trs(consumed),
                    },
                });
            }
            Event3p::Daiminkan {
                actor,
                target,
                pai,
                consumed,
            } => {
                p.out.push(MatchlogEvent::Call {
                    actor: *actor,
                    meld: MatchlogMeld {
                        kind: MeldKind::Daiminkan,
                        from: Some(*target),
                        claimed: Some(tr(*pai)),
                        consumed: trs(consumed),
                    },
                });
                p.creators.push_back(DoraCreatedBy::Daiminkan);
                p.pending = Pending::Kan(DoraCreatedBy::Daiminkan, RobberyKind::Kakan);
            }
            Event3p::Kakan {
                actor,
                pai,
                consumed,
            } => {
                let from = p.pon_from.get(&(*actor, pai.deaka().kind())).copied();
                p.out.push(MatchlogEvent::Call {
                    actor: *actor,
                    meld: MatchlogMeld {
                        kind: MeldKind::Kakan,
                        from,
                        claimed: Some(tr(*pai)),
                        consumed: trs(consumed),
                    },
                });
                p.creators.push_back(DoraCreatedBy::Kakan);
                p.pending = Pending::Kan(DoraCreatedBy::Kakan, RobberyKind::Kakan);
                p.open_window(RobberyKind::Kakan);
                p.close_window_if_open(matches!(next, Some(Event3p::Hora { .. })));
            }
            Event3p::Ankan { actor, consumed } => {
                p.out.push(MatchlogEvent::Call {
                    actor: *actor,
                    meld: MatchlogMeld {
                        kind: MeldKind::Ankan,
                        from: None,
                        claimed: None,
                        consumed: trs(consumed),
                    },
                });
                p.creators.push_back(DoraCreatedBy::Ankan);
                p.pending = Pending::Kan(DoraCreatedBy::Ankan, RobberyKind::Ankan);
                p.open_window(RobberyKind::Ankan);
                p.close_window_if_open(matches!(next, Some(Event3p::Hora { .. })));
            }
            Event3p::Nukidora { actor, pai } => {
                p.out.push(MatchlogEvent::Call {
                    actor: *actor,
                    meld: MatchlogMeld {
                        kind: MeldKind::Kita,
                        from: None,
                        claimed: Some(tr(*pai)),
                        consumed: vec![],
                    },
                });
                // Nukidora creates no indicator, so it is not queued as a creator.
                p.pending = Pending::Kita;
                p.open_window(RobberyKind::Kita);
                p.close_window_if_open(matches!(next, Some(Event3p::Hora { .. })));
            }
            Event3p::Dora { dora_marker } => {
                let created_by = p.creators.pop_front().unwrap_or(DoraCreatedBy::Kakan);
                p.out.push(MatchlogEvent::Dora {
                    marker: tr(*dora_marker),
                    created_by,
                });
            }
            Event3p::Reach { .. } => p.reach_declared = true,
            Event3p::ReachAccepted { actor, .. } => {
                p.out.push(MatchlogEvent::ReachAccepted { actor: *actor });
            }
            _ => {}
        }
        i += 1;
    }
    map.push(p.out.len());
    (p.out, map)
}
