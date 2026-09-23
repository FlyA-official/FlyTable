//! Canonical versioned wire types: `flya-actions-v2` and `flya-mahjong-events-v2`,
//! with frozen v1 decoding.
//!
//! This is the public canonical contract for model decisions, separate from the
//! host-translator protocol `flya-inference-v2`:
//!
//! - [`CanonicalAction`] (`flya-actions-v2`): a plain `dahai` and a riichi discard
//!   `riichi_dahai` are separate tagged variants rather than one `riichi: bool`, and
//!   every legal action carries a stable `action_id` ([`CanonicalLegalAction`]).
//!   3-player nukidora is named `kita` and accepts the `nukidora` alias on input. v2
//!   adds the originless dealer opening discard.
//! - [`MahjongEvents`] (`flya-mahjong-events-v2`): per-seat visible event envelope
//!   reusing [`Event4p`] / [`Event3p`], adding version, rule line, visibility source
//!   and seq metadata. Reads frozen v1 and v2, writes v2, rejects other majors.
//!
//! [`CanonicalAction::from_legacy_value`] / [`CanonicalAction::to_legal_4p`] and
//! friends convert to and from the inference wire. The frozen v1
//! `{type:"dahai", riichi:bool}` shape is unchanged.

use serde::{Deserialize, Serialize};

use flytable_core::tile::Tile;
use flytable_event::{Event3p, Event4p};

use crate::{LegalAction3p, LegalAction4p, ResponseOpportunities, RuleLine};

/// `flya-actions-v1` contract name.
pub const FLYA_ACTIONS_V1: &str = "flya-actions-v1";
/// Canonical action contract name including the originless dealer opening discard.
pub const FLYA_ACTIONS_V2: &str = "flya-actions-v2";
/// `flya-mahjong-events-v1` contract name.
pub const FLYA_MAHJONG_EVENTS_V1: &str = "flya-mahjong-events-v1";
/// Canonical envelope contract name including `dealer_opening*` events.
pub const FLYA_MAHJONG_EVENTS_V2: &str = "flya-mahjong-events-v2";

/// Conversion error between canonical actions and internal types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalConvertError(pub String);

impl std::fmt::Display for CanonicalConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for CanonicalConvertError {}

fn err<T>(msg: impl Into<String>) -> Result<T, CanonicalConvertError> {
    Err(CanonicalConvertError(msg.into()))
}

/// Rule line on the wire (`riichi4p` / `riichi3p`), convertible to [`RuleLine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleLineWire {
    #[serde(rename = "riichi4p")]
    Riichi4p,
    #[serde(rename = "riichi3p")]
    Riichi3p,
}

impl From<RuleLine> for RuleLineWire {
    fn from(value: RuleLine) -> Self {
        match value {
            RuleLine::Riichi4p => RuleLineWire::Riichi4p,
            RuleLine::Riichi3p => RuleLineWire::Riichi3p,
        }
    }
}
impl From<RuleLineWire> for RuleLine {
    fn from(value: RuleLineWire) -> Self {
        match value {
            RuleLineWire::Riichi4p => RuleLine::Riichi4p,
            RuleLineWire::Riichi3p => RuleLine::Riichi3p,
        }
    }
}

/// Source of events. Regular models always receive `observed` projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Observed,
    Authoritative,
}

/// Envelope source metadata, for consistency checks and diagnostics only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSource {
    pub kind: SourceKind,
    #[serde(default)]
    pub epoch: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

/// Opportunities a single pass gives up (canonical wire), convertible to [`ResponseOpportunities`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Declines {
    pub ron: bool,
    pub call: bool,
}

impl From<ResponseOpportunities> for Declines {
    fn from(value: ResponseOpportunities) -> Self {
        Declines {
            ron: value.ron,
            call: value.call,
        }
    }
}
impl From<Declines> for ResponseOpportunities {
    fn from(value: Declines) -> Self {
        ResponseOpportunities {
            ron: value.ron,
            call: value.call,
        }
    }
}

/// `flya-actions-v2` canonical action; the frozen v1 subset excludes `DealerOpening*`.
///
/// A plain discard [`CanonicalAction::Dahai`] and a riichi discard
/// [`CanonicalAction::RiichiDahai`] are separate variants. When a tile can be
/// discarded either way, `legal_actions` lists two distinct actions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CanonicalAction {
    /// Plain discard.
    Dahai {
        pai: Tile,
        tsumogiri: bool,
    },
    /// The dealer's first discard (Mahjong Soul); no drawn/hand origin.
    DealerOpeningDahai {
        pai: Tile,
    },
    /// Riichi discard.
    RiichiDahai {
        pai: Tile,
        tsumogiri: bool,
    },
    /// The dealer's first discard with riichi (Mahjong Soul); no drawn/hand origin.
    DealerOpeningRiichiDahai {
        pai: Tile,
    },
    /// Chi (4-player only). `consumed` are the two tiles from hand.
    Chi {
        pai: Tile,
        consumed: Vec<Tile>,
    },
    Pon {
        pai: Tile,
        consumed: Vec<Tile>,
    },
    Daiminkan {
        pai: Tile,
        consumed: Vec<Tile>,
    },
    Ankan {
        pai: Tile,
        consumed: Vec<Tile>,
    },
    Kakan {
        pai: Tile,
        consumed: Vec<Tile>,
    },
    /// Nukidora (3-player only). Canonical name `kita`; `nukidora` accepted on input.
    #[serde(rename = "kita", alias = "nukidora")]
    Kita,
    /// Tsumo (`pai` optional, defaults to the drawn tile).
    Tsumo {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pai: Option<Tile>,
    },
    Ron {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pai: Option<Tile>,
        target: u8,
    },
    Kyushukyuhai,
    /// Passes on every opportunity in this response window (MJAI `none` on the response side).
    PassAll {
        declines: Declines,
    },
}

impl<'de> Deserialize<'de> for CanonicalAction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        Self::from_canonical_value(&value).map_err(serde::de::Error::custom)
    }
}

impl CanonicalAction {
    /// Wire `type` string (same as the serde tag).
    pub const fn wire_type(&self) -> &'static str {
        match self {
            CanonicalAction::Dahai { .. } => "dahai",
            CanonicalAction::DealerOpeningDahai { .. } => "dealer_opening_dahai",
            CanonicalAction::RiichiDahai { .. } => "riichi_dahai",
            CanonicalAction::DealerOpeningRiichiDahai { .. } => "dealer_opening_riichi_dahai",
            CanonicalAction::Chi { .. } => "chi",
            CanonicalAction::Pon { .. } => "pon",
            CanonicalAction::Daiminkan { .. } => "daiminkan",
            CanonicalAction::Ankan { .. } => "ankan",
            CanonicalAction::Kakan { .. } => "kakan",
            CanonicalAction::Kita => "kita",
            CanonicalAction::Tsumo { .. } => "tsumo",
            CanonicalAction::Ron { .. } => "ron",
            CanonicalAction::Kyushukyuhai => "kyushukyuhai",
            CanonicalAction::PassAll { .. } => "pass_all",
        }
    }

    /// Canonical form of a 4-player legal action (riichi discards split out).
    pub fn from_legal_4p(action: &LegalAction4p) -> CanonicalAction {
        match action {
            LegalAction4p::Discard {
                pai,
                tsumogiri,
                riichi,
            } => {
                if *riichi {
                    CanonicalAction::RiichiDahai {
                        pai: *pai,
                        tsumogiri: *tsumogiri,
                    }
                } else {
                    CanonicalAction::Dahai {
                        pai: *pai,
                        tsumogiri: *tsumogiri,
                    }
                }
            }
            LegalAction4p::DealerOpeningDiscard { pai, riichi } => {
                if *riichi {
                    CanonicalAction::DealerOpeningRiichiDahai { pai: *pai }
                } else {
                    CanonicalAction::DealerOpeningDahai { pai: *pai }
                }
            }
            LegalAction4p::Kan {
                pai,
                kind,
                consumed,
            } => kan_to_canonical(*pai, *kind, consumed.clone()),
            LegalAction4p::Tsumo { pai } => CanonicalAction::Tsumo { pai: Some(*pai) },
            LegalAction4p::DealerOpeningTsumo => CanonicalAction::Tsumo { pai: None },
            LegalAction4p::Kyushukyuhai => CanonicalAction::Kyushukyuhai,
            LegalAction4p::PassAll { declines } => CanonicalAction::PassAll {
                declines: (*declines).into(),
            },
            LegalAction4p::Pon { pai, consumed } => CanonicalAction::Pon {
                pai: *pai,
                consumed: consumed.to_vec(),
            },
            LegalAction4p::Chi { pai, consumed } => CanonicalAction::Chi {
                pai: *pai,
                consumed: consumed.to_vec(),
            },
            LegalAction4p::Ron { pai, target } => CanonicalAction::Ron {
                pai: Some(*pai),
                target: *target,
            },
        }
    }

    /// Canonical form of a 3-player legal action (riichi discards split out, nukidora as `kita`).
    pub fn from_legal_3p(action: &LegalAction3p) -> CanonicalAction {
        match action {
            LegalAction3p::Discard {
                pai,
                tsumogiri,
                riichi,
            } => {
                if *riichi {
                    CanonicalAction::RiichiDahai {
                        pai: *pai,
                        tsumogiri: *tsumogiri,
                    }
                } else {
                    CanonicalAction::Dahai {
                        pai: *pai,
                        tsumogiri: *tsumogiri,
                    }
                }
            }
            LegalAction3p::DealerOpeningDiscard { pai, riichi } => {
                if *riichi {
                    CanonicalAction::DealerOpeningRiichiDahai { pai: *pai }
                } else {
                    CanonicalAction::DealerOpeningDahai { pai: *pai }
                }
            }
            LegalAction3p::Kan {
                pai,
                kind,
                consumed,
            } => kan_to_canonical(*pai, *kind, consumed.clone()),
            LegalAction3p::Nukidora => CanonicalAction::Kita,
            LegalAction3p::Tsumo { pai } => CanonicalAction::Tsumo { pai: Some(*pai) },
            LegalAction3p::DealerOpeningTsumo => CanonicalAction::Tsumo { pai: None },
            LegalAction3p::Kyushukyuhai => CanonicalAction::Kyushukyuhai,
            LegalAction3p::PassAll { declines } => CanonicalAction::PassAll {
                declines: (*declines).into(),
            },
            LegalAction3p::Pon { pai, consumed } => CanonicalAction::Pon {
                pai: *pai,
                consumed: consumed.to_vec(),
            },
            LegalAction3p::Ron { pai, target } => CanonicalAction::Ron {
                pai: Some(*pai),
                target: *target,
            },
        }
    }

    /// Canonical to 4-player legal action (`riichi_dahai` becomes `Discard { riichi: true }`).
    pub fn to_legal_4p(&self) -> Result<LegalAction4p, CanonicalConvertError> {
        use crate::KanKind;
        match self {
            CanonicalAction::Dahai { pai, tsumogiri } => Ok(LegalAction4p::Discard {
                pai: *pai,
                tsumogiri: *tsumogiri,
                riichi: false,
            }),
            CanonicalAction::DealerOpeningDahai { pai } => {
                Ok(LegalAction4p::DealerOpeningDiscard {
                    pai: *pai,
                    riichi: false,
                })
            }
            CanonicalAction::RiichiDahai { pai, tsumogiri } => Ok(LegalAction4p::Discard {
                pai: *pai,
                tsumogiri: *tsumogiri,
                riichi: true,
            }),
            CanonicalAction::DealerOpeningRiichiDahai { pai } => {
                Ok(LegalAction4p::DealerOpeningDiscard {
                    pai: *pai,
                    riichi: true,
                })
            }
            CanonicalAction::Chi { pai, consumed } => Ok(LegalAction4p::Chi {
                pai: *pai,
                consumed: consumed_array_2(consumed)?,
            }),
            CanonicalAction::Pon { pai, consumed } => Ok(LegalAction4p::Pon {
                pai: *pai,
                consumed: consumed_array_2(consumed)?,
            }),
            CanonicalAction::Daiminkan { pai, consumed } => Ok(LegalAction4p::Kan {
                pai: *pai,
                kind: KanKind::Daiminkan,
                consumed: check_kan_consumed(consumed)?,
            }),
            CanonicalAction::Ankan { pai, consumed } => Ok(LegalAction4p::Kan {
                pai: *pai,
                kind: KanKind::Ankan,
                consumed: check_kan_consumed(consumed)?,
            }),
            CanonicalAction::Kakan { pai, consumed } => Ok(LegalAction4p::Kan {
                pai: *pai,
                kind: KanKind::Kakan,
                consumed: check_kan_consumed(consumed)?,
            }),
            CanonicalAction::Kita => err("kita is a 3p-only action"),
            CanonicalAction::Tsumo { pai: Some(pai) } => Ok(LegalAction4p::Tsumo { pai: *pai }),
            CanonicalAction::Tsumo { pai: None } => Ok(LegalAction4p::DealerOpeningTsumo),
            CanonicalAction::Ron { pai, target } => Ok(LegalAction4p::Ron {
                pai: pai.ok_or_else(|| CanonicalConvertError("ron requires pai".into()))?,
                target: *target,
            }),
            CanonicalAction::Kyushukyuhai => Ok(LegalAction4p::Kyushukyuhai),
            CanonicalAction::PassAll { declines } => Ok(LegalAction4p::PassAll {
                declines: (*declines).into(),
            }),
        }
    }

    /// Canonical to 3-player legal action (`kita` / `nukidora` become `Nukidora`; `chi` is rejected).
    pub fn to_legal_3p(&self) -> Result<LegalAction3p, CanonicalConvertError> {
        use crate::KanKind;
        match self {
            CanonicalAction::Dahai { pai, tsumogiri } => Ok(LegalAction3p::Discard {
                pai: *pai,
                tsumogiri: *tsumogiri,
                riichi: false,
            }),
            CanonicalAction::DealerOpeningDahai { pai } => {
                Ok(LegalAction3p::DealerOpeningDiscard {
                    pai: *pai,
                    riichi: false,
                })
            }
            CanonicalAction::RiichiDahai { pai, tsumogiri } => Ok(LegalAction3p::Discard {
                pai: *pai,
                tsumogiri: *tsumogiri,
                riichi: true,
            }),
            CanonicalAction::DealerOpeningRiichiDahai { pai } => {
                Ok(LegalAction3p::DealerOpeningDiscard {
                    pai: *pai,
                    riichi: true,
                })
            }
            CanonicalAction::Chi { .. } => err("chi is not a legal 3p action"),
            CanonicalAction::Pon { pai, consumed } => Ok(LegalAction3p::Pon {
                pai: *pai,
                consumed: consumed_array_2(consumed)?,
            }),
            CanonicalAction::Daiminkan { pai, consumed } => Ok(LegalAction3p::Kan {
                pai: *pai,
                kind: KanKind::Daiminkan,
                consumed: check_kan_consumed(consumed)?,
            }),
            CanonicalAction::Ankan { pai, consumed } => Ok(LegalAction3p::Kan {
                pai: *pai,
                kind: KanKind::Ankan,
                consumed: check_kan_consumed(consumed)?,
            }),
            CanonicalAction::Kakan { pai, consumed } => Ok(LegalAction3p::Kan {
                pai: *pai,
                kind: KanKind::Kakan,
                consumed: check_kan_consumed(consumed)?,
            }),
            CanonicalAction::Kita => Ok(LegalAction3p::Nukidora),
            CanonicalAction::Tsumo { pai: Some(pai) } => Ok(LegalAction3p::Tsumo { pai: *pai }),
            CanonicalAction::Tsumo { pai: None } => Ok(LegalAction3p::DealerOpeningTsumo),
            CanonicalAction::Ron { pai, target } => Ok(LegalAction3p::Ron {
                pai: pai.ok_or_else(|| CanonicalConvertError("ron requires pai".into()))?,
                target: *target,
            }),
            CanonicalAction::Kyushukyuhai => Ok(LegalAction3p::Kyushukyuhai),
            CanonicalAction::PassAll { declines } => Ok(LegalAction3p::PassAll {
                declines: (*declines).into(),
            }),
        }
    }

    /// Compatibility entry point: parses the legacy `flya-inference-v1` wire
    /// (`{type:"dahai", riichi:bool}`, `{type:"nukidora"}`, ...) into canonical form,
    /// splitting `dahai` into [`CanonicalAction::Dahai`] / [`CanonicalAction::RiichiDahai`]
    /// by `riichi`. Extra legacy fields such as `action_id` and `actor` are ignored.
    pub fn from_legacy_value(
        value: &serde_json::Value,
    ) -> Result<CanonicalAction, CanonicalConvertError> {
        let obj = value
            .as_object()
            .ok_or_else(|| CanonicalConvertError("legacy action must be a JSON object".into()))?;
        let kind = obj
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CanonicalConvertError("legacy action missing string `type`".into()))?;

        let pai = || -> Result<Tile, CanonicalConvertError> {
            obj.get("pai")
                .and_then(|v| v.as_str())
                .ok_or_else(|| CanonicalConvertError(format!("{kind} requires string `pai`")))
                .and_then(|s| {
                    s.parse::<Tile>()
                        .map_err(|_| CanonicalConvertError(format!("invalid tile {s:?}")))
                })
        };
        let opt_pai = || -> Result<Option<Tile>, CanonicalConvertError> {
            match obj.get("pai") {
                Some(v) => v
                    .as_str()
                    .ok_or_else(|| CanonicalConvertError("`pai` must be a string".into()))
                    .and_then(|s| {
                        s.parse::<Tile>()
                            .map(Some)
                            .map_err(|_| CanonicalConvertError(format!("invalid tile {s:?}")))
                    }),
                None => Ok(None),
            }
        };
        let consumed = || -> Result<Vec<Tile>, CanonicalConvertError> {
            let arr = obj
                .get("consumed")
                .and_then(|v| v.as_array())
                .ok_or_else(|| CanonicalConvertError(format!("{kind} requires `consumed`")))?;
            arr.iter()
                .map(|v| {
                    v.as_str()
                        .ok_or_else(|| CanonicalConvertError("consumed entry not a string".into()))
                        .and_then(|s| {
                            s.parse::<Tile>()
                                .map_err(|_| CanonicalConvertError(format!("invalid tile {s:?}")))
                        })
                })
                .collect()
        };
        let tsumogiri = || -> Result<bool, CanonicalConvertError> {
            obj.get("tsumogiri")
                .and_then(|v| v.as_bool())
                .ok_or_else(|| CanonicalConvertError("dahai requires bool `tsumogiri`".into()))
        };

        match kind {
            "dahai" => {
                let riichi = match obj.get("riichi") {
                    None => false,
                    Some(v) => v.as_bool().ok_or_else(|| {
                        CanonicalConvertError("legacy `riichi` must be a bool".into())
                    })?,
                };
                let (pai, tsumogiri) = (pai()?, tsumogiri()?);
                Ok(if riichi {
                    CanonicalAction::RiichiDahai { pai, tsumogiri }
                } else {
                    CanonicalAction::Dahai { pai, tsumogiri }
                })
            }
            "dealer_opening_dahai" => Ok(CanonicalAction::DealerOpeningDahai { pai: pai()? }),
            "dealer_opening_riichi_dahai" => {
                Ok(CanonicalAction::DealerOpeningRiichiDahai { pai: pai()? })
            }
            // The canonical `riichi_dahai` is also accepted here (idempotent).
            "riichi_dahai" => Ok(CanonicalAction::RiichiDahai {
                pai: pai()?,
                tsumogiri: tsumogiri()?,
            }),
            "chi" => Ok(CanonicalAction::Chi {
                pai: pai()?,
                consumed: consumed()?,
            }),
            "pon" => Ok(CanonicalAction::Pon {
                pai: pai()?,
                consumed: consumed()?,
            }),
            "daiminkan" => Ok(CanonicalAction::Daiminkan {
                pai: pai()?,
                consumed: consumed()?,
            }),
            "ankan" => Ok(CanonicalAction::Ankan {
                pai: pai()?,
                consumed: consumed()?,
            }),
            "kakan" => Ok(CanonicalAction::Kakan {
                pai: pai()?,
                consumed: consumed()?,
            }),
            "kita" | "nukidora" => Ok(CanonicalAction::Kita),
            "tsumo" => Ok(CanonicalAction::Tsumo { pai: opt_pai()? }),
            "ron" => Ok(CanonicalAction::Ron {
                pai: opt_pai()?,
                target: u8::try_from(obj.get("target").and_then(|v| v.as_u64()).ok_or_else(
                    || CanonicalConvertError("ron requires integer `target`".into()),
                )?)
                .map_err(|_| CanonicalConvertError("ron target exceeds u8".into()))?,
            }),
            "kyushukyuhai" => Ok(CanonicalAction::Kyushukyuhai),
            "pass_all" => {
                let declines = obj
                    .get("declines")
                    .ok_or_else(|| CanonicalConvertError("pass_all requires `declines`".into()))?
                    .as_object()
                    .ok_or_else(|| CanonicalConvertError("`declines` must be an object".into()))?;
                for key in declines.keys() {
                    if key != "ron" && key != "call" {
                        return err(format!("declines has disallowed field {key:?}"));
                    }
                }
                let flag = |name: &str| -> Result<bool, CanonicalConvertError> {
                    match declines.get(name) {
                        None => Ok(false),
                        Some(v) => v.as_bool().ok_or_else(|| {
                            CanonicalConvertError(format!("declines.{name} must be bool"))
                        }),
                    }
                };
                let ron = flag("ron")?;
                let call = flag("call")?;
                Ok(CanonicalAction::PassAll {
                    declines: Declines { ron, call },
                })
            }
            other => err(format!("unknown legacy action type {other:?}")),
        }
    }

    /// Strict canonical parsing (fail-closed).
    ///
    /// Rejects keys outside each `type`'s allowed set, in particular the legacy-only
    /// `riichi` key on `dahai`, so the same JSON cannot parse to opposite meanings on
    /// the two paths: `{"type":"dahai","riichi":true}` fails here and only
    /// [`CanonicalAction::from_legacy_value`] interprets `riichi`. Type errors (non-bool
    /// `tsumogiri`, invalid `pai`, `target` out of `u8` range, malformed `declines`) are
    /// rejected rather than defaulted or truncated.
    pub fn from_canonical_value(
        value: &serde_json::Value,
    ) -> Result<CanonicalAction, CanonicalConvertError> {
        let obj = value.as_object().ok_or_else(|| {
            CanonicalConvertError("canonical action must be a JSON object".into())
        })?;
        let kind = obj.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
            CanonicalConvertError("canonical action missing string `type`".into())
        })?;

        // Allowed keys per type, including `type` but not `action_id` (an outer field of the legal action).
        let allowed: &[&str] = match kind {
            "dahai" | "riichi_dahai" => &["type", "pai", "tsumogiri"],
            "dealer_opening_dahai" | "dealer_opening_riichi_dahai" => &["type", "pai"],
            "chi" | "pon" | "daiminkan" | "ankan" | "kakan" => &["type", "pai", "consumed"],
            "kita" => &["type"],
            "tsumo" => &["type", "pai"],
            "ron" => &["type", "pai", "target"],
            "kyushukyuhai" => &["type"],
            "pass_all" => &["type", "declines"],
            other => return err(format!("unknown canonical action type {other:?}")),
        };
        for key in obj.keys() {
            if !allowed.contains(&key.as_str()) {
                return err(format!(
                    "canonical action {kind:?} has disallowed field {key:?} \
                     (legacy-only fields like `riichi` must use the explicit legacy entry)"
                ));
            }
        }

        let req_pai = || -> Result<Tile, CanonicalConvertError> {
            obj.get("pai")
                .and_then(|v| v.as_str())
                .ok_or_else(|| CanonicalConvertError(format!("{kind} requires string `pai`")))
                .and_then(|s| {
                    s.parse::<Tile>()
                        .map_err(|_| CanonicalConvertError(format!("invalid tile {s:?}")))
                })
        };
        let opt_pai = || -> Result<Option<Tile>, CanonicalConvertError> {
            match obj.get("pai") {
                None => Ok(None),
                Some(v) => v
                    .as_str()
                    .ok_or_else(|| CanonicalConvertError("`pai` must be a string".into()))
                    .and_then(|s| {
                        s.parse::<Tile>()
                            .map(Some)
                            .map_err(|_| CanonicalConvertError(format!("invalid tile {s:?}")))
                    }),
            }
        };
        let consumed = || -> Result<Vec<Tile>, CanonicalConvertError> {
            let arr = obj
                .get("consumed")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    CanonicalConvertError(format!("{kind} requires array `consumed`"))
                })?;
            arr.iter()
                .map(|v| {
                    v.as_str()
                        .ok_or_else(|| CanonicalConvertError("consumed entry not a string".into()))
                        .and_then(|s| {
                            s.parse::<Tile>()
                                .map_err(|_| CanonicalConvertError(format!("invalid tile {s:?}")))
                        })
                })
                .collect()
        };
        let tsumogiri = || -> Result<bool, CanonicalConvertError> {
            obj.get("tsumogiri")
                .ok_or_else(|| CanonicalConvertError(format!("{kind} requires bool `tsumogiri`")))?
                .as_bool()
                .ok_or_else(|| CanonicalConvertError("`tsumogiri` must be a bool".into()))
        };

        match kind {
            "dahai" => Ok(CanonicalAction::Dahai {
                pai: req_pai()?,
                tsumogiri: tsumogiri()?,
            }),
            "dealer_opening_dahai" => Ok(CanonicalAction::DealerOpeningDahai { pai: req_pai()? }),
            "riichi_dahai" => Ok(CanonicalAction::RiichiDahai {
                pai: req_pai()?,
                tsumogiri: tsumogiri()?,
            }),
            "dealer_opening_riichi_dahai" => {
                Ok(CanonicalAction::DealerOpeningRiichiDahai { pai: req_pai()? })
            }
            "chi" => Ok(CanonicalAction::Chi {
                pai: req_pai()?,
                consumed: consumed()?,
            }),
            "pon" => Ok(CanonicalAction::Pon {
                pai: req_pai()?,
                consumed: consumed()?,
            }),
            "daiminkan" => Ok(CanonicalAction::Daiminkan {
                pai: req_pai()?,
                consumed: consumed()?,
            }),
            "ankan" => Ok(CanonicalAction::Ankan {
                pai: req_pai()?,
                consumed: consumed()?,
            }),
            "kakan" => Ok(CanonicalAction::Kakan {
                pai: req_pai()?,
                consumed: consumed()?,
            }),
            "kita" => Ok(CanonicalAction::Kita),
            "tsumo" => Ok(CanonicalAction::Tsumo { pai: opt_pai()? }),
            "ron" => {
                let target_raw = obj
                    .get("target")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| CanonicalConvertError("ron requires integer `target`".into()))?;
                let target = u8::try_from(target_raw).map_err(|_| {
                    CanonicalConvertError(format!("ron target {target_raw} exceeds u8"))
                })?;
                Ok(CanonicalAction::Ron {
                    pai: opt_pai()?,
                    target,
                })
            }
            "kyushukyuhai" => Ok(CanonicalAction::Kyushukyuhai),
            "pass_all" => {
                let declines = match obj.get("declines") {
                    None => Declines::default(),
                    Some(d) => {
                        let d = d.as_object().ok_or_else(|| {
                            CanonicalConvertError("`declines` must be an object".into())
                        })?;
                        for k in d.keys() {
                            if k != "ron" && k != "call" {
                                return err(format!("declines has disallowed field {k:?}"));
                            }
                        }
                        let flag = |name: &str| -> Result<bool, CanonicalConvertError> {
                            match d.get(name) {
                                None => Ok(false),
                                Some(v) => v.as_bool().ok_or_else(|| {
                                    CanonicalConvertError(format!("declines.{name} must be bool"))
                                }),
                            }
                        };
                        Declines {
                            ron: flag("ron")?,
                            call: flag("call")?,
                        }
                    }
                };
                Ok(CanonicalAction::PassAll { declines })
            }
            _ => unreachable!("kind already validated against allowed set"),
        }
    }
}

fn kan_to_canonical(pai: Tile, kind: crate::KanKind, consumed: Vec<Tile>) -> CanonicalAction {
    use crate::KanKind;
    match kind {
        KanKind::Ankan => CanonicalAction::Ankan { pai, consumed },
        KanKind::Kakan => CanonicalAction::Kakan { pai, consumed },
        KanKind::Daiminkan => CanonicalAction::Daiminkan { pai, consumed },
    }
}

fn consumed_array_2(consumed: &[Tile]) -> Result<[Tile; 2], CanonicalConvertError> {
    if consumed.len() == 2 {
        Ok([consumed[0], consumed[1]])
    } else {
        err(format!("expected 2 consumed tiles, got {}", consumed.len()))
    }
}

fn check_kan_consumed(consumed: &[Tile]) -> Result<Vec<Tile>, CanonicalConvertError> {
    if (3..=4).contains(&consumed.len()) {
        Ok(consumed.to_vec())
    } else {
        err(format!(
            "expected 3 or 4 consumed tiles for kan, got {}",
            consumed.len()
        ))
    }
}

/// A canonical legal action with a stable `action_id` (its index in `legal_actions`).
///
/// Wire shape `{"action_id":N,"type":"dahai",...}`, with `action_id` flattened
/// alongside the action. A successful model result must pick one of these.
///
/// Deserialization goes through [`CanonicalAction::from_canonical_value`]:
/// `action_id` must be an integer, the other fields are checked against the
/// `type`'s allowed set, and a legacy `riichi` key on `dahai` is rejected.
/// Serialization still flattens (same shape as the old wire).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CanonicalLegalAction {
    pub action_id: usize,
    #[serde(flatten)]
    pub action: CanonicalAction,
}

impl<'de> Deserialize<'de> for CanonicalLegalAction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let obj = value
            .as_object()
            .ok_or_else(|| serde::de::Error::custom("legal action must be a JSON object"))?;
        let action_id = obj
            .get("action_id")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| serde::de::Error::custom("legal action requires integer `action_id`"))?;
        let action_id = usize::try_from(action_id)
            .map_err(|_| serde::de::Error::custom("action_id out of range"))?;
        let mut action_obj = obj.clone();
        action_obj.remove("action_id");
        let action = CanonicalAction::from_canonical_value(&serde_json::Value::Object(action_obj))
            .map_err(serde::de::Error::custom)?;
        Ok(CanonicalLegalAction { action_id, action })
    }
}

impl CanonicalLegalAction {
    pub fn new(action_id: usize, action: CanonicalAction) -> Self {
        CanonicalLegalAction { action_id, action }
    }

    /// Converts a 4-player legal action table into an indexed canonical table.
    pub fn table_from_legal_4p(actions: &[LegalAction4p]) -> Vec<CanonicalLegalAction> {
        actions
            .iter()
            .enumerate()
            .map(|(i, a)| CanonicalLegalAction::new(i, CanonicalAction::from_legal_4p(a)))
            .collect()
    }

    /// Converts a 3-player legal action table into an indexed canonical table.
    pub fn table_from_legal_3p(actions: &[LegalAction3p]) -> Vec<CanonicalLegalAction> {
        actions
            .iter()
            .enumerate()
            .map(|(i, a)| CanonicalLegalAction::new(i, CanonicalAction::from_legal_3p(a)))
            .collect()
    }
}

/// Schema version tag of `flya-mahjong-events`.
///
/// Serializes as v2. Deserialization accepts frozen v1 and v2 (with any minor) and
/// rejects other majors. The envelope itself does not use `deny_unknown_fields`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MahjongEventsSchema(String);

impl Default for MahjongEventsSchema {
    fn default() -> Self {
        Self(FLYA_MAHJONG_EVENTS_V2.to_string())
    }
}

impl MahjongEventsSchema {
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Serialize for MahjongEventsSchema {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.0)
    }
}
impl<'de> Deserialize<'de> for MahjongEventsSchema {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        if schema_major_matches(&s, "flya-mahjong-events", 1)
            || schema_major_matches(&s, "flya-mahjong-events", 2)
        {
            Ok(MahjongEventsSchema(s))
        } else {
            Err(serde::de::Error::custom(format!(
                "unsupported schema {s:?}: this build accepts flya-mahjong-events major v1 or v2"
            )))
        }
    }
}

/// Strictly checks `<base>-v<major>` or `<base>-v<major>.<minor>...` with a matching major.
///
/// Every segment must be a non-empty run of digits; `-v1.`, `-v1.foo`, `-v1abc` and
/// `-vfoo` are rejected. Only forms like `1`, `1.2` or `1.2.0` pass.
pub fn schema_major_matches(schema: &str, base: &str, major: u32) -> bool {
    let Some(rest) = schema.strip_prefix(base) else {
        return false;
    };
    let Some(ver) = rest.strip_prefix("-v") else {
        return false;
    };
    let mut parts = ver.split('.');
    let Some(major_str) = parts.next() else {
        return false;
    };
    if major_str.is_empty() || !major_str.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    // Every minor/patch segment must be non-empty digits.
    for part in parts {
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
    }
    major_str
        .parse::<u32>()
        .map(|m| m == major)
        .unwrap_or(false)
}

/// `flya-mahjong-events-v2` per-seat visible event envelope, generic over
/// [`Event4p`] / [`Event3p`].
///
/// Only adds version, rule line, visibility and seq metadata. Invariants: 4-player
/// uses 4 seats and 3-player 3 (decided by `E`); authoritative seeds, walls and
/// hidden tiles never cross the projection boundary (regular models always receive
/// observed events).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MahjongEvents<E> {
    pub schema: MahjongEventsSchema,
    pub rule_line: RuleLineWire,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule_profile: Option<String>,
    pub viewer_seat: u8,
    pub source: EventSource,
    pub from_seq: u64,
    pub to_seq: u64,
    pub events: Vec<E>,
    pub legal_actions: Vec<CanonicalLegalAction>,
    /// Extension namespace. Unknown majors are rejected; optional fields are tolerated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ext: Option<serde_json::Value>,
}

/// 4-player event envelope.
pub type MahjongEvents4p = MahjongEvents<Event4p>;
/// 3-player event envelope.
pub type MahjongEvents3p = MahjongEvents<Event3p>;
