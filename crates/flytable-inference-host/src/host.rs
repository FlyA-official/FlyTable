use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command as ProcessCommand, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::OnceLock;
use std::thread;
use std::time::Duration;

use flytable_core::tile::Tile;
use flytable_event::{Actor3, Actor4, Event3p, Event4p};
use flytable_seat::contract::{
    self, ActionAnnotation, Annotation, AnnotationDisplay, AnnotationValue, EngineCaps,
    InferenceDecision, InferenceError, InferenceErrorKind, KanKind, LegalAction3p, LegalAction4p,
    NumberFormat, PrimaryMetric, Recommendation, ResponseOpportunities, RiichiStyle, RuleLine,
    VisibleBody3p, VisibleBody4p, VisibleEvent3p, VisibleEvent4p, FLYA_INFERENCE_PROTOCOL_V1,
    FLYA_INFERENCE_PROTOCOL_V2,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub const DEFAULT_INFER_TIMEOUT: Duration = Duration::from_secs(5);
pub const MAX_META_BYTES: usize = 64 * 1024;

/// Order in which optional fields are trimmed when `meta` is over the limit (largest
/// and least used first). `recommendation` is never trimmed.
const META_TRIM_ORDER: [&str; 5] = [
    "state",
    "debug",
    "model_output",
    "table_context",
    "behavior",
];

/// Structural limits for recommendation annotations, so a broken or hostile
/// translator cannot flood the pass-through. Entries over the limit are dropped one
/// by one; play is unaffected and no partial data reaches the frontend.
pub const MAX_RECOMMENDATION_ACTIONS: usize = 128;
pub const MAX_ANNOTATIONS_PER_SCOPE: usize = 32;
pub const MAX_ANNOTATION_TEXT_BYTES: usize = 4 * 1024;

#[derive(Debug, Clone)]
pub struct EngineProcessConfig {
    pub executable: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: Option<PathBuf>,
    pub timeout: Duration,
    pub session_id: String,
    pub seat: u8,
    pub rule_line: RuleLine,
    pub match_context: Value,
    /// host-translator protocols allowed in negotiation, in priority order.
    pub protocol_versions: Vec<Box<str>>,
}

impl EngineProcessConfig {
    pub fn new(
        executable: impl Into<PathBuf>,
        seat: u8,
        rule_line: RuleLine,
        session_id: impl Into<String>,
    ) -> Self {
        Self {
            executable: executable.into(),
            args: Vec::new(),
            env: Vec::new(),
            cwd: None,
            timeout: DEFAULT_INFER_TIMEOUT,
            session_id: session_id.into(),
            seat,
            rule_line,
            match_context: Value::Null,
            protocol_versions: vec![
                FLYA_INFERENCE_PROTOCOL_V2.into(),
                FLYA_INFERENCE_PROTOCOL_V1.into(),
            ],
        }
    }
}

/// Remote runtime HTTP inference config.
///
/// The network counterpart of [`EngineProcessConfig`]. It lives in
/// `flytable-inference-host` rather than the rule or runtime crates, so products can
/// use remote models without networking leaking into the rule crates.
///
/// Does not derive `Debug`: `bearer_token` is a short-lived credential and must never
/// reach a `{:?}` sink (anyhow context, traces, logs). The manual `Debug` mirrors
/// `RemoteHttpHost` and only reports `has_bearer_token: bool`.
#[derive(Clone)]
pub struct RemoteHttpConfig {
    /// Full endpoint URL, e.g. `https://api.example.com/api/inference/model4p/infer`.
    pub endpoint: String,
    /// Short-lived account/session token supplied by the product layer.
    pub bearer_token: Option<String>,
    pub timeout: Duration,
    pub session_id: String,
    pub seat: u8,
    pub rule_line: RuleLine,
    pub match_context: Value,
}

impl std::fmt::Debug for RemoteHttpConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteHttpConfig")
            .field("endpoint", &self.endpoint)
            .field("has_bearer_token", &self.bearer_token.is_some())
            .field("timeout", &self.timeout)
            .field("session_id", &self.session_id)
            .field("seat", &self.seat)
            .field("rule_line", &self.rule_line)
            .field("match_context", &self.match_context)
            .finish()
    }
}

impl RemoteHttpConfig {
    pub fn new(
        endpoint: impl Into<String>,
        seat: u8,
        rule_line: RuleLine,
        session_id: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            bearer_token: None,
            timeout: DEFAULT_INFER_TIMEOUT,
            session_id: session_id.into(),
            seat,
            rule_line,
            match_context: Value::Null,
        }
    }
}

/// HTTP host for remote models.
pub struct RemoteHttpHost {
    config: RemoteHttpConfig,
}

impl std::fmt::Debug for RemoteHttpHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteHttpHost")
            .field("endpoint", &self.config.endpoint)
            .field("seat", &self.config.seat)
            .field("rule_line", &self.config.rule_line)
            .field("timeout", &self.config.timeout)
            .field("has_bearer_token", &self.config.bearer_token.is_some())
            .finish_non_exhaustive()
    }
}

impl RemoteHttpHost {
    pub fn new(config: RemoteHttpConfig) -> std::result::Result<Self, InferenceError> {
        if config.endpoint.trim().is_empty() {
            return Err(inference_error(
                InferenceErrorKind::Protocol,
                "remote endpoint is empty",
            ));
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> &RemoteHttpConfig {
        &self.config
    }

    pub fn infer_4p(
        &self,
        input: InferenceInput4p<'_>,
    ) -> std::result::Result<InferenceDecision, InferenceError> {
        if self.config.rule_line != RuleLine::Riichi4p {
            return Err(inference_error(
                InferenceErrorKind::Protocol,
                "4p infer called on non-riichi4p remote host",
            ));
        }
        let body = build_remote_infer_request_4p(&self.config, &input)?;
        let legal_actions = input
            .legal_actions
            .iter()
            .enumerate()
            .map(|(index, action)| legal_action_4p_to_wire(index, action))
            .collect::<Vec<_>>();
        self.request_remote_infer(body, &input.decision_id, &legal_actions)
    }

    pub fn infer_3p(
        &self,
        input: InferenceInput3p<'_>,
    ) -> std::result::Result<InferenceDecision, InferenceError> {
        if self.config.rule_line != RuleLine::Riichi3p {
            return Err(inference_error(
                InferenceErrorKind::Protocol,
                "3p infer called on non-riichi3p remote host",
            ));
        }
        let body = build_remote_infer_request_3p(&self.config, &input)?;
        let legal_actions = input
            .legal_actions
            .iter()
            .enumerate()
            .map(|(index, action)| legal_action_3p_to_wire(index, action))
            .collect::<Vec<_>>();
        self.request_remote_infer(body, &input.decision_id, &legal_actions)
    }

    fn request_remote_infer(
        &self,
        body: Value,
        decision_id: &str,
        legal_actions: &[Value],
    ) -> std::result::Result<InferenceDecision, InferenceError> {
        let config = self.config.clone();
        let (status, value) = thread::spawn(move || request_remote_infer_blocking(config, body))
            .join()
            .map_err(|_| {
                inference_error(
                    InferenceErrorKind::Internal,
                    "remote inference worker thread panicked",
                )
            })??;
        if !(200..300).contains(&status) {
            let detail = value
                .get("message")
                .or_else(|| value.get("error"))
                .and_then(Value::as_str)
                .unwrap_or("remote inference failed");
            let kind = match status {
                401..=403 => InferenceErrorKind::EngineUnavailable,
                408 | 504 => InferenceErrorKind::Timeout,
                409 => InferenceErrorKind::StaleDecision,
                _ => InferenceErrorKind::Protocol,
            };
            return Err(inference_error(
                kind,
                format!("remote status {status}: {detail}"),
            ));
        }
        parse_remote_infer_response(&value, decision_id, legal_actions)
    }
}

/// Process-wide blocking client for remote inference.
///
/// Building a new blocking client per call (each with its own connection pool and
/// internal runtime) opens a new TCP connection per request under load, which can
/// exhaust ephemeral ports on Windows through TIME_WAIT (`os error 10048`). A single
/// shared client keeps connections alive; timeouts are set per request with
/// `.timeout(config.timeout)`, and redirects are disabled.
fn remote_http_client() -> &'static reqwest::blocking::Client {
    static CLIENT: OnceLock<reqwest::blocking::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .pool_idle_timeout(Duration::from_secs(90))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("build pooled remote inference http client")
    })
}

fn request_remote_infer_blocking(
    config: RemoteHttpConfig,
    body: Value,
) -> std::result::Result<(u16, Value), InferenceError> {
    let mut request = remote_http_client()
        .post(&config.endpoint)
        .timeout(config.timeout)
        .json(&body);
    if let Some(token) = config
        .bearer_token
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        request = request.bearer_auth(token);
    }
    let response = request.send().map_err(|err| {
        let kind = if err.is_timeout() {
            InferenceErrorKind::Timeout
        } else {
            InferenceErrorKind::EngineUnavailable
        };
        inference_error(kind, format!("remote inference request failed: {err}"))
    })?;
    let status = response.status().as_u16();
    let value = response.json::<Value>().map_err(|err| {
        inference_error(
            InferenceErrorKind::Protocol,
            format!("remote inference returned non-json response: {err}"),
        )
    })?;
    Ok((status, value))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionPhase {
    Discard,
    Response,
}

impl DecisionPhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            DecisionPhase::Discard => "discard",
            DecisionPhase::Response => "response",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Authoritative,
    Observed,
}

impl SourceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            SourceKind::Authoritative => "authoritative",
            SourceKind::Observed => "observed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceInfo {
    pub kind: SourceKind,
    pub epoch: u64,
    pub status: String,
}

impl SourceInfo {
    pub fn authoritative() -> Self {
        Self {
            kind: SourceKind::Authoritative,
            epoch: 0,
            status: "reconstructable".to_string(),
        }
    }

    fn digest_value(&self) -> Value {
        json!({
            "kind": self.kind.as_str(),
            "epoch": self.epoch,
        })
    }

    fn wire_value(&self) -> Value {
        json!({
            "kind": self.kind.as_str(),
            "epoch": self.epoch,
            "status": self.status,
        })
    }
}

#[derive(Debug, Clone)]
pub struct InferenceInput4p<'a> {
    pub decision_id: String,
    pub phase: DecisionPhase,
    pub events: &'a [Event4p],
    pub legal_actions: &'a [LegalAction4p],
    pub wall_remaining: u32,
    pub source: SourceInfo,
    /// Short-lived ticket for remote calls (upstream service -> host -> translator); `None` for certification smoke tests.
    pub remote_auth: Option<String>,
    /// Overrides the viewer seat from the spawn config (`None` keeps `config.seat`).
    pub seat: Option<u8>,
}

#[derive(Debug, Clone)]
pub struct InferenceInput3p<'a> {
    pub decision_id: String,
    pub phase: DecisionPhase,
    pub events: &'a [Event3p],
    pub legal_actions: &'a [LegalAction3p],
    pub wall_remaining: u32,
    pub source: SourceInfo,
    pub remote_auth: Option<String>,
    /// Overrides the viewer seat from the spawn config (`None` keeps `config.seat`).
    pub seat: Option<u8>,
    /// Overrides `match_context` from the spawn config (`None` keeps `config.match_context`).
    ///
    /// Workers are long-lived subprocesses reused across requests, so their
    /// `match_context` is fixed at spawn time. Some models take table goals (such as rank
    /// points) per request, so this has to be per request: `Some(..)` replaces the
    /// handshake value in the `state` of this infer request.
    pub match_context: Option<Value>,
}

/// Stage-aware subprocess response. `DeclareReach` is an internal intermediate signal and is never
/// exposed as a public action.
pub enum HostStageDecision {
    Decision(InferenceDecision),
    DeclareReach,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackPolicy {
    ProductBasicDecider,
    FailMatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FallbackReason {
    Abstain,
    Error(InferenceErrorKind),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostInferenceOutcome {
    Engine(InferenceDecision),
    Fallback {
        policy: FallbackPolicy,
        reason: FallbackReason,
        decision: InferenceDecision,
    },
}

pub struct SubprocessHost {
    config: EngineProcessConfig,
    child: Option<Child>,
    stdin: Option<BufWriter<ChildStdin>>,
    reader: Option<Receiver<io::Result<String>>>,
    caps: Option<EngineCaps>,
    negotiated_protocol: Option<Box<str>>,
    needs_rebuild: bool,
}

impl SubprocessHost {
    pub fn start(config: EngineProcessConfig) -> std::result::Result<Self, InferenceError> {
        let mut host = Self {
            config,
            child: None,
            stdin: None,
            reader: None,
            caps: None,
            negotiated_protocol: None,
            needs_rebuild: false,
        };
        host.restart()?;
        Ok(host)
    }

    pub fn caps(&self) -> Option<&EngineCaps> {
        self.caps.as_ref()
    }

    pub fn negotiated_protocol(&self) -> Option<&str> {
        self.negotiated_protocol.as_deref()
    }

    pub fn needs_rebuild(&self) -> bool {
        self.needs_rebuild
    }

    pub fn restart(&mut self) -> std::result::Result<(), InferenceError> {
        self.stop_child();
        let mut command = ProcessCommand::new(&self.config.executable);
        command
            .args(&self.config.args)
            // Keep Python translators from writing .pyc files (fewer transient `__pycache__`
            // files, so package_hash and model_id stay stable). Applied before `config.env` so
            // callers can still override it. No effect on non-Python engines.
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .envs(self.config.env.iter().map(|(key, value)| (key, value)))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(cwd) = &self.config.cwd {
            command.current_dir(cwd);
        }
        let mut child = command.spawn().map_err(|err| {
            inference_error(
                InferenceErrorKind::EngineUnavailable,
                format!(
                    "failed to spawn inference engine {}: {err}",
                    self.config.executable.display()
                ),
            )
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            inference_error(
                InferenceErrorKind::EngineUnavailable,
                "failed to open inference engine stdin",
            )
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            inference_error(
                InferenceErrorKind::EngineUnavailable,
                "failed to open inference engine stdout",
            )
        })?;
        self.stdin = Some(BufWriter::new(stdin));
        self.reader = Some(spawn_stdout_reader(stdout));
        self.child = Some(child);
        self.needs_rebuild = false;
        self.handshake()
    }

    pub fn infer_4p(
        &mut self,
        input: InferenceInput4p<'_>,
    ) -> std::result::Result<InferenceDecision, InferenceError> {
        let timeout = self.config.timeout;
        self.infer_4p_within(input, timeout)
    }

    /// Same as [`Self::infer_4p`], but this call's response read timeout is `call_timeout`
    /// (per-call `min(remaining, cap)` on a reused worker). Handshake and process start
    /// still use `config.timeout`.
    pub fn infer_4p_within(
        &mut self,
        input: InferenceInput4p<'_>,
        call_timeout: Duration,
    ) -> std::result::Result<InferenceDecision, InferenceError> {
        if self.config.rule_line != RuleLine::Riichi4p {
            return Err(inference_error(
                InferenceErrorKind::Protocol,
                "4p infer called on non-riichi4p host",
            ));
        }
        self.ensure_running()?;
        let built = build_infer_request_4p(
            &self.config,
            self.negotiated_protocol
                .as_deref()
                .expect("running host completed protocol negotiation"),
            &input,
        )?;
        self.request_infer(&built.request)?;
        let response = self.read_response_json(call_timeout)?;
        self.handle_response_4p(response, &built, input.legal_actions)
    }

    pub fn infer_stage_4p_within(
        &mut self,
        input: InferenceInput4p<'_>,
        call_timeout: Duration,
        second_stage: bool,
    ) -> std::result::Result<HostStageDecision, InferenceError> {
        if self.config.rule_line != RuleLine::Riichi4p {
            return Err(inference_error(
                InferenceErrorKind::Protocol,
                "4p stage infer called on non-riichi4p host",
            ));
        }
        self.ensure_running()?;
        let mut built = build_infer_request_4p(
            &self.config,
            self.negotiated_protocol
                .as_deref()
                .expect("running host completed protocol negotiation"),
            &input,
        )?;
        built.request["driver_stage"] = json!(if second_stage {
            "riichi_second_stage"
        } else {
            "primary"
        });
        self.request_infer(&built.request)?;
        let response = self.read_response_json(call_timeout)?;
        self.handle_common_response(&response, &built)?;
        if response_branch(&response)? == ResponseBranch::DeclareReach {
            return Ok(HostStageDecision::DeclareReach);
        }
        self.handle_response_4p(response, &built, input.legal_actions)
            .map(HostStageDecision::Decision)
    }

    pub fn infer_3p(
        &mut self,
        input: InferenceInput3p<'_>,
    ) -> std::result::Result<InferenceDecision, InferenceError> {
        let timeout = self.config.timeout;
        self.infer_3p_within(input, timeout)
    }

    /// 3-player version of [`Self::infer_4p_within`].
    pub fn infer_3p_within(
        &mut self,
        input: InferenceInput3p<'_>,
        call_timeout: Duration,
    ) -> std::result::Result<InferenceDecision, InferenceError> {
        if self.config.rule_line != RuleLine::Riichi3p {
            return Err(inference_error(
                InferenceErrorKind::Protocol,
                "3p infer called on non-riichi3p host",
            ));
        }
        self.ensure_running()?;
        let built = build_infer_request_3p(
            &self.config,
            self.negotiated_protocol
                .as_deref()
                .expect("running host completed protocol negotiation"),
            &input,
        )?;
        self.request_infer(&built.request)?;
        let response = self.read_response_json(call_timeout)?;
        self.handle_response_3p(response, &built, input.legal_actions)
    }

    pub fn infer_stage_3p_within(
        &mut self,
        input: InferenceInput3p<'_>,
        call_timeout: Duration,
        second_stage: bool,
    ) -> std::result::Result<HostStageDecision, InferenceError> {
        if self.config.rule_line != RuleLine::Riichi3p {
            return Err(inference_error(
                InferenceErrorKind::Protocol,
                "3p stage infer called on non-riichi3p host",
            ));
        }
        self.ensure_running()?;
        let mut built = build_infer_request_3p(
            &self.config,
            self.negotiated_protocol
                .as_deref()
                .expect("running host completed protocol negotiation"),
            &input,
        )?;
        built.request["driver_stage"] = json!(if second_stage {
            "riichi_second_stage"
        } else {
            "primary"
        });
        self.request_infer(&built.request)?;
        let response = self.read_response_json(call_timeout)?;
        self.handle_common_response(&response, &built)?;
        if response_branch(&response)? == ResponseBranch::DeclareReach {
            return Ok(HostStageDecision::DeclareReach);
        }
        self.handle_response_3p(response, &built, input.legal_actions)
            .map(HostStageDecision::Decision)
    }

    /// Default inference timeout configured at process start (drivers cap absolute deadlines with it).
    #[must_use]
    pub fn configured_timeout(&self) -> Duration {
        self.config.timeout
    }

    pub fn infer_4p_or_fallback<F>(
        &mut self,
        input: InferenceInput4p<'_>,
        policy: FallbackPolicy,
        fallback_select: F,
    ) -> std::result::Result<HostInferenceOutcome, InferenceError>
    where
        F: FnOnce(&[LegalAction4p]) -> Option<usize>,
    {
        match self.infer_4p(input.clone()) {
            Ok(InferenceDecision::Abstain) => self.fallback_4p(
                policy,
                FallbackReason::Abstain,
                input.legal_actions,
                fallback_select,
            ),
            Ok(decision) => Ok(HostInferenceOutcome::Engine(decision)),
            Err(err) if policy == FallbackPolicy::ProductBasicDecider => self.fallback_4p(
                policy,
                FallbackReason::Error(err.kind),
                input.legal_actions,
                fallback_select,
            ),
            Err(err) => Err(err),
        }
    }

    pub fn infer_3p_or_fallback<F>(
        &mut self,
        input: InferenceInput3p<'_>,
        policy: FallbackPolicy,
        fallback_select: F,
    ) -> std::result::Result<HostInferenceOutcome, InferenceError>
    where
        F: FnOnce(&[LegalAction3p]) -> Option<usize>,
    {
        match self.infer_3p(input.clone()) {
            Ok(InferenceDecision::Abstain) => self.fallback_3p(
                policy,
                FallbackReason::Abstain,
                input.legal_actions,
                fallback_select,
            ),
            Ok(decision) => Ok(HostInferenceOutcome::Engine(decision)),
            Err(err) if policy == FallbackPolicy::ProductBasicDecider => self.fallback_3p(
                policy,
                FallbackReason::Error(err.kind),
                input.legal_actions,
                fallback_select,
            ),
            Err(err) => Err(err),
        }
    }

    pub fn end(&mut self) -> std::result::Result<(), InferenceError> {
        if self.child.is_none() {
            return Ok(());
        }
        let request = json!({
            "v": protocol_number(self.negotiated_protocol.as_deref().unwrap_or(FLYA_INFERENCE_PROTOCOL_V1)),
            "cmd": "end",
            "session_id": self.config.session_id,
        });
        self.write_json_line(&request)?;
        let response = self.read_response_json(self.config.timeout)?;
        if response.get("ok").and_then(Value::as_bool) == Some(true) {
            Ok(())
        } else {
            Err(inference_error(
                InferenceErrorKind::Protocol,
                format!("engine end failed: {response}"),
            ))
        }
    }

    fn fallback_4p<F>(
        &self,
        policy: FallbackPolicy,
        reason: FallbackReason,
        legal_actions: &[LegalAction4p],
        fallback_select: F,
    ) -> std::result::Result<HostInferenceOutcome, InferenceError>
    where
        F: FnOnce(&[LegalAction4p]) -> Option<usize>,
    {
        if policy == FallbackPolicy::FailMatch {
            return Err(inference_error(
                InferenceErrorKind::EngineUnavailable,
                "engine abstained under fail-match fallback policy",
            ));
        }
        let Some(index) = fallback_select(legal_actions) else {
            return Err(inference_error(
                InferenceErrorKind::Internal,
                "fallback selector produced no legal action",
            ));
        };
        if index >= legal_actions.len() {
            return Err(inference_error(
                InferenceErrorKind::InvalidAction,
                format!("fallback selector returned out-of-range legal index {index}"),
            ));
        }
        Ok(HostInferenceOutcome::Fallback {
            policy,
            reason,
            decision: InferenceDecision::Select { index, meta: None },
        })
    }

    fn fallback_3p<F>(
        &self,
        policy: FallbackPolicy,
        reason: FallbackReason,
        legal_actions: &[LegalAction3p],
        fallback_select: F,
    ) -> std::result::Result<HostInferenceOutcome, InferenceError>
    where
        F: FnOnce(&[LegalAction3p]) -> Option<usize>,
    {
        if policy == FallbackPolicy::FailMatch {
            return Err(inference_error(
                InferenceErrorKind::EngineUnavailable,
                "engine abstained under fail-match fallback policy",
            ));
        }
        let Some(index) = fallback_select(legal_actions) else {
            return Err(inference_error(
                InferenceErrorKind::Internal,
                "fallback selector produced no legal action",
            ));
        };
        if index >= legal_actions.len() {
            return Err(inference_error(
                InferenceErrorKind::InvalidAction,
                format!("fallback selector returned out-of-range legal index {index}"),
            ));
        }
        Ok(HostInferenceOutcome::Fallback {
            policy,
            reason,
            decision: InferenceDecision::Select { index, meta: None },
        })
    }

    fn ensure_running(&mut self) -> std::result::Result<(), InferenceError> {
        if self.child.is_none() || self.needs_rebuild {
            self.restart()?;
        }
        Ok(())
    }

    fn handshake(&mut self) -> std::result::Result<(), InferenceError> {
        if self.config.protocol_versions.is_empty() {
            return Err(inference_error(
                InferenceErrorKind::Protocol,
                "host protocol_versions must not be empty",
            ));
        }
        if self.config.protocol_versions.iter().any(|protocol| {
            !matches!(
                protocol.as_ref(),
                FLYA_INFERENCE_PROTOCOL_V1 | FLYA_INFERENCE_PROTOCOL_V2
            )
        }) {
            return Err(inference_error(
                InferenceErrorKind::Protocol,
                "host protocol_versions contains an unsupported version",
            ));
        }
        let request = json!({
            "v": 1,
            "cmd": "hello",
            "seat": self.config.seat,
            "rule_line": rule_line_wire(self.config.rule_line),
            "session_id": self.config.session_id,
            "match_context": self.config.match_context,
            "host_protocols": &self.config.protocol_versions,
        });
        self.write_json_line(&request)?;
        // The handshake read uses `config.timeout`, not the per-call deadline.
        let response = self.read_response_json(self.config.timeout)?;
        if response.get("ok").and_then(Value::as_bool) != Some(true) {
            self.needs_rebuild = true;
            return Err(inference_error(
                InferenceErrorKind::Protocol,
                format!("engine hello failed: {response}"),
            ));
        }
        if response.get("session_id").and_then(Value::as_str) != Some(&self.config.session_id) {
            self.needs_rebuild = true;
            return Err(inference_error(
                InferenceErrorKind::Protocol,
                "engine hello session_id mismatch",
            ));
        }

        let protocol_versions =
            string_array(response.get("protocol_versions")).ok_or_else(|| {
                inference_error(
                    InferenceErrorKind::Protocol,
                    "engine hello missing protocol_versions",
                )
            })?;
        let negotiated = self
            .config
            .protocol_versions
            .iter()
            .find(|host| {
                protocol_versions
                    .iter()
                    .any(|engine| engine == host.as_ref())
            })
            .cloned();
        let Some(negotiated) = negotiated else {
            self.needs_rebuild = true;
            return Err(inference_error(
                InferenceErrorKind::Protocol,
                "engine protocol_versions has no supported host intersection",
            ));
        };

        let caps = parse_engine_caps(&response)?;
        if !caps.rule_lines.contains(&self.config.rule_line) {
            self.needs_rebuild = true;
            return Err(inference_error(
                InferenceErrorKind::Protocol,
                "engine caps do not include requested rule line",
            ));
        }
        self.caps = Some(caps);
        self.negotiated_protocol = Some(negotiated);
        Ok(())
    }

    fn request_infer(&mut self, request: &Value) -> std::result::Result<(), InferenceError> {
        self.write_json_line(request)
    }

    fn handle_response_4p(
        &mut self,
        response: Value,
        built: &BuiltInferenceRequest,
        legal_actions: &[LegalAction4p],
    ) -> std::result::Result<InferenceDecision, InferenceError> {
        self.handle_common_response(&response, built)?;
        let protocol = self
            .negotiated_protocol
            .as_deref()
            .expect("running host completed protocol negotiation");
        let branch = response_branch(&response)?;
        match branch {
            ResponseBranch::Select => select_decision(&response, legal_actions.len()),
            ResponseBranch::Action => {
                let action = response.get("action").ok_or_else(|| {
                    inference_error(InferenceErrorKind::Protocol, "missing action branch")
                })?;
                let action = parse_response_action_4p(action, built.viewer_seat, protocol)?;
                let index = match_index_4p(&action, legal_actions)?;
                Ok(InferenceDecision::Select {
                    index,
                    meta: response_meta(&response, legal_actions.len()),
                })
            }
            ResponseBranch::Ranked => {
                if !self
                    .caps
                    .as_ref()
                    .map(|caps| caps.returns_ranked_actions)
                    .unwrap_or(false)
                {
                    return Err(inference_error(
                        InferenceErrorKind::Protocol,
                        "engine returned ranked actions without declaring returns_ranked_actions",
                    ));
                }
                ranked_decision_4p(&response, legal_actions, built.viewer_seat, protocol)
            }
            ResponseBranch::Abstain => Ok(InferenceDecision::Abstain),
            ResponseBranch::DeclareReach => Err(inference_error(
                InferenceErrorKind::Protocol,
                "declare_reach is only valid through the stage driver entry",
            )),
        }
    }

    fn handle_response_3p(
        &mut self,
        response: Value,
        built: &BuiltInferenceRequest,
        legal_actions: &[LegalAction3p],
    ) -> std::result::Result<InferenceDecision, InferenceError> {
        self.handle_common_response(&response, built)?;
        let protocol = self
            .negotiated_protocol
            .as_deref()
            .expect("running host completed protocol negotiation");
        let branch = response_branch(&response)?;
        match branch {
            ResponseBranch::Select => select_decision(&response, legal_actions.len()),
            ResponseBranch::Action => {
                let action = response.get("action").ok_or_else(|| {
                    inference_error(InferenceErrorKind::Protocol, "missing action branch")
                })?;
                let action = parse_response_action_3p(action, built.viewer_seat, protocol)?;
                let index = match_index_3p(&action, legal_actions)?;
                Ok(InferenceDecision::Select {
                    index,
                    meta: response_meta(&response, legal_actions.len()),
                })
            }
            ResponseBranch::Ranked => {
                if !self
                    .caps
                    .as_ref()
                    .map(|caps| caps.returns_ranked_actions)
                    .unwrap_or(false)
                {
                    return Err(inference_error(
                        InferenceErrorKind::Protocol,
                        "engine returned ranked actions without declaring returns_ranked_actions",
                    ));
                }
                ranked_decision_3p(&response, legal_actions, built.viewer_seat, protocol)
            }
            ResponseBranch::Abstain => Ok(InferenceDecision::Abstain),
            ResponseBranch::DeclareReach => Err(inference_error(
                InferenceErrorKind::Protocol,
                "declare_reach is only valid through the stage driver entry",
            )),
        }
    }

    fn handle_common_response(
        &mut self,
        response: &Value,
        built: &BuiltInferenceRequest,
    ) -> std::result::Result<(), InferenceError> {
        match response.get("ok").and_then(Value::as_bool) {
            Some(false) => {
                let err = error_response(response);
                if matches!(
                    err.kind,
                    InferenceErrorKind::Protocol
                        | InferenceErrorKind::EngineUnavailable
                        | InferenceErrorKind::Timeout
                ) {
                    self.needs_rebuild = true;
                }
                return Err(err);
            }
            Some(true) => {}
            None => {
                self.needs_rebuild = true;
                return Err(inference_error(
                    InferenceErrorKind::Protocol,
                    "engine response missing ok boolean",
                ));
            }
        }

        if response.get("decision_id").and_then(Value::as_str) != Some(&built.decision_id) {
            return Err(inference_error(
                InferenceErrorKind::StaleDecision,
                "engine decision_id echo mismatch",
            ));
        }
        if response.get("legal_digest").and_then(Value::as_str) != Some(&built.legal_digest) {
            return Err(inference_error(
                InferenceErrorKind::StaleDecision,
                "engine legal_digest echo mismatch",
            ));
        }
        if response.get("state_digest").and_then(Value::as_str) != Some(&built.state_digest) {
            return Err(inference_error(
                InferenceErrorKind::StaleDecision,
                "engine state_digest echo mismatch",
            ));
        }
        if response.get("engine_ack_seq").and_then(Value::as_u64) != Some(built.to_seq) {
            return Err(inference_error(
                InferenceErrorKind::StateSyncRequired,
                "engine_ack_seq does not match request to_seq",
            ));
        }
        if response
            .get("engine_ack_state_digest")
            .and_then(Value::as_str)
            != Some(&built.state_digest)
        {
            return Err(inference_error(
                InferenceErrorKind::StateSyncRequired,
                "engine_ack_state_digest does not match authoritative state_digest",
            ));
        }
        Ok(())
    }

    fn write_json_line(&mut self, value: &Value) -> std::result::Result<(), InferenceError> {
        let Some(stdin) = self.stdin.as_mut() else {
            self.needs_rebuild = true;
            return Err(inference_error(
                InferenceErrorKind::EngineUnavailable,
                "inference engine stdin is not available",
            ));
        };
        serde_json::to_writer(&mut *stdin, value).map_err(|err| {
            inference_error(
                InferenceErrorKind::Protocol,
                format!("failed to serialize engine request: {err}"),
            )
        })?;
        stdin.write_all(b"\n").map_err(|err| {
            self.needs_rebuild = true;
            inference_error(
                InferenceErrorKind::EngineUnavailable,
                format!("failed to write engine request newline: {err}"),
            )
        })?;
        stdin.flush().map_err(|err| {
            self.needs_rebuild = true;
            inference_error(
                InferenceErrorKind::EngineUnavailable,
                format!("failed to flush engine request: {err}"),
            )
        })
    }

    fn read_response_json(
        &mut self,
        timeout: Duration,
    ) -> std::result::Result<Value, InferenceError> {
        let line = self.read_line_with_timeout(timeout)?;
        serde_json::from_str(&line).map_err(|err| {
            self.needs_rebuild = true;
            inference_error(
                InferenceErrorKind::Protocol,
                format!("engine returned invalid JSON: {err}; line={line:?}"),
            )
        })
    }

    fn read_line_with_timeout(
        &mut self,
        timeout: Duration,
    ) -> std::result::Result<String, InferenceError> {
        let Some(reader) = self.reader.as_ref() else {
            self.needs_rebuild = true;
            return Err(inference_error(
                InferenceErrorKind::EngineUnavailable,
                "inference engine stdout reader is not available",
            ));
        };
        match reader.recv_timeout(timeout) {
            Ok(Ok(line)) => Ok(line),
            Ok(Err(err)) => {
                self.mark_unavailable();
                Err(inference_error(
                    InferenceErrorKind::EngineUnavailable,
                    format!("inference engine stdout closed: {err}"),
                ))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.mark_unavailable();
                Err(inference_error(
                    InferenceErrorKind::Timeout,
                    format!(
                        "inference engine timed out after {} ms",
                        timeout.as_millis()
                    ),
                ))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.mark_unavailable();
                Err(inference_error(
                    InferenceErrorKind::EngineUnavailable,
                    "inference engine stdout reader disconnected",
                ))
            }
        }
    }

    fn mark_unavailable(&mut self) {
        self.needs_rebuild = true;
        self.stop_child();
    }

    fn stop_child(&mut self) {
        self.stdin.take();
        self.reader.take();
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for SubprocessHost {
    fn drop(&mut self) {
        let _ = self.end();
        self.stop_child();
    }
}

fn spawn_stdout_reader(stdout: std::process::ChildStdout) -> Receiver<io::Result<String>> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    let _ = tx.send(Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "stdout EOF",
                    )));
                    break;
                }
                Ok(_) => {
                    while line.ends_with(['\n', '\r']) {
                        line.pop();
                    }
                    if tx.send(Ok(line)).is_err() {
                        break;
                    }
                }
                Err(err) => {
                    let _ = tx.send(Err(err));
                    break;
                }
            }
        }
    });
    rx
}

#[derive(Debug, Clone)]
struct BuiltInferenceRequest {
    request: Value,
    decision_id: String,
    legal_digest: String,
    state_digest: String,
    to_seq: u64,
    viewer_seat: u8,
}

fn effective_seat(config: &EngineProcessConfig, override_seat: Option<u8>) -> u8 {
    override_seat.unwrap_or(config.seat)
}

fn build_infer_request_4p(
    config: &EngineProcessConfig,
    protocol: &str,
    input: &InferenceInput4p<'_>,
) -> std::result::Result<BuiltInferenceRequest, InferenceError> {
    let viewer_seat = effective_seat(config, input.seat);
    let seat = viewer_seat as Actor4;
    let visible_events = visible_events_4p_to_wire(input.events, seat, protocol);
    let opening_tile = dealer_opening_tile_4p(input.events, seat);
    let legal_actions = input
        .legal_actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            legal_action_4p_to_protocol_wire(index, action, protocol, opening_tile)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    build_infer_request_common(
        config,
        protocol,
        viewer_seat,
        input.decision_id.clone(),
        input.phase,
        input.wall_remaining,
        input.source.clone(),
        visible_events,
        legal_actions,
        4,
        input.events.len() as u64,
        input.remote_auth.as_deref(),
        None,
    )
}

fn build_infer_request_3p(
    config: &EngineProcessConfig,
    protocol: &str,
    input: &InferenceInput3p<'_>,
) -> std::result::Result<BuiltInferenceRequest, InferenceError> {
    let viewer_seat = effective_seat(config, input.seat);
    let seat = viewer_seat as Actor3;
    let visible_events = visible_events_3p_to_wire(input.events, seat, protocol);
    let opening_tile = dealer_opening_tile_3p(input.events, seat);
    let legal_actions = input
        .legal_actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            legal_action_3p_to_protocol_wire(index, action, protocol, opening_tile)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    build_infer_request_common(
        config,
        protocol,
        viewer_seat,
        input.decision_id.clone(),
        input.phase,
        input.wall_remaining,
        input.source.clone(),
        visible_events,
        legal_actions,
        3,
        input.events.len() as u64,
        input.remote_auth.as_deref(),
        input.match_context.as_ref(),
    )
}

#[allow(clippy::too_many_arguments)]
fn build_infer_request_common(
    config: &EngineProcessConfig,
    protocol: &str,
    viewer_seat: u8,
    decision_id: String,
    phase: DecisionPhase,
    wall_remaining: u32,
    source: SourceInfo,
    visible_events: Vec<Value>,
    legal_actions: Vec<Value>,
    player_count: u8,
    to_seq: u64,
    remote_auth: Option<&str>,
    match_context: Option<&Value>,
) -> std::result::Result<BuiltInferenceRequest, InferenceError> {
    let legal_digest = digest_value(&json!({
        "protocol_version": protocol,
        "rule_line": rule_line_wire(config.rule_line),
        "viewer_seat": viewer_seat,
        "phase": phase.as_str(),
        "seq": to_seq,
        "wall_remaining": wall_remaining,
        "legal_actions": legal_actions,
    }));
    let state_digest = digest_value(&json!({
        "protocol_version": protocol,
        "rule_line": rule_line_wire(config.rule_line),
        "viewer_seat": viewer_seat,
        "source": source.digest_value(),
        "to_seq": to_seq,
        "wall_remaining": wall_remaining,
        "events": visible_events,
    }));
    let mut request = json!({
        "v": protocol_number(protocol),
        "cmd": "infer",
        "seat": viewer_seat,
        "decision_id": decision_id,
        "phase": phase.as_str(),
        "legal_digest": legal_digest,
        "state_digest": state_digest,
        "session_id": config.session_id,
        "state_sync": "full",
        "base_seq": Value::Null,
        "to_seq": to_seq,
        "state": {
            "rule_line": rule_line_wire(config.rule_line),
            "player_count": player_count,
            // Use this request's value if given, otherwise the handshake value.
            "match_context": match_context.unwrap_or(&config.match_context),
            "wall_remaining": wall_remaining,
            "source": source.wire_value(),
            "events": visible_events,
            "legal_actions": legal_actions,
        },
    });
    if let Some(ticket) = remote_auth.filter(|value| !value.is_empty()) {
        request["remote_auth"] = json!(ticket);
    }
    Ok(BuiltInferenceRequest {
        request,
        decision_id,
        legal_digest,
        state_digest,
        to_seq,
        viewer_seat,
    })
}

fn build_remote_infer_request_4p(
    config: &RemoteHttpConfig,
    input: &InferenceInput4p<'_>,
) -> std::result::Result<Value, InferenceError> {
    let events = input
        .events
        .iter()
        .map(|event| {
            serde_json::to_value(event).map_err(|err| {
                inference_error(
                    InferenceErrorKind::Internal,
                    format!("serialize 4p event for remote inference: {err}"),
                )
            })
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let legal_actions = input
        .legal_actions
        .iter()
        .enumerate()
        .map(|(index, action)| legal_action_4p_to_wire(index, action))
        .collect::<Vec<_>>();
    Ok(remote_infer_request_common(
        config,
        input.decision_id.clone(),
        input.phase,
        input.wall_remaining,
        events,
        legal_actions,
    ))
}

fn build_remote_infer_request_3p(
    config: &RemoteHttpConfig,
    input: &InferenceInput3p<'_>,
) -> std::result::Result<Value, InferenceError> {
    let events = input
        .events
        .iter()
        .map(|event| {
            serde_json::to_value(event).map_err(|err| {
                inference_error(
                    InferenceErrorKind::Internal,
                    format!("serialize 3p event for remote inference: {err}"),
                )
            })
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let legal_actions = input
        .legal_actions
        .iter()
        .enumerate()
        .map(|(index, action)| legal_action_3p_to_wire(index, action))
        .collect::<Vec<_>>();
    Ok(remote_infer_request_common(
        config,
        input.decision_id.clone(),
        input.phase,
        input.wall_remaining,
        events,
        legal_actions,
    ))
}

fn remote_infer_request_common(
    config: &RemoteHttpConfig,
    decision_id: String,
    phase: DecisionPhase,
    wall_remaining: u32,
    events: Vec<Value>,
    legal_actions: Vec<Value>,
) -> Value {
    json!({
        "session_id": config.session_id,
        "seat": config.seat,
        "phase": phase.as_str(),
        "decision_id": decision_id,
        "wall_remaining": wall_remaining,
        "match_context": config.match_context,
        "events": events,
        "legal_actions": legal_actions,
    })
}

fn parse_remote_infer_response(
    response: &Value,
    decision_id: &str,
    legal_actions: &[Value],
) -> std::result::Result<InferenceDecision, InferenceError> {
    match response.get("ok").and_then(Value::as_bool) {
        Some(false) => {
            let detail = response
                .get("message")
                .or_else(|| response.get("error"))
                .and_then(Value::as_str)
                .unwrap_or("remote inference returned ok:false");
            return Err(inference_error(InferenceErrorKind::Protocol, detail));
        }
        Some(true) => {}
        None => {
            return Err(inference_error(
                InferenceErrorKind::Protocol,
                "remote inference response missing ok boolean",
            ));
        }
    }

    if response.get("decision_id").and_then(Value::as_str) != Some(decision_id) {
        return Err(inference_error(
            InferenceErrorKind::StaleDecision,
            "remote inference decision_id echo mismatch",
        ));
    }
    let index = response
        .get("selected_index")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            inference_error(
                InferenceErrorKind::Protocol,
                "remote inference response missing selected_index",
            )
        })? as usize;
    let Some(expected_action) = legal_actions.get(index) else {
        return Err(inference_error(
            InferenceErrorKind::InvalidAction,
            format!("remote selected_index {index} out of range"),
        ));
    };
    if response.get("action") != Some(expected_action) {
        return Err(inference_error(
            InferenceErrorKind::InvalidAction,
            "remote inference action echo does not match selected_index",
        ));
    }
    let meta = response
        .get("meta")
        .filter(|value| !value.is_null())
        .map(|value| value.to_string().into_boxed_str());
    Ok(InferenceDecision::Select { index, meta })
}

fn protocol_number(protocol: &str) -> u8 {
    match protocol {
        FLYA_INFERENCE_PROTOCOL_V2 => 2,
        FLYA_INFERENCE_PROTOCOL_V1 => 1,
        _ => unreachable!("protocol was validated during handshake"),
    }
}

fn dealer_opening_tile_4p(events: &[Event4p], seat: Actor4) -> Option<Tile> {
    events.iter().rev().find_map(|event| match event {
        Event4p::DealerOpening { actor, pai } if *actor == seat => Some(*pai),
        _ => None,
    })
}

fn dealer_opening_tile_3p(events: &[Event3p], seat: Actor3) -> Option<Tile> {
    events.iter().rev().find_map(|event| match event {
        Event3p::DealerOpening { actor, pai } if *actor == seat => Some(*pai),
        _ => None,
    })
}

fn visible_events_4p_to_wire(events: &[Event4p], seat: Actor4, protocol: &str) -> Vec<Value> {
    let mut opening_tiles = [None; 4];
    events
        .iter()
        // Forced autoplay only exists in v2 and has no mjai equivalent. Older engines fail
        // the whole stream on unknown events rather than ignoring them, so these are
        // filtered out. They only describe opponent behavior, so replay is unaffected.
        .filter(|event| {
            protocol != FLYA_INFERENCE_PROTOCOL_V1
                || !matches!(
                    event,
                    Event4p::SeatForcedAutoplay { .. } | Event4p::SeatResumed { .. }
                )
        })
        .map(|event| {
            let visible = contract::project(event, seat);
            let mut wire = visible_event_4p_to_wire(&visible);
            if protocol == FLYA_INFERENCE_PROTOCOL_V1 {
                match visible.event {
                    VisibleBody4p::DealerOpening { actor, pai } => {
                        opening_tiles[actor as usize] = Some(pai);
                        wire["type"] = json!("tsumo");
                    }
                    VisibleBody4p::DealerOpeningDahai { actor, pai } => {
                        wire["type"] = json!("dahai");
                        wire["tsumogiri"] =
                            json!(opening_tiles[actor as usize].take() == Some(pai));
                    }
                    _ => {}
                }
            }
            wire
        })
        .collect()
}

fn visible_events_3p_to_wire(events: &[Event3p], seat: Actor3, protocol: &str) -> Vec<Value> {
    let mut opening_tiles = [None; 3];
    events
        .iter()
        // Forced autoplay only exists in v2 and has no mjai equivalent. Older engines fail
        // the whole stream on unknown events rather than ignoring them, so these are
        // filtered out. They only describe opponent behavior, so replay is unaffected.
        .filter(|event| {
            protocol != FLYA_INFERENCE_PROTOCOL_V1
                || !matches!(
                    event,
                    Event3p::SeatForcedAutoplay { .. } | Event3p::SeatResumed { .. }
                )
        })
        .map(|event| {
            let visible = contract::project(event, seat);
            let mut wire = visible_event_3p_to_wire(&visible);
            if protocol == FLYA_INFERENCE_PROTOCOL_V1 {
                match visible.event {
                    VisibleBody3p::DealerOpening { actor, pai } => {
                        opening_tiles[actor as usize] = Some(pai);
                        wire["type"] = json!("tsumo");
                    }
                    VisibleBody3p::DealerOpeningDahai { actor, pai } => {
                        wire["type"] = json!("dahai");
                        wire["tsumogiri"] =
                            json!(opening_tiles[actor as usize].take() == Some(pai));
                    }
                    _ => {}
                }
            }
            wire
        })
        .collect()
}

fn visible_event_4p_to_wire(visible: &VisibleEvent4p) -> Value {
    let viewer_seat = visible.viewer_seat;
    let mut event = match &visible.event {
        VisibleBody4p::None => json!({"type": "none"}),
        VisibleBody4p::StartGame { names } => {
            json!({"type": "start_game", "names": names})
        }
        VisibleBody4p::StartKyoku {
            bakaze,
            dora_marker,
            kyoku,
            honba,
            kyotaku,
            oya,
            scores,
            tehais,
        } => json!({
            "type": "start_kyoku",
            "bakaze": bakaze,
            "dora_marker": dora_marker,
            "kyoku": kyoku,
            "honba": honba,
            "kyotaku": kyotaku,
            "oya": oya,
            "scores": scores,
            "tehais": tehais,
        }),
        VisibleBody4p::Tsumo { actor, pai } => {
            json!({"type": "tsumo", "actor": actor, "pai": pai})
        }
        VisibleBody4p::DealerOpening { actor, pai } => {
            json!({"type": "dealer_opening", "actor": actor, "pai": pai})
        }
        VisibleBody4p::Dahai {
            actor,
            pai,
            tsumogiri,
        } => json!({"type": "dahai", "actor": actor, "pai": pai, "tsumogiri": tsumogiri}),
        VisibleBody4p::DealerOpeningDahai { actor, pai } => {
            json!({"type": "dealer_opening_dahai", "actor": actor, "pai": pai})
        }
        VisibleBody4p::SeatForcedAutoplay { actor } => {
            json!({"type": "seat_forced_autoplay", "actor": actor})
        }
        VisibleBody4p::SeatResumed { actor } => {
            json!({"type": "seat_resumed", "actor": actor})
        }
        VisibleBody4p::Chi {
            actor,
            target,
            pai,
            consumed,
        } => {
            json!({"type": "chi", "actor": actor, "target": target, "pai": pai, "consumed": consumed})
        }
        VisibleBody4p::Pon {
            actor,
            target,
            pai,
            consumed,
        } => {
            json!({"type": "pon", "actor": actor, "target": target, "pai": pai, "consumed": consumed})
        }
        VisibleBody4p::Daiminkan {
            actor,
            target,
            pai,
            consumed,
        } => {
            json!({"type": "daiminkan", "actor": actor, "target": target, "pai": pai, "consumed": consumed})
        }
        VisibleBody4p::Kakan {
            actor,
            pai,
            consumed,
        } => json!({"type": "kakan", "actor": actor, "pai": pai, "consumed": consumed}),
        VisibleBody4p::Ankan { actor, consumed } => {
            json!({"type": "ankan", "actor": actor, "consumed": consumed})
        }
        VisibleBody4p::Dora { dora_marker } => {
            json!({"type": "dora", "dora_marker": dora_marker})
        }
        VisibleBody4p::Reach { actor } => json!({"type": "reach", "actor": actor}),
        VisibleBody4p::ReachAccepted { actor } => {
            json!({"type": "reach_accepted", "actor": actor})
        }
        VisibleBody4p::Hora {
            actor,
            target,
            deltas,
            ura_markers,
            scoring,
        } => json!({
            "type": "hora",
            "actor": actor,
            "target": target,
            "deltas": deltas,
            "ura_markers": ura_markers,
            "scoring": scoring,
        }),
        VisibleBody4p::Ryukyoku { deltas } => {
            json!({"type": "ryukyoku", "deltas": deltas})
        }
        VisibleBody4p::EndKyoku => json!({"type": "end_kyoku"}),
        VisibleBody4p::EndGame => json!({"type": "end_game"}),
    };
    event["viewer_seat"] = json!(viewer_seat);
    event
}

fn visible_event_3p_to_wire(visible: &VisibleEvent3p) -> Value {
    let viewer_seat = visible.viewer_seat;
    let mut event = match &visible.event {
        VisibleBody3p::None => json!({"type": "none"}),
        VisibleBody3p::StartGame { names } => {
            json!({"type": "start_game", "names": names})
        }
        VisibleBody3p::StartKyoku {
            bakaze,
            dora_marker,
            kyoku,
            honba,
            kyotaku,
            oya,
            scores,
            tehais,
        } => json!({
            "type": "start_kyoku",
            "bakaze": bakaze,
            "dora_marker": dora_marker,
            "kyoku": kyoku,
            "honba": honba,
            "kyotaku": kyotaku,
            "oya": oya,
            "scores": scores,
            "tehais": tehais,
        }),
        VisibleBody3p::Tsumo { actor, pai } => {
            json!({"type": "tsumo", "actor": actor, "pai": pai})
        }
        VisibleBody3p::DealerOpening { actor, pai } => {
            json!({"type": "dealer_opening", "actor": actor, "pai": pai})
        }
        VisibleBody3p::Dahai {
            actor,
            pai,
            tsumogiri,
        } => json!({"type": "dahai", "actor": actor, "pai": pai, "tsumogiri": tsumogiri}),
        VisibleBody3p::DealerOpeningDahai { actor, pai } => {
            json!({"type": "dealer_opening_dahai", "actor": actor, "pai": pai})
        }
        VisibleBody3p::SeatForcedAutoplay { actor } => {
            json!({"type": "seat_forced_autoplay", "actor": actor})
        }
        VisibleBody3p::SeatResumed { actor } => {
            json!({"type": "seat_resumed", "actor": actor})
        }
        VisibleBody3p::Pon {
            actor,
            target,
            pai,
            consumed,
        } => {
            json!({"type": "pon", "actor": actor, "target": target, "pai": pai, "consumed": consumed})
        }
        VisibleBody3p::Daiminkan {
            actor,
            target,
            pai,
            consumed,
        } => {
            json!({"type": "daiminkan", "actor": actor, "target": target, "pai": pai, "consumed": consumed})
        }
        VisibleBody3p::Kakan {
            actor,
            pai,
            consumed,
        } => json!({"type": "kakan", "actor": actor, "pai": pai, "consumed": consumed}),
        VisibleBody3p::Ankan { actor, consumed } => {
            json!({"type": "ankan", "actor": actor, "consumed": consumed})
        }
        VisibleBody3p::Nukidora { actor, pai } => {
            json!({"type": "nukidora", "actor": actor, "pai": pai})
        }
        VisibleBody3p::Dora { dora_marker } => {
            json!({"type": "dora", "dora_marker": dora_marker})
        }
        VisibleBody3p::Reach { actor } => json!({"type": "reach", "actor": actor}),
        VisibleBody3p::ReachAccepted { actor } => {
            json!({"type": "reach_accepted", "actor": actor})
        }
        VisibleBody3p::Hora {
            actor,
            target,
            deltas,
            ura_markers,
            scoring,
        } => json!({
            "type": "hora",
            "actor": actor,
            "target": target,
            "deltas": deltas,
            "ura_markers": ura_markers,
            "scoring": scoring,
        }),
        VisibleBody3p::Ryukyoku { deltas } => {
            json!({"type": "ryukyoku", "deltas": deltas})
        }
        VisibleBody3p::EndKyoku => json!({"type": "end_kyoku"}),
        VisibleBody3p::EndGame => json!({"type": "end_game"}),
    };
    event["viewer_seat"] = json!(viewer_seat);
    event
}

/// Serializes a 4-player legal action into the current native wire JSON (with the
/// `action_id` index).
///
/// Networked hosts and any client holding Rust `LegalAction4p` values use this to
/// produce the canonical wire shape. Inverse of [`parse_legal_action_4p`].
pub fn legal_action_4p_to_wire(index: usize, action: &LegalAction4p) -> Value {
    let mut value = match action {
        LegalAction4p::Discard {
            pai,
            tsumogiri,
            riichi,
        } => json!({
            "type": action.wire_type(),
            "pai": pai,
            "tsumogiri": tsumogiri,
            "riichi": riichi,
        }),
        LegalAction4p::DealerOpeningDiscard { pai, riichi } => json!({
            "type": action.wire_type(),
            "pai": pai,
            "riichi": riichi,
        }),
        LegalAction4p::Kan {
            pai,
            kind: _,
            consumed,
        } => json!({
            "type": action.wire_type(),
            "pai": pai,
            "consumed": consumed,
        }),
        LegalAction4p::Tsumo { pai } => json!({
            "type": action.wire_type(),
            "pai": pai,
        }),
        LegalAction4p::DealerOpeningTsumo => json!({"type": action.wire_type()}),
        LegalAction4p::Kyushukyuhai => json!({"type": action.wire_type()}),
        LegalAction4p::PassAll { declines } => json!({
            "type": action.wire_type(),
            "declines": declines_to_wire(*declines),
        }),
        LegalAction4p::Pon { pai, consumed } => json!({
            "type": action.wire_type(),
            "pai": pai,
            "consumed": consumed,
        }),
        LegalAction4p::Chi { pai, consumed } => json!({
            "type": action.wire_type(),
            "pai": pai,
            "consumed": consumed,
        }),
        LegalAction4p::Ron { pai, target } => json!({
            "type": action.wire_type(),
            "pai": pai,
            "target": target,
        }),
    };
    value["action_id"] = json!(index);
    value
}

fn legal_action_4p_to_protocol_wire(
    index: usize,
    action: &LegalAction4p,
    protocol: &str,
    opening_tile: Option<Tile>,
) -> std::result::Result<Value, InferenceError> {
    if protocol != FLYA_INFERENCE_PROTOCOL_V1 {
        return Ok(legal_action_4p_to_wire(index, action));
    }
    Ok(match action {
        LegalAction4p::DealerOpeningDiscard { pai, riichi } => json!({
            "type": "dahai",
            "pai": pai,
            "tsumogiri": opening_tile == Some(*pai),
            "riichi": riichi,
            "action_id": index,
        }),
        LegalAction4p::DealerOpeningTsumo => json!({
            "type": "tsumo",
            "pai": opening_tile.ok_or_else(|| inference_error(
                InferenceErrorKind::Protocol,
                "v1 dealer-opening compatibility requires the opening tile event",
            ))?,
            "action_id": index,
        }),
        _ => legal_action_4p_to_wire(index, action),
    })
}

/// Serializes a 3-player legal action into the current native wire JSON (with the
/// `action_id` index). Inverse of [`parse_legal_action_3p`]; see [`legal_action_4p_to_wire`].
pub fn legal_action_3p_to_wire(index: usize, action: &LegalAction3p) -> Value {
    let mut value = match action {
        LegalAction3p::Discard {
            pai,
            tsumogiri,
            riichi,
        } => json!({
            "type": action.wire_type(),
            "pai": pai,
            "tsumogiri": tsumogiri,
            "riichi": riichi,
        }),
        LegalAction3p::DealerOpeningDiscard { pai, riichi } => json!({
            "type": action.wire_type(),
            "pai": pai,
            "riichi": riichi,
        }),
        LegalAction3p::Kan {
            pai,
            kind: _,
            consumed,
        } => json!({
            "type": action.wire_type(),
            "pai": pai,
            "consumed": consumed,
        }),
        LegalAction3p::Nukidora => json!({"type": action.wire_type()}),
        LegalAction3p::Tsumo { pai } => json!({
            "type": action.wire_type(),
            "pai": pai,
        }),
        LegalAction3p::DealerOpeningTsumo => json!({"type": action.wire_type()}),
        LegalAction3p::Kyushukyuhai => json!({"type": action.wire_type()}),
        LegalAction3p::PassAll { declines } => json!({
            "type": action.wire_type(),
            "declines": declines_to_wire(*declines),
        }),
        LegalAction3p::Pon { pai, consumed } => json!({
            "type": action.wire_type(),
            "pai": pai,
            "consumed": consumed,
        }),
        LegalAction3p::Ron { pai, target } => json!({
            "type": action.wire_type(),
            "pai": pai,
            "target": target,
        }),
    };
    value["action_id"] = json!(index);
    value
}

fn legal_action_3p_to_protocol_wire(
    index: usize,
    action: &LegalAction3p,
    protocol: &str,
    opening_tile: Option<Tile>,
) -> std::result::Result<Value, InferenceError> {
    if protocol != FLYA_INFERENCE_PROTOCOL_V1 {
        return Ok(legal_action_3p_to_wire(index, action));
    }
    Ok(match action {
        LegalAction3p::DealerOpeningDiscard { pai, riichi } => json!({
            "type": "dahai",
            "pai": pai,
            "tsumogiri": opening_tile == Some(*pai),
            "riichi": riichi,
            "action_id": index,
        }),
        LegalAction3p::DealerOpeningTsumo => json!({
            "type": "tsumo",
            "pai": opening_tile.ok_or_else(|| inference_error(
                InferenceErrorKind::Protocol,
                "v1 dealer-opening compatibility requires the opening tile event",
            ))?,
            "action_id": index,
        }),
        _ => legal_action_3p_to_wire(index, action),
    })
}

fn declines_to_wire(declines: ResponseOpportunities) -> Value {
    json!({
        "ron": declines.ron,
        "call": declines.call,
    })
}

#[derive(Debug, Deserialize)]
struct WireAction {
    #[serde(rename = "type")]
    kind: String,
    actor: Option<u8>,
    pai: Option<Tile>,
    tsumogiri: Option<bool>,
    riichi: Option<bool>,
    target: Option<u8>,
    consumed: Option<Vec<Tile>>,
    declines: Option<WireDeclines>,
}

#[derive(Debug, Deserialize)]
struct WireDeclines {
    ron: bool,
    call: bool,
}

fn validate_response_action_protocol(
    value: &Value,
    protocol: &str,
) -> std::result::Result<(), InferenceError> {
    if protocol != FLYA_INFERENCE_PROTOCOL_V1 {
        return Ok(());
    }
    let kind = value.get("type").and_then(Value::as_str);
    if kind.is_some_and(|kind| kind.starts_with("dealer_opening")) {
        return Err(inference_error(
            InferenceErrorKind::Protocol,
            "dealer_opening actions require flya-inference-v2",
        ));
    }
    if kind == Some("tsumo") && matches!(value.get("pai"), None | Some(Value::Null)) {
        return Err(inference_error(
            InferenceErrorKind::Protocol,
            "v1 tsumo actions require pai",
        ));
    }
    Ok(())
}

fn parse_response_action_4p(
    value: &Value,
    seat: u8,
    protocol: &str,
) -> std::result::Result<LegalAction4p, InferenceError> {
    validate_response_action_protocol(value, protocol)?;
    parse_action_4p(value, seat)
}

fn parse_response_action_3p(
    value: &Value,
    seat: u8,
    protocol: &str,
) -> std::result::Result<LegalAction3p, InferenceError> {
    validate_response_action_protocol(value, protocol)?;
    parse_action_3p(value, seat)
}

fn parse_action_4p(value: &Value, seat: u8) -> std::result::Result<LegalAction4p, InferenceError> {
    let wire: WireAction = serde_json::from_value(value.clone()).map_err(|err| {
        inference_error(
            InferenceErrorKind::Protocol,
            format!("invalid 4p action wire shape: {err}"),
        )
    })?;
    ensure_actor_matches(wire.actor, seat)?;
    match wire.kind.as_str() {
        "dahai" => Ok(LegalAction4p::Discard {
            pai: require_tile(wire.pai, "pai")?,
            tsumogiri: require_bool(wire.tsumogiri, "tsumogiri")?,
            riichi: require_bool(wire.riichi, "riichi")?,
        }),
        "dealer_opening_dahai" => Ok(LegalAction4p::DealerOpeningDiscard {
            pai: require_tile(wire.pai, "pai")?,
            riichi: require_bool(wire.riichi, "riichi")?,
        }),
        "ankan" | "kakan" | "daiminkan" => Ok(LegalAction4p::Kan {
            pai: require_tile(wire.pai, "pai")?,
            kind: parse_kan_kind(&wire.kind)?,
            consumed: require_consumed_len(wire.consumed, 3, 4)?,
        }),
        "tsumo" => Ok(match wire.pai {
            Some(pai) => LegalAction4p::Tsumo { pai },
            None => LegalAction4p::DealerOpeningTsumo,
        }),
        "kyushukyuhai" => Ok(LegalAction4p::Kyushukyuhai),
        "pass_all" => Ok(LegalAction4p::PassAll {
            declines: require_declines(wire.declines)?,
        }),
        "pon" => Ok(LegalAction4p::Pon {
            pai: require_tile(wire.pai, "pai")?,
            consumed: consumed_array_2(wire.consumed)?,
        }),
        "chi" => Ok(LegalAction4p::Chi {
            pai: require_tile(wire.pai, "pai")?,
            consumed: consumed_array_2(wire.consumed)?,
        }),
        "ron" => Ok(LegalAction4p::Ron {
            pai: require_tile(wire.pai, "pai")?,
            target: require_target(wire.target)?,
        }),
        other => Err(inference_error(
            InferenceErrorKind::Protocol,
            format!("unknown 4p action type {other:?}"),
        )),
    }
}

fn parse_action_3p(value: &Value, seat: u8) -> std::result::Result<LegalAction3p, InferenceError> {
    let wire: WireAction = serde_json::from_value(value.clone()).map_err(|err| {
        inference_error(
            InferenceErrorKind::Protocol,
            format!("invalid 3p action wire shape: {err}"),
        )
    })?;
    ensure_actor_matches(wire.actor, seat)?;
    match wire.kind.as_str() {
        "dahai" => Ok(LegalAction3p::Discard {
            pai: require_tile(wire.pai, "pai")?,
            tsumogiri: require_bool(wire.tsumogiri, "tsumogiri")?,
            riichi: require_bool(wire.riichi, "riichi")?,
        }),
        "dealer_opening_dahai" => Ok(LegalAction3p::DealerOpeningDiscard {
            pai: require_tile(wire.pai, "pai")?,
            riichi: require_bool(wire.riichi, "riichi")?,
        }),
        "ankan" | "kakan" | "daiminkan" => Ok(LegalAction3p::Kan {
            pai: require_tile(wire.pai, "pai")?,
            kind: parse_kan_kind(&wire.kind)?,
            consumed: require_consumed_len(wire.consumed, 3, 4)?,
        }),
        "nukidora" => Ok(LegalAction3p::Nukidora),
        "tsumo" => Ok(match wire.pai {
            Some(pai) => LegalAction3p::Tsumo { pai },
            None => LegalAction3p::DealerOpeningTsumo,
        }),
        "kyushukyuhai" => Ok(LegalAction3p::Kyushukyuhai),
        "pass_all" => Ok(LegalAction3p::PassAll {
            declines: require_declines(wire.declines)?,
        }),
        "pon" => Ok(LegalAction3p::Pon {
            pai: require_tile(wire.pai, "pai")?,
            consumed: consumed_array_2(wire.consumed)?,
        }),
        "ron" => Ok(LegalAction3p::Ron {
            pai: require_tile(wire.pai, "pai")?,
            target: require_target(wire.target)?,
        }),
        "chi" => Err(inference_error(
            InferenceErrorKind::Protocol,
            "3p action cannot be chi",
        )),
        other => Err(inference_error(
            InferenceErrorKind::Protocol,
            format!("unknown 3p action type {other:?}"),
        )),
    }
}

/// Parses an inference wire JSON legal action back into `LegalAction4p`.
///
/// A networked host that receives `legal_actions` in wire form rebuilds
/// `LegalAction4p` with this to drive [`SubprocessHost::infer_4p`], so canonical
/// parsing has one implementation. `seat` is the deciding seat (wire legal actions
/// have no `actor`, so it is only used for consistency checks). Inverse of
/// [`legal_action_4p_to_wire`].
pub fn parse_legal_action_4p(
    value: &Value,
    seat: u8,
) -> std::result::Result<LegalAction4p, InferenceError> {
    parse_action_4p(value, seat)
}

/// Parses an inference wire JSON legal action back into `LegalAction3p`. See
/// [`parse_legal_action_4p`]; inverse of [`legal_action_3p_to_wire`].
pub fn parse_legal_action_3p(
    value: &Value,
    seat: u8,
) -> std::result::Result<LegalAction3p, InferenceError> {
    parse_action_3p(value, seat)
}

fn ensure_actor_matches(actor: Option<u8>, seat: u8) -> std::result::Result<(), InferenceError> {
    if actor.is_some_and(|actor| actor != seat) {
        return Err(inference_error(
            InferenceErrorKind::InvalidAction,
            "action actor does not match requested seat",
        ));
    }
    Ok(())
}

fn require_tile(value: Option<Tile>, field: &str) -> std::result::Result<Tile, InferenceError> {
    value.ok_or_else(|| {
        inference_error(
            InferenceErrorKind::Protocol,
            format!("action missing required tile field {field}"),
        )
    })
}

fn require_bool(value: Option<bool>, field: &str) -> std::result::Result<bool, InferenceError> {
    value.ok_or_else(|| {
        inference_error(
            InferenceErrorKind::Protocol,
            format!("action missing required boolean field {field}"),
        )
    })
}

fn require_target(value: Option<u8>) -> std::result::Result<u8, InferenceError> {
    value.ok_or_else(|| {
        inference_error(
            InferenceErrorKind::Protocol,
            "action missing required target field",
        )
    })
}

fn require_declines(
    declines: Option<WireDeclines>,
) -> std::result::Result<ResponseOpportunities, InferenceError> {
    let declines = declines.ok_or_else(|| {
        inference_error(
            InferenceErrorKind::Protocol,
            "pass_all action missing declines",
        )
    })?;
    Ok(ResponseOpportunities {
        ron: declines.ron,
        call: declines.call,
    })
}

fn consumed_array_2(value: Option<Vec<Tile>>) -> std::result::Result<[Tile; 2], InferenceError> {
    let consumed = require_consumed_len(value, 2, 2)?;
    let tiles: [Tile; 2] = consumed.try_into().map_err(|_| {
        inference_error(
            InferenceErrorKind::Protocol,
            "consumed length changed while parsing pair",
        )
    })?;
    Ok(tiles)
}

fn require_consumed_len(
    value: Option<Vec<Tile>>,
    min: usize,
    max: usize,
) -> std::result::Result<Vec<Tile>, InferenceError> {
    let consumed = value.ok_or_else(|| {
        inference_error(
            InferenceErrorKind::Protocol,
            "action missing required consumed field",
        )
    })?;
    if consumed.len() < min || consumed.len() > max {
        return Err(inference_error(
            InferenceErrorKind::Protocol,
            format!(
                "action consumed length {} outside required range {min}..={max}",
                consumed.len()
            ),
        ));
    }
    Ok(consumed)
}

fn parse_kan_kind(kind: &str) -> std::result::Result<KanKind, InferenceError> {
    match kind {
        "ankan" => Ok(KanKind::Ankan),
        "kakan" => Ok(KanKind::Kakan),
        "daiminkan" => Ok(KanKind::Daiminkan),
        other => Err(inference_error(
            InferenceErrorKind::Protocol,
            format!("unknown kan action type {other:?}"),
        )),
    }
}

fn equivalent_4p(legal: &LegalAction4p, action: &LegalAction4p) -> bool {
    match (legal, action) {
        (
            LegalAction4p::DealerOpeningDiscard {
                pai: legal_pai,
                riichi: legal_riichi,
            },
            LegalAction4p::Discard {
                pai,
                riichi,
                tsumogiri: _,
            }
            | LegalAction4p::DealerOpeningDiscard { pai, riichi },
        ) => legal_pai == pai && legal_riichi == riichi,
        (
            LegalAction4p::DealerOpeningTsumo,
            LegalAction4p::DealerOpeningTsumo | LegalAction4p::Tsumo { .. },
        ) => true,
        _ => legal == action,
    }
}

fn equivalent_3p(legal: &LegalAction3p, action: &LegalAction3p) -> bool {
    match (legal, action) {
        (
            LegalAction3p::DealerOpeningDiscard {
                pai: legal_pai,
                riichi: legal_riichi,
            },
            LegalAction3p::Discard {
                pai,
                riichi,
                tsumogiri: _,
            }
            | LegalAction3p::DealerOpeningDiscard { pai, riichi },
        ) => legal_pai == pai && legal_riichi == riichi,
        (
            LegalAction3p::DealerOpeningTsumo,
            LegalAction3p::DealerOpeningTsumo | LegalAction3p::Tsumo { .. },
        ) => true,
        _ => legal == action,
    }
}

fn match_index_4p(
    action: &LegalAction4p,
    legal_actions: &[LegalAction4p],
) -> std::result::Result<usize, InferenceError> {
    legal_actions
        .iter()
        .position(|legal| equivalent_4p(legal, action))
        .ok_or_else(|| {
            inference_error(
                InferenceErrorKind::InvalidAction,
                format!("engine action does not exactly match current 4p legal list: {action:?}"),
            )
        })
}

fn match_index_3p(
    action: &LegalAction3p,
    legal_actions: &[LegalAction3p],
) -> std::result::Result<usize, InferenceError> {
    legal_actions
        .iter()
        .position(|legal| equivalent_3p(legal, action))
        .ok_or_else(|| {
            inference_error(
                InferenceErrorKind::InvalidAction,
                format!("engine action does not exactly match current 3p legal list: {action:?}"),
            )
        })
}

fn select_decision(
    response: &Value,
    legal_len: usize,
) -> std::result::Result<InferenceDecision, InferenceError> {
    let index = response
        .get("select")
        .and_then(|select| select.get("legal_index"))
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            inference_error(
                InferenceErrorKind::Protocol,
                "select branch missing legal_index",
            )
        })? as usize;
    if index >= legal_len {
        return Err(inference_error(
            InferenceErrorKind::InvalidAction,
            format!("engine selected out-of-range legal index {index}"),
        ));
    }
    Ok(InferenceDecision::Select {
        index,
        meta: response_meta(response, legal_len),
    })
}

fn ranked_decision_4p(
    response: &Value,
    legal_actions: &[LegalAction4p],
    seat: u8,
    protocol: &str,
) -> std::result::Result<InferenceDecision, InferenceError> {
    let ranked = ranked_entries(response)?;
    for entry in &ranked {
        validate_response_action_protocol(&entry.action, protocol)?;
    }
    for entry in ranked {
        let action = parse_action_4p(&entry.action, seat)?;
        if let Some(index) = legal_actions
            .iter()
            .position(|legal| equivalent_4p(legal, &action))
        {
            return Ok(InferenceDecision::Select {
                index,
                meta: response_meta(response, legal_actions.len()),
            });
        }
    }
    Err(inference_error(
        InferenceErrorKind::InvalidAction,
        "no ranked 4p action exactly matched current legal list",
    ))
}

fn ranked_decision_3p(
    response: &Value,
    legal_actions: &[LegalAction3p],
    seat: u8,
    protocol: &str,
) -> std::result::Result<InferenceDecision, InferenceError> {
    let ranked = ranked_entries(response)?;
    for entry in &ranked {
        validate_response_action_protocol(&entry.action, protocol)?;
    }
    for entry in ranked {
        let action = parse_action_3p(&entry.action, seat)?;
        if let Some(index) = legal_actions
            .iter()
            .position(|legal| equivalent_3p(legal, &action))
        {
            return Ok(InferenceDecision::Select {
                index,
                meta: response_meta(response, legal_actions.len()),
            });
        }
    }
    Err(inference_error(
        InferenceErrorKind::InvalidAction,
        "no ranked 3p action exactly matched current legal list",
    ))
}

#[derive(Debug, Clone)]
struct RankedEntry {
    action: Value,
    score: f64,
    original_index: usize,
}

fn ranked_entries(response: &Value) -> std::result::Result<Vec<RankedEntry>, InferenceError> {
    let items = response
        .get("ranked")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            inference_error(
                InferenceErrorKind::Protocol,
                "ranked branch must be an array",
            )
        })?;
    let mut entries = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let action = item.get("action").cloned().ok_or_else(|| {
            inference_error(InferenceErrorKind::Protocol, "ranked entry missing action")
        })?;
        let score = item.get("score").and_then(Value::as_f64).ok_or_else(|| {
            inference_error(
                InferenceErrorKind::Protocol,
                "ranked entry missing numeric score",
            )
        })?;
        entries.push(RankedEntry {
            action,
            score,
            original_index: index,
        });
    }
    entries.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.original_index.cmp(&b.original_index))
    });
    Ok(entries)
}

/// Turns the translator's `meta` into the opaque meta text the host passes to the frontend.
///
/// `meta.recommendation` is validated against its typed envelope and normalized: the
/// host parses it into a [`Recommendation`] (checking `action_ref` in `0..legal_len`,
/// probabilities in [0, 1], valid type tags) and writes it back in canonical form.
/// `id` / `title` / `label` pass through uninterpreted. Validation failures only drop
/// annotations, whole or individually, and never change play. Other meta fields are
/// kept as bounded opaque JSON.
fn response_meta(response: &Value, legal_len: usize) -> Option<Box<str>> {
    let meta = response.get("meta")?;
    let mut meta = meta.clone();
    if let Some(object) = meta.as_object_mut() {
        if object.contains_key("recommendation") {
            match parse_recommendation(&object["recommendation"], legal_len) {
                Some(rec) => {
                    object.insert("recommendation".to_string(), recommendation_to_wire(&rec));
                }
                None => {
                    object.remove("recommendation");
                    eprintln!(
                        "flya-inference-host: dropped malformed/empty meta.recommendation (legal_len={legal_len})"
                    );
                }
            }
        }
    }
    let mut raw = serde_json::to_string(&meta).ok()?;
    if raw.len() > MAX_META_BYTES {
        // Over the limit, trim optional fields in priority order and record it, keeping the recommendation.
        for key in META_TRIM_ORDER {
            if raw.len() <= MAX_META_BYTES {
                break;
            }
            let removed = meta
                .as_object_mut()
                .map(|object| object.remove(key).is_some())
                .unwrap_or(false);
            if removed {
                eprintln!(
                    "flya-inference-host: meta exceeded {MAX_META_BYTES} bytes; dropped optional field `{key}` (legal_len={legal_len})"
                );
                raw = serde_json::to_string(&meta).ok()?;
            }
        }
        if raw.len() > MAX_META_BYTES {
            if let Some(object) = meta.as_object_mut() {
                let recommendation = object.remove("recommendation");
                object.clear();
                if let Some(recommendation) = recommendation {
                    object.insert("recommendation".to_string(), recommendation);
                }
            }
            eprintln!(
                "flya-inference-host: meta still exceeded {MAX_META_BYTES} bytes; kept only meta.recommendation (legal_len={legal_len})"
            );
            raw = serde_json::to_string(&meta).ok()?;
        }
        if raw.len() > MAX_META_BYTES {
            eprintln!(
                "flya-inference-host: meta.recommendation alone exceeded {MAX_META_BYTES} bytes; dropping meta (legal_len={legal_len})"
            );
            return None;
        }
    }
    Some(raw.into_boxed_str())
}

// Recommendation annotations: typed envelope parsing and normalization
//
// Parsing is lenient per entry: a non-object top level drops everything; a malformed
// action or annotation drops only that entry. The frontend always gets a clean whole
// or a clean subset, and one bad annotation never discards the rest.

fn parse_recommendation(value: &Value, legal_len: usize) -> Option<Recommendation> {
    let object = value.as_object()?;
    let actions = object
        .get("actions")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| parse_action_annotation(item, legal_len))
                .take(MAX_RECOMMENDATION_ACTIONS)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let global = object
        .get("global")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(parse_annotation)
                .take(MAX_ANNOTATIONS_PER_SCOPE)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if actions.is_empty() && global.is_empty() {
        return None;
    }
    Some(Recommendation { actions, global })
}

fn parse_action_annotation(value: &Value, legal_len: usize) -> Option<ActionAnnotation> {
    let object = value.as_object()?;
    // `action_ref` must be within the legal actions, the only semantic check on the
    // display channel, so the frontend can bind the annotation to an action. Out of
    // range drops the entry.
    let action_ref = object.get("action_ref").and_then(Value::as_u64)? as usize;
    if legal_len != 0 && action_ref >= legal_len {
        return None;
    }
    let primary = object.get("primary").and_then(parse_primary_metric);
    let label = object
        .get("label")
        .and_then(Value::as_str)
        .map(str::to_string);
    let attributes = object
        .get("attributes")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(parse_annotation)
                .take(MAX_ANNOTATIONS_PER_SCOPE)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Some(ActionAnnotation {
        action_ref,
        primary,
        label,
        attributes,
    })
}

fn parse_primary_metric(value: &Value) -> Option<PrimaryMetric> {
    let object = value.as_object()?;
    match object.get("kind").and_then(Value::as_str)? {
        "probability" => {
            let prob = object.get("value").and_then(Value::as_f64)?;
            if !prob.is_finite() || !(0.0..=1.0).contains(&prob) {
                return None;
            }
            Some(PrimaryMetric::Probability(prob))
        }
        "rank" => {
            let rank = object.get("value").and_then(Value::as_u64)?;
            if rank == 0 {
                return None;
            }
            Some(PrimaryMetric::Rank(rank as u32))
        }
        _ => None,
    }
}

fn parse_annotation(value: &Value) -> Option<Annotation> {
    let object = value.as_object()?;
    let id = object.get("id").and_then(Value::as_str)?.trim().to_string();
    if id.is_empty() {
        return None;
    }
    let title = object
        .get("title")
        .and_then(Value::as_str)
        .map(str::to_string);
    let value = parse_annotation_value(object)?;
    // Known style tokens map to variants; unknown tokens pass through as `Other` for the frontend to render by type.
    let display = match object.get("display").and_then(Value::as_str) {
        Some(token) => AnnotationDisplay::from_wire(token),
        None => AnnotationDisplay::Follow,
    };
    Some(Annotation {
        id,
        title,
        value,
        display,
    })
}

fn parse_annotation_value(object: &serde_json::Map<String, Value>) -> Option<AnnotationValue> {
    match object.get("type").and_then(Value::as_str)? {
        "number" => {
            let number = object.get("value").and_then(Value::as_f64)?;
            if !number.is_finite() {
                return None;
            }
            let format = match object.get("format").and_then(Value::as_str) {
                Some("percent") => NumberFormat::Percent,
                _ => NumberFormat::Raw,
            };
            Some(AnnotationValue::Number {
                value: number,
                format,
            })
        }
        "semantic_id" => {
            let id = object.get("value").and_then(Value::as_str)?.to_string();
            if id.is_empty() {
                return None;
            }
            let label = object
                .get("label")
                .and_then(Value::as_str)
                .map(str::to_string);
            Some(AnnotationValue::SemanticId { value: id, label })
        }
        "text" => {
            let mut text = object.get("value").and_then(Value::as_str)?.to_string();
            if text.is_empty() {
                return None;
            }
            if text.len() > MAX_ANNOTATION_TEXT_BYTES {
                // Truncate on a character boundary so multi-byte UTF-8 is not split.
                let mut end = MAX_ANNOTATION_TEXT_BYTES;
                while end > 0 && !text.is_char_boundary(end) {
                    end -= 1;
                }
                text.truncate(end);
                text.push('…');
            }
            Some(AnnotationValue::Text { value: text })
        }
        _ => None,
    }
}

fn recommendation_to_wire(rec: &Recommendation) -> Value {
    json!({
        "actions": rec.actions.iter().map(action_annotation_to_wire).collect::<Vec<_>>(),
        "global": rec.global.iter().map(annotation_to_wire).collect::<Vec<_>>(),
    })
}

fn action_annotation_to_wire(action: &ActionAnnotation) -> Value {
    let mut value = json!({
        "action_ref": action.action_ref,
        "attributes": action.attributes.iter().map(annotation_to_wire).collect::<Vec<_>>(),
    });
    if let Some(primary) = &action.primary {
        value["primary"] = primary_metric_to_wire(primary);
    }
    if let Some(label) = &action.label {
        value["label"] = json!(label);
    }
    value
}

fn primary_metric_to_wire(primary: &PrimaryMetric) -> Value {
    match primary {
        PrimaryMetric::Probability(p) => json!({"kind": primary.wire_kind(), "value": p}),
        PrimaryMetric::Rank(r) => json!({"kind": primary.wire_kind(), "value": r}),
    }
}

fn annotation_to_wire(annotation: &Annotation) -> Value {
    let mut value = match &annotation.value {
        AnnotationValue::Number { value, format } => json!({
            "type": annotation.value.wire_type(),
            "value": value,
            "format": format.wire_value(),
        }),
        AnnotationValue::SemanticId { value, label } => {
            let mut out = json!({
                "type": annotation.value.wire_type(),
                "value": value,
            });
            if let Some(label) = label {
                out["label"] = json!(label);
            }
            out
        }
        AnnotationValue::Text { value } => json!({
            "type": annotation.value.wire_type(),
            "value": value,
        }),
    };
    value["id"] = json!(annotation.id);
    if let Some(title) = &annotation.title {
        value["title"] = json!(title);
    }
    value["display"] = json!(annotation.display.wire_value());
    value
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseBranch {
    Select,
    Action,
    Ranked,
    Abstain,
    DeclareReach,
}

fn response_branch(response: &Value) -> std::result::Result<ResponseBranch, InferenceError> {
    let mut branches = Vec::new();
    if response.get("select").is_some() {
        branches.push(ResponseBranch::Select);
    }
    if response.get("action").is_some() {
        branches.push(ResponseBranch::Action);
    }
    if response.get("ranked").is_some() {
        branches.push(ResponseBranch::Ranked);
    }
    if response.get("abstain").and_then(Value::as_bool) == Some(true) {
        branches.push(ResponseBranch::Abstain);
    }
    if response.get("declare_reach").and_then(Value::as_bool) == Some(true) {
        branches.push(ResponseBranch::DeclareReach);
    }
    match branches.as_slice() {
        [branch] => Ok(*branch),
        [] => Err(inference_error(
            InferenceErrorKind::Protocol,
            "ok:true response missing select|action|ranked|abstain|declare_reach branch",
        )),
        _ => Err(inference_error(
            InferenceErrorKind::Protocol,
            "ok:true response has multiple select|action|ranked|abstain|declare_reach branches",
        )),
    }
}

fn error_response(response: &Value) -> InferenceError {
    let kind = match response.get("error_kind").and_then(Value::as_str) {
        Some("protocol") => InferenceErrorKind::Protocol,
        Some("internal") => InferenceErrorKind::Internal,
        Some("stale") => InferenceErrorKind::StaleDecision,
        Some("unavailable") => InferenceErrorKind::EngineUnavailable,
        Some("state_sync_required") => InferenceErrorKind::StateSyncRequired,
        _ => InferenceErrorKind::Protocol,
    };
    let detail = response
        .get("detail")
        .and_then(Value::as_str)
        .unwrap_or("engine returned ok:false error");
    inference_error(kind, detail)
}

fn parse_engine_caps(response: &Value) -> std::result::Result<EngineCaps, InferenceError> {
    let caps = response.get("caps").ok_or_else(|| {
        inference_error(InferenceErrorKind::Protocol, "engine hello missing caps")
    })?;
    let caps_version = response
        .get("caps_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            inference_error(
                InferenceErrorKind::Protocol,
                "engine hello missing caps_version",
            )
        })? as u32;
    let protocol_versions = string_array(response.get("protocol_versions"))
        .ok_or_else(|| {
            inference_error(
                InferenceErrorKind::Protocol,
                "engine hello missing protocol_versions",
            )
        })?
        .into_iter()
        .map(Into::into)
        .collect();
    let rule_lines = string_array(caps.get("rule_lines"))
        .ok_or_else(|| {
            inference_error(
                InferenceErrorKind::Protocol,
                "engine caps missing rule_lines",
            )
        })?
        .into_iter()
        .map(|value| parse_rule_line(&value))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let riichi_style = caps
        .get("riichi_style")
        .and_then(Value::as_str)
        .map(parse_riichi_style)
        .transpose()?
        .ok_or_else(|| {
            inference_error(
                InferenceErrorKind::Protocol,
                "engine caps missing riichi_style",
            )
        })?;
    let supports_incremental = caps
        .get("supports_incremental")
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            inference_error(
                InferenceErrorKind::Protocol,
                "engine caps missing supports_incremental",
            )
        })?;
    let returns_ranked_actions = caps
        .get("returns_ranked_actions")
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            inference_error(
                InferenceErrorKind::Protocol,
                "engine caps missing returns_ranked_actions",
            )
        })?;
    Ok(EngineCaps {
        caps_version,
        protocol_versions,
        rule_lines,
        riichi_style,
        supports_incremental,
        returns_ranked_actions,
    })
}

fn string_array(value: Option<&Value>) -> Option<Vec<String>> {
    value.and_then(Value::as_array).map(|items| {
        items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect()
    })
}

fn parse_rule_line(value: &str) -> std::result::Result<RuleLine, InferenceError> {
    match value {
        "riichi4p" => Ok(RuleLine::Riichi4p),
        "riichi3p" => Ok(RuleLine::Riichi3p),
        other => Err(inference_error(
            InferenceErrorKind::Protocol,
            format!("unknown rule_line {other:?}"),
        )),
    }
}

fn parse_riichi_style(value: &str) -> std::result::Result<RiichiStyle, InferenceError> {
    match value {
        "parallel_discard" => Ok(RiichiStyle::ParallelDiscard),
        "declare_then_discard" => Ok(RiichiStyle::DeclareThenDiscard),
        other => Err(inference_error(
            InferenceErrorKind::Protocol,
            format!("unknown riichi_style {other:?}"),
        )),
    }
}

fn rule_line_wire(rule_line: RuleLine) -> &'static str {
    rule_line.wire_value()
}

fn digest_value(value: &Value) -> String {
    let canonical = canonical_json(value);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).expect("string serialization"),
        Value::Array(items) => {
            let body = items
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{body}]")
        }
        Value::Object(object) => {
            let sorted = object
                .iter()
                .map(|(key, value)| (key.as_str(), value))
                .collect::<BTreeMap<_, _>>();
            let body = sorted
                .into_iter()
                .map(|(key, value)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(key).expect("key serialization"),
                        canonical_json(value)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
    }
}

fn inference_error(kind: InferenceErrorKind, detail: impl Into<String>) -> InferenceError {
    InferenceError {
        kind,
        detail: detail.into().into_boxed_str(),
    }
}
