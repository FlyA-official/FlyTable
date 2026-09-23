//! `InferenceDriver`, a common abstraction over inference executors.
//!
//! `SubprocessHost` (local translator subprocess, stateful `&mut self` pipe) and
//! `RemoteHttpHost` (remote HTTP, stateless `&self`) have matching signatures but no
//! shared trait. [`InferenceDriver`] is an object-safe trait over both, so a
//! scheduler such as a worker pool can hold `Arc<dyn InferenceDriver>`.
//!
//! - `&self` with interior mutability: `LocalProcessDriver` wraps the `&mut`
//!   subprocess in a `Mutex`, so one driver is one subprocess and one unit of
//!   concurrency (`max_concurrency() == 1`). Concurrency for one model comes from
//!   several driver workers in a pool. `RemoteProviderDriver` is naturally
//!   concurrent and reports the provider budget as `max_concurrency()`.
//! - No live in-flight count in the trait, only the static `max_concurrency()` hint.
//!   In-flight counts are per-worker `AtomicUsize`s in the pool, so there is a single
//!   source of truth.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use flytable_seat::{InferenceDecision, InferenceError, InferenceErrorKind};

use crate::host::{
    HostStageDecision, InferenceInput3p, InferenceInput4p, RemoteHttpHost, SubprocessHost,
};

/// Internal stage of two-stage riichi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverStage {
    /// Main request.
    Primary,
    /// After `declare_then_discard` declares reach, ask which tile to discard.
    RiichiSecondStage,
}

/// Output of a stage: a regular decision, or (`declare_then_discard` stage 1) the
/// internal intermediate result of declaring riichi.
///
/// `DeclareReach` is never a public success response; it only appears inside the
/// host's two-stage folding and ends up as a single `riichi_dahai`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageDecision {
    Decision(InferenceDecision),
    DeclareReach,
}

/// Cheap, non-blocking driver health snapshot for scheduling and monitoring; not used for correctness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverHealth {
    Healthy,
    Unhealthy(String),
    Unknown,
}

/// Common inference executor: takes the full state and the authoritative legal
/// actions and returns a decision (selected index or abstain).
///
/// Object-safe (`&self`, no generic methods, owned returns), so it works as
/// `Arc<dyn InferenceDriver>`. Canonical state and actions are normalized before
/// leaving this trait; third-party observations, protocols and credentials stay
/// inside implementations and never cross this boundary.
pub trait InferenceDriver: Send + Sync {
    /// 4-player inference for one decision.
    fn infer_4p(&self, input: InferenceInput4p<'_>) -> Result<InferenceDecision, InferenceError>;
    /// 3-player inference for one decision.
    fn infer_3p(&self, input: InferenceInput3p<'_>) -> Result<InferenceDecision, InferenceError>;
    /// Warm-up (lazy weight loading or connections). No-op by default.
    fn warmup(&self) -> Result<(), InferenceError> {
        Ok(())
    }
    /// Health snapshot. `Unknown` by default.
    fn health(&self) -> DriverHealth {
        DriverHealth::Unknown
    }
    /// Maximum concurrency of one driver instance (local = 1, remote = provider budget), used as the per-worker cap.
    fn max_concurrency(&self) -> usize {
        1
    }
    /// Drain or shut down (best effort). No-op by default.
    fn drain(&self) {}

    /// Two-stage riichi stage inference. The default wraps `infer_4p` as a `Decision`, so
    /// every existing `parallel_discard` driver keeps its behavior and never returns
    /// `DeclareReach`. `declare_then_discard` drivers override this and return
    /// `DeclareReach` in stage 1; the host then appends a temporary reach event, runs
    /// stage 2 and folds both into one `riichi_dahai`. Public responses never contain a
    /// bare reach.
    fn infer_stage_4p(
        &self,
        input: InferenceInput4p<'_>,
        _stage: DriverStage,
    ) -> Result<StageDecision, InferenceError> {
        Ok(StageDecision::Decision(self.infer_4p(input)?))
    }
    /// 3-player version of [`InferenceDriver::infer_stage_4p`].
    fn infer_stage_3p(
        &self,
        input: InferenceInput3p<'_>,
        _stage: DriverStage,
    ) -> Result<StageDecision, InferenceError> {
        Ok(StageDecision::Decision(self.infer_3p(input)?))
    }

    /// Stage inference with a per-call absolute deadline (so a reused worker can use
    /// `min(remaining, cap)`). The default ignores the deadline and falls back to
    /// [`InferenceDriver::infer_stage_4p`]. Only drivers that can turn the deadline into a
    /// read timeout for the call (`LocalProcessDriver`) override it;
    /// `ProviderScheduledDriver` overrides it to pass the deadline through within its retry
    /// budget. `None` uses the node's default timeout.
    fn infer_stage_4p_within(
        &self,
        input: InferenceInput4p<'_>,
        stage: DriverStage,
        _deadline: Option<Instant>,
    ) -> Result<StageDecision, InferenceError> {
        self.infer_stage_4p(input, stage)
    }
    /// 3-player version of [`InferenceDriver::infer_stage_4p_within`].
    fn infer_stage_3p_within(
        &self,
        input: InferenceInput3p<'_>,
        stage: DriverStage,
        _deadline: Option<Instant>,
    ) -> Result<StageDecision, InferenceError> {
        self.infer_stage_3p(input, stage)
    }
}

/// Turns an absolute deadline into the read timeout for this call. An expired
/// deadline returns ZERO so the caller times out immediately; no extra margin is
/// added, so the ingress budget is never exceeded.
fn call_timeout_from_deadline(deadline: Option<Instant>, cap: Duration) -> Duration {
    match deadline {
        None => cap,
        Some(d) => d.saturating_duration_since(Instant::now()).min(cap),
    }
}

/// Local translator subprocess driver: wraps `&mut SubprocessHost` in a `Mutex`; one instance is one subprocess.
pub struct LocalProcessDriver {
    host: Mutex<SubprocessHost>,
}

impl LocalProcessDriver {
    #[must_use]
    pub fn new(host: SubprocessHost) -> Self {
        Self {
            host: Mutex::new(host),
        }
    }

    /// Starts the subprocess and wraps it.
    pub fn start(config: crate::host::EngineProcessConfig) -> Result<Self, InferenceError> {
        Ok(Self::new(SubprocessHost::start(config)?))
    }
}

impl std::fmt::Debug for LocalProcessDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalProcessDriver").finish_non_exhaustive()
    }
}

impl InferenceDriver for LocalProcessDriver {
    fn infer_4p(&self, input: InferenceInput4p<'_>) -> Result<InferenceDecision, InferenceError> {
        let mut host = self.host.lock().unwrap_or_else(|e| e.into_inner());
        host.infer_4p(input)
    }

    fn infer_3p(&self, input: InferenceInput3p<'_>) -> Result<InferenceDecision, InferenceError> {
        let mut host = self.host.lock().unwrap_or_else(|e| e.into_inner());
        host.infer_3p(input)
    }

    fn health(&self) -> DriverHealth {
        // Cheap snapshot: busy (lock held) means alive; otherwise check whether a restart is pending; poisoned means unhealthy.
        match self.host.try_lock() {
            Ok(host) if host.needs_rebuild() => {
                DriverHealth::Unhealthy("subprocess needs rebuild".to_string())
            }
            Ok(_) => DriverHealth::Healthy,
            Err(std::sync::TryLockError::WouldBlock) => DriverHealth::Healthy,
            Err(std::sync::TryLockError::Poisoned(_)) => {
                DriverHealth::Unhealthy("subprocess mutex poisoned".to_string())
            }
        }
    }

    fn max_concurrency(&self) -> usize {
        1
    }

    fn drain(&self) {
        let mut host = self.host.lock().unwrap_or_else(|e| e.into_inner());
        let _ = host.end();
    }

    fn infer_stage_4p_within(
        &self,
        input: InferenceInput4p<'_>,
        stage: DriverStage,
        deadline: Option<Instant>,
    ) -> Result<StageDecision, InferenceError> {
        // parallel_discard: the engine folds riichi itself, so the stage is just infer_4p; the deadline caps the read timeout.
        let mut host = self.host.lock().unwrap_or_else(|e| e.into_inner());
        let call_timeout = call_timeout_from_deadline(deadline, host.configured_timeout());
        if call_timeout.is_zero() {
            return Err(InferenceError {
                kind: InferenceErrorKind::Timeout,
                detail: "decision deadline exhausted before driver call".into(),
            });
        }
        match host.infer_stage_4p_within(
            input,
            call_timeout,
            stage == DriverStage::RiichiSecondStage,
        )? {
            HostStageDecision::Decision(decision) => Ok(StageDecision::Decision(decision)),
            HostStageDecision::DeclareReach => Ok(StageDecision::DeclareReach),
        }
    }

    fn infer_stage_3p_within(
        &self,
        input: InferenceInput3p<'_>,
        stage: DriverStage,
        deadline: Option<Instant>,
    ) -> Result<StageDecision, InferenceError> {
        let mut host = self.host.lock().unwrap_or_else(|e| e.into_inner());
        let call_timeout = call_timeout_from_deadline(deadline, host.configured_timeout());
        if call_timeout.is_zero() {
            return Err(InferenceError {
                kind: InferenceErrorKind::Timeout,
                detail: "decision deadline exhausted before driver call".into(),
            });
        }
        match host.infer_stage_3p_within(
            input,
            call_timeout,
            stage == DriverStage::RiichiSecondStage,
        )? {
            HostStageDecision::Decision(decision) => Ok(StageDecision::Decision(decision)),
            HostStageDecision::DeclareReach => Ok(StageDecision::DeclareReach),
        }
    }
}

/// Remote provider driver wrapping `RemoteHttpHost` (itself concurrent through `&self`).
/// `max_concurrency` is the provider's budget, so the pool allows several in-flight calls on one worker.
pub struct RemoteProviderDriver {
    host: RemoteHttpHost,
    max_concurrency: usize,
}

impl RemoteProviderDriver {
    #[must_use]
    pub fn new(host: RemoteHttpHost, max_concurrency: usize) -> Self {
        Self {
            host,
            max_concurrency: max_concurrency.max(1),
        }
    }
}

impl std::fmt::Debug for RemoteProviderDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteProviderDriver")
            .field("max_concurrency", &self.max_concurrency)
            .finish_non_exhaustive()
    }
}

impl InferenceDriver for RemoteProviderDriver {
    fn infer_4p(&self, input: InferenceInput4p<'_>) -> Result<InferenceDecision, InferenceError> {
        self.host.infer_4p(input)
    }

    fn infer_3p(&self, input: InferenceInput3p<'_>) -> Result<InferenceDecision, InferenceError> {
        self.host.infer_3p(input)
    }

    fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }

    fn health(&self) -> DriverHealth {
        DriverHealth::Healthy
    }
}

/// Provider scheduling policy.
#[derive(Debug, Clone)]
pub struct ProviderPolicy {
    /// Concurrency budget (maximum in flight). `0` delegates to `inner.max_concurrency()`
    /// (1 for a local translator pipe). Do not set it above what the pipe can handle; the
    /// extra calls would only serialize on the host lock.
    pub max_concurrency: usize,
    /// Bounded retry budget. Only pre-send-safe `EngineUnavailable` errors are retried,
    /// never `Timeout`: after sending, a retry could charge a metered remote call twice.
    pub max_retries: usize,
    /// Consecutive failures that open the circuit breaker.
    pub trip_threshold: usize,
    /// Time after which an open breaker becomes half-open and lets one probe through.
    pub cooldown: Duration,
}

impl Default for ProviderPolicy {
    fn default() -> Self {
        Self {
            max_concurrency: 0, // delegate to inner (1 for a local translator pipe)
            max_retries: 0,     // metered calls: no retries by default
            trip_threshold: 5,
            cooldown: Duration::from_secs(10),
        }
    }
}

/// Minimal counting semaphore (`Mutex` + `Condvar`; this crate has no tokio or crossbeam).
struct Semaphore {
    permits: Mutex<usize>,
    cv: Condvar,
}

impl Semaphore {
    fn new(n: usize) -> Self {
        Self {
            permits: Mutex::new(n.max(1)),
            cv: Condvar::new(),
        }
    }

    fn acquire_until(&self, deadline: Option<Instant>) -> Result<SemGuard<'_>, InferenceError> {
        let mut p = self.permits.lock().unwrap_or_else(|e| e.into_inner());
        while *p == 0 {
            match deadline {
                None => p = self.cv.wait(p).unwrap_or_else(|e| e.into_inner()),
                Some(deadline) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return Err(InferenceError {
                            kind: InferenceErrorKind::Timeout,
                            detail: "provider concurrency wait exceeded decision deadline".into(),
                        });
                    }
                    let (guard, wait) = self
                        .cv
                        .wait_timeout(p, remaining)
                        .unwrap_or_else(|e| e.into_inner());
                    p = guard;
                    if wait.timed_out() && *p == 0 {
                        return Err(InferenceError {
                            kind: InferenceErrorKind::Timeout,
                            detail: "provider concurrency wait exceeded decision deadline".into(),
                        });
                    }
                }
            }
        }
        *p -= 1;
        Ok(SemGuard { sem: self })
    }
}

struct SemGuard<'a> {
    sem: &'a Semaphore,
}

impl Drop for SemGuard<'_> {
    fn drop(&mut self) {
        let mut p = self.sem.permits.lock().unwrap_or_else(|e| e.into_inner());
        *p += 1;
        self.sem.cv.notify_one();
    }
}

struct BreakerState {
    consecutive_failures: usize,
    open_until: Option<Instant>,
    half_open_inflight: bool,
}

/// Provider snapshot (concurrency, retries, breaker).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderMetrics {
    pub calls: usize,
    pub failures: usize,
    pub rejected_open: usize,
    pub circuit_open: bool,
}

/// Provider scheduling decorator: adds an independent concurrency budget, bounded
/// idempotency-safe retries, a circuit breaker, health and metrics to any
/// `InferenceDriver`, without touching credentials (remote tickets still flow
/// opaquely through `InferenceInput.remote_auth` to the inner translator and are only
/// used at the actual remote boundary).
///
/// Typical use: `inner` is a local translator `LocalProcessDriver` that does the remote
/// protocol conversion, and this decorator adds provider scheduling around it so
/// unbounded or serialized HTTP calls inside the translator do not hide scheduling
/// problems.
pub struct ProviderScheduledDriver {
    inner: Arc<dyn InferenceDriver>,
    policy: ProviderPolicy,
    concurrency: usize,
    sem: Semaphore,
    breaker: Mutex<BreakerState>,
    calls: AtomicUsize,
    failures: AtomicUsize,
    rejected_open: AtomicUsize,
}

impl ProviderScheduledDriver {
    #[must_use]
    pub fn new(inner: Arc<dyn InferenceDriver>, policy: ProviderPolicy) -> Self {
        let concurrency = if policy.max_concurrency == 0 {
            inner.max_concurrency().max(1)
        } else {
            policy.max_concurrency
        };
        Self {
            inner,
            policy,
            concurrency,
            sem: Semaphore::new(concurrency),
            breaker: Mutex::new(BreakerState {
                consecutive_failures: 0,
                open_until: None,
                half_open_inflight: false,
            }),
            calls: AtomicUsize::new(0),
            failures: AtomicUsize::new(0),
            rejected_open: AtomicUsize::new(0),
        }
    }

    /// Metrics snapshot.
    #[must_use]
    pub fn metrics(&self) -> ProviderMetrics {
        let open = {
            let st = self.breaker.lock().unwrap_or_else(|e| e.into_inner());
            matches!(st.open_until, Some(until) if Instant::now() < until)
        };
        ProviderMetrics {
            calls: self.calls.load(Ordering::SeqCst),
            failures: self.failures.load(Ordering::SeqCst),
            rejected_open: self.rejected_open.load(Ordering::SeqCst),
            circuit_open: open,
        }
    }

    /// Breaker precheck: open and still cooling down rejects without calling inner; after
    /// the cooldown it goes half-open and lets one probe through.
    fn precheck_breaker(&self) -> Result<(), InferenceError> {
        let mut st = self.breaker.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(until) = st.open_until {
            if Instant::now() < until {
                drop(st);
                self.rejected_open.fetch_add(1, Ordering::SeqCst);
                return Err(InferenceError {
                    kind: InferenceErrorKind::EngineUnavailable,
                    detail: "provider circuit open".into(),
                });
            }
            if st.half_open_inflight {
                drop(st);
                self.rejected_open.fetch_add(1, Ordering::SeqCst);
                return Err(InferenceError {
                    kind: InferenceErrorKind::EngineUnavailable,
                    detail: "provider half-open probe already in flight".into(),
                });
            }
            st.half_open_inflight = true;
        }
        Ok(())
    }

    fn on_success(&self) {
        let mut st = self.breaker.lock().unwrap_or_else(|e| e.into_inner());
        st.consecutive_failures = 0;
        st.open_until = None;
        st.half_open_inflight = false;
    }

    fn on_failure(&self) {
        self.failures.fetch_add(1, Ordering::SeqCst);
        let mut st = self.breaker.lock().unwrap_or_else(|e| e.into_inner());
        st.consecutive_failures += 1;
        st.half_open_inflight = false;
        if st.consecutive_failures >= self.policy.trip_threshold {
            st.open_until = Some(Instant::now() + self.policy.cooldown);
        }
    }

    /// Common execution: breaker, then a concurrency slot, then `one_attempt` with bounded,
    /// idempotency-safe retries.
    fn execute<T>(
        &self,
        deadline: Option<Instant>,
        mut one_attempt: impl FnMut() -> Result<T, InferenceError>,
    ) -> Result<T, InferenceError> {
        let _permit = self.sem.acquire_until(deadline)?;
        // Another call may have opened the breaker while waiting for a slot; check again and take the half-open probe exclusively.
        self.precheck_breaker()?;
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut attempt = 0;
        loop {
            match one_attempt() {
                Ok(v) => {
                    self.on_success();
                    return Ok(v);
                }
                Err(e) => {
                    // Only pre-send-safe EngineUnavailable is retried; Timeout never is.
                    if e.kind == InferenceErrorKind::PreSendUnavailable
                        && attempt < self.policy.max_retries
                        && deadline.is_none_or(|d| Instant::now() < d)
                    {
                        attempt += 1;
                        continue;
                    }
                    self.on_failure();
                    return Err(e);
                }
            }
        }
    }
}

impl std::fmt::Debug for ProviderScheduledDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderScheduledDriver")
            .field("concurrency", &self.concurrency)
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

impl InferenceDriver for ProviderScheduledDriver {
    fn infer_4p(&self, input: InferenceInput4p<'_>) -> Result<InferenceDecision, InferenceError> {
        self.execute(None, || self.inner.infer_4p(input.clone()))
    }

    fn infer_3p(&self, input: InferenceInput3p<'_>) -> Result<InferenceDecision, InferenceError> {
        self.execute(None, || self.inner.infer_3p(input.clone()))
    }

    fn infer_stage_4p(
        &self,
        input: InferenceInput4p<'_>,
        stage: DriverStage,
    ) -> Result<StageDecision, InferenceError> {
        self.execute(None, || self.inner.infer_stage_4p(input.clone(), stage))
    }

    fn infer_stage_3p(
        &self,
        input: InferenceInput3p<'_>,
        stage: DriverStage,
    ) -> Result<StageDecision, InferenceError> {
        self.execute(None, || self.inner.infer_stage_3p(input.clone(), stage))
    }

    fn infer_stage_4p_within(
        &self,
        input: InferenceInput4p<'_>,
        stage: DriverStage,
        deadline: Option<Instant>,
    ) -> Result<StageDecision, InferenceError> {
        // Breaker and retries still apply, and the deadline is passed to inner (shared across retries).
        self.execute(deadline, || {
            self.inner
                .infer_stage_4p_within(input.clone(), stage, deadline)
        })
    }

    fn infer_stage_3p_within(
        &self,
        input: InferenceInput3p<'_>,
        stage: DriverStage,
        deadline: Option<Instant>,
    ) -> Result<StageDecision, InferenceError> {
        self.execute(deadline, || {
            self.inner
                .infer_stage_3p_within(input.clone(), stage, deadline)
        })
    }

    fn warmup(&self) -> Result<(), InferenceError> {
        self.inner.warmup()
    }

    fn max_concurrency(&self) -> usize {
        self.concurrency
    }

    fn health(&self) -> DriverHealth {
        let open = {
            let st = self.breaker.lock().unwrap_or_else(|e| e.into_inner());
            matches!(st.open_until, Some(until) if Instant::now() < until)
        };
        if open {
            DriverHealth::Unhealthy("provider circuit open".to_string())
        } else {
            self.inner.health()
        }
    }

    fn drain(&self) {
        self.inner.drain();
    }
}
