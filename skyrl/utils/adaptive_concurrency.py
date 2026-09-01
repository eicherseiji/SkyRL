"""Backend-neutral contracts for adaptive asynchronous sampling admission."""

import asyncio
import math
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias, runtime_checkable

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

ENGINE_LOADS_METRIC = "engine.loads"


@dataclass(frozen=True)
class VLLMEngineLoad:
    """One coherent vLLM engine-load observation.

    This is deliberately a value object instead of a loose metrics dictionary:
    policies that depend on KV pressure, waiting work, or preemptions must not
    accidentally combine values from different engines or scrape intervals.
    """

    engine_id: str
    role: str | None
    kv_capacity_tokens: int | None
    max_model_len: int | None
    kv_usage: float
    running: int
    waiting: int
    waiting_capacity: int | None
    preemptions_delta: int

    def __post_init__(self) -> None:
        if not self.engine_id:
            raise ValueError("engine_id must not be empty")
        for name in ("kv_capacity_tokens", "max_model_len", "waiting_capacity"):
            value = getattr(self, name)
            if value is not None:
                _require_non_negative_int(value, path=name)
        for name in ("running", "waiting", "preemptions_delta"):
            _require_non_negative_int(getattr(self, name), path=name)
        if isinstance(self.kv_usage, bool) or not isinstance(self.kv_usage, (int, float)):
            raise TypeError(f"kv_usage must be numeric, got {type(self.kv_usage).__name__}")
        if not math.isfinite(self.kv_usage) or self.kv_usage < 0:
            raise ValueError(f"kv_usage must be a finite non-negative value, got {self.kv_usage}")


@dataclass(frozen=True)
class SamplingFeedback:
    """Generic backend feedback for custom integrations.

    Built-in backends should prefer :class:`VLLMEngineSamplingFeedback` or
    :class:`RequestSamplingFeedback`. ``metrics`` remains deliberately
    extensible for custom backends; common names and units are validated so the
    built-in policies can also consume a generic event when appropriate.
    """

    metrics: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metrics = dict(self.metrics)
        _validate_json_value(metrics, path="metrics")
        _validate_common_metrics(metrics)
        object.__setattr__(self, "metrics", metrics)

    def number(self, name: str) -> float | None:
        """Return an optional numeric metric after excluding booleans."""

        value = self.metrics.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
        return float(value)

    def boolean(self, name: str) -> bool | None:
        """Return an optional boolean metric."""

        value = self.metrics.get(name)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")
        return value


@dataclass(frozen=True)
class RequestSamplingFeedback(SamplingFeedback):
    """Typed request-level feedback shared by hosted generation backends.

    Tinker can populate request latency, tokens, and cache hits. Fireworks
    dedicated deployments additionally expose prefill queue duration in
    response headers; serverless Fireworks deployments instead communicate
    overload through HTTP 429/503. Missing fields are expected and remain
    ``None`` rather than being guessed.
    """

    source: Literal["tinker", "fireworks", "hosted", "unknown"] = "unknown"
    queue_duration_s: float | None = None
    request_latency_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_hit_tokens: int | None = None
    rate_limited: bool = False

    def __post_init__(self) -> None:
        metrics = dict(self.metrics)
        for name in ("prompt_tokens", "completion_tokens", "cache_hit_tokens"):
            value = getattr(self, name)
            if value is not None:
                _require_non_negative_int(value, path=name)
        typed_values: dict[str, JsonValue] = {
            "queue_duration_s": self.queue_duration_s,
            "request_latency_s": self.request_latency_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "rate_limited": self.rate_limited,
        }
        for name, value in typed_values.items():
            if value is not None and (name != "rate_limited" or value):
                existing = metrics.get(name)
                if existing is not None and existing != value:
                    raise ValueError(f"{name} was provided twice with different values")
                metrics[name] = value
        object.__setattr__(self, "metrics", metrics)
        super().__post_init__()

    @classmethod
    def from_fireworks_headers(
        cls,
        headers: Mapping[str, object],
        *,
        request_latency_s: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        http_status_code: int | None = None,
        transport_error: bool = False,
    ) -> "RequestSamplingFeedback":
        """Normalize Fireworks response headers without importing its SDK.

        Fireworks accepts both bare and ``fireworks-``-prefixed metric header
        names. Durations are seconds. Unknown headers are intentionally not
        copied into the generic metrics bag because HTTP headers often contain
        credentials or unrelated transport metadata.
        """

        normalized = {str(key).lower(): value for key, value in headers.items()}

        def value(name: str) -> object | None:
            return normalized.get(name, normalized.get(f"fireworks-{name}"))

        def optional_float(name: str) -> float | None:
            raw = value(name)
            if raw in (None, ""):
                return None
            try:
                return float(str(raw))
            except (TypeError, ValueError):
                return None

        def optional_int(name: str) -> int | None:
            raw = value(name)
            if raw in (None, ""):
                return None
            try:
                return int(str(raw))
            except (TypeError, ValueError):
                return None

        queue_duration_s = optional_float("prefill-queue-duration")
        prompt_tokens = optional_int("prompt-tokens") or prompt_tokens
        cache_hit_tokens = optional_int("cached-prompt-tokens")
        metrics: dict[str, JsonValue] = {}
        for header, metric in (
            ("generation-queue-duration", "fireworks.generation_queue_duration_s"),
            ("server-time-to-first-token", "fireworks.server_ttft_s"),
            ("server-processing-time", "fireworks.server_processing_time_s"),
            ("tokenizer-queue-duration", "fireworks.tokenizer_queue_duration_s"),
            ("tokenizer-duration", "fireworks.tokenizer_duration_s"),
            ("prefill-duration", "fireworks.prefill_duration_s"),
            ("generation-duration", "fireworks.generation_duration_s"),
        ):
            parsed = optional_float(header)
            if parsed is not None:
                metrics[metric] = parsed
        concurrent = optional_int("num-concurrent-requests")
        if concurrent is not None:
            metrics["fireworks.num_concurrent_requests"] = concurrent
        if transport_error:
            metrics["fireworks.transport_error"] = True

        return cls(
            metrics=metrics,
            source="fireworks",
            queue_duration_s=queue_duration_s,
            request_latency_s=request_latency_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_hit_tokens=cache_hit_tokens,
            rate_limited=http_status_code in (429, 503) or transport_error,
        )


@dataclass(frozen=True)
class VLLMEngineSamplingFeedback(SamplingFeedback):
    """Typed coherent snapshot from a managed vLLM deployment."""

    engine_loads: tuple[VLLMEngineLoad, ...] = ()

    def __post_init__(self) -> None:
        if not self.engine_loads:
            raise ValueError("engine_loads must not be empty")
        metrics = dict(self.metrics)
        if ENGINE_LOADS_METRIC in metrics:
            raise ValueError(f"{ENGINE_LOADS_METRIC} is derived from engine_loads and must not be supplied in metrics")
        metrics[ENGINE_LOADS_METRIC] = [
            {
                "engine_id": load.engine_id,
                "role": load.role,
                "kv_capacity_tokens": load.kv_capacity_tokens,
                "max_model_len": load.max_model_len,
                "kv_usage": load.kv_usage,
                "running": load.running,
                "waiting": load.waiting,
                "waiting_capacity": load.waiting_capacity,
                "preemptions_delta": load.preemptions_delta,
            }
            for load in self.engine_loads
        ]
        object.__setattr__(self, "metrics", metrics)
        super().__post_init__()


@dataclass(frozen=True)
class SamplingCompletion:
    """Lifecycle event for completed sampling admission units.

    One admission unit is one active sampled sequence. Native SkyRL admits each
    prompt request with weight one. A durable request that produces multiple
    sequences, such as Tinker's ``num_samples``, reports that weight through
    ``admission_units`` so policy turnover uses the same unit as admission.
    """

    tokens: int
    duration_s: float
    admission_units: int = 1
    metrics: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.tokens, bool) or not isinstance(self.tokens, int):
            raise TypeError(f"tokens must be an integer, got {type(self.tokens).__name__}")
        if self.tokens < 0:
            raise ValueError(f"tokens must be non-negative, got {self.tokens}")
        if isinstance(self.duration_s, bool) or not isinstance(self.duration_s, (int, float)):
            raise TypeError(f"duration_s must be numeric, got {type(self.duration_s).__name__}")
        if not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError(f"duration_s must be a finite non-negative value, got {self.duration_s}")
        if isinstance(self.admission_units, bool) or not isinstance(self.admission_units, int):
            raise TypeError(f"admission_units must be an integer, got {type(self.admission_units).__name__}")
        if self.admission_units < 1:
            raise ValueError(f"admission_units must be at least 1, got {self.admission_units}")
        metrics = dict(self.metrics)
        _validate_json_value(metrics, path="metrics")
        object.__setattr__(self, "metrics", metrics)


@dataclass(frozen=True)
class ConcurrencyContext:
    """Admission state captured by the owner when a policy hook runs.

    ``current_limit`` and ``in_flight`` use sampled-sequence admission units,
    not API rows, trajectory groups, or generator worker counts.
    """

    current_limit: int
    in_flight: int
    observed_at_s: float

    def __post_init__(self) -> None:
        if isinstance(self.current_limit, bool) or not isinstance(self.current_limit, int):
            raise TypeError(f"current_limit must be an integer, got {type(self.current_limit).__name__}")
        if self.current_limit < 1:
            raise ValueError(f"current_limit must be at least 1, got {self.current_limit}")
        if isinstance(self.in_flight, bool) or not isinstance(self.in_flight, int):
            raise TypeError(f"in_flight must be an integer, got {type(self.in_flight).__name__}")
        if self.in_flight < 0:
            raise ValueError(f"in_flight must be non-negative, got {self.in_flight}")
        if isinstance(self.observed_at_s, bool) or not isinstance(self.observed_at_s, (int, float)):
            raise TypeError(f"observed_at_s must be numeric, got {type(self.observed_at_s).__name__}")
        if not math.isfinite(self.observed_at_s) or self.observed_at_s < 0:
            raise ValueError(
                f"observed_at_s must be a finite non-negative monotonic timestamp, got {self.observed_at_s}"
            )


@dataclass(frozen=True)
class ConcurrencyDecision:
    """Effects requested by a policy and applied by the admission owner.

    ``desired_limit`` uses sampled-sequence admission units. ``shed_count`` is
    a count of owner-managed cancellation candidates—for native fully async
    training, individual trajectory tasks—rather than a durable-row count.
    """

    desired_limit: int
    shed_count: int = 0
    reason: str = "unspecified"

    def __post_init__(self) -> None:
        if isinstance(self.desired_limit, bool) or not isinstance(self.desired_limit, int):
            raise TypeError(f"desired_limit must be an integer, got {type(self.desired_limit).__name__}")
        if self.desired_limit < 1:
            raise ValueError(f"desired_limit must be at least 1, got {self.desired_limit}")
        if isinstance(self.shed_count, bool) or not isinstance(self.shed_count, int):
            raise TypeError(f"shed_count must be an integer, got {type(self.shed_count).__name__}")
        if self.shed_count < 0:
            raise ValueError(f"shed_count must be non-negative, got {self.shed_count}")
        if not self.reason:
            raise ValueError("reason must not be empty")


@runtime_checkable
class ConcurrencyPolicy(Protocol):
    """Stateful strategy shared by sampling admission owners.

    The policy requests effects but never owns admission permits, durable rows,
    or trajectory tasks. The component that already owns pending work supplies
    a context snapshot and applies any returned decision.
    """

    def on_feedback(self, feedback: SamplingFeedback, context: ConcurrencyContext) -> ConcurrencyDecision | None: ...

    def on_completion(
        self, completion: SamplingCompletion, context: ConcurrencyContext
    ) -> ConcurrencyDecision | None: ...


class ResizableConcurrencyLimiter:
    """An async concurrency limiter whose limit can change at runtime.

    Lowering the limit never cancels work that already holds a slot. New work
    waits until the number of holders falls below the new limit. Owners that
    support active shedding apply ``ConcurrencyDecision.shed_count`` separately.
    """

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        self._limit = limit
        self._in_flight = 0
        self._waiting = 0
        self._condition = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def waiting(self) -> int:
        return self._waiting

    async def acquire(self, units: int = 1) -> None:
        if units < 1:
            raise ValueError(f"units must be at least 1, got {units}")
        async with self._condition:
            self._waiting += 1
            try:
                await self._condition.wait_for(
                    lambda: self._in_flight + units <= self._limit or (units > self._limit and self._in_flight == 0)
                )
                self._in_flight += units
            finally:
                self._waiting -= 1

    async def release(self, units: int = 1) -> None:
        if units < 1:
            raise ValueError(f"units must be at least 1, got {units}")
        async with self._condition:
            if self._in_flight < units:
                raise RuntimeError(f"cannot release {units} concurrency units when only {self._in_flight} are held")
            self._in_flight -= units
            self._condition.notify_all()

    async def resize(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        async with self._condition:
            self._limit = limit
            self._condition.notify_all()

    @asynccontextmanager
    async def slot(self, units: int = 1) -> AsyncIterator[None]:
        """Hold sampled-sequence admission units for one model request."""

        await self.acquire(units)
        try:
            yield
        finally:
            await self.release(units)


class SamplingConcurrencyController:
    """Co-locate one concurrency policy with the limiter it controls.

    Native ``SkyRLGymGenerator`` owns one controller and shares it with the
    ``RemoteInferenceClient`` request path. The client calls :meth:`slot` but
    does not own a second adaptive limit. Policy decisions resize this
    controller's limiter atomically with respect to other policy callbacks.

    ``shed_count`` is deliberately returned to the generator: the controller
    owns sampling permits, while the generator owns cancellable trajectories.
    Durable admission owners such as ``TinkerEngine`` reuse the policy
    contracts directly and apply decisions to their database transition rather
    than using this process-local limiter.
    """

    def __init__(
        self,
        *,
        policy: ConcurrencyPolicy,
        initial_limit: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._policy = policy
        self._limiter = ResizableConcurrencyLimiter(initial_limit)
        self._clock = clock
        self._policy_lock = asyncio.Lock()

    @property
    def policy(self) -> ConcurrencyPolicy:
        return self._policy

    @property
    def current_limit(self) -> int:
        return self._limiter.limit

    @property
    def in_flight(self) -> int:
        return self._limiter.in_flight

    @property
    def waiting(self) -> int:
        return self._limiter.waiting

    def context(self) -> ConcurrencyContext:
        """Capture the limiter state supplied to the next policy hook."""

        return ConcurrencyContext(
            current_limit=self.current_limit,
            in_flight=self.in_flight,
            observed_at_s=self._clock(),
        )

    @asynccontextmanager
    async def slot(self, units: int = 1) -> AsyncIterator[None]:
        """Hold model-request units in the shared sampling window."""

        async with self._limiter.slot(units):
            yield

    async def on_feedback(self, feedback: SamplingFeedback) -> ConcurrencyDecision | None:
        """Run the policy feedback hook and apply its requested limit."""

        async with self._policy_lock:
            decision = self._policy.on_feedback(feedback, self.context())
            return await self._apply(decision)

    async def on_completion(self, completion: SamplingCompletion) -> ConcurrencyDecision | None:
        """Run the policy completion hook and apply its requested limit."""

        async with self._policy_lock:
            decision = self._policy.on_completion(completion, self.context())
            return await self._apply(decision)

    async def _apply(self, decision: ConcurrencyDecision | None) -> ConcurrencyDecision | None:
        if decision is not None:
            await self._limiter.resize(decision.desired_limit)
        return decision


class FixedConcurrencyPolicy:
    """Safe fallback for backends with no trustworthy pressure signal."""

    def on_feedback(self, feedback: SamplingFeedback, context: ConcurrencyContext) -> ConcurrencyDecision | None:
        return None

    def on_completion(self, completion: SamplingCompletion, context: ConcurrencyContext) -> ConcurrencyDecision | None:
        return None


class EngineLoadConcurrencyPolicy:
    """V1 adaptive policy for coherent per-engine vLLM load snapshots.

    The policy grows by pipeline turnover while the most recent scrape is
    clear, soft-trims when KV usage loses headroom, and cuts on preemptions or
    a persistent capacity queue.  It is intentionally a state machine only:
    the controller applies the requested limit and a trajectory owner may
    later consume ``shed_count``.  Native V1 keeps active shedding disabled,
    so every downward resize drains naturally.

    Thresholds are conservative implementation constants rather than user
    hyperparameters.  The public tuning surface is the initial/min/max window;
    exposing every threshold before there is operating data would merely move
    manual concurrency tuning into a larger configuration space.
    """

    TURNOVER_GROWTH = 1.25
    BINDING_FRACTION = 0.9
    KV_USAGE_GROW = 0.6
    KV_USAGE_SOFT_CAP = 0.8
    KV_USAGE_HARD_CAP = 0.9
    KV_USAGE_TARGET = 0.7
    KV_TRIM_COOLDOWN_POLLS = 6
    QUEUE_RATIO = 0.5
    QUEUE_PERSISTENCE_POLLS = 6
    QUEUE_CUT_FRACTION = 0.9
    PREEMPTION_CUT_FRACTION = 0.8
    ESCALATED_CUT_FRACTION = 0.5
    ESCALATION_GRACE_POLLS = 6
    GROWTH_GATE_TTL_S = 15.0

    def __init__(
        self,
        *,
        min_limit: int = 1,
        max_limit: int,
        enable_active_shedding: bool = False,
    ) -> None:
        if min_limit < 1:
            raise ValueError(f"min_limit must be at least 1, got {min_limit}")
        if max_limit < min_limit:
            raise ValueError(f"max_limit must be at least min_limit ({min_limit}), got {max_limit}")
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.enable_active_shedding = enable_active_shedding

        self._cap: float | None = None
        self._turnover = 0.0
        self._can_grow = False
        self._can_grow_until_s = 0.0
        self._previous_waiting: dict[str, int] = {}
        self._queue_overload_polls = 0
        self._trim_cooldown_polls = 0
        self._draining = False
        self._escalated = False
        self._escalation_grace_polls = 0

    def on_feedback(self, feedback: SamplingFeedback, context: ConcurrencyContext) -> ConcurrencyDecision | None:
        """Classify one coherent scrape and request a resize when needed."""

        if not isinstance(feedback, VLLMEngineSamplingFeedback):
            return None

        loads = tuple(load for load in feedback.engine_loads if load.role != "prefill")
        if not loads:
            return None
        self._sync_cap(context)

        max_usage = max(load.kv_usage for load in loads)
        total_running = sum(load.running for load in loads)
        total_capacity_waiting = sum(
            load.waiting_capacity if load.waiting_capacity is not None else load.waiting for load in loads
        )
        preempted = any(load.preemptions_delta > 0 for load in loads)

        if total_running > 0 and total_capacity_waiting > self.QUEUE_RATIO * total_running:
            self._queue_overload_polls += 1
        else:
            self._queue_overload_polls = 0
        queue_overload = self._queue_overload_polls >= self.QUEUE_PERSISTENCE_POLLS

        # A queue observed in two successive polls is pressure even before it
        # reaches the hard, ratio-based cut.  It closes the growth gate without
        # overreacting to a single turn-completion burst.
        repeated_waiting = any(load.waiting > 0 and self._previous_waiting.get(load.engine_id, 0) > 0 for load in loads)
        self._previous_waiting = {load.engine_id: load.waiting for load in loads}

        if self._draining and context.in_flight <= context.current_limit and not preempted and not queue_overload:
            self._draining = False
            self._escalation_grace_polls = self.ESCALATION_GRACE_POLLS
        if not self._draining and self._escalated:
            self._escalation_grace_polls -= 1
            if self._escalation_grace_polls <= 0:
                self._escalated = False

        self._trim_cooldown_polls = max(0, self._trim_cooldown_polls - 1)
        self._can_grow = (
            max_usage <= self.KV_USAGE_GROW
            and total_capacity_waiting == 0
            and not repeated_waiting
            and not preempted
            and not self._draining
        )
        self._can_grow_until_s = context.observed_at_s + self.GROWTH_GATE_TTL_S

        if self._draining:
            return None

        if preempted or queue_overload:
            cut_fraction = (
                self.ESCALATED_CUT_FRACTION
                if self._escalated
                else (self.QUEUE_CUT_FRACTION if queue_overload else self.PREEMPTION_CUT_FRACTION)
            )
            target = self._clamp(math.floor(context.in_flight * cut_fraction))
            reason = "engine_queue_overload" if queue_overload else "engine_preemptions"
            self._queue_overload_polls = 0
            self._draining = True
            self._escalated = True
            return self._resize_down(target, context, reason=reason, hard=True)

        if max_usage > self.KV_USAGE_SOFT_CAP and context.in_flight > 0 and self._trim_cooldown_polls == 0:
            target = self._clamp(math.floor(context.in_flight * self.KV_USAGE_TARGET / max_usage))
            hard = max_usage > self.KV_USAGE_HARD_CAP
            self._trim_cooldown_polls = self.KV_TRIM_COOLDOWN_POLLS
            return self._resize_down(
                target,
                context,
                reason="kv_hard_trim" if hard else "kv_soft_trim",
                hard=hard,
            )

        return None

    def on_completion(self, completion: SamplingCompletion, context: ConcurrencyContext) -> ConcurrencyDecision | None:
        """Pace multiplicative growth by completed admission-unit turnover."""

        self._sync_cap(context)
        completed_in_flight = context.in_flight + completion.admission_units
        fraction = completion.admission_units / max(completed_in_flight, completion.admission_units)
        self._turnover += fraction
        if completion.tokens <= 0:
            return None
        if not (
            self._can_grow
            and context.observed_at_s < self._can_grow_until_s
            and completed_in_flight >= self.BINDING_FRACTION * context.current_limit
        ):
            return None

        assert self._cap is not None
        self._cap = self._clamp_float(self._cap * self.TURNOVER_GROWTH**fraction)
        desired_limit = int(self._cap)
        if desired_limit == context.current_limit:
            return None
        return ConcurrencyDecision(desired_limit=desired_limit, reason="clear_engine_turnover")

    @property
    def turnover(self) -> float:
        """Completed pipeline turnovers, exposed for observability and tests."""

        return self._turnover

    def _sync_cap(self, context: ConcurrencyContext) -> None:
        if self._cap is None:
            self._cap = float(context.current_limit)

    def _resize_down(
        self,
        target: int,
        context: ConcurrencyContext,
        *,
        reason: str,
        hard: bool,
    ) -> ConcurrencyDecision | None:
        target = min(target, context.current_limit)
        self._cap = float(target)
        if target == context.current_limit:
            return None
        shed_count = max(0, context.in_flight - target) if hard and self.enable_active_shedding else 0
        return ConcurrencyDecision(desired_limit=target, shed_count=shed_count, reason=reason)

    def _clamp(self, value: int) -> int:
        return min(self.max_limit, max(self.min_limit, value))

    def _clamp_float(self, value: float) -> float:
        return min(float(self.max_limit), max(float(self.min_limit), value))


@dataclass
class _QueueDelayAccumulator:
    sample_count: int = 0
    queue_duration_sum_s: float = 0.0
    queue_duration_count: int = 0
    rate_limited: bool = False

    def add(self, feedback: SamplingFeedback) -> None:
        self.sample_count += 1
        queue_duration_s = feedback.number("queue_duration_s")
        if queue_duration_s is not None:
            self.queue_duration_sum_s += queue_duration_s
            self.queue_duration_count += 1
        self.rate_limited = self.rate_limited or bool(feedback.boolean("rate_limited"))


class QueueDelayConcurrencyPolicy:
    """AIMD policy for backends that expose queue delay or rate limiting.

    This is intentionally less informed than an engine-load policy. It adapts
    over completed feedback intervals and leaves lifecycle completions unused;
    richer policies can use ``on_completion`` to pace growth by pipeline
    turnover while a recent engine snapshot says growth is safe.
    """

    def __init__(
        self,
        *,
        min_limit: int = 1,
        max_limit: int,
        queue_duration_target_s: float,
        adjustment_interval: int = 32,
        additive_step: int = 1,
        decrease_ratio: float = 0.5,
    ):
        if min_limit < 1:
            raise ValueError(f"min_limit must be at least 1, got {min_limit}")
        if max_limit < min_limit:
            raise ValueError(f"max_limit must be at least min_limit ({min_limit}), got {max_limit}")
        if not math.isfinite(queue_duration_target_s) or queue_duration_target_s < 0:
            raise ValueError(
                f"queue_duration_target_s must be a finite non-negative value, got {queue_duration_target_s}"
            )
        if adjustment_interval < 1:
            raise ValueError(f"adjustment_interval must be at least 1, got {adjustment_interval}")
        if additive_step < 1:
            raise ValueError(f"additive_step must be at least 1, got {additive_step}")
        if not 0 < decrease_ratio < 1:
            raise ValueError(f"decrease_ratio must be between 0 and 1, got {decrease_ratio}")

        self.min_limit = min_limit
        self.max_limit = max_limit
        self.queue_duration_target_s = queue_duration_target_s
        self.adjustment_interval = adjustment_interval
        self.additive_step = additive_step
        self.decrease_ratio = decrease_ratio
        self._feedback = _QueueDelayAccumulator()

    def on_feedback(self, feedback: SamplingFeedback, context: ConcurrencyContext) -> ConcurrencyDecision | None:
        """Record one feedback sample and decide when the interval is full."""

        self._feedback.add(feedback)
        if self._feedback.sample_count < self.adjustment_interval:
            return None
        return self._adjust(context)

    def on_completion(self, completion: SamplingCompletion, context: ConcurrencyContext) -> ConcurrencyDecision | None:
        """Queue-delay AIMD does not use completion-paced growth."""

        return None

    def flush(self, context: ConcurrencyContext) -> ConcurrencyDecision | None:
        """Decide from a partially filled interval, if feedback exists."""

        if self._feedback.sample_count == 0:
            return None
        return self._adjust(context)

    def _adjust(self, context: ConcurrencyContext) -> ConcurrencyDecision:
        feedback = self._feedback
        average_queue_duration_s = _average(feedback.queue_duration_sum_s, feedback.queue_duration_count)

        if feedback.rate_limited:
            reason = "rate_limited"
            desired_limit = math.floor(context.current_limit * self.decrease_ratio)
        elif average_queue_duration_s is None:
            reason = "no_queue_feedback"
            desired_limit = context.current_limit
        elif average_queue_duration_s <= self.queue_duration_target_s:
            reason = "below_queue_target"
            desired_limit = context.current_limit + self.additive_step
        else:
            reason = "above_queue_target"
            desired_limit = math.floor(context.current_limit * self.decrease_ratio)

        desired_limit = min(self.max_limit, max(self.min_limit, desired_limit))
        self._feedback = _QueueDelayAccumulator()
        return ConcurrencyDecision(desired_limit=desired_limit, reason=reason)


def _validate_common_metrics(metrics: Mapping[str, JsonValue]) -> None:
    for name in (
        "queue_duration_s",
        "prompt_tokens",
        "completion_tokens",
        "request_latency_s",
        "cache_hit_tokens",
    ):
        value = metrics.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative value, got {value}")

    rate_limited = metrics.get("rate_limited")
    if rate_limited is not None and not isinstance(rate_limited, bool):
        raise TypeError(f"rate_limited must be a boolean, got {type(rate_limited).__name__}")

    prompt_tokens = metrics.get("prompt_tokens")
    cache_hit_tokens = metrics.get("cache_hit_tokens")
    if (
        isinstance(prompt_tokens, (int, float))
        and not isinstance(prompt_tokens, bool)
        and isinstance(cache_hit_tokens, (int, float))
        and not isinstance(cache_hit_tokens, bool)
        and cache_hit_tokens > prompt_tokens
    ):
        raise ValueError(f"cache_hit_tokens ({cache_hit_tokens}) cannot exceed prompt_tokens ({prompt_tokens})")

    engine_loads = metrics.get(ENGINE_LOADS_METRIC)
    if engine_loads is not None:
        _validate_engine_loads(engine_loads)


def _validate_engine_loads(value: JsonValue) -> None:
    if not isinstance(value, list):
        raise TypeError(f"{ENGINE_LOADS_METRIC} must be a list, got {type(value).__name__}")

    required = {
        "engine_id",
        "role",
        "kv_capacity_tokens",
        "max_model_len",
        "kv_usage",
        "running",
        "waiting",
        "waiting_capacity",
        "preemptions_delta",
    }
    for index, load in enumerate(value):
        if not isinstance(load, dict):
            raise TypeError(f"{ENGINE_LOADS_METRIC}[{index}] must be an object, got {type(load).__name__}")
        missing = required - load.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{ENGINE_LOADS_METRIC}[{index}] is missing: {names}")
        prefix = f"{ENGINE_LOADS_METRIC}[{index}]"
        _require_type(load["engine_id"], str, path=f"{prefix}.engine_id")
        if load["role"] is not None:
            _require_type(load["role"], str, path=f"{prefix}.role")
        for name in ("kv_capacity_tokens", "max_model_len", "waiting_capacity"):
            field_value = load[name]
            if field_value is not None:
                _require_non_negative_int(field_value, path=f"{prefix}.{name}")
        for name in ("running", "waiting", "preemptions_delta"):
            _require_non_negative_int(load[name], path=f"{prefix}.{name}")
        kv_usage = load["kv_usage"]
        if isinstance(kv_usage, bool) or not isinstance(kv_usage, (int, float)):
            raise TypeError(f"{prefix}.kv_usage must be numeric, got {type(kv_usage).__name__}")
        if not math.isfinite(kv_usage) or kv_usage < 0:
            raise ValueError(f"{prefix}.kv_usage must be a finite non-negative value, got {kv_usage}")


def _require_type(value: object, expected: type, *, path: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{path} must be {expected.__name__}, got {type(value).__name__}")


def _require_non_negative_int(value: object, *, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{path} must be non-negative, got {value}")


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} object keys must be strings")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise TypeError(f"{path} must contain only JSON-compatible values, got {type(value).__name__}")


def _average(total: float, count: int) -> float | None:
    return total / count if count else None
