"""Backend-neutral contracts for adaptive asynchronous sampling admission."""

import asyncio
import math
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, TypedDict, runtime_checkable

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

ENGINE_LOADS_METRIC = "engine.loads"


class EngineLoadMetrics(TypedDict):
    """Standard shape for one entry in an ``engine.loads`` snapshot.

    Backends that cannot populate this complete shape should omit
    ``engine.loads`` and report only the metrics they can observe.
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


@dataclass(frozen=True)
class SamplingFeedback:
    """One coherent snapshot of backend pressure signals.

    ``metrics`` is deliberately extensible: common names and units are stable,
    missing keys are expected, and specialized backends may publish namespaced
    additions. ``engine.loads`` contains a list of :class:`EngineLoadMetrics`
    dictionaries from the same metrics poll when per-engine visibility exists.
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

    async def acquire(self) -> None:
        async with self._condition:
            self._waiting += 1
            try:
                await self._condition.wait_for(lambda: self._in_flight < self._limit)
                self._in_flight += 1
            finally:
                self._waiting -= 1

    async def release(self) -> None:
        async with self._condition:
            if self._in_flight == 0:
                raise RuntimeError("cannot release a concurrency slot that is not held")
            self._in_flight -= 1
            self._condition.notify_all()

    async def resize(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        async with self._condition:
            self._limit = limit
            self._condition.notify_all()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one sampled-sequence admission unit for one model request."""

        await self.acquire()
        try:
            yield
        finally:
            await self.release()


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
    async def slot(self) -> AsyncIterator[None]:
        """Hold one native model-request unit in the shared sampling window."""

        async with self._limiter.slot():
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

    required = EngineLoadMetrics.__required_keys__
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
