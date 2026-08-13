"""Backend-neutral adaptive admission control for asynchronous sampling."""

import asyncio
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SamplingFeedback:
    """Signals a sampling adapter can report without exposing its generator.

    ``queue_duration_s`` is the congestion signal used for admission control.
    Token and latency fields are retained in adjustment summaries for
    observability and future token-aware scheduling, but do not change the
    concurrency limit by themselves.
    """

    queue_duration_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    request_latency_s: float | None = None
    cache_hit_tokens: int | None = None
    rate_limited: bool = False

    def __post_init__(self) -> None:
        for name in ("queue_duration_s", "request_latency_s"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite non-negative value, got {value}")

        for name in ("prompt_tokens", "completion_tokens", "cache_hit_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        if (
            self.prompt_tokens is not None
            and self.cache_hit_tokens is not None
            and self.cache_hit_tokens > self.prompt_tokens
        ):
            raise ValueError(
                f"cache_hit_tokens ({self.cache_hit_tokens}) cannot exceed prompt_tokens ({self.prompt_tokens})"
            )


@dataclass(frozen=True)
class ConcurrencyAdjustment:
    """One completed control interval and its workload summary."""

    previous_limit: int
    limit: int
    reason: Literal["below_queue_target", "above_queue_target", "rate_limited", "no_queue_feedback"]
    sample_count: int
    average_queue_duration_s: float | None
    average_prompt_tokens: float | None
    average_completion_tokens: float | None
    average_request_latency_s: float | None
    cache_hit_rate: float | None


class ResizableConcurrencyLimiter:
    """An async concurrency limiter whose limit can change at runtime.

    Lowering the limit never cancels work that already holds a slot. New work
    waits until the number of holders falls below the new limit.
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
        """Hold one admission slot for the duration of a sampling request."""

        await self.acquire()
        try:
            yield
        finally:
            await self.release()


@dataclass
class _FeedbackAccumulator:
    sample_count: int = 0
    queue_duration_sum_s: float = 0.0
    queue_duration_count: int = 0
    prompt_tokens_sum: int = 0
    prompt_tokens_count: int = 0
    completion_tokens_sum: int = 0
    completion_tokens_count: int = 0
    request_latency_sum_s: float = 0.0
    request_latency_count: int = 0
    cache_hit_tokens_sum: int = 0
    cache_prompt_tokens_sum: int = 0
    rate_limited: bool = False

    def add(self, feedback: SamplingFeedback) -> None:
        self.sample_count += 1
        if feedback.queue_duration_s is not None:
            self.queue_duration_sum_s += feedback.queue_duration_s
            self.queue_duration_count += 1
        if feedback.prompt_tokens is not None:
            self.prompt_tokens_sum += feedback.prompt_tokens
            self.prompt_tokens_count += 1
        if feedback.completion_tokens is not None:
            self.completion_tokens_sum += feedback.completion_tokens
            self.completion_tokens_count += 1
        if feedback.request_latency_s is not None:
            self.request_latency_sum_s += feedback.request_latency_s
            self.request_latency_count += 1
        if feedback.prompt_tokens is not None and feedback.cache_hit_tokens is not None:
            self.cache_prompt_tokens_sum += feedback.prompt_tokens
            self.cache_hit_tokens_sum += feedback.cache_hit_tokens
        self.rate_limited = self.rate_limited or feedback.rate_limited


class AdaptiveSamplingConcurrencyController:
    """Adjust sampling admission with additive increase/multiplicative decrease.

    Sampling adapters call ``observe`` after each request. Once an adjustment
    interval completes, queue duration below the target adds ``additive_step``
    slots; queue duration above it multiplies the current limit by
    ``decrease_ratio``. A rate-limited response also selects the decrease path.

    If a backend cannot report queue duration, the controller deliberately
    leaves the limit unchanged. End-to-end latency and token counts alone do
    not reveal whether the backend has spare capacity.
    """

    def __init__(
        self,
        *,
        initial_limit: int,
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
        if not min_limit <= initial_limit <= max_limit:
            raise ValueError(f"initial_limit must be in [{min_limit}, {max_limit}], got {initial_limit}")
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
        self.limiter = ResizableConcurrencyLimiter(initial_limit)
        self._feedback = _FeedbackAccumulator()
        self._feedback_lock = asyncio.Lock()

    async def observe(self, feedback: SamplingFeedback) -> ConcurrencyAdjustment | None:
        """Record one completed sample and adjust when the interval is full."""

        async with self._feedback_lock:
            self._feedback.add(feedback)
            if self._feedback.sample_count < self.adjustment_interval:
                return None
            return await self._adjust()

    async def flush(self) -> ConcurrencyAdjustment | None:
        """Adjust from a partially filled interval, if any feedback exists."""

        async with self._feedback_lock:
            if self._feedback.sample_count == 0:
                return None
            return await self._adjust()

    async def _adjust(self) -> ConcurrencyAdjustment:
        feedback = self._feedback
        previous_limit = self.limiter.limit
        average_queue_duration_s = _average(feedback.queue_duration_sum_s, feedback.queue_duration_count)

        if feedback.rate_limited:
            reason = "rate_limited"
            limit = max(self.min_limit, math.floor(previous_limit * self.decrease_ratio))
        elif average_queue_duration_s is None:
            reason = "no_queue_feedback"
            limit = previous_limit
        elif average_queue_duration_s <= self.queue_duration_target_s:
            reason = "below_queue_target"
            limit = min(self.max_limit, previous_limit + self.additive_step)
        else:
            reason = "above_queue_target"
            limit = max(self.min_limit, math.floor(previous_limit * self.decrease_ratio))

        await self.limiter.resize(limit)
        adjustment = ConcurrencyAdjustment(
            previous_limit=previous_limit,
            limit=limit,
            reason=reason,
            sample_count=feedback.sample_count,
            average_queue_duration_s=average_queue_duration_s,
            average_prompt_tokens=_average(feedback.prompt_tokens_sum, feedback.prompt_tokens_count),
            average_completion_tokens=_average(feedback.completion_tokens_sum, feedback.completion_tokens_count),
            average_request_latency_s=_average(feedback.request_latency_sum_s, feedback.request_latency_count),
            cache_hit_rate=(
                feedback.cache_hit_tokens_sum / feedback.cache_prompt_tokens_sum
                if feedback.cache_prompt_tokens_sum
                else None
            ),
        )
        self._feedback = _FeedbackAccumulator()
        return adjustment


def _average(total: float, count: int) -> float | None:
    return total / count if count else None
