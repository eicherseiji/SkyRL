import asyncio

import pytest

from skyrl.utils.adaptive_concurrency import (
    AdaptiveSamplingConcurrencyController,
    ResizableConcurrencyLimiter,
    SamplingFeedback,
)


@pytest.mark.asyncio
async def test_resize_up_admits_waiting_work():
    limiter = ResizableConcurrencyLimiter(2)
    await limiter.acquire()
    await limiter.acquire()

    waiter = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)
    assert limiter.in_flight == 2
    assert limiter.waiting == 1

    await limiter.resize(3)
    await asyncio.wait_for(waiter, timeout=1)
    assert limiter.in_flight == 3
    assert limiter.waiting == 0

    for _ in range(3):
        await limiter.release()


@pytest.mark.asyncio
async def test_resize_down_does_not_cancel_in_flight_work():
    limiter = ResizableConcurrencyLimiter(3)
    for _ in range(3):
        await limiter.acquire()

    await limiter.resize(1)
    waiter = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)

    await limiter.release()
    await limiter.release()
    await asyncio.sleep(0)
    assert not waiter.done()

    await limiter.release()
    await asyncio.wait_for(waiter, timeout=1)
    assert limiter.in_flight == 1
    await limiter.release()


@pytest.mark.asyncio
async def test_slot_releases_after_exception():
    limiter = ResizableConcurrencyLimiter(1)

    with pytest.raises(RuntimeError, match="sampling failed"):
        async with limiter.slot():
            raise RuntimeError("sampling failed")

    assert limiter.in_flight == 0
    async with limiter.slot():
        assert limiter.in_flight == 1


@pytest.mark.asyncio
async def test_low_queue_duration_additively_increases_limit_and_summarizes_workload():
    controller = AdaptiveSamplingConcurrencyController(
        initial_limit=4,
        max_limit=8,
        queue_duration_target_s=0.2,
        adjustment_interval=2,
    )

    assert (
        await controller.observe(
            SamplingFeedback(
                queue_duration_s=0.05,
                prompt_tokens=100,
                completion_tokens=20,
                request_latency_s=0.5,
                cache_hit_tokens=40,
            )
        )
        is None
    )
    adjustment = await controller.observe(
        SamplingFeedback(
            queue_duration_s=0.15,
            prompt_tokens=300,
            completion_tokens=60,
            request_latency_s=1.5,
            cache_hit_tokens=160,
        )
    )

    assert adjustment is not None
    assert adjustment.previous_limit == 4
    assert adjustment.limit == 5
    assert adjustment.reason == "below_queue_target"
    assert adjustment.sample_count == 2
    assert adjustment.average_queue_duration_s == pytest.approx(0.1)
    assert adjustment.average_prompt_tokens == 200
    assert adjustment.average_completion_tokens == 40
    assert adjustment.average_request_latency_s == 1
    assert adjustment.cache_hit_rate == pytest.approx(0.5)
    assert controller.limiter.limit == 5


@pytest.mark.asyncio
async def test_high_queue_duration_multiplicatively_decreases_limit():
    controller = AdaptiveSamplingConcurrencyController(
        initial_limit=9,
        min_limit=2,
        max_limit=16,
        queue_duration_target_s=0.2,
        adjustment_interval=1,
        decrease_ratio=0.5,
    )

    adjustment = await controller.observe(SamplingFeedback(queue_duration_s=0.3))

    assert adjustment is not None
    assert adjustment.reason == "above_queue_target"
    assert adjustment.limit == 4


@pytest.mark.asyncio
async def test_rate_limit_takes_precedence_over_queue_duration():
    controller = AdaptiveSamplingConcurrencyController(
        initial_limit=8,
        max_limit=16,
        queue_duration_target_s=0.2,
        adjustment_interval=2,
    )

    await controller.observe(SamplingFeedback(queue_duration_s=0.01, rate_limited=True))
    adjustment = await controller.observe(SamplingFeedback(queue_duration_s=0.01))

    assert adjustment is not None
    assert adjustment.reason == "rate_limited"
    assert adjustment.limit == 4


@pytest.mark.asyncio
async def test_token_and_latency_feedback_do_not_guess_backend_capacity():
    controller = AdaptiveSamplingConcurrencyController(
        initial_limit=4,
        max_limit=16,
        queue_duration_target_s=0.2,
        adjustment_interval=1,
    )

    adjustment = await controller.observe(
        SamplingFeedback(prompt_tokens=1_000, completion_tokens=2_000, request_latency_s=30)
    )

    assert adjustment is not None
    assert adjustment.reason == "no_queue_feedback"
    assert adjustment.limit == 4


@pytest.mark.asyncio
async def test_flush_adjusts_a_partial_interval_and_then_returns_none():
    controller = AdaptiveSamplingConcurrencyController(
        initial_limit=2,
        max_limit=4,
        queue_duration_target_s=0.2,
        adjustment_interval=10,
    )
    await controller.observe(SamplingFeedback(queue_duration_s=0.1))

    adjustment = await controller.flush()

    assert adjustment is not None
    assert adjustment.sample_count == 1
    assert adjustment.limit == 3
    assert await controller.flush() is None


@pytest.mark.parametrize(
    "feedback",
    [
        SamplingFeedback(queue_duration_s=None),
        SamplingFeedback(prompt_tokens=0),
        SamplingFeedback(completion_tokens=0),
    ],
)
def test_feedback_accepts_missing_and_zero_values(feedback):
    assert feedback is not None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("queue_duration_s", -0.1),
        ("queue_duration_s", float("inf")),
        ("request_latency_s", float("nan")),
        ("prompt_tokens", -1),
        ("completion_tokens", -1),
        ("cache_hit_tokens", -1),
    ],
)
def test_feedback_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        SamplingFeedback(**{field: value})


def test_feedback_rejects_more_cache_hits_than_prompt_tokens():
    with pytest.raises(ValueError, match="cache_hit_tokens"):
        SamplingFeedback(prompt_tokens=10, cache_hit_tokens=11)


def test_controller_validates_configuration():
    with pytest.raises(ValueError, match="initial_limit"):
        AdaptiveSamplingConcurrencyController(
            initial_limit=9,
            max_limit=8,
            queue_duration_target_s=0.2,
        )
    with pytest.raises(ValueError, match="decrease_ratio"):
        AdaptiveSamplingConcurrencyController(
            initial_limit=4,
            max_limit=8,
            queue_duration_target_s=0.2,
            decrease_ratio=1,
        )
