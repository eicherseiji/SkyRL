import asyncio

import pytest

from skyrl.utils.adaptive_concurrency import (
    ConcurrencyContext,
    ConcurrencyDecision,
    ConcurrencyPolicy,
    FixedConcurrencyPolicy,
    QueueDelayConcurrencyPolicy,
    ResizableConcurrencyLimiter,
    SamplingCompletion,
    SamplingFeedback,
)


def context(*, current_limit: int, in_flight: int | None = None) -> ConcurrencyContext:
    return ConcurrencyContext(
        current_limit=current_limit,
        in_flight=current_limit if in_flight is None else in_flight,
        observed_at_s=123.0,
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


def test_queue_delay_policy_increases_limit_after_feedback_interval():
    policy = QueueDelayConcurrencyPolicy(
        max_limit=8,
        queue_duration_target_s=0.2,
        adjustment_interval=2,
    )

    assert (
        policy.on_feedback(
            SamplingFeedback(
                {
                    "queue_duration_s": 0.05,
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "request_latency_s": 0.5,
                    "cache_hit_tokens": 40,
                }
            ),
            context(current_limit=4),
        )
        is None
    )
    decision = policy.on_feedback(
        SamplingFeedback(
            {
                "queue_duration_s": 0.15,
                "prompt_tokens": 300,
                "completion_tokens": 60,
                "request_latency_s": 1.5,
                "cache_hit_tokens": 160,
            }
        ),
        context(current_limit=4),
    )

    assert decision == ConcurrencyDecision(desired_limit=5, reason="below_queue_target")


def test_queue_delay_policy_decreases_limit_on_high_queue_duration():
    policy = QueueDelayConcurrencyPolicy(
        min_limit=2,
        max_limit=16,
        queue_duration_target_s=0.2,
        adjustment_interval=1,
        decrease_ratio=0.5,
    )

    decision = policy.on_feedback(SamplingFeedback({"queue_duration_s": 0.3}), context(current_limit=9))

    assert decision == ConcurrencyDecision(desired_limit=4, reason="above_queue_target")


def test_rate_limit_takes_precedence_over_queue_duration():
    policy = QueueDelayConcurrencyPolicy(
        max_limit=16,
        queue_duration_target_s=0.2,
        adjustment_interval=2,
    )

    policy.on_feedback(
        SamplingFeedback({"queue_duration_s": 0.01, "rate_limited": True}),
        context(current_limit=8),
    )
    decision = policy.on_feedback(SamplingFeedback({"queue_duration_s": 0.01}), context(current_limit=8))

    assert decision == ConcurrencyDecision(desired_limit=4, reason="rate_limited")


def test_feedback_without_pressure_does_not_guess_backend_capacity():
    policy = QueueDelayConcurrencyPolicy(
        max_limit=16,
        queue_duration_target_s=0.2,
        adjustment_interval=1,
    )

    decision = policy.on_feedback(
        SamplingFeedback({"prompt_tokens": 1_000, "completion_tokens": 2_000, "request_latency_s": 30}),
        context(current_limit=4),
    )

    assert decision == ConcurrencyDecision(desired_limit=4, reason="no_queue_feedback")


def test_flush_uses_the_latest_owner_context():
    policy = QueueDelayConcurrencyPolicy(
        max_limit=8,
        queue_duration_target_s=0.2,
        adjustment_interval=10,
    )
    policy.on_feedback(SamplingFeedback({"queue_duration_s": 0.1}), context(current_limit=2))

    decision = policy.flush(context(current_limit=5))

    assert decision == ConcurrencyDecision(desired_limit=6, reason="below_queue_target")
    assert policy.flush(context(current_limit=6)) is None


def test_policy_protocol_supports_feedback_and_completion_hooks():
    policy = FixedConcurrencyPolicy()

    assert isinstance(policy, ConcurrencyPolicy)
    assert policy.on_feedback(SamplingFeedback(), context(current_limit=4)) is None
    assert policy.on_completion(SamplingCompletion(tokens=128, duration_s=1.25), context(current_limit=4)) is None


def test_queue_delay_policy_leaves_completion_growth_to_richer_policies():
    policy = QueueDelayConcurrencyPolicy(max_limit=8, queue_duration_target_s=0.2)

    assert policy.on_completion(SamplingCompletion(tokens=128, duration_s=1.25), context(current_limit=4)) is None


def test_feedback_accepts_a_coherent_engine_load_snapshot():
    loads = [
        {
            "engine_id": "decode-0",
            "role": "decode",
            "kv_capacity_tokens": 131_072,
            "max_model_len": 32_768,
            "kv_usage": 0.72,
            "running": 12,
            "waiting": 2,
            "waiting_capacity": 1,
            "preemptions_delta": 0,
        },
        {
            "engine_id": "prefill-0",
            "role": "prefill",
            "kv_capacity_tokens": None,
            "max_model_len": 32_768,
            "kv_usage": 0.1,
            "running": 4,
            "waiting": 0,
            "waiting_capacity": None,
            "preemptions_delta": 0,
        },
    ]

    feedback = SamplingFeedback({"engine.loads": loads, "backend.router.requests": 16})

    assert feedback.metrics["engine.loads"] == loads


@pytest.mark.parametrize(
    ("metrics", "error"),
    [
        ({"queue_duration_s": -0.1}, ValueError),
        ({"queue_duration_s": float("inf")}, ValueError),
        ({"request_latency_s": float("nan")}, ValueError),
        ({"prompt_tokens": -1}, ValueError),
        ({"completion_tokens": -1}, ValueError),
        ({"cache_hit_tokens": -1}, ValueError),
        ({"rate_limited": 1}, TypeError),
        ({"backend.bad": object()}, TypeError),
        ({"engine.loads": {}}, TypeError),
        ({"engine.loads": [{"engine_id": "incomplete"}]}, ValueError),
    ],
)
def test_feedback_rejects_invalid_metrics(metrics, error):
    with pytest.raises(error):
        SamplingFeedback(metrics)


def test_feedback_rejects_more_cache_hits_than_prompt_tokens():
    with pytest.raises(ValueError, match="cache_hit_tokens"):
        SamplingFeedback({"prompt_tokens": 10, "cache_hit_tokens": 11})


@pytest.mark.parametrize(
    "value",
    [
        ConcurrencyContext(current_limit=1, in_flight=0, observed_at_s=0),
        SamplingCompletion(tokens=0, duration_s=0),
        ConcurrencyDecision(desired_limit=1),
    ],
)
def test_contracts_accept_boundary_values(value):
    assert value is not None


def test_contracts_reject_invalid_values():
    with pytest.raises(ValueError, match="current_limit"):
        ConcurrencyContext(current_limit=0, in_flight=0, observed_at_s=1)
    with pytest.raises(ValueError, match="in_flight"):
        ConcurrencyContext(current_limit=1, in_flight=-1, observed_at_s=1)
    with pytest.raises(ValueError, match="observed_at_s"):
        ConcurrencyContext(current_limit=1, in_flight=0, observed_at_s=float("inf"))
    with pytest.raises(ValueError, match="tokens"):
        SamplingCompletion(tokens=-1, duration_s=1)
    with pytest.raises(ValueError, match="duration_s"):
        SamplingCompletion(tokens=1, duration_s=-1)
    with pytest.raises(ValueError, match="shed_count"):
        ConcurrencyDecision(desired_limit=1, shed_count=-1)


def test_queue_delay_policy_validates_configuration():
    with pytest.raises(ValueError, match="max_limit"):
        QueueDelayConcurrencyPolicy(
            min_limit=9,
            max_limit=8,
            queue_duration_target_s=0.2,
        )
    with pytest.raises(ValueError, match="decrease_ratio"):
        QueueDelayConcurrencyPolicy(
            max_limit=8,
            queue_duration_target_s=0.2,
            decrease_ratio=1,
        )
