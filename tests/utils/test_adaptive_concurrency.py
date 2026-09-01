import asyncio

import pytest

from skyrl.utils.adaptive_concurrency import (
    ConcurrencyContext,
    ConcurrencyDecision,
    ConcurrencyPolicy,
    FixedConcurrencyPolicy,
    QueueDelayConcurrencyPolicy,
    RequestSamplingFeedback,
    ResizableConcurrencyLimiter,
    SamplingCompletion,
    SamplingConcurrencyController,
    SamplingFeedback,
    VLLMEngineLoad,
    VLLMEngineSamplingFeedback,
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


@pytest.mark.asyncio
async def test_weighted_slot_accounts_in_sampled_sequence_units():
    limiter = ResizableConcurrencyLimiter(4)

    async with limiter.slot(units=3):
        assert limiter.in_flight == 3
        waiter = asyncio.create_task(limiter.acquire(units=2))
        await asyncio.sleep(0)
        assert not waiter.done()

    await asyncio.wait_for(waiter, timeout=1)
    assert limiter.in_flight == 2
    await limiter.release(units=2)


@pytest.mark.asyncio
async def test_controller_co_locates_policy_and_request_limiter():
    policy = QueueDelayConcurrencyPolicy(
        max_limit=8,
        queue_duration_target_s=0.2,
        adjustment_interval=1,
    )
    controller = SamplingConcurrencyController(policy=policy, initial_limit=2, clock=lambda: 456.0)

    async with controller.slot():
        assert controller.in_flight == 1
        assert controller.context() == ConcurrencyContext(current_limit=2, in_flight=1, observed_at_s=456.0)
        decision = await controller.on_feedback(SamplingFeedback({"queue_duration_s": 0.1}))

    assert controller.policy is policy
    assert decision == ConcurrencyDecision(desired_limit=3, reason="below_queue_target")
    assert controller.current_limit == 3
    assert controller.in_flight == 0


@pytest.mark.asyncio
async def test_controller_returns_shedding_to_trajectory_owner():
    class SheddingPolicy:
        def on_feedback(self, feedback, policy_context):
            return ConcurrencyDecision(desired_limit=2, shed_count=3, reason="hard_overload")

        def on_completion(self, completion, policy_context):
            return None

    controller = SamplingConcurrencyController(policy=SheddingPolicy(), initial_limit=8)

    decision = await controller.on_feedback(SamplingFeedback())

    assert decision == ConcurrencyDecision(desired_limit=2, shed_count=3, reason="hard_overload")
    assert controller.current_limit == 2


@pytest.mark.asyncio
async def test_controller_forwards_weighted_completion_to_policy():
    class CompletionPolicy:
        completion = None

        def on_feedback(self, feedback, policy_context):
            return None

        def on_completion(self, completion, policy_context):
            self.completion = completion
            return ConcurrencyDecision(
                desired_limit=policy_context.current_limit + completion.admission_units,
                reason="completion_growth",
            )

    policy = CompletionPolicy()
    controller = SamplingConcurrencyController(policy=policy, initial_limit=4)
    completion = SamplingCompletion(tokens=512, duration_s=2.5, admission_units=3)

    decision = await controller.on_completion(completion)

    assert policy.completion is completion
    assert decision == ConcurrencyDecision(desired_limit=7, reason="completion_growth")
    assert controller.current_limit == 7


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


def test_fireworks_headers_become_typed_request_feedback():
    feedback = RequestSamplingFeedback.from_fireworks_headers(
        {
            "fireworks-prefill-queue-duration": "0.25",
            "cached-prompt-tokens": "80",
            "prompt-tokens": "100",
            "generation-queue-duration": "0.05",
            "num-concurrent-requests": "12",
        },
        request_latency_s=0.8,
        completion_tokens=20,
        http_status_code=200,
    )

    assert feedback.source == "fireworks"
    assert feedback.queue_duration_s == 0.25
    assert feedback.cache_hit_tokens == 80
    assert feedback.prompt_tokens == 100
    assert feedback.completion_tokens == 20
    assert feedback.metrics["fireworks.generation_queue_duration_s"] == 0.05
    assert feedback.metrics["fireworks.num_concurrent_requests"] == 12


def test_fireworks_serverless_overload_becomes_rate_limit_feedback():
    feedback = RequestSamplingFeedback.from_fireworks_headers({}, http_status_code=503)

    assert feedback.queue_duration_s is None
    assert feedback.rate_limited is True
    assert feedback.boolean("rate_limited") is True

    transport_feedback = RequestSamplingFeedback.from_fireworks_headers({}, transport_error=True)
    assert transport_feedback.rate_limited is True
    assert transport_feedback.metrics["fireworks.transport_error"] is True


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


def test_completion_reports_weighted_sampled_sequence_units():
    completion = SamplingCompletion(tokens=512, duration_s=2.5, admission_units=4)

    assert completion.admission_units == 4


def test_queue_delay_policy_leaves_completion_growth_to_richer_policies():
    policy = QueueDelayConcurrencyPolicy(max_limit=8, queue_duration_target_s=0.2)

    assert policy.on_completion(SamplingCompletion(tokens=128, duration_s=1.25), context(current_limit=4)) is None


def test_feedback_accepts_a_coherent_engine_load_snapshot():
    loads = (
        VLLMEngineLoad(
            engine_id="decode-0",
            role="decode",
            kv_capacity_tokens=131_072,
            max_model_len=32_768,
            kv_usage=0.72,
            running=12,
            waiting=2,
            waiting_capacity=1,
            preemptions_delta=0,
        ),
        VLLMEngineLoad(
            engine_id="prefill-0",
            role="prefill",
            kv_capacity_tokens=None,
            max_model_len=32_768,
            kv_usage=0.1,
            running=4,
            waiting=0,
            waiting_capacity=None,
            preemptions_delta=0,
        ),
    )

    feedback = VLLMEngineSamplingFeedback(metrics={"backend.router.requests": 16}, engine_loads=loads)

    assert feedback.engine_loads == loads
    assert feedback.metrics["engine.loads"][0]["engine_id"] == "decode-0"


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
    with pytest.raises(ValueError, match="admission_units"):
        SamplingCompletion(tokens=1, duration_s=1, admission_units=0)
    with pytest.raises(TypeError, match="admission_units"):
        SamplingCompletion(tokens=1, duration_s=1, admission_units=True)
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
