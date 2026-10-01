import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl.train.sampling_service import SamplingClient, SamplingService
from skyrl.utils.adaptive_concurrency import (
    FixedConcurrencyPolicy,
    SamplingConcurrencyController,
)


@pytest.mark.asyncio
async def test_service_owns_waiting_requests_and_limits_backend_dispatch():
    backend = MagicMock()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    active = 0
    max_active = 0

    async def chat_completion(payload):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if not first_started.is_set():
            first_started.set()
            await release_first.wait()
        active -= 1
        return {"id": payload["json"]["session_id"]}

    backend.chat_completion = chat_completion
    controller = SamplingConcurrencyController(policy=FixedConcurrencyPolicy(), initial_limit=1)
    service = SamplingService(backend, controller=controller)

    first = asyncio.create_task(service.chat_completion({"json": {"session_id": "a"}}, attempt_id="a"))
    await first_started.wait()
    second = asyncio.create_task(service.chat_completion({"json": {"session_id": "b"}}, attempt_id="b"))
    for _ in range(10):
        if controller.waiting == 1:
            break
        await asyncio.sleep(0)

    assert service.in_flight == 1
    assert service.pending == 1
    assert controller.in_flight == 1
    assert controller.waiting == 1
    assert max_active == 1

    release_first.set()
    assert await asyncio.gather(first, second) == [{"id": "a"}, {"id": "b"}]
    assert service.in_flight == 0
    assert service.pending == 0
    assert max_active == 1


@pytest.mark.asyncio
async def test_sampling_client_splits_generate_batch_into_service_requests():
    backend = MagicMock()

    async def generate(input_batch, model=None):
        prompt = input_batch["prompt_token_ids"][0]
        return {
            "responses": [str(prompt[-1])],
            "response_ids": [[prompt[-1]]],
            "stop_reasons": ["stop"],
            "response_logprobs": [[-0.1]],
            "prompt_logprobs": None,
            "rollout_expert_indices": None,
        }

    backend.generate = generate
    backend.model_name = "model"
    service = SamplingService(backend)
    client = SamplingClient(service, backend)

    result = await client.generate(
        {
            "prompt_token_ids": [[1, 2], [3, 4]],
            "prompts": None,
            "sampling_params": {"max_tokens": 1},
            "session_ids": ["attempt-a", "attempt-b"],
            "mm_features": None,
            "cache_salt": None,
        }
    )

    assert result["responses"] == ["2", "4"]
    assert result["response_ids"] == [[2], [4]]
    assert result["response_logprobs"] == [[-0.1], [-0.1]]


@pytest.mark.asyncio
async def test_client_cancellation_cancels_service_owned_backend_work():
    backend = MagicMock()
    backend.chat_completion = AsyncMock(side_effect=asyncio.Event().wait)
    controller = SamplingConcurrencyController(policy=FixedConcurrencyPolicy(), initial_limit=1)
    service = SamplingService(backend, controller=controller)

    request = asyncio.create_task(service.chat_completion({"json": {}}, attempt_id="attempt"))
    await asyncio.sleep(0)
    request.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request
    assert service.pending == 0
    assert service.in_flight == 0
    assert controller.in_flight == 0
