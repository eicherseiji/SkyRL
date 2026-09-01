import json

import httpx
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import FutureDB, RequestStatus
from skyrl.tinker.extra.external_inference import ExternalInferenceClient
from skyrl.utils.adaptive_concurrency import (
    FixedConcurrencyPolicy,
    SamplingConcurrencyController,
)


class RecordingPolicy(FixedConcurrencyPolicy):
    def __init__(self):
        self.feedback = []
        self.completions = []

    def on_feedback(self, feedback, context):
        self.feedback.append(feedback)

    def on_completion(self, completion, context):
        self.completions.append(completion)


@pytest.mark.asyncio
async def test_fireworks_forwarding_reports_feedback_and_completes_dispatched_row(monkeypatch):
    db_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
    )
    async with db_engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    request_data = types.SampleInput(
        base_model="test-model",
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]),
        sampling_params=types.SamplingParams(temperature=0.0, max_tokens=4, seed=0),
        num_samples=2,
        checkpoint_id="",
        prompt_logprobs=False,
    )
    async with AsyncSession(db_engine) as session:
        row = FutureDB(
            request_type=types.RequestType.EXTERNAL,
            model_id="",
            request_data=request_data.model_dump(mode="json"),
            status=RequestStatus.DISPATCHED,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        request_id = row.request_id
        assert request_id is not None

    policy = RecordingPolicy()
    controller = SamplingConcurrencyController(policy=policy, initial_limit=2)
    client = ExternalInferenceClient(
        EngineConfig(
            base_model="test-model",
            external_inference_url="https://api.fireworks.ai",
            external_inference_provider="fireworks",
        ),
        db_engine,
        controller,
    )

    async def forward(*args, **kwargs):
        return (
            types.SampleOutput(
                sequences=[
                    types.GeneratedSequence(tokens=[10, 11], logprobs=[-0.1, -0.2], stop_reason="stop"),
                    types.GeneratedSequence(tokens=[12], logprobs=[-0.3], stop_reason="stop"),
                ]
            ),
            {
                "fireworks-prefill-queue-duration": "0.2",
                "fireworks-cached-prompt-tokens": "2",
                "fireworks-prompt-tokens": "3",
            },
            200,
            3,
        )

    monkeypatch.setattr(client, "_forward_to_engine", forward)
    await client.call_and_store_result(request_id, request_data, "", "", base_model="test-model")

    async with AsyncSession(db_engine) as session:
        completed = await session.get(FutureDB, request_id)
        assert completed is not None
        assert completed.status == RequestStatus.COMPLETED
        assert completed.result_data is not None
        assert len(completed.result_data["sequences"]) == 2

    assert controller.in_flight == 0
    assert len(policy.feedback) == 1
    assert policy.feedback[0].source == "fireworks"
    assert policy.feedback[0].queue_duration_s == 0.2
    assert policy.feedback[0].cache_hit_tokens == 2
    assert len(policy.completions) == 1
    assert policy.completions[0].admission_units == 2
    assert policy.completions[0].tokens == 3

    await db_engine.dispose()


@pytest.mark.asyncio
async def test_fireworks_transport_normalizes_token_ids_and_sampling_logprobs():
    captured_request = None

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(
            200,
            headers={"prefill-queue-duration": "0.125"},
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "raw_output": {"completion_token_ids": [10, 11]},
                        "logprobs": {
                            "content": [
                                {"logprob": -9.0, "sampling_logprob": -0.1},
                                {"logprob": -8.0, "sampling_logprob": -0.2},
                            ]
                        },
                    }
                ]
            },
        )

    config = EngineConfig(
        base_model="test-model",
        external_inference_url="https://api.fireworks.ai",
        external_inference_provider="fireworks",
    )
    client = ExternalInferenceClient(config, db_engine=None)
    request_data = types.SampleInput(
        base_model="accounts/test/models/deployed-model",
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]),
        sampling_params=types.SamplingParams(temperature=0.5, max_tokens=4, seed=0),
        num_samples=1,
        checkpoint_id="",
        prompt_logprobs=False,
    )

    async with httpx.AsyncClient(
        base_url=client.base_url,
        transport=httpx.MockTransport(respond),
    ) as http_client:
        output, headers, status, prompt_tokens = await client._forward_to_engine(
            request_data,
            model_id="",
            checkpoint_id="",
            http_client=http_client,
            base_model=request_data.base_model,
        )

    assert captured_request is not None
    assert str(captured_request.url) == "https://api.fireworks.ai/inference/v1/completions"
    assert json.loads(captured_request.content)["raw_output"] is True
    assert output.sequences[0].tokens == [10, 11]
    assert output.sequences[0].logprobs == [-0.1, -0.2]
    assert headers["prefill-queue-duration"] == "0.125"
    assert status == 200
    assert prompt_tokens == 3
