from contextlib import contextmanager

import httpx
import pytest
from fastapi import Request
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from tinker import types

from skyrl.tinker.api import (
    EncodedTextChunk,
    ModelInput,
    SampleFutureResponse,
    SampleRequest,
    SamplingParams,
    app,
    asample,
)
from skyrl.tinker.config import EngineConfig, config_to_argv
from skyrl.tinker.sampling_feedback import (
    AdaptiveTinkerSamplingClient,
    LatestSamplingFeedbackStore,
)
from skyrl.utils.adaptive_concurrency import (
    EngineLoadConcurrencyPolicy,
    SamplingConcurrencyController,
    VLLMEngineLoad,
    VLLMEngineSamplingFeedback,
    sampling_feedback_to_metadata,
)


def overloaded_feedback() -> VLLMEngineSamplingFeedback:
    return VLLMEngineSamplingFeedback(
        engine_loads=(
            VLLMEngineLoad(
                engine_id="decode-0",
                role="decode",
                kv_capacity_tokens=131_072,
                max_model_len=32_768,
                kv_usage=0.95,
                running=8,
                waiting=4,
                waiting_capacity=4,
                preemptions_delta=1,
            ),
        )
    )


@pytest.mark.asyncio
async def test_latest_feedback_store_serializes_response_metadata():
    store = LatestSamplingFeedbackStore()

    assert store.snapshot() is None
    await store.on_feedback(overloaded_feedback())

    response = SampleFutureResponse(
        future_id="12",
        request_id="12",
        sampling_feedback=store.snapshot(),
    )
    assert response.sampling_feedback is not None
    assert response.sampling_feedback["schema_version"] == 1
    assert response.sampling_feedback["kind"] == "vllm_engine"
    assert response.sampling_feedback["engine_loads"][0]["engine_id"] == "decode-0"


@pytest.mark.asyncio
async def test_asample_returns_latest_feedback_when_client_requests_metadata():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    store = LatestSamplingFeedbackStore()
    await store.on_feedback(overloaded_feedback())
    app.state.sampling_feedback_store = store
    app.state.external_inference_client = None
    request = Request(
        {
            "type": "http",
            "app": app,
            "headers": [(b"x-tinker-sampling-backpressure", b"1")],
        }
    )
    sample_request = SampleRequest(
        base_model="test-model",
        prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2])]),
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
        seq_id=7,
    )

    async with AsyncSession(engine) as session:
        response = await asample(sample_request, request, session)
    await engine.dispose()

    assert response.sampling_feedback is not None
    assert response.sampling_feedback["kind"] == "vllm_engine"


class _FakeGeneratedClient:
    def __init__(self, payload):
        self.payload = payload
        self.options = None

    async def post(self, path, *, body, options, cast_to):
        assert path == "/api/v1/asample"
        assert body["sampling_session_id"] == "sampling-session"
        assert cast_to is httpx.Response
        self.options = options
        return httpx.Response(
            200,
            json=self.payload,
            request=httpx.Request("POST", "http://tinker.test/api/v1/asample"),
        )


class _FakeHolder:
    def __init__(self, generated_client):
        self.generated_client = generated_client

    @contextmanager
    def aclient(self, pool_type):
        yield self.generated_client

    def _should_pause_on_billing(self, status_code, message):
        return False


@pytest.mark.asyncio
async def test_adaptive_tinker_client_consumes_asample_response_metadata():
    payload = {
        "future_id": "12",
        "request_id": "12",
        "status": "pending",
        "sampling_feedback": sampling_feedback_to_metadata(overloaded_feedback()),
    }
    generated_client = _FakeGeneratedClient(payload)
    controller = SamplingConcurrencyController(
        policy=EngineLoadConcurrencyPolicy(min_limit=1, max_limit=16),
        initial_limit=8,
    )
    client = object.__new__(AdaptiveTinkerSamplingClient)
    client.holder = _FakeHolder(generated_client)
    client._sampling_session_id = "sampling-session"
    client._sampling_concurrency_controller = controller

    future = await client._send_asample_request(
        request_id=7,
        num_samples=1,
        prompt=types.ModelInput.from_ints([1, 2]),
        sampling_params=types.SamplingParams(max_tokens=4, temperature=0.0),
        include_prompt_logprobs=False,
        topk_prompt_logprobs=0,
    )

    assert future.request_id == "12"
    assert controller.current_limit == 1
    assert generated_client.options["headers"] == {"X-Tinker-Sampling-Backpressure": "1"}
    assert generated_client.options["max_retries"] == 0


def test_tinker_feedback_metrics_urls_survive_subprocess_argv():
    config = EngineConfig(
        base_model="test-model",
        sampling_feedback_metrics_urls=["http://node-a:8080/metrics", "http://node-b:8080/metrics"],
    )

    argv = config_to_argv(config)

    index = argv.index("--sampling-feedback-metrics-urls")
    assert argv[index + 1] == '["http://node-a:8080/metrics", "http://node-b:8080/metrics"]'
