"""Run-scoped sampling queue and the lightweight client submitted to by generators.

The service owns request admission and backend feedback.  The client preserves
``InferenceEngineInterface`` so generators only change which implementation is
injected; they continue to execute their own environment and tool loops.

The first transport is in-process because SkyRL currently constructs one
generator in the training entrypoint process.  The boundary is deliberately
request-shaped (not callback-shaped), so a Ray or HTTP transport can implement
the same methods without serializing generator execution.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

from skyrl.backends.skyrl_train.inference_servers.base import (
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
)
from skyrl.utils.adaptive_concurrency import (
    SamplingCompletion,
    SamplingConcurrencyController,
)

_ResultT = TypeVar("_ResultT")


class SamplingFeedbackProducer(Protocol):
    """Lifecycle shared by service-owned backend feedback producers."""

    def start(self) -> None: ...

    async def aclose(self) -> None: ...


@dataclass
class _SamplingWork(Generic[_ResultT]):
    request_id: str
    operation: str
    attempt_id: str | None
    submitted_at_s: float
    completion: asyncio.Future[_ResultT]
    task: asyncio.Task[None] | None = None
    dispatched_at_s: float | None = None


class SamplingService:
    """Long-lived owner of queued sampling requests for one training run.

    Requests are accepted immediately into service-owned work records.  Each
    runner waits on the resizable controller before dispatching to the backend,
    so callers can submit freely while engine pressure controls only the
    ``PENDING -> DISPATCHED`` transition.
    """

    def __init__(
        self,
        backend: InferenceEngineInterface,
        *,
        controller: SamplingConcurrencyController | None = None,
        feedback_producer: SamplingFeedbackProducer | None = None,
    ) -> None:
        self._backend = backend
        self._controller = controller
        self._feedback_producer = feedback_producer
        self._work: dict[str, _SamplingWork[Any]] = {}
        self._started = False
        self._closed = False

    @property
    def controller(self) -> SamplingConcurrencyController | None:
        return self._controller

    @property
    def pending(self) -> int:
        return sum(work.dispatched_at_s is None for work in self._work.values())

    @property
    def in_flight(self) -> int:
        return sum(work.dispatched_at_s is not None for work in self._work.values())

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("sampling service is closed")
        if self._started:
            return
        self._started = True
        if self._feedback_producer is not None:
            self._feedback_producer.start()

    async def generate(
        self,
        input_batch: InferenceEngineInput,
        *,
        model: str | None,
        attempt_id: str | None,
    ) -> InferenceEngineOutput:
        return await self._submit(
            operation="generate",
            attempt_id=attempt_id,
            invoke=lambda: self._backend.generate(input_batch, model=model),
        )

    async def sample(self, request_payload: dict[str, Any], *, attempt_id: str | None) -> Any:
        sample = getattr(self._backend, "sample")
        return await self._submit(
            operation="sample",
            attempt_id=attempt_id,
            invoke=lambda: sample(request_payload),
        )

    async def chat_completion(self, request_payload: dict[str, Any], *, attempt_id: str | None) -> dict[str, Any]:
        return await self._submit(
            operation="chat_completion",
            attempt_id=attempt_id,
            invoke=lambda: self._backend.chat_completion(request_payload),
        )

    async def completion(self, request_payload: dict[str, Any], *, attempt_id: str | None) -> dict[str, Any]:
        return await self._submit(
            operation="completion",
            attempt_id=attempt_id,
            invoke=lambda: self._backend.completion(request_payload),
        )

    async def _submit(
        self,
        *,
        operation: str,
        attempt_id: str | None,
        invoke: Callable[[], Awaitable[_ResultT]],
    ) -> _ResultT:
        self.start()
        loop = asyncio.get_running_loop()
        request_id = uuid.uuid4().hex
        completion: asyncio.Future[_ResultT] = loop.create_future()
        work = _SamplingWork(
            request_id=request_id,
            operation=operation,
            attempt_id=attempt_id,
            submitted_at_s=time.monotonic(),
            completion=completion,
        )
        self._work[request_id] = work
        work.task = asyncio.create_task(self._execute(work, invoke), name=f"sampling-{operation}-{request_id}")

        try:
            return await completion
        except asyncio.CancelledError:
            # A dead generator must not leave an orphan request consuming engine
            # capacity. Policy-driven shedding is layered on this same ownership
            # point in the next stack entry.
            work.task.cancel()
            await asyncio.gather(work.task, return_exceptions=True)
            raise

    async def _execute(
        self,
        work: _SamplingWork[_ResultT],
        invoke: Callable[[], Awaitable[_ResultT]],
    ) -> None:
        started_at_s = 0.0
        succeeded = False
        try:
            if self._controller is None:
                work.dispatched_at_s = started_at_s = time.monotonic()
                result = await invoke()
            else:
                async with self._controller.slot():
                    work.dispatched_at_s = started_at_s = time.monotonic()
                    result = await invoke()
                succeeded = True
                await self._controller.on_completion(SamplingCompletion(duration_s=time.monotonic() - started_at_s))
            if self._controller is None:
                succeeded = True
            if not work.completion.done():
                work.completion.set_result(result)
        except asyncio.CancelledError:
            if not work.completion.done():
                work.completion.cancel()
            raise
        except BaseException as exc:
            if not work.completion.done():
                work.completion.set_exception(exc)
        finally:
            # ``succeeded`` is intentionally retained for debugger visibility;
            # failed requests do not pace policy growth.
            _ = succeeded
            self._work.pop(work.request_id, None)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = [work.task for work in self._work.values() if work.task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._feedback_producer is not None:
            await self._feedback_producer.aclose()


def _request_attempt_id(request_payload: dict[str, Any]) -> str | None:
    body = request_payload.get("json") or {}
    session_id = body.get("session_id") or body.get("sampling_session_id")
    if session_id is not None:
        return str(session_id)
    headers = request_payload.get("headers") or {}
    header_session_id = headers.get("X-Session-ID") or headers.get("x-session-id")
    return str(header_session_id) if header_session_id is not None else None


def _merge_optional_batches(results: list[InferenceEngineOutput], key: str) -> list[Any] | None:
    values = [result.get(key) for result in results]
    if any(value is None for value in values):
        return None
    return [item for value in cast(list[list[Any]], values) for item in value]


class SamplingClient(InferenceEngineInterface):
    """Lightweight generator-facing client for a run-scoped service.

    Sampling methods submit request data to ``SamplingService``. Control-plane
    and rendering methods remain direct delegates because they neither consume
    generation capacity nor belong in the sampling queue.
    """

    def __init__(self, service: SamplingService, backend: InferenceEngineInterface) -> None:
        self._service = service
        self._backend = backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    @property
    def weight_version(self) -> int | None:
        return getattr(self._backend, "weight_version", None)

    def increment_weight_version(self) -> None:
        self._backend.increment_weight_version()

    def get_endpoint_url(self) -> str:
        # Endpoint-based custom harnesses still bypass the in-process transport.
        # The TITO proxy can become an HTTP transport for this same service
        # contract without changing generators that call this client directly.
        return self._backend.get_endpoint_url()

    async def generate(
        self,
        input_batch: InferenceEngineInput,
        model: str | None = None,
    ) -> InferenceEngineOutput:
        primary = input_batch.get("prompt_token_ids") or input_batch.get("prompts")
        if primary is None:
            raise ValueError("InferenceEngineInput requires prompt_token_ids or prompts")

        session_ids = input_batch.get("session_ids")
        requests: list[Awaitable[InferenceEngineOutput]] = []
        for index in range(len(primary)):
            single = dict(input_batch)
            for key in ("prompts", "prompt_token_ids", "session_ids", "mm_features"):
                value = input_batch.get(key)  # type: ignore[literal-required]
                if value is not None:
                    single[key] = [value[index]]
            attempt_id = str(session_ids[index]) if session_ids is not None else None
            requests.append(
                self._service.generate(
                    cast(InferenceEngineInput, single),
                    model=model,
                    attempt_id=attempt_id,
                )
            )

        results = await asyncio.gather(*requests)
        return InferenceEngineOutput(
            responses=[item for result in results for item in result["responses"]],
            response_ids=[item for result in results for item in result["response_ids"]],
            stop_reasons=[item for result in results for item in result["stop_reasons"]],
            response_logprobs=_merge_optional_batches(results, "response_logprobs"),
            prompt_logprobs=_merge_optional_batches(results, "prompt_logprobs"),
            rollout_expert_indices=_merge_optional_batches(results, "rollout_expert_indices"),
        )

    async def sample(self, request_payload: dict[str, Any]) -> Any:
        return await self._service.sample(
            request_payload,
            attempt_id=_request_attempt_id(request_payload),
        )

    async def chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        return await self._service.chat_completion(
            request_payload,
            attempt_id=_request_attempt_id(request_payload),
        )

    async def completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        return await self._service.completion(
            request_payload,
            attempt_id=_request_attempt_id(request_payload),
        )

    async def render_chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        return await self._backend.render_chat_completion(request_payload)

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self._backend.wake_up(*args, **kwargs)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self._backend.sleep(*args, **kwargs)

    async def init_weight_update_communicator(self, init_info):
        return await self._backend.init_weight_update_communicator(init_info)

    async def update_named_weights(self, request):
        return await self._backend.update_named_weights(request)

    async def reset_prefix_cache(self, reset_running_requests: bool = False):
        return await self._backend.reset_prefix_cache(reset_running_requests)

    async def pause_generation(self) -> None:
        return await self._backend.pause_generation()

    async def resume_generation(self) -> None:
        return await self._backend.resume_generation()

    async def finish_session(self, session_id: str) -> None:
        return await self._backend.finish_session(session_id)

    async def get_world_size(self) -> tuple[int, int]:
        return await self._backend.get_world_size()

    async def teardown(self) -> None:
        await self._service.aclose()
        await self._backend.teardown()

    async def aclose(self) -> None:
        await self.teardown()
