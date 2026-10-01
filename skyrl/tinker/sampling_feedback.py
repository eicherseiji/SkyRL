"""Tinker response-metadata bridge for adaptive sampling admission."""

from __future__ import annotations

import copy
import time
from typing import Any

import httpx
import tinker
from tinker import types
from tinker._base_client import make_request_options
from tinker._compat import model_dump
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.sampling_client import SamplingClient

from skyrl.utils.adaptive_concurrency import (
    ConcurrencyDecision,
    SamplingCompletion,
    SamplingConcurrencyController,
    SamplingFeedback,
    sampling_feedback_from_metadata,
    sampling_feedback_to_metadata,
)
from skyrl.utils.log import logger


class LatestSamplingFeedbackStore:
    """Keep the latest backend observation for the next ``/asample`` response.

    A managed-vLLM producer can publish into this store while the Tinker API
    process serializes its latest snapshot into response metadata. The store is
    deliberately policy-free: the long-lived SDK client remains the admission
    owner and interprets the observation with its own controller.
    """

    def __init__(self) -> None:
        self._metadata: dict[str, Any] | None = None

    async def on_feedback(self, feedback: SamplingFeedback) -> ConcurrencyDecision | None:
        self._metadata = sampling_feedback_to_metadata(feedback)
        return None

    def snapshot(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._metadata)


class AdaptiveTinkerSamplingClient(SamplingClient):
    """Tinker ``SamplingClient`` that gates requests using response metadata.

    This POC preserves the public ``sample`` and ``sample_async`` surface. It
    overrides the SDK's submission hook only because tinker 0.24 parses
    ``/asample`` directly into ``UntypedAPIFuture`` and otherwise discards extra
    response fields. Once the upstream SDK exposes successful response metadata,
    this class can become a thin composition wrapper instead of a subclass.
    """

    def __init__(
        self,
        holder,
        *,
        sampling_session_id: str,
        sampling_concurrency_controller: SamplingConcurrencyController,
        retry_config=None,
    ) -> None:
        super().__init__(
            holder,
            sampling_session_id=sampling_session_id,
            retry_config=retry_config,
            subprocess_sampling=False,
        )
        self._sampling_concurrency_controller = sampling_concurrency_controller

    @classmethod
    def from_client(
        cls,
        client: SamplingClient,
        controller: SamplingConcurrencyController,
    ) -> AdaptiveTinkerSamplingClient:
        """Replace a freshly-created SDK client while keeping its session.

        The returned client must replace ``client`` at the call site; using both
        would duplicate the SDK request sequence. Subprocess/sidecar sampling is
        intentionally outside this POC because the controller has process-local
        state.
        """

        if getattr(client, "_sampling_client_sidecar_handle", None) is not None:
            raise ValueError("AdaptiveTinkerSamplingClient POC does not support subprocess sampling")
        wrapped = cls(
            client.holder,
            sampling_session_id=client._sampling_session_id,
            sampling_concurrency_controller=controller,
        )
        wrapped.retry_handler = client.retry_handler
        wrapped.feature_gates = set(client.feature_gates)
        wrapped._last_queue_state_logged = client._last_queue_state_logged
        wrapped._request_id_counter = client._request_id_counter
        return wrapped

    async def _send_asample_request(
        self,
        request_id: int,
        num_samples: int,
        prompt: types.ModelInput,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int,
    ):
        """Submit normally, retaining the successful response metadata."""

        try:
            request = types.SampleRequest(
                sampling_session_id=self._sampling_session_id,
                seq_id=request_id,
                num_samples=num_samples,
                prompt=prompt,
                sampling_params=sampling_params,
                prompt_logprobs=include_prompt_logprobs,
                topk_prompt_logprobs=topk_prompt_logprobs,
            )
            options = make_request_options(
                extra_headers={"X-Tinker-Sampling-Backpressure": "1"},
            )
            options["max_retries"] = 0
            with self.holder.aclient(ClientConnectionPoolType.SAMPLE) as client:
                response = await client.post(
                    "/api/v1/asample",
                    body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
                    options=options,
                    cast_to=httpx.Response,
                )

            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("Tinker /asample response must be a JSON object")
            metadata = payload.get("sampling_feedback")
            if metadata is not None:
                try:
                    if not isinstance(metadata, dict):
                        raise TypeError("sampling_feedback response metadata must be an object")
                    feedback = sampling_feedback_from_metadata(metadata)
                    await self._sampling_concurrency_controller.on_feedback(feedback)
                except (TypeError, ValueError) as exc:
                    logger.warning("Ignoring invalid Tinker sampling feedback metadata: %s", exc)
            return types.UntypedAPIFuture.model_validate(payload)
        except tinker.APIStatusError as exc:
            if exc.status_code == 429 or self.holder._should_pause_on_billing(exc.status_code, exc.message):
                return None
            raise

    async def _sample_async_impl(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int = 0,
    ) -> types.SampleResponse:
        """Hold one client permit for the complete hosted inference request."""

        async with self._sampling_concurrency_controller.slot():
            started_at = time.monotonic()
            result = await super()._sample_async_impl(
                prompt,
                num_samples,
                sampling_params,
                include_prompt_logprobs,
                topk_prompt_logprobs,
            )
        await self._sampling_concurrency_controller.on_completion(
            SamplingCompletion(duration_s=time.monotonic() - started_at)
        )
        return result
