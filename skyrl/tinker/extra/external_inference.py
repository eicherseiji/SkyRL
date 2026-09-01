import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from cloudpathlib import AnyPath
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.backends.renderer import render_model_input
from skyrl.backends.utils import convert_vllm_prompt_logprobs
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import FutureDB, RequestStatus
from skyrl.utils.adaptive_concurrency import (
    RequestSamplingFeedback,
    SamplingCompletion,
    SamplingConcurrencyController,
)
from skyrl.utils.log import logger
from skyrl.utils.storage import download_and_unpack

if TYPE_CHECKING:
    from skyrl.tinker.api import SampleRequest


def _extract_checkpoint_sync(checkpoint_path: AnyPath, target_dir: Path) -> None:
    """Extract a LoRA checkpoint to disk for vLLM to load.

    This is a blocking operation (filesystem/network I/O) and should be called
    via asyncio.to_thread() to avoid blocking the event loop.
    """
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    # Extract the checkpoint if it doesn't already exist
    if not target_dir.exists():
        try:
            with download_and_unpack(checkpoint_path) as extracted_path:
                extracted_path.rename(target_dir)
        except FileExistsError:
            # This could happen if two processes try to download the file.
            # In that case the other process won the race and created target_dir.
            pass


class ExternalInferenceClient:
    """Client for calling external inference engines (vLLM or Fireworks)."""

    def __init__(
        self,
        engine_config: EngineConfig,
        db_engine,
        sampling_concurrency_controller: SamplingConcurrencyController | None = None,
    ):
        external_url = str(engine_config.external_inference_url).rstrip("/")
        self.base_url = (
            external_url if engine_config.external_inference_provider == "fireworks" else f"{external_url}/v1"
        )
        self.api_key = engine_config.external_inference_api_key
        self.checkpoints_base = engine_config.checkpoints_base
        self.lora_base_dir = engine_config.external_inference_lora_base
        self.db_engine = db_engine
        self.provider = engine_config.external_inference_provider
        self.sampling_concurrency_controller = sampling_concurrency_controller

    async def call_and_store_result(
        self,
        request_id: int,
        sample_req,
        model_id: str,
        checkpoint_id: str,
        *,
        base_model: str | None = None,
    ):
        """Background task to call external engine and store result in database."""
        started_at = time.monotonic()
        feedback: RequestSamplingFeedback | None = None
        completion_tokens = 0
        try:

            async def _call():
                async with httpx.AsyncClient(
                    base_url=self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=httpx.Timeout(300.0, connect=10.0),  # 5 minutes for inference, 10s for connect
                ) as http_client:
                    return await self._forward_to_engine(
                        sample_req, model_id, checkpoint_id, http_client, base_model=base_model
                    )

            if self.sampling_concurrency_controller is None:
                result, response_headers, http_status_code, prompt_tokens = await _call()
            else:
                async with self.sampling_concurrency_controller.slot(units=sample_req.num_samples):
                    result, response_headers, http_status_code, prompt_tokens = await _call()

            completion_tokens = sum(len(sequence.tokens) for sequence in result.sequences)
            if self.provider == "fireworks":
                feedback = RequestSamplingFeedback.from_fireworks_headers(
                    response_headers,
                    request_latency_s=time.monotonic() - started_at,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    http_status_code=http_status_code,
                )
            else:
                feedback = RequestSamplingFeedback(
                    source="tinker",
                    request_latency_s=time.monotonic() - started_at,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            result_data = result.model_dump()
            status = RequestStatus.COMPLETED
        except httpx.HTTPStatusError as e:
            if self.provider == "fireworks":
                feedback = RequestSamplingFeedback.from_fireworks_headers(
                    e.response.headers,
                    request_latency_s=time.monotonic() - started_at,
                    completion_tokens=0,
                    http_status_code=e.response.status_code,
                )
            logger.exception("External engine error")
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED
        except httpx.RequestError as e:
            if self.provider == "fireworks":
                feedback = RequestSamplingFeedback.from_fireworks_headers(
                    {},
                    request_latency_s=time.monotonic() - started_at,
                    completion_tokens=0,
                    transport_error=True,
                )
            logger.exception("External engine transport error")
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED
        except Exception as e:
            logger.exception("External engine error")
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED

        if self.sampling_concurrency_controller is not None:
            if feedback is not None:
                await self.sampling_concurrency_controller.on_feedback(feedback)
            await self.sampling_concurrency_controller.on_completion(
                SamplingCompletion(
                    tokens=completion_tokens,
                    duration_s=time.monotonic() - started_at,
                    admission_units=sample_req.num_samples,
                )
            )

        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            if future is None:
                logger.warning("FutureDB row %s missing on completion write — skipping", request_id)
                return
            future.result_data = result_data
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def _forward_to_engine(
        self,
        request: "SampleRequest | types.SampleInput",
        model_id: str,
        checkpoint_id: str,
        http_client: httpx.AsyncClient,
        *,
        base_model: str | None = None,
    ) -> tuple[types.SampleOutput, dict[str, str], int, int]:
        """Forward request to vLLM with dynamic LoRA loading.

        Extracts the checkpoint to the configured external_inference_lora_base and references it by a model name
        that vLLM can dynamically load via the lora_filesystem_resolver plugin.

        For base model sampling (no LoRA), the request is sent directly using the base model name.
        """
        model_input = request.prompt.to_types() if hasattr(request.prompt, "to_types") else request.prompt
        prompt_tokens = render_model_input([model_input])[0].prompt_ids

        if self.provider == "fireworks" and not base_model:
            raise ValueError(
                "Fireworks sampling requires SampleInput.base_model to name an already deployed Fireworks model"
            )

        if base_model:
            # Base model sampling: use the model name directly, no LoRA checkpoint needed
            model_name = base_model
        else:
            # LoRA sampling: extract checkpoint and reference it by name for dynamic loading
            model_name = f"{model_id}_{checkpoint_id}"
            checkpoint_path = self.checkpoints_base / model_id / "sampler_weights" / f"{checkpoint_id}.tar.gz"
            target_dir = self.lora_base_dir / model_name

            await asyncio.to_thread(_extract_checkpoint_sync, checkpoint_path, target_dir)

        payload = {
            "model": model_name,
            "prompt": prompt_tokens,
            "n": request.num_samples,
            "seed": request.sampling_params.seed,
            "max_tokens": request.sampling_params.max_tokens,
            "temperature": request.sampling_params.temperature,
            "top_p": request.sampling_params.top_p,
            "top_k": request.sampling_params.top_k,
            "logprobs": True,
            "stream": False,
            "return_token_ids": True,
        }
        # vLLM's `prompt_logprobs` is an int: 0 returns just the prompt tokens'
        # own logprobs, k>0 also returns the top-k per position.
        topk_prompt_logprobs = getattr(request, "topk_prompt_logprobs", 0) or 0
        want_prompt_logprobs = bool(request.prompt_logprobs) or topk_prompt_logprobs > 0
        if self.provider == "fireworks" and want_prompt_logprobs:
            raise NotImplementedError("Fireworks prompt logprobs are not yet normalized by this transport adapter")
        if want_prompt_logprobs:
            payload["prompt_logprobs"] = topk_prompt_logprobs

        # Pass X-Session-ID for deterministic routing
        headers = {}
        session_id = types.make_routing_session_id(request.sampling_session_id, request.seq_id)
        if session_id is not None:
            headers["X-Session-ID"] = session_id

        if self.provider == "fireworks":
            payload["raw_output"] = True
            completion_path = "/inference/v1/completions"
        else:
            completion_path = "/completions"

        response = await http_client.post(completion_path, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        prompt_logprobs = None
        topk = None
        if want_prompt_logprobs:
            # All `n` choices share one prompt, so vLLM repeats the same prompt
            # logprobs on each choice; read them off the first.
            raw = result["choices"][0].get("prompt_logprobs") if result["choices"] else None
            if raw is None:
                logger.warning("Requested prompt logprobs but vLLM /completions returned none")
            prompt_logprobs, topk = convert_vllm_prompt_logprobs(prompt_tokens, raw, topk=topk_prompt_logprobs)

        sequences = []
        for choice in result["choices"]:
            if self.provider == "fireworks":
                raw_output = choice.get("raw_output") or {}
                tokens = raw_output.get("completion_token_ids")
                if tokens is None:
                    raise RuntimeError("Fireworks completion omitted raw_output.completion_token_ids")
                content = (choice.get("logprobs") or {}).get("content") or []
                raw_logprobs = [
                    item.get("sampling_logprob") if item.get("sampling_logprob") is not None else item.get("logprob")
                    for item in content
                ]
                if len(raw_logprobs) != len(tokens) or any(value is None for value in raw_logprobs):
                    raise RuntimeError("Fireworks completion omitted per-token sampling logprobs")
                logprobs = [float(value) for value in raw_logprobs if value is not None]
                finish_reason = choice["finish_reason"]
                stop_reason = "stop" if finish_reason in ("stop", "stop_token") else "length"
            else:
                lp = choice["logprobs"]
                tokens = choice["token_ids"]
                logprobs = lp["token_logprobs"]
                stop_reason = choice["finish_reason"]
            sequences.append(
                types.GeneratedSequence(
                    tokens=tokens,
                    logprobs=logprobs,
                    stop_reason=stop_reason,
                )
            )

        return (
            types.SampleOutput(
                sequences=sequences,
                prompt_logprobs=prompt_logprobs,
                topk_prompt_logprobs=topk,
            ),
            dict(response.headers),
            response.status_code,
            len(prompt_tokens),
        )
