"""Compose client-side sampling admission with an inference backend."""

from loguru import logger

from skyrl.backends.skyrl_train.inference_servers.base import InferenceEngineInterface
from skyrl.train.config import GeneratorConfig
from skyrl.utils.adaptive_concurrency import (
    EngineLoadConcurrencyPolicy,
    FixedConcurrencyPolicy,
    SamplingConcurrencyController,
)


def build_sampling_concurrency_controller(
    generator_cfg: GeneratorConfig,
) -> SamplingConcurrencyController | None:
    """Build the controller installed in the generator's inference client."""

    concurrency_cfg = generator_cfg.sampling_concurrency
    if not concurrency_cfg.enabled:
        return None
    if concurrency_cfg.policy == "fixed":
        policy = FixedConcurrencyPolicy()
    elif concurrency_cfg.policy == "engine_load":
        if not generator_cfg.inference_engine.run_engines_locally:
            raise ValueError(
                "sampling concurrency policy 'engine_load' currently requires SkyRL-managed local vLLM engines"
            )
        if not generator_cfg.inference_engine.enable_ray_prometheus_stats:
            raise ValueError(
                "sampling concurrency policy 'engine_load' requires "
                "generator.inference_engine.enable_ray_prometheus_stats=true"
            )
        if generator_cfg.inference_engine.enable_pd:
            raise ValueError(
                "sampling concurrency policy 'engine_load' does not yet support prefill/decode role attribution"
            )
        policy = EngineLoadConcurrencyPolicy(
            min_limit=concurrency_cfg.min_limit,
            max_limit=concurrency_cfg.max_limit,
        )
    else:
        raise ValueError(f"Unknown sampling concurrency policy: {concurrency_cfg.policy}")

    return SamplingConcurrencyController(
        policy=policy,
        initial_limit=concurrency_cfg.initial_limit,
    )


def attach_sampling_concurrency(
    inference_engine_client: InferenceEngineInterface,
    controller: SamplingConcurrencyController | None,
) -> None:
    """Give the SDK client ownership of request admission and feedback."""

    if controller is None:
        return
    admission_attached = inference_engine_client.set_sampling_concurrency_controller(controller)
    if not admission_attached:
        raise ValueError(
            f"{type(inference_engine_client).__name__} does not implement client-side sampling admission; "
            "override set_sampling_concurrency_controller() to opt in"
        )
    logger.info(
        "Client-side sampling concurrency enabled with policy={} and initial_limit={}",
        type(controller.policy).__name__,
        controller.current_limit,
    )
