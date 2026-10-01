from unittest.mock import MagicMock

import pytest

from skyrl.train.config import GeneratorConfig
from skyrl.train.sampling_concurrency import (
    attach_sampling_concurrency,
    build_sampling_concurrency_controller,
)
from skyrl.utils.adaptive_concurrency import EngineLoadConcurrencyPolicy


def test_builds_client_owned_engine_load_controller():
    config = GeneratorConfig()
    config.batched = False
    config.sampling_concurrency.enabled = True
    config.sampling_concurrency.policy = "engine_load"
    config.sampling_concurrency.initial_limit = 3

    controller = build_sampling_concurrency_controller(config)

    assert controller is not None
    assert controller.current_limit == 3
    assert isinstance(controller.policy, EngineLoadConcurrencyPolicy)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("run_engines_locally", "requires SkyRL-managed local vLLM"),
        ("enable_ray_prometheus_stats", "requires.*enable_ray_prometheus_stats"),
        ("enable_pd", "does not yet support prefill/decode"),
    ],
)
def test_rejects_engine_load_without_coherent_managed_vllm_metrics(field, message):
    config = GeneratorConfig()
    config.sampling_concurrency.enabled = True
    config.sampling_concurrency.policy = "engine_load"
    if field == "run_engines_locally":
        config.inference_engine.run_engines_locally = False
    elif field == "enable_ray_prometheus_stats":
        config.inference_engine.enable_ray_prometheus_stats = False
    else:
        config.inference_engine.enable_pd = True

    with pytest.raises(ValueError, match=message):
        build_sampling_concurrency_controller(config)


def test_rejects_engine_load_policy_without_client_admission_support():
    config = GeneratorConfig()
    config.sampling_concurrency.enabled = True
    config.sampling_concurrency.policy = "engine_load"
    controller = build_sampling_concurrency_controller(config)
    inference_client = MagicMock()
    inference_client.set_sampling_concurrency_controller.return_value = False

    with pytest.raises(ValueError, match="does not implement client-side sampling admission"):
        attach_sampling_concurrency(inference_client, controller)
