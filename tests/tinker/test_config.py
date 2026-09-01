import argparse

from skyrl.tinker.config import EngineConfig, add_model, config_to_argv


def test_sampling_concurrency_config_round_trips_to_engine_subprocess_argv():
    config = EngineConfig.model_validate(
        {
            "base_model": "test-model",
            "external_inference_url": "https://api.fireworks.ai/inference",
            "external_inference_provider": "fireworks",
            "sampling_concurrency": {
                "enabled": True,
                "policy": "queue_delay",
                "initial_limit": 16,
                "max_limit": 128,
                "queue_duration_target_s": 0.25,
            },
        }
    )
    parser = argparse.ArgumentParser()
    add_model(parser, EngineConfig)

    parsed = parser.parse_args(config_to_argv(config))
    restored = EngineConfig.model_validate(vars(parsed))

    assert restored.external_inference_provider == "fireworks"
    assert restored.sampling_concurrency.enabled is True
    assert restored.sampling_concurrency.policy == "queue_delay"
    assert restored.sampling_concurrency.initial_limit == 16
    assert restored.sampling_concurrency.max_limit == 128
    assert restored.sampling_concurrency.queue_duration_target_s == 0.25
