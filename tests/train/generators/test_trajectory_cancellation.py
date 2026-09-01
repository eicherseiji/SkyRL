import asyncio
from collections import Counter
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skyrl.train.config import GeneratorConfig
from skyrl.train.generators.base import BatchMetadata, TrajectoryID
from skyrl.train.generators.skyrl_gym_generator import (
    SkyRLGymGenerator,
    TrajectoryOutput,
)
from skyrl.train.generators.trajectory_cancellation import (
    TrajectoryCancellationCandidate,
    YoungestTrainingTrajectoryCancellationPolicy,
)
from skyrl.utils.adaptive_concurrency import (
    ConcurrencyDecision,
    SamplingConcurrencyController,
    SamplingFeedback,
)


class _SheddingPolicy:
    def on_feedback(self, feedback, context):
        return ConcurrencyDecision(desired_limit=1, shed_count=1, reason="hard_pressure")

    def on_completion(self, completion, context):
        return None


def _generator_config() -> GeneratorConfig:
    config = GeneratorConfig()
    config.batched = False
    config.max_turns = 1
    config.max_input_length = 64
    config.sampling_params.max_generate_length = 8
    config.sampling_params.logprobs = None
    config.chat_template.source = "name"
    config.chat_template.name_or_path = None
    return config


def _tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [1, 2]
    tokenizer.eos_token_id = 2
    tokenizer.eos_token = "</s>"
    return tokenizer


def test_youngest_trajectory_policy_selects_exact_count_and_excludes_eval():
    policy = YoungestTrainingTrajectoryCancellationPolicy()
    candidates = [
        TrajectoryCancellationCandidate("old", "group-a", 0, 1.0, "train"),
        TrajectoryCancellationCandidate("eval", "group-eval", 0, 3.0, "eval"),
        TrajectoryCancellationCandidate("young", "group-b", 1, 2.0, "train"),
    ]

    assert policy.select_trajectories(candidates, shed_count=1) == ("young",)
    assert policy.select_trajectories(candidates, shed_count=5) == ("young", "old")


@pytest.mark.asyncio
async def test_generator_cancels_and_retries_only_one_youngest_trajectory():
    controller = SamplingConcurrencyController(policy=_SheddingPolicy(), initial_limit=4)
    inference_client = MagicMock()
    inference_client.set_sampling_concurrency_controller.return_value = True
    inference_client.finish_session = AsyncMock()
    env_config = MagicMock(max_env_workers=0)
    generator = SkyRLGymGenerator(
        generator_cfg=_generator_config(),
        skyrl_gym_cfg=env_config,
        inference_engine_client=inference_client,
        tokenizer=_tokenizer(),
        sampling_concurrency_controller=controller,
    )

    release = asyncio.Event()
    all_started = asyncio.Event()
    starts: list[tuple[str, int]] = []
    cancelled: list[tuple[str, int]] = []

    async def agent_loop(*args, trajectory_id, **kwargs):
        identity = (trajectory_id.instance_id, trajectory_id.repetition_id)
        starts.append(identity)
        if len(starts) >= 4:
            all_started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(identity)
            raise
        return TrajectoryOutput(
            response_ids=[trajectory_id.repetition_id + 10],
            reward=[1.0],
            stop_reason="stop",
            loss_mask=[1],
            prompt_ids=[1],
            rollout_logprobs=None,
            env_metrics={},
        )

    generator.agent_loop = agent_loop
    trajectory_ids = [
        TrajectoryID(instance_id=group, repetition_id=repetition)
        for group in ("old", "young")
        for repetition in range(2)
    ]
    input_batch = {
        "prompts": [[{"role": "user", "content": group}] for group in ("old", "old", "young", "young")],
        "env_classes": ["gsm8k"] * 4,
        "env_extras": [{} for _ in range(4)],
        "sampling_params": None,
        "trajectory_ids": trajectory_ids,
        "batch_metadata": BatchMetadata(global_step=0, training_phase="train"),
    }

    generation = asyncio.create_task(generator.generate(input_batch, disable_tqdm=True))
    await asyncio.wait_for(all_started.wait(), timeout=1)
    await controller.on_feedback(SamplingFeedback())

    async def one_young_trajectory_restarted():
        while sum(group == "young" for group, _ in starts) < 3:
            await asyncio.sleep(0)

    await asyncio.wait_for(one_young_trajectory_restarted(), timeout=1)
    release.set()
    output = await asyncio.wait_for(generation, timeout=1)

    assert cancelled == [("young", 1)]
    assert Counter(starts) == Counter(
        {
            ("old", 0): 1,
            ("old", 1): 1,
            ("young", 0): 1,
            ("young", 1): 2,
        }
    )
    assert output["response_ids"] == [[10], [11], [10], [11]]


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_agent_loop_closes_environment_when_cancelled(mock_make):
    config = _generator_config()
    tokenizer = _tokenizer()
    inference_client = MagicMock()
    inference_client.finish_session = AsyncMock()
    request_started = asyncio.Event()

    async def block_generation(*args, **kwargs):
        request_started.set()
        await asyncio.Event().wait()

    inference_client.generate = AsyncMock(side_effect=block_generation)
    env = MagicMock()
    env.init.return_value = ([{"role": "user", "content": "prompt"}], {})
    mock_make.return_value = env
    generator = SkyRLGymGenerator(
        generator_cfg=config,
        skyrl_gym_cfg=MagicMock(max_env_workers=0),
        inference_engine_client=inference_client,
        tokenizer=tokenizer,
    )

    task = asyncio.create_task(
        generator.agent_loop(
            [{"role": "user", "content": "prompt"}],
            "test",
            {},
            max_tokens=8,
            max_input_length=64,
            trajectory_id=TrajectoryID("group", 0),
        )
    )
    await asyncio.wait_for(request_started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    env.close.assert_called_once_with()
    inference_client.finish_session.assert_awaited_once_with("group_0")
