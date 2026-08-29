from types import SimpleNamespace

import numpy as np
import torch
from datasets import Dataset

from lerobot.datasets.compute_stats import compute_relative_action_stats
from lerobot.policies.factory import (
    _reconnect_relative_absolute_steps,
    make_policy,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.core import TransitionKey
from lerobot.processor.tokenizer_processor import TokenizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

ACTION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def test_mixed_relative_action_roundtrip_keeps_gripper_absolute():
    state = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 7.0]])
    actions = torch.tensor([[[11.0, 18.0, 35.0, 44.0, 49.0, 2.0], [12.0, 25.0, 31.0, 38.0, 53.0, 9.0]]])
    relative_step = RelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=["gripper"],
        action_names=ACTION_NAMES,
    )
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)

    relative_transition = relative_step(
        {
            TransitionKey.OBSERVATION: {OBS_STATE: state},
            TransitionKey.ACTION: actions,
        }
    )
    relative_actions = relative_transition[TransitionKey.ACTION]

    torch.testing.assert_close(relative_actions[..., :5], actions[..., :5] - state[:, None, :5])
    torch.testing.assert_close(relative_actions[..., 5], actions[..., 5])

    recovered = absolute_step({TransitionKey.ACTION: relative_actions})[TransitionKey.ACTION]
    torch.testing.assert_close(recovered, actions)


def test_pi05_processor_factory_uses_shared_relative_step(monkeypatch):
    monkeypatch.setattr(TokenizerProcessorStep, "__post_init__", lambda self: None)
    config = PI05Config(
        device="cpu",
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        action_feature_names=ACTION_NAMES,
    )

    preprocessor, postprocessor = make_pi05_pre_post_processors(config, dataset_stats=None)
    relative_step = next(
        step for step in preprocessor.steps if isinstance(step, RelativeActionsProcessorStep)
    )
    absolute_step = next(
        step for step in postprocessor.steps if isinstance(step, AbsoluteActionsProcessorStep)
    )

    assert relative_step.enabled
    assert relative_step._build_mask(6) == [True, True, True, True, True, False]
    assert absolute_step.enabled
    assert absolute_step.relative_step is relative_step
    assert list(map(type, preprocessor.steps)).index(RelativeActionsProcessorStep) < list(
        map(type, preprocessor.steps)
    ).index(NormalizerProcessorStep)
    assert list(map(type, postprocessor.steps)).index(UnnormalizerProcessorStep) < list(
        map(type, postprocessor.steps)
    ).index(AbsoluteActionsProcessorStep)


def test_reconnect_relative_absolute_steps_after_deserialization():
    relative_step = RelativeActionsProcessorStep(enabled=True)
    absolute_step = AbsoluteActionsProcessorStep(enabled=True)
    preprocessor = PolicyProcessorPipeline(steps=[relative_step], name="policy_preprocessor")
    postprocessor = PolicyProcessorPipeline(steps=[absolute_step], name="policy_postprocessor")

    assert absolute_step.relative_step is None
    _reconnect_relative_absolute_steps(preprocessor, postprocessor)
    assert absolute_step.relative_step is relative_step


def test_compute_relative_action_stats_uses_chunks_and_excludes_gripper():
    actions = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 11.0, 101.0],
            [4.0, 14.0, 102.0],
            [20.0, 30.0, 200.0],
            [23.0, 35.0, 201.0],
            [27.0, 40.0, 202.0],
        ],
        dtype=np.float32,
    )
    states = np.array(
        [
            [0.0, 8.0, 900.0],
            [1.0, 9.0, 901.0],
            [3.0, 12.0, 902.0],
            [18.0, 28.0, 903.0],
            [21.0, 31.0, 904.0],
            [25.0, 36.0, 905.0],
        ],
        dtype=np.float32,
    )
    episode_indices = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    dataset = Dataset.from_dict(
        {
            ACTION: actions.tolist(),
            OBS_STATE: states.tolist(),
            "episode_index": episode_indices.tolist(),
        }
    )
    features = {
        ACTION: {
            "dtype": "float32",
            "shape": [3],
            "names": ["joint_0.pos", "joint_1.pos", "gripper.pos"],
        },
        OBS_STATE: {
            "dtype": "float32",
            "shape": [3],
            "names": ["joint_0.pos", "joint_1.pos", "gripper.pos"],
        },
    }

    stats = compute_relative_action_stats(
        dataset,
        features,
        chunk_size=2,
        exclude_joints=["gripper"],
    )

    expected_chunks = []
    for start in [0, 1, 3, 4]:
        chunk = actions[start : start + 2].copy()
        chunk[:, :2] -= states[start, :2]
        expected_chunks.append(chunk)
    expected = np.concatenate(expected_chunks, axis=0)

    np.testing.assert_allclose(stats["mean"], expected.mean(axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(stats["std"], expected.std(axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(stats["min"], expected.min(axis=0))
    np.testing.assert_allclose(stats["max"], expected.max(axis=0))
    np.testing.assert_array_equal(stats["count"], np.array([8]))
    np.testing.assert_allclose(expected[:, 2], np.array([100, 101, 101, 102, 200, 201, 201, 202]))


def test_make_policy_populates_action_feature_names(monkeypatch):
    class DummyPolicy(torch.nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.config = config

    monkeypatch.setattr("lerobot.policies.factory.get_policy_class", lambda _: DummyPolicy)
    metadata = SimpleNamespace(
        features={
            ACTION: {"dtype": "float32", "shape": [6], "names": ACTION_NAMES},
            OBS_STATE: {"dtype": "float32", "shape": [6], "names": ACTION_NAMES},
        },
        stats={},
    )
    config = PI05Config(device="cpu")

    policy = make_policy(config, ds_meta=metadata, rename_map={"unused": "unused"})

    assert policy.config.action_feature_names == ACTION_NAMES


def test_pi05_relative_action_defaults():
    config = PI05Config(device="cpu")
    assert config.use_relative_actions is False
    assert config.relative_exclude_joints == ["gripper"]
    assert config.action_feature_names is None
