#!/usr/bin/env python

import pytest
import torch

from lerobot.configs.train import ACPConfig
from lerobot.rl.acp_success_filter import (
    ACPSuccessLossWeights,
    build_acp_success_loss_weights,
    find_success_episode_indices,
    resolve_acp_dataset_episodes,
)


class DummyEpisodes(dict):
    @property
    def column_names(self):
        return list(self.keys())


class DummyMeta:
    def __init__(self, episodes):
        self.episodes = episodes


class DummyDataset:
    def __init__(self, episodes, selected_episodes=None):
        self.meta = DummyMeta(episodes)
        self.episodes = selected_episodes


def test_find_success_episode_indices_from_episode_metadata():
    episodes = DummyEpisodes(
        {
            "episode_index": [3, 4, 5],
            "episode_success": ["success", "failure", "success"],
        }
    )

    assert find_success_episode_indices(episodes, "episode_success") == [3, 5]


def test_resolve_acp_dataset_episodes_drop_keeps_only_success_requested_episodes():
    episodes = DummyEpisodes(
        {
            "episode_index": [0, 1, 2, 3],
            "episode_success": ["success", "failure", "success", "failure"],
        }
    )
    cfg = ACPConfig(enable=True, failure_loss_mode="drop", success_field="episode_success")

    kept = resolve_acp_dataset_episodes(
        requested_episodes=[1, 2, 3],
        all_episodes=episodes,
        cfg=cfg,
    )

    assert kept == [2]


def test_resolve_acp_dataset_episodes_mask_loss_keeps_original_episode_selection():
    episodes = DummyEpisodes({"episode_success": ["success", "failure"]})
    cfg = ACPConfig(enable=True, failure_loss_mode="mask_loss")

    kept = resolve_acp_dataset_episodes(
        requested_episodes=[0, 1],
        all_episodes=episodes,
        cfg=cfg,
    )

    assert kept == [0, 1]


def test_acp_success_loss_weights_masks_failure_samples():
    weights = ACPSuccessLossWeights(success_episode_indices=[0, 2], device=torch.device("cpu"))

    batch_weights, stats = weights.compute_batch_weights(
        {"episode_index": torch.tensor([0, 1, 2, 1], dtype=torch.int64)}
    )

    assert torch.equal(batch_weights, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert stats["acp_success_loss_num_success"] == 2
    assert stats["acp_success_loss_num_failure"] == 2


def test_build_acp_success_loss_weights_respects_selected_episodes():
    episodes = DummyEpisodes(
        {
            "episode_index": [0, 1, 2],
            "episode_success": ["success", "success", "failure"],
        }
    )
    dataset = DummyDataset(episodes, selected_episodes=[1, 2])
    cfg = ACPConfig(enable=True, failure_loss_mode="mask_loss")

    weights = build_acp_success_loss_weights(dataset=dataset, cfg=cfg, device=torch.device("cpu"))

    assert weights is not None
    assert weights.success_episode_indices == {1}


def test_drop_mode_raises_when_no_success_episode_is_selected():
    episodes = DummyEpisodes(
        {
            "episode_index": [0, 1],
            "episode_success": ["failure", "failure"],
        }
    )
    cfg = ACPConfig(enable=True, failure_loss_mode="drop")

    with pytest.raises(ValueError, match="no successful episodes"):
        resolve_acp_dataset_episodes(requested_episodes=None, all_episodes=episodes, cfg=cfg)
