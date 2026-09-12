#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ACP success-only training helpers."""

from __future__ import annotations

from typing import Any

import torch

from lerobot.configs.train import ACPConfig
from lerobot.utils.recording_annotations import EPISODE_SUCCESS, normalize_episode_success_label


def _to_python_int(value: Any) -> int:
    """把 tensor/numpy 标量统一转成 Python int，方便 episode 索引查表。"""
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _column_names(table: Any) -> set[str]:
    """读取 HF Dataset / pandas / dict 里的列名。"""
    if hasattr(table, "column_names"):
        return set(table.column_names)
    if hasattr(table, "columns"):
        return set(table.columns)
    if isinstance(table, dict):
        return set(table)
    return set()


def _column_values(table: Any, column: str) -> list[Any]:
    """按列读取 episode metadata，避免依赖某一种表格实现。"""
    return list(table[column])


def find_success_episode_indices(episodes: Any, success_field: str) -> list[int]:
    """从 episode metadata 中找出标注为 success 的 episode_index。"""
    names = _column_names(episodes)
    if success_field not in names:
        raise KeyError(f"ACP success field '{success_field}' is missing from episode metadata.")

    labels = _column_values(episodes, success_field)
    if "episode_index" in names:
        episode_indices = [_to_python_int(v) for v in _column_values(episodes, "episode_index")]
    else:
        episode_indices = list(range(len(labels)))

    success_indices: list[int] = []
    for episode_index, label in zip(episode_indices, labels, strict=True):
        normalized = normalize_episode_success_label(label)
        if normalized == EPISODE_SUCCESS:
            success_indices.append(episode_index)
    return success_indices


def resolve_acp_dataset_episodes(
    *,
    requested_episodes: list[int] | None,
    all_episodes: Any,
    cfg: ACPConfig,
) -> list[int] | None:
    """根据 ACP 失败样本策略，决定 LeRobotDataset 实际加载哪些 episode。"""
    if not cfg.enable or cfg.failure_loss_mode != "drop":
        return requested_episodes

    names = _column_names(all_episodes)
    success_indices = set(find_success_episode_indices(all_episodes, cfg.success_field))
    if requested_episodes is not None:
        candidate_indices = list(requested_episodes)
    elif "episode_index" in names:
        candidate_indices = [_to_python_int(v) for v in _column_values(all_episodes, "episode_index")]
    else:
        candidate_indices = list(range(len(_column_values(all_episodes, cfg.success_field))))
    kept_indices = [episode_index for episode_index in candidate_indices if episode_index in success_indices]
    if not kept_indices:
        raise ValueError(
            "ACP failure_loss_mode='drop' selected no successful episodes. "
            f"Check acp.success_field='{cfg.success_field}' and dataset labels."
        )
    return kept_indices


class ACPSuccessLossWeights:
    """根据 batch 的 episode_index 生成成功样本 loss 权重。"""

    def __init__(self, success_episode_indices: list[int], device: torch.device):
        if not success_episode_indices:
            raise ValueError("ACP success-only loss requires at least one successful episode.")
        self.success_episode_indices = set(success_episode_indices)
        self.device = device

    def compute_batch_weights(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float | int]]:
        """返回 batch 级 0/1 权重；失败样本会 forward 但不产生 loss。"""
        if "episode_index" not in batch:
            raise KeyError("ACP success-only loss requires 'episode_index' in the training batch.")

        episode_indices = batch["episode_index"]
        if isinstance(episode_indices, torch.Tensor):
            values = episode_indices.detach().cpu().tolist()
        else:
            values = list(episode_indices)

        raw_weights = [
            1.0 if _to_python_int(episode_index) in self.success_episode_indices else 0.0
            for episode_index in values
        ]
        weights = torch.tensor(raw_weights, dtype=torch.float32, device=self.device)
        success_count = int(weights.sum().item())
        failure_count = int(weights.numel() - success_count)
        stats = {
            "acp_success_loss_weight_mean": float(weights.mean().item()) if weights.numel() > 0 else 0.0,
            "acp_success_loss_num_success": success_count,
            "acp_success_loss_num_failure": failure_count,
        }
        return weights, stats


def build_acp_success_loss_weights(
    *, dataset: Any, cfg: ACPConfig, device: torch.device
) -> ACPSuccessLossWeights | None:
    """按配置创建 success-only loss 权重器。"""
    if not cfg.enable or cfg.failure_loss_mode != "mask_loss":
        return None

    success_indices = find_success_episode_indices(dataset.meta.episodes, cfg.success_field)
    if dataset.episodes is not None:
        selected_indices = set(dataset.episodes)
        success_indices = [
            episode_index for episode_index in success_indices if episode_index in selected_indices
        ]
    return ACPSuccessLossWeights(success_indices, device=device)
