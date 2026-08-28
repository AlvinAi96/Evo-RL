#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.delta_action_processor import (
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
)
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_STATE

__all__ = [
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "RelativeActionsProcessorStep",
    "AbsoluteActionsProcessorStep",
    "to_relative_actions",
    "to_absolute_actions",
]


def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """把绝对 action 转成相对 action：relative = action - state。"""
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]

    # state 可能来自 CPU 缓存，这里对齐到 action 的设备和 dtype，避免 CUDA 推理时报错。
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)

    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)

    actions = actions.clone()
    actions[..., :dims] -= state_offset
    return actions


def to_absolute_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """把相对 action 还原成绝对 action：absolute = relative + state。"""
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]

    # state 可能来自 CPU 缓存，这里对齐到 action 的设备和 dtype，避免 CUDA 推理时报错。
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)

    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)

    actions = actions.clone()
    actions[..., :dims] += state_offset
    return actions


@ProcessorStepRegistry.register("relative_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """将 action 从绝对关节目标转换成相对当前 state 的增量。"""

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, action_dim: int) -> list[bool]:
        """根据 exclude_joints 生成 action 维度 mask，True 表示该维度转相对量。"""
        if not self.exclude_joints or self.action_names is None:
            return [True] * action_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * action_dim

        mask = []
        for name in self.action_names[:action_dim]:
            action_name = str(name).lower()
            is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
            mask.append(not is_excluded)

        if len(mask) < action_dim:
            mask.extend([True] * (action_dim - len(mask)))
        return mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """缓存 observation.state，并在开启时把 action 转成相对 action。"""
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None

        # 后处理恢复绝对 action 时需要复用本次 observation.state。
        if state is not None:
            self._last_state = state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition

        new_transition[TransitionKey.ACTION] = to_relative_actions(
            action,
            state,
            self._build_mask(action.shape[-1]),
        )
        return new_transition

    def get_cached_state(self) -> torch.Tensor | None:
        """返回最近缓存的 observation.state，供 absolute_actions_processor 使用。"""
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        """导出可序列化配置，保证模型保存/加载时 processor 配置可复用。"""
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """relative 转换不改变特征 schema。"""
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    """将模型输出的相对 action 还原成机器人可执行的绝对 action。"""

    enabled: bool = False
    relative_step: RelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """使用 preprocessor 缓存的 state，把相对 action 加回当前 state。"""
        if not self.enabled:
            return transition

        if self.relative_step is None:
            raise RuntimeError("AbsoluteActionsProcessorStep requires a paired RelativeActionsProcessorStep.")

        cached_state = self.relative_step.get_cached_state()
        if cached_state is None:
            raise RuntimeError("RelativeActionsProcessorStep has no cached observation.state.")

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        mask = self.relative_step._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_absolute_actions(action, cached_state, mask)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        """导出可序列化配置；relative_step 是运行时引用，不写入配置。"""
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """absolute 转换不改变特征 schema。"""
        return features
