# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_STATE

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry


def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert selected absolute action dimensions to offsets from the current state."""
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] -= state_offset
    return actions


def to_absolute_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert selected relative action dimensions back to absolute positions."""
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
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
    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, action_dim: int) -> list[bool]:
        if not self.exclude_joints or self.action_names is None:
            return [True] * action_dim
        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        mask = []
        for name in self.action_names[:action_dim]:
            action_name = str(name).lower()
            mask.append(not any(token == action_name or token in action_name for token in exclude_tokens))
        if len(mask) < action_dim:
            mask.extend([True] * (action_dim - len(mask)))
        return mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None:
            self._last_state = state
        if not self.enabled:
            return transition
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition
        mask = self._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_relative_actions(action, state, mask)
        return new_transition

    def get_cached_state(self) -> torch.Tensor | None:
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    enabled: bool = False
    relative_step: RelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if self.relative_step is None:
            raise RuntimeError("AbsoluteActionsProcessorStep requires a paired relative action processor.")
        cached_state = self.relative_step.get_cached_state()
        if cached_state is None:
            raise RuntimeError("No state has been cached by RelativeActionsProcessorStep.")
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition
        mask = self.relative_step._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_absolute_actions(action, cached_state, mask)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
