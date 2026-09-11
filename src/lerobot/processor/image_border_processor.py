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

from dataclasses import dataclass, field

import numpy as np

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep


@dataclass
class ImageBorderConfig:
    """图像内边框配置，用于在原始相机图像周围添加彩色边框。"""
    enable: bool = False
    width_px: int = 5
    color_rgb: tuple[int, int, int] = (128, 128, 128)
    keys: list[str] = field(default_factory=list)


@dataclass
class ImageBorderProcessorStep(ProcessorStep):
    """在 RobotObservation 的指定图像键上添加彩色内边框。"""

    width_px: int = 5
    color_rgb: tuple[int, int, int] = (128, 128, 128)
    keys: list[str] = field(default_factory=list)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """对观察结果中 keys 列表指定的图像添加边框。"""
        if not self.keys:
            return transition

        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return transition

        new_transition = transition.copy()
        new_obs = dict(observation)

        for key in self.keys:
            if key not in new_obs:
                continue
            img = new_obs[key]
            if img is None:
                continue
            new_obs[key] = self._add_border(img)

        new_transition[TransitionKey.OBSERVATION] = new_obs
        return new_transition

    def _add_border(self, img):
        """在图像内侧添加彩色边框，支持 numpy（H, W, C）和 torch tensor（C, H, W）。"""
        import torch

        w_px = self.width_px
        if isinstance(img, torch.Tensor):
            return self._add_border_tensor(img, w_px)
        else:
            return self._add_border_numpy(np.asarray(img), w_px)

    def _add_border_numpy(self, img: np.ndarray, w_px: int) -> np.ndarray:
        h, w = img.shape[:2]
        w_px = min(w_px, h // 2, w // 2)

        bordered = img.copy()
        color = list(self.color_rgb)

        if img.ndim == 2:
            color_val = int(np.mean(self.color_rgb))
            bordered[:w_px, :] = color_val
            bordered[-w_px:, :] = color_val
            bordered[:, :w_px] = color_val
            bordered[:, -w_px:] = color_val
        else:
            bordered[:w_px, :] = color
            bordered[-w_px:, :] = color
            bordered[:, :w_px] = color
            bordered[:, -w_px:] = color

        return bordered

    def _add_border_tensor(self, img, w_px: int):
        import torch

        # 假设 (C, H, W) 格式
        _, h, w = img.shape
        w_px = min(w_px, h // 2, w // 2)

        bordered = img.clone()
        color = torch.tensor(self.color_rgb, dtype=img.dtype, device=img.device)

        for c in range(min(len(color), img.shape[0])):
            bordered[c, :w_px, :] = color[c]
            bordered[c, -w_px:, :] = color[c]
            bordered[c, :, :w_px] = color[c]
            bordered[c, :, -w_px:] = color[c]

        return bordered

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """添加图像边框不改变特征形状/类型。"""
        return features