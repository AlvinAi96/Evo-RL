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

from .converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from .core import RobotAction, RobotObservation
from .image_border_processor import ImageBorderConfig, ImageBorderProcessorStep
from .pipeline import IdentityProcessorStep, RobotProcessorPipeline


def make_default_teleop_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_action_processor


def make_default_robot_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return robot_action_processor


def make_default_robot_observation_processor(
    image_border: ImageBorderConfig | None = None,
) -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    """构造机器人 observation pipeline；默认不改图，按配置可插入图像边框步骤。"""
    steps = []
    if image_border is not None and image_border.enable and image_border.width_px > 0:
        # 在原始相机图像阶段加边框，后续 dataset/Rerun/policy 都能看到同一份 observation。
        steps.append(
            ImageBorderProcessorStep(
                width_px=image_border.width_px,
                color_rgb=image_border.color_rgb,
                keys=image_border.keys,
            )
        )
    if not steps:
        steps = [IdentityProcessorStep()]

    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=steps,
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


def make_default_processors(image_border: ImageBorderConfig | None = None):
    """构造默认 action/observation processors；图像边框只影响 observation 分支。"""
    teleop_action_processor = make_default_teleop_action_processor()
    robot_action_processor = make_default_robot_action_processor()
    robot_observation_processor = make_default_robot_observation_processor(image_border=image_border)
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)
