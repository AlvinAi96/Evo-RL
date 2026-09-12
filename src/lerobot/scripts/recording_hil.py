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

"""Human-in-loop recording helpers used by `lerobot_record.py`."""

import inspect
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, RobotAction
from lerobot.processor.core import TransitionKey
from lerobot.rl.acp_tags import build_acp_tagged_task
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.control_utils import predict_action


@dataclass
class ACPInferenceConfig:
    enable: bool = False
    use_cfg: bool = False
    cfg_beta: float = 1.0
    velocity_cfg: bool = False

    def validate(self) -> None:
        """校验 ACP 推理配置组合，避免命令看似生效但实际走错路径。"""
        if self.use_cfg and not self.enable:
            raise ValueError("`acp_inference.use_cfg=true` requires `acp_inference.enable=true`.")
        if self.velocity_cfg and not self.use_cfg:
            raise ValueError(
                "`acp_inference.velocity_cfg=true` requires `acp_inference.use_cfg=true`."
            )
        if self.cfg_beta < 0:
            raise ValueError("`acp_inference.cfg_beta` must be >= 0.")


POLICY_RUNTIME_STATE_KEYS = ("_action_queue", "_queues", "_prev_mean")


INTERVENTION_STATE_POLICY = 0.0
INTERVENTION_STATE_ACTIVE = 1.0
INTERVENTION_STATE_RELEASE = 2.0


def _get_torch_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


def _set_torch_rng_state(
    device: torch.device, cpu_state: torch.Tensor, cuda_state: torch.Tensor | None
) -> None:
    torch.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def _clone_runtime_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, deque):
        return deque((_clone_runtime_value(item) for item in value), maxlen=value.maxlen)
    if isinstance(value, dict):
        return {key: _clone_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_runtime_value(item) for item in value)
    return deepcopy(value)


def _capture_policy_runtime_state(policy: PreTrainedPolicy) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key in POLICY_RUNTIME_STATE_KEYS:
        if hasattr(policy, key):
            state[key] = _clone_runtime_value(getattr(policy, key))
    return state


def _restore_policy_runtime_state(policy: PreTrainedPolicy, state: dict[str, Any]) -> None:
    for key, value in state.items():
        setattr(policy, key, _clone_runtime_value(value))


def _predict_policy_action_with_runtime_state(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
    runtime_state: dict[str, Any],
) -> PolicyAction:
    _restore_policy_runtime_state(policy, runtime_state)
    action = predict_action(
        observation=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
    )
    runtime_state.clear()
    runtime_state.update(_capture_policy_runtime_state(policy))
    return action


def _prepare_policy_observation_batch(
    *,
    observation_frame: dict[str, np.ndarray],
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    task: str | None,
    robot_type: str | None,
) -> dict[str, Any]:
    """按指定 task 走完整 preprocessor，得到 policy 可直接消费的 batch。"""
    observation = copy(observation_frame)
    observation = prepare_observation_for_inference(observation, device, task, robot_type)
    return preprocessor(observation)


def _find_preprocessor_step(
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    class_name: str,
) -> Any | None:
    """按类名查找 processor step，避免 recording_hil 强依赖某个 policy 模块。"""
    for step in getattr(preprocessor, "steps", []):
        if step.__class__.__name__ == class_name:
            return step
    return None


def _build_positive_language_batch_from_base(
    *,
    base_batch: dict[str, Any],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    positive_task: str,
) -> dict[str, Any] | None:
    """复用 base batch 的归一化 state，只重新生成 positive prompt 的 tokens/masks。"""
    prompt_step = _find_preprocessor_step(preprocessor, "Pi05PrepareStateTokenizerProcessorStep")
    tokenizer_step = _find_preprocessor_step(preprocessor, "TokenizerProcessorStep")
    if prompt_step is None or tokenizer_step is None or OBS_STATE not in base_batch:
        return None

    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: base_batch[OBS_STATE]},
        TransitionKey.COMPLEMENTARY_DATA: {"task": [positive_task]},
    }
    # 先用 Pi0.5 的 state prompt 规则拼出完整文本，再只跑 tokenizer step。
    transition = prompt_step(transition)
    transition = tokenizer_step(transition)
    positive_observation = transition[TransitionKey.OBSERVATION]

    positive_batch = dict(base_batch)
    positive_batch[OBS_LANGUAGE_TOKENS] = positive_observation[OBS_LANGUAGE_TOKENS]
    positive_batch[OBS_LANGUAGE_ATTENTION_MASK] = positive_observation[OBS_LANGUAGE_ATTENTION_MASK]
    return positive_batch


def _predict_policy_action_with_velocity_cfg(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    conditional_task: str,
    robot_type: str | None,
    cfg_beta: float,
) -> PolicyAction:
    """在 policy 内部每个 flow-matching velocity step 做 CFG 引导。"""
    select_action = getattr(policy, "select_action", None)
    if not callable(select_action) or "cfg_positive_batch" not in inspect.signature(select_action).parameters:
        raise NotImplementedError(
            f"`acp_inference.velocity_cfg=true` is not supported by policy {policy.__class__.__name__}."
        )

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # base batch 保持原始 task，positive batch 只替换语言条件；CFG 在 policy 内、postprocessor 前完成。
        base_batch = _prepare_policy_observation_batch(
            observation_frame=observation_frame,
            device=device,
            preprocessor=preprocessor,
            task=task,
            robot_type=robot_type,
        )
        positive_batch = _build_positive_language_batch_from_base(
            base_batch=base_batch,
            preprocessor=preprocessor,
            positive_task=conditional_task,
        )
        if positive_batch is None:
            positive_batch = _prepare_policy_observation_batch(
                observation_frame=observation_frame,
                device=device,
                preprocessor=preprocessor,
                task=conditional_task,
                robot_type=robot_type,
            )
        action = select_action(
            base_batch,
            cfg_positive_batch=positive_batch,
            cfg_beta=cfg_beta,
        )
        return postprocessor(action)


def _predict_policy_action_with_acp_inference(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
    acp_inference: ACPInferenceConfig,
    cond_runtime_state: dict[str, Any] | None = None,
    uncond_runtime_state: dict[str, Any] | None = None,
) -> PolicyAction:
    acp_inference.validate()
    if not acp_inference.enable:
        return predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )

    conditional_task = build_acp_tagged_task(task, is_positive=True)
    if acp_inference.velocity_cfg and acp_inference.use_cfg:
        return _predict_policy_action_with_velocity_cfg(
            observation_frame=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            conditional_task=conditional_task,
            robot_type=robot_type,
            cfg_beta=acp_inference.cfg_beta,
        )

    if not acp_inference.use_cfg:
        return predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=conditional_task,
            robot_type=robot_type,
        )

    if cond_runtime_state is None or uncond_runtime_state is None:
        raise ValueError("CFG inference requires cond/uncond runtime states.")

    cpu_state, cuda_state = _get_torch_rng_state(device)
    action_cond = _predict_policy_action_with_runtime_state(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=conditional_task,
        robot_type=robot_type,
        runtime_state=cond_runtime_state,
    )
    _set_torch_rng_state(device, cpu_state, cuda_state)
    action_uncond = _predict_policy_action_with_runtime_state(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
        runtime_state=uncond_runtime_state,
    )
    return action_uncond + acp_inference.cfg_beta * (action_cond - action_uncond)


class PolicySyncDualArmExecutor:
    """Broadcast one policy-derived robot action to follower + teleop arm."""

    def __init__(self, robot: Robot, teleop: Teleoperator, parallel_dispatch: bool = True):
        self.robot = robot
        self.teleop = teleop
        self.parallel_dispatch = parallel_dispatch
        self._pool = ThreadPoolExecutor(max_workers=2) if parallel_dispatch else None

    def send_action(self, action: RobotAction) -> RobotAction:
        if self._pool is None:
            sent_action = self.robot.send_action(action)
            self.teleop.send_feedback(action)
            return sent_action

        robot_future = self._pool.submit(self.robot.send_action, action)
        teleop_future = self._pool.submit(self.teleop.send_feedback, action)
        sent_action = robot_future.result()
        teleop_future.result()
        return sent_action

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
