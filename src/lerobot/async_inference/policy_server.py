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

"""
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.processor.rename_processor import RenameObservationsProcessorStep
from lerobot.scripts.recording_hil import (
    ACPInferenceConfig,
    _capture_policy_runtime_state,
    _predict_policy_action_with_acp_inference,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.utils.constants import ACTION, OBS_STR

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    BufferedActions,
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None
        self.rename_map: dict[str, str] = {}
        self.action_names: list[str] = []
        self.robot_type: str | None = None
        self.inference_mode: str = "action_chunk"
        self.acp_inference = ACPInferenceConfig()
        self.cond_policy_runtime_state: dict[str, Any] | None = None
        self.uncond_policy_runtime_state: dict[str, Any] | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )
        inference_mode = getattr(policy_specs, "inference_mode", "action_chunk")
        action_names = list(getattr(policy_specs, "action_names", []))
        # 兼容原有 chunk/单步模式，以及新增的本地 buffer 补货模式。
        if inference_mode not in {"action_chunk", "select_action", "select_action_buffered"}:
            raise ValueError(f"Unsupported inference_mode: {inference_mode}")
        if inference_mode in {"select_action", "select_action_buffered"} and not action_names:
            raise ValueError("select_action mode requires action_names from the robot client.")

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self.action_names = action_names
        self.robot_type = getattr(policy_specs, "robot_type", None)
        self.inference_mode = inference_mode
        self.acp_inference = ACPInferenceConfig(
            enable=getattr(policy_specs, "acp_enable", False),
            use_cfg=getattr(policy_specs, "acp_use_cfg", False),
            cfg_beta=getattr(policy_specs, "acp_cfg_beta", 1.0),
        )

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        preprocessor_overrides = {"device_processor": device_override}
        if policy_specs.rename_map:
            # 客户端显式传入时才覆盖模型目录里的 rename 配置。
            # 这样可以避免把已保存的映射清空。
            preprocessor_overrides["rename_observations_processor"] = {
                "rename_map": policy_specs.rename_map
            }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides={"device_processor": device_override},
        )
        self.rename_map = self._extract_rename_map(self.preprocessor)
        self.logger.info("Loaded preprocessor rename_map: %s", self.rename_map)
        self._reset_policy_runtime_state()

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def _reset_policy_runtime_state(self) -> None:
        """重置 policy 与 processor 的推理缓存，和 human-inloop 释放接管时保持一致。"""
        # HIL 释放人工接管后，原版会重置 policy/preprocessor/postprocessor。
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()
        if self.preprocessor is not None:
            self.preprocessor.reset()
        if self.postprocessor is not None:
            self.postprocessor.reset()

        self.cond_policy_runtime_state = None
        self.uncond_policy_runtime_state = None
        if self.policy is not None and self.acp_inference.enable and self.acp_inference.use_cfg:
            self.cond_policy_runtime_state = _capture_policy_runtime_state(self.policy)
            self.uncond_policy_runtime_state = _capture_policy_runtime_state(self.policy)

    def _extract_rename_map(
        self, preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None
    ) -> dict[str, str]:
        """读取 policy preprocessor 中保存的 observation 重命名关系。"""
        # 从已加载的预处理器中读取图像重命名关系。
        # 这个映射会供 raw observation resize 阶段使用。
        if preprocessor is None:
            return {}
        for step in preprocessor.steps:
            if isinstance(step, RenameObservationsProcessorStep):
                return dict(step.rename_map)
        return {}

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            if self.inference_mode == "select_action":
                action_payload = self._predict_select_action(obs)
            elif self.inference_mode == "select_action_buffered":
                action_payload = self._predict_select_action_buffered(obs)
            else:
                action_payload = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_payload)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if self.inference_mode in {"select_action", "select_action_buffered"}:
            # HIL 对齐模式不做 async 的相似帧过滤；原版 human-inloop 每轮都会用当前观测推理。
            if self.observation_queue.full():
                _ = self.observation_queue.get_nowait()
            self.observation_queue.put(obs)
            return True

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _dataset_features_for_action(self) -> dict[str, dict]:
        """构造 make_robot_action 需要的最小 action feature 描述。"""
        # 原版 human-inloop 通过 dataset.features 映射 action 维度；远端纯推理没有 dataset，
        # 因此这里用客户端传来的 robot.action_features 顺序构造等价的 action names。
        return {
            ACTION: {
                "dtype": "float32",
                "shape": [len(self.action_names)],
                "names": self.action_names,
            }
        }

    def _postprocess_policy_action(self, action_tensor: PolicyAction) -> dict[str, float]:
        """把单步 policy action tensor 转成 robot.send_action 需要的 dict。"""
        return make_robot_action(action_tensor, self._dataset_features_for_action())

    def _get_select_action_queue(self, runtime_state: dict[str, Any] | None = None):
        """读取 select_action 已经缓存好的动作队列。"""
        # 不同 policy 的缓存实现不同：pi0/pi05 常用 `_action_queue`，smolvla 使用 `_queues[action]`。
        state_owner: Any = runtime_state if runtime_state is not None else self.policy
        if state_owner is None:
            return None

        if isinstance(state_owner, dict):
            action_queue = state_owner.get("_action_queue")
            if action_queue is not None:
                return action_queue
            queues = state_owner.get("_queues")
            if isinstance(queues, dict):
                return queues.get(ACTION)
            return None

        action_queue = getattr(state_owner, "_action_queue", None)
        if action_queue is not None:
            return action_queue
        queues = getattr(state_owner, "_queues", None)
        if isinstance(queues, dict):
            return queues.get(ACTION)
        return None

    def _get_available_buffered_select_actions(self, runtime_state: dict[str, Any] | None = None) -> int:
        """返回当前 observation 还剩多少个无需再次推理的后续动作。"""
        action_queue = self._get_select_action_queue(runtime_state)
        return len(action_queue) if action_queue is not None else 0

    def _pop_buffered_select_action(self, runtime_state: dict[str, Any] | None = None) -> PolicyAction | None:
        """弹出一个已经在队列里的原始 action，避免重复触发新的 chunk 推理。"""
        action_queue = self._get_select_action_queue(runtime_state)
        if action_queue is None or len(action_queue) == 0:
            return None
        return action_queue.popleft()

    def _predict_select_action(self, observation_t: TimedObservation) -> dict[str, float]:
        """按原版 human-inloop 的 predict_action/select_action 路径预测单步动作。"""
        if observation_t.should_reset_policy():
            self.logger.info("Resetting policy runtime state before observation #%s", observation_t.get_timestep())
            self._reset_policy_runtime_state()

        raw_observation = dict(observation_t.get_observation())
        task = raw_observation.pop("task", "")

        # 先构造和 dataset frame 同名的 observation，再交给原版 ACP/predict_action helper。
        observation_frame = build_dataset_frame(self.lerobot_features, raw_observation, prefix=OBS_STR)
        action_tensor = _predict_policy_action_with_acp_inference(
            observation_frame=observation_frame,
            policy=self.policy,
            device=torch.device(self.device),
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=self.policy.config.use_amp,
            task=task,
            robot_type=self.robot_type,
            acp_inference=self.acp_inference,
            cond_runtime_state=self.cond_policy_runtime_state,
            uncond_runtime_state=self.uncond_policy_runtime_state,
        )
        action = make_robot_action(action_tensor, self._dataset_features_for_action())
        self.last_processed_obs = observation_t
        self.logger.info(
            "Select action #%s generated | action_dim=%d",
            observation_t.get_timestep(),
            len(action),
        )
        return action

    def _predict_select_action_buffered(self, observation_t: TimedObservation) -> BufferedActions:
        """按原版 select_action 的队列边界，一次返回当前 observation 对应的一整段动作。"""
        if observation_t.should_reset_policy():
            self.logger.info("Resetting policy runtime state before observation #%s", observation_t.get_timestep())
            self._reset_policy_runtime_state()

        raw_observation = dict(observation_t.get_observation())
        task = raw_observation.pop("task", "")

        # 第一拍严格复用原版 HIL helper，保证 preprocess / ACP / CFG / postprocess 完全一致。
        observation_frame = build_dataset_frame(self.lerobot_features, raw_observation, prefix=OBS_STR)
        action_tensors: list[PolicyAction] = [
            _predict_policy_action_with_acp_inference(
                observation_frame=observation_frame,
                policy=self.policy,
                device=torch.device(self.device),
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                use_amp=self.policy.config.use_amp,
                task=task,
                robot_type=self.robot_type,
                acp_inference=self.acp_inference,
                cond_runtime_state=self.cond_policy_runtime_state,
                uncond_runtime_state=self.uncond_policy_runtime_state,
            )
        ]

        # 后续动作直接从本次 select_action 已经生成好的内部队列里取，避免重复跑 preprocess。
        # 这里按“原版队列耗尽”为边界，不截断成更短的 chunk，避免破坏 select_action 语义。
        if self.acp_inference.enable and self.acp_inference.use_cfg:
            remaining = min(
                self._get_available_buffered_select_actions(self.cond_policy_runtime_state),
                self._get_available_buffered_select_actions(self.uncond_policy_runtime_state),
            )
            for _ in range(remaining):
                cond_action = self._pop_buffered_select_action(self.cond_policy_runtime_state)
                uncond_action = self._pop_buffered_select_action(self.uncond_policy_runtime_state)
                if cond_action is None or uncond_action is None:
                    break
                processed_cond = self.postprocessor(cond_action)
                processed_uncond = self.postprocessor(uncond_action)
                action_tensors.append(
                    processed_uncond + self.acp_inference.cfg_beta * (processed_cond - processed_uncond)
                )
        else:
            remaining = self._get_available_buffered_select_actions()
            for _ in range(remaining):
                queued_action = self._pop_buffered_select_action()
                if queued_action is None:
                    break
                action_tensors.append(self.postprocessor(queued_action))

        buffered_actions = [
            TimedAction(
                timestamp=observation_t.get_timestamp() + i * self.config.environment_dt,
                timestep=observation_t.get_timestep() + i,
                action=self._postprocess_policy_action(action_tensor),
            )
            for i, action_tensor in enumerate(action_tensors)
        ]
        self.last_processed_obs = observation_t
        self.logger.info(
            "Buffered select_action #%s generated | buffered_actions=%d",
            observation_t.get_timestep(),
            len(buffered_actions),
        )
        return BufferedActions(
            observation_timestep=observation_t.get_timestep(),
            actions=buffered_actions,
        )

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
            rename_map=self.rename_map,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_inference
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
