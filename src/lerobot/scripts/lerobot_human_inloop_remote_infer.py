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

"""通过远端 async policy server 做 human-in-loop 真机推理。

这个脚本只负责真机控制与远端推理，不创建 LeRobotDataset，也不保存任何帧。
"""

import logging
import pickle  # nosec
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Any, TypeVar

import grpc
import rerun as rr

from lerobot.async_inference.configs import get_aggregate_function
from lerobot.async_inference.helpers import (
    BufferedActions,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    map_robot_keys_to_lerobot_features,
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import ImageBorderConfig, RobotAction, make_default_processors
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_piper_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    piper_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.scripts.recording_hil import (
    INTERVENTION_STATE_ACTIVE,
    INTERVENTION_STATE_POLICY,
    INTERVENTION_STATE_RELEASE,
    ACPInferenceConfig,
    PolicySyncDualArmExecutor,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_piper_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    piper_leader,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.control_utils import init_keyboard_listener, sanity_check_bimanual_piper_pair
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

T = TypeVar("T")

SUPPORTED_REMOTE_INFERENCE_MODES = {
    "select_action",
    "select_action_buffered",
    "select_action_buffered_async",
}


@dataclass
class HumanInloopRemoteInferConfig:
    """远端 HIL 纯推理配置，保留 robot/teleop 控制项，移除 dataset 录制项。"""

    robot: RobotConfig
    pretrained_name_or_path: str
    policy_type: str = "pi05"
    task: str = ""
    server_address: str = "localhost:8080"
    actions_per_chunk: int = 50
    policy_device: str = "cuda"
    client_device: str = "cpu"
    inference_mode: str = "select_action_buffered"
    chunk_size_threshold: float = 0.5
    aggregate_fn_name: str = "conservative"
    fps: int = 30
    control_time_s: float | None = None
    teleop: TeleoperatorConfig | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    image_border: ImageBorderConfig = field(default_factory=ImageBorderConfig)
    play_sounds: bool = True
    policy_sync_to_teleop: bool = True
    policy_sync_parallel: bool = True
    intervention_state_machine_enabled: bool = True
    intervention_toggle_key: str = "i"
    acp_inference: ACPInferenceConfig = field(default_factory=ACPInferenceConfig)
    communication_retry_timeout_s: float = 2.0
    communication_retry_interval_s: float = 0.1
    debug_log_queue_size: bool = False

    def __post_init__(self):
        """校验远端推理所需的最小配置，避免真机启动后才报错。"""
        if self.teleop is None:
            raise ValueError("`lerobot-human-inloop-remote-infer` requires `teleop` config.")
        if not self.pretrained_name_or_path:
            raise ValueError("`pretrained_name_or_path` cannot be empty.")
        if self.actions_per_chunk <= 0:
            raise ValueError("`actions_per_chunk` must be positive.")
        if self.inference_mode not in SUPPORTED_REMOTE_INFERENCE_MODES:
            raise ValueError(
                f"`inference_mode` must be one of {sorted(SUPPORTED_REMOTE_INFERENCE_MODES)}, "
                f"got {self.inference_mode!r}."
            )
        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError("`chunk_size_threshold` must be between 0 and 1.")
        get_aggregate_function(self.aggregate_fn_name)
        if self.fps <= 0:
            raise ValueError("`fps` must be positive.")
        if not self.intervention_toggle_key or len(self.intervention_toggle_key) != 1:
            raise ValueError("`intervention_toggle_key` must be a single character.")
        if self.communication_retry_timeout_s < 0:
            raise ValueError("`communication_retry_timeout_s` must be >= 0.")
        if self.communication_retry_interval_s <= 0:
            raise ValueError("`communication_retry_interval_s` must be > 0.")
        self.acp_inference.validate()

    @property
    def environment_dt(self) -> float:
        """返回单个控制周期的秒数。"""
        return 1 / self.fps


class RemotePolicyActionClient:
    """本地机器人侧 gRPC 客户端，按原版 select_action 语义整段补货并本地逐步消费。"""

    def __init__(self, cfg: HumanInloopRemoteInferConfig, robot: Robot):
        """基于本地 robot schema 构造远端 policy 所需的特征描述。"""
        self.cfg = cfg
        self.inference_mode = getattr(cfg, "inference_mode", "select_action_buffered")
        self.aggregate_fn = get_aggregate_function(
            getattr(cfg, "aggregate_fn_name", "conservative")
        )
        self.action_names = list(robot.action_features)
        self.policy_config = RemotePolicyConfig(
            policy_type=cfg.policy_type,
            pretrained_name_or_path=cfg.pretrained_name_or_path,
            lerobot_features=map_robot_keys_to_lerobot_features(robot),
            actions_per_chunk=cfg.actions_per_chunk,
            device=cfg.policy_device,
            action_names=self.action_names,
            robot_type=robot.robot_type,
            inference_mode=self.inference_mode,
            acp_enable=cfg.acp_inference.enable,
            acp_use_cfg=cfg.acp_inference.use_cfg,
            acp_cfg_beta=cfg.acp_inference.cfg_beta,
            acp_velocity_cfg=cfg.acp_inference.velocity_cfg,
        )
        self.channel = grpc.insecure_channel(
            cfg.server_address, grpc_channel_options(initial_backoff=f"{cfg.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self._running = False
        self.latest_action = -1
        self.reset_policy_on_next_observation = False
        self._buffered_actions: deque[TimedAction] = deque()
        self._buffer_lock = threading.Lock()
        self._async_request_lock = threading.Lock()
        self._async_request_in_flight = False
        self._request_generation = 0
        self._async_thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """判断当前 gRPC 客户端是否仍在运行。"""
        return self._running

    def start(self) -> bool:
        """连接远端 policy server，并发送模型路径、动作维度和机器人特征。"""
        try:
            self.stub.Ready(services_pb2.Empty())
            policy_setup = services_pb2.PolicySetup(data=pickle.dumps(self.policy_config))
            logging.info("Sending remote policy instructions to %s", self.cfg.server_address)
            logging.info("Remote policy action names: %s", self.action_names)
            self.stub.SendPolicyInstructions(policy_setup)
            self._running = True
            return True
        except grpc.RpcError as error:
            logging.error("Failed to connect to remote policy server: %s", error)
            return False

    def stop(self) -> None:
        """停止本地客户端并关闭 gRPC channel。"""
        self._running = False
        self.channel.close()
        if self._async_thread is not None and self._async_thread.is_alive():
            # 退出时短暂等待后台请求线程收尾，避免关闭 channel 后仍继续合并旧动作。
            self._async_thread.join(timeout=1.0)

    def request_policy_reset(self) -> None:
        """标记下一次远端推理前重置 policy 运行态，并清空本地动作缓存。"""
        self.reset_policy_on_next_observation = True
        with self._buffer_lock:
            self._buffered_actions.clear()
        with self._async_request_lock:
            # bump generation 后，旧的后台响应即使回来也会被丢弃。
            self._request_generation += 1

    def send_observation(self, observation: RawObservation, task: str) -> bool:
        """把当前 observation 和 task 打包成 TimedObservation 发给远端。"""
        if not self.running:
            raise RuntimeError("Remote policy client is not running.")

        raw_observation = dict(observation)
        raw_observation["task"] = task

        timed_observation = TimedObservation(
            timestamp=time.time(),
            timestep=max(self.latest_action + 1, 0),
            observation=raw_observation,
            must_go=True,
            reset_policy=self.reset_policy_on_next_observation,
        )
        try:
            # 将 observation 发到远端 VLA 服务，远端按指定 inference_mode 推理动作。
            observation_iterator = send_bytes_in_chunks(
                pickle.dumps(timed_observation),
                services_pb2.Observation,
                log_prefix="[REMOTE_INFER] Observation",
                silent=True,
            )
            self.stub.SendObservations(observation_iterator)
            self.reset_policy_on_next_observation = False
            return True
        except grpc.RpcError as error:
            logging.error("Error sending observation #%s: %s", timed_observation.get_timestep(), error)
            return False

    def get_policy_actions(self) -> BufferedActions | None:
        """同步等待远端返回当前 observation 对应的一段 buffered actions。"""
        try:
            actions = self.stub.GetActions(services_pb2.Empty())
            if len(actions.data) == 0:
                return None
            action_payload = pickle.loads(actions.data)  # nosec
            if isinstance(action_payload, BufferedActions):
                return action_payload
            if isinstance(action_payload, list):
                # 兼容旧服务端直接返回 list[TimedAction] / list[dict] 的场景。
                buffered_actions = []
                for offset, action in enumerate(action_payload):
                    if isinstance(action, TimedAction):
                        buffered_actions.append(action)
                    else:
                        buffered_actions.append(
                            TimedAction(
                                timestamp=time.time() + offset * self.cfg.environment_dt,
                                timestep=max(self.latest_action + 1, 0) + offset,
                                action=action,
                            )
                        )
                return BufferedActions(
                    observation_timestep=max(self.latest_action + 1, 0),
                    actions=buffered_actions,
                )
            if isinstance(action_payload, dict):
                return BufferedActions(
                    observation_timestep=max(self.latest_action + 1, 0),
                    actions=[
                        TimedAction(
                            timestamp=time.time(),
                            timestep=max(self.latest_action + 1, 0),
                            action=action_payload,
                        )
                    ],
                )
            raise TypeError(f"Unsupported remote action payload type: {type(action_payload)}")
        except grpc.RpcError as error:
            logging.error("Error receiving remote policy actions: %s", error)
            return None
        except Exception as error:
            logging.error("Error decoding remote policy actions: %s", error)
            return None

    def _refill_action_buffer(self, observation: RawObservation, task: str) -> bool:
        """当本地 buffer 为空时，用当前 observation 向远端整段补货。"""
        if not self.send_observation(observation, task):
            return False
        buffered_actions = self.get_policy_actions()
        if buffered_actions is None:
            return False
        with self._buffer_lock:
            self._buffered_actions = deque(buffered_actions.get_actions())
        logging.info(
            "Received buffered remote actions for observation #%s | buffered_actions=%d",
            buffered_actions.get_observation_timestep(),
            len(buffered_actions.get_actions()),
        )
        return len(buffered_actions.get_actions()) > 0

    def _is_async_mode(self) -> bool:
        """判断当前是否启用提前补货和新旧动作融合。"""
        return self.inference_mode == "select_action_buffered_async"

    def _buffer_size(self) -> int:
        """读取本地待执行动作数量。"""
        with self._buffer_lock:
            return len(self._buffered_actions)

    def _async_request_running(self) -> bool:
        """判断是否已有后台补货请求在进行中。"""
        with self._async_request_lock:
            return self._async_request_in_flight

    def _ready_to_send_observation(self) -> bool:
        """根据 chunk_size_threshold 判断是否应该提前发送新 observation。"""
        buffer_size = self._buffer_size()
        return buffer_size / self.cfg.actions_per_chunk <= self.cfg.chunk_size_threshold

    def _aggregate_action_values(
        self,
        old_action: RobotAction,
        new_action: RobotAction,
    ) -> RobotAction:
        """对同一个 timestep 的新旧 dict action 做逐关节融合。"""
        fused_action: RobotAction = {}
        # 逐 key 融合复用原生 async 权重，同时保留 send_action 需要的 dict 结构。
        for key in old_action.keys() | new_action.keys():
            if key not in old_action:
                fused_action[key] = new_action[key]
            elif key not in new_action:
                fused_action[key] = old_action[key]
            else:
                fused_action[key] = self.aggregate_fn(old_action[key], new_action[key])
        return fused_action

    def _merge_buffered_actions(self, incoming_actions: list[TimedAction]) -> None:
        """按 timestep 合并后台动作，重叠位置做 temporal ensemble。"""
        if len(incoming_actions) == 0:
            return

        with self._buffer_lock:
            current_actions = {
                action.get_timestep(): action
                for action in self._buffered_actions
                if action.get_timestep() > self.latest_action
            }

            for incoming_action in incoming_actions:
                timestep = incoming_action.get_timestep()
                if timestep <= self.latest_action:
                    continue

                current_action = current_actions.get(timestep)
                if current_action is None:
                    current_actions[timestep] = incoming_action
                    continue

                old_action = current_action.get_action()
                new_action = incoming_action.get_action()
                if not isinstance(old_action, dict) or not isinstance(new_action, dict):
                    raise TypeError("select_action_buffered_async expects dict robot actions.")

                # 同一未来 timestep 被新旧 observation 都预测到时，做平滑融合。
                current_actions[timestep] = TimedAction(
                    timestamp=incoming_action.get_timestamp(),
                    timestep=timestep,
                    action=self._aggregate_action_values(old_action, new_action),
                )

            self._buffered_actions = deque(
                current_actions[timestep] for timestep in sorted(current_actions)
            )

        logging.info(
            "Merged async remote actions | incoming_actions=%d | buffer_size=%d",
            len(incoming_actions),
            self._buffer_size(),
        )

    def _async_refill_action_buffer(
        self,
        observation: RawObservation,
        task: str,
        request_generation: int,
    ) -> None:
        """后台发送 observation 并接收动作，回来后和本地未来动作队列融合。"""
        try:
            if not self.send_observation(observation, task):
                return
            buffered_actions = self.get_policy_actions()
            if buffered_actions is None:
                return
            with self._async_request_lock:
                if request_generation != self._request_generation:
                    logging.info("Discarding stale async remote actions after policy reset.")
                    return
            self._merge_buffered_actions(buffered_actions.get_actions())
        finally:
            with self._async_request_lock:
                self._async_request_in_flight = False

    def _maybe_start_async_refill(self, observation: RawObservation, task: str) -> None:
        """buffer 低于阈值时异步请求新动作，主循环不等待远端推理。"""
        if not self._is_async_mode() or not self._ready_to_send_observation():
            return

        with self._async_request_lock:
            if self._async_request_in_flight:
                return
            self._async_request_in_flight = True
            request_generation = self._request_generation

        # observation 复制一份交给后台线程，避免主循环后续改动同一个 dict。
        async_observation = dict(observation)
        self._async_thread = threading.Thread(
            target=self._async_refill_action_buffer,
            args=(async_observation, task, request_generation),
            daemon=True,
        )
        self._async_thread.start()

    def predict_action(self, observation: RawObservation, task: str) -> RobotAction | None:
        """按 select_action 队列边界补货，再从本地 buffer 逐步取动作执行。"""
        if self._buffer_size() == 0:
            if self._is_async_mode() and self._async_request_running():
                # 后台推理已在进行中，本周期不再发第二个同步请求，避免抢响应。
                return None
            if not self._refill_action_buffer(observation, task):
                return None

        self._maybe_start_async_refill(observation, task)
        with self._buffer_lock:
            timed_action = self._buffered_actions.popleft()
        self.latest_action = timed_action.get_timestep()
        action = timed_action.get_action()
        if not isinstance(action, dict):
            raise TypeError(f"Expected robot action dict, got {type(action)}")
        return action


def _run_with_connection_retry(
    action_name: str,
    fn: Callable[[], T],
    timeout_s: float,
    interval_s: float,
) -> T:
    """对真机通信调用做短暂重试，降低串口/相机瞬时错误的影响。"""
    # 真机串口/相机偶发抖动时短暂重试，避免一次瞬时通信错误直接中断推理。
    deadline_t = time.perf_counter() + timeout_s
    first_error: ConnectionError | None = None
    while True:
        try:
            return fn()
        except ConnectionError as error:
            if first_error is None:
                first_error = error
                logging.warning(
                    "%s failed with transient communication error; retrying for up to %.2fs (%s)",
                    action_name,
                    timeout_s,
                    error,
                )
            if timeout_s <= 0.0 or time.perf_counter() >= deadline_t:
                raise
            time.sleep(min(interval_s, max(deadline_t - time.perf_counter(), 0.0)))


def _set_teleop_manual_control(teleop: Teleoperator | None, enabled: bool) -> None:
    """切换 leader arm 是否允许人工拖动。"""
    # SO leader 支持 manual_control 时，policy 模式下关闭手动回驱，人工接管时再打开。
    if teleop is not None and hasattr(teleop, "set_manual_control"):
        teleop.set_manual_control(enabled)


def _build_remote_task(task: str, acp_inference: ACPInferenceConfig) -> str:
    """返回发给远端服务端的原始 task 文本。"""
    # 为了对齐原版 human-inloop，ACP 正样本标签和 CFG 两路推理由远端服务端统一处理。
    return task


def _remote_human_inloop_loop(
    *,
    cfg: HumanInloopRemoteInferConfig,
    robot: Robot,
    teleop: Teleoperator | None,
    remote_client: RemotePolicyActionClient,
    events: dict[str, Any],
    policy_sync_executor: PolicySyncDualArmExecutor | None,
    display_compressed_images: bool,
) -> None:
    """执行不写数据集的 HIL 控制循环。"""
    # 纯远程推理也复用边框 processor，保证与重新采集/训练的数据分布一致。
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors(
        image_border=cfg.image_border
    )
    has_teleop = teleop is not None
    intervention_enabled = cfg.intervention_state_machine_enabled and has_teleop
    intervention_state = INTERVENTION_STATE_POLICY
    zero_policy_action = dict.fromkeys(remote_client.action_names, 0.0)
    last_teleop_action: RobotAction | None = None
    task = _build_remote_task(cfg.task, cfg.acp_inference)
    logging.info(
        "Remote inference task text:\n%s\nACP inference: enable=%s use_cfg=%s "
        "velocity_cfg=%s cfg_beta=%s",
        task,
        cfg.acp_inference.enable,
        cfg.acp_inference.use_cfg,
        cfg.acp_inference.velocity_cfg,
        cfg.acp_inference.cfg_beta,
    )
    start_t = time.perf_counter()

    if intervention_enabled:
        _set_teleop_manual_control(teleop, False)

    while not events["stop_recording"]:
        loop_start = time.perf_counter()
        if cfg.control_time_s is not None and loop_start - start_t >= cfg.control_time_s:
            break
        if events["exit_early"]:
            events["exit_early"] = False
            break

        if events.get("toggle_intervention", False):
            events["toggle_intervention"] = False
            if intervention_enabled:
                if intervention_state == INTERVENTION_STATE_POLICY:
                    _set_teleop_manual_control(teleop, True)
                    intervention_state = INTERVENTION_STATE_ACTIVE
                    logging.info("Intervention enabled: teleop actions override remote policy.")
                else:
                    _set_teleop_manual_control(teleop, False)
                    remote_client.request_policy_reset()
                    intervention_state = INTERVENTION_STATE_RELEASE
                    logging.info("Intervention released: returning control to remote policy.")
            else:
                logging.info("Intervention toggle ignored because no teleoperator is configured.")

        obs = _run_with_connection_retry(
            "robot.get_observation",
            robot.get_observation,
            cfg.communication_retry_timeout_s,
            cfg.communication_retry_interval_s,
        )
        obs_processed = robot_observation_processor(obs)

        act_processed_policy = None
        if not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE):
            # 远端服务端内部按原版 preprocessor -> select_action -> postprocessor 执行。
            act_processed_policy = remote_client.predict_action(obs_processed, task)

        act_processed_teleop = None
        if has_teleop:
            teleop_action = _run_with_connection_retry(
                "teleop.get_action",
                teleop.get_action,
                cfg.communication_retry_timeout_s,
                cfg.communication_retry_interval_s,
            )
            act_processed_teleop = teleop_action_processor((teleop_action, obs))
            last_teleop_action = act_processed_teleop

        if intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE:
            action_values = act_processed_teleop or last_teleop_action or act_processed_policy or zero_policy_action
        else:
            action_values = act_processed_policy if act_processed_policy is not None else act_processed_teleop

        if action_values is None:
            precise_sleep(max(cfg.environment_dt - (time.perf_counter() - loop_start), 0.0))
            continue

        robot_action_to_send = robot_action_processor((action_values, obs))
        selected_from_policy = act_processed_policy is not None and action_values is act_processed_policy
        if policy_sync_executor is not None and selected_from_policy:
            _sent_action = _run_with_connection_retry(
                "policy_sync_executor.send_action",
                lambda robot_action_to_send=robot_action_to_send: policy_sync_executor.send_action(
                    robot_action_to_send
                ),
                cfg.communication_retry_timeout_s,
                cfg.communication_retry_interval_s,
            )
        else:
            _sent_action = _run_with_connection_retry(
                "robot.send_action",
                lambda robot_action_to_send=robot_action_to_send: robot.send_action(robot_action_to_send),
                cfg.communication_retry_timeout_s,
                cfg.communication_retry_interval_s,
            )

        if cfg.display_data:
            log_rerun_data(
                observation=obs_processed,
                action=action_values,
                compress_images=display_compressed_images,
            )

        if intervention_state == INTERVENTION_STATE_RELEASE:
            intervention_state = INTERVENTION_STATE_POLICY
        precise_sleep(max(cfg.environment_dt - (time.perf_counter() - loop_start), 0.0))


@parser.wrap()
def human_inloop_remote_infer(cfg: HumanInloopRemoteInferConfig) -> None:
    """命令入口：连接 robot/teleop、启动远端客户端、运行纯推理 HIL 循环。"""
    init_logging()
    sanity_check_bimanual_piper_pair(cfg.robot, cfg.teleop)
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="human_inloop_remote_infer", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    remote_client: RemotePolicyActionClient | None = None
    policy_sync_executor: PolicySyncDualArmExecutor | None = None
    listener = None

    try:
        robot.connect()
        if teleop is not None:
            teleop.connect()
        if cfg.policy_sync_to_teleop:
            if teleop is None:
                raise ValueError("`policy_sync_to_teleop=true` requires `teleop` config.")
            # 复用 human-inloop-record 的同步逻辑，让 policy 动作同时回驱 leader arm，方便接管。
            policy_sync_executor = PolicySyncDualArmExecutor(
                robot=robot,
                teleop=teleop,
                parallel_dispatch=cfg.policy_sync_parallel,
            )

        remote_client = RemotePolicyActionClient(cfg, robot)
        if not remote_client.start():
            raise RuntimeError("Failed to start remote policy client.")

        listener, events = init_keyboard_listener(intervention_toggle_key=cfg.intervention_toggle_key)
        logging.info(
            "Human-in-loop remote inference is running. Press '%s' to toggle takeover, right arrow to "
            "exit current loop, Esc to stop.",
            cfg.intervention_toggle_key,
        )
        _remote_human_inloop_loop(
            cfg=cfg,
            robot=robot,
            teleop=teleop,
            remote_client=remote_client,
            events=events,
            policy_sync_executor=policy_sync_executor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        log_say("Stop human-in-loop remote inference", cfg.play_sounds, blocking=True)
        if remote_client is not None:
            remote_client.stop()
        if policy_sync_executor is not None:
            policy_sync_executor.shutdown()
        if listener and hasattr(listener, "stop"):
            listener.stop()
        if cfg.display_data:
            rr.rerun_shutdown()
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()
        log_say("Exiting", cfg.play_sounds)


def main():
    """注册第三方插件后交给 draccus/parser 解析 CLI 参数。"""
    register_third_party_plugins()
    human_inloop_remote_infer()


if __name__ == "__main__":
    main()
