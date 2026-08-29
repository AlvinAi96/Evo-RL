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
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pprint import pformat
from queue import Queue
from typing import Any, TypeVar

import grpc
import rerun as rr

from lerobot.async_inference.configs import get_aggregate_function
from lerobot.async_inference.helpers import (
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    map_robot_keys_to_lerobot_features,
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import RobotAction, make_default_processors
from lerobot.rl.acp_tags import build_acp_tagged_task
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
    chunk_size_threshold: float = 0.5
    aggregate_fn_name: str = "conservative"
    fps: int = 30
    control_time_s: float | None = None
    teleop: TeleoperatorConfig | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    policy_sync_to_teleop: bool = False
    policy_sync_parallel: bool = True
    intervention_state_machine_enabled: bool = True
    intervention_toggle_key: str = "i"
    acp_inference: ACPInferenceConfig = field(default_factory=ACPInferenceConfig)
    communication_retry_timeout_s: float = 2.0
    communication_retry_interval_s: float = 0.1
    debug_log_queue_size: bool = False

    def __post_init__(self):
        """校验远端推理所需的最小配置，避免真机启动后才报错。"""
        if not self.pretrained_name_or_path:
            raise ValueError("`pretrained_name_or_path` cannot be empty.")
        if self.actions_per_chunk <= 0:
            raise ValueError("`actions_per_chunk` must be positive.")
        if self.fps <= 0:
            raise ValueError("`fps` must be positive.")
        if not self.intervention_toggle_key or len(self.intervention_toggle_key) != 1:
            raise ValueError("`intervention_toggle_key` must be a single character.")
        if self.acp_inference.use_cfg:
            raise ValueError(
                "`lerobot-human-inloop-remote-infer` currently supports ACP positive tagging only; "
                "set `acp_inference.use_cfg=false`."
            )
        if self.communication_retry_timeout_s < 0:
            raise ValueError("`communication_retry_timeout_s` must be >= 0.")
        if self.communication_retry_interval_s <= 0:
            raise ValueError("`communication_retry_interval_s` must be > 0.")
        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @property
    def environment_dt(self) -> float:
        """返回单个控制周期的秒数。"""
        return 1 / self.fps


class RemotePolicyActionClient:
    """本地机器人侧 gRPC 客户端，负责发 observation、收远端 action chunk。"""

    def __init__(self, cfg: HumanInloopRemoteInferConfig, robot: Robot):
        """基于本地 robot schema 构造远端 policy 所需的特征描述。"""
        self.cfg = cfg
        self.action_names = list(robot.action_features)
        self.policy_config = RemotePolicyConfig(
            policy_type=cfg.policy_type,
            pretrained_name_or_path=cfg.pretrained_name_or_path,
            lerobot_features=map_robot_keys_to_lerobot_features(robot),
            actions_per_chunk=cfg.actions_per_chunk,
            device=cfg.policy_device,
        )
        self.channel = grpc.insecure_channel(
            cfg.server_address, grpc_channel_options(initial_backoff=f"{cfg.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.shutdown_event = threading.Event()
        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = cfg.actions_per_chunk
        self.must_go = threading.Event()
        self.must_go.set()

    @property
    def running(self) -> bool:
        """判断当前 gRPC 客户端是否仍在运行。"""
        return not self.shutdown_event.is_set()

    def start(self) -> bool:
        """连接远端 policy server，并发送模型路径、动作维度和机器人特征。"""
        try:
            self.stub.Ready(services_pb2.Empty())
            policy_setup = services_pb2.PolicySetup(data=pickle.dumps(self.policy_config))
            logging.info("Sending remote policy instructions to %s", self.cfg.server_address)
            logging.info("Remote policy action names: %s", self.action_names)
            self.stub.SendPolicyInstructions(policy_setup)
            self.shutdown_event.clear()
            return True
        except grpc.RpcError as error:
            logging.error("Failed to connect to remote policy server: %s", error)
            return False

    def stop(self) -> None:
        """停止本地客户端并关闭 gRPC channel。"""
        self.shutdown_event.set()
        self.channel.close()

    def _aggregate_action_queues(self, incoming_actions: list[TimedAction]) -> None:
        """合并远端新 chunk 和本地未执行 chunk，保留未来时间步的动作。"""
        # 保留 async_inference 的 chunk 聚合逻辑，避免新旧 action chunk 互相覆盖过急。
        future_action_queue = Queue()
        with self.action_queue_lock:
            current_action_queue = {
                action.get_timestep(): action.get_action() for action in self.action_queue.queue
            }

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action
            if new_action.get_timestep() <= latest_action:
                continue
            if new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=self.cfg.aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self) -> None:
        """后台线程循环接收远端 action chunk。"""
        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                timed_actions: list[TimedAction] = pickle.loads(actions_chunk.data)  # nosec
                if self.cfg.client_device != "cpu":
                    for timed_action in timed_actions:
                        timed_action.action = timed_action.get_action().to(self.cfg.client_device)

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                self._aggregate_action_queues(timed_actions)
                self.must_go.set()
            except grpc.RpcError as error:
                if self.running:
                    logging.error("Error receiving remote actions: %s", error)

    def actions_available(self) -> bool:
        """判断本地 action 队列里是否有可执行动作。"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def clear_actions(self) -> None:
        """清空本地 action 队列，用于人工接管和释放接管的边界。"""
        # 接管状态切换时丢弃旧 chunk，避免释放后执行过期的 policy 动作。
        with self.action_queue_lock:
            self.action_queue = Queue()
        self.must_go.set()

    def ready_to_send_observation(self) -> bool:
        """根据剩余 action 比例判断是否需要向远端补发新 observation。"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self.cfg.chunk_size_threshold

    def send_observation(self, observation: RawObservation, task: str) -> bool:
        """把当前 observation 和 task 打包成 TimedObservation 发给远端。"""
        if not self.running:
            raise RuntimeError("Remote policy client is not running.")

        raw_observation = dict(observation)
        raw_observation["task"] = task
        with self.latest_action_lock:
            latest_action = self.latest_action
        with self.action_queue_lock:
            must_go = self.must_go.is_set() and self.action_queue.empty()

        timed_observation = TimedObservation(
            timestamp=time.time(),
            timestep=max(latest_action, 0),
            observation=raw_observation,
            must_go=must_go,
        )
        try:
            # 将当前 observation 分块发到远端 VLA 服务，远端负责预处理和模型推理。
            observation_iterator = send_bytes_in_chunks(
                pickle.dumps(timed_observation),
                services_pb2.Observation,
                log_prefix="[REMOTE_INFER] Observation",
                silent=True,
            )
            self.stub.SendObservations(observation_iterator)
            if must_go:
                self.must_go.clear()
            return True
        except grpc.RpcError as error:
            logging.error("Error sending observation #%s: %s", timed_observation.get_timestep(), error)
            return False

    def pop_policy_action(self) -> RobotAction | None:
        """从本地队列取出一个 policy action，并转成 robot.send_action 需要的 dict。"""
        with self.action_queue_lock:
            if self.action_queue.empty():
                return None
            if self.cfg.debug_log_queue_size:
                logging.info("Remote action queue size: %s", self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()

        action_tensor = timed_action.get_action()
        action = {key: action_tensor[idx].item() for idx, key in enumerate(self.action_names)}
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
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
    """根据 ACP 推理开关生成发给远端 VLA 的最终 task 文本。"""
    # ACP 训练出的策略推理时需要正向 Advantage 标签，远端服务端只看到这个字符串。
    if acp_inference.enable:
        return build_acp_tagged_task(task, is_positive=True)
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
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    has_teleop = teleop is not None
    intervention_enabled = cfg.intervention_state_machine_enabled and has_teleop
    intervention_state = INTERVENTION_STATE_POLICY
    last_teleop_action: RobotAction | None = None
    last_policy_action: RobotAction | None = None
    task = _build_remote_task(cfg.task, cfg.acp_inference)
    logging.info("Remote inference task text:\n%s", task)
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
                    remote_client.clear_actions()
                    intervention_state = INTERVENTION_STATE_ACTIVE
                    logging.info("Intervention enabled: teleop actions override remote policy.")
                else:
                    _set_teleop_manual_control(teleop, False)
                    remote_client.clear_actions()
                    last_policy_action = None
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

        if remote_client.ready_to_send_observation():
            # 每个控制周期只发 observation，不写 dataset frame。
            remote_client.send_observation(obs_processed, task)

        act_processed_policy = None
        if not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE):
            act_processed_policy = remote_client.pop_policy_action()

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

        if act_processed_policy is not None:
            last_policy_action = act_processed_policy

        if intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE:
            action_values = act_processed_teleop or last_teleop_action or last_policy_action
        else:
            action_values = act_processed_policy or last_policy_action

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
    receiver_thread: threading.Thread | None = None
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

        receiver_thread = threading.Thread(target=remote_client.receive_actions, daemon=True)
        receiver_thread.start()

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
        if receiver_thread is not None:
            receiver_thread.join(timeout=2)
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
