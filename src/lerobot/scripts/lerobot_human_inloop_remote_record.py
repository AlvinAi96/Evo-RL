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

"""通过远端 async policy server 做 human-in-loop 真机录制。"""

import logging
from dataclasses import dataclass
from pathlib import Path

from lerobot.configs import parser
from lerobot.scripts.lerobot_human_inloop_record import _HumanInloopFailureResetController
from lerobot.scripts.lerobot_human_inloop_remote_infer import RemotePolicyActionClient
from lerobot.scripts.lerobot_record import RecordConfig, record
from lerobot.utils.import_utils import register_third_party_plugins


def _infer_remote_policy_version(pretrained_name_or_path: str, policy_type: str) -> str:
    """为采集元数据推断一个稳定的远端 policy 版本标识。"""
    normalized = pretrained_name_or_path.rstrip("/").strip()
    if normalized:
        return Path(normalized).name or normalized
    return policy_type


@dataclass
class HumanInloopRemoteRecordConfig(RecordConfig):
    """远端 HIL 录制配置，沿用 RecordConfig 的 dataset/episode 写入行为。"""

    pretrained_name_or_path: str = ""
    policy_type: str = "pi05"
    server_address: str = "localhost:8080"
    actions_per_chunk: int = 50
    policy_device: str = "cuda"
    client_device: str = "cpu"
    chunk_size_threshold: float = 0.5
    aggregate_fn_name: str = "conservative"
    debug_log_queue_size: bool = False

    def __post_init__(self):
        """补充远端推理字段校验，同时复用原版录制参数校验。"""
        super().__post_init__()
        if self.teleop is None:
            raise ValueError("`lerobot-human-inloop-remote-record` requires `teleop` config.")
        if not self.pretrained_name_or_path:
            raise ValueError("`pretrained_name_or_path` cannot be empty.")
        if self.actions_per_chunk <= 0:
            raise ValueError("`actions_per_chunk` must be positive.")
        if not self.server_address:
            raise ValueError("`server_address` cannot be empty.")

    @property
    def environment_dt(self) -> float:
        """返回远端 buffered action 的时间步长，与录制 fps 严格一致。"""
        return 1 / self.dataset.fps


@parser.wrap()
def human_inloop_remote_record(cfg: HumanInloopRemoteRecordConfig):
    """命令入口：复用原版 record()，只把本地 policy 后端替换成远端 gRPC 服务。"""
    if cfg.policy is not None:
        raise ValueError(
            "`lerobot-human-inloop-remote-record` does not use `--policy.path`. "
            "Please pass remote model info with `--pretrained_name_or_path` instead."
        )

    cfg.policy_sync_to_teleop = True
    cfg.intervention_state_machine_enabled = True
    cfg.enable_episode_outcome_labeling = True
    cfg.default_episode_success = "failure"
    cfg.enable_collector_policy_id = True
    if cfg.collector_policy_id_policy is None:
        cfg.collector_policy_id_policy = _infer_remote_policy_version(
            cfg.pretrained_name_or_path, cfg.policy_type
        )

    failure_reset_controller = _HumanInloopFailureResetController(cfg)
    cfg._on_record_connected = failure_reset_controller.on_record_connected
    cfg._on_record_episode_outcome = failure_reset_controller.on_episode_outcome

    def build_remote_policy_runtime(robot):
        """基于当前 robot schema 创建远端 policy 客户端。"""
        return RemotePolicyActionClient(cfg, robot)

    # 将远端 policy runtime 注入到 record()，从而复用原版 dataset 写入和 episode 边界逻辑。
    cfg._build_policy_runtime = build_remote_policy_runtime

    logging.info(
        "Human-in-loop remote recording is enabled. Press '%s' to toggle takeover. "
        "Press '%s' to mark success and end, '%s' to mark failure and end. "
        "Recorded `action` is the executed action. "
        "Policy output (when policy is enabled) is stored in `complementary_info.policy_action`. "
        "Collector source is stored in `complementary_info.collector_policy_id`. "
        "Remote policy server: %s | policy_type=%s | policy=%s. "
        "ACP inference: enable=%s use_cfg=%s cfg_beta=%.3f.",
        cfg.intervention_toggle_key,
        cfg.episode_success_key,
        cfg.episode_failure_key,
        cfg.server_address,
        cfg.policy_type,
        cfg.pretrained_name_or_path,
        cfg.acp_inference.enable,
        cfg.acp_inference.use_cfg,
        cfg.acp_inference.cfg_beta,
    )
    return record(cfg)


def main():
    register_third_party_plugins()
    human_inloop_remote_record()


if __name__ == "__main__":
    main()
