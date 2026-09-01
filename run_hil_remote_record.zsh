#!/usr/bin/env zsh
set -euo pipefail

# 中文说明：固定使用 Evo-RL 的 conda Python 和本地 src，避免误用 base 环境或旧版 lerobot。
REPO_DIR="/Users/bytedance/Evo-RL"
PYTHON_BIN="/opt/miniconda3/envs/evo-rl/bin/python"
CONFIG_PATH="${REPO_DIR}/configs/hil_remote_record_so101.yaml"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

cd "${REPO_DIR}"

mode="${1:-check}"

if [[ "${mode}" == "check" ]]; then
  echo "[check] python: ${PYTHON_BIN}"
  "${PYTHON_BIN}" -c 'import lerobot.scripts.lerobot_human_inloop_remote_record as m; print("[check] module:", m.__file__); print("[check] remote args:", m.HumanInloopRemoteRecordConfig.__annotations__)'
  echo "[check] help contains remote args:"
  "${PYTHON_BIN}" -m lerobot.scripts.lerobot_human_inloop_remote_record --help 2>&1 \
    | grep -n -- "pretrained_name_or_path\|policy_type\|server_address\|actions_per_chunk\|policy_device"
  echo "[check] config: ${CONFIG_PATH}"
  # 中文说明：只解析配置文件，不连接机械臂，确认 YAML 可被 HumanInloopRemoteRecordConfig 正确读取。
  "${PYTHON_BIN}" -c 'import draccus; from lerobot.scripts.lerobot_human_inloop_remote_record import HumanInloopRemoteRecordConfig; cfg=draccus.parse(config_class=HumanInloopRemoteRecordConfig, config_path="'${CONFIG_PATH}'", args=[]); print("[check] parse config OK"); print("[check] robot:", cfg.robot.type, cfg.robot.port); print("[check] teleop:", cfg.teleop.type, cfg.teleop.port); print("[check] remote:", cfg.pretrained_name_or_path, cfg.policy_type, cfg.server_address)'
  echo "[check] OK. To really record, run: /bin/zsh ${REPO_DIR}/run_hil_remote_record.zsh run"
elif [[ "${mode}" == "run" ]]; then
  RUN_ROOT="${DATASET_ROOT:-/Users/bytedance/lerobot_dataset_0824_hil_r2_$(date +%Y%m%d_%H%M%S)}"
  echo "[run] dataset.root: ${RUN_ROOT}"
  # 中文说明：真正启动 human-in-loop remote recording，会连接机械臂、摄像头、远端 policy server 并写入数据集。
  exec "${PYTHON_BIN}" -m lerobot.scripts.lerobot_human_inloop_remote_record \
    --config_path="${CONFIG_PATH}" \
    --dataset.root="${RUN_ROOT}"
else
  echo "Usage: /bin/zsh ${REPO_DIR}/run_hil_remote_record.zsh [check|run]" >&2
  exit 2
fi
