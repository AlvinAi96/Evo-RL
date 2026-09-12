# ACP RLinf Alignment

本文记录当前分支对 Advantage-Conditioned Policy 训练侧和推理侧 CFG 的修改背景、配置语义与操作命令。

## 背景

原 Evo-RL ACP 训练会根据 `complementary_info.acp_indicator` 给正负样本分别注入：

- `Task: xxx\nAdvantage: positive`
- `Task: xxx\nAdvantage: negative`

同时对所有样本使用统一 dropout。这样会让 unconditioned prompt 同时混入正负样本，也会训练大量推理侧不用的 negative-conditioned prompt。

本次对齐 RLinf 推荐的训练分布：

- 正样本：大部分使用 `Advantage: positive`，少量通过 dropout 保留 base task。
- 负样本：默认只使用 base task，不再追加 `Advantage: negative`。
- Loss：默认只用成功轨迹产生 loss，失败轨迹的处理由配置决定。

## Prompt 配置

ACP 配置位于 `src/lerobot/configs/train.py` 的 `ACPConfig`。

关键字段：

- `acp.enable`: 是否开启 ACP prompt hook。
- `acp.indicator_field`: 数据集中 0/1 Advantage indicator 字段。
- `acp.indicator_dropout_prob`: 正样本 tag dropout 概率，默认 `0.1`。
- `acp.tag_negative_prompts`: 是否继续给负样本追加 `Advantage: negative`，默认 `false`。

默认行为：

| 样本 | 训练 prompt |
| --- | --- |
| 正样本且未 dropout | `Task: xxx\nAdvantage: positive` |
| 正样本且 dropout | `Task: xxx` |
| 负样本 | `Task: xxx` |

如需恢复旧 Evo-RL 行为，可显式设置：

```bash
--acp.indicator_dropout_prob=0.3 \
--acp.tag_negative_prompts=true
```

## 失败轨迹 Loss 配置

`acp.failure_loss_mode` 控制失败轨迹如何参与 ACP 训练，默认 `mask_loss`。

| 配置 | 行为 |
| --- | --- |
| `mask_loss` | 失败轨迹进入 dataloader 和模型 forward，但 loss 权重为 0；只对成功轨迹反传。 |
| `drop` | 构造 dataset 时只加载成功 episode，失败轨迹不进入 dataloader。 |
| `keep_loss` | 失败轨迹正常进入 dataloader 且正常计算 loss，用于兼容旧训练方式。 |

成功标签从 episode metadata 读取：

```bash
--acp.success_field=episode_success
```

`mask_loss` 模式依赖 batch 中的 `episode_index`，训练时会根据成功 episode 集合生成 0/1 per-sample loss 权重。若同时开启 RA-BC，两者权重会相乘后再归一。

推荐 ACP 命令片段：

```bash
--acp.enable=true \
--acp.indicator_field=complementary_info.acp_indicator_<TAG> \
--acp.indicator_dropout_prob=0.1 \
--acp.tag_negative_prompts=false \
--acp.failure_loss_mode=mask_loss \
--acp.success_field=episode_success
```

如果希望失败轨迹完全不进 dataloader：

```bash
--acp.failure_loss_mode=drop
```

## 推理侧 CFG

推理侧默认保持旧行为：如果只设置 `acp_inference.use_cfg=true`，CFG 仍然是动作级融合：

```python
action = action_uncond + beta * (action_cond - action_uncond)
```

如果额外开启 `acp_inference.velocity_cfg=true`，Pi0.5 会在 flow matching 采样循环里对每个 denoise step 做 RLinf 风格 velocity-level CFG：

```python
v = v_uncond + beta * (v_cond - v_uncond)
x_t = x_t + dt * v
```

配置关系：

- `acp_inference.enable=true`: 使用 `Task + Advantage: positive` 做正向条件推理。
- `acp_inference.use_cfg=true`: 同时使用 base task 作为 uncond 分支并执行 CFG。
- `acp_inference.velocity_cfg=true`: 在 Pi0.5 采样循环内逐步引导 velocity；默认 `false` 时保持动作级 CFG。
- `acp_inference.cfg_beta`: CFG 强度，默认 `1.0`。

`velocity_cfg=true` 当前只支持 Pi0.5；其它 policy 没有对应 flow-matching velocity 接口时会直接报错。

## 端到端命令流程

下面命令按当前分支的新 ACP 训练逻辑整理，覆盖：

1. 本地纯遥操检查。
2. 本地纯人工 HIL 采数。
3. 数据集检查与上传。
4. 远端 value model 训练与 value/advantage/indicator 标注。
5. 远端 ACP policy 训练。
6. 远端 policy server + 本地纯推理。
7. 远端 policy server + 本地 HIL 续采。

环境边界：

- 本地 Mac 负责真机、相机、leader arm、dataset 写入。
- AutoDL/GPU 机器负责 policy server、value 训练、ACP policy 训练。
- `/etc/network_turbo` 只在 AutoDL/GPU 机器上执行，不要在本地 Mac 上执行。
- 远端推理命令里的 `--pretrained_name_or_path` 是远端 GPU 机器上的路径，不是本地 Mac 路径。

### 本地公共变量

在本地 Mac 上执行：

```bash
conda activate evo-rl
cd /Users/bytedance/Evo-RL
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

export TASK="insert carrot into the hole"
export DATASET_REPO_R1="AlvinAi/insert_carrot_into_the_hole_hil_v2_r1"
export DATASET_ROOT_R1="/Users/bytedance/lerobot_dataset_v2_r1"
export DATASET_REPO_R2="AlvinAi/insert_carrot_into_the_hole_hil_v2_r2"
export DATASET_ROOT_R2="/Users/bytedance/lerobot_dataset_v2_r2"

export ROBOT_CAMERAS='{ front: {type: opencv, index_or_path: 0, width: 1280, height: 960, fps: 30, warmup_s: 10}, side: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30, warmup_s: 10}}'
export ROBOT_CAMERAS_15FPS='{ front: {type: opencv, index_or_path: 0, width: 1280, height: 960, fps: 15, warmup_s: 30}, side: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 15, warmup_s: 30}}'
```

说明：

- `ROBOT_CAMERAS` 不要在 `{ ... }` 字符串中间断行，否则 YAML/Draccus 解析容易失败。
- 30fps 用于纯遥操检查和首轮采数；如果相机带宽不稳，远端 HIL 续采可以改用 `ROBOT_CAMERAS_15FPS`，并把 `--dataset.fps` 也设为 `15`。

### 1. 纯遥操检查，不写数据

在本地 Mac 上执行：

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/cu.usbmodem5B790505591 \
  --robot.id=hongfeng_follower_arm \
  --robot.cameras="$ROBOT_CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/cu.usbmodem5B790500991 \
  --teleop.id=hongfeng_leader_arm \
  --fps=30 \
  --display_data=true
```

说明：

- 这一步只检查 follower、leader、双相机和 Rerun 展示，不创建 `LeRobotDataset`。
- 如果相机打不开，先单独排查 camera index、USB 带宽和 `MJPG`/分辨率/FPS 配置。

### 2. 首轮纯人工 HIL 采数

在本地 Mac 上执行：

```bash
python -m lerobot.scripts.lerobot_human_inloop_record \
  --robot.type=so101_follower \
  --robot.port=/dev/cu.usbmodem5B790505591 \
  --robot.id=hongfeng_follower_arm \
  --robot.cameras="$ROBOT_CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/cu.usbmodem5B790500991 \
  --teleop.id=hongfeng_leader_arm \
  --dataset.repo_id="$DATASET_REPO_R1" \
  --dataset.root="$DATASET_ROOT_R1" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes=80 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=2 \
  --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --dataset.vcodec=h264 \
  --display_data=true \
  --image_border.enable=true \
  --image_border.width_px=12 \
  --image_border.color_rgb='[0, 96, 255]' \
  --image_border.keys='[front, side]' \
  --resume=false
```

说明：

- 这是纯人工数据采集，不加载 policy，不走远端 inference。
- `s` 标记成功并结束当前 episode，`f` 标记失败并结束当前 episode。
- `episode_success` 会写入 episode metadata，后续 `mask_loss/drop` 都依赖这个字段。
- 主 `action` 字段写的是人工执行动作。

### 3. 数据集检查与上传

在本地 Mac 上执行：

```bash
lerobot-dataset-report --dataset "$DATASET_ROOT_R1"
```

重点检查：

- `episode_success` 是否同时包含 success/failure。
- episode 数、frame 数、FPS、相机字段是否符合预期。
- `complementary_info.is_intervention` 和 `collector_policy_id` 是否存在。

上传到 Hugging Face：

```bash
hf repo create "$DATASET_REPO_R1" --type dataset --private
hf upload "$DATASET_REPO_R1" "$DATASET_ROOT_R1" . --repo-type dataset
```

说明：

- 如果数据集 repo 已存在，`hf repo create` 失败可以忽略，直接执行 `hf upload`。
- `hf upload` 的本地路径必须和采数时的 `--dataset.root` 一致。
- `lerobot-value-infer` 只会改本地 dataset；如果处理后还想同步到 HF，需要再次 `hf upload`。

### 4. 远端 value model 训练

在 AutoDL/GPU 机器上执行：

```bash
conda activate evo-rl
cd /root/autodl-tmp/Evo-RL
source /etc/network_turbo
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

python -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=AlvinAi/insert_carrot_into_the_hole_hil_v2_r1 \
  --dataset.root=/root/autodl-tmp/data/lerobot/AlvinAi/insert_carrot_into_the_hole_hil_v2_r1 \
  --value.type=pistar06 \
  --value.device=cuda \
  --value.dtype=bfloat16 \
  --value.camera_features='[observation.images.front, observation.images.side]' \
  --value.push_to_hub=false \
  --batch_size=64 \
  --steps=8000 \
  --output_dir=/root/autodl-tmp/outputs/value_train/pistar06_insert_carrot_v2_r1 \
  --job_name=pistar06_insert_carrot_v2_r1 \
  --wandb.enable=false
```

说明：

- `pistar06` 是 value model，不是动作 policy。
- value model 读取 `episode_success`，生成每帧 value target。
- 多 GPU 时用 `accelerate launch`，并把单卡 `--batch_size` 按 GPU 数拆分。

### 5. value/advantage/indicator 标注

在 AutoDL/GPU 机器上执行：

```bash
python -m lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id=AlvinAi/insert_carrot_into_the_hole_hil_v2_r1 \
  --dataset.root=/root/autodl-tmp/data/lerobot/AlvinAi/insert_carrot_into_the_hole_hil_v2_r1 \
  --inference.checkpoint_path=/root/autodl-tmp/outputs/value_train/pistar06_insert_carrot_v2_r1 \
  --inference.checkpoint_ref=last \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --runtime.num_workers=4 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field=complementary_info.value_pistar06_v2_r1_n50_r30 \
  --acp.advantage_field=complementary_info.advantage_pistar06_v2_r1_n50_r30 \
  --acp.indicator_field=complementary_info.acp_indicator_pistar06_v2_r1_n50_r30 \
  --output_dir=/root/autodl-tmp/outputs/value_infer/pistar06_insert_carrot_v2_r1_n50_r30 \
  --job_name=pistar06_insert_carrot_v2_r1_n50_r30
```

说明：

- 这一步会把 `value/advantage/acp_indicator` 写回 `--dataset.root` 对应的数据集。
- `--acp.positive_ratio=0.3` 表示按 task 内 advantage 取 top 30% 为正样本。
- ACP policy 训练时的 `--acp.indicator_field` 必须和这里写出的 indicator 字段完全一致。

如果要把标注后的数据集同步回 Hugging Face：

```bash
hf upload AlvinAi/insert_carrot_into_the_hole_hil_v2_r1 /root/autodl-tmp/data/lerobot/AlvinAi/insert_carrot_into_the_hole_hil_v2_r1 . --repo-type dataset
```

### 6. 远端 ACP policy 训练

在 AutoDL/GPU 机器上执行：

```bash
export CUDA_VISIBLE_DEVICES=0

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=AlvinAi/insert_carrot_into_the_hole_hil_v2_r1 \
  --dataset.root=/root/autodl-tmp/data/lerobot/AlvinAi/insert_carrot_into_the_hole_hil_v2_r1 \
  --policy.path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --batch_size=64 \
  --num_workers=8 \
  --steps=30000 \
  --save_freq=3000 \
  --log_freq=10 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_pistar06_v2_r1_n50_r30 \
  --acp.indicator_dropout_prob=0.1 \
  --acp.tag_negative_prompts=false \
  --acp.failure_loss_mode=mask_loss \
  --acp.success_field=episode_success \
  --output_dir=/root/autodl-tmp/outputs/train/pi05_insert_carrot_v2_r1_acp_n50_r30 \
  --job_name=pi05_insert_carrot_v2_r1_acp_n50_r30 \
  --wandb.enable=false \
  --policy.push_to_hub=false
```

多 GPU 版本：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --dataset.repo_id=AlvinAi/insert_carrot_into_the_hole_hil_v2_r1 \
  --dataset.root=/root/autodl-tmp/data/lerobot/AlvinAi/insert_carrot_into_the_hole_hil_v2_r1 \
  --policy.path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --batch_size=64 \
  --num_workers=8 \
  --steps=30000 \
  --save_freq=3000 \
  --save_training_state=false \
  --log_freq=10 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_pistar06_v2_r1_n50_r30 \
  --acp.indicator_dropout_prob=0.1 \
  --acp.tag_negative_prompts=false \
  --acp.failure_loss_mode=mask_loss \
  --acp.success_field=episode_success \
  --output_dir=/root/autodl-tmp/outputs/train/pi05_insert_carrot_v2_r1_acp_n50_r30 \
  --job_name=pi05_insert_carrot_v2_r1_acp_n50_r30 \
  --wandb.enable=true \
  --wandb.project=Evo-RL \
  --wandb.disable_artifact=true \
  --policy.push_to_hub=false
```

说明：

- `mask_loss`：失败轨迹进入 dataloader 和模型 forward，但 loss 权重为 0。
- `drop`：失败 episode 在 dataset 构造阶段被过滤，不进入 dataloader。
- `keep_loss`：失败轨迹正常计算 loss，用于旧实验兼容。
- 当前默认 CFG 训练分布是：正样本 90% positive tag、10% base；负样本 100% base。

### 7. 启动远端 policy server

在 AutoDL/GPU 机器上执行：

```bash
conda activate evo-rl
cd /root/autodl-tmp/Evo-RL
source /etc/network_turbo
export HF_HOME=/root/autodl-tmp/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=8000 \
  --fps=30 \
  --inference_latency=0.0 \
  --obs_queue_timeout=20
```

如果本地 Mac 通过 SSH 访问 AutoDL，在本地 Mac 另开一个终端建立隧道：

```bash
ssh -N -L 8000:127.0.0.1:8000 -p <SSH_PORT> root@<AUTODL_HOST>
```

说明：

- `policy_server` 启动时不传模型路径；模型路径由本地 client 连接后发给 server。
- `--pretrained_name_or_path` 必须是 AutoDL/GPU 机器上真实存在的 checkpoint 路径。

### 8. 远端 policy 纯推理，不写数据

在本地 Mac 上执行：

```bash
python -m lerobot.scripts.lerobot_human_inloop_remote_infer \
  --robot.type=so101_follower \
  --robot.port=/dev/cu.usbmodem5B790505591 \
  --robot.id=hongfeng_follower_arm \
  --robot.cameras="$ROBOT_CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/cu.usbmodem5B790500991 \
  --teleop.id=hongfeng_leader_arm \
  --task="$TASK" \
  --server_address=127.0.0.1:8000 \
  --policy_type=pi05 \
  --pretrained_name_or_path=/root/autodl-tmp/models/pi05_insert_carrot_hil_r1r2r3_acp_n50_r30 \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=20 \
  --inference_mode=select_action_buffered_async \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --policy_sync_to_teleop=false \
  --fps=30 \
  --acp_inference.enable=true \
  --acp_inference.use_cfg=true \
  --acp_inference.velocity_cfg=true \
  --acp_inference.cfg_beta=1.0 \
  --image_border.enable=true \
  --image_border.width_px=12 \
  --image_border.color_rgb='[0, 96, 255]' \
  --image_border.keys='[front, side]' \
  --display_data=true
```

说明：

- 这是 dataset-free 真机推理，不创建 dataset，不写 parquet/video。
- `--acp_inference.enable=true` 会让 server 使用 `Task + Advantage: positive`。
- `--acp_inference.use_cfg=true --acp_inference.velocity_cfg=true` 会在 Pi0.5 每个 denoise step 做 velocity CFG。
- 如需回到旧的最终动作融合方式，保留 `--acp_inference.use_cfg=true`，但把 `--acp_inference.velocity_cfg=false` 或删掉该参数。
- `select_action_buffered_async` 是 `--inference_mode`，不是 `--aggregate_fn_name`。
- `aggregate_fn_name` 只能是 `weighted_average`、`conservative`、`average`、`latest_only`。

### 9. 远端 policy HIL 续采并写数据

在本地 Mac 上执行：

```bash
python -m lerobot.scripts.lerobot_human_inloop_remote_record \
  --robot.type=so101_follower \
  --robot.port=/dev/cu.usbmodem5B790505591 \
  --robot.id=hongfeng_follower_arm \
  --robot.cameras="$ROBOT_CAMERAS_15FPS" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/cu.usbmodem5B790500991 \
  --teleop.id=hongfeng_leader_arm \
  --dataset.repo_id="$DATASET_REPO_R2" \
  --dataset.root="$DATASET_ROOT_R2" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes=80 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=2 \
  --dataset.fps=15 \
  --dataset.push_to_hub=false \
  --dataset.vcodec=h264 \
  --display_data=true \
  --pretrained_name_or_path=/root/autodl-tmp/models/pi05_insert_carrot_hil_v2_r1r2 \
  --policy_type=pi05 \
  --server_address=127.0.0.1:8000 \
  --actions_per_chunk=50 \
  --policy_device=cuda \
  --client_device=cpu \
  --inference_mode=select_action_buffered_async \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --acp_inference.enable=true \
  --acp_inference.use_cfg=true \
  --acp_inference.velocity_cfg=true \
  --acp_inference.cfg_beta=1.0 \
  --image_border.enable=true \
  --image_border.width_px=12 \
  --image_border.color_rgb='[0, 96, 255]' \
  --image_border.keys='[front, side]' \
  --resume=false
```

说明：

- 这是 policy-assisted HIL 采数，会创建/写入 `LeRobotDataset`。
- policy 控制时，主 `action` 写入实际执行的 policy action；人工接管时，主 `action` 写入人工动作。
- `complementary_info.policy_action` 单独保存 policy 输出，便于后续分析。
- `i` 切换人工接管/释放接管，释放时会重置远端 policy buffer，避免旧 action chunk 跨状态残留。
- 这里使用 15fps 相机时，`--dataset.fps` 也设为 15；如果希望 30fps 训练数据，则把相机和 dataset 都改回 30。

采完后检查并上传：

```bash
lerobot-dataset-report --dataset "$DATASET_ROOT_R2"
hf repo create "$DATASET_REPO_R2" --type dataset --private
hf upload "$DATASET_REPO_R2" "$DATASET_ROOT_R2" . --repo-type dataset
```
