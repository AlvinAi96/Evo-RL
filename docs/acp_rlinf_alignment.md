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
- Loss：正负样本都参与 policy BC loss；负样本负责学习无条件/base 分支。

注意：RLinf/RECAP 的正负样本来自逐帧 Advantage 分位数，而不是直接把成功 episode
视为正样本、失败 episode 视为负样本。默认 top 30% 为正样本，再对正样本做 0.1
unconditional dropout，因此最终约 27% 是 positive prompt、73% 是 base prompt。

已核对 RLinf `ee2cab65` 源码：CFG dataloader 不按 episode 成败过滤数据，
`openpi_cfg_action_model.py` 的 flow matching loss 对整个 batch 直接求均值。负
Advantage 样本虽然被路由到 unconditional/base prompt，但仍产生动作 loss。配置项
`train_expert_only` 表示只更新 action expert 参数并冻结 VLM，不表示只使用成功专家轨迹。
“只训练成功专家轨迹”只适用于 CFG 之前的初始 SFT 数据来源描述，不适用于 RECAP 的
CFG policy optimization 阶段。

## Prompt 配置

ACP 配置位于 `src/lerobot/configs/train.py` 的 `ACPConfig`。

关键字段：

- `acp.enable`: 是否开启 ACP prompt hook。
- `acp.prompt_source`: `indicator` 使用逐帧 Advantage 标签；`episode_success` 使用整条
  episode 的成功/失败标签。
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

`acp.failure_loss_mode` 控制失败轨迹如何参与 ACP 训练。默认 `keep_loss`，与
RLinf/RECAP 中无条件样本同样参与 policy loss 的做法一致。

| 配置 | 行为 |
| --- | --- |
| `mask_loss` | 失败轨迹进入 dataloader 和模型 forward，但 loss 权重为 0；只对成功轨迹反传。 |
| `drop` | 构造 dataset 时只加载成功 episode，失败轨迹不进入 dataloader。 |
| `keep_loss` | 失败轨迹正常进入 dataloader 且正常计算 loss。 |

成功标签从 episode metadata 读取：

```bash
--acp.success_field=episode_success
```

`mask_loss` 模式依赖 batch 中的 `episode_index`，训练时会根据成功 episode 集合生成 0/1 per-sample loss 权重。若同时开启 RA-BC，两者权重会相乘后再归一。

严格按 RLinf/RECAP 的 Advantage 标签训练：

```bash
--acp.enable=true \
--acp.prompt_source=indicator \
--acp.indicator_field=complementary_info.acp_indicator_<TAG> \
--acp.indicator_dropout_prob=0.1 \
--acp.tag_negative_prompts=false \
--acp.failure_loss_mode=keep_loss
```

按 episode 成败组织 prompt 是本项目额外支持的策略，并非 RLinf 原始定义：

```bash
--acp.prompt_source=episode_success \
--acp.success_field=episode_success \
--acp.indicator_dropout_prob=0.1 \
--acp.failure_loss_mode=keep_loss
```

这会得到成功样本期望 90% positive、10% base，失败样本 100% base，且两者都参与
policy BC loss。

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

实现边界：

- Pi0.5 的 `sample_actions()` 接收可选 `cfg_positive_tokens/cfg_positive_masks/cfg_beta`。
- `tokens/masks` 是 base task；`cfg_positive_tokens/cfg_positive_masks` 是 `Task + Advantage: positive`。
- 进入 denoise loop 前会分别缓存 base prefix 和 positive prefix。
- 每一步都在同一个 `x_t` 上算 `v_uncond` 与 `v_cond`，再合成 `v_t`。
- CFG 在 policy 内部的归一化动作空间完成，且发生在 postprocessor/反归一化/相对动作还原之前。
- `select_action` 队列里缓存的是已经 velocity-guided 的 raw action；`select_action_buffered_async` 后续从同一个 guided queue 取动作，不再对 cond/uncond 两条 postprocessed action 做相减。
- RTC 当前不和 velocity CFG 同时启用；如果 Pi0.5 配置开启 RTC，再打开 `velocity_cfg=true` 会显式报错。

## 端到端命令流程

下面命令按当前分支的新 ACP 训练逻辑整理。部署形态固定为：

```text
远端 AutoDL/GPU 机器：只跑 policy server 和训练任务，不连接真机、不连接相机。
本地 Mac：连接 SO101 真机、leader arm、相机；通过 gRPC/SSH 隧道请求远端 policy action。
```

覆盖：

1. 本地纯遥操检查。
2. 本地纯人工 HIL 采数。
3. 数据集检查与上传。
4. 远端 value model 训练与 value/advantage/indicator 标注。
5. 远端 ACP policy 训练。
6. 远端 policy server。
7. 本地真机连接远端服务纯推理。
8. 本地真机连接远端服务 HIL 续采。

环境边界：

- 本地 Mac 负责真机、相机、leader arm、dataset 写入。
- AutoDL/GPU 机器负责 policy server、value 训练、ACP policy 训练。
- `/etc/network_turbo` 只在 AutoDL/GPU 机器上执行，不要在本地 Mac 上执行。
- 本地 client 命令里的 `--server_address=127.0.0.1:8000` 指向 SSH 隧道的本地端口；真正执行模型 forward 的是远端 `policy_server`。
- 本地 client 命令里的 `--pretrained_name_or_path` 会被发送给远端 `policy_server`，因此它必须是 AutoDL/GPU 机器上的 checkpoint 路径，不是本地 Mac 路径。

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
- `lerobot-value-infer` 只会修改当前执行机器上的 `--dataset.root`；如果处理后还想同步到 HF，需要再次 `hf upload`。

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

### 远端 LingBot-Depth 依赖与路径

只在远端 ACP policy 训练开启 `--policy.depth_align.enable=true` 时需要。推理阶段不需要 MoGe/LingBot-Depth，因为 depth 分支只作为训练期辅助监督。

在 AutoDL/GPU 机器上执行：

```bash
conda activate evo-rl
cd /root/autodl-tmp/Evo-RL

# 如果远端还没有安装 LingBot-Depth/MoGe 依赖，先按实际路径安装或确认可 import。
pip install -e /root/autodl-tmp/lingbot-vla/lingbotvla/models/vla/vision_models/lingbot-depth --no-deps

# 这两个路径都属于远端 GPU 机器；本地 Mac 不需要有这些文件。
export MOGE_PATH=/root/autodl-tmp/models/moge2-vitb-normal.pt
export LINGBOT_DEPTH_PATH=/root/autodl-tmp/models/lingbot_depth/model_mdm_pre.pt
```

快速检查：

```bash
python - <<'PY'
from mdm.model.v2 import MDMModel
from moge.model.v2 import MoGeModel
print("depth deps ok")
PY
```

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
  --acp.target_field=complementary_info.value_target_pistar06_v2_r1_n50_r30 \
  --acp.advantage_field=complementary_info.advantage_pistar06_v2_r1_n50_r30 \
  --acp.indicator_field=complementary_info.acp_indicator_pistar06_v2_r1_n50_r30 \
  --output_dir=/root/autodl-tmp/outputs/value_infer/pistar06_insert_carrot_v2_r1_n50_r30 \
  --job_name=pistar06_insert_carrot_v2_r1_n50_r30
```

说明：

- 这一步会把 `value/value_target/advantage/acp_indicator` 写回 `--dataset.root` 对应的数据集。
- `--acp.positive_ratio=0.3` 表示按 task 内 advantage 取 top 30% 为正样本。
- 下方推荐的 RLinf/RECAP 式 policy 训练直接使用该 indicator：top 30% Advantage
  样本进入 positive prompt，其余样本进入 base prompt。

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
  --acp.prompt_source=indicator \
  --acp.indicator_field=complementary_info.acp_indicator_pistar06_v2_r1_n50_r30 \
  --acp.indicator_dropout_prob=0.1 \
  --acp.tag_negative_prompts=false \
  --acp.failure_loss_mode=keep_loss \
  --output_dir=/root/autodl-tmp/outputs/train/pi05_insert_carrot_v2_r1_acp_n50_r30 \
  --job_name=pi05_insert_carrot_v2_r1_acp_n50_r30 \
  --wandb.enable=false \
  --policy.push_to_hub=false
```

如果本轮要启用 LingBot-Depth 蒸馏，在上面的 `lerobot_train` 命令中额外加入：

```bash
  --policy.depth_align.enable=true \
  --policy.depth_align.moge_path="$MOGE_PATH" \
  --policy.depth_align.lingbot_depth_path="$LINGBOT_DEPTH_PATH" \
  --policy.depth_align.loss_weight=0.004 \
  --policy.depth_align.target_token_size=16 \
  --policy.depth_align.target_dim=1024 \
  --policy.depth_align.target_num_tokens=256 \
  --policy.depth_align.resolution_level=3 \
  --policy.depth_align.visualize.enable=true \
  --policy.depth_align.visualize.output_dir=/root/autodl-tmp/outputs/depth_align_viz/pi05_insert_carrot_v2_r1_acp_n50_r30 \
  --policy.depth_align.visualize.interval_steps=500 \
  --policy.depth_align.visualize.max_items=4 \
```

等确认可视化正常后，把可视化关掉即可：

```bash
  --policy.depth_align.enable=true \
  --policy.depth_align.moge_path="$MOGE_PATH" \
  --policy.depth_align.lingbot_depth_path="$LINGBOT_DEPTH_PATH" \
  --policy.depth_align.visualize.enable=false \
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
  --acp.prompt_source=indicator \
  --acp.indicator_field=complementary_info.acp_indicator_pistar06_v2_r1_n50_r30 \
  --acp.indicator_dropout_prob=0.1 \
  --acp.tag_negative_prompts=false \
  --acp.failure_loss_mode=keep_loss \
  --output_dir=/root/autodl-tmp/outputs/train/pi05_insert_carrot_v2_r1_acp_n50_r30 \
  --job_name=pi05_insert_carrot_v2_r1_acp_n50_r30 \
  --wandb.enable=true \
  --wandb.project=Evo-RL \
  --wandb.disable_artifact=true \
  --policy.push_to_hub=false
```

多 GPU 训练同样可以加入 LingBot-Depth 参数块；`MOGE_PATH` 和 `LINGBOT_DEPTH_PATH` 仍然必须是每个远端训练进程都能访问到的路径。

说明：

- 当前命令采用 RLinf/RECAP 式 `indicator + keep_loss`：按逐帧 Advantage top 30%
  生成 indicator；正样本中 90% 使用 positive prompt、10% 使用 base prompt，负样本
  全部使用 base prompt，所有样本都参与 loss。
- `mask_loss`：失败轨迹进入 dataloader 和模型 forward，但 loss 权重为 0。
- `drop`：失败 episode 在 dataset 构造阶段被过滤，不进入 dataloader。
- `keep_loss`：失败轨迹正常计算 loss，也是 RLinf/RECAP 无条件样本的训练语义。
- `positive_ratio=0.3` 与 `indicator_dropout_prob=0.1` 组合后，期望约 27% 样本使用
  positive prompt、73% 使用 base prompt。
- 不加 `--policy.depth_align.enable=true` 时就是原 ACP policy 训练，不加载 MoGe/LingBot-Depth。
- 开启 depth 后，MoGe/LingBot-Depth 冻结产 teacher target，梯度只回传到 Pi0.5 可训练参数和新增 `depth_align_head`。
- `depth_align.visualize.enable=true` 只保存 RGB/teacher feature/policy feature 对齐图，方便初期检查；确认无误后建议关闭以减少训练 I/O。

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

- `policy_server` 只负责接收 observation、在远端 GPU 上跑 policy forward、返回 action；它不连接真机、不写数据集。
- `policy_server` 启动时不传模型路径；模型路径由本地 client 连接后发给 server。
- `--pretrained_name_or_path` 必须是 AutoDL/GPU 机器上真实存在的 checkpoint 路径。
- 远端 server 如果绑定 `127.0.0.1`，本地必须通过 SSH 隧道访问；如果绑定公网/内网 IP，要同步修改本地 client 的 `--server_address`。

### 8. 本地真机连接远端服务纯推理，不写数据

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
  --display_data=true
```

说明：

- 这是 dataset-free 真机推理：本地 Mac 连接真机和相机，远端 GPU 只返回 action，不创建 dataset，不写 parquet/video。
- `--server_address=127.0.0.1:8000` 是本地 SSH 隧道入口；如果没有 SSH 隧道，就改成远端 server 的可访问地址。
- `--pretrained_name_or_path=/root/...` 虽然写在本地命令里，但会发给远端 server 加载；路径必须存在于 AutoDL/GPU 机器。
- LingBot-Depth 蒸馏只影响训练后的 checkpoint；这条推理命令不需要传 `policy.depth_align.*`，也不会在推理时输出 depth map。
- `--acp_inference.enable=true` 会让 server 使用 `Task + Advantage: positive`。
- `--acp_inference.use_cfg=true --acp_inference.velocity_cfg=true` 会在 Pi0.5 每个 denoise step 做 velocity CFG。
- 如需回到旧的最终动作融合方式，保留 `--acp_inference.use_cfg=true`，但把 `--acp_inference.velocity_cfg=false` 或删掉该参数。
- `select_action_buffered_async` 是 `--inference_mode`，不是 `--aggregate_fn_name`。
- `aggregate_fn_name` 只能是 `weighted_average`、`conservative`、`average`、`latest_only`。

### 9. 本地真机连接远端服务 HIL 续采并写数据

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
  --resume=false
```

说明：

- 这是 policy-assisted HIL 采数：本地 Mac 创建/写入 `LeRobotDataset`，远端 GPU 只负责 policy 推理。
- `--pretrained_name_or_path=/root/...` 仍然是远端 checkpoint 路径，不是本地路径。
- 若 checkpoint 是开启 LingBot-Depth 蒸馏训练得到的，HIL 续采命令也不需要传 `policy.depth_align.*`；推理只用已经训练好的 VLA/action 参数。
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
