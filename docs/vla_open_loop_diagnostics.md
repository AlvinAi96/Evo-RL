# VLA 开环、Value 与深度诊断

## 1. 开环动作推理

在带 GPU 和 checkpoint 的机器运行：

```bash
python examples/open_loop_trainset_eval.py \
  --checkpoint /path/to/checkpoint/pretrained_model \
  --output-dir outputs/open_loop/run_001 \
  --num-samples 400 \
  --batch-size 16 \
  --cfg-beta 0.6 \
  --device cuda
```

脚本对相同观测和相同 diffusion noise 分别推理：

- `base`
- `positive`
- 数据原标签对应的 `train_tag`
- denoise 每一步执行的 `velocity_cfg_beta_0.6`

输出包含 `predictions.npz`、`summary.json/csv`、逐样本动作曲线和 HTML 报告。不要用
`base + beta * (positive - base)` 的最终 action 线性插值代替 velocity CFG；两者不是同一算法。

## 2. Prompt、关节、累计步数和轨迹阶段 MAE

```bash
lerobot-vla-analysis action \
  --predictions outputs/open_loop/run_001/predictions.npz \
  --summary outputs/open_loop/run_001/summary.json \
  --dataset-root /path/to/evaluated_dataset \
  --output-dir outputs/open_loop/run_001/detailed \
  --horizons 1 5 10 20 30 40 50 \
  --stage-bins 5 \
  --fps 30
```

主要输出：

- `prompt_joint_horizon_mae.csv`：各 prompt、关节、累计前 H 步的 MAE/RMSE
- `prompt_joint_stage_mae.csv`：各 prompt、关节、轨迹 0–20% 至 80–100% 阶段 MAE
- `*_joint_horizon_mae.png` 与 `*_joint_stage_mae.png`：热力图
- `sample_stages.csv`：每个开环窗口的 episode、frame 和阶段

轨迹阶段是时间进度分桶，不等同于“接近、抓取、搬运、插入”等语义阶段。需要语义阶段时，
应先给数据增加阶段标注，再按该字段分组。

## 3. Value → Advantage 与过拟合

先分别对训练集和真正未参与训练的验证集执行 Value 推理：

```bash
lerobot-value-infer \
  --dataset.repo_id=local/train \
  --dataset.root=/path/to/train_dataset \
  --inference.checkpoint_path=/path/to/value_run \
  --acp.enable=true \
  --acp.n_step=50 \
  --runtime.device=cuda

lerobot-value-infer \
  --dataset.repo_id=local/heldout \
  --dataset.root=/path/to/heldout_dataset \
  --inference.checkpoint_path=/path/to/value_run \
  --acp.enable=true \
  --acp.n_step=50 \
  --runtime.device=cuda
```

推理会写入：

- `complementary_info.value`
- `complementary_info.value_target`
- `complementary_info.advantage`
- `complementary_info.acp_indicator`

然后比较训练集与验证集：

```bash
lerobot-vla-analysis value \
  --train-dataset /path/to/train_dataset \
  --eval-dataset /path/to/heldout_dataset \
  --output-dir outputs/value_audit/run_001
```

`value_audit.json` 会给出 Value MAE/RMSE、预测与 target 相关性、Advantage 分布和
generalization gap。只有训练集结果时，工具会明确标记“不能判断过拟合”；训练误差很小本身
不是泛化证据。

## 4. 深度与 padding 对照

```bash
lerobot-depth-audit \
  --dataset-root /path/to/dataset \
  --moge-checkpoint /path/to/moge/model.pt \
  --moge-source /path/to/MoGe \
  --output-dir outputs/depth_audit/run_001 \
  --samples 29:0.45:front 29:0.45:side 56:0.65:front 56:0.65:side \
  --device cuda
```

每个 sample 会保存原图、224 方形 letterbox 输入、原始宽高比深度图和 letterbox 深度图的
对照，以及 `.npz` 原始数组和 `depth_audit.csv/json`。

## 5. LingBot-VLA 与本项目的处理差异

官方 LingBot-VLA 在 `prepare_images` 中先把 RGB 从 `[0,255]` 归一化到 `[-1,1]`，再调用
`resize_with_pad_item(..., pad_value=-1)`。因此黑色 padding 保持为 `-1`，不会产生 `-3`。
它仍使用 letterbox，所以仍有内容分辨率损失，但数值范围正确。

本项目原实现是在 `[0,1]` 图像上填 `-1`，随后整体执行 `2*x-1`，使 padding 变成 `-3`。
现在 Pi0、Pi0.5 和 Pi0-Fast 都统一在 `[0,1]` 阶段填 `0`，归一化后恰好是 `-1`。此外，
Depth teacher 改为直接读取
未做方形 padding 的相机 RGB，并逐相机处理不同原始分辨率；Pi0.5/SigLIP 分支仍保留
letterbox 以兼容模型结构。

## 6. 按 episode 成败组织 Prompt

训练时使用：

```bash
--acp.enable=true \
--acp.prompt_source=episode_success \
--acp.success_field=episode_success \
--acp.indicator_dropout_prob=0.1 \
--acp.failure_loss_mode=keep_loss \
--acp.tag_negative_prompts=false
```

含义：

- 成功 episode 的样本先视为 positive，再以 0.1 概率 dropout 为 base，期望比例为
  90% positive + 10% base；
- 失败 episode 全部使用 base prompt；
- `keep_loss` 保证失败样本仍参与 BC loss；
- 该模式不依赖 Value 推理生成的 `acp_indicator`。

固定 seed 时 prompt 抽样可复现。90/10 是训练采样期望比例，不是把数据集永久切成两个互斥副本。

这是本项目按任务需求增加的简化策略，不是 RLinf/RECAP 的原始标签定义。RLinf/RECAP
先通过 Value 模型计算逐帧 Advantage，以分位数选出正样本；低优势样本使用 base prompt，
并同样参与 policy loss。其默认 top 30% 与 0.1 positive dropout 会产生约 27%
positive prompt。
