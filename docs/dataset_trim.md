# 数据集起始静止裁剪

## 目标

`lerobot-trim-dataset` 分析每条 episode 从初始 action 到首次持续运动的静止时长，并可生成
裁掉过长静止前缀的新 LeRobot v3 数据集。

源数据集不会被原地修改。输出先写入 `.incomplete` 临时目录，全部 episode 成功后才重命名
为目标目录。

## 当前数据集的分析规则

- 运动阈值：任一关节相对初始 action 的绝对变化大于 `0.01`
- 持续运动：连续 5 帧满足阈值
- 保留上下文：首次持续运动前保留 1 秒静止画面

当前 `/Users/bytedance/lerobot_dataset_v2_r1` 的分析结果：

- 80 episodes，原始 43173 帧
- 起始静止中位数 2.02 秒，最长 18.37 秒
- 9 条 episode 起始静止超过 5 秒
- 按上述规则将裁掉 4215 帧，即 140.5 秒
- 输出预计 38958 帧

## 只分析，不改数据

```bash
lerobot-trim-dataset \
  --source /path/to/source_dataset \
  --analysis-only \
  --report-dir outputs/dataset_trim/my_dataset \
  --motion-threshold 0.01 \
  --min-moving-frames 5 \
  --keep-static-seconds 1.0
```

先查看 `episode_trim_plan.csv`，尤其是
`first_sustained_motion_frame`、`trim_frames` 和 `output_frames`。如果机器人有非常慢的
起步动作，应降低 `--motion-threshold`，不要直接沿用默认值。

## 生成裁剪后的新数据集

```bash
lerobot-trim-dataset \
  --source /path/to/source_dataset \
  --output-dir /path/to/new_dataset \
  --report-dir outputs/dataset_trim/my_dataset_trimmed \
  --motion-threshold 0.01 \
  --min-moving-frames 5 \
  --keep-static-seconds 1.0 \
  --vcodec h264
```

## 验证

```bash
lerobot-dataset-report --dataset /path/to/new_dataset
```

重点检查：

- episode 数与源数据一致；
- 总帧数等于 `summary.json` 的 `output_frames`；
- `episode_success` 等 episode metadata 仍存在；
- 两路视频分辨率、FPS 和 episode 时间范围正确；
- 抽查每条轨迹在约 1 秒后开始运动。

若中途失败，脚本会保留 `<output-dir>.incomplete` 便于排查；确认无用后再删除。
