---
library_name: lerobot
tags:
  - lerobot
  - robotics
  - value-function
  - vision-language
  - human-in-the-loop
  - pistar06
datasets:
  - AlvinAi/insert_carrot_into_the_hole_hil
---

# Pi*0.6 Value Function — Insert Carrot

This repository contains the final step-2000 Pi*0.6 value-function checkpoint trained for the task **"insert carrot into the hole"**. It was trained with the [MINT-SJTU/Evo-RL](https://github.com/MINT-SJTU/Evo-RL) pipeline on human-in-the-loop robot demonstrations.

This is a **value model**, not an action policy. Its intended role is to estimate per-frame value/return-to-go signals that Evo-RL can convert into n-step advantages and binary Advantage-Conditioned Policy (ACP) indicators.

## Training data

- Dataset: [AlvinAi/insert_carrot_into_the_hole_hil](https://huggingface.co/datasets/AlvinAi/insert_carrot_into_the_hole_hil)
- Frames: 40,353
- Episodes: 83
- Task: insert carrot into the hole
- Observation cameras: `observation.images.front` and `observation.images.side`
- Robot state: `observation.state`, 6 dimensions

## Model configuration

- Value model type: `pistar06`
- Vision backbone: `google/siglip-so400m-patch14-384`
- Language backbone: `google/gemma-3-270m`
- Value support: 201 bins over `[-1.0, 0.0]`
- Training dtype: bfloat16
- Vision and language backbones were fine-tuned
- Gradient checkpointing: enabled

## Training configuration

| Parameter | Value |
|---|---:|
| GPUs | 4 |
| Batch size per GPU | 64 |
| Global batch size | 256 |
| Optimizer steps | 2,000 |
| Approximate dataset passes | 12.7 |
| Optimizer | AdamW |
| Peak learning rate | 5e-5 |
| Warmup steps | 125 |
| Scheduler decay steps | 2,000 |
| Final checkpoint | step 2,000 |

The final W&B training summary reported `train/loss=0.5956` and `train/value_mae=0.00191`. These are training metrics, not held-out evaluation or robot rollout success rates.

## Value inference and ACP annotation

Run from an Evo-RL environment:

```bash
lerobot-value-infer \
  --dataset.repo_id=AlvinAi/insert_carrot_into_the_hole_hil \
  --inference.checkpoint_path=Aikwed/pistart0.6-insert-carrot \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field=complementary_info.value_pistar06 \
  --acp.advantage_field=complementary_info.advantage_pistar06 \
  --acp.indicator_field=complementary_info.acp_indicator_pistar06 \
  --output_dir=outputs/value_infer/pistar06_insert_carrot
```

The checkpoint includes the model weights, model/training configuration, processor definitions, and normalization statistics. Optimizer, scheduler, and RNG training states are not included in this Hub repository.

## Limitations

- The model is task- and embodiment-specific and has not been validated outside the insert-carrot dataset distribution.
- Camera keys, image shapes, robot-state layout, and preprocessing must match the saved configuration.
- Low offline training loss does not by itself establish calibrated values or improved closed-loop robot success; downstream rollout evaluation is required.
