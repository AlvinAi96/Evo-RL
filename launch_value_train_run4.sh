#!/usr/bin/env bash
set -euo pipefail

cd /scratch/u6pw/ql337.u6pw/projects/Evo-RL

exec .venv/bin/accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  .venv/bin/lerobot-value-train \
  --dataset.repo_id=AlvinAi/insert_carrot_into_the_hole_hil \
  --dataset.root=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/data/lerobot/AlvinAi/insert_carrot_into_the_hole_hil \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.use_gradient_checkpointing=true \
  --batch_size=64 \
  --num_workers=8 \
  --steps=2000 \
  --save_freq=500 \
  --log_freq=25 \
  --value.scheduler_warmup_steps=125 \
  --value.scheduler_decay_steps=2000 \
  --output_dir=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/outputs/value_train/pistart0.6-insert-carrot \
  --job_name=pistart0.6-insert-carrot \
  --value.push_to_hub=false \
  --wandb.enable=true \
  --wandb.project=evo-rl
