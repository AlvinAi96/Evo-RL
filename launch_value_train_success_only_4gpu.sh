#!/usr/bin/env bash
set -euo pipefail

cd /scratch/u6pw/ql337.u6pw/projects/Evo-RL

exec .venv/bin/accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  .venv/bin/lerobot-value-train \
  --dataset.repo_id=Aikwed/insert_carrot_into_the_hole_hil_success_only \
  --dataset.root=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/data/lerobot/Aikwed/insert_carrot_into_the_hole_hil_success_only \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.use_gradient_checkpointing=true \
  --batch_size=64 \
  --num_workers=8 \
  --steps=1731 \
  --save_freq=433 \
  --log_freq=22 \
  --value.scheduler_warmup_steps=108 \
  --value.scheduler_decay_steps=1731 \
  --output_dir=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/outputs/value_train/pistar06_insert_carrot_hil_success_only \
  --job_name=pistar06_insert_carrot_hil_success_only \
  --value.push_to_hub=true \
  --value.repo_id=Aikwed/pistar06_insert_carrot_hil_success_only \
  --wandb.enable=true \
  --wandb.project=evo-rl
