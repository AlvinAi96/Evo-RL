#!/usr/bin/env bash
set -euo pipefail

cd /scratch/u6pw/ql337.u6pw/projects/Evo-RL
. ./r1_training_config.sh

exec .venv/bin/accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  .venv/bin/lerobot-value-train \
  --dataset.repo_id="$DATASET_REPO_R1" \
  --dataset.root="$DATASET_ROOT_R1" \
  --value.type=pistar06 \
  --value.camera_features='[observation.images.front, observation.images.side]' \
  --value.dtype=bfloat16 \
  --value.use_gradient_checkpointing=true \
  --batch_size=64 \
  --num_workers=8 \
  --steps=8000 \
  --save_freq=2000 \
  --log_freq=25 \
  --value.scheduler_warmup_steps=500 \
  --value.scheduler_decay_steps=8000 \
  --output_dir="$VALUE_OUTPUT_R1" \
  --job_name=pistar06_insert_carrot_v2_r1 \
  --value.push_to_hub=false \
  --wandb.enable=true \
  --wandb.project=evo-rl
