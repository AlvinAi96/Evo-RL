#!/usr/bin/env bash
set -euo pipefail

cd /scratch/u6pw/ql337.u6pw/projects/Evo-RL

.venv/bin/accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=bf16 \
  .venv/bin/lerobot-train \
  --dataset.repo_id=AlvinAi/insert_carrot_into_the_hole_hil \
  --dataset.root=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/data/lerobot/AlvinAi/insert_carrot_into_the_hole_hil_acp_pistar06_run4_n50_r30 \
  --policy.path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.side":"observation.images.left_wrist_0_rgb"}' \
  --batch_size=64 \
  --num_workers=8 \
  --steps=1580 \
  --save_freq=158 \
  --save_training_state=false \
  --log_freq=10 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_pistar06_run4_n50_r30 \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/outputs/train/pi05_insert_carrot_hil_acp_bs256_e10 \
  --job_name=pi05_insert_carrot_hil_acp_bs256_e10 \
  --wandb.enable=true \
  --wandb.project=Evo-RL \
  --wandb.disable_artifact=true \
  --policy.push_to_hub=false \
  --policy.repo_id=Aikwed/pistar06_insert_carrot_into_the_hole_acp_r1

.venv/bin/python upload_pi05_acp_selected_epochs.py
