#!/usr/bin/env bash
set -euo pipefail

cd /scratch/u6pw/ql337.u6pw/projects/Evo-RL
. ./r1_training_config.sh

.venv/bin/accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=bf16 \
  .venv/bin/lerobot-train \
  --dataset.repo_id="$DATASET_REPO_R1" \
  --dataset.root="$DATASET_ROOT_R1" \
  --policy.path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --rename_map="$PI05_RENAME_MAP_R1" \
  --batch_size=64 \
  --num_workers=8 \
  --steps=1580 \
  --save_freq=158 \
  --save_training_state=false \
  --log_freq=10 \
  --acp.enable=true \
  --acp.indicator_field="$ACP_INDICATOR_FIELD_R1" \
  --acp.indicator_dropout_prob=0.1 \
  --acp.tag_negative_prompts=false \
  --acp.failure_loss_mode=mask_loss \
  --acp.success_field=episode_success \
  --output_dir=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/outputs/train/pi05_insert_carrot_v2_r1_acp_bs256_e10 \
  --job_name=pi05_insert_carrot_v2_r1_acp_bs256_e10 \
  --wandb.enable=true \
  --wandb.project=Evo-RL \
  --wandb.disable_artifact=true \
  --policy.push_to_hub=false \
  --policy.repo_id=Aikwed/pi05_insert_carrot_v2_r1_acp_n50_r30

.venv/bin/python upload_pi05_acp_selected_epochs.py
