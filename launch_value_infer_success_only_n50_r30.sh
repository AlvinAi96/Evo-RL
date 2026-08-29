#!/usr/bin/env bash
set -euo pipefail

cd /scratch/u6pw/ql337.u6pw/projects/Evo-RL

exec .venv/bin/accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=bf16 \
  .venv/bin/lerobot-value-infer \
  --dataset.repo_id=Aikwed/insert_carrot_into_the_hole_hil_success_only \
  --dataset.root=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/data/lerobot/Aikwed/insert_carrot_into_the_hole_hil_success_only_acp_pistar06_n50_r30 \
  --inference.checkpoint_path=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/outputs/value_train/pistar06_insert_carrot_hil_success_only \
  --inference.checkpoint_ref=last \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --runtime.num_workers=8 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field=complementary_info.value_pistar06_success_only_n50_r30 \
  --acp.advantage_field=complementary_info.advantage_pistar06_success_only_n50_r30 \
  --acp.indicator_field=complementary_info.acp_indicator_pistar06_success_only_n50_r30 \
  --output_dir=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/outputs/value_infer/pistar06_success_only_n50_r30 \
  --job_name=pistar06_success_only_n50_r30.infer
