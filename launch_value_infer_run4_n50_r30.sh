#!/usr/bin/env bash
#SBATCH --job-name=evo-value-infer-r1
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000M
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/logs/slurm-%x-%j.err
set -euo pipefail

cd /scratch/u6pw/ql337.u6pw/projects/Evo-RL
. ./r1_training_config.sh

exec .venv/bin/accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  .venv/bin/lerobot-value-infer \
  --dataset.repo_id="$DATASET_REPO_R1" \
  --dataset.root="$DATASET_ROOT_R1" \
  --inference.checkpoint_path="$VALUE_CHECKPOINT_R1" \
  --inference.checkpoint_ref=last \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --runtime.num_workers=8 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field="$ACP_VALUE_FIELD_R1" \
  --acp.advantage_field="$ACP_ADVANTAGE_FIELD_R1" \
  --acp.indicator_field="$ACP_INDICATOR_FIELD_R1" \
  --output_dir="$VALUE_INFER_OUTPUT_R1" \
  --job_name=pistar06_insert_carrot_v2_r1_n50_r30.infer
