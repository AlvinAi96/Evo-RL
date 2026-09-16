#!/usr/bin/env bash
# Continue from weights: checkpoint 000500 has no optimizer/RNG state.
# Remaining 1500 updates use a fresh optimizer and cosine LR schedule.
#SBATCH --job-name=evo-value-r1-cont
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

checkpoint="$VALUE_OUTPUT_R1/checkpoints/000500/pretrained_model"
output_dir="$VALUE_OUTPUT_R1/continued_from_000500_${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
test -s "$checkpoint/model.safetensors"

exec .venv/bin/accelerate launch \
  --multi_gpu --num_processes=4 --num_machines=1 \
  --mixed_precision=no --dynamo_backend=no \
  .venv/bin/lerobot-value-train \
  --dataset.repo_id="$DATASET_REPO_R1" \
  --dataset.root="$DATASET_ROOT_R1" \
  --value.path="$checkpoint" \
  --value.device=cuda \
  --value.dtype=bfloat16 \
  --value.use_gradient_checkpointing=true \
  --value.scheduler_warmup_steps=0 \
  --value.scheduler_decay_steps=1500 \
  --batch_size=64 --num_workers=8 \
  --steps=1500 --save_freq=500 --log_freq=25 \
  --save_training_state=true \
  --output_dir="$output_dir" \
  --job_name=pistar06_insert_carrot_v2_r1_continued \
  --value.push_to_hub=false \
  --wandb.enable=true --wandb.project=evo-rl
