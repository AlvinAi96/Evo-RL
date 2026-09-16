#!/usr/bin/env bash
#SBATCH --job-name=evo-pi05-acp-depth-r1
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --exclude=nid010659,nid010910
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000M
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/logs/slurm-%x-%j.err
set -euo pipefail

cd /scratch/u6pw/ql337.u6pw/projects/Evo-RL
. ./r1_training_config.sh

export MOGE_PATH=/lus/lfs1aip2/projects/u6pw/models/evo_rl_depth/moge-2-vitb-normal/model.pt
export LINGBOT_DEPTH_PATH=/lus/lfs1aip2/projects/u6pw/models/evo_rl_depth/lingbot-depth-pretrain-vitl-14/model.pt
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

visible_gpus=$(.venv/bin/python -c 'import torch; print(torch.cuda.device_count())')
if [[ "$visible_gpus" != "4" ]]; then
  echo "Expected 4 visible GPUs, but CUDA reports $visible_gpus; refusing to oversubscribe a GPU." >&2
  exit 42
fi

output_dir=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/outputs/train/pi05_insert_carrot_v2_r1_acp_depth_bs256_e10
resume_config="$output_dir/checkpoints/last/pretrained_model/train_config.json"

if [[ -f "$resume_config" ]]; then
  train_args=(
    --config_path="$resume_config"
    --resume=true
  )
else
  train_args=(
  --dataset.repo_id="$DATASET_REPO_R1"
  --dataset.root="$DATASET_ROOT_R1"
  --policy.path=lerobot/pi05_base
  --policy.device=cuda
  --policy.dtype=bfloat16
  --policy.gradient_checkpointing=true
  --policy.depth_align.enable=true
  --policy.depth_align.moge_path="$MOGE_PATH"
  --policy.depth_align.lingbot_depth_path="$LINGBOT_DEPTH_PATH"
  --policy.depth_align.loss_weight=0.004
  --policy.depth_align.target_token_size=16
  --policy.depth_align.target_dim=1024
  --policy.depth_align.target_num_tokens=256
  --policy.depth_align.resolution_level=3
  --policy.depth_align.visualize.enable=true
  --policy.depth_align.visualize.output_dir=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/outputs/depth_align_viz/pi05_insert_carrot_v2_r1_acp_depth_bs256_e10
  --policy.depth_align.visualize.interval_steps=100
  --policy.depth_align.visualize.max_items=4
  --rename_map="$PI05_RENAME_MAP_R1"
  --batch_size=64
  --num_workers=8
  --steps=1687
  --save_freq=169
  --save_training_state=true
  --log_freq=10
  --acp.enable=true
  --acp.indicator_field="$ACP_INDICATOR_FIELD_R1"
  --acp.indicator_dropout_prob=0.1
  --acp.tag_negative_prompts=false
  --acp.failure_loss_mode=mask_loss
  --acp.success_field=episode_success
  --output_dir="$output_dir"
  --job_name=pi05_insert_carrot_v2_r1_acp_depth_bs256_e10
  --wandb.enable=true
  --wandb.project=Evo-RL
  --wandb.disable_artifact=true
  --policy.push_to_hub=false
  --policy.repo_id=Aikwed/pi05_insert_carrot_v2_r1_acp_depth_n50_r30
  )
fi

exec .venv/bin/accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  .venv/bin/lerobot-train \
  "${train_args[@]}"
