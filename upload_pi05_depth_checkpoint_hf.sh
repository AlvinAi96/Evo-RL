#!/usr/bin/env bash
#SBATCH --job-name=evo-pi05-hf-upload
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/u6pw/ql337.u6pw/projects/Evo-RL/logs/slurm-%x-%j.err
set -euo pipefail

cd /scratch/u6pw/ql337.u6pw/projects/Evo-RL
export PYTHONUNBUFFERED=1

exec .venv/bin/python scripts/upload_pi05_depth_checkpoint_hf.py
