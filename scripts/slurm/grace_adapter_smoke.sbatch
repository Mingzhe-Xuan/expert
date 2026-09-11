#!/bin/bash
#SBATCH --job-name=expert-grace-smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/grace-adapter-%j.out
#SBATCH --error=logs/slurm/grace-adapter-%j.err

set -euo pipefail

: "${EXPERT_VENV:?Set EXPERT_VENV to the recorded GRACE-compatible environment}"
source "${EXPERT_VENV}/bin/activate"
export TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p logs/slurm results/backbones
python -m src.cli.grace_smoke \
  --device cuda \
  --output "results/backbones/grace-${SLURM_JOB_ID}.json"
