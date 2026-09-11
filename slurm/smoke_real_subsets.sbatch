#!/bin/bash
#SBATCH --job-name=expert-real-smoke
#SBATCH --partition=compute
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/real-smoke-%A_%a.out
#SBATCH --error=logs/slurm/real-smoke-%A_%a.err

set -euo pipefail

: "${EXPERT_VENV:?Set EXPERT_VENV to the recorded project virtual environment}"
source "${EXPERT_VENV}/bin/activate"
mkdir -p logs/slurm results/real-smoke
python -m src.cli.real_subset_smoke \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --device cuda \
  --output-root results/real-smoke
