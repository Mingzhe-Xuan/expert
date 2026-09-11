#!/bin/bash
#SBATCH --job-name=expert-mace-smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/mace-adapter-%j.out
#SBATCH --error=logs/slurm/mace-adapter-%j.err

set -euo pipefail

: "${EXPERT_VENV:?Set EXPERT_VENV to the recorded project virtual environment}"
source "${EXPERT_VENV}/bin/activate"
mkdir -p logs/slurm results/backbones
python -m src.cli.mace_smoke \
  --device cuda \
  --output "results/backbones/mace-${SLURM_JOB_ID}.json"
