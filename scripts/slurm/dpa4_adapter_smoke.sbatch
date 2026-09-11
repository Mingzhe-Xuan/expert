#!/bin/bash
#SBATCH --job-name=expert-dpa4-smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/dpa4-adapter-%j.out
#SBATCH --error=logs/slurm/dpa4-adapter-%j.err

set -euo pipefail

: "${EXPERT_DPA4_VENV:?Set EXPERT_DPA4_VENV to the recorded DPA4 environment}"
source "${EXPERT_DPA4_VENV}/bin/activate"
mkdir -p logs/slurm results/backbones
python -m src.cli.dpa4_smoke \
  --device cuda \
  --output "results/backbones/dpa4-${SLURM_JOB_ID}.json"
