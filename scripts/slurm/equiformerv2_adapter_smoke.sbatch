#!/bin/bash
#SBATCH --job-name=expert-eqv2-smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/equiformerv2-adapter-%j.out
#SBATCH --error=logs/slurm/equiformerv2-adapter-%j.err

set -euo pipefail

: "${EXPERT_VENV:?Set EXPERT_VENV to the recorded fairchem-core 1.10.0 environment}"
source "${EXPERT_VENV}/bin/activate"
mkdir -p logs/slurm results/backbones
python -m src.cli.equiformerv2_smoke \
  --device cuda \
  --output "results/backbones/equiformerv2-${SLURM_JOB_ID}.json"
