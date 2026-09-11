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

: "${EXPERT_MACE_VENV:?Set EXPERT_MACE_VENV to the recorded MACE/core environment}"
source "${EXPERT_MACE_VENV}/bin/activate"
mkdir -p logs/slurm results/backbones
git rev-parse HEAD > "results/backbones/mace-git-${SLURM_JOB_ID}.txt"
python -m pip freeze > "results/backbones/mace-environment-${SLURM_JOB_ID}.txt"
python -m src.cli.mace_smoke \
  --device cuda \
  --output "results/backbones/mace-${SLURM_JOB_ID}.json" \
  --junit "results/backbones/mace-${SLURM_JOB_ID}.xml"
