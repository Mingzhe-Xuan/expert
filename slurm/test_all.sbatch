#!/bin/bash
#SBATCH --job-name=expert-test-all
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/test-all-%j.out
#SBATCH --error=logs/slurm/test-all-%j.err

set -euo pipefail

: "${EXPERT_VENV:?Set EXPERT_VENV to the recorded project virtual environment}"
source "${EXPERT_VENV}/bin/activate"
mkdir -p logs/slurm results/test-all
git rev-parse HEAD > "results/test-all/git-${SLURM_JOB_ID}.txt"
python -m pip freeze > "results/test-all/environment-${SLURM_JOB_ID}.txt"
python -m src.cli.test_all \
  --junit "results/test-all/junit-${SLURM_JOB_ID}.xml" \
  --summary "results/test-all/summary-${SLURM_JOB_ID}.json"
