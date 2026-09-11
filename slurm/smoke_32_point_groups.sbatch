#!/bin/bash
#SBATCH --job-name=expert-pg32-smoke
#SBATCH --partition=compute
#SBATCH --array=0-57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/pg32-smoke-%A_%a.out
#SBATCH --error=logs/slurm/pg32-smoke-%A_%a.err

set -euo pipefail

: "${EXPERT_VENV:?Set EXPERT_VENV to the recorded project virtual environment}"
: "${EXPERT_PG_FIXTURES:?Set EXPERT_PG_FIXTURES to the promoted 32-PG manifest}"
source "${EXPERT_VENV}/bin/activate"
mkdir -p logs/slurm results/point-group-smoke
git rev-parse HEAD > "results/point-group-smoke/git-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
python -m pip freeze > "results/point-group-smoke/environment-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
python -m src.cli.point_group_smoke \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --fixtures "${EXPERT_PG_FIXTURES}" \
  --device cuda \
  --output "results/point-group-smoke/summary-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json" \
  --junit "results/point-group-smoke/junit-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.xml"
