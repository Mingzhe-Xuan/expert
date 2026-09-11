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

source slurm/select_backbone_venv.sh
select_backbone_venv "${SLURM_ARRAY_TASK_ID}"
source "${EXPERT_SELECTED_VENV}/bin/activate"
mkdir -p logs/slurm results/real-smoke
git rev-parse HEAD > "results/real-smoke/git-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
python -m pip freeze > "results/real-smoke/environment-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
python -m src.cli.real_subset_smoke \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --device cuda \
  --output-root results/real-smoke \
  --summary "results/real-smoke/summary-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json" \
  --junit "results/real-smoke/junit-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.xml"
