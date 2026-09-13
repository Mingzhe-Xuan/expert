#!/bin/bash
#SBATCH --job-name=expert-pg-reduce
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/pg-reduce-%j.out
#SBATCH --error=logs/slurm/pg-reduce-%j.err

set -euo pipefail

: "${EXPERT_MACE_VENV:?Set the recorded MACE/core environment}"
source "${EXPERT_MACE_VENV}/bin/activate"
candidate="results/tensor-curation/reduced-candidate-${SLURM_JOB_ID}"
mkdir -p logs/slurm "${candidate}" data/processed/curated_tensors/reduced_gt_5pct
git rev-parse HEAD > "${candidate}/git.txt"
python -m pip freeze > "${candidate}/environment.txt"
python -m data.curation.reduce_point_groups \
  --output-dir data/processed/curated_tensors/reduced_gt_5pct \
  --output-manifest "${candidate}/curated_tensors_reduced_gt_5pct.json"
