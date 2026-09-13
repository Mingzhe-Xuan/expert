#!/bin/bash
#SBATCH --job-name=expert-tensor-curation
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/tensor-curation-%j.out
#SBATCH --error=logs/slurm/tensor-curation-%j.err

set -euo pipefail

: "${EXPERT_MACE_VENV:?Set the recorded MACE/core environment}"
source "${EXPERT_MACE_VENV}/bin/activate"
candidate="results/tensor-curation/candidate-${SLURM_JOB_ID}"
mkdir -p logs/slurm "${candidate}" data/processed/curated_tensors
git rev-parse HEAD > "${candidate}/git.txt"
python -m pip freeze > "${candidate}/environment.txt"
python data/curate_tensor_datasets.py \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --output-dir data/processed/curated_tensors \
  --manifest "${candidate}/curated_tensors.json" \
  --report-json "${candidate}/curated_tensor_datasets.json" \
  --report-markdown "${candidate}/curated_tensor_datasets.md"
