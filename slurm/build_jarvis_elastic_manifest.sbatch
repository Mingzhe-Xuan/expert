#!/bin/bash
#SBATCH --job-name=expert-elastic-protocol
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/elastic-protocol-%j.out
#SBATCH --error=logs/slurm/elastic-protocol-%j.err

set -euo pipefail

: "${EXPERT_MACE_VENV:?Set the recorded MACE/core environment}"
source "${EXPERT_MACE_VENV}/bin/activate"
mkdir -p logs/slurm results/dataset-protocol
git rev-parse HEAD > "results/dataset-protocol/git-${SLURM_JOB_ID}.txt"
python -m pip freeze > "results/dataset-protocol/environment-${SLURM_JOB_ID}.txt"
python data/build_manifest.py \
  --only elastic \
  --output-dir "results/dataset-protocol/candidate-${SLURM_JOB_ID}"
