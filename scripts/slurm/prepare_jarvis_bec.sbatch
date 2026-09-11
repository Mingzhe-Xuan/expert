#!/usr/bin/env bash
#SBATCH --job-name=jarvis-bec
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=data/processed/slurm-jarvis-bec-%j.out
#SBATCH --error=data/processed/slurm-jarvis-bec-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
: "${EXPERT_MACE_VENV:?Set EXPERT_MACE_VENV to the recorded MACE/core environment}"
source "${EXPERT_MACE_VENV}/bin/activate"
mkdir -p data/raw/jarvis_dfpt/archives data/processed

git rev-parse HEAD > "data/processed/jarvis-dfpt-bec-git-${SLURM_JOB_ID}.txt"
python -m pip freeze > "data/processed/jarvis-dfpt-bec-environment-${SLURM_JOB_ID}.txt"
python -m src.cli.prepare_bec_job \
  --index-zip data/raw/jarvis_dfpt/raw_file_index.zip \
  --dataset-output data/processed/jarvis_dfpt_bec.jsonl \
  --archive-dir data/raw/jarvis_dfpt/archives \
  --errors data/processed/jarvis_dfpt_bec_errors.jsonl \
  --dataset-summary data/processed/jarvis_dfpt_bec_summary.json \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --summary "data/processed/jarvis-dfpt-bec-job-${SLURM_JOB_ID}.json" \
  --junit "data/processed/jarvis-dfpt-bec-job-${SLURM_JOB_ID}.xml"
