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

python data/prepare_jarvis_bec.py \
  --index-zip data/raw/jarvis_dfpt/raw_file_index.zip \
  --output data/processed/jarvis_dfpt_bec.jsonl \
  --archive-dir data/raw/jarvis_dfpt/archives \
  --errors data/processed/jarvis_dfpt_bec_errors.jsonl \
  --workers "${SLURM_CPUS_PER_TASK}"

python data/summarize_jarvis_bec.py \
  data/processed/jarvis_dfpt_bec.jsonl \
  --errors data/processed/jarvis_dfpt_bec_errors.jsonl \
  --output data/processed/jarvis_dfpt_bec_summary.json
