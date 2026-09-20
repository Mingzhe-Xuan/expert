#!/bin/bash
#SBATCH --job-name=expert-sg-errors
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/space-group-errors-%j.out
#SBATCH --error=logs/slurm/space-group-errors-%j.err

set -euo pipefail
: "${EXPERT_CGCNN_VENV:?Set the recorded CGCNN environment}"
: "${EXPERT_REDUCED_DATASET:?Set the frozen reduced dielectric-total JSONL}"
: "${EXPERT_REDUCED_CGCNN_PREDICTIONS:?Set the accepted current-pg prediction JSONL}"
: "${EXPERT_REDUCED_GMTNET_PREDICTIONS:?Set the accepted GMTNet prediction JSONL}"

source "${EXPERT_CGCNN_VENV}/bin/activate"
run_root="results/reduced-benchmark/space-group-analysis"
mkdir -p logs/slurm "${run_root}"

python -m src.cli.analyze_space_group_errors \
  --dataset "${EXPERT_REDUCED_DATASET}" \
  --current-pg "${EXPERT_REDUCED_CGCNN_PREDICTIONS}" \
  --gmtnet "${EXPERT_REDUCED_GMTNET_PREDICTIONS}" \
  --json "${run_root}/space-groups-${SLURM_JOB_ID}.json" \
  --csv "${run_root}/space-groups-${SLURM_JOB_ID}.csv" \
  --markdown "${run_root}/space-groups-${SLURM_JOB_ID}.md" \
  --plot "${run_root}/space-groups-${SLURM_JOB_ID}.svg"
