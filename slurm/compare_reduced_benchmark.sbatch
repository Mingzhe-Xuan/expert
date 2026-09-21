#!/bin/bash
#SBATCH --job-name=expert-red-compare
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/reduced-comparison-%j.out
#SBATCH --error=logs/slurm/reduced-comparison-%j.err

set -euo pipefail
: "${EXPERT_CGCNN_VENV:?Set the recorded comparison environment}"
: "${EXPERT_REDUCED_DPA4_PREDICTIONS:?Set the accepted DPA4 prediction JSONL}"
: "${EXPERT_REDUCED_GMTNET_PREDICTIONS:?Set the accepted GMTNet prediction JSONL}"
: "${EXPERT_REDUCED_CGCNN_PREDICTIONS:?Set the completed CGCNN prediction JSONL}"
: "${EXPERT_REDUCED_CGCNN_PARENT_PREDICTIONS:?Set the completed all-ancestor PG-parent-DAG prediction JSONL}"
: "${EXPERT_REDUCED_CGCNN_RELATIVE_PARENT_PREDICTIONS:?Set the completed relative-PG prediction JSONL}"

source "${EXPERT_CGCNN_VENV}/bin/activate"
run_root="results/reduced-benchmark/comparison"
mkdir -p logs/slurm "${run_root}"

python -m src.cli.compare_reduced_benchmark \
  --dpa4 "${EXPERT_REDUCED_DPA4_PREDICTIONS}" \
  --gmtnet "${EXPERT_REDUCED_GMTNET_PREDICTIONS}" \
  --cgcnn-full-pg "${EXPERT_REDUCED_CGCNN_PREDICTIONS}" \
  --cgcnn-parent-dag "${EXPERT_REDUCED_CGCNN_PARENT_PREDICTIONS}" \
  --cgcnn-relative-parent-dag "${EXPERT_REDUCED_CGCNN_RELATIVE_PARENT_PREDICTIONS}" \
  --summary "${run_root}/summary-${SLURM_JOB_ID}.json" \
  --table "${run_root}/table-${SLURM_JOB_ID}.md"
