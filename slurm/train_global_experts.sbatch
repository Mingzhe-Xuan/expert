#!/bin/bash
#SBATCH --job-name=global-pg-experts
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/global-pg-experts-%j.out
#SBATCH --error=logs/slurm/global-pg-experts-%j.err

set -euo pipefail
: "${EXPERT_GMTNET_VENV:?Set the GMTNet virtual environment}"
: "${EXPERT_GMTNET_ROOT:?Set the pinned official GMTNet checkout}"
: "${EXPERT_GLOBAL_GRAPH_CACHE:?Set the matching official GMTNet graph cache}"
: "${EXPERT_GLOBAL_ROUTING_CACHE:?Set a new global-experts routing cache path}"
source "${EXPERT_GMTNET_VENV}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
launch_root="results/global-pg-experts/launch-${SLURM_JOB_ID}"
mkdir -p "${launch_root}"
git rev-parse HEAD > "${launch_root}/revision.txt"
printf '%s\n' "$@" > "${launch_root}/arguments.txt"
python -m src.cli.global_experts_train \
  --official-root "${EXPERT_GMTNET_ROOT}" \
  --graph-cache "${EXPERT_GLOBAL_GRAPH_CACHE}" \
  --routing-cache "${EXPERT_GLOBAL_ROUTING_CACHE}" \
  --output-dir "results/global-pg-experts/${SLURM_JOB_ID}" \
  --device cuda "$@"
