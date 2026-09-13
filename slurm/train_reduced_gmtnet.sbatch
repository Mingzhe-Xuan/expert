#!/bin/bash
#SBATCH --job-name=expert-red-gmtnet
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/reduced-gmtnet-%j.out
#SBATCH --error=logs/slurm/reduced-gmtnet-%j.err

set -euo pipefail
: "${EXPERT_GMTNET_VENV:?Set the recorded GMTNet environment}"
: "${EXPERT_GMTNET_ROOT:?Set the pinned official GMTNet checkout}"
source "${EXPERT_GMTNET_VENV}/bin/activate"
run_root="results/reduced-benchmark/gmtnet"
mkdir -p logs/slurm "${run_root}" results/reduced-benchmark/cache
git rev-parse HEAD > "${run_root}/git-${SLURM_JOB_ID}.txt"
python -m pip freeze > "${run_root}/environment-${SLURM_JOB_ID}.txt"
smoke_args=()
cache_name="gmtnet-graphs.pt"
if [[ "${EXPERT_REDUCED_SMOKE:-0}" == "1" ]]; then
  smoke_args+=(--smoke)
  cache_name="gmtnet-smoke-graphs.pt"
fi
python -m src.cli.reduced_gmtnet_train \
  --official-root "${EXPERT_GMTNET_ROOT}" \
  --device cuda \
  --cache "results/reduced-benchmark/cache/${cache_name}" \
  --checkpoint "${run_root}/checkpoint-${SLURM_JOB_ID}.pt" \
  --predictions "${run_root}/predictions-${SLURM_JOB_ID}.jsonl" \
  --summary "${run_root}/summary-${SLURM_JOB_ID}.json" \
  --junit "${run_root}/junit-${SLURM_JOB_ID}.xml" \
  --epochs "${EXPERT_REDUCED_GMTNET_EPOCHS:-200}" \
  --batch-size "${EXPERT_REDUCED_GMTNET_BATCH_SIZE:-64}" \
  "${smoke_args[@]}"
