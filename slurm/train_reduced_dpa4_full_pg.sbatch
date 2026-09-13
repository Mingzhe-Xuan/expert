#!/bin/bash
#SBATCH --job-name=expert-red-dpa4
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/reduced-dpa4-%j.out
#SBATCH --error=logs/slurm/reduced-dpa4-%j.err

set -euo pipefail
: "${EXPERT_DPA4_VENV:?Set the recorded DPA4 environment}"
source "${EXPERT_DPA4_VENV}/bin/activate"
run_root="results/reduced-benchmark/dpa4"
mkdir -p logs/slurm "${run_root}" results/reduced-benchmark/cache
git rev-parse HEAD > "${run_root}/git-${SLURM_JOB_ID}.txt"
python -m pip freeze > "${run_root}/environment-${SLURM_JOB_ID}.txt"
smoke_args=()
if [[ "${EXPERT_REDUCED_SMOKE:-0}" == "1" ]]; then
  smoke_args+=(--smoke)
fi
python -m src.cli.reduced_dpa4_train \
  --device cuda \
  --cache-root results/reduced-benchmark/cache \
  --checkpoint "${run_root}/checkpoint-${SLURM_JOB_ID}.pt" \
  --predictions "${run_root}/predictions-${SLURM_JOB_ID}.jsonl" \
  --summary "${run_root}/summary-${SLURM_JOB_ID}.json" \
  --junit "${run_root}/junit-${SLURM_JOB_ID}.xml" \
  --max-epochs "${EXPERT_REDUCED_DPA4_EPOCHS:-200}" \
  --batch-size "${EXPERT_REDUCED_DPA4_BATCH_SIZE:-16}" \
  --patience "${EXPERT_REDUCED_DPA4_PATIENCE:-30}" \
  "${smoke_args[@]}"
