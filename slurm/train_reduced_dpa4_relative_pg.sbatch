#!/bin/bash
#SBATCH --job-name=expert-dpa-relpg
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/dpa4-relative-pg-%j.out
#SBATCH --error=logs/slurm/dpa4-relative-pg-%j.err

set -euo pipefail
: "${EXPERT_DPA4_VENV:?Set the recorded DPA4 environment}"
source "${EXPERT_DPA4_VENV}/bin/activate"
run_root="results/reduced-benchmark/dpa4-relative-pg"
mkdir -p logs/slurm "${run_root}" results/reduced-benchmark/cache
git rev-parse HEAD > "${run_root}/git-${SLURM_JOB_ID}.txt"
python -m pip freeze > "${run_root}/environment-${SLURM_JOB_ID}.txt"

smoke_args=()
feature_shards="${EXPERT_DPA4_FEATURE_SHARDS:-64}"
if [[ "${EXPERT_REDUCED_SMOKE:-0}" == "1" ]]; then
  smoke_args+=(--smoke)
  feature_shards="${EXPERT_DPA4_FEATURE_SHARDS:-1}"
fi
if ! [[ "${feature_shards}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPERT_DPA4_FEATURE_SHARDS must be a positive integer" >&2
  exit 2
fi

python -m src.cli.reduced_dpa4_relative_pg_train \
  --device cuda \
  --cache-root results/reduced-benchmark/cache \
  --epochs "${EXPERT_REDUCED_DPA4_EPOCHS:-200}" \
  --batch-size "${EXPERT_REDUCED_DPA4_BATCH_SIZE:-64}" \
  --learning-rate "${EXPERT_REDUCED_DPA4_LEARNING_RATE:-0.001}" \
  --end-learning-rate "${EXPERT_REDUCED_DPA4_END_LEARNING_RATE:-0.00001}" \
  --weight-decay "${EXPERT_REDUCED_DPA4_WEIGHT_DECAY:-0.00001}" \
  --seed "${EXPERT_REDUCED_DPA4_SEED:-42}" \
  --checkpoint-interval "${EXPERT_DPA4_CHECKPOINT_INTERVAL:-20}" \
  --feature-shards "${feature_shards}" \
  "${smoke_args[@]}" \
  --checkpoint "${run_root}/checkpoint-${SLURM_JOB_ID}.pt" \
  --predictions "${run_root}/predictions-${SLURM_JOB_ID}.jsonl" \
  --summary "${run_root}/summary-${SLURM_JOB_ID}.json" \
  --junit "${run_root}/junit-${SLURM_JOB_ID}.xml"
