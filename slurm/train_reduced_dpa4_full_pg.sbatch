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
feature_shards="${EXPERT_DPA4_FEATURE_SHARDS:-4}"
if ! [[ "${feature_shards}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPERT_DPA4_FEATURE_SHARDS must be a positive integer" >&2
  exit 2
fi
common_args=(
  --device cuda \
  --cache-root results/reduced-benchmark/cache \
  --max-epochs "${EXPERT_REDUCED_DPA4_EPOCHS:-200}" \
  --batch-size "${EXPERT_REDUCED_DPA4_BATCH_SIZE:-16}" \
  --patience "${EXPERT_REDUCED_DPA4_PATIENCE:-30}" \
  --feature-shards "${feature_shards}" \
  "${smoke_args[@]}"
)

if (( feature_shards > 1 )); then
  worker_threads=$(( SLURM_CPUS_PER_TASK / feature_shards ))
  if (( worker_threads < 1 )); then
    worker_threads=1
  fi
  worker_pids=()
  for (( shard_index=0; shard_index<feature_shards; shard_index++ )); do
    env OMP_NUM_THREADS="${worker_threads}" \
      DP_INTRA_OP_PARALLELISM_THREADS="${worker_threads}" \
      DP_INTER_OP_PARALLELISM_THREADS=1 \
      python -m src.cli.reduced_dpa4_train \
        "${common_args[@]}" \
        --prepare-only \
        --feature-shard-index "${shard_index}" \
        --checkpoint "${run_root}/prepare-checkpoint-${SLURM_JOB_ID}-${shard_index}.pt" \
        --predictions "${run_root}/prepare-predictions-${SLURM_JOB_ID}-${shard_index}.jsonl" \
        --summary "${run_root}/prepare-summary-${SLURM_JOB_ID}-${shard_index}.json" \
        --junit "${run_root}/prepare-junit-${SLURM_JOB_ID}-${shard_index}.xml" \
        > "${run_root}/prepare-stdout-${SLURM_JOB_ID}-${shard_index}.log" \
        2> "${run_root}/prepare-stderr-${SLURM_JOB_ID}-${shard_index}.log" &
    worker_pids+=("$!")
  done
  worker_status=0
  for worker_pid in "${worker_pids[@]}"; do
    if ! wait "${worker_pid}"; then
      worker_status=1
    fi
  done
  if (( worker_status != 0 )); then
    echo "one or more DPA4 feature shard workers failed" >&2
    exit 1
  fi
fi

python -m src.cli.reduced_dpa4_train \
  "${common_args[@]}" \
  --checkpoint "${run_root}/checkpoint-${SLURM_JOB_ID}.pt" \
  --predictions "${run_root}/predictions-${SLURM_JOB_ID}.jsonl" \
  --summary "${run_root}/summary-${SLURM_JOB_ID}.json" \
  --junit "${run_root}/junit-${SLURM_JOB_ID}.xml"
