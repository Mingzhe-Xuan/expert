#!/bin/bash
#SBATCH --job-name=expert-dpa-gmt
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/dpa4-gmtnet-%j.out
#SBATCH --error=logs/slurm/dpa4-gmtnet-%j.err

set -euo pipefail
: "${EXPERT_GMTNET_VENV:?Set the recorded GMTNet environment}"
: "${EXPERT_GMTNET_ROOT:?Set the pinned official GMTNet checkout}"
source "${EXPERT_GMTNET_VENV}/bin/activate"
run_root="results/reduced-benchmark/dpa4-gmtnet"
mkdir -p logs/slurm "${run_root}" results/reduced-benchmark/cache
git rev-parse HEAD > "${run_root}/git-${SLURM_JOB_ID}.txt"
python -m pip freeze > "${run_root}/environment-${SLURM_JOB_ID}.txt"

smoke_args=()
graph_cache="results/reduced-benchmark/cache/gmtnet-graphs.pt"
checkpoint_interval="${EXPERT_DPA4_GMTNET_CHECKPOINT_INTERVAL:-20}"
minimum_checkpoint_epoch_exclusive="${EXPERT_GMTNET_MINIMUM_CHECKPOINT_EPOCH_EXCLUSIVE:-100}"
if [[ "${EXPERT_REDUCED_SMOKE:-0}" == "1" ]]; then
  smoke_args+=(--smoke)
  graph_cache="results/reduced-benchmark/cache/gmtnet-smoke-graphs.pt"
  checkpoint_interval="${EXPERT_DPA4_GMTNET_CHECKPOINT_INTERVAL:-1}"
  minimum_checkpoint_epoch_exclusive="${EXPERT_GMTNET_MINIMUM_CHECKPOINT_EPOCH_EXCLUSIVE:-0}"
fi

python -m src.cli.reduced_dpa4_gmtnet_train \
  --official-root "${EXPERT_GMTNET_ROOT}" \
  --device cuda \
  --cache "${graph_cache}" \
  --dpa-cache-root results/reduced-benchmark/cache \
  --feature-shards "${EXPERT_DPA4_FEATURE_SHARDS:-64}" \
  --epochs "${EXPERT_DPA4_GMTNET_EPOCHS:-200}" \
  --batch-size "${EXPERT_DPA4_GMTNET_BATCH_SIZE:-64}" \
  --learning-rate "${EXPERT_DPA4_GMTNET_LEARNING_RATE:-0.001}" \
  --end-learning-rate "${EXPERT_DPA4_GMTNET_END_LEARNING_RATE:-0.00001}" \
  --weight-decay "${EXPERT_DPA4_GMTNET_WEIGHT_DECAY:-0.00001}" \
  --seed "${EXPERT_DPA4_GMTNET_SEED:-42}" \
  --checkpoint-interval "${checkpoint_interval}" \
  --minimum-checkpoint-epoch-exclusive "${minimum_checkpoint_epoch_exclusive}" \
  "${smoke_args[@]}" \
  --checkpoint "${run_root}/checkpoint-${SLURM_JOB_ID}.pt" \
  --predictions "${run_root}/predictions-${SLURM_JOB_ID}.jsonl" \
  --summary "${run_root}/summary-${SLURM_JOB_ID}.json" \
  --junit "${run_root}/junit-${SLURM_JOB_ID}.xml"
