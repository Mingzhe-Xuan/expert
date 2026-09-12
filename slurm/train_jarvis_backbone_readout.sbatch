#!/bin/bash
#SBATCH --job-name=expert-jarvis-br
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/benchmark-%j.out
#SBATCH --error=logs/slurm/benchmark-%j.err

set -euo pipefail

: "${EXPERT_BENCHMARK_BACKBONE:?Set mace, grace, or dpa4}"
: "${EXPERT_BENCHMARK_TARGET:?Set dielectric or elastic}"
case "${EXPERT_BENCHMARK_BACKBONE}" in
  mace)
    : "${EXPERT_MACE_VENV:?Set the recorded MACE/core environment}"
    EXPERT_SELECTED_VENV="${EXPERT_MACE_VENV}"
    ;;
  grace)
    : "${EXPERT_GRACE_VENV:?Set the recorded GRACE environment}"
    EXPERT_SELECTED_VENV="${EXPERT_GRACE_VENV}"
    export EXPERT_GRACE_TF_DEVICE="${EXPERT_GRACE_TF_DEVICE:-cpu}"
    ;;
  dpa4)
    : "${EXPERT_DPA4_VENV:?Set the recorded DPA4 environment}"
    EXPERT_SELECTED_VENV="${EXPERT_DPA4_VENV}"
    ;;
  *)
    echo "Only non-Eq backbones are enabled while EquiformerV2 download is paused" >&2
    exit 2
    ;;
esac

source "${EXPERT_SELECTED_VENV}/bin/activate"
run_root="results/benchmarks/${EXPERT_BENCHMARK_BACKBONE}-${EXPERT_BENCHMARK_TARGET}"
mkdir -p logs/slurm "${run_root}" results/benchmark-cache
git rev-parse HEAD > "${run_root}/git-${SLURM_JOB_ID}.txt"
python -m pip freeze > "${run_root}/environment-${SLURM_JOB_ID}.txt"
python -m src.cli.benchmark_train \
  --backbone "${EXPERT_BENCHMARK_BACKBONE}" \
  --target "${EXPERT_BENCHMARK_TARGET}" \
  --device cuda \
  --cache-root results/benchmark-cache \
  --checkpoint "${run_root}/checkpoint-${SLURM_JOB_ID}.pt" \
  --summary "${run_root}/summary-${SLURM_JOB_ID}.json" \
  --junit "${run_root}/junit-${SLURM_JOB_ID}.xml" \
  --max-epochs "${EXPERT_BENCHMARK_EPOCHS:-200}" \
  --batch-size "${EXPERT_BENCHMARK_BATCH_SIZE:-16}" \
  --learning-rate "${EXPERT_BENCHMARK_LR:-0.001}" \
  --patience "${EXPERT_BENCHMARK_PATIENCE:-30}"
