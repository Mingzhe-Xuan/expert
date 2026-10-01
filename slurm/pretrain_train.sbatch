#!/bin/bash
#SBATCH --job-name=pretrain-train
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/pretrain-train-%A_%a.out
#SBATCH --error=logs/slurm/pretrain-train-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
: "${EXPERT_GMTNET_ROOT:=/home/xmz/expert-data/vendor/GMTNet-pinned-7a606a4}"
: "${EXPERT_GMTNET_VENV:=/home/xmz/expert-envs/gmtnet-py310}"
: "${EXPERT_PRETRAIN_ROOT:=results/pretrain/20261001}"
extra=()
if [[ "${EXPERT_PRETRAIN_SMOKE:-0}" == 1 ]]; then extra+=(--smoke --epochs 2); fi
"${EXPERT_GMTNET_VENV}/bin/python" -m src.experiments.pretrain.train \
  --index "${SLURM_ARRAY_TASK_ID:?array index required}" \
  --official-root "${EXPERT_GMTNET_ROOT}" --cache "${EXPERT_PRETRAIN_ROOT}/cache" \
  --output "${EXPERT_PRETRAIN_ROOT}/runs" "${extra[@]}"
