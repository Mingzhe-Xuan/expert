#!/bin/bash
#SBATCH --job-name=pretrain-prepare
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/pretrain-prepare-%A_%a.out
#SBATCH --error=logs/slurm/pretrain-prepare-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
: "${EXPERT_GMTNET_ROOT:=/home/xmz/expert-data/vendor/GMTNet}"
: "${EXPERT_GMTNET_VENV:=/home/xmz/expert-envs/gmtnet-py310}"
: "${EXPERT_DPA4_VENV:=/home/xmz/expert-envs/dpa4-py310}"
: "${EXPERT_PRETRAIN_ROOT:=results/pretrain/20261001}"
tasks=(dielectric elastic)
task="${tasks[${SLURM_ARRAY_TASK_ID:-0}]}"
extra=()
if [[ "${EXPERT_PRETRAIN_SMOKE:-0}" == 1 ]]; then extra+=(--smoke); fi
cache="${EXPERT_PRETRAIN_ROOT}/cache/${task}"
mkdir -p "${cache}"
git rev-parse HEAD > "${cache}/git.txt"
"${EXPERT_DPA4_VENV}/bin/python" -m pip freeze > "${cache}/feature-environment.txt"
"${EXPERT_GMTNET_VENV}/bin/python" -m pip freeze > "${cache}/training-environment.txt"
if [[ "${EXPERT_PRETRAIN_SMOKE:-0}" == 1 ]]; then
  "${EXPERT_DPA4_VENV}/bin/python" -m src.experiments.pretrain.prepare \
    --task "${task}" --mode verify --output "${cache}/native-equivalence.json" --smoke
fi
"${EXPERT_DPA4_VENV}/bin/python" -m src.experiments.pretrain.prepare \
  --task "${task}" --mode features --output "${cache}/features.pt" "${extra[@]}"
"${EXPERT_DPA4_VENV}/bin/python" -m src.experiments.pretrain.prepare \
  --task "${task}" --mode features --inference --output "${cache}/inference_features.pt" "${extra[@]}"
"${EXPERT_GMTNET_VENV}/bin/python" -m src.experiments.pretrain.prepare \
  --task "${task}" --mode graphs --official-root "${EXPERT_GMTNET_ROOT}" \
  --output "${cache}/graphs.pt" "${extra[@]}"
