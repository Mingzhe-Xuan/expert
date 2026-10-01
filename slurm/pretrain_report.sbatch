#!/bin/bash
#SBATCH --job-name=pretrain-report
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/pretrain-report-%j.out
#SBATCH --error=logs/slurm/pretrain-report-%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
: "${EXPERT_GMTNET_VENV:=/home/xmz/expert-envs/gmtnet-py310}"
: "${EXPERT_PRETRAIN_ROOT:=results/pretrain/20261001}"
"${EXPERT_GMTNET_VENV}/bin/python" good_result/pretrain/report.py \
  --runs "${EXPERT_PRETRAIN_ROOT}/runs" --output good_result/pretrain
tar -czf "${EXPERT_PRETRAIN_ROOT}/delivery.tar.gz" \
  good_result/pretrain/*.png good_result/pretrain/*.svg \
  good_result/pretrain/*.json good_result/pretrain/report.html \
  "${EXPERT_PRETRAIN_ROOT}"/runs/*/*/*/*/summary.json \
  "${EXPERT_PRETRAIN_ROOT}"/runs/*/*/*/*/predictions.jsonl
sha256sum "${EXPERT_PRETRAIN_ROOT}/delivery.tar.gz" > "${EXPERT_PRETRAIN_ROOT}/delivery.sha256"
