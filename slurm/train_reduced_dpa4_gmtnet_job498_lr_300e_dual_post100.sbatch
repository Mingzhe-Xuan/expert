#!/bin/bash
#SBATCH --job-name=expert-dpa-gmt498lr300p100
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/dpa4-gmtnet-job498-lr-300e-dual-post100-%j.out
#SBATCH --error=logs/slurm/dpa4-gmtnet-job498-lr-300e-dual-post100-%j.err

set -euo pipefail
: "${EXPERT_GMTNET_VENV:?Set the recorded GMTNet environment}"
: "${EXPERT_GMTNET_ROOT:?Set the pinned official GMTNet checkout}"
source "${EXPERT_GMTNET_VENV}/bin/activate"
run_root="results/reduced-benchmark/dpa4-gmtnet-job498-lr-300e-dual-post100"
mkdir -p logs/slurm "${run_root}" results/reduced-benchmark/cache
git rev-parse HEAD > "${run_root}/git-${SLURM_JOB_ID}.txt"
python -m pip freeze > "${run_root}/environment-${SLURM_JOB_ID}.txt"

python -m src.cli.reduced_dpa4_gmtnet_train \
  --official-root "${EXPERT_GMTNET_ROOT}" \
  --device cuda \
  --cache results/reduced-benchmark/cache/gmtnet-graphs.pt \
  --dpa-cache-root results/reduced-benchmark/cache \
  --feature-shards 64 \
  --epochs 300 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --end-learning-rate 0.00001 \
  --learning-rate-decay-epochs 200 \
  --minimum-checkpoint-epoch-exclusive 100 \
  --weight-decay 0.00001 \
  --seed 42 \
  --checkpoint-interval 20 \
  --select-validation-fnorm \
  --checkpoint "${run_root}/checkpoint-${SLURM_JOB_ID}-validation-mae.pt" \
  --predictions "${run_root}/predictions-${SLURM_JOB_ID}-validation-mae.jsonl" \
  --fnorm-checkpoint "${run_root}/checkpoint-${SLURM_JOB_ID}-validation-fnorm.pt" \
  --fnorm-predictions "${run_root}/predictions-${SLURM_JOB_ID}-validation-fnorm.jsonl" \
  --summary "${run_root}/summary-${SLURM_JOB_ID}.json" \
  --junit "${run_root}/junit-${SLURM_JOB_ID}.xml"
