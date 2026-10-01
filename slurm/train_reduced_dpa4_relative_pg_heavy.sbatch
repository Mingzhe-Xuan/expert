#!/bin/bash
#SBATCH --job-name=expert-dpa-relpg-heavy
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/dpa4-relative-pg-heavy-%j.out
#SBATCH --error=logs/slurm/dpa4-relative-pg-heavy-%j.err

set -euo pipefail
: "${EXPERT_DPA4_VENV:?Set the recorded DPA4 environment}"
source "${EXPERT_DPA4_VENV}/bin/activate"
export OMP_NUM_THREADS=8
run_root="results/reduced-benchmark/dpa4-relative-pg-heavy/${SLURM_JOB_ID}"
mkdir -p "${run_root}"
git rev-parse HEAD > "${run_root}/git.txt"
python -m pip freeze > "${run_root}/environment.txt"
python -m pytest tests/test_heavy_pg.py tests/test_training_checkpoint.py -q \
  --junitxml="${run_root}/preflight.xml"
python -m src.cli.reduced_dpa4_relative_pg_train \
  --device cuda --cache-root results/reduced-benchmark/cache \
  --epochs 200 --batch-size 64 --learning-rate 0.001 --end-learning-rate 0.00001 \
  --weight-decay 0.00001 --seed 42 --checkpoint-interval 20 \
  --minimum-checkpoint-epoch-exclusive 100 --feature-shards 64 \
  --hidden-multiplicities 16 2 2 2 2 --pg-expert-depth 12 \
  --expected-trainable-parameters 769156 \
  --pg-weighting within_cross_chain --grouped-pg-gates --vectorized-pg-routing \
  --checkpoint "${run_root}/checkpoint.pt" \
  --predictions "${run_root}/predictions.jsonl" \
  --summary "${run_root}/summary.json" --junit "${run_root}/junit.xml"
