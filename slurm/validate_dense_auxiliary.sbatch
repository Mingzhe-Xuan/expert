#!/bin/bash
#SBATCH --job-name=validate-dense-o3
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/dense-validation-%j.out
#SBATCH --error=logs/slurm/dense-validation-%j.err

set -euo pipefail
source /home/xmz/expert-envs/gmtnet-py310/bin/activate
export OMP_NUM_THREADS=8
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export EXPERT_GMTNET_ROOT=/home/xmz/expert-data/vendor/GMTNet-pinned-7a606a4
run_root="results/dense-validation/${SLURM_JOB_ID}"
mkdir -p "${run_root}"
git rev-parse HEAD > "${run_root}/revision.txt"
python -m pytest tests/test_dense_o3.py tests/test_global_experts_schedule.py \
  tests/test_global_experts_checkpoint_evaluation.py -q -p no:cacheprovider \
  --junitxml="${run_root}/tests.xml"
python -m src.cli.global_experts_profile \
  --official-root "${EXPERT_GMTNET_ROOT}" \
  --graph-cache results/reduced-benchmark/cache/gmtnet-graphs.pt \
  --routing-cache results/global-pg-experts/cache/full-20260930-routing.pt \
  --output-dir "${run_root}/profile" --device cuda --batch-size 64 \
  --auxiliary-type dense --match-pg-active-budget --epochs 300 --decay-epochs 200
