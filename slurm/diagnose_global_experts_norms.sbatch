#!/bin/bash
#SBATCH --job-name=global-pg-norms
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/global-pg-norms-%j.out
#SBATCH --error=logs/slurm/global-pg-norms-%j.err
set -euo pipefail
source /home/xmz/expert-envs/gmtnet-py310/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
run_root="results/global-pg-norms/${SLURM_JOB_ID}"
mkdir -p "${run_root}"
git rev-parse HEAD > "${run_root}/revision.txt"
python -m pytest tests/test_global_experts_norms.py -q \
  --basetemp="${run_root}/test-tmp" --junitxml="${run_root}/tests.xml"
python -m src.cli.global_experts_norms \
  --official-root /home/xmz/expert-data/vendor/GMTNet-pinned-7a606a4 \
  --graph-cache results/reduced-benchmark/cache/gmtnet-graphs.pt \
  --routing-cache results/global-pg-experts/cache/full-20260930-routing.pt \
  --output-dir "${run_root}/diagnostics" --device cuda "$@"
