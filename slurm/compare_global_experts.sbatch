#!/bin/bash
#SBATCH --job-name=compare-global-pg
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/compare-global-pg-%j.out
#SBATCH --error=logs/slurm/compare-global-pg-%j.err
set -euo pipefail
source /home/xmz/expert-envs/gmtnet-py310/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
run_root="results/global-pg-compare/${SLURM_JOB_ID}"
mkdir -p "${run_root}"
git rev-parse HEAD > "${run_root}/revision.txt"
python -m src.cli.global_experts_compare \
  --official-root /home/xmz/expert-data/vendor/GMTNet-pinned-7a606a4 \
  --graph-cache results/global-pg-profile/cache/smoke-graphs.pt \
  --routing-cache results/global-pg-profile/cache/smoke-routing.pt \
  --output-dir "${run_root}/smoke" \
  --smoke --device cuda --batch-size 64 "$@"
python -m src.cli.global_experts_compare \
  --official-root /home/xmz/expert-data/vendor/GMTNet-pinned-7a606a4 \
  --graph-cache results/global-pg-compare/cache/batch64-graphs.pt \
  --routing-cache results/global-pg-compare/cache/batch64-routing.pt \
  --output-dir "${run_root}/batch64" \
  --smoke --train-samples 64 --device cuda --batch-size 64 "$@"
