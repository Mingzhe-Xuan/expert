#!/bin/bash
#SBATCH --job-name=fused-pg-probe
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/fused-pg-probe-%j.out
#SBATCH --error=logs/slurm/fused-pg-probe-%j.err
set -euo pipefail
source /home/xmz/expert-envs/gmtnet-py310/bin/activate
export OMP_NUM_THREADS=8
run_root="results/fused-pg-probe/${SLURM_JOB_ID}"
mkdir -p "${run_root}"
git rev-parse HEAD > "${run_root}/git.txt"
python -m pip freeze > "${run_root}/environment.txt"
python -m pytest tests/test_pg_probe.py tests/test_fused_pg_probe.py -q --junitxml="${run_root}/tests.xml"
python -m src.probes.fused_pg \
  --official-root /home/xmz/expert-data/vendor/GMTNet-pinned-7a606a4 \
  --graph-cache results/reduced-benchmark/cache/gmtnet-graphs.pt \
  --routing-cache results/global-pg-experts/cache/full-20260930-routing.pt \
  --output-dir "${run_root}" --device cuda
