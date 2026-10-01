#!/bin/bash
#SBATCH --job-name=dpa-gmtnet-probe
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/gmtnet-pg-probe-%j.out
#SBATCH --error=logs/slurm/gmtnet-pg-probe-%j.err
set -euo pipefail
source /home/xmz/expert-envs/gmtnet-py310/bin/activate
export OMP_NUM_THREADS=4
run_root="results/gmtnet-pg-probe/${SLURM_JOB_ID}"
mkdir -p "${run_root}"
git rev-parse HEAD > "${run_root}/git.txt"
python -m pip freeze > "${run_root}/environment.txt"
python -m pytest tests/test_pg_probe.py tests/test_gmtnet_probe.py -q --junitxml="${run_root}/tests.xml"
python -m src.probes.gmtnet_pg --official-root /home/xmz/expert-data/vendor/GMTNet-pinned-7a606a4 --output "${run_root}" --device cuda
