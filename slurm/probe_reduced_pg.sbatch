#!/bin/bash
#SBATCH --job-name=dpa-pg-probe
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/pg-probe-%j.out
#SBATCH --error=logs/slurm/pg-probe-%j.err
set -euo pipefail
source /home/xmz/expert-envs/dpa4-py310/bin/activate
export OMP_NUM_THREADS=4
run_root="results/pg-probe/${SLURM_JOB_ID}"
mkdir -p "${run_root}"
git rev-parse HEAD > "${run_root}/git.txt"
python -m pip freeze > "${run_root}/environment.txt"
python -m pytest tests/test_pg_probe.py -q --junitxml="${run_root}/tests.xml"
python -m src.probes.pg_linear --output "${run_root}" --device cuda
