#!/bin/bash
#SBATCH --job-name=compare-pure-pg
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/compare-pure-pg-%j.out
#SBATCH --error=logs/slurm/compare-pure-pg-%j.err
set -euo pipefail
source /home/xmz/expert-envs/gmtnet-py310/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export EXPERT_GMTNET_ROOT=/home/xmz/expert-data/vendor/GMTNet-pinned-7a606a4
run_root="results/standalone-pg-compare/${SLURM_JOB_ID}"
mkdir -p "${run_root}"
git rev-parse HEAD > "${run_root}/revision.txt"
python -m pytest tests/test_standalone_pg_optimized.py tests/test_global_experts_optimized.py \
  tests/test_parent_dag_training.py tests/test_dispatcher_readout.py tests/test_training_checkpoint.py -q -p no:faulthandler \
  --junitxml="${run_root}/tests.xml"
python -m src.cli.standalone_pg_compare --output-dir "${run_root}/smoke" --train-samples 7 "$@"
python -m src.cli.standalone_pg_compare --output-dir "${run_root}/batch64" --train-samples 64 "$@"
