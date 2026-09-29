#!/bin/bash
#SBATCH --job-name=gmtnet-attn-test
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/gmtnet-attn-test-%j.out
#SBATCH --error=logs/slurm/gmtnet-attn-test-%j.err

set -euo pipefail
: "${EXPERT_GMTNET_VENV:?Set the recorded GMTNet environment}"
: "${EXPERT_GMTNET_ROOT:?Set the pinned official GMTNet checkout}"
test -f "${EXPERT_GMTNET_ROOT}/gmtnet.py"
source "${EXPERT_GMTNET_VENV}/bin/activate"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
run_root="results/gmtnet-attention-test/${SLURM_JOB_ID}"
mkdir -p "${run_root}"
git rev-parse HEAD
python -m pytest tests/test_gmtnet_attention.py::test_legacy_and_enabled_full_model \
  -v --junitxml="${run_root}/junit.xml"
