#!/bin/bash
#SBATCH --job-name=expert-build-pg32
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/build-pg32-%j.out
#SBATCH --error=logs/slurm/build-pg32-%j.err

set -euo pipefail

: "${EXPERT_MACE_VENV:?Set EXPERT_MACE_VENV to the recorded MACE/core environment}"
source "${EXPERT_MACE_VENV}/bin/activate"
mkdir -p logs/slurm results/point-groups
python -m src.cli.build_point_group_fixtures \
  --output results/point-groups/crystallographic_point_groups.json
