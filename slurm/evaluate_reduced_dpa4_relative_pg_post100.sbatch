#!/bin/bash
#SBATCH --job-name=expert-dpa-relpg-post100
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/dpa4-relative-pg-post100-%j.out
#SBATCH --error=logs/slurm/dpa4-relative-pg-post100-%j.err

set -euo pipefail
: "${EXPERT_DPA4_VENV:?Set the recorded DPA4 environment}"
source "${EXPERT_DPA4_VENV}/bin/activate"
run_root="results/reduced-benchmark/dpa4-relative-pg-post100-eval"
mkdir -p logs/slurm "${run_root}"
git rev-parse HEAD > "${run_root}/git-${SLURM_JOB_ID}.txt"
python -m pip freeze > "${run_root}/environment-${SLURM_JOB_ID}.txt"

evaluate_one() {
  local label="$1"
  local source_root="$2"
  local source_job="$3"
  local checkpoint_sha256="$4"
  shift 4
  local checkpoint="${source_root}/checkpoint-${source_job}-epoch-120.pt"
  echo "${checkpoint_sha256}  ${checkpoint}" | sha256sum -c -
  python -m src.cli.reduced_dpa4_relative_pg_evaluate \
    --device cuda \
    --cache-root results/reduced-benchmark/cache \
    --feature-shards 64 \
    --batch-size 64 \
    --minimum-checkpoint-epoch-exclusive 100 \
    --hidden-multiplicities "$@" \
    --source-summary "${source_root}/summary-${source_job}.json" \
    --checkpoint "${checkpoint}" \
    --predictions "${run_root}/predictions-${SLURM_JOB_ID}-${label}.jsonl" \
    --summary "${run_root}/summary-${SLURM_JOB_ID}-${label}.json" \
    --junit "${run_root}/junit-${SLURM_JOB_ID}-${label}.xml"
}

evaluate_one 56d results/reduced-benchmark/dpa4-relative-pg 488 \
  e0a950b5e30ceefa1465a7c3943271b716157ad0a3ecb3e7c4deb4dae2d589c7 8 2 2 2 2
evaluate_one 64d results/reduced-benchmark/dpa4-relative-pg-64d 505 \
  d8f0094501e65c5214002cfd7409d047540a83121b0ab235147aecac1f9fdb2e 16 2 2 2 2
evaluate_one 80d results/reduced-benchmark/dpa4-relative-pg-80d 506 \
  195b2096f89dcca4a456b4b95bc76dd79354ccca6064ab6242803db6541fd88a 8 3 3 3 3
