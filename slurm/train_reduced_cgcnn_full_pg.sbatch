#!/bin/bash
#SBATCH --job-name=expert-red-cgcnn-pg
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/reduced-cgcnn-full-pg-%j.out
#SBATCH --error=logs/slurm/reduced-cgcnn-full-pg-%j.err

set -euo pipefail
: "${EXPERT_DPA4_VENV:?Set the recorded DPA4/core environment}"
source "${EXPERT_DPA4_VENV}/bin/activate"
run_root="results/reduced-benchmark/cgcnn-full-pg"
mkdir -p logs/slurm "${run_root}" results/reduced-benchmark/cache
git rev-parse HEAD > "${run_root}/git-${SLURM_JOB_ID}.txt"
python -m pip freeze > "${run_root}/environment-${SLURM_JOB_ID}.txt"
smoke_args=()
if [[ "${EXPERT_REDUCED_SMOKE:-0}" == "1" ]]; then
  smoke_args+=(--smoke)
fi
python -m src.cli.reduced_cgcnn_full_pg_train \
  --device cuda \
  --cache-root results/reduced-benchmark/cache \
  --checkpoint "${run_root}/checkpoint-${SLURM_JOB_ID}.pt" \
  --predictions "${run_root}/predictions-${SLURM_JOB_ID}.jsonl" \
  --summary "${run_root}/summary-${SLURM_JOB_ID}.json" \
  --junit "${run_root}/junit-${SLURM_JOB_ID}.xml" \
  --epochs "${EXPERT_REDUCED_CGCNN_EPOCHS:-200}" \
  --batch-size "${EXPERT_REDUCED_CGCNN_BATCH_SIZE:-64}" \
  --learning-rate "${EXPERT_REDUCED_CGCNN_LEARNING_RATE:-0.001}" \
  --end-learning-rate "${EXPERT_REDUCED_CGCNN_END_LEARNING_RATE:-0.00001}" \
  --weight-decay "${EXPERT_REDUCED_CGCNN_WEIGHT_DECAY:-0.00001}" \
  "${smoke_args[@]}"
