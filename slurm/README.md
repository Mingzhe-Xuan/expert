# Slurm entry points

`train_reduced_dpa4_full_pg.sbatch` and `train_reduced_gmtnet.sbatch` form the paired custom
dielectric-total benchmark. They consume the same manifest-frozen 5,001/637/677 split and export
per-ID test predictions for the common evaluator. The DPA4 job fixes `B+A+PGE+R/full_pg/full_o3`
with seven current-point-group experts. The GMTNet job verifies the official checkout commit before
using its model and records the dedicated environment. Preprocessing caches and all run artifacts
remain under ignored `results/reduced-benchmark/`.

Full DPA4 extraction defaults to 64 deterministic cache partitions consumed by four persistent
GPU workers. `EXPERT_DPA4_FEATURE_SHARDS` controls recovery granularity, while
`EXPERT_DPA4_FEATURE_WORKERS` controls concurrency; they are deliberately separate so variable-cost
crystals cannot strand one monolithic worker. Each completed partition is independently provenance
checked and reusable, and the final process restores the frozen manifest order before training.
Smoke mode retains four partitions by default.

These non-interactive jobs are the only supported path for full tests, model inference,
training smoke runs, and batch data processing on Guqq. Activate the recorded project
backbone-specific virtual environments through `EXPERT_MACE_VENV`, `EXPERT_GRACE_VENV`,
`EXPERT_DPA4_VENV`, and `EXPERT_EQUIFORMERV2_VENV`; outputs belong under ignored `logs/` and
`results/` directories. Every server session must first follow `AGENTS.md`: record the
connection purpose, connect, and run `git pull` before submitting a job.

The three acceptance launchers persist Git and package fingerprints, JUnit XML,
machine-readable summaries, Slurm identifiers, and scheduler-captured stdout/stderr.
After completion, acceptance still requires checking every task with `sacct`; a generated
report is not evidence that the scheduler state was `COMPLETED`.

The four standalone real-checkpoint adapter launchers follow the same evidence contract: exact
Git revision, `pip freeze`, JSON execution metadata, JUnit, and scheduler-captured stdout/stderr.
Their CLI wrappers persist failure artifacts before returning a nonzero exit code.

The required 32-PG fixture builder and JARVIS-DFPT preparation jobs also emit job-level JSON and
JUnit through the same wrapper, in addition to Git and environment fingerprints. The BEC job only
passes when validation is clean and the extracted record count equals the official index count.

`train_jarvis_backbone_readout.sbatch` runs one full published-split JARVIS dielectric or elastic
experiment selected by `EXPERT_BENCHMARK_BACKBONE` and `EXPERT_BENCHMARK_TARGET`. Frozen source
features are cached under ignored `results/benchmark-cache/` so readout hyperparameter retries do
not rerun the checkpoint. The job emits Git/environment fingerprints, a best checkpoint, JSON,
JUnit, and scheduler logs. The launcher permits MACE/GRACE/DPA4 only while EqV2 access is paused.

`build_jarvis_elastic_manifest.sbatch` reproduces GMTNet's second-stage structural-symmetry screen
inside Slurm and writes a candidate 14,220-record manifest under ignored `results/dataset-protocol/`.
The candidate must be copied back, inspected, tested, and committed locally; the server job never
edits the Git-managed production manifest.

`curate_tensor_datasets.sbatch` performs the full DTNet/GMTNet/MatTen physical audit,
point-group projection checks, robust outlier analysis, and conservative duplicate merge on CPU.
Large JSONL outputs go to ignored `data/processed/curated_tensors/`; compact candidate manifests and
reports go to `results/tensor-curation/candidate-$SLURM_JOB_ID/` for local inspection and promotion.

Both mixed-backbone arrays use `select_backbone_venv.sh`. Their frozen schedules map array
index modulo four to MACE, GRACE, DPA4, and EquiformerV2 respectively. The full core test suite,
fixture builder, and BEC preparation use the MACE/core environment; standalone adapter jobs use
their matching environment. A single combined environment is unsupported because the frozen
backbone releases have incompatible e3nn and PyTorch requirements.

After every submitted job is terminal, create an ignored manifest containing the real IDs:

```json
{
  "schema_version": 1,
  "jobs": [
    {"name": "test_all", "job_id": "12345"},
    {"name": "point_group_smoke", "job_id": "12346", "array": {"start": 0, "end": 57}},
    {"name": "real_subset_smoke", "job_id": "12347", "array": {"start": 0, "end": 19}}
  ]
}
```

Run the lightweight login-node audit with
`python -m src.cli.slurm_audit --manifest results/acceptance/jobs.json --raw results/acceptance/sacct.txt --output results/acceptance/sacct-audit.json`.
It queries allocation-only pipe-delimited records and fails unless every expected single job or
array task appears exactly once with state `COMPLETED` and exit code `0:0`. Array parent and step
rows cannot conceal a missing task.
