# CLI

Non-interactive entry points prepare data, train independent units, run test suites, and
serialize reports. Every command accepts frozen configs and seeds, returns nonzero on
failure, and is suitable for the Slurm scripts required by `GOAL.md`.

```text
python -m src.cli.test --config path/to/run.json
```

`python -m src.cli.mace_smoke --device cuda --output results/backbones/mace.json --junit results/backbones/mace.xml`
runs the real manifest-pinned MACE checkpoint feature/frozen-gradient/equivariance smoke.
The provided `scripts/slurm/mace_adapter_smoke.sbatch` is the required execution path on Guqq;
the login node must not run this command directly.

All four standalone adapter CLIs use the same evidence wrapper: success and failure both write
JSON execution metadata plus a one-case JUnit file, while failures are re-raised for a nonzero
Slurm exit code.

`python -m src.cli.real_subset_smoke --index N --device cuda --output-root results/real-smoke`
runs one entry from the frozen 20-run real-data coverage schedule. On Guqq it is launched
only through `slurm/smoke_real_subsets.sbatch` and records checkpoint, metrics, Git commit,
runtime, CUDA device, and Slurm identifiers.

`python -m src.cli.benchmark_train --backbone mace --target dielectric ...` materializes the
frozen pre-interface backbone tap once, caches it with checkpoint/split metadata gates, and trains
the direct `B+R` readout on the complete published JARVIS split. It reports sample-mean Cartesian
Fnorm and EwT 25/10/5 in the original physical frame. On Guqq it must be run through
`slurm/train_jarvis_backbone_readout.sbatch`; the launcher intentionally rejects EquiformerV2 while
its gated checkpoint download is paused.
The CLI validates exact public split sizes before loading a backbone: `3770/471/471` for dielectric
and `11376/1422/1422` for elastic. Consequently, the known 14,480-record first-stage elastic
manifest is rejected until the official second-stage candidate is generated and promoted.

`python -m src.cli.reduced_cgcnn_full_pg_train ...` is the additive, explicitly
`current-group only` GMTNet-input ablation. It
leaves DPA4 untouched, materializes the exact 92D JARVIS CGCNN feature on the repository's
canonical PBC graph, applies a learned `92 -> 128` scalar embedding, and trains
`B+A+PGE+R/full_pg` with GMTNet-aligned Cartesian Huber, AdamW, linear learning-rate decay, and
validation-MAE selection. It emits the same prediction JSONL and common RMSE/Fnorm/EwT metrics as
the existing reduced runners and is launched only by `slurm/train_reduced_cgcnn_full_pg.sbatch`.

`python -m src.cli.reduced_cgcnn_parent_dag_train ...` is its matched parent-DAG experiment. It
reuses the immutable CGCNN feature caches and separately stores only the current PG number for each
ordered sample ID. A validated offline 32-point-group cover DAG supplies every transitive parent;
online forward passes activate and equally fuse the current expert plus all deduplicated parent PG
experts. No relaxed structure search, Hall embedding, residual, or `max_parents` cutoff participates.
Its optimizer, split, seed defaults, loss, schedule, checkpoint selection, and metrics match the
current-group-only run. On Guqq it is launched only by
`slurm/train_reduced_cgcnn_parent_dag.sbatch`.

`python -m src.cli.plot_training_history --summary ... --expected-sha256 ... --svg ... --png ...`
validates and renders the accepted CGCNN current-group-only epoch history. It is visualization only:
no prediction, target, or metric is recomputed.
Optional paired `--gmtnet-summary/--gmtnet-expected-sha256` arguments overlay GMTNet's natively
recorded training Huber loss and validation MAE in that same figure.

`python -m src.cli.build_point_group_fixtures --output results/point-groups/...json --summary results/point-groups/summary.json --junit results/point-groups/junit.xml`
scans the three frozen equilibrium structure sources and selects the canonical 32 fixtures.
It is batch data processing and must run through `slurm/build_point_group_fixtures.sbatch`
on Guqq, never directly on the login node.

`python -m src.cli.prepare_bec_job ... --summary ... --junit ...` owns the resumable full
JARVIS-DFPT preparation plus final validation as one auditable job. It rejects extraction errors,
invalid records, duplicates, and any record count short of the official index while preserving
the gathered counts in its failure summary.

`slurm/smoke_32_point_groups.sbatch` runs the frozen 58-row matrix: 32 mandatory
per-group PGE rows plus the exhaustive 26 architecture rows. Each row uses a real
checkpoint and records graph automorphism, target symmetry, backward, and parameter
budget evidence. `slurm/test_all.sbatch` is the aggregate project test entry point.
Its `src.cli.test_all` wrapper emits JUnit plus a JSON count/status summary even when
pytest returns a nonzero exit code, and makes any skipped/xfail item fail acceptance.
Both GPU array CLIs likewise preserve a one-case JUnit file and failure summary before
returning nonzero; their Slurm launchers also fingerprint Git and installed packages.

`python -m src.cli.slurm_audit --manifest ... --raw ... --output ...` is the strict post-run
accounting gate. It expands frozen array ranges and rejects missing/duplicate allocations,
non-`COMPLETED` states, and any exit code other than `0:0`.

`python -m src.cli.efficiency_report --point-group-glob 'results/point-group-smoke/summary-JOB_*.json' --real-subset-glob 'results/real-smoke/summary-JOB_*.json' --json-output results/efficiency/report.json --markdown-output results/efficiency/report.md`
requires the complete 58/20 row matrices and produces the paired machine-readable and Markdown
runtime report. Validation is fail-closed, including row identity, parameter budget, numeric
consistency, and the explicitly limited active-downstream FLOPs scope.
