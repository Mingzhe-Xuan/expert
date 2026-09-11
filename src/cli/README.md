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
