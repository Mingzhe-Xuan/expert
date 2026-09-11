# CLI

Non-interactive entry points prepare data, train independent units, run test suites, and
serialize reports. Every command accepts frozen configs and seeds, returns nonzero on
failure, and is suitable for the Slurm scripts required by `GOAL.md`.

```text
python -m src.cli.test --config path/to/run.json
```

`python -m src.cli.mace_smoke --device cuda --output results/backbones/mace.json`
runs the real manifest-pinned MACE checkpoint feature/frozen-gradient/equivariance smoke.
The provided `scripts/slurm/mace_adapter_smoke.sbatch` is the required execution path on Guqq;
the login node must not run this command directly.
