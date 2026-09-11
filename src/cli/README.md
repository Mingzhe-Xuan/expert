# CLI

Non-interactive entry points prepare data, train independent units, run test suites, and
serialize reports. Every command accepts frozen configs and seeds, returns nonzero on
failure, and is suitable for the Slurm scripts required by `GOAL.md`.

```text
python -m src.cli.test --config path/to/run.json
```
