# Slurm entry points

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

Both mixed-backbone arrays use `select_backbone_venv.sh`. Their frozen schedules map array
index modulo four to MACE, GRACE, DPA4, and EquiformerV2 respectively. The full core test suite,
fixture builder, and BEC preparation use the MACE/core environment; standalone adapter jobs use
their matching environment. A single combined environment is unsupported because the frozen
backbone releases have incompatible e3nn and PyTorch requirements.
