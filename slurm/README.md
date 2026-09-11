# Slurm entry points

These non-interactive jobs are the only supported path for full tests, model inference,
training smoke runs, and batch data processing on Guqq. Activate the recorded project
virtual environment through `EXPERT_VENV`; outputs belong under ignored `logs/` and
`results/` directories. Every server session must first follow `AGENTS.md`: record the
connection purpose, connect, and run `git pull` before submitting a job.
