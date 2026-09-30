# Global-experts runtime profiling

`python -m src.cli.global_experts_profile` accepts the training CLI arguments.
Use separate smoke graph/routing caches and `--smoke` for a real PG-stratified batch.
Default DPA feature caches, model parameters and loss are identical to training.
The first training batch is used; actual IDs, PGs, nodes and edges are reported.

Two warmup and three uninstrumented synchronized forward/backward passes precede
one torch.profiler pass. No optimizer steps occur; train-mode BatchNorm buffers
do update. DPA extraction, cache preparation and construction are excluded.
Outputs in a fresh directory: summary.json, trace.json (Chrome/Perfetto), and
CPU/CUDA operator tables. CUDA memory reports whole-model allocated/reserved peaks.
Module ranges describe forward calls, not module-attributed backward. Nested
inclusive times overlap; do not sum them or confuse kernel time with wall time.
Full backward operators and launch gaps are available in the trace. Local CPU
profiling validates instrumentation only; Guqq computation must use Slurm.

The report records actual interpreter/torch/e3nn paths and versions. Frame Wigner-D
construction has a separate `stage/frame_representation` range; this runs on CPU
for compatibility with older e3nn and is included in overall forward wall time.
