# Global-experts runtime profiling

## Standalone pure PG

`python -m src.cli.standalone_pg_compare --output-dir <fresh-path> --train-samples 64`
compares a legacy-weighting baseline and new-weighting reference/grouped/optimized models. The full pass
includes the 3200D frozen-DPA O(3) source interface, Adapter, full_pg and tensor readout;
unlike the wrapper it uses native DPA graphs. New weighting shares the wrapper router. Fresh routing
caches are derived from those graphs. CUDA expert streams are required and recorded.
Optimization flags preserve same-algorithm state-dict keys. Legacy scales are explicitly
migrated to equal positive sigmas for the new algorithm. Six warmups, seven synchronized passes, identical non-routing parameters/data/RNG,
Huber Cartesian loss, no optimizer update. Output and all parameter-gradient parity
must pass between the three new-weighting variants; legacy output equality is not expected.
Report same-algorithm speedup separately from legacy-to-new total change.
Traces and summaries are stored separately for each variant.
`slurm/compare_standalone_pg.sbatch` runs regression tests and real 7/64-crystal batches
in the same GPU allocation. Preparation is excluded from timed passes. Kernel counts
are device events, not CPU launch counts; nested module totals must not be summed.

`python -m src.cli.global_experts_compare` uses the same data/CLI and compares the
reference, grouped gates, grouped gates + frame cache, and all three optimizations.
Every variant starts from identical weights/buffers and RNG; outputs and parameter
gradients must match within explicit float32 tolerances before measurement. Six
warmups and seven timed passes precede each full trace. `compare_global_experts.sbatch`
runs the paired comparison on one GPU allocation. No optimizer update is performed.
For a real 64-crystal batch use `--smoke --train-samples 64 --batch-size 64` with
separate graph/routing cache filenames: this retains all seven PG representatives
then fills deterministically from the frozen training split, without mixing held-out IDs.

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

`python -m src.profiling.analyze_trace TRACE_JSON OUTPUT_JSON` attributes actual
kernel events using external IDs and forward/backward flow links. It excludes
GPU annotation spans and keeps unmatched work separate. The synthetic regression
test covers attribution across autograd threads and annotation exclusion.
