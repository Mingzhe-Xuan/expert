# Experts

`PointGroupTensorModel` defaults to `grouped_pg_gates=True` and
`vectorized_pg_routing=True`; `CachedBackboneTensorModel` exposes the same keyword
options. Set both false for the original execution path. Grouping lives in
`optimized.py`, shared with the GMTNet wrapper and retaining identical state-dict keys.
The general model constructor keeps `pg_weighting="legacy"` for existing callers.
Set `pg_weighting="within_cross_chain"` for the same router as the GMTNet wrapper,
shared under `chain_routing.py` (wrapper import paths remain compatible). It uses
near-current within-chain stick breaking and softmax over immediate-parent energies.
The dedicated DPA relative-PG training CLI defaults to this **new** weighting. Exposed
controls: `chain_temperature=1.0`, `initial_sigma=.08`, `sigma_floor=1e-8`.
There are no independent chain logits: edge sigmas are trainable, residuals are detached
metadata, exactly as in the wrapper. Static ancestor routing and O(3)/A1 experts retain
their established routing. Legacy dynamic PG routing uses root-to-current order and
node-count priors, preserving its residual gradients.
Topology caches are bounded and contain no autograd values.
PG-only dispatch skips unused local edge graphs, retaining independent
CUDA streams. O(3) expert dispatch remains on the reference routing/graph path.
Canonicalization is already performed at input preparation; no wrapper frame cache is
needed here. Input/output contracts, training losses and optimizer ownership are unchanged.

Old full-PG checkpoints load directly with `pg_weighting="legacy"`. To explicitly change
algorithms, use `migrate_legacy_edge_scales(state_dict, model.edge_gate)`; for a cached
backbone model pass `model.downstream.edge_gate` and `prefix="downstream.edge_gate."`.
This converts per-edge logits to the shared vector while preserving positive sigmas and
all non-routing weights. It intentionally changes predictions; create a **new optimizer**
rather than reuse the old optimizer's parameter state. New checkpoints record routing
temperature/floor/edge order and reject mismatches before state mutation.

This module contains shared O(3) adaptation, routed O(3) experts, A1-only PG experts,
Full-PG experts, continuous residual gates, and hierarchical fusion. Each active PG
expert contains exactly two independently parameterized blocks. Fusion occurs only in a
common O(3) target space.

The default hidden O(3) layout is shared by every architecture mode:
`8x0e + 2x1o + 2x2e + 2x3o + 2x4e` (56 components). Checkpoints created with the former
28-component full-PG layout are shape-incompatible and must not be resumed as the new layout.

```python
output = expert(features, graph, symmetry, parent_dag)
```

`O3Adaptation` and `RoutedO3Expert` place the selected TP backend on complete directed
PBC edges. `A1PointGroupExpert` processes only invariant coordinates;
`FullPointGroupExpert` subduces and retains every real finite-group carrier. Both PG modes
contain exactly two independent blocks (with an explicit C1 identity bypass). Material routing
enumerates every maximal current-to-root point-group path, gives paths the normalized prior
`len(path) / sum(len(paths))`, converts every incremental edge residual with
`a = 1 - exp(-(r/sigma)^2)`, and performs root-to-current stick-breaking inside the path. It then
sums contributions for a PG reached by multiple paths before normalized common-space fusion. Sigma
is shared by stable offline edge-template ID; material embedding checksums only identify residual
instances and never create per-sample parameters.

`PointGroupTensorModel` is the frozen five-branch downstream dispatcher. It consumes a
real `O3FeatureBatch`, conditionally executes only configured modules, routes only through
validated point-group DAG nodes, fuses in the shared O(3) layout, and invokes one task readout.
It reports deduplicated active non-backbone parameters and per-sample active expert counts;
expert-free branches report zero, while routed branches count each unique active PG expert once.
For the offline class-DAG ablation, the dispatcher instead accepts one validated PG number per sample,
looks up current plus every transitive parent class, executes every deduplicated expert, and fuses
them with equal weights in the shared O(3) layout. Residual-weighted and static PG inputs are mutually
exclusive, and current-only routing retains the original interface.

Execution is expert-batched rather than structure-serial. The dispatcher first plans every
structure's active experts and weights, then collates all structures assigned to the same expert
into one node/graph sub-batch. Each non-empty expert therefore runs once per input batch. Weighted
outputs are scattered back to the original node order before readout. On CUDA, distinct expert
buckets are launched on persistent per-expert streams with explicit producer and join dependencies;
on CPU, the same grouped algorithm runs synchronously and deterministically. Routing weights and
checkpoint parameters are unchanged by this scheduling policy.
