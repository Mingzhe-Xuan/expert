# Experts

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
