# Experts

This module contains shared O(3) adaptation, routed O(3) experts, A1-only PG experts,
Full-PG experts, continuous residual gates, and hierarchical fusion. Each active PG
expert contains exactly two independently parameterized blocks. Fusion occurs only in a
common O(3) target space.

```python
output = expert(features, graph, symmetry, parent_dag)
```

`O3Adaptation` and `RoutedO3Expert` place the selected TP backend on complete directed
PBC edges. `A1PointGroupExpert` processes only invariant coordinates;
`FullPointGroupExpert` subduces and retains every real finite-group carrier. Both PG modes
contain exactly two independent blocks (with an explicit C1 identity bypass). Continuous
residual gates operate only on Hall-validated, deduplicated active sets before fusion in
the common O(3) layout.

`PointGroupTensorModel` is the frozen five-branch downstream dispatcher. It consumes a
real `O3FeatureBatch`, conditionally executes only configured modules, routes only through
validated Hall DAG nodes, fuses in the shared O(3) layout, and invokes one task readout.
It reports deduplicated active non-backbone parameters and per-sample active expert counts;
expert-free branches report zero, while routed branches count each validated Hall node once.
