# Graphs

This module owns the cutoff periodic multigraph contract and covariance audit. A
`PeriodicGraph` contains directed image edges, integer cell shifts, displacements,
distances, and node-to-crystal mapping. Edge shells are never truncated by an arbitrary
neighbour count.

```python
from src.graphs import PeriodicGraph

graph = PeriodicGraph(positions, cell, species, batch, edges, shifts, vectors,
                      vectors.norm(dim=-1), cutoff=6.0)
```

`build_periodic_graph` derives adaptive reciprocal-space image bounds and enumerates all
images under the strict cutoff. `collate_periodic_graphs` offsets node and graph indices
without merging multiedges.
