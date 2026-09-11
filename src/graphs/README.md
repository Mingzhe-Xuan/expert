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
