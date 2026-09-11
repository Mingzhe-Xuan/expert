# Experts

This module contains shared O(3) adaptation, routed O(3) experts, A1-only PG experts,
Full-PG experts, continuous residual gates, and hierarchical fusion. Each active PG
expert contains exactly two independently parameterized blocks. Fusion occurs only in a
common O(3) target space.

```python
output = expert(features, graph, symmetry, parent_dag)
```
