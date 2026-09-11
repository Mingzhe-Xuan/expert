# Backbones

Adapters load the revision-pinned MACE, GRACE, DPA4, and EquiformerV2 checkpoints and
return `O3FeatureBatch` before each model's scalar readout. Inputs are
`PeriodicGraph` batches; output layouts explicitly record degree, parity, multiplicity,
and component order. Backbones are frozen by default. DPA4 and EquiformerV2 must use
the inversion-paired wrapper rather than relabelling SO(3) features.

```python
# Phase B public shape
features = adapter(graph)  # O3FeatureBatch
```
