# Irreps

Immutable `IrrepLayout` values define real-harmonic component and copy order.
`O3FeatureBatch` pairs those layouts with node/edge tensors and mappings. Repeated target
copies use distinct labels. `ConventionMetadata` is embedded in checkpoints and rejects
incompatible subduction, CG, path, copy, or harmonic conventions.

```python
layout = IrrepLayout((IrrepTerm(2, 0, "e", "scalars"),))
features = O3FeatureBatch(x, layout, node_batch)
```
