# Irreps

Immutable `IrrepLayout` values define real-harmonic component and copy order.
`O3FeatureBatch` pairs those layouts with node/edge tensors and mappings. Repeated target
copies use distinct labels. `ConventionMetadata` is embedded in checkpoints and rejects
incompatible subduction, CG, path, copy, or harmonic conventions.

```python
layout = IrrepLayout((IrrepTerm(2, 0, "e", "scalars"),))
features = O3FeatureBatch(x, layout, node_batch)
```

`build_subduction_plan` restricts every provenance-labelled O(3) copy to a point group
and deterministically resolves all real finite-group irreducible copies through the
symmetric commutant. `finite_group_intertwiners` constructs CG paths from a Hom-space
Reynolds projector; both expose stable ordering checksums.
