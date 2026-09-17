# Symmetry

Symmetry canonicalization produces a `SymmetryRecord`; point-group permutations are
audit-only. Physical parent routing consumes a versioned `ParentDAGSpec` whose Hall-level
embeddings include settings, transforms, full operations, atom correspondence, Wyckoff
splitting, domain variant, and checksum. The generic 32-group lattice is not a physical
parent DAG.

```python
dag = ParentDAGSpec(material_id="sample", current_hall_number=1, embeddings=())
```

`PointGroupRegistry()` validates the frozen 32-group candidate asset and provides O(3)
representation matrices, invariant projectors, deterministic bases, and basis checksums.
`canonicalize_structure` repeats symmetry discovery with an explicit Hall number, rotates
positions/cell into spglib's idealized frame without reordering sites, and records audited
species-preserving permutations for diagnostics only.

`discover_material_parent_routing` constructs an auditable material-specific DAG from the actual
structure. It repeats spglib discovery on a frozen increasing `symprec` schedule, accepts only a
distinct higher point group whose common-cell affine operations contain the current operations,
limits the operation-index jump, and computes a continuous residual from species-matched periodic
site displacement plus lattice-metric violation. Each accepted common-cell Hall embedding carries
the full parent operations, identity site correspondence, Wyckoff splitting, convention/version,
and checksum. `save_parent_routing_cache` freezes these records separately from backbone features.
