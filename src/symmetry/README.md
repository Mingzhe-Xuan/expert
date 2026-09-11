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
