# Symmetry

Symmetry canonicalization produces a `SymmetryRecord`; point-group permutations are audit-only.
Relative-position parent routing consumes a versioned `ParentDAGSpec` built deterministically from
the frozen 32-group asset. Every cover edge stores the full parent rotation group, all oriented
maximal child subsets, the source-asset hash, and a checksum.

```python
dag = ParentDAGSpec(material_id="sample", current_point_group_number=32, embeddings=())
```

`PointGroupRegistry()` validates the frozen 32-group candidate asset and provides O(3)
representation matrices, invariant projectors, deterministic bases, and basis checksums.
`canonicalize_structure` repeats symmetry discovery with an explicit Hall number, rotates
positions/cell into spglib's idealized frame without reordering sites, and records both model-facing
Cartesian rotations and Hall-affine fractional rotations. The material fractional-cell/canonical-
Cartesian-frame common-cell convention is explicit; audited species-preserving permutations remain
diagnostics only.

`PointGroupAncestorDAG.maximal_paths()` supplies the stable class topology.
`build_point_group_parent_dag` reads the same asset's oriented subgroup instances and constructs
every reachable cover edge. `validate_point_group_parent_dag` requires exact cover-edge and
maximal-path equality with the asset. This route has no relaxed multi-`symprec` discovery and no
external Hall/common-cell registry.

`route_material_on_point_group_dag` computes one residual per cover edge from species-labelled graph
`edge_vectors`. For every stored child orientation it applies only rotations in `parent \\ child`,
performs species-pair-preserving Hungarian matching against the original relative-vector multiset,
and retains the minimum oriented RMS. This is translation- and origin-independent. Cache schema 3
stores sample ID, current PG and residuals; the full DAG is reconstructed and hash-checked on load.

`PointGroupAncestorDAG` is the separate class-level routing contract used by the reduced CGCNN
parent experiment. It validates the frozen 32-class/80-cover-edge asset offline, enumerates every
maximal current-to-root path, and also retains the old deduplicated all-ancestor lookup as a
separately labelled equal-weight ablation.
