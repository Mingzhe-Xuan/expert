# Tensor dataset curation

This module normalizes and audits DTNet/GMTNet dielectric tensors and GMTNet/MatTen elastic
stiffness tensors. It deliberately separates property semantics from source names:

- `dielectric_electronic`: clamped-ion/electronic relative permittivity;
- `dielectric_ionic`: lattice/ionic contribution to relative permittivity;
- `dielectric_total`: electronic plus ionic static relative permittivity;
- `elastic_stiffness`: fourth-rank stiffness tensor in GPa.

`sources.py` contains hash-gated source adapters. `physics.py` performs structural, intrinsic tensor,
positive-(semi)definiteness, mechanical-stability, and point-group checks. `dedup.py` constructs a
conservative primitive-cell fingerprint and resolves only same-subtype duplicates. `pipeline.py`
orchestrates screening, robust outlier flags, conflict-safe merging, deterministic group splits, and
machine-readable reports.

Generated full datasets and row-level audits are written below `data/processed/curated_tensors/` and
are intentionally Git-ignored. Compact manifests and descriptive/point-group reports are written to
`data/manifests/curated_tensors.json` and `docs/analysis/curated_tensor_datasets.{json,md}`.

Render the recommended-set point-group frequency figure directly from the compact JSON report
(no full dataset scan or plotting dependency is required):

```text
python -m data.curation.plot_point_groups
```

The deterministic SVG uses one count axis per property subtype and writes to
`docs/analysis/curated_tensor_point_group_frequency.svg` by default.

Build per-property reduced datasets containing only point groups whose frequency in that property's
recommended set is strictly greater than 5%:

```text
python -m data.curation.reduce_point_groups
```

Full real-data reduction must use `slurm/reduce_curated_tensor_point_groups.sbatch`. Outputs default
to ignored `data/processed/curated_tensors/reduced_gt_5pct/`. Every row preserves its original
structure, tensor, provenance and split, and adds a `reduction` object containing the strict
threshold, the property's complete `property_available_point_groups` list, and the row point group's
count/frequency. The compact candidate manifest records these fields again at dataset level together
with output hashes, byte sizes, record counts, point-group counts and split counts.

Run the complete real-data job through Slurm:

```text
python data/curate_tensor_datasets.py
```

Do not interpret a statistical outlier as automatically unphysical. The report exposes
`physical_valid` and `recommended` separately: the latter additionally removes extreme robust
outliers and unresolved duplicate-label conflicts.
