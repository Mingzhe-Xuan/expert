# Reduced tensor datasets: point-group frequency >5%

These datasets are filtered independently within each property subtype from the verified
`recommended` datasets. A point group is retained only when
`point_group_count / property_recommended_records > 0.05`; a group at exactly 5% is excluded.

| Property subtype | Retained | Original recommended | Retained share | Available point groups |
|---|---:|---:|---:|---|
| `dielectric_electronic` | 6,150 | 10,548 | 58.30% | `2/m`, `mm2`, `mmm`, `4/mmm`, `-3m`, `-43m`, `m-3m` |
| `dielectric_ionic` | 6,230 | 10,961 | 56.84% | `2/m`, `mm2`, `mmm`, `4/mmm`, `-3m`, `-43m`, `m-3m` |
| `dielectric_total` | 6,315 | 11,088 | 56.95% | `2/m`, `mm2`, `mmm`, `4/mmm`, `-3m`, `-43m`, `m-3m` |
| `elastic_stiffness` | 15,864 | 21,143 | 75.03% | `2/m`, `mmm`, `4/mmm`, `-3m`, `6/mmm`, `m-3m` |

The four JSONL files are under
`data/processed/curated_tensors/reduced_gt_5pct/` and retain every original structure, tensor,
provenance, quality, duplicate and split field. Each row adds:

```json
{
  "reduction": {
    "source_scope": "recommended",
    "criterion": "within_property_point_group_frequency_strictly_greater_than",
    "frequency_threshold": 0.05,
    "property_available_point_groups": ["..."],
    "point_group_count": 1132,
    "point_group_frequency": 0.10731892301858172,
    "property_recommended_records": 10548
  }
}
```

Exact point-group and split counts, file sizes and SHA-256 digests are recorded in
[`data/manifests/curated_tensors_reduced_gt_5pct.json`](../../data/manifests/curated_tensors_reduced_gt_5pct.json).
The extraction ran as Slurm job 436 from commit `61797c3` and completed with exit `0:0` in 5 seconds.
