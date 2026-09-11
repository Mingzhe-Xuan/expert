# Data

Data modules are independent for JARVIS dielectric, JARVIS elastic, MatTen elastic, and
JARVIS-DFPT BEC. Each owns its split manifest and train-only normalizer. Published splits
take priority; otherwise seed `20260911` produces a leakage-audited 8:1:1 split.

```python
unit = load_training_unit("jarvis_dfpt", "bec", split_manifest)
```

`TrainingUnit` accepts only the four frozen pairings. `make_seeded_split` performs the
fallback split over material/duplicate groups, never individual records.
