# Data

Data modules are independent for JARVIS dielectric, JARVIS elastic, DTNet Materials Project
dielectric, MatTen elastic, and JARVIS-DFPT BEC. Each owns its split manifest and train-only normalizer. Published splits
take priority; otherwise seed `20260911` produces a leakage-audited 8:1:1 split.

```python
from src.data import TrainingUnit, load_training_dataset

dataset = load_training_dataset(TrainingUnit("dtnet", "dielectric"))
```

`TrainingUnit` accepts only the five registered pairings. `make_seeded_split` performs the
fallback split over material/duplicate groups, never individual records.

`load_training_dataset` verifies the frozen source size and SHA-256 before parsing.
It supports the GMTNet JARVIS pickle schemas, DTNet's normalized JSONL, MatTen's column-oriented JSON, and the
calculation-matched JARVIS-DFPT JSONL. All become immutable `TensorSample` objects
with fractional structure data plus Cartesian and frozen-order irrep targets.
`load_five_structure_smoke` selects the first 3/1/1 records from the retained
published or seeded split; it does not redefine or shrink the underlying split.

DTNet's canonical `dielectric` label is the symmetric part of its published total tensor. The
processed row also retains raw electronic, ionic, and total tensors plus their antisymmetric
residuals. Its split exactly reproduces the two seed-3 `train_test_split` calls in the official
training script (5,318/665/665 for the published 6,648 rows).

`load_structure_candidates` is the structure-only, checksum-gated iterator used by
the 32-point-group fixture builder. It preserves source sample IDs and provenance,
and deliberately rejects BEC because that source is not an equilibrium benchmark.
