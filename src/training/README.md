# Training

Training computes supervised loss in labelled irrep coefficient blocks and keeps each
dataset-by-property unit independent. Checkpoints store model/optimizer state,
train-split normalizers, configuration, and all numerical convention checksums; loading
fails closed on incompatibility.

`BenchmarkConfig.checkpoint_interval` is an opt-in exact-epoch archive. A positive interval writes
`<best-stem>-epoch-NNN.pt` after that epoch while leaving the validation-selected checkpoint at its
original path. Each archive contains the full resumable checkpoint payload, and the returned report
records its epoch, path, byte size, and SHA-256. The default `0` preserves prior runner behavior.

```python
losses = coefficient_loss(prediction, target, normalizer)
```

`CoefficientNormalizer.fit(..., split="train")` stores one statistic per labelled irrep
copy and rejects fitting on validation/test data or loading state from another unit.
`coefficient_mse` keeps those copies separate. Checkpoint loading verifies architecture,
unit, layout, and full convention metadata before mutating model or optimizer state.

`prepare_tensor_batch` canonicalizes structures and targets in the same frame while
preserving BEC site order. `run_five_structure_smoke` executes the frozen 3/1/1 chain:
train-only normalization, one optimizer update, validation, checkpoint save/load,
test inference, and per-copy raw-unit metrics. Its default builder always constructs
a manifest-pinned real backbone; dependency injection is only a unit-test seam and
does not count as backbone acceptance evidence.
