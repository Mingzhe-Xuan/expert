# Training

Training computes supervised loss in labelled irrep coefficient blocks and keeps each
dataset-by-property unit independent. Checkpoints store model/optimizer state,
train-split normalizers, configuration, and all numerical convention checksums; loading
fails closed on incompatibility.

```python
losses = coefficient_loss(prediction, target, normalizer)
```
