# Backbones

Adapters load the revision-pinned MACE, GRACE, DPA4, and EquiformerV2 checkpoints and
return `O3FeatureBatch` before each model's scalar readout. Inputs are
`PeriodicGraph` batches; output layouts explicitly record degree, parity, multiplicity,
and component order. Backbones are frozen by default. DPA4 and EquiformerV2 must use
the inversion-paired wrapper rather than relabelling SO(3) features.

```python
# Phase B public shape
features = adapter(graph)  # O3FeatureBatch
```

`BackboneResourceRegistry` validates the canonical four-entry manifest and fails closed
on absent, gated, size-mismatched, or checksum-mismatched checkpoints.
`InversionPairedReynolds` accepts a frozen SO(3) extractor, evaluates both the structure
and its periodic inversion with the same checkpoint, and emits explicit even/odd O(3)
blocks. `O3InterfaceProjector` is the trainable equivariant map into a downstream layout;
its parameters count as non-backbone parameters.
