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

`MACEBackboneAdapter` loads only the manifest-pinned local MACE-MP-0b3 medium file,
requires the frozen runtime version, and taps the first interaction state before the
energy readout. It converts MACE sender/receiver edges to the downstream target-to-source
vector convention without rebuilding or truncating the checkpoint graph.

`DPA4BackboneAdapter` loads the manifest-pinned DPA4-Plus checkpoint through DeePMD's
official `Tester`, reuses its sparse neighbor schema, and taps the SeZM descriptor's
final `[node, lm, 1, channel]` state. The adapter converts each degree block to e3nn's
copy-major layout, obtains explicit even/odd O(3) carriers from two real checkpoint
calls, and trains only the final interface projector.

`EquiformerV2BackboneAdapter` requires the exact gated `facebook/OMAT24`
`eqV2_31M_mp.pt` artifact and fairchem-core 1.10.0. It uses fairchem's official
checkpoint loader, taps the backbone's final normalized `[node, lm, channel]`
`node_embedding`, reuses the returned periodic radius graph, and applies the same
two-real-forward inversion construction as DPA4. The adapter fails closed before
importing fairchem while the gated artifact is unavailable; an OC20 checkpoint is
not a valid substitute.

`GRACEBackboneAdapter` verifies both the published archive and every extracted
TensorFlow checkpoint artifact. It taps the 4,512-component natural-parity `AA`
state—the common last equivariant state before the two higher-order scalar branches—
and derives a strictly validated TensorPotential-to-e3nn real-harmonic basis map.
