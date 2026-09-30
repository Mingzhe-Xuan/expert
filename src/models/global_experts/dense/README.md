# Dense O(3) auxiliary branch

Replaces PG experts and chain routing with always-active residual message blocks.
Input/output: node features in the configured expert irreps, edge_index (receiver,
sender), and periodic displacement vectors. Output retains the input carrier.

Each block uses internally learned/shared full-O(3) TP path weights, spherical
harmonics up to edge_lmax, one radial-MLP scalar per edge, cosine cutoff, and
sqrt-degree aggregation. Even scalars and per-copy norms feed an invariant MLP;
its sigmoid output gates entire irrep copies. A bias-free equivariant Linear and
tanh-scaled identity residual finish the block. Geometry is reused within forward.
Hidden-carrier projections wrap only the residual, preserving exact identity at
zero residual logits. Internal block gradients can be zero on the first update;
they must become active after the residual logits move away from zero.

Dense depth, hidden irreps, radial/gate widths are real trainable-capacity controls.
Parameter matching is not FLOP matching: multiple blocks enlarge receptive fields.
This module does not standardize frames or use point-group metadata.
