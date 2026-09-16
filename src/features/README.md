# Fixed input features

This module owns deterministic, non-learned node descriptors used before an equivariant model.
`cgcnn.py` reproduces GMTNet's `jarvis.core.specie.get_node_attributes(...,
atom_features="cgcnn")` input exactly: one 92-component even-scalar vector per atom. It validates
the full element table, exposes a content digest for cache provenance, and never constructs graphs
or targets.

The downstream learned `92 -> 128` scalar embedding lives in `src/models/cgcnn.py`. Keeping the
fixed descriptor separate from the learned model prevents this additive branch from changing any
pretrained-backbone adapter or cache.
