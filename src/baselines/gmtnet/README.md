# GMTNet baseline adapter

## Optional equivariant attention

`GMTNetConfig(use_equiv_attn=True)` enables invariant scalar attention in all three
equivariant TP layers. Both training CLIs accept `--use-equiv-attn`. The default is
False: the official model, parameter keys, initialization, and forward call remain
unchanged. The option is saved in checkpoint/report config and also supports DPA4 inputs.

For direct construction, use `build_gmtnet(official_model, args, use_equiv_attn=True)`;
the factory also reads `args.use_equiv_attn` when the keyword is omitted. Existing
args objects need no new fields. An already constructed official model can be passed
to `configure_equivariant_attention(model, use_equiv_attn=True)` before creating the
optimizer. Both return the model with its original `forward(data, feat_mask, equality)`.
The external official checkout is unchanged; direct calls to its constructor do not
activate this repository's extension.

`attention.py` retains each layer's TP and edge-weight MLP. It extracts signed `0e`
scalars and the norm of every other irrep copy (including `0o`), then computes
`MLP(concat(inv_i, inv_j, edge_features))`. The edge features are GMTNet's invariant
distance embedding. Softmax is over each receiver's neighbors; a single scalar
multiplies the entire TP message. Weighted **sum** replaces the original mean,
without a second degree normalization. Residual behavior is unchanged. In the official
TP implementation the receiver is `edge_index[0]`, unlike PyG's usual naming convention.

Old checkpoints load strictly with the default option. Enabled checkpoints include
additional `equi_update.nlayer_*.attn_mlp.*` parameters and must be reconstructed with
the option enabled. To initialize attention from a legacy checkpoint, explicitly use
`load_state_dict(..., strict=False)` and verify that only attention keys are missing;
these new weights require training. Attention preserves the TP stack's O(3) covariance;
it does not alter or strengthen the official downstream symmetry-mask/equality rules.

This module runs the official dielectric `GMTNet` model at commit
`7a606a459ee48a320ed38450e391811fb43d5e19`. It does not vendor that repository;
the caller supplies a checkout whose commit is verified before import.

The adapter recreates the official `symprec=1e-5` symmetry masks, equality masks,
16-nearest-neighbour graphs, Huber loss, AdamW optimizer, and linear polynomial
learning-rate decay. It deliberately removes WandB/path placeholders and evaluates
every validation/test record (the released script drops the last validation batch).
If the retired `torch_scatter` extension is unavailable, the adapter supplies
PyG's API-compatible `torch_geometric.utils.scatter`; GMTNet only uses sum/mean
reductions, so this does not change its model computation.
The released dielectric transformer imports `torch_sparse.SparseTensor` only as
a type annotation; when that retired extension is absent, the adapter provides
an inert annotation placeholder and no sparse operation is replaced.
The curated split and labels are never regenerated. Predictions are exported by
record ID and scored by `src.evaluation.tensor_benchmark_metrics`.
Every validation epoch records both component MAE and the mean per-sample
Frobenius distance (`validation_fnorm`) using that same dielectric metric
implementation. Best-checkpoint selection remains based on validation MAE.
Checkpoint selection is restricted by `minimum_checkpoint_epoch_exclusive`, which defaults to 100:
epoch 100 is ineligible and epoch 101 is the first eligible checkpoint. The same strict threshold
applies to the optional independent validation-Fnorm selector. The training horizon must contain at
least one eligible epoch; deliberately short diagnostic runs must explicitly lower the threshold.

`GMTNetConfig.learning_rate_decay_epochs` optionally decouples the linear decay horizon from the
total training horizon. The default `None` preserves decay across all epochs; setting it to 200 in a
300-epoch run reproduces the 200-epoch schedule and holds the endpoint LR for the remaining epochs.
`select_validation_fnorm=True`, together with distinct Fnorm checkpoint and prediction paths, adds
an independent Fnorm selector while retaining the legacy top-level MAE-selected report fields.
Both selectors are also recorded under `checkpoint_selections` with their own held-out metrics.


The runner also supports the explicit DPA4-input ablation used by
`src.cli.reduced_dpa4_gmtnet_train`. It reuses the frozen pre-interface DPA4 O(3)
feature cache and converts each irrep copy into one scalar: even degree-zero copies
retain their signed value, while every other copy contributes its Euclidean norm.
The resulting 640 invariant scalars replace only GMTNet's 92D CGCNN atom input;
the original atom projection output remains 128D and the official graph, message
passing, symmetry masks, tensor readout, loss, optimizer, schedule, and split stay
unchanged. Sample IDs, atom order, source width, cache checkpoint hash, and dataset
hash are validated before training.
