# GMTNet baseline adapter

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


The runner also supports the explicit DPA4-input ablation used by
`src.cli.reduced_dpa4_gmtnet_train`. It reuses the frozen pre-interface DPA4 O(3)
feature cache and converts each irrep copy into one scalar: even degree-zero copies
retain their signed value, while every other copy contributes its Euclidean norm.
The resulting 640 invariant scalars replace only GMTNet's 92D CGCNN atom input;
the original atom projection output remains 128D and the official graph, message
passing, symmetry masks, tensor readout, loss, optimizer, schedule, and split stay
unchanged. Sample IDs, atom order, source width, cache checkpoint hash, and dataset
hash are validated before training.
