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
