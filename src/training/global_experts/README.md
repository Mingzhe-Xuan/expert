# Global plus experts training

`data.py` binds canonical frames and detached Hungarian routing residuals to exact
official graph rows, dataset/split identities, registry hash, and cache SHA-256.
`runner.py` trains all enabled branches using masked Huber loss and AdamW, saves the
best validation-MAE checkpoint and interval checkpoints, and evaluates the test split
with the selected checkpoint. It does not change the existing benchmark runners.

Call `train_global_experts(model, splits, output_dir=..., provenance=..., config=...)`.
Each row contains the official graph, feature/equality masks, target, sample ID, and a
`CrystalRouting`; optional boolean `target_mask` applies to training targets. Validation
and test require complete targets for the existing full-tensor benchmark definition.
Use a fresh output directory. The final partial training batch is retained.

## CLI and Slurm

```bash
python -m src.cli.global_experts_train \
  --official-root data/sources/GMTNet \
  --graph-cache results/global-pg/cache/graphs.pt \
  --routing-cache results/global-pg/cache/routing.pt \
  --dpa-cache-root results/reduced-benchmark/cache --feature-shards 64 \
  --output-dir results/global-pg/run-001 \
  --device cuda --epochs 200 --batch-size 64 \
  --expert-irreps '8x0e + 2x1o + 2x2e + 2x3o + 2x4e' \
  --initial-sigma 0.08 --chain-temperature 1.0 --branch-initial-logit -4
```

The CLI currently uses the repository's frozen reduced dielectric dataset and pinned
official graph builder. `--manifest` selects its manifest; `--smoke` selects the
existing PG-stratified smoke IDs and requires separate matching cache paths.
`--prepare-only` builds/validates caches without training. Every model and training
dataclass field has a CLI option (underscores become hyphens). Boolean options have
both positive and `--no-...` forms. For example, `--use-equiv-attn`, `--freeze-global`,
`--no-auxiliary-enabled`; `--adapter-initial-logit 0` is the algorithm default.
Run `python -m src.cli.global_experts_train --help` for the complete interface.

Default input is `--input-features dpa4`: reuse the existing frozen DPA4 feature shards
under `results/reduced-benchmark/cache` (64 shards per split by default). The existing
loader checks backbone/checkpoint/dataset/split identities and node species/order;
each irrep copy is reduced to an O(3)-invariant scalar before replacing graph inputs.
DPA stays frozen; GMTNet, Adapter, experts and fusion/routing parameters train jointly.
Missing or incompatible caches fail rather than silently falling back. Override the
cache root/shard count using the options above; Slurm forwards these CLI options too.
Use `--input-features cgcnn` to restore the original 92-dimensional input explicitly.

`--global-checkpoint PATH --global-checkpoint-sha256 HASH` initializes the global
branch from an official GMTNet checkpoint. Only new attention parameters may be missing
when attention is explicitly enabled. Incompatible carrier/input shapes fail closed.
`--node-features PATH --node-feature-sha256 HASH` accepts a frozen invariant-node
archive, including invariant DPA features, with per-row official graph digests and
unchanged node order. See `attach_invariant_inputs` for its tensor-only archive schema.
This explicit archive overrides default DPA shard loading (its actual kind is recorded);
it cannot be combined with `--input-features cgcnn`.
Unreduced equivariant components must not be passed as invariant node features.

Outputs: `best.pt`, optional `epoch-NNNN.pt`, `summary.json`, and ordered
`predictions.jsonl`. Checkpoints include model/config/asset/graph/routing provenance,
optimizer state and step; `load_checkpoint` checks provenance before strict loading.
The CLI supports global warm-start, not automatic interrupted-training continuation.
Only train targets may have missing-entry masks; held-out full-tensor metrics require
complete targets. All graph preparation and training on Guqq must run through Slurm.

From the repository root, set `EXPERT_GMTNET_VENV`, `EXPERT_GMTNET_ROOT`,
`EXPERT_GLOBAL_GRAPH_CACHE`, and `EXPERT_GLOBAL_ROUTING_CACHE`, ensure `logs/slurm`
exists, then submit `sbatch slurm/train_global_experts.sbatch [CLI options]`.
Its result directory is isolated by Slurm job ID. No existing launcher is changed.
