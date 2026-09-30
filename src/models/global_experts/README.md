# Global GMTNet plus hierarchical Full-PG experts

Performance paths default on for this additive model only: `grouped_pg_gates`,
`cache_frames`, `vectorized_routing`. Disable all three to reproduce the reference
execution path. `optimized.py` groups copy gates by width/type and owns a value-keyed
LRU frame cache (`frame_cache_size=8192`, cleared on model device/dtype conversion).
Derived indices/caches are nonpersistent; learned state-dict keys stay unchanged.
`vectorized_routing.py` caches validated immutable DAG plans and evaluates padded
prefix products in batches, including exact zero-gate cases. Residuals and learned
scales are never cached. With vectorized routing, the bias-free output projection
is applied once after weighted pooling; requested diagnostics still expose per-PG
projected features. Floating-point reduction order may differ, not the algorithm.

Standard-frame representation matrices are built from detached geometric metadata
on CPU, then moved to the feature device/dtype. This supports e3nn versions whose
Wigner generators otherwise mix CPU constants with CUDA angles. It does not detach
the learned features or alter feature/parameter gradients.

This package implements `docs/analysis/algorithm.md` independently of the existing
GMTNet baseline runner and PG dispatcher. It reuses the official encoder/readout and
the existing two-block `FullPointGroupExpert`, with new routing and Adapter modules.

`routing.py` consumes a validated complete material PG DAG and detached edge residuals.
It returns per-chain conditional weights, immediate-parent softmax probabilities, and
exactly collapsed per-PG coefficients. Edge scales are shared by offline edge ID.
Traversal starts with the edge nearest the current PG, as required by the algorithm;
the existing legacy router retains its own ordering and fixed length priors.

## Model API

```python
from src.models.global_experts import GlobalExpertsConfig, GlobalExpertsModel, CrystalRouting

config = GlobalExpertsConfig(
    expert_irreps="8x0e + 2x1o + 2x2e + 2x3o + 2x4e",
    initial_sigma=0.08, chain_temperature=1.0,
    branch_initial_logit=-4.0, use_equiv_attn=False,
)
model = GlobalExpertsModel(
    official_model, official_module.equality_adjustment,
    expert_numbers=active_pg_numbers, edge_ids=offline_edge_ids, config=config,
)
prediction, diagnostics = model(
    graph_batch, feature_masks, equality_masks, crystal_routing,
    return_diagnostics=True,
)
```

`graph_batch` uses the official PyG node/edge layout. `crystal_routing` contains one
`CrystalRouting(sample_id, dag, residuals, input_to_standard)` per crystal in the same
order. PG membership, all maximal paths, residual coverage, sample IDs (when supplied
on the graph), orthogonal frames, and carrier dimensions are validated. The data layer
binds node order and graph contents to cache digests; direct API callers must supply
the matching records. Output shape is the original readout's shape (dielectric `[B,3,3]`).

`adapter.py` applies an exact identity residual plus `tanh(zeta)` times a radial TP
message; `zeta=0` initially. There is no learned self connection. `frames.py` first
uses the row-lattice polar frame, then spglib standardization. The resulting frame
may have determinant -1 and is passed to the full parity-aware irrep representation.
It satisfies `R(gx) g = R(x)` for external rotations/reflections of the same lattice
and atom representation. Geometry and strict detection are detached preprocessing.

Each PG is evaluated once per mini-batch on its selected nodes. Outputs are restored
to the input frame, mean-pooled separately for each crystal, and mapped back to the
GMTNet carrier. Diagnostics expose omega, immediate-edge energies, pi, alpha, pooled
PG features, the global/fused carrier, and sigmoid(branch_logit). No per-PG or per-chain
trainable fusion logits are introduced. The branch is auxiliary by default;
`auxiliary_enabled=False` calls the supplied global model exactly.

All dataclass fields are public inputs. These include expert irrep multiplicities and
parities, Adapter backend (`full_o3`/`o2_tp`), mmax/lmax/cutoff/radial width/initial gate,
sigma initialization/floor, fixed chain temperature, branch initialization, C1 bypass,
global freeze, auxiliary disable, and optional equivariant attention. Freezing the global
branch also freezes its BatchNorm statistics. `metadata()` records layouts, interface
maps, conventions and asset identities for strict checkpoint reconstruction.

## Equivariance and compatibility

The new carrier branch is tested under proper and improper O(3) transforms, including
the data-derived frame and invariant Hungarian residuals. For tensor readout tests,
feature masks must transform with their carrier; Cartesian equality constraints must
also correspond to the chosen frame. The unchanged official componentwise equality
adjustment and clipped mask preprocessing are not a new proof of arbitrary-frame
equivariance. In particular, a fixed Cartesian equality mask cannot be reused after
arbitrary rotation. The standard-frame construction assumes the same ordered lattice
representation; invariance under a change of lattice basis is a separate issue.

Existing GMTNet and PG classes, runners, initialization, and routing remain unchanged.
Training uses the new `src.cli.global_experts_train` entry point; see
`src/training/global_experts/README.md`. No formal ablation or accuracy improvement is
claimed by the implementation/unit tests.

## Dense auxiliary ablation

`GlobalExpertsConfig(auxiliary_type="dense")` replaces experts/router with the
always-active [dense O(3) module](dense/README.md), keeping the input/output maps,
base Adapter, mean pooling, sigmoid lambda, mask and tensor readout. `pg` remains
the default; its checkpoint metadata and state-dict keys are preserved. `none`
disables the auxiliary contribution. Dense forwarding needs no routing records.

The training CLI accepts `--auxiliary-type dense --match-pg-active-budget` to select
depth, internal multiplicities and invariant gate width using only mean structural
active PG/router parameter counts over training crystals. It explicitly copies
shared initialization from a same-seed PG reference and verifies the analytic count
against instantiated parameters. Matching must be within1% of total auxiliary
capacity; actual replacement and total errors are stored in provenance. Explicit
`--dense-depth`, `--dense-hidden-irreps`, `--dense-radial-width`, `--dense-gate-width`
and `--dense-initial-logit` also support manual configurations without auto matching.
