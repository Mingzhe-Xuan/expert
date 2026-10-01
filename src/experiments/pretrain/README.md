# Paired label-efficiency experiment

Inputs: checksum-pinned `data/manifests/curated_tensors.json` recommended records,
restricted to provenance `source_dataset=gmtnet` (JARVIS-DFT). Total dielectric is
electronic + ionic, dimensionless; elastic stiffness is in GPa. Labels have already
been screened and projected onto crystal symmetry by the recorded curation pipeline.
This is a curated JARVIS benchmark, not the original paper's unmodified benchmark.
All available point groups are retained. Existing duplicate-group train/validation/test
membership is preserved; other source datasets are excluded.

## Modules and interfaces

- `protocol.py`: source hash validation, deterministic nested 25/50/75/100% subsets,
  fixed sampling seed 20261001, 48-run grid (two tasks, two models, seeds 42/43/44).
- `data.py`: tensor contracts and explicit elastic component conversion.
- `prepare.py`: Slurm GPU feature extraction to compact per-node invariant embeddings;
  official graph and mask construction; fresh test-structure feature timing.
- `features.py`: preserve canonicalization and the two-pass even/odd parity formula,
  using the pretrained model's native neighbor builder. Omit redundant generic
  Python neighbor enumeration, which is ignored by that extractor. Slurm smoke
  compares two actual structures against the original extraction path per task.
- `constraints.py`: batch the original equality adjustment across structures while
  retaining the original ordered pair updates. Every training run verifies values
  and gradients against the original function on up to 64 actual training masks.
- `train.py`: one grid entry per process, original per-task model architecture,
  paired training, checkpointing, test metrics and inference timing.
- `good_result/pretrain/`: report and figures from recorded artifacts.

The two models differ only in the scalar atom inputs (92 elemental descriptors versus
640 frozen per-irrep invariant features). Both use the pinned original model without
the optional extra equivariant attention. Original per-task defaults are preserved:
Huber loss, AdamW (1e-3 LR, 1e-5 weight decay), linear per-step decay to 1e-5,
batch 64 with training drop-last, 200 epochs; dielectric mask enabled, elastic
feature mask disabled, official equality adjustment in both. No early stopping.
Training seeds do not alter subsets. Validation/test are complete, without drop-last.
The prespecified checkpoint selector is validation Fnorm in the second half of
training, shared by both models/tasks. This explicitly differs from the historical
dielectric runner's MAE selector. Training curves show online training loss and
validation loss/Fnorm, without comparing losses between tasks.

Elastic output uses the original xx,yy,zz,xy,yz,xz convention. Reports use the standard
xx,yy,zz,yz,xz,xy permutation; Fnorm/RMSE/EwT are invariant to this joint permutation.
Fnorm is mean per-structure matrix Frobenius error (3x3 or 6x6), not full rank-four norm.
EwT is a percentage below 5/10/25% relative Fnorm with denominator norm(target)+1e-5.

## Timing

CUDA is synchronized at stage boundaries. Training seconds include optimization,
validation, checkpoint writes and per-epoch history writes; exclude loading, graph/
feature preparation and final test/inference. Preparation is measured record by record.
Each subset's preparation cost is the sum for its actual train+validation records,
plus initialization and the full cache serialization overhead (conservative).
This is allocated measured work, not a separate cold wall-time run per seed. The
features are physically computed once and reused across seeds/models/fractions.
Report graph preparation separately. Total = feature preparation + graph preparation
+ cached training. Test preparation is excluded from training cost.

New-structure latency includes freshly recomputed features, graph/mask construction,
and batch-one prediction on the first 32 frozen test structures. Independent stages
are measured in their compatible environments with resident models and summed per
structure; feature initialization is reported separately. It is not a monolithic
service latency or a cache-hit latency. The first feature extraction can include
kernel warmup, making this estimate conservative. All models use the same IDs.

## Execution

Use recorded `/home/xmz/expert-envs/{dpa4,gmtnet}-py310` environments. No installation
or source edits on the server. Preparation and training must run through Slurm.
Use unique result roots for smoke retries; completed artifacts are never overwritten.

```bash
EXPERT_PRETRAIN_ROOT=results/pretrain/smoke-v1 EXPERT_PRETRAIN_SMOKE=1 \
  sbatch --array=0-1%1 slurm/pretrain_prepare.sbatch
# After successful preparation, run smoke indexes 0,1,24,25 with the same variables.
sbatch --array=0,1,24,25%1 slurm/pretrain_train.sbatch
# Only after smoke acceptance, prepare full cache then submit the 48-run grid:
sbatch --array=0-1%1 slurm/pretrain_prepare.sbatch
sbatch --array=0-47%1 slurm/pretrain_train.sbatch
```

Set and export `EXPERT_PRETRAIN_ROOT` and `EXPERT_PRETRAIN_SMOKE` for every submission;
the smoke environment prefix in the first example applies only to that command.
Schedulers should add `afterok` dependencies when queueing preparation and training.
Local focused checks: `python -m pytest tests/test_pretrain_protocol.py -q`.
