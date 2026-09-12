# Local dataset point-group frequency and expert-balance audit

Generated: 2026-09-12 (Asia/Shanghai).

## Scope and interpretation

Counts use each frozen dataset split and spglib with `symprec=1e-5`, `angle_tolerance=-1`, matching model symmetry discovery. “Expert balance” below means the load of the **current-PG expert** on the training split. Exact routed load cannot yet be computed: parent experts require a validated, material-specific Hall-level `ParentDAGSpec`; the class-level subgroup lattice must not be substituted for it.

JARVIS-DFPT BEC is explicitly excluded from training-balance conclusions because its full processed output and split are pending; only the 10 successful local records are shown.

## Training-split balance summary

| Dataset | Train N | PG coverage | Zero PGs | Min nonzero | Max | Max/min | Max/uniform | CV (32 PGs) | Entropy / log 32 | Effective PGs | Largest share | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| jarvis_dielectric | 3770 | 30/32 | 2 | 5 | 419 | 83.8 | 3.56 | 0.994 | 0.852 | 19.2 | 11.1% | 严重不平衡 |
| jarvis_elastic | 11584 | 32/32 | 0 | 1 | 2571 | 2571.0 | 7.10 | 1.559 | 0.749 | 13.4 | 22.2% | 严重不平衡 |
| matten_elastic | 8230 | 32/32 | 0 | 1 | 2792 | 2792.0 | 10.86 | 2.108 | 0.657 | 9.7 | 33.9% | 严重不平衡 |
| jarvis_dfpt_bec_local_snapshot | — | — | — | — | — | — | — | — | — | — | — | 不可判定（正式 split 缺失） |

`Max/uniform` is the busiest expert count divided by `Train N / 32`; CV includes all 32 experts, including zero-count groups. Effective PGs is `exp(Shannon entropy)`.

## Frequency × PG

### jarvis_dielectric

GMTNet calculation-matched JARVIS dielectric. frozen train/validation/test IDs; pre-filter and unused records excluded.

| PG | train | validation | test | total |
|---|---:|---:|---:|---:|
| 1 | 178 | 28 | 20 | 226 |
| -1 | 78 | 12 | 8 | 98 |
| 2 | 180 | 18 | 14 | 212 |
| m | 249 | 17 | 27 | 293 |
| 2/m | 251 | 35 | 37 | 323 |
| 222 | 77 | 12 | 15 | 104 |
| mm2 | 287 | 42 | 40 | 369 |
| mmm | 263 | 43 | 28 | 334 |
| 4 | 12 | 0 | 2 | 14 |
| -4 | 51 | 6 | 8 | 65 |
| 4/m | 9 | 1 | 1 | 11 |
| 422 | 25 | 5 | 4 | 34 |
| 4mm | 43 | 3 | 2 | 48 |
| -42m | 184 | 26 | 21 | 231 |
| 4/mmm | 308 | 31 | 38 | 377 |
| 3 | 13 | 1 | 2 | 16 |
| -3 | 26 | 6 | 3 | 35 |
| 32 | 39 | 3 | 6 | 48 |
| 3m | 245 | 31 | 44 | 320 |
| -3m | 419 | 58 | 44 | 521 |
| 6 | 18 | 3 | 2 | 23 |
| -6 | 6 | 1 | 0 | 7 |
| 6/m | 0 | 0 | 0 | 0 |
| 622 | 5 | 1 | 1 | 7 |
| 6mm | 74 | 9 | 11 | 94 |
| -6m2 | 53 | 8 | 12 | 73 |
| 6/mmm | 114 | 7 | 7 | 128 |
| 23 | 36 | 2 | 8 | 46 |
| m-3 | 7 | 0 | 1 | 8 |
| 432 | 0 | 0 | 0 | 0 |
| -43m | 244 | 34 | 26 | 304 |
| m-3m | 276 | 28 | 39 | 343 |
| **Total** | **3770** | **471** | **471** | **4712** |

### jarvis_elastic

GMTNet calculation-matched JARVIS elastic. frozen train/validation/test IDs; pre-filter and unused records excluded.

| PG | train | validation | test | total |
|---|---:|---:|---:|---:|
| 1 | 163 | 19 | 22 | 204 |
| -1 | 275 | 32 | 48 | 355 |
| 2 | 122 | 14 | 16 | 152 |
| m | 185 | 21 | 18 | 224 |
| 2/m | 913 | 120 | 125 | 1158 |
| 222 | 61 | 8 | 11 | 80 |
| mm2 | 291 | 20 | 26 | 337 |
| mmm | 1090 | 131 | 143 | 1364 |
| 4 | 4 | 0 | 2 | 6 |
| -4 | 52 | 2 | 5 | 59 |
| 4/m | 79 | 5 | 13 | 97 |
| 422 | 19 | 2 | 3 | 24 |
| 4mm | 453 | 54 | 63 | 570 |
| -42m | 551 | 69 | 55 | 675 |
| 4/mmm | 1790 | 220 | 224 | 2234 |
| 3 | 9 | 1 | 3 | 13 |
| -3 | 83 | 10 | 11 | 104 |
| 32 | 32 | 2 | 4 | 38 |
| 3m | 218 | 26 | 29 | 273 |
| -3m | 786 | 128 | 112 | 1026 |
| 6 | 9 | 0 | 0 | 9 |
| -6 | 1 | 2 | 1 | 4 |
| 6/m | 7 | 0 | 0 | 7 |
| 622 | 15 | 2 | 0 | 17 |
| 6mm | 119 | 10 | 8 | 137 |
| -6m2 | 210 | 32 | 31 | 273 |
| 6/mmm | 529 | 55 | 74 | 658 |
| 23 | 32 | 4 | 6 | 42 |
| m-3 | 31 | 6 | 6 | 43 |
| 432 | 2 | 0 | 0 | 2 |
| -43m | 882 | 124 | 89 | 1095 |
| m-3m | 2571 | 329 | 300 | 3200 |
| **Total** | **11584** | **1448** | **1448** | **14480** |

### matten_elastic

MatTen elasticity tensors of 10276 crystals. published train/val/test split.

| PG | train | validation | test | total |
|---|---:|---:|---:|---:|
| 1 | 57 | 7 | 6 | 70 |
| -1 | 80 | 6 | 5 | 91 |
| 2 | 26 | 4 | 1 | 31 |
| m | 111 | 15 | 11 | 137 |
| 2/m | 484 | 56 | 62 | 602 |
| 222 | 25 | 5 | 3 | 33 |
| mm2 | 209 | 27 | 36 | 272 |
| mmm | 1048 | 132 | 134 | 1314 |
| 4 | 1 | 0 | 0 | 1 |
| -4 | 21 | 4 | 9 | 34 |
| 4/m | 33 | 3 | 6 | 42 |
| 422 | 15 | 3 | 1 | 19 |
| 4mm | 38 | 6 | 3 | 47 |
| -42m | 123 | 12 | 16 | 151 |
| 4/mmm | 1297 | 162 | 151 | 1610 |
| 3 | 4 | 0 | 0 | 4 |
| -3 | 55 | 9 | 8 | 72 |
| 32 | 15 | 5 | 4 | 24 |
| 3m | 102 | 11 | 10 | 123 |
| -3m | 447 | 61 | 60 | 568 |
| 6 | 2 | 0 | 0 | 2 |
| -6 | 2 | 0 | 0 | 2 |
| 6/m | 7 | 0 | 0 | 7 |
| 622 | 11 | 1 | 1 | 13 |
| 6mm | 87 | 10 | 15 | 112 |
| -6m2 | 172 | 20 | 19 | 211 |
| 6/mmm | 487 | 66 | 60 | 613 |
| 23 | 44 | 4 | 1 | 49 |
| m-3 | 63 | 14 | 5 | 82 |
| 432 | 4 | 1 | 1 | 6 |
| -43m | 368 | 36 | 25 | 429 |
| m-3m | 2792 | 345 | 368 | 3505 |
| **Total** | **8230** | **1025** | **1021** | **10276** |

### jarvis_dfpt_bec_local_snapshot

JARVIS-DFPT BEC local successful-extraction snapshot. 10 local successful records; full extraction and frozen split pending.

| PG | local_available | total |
|---|---:|---:|
| 1 | 1 | 1 |
| -1 | 1 | 1 |
| 2 | 0 | 0 |
| m | 0 | 0 |
| 2/m | 1 | 1 |
| 222 | 0 | 0 |
| mm2 | 0 | 0 |
| mmm | 0 | 0 |
| 4 | 0 | 0 |
| -4 | 0 | 0 |
| 4/m | 0 | 0 |
| 422 | 0 | 0 |
| 4mm | 0 | 0 |
| -42m | 0 | 0 |
| 4/mmm | 0 | 0 |
| 3 | 0 | 0 |
| -3 | 1 | 1 |
| 32 | 0 | 0 |
| 3m | 0 | 0 |
| -3m | 5 | 5 |
| 6 | 0 | 0 |
| -6 | 0 | 0 |
| 6/m | 0 | 0 |
| 622 | 0 | 0 |
| 6mm | 1 | 1 |
| -6m2 | 0 | 0 |
| 6/mmm | 0 | 0 |
| 23 | 0 | 0 |
| m-3 | 0 | 0 |
| 432 | 0 | 0 |
| -43m | 0 | 0 |
| m-3m | 0 | 0 |
| **Total** | **10** | **10** |

## Consequences for expert training

1. Uniform random sampling trains current-PG experts in direct proportion to the tables above; it does not produce balanced expert updates.
2. Report both natural-distribution metrics and a balanced-training ablation. For the latter, use inverse-frequency or temperature-smoothed PG sampling **within the train split only**; never rebalance validation/test.
3. A PG with very few structures cannot be repaired by oversampling alone. Prefer shared adaptation, hierarchical parameter sharing/regularization, and report per-PG metrics with confidence intervals; merge neither labels nor datasets silently.
4. Recompute actual expert activation counts after material-specific parent DAGs exist. A parent expert may receive gradients from several child PGs, so current-PG frequency is a lower-bound view of routed load, not the final load profile.
5. Finish BEC extraction and freeze its split before choosing BEC sampling weights.

## Reproduction

```text
python tools/dataset_pg_statistics.py
```

Machine-readable counts and metrics: [dataset_pg_balance.json](dataset_pg_balance.json).
