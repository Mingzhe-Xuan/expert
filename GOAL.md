# Goal: Complete point-group-specialized tensor model implementation

## 1. Objective

在 `src` 中完成 `proposal.md` 定义的晶体张量预测系统，使四类真实 pretrained backbone、五个 architecture branches、两种 PG hidden modes、两种 O(3) tensor-product backends，以及 dielectric、elastic、BEC 三类 target 均具有完整且可训练的 forward/backward 实现。

最终实现必须在 Guqq 上通过：

1. 全部单元测试；
2. O(3)、PG、PBC graph 与 BEC 联合等变性测试；
3. 32 个 crystallographic point groups 各一个结构的 forward/backward smoke test；
4. 每个真实 `dataset × target property` 独立训练单元的 5-structure train/validation/test smoke test；
5. 所有合法 architecture/mode/backend 分支的构造、forward、loss、backward 与 checkpoint round-trip 测试。

本 Goal 不要求全量训练、收敛性实验、SOTA 结果或完整超参数搜索，但不允许以 mock backbone、dummy output、跳过测试或只实现配置入口来宣称完成。

## 2. Sources of truth

- 数学、模型、数据和实验定义：[proposal.md](proposal.md)
- 完整 32 点群子群资料：[docs/ref/crystallographic_point_group_subgroups.md](docs/ref/crystallographic_point_group_subgroups.md)
- 可机读点群格：[assets/docs/subgroup_chain.json](assets/docs/subgroup_chain.json)
- 当前实现原型：[assets/model_code](assets/model_code)
- 开发与 Guqq 权限规范：[AGENTS.md](AGENTS.md)

发生冲突时，优先级为：本 Goal 的冻结决策 > `proposal.md` 中与本 Goal 一致的最新定义 > 当前原型代码。不得为了迁就现有原型而缩减验收范围。

## 3. Starting state

当前代码只能视为 reference prototype：

- 已有 MACE 接口和部分前向；
- 已有 A1-oriented dielectric/elastic 原型、参数报告与部分等变测试；
- 尚未完整实现 GRACE、DPA4、EquiformerV2；
- 尚未完整实现 `full_pg`、BEC、五分支 dispatcher 和所有 `full_o3 | o2_tp` placement；
- JARVIS dielectric、JARVIS elastic、MatTen elastic 已在本地准备；
- JARVIS-DFPT BEC 只有索引和 10 个验证 archive，完整约 5,000 archive 的下载、提取与审计尚未完成；
- EquiformerV2 checkpoint 的访问限制必须真实解决，不能用随机权重替代最终验收。

## 4. Frozen model matrix

### 4.1 Five architecture branches

记 `B` 为 O(3) feature-interface backbone，`A` 为 shared O(3) adaptation，`O3E` 为 symmetry-routed O(3) experts，`PGE` 为 point-group experts，`R` 为 O(3)-equivariant task readout。只实现以下五个顶层分支：

| ID | Name | Forward path |
|---:|---|---|
| 1 | `B+R` | backbone → O(3) readout |
| 2 | `B+A+R` | backbone → O(3) adaptation → O(3) readout |
| 3 | `B+A+O3E+R` | backbone → O(3) adaptation → routed O(3) experts → hierarchical fusion → O(3) readout |
| 4 | `B+PGE+R` | backbone → point-group experts → inverse subduction/fusion → O(3) readout |
| 5 | `B+A+PGE+R` | backbone → O(3) adaptation → point-group experts → inverse subduction/fusion → O(3) readout |

`O3E` 与 `PGE` 使用相同的 current-plus-compatible-parent active set、continuous residual gates 和 hierarchical fusion，但二者互斥；第一版不增加同时含 O3E 与 PGE 的第六分支。

### 4.2 PG hidden modes

Architectures 4/5 必须同时支持：

- `a1_only`：natural multiplicity `[8, 2, 2, 2, 2]`；保留 provenance-labelled trivial irreps；
- `full_pg`：natural multiplicity `[4, 1, 1, 1, 1]`；保留全部允许的 finite-group irreps、subduction copies 与 fusion copies。

每个 active PG expert 串联两个参数独立、不共享权重的 PG blocks。`C1` 可按冻结策略 bypass 无群论收益的 PG-TP，但必须保持相同输入输出 contract。

### 4.3 O(3) TP backends

每个实际存在的 O(3) TP placement 都必须独立支持：

- `full_o3`；
- `o2_tp`，默认 `mmax=2`，并支持 full-`m` control。

placements 为：

- shared adaptation；
- routed O(3) expert；
- final O(3) readout。

不存在对应模块时配置值必须为 `none`，不得实例化空模块。五个结构分支展开为 26 个合法 mode/backend configurations：`2 + 4 + 8 + 4 + 8`。

### 4.4 Backbone adapters

必须接入真实 checkpoint：

| Adapter | Frozen checkpoint for acceptance | Current resource state |
|---|---|---|
| MACE | MACE-MP `medium-0b3` | prototype adapter exists; version/checksum must be frozen |
| GRACE | `GRACE-1L-OMAT-medium-ft-E-checkpoint.tar.gz` | downloaded and checksum-verified |
| DPA4 | `DPA4-Plus-OMat24-v20260805.pt` | downloaded; selected because `lmax=4` |
| EquiformerV2 | `eqV2_31M_mp.pt` from `facebook/OMAT24` | gated access currently unresolved |

具体 revision、license、runtime、local path 与 checksum 以 `data/manifests/backbones.json` 为资源 registry；MACE 缺失的 manifest entry 必须在验收前补齐。

所有 adapter 输出统一的 `O3FeatureBatch`，至少包含 node features、明确的 `(l, parity, multiplicity, component order)`、batch/node mapping，以及下游图所需的 edge geometry。必须在原模型 scalar readout 前提取 features。

MACE/GRACE 逐 checkpoint 审计 parity convention。只保证 SO(3) 的 DPA4/EquiformerV2 必须使用固定、可测试且在 train/validation/test 一致启用的 inversion-paired Reynolds/parity wrapper，转换为统一 O(3) interface。不得只修改 metadata 宣称 SO(3) features 已成为 O(3) features。

所有 backbone 默认 frozen；partial/full finetuning 只保留为后续实验开关，不是本 Goal 的 smoke-test 前提。

## 5. Target contracts

每个 target property 先解析 O(3) decomposition，只加入相对 natural parity `p=(-1)^l` 缺少的 carriers。

| Property | Scope | Output irreps | Extra non-natural carrier |
|---|---|---|---|
| dielectric | global | `1x0e + 1x2e` | none |
| elastic | global | `2x0e + 2x2e + 1x4e` | none |
| BEC | node | `1x0e + 1x1e + 1x2e` | `1x1e` |

dielectric/elastic 使用 permutation-invariant global pooling 和 point-group fixed-subspace constraint。BEC 不做 global pooling，输出 shape 为 `[num_atoms, 3, 3]`，保留 atom/site order 和 O(3) node carriers。

BEC 默认 forward 不接收 `pi_g`，也不依赖联合 Reynolds projector；所需关系

```text
Z[pi_g(i)] = R_g @ Z[i] @ R_g.T
```

必须由 cutoff PBC graph covariance、GNN node-permutation equivariance 与 feature-fiber equivariance 内蕴保证。`pi_g` 只用于 graph/equivariance audit 和非默认 hard-projection control。ASR projection 独立实现，不依赖 `pi_g`，并分别报告 raw、symmetry-control 和 ASR 后的结果。

### 5.1 Loss and normalization contract

supervised loss 在不可约系数空间计算，并对 repeated copies 分别处理。normalization statistics 只能由当前独立训练单元的 train split 得到并随 checkpoint 保存：

- dielectric：分别计算 `0e` 与 `2e` coefficient loss；
- elastic：分别计算两个 `0e` copies、两个 `2e` copies 与一个 `4e` copy 的 loss；
- BEC：逐原子计算 `0e`、`1e`、`2e` coefficient loss，不跨原子 pooling。

实现必须支持 train-RMS/variance weighting、raw physical-unit inverse transform 和每-irrep metrics。BEC raw head、ASR-projected output 与 optional symmetry-control output 必须同时可取；smoke training 默认对 raw head 使用有限的 differentiable coefficient MSE，ASR/projector 不得掩盖 raw equivariance failure。完整训练采用哪一种 ASR loss/projection组合留给后续实验配置，不影响本 Goal 的执行链验收。

## 6. Subgroup-chain and physical parent DAG

`assets/docs/subgroup_chain.json` 必须：

- 使用固定 schema version；
- 覆盖且仅覆盖 32 个 crystallographic point groups；
- 保留 representative operation matrices；
- 保留所有 concrete oriented subgroup instances 与 operation indices；
- 保留 maximal class cover edges、embedding multiplicity 和全部 maximal class-chain sequences；
- 记录生成器与 spglib version；
- 通过自动 schema、count、closure、cover-edge 与 chain validation。

该 JSON 只是 point-group candidate lattice，不能直接作为某个材料的物理 parent DAG。运行时 `ParentEmbeddingSpec`/`ParentDAGSpec` 还必须冻结：

- parent/child Hall number 与 settings；
- basis/origin 和 common-cell/supercell transforms；
- 完整 `(W_g, t_g)` operations；
- species-preserving atom correspondence；
- Wyckoff splitting；
- domain/orientation variant；
- convention ID、version 与 checksum。

spglib 负责当前结构 symmetry detection、standardization 和 operation validation，不负责凭 `symprec` 自动发现物理 parent phase。没有通过 Hall-level embedding validation 的 candidate 不得进入 hierarchical runtime active set。

## 7. Data and graph protocol

### 7.1 Independent training units

每个 `dataset × target property` 独立训练，分别持有 config、normalizer、model parameters、optimizer、checkpoint 和 metrics：

1. JARVIS tensor — dielectric；
2. JARVIS tensor — elastic；
3. MatTen — elastic；
4. JARVIS-DFPT — BEC。

不得进行跨性质或跨数据集联合训练；同一 property 换数据集也必须重新训练 downstream model。

### 7.2 Splits

优先原样复用 published benchmark split。没有可复用 split 时，用冻结随机种子 `20260911` 按 `8:1:1` 生成并保存 manifest。划分必须在训练/调参前完成，并避免相同材料或重复结构跨 split 泄漏。

### 7.3 PBC graph

优先复用 backbone 已构造的 cutoff PBC graph；若 API 不暴露图，则按 checkpoint 的 graph specification 确定性重建。只有二者均不可得时才使用 `6.0 Å` cutoff。

图数据至少包含：

- node order；
- directed `edge_index`；
- integer cell shifts；
- periodic displacement vectors/distances；
- batch mapping；
- cutoff/boundary convention。

不得用任意 `max_neighbors` 切断完整等距 symmetry shell。对 nonsymmorphic operations、跨边界边和相同原子对的不同 image edges 必须保持闭合。

### 7.4 BEC preparation

完成全部可访问 JARVIS-DFPT archives 的可恢复下载、checksum 验证和 calculation-matched 提取。每条记录必须保存同一 `vasprun.xml` 中的 final structure、site order、完整 `N x 3 x 3` BEC、单位、archive/checksum 和 ASR residual。生成有效/失败/排除计数及原因，不得把索引数当作有效样本数。

下载为登录节点允许的轻量操作；完整提取、校验和批处理必须通过 Guqq Slurm 执行。

## 8. Required package structure and interfaces

文件结构如下；每个一级模块目录必须有 README，说明输入、输出、conventions 和最小用例：

```text
src/
  backbones/       # four real checkpoint adapters and O(3) wrapper
  graphs/          # cutoff PBC graph contract and covariance audit
  symmetry/        # canonicalization, PG registry, parent embeddings, gates
  irreps/          # O(3)/PG layouts, subduction, CG, Cartesian transforms
  tensor_products/ # full_o3 and complete local o2_tp
  experts/         # adaptation, routed O3E, A1-only PGE, Full-PG PGE
  heads/           # dielectric, elastic, BEC, optional projectors/ASR
  data/            # four independent data modules and split manifests
  training/        # loss, optimizer, checkpoint and smoke trainer
  evaluation/      # metrics, symmetry and efficiency reports
  configs/         # validated model/data/run schemas
  cli/             # prepare/train/test/report entry points
```

核心 typed contracts：

- `PeriodicGraph`：结构、周期 multiedges、cell shifts、batch/site mapping；
- `O3FeatureBatch`：node/edge carriers 和 immutable O(3) irrep layout；
- `SymmetryRecord`：canonical frame、current PG/SG、operations 和 audit-only permutations；
- `ParentEmbeddingSpec`/`ParentDAGSpec`：物理 parent embeddings；
- `ArchitectureConfig`：只允许五个 branch names 和条件合法的 mode/backend；
- `TensorPrediction`：canonical/raw Cartesian output、scope、task metadata。

所有 subduction、CG、path ordering、copy ordering、real-harmonic convention 和 checksums 必须进入 checkpoint metadata；不兼容 checkpoint 必须 fail closed。

## 9. Forward implementation requirements

完整 forward 顺序为：

1. 读取/构造 `PeriodicGraph`；
2. canonicalize 并生成 `SymmetryRecord`；
3. 真实 backbone feature extraction 和必要的 SO(3)→O(3) wrapper；
4. 根据 architecture 有条件地执行 O(3) adaptation；
5. expert-free 直接进入 readout，或构造 validated current/parent active set 与连续 gates；
6. 执行 routed O3E，或执行 subduction → 两个独立 PG blocks → inverse subduction；
7. 仅在公共 O(3) target space 中做 hierarchical fusion；
8. 执行恰好一个 O(3) TP readout；
9. 转换到 Cartesian tensor 并 de-canonicalize；BEC 同时恢复 site order；
10. 可选执行 diagnostic/hard BEC projector 和 ASR，但 headline raw output 不依赖它们。

任何配置均须支持 mixed-size batched crystals、empty/degenerate edge fallback、CPU/GPU device transfer、float32；equivariance reference tests可使用 float64。

## 10. Parameter and runtime constraints

- 每个样本去重后的 active non-backbone parameters 必须严格 `< 5,000,000`；
- 分别报告 total、trainable、active/sample、active expert count、FLOPs、latency 和 peak GPU memory；
- 按 architecture、task、PG、PG mode 与 TP backend 生成机器可读和 Markdown 报告；
- 旧的 A1 dielectric/elastic 参数报告只能作为参考，必须在 Full-PG、BEC 和五分支实现后重新生成；
- `full_o3`/`o2_tp` 除原生宽度外必须支持 parameter-或 FLOPs-matched comparison。

超过 5M 的任何合法 mandatory configuration 都是失败，必须通过减小 multiplicity/path width 等不破坏等变性的方式修正，不得修改统计口径。

## 11. Test requirements

### 11.1 Unit tests

至少覆盖：

- config 合法/非法组合和 26-config enumeration；
- subgroup-chain schema、counts、group closure、cover edges 与 maximal chains；
- spglib canonicalization、frame round-trip 和 deterministic setting；
- PBC neighbor/image enumeration、shell completeness 和 batch collation；
- O(3) irrep layout、parity、subduction/inverse-subduction round-trip；
- finite-group representation matrices、CG intertwining、fusion multiplicity/path ordering；
- `full_o3` TP 与 `o2_tp` 的 shape、gradient、gauge、edge reversal、`0e/0o`、polar/pseudo cases；
- A1 path presence 和 Full-PG non-trivial paths；
- continuous residuals、gates、normalization、multiple-path de-duplication 和 parent-limit behavior；
- dielectric/elastic Cartesian↔irrep round-trip与 forbidden components；
- BEC `0e+1e+2e` round-trip、node scope、ASR 和 optional projector；
- 四 backbone adapters 的 checkpoint metadata、feature tap、irrep contract 和 frozen-gradient policy；
- independent dataset/property configs、split determinism、normalization leakage prevention；
- loss、optimizer update、checkpoint save/load/resume 和 report serialization。

### 11.2 Equivariance tests

对随机 proper/improper O(3) transforms 验证：

- backbone wrapper features；
- adaptation 的两种 TP backend；
- O3E 的两种 TP backend；
- final readout 的两种 TP backend；
- A1-only 和 Full-PG experts；
- 三类 final Cartesian outputs；
- forward 后 loss/backward 的 finite gradients。

对 32 点群验证每个 PG block 满足相应 finite-group equivariance。对 BEC 额外验证：

- graph node/edge/cell-shift automorphism；
- `Z[pi_g(i)] = R_g Z[i] R_g.T`；
- `pi_g` 不属于默认 model input；
- symmetry projector 关闭时 raw output 仍通过 tolerance；
- ASR 前后和 projector 交换性/兼容性。

误差 tolerance 必须按 dtype 和模块层级预先登记；不得在测试失败后针对单个 case 放宽。

### 11.3 Complete architecture coverage

生成冻结的 `architecture_variant_manifest.json`，枚举全部 26 个合法 configurations。测试必须证明每个 configuration：

- 可构造；
- 只实例化所需模块；
- forward 输出 scope/shape 正确；
- loss 可计算；
- backward 梯度有限且到达所有应训练模块；
- active parameter count `<5M`；
- checkpoint round-trip 后输出在 tolerance 内一致。

mandatory matrix 中不得存在 `skip`、`xfail` 或捕获异常后伪装通过。仅 GPU/外部 checkpoint 测试可以在本地标记为 Guqq-only，但 Guqq 最终报告中不得跳过。

## 12. Smoke-test acceptance

### 12.1 32-point-group smoke

建立恰好 32 个版本化 equilibrium PBC fixtures，每个 crystallographic PG 一个结构，并记录来源/构造方式、expected PG、Hall setting、species、lattice、fractional coordinates 和 checksum。

每个 fixture 至少完成一次：graph → symmetry → model forward → target-shaped scalar loss → backward。完整参数化 suite 必须覆盖五个 architectures、所有 O(3) TP placements 的 `full_o3 | o2_tp`、PG `a1_only | full_pg` 和三类 heads；每个点群至少有一次 PG-expert forward/backward。输出还须通过该点群 dielectric/elastic forbidden-component 检查；选定 BEC fixtures 通过联合 node-permutation/tensor-rotation 检查。

### 12.2 Real-data 5-structure smoke

对四个独立训练单元各固定 5 个真实结构：优先从正式 train/validation/test split 取 `3/1/1`。每个训练单元必须跑通：

1. raw record parsing；
2. graph/canonicalization/collation；
3. 至少一次 optimizer update；
4. validation inference；
5. checkpoint save/load；
6. test inference 与 metrics serialization。

五个 architecture branches 必须在每个训练单元上至少各完成一次端到端 smoke；四 backbone adapters 必须分别在每种 target contract 上至少完成一次真实 checkpoint 端到端 smoke。26-config exhaustive 行为由 synthetic/module matrix 保证，真实数据 smoke 可使用冻结的 pairwise coverage schedule，但不得遗漏任何 branch、PG mode、TP backend 或 backbone。

5 个样本只验证执行链，不报告科学精度，也不能代替正式 split。

## 13. Guqq execution and evidence

所有训练、推理、批量数据处理和完整测试必须通过 Slurm，禁止直接在登录节点运行。每次连接前按 `AGENTS.md` 更新 `docs/agents/gpu.md`；连接后先 `git pull`。

服务器环境必须使用独立 Python venv，并在 `docs/agents/env.md` 记录 Python、CUDA、PyTorch、e3nn、spglib、四 backbone 包及 checkpoint versions/checksums。

至少提供三个 Slurm entry points：

```text
slurm/test_all.sbatch
slurm/smoke_32_point_groups.sbatch
slurm/smoke_real_subsets.sbatch
```

每个 job 必须：

- 使用非交互命令和固定 config/seed；
- 失败时返回非零 exit code；
- 保存 stdout/stderr、JUnit XML、machine-readable summary、git commit、environment fingerprint 和 Slurm job ID；
- 记录每项 passed/failed/skipped 数；
- 通过 `sacct` 核验最终 state 和 exit code。

结果通过 `scp` 拉回本地任务约定的 ignored results 目录；摘要写入 `docs/agents/test.md`。日志、dataset、checkpoint 和批量结果不得普通 Git 提交。

## 14. Implementation phases and gates

### Phase A — Schemas and mathematical registries

交付 config schemas、subgroup/embedding registries、O(3)/PG conventions、tensor decompositions和验证 tests。退出条件：registry/round-trip/group tests 全绿。

### Phase B — Graphs and four backbone adapters

交付统一 `PeriodicGraph`/`O3FeatureBatch`、四个真实 adapter 和 SO(3)→O(3) wrapper。退出条件：四 checkpoint 的 feature/rotation/reflection tests 全绿。

### Phase C — TP, adaptation and experts

交付 `full_o3`/`o2_tp`、A、O3E、A1-PGE、Full-PGE、continuous routing/fusion。退出条件：26 configurations 全部构造、forward/backward、equivariance 和 `<5M`。

### Phase D — Three heads and data modules

交付 dielectric、elastic、BEC heads/projectors/losses，以及四个独立数据模块和完整 BEC 提取。退出条件：target round-trip、split/leakage 和 5-structure local fixture tests 全绿。

### Phase E — Guqq verification

提交三个 Slurm suites。退出条件：全部 jobs `COMPLETED`、exit code 0、mandatory tests 无 skip/xfail，并将报告拉回本地。

任何 phase 测试失败不得进入完成状态。连续三次同类失败时按 `AGENTS.md` 查阅/更新 `docs/agents/lessons.md`。

## 15. Definition of done

只有同时满足以下条件，Goal 才可标记 complete：

- 四个真实 pretrained backbone adapters 全部可用；
- 五个 architecture branches 和全部 26 legal configurations 实现完整；
- `a1_only | full_pg`、`full_o3 | o2_tp` 没有占位路径；
- dielectric、elastic、BEC 分别训练，target carriers/head/pooling 正确；
- BEC 完整可访问数据已下载、提取、审计并生成 split；
- 32 点群 fixture 齐全且 forward/backward smoke 全绿；
- 四个真实训练单元的 5-structure train/test smoke 全绿；
- O(3)、PG、PBC graph、BEC 联合等变性和全部单元测试全绿；
- Guqq Slurm 证据显示 mandatory tests 0 failed、0 skipped、0 xfailed；
- 所有合法配置 active non-backbone parameters 严格小于 5M；
- README、config、environment、data manifest、parameter report 和 test evidence 完整；
- 工作树中的源码改动已在本地通过测试后提交并推送，Guqq 从同一 commit `git pull` 执行。

以下不属于完成条件：全量训练、最终精度、SOTA 对比、超参数搜索、low-data curves 和论文作图。这些工作只能在本 Goal 完成后开始。
