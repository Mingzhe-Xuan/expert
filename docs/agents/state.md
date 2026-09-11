# Agent state

## 并行 Goal 状态（Guqq per-backbone environments）

官方包元数据证明单一 venv 无法同时满足冻结的 MACE、GRACE、DPA4 与 Equiformer runtimes。
所有 Slurm 入口现已使用四个显式 venv contract；两个 mixed arrays 按已冻结且验证的
`index % 4` backbone schedule 选择环境，避免 resolver 冲突或运行时静默替换版本。

## 并行 Goal 变更记录（Guqq per-backbone environments）

- 2026-09-12：确认 MACE/e3nn 0.4.4、fairchem/Torch 2.4、DeepMD/Torch 2.11/e3nn 0.5.9
  与 GRACE/TensorFlow 的不可合并边界；冻结四变量 Slurm selector 与静态映射测试。
- 2026-09-12：四环境 selector、九个相关 launchers 与 fail-closed 测试完成；targeted 13、
  完整 144 tests 全绿，全部 shell syntax/compile/diff checks 通过。下一步在 Guqq 建立三套
  剩余 venv，并安装/记录四套精确环境。

## 并行 Goal 状态（material-oriented target projection）

已将 dielectric/elastic global readout 从 point-group symbol 的 representative Hall projector
改为逐材料 `SymmetryRecord.rotations` Reynolds projector。非代表 orientation、mixed batch、
幂等/不变性/梯度与完整 26-config 回归均通过；BEC raw head 继续不使用该 projector。

## 并行 Goal 变更记录（material-oriented target projection）

- 2026-09-12：冻结 actual-operation projector 的 shape/task/device、幂等、不变性、梯度与
  non-representative orientation 测试；下一步实现 heads/readout 接口并跑完整套件。
- 2026-09-12：operation-aware projector 与 readout 已完成；targeted 12/15 tests 及完整
  133-test suite 全绿，compile/diff checks 通过。下一步回到 Guqq 环境与资源准备。

## 并行 Goal 状态（32-PG equilibrium fixtures）

已完成 source iterator、deterministic selector、versioned validator、58-row real-checkpoint
smoke runner，以及三个 Goal 必需 Slurm entry points 的完整证据输出。实际 32-record asset
仍必须由 Guqq Slurm 生成并经 scp 回本地，prototype synthetic structures 不能充当最终验收证据。

## 并行 Goal 变更记录（32-PG equilibrium fixtures）

- 2026-09-12：冻结真实 equilibrium 来源、32-group exact coverage、Hall re-detection、record
  checksum 与 missing/tamper failure 测试；下一步实现并验证生成器和 Slurm entry point。
- 2026-09-12：完成 32-PG fixture selector/validator、58-row schedule、真实 adapter forward/backward、
  backbone-native periodic edge automorphism 与 target/BEC joint audit；三个验收 launcher 均保存
  JUnit、JSON、Git/environment/Slurm metadata，且全量 pytest 的 skip/xfail 会非零失败。本地 129 tests
  全绿；下一步必须在 Guqq 生成真实 fixture 并运行全部 Slurm jobs。

## 当前状态（Phase A 正式实现启动）

已建立正式 `src/` 包的 12 个一级模块边界和 README，落地六项核心 typed
contracts、三类 target layouts、checkpoint convention fail-closed 校验、五分支
配置 schema 与冻结 26-config manifest。正式 32 点群 registry 现已验证 operation
closure/identity/inverse/正交化，并可生成 `l<=4` O(3) 表示、invariant projector、
确定性 basis/checksum。三类 target 现具备高精度 Cartesian↔irrep、O(3) transform、
global PG fixed-space projection、de-canonicalization；BEC 另具独立 ASR 与显式可选
joint projector。Phase B 已增加无任意邻居截断的完整 cutoff PBC image enumeration、
mixed-size collation 和 device/dtype transfer。合并本地套件 45 passed、1 个旧原型
opt-in skip。四个独立训练单元现已有合法 pairing、group-safe seeded split 和
train-only copy-aware RMS/variance normalizer contract；训练层已增加逐 copy loss/raw
metrics、optimizer update 和 metadata-first atomic checkpoint round-trip。合并本地
套件 59 passed、1 个旧原型 opt-in skip。现已加入显式 Hall-setting spglib
canonicalization、site-order-preserving frame、完整 operation permutations，以及
affine/species/checksum/DAG parent validation。Full-PG 数学层现可逐 O(3) provenance
copy 分解全部实有限群 irreps、正交 subduction/inverse，并从 Hom-space Reynolds
projector 构造 finite-group CG paths；合并套件 75 passed、1 个旧 skip。Goal 尚未
具备完成证据。Phase C 现已实现 full-O(3) TP 与真正逐 degree/copy triple 枚举完整
bandlimited Hom paths 的 local-O(2) TP，支持 `mmax=2`/full-m、local O(2) gauge 和
global improper O(3)。shared adaptation、routed O3E、A1/Full-PG 两 block experts、
continuous Hall active-set gates 和公共 O(3) fusion 已接通；合并套件 103 passed、
1 个旧 skip。五分支 dispatcher 与 dielectric/elastic/BEC 三类 final readout 已接通，
26 configs 均已完成本地 forward/loss/backward/checkpoint 和 active `<5M` 验证；global
outputs 在 canonical frame 投影 PG fixed space，BEC 保留 node scope 并独立返回 raw/ASR。
Goal 尚未具备完成证据。

## 当前计划（Phase A 正式实现启动）

1. 实现四个真实预训练 backbone adapters，并冻结各自真实中间 O(3) layout contract。
2. 将 backbone adapter 接入统一 dispatcher，补真实 checkpoint 的最小 batch 前向/反传。
3. 完成真实数据集与 32 点群两级 smoke entrypoints，再在 Guqq 经 Slurm 验证。

## 变更记录（Phase A 正式实现启动）

- 2026-09-11：恢复被中断的续跑；工作树仍停在 clean commit `8c65da9`，无半成品
  修改或本轮残留进程。进入 deterministic spglib canonicalization 单元，先登记
  Hall setting、frame/site round-trip 与完整 operation-permutation 测试。
- 2026-09-11：canonicalization 目标测试发现 improper operations 被正交化为 proper
  的 determinant 错误，已在不放宽断言的前提下修复，6/6 通过。继续进入 Hall-level
  physical parent embedding validation，并先登记 affine closure/species/DAG 测试。
- 2026-09-11：完成 deterministic Hall canonicalization 和 physical-parent validation；
  保留原 site order、生成 audit permutations，验证 affine closure/species mapping、
  orientation variants 与 DAG connectivity。目标联合 25 passed，完整套件 69 passed。
- 2026-09-11：完成五分支 dispatcher 与三类 tensor readout；首轮测试发现并修复
  float32 fixed-space 判秩污染 forbidden components。26 configs 已逐项通过
  forward/loss/backward/checkpoint/active `<5M`，下一阶段进入四个真实 backbone adapters。
- 2026-09-11：进入 Full-PG subduction/finite-group CG 实现；方案冻结为逐 O(3)
  provenance copy 的实 commutant decomposition、正交 inverse，以及 Hom-space Reynolds
  intertwiners，避免把 invariant/A1 projector 冒充 Full-PG。
- 2026-09-11：完成 32 群 Full-PG real-irrep subduction/inverse 与 finite-group CG
  registry；保留 source/copy provenance、确定性 path order/checksum，cubic `l=2`
  明确产生 `2+3` 非 A1 blocks。目标测试 6/6，合并套件 75 passed。
- 2026-09-11：Phase A 数学 registry/canonicalization gate 的本地基础测试已绿，进入
  Phase C 双 TP backend；`o2_tp` 将逐 degree/copy triple 枚举 local O(2) intertwiners，
  不复用旧原型“full O(3) TP 后裁 m-band”的不完整实现。
- 2026-09-11：完成 `full_o3 | o2_tp` backend；O2 coupling 由非混叠 dihedral
  quadrature 的完整 Hom basis 构造，逐 multiplicity path 独立学习。目标测试 12/12，
  proper/improper、gauge、mmax、parity 与 reversal 全绿，合并套件 87 passed。
- 2026-09-11：进入 adaptation/O3E/PGE/routing/fusion 实现；双 O3 placements 复用已
  验证 TP contract，PGE 两层独立 block 分为 A1 invariant 与 Full-PG subduced modes，
  并先登记 32 群/continuous gate/active-parameter tests。
- 2026-09-11：完成 shared adaptation、routed O3E、A1/Full-PG PGE、continuous gate
  与 hierarchical fusion；32 群双 PGE mode、双 TP backend proper/improper、empty
  fallback、两 block 独立和 `<5M` 初审全绿。目标 16 passed，合并 103 passed。
- 2026-09-11：进入五分支 dispatcher/三 readouts 实现；冻结 26-config conditional
  module、mixed batches、global PG constraint、node-wise BEC raw+ASR、validated parent
  routing、loss/backward/checkpoint 和 `<5M` 全矩阵测试。
- 2026-09-11：退出 dispatcher/readout 阶段：三类输出、父 Hall 校验和 26-config
  train/checkpoint 矩阵均通过；期间修复 float32 fixed-space 判秩缺陷。下一步实现并
  验证四个真实预训练 backbone adapters。
- 2026-09-11：进入四真实 backbone adapter 阶段；先冻结资源 metadata、真实 feature
  tap、graph reuse/rebuild、frozen-gradient 和 parity 验收。DPA4/EquiformerV2 必须以
  双真实前向的 inversion-paired Reynolds wrapper 接入，Equiformer gated 资源仍需解决。
- 2026-09-11：完成 backbone resource/parity/interface 基础层：manifest 现含四项冻结
  metadata 与 MACE 文件 SHA，资源校验 fail closed；SO(3) wrapper 在保留右手 cell 的
  周期反演对上执行两次 extractor 并产生显式 even/odd blocks。真实四 adapter 尚未完成。
- 2026-09-11：进入真实 MACE adapter 单元；冻结官方 calculator、本地 checkpoint、
  first-interaction `128x0e+128x1o` tap、实际 MACE edge geometry 与 mixed-batch contract。
- 2026-09-11：MACE adapter 与 Slurm smoke 候选已完成本地 contract 验证；合并套件
  114 passed、1 旧 skip。Guqq 仓库 `git pull` 本次 60 秒无响应，真实 checkpoint
  forward/reflection/frozen-gradient 验收仍待同步后提交 Slurm，未误标完成。
- 2026-09-11：进入真实 DPA4-Plus adapter；官方 3.2.0 源码确认
  `forward_with_edges` 直接返回 `[N,25,1,64]` final equivariant latent，且 neighbor graph
  方向与本项目一致。将基于公开返回值和 graph builder 接入双前向 parity wrapper。
- 2026-09-11：DPA4-Plus adapter 与 Slurm smoke 候选完成本地验收：严格验证 checkpoint
  和伴随 config，复用官方 sparse edges，显式转换 `64x(l=0..4)` copy-major layout，并以
  双真实调用 Reynolds wrapper 补全 O(3) parity。目标 9 passed、合并 116 passed、1 旧
  skip；Guqq 本轮在 shell 建立前断开，真实 runtime/layout/equivariance 仍待 Slurm。
- 2026-09-11：进入真实 GRACE adapter 单元；先从发布 archive 的 `model.yaml` 与官方
  TensorPotential loader/instruction graph 冻结精确 restore API、`rho` tap、O(3) layout 和
  bond geometry contract。要求的 0.5.10 不在 PyPI，必须先解决可复现 runtime 来源。
- 2026-09-11：GRACE adapter 候选完成本地验收；源码审计纠正 `rho` scalar-only 误标，
  改取共同高阶分支 `AA` 的 27-copy/4512-component 自然宇称状态，并对 runtime 球谐基
  到 e3nn 的正交转换严格求解。目标 11 passed、合并 118 passed、1 旧 skip；0.6.0
  兼容 restore 和真实数值等变性仍待 Guqq Slurm，未误标完成。
- 2026-09-11：进入 EquiformerV2 adapter 单元；官方模型页再次确认 checkpoint 文件可列出
  但下载必须由用户接受 OMat24 协议并共享联系信息。先实现官方源码/API、SO(3) tap、graph
  和 parity contract；资源继续严格 gated，不代替用户接受许可。
- 2026-09-11：完成 EquiformerV2 adapter 候选与 Slurm smoke；严格使用 fairchem-core
  1.10.0 官方 loader，冻结 final-normalized `[N,25,128]` node tap，并复用返回的周期图。
  目标 13 passed、完整项目 114 passed；exact OMat24 checkpoint 仍待用户许可和 Guqq
  真实 restore/equivariance，因此该 backbone 未误标为完成，下一阶段转入统一真实数据管线。
- 2026-09-11：进入四个独立真实数据模块实现单元；已审计 GMTNet JARVIS pickle、MatTen
  column-oriented JSON 和 JARVIS-DFPT processed JSONL schema。模块边界冻结为 resource/split
  manifest 验证、统一 sample contract、逐来源 parser 和确定性 3/1/1 smoke selection；完整
  BEC 提取和真实模型训练仍只经 Guqq Slurm。
- 2026-09-11：四个独立数据 loader 候选完成；统一 sample 同时保留结构、Cartesian target、
  irrep coefficients、单位与来源元数据，严格校验资源及 split，并固定 3/1/1 selection。
  三个现有真实源的五样本解析通过，完整项目 119 passed；BEC 完整 JSONL 仍待 Guqq Slurm。
  下一实现单元转入 end-to-end smoke trainer/report 与三个总 Slurm entry points。
- 2026-09-12：进入 end-to-end execution 单元；模块边界冻结为 `models/` 中真实
  backbone→backbone graph→dispatcher 组合，以及 `training/` 中 sample canonicalization、
  train-only normalization、3/1/1 update/eval/checkpoint/report。单元测试可注入解析型 extractor，
  但 Guqq 最终证据必须运行真实四 checkpoint。
- 2026-09-12：完成 end-to-end composition、canonical 3/1/1 smoke runner、20-run pairwise
  schedule 和 real-subset Slurm array 入口；目标 4 passed、完整项目 123 passed、0 skip。
  当前仅 orchestration 通过，20 个真实 jobs 尚未运行；下一单元补 32-PG versioned fixtures/
  smoke runner 与 `test_all`、`smoke_32_point_groups` 总入口。

- 2026-09-11：读取 `GOAL.md`、原型、资源 manifest、Git 状态和既有代理记录；
  确认正式实现尚未开始。进入 Phase A，先完成包骨架、核心 contracts 和 26 配置
  schema，并在实现前登记对应测试。
- 2026-09-11：完成正式模块骨架、typed contracts、target layouts、convention
  checksum 和 26-config schema/manifest；新增测试 15/15 通过，完整本地集合为
  27 passed、1 个旧 opt-in skip。下一实现单元转入数学 registries 与 transforms。
- 2026-09-11：进入 Phase A 点群 registry/invariant-subspace 实现单元；测试计划已
  冻结为 32 群 closure、O(3) 表示同态、projector/basis 与 target fixed-space 检查。
  Guqq 同步尝试仅到达跳板欢迎信息后悬挂，已终止相关 SSH 进程，未运行远端计算。
- 2026-09-11：完成 32 点群 registry、fractional→Cartesian 正交化、O(3) 表示、
  invariant projector 和确定性 basis/checksum；目标测试 4/4、合并套件 31 passed。
  PyTorch/e3nn 兼容处理仅登记 `slice` safe global，未禁用安全 checkpoint 加载。
- 2026-09-11：进入 target transforms/BEC controls 实现单元；先冻结 round-trip、
  O(3) covariance、32 群 fixed-space、逐 crystal ASR、联合 projector 与交换性测试。
- 2026-09-11：完成三 target 高精度 transforms、global fixed-space projector、
  frame/site-order 恢复、BEC ASR 与 optional joint projector；目标测试 8/8，合并
  套件 39 passed。通过 basis 极分解修复 e3nn default-float 的约 `1e-7` 精度上限。
- 2026-09-11：进入 Phase B cutoff PBC graph 实现单元；测试计划冻结为完整 image
  shell、skew cell 自适应 bounds、mixed-size collation、empty fallback、device/dtype
  transfer 与 proper/improper covariance。
- 2026-09-11：完成自适应 reciprocal-bound PBC graph 构造、完整 directed image
  multiedges、strict cutoff、mixed-size collation 与 `.to()`；skew brute-force 和
  proper/improper covariance 测试通过，合并套件 45 passed。
- 2026-09-11：进入独立训练单元/split/normalization 实现；先冻结四个合法
  dataset-property 配对、按 duplicate group 的确定性 8:1:1 fallback、train-only
  repeated-copy normalizer 与 fail-closed state tests。
- 2026-09-11：完成四独立 training-unit contract、seed 20260911 group-preserving
  split 和 train-only coefficient normalizer；5-group 产生 3/1/1，RMS/variance
  均按显式 irrep copy 保存并支持 inverse，合并套件 51 passed。
- 2026-09-11：进入 training loss/metrics/checkpoint 实现单元；测试冻结为逐 copy
  coefficient loss、BEC node scope、raw physical metrics、optimizer update、严格且原子
  checkpoint round-trip，以及 architecture/unit/layout/convention 不兼容拒绝。
- 2026-09-11：完成三 target copy-aware coefficient loss、raw MAE/RMSE 与原子
  checkpoint；加载在 state mutation 前验证 architecture、unit、convention 和
  normalizer layout，目标测试 8/8、合并套件 59 passed。

## 当前状态（GOAL 与 subgroup-chain 资产整理）

已将 `proposal.md` 与 32 点群子群资料收敛为可执行的 `GOAL.md`，并生成供实现消费的 `assets/docs/subgroup_chain.json`；JSON 完整性测试和当前原型测试通过。当前代码仍仅为 MACE + A1 dielectric/elastic 原型，BEC、Full-PG、其余三类 backbone、五分支 dispatcher 与完整 Guqq 测试矩阵属于后续 Goal 执行内容，尚未误标为完成。

## 当前计划（GOAL 与 subgroup-chain 资产整理）

1. 后续按 GOAL Phase A--E 实现 registries、四 backbone、五分支、三类 heads、数据与训练系统。
2. 完成 BEC 全量可访问 archive 提取并解除 EquiformerV2 gated checkpoint 阻塞。
3. 在 Guqq 上通过 Slurm 执行全部 unit/equivariance/32-PG/real-subset suites，满足 0 mandatory skip/xfail 后才标记 Goal complete。

## 变更记录（GOAL 与 subgroup-chain 资产整理）

- 2026-09-11：开始从 proposal 与点群子群资料生成实现 GOAL 和 runtime subgroup-chain asset；先审计现有原型、数据资源与测试覆盖，避免将待实现项误标为已完成。
- 2026-09-11：完成 `GOAL.md`，冻结五分支/26 配置、四真实 backbone、PG/O(3) 双 mode/backend、独立数据任务、模块接口、参数预算、完整测试矩阵与 Guqq/Slurm Definition of Done。
- 2026-09-11：生成 `assets/docs/subgroup_chain.json`（32 群、80 cover edges、433 oriented instances、222 maximal class-chain sequences），增加可复现生成入口与 closure/chain test，并将原型默认 DAG 路径切换到该资产。

## 当前状态（五分支架构收敛）

已将 proposal 的顶层 architecture matrix 收敛为用户指定的五个分支，并统一 PG hidden mode 与 O(3) TP backend 两个正交配置轴；文档一致性检查通过。本轮未修改 `assets/model_code`。

## 当前计划（五分支架构收敛）

1. 后续以五分支 dispatcher 为唯一实现入口。
2. 分支 4/5 实现 `a1_only | full_pg`，实际存在的 adaptation/O3E/readout 分别实现 `full_o3 | o2_tp`。
3. 完成实现后按 architecture、点群、task 与 backend 重新统计参数量并运行 smoke/equivariance tests。

## 变更记录（五分支架构收敛）

- 2026-09-11：冻结训练/数据/smoke 协议：每个 dataset × property 独立训练；优先复用 benchmark split，否则固定 8:1:1；优先复用 backbone cutoff graph，否则 6 Å；32 点群各一结构做 forward/backward，且每个真实训练单元用 5 个结构跑通 train/test。
- 2026-09-11：将额外 parity carrier 改为逐 target decomposition 决定；dielectric/elastic 不添加 unnatural parity，BEC 在两种 PG mode 下均只额外加入 `1x1e`。
- 2026-09-11：澄清 BEC 约束边界：默认 forward 不输入 \(\pi_g\) 且不依赖联合 projector；\(\pi_g\) 仅供 graph/equivariance audit 与非默认 hard-projection control，ASR 独立且不需要 \(\pi_g\)。
- 2026-09-11：将顶层实验架构冻结为 `B+R`、`B+A+R`、`B+A+O3E+R`、`B+PGE+R` 与 `B+A+PGE+R`；同步摘要、方法图、伪代码、配置、实验矩阵、阶段计划与总结。
- 2026-09-11：明确 O3E/PGE 使用相同 parent DAG、连续 gates 与 hierarchical fusion，二者不在同一正式分支中叠加；PG mode 为 `a1_only | full_pg`，所有实际存在的 O(3) TP backend 为 `full_o3 | o2_tp`。
- 2026-09-11：移除 identity readout 作为正式架构选项，并完成 Markdown 围栏、公式围栏、路径与 diff whitespace 检查。

## 当前状态

已完成 proposal 的 BEC 正式 benchmark、task-aware node head/input irreps、联合等变性说明，以及 A1-only / Full-PG 双实现范围修订；文档一致性检查通过。

## 当前计划

1. 后续实现 `full_pg` path table、BEC node head 与协变 cutoff PBC graph constructor。
2. 实现完成后重新统计两种 mode 含 BEC head/`1e` carrier 的逐点群 active parameters。
3. 对 32 点群运行 forward/backward、graph automorphism 与 BEC 联合等变性 smoke tests。

## 变更记录（BEC benchmark 与双 PG hidden mode）

- 2026-09-11：将 BEC 从 future work 提升为第一阶段 benchmark；明确 `output_scope=node`、`0e+1e+2e` head、task-aware `1e` input carrier、联合 node-permutation--tensor-rotation 等变性、可选 Reynolds projector 与 ASR。
- 2026-09-11：将 `a1_only` 与 `full_pg` 固定为两种必须实现和评测的 PG hidden mode，并加入参数匹配 layout、32 点群 smoke/equivariance 验收与 BEC 数据协议。
- 2026-09-11：完成 Markdown 围栏、diff whitespace、本地路径与旧范围声明检查；本轮仅修订文档，`assets/model_code` 尚未实现新增 Full-PG/BEC forward。

## 当前状态

已将本机 SSH 别名 `Guqq` 配置为经 `vlab` 跳转，并完成端到端免密验收。
`vlab true` 与 `Guqq true` 均以状态 0 退出；Guqq 的首个远程操作确为
`git pull`，但默认登录目录不是 Git 仓库，因此该 Git 命令本身以状态 1 退出。

## 当前计划

1. 后续直接使用 `ssh Guqq` 经 vlab 登录。
2. 若需在 Guqq 拉取代码，先切换到服务器上的项目仓库目录再运行 `git pull`。
3. 保留 `C:\Users\asus\.ssh\config.bak-vlab-20260911` 作为回滚备份。

## 变更记录

- 2026-09-11：开始其他 backbone 与 BEC 数据准备阶段；先进行官方资源审计。
- 2026-09-11：锁定并下载 GRACE-1L-OMAT-medium-ft-E；内部 YAML 确认中间
  feature `lmax=4`，权重 SHA-256 与 Hugging Face LFS 值一致。
- 2026-09-11：DPA4 发布配置审计显示 Nano/Mini/Neo/Air 的 `lmax` 分别为
  1/2/3/3，改选唯一满足条件的 Plus（`lmax=4, mmax=1`）并下载。
- 2026-09-11：确认 JARVIS 冻结 raw index 只有 5,000 个 DFPT 档案、约
  9.95 GiB，比论文的 5,015 少 15 个；完成 calculation-matched BEC 提取器。
- 2026-09-11：本地 10 样本提取验证通过；服务器 SSH 首次连接被远端关闭，
  完整提取暂未提交，下一步重试连接而不在本地执行重批处理。
- 2026-09-11：用户明确授权后已将 `e65c0e0` 推送至 `origin/main`。SSH 连续
  三次在 key exchange 前由远端关闭；端口 22 可达且本地 alias/用户/密钥配置
  正确，因此停止盲目重试并将完整 Slurm 提取标为外部阻塞。
- 2026-09-11：按用户要求重新进入 Guqq SSH 连接诊断阶段；已先记录服务器连接
  用途，下一步以 `git pull` 为首个远程命令进行限时 verbose SSH 测试，并按
  TCP、密钥交换、认证和远程命令四层定位故障。
- 2026-09-11：Guqq 复测完成并再次阻塞：TCP 22 建连成功，服务端在发送 SSH
  banner 和进入 key exchange 前关闭；无认证尝试、无远程命令执行。下一步需由
  服务器管理员检查 sshd、`MaxStartups`、源 IP 过滤及 fail2ban/CrowdSec 等
  pre-auth 策略，或待该状态解除后再连接并首先执行 `git pull`。
- 2026-09-11：开始配置 `vlab` 作为 `Guqq` 跳板机；确认现有 `vlab` 专用密钥和
  `Guqq-server` 跳转条目可复用，下一步备份用户级配置、修改 `Guqq` 并执行
  静态解析与端到端免密验收。
- 2026-09-11：完成 `vlab` 跳板配置和免密验收；`Guqq` 已增加
  `ProxyJump vlab`，静态解析正确，vlab 与 Guqq 的 BatchMode 连接均返回 0。
  Guqq 首次远程 `git pull` 已执行但因登录目录不是仓库返回 1；SSH 配置任务完成。
