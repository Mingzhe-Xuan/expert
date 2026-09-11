# Agent progress updates

- 2026-09-11：开始 MACE/GRACE/DPA4/EquiformerV2 正式 adapters；测试计划冻结为
  manifest/checksum/runtime fail-closed、scalar readout 前真实 tap、统一 `O3FeatureBatch`、
  graph reuse、checkpoint frozen policy，以及 SO(3) 模型的双前向 parity Reynolds wrapper。
- 2026-09-11：完成 backbone 基础提交候选：补齐 MACE-MP medium-0b3 revision/size/SHA
  和四资源 feature/parity/graph metadata；实现严格 resource registry、周期反演双调用
  Reynolds wrapper 与 O(3) interface projector。目标 5 passed，合并 112 passed、1 旧 skip；
  四个真实 checkpoint feature taps 仍是下一实现单元。
- 2026-09-11：开始真实 MACE-MP medium-0b3 adapter；验收边界为严格 resource/runtime、
  first-interaction parity layout、MACE graph reuse、mixed-batch node mapping、冻结 backbone
  与仅 interface projector 可训练。真实 checkpoint 数值测试将通过 Guqq Slurm 执行。
- 2026-09-11：完成 MACE adapter 候选与 `mace_adapter_smoke.sbatch`；本地目标 7 passed、
  合并 114 passed、1 旧 skip。Guqq 首条仓库 `git pull` 等待 60 秒后中止，因此真实
  MACE Slurm 尚未运行；下一步先推送候选，再做有界 pull/环境检查/作业提交。

- 2026-09-11：恢复续跑并确认工作树 clean；开始 deterministic spglib
  canonicalization，实现范围包括显式 Hall setting、input/canonical frame、完整
  `(W,t)` operations 与 species-preserving audit permutations。
- 2026-09-11：canonicalization 已通过 6 项目标测试；测试曾捕获 improper 群操作
  被误强制为 det=+1 的错误，修复后 `m-3m` 保留完整 48 操作。随后开始 Hall-level
  physical parent embedding 的 affine group、species mapping、checksum 与 DAG 验证。
- 2026-09-11：完成 explicit-Hall spglib canonicalization 与严格 ParentEmbedding/DAG
  validation：frame/site/tensor round-trip、proper-input rotation covariance、完整
  `(W,t)` species-preserving permutations、affine group、Wyckoff split mapping、domain
  variants/checksum/cycle/connectivity 均有测试；完整本地结果 69 passed、1 个旧 skip。
- 2026-09-11：开始 Full-PG 数学 registry：逐 labelled O(3) copy 做实有限群
  commutant decomposition，生成可逆 subduction/copy provenance；finite-group CG
  使用 Hom-space Reynolds projector 并逐 operation 验证 intertwining。
- 2026-09-11：完成 Full-PG subduction/CG registry：symmetric commutant 确定性拆分
  全部实 irreducible copies，正交 inverse 保证 feature round-trip，Hom-space Reynolds
  生成允许/禁止 CG paths；32 群逐操作 tests、provenance、非平凡 paths 和 checksums
  全绿。完整本地结果 75 passed、1 个旧 opt-in skip。
- 2026-09-11：开始 Phase C `full_o3 | o2_tp` backends；Full-O(3) 使用 e3nn 固定
  convention，local-O(2) 将按 degree/copy triples 枚举完整 bandlimited Hom paths，
  并验证 global O(3)、local gauge、improper/parity 与 edge reversal。
- 2026-09-11：完成双 TP backend：`full_o3` 使用完整 e3nn paths；`o2_tp` 对每个
  degree/copy triple 以非混叠 O(2) quadrature 枚举完整 Hom paths 并保留独立权重，
  支持默认 `mmax=2` 与 full-m。目标测试 12 passed，完整本地 87 passed、1 个旧 skip。
- 2026-09-11：开始 shared adaptation、routed O3E、A1/Full-PG experts、continuous
  routing 与 O(3)-space fusion；测试覆盖双 backend placements、empty edges、O(3)/32-PG
  equivariance、两层权重独立、parent limits、path dedup、梯度与 `<5M`。
- 2026-09-11：完成 Phase C 核心 modules：adaptation/O3E 在完整 PBC edges 上放置
  独立选择的 TP backend；A1 与 Full-PG experts 各含两个独立 blocks，C1 可显式
  bypass；continuous residual gates 对 Hall active set 去重归一，fusion 回公共 O(3)
  space。目标测试 16 passed，完整本地 103 passed、1 个旧 skip。
- 2026-09-11：开始五分支 dispatcher 与 dielectric/elastic/BEC final readouts；将对
  冻结 26 configs 逐项验证 conditional construction、forward/loss/backward、active
  `<5M` 和 checkpoint，同时覆盖 mixed-size global/node scopes 与 PG constraints。
- 2026-09-11：完成统一五分支 dispatcher 与三类 final readout；global heads 做
  per-crystal pooling/PG fixed-space/decanonicalization，BEC 保留逐节点 raw 并独立给出
  ASR。修复 float32 fixed-space 判秩污染；4 项目标测试覆盖全部 26 configs 的
  forward/loss/backward/checkpoint 与 `<5M`，下一单元进入真实 backbone adapters。

- 2026-09-11：启动 `GOAL.md` Phase A 正式实现；审计确认仓库尚无 `src/` 包，
  本实现单元将建立规定模块边界、核心 typed contracts、五分支配置校验与冻结的
  26-config manifest。测试范围已先写入 `docs/agents/test.md`。
- 2026-09-11：完成首个 Phase A 实现单元：正式 `src/` 模块骨架和 README、
  `PeriodicGraph`/`O3FeatureBatch`/symmetry/parent DAG/`TensorPrediction` contracts、
  三类 target layouts、checkpoint convention checksum，以及与运行时枚举严格一致的
  26-config manifest；新增测试 15 passed，合并本地套件 27 passed、1 个旧 opt-in skip。
- 2026-09-11：开始实现正式 32 点群 registry、`l=0..4` O(3) 表示和 invariant
  subspace；Guqq 同步命令经 ProxyJump 仅返回 vlab 欢迎信息后悬挂，相关 SSH
  进程已终止，尚无 Guqq checkout 同步证据。
- 2026-09-11：完成正式 32 点群 registry 和 invariant-subspace 单元：构造时验证
  operation count/identity/inverse/closure，生成正交 Cartesian operations、O(3)
  表示、幂等 projector、确定性 basis 与 checksum；测试覆盖全部群和三类 targets，
  合并本地结果 31 passed、1 个既有 opt-in skip。
- 2026-09-11：开始 target Cartesian transforms 与 BEC controls 实现；测试范围已
  覆盖三类 target round-trip/O(3) covariance、32 群 fixed-space、ASR、optional
  joint projector 及两者交换性。
- 2026-09-11：完成三类 target 的高精度 Cartesian↔irrep/O(3) transforms、global
  PG fixed-space projection、de-canonicalization，以及 BEC 独立 ASR 和显式 optional
  joint projector；使用 float64 极分解修复 e3nn change-of-basis 的单精度遗留误差，
  未放宽测试标准。目标测试 8 passed，合并本地结果 39 passed、1 个既有 opt-in skip。
- 2026-09-11：开始 Phase B cutoff PBC graph constructor/collation；测试范围先冻结为
  image multiedges、完整等距 shell、skew cells、严格 boundary、mixed batches、空边
  fallback、device/dtype transfer 和 O(3) covariance。
- 2026-09-11：完成 Phase B PBC graph 基础：通过 reciprocal bounds 自适应枚举全部
  strict-cutoff image edges，保留 directed multiedges/cell shifts，支持 mixed-size
  collation、空边和统一 `.to()`；目标测试 6 passed，合并本地结果 45 passed、
  1 个既有 opt-in skip。
- 2026-09-11：开始四个独立训练单元的 config/split/normalizer 基础；测试计划已覆盖
  合法 pairing、seed 20260911、duplicate-group 防泄漏、5-group 3/1/1、train-only
  fitting、RMS/variance inverse 和 state compatibility。
- 2026-09-11：完成四个合法 `dataset × property` contract、seed 20260911 的
  duplicate-group-safe fallback split，以及只允许 train fitting 的 copy-aware
  coefficient normalizer；支持 RMS/variance、物理单位 inverse 和跨单元/layout
  fail-closed loading。目标测试 6 passed，合并本地结果 51 passed、1 个既有 skip。
- 2026-09-11：开始 copy-aware coefficient loss/metrics 与 checkpoint 实现；测试范围
  覆盖三 targets、normalizer weighting、optimizer update、输出 round-trip、原子保存
  和 architecture/unit/layout/convention 全部 fail-closed gates。
- 2026-09-11：完成逐 labelled irrep-copy coefficient MSE/raw MAE-RMSE、optimizer
  update 和 atomic checkpoint save/load；checkpoint 用安全 tensor-only load，且在
  模型/优化器 mutation 前严格验证 architecture、training unit、完整 conventions
  与 normalizer layout。目标测试 8 passed，合并本地结果 59 passed、1 个既有 skip。

- 2026-09-11：完成 `GOAL.md` 与 `assets/docs/subgroup_chain.json`：Goal 将完整模型验收冻结为四真实 backbone、五分支/26 合法配置、A1/Full-PG、full-O3/O2、三类性质、四独立训练单元、严格 `<5M`、全部等变/单元测试及 Guqq Slurm 两级 smoke；subgroup asset 保留 32 群、80 cover edges、433 oriented instances 和 222 maximal chains，并由新增测试验证。

- 2026-09-11：冻结 dataset × property 独立训练、官方 split/8:1:1 fallback、backbone graph/6 Å fallback 规则；定义 32 点群单结构 forward/backward 与每训练单元 5 结构 train/test 两级 smoke，并将 unnatural-parity carrier 改为逐 target 审计（仅 BEC 加 `1x1e`）。

- 2026-09-11：在 proposal 中明确 BEC 默认 forward 不输入 \(\pi_g\)、不依赖联合空间群 projector；\(\pi_g\) 仅用于图/等变性审计和非默认 hard-projection control，ASR 作为无需 \(\pi_g\) 的独立物理约束。

- 2026-09-11：修订 `proposal.md` 的正式任务范围：保留 `a1_only | full_pg` 两种 PG hidden 实现；将 JARVIS-DFPT BEC 纳入第一阶段 benchmark；增加 node-wise pooling/head 分流、BEC `1e` input carrier、PBC graph automorphism、联合等变性/可选投影/ASR 约束、32 点群 smoke tests 与参数预算重统计要求。

- 2026-09-11：启动 GRACE、DPA4/SeZM、EquiformerV2 checkpoint 与 JARVIS-DFPT
  BEC 数据准备；建立资源审计和测试记录。
- 2026-09-11：完成可追溯资源准备单元：新增 backbone/BEC manifests、可恢复并发
  下载与逐原子 BEC 提取器、流式校验汇总器及 Slurm 作业脚本；GRACE 和
  DPA4-Plus 已落盘，BEC 通过 10 个真实样本验证，EquiformerV2 明确记录为门控阻塞。
- 2026-09-11：经用户明确授权，将资源准备提交 `e65c0e0` 推送到
  `origin/main`；随后确认 Guqq SSH 在密钥交换前由服务端关闭，完整 BEC Slurm
  作业尚未提交，并按连续三次失败规则记录网络诊断经验。

- 2026-09-11：将 proposal 顶层架构收敛为五个固定分支：`B+R`、`B+A+R`、`B+A+O3E+R`、`B+PGE+R`、`B+A+PGE+R`；统一 PG expert 的 `a1_only | full_pg` 与 adaptation/O3E/readout 的 `full_o3 | o2_tp` 配置轴，并同步方法图、前向伪代码、参数口径、实验矩阵、phase/MVP 和总结。
- 2026-09-11：按用户要求复测 Guqq 连接；verbose SSH 与 `ssh-keyscan` 均确认
  TCP 22 可达但服务端在 SSH banner/key exchange 前主动断开，本地 alias、用户、
  端口和 identity 配置正确。故障继续定位为服务端 sshd/pre-auth 网络策略问题，
  `git pull` 和后续 Slurm 操作均未能执行。
- 2026-09-11：完成本机 SSH 跳板配置：备份用户级 config，为 `Host Guqq` 增加
  `ProxyJump vlab`，并通过 `ssh -G` 静态解析、vlab BatchMode 登录及 Guqq
  BatchMode 端到端状态 0 验收。首次 Guqq 远程操作按规范执行 `git pull`，但因
  默认登录目录不是 Git 仓库返回状态 1；SSH 免密链路本身已确认正常。
