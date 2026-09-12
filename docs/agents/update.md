# Agent progress updates

- 2026-09-12：开始四个本地训练单元的 `frequency × PG` 描述性统计；冻结使用
  manifest split 与模型相同 spglib 容差，并将 current-PG 样本平衡和 parent-DAG 实际路由
  负载分开解释。
- 2026-09-12：完成 29,478 条正式 split/本地成功结构的 32-PG 频数统计，发布 Markdown
  与机器可读 JSON；三个完整 train split 均严重不平衡，BEC 因仅 10 条本地样本不可判定。

- 2026-09-12：用户确认 Guqq 网络恢复，继续资源与 Slurm 阶段；按指示暂停
  EquiformerV2 checkpoint 下载，只推进十文件非门控资源及不依赖 Eq 的验收任务。
- 2026-09-12：Guqq 可达但服务器 GitHub HTTPS 仍报 GnuTLS `-110`；改用本地 committed
  `main` Git bundle 经 SCP 传入，再以该 bundle 的 `git pull --ff-only` 作为首个远程操作。
- 2026-09-12：23.2 MB complete-history bundle 的 SCP 被跳板中断，未采信远端副本；
  改为以服务器已有 `6395e15` 为 prerequisite 的最小增量 bundle，再由 Git pull 验证。
- 2026-09-12：6.9 KB 增量 bundle pull 成功并创建 staging；四个小资源上传成功，
  17.85 MB shard 中断。剩余资源改为 8 MiB 分块 SCP，重组后全量 size/SHA 原子提升。
- 2026-09-12：已传完 DPA4、JARVIS elastic、MatTen elastic 和 MACE 前六个 8 MiB 分块；
  MACE 第七块连续三次被跳板重置，按规范停止盲重试并补充经验，改为该块的 4 MiB
  唯一命名子块。远端部分文件继续视为不可信 staging，整文件哈希通过后才原子提升。
- 2026-09-12：非 Eq 分块已全部传至 staging，bundle pull 成功；整体验证在首个 dataset
  SHA 即因缩略记录转抄错误而 fail closed，未提升任何最终文件。现改用 committed manifest
  原值与本地独立哈希双重核对，再执行 all-verify-before-any-rename。
- 2026-09-12：十个非 Eq 数据/checkpoint 文件已全部通过 manifest size/SHA 并在 Guqq
  原子提升；下一阶段提交 full tests、fixture builder、三 standalone smokes 及排除
  `index % 4 == 3` 的两组 Slurm arrays。完整 BEC 仍因 12 GiB 余量不足不提交。
- 2026-09-12：Guqq 已提交 jobs `360`–`365`：full tests、MACE/GRACE/DPA4 standalone、
  fixture builder 及 12 个 non-Eq/non-BEC real rows；Eq 和未完成 BEC rows 均未提交。
  下一步提交依赖 `afterok:364` 的 44-row 非 Eq PG array 并监控全部终态。
- 2026-09-12：已提交 non-Eq PG array `366`（44 rows，`afterok:364`）。集群明确返回
  `Slurm accounting storage is disabled`，因此 strict `sacct` audit 无法执行；后续使用
  `scontrol` 与 stdout/stderr、JSON/JUnit 证据交叉监控，并显式保留此验收限制。
- 2026-09-12：`360/361` 已 `COMPLETED 0:0`；`362/363` 和多个 real rows 非零失败，
  row 10 通过，`364` 仍运行、`366` 等待依赖。因同类 real smoke 连续失败超过三次，
  已查阅 separate-runtime 经验并转入 JSON/JUnit/日志诊断，不盲目重提。
- 2026-09-12：诊断确认 real float32 frame 逆矩阵阈值过严、DPA4 neighbor schema 混用
  CPU/CUDA，以及 TensorFlow 2.20 不支持 RTX 5090 sm_120 CUDA kernel。先取消待运行的
  365/366 工作，再本地修复并以 targeted/full tests 验证后重提。
- 2026-09-12：364 fixture builder 已 `COMPLETED 0:0`；366 在依赖解除后已启动，现已取消
  余下任务。完成 dtype-aware frame tolerance、DPA4 schema device 归一和显式 GRACE
  TF-CPU fallback；targeted 42、完整 182 tests、compile/shell/diff checks 全部通过。
- 2026-09-12：`6e7043f` 已同步 Guqq，并提交最小修复探针 402（GRACE）、403（DPA4）、
  404（real 0–2）。初始均 pending；五项全部通过前不恢复大数组。
- 2026-09-12：real 0 已 `COMPLETED 0:0`，frame 修复成立；GRACE CPU fallback 越过 PTX
  后发现 `float_dtype` 字符串契约，DPA4 尚余一处 device mismatch。先读取精确 traceback，
  不重提或扩大数组。
- 2026-09-12：403 traceback 确认 DPA4 adapter forward 已通过，余错位于 e3nn validation
  matrix 的 CPU/CUDA 边界；404_2 则独立失败于 O(2) CG。Slurm 407 在 e3nn 0.5.9 下复现
  992/1000 组合失败，定位为 Wigner generator 默认 float32 污染。现已实现显式 float64
  generator、CPU-first smoke matrix 及 GRACE 字符串 dtype；targeted 51、full 194 tests 全绿。
- 2026-09-12：`5472bb2` 已推送并同步 Guqq，提交 408 GRACE、409 DPA4 与 410 real 0–2。
  后续三次监控连接均只返回跳板欢迎信息；已查阅既有网络 lesson 并停止盲重连，等待跨轮
  间隔后再收集终态。Eq checkpoint 与 Eq index 始终未访问。

- 2026-09-12：完成两项本地 DoD 补缺并推送后，登记一次有界 Guqq 资源同步恢复检查；
  仍以 HTTP/1.1 `git pull --ff-only` 为首个远程操作，失败即停止且不写资源。
- 2026-09-12：Guqq pull 已恢复并同步至 `6395e15`；远端现有 dielectric 文件大小不符
  manifest，未采信也未覆盖。登记同一持久 transport 下的逐文件临时上传、校验与原子替换。
- 2026-09-12：SSH control-master 握手被跳板重置且未写文件；调整为许可内的逐文件
  `.upload-part` SCP，之后在 pull-first 校验连接中统一核对 size/SHA 并原子替换。
- 2026-09-12：SCP staging 的 pull-first 连接再次以 GnuTLS `-110` 结束，guarded mkdir
  未执行、传输未开始；停止本轮即时重试，保留已核验 10-file contract 与原子上传方案。

- 2026-09-12：开始 efficiency report aggregation 单元；将 58/20 smoke 内嵌记录汇总为
  同源、稳定排序且 fail-closed 的 JSON/Markdown 交付物。
- 2026-09-12：完成严格 58/20 coverage 与 efficiency schema/numeric/CUDA/scope 校验、
  稳定 JSON/Markdown 渲染和 CLI；targeted 14、完整项目 178 tests 全绿。

- 2026-09-12：开始 runtime efficiency evidence 单元；补齐 README 已声明但代码缺失的参数、
  active experts、active downstream FLOPs、end-to-end latency 与 CUDA peak memory 报告。
- 2026-09-12：完成并接入两套 smoke 的 efficiency profiler；GRACE external frozen variables
  纳入 total parameter 口径，FLOPs 显式限定 active downstream scope。Targeted 16、完整 170 tests 全绿。

- 2026-09-12：开始 strict Slurm terminal-state audit 单元；将用 versioned job manifest 精确
  展开并核验全部单作业与 array tasks，替代无法形成完成证明的人工 `sacct` 浏览。
- 2026-09-12：完成 strict sacct CLI、manifest schema 与数组完整性校验；targeted 12、完整
  167 tests 全绿，CLI help/compile/diff 通过。真实 audit JSON 待 Guqq jobs 结束后生成。

- 2026-09-12：开始 required data-job evidence 单元；目标是让 32-PG fixture 构建和完整 BEC
  prepare/validate 作业也具备失败保留的 summary/JUnit、Git/env 与 Slurm-ID 证据。
- 2026-09-12：完成 32-PG builder 与 BEC prepare/validate job-level evidence；修正 BEC resume
  的旧错误污染并强制全索引计数。Targeted 27、完整 161 tests、scoped compile/shell/diff 全绿。
- 2026-09-12：间隔两个本地实现单元后做一次 pull-first 恢复检查，仍以 GnuTLS `-110`
  失败且未开始上传；遵守单次界限，本回合不再连接 Guqq。

- 2026-09-12：DoD 审计发现 standalone adapter jobs 缺少失败 JSON/JUnit 与独立 Git/env
  fingerprints，并发现 DPA4 smoke 的 source-layout 符号未导入；已冻结对应行为与静态测试。
- 2026-09-12：完成共用 fail-preserving adapter runner、四套 sbatch Git/env/JSON/JUnit 证据链
  与 DPA4 layout import 修复；targeted 17、完整 157 tests 全绿，compile/shell/diff checks 通过。

- 2026-09-12：开始 Guqq 资源同步单元；已复核三份 benchmark 数据和三套非门控骨干的
  10 个 contract 文件。首个 tar stream 在成功 pull 后被跳板重置；改为先 pull 的 multiplex
  transport 加逐文件原子 scp，且剔除未进入 GRACE loader contract 的额外 GMM artifact。
- 2026-09-12：资源同步后两次 pull 分别以 GnuTLS `-110` 和 GitHub 443 超时失败；连续
  三次未形成完整上传，已查阅 `lessons.md` 并暂停盲连。保留已验证本地资源和原子复制方案。

- 2026-09-12：用 scoped HTTP/1.1 恢复 Guqq pull，拦截 wheel-only 导致的旧 Hydra 回退；
  完成 Equiformer venv（fairchem-core 1.10.0、Torch 2.4.1+cu121、e3nn 0.5.9），
  `pip check` clean，freeze SHA 已记录，剩余 15 GiB。
- 2026-09-12：固化 Equiformer 119-entry exact lock 和 modern Hydra/OmegaConf guard；
  targeted 9 passed，完整本地 suite 153 passed、0 failed/skip，compile/diff checks 通过。

- 2026-09-12：Equiformer runtime 三次尝试均被强制 pull 的出站网络故障拦截，pip 未执行；
  本轮停止盲连。前三套环境及 locks 已验证，最后 runtime 与 Slurm evidence 尚待完成。

- 2026-09-12：完成 `requirements/guqq/dpa4.txt`：80 个唯一 exact pins 与 7 个关键版本
  断言；targeted 7、完整 151 tests 全绿，compile/diff 通过，准备 Equiformer runtime。

- 2026-09-12：跨回合间隔后 Guqq pull 恢复；完成 DPA4 venv（DeepMD-kit 3.2.0、Torch
  2.11.0+cu128、e3nn 0.5.9），`pip check` clean，freeze SHA 已记录，剩余 24 GiB。

- 2026-09-12：DPA4 安装三次均在强制 pull 阶段无输出终止，pip 未执行；按三次失败规则
  暂停本轮 Guqq 重连。MACE/GRACE 已验证，DPA4/Equiformer 与 Slurm evidence 仍待完成。

- 2026-09-12：完成 `requirements/guqq/grace.txt`：95 个唯一 exact pins 与 8 个关键版本
  断言；targeted 5、完整 149 tests 全绿，compile/diff 通过，准备 DPA4 安装。

- 2026-09-12：Guqq 间隔重试恢复；完成 GRACE venv（TensorPotential 0.6.0、TensorFlow
  2.20.0、Torch 2.11.0+cu128、e3nn 0.5.9），`pip check` clean，剩余 31 GiB。

- 2026-09-12：完成并验证 `requirements/guqq/mace-core.txt`：71 个唯一 exact pins、官方
  cu128 index、9 个关键版本断言；targeted 3、完整 147 tests 全绿，compile/diff 通过。

- 2026-09-12：GRACE 安装三次均由首个强制 pull 的 Guqq→GitHub TLS/443 故障安全拦截；
  环境未改变，停止盲目重连并启动已验证 MACE/core 完整 freeze lock 固化。

- 2026-09-12：完成 Guqq MACE/core venv：Torch 2.11.0+cu128、MACE 0.3.16、e3nn 0.4.4
  与冻结科学栈安装成功，`pip check` clean；安装后根文件系统尚余 41 GiB。

- 2026-09-12：MACE/core 首次依赖解析在安装前因 `python-hostlist` 无 wheel 而安全停止；
  同时识别 PyPI Torch 2.11 默认 CUDA 13 偏离冻结 CUDA 12.8，改用官方 cu128 wheel 索引。

- 2026-09-12：Guqq 已按规定先 pull；MACE/core、GRACE、DPA4、EquiformerV2 四套独立
  Python 3.10.12 venv 均已创建并确认不继承 system site packages，进入逐套安装阶段。

- 2026-09-12：官方 PyPI metadata 审计确认四 backbone runtime 不能共用一个 venv；启动
  `EXPERT_MACE_VENV|EXPERT_GRACE_VENV|EXPERT_DPA4_VENV|EXPERT_EQUIFORMERV2_VENV`
  Slurm contract 与 frozen array-index selector 实现。
- 2026-09-12：完成四 venv Slurm dispatch；58/20 rows 的 modulo-four mapping、缺变量失败、
  standalone/core launchers 与 shell syntax 均通过，完整项目 144 passed、0 skip。

- 2026-09-12：验收审计发现 global head 仍使用代表 Hall orientation，而非材料实际
  `SymmetryRecord.rotations`；启动 operation-aware Reynolds projector 修复与非代表 orientation
  回归测试，保持 BEC raw head 不依赖 symmetry projector。
- 2026-09-12：完成逐材料 operation-aware global projector；非代表 `mm2` orientation 与
  mixed-crystal readout 回归证明实际 fixed space、不变性、幂等和 backward，完整套件
  133 passed、0 skip，compile/diff checks 通过。

- 2026-09-12：开始 32-PG real-equilibrium fixture pipeline；将从已冻结 JARVIS/MatTen
  structures 经 Slurm 选择每点群一条，保存 Hall/source/checksum，并拒绝 synthetic prototype
  充当最终 fixture。保留同工作区 published benchmark 审计的全部并发改动。
- 2026-09-12：完成 equilibrium structure iterator、32-group deterministic fixture manifest、
  58-row real-checkpoint smoke、periodic graph/BEC joint audit 和完整 Slurm evidence wrappers；
  targeted 11 passed，完整本地套件 129 passed、0 skip，compile/sbatch syntax/diff checks 通过。
  未生成 synthetic acceptance asset；真实 fixture 与 GPU/Slurm 证据仍待 Guqq。

- 2026-09-12：开始 JARVIS tensor、MatTen elastic 与 JARVIS-DFPT BEC 的 published
  score 审计；将只从 primary sources 提取数值，下载公开来源到 `docs/`，并按数据与
  split 可比性生成表格。
- 2026-09-12：完成 `docs/benchmarks/README.md` 跑分汇总与 `SOURCES.md` 来源清单；
  下载 MatTen、CEITNet、JARVIS-DFPT 原文和 ALIGNN 官方 README 快照，区分原始报告、
  后续统一重跑与不同 target/split。链接、表格、PDF、checksum 和 whitespace 检查通过。

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
- 2026-09-11：开始 DPA4-Plus adapter；精确源码审计确认公开 sparse-edge builder、
  `src=neighbor/dst=center` 方向、e3nn-compatible packed real `(l,m)` order，以及 descriptor
  直接返回 scalar 前 final latent，避免依赖脆弱的 module forward hook。
- 2026-09-11：完成 DPA4-Plus adapter 候选、共享 parity edge-geometry 透传与
  `dpa4_adapter_smoke.sbatch`；目标测试 9 passed、合并 116 passed、1 旧 skip，compile/diff
  检查通过。真实 checkpoint 的 DeePMD 3.2/GPU 数值验收仍需在 Guqq 经 Slurm 执行。
- 2026-09-11：开始 GRACE adapter；测试计划先冻结 archive metadata、官方 restore、
  scalar readout 前 `rho` feature tap、真实 bond geometry、冻结梯度与 O(3) 数值验收。
- 2026-09-11：完成 GRACE adapter 候选；确认 `rho` 实为 17-channel scalar-only 并将 tap
  修正为 4512-component `AA`，实现 27 个 history copy 的显式 layout、runtime-derived
  TensorPotential→e3nn 基变换、artifact 逐文件 checksum 与 Slurm smoke。目标 11 passed、
  合并 118 passed、1 旧 skip；真实 checkpoint restore 尚待 Guqq。
- 2026-09-11：开始 EquiformerV2 adapter；先审计官方 fairchem checkpoint loader、最终
  node SO(3) coefficient tap 和 radius graph contract，同时保持 OMat24 gated resource fail closed。
- 2026-09-11：完成 EquiformerV2 adapter 候选；实现官方 checkpoint loader、final-normalized
  3200-component SO(3) tap、fairchem graph geometry 复用、paired-inversion O(3) completion、
  frozen/interface gradient smoke 与 Slurm 入口。目标 13 passed、完整项目 114 passed；受限
  checkpoint 的 checksum、restore 和数值验收仍待用户完成 OMat24 access 后在 Guqq 执行。
- 2026-09-11：开始四个独立 real-data modules；测试计划先冻结统一 tensor sample、三种
  source schema、published split、resource checksum、BEC site order 和 3/1/1 smoke selection。
- 2026-09-11：完成 real-data loader 候选；JARVIS dielectric/elastic、MatTen elastic 与
  JARVIS-DFPT BEC 统一为 immutable `TensorSample`，支持 frozen resource gate、published/
  seeded split、target irrep conversion、node order 和 3/1/1 smoke selection。目标 11 passed、
  完整项目 119 passed；三个本地真实源实读通过，完整 BEC output 继续 fail closed。
- 2026-09-12：开始 end-to-end model/smoke runner；先冻结 canonical frame target、backbone
  graph reuse、3/1/1 optimizer/checkpoint/metrics 和真实-vs-test-seam 边界，再实现 Slurm CLI。
- 2026-09-12：完成真实 backbone→原生 graph→dispatcher composition、canonical target batch、
  3/1/1 optimizer/checkpoint/metrics runner、20-run coverage schedule 与 Slurm array；定向
  4 passed、完整项目 123 passed。真实 jobs、GPU 指标与 checkpoint evidence 仍待 Guqq。

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
# 2026-09-12 — Full JARVIS backbone + direct-readout benchmark runner

- Added full published-split training for JARVIS dielectric and elastic with deterministic
  minibatches, train-only copy-aware normalization, AdamW, validation scheduling/early stopping,
  strict best-checkpoint restore, and original-frame Fnorm/EwT25/10/5 reporting.
- Added one-time frozen pre-interface feature extraction for all four adapters and tensor-only,
  atomic caches gated by backbone checkpoint SHA, training unit, split, exact sample IDs, and layout.
- Expanded readout edge harmonics to each target's maximum degree, restoring a legal elastic l=4
  path for the MACE l<=1 source tap. Added a non-Eq-only Slurm launcher while EqV2 is paused.
- Verification: benchmark targeted tests 5 passed; complete suite 199 passed; compile, CLI help,
  Slurm shell syntax, and diff checks passed. No full benchmark score has been claimed yet.
- Committed and pushed as `1e4958a`. Three subsequent pull-first Guqq attempts failed at the Vlab
  jump-host boundary, so probe/capacity gates could not be verified and no full training job was
  submitted; EquiformerV2 remained untouched.
# 2026-09-12 — Correct GMTNet elastic benchmark population

- Audited frozen official GMTNet source and found the existing 14,480-record manifest represented
  only its first magnitude screen, while the reported benchmark uses a second structural-symmetry
  forbidden-component screen before the seed-32 split and contains 14,220 records.
- Reproduced the official O(3)-probe support-mask algorithm, persisted compact per-record masks,
  applied the same zero projection in the loader, and made full benchmark training reject any split
  other than 11,376/1,422/1,422 for elastic (3,770/471/471 for dielectric).
- Added a CPU Slurm candidate-manifest job so server-side batch processing never edits tracked source.
  Verification: targeted 13 passed, full suite 202 passed, relevant compile/shell/diff checks passed.
  Full candidate generation awaits Guqq connectivity; no incomparable elastic job was submitted.
- Audited the official evaluation loop and corrected elastic Fnorm/EwT to contract predictions to
  the released 6x6 Voigt order before flattening, using the published `norm(label)+1e-5` denominator.
  The final regression state is 203 passed; another Guqq sync attempt failed at the same jump host.
- Pushed the protocol and metric fixes as `f209a1a` and `13c4485`. The final bounded connection was
  explicitly closed by the Vlab SSH endpoint, so no remote pull or job submission occurred.

# 2026-09-12 — Benchmark Goal externally blocked after third goal turn

- A third consecutive Goal-turn pull-first attempt again stopped at the Vlab→Guqq SSH boundary with
  no Guqq command output. No manifest, extraction, training, or evaluation job was submitted.
- All remaining acceptance work requires Slurm, so the active benchmark Goal is formally blocked
  rather than represented as complete. Resume from `main` after connectivity returns: pull first,
  collect 408–410, verify capacity, submit the elastic protocol candidate, then launch training.

# 2026-09-12 — Post-block recovery attempt 1

- The automatic Goal continuation triggered a fresh recovery audit. After correcting a local quoting
  error, a bounded pull-first SSH attempt remained at the Vlab banner for 60 seconds and was stopped;
  no Guqq command, scheduler query, or Slurm submission ran.
- The benchmark remains externally blocked. EquiformerV2 access stayed excluded.

# 2026-09-12 — Post-block recovery attempt 2

- A loop-free, fixed-argument pull-first SSH command again stopped after the Vlab banner. It produced
  no Guqq output for 60 seconds and was terminated; no remote mutation or Slurm submission is
  claimed.
- This is the second consecutive same-condition turn in the fresh recovery audit. Benchmark metrics
  remain unavailable and EquiformerV2 remains excluded.

# 2026-09-12 — Post-block recovery attempt 3 and renewed blocker

- Effective host configuration confirmed the intended Vlab proxy jump. Verbose evidence showed that
  both SSH authentications and Guqq command acceptance succeeded, followed by `Broken pipe` before
  mandatory pull output. Pull completion remains unknown and the guarded later commands have no
  execution evidence.
- This is the third consecutive same-condition recovery turn. All remaining benchmark work requires
  Slurm behind the mandatory pull gate, so the Goal is formally blocked again without any metric
  claim. EquiformerV2 remained untouched.

# 2026-09-12 — User-confirmed Guqq recovery

- The benchmark Goal is reactivated after the user confirmed network recovery. The next connection
  remains pull-first and will collect exact probe/capacity evidence before any Slurm submission.
- EquiformerV2 remains paused. Elastic training remains gated on generating, returning, validating,
  and committing the exact 14,220-record protocol manifest.
- The first recovery turn tried both non-PTY and PTY pull-first sessions. Each stopped at the Vlab
  banner without Guqq output and was boundedly terminated; no job was submitted. This starts a fresh
  blocked audit at attempt 1 while the Goal remains active.
