# Agent progress updates

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
