# Agent state

## 当前状态（Phase A 正式实现启动）

已建立正式 `src/` 包的 12 个一级模块边界和 README，落地六项核心 typed
contracts、三类 target layouts、checkpoint convention fail-closed 校验、五分支
配置 schema 与冻结 26-config manifest。新增测试 15/15 通过，合并现有本地套件
27 passed、1 个旧原型 opt-in skip；Goal 仍处于 Phase A，尚不具备完成证据。

## 当前计划（Phase A 正式实现启动）

1. 完成 Phase A 数学 registries：点群表示、subduction/inverse、CG/path/copy order
   和 Cartesian↔irrep transforms，并补齐 round-trip/intertwining tests。
2. 实现 canonicalization 与 Hall-level embedding operation validation。
3. Phase A gate 全绿后进入 Phase B 的 cutoff PBC graph 和真实 backbone adapters。

## 变更记录（Phase A 正式实现启动）

- 2026-09-11：读取 `GOAL.md`、原型、资源 manifest、Git 状态和既有代理记录；
  确认正式实现尚未开始。进入 Phase A，先完成包骨架、核心 contracts 和 26 配置
  schema，并在实现前登记对应测试。
- 2026-09-11：完成正式模块骨架、typed contracts、target layouts、convention
  checksum 和 26-config schema/manifest；新增测试 15/15 通过，完整本地集合为
  27 passed、1 个旧 opt-in skip。下一实现单元转入数学 registries 与 transforms。

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
