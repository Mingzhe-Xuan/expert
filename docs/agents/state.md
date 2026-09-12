# Agent state

## 当前状态（JARVIS backbone + readout benchmark）

正式 benchmark 训练入口已完成本地实现与完整回归：支持完整 published split、冻结 source
feature 一次提取与 checksum/split-gated cache、direct `B+R` 多 epoch 训练、validation early
stopping/best checkpoint，以及原始物理坐标下 Fnorm/EwT25/10/5 和公开目标逐项比较。
Guqq 上 408、409、410_[0-2] 仍是全量训练前的非 Eq 真实 backbone 最小运行探针，终态待收集；
EquiformerV2 checkpoint 下载继续按用户要求暂停，正式 benchmark 尚未产出数值。

提交前协议复核发现 production elastic manifest 当前筛选为 14,480，而 GMTNet A 级协议及
官方第二阶段预处理实际要求 14,220。正式 elastic CLI 将先 fail closed；正在实现官方
structural-symmetry zero screening、compact support mask 与 Slurm candidate-manifest 生成路径。

上述协议修复与 exact Voigt metric 现已完成本地实现并通过 203 项完整回归；`build_jarvis_elastic_manifest.sbatch`
会在 compute partition 生成候选，不在服务器修改 Git 源码。下一步需先同步并运行该 CPU job，
验证输出确为 14,220 后 scp 回本地提升 manifest；dielectric 不受此数据协议差异影响。

用户已再次确认 Guqq 网络恢复，当前从 external blocked 转入 fresh recovery：先提交本地
状态记录，再严格 pull-first 收集 408–410、容量和 scheduler 证据。只有门控通过才提交
elastic candidate 与 MACE dielectric；elastic 训练仍等待 14,220 manifest 本地提升。

## 当前计划（JARVIS backbone + readout benchmark）

1. pull-first 连接 Guqq，收集 408、409、410_[0-2] 的 scheduler、JSON、JUnit 和日志证据。
2. 通过 Slurm 重建 14,220 elastic candidate manifest，scp 回本地验证/提交；在此之前只允许
   dielectric 进入训练门控，elastic 必须拒绝当前 14,480 split。
3. 通过 Slurm 运行 MACE/GRACE/DPA4；按验证结果逐步调参，
   与 `docs/benchmarks/README.md` 的 A 级指标逐项比较。

## 变更记录（JARVIS backbone + readout benchmark）

- 2026-09-12 16:36 +08:00：恢复持久 Goal 并完成初始审计。确认公开对照表和完整 split 已冻结，
  但现有训练入口仅为五结构 smoke，不能产生 benchmark 验收指标；先收集非 Eq runtime 探针，
  再进入正式 trainer 实现，Eq checkpoint 保持暂停。
- 2026-09-12：完成正式 backbone + direct readout trainer 基础：四 adapter 暴露 frozen
  pre-interface tap，缓存严格绑定 checkpoint SHA/split IDs；readout edge basis 按 target 扩展到
  elastic `l=4`；完整项目 199 passed，compile、CLI help、shell syntax 与 diff 检查通过。
  下一阶段为提交同步并在 Guqq 先跑 MACE 两个 JARVIS 单元。
- 2026-09-12：实现与研究记录已提交并推送为 `1e4958a`。随后三次 pull-first Guqq 连接均只
  返回 Vlab banner 并以状态 1 结束，未取得 pull、408–410 探针或容量证据；按既有网络经验
  停止本轮盲连，训练提交门控保持关闭。下一次恢复时仍从同一只读证据收集步骤继续。
- 2026-09-12：跨轮 Guqq 恢复仍在 Vlab 层以状态 1 退出。并行完成 GMTNet elastic 官方源码
  审计与协议修复：当前 14,480 manifest 在正式 CLI 中 fail closed；新增 14,220 candidate
  Slurm generator、逐记录 36-bit support mask 和 loader zero projection，完整 202 tests 通过。
- 2026-09-12：进一步从 GMTNet official test loop 确认 elastic Fnorm/EwT 使用 6x6 Voigt
  flatten 且相对误差分母加 `1e-5`；修复 runner 的 Cartesian shear 重复计数后完整 203 tests
  通过。第二次同步仍停在 Vlab，candidate job 未提交。
- 2026-09-12：协议与指标修复分别推送为 `f209a1a`、`13c4485`；本轮第三次连接明确由
  Vlab `202.38.75.226:22` 主动关闭。未发生 pull、容量检查或 Slurm 提交，远端状态不变。
- 2026-09-12：第三个连续 Goal 回合的 bounded pull-first 连接仍只返回 Vlab banner 后状态 1
  退出。由于剩余工作全部要求 Slurm 且无合法替代路径，阶段正式转为 external blocked；
  未把本地测试、入口实现或不可比数据误当作 benchmark 完成证据。
- 2026-09-12：Goal 自动续跑后的 fresh recovery audit 第一次检查仍只到达 Vlab；修正本地
  quoting 后等待 60 秒也没有任何 Guqq pull 输出，终止时两个 SSH 层由远端关闭。没有进行
  登录节点计算或 Slurm 提交，外部阻塞状态保持不变。
- 2026-09-12：fresh recovery audit 第二回合使用无循环的固定参数命令，仍在 60 秒内仅返回
  Vlab banner，未获得任何 Guqq pull 或 Slurm/容量输出后终止。该恢复审计已连续两回合同类
  失败；训练指标和远端作业终态仍未知。
- 2026-09-12：fresh recovery audit 第三回合的 `ssh -vv` 证明 Vlab 与 Guqq 均认证成功且
  Guqq 接受了命令，但在强制 `git pull` 返回前通道 `Broken pipe`；pull 终态未知，`&&` 后的
  scheduler/Slurm 命令无执行证据。连续三回合相同传输阻塞再次满足 external-blocked 阈值。
- 2026-09-12：用户确认网络恢复；重新激活 benchmark Goal。下一步先提交该状态，再执行
  pull-first 远端检查和严格门控的 Slurm 提交；EquiformerV2 继续暂停。

## 当前状态（Dataset point-group balance）

已完成本地四个训练单元的 crystallographic point-group 描述性统计与 current-PG expert
训练负载审计。三个完整训练集均严重不平衡；完整 JARVIS-DFPT BEC 尚未落盘，因此只报告
10 条成功提取样本，不据此判断正式训练平衡性。

## 当前计划（Dataset point-group balance）

1. 将 `docs/analysis/dataset_pg_balance.md` 与机器可读 JSON 作为当前基线。
2. 完成 BEC 全量提取并冻结 split 后重跑同一统计。
3. material-specific parent DAG 可用后，补充实际 active-expert frequency，而非只看 current PG。

## 变更记录（Dataset point-group balance）

- 2026-09-12：开始本地数据集 PG 描述性统计；已确认 dielectric、elastic、MatTen 为完整
  本地冻结资源，JARVIS-DFPT manifest 仍为 full extraction pending，本地仅有 10 条成功样本。
- 2026-09-12：完成 29,478 条正式 split/本地成功结构的 PG 重判、频数守恒与不均衡审计；
  三个完整 train split 均为严重不平衡。报告、JSON 和可复现生成器已完成校验。

## 当前状态（Guqq network recovery）

用户确认 Guqq 网络恢复，Goal 从外部网络阻塞中继续；EquiformerV2 checkpoint 下载按用户
要求暂停，不尝试访问 gated 资源。十个非 Eq 资源已完成校验和提升；修复提交 `5472bb2`
已同步 Guqq，最小探针 408、409、410_[0-2] 已提交。其后跳板连续三次只返回欢迎信息，
当前停止盲监控且不推断作业终态。

## 当前计划（Guqq network recovery）

1. 间隔一轮后先 pull-first 恢复只读连接，收集 408、409、410_[0-2] 的终态及 JSON/JUnit。
2. 仅当五项全部通过，才恢复剩余 non-Eq arrays；否则从精确失败证据进入下一修复单元。
3. 保存 Git/env/Slurm 证据；`sacct` 因 accounting storage 未配置保留为明确限制。

## 变更记录（Guqq network recovery）

- 2026-09-12：收到网络恢复通知并恢复 Goal；明确暂停 EquiformerV2 checkpoint 下载，
  先完成 JARVIS/MatTen 与 MACE/GRACE/DPA4 资源同步及可独立验收工作。
- 2026-09-12：增量 Git bundle pull 成功，Guqq 同步至 `9bcd5cf` 并创建 staging；四个
  小文件上传成功，但 17.85 MB GRACE shard 被跳板中断。调整为 8 MiB 分块、重组后完整
  size/SHA 校验、通过才原子提升；远端仅余 13 GiB，完整 BEC job 需另行容量门控。
- 2026-09-12：DPA4、JARVIS elastic、MatTen elastic 及前六个 MACE 分块已传至隔离
  staging；`mace_medium.part.006` 连续三次被跳板重置，已查阅并补充网络经验，改为只把
  该块细分为 4 MiB 唯一命名子块。任何远端分块在整文件 size/SHA 通过前均不采信。
- 2026-09-12：全部非 Eq 分块已传完且 Guqq pull 至 `2b3f3a4`；首次整体验证因从缩略
  进度记录抄入了错误的数据集 SHA 而在首文件立即失败，guard 保证十个最终路径均未提升。
  已改为直接读取 committed manifests 的完整 SHA，并在下一 pull-first 连接重新整体验证。
- 2026-09-12：十个非 Eq 文件已在 Guqq 全部通过固定 size/SHA 并原子提升，服务器同步至
  `7288f36`，余量约 12 GiB。进入 Slurm 提交阶段：排除全部 Eq indices；完整 BEC 因
  input/output 容量安全边界不足继续门控，不提交。
- 2026-09-12：已提交 Slurm `360` full tests、`361/362/363` 三 backbone smokes、`364`
  fixture builder 与 `365` 的 12 个 non-Eq/non-BEC real rows。下一步以 `afterok:364`
  提交 44 个非 Eq PG rows，并持续审计至终态。
- 2026-09-12：44-row non-Eq PG array `366` 已以 `afterok:364` 提交。Guqq 禁用了 Slurm
  accounting storage，`sacct` 不可用；改用 `scontrol`（记录仍在时）及任务自带 stdout、
  stderr、JSON、JUnit 交叉验收，并把 accounting 缺口保留为明确限制。
- 2026-09-12：`360` full tests、`361` MACE standalone 已 `COMPLETED 0:0`；GRACE `362`、
  DPA4 `363` 与 real rows 0/1/2/4/5/6/8/9 已失败。连续失败超过三次后已查阅 runtime
  lessons，进入只读日志诊断，禁止盲重提；fixture `364` 尚在运行，PG array `366` 等待依赖。
- 2026-09-12：失败根因冻结为 float32 frame 阈值、DPA4 schema device、TensorFlow 2.20
  对 RTX 5090 sm_120 无 CUDA kernel 三项。先取消 365 未完成子任务与 366，保留 364；再在
  本地实现 dtype-aware 校验、DPA device 归一及显式可审计的 GRACE TF-CPU fallback。
- 2026-09-12：`364` 已 `COMPLETED 0:0`；366 已在依赖解除后启动，随即取消剩余工作。
  三项修复及回归测试已完成，本地 targeted 42、完整 182 tests 全绿；下一步只重提
  GRACE/DPA4 standalone 与 real 0–2 最小探针，通过后才恢复更大 arrays。
- 2026-09-12：修复提交 `6e7043f` 已同步 Guqq；最小 Slurm 探针为 402 GRACE、403 DPA4、
  404 real 0–2，初始均 pending。只有五项全部 `0:0` 且 JSON/JUnit passed 后才恢复 arrays。
- 2026-09-12：real row 0 已 `COMPLETED 0:0`，确认 frame 修复；GRACE 已绕过 PTX 后暴露
  `float_dtype` 必须为字符串的 API 错误；DPA4 仍有 device mismatch，待读取 403/404_2
  精确 traceback 后再修。大数组继续暂停。
- 2026-09-12：traceback 已将两条 DPA 路径分离：standalone 403 已完成 adapter forward，
  失败于 e3nn 0.5.9 用 CUDA rotation 构造验证矩阵；real 404_2 则在 adapter 构造前失败于
  O(2) finite-group CG intertwining。先修确定的 API/device 边界，再以 Slurm 复现 CG 数值差异。
- 2026-09-12：显式 float64 generator、CPU-first 验证矩阵及 GRACE dtype 修复已提交为
  `5472bb2`，Guqq 同步后提交 408/409/410_[0-2]。随后三次监控连接均只返回跳板欢迎信息；
  已按 lessons 停止盲重试，保留作业运行/结果且不宣称终态。

## 当前状态（Efficiency report aggregation）

已完成逐次 smoke 的 runtime efficiency 采样及跨 58-row PG 与 20-row real-subset
结果的严格汇总器；机器可读 JSON 和 Markdown 报告共享同一组已校验记录。

## 当前计划（Efficiency report aggregation）

1. 定义必需字段、实验身份和数值约束，缺失、失败或重复记录一律拒绝。
2. 递归读取 smoke JSON，生成稳定排序的 versioned JSON 与 Markdown 表。
3. 添加 CLI、单元测试和 README 用例，执行 targeted/full pytest、compile/help/diff checks。

## 变更记录（Efficiency report aggregation）

- 2026-09-12：开始汇总报告实现单元；以已嵌入 58/20 smoke JSON 的 efficiency record
  为唯一事实来源，冻结 architecture/task/PG/mode/backend 身份与 active-downstream FLOPs 口径。
- 2026-09-12：efficiency JSON/Markdown 汇总器完成；严格核验 58/20 index coverage、passed
  状态、身份/数值/参数预算、CUDA 来源和 scope。Targeted 14、项目完整 178 tests 全绿；
  compile/help/diff checks 通过。真实报告仍待 Guqq arrays 产出后生成。

## 并行 Goal 状态（Guqq per-backbone environments）

官方包元数据证明单一 venv 无法同时满足冻结的 MACE、GRACE、DPA4 与 Equiformer runtimes。
所有 Slurm 入口现已使用四个显式 venv contract；四套隔离 Python 3.10.12 环境均已在 Guqq
幂等创建。两个 mixed arrays 按已冻结且验证的 `index % 4` backbone schedule 选择环境；
MACE/core 已完成 CUDA 12.8 精确依赖安装并通过 `pip check`。GRACE 安装连续三次被 Guqq
四套隔离 runtime 均已在 Guqq 完成精确安装并通过 `pip check`；Equiformer 使用 scoped
HTTP/1.1 pull 恢复网络，并修复 wheel-only resolver 的旧 Hydra/OmegaConf 回退。四套 runtime
lock 已固化。资源同步连续三次因跳板/Guqq 出站网络中断，已按 `lessons.md` 暂停盲连；
本地 10-file contract 完整。四个 standalone adapter smoke 的失败证据、JUnit、Git/environment
fingerprint contract 已完成并通过全量测试；Equiformer gated checkpoint 与全部 Slurm 验收仍未完成。

## 并行 Goal 变更记录（Guqq per-backbone environments）

- 2026-09-12：确认 MACE/e3nn 0.4.4、fairchem/Torch 2.4、DeepMD/Torch 2.11/e3nn 0.5.9
  与 GRACE/TensorFlow 的不可合并边界；冻结四变量 Slurm selector 与静态映射测试。
- 2026-09-12：四环境 selector、九个相关 launchers 与 fail-closed 测试完成；targeted 13、
  完整 144 tests 全绿，全部 shell syntax/compile/diff checks 通过。下一步在 Guqq 建立三套
  剩余 venv，并安装/记录四套精确环境。
- 2026-09-12：Guqq 先拉取至 `e2aa7cd`，四套隔离 venv 均确认 Python 3.10.12、无 system
  site packages；环境创建阶段完成，转入 MACE/core 起始的逐套安装与版本验证。
- 2026-09-12：MACE/core 安装 Torch 2.11.0+cu128、e3nn 0.4.4、MACE 0.3.16 及冻结科学栈，
  `pip check` 无冲突，磁盘剩余 41 GiB；转入 GRACE/TensorPotential 环境安装。
- 2026-09-12：GRACE 三次尝试均在首个强制 Git pull 因 Guqq 出站 GitHub 网络失败而终止，
  未执行安装；按 lessons 暂停重试，转为本地 MACE freeze lock 与静态验收单元。
- 2026-09-12：完成 71-entry MACE/core freeze lock、官方 cu128 index 与关键版本静态验收；
  targeted 3、完整 147 tests 全绿，compile/diff checks 通过。
- 2026-09-12：间隔重试 pull 恢复；GRACE 安装 TensorPotential 0.6.0、TensorFlow 2.20.0、
  Torch 2.11.0+cu128 与 e3nn 0.5.9，`pip check` clean，freeze SHA 已记录，剩余 31 GiB。
- 2026-09-12：完成 95-entry GRACE freeze lock 与关键版本静态验收；targeted 5、完整
  149 tests 全绿，compile/diff checks 通过。下一步安装 DPA4 runtime。
- 2026-09-12：DPA4 三次连接均未完成首个强制 pull，全部安装被 `set -e` 门控且 venv
  未改变；停止本轮网络重试，保留 MACE/GRACE 已验证边界等待下一次间隔恢复。
- 2026-09-12：跨回合间隔后 pull 恢复；DPA4 安装 DeepMD-kit 3.2.0、Torch 2.11.0+cu128、
  e3nn 0.5.9 与冻结科学栈，`pip check` clean，freeze SHA 已记录，剩余 24 GiB。
- 2026-09-12：完成 80-entry DPA4 freeze lock 与关键版本静态验收；targeted 7、完整
  151 tests 全绿，compile/diff checks 通过。下一步安装 EquiformerV2 runtime。
- 2026-09-12：Equiformer runtime 三次连接均未完成强制 pull，wheel 安装未执行且 venv
  保持空环境；按三次失败规则暂停本轮 Guqq 重试。
- 2026-09-12：HTTP/1.1 pull 恢复后，拦截并修复旧 Hydra/OmegaConf 回退；Equiformer 安装
  fairchem-core 1.10.0、Torch 2.4.1+cu121、e3nn 0.5.9，`pip check` clean，剩余 15 GiB。
- 2026-09-12：Equiformer 119-entry exact lock 与 resolver guard 完成；targeted 9 tests、完整
  153-test suite、compile/diff checks 全绿。四套 Guqq runtime 阶段完成，下一步上传已校验资源。
- 2026-09-12：开始资源同步阶段；冻结 10 个 contract 传输对象及其本地 size/SHA-256，采用
  单一 SSH transport 保证远端先 pull、后写入 ignored data/checkpoint 路径并逐项复核。
- 2026-09-12：首个 tar stream 在 pull 成功后约 21 秒被跳板重置；不采信任何半成品。
  调整为先 pull 的 multiplex transport 加逐文件原子 scp；未被 manifest 引用的 GRACE
  `gmm_artifacts.npz` 不属于 loader contract，已从 11-file 初稿范围剔除。
- 2026-09-12：资源同步第二、三次均未通过 mandatory pull，分别为 GnuTLS `-110` 与
  GitHub 443 超时 133932 ms；已查阅 `lessons.md` 的出站网络经验并暂停本阶段重试。
  三次均未形成可验收上传，下一步保留原子复制方案，等待外部网络状态变化。
- 2026-09-12：Definition-of-Done 审计发现四个 standalone adapter jobs 仅成功时写 JSON，
  缺少失败摘要、JUnit 与独立 Git/environment fingerprints；开始统一证据 contract，另修复
  DPA4 smoke 对未导入 `DPA4_SO3_LAYOUT` 的运行时引用。
- 2026-09-12：standalone adapter evidence contract 完成；成功/失败均持久化 JSON/JUnit，
  四个 sbatch 固化 Git/env artifacts，DPA4 symbol regression 修复。Targeted 17、完整 157 tests
  全绿，compile、四个 `bash -n` 与 diff checks 通过。下一步等待 Guqq 网络恢复后上传资源。
- 2026-09-12：继续 DoD 审计，开始补齐 32-PG fixture builder 与完整 BEC preparation 两个
  required data jobs 的 Git/env/summary/JUnit/Slurm-ID 证据 contract；先抽象无设备单 case runner。
- 2026-09-12：两个 required data jobs 的证据 contract 完成；BEC 当前尝试 error log 可恢复，
  全索引 record-count fail-closed。Targeted 27、完整 161 tests 全绿，scoped compile、两个
  `bash -n` 与 diff checks 通过。资源与 Guqq Slurm 实跑仍等待网络恢复。
- 2026-09-12：完成两个本地 DoD 单元后进行一次有界恢复检查；Guqq 可连接，但首条 scoped
  HTTP/1.1 pull 再次以 GnuTLS `-110` 结束，未到 `MASTER_READY`、未写入资源。本回合停止连接。
- 2026-09-12：继续最终证据审计；发现仓库仅文字要求人工查看 `sacct`，没有逐任务机器校验。
  开始实现冻结 job manifest 与 strict sacct parser/auditor，覆盖 58/20 arrays 及所有单作业。
- 2026-09-12：strict sacct audit 完成；精确展开 expected `JobIDRaw`，拒绝缺失、重复、
  非 `COMPLETED` 与非 `0:0`，并保存 raw/JSON。Targeted 12、完整 167 tests 全绿，CLI
  help、compile/diff checks 通过。最终 scheduler evidence 仍需 Guqq 实跑后生成。
- 2026-09-12：DoD 审计继续发现 efficiency README 无对应实现；开始增加真实 forward latency、
  CUDA peak allocation、PyTorch-dispatched active downstream FLOPs，以及 total/trainable/
  active parameters 和逐样本 active-expert counts，并接入 58/20 smoke reports。
- 2026-09-12：runtime efficiency evidence 完成并接入两套 smoke；GRACE 外部 TensorFlow
  frozen variables 单列计入 total parameters，FLOPs scope 明确不冒充 TF 总量。Targeted 16、
  完整 170 tests 全绿，compile/diff checks 通过。真实 CUDA 数值待 Guqq jobs 生成。
- 2026-09-12：两项 efficiency 补缺推送后，Guqq 一次 pull 成功并同步至 `6395e15`，确认
  远端资源仍不完整；后续 control-master 被重置，SCP staging 的 pull-first 连接又以 GnuTLS
  `-110` 失败。未创建 staging、未上传或覆盖文件，保留原子方案等待间隔恢复。

## 并行 Goal 状态（material-oriented target projection）

已将 dielectric/elastic global readout 从 point-group symbol 的 representative Hall projector
改为逐材料 `SymmetryRecord.rotations` Reynolds projector。非代表 orientation、mixed batch、
幂等/不变性/梯度与完整 26-config 回归均通过；BEC raw head 继续不使用该 projector。

## 并行 Goal 变更记录（material-oriented target projection）

- 2026-09-12：冻结 actual-operation projector 的 shape/task/device、幂等、不变性、梯度与
  non-representative orientation 测试；下一步实现 heads/readout 接口并跑完整套件。
- 2026-09-12：operation-aware projector 与 readout 已完成；targeted 12/15 tests 及完整
  133-test suite 全绿，compile/diff checks 通过。下一步回到 Guqq 环境与资源准备。

## 当前状态（Published benchmark 跑分审计）

已完成 JARVIS tensor dielectric/elastic、MatTen elastic 与 JARVIS-DFPT BEC
四个 benchmark 的公开模型结果审计。汇总表按 A/B/C 三级区分直接可比、论文内可比
和不可直接比较结果；新增 MatTen、CEITNet、JARVIS-DFPT 原文及 ALIGNN 官方表快照，
并记录全部本地来源的 URL、SHA-256 和访问状态。

## 当前计划（Published benchmark 跑分审计）

1. 后续得到本项目正式跑分后，优先追加到相同 split 的 A 级表。
2. 若 OpenReview/RSC 端点恢复，补归档 IrredNet PDF 与 MatTen ESI 并校验 checksum。
3. BEC 正式比较前，在冻结 split 上重跑 ETGNN，避免把未发布 split IDs 的 0.045 e
   直接当作同 test set 排名。

## 变更记录（Published benchmark 跑分审计）

- 2026-09-12：开始公开 benchmark 跑分审计；冻结四个训练单元和 primary-source-only
  证据标准，先登记文档验收检查，再开展网络检索与来源归档。
- 2026-09-12：完成公开跑分表和来源归档；识别 GMTNet/GoeCTP/IrredNet 的不同重跑
  family、MatTen 的 derived-modulus 指标边界，以及 ETGNN BEC 0.045 e 的 split-ID 缺口。
  本地链接、表格列、PDF magic/Poppler 解析、SHA-256 与 `git diff --check` 均通过。

## 并行 Goal 状态（32-PG equilibrium fixtures）

在不覆盖 published benchmark 审计改动的前提下，进入 32 点群真实平衡结构 fixture 管线。
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

## 历史状态（Phase A 正式实现启动）

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
