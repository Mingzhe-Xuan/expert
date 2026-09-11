# Test plan and results

## 2026-09-11 — Phase A point-group registry and invariant subspaces

计划检查：

- 正式 registry 从冻结 asset 读取且仅读取 32 个 crystallographic point groups，
  验证每组 operation closure、identity/inverse、order 和笛卡尔正交化。
- 对 `l=0..4` 的 natural O(3) carriers 生成确定性点群表示与 invariant subspace；
  projector 幂等、basis 正交且在全部群操作下不变。
- dielectric、elastic、BEC 三个 target layout 的 fixed-subspace bases 维数稳定，
  basis checksum 可进入 checkpoint conventions。
- 针对 proper/improper operation 验证表示同态；执行目标 pytest、compile 和
  `git diff --check`，结果在提交前补录。

实际结果：

- 首次测试收集因 PyTorch 2.6+ `weights_only=True` 与 e3nn 内置 Wigner 常量中的
  `slice` 对象不兼容而失败；实现改为仅向 safe globals 登记 Python 内置 `slice`，
  未关闭安全加载，也未放宽任何数学 tolerance。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_point_group_registry.py -q`：
  4 passed，覆盖 32 群 registry/正交化、抽样表示同态、全部 target invariant
  projector/basis 和 C1/Ci/cubic 已知 fixed-space 维数。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests assets/model_code/tests -q`：
  31 passed，0 failed，1 个既有 opt-in skip。
- `python -m compileall -q src tests/test_point_group_registry.py` 与
  `git diff --check`：通过；后者仅有 LF→CRLF 提示。

## 2026-09-11 — Phase A contracts and architecture schema

计划检查：

- `src/` 的 12 个一级模块均可导入且各自包含 README。
- `ArchitectureConfig` 只接受五个冻结分支，并对不存在的 adaptation、O3E、PGE
  placement 强制使用 `none`；PGE 分支要求 `a1_only | full_pg`。
- 枚举结果恰好为 `2 + 4 + 8 + 4 + 8 = 26`，manifest 与运行时枚举逐项一致且稳定。
- `PeriodicGraph`、`O3FeatureBatch`、`SymmetryRecord`、`ParentEmbeddingSpec`、
  `ParentDAGSpec` 与 `TensorPrediction` 对 shape、dtype、scope 和关键不变量 fail closed。
- 执行本实现单元的 pytest、compile/import 检查和 `git diff --check`；实际结果须在
  提交前补录，任一失败均不提交。

实际结果：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_phase_a_contracts.py -q`：
  15 passed，0 failed，0 skipped。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests assets/model_code/tests -q`：
  27 passed，0 failed，1 skipped；唯一 skip 是旧原型中由
  `RUN_FULL_PG_OUTPUT_TEST=1` 控制的 32-PG/64-forward regression，不属于本次
  Phase A 新测试。它仍是完整 Goal 后续必须消除的验收缺口，未据此宣称完成。
- `python -m compileall -q src tests/test_phase_a_contracts.py`：通过。
- `python -m json.tool src/configs/architecture_variant_manifest.json`：通过。
- `git diff --check`：通过；仅报告 Git 的 LF→CRLF 工作树提示，无 whitespace error。

## 2026-09-11 — vlab jump host for Guqq

计划检查：

- `ssh -G Guqq` 必须解析出目标 `211.86.155.221`、用户 `xmz`、目标 Ed25519
  密钥及 `proxyjump vlab`。
- `ssh -G vlab` 必须解析出 `vlab.ustc.edu.cn`、用户 `ubuntu` 和专用 PEM 密钥。
- `ssh -o BatchMode=yes -o ConnectTimeout=15 vlab true` 必须无需交互认证并以
  状态 0 退出。
- `ssh -o BatchMode=yes -o ConnectTimeout=20 Guqq git pull` 必须经 vlab 跳转、
  无密码提示并在 Guqq 调用规范要求的首个远程操作 `git pull`；只有默认登录目录
  本身是 Git 仓库时，才要求该 Git 命令返回状态 0。
- 修改前备份用户级 SSH config；修改后不改变其他 Host 条目。

实际结果：

- `ssh -G Guqq`：解析为 `xmz@211.86.155.221`、
  `~/.ssh/id_ed25519_codex` 和 `proxyjump vlab`，通过。
- `ssh -G vlab`：解析为 `ubuntu@vlab.ustc.edu.cn` 和
  `C:\Users\asus\.ssh\vlab-vm12818.pem`，通过。
- `ssh -o BatchMode=yes -o ConnectTimeout=15 vlab true`：状态 0，无密码提示。
- `ssh -o BatchMode=yes -o ConnectTimeout=20 Guqq git pull`：成功经 vlab 登录并
  在 Guqq 执行远程命令；因默认登录目录不是 Git 仓库而返回状态 1。该结果不属于
  SSH 失败，但表明后续拉取前必须先进入项目仓库目录。
- 验收边界修订依据：用户目标是 SSH 免密登录，项目规范要求连接后把 `git pull`
  作为首个远程操作，但未规定项目仓库位于默认登录目录；远端返回 Git 自身的
  `not a git repository` 已证明 SSH 认证、跳转和远程命令执行均已完成。
- `ssh -o BatchMode=yes -o ConnectTimeout=20 Guqq true`：状态 0，无密码提示，
  端到端免密验收通过。
- 原配置已备份到 `C:\Users\asus\.ssh\config.bak-vlab-20260911`；配置差异检查
  仅应包含 `Host Guqq` 下新增的 `ProxyJump vlab`。

## 2026-09-11 — GOAL and subgroup-chain artifacts

计划检查：

- `assets/docs/subgroup_chain.json` 可解析，包含且仅包含 32 个 crystallographic point groups；
- cover edge、oriented subgroup instance 与 maximal-chain 总数和权威生成源一致；每条 chain 的相邻节点都是合法 cover edge并终止于 `1`；
- `GOAL.md` 明确五分支、两种 PG mode、两种 O(3) TP backend、四类 backbone、三类性质/四个独立训练单元及 `<5M` 验收；
- GOAL 明确 32 点群 forward/backward smoke、每训练单元 5 结构 train/test smoke、O(3)/PG/BEC 等变测试和全部单元测试；
- GOAL 明确 Guqq 上所有计算任务只能通过 Slurm、当前未完成资源和禁止以 skip/dummy 通过验收；
- Markdown、JSON、路径与 diff whitespace 检查通过。

实际结果：

- 生成器命令成功，`assets/docs/subgroup_chain.json` 可解析：schema 1、32 point groups、80 class cover edges、433 oriented subgroup instances、222 maximal chain class sequences；
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest assets/model_code/tests/test_subgroup_chain_asset.py -q`：1 passed；包含 subgroup matrix closure、maximal counts 和 chain-adjacency 验证；
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest assets/model_code/tests -q`：6 passed、1 skipped；skip 是当前旧的 32-PG/64-forward opt-in regression，必须在未来 Guqq Goal 验收中启用且最终 mandatory skips 为 0；
- 首次 pytest 在 collection 前因全局 LangSmith 插件缺少 `requests_toolbelt` 失败；禁用无关第三方 plugin autoload 后测试正常执行，未降低测试范围；
- GOAL 中 5 个本地链接全部存在；Markdown code fences 为 6，配对；
- `git diff --check`：通过，仅有工作树 LF→CRLF 提示；
- 本轮未连接 Guqq、未运行计算型 smoke，也未声称完整模型已经实现。

## 2026-09-11 — Training/data/smoke protocol clarification

计划与结果：

- 独立训练单元明确为 dataset × target property，JARVIS dielectric/elastic 不共享 downstream training：通过；
- graph policy 明确为优先复用 backbone、缺失时 6 Å fallback：通过；
- split policy 明确为优先官方 split、缺失时持久化 8:1:1：通过；
- smoke protocol 明确区分 32 点群各一结构 forward/backward 与每训练单元 5 结构 train/test pipeline：通过；
- parity audit 明确 dielectric/elastic 无额外 irreps、BEC 额外 `1x1e`：通过；
- Markdown code fences 30、display-math fences 602，均配对；`git diff --check -- proposal.md docs/agents/{state,test,update}.md` 通过，仅有工作树 LF→CRLF 提示；
- 本轮仅修改文档，未运行模型测试。

## 2026-09-11 — BEC permutation/projector boundary clarification

计划与结果：

- 检查默认 BEC forward 明确不接收 \(\pi_g\)：通过；
- 检查联合 Reynolds projector 仅为 diagnostic / 非默认 hard control：通过；
- 检查 ASR 被声明为不依赖 \(\pi_g\) 的独立约束：通过；
- `git diff --check -- proposal.md docs/agents/{state,test,update}.md`：通过，仅有工作树 LF→CRLF 提示；
- 本轮仅修改文档，未运行模型测试。

## 2026-09-11 — Five-branch architecture revision

计划检查：

- proposal 顶层架构只保留五个用户指定分支；
- PG expert 均开放 `a1_only | full_pg`；
- O(3) adaptation、O(3) expert 与 O(3) readout 的 TP backend 均开放 `full_o3 | o2_tp`；
- 旧 architecture 名称、MVP 与实验矩阵不存在冲突；
- Markdown diff、公式/代码围栏与相关本地路径检查通过。

实际结果：

- `git diff --check -- proposal.md docs/agents/{state,test,update}.md`：通过，仅有工作树 LF→CRLF 提示；
- Markdown code fences：30，偶数；display-math fences：598，偶数；
- 旧顶层 architecture 名称、`N_s`/shared-experts 与 identity-readout ablation 冲突检索：0 项；FAQ 中仅保留 Output Projection 作为概念性外部 baseline 讨论；
- `docs/ref/o2 tp.pdf`、点群 DAG JSON 与逐点群参数报告路径：均存在；
- 本轮仅修改文档，未运行模型测试。

## 2026-09-11 — Proposal BEC/PG-mode revision

计划检查：

- `proposal.md` 无 whitespace/diff 格式错误，公式与代码围栏配对；
- 相关本地链接存在；
- 不再残留 BEC 为 future work/global-only 或 Full-PG 仅作 ablation 的冲突范围声明；
- pooling scope、BEC input/output irreps、联合等变性与两种 PG hidden mode 在摘要、方法、benchmark、phase、FAQ 和总结中一致。

实际结果：

- `git diff --check -- proposal.md docs/agents/{state,update,test}.md`：通过，仅有既有 LF→CRLF 提示；
- Markdown code fences：26，偶数；display-math fences：622，偶数；
- BEC future-work/global-only、Full-PG-only-ablation、`full_to_a1` 冲突检索：0 项；
- `docs/ref/o2 tp.pdf`、点群 DAG JSON 与参数报告本地路径：均存在。

## 2026-09-11 — Backbone/BEC resource preparation

计划检查：

- 下载文件大小和发布方 checksum（若提供）一致；另记录 SHA-256。
- checkpoint 可由其官方包或安全的 metadata-only 方法识别，不执行大规模推理。
- 每个 backbone 记录具体模型版本、feature-tap 候选、连续群/parity 约定和元素覆盖。
- BEC 数据逐样本具有结构、与原子数一致的 N x 3 x 3 tensor、稳定样本 ID；统计
  缺失值、形状错误和 split 情况。
- raw 数据、checkpoint 和克隆的上游仓库均被 Git 忽略。

实际结果：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_prepare_jarvis_bec.py -q`：
  2 passed。
- 10 个真实 JARVIS-DFPT archive 下载、MD5 校验和 XML 提取：10/10 成功，
  0 个提取错误；原子数范围 3–27，结构站点数与 BEC 第一维全部一致。
- 10 样本 summary validator：`valid=true`、无重复 ID、无非有限值或形状错误。
- JVASP-29884 单样本：14 个原子，BEC 形状 14 x 3 x 3，ASR Frobenius
  residual 为 3.1623e-8 e。
- GRACE 文件 85,894,694 bytes，SHA-256
  `19a006ab5668e6aa3426de6a78ee4a9a9da373680ec9a6171787b0259a67bc20`，
  与官方 LFS checksum 一致。
- DPA4-Plus 文件 35,550,605 bytes，SHA-256
  `6820a320dc241a4002e6c257ad21983950df178b1f80271a6e6af199adb18567`；
  配置 JSON 语法和 `lmax=4, mmax=1` 已核验。
- JARVIS raw index MD5
  `0bbf6fb41120357b3ff8b19eb3690b8b`，与 Figshare 元数据一致。
- 两个 manifest 均通过 `python -m json.tool`；新 Python 文件通过
  `python -m compileall -q`。
- EquiformerV2 下载返回 HTTP 401/GatedRepo；本机 `hf auth whoami` 为
  `Not logged in`，故未伪造完成状态。
- Guqq 连通性诊断：TCP 22 成功；`ssh -G Guqq` 在批准环境中解析为用户 `xmz`
  和指定 Ed25519 identity；`ssh -vv Guqq git pull` 在认证前返回
  `kex_exchange_identification: Connection closed by remote host`，因此没有远端
  命令得到执行，也没有绕过 Slurm 在登录节点运行批处理。
- 2026-09-11 用户要求复测：`ssh -vvv -o BatchMode=yes -o ConnectTimeout=15
  Guqq git pull` 再次确认 TCP 建连成功，但远端在收到客户端版本字符串后、发送
  SSH banner/进入 key exchange 前关闭连接；`ssh-keyscan -T 10` 同样未收到
  banner 或 host key。`ssh -G Guqq` 仍解析为 `xmz@211.86.155.221:22` 和预期
  Ed25519 identity，因此排除仓库路径、远程 Git 命令和客户端密钥认证阶段。
