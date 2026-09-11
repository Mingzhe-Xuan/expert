# Test plan and results

## 2026-09-11 — Five-branch dispatcher and three tensor readouts

计划检查：

- 冻结 manifest 的 26 configs 全部可构造，且 `adaptation/o3e/pge` 缺失 placement
  为 `None`、不实例化空模块；O3E/PGE 互斥，final readout 恰有一个 backend。
- mixed-size `O3FeatureBatch + PeriodicGraph + SymmetryRecord` 对 dielectric/elastic
  输出 `[B,3,3]`/`[B,3,3,3,3]`，先 permutation-invariant pooling 再 PG fixed-space。
- BEC 输出 `[N,3,3]`、保留 node/site order，不 global pooling；默认 forward 不接受
  `pi_g`/不执行 joint projector，同时返回 raw 与 independent ASR tensor。
- 五分支按冻结顺序执行 A、O3E 或 A1/Full-PGE、公共 O(3) fusion、恰好一个 readout；
  active parent 只来自 validated `ParentDAGSpec`，Hall/PG expert 参数去重。
- 26 configs 的 forward、target coefficient loss、backward、active non-backbone `<5M`
  与 checkpoint round-trip 全覆盖；mixed batch/empty edge/float32 另测。
- 三类 final Cartesian outputs 做 proper/improper O(3) 与 PG forbidden-component tests；
  运行完整 pytest、compile/diff，提交前补录结果。

实际结果：

- 首轮测试捕获 float32 下 PG invariant basis 以 `1e-9` 判秩时将数值残差误收为
  forbidden basis 的缺陷；现固定以 float64 决定 fixed space 后转换至模型 dtype，
  未降低约束断言。
- `$env:PYTHONPATH='.'; uv run pytest tests/test_dispatcher_readout.py -q
  -p no:cacheprovider --basetemp=.test-tmp/dispatcher`：4 passed；覆盖三 head scope/shape、
  mixed batch/empty edge、PG fixed space、BEC raw+ASR/no-`pi_g`、proper/improper O(3)、
  父 Hall fail-closed，以及全部 26 configs 的 forward/loss/backward/checkpoint/`<5M`。
- `$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; uv run python -m pytest
  tests assets/model_code/tests -q -p no:cacheprovider --basetemp=.test-tmp/full`：
  107 passed，1 个既有 opt-in skip，0 failed。
- `uv run python -m compileall -q src tests/test_dispatcher_readout.py` 与
  `git diff --check`：通过；后者仅报告 Windows LF→CRLF 提示。

## 2026-09-11 — Adaptation, O3E, dual-mode PGE, routing and fusion

计划检查：

- shared adaptation 与 routed O3E 的每个实际 TP placement 分别构造 `full_o3`、
  `o2_tp`，使用完整 PBC directed edges，支持 empty-edge self fallback、梯度和随机
  proper/improper O(3) equivariance。
- 每个 PG expert 恰有两个参数对象不共享的 blocks；C1 bypass 仍保持相同 I/O contract。
- `a1_only` 只保留逐 O(3) source/copy 标记的 invariant coordinates；`full_pg` 使用
  正交 subduction，保留全部 non-trivial blocks，并对 32 群逐 operation 等变。
- continuous residual gates 权重非负、归一、可微，current/compatible parents active
  set 去重；parent residual 极限和 duplicate paths 不重复计参数/feature。
- hierarchical fusion 只接受公共 O(3) layout，按归一权重融合；双模式 backward
  到达两个 blocks 和所有应训练 routing/fusion 参数，active parameter 初步审计 `<5M`。
- 运行目标 pytest、完整 pytest、compile/diff 检查，提交前补录结果。

实际结果：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_expert_modules.py -q`：
  16 passed；adaptation/O3E 双 backend×proper/improper、empty fallback、32 群双 PGE
  mode、两 block 参数独立、C1 bypass、Full-PG nontrivial carrier、continuous gates、
  fusion 与 active `<5M` 初审全部通过。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests assets/model_code/tests -q`：
  103 passed，0 failed，1 个既有 opt-in skip。
- `python -m compileall -q src tests/test_expert_modules.py` 与 `git diff --check`：
  通过；后者仅有 LF→CRLF 提示。

## 2026-09-11 — Full-O(3) and complete local-O(2) tensor products

计划检查：

- `full_o3` 与 `o2_tp` 使用同一 `IrrepLayout`/batch contract，输出 shape 正确、参数
  梯度及两输入梯度有限；无允许 parity path 时显式输出零宽/零值而非伪路径。
- `full_o3` 对随机 proper/improper O(3) 满足 equivariance，覆盖 `0e/0o`、polar 与
  pseudo carriers。
- `o2_tp` 按每个 O(3) degree/copy triple 枚举完整 local O(2) Hom-space paths，而非
  full-O(3) TP 加 projector；默认 `mmax=2`，支持 full-m control。
- `o2_tp` 对随机 proper/improper global transforms 等变，并对 local-frame SO(2)
  gauge rotation 与 O(2) reflection gauge 不变；edge-axis reversal cases 单独覆盖。
- path/copy ordering 与 fixed coupling checksum 稳定，parameter count 可审计；运行
  目标 pytest、完整 pytest、compile/diff 检查，提交前补录结果。

实际结果：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_tensor_product_backends.py -q`：
  12 passed；双 backend shape/grad/parameter audit、proper/improper O(3)、local SO(2)
  rotation/reflection gauges、mmax/full-m、forbidden parity、polar/pseudo edge reversal
  与 fail-closed inputs 全部通过。
- full-m vector×vector→scalar 枚举两个独立 local-O(2) paths，`mmax=0` 仅一个，证明
  实现没有退化为 full-O(3) scalar dot-product path。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests assets/model_code/tests -q`：
  87 passed，0 failed，1 个既有 opt-in skip。
- `python -m compileall -q src tests/test_tensor_product_backends.py` 与
  `git diff --check`：通过；后者仅有 LF→CRLF 提示。

## 2026-09-11 — Full-PG subduction and finite-group CG registry

计划检查：

- 对 32 点群、`l=0..4` natural carriers，逐 O(3) labelled copy 构造确定性实有限群
  irreducible subspaces；subduction matrix 正交、copy/path order 稳定且 checksum 可复现。
- subduction→inverse 对 float64 batch round-trip；每个 PG block 在全部群操作下闭合，
  block-off-diagonal leakage 低于预注册 module tolerance。
- A1 invariant copies 与 Full-PG copies 均保留 O(3) source label/copy provenance；
  Full-PG 在非平凡群/`l>0` 中存在非 A1 paths，不能退化为旧 A1-only 原型。
- finite-group CG/intertwiner basis 由 Hom-space Reynolds projector 构造，path basis
  正交确定，逐 operation 满足 `K^T W = W C^T`；无允许 path 时返回显式零维 basis。
- subduction、CG、path/copy ordering checksum 可组装进 `ConventionMetadata`；运行
  目标 pytest、完整 pytest、compile/diff 检查，提交前补录结果。

实际结果：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_subduction_cg.py -q`：
  6 passed；32 群 `l=0..4` round-trip/逐操作 equivariance/checksum、cubic `l=2`
  `2+3` 非平凡分解、repeated provenance、32 群 vector×vector→scalar intertwiners、
  forbidden zero-path 与 convention checksum 均通过。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests assets/model_code/tests -q`：
  75 passed，0 failed，1 个既有 opt-in skip。
- `python -m compileall -q src tests/test_subduction_cg.py` 与 `git diff --check`：
  通过；后者仅有 LF→CRLF 提示。

## 2026-09-11 — Hall-level physical parent embedding validation

计划检查：

- `ParentEmbeddingSpec` 强制非空 parent/child settings、可逆 basis/supercell transforms、
  完整 affine `(W,t)` operation group 的 identity/uniqueness/closure/inverse。
- child→parent atom correspondence 覆盖全部 child sites、索引有效并逐 site 保持
  species；允许 Wyckoff splitting 的多 child→单 parent，不误要求 bijection。
- checksum 覆盖 Hall/settings/transforms/operations/species/correspondence/Wyckoff/domain
  全字段，任何篡改 fail closed。
- `ParentDAGSpec` 允许同一 Hall edge 的不同 orientation variants，但拒绝完全重复
  embedding、cycle、断开的 current Hall 与未验证 edge。
- 运行目标 pytest、完整本地 pytest、compile 和 diff 检查，提交前补录结果。

实际结果：

- 与 canonicalization/既有 Phase A contracts 联合执行：
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_canonicalization.py
  tests/test_parent_embeddings.py tests/test_phase_a_contracts.py -q`：25 passed。
- affine identity/unique/inverse/closure、可逆 transforms、species-preserving 多对一
  mapping、checksum tamper、orientation variants、duplicate/cycle/disconnected DAG
  均有直接通过/拒绝断言。

## 2026-09-11 — Deterministic spglib canonicalization

计划检查：

- 使用 spglib 2.6.0 默认最小 Hall setting 检测后，以检测到的 Hall number 再次显式
  查询并冻结 setting；重复调用和 atom-order permutation 得到相同 SG/PG/Hall 元数据。
- `std_rotation_matrix` 经正交审计后作为 input→canonical active rotation；positions、
  cell 与 rank-2/rank-4 tensor 的 canonical→input round-trip 在 float64 tolerance 内。
- 保留原始 site order；从完整 `(W,t)` operations 生成 species-preserving audit-only
  permutations，并验证每项为 bijection 及周期坐标映射闭合。
- fractional operations 转换到 canonical Cartesian frame 后正交、与 registry PG order
  一致；无效 cell、spglib failure 或不闭合 site mapping fail closed。
- 运行目标 pytest、完整本地 pytest、compile 和 diff 检查，提交前补录结果。

实际结果：

- 首轮目标测试捕获 operation polar repair 将 improper matrices 强制为 det=+1，导致
  `m-3m` 从 48 个操作坍缩成 24 个；现已仅对 canonical frame 强制 proper，群操作
  保留原 determinant，原断言随后 6/6 通过。
- 与 parent/contracts 合并目标测试 25 passed；完整
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests assets/model_code/tests -q`：
  69 passed，0 failed，1 个既有 opt-in skip。
- `python -m compileall -q src tests/test_canonicalization.py
  tests/test_parent_embeddings.py` 与 `git diff --check`：通过；后者仅有 LF→CRLF 提示。

## 2026-09-11 — Copy-aware loss, metrics, and checkpoint round-trip

计划检查：

- dielectric/elastic/BEC coefficient MSE 分别按冻结 layout 的每个 labelled copy
  计算；BEC 保留 atom item 维度而不做跨原子 target pooling，total 可反向传播。
- normalizer scale 产生 train-stat weighting；raw physical-unit per-copy MAE/RMSE
  可序列化且 repeated copies 不合并。
- 最小可训练模块完成 optimizer update，checkpoint 原子保存/严格加载后输出一致，
  optimizer step、normalizer、architecture、training-unit 与 convention metadata 恢复。
- architecture/unit/layout/convention 任一不匹配均在 state mutation 前 fail closed；
  使用安全 `weights_only=True` 加载普通 tensor/state 数据。
- 运行目标 pytest、完整本地 pytest、compile 和 diff 检查，提交前补录结果。

实际结果：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_training_checkpoint.py -q`：
  8 passed；三 target copy-aware loss/gradient/metrics、normalizer weighting、Adam
  update、原子 checkpoint round-trip，以及 architecture/unit/convention mismatch 在
  model mutation 前拒绝均通过。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests assets/model_code/tests -q`：
  59 passed，0 failed，1 个既有 opt-in skip。
- `python -m compileall -q src tests/test_training_checkpoint.py` 与
  `git diff --check`：通过；后者仅有 LF→CRLF 提示。

## 2026-09-11 — Independent training units, splits, and normalization

计划检查：

- 只允许 JARVIS dielectric、JARVIS elastic、MatTen elastic、JARVIS-DFPT BEC 四个
  `dataset × property` 配对，配置/normalizer/checkpoint namespace 不可跨单元共享。
- fallback split 使用 seed `20260911`、按 material/duplicate group 划分，结果确定、
  全覆盖且 train/validation/test 无样本或 group 泄漏；5 groups 产生 3/1/1。
- coefficient normalizer 仅允许 `split=train` 拟合，按 target layout 的显式 repeated
  copy 分别保存 scale；支持 RMS/variance、有限小样本 fallback 和物理单位 inverse。
- normalizer state 含 layout/copy order 与训练单元 identity，不兼容加载 fail closed。
- 运行目标 pytest、完整本地 pytest、compile 和 diff 检查，提交前补录结果。

实际结果：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_training_units.py -q`：
  6 passed；四合法 pairing/非法 pairing、确定性 duplicate-group split、5-group
  3/1/1、RMS/variance copy-aware inverse、非 train fitting 拒绝及跨单元/layout
  state 拒绝均通过。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests assets/model_code/tests -q`：
  51 passed，0 failed，1 个既有 opt-in skip。
- `python -m compileall -q src tests/test_training_units.py` 与 `git diff --check`：
  通过；后者仅有 LF→CRLF 提示。

## 2026-09-11 — Complete cutoff PBC graph construction

计划检查：

- 单原子 cubic cell 在跨边界 cutoff 下枚举完整六邻居等距 shell，并保留同一原子对
  的不同 image multiedges、整数 shifts 与双向 edges；严格 cutoff boundary 不纳入。
- skew/non-reduced cell 的自适应 image bounds 与较大 brute-force enumeration 一致，
  不使用 `max_neighbors` 截断。
- mixed-size graph collation 正确偏移 node/graph indices 并保持 edge geometry；空边、
  退化小 cutoff、float32/float64 与 CPU/GPU `.to()` contract 可用。
- 对 cell/positions 同时施加 proper/improper O(3) 变换时，edge topology/shifts 不变，
  vectors 协变、distances 不变。
- 运行目标 pytest、完整本地 pytest、compile 和 diff 检查，提交前补录结果。

实际结果：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_periodic_graph_builder.py -q`：
  6 passed；cubic 六邻居 shell、严格 boundary、双站点 image multiedges、skew cell
  brute-force 对照、mixed collation/dtype transfer 与 proper/improper covariance 全绿。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests assets/model_code/tests -q`：
  45 passed，0 failed，1 个既有 opt-in skip。
- `python -m compileall -q src tests/test_periodic_graph_builder.py` 与
  `git diff --check`：通过；后者仅有 LF→CRLF 提示。

## 2026-09-11 — Target Cartesian transforms and BEC controls

计划检查：

- dielectric、elastic、BEC 的 Cartesian↔irrep round-trip 在 float64 下保持目标
  tensor symmetry，且 coefficient width 与冻结 layout 一致。
- proper/improper O(3) 旋转前后的 coefficient 表示和 Cartesian 变换一致。
- point-group fixed-space coefficient projector 对 32 群均幂等，投影结果满足所有
  群操作；global heads 可使用它而 BEC raw head 不隐式调用它。
- BEC ASR 按 crystal 独立去除逐分量原子和，不依赖 `pi_g`；optional joint
  permutation/tensor projector 满足 `Z[pi_g(i)]=R_g Z[i]R_g^T`，并验证与 ASR 交换。
- 运行目标 pytest、完整本地 pytest、compile 和 diff 检查，提交前补录结果。

实际结果：

- 首轮 round-trip 的三项参数化 case 均显示约 `0.9–1.9e-7` 最大误差；诊断证明
  e3nn 0.5 的符号 change-of-basis 经 default-float 中间量生成，事后转换 float64
  仍保留该误差。实现改为对冻结 basis 做 float64 极分解正交化，并从同一 basis
  诱导 target representation；未放宽原测试 tolerance。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_target_transforms.py -q`：
  8 passed，覆盖三 target round-trip 与 proper/improper covariance、tensor intrinsic
  symmetries、32 群 global projector、BEC ASR 梯度、joint projector/ASR 交换和
  frame/site-order 恢复。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests assets/model_code/tests -q`：
  39 passed，0 failed，1 个既有 opt-in skip。
- `python -m compileall -q src tests/test_target_transforms.py` 与
  `git diff --check`：通过；后者仅报告 LF→CRLF 提示。

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
