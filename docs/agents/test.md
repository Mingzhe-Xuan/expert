# Test plan and results

## 2026-09-12 — Efficiency report aggregation

计划检查：

- 成功读取 point-group 与 real-subset smoke JSON 中的 `efficiency` record，并按来源稳定排序；
- 严格拒绝非 passed summary、缺少必需字段、非法数值、未知 FLOPs scope 与重复实验身份；
- JSON 报告保留完整机器字段，Markdown 表明确标注 active downstream FLOPs、单次真实
  end-to-end latency 与 CUDA peak memory，并覆盖 architecture/task/PG/mode/backend keys；
- CLI 同时原子写入 JSON/Markdown，失败返回非零且不发布部分输出；
- 运行 targeted、完整 pytest、compile、CLI help 和 `git diff --check`。

实际结果：

- targeted：14 passed、68 warnings；完整项目 suite：178 passed、0 failed、0 skipped、
  496 warnings，229.84s；
- 精确覆盖、稳定排序、Markdown scope/columns，以及 failed/missing/budget/scope/mean/CUDA
  和重复 index 拒绝测试全部通过；
- `python -m src.cli.efficiency_report --help`、`python -m compileall -q src tests` 与
  `git diff --check` 通过；`uv run ruff` 因当前本地 uv 环境未安装 ruff 而未执行；
- 一次未限定路径的 `pytest -q` 在收集用户下载的 `data/sources/matten` 上游测试时因本机
  `torch_spline_conv` DLL 不兼容退出；该目录不属于项目 `tests/` 验收范围，随后显式完整
  `pytest -q tests` 全绿，未降低既定项目测试范围。

## 2026-09-12 — Runtime efficiency evidence

计划检查：

- profiler 必须报告 total/trainable/active non-backbone parameters、逐样本 active expert
  count（mean/max）、真实 end-to-end forward latency、CUDA peak allocated bytes；
- active FLOPs 用缓存 backbone feature 上的实际 downstream forward，经 Torch profiler
  统计 dispatched operators，并明确 scope，避免把 TensorFlow GRACE 未观测部分冒充总 FLOPs；
- profiling 保持并恢复 model train/eval 状态，CUDA 测量前后同步，非 CUDA 明确将 peak 记为 null；
- 58-row PG smoke 与 20-row real-subset smoke 均嵌入带 architecture/task/PG/mode/backend keys
  的 efficiency record；
- 添加 CPU linear reference、active-expert counts 与 smoke integration 测试，运行 targeted、
  完整 pytest、compile 和 `git diff --check`。

实际结果：

- efficiency record 包含 architecture/task/PG/mode/三个 backend keys，以及 total、Torch
  registered、external frozen、trainable、active non-backbone parameters；
- active expert counts 按样本去重并报告 list/mean/max；expert-free 为 0，current-only routed 为 1；
- end-to-end latency 为同步单次真实 forward；CUDA incremental peak 在 CUDA 上测量、CPU 为 null；
- active downstream FLOPs 由缓存 backbone feature 上的 Torch profiler 实测，并以
  `torch_dispatched_active_downstream_forward` 明确 scope；GRACE TensorFlow 参数独立计数；
- 58-row point-group 与 20-row real-subset JSON 均嵌入 efficiency record；
- targeted：16 passed；完整本地 suite：170 passed、0 failed、0 skipped，496 warnings，219.24s；
- `python -m compileall -q src tests` 与 `git diff --check` 通过。

## 2026-09-12 — Non-Equiformer resource atomic promotion

计划检查：十个非 Eq contract 文件必须先在隔离 staging 重组，逐个匹配 committed manifest
的固定 byte count 与 SHA-256；只有全部通过后才允许原子提升，任一失败不得修改最终路径。

实际结果：本地十文件 size/SHA 复算完成；首轮远端验证因缩略记录中的 dataset SHA 转抄错误
按预期立即非零退出，十个 `mv` 均未执行。改用 committed manifests 原值后，Guqq 十文件
的固定大小和完整 SHA-256 全部通过，随后一次性原子提升；`git diff --check` 通过。

## 2026-09-12 — Guqq runtime smoke failure fixes

计划检查：

- float32 非恒等 canonical frame 与其转置在 dtype-aware 数值阈值内通过，明显非逆矩阵仍失败；
- DPA4 neighbor schema 的 index/vector/mask 在进入 descriptor 和几何运算前统一到模型设备；
- GRACE launcher 显式选择 TensorFlow CPU fallback，adapter 在任何 TensorFlow op 前应用，
  同时保持 PyTorch interface/device 输出在请求的 CUDA device，并在结果中披露 backend；
- 运行 canonicalization、backbone contracts、CLI/Slurm launcher targeted tests，随后运行完整
  `pytest -q tests`、compile、shell syntax 与 `git diff --check`，全部通过才提交源码修复。

实际结果：targeted 42 passed、55 warnings；完整项目 182 passed、496 warnings（205.36s）。
`python -m compileall -q src tests`、两个修改后 shell launcher 的 `bash -n` 与
`git diff --check` 全部通过。前两次启动分别因未显式设置 `PYTHONPATH=.` 和受限 uv cache
未进入测试收集；按既有记录改用本地 Python、禁用第三方 plugin/cache 后在原范围全绿。

## 2026-09-12 — Strict Slurm terminal-state audit

计划检查：

- 定义 versioned acceptance job manifest，单作业与 array range 展开为精确 `JobIDRaw` 集合；
- 解析 `sacct -X -n -P` 输出，拒绝缺失/重复期望任务、未知期望字段、非 `COMPLETED`
  state、非 `0:0` exit code；允许并忽略 array parent allocation 行与无关 step 行；
- CLI 调用固定字段的 `sacct`，持久化 raw output 和 machine-readable audit JSON，任何失败返回非零；
- 对成功、数组缺口、失败 state/exit、duplicate 和 CLI subprocess contract 增加测试，再运行
  targeted、完整 pytest、compile 与 `git diff --check`。

实际结果：

- manifest schema 精确展开单作业和闭区间 arrays；未知字段、非法 ID/range fail closed；
- parser 固定六列 allocation-only `sacct` 格式，auditor 拒绝缺失、重复、失败 state/exit；
  array parent 与 `.batch` 等非期望行被忽略且不能填补 task 缺口；
- CLI 使用 `sacct -X -n -P`、一次查询全部基础 job IDs，并分别保存 raw 与 JSON audit；
- targeted：12 passed；完整本地 suite：167 passed、0 failed、0 skipped，484 warnings，214.42s；
- CLI help、`python -m compileall -q src tests` 与 `git diff --check` 通过。

## 2026-09-12 — Required data-job Slurm evidence contract

计划检查：

- 将通用 success/failure evidence runner 扩展到无 device 的数据 job，同时保持 adapter wrapper
  兼容；
- 32-PG fixture builder 在成功与异常路径都写独立 summary/JUnit/execution metadata，sbatch
  另存 Git revision 与 `pip freeze`；
- JARVIS-DFPT BEC preparation 的 prepare + validation sequence 同样必须形成单一 job-level
  summary/JUnit，任何阶段失败均返回非零且不伪装为完成；
- 增加 runner、CLI/launcher contract 测试，运行 targeted、完整 pytest、compile、相关
  sbatch `bash -n` 与 `git diff --check`。

实际结果：

- 32-PG builder 与 BEC prepare/validate 均通过共用 runner 持久化成功或失败 summary/JUnit；
- 两个 sbatch 均保存 Git revision、`pip freeze`，artifact 名包含 Slurm job ID；
- BEC wrapper 拒绝 extraction errors、invalid/duplicate records 与小于完整索引的 record count；
  每次 resume 重建当前 attempt error log，已恢复的暂态错误不再永久污染最终验收；
- targeted：27 passed；完整本地 suite：161 passed、0 failed、0 skipped，484 warnings，216.02s；
- `compileall src tests`、两个 data 脚本 `py_compile`、两个 sbatch `bash -n`、CLI help 与
  `git diff --check` 通过。

诊断记录：一次过宽的 `compileall src data tests` 扫入 ignored `data/vendor/fairchem` 上游源码，
命中其既存 future-import 问题；随后对本单元两个 data 脚本做精确 `py_compile` 并通过，未修改 vendor。

## 2026-09-12 — Standalone adapter Slurm evidence contract

计划检查：

- 共用 CLI runner 在成功与异常路径都写 machine-readable JSON、单 case JUnit 和 execution
  metadata；异常路径保留原异常并产生非零退出；
- 四个 adapter CLI 均要求 `--junit`，四个 sbatch 均保存 Git commit、完整 environment
  fingerprint，并把 job/task ID 编入 artifact 名；
- DPA4 smoke 必须显式导入并能读取其 source layout，避免真实 checkpoint 成功后才触发
  `NameError`；
- 对共用 runner 的成功/失败行为和四个 launcher contract 增加单元测试，随后运行 targeted、
  完整 pytest、compile、四个 sbatch `bash -n` 与 `git diff --check`。

实际结果：

- 共用 runner 成功路径写入 metadata/JSON/JUnit，失败路径先写 failure JSON/JUnit 再原样抛出；
- 四个 standalone sbatch 均包含 Git revision、`pip freeze`、JSON、JUnit 与 Slurm-ID artifact；
- DPA4 smoke 现显式导入 `DPA4_SO3_LAYOUT`，回归断言 dimension 1600；
- targeted：17 passed；完整本地 suite：157 passed、0 failed、0 skipped，484 warnings，214.34s；
- `python -m compileall -q src tests`、四个 sbatch `bash -n` 与 `git diff --check` 均通过。

诊断记录：前两次测试启动分别因未设置 `PYTHONPATH=.` 和受限 uv cache 失败，随后使用既定
本地 Python、显式 `PYTHONPATH=.`、禁用第三方 pytest plugin autoload/cache 后通过；测试范围未降低。

## 2026-09-12 — Guqq benchmark/checkpoint resource synchronization

计划检查：

- 上传前复核三份 benchmark 数据、MACE checkpoint、GRACE archive/三个 manifest loader
  artifacts 与 DPA4 checkpoint/config 共 10 个 contract 对象的本地 size 和 SHA-256；
- 单一 SSH 会话必须先在 `/home/xmz/expert` 执行 scoped HTTP/1.1 fast-forward pull，再解包
  到仓库内 Git-ignored 原路径；
- 上传后逐项核对远端 size/SHA-256，并确认工作树没有因资源同步产生 tracked source 修改；
- 本步骤仅做网络传输和文件校验，不直接在登录节点运行任何模型计算或批量数据处理。

调整说明：首个整包 tar stream 在 pull 后被跳板重置；改用一次先 pull 的 multiplex transport
逐文件原子复制。额外 `gmm_artifacts.npz` 未被 manifest 或 adapter loader contract 引用，故不上传。

实际结果：未通过，且未降低验收标准。第一次在 pull 成功后传输被重置；第二次 mandatory
HTTP/1.1 pull 以 GnuTLS `-110` 结束；第三次连接 GitHub 443 在 133932 ms 后超时。
三次均未完成远端 10-file size/SHA-256 验证，因此不将任何可能的半文件计为已同步。

## 2026-09-12 — Reproducible Guqq EquiformerV2 environment lock

计划检查：

- 将修复后 clean `pip check` 的 Equiformer snapshot 固化为独立 exact-pin lock；
- 通用 parser 断言无重复、全部 requirements 精确 `==`，并包含官方 cu121 index；
- 关键版本必须为 fairchem-core 1.10.0、Torch 2.4.1+cu121、e3nn 0.5.9、Hydra 1.3.2、
  OmegaConf 2.3.0、NumPy 1.26.4、SciPy 1.15.3 与 spglib 2.6.0；
- 明确禁止旧 `hydra-core==0.11.3` / `omegaconf==1.4.1` 回退；执行 targeted、完整 pytest、
  compile 与 `git diff --check`。

实际结果：

- `requirements/guqq/equiformerv2.txt` 固化 119 个精确 pin，并保留官方 cu121 index；
- targeted environment-lock suite：9 passed；完整本地 suite：153 passed、0 failed、0 skipped，
  484 warnings，220.63s；
- `python -m compileall -q src tests` 与 `git diff --check` 均通过。

## 2026-09-12 — Reproducible Guqq DPA4 environment lock

计划检查：

- 将 Guqq clean `pip check` 后的 DPA4 snapshot 固化为独立 exact-pin lock；
- 通用 parser 断言无重复、全部 requirements 精确 `==`，并包含官方 cu128 index；
- 关键版本必须为 DeepMD-kit 3.2.0、Torch 2.11.0+cu128、CUDA toolkit 12.8.1、
  e3nn 0.5.9、NumPy 1.26.4、SciPy 1.15.3 与 spglib 2.6.0；
- 执行 targeted pytest、完整本地 pytest、compile 与 `git diff --check`。

实际结果：

- `dpa4.txt` 固化 80 个唯一 exact pins、官方 cu128 index 与全部关键版本；
- targeted：7 passed；完整本地 `tests/`：151 passed，0 failed，0 skipped（229.92 s）；
- `python -m compileall -q src tests` 与 `git diff --check` 通过。

## 2026-09-12 — Reproducible Guqq GRACE environment lock

计划检查：

- 将 Guqq clean `pip check` 后的 GRACE package snapshot 固化为独立 exact-pin lock；
- 复用通用 lock parser，断言无重复、全部有效 requirements 精确 `==`，并包含官方 cu128 index；
- 关键版本必须为 TensorPotential 0.6.0、TensorFlow 2.20.0、Torch 2.11.0+cu128、
  CUDA toolkit 12.8.1、e3nn 0.5.9、NumPy 1.26.4、SciPy 1.15.3 与 spglib 2.6.0；
- 执行 targeted pytest、完整本地 pytest、compile 与 `git diff --check`。

实际结果：

- `grace.txt` 固化 95 个唯一 exact pins、官方 cu128 index 与全部关键版本；
- targeted：5 passed；完整本地 `tests/`：149 passed，0 failed，0 skipped（222.33 s）；
- `python -m compileall -q src tests` 与 `git diff --check` 通过。

## 2026-09-12 — Reproducible Guqq MACE environment lock

计划检查：

- 将 Guqq 已通过 `pip check` 的完整 MACE/core `pip freeze` 固化为独立 requirements lock；
- lock 中每个有效 requirement 必须使用精确 `==` pin，包名不得重复，并显式包含 PyTorch
  官方 cu128 index；
- 静态测试必须断言关键验收版本 Torch 2.11.0+cu128、CUDA toolkit 12.8.1、e3nn 0.4.4、
  MACE 0.3.16、spglib 2.6.0、ASE 3.26.0、NumPy 1.26.4、SciPy 1.15.3 与 pytest 8.4.2；
- 执行 targeted pytest、完整本地 pytest、compile 与 `git diff --check`。

实际结果：

- `mace-core.txt` 固化 Guqq clean `pip freeze` 的 71 个唯一精确 pins，并包含官方 cu128 index；
- targeted：3 passed；完整本地 `tests/`：147 passed，0 failed，0 skipped（182.87 s）；
- `python -m compileall -q src tests` 与 `git diff --check` 通过。首次完整命令使用 pytest
  console entry point 时因提升权限会话的非 ASCII cwd 编码丢失 `src` 路径；改用项目既有
  `uv run python -m pytest` 后正常收集并全绿，不属于代码失败。

## 2026-09-12 — Per-backbone Guqq environment dispatch

计划检查：

- 根据官方 `Requires-Dist` 冻结不可合并约束：MACE `e3nn==0.4.4`，Equiformer/fairchem
  `e3nn>=0.5` + Torch 2.4，DPA4/DeepMD `e3nn>=0.5.9` + Torch 2.11，GRACE TensorFlow；
- 两个 mixed-backbone array launchers 必须按 frozen schedule 的 `index % 4` 映射
  `mace|grace|dpa4|equiformerv2`，分别要求 `EXPERT_*_VENV`，缺变量非零失败；
- `test_all` 与 fixture builder 明确使用 MACE/core venv；四个 standalone adapter smoke
  各自只接受对应变量，禁止以一个无法解析的通用 `EXPERT_VENV` 冒充全部 runtime；
- 静态测试解析 schedule 并核对 58/20 rows 的 shell selector 映射、四变量完整性、无旧通用变量，
  另执行 sbatch `bash -n`、完整 pytest、compile 与 diff checks。

实际结果：

- 官方 PyPI JSON metadata 确认四组冻结约束不可合并；未使用 `--no-deps` 或放宽版本绕过；
- 58-row PG 与 20-row real-data schedules 每行均满足 `backbone == order[index % 4]`；
  shell selector 对 0..7 映射正确，缺 DPA4 变量时非零失败并指名缺失变量；
- 所有 mixed/standalone/core/BEC launchers 均无旧 `EXPERT_VENV`，且使用预期专用变量；
- targeted：13 passed；完整本地 `tests/`：144 passed，0 failed，0 skipped（204.15 s）；
  全部 sbatch/selector `bash -n`、compile 与 `git diff --check` 通过。

## 2026-09-12 — Material-operation-aware global tensor projection

计划检查：

- 新增只接受 dielectric/elastic 的 Reynolds projector，直接由 `SymmetryRecord.rotations`
  构造 target coefficient representations；拒绝 BEC、空操作集、错误 shape/device/width；
- 对非代表 orientation（将非平凡 crystallographic group 共轭到另一 Cartesian orientation）
  验证投影结果逐操作不变、幂等且保持有限梯度；明确证明代表 registry projector 与该实际
  orientation 的固定子空间不同，防止测试退化为既有 representative-Hall case；
- `TensorReadout` 必须逐 crystal 使用对应 `SymmetryRecord` operations，mixed batch 不串扰，
  dielectric/elastic 最终 Cartesian intrinsic symmetry 与 O(3) contract 保持不变；
- targeted 与完整 `tests/` 均须 0 failed/skip，另执行 compile 与 diff whitespace 检查。

实际结果：

- 非代表 `mm2` orientation 的 dielectric/elastic projector 验证了逐操作不变、幂等、有限梯度，
  且与 representative-Hall fixed space 明确不同；空 operation、错误 width 与 BEC 均 fail closed；
- mixed-crystal `TensorReadout` 对两个不同 Cartesian orientations 独立投影并完成 backward；
- targeted projector/mixed-batch：12 passed；含全部 26 configs 的 dispatcher suite：15 passed；
- 完整本地 `tests/`：133 passed，0 failed，0 skipped（181.14 s）；
  `python -m compileall -q src tests` 与 `git diff --check` 通过。

## 2026-09-12 — Real-equilibrium 32-point-group fixture pipeline

计划检查：

- 从 checksum-verified GMTNet JARVIS 与 MatTen equilibrium structure records 中选择 fixture，
  禁止沿用 prototype 的 synthetic Wyckoff orbit 作为最终 32-PG acceptance evidence；
- selection 使用冻结 source priority、最少原子数和 stable sample ID 排序，经 spglib explicit
  Hall re-detection 后恰好覆盖 32 crystallographic point groups，每组一条；缺组必须非零失败；
- versioned manifest 为每条记录保存 dataset/source manifest、sample ID、expected PG/SG/Hall、
  species、lattice、fractional coordinates、construction method、spglib version 和 record SHA-256；
- validator 重新计算 record checksum、site shape/species、晶格可逆性与 expected PG/Hall，并拒绝
  重复/缺失/额外点群；fixture 生成属于批量数据处理，仅通过 Guqq Slurm，结果 scp 回本地后提交；
- 使用 small-schema candidates 覆盖 deterministic selection、32-group completeness、tamper
  rejection 和 missing-group failure；这些 synthetic test candidates 不计作最终 equilibrium fixtures。

实际结果：

- targeted：`tests/test_point_group_fixtures.py`、`tests/test_real_data_modules.py`、
  `tests/test_cli_reports.py` 共 11 passed，0 failed，0 skipped；覆盖 selector/validator、真实 schema
  structure iterator、58-row schedule、周期 image-edge multiset audit 与成功/失败 JUnit 序列化；
- full local suite：129 passed，0 failed，0 skipped（183.08 s）；
- `python -m compileall -q src tests`、四个相关 sbatch 的 `bash -n` 与 `git diff --check` 均通过；
- 本地未生成 synthetic acceptance asset，也未运行真实 checkpoint smoke。最终 32-record fixture、58 个
  GPU rows、四个 real-subset training units 与 scheduler `sacct` 证据仍必须在 Guqq 经 Slurm 完成。

## 2026-09-12 — End-to-end model and five-structure smoke runner

计划检查：

- 组合四类 adapter 与 `PointGroupTensorModel`，确保 adapter 直接投影到当前 architecture 的
  hidden layout，downstream 复用 backbone 返回的 edge geometry，而不是静默使用输入 fallback graph；
- 将每个 `TensorSample` 逐结构 canonicalize、保持 site order、collate，并把 raw target 旋转到
  canonical coefficient frame；global target 按 crystal 拼接，BEC 按 atom 拼接；
- smoke runner 严格执行 3 train / 1 validation / 1 test、train-only normalizer、一次 optimizer
  update、validation、checkpoint save/load、test metrics JSON；backbone 冻结而 interface/downstream
  获得有限梯度；
- 提供真实 backbone factory 和依赖注入的单元测试 seam；测试 seam 只验证 orchestration，最终
  acceptance 仍必须由四个真实 checkpoint 的 Slurm runs 提供，不能以 analytic extractor 替代；
- report 固定记录 commit、unit、architecture、backbone、sample IDs、loss、per-copy metrics、
  active/trainable parameters 与 checkpoint path，缺少资源或不兼容 metadata 时非零失败。

实际结果：

- 新增 end-to-end composition，单元测试证明 downstream 使用 adapter 返回的原生 edge tensors
  （共享 storage），缺失任一 geometry field 时 fail closed；canonical batch 的 target scope 与
  symmetry/sample mapping 对齐。
- orchestration-only analytic seam 完成 3/1/1 train-only normalization、optimizer update、
  validation、atomic checkpoint reload、test per-copy metrics；明确不计作真实 backbone 证据。
- 20-run schedule 自动检查：四独立 unit 各覆盖五 branches，每个 target 覆盖四 backbones，
  `a1_only|full_pg` 与 adaptation/O3E/readout 各自 `full_o3|o2_tp` 均有覆盖。
- `tests/test_end_to_end_smoke.py`：4 passed；完整项目 `tests/`：123 passed，0 failed，
  0 skip；compileall、schedule JSON 和 diff whitespace 检查通过。
- `slurm/smoke_real_subsets.sbatch` 尚未提交；真实 checkpoint restore、GPU forward/backward、
  latency/memory 和科学数据链结果仍待 Guqq，不能由本地 seam 结果替代。


## 2026-09-11 — Four independent real-data modules

计划检查：

- 定义统一、不可变的 tensor sample contract，严格验证 lattice、fractional sites、atomic
  numbers、global/node target shape、finite values、单位和独立 `TrainingUnit`；
- JARVIS dielectric/elastic loader 复用已冻结 GMTNet published split 和过滤规则，分别解析
  3x3 dielectric 与 kbar→GPa 后的 6x6 Voigt elastic，并还原完整 minor/major-symmetric
  3x3x3x3 tensor；
- MatTen loader 复用文件内 train/val/test split，解析 pymatgen structure dict 与原生完整
  elastic tensor，不依赖 pymatgen runtime；
- JARVIS-DFPT loader 流式解析 calculation-matched JSONL，保持 finalpos site order、逐原子
  N×3×3 BEC、source hashes 和 ASR metadata；processed output 缺失时严格失败；
- 四个 loader 都验证 manifest resource size/checksum、sample ID 唯一性、split 覆盖与无泄漏，
  并提供固定 3/1/1 五结构 smoke selection；使用小型临时真实-schema fixtures 做单元测试，
  完整文件解析及端到端训练留给 Guqq Slurm。

实际结果：

- synthetic real-schema fixtures 覆盖四个 loader、Voigt expansion、Cartesian↔irrep、node
  scope/site order、source metadata、resource gate、pending BEC gate 和 graph construction；
  `tests/test_real_data_modules.py + test_training_units.py`：11 passed。
- 三个本地可用真实数据源的冻结 3/1/1 均通过 size/SHA 和解析：JARVIS dielectric
  5 structures（5–46 atoms）、JARVIS elastic 5 structures（6–24 atoms）、MatTen elastic
  5 structures（2–5 atoms）；target shape/units 分别为 3×3 dimensionless 与
  3×3×3×3 GPa。
- 完整项目 `uv run python -m pytest tests -q --basetemp ... -p no:cacheprovider`：
  119 passed，0 failed，0 skip；`python -m compileall -q src tests` 与
  `git diff --check` 通过。
- 首次定向收集暴露 `TARGET_LAYOUTS` 错误模块边界（从 `src.irreps` 导入）；修正为公开
  `src.heads` 接口后全部通过。完整 BEC processed JSONL 仍缺失，默认 loader 按计划
  fail closed，不能将 10 个下载样本冒充完整数据。


## 2026-09-11 — Real EquiformerV2 SO(3) adapter

计划检查：

- 保持 Goal 冻结的 `facebook/OMAT24@8a5a.../eqV2_31M_mp.pt`，严格验证 gated
  checkpoint size/SHA、官方 fairchem runtime/config 与内部 SO(3) coefficient layout；
  不用 OC20、随机权重或其他文件替换最终验收资源。
- 通过官方 fairchem graph/model API 加载 checkpoint，在 energy/force scalar/vector head 前
  取最后 node SO(3) embedding；验证 resolution/lmax/mmax/channel ordering 和 node mapping。
- 复用官方 radius graph、cell offsets、edge vectors与 cutoff；用周期反演 pair 两次真实
  forward 构造 O(3) even/odd blocks，再进入 trainable interface projector。
- 本地覆盖 resource gate 先于 fairchem import、layout flatten、双调用/frozen policy 和 Slurm
  smoke contract；真实 checkpoint proper/improper/backward 仅经 Guqq Slurm，0 skip/xfail。
- 官方页面若继续要求用户同意许可并共享联系信息，保持 fail closed 并记录最小人工动作；
  不自动代表用户接受协议。

实际结果：

- 官方 fairchem-core 1.10.0 源码确认 checkpoint 使用 HydraModel，最终 backbone
  `forward` 在所有 8 层和 final norm 后返回 `node_embedding` 与实际 `graph`；冻结布局为
  `[N,25,128]`、`lmax=[4]`、`mmax=[2]`、cutoff 12 Å。
- 新增 copy-major flatten/layout 与严格 gated resource 测试；
  `uv run python -m pytest tests/test_backbone_contracts.py -q --basetemp ...`：13 passed。
- `uv run python -m pytest tests -q --basetemp ...`：114 passed，0 failed，0 skip；
  `python -m compileall -q src tests`、manifest `json.tool` 和 `git diff --check` 通过。
- 一次未限定路径的 `pytest` 错误收集了 Git 忽略的 `data/sources/*` 和 `data/vendor/*`
  上游仓库测试，产生 48 个缺失上游可选依赖/fixture 的 collection errors；这超出已冻结的
  项目测试范围，随后显式限定 `tests/` 的完整项目套件通过，未降低任何项目测试标准。
- exact checkpoint 仍因 Hugging Face OMat24 人工许可门控而不可下载；因此本地只验收了
  fail-closed、布局、脚本和静态 contract，真实 restore/proper/improper/backward Slurm
  验收继续待资源访问，未用 OC20 权重代替、未声明真实 smoke 已通过。


## 2026-09-11 — Real GRACE tensorpotential adapter

计划检查：

- 解析发布 archive 中的 `model.yaml`，严格验证 TensorPotential 版本、element map、cutoff、
  `rho` instruction 类型和最终 scalar readout 的连接关系；不凭名称猜测 feature tap。
- 通过官方 TensorPotential restore/load API 恢复权重，在 scalar readout 前提取明确的
  `rho` 中间张量；验证实际 shape、allowed `(l, parity)` blocks 与 component ordering。
- 复用 TensorPotential 自身构造的 bond indices、周期 shifts/vectors 和 node ordering，输出
  统一 `O3FeatureBatch`；checkpoint 永久冻结，仅 interface projector 接受梯度。
- 对真实小结构测试 identity/proper/improper/reflection、mixed batch、edge round-trip、有限
  backward；真实 checkpoint 运算只经 Guqq Slurm，且不得以随机网络或 skip 代替。
- 若发布要求的 `tensorpotential==0.5.10` 无可安装 artifact，则必须定位可复现的官方源码
  revision 或明确 fail closed，不静默使用 0.5.9/0.6.x。

阶段性实际结果（真实 Slurm 尚待执行）：

- YAML/instruction audit 证实 `rho.only_invar=true`、`ls_max=[0,0,0,0]`、17 scalar
  channels；原 manifest 的 (l\le4) `rho` 描述错误。正式 tap 改为共同馈入 `AA1/AA2`
  两条高阶分支的 `AA`：32 channels、141 angular/history slots、总宽 4,512。
- `AA` metadata 精确分成 27 个自然宇称 history copies：各阶总 multiplicity 为
  `l0=160, l1=128, l2=224, l3=160, l4=192`；转换保持每个 32-channel copy 的
  contiguous m-block。
- 用官方 TensorPotential 球谐和 e3nn 在 64 个确定性单位向量上求正交基变换；`l=0..4`
  最大残差为 `3.6e-15`。adapter 构造时会对实际 runtime 重新推导并以 `2e-12` fail closed。
- checkpoint 声明版本 0.5.10，但 PyPI 与官方 tags 均无该发行版；选择 checkpoint 后首个
  正式官方兼容 loader `tensorpotential==0.6.0`，同时保留 metadata 0.5.10 双重校验，
  不伪称存在 0.5.10 artifact。
- archive、`model.yaml`、TensorFlow checkpoint index/data 均已记录独立 size/SHA 并纳入
  registry 校验；创建 `grace_adapter_smoke.sbatch` 做真实 restore/proper/improper/backward。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_backbone_contracts.py -q`：
  11 passed；完整套件：118 passed、1 个既有 opt-in skip、0 failed。
- `python -m compileall -q src tests/test_backbone_contracts.py`、manifest JSON 与
  `git diff --check`：通过。`uv run ruff check ...` 未执行，因为当前 uv 环境没有 ruff
  executable；未把工具缺失记作代码通过。


## 2026-09-11 — Real DPA4-Plus SO(3) adapter

计划检查：

- 构造时验证 DPA4-Plus checkpoint/config size+SHA 与 `deepmd-kit==3.2.0`、PyTorch 2.11
  runtime；用官方 `Tester/get_model` 恢复 checkpoint，不随机初始化或冻结成 `.pt2`。
- 调用官方 `build_neighbor_list` 与 descriptor `forward_with_edges`，在 l=0 readout 前取
  final equivariant latent `[N,25,1,64]`；过滤官方 guard edges并复用
  `src=neighbor,dst=center,edge_vec=r_src-r_dst` 图约定。
- 将 `[l,m,channel]` 逐阶变换为 e3nn multiplicity-major `64x(l=0..4)` SO(3) layout，
  component-order 必须经官方 e3nn grid/Wigner convention 审计，不静默假定。
- 使用周期反演 pair 的两次真实 DPA4 forward 生成 even/odd O(3) blocks，再投影到共享
  hidden layout；测试两次调用、冻结梯度、proper/improper/reflection、mixed node mapping、
  edge/cell-shift round-trip和 finite projector gradients。
- 本地测试 runtime/resource/layout fail-closed；真实 checkpoint 数值 suite 仅经 Guqq Slurm，
  预先固定 float32 tolerance，最终不得 skip/xfail。

阶段性实际结果（真实 Slurm 尚待执行）：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_backbone_contracts.py -q`：
  9 passed；新增覆盖 DPA4 资源 gate、checkpoint/config 路径与 checksum contract、
  `[N,25,1,64]` shape fail-closed、五个 degree block 及 copy-major 精确索引映射。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests assets/model_code/tests -q`：
  116 passed、1 个既有 opt-in skip、0 failed。
- `python -m compileall -q src tests/test_backbone_contracts.py` 与 `git diff --check`：
  通过；后者只有工作树 LF→CRLF 提示。
- 已创建真实 2-node Si、proper/improper、冻结梯度与 interface backward 的
  `dpa4_adapter_smoke.sbatch`，固定 float32 tolerance `1e-3`。Guqq 连接在远程 shell
  建立前被关闭，故真实 checkpoint 数值结果仍未产生，不能据此结束 DPA4 验收。


## 2026-09-11 — Real MACE-MP medium-0b3 adapter

计划检查：

- 构造时验证 manifest、79,472,952-byte checkpoint SHA-256 与 `mace-torch==0.3.16`；
  使用官方 `MACECalculator` 加载本地路径，不隐式下载或回退随机权重。
- 对真实 ASE/PBC structure 调用 MACE graph/model，在 energy scalar readout 前截取
  `products[0].linear.irreps_out == 128x0e + 128x1o` 的 first-interaction node state，
  不接受 tap layout 或 feature width 漂移。
- 输出统一 `O3FeatureBatch`，保留 mixed batch node order，并登记 MACE 实际 edge index、
  Cartesian shifts/vectors/distances、cell/positions/species；interface projector 进入目标 layout。
- checkpoint 参数全部 frozen 且 `train()` 后仍 eval/no-grad；projector loss backward 有限且
  checkpoint gradients 恒为零。真实 identity/proper/improper/reflection 测试在 Guqq Slurm
  使用同一 checkpoint 执行，本地仅做不加载大模型的 adapter API/fail-closed 单测。

阶段性实际结果（真实 Slurm 尚待执行）：

- `uv run python -m pytest tests/test_backbone_contracts.py -q`：7 passed；新增覆盖精确
  MACE runtime、`128x0e+128x1o` layout 转换、resource gate 先于模型加载，以及固定
  2-node/10-component/`3e-4` tolerance 的 Slurm smoke contract。
- `uv run python -m pytest tests assets/model_code/tests -q`：114 passed，1 个既有 opt-in
  skip，0 failed。真实 checkpoint 数值 smoke 尚未执行，不能据此结束 MACE 验收。



## 2026-09-11 — Four real pretrained backbone adapters

计划检查：

- `data/manifests/backbones.json` 必须冻结 MACE/GRACE/DPA4/EquiformerV2 四项的
  repository/revision/license/runtime/local path/checksum/feature tap/parity/graph contract；
  缺文件、checksum 不符、版本不兼容或 gated 权限未解决时 fail closed。
- 四 adapter 在原 scalar readout 前提取真实 checkpoint node features，转换为明确
  `IrrepLayout`/component order/node batch/edge geometry 的 `O3FeatureBatch`；优先复用
  backbone graph，否则按冻结 graph spec 确定性重建。
- MACE/GRACE 逐 checkpoint 审计并测试 O(3) parity；DPA4/EquiformerV2 使用固定的
  inversion-paired Reynolds wrapper，以两次真实 backbone evaluation 构造 parity
  channels，禁止仅改 metadata。
- 默认冻结 checkpoint 参数并固定 eval 行为；允许的 interface projection 参数可反传，
  active non-backbone 计数包含它们。测试 batch/node mapping、float/device、empty-edge、
  proper/improper transforms、反射配对和 finite gradients。
- 本地运行纯 contract/parity/manifest tests；四个 checkpoint 的真实最小 batch
  feature/rotation/reflection/frozen-gradient 测试登记为 Guqq-only，但最终 Slurm 报告
  必须四项全绿且无 skip/xfail。

阶段性实际结果（resource/parity/interface 基础层）：

- `uv run python -m pytest tests/test_backbone_contracts.py -q`：5 passed；覆盖四项 manifest
  顺序与必填元数据、gated/missing/size/SHA/path-escape fail-closed、周期反演 involution、
  双真实 extractor 调用、proper/improper O(3) 协变、冻结策略及 interface projector 梯度。
- `uv run python -m pytest tests assets/model_code/tests -q`：112 passed，1 个既有 opt-in
  skip，0 failed。此结果不等同于四个真实 checkpoint acceptance；后者仍待 adapter 与
  Guqq Slurm suite。



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

# 2026-09-12 — Guqq second-stage runtime compatibility fixes

## 测试范围与预期结果

- GRACE `GeometricalDataBuilder` 必须使用 TensorPotential 0.6.0 接受的字符串 dtype，且现有
  TensorFlow CPU fallback 与审计字段保持不变。
- smoke 的 e3nn 表示矩阵必须先由 CPU rotation 构造，再显式转换到 feature 的 device/dtype；
  用 fake representation 验证该边界，防止 e3nn 0.5.9 CPU 常量与 CUDA rotation 混用。
- O(2) 有限群路径构造须在 e3nn 0.4.4 与 0.5.9 上保持相同 path count/checksum，并满足既有
  intertwining 阈值；先用 Guqq 的隔离 DPA4 环境通过 Slurm 诊断精确数值差异，再决定修复，
  不通过放宽验收阈值掩盖误差。
- 本地运行新增目标测试、相关 subduction/tensor-product/backbone 测试及完整测试；Guqq 仅重提
  GRACE/DPA4 standalone 和 real 0–2 最小 Slurm 探针，五项全绿前不恢复较大数组。
- EquiformerV2 checkpoint 继续暂停：不下载、不校验、不加载，也不提交依赖它的作业。

## 实际结果

- Guqq CPU-only Slurm diagnostic `407`：`COMPLETED 0:0`；Torch 2.11/e3nn 0.5.9 下旧实现
  有 992/1000 degree/parity 组合触发 strict intertwining failure，确认不是单一路径尾差。
- 新实现以显式 float64 real-basis generators 计算 O(3) 矩阵，不修改全局默认 dtype；新增测试
  验证 degree 0–4、e/o parity 与 e3nn 0.4.4 convention 一致，并验证 sampled dihedral group law。
- 目标回归：`51 passed, 67 warnings`；完整本地回归：`194 passed, 496 warnings`，耗时
  247.42 s；均无 skip/xfail/failure。
- `git diff --check`：通过，仅有工作树既有 LF→CRLF 提示。
- `uv run ruff check ...`：未执行，当前 uv 环境没有 `ruff` executable；ruff 不属于冻结测试
  计划，且完整 Python 测试已覆盖所有修改模块的导入与执行，因此不以缺失工具替代或降低测试。
- Guqq second-stage GPU 验收：408、409、410_[0-2] 已成功提交；其后连续三次监控连接未
  到达可验证的 pull/queue 输出，故五项 terminal/JSON/JUnit 结果仍为 pending，不伪报通过。
