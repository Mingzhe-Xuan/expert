# 开题报告草案：面向晶体张量性质预测的预训练 O(3) 表示与连续层级点群特化

> **暂定题目（中文）**：面向平衡晶体张量性质预测的预训练 O(3)-等变表示与连续层级点群特化学习  
> **Tentative Title (English)**: **Pretrain at the Universal Symmetry Level, Specialize along the Symmetry-Breaking Hierarchy: Continuity-Aware Point-Group Equivariant Learning for Crystal Tensor Properties**  
> **版本**：Proposal v0.1  
> **日期**：2026-09-04

---

## 摘要

晶体材料的介电张量、弹性张量等性质不仅取决于化学组成和局域几何环境，还严格受到晶体对称性的约束。近年来，基于 O(3)/E(3) 等变神经网络的模型已经能够从原子结构直接预测张量性质，并保证坐标旋转或反射下的物理协变性。然而，标准 O(3)-等变网络通常只利用“任意三维旋转/反射下的连续群对称性”，并未显式利用 relaxed equilibrium crystal 所具有的、结构依赖的离散 stabilizer symmetry，即 crystallographic point group。与此同时，当前高质量张量标签仍显著少于能量、力和结构数据，使得直接为每个张量任务或每个点群从头训练模型存在明显的数据效率问题。

本研究拟提出一种 **pretrained O(3)-equivariant backbone + shared O(3) adaptation + hierarchical symmetry-breaking gates + full point-group branches** 的晶体表示框架。核心思想是：首先利用在大规模原子数据上预训练的 O(3)-等变 backbone 提取具有通用化学和几何意义、同时保留方向信息的高质量表示；再通过跨点群共享的 O(3) task adaptation 形成下游表示；随后对当前点群及其物理兼容的父群分别激活完整分支。对每个激活群 \(K\)，该分支依次执行 \(O(3)\rightarrow K\) subduction、point-group tensor product、point-group equivariant MLP 与 PG-equivariant norm。global tensor 任务随后进行 pooling 与 \(\mathrm{Fix}_K(V_T)\)-constrained readout；atom-resolved BEC 则跳过 pooling，直接使用共享的 node-wise equivariant tensor readout。各分支输出再提升回统一 O(3) tensor-irrep space，并根据当前结构与各父群 symmetry operations 之间的 normalized residual 构造连续、O(3)-invariant gates，进行层级加权平均；融合结果可再经过一个共享的 O(3)-equivariant readout，最后统一转换回 Cartesian basis 和原始输入坐标系。子群分支权重在父群对称性恢复时严格趋于零，从而同时利用跨点群迁移、离散对称性特化与结构形变路径上的连续性。

本研究聚焦 relaxed equilibrium structures 上的张量性质，以 global dielectric / elastic tensors 和 atom-resolved Born effective charge（BEC）为主要对象。三类任务共享同一 node-wise backbone、shared O(3) adaptation 和 PG branches；实现层面的主要任务开关是 **是否进行 global pooling**：dielectric / elastic 在 PG norm 后 pooling，BEC 保留逐原子输出。主实验采用 **JARVIS tensor benchmark**、**MatTen elastic benchmark** 与 **JARVIS-DFPT BEC**；**MP-Dielectric BEC** 作为跨数据库候选 benchmark，在锁定数据发布版并核验底层 DFPT task 的完整 BEC 字段后使用。研究将系统区分并量化四类增益：**pretraining gain、hard-routing gain、hierarchical continuity gain 和 point-group representation gain**。同时通过 anisotropy-conditioned analysis、irrep-wise error、symmetry violation、symmetry-breaking path continuity、sample-efficiency curve、parameter-matched ablation 等实验回答一个更一般的科学问题：

> **How can universally pretrained O(3)-equivariant representations be efficiently specialized to the discrete stabilizer symmetries of equilibrium crystals for tensorial property prediction?**

本研究的核心 thesis 可概括为：

> **Pretrain at the universal symmetry level, specialize at the material symmetry level.**

---

# 1. 研究背景

## 1.1 从标量性质预测走向张量性质预测

材料机器学习长期以 formation energy、band gap、bulk modulus 等标量性质为主要预测对象。对于标量目标，只需要保证模型输出对整体旋转和平移不变：

$$
f(Rx)=f(x).
$$

然而大量关键材料性质具有明确的方向依赖，例如：

- dielectric tensor \(\epsilon_{ij}\)；
- elastic tensor \(C_{ijkl}\)；
- piezoelectric tensor \(e_{ijk}\)；
- Born effective charge \(Z^*_{\kappa,ij}\)；
- magnetic susceptibility；
- thermal conductivity 等。

对于这些量，旋转晶体结构后，预测结果必须按照对应的张量表示同步变换，而不能保持不变。例如 rank-2 tensor 满足：

$$
T' = RTR^\top,
$$

rank-4 elastic tensor 满足：

$$
C'_{ijkl}
=
R_{ia}R_{jb}R_{kc}R_{ld}C_{abcd}.
$$

因此，张量性质预测不仅是一个数值回归问题，更是一个：

> **representation learning under symmetry constraints**

的问题。

---

## 1.2 O(3)-equivariant neural networks 的作用与局限

现代等变神经网络通过在 O(3)/SO(3) irreducible representations（irreps）上组织特征，将网络内部表示写为：

$$
h=\bigoplus_{\ell,p}n_{\ell,p}D^{(\ell,p)},
$$

其中 \(\ell\) 表示 angular momentum order，\(p\) 表示 parity。

旋转或反射输入结构时：

$$
h^{(\ell,p)}(Rx)
=
D^{(\ell,p)}(R)h^{(\ell,p)}(x).
$$

通过 Clebsch–Gordan tensor product，可以在保持等变性的同时进行不同 angular channels 之间的非线性交互：

$$
D^{(\ell_1)}
\otimes
D^{(\ell_2)}
=
\bigoplus_{\ell=|\ell_1-\ell_2|}^{\ell_1+\ell_2}
D^{(\ell)}.
$$

Tensor Field Networks、e3nn、NequIP、MACE、Equiformer 等工作已经证明，O(3)/E(3)-等变表示对于三维原子系统具有很高的数据效率与物理一致性。针对材料张量预测，MatTen、AnisoNet、GMTNet 等进一步证明了等变模型能够直接预测 elasticity、dielectric 等方向相关性质。

但标准 O(3)-equivariant network 存在一个关键结构性限制：

对于固定 \(\ell\)，它把整个 \((2\ell+1)\)-dimensional subspace 视为一个连续旋转群 irrep；网络参数不能根据一个具体晶体所具有的离散 point-group symmetry，对同一个 \(\ell\) 内不同 symmetry-adapted subspaces 使用不同的交互规则。

换句话说，O(3)-equivariance 告诉模型：

$$
f(Rx)
=
\rho(R)f(x),
\qquad
R\in O(3),
$$

但对 equilibrium crystal \(x\)，还存在一个非平凡 stabilizer subgroup：

$$
G_x
=
\{g\in O(3)\mid gx=x\}.
$$

因此对于：

$$
g\in G_x,
$$

有：

$$
f(x)=\rho(g)f(x).
$$

标准 O(3) 模型能够在原则上“学到”这种约束，却通常不会在参数化层面显式利用 \(G_x\) 的离散结构。

---

## 1.3 为什么 equilibrium crystal 的 point group 值得显式利用

对于 relaxed equilibrium structures，晶体结构的空间群/点群通常是一个稳定且具有物理意义的结构属性。

根据 Neumann principle，材料物性的对称性必须包含晶体点群的对称操作。因此目标 tensor 并非分布在任意 tensor space 中，而是位于当前点群允许的固定子空间：

$$
\mathrm{Fix}_G(V_T)
=
\left\{
T\in V_T
\mid
\rho_T(g)T=T,
\ \forall g\in G
\right\}.
$$

因此不同点群对应不同的：

1. tensor independent components；
2. symmetry-forbidden components；
3. O(3) irreps 在离散群下的 branching / subduction；
4. symmetry-adapted nonlinear interaction channels。

例如，同一个 \(\ell=2\) O(3) irrep 在不同点群中会分解为不同的有限群 irreps：

$$
D^{(2)}
\downarrow_G
=
\bigoplus_{\Gamma\in\widehat G}
m_{2,\Gamma}\Gamma.
$$

这种 within-\(\ell\) specificity 正是 point-group-aware representation 相比纯 O(3) representation 可以利用的额外信息。

PGEqNN 的近期工作进一步表明：将 O(3)/SO(3) rotational order 进一步按照 point-group irreps 划分，可以在部分 dielectric / elastic 任务中提取额外的 symmetry-specific signal。

其分析还表明，许多 global equilibrium tensor 的主要预测信息集中于 trivial irrep \(A_1\) subspaces，而 full point-group representation 的额外优势主要在存在真正可学习 anisotropic signal 的场景中出现。

这说明：

> **point-group awareness 并非“任何数据上都自动有效”，而需要针对 anisotropy、tensor order 和 point-group structure 进行机制化验证。**

---

## 1.4 为什么先使用 pretrained O(3)-equivariant backbone？

这是本研究需要主动回答的核心 reviewer question。

### 原因 1：利用 universal pretraining 提升 label efficiency

高质量 tensor labels 的规模通常只有：

$$
10^3\sim10^4,
$$

远小于能量、力或通用结构预训练可利用的数据量。

如果对每个 tensor task、每个 point group 都从头学习：

- chemistry；
- coordination；
- bonding；
- local geometry；
- many-body interactions；

会造成大量重复学习。

因此采用：

$$
\text{large-scale atomistic pretraining}
\rightarrow
\text{universal representation}
\rightarrow
\text{small tensor dataset specialization}
$$

可以提高下游样本效率。

---

### 原因 2：O(3)-equivariant features 保留 tensor task 所需方向信息

如果 pretrained backbone 只输出 invariant scalars：

$$
h_{\mathrm{inv}}(Rx)
=
h_{\mathrm{inv}}(x),
$$

则与局域方向、键角、各向异性相关的信息可能已经被压缩，后续 PG module 需要重新恢复方向结构。

而 O(3)-equivariant pretrained representation 保留：

$$
h
=
h^{(0)}
\oplus
h^{(1)}
\oplus
h^{(2)}
\oplus\cdots,
$$

使 downstream tensor task 能直接利用具有正确 transformation law 的 directional features。

---

### 原因 3：O(3) 是所有 crystallographic point groups 的共同母空间

对于任意 crystallographic point group：

$$
G\subset O(3).
$$

因此可以形成自然的 symmetry hierarchy：

$$
\boxed{
O(3)
\rightarrow
G
\rightarrow
\mathrm{Fix}_G(V_T)
}.
$$

这意味着 pretrained O(3) representation 可以作为跨所有 point groups 的统一 representation interface。

不同结构只需在下游根据其 stabilizer group 做 specialization，而不必分别预训练 32 套完全独立的模型。

---

### 原因 4：跨点群 transferable chemistry 与 PG-specific physics 应当解耦

局域 Si–O coordination、transition-metal polyhedra、chemical bonding motifs 等知识本质上可跨 point groups 共享。

将所有知识都交给独立 PG model 会导致：

$$
\text{data fragmentation}.
$$

因此我们希望把知识分解为：

$$
\underbrace{
\theta_{\mathrm{pre}}
}_{\text{universal chemistry / geometry}}
+
\underbrace{
\Delta\theta_{\mathrm{task}}
}_{\text{tensor-task adaptation}}
+
\underbrace{
\left\{
w_K(r(x))\Delta\theta_K
\right\}_{K\in\mathcal A(x)}
}_{\text{hierarchical symmetry specialization}}.
$$

本研究因此不是“用 point group 替代 O(3)”，而是主张：

> **Point-group symmetry should specialize universal equivariant representations rather than replace them.**

---

# 2. 研究问题与科学假设

## 2.1 总体研究问题

> **How can universally pretrained O(3)-equivariant representations be efficiently specialized to the discrete stabilizer symmetries of equilibrium crystals for tensorial property prediction?**

中文可表述为：

> 如何将大规模预训练得到的通用 O(3)-等变晶体表示，高效特化到 relaxed equilibrium crystal 的离散点群 stabilizer symmetry，从而提高全局张量性质预测的数据效率、对称性一致性与各向异性建模能力？

---

## 2.2 子问题

### RQ1：Universal pretraining 是否显著提高 tensor label efficiency？

比较：

$$
\text{same architecture from scratch}
\quad\text{vs.}\quad
\text{pretrained O(3) backbone}.
$$

重点不只关注 full-data accuracy，而要关注达到同一误差所需的 labels 数量。

---

### RQ2：父群—子群层级条件化是否具有独立价值？

比较：

$$
\text{Shared-O3}
\quad\text{vs.}\quad
\text{Shared-O3 + hierarchical full branches}.
$$

该实验回答：

> **hierarchical symmetry-breaking awareness**

是否具有独立价值。

---

### RQ3：显式进入 point-group representation space 是否提供层级条件化之外的额外收益？

比较：

$$
\text{Hierarchical-O3}
\quad\text{vs.}\quad
\text{Full PG representation model}.
$$

该实验回答：

> **representation-level PG awareness**

是否具有独立价值。

---

### RQ4：PG-aware representation 在什么条件下真正有效？

重点考察：

- anisotropy 强弱；
- tensor order；
- point-group symmetry level；
- 每个 PG 的数据规模；
- \(\ell=0,2,4\) 等 harmonic channels 的可学习程度。

---

### RQ5：shared O(3) adaptation 与 current-plus-parent full branches 是否具有互补性？

假设：

- shared O(3) adaptation 学习跨 point group transferable 的 tensor knowledge；
- current / parent branches 学习由连续 parent-group residuals 调制的层级 specialization。

---

### RQ6：中间 non-trivial PG irreps 是否必要？

由于已有结果表明很多 global equilibrium tensor 预测信息集中于 trivial \(A_1\) blocks，因此必须比较：

$$
\text{Full PG intermediate representation}
\quad\text{vs.}\quad
A_1\text{-only intermediate representation}.
$$

---

## 2.3 核心科学假设

### H1 — Pretraining hypothesis

预训练 O(3) backbone 的主要收益体现为：

> **label efficiency**

尤其在 low-data regime 下更加明显。

---

### H2 — Hierarchical continuity hypothesis

对当前点群及其物理兼容父群运行完整分支，并使用连续、O(3)-invariant 的 symmetry-breaking gates 融合其输出，可以在保留 PG specialization 的同时缓解 hard point-group switching 导致的表示不连续。关键边界条件是：当父群对称性恢复时，相应子群分支权重必须趋于零。

---

### H3 — Representation specialization hypothesis

将 task-adapted O(3) representation 分别 subduce 到当前点群及兼容父群，并在各分支进行：

$$
PG\ tensor\ product
+
PG\text{-equivariant nonlinear processing}
$$

可以进一步利用 within-\(\ell\) 的离散 symmetry specificity。

---

### H4 — Anisotropy hypothesis

PG-aware model 相比纯 O(3) model 的收益应随可学习 anisotropic signal 增强而上升，而不是在所有样本、所有点群上均匀增加。

---

### H5 — Long-tail hypothesis

在每个 PG 的数据量较少时：

$$
\text{Hierarchical-Full}
>
\text{Branch-only},
$$

因为 shared O(3) adaptation 能跨点群迁移 tensor-task knowledge，降低数据碎片化。

---

# 3. 研究对象与任务范围

## 3.1 结构对象

第一阶段仅研究：

> **relaxed equilibrium crystal structures**

原因包括：

1. equilibrium symmetry 具有清晰物理意义；
2. 可以通过 spglib / pymatgen 获得 point group、space group 和 standard setting；
3. Neumann principle 对 equilibrium bulk tensor 约束直接；
4. 避免第一阶段同时处理 thermal disorder、defect、phonon displacement 和 approximate symmetry。

后续扩展可研究：

- weakly symmetry-broken structures；
- strained structures；
- finite-temperature snapshots；
- defects / substitutions；
- 更一般的 subgroup DAG、近似 symmetry embeddings 与有限温结构上的连续 gate。

---

## 3.2 第一阶段目标性质

### 3.2.1 Dielectric tensor

选择 symmetric rank-2 dielectric tensor：

$$
\epsilon_{ij}
=
\epsilon_{ji}.
$$

其 O(3)/SO(3) irreducible decomposition 为：

$$
V_\epsilon
\cong
D^{(0,+)}
\oplus
D^{(2,+)}.
$$

其中：

- \(\ell=0\)：isotropic trace；
- \(\ell=2\)：traceless anisotropic part。

它是研究：

> isotropic vs anisotropic signal

以及 point-group specialization 的理想起点。

---

### 3.2.2 Elastic tensor

Elastic tensor 满足 intrinsic symmetries：

$$
C_{ijkl}
=
C_{jikl}
=
C_{ijlk}
=
C_{klij}.
$$

因此其独立维度为 21，并可在 SO(3) 下分解为：

$$
V_C
\cong
2D^{(0,+)}
\oplus
2D^{(2,+)}
\oplus
D^{(4,+)}.
$$

维度检查：

$$
2\times1
+
2\times5
+
9
=
21.
$$

相比 dielectric，elastic tensor 拥有更高 angular order 和更复杂的 anisotropic structure，因此预计更能体现 PG specialization 的价值。

---

### 3.2.3 Atom-resolved 任务：Born effective charge

Born effective charge 是 atom-resolved rank-2 tensor：

$$
Z^*_{\kappa,\alpha\beta}
=
\frac{\partial P_\alpha}
{\partial u_{\kappa\beta}}.
$$

BEC 是一般的 \(3\times3\) 张量，不预设对称性；其 O(3) representation 为：

$$
V_Z
\cong
D^{(0,+)}
\oplus
D^{(1,+)}
\oplus
D^{(2,+)},
$$

维数为 \(1+3+5=9\)。其直接约束不仅来自 global crystal point group，还涉及：

- site symmetry；
- Wyckoff orbit；
- symmetry-related atoms 之间的 permutation-covariance。

与 dielectric / elastic 的网络主体保持一致，区别集中在 PG norm 之后：

$$
\begin{aligned}
\text{dielectric / elastic:}&\quad \{q_i\}\rightarrow \operatorname{Pool}_i(q_i)\rightarrow \text{global readout},\\
\text{BEC:}&\quad \{q_i\}\rightarrow \{\text{shared node-wise equivariant readout}(q_i)\}.
\end{aligned}
$$

即 BEC 不做 global pooling，输出完整的 \(N\times3\times3\) tensor。需要强调：这是同一主干上的 pooling 开关，而不是把每个原子分别投影到全局 \(\mathrm{Fix}_K(V_Z)\)；对任意 \(g\in K\)，逐原子输出必须满足：

$$
\hat Z^*_{\pi_g(\kappa)}
=
R_g\hat Z^*_{\kappa}R_g^\top,
$$

其中 \(\pi_g\) 是空间群操作诱导的原子置换。该关系不需要额外的 Wyckoff-specific module：只要 PG-equivariant PBC message passing 对节点、周期边和 feature fiber 的联合群作用严格等变，且 tensor readout 在所有节点间共享，它便由图网络的 permutation-equivariance 与 PG-equivariance 自动推出；当 \(\pi_g(\kappa)=\kappa\)（模晶格平移）时，又自动退化为该 site stabilizer 的 fixed-subspace constraint。第一版可进一步对预测施加 acoustic sum rule 投影 \(\sum_\kappa \hat Z^*_{\kappa,\alpha\beta}=0\)，并同时报告投影前后的误差。

---

# 4. 数学基础

## 4.1 O(3)-equivariance

给定输入空间和特征空间上的群表示：

$$
\rho_{\mathrm{in}},
\qquad
\rho_{\mathrm{out}},
$$

函数 \(f\) 对群 \(G\) 等变，当且仅当：

$$
f(\rho_{\mathrm{in}}(g)x)
=
\rho_{\mathrm{out}}(g)f(x),
\qquad
\forall g\in G.
$$

对于三维材料，完整 frame transformation 可考虑 \(O(3)\)，包括 rotations 与 inversion / reflection。

---

## 4.2 O(3) irreducible representations

网络特征组织为：

$$
h
=
\bigoplus_{\ell,p}
h^{(\ell,p)}.
$$

每个 \(\ell\) block 在旋转下：

$$
h^{(\ell)}
\mapsto
D^{(\ell)}(R)h^{(\ell)}.
$$

Scalar、vector、quadrupolar 等分别对应：

$$
\ell=0,1,2,\ldots
$$

---

## 4.3 O(3) tensor product

两个 irreps 的 tensor product 经 Clebsch–Gordan decomposition 得到允许的输出 angular orders：

$$
h^{(\ell_1)}
\otimes
h^{(\ell_2)}
\xrightarrow{CG}
\bigoplus_{\ell=|\ell_1-\ell_2|}^{\ell_1+\ell_2}
h^{(\ell)}.
$$

本文中的 TP 不是脱离图结构的单次 feature product，而是在原子/周期分子图上构造并聚合消息。设有向边 \(j\to i\) 的源节点为 \(j=\mathrm{src}\)，边的等变特征为 \(e_{ij}\)，则一条允许的 CG path \(\pi\) 产生：

$$
m_{i\leftarrow j,\pi}
=
w_\pi\!\left([h_j^{\mathrm{inv}}\mid h_i^{\mathrm{inv}}]\right)
\left[
h_j^{(\ell_1,p_1)}\otimes e_{ij}^{(\ell_2,p_2)}
\right]_{\pi\rightarrow(\ell_{\mathrm{out}},p_{\mathrm{out}})},
$$

其中：

$$
w_\pi\!\left([h_j^{\mathrm{inv}}\mid h_i^{\mathrm{inv}}]\right)
=
\operatorname{MLP}_\pi\!\left([h_j^{\mathrm{inv}}\mid h_i^{\mathrm{inv}}]\right),
$$

其中 \([h_{\mathrm{src}}\mid h_{\mathrm{tgt}}]\) 表示 source 与 target node features 的 concatenation，\(w_\pi\) 是该 TP path 的标量权重。随后在目标节点聚合：

$$
u_i
=
\sum_{j\in\mathcal N(i)}
\sum_{\pi}m_{i\leftarrow j,\pi}.
$$

这里 MLP 的两部分输入必须分别取 source 与 target 节点的 invariant/scalar channels，或由各自节点 irreps 构造的不变量；不能将非平凡 irrep components 直接送入普通 MLP 生成路径权重。这样 \(w_\pi\) 在 O(3) 下为标量，整个消息构造与邻域求和才保持 equivariance。

shared adaptation 的图消息 TP 保留两种可切换实现，由配置参数控制：

$$
\texttt{shared\_tp\_backend}
\in
\{\texttt{full\_o3},\texttt{so2\_reduced}\}.
$$

- `full_o3`：直接枚举允许的完整 O(3) Clebsch–Gordan paths；
- `so2_reduced`：先将特征旋转到 edge-aligned local frame，在其中按 SO(2) \(m\)-selection rules 执行 TP / convolution，再旋转回 global frame。

两种实现必须输出完全相同的 O(3) irrep layout，因而后续 \(O(3)\rightarrow K\) subduction 与 PG branches 无需改变。`so2_reduced` 还必须显式处理 parity、reflection、edge reversal 与 local-frame gauge；若只满足 proper rotations，则只能声称 SO(3)-equivariance。该开关只作用于 shared O(3) adaptation，后续 finite-group PG TP 始终保留完整的 PG fusion rules。

---

## 4.4 Equivariant MLP

Equivariant MLP 在消息聚合后对每个节点分别作用：

$$
h_i'
=
\operatorname{EqMLP}_{O(3)}(u_i).
$$

它不跨节点完成邻域聚合，也不对 irrep component 随意使用普通 element-wise MLP，而是在 multiplicity / channel space 中进行线性 mixing，并配合允许的 equivariant / gated nonlinearities。所有跨节点通信均由前述 TP message passing 完成。

抽象记为：

$$
\mathrm{EqMLP}_{O(3)}:
\bigoplus_{\ell,p}
n_{\ell,p}D^{(\ell,p)}
\rightarrow
\bigoplus_{\ell,p}
n'_{\ell,p}D^{(\ell,p)}.
$$

---

## 4.5 Crystallographic point group 是 O(3) 的有限子群

对于 equilibrium crystal，记 point group 为 \(G\)：

$$
G\subset O(3).
$$

在 canonical frame 中，\(G\) 的对称轴具有标准方向，因此可以使用预先构造的：

- group representation matrices；
- character tables；
- Clebsch–Gordan coefficients；
- symmetry-adapted bases。

---

## 4.6 Subduction / restriction：O(3) → point group

参考 Heilman 与 Yan 的 PGEqNN（arXiv:2607.16871，Eq. (2)）所采用的 point-group harmonics（PGH）构造，对每个 O(3) irrep \(D^{(\ell,p)}\) 及嵌入已固定的 crystallographic point group \(K\subset O(3)\)，其 restriction 分解为：

$$
D^{(\ell,p)}
\downarrow_K
=
\bigoplus_{\Gamma\in\widehat K}
m_{\ell p,\Gamma}\Gamma,
$$

其中 \(\widehat K\) 表示 \(K\) 的 irreps 集合，重数可由 character inner product 得到：

$$
m_{\ell p,\Gamma}
=
\frac{1}{|K|}
\sum_{g\in K}
\chi^{(\ell,p)}(g)
\chi^\Gamma(g)^*.
$$

在 standard real spherical-harmonic basis \(\{Y_m^{(\ell,p)}\}_{m=-\ell}^{\ell}\) 与 PGH basis 之间定义完整、不降维的 unitary basis change：

$$
H_{\Gamma a\mu}^{(\ell,p)}
=
\sum_{m=-\ell}^{\ell}
\left[U_K^{(\ell,p)}\right]_{\Gamma a\mu,m}
Y_m^{(\ell,p)},
$$

其中 \(a=1,\ldots,m_{\ell p,\Gamma}\) 区分 repeated copies，\(\mu=1,\ldots,d_\Gamma\) 是 irrep \(\Gamma\) 内部的 component index。维数保持关系为：

$$
\sum_{\Gamma\in\widehat K}
m_{\ell p,\Gamma}d_\Gamma
=
2\ell+1.
$$

与 PGEqNN Eq. (2) 一致，\(U_K^{(\ell,p)}\) 将 standard harmonic basis 变为 symmetry-adapted basis，并同时 block-diagonalize \(K\) 中的所有表示矩阵：

$$
U_K^{(\ell,p)}
D^{(\ell,p)}(g)
U_K^{(\ell,p)\dagger}
=
\bigoplus_{\Gamma\in\widehat K}
\left(
I_{m_{\ell p,\Gamma}}
\otimes
D^\Gamma(g)
\right),
\qquad g\in K.
$$

因此 \(U_K^{(\ell,p)}\) 是 restriction representation 与 PG-irrep direct sum 之间的 intertwiner。选择 orthonormal PGH 时：

$$
U_K^{(\ell,p)\dagger}U_K^{(\ell,p)}
=
U_K^{(\ell,p)}U_K^{(\ell,p)\dagger}
=I.
$$

对带 channel index \(c\) 的 feature，完整正向 subduction 为：

$$
h^{PG}_{c,\Gamma a\mu}
=
\sum_{m=-\ell}^{\ell}
\left[U_K^{(\ell,p)}\right]_{\Gamma a\mu,m}
h^{O(3)}_{c,m}.
$$

若没有丢弃任何 \(\Gamma\) block，则 inverse basis change 为：

$$
h^{O(3)}_{c,m}
=
\sum_{\Gamma,a,\mu}
\left[U_K^{(\ell,p)}\right]^*_{\Gamma a\mu,m}
h^{PG}_{c,\Gamma a\mu},
$$

即：

$$
h^{O(3)}
=
U_K^{(\ell,p)\dagger}h^{PG}.
$$

PGEqNN 使用 real spherical harmonics 和实 PGH matrices；在该 convention 下 \(U_K^{(\ell,p)}\) 为实正交矩阵，因而：

$$
U_K^{(\ell,p)\dagger}
=
U_K^{(\ell,p)\top}.
$$

若采用 complex harmonics，则必须保留 Hermitian transpose \(U^\dagger\)，不能直接写 \(U^\top\)。该步骤本质上是同维度的 symmetry-adapted basis change，而不是降维；只有随后显式选择部分 PG irreps 时才发生 restriction / projection。

### 4.6.1 符号含义与来源

| 符号 | 含义 | 如何得到 |
| --- | --- | --- |
| \(K\) | 当前分支采用的、embedding 已固定的 crystallographic point group | 由 canonicalized structure、spglib symmetry operations 与 compatible-parent embedding metadata 确定 |
| \(g\in K\) | \(K\) 的一个群元素 | 对应 canonical frame 中的正交矩阵 \(R_g\)；涉及原子映射时还同时保存空间群平移 \(t_g\) |
| \(\widehat K\) | \(K\) 的全部 inequivalent irreps 构成的集合 | 来自固定 point-group convention / character table |
| \(\ell\) | O(3)/SO(3) angular order，标准 basis 的维数为 \(2\ell+1\) | 由 backbone feature layout 或目标张量的 harmonic decomposition 给定 |
| \(p\in\{+1,-1\}\) | O(3) irrep 在 inversion 下的 parity | 由 feature/target 的 polar、axial 与 tensor-product 类型确定 |
| \(m\) | standard spherical-harmonic basis 中的 component index，\(m=-\ell,\ldots,\ell\) | 固定 basis label，不学习 |
| \(Y_m^{(\ell,p)}\) | standard real spherical-harmonic / O(3)-irrep basis vector | 由选定的 harmonic normalization、axis 和 parity convention 固定 |
| \(\Gamma\) | point group \(K\) 的一个 irrep | 来自固定 convention 下的 character table / representation library |
| \(D^\Gamma(g)\) | \(g\) 在 irrep \(\Gamma\) 中的 representation matrix | 从 character-table-compatible irrep matrices / MultiPie 获得 |
| \(\chi^\Gamma(g)\) | \(\Gamma\) 的 character | \(\chi^\Gamma(g)=\operatorname{Tr}D^\Gamma(g)\) |
| \(d_\Gamma\) | \(\Gamma\) 的维数 | \(d_\Gamma=\chi^\Gamma(e)\) |
| \(m_{\ell p,\Gamma}\) | \(\Gamma\) 在 \(D^{(\ell,p)}\downarrow_K\) 中出现的次数 | 由 character inner product 计算 |
| \(a\) | repeated \(\Gamma\) copies 的编号 | \(a=1,\ldots,m_{\ell p,\Gamma}\)，通过确定性的 copy-order convention 固定 |
| \(\mu\) | irrep \(\Gamma\) 内部的 component index | \(\mu=1,\ldots,d_\Gamma\) |
| \(H_{\Gamma a\mu}^{(\ell,p)}\) | 按 \((\Gamma,a,\mu)\) 标记的 point-group harmonic basis vector | 由 \(U_K^{(\ell,p)}Y^{(\ell,p)}\) 得到 |
| \(c\) | 相同 \(D^{(\ell,p)}\) 的 learnable channel/copy index | 由网络宽度 \(n_{\ell,p}\) 给定；subduction 不混合该 index |
| \(n_{\ell,p}\) | 网络 feature 中 \(D^{(\ell,p)}\) 的 channel multiplicity | 由 architecture configuration 指定 |
| \(U_K^{(\ell,p)}\) | standard harmonic basis 到 PGH basis 的 subduction / basis-change matrix | 从群表示矩阵确定性构造，或按 PGEqNN 从 MultiPie/已核验表格读取；不是可学习参数 |
| \(P_{\Gamma;\mu\nu}^{(\ell,p)}\) | 从 \(D^{(\ell,p)}\) 空间抽取 \(\Gamma\) 型子空间的 matrix-valued projection operator | 由 \(D^\Gamma(g)\) 与 \(D^{(\ell,p)}(R_g)\) 的有限群求和得到 |

对于 proper 与 improper operations，先在 canonical Cartesian frame 中构造完整的 O(3) representation：

$$
D^{(\ell,p)}(R_g)
=
p^{(1-\det R_g)/2}
D^{(\ell)}\!\left((\det R_g)R_g\right),
$$

其中 \((\det R_g)R_g\in SO(3)\)，并定义：

$$
\chi^{(\ell,p)}(g)
=
\operatorname{Tr}D^{(\ell,p)}(R_g).
$$

这说明 character multiplicity 公式中的 \(\chi^{(\ell,p)}\) 不是额外数据标签，而是由 spglib 返回的 canonical-frame rotations、parity convention 和 Wigner-\(D\) implementation 直接计算。

记号上，\(e\) 表示群的 identity element，\(^*\) 表示复共轭，\({}^\top\) 表示 transpose，\({}^\dagger\) 表示 conjugate transpose；real orthonormal convention 下 \({}^\dagger={}^\top\)。

### 4.6.2 \(U_K^{(\ell,p)}\) 的确定性构造

首先必须把 spglib operations 与 irrep library 中的抽象群元素一一对齐。设 canonical lattice matrix 为 \(A\)，spglib 给出的 fractional-coordinate rotation 为 \(W_g\)，则在本文固定的 column-vector lattice convention 下：

$$
R_g
=
A W_g A^{-1}.
$$

若代码采用 row-vector lattice convention，则应使用相应的 transpose 形式，不能混用。利用已固定的 group embedding、矩阵乘法表和 tolerance，将每个 \(R_g\) 与 library 中同一个 \(g\) 的 \(D^\Gamma(g)\) 配对；仅按 conjugacy class 配对不足以确定多维 irrep 的 component matrices。

第一种实现与 PGEqNN 一致：在固定 point-group setting、real spherical-harmonic convention、irrep naming 和 component order 后，从 MultiPie 或预计算 PGH tables 读取 \(U_K^{(\ell,p)}\)。载入后必须验证前述 intertwining relation、unitarity 和 dimension sum。

若自行生成，可使用 finite-group projection operators：

$$
P_{\Gamma;\mu\nu}^{(\ell,p)}
=
\frac{d_\Gamma}{|K|}
\sum_{g\in K}
\left[D^\Gamma(g)\right]^*_{\mu\nu}
D^{(\ell,p)}(R_g).
$$

具体流程为：

1. 对每个 \(\Gamma\) 计算 \(m_{\ell p,\Gamma}\)，只处理 multiplicity 非零的 blocks；
2. 将 \(P_{\Gamma;\mu\nu}^{(\ell,p)}\) 作用于一组 deterministic seed vectors；
3. 使用 SVD/QR 提取其 image，并在 repeated-copy space 中正交化；
4. 按固定的 \((\Gamma,a,\mu)\) 顺序堆叠所得 basis vectors，形成 \(U_K^{(\ell,p)}\) 的 rows；
5. 固定每个 basis vector 的 sign/phase、degenerate-copy gauge 和 irrep component order；
6. 数值验证
   \[
   UDU^\dagger=\bigoplus_\Gamma(I_m\otimes D^\Gamma),
   \qquad
   UU^\dagger=U^\dagger U=I.
   \]

其中 SVD/QR 只用于离线生成固定 matrices，不参与训练。所有数据 split 和所有模型必须复用同一套 \(U\)、sign/phase 与 copy-order convention，否则不同 PG branches 的 coefficients 不能安全地 inverse-subduce 到同一公共空间。

---

## 4.7 Point-group tensor product

在 PG representation space 中，设 source、edge/filter 与 output irreps 分别为 \(\Gamma_s,\Gamma_e,\Gamma_o\)。其 fusion rule 为：

$$
\Gamma_s
\otimes
\Gamma_e
=
\bigoplus_{\Gamma_o\in\widehat K}
N^{\Gamma_o}_{\Gamma_s\Gamma_e}
\Gamma_o,
$$

其中 fusion multiplicity 为：

$$
N^{\Gamma_o}_{\Gamma_s\Gamma_e}
=
\frac{1}{|K|}
\sum_{g\in K}
\chi^{\Gamma_s}(g)
\chi^{\Gamma_e}(g)
\chi^{\Gamma_o}(g)^*.
$$

若 \(N^{\Gamma_o}_{\Gamma_s\Gamma_e}>0\)，使用

$$
\eta
=
1,\ldots,
N^{\Gamma_o}_{\Gamma_s\Gamma_e}
$$

标记重复的 fusion channels。定义固定的 finite-group Clebsch-Gordan intertwiner：

$$
C^{K;\Gamma_o,\eta}:
V_{\Gamma_s}\otimes V_{\Gamma_e}
\rightarrow
V_{\Gamma_o},
$$

满足：

$$
C^{K;\Gamma_o,\eta}
\left[
D^{\Gamma_s}(g)
\otimes
D^{\Gamma_e}(g)
\right]
=
D^{\Gamma_o}(g)
C^{K;\Gamma_o,\eta},
\qquad g\in K.
$$

对 \(x^{\Gamma_s}\in V_{\Gamma_s}\) 与 \(y^{\Gamma_e}\in V_{\Gamma_e}\)，一次无参数的 PG tensor product contraction 为：

$$
\left[
x^{\Gamma_s}
\otimes_K
y^{\Gamma_e}
\right]_{\mu_o}^{\Gamma_o,\eta}
=
\sum_{\mu_s=1}^{d_{\Gamma_s}}
\sum_{\mu_e=1}^{d_{\Gamma_e}}
C_{\mu_o;\mu_s\mu_e}^{K;\Gamma_o,\eta}
x_{\mu_s}^{\Gamma_s}
y_{\mu_e}^{\Gamma_e}.
$$

这里：

- \(\mu_s,\mu_e,\mu_o\) 分别是三个 irreps 内部的 component indices；
- \(\eta\) 区分同一 \(\Gamma_o\) 的 repeated fusion copies；
- \(C^{K;\Gamma_o,\eta}\) 是由 \(D^{\Gamma_s},D^{\Gamma_e},D^{\Gamma_o}\) 离线求解 intertwiner space、正交化并固定 gauge 后得到的常数，也可由经过 convention 核验的 finite-group CG library 读取；
- CG coefficients 不参与学习；可学习参数只能作用在 channel、copy、path multiplicity 等不会破坏 irrep component transformation law 的空间上。

因此 PG TP 的表示分量 contraction 由群论固定，而网络学习“哪些允许路径应具有多大权重”。与纯 O(3) TP 相比，它允许对不同 \(\Gamma\)、subduction copy \(a\) 与 fusion copy \(\eta\) 分配不同参数，从而形成更细粒度的 within-\(\ell\) parameterization。

---

## 4.8 Neumann principle 与 fixed subspace

对于目标 tensor representation \(V_T\)，物理允许空间为：

$$
\mathrm{Fix}_G(V_T)
=
\left\{
T\in V_T
\mid
\rho_T(g)T=T,
\ \forall g\in G
\right\}.
$$

在 point-group adapted basis 中，该空间对应 trivial representation \(A_1\) 的 copies。

因此 final readout 应 hard enforce：

$$
\hat T_G
\in
\mathrm{Fix}_G(V_T).
$$

---

## 4.9 Reynolds projection

任意预测 \(\hat T\) 都可以通过群平均投影到 fixed subspace：

$$
P_G(\hat T)
=
\frac{1}{|G|}
\sum_{g\in G}
\rho_T(g)\hat T.
$$

这是一个极强的 conceptual baseline，因为它可以在不修改 internal representation 的情况下确保输出满足 point-group symmetry。

本研究必须证明：

> internal PG specialization 的价值不仅是让输出“合法”，而是改善 representation learning 和 sample efficiency。

---

# 5. Our Method

## 5.1 总体思想

模型采用以下 hierarchy：

$$
\boxed{
\text{Universal O(3) representation}
\rightarrow
\text{task/symmetry-conditioned O(3) adaptation}
\rightarrow
\text{point-group representation}
\rightarrow
\text{symmetry-allowed tensor}
}
$$

核心结构为：

1. spglib canonicalization；
2. pretrained O(3)-equivariant backbone；
3. 1–2 个 shared O(3) experts，其 TP backend 由参数在 full O(3) 与 SO(2)-reduced 实现间切换；
4. 确定当前点群 \(G_x\)、物理兼容的父群集合及其嵌入关系；
5. 根据当前结构对各父群操作的连续 parent-group residual 构造 hierarchical gates；
6. 对每个激活群 \(K\in\mathcal A(x)\) 运行完整分支：\(O(3)\rightarrow K\) subduction、PG tensor product、PG equivariant MLP 与 PG-equivariant norm；global 任务再 pooling 并进入 \(\mathrm{Fix}_K(V_T)\) readout，BEC 则跳过 pooling 并对每个节点使用同一个 equivariant tensor readout；
7. 将各分支输出 inverse-subduce 到统一 O(3) tensor-irrep space；
8. 使用满足父群极限条件的权重做加权平均；
9. 可选地执行一个融合后的共享 O(3)-equivariant readout；
10. 统一执行 irrep-to-Cartesian；
11. de-canonicalization 回原始输入 frame。

这里的“父群”不是当前点群的所有抽象 supergroups，而是具有明确 orientation / setting embedding、并对应可实现结构畸变路径的 compatible parent groups。

---

## 5.2 架构图

```mermaid
flowchart TD
    X["Relaxed crystal x"] --> C["spglib canonicalization"]
    C --> G["Current point group Gx"]
    C --> A["Compatible parent groups and embeddings"]
    C --> RES["Parent-group operation residuals r_K(x)"]
    C --> Q["Cartesian frame transform Qx"]
    C --> XC["Canonicalized crystal x_bar"]

    XC --> B["Pretrained O(3)-equivariant backbone"]
    B --> H["Universal O(3) features h"]

    H --> SH["Shared O(3) adaptation: full_o3 or so2_reduced"]
    H -. optional residual .-> BASE["Shared O(3) branch input"]
    SH --> BASE

    G --> SET["Active set A(x): current group plus compatible parents"]
    A --> SET
    SET --> ZK["For each K in A(x): branch input z_K = shared"]
    BASE --> ZK

    ZK --> SUBK["Subduction O(3) to K"]
    SUBK --> PGK["PG_K TP message passing and per-node EqMLP"]
    PGK --> NORMK["PG-equivariant norm per node"]
    NORMK --> SCOPE{"output_scope"}
    SCOPE -->|global: dielectric / elastic| POOLK["Permutation-invariant global pool"]
    POOLK --> FIXK["Readout in Fix_K(V_T)"]
    SCOPE -->|site: BEC| NODEK["Shared node-wise equivariant tensor readout; no pool"]
    NODEK --> LIFT
    FIXK --> LIFT["Inverse subduction to common O(3) tensor-irrep space"]

    RES --> W["Continuous hierarchical weights w_K"]
    SET --> W
    W --> MIX["Weighted average across full branches"]
    LIFT --> MIX
    MIX --> O3RO["Shared O(3)-equivariant readout (optional)"]
    O3RO --> CART["Irrep to Cartesian tensor"]
    Q --> DEC["De-canonicalization"]
    CART --> DEC
    DEC --> OUT["Tensor in original Cartesian frame"]
```

---

## 5.3 Step 1：Canonicalization 与 symmetry metadata

输入 relaxed structure \(x\)，使用 spglib 获取：

- standardized / idealized structure \(\bar x\)；
- current point-group symbol \(G_x\)；
- compatible parent groups、具体 subgroup embeddings 与 reference settings；
- normalized parent-group operation residuals；
- symmetry operations；
- transformation matrix / standard setting；
- Cartesian rigid rotation；
- atom index mapping，以及每个空间群操作诱导的 site permutation \(\pi_g\) 等。

抽象写为：

$$
C(x)
=
(\bar x,G_x,\mathcal A(x),r(x),Q_x,\mathcal S),
$$

其中 \(\mathcal A(x)\) 表示当前点群与物理兼容父群构成的激活集合，\(r(x)=\{r_K(x)\}_{K\in\mathcal A(x)}\) 表示当前结构相对各父群的连续 symmetry residuals，\(Q_x\) 表示从原始 Cartesian frame 到 canonical frame 所需记录的正交变换信息，\(\mathcal S\) 保存各激活群的实际 symmetry representation 与 embedding metadata。

需要特别注意：

> **lattice change-of-basis matrix 与 Cartesian tensor rotation matrix 不是同一个对象。**

spglib 的标准化也并非严格唯一，可能存在 group-equivalent choices。

因此模型仍需对 canonical frame 中剩余的 group-equivalent choices 保持一致性。

PG-equivariant stage 正好提供这一保证。

---

## 5.4 Step 2：Pretrained O(3)-equivariant backbone

在 canonicalized structure 上运行通用预训练 backbone：

$$
h
=
B_{\Theta}(\bar x),
$$

其中：

$$
h
=
\bigoplus_{\ell,p}
h^{(\ell,p)}.
$$

Backbone 的角色是提供：

1. universal chemical environment knowledge；
2. local / many-body geometry；
3. directional equivariant information；
4. 跨 point groups 的统一 feature interface。

第一阶段优先选择已有成熟 pretrained O(3)/E(3) atomistic model。

具体 backbone 可根据：

- pretrained weights；
- license；
- 元素覆盖；
- feature extraction interface；
- \(\ell_{\max}\)；
- computational cost；

决定。

论文的 scientific contribution 不绑定某一个 backbone。

---

## 5.5 Step 3：Shared O(3) experts

设置：

$$
N_s=1\text{ or }2
$$

个 shared experts，所有 configurations 均激活。

第 \(k\) 个 shared expert：

$$
s_{k,i}
=
E_{\theta_k}^{\mathrm{shared}}(h)_i
=
M^{O(3)}_{\theta_k}
\left(
\sum_{j\in\mathcal N(i)}
T^{O(3)}_{\theta_k}
\left(h_j,e_{ij};
w_{k,\pi}\!\left([h_j^{\mathrm{inv}}\mid h_i^{\mathrm{inv}}]\right)
;\texttt{shared\_tp\_backend}
\right)
\right).
$$

其中 \(T^{O(3)}_{\theta_k}\) 根据 `shared_tp_backend` 选择 full O(3) CG TP 或 edge-frame SO(2)-reduced TP。即：

$$
\boxed{
\{Full\ O(3)\ TP\ \mid\ SO(2)\text{-}reduced\ TP\}
\ +\ NeighborAggregation
\rightarrow
O(3)\ EquivariantMLP\ per\ node
}
$$

其中每条 TP path 的权重由源节点与目标节点 invariant features 的拼接产生：

$$
w_{k,\pi}\!\left([h_{\mathrm{src}}\mid h_{\mathrm{tgt}}]\right)
=
\operatorname{MLP}_{k,\pi}
\!\left(
[h_{\mathrm{src}}^{\mathrm{inv}}\mid h_{\mathrm{tgt}}^{\mathrm{inv}}]
\right).
$$

因此 shared adaptation 的输出仍是一组 node-wise O(3)-equivariant features；TP 负责图上的消息构造与聚合，EqMLP 只对各节点的聚合结果分别处理。

主设置默认使用 `full_o3`；`so2_reduced` 作为参数匹配的 efficiency branch。两者不是同时激活的 mixture branches，而是同一 shared expert 的互斥实现选项。

注意：

> **Shared expert 到这里结束，不包含 PG tensor product / PG MLP。**

其任务是从通用 pretrained features 中形成跨 point groups 可迁移的：

> **tensor-task-aware O(3) representation**

---

## 5.6 Step 4：Current-plus-parent active set 与连续 gates

对当前 structure 的 point group \(G_x\)，构造：

$$
\mathcal A(x)
=
\{G_x\}
\cup
\{K\mid K\text{ 是 }G_x\text{ 的物理兼容父群}\}.
$$

兼容性必须包含具体的 subgroup embedding、canonical orientation 与结构 correspondence，而不能只根据抽象 point-group symbol 判断。

候选 parent DAG 应由结构 family / prototype、space-group subgroup relation 与原子 correspondence 预先确定，并在所研究的连续路径邻域内保持不变；不得使用 \(r_K<\texttt{symprec}\) 之类的 hard threshold 动态筛选父群。若工程上允许 candidate set 改变，则新加入或移除分支在切换边界的权重必须严格为零。

对于一条 symmetry-breaking edge：

$$
K_{i-1}\rightarrow K_i,
\qquad
K_i\subset K_{i-1},
$$

令 canonical periodic crystal 写为：

$$
x
=
\left(
A,
\{(s_i,Z_i)\}_{i=1}^{N}
\right),
$$

其中 \(A=[a_1,a_2,a_3]\in\mathbb R^{3\times3}\) 的列为 Cartesian lattice vectors，\(s_i\in[0,1)^3\) 为 fractional coordinates，\(Z_i\) 为元素类型。父群 \(H=K_{i-1}\) 中的 embedded space-group operation 记为：

$$
g=(W_g,t_g),
\qquad
s\mapsto W_gs+t_g,
$$

其中 \(W_g\in GL(3,\mathbb Z)\)，\(t_g\) 为 fractional translation。若实现接口使用 row-vector lattice convention，必须在边界处显式 transpose，不能混用下述 column-vector 公式。

### 5.6.1 Lattice metric residual

定义 Gram matrix：

$$
M=A^\top A.
$$

精确 lattice symmetry 满足：

$$
W_g^\top M W_g=M.
$$

因此定义无量纲 lattice residual：

$$
r_{g,\mathrm{cell}}^2(x)
=
\frac{
\left\|
W_g^\top M W_g-M
\right\|_F^2
}{
\|M\|_F^2+\epsilon
}.
$$

该项确保 cubic \(\rightarrow\) tetragonal / orthorhombic strain 即使不改变 fractional atomic coordinates，也会产生非零 residual。例如 \(A=\operatorname{diag}(a,b,c)\) 且 \(W_g\) 交换前两轴时，该 residual 随 \((a^2-b^2)^2\) 增长。

### 5.6.2 Periodic atomic-position residual

精确 symmetry 要求对每个原子 \(i\)，存在同元素原子 \(\pi_g(i)\) 与 lattice translation \(n_i\in\mathbb Z^3\)，使：

$$
W_gs_i+t_g
=
s_{\pi_g(i)}+n_i.
$$

令 characteristic length 为：

$$
\ell_x
=
\left(
\frac{|\det A|}{N}
\right)^{1/3}.
$$

若 parent embedding 已给出固定 atom correspondence \(\pi_g\)，定义：

$$
r_{g,\mathrm{pos}}^2(x)
=
\frac{1}{N\ell_x^2}
\sum_{i=1}^{N}
\min_{n_i\in\mathbb Z^3}
\left\|
A\left(
W_gs_i+t_g-s_{\pi_g(i)}-n_i
\right)
\right\|_2^2.
$$

固定 \(\pi_g\) 是优先方案，因为它保持 parent-to-child 路径上的 correspondence 稳定。若 mapping 未知，则使用 species-preserving assignment：

$$
r_{g,\mathrm{pos}}^2(x)
=
\frac{1}{N\ell_x^2}
\min_{\pi\in\Pi_Z}
\sum_{i=1}^{N}
\min_{n_i\in\mathbb Z^3}
\left\|
A\left(
W_gs_i+t_g-s_{\pi(i)}-n_i
\right)
\right\|_2^2,
$$

其中 \(\Pi_Z\) 只允许相同元素之间的 permutation。工程上可由 periodic pairwise-distance cost matrix 加 Hungarian assignment 实现。对 skewed cell，\(n_i\) 必须解 closest-lattice-vector problem；不能无条件对 fractional difference 逐分量 `round`。最小 assignment 或最近镜像发生切换时，距离值保持连续但梯度可能不连续；需要结构导数时应优先固定 parent mapping，或另行研究 smooth assignment。

### 5.6.3 Operation residual 与 group aggregation

组合 operation residual：

$$
d(gx,x)^2
=
r_g(x)^2
=
\lambda_{\mathrm{cell}}\,
r_{g,\mathrm{cell}}^2(x)
+
\lambda_{\mathrm{pos}}\,
r_{g,\mathrm{pos}}^2(x),
$$

其中：

$$
\lambda_{\mathrm{cell}},\lambda_{\mathrm{pos}}\ge0,
\qquad
\lambda_{\mathrm{cell}}+\lambda_{\mathrm{pos}}=1.
$$

两部分均已无量纲化；第一版默认可取各 \(1/2\)，并将二者比例作为 ablation。然后定义 normalized parent-group residual：

$$
r_H^2(x)
=
\frac{1}{|H|}
\sum_{g\in H}
d(gx,x)^2.
$$

单位元贡献恒为零。可保留它以保持统一群平均，也可使用 \(1/(|H|-1)\) 对非单位操作平均；两种 convention 不得混用，并需分别 calibration \(\sigma_H\)。如果当前结构严格保留父群 \(H\)，则 \(r_H(x)=0\)；偏离父群越强，\(r_H(x)\) 越大。

这里不得在 \(gx\) 与 \(x\) 之间再次最小化任意 global O(3) alignment，否则可用 \(g^{-1}\) 抵消被测试的 symmetry operation，使 residual 失去意义。距离应在同一个 canonical / transported frame 中计算；当输入整体变换为 \(Rx\) 时，父群 embedding 同步共轭为 \(RHR^{-1}\)，从而：

$$
r_H(Rx)=r_H(x),
\qquad R\in O(3).
$$

对于 nonsymmorphic operations，spglib 的 rotation 与 translation 必须按相同 index 成对使用。对于 cell multiplication、translation-symmetry change 或 Wyckoff splitting，应先建立共同 supercell 与明确的 parent space-group / atom mapping；仅使用抽象 point-group rotation 或按原子 index 比较是不充分的。

### 5.6.4 Residual-to-gate map

对 edge \(H\rightarrow K_i\)，定义 symmetry-breaking gate：

$$
a_{H\rightarrow K_i}(x)
=
1-
\exp\!\left(
-\frac{r_H(x)^2}{\sigma_{H\rightarrow K_i}^2}
\right).
$$

同时定义 parent-preservation score：

$$
s_{H\rightarrow K_i}(x)
=
\exp\!\left(
-\frac{r_H(x)^2}{\sigma_{H\rightarrow K_i}^2}
\right)
=
1-a_{H\rightarrow K_i}(x).
$$

这里 \(a_{H\rightarrow K_i}\) 表示“父群被破坏后，子群分支应开启多少”，不能直接解释为父群 expert 的权重。父群恢复时：

$$
r_H\to0
\quad\Longrightarrow\quad
a_{H\rightarrow K_i}\to0,
\qquad
s_{H\rightarrow K_i}\to1.
$$

因此相应子群 branch 自动关闭，父群 branch 保留。几何量固定定义 \(r_H\)，模型最多只学习受约束的 \(\sigma_{H\rightarrow K_i}>0\) 或单调 calibration，不使用普通 learned softmax 取代这一边界条件。

若同一父群可沿多个 distortion directions 降到不同子群，可以额外把 \(x-P_Hx\) 分解到 parent-group irreps，作为 path compatibility 或解释性特征；但第一版 branch strength 的主尺度由上述 operation-level parent residual 决定。

---

## 5.7 Step 5：Hierarchical convex weights

先考虑完整 DAG 中任一条 embedded subgroup path：

$$
K_0\supset K_1\supset\cdots\supset K_L=G_x,
$$

使用 stick-breaking 形式把 edge gates 转换为 branch weights：

$$
w_0=1-a_1,
$$

$$
w_k
=
\left(\prod_{j=1}^{k}a_j\right)
(1-a_{k+1}),
\qquad 0<k<L,
$$

$$
w_L
=
\prod_{j=1}^{L}a_j.
$$

因此：

$$
w_k\ge0,
\qquad
\sum_{k=0}^{L}w_k=1.
$$

当任一父群 \(K_{i-1}\) 的 symmetry 恢复时，\(a_i\to0\)，所有更深子群分支的权重自动趋于零。这个边界条件，而非“权重和为 1”本身，提供跨 symmetry boundary 的连续性。

对于完整 subgroup DAG，令 \(\mathcal P(x)\) 为所有物理兼容、具有明确 embedding 的 parent-to-current paths。每条 path 内使用上述 stick-breaking 规则，并给 path 一个连续、归一化的几何 compatibility prior \(\pi_p(x)\)。同一群 \(K\) 可能出现在多条 path 中，其未归一化权重为：

$$
\beta_K(x)
=
\sum_{p\in\mathcal P(x):K\in p}
\pi_p(x)w_{K\mid p}(x),
$$

最终取：

$$
w_K(x)
=
\frac{\beta_K(x)}{
\sum_{J\in\mathcal A(x)}\beta_J(x)
}.
$$

这样当前群与所有 compatible parents 均运行且可同时具有非零贡献，同时避免同一父群因多条路径被重复计算。所有 path priors、edge gates 与 normalization 都必须连续；当某条 parent-to-child edge 恢复 symmetry 时，所有经过该 edge 的更深路径贡献必须趋于零。第一版可以只支持 benchmark 中预先枚举的 embedded DAG paths，但不能把 DAG 偷换成 one-hot parent selection。

---

## 5.8 Step 6：每个激活群运行完整分支

先形成所有分支共享的 O(3) task representation：

$$
z_{\mathrm{shared},i}
=
\alpha_0h_{\mathrm{skip},i}
+
\sum_{k=1}^{N_s}s_{k,i}.
$$

对每个激活群 \(K\in\mathcal A(x)\)，直接以共享 task representation 作为分支输入：

$$
z_{K,i}=z_{\mathrm{shared},i}.
$$

随后该分支独立执行 subduction：

$$
z_{K,i}^{PG}
=
\mathcal U_K z_{K,i},
$$

其中：

$$
\mathcal U_K
=
\bigoplus_{\ell,p}
\left(
I_{n_{\ell,p}}
\otimes
U_K^{(\ell,p)}
\right),
$$

其中 \(n_{\ell,p}\) 是该 O(3) irrep 的 channel multiplicity；\(U_K^{(\ell,p)}\) 只作用于 \(m\)-components，不混合 learnable channels。完整 PG layout 保留全部 \((\Gamma,a,\mu)\) 分量，因此：

$$
\mathcal U_K^{-1}
=
\mathcal U_K^\dagger,
\qquad
\mathcal U_K^\dagger\mathcal U_K
=I.
$$

在本文采用的 real orthonormal PGH convention 下可实现为 \(\mathcal U_K^\top\)。如果某个 ablation 只保留部分 PG irreps，则该操作成为投影而不再可逆，不得称为 inverse subduction。

对于每个 \((\ell,p)\)：

$$
D^{(\ell,p)}
\downarrow_K
=
\bigoplus_{\Gamma}
m_{\ell p,\Gamma}\Gamma.
$$

在该步骤之后，feature label 从粗粒度的：

$$
(\ell,p,c)
$$

变为更细粒度的：

$$
(\ell,p,\Gamma,a,c).
$$

---

## 5.9 Step 7：Point-group tensor product

对 PG-adapted node/edge features 在同一 PBC graph 上执行 finite-group tensor-product message passing。为压缩指标，定义一个 PGH block label：

$$
\lambda
=
(\ell,p,\Gamma,a),
$$

其中只有 \(\Gamma\) 决定该 block 在 \(K\) 下的 transformation law，\((\ell,p,a)\) 记录它来自哪个 O(3) order、parity 与 repeated subduction copy，并允许网络对这些来源分配不同参数。

若 edge features 直接由几何构造，可写为：

$$
e^{K,\lambda_e}_{ij,q,\mu_e}
=
B_q(r_{ij})
H_{\Gamma_e a_e\mu_e}^{(\ell_e,p_e)}
(\widehat r_{ij}),
$$

其中 \(B_q(r)\) 是固定 radial basis，\(q\) 是 radial channel。若 backbone 已经返回 equivariant edge attributes，则直接对其做 \(\mathcal U_K\) subduction，不重复构造上述 edge basis。

一条完整的 PG TP path 记为：

$$
\pi
=
\left(
\lambda_s,c_s;
\lambda_e,q;
\lambda_o,c_o;
\eta
\right),
$$

且仅当

$$
N_{\Gamma_s\Gamma_e}^{\Gamma_o}>0
$$

时才枚举该 path。其固定、无参数的 CG contraction 为：

$$
\tau^{K,\pi}_{ij,\mu_o}
=
\sum_{\mu_s,\mu_e}
C_{\mu_o;\mu_s\mu_e}^{K;\Gamma_o,\eta}
z^{K,\lambda_s}_{j,c_s,\mu_s}
e^{K,\lambda_e}_{ij,q,\mu_e}.
$$

### 5.9.1 Endpoint-conditioned learnable path weights

当前框架不直接照搬 PGEqNN 中只依赖 \(r_{ij}\) 的 learnable radial filter。为了使同一个 router trunk 能在不同 \(K\) 之间共享，主设置从 subduction 之前的 \(z_{\mathrm{shared}}\) 中读取固定宽度的 O(3)-invariant scalar channels：

$$
s_{ij}
=
\left[
\left(z_{\mathrm{shared},j}\right)^{(0,+)}
\mid
\left(z_{\mathrm{shared},i}\right)^{(0,+)}
\right],
$$

其中 \((0,+)\) 表示真正的 O(3)-trivial channels。由于 \(s_{ij}\) 的维度与 \(K\) 无关，并且在 O(3) 下不变，普通 MLP 可以生成每条允许路径的动态标量权重。先写成未因式分解的概念形式：

$$
\omega^K_{ij,\pi}
=
\left[
\operatorname{MLP}_{\theta_K}^{K}
\left(s_{ij}\right)
\right]_\pi.
$$

因此，和 PGEqNN 的 learnable radial filter 对应的有效 path filter 可写为：

$$
F^{K,\pi}_{ij}
=
\omega^K_{ij,\pi}(s_{ij})
B_{q(\pi)}(r_{ij}).
$$

这里 \(q(\pi)\) 是 path \(\pi\) 携带的 radial-channel index。径向基 \(B_q\) 固定，而 \(\omega\) 可学习；不同的 \(q\) 被视为不同 paths，所以对这些 paths 的加权本身就能学习径向组合。主设置仍严格采用 \(\operatorname{MLP}[h_{\mathrm{src}}^{\mathrm{inv}}\mid h_{\mathrm{tgt}}^{\mathrm{inv}}]\)，不再把 \(r_{ij}\) 直接输入 router，以免改变前文约定。

限制 router 只读取不变 endpoint features 是保证 \(\omega^K_{ij,\pi}\) 为标量的充分且易验证的实现。若希望利用由高阶 \(\ell\) subduce 出的 PG-\(A_1\) channels，则必须先为每个 \(K\) 构造固定宽度 summary，并使用 group-specific input projection；该版本作为 ablation，不作为跨 PG 共享 router 的默认输入。

于是完整 edge message 为：

$$
m^{K,\lambda_o}_{i\leftarrow j,c_o,\mu_o}
=
\sum_{\pi\mapsto(\lambda_o,c_o)}
\omega^K_{ij,\pi}
\tau^{K,\pi}_{ij,\mu_o}.
$$

\(\omega^K_{ij,\pi}\) 可以依赖 \(\Gamma_s,\Gamma_e,\Gamma_o\)、subduction copies、input/output channels 与 fusion copy \(\eta\)，但必须是标量，并且不能依赖 irrep component \(\mu_o\)。否则会破坏 Schur structure 和 \(K\)-equivariance。

### 5.9.2 与当前多点群框架适配的参数化

若为每个 \(K\) 的每条 path 分别放置完整 MLP，参数量与 active PG 数会迅速增大，并加剧 rare-PG overfitting。主实现采用 shared router trunk 与 group-specific low-rank path head：

$$
\phi_{ij}
=
\operatorname{MLP}_{\theta_{\mathrm{route}}}
\left(s_{ij}\right)
\in
\mathbb R^{r_{\mathrm{route}}},
$$

$$
\boldsymbol\omega^K_{ij}
=
b^K
+
A^K\phi_{ij},
\qquad
A^K\in
\mathbb R^{|\mathcal P_K|\times r_{\mathrm{route}}},
$$

其中 \(\mathcal P_K\) 是群 \(K\) 在当前 layer 中枚举出的全部 allowed paths，\(r_{\mathrm{route}}\) 是由配置指定的共享 router bottleneck width。共享参数 \(\theta_{\mathrm{route}}\) 学习跨 PG 可迁移的 endpoint conditioning；\(A^K,b^K\) 是轻量的 PG-specific learnable parameters，用于将公共 latent router features 映射到该群自己的 path set。完整的 per-group/per-path MLP 仅作为 capacity ablation。

随后加入一个 PG-equivariant self interaction 并在目标节点聚合：

$$
\tilde z_{K,i}^{PG}
=
W_{\mathrm{self}}^K z_{K,i}^{PG}
+
\frac{1}{\sqrt{|\mathcal N(i)|}}
\sum_{j\in\mathcal N(i)}
m^K_{i\leftarrow j},
$$

其中：

$$
W_{\mathrm{self}}^K
=
\bigoplus_{\Gamma}
\left(
W_{\Gamma,\mathrm{mult}}^K
\otimes
I_{d_\Gamma}
\right).
$$

\(W_{\Gamma,\mathrm{mult}}^K\) 只在具有相同 \(\Gamma\) 的 channel/copy multiplicity space 中学习 mixing；\(I_{d_\Gamma}\) 保证同一 irrep 内的 components 共享权重。

省略 layer superscript 后，这一层新增的 trainable parameter count 为：

$$
N_{\mathrm{PGTP}}
=
N(\theta_{\mathrm{route}})
+
\sum_K
\left(
|\mathcal P_K|r_{\mathrm{route}}
+
|\mathcal P_K|
+
\sum_\Gamma N(W^K_{\Gamma,\mathrm{mult}})
\right).
$$

单个样本只激活 \(K\in\mathcal A(x)\) 的 group-specific heads 与 self interactions；\(\theta_{\mathrm{route}}\) 只计算一次公共 edge latent，随后供所有 active branches 复用。因而引入可学习参数是必要的，但不应把 CG coefficients 本身设为可学习参数。

### 5.9.3 固定量与可学习量

| 类别 | 符号 | 作用 |
| --- | --- | --- |
| 固定群论量 | \(U_K^{(\ell,p)}\)、\(C^{K;\Gamma_o,\eta}\)、allowed path set \(\mathcal P_K\) | basis change、PG-CG contraction 与 path legality |
| 固定图/几何量 | edge index、periodic image、\(r_{ij},\widehat r_{ij},B_q(r_{ij})\) | PBC neighbor relation 与 edge geometry |
| 共享可学习量 | \(\theta_{\mathrm{route}}\) | 从 endpoint trivial features 提取跨 PG 共用的 routing representation |
| PG-specific 可学习量 | \(A^K,b^K,W_{\Gamma,\mathrm{mult}}^K\) | path weighting 与同-irrep multiplicity mixing |
| 后续可学习量 | PG EqMLP、PG norm 的 scalar gains、task readout | node-wise nonlinear adaptation、normalization 与输出 |

PGEqNN Eq. (3) 中的 CG contraction 与 learnable radial filter 分别对应这里的固定 \(C\) 与可学习 path weighting。本文的适配点是：保留有限群 intertwiner，使用端点 invariant features 产生动态 path weights，并通过 shared-low-rank parameterization 支持 current group 与多个 compatible-parent branches。

### 5.9.4 \(\alpha\otimes\beta\rightarrow\gamma\) 的代码表示与维护

在实现中令

$$
\alpha=\Gamma_s,
\qquad
\beta=\Gamma_e,
\qquad
\gamma=\Gamma_o.
$$

三者不是 feature tensor 本身，而是群 \(K\) 内的 **irrep IDs**。同一个 `A1`、`E` 等名称在不同点群中不是同一个对象，因此 irrep 的全局标识必须至少是：

$$
\texttt{IrrepKey}=(\texttt{group\_id},\texttt{irrep\_id}),
$$

不能只保存 irrep name。每个点群维护一个只读的 `PointGroupSpec`：

```python
PointGroupSpec(
    group_id,
    elements, multiplication_table,
    irreps={irrep_id: IrrepSpec(dim, representation_matrices, characters)},
    fusion={(alpha, beta): [(gamma, multiplicity), ...]},
    cg={(alpha, beta, gamma, eta): Tensor[d_gamma, d_alpha, d_beta]},
    trivial_irrep_id,
    convention_id, gauge_checksum,
)
```

其中 `fusion[(alpha, beta)]` 就是 \(\alpha\otimes\beta\rightarrow\gamma\) 的唯一 legality source；`cg[...]` 保存每个 fusion copy \(\eta\) 的固定 contraction tensor，并注册为 non-trainable buffer。`fusion` 不应由训练过程动态修改，而应由 character product 离线生成，或从与表示矩阵采用同一 convention 的表格载入。

PG feature space 在数学上维护为 block direct sum：

$$
\mathcal F_K
=
\bigoplus_b
\left(
\mathbb R^{n_b}
\otimes
V_{\Gamma_b}
\right),
$$

其中 \(b=(\ell,p,\Gamma,a)\)，\(n_b\) 是 learnable channel multiplicity。推荐的实际存储是 block dictionary：

```python
PGBlockKey = (ell, parity, irrep_id, subduction_copy)
features[PGBlockKey].shape == [num_nodes, num_channels, irrep_dim]
edges[PGBlockKey].shape    == [num_edges, num_radial_channels,
                               num_channels, irrep_dim]
```

最后一个轴才是 irrep component \(\mu\)，它由 \(D^\Gamma(g)\) 作用；`num_channels` 与 `subduction_copy` 都属于 multiplicity information，不应被 \(D^\Gamma(g)\) 混合。也就是说，代码中的一个 irrep 空间并不需要为每个向量实例化一个抽象对象，而由 `PGBlockKey + IrrepSpec + 最后一维坐标` 共同表达。

初始化 PG-TP layer 时，根据输入、edge 与期望输出 layouts 枚举并冻结 path table：

```python
TPPathSpec(
    path_id,
    src_block, edge_block, out_block,
    src_channel, out_channel,
    radial_channel_q, fusion_copy_eta,
    cg_key=(alpha, beta, gamma, eta),
    weight_offset,
)
```

枚举规则为：读取 `src_block.irrep_id = alpha` 与 `edge_block.irrep_id = beta`，查询 `fusion[(alpha, beta)]`；只有期望 output layout 中存在相应 \(\gamma\) 时，才为每个 \(\eta=1,\ldots,N_{\alpha\beta}^{\gamma}\)、径向通道和输入/输出 channel 创建 path。将所有 paths 按

$$
(\alpha,\beta,\gamma,\eta,
\lambda_s,\lambda_e,\lambda_o,
q,c_s,c_o)
$$

确定性排序后分配 `path_id` 与 `weight_offset`。于是 router 输出向量的第 `weight_offset` 项始终对应同一条 path；该 path table、group convention 与 checksum 必须随 checkpoint 保存。

运行时只执行：

```python
cg = group_spec.cg[path.cg_key]          # buffer, no gradient
w  = path_weights[:, path.weight_offset] # learned scalar per edge
out[path.out_block] += w * contract(cg, src[path.src_block],
                                    edge[path.edge_block])
```

CG tensor 的 basis/gauge 必须和 `IrrepSpec.representation_matrices` 以及 subduction matrix \(U_K^{(\ell,p)}\) 完全一致；仅仅 irrep 名称与维数相同并不足够。载入每个 `PointGroupSpec` 时至少执行四类自动测试：

1. fusion multiplicity 与 character inner product 一致；
2. 每个 CG tensor 满足 intertwiner equation，并在 fusion-copy space 中正交归一；
3. 随机检查 \(\operatorname{TP}(D^\alpha x,D^\beta y)=D^\gamma\operatorname{TP}(x,y)\)；
4. PG feature 的 pack/unpack 以及 subduction/inverse-subduction round trip 在容差内成立。

因此，\(\alpha\otimes\beta\rightarrow\gamma\) 应由 **immutable group registry + deterministic path table** 维护；训练只更新与 `weight_offset` 对应的标量 path weights 和 multiplicity-space maps，不能改变 fusion rule、irrep component ordering 或 CG gauge。

### 5.9.5 晶体学库的复用边界

PGEqNN 使用 spglib / pymatgen 完成结构标准化，使用 pymatgen 的 CrystalNN 构图，并使用 MultiPie 提供 crystallographic point-group harmonics matrices。本文复用这些成熟工具，但不让它们直接决定训练时的动态表示布局。推荐职责划分如下：

| 组件 | 本文中的职责 | 执行阶段 |
| --- | --- | --- |
| [spglib](https://spglib.readthedocs.io/en/stable/python-interface.html) | space/point-group identification、标准胞、对称操作、Wyckoff/equivalent-site mapping 与输入到标准胞的变换 | dataset preprocessing；每个结构一次 |
| [pymatgen](https://pymatgen.org/) | structure I/O、坐标/晶格对象及 tensor metadata；CrystalNN 仅用于复现 PGEqNN graph 或 graph ablation | dataset preprocessing |
| [MultiPie](https://github.com/CMT-MU/MultiPie) | crystallographic PG irreps、characters/product tables、symmetry-adapted multipole/PGH basis 与候选 \(U_K^{(\ell,p)}\) | offline group-registry generation |
| e3nn / backbone implementation | O(3) irreps、real spherical harmonics、O(3) feature tensors 与 shared adaptation | differentiable forward |
| 本文的 PG registry / PG-TP code | convention conversion、finite-group CG construction、path enumeration、PyTorch buffer packing 与 validation | 初始化和 differentiable forward |

具体数据流为：

$$
\boxed{
\text{spglib canonicalization}
\rightarrow
\text{MultiPie-assisted offline PG tables}
\rightarrow
\text{validated immutable registry}
\rightarrow
\text{PyTorch/e3nn forward}
}.
$$

#### spglib：唯一的结构标准化真源

主实现只允许一个 canonicalization source。推荐直接使用 spglib，并锁定 `symprec`、`angle_tolerance`、`to_primitive` 与 `no_idealize`。对每个样本保存：

```python
SymmetryRecord(
    spglib_version,
    hall_number, spacegroup_number, pointgroup_symbol,
    transformation_matrix, origin_shift, std_rotation_matrix,
    rotations, translations,
    mapping_to_primitive, std_mapping_to_primitive,
    equivalent_atoms, wyckoffs,
    symprec, angle_tolerance, to_primitive, no_idealize,
)
```

pymatgen 的 `SpacegroupAnalyzer` 底层同样使用 spglib；若采用其接口，不能再对同一结构独立运行第二套 standardization 并混合两次结果。canonical atom ordering、tensor rotation \(Q_x\) 和所有跨胞 edge image 必须从同一份 `SymmetryRecord` 派生。

#### MultiPie：离线数据源，而不是训练时依赖

MultiPie 可用于取得或辅助构造 \(D^\Gamma(g)\)、\(\chi^\Gamma(g)\)、harmonic decomposition、product table 与 PGH/subduction matrices。PGEqNN 明确使用它提供 PGH matrices；但本文不预设其公共 API 会直接返回符合当前 convention 的全部 finite-group CG tensors。因此：

1. 从 MultiPie 导出候选 group/irrep/PGH 数据；
2. 转换到本文固定的 real-harmonic、parity、operation-order 与 component-order convention；
3. 由转换后的 \(D^\Gamma(g)\) 自行求解 intertwiner space，生成 \(C^{K;\gamma,\eta}\)；
4. 运行 5.9.4 中的 fusion、intertwining、equivariance 与 round-trip tests；
5. 将通过验证的数据序列化，训练时只加载固定 arrays。

推荐缓存格式为：

```text
group_registry/
  <group_id>/
    irreps.npz
    subduction_l0_lmax.npz
    fusion.json
    cg.npz
    convention.json
    checksums.json
```

`convention.json` 至少记录 MultiPie version、group setting、operation ordering、Cartesian-axis convention、real/complex harmonics、parity rule、irrep component ordering、repeated-copy ordering 与 CG gauge。版本或 checksum 不同的 registry 不允许与旧 checkpoint 静默混用。

#### pymatgen / CrystalNN：只作为可控图构造选项

CrystalNN 可用于严格复现 PGEqNN 的 Wyckoff-graph / neighbor-selection setting，但主模型优先沿用 pretrained backbone 的 PBC cutoff graph，原因是：

- 避免预训练和下游使用不同 graph semantics；
- radial-cutoff graph 更易保持 node/edge permutation closure；
- CrystalNN 的离散 neighbor selection 可能沿 symmetry-breaking path 改变拓扑，从而给 continuity experiment 引入额外 jump。

因此图构造由配置控制：

```python
graph_backend = "backbone_cutoff"  # main
# alternatives: "crystalnn_reproduction", "fixed_radius_ablation"
```

无论选择何种 backend，都在 preprocessing 后缓存 `edge_index`、periodic image vector 和构图参数；不得在每个 epoch 重新调用 CrystalNN。

综上，成熟晶体学库负责提供经过同行验证的 crystallographic metadata 与候选 basis data，而模型仓库负责统一 convention、生成 CG/path tables 并执行严格等变性验收。这样既复用 PGEqNN 的可靠基础设施，又保持本文多 PG branches、inverse subduction 和 common-space fusion 所需的完整 gauge control。

这里的等变性必须作用于整个 PBC graph，而不只是 feature components。对空间群操作 \(g=(R_g,t_g)\)，其联合动作同时包含：

$$
i\mapsto\pi_g(i),
\qquad
e_{ij}^{\mathbf n}\mapsto
e_{\pi_g(i)\pi_g(j)}^{\mathbf n'},
\qquad
(g\cdot z)_{\pi_g(i)}
=
\rho_K(R_g)z_i,
$$

其中 \(\mathbf n\) 与 \(\mathbf n'\) 记录相应的 periodic image。只要 neighbor construction 在该映射下闭合、所有边共享 message rule、节点使用 permutation-equivariant sum aggregation，并正确处理 nonsymmorphic translation 与跨胞边，PG-equivariant PBC message passing 就对节点置换与 feature rotation 的联合动作严格等变。

其核心价值在于：

> 不只是将 O(3) features 换一个 basis，而是真正利用 finite-group fusion rules 在更细粒度 symmetry subspaces 上进行 nonlinear interaction。

这是区别于：

- canonicalize + ordinary MLP；
- output projection；
- 纯 O(3) tensor head；

的关键。

---

## 5.10 Step 8：PG-equivariant MLP

聚合完成后，PG-equivariant MLP 对每个节点分别作用：

$$
q_{K,i}
=
M_K^{PG}
(\tilde z_{K,i}^{PG}).
$$

PG-equivariant MLP 不负责跨节点通信；它只在单个节点的 multiplicity / channel space 内 mixing，并保持有限群 representation structure。

可以考虑：

- blockwise linear maps by \(\Gamma\)；
- gated nonlinearities；
- norm-based nonlinearities；
- repeated PG TP + PG EqMLP blocks。

第一版建议只使用 1–2 层，避免模型过重。

在进入 constrained readout 前，先对各节点的 \(q_{K,i}\) 施加 PG-equivariant normalization。随后由任务参数

$$
\texttt{output\_scope}\in\{\texttt{global},\texttt{site}\}
$$

控制是否做 permutation-invariant global pooling。对于 dielectric / elastic，\(\texttt{output\_scope=global}\)：

$$
\bar q_K
=
\operatorname{Pool}_{i}
\left[
\operatorname{Norm}^{PG}_K(q_{K,i})
\right].
$$

对于 BEC，\(\texttt{output\_scope=site}\)，直接保留：

$$
\bar q_{K,i}=\operatorname{Norm}^{PG}_K(q_{K,i}),
$$

不执行 pooling。global pooling 第一版采用按节点求和或求均值；它不混合 representation components，因此保持 PG-equivariance。norm 必须按 PG irrep block 使用不变二次范数进行归一化，并在 representation / multiplicity 维度上保持 \(K\)-equivariance；不能对非平凡 irrep 的分量使用会破坏表示结构的普通逐元素 LayerNorm。可采用带 \(\epsilon\) 与每个 irrep/channel 可学习标量增益的 equivariant RMSNorm；加性 bias 仅允许用于 trivial \(A_1\) blocks。

---

## 5.11 Step 9：Symmetry-constrained tensor readout

对任务 \(t\)，定义目标 representation \(V_t\)。

Dielectric：

$$
V_\epsilon
=
D^{(0,+)}
\oplus
D^{(2,+)}.
$$

Elastic：

$$
V_C
=
2D^{(0,+)}
\oplus
2D^{(2,+)}
\oplus
D^{(4,+)}.
$$

BEC：

$$
V_Z
=
D^{(0,+)}
\oplus
D^{(1,+)}
\oplus
D^{(2,+)}.
$$

对 global 任务，在每个激活分支中限制到该分支 point group \(K\)：

$$
V_t
\downarrow_K.
$$

只读取其中满足：

$$
\Gamma=A_1
$$

的 independent copies，即：

$$
\hat c_K
=
R_K(\bar q_K),
$$

并满足：

$$
\hat c_K
\in
\mathrm{Fix}_K(V_t).
$$

因此 global 输出在 architecture level 上严格满足 point-group constraints，而不是依赖 loss“学会”哪些 tensor components 应该为零或相等。

对 BEC 不使用 global \(A_1\) readout，也不引入独立的 Wyckoff-specific readout。对所有节点共享同一个 PG-equivariant tensor map \(R^{\mathrm{node}}_{K,Z}\)：

$$
\hat Z^*_{K,i}
=
R^{\mathrm{node}}_{K,Z}(\bar q_{K,i}).
$$

由于前述 PBC message passing 对联合的节点置换—feature 变换等变，且 \(R^{\mathrm{node}}_{K,Z}\) 在节点间共享，所以直接得到：

$$
\hat Z^*_{K,\pi_g(i)}
=
\rho_Z(R_g)\hat Z^*_{K,i}
=
R_g\hat Z^*_{K,i}R_g^\top.
$$

因此，同一 Wyckoff orbit 上的 tensor relation 已由网络自动保证；若 \(g\) 固定节点 \(i\)（模晶格平移），则 \(\hat Z^*_{K,i}\in\mathrm{Fix}_{K_i}(V_Z)\) 也自动成立。无需只预测 orbit representative，也无需手工展开 Wyckoff orbit。任务切换在模型结构上主要就是是否 pooling。

---

## 5.12 Step 10：提升到公共空间并进行 branch fusion

每个完整分支的最终 target representation 先映射回相同的 O(3) tensor-irrep space。这里 inverse subduction 只表示最终 target coordinates 的逆基变换，并不声称能够逆转分支内的 TP、非线性或 pooling。

对任务 \(t\) 的 target layout 定义：

$$
\mathcal U_{K,t}
=
\bigoplus_{(\ell,p)\in V_t}
\left(
I_{n_{\ell,p}^{(t)}}
\otimes
U_K^{(\ell,p)}
\right),
$$

其中 \(n_{\ell,p}^{(t)}\) 是 \(V_t\) 中该 O(3) irrep 的 copy multiplicity。例如 elastic tensor 的 \(\ell=0\) 与 \(\ell=2\) 均各有两个 copies，必须用固定顺序保留，不能在不同分支中任意交换。

对于 global tensor，readout 首先预测 \(\mathrm{Fix}_K(V_t)\) 中的 independent \(A_1\) coefficients：

$$
\hat a_K
=
R_{K,t}(\bar q_K)
\in
\mathbb R^{d_{K,t}},
\qquad
d_{K,t}
=
\dim\mathrm{Fix}_K(V_t).
$$

参考 PGEqNN 的 tensor reconstruction，定义固定的 zero-padding / injection matrix：

$$
J_{K,t}:
\mathbb R^{d_{K,t}}
\hookrightarrow
V_t\downarrow_K,
\qquad
\tilde c_K^{PG}
=
J_{K,t}\hat a_K.
$$

\(J_{K,t}\) 将各 independent coefficients 放入对应的 repeated \(A_1\) slots，并将所有 nontrivial-irrep slots 置零。随后才执行完整 target basis 的 inverse subduction：

$$
J_{K,t}^{\dagger}J_{K,t}
=
I_{d_{K,t}},
\qquad
J_{K,t}J_{K,t}^{\dagger}
=
P^{PG}_{A_1,K,t}.
$$

$$
\boxed{
\hat c_K^{O(3)}
=
\mathcal U_{K,t}^{\dagger}
J_{K,t}\hat a_K
}.
$$

在 real orthonormal PGH convention 下，这与 PGEqNN 使用的 zero-padding 后乘 \([\mathcal U_{K,t}]^\top\) 完全一致。相应的 O(3)-basis fixed-subspace projector 为：

$$
P^{O(3)}_{K,t}
=
\mathcal U_{K,t}^{\dagger}
J_{K,t}J_{K,t}^{\dagger}
\mathcal U_{K,t}.
$$

对于 BEC，node-wise readout 不执行全局 \(A_1\) restriction，而是为每个 canonical node 输出完整的 \(V_Z\downarrow_K\) coordinates，因此直接逐节点变换：

$$
\boxed{
\hat c_{K,i}^{O(3)}
=
\mathcal U_{K,Z}^{\dagger}
\hat c_{K,i}^{PG}
}.
$$

所有分支必须使用同一个 target-irrep ordering、copy convention 和 canonical atom mapping，才能在公共 O(3) space 中融合。

### 5.12.1 Target lifting 中各符号的含义

| 符号 | 含义 | 如何得到 / 是否学习 |
| --- | --- | --- |
| \(t\) | 任务类型，例如 dielectric、elastic 或 BEC | 由 dataset/task configuration 指定 |
| \(V_t\) | 任务 \(t\) 的完整 O(3) target representation | 由目标张量的 intrinsic permutation symmetries 做 harmonic decomposition 得到 |
| \(n_{\ell,p}^{(t)}\) | \(V_t\) 中 \(D^{(\ell,p)}\) 的 copy multiplicity | 由 \(V_t\) 的解析 decomposition 固定；例如 elastic 的 \(\ell=0,2\) 各有两个 copies |
| \(\mathcal U_{K,t}\) | 只作用于 target layout 的完整 subduction matrix | 用对应的 \(U_K^{(\ell,p)}\) 按 target copy order 做 block direct sum；固定、不学习 |
| \(d_{K,t}\) | global tensor 在 \(K\) 下允许的 independent coefficient 数 | \(d_{K,t}=\dim\mathrm{Fix}_K(V_t)\)，也等于 \(V_t\downarrow_K\) 中所有 \(A_1\) copies 的总数 |
| \(\bar q_K\) | PG norm 与 global pooling 后的 graph representation | 由网络前向传播计算 |
| \(R_{K,t}\) | 从 \(\bar q_K\) 到 independent \(A_1\) coefficients 的 readout | 可学习的 \(K\)-equivariant/trivial-block linear map 或小型 MLP |
| \(\hat a_K\) | \(R_{K,t}\) 预测的 \(d_{K,t}\) 个 independent coefficients | 模型输出，参与监督学习 |
| \(J_{K,t}\) | 将 \(\hat a_K\) 放回完整 PGH target vector 的 injection/zero-padding matrix | 根据 target PGH layout 中的 \(A_1\) slot indices 自动生成；固定、不学习 |
| \(\tilde c_K^{PG}\) | zero-padding 后的完整 PGH target vector | \(\tilde c_K^{PG}=J_{K,t}\hat a_K\) |
| \(\hat c_K^{O(3)}\) | inverse-subduce 后、可跨分支比较的公共 O(3) coefficients | \(\mathcal U_{K,t}^\dagger\tilde c_K^{PG}\) |
| \(P^{PG}_{A_1,K,t}\) | PGH basis 中保留全部 trivial-irrep slots 的 projector | \(J_{K,t}J_{K,t}^\dagger\)，固定、不学习 |
| \(P^{O(3)}_{K,t}\) | 同一 fixed subspace 在 O(3) harmonic basis 中的 projector | 由 \(\mathcal U_{K,t}^\dagger P^{PG}_{A_1,K,t}\mathcal U_{K,t}\) 共轭变换得到 |
| \(\mathcal U_{K,Z}\) | BEC target \(V_Z=D^{(0,+)}\oplus D^{(1,+)}\oplus D^{(2,+)}\) 的 target-specific subduction | 将三块对应的 \(U_K^{(\ell,+)}\) 直接求和；固定、不学习 |

### 5.12.2 \(J_{K,t}\) 的自动生成

设完整 PGH target vector 的固定索引顺序为：

$$
s=(\ell,p,r,\Gamma,a,\mu),
$$

其中 \(r=1,\ldots,n_{\ell,p}^{(t)}\) 表示 target 中 repeated O(3) copies。枚举所有满足

$$
\Gamma=A_1,
\qquad
\mu=1
$$

的 slot indices：

$$
\mathcal I_{K,t}^{A_1}
=
\{s_1,\ldots,s_{d_{K,t}}\}.
$$

令 \(e_{s_j}\) 为完整 PGH target space 中第 \(s_j\) 个 standard basis vector，则：

$$
J_{K,t}
=
\begin{bmatrix}
e_{s_1}&e_{s_2}&\cdots&e_{s_{d_{K,t}}}
\end{bmatrix}.
$$

因此 \(J_{K,t}\) 只是由 PGH layout 自动生成的稀疏 \(0/1\) matrix；它不需要训练，也不依赖样本。对每个 \((K,t)\)，初始化时生成一次并缓存即可。若使用的 real irrep convention 将 trivial representation 命名为 \(A_g\)、\(A'\) 等，则程序应依据“所有表示矩阵均为 1”识别 trivial irrep，而不能把名称 A1 写死。

随后才进行跨分支加权平均：

$$
\hat c_{O(3)}(x)
=
\sum_{K\in\mathcal A(x)}
w_K(x)\hat c_K^{O(3)}(x).
$$

不能直接平均不同 PG basis 中维数和语义不同的 coefficients。对于 global 任务，由于任一父群 \(K\supseteq G_x\) 均满足：

$$
\mathrm{Fix}_K(V_t)
\subseteq
\mathrm{Fix}_{G_x}(V_t),
$$

所有 current / parent branch 输出提升后的加权平均仍属于当前点群允许的 tensor subspace。对于 BEC，则对相同 canonical node index 逐原子融合；由于 \(w_K(x)\) 是不变标量，且每个分支都对联合的 node permutation–tensor rotation 等变，加权结果仍满足

$$
\hat Z^*_{\pi_g(i)}
=
R_g\hat Z^*_iR_g^\top.
$$

在 weighted average 之后，可加入一个由所有点群共享的 O(3)-equivariant readout：

$$
\hat c_{\mathrm{out}}^{O(3)}
=
R_t^{O(3)}\!\left(\hat c_{O(3)}\right).
$$

该层用于在公共 O(3) tensor-irrep space 中做最终的 channel/copy mixing 与任务校准，而不是重新执行 point-group routing；在 BEC 中它同样按节点共享地作用。由于 \(R_t^{O(3)}\) 与任意 \(g\in O(3)\) 的作用对易，它既不破坏 global 任务的 \(\mathrm{Fix}_{G_x}(V_t)\) constraint，也不破坏 BEC 的 node permutation–tensor rotation equivariance。主实验应比较：

$$
R_t^{O(3)}=I
\qquad\text{vs.}\qquad
R_t^{O(3)}=\text{learned equivariant readout}.
$$

若分支在 inverse subduction 后已经只输出最终 tensor 所需的最小系数、没有额外 channel multiplicity，则该层可能退化为简单的 copy mixing 或缩放；此时应优先使用 identity，以避免无效增参。

对于连续结构路径 \(x(t)\to x(0)\)，若 \(x(0)\) 恢复父群 \(K_0\)，并且 branch maps、group embeddings 与 frame transport 在公共 gauge 下连续，则所有严格低于 \(K_0\) 的分支满足：

$$
w_K(x(t))\rightarrow0,
\qquad K\subsetneq K_0.
$$

于是：

$$
\lim_{t\to0}\hat c_{O(3)}(x(t))
=
\hat c_{O(3)}(x(0)).
$$

因此 active set 即使在精确 symmetry boundary 改变，也不会产生有限 jump；新加入或移除的分支必须具有零边界权重。该论证保证的是 \(C^0\) continuity。若训练目标涉及 forces、higher-order responses 或路径导数，还需对 gate、matching 与 canonical frame map 额外验证 \(C^1/C^2\) regularity。

最后通过固定的 spherical / Cartesian intertwiner 转换为 canonical Cartesian tensor：

$$
\hat T_{\mathrm{can}}
=
\mathcal C_t
(\hat c_{\mathrm{out}}^{O(3)}).
$$

这里：

$$
\mathcal C_t
$$

不应由普通 MLP 学习，而应使用固定的 group-theoretic tensor basis transformation。

---

## 5.13 Step 11：De-canonicalization

最后将 canonical-frame prediction 转回原始输入 Cartesian frame：

$$
\hat T_{\mathrm{orig}}
=
\rho_t(Q_x^{-1})
\hat T_{\mathrm{can}}.
$$

对 rank-2：

$$
\hat T_{\mathrm{orig}}
=
Q_x^{-1}
\hat T_{\mathrm{can}}
Q_x^{-\top}.
$$

对 rank-4 则相应地对四个 indices 变换。

对于 BEC，除对每个 \(3\times3\) tensor 做上述 rank-2 frame transformation 外，还要用 canonicalization 保存的 inverse atom mapping 将节点顺序恢复为原始 structure 的 site order。

完整 pipeline 需要验证：

$$
F(Rx)
=
\rho_t(R)F(x)
$$

在随机 frame transformations 下数值成立。

---

## 5.14 两层 point-group awareness

本方法有意区分两类 PG 信息。

### Hierarchy-level PG awareness

$$
\{r_H(x)\}_{H\in\mathcal A(x)}
\rightarrow
\{w_K(x)\}_{K\in\mathcal A(x)}.
$$

连续 parent-group residuals 决定当前群与兼容父群完整分支的贡献。point-group identity 只确定候选 subgroup structure，不直接充当 one-hot gate。

---

### Representation-level PG awareness

$$
O(3)
\downarrow
K
\rightarrow
PG_K\text{-TP}
\rightarrow
PG_K\text{-EqMLP}
\rightarrow
\begin{cases}
\operatorname{Pool}+\mathrm{Fix}_K(V_T), & \text{global tensor},\\
R^{\mathrm{node}}_{K,Z}\text{ on every node}, & \text{BEC}.
\end{cases}
$$

每个激活群的完整分支显式利用自身 finite-group irreps 与 fusion rules。global tensor 通过 fixed-subspace readout 保证输出约束；BEC 的 Wyckoff/site constraints 则由 PG-equivariant PBC message passing 的联合节点置换—feature 等变性自动保证。之后，各分支输出提升回公共 O(3) tensor space 再融合。

这两个机制可以被独立 ablate，是方法可解释性的重要组成部分。

---

## 5.15 前向传播伪代码

```python
# x: relaxed crystal structure
x_bar, Gx, Qx, symmetry_meta = canonicalize_with_spglib(x)

# Embedded current-to-parent DAG paths and parent-symmetry residuals
paths = compatible_parent_paths(Gx, symmetry_meta)
active_groups = unique_groups(paths)  # current group plus all compatible parents
parent_residual = parent_group_operation_residuals(x_bar, paths)
a_edge = residual_breaking_gates(parent_residual)  # invariant; a(0) = 0
path_prior = continuous_path_compatibility(x_bar, paths)
weights = aggregate_path_stick_breaking_weights(
    paths, path_prior, a_edge
)  # one weight per active group; nonnegative; sum = 1

# universal node-wise equivariant representation and periodic graph
h, edge_attr, edge_index = pretrained_O3_backbone(x_bar)

# Mutually exclusive shared-TP implementation switch
shared_tp_backend = config.shared_tp_backend  # "full_o3" | "so2_reduced"

# shared O(3) branch input
shared = sum(
    shared_expert_k(
        h,
        edge_attr,
        edge_index,
        tp_backend=shared_tp_backend,
        path_weight_from_endpoint_invariants="concat(src, tgt)",
    )  # TP messages + neighbor sum + node-wise EqMLP
    for k in range(N_s)
)

if use_backbone_skip:
    shared = shared + project_if_needed(h)

# Every active current/parent group runs a complete branch
# Compute the fixed-width invariant endpoint latent once and reuse it across PGs
route_latent = shared_endpoint_router(
    concat_o3_trivial_channels(shared, edge_index)  # [src | tgt] per edge
)
branch_coeff_o3 = []
for group_K in active_groups:
    weight_K = weights[group_K]
    z_K = shared
    z_pg_K = subduction[group_K](z_K)
    edge_pg_K = subduction[group_K](edge_attr)
    z_pg_K = PG_tensor_product_message_passing[group_K](
        z_pg_K,
        edge_pg_K,
        edge_index,
        fixed_cg=fixed_PG_CG[group_K],
        allowed_paths=PG_path_set[group_K],
        path_weights=low_rank_group_path_head[group_K](route_latent),
        self_interaction=PG_self_interaction[group_K],
    )
    z_pg_K = PG_equivariant_mlp_per_node[group_K](z_pg_K)
    z_pg_K = PG_equivariant_norm_per_node[group_K](z_pg_K)
    if config.output_scope == "global":  # dielectric / elastic
        branch_input_K = permutation_invariant_node_pool(z_pg_K)
        independent_A1_K = readout_fix_subspace[group_K, task](branch_input_K)
        coeff_pg_K = inject_A1_into_full_PGH[group_K, task](independent_A1_K)
    else:  # BEC: preserve nodes; equivariance already includes site permutations
        coeff_pg_K = shared_nodewise_equivariant_readout[group_K, task](z_pg_K)
    coeff_o3_K = inverse_subduction_dagger_to_tensor_irreps[
        group_K, task
    ](coeff_pg_K)
    branch_coeff_o3.append((weight_K, coeff_o3_K))

# Fusion is legal only after every branch is lifted to the common O(3) space
coeff_o3 = sum(
    weight_K * coeff_o3_K
    for weight_K, coeff_o3_K in branch_coeff_o3
)

# Optional shared readout after common-space weighted fusion
coeff_o3_out = shared_O3_readout[task](coeff_o3)

T_can = irrep_to_cartesian[task](coeff_o3_out)

# recover tensor in original Cartesian frame
T_pred = decanonicalize_tensor(
    T_can, Qx, task
)

return T_pred
```

---

# 6. 训练策略

## 6.1 主设置：Frozen pretrained backbone

第一组 headline experiments 建议固定 backbone：

$$
\Theta_{\mathrm{backbone}}
=
\mathrm{frozen}.
$$

只训练：

- shared O(3) experts；
- current / parent PG stages 与 readouts；
- 融合后的共享 O(3)-equivariant readout（若启用）；
- 受约束的 gate calibration parameters（若启用）。

优点：

1. 最干净地检验 downstream architecture；
2. 避免 full finetune 把 PG module 的贡献吸收到 backbone 中；
3. 强化 parameter-efficient transfer learning 的论文定位；
4. 降低计算成本。

---

## 6.2 Finetuning ablation

只对少数关键模型比较：

### Frozen

Backbone 全冻结。

### Partial FT

仅最后 1–2 个 backbone blocks 可训练。

### Full FT

全模型可训练。

不建议每个 baseline 都乘上三种 finetune setting，以免实验组合爆炸。

优先比较：

- O(3)-Base；
- Single-PG-Rep；
- Hierarchical-Full。

---

## 6.3 Loss function

推荐在 irreducible tensor coefficient space 中训练，而不是直接使用 naive Cartesian MSE。

### Dielectric

$$
\mathcal L_\epsilon
=
\lambda_0
\left\|
\hat\epsilon^{(0)}
-
\epsilon^{(0)}
\right\|^2
+
\lambda_2
\left\|
\hat\epsilon^{(2)}
-
\epsilon^{(2)}
\right\|^2.
$$

---

### Elastic

对 decomposition 中各 copies 分别计算：

$$
\mathcal L_C
=
\sum_{a\in 2\times\ell=0}
\lambda_{0,a}\mathcal L_{0,a}
+
\sum_{b\in 2\times\ell=2}
\lambda_{2,b}\mathcal L_{2,b}
+
\lambda_4\mathcal L_4.
$$

需要避免：

$$
\ell=0
$$

数值尺度过大导致 anisotropic channels 被淹没。

建议：

- per-irrep standardization；
- train-set RMS normalization；
- variance-based weighting；
- uncertainty weighting。

主论文必须报告 raw physical-unit metrics。

---

## 6.4 Point-group imbalance

PG-specific full branches 可能面临：

$$
|\mathcal D_{G_1}|
\gg
|\mathcal D_{G_2}|.
$$

候选策略：

1. group-balanced batch sampling；
2. shared expert + hierarchical full branches，自然缓解 rare-PG overfitting；
3. PG-specific stages 使用更强 weight decay；
4. 使用 bottleneck / low-rank parameterization；
5. minimum-sample threshold；
6. branch dropout，但不得破坏 gate 的父群极限条件。

例如训练时随机关闭某个非根 residual branch：

$$
w_K\hat c_K^{O(3)}\rightarrow0,
$$

迫使 shared path 保持足够的普适能力。

第一版不使用普通 learned router。当前点群与 compatible parent DAG 由晶体学 metadata 确定，但 branch weights 由连续 parent-group operation residuals 决定。这样可以避免：

- load balancing；
- router collapse；
- routing entropy；

等额外 confounds。

---

## 6.5 Parameter-efficient design

对于每个 PG-specific stage，可控制：

- hidden multiplicity；
- bottleneck channels；
- low-rank channel mixing；
- shallow TP / MLP blocks。

shared adaptation 通过 `shared_tp_backend` 在 `full_o3` 与 `so2_reduced` 之间切换。除报告各自原生规模外，还需调整 channel multiplicity、path multiplicity 或 EqMLP width，构造：

$$
N_{\mathrm{trainable}}^{\mathrm{full\_o3}}
\approx
N_{\mathrm{trainable}}^{\mathrm{so2\_reduced}},
$$

的 parameter-matched comparison，以区分收益来自 TP parameterization 还是单纯的参数量变化。

应报告：

$$
N_{\mathrm{total}},
$$

$$
N_{\mathrm{trainable}},
$$

$$
N_{\mathrm{active/sample}},
$$

以及：

- FLOPs；
- inference latency；
- peak GPU memory。

hierarchical full-branch model 的单样本 active cost 为：

$$
N_s\text{ shared experts}
+
|\mathcal A(x)|\text{ complete PG branches}.
$$

因此必须额外报告平均 / 最大 active branch 数、各 embedded subgroup DAG 的 FLOPs，以及相对 single-branch hard routing 的 latency overhead。第一版通过限制 benchmark 中允许的 compatible parent embeddings、共享 branch parameters 与浅层 PG blocks 控制成本；不能以 hard top-\(k\) 截断破坏连续性作为默认实现。

---

# 7. Baselines

Baseline 分为两类：

1. **same-backbone causal baselines**；
2. **published baselines**。

第一类用于证明每个模块的因果作用。

第二类用于与领域已有方法比较绝对性能。

---

## 7.1 Same-backbone causal baselines

| Baseline | Shared O(3) | PG branches | Continuous hierarchy | PG TP/MLP + norm + constrained readout | 核心问题 |
|---|---:|---:|---:|---:|---|
| **O(3)-Base** | ✗ | ✗ | ✗ | ✗ | pretrained O(3) representation 本身能做到什么？ |
| **O(3)+Projection** | ✗ | ✗ | ✗ | output only | 只在 output enforce symmetry 是否已经足够？ |
| **Shared-O3** | ✓ | ✗ | ✗ | ✗ | task-specific O(3) adaptation 是否有用？ |
| **Hard-Routing-O3** | ✓ | current only | ✗ | ✗ | 原始 one-hot routing 的收益与不连续性是什么？ |
| **Single-PG-Rep** | ✓ | current only | ✗ | ✓ | 单一 current-group PG representation 是否有用？ |
| **Parent-Average-Control** | ✓ | current + parents | arbitrary average | ✓ | 增益是否只来自额外容量 / ensemble？ |
| **Hierarchical-Full** | ✓ | current + parents | symmetry-breaking gates | ✓ | 完整模型是否兼顾精度、对称性与连续性？ |
| **Branch-only** | ✗ | current + parents | symmetry-breaking gates | ✓ | shared transfer 是否缓解 long-tail branch 的数据碎片化？ |

---

### 7.1.1 O(3)-Base

同一个 pretrained backbone + 标准 O(3)-equivariant tensor head。

作为最基本对照。

---

### 7.1.2 O(3)+Projection

先正常输出：

$$
\hat T,
$$

再进行：

$$
\hat T_{\mathrm{sym}}
=
P_G(\hat T).
$$

即：

$$
P_G(\hat T)
=
\frac1{|G|}
\sum_{g\in G}
\rho_T(g)\hat T.
$$

这是必须击败的 conceptual baseline。

如果 Hierarchical-Full 仅比 Base 好，却和 Projection 相同，则说明 PG module 的主要价值可能只是 output validity，而不是 representation learning。

---

### 7.1.3 Shared-O3

验证 pretrained features 在直接进入 downstream readout 前，是否需要：

$$
O(3)\ TP
+
O(3)\ EqMLP.
$$

它回答：

> pretrained O(3) features 是否已经 task-ready？

---

### 7.1.4 Hard-Routing-O3

在不进入 PG representation space 的情况下加入 one-hot current-PG O(3) expert，保留为原始方案 baseline。

用于专门测试：

> hard routing-level point-group awareness 及其 symmetry-boundary discontinuity

是否独立有效。

---

### 7.1.5 Single-PG-Rep

保留 shared O(3) expert，只运行 current-group PG branch，不激活父群分支，并进行：

$$
O(3)\rightarrow G
\rightarrow
PG\ TP/MLP.
$$

用于专门测：

> representation-level point-group awareness

---

### 7.1.6 Parent-Average-Control

激活与完整模型相同的 current / parent full branches，但使用固定平均或不满足父群极限条件的普通 normalized similarity weights。它用于检验改善是否只来自：

- 更大的 active capacity；
- 多分支 ensemble；
- 普通平滑平均。

该 control 不作为最终方法，因为它没有结构连续性的理论保证。

---

### 7.1.7 Hierarchical-Full

完整模型：

$$
\boxed{
\text{Pretrained O3}
\rightarrow
\{\text{current + compatible parent full PG branches}\}
\rightarrow
\text{inverse subduction to common O3 space}
\rightarrow
\text{hierarchical weighted fusion}
\rightarrow
\text{shared O3 readout (optional)}
}
$$

---

### 7.1.8 Branch-only

移除 shared O(3) experts，其余 current / parent full branches、continuous gates 与 common-space fusion 均与 Hierarchical-Full 相同。用于检验 shared transfer 对 long-tail PG 的作用。

---

# 7.2 Published Baselines

## 7.2.1 MatTen

MatTen 是基于 e3nn 的 elasticity tensor equivariant GNN。

其特点是：

- 直接预测 rank-4 elastic tensor；
- 统一处理七大晶系；
- 使用 O(3)-equivariant representation；
- 具有公开 MP elastic benchmark。

**任务：Elastic。**

它代表：

> direct O(3)-equivariant tensor model

这一类 baseline。

---

## 7.2.2 GMTNet

GMTNet 面向：

- dielectric；
- piezoelectric；
- elastic；

等通用晶体张量。

其核心贡献包括：

- O(3) tensor representation；
- crystal symmetry；
- 对应 DFT calculation structure 的严格数据配对。

**任务：Dielectric + Elastic。**

GMTNet 是 symmetry-aware tensor prediction 的关键 published baseline。

---

## 7.2.3 ETGNN

ETGNN 通过 edge-based tensor expansion 实现 tensor output equivariance。

其一个重要特点是：

> backbone 本身不一定需要显式高阶 equivariant representation。

因此它代表：

> output construction / tensor expansion ensures equivariance

这一类方法。

---

## 7.2.4 AnisoNet

AnisoNet 是 dielectric tensor 的直接 equivariant predictor。

可将 dielectric 输出组织为：

$$
0e+2e.
$$

即：

$$
D^{(0,+)}
\oplus
D^{(2,+)}.
$$

它特别强调 dielectric anisotropy，因此非常适合作为 dielectric benchmark 中的直接 baseline。

---

## 7.2.5 DTNet

DTNet 从 pretrained atomistic potential 中提取 latent equivariant information，并冻结 parent potential 进行 dielectric prediction。

其 conceptual position 与本研究非常接近：

$$
\text{pretrained atomistic model}
\rightarrow
\text{tensor downstream task}.
$$

因此 DTNet 是检验：

> 为什么 pretrained representation 有价值

这一叙事的重要 baseline。

---

## 7.2.6 GoeCTP

GoeCTP 强调 canonicalization / polar decomposition，将 equivariance guarantee 部分外置，使得后续网络能够采用更普通、更高效的网络结构。

它是本研究的重要对照，因为 reviewer 很可能会问：

> 如果 canonicalization 已经足够，为什么还需要显式 O(3)/PG equivariant representation？

---

## 7.2.7 PGEqNN

PGEqNN 是最直接的 point-group-aware baseline。

其核心思想是将 SO(3) rotational order 进一步按 point-group irreps partition，并比较：

- scalar invariant model；
- SO(3) partition；
- full PG partition；
- \(A_1\)-restricted PG partition。

本研究与其关键区别在于：

1. 使用 **universally pretrained O(3) backbone**；
2. 对 current group 与 compatible parents 运行 **完整 PG branches**；
3. 强调 **parameter-efficient downstream specialization**；
4. 强调 **label efficiency**；
5. 将 hierarchy-level 与 representation-level PG awareness 解耦；
6. 研究跨 PG transfer 与 long-tail regime。

PGEqNN-Full 与 PGEqNN-\(A_1\)-only 均应作为 baseline。

---

# 8. Benchmarks

本研究按输出粒度选用以下 benchmark：

1. global tensor：JARVIS tensor benchmark；
2. global elastic tensor：MatTen elastic benchmark；
3. atom-resolved BEC：JARVIS-DFPT（主）与 MP-Dielectric（候选跨库验证）。

各 benchmark 分别使用锁定的数据版本、公开 protocol 与固定 split。point-group-controlled、anisotropy-conditioned、low-data 和 long-tail 实验均从相应 benchmark 的训练集派生。MP-Dielectric 只有在底层 DFPT task 中能稳定取得与结构逐原子对齐的完整 BEC 后才进入正式主表，不能用汇总 dielectric 记录数替代有效 BEC 样本数。

---

## 8.1 Benchmark A：JARVIS Tensor Published Benchmark

第一阶段使用其中与任务范围一致的 dielectric 与 elastic tensor 子任务；piezoelectric 等其他 tensor 仅作为后续扩展。所有结果优先复现 published benchmark 的数据清洗、单位和 split。

GMTNet 系列 benchmark 的重要优点是：

> tensor property 与对应 DFT calculation structure 被严格匹配。

这可以减少：

$$
\text{structure symmetry}
\neq
\text{tensor symmetry}
$$

的问题。

公开 benchmark 中 dielectric 常见规模约为：

$$
N\approx4713.
$$

Elastic 数据规模会随 JARVIS release / curation protocol 有所变化。

因此本研究不应简单混用不同工作中的数字，而应：

1. 优先使用 baseline 官方 split；
2. 固定数据版本；
3. 保存 sample IDs；
4. 在主表中只比较相同 split 下数字。

---

### GMTNet-style metrics

#### Frobenius error

$$
\mathrm{Fnorm}
=
\|\hat T-T\|_F.
$$

#### Relative tensor error

$$
r_i
=
\frac{
\|\hat T_i-T_i\|_F
}{
\|T_i\|_F
}.
$$

#### Error-within-Threshold

报告：

- EwT 25%；
- EwT 10%；
- EwT 5%。

该 benchmark 用于回答：

> Ours 与 general tensor SOTA 相比总体表现如何？

---

## 8.2 Benchmark B：MatTen Elastic

MatTen 数据集包含约：

$$
N=10,276
$$

个 DFT elastic tensors。

用途包括：

1. 与 MatTen 直接 comparison；
2. 验证 JARVIS → MP database transfer；
3. 验证不同 DFT workflow 下结论是否稳定；
4. 作为主要的 larger elasticity benchmark。

---

## 8.3 Benchmark C：Born Effective Charge

BEC benchmark 预测 relaxed equilibrium crystal 中每个原子的完整 \(3\times3\) Born effective charge，即每个样本的标签形状为 \(N\times3\times3\)。

| 数据集 | 有效样本数 | Full \(N\times3\times3\) BEC | equilibrium structure | 用途 |
| --- | ---: | :---: | --- | --- |
| **JARVIS-DFPT** | **5015** | ✅ | ✅ 严格 relaxed | **主 benchmark** |
| **MP-Dielectric** | 待锁定 release 后审计 | 需从底层 DFPT task 核验 | relaxed calculation structure 与 BEC 必须逐原子对齐 | 候选跨数据库验证 |

JARVIS-DFPT 的 5015 条作为当前 proposal 的预期规模；正式实验开始前仍需固定 release、保存 sample IDs，并检查缺失值、单位、原子顺序和重复结构。MP 的公开 dielectric derived collection 不直接等价于完整 BEC 数据集；正式样本数应定义为同时满足以下条件的 task 数：

1. `output.dielectric_properties.born_charges` 非空且形状为 \(N\times3\times3\)；
2. calculation structure 为目标 relaxed equilibrium structure；
3. BEC 的原子顺序可与 structure sites 一一对应；
4. calculation 收敛且通过数值质量检查。

BEC 主指标为 atom-wise Frobenius MAE / RMSE，并补充 element-macro、Wyckoff-orbit-macro、equivariance violation 与 acoustic-sum-rule residual。数据划分优先按 material / reduced composition 分组，避免同一材料的近重复结构跨 split 泄漏。

---

## 8.4 各主 Benchmark 上的 Low-data / Long-tail PG Protocol

这是本研究非常关键的特色 benchmark。

该 protocol 分别在 JARVIS tensor、MatTen elastic 与 JARVIS-DFPT BEC 的固定 split 内运行。对每个 target PG \(G\)，固定 validation / test set，只减少该 PG 的训练 labels；任何 PG 子集选择都只使用训练集统计，不改变主 benchmark 的测试集。

---

### 方案 A：比例式

$$
1\%,5\%,10\%,25\%,50\%,100\%.
$$

---

### 方案 B：固定样本数

$$
N_G
\in
\{32,64,128,256,\mathrm{full}\}.
$$

---

比较：

$$
\text{Branch-only}
$$

$$
\text{Shared-only}
$$

$$
\boxed{
\text{Hierarchical-Full}
}
$$

核心预期：

$$
\text{Hierarchical-Full}
-
\text{Branch-only}
$$

在 low-data 时最大，并随 \(N_G\) 增加逐渐缩小。

这将直接验证：

> shared experts 是否真正缓解了 point-group data fragmentation。

---

# 9. Evaluation Metrics 与 Analytical Experiments

## 9.1 Standard Tensor Metrics

### Frobenius norm error

$$
E_F
=
\|\hat T-T\|_F.
$$

报告：

- MAE；
- RMSE；
- physical units。

---

### Relative Frobenius error

$$
E_{\mathrm{rel}}
=
\frac{
\|\hat T-T\|_F
}{
\|T\|_F+\epsilon
}.
$$

---

### EwT

报告：

- EwT5；
- EwT10；
- EwT25。

方便与 GMTNet / GoeCTP 系列工作比较。

---

# 9.2 Irrep-wise Error

只报 Cartesian MAE 会掩盖模型究竟提升了：

- isotropic information；
- 还是 anisotropic information。

因此需要拆开评估。

---

## Dielectric

分别报告：

$$
MAE_{\ell=0},
$$

$$
MAE_{\ell=2}.
$$

---

## Elastic

至少报告：

$$
MAE_{\ell=0},
$$

$$
MAE_{\ell=2},
$$

$$
MAE_{\ell=4}.
$$

最好进一步区分 elasticity decomposition 中 repeated copies：

$$
2D^{(0)}
\oplus
2D^{(2)}
\oplus
D^{(4)}.
$$

---

## No-skill floor

同时计算各 block 的简单 baseline：

$$
\mathrm{MAD}_{\ell}
=
\mathbb E
\left[
|y_{\ell}-\mathrm{median}(y_{\ell})|
\right].
$$

或使用 mean predictor。

这样可以判断：

> 某个 anisotropic irrep 是否本身就几乎不可学习。

否则“PG 没提升”可能只是目标 signal 过弱。

---

# 9.3 Anisotropy-conditioned Analysis

这是最重要的 analytical experiment 之一。

---

## Dielectric anisotropy score

定义：

$$
A_\epsilon
=
\frac{
\|\epsilon^{(2)}\|_F
}{
\|\epsilon\|_F+\epsilon_0
}.
$$

将 test set 分为：

- low anisotropy；
- medium anisotropy；
- high anisotropy。

定义 PG gain：

$$
\Delta_{\mathrm{PG}}(A)
=
E_{O(3)}(A)
-
E_{\mathrm{Ours}}(A).
$$

核心 hypothesis：

$$
\frac{
d\Delta_{\mathrm{PG}}
}{
dA
}
>0
$$

至少在 anisotropic signal 具有足够可学习性时成立。

---

## 预期图

x-axis：

$$
A_T
$$

y-axis：

$$
E_{O(3)}
-
E_{\mathrm{Ours}}.
$$

如果随着 anisotropy 增大，Ours 的优势系统性增大，将强力支持核心 motivation。

---

# 9.4 PG-specific Performance Analysis

分别对不同 point groups 报：

- total error；
- irrep-wise error；
- PG gain；
- parameter efficiency。

不要只给一个 mixed-dataset global MAE。

可信的结果可能是：

$$
\Delta_{\mathrm{PG}}(O_h)
\approx0,
$$

但：

$$
\Delta_{\mathrm{PG}}(D_{3d})
>0.
$$

这并非失败。

相反，它支持：

> PG awareness 只在 symmetry-resolved signal 具有信息时真正有价值。

---

# 9.5 Symmetry Violation Metric

定义：

$$
\delta_G(T)
=
\frac1{|G|}
\sum_{g\in G}
\frac{
\|\rho_T(g)T-T\|_F
}{
\|T\|_F+\epsilon
}.
$$

对于 hard-constrained final output：

$$
\delta_G(\hat T_{\mathrm{final}})
\approx0.
$$

但更有价值的是测量：

$$
\delta_G(
\hat T_{\mathrm{pre\text{-}projection}}
).
$$

用于区分：

1. 模型内部已经学习出 symmetry；
2. 最后的 projector 才强行修正 symmetry。

---

# 9.6 Sample-efficiency Curves

比较：

1. scratch O(3)；
2. pretrained O(3)；
3. pretrained + Shared-O3；
4. pretrained + Hierarchical-Full。

训练比例：

$$
1\%,5\%,10\%,25\%,50\%,100\%.
$$

定义达到目标误差 \(e\) 所需 labels：

$$
N_M(e).
$$

进一步定义：

$$
\mathrm{LabelEfficiencyGain}(e)
=
\frac{
N_{\mathrm{scratch}}(e)
}{
N_{\mathrm{pretrained}}(e)
}.
$$

这将直接量化：

> 为什么使用 universally pretrained O(3) backbone。

---

# 9.7 Shared vs Hierarchical Branches

核心消融：

$$
\text{Shared only},
$$

$$
\text{Branch-only},
$$

$$
\text{Hierarchical-Full}.
$$

预期：

- full-data/high-frequency PG：三者差距可能缩小；
- low-data/rare PG：Hierarchical-Full 优势显著。

可以进一步分析 feature norms：

$$
\|z_{\mathrm{shared}}\|,
\qquad
\|w_K\hat c_K^{O(3)}\|,
$$

观察不同 PG 与不同 distortion amplitudes 对 shared / branch paths 的依赖程度。

---

# 9.8 Hierarchy-level vs Representation-level PG Awareness

通过：

$$
\text{Shared-O3}
\rightarrow
\text{Hard-Routing-O3}
\rightarrow
\text{Single-PG-Rep}
\rightarrow
\text{Hierarchical-Full}
$$

回答：

1. one-hot PG identity 本身提供多少信息及多少 discontinuity？
2. explicit PG irreps 提供多少额外信息？
3. compatible parents 与 continuous gates 是否提供超出普通 ensemble 的收益？
4. hierarchy-level 与 representation-level mechanisms 是否 complementary？

---

# 9.9 Full PG vs \(A_1\)-only Intermediate Representation

鉴于已有 point-group tensor work 的结果，必须比较：

## Full-irrep PG

保留所有 intermediate PG irreps。

## \(A_1\)-only

只保留 trivial blocks。

可能出现三种结果。

### Case 1

$$
\text{Full-irrep}
>
A_1\text{-only}.
$$

说明 non-trivial intermediate channels 通过 nonlinear coupling 回流到 target \(A_1\) 有价值。

---

### Case 2

$$
\text{Full-irrep}
\approx
A_1\text{-only}.
$$

说明 global equilibrium tensor 主要由 trivial channels 决定，可以大幅压缩模型。

---

### Case 3

$$
A_1\text{-only}
>
\text{Full-irrep}.
$$

说明当前数据规模不足以稳定训练 non-trivial channels，full representation 可能增加方差。

三种结果都具有科学价值。

---

# 9.10 Parameter-matched Comparison

必须避免 reviewer 质疑：

> PG model 只是因为参数更多所以更好。

因此需要构造：

- matched trainable parameters；
- 或 matched active parameters / sample；
- 或 matched FLOPs / hidden multiplicity。

至少报告：

$$
N_{\mathrm{total}},
$$

$$
N_{\mathrm{trainable}},
$$

$$
N_{\mathrm{active/sample}}.
$$

---

# 9.11 Computational Efficiency

报告：

- training GPU-hours；
- samples / sec；
- inference latency；
- peak GPU memory；
- PG-stage overhead；
- canonicalization preprocessing cost。

本研究不一定追求绝对最快，但应证明：

> specialization gain / computational cost

之间的 trade-off 合理。

---

# 9.12 Rotation / Frame Covariance Test

对每个 test structure 随机采样：

$$
R\in O(3),
$$

验证：

$$
\epsilon_{\mathrm{cov}}
=
\frac{
\|F(Rx)-\rho_T(R)F(x)\|_F
}{
\|F(x)\|_F+\epsilon
}.
$$

同时测试：

- proper rotations；
- inversion / reflection；
- canonicalization 前后的 frame consistency。

理论上应接近 numerical precision / implementation tolerance。

---

# 9.13 Canonicalization Robustness

spglib symmetry detection 依赖 `symprec`。

应测试：

$$
\mathrm{symprec}
\in
\{10^{-2},10^{-3},10^{-4},10^{-5}\}
$$

或以实际数据精度为中心选取范围。

观察：

1. PG assignment stability；
2. prediction stability；
3. tolerance-induced active-set switches；
4. parent residual、gate 与 branch-weight stability；
5. canonical-frame tensor transform consistency。

`symprec` 只用于确定候选群与 embedding，不得直接产生 branch weights。即使候选 active set 因数值容差改变，新加入或移除分支也必须在边界具有趋于零的权重。明显不满足这一条件的样本应单独标记为 embedding-unstable，而不能通过剔除掩盖主要连续性结果。

---

# 9.14 Symmetry-breaking Path Continuity Test

对具有已知 parent \(K_0\) 的结构构造稠密连续畸变路径：

$$
x(t)=x_0+t\,u_\lambda,
\qquad
t\in[-t_{\max},t_{\max}],
$$

其中 \(u_\lambda\) 属于父群的指定 symmetry-breaking irrep，并使 \(G_{x(0)}=K_0\)、\(G_{x(t\ne0)}\subset K_0\)。报告：

- 每个父群的 \(r_{H,\mathrm{cell}}(t)\)、\(r_{H,\mathrm{pos}}(t)\)、总 residual \(r_H(t)\)、preservation score \(s_H(t)\)、edge gate \(a_i(t)\) 与 branch weight \(w_K(t)\)；
- prediction path \(\hat T(t)\)；
- symmetry boundary 左右极限误差；
- 离散 Lipschitz / jump 指标：

$$
J(\delta)
=
\frac{
\|\hat T(x(\delta))-\hat T(x(-\delta))\|_F
}{
\|x(\delta)-x(-\delta)\|+\epsilon
};
$$

- 对 hard routing、ordinary parent average 与 hierarchical full branches 的配对比较；
- 若任务需要 force / response derivatives，额外测试一阶导数连续性。

必须分别覆盖 lattice strain、internal-coordinate displacement 与多个 domain directions。真实一阶相变或结构优化跳到不同能量极小值的样本不用于宣称跨相连续。

---

# 9.15 Chemical Split / Structural OOD

如果计算资源允许，增加：

- composition-disjoint split；
- prototype-disjoint split；
- leave-one-chemistry-family-out。

用于回答：

> pretrained backbone 是否真的提升跨 chemistry transfer，而不只是 random split interpolation。

---

# 10. 预期结果

## 10.1 预期 1：Pretrained O(3) 提高低数据性能

预期在 low-data regime：

$$
E_{\mathrm{pretrained}}
\ll
E_{\mathrm{scratch}}.
$$

且差距在：

$$
1\%-25\%
$$

labels 时最大，在 full-data 时缩小。

这支持：

> pretraining 的主要价值是 sample / label efficiency，而不一定是无限提高 asymptotic accuracy。

---

## 10.2 预期 2：Shared-O3 优于直接 backbone readout

Pretrained backbone features 虽然：

> symmetry-correct

但未必：

> task-aligned。

通过：

$$
O(3)\ TP
\rightarrow
O(3)\ EqMLP
$$

重新组合 angular channels，应获得更好的 tensor-specific representation。

即预期：

$$
\text{Shared-O3}
>
\text{O3-Base}.
$$

---

## 10.3 预期 3：Hierarchical full branches 改善 symmetry-boundary behavior

预期：

$$
\text{Hierarchical-Full}
>
\text{Hard-Routing-O3}
$$

这一优势应同时体现为：

- symmetry-breaking path 上更低的 prediction jump；
- 对 `symprec` 与微扰更稳定；
- anisotropic / higher-order tensor channels 上更好的连续拟合；
- 低数据子群从父群获得更有效的 transfer。

若只提高平均精度而不改善 boundary continuity，则不能宣称该层级设计解决了 hard routing discontinuity。

---

## 10.4 预期 4：PG Representation Gain 与 Anisotropy 相关

不预期 Hierarchical-Full 在所有 point groups 上都大幅领先。

更可信的结果为：

- \(O_h\) elastic：Hierarchical-Full 与 O(3) 接近；
- high-anisotropy subsets：Hierarchical-Full 明显改善；
- dielectric \(\ell=0\)：差距较小；
- dielectric \(\ell=2\)：差距明显；
- elastic \(\ell=4\)：可能出现更强 PG benefit。

即：

$$
\boxed{
\text{PG gain}
\uparrow
\quad
\text{with}
\quad
\text{learnable anisotropic content}
}
$$

---

## 10.5 预期 5：Shared + Hierarchical Branches 在 Long-tail 下最好

当 target PG labels 很少时：

$$
\text{Hierarchical-Full}
>
\text{Branch-only}.
$$

随数据增加，PG branches 可以独立学习更多，二者差距减小。

这将成为本研究区别于纯 PG-from-scratch model 的核心结果。

---

## 10.6 预期 6：Hard-constrained Readout 保证严格 Tensor Symmetry

Hierarchical-Full model 的 final output 应满足：

$$
\delta_G
\approx0.
$$

如果 Hierarchical-Full 在 readout 前 intermediate prediction 本身也更接近 fixed subspace，则可以进一步证明：

> internal representation 本身更加 symmetry-aware。

---

# 11. 哪些结果会削弱核心 Hypothesis？

## Case A：O(3)+Projection ≈ Hierarchical-Full

说明 explicit PG representation 对 accuracy 无额外帮助，主要价值只是 output constraints。

此时可转向研究：

- efficiency；
- compact readout；
- low-data regime；
- higher-order tensors；
- stronger anisotropy；
- approximate symmetry。

---

## Case B：Hard-Routing-O3 ≈ Hierarchical-Full 且明显 > Shared-O3

说明：

> PG identity 很重要，但 parent branches、continuous gates 与 finite-group TP/MLP 未提供额外收益。

此时可以简化模型并将贡献聚焦到：

> hard symmetry-routed parameter-efficient adaptation，并明确承认其连续性限制。

---

## Case C：Single-PG-Rep > Hard-Routing-O3，但 Hierarchical-Full ≈ Single-PG-Rep

说明 representation-level PG awareness 有用，但 current-plus-parent hierarchy 在该数据设置下没有带来额外收益。

---

## Case D：A1-only consistently > Full PG

说明当前数据规模下 non-trivial intermediate PG irreps 主要增加 variance。

此时可将轻量化：

$$
A_1\text{-only}
$$

版本作为最终架构。

---

## Case E：Pretraining 对 Low-data 无明显帮助

需要检查：

- pretrained task/domain mismatch；
- backbone features 是否包含足够高阶 channels；
- frozen backbone 是否限制过强；
- canonicalization 是否改变 backbone distribution；
- target tensor 是否依赖预训练未覆盖的 long-range physics。

---

## Case F：Hierarchical-Full 精度提高但 boundary jump 不下降

说明多分支收益可能只来自额外容量或 ensemble，而 symmetry-breaking gates、parent embedding 或 canonical frame transport 没有实现预期连续极限。此时不得声称模型解决了结构—表示不连续；必须依次检查：

- 子群 branch weight 是否在父群恢复时数值趋于零；
- DAG 多路径聚合是否重复计数；
- candidate parent set 是否由 hard tolerance 突然改变；
- atom matching / lattice correspondence 是否跨路径连续；
- inverse subduction 与 de-canonicalization 是否使用一致 embedding。

---

# 12. 风险与备选方案

## 12.1 Canonicalization 非唯一或不稳定

### 风险

spglib standard setting 并非严格数学唯一，symmetry detection 还受到 tolerance 影响。

### 应对

- 保存完整 transform metadata；
- 为每个 active branch 保存具体 group embedding，并在公共 O(3) tensor space 中融合；
- canonicalization robustness test；
- 对 embedding-unstable samples 单独标记；
- gate 必须由连续几何量构造，不能直接使用 `symprec` one-hot label。

---

## 12.2 多个完整 PG Branches 导致参数与计算碎片化

### 风险

rare PG 数据量很小。

### 应对

- shared experts；
- 参数共享的轻量 PG TP/MLP block；
- first benchmark 先覆盖有足够数据的 PG；
- minimum support threshold；
- 第一版只枚举 benchmark 中物理明确的 embedded subgroup DAG paths，但对已纳入的当前群激活全部 compatible parents；
- 使用 hierarchy-aware parameter sharing：

$$
\text{parent branch parameters}
+
\text{child residual parameters}.
$$

不得用不连续的 hard top-\(k\) 作为默认降本方式。若推理时进行数值剪枝，必须证明被剪分支权重低于预设误差预算，并单独报告未经剪枝的连续性结果。

---

## 12.3 备选：分支内 K-conditioned O(3) adapter

主模型不在完整 PG branch 内额外放置 O(3) expert；每个分支均直接从共享 O(3) task representation 开始 subduction。若后续实验表明 low-data、long-tail PG 或强各向异性设置需要更强的分支容量，可将下式作为**可选增强与消融项**：

$$
z_K=z_{\mathrm{shared}}+E_K^{O(3)}(h).
$$

其中 \(E_K^{O(3)}\) 是按点群 \(K\) 索引参数、但保持 O(3)-equivariance 的 adapter。启用它时，必须进行 parameter-matched ablation，以区分性能收益来自额外容量还是后续的 PG representation learning。

---

## 12.4 PG Stage 计算复杂

### 风险

finite-group CG tables / repeated multiplicities 实现复杂。

### 应对

第一版限制：

- \(\ell_{\max}\) 与 target task 对齐；
- 1 个 PG TP + EqMLP block；
- 只支持 benchmark 中核心 PG；
- 验证后再扩展 32 groups。

---

## 12.5 Pretrained Backbone 不适配 Canonicalized Cell Convention

### 风险

如果 backbone 在 primitive cell / arbitrary orientation 上预训练，canonical conventional cell 可能改变：

- 图规模；
- neighbor statistics；
- distribution。

### 应对

- canonicalization 优先控制 orientation / setting；
- 不盲目扩大 conventional cell；
- 尽可能使用 primitive representation；
- 保存 canonical orientation metadata；
- 对 primitive vs conventional 实现做比较。

---

## 12.6 Tensor Label 与 Structure Symmetry 不一致

### 风险

数据库中的 tensor 可能对应不同 relaxation 或不同 DFT structure。

### 应对

优先采用 calculation-matched structures。

并自动验证：

$$
\max_{g\in G}
\frac{
\|\rho_T(g)T-T\|
}{
\|T\|
}
<
\tau.
$$

不满足时：

- 剔除；
- 降级 symmetry；
- 或单独分析。

---

# 13. 开题阶段工作计划与里程碑

## Phase 0：数学与工程验证

完成：

- O(3) tensor irrep decomposition；
- dielectric Cartesian ↔ irrep transformation；
- elastic Cartesian ↔ irrep transformation；
- Reynolds projector；
- spglib canonicalization + frame tracking；
- tensor de-canonicalization。

**交付物：**

> group / tensor test suite

---

## Phase 1：Minimal PG Layer

优先选择：

$$
D_{2h},
\quad
D_{3d},
\quad
O_h
$$

等代表 PG。

完成：

- subduction matrices；
- PG irreps；
- PG CG / tensor products；
- PG equivariance tests；
- synthetic symmetry task。

**交付物：**

> standalone PG-equivariant block

---

## Phase 2：Pretrained Backbone + Shared O(3) Adapters

完成：

- 对接 pretrained O(3) backbone；
- frozen backbone pipeline；
- shared O(3) TP + EqMLP；
- O3-Base；
- Shared-O3。

**交付物：**

> reliable pretrained O(3) tensor downstream baseline

---

## Phase 3：Hierarchical Full PG Branches

实现：

- compatible parent DAG paths 与 group-embedding metadata；
- periodic parent-group operation residuals；
- continuous edge gates 与 stick-breaking weights；
- 每个激活群独立的 subduction、PG TP/MLP、PG-equivariant norm 与 constrained readout；
- inverse subduction 后的 common-space weighted fusion；
- 融合后的共享 O(3)-equivariant readout（可选）。

**交付物：**

> Hierarchical-Full model

---

## Phase 4：PG-controlled Benchmark

优先完成：

- JARVIS tensor 中的 dielectric representative PG subsets；
- JARVIS tensor 中的 elastic representative PG subsets；
- MatTen elastic 中的 high-symmetry control 与 lower-symmetry PG subsets；
- JARVIS-DFPT BEC 中的 representative PG / Wyckoff-orbit subsets，并验证 no-pooling shared node-wise readout 及其隐式 Wyckoff equivariance。

具体 PG 仅在训练集完成样本量与 embedding-stability 审计后冻结；优先考察 \(D_{3d}\) dielectric、\(O_h\) elastic 与 \(D_{2h}\) elastic，但不为凑齐类别改变 published test split。

---

## Phase 5：SOTA Benchmark + Low-data Analysis

完成：

- JARVIS tensor published benchmark；
- MatTen elastic published benchmark；
- JARVIS-DFPT BEC benchmark；
- MP-Dielectric BEC 字段与 structure-site alignment 审计；
- learning curves；
- long-tail PG；
- parameter matching；
- runtime profiling。

---

## Phase 6：Representation Analysis 与论文组织

完成：

- anisotropy vs gain；
- irrep-wise error；
- PG gain heatmap；
- expert contribution；
- symmetry violation；
- A1-only vs Full PG；
- reviewer FAQ；
- failure analysis。

---

# 14. 预期创新点与贡献

## Contribution 1：Universal-to-discrete Symmetry Specialization Framework

提出：

$$
\boxed{
O(3)
\rightarrow
G
\rightarrow
\mathrm{Fix}_G(V_T)
}
$$

的层次化 representation-learning framework，将 universal continuous-symmetry pretraining 与 material-specific discrete-symmetry specialization 统一起来。

---

## Contribution 2：Continuity-aware Current-plus-parent Full Branches

提出 current point group 与 compatible parent groups 各自运行完整 PG-specialized branch，并以当前结构相对父群 operations 的 normalized residual 构造满足父群极限条件的层级权重：

$$
\hat c_{\mathrm{out}}^{O(3)}(x)
=
R_t^{O(3)}\!\left[
\sum_{K\in\mathcal A(x)}
w_K(x)
\mathcal U_{K,t}^{\dagger}J_{K,t}R_K\!\left(
\operatorname{Pool}_i\!\left[
\operatorname{Norm}^{PG}_K\!\left(
M_K^{PG}\!\left(T_{K,\mathrm{graph}}^{PG}(\mathcal U_Kz_K,\mathcal U_Ke)_i\right)
\right)
\right]
\right)
\right].
$$

该设计同时利用：

- cross-PG transfer；
- PG-specific representation learning；
- group-subgroup hierarchy；
- symmetry-breaking path continuity。

---

## Contribution 3：Hierarchy-level 与 Representation-level Symmetry Awareness 的因果分解

通过系统 ablation 单独测量：

- hard point-group routing vs continuous hierarchical gates；
- ordinary parent averaging vs symmetry-breaking-weighted averaging；
- explicit PG irreps / fusion rules；
- final symmetry projection / readout。

避免把所有收益模糊归因于：

> symmetry-aware model。

---

## Contribution 4：面向 Tensor Task 的 Symmetry-aware Benchmark Methodology

不只报告 mixed-dataset average，而按：

- point group；
- anisotropy；
- irrep channel；
- label budget；
- symmetry compliance；

系统解释：

> 什么时候 PG awareness 真正有价值？

---

## Contribution 5：Parameter-efficient Tensor Specialization of Pretrained Atomistic Models

研究 universal equivariant pretraining 如何降低 tensor downstream labels 需求，为 materials foundation models 向高阶 response properties 迁移提供更一般的范式。

---

# 15. 建议的主论文图表结构

## Figure 1 — Method Overview

```text
Relaxed crystal
     |
spglib canonicalization -----> Gx, parent DAG, r_H(x), Qx
     |
Pretrained O(3) backbone
     |
Shared O(3) adaptation
     |
     +--------------------+--------------------+
     |                    |                    |
Full branch K0       Full branch K1       Full branch Gx
O3 -> K0             O3 -> K1             O3 -> Gx
Graph PG TP           Graph PG TP           Graph PG TP
Node EqMLP            Node EqMLP            Node EqMLP
PG norm               PG norm               PG norm
global: pool + Fix_K readout / BEC: shared node-wise equivariant readout
     |                    |                    |
     +------ inverse subduction to common O(3) space -----+
                              |
              hierarchical weighted average w_K(r)
                              |
                 shared O(3) readout (optional)
                              |
                     irrep -> Cartesian
                              |
                     de-canonicalization
```

---

## Figure 2 — Where Does PG Awareness Help?

x-axis：

$$
\text{anisotropy score}
$$

y-axis：

$$
E_{O(3)}
-
E_{\mathrm{Ours}}.
$$

---

## Figure 3 — Data Efficiency

x-axis：

$$
\text{number / fraction of tensor labels}
$$

y-axis：

$$
\text{test error}
$$

比较：

- scratch；
- pretrained；
- pretrained + shared；
- pretrained + full PG。

---

## Figure 4 — Per-PG / Per-irrep Heatmap

行：

> point groups

列：

$$
\ell=0,2,4
$$

颜色：

> relative error reduction

---

## Table 1 — Standard SOTA Benchmark

JARVIS tensor / MatTen elastic published test sets，以及 JARVIS-DFPT BEC fixed test set。

---

## Table 2 — Causal Ablation

比较：

- O3-Base；
- Projection；
- Shared；
- Hard-Routing-O3；
- Single-PG-Rep；
- Parent-Average-Control；
- Branch-only；
- Hierarchical-Full。

---

## Table 3 — Efficiency

报告：

- total parameters；
- trainable parameters；
- active parameters；
- FLOPs；
- latency；
- label efficiency。

---

# 16. Reviewer FAQ

## Q1：为什么一定要先用一个 O(3)-equivariant backbone？

核心回答有三层。

### 第一层：Universal Pretraining

Tensor datasets 相对较小，而 large-scale atomistic pretraining 可以提供 transferable chemical / geometric knowledge。

因此：

$$
\text{pretraining}
\rightarrow
\text{better downstream label efficiency}.
$$

---

### 第二层：Directional Information

与 invariant backbone 不同，O(3)-equivariant representation 保留：

$$
h^{(0)},
h^{(1)},
h^{(2)},
\ldots
$$

等方向相关信息。

这与 tensorial targets 的表示结构天然兼容。

---

### 第三层：Universal Representation Interface

由于：

$$
G\subset O(3)
$$

对所有 crystallographic point groups 成立，因此 O(3) 提供一个统一的母表示空间：

$$
\boxed{
O(3)
\rightarrow
G
\rightarrow
\mathrm{Fix}_G(V_T)
}.
$$

因此 point-group symmetry 应被用于：

> **specialize universal representation**

而不是替代 universal representation。

推荐回答：

> We use a pretrained O(3)-equivariant backbone not only to reuse universal chemical and geometric knowledge, but also to preserve directional information in a representation space naturally compatible with tensorial targets. Since every crystallographic point group is a subgroup of O(3), this representation provides a universal interface for downstream point-group specialization.

---

## Q2：为什么不直接做 PG-equivariant network from scratch？

因为每个 PG 的数据远少于完整数据集。

如果分别训练：

$$
\theta_G
\leftarrow
\mathcal D_G,
$$

会造成严重的 data fragmentation。

另一方面，chemistry / local geometry 大量知识是跨 PG transferable 的。

因此本研究采用：

$$
\theta_{\mathrm{pre}}
+
\Delta\theta_{\mathrm{shared}}
+
\left\{
w_K(r(x))\Delta\theta_K
\right\}_{K\in\mathcal A(x)}.
$$

---

## Q3：为什么不直接做 Output Projection？

Reynolds projection：

$$
P_G(T)
=
\frac1{|G|}
\sum_{g\in G}
\rho_T(g)T
$$

确实能够保证 tensor 合法。

所以它必须作为强 baseline。

但 projection 只改变：

> output space

而不改变：

> internal representation learning。

本研究真正检验的是：

> 显式 PG specialization 是否能够改善 anisotropic representation、sample efficiency 和 long-tail performance。

---

## Q4：Canonicalization 后为什么还需要 Equivariance？

Canonicalization 消除了大部分 arbitrary global frame freedom，但：

1. standardized frame 未必严格唯一；
2. 存在 symmetry-equivalent canonical choices；
3. pretrained O(3) backbone 本身已经包含 directional representations；
4. 我们希望在 PG irreps 中进行 physically structured nonlinear interactions，而不是固定 frame 后全部退化为 ordinary MLP。

因此 canonicalization 与 equivariance 并非互斥。

---

## Q5：这和 PGEqNN 有什么本质区别？

PGEqNN 更直接的问题是：

> 在从头训练的 equivariant GNN 中，将 SO(3) rotational order 按 PG irreps 进一步 partition 是否有价值？

本研究的问题是：

> 大规模预训练得到的 universal O(3) representation，如何通过 parameter-efficient shared adaptation 与 current-plus-parent full branches，被连续地特化到结构的 discrete stabilizer symmetry？

新增维度包括：

- universal pretraining；
- sample efficiency；
- shared vs PG-specific transfer；
- symmetry-breaking-coordinate gating；
- current / compatible-parent complete PG branches；
- hierarchy-level vs representation-level awareness；
- low-data / long-tail PG specialization。

---

## Q6：如果 \(A_1\)-only 已经足够，为什么需要 Full PG TP？

这是需要实验回答的问题。

虽然 final equilibrium tensor 一定位于：

$$
\mathrm{Fix}_G(V_T),
$$

即 trivial irreps，但 intermediate non-trivial PG irreps 仍可能通过 tensor products：

$$
\Gamma
\otimes
\Gamma'
\supset
A_1
$$

影响最终 trivial channels。

因此必须比较：

$$
\text{Full PG}
\quad\text{vs.}\quad
A_1\text{-only}.
$$

---

## Q7：加入 Born Effective Charge 是否需要另一套模型？

不需要另一套 backbone 或 PG branch。Born effective charge 是 atom-resolved tensor，因此与 global dielectric / elastic 的主要实现差别是：

$$
\text{BEC: no pooling}
\qquad\text{vs.}\qquad
\text{dielectric / elastic: global pooling}.
$$

其前提是 PG-equivariant PBC graph 正确处理：

- \((R_g,t_g)\) 诱导的 atomic-site permutation；
- periodic image / cross-boundary edge permutation；
- feature fiber 上的 \(\rho_K(R_g)\) 变换。

在该前提下，Wyckoff-orbit equivariance 和 site-stabilizer constraint 都是图网络联合等变性的直接推论，不需要额外的 Wyckoff module 或 orbit expansion。因此统一模型使用 `output_scope = global | site` 控制 pooling：共享 node-wise 表示与消息传递，BEC 只增加一个在所有节点间共享的 tensor readout。

---

# 17. 开题阶段最小可行实验（MVP）

如果计算资源有限，可以先只做三个数据 setting。

## Dataset

### Positive anisotropy case

JARVIS tensor dielectric：

$$
D_{3d}.
$$

### Negative/control case

MatTen elastic：

$$
O_h.
$$

### Lower-symmetry case

MatTen elastic：

$$
D_{2h}.
$$

---

## Models

只需要先训练：

1. O(3)-Base；
2. O(3)+Projection；
3. Shared-O3；
4. Hard-Routing-O3；
5. Single-PG-Rep；
6. Parent-Average-Control；
7. Branch-only；
8. Hierarchical-Full。

---

## Metrics

至少报告：

- total tensor Fnorm / MAE；
- irrep-wise MAE；
- anisotropy-conditioned gain；
- 10% / 100% labels；
- symmetry violation；
- active parameters。

---

## Early Go / No-Go Criterion

如果观察到：

$$
\text{Hierarchical-Full}_{D_{3d}}
>
\text{O3}_{D_{3d}},
$$

同时：

$$
\text{Hierarchical-Full}_{O_h}
\approx
\text{O3}_{O_h},
$$

并且 low-data 下：

$$
\text{Hierarchical-Full}
>
\text{Branch-only},
$$

则核心 research thesis 已得到初步支持。

---

# 18. 更完整的实验矩阵

| Experiment | 主要比较 | 回答的问题 |
|---|---|---|
| Pretraining | scratch vs pretrained | universal pretraining 是否提高 label efficiency？ |
| Shared adaptation | O3-Base vs Shared-O3 | pretrained features 是否需要 task adaptation？ |
| Shared TP backend | `full_o3` vs `so2_reduced`（原生规模与参数匹配） | SO(2) reduction 的精度—效率收益是否独立于参数量？ |
| Hard PG routing | Shared-O3 vs Hard-Routing-O3 | PG identity 本身是否有信息，并造成多大 boundary jump？ |
| PG representation | Shared-O3 vs Single-PG-Rep | finite-group representation 是否有额外价值？ |
| Parent ensemble control | Parent-Average-Control vs Hierarchical-Full | 收益是否来自物理 gate 而非普通 ensemble？ |
| Full combination | Hard-Routing-O3 / Single-PG-Rep vs Hierarchical-Full | hierarchy-level 与 representation-level awareness 是否互补？ |
| Output constraint | O3+Projection vs Hierarchical-Full | internal specialization 是否超过后处理？ |
| Long-tail | Branch-only vs Hierarchical-Full | shared experts 是否缓解 PG data fragmentation？ |
| Path continuity | Hard-Routing-O3 vs Parent-Average-Control vs Hierarchical-Full | 父群极限是否连续？ |
| Intermediate irreps | Full PG vs A1-only | non-trivial intermediate irreps 是否必要？ |
| PG-TP parameterization | static radial coefficients vs per-group full MLP vs shared router + low-rank PG head | 动态 path weighting 是否必要，以及共享低秩适配能否兼顾容量、参数量与 rare-PG 泛化？ |
| Post-fusion O(3) readout | identity vs shared O(3)-equivariant readout | 公共空间中的最终 channel/copy mixing 是否有额外价值？ |
| Backbone training | frozen vs partial vs full | specialization 是否 parameter-efficient？ |
| Anisotropy | low/med/high anisotropy | PG gain 是否随 anisotropy 增大？ |
| Harmonic error | \(\ell=0,2,4\) | improvement 来自哪些 angular channels？ |
| Symmetry | \(\delta_G\) | 模型是否严格满足 symmetry？ |
| Frame covariance | random rotations | canonicalization pipeline 是否正确？ |
| Efficiency | params/FLOPs/time | gain 是否值得计算开销？ |

---

# 19. 预期论文核心叙事

整篇工作的逻辑最好不要写成：

> “Point group 能提高 tensor prediction。”

这个论述太宽泛，也太容易被现有工作覆盖。

更好的叙事是：

### Observation 1

现代 atomistic foundation / pretrained models 能学习高质量通用表示，但 downstream tensor labels 很少。

### Observation 2

Tensor targets 需要 directional equivariant information，因此 O(3)-equivariant pretrained representations 是天然接口。

### Observation 3

Relaxed equilibrium crystals 还拥有额外的 structure-dependent discrete stabilizer symmetry：

$$
G_x\subset O(3).
$$

### Question

如何在不丢失 universal pretraining transferability 的情况下利用这个离散 symmetry？

### Proposed answer

先在 universal O(3) representation space 中进行：

$$
\text{shared task adaptation},
$$

再对 current point group 与 compatible parents 分别执行：

$$
O(3)\downarrow K
\rightarrow
PG_K\text{-TP/MLP}
\rightarrow
\mathrm{Fix}_K(V_T),
$$

然后 inverse-subduce 回公共 O(3) tensor space，以 symmetry-breaking gates 做层级加权融合，并可通过共享 O(3)-equivariant readout 完成最终的 channel/copy mixing。模型由此不仅利用 discrete stabilizer symmetry，也显式满足父群恢复极限下的连续性条件。

因此核心 pipeline 为：

$$
\boxed{
\text{Universal Representation}
\rightarrow
\text{Task Adaptation}
\rightarrow
\text{Material-Symmetry Specialization}
\rightarrow
\text{Physical Tensor}
}
$$

---

# 20. 核心 Thesis

本研究最终希望验证的不仅是一种模型结构，而是一种更一般的 representation-learning principle：

> **Universal symmetries are suitable for pretraining; material-specific stabilizer symmetries are suitable for downstream specialization.**

即：

$$
\boxed{
\textbf{
Pretrain at the universal symmetry level,
specialize at the material symmetry level.
}
}
$$

对于晶体张量性质预测，这一原则具体体现为：

$$
\boxed{
O(3)
\rightarrow
\{K\in\mathcal A(x)\}
\rightarrow
\text{common-space hierarchical fusion}
\rightarrow
\text{shared O(3) readout}
\rightarrow
\mathrm{Fix}_{G_x}(V_T)
}
$$

其中：

- \(O(3)\)：跨所有材料共享的 universal geometric symmetry；
- \(\mathcal A(x)\)：当前 discrete stabilizer \(G_x\) 与所有物理兼容父群的 embedded active set；
- \(\mathrm{Fix}_{G_x}(V_T)\)：当前物性真正允许的 tensor subspace。

在数据层面，对应：

$$
\boxed{
\text{large-scale generic data}
\rightarrow
\text{small tensor data}
\rightarrow
\text{small per-PG data}
}
$$

在参数层面，对应：

$$
\boxed{
\theta_{\mathrm{pre}}
\rightarrow
\Delta\theta_{\mathrm{shared}}
\rightarrow
\{w_K(r(x))\Delta\theta_K\}_{K\in\mathcal A(x)}
}
$$

这三组对应关系共同构成本课题的理论与实验主线。

---

# 21. 参考工作框架

开题阶段建议重点阅读和对比以下类别的工作。

## Tensor-equivariant models

- Tensor Field Networks
- e3nn
- MatTen
- ETGNN
- AnisoNet

重点理解：

- Cartesian tensor ↔ spherical tensor；
- irreducible decomposition；
- equivariant readout。

---

## Crystal-symmetry-aware tensor models

- GMTNet
- PGEqNN
- GoeCTP

重点理解：

- global O(3) covariance 与 crystal symmetry 的关系；
- canonicalization；
- space-group / point-group information；
- output symmetry enforcement；
- point-group-adapted basis。

---

## Pretrained atomistic models for downstream tensor tasks

重点关注：

- DTNet；
- pretrained equivariant interatomic potentials；
- foundation atomistic representations。

核心问题：

> pretrained latent representation 是否真正保留 downstream tensor 所需的 angular information？

---

## Structural symmetry breaking 与 continuous symmetry

- Landau / isotropy-subgroup description of structural phase transitions；
- parent-space-group irreps 与 symmetry-mode decomposition；
- continuous symmetry measures；
- group–subgroup embeddings、Wyckoff splitting 与 domain variants。

重点理解：

- 如何构造 parent-symmetric reference \(P_Kx\)；
- 如何从 periodic parent-group operation residual 得到连续且 invariant 的 gate；
- 如何区分 point-group breaking 与 translation / cell-multiplication transitions；
- 如何保证 active-set 切换时新分支具有零边界权重。

---

# 22. 开题阶段需要进一步确认的技术问题

在正式 implementation 前，需要逐项确认：

1. 主 pretrained O(3) backbone 选型；
2. backbone 能输出哪些 \(\ell\)；
3. dielectric 是否要求至少 \(\ell_{\max}=2\)；
4. elastic 是否要求 backbone / expert 支持 \(\ell_{\max}=4\)；
5. 是否允许 expert 通过 TP 从低阶 channels 构造 \(\ell=4\)；
6. `shared_tp_backend` 使用 `full_o3` 还是带 parity/gauge 处理的 `so2_reduced`；
7. `output_scope` 按任务使用 `global`（dielectric / elastic）还是 `site`（BEC）；
8. PBC graph 在 \((R_g,t_g)\) 下的 site/edge permutation equivariance，以及 BEC acoustic sum rule policy；
9. MP-Dielectric 的数据 release、底层 DFPT task 可访问性及 structure-site alignment；
10. 32 crystallographic PG 的 irrep convention；
11. subduction matrix convention；
12. finite-group CG coefficient convention；
13. PG-TP allowed-path enumeration、fusion-copy index \(\eta\) 与 path ordering 是否在 checkpoint 中固定；
14. router 默认只读固定宽度 O(3)-trivial endpoint channels，高阶 PG-\(A_1\) summary 仅作 ablation；
15. \(r_{\mathrm{route}}\)、per-group full head 与 shared-low-rank head 的参数匹配；
16. repeated irreps 的 multiplicity handling；
17. Cartesian tensor basis convention；
18. Voigt notation 仅用于 evaluation，还是训练；
19. spglib standardization / idealization policy；
20. tensor label 是否需要先进行 symmetry cleaning；
21. improper rotations / parity conventions；
22. primitive vs conventional cell；
23. batch 内不同 active branch sets 的动态计算组织；
24. full-branch 参数存储、hierarchical weights 与 common-space fusion；
25. compatible parent space-group embeddings、\((W_g,t_g)\) convention、Wyckoff / atom correspondence 与 domain variants；
26. lattice / position residual 的无量纲化、\(\lambda_{\mathrm{cell}}/\lambda_{\mathrm{pos}}\)、closest-lattice-vector 与 assignment policy；
27. gate scale \(\sigma_i\) 的初始化、单位无关 calibration 与约束；
28. subgroup DAG 多路径权重聚合是否避免重复计数；
29. active-set 边界的 \(C^0\) continuity，以及导数任务所需的 \(C^1/C^2\) regularity；
30. distributed training 时 rare PG 的 batch imbalance；
31. spglib、pymatgen、MultiPie 与 e3nn/backbone 的版本锁定及 license/citation 记录；
32. 是否保证 spglib 为唯一 canonicalization source，并完整缓存 `SymmetryRecord`；
33. MultiPie 数据转换后的 registry checksum、gauge validation 与 checkpoint compatibility；
34. `graph_backend` 使用 pretrained-backbone cutoff、CrystalNN reproduction 还是 fixed-radius ablation。

---

# 23. 一个可能的正式模型名称

后续可以考虑以下命名。

### PG-Adapter

**Point-Group Equivariant Adapter for Pretrained Atomistic Models**

简单，但略普通。

### SymmAdapt

强调 symmetry specialization。

### UniPG

强调：

> universal O(3) → point group

### StabilizerNet / StabilizerAdapter

更理论化，直接强调：

$$
G_x=\mathrm{Stab}(x).
$$

### PGS-Former / PGS-Net

Point-Group-Specialized network。

目前从论文概念角度，个人更偏向：

> **Stabilizer-aware Equivariant Adaptation**

因为 point group 本质上就是 relaxed crystal 在 \(O(3)\) 中的 stabilizer subgroup。

---

# 24. 一句话总结

本研究不是简单地“把 point group 加进 tensor predictor”，而是尝试建立一种更一般的 materials foundation-model specialization 范式：

$$
\boxed{
\textbf{
Pretrain at the universal symmetry level,
specialize at the material symmetry level.
}
}
$$

即：

利用大规模 O(3)-equivariant pretraining 学习跨材料、跨点群可迁移的化学与几何知识，再让当前点群与所有物理兼容父群分别运行完整 PG-specialized branch；各分支输出提升回公共 O(3) tensor space，并依据当前结构相对父群对称操作的连续 residuals 进行层级加权平均。该框架在严格物理允许的 tensor subspace 中完成预测，同时把父群恢复极限下的结构—表示连续性作为显式架构条件与实验验收指标。
