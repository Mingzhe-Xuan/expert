# Global GMTNet with hierarchical point-group experts

## 1. Scope and design contract

This document specifies the planned end-to-end model for crystal tensor prediction. A pinned
GMTNet branch supplies the global $O(3)$-equivariant representation. Point-group (PG) experts form
an auxiliary residual branch. Within every maximal point-group chain, edge-residual gates perform
stick breaking; across chains, an immediate-parent residual softmax selects the plausible subgroup
direction. One learned scalar scales the complete auxiliary branch before it is added to the GMTNet
crystal feature.

The design makes four commitments:

1. The global branch remains available for every structure and is not replaced by discrete routing.
2. Every PG expert is pooled at crystal level before hierarchical fusion.
3. A PG shared by several chains is evaluated once; its chain contributions are combined exactly.
4. Routing uses invariant scalar weights, so it does not change the $O(3)$ carrier law.

This is the normative contract for the incremental implementation linked in Section 18. The legacy
fixed chain-length prior and node-level expert fusion remain unchanged in the original models;
the new global-plus-experts model uses the equations below.

## 2. Symbols

### 2.1 Input, graph, and batch symbols

| Symbol | Definition |
|---|---|
| $x$ | One periodic crystal structure. |
| $B$ | Number of crystals in a mini-batch. |
| $b\in\{1,\ldots,B\}$ | Crystal index. |
| $V_b$, $E_b$ | Atoms and directed periodic neighbor edges of crystal $b$. |
| $N_b=\|V_b\|$ | Number of atoms in crystal $b$. |
| $Z_i$ | Atomic number of atom $i$. |
| $\mathbf r_i\in\mathbb R^3$ | Cartesian coordinate of atom $i$. |
| $\mathbf n_{ij}\in\mathbb Z^3$ | Periodic image offset for edge $i\to j$. |
| $\mathbf L_b\in\mathbb R^{3\times3}$ | Lattice matrix of crystal $b$. |
| $\mathbf d_{ij}$ | Periodic relative edge vector. |
| $d_{ij}=\|\mathbf d_{ij}\|_2$ | Edge distance. |
| $y_b$ | Ground-truth response tensor. |
| $\widehat y_b$ | Model prediction. |

The periodic edge geometry is

\[
\mathbf d_{ij}
=
\mathbf r_j+\mathbf L_b\mathbf n_{ij}-\mathbf r_i,
\qquad
d_{ij}=\|\mathbf d_{ij}\|_2.
\]

Only relative geometry enters the routing residual described below, matching the geometry visible to
the graph model.

### 2.2 Representation symbols

| Symbol | Definition |
|---|---|
| $g\in O(3)$ | A proper or improper orthogonal transformation. |
| $\rho(g)$ | Its matrix representation on the public feature carrier. |
| $\mathcal V$ | The shared finite-dimensional $O(3)$ carrier space. |
| $C=\dim\mathcal V$ | Flattened carrier dimension. |
| $h_i^{(t)}\in\mathcal V$ | Node feature after GMTNet layer $t$. |
| $h_i=h_i^{(T)}$ | Final global node feature before crystal pooling. |
| $g_b\in\mathcal V$ | GMTNet global crystal feature. |
| $z_{b,H}\in\mathcal V$ | Crystal-pooled output of PG expert $H$. |
| $z_b^{\mathrm{PG}}\in\mathcal V$ | Hierarchically fused PG auxiliary feature. |
| $\widetilde g_b\in\mathcal V$ | Global-plus-expert fused crystal feature. |

The carrier can be written as a direct sum of irreducible representations,

\[
\mathcal V
=
\bigoplus_{\ell,p}m_{\ell,p}V_{\ell,p},
\]

where $\ell$ is angular degree, $p$ is parity, and $m_{\ell,p}$ is multiplicity. Additions and
scalar multiplications below are performed between tensors with the same ordered layout. If an expert
uses a different internal layout, a learned equivariant adapter maps it to $\mathcal V$ before
pooling or fusion.

### 2.3 Point-group hierarchy symbols

| Symbol | Definition |
|---|---|
| $G_b$ | Strictly detected current PG of crystal $b$. |
| $\mathcal D_b$ | Validated material PG DAG rooted upward from $G_b$. |
| $\mathcal P_b=\{P_{b,k}\}_{k=1}^{K_b}$ | Complete set of maximal current-to-root chains. |
| $K_b$ | Number of maximal chains for crystal $b$. |
| $D_{b,k}$ | Number of cover edges in chain $k$. |
| $H_{b,k,d}$ | PG at depth $d$ on chain $k$. |
| $e_{b,k,d}$ | Cover edge $H_{b,k,d}\to H_{b,k,d-1}$. |
| $\mathcal A_b$ | Set of unique PGs occurring in all active chains. |

Every chain is oriented from the current group toward its ancestors:

\[
P_{b,k}:
H_{b,k,0}=G_b
\leftarrow H_{b,k,1}
\leftarrow\cdots\leftarrow
H_{b,k,D_{b,k}}.
\]

Thus $H_{b,k,1}$ is the immediate parent of $G_b$ on chain $k$, and

\[
e_{b,k,d}
=
\bigl(H_{b,k,d}\to H_{b,k,d-1}\bigr).
\]

The short notation $H_{b,k}$ always means $H_{b,k,1}$, not the whole chain and not an expert
feature.

### 2.4 Routing and parameter symbols

| Symbol | Status | Definition |
|---|---|---|
| $r_{b,e}\ge0$ | geometry-derived | Residual for PG cover edge $e$. |
| $s_e\in\mathbb R$ | learned | Unconstrained edge-scale parameter. |
| $\sigma_e=\operatorname{softplus}(s_e)+\epsilon_\sigma$ | learned positive | Shared positive scale for edge $e$. |
| $q_{b,e}=r_{b,e}/\sigma_e$ | derived | Dimensionless normalized residual. |
| $a_{b,e}=1-e^{-q_{b,e}^2}$ | derived | Continue-to-current stick-breaking gate. |
| $\omega_{b,k,d}$ | derived | Conditional weight of depth $d$ within chain $k$. |
| $E_{b,k}$ | derived | Immediate-parent chain energy. |
| $\tau>0$ | fixed initially | Between-chain softmax temperature. |
| $\pi_{b,k}$ | derived | Weight of chain $k$. |
| $\alpha_{b,H}$ | derived | Effective coefficient of unique PG $H$. |
| $\ell\in\mathbb R$ | learned | Unconstrained whole-branch fusion logit. |
| $\lambda=\operatorname{sigmoid}(\ell)$ | learned bounded | Whole PG-branch scale. |

The initial implementation keeps $\tau$ fixed so the effect of the single learned branch scale is
identifiable. A later ablation may learn
$\tau=\operatorname{softplus}(t)+\epsilon_\tau$, but that is not part of the initial contract.
Edge scales $\sigma_e$ remain learned because they calibrate residual units shared by all materials
using the same offline cover edge.

## 3. End-to-end forward model

### 3.1 Periodic graph construction

For every crystal, construct a deterministic periodic neighbor graph with the same cutoff, image
convention, atom ordering, and graph-cache identity used by the pinned GMTNet implementation. The
initial atom and edge embeddings are

\[
h_i^{(0)}=\Phi_{\mathrm{atom}}(Z_i),
\qquad
u_{ij}=\Phi_{\mathrm{radial}}(d_{ij}).
\]

If frozen DPA features are used, only the documented input projection changes; the downstream
carrier width and graph order remain the same. The choice of atom features is orthogonal to the
hierarchical fusion algorithm.

### 3.2 GMTNet global node encoder

Let $\mathcal M_t$ denote the pinned GMTNet equivariant attention/message-passing layer. Abstractly,

\[
m_{ij}^{(t)}
=
\mathcal M_t
\left(h_i^{(t)},h_j^{(t)},u_{ij},\widehat{\mathbf d}_{ij}\right),
\]

\[
h_i^{(t+1)}
=
\mathcal U_t
\left(h_i^{(t)},
\sum_{j:(j,i)\in E_b}m_{ji}^{(t)}\right),
\qquad t=0,\ldots,T-1.
\]

Here $\widehat{\mathbf d}_{ij}=\mathbf d_{ij}/d_{ij}$, and angular dependence is represented in the
official model through equivariant spherical/tensor operations. The exact internal attention
parameterization remains the pinned official GMTNet implementation; the required public contract is

\[
h_i(gx)=\rho(g)h_i(x).
\]

The global crystal feature is mean pooled:

\[
g_b
=
\operatorname{Pool}_b(\{h_i\})
=
\frac{1}{N_b}\sum_{i\in V_b}h_i.
\]

The auxiliary branch starts from the same final node features $h_i$. The official feature mask and
response readout are deliberately postponed until after global-plus-expert fusion.

### 3.3 Unique PG expert evaluation and pooling

For each unique active PG $H\in\mathcal A_b$, let

\[
\mathcal E_H:
\{(h_i,\mathcal G_b)\}_{i\in V_b}
\longrightarrow
\{e_{i,H}\in\mathcal V\}_{i\in V_b}
\]

be a lightweight PG-conditioned equivariant residual expert. It satisfies

\[
\mathcal E_H(\rho(g)h,g\mathcal G_b)
=
\rho(g)\mathcal E_H(h,\mathcal G_b).
\]

Its crystal-level feature is independently pooled:

\[
z_{b,H}
=
\frac{1}{N_b}
\sum_{i\in V_b}e_{i,H}.
\]

The expert output is interpreted as a residual carrier feature, not as a separate physical
prediction. Experts from different crystals are never pooled together. Batching by PG is an
implementation optimization only; segment indices restore the $b,H$ ownership before pooling.

## 4. Geometry-derived cover-edge residual

### 4.1 Parent-minus-child operations

For cover edge $e=(H^+\to H^-)$, the validated embedding asset supplies parent rotations
$\mathcal R^+_e$ and allowed oriented variants
${\mathcal R^-_{e,v}}_{v\in\mathcal V_e}$ of the child rotations in the same frame. For variant
$v$, the rotations added by the parent hypothesis are

\[
\Delta\mathcal R_{e,v}
=
\mathcal R^+_e\setminus\mathcal R^-_{e,v}.
\]

Every valid cover edge must add at least one rotation.

### 4.2 Species-pair-preserving edge matching

Let $\mathcal I_{b,c}$ be graph edges whose ordered endpoint species pair equals $c=(Z_i,Z_j)$.
For an added rotation $R\in\Delta\mathcal R_{e,v}$, rotate every relative edge vector and form
the squared cost matrix inside the same species-pair block:

\[
C^{(b,c,R)}_{pq}
=
\left\|R\mathbf d_p-\mathbf d_q\right\|_2^2,
\qquad p,q\in\mathcal I_{b,c}.
\]

Let $\Pi(\mathcal I_{b,c})$ denote permutations of that block. Hungarian assignment yields

\[
D_{b,c,R}
=
\min_{\varphi\in\Pi(\mathcal I_{b,c})}
\sum_{p\in\mathcal I_{b,c}}
C^{(b,c,R)}_{p,\varphi(p)}.
\]

The residual of orientation variant $v$ is the RMS over all matched entries for all added
rotations and species blocks:

\[
r_{b,e,v}
=
\sqrt{
\frac{
\displaystyle
\sum_{R\in\Delta\mathcal R_{e,v}}
\sum_c D_{b,c,R}
}{
\displaystyle
|\Delta\mathcal R_{e,v}|\sum_c|\mathcal I_{b,c}|
}
}.
\]

The edge residual chooses the best allowed child orientation:

\[
r_{b,e}=\min_{v\in\mathcal V_e}r_{b,e,v}.
\]

This reproduces the current `minimum-oriented-parent-minus-child-relative-edge-rms-v1` convention.
If the observed edge set is empty, its empty RMS is defined as zero, matching the cache
implementation. In the initial system these residuals are precomputed and detached; gradients flow
to learned $\sigma_e$, experts, GMTNet, and the final branch logit, but not through Hungarian
assignment or strict PG detection.

## 5. Within-chain stick breaking

### 5.1 Edge gate

For every edge on an active chain, define

\[
q_{b,k,d}
=
\frac{r_{b,e_{b,k,d}}}{\sigma_{e_{b,k,d}}},
\qquad
a_{b,k,d}
=
1-\exp\!\left(-q_{b,k,d}^2\right).
\]

Because $r\ge0$ and $\sigma>0$,

\[
0\le a_{b,k,d}<1.
\]

The orientation of this gate is important:

- $r=0\Rightarrow a=0$: the parent symmetry is exactly supported, so mass stops at that parent.
- $r\gg\sigma\Rightarrow a\to1$: the parent operation is violated, so mass passes through the
  ancestor tests and ultimately remains closer to the detected current group.

### 5.2 Recursive construction

Initialize the unallocated mass of chain $k$ by

\[
R_{b,k,0}=1.
\]

For depths $d=1,\ldots,D_{b,k}$, assign mass to ancestor $H_{b,k,d}$ and update the remaining
mass:

\[
\omega_{b,k,d}
=
R_{b,k,d-1}(1-a_{b,k,d}),
\qquad
R_{b,k,d}
=
R_{b,k,d-1}a_{b,k,d}.
\]

After the last edge, assign the remaining mass to the current group:

\[
\omega_{b,k,0}=R_{b,k,D_{b,k}}.
\]

The closed forms are therefore

\[
\omega_{b,k,d}
=
\left(\prod_{j=1}^{d-1}a_{b,k,j}\right)
(1-a_{b,k,d}),
\qquad 1\le d\le D_{b,k},
\]

and

\[
\omega_{b,k,0}
=
\prod_{j=1}^{D_{b,k}}a_{b,k,j}.
\]

The empty product equals one. For a singleton chain with $D_{b,k}=0$, define
$\omega_{b,k,0}=1$.

### 5.3 Normalization proof

Since

\[
\omega_{b,k,d}
=
R_{b,k,d-1}-R_{b,k,d},
\]

the ancestor terms telescope:

\[
\sum_{d=1}^{D_{b,k}}\omega_{b,k,d}
=
R_{b,k,0}-R_{b,k,D_{b,k}}
=
1-R_{b,k,D_{b,k}}.
\]

Adding the current-group remainder gives

\[
\sum_{d=0}^{D_{b,k}}\omega_{b,k,d}=1.
\]

Every $\omega_{b,k,d}$ is non-negative, so each chain feature is a convex combination:

\[
c_{b,k}
=
\sum_{d=0}^{D_{b,k}}
\omega_{b,k,d}z_{b,H_{b,k,d}}.
\]

## 6. Between-chain residual weighting

### 6.1 Immediate-parent energy

The compatibility of chain $k$ is determined only by its first cover edge adjacent to $G_b$:

\[
E_{b,k}
=
q_{b,k,1}^2
=
\left(
\frac{r_{b,e_{b,k,1}}}{\sigma_{e_{b,k,1}}}
\right)^2.
\]

Edges above the immediate parent are intentionally excluded from this energy. Such edges answer how
far upward to move *after* entering a chain and may remain inconsistent even when its immediate
parent is correct. They already affect $c_{b,k}$ through within-chain stick breaking.

For the degenerate case of one singleton chain, set $\pi_{b,1}=1$ without constructing an energy.

### 6.2 Negative-energy softmax

For $K_b>1$, define

\[
\pi_{b,k}
=
\frac{\exp(-E_{b,k}/\tau)}
{\sum_{j=1}^{K_b}\exp(-E_{b,j}/\tau)},
\qquad \tau>0.
\]

Immediately,

\[
\pi_{b,k}>0,
\qquad
\sum_{k=1}^{K_b}\pi_{b,k}=1.
\]

For two chains $i,j$, their odds are

\[
\frac{\pi_{b,i}}{\pi_{b,j}}
=
\exp\!\left(-\frac{E_{b,i}-E_{b,j}}{\tau}\right).
\]

Hence $E_{b,i}<E_{b,j}\Rightarrow\pi_{b,i}>\pi_{b,j}$. Equal immediate-edge energies produce
equal weights. As $\tau\to0^+$, routing approaches the minimum-energy chain; as
$\tau\to\infty$, it approaches the uniform distribution. Unlike reciprocal residual weighting,
the softmax is finite at exact symmetry and requires no arbitrary additive $\varepsilon$.

The conceptual chain-level auxiliary feature is

\[
z_b^{\mathrm{PG}}
=
\sum_{k=1}^{K_b}\pi_{b,k}c_{b,k}.
\]

## 7. Exact collapse to unique PG experts

Explicitly materializing every $c_{b,k}$ would repeatedly reference a PG shared by several chains.
Substituting the chain definition gives

\[
z_b^{\mathrm{PG}}
=
\sum_{k=1}^{K_b}\pi_{b,k}
\sum_{d=0}^{D_{b,k}}
\omega_{b,k,d}z_{b,H_{b,k,d}}.
\]

Because all sums are finite and fusion is linear, regroup terms by unique PG. Define

\[
\alpha_{b,H}
=
\sum_{k=1}^{K_b}
\sum_{d=0}^{D_{b,k}}
\mathbf 1[H_{b,k,d}=H]
\pi_{b,k}\omega_{b,k,d}.
\]

Then

\[
\boxed{
z_b^{\mathrm{PG}}
=
\sum_{H\in\mathcal A_b}
\alpha_{b,H}z_{b,H}
}
\]

is exactly equal to explicit chain-feature fusion. It is not an approximation and produces identical
gradients. Since $\pi\ge0$ and $\omega\ge0$, $\alpha_{b,H}\ge0$. Its normalization is

\[
\begin{aligned}
\sum_{H\in\mathcal A_b}\alpha_{b,H}
&=
\sum_{k=1}^{K_b}\pi_{b,k}
\sum_{d=0}^{D_{b,k}}\omega_{b,k,d}\\
&=
\sum_{k=1}^{K_b}\pi_{b,k}\\
&=1.
\end{aligned}
\]

Thus the optimized auxiliary feature remains a convex combination of unique pooled expert features.
The implementation should retain $\pi_{b,k}$, $\omega_{b,k,d}$, and $\alpha_{b,H}$ as diagnostics,
while evaluating every $z_{b,H}$ once.

## 8. Global-plus-expert residual merge

Use one learned scalar for the complete PG auxiliary branch:

\[
\lambda
=
\operatorname{sigmoid}(\ell)
\in(0,1).
\]

Initialize $\ell$ to a negative value so training begins near the pinned GMTNet path without
eliminating the gradient. The fused crystal carrier is

\[
\boxed{
\widetilde g_b
=
g_b+\lambda z_b^{\mathrm{PG}}
}
\]

There are no learned per-PG fusion logits and no learned per-chain fusion logits. PG-specific
parameters live inside the experts; material-dependent mixture weights come only from geometric
residuals and learned positive edge scales.

The formal recovery limit is

\[
\lim_{\ell\to-\infty}\widetilde g_b=g_b.
\]

For exact finite-parameter baseline tests, the wrapper may expose an explicit auxiliary-disable flag;
this is a testing/configuration switch rather than an additional learned gate.

## 9. GMTNet mask, tensor readout, and loss

After fusion, preserve the official terminal order. Let $M_b$ be GMTNet's feature/symmetry mask,
$\mathcal R_\theta$ its response block, and $\mathcal Q_b$ its final equality/symmetry adjustment:

\[
\bar g_b=M_b\widetilde g_b,
\]

\[
o_b=\mathcal R_\theta(\bar g_b),
\]

\[
\widehat y_b=\mathcal Q_b(o_b).
\]

For a rank-two response, $o_b$ is reshaped into the task's matrix representation before the final
adjustment. The precise dielectric or elastic equality mask remains the pinned task implementation.
No expert-specific readout is introduced.

With entrywise mask $W_b$ for observed/valid targets and a scalar robust loss $\phi$, the training
objective is

\[
\mathcal L_{\mathrm{task}}
=
\frac{
\displaystyle\sum_{b=1}^{B}
\sum_{p,q}W_{b,pq}\,
\phi(\widehat y_{b,pq}-y_{b,pq})
}{
\displaystyle\sum_{b=1}^{B}\sum_{p,q}W_{b,pq}
}.
\]

For the current GMTNet-aligned protocol, $\phi$ is the configured Huber loss. Optional weight decay
is applied by the optimizer rather than by changing the routing equations:

\[
\mathcal L
=
\mathcal L_{\mathrm{task}}
+\beta\|\Theta_{\mathrm{decay}}\|_2^2.
\]

The complete trainable set is

\[
\Theta
=
\Theta_{\mathrm{GMTNet}}
\cup
\{\Theta_H:H\in\mathcal A_{\mathrm{registry}}\}
\cup
\{s_e:e\in\mathcal E_{\mathrm{registry}}\}
\cup
\{\ell\},
\]

plus any required equivariant carrier adapters. Whether the GMTNet backbone is initialized from a
checkpoint or trained from scratch is an experiment configuration, not a change to the algorithm.

## 10. Equivariance derivation

Assume:

1. GMTNet node features obey $h_i(gx)=\rho(g)h_i(x)$.
2. Every expert and any carrier adapter are $O(3)$-equivariant.
3. Mean pooling uses only invariant segment membership.
4. Residuals are invariant: $r_{b,e}(gx)=r_{b,e}(x)$.
5. The active DAG and chain enumeration are frame independent.

Mean pooling commutes with the group action:

\[
g_b(gx)
=
\frac1{N_b}\sum_i\rho(g)h_i(x)
=
\rho(g)g_b(x).
\]

Likewise,

\[
z_{b,H}(gx)=\rho(g)z_{b,H}(x).
\]

Residual invariance implies invariance of every scalar derived from it:

\[
q_{b,e}(gx)=q_{b,e}(x),
\quad
a_{b,e}(gx)=a_{b,e}(x),
\quad
\omega_{b,k,d}(gx)=\omega_{b,k,d}(x),
\quad
\pi_{b,k}(gx)=\pi_{b,k}(x),
\quad
\alpha_{b,H}(gx)=\alpha_{b,H}(x).
\]

Therefore

\[
\begin{aligned}
z_b^{\mathrm{PG}}(gx)
&=
\sum_H\alpha_{b,H}(x)\rho(g)z_{b,H}(x)\\
&=
\rho(g)z_b^{\mathrm{PG}}(x),
\end{aligned}
\]

and, because $\lambda$ is a scalar,

\[
\widetilde g_b(gx)
=
\rho(g)g_b(x)+\lambda\rho(g)z_b^{\mathrm{PG}}(x)
=
\rho(g)\widetilde g_b(x).
\]

Consequently the unchanged equivariant GMTNet readout produces the required tensor transformation
law. For a Cartesian rank-two response this is

\[
\widehat Y(gx)=g\widehat Y(x)g^{\mathsf T}.
\]

This construction does not add a new symmetry beyond $O(3)$. It changes finite-capacity parameter
sharing across input stabilizer strata while preserving the same transformation law.

## 11. Continuity and differentiability boundary

For a fixed active DAG, fixed chain support, fixed matching/orientation branch, positive
$\sigma_e$, and positive $\tau$, the maps

\[
r
\mapsto q
\mapsto a
\mapsto(\omega,\pi)
\mapsto\alpha
\mapsto z^{\mathrm{PG}}
\]

are continuous. Softmax and the analytic gate provide finite derivatives at $r=0$. Minimum
orientation selection and optimal assignment can be nondifferentiable when minimizers exchange, but
their minimum value remains continuous when the underlying costs are continuous.

The initial cached implementation does **not** backpropagate through $r$, because strict PG
detection, Hungarian assignment, and routing-cache construction are outside the model graph. It does
backpropagate through every $\sigma_e$, the experts, GMTNet, and $\ell$.

No global continuity claim is made across a change of strict current-PG label if that change inserts
or removes DAG nodes or chains. Residual-dependent chain weights remove the fixed positive mass of an
incompatible chain, but a formal global guarantee additionally requires a fixed candidate support,
such as a unified PG DAG whose nodes remain present while only continuous weights change.

## 12. Numerical and validation rules

The implementation must enforce the following conditions:

1. Use `softplus` plus a small positive floor for every $\sigma_e$.
2. Compute chain softmax with log-sum-exp stabilization.
3. Require residual keys to cover exactly the validated material DAG edges.
4. Require every active chain to be a complete offline maximal path.
5. Require $\omega$, $\pi$, and $\alpha$ to be finite and non-negative.
6. Check their respective sums against one within a documented numerical tolerance.
7. Deduplicate experts by PG number before execution and preserve crystal segment ownership.
8. Require every pooled expert feature to use the exact GMTNet public carrier layout.
9. Apply the auxiliary merge before the unchanged GMTNet feature mask and output block.
10. Record the official GMTNet commit, graph/cache hashes, PG registry and DAG hashes, residual
    convention, chain-weighting convention, $\tau$, edge-scale parameterization, branch-scale
    parameterization, and carrier layout in checkpoints and result summaries.
11. Fail closed on stale cache identity, missing edges, inconsistent sample/node order, or layout
    mismatch.

## 13. Batched algorithm

```text
Input:
    crystals x_1, ..., x_B
    validated graph cache and material PG routing cache
    GMTNet parameters, PG expert parameters
    learned edge-scale parameters {s_e}, learned branch logit ell
    fixed temperature tau

1. Build/load the periodic batch graph without changing crystal or node order.
2. Run pinned GMTNet through its final equivariant node update:
       h_i = GMTNetNodeEncoder(x)_i
3. Mean-pool h_i per crystal:
       g_b = mean_{i in V_b} h_i
4. For every crystal b:
       enumerate all maximal chains P_{b,k}
       collect the unique active PG set A_b
5. Bucket requests by unique PG, execute each requested expert once per crystal,
   and mean-pool its node outputs:
       z_{b,H} = mean_{i in V_b} E_H(h, graph)_i
6. For every crystal b and active cover edge e:
       sigma_e = softplus(s_e) + epsilon_sigma
       q_{b,e} = r_{b,e} / sigma_e
       a_{b,e} = 1 - exp(-q_{b,e}^2)
7. For every chain k:
       compute within-chain omega_{b,k,d} by stick breaking
       E_{b,k} = q_{b,e_{b,k,1}}^2
8. Compute chain weights:
       pi_{b,:} = softmax(-E_{b,:} / tau)
9. Collapse repeated PG contributions:
       alpha_{b,H} = sum_{k,d: H_{b,k,d}=H} pi_{b,k} omega_{b,k,d}
       z_b^PG = sum_H alpha_{b,H} z_{b,H}
10. Merge branches:
       lambda = sigmoid(ell)
       g_tilde_b = g_b + lambda z_b^PG
11. Preserve official GMTNet terminal processing:
       y_hat_b = EqualityAdjustment(OutputBlock(FeatureMask(g_tilde_b)))
12. During training, evaluate masked task loss, backpropagate, and update all
    configured trainable parameters.

Output:
    predictions y_hat_b
    optional diagnostics {omega_{b,k,d}, E_{b,k}, pi_{b,k}, alpha_{b,H}, lambda}
```

## 14. Required ablations and interpretation

The algorithm must be evaluated against at least:

- pinned GMTNet alone;
- global plus a parameter/FLOP-matched dense branch;
- global plus a generic learned MoE;
- shuffled or random PG routing;
- flat PG fusion without chain structure;
- the old fixed chain-length prior;
- hierarchical routing without the global residual path;
- the proposed global plus hierarchical PG experts.

The scientific claim is supported only if the hierarchy improves finite-data or finite-capacity
behavior beyond parameter count alone, while retaining the same tensor equivariance and symmetry
consistency. The model does not claim that PG experts provide information absent from the structure,
nor that they are required for exact point-group-valid outputs from an already exact
$O(3)$-equivariant model.

## 15. Relation to the focused weighting analysis

The rationale for replacing the fixed path-length prior, the choice of immediate-parent rather than
full-chain energy, and the remaining fixed-support continuity limitation are developed separately in
[`point_group_subchain_weighting.md`](point_group_subchain_weighting.md). The present document turns
that local routing proposal into the complete global-plus-experts model algorithm.

## 16. GMTNet algorithm

This appendix expands the pinned GMTNet branch used above. The checked-in dielectric model uses
scalar width \(C_s=128\), two scalar attention layers, and an equivariant update with spherical
harmonics through \(\ell=2\).

### 16.1 Atom, edge, and scalar-attention stages

From the 92-dimensional atom descriptor \(x_i^{\mathrm{atom}}\), GMTNet initializes

\[
s_i^{(0)}=W_{\mathrm{atom}}x_i^{\mathrm{atom}}+b_{\mathrm{atom}}
\in\mathbb R^{128}.
\]

For periodic edge vector \(\mathbf d_{ij}\), define

\[
\xi_{ij}=-\frac{0.75}{\|\mathbf d_{ij}\|_2},
\qquad
u_{ij}
=
\operatorname{Softplus}
\left[W_r\operatorname{RBF}_{512}(\xi_{ij})+b_r\right],
\]

where the fixed RBF centers cover \([-4,0]\). In scalar attention layer \(t\),

\[
q_i=W_qs_i^{(t)},\quad
k_i=W_ks_i^{(t)},\quad
v_i=W_vs_i^{(t)},\quad
\bar u_{ij}=W_eu_{ij}.
\]

The edge-conditioned key and value message are

\[
\widetilde k_{ij}=\operatorname{MLP}_k([k_i,k_j,\bar u_{ij}]),
\qquad
\widetilde m_{ij}=\operatorname{MLP}_m([v_i,v_j,\bar u_{ij}]).
\]

The pinned implementation uses an elementwise sigmoid gate rather than a neighbor-softmax:

\[
\gamma_{ij}
=
\operatorname{sigmoid}
\left[
\operatorname{BN}
\left(
\frac{q_i\odot\widetilde k_{ij}}{\sqrt{C_s}}
\right)
\right],
\qquad
m_{ij}=\widetilde m_{ij}\odot\gamma_{ij}.
\]

It sums messages and applies a residual Softplus update:

\[
m_i=\sum_{j:(j,i)\in E_b}m_{ji},
\qquad
s_i^{(t+1)}
=
\operatorname{Softplus}(s_i^{(t)}+W_om_i+b_o).
\]

This layer is applied twice.

### 16.2 Equivariant tensor-product stage

Project \(s_i^{(2)}\) into \(\mathcal V^{(0)}=16\times0e\), and compute

\[
Y_{ij}=Y_{\le2}(\widehat{\mathbf d}_{ij})
\in0e\oplus1o\oplus2e.
\]

The three tensor-product layers use

\[
\begin{aligned}
\mathcal V^{(0)}
&=16\times0e,\\
\mathcal V^{(1)}
&=16\times0e\oplus2\times1o\oplus2\times2e,\\
\mathcal V^{(2)}
&=16\times0e\oplus2\times1o\oplus2\times1e
  \oplus2\times2e\oplus2\times2o,\\
\mathcal V^{(3)}
&=0e\oplus0o\oplus1e\oplus1o\oplus2e\oplus2o\oplus3e\oplus3o.
\end{aligned}
\]

For layer \(r\), the radial network generates tensor-product weights:

\[
w_{ij}^{(r)}=\operatorname{MLP}_{\mathrm{TP}}^{(r)}(u_{ij}),
\qquad
\mu_{ij}^{(r)}
=
\operatorname{TP}^{(r)}
\left(t_j^{(r)},Y_{ij};w_{ij}^{(r)}\right).
\]

Mean aggregation, with the configured padded residual where enabled, gives

\[
t_i^{(r+1)}
=
\frac1{|\mathcal N(i)|}
\sum_{j\in\mathcal N(i)}\mu_{ij}^{(r)}
+\operatorname{Pad}(t_i^{(r)}).
\]

The cached directed-edge orientation must match the official source/target convention. The final
node feature exposed to both branches is

\[
h_i=t_i^{(3)}\in\mathcal V^{(3)}.
\]

### 16.3 Pooling and response readout

GMTNet mean-pools \(g_b=N_b^{-1}\sum_i h_i\). The combined model inserts
\(\widetilde g_b=g_b+\lambda z_b^{\mathrm{PG}}\), then retains the official order

\[
\bar g_b=M_b\widetilde g_b,
\qquad
\widehat y_b=\mathcal Q_b(\mathcal R_\theta(\bar g_b)).
\]

For dielectric prediction, \(\mathcal R_\theta\) introduces a differentiable polar probe
\(\mathbf E\in1o\), constructs

\[
\mathbf D_b(\mathbf E)
=
\operatorname{TP}_{\mathrm{dielectric}}
\left(\bar g_b,Y_{1o}(\mathbf E);\mathbf1\right),
\]

and returns its Jacobian

\[
\widehat\varepsilon_{b,ac}
=
\frac{\partial D_{b,a}}{\partial E_c}.
\]

The equality adjustment \(\mathcal Q_b\) averages tensor entries declared equivalent by the supplied
crystallographic mask. Piezoelectric and elastic tasks use analogous automatic-differentiation
constructions with symmetric stress or strain probes.

### 16.4 GMTNet forward procedure

1. Embed 92 atom features into 128 scalar channels.
2. Expand \(-0.75/d_{ij}\) into 512 RBF channels and project to 128 edge channels.
3. Apply two scalar Comformer attention layers.
4. Project to \(16\times0e\), compute \(0e\oplus1o\oplus2e\) edge harmonics, and apply three
   equivariant tensor-product convolutions.
5. Mean-pool the resulting \(\mathcal V^{(3)}\) node carrier.
6. Insert hierarchical expert fusion.
7. Apply the unchanged feature mask, response block, and equality adjustment.

## 17. PG-expert algorithm: Adapter plus Full-PG experts

This section specifies the complete PG-expert branch intended by this model. It is not the generic
RoutedO3Expert. For crystal \(b\), the branch is

\[
\{h_i\}_{i\in V_b}
\xrightarrow{\text{\(O(3)\) Adapter}}
\{a_i\}_{i\in V_b}
\xrightarrow{\text{Full-PG expert }H}
\{e_{i,H}\}_{i\in V_b}
\xrightarrow{\text{crystal pooling}}
z_{b,H}.
\]

The hierarchical fuser in Sections 5--8 then combines the \(z_{b,H}\). The Adapter is shared by all
PGs; each \(H\) has its own two-block Full-PG expert.

### 17.1 Carrier interfaces

Let \(\mathcal V_G\) be GMTNet's node carrier and \(\mathcal V_E\) the public PG-expert carrier. In
the existing expert family,

\[
\mathcal V_E
=
8\times0e
\oplus2\times1o
\oplus2\times2e
\oplus2\times3o
\oplus2\times4e,
\]

whose flattened dimension is 56. The implementation must not assume that
\(\mathcal V_G=\mathcal V_E\). It therefore exposes equivariant interface maps

\[
L_{\mathrm{in}}:\mathcal V_G\to\mathcal V_E,
\qquad
L_{\mathrm{out}}:\mathcal V_E\to\mathcal V_G.
\]

They are block-diagonal over matching \(O(3)\) irreps. An output irrep absent from the input cannot be
created by a linear map alone; it is initialized as zero and may subsequently be generated by the
Adapter's tensor products with edge spherical harmonics. Define

\[
u_i=L_{\mathrm{in}}h_i.
\]

If a concrete GMTNet configuration already emits \(\mathcal V_E\), \(L_{\mathrm{in}}\) may be the
identity. Checkpoints must record both layouts and both interface-map identities.

### 17.2 Shared O(3) Adapter

The Adapter is one shared periodic \(O(3)\)-equivariant message block. It adapts GMTNet features to
the distribution and receptive field expected by every PG expert without introducing PG-specific
parameters.

For directed edge \(j\to i\), compute the edge carrier

\[
y_{ij}
=
Y_{\le2}(\widehat{\mathbf d}_{ij})
\in
0e\oplus1o\oplus2e.
\]

Let \(\operatorname{TP}_A\) be an equivariant tensor product

\[
\operatorname{TP}_A:
\mathcal V_E\otimes
(0e\oplus1o\oplus2e)
\longrightarrow
\mathcal V_E.
\]

The unmodulated message is

\[
m_{ij}^{A}
=
\operatorname{TP}_A
\left(u_j,y_{ij};\mathbf d_{ij}\right).
\]

The current backend accepts the relative vector as additional geometric input when constructing its
tensor-product coefficients. With cutoff \(r_c\), define normalized distance and cosine envelope

\[
\delta_{ij}=\frac{d_{ij}}{r_c},
\]

\[
\chi(\delta)
=
\begin{cases}
\frac12[\cos(\pi\delta)+1],&0\le\delta\le1,\\
0,&\delta>1.
\end{cases}
\]

A scalar radial MLP produces

\[
\kappa_{ij}^{A}
=
\psi_A(\delta_{ij})\chi(\delta_{ij}),
\qquad
\widetilde m_{ij}^{A}
=
\kappa_{ij}^{A}m_{ij}^{A}.
\]

The aggregated message is normalized by the square root of target degree:

\[
\bar m_i^{A}
=
\frac{
\sum_{j\in\mathcal N(i)}
\widetilde m_{ij}^{A}
}{
\sqrt{\max(1,|\mathcal N(i)|)}
}.
\]

The Adapter uses no learned self-connection. Its self path is the exact identity. Introduce one
bounded scalar residual gate

\[
\gamma_A=\tanh(\zeta_A),
\qquad
\zeta_A\big|_{\mathrm{init}}=0,
\]

and define the Adapter output by

\[
\boxed{
a_i
=
u_i+\gamma_A\bar m_i^{A}
}.
\]

Therefore the Adapter is exactly the identity at initialization:

\[
a_i\big|_{\mathrm{init}}=u_i.
\]

For a graph with no edges, \(\bar m_i^{A}=0\), so it remains the identity for every value of
\(\gamma_A\). The scalar gate cannot mix irrep types or magnetic components. Since the message is
equivariant and \(\gamma_A\) is invariant, the Adapter satisfies

\[
a_i(gx)=\rho_E(g)a_i(x).
\]

The identity residual deliberately replaces both the former learned self-connection
\(L_A^{\mathrm{self}}u_i\) and the former input scaling \(\eta_Au_i\). This removes their redundant
copy mixing before Full-PG subduction and gives the auxiliary branch an auditable zero-message
starting point.

The same \(a_i\) is reused by every active Full-PG expert.

### 17.3 Standardizing the frame before and after the PG expert

Full-PG subduction matrices are tabulated in a standard crystallographic Cartesian frame. Therefore
the adapted feature must be rotated into that frame before \(U_H\) is applied, and the expert output
must be rotated back before pooling and fusion with GMTNet.

For crystal \(b\), let

\[
R_b=R_{b,\mathrm{input}\to\mathrm{std}}\in SO(3),
\qquad
R_b^{-1}=R_{b,\mathrm{std}\to\mathrm{input}}.
\]

The corresponding representation on the expert carrier is \(D_E(R_b)=\rho_E(R_b)\). The
implementation stores carrier coefficients as row vectors, so the feature entering Full-PG is

\[
\boxed{
a_{i,b}^{\mathrm{std}}
=
a_{i,b}^{\mathrm{input}}D_E(R_b)^{\mathsf T}
}.
\]

The coordinate convention is

\[
\mathbf r_i^{\mathrm{std}}=R_b\mathbf r_i^{\mathrm{input}},
\]

or, for stored row vectors,

\[
(\mathbf r_i^{\mathrm{std}})^{\mathsf T}
=
(\mathbf r_i^{\mathrm{input}})^{\mathsf T}R_b^{\mathsf T}.
\]

If Full-PG produces \(e_{i,H}^{\mathrm{std}}\), rotate it back with the inverse frame:

\[
\boxed{
e_{i,H}^{\mathrm{input}}
=
e_{i,H}^{\mathrm{std}}
D_E(R_b^{-1})^{\mathsf T}
}.
\]

Because \(D_E(R_b^{-1})=D_E(R_b)^{-1}\), these rotations are inverses:

\[
\left[aD_E(R_b)^{\mathsf T}\right]
D_E(R_b^{-1})^{\mathsf T}
=a.
\]

They are fixed geometric transforms, not trainable routing weights.

For an external proper rotation \(Q\), deterministic standardization must satisfy

\[
R_b(Qx)Q=R_b(x).
\]

In column-vector notation,

\[
\rho_E(R_b(Qx))\rho_E(Q)
=
\rho_E(R_b(x)),
\]

so the standardized expert input is unchanged by a global frame rotation. Restoring the inverse
frame consequently gives

\[
e^{\mathrm{input}}(Qx)
=
\rho_E(Q)e^{\mathrm{input}}(x).
\]

The current canonicalization contract forces the standardization matrix to be a proper rotation.
Therefore this pre/post rotation directly establishes frame covariance for \(SO(3)\). A claim for all
of \(O(3)\), including improper transforms, additionally requires either:

1. a parity-aware standard-frame rule satisfying \(R_b(gx)g=R_b(x)\) for every \(g\in O(3)\); or
2. an explicit improper-frame parity transform on every \((\ell,p)\) carrier block.

The implementation and tests must choose one before claiming full \(O(3)\) equivariance. A
proper-rotation round-trip test alone is insufficient.

### 17.4 Restricting the \(O(3)\) carrier to point group \(H\)

Let \(H\subset O(3)\) be one active finite point group and
\(\rho_E|_H\) the restriction of the public carrier to \(H\). The offline registry constructs a real
orthogonal subduction matrix \(U_H\) such that

\[
D_H(q)
=
U_H^{\mathsf T}\rho_E(q)U_H,
\qquad q\in H,
\]

is block-organized into copies of irreducible \(H\)-representations. Using the row-vector convention
of the implementation, transform an adapted node feature into PG coordinates:

\[
v_{i,H}^{(0)}=a_i^{\mathrm{std}}U_H.
\]

The subduction transformation is a change of basis, not a projection: Full-PG keeps both trivial and
nontrivial finite-group carriers. This distinguishes it from the A1-only expert.

Let the copy metadata partition the PG-coordinate vector into slices

\[
v_{i,H}
=
v_{i,H}^{[1]}\oplus\cdots\oplus v_{i,H}^{[C_H]}.
\]

Every slice contains one real irrep copy. The metadata records its indices and whether it is a
one-dimensional trivial copy.

### 17.5 One Full-PG equivariant block

Each Full-PG expert contains two independently parameterized finite-group blocks. For block
\(r\in\{1,2\}\), start from an unconstrained matrix
\(W_{H,r}\) and average it over the finite group:

\[
\overline W_{H,r}
=
\frac1{|H|}
\sum_{q\in H}
D_H(q)W_{H,r}D_H(q)^{\mathsf T}.
\]

For any \(p\in H\), reindexing \(q\mapsto pq\) gives

\[
\begin{aligned}
D_H(p)\overline W_{H,r}D_H(p)^{\mathsf T}
&=
\frac1{|H|}
\sum_{q\in H}
D_H(pq)W_{H,r}D_H(pq)^{\mathsf T}\\
&=
\overline W_{H,r}.
\end{aligned}
\]

Therefore \(\overline W_{H,r}\) commutes with the \(H\)-action and is an
\(H\)-equivariant linear map.

The invariant projector is

\[
P_H
=
\frac1{|H|}
\sum_{q\in H}D_H(q).
\]

It obeys \(D_H(p)P_H=P_H\), so the admissible bias is

\[
\overline b_{H,r}=b_{H,r}P_H.
\]

For block input \(v_{i,H}^{(r-1)}\), form

\[
\widetilde v_{i,H}^{(r)}
=
v_{i,H}^{(r-1)}\overline W_{H,r}
+
\overline b_{H,r}.
\]

A componentwise nonlinearity is not valid on a general multidimensional irrep. Nonlinearity is
therefore applied copy by copy.

For a trivial one-dimensional copy \(c\),

\[
\Gamma_{H,r}^{[c]}(w)
=
\operatorname{SiLU}(w).
\]

For a nontrivial copy \(c\), its norm is invariant under the orthogonal irrep action. Use the scalar
norm gate

\[
g_{H,r}^{[c]}
=
\operatorname{sigmoid}
\left(
\gamma_{H,r}^{[c]}
\|\widetilde v_{i,H}^{(r),[c]}\|_2
+
\beta_{H,r}^{[c]}
\right),
\]

\[
\Gamma_{H,r}^{[c]}
\left(
\widetilde v_{i,H}^{(r),[c]}
\right)
=
g_{H,r}^{[c]}
\widetilde v_{i,H}^{(r),[c]}.
\]

Concatenate all processed copies and add the residual:

\[
\boxed{
v_{i,H}^{(r)}
=
v_{i,H}^{(r-1)}
+
\bigoplus_{c=1}^{C_H}
\Gamma_{H,r}^{[c]}
\left(
\widetilde v_{i,H}^{(r),[c]}
\right)
}.
\]

Because the linear map, bias, copywise scalar gates, and residual all commute with \(D_H(q)\), each
block is \(H\)-equivariant.

### 17.6 Two-block Full-PG expert and inverse subduction

Apply two independent blocks:

\[
v_{i,H}^{(1)}
=
\mathcal F_{H,1}(v_{i,H}^{(0)}),
\qquad
v_{i,H}^{(2)}
=
\mathcal F_{H,2}(v_{i,H}^{(1)}).
\]

Map back into the public expert carrier:

\[
\boxed{
e_{i,H}^{\mathrm{std}}
=
v_{i,H}^{(2)}U_H^{\mathsf T}
}.
\]

Equivalently,

\[
\mathcal E_H(a_i^{\mathrm{std}})
=
\left[
\mathcal F_{H,2}
\left(
\mathcal F_{H,1}(a_i^{\mathrm{std}}U_H)
\right)
\right]
U_H^{\mathsf T}.
\]

For the identity point group \(C_1\), the configured bypass makes both blocks identities, hence

\[
\mathcal E_{C_1}(a_i^{\mathrm{std}})=a_i^{\mathrm{std}}.
\]

This prevents the least constrained group from receiving an arbitrary special transformation merely
because it is routed through the PG branch.

After inverse subduction, apply the standard-to-input carrier rotation from Section 17.3 to obtain
\(e_{i,H}^{\mathrm{input}}\).

### 17.7 Crystal pooling and output carrier

Pool every unique expert independently:

\[
\bar z_{b,H}
=
\frac1{N_b}
\sum_{i\in V_b}
e_{i,H}^{\mathrm{input}}
\in\mathcal V_E.
\]

Project it into GMTNet's fusion carrier:

\[
\boxed{
z_{b,H}
=
L_{\mathrm{out}}\bar z_{b,H}
\in\mathcal V_G
}.
\]

Pooling and \(L_{\mathrm{out}}\) occur before chain fusion. Thus all \(z_{b,H}\), as well as the global
feature \(g_b\), inhabit exactly \(\mathcal V_G\). A shared PG that occurs in several chains is
adapted, evaluated, pooled, and projected once.

The chain fuser defined earlier then computes

\[
c_{b,k}
=
\sum_{d=0}^{D_{b,k}}
\omega_{b,k,d}z_{b,H_{b,k,d}},
\]

\[
z_b^{\mathrm{PG}}
=
\sum_{k=1}^{K_b}\pi_{b,k}c_{b,k}
=
\sum_{H\in\mathcal A_b}\alpha_{b,H}z_{b,H},
\]

followed by

\[
\widetilde g_b
=
g_b+\lambda z_b^{\mathrm{PG}}.
\]

No learned per-PG or per-chain scalar is introduced after pooling. The learned quantities in this
branch are the shared Adapter message parameters and its zero-initialized residual logit
\(\zeta_A\), each Full-PG expert, the carrier interface maps, the edge scales \(\sigma_e\), and the
single final branch coefficient \(\lambda\).

### 17.8 Equivariance scope of Full-PG experts

For a fixed oriented \(H\), the two finite-group blocks are exactly \(H\)-equivariant. To make the
whole PG branch covariant under an arbitrary external \(g\in O(3)\), the oriented group and its
subduction basis must be transported with the input:

\[
H(gx)=gH(x)g^{-1},
\]

\[
U_{H(gx)}
=
\rho_E(g)U_{H(x)}C_g,
\]

where \(C_g\) is an admissible orthogonal change of PG-adapted coordinates. Under this condition,

\[
\mathcal E_{H(gx)}
\left(
\rho_E(g)a(x)
\right)
=
\rho_E(g)
\mathcal E_{H(x)}(a(x)).
\]

Together with equivariant \(L_{\mathrm{in}}\), Adapter, \(L_{\mathrm{out}}\), and invariant routing
weights, this supplies the assumption used in Section 10.

The explicit pre/post standard-frame rotation in Section 17.3 implements this transported-basis
idea: features move into the frame where \(U_H\) is defined and then move back. Its guarantee is only
as strong as the covariance contract of the standard-frame construction, including its handling of
improper transformations.

### 17.9 Complete PG-expert procedure

1. Run the shared equivariant lift \(u_i=L_{\mathrm{in}}h_i\).
2. Run one shared periodic O(3) Adapter message block and form
   \(a_i=u_i+\tanh(\zeta_A)\bar m_i^A\), with \(\zeta_A=0\) at initialization.
3. Enumerate the unique active PG set \(\mathcal A_b\) for every crystal.
4. Bucket crystals by PG so each nonempty Full-PG expert is called once per mini-batch.
5. Rotate adapted features into the standard frame:
   \(a_i^{\mathrm{std}}=a_iD_E(R_b)^{\mathsf T}\).
6. For each \(H\), subduce \(a_i^{\mathrm{std}}\) with \(U_H\).
7. Apply two independent Full-PG blocks, each consisting of group-averaged linear weights, invariant
   bias, copywise SiLU/norm gates, and a residual connection.
8. Map back to the standard public carrier with \(U_H^{\mathsf T}\).
9. Rotate expert outputs back with \(D_E(R_b^{-1})^{\mathsf T}\).
10. Mean-pool per crystal and project with \(L_{\mathrm{out}}\), producing \(z_{b,H}\).
11. Compute within-chain \(\omega\), between-chain \(\pi\), and deduplicated coefficients \(\alpha\).
12. Form \(z_b^{\mathrm{PG}}=\sum_H\alpha_{b,H}z_{b,H}\).
13. Add the single scaled auxiliary feature to GMTNet:
    \(\widetilde g_b=g_b+\lambda z_b^{\mathrm{PG}}\).

This is the complete Adapter plus Full-PG-experts algorithm used by the model plan.

## 18. Incremental implementation interfaces (2026-09-30)

The implementation lives in [`src/models/global_experts`](../../src/models/global_experts/README.md).
It does not replace the legacy GMTNet runner or `PointGroupTensorModel`. The model separates
`IdentityMessageAdapter`, `HierarchicalChainRouter`, and `GlobalExpertsModel`, reusing the existing
two-block `FullPointGroupExpert` and the pinned GMTNet terminal processing.

The shared expert carrier defaults to the 56-component layout in Section 17.1. Configuration
exposes the carrier, Adapter backend/lmax/mmax/cutoff/radial width, edge sigma initialization/floor,
fixed chain temperature, branch logit, C1 bypass, global freeze, auxiliary disable, and optional
GMTNet equivariant attention. Defaults are sigma=0.08, tau=1, zeta_A=0, and branch logit=-4.
Omega, pi, alpha, pooled expert features, and global/fused carriers are available as diagnostics.

For the improper-frame choice required by Section 17.3, the implementation first decomposes the
row lattice as L=S Q using its polar factor Q in O(3), then runs the existing proper spglib
standardization on the frame-independent S. If that standardization supplies R_S, the complete
input-to-standard transform is R=R_S Q. Under an external g, Q'=Q g^T and S'=S, so R' g=R.
The full e3nn representation of R transports every parity block into/out of the expert frame.
This extends the new branch's frame contract without changing legacy canonicalization.

The unchanged official mask/equality operations retain their own frame contract. Tensor
equivariance requires transporting the feature mask and using equality constraints valid in
the new Cartesian frame; reusing a fixed Cartesian equality mask after a generic rotation is
not valid. Tests cover the carrier, readout, and nontrivial transported feature masks.

The independent [training module](../../src/training/global_experts/README.md) and
[`global_experts_train` CLI](../../src/cli/global_experts_train.py) bind routing residuals to the
exact official graph cache, dataset/split IDs, and registry/DAG identities. They provide masked
Huber training, best-validation-MAE selection, strict checkpoint reload, ordered test predictions,
and an isolated [Slurm launcher](../../slurm/train_global_experts.sbatch). The production entry
point currently targets the same reduced dielectric task as the existing GMTNet baseline.

Implementation checks are in [`test_global_experts.py`](../../tests/test_global_experts.py).
The scientific ablation program in Section 14 remains an experiment plan; unit-test success does
not constitute evidence of improved predictive accuracy or global continuity across DAG changes.
