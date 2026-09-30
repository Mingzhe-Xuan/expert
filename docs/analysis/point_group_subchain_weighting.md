# Point-group subchain weighting analysis

## 1. Problem setting

For a structure whose currently detected point group is \(G\), the offline point-group DAG may
contain several maximal subgroup chains starting at \(G\). Write the \(k\)-th chain as

\[
G \leftarrow H_{k,1} \leftarrow H_{k,2} \leftarrow \cdots.
\]

Here \(H_{k,1}\) is the **immediate parent group** of \(G\) on chain \(k\). In the shorter notation
\(H_k\), the subscript indexes a chain and \(H_k\) always means this immediate parent, rather than
the whole chain or a PG expert. The cover edge adjacent to the current group is

\[
e_k=(H_k\to G),
\]

and \(r_{e_k}\ge 0\) measures the geometric violation of the additional operations contributed by
\(H_k\) relative to \(G\). A smaller residual means greater consistency with that parent-group
hypothesis.

## 2. Current behavior

For every edge \(e\), the current analytic gate is

\[
a_e=1-\exp\!\left[-\left(\frac{r_e}{\sigma_e}\right)^2\right],
\]

where \(\sigma_e>0\) is learned. Each maximal chain performs stick breaking internally. Its
conditional node weights sum to one, after which the whole chain receives the fixed length prior

\[
\pi_k^{\mathrm{fixed}}
=
\frac{|P_k|}{\sum_j |P_j|}.
\]

Thus a chain does not contribute unit mass to the final mixture, but it is guaranteed the positive
mass \(\pi_k^{\mathrm{fixed}}\), independently of whether the observed structure is geometrically
consistent with that chain. Edge residuals decide where this mass stops inside the chain; they
cannot suppress an incompatible chain as a whole.

Suppose \(G\) has two immediate parents \(H_1\) and \(H_2\), while a continuous deformation makes
the structure approach \(H_1\). The first edge residual can approach zero while the second remains
large. The \(H_1\) chain transfers its mass toward \(H_1\), but the fixed mass of the incompatible
\(H_2\) chain can remain at \(G\). When strict symmetry detection changes the current label from
\(G\) to \(H_1\), the active DAG and its normalization change. Consequently, the current routing is
continuous inside a fixed-DAG region but is not guaranteed to be continuous across the label
change.

The fixed path prior is therefore an important source of the boundary mismatch. It is not the mere
fact that conditional weights within a chain sum to one; the issue is assigning every candidate
chain a residual-independent, nonzero total mass.

## 3. Residual-dependent chain weights

A natural replacement is to normalize a consistency score over chains. A direct inverse-residual
construction would be

\[
\pi_k
=
\frac{(\varepsilon+E_k)^{-1}}
     {\sum_j(\varepsilon+E_j)^{-1}},
\]

where \(E_k\) is a normalized chain inconsistency. Although this gives larger weight to more
consistent chains, it is sensitive to the arbitrary choice of \(\varepsilon\), becomes sharply
conditioned near zero, and has a singular zero-residual limit if \(\varepsilon\) is omitted.

A smoother parameterization is a negative-energy softmax:

\[
\pi_k
=
\frac{\exp(-E_k/\tau)}
     {\sum_j\exp(-E_j/\tau)},
\qquad \tau>0.
\]

It guarantees \(\pi_k\ge0\), \(\sum_k\pi_k=1\), finite values at exact symmetry, and continuous
gradients. The temperature \(\tau\) controls the transition between diffuse chain fusion and
near-hard chain selection.

## 4. Which residuals should define chain consistency?

Raw residuals should not be added directly because edge scales and chain lengths differ. If a
whole-chain score is required, a dimensionless length-normalized energy is

\[
E_k^{\mathrm{full}}
=
\frac{1}{|P_k|}
\sum_{e\in P_k}
\left(\frac{r_e}{\sigma_e}\right)^2.
\]

However, every maximal chain extends from \(G\) to a root group. If a structure approaches an
intermediate parent \(H_k\), edges above \(H_k\) can correctly retain large residuals. Including
those edges in \(E_k^{\mathrm{full}}\) would wrongly penalize a valid immediate-parent transition.

For selecting among subchains at the first branch, the recommended score therefore uses only the
immediate-parent cover edge:

\[
E_k
=
\left(\frac{r_{H_k\to G}}{\sigma_{H_k\to G}}\right)^2,
\qquad
\pi_k=\operatorname{softmax}_k(-E_k/\tau).
\]

This definition has a direct interpretation: \(\pi_k\) selects which immediate supergroup
hypothesis best explains the observed geometry, while the existing stick-breaking process handles
how far upward to move after entering that chain.

For a more general score assigned to an arbitrary ancestor \(H_{k,d}\), only the prefix from \(G\)
to that ancestor should be used:

\[
E_{k,d}
=
\frac{1}{d}
\sum_{e\in P_{G\to H_{k,d}}}
\left(\frac{r_e}{\sigma_e}\right)^2.
\]

If the same PG is reachable through several chains, its contributions should be summed before the
final normalization so that only one expert instance is evaluated for that PG.

## 5. Continuity implications

With a fixed graph, fixed candidate DAG, and positive \(\sigma_e\), the residuals, analytic gates,
softmax chain weights, stick-breaking products, and final normalization are continuous functions of
the geometry. Optimal assignment or minimum-orientation changes can create nondifferentiable points,
but the minimum value remains continuous.

Residual-dependent chain weights remove the guaranteed mass assigned to incompatible paths and can
make a transition such as \(G\to H_k\) approach the parent expert much more cleanly. They do not,
by themselves, prove global continuity if the current-PG label still changes the active DAG. A
strict guarantee requires a fixed candidate support, for example a unified PG-DAG whose experts are
never inserted or removed by discrete current-PG detection. Geometry would then affect only
continuous residual-derived scores.

## 6. Recommended initial design

1. Let \(H_k\) denote the immediate parent of \(G\) on subchain \(k\).
2. Normalize its edge residual by the learned positive edge scale \(\sigma_{H_k\to G}\).
3. Compute chain weights with a negative-energy softmax rather than a raw reciprocal.
4. Retain conditional stick breaking inside each selected chain.
5. Sum contributions that reach the same PG before the final expert fusion.
6. Treat fixed-support unified-DAG routing as the additional change required for a formal global
   continuity claim.

The proposed chain gate fixes the most immediate modeling issue: chain-level mass becomes evidence
dependent instead of being imposed by path length. It should first be evaluated against the current
fixed-prior implementation using transition sweeps and matched training settings.
