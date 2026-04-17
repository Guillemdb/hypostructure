# C3 critical \(L^3\)-normalization bridge

This note implements the C3 bridge in the classification-completeness program.
Its purpose is to remove "normalization failure" as an ambiguous Type II
survivor mechanism.

The important point is that critical \(L^3\)-normalization must not be
implemented by multiplying the velocity by a time-dependent scalar. That would
not preserve the Navier-Stokes equation. Instead, the compact Type II barrier
only needs a non-collapse condition in the critical topology. The exact
normalization

```{math}
\|V(\tau)\|_{L^3(\mathbb R^3)}=1
```

is therefore replaced by the invariant certificate

```{math}
0<\eta\le \|V(\tau)\|_{L^3(\mathbb R^3)}\le M<\infty.
```

This is the rigorous meaning of \(K_{L^3\mathrm{Norm}}^+\) in the
classification ledger.

## Critical mass certificates

::::{prf:definition} Critical \(L^3\)-mass along a represented Type II orbit
:label: def-critical-l3-mass

Let \((V,P,a,b)\) be a repaired-gauge renormalized Navier-Stokes Type II
candidate on \([\tau_0,\infty)\). Its critical mass function is

```{math}
N_3(\tau):=\|V(\tau)\|_{L^3(\mathbb R^3)}.
```

The orbit has a **positive finite critical-mass certificate** if there exist
constants \(0<\eta\le M<\infty\) such that

```{math}
\eta\le N_3(\tau)\le M
\qquad
\text{for all }\tau\ge\tau_0.
```

The certificate is denoted

```{math}
K_{L^3\mathrm{Mass}}^+(\eta,M).
```

::::

::::{prf:definition} \(L^3\)-normalization bridge certificate
:label: def-l3norm-bridge-certificate

The bridge certificate

```{math}
K_{L^3\mathrm{Norm}}^+
```

means that the represented Type II orbit carries a positive finite
critical-mass certificate:

```{math}
K_{L^3\mathrm{Norm}}^+
\equiv
\exists\,0<\eta\le M<\infty:
K_{L^3\mathrm{Mass}}^+(\eta,M).
```

No amplitude rescaling is performed. The word "normalization" means that the
orbit lies in a fixed nonzero finite annulus of the scale-critical \(L^3\)
topology.

::::

## Degenerate certificates

If \(K_{L^3\mathrm{Norm}}^+\) fails for a represented orbit, the failure is not
left unnamed.

::::{prf:definition} Critical normalization defects
:label: def-critical-normalization-defects

For a represented renormalized orbit, define the following defect certificates:

1. **Critical zero-collapse**
   ```{math}
   K_{L^3\mathrm{Zero}}^-:
   \qquad
   \sup_{\tau\ge\tau_0}N_3(\tau)<\infty
   \quad\text{and}\quad
   \liminf_{\tau\to\infty}N_3(\tau)=0.
   ```
2. **Critical infinite-mass defect**
   ```{math}
   K_{L^3\mathrm{Inf}}^-:
   \qquad
   \sup_{\tau\ge\tau_0}N_3(\tau)=\infty
   \quad\text{or}\quad
   N_3(\tau)=\infty\text{ for some }\tau.
   ```
3. **Critical measurability/domain defect**
   ```{math}
   K_{L^3\mathrm{Dom}}^-:
   \qquad
   N_3(\tau)\text{ is not a well-defined extended measurable critical norm
   on the represented branch.}
   ```

The zero-collapse certificate includes boundedness so that
\(K_{L^3\mathrm{Zero}}^-\) and \(K_{L^3\mathrm{Inf}}^-\) are disjoint. If an
orbit both has arbitrarily small critical mass and unbounded critical mass, it
is classified as \(K_{L^3\mathrm{Inf}}^-\) in the ordered theorem below.

::::

## Nonzero-mass version of the compact barrier

The exact equality \(\|V(\tau)\|_3=1\) in Theorem A'' can be replaced by
\(K_{L^3\mathrm{Norm}}^+\).

::::{prf:theorem} Nonzero-mass good-window compact Type II barrier
:label: thm-nonzero-mass-good-window-barrier

Let \((V,P,a,b)\) be a repaired-gauge renormalized Type II candidate. Assume:

1. positive finite critical mass:
   ```{math}
   K_{L^3\mathrm{Mass}}^+(\eta,M)
   ```
   for some \(0<\eta\le M<\infty\);
2. uniform global \(L^3\)-tightness:
   ```{math}
   \forall\varepsilon>0\ \exists R_\varepsilon:
   \sup_{\tau\ge\tau_0}
   \int_{|y|>R_\varepsilon}|V(y,\tau)|^3\,dy<\varepsilon;
   ```
3. uniform local windowed \(L^2_\tau H^1_y\) bounds:
   ```{math}
   \forall m\ge1:\quad
   \sup_{n\in\mathbb N}
   \int_{\tau_0+n}^{\tau_0+n+1}
   \|V(\tau)\|_{H^1(B_m)}^2\,d\tau<\infty.
   ```

Then finite total localized renormalization cost is impossible:

```{math}
\int_{\tau_0}^{\infty}
\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
```

::::

:::{prf:proof}
The proof is the proof of Theorem A'' with the exact normalization
\(\|V(\tau)\|_3=1\) replaced by the lower bound \(N_3(\tau)\ge\eta\).
We spell out the only place where the normalization is used.

Assume for contradiction that the total localized renormalization cost is
finite. The good-window selection and compactness portion of Theorem A'' gives
times \(\sigma_j\to\infty\) and a subsequence, not relabeled, such that:

1. \(V(\sigma_j)\to V_\infty\) strongly in \(L^3_{\mathrm{loc}}\);
2. the selected sequence has vanishing local dissipation on the exhaustion used
   in Theorem A'';
3. consequently \(\nabla V_\infty=0\) distributionally on every fixed ball.

These conclusions are exactly the pre-rigidity conclusions of Theorem A''. They
do not use the exact value \(\|V(\tau)\|_3=1\); they use only tightness, local
windowed \(H^1\) control, and finite total cost.

The low-dissipation limit has zero gradient on each fixed ball, hence is
spatially constant on each fixed ball. The constants are compatible as the
balls increase, so the local limit is a single constant vector \(c\) on
\(\mathbb R^3\). The upper critical-mass bound \(N_3(\sigma_j)\le M\) and
strong convergence on \(B_R\) give
\[
|c|^3 |B_R|
=
\lim_{j\to\infty}\int_{B_R}|V(y,\sigma_j)|^3\,dy
\le M^3
\]
for every \(R\). Letting \(R\to\infty\) forces \(c=0\). Thus the selected
states converge to zero strongly in \(L^3_{\mathrm{loc}}\).

Global tightness upgrades local convergence to global \(L^3\)-smallness:
choose \(R\) so that the \(L^3\)-tail outside \(B_R\) is \(<\eta^3/4\) uniformly
in \(\tau\). Strong convergence on \(B_R\) gives
\[
\int_{B_R}|V(y,\sigma_j)|^3\,dy<\eta^3/4
\]
for all sufficiently large \(j\). Hence
\[
\|V(\sigma_j)\|_{L^3(\mathbb R^3)}^3<\eta^3/2,
\]
contradicting \(N_3(\sigma_j)\ge\eta\). Therefore the total localized
renormalization cost must be infinite. \(\square\)
:::

## C3 bridge theorem

::::{prf:theorem} C3 critical \(L^3\)-normalization bridge
:label: thm-c3-l3-normalization-bridge

Let \((V,P,a,b)\) be a represented repaired-gauge Type II candidate, and attempt
to form
\[
N_3(\tau)=\|V(\tau)\|_{L^3}.
\]
Then exactly one of the following ordered outcomes holds:

1. \(K_{L^3\mathrm{Dom}}^-\) holds.
2. \(K_{L^3\mathrm{Inf}}^-\) holds.
3. \(K_{L^3\mathrm{Zero}}^-\) holds.
4. \(K_{L^3\mathrm{Norm}}^+\) holds, i.e. the candidate lies in a positive
   finite critical \(L^3\)-annulus after discarding at most a finite initial
   renormalized-time interval.

The ordering is: domain defect first, infinite-mass defect second,
zero-collapse third, positive finite annulus fourth. If outcome 4 holds, the
exact normalization hypothesis in Theorem A'' may be
replaced by \(K_{L^3\mathrm{Norm}}^+\), and the compact Type II barrier remains
valid by Theorem {prf:ref}`thm-nonzero-mass-good-window-barrier`.

Here "exactly one" means exactly one **ordered output certificate** is emitted:
the first applicable item in the displayed order. Later conditions are not
evaluated once an earlier defect has fired.

::::

:::{prf:proof}
If \(N_3\) is not a well-defined measurable critical norm on the represented
branch, then \(K_{L^3\mathrm{Dom}}^-\) holds.

Assume \(N_3\) is well-defined as an extended nonnegative measurable function.
If \(N_3(\tau)=\infty\) for some \(\tau\), or if
\(\sup_{\tau\ge\tau_0}N_3(\tau)=\infty\), then
\(K_{L^3\mathrm{Inf}}^-\) holds. If the supremum is finite but
\(\liminf_{\tau\to\infty}N_3(\tau)=0\), then \(K_{L^3\mathrm{Zero}}^-\) holds.

It remains to consider the case
\[
0<\liminf_{\tau\to\infty}N_3(\tau)
\quad\text{and}\quad
\sup_{\tau\ge\tau_0}N_3(\tau)<\infty.
\]
Discarding a finite initial interval if necessary, there are constants
\(0<\eta\le M<\infty\) such that
\(\eta\le N_3(\tau)\le M\) for all remaining \(\tau\). Since the Type II
barrier concerns the tail \(\tau\to\infty\), shifting \(\tau_0\) to this later
time does not change any infinite-cost or finite-cost conclusion. Hence the
tail version of \(K_{L^3\mathrm{Norm}}^+\) holds, and we relabel the shifted
tail initial time as \(\tau_0\).

The final statement follows from Theorem
{prf:ref}`thm-nonzero-mass-good-window-barrier`. \(\square\)
:::

## Consequence for the Type II classification

After C3, "failure of normalization" is no longer an unstructured gap. A
represented candidate either:

1. enters the nonzero finite critical branch \(K_{L^3\mathrm{Norm}}^+\), where
   the compact barrier applies;
2. collapses to zero critical mass, which is a degenerate/non-Type-II profile
   unless the representation theorem explicitly permits vanishing critical
   profiles;
3. has infinite critical mass, which lies outside the \(L^3\)-critical compact
   Type II branch;
4. fails to have a well-defined critical norm, which is a representation/domain
   defect rather than a compact Type II survivor.

Thus the classification ledger can replace the vague entry "normalization
failure" by the explicit certificates

```{math}
K_{L^3\mathrm{Zero}}^-,
\qquad
K_{L^3\mathrm{Inf}}^-,
\qquad
K_{L^3\mathrm{Dom}}^-.
```

To remove these defects completely from the Type II universe, the representation
bridge C2 should prove that genuine Type II candidates have finite critical
mass bounded away from zero on the renormalized tail.
