# C12 NS3D localized monotonicity translation

This note proves the corrected localized monotonicity certificate

```{math}
K_{\mathrm{NSLocMonoTrans}}^+
```

needed by C11 to apply the formal theorem {prf:ref}`mt-up-type-ii` to a
declared NS3D repaired-gauge Type II branch.

The proof is deliberately exact about the obstruction. The renormalized local
energy identity does not give the heat-model monotonicity formula for free.
Pressure flux, cutoff transport, translation drift, and the signed scale term
must be absorbed into a finite tail correction. Once that correction is
available, the formal monotonicity input is recovered.

## Setup

Let \((V,P,a,b)\) be a repaired-gauge renormalized NS3D Type II candidate on
the tail \([\tau_0,\infty)\). Let
\(\phi=\phi_{R_0}\in C_c^\infty(\mathbb R^3)\), \(0\le\phi\le1\), be the
cutoff used in C4. Define

```{math}
\mathcal E_\phi(\tau)
:=
\frac12\int_{\mathbb R^3}|V(y,\tau)|^2\phi(y)\,dy,
```

```{math}
A_\phi(\tau)
:=
\int_{\mathbb R^3}|\nabla V(y,\tau)|^2\phi(y)\,dy,
\qquad
M_\phi(\tau)
:=
\int_{\mathbb R^3}|V(y,\tau)|^2\phi(y)\,dy,
```

and

```{math}
\tilde{\mathfrak D}_{R_0}(\tau)
:=
\nu A_\phi(\tau)+a_+(\tau)M_\phi(\tau),
\qquad
a_+:=\max(a,0).
```

The localized energy identity from Lemma 4 of
[compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md)
is

```{math}
\begin{aligned}
\frac{d}{d\tau}\mathcal E_\phi
=&
-\nu A_\phi
+\frac{\nu}{2}\int |V|^2\Delta\phi
+\frac12\int |V|^2V\cdot\nabla\phi
+\int P\,V\cdot\nabla\phi \\
&-\frac{a}{2}M_\phi
-\frac{a}{2}\int |V|^2 y\cdot\nabla\phi
-\frac12\,b\cdot\int |V|^2\nabla\phi .
\end{aligned}
```

## Explicit monotonicity-error density

::::{prf:definition} C12 monotonicity error density
:label: def-c12-monotonicity-error-density

Define

```{math}
\begin{aligned}
B_{R_0}(\tau)
:={}&
\frac{\nu}{2}\left|\int |V|^2\Delta\phi\right|
+\frac12\left|\int |V|^2V\cdot\nabla\phi\right|
+\left|\int P\,V\cdot\nabla\phi\right|\\
&+\frac12|b(\tau)|\left|\int |V|^2\nabla\phi\right|
+\frac12 a_-(\tau)M_\phi(\tau)
+\frac12|a(\tau)|\left|\int |V|^2 y\cdot\nabla\phi\right|,
\end{aligned}
```

where \(a_-:=\max(-a,0)\).

This definition is independent of the pressure normalization. Indeed, replacing
\(P\) by \(P+c(\tau)\) changes the pressure term by

```{math}
c(\tau)\int V\cdot\nabla\phi
=
-c(\tau)\int \phi\,\nabla\cdot V
=0.
```

Thus \(B_{R_0}\) is well-defined modulo the usual time-dependent pressure
constants. The certificate includes measurability of each displayed term, so
\(B_{R_0}\) is a nonnegative measurable function on the tail.

The finite monotonicity-error certificate is

```{math}
K_{\mathrm{FiniteMonoErr}}^+:
\qquad
\int_{\tau_0}^{\infty}B_{R_0}(\tau)\,d\tau<\infty.
```

::::

The term \(a_-M_\phi\) is essential. The C4 cost contains \(a_+M_\phi\), while
the local energy identity contains \(-aM_\phi/2\). Thus the scale term has the
right sign for \(a\ge0\), but for \(a<0\) it produces a positive error not
included in the C4 cost.

## Corrected monotonicity theorem

::::{prf:theorem} C12 corrected localized monotonicity
:label: thm-c12-corrected-localized-monotonicity

Assume the repaired-gauge orbit is regular enough for the localized energy
identity above to hold on the tail and assume
\(K_{\mathrm{FiniteMonoErr}}^+\). Define the corrected local energy

```{math}
\mathcal E_\phi^{\mathrm{corr}}(\tau)
:=
\mathcal E_\phi(\tau)
+\int_{\tau}^{\infty}B_{R_0}(s)\,ds.
```

Then, in the distributional sense on \((\tau_0,\infty)\),

```{math}
\frac{d}{d\tau}\mathcal E_\phi^{\mathrm{corr}}(\tau)
+\frac12\tilde{\mathfrak D}_{R_0}(\tau)
\le0.
```

Consequently \(K_{\mathrm{NSLocMonoTrans}}^+\) is emitted with cost comparable
to the C4 Type II barrier cost.

::::

:::{prf:proof}
Start from the localized identity. Add

```{math}
\frac12\tilde{\mathfrak D}_{R_0}
=
\frac{\nu}{2}A_\phi+\frac12a_+M_\phi
```

to both sides. The gradient contribution becomes

```{math}
-\nu A_\phi+\frac{\nu}{2}A_\phi
=
-\frac{\nu}{2}A_\phi
\le0.
```

The core scale contribution is

```{math}
-\frac{a}{2}M_\phi+\frac12a_+M_\phi
=
\frac12a_-M_\phi
\le
B_{R_0}(\tau).
```

Every remaining term in the localized identity is bounded above by its
corresponding absolute-value contribution in \(B_{R_0}(\tau)\). Therefore

```{math}
\frac{d}{d\tau}\mathcal E_\phi(\tau)
+\frac12\tilde{\mathfrak D}_{R_0}(\tau)
\le
B_{R_0}(\tau)
```

in distributions. Since \(K_{\mathrm{FiniteMonoErr}}^+\) gives
\(\int_{\tau_0}^\infty B_{R_0}<\infty\), the correction tail is finite for
every \(\tau\ge\tau_0\), absolutely continuous on compact subintervals, and

```{math}
\frac{d}{d\tau}
\left(\int_{\tau}^{\infty}B_{R_0}(s)\,ds\right)
=
-B_{R_0}(\tau).
```

Adding this identity to the previous differential inequality gives

```{math}
\frac{d}{d\tau}\mathcal E_\phi^{\mathrm{corr}}(\tau)
+\frac12\tilde{\mathfrak D}_{R_0}(\tau)
\le0.
```

Multiplication of the Type II barrier cost by the positive constant \(1/2\)
does not change divergence or the blocked `BarrierTypeII` certificate. Hence
the corrected monotonicity formula is compatible with the C4 cost and supplies
\(K_{\mathrm{NSLocMonoTrans}}^+\). \(\square\)
:::

## Certificate form

::::{prf:definition} Localized energy identity certificate
:label: def-c12-local-energy-identity-certificate

The certificate

```{math}
K_{\mathrm{NSLocEnergyId}}^+
```

means that the C2 repaired-gauge orbit has enough regularity to justify the
localized energy identity of Lemma 4 in
[compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md),
or is obtained by a standard suitable-solution approximation for which that
identity passes to the limit as a local energy inequality with the same error
terms used in Definition {prf:ref}`def-c12-monotonicity-error-density`.

::::

::::{prf:corollary} C12 emits \(K_{\mathrm{NSLocMonoTrans}}^+\)
:label: cor-c12-emits-ns-localized-monotonicity

Assume

```{math}
K_{\mathrm{RepBridge}}^+,
\qquad
K_{\mathrm{CostBridge}}^+,
\qquad
K_{\mathrm{NSLocEnergyId}}^+,
\qquad
K_{\mathrm{FiniteMonoErr}}^+,
```

Then

```{math}
K_{\mathrm{NSLocMonoTrans}}^+.
```

::::

:::{prf:proof}
\(K_{\mathrm{RepBridge}}^+\) supplies the repaired-gauge orbit
\((V,P,a,b)\). \(K_{\mathrm{CostBridge}}^+\) identifies the C4 cost as
\(\tilde{\mathfrak D}_{R_0}\). \(K_{\mathrm{NSLocEnergyId}}^+\) permits use of
the localized energy identity or its suitable-solution inequality limit. The
finite-error certificate gives the tail correction. Theorem
{prf:ref}`thm-c12-corrected-localized-monotonicity` then emits
\(K_{\mathrm{NSLocMonoTrans}}^+\). \(\square\)
:::

## How to discharge the finite-error certificate

The remaining analytic content is now completely explicit:

```{math}
\int_{\tau_0}^{\infty}B_{R_0}(\tau)\,d\tau<\infty.
```

This breaks into the following named subcertificates:

```{math}
K_{\mathrm{ViscCutErr}}^+,\quad
K_{\mathrm{ConvFluxErr}}^+,\quad
K_{\mathrm{PressureFluxErr}}^+,\quad
K_{\mathrm{CenterDriftErr}}^+,\quad
K_{\mathrm{ScaleNegErr}}^+,\quad
K_{\mathrm{ScaleCutErr}}^+.
```

They respectively assert tail integrability of the six terms in
Definition {prf:ref}`def-c12-monotonicity-error-density`. Their conjunction is
equivalent to \(K_{\mathrm{FiniteMonoErr}}^+\).

The implemented C6/C7 estimates give uniform window bounds for several of
these quantities, but uniform window bounds are not the same as tail
integrability over \([\tau_0,\infty)\). Therefore C12 proves the monotonicity
translation once the finite-tail certificate is supplied; it does not claim
that tightness and windowed \(H^1\) alone force finite monotonicity error.

## Consequence for C11

Combining C12 with C11 gives the formal `UP-TypeII` route:

```{math}
K_{\mathrm{Auto}}^+
\wedge
K_{\mathrm{TypeIIRoute}}^+
\wedge
K_{D_E}^+
\wedge
K_{\mathrm{RepBridge}}^+
\wedge
K_{\mathrm{CostBridge}}^+
\wedge
K_{\mathrm{NSLocEnergyId}}^+
\wedge
K_{\mathrm{FiniteMonoErr}}^+
\Longrightarrow
K_{\mathrm{GenericUPTypeIIAdmiss}}^+.
```

Thus the exact remaining task is no longer vague localized monotonicity; it is
the finite-tail estimate for the explicit error density \(B_{R_0}\).
