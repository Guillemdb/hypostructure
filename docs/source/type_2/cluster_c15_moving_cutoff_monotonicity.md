# C15 moving-cutoff monotonicity replacement

This note replaces the fixed-cutoff finite-error certificate

```{math}
K_{\mathrm{FiniteMonoErr}}^+
```

from C12 by a moving-cutoff version. The purpose is to make the finite-error
tail plausible under no-radiation/tightness information: instead of forcing all
cutoff fluxes through a fixed annulus forever, we let the annulus move outward.

This does not make the finite-error certificate automatic. A moving cutoff
introduces a new cutoff-speed term, and the scale-negative contribution remains
a genuine obstruction. C15 isolates the exact summability certificates needed.

## Moving cutoff setup

Let \(\phi\in C_c^\infty(\mathbb R^3)\) be radial, \(0\le\phi\le1\),
\(\phi=1\) on \(B_1\), and \(\phi=0\) outside \(B_2\). Let

```{math}
\phi_R(y):=\phi(y/R).
```

Let \(R:[\tau_0,\infty)\to[R_0,\infty)\) be locally absolutely continuous and
nondecreasing. Define

```{math}
\Phi(\tau,y):=\phi_{R(\tau)}(y),
```

```{math}
\mathcal E_R(\tau)
:=
\frac12\int |V(y,\tau)|^2\Phi(\tau,y)\,dy,
```

and

```{math}
\tilde{\mathfrak D}_{R(\tau)}(\tau)
:=
\nu\int |\nabla V|^2\Phi
+a_+(\tau)\int |V|^2\Phi.
```

The moving annulus is

```{math}
A_{R(\tau)}:=\{R(\tau)\le |y|\le 2R(\tau)\}.
```

## Moving-cutoff error density

::::{prf:definition} Moving monotonicity-error density
:label: def-c15-moving-monotonicity-error-density

Define

```{math}
\begin{aligned}
B_{\mathrm{mov}}(\tau)
:={}&
\frac12\left|\int |V|^2\partial_\tau\Phi\right|
+\frac{\nu}{2}\left|\int |V|^2\Delta\Phi\right|
+\frac12\left|\int |V|^2V\cdot\nabla\Phi\right|
+\left|\int P\,V\cdot\nabla\Phi\right|\\
&+\frac12|b(\tau)|\left|\int |V|^2\nabla\Phi\right|
+\frac12 a_-(\tau)\int |V|^2\Phi
+\frac12|a(\tau)|\left|\int |V|^2y\cdot\nabla\Phi\right|.
\end{aligned}
```

The moving finite-error certificate is

```{math}
K_{\mathrm{MovingFiniteMonoErr}}^+:
\qquad
\int_{\tau_0}^{\infty}B_{\mathrm{mov}}(\tau)\,d\tau<\infty.
```

::::

The pressure term is again independent of the pressure normalization because
\(\int V\cdot\nabla\Phi=0\).

## Moving corrected monotonicity

::::{prf:theorem} C15 moving-cutoff corrected monotonicity
:label: thm-c15-moving-cutoff-corrected-monotonicity

Assume \(K_{\mathrm{RepBridge}}^+\), \(K_{\mathrm{CostBridge}}^+\),
\(K_{\mathrm{NSLocEnergyId}}^+\), and
\(K_{\mathrm{MovingFiniteMonoErr}}^+\). Then the moving corrected energy

```{math}
\mathcal E_R^{\mathrm{corr}}(\tau)
:=
\mathcal E_R(\tau)
+\int_{\tau}^{\infty}B_{\mathrm{mov}}(s)\,ds
```

satisfies

```{math}
\frac{d}{d\tau}\mathcal E_R^{\mathrm{corr}}(\tau)
+\frac12\tilde{\mathfrak D}_{R(\tau)}(\tau)
\le0
```

in distributions. Consequently the moving-cutoff version of
\(K_{\mathrm{NSLocMonoTrans}}^+\) is emitted.

::::

:::{prf:proof}
Use the localized energy identity with the time-dependent test function
\(\Phi(\tau,y)\). Compared with C12, the only new term is
\(\frac12\int |V|^2\partial_\tau\Phi\), coming from differentiating the cutoff
inside \(\mathcal E_R\). All other terms are identical with \(\phi\) replaced
by \(\Phi\). Add \(\frac12\tilde{\mathfrak D}_{R(\tau)}\) to both sides. The
gradient term leaves \(-\frac{\nu}{2}\int|\nabla V|^2\Phi\le0\), and the core
scale term contributes \(\frac12a_-\int|V|^2\Phi\). Every remaining term is
bounded above by the corresponding absolute-value term in
\(B_{\mathrm{mov}}\). Hence

```{math}
\frac{d}{d\tau}\mathcal E_R(\tau)
+\frac12\tilde{\mathfrak D}_{R(\tau)}(\tau)
\le
B_{\mathrm{mov}}(\tau).
```

The moving finite-error certificate makes the correction tail finite and
absolutely continuous on compact intervals. Subtracting
\(B_{\mathrm{mov}}\) by differentiating the correction tail gives the claimed
monotonicity inequality. \(\square\)
:::

## Summable-annulus sufficient package

Moving cutoffs are useful only if the annular errors become summable. C15
therefore defines the exact sufficient package.

::::{prf:definition} Summable moving-annulus error package
:label: def-c15-summable-moving-annulus-package

The certificate

```{math}
K_{\mathrm{MoveAnnErr}}^+
```

means that there exists a nondecreasing locally absolutely continuous
\(R(\tau)\to\infty\) such that each of the following seven quantities is
integrable on \([\tau_0,\infty)\):

```{math}
\left|\int |V|^2\partial_\tau\Phi\right|,
\quad
\left|\int |V|^2\Delta\Phi\right|,
\quad
\left|\int |V|^2V\cdot\nabla\Phi\right|,
\quad
\left|\int P\,V\cdot\nabla\Phi\right|,
```

```{math}
|b|\left|\int |V|^2\nabla\Phi\right|,
\quad
a_-\int |V|^2\Phi,
\quad
|a|\left|\int |V|^2y\cdot\nabla\Phi\right|.
```

Equivalently, \(K_{\mathrm{MoveAnnErr}}^+\) is the conjunction

```{math}
K_{\partial_\tau\mathrm{CutErr}}^+
\wedge
K_{\mathrm{MovViscCutErr}}^+
\wedge
K_{\mathrm{MovConvFluxErr}}^+
\wedge
K_{\mathrm{MovPressureFluxErr}}^+
\wedge
K_{\mathrm{MovCenterDriftErr}}^+
\wedge
K_{\mathrm{ScaleNegL^1}}^+
\wedge
K_{\mathrm{MovScaleCutErr}}^+.
```

::::

::::{prf:lemma} Summable moving-annulus errors imply moving finite error
:label: lem-c15-summable-moving-annulus-implies-moving-finite-error

If \(K_{\mathrm{MoveAnnErr}}^+\) holds, then

```{math}
K_{\mathrm{MovingFiniteMonoErr}}^+.
```

::::

:::{prf:proof}
\(B_{\mathrm{mov}}\) is the finite positive linear combination of the seven
quantities listed in Definition
{prf:ref}`def-c15-summable-moving-annulus-package`. If each is integrable, then
so is \(B_{\mathrm{mov}}\). \(\square\)
:::

## How tightness helps, and what it cannot do

Local compact-cylinder mass retention can help produce the moving-annulus certificates for
the cutoff-supported terms by choosing \(R(\tau)\) so that the annular \(L^3\)
mass is small on successive unit windows. This is a summability upgrade of
tightness, not a formal consequence of tightness alone.

::::{prf:definition} Summable tightness schedule
:label: def-c15-summable-tightness-schedule

The certificate

```{math}
K_{\mathrm{SummTightSched}}^+
```

means that there exists a nondecreasing sequence \(R_n\to\infty\) such that on
unit windows \(I_n=[\tau_0+n,\tau_0+n+1]\), the annular critical tails obey

```{math}
\sum_{n=0}^{\infty}
\sup_{\tau\in I_n}
\|V(\tau)\|_{L^3(A_{R_n})}^3
<\infty,
```

and the same schedule is compatible with the pressure-tail normalization used
for \(P\).

::::

This schedule is stronger than \(K_{L^3\mathrm{Tight}}^+\). Uniform tightness
allows each window tail to be made small after choosing a large radius, but it
does not by itself provide one monotone radius schedule with a summable series.

## Main moving-cutoff replacement theorem

::::{prf:definition} Moving-cost barrier admissibility
:label: def-c15-moving-cost-barrier-admissibility

The certificate

```{math}
K_{\mathrm{MovingCostBridge}}^+
```

means that the moving cost

```{math}
\tilde{\mathfrak D}_{R(\tau)}(\tau)
=
\nu\int |\nabla V|^2\phi_{R(\tau)}
+a_+(\tau)\int |V|^2\phi_{R(\tau)}
```

is accepted by the declared `BarrierTypeII` evaluator as the active Type II
barrier cost, or is compared to the fixed C4 cost by a positive constant on
the tail so that divergence and the blocked certificate are unchanged.

Without this certificate, C15 gives a moving corrected monotonicity formula but
does not by itself license replacing the fixed C4 cost in the formal
`UP-TypeII` application package.

::::

::::{prf:theorem} C15 moving-cutoff replacement for fixed finite error
:label: thm-c15-moving-cutoff-replacement

Assume

```{math}
K_{\mathrm{RepBridge}}^+,
\qquad
K_{\mathrm{CostBridge}}^+,
\qquad
K_{\mathrm{MovingCostBridge}}^+,
\qquad
K_{\mathrm{NSLocEnergyId}}^+,
\qquad
K_{\mathrm{MoveAnnErr}}^+.
```

Then the moving-cutoff monotonicity translation needed by the generic
`UP-TypeII` route is available.

::::

:::{prf:proof}
By Lemma {prf:ref}`lem-c15-summable-moving-annulus-implies-moving-finite-error`,
\(K_{\mathrm{MoveAnnErr}}^+\) emits
\(K_{\mathrm{MovingFiniteMonoErr}}^+\). Apply Theorem
{prf:ref}`thm-c15-moving-cutoff-corrected-monotonicity`. The certificate
\(K_{\mathrm{MovingCostBridge}}^+\) identifies this moving monotonicity cost
with the active `BarrierTypeII` cost, or with an equivalent tail-comparable
cost, so the generic `UP-TypeII` route may use it in place of the fixed C12
finite-error input. \(\square\)
:::

## Remaining hard term

The moving cutoff addresses the annular cutoff terms. It does not remove the
scale-negative contribution

```{math}
K_{\mathrm{ScaleNegL^1}}^+:
\qquad
\int_{\tau_0}^{\infty}
a_-(\tau)\int |V(y,\tau)|^2\Phi(\tau,y)\,dy\,d\tau
<\infty.
```

This is the genuine scale-drift obstruction. A proof of eventual one-sided
scale drift \(a_-\in L^1_\tau\), or an equivalent repaired-gauge sign theorem,
would discharge it.
