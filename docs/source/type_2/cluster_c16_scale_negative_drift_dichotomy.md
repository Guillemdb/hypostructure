# C16 scale-negative drift dichotomy

C15 isolates the scale-negative term

```{math}
K_{\mathrm{ScaleNegL^1}}^+:
\qquad
\int_{\tau_0}^{\infty}
a_-(\tau)\int |V(y,\tau)|^2\Phi(\tau,y)\,dy\,d\tau<\infty.
```

This note proves the exact repaired-gauge dichotomy for that term. It does not
claim that genuine scale collapse automatically gives
\(K_{\mathrm{ScaleNegL^1}}^+\). With the convention of the master note,

```{math}
a(\tau)=\lambda(t)\lambda_t(t)
\qquad\text{and}\qquad
\frac{d}{d\tau}\log\lambda(\tau)=a(\tau),
```

physical concentration \(\lambda(\tau)\to0\) forces negative scale drift, not
positive scale drift. Therefore the best rigorous statement is a dichotomy:
either the negative drift is integrable after weighting by the moving localized
\(L^2\) mass, or the candidate is routed into an explicit
scale-collapse-drift obstruction.

## Setup

Let \(R(\tau)\) and \(\Phi(\tau,y)=\phi_{R(\tau)}(y)\) be as in C15. Define the
moving localized \(L^2\) mass

```{math}
M_R(\tau):=\int |V(y,\tau)|^2\Phi(\tau,y)\,dy.
```

The C15 scale-negative certificate is exactly

```{math}
\int_{\tau_0}^{\infty}a_-(\tau)M_R(\tau)\,d\tau<\infty.
```

::::{prf:definition} Negative scale-drift certificates
:label: def-c16-negative-scale-drift-certificates

The finite unweighted negative-drift certificate is

```{math}
K_{\mathrm{NegDriftL^1}}^+:
\qquad
\int_{\tau_0}^{\infty}a_-(\tau)\,d\tau<\infty.
```

The moving localized \(L^2\)-upper certificate is

```{math}
K_{\mathrm{CoreL^2Bd}}^+:
\qquad
\sup_{\tau\ge\tau_0}M_R(\tau)<\infty.
```

The moving localized \(L^2\)-floor certificate is

```{math}
K_{\mathrm{CoreL^2Floor}}^+:
\qquad
\exists m_0>0,\ \exists\tau_1\ge\tau_0
\quad\text{such that}\quad
M_R(\tau)\ge m_0
\quad\text{for a.e. }\tau\ge\tau_1.
```

The scale-collapse-drift obstruction is

```{math}
K_{\mathrm{ScaleCollapseDrift}}^-:
\qquad
\int_{\tau_0}^{\infty}a_-(\tau)M_R(\tau)\,d\tau=\infty.
```

::::

The obstruction is named with a minus sign because it is exactly the failure of
the C15 payload \(K_{\mathrm{ScaleNegL^1}}^+\).

## Finite negative drift discharges the C15 scale term

::::{prf:lemma} Bounded core mass plus finite negative drift gives scale-negative integrability
:label: lem-c16-bounded-core-finite-neg-drift

Assume \(K_{\mathrm{CoreL^2Bd}}^+\) and
\(K_{\mathrm{NegDriftL^1}}^+\). Then

```{math}
K_{\mathrm{ScaleNegL^1}}^+.
```

::::

:::{prf:proof}
Let \(M_*:=\sup_{\tau\ge\tau_0}M_R(\tau)<\infty\). Since
\(a_-\ge0\),

```{math}
\int_{\tau_0}^{\infty}a_-(\tau)M_R(\tau)\,d\tau
\le
M_*\int_{\tau_0}^{\infty}a_-(\tau)\,d\tau
<\infty.
```

This is precisely \(K_{\mathrm{ScaleNegL^1}}^+\). \(\square\)
:::

::::{prf:corollary} One-sided repaired-gauge route to C15
:label: cor-c16-one-sided-gauge-route-to-c15

Assume the C15 moving-cutoff hypotheses other than
\(K_{\mathrm{ScaleNegL^1}}^+\), and assume

```{math}
K_{\mathrm{CoreL^2Bd}}^+
\wedge
K_{\mathrm{NegDriftL^1}}^+.
```

Then the scale-negative component of \(K_{\mathrm{MoveAnnErr}}^+\) is
discharged. Hence, once the remaining moving annular terms in
\(K_{\mathrm{MoveAnnErr}}^+\) are supplied, the C15 moving-cutoff replacement
can be used in the C14 `UP-TypeII` checklist.

::::

:::{prf:proof}
Apply Lemma {prf:ref}`lem-c16-bounded-core-finite-neg-drift` to obtain
\(K_{\mathrm{ScaleNegL^1}}^+\). C15 consumes that certificate as the sixth
component of \(K_{\mathrm{MoveAnnErr}}^+\). \(\square\)
:::

## The exact finite-or-infinite dichotomy

::::{prf:theorem} C16 scale-negative finite/infinite alternative
:label: thm-c16-scale-negative-finite-infinite-alternative

For every represented repaired-gauge orbit for which \(a_-\) and \(M_R\) are
measurable and nonnegative, exactly one of the following alternatives holds:

```{math}
K_{\mathrm{ScaleNegL^1}}^+
\qquad\text{or}\qquad
K_{\mathrm{ScaleCollapseDrift}}^-.
```

::::

:::{prf:proof}
The quantity

```{math}
I_R:=\int_{\tau_0}^{\infty}a_-(\tau)M_R(\tau)\,d\tau
```

is an extended nonnegative number. Therefore either \(I_R<\infty\) or
\(I_R=\infty\), and not both. The first alternative is exactly
\(K_{\mathrm{ScaleNegL^1}}^+\). The second is exactly
\(K_{\mathrm{ScaleCollapseDrift}}^-\). \(\square\)
:::

This theorem is tautological at the measure-theoretic level, but important at
the certificate level: the C15 obstruction is no longer an unnamed analytic
gap. Its failure is a declared residual class.

## Genuine scale collapse forces the obstruction when the core mass does not vanish

The sign of \(a\) is fixed by the renormalization convention in the master
note.

::::{prf:lemma} Repaired-gauge logarithmic scale identity
:label: lem-c16-logarithmic-scale-identity

For every represented orbit using the master-note renormalization convention,

```{math}
\int_{\tau_1}^{\tau_2}a(\tau)\,d\tau
=
\log\lambda(\tau_2)-\log\lambda(\tau_1)
```

whenever \(\tau_0\le\tau_1<\tau_2<\infty\).

::::

:::{prf:proof}
In the master-note convention,
\(\tau_t=\lambda(t)^{-2}\) and \(a(\tau)=\lambda(t)\lambda_t(t)\). Hence

```{math}
\frac{d}{d\tau}\log\lambda
=
\frac{\lambda_t}{\lambda}\frac{dt}{d\tau}
=
\frac{\lambda_t}{\lambda}\lambda^2
=
\lambda\lambda_t
=a.
```

Integrating over \([\tau_1,\tau_2]\) gives the identity. \(\square\)
:::

::::{prf:theorem} Genuine collapse with nonvanishing core mass fails the C15 scale-negative payload
:label: thm-c16-collapse-core-floor-forces-scale-drift-obstruction

Assume the represented repaired-gauge orbit satisfies

```{math}
\lambda(\tau)\to0
\qquad\text{as}\qquad
\tau\to\infty,
```

and assume \(K_{\mathrm{CoreL^2Floor}}^+\). Then

```{math}
K_{\mathrm{ScaleCollapseDrift}}^-.
```

In particular, under these hypotheses the C15 moving-cutoff route cannot
discharge \(K_{\mathrm{ScaleNegL^1}}^+\).

::::

:::{prf:proof}
By Lemma {prf:ref}`lem-c16-logarithmic-scale-identity`,

```{math}
\int_{\tau_1}^{\tau_2}a(\tau)\,d\tau
=
\log\lambda(\tau_2)-\log\lambda(\tau_1).
```

Since \(\lambda(\tau)\to0\), the right-hand side tends to \(-\infty\) as
\(\tau_2\to\infty\). If \(\int_{\tau_1}^{\infty}a_-(\tau)\,d\tau<\infty\),
then

```{math}
\int_{\tau_1}^{\tau_2}a(\tau)\,d\tau
=
\int_{\tau_1}^{\tau_2}a_+(\tau)\,d\tau
-
\int_{\tau_1}^{\tau_2}a_-(\tau)\,d\tau
\ge
-
\int_{\tau_1}^{\infty}a_-(\tau)\,d\tau
>-\infty,
```

contradicting convergence to \(-\infty\). Thus
\(\int_{\tau_1}^{\infty}a_-(\tau)\,d\tau=\infty\).

By \(K_{\mathrm{CoreL^2Floor}}^+\), there are \(m_0>0\) and
\(\tau_1\ge\tau_0\) such that \(M_R(\tau)\ge m_0\) for a.e.
\(\tau\ge\tau_1\). Therefore

```{math}
\int_{\tau_0}^{\infty}a_-(\tau)M_R(\tau)\,d\tau
\ge
m_0\int_{\tau_1}^{\infty}a_-(\tau)\,d\tau
=\infty.
```

This is \(K_{\mathrm{ScaleCollapseDrift}}^-\). \(\square\)
:::

## Classification consequence for the moving-cutoff `UP-TypeII` route

::::{prf:corollary} C16 scale-drift routing for C15
:label: cor-c16-scale-drift-routing-for-c15

Assume the C15 moving-cutoff setup and the already declared NS3D repaired-gauge
backend. Then the scale-negative part of \(K_{\mathrm{MoveAnnErr}}^+\) has the
following exhaustive routing:

1. If
   \(K_{\mathrm{CoreL^2Bd}}^+\wedge K_{\mathrm{NegDriftL^1}}^+\) holds, then
   \(K_{\mathrm{ScaleNegL^1}}^+\) holds.
2. If the weighted integral is infinite, the candidate emits
   \(K_{\mathrm{ScaleCollapseDrift}}^-\).
3. If, in addition, \(\lambda(\tau)\to0\) and
   \(K_{\mathrm{CoreL^2Floor}}^+\) hold, then the second alternative is forced.

::::

:::{prf:proof}
The first statement is Lemma
{prf:ref}`lem-c16-bounded-core-finite-neg-drift`. The exhaustive
finite/infinite split is Theorem
{prf:ref}`thm-c16-scale-negative-finite-infinite-alternative`. The final
statement is Theorem
{prf:ref}`thm-c16-collapse-core-floor-forces-scale-drift-obstruction`.
\(\square\)
:::

## What C16 does and does not prove

C16 proves that the C15 scale-negative obstruction is fully classified.
It does not prove that the obstruction is absent for genuine Type II collapse.
Indeed, under the master-note sign convention, genuine scale collapse plus a
nonvanishing localized \(L^2\) core forces the obstruction.

Therefore the moving-cutoff `UP-TypeII` route can be completed in either of two
ways:

1. prove a finite negative-drift or mass-damped negative-drift theorem for the
   repaired gauge, yielding \(K_{\mathrm{ScaleNegL^1}}^+\);
2. accept \(K_{\mathrm{ScaleCollapseDrift}}^-\) as an additional residual Type
   II survivor class and try to rule it out by a different barrier, by changing
   the active Type II cost, or by replacing the monotonicity functional so that
   negative scale collapse contributes on the coercive side rather than as an
   error.

The second route is likely the honest one for physical scale collapse: the C15
moving cutoff repairs annular radiation errors, but it cannot by itself turn
the interior negative logarithmic scale drift into an integrable tail.
