# C14 explicit checklist for applying `UP-TypeII` to NS3D

This note lists every certificate needed to apply the formal theorem
{prf:ref}`mt-up-type-ii` to a declared 3D Navier-Stokes Type II candidate.

It has two forms:

1. a **direct application package**, where the blocked barrier certificate is
   supplied as an input;
2. an **expanded compact-barrier package**, where the blocked barrier
   certificate is produced from the compact Type II PDE theorem and C4.

## Direct application package

::::{prf:definition} Direct NS3D `UP-TypeII` package
:label: def-c14-direct-ns3d-up-typeii-package

For \(\omega\in\mathcal U_{\mathrm{II}}^{NS}\), define

```{math}
K_{\mathrm{UPTypeIIReady},NS3D}^{\mathrm{dir}}(\omega)
```

to be the conjunction

```{math}
\begin{aligned}
K_{\mathrm{UPTypeIIReady},NS3D}^{\mathrm{dir}}(\omega)
:={}&
K_{\mathrm{Auto}}^+
\wedge
K_{\mathrm{TypeIIRoute}}^+(\omega)
\wedge
K_{D_E}^+(\omega)
\wedge
K_{\mathrm{RepBridge}}^+(\omega)\\
&\wedge
K_{\mathrm{CostBridge}}^+(\omega)
\wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega)
\wedge
K_{\mathrm{NSLocEnergyId}}^+(\omega)
\wedge
K_{\mathrm{FiniteMonoErr}}^+(\omega).
\end{aligned}
```

The entries mean:

| Certificate | Role |
|---|---|
| \(K_{\mathrm{Auto}}^+\) | NS3D is registered as the declared parabolic hypostructure backend. |
| \(K_{\mathrm{TypeIIRoute}}^+\) | C1 routes the candidate through \(K_{C_\mu}^+\wedge K_{\mathrm{SC}_\lambda}^-\wedge K_{\mathrm{Prof}_{NS}}^+\). |
| \(K_{D_E}^+\) | The physical energy inequality supplies bounded-energy Type II concentration. |
| \(K_{\mathrm{RepBridge}}^+\) | C2 supplies the repaired-gauge orbit \((V,P,a,b)\). |
| \(K_{\mathrm{CostBridge}}^+\) | C4 identifies the NS3D barrier cost with \(\tilde{\mathfrak D}_{R_0}\). |
| \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) | The Type II barrier has actually emitted the blocked certificate. |
| \(K_{\mathrm{NSLocEnergyId}}^+\) | C12 may use the localized energy identity or suitable-solution local energy inequality. |
| \(K_{\mathrm{FiniteMonoErr}}^+\) | The explicit C12 monotonicity-error density has finite tail integral. |

::::

::::{prf:theorem} Direct checklist applies formal `UP-TypeII`
:label: thm-c14-direct-checklist-applies-up-typeii

If \(K_{\mathrm{UPTypeIIReady},NS3D}^{\mathrm{dir}}(\omega)\) holds, then the
formal theorem {prf:ref}`mt-up-type-ii` applies to \(\omega\), and

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}(\omega)
```

is emitted.

::::

:::{prf:proof}
The direct package is exactly the C13 application package
\(K_{\mathrm{FormalUPTypeIIApp},NS3D}^+(\omega)\). Apply Theorem
{prf:ref}`thm-c13-formal-up-typeii-application-ns3d`. \(\square\)
:::

## Expanded compact-barrier package

The direct package treats
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) as an input. In the compact Type II
route, this blocked certificate is produced by Theorem A'' plus C4.

::::{prf:definition} Compact-barrier blocked-output payload
:label: def-c14-compact-barrier-blocked-output-payload

For a declared repaired-gauge candidate, define

```{math}
K_{\mathrm{CompactBarrierBlk},NS3D}^+(\omega)
```

to mean that the compact Type II hypotheses needed by Theorem A'' are present
and C4 compiles the resulting infinite localized renormalization cost into
`BarrierTypeII`:

```{math}
\int_{\tau_0}^{\infty}
\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau
=\infty
\quad\Longrightarrow\quad
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega).
```

Equivalently, this payload may be supplied by the already named C-series route

```{math}
K_{\mathrm{ClassComplete}}^+(\omega)
\wedge
K_{L^3\mathrm{Tight}}^+(\omega)
\wedge
K_{\mathrm{WinH1}}^+(\omega),
```

together with C4's cost bridge, whenever those certificates are the active
route to Theorem A''.

::::

::::{prf:definition} Expanded NS3D `UP-TypeII` package
:label: def-c14-expanded-ns3d-up-typeii-package

Define

```{math}
K_{\mathrm{UPTypeIIReady},NS3D}^{\mathrm{exp}}(\omega)
```

to be the conjunction

```{math}
\begin{aligned}
K_{\mathrm{UPTypeIIReady},NS3D}^{\mathrm{exp}}(\omega)
:={}&
K_{\mathrm{Auto}}^+
\wedge
K_{\mathrm{TypeIIRoute}}^+(\omega)
\wedge
K_{D_E}^+(\omega)
\wedge
K_{\mathrm{RepBridge}}^+(\omega)\\
&\wedge
K_{\mathrm{CostBridge}}^+(\omega)
\wedge
K_{\mathrm{CompactBarrierBlk},NS3D}^+(\omega)
\wedge
K_{\mathrm{NSLocEnergyId}}^+(\omega)
\wedge
K_{\mathrm{FiniteMonoErr}}^+(\omega).
\end{aligned}
```

::::

::::{prf:theorem} Expanded checklist applies formal `UP-TypeII`
:label: thm-c14-expanded-checklist-applies-up-typeii

If \(K_{\mathrm{UPTypeIIReady},NS3D}^{\mathrm{exp}}(\omega)\) holds, then the
formal theorem {prf:ref}`mt-up-type-ii` applies to \(\omega\), and

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}(\omega)
```

is emitted.

::::

:::{prf:proof}
The compact-barrier payload emits
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega)\). Replacing
\(K_{\mathrm{CompactBarrierBlk},NS3D}^+(\omega)\) in the expanded package by
this emitted blocked certificate gives the direct package of Definition
{prf:ref}`def-c14-direct-ns3d-up-typeii-package`. The direct theorem then
applies. \(\square\)
:::

## Fully explicit remaining analytic tail

The only term in the direct checklist not already represented by C1--C4/C10--C13
or the dataset route is

```{math}
K_{\mathrm{FiniteMonoErr}}^+.
```

Explicitly, it is

```{math}
\int_{\tau_0}^{\infty}B_{R_0}(\tau)\,d\tau<\infty,
```

where

```{math}
\begin{aligned}
B_{R_0}(\tau)
:={}&
\frac{\nu}{2}\left|\int |V|^2\Delta\phi\right|
+\frac12\left|\int |V|^2V\cdot\nabla\phi\right|
+\left|\int P\,V\cdot\nabla\phi\right|\\
&+\frac12|b(\tau)|\left|\int |V|^2\nabla\phi\right|
+\frac12 a_-(\tau)\int |V|^2\phi
+\frac12|a(\tau)|\left|\int |V|^2 y\cdot\nabla\phi\right|.
\end{aligned}
```

Thus a completely explicit proof that the formal theorem applies to NS3D is
reduced to the six finite-tail subcertificates:

```{math}
K_{\mathrm{ViscCutErr}}^+,\quad
K_{\mathrm{ConvFluxErr}}^+,\quad
K_{\mathrm{PressureFluxErr}}^+,\quad
K_{\mathrm{CenterDriftErr}}^+,\quad
K_{\mathrm{ScaleNegErr}}^+,\quad
K_{\mathrm{ScaleCutErr}}^+.
```

These are not windowed estimates. Each is a tail-integrability statement over
\([\tau_0,\infty)\).

## Final answer encoded by C14

The formal theorem {prf:ref}`mt-up-type-ii` can be used for NS3D exactly under
one of the two packages:

```{math}
K_{\mathrm{UPTypeIIReady},NS3D}^{\mathrm{dir}}
\quad\text{or}\quad
K_{\mathrm{UPTypeIIReady},NS3D}^{\mathrm{exp}}.
```

Without one of these packages, using generic `UP-TypeII` for NS3D is not
licensed by the current proof stack.

## Moving-cutoff alternative

C15 provides an alternative to the fixed-cutoff certificate
\(K_{\mathrm{FiniteMonoErr}}^+\). Instead of requiring finite tail integrability
of the fixed-annulus density \(B_{R_0}\), one may supply the moving-annulus
package

```{math}
K_{\mathrm{MoveAnnErr}}^+,
```

which emits \(K_{\mathrm{MovingFiniteMonoErr}}^+\) and gives the same corrected
monotonicity input with a moving cutoff \(R(\tau)\to\infty\).

Thus the C14 checklist may replace

```{math}
K_{\mathrm{FiniteMonoErr}}^+
```

by

```{math}
K_{\mathrm{MoveAnnErr}}^+
```

provided \(K_{\mathrm{MovingCostBridge}}^+\) holds, i.e. the C4 barrier cost is
interpreted with the corresponding moving cutoff or is tail-comparable to it in
the declared `BarrierTypeII` evaluator. The moving route still requires the
scale-negative tail certificate
\(K_{\mathrm{ScaleNegL^1}}^+\).
