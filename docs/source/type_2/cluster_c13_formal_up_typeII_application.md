# C13 formal `UP-TypeII` application to NS3D

This note is the final composition theorem for the generic `UP-TypeII` route.
It answers the operational question:

When may the formal theorem {prf:ref}`mt-up-type-ii` be applied to a declared
NS3D Type II candidate?

The answer is: exactly when the C1--C4 route is present, the blocked
`BarrierTypeII` certificate has actually been emitted, and the C12 finite
monotonicity-error tail has been supplied.

## Application package

::::{prf:definition} NS3D formal-UP application package
:label: def-c13-ns3d-formal-up-application-package

For a declared candidate \(\omega\in\mathcal U_{\mathrm{II}}^{NS}\), define

```{math}
K_{\mathrm{FormalUPTypeIIApp},NS3D}^+(\omega)
```

to be the conjunction

```{math}
\begin{aligned}
K_{\mathrm{FormalUPTypeIIApp},NS3D}^+(\omega)
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

Here \(K_{\mathrm{FiniteMonoErr}}^+\) is the C12 finite-tail certificate for
the explicit monotonicity-error density \(B_{R_0}\).

::::

## Formal application theorem

::::{prf:theorem} C13 formal `UP-TypeII` application theorem for NS3D
:label: thm-c13-formal-up-typeii-application-ns3d

Assume \(K_{\mathrm{FormalUPTypeIIApp},NS3D}^+(\omega)\). Then the formal
theorem {prf:ref}`mt-up-type-ii` applies to \(\omega\), and the declared NS3D
backend emits

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}(\omega).
```

::::

:::{prf:proof}
By \(K_{\mathrm{RepBridge}}^+(\omega)\),
\(K_{\mathrm{CostBridge}}^+(\omega)\),
\(K_{\mathrm{NSLocEnergyId}}^+(\omega)\), and
\(K_{\mathrm{FiniteMonoErr}}^+(\omega)\), C12 emits
\(K_{\mathrm{NSLocMonoTrans}}^+(\omega)\). Then C11 applies to

```{math}
K_{\mathrm{Auto}}^+,
\quad
K_{\mathrm{TypeIIRoute}}^+(\omega),
\quad
K_{D_E}^+(\omega),
\quad
K_{\mathrm{RepBridge}}^+(\omega),
\quad
K_{\mathrm{CostBridge}}^+(\omega),
\quad
K_{\mathrm{NSLocMonoTrans}}^+(\omega),
```

and emits

```{math}
K_{\mathrm{GenericUPTypeIIAdmiss}}^+(\omega).
```

C1 gives \(K_{\mathrm{SC}_\lambda}^-(\omega)\) as part of
\(K_{\mathrm{TypeIIRoute}}^+(\omega)\). The application package also includes
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega)\). Therefore Lemma
{prf:ref}`lem-c10-generic-up-admissibility-licenses-up-typeii` applies the
formal theorem {prf:ref}`mt-up-type-ii` to \(\omega\) and emits
\(K_{\mathrm{SC}_\lambda}^{\sim}(\omega)\). \(\square\)
:::

## Necessity ledger for the current route

::::{prf:corollary} What is still needed to apply formal `UP-TypeII`
:label: cor-c13-what-is-needed-for-formal-up-typeii

On the declared NS3D C-series route, C1, C2, C4, C10, C11, and C12 reduce the
formal `UP-TypeII` application problem to the following checklist:

```{math}
K_{\mathrm{Auto}}^+,
\quad
K_{\mathrm{TypeIIRoute}}^+,
\quad
K_{D_E}^+,
\quad
K_{\mathrm{RepBridge}}^+,
\quad
K_{\mathrm{CostBridge}}^+,
\quad
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}},
\quad
K_{\mathrm{NSLocEnergyId}}^+,
\quad
K_{\mathrm{FiniteMonoErr}}^+.
```

The route, energy, representation, and cost-adapter entries are supplied by the
current C-series route once the declared backend payloads are accepted. The
blocked entry \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) must still actually be
emitted, for example by C4 after the compact barrier gives infinite localized
renormalization cost. The local-energy-identity entry
\(K_{\mathrm{NSLocEnergyId}}^+\) is the regularity/suitable-solution payload
needed to use the C12 localized energy identity. Once the blocked certificate
and the local-energy-identity payload are present, the remaining analytic
obstruction to applying the formal theorem {prf:ref}`mt-up-type-ii` through the
generic route is the finite tail estimate

```{math}
\int_{\tau_0}^{\infty}B_{R_0}(\tau)\,d\tau<\infty.
```

::::

:::{prf:proof}
The checklist is exactly Definition
{prf:ref}`def-c13-ns3d-formal-up-application-package`. The preceding theorem
proves sufficiency. The C-series documents discharge the route, energy,
representation, and cost-adapter entries. The blocked-barrier entry is supplied
only after \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) has actually been emitted
by the barrier route. C12 identifies the remaining localized monotonicity entry
with \(K_{\mathrm{FiniteMonoErr}}^+\). \(\square\)
:::

## Non-overclaim clause

C13 does not prove that \(K_{\mathrm{FiniteMonoErr}}^+\) follows from
tightness, windowed \(H^1\), or finite physical energy. Those hypotheses supply
local or windowed controls, while \(K_{\mathrm{FiniteMonoErr}}^+\) is a global
tail-integrability statement over the infinite renormalized-time interval.

Therefore the correct final statement is:

```{math}
K_{\mathrm{FormalUPTypeIIApp},NS3D}^+
\Longrightarrow
\text{formal } \mathrm{UP}\text{-}\mathrm{TypeII}\text{ applies to NS3D}.
```

Without \(K_{\mathrm{FiniteMonoErr}}^+\), the formal theorem is not yet
licensed for NS3D through the generic route.
