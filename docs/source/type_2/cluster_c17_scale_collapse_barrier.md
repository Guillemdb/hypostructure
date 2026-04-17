# C17 scale-collapse drift barrier

C16 shows that the moving-cutoff monotonicity route has a genuine obstruction:

```{math}
K_{\mathrm{ScaleCollapseDrift}}^-:
\qquad
\int_{\tau_0}^{\infty}a_-(\tau)M_R(\tau)\,d\tau=\infty,
\qquad
M_R(\tau)=\int |V|^2\phi_{R(\tau)}.
```

For genuine collapse \(\lambda(\tau)\to0\) with a nonvanishing localized
\(L^2\) core, C16 proves that this obstruction is forced. Therefore the right
next move is not to treat \(a_-M_R\) as a finite error. The right move is to
register it as a second Type II barrier cost.

This note implements that route.

## Scale-collapse cost

::::{prf:definition} Scale-collapse renormalization cost
:label: def-c17-scale-collapse-renormalization-cost

For a represented repaired-gauge orbit, define the scale-collapse cost density

```{math}
\mathfrak C_{\mathrm{sc}}(\tau)
:=
a_-(\tau)M_R(\tau),
\qquad
M_R(\tau):=\int |V(y,\tau)|^2\phi_{R(\tau)}(y)\,dy.
```

The cumulative scale-collapse cost is

```{math}
\mathcal C_{\mathrm{sc}}[\tau_0,\infty)
:=
\int_{\tau_0}^{\infty}\mathfrak C_{\mathrm{sc}}(\tau)\,d\tau.
```

Thus

```{math}
K_{\mathrm{ScaleCollapseDrift}}^-
\quad\Longleftrightarrow\quad
\mathcal C_{\mathrm{sc}}[\tau_0,\infty)=\infty.
```

::::

The density is nonnegative and local-in-renormalized-core. It measures the
amount of inward logarithmic scale drift carried by the localized \(L^2\) core.

## Backend bridge

::::{prf:definition} Scale-collapse cost bridge
:label: def-c17-scale-collapse-cost-bridge

The certificate

```{math}
K_{\mathrm{ScaleCollapseCostBridge}}^+
```

means that the declared NS3D `BarrierTypeII` evaluator accepts
\(\mathfrak C_{\mathrm{sc}}\) as an admissible Type II barrier cost for the
scale-collapse branch. Equivalently, the evaluator has been extended from the
default fixed cost

```{math}
\tilde{\mathfrak D}_{R_0}
=
\nu\int|\nabla V|^2\phi_{R_0}
+a_+\int|V|^2\phi_{R_0}
```

to the branchwise cost family containing

```{math}
\mathfrak C_{\mathrm{sc}}
=a_-M_R.
```

The bridge records three requirements:

1. \(\mathfrak C_{\mathrm{sc}}\) is measurable, nonnegative, and locally
   integrable on finite \(\tau\)-intervals for represented orbits.
2. Divergence of \(\mathcal C_{\mathrm{sc}}[\tau_0,\infty)\) is a registered
   blocking condition for `BarrierTypeII`.
3. The blocked certificate emitted by this branch is the same framework-level
   certificate \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) used by C4, C10, and
   C13.

::::

This is an evaluator-registration payload, analogous to C4's identity-cost
registration. It is not a PDE estimate.

## Direct barrier theorem

::::{prf:theorem} C17 scale-collapse drift emits the blocked Type II barrier
:label: thm-c17-scale-collapse-drift-emits-blocked-typeII

Assume

```{math}
K_{\mathrm{ScaleCollapseDrift}}^-
\wedge
K_{\mathrm{ScaleCollapseCostBridge}}^+.
```

Then

```{math}
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```

::::

:::{prf:proof}
By Definition {prf:ref}`def-c17-scale-collapse-renormalization-cost`,
\(K_{\mathrm{ScaleCollapseDrift}}^-\) is exactly the divergence statement

```{math}
\mathcal C_{\mathrm{sc}}[\tau_0,\infty)=\infty.
```

By \(K_{\mathrm{ScaleCollapseCostBridge}}^+\), divergence of this registered
nonnegative Type II branch cost is accepted by the declared `BarrierTypeII`
evaluator as a blocking condition and emits the framework-level blocked
certificate \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\). \(\square\)
:::

## Promotion consequence

::::{prf:corollary} Scale-collapse branch suppression under NS-valid promotion
:label: cor-c17-scale-collapse-branch-suppression

Assume

```{math}
K_{\mathrm{SC}_\lambda}^-
\wedge
K_{\mathrm{ScaleCollapseDrift}}^-
\wedge
K_{\mathrm{ScaleCollapseCostBridge}}^+
\wedge
K_{\mathrm{NS\text{-}UPTypeII}}^+.
```

Then

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}.
```

::::

:::{prf:proof}
Theorem {prf:ref}`thm-c17-scale-collapse-drift-emits-blocked-typeII` gives
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\). The NS-valid promotion bridge
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\) is exactly the C10 payload licensing

```{math}
K_{\mathrm{SC}_\lambda}^-
\wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}
\Longrightarrow
K_{\mathrm{SC}_\lambda}^{\sim}.
```

Applying that bridge gives the conclusion. \(\square\)
:::

## Absolute scale-cost variant

Some evaluators may prefer a single enlarged Type II cost rather than two
branch costs. C17 also records the equivalent sufficient option.

::::{prf:definition} Absolute scale-drift barrier cost
:label: def-c17-absolute-scale-drift-barrier-cost

Define

```{math}
\mathfrak C_{\mathrm{abs}}(\tau)
:=
\nu\int|\nabla V|^2\phi_{R(\tau)}
+|a(\tau)|M_R(\tau).
```

The certificate

```{math}
K_{\mathrm{AbsScaleCostBridge}}^+
```

means that `BarrierTypeII` accepts \(\mathfrak C_{\mathrm{abs}}\) as an
admissible Type II barrier cost and emits
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) whenever
\(\int_{\tau_0}^{\infty}\mathfrak C_{\mathrm{abs}}=\infty\).

::::

::::{prf:lemma} Scale-collapse drift forces absolute scale-cost divergence
:label: lem-c17-scale-collapse-forces-absolute-cost

If \(K_{\mathrm{ScaleCollapseDrift}}^-\) holds, then

```{math}
\int_{\tau_0}^{\infty}\mathfrak C_{\mathrm{abs}}(\tau)\,d\tau=\infty.
```

::::

:::{prf:proof}
Since

```{math}
\mathfrak C_{\mathrm{abs}}(\tau)
=
\nu\int|\nabla V|^2\phi_{R(\tau)}
+|a(\tau)|M_R(\tau)
\ge
a_-(\tau)M_R(\tau),
```

divergence of \(\int a_-M_R\) forces divergence of
\(\int\mathfrak C_{\mathrm{abs}}\). \(\square\)
:::

::::{prf:corollary} Absolute scale-cost bridge blocks scale-collapse drift
:label: cor-c17-absolute-scale-cost-blocks-scale-collapse

Assume

```{math}
K_{\mathrm{ScaleCollapseDrift}}^-
\wedge
K_{\mathrm{AbsScaleCostBridge}}^+.
```

Then

```{math}
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```

::::

:::{prf:proof}
By Lemma {prf:ref}`lem-c17-scale-collapse-forces-absolute-cost`,
\(\int\mathfrak C_{\mathrm{abs}}=\infty\). The bridge
\(K_{\mathrm{AbsScaleCostBridge}}^+\) registers this divergence as a
`BarrierTypeII` blocking condition and emits
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\). \(\square\)
:::

## Exhaustive moving-cutoff route after C16 and C17

::::{prf:theorem} C15-C17 moving route with scale-collapse fallback
:label: thm-c17-moving-route-with-scale-collapse-fallback

Assume the C15 moving-cutoff setup, the C16 scale-drift dichotomy, and the
declared NS3D Type II route \(K_{\mathrm{SC}_\lambda}^-\). Then the
scale-negative branch has the following exhaustive outcome:

1. If \(K_{\mathrm{ScaleNegL^1}}^+\) holds and the remaining C15 annular
   moving-error certificates hold, then the moving-cutoff monotonicity route is
   available for the formal `UP-TypeII` checklist.
2. If \(K_{\mathrm{ScaleCollapseDrift}}^-\) holds and either
   \(K_{\mathrm{ScaleCollapseCostBridge}}^+\) or
   \(K_{\mathrm{AbsScaleCostBridge}}^+\) holds, then
   \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) is emitted directly.
3. If the second alternative also has \(K_{\mathrm{NS\text{-}UPTypeII}}^+\),
   then \(K_{\mathrm{SC}_\lambda}^{\sim}\) is emitted.

::::

:::{prf:proof}
The first branch is C15 after the C16 positive certificate supplies
\(K_{\mathrm{ScaleNegL^1}}^+\). The second branch is Theorem
{prf:ref}`thm-c17-scale-collapse-drift-emits-blocked-typeII` or Corollary
{prf:ref}`cor-c17-absolute-scale-cost-blocks-scale-collapse`, depending on
which cost bridge is registered. The third branch is Corollary
{prf:ref}`cor-c17-scale-collapse-branch-suppression`. \(\square\)
:::

## What remains after C17

C17 does not prove that the default C4 identity cost already includes
\(\mathfrak C_{\mathrm{sc}}\). It adds the precise extra backend payload needed
to use scale-collapse drift as a barrier:

```{math}
K_{\mathrm{ScaleCollapseCostBridge}}^+
\quad\text{or}\quad
K_{\mathrm{AbsScaleCostBridge}}^+.
```

Once one of these is registered, the C16 obstruction is no longer merely a
failure of the C15 monotonicity route. It is itself a barrier-blocking branch.
Without one of these bridges, the rigorous classification must still list
\(K_{\mathrm{ScaleCollapseDrift}}^-\) as a residual Type II class.
