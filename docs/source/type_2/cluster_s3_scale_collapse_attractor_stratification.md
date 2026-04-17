# S3 scale-collapse generalized self-similar reduction

This note implements the S3 stratification route for the scale-collapse drift
survivor isolated by C16.

C16 proves that genuine repaired-gauge scale collapse with a nonvanishing moving
localized core forces the obstruction

```{math}
K_{\mathrm{ScaleCollapseDrift}}^-:
\qquad
\int_{\tau_0}^{\infty}a_-(\tau)M_R(\tau)\,d\tau=\infty,
\qquad
M_R(\tau)=\int |V(y,\tau)|^2\phi_{R(\tau)}(y)\,dy.
```

C17 handles this obstruction by cost registration: register either
\(a_-M_R\) or an absolute scale-drift cost as an admissible Type II barrier
cost. S3 is the PDE alternative. It asks what the obstruction forces inside
the repaired-gauge state space.

The honest output is a reduction theorem, not an unconditional closure. Under
compactness and modulation-limit payloads, S3 produces a nonzero autonomous
reduced Navier-Stokes limit. If that autonomous limit lies in a rigidity class
covered by a self-similar or stationary Liouville theorem, S3 blocks the
scale-collapse survivor. Otherwise the remaining obstruction is a precise generalized
self-similar or stationary Liouville problem.

## Scale-collapse stratum

::::{prf:definition} S3 scale-collapse stratum
:label: def-s3-scale-collapse-stratum

A represented repaired-gauge Type II branch is in the S3 scale-collapse stratum
when it satisfies

```{math}
K_{\mathrm{S3ScaleCollapse}}^+
:=
K_{\mathrm{RepBridge}}^+
\wedge
K_{L^3\mathrm{Norm}}^+
\wedge
K_{\mathrm{CoreL^2Floor}}^+
\wedge
K_{\mathrm{ModBd}}^+
\wedge
K_{\mathrm{WinH1}}^+
\wedge
\{\lambda(\tau)\to0\}
\wedge
K_{\mathrm{ScaleCollapseDrift}}^-.
```

Here \(K_{\mathrm{ModBd}}^+\) means the repaired modulation parameters are
bounded on the branch, and \(K_{\mathrm{WinH1}}^+\) supplies the local
Caccioppoli compactness needed to pass nonlinear terms on bounded cylinders.
The C16 core floor supplies constants \(m_0>0\), \(\tau_1\), and moving cutoffs
such that

```{math}
M_R(\tau)\ge m_0
\quad\text{for a.e. }\tau\ge\tau_1.
```

::::

The stratum is the residual scale-drift class: it is not the radiative bucket
and not the rough-core bucket. It appears when the moving-cutoff monotonicity
route cannot absorb the negative scale term.

## Window and modulation payloads

The divergence in \(K_{\mathrm{ScaleCollapseDrift}}^-\) gives infinitely much
weighted negative drift, but it does not by itself select a fixed-length window
with a uniform negative average. A bounded nonnegative function can have
infinite integral while its unit-window averages tend to zero. S3 therefore
separates the true consequence of C16 from the extra thickness needed for a
self-similar reduction.

::::{prf:definition} S3 thick negative-drift windows
:label: def-s3-thick-negative-windows

The certificate \(K_{\mathrm{S3ThickDrift}}^+(T,c;\tau_n)\) means that there are
\(T>0\), \(c>0\), and a fixed window sequence \(\tau_n\to\infty\) such that

```{math}
\frac1T\int_{\tau_n}^{\tau_n+T} a_-(\tau)M_R(\tau)\,d\tau\ge c
\qquad\text{for every }n.
```

If \(M_R\) is bounded above by \(M_1\) on these windows, then

```{math}
\frac1T\int_{\tau_n}^{\tau_n+T} a_-(\tau)\,d\tau\ge c/M_1.
```

Thus thick weighted drift forces genuinely negative modulation on the selected
windows. Without \(K_{\mathrm{S3ThickDrift}}^+\), S3 records the branch as a
thin-drift defect rather than claiming self-similar asymptotics.

::::

::::{prf:definition} S3 autonomous modulation limit
:label: def-s3-autonomous-modulation-limit

The certificate

```{math}
K_{\mathrm{S3ModLim}}^+(a_\infty,b_\infty;T,\tau_n)
```

means that on the fixed window sequence \([\tau_n,\tau_n+T]\), after translation,

```{math}
a(\tau_n+\cdot)\to a_\infty,
\qquad
b(\tau_n+\cdot)\to b_\infty
```

in a coefficient topology strong enough to pass the products
\(a(V+y\cdot\nabla V)\) and \(b\cdot\nabla V\) to the limit in distributions.
Strong \(L^1(0,T)\) convergence is sufficient; mere convergence of averages is
not sufficient unless a separate compensated-convergence argument is supplied.

The parameter \(a_\infty\) may be negative, zero, or an exceptional value
allowed by the bounded modulation range. The case \(a_\infty<0\) is a
generalized self-similar regime. The case \(a_\infty=0\) is a stationary
Navier-Stokes regime, possibly with translation drift \(b_\infty\).

::::

::::{prf:lemma} Thick autonomous drift forces a negative scale parameter
:label: lem-s3-thick-drift-negative-parameter

Assume \(K_{\mathrm{S3ThickDrift}}^+(T,c;\tau_n)\) on windows
\([\tau_n,\tau_n+T]\), assume \(M_R\le M_1\) on those windows, and assume
\(a(\tau_n+\cdot)\to a_\infty\) in \(L^1(0,T)\). Then

```{math}
a_\infty\le -c/M_1<0.
```

::::

:::{prf:proof}
The upper mass bound and thick weighted drift give

```{math}
\frac1T\int_{\tau_n}^{\tau_n+T}a_-(\tau)\,d\tau\ge c/M_1.
```

After translation and \(L^1\) convergence of \(a\) to the constant
\(a_\infty\), the left side converges to \((-a_\infty)_+\). Hence
\((-a_\infty)_+\ge c/M_1\), so \(a_\infty\le -c/M_1<0\). \(\square\)
:::

::::{prf:definition} S3 compactness payload
:label: def-s3-compactness-payload

The certificate \(K_{\mathrm{S3Compact}}^+(T,\tau_n)\) means that on the same
fixed window sequence, for every bounded ball \(B_R\), the translated sequence

```{math}
V_n(y,s)=V(y,\tau_n+s),\qquad 0<s<T,
```

is precompact strongly in \(L^2(0,T;L^q(B_R))\) for every \(q<6\), weakly in
\(L^2(0,T;H^1(B_R))\), and weak-* in \(L^\infty(0,T;L^3(\mathbb R^3))\). The
pressures are normalized modulo constants and converge in a topology sufficient
to pass

```{math}
-\Delta P=\partial_i\partial_j(V_iV_j)
```

to the limit locally.

This is the Aubin--Lions/Caccioppoli payload supplied by the local regularity
layer, not a consequence of \(K_{L^3\mathrm{Norm}}^+\) alone. In every S3
composition below, \(K_{\mathrm{S3Compact}}^+\), \(K_{\mathrm{S3ModLim}}^+\),
and any optional \(K_{\mathrm{S3ThickDrift}}^+\) are required on the same
window sequence \(\tau_n\).

::::

## Autonomous reduction

::::{prf:theorem} S3 autonomous reduced-limit theorem
:label: thm-s3-autonomous-reduced-limit

Assume

```{math}
K_{\mathrm{S3ScaleCollapse}}^+
\wedge
K_{\mathrm{S3Compact}}^+(T,\tau_n)
\wedge
K_{\mathrm{S3ModLim}}^+(a_\infty,b_\infty;T,\tau_n).
```

Then, after passing to a subsequence, the translated branch converges to a
nonzero weak solution \((U,\Pi)\) on
\(\mathbb R^3\times(0,T)\) of the autonomous reduced equation

```{math}
\partial_s U+(U\cdot\nabla)U+\nabla\Pi
=
\nu\Delta U+a_\infty(U+y\cdot\nabla U)+b_\infty\cdot\nabla U,
\qquad
\nabla\cdot U=0.
```

Moreover \(U\in L^\infty(0,T;L^3(\mathbb R^3))\), the pressure satisfies the
Navier-Stokes pressure reconstruction, and the localized core floor passes to
the limit on a retained compact subcutoff. In particular, \(U\not\equiv0\).

::::

:::{prf:proof}
The weak-* \(L^\infty L^3\) bound follows from \(K_{L^3\mathrm{Norm}}^+\).
The local \(L^2H^1\) bound and time-derivative compactness are part of
\(K_{\mathrm{S3Compact}}^+(T,\tau_n)\), so Aubin--Lions gives strong local convergence
in \(L^2L^q\), \(q<6\). This strong convergence and weak \(L^2H^1\) convergence
pass the nonlinear term to \((U\cdot\nabla)U\) in distributions. The pressure
payload passes the Poisson reconstruction to \(\Pi\).

The modulation-limit payload passes the coefficient terms to
\(a_\infty(U+y\cdot\nabla U)\) and \(b_\infty\cdot\nabla U\). Therefore the
limit solves the autonomous reduced equation.

Finally, \(K_{\mathrm{CoreL^2Floor}}^+\) gives a positive localized \(L^2\)
floor on the selected branch. The compactness payload includes convergence
strong enough for this localized pairing, or for a compact subcutoff below it,
to pass to the limit. Hence \(U\not\equiv0\). \(\square\)
:::

## From autonomous limits to stationary profiles

Theorem {prf:ref}`thm-s3-autonomous-reduced-limit` gives an autonomous
solution on translated windows. It does not automatically give a stationary
profile. The additional step is another genuine payload.

::::{prf:definition} S3 stationary omega-limit payload
:label: def-s3-stationary-omega-payload

The certificate

```{math}
K_{\mathrm{S3StatLim}}^+(a_\infty,b_\infty)
```

means that the autonomous limit supplied by Theorem
{prf:ref}`thm-s3-autonomous-reduced-limit` has a nonzero stationary omega-limit
\((W,Q)\) satisfying

```{math}
(W\cdot\nabla)W+\nabla Q
=
\nu\Delta W+a_\infty(W+y\cdot\nabla W)+b_\infty\cdot\nabla W,
\qquad
\nabla\cdot W=0,
```

with \(W\in L^3(\mathbb R^3)\) and with the admissibility hypotheses required
by the rigidity theorem to be invoked.

::::

This payload is necessary because autonomous bounded trajectories need not be
stationary merely by time translation. A Lyapunov or asymptotic-compactness
argument is required to produce \(W\).

## Rigidity and Liouville routing

::::{prf:definition} S3 rigidity-route payload
:label: def-s3-rigidity-route-payload

For a parameter pair \((a_\infty,b_\infty)\), the certificate

```{math}
K_{\mathrm{S3Rig}}^+(a_\infty,b_\infty)
```

means that every admissible stationary solution of

```{math}
(W\cdot\nabla)W+\nabla Q
=
\nu\Delta W+a_\infty(W+y\cdot\nabla W)+b_\infty\cdot\nabla W,
\qquad
W\in L^3(\mathbb R^3),
```

is zero, or at least is not a terminal Type II blowup branch in the declared
backend.

For the classical backward self-similar normalization, after rescaling and
removing harmless translations, this is the role played by the
Nečas--Růžička--Šverák rigidity theorem. For forward self-similar solution
classes, the relevant input is not nonexistence; known forward self-similar
solutions may exist. The needed certificate is the narrower statement that the
admissible forward profile is not a terminal Type II blowup branch.

For parameter values not covered by an available rigidity theorem, S3 emits a
Liouville defect rather than a contradiction.

::::

::::{prf:theorem} S3 conditional scale-collapse blocker
:label: thm-s3-conditional-scale-collapse-blocker

Assume

```{math}
K_{\mathrm{S3ScaleCollapse}}^+
\wedge
K_{\mathrm{S3Compact}}^+(T,\tau_n)
\wedge
K_{\mathrm{S3ModLim}}^+(a_\infty,b_\infty;T,\tau_n)
\wedge
K_{\mathrm{S3StatLim}}^+(a_\infty,b_\infty)
\wedge
K_{\mathrm{S3Rig}}^+(a_\infty,b_\infty).
```

Then the branch is not an unresolved scale-collapse Type II survivor. Equivalently,
S3 emits

```{math}
K_{\mathrm{ScaleCollapseBlk}}^+.
```

::::

:::{prf:proof}
The autonomous reduced-limit theorem gives a nonzero autonomous limit. The
stationary omega-limit payload converts it into a nonzero stationary profile
\((W,Q)\) in the admissible class for the parameter pair
\((a_\infty,b_\infty)\). The rigidity-route payload says that no such nonzero
profile can represent an unresolved terminal Type II survivor. This contradicts
the assumption that the original branch remains in the S3 survivor class.
Therefore the scale-collapse branch is blocked. \(\square\)
:::

## Promotion consequence

::::{prf:corollary} S3 scale-collapse suppression under NS-valid promotion
:label: cor-s3-scale-collapse-suppression

Assume the hypotheses of Theorem
{prf:ref}`thm-s3-conditional-scale-collapse-blocker`, assume the branch is on
the Type II scale route \(K_{\mathrm{SC}_\lambda}^-\), and assume the declared
backend accepts the S3 blocker as a Type II branch-exclusion certificate. If
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\) is present, then the branch emits

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}.
```

::::

:::{prf:proof}
Theorem {prf:ref}`thm-s3-conditional-scale-collapse-blocker` gives the blocked
Type II outcome for the scale-collapse branch. The C10 promotion bridge then
licenses

```{math}
K_{\mathrm{SC}_\lambda}^-
\wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}
\Longrightarrow
K_{\mathrm{SC}_\lambda}^{\sim}.
```

Thus \(K_{\mathrm{SC}_\lambda}^{\sim}\) is emitted. \(\square\)
:::

## What C16 alone gives, and what it does not give

The C16 obstruction implies

```{math}
\int_{\tau_0}^{\infty} a_-(\tau)\,d\tau=\infty
```

only when \(M_R\) is bounded above on the same moving cutoffs, since then
\(a_-M_R\le M_1a_-\). The local mass floor is used for nontriviality of
limits, not for this unweighted-drift implication. Thus C16 proves sustained
weighted contraction, but by itself it does not prove that
\(a(\tau)\) converges to a negative constant, and it does not prove uniformly
negative fixed-window averages. Therefore C16 alone does not place the branch
inside the classical self-similar regime.

The correct S3 routing is:

1. **Thick autonomous drift.** If
   \(K_{\mathrm{S3ThickDrift}}^+\), \(K_{\mathrm{S3ModLim}}^+\),
   \(K_{\mathrm{S3Compact}}^+\), and \(K_{\mathrm{S3StatLim}}^+\) are supplied,
   the branch reduces to a generalized self-similar stationary profile. It is
   blocked exactly in the parameter ranges covered by
   \(K_{\mathrm{S3Rig}}^+\).
2. **Thin or oscillatory drift.** If the infinite drift is accumulated through
   vanishing averages or oscillatory modulation, the self-similar rigidity
   route is not licensed. This is the defect
   \(K_{\mathrm{S3ThinDrift}}^-\) or \(K_{\mathrm{S3ModLim}}^-\), not a closed
   contradiction.
3. **Degenerate autonomous drift.** If an autonomous limit has
   \(a_\infty=0\), the stationary equation becomes the stationary NS3D equation
   with possible constant advection. Its triviality in the general
   \(L^3(\mathbb R^3)\) class is a stationary Liouville problem, so S3 records
   \(K_{\mathrm{StationaryL^3Liouville}}^-\) unless an appropriate Liouville
   theorem is supplied.

## Relation to C16 and C17

C16 proves that genuine scale collapse with a nonvanishing localized core
forces \(K_{\mathrm{ScaleCollapseDrift}}^-\). C17 handles that obstruction by
cost registration:

```{math}
K_{\mathrm{ScaleCollapseDrift}}^-
\wedge
K_{\mathrm{ScaleCollapseCostBridge}}^+
\Longrightarrow
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```

S3 handles the same obstruction by PDE reduction:

```{math}
K_{\mathrm{ScaleCollapseDrift}}^-
\rightsquigarrow
\text{autonomous generalized self-similar or stationary Liouville problem}.
```

It closes only after the compactness, autonomous-modulation, stationary-limit,
and rigidity payloads for the selected parameter regime are supplied.

## Remaining S3 defects

The ordered defects for the S3 route are:

```{math}
K_{\mathrm{S3ThinDrift}}^-,
\qquad
K_{\mathrm{S3Compact}}^-,
\qquad
K_{\mathrm{S3ModLim}}^-,
\qquad
K_{\mathrm{S3StatLim}}^-,
\qquad
K_{\mathrm{S3Rig}}^-(a_\infty,b_\infty).
```

The last defect includes the classical self-similar rigidity-class verification,
generalized self-similar Liouville problems for \(a_\infty<0\) outside the
covered range, and the stationary \(L^3\)-Liouville problem when
\(a_\infty=0\).

## References for the rigidity payload

The backward self-similar rigidity input is modeled on the
Nečas--Růžička--Šverák theorem excluding nontrivial Leray self-similar
Navier-Stokes profiles in the classical admissible class. The forward
self-similar input must be stated as a non-Type-II classification result, not
as a nonexistence theorem, because forward self-similar Navier-Stokes solutions
may exist in appropriate large-data settings. Generalized parameter values and
the stationary \(L^3\) regime remain explicit Liouville-type payloads until the
needed theorem is supplied.
