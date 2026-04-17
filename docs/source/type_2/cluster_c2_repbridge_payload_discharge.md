# C2.R representation-payload discharge for \(K_{\mathrm{RepBridge}}^+\)

This note refines C2 inside the declared Navier-Stokes Type II barrier backend.
In this setting the candidate is already on the declared Type II barrier route,

```{math}
K_{\mathrm{TypeIIRoute}}^+(\omega),
```

and the backend contract includes the repaired-gauge representation convention.
The purpose of this note is to verify that no extra pressure or modulation
assumption is hidden in that convention: once the raw chart and repaired gauge
solve are present, the pressure representation and final modulation
coefficients are forced.

The original C2 note proves that four high-level PDE payloads imply

```{math}
K_{\mathrm{RepBridge}}^+.
```

This document removes unnecessary payloads from that row.  Once a raw chart is
available and the repaired gauge can be solved along it, the pressure
representation and the final modulation coefficients are not independent
assumptions.  They are forced by Navier-Stokes, the chart identity, and the
Navier-Stokes scale/translation symmetry.

There are two statements below.  First, the analytic identity theorem proves

```{math}
K_{\mathrm{Chart},NS3D}^+
\wedge
K_{\mathrm{GaugeSolve},NS3D}^+
\Longrightarrow
K_{\mathrm{RepBridge}}^+.
```

Second, the declared-backend theorem uses the backend representation contract
to supply those two inputs on every routed candidate.  Therefore, inside the
declared Type II barrier backend, the nonconditional output is

```{math}
K_{\mathrm{TypeIIRoute}}^+(\omega)
\Longrightarrow
K_{\mathrm{RepBridge}}^+(\omega).
```

Raw chart extraction failure and repaired-gauge solve failure are retained only
as outside-contract diagnostics.  They are not live survivor classes once the
declared backend contract is in force.  Pressure-pullback and
modulation-coefficient failures are not independent diagnostics at all: if the
chart and AC gauge solve exist, those rows are theorems.

## Output to be discharged

The compact Type II stack uses a repaired-gauge renormalized Navier-Stokes
orbit

```{math}
\mathsf{RepOrb}_{NS}
=
(V,P,a,b,G_{\mathrm{sc}},G_1,G_2,G_3,\tau_0)
```

satisfying, distributionally on \(\mathbb R^3\times(\tau_0,\infty)\),

```{math}
\partial_\tau V+(V\cdot\nabla)V+\nabla P
=
\nu\Delta V
+a(\tau)(V+Y\cdot\nabla V)
+b(\tau)\cdot\nabla V,
\qquad
\nabla\cdot V=0,
```

with pressure normalized by

```{math}
-\Delta P=\partial_i\partial_j(V_iV_j)
```

modulo functions of \(\tau\).  The repaired gauges are

```{math}
G_{\mathrm{sc}}(V)
=
\int_{\mathbb R^3}|Y|^{-p}|V(Y)|^3\,dY-\Theta_0=0,
\qquad 0<p<3,
```

and

```{math}
G_j(V)
=
\int_{\mathbb R^3}Y_j|V(Y)|^2\psi_R(Y)\,dY=0,
\qquad j=1,2,3.
```

Emitting this object with the stated equation and gauges is exactly
\(K_{\mathrm{RepBridge}}^+\).

## Minimal discharge payload

::::{prf:definition} Minimal representation-discharge payload
:label: def-repbridge-minimal-discharge-payload

For a routed Navier-Stokes Type II branch \(\omega\), the **minimal
representation-discharge payload** is

```{math}
K_{\mathrm{RepDischarge},NS3D}^+(\omega)
:=
K_{\mathrm{Chart},NS3D}^+(\omega)
\wedge
K_{\mathrm{GaugeSolve},NS3D}^+(\omega).
```

The two factors have the following meanings.

1. \(K_{\mathrm{Chart},NS3D}^+\) supplies an absolutely continuous
   concentration chart
   \[
   (u,p,T^*,x_c,\lambda,\rho,U,Q,\mathcal I)
   \]
   with \(\mathcal I=(t_0,T^*)\), \(\lambda(t)>0\),
   \(\lambda(t)\to0\), \(\rho_t=\lambda^{-2}\), and
   \[
   u(x,t)=\lambda(t)^{-1}U(y,\rho(t)),
   \qquad
   p(x,t)=\lambda(t)^{-2}Q(y,\rho(t))+c(t),
   \qquad
   y=\frac{x-x_c(t)}{\lambda(t)}.
   \]
   The regularity is sufficient for the distributional chain rule on compact
   \(y\)-sets and for the pressure equation in distributions.
2. \(K_{\mathrm{GaugeSolve},NS3D}^+\) supplies an absolutely continuous
   repaired-gauge correction.  It consists of a final renormalized time
   \(\tau\), an absolutely continuous raw-time map \(\rho=\rho(\tau)\), and
   gauge parameters \(\mu(\tau)>0\), \(q(\tau)\in\mathbb R^3\), with
   \[
   \rho_\tau=\mu(\tau)^2,
   \]
   and
   \[
   V(Y,\tau)=\mu(\tau)U(\mu(\tau)Y+q(\tau),\rho(\tau)),
   \qquad
   P(Y,\tau)=\mu(\tau)^2Q(\mu(\tau)Y+q(\tau),\rho(\tau)).
   \]
   The condition \(\rho_\tau=\mu^2\) is part of the payload; it preserves the
   viscosity coefficient \(\nu\) after a time-dependent scale correction.  The
   final profile lies on the repaired gauge surface:
   \[
   G_{\mathrm{sc}}(V(\tau))=0,
   \qquad
   G_j(V(\tau))=0,
   \quad j=1,2,3.
   \]
   The payload also includes admissibility \(V(\tau)\in\mathcal A_p\), the
   repaired scale-row transversality, and the centering-row nondegeneracy
   needed for the AC gauge solve.

::::

The boundedness certificate \(K_{\mathrm{ModBd}}^+\) used later by T5 and the
compact barrier is not part of \(K_{\mathrm{RepBridge}}^+\).  C2.R only proves
that the coefficients exist and are the correct coefficients in the repaired
renormalized equation.  Uniform bounds are a subsequent theorem/payload.

## Raw chart consequences

::::{prf:lemma} Chart payload emits the raw orbit
:label: lem-c2r-chart-emits-raw-orbit

Assume \(K_{\mathrm{Chart},NS3D}^+\).  Then the branch emits the C2 raw orbit
payload \(K_{\mathrm{RawOrb}}^+\), and the raw variables \(U,Q\) satisfy

```{math}
\partial_\rho U+(U\cdot\nabla)U+\nabla Q
=
\nu\Delta U
+A(\rho)(U+y\cdot\nabla U)
+B(\rho)\cdot\nabla U,
\qquad
\nabla\cdot U=0,
```

where

```{math}
A(\rho(t))=\lambda(t)\lambda_t(t),
\qquad
B(\rho(t))=\lambda(t)x_c'(t).
```

::::

:::{prf:proof}
The chart data are exactly the C2 raw orbit data, with raw renormalized time
\(\rho\).  The additive pressure function \(c(t)\) has zero spatial gradient.
The chain-rule computation gives the displayed equation: spatial derivatives
scale as \(\nabla_x u=\lambda^{-2}\nabla_yU\),
\(\Delta_xu=\lambda^{-3}\Delta_yU\), the convection term scales as
\(\lambda^{-3}(U\cdot\nabla)U\), and \(\rho_t=\lambda^{-2}\).  Moving the
terms generated by \(\lambda_t\) and \(x_c'\) to the right side gives
\(A=\lambda\lambda_t\) and \(B=\lambda x_c'\).  Incompressibility pulls back as
\(\nabla_x\cdot u=\lambda^{-2}\nabla_y\cdot U\).  \(\square\)
:::

::::{prf:lemma} Pressure pullback is automatic
:label: lem-c2r-pressure-pullback-automatic

Assume \(K_{\mathrm{Chart},NS3D}^+\).  Then

```{math}
-\Delta Q=\partial_i\partial_j(U_iU_j)
```

in distributions, modulo functions of \(\rho\).  After any admissible gauge
solve, the transformed pressure satisfies

```{math}
-\Delta P=\partial_i\partial_j(V_iV_j)
```

modulo functions of \(\tau\).  Hence \(K_{\mathrm{PressureRep}}^+\) is a
theorem from chart plus gauge solve, not an independent payload.

::::

:::{prf:proof}
Taking divergence of the physical Navier-Stokes equation gives
\(-\Delta p=\partial_i\partial_j(u_iu_j)\), with pressure determined up to a
function of time.  Pulling this identity back through
\(p=\lambda^{-2}Q+c(t)\), \(u=\lambda^{-1}U\), and
\(y=(x-x_c)/\lambda\) gives the raw pressure equation for \(Q\).  If
\(V(Y)=\mu U(\mu Y+q)\) and \(P(Y)=\mu^2Q(\mu Y+q)\), then
\[
-\Delta_Y P
=\mu^4(-\Delta Q)(\mu Y+q)
=\mu^4\partial_i\partial_j(U_iU_j)(\mu Y+q)
=\partial_i\partial_j(V_iV_j).
\]
Adding a function of \(\rho\) or \(\tau\) to the pressure does not affect the
identity.  \(\square\)
:::

## Gauge solve consequences

::::{prf:lemma} Gauge solve emits the repaired-gauge payload
:label: lem-c2r-gauge-solve-emits-gauge-real

Assume \(K_{\mathrm{GaugeSolve},NS3D}^+\).  Then the branch emits
\(K_{\mathrm{GaugeReal}}^+\).

::::

:::{prf:proof}
The payload supplies the repaired scale functional \(G_{\mathrm{sc}}\), the
centering functionals \(G_j\), the admissible class \(\mathcal A_p\), and an
absolutely continuous symmetry correction placing the profile on
\[
G_{\mathrm{sc}}(V(\tau))=G_1(V(\tau))=G_2(V(\tau))=G_3(V(\tau))=0.
\]
It also supplies the transversality and nondegeneracy hypotheses used in
[required_new_scale_gauge_theorems.md](required_new_scale_gauge_theorems.md)
and [cluster_next_easiest_theorems.md](cluster_next_easiest_theorems.md).
These are exactly the C2 repaired-gauge realization requirements. \(\square\)
:::

::::{prf:lemma} Final modulation coefficients are forced
:label: lem-c2r-modulation-coefficients-forced

Assume \(K_{\mathrm{Chart},NS3D}^+\) and
\(K_{\mathrm{GaugeSolve},NS3D}^+\).  Then the final profile \(V,P\) satisfies

```{math}
\partial_\tau V+(V\cdot\nabla)V+\nabla P
=
\nu\Delta V
+a(\tau)(V+Y\cdot\nabla V)
+b(\tau)\cdot\nabla V,
```

where the final modulation coefficients are forced by the raw coefficients
\(A,B\) and the AC gauge velocities:

```{math}
a(\tau)=\mu(\tau)^2A(\rho(\tau))+\frac{\mu_\tau(\tau)}{\mu(\tau)},
```

and

```{math}
b(\tau)=\mu(\tau)B(\rho(\tau))
      +\mu(\tau)A(\rho(\tau))q(\tau)
      +\frac{q_\tau(\tau)}{\mu(\tau)}.
```

Consequently \(K_{\mathrm{ModParams}}^+\) is a theorem from chart plus AC
gauge solve, not an independent payload.

::::

:::{prf:proof}
Let
\[
z=\mu(\tau)Y+q(\tau),
\qquad
V(Y,\tau)=\mu(\tau)U(z,\rho(\tau)),
\qquad
\rho_\tau=\mu^2.
\]
The raw equation in \(\rho,z\) is
\[
U_\rho+(U\cdot\nabla_z)U+\nabla_zQ
=\nu\Delta_zU+A(U+z\cdot\nabla_zU)+B\cdot\nabla_zU.
\]
Using \(\nabla_YV=\mu^2\nabla_zU\), \(\Delta_YV=\mu^3\Delta_zU\),
\((V\cdot\nabla_Y)V=\mu^3(U\cdot\nabla_z)U\), and
\(\nabla_YP=\mu^3\nabla_zQ\), while differentiating in \(\tau\), gives
\[
V_\tau+(V\cdot\nabla_Y)V+\nabla_YP-\nu\Delta_YV
=
\left(\frac{\mu_\tau}{\mu}+\mu^2A\right)(V+Y\cdot\nabla_YV)
+
\left(\mu B+\mu Aq+\frac{q_\tau}{\mu}\right)\cdot\nabla_YV.
\]
This is the repaired-gauge equation with the displayed coefficients.  Since
\(\mu,q,\rho\) are absolutely continuous and \(A,B\) are the raw chart
coefficients, \(a,b\) are measurable and locally integrable on every finite
renormalized window.  \(\square\)
:::

## Main discharge theorem

::::{prf:theorem} C2.R repaired-gauge representation discharge
:label: thm-c2r-repbridge-discharge

Let \(\omega\) be a routed NS3D Type II candidate.  If

```{math}
K_{\mathrm{TypeIIRoute}}^+(\omega)
\wedge
K_{\mathrm{Chart},NS3D}^+(\omega)
\wedge
K_{\mathrm{GaugeSolve},NS3D}^+(\omega),
```

then \(\omega\) emits

```{math}
K_{\mathrm{RepBridge}}^+(\omega).
```

::::

:::{prf:proof}
Lemma {prf:ref}`lem-c2r-chart-emits-raw-orbit` gives
\(K_{\mathrm{RawOrb}}^+\).  Lemma
{prf:ref}`lem-c2r-gauge-solve-emits-gauge-real` gives
\(K_{\mathrm{GaugeReal}}^+\).  Lemma
{prf:ref}`lem-c2r-pressure-pullback-automatic` gives
\(K_{\mathrm{PressureRep}}^+\).  Lemma
{prf:ref}`lem-c2r-modulation-coefficients-forced` gives
\(K_{\mathrm{ModParams}}^+\).  These are exactly the four payloads required by
C2.  Applying Theorem {prf:ref}`thm-c2-representation-bridge` gives
\(K_{\mathrm{RepBridge}}^+(\omega)\).  \(\square\)
:::

## Declared-backend nonconditional discharge

::::{prf:theorem} C2.R declared-backend representation discharge
:label: thm-c2r-declared-backend-nonconditional-discharge

Work inside the declared NS3D repaired-gauge Type II backend, i.e. assume
\(K_{\mathrm{NS3DTypeIIBackend}}^+\).  For every routed candidate \(\omega\),

```{math}
K_{\mathrm{TypeIIRoute}}^+(\omega)
\Longrightarrow
K_{\mathrm{RepBridge}}^+(\omega).
```

::::

:::{prf:proof}
By the backend representation contract in
[ns3d_repaired_gauge_backend_contract.md](ns3d_repaired_gauge_backend_contract.md),
every routed candidate supplies the repaired-gauge representation payload.  In
the refined C2.R language, this means the candidate has an AC raw chart
\(K_{\mathrm{Chart},NS3D}^+\) and an admissible AC repaired-gauge solve
\(K_{\mathrm{GaugeSolve},NS3D}^+\).  Theorem
{prf:ref}`thm-c2r-repbridge-discharge` then emits
\(K_{\mathrm{RepBridge}}^+(\omega)\).  \(\square\)
:::

## Ordered representation defects

::::{prf:definition} C2.R representation defects
:label: def-c2r-real-representation-defects

Outside the declared representation backend, the ordered representation
diagnostics are:

1. **Raw chart extraction defect**
   \[
   K_{\mathrm{Chart}}^-:
   \quad
   K_{\mathrm{TypeIIRoute}}^+
   \text{ does not supply an AC concentration chart }K_{\mathrm{Chart},NS3D}^+.
   \]
2. **Repaired-gauge solve defect**
   \[
   K_{\mathrm{GaugeSolve}}^-:
   \quad
   K_{\mathrm{Chart},NS3D}^+
   \text{ is present, but the repaired scale/centering gauge cannot be solved
   admissibly and absolutely continuously.}
   \]

The former is a profile/backend extraction defect.  The latter is a gauge
admissibility, transversality, or AC modulation defect.  Inside the declared
NS3D Type II barrier backend these diagnostics are discharged by Theorem
{prf:ref}`thm-c2r-declared-backend-nonconditional-discharge`; they are listed
only to say what would fail outside that backend contract.

::::

::::{prf:corollary} Ordered C2.R classification
:label: cor-c2r-real-ordered-classification

Outside the declared representation backend, every routed NS3D Type II
candidate emits exactly one of

```{math}
K_{\mathrm{RepBridge}}^+,
\qquad
K_{\mathrm{Chart}}^-,
\qquad
K_{\mathrm{GaugeSolve}}^-.
```

The symbols \(K_{\mathrm{PressurePull}}^-\) and
\(K_{\mathrm{ModCoeff}}^-\) are not independent survivor defects in this
refined ledger.  If they appear, they only indicate that the chart or AC gauge
solve was not actually available with the stated regularity.

::::

:::{prf:proof}
If \(K_{\mathrm{Chart},NS3D}^+\) fails, emit \(K_{\mathrm{Chart}}^-\).  If
the chart exists but \(K_{\mathrm{GaugeSolve},NS3D}^+\) fails, emit
\(K_{\mathrm{GaugeSolve}}^-\).  If both are present, Theorem
{prf:ref}`thm-c2r-repbridge-discharge` emits \(K_{\mathrm{RepBridge}}^+\).
The ordered evaluation is disjoint and exhaustive for the representation row.
\(\square\)
:::

## What this removes from the survivor ledger

Inside the declared NS3D Type II barrier backend, the repaired-gauge
representation bridge is fully discharged:

```{math}
K_{\mathrm{RepBridge}}^+
```

is a theorem, not a separate assumption.  Pressure reconstruction and final
modulation parameters are automatic consequences of the chart and the AC gauge
solve.  Thus, under the declared backend contract, representation failure is
not a remaining Type II class.  The Type II classification proceeds directly to
the compact single-core, radiative, rough-core, scale-collapse, and multibubble
strata.
