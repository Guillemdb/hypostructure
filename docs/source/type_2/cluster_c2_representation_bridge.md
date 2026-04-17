# C2 representation bridge: from sieve profiles to repaired-gauge Type II orbits

This note implements C2 in the classification-completeness program. Its purpose
is to turn the abstract sieve/profile output

```{math}
K_{C_\mu}^+\wedge K_{\mathrm{Prof}_{NS}}^+
```

into the concrete PDE object used by the compact Type II barrier:

```{math}
(V,P,a,b)
```

satisfying the repaired-gauge renormalized Navier-Stokes equation.

The bridge is intentionally explicit. We do not claim that a generic
concentration certificate automatically contains a smooth repaired-gauge orbit.
Instead, C2 states the exact payloads needed from the Navier-Stokes profile
backend and proves that those payloads compile into

```{math}
K_{\mathrm{RepBridge}}^+.
```

## Target PDE object

The compact Type II proof stack requires a renormalized profile satisfying

```{math}
\partial_\tau V +(V\cdot\nabla)V+\nabla P
=
\nu\Delta V
+a(\tau)(V+y\cdot\nabla V)
+b(\tau)\cdot\nabla V,
\qquad
\nabla\cdot V=0.
```

It also requires the repaired scale gauge and centering gauges used in
[required_new_scale_gauge_theorems.md](required_new_scale_gauge_theorems.md)
and [cluster_next_easiest_theorems.md](cluster_next_easiest_theorems.md):

```{math}
G_{\mathrm{sc}}(V)
=
\int_{\mathbb R^3}|y|^{-p}|V(y)|^3\,dy-\Theta_0
=0,
\qquad
0<p<3,
```

and

```{math}
G_j(V)
=
\int_{\mathbb R^3}y_j|V(y)|^2\psi_R(y)\,dy
=0,
\qquad j=1,2,3.
```

The representation bridge is the certificate that the abstract Type II profile
has been realized in exactly this PDE class.

## Raw Type II orbit payload

::::{prf:definition} Raw Navier-Stokes Type II orbit payload
:label: def-raw-ns-typeII-orbit-payload

A **raw Navier-Stokes Type II orbit payload** is a tuple

```{math}
\mathsf{RawOrb}_{NS}
=
\bigl(
u,p,T^*,x_c,\lambda,\tau,V,P,\mathcal I
\bigr)
```

with the following data.

1. \(u,p\) solve the 3D incompressible Navier-Stokes equations on a terminal
   interval \(\mathcal I=(t_0,T^*)\):
   ```{math}
   \partial_t u+(u\cdot\nabla)u+\nabla p=\nu\Delta u,
   \qquad
   \nabla\cdot u=0.
   ```
2. \(x_c:\mathcal I\to\mathbb R^3\) and
   \(\lambda:\mathcal I\to(0,\infty)\) are absolutely continuous modulation
   functions, with \(\lambda(t)\to0\) as \(t\uparrow T^*\).
3. The renormalized time is absolutely continuous and satisfies
   ```{math}
   \frac{d\tau}{dt}=\lambda(t)^{-2},
   \qquad
   \tau(t)\to\infty
   \quad\text{as }t\uparrow T^*.
   ```
4. The renormalized variables are
   ```{math}
   y=\frac{x-x_c(t)}{\lambda(t)},
   \qquad
   u(x,t)=\lambda(t)^{-1}V(y,\tau(t)),
   \qquad
   p(x,t)=\lambda(t)^{-2}P(y,\tau(t)).
   ```
5. The regularity is sufficient to justify the change of variables and the
   distributional chain rule on compact \(y\)-sets.

The positive certificate that such a payload is supplied by the profile backend
is denoted

```{math}
K_{\mathrm{RawOrb}}^+.
```

::::

## Raw renormalized equation

::::{prf:lemma} Raw orbit gives the renormalized Navier-Stokes equation
:label: lem-raw-orbit-renormalized-equation

Assume \(K_{\mathrm{RawOrb}}^+\). Define

```{math}
a_{\mathrm{raw}}(\tau(t)):=\lambda(t)\lambda_t(t),
\qquad
b_{\mathrm{raw}}(\tau(t)):=\lambda(t)x_c'(t).
```

Then \(V,P\) satisfy, distributionally on compact \(y\)-sets,

```{math}
\partial_\tau V +(V\cdot\nabla)V+\nabla P
=
\nu\Delta V
+a_{\mathrm{raw}}(\tau)(V+y\cdot\nabla V)
+b_{\mathrm{raw}}(\tau)\cdot\nabla V,
\qquad
\nabla\cdot V=0.
```

::::

:::{prf:proof}
Write

```{math}
u(x,t)=\lambda(t)^{-1}V(y,\tau(t)),
\qquad
y=\frac{x-x_c(t)}{\lambda(t)},
\qquad
\tau_t=\lambda^{-2}.
```

The spatial derivatives are

```{math}
\nabla_x u=\lambda^{-2}\nabla_y V,
\qquad
\Delta_x u=\lambda^{-3}\Delta_y V,
\qquad
(u\cdot\nabla_x)u=\lambda^{-3}(V\cdot\nabla_y)V.
```

Since

```{math}
y_t
=
-\frac{x_c'(t)}{\lambda(t)}
-\frac{\lambda_t(t)}{\lambda(t)}y,
```

the time derivative is

```{math}
\partial_t u
=
-\lambda_t\lambda^{-2}V
+\lambda^{-1}\tau_t\partial_\tau V
+\lambda^{-1}y_t\cdot\nabla_y V.
```

Using \(\tau_t=\lambda^{-2}\), this becomes

```{math}
\partial_t u
=
\lambda^{-3}\partial_\tau V
-\lambda_t\lambda^{-2}(V+y\cdot\nabla V)
-\lambda^{-2}x_c'(t)\cdot\nabla V.
```

Also \(\nabla_x p=\lambda^{-3}\nabla_y P\). Substituting these identities into
Navier-Stokes and multiplying by \(\lambda^3\) gives

```{math}
\partial_\tau V
-\lambda\lambda_t(V+y\cdot\nabla V)
-\lambda x_c'(t)\cdot\nabla V
+(V\cdot\nabla)V+\nabla P
=
\nu\Delta V.
```

Moving the modulation terms to the right-hand side gives the claimed equation
with \(a_{\mathrm{raw}}=\lambda\lambda_t\) and
\(b_{\mathrm{raw}}=\lambda x_c'\). Finally,
\(\nabla_x\cdot u=\lambda^{-2}\nabla_y\cdot V\), so incompressibility of \(u\)
implies \(\nabla_y\cdot V=0\). \(\square\)
:::

## Repaired-gauge realization payload

::::{prf:definition} Repaired-gauge realization payload
:label: def-repaired-gauge-realization-payload

A **repaired-gauge realization payload** for a raw orbit is a tuple

```{math}
\mathsf{GaugeReal}_{NS}
=
\bigl(
p,\Theta_0,R,\psi_R,
\mathcal A_p,
G_{\mathrm{sc}},G_1,G_2,G_3,
\mathsf{gauge\_surface},
\mathsf{regularity},
\mathsf{modulation\_update}
\bigr)
```

such that:

1. \(0<p<3\), \(\Theta_0>0\), and the scale gauge is
   ```{math}
   G_{\mathrm{sc}}(V)=
   \int_{\mathbb R^3}|y|^{-p}|V(y)|^3\,dy-\Theta_0.
   ```
2. The centering gauges are
   ```{math}
   G_j(V)=\int_{\mathbb R^3}y_j|V(y)|^2\psi_R(y)\,dy,
   \qquad j=1,2,3.
   ```
3. The realized profile lies on the repaired gauge surface:
   ```{math}
   G_{\mathrm{sc}}(V(\tau))=0,
   \qquad
   G_j(V(\tau))=0,\quad j=1,2,3,
   \qquad
   \tau\ge\tau_0.
   ```
4. \(V(\tau)\in\mathcal A_p\) for the repaired scale gauge and has enough local
   regularity for the gauge derivatives and modulation identities used in
   [required_new_scale_gauge_theorems.md](required_new_scale_gauge_theorems.md)
   and [cluster_next_easiest_theorems.md](cluster_next_easiest_theorems.md).
5. Any time-dependent symmetry adjustment used to impose the gauges has already
   been absorbed into updated modulation parameters \(a(\tau),b(\tau)\), and
   the resulting equation has the standard repaired-gauge form.

The positive certificate for this payload is denoted

```{math}
K_{\mathrm{GaugeReal}}^+.
```

::::

This payload is deliberately not hidden inside notation. It is the place where
the profile backend must certify that the abstract concentration object has
actually been placed on the repaired scale and centering gauge surface.

## Representation bridge certificate

::::{prf:definition} Repaired-gauge representation bridge
:label: def-repbridge-certificate

The certificate

```{math}
K_{\mathrm{RepBridge}}^+
```

is the assertion that the Navier-Stokes Type II candidate has:

1. a raw orbit payload \(K_{\mathrm{RawOrb}}^+\);
2. a repaired-gauge realization payload \(K_{\mathrm{GaugeReal}}^+\);
3. a pressure payload \(K_{\mathrm{PressureRep}}^+\) certifying that \(P\) is
   the pressure associated with \(V\), modulo the local constants used in the
   pressure estimates;
4. a modulation payload \(K_{\mathrm{ModParams}}^+\) certifying that the final
   modulation parameters \(a,b\) are the coefficients appearing in the repaired
   renormalized equation.

The output object is the tuple

```{math}
\mathsf{RepOrb}_{NS}
=
(V,P,a,b,G_{\mathrm{sc}},G_1,G_2,G_3,\tau_0).
```

::::

## Main C2 theorem

::::{prf:theorem} C2 representation bridge
:label: thm-c2-representation-bridge

Assume a Navier-Stokes Type II profile branch supplies:

```{math}
K_{C_\mu}^+,\qquad
K_{\mathrm{Prof}_{NS}}^+,\qquad
K_{\mathrm{RawOrb}}^+,\qquad
K_{\mathrm{GaugeReal}}^+,\qquad
K_{\mathrm{PressureRep}}^+,\qquad
K_{\mathrm{ModParams}}^+.
```

Then the branch emits

```{math}
K_{\mathrm{RepBridge}}^+.
```

In particular, the candidate is represented by a repaired-gauge renormalized
Navier-Stokes orbit \((V,P,a,b)\) satisfying the PDE class required by the
compact Type II barrier.

::::

:::{prf:proof}
The raw orbit payload gives the change of variables and the raw renormalized
profile \(V,P\). By Lemma {prf:ref}`lem-raw-orbit-renormalized-equation`, this
profile satisfies the renormalized Navier-Stokes equation with raw modulation
coefficients.

The repaired-gauge realization payload certifies that the profile has been
placed on the repaired scale and centering gauge surface and that any
time-dependent symmetry adjustment has been absorbed into updated modulation
parameters. The pressure payload certifies that the pressure variable is the
Navier-Stokes pressure associated with \(V\), modulo the constants allowed in
the local pressure theory. The modulation payload identifies the final
coefficients \(a,b\) in the repaired-gauge equation.

Combining these payloads gives exactly the output tuple
\(\mathsf{RepOrb}_{NS}\) in Definition
{prf:ref}`def-repbridge-certificate`. Therefore
\(K_{\mathrm{RepBridge}}^+\) is emitted. \(\square\)
:::

## NS3D representation payload

We now instantiate the C2 payload for the declared NS3D Type II backend. This
does not assert that arbitrary weak concentration data automatically has this
regularity. It fixes the representation contract for the NS3D backend branch
that is meant to feed the repaired-gauge Type II proof.

::::{prf:definition} NS3D repaired-gauge representation payload
:label: def-ns3d-representation-payload

For a declared NS3D Type II candidate \(\omega\) that has entered the C1 route

```{math}
K_{\mathrm{TypeIIRoute}}^+(\omega),
```

the **NS3D repaired-gauge representation payload** is

```{math}
\mathsf{RepPayload}_{NS3D}(\omega)
=
\bigl(
K_{\mathrm{RawOrb},NS3D}^+(\omega),
K_{\mathrm{GaugeReal},NS3D}^+(\omega),
K_{\mathrm{PressureRep},NS3D}^+(\omega),
K_{\mathrm{ModParams},NS3D}^+(\omega)
\bigr),
```

where:

1. \(K_{\mathrm{RawOrb},NS3D}^+(\omega)\) supplies the raw orbit payload of
   Definition {prf:ref}`def-raw-ns-typeII-orbit-payload`;
2. \(K_{\mathrm{GaugeReal},NS3D}^+(\omega)\) supplies the repaired-gauge
   realization payload of Definition
   {prf:ref}`def-repaired-gauge-realization-payload`;
3. \(K_{\mathrm{PressureRep},NS3D}^+(\omega)\) supplies the pressure
   representation payload for the renormalized pressure \(P\);
4. \(K_{\mathrm{ModParams},NS3D}^+(\omega)\) supplies the final modulation
   coefficients \(a,b\) in the repaired-gauge renormalized equation.

The positive certificate that the declared NS3D repaired-gauge Type II backend
uses this representation contract is denoted

```{math}
K_{\mathrm{RepPayload},NS3D}^+.
```

::::

::::{prf:lemma} NS3D backend supplies the repaired-gauge representation payload
:label: lem-ns3d-supplies-reppayload

For every declared NS3D Type II candidate \(\omega\) with
\(K_{\mathrm{TypeIIRoute}}^+(\omega)\), the declared repaired-gauge Type II
backend supplies

```{math}
\mathsf{RepPayload}_{NS3D}(\omega).
```

Equivalently,

```{math}
K_{\mathrm{RepPayload},NS3D}^+
```

is available for the declared NS3D repaired-gauge Type II branch.

::::

:::{prf:proof}
The C1 route supplies the concentration/profile branch

```{math}
K_{C_\mu}^+(\omega)
\wedge
K_{\mathrm{SC}_\lambda}^-(\omega)
\wedge
K_{\mathrm{Prof}_{NS}}^+(\omega).
```

By Definition {prf:ref}`def-ns3d-representation-payload`, the declared NS3D
repaired-gauge Type II backend attaches to that branch the four representation
payloads

```{math}
K_{\mathrm{RawOrb},NS3D}^+(\omega),
\quad
K_{\mathrm{GaugeReal},NS3D}^+(\omega),
\quad
K_{\mathrm{PressureRep},NS3D}^+(\omega),
\quad
K_{\mathrm{ModParams},NS3D}^+(\omega).
```

These are exactly the components of
\(\mathsf{RepPayload}_{NS3D}(\omega)\). Hence
\(K_{\mathrm{RepPayload},NS3D}^+\) is available for the declared backend.
\(\square\)
:::

::::{prf:corollary} NS3D representation bridge
:label: cor-ns3d-representation-bridge

For every declared NS3D Type II candidate \(\omega\) with
\(K_{\mathrm{TypeIIRoute}}^+(\omega)\), the declared repaired-gauge Type II
backend emits

```{math}
K_{\mathrm{RepBridge}}^+(\omega).
```

::::

:::{prf:proof}
By Lemma {prf:ref}`lem-ns3d-supplies-reppayload`, the candidate has the four
payloads \(K_{\mathrm{RawOrb}}^+\), \(K_{\mathrm{GaugeReal}}^+\),
\(K_{\mathrm{PressureRep}}^+\), and \(K_{\mathrm{ModParams}}^+\). Applying
Theorem {prf:ref}`thm-c2-representation-bridge` gives
\(K_{\mathrm{RepBridge}}^+(\omega)\). \(\square\)
:::

## Ordered representation defects

If the bridge cannot be emitted, the failure is classified rather than left
ambiguous.

::::{prf:definition} Representation defect certificates
:label: def-representation-defect-certificates

The ordered representation defects are:

1. **Profile extraction defect**
   ```{math}
   K_{\mathrm{ProfOrb}}^-:
   \quad
   K_{C_\mu}^+\wedge K_{\mathrm{Prof}_{NS}}^+
   \text{ does not supply }K_{\mathrm{RawOrb}}^+.
   ```
2. **Gauge realization defect**
   ```{math}
   K_{\mathrm{GaugeReal}}^-:
   \quad
   K_{\mathrm{RawOrb}}^+\text{ is available, but }K_{\mathrm{GaugeReal}}^+
   \text{ is not.}
   ```
3. **Pressure representation defect**
   ```{math}
   K_{\mathrm{PressureRep}}^-:
   \quad
   K_{\mathrm{PressureRep}}^+\text{ is unavailable.}
   ```
4. **Modulation parameter defect**
   ```{math}
   K_{\mathrm{ModParams}}^-:
   \quad
   K_{\mathrm{ModParams}}^+\text{ is unavailable.}
   ```

The first applicable defect in this order is the emitted representation-failure
certificate.

::::

::::{prf:corollary} Ordered C2 representation classification
:label: cor-c2-representation-classification

For every Navier-Stokes Type II profile branch in the declared backend, exactly
one ordered output certificate is emitted:

1. \(K_{\mathrm{RepBridge}}^+\);
2. \(K_{\mathrm{ProfOrb}}^-\);
3. \(K_{\mathrm{GaugeReal}}^-\);
4. \(K_{\mathrm{PressureRep}}^-\);
5. \(K_{\mathrm{ModParams}}^-\).

::::

:::{prf:proof}
Attempt to verify the C2 payloads in order. If all positive payloads are
present, Theorem {prf:ref}`thm-c2-representation-bridge` emits
\(K_{\mathrm{RepBridge}}^+\). If not, the first missing payload emits the
corresponding defect certificate in Definition
{prf:ref}`def-representation-defect-certificates`. This is exhaustive by typed
certificate evaluation and disjoint as an ordered output classification.
\(\square\)
:::

## What C2 discharges

C2 removes "representation failure" as a vague bucket. It replaces it by the
following precise statement:

```{math}
K_{\mathrm{RepBridge}}^+
\quad\text{or}\quad
K_{\mathrm{ProfOrb}}^-
\quad\text{or}\quad
K_{\mathrm{GaugeReal}}^-
\quad\text{or}\quad
K_{\mathrm{PressureRep}}^-
\quad\text{or}\quad
K_{\mathrm{ModParams}}^-.
```

The bridge is fully positive once the profile backend supplies the four
concrete PDE payloads:

```{math}
K_{\mathrm{RawOrb}}^+,\qquad
K_{\mathrm{GaugeReal}}^+,\qquad
K_{\mathrm{PressureRep}}^+,\qquad
K_{\mathrm{ModParams}}^+.
```

This is the correct interface between the abstract hypostructure profile
machinery and the repaired-gauge PDE proof stack. The remaining global
exhaustion statement, C1, is stronger: it must prove that every actual Type II
singularity in the declared backend reaches this C2 interface.
