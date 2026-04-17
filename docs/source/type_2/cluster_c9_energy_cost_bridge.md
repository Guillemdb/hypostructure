# C9 energy-cost bridge: physical energy versus renormalized Type II cost

This note implements C9 in the Type II classification program. It relates the
unweighted renormalized Type II barrier cost to the physical Navier-Stokes
energy budget.

The main conclusion is:

```{math}
\text{finite physical energy controls a scale-weighted renormalized cost,}
```

not the unweighted cost used by the compact Type II barrier.

This distinction matters because the generic `UP-TypeII` theorem in the
Hypostructure framework is written for semilinear heat-type parabolic models.
It is not automatically applicable to 3D Navier-Stokes. For NS3D we therefore
separate:

1. the blocked Type II barrier certificate
   \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\), produced by C4 from infinite
   unweighted cost; and
2. the promotion
   \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\Rightarrow
   K_{\mathrm{SC}_\lambda}^{\sim}\), which requires an additional
   Navier-Stokes applicability certificate.

## UP-TypeII applicability for NS3D

::::{prf:definition} Navier-Stokes UP-TypeII applicability certificate
:label: def-ns-uptypeii-applicability

The certificate

```{math}
K_{\mathrm{NS\text{-}UPTypeII}}^+
```

means that the hypotheses of the framework theorem `UP-TypeII` have been
verified for the declared NS3D repaired-gauge Type II backend, or that a
Navier-Stokes-specific replacement promotion theorem has been supplied.

The full payload is defined in
[cluster_c10_ns_up_typeII_promotion.md](cluster_c10_ns_up_typeII_promotion.md)
as \(K_{\mathrm{NSPromPayload}}^+\). The definition below records the C9
energy-cost consequences of that promotion bridge.

Without \(K_{\mathrm{NS\text{-}UPTypeII}}^+\), C4 emits
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) but not automatically
\(K_{\mathrm{SC}_\lambda}^{\sim}\).

::::

::::{prf:lemma} Conditional NS3D Type II promotion
:label: lem-conditional-ns3d-typeii-promotion

Assume

```{math}
K_{\mathrm{SC}_\lambda}^-,
\qquad
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}},
\qquad
K_{\mathrm{NS\text{-}UPTypeII}}^+.
```

Then

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}
```

is emitted.

::::

:::{prf:proof}
The certificate \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) is exactly the
applicability bridge that permits the `UP-TypeII` promotion, or its
Navier-Stokes-specific replacement, to be applied to the NS3D blocked Type II
barrier. Applying that promotion to
\(K_{\mathrm{SC}_\lambda}^-\wedge K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\)
emits \(K_{\mathrm{SC}_\lambda}^{\sim}\). \(\square\)
:::

## Setup

Let

```{math}
u(x,t)
=
\lambda(t)^{-1}
V\left(\frac{x-x_c(t)}{\lambda(t)},\tau(t)\right),
\qquad
\frac{d\tau}{dt}=\lambda(t)^{-2}.
```

The physical energy inequality is

```{math}
E(t)
+
\nu\int_{0}^{t}\|\nabla_x u(s)\|_{L^2_x}^2\,ds
\le
E(0),
\qquad
E(t)=\frac12\|u(t)\|_{L^2_x}^2.
```

The compact-barrier cost is

```{math}
\tilde{\mathfrak D}_{R_0}(\tau)
=
\nu\int|\nabla_y V|^2\phi_{R_0}
+
a_+(\tau)\int |V|^2\phi_{R_0}.
```

## Scaling identity for dissipation

::::{prf:lemma} Physical dissipation in renormalized variables
:label: lem-physical-dissipation-renormalized

For the renormalization above,

```{math}
\|\nabla_x u(t)\|_{L^2_x}^2
=
\lambda(t)^{-1}
\|\nabla_y V(\tau(t))\|_{L^2_y}^2.
```

Consequently,

```{math}
\int_{t_1}^{t_2}
\|\nabla_x u(t)\|_{L^2_x}^2\,dt
=
\int_{\tau(t_1)}^{\tau(t_2)}
\lambda(\tau)
\|\nabla_y V(\tau)\|_{L^2_y}^2\,d\tau.
```

::::

:::{prf:proof}
Since \(u=\lambda^{-1}V\) and \(y=(x-x_c)/\lambda\),
\(\nabla_x u=\lambda^{-2}\nabla_yV\). With \(dx=\lambda^3dy\),

```{math}
\|\nabla_x u\|_{L^2_x}^2
=
\int \lambda^{-4}|\nabla_yV|^2\lambda^3dy
=
\lambda^{-1}\|\nabla_yV\|_{L^2_y}^2.
```

Since \(d\tau/dt=\lambda^{-2}\), \(dt=\lambda^2d\tau\). Multiplying gives the
second identity. \(\square\)
:::

## Physical-energy controlled cost

::::{prf:definition} Physical-renormalized dissipation cost
:label: def-physical-renormalized-dissipation-cost

Define

```{math}
\mathfrak D_{\mathrm{phys-ren}}(\tau)
:=
\nu\lambda(\tau)\|\nabla_yV(\tau)\|_{L^2_y}^2.
```

and the localized version

```{math}
\mathfrak D_{\mathrm{phys-ren},R_0}(\tau)
:=
\nu\lambda(\tau)\int|\nabla_yV|^2\phi_{R_0}.
```

::::

::::{prf:theorem} Physical energy controls scale-weighted renormalized dissipation
:label: thm-energy-controls-scale-weighted-dissipation

Assume \(K_{D_E}^+\), i.e. finite physical energy and the Navier-Stokes energy
inequality. Then, for every \(t_1<T^*\),

```{math}
\int_{\tau(t_1)}^{\infty}
\mathfrak D_{\mathrm{phys-ren}}(\tau)\,d\tau
\le
E(t_1)
\le
E(0).
```

In particular,

```{math}
\int_{\tau(t_1)}^{\infty}
\mathfrak D_{\mathrm{phys-ren},R_0}(\tau)\,d\tau
<\infty.
```

::::

:::{prf:proof}
The physical energy inequality on \([t_1,t]\) gives

```{math}
\nu\int_{t_1}^{t}\|\nabla_xu(s)\|_2^2ds\le E(t_1).
```

Use Lemma {prf:ref}`lem-physical-dissipation-renormalized` and let
\(t\uparrow T^*\). Monotone convergence gives the full weighted bound. The
localized estimate follows from \(0\le\phi_{R_0}\le1\). \(\square\)
:::

## The \(a_+\)-term

The compact-barrier cost also contains \(a_+\int |V|^2\phi_{R_0}\). Physical
energy controls \(\lambda\|V\|_2^2\), but not automatically
\(\lambda a_+\|V\|_2^2\). We therefore isolate the needed input.

::::{prf:definition} Weighted \(a_+\)-cost control
:label: def-weighted-aplus-cost-control

The certificate

```{math}
K_{a_+\mathrm{Weight}}^+
```

means

```{math}
\int_{\tau_1}^{\infty}
\lambda(\tau)a_+(\tau)
\int |V(y,\tau)|^2\phi_{R_0}(y)\,dy\,d\tau
<\infty.
```

::::

::::{prf:definition} Scale-weighted Type II barrier cost
:label: def-scale-weighted-typeII-cost

Define

```{math}
\mathfrak C_{\mathrm{II,phys}}^{NS}(\tau)
:=
\lambda(\tau)\tilde{\mathfrak D}_{R_0}(\tau).
```

::::

::::{prf:theorem} Finite physical energy controls the scale-weighted Type II cost
:label: thm-energy-controls-scale-weighted-typeII-cost

Assume \(K_{D_E}^+\) and \(K_{a_+\mathrm{Weight}}^+\). Then

```{math}
\int_{\tau_1}^{\infty}
\mathfrak C_{\mathrm{II,phys}}^{NS}(\tau)\,d\tau
<\infty.
```

::::

:::{prf:proof}
By definition,

```{math}
\mathfrak C_{\mathrm{II,phys}}^{NS}
=
\lambda\nu\int|\nabla V|^2\phi_{R_0}
+
\lambda a_+\int |V|^2\phi_{R_0}.
```

The first term is integrable by Theorem
{prf:ref}`thm-energy-controls-scale-weighted-dissipation`; the second is
integrable by \(K_{a_+\mathrm{Weight}}^+\). \(\square\)
:::

## Infinite-cost classification

::::{prf:definition} Scale-decoupled infinite Type II cost
:label: def-scale-decoupled-infinite-cost

The certificate

```{math}
K_{\mathrm{ScaleDecCost}}^+
```

means

```{math}
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)d\tau=\infty,
\qquad
\int_{\tau_0}^{\infty}\lambda(\tau)\tilde{\mathfrak D}_{R_0}(\tau)d\tau<\infty.
```

::::

::::{prf:definition} Physically forbidden weighted infinite cost
:label: def-phys-forbidden-weighted-infinite-cost

The certificate

```{math}
K_{\mathrm{PhysCostInf}}^-
```

means

```{math}
\int_{\tau_0}^{\infty}
\mathfrak C_{\mathrm{II,phys}}^{NS}(\tau)\,d\tau
=\infty.
```

Under \(K_{D_E}^+\wedge K_{a_+\mathrm{Weight}}^+\), this is impossible by
Theorem {prf:ref}`thm-energy-controls-scale-weighted-typeII-cost`.

::::

::::{prf:theorem} C9 infinite-cost trichotomy
:label: thm-c9-infinite-cost-trichotomy

Assume \(K_{D_E}^+\) and \(K_{a_+\mathrm{Weight}}^+\). Let a represented Type II
candidate satisfy

```{math}
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)d\tau=\infty.
```

Then:

1. C4 emits \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\).
2. If \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) also holds, then
   \(K_{\mathrm{SC}_\lambda}^{\sim}\) is emitted.
3. Relative to physical energy, exactly one of the following holds:
   \(K_{\mathrm{PhysCostInf}}^-\) or \(K_{\mathrm{ScaleDecCost}}^+\).

::::

:::{prf:proof}
The unweighted infinite-cost conclusion is exactly the default NS3D
`BarrierTypeII` cost divergence from C4, so C4 emits
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\). If
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\) is present, Lemma
{prf:ref}`lem-conditional-ns3d-typeii-promotion` emits
\(K_{\mathrm{SC}_\lambda}^{\sim}\).

For the physical-energy classification, evaluate
\(\int\mathfrak C_{\mathrm{II,phys}}^{NS}\). If it is infinite, then
\(K_{\mathrm{PhysCostInf}}^-\) fires, contradicting
\(K_{D_E}^+\wedge K_{a_+\mathrm{Weight}}^+\). If it is finite, then the
candidate is scale-decoupled by Definition
{prf:ref}`def-scale-decoupled-infinite-cost`. The two physical-energy outcomes
are disjoint and exhaustive. \(\square\)
:::

## Corrected conclusion

The rigorous replacement for the informal slogan is:

```{math}
\text{infinite scale-weighted renormalized cost}
\Longrightarrow
\text{infinite physical energy dissipation}.
```

But:

```{math}
\text{infinite unweighted renormalized cost}
\centernot\Longrightarrow
\text{infinite physical energy}
```

without an additional scale-weight bridge.

Also, for 3D Navier-Stokes:

```{math}
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}
\centernot\Longrightarrow
K_{\mathrm{SC}_\lambda}^{\sim}
```

unless the applicability bridge \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) is
present. The generic `UP-TypeII` proof is heat-model-oriented; NS3D must either
verify its hypotheses or provide a Navier-Stokes-specific replacement
promotion.
