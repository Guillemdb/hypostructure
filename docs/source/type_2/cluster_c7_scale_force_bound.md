# C7 defect discharge: pointwise weighted scale-force decomposition

This note refines the pointwise T14 scale-row forcing certificate

```{math}
K_{\mathrm{ScaleForceBd}}^+:
\qquad
\sup_{\tau\ge\tau_0}
\left|
\int_{\mathbb R^3}|y|^{-p}|V|V\cdot H(V,P)\,dy
\right|<\infty,
```

where

```{math}
H(V,P)=\nu\Delta V-(V\cdot\nabla)V-\nabla P,
\qquad 0<p<3.
```

The purpose is to separate the weighted scale row into the three analytic
pieces that can be proved independently. It also discharges the nonlinear piece
under an explicit weighted fourth-moment hypothesis.

## Scale-force pieces

::::{prf:definition} Pointwise scale-force payload
:label: def-c7-scale-force-payload

For a represented repaired-gauge Type II orbit, define

```{math}
I_\Delta(\tau)
:=\int_{\mathbb R^3}|y|^{-p}|V|V\cdot \Delta V\,dy,
```

```{math}
I_{\mathrm{nl}}(\tau)
:=\int_{\mathbb R^3}|y|^{-p}|V|V\cdot ((V\cdot\nabla)V)\,dy,
```

and

```{math}
I_P(\tau)
:=\int_{\mathbb R^3}|y|^{-p}|V|V\cdot \nabla P\,dy,
```

whenever these integrals are absolutely convergent or are defined by the
specified distributional pairing used for the represented branch.

Define the positive payloads

```{math}
K_{\mathrm{ScaleLapBd}}^+:
\qquad
\sup_{\tau\ge\tau_0}|I_\Delta(\tau)|<\infty,
```

```{math}
K_{\mathrm{ScaleNonlinBd}}^+:
\qquad
\sup_{\tau\ge\tau_0}|I_{\mathrm{nl}}(\tau)|<\infty,
```

and

```{math}
K_{\mathrm{ScalePressureBd}}^+:
\qquad
\sup_{\tau\ge\tau_0}|I_P(\tau)|<\infty.
```

The aggregate pointwise scale-force decomposition payload is

```{math}
K_{\mathrm{ScaleForcePayload}}^+
:=
K_{\mathrm{ScaleLapBd}}^+
\wedge
K_{\mathrm{ScaleNonlinBd}}^+
\wedge
K_{\mathrm{ScalePressureBd}}^+.
```

::::

The signs in this note follow the T14 convention
\(H=\nu\Delta V-(V\cdot\nabla)V-\nabla P\). Thus the scale-force row is
\(\nu I_\Delta-I_{\mathrm{nl}}-I_P\).

## Decomposition theorem

::::{prf:lemma} Scale-force payload implies pointwise scale-force bound
:label: lem-c7-scale-force-payload-implies-bound

For every represented repaired-gauge Type II orbit,

```{math}
K_{\mathrm{ScaleForcePayload}}^+
\Longrightarrow
K_{\mathrm{ScaleForceBd}}^+.
```

More explicitly, if

```{math}
\sup_\tau |I_\Delta(\tau)|\le C_\Delta,
\qquad
\sup_\tau |I_{\mathrm{nl}}(\tau)|\le C_{\mathrm{nl}},
\qquad
\sup_\tau |I_P(\tau)|\le C_P,
```

then

```{math}
\sup_{\tau\ge\tau_0}
\left|
\int |y|^{-p}|V|V\cdot H(V,P)\,dy
\right|
\le
\nu C_\Delta+C_{\mathrm{nl}}+C_P.
```

::::

:::{prf:proof}
By the definition of \(H\),

```{math}
\int |y|^{-p}|V|V\cdot H(V,P)\,dy
=
\nu I_\Delta-I_{\mathrm{nl}}-I_P.
```

The triangle inequality gives the displayed uniform bound. \(\square\)
:::

## Nonlinear weighted piece

::::{prf:definition} Weighted fourth-moment certificate
:label: def-c7-scale-l4-moment

Define

```{math}
K_{\mathrm{ScaleL4Mom}}^+:
\qquad
\sup_{\tau\ge\tau_0}
\int_{\mathbb R^3}|y|^{-p-1}|V(y,\tau)|^4\,dy<\infty.
```

::::

This is stronger than the weighted first-moment cubic bound used for the mixed
modulation block in T3. The T3 bound controls
\(\int |y|^{-p-1}|V|^3\); the nonlinear scale-force row naturally produces the
weighted fourth moment after integration by parts.

::::{prf:lemma} Weighted fourth moment controls the nonlinear scale row
:label: lem-c7-scale-nonlinear-from-l4-moment

Assume \(V(\tau)\) is divergence-free and belongs to the admissibility class
needed to justify the weighted integration by parts, for instance by smooth
compact approximation in the repaired-gauge class. Then

```{math}
K_{\mathrm{ScaleL4Mom}}^+
\Longrightarrow
K_{\mathrm{ScaleNonlinBd}}^+.
```

More precisely,

```{math}
|I_{\mathrm{nl}}(\tau)|
\le
\frac{p}{3}
\int_{\mathbb R^3}|y|^{-p-1}|V(y,\tau)|^4\,dy.
```

::::

:::{prf:proof}
For smooth compactly supported approximants, use

```{math}
|V|V\cdot((V\cdot\nabla)V)
=\frac13 V\cdot\nabla(|V|^3).
```

Since \(\nabla\cdot V=0\), integration by parts gives

```{math}
I_{\mathrm{nl}}
=\frac13\int |y|^{-p}V\cdot\nabla(|V|^3)\,dy
=-\frac13\int |V|^3 V\cdot\nabla(|y|^{-p})\,dy.
```

Because

```{math}
\nabla(|y|^{-p})=-p|y|^{-p-2}y,
```

we obtain

```{math}
I_{\mathrm{nl}}
=\frac{p}{3}\int |y|^{-p-2}(V\cdot y)|V|^3\,dy.
```

Therefore

```{math}
|I_{\mathrm{nl}}|
\le
\frac{p}{3}\int |y|^{-p-1}|V|^4\,dy.
```

Taking the supremum in \(\tau\) proves the certificate implication. The general
admissible case follows by the same approximation convention used for the
repaired weighted scale gauge. \(\square\)
:::

## C7 consequence

::::{prf:corollary} Pointwise scale-force route after nonlinear discharge
:label: cor-c7-scale-force-route-after-nonlinear

For every represented repaired-gauge Type II orbit,

```{math}
K_{\mathrm{ScaleLapBd}}^+
\wedge
K_{\mathrm{ScaleL4Mom}}^+
\wedge
K_{\mathrm{ScalePressureBd}}^+
\Longrightarrow
K_{\mathrm{ScaleForceBd}}^+.
```

::::

:::{prf:proof}
Lemma {prf:ref}`lem-c7-scale-nonlinear-from-l4-moment` turns
\(K_{\mathrm{ScaleL4Mom}}^+\) into \(K_{\mathrm{ScaleNonlinBd}}^+\). Together
with \(K_{\mathrm{ScaleLapBd}}^+\) and \(K_{\mathrm{ScalePressureBd}}^+\), this
is exactly \(K_{\mathrm{ScaleForcePayload}}^+\). Lemma
{prf:ref}`lem-c7-scale-force-payload-implies-bound` then emits
\(K_{\mathrm{ScaleForceBd}}^+\). \(\square\)
:::

## Updated ordered defects

T14 reduces the pointwise modulation-force defect to translation forcing and
weighted scale forcing. This note refines the weighted scale forcing defect as

```{math}
K_{\mathrm{ScaleForceBd}}^-
\leadsto
K_{\mathrm{ScaleLapBd}}^-
\quad\text{or}\quad
K_{\mathrm{ScaleL4Mom}}^-
\quad\text{or}\quad
K_{\mathrm{ScalePressureBd}}^-.
```

The nonlinear scale-row term is therefore no longer a separate obstruction once
\(K_{\mathrm{ScaleL4Mom}}^+\) is available. The remaining analytic pointwise
scale-row work is the weighted Laplacian term and the weighted pressure term,
plus the weighted fourth-moment input if it is not already supplied by the
backend.
