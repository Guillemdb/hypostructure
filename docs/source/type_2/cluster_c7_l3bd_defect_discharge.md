# C7 defect discharge: critical boundedness from C3

This note discharges the first non-representation defect in the C7 rough-core
bridge. The C7 input package needs

```{math}
K_{L^3\mathrm{Bd}}^+:
\qquad
\sup_{\tau\ge\tau_0}\|V(\tau)\|_{L^3(\mathbb R^3)}<\infty.
```

C3 already supplies the stronger positive finite critical-mass annulus
certificate

```{math}
K_{L^3\mathrm{Norm}}^+:
\qquad
\exists\,0<\eta\le M<\infty:
\eta\le\|V(\tau)\|_{L^3(\mathbb R^3)}\le M
\quad\forall\tau\ge\tau_0.
```

Therefore the bounded-critical-norm defect
\(K_{L^3\mathrm{Bd}}^-\) is not an independent rough-core survivor once C3 has
emitted \(K_{L^3\mathrm{Norm}}^+\).

## Bounded critical norm certificate

::::{prf:definition} Bounded critical norm certificate
:label: def-c7-l3bd-certificate

For a represented repaired-gauge Type II orbit \((V,P,a,b)\), define

```{math}
K_{L^3\mathrm{Bd}}^+
```

as the certificate that

```{math}
\sup_{\tau\ge\tau_0}\|V(\tau)\|_{L^3(\mathbb R^3)}<\infty.
```

Its ordered defect is

```{math}
K_{L^3\mathrm{Bd}}^-:
\quad
\sup_{\tau\ge\tau_0}\|V(\tau)\|_{L^3(\mathbb R^3)}=\infty
\quad\text{or the critical norm is unavailable on the represented branch.}
```

::::

The second alternative is included only for typed certificate accounting. In
the C7 bridge, the orbit is already represented by C2, and C3 has already
evaluated the critical norm.

## C3 implies bounded critical norm

::::{prf:lemma} C3 critical normalization discharges boundedness
:label: lem-c3-l3norm-implies-l3bd

For every represented repaired-gauge Type II orbit,

```{math}
K_{L^3\mathrm{Norm}}^+
\Longrightarrow
K_{L^3\mathrm{Bd}}^+.
```

::::

:::{prf:proof}
By the definition of \(K_{L^3\mathrm{Norm}}^+\) in C3, there exist constants
\(0<\eta\le M<\infty\) such that

```{math}
\eta\le\|V(\tau)\|_{L^3(\mathbb R^3)}\le M
\qquad\forall\tau\ge\tau_0.
```

Taking the supremum in \(\tau\) gives

```{math}
\sup_{\tau\ge\tau_0}\|V(\tau)\|_{L^3(\mathbb R^3)}\le M<\infty.
```

This is exactly \(K_{L^3\mathrm{Bd}}^+\). \(\square\)
:::

## C7 consequence

::::{prf:corollary} C7 rough-core input package with C3 boundedness
:label: cor-c7-input-with-c3-boundedness

Assume a declared Type II candidate has

```{math}
K_{\mathrm{RepBridge}}^+,
\qquad
K_{L^3\mathrm{Norm}}^+,
\qquad
K_{\mathrm{ModMatrixInv}}^+,
\qquad
K_{\mathrm{ModForceBd}}^+,
\qquad
K_{\mathrm{CaccioppoliReg}}^+.
```

Then the candidate emits

```{math}
K_{\mathrm{WinH1}}^+.
```

Consequently, the pointwise rough-core blocker payload used by C8 is available
for that represented branch.

::::

:::{prf:proof}
By Lemma {prf:ref}`lem-c3-l3norm-implies-l3bd`,
\(K_{L^3\mathrm{Norm}}^+\) gives \(K_{L^3\mathrm{Bd}}^+\). The remaining
hypotheses are exactly the C7.5 hypotheses in
[cluster_c7_win_h1_certificate_composition.md](cluster_c7_win_h1_certificate_composition.md):
representation, bounded critical norm, modulation-matrix invertibility,
modulation forcing bounds, and Caccioppoli regularity. Therefore C7.5 emits
\(K_{\mathrm{WinH1}}^+\). \(\square\)
:::

## C-series C7 discharge

::::{prf:corollary} C-series rough-core bridge after C3
:label: cor-c7-after-c3-l3bd-discharge

On the C-series route, assume a declared Type II candidate has

```{math}
K_{\mathrm{RepBridge}}^+,
\qquad
K_{L^3\mathrm{Norm}}^+,
\qquad
K_{\mathrm{ModMatrixInv}}^+,
\qquad
K_{\mathrm{ModForceBd}}^+,
\qquad
K_{\mathrm{CaccioppoliReg}}^+.
```

Then it emits the pointwise rough-core blocker

```{math}
K_{\mathrm{RoughCoreBlk}}^+(\omega),
```

equivalently \(K_{\mathrm{WinH1}}^+(\omega)\) for that candidate.

::::

:::{prf:proof}
Lemma {prf:ref}`lem-c3-l3norm-implies-l3bd` replaces the C7 input
\(K_{L^3\mathrm{Bd}}^+\) by the already available C3 certificate
\(K_{L^3\mathrm{Norm}}^+\). The hypotheses are then exactly the pointwise
C7.5 hypotheses in
[cluster_c7_win_h1_certificate_composition.md](cluster_c7_win_h1_certificate_composition.md).
C7.5 emits \(K_{\mathrm{WinH1}}^+(\omega)\), and C7.6 identifies this with
the pointwise rough-core blocker payload used by C8. \(\square\)
:::

## Updated ordered defects

After this discharge, the C7 rough-core bridge no longer treats
\(K_{L^3\mathrm{Bd}}^-\) as an independent remaining defect whenever
\(K_{L^3\mathrm{Norm}}^+\) is present. The remaining nontrivial C7 defects are:

```{math}
K_{\mathrm{RepBridge}}^-,
\qquad
K_{\mathrm{ModForceBd}}^-,
\qquad
K_{\mathrm{CaccioppoliReg}}^-.
```

If the repaired-gauge nondegeneracy payload of
[cluster_c7_modmatrix_inverse_discharge.md](cluster_c7_modmatrix_inverse_discharge.md)
is not imported, then \(K_{\mathrm{ModMatrixInv}}^-\) remains as the corresponding
gauge-layer defect.

In the C-series route, \(K_{\mathrm{RepBridge}}^+\) and
\(K_{L^3\mathrm{Norm}}^+\) are already part of
\(K_{\mathrm{ClassComplete}}^+\). Thus the next useful C7 discharges are the modulation-force bound and
Caccioppoli-regularity certificates, after importing the modulation-matrix
inverse discharge. If \(K_{\mathrm{RepBridge}}^+\) is not imported from C2/C5, then
\(K_{\mathrm{RepBridge}}^-\) remains the first ordered C7 defect.
