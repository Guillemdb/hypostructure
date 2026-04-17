# S13: local compact-cylinder terminal sequences

S12 reduces local terminal compactness to the standard critical
Navier-Stokes profile-decomposition theorem. The only internal boundedness
input left there is

```{math}
K_{\mathrm{BoundedCritTermSeq}}^+.
```

This note proves that input on the C3 positive finite critical-mass branch.

## Terminal sequence routing

::::{prf:definition} Terminal sequences sampled from the represented orbit
:label: def-s13-terminal-sequence-routing

The certificate

```{math}
K_{\mathrm{TermSeqFromOrbit}}^+
```

means that every terminal active-camera sequence used by S8 and S12 is obtained
by sampling the represented repaired-gauge Type II orbit and applying only
Navier-Stokes critical symmetries:

```{math}
u_{0,n}
=
\Lambda_{\lambda_n,x_n} V(\tau_n)
```

or the inverse equivalent camera normalization, with
\(\tau_n\to\infty\), \(\lambda_n>0\), and \(x_n\in\mathbb R^3\). The
normalization maps preserve the critical \(L^3\) norm:

```{math}
\|\Lambda_{\lambda_n,x_n} V(\tau_n)\|_{L^3(\mathbb R^3)}
=
\|V(\tau_n)\|_{L^3(\mathbb R^3)}.
```

If a terminal active-camera sequence is not obtained this way, the defect is
denoted

```{math}
K_{\mathrm{TermSeqRoute}}^-.
```

::::

This is a routing certificate, not an analytic estimate. It ensures that the
sequences to which S12 applies are the same sequences controlled by the C3
critical-mass ledger.

## Boundedness from C3

::::{prf:lemma} Critical symmetry preserves the \(L^3\) norm
:label: lem-s13-critical-symmetry-preserves-l3

For

```{math}
(\Lambda_{\lambda,x_0}f)(x)
:=
\lambda^{-1}f\left(\frac{x-x_0}{\lambda}\right),
```

one has

```{math}
\|\Lambda_{\lambda,x_0}f\|_{L^3(\mathbb R^3)}
=
\|f\|_{L^3(\mathbb R^3)}.
```

::::

:::{prf:proof}
By the change of variables \(y=(x-x_0)/\lambda\),

```{math}
\|\Lambda_{\lambda,x_0}f\|_3^3
=
\int_{\mathbb R^3}\lambda^{-3}
\left|f\left(\frac{x-x_0}{\lambda}\right)\right|^3\,dx
=
\int_{\mathbb R^3}|f(y)|^3\,dy.
```

Taking cube roots gives the claim. \(\square\)
:::

::::{prf:theorem} S13 bounded terminal sequences from \(L^3\)-normalization
:label: thm-s13-bounded-terminal-sequences-from-l3norm

Assume

```{math}
K_{L^3\mathrm{Norm}}^+
\wedge
K_{\mathrm{TermSeqFromOrbit}}^+.
```

Then

```{math}
K_{\mathrm{BoundedCritTermSeq}}^+.
```

::::

:::{prf:proof}
By \(K_{L^3\mathrm{Norm}}^+\), there is \(M<\infty\) such that the represented
renormalized orbit satisfies

```{math}
\|V(\tau)\|_{L^3(\mathbb R^3)}\le M
\qquad
\text{for all }\tau\ge\tau_0.
```

Let \(u_{0,n}\) be any terminal active-camera sequence used by S8/S12. By
\(K_{\mathrm{TermSeqFromOrbit}}^+\), it is obtained from samples
\(V(\tau_n)\) by critical Navier-Stokes symmetries. Lemma
{prf:ref}`lem-s13-critical-symmetry-preserves-l3` gives

```{math}
\|u_{0,n}\|_3
=
\|V(\tau_n)\|_3
\le M.
```

Thus every such terminal sequence is uniformly bounded in \(L^3\). This is
\(K_{\mathrm{BoundedCritTermSeq}}^+\). \(\square\)
:::

## Failure classification

::::{prf:theorem} S13 bounded terminal sequence dichotomy
:label: thm-s13-bounded-terminal-sequence-dichotomy

For a represented Type II candidate reaching the S8/S12 terminal-camera
analysis, exactly one of the following ordered outcomes occurs:

1. \(K_{\mathrm{TermSeqRoute}}^-\);
2. \(K_{L^3\mathrm{Dom}}^-\);
3. \(K_{L^3\mathrm{Inf}}^-\);
4. \(K_{L^3\mathrm{Zero}}^-\);
5. \(K_{\mathrm{BoundedCritTermSeq}}^+\), after the positive C3 branch
   \(K_{L^3\mathrm{Norm}}^+\) is emitted.

::::

:::{prf:proof}
First check whether the terminal active-camera sequences are sampled from the
represented orbit by critical symmetries. If not, the first defect
\(K_{\mathrm{TermSeqRoute}}^-\) is emitted.

Assume \(K_{\mathrm{TermSeqFromOrbit}}^+\). Apply the C3 ordered
critical-mass evaluator to the represented orbit. C3 emits exactly one ordered
output among \(K_{L^3\mathrm{Dom}}^-\), \(K_{L^3\mathrm{Inf}}^-\),
\(K_{L^3\mathrm{Zero}}^-\), and \(K_{L^3\mathrm{Norm}}^+\). In the first three
cases S13 records the same critical-mass defect. In the positive finite
critical-mass case, Theorem
{prf:ref}`thm-s13-bounded-terminal-sequences-from-l3norm` gives
\(K_{\mathrm{BoundedCritTermSeq}}^+\).

On the C18 terminal package, \(K_{L^3\mathrm{Norm}}^+\) is included
explicitly, so the fifth outcome is the relevant terminal route. \(\square\)
:::

## Consequence for S12 and C18

::::{prf:corollary} S13 discharges the S12 boundedness input
:label: cor-s13-discharges-s12-boundedness-input

In S12, the hypothesis

```{math}
K_{\mathrm{BoundedCritTermSeq}}^+
```

may be replaced by

```{math}
K_{L^3\mathrm{Norm}}^+
\wedge
K_{\mathrm{TermSeqFromOrbit}}^+.
```

::::

:::{prf:proof}
This is Theorem
{prf:ref}`thm-s13-bounded-terminal-sequences-from-l3norm`. \(\square\)
:::

::::{prf:corollary} C18 terminal package with explicit boundedness discharge
:label: cor-s13-c18-terminal-package-with-explicit-boundedness

The C18 terminal payload may replace \(K_{\mathrm{TermCritProfThm},NS3D}^+\)
by the explicit conjunction

```{math}
K_{L^3\mathrm{Norm}}^+
\wedge
K_{\mathrm{RepBridge}}^+
\wedge
K_{\mathrm{SmallDataStab}_{L^3}}^+
\wedge
K_{\mathrm{CriticalNSProfDecomp}}^+.
```

::::

:::{prf:proof}
By S14, \(K_{\mathrm{RepBridge}}^+\) gives
\(K_{\mathrm{TermSeqFromOrbit}}^+\) in the declared terminal backend. Then
\(K_{L^3\mathrm{Norm}}^+\wedge K_{\mathrm{TermSeqFromOrbit}}^+\) gives
\(K_{\mathrm{BoundedCritTermSeq}}^+\) by Corollary
{prf:ref}`cor-s13-discharges-s12-boundedness-input`. Together with
\(K_{\mathrm{SmallDataStab}_{L^3}}^+\) and
\(K_{\mathrm{CriticalNSProfDecomp}}^+\), this is exactly
\(K_{\mathrm{TermCritProfThm},NS3D}^+\) from S12. C18 then applies. \(\square\)
:::
