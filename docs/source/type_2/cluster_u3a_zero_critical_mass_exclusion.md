# U3a: zero critical mass exclusion

This note discharges the formal part of U3a. It proves that the C3 zero
critical-mass defect is not an admissible Type II core once the branch carries
a tail nontrivial-core certificate. It also separates the analytic persistence
input needed to upgrade subsequential concentration into a tail lower bound.

The result does not prove the full positive finite annulus
\(K_{L^3\mathrm{Norm}}^+\). It removes only the zero branch
\(K_{L^3\mathrm{Zero}}^-\). The infinite critical-mass branch
\(K_{L^3\mathrm{Inf}}^-\) remains a separate U3b target.

## Nontrivial core certificate

::::{prf:definition} Nontrivial critical core certificate
:label: def-u3a-nontrivial-critical-core

For a represented repaired-gauge Type II branch \((V,P,a,b)\), the certificate

```{math}
K_{\mathrm{CoreNontriv}}^+
```

means that there exist \(R_*<\infty\), \(\eta_*>0\), and
\(\tau_*\ge\tau_0\) such that

```{math}
\|V(\tau)\|_{L^3(B_{R_*})}\ge \eta_*
\qquad
\text{for all }\tau\ge\tau_*.
```

Equivalently, the represented terminal core remains nonzero in the
scale-critical topology on the renormalized tail.

::::

This is the minimal critical-topology content needed to rule out the C3
zero-collapse alternative. It is stronger than mere subsequential
concentration: it is a no-return statement saying that the retained terminal
core cannot repeatedly return arbitrarily close to zero in the critical
topology.

::::{prf:definition} Subsequence nontriviality certificate
:label: def-u3a-subseq-nontrivial-critical-core

The certificate

```{math}
K_{\mathrm{CoreSubseqNontriv}}^+
```

means that there exist \(R_*<\infty\), \(\eta_*>0\), and a sequence
\(\tau_n\to\infty\) such that

```{math}
\|V(\tau_n)\|_{L^3(B_{R_*})}\ge \eta_*
\qquad
\text{for all }n.
```

::::

::::{prf:definition} Core persistence certificate
:label: def-u3a-core-persistence

The certificate

```{math}
K_{\mathrm{CorePersist}}^+
```

means that subsequential nontriviality of a retained terminal Type II core
upgrades to tail nontriviality:

```{math}
K_{\mathrm{CoreSubseqNontriv}}^+
\Longrightarrow
K_{\mathrm{CoreNontriv}}^+.
```

Failure of this upgrade is denoted

```{math}
K_{\mathrm{CorePersist}}^-.
```

::::

::::{prf:definition} Zero-core defect
:label: def-u3a-zero-core-defect

The defect

```{math}
K_{\mathrm{CoreNontriv}}^-
```

means that no such tail lower bound exists. A branch carrying this defect is a
failed extraction/nontriviality branch, not an admissible Type II core.

::::

## Concentration extraction emits subsequential nontriviality

::::{prf:theorem} Type II concentration emits a subsequential nontrivial critical core
:label: thm-u3a-concentration-emits-core-nontriv

Assume a branch has entered the declared Type II backend through concentration
extraction and repaired-gauge representation:

```{math}
K_{\mathrm{CmuExtract}}^+
\wedge
K_{\mathrm{RepBridge}}^+.
```

Assume also that the extracted branch is retained as a terminal Type II core,
not discarded as scattering, exterior regular mass, or a lost-profile defect.
Then

```{math}
K_{\mathrm{CoreSubseqNontriv}}^+.
```

::::

:::{prf:proof}
The concentration extraction certificate supplies a nonzero profile/germ in
the declared NS3D profile backend. The repaired-gauge bridge transports that
profile to the renormalized orbit \(V(\tau)\) by Navier-Stokes critical
symmetries and admissible chart maps. These symmetries preserve the
scale-critical \(L^3\) norm.

Because the extracted profile is nonzero in \(L^3_{\mathrm{loc}}\) after
choosing its terminal camera, there are \(R_*<\infty\), \(\eta_*>0\), and the
corresponding extraction times \(\tau_n\to\infty\) such that
\(\|V(\tau_n)\|_{L^3(B_{R_*})}\ge\eta_*\). This is exactly
\(K_{\mathrm{CoreSubseqNontriv}}^+\). \(\square\)
:::

Theorem {prf:ref}`thm-u3a-concentration-emits-core-nontriv` does not by itself
rule out \(\liminf_{\tau\to\infty}\|V(\tau)\|_3=0\). That stronger conclusion
requires \(K_{\mathrm{CorePersist}}^+\). This distinction prevents a
subsequence extraction theorem from being used as a uniform no-return theorem.

## Zero critical mass is impossible for a nontrivial core

::::{prf:theorem} U3a zero critical mass exclusion
:label: thm-u3a-zero-critical-mass-exclusion

For a represented repaired-gauge Type II branch,

```{math}
K_{\mathrm{CorePersist}}^+
\wedge
K_{\mathrm{CoreSubseqNontriv}}^+
\Longrightarrow
\neg K_{L^3\mathrm{Zero}}^-.
```

Equivalently, if the C3 ordered evaluator emits
\(K_{L^3\mathrm{Zero}}^-\), then the branch cannot be an admissible retained
Type II core satisfying persistence; it must emit at least one upstream defect

```{math}
K_{\mathrm{CoreSubseqNontriv}}^-,
\qquad
K_{\mathrm{CorePersist}}^-.
```

::::

:::{prf:proof}
By \(K_{\mathrm{CorePersist}}^+\wedge K_{\mathrm{CoreSubseqNontriv}}^+\), the
tail nontriviality certificate \(K_{\mathrm{CoreNontriv}}^+\) holds. Hence
there are
\(R_*<\infty\), \(\eta_*>0\), and \(\tau_*\) such that

```{math}
\|V(\tau)\|_{L^3(B_{R_*})}\ge \eta_*
\qquad
\text{for all }\tau\ge\tau_*.
```

Since \(B_{R_*}\subset\mathbb R^3\), this gives the global lower bound

```{math}
\|V(\tau)\|_{L^3(\mathbb R^3)}\ge \eta_*
\qquad
\text{for all }\tau\ge\tau_*.
```

Hence

```{math}
\liminf_{\tau\to\infty}\|V(\tau)\|_{L^3(\mathbb R^3)}
\ge \eta_*>0.
```

But the C3 zero-collapse defect is

```{math}
K_{L^3\mathrm{Zero}}^-:
\qquad
\sup_{\tau\ge\tau_0}\|V(\tau)\|_3<\infty,
\quad
\liminf_{\tau\to\infty}\|V(\tau)\|_3=0.
```

The two statements are incompatible. Therefore a retained nontrivial core
cannot emit \(K_{L^3\mathrm{Zero}}^-\). \(\square\)
:::

## Consequence for C3 and S13

::::{prf:corollary} C3 after U3a
:label: cor-u3a-c3-after-zero-exclusion

On represented retained Type II cores satisfying
\(K_{\mathrm{CorePersist}}^+\wedge K_{\mathrm{CoreSubseqNontriv}}^+\), the
ordered C3 alternatives reduce to

```{math}
K_{L^3\mathrm{Dom}}^-,
\qquad
K_{L^3\mathrm{Inf}}^-,
\qquad
K_{L^3\mathrm{Norm}}^+.
```

The zero alternative is no longer admissible.

::::

:::{prf:proof}
Apply Theorem {prf:ref}`thm-u3a-zero-critical-mass-exclusion` to remove
\(K_{L^3\mathrm{Zero}}^-\) from the C3 ordered list. \(\square\)
:::

::::{prf:corollary} S13 zero-route removal
:label: cor-u3a-s13-zero-route-removal

In the S13 bounded-terminal-sequence dichotomy, after
\(K_{\mathrm{CorePersist}}^+\wedge K_{\mathrm{CoreSubseqNontriv}}^+\) is
supplied, the zero critical-mass route is not a terminal-sequence obstruction.
The remaining S13 defects are

```{math}
K_{\mathrm{TermSeqRoute}}^-,
\qquad
K_{L^3\mathrm{Dom}}^-,
\qquad
K_{L^3\mathrm{Inf}}^-.
```

Inside the declared S8 terminal backend, S14 reduces
\(K_{\mathrm{TermSeqRoute}}^-\) to \(K_{\mathrm{RepBridge}}^-\).

::::

:::{prf:proof}
S13 imports the C3 ordered critical-mass alternatives. Corollary
{prf:ref}`cor-u3a-c3-after-zero-exclusion` removes
\(K_{L^3\mathrm{Zero}}^-\) on retained nontrivial cores. The final sentence is
S14. \(\square\)
:::

## Remaining U3 boundary

U3a proves only the lower-bound half of critical normalization, modulo the
subsequential nontriviality and persistence certificates. The remaining U3
tasks are:

1. persistence/no-return:
   ```{math}
   K_{\mathrm{CorePersist}}^+;
   ```
2. domain well-definedness:
   ```{math}
   \neg K_{L^3\mathrm{Dom}}^-;
   ```
3. infinite critical-mass routing:
   ```{math}
   K_{L^3\mathrm{Inf}}^-
   \Longrightarrow
   \text{accepted barrier defect or outside-Type-II classification};
   ```
4. upper finite critical-mass control on the retained represented branch:
   ```{math}
   \sup_{\tau\gg1}\|V(\tau)\|_3<\infty.
   ```

Once these are supplied, C3 emits the full positive finite annulus
\(K_{L^3\mathrm{Norm}}^+\).
