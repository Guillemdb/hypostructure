# C5 two-bucket finite-cost classification

This note implements C5 in the classification-completeness program. It consumes
C1--C4 and the compact Type II good-window barrier to produce an ordered,
certificate-level classification of declared Type II candidates:

```{math}
\text{declared Type II}
\Longrightarrow
\text{suppressed}
\quad\vee\quad
\text{radiative/noncompact}
\quad\vee\quad
\text{rough-core}.
```

Consequently, every finite-cost non-suppressed declared Type II candidate is in
one of the two survivor buckets:

```{math}
\text{finite-cost non-suppressed Type II}
\Longrightarrow
\text{radiative/noncompact}
\quad\vee\quad
\text{rough-core}.
```

The result is still Type II-specific. It does not prove global regularity, and
it does not eliminate the two remaining PDE survivor buckets. Those are the
roles of C6 and C7.

## Scope and typed outputs

C5 is a theorem about the **declared** Navier-Stokes Type II backend. All
certificates below are evaluated on a fixed declared candidate
\(\omega\in\mathcal U_{\mathrm{II}}^{NS}\), after C1 has routed it into the
Type II sieve branch.

The proof uses typed certificate evaluation, not informal negation. For each
remaining compact-barrier input, the backend emits exactly one ordered output:
a positive certificate or its corresponding defect certificate. This is the
extra bookkeeping needed to make the two-bucket conclusion airtight.

## Inputs from C1--C4

The C5 theorem uses the implemented bridge certificates:

```{math}
K_{\mathrm{TypeIIExhaust}}^+,\qquad
K_{\mathrm{RepBridge}}^+,\qquad
K_{L^3\mathrm{Norm}}^+,\qquad
K_{\mathrm{CostBridge}}^+.
```

Their meanings are:

1. C1: every declared Type II candidate enters the
   \(K_{C_\mu}^+\wedge K_{\mathrm{SC}_\lambda}^-\wedge
   K_{\mathrm{Prof}_{NS}}^+\) route;
2. C2: the routed profile supplies a repaired-gauge renormalized
   Navier-Stokes orbit \((V,P,a,b)\);
3. C3: the orbit lies in a positive finite critical \(L^3\)-mass annulus;
4. C4: the localized PDE cost is C4-ready, meaning it is measurable,
   nonnegative, and locally integrable on finite \(\tau\)-intervals, and
   the PDE infinite-cost conclusion from Theorem A'' compiles into the
   framework-level `BarrierTypeII` blocked certificate.

These bridges remove exhaustion, representation, normalization, and
cost-adapter defects from the survivor list. They do not, by themselves, prove
global critical tightness or local windowed \(H^1\) control.

## Remaining compact-barrier tests

After C1--C4, the compact Type II barrier has only two unresolved PDE inputs.

::::{prf:definition} C5 compact-barrier test certificates
:label: def-c5-compact-barrier-tests

For a represented repaired-gauge Type II orbit, define

```{math}
K_{L^3\mathrm{Tight}}^+
```

to mean uniform global critical tightness:

```{math}
\forall\varepsilon>0\ \exists R_\varepsilon:
\sup_{\tau\ge\tau_0}
\int_{|y|>R_\varepsilon}|V(y,\tau)|^3\,dy<\varepsilon.
```

Its defect certificate is

```{math}
K_{L^3\mathrm{Tight}}^-:
\quad
\exists\varepsilon_0>0\ \forall R>0\ \exists\tau_R\ge\tau_0:
\int_{|y|>R}|V(y,\tau_R)|^3\,dy\ge\varepsilon_0.
```

Define

```{math}
K_{\mathrm{WinH1}}^+
```

to mean uniform local windowed \(H^1\) control:

```{math}
\forall m\ge1:\quad
\sup_{n\in\mathbb N}
\int_{\tau_0+n}^{\tau_0+n+1}
\|V(\tau)\|_{H^1(B_m)}^2\,d\tau<\infty.
```

Its defect certificate is

```{math}
K_{\mathrm{WinH1}}^-:
\quad
\exists m\ge1:
\sup_{n\in\mathbb N}
\int_{\tau_0+n}^{\tau_0+n+1}
\|V(\tau)\|_{H^1(B_m)}^2\,d\tau=\infty.
```

::::

::::{prf:definition} C5 total test evaluator
:label: def-c5-total-test-evaluator

The C5 compact-barrier evaluator is ordered as follows.

1. Evaluate global critical tightness. Emit exactly one of
   \(K_{L^3\mathrm{Tight}}^+\) or \(K_{L^3\mathrm{Tight}}^-\).
2. If \(K_{L^3\mathrm{Tight}}^+\) is emitted, evaluate local windowed
   \(H^1\) control. Emit exactly one of
   \(K_{\mathrm{WinH1}}^+\) or \(K_{\mathrm{WinH1}}^-\).

The first negative output is the surviving bucket. If both positive outputs
are emitted, the candidate enters the compact Type II barrier.

::::

The PDE interpretation is:

```{math}
K_{L^3\mathrm{Tight}}^-
\quad\leadsto\quad
\text{radiative/noncompact Type II},
```

and

```{math}
K_{\mathrm{WinH1}}^-
\quad\leadsto\quad
\text{rough-core Type II}.
```

## Classification-completeness certificate

::::{prf:definition} C5 classification-completeness certificate
:label: def-c5-classcomplete-certificate

The certificate

```{math}
K_{\mathrm{ClassComplete}}^+
```

means that, for every declared Type II candidate in
\(\mathcal U_{\mathrm{II}}^{NS}\), the backend supplies the bridge package

```{math}
K_{\mathrm{TypeIIExhaust}}^+
\wedge
K_{\mathrm{RepBridge}}^+
\wedge
K_{L^3\mathrm{Norm}}^+
\wedge
K_{\mathrm{CostBridge}}^+.
```

Equivalently, every declared Type II candidate reaches the C5 total test
evaluator with a repaired-gauge orbit, positive finite critical mass, the Type
II scale branch \(K_{\mathrm{SC}_\lambda}^-\), C4-ready localized cost
regularity, and the C4 cost adapter.

This certificate is a classification-completeness certificate for the compact
Type II barrier route. It does not include \(K_{L^3\mathrm{Tight}}^+\) or
\(K_{\mathrm{WinH1}}^+\); those are exactly the two remaining branch tests.

::::

## Compact-positive lemma

::::{prf:lemma} C5 compact-positive branch is suppressed
:label: lem-c5-compact-positive-branch-suppressed

Assume \(K_{\mathrm{ClassComplete}}^+\) and
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\) for the declared backend. If a declared
Type II candidate emits

```{math}
K_{L^3\mathrm{Tight}}^+
\qquad\text{and}\qquad
K_{\mathrm{WinH1}}^+,
```

then it is suppressed by the compact Type II barrier:

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}.
```

::::

:::{prf:proof}
By \(K_{\mathrm{ClassComplete}}^+\), C1 routes the candidate through
\(K_{\mathrm{SC}_\lambda}^-\), C2 gives a repaired-gauge orbit \((V,P,a,b)\),
C3 gives \(K_{L^3\mathrm{Norm}}^+\), and C4 gives both the localized cost
regularity required by the C4 theorem and \(K_{\mathrm{CostBridge}}^+\).

The certificates \(K_{L^3\mathrm{Norm}}^+\),
\(K_{L^3\mathrm{Tight}}^+\), and \(K_{\mathrm{WinH1}}^+\) are exactly the
hypotheses of the nonzero-mass good-window compact Type II barrier from C3,
which is Theorem A'' with exact unit normalization replaced by the positive
finite critical-mass annulus. Hence

```{math}
\int_{\tau_0}^{\infty}
\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
```

C4 compiles this PDE divergence into
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\). Combining
\(K_{\mathrm{SC}_\lambda}^-\) with
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) emits
\(K_{\mathrm{SC}_\lambda}^{\sim}\) only if the NS applicability bridge
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\) is available. Without that bridge, the
compact-positive branch emits \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) but
not the post-promotion suppression certificate. \(\square\)
:::

## C5 theorem

::::{prf:theorem} C5 suppressed-or-two-bucket classification
:label: thm-c5-suppressed-or-two-bucket-classification

Assume \(K_{\mathrm{ClassComplete}}^+\) and
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\) for the declared Navier-Stokes Type II
backend. Then every declared Type II candidate emits exactly one ordered output
from the following list:

1. **suppressed compact Type II:** \(K_{\mathrm{SC}_\lambda}^{\sim}\);
2. **radiative/noncompact Type II:** \(K_{L^3\mathrm{Tight}}^-\);
3. **rough-core Type II:** \(K_{\mathrm{WinH1}}^-\), after
   \(K_{L^3\mathrm{Tight}}^+\) has been emitted.

::::

:::{prf:proof}
Fix a declared Type II candidate. By \(K_{\mathrm{ClassComplete}}^+\), it
reaches the ordered C5 total test evaluator.

First evaluate tightness. If the evaluator emits
\(K_{L^3\mathrm{Tight}}^-\), the candidate is in outcome 2. If it emits
\(K_{L^3\mathrm{Tight}}^+\), evaluate windowed \(H^1\) control. If the
evaluator emits \(K_{\mathrm{WinH1}}^-\), the candidate is in outcome 3. If it
emits \(K_{\mathrm{WinH1}}^+\), Lemma
{prf:ref}`lem-c5-compact-positive-branch-suppressed` gives
\(K_{\mathrm{SC}_\lambda}^{\sim}\), so the candidate is in outcome 1.

The outputs are exhaustive by Definition
{prf:ref}`def-c5-total-test-evaluator`. They are disjoint as ordered outputs:
\(K_{L^3\mathrm{Tight}}^-\) stops the evaluator before the windowed \(H^1\)
check, \(K_{\mathrm{WinH1}}^-\) is emitted only after
\(K_{L^3\mathrm{Tight}}^+\), and the suppressed output is emitted only after
both positive tests. \(\square\)
:::

## Finite-cost convention

In C5, **finite-cost** means that the declared Type II cost used by the active
`BarrierTypeII` backend is finite on the candidate branch:

```{math}
\int_{\tau_0}^{\infty}\mathfrak C_{\mathrm{II}}(\tau)\,d\tau<\infty.
```

In the default Navier-Stokes backend of C4 this is the same as finite localized
PDE cost because \(\mathfrak C_{\mathrm{II}}^{NS}=\tilde{\mathfrak D}_{R_0}\).
The finite-cost assumption is not used to force the ordered output; it records
that the remaining non-suppressed branch is a finite-cost survivor. The ordered
output itself comes from Theorem {prf:ref}`thm-c5-suppressed-or-two-bucket-classification`.

## Finite-cost corollary

::::{prf:corollary} C5 two-bucket finite-cost classification
:label: cor-c5-two-bucket-finite-cost-classification

Assume \(K_{\mathrm{ClassComplete}}^+\) and the NS-valid promotion certificate
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\). Let a declared Type II candidate be
finite-cost and non-suppressed, meaning the output
\(K_{\mathrm{SC}_\lambda}^{\sim}\) is not emitted for that candidate. Then the
candidate emits at least one of

```{math}
K_{L^3\mathrm{Tight}}^-,
\qquad
K_{\mathrm{WinH1}}^-.
```

Equivalently, every finite-cost non-suppressed declared Type II candidate is
either radiative/noncompact or rough-core.

::::

:::{prf:proof}
Apply Theorem
{prf:ref}`thm-c5-suppressed-or-two-bucket-classification`. Since the candidate
is assumed non-suppressed, outcome 1 is unavailable. Therefore the ordered
output is outcome 2 or outcome 3. These are exactly
\(K_{L^3\mathrm{Tight}}^-\) and \(K_{\mathrm{WinH1}}^-\). The finite-cost
hypothesis is retained to identify the survivor as a finite-cost survivor, but
the certificate exclusion of the suppressed case uses the explicit
non-suppression assumption. \(\square\)
:::

## What C5 discharges

C5 removes the following survivor mechanisms from the finite-cost
classification ledger:

- backend exhaustion failure, once \(K_{\mathrm{TypeIIExhaust}}^+\) is
  available;
- missing repaired-gauge PDE representation, once \(K_{\mathrm{RepBridge}}^+\)
  is available;
- ambiguous \(L^3\)-normalization failure, once \(K_{L^3\mathrm{Norm}}^+\) is
  available;
- cost-adapter failure, once \(K_{\mathrm{CostBridge}}^+\) is available.

After C5, the only finite-cost non-suppressed Type II buckets left in the
declared backend are:

```{math}
\text{radiative/noncompact}
\quad\vee\quad
\text{rough-core}.
```

Thus C6 and C7 have precise targets:

```{math}
K_{\mathrm{RadBlk}}^+
\quad\text{and}\quad
K_{\mathrm{RoughCoreBlk}}^+.
```
