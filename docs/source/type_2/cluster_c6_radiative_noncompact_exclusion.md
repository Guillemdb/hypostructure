# C6 radiative/noncompact branch exclusion

This note implements C6 in the classification-completeness program. It targets
the radiative/noncompact survivor from C5:

```{math}
K_{L^3\mathrm{Tight}}^-.
```

The output certificate is

```{math}
K_{\mathrm{RadBlk}}^+,
```

meaning that the radiative/noncompact Type II branch is blocked. In the compact
Type II proof stack this is exactly the same as emitting uniform critical
tightness:

```{math}
K_{\mathrm{RadBlk}}^+
\equiv
K_{L^3\mathrm{Tight}}^+.
```

C6 is not a global regularity theorem and does not prove no-radiation from
bare Navier-Stokes data. It gives explicit no-radiation certificates that are
sufficient to rule out the radiative/noncompact bucket in the declared NS3D
repaired-gauge Type II backend.

## Target certificate

::::{prf:definition} Radiative branch blocker
:label: def-radblk-certificate

For a represented repaired-gauge Type II orbit \(V(\tau)\), define

```{math}
K_{\mathrm{RadBlk}}^+
```

to mean uniform global critical \(L^3\)-tightness:

```{math}
\forall\varepsilon>0\ \exists R_\varepsilon:
\sup_{\tau\ge\tau_0}
\int_{|y|>R_\varepsilon}|V(y,\tau)|^3\,dy<\varepsilon.
```

Equivalently,

```{math}
K_{\mathrm{RadBlk}}^+ := K_{L^3\mathrm{Tight}}^+.
```

Its negation in the ordered C5 evaluator is the radiative/noncompact bucket

```{math}
K_{L^3\mathrm{Tight}}^-.
```

::::

## Route 1: global compactness

The cleanest no-radiation certificate is global precompactness in the critical
topology.

::::{prf:definition} Global \(L^3\)-compactness certificate
:label: def-global-l3-compactness-certificate

The certificate

```{math}
K_{\mathrm{GlobComp}_{L^3}}^+
```

means that the renormalized orbit

```{math}
\mathcal O:=\{V(\tau):\tau\ge\tau_0\}
```

is contained in a compact subset of \(L^3(\mathbb R^3)\).

::::

::::{prf:lemma} Global \(L^3\)-compactness blocks radiation
:label: lem-global-l3-compactness-blocks-radiation

If \(K_{\mathrm{GlobComp}_{L^3}}^+\) holds, then

```{math}
K_{\mathrm{RadBlk}}^+
```

holds.

::::

:::{prf:proof}
Compact subsets of \(L^3(\mathbb R^3)\) are uniformly tight. Indeed, let
\(K\subset L^3\) be compact and fix \(\varepsilon>0\). Choose a finite
\(\varepsilon/2\)-net \(f_1,\dots,f_N\) in \(L^3\). For each \(i\), choose
\(R_i\) so that
\[
\int_{|y|>R_i}|f_i(y)|^3\,dy<(\varepsilon/2)^3.
\]
Let \(R=\max_i R_i\). For any \(f\in K\), choose \(i\) with
\(\|f-f_i\|_3<\varepsilon/2\). Then
\[
\|f\|_{L^3(|y|>R)}
\le
\|f-f_i\|_3+\|f_i\|_{L^3(|y|>R)}
<\varepsilon.
\]
Taking the cube gives uniform smallness of the \(L^3\) tail. Applying this to
the compact set containing \(\mathcal O\) gives \(K_{L^3\mathrm{Tight}}^+\),
which is \(K_{\mathrm{RadBlk}}^+\). \(\square\)
:::

## Route 2: compact core plus vanishing critical remainder

Radiation is also blocked if the orbit decomposes into a compact critical core
plus a vanishing \(L^3\) remainder.

::::{prf:definition} Compact-core vanishing-remainder certificate
:label: def-compact-core-vanishing-remainder

The certificate

```{math}
K_{\mathrm{CoreRem}}^+
```

means that there exist:

1. a compact set \(K_{\mathrm{core}}\subset L^3(\mathbb R^3)\);
2. a map \(Q:[\tau_0,\infty)\to K_{\mathrm{core}}\);
3. a remainder \(r(\tau)\in L^3(\mathbb R^3)\);

such that

```{math}
V(\tau)=Q(\tau)+r(\tau),
\qquad
\lim_{\tau\to\infty}\|r(\tau)\|_{L^3}=0.
```

It also includes finite-initial-window tightness:
for every \(T>\tau_0\), the set

```{math}
\{V(\tau):\tau\in[\tau_0,T]\}
```

is uniformly \(L^3\)-tight.

::::

::::{prf:lemma} Compact core plus vanishing remainder blocks radiation
:label: lem-core-rem-blocks-radiation

If \(K_{\mathrm{CoreRem}}^+\) holds, then

```{math}
K_{\mathrm{RadBlk}}^+
```

holds.

::::

:::{prf:proof}
Fix \(\varepsilon>0\). By compactness of \(K_{\mathrm{core}}\), Lemma
{prf:ref}`lem-global-l3-compactness-blocks-radiation` applied to
\(K_{\mathrm{core}}\) gives \(R_1\) such that
\[
\sup_{q\in K_{\mathrm{core}}}\|q\|_{L^3(|y|>R_1)}<\varepsilon/3.
\]
Since \(\|r(\tau)\|_3\to0\), choose \(T\) so that
\[
\|r(\tau)\|_3<\varepsilon/3
\qquad
\text{for all }\tau\ge T.
\]
Then for \(\tau\ge T\),
\[
\|V(\tau)\|_{L^3(|y|>R_1)}
\le
\|Q(\tau)\|_{L^3(|y|>R_1)}+\|r(\tau)\|_3
<2\varepsilon/3.
\]
By finite-initial-window tightness on \([\tau_0,T]\), choose \(R_2\) such that
\[
\sup_{\tau\in[\tau_0,T]}
\|V(\tau)\|_{L^3(|y|>R_2)}<\varepsilon.
\]
With \(R=\max(R_1,R_2)\), the \(L^3\) tail of \(V(\tau)\) is \(<\varepsilon\)
for all \(\tau\ge\tau_0\). This is \(K_{L^3\mathrm{Tight}}^+\), hence
\(K_{\mathrm{RadBlk}}^+\). \(\square\)
:::

## Route 3: no-splitting plus compact profiles

The profile-decomposition route blocks radiation by proving that all secondary
profiles and remainders vanish.

::::{prf:definition} No-splitting/no-radiation profile certificate
:label: def-nosplit-norad-profile-certificate

The certificate

```{math}
K_{\mathrm{NoSplitRad}}^+
```

means the declared profile decomposition of the Type II branch has:

1. a primary compact core family \(K_{\mathrm{core}}\subset L^3\);
2. asymptotic \(L^3\)-decoupling of core, secondary profiles, and remainder;
3. single-profile saturation, so every non-primary profile has zero critical
   \(L^3\)-mass;
4. a remainder whose \(L^3\)-norm vanishes on the renormalized tail;
5. finite-initial-window tightness.

::::

::::{prf:lemma} No-splitting/no-radiation profile certificate gives compact core remainder
:label: lem-nosplit-rad-gives-core-rem

If \(K_{\mathrm{NoSplitRad}}^+\) holds, then

```{math}
K_{\mathrm{CoreRem}}^+
```

holds.

::::

:::{prf:proof}
By single-profile saturation and nonnegative \(L^3\)-mass decoupling, every
secondary profile has zero critical \(L^3\)-mass and therefore vanishes in
\(L^3\). The profile decomposition then reduces to a primary compact core plus
a remainder whose \(L^3\)-norm tends to zero on the renormalized tail. These are
exactly the core \(Q(\tau)\), compact set \(K_{\mathrm{core}}\), and remainder
\(r(\tau)\) required by Definition
{prf:ref}`def-compact-core-vanishing-remainder`, with finite-initial-window
tightness supplied as part of \(K_{\mathrm{NoSplitRad}}^+\). Hence
\(K_{\mathrm{CoreRem}}^+\) holds. \(\square\)
:::

::::{prf:corollary} No-splitting/no-radiation blocks radiation
:label: cor-nosplit-rad-blocks-radiation

If \(K_{\mathrm{NoSplitRad}}^+\) holds, then

```{math}
K_{\mathrm{RadBlk}}^+.
```

::::

:::{prf:proof}
Lemma {prf:ref}`lem-nosplit-rad-gives-core-rem` gives
\(K_{\mathrm{CoreRem}}^+\). Lemma
{prf:ref}`lem-core-rem-blocks-radiation` then gives
\(K_{\mathrm{RadBlk}}^+\). \(\square\)
:::

## Route 4: finite library tightness

The NS3D dataset already records a profile/library package
\(K_{\mathrm{Prof}_{NS}}^+\), \(K_{\mathrm{Germ}}^+\), and
\(K_{\mathrm{CatLib}}^+\). This does not by itself imply no radiation, but it
becomes a powerful no-radiation route once the library approximation is in the
critical \(L^3\) topology and the library profiles are uniformly tight.

::::{prf:definition} Finite-library \(L^3\)-tight approximation certificate
:label: def-finite-library-tight-approx

The certificate

```{math}
K_{\mathrm{LibTightApprox}}^+
```

means:

1. the profile branch has the finite certified library package
   ```{math}
   K_{\mathrm{Prof}_{NS}}^+
   \wedge
   K_{\mathrm{Germ}}^+
   \wedge
   K_{\mathrm{CatLib}}^+;
   ```
2. the library elements form a finite family
   \(\mathcal B_{NS}^{II}=\{B_1,\dots,B_N\}\subset L^3(\mathbb R^3)\) after
   repaired-gauge realization;
3. the realized orbit is approximated by the library in the critical topology:
   ```{math}
   \forall\delta>0\ \exists \tau_\delta:
   \forall\tau\ge\tau_\delta\ \exists i(\tau)\in\{1,\dots,N\}:
   \|V(\tau)-B_{i(\tau)}\|_{L^3}<\delta;
   ```
4. the finite initial window \(\{V(\tau):\tau\in[\tau_0,\tau_\delta]\}\) is
   uniformly \(L^3\)-tight for each \(\delta>0\).

::::

::::{prf:lemma} Finite library tight approximation blocks radiation
:label: lem-finite-library-tight-approx-blocks-radiation

If \(K_{\mathrm{LibTightApprox}}^+\) holds, then

```{math}
K_{\mathrm{RadBlk}}^+.
```

::::

:::{prf:proof}
Fix \(\varepsilon>0\). Since the library is finite and each
\(B_i\in L^3(\mathbb R^3)\), there is \(R_1\) such that
\[
\max_{1\le i\le N}\|B_i\|_{L^3(|y|>R_1)}<\varepsilon/3.
\]
Choose \(\delta=\varepsilon/3\). For \(\tau\ge\tau_\delta\), pick
\(i(\tau)\) with \(\|V(\tau)-B_{i(\tau)}\|_3<\delta\). Then
\[
\|V(\tau)\|_{L^3(|y|>R_1)}
\le
\|V(\tau)-B_{i(\tau)}\|_3
+
\|B_{i(\tau)}\|_{L^3(|y|>R_1)}
<2\varepsilon/3.
\]
By finite-initial-window tightness, choose \(R_2\) so that the \(L^3\) tail is
\(<\varepsilon\) on \([\tau_0,\tau_\delta]\). With \(R=\max(R_1,R_2)\), the
orbit is uniformly \(L^3\)-tight. Hence \(K_{\mathrm{RadBlk}}^+\) holds.
\(\square\)
:::

## Route 5: tame finite-description compactness

The certificates \(K_{\mathrm{TB}_O}^+\) and \(K_{\mathrm{RepDesc}_K}^+\) are
useful when they are paired with a compact parameter realization of the profile
family. This converts tame/finite-description data into actual \(L^3\)
precompactness.

::::{prf:definition} Tame finite-description compact-realization certificate
:label: def-tame-fd-compact-realization

The certificate

```{math}
K_{\mathrm{TameFDComp}}^+
```

means:

1. \(K_{\mathrm{TB}_O}^+\) holds for the declared profile backend;
2. \(K_{\mathrm{RepDesc}_K}^+\) holds for the thin trace representation;
3. the repaired-gauge profiles are parameterized by a compact definable
   parameter set \(\Theta_{\mathrm{II}}\);
4. the realization map
   ```{math}
   \Theta_{\mathrm{II}}\ni\theta\mapsto V_\theta\in L^3(\mathbb R^3)
   ```
   is continuous;
5. the orbit satisfies \(V(\tau)=V_{\theta(\tau)}\) for some
   \(\theta(\tau)\in\Theta_{\mathrm{II}}\), up to a remainder
   \(r(\tau)\to0\) in \(L^3\).
6. finite-initial-window tightness holds: for every \(T>\tau_0\), the set
   \[
   \{V(\tau):\tau\in[\tau_0,T]\}
   \]
   is uniformly \(L^3\)-tight.

::::

::::{prf:lemma} Tame finite-description compact realization blocks radiation
:label: lem-tame-fd-compact-realization-blocks-radiation

If \(K_{\mathrm{TameFDComp}}^+\) holds, then

```{math}
K_{\mathrm{RadBlk}}^+.
```

::::

:::{prf:proof}
The continuous image of compact \(\Theta_{\mathrm{II}}\) in \(L^3\) is compact.
Thus the core family
\[
K_{\mathrm{core}}:=\{V_\theta:\theta\in\Theta_{\mathrm{II}}\}
\]
is compact in \(L^3\). The representation
\(V(\tau)=V_{\theta(\tau)}+r(\tau)\) with \(r(\tau)\to0\) in \(L^3\) is exactly
the compact-core vanishing-remainder certificate \(K_{\mathrm{CoreRem}}^+\),
because finite-initial-window tightness is included in
Definition {prf:ref}`def-tame-fd-compact-realization`. Lemma
{prf:ref}`lem-core-rem-blocks-radiation` gives
\(K_{\mathrm{RadBlk}}^+\). \(\square\)
:::

## Route 6: a-posteriori discharge of tightness INC

The framework's a-posteriori inconclusive-discharge rule can make C6 stronger
operationally. Instead of requiring no-radiation information to be available
when the tightness check is first reached, the backend may record an
inconclusive tightness certificate and discharge it later when one of the
routes above is proved.

::::{prf:definition} C6 route certificate
:label: def-c6-route-certificate

The certificate

```{math}
K_{\mathrm{C6Route}}^+
```

is emitted when at least one of the C6 route certificates is available:

```{math}
K_{\mathrm{GlobComp}_{L^3}}^+,
\quad
K_{\mathrm{CoreRem}}^+,
\quad
K_{\mathrm{NoSplitRad}}^+,
\quad
K_{\mathrm{LibTightApprox}}^+,
\quad
K_{\mathrm{TameFDComp}}^+.
```

Its payload records which route fired.

::::

::::{prf:lemma} C6 route certificate blocks radiation
:label: lem-c6-route-blocks-radiation

If \(K_{\mathrm{C6Route}}^+\) holds, then

```{math}
K_{\mathrm{RadBlk}}^+.
```

::::

:::{prf:proof}
Inspect the payload of \(K_{\mathrm{C6Route}}^+\). If it records
\(K_{\mathrm{GlobComp}_{L^3}}^+\), use Lemma
{prf:ref}`lem-global-l3-compactness-blocks-radiation`. If it records
\(K_{\mathrm{CoreRem}}^+\), use Lemma
{prf:ref}`lem-core-rem-blocks-radiation`. If it records
\(K_{\mathrm{NoSplitRad}}^+\), use Corollary
{prf:ref}`cor-nosplit-rad-blocks-radiation`. If it records
\(K_{\mathrm{LibTightApprox}}^+\), use Lemma
{prf:ref}`lem-finite-library-tight-approx-blocks-radiation`. If it records
\(K_{\mathrm{TameFDComp}}^+\), use Lemma
{prf:ref}`lem-tame-fd-compact-realization-blocks-radiation`. In every case,
\(K_{\mathrm{RadBlk}}^+\) follows. \(\square\)
:::

::::{prf:definition} Tightness inconclusive certificate
:label: def-tightness-inc-certificate

The certificate

```{math}
K_{L^3\mathrm{Tight}}^{\mathrm{inc}}
```

has obligation \(K_{L^3\mathrm{Tight}}^+\) and missing set

```{math}
\mathsf{missing}
=
\{K_{\mathrm{C6Route}}^+\}.
```

Its trace records the deferred no-radiation route search. The point of using
the aggregate certificate \(K_{\mathrm{C6Route}}^+\) is that
`UP-IncAposteriori` requires all certificates in the missing set to be supplied;
the disjunction among possible no-radiation routes is resolved before the
inconclusive tightness certificate is upgraded.

::::

::::{prf:theorem} A-posteriori tightness discharge
:label: thm-aposteriori-tightness-discharge

Assume \(K_{L^3\mathrm{Tight}}^{\mathrm{inc}}\) is present in the certificate
context. If later certificates emit \(K_{\mathrm{C6Route}}^+\), then promotion
closure emits

```{math}
K_{L^3\mathrm{Tight}}^+
```

and hence

```{math}
K_{\mathrm{RadBlk}}^+.
```

::::

:::{prf:proof}
By Lemma {prf:ref}`lem-c6-route-blocks-radiation`,
\(K_{\mathrm{C6Route}}^+\) implies \(K_{\mathrm{RadBlk}}^+\), equivalently
\(K_{L^3\mathrm{Tight}}^+\). Thus the discharge condition required by
`UP-IncAposteriori` is satisfied: the later YES certificate
\(K_{\mathrm{C6Route}}^+\) fills the singleton missing set of the inconclusive
tightness obligation. Promotion closure upgrades
\(K_{L^3\mathrm{Tight}}^{\mathrm{inc}}\) to \(K_{L^3\mathrm{Tight}}^+\), hence
to \(K_{\mathrm{RadBlk}}^+\). \(\square\)
:::

## C6 theorem

::::{prf:theorem} C6 radiative/noncompact branch exclusion
:label: thm-c6-radiative-noncompact-exclusion

For a declared NS3D repaired-gauge Type II candidate, assume at least one of:

```{math}
K_{\mathrm{GlobComp}_{L^3}}^+,
\qquad
K_{\mathrm{CoreRem}}^+,
\qquad
K_{\mathrm{NoSplitRad}}^+,
\qquad
K_{\mathrm{LibTightApprox}}^+,
\qquad
K_{\mathrm{TameFDComp}}^+.
```

Then

```{math}
K_{\mathrm{RadBlk}}^+
\equiv
K_{L^3\mathrm{Tight}}^+
```

is emitted. Consequently, the C5 radiative/noncompact output

```{math}
K_{L^3\mathrm{Tight}}^-
```

is unavailable for that candidate.

::::

:::{prf:proof}
If \(K_{\mathrm{GlobComp}_{L^3}}^+\) holds, apply Lemma
{prf:ref}`lem-global-l3-compactness-blocks-radiation`. If
\(K_{\mathrm{CoreRem}}^+\) holds, apply Lemma
{prf:ref}`lem-core-rem-blocks-radiation`. If \(K_{\mathrm{NoSplitRad}}^+\)
holds, apply Corollary {prf:ref}`cor-nosplit-rad-blocks-radiation`. In every
case \(K_{\mathrm{RadBlk}}^+\), equivalently \(K_{L^3\mathrm{Tight}}^+\), is
emitted. The two additional cases are handled by Lemma
{prf:ref}`lem-finite-library-tight-approx-blocks-radiation` and Lemma
{prf:ref}`lem-tame-fd-compact-realization-blocks-radiation`. Since the C5 evaluator emits exactly one of
\(K_{L^3\mathrm{Tight}}^+\) and \(K_{L^3\mathrm{Tight}}^-\), the negative
radiative/noncompact output is unavailable. \(\square\)
:::

## Coupling with C5

::::{prf:corollary} C5 plus C6 leaves only rough-core finite-cost survivors
:label: cor-c5-c6-rough-core-only

Assume \(K_{\mathrm{ClassComplete}}^+\). Let a declared Type II candidate be
finite-cost and non-suppressed. If C6 emits \(K_{\mathrm{RadBlk}}^+\) for that
candidate, then the candidate emits

```{math}
K_{\mathrm{WinH1}}^-.
```

Equivalently, after radiative/noncompact exclusion, every finite-cost
non-suppressed declared Type II survivor is rough-core.

::::

:::{prf:proof}
By the C5 finite-cost classification, a finite-cost non-suppressed declared
Type II candidate emits either \(K_{L^3\mathrm{Tight}}^-\) or
\(K_{\mathrm{WinH1}}^-\). C6 emits \(K_{\mathrm{RadBlk}}^+\), equivalently
\(K_{L^3\mathrm{Tight}}^+\), so the ordered C5 tightness defect
\(K_{L^3\mathrm{Tight}}^-\) is unavailable. Therefore the only remaining C5
survivor output is \(K_{\mathrm{WinH1}}^-\). \(\square\)
:::

## What C6 discharges

C6 turns no-radiation/no-splitting information into the exact certificate C5
needs:

```{math}
K_{\mathrm{RadBlk}}^+
=
K_{L^3\mathrm{Tight}}^+.
```

It gives three sufficient routes:

1. global \(L^3\)-compactness of the renormalized orbit;
2. compact core plus vanishing \(L^3\) remainder;
3. no-splitting/no-radiation profile decomposition.
4. finite certified profile library plus \(L^3\)-approximation;
5. tame finite-description compact realization;
6. a-posteriori discharge of \(K_{L^3\mathrm{Tight}}^{\mathrm{inc}}\) once the
   aggregate route certificate \(K_{\mathrm{C6Route}}^+\) is later produced.

With C6 in place, the remaining finite-cost non-suppressed Type II bucket is
only:

```{math}
K_{\mathrm{WinH1}}^-,
```

the rough-core branch targeted by C7.
