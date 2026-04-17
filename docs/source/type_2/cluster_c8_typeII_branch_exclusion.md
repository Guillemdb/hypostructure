# C8 Type II branch exclusion theorem

This note implements C8 in the classification-completeness program. It is the
assembly theorem after C1--C7:

```{math}
K_{\mathrm{ClassComplete}}^+
\wedge
K_{\mathrm{RadBlk}}^+
\wedge
K_{\mathrm{RoughCoreBlk}}^+
\wedge
K_{\mathrm{NS\text{-}UPTypeII}}^+
\Longrightarrow
\text{no non-suppressed declared Type II branch remains}.
```

The conclusion is Type II-specific. It says that the declared NS3D Type II
backend has no admissible **unresolved** Type II branch after the C-series
certificates are applied. It does not rule out Type I behavior, continuation
issues outside the declared Type II backend, or any global Navier-Stokes route
not represented by the declared backend.

## Scope

All C8 certificates are evaluated over the declared Type II universe
\(\mathcal U_{\mathrm{II}}^{NS}\) from C1. The theorem is vacuous if this
universe is empty. Otherwise, every statement below is pointwise in a declared
candidate

```{math}
\omega\in\mathcal U_{\mathrm{II}}^{NS}.
```

This pointwise convention matters because the C5 evaluator emits ordered
outputs candidate-by-candidate.

## Blocker certificates

C8 uses the exact positive certificates consumed by the C5 ordered evaluator.

::::{prf:definition} Pointwise C8 branch blockers
:label: def-c8-pointwise-branch-blockers

For a declared repaired-gauge Type II candidate \(\omega\), the certificate

```{math}
K_{\mathrm{RadBlk}}^+(\omega)
```

means that the C6 radiative blocker has emitted the positive tightness payload

```{math}
K_{L^3\mathrm{Tight}}^+(\omega).
```

Thus the radiative/noncompact output
\(K_{L^3\mathrm{Tight}}^-(\omega)\) from C5 is unavailable whenever
\(K_{\mathrm{RadBlk}}^+(\omega)\) is emitted.

For the same candidate, the certificate

```{math}
K_{\mathrm{RoughCoreBlk}}^+(\omega)
```

means that the C7 rough-core blocker has emitted the positive windowed-control
payload

```{math}
K_{\mathrm{WinH1}}^+(\omega).
```

Thus the rough-core output \(K_{\mathrm{WinH1}}^-(\omega)\) from C5 is
unavailable whenever \(K_{\mathrm{RoughCoreBlk}}^+(\omega)\) is emitted.

::::

The C6 note realizes \(K_{\mathrm{RadBlk}}^+(\omega)\) by proving
\(K_{L^3\mathrm{Tight}}^+(\omega)\). The C7 notes realize
\(K_{\mathrm{RoughCoreBlk}}^+(\omega)\) by proving
\(K_{\mathrm{WinH1}}^+(\omega)\), either directly from the windowed \(H^1\)
bridge or after the ordered upstream rough-core defects have been discharged.

::::{prf:definition} Universal C8 branch blockers
:label: def-c8-universal-branch-blockers

The global certificates used in the C8 theorem are universal closures of the
pointwise blockers:

```{math}
K_{\mathrm{RadBlk}}^+
:\Longleftrightarrow
\forall\omega\in\mathcal U_{\mathrm{II}}^{NS},\quad
K_{\mathrm{RadBlk}}^+(\omega).
```

and

```{math}
K_{\mathrm{RoughCoreBlk}}^+
:\Longleftrightarrow
\forall\omega\in\mathcal U_{\mathrm{II}}^{NS},\quad
K_{\mathrm{RoughCoreBlk}}^+(\omega).
```

::::

These universal certificates are stronger than saying that C6 or C7 has a
conditional route available. They assert that the route has actually been
discharged for every declared candidate in the backend.

## Admissible unresolved Type II branch

::::{prf:definition} Admissible unresolved Type II branch
:label: def-c8-admissible-unresolved-typeII-branch

A declared Type II candidate
\(\omega\in\mathcal U_{\mathrm{II}}^{NS}\) is an **admissible unresolved Type II
branch** if it reaches the C5 evaluator under \(K_{\mathrm{ClassComplete}}^+\)
and does not emit the suppressed Type II output

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}(\omega).
```

Equivalently, it is a declared Type II branch that survives the compact
`BarrierTypeII` suppression route and, when
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\) is available, the corresponding
NS-valid post-promotion suppression route.

::::

This definition is deliberately narrower than "Navier-Stokes singularity." It
only refers to the declared Type II backend and the C-series certificate stack.

## C8 theorem

::::{prf:theorem} C8 Type II branch exclusion
:label: thm-c8-typeII-branch-exclusion

Assume the declared NS3D repaired-gauge Type II backend emits the universal
certificates

```{math}
K_{\mathrm{ClassComplete}}^+,
\qquad
K_{\mathrm{RadBlk}}^+,
\qquad
K_{\mathrm{RoughCoreBlk}}^+,
\qquad
K_{\mathrm{NS\text{-}UPTypeII}}^+.
```

Then every declared Type II candidate \(\omega\in\mathcal U_{\mathrm{II}}^{NS}\)
emits

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}(\omega).
```

Consequently, there is no admissible unresolved Type II branch in the declared
backend.

::::

:::{prf:proof}
Fix a declared Type II candidate
\(\omega\in\mathcal U_{\mathrm{II}}^{NS}\). By
\(K_{\mathrm{ClassComplete}}^+\), the candidate reaches the C5 ordered
evaluator. C5 gives exactly one ordered output:

1. suppressed compact Type II: \(K_{\mathrm{SC}_\lambda}^{\sim}(\omega)\);
2. radiative/noncompact Type II: \(K_{L^3\mathrm{Tight}}^-(\omega)\);
3. rough-core Type II: \(K_{\mathrm{WinH1}}^-(\omega)\), after
   \(K_{L^3\mathrm{Tight}}^+(\omega)\) has been emitted.

By the universal certificate \(K_{\mathrm{RadBlk}}^+\) and Definition
{prf:ref}`def-c8-universal-branch-blockers`, the pointwise blocker
\(K_{\mathrm{RadBlk}}^+(\omega)\) is available. By Definition
{prf:ref}`def-c8-pointwise-branch-blockers`, this emits
\(K_{L^3\mathrm{Tight}}^+(\omega)\). Since the C5 tightness evaluator emits
exactly one of \(K_{L^3\mathrm{Tight}}^+(\omega)\) and
\(K_{L^3\mathrm{Tight}}^-(\omega)\), the radiative/noncompact output is
unavailable.

By the universal certificate \(K_{\mathrm{RoughCoreBlk}}^+\) and Definition
{prf:ref}`def-c8-universal-branch-blockers`, the pointwise blocker
\(K_{\mathrm{RoughCoreBlk}}^+(\omega)\) is available. By Definition
{prf:ref}`def-c8-pointwise-branch-blockers`, this emits
\(K_{\mathrm{WinH1}}^+(\omega)\). Since the C5 windowed \(H^1\) evaluator emits
exactly one of \(K_{\mathrm{WinH1}}^+(\omega)\) and
\(K_{\mathrm{WinH1}}^-(\omega)\), the rough-core output is unavailable.

The only remaining C5 ordered output is therefore
\(K_{\mathrm{SC}_\lambda}^{\sim}(\omega)\). Since \(\omega\) was arbitrary,
every declared Type II candidate is suppressed. By Definition
{prf:ref}`def-c8-admissible-unresolved-typeII-branch`, no admissible unresolved
Type II branch remains. \(\square\)
:::

## Finite-cost version

::::{prf:corollary} C8 finite-cost branch exclusion
:label: cor-c8-finite-cost-branch-exclusion

Under the hypotheses of Theorem
{prf:ref}`thm-c8-typeII-branch-exclusion`, there is no finite-cost
non-suppressed declared Type II candidate.

::::

:::{prf:proof}
Theorem {prf:ref}`thm-c8-typeII-branch-exclusion` shows that every declared
Type II candidate emits \(K_{\mathrm{SC}_\lambda}^{\sim}\). Hence no declared
candidate is non-suppressed, regardless of whether its Type II cost is finite.
\(\square\)
:::

## What C8 discharges

C8 is the final Type II-specific C-series assembly theorem.

- C1--C4 give \(K_{\mathrm{ClassComplete}}^+\), so every declared Type II
  candidate reaches the C5 evaluator with the representation, normalization,
  and cost-adapter defects discharged.
- C5 says the only ordered outcomes are suppression, radiative/noncompact, or
  rough-core.
- C6 emits \(K_{\mathrm{RadBlk}}^+\) only after it supplies the positive
  tightness payload \(K_{L^3\mathrm{Tight}}^+\) for every declared candidate.
- C7 emits \(K_{\mathrm{RoughCoreBlk}}^+\) only after it supplies the positive
  windowed-control payload \(K_{\mathrm{WinH1}}^+\) for every declared
  candidate.

Therefore the declared Type II backend has no unresolved Type II branch left.
The remaining work after C8 is not another formal C-series assembly step; it is
to discharge the upstream hypotheses behind C6/C7 in any concrete backend where
they have not already been certified.
