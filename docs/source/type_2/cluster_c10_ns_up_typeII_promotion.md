# C10 NS3D Type II promotion bridge

This note implements C10 in the Type II classification-completeness program.
It replaces the informal sentence

```{math}
\text{apply generic } \mathrm{UP}\text{-}\mathrm{TypeII}\text{ to NS3D}
```

by an explicit Navier-Stokes promotion certificate.

C10 is a certificate-interface theorem. It does not prove the analytic
Navier-Stokes barrier mechanism. Instead, it states the exact payload that must
be supplied before a blocked NS3D Type II branch may be treated as suppressed
by the scale interface.

The generic proof-object
[../proofs/proof-mt-up-type-ii.md](../proofs/proof-mt-up-type-ii.md)
is written for semilinear heat-type parabolic models. It uses a scalar
energy-supercritical heat equation, a Merle-Zaag-type localized monotonicity
formula, and a model renormalization-cost barrier. Those hypotheses are not
automatic consequences of the 3D Navier-Stokes dataset route.

Therefore the NS3D Type II stack has two distinct outputs:

```{math}
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}
```

from the declared Navier-Stokes `BarrierTypeII` evaluator, and

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}
```

only after the promotion payload in this note is supplied.

## Promotion target

::::{prf:definition} NS3D Type II promotion target
:label: def-c10-ns3d-typeii-promotion-target

For a declared repaired-gauge Type II candidate
\(\omega\in\mathcal U_{\mathrm{II}}^{NS}\), the target certificate

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}(\omega)
```

means that the candidate is suppressed at the scale interface in the same typed
sense as the framework output of `UP-TypeII`: it is no longer an admissible
unresolved Type II continuation-failure branch after the scale-check route has
processed

```{math}
K_{\mathrm{SC}_\lambda}^-(\omega)
\wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega).
```

This target is stronger than the blocked certificate. The blocked certificate
only says that the candidate hits the Type II barrier evaluator. The suppressed
certificate says that the barrier evaluator is sound for the NS3D declared
backend and may be used by downstream classification theorems as a resolved
Type II branch.

::::

## Primitive promotion payload

::::{prf:definition} NS3D promotion payload
:label: def-c10-ns3d-promotion-payload

The certificate

```{math}
K_{\mathrm{NSPromPayload}}^+(\omega)
```

is the conjunction of the following candidate-level witnesses.

1. **Route admissibility**

   ```{math}
   K_{\mathrm{PromRoute}}^+(\omega)
   ```

   asserts that \(\omega\) lies in the declared backend universe
   \(\mathcal U_{\mathrm{II}}^{NS}\), carries the C1 Type II route
   \(K_{\mathrm{TypeIIRoute}}^+(\omega)\), and has not left the scale-check
   branch before the Type II barrier is evaluated.

2. **Blocked-barrier admissibility**

   ```{math}
   K_{\mathrm{BarrierAdmiss}}^+(\omega)
   ```

   asserts that the active `BarrierTypeII` evaluator is the declared NS3D
   evaluator of C4, with identity cost

   ```{math}
   \mathfrak C_{\mathrm{II}}^{NS}
   =
   \tilde{\mathfrak D}_{R_0},
   ```

   and that the emitted
   \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega)\) is the evaluator's
   official blocked output for this candidate.

3. **Scale-route stability**

   ```{math}
   K_{\mathrm{ScaleRouteStable}}^+(\omega)
   ```

   asserts that \(K_{\mathrm{SC}_\lambda}^-(\omega)\) remains the active
   scale-interface diagnosis after the repaired-gauge representation, cost
   bridge, and barrier evaluation are attached. In particular, the candidate is
   not reclassified as Type I, continuation-success, boundary/open-system
   failure, or non-NS3D-domain failure during the C1--C4 bridge route.

4. **No barrier leakage**

   ```{math}
   K_{\mathrm{NoBarrierLeak}}^+(\omega)
   ```

   asserts that every declared continuation of the same candidate branch that
   preserves
   \(K_{\mathrm{SC}_\lambda}^-(\omega)\) and
   \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega)\) is routed back into the
   same blocked Type II barrier state. Equivalently, there is no typed
   downstream escape from the blocked Type II state to a non-suppressed Type II
   survivor without first triggering one of the ordered C1--C7 defect
   certificates.

5. **Promotion soundness**

   ```{math}
   K_{\mathrm{PromotionSound}}^+(\omega)
   ```

   asserts that the backend supplies a typed promotion morphism

   ```{math}
   \Pi_{\mathrm{NS,II}}(\omega):
   K_{\mathrm{SC}_\lambda}^-(\omega)
   \wedge
   K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega)
   \Longrightarrow
   K_{\mathrm{SC}_\lambda}^{\sim}(\omega).
   ```

   The morphism must be obtained by one of the following two alternatives.

   **Generic verification route.** The hypotheses of the framework theorem
   `UP-TypeII` have been verified for this NS3D repaired-gauge candidate, after
   translating the NS3D variables and the declared identity cost into the
   theorem's typed inputs.

   **NS-specific replacement route.** A Navier-Stokes-specific replacement
   theorem has been supplied with the same input/output certificate interface.

   In either case, the morphism must be internal to the declared NS3D backend
   and must use the same meaning of \(K_{\mathrm{SC}_\lambda}^{\sim}\) as
   Definition {prf:ref}`def-c10-ns3d-typeii-promotion-target`.

The aggregate payload is

```{math}
K_{\mathrm{NSPromPayload}}^+(\omega)
:=
K_{\mathrm{PromRoute}}^+(\omega)
\wedge
K_{\mathrm{BarrierAdmiss}}^+(\omega)
\wedge
K_{\mathrm{ScaleRouteStable}}^+(\omega)
\wedge
K_{\mathrm{NoBarrierLeak}}^+(\omega)
\wedge
K_{\mathrm{PromotionSound}}^+(\omega).
```

::::

## The NS-UPTypeII certificate

::::{prf:definition} NS-valid UP-TypeII certificate
:label: def-c10-ns-valid-uptypeii

The certificate

```{math}
K_{\mathrm{NS\text{-}UPTypeII}}^+
```

means that the NS3D promotion payload is available uniformly on the declared
Type II universe:

```{math}
\forall \omega\in\mathcal U_{\mathrm{II}}^{NS},
\qquad
K_{\mathrm{NSPromPayload}}^+(\omega).
```

The pointwise version \(K_{\mathrm{NS\text{-}UPTypeII}}^+(\omega)\) means
\(K_{\mathrm{NSPromPayload}}^+(\omega)\) for the single candidate \(\omega\).

This definition deliberately does not say that the generic heat-model
`UP-TypeII` theorem automatically applies to Navier-Stokes. It says exactly
which payload is required before a Navier-Stokes Type II branch may use the
same output certificate.

::::

## Generic `UP-TypeII` admissibility route

The generic route in Definition {prf:ref}`def-c10-ns3d-promotion-payload` is
not a name-only assertion. It must verify the actual hypotheses of the formal
framework theorem {prf:ref}`mt-up-type-ii`. C11 implements the corresponding
payload compiler:
[cluster_c11_generic_up_typeII_payload_discharge.md](cluster_c11_generic_up_typeII_payload_discharge.md).

::::{prf:definition} Generic-UP admissibility payload for NS3D
:label: def-c10-generic-up-admissibility-payload

For a declared candidate \(\omega\), the certificate

```{math}
K_{\mathrm{GenericUPTypeIIAdmiss}}^+(\omega)
```

means that the following translation witnesses are available.

1. **Formal theorem anchor**

   ```{math}
   K_{\mathrm{UPTypeIIAnchor}}^+(\omega)
   ```

   asserts that the promotion is being made by the formal theorem
   {prf:ref}`mt-up-type-ii`, with implemented proof
   [../proofs/proof-mt-up-type-ii.md](../proofs/proof-mt-up-type-ii.md), not by
   an informal analogy.

2. **Hypothesis translation**

   ```{math}
   K_{\mathrm{UPTypeIIHypTrans}}^+(\omega)
   ```

   asserts that the declared NS3D repaired-gauge branch supplies the three
   hypotheses listed in {prf:ref}`mt-up-type-ii`:

   ```{math}
   \begin{aligned}
   &\text{supercritical scale diagnosis corresponding to }
     K_{\mathrm{SC}_\lambda}^-(\omega),\\
   &\text{a declared Type II concentration scenario with bounded physical
     energy on the branch},\\
   &\text{a localized energy monotonicity formula at the active scale }
     \lambda(t).
   \end{aligned}
   ```

   The monotonicity formula must be an NS3D statement strong enough to play the
   role of the formal theorem's localized energy monotonicity input. Ordinary
   global energy dissipation alone is not sufficient unless the translation
   proof identifies it with the required localized monotonicity object.

3. **Barrier predicate translation**

   ```{math}
   K_{\mathrm{UPTypeIICostTrans}}^+(\omega)
   ```

   asserts that the C4 blocked certificate
   \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega)\) is exactly the blocked
   `BarrierTypeII` input used by {prf:ref}`mt-up-type-ii`, after translating the
   NS3D identity cost

   ```{math}
   \mathfrak C_{\mathrm{II}}^{NS}
   =
   \tilde{\mathfrak D}_{R_0}
   ```

   into the theorem's renormalization-cost integral.

4. **Domain embedding**

   ```{math}
   K_{\mathrm{UPTypeIIDomainEmb}}^+(\omega)
   ```

   asserts that the repaired-gauge NS3D orbit and its scale/gauge variables
   embed into the parabolic Hypostructure domain expected by the formal
   theorem, without changing the meaning of
   \(K_{\mathrm{SC}_\lambda}^-\),
   \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\), or
   \(K_{\mathrm{SC}_\lambda}^{\sim}\).

5. **Conclusion import**

   ```{math}
   K_{\mathrm{UPTypeIIConclusionImport}}^+(\omega)
   ```

   asserts that the theorem's output certificate is the same scale-interface
   target fixed in Definition
   {prf:ref}`def-c10-ns3d-typeii-promotion-target`.

The aggregate certificate is

```{math}
K_{\mathrm{GenericUPTypeIIAdmiss}}^+(\omega)
:=
K_{\mathrm{UPTypeIIAnchor}}^+(\omega)
\wedge
K_{\mathrm{UPTypeIIHypTrans}}^+(\omega)
\wedge
K_{\mathrm{UPTypeIICostTrans}}^+(\omega)
\wedge
K_{\mathrm{UPTypeIIDomainEmb}}^+(\omega)
\wedge
K_{\mathrm{UPTypeIIConclusionImport}}^+(\omega).
```

::::

::::{prf:lemma} Generic-UP admissibility licenses the formal `UP-TypeII` theorem
:label: lem-c10-generic-up-admissibility-licenses-up-typeii

Assume

```{math}
K_{\mathrm{SC}_\lambda}^-(\omega),
\qquad
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega),
\qquad
K_{\mathrm{GenericUPTypeIIAdmiss}}^+(\omega).
```

Then the formal theorem {prf:ref}`mt-up-type-ii` may be applied to \(\omega\),
and it emits

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}(\omega).
```

::::

:::{prf:proof}
The anchor witness selects the formal theorem {prf:ref}`mt-up-type-ii`. The
hypothesis-translation witness supplies the theorem's three analytic
hypotheses: supercritical scaling, bounded-energy Type II concentration, and a
localized energy monotonicity formula. The cost-translation witness identifies
the C4 blocked certificate with the theorem's `BarrierTypeII` blocked input.
The domain-embedding witness places the repaired-gauge NS3D branch in the
theorem's parabolic Hypostructure domain. The conclusion-import witness
identifies the theorem's output with Definition
{prf:ref}`def-c10-ns3d-typeii-promotion-target`.

Therefore the formal theorem consumes
\(K_{\mathrm{SC}_\lambda}^-(\omega)\wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega)\) and emits
\(K_{\mathrm{SC}_\lambda}^{\sim}(\omega)\) for this translated candidate.
\(\square\)
:::

## Promotion theorem

::::{prf:theorem} C10 pointwise NS3D Type II promotion
:label: thm-c10-pointwise-ns3d-typeii-promotion

Let \(\omega\in\mathcal U_{\mathrm{II}}^{NS}\). Assume

```{math}
K_{\mathrm{SC}_\lambda}^-(\omega),
\qquad
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega),
\qquad
K_{\mathrm{NSPromPayload}}^+(\omega).
```

Then

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}(\omega)
```

is emitted.

::::

:::{prf:proof}
Expand \(K_{\mathrm{NSPromPayload}}^+(\omega)\). The route admissibility and
scale-route-stability witnesses ensure that \(\omega\) is being evaluated in
the declared Type II scale branch and that
\(K_{\mathrm{SC}_\lambda}^-(\omega)\) is the active scale diagnosis. The
blocked-barrier-admissibility witness ensures that
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega)\) is the official output of
the declared NS3D `BarrierTypeII` evaluator, not an external or incomparable
cost statement. The no-barrier-leakage witness rules out a typed downstream
escape from the blocked Type II state to a non-suppressed Type II survivor
inside the declared backend.

It remains only to justify the transition from blocked to suppressed.
By \(K_{\mathrm{PromotionSound}}^+(\omega)\), the backend supplies the typed
promotion morphism \(\Pi_{\mathrm{NS,II}}(\omega)\). Applying this morphism to
the two available inputs
\(K_{\mathrm{SC}_\lambda}^-(\omega)\) and
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega)\) emits
\(K_{\mathrm{SC}_\lambda}^{\sim}(\omega)\).

The two construction routes for \(\Pi_{\mathrm{NS,II}}\) are not required to be
disjoint; either is sufficient. In the generic construction route, the morphism
\(\Pi_{\mathrm{NS,II}}\) is precisely the one produced by Lemma
{prf:ref}`lem-c10-generic-up-admissibility-licenses-up-typeii`.
Thus the candidate emits the desired suppressed scale-interface certificate.
\(\square\)
:::

::::{prf:corollary} C10 uniform NS3D Type II promotion
:label: cor-c10-uniform-ns3d-typeii-promotion

Assume \(K_{\mathrm{NS\text{-}UPTypeII}}^+\). Then for every declared Type II
candidate \(\omega\in\mathcal U_{\mathrm{II}}^{NS}\),

```{math}
K_{\mathrm{SC}_\lambda}^-(\omega)
\wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega)
\Longrightarrow
K_{\mathrm{SC}_\lambda}^{\sim}(\omega).
```

::::

:::{prf:proof}
By Definition {prf:ref}`def-c10-ns-valid-uptypeii`,
\(K_{\mathrm{NSPromPayload}}^+(\omega)\) is available for every
\(\omega\in\mathcal U_{\mathrm{II}}^{NS}\). Apply Theorem
{prf:ref}`thm-c10-pointwise-ns3d-typeii-promotion` pointwise. \(\square\)
:::

## Failure ledger

::::{prf:definition} C10 ordered promotion defects
:label: def-c10-ordered-promotion-defects

If \(K_{\mathrm{SC}_\lambda}^-\) and
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) are present but the promoted
certificate \(K_{\mathrm{SC}_\lambda}^{\sim}\) is not emitted, the C10
promotion evaluator records the first missing payload in the following ordered
list:

```{math}
K_{\mathrm{PromRoute}}^-,
\quad
K_{\mathrm{BarrierAdmiss}}^-,
\quad
K_{\mathrm{ScaleRouteStable}}^-,
\quad
K_{\mathrm{NoBarrierLeak}}^-,
\quad
K_{\mathrm{PromotionSound}}^-.
```

These are framework/promotion defects, not additional PDE survivor buckets.

::::

::::{prf:lemma} C10 defects are not Type II survivor mechanisms
:label: lem-c10-defects-not-survivors

Assume a candidate has already emitted

```{math}
K_{\mathrm{SC}_\lambda}^-,
\qquad
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```

If C10 emits one of the ordered defects in Definition
{prf:ref}`def-c10-ordered-promotion-defects`, then the candidate is not being
classified as radiative/noncompact or rough-core by C10. The output records a
missing promotion-interface payload.

::::

:::{prf:proof}
The radiative/noncompact and rough-core buckets are C5 outputs:
\(K_{L^3\mathrm{Tight}}^-\) and \(K_{\mathrm{WinH1}}^-\). The C10 evaluator is
entered only after the scale defect and blocked Type II barrier certificate are
already present. Its ordered defects concern route admissibility, barrier
admissibility, route stability, leakage, or promotion soundness. None of these
certificates asserts failure of tightness or failure of windowed local
\(H^1\)-control. Hence a C10 defect is a promotion-interface gap, not a new
PDE Type II bucket. \(\square\)
:::

## Relation to C4, C5, C8, and C9

The C-series should now be read with the following separation.

```{math}
\text{C4}
\quad\Longrightarrow\quad
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}
```

inside the declared NS3D backend when the compact PDE theorem gives infinite
localized renormalization cost.

```{math}
\text{C10}
\quad\Longrightarrow\quad
\left(
K_{\mathrm{SC}_\lambda}^-
\wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}
\Rightarrow
K_{\mathrm{SC}_\lambda}^{\sim}
\right)
```

once \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) is supplied.

Thus C5 and C8 may use the suppressed output only under the C10 promotion
payload. Without C10, the strongest unconditional compact conclusion is the
blocked barrier certificate \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\).

## What remains open

C10 makes the promotion interface rigorous, but it does not prove the analytic
Navier-Stokes promotion morphism by itself. The remaining mathematical task is
to discharge \(K_{\mathrm{PromotionSound}}^+\) by constructing
\(\Pi_{\mathrm{NS,II}}\) through one of the two accepted routes:

1. verify every hypothesis of the generic `UP-TypeII` theorem after translating
   the declared NS3D repaired-gauge backend into that theorem's input format,
   i.e. prove \(K_{\mathrm{GenericUPTypeIIAdmiss}}^+\);
2. prove a Navier-Stokes-specific replacement theorem that consumes
   \(K_{\mathrm{SC}_\lambda}^-\wedge
   K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) and emits
   \(K_{\mathrm{SC}_\lambda}^{\sim}\) with the meaning fixed in Definition
   {prf:ref}`def-c10-ns3d-typeii-promotion-target`.

Until one of these is supplied, the Type II stack rigorously proves blocked
compact Type II, not automatic post-UP suppression for NS3D.
