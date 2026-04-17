# C11 generic `UP-TypeII` admissibility payload discharge

This note implements the payload behind

```{math}
K_{\mathrm{GenericUPTypeIIAdmiss}}^+
```

from C10. Its purpose is precise: identify which parts of the generic
`UP-TypeII` admissibility payload are already supplied by the declared NS3D
Type II backend, and isolate the remaining analytic witness needed before the
formal theorem {prf:ref}`mt-up-type-ii` can be invoked.

The conclusion is not that the generic theorem automatically applies to 3D
Navier-Stokes. The conclusion is the conditional compiler

```{math}
K_{\mathrm{GenUPPayload},NS3D}^+
\Longrightarrow
K_{\mathrm{GenericUPTypeIIAdmiss}}^+.
```

## Formal target from C10

C10 defines

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

Once this certificate is emitted, Lemma
{prf:ref}`lem-c10-generic-up-admissibility-licenses-up-typeii` permits applying
the formal theorem {prf:ref}`mt-up-type-ii`.

## Discharge payload

::::{prf:definition} NS3D generic-UP payload
:label: def-c11-ns3d-generic-up-payload

For a declared candidate \(\omega\in\mathcal U_{\mathrm{II}}^{NS}\), define

```{math}
K_{\mathrm{GenUPPayload},NS3D}^+(\omega)
```

to be the conjunction of the following witnesses.

1. **C10 anchor registration**

   ```{math}
   K_{\mathrm{C10Anchor}}^+(\omega)
   ```

   asserts that the theorem being invoked is exactly
   {prf:ref}`mt-up-type-ii`, with target certificate
   \(K_{\mathrm{SC}_\lambda}^{\sim}(\omega)\) interpreted as in C10.

2. **Scale-supercritical route**

   ```{math}
   K_{\mathrm{ScaleSupercritRoute}}^+(\omega)
   ```

   asserts that the C1 route supplies
   \(K_{\mathrm{TypeIIRoute}}^+(\omega)\), in particular
   \(K_{\mathrm{SC}_\lambda}^-(\omega)\), and that this is the same
   supercritical scale diagnosis consumed by {prf:ref}`mt-up-type-ii`.

3. **Bounded-energy Type II concentration**

   ```{math}
   K_{\mathrm{BoundedEnergyTypeII}}^+(\omega)
   ```

   asserts that the candidate is a declared Type II concentration branch with
   bounded physical energy. Concretely, it is supplied by

   ```{math}
   K_{\mathrm{TypeIIRoute}}^+(\omega)
   \wedge
   K_{D_E}^+(\omega),
   ```

   where \(K_{\mathrm{TypeIIRoute}}^+\) gives the concentration/profile route
   and \(K_{D_E}^+\) gives the Navier-Stokes energy inequality on the physical
   branch.

4. **Localized monotonicity translation**

   ```{math}
   K_{\mathrm{NSLocMonoTrans}}^+(\omega)
   ```

   asserts that the renormalized NS3D local energy identity/Caccioppoli layer
   supplies a localized energy monotonicity object strong enough to replace the
   monotonicity hypothesis in {prf:ref}`mt-up-type-ii`.

   This is the only genuinely analytic part of the generic-UP payload. It is
   not the same as the global energy inequality \(K_{D_E}^+\). It must prove
   that all pressure, flux, transport, and repaired-gauge modulation terms in
   the localized renormalized energy identity are either nonpositive, absorbed
   into the declared barrier cost, or controlled by already declared positive
   certificates.

5. **Cost translation**

   ```{math}
   K_{\mathrm{C4CostAsUPCost}}^+(\omega)
   ```

   asserts that C4's identity `BarrierTypeII` cost

   ```{math}
   \mathfrak C_{\mathrm{II}}^{NS}
   =
   \tilde{\mathfrak D}_{R_0}
   ```

   is the renormalization-cost input used by the formal theorem.

6. **Parabolic domain embedding**

   ```{math}
   K_{\mathrm{NSParabolicDomainEmb}}^+(\omega)
   ```

   asserts that the repaired-gauge NS3D orbit belongs to the parabolic
   Hypostructure domain expected by {prf:ref}`mt-up-type-ii`, with the same
   scale, barrier, and suppression certificate semantics. It is supplied by the
   conjunction of the NS3D automation/type witness, the C1 route, and the C2
   repaired-gauge representation bridge.

7. **Conclusion identity**

   ```{math}
   K_{\mathrm{UPConclusionId}}^+(\omega)
   ```

   asserts that the output of {prf:ref}`mt-up-type-ii` is exactly the C10
   target \(K_{\mathrm{SC}_\lambda}^{\sim}(\omega)\), not a distinct
   heat-model suppression certificate.

The aggregate payload is

```{math}
\begin{aligned}
K_{\mathrm{GenUPPayload},NS3D}^+(\omega)
:={}&
K_{\mathrm{C10Anchor}}^+(\omega)
\wedge
K_{\mathrm{ScaleSupercritRoute}}^+(\omega)
\wedge
K_{\mathrm{BoundedEnergyTypeII}}^+(\omega)\\
&\wedge
K_{\mathrm{NSLocMonoTrans}}^+(\omega)
\wedge
K_{\mathrm{C4CostAsUPCost}}^+(\omega)
\wedge
K_{\mathrm{NSParabolicDomainEmb}}^+(\omega)
\wedge
K_{\mathrm{UPConclusionId}}^+(\omega).
\end{aligned}
```

::::

## Component discharge lemmas

::::{prf:lemma} C11 anchor and conclusion discharge
:label: lem-c11-anchor-conclusion-discharge

For every declared Type II candidate \(\omega\), C10 supplies

```{math}
K_{\mathrm{C10Anchor}}^+(\omega)
\wedge
K_{\mathrm{UPConclusionId}}^+(\omega).
```

::::

:::{prf:proof}
C10 defines the promotion target
\(K_{\mathrm{SC}_\lambda}^{\sim}(\omega)\) and explicitly anchors the generic
route to the formal theorem {prf:ref}`mt-up-type-ii`. Therefore the theorem
anchor and the conclusion identity are definitional consequences of C10's
promotion-interface contract. \(\square\)
:::

::::{prf:lemma} C1 and the dataset discharge scale and bounded-energy inputs
:label: lem-c11-scale-energy-discharge

Assume \(K_{\mathrm{TypeIIRoute}}^+(\omega)\) and \(K_{D_E}^+(\omega)\). Then

```{math}
K_{\mathrm{ScaleSupercritRoute}}^+(\omega)
\wedge
K_{\mathrm{BoundedEnergyTypeII}}^+(\omega).
```

::::

:::{prf:proof}
By definition of the C1 Type II route,

```{math}
K_{\mathrm{TypeIIRoute}}^+(\omega)
=
K_{C_\mu}^+(\omega)
\wedge
K_{\mathrm{SC}_\lambda}^-(\omega)
\wedge
K_{\mathrm{Prof}_{NS}}^+(\omega).
```

Thus the candidate carries the supercritical scale diagnosis consumed by the
Type II barrier route, and it is a declared concentration/profile branch.
The dataset certificate \(K_{D_E}^+\) supplies bounded physical energy through
the Navier-Stokes energy inequality. These are exactly
\(K_{\mathrm{ScaleSupercritRoute}}^+\) and
\(K_{\mathrm{BoundedEnergyTypeII}}^+\). \(\square\)
:::

::::{prf:lemma} C4 discharges generic-UP cost translation
:label: lem-c11-cost-translation-discharge

Assume \(K_{\mathrm{CostBridge}}^+(\omega)\) in the default NS3D
`BarrierTypeII` backend. Then

```{math}
K_{\mathrm{C4CostAsUPCost}}^+(\omega).
```

::::

:::{prf:proof}
C4 defines the active NS3D Type II barrier cost by identity:

```{math}
\mathfrak C_{\mathrm{II}}^{NS}
=
\tilde{\mathfrak D}_{R_0}.
```

It also proves that divergence of this PDE cost emits the framework blocked
certificate \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega)\). Therefore the
cost used by the declared NS3D `BarrierTypeII` evaluator is the
renormalization-cost input passed to the formal Type II promotion theorem.
\(\square\)
:::

::::{prf:lemma} C1--C2 discharge the parabolic domain embedding
:label: lem-c11-domain-embedding-discharge

Assume

```{math}
K_{\mathrm{Auto}}^+,
\qquad
K_{\mathrm{TypeIIRoute}}^+(\omega),
\qquad
K_{\mathrm{RepBridge}}^+(\omega).
```

Then

```{math}
K_{\mathrm{NSParabolicDomainEmb}}^+(\omega).
```

::::

:::{prf:proof}
The automation/type witness \(K_{\mathrm{Auto}}^+\) records that the NS3D
instance is treated as a parabolic transport-diffusion Hypostructure by the
declared backend. The C1 route places \(\omega\) on the Type II scale/profile
branch. The C2 representation bridge supplies the repaired-gauge orbit
\((V,P,a,b)\) with scale, center, pressure, and modulation data. Together these
data embed the candidate into the parabolic domain expected by
{prf:ref}`mt-up-type-ii` while preserving the meanings of
\(K_{\mathrm{SC}_\lambda}^-\), \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\), and
\(K_{\mathrm{SC}_\lambda}^{\sim}\). \(\square\)
:::

## Localized monotonicity witness

::::{prf:definition} NS3D localized monotonicity translation
:label: def-c11-ns-localized-monotonicity-translation

The certificate

```{math}
K_{\mathrm{NSLocMonoTrans}}^+(\omega)
```

means that there exists a localized renormalized energy functional
\(\mathcal E_{R_0}(\tau)\), compatible with the repaired-gauge orbit of
\(\omega\), and a constant \(c_0>0\) such that on the tail of the branch

```{math}
\frac{d}{d\tau}\mathcal E_{R_0}(\tau)
+
c_0\tilde{\mathfrak D}_{R_0}(\tau)
+
\mathcal R_{R_0}(\tau)
\le 0
```

in the distributional sense, where:

1. \(\tilde{\mathfrak D}_{R_0}\ge0\) is the C4 Type II barrier cost, and the
   fixed factor \(c_0>0\) is allowed because multiplying a nonnegative barrier
   cost by a positive constant does not change divergence or the blocked
   `BarrierTypeII` certificate;
2. \(\mathcal R_{R_0}\ge0\) is an optional nonnegative remainder, or
   \(\mathcal R_{R_0}=0\);
3. every pressure, cutoff-flux, transport, and modulation term in the
   renormalized local energy identity is either nonpositive, absorbed into
   \(\tilde{\mathfrak D}_{R_0}\), or controlled by a finite endpoint/tail error
   that does not invalidate the monotonicity input of {prf:ref}`mt-up-type-ii`.

If the best available statement has a finite error \(B_{R_0}(\tau)\), then the
certificate requires an explicitly corrected monotone functional

```{math}
\mathcal E_{R_0}^{\mathrm{corr}}(\tau)
:=
\mathcal E_{R_0}(\tau)
+
\int_{\tau}^{\infty} B_{R_0}(s)\,ds
```

with finite correction tail and the same conclusion.

::::

C12 proves this certificate from the corrected local energy identity and the
finite monotonicity-error certificate \(K_{\mathrm{FiniteMonoErr}}^+\):
[cluster_c12_ns_localized_monotonicity_translation.md](cluster_c12_ns_localized_monotonicity_translation.md).

::::{prf:lemma} Caccioppoli alone does not discharge generic-UP monotonicity
:label: lem-c11-caccioppoli-not-enough-for-up-monotonicity

The certificates \(K_{D_E}^+\) and \(K_{\mathrm{CaccioppoliReg}}^+\) do not by
themselves imply \(K_{\mathrm{NSLocMonoTrans}}^+\).

::::

:::{prf:proof}
\(K_{D_E}^+\) is a global physical energy inequality. It controls total
physical energy and physical dissipation, but it does not identify a localized
renormalized energy functional at the active Type II scale whose derivative is
monotone after pressure, transport, cutoff, and modulation terms are included.

\(K_{\mathrm{CaccioppoliReg}}^+\) supplies local spacetime \(H^1\)-control or
windowed gradient bounds from the renormalized local energy inequality. Such an
estimate is an a priori bound, not necessarily a monotonicity formula of the
form required by {prf:ref}`mt-up-type-ii`. Therefore these two certificates
support the construction of \(K_{\mathrm{NSLocMonoTrans}}^+\), but do not
imply it without an additional absorption/correction argument. \(\square\)
:::

## Payload compiler

::::{prf:theorem} C11 generic-UP payload compiler
:label: thm-c11-generic-up-payload-compiler

Let \(\omega\in\mathcal U_{\mathrm{II}}^{NS}\). Assume

```{math}
K_{\mathrm{Auto}}^+,
\quad
K_{\mathrm{TypeIIRoute}}^+(\omega),
\quad
K_{D_E}^+(\omega),
\quad
K_{\mathrm{RepBridge}}^+(\omega),
\quad
K_{\mathrm{CostBridge}}^+(\omega),
\quad
K_{\mathrm{NSLocMonoTrans}}^+(\omega).
```

Then

```{math}
K_{\mathrm{GenericUPTypeIIAdmiss}}^+(\omega).
```

::::

:::{prf:proof}
Lemma {prf:ref}`lem-c11-anchor-conclusion-discharge` gives
\(K_{\mathrm{C10Anchor}}^+\) and \(K_{\mathrm{UPConclusionId}}^+\). Lemma
{prf:ref}`lem-c11-scale-energy-discharge` gives
\(K_{\mathrm{ScaleSupercritRoute}}^+\) and
\(K_{\mathrm{BoundedEnergyTypeII}}^+\). Lemma
{prf:ref}`lem-c11-cost-translation-discharge` gives
\(K_{\mathrm{C4CostAsUPCost}}^+\). Lemma
{prf:ref}`lem-c11-domain-embedding-discharge` gives
\(K_{\mathrm{NSParabolicDomainEmb}}^+\). The remaining hypothesis is exactly
\(K_{\mathrm{NSLocMonoTrans}}^+\).

These seven witnesses are precisely
\(K_{\mathrm{GenUPPayload},NS3D}^+(\omega)\), and by Definition
{prf:ref}`def-c10-generic-up-admissibility-payload` they emit
\(K_{\mathrm{GenericUPTypeIIAdmiss}}^+(\omega)\). \(\square\)
:::

::::{prf:corollary} C11 route to formal `UP-TypeII`
:label: cor-c11-route-to-formal-up-typeii

Under the hypotheses of Theorem
{prf:ref}`thm-c11-generic-up-payload-compiler`, if also

```{math}
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega)
```

has been emitted by C4, then the formal theorem {prf:ref}`mt-up-type-ii` may be
applied to \(\omega\), and C10 emits

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}(\omega).
```

::::

:::{prf:proof}
Theorem {prf:ref}`thm-c11-generic-up-payload-compiler` gives
\(K_{\mathrm{GenericUPTypeIIAdmiss}}^+(\omega)\). C1 gives
\(K_{\mathrm{SC}_\lambda}^-(\omega)\) as part of
\(K_{\mathrm{TypeIIRoute}}^+(\omega)\), and C4 gives
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}(\omega)\). Lemma
{prf:ref}`lem-c10-generic-up-admissibility-licenses-up-typeii` applies the
formal theorem {prf:ref}`mt-up-type-ii` and emits
\(K_{\mathrm{SC}_\lambda}^{\sim}(\omega)\). \(\square\)
:::

## Ordered C11 defects

If the generic-UP payload is not emitted, C11 records the first missing witness:

```{math}
K_{\mathrm{C10Anchor}}^-,
\quad
K_{\mathrm{ScaleSupercritRoute}}^-,
\quad
K_{\mathrm{BoundedEnergyTypeII}}^-,
\quad
K_{\mathrm{NSLocMonoTrans}}^-,
\quad
K_{\mathrm{C4CostAsUPCost}}^-,
\quad
K_{\mathrm{NSParabolicDomainEmb}}^-,
\quad
K_{\mathrm{UPConclusionId}}^-.
```

On the current NS3D C-series route, all entries except
\(K_{\mathrm{NSLocMonoTrans}}^-\) are discharged by C1, C2, C4, C10, and the
dataset certificates. Thus the remaining obstruction to using the formal
generic `UP-TypeII` theorem is exactly the localized monotonicity translation.
C12 reduces that obstruction to the explicit finite-tail estimate
\(K_{\mathrm{FiniteMonoErr}}^+\).

This is why the promoted C8 theorem remains conditional on
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\): C11 explains how to obtain that
certificate through the generic theorem route once
\(K_{\mathrm{NSLocMonoTrans}}^+\) is proved.
