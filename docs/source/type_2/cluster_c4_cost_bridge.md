# C4 cost bridge: compiling PDE renormalization cost into `BarrierTypeII`

This note implements the lowest-hanging bridge in the
classification-completeness program: the adapter from the localized PDE
renormalization cost in the compact Type II notes to the framework-level
`BarrierTypeII` blocked certificate. Promotion to the post-UP suppression
certificate is a separate NS3D applicability question handled by C9.

The goal is not to prove global regularity. The goal is to make the following
certificate implication precise:

```{math}
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty
\quad\Longrightarrow\quad
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```

Once this bridge is available, Theorem A'' in
[compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md)
can feed directly into the Hypostructure `BarrierTypeII` node. It feeds into
post-promotion Type II suppression only after the declared NS3D backend also
supplies \(K_{\mathrm{NS\text{-}UPTypeII}}^+\).

## Framework target

In the Hypostructure framework, Node 4 `ScaleCheck` sends a supercritical
scaling failure

```{math}
K_{\mathrm{SC}_\lambda}^-
```

to `BarrierTypeII`. The relevant blocked certificate is

```{math}
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```

The universal promotion pattern is

```{math}
K_{\mathrm{SC}_\lambda}^-
\wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}
\Longrightarrow
K_{\mathrm{SC}_\lambda}^{\sim}.
```

For 3D Navier-Stokes this last promotion is not automatic. C4 proves the
blocked certificate \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\). The promoted
certificate \(K_{\mathrm{SC}_\lambda}^{\sim}\) requires the additional
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\) bridge from C9.

See:

- [../1_hypostructure_formalism/04_nodes/02_barrier_nodes.md](../1_hypostructure_formalism/04_nodes/02_barrier_nodes.md), `BarrierTypeII`;
- [../1_hypostructure_formalism/08_upgrades/01_instantaneous.md](../1_hypostructure_formalism/08_upgrades/01_instantaneous.md), `UP-TypeII`;
- [../proofs/proof-mt-up-type-ii.md](../proofs/proof-mt-up-type-ii.md), proof-object for `UP-TypeII`.

## PDE cost used by the compact Type II theorem

The compact Type II master note uses the localized nonnegative cost

```{math}
\tilde{\mathfrak D}_{R_0}(\tau)
=
\nu\int_{\mathbb R^3}|\nabla V(y,\tau)|^2\phi_{R_0}(y)\,dy
+
a_+(\tau)\int_{\mathbb R^3}|V(y,\tau)|^2\phi_{R_0}(y)\,dy.
```

Here \(V\) is the repaired-gauge renormalized velocity profile,
\(a_+=\max(a,0)\), \(\phi_{R_0}\) is the cutoff used by the master note, and
\(\nu>0\) is the viscosity.

Theorem A'' proves, under global \(L^3\)-normalization, uniform \(L^3\)-tightness,
and local windowed \(H^1\) control, that

```{math}
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
```

This is the PDE-side infinite renormalization cost.

## Framework cost object

The generic `BarrierTypeII` statement is formulated in terms of an abstract
renormalization cost. For the Navier-Stokes Type II backend we make that cost
object explicit.

::::{prf:definition} Navier-Stokes Type II barrier cost
:label: def-ns-typeII-barrier-cost

Let \((V,P,a,b)\) be a repaired-gauge renormalized Navier-Stokes Type II
candidate. A **Navier-Stokes Type II barrier cost** is a measurable function

```{math}
\mathfrak C_{\mathrm{II}}:[\tau_0,\infty)\to[0,\infty]
```

together with a certificate payload

```{math}
\mathsf{CostPayload}
=
\bigl(
\mathfrak C_{\mathrm{II}},
\tau_0,
\mathsf{nonnegativity},
\mathsf{measurability},
\mathsf{local\_finite\_interval\_integrability},
\mathsf{divergence\_meaning}
\bigr),
```

where `divergence_meaning` records that

```{math}
\int_{\tau_0}^{\infty}\mathfrak C_{\mathrm{II}}(\tau)\,d\tau=\infty
```

is accepted by the `BarrierTypeII` evaluator as the blocked Type II barrier
condition.

::::

This definition is backend-specific. It does not change the abstract framework.
It declares which concrete Navier-Stokes cost is submitted to the Type II
barrier evaluator.

## Navier-Stokes `BarrierTypeII` backend convention

The remaining ambiguity is removed by fixing the Navier-Stokes specialization
of `BarrierTypeII`.

::::{prf:definition} Navier-Stokes `BarrierTypeII` evaluator
:label: def-ns-barriertypeii-evaluator

For the Navier-Stokes Type II backend, the `BarrierTypeII` evaluator is defined
with barrier cost

```{math}
\mathfrak C_{\mathrm{II}}^{NS}(\tau)
:=
\tilde{\mathfrak D}_{R_0}(\tau).
```

Its blocked certificate is emitted by the rule

```{math}
\int_{\tau_0}^{\infty}
\mathfrak C_{\mathrm{II}}^{NS}(\tau)\,d\tau
=\infty
\quad\Longrightarrow\quad
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```

Equivalently, the backend registration certificate

```{math}
K_{\mathrm{BTII}\leftarrow \mathfrak C_{\mathrm{II}}^{NS}}^+
```

is part of the Navier-Stokes Type II backend definition, not an additional
analytic hypothesis.

::::

This convention is the precise implementation of the phrase "the identity-cost
backend." It makes the framework cost and the PDE cost identical for the
Navier-Stokes Type II branch. Any alternative backend may still choose a
different cost, but that is no longer the default route used in this
classification-completeness program.

## Cost comparison witness

The bridge from PDE cost to framework cost is a domination statement.

::::{prf:definition} Cost comparison witness
:label: def-cost-comparison-witness

For a repaired-gauge Type II candidate, a **cost comparison witness** is a tuple

```{math}
W_{\mathrm{cost}}
=
\bigl(
R_0,\phi_{R_0},
\mathfrak C_{\mathrm{II}},
c_0,
\tau_1,
\mathsf{comparison\_proof}
\bigr)
```

with \(c_0>0\), \(\tau_1\ge\tau_0\), and

```{math}
\mathfrak C_{\mathrm{II}}(\tau)
\ge
c_0\,\tilde{\mathfrak D}_{R_0}(\tau)
\qquad
\text{for a.e. }\tau\ge\tau_1.
```

The associated positive certificate is denoted

```{math}
K_{\mathrm{CostCompare}}^+.
```

::::

The simplest backend choice is

```{math}
\mathfrak C_{\mathrm{II}}(\tau)
:=
\mathfrak C_{\mathrm{II}}^{NS}(\tau)
:=
\tilde{\mathfrak D}_{R_0}(\tau),
\qquad
c_0=1,
\qquad
\tau_1=\tau_0.
```

Then \(K_{\mathrm{CostCompare}}^+\) is the identity comparison. If a different
framework cost is used, \(K_{\mathrm{CostCompare}}^+\) records the analytic
comparison needed to dominate the PDE cost.

## The bridge certificate

::::{prf:definition} PDE-to-framework Type II cost bridge
:label: def-costbridge-certificate

The certificate

```{math}
K_{\mathrm{CostBridge}}^+
```

consists of:

1. a Navier-Stokes Type II barrier cost
   \(\mathfrak C_{\mathrm{II}}\) in the sense of
   Definition {prf:ref}`def-ns-typeII-barrier-cost`;
2. a cost comparison witness \(K_{\mathrm{CostCompare}}^+\) in the sense of
   Definition {prf:ref}`def-cost-comparison-witness`;
3. a backend admissibility certificate
   ```{math}
   K_{\mathrm{BTII}\leftarrow \mathfrak C_{\mathrm{II}}}^+
   ```
   stating that the `BarrierTypeII` evaluator accepts divergence of
   \(\mathfrak C_{\mathrm{II}}\) as its blocked condition.

Equivalently, the bridge certificate packages the data needed to prove the
verifier

```{math}
\mathsf{CostBridgeVerify}
:
\left[
\int_{\tau_0}^{\infty}
\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty
\right]
\to
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}},
```

but it does not include that verifier as an assumption.

::::

For the default Navier-Stokes Type II backend, Definition
{prf:ref}`def-ns-barriertypeii-evaluator` supplies item 3 with
\(\mathfrak C_{\mathrm{II}}=\mathfrak C_{\mathrm{II}}^{NS}\), and the identity
comparison supplies item 2. Thus \(K_{\mathrm{CostBridge}}^+\) is automatic for
the default backend once the PDE cost is measurable, nonnegative, and locally
integrable on finite \(\tau\)-intervals.

::::{prf:lemma} Automatic identity cost bridge
:label: lem-automatic-identity-cost-bridge

Let \((V,P,a,b)\) be a repaired-gauge renormalized Navier-Stokes Type II
candidate for which \(\tilde{\mathfrak D}_{R_0}\) is measurable, nonnegative,
and locally integrable on finite \(\tau\)-intervals. In the Navier-Stokes
`BarrierTypeII` backend of Definition
{prf:ref}`def-ns-barriertypeii-evaluator`,

```{math}
K_{\mathrm{CostBridge}}^+
```

holds.

::::

:::{prf:proof}
Set

```{math}
\mathfrak C_{\mathrm{II}}
=
\mathfrak C_{\mathrm{II}}^{NS}
=
\tilde{\mathfrak D}_{R_0}.
```

The assumed measurability, nonnegativity, and local finite-interval
integrability give the cost payload of Definition
{prf:ref}`def-ns-typeII-barrier-cost`. The identity
\(\mathfrak C_{\mathrm{II}}=\tilde{\mathfrak D}_{R_0}\) gives
\(K_{\mathrm{CostCompare}}^+\) with \(c_0=1\) and \(\tau_1=\tau_0\). Definition
{prf:ref}`def-ns-barriertypeii-evaluator` supplies
\(K_{\mathrm{BTII}\leftarrow \mathfrak C_{\mathrm{II}}}^+\). These are exactly
the components of \(K_{\mathrm{CostBridge}}^+\). \(\square\)
:::

## Main C4 theorem

::::{prf:theorem} C4 cost bridge
:label: thm-c4-cost-bridge

Let \((V,P,a,b)\) be a repaired-gauge renormalized Navier-Stokes Type II
candidate. Assume:

1. the localized PDE cost \(\tilde{\mathfrak D}_{R_0}\) is measurable,
   nonnegative, and locally integrable on finite \(\tau\)-intervals;
2. the PDE infinite-cost statement holds:
   ```{math}
   \int_{\tau_0}^{\infty}
   \tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty;
   ```
3. either the default Navier-Stokes `BarrierTypeII` backend of Definition
   {prf:ref}`def-ns-barriertypeii-evaluator` is used, or
   \(K_{\mathrm{CostBridge}}^+\) is available for an alternative backend cost.

Then the framework-level blocked Type II barrier certificate is emitted:

```{math}
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```

::::

:::{prf:proof}
If the default Navier-Stokes backend is used, Lemma
{prf:ref}`lem-automatic-identity-cost-bridge` supplies
\(K_{\mathrm{CostBridge}}^+\). Otherwise this certificate is assumed for the
alternative backend cost. In either case, there is a framework Type II barrier cost
\(\mathfrak C_{\mathrm{II}}\) and a comparison witness
\(K_{\mathrm{CostCompare}}^+\). Hence there exist \(c_0>0\) and
\(\tau_1\ge\tau_0\) such that, for almost every \(\tau\ge\tau_1\),

```{math}
\mathfrak C_{\mathrm{II}}(\tau)
\ge
c_0\tilde{\mathfrak D}_{R_0}(\tau).
```

Since \(\tilde{\mathfrak D}_{R_0}\ge0\) and is locally integrable on finite
intervals, divergence on \([\tau_0,\infty)\) implies tail divergence:

```{math}
\int_{\tau_1}^{\infty}
\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
```

Therefore

```{math}
\int_{\tau_1}^{\infty}
\mathfrak C_{\mathrm{II}}(\tau)\,d\tau
\ge
c_0
\int_{\tau_1}^{\infty}
\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau
=\infty.
```

Since \(\mathfrak C_{\mathrm{II}}\ge0\) and is locally integrable on finite
intervals by its cost payload,

```{math}
\int_{\tau_0}^{\infty}
\mathfrak C_{\mathrm{II}}(\tau)\,d\tau
=
\int_{\tau_0}^{\tau_1}
\mathfrak C_{\mathrm{II}}(\tau)\,d\tau
+
\int_{\tau_1}^{\infty}
\mathfrak C_{\mathrm{II}}(\tau)\,d\tau
=\infty.
```

By the `divergence_meaning` payload in the Navier-Stokes Type II barrier cost
and the admissibility certificate
\(K_{\mathrm{BTII}\leftarrow \mathfrak C_{\mathrm{II}}}^+\), divergence of
\(\mathfrak C_{\mathrm{II}}\) is exactly the blocked condition accepted by the
`BarrierTypeII` evaluator. Hence the evaluator emits

```{math}
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```

This is the desired bridge. \(\square\)
:::

## Identity-cost corollary

The cost bridge is unconditional in the default Navier-Stokes backend because
that backend declares the framework Type II cost to be the PDE localized
renormalization cost.

::::{prf:corollary} Identity-cost bridge
:label: cor-identity-cost-bridge

In the Navier-Stokes `BarrierTypeII` backend of Definition
{prf:ref}`def-ns-barriertypeii-evaluator`, set

```{math}
\mathfrak C_{\mathrm{II}}(\tau)
:=
\tilde{\mathfrak D}_{R_0}(\tau).
```

Then \(K_{\mathrm{CostCompare}}^+\) holds with \(c_0=1\) and
\(\tau_1=\tau_0\), and the backend registration
\(K_{\mathrm{BTII}\leftarrow \mathfrak C_{\mathrm{II}}}^+\) is automatic.
Consequently, every PDE infinite-cost conclusion of Theorem A'' whose cost is
locally integrable on finite \(\tau\)-intervals compiles into

```{math}
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```

::::

:::{prf:proof}
The comparison inequality is the identity

```{math}
\mathfrak C_{\mathrm{II}}(\tau)
=
\tilde{\mathfrak D}_{R_0}(\tau).
```

Therefore Definition {prf:ref}`def-cost-comparison-witness` holds with
\(c_0=1\) and \(\tau_1=\tau_0\). Definition
{prf:ref}`def-ns-barriertypeii-evaluator` gives the backend registration
\(K_{\mathrm{BTII}\leftarrow \mathfrak C_{\mathrm{II}}}^+\), hence Lemma
{prf:ref}`lem-automatic-identity-cost-bridge` gives
\(K_{\mathrm{CostBridge}}^+\). Theorem
{prf:ref}`thm-c4-cost-bridge` then gives
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\). \(\square\)
:::

## Coupling with Theorem A''

Combining Theorem A'' with the cost bridge gives the exact certificate used by
the sieve.

::::{prf:theorem} Theorem A'' emits `BarrierTypeII` after C4
:label: thm-a-double-prime-emits-barriertypeii

Assume a repaired-gauge renormalized Type II candidate satisfies the hypotheses
of Theorem A'':

1. global \(L^3\)-normalization,
2. uniform global \(L^3\)-tightness,
3. uniform local windowed \(L^2_\tau H^1_y\) bounds.

Use the default Navier-Stokes `BarrierTypeII` backend of Definition
{prf:ref}`def-ns-barriertypeii-evaluator`. Then the candidate emits

```{math}
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```

If the Navier-Stokes sieve run also supplies \(K_{\mathrm{SC}_\lambda}^-\) and
the NS applicability bridge \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) from
[cluster_c9_energy_cost_bridge.md](cluster_c9_energy_cost_bridge.md), then the
NS-valid Type II promotion emits

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}.
```

::::

:::{prf:proof}
Theorem A'' gives

```{math}
\int_{\tau_0}^{\infty}
\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
```

By Lemma {prf:ref}`lem-automatic-identity-cost-bridge`, the default
Navier-Stokes backend supplies \(K_{\mathrm{CostBridge}}^+\). The C4 cost
bridge converts the infinite PDE cost into
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\). Combining this blocked certificate
with the supercritical scaling certificate \(K_{\mathrm{SC}_\lambda}^-\) gives
the post-promotion certificate \(K_{\mathrm{SC}_\lambda}^{\sim}\) only after
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\) has verified that the Type II promotion
is valid for the NS3D backend. Without that bridge, C4 emits the blocked
certificate \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\), not
\(K_{\mathrm{SC}_\lambda}^{\sim}\). \(\square\)
:::

## What this discharges

This note implements C4 in the roadmap at the certificate-adapter level.

The remaining analytic issue is not the formal implication

```{math}
\int\tilde{\mathfrak D}_{R_0}=\infty
\Rightarrow
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```

That implication is now packaged by \(K_{\mathrm{CostBridge}}^+\). In the
default Navier-Stokes backend, the convention is fixed:

1. **Default identity backend.** Declare
   \(\mathfrak C_{\mathrm{II}}^{NS}=\tilde{\mathfrak D}_{R_0}\). Registration
   with `BarrierTypeII` is part of Definition
   {prf:ref}`def-ns-barriertypeii-evaluator`. Then C4 is fully discharged by
   Lemma {prf:ref}`lem-automatic-identity-cost-bridge` and Corollary
   {prf:ref}`cor-identity-cost-bridge`.
2. **External framework-cost backend.** Keep the generic model cost used by
   `UP-TypeII`. Then one must prove \(K_{\mathrm{CostCompare}}^+\) by comparing
   that model cost to \(\tilde{\mathfrak D}_{R_0}\).

For the Type II classification program, the identity backend is the cleanest:
it makes the compact PDE barrier directly consumable by the hypostructure
`BarrierTypeII` node without importing a separate model cost.
