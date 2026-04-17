# Declared NS3D repaired-gauge Type II backend contract

This document fixes the local backend contract used by the Type II notes in
this folder.

The phrase **declared NS3D repaired-gauge Type II backend** means the
Navier-Stokes hypostructure backend obtained by combining:

1. the NS3D dataset certificate trail in
   [../dataset/navier_stokes_3d.md](../dataset/navier_stokes_3d.md);
2. the repaired-gauge renormalized orbit class in
   [compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md);
3. the repaired scale-gauge and modulation machinery in
   [required_new_scale_gauge_theorems.md](required_new_scale_gauge_theorems.md)
   and [cluster_next_easiest_theorems.md](cluster_next_easiest_theorems.md).

This backend contract is not a global regularity theorem. It defines the
certificate interface used to classify declared Type II candidates.

The C1--C4 bridge documents are consequences of this contract, not part of the
definition of the contract. This avoids circularity: the backend supplies the
primitive route, representation, critical-mass, and cost conventions; C1--C4
prove what follows from them.

## Backend universe

The backend universe is the declared Type II class

```{math}
\mathcal U_{\mathrm{II}}^{NS}.
```

A branch \(\omega=(u,p,T^*)\) belongs to
\(\mathcal U_{\mathrm{II}}^{NS}\) exactly when the NS3D dataset backend routes
it as a Type II candidate rather than as continuation-success, Type I,
boundary/open-system failure, non-Navier-Stokes domain failure, or another
non-Type-II diagnostic branch.

The backend does not assert that \(\mathcal U_{\mathrm{II}}^{NS}\) is nonempty.
It says what certificate route must be followed if a branch is declared Type II.

## Required positive route

Every \(\omega\in\mathcal U_{\mathrm{II}}^{NS}\) is routed through

```{math}
K_{\mathrm{TypeIIRoute}}^+(\omega)
:=
K_{C_\mu}^+(\omega)
\wedge
K_{\mathrm{SC}_\lambda}^-(\omega)
\wedge
K_{\mathrm{Prof}_{NS}}^+(\omega).
```

Equivalently, the backend carries

```{math}
K_{\mathrm{TypeIIExhaust}}^+.
```

The local implementation is C1:
[cluster_c1_typeII_branch_exhaustion.md](cluster_c1_typeII_branch_exhaustion.md).

## Repaired-gauge representation contract

For every routed candidate \(\omega\), the backend attaches the refined
representation-discharge payload

```{math}
K_{\mathrm{RepDischarge},NS3D}^+(\omega)
:=
K_{\mathrm{Chart},NS3D}^+(\omega)
\wedge
K_{\mathrm{GaugeSolve},NS3D}^+(\omega).
```

C2.R proves from this payload the four C2 representation components

```{math}
K_{\mathrm{RawOrb},NS3D}^+(\omega),
\quad
K_{\mathrm{GaugeReal},NS3D}^+(\omega),
\quad
K_{\mathrm{PressureRep},NS3D}^+(\omega),
\quad
K_{\mathrm{ModParams},NS3D}^+(\omega).
```

Their conjunction is the high-level representation payload
\(K_{\mathrm{RepPayload},NS3D}^+(\omega)\). This emits

```{math}
K_{\mathrm{RepBridge}}^+(\omega).
```

The local implementation is C2:
[cluster_c2_representation_bridge.md](cluster_c2_representation_bridge.md). The
finer discharge into the two genuine inputs, chart extraction and admissible
AC gauge solve, is C2.R:
[cluster_c2_repbridge_payload_discharge.md](cluster_c2_repbridge_payload_discharge.md).

## Critical \(L^3\)-mass contract

For every represented candidate, the backend evaluates the critical mass

```{math}
N_3(\tau)=\|V(\tau)\|_{L^3(\mathbb R^3)}.
```

The output is ordered:

```{math}
K_{L^3\mathrm{Dom}}^-,
\quad
K_{L^3\mathrm{Inf}}^-,
\quad
K_{L^3\mathrm{Zero}}^-,
\quad
K_{L^3\mathrm{Norm}}^+.
```

On the positive branch,

```{math}
K_{L^3\mathrm{Norm}}^+
\quad\text{means}\quad
0<\eta\le \|V(\tau)\|_{L^3}\le M<\infty
```

on the renormalized tail. Exact unit normalization is not required and is not
implemented by amplitude rescaling.

The local implementation is C3:
[cluster_c3_l3_normalization_bridge.md](cluster_c3_l3_normalization_bridge.md).

## Type II barrier cost contract

The backend uses the identity Type II barrier cost:

```{math}
\mathfrak C_{\mathrm{II}}^{NS}(\tau)
:=
\tilde{\mathfrak D}_{R_0}(\tau).
```

Thus divergence of the PDE localized renormalization cost is, by definition,
the blocked `BarrierTypeII` condition:

```{math}
\int_{\tau_0}^{\infty}
\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty
\Longrightarrow
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```

The local implementation is C4:
[cluster_c4_cost_bridge.md](cluster_c4_cost_bridge.md).

This cost contract emits the blocked Type II barrier certificate. It does not
by itself apply the generic `UP-TypeII` theorem to 3D Navier-Stokes. The
post-promotion suppression certificate

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}
```

requires the separate NS-valid applicability/replacement certificate

```{math}
K_{\mathrm{NS\text{-}UPTypeII}}^+,
```

defined in [cluster_c9_energy_cost_bridge.md](cluster_c9_energy_cost_bridge.md).
Thus the declared backend makes C1--C4 unconditional, while the promotion from
`BarrierTypeII` blocked to suppressed is conditional on C9's explicit
applicability bridge, whose full payload is made explicit in
[cluster_c10_ns_up_typeII_promotion.md](cluster_c10_ns_up_typeII_promotion.md).

## Complete local C1--C4 package

The primitive declared backend package is

```{math}
K_{\mathrm{NS3DTypeIIBackend}}^+
:=
K_{\mathrm{ExhPayload},NS3D}^+
\wedge
K_{\mathrm{RepPayload},NS3D}^+
\wedge
\mathsf{CriticalMassEval}_{NS3D}
\wedge
\mathsf{IdentityCost}_{NS3D}.
```

Here:

- \(K_{\mathrm{ExhPayload},NS3D}^+\) is the C1 route/exhaustion payload;
- \(K_{\mathrm{RepPayload},NS3D}^+\) is a universal per-candidate payload, i.e.
  it supplies the refined local representation-discharge payload
  \[
  K_{\mathrm{RepDischarge},NS3D}^+
  :=K_{\mathrm{Chart},NS3D}^+\wedge K_{\mathrm{GaugeSolve},NS3D}^+
  \]
  for every routed \(\omega\in\mathcal U_{\mathrm{II}}^{NS}\). C2.R proves
  pressure pullback and modulation coefficients from those two inputs and
  therefore
  \(K_{\mathrm{RepDischarge},NS3D}^+\Rightarrow
  K_{\mathrm{RepPayload},NS3D}^+\Rightarrow K_{\mathrm{RepBridge}}^+\);
- \(\mathsf{CriticalMassEval}_{NS3D}\) means the backend evaluates
  \(N_3(\tau)=\|V(\tau)\|_3\) and emits exactly one ordered C3 output for each
  represented candidate;
- \(\mathsf{IdentityCost}_{NS3D}\) means the backend uses
  \(\mathfrak C_{\mathrm{II}}^{NS}=\tilde{\mathfrak D}_{R_0}\) as its
  `BarrierTypeII` cost.

Under this primitive package, C1--C4 are explicit and unconditional inside the
declared backend:

```{math}
K_{\mathrm{NS3DTypeIIBackend}}^+
\Longrightarrow
K_{\mathrm{TypeIIExhaust}}^+
```

and for every routed candidate \(\omega\in\mathcal U_{\mathrm{II}}^{NS}\),

```{math}
K_{\mathrm{NS3DTypeIIBackend}}^+
\wedge
K_{\mathrm{TypeIIRoute}}^+(\omega)
\Longrightarrow
\bigl(
K_{\mathrm{RepBridge}}^+(\omega),
\text{exactly one ordered }L^3\text{-mass output for }\omega,
K_{\mathrm{CostBridge}}^+(\omega)
\bigr).
```

The first implication is C1. The per-candidate implication is the combination of
C2, C3, and C4.

## What this backend does not claim

This backend contract does not prove:

- that \(\mathcal U_{\mathrm{II}}^{NS}\) is nonempty;
- that every conceivable weak Navier-Stokes singularity outside the declared
  dataset backend is represented here;
- that Type I singularities are excluded;
- that radiative/noncompact Type II or rough-core Type II are excluded.
- that the generic `UP-TypeII` proof-object applies directly to NS3D without
  the explicit \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) bridge.

It only makes the Type II classification interface explicit. Once a candidate
is declared Type II by this backend, C1--C4 route it into the bridge ledger with
no hidden payload assumptions.
