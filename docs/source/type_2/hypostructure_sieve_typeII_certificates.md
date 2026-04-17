# Sieve certificates for strengthening the Type II barrier

This note links the compact Type II proof stack in this folder to the
Hypostructure sieve framework. The goal here is narrower than the global
regularity theorem in [../dataset/navier_stokes_3d.md](../dataset/navier_stokes_3d.md):
we use the certificates produced by a Navier-Stokes sieve run as additional
typed hypotheses for the Type II exclusion program.

The intended output is a stronger conditional Type II theorem:

```{math}
\text{sieve certificates}
\quad+\quad
\text{repaired-gauge PDE bridges}
\quad+\quad
\text{compact Type II barrier}
\quad\Longrightarrow\quad
\text{fewer admissible Type II scenarios}.
```

This does not assert
```{math}
K_{\mathrm{Reg}_{NS}}^+.
```
It only says that the available structural certificates can be imported into
the compact Type II analysis before invoking the categorical Lock or the final
continuation bridge.

## Framework references

The relevant framework components are:

- [../1_hypostructure_formalism/intro_hypostructure.md](../1_hypostructure_formalism/intro_hypostructure.md), for the typed certificate model and the gate/barrier/surgery sieve.
- [../1_hypostructure_formalism/04_nodes/01_gate_nodes.md](../1_hypostructure_formalism/04_nodes/01_gate_nodes.md), especially Node 3 `CompactCheck`, Node 4 `ScaleCheck`, and the routing from a scaling failure to `BarrierTypeII`.
- [../1_hypostructure_formalism/04_nodes/02_barrier_nodes.md](../1_hypostructure_formalism/04_nodes/02_barrier_nodes.md), especially `BarrierTypeII`, whose blocked certificate is
  ```{math}
  K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
  ```
- [../1_hypostructure_formalism/08_upgrades/01_instantaneous.md](../1_hypostructure_formalism/08_upgrades/01_instantaneous.md), especially `UP-TypeII`, which promotes
  ```{math}
  K_{\mathrm{SC}_\lambda}^- \wedge K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}
  \Longrightarrow
  K_{\mathrm{SC}_\lambda}^{\sim}.
  ```
  For 3D Navier-Stokes this promotion is not automatic. It is available only
  after the declared backend supplies the NS applicability/replacement
  certificate \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) from
  [cluster_c9_energy_cost_bridge.md](cluster_c9_energy_cost_bridge.md).
- [../proofs/proof-mt-up-type-ii.md](../proofs/proof-mt-up-type-ii.md), for the proof-object interpretation of the Type II renormalization barrier certificate.
- [../dataset/navier_stokes_3d.md](../dataset/navier_stokes_3d.md), for the concrete 3D Navier-Stokes certificate trail.

The compact Type II PDE inputs are:

- [ns3d_repaired_gauge_backend_contract.md](ns3d_repaired_gauge_backend_contract.md), for the local backend contract under which C1--C4 are explicit and unconditional inside the declared NS3D repaired-gauge Type II backend.
- [compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md), especially Theorem A'', the good-window compact Type II barrier, and Corollary 16, the finite-cost contrapositive classification.
- [cluster_next_easiest_theorems.md](cluster_next_easiest_theorems.md), for pressure, repaired-gauge nondegeneracy, bounded modulation, and time-regularity theorem clusters.
- [cluster_t7_L3_compactness_upgrade.md](cluster_t7_L3_compactness_upgrade.md), for the strong local \(L^3\)-compactness upgrade and good-time selection.
- [required_new_scale_gauge_theorems.md](required_new_scale_gauge_theorems.md), for the repaired scale gauge and modulation matrix layer.
- [cluster_c1_typeII_branch_exhaustion.md](cluster_c1_typeII_branch_exhaustion.md), for the declared Type II branch exhaustion certificate.
- [cluster_c2_representation_bridge.md](cluster_c2_representation_bridge.md), for the bridge from sieve/profile data to repaired-gauge renormalized Navier-Stokes orbits.
- [cluster_c3_l3_normalization_bridge.md](cluster_c3_l3_normalization_bridge.md), for the critical \(L^3\)-normalization bridge and the classification of normalization defects.
- [cluster_c4_cost_bridge.md](cluster_c4_cost_bridge.md), for the implemented certificate adapter \(K_{\mathrm{CostBridge}}^+\) from PDE infinite renormalization cost to the framework-level `BarrierTypeII` blocked certificate.
- [cluster_c5_two_bucket_classification.md](cluster_c5_two_bucket_classification.md), for the implemented C5 finite-cost two-bucket classification after C1--C4 discharge representation, normalization, exhaustion, and cost-adapter defects.
- [cluster_c6_radiative_noncompact_exclusion.md](cluster_c6_radiative_noncompact_exclusion.md), for the no-radiation/tightness certificate \(K_{\mathrm{RadBlk}}^+\equiv K_{L^3\mathrm{Tight}}^+\).
- [cluster_c7_win_h1_certificate_composition.md](cluster_c7_win_h1_certificate_composition.md), for the rough-core/windowed-\(H^1\) certificate composition whose positive rough-core blocker carries the \(K_{\mathrm{WinH1}}^+\) payload.
- [cluster_c7_l3bd_defect_discharge.md](cluster_c7_l3bd_defect_discharge.md), for the discharge of the C7 bounded-critical-norm defect from \(K_{L^3\mathrm{Norm}}^+\).
- [cluster_c7_modmatrix_inverse_discharge.md](cluster_c7_modmatrix_inverse_discharge.md), for the discharge of the C7 modulation-matrix inverse defect from repaired-gauge nondegeneracy.
- [cluster_c7_scale_force_bound.md](cluster_c7_scale_force_bound.md), for the pointwise weighted scale-force decomposition and nonlinear row discharge from \(K_{\mathrm{ScaleL4Mom}}^+\).
- [cluster_c8_typeII_branch_exclusion.md](cluster_c8_typeII_branch_exclusion.md), for the final Type II-specific C-series assembly theorem excluding admissible unresolved Type II branches after classification completeness, radiative suppression, and rough-core suppression.
- [cluster_c10_ns_up_typeII_promotion.md](cluster_c10_ns_up_typeII_promotion.md), for the explicit NS3D promotion payload \(K_{\mathrm{NSPromPayload}}^+\) behind \(K_{\mathrm{NS\text{-}UPTypeII}}^+\).
- [cluster_c11_generic_up_typeII_payload_discharge.md](cluster_c11_generic_up_typeII_payload_discharge.md), for the generic `UP-TypeII` admissibility payload and the remaining localized monotonicity witness \(K_{\mathrm{NSLocMonoTrans}}^+\).
- [cluster_c12_ns_localized_monotonicity_translation.md](cluster_c12_ns_localized_monotonicity_translation.md), for the corrected localized monotonicity theorem reducing \(K_{\mathrm{NSLocMonoTrans}}^+\) to finite tail control of the explicit monotonicity-error density.
- [cluster_c13_formal_up_typeII_application.md](cluster_c13_formal_up_typeII_application.md), for the exact NS3D application package that licenses invoking the formal theorem {prf:ref}`mt-up-type-ii`.
- [cluster_c14_explicit_up_typeII_ns3d_checklist.md](cluster_c14_explicit_up_typeII_ns3d_checklist.md), for the direct and expanded complete checklists for applying formal `UP-TypeII` to NS3D.
- [cluster_c15_moving_cutoff_monotonicity.md](cluster_c15_moving_cutoff_monotonicity.md), for the moving-cutoff replacement of the fixed finite-error certificate.
- [cluster_c16_scale_negative_drift_dichotomy.md](cluster_c16_scale_negative_drift_dichotomy.md), for the finite-or-infinite routing of the moving-cutoff scale-negative drift term.
- [cluster_c17_scale_collapse_barrier.md](cluster_c17_scale_collapse_barrier.md), for the scale-collapse drift barrier fallback when the C16 obstruction is infinite.
- [cluster_c18_final_up_typeII_ns3d.md](cluster_c18_final_up_typeII_ns3d.md), for the final declared-terminal-backend `UP-TypeII` assembly theorem for NS3D.
- [cluster_s12_terminal_profile_completeness.md](cluster_s12_terminal_profile_completeness.md), for the Type-II-local terminal profile-completeness discharge from critical NS profile decomposition, replacing the global `CatLib` placeholder in the C18 terminal package.
- [cluster_s13_bounded_critical_terminal_sequences.md](cluster_s13_bounded_critical_terminal_sequences.md), for the bounded terminal critical-sequence input in S12, discharged from \(K_{L^3\mathrm{Norm}}^+\) and terminal sequence routing.
- [cluster_s14_terminal_sequence_routing.md](cluster_s14_terminal_sequence_routing.md), for discharging terminal sequence routing from repaired-gauge representation and terminal-camera construction.
- [cluster_s3_scale_collapse_attractor_stratification.md](cluster_s3_scale_collapse_attractor_stratification.md), for the scale-collapse generalized self-similar reduction alternative to the C17 cost bridge.

## Certificates already available from the Navier-Stokes dataset route

The Navier-Stokes dataset records the following core certificates before the
final Lock/continuation conclusion is used.

| Certificate | Source in sieve | Meaning for Type II analysis |
|---|---|---|
| \(K_{\mathrm{Auto}}^+\) | automation witness | The parabolic instance is admissible for the universal singularity modules and factory backend architecture. |
| \(K_{D_E}^+\) | Node 1, energy interface | Physical kinetic energy is bounded and dissipative: \(E(t)+\int_0^t\mathfrak D(s)\,ds\le E(0)\). |
| \(K_{C_\mu}^+\) | Node 3, compactness interface | Concentration profile / blow-up germ exists modulo the declared Navier-Stokes symmetry group. |
| \(K_{\mathrm{SC}_\lambda}^-\) | Node 4, scaling interface | Energy scaling is supercritical; this is the branch that routes to `BarrierTypeII`. |
| \(K_{\mathrm{SC}_{\partial c}}^+\) | Node 5, parameter interface | The structural parameters \((n,\nu)=(3,\nu)\) are fixed along the run. |
| \(K_{\mathrm{Cap}_H}^+\) | Node 6, capacity interface | Capacity/codimension data is available for the declared singularity package. |
| \(K_{\mathrm{TB}_\pi}^+\) | Node 8, topology interface | The divergence-free sector is preserved. |
| \(K_{\mathrm{TB}_O}^+\) | Node 9, tameness interface | The declared profile backend is tame/stratified. |
| \(K_{\mathrm{RepDesc}_K}^+\) | Node 11, finite-description interface | The thin trace has a finite-description representation usable by the backend. |
| \(K_{\mathrm{Bound}_\partial}^-\) | Node 13, boundary interface | The problem is routed as a closed-system instance toward the Lock. |

The dataset also records structured inconclusive certificates:

| Certificate | Meaning | Type II use |
|---|---|---|
| \(K_{\mathrm{Rec}_N}^{\mathrm{inc}}\) | event finiteness is not supplied by this route | Do not use it as a Type II exclusion input unless upgraded. |
| \(K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}\) | stiffness package missing | Cannot be used to rule out rough-core oscillation without a separate stiffness bridge. |
| \(K_{\mathrm{TB}_\rho}^{\mathrm{inc}}\) | mixing backend missing | Cannot be used to exclude recurrent radiation without a separate ergodic/mixing bridge. |
| \(K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}\) | gradient/Lyapunov compatibility missing | Cannot be used to suppress local frequency cascades without a separate oscillation bridge. |

These `inc` certificates are still useful: their missing payloads identify exactly
which extra structural hypotheses would narrow the remaining Type II branches.

## Certificates from the singularity module

After Node 3, the Navier-Stokes dataset invokes the universal singularity
machinery and records:

| Certificate | Meaning for Type II analysis |
|---|---|
| \(K_{\mathrm{Prof}_{NS}}^+\) | The extracted profile lies in the declared profile backend, either \(K_{\mathrm{lib}}^+\) or \(K_{\mathrm{strat}}^+\). |
| \(K_{\mathrm{Germ}}^+\) | The classifiable germ package is set-sized and usable by the library machinery. |
| \(K_{\mathrm{init}}^+\) | The universal bad object \(\mathbb H_{\mathrm{bad}}^{NS}\) exists for the declared package. |
| \(K_{\mathrm{CatLib}}^+\) | The finite bad-pattern library \(\mathcal B_{NS}\) is complete relative to the declared backend. |

For the Type II program, the useful part is not the final categorical Lock.
The useful part is that a Type II candidate can be treated as a profile/germ
object modulo translations, rotations, time shifts, and scaling. This is the
correct framework-level source for the renormalized-orbit representation
hypothesis used in [compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md).

However, the translation is not automatic. It requires the bridge certificate
defined below.

## UP-metatheorem outputs relevant to Type II

The most important universal-property output is the Type II promotion pattern:

```{math}
K_{\mathrm{SC}_\lambda}^-
\wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}
\Longrightarrow
K_{\mathrm{SC}_\lambda}^{\sim}.
```

Interpretation:

- \(K_{\mathrm{SC}_\lambda}^-\) is already present in the Navier-Stokes dataset and records supercritical energy scaling.
- \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) is the Type II barrier-block certificate. In the generic framework it is phrased as divergence of a renormalization cost.
- \(K_{\mathrm{SC}_\lambda}^{\sim}\) is the post-UP certificate saying that the Type II obstruction behaves as effectively suppressed for the purpose of the scaling interface.

The generic `UP-TypeII` proof-object is heat-model-oriented. Therefore the
Navier-Stokes Type II stack separates two outputs:

```{math}
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}
\quad\text{from the NS3D } \mathrm{BarrierTypeII}\text{ backend},
```

and

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}
\quad\text{only after}\quad
K_{\mathrm{NS\text{-}UPTypeII}}^+.
```

Here \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) means the uniform C10 payload
\(K_{\mathrm{NSPromPayload}}^+\): either the generic `UP-TypeII` hypotheses
have been verified for the NS3D repaired-gauge backend, or an NS-specific
replacement promotion theorem has been registered with the same input/output
type.

C11 makes the first alternative explicit: the formal theorem
{prf:ref}`mt-up-type-ii` may be invoked once
\(K_{\mathrm{GenericUPTypeIIAdmiss}}^+\) is emitted. The current C-series stack
reduces this to the localized monotonicity translation
\(K_{\mathrm{NSLocMonoTrans}}^+\) plus already registered route, representation,
energy, cost, and conclusion-identity certificates.
C12 proves \(K_{\mathrm{NSLocMonoTrans}}^+\) from
\(K_{\mathrm{FiniteMonoErr}}^+\), the finite tail integral of the explicit
localized energy error density.
C13 then composes C10--C12: once the application package
\(K_{\mathrm{FormalUPTypeIIApp},NS3D}^+\) is present, the formal theorem
{prf:ref}`mt-up-type-ii` applies and emits
\(K_{\mathrm{SC}_\lambda}^{\sim}\).

C15 provides the moving-cutoff replacement for the fixed finite-error package.
It reduces the annular error terms by allowing the cutoff annulus to move
outward, but it leaves the weighted negative scale drift
\(K_{\mathrm{ScaleNegL^1}}^+\). C16 makes this last term explicit: finite
negative repaired-gauge drift with bounded moving localized \(L^2\) mass emits
the positive certificate, while failure of the weighted integral is the named
obstruction \(K_{\mathrm{ScaleCollapseDrift}}^-\). Under the master-note
convention \(a=d_\tau\log\lambda\), genuine scale collapse
\(\lambda(\tau)\to0\) with nonvanishing localized \(L^2\) core forces this
obstruction.

C17 then gives the barrier fallback: if the declared `BarrierTypeII` evaluator
accepts either the scale-collapse cost \(a_-M_R\) or the absolute scale-drift
cost \(\nu\int|\nabla V|^2\phi_{R(\tau)}+|a|M_R\), then
\(K_{\mathrm{ScaleCollapseDrift}}^-\) emits
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) directly. With the C10 NS-valid
promotion payload, this blocked certificate emits
\(K_{\mathrm{SC}_\lambda}^{\sim}\).

C18 is the final assembly theorem. It packages the abstract C8 route
\[
K_{\mathrm{ClassComplete}}^+
\wedge K_{\mathrm{RadBlk}}^+
\wedge K_{\mathrm{RoughCoreBlk}}^+
\wedge K_{\mathrm{NS\text{-}UPTypeII}}^+
\Longrightarrow
\forall\omega\in\mathcal U_{\mathrm{II}}^{NS},
K_{\mathrm{SC}_\lambda}^{\sim}(\omega),
\]
and expands the terminal backend payload using S14, S13, and S12 for terminal
profile completeness, the C6 no-radiation/tightness route, S3 and S4--S8 for
scale-collapse and multibubble/radiative closure, C7 for rough-core closure,
and C10 for NS-valid promotion.

The compact Type II theorem in this folder supplies a PDE-specific way to
produce a barrier conclusion in the finite-cost compact branch:

```{math}
\text{global } L^3 \text{ normalization}
\wedge
\text{uniform } L^3 \text{ tightness}
\wedge
\text{local windowed } L^2_\tau H^1_y \text{ control}
\Longrightarrow
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
```

This is not literally the same cost as the model cost in `UP-TypeII`.
Therefore, the sieve-strengthened Type II theorem must record a cost translator.

## Bridge certificates needed by the Type II proof

The sieve certificates strengthen the Type II proof only after the following
bridges are supplied.

### 1. Representation bridge

```{math}
K_{\mathrm{RepBridge}}^+:
\quad
K_{C_\mu}^+
\wedge
K_{\mathrm{Prof}_{NS}}^+
\Longrightarrow
\text{renormalized repaired-gauge orbit }(V,P,a,b).
```

Payload:

- a concentration center \(x_c(t)\),
- a scale \(\lambda(t)\),
- renormalized time \(\tau_t=\lambda(t)^{-2}\),
- a profile \(V(y,\tau)\),
- pressure \(P(y,\tau)\),
- modulation parameters \(a(\tau),b(\tau)\),
- compatibility with the repaired scale and centering gauges.

This bridge turns the framework profile/germ certificate into the PDE object
used by the compact Type II notes.

The bridge is implemented in
[cluster_c2_representation_bridge.md](cluster_c2_representation_bridge.md). It
requires four concrete PDE payloads:

```{math}
K_{\mathrm{RawOrb}}^+,\qquad
K_{\mathrm{GaugeReal}}^+,\qquad
K_{\mathrm{PressureRep}}^+,\qquad
K_{\mathrm{ModParams}}^+.
```

For the declared NS3D repaired-gauge Type II backend, that note explicitly
provides the concrete representation payload

```{math}
K_{\mathrm{RepPayload},NS3D}^+
```

for candidates routed by C1, and derives
\(K_{\mathrm{RepBridge}}^+\) from it. The refined discharge document
[cluster_c2_repbridge_payload_discharge.md](cluster_c2_repbridge_payload_discharge.md)
shows that this payload follows from the more local certificate

```{math}
K_{\mathrm{RepDischarge},NS3D}^+
=
K_{\mathrm{Chart},NS3D}^+\wedge
K_{\mathrm{GaugeSolve},NS3D}^+.
```

In that refined ledger, pressure pullback and final modulation coefficients are
proved consequences of the chart plus AC gauge solve, not independent bridge
assumptions.

In the refined C2.R ledger, if representation cannot be emitted, the candidate
emits one of the two genuine ordered representation-defect certificates:

```{math}
K_{\mathrm{Chart}}^-,
\quad
K_{\mathrm{GaugeSolve}}^-.
```

The older high-level names \(K_{\mathrm{PressureRep}}^-\) and
\(K_{\mathrm{ModParams}}^-\) are not independent survivor defects once the
chart and AC gauge solve exist; pressure and modulation are then proved inside
C2.R.

### 2. Critical normalization bridge

```{math}
K_{L^3\mathrm{Norm}}^+:
\quad
\text{critical branch data}
\Longrightarrow
\|V(\tau)\|_{L^3(\mathbb R^3)}=1
\text{ after admissible normalization}.
```

This bridge is necessary because the sieve's scaling certificate records the
structural scaling exponents, while the Type II barrier uses the scale-critical
Navier-Stokes norm \(L^3\).

The bridge is implemented in
[cluster_c3_l3_normalization_bridge.md](cluster_c3_l3_normalization_bridge.md).
There, exact unit normalization is replaced by the invariant annulus condition

```{math}
0<\eta\le \|V(\tau)\|_{L^3(\mathbb R^3)}\le M<\infty.
```

This avoids the invalid operation of multiplying the Navier-Stokes velocity by
a time-dependent scalar. If the annulus condition fails, the candidate emits one
of the explicit defect certificates

```{math}
K_{L^3\mathrm{Zero}}^-,
\qquad
K_{L^3\mathrm{Inf}}^-,
\qquad
K_{L^3\mathrm{Dom}}^-.
```

### 3. Cost bridge

```{math}
K_{\mathrm{CostBridge}}^+:
\quad
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty
\Longrightarrow
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```

This bridge is the key adapter from the PDE theorem to the hypostructure
`BarrierTypeII` node. It must identify the localized renormalized cost
\(\tilde{\mathfrak D}_{R_0}\) with, or prove it dominates, the cost accepted by
the framework's Type II barrier.

Without \(K_{\mathrm{CostBridge}}^+\), Theorem A'' is still a valid PDE
barrier, but it has not yet been compiled into the framework's
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) certificate.

The certificate-adapter implementation is now isolated in
[cluster_c4_cost_bridge.md](cluster_c4_cost_bridge.md). That note defines the
default Navier-Stokes `BarrierTypeII` evaluator with identity cost
\(\mathfrak C_{\mathrm{II}}^{NS}=\tilde{\mathfrak D}_{R_0}\), so
\(K_{\mathrm{CostBridge}}^+\) is automatic in the default backend once the PDE
cost is measurable, nonnegative, and locally integrable on finite
\(\tau\)-intervals. If one deliberately uses a different model cost, the
remaining obligation is the comparison witness \(K_{\mathrm{CostCompare}}^+\).

### 4. Tightness bridge

```{math}
K_{L^3\mathrm{Tight}}^+:
\quad
\text{compact/profile branch certificates}
\Longrightarrow
\forall\varepsilon>0\ \exists R_\varepsilon:
\sup_\tau\int_{|y|>R_\varepsilon}|V(y,\tau)|^3<\varepsilon.
```

Candidate upstream sources are \(K_{C_\mu}^+\), \(K_{\mathrm{Prof}_{NS}}^+\),
\(K_{\mathrm{TB}_O}^+\), and \(K_{\mathrm{RepDesc}_K}^+\). The current PDE
notes do not prove this bridge unconditionally; it is exactly the T8--T10
structural compactness/no-radiation/no-splitting program.

### 5. Windowed \(H^1\) bridge

```{math}
K_{\mathrm{WinH1}}^+:
\quad
K_{\mathrm{RepBridge}}^+
\wedge K_{L^3\mathrm{Bd}}^+
\wedge K_{\mathrm{ModMatrixInv}}^+
\wedge K_{\mathrm{ModForceBd}}^+
\wedge K_{\mathrm{CaccioppoliReg}}^+
\Longrightarrow
\sup_n\int_{\tau_0+n}^{\tau_0+n+1}
\|V(\tau)\|_{H^1(B_m)}^2\,d\tau<\infty
```
for every fixed \(m\).

This is the bridge to the renormalized local windowed \(H^1\) control required
by Theorem A''. The conditional PDE bridge is implemented in
[cluster_c6_windowed_h1_bridge.md](cluster_c6_windowed_h1_bridge.md): global
\(L^\infty_\tau L^3_y\) control, bounded repaired-gauge modulation, pressure
reconstruction, and the renormalized Caccioppoli estimate imply
\(K_{\mathrm{WinH1}}^+\). U5a
[cluster_u5a_bare_data_caccioppoli.md](cluster_u5a_bare_data_caccioppoli.md)
proves the compact-cylinder certificate \(K_{\mathrm{CaccioppoliReg}}^+\) for represented suitable branches, with pressure and AC gauge/modulation supplied by C2.R. This U5a input is local and does not assume global \(L^3\), tightness, bounded modulation, or uniform windowed \(H^1\). The upstream certificate composition is implemented
in
[cluster_c7_win_h1_certificate_composition.md](cluster_c7_win_h1_certificate_composition.md):
\[
K_{\mathrm{RepBridge}}^+
\wedge K_{L^3\mathrm{Bd}}^+
\wedge K_{\mathrm{ModMatrixInv}}^+
\wedge K_{\mathrm{ModForceBd}}^+
\wedge K_{\mathrm{CaccioppoliReg}}^+
\Longrightarrow
K_{\mathrm{WinH1}}^+.
\]
The modulation forcing input in this composition has the T14 reduction
\[
K_{\mathrm{TransForceBd}}^+
\wedge
K_{\mathrm{ScaleForceBd}}^+
\Longrightarrow
K_{\mathrm{ModForceBd}}^+.
\]
The translation forcing part is a compactly supported local estimate from
local \(L^3\), local \(H^1\), and local pressure control, with the local
\(H^1\) input supplied independently of the C7 conclusion. The remaining
pointwise scale part is decomposed in
[cluster_c7_scale_force_bound.md](cluster_c7_scale_force_bound.md) into weighted
Laplacian, weighted fourth-moment, and weighted pressure inputs; the nonlinear
row is discharged from \(K_{\mathrm{ScaleL4Mom}}^+\).
The non-circular good-window replacement is the T15--T17 integrated route:
\[
K_{\mathrm{ModMatrixInv}}^+
\wedge
K_{\mathrm{TransForceL^1Win}}^+
\wedge
K_{\mathrm{ScaleForceL^1Win}}^+
\Longrightarrow
K_{\mathrm{ModL^1Win}}^+.
\]
It gives selected good times with vanishing cost and uniformly bounded
modulation. This route supports selected-time compactness arguments but is not
identical to the pointwise certificate \(K_{\mathrm{ModBd}}^+\).
T18 decomposes the scale-force payload as
\[
K_{\mathrm{ScaleDiffL^1Win}}^+
\wedge
K_{\mathrm{AnnConvReg}}^+
\wedge
K_{\mathrm{ScaleV4L^1Win}}^+
\wedge
K_{\mathrm{ScalePressL^1Win}}^+
\Longrightarrow
K_{\mathrm{ScaleForceL^1Win}}^+.
\]
On represented repaired-gauge branches this is exactly the rough-core blocker:
\[
K_{\mathrm{WinH1}}^+
\Longleftrightarrow
K_{\mathrm{RoughCoreBlk}}^+.
\]

### 6. Rough-core suppression bridge

```{math}
K_{\mathrm{RoughCoreBlk}}^+:
\quad
K_{\mathrm{LS}_\sigma}^+
\ \text{or}\
K_{\mathrm{GC}_\nabla}^+
\Longrightarrow
K_{\mathrm{WinH1}}^+.
```

The dataset currently has \(K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}\) and
\(K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}\), not positive certificates. This is
why the rough-core branch remains possible. If either stiffness or oscillation
control is upgraded, the rough-core branch can be narrowed.

### 7. Radiation suppression bridge

```{math}
K_{\mathrm{RadBlk}}^+:
\quad
K_{\mathrm{TB}_\rho}^+
\ \text{or a scattering/no-radiation backend}
\Longrightarrow
K_{L^3\mathrm{Tight}}^+.
```

The dataset currently has \(K_{\mathrm{TB}_\rho}^{\mathrm{inc}}\), not a
positive mixing certificate. This is why noncompact/radiative Type II remains
possible. If a mixing, scattering, or no-radiation backend is supplied, it
feeds directly into \(K_{L^3\mathrm{Tight}}^+\).

## Sieve-strengthened Type II exclusion theorem

::::{prf:theorem} Sieve-strengthened compact Type II barrier
:label: thm-sieve-strengthened-typeII-barrier

Let \(\mathcal H_{NS}\) be the 3D Navier-Stokes hypostructure in
[../dataset/navier_stokes_3d.md](../dataset/navier_stokes_3d.md). Assume the
sieve run has produced
```{math}
K_{D_E}^+,\quad
K_{C_\mu}^+,\quad
K_{\mathrm{SC}_\lambda}^-,\quad
K_{\mathrm{SC}_{\partial c}}^+,\quad
K_{\mathrm{TB}_\pi}^+,\quad
K_{\mathrm{TB}_O}^+,\quad
K_{\mathrm{RepDesc}_K}^+,\quad
K_{\mathrm{Prof}_{NS}}^+.
```
Assume in addition that the bridge certificates
```{math}
K_{\mathrm{RepBridge}}^+,\quad
K_{L^3\mathrm{Norm}}^+,\quad
K_{L^3\mathrm{Tight}}^+,\quad
K_{\mathrm{WinH1}}^+,\quad
K_{\mathrm{CostBridge}}^+
```
are available for the candidate Type II profile.

Then the candidate emits the blocked Type II barrier certificate
```{math}
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
```
If, in addition, \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) is available, the
blocked certificate is promoted by the NS-valid Type II promotion:
```{math}
K_{\mathrm{SC}_\lambda}^-
\wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}
\Longrightarrow
K_{\mathrm{SC}_\lambda}^{\sim}.
```

Thus the candidate is excluded from the compact finite-cost Type II branch and
is promoted to effective Type II suppression at the scale interface.

::::

:::{prf:proof}
The representation, normalization, tightness, and windowed \(H^1\) bridge
certificates produce exactly the hypotheses of Theorem A'' in
[compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md).
Theorem A'' gives
```{math}
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
```
Applying \(K_{\mathrm{CostBridge}}^+\) converts this PDE divergence statement
into the framework-level blocked Type II certificate
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\). Since
\(K_{\mathrm{SC}_\lambda}^-\) is already produced by the Navier-Stokes dataset
route, the candidate has the inputs to the Type II promotion pattern. The
post-promotion certificate \(K_{\mathrm{SC}_\lambda}^{\sim}\) is emitted only
when the declared NS3D backend also supplies
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\). Without that applicability/replacement
certificate, the rigorous conclusion is the blocked barrier certificate
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\). \(\square\)
:::

## Contrapositive sieve classification

The same theorem gives a useful contrapositive. Suppose a Type II candidate
appears in the Navier-Stokes sieve run and cannot be promoted to
\(K_{\mathrm{SC}_\lambda}^{\sim}\) by the compact Type II barrier route.
Then at least one of the following certificates or bridges is missing or false.

| Failure | Mathematical interpretation | Surviving Type II bucket |
|---|---|---|
| outside declared backend: \(K_{\mathrm{Chart}}^-\) or \(K_{\mathrm{GaugeSolve}}^-\) | The framework profile/germ was not converted into a repaired-gauge orbit. Inside the declared backend C2.R discharges this row. | outside-contract representation diagnostic |
| \(K_{L^3\mathrm{Norm}}^-\) or missing | The candidate was not placed on the critical \(L^3\)-normalized branch. | normalization failure |
| \(K_{L^3\mathrm{Tight}}^-\) or missing | Critical mass is not uniformly tight in renormalized variables. | noncompact/radiative Type II |
| \(K_{\mathrm{WinH1}}^-\) or missing | Local windowed \(H^1\) control fails in a bounded renormalized core. | rough-core Type II |
| \(K_{\mathrm{CostBridge}}^-\) or missing | PDE infinite cost has not been compiled into `BarrierTypeII`. | framework-adapter failure |
| \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) true and \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) available | NS-valid Type II promotion emits \(K_{\mathrm{SC}_\lambda}^{\sim}\). | Type II suppressed at the scale interface |

This is stronger than the purely PDE contrapositive because the sieve identifies
which upstream structural certificates would eliminate each survivor:

- \(K_{\mathrm{TB}_\rho}^+\) or a scattering/no-radiation backend targets the noncompact/radiative branch through \(K_{\mathrm{RadBlk}}^+\).
- \(K_{\mathrm{LS}_\sigma}^+\) or \(K_{\mathrm{GC}_\nabla}^+\) targets the rough-core branch through \(K_{\mathrm{RoughCoreBlk}}^+\).
- \(K_{\mathrm{TB}_O}^+\), \(K_{\mathrm{RepDesc}_K}^+\), and \(K_{\mathrm{Prof}_{NS}}^+\) are natural inputs for proving tightness. The representation bridge is discharged inside the declared backend by C2.R.

## Complete certificate-level classification

The rigorous classification statement that is currently available is a
certificate-level classification, not yet a two-bucket PDE classification. This
distinction matters.

::::{prf:theorem} Complete certificate-level Type II classification
:label: thm-complete-certificate-typeII-classification

Fix a candidate Type II singularity detected by the Navier-Stokes sieve route,
and assume the candidate has entered the supercritical scaling branch
\(K_{\mathrm{SC}_\lambda}^-\). Then exactly one of the following mutually
exclusive certificate outcomes occurs after attempting the compact Type II
barrier compilation:

1. **Suppressed Type II.**
   The certificates
   ```{math}
   K_{\mathrm{RepBridge}}^+,\quad
   K_{L^3\mathrm{Norm}}^+,\quad
   K_{L^3\mathrm{Tight}}^+,\quad
   K_{\mathrm{WinH1}}^+,\quad
   K_{\mathrm{CostBridge}}^+
   ```
   are all present. Then Theorem A'' emits infinite PDE renormalization cost,
   \(K_{\mathrm{CostBridge}}^+\) compiles it into
   \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\). If
   \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) is also available, the NS-valid Type II
   promotion emits \(K_{\mathrm{SC}_\lambda}^{\sim}\).

2. **Representation/normalization defect.**
   At least one of
   ```{math}
   K_{\mathrm{RepBridge}}^+,\qquad K_{L^3\mathrm{Norm}}^+
   ```
   is unavailable or false. The candidate has not yet been placed into the
   normalized repaired-gauge PDE class to which Theorem A'' applies.

3. **Radiative/noncompact Type II.**
   The representation and normalization bridges are present, but
   \(K_{L^3\mathrm{Tight}}^+\) fails. Equivalently, in the repaired-gauge
   renormalized variables,
   ```{math}
   \exists\varepsilon_0>0\ \forall R>0\ \exists\tau_R:
   \int_{|y|>R}|V(y,\tau_R)|^3\,dy\ge\varepsilon_0.
   ```

4. **Rough-core Type II.**
   The representation, normalization, and tightness bridges are present, but
   \(K_{\mathrm{WinH1}}^+\) fails. Equivalently, there is a bounded
   renormalized core on which local windowed \(H^1\) control fails:
   ```{math}
   \exists m\ge1:
   \sup_n\int_{\tau_0+n}^{\tau_0+n+1}
   \|V(\tau)\|_{H^1(B_m)}^2\,d\tau=\infty.
   ```

5. **Cost-adapter defect.**
   The PDE hypotheses of Theorem A'' are available and the PDE theorem gives
   infinite localized renormalization cost, but \(K_{\mathrm{CostBridge}}^+\)
   is unavailable or false. The PDE barrier has not yet been compiled into the
   framework's `BarrierTypeII` blocked certificate.

These alternatives are exhaustive at the certificate level.

::::

:::{prf:proof}
Attempt the bridge checks in the ordered list
```{math}
K_{\mathrm{RepBridge}},\quad
K_{L^3\mathrm{Norm}},\quad
K_{L^3\mathrm{Tight}},\quad
K_{\mathrm{WinH1}},\quad
K_{\mathrm{CostBridge}}.
```
Typed certificate logic gives a positive, negative, or missing/inconclusive
outcome for each check. If all checks are positive, Theorem A'' gives the
blocked barrier certificate, and alternative 1 follows only when
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\) is also available. If the first or second
check is not positive,
we are in alternative 2. If the first two are positive and tightness is not
positive, we are in alternative 3. If tightness is positive but windowed
\(H^1\) is not positive, we are in alternative 4. If the PDE hypotheses are
positive but the cost compilation is not positive, we are in alternative 5.
The cases are disjoint by construction of the ordered first-failure
classification and exhaustive because every attempted certificate evaluation
has a typed outcome. \(\square\)
:::

## When the classification collapses to the two PDE buckets

The certificate-level theorem becomes the desired PDE classification once the
representation, normalization, and cost-adapter defects are discharged.

::::{prf:corollary} Two-bucket classification after bridge discharge
:label: cor-two-bucket-typeII-classification

Assume every Type II candidate in the declared Navier-Stokes sieve branch
satisfies
```{math}
K_{\mathrm{RepBridge}}^+,\qquad
K_{L^3\mathrm{Norm}}^+,\qquad
K_{\mathrm{CostBridge}}^+.
```
Assume also \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) for the declared NS3D backend.
Then every Type II candidate is either:

1. suppressed by the compact Type II barrier and NS-valid Type II promotion,
2. radiative/noncompact, meaning \(K_{L^3\mathrm{Tight}}^+\) fails,
3. rough-core, meaning \(K_{\mathrm{WinH1}}^+\) fails.

If, in addition, the candidate is assumed to have finite framework Type II cost
and the cost bridge is bidirectional on the declared branch, then every
finite-cost non-suppressed Type II candidate is either radiative/noncompact or
rough-core.

::::

:::{prf:proof}
Apply Theorem {prf:ref}`thm-complete-certificate-typeII-classification`.
The assumed bridge certificates eliminate alternatives 2 and 5. The remaining
alternatives are suppressed, radiative/noncompact, or rough-core. Under the
additional finite-cost and bidirectional cost-bridge assumptions, the suppressed
alternative is incompatible with finite non-suppressed Type II, leaving only the
two PDE survivor buckets. \(\square\)
:::

## What must be proved to make the classification fully PDE-intrinsic

The theorem above is rigorous because it classifies certificate outcomes. To
make it a classification of actual Navier-Stokes Type II singularities rather
than a classification of the current proof state, the following bridge
completeness statements must be proved.

### Classification-completeness axiom package

Define
```{math}
K_{\mathrm{ClassComplete}}^+
:=
K_{\mathrm{RepBridge}}^+
\wedge
K_{L^3\mathrm{Norm}}^+
\wedge
K_{\mathrm{CostBridge}}^+
\wedge
K_{\mathrm{TypeIIExhaust}}^+.
```

Here \(K_{\mathrm{TypeIIExhaust}}^+\) means every Type II candidate in the
declared Navier-Stokes backend is detected by the sieve branch
\(K_{C_\mu}^+\wedge K_{\mathrm{SC}_\lambda}^-\) and is represented by the
profile package \(K_{\mathrm{Prof}_{NS}}^+\).

This exhaustion certificate is implemented in
[cluster_c1_typeII_branch_exhaustion.md](cluster_c1_typeII_branch_exhaustion.md)
as a declared-backend theorem. For the NS3D dataset, that note explicitly
provides the concrete payload \(K_{\mathrm{ExhPayload},NS3D}^+\) from the
dataset outputs

```{math}
K_{C_\mu}^+,\qquad
K_{\mathrm{SC}_\lambda}^-,\qquad
K_{\mathrm{Prof}_{NS}}^+.
```

Failures are classified as

```{math}
K_{\mathrm{CmuExtract}}^-,
\qquad
K_{\mathrm{ScaleRoute}}^-,
\qquad
K_{\mathrm{ProfComplete}}^-,
\qquad
K_{\mathrm{LostTypeII}}^-.
```

With this package, the remaining Type II universe is rigorously reduced to:

```{math}
\text{suppressed}
\quad\vee\quad
\text{radiative/noncompact}
\quad\vee\quad
\text{rough-core}.
```

To rule out all Type II singularities, one must further prove:

```{math}
K_{\mathrm{RadBlk}}^+
\quad\text{and}\quad
K_{\mathrm{RoughCoreBlk}}^+.
```

That would give
```{math}
K_{\mathrm{ClassComplete}}^+
\wedge
K_{\mathrm{RadBlk}}^+
\wedge
K_{\mathrm{RoughCoreBlk}}^+
\wedge
K_{\mathrm{NS\text{-}UPTypeII}}^+
\Longrightarrow
\text{no admissible unresolved Type II branch in the declared sieve backend}.
```

This remains weaker than global regularity: it rules out the Type II branch
inside the promoted declared backend, but says nothing by itself about Type I
scenarios, continuation criteria, or the final Lock.

## Practical sieve use

To strengthen the current Type II exclusion theorem, run the Navier-Stokes
sieve only up to the structural certificates needed for the Type II branch.
Do not invoke the final global regularity route.

The useful import package is:

```{math}
\mathcal K_{\mathrm{TypeII,Sieve}}
=
\left(
K_{D_E}^+,
K_{C_\mu}^+,
K_{\mathrm{SC}_\lambda}^-,
K_{\mathrm{SC}_{\partial c}}^+,
K_{\mathrm{Cap}_H}^+,
K_{\mathrm{TB}_\pi}^+,
K_{\mathrm{TB}_O}^+,
K_{\mathrm{RepDesc}_K}^+,
K_{\mathrm{Prof}_{NS}}^+
\right).
```

Then the Type II proof ledger becomes:

| Target | Best certificate route | Status |
|---|---|---|
| repaired-gauge renormalized orbit | \(K_{\mathrm{RepBridge}}^+\) from \(K_{C_\mu}^+\wedge K_{\mathrm{Prof}_{NS}}^+\) plus the C2 payloads | implemented in `cluster_c2_representation_bridge.md`; backend payload defects are classified |
| critical \(L^3\) normalization | \(K_{L^3\mathrm{Norm}}^+\), implemented as positive finite critical-mass annulus control | implemented in `cluster_c3_l3_normalization_bridge.md`; normalization defects are classified |
| global critical tightness | \(K_{L^3\mathrm{Tight}}^+\) from compact/profile/tame/finite-description data | T8--T10 target |
| local windowed \(H^1\) control / rough-core blocker | \(K_{\mathrm{WinH1}}^+\equiv K_{\mathrm{RoughCoreBlk}}^+\) from global \(L^3\), Caccioppoli, pressure reconstruction, and bounded repaired-gauge modulation | PDE bridge implemented in `cluster_c6_windowed_h1_bridge.md`; full C7 blocker implemented in `cluster_c7_win_h1_certificate_composition.md`; bounded critical norm discharged from C3 in `cluster_c7_l3bd_defect_discharge.md`; modulation inverse discharged in `cluster_c7_modmatrix_inverse_discharge.md` |
| finite-cost compact branch exclusion | Theorem A'' | implemented |
| finite-cost contrapositive | Corollary 16 | implemented |
| compile PDE infinite cost into `BarrierTypeII` | \(K_{\mathrm{CostBridge}}^+\) | implemented in `cluster_c4_cost_bridge.md` for the default identity-cost backend; non-default costs require \(K_{\mathrm{CostCompare}}^+\) |
| post-UP Type II suppression | \(K_{\mathrm{SC}_\lambda}^{\sim}\) | available once \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) is compiled and \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) is supplied |

## Lowest-hanging improvements

The most immediate way to use the sieve to narrow Type II is not to prove the
Lock. It is to discharge bridge certificates in this order:

1. Finish the remaining C7 rough-core inputs after the C3 bounded-norm and
   repaired-gauge modulation-inverse discharges: \(K_{\mathrm{ModForceBd}}^+\)
   and \(K_{\mathrm{CaccioppoliReg}}^+\). U5a now discharges
   \(K_{\mathrm{CaccioppoliReg}}^+\) on represented suitable branches by a local
   compact-cylinder argument, so the immediate remaining rough-core input is
   the modulation-force side. The pointwise scale-force side of
   \(K_{\mathrm{ModForceBd}}^+\) now reduces to \(K_{\mathrm{ScaleLapBd}}^+\),
   \(K_{\mathrm{ScaleL4Mom}}^+\), and \(K_{\mathrm{ScalePressureBd}}^+\), plus
   the independent translation-force hypotheses. If C2/C5 representation is not
   being imported, \(K_{\mathrm{RepBridge}}^+\) is still the first ordered input;
   if the repaired-gauge nondegeneracy payload is not imported,
   \(K_{\mathrm{ModMatrixInv}}^+\) remains the gauge-layer input.
2. Prove \(K_{L^3\mathrm{Tight}}^+\) using the compact/profile/tame and
   finite-description certificates. This attacks the radiative/noncompact
   bucket.
3. Supply \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) wherever the output needs the
   promoted post-UP suppression certificate \(K_{\mathrm{SC}_\lambda}^{\sim}\),
   not merely the blocked Type II barrier.
4. Upgrade \(K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}\),
   \(K_{\mathrm{TB}_\rho}^{\mathrm{inc}}\), or
   \(K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}\) only if the corresponding backend
   is actually available. These upgrades would further suppress rough-core or
   radiative survivor mechanisms.

This gives a precise sieve agenda: every surviving Type II candidate must now
be explained as a failed bridge or a failed positive certificate, not merely as
an unspecified gap in the compactness proof.
