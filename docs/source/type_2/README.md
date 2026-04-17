# Compact Type II proof architecture

This folder organizes a conditional compact Type II exclusion program for the 3D incompressible Navier-Stokes equations. The main reference is [compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md). The other files isolate reusable functional-analytic tools, repaired gauge identities, and theorem clusters needed to close the master argument.

The result is not a bare-data exclusion of all Type II singularities. It proves a barrier theorem for a compact, normalized, gauge-fixed renormalized orbit under explicit compactness, pressure, modulation, and local regularity hypotheses.

## Hypostructure DAG convention

The proof stack is a directed acyclic certificate graph.  A theorem is always
read at its current node.  All upstream certificates already emitted on the
path to that node are safe assumptions, and no downstream theorem reopens an
upstream branch.  If a live certificate fails at the current node, the output
is a named downstream defect stratum, not a request to revisit scattering,
concentration, terminal routing, or another earlier decision.

For example, after the route emits \(K_{\mathrm{ScattBranch}}^-\), the
scattering branch is no longer part of the local state space.  Later
multibubble and representation nodes may assume that branch has been removed.
Likewise, U2a is not a theorem about arbitrary Navier-Stokes solutions; it is
the representation identity theorem on the represented node.

## Current folder status

The compact single-core barrier is assembled. The master note proves the
good-window contradiction once the represented branch has positive finite
critical \(L^3\) mass, global \(L^3\)-tightness, and uniform local
windowed \(H^1\) control.

The implemented bridge stack now contains:

1. pressure control and local \(H^{-1}\) time regularity, via T1--T6;
2. good-window compactness and the finite-cost contrapositive, via T7 and
   Theorem A'';
3. conditional no-radiation and no-splitting criteria, via T8--T10;
4. classification-completeness bridges C1--C5;
5. radiative and rough-core blocker routes C6--C8;
6. NS-valid Type II promotion and formal `UP-TypeII` application checklists,
   via C9--C18;
7. pointwise and integrated modulation-force reductions, via T14--T18;
8. same-point multiscale cascade exhaustion and regular cascade exclusion,
   via S5;
9. multibubble camera reduction to same-point and separated-point
   no-splitting payloads, via S6;
10. discharge of both multibubble decoupling payloads from the nonlinear
    profile-evolution theorem \(K_{\mathrm{NLProfDec},NS3D}^+\), via S7.
11. proof of \(K_{\mathrm{NLProfDec},NS3D}^+\) for terminal active cameras
    from terminal profile completeness, emitted \(K_{\mathrm{ScattBranch}}^-\), exterior-regular
    discard, repaired-gauge representation, and Caccioppoli regularity, via
    S8.
12. discharge of the small-data scattering branch, exterior-regular discard,
    and Caccioppoli regularity certificates, via S9--S11 and U5a;
13. discharge of terminal profile completeness from the standard critical
    Navier-Stokes profile decomposition and small-data stability theorem, via
    S12;
14. discharge of the bounded terminal critical sequence input from the C3
    \(L^3\)-normalization branch and terminal sequence routing, via S13;
15. discharge of terminal sequence routing from repaired-gauge representation,
    via S14. Terminal-camera construction is built into the declared S8
    terminal backend, so it is no longer an independent analytic hypothesis.

The remaining unexcluded Type II scenarios are not pressure-tail failures,
local pressure failures, repaired-gauge matrix degeneracy after its payload is
imported, or singular-boundary terms in the convective scale row. Those have
been converted into proved estimates or explicit upstream certificates. The
remaining scenarios are the survivor buckets listed in
[Current Type II singularity classification](#current-type-ii-singularity-classification).

The current low-hanging terminal-backend gaps have been narrowed as follows.
The terminal-camera construction certificate is now definitional in the S8
terminal backend, and S14 proves
\[
K_{\mathrm{RepBridge}}^+
\Longrightarrow
K_{\mathrm{TermSeqFromOrbit}}^+ .
\]
S12 now records Kato small-data \(L^3\) stability as the theorem emitting
\(K_{\mathrm{SmallDataStab}_{L^3}}^+\), and it states the accepted critical
Navier-Stokes profile theorem in the exact terminal-profile form needed by
S8. S10 now proves that positive-physical-distance profiles are exterior:
they vanish in compact Type II core cameras and their pressure contribution is
harmless modulo constants.

Thus \(K_{\mathrm{TermCamConstruct}}^+\) and
\(K_{\mathrm{TermSeqFromOrbit}}^+\) are not independent remaining blockers
inside the declared terminal backend. U3a separates subsequential
nontriviality from the extra no-return input needed to remove the zero
critical-mass branch:
\[
K_{\mathrm{CorePersist}}^+
\wedge
K_{\mathrm{CoreSubseqNontriv}}^+
\Longrightarrow
\neg K_{L^3\mathrm{Zero}}^- .
\]
At the current DAG frontier, the remaining live downstream strata/payload
nodes are:
\[
K_{\mathrm{ClassComplete}}^+,\quad
K_{\mathrm{RepBridge}}^+,\quad
K_{\mathrm{CorePersist}}^+,\quad
K_{\mathrm{RawOrbL3},NS3D}^+,\quad
K_{\mathrm{StateStratExh}}^+ and K_{\mathrm{StratCritMass}}^+,\quad
K_{\mathrm{C6Route}}^+,\quad
K_{\mathrm{CaccioppoliReg}}^+,\quad
K_{\mathrm{S3NRSPayload}}^+,
\]
together with the imported critical profile theorem
\(K_{\mathrm{CriticalNSProfDecomp}}^+\) and, if one wants the formal
hypostructure output rather than only a blocked Type II certificate,
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\).

After the technical bridge payloads, including the C2.R representation-discharge payload, single-point exterior discard payload,
rough-core Caccioppoli payload, and S3 scale-collapse reduction payloads are
supplied, every non-multibubble Type II stratum is closed. S5 also removes
regular strict same-point cascades under its profile-decoupling,
active-bubble mass-floor, inner-representation, and perturbative S3 payloads.
The remaining failures are downstream payload defects at their respective
nodes: bounded critical terminal sequence failure for the S12 profile theorem,
exterior discard, repaired-gauge representation, Caccioppoli regularity, or
S3 rigidity.
Technical bridge failures remain in the ledger as payload defects, not as
additional singularity classes.

S6 refines the multibubble residue. If
\[
K_{\mathrm{SamePointNLDec}}^+
\wedge
K_{\mathrm{MultiPointCamDec}}^+
\]
is supplied, then every multibubble candidate reduces to a single-camera S3
contradiction. Without those payloads, the exact multibubble obstructions are
\(K_{\mathrm{SamePointNLDec}}^-\) and
\(K_{\mathrm{MultiPointCamDec}}^-\).
S7 discharges both payloads from the single nonlinear profile-evolution
decoupling theorem
\[
K_{\mathrm{NLProfDec},NS3D}^+.
\]
S8 proves this theorem in the terminal-camera form. Therefore multibubble
failure is no longer an independent singularity class inside the declared
terminal-complete backend; it can occur only through an upstream failure of
bounded critical terminal sequence/profile-theorem input, exterior-regular
discard, repaired-gauge representation, Caccioppoli regularity, or the S3
rigidity payload.  The scattering branch has already been removed by
\(K_{\mathrm{ScattBranch}}^-\) before this node.

## Certificate ledger for excluded Type II strata

Every Type II exclusion statement in this folder is conditional on an explicit
certificate package. The table below records the singularity stratum, the
certificates required to exclude it, and the theorem file proving the
exclusion.

| Type II stratum | Required certificates | Proof location |
|---|---|---|
| Represented compact single-core candidate | \(K_{\mathrm{RepBridge}}^+\) (discharged by C2.R from \(K_{\mathrm{RepDischarge},NS3D}^+\)), \(K_{\mathrm{StratCritPacket}}^+\), \(K_{L^3\mathrm{Tight}}^+\), \(K_{\mathrm{WinH1}}^+\), pressure reconstruction, bounded modulation, and the good-window compactness inputs | Theorem A'' in [compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md), with T1--T7 supplied by [cluster_next_easiest_theorems.md](cluster_next_easiest_theorems.md) and [cluster_t7_L3_compactness_upgrade.md](cluster_t7_L3_compactness_upgrade.md) |
| Finite-cost non-suppressed branch classification | \(K_{\mathrm{ClassComplete}}^+\), \(K_{\mathrm{CostBridge}}^+\), and the represented positive finite critical-mass branch | C4 in [cluster_c4_cost_bridge.md](cluster_c4_cost_bridge.md) and C5 in [cluster_c5_two_bucket_classification.md](cluster_c5_two_bucket_classification.md) |
| Small-data scattering profile branch | Small critical profile norm \(\|\phi\|_3\le\varepsilon_{\mathrm{sd}}\), Kato critical mild stability, and the scattering ledger | S9 in [cluster_s9_scattering_branch_discharge.md](cluster_s9_scattering_branch_discharge.md), proving \(K_{\mathrm{ScattBranch}}^-\) |
| Non-bubble radiative/noncompact branch | A positive tightness blocker \(K_{\mathrm{RadBlk}}^+\equiv K_{L^3\mathrm{Tight}}^+\), supplied by one of the C6 routes: global \(L^3\)-compactness, compact core plus vanishing remainder, no-radiation/no-splitting, finite-library approximation, or tame finite-description compact realization | C6 in [cluster_c6_radiative_noncompact_exclusion.md](cluster_c6_radiative_noncompact_exclusion.md) |
| Physically exterior far radiation | \(K_{\mathrm{SinglePointBlowup}}^+\wedge K_{\mathrm{ExtRegDiscard}}^+\) | S4 in [cluster_s4_multibubble_residue_classification.md](cluster_s4_multibubble_residue_classification.md), with \(K_{\mathrm{ExtRegDiscard}}^+\) proved in S10 [cluster_s10_exterior_regular_discard.md](cluster_s10_exterior_regular_discard.md) |
| Rough-core branch \(K_{\mathrm{WinH1}}^-\) | \(K_{\mathrm{RepBridge}}^+\), \(K_{L^3\mathrm{Bd}}^+\), \(K_{\mathrm{ModMatrixInv}}^+\), \(K_{\mathrm{ModForceBd}}^+\), \(K_{\mathrm{CaccioppoliReg}}^+\), and pressure reconstruction | C6/T13 in [cluster_c6_windowed_h1_bridge.md](cluster_c6_windowed_h1_bridge.md), composed by C7 in [cluster_c7_win_h1_certificate_composition.md](cluster_c7_win_h1_certificate_composition.md); \(K_{\mathrm{CaccioppoliReg}}^+\) is proved from compact-cylinder physical suitability and declared repaired-gauge representation in U5a [cluster_u5a_bare_data_caccioppoli.md](cluster_u5a_bare_data_caccioppoli.md), strengthening S11 [cluster_s11_caccioppoli_regularity_discharge.md](cluster_s11_caccioppoli_regularity_discharge.md); U5a does not assume global \(L^3\), tightness, bounded modulation, or uniform windowed \(H^1\); bounded critical norm and modulation-matrix inverse are discharged by [cluster_c7_l3bd_defect_discharge.md](cluster_c7_l3bd_defect_discharge.md) and [cluster_c7_modmatrix_inverse_discharge.md](cluster_c7_modmatrix_inverse_discharge.md) |
| Vorticity-controlled rough-core subtype | \(K_{\mathrm{RepBridge}}^+\wedge K_{L^3\mathrm{Bd}}^+\wedge K_{\mathrm{VortL^2Win}}^+\) | S2 in [cluster_s2_rough_core_vorticity_exclusion.md](cluster_s2_rough_core_vorticity_exclusion.md) |
| Scale-collapse drift | Either \(K_{\mathrm{ScaleCollapseCostBridge}}^+\) or \(K_{\mathrm{AbsScaleCostBridge}}^+\) for the cost-barrier route, or the S3 compactness, autonomous-modulation, stationary-limit, and Liouville/NRŠ payloads \(K_{\mathrm{S3NRSPayload}}^+\) | C16 in [cluster_c16_scale_negative_drift_dichotomy.md](cluster_c16_scale_negative_drift_dichotomy.md), C17 in [cluster_c17_scale_collapse_barrier.md](cluster_c17_scale_collapse_barrier.md), and S3 in [cluster_s3_scale_collapse_attractor_stratification.md](cluster_s3_scale_collapse_attractor_stratification.md) |
| Regular strict same-point cascade | S5 profile-decoupling, active-bubble mass floor, inner-representation, perturbative S3 robustness, and \(K_{\mathrm{S3NRSPayload}}^+\) | S5 in [cluster_s5_multiscale_cascade_exclusion.md](cluster_s5_multiscale_cascade_exclusion.md) |
| General same-point multibubble and separated multipoint multibubble | \(K_{\mathrm{TechTypeII}}^{\mathrm{blk},+}\), \(K_{\mathrm{StratCritPacket}}^+\), \(K_{\mathrm{SmallDataStab}_{L^3}}^+\), \(K_{\mathrm{CriticalNSProfDecomp}}^+\), \(K_{\mathrm{ScattBranch}}^-\), \(K_{\mathrm{ExtRegDiscard}}^+\), \(K_{\mathrm{RepBridge}}^+\), \(K_{\mathrm{CaccioppoliReg}}^+\), and \(K_{\mathrm{S3NRSPayload}}^+\) | S6 in [cluster_s6_multibubble_camera_reduction.md](cluster_s6_multibubble_camera_reduction.md), S7 in [cluster_s7_decoupling_payload_discharge.md](cluster_s7_decoupling_payload_discharge.md), S8 in [cluster_s8_terminal_nonlinear_profile_decoupling.md](cluster_s8_terminal_nonlinear_profile_decoupling.md), S12 in [cluster_s12_terminal_profile_completeness.md](cluster_s12_terminal_profile_completeness.md), S13 in [cluster_s13_bounded_critical_terminal_sequences.md](cluster_s13_bounded_critical_terminal_sequences.md), and S14 in [cluster_s14_terminal_sequence_routing.md](cluster_s14_terminal_sequence_routing.md), with \(K_{\mathrm{ScattBranch}}^-\), \(K_{\mathrm{ExtRegDiscard}}^+\), and \(K_{\mathrm{CaccioppoliReg}}^+\) proved in S9--S11 and strengthened by U5a |
| Promotion from blocked Type II to post-UP suppression | \(K_{\mathrm{NS\text{-}UPTypeII}}^+\), \(K_{\mathrm{FormalUPTypeIIApp},NS3D}^+\), emitted \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\), and the localized monotonicity/finite-error payloads | C10--C14 in [cluster_c10_ns_up_typeII_promotion.md](cluster_c10_ns_up_typeII_promotion.md), [cluster_c11_generic_up_typeII_payload_discharge.md](cluster_c11_generic_up_typeII_payload_discharge.md), [cluster_c12_ns_localized_monotonicity_translation.md](cluster_c12_ns_localized_monotonicity_translation.md), [cluster_c13_formal_up_typeII_application.md](cluster_c13_formal_up_typeII_application.md), and [cluster_c14_explicit_up_typeII_ns3d_checklist.md](cluster_c14_explicit_up_typeII_ns3d_checklist.md) |

Thus the folder excludes exactly those Type II strata for which the
corresponding certificate package in this ledger is supplied. If one package
is missing, the candidate is classified by the named missing certificate
rather than by an unnamed singularity mechanism.

## Full proof strategy

The strategy is a classification-and-exclusion pipeline for declared NS3D Type
II candidates.

1. **Enter the declared Type II backend.** C1 is the exhaustion gate. It asks the
   backend to supply concentration extraction, scale routing, profile
   completeness, and no lost Type II branch. If this fails, the candidate is a
   backend/profile defect, not a PDE survivor class.
2. **Build the repaired-gauge orbit.** C2 and C2.R turn
   the profile branch into a renormalized Navier-Stokes orbit
   \((V,P,a,b)\), with pressure reconstruction, repaired gauge, and admissible modulation.
   C2.R shows that chart extraction plus an admissible AC repaired-gauge solve discharge the bridge; pressure pullback and modulation coefficients are then theorems, while T1--T6 supply later pressure estimates, time regularity, and modulation nondegeneracy inputs.
3. **Stratify the critical mass.** The terminal backend no longer uses a
   global \(L^3\) normalization of the full solution. U3b works stratum by
   stratum: every retained active terminal profile state
   \(\Phi_{\mathfrak s}\) has
   \(0<\|\Phi_{\mathfrak s}\|_3<\infty\), and bounded terminal packets emit
   \(K_{\mathrm{StratCritPacket}}^+\). Global critical-mass pathologies are
   routed through the radiation, multistrata, cascade, rough-core, or
   lost-profile ledgers, not used in the local stratum estimate.
4. **Compile the compact single-core barrier.** Theorem A'' says that a
   represented orbit with positive finite critical mass, global
   \(L^3\)-tightness, and windowed local \(H^1\) control cannot have finite
   localized Type II cost. C4 compiles this PDE divergence into the declared
   `BarrierTypeII` blocked certificate.
5. **Classify finite-cost failures.** C5 says that a finite-cost, non-suppressed
   candidate that passes the bridge stack must fail either global
   \(L^3\)-tightness or windowed local \(H^1\). These are the two basic
   noncompact/radiative and rough-core branches.
6. **Remove non-bubble radiation and isolate bubbles.** C6 gives
   no-radiation/tightness routes. S4 separates physically far radiation from
   genuine core radiation: under single-point blowup and exterior-regular
   discard, far radiation is removed from the Type II core ledger. What remains
   in the radiative bucket is multibubble radiation: a secondary same-point
   scale or another physical concentration center.
7. **Remove regular same-point cascades.** S5 proves that bounded critical mass
   gives finite cascade length, and that a regular strict same-point cascade
   reduces at the innermost scale to a perturbative single-bubble S3 branch.
   Under S5's decoupling, active-bubble, inner-expansion, and perturbative S3
   payloads, regular strict same-point cascades are excluded. The same-point
   residue that remains is the non-regular no-splitting/profile-compatibility
   case.
8. **Remove rough cores.** C6/T13 proves windowed local \(H^1\) from bounded
   critical \(L^3\), pressure reconstruction, bounded modulation, and
   Caccioppoli regularity. C7 records the certificate composition and ordered
   defects. S2 adds a vorticity-controlled subtype exclusion, but the main
   rough-core closure is the Caccioppoli bridge.
9. **Route scale-collapse drift.** C16 isolates the negative-scale drift
   alternative. C17 closes it if the backend accepts the scale-collapse cost as
   a Type II barrier. S3 gives the PDE alternative: on a common window sequence,
   compactness plus autonomous modulation plus a stationary omega-limit plus a
   parameter-specific Liouville theorem blocks the scale-collapse branch. If
   those S3 payloads are missing, the missing item is a named analytic defect,
   not a new singularity class.
10. **Promote blocked Type II when licensed.** C10--C14 explain when a blocked
   Type II outcome can be promoted to \(K_{\mathrm{SC}_\lambda}^{\sim}\)
   through the formal `UP-TypeII` theorem. Without the promotion payload, the
   rigorous output remains a blocked Type II certificate.
11. **Close the multibubble residue in the terminal backend.** After the
    technical bridge payloads and closeable non-bubble strata are supplied, S5
    removes regular strict same-point cascades, S6 reduces all remaining
    multibubble cases to same-point or separated-point decoupling payloads,
    S7 discharges those payloads from nonlinear profile decoupling, S8 proves
    nonlinear profile decoupling in terminal active cameras, and S14--S13--S12
    supply the terminal profile-completeness route. Thus multibubble is no
    longer an independent survivor inside the declared terminal-complete
    backend. It can only reappear as one of the named upstream defects:
    repaired-gauge representation failure, critical-mass defect, critical
    profile theorem failure, scattering-removal failure, exterior-discard
    failure, Caccioppoli failure, or S3 rigidity failure.

Thus the folder does not claim a bare-data global regularity proof. It proves
the declared terminal-backend Type II exclusion once the explicit C18 terminal
payload is supplied. Outside that terminal backend, every failure is named as a
certificate defect rather than left as an unnamed singularity mechanism.

## Main contradiction mechanism

The proof starts from a renormalized candidate singularity
\[
u(x,t)=\lambda(t)^{-1}V\left(\frac{x-x_c(t)}{\lambda(t)},\tau(t)\right),
\qquad
\tau_t=\lambda(t)^{-2}.
\]
The renormalized profile solves
\[
\partial_\tau V +(V\cdot\nabla)V+\nabla P
=\nu\Delta V+a(\tau)(V+y\cdot\nabla V)+b(\tau)\cdot\nabla V,
\qquad \nabla\cdot V=0.
\]

The barrier uses the nonnegative localized renormalization cost
\[
\tilde{\mathfrak D}_{R_0}(\tau)
=
\nu\int |\nabla V|^2\phi_{R_0}
+a_+(\tau)\int |V|^2\phi_{R_0}.
\]
If this cost were integrable in renormalized time, then nonnegativity gives a sequence \(\tau_n\to\infty\) such that
\[
\tilde{\mathfrak D}_{R_0}(\tau_n)\to0.
\]
The local energy/Caccioppoli layer turns this into vanishing core dissipation:
\[
\int_{B_{R_0}}|\nabla V(\tau_n)|^2\to0.
\]
Compactness and time regularity then produce a strong local limit of \(V(\tau_n)\). The low-dissipation rigidity lemma says a normalized, tight, strongly compact sequence cannot converge this way while losing all core dissipation. This contradiction forces
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty
\]
for every orbit satisfying the stated hypotheses.

## How the files fit together

[compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md) is the complete internal proof chain. It contains the renormalized equation, local energy identities, Caccioppoli estimate, pressure-tail control, verification of the local \(L^2\) pressure hypotheses, the local \(L^2\) pressure estimate modulo constants, time-regularity lemma, Aubin-Lions closure, low-dissipation rigidity theorem, final conditional barrier theorem, repaired gauge appendix, and Theorem A'' good-window closure.

[required_new_scale_gauge_theorems.md](required_new_scale_gauge_theorems.md) proves the repaired weighted-moment scale gauge facts used by the master note. The key identity is
\[
DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)]=p\Theta_0
\]
on the gauge surface. This fixes the scale row of the modulation matrix without relying on annular lower bounds.

[cluster_next_easiest_theorems.md](cluster_next_easiest_theorems.md) packages T1--T6 from the roadmap. It records pressure-tail control, verification of the T1.5 hypotheses from \(L^\infty_\tau L^3_y\) and local \(L^2_\tau H^1_y\) control (T1.4), the local \(L^2_\tau L^2_y\) pressure estimate modulo constants (T1.5), translation-block invertibility, cross-term control, full repaired-gauge nondegeneracy, bounded modulation parameters \((a,b)\), and local \(H^{-1}\) time regularity.

[cluster_t7_L3_compactness_upgrade.md](cluster_t7_L3_compactness_upgrade.md) packages T7. It separates spacetime compactness from fixed sampled-time compactness: Aubin-Lions gives strong spacetime \(L^2\), interpolation gives spacetime \(L^3\), and sampled-time \(L^3\) requires sampled-time \(L^2\)-compactness plus a local \(L^6\)-type bound. Its T7.4 good-time lemma discharges the sampled-time compactness input when low average cost and local \(L^2_\tau H^1_y\) control hold on common windows.

[cluster_structural_t8_t11.md](cluster_structural_t8_t11.md) starts the structural cluster with the conditional form of T8, the no-radiation criteria T9, and the no-splitting criteria T10. It proves that a globally \(L^3\)-compact gauge-fixed branch is uniformly \(L^3\)-tight, sampled-precompact in \(L^3\), and compatible with the compact Type II barrier once the local windowed \(H^1\) bounds are available. It also proves that compact-core decompositions with vanishing critical \(L^3\) remainders, or equivalent \(L^3\)-decoupling certificates, eliminate radiative critical tails. Finally, it proves that asymptotic \(L^3\)-decoupling plus single-profile saturation kills all secondary profiles and remainders. The implication from abstract Type II branch membership to global single-profile \(L^3\)-compactness belongs to the no-radiation/no-splitting/irreducibility program.

[abstract_and_ns_theorems.md](abstract_and_ns_theorems.md) isolates reusable abstract tools: compactness implies tightness, Aubin-Lions compactness, trace compactness, and the 3D Navier-Stokes instantiation.

[type2_roadmap.md](type2_roadmap.md) tracks theorem-level status. It separates implemented technical closure notes from the global compactness/no-radiation/no-splitting program.

[hypostructure_sieve_typeII_certificates.md](hypostructure_sieve_typeII_certificates.md) links this PDE proof stack to the Hypostructure sieve and to the Navier-Stokes dataset certificate trail. It does not use the global regularity conclusion; instead it records which post-sieve certificates can be imported to strengthen the Type II exclusion theorem, which bridge certificates remain open, and why post-UP Type II suppression for NS3D requires the explicit \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) bridge.

[ns3d_repaired_gauge_backend_contract.md](ns3d_repaired_gauge_backend_contract.md) fixes the local meaning of the declared NS3D repaired-gauge Type II backend. It states the backend universe, the route certificate \(K_{\mathrm{TypeIIRoute}}^+\), the NS3D representation payload, the critical \(L^3\)-mass output, the identity `BarrierTypeII` cost convention used by C1--C4, and the boundary between blocked Type II and post-promotion suppression.

[cluster_c1_typeII_branch_exhaustion.md](cluster_c1_typeII_branch_exhaustion.md) implements the declared Type II branch exhaustion certificate \(K_{\mathrm{TypeIIExhaust}}^+\). It proves that, once the Navier-Stokes backend supplies concentration extraction, scale routing, profile completeness, and no-leakage payloads, every declared Type II branch enters the C2--C4 bridge ledger.

[cluster_u3a_zero_critical_mass_exclusion.md](cluster_u3a_zero_critical_mass_exclusion.md) implements U3a. It proves that concentration extraction gives subsequential nontriviality, and that a retained core with the additional persistence/no-return payload cannot emit the zero-collapse defect for that retained core. U3b supersedes the global critical-mass route in the terminal backend by working locally on terminal strata.

[cluster_c2_representation_bridge.md](cluster_c2_representation_bridge.md) implements the representation bridge from sieve/profile certificates to repaired-gauge renormalized Navier-Stokes orbits. It defines the raw orbit, repaired-gauge, pressure, and modulation payloads needed to emit \(K_{\mathrm{RepBridge}}^+\), and classifies representation defects when a payload is missing.

[cluster_c2_repbridge_payload_discharge.md](cluster_c2_repbridge_payload_discharge.md) refines C2 by discharging the high-level representation payload into the two genuine inputs: raw chart extraction and repaired-gauge solve with the required time reparametrization. It proves pressure pullback and final modulation-coefficient realization as consequences of those inputs. Once \(K_{\mathrm{RepDischarge},NS3D}^+\) is supplied, \(K_{\mathrm{RepBridge}}^+\) is a theorem rather than a separate technical assumption.

[cluster_u2a_repaired_gauge_representation_pieces.md](cluster_u2a_repaired_gauge_representation_pieces.md) implements U2a. It proves the automatic repaired-gauge representation pieces from an absolutely continuous raw chart and an absolutely continuous repaired-gauge path: final physical chart identity, repaired-gauge equation, pressure reconstruction modulo constants, modulation coefficient formulas, AC gauge regularity, and critical \(L^3\)-norm invariance. After U2a, the live representation payloads are chart extraction, repaired-gauge root solvability, AC root selection, and terminal admissibility of the final scale-time chart.

[cluster_c3_l3_normalization_bridge.md](cluster_c3_l3_normalization_bridge.md) implements the critical \(L^3\)-normalization bridge. It replaces exact unit normalization by the invariant positive finite critical-mass certificate \(0<\eta\le\|V(\tau)\|_3\le M<\infty\), proves the corresponding good-window barrier variant, and classifies normalization defects.

[cluster_u3b_critical_mass_completion.md](cluster_u3b_critical_mass_completion.md) implements the local stratified U3 ledger. It proves positive finite critical mass for every retained active terminal stratum from the exhaustive state-space partition and terminal profile completeness, and proves finite active-packet bounds from profile mass decoupling. It is explicitly not a global \(L^3\) estimate for the full solution.

[cluster_c4_cost_bridge.md](cluster_c4_cost_bridge.md) implements the first classification-completeness bridge. It defines the Navier-Stokes Type II barrier cost, the cost-comparison witness, and the certificate \(K_{\mathrm{CostBridge}}^+\) that compiles Theorem A'' into the framework-level `BarrierTypeII` blocked certificate.

[cluster_c5_two_bucket_classification.md](cluster_c5_two_bucket_classification.md) implements C5. It combines C1--C4 with the compact Type II good-window barrier to prove that every finite-cost non-suppressed declared Type II candidate is either radiative/noncompact, meaning \(K_{L^3\mathrm{Tight}}^-\), or rough-core, meaning \(K_{\mathrm{WinH1}}^-\).

[cluster_c6_radiative_noncompact_exclusion.md](cluster_c6_radiative_noncompact_exclusion.md) implements C6. It defines \(K_{\mathrm{RadBlk}}^+\equiv K_{L^3\mathrm{Tight}}^+\) and proves several sufficient no-radiation routes: global \(L^3\)-compactness, compact core plus vanishing \(L^3\) remainder, no-splitting/no-radiation profile decomposition, finite library tight approximation, tame finite-description compact realization, and a-posteriori discharge of tightness INC certificates through the aggregate route certificate \(K_{\mathrm{C6Route}}^+\). Combined with C5, it leaves only the rough-core bucket.

[cluster_c6_windowed_h1_bridge.md](cluster_c6_windowed_h1_bridge.md) implements C6/T13. It proves the windowed local \(H^1\) bridge \(K_{\mathrm{WinH1}}^+\) from the renormalized Caccioppoli estimate, global \(L^\infty_\tau L^3_y\) control, bounded modulation, and pressure reconstruction. The pressure input uses the non-circular \(L^{3/2}\) pressure estimate from T1.

[cluster_c7_win_h1_certificate_composition.md](cluster_c7_win_h1_certificate_composition.md) implements the rough-core blocker \(K_{\mathrm{RoughCoreBlk}}^+\). It composes the upstream certificates needed by C6 and proves that \(K_{\mathrm{RepBridge}}^+\), global critical \(L^3\) boundedness, modulation-matrix invertibility, modulation forcing bounds, and Caccioppoli regularity emit \(K_{\mathrm{WinH1}}^+\), equivalently \(K_{\mathrm{RoughCoreBlk}}^+\), and it classifies the ordered defects when the rough-core bridge cannot be emitted.

[cluster_t14_modulation_force_bound.md](cluster_t14_modulation_force_bound.md) implements T14. It reduces \(K_{\mathrm{ModForceBd}}^+\) to compactly supported translation-force bounds and the weighted scale-forcing payload \(K_{\mathrm{ScaleForceBd}}^+\), with the translation-force estimate requiring independent pointwise local \(H^1\) control.

[cluster_c7_scale_force_bound.md](cluster_c7_scale_force_bound.md) refines the pointwise weighted scale-force payload. It proves the decomposition \(K_{\mathrm{ScaleLapBd}}^+\wedge K_{\mathrm{ScaleNonlinBd}}^+\wedge K_{\mathrm{ScalePressureBd}}^+\Rightarrow K_{\mathrm{ScaleForceBd}}^+\) and discharges the nonlinear scale row from the weighted fourth-moment certificate \(K_{\mathrm{ScaleL4Mom}}^+\).

[cluster_t15_t16_integrated_modulation.md](cluster_t15_t16_integrated_modulation.md) implements T15--T17. It replaces pointwise forcing by window-integrated forcing, proves good-time modulation selection, proves the integrated translation-force estimate from windowed gradient control, and isolates the integrated weighted scale-force payload.

[cluster_t18_scale_force_decomposition.md](cluster_t18_scale_force_decomposition.md) implements T18. It decomposes the integrated scale-force payload into diffusive, convective, and pressure contributions, discharges singular-weight convective integration by parts by a cutoff argument from annular convective regularity and weighted \(L^4\)-window control, and isolates the remaining weighted diffusion and pressure defects.

[cluster_s2_rough_core_vorticity_exclusion.md](cluster_s2_rough_core_vorticity_exclusion.md) implements S2. It records that the 3D vorticity/enstrophy route does not autonomously close rough-core because of vortex stretching, and proves the conditional subtype result that represented bounded-critical-norm branches with local vorticity-window control have windowed local \(H^1\) control.

[cluster_s4_multibubble_residue_classification.md](cluster_s4_multibubble_residue_classification.md) implements S4. It states the current NS3D endpoint: once the technical bridge payloads, exterior-regular far-radiation discard, rough-core Caccioppoli closure, and either the C17 scale-collapse cost bridge or the S3 compactness/modulation/stationary-limit/Liouville route are supplied, every remaining Type II survivor is multibubble: same-point no-splitting/profile-compatibility residue or multi-point concentration.

[cluster_s5_multiscale_cascade_exclusion.md](cluster_s5_multiscale_cascade_exclusion.md) implements S5. It proves \(L^3\) profile decoupling for orthogonal finite profile families, derives the critical-mass bound on cascade length, expands regular outer bubbles in innermost variables, proves escaping-offset outer profiles vanish locally, absorbs the leading ambient velocity into the translation gauge, discharges perturbative S3 robustness under explicit local estimates, and rules out regular strict same-point cascades. It also proves that \(K_{\mathrm{SamePointNLDec}}^+\) rules out all same-point multibubble cascades. The remaining same-point case is exactly the no-splitting/profile-compatibility defect \(K_{\mathrm{SamePointNLDec}}^-\).

[cluster_s6_multibubble_camera_reduction.md](cluster_s6_multibubble_camera_reduction.md) implements S6. It formalizes the camera-on-active-bubble reduction for multibubbles: comparable same-point profiles are compound profiles, strict same-point cascades route through S5, separated physical points are locally invisible in one another's cameras, and the full multibubble candidate is ruled out under \(K_{\mathrm{SamePointNLDec}}^+\), \(K_{\mathrm{MultiPointCamDec}}^+\), and \(K_{\mathrm{S3NRSPayload}}^+\). S7 and S8 discharge the two decoupling payloads in terminal cameras; remaining failures are upstream profile-completeness, exterior-discard, representation, U5a/suitability, or S3 defects.

[cluster_s7_decoupling_payload_discharge.md](cluster_s7_decoupling_payload_discharge.md) implements S7. It proves local perturbation stability in \(L^1_\tau H^{-1}_y\), static same-point and separated-point velocity decoupling from \(L^3\), local pressure decoupling from Calderon-Zygmund kernel tails, and then proves
\[
K_{\mathrm{NLProfDec},NS3D}^+
\Longrightarrow
K_{\mathrm{SamePointNLDec}}^+
\wedge
K_{\mathrm{MultiPointCamDec}}^+.
\]
Thus the multibubble residue is reduced to the single nonlinear profile-decoupling theorem \(K_{\mathrm{NLProfDec},NS3D}^+\).

[cluster_s8_terminal_nonlinear_profile_decoupling.md](cluster_s8_terminal_nonlinear_profile_decoupling.md) implements S8. It proves \(K_{\mathrm{NLProfDec},NS3D}^+\) in the terminal active camera sense from the terminal windowed nonlinear profile-completeness payload \(K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+\), removal of the scattering branch, exterior-regular discard, repaired-gauge representation, and Caccioppoli regularity. S12 now supplies the profile-completeness payload from bounded terminal critical sequences, Kato small-data stability, and the accepted critical NS profile theorem. S8 also records why the terminal quantifier is necessary: a nonterminal same-point camera can still see a smaller active bubble.

[cluster_s9_scattering_branch_discharge.md](cluster_s9_scattering_branch_discharge.md) implements S9. It proves the small-data critical \(L^3\) scattering branch discharge: profiles with \(\|\phi\|_3\le\varepsilon_{\mathrm{sd}}\) generate global perturbative Kato mild solutions and are assigned to the scattering ledger, so every active profile has \(\|\phi\|_3>\varepsilon_{\mathrm{sd}}\).

[cluster_s10_exterior_regular_discard.md](cluster_s10_exterior_regular_discard.md) implements S10. It proves \(K_{\mathrm{ExtRegDiscard}}^+\) from single-point exterior regularity by constructing a divergence-free core/exterior decomposition and showing that exterior velocity, pressure, and localization errors vanish in compact Type II core cameras. It also proves that positive-physical-distance profiles belong to the exterior ledger and cannot contribute to compact terminal Type II core cameras.

[cluster_u5a_bare_data_caccioppoli.md](cluster_u5a_bare_data_caccioppoli.md) is the current Caccioppoli regularity discharge. For represented suitable branches, C2.R supplies pressure reconstruction and AC gauge/modulation, so U5a emits the local certificate \(K_{\mathrm{CaccioppoliReg}}^+\). The proof is compact-cylinder only: it does not assume global \(L^3\), global tightness, bounded modulation, finite energy at infinity, global compactness, or uniform windowed \(H^1\). [cluster_s11_caccioppoli_regularity_discharge.md](cluster_s11_caccioppoli_regularity_discharge.md) contains the underlying renormalized local energy identity and Caccioppoli estimate used by that route.

[cluster_c7_l3bd_defect_discharge.md](cluster_c7_l3bd_defect_discharge.md) discharges the bounded-critical-norm input in C7 by proving \(K_{\mathrm{StratCritPacket}}^+\Rightarrow K_{L^3\mathrm{Bd}}^+\), so \(K_{L^3\mathrm{Bd}}^-\) is not an independent remaining rough-core defect on the C-series route.

[cluster_c7_modmatrix_inverse_discharge.md](cluster_c7_modmatrix_inverse_discharge.md) discharges the C7 modulation-matrix inverse input by packaging the repaired-gauge nondegeneracy payload from T2--T4 into \(K_{\mathrm{ModMatrixInv}}^+\).

[cluster_c8_typeII_branch_exclusion.md](cluster_c8_typeII_branch_exclusion.md) implements C8. It combines universal \(K_{\mathrm{ClassComplete}}^+\), \(K_{\mathrm{RadBlk}}^+\), \(K_{\mathrm{RoughCoreBlk}}^+\), and the NS-valid promotion certificate \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) to prove that every declared Type II candidate emits the post-promotion suppression certificate, so no admissible unresolved Type II branch remains in the declared backend.

[cluster_c9_energy_cost_bridge.md](cluster_c9_energy_cost_bridge.md) implements C9. It proves that finite physical energy controls the scale-weighted renormalized dissipation, not the unweighted Type II barrier cost; introduces \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) as the applicability bridge needed before applying Type II promotion to NS3D; and classifies infinite unweighted cost into blocked, physically forbidden weighted-infinite-cost, or scale-decoupled infinite-cost scenarios.

[cluster_c10_ns_up_typeII_promotion.md](cluster_c10_ns_up_typeII_promotion.md) implements C10. It makes \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) explicit as a uniform NS3D promotion payload \(K_{\mathrm{NSPromPayload}}^+\), separates blocked Type II from post-promotion suppression, and proves that \(K_{\mathrm{SC}_\lambda}^-\wedge K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) emits \(K_{\mathrm{SC}_\lambda}^{\sim}\) only when the promotion-soundness route is supplied. Its generic route defines \(K_{\mathrm{GenericUPTypeIIAdmiss}}^+\), the exact payload that licenses applying the formal theorem {prf:ref}`mt-up-type-ii` to an NS3D repaired-gauge branch.

[cluster_c11_generic_up_typeII_payload_discharge.md](cluster_c11_generic_up_typeII_payload_discharge.md) implements C11. It discharges the generic `UP-TypeII` admissibility payload as far as the current NS3D stack permits: C1, C2, C4, C10, and the dataset certificates supply the anchor, scale route, bounded-energy Type II concentration, cost translation, parabolic domain embedding, and conclusion identity. The remaining analytic witness is \(K_{\mathrm{NSLocMonoTrans}}^+\), the localized NS3D monotonicity translation needed before the formal theorem {prf:ref}`mt-up-type-ii` can be applied.

[cluster_c12_ns_localized_monotonicity_translation.md](cluster_c12_ns_localized_monotonicity_translation.md) implements C12. It proves \(K_{\mathrm{NSLocMonoTrans}}^+\) from the repaired-gauge localized energy identity once the explicit finite monotonicity-error certificate \(K_{\mathrm{FiniteMonoErr}}^+\) is supplied. The error density contains the viscous cutoff, convective flux, pressure flux, center drift, negative-scale, and scale-cutoff terms; C12 shows these are exactly the terms that must have finite tail integral to use the formal `UP-TypeII` theorem for NS3D.

[cluster_c13_formal_up_typeII_application.md](cluster_c13_formal_up_typeII_application.md) implements C13. It composes C10--C12 into the exact formal-application theorem: \(K_{\mathrm{FormalUPTypeIIApp},NS3D}^+\) implies that {prf:ref}`mt-up-type-ii` applies to the declared NS3D Type II candidate and emits \(K_{\mathrm{SC}_\lambda}^{\sim}\). It also records that the blocked certificate \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) must actually be emitted and \(K_{\mathrm{NSLocEnergyId}}^+\) must be supplied; after those are present, the remaining analytic obstruction on the current C-series route is \(K_{\mathrm{FiniteMonoErr}}^+\).

[cluster_c14_explicit_up_typeII_ns3d_checklist.md](cluster_c14_explicit_up_typeII_ns3d_checklist.md) implements C14. It gives the complete direct and expanded checklists for applying formal `UP-TypeII` to NS3D, including \(K_{\mathrm{NSLocEnergyId}}^+\), \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\), and the six finite-tail subcertificates whose conjunction is \(K_{\mathrm{FiniteMonoErr}}^+\).

[cluster_c15_moving_cutoff_monotonicity.md](cluster_c15_moving_cutoff_monotonicity.md) implements C15. It replaces the fixed-annulus finite-error demand by a moving-cutoff package \(K_{\mathrm{MoveAnnErr}}^+\), adding the new \(\partial_\tau\phi_{R(\tau)}\) error and isolating the scale-negative tail \(K_{\mathrm{ScaleNegL^1}}^+\) as the remaining hard drift term. It also adds \(K_{\mathrm{MovingCostBridge}}^+\), because the moving cost must be accepted by, or compared to, the declared `BarrierTypeII` cost before it can replace the fixed C12 input.

[cluster_c16_scale_negative_drift_dichotomy.md](cluster_c16_scale_negative_drift_dichotomy.md) implements C16. It proves the exact dichotomy for the C15 scale-negative term: bounded moving localized \(L^2\) mass plus finite negative repaired-gauge drift emits \(K_{\mathrm{ScaleNegL^1}}^+\), while failure of the weighted integral is the named obstruction \(K_{\mathrm{ScaleCollapseDrift}}^-\). It also records the sign consequence of the master-note convention \(a=d_\tau\log\lambda\): genuine collapse \(\lambda(\tau)\to0\) with a nonvanishing localized \(L^2\) core forces the obstruction rather than discharging it.

[cluster_c17_scale_collapse_barrier.md](cluster_c17_scale_collapse_barrier.md) implements C17. It turns the C16 obstruction into a direct Type II barrier branch once the backend registers either the scale-collapse cost \(a_-M_R\) or the absolute scale-drift cost \(\nu\int|\nabla V|^2\phi_{R(\tau)}+|a|M_R\). Under that cost bridge, \(K_{\mathrm{ScaleCollapseDrift}}^-\) emits \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\), and with C10's NS-valid promotion payload it emits \(K_{\mathrm{SC}_\lambda}^{\sim}\).

[cluster_c18_final_up_typeII_ns3d.md](cluster_c18_final_up_typeII_ns3d.md) implements C18. It is the final assembly theorem for the declared terminal NS3D repaired-gauge Type II backend: C8 gives the abstract implication from classification completeness, radiative blocker, rough-core blocker, and NS-valid promotion to universal \(K_{\mathrm{SC}_\lambda}^{\sim}\); C18 then expands the terminal payload using S3 and S4--S8 for scale-collapse and multibubble/radiative closure, C7 for rough-core closure, and C10 for promotion.

[cluster_s12_terminal_profile_completeness.md](cluster_s12_terminal_profile_completeness.md) implements S12. It replaces the global `CatLib` placeholder for the Type-II terminal profile slot by the concrete analytic theorem actually needed: bounded terminal critical sequences, Kato small-data \(L^3\) stability, and the accepted critical Navier-Stokes profile decomposition/nonlinear stability theorem emit \(K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+\).

[cluster_s13_bounded_critical_terminal_sequences.md](cluster_s13_bounded_critical_terminal_sequences.md) implements S13. It proves the bounded terminal critical-sequence input used by S12 from \(K_{\mathrm{StratCritPacket}}^+\) and terminal sequence routing, using scale invariance of the \(L^3\) norm. With S14 imported, the C18 terminal backend uses \(K_{\mathrm{StratCritPacket}}^+\wedge K_{\mathrm{RepBridge}}^+\) for this boundedness step. Failure is classified by the ordered S13 outcomes \(K_{\mathrm{TermSeqRoute}}^-\), \(K_{L^3\mathrm{Dom}}^-\), \(K_{L^3\mathrm{Inf}}^-\), or \(K_{L^3\mathrm{Zero}}^-\), with \(K_{\mathrm{TermSeqRoute}}^-\) reduced to \(K_{\mathrm{RepBridge}}^-\) inside the declared terminal backend.

[cluster_s14_terminal_sequence_routing.md](cluster_s14_terminal_sequence_routing.md) implements S14. It proves \(K_{\mathrm{TermSeqFromOrbit}}^+\) from repaired-gauge representation and terminal-camera construction, and classifies failure as \(K_{\mathrm{RepBridge}}^-\) or \(K_{\mathrm{TermCamConstruct}}^-\).

[cluster_s3_scale_collapse_attractor_stratification.md](cluster_s3_scale_collapse_attractor_stratification.md) implements S3. It rewrites the scale-collapse drift survivor as a generalized self-similar reduction problem: thick/autonomous drift plus compactness gives a nonzero autonomous reduced NS limit, a stationary omega-limit payload turns that into a profile equation, and parameter-specific Liouville or self-similar rigidity is required before the branch is blocked. Thin drift, failed autonomous modulation, failed stationary extraction, and uncovered Liouville ranges remain explicit S3 defects.

## Dependency chain

The analytic flow is:

1. Renormalize the physical solution into \((V,P,a,b)\).
2. Fix the orbit by the repaired scale gauge and centering gauges.
3. Use gauge identities to build the modulation matrix \(M(V)\).
4. Prove matrix nondegeneracy from the scale transversality, translation block, and cross-term control.
5. Use nondegeneracy to bound \((a,b)\).
6. Use the \(L^2\) pressure bridge T1.5 and bounded \((a,b)\) to prove local \(H^{-1}\) time regularity.
7. Use Caccioppoli for local \(L^2_\tau H^1_y\) control.
8. Use Aubin-Lions and the T7 upgrade to get spacetime compactness, then use either sampled-time \(L^2\)-compactness or the T7.4 good-time selection lemma plus local \(H^1\) control to get strong local \(L^3\)-compactness of sampled states.
9. Combine low-cost subsequence extraction, low core dissipation, tightness, normalization, and compactness.
10. Apply the low-dissipation rigidity lemma to contradict finite total cost.

The repaired gauge is essential in steps 2--5. The pressure layer has three distinct roles: T1 gives local \(L^{3/2}\) pressure control from the standard pressure tail, T1.4 verifies the stronger T1.5 hypotheses from \(L^\infty_\tau L^3_y\) and local \(L^2_\tau H^1_y\), and T1.5 gives the \(L^2_\tau L^2_y\) pressure control needed for the \(H^{-1}\) time-regularity theorem. The \(L^{3/2}\) pressure bound alone does not give \(\nabla P\in H^{-1}\), because the \(H^{-1}\) pairing differentiates the test function. The compactness and pressure/time-regularity estimates are essential in steps 6--8. The final contradiction happens only after all these inputs are present simultaneously.

The preferred streamlined closure is Theorem A''. It bypasses endpoint trace compactness and \(H^{-1}\)-time regularity in the final contradiction step by selecting good times from low-cost windows with local \(L^2_\tau H^1_y\) control.

## Current conditional hypotheses

The final theorem uses the following core assumptions on the renormalized orbit:

1. Global \(L^3\)-normalization:
\[
\|V(\tau)\|_{L^3(\mathbb R^3)}=1.
\]
2. Uniform global \(L^3\)-tightness:
\[
\forall\varepsilon>0\ \exists R_\varepsilon:
\sup_\tau\int_{|y|>R_\varepsilon}|V(y,\tau)|^3<\varepsilon.
\]
3. Local \(L^2_\tau H^1_y\) control, typically supplied by the renormalized Caccioppoli estimate.
4. Repaired gauge admissibility and modulation nondegeneracy sufficient to bound \((a,b)\).
5. Local \(L^2_\tau H^{-1}_y\) time regularity for \(\partial_\tau V\), when using the Aubin-Lions route. T1, T1.4, and T1.5 supply the required pressure input from \(L^\infty_\tau L^3_y\) and local \(L^2_\tau H^1_y\) control.
6. A sampled-time strong \(L^2_{\mathrm{loc}}\)-compactness mechanism plus a local \(L^6\)-type bound, or the T7.4 good-time mechanism from low average cost and local \(L^2_\tau H^1_y\) control on common windows. Equivalently, any direct mechanism giving strong sampled-time \(L^3_{\mathrm{loc}}\)-compactness suffices.

Theorem A'' gives a sufficient package for the compactness contradiction:

1. global \(L^3\)-normalization,
2. uniform \(L^3\)-tightness,
3. uniform local \(L^2_\tau H^1_y\) bounds on unit windows.

Under those three assumptions, finite total cost itself produces the compact low-cost samples needed for Lemma 8.

The remaining closure points are certificate-level defects:

- \(K_{L^3\mathrm{Tight}}^-\): failure of global critical tightness; C6 gives sufficient routes through global \(L^3\)-compactness, no-radiation, no-splitting, finite-library approximation, or tame finite-description compact realization.
- \(K_{\mathrm{WinH1}}^-\): failure of local windowed \(H^1\) control; C6/T13 and C7 reduce this to ordered upstream defects. On the C-series route, \(K_{L^3\mathrm{Bd}}^-\) is discharged by C3 and \(K_{\mathrm{ModMatrixInv}}^-\) is discharged by the repaired-gauge nondegeneracy payload. T14 refines \(K_{\mathrm{ModForceBd}}^-\) into translation-force or weighted scale-force failure only when its independent local translation-force hypotheses are available; the pointwise scale-force side is split by [cluster_c7_scale_force_bound.md](cluster_c7_scale_force_bound.md) into \(K_{\mathrm{ScaleLapBd}}^-\), \(K_{\mathrm{ScaleL4Mom}}^-\), or \(K_{\mathrm{ScalePressureBd}}^-\), with the nonlinear row discharged by \(K_{\mathrm{ScaleL4Mom}}^+\). T15--T17 give a weaker non-circular good-window replacement using \(K_{\mathrm{TransForceL^1Win}}^+\) and \(K_{\mathrm{ScaleForceL^1Win}}^+\). T18 decomposes \(K_{\mathrm{ScaleForceL^1Win}}^-\) into weighted diffusion, annular convective regularity, weighted \(L^4\)-convective, and weighted pressure defects; the singular boundary terms are discharged by the T18 cutoff argument. U5a discharges \(K_{\mathrm{CaccioppoliReg}}^+\) from physical suitability and the declared repaired-gauge representation; C2.R supplies pressure reconstruction and AC gauge/modulation. After importing representation, gauge nondegeneracy, and U5a, the remaining rough-core defects are \(K_{\mathrm{ModForceBd}}^-\), bounded-modulation failure, or failure of physical suitability, with the integrated route available for selected-time compactness arguments.
- \(K_{\mathrm{CostCompare}}^-\): only relevant if the backend uses a Type II barrier cost different from \(\tilde{\mathfrak D}_{R_0}\).
- C1 backend defects \(K_{\mathrm{CmuExtract}}^-\), \(K_{\mathrm{ScaleRoute}}^-\), \(K_{\mathrm{ProfComplete}}^-\), and \(K_{\mathrm{LostTypeII}}^-\), which concern whether a declared Type II candidate enters the certified profile route.

## What the proof excludes

The barrier excludes a compact Type II renormalized orbit satisfying the hypotheses above from having finite total renormalization cost. Equivalently, a normalized tight orbit with the required local compactness, local \(H^1\) control, and modulation properties cannot admit a low-cost subsequence whose core dissipation vanishes.

This is strongest against a single compact core scenario: one profile, no radiation, no splitting, with enough gauge control to keep the renormalized dynamics uniformly estimated.

## Current Type II singularity classification

The current classification is a DAG-local certificate-level classification,
not a claim about arbitrary Navier-Stokes solutions outside this route. It
applies to a candidate declared by the Navier-Stokes backend as Type II, then
routed through C1 into the sieve Type II branch \(K_{\mathrm{SC}_\lambda}^-\),
and finally compiled through the repaired-gauge compact barrier stack.

The current NS3D endpoint is the S4--S8 reduction: after the technical bridge
payloads are supplied, exterior-regular far radiation is discarded from the
core ledger by \(K_{\mathrm{ExtRegDiscard}}^+\), rough core is closed by
C6/T13, and scale-collapse drift is closed either by the C17 cost bridge or by
the S3 compactness/modulation/stationary-limit/Liouville payloads. S5 closes
regular strict same-point cascades under its explicit cascade payloads, and
closes all same-point cascades under \(K_{\mathrm{SamePointNLDec}}^+\). S6
closes separated-point multibubbles under \(K_{\mathrm{MultiPointCamDec}}^+\).
S7 proves both decoupling payloads from \(K_{\mathrm{NLProfDec},NS3D}^+\), and
S8 proves \(K_{\mathrm{NLProfDec},NS3D}^+\) in terminal active cameras. Thus
multibubble concentration is not a separate survivor class inside the
terminal-complete profile backend. A multibubble failure can occur only
through one of the following upstream defects:

1. profile-completeness or terminal extraction failure
   \(K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^-\);
2. repaired-gauge representation or Caccioppoli regularity failure;
3. failure of the S3 rigidity payload.

These are technical or rigidity payload defects, not additional multibubble
singularity mechanisms.

C1 is implemented as a declared-backend exhaustion theorem: if the backend supplies \(K_{\mathrm{ExhPayload}}^+\), then it emits \(K_{\mathrm{TypeIIExhaust}}^+\). If that payload cannot be supplied, the first possible singularity classification is an exhaustion/backend routing defect. Once C1 routes the candidate successfully, every attempted compact-barrier compilation lands in one of the downstream outcomes below.

1. **Exhaustion/backend routing defect.** The candidate is declared Type II, but the backend does not certify that it enters the concentration/scale/profile route consumed by C2--C4. The ordered C1 defects are:
\[
K_{\mathrm{CmuExtract}}^-,
\quad
K_{\mathrm{ScaleRoute}}^-,
\quad
K_{\mathrm{ProfComplete}}^-,
\quad
K_{\mathrm{LostTypeII}}^-.
\]
These respectively mean failure to emit the concentration profile certificate \(K_{C_\mu}^+\), failure to route the branch through \(K_{\mathrm{SC}_\lambda}^-\), failure to emit \(K_{\mathrm{Prof}_{NS}}^+\), or leakage of a declared Type II branch outside the certified route.

2. **Suppressed compact Type II.** The candidate has all bridge certificates needed by the compact barrier:
\[
K_{\mathrm{RepBridge}}^+,\quad
K_{\mathrm{StratCritPacket}}^+,\quad
K_{L^3\mathrm{Tight}}^+,\quad
K_{\mathrm{WinH1}}^+,\quad
K_{\mathrm{CostBridge}}^+.
\]
Theorem A'' then gives infinite localized PDE renormalization cost, and \(K_{\mathrm{CostBridge}}^+\) compiles this into the blocked `BarrierTypeII` certificate \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\). The stronger post-promotion certificate \(K_{\mathrm{SC}_\lambda}^{\sim}\) is emitted only after the NS-valid applicability/replacement bridge \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) of C10 is supplied. This is the compact single-core class currently blocked by the implemented PDE theorem and suppressed only in the C10-valid promoted backend.

3. **Representation bridge.** Inside the declared NS3D Type II barrier backend, C2.R makes this nonconditional: every routed candidate emits \(K_{\mathrm{RepBridge}}^+\). The outside-contract diagnostics are
\[
K_{\mathrm{Chart}}^-,
\quad
K_{\mathrm{GaugeSolve}}^-.
\]
These respectively mean failure to extract an absolutely continuous raw concentration chart from the routed profile package, or failure to solve the repaired scale/centering gauge admissibly and absolutely continuously. They are not live survivor classes once the declared backend contract is in force. Pressure representation and final modulation parameters are theorems from the chart plus AC gauge solve.
U2a refines this further: after an AC chart and AC repaired-gauge path are
present, pressure reconstruction, modulation coefficients, final chart
identity, AC gauge regularity, and critical \(L^3\)-invariance are automatic.
The downstream representation strata are chart extraction, repaired-gauge
root solvability, AC root selection, and terminal admissibility of the final
scale-time chart.

4. **Critical \(L^3\)-normalization defect.** The represented orbit is not on the positive finite critical-mass branch used by the compact barrier. C3 replaces exact unit normalization by the invariant annulus condition
\[
0<\eta\le\|V(\tau)\|_{L^3(\mathbb R^3)}\le M<\infty
\]
after discarding at most a finite initial \(\tau\)-interval. If this fails, the ordered defects are:
\[
K_{L^3\mathrm{Dom}}^-,
\quad
K_{L^3\mathrm{Inf}}^-,
\quad
K_{L^3\mathrm{Zero}}^-.
\]
These respectively mean the critical norm is not a well-defined measurable branch quantity, the critical mass is infinite or unbounded, or the critical mass has bounded zero-collapse. These are now named defects, not an ambiguous "normalization failure."

5. **Noncompact/radiative Type II.** The representation and normalization bridges are present, but uniform \(L^3\)-tightness fails:
\[
\exists\varepsilon_0>0\ \forall R>0\ \exists\tau_R:
\int_{|y|>R}|V(y,\tau_R)|^3\,dy\ge\varepsilon_0.
\]
This bucket includes radiation tails, outward-moving secondary profiles, multi-profile splitting, cascade behavior, and failures of global critical precompactness. S4 separates physically far radiation from genuine multibubble radiation. Far radiation is regular exterior mass under the single-point blowup payload and is removed from the Type II core ledger only when \(K_{\mathrm{ExtRegDiscard}}^+\) is supplied. S5 removes regular strict same-point cascades and proves that \(K_{\mathrm{SamePointNLDec}}^+\) removes all same-point cascades. S6 reduces the multibubble part to camera-decoupling payloads, S7 reduces those payloads to \(K_{\mathrm{NLProfDec},NS3D}^+\), and S8 proves that payload in terminal cameras from \(K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+\) and the repaired-gauge/Caccioppoli bridge. The remaining radiative defects are upstream profile-completeness, exterior-discard, representation, U5a/suitability, or S3-rigidity defects.

6. **Rough-core Type II.** The representation, normalization, and tightness bridges are present, but local windowed \(H^1\) control fails:
\[
\exists m\ge1:
\sup_n\int_{\tau_0+n}^{\tau_0+n+1}
\|V(\tau)\|_{H^1(B_m)}^2\,d\tau=\infty.
\]
This bucket includes compact-looking \(L^3\)-tight cores with persistent local high-frequency activity, oscillation, or gradient concentration in bounded renormalized regions. C6/T13 proves \(K_{\mathrm{WinH1}}^+\) from bounded global \(L^3\), pressure reconstruction, bounded modulation, and Caccioppoli regularity. C7 composes this with upstream certificates and identifies \(K_{\mathrm{WinH1}}^+\) with \(K_{\mathrm{RoughCoreBlk}}^+\) on represented repaired-gauge branches. Failure of the rough-core blocker is reduced to ordered upstream defects. On the C-series route, [cluster_c7_l3bd_defect_discharge.md](cluster_c7_l3bd_defect_discharge.md) discharges the bounded-critical-norm defect and [cluster_c7_modmatrix_inverse_discharge.md](cluster_c7_modmatrix_inverse_discharge.md) discharges the modulation-matrix inverse defect. U5a discharges Caccioppoli regularity from physical suitability and the declared repaired-gauge representation, with pressure and AC gauge/modulation supplied by C2.R. After importing C2/C5 representation, repaired-gauge nondegeneracy, and U5a, the remaining nontrivial rough-core defects are
\[
K_{\mathrm{ModForceBd}}^-,
\]
or failure of physical suitability.
The pointwise modulation-force defect is refined by T14 and
[cluster_c7_scale_force_bound.md](cluster_c7_scale_force_bound.md). The
good-window replacement is refined by T15--T18 into
\[
K_{\mathrm{TransForceL^1Win}}^-,
\quad
K_{\mathrm{ScaleDiffL^1Win}}^-,
\quad
K_{\mathrm{AnnConvReg}}^-,
\quad
K_{\mathrm{ScaleV4L^1Win}}^-,
\quad
K_{\mathrm{ScalePressL^1Win}}^-.
\]
The singular convective boundary terms are not a separate survivor: T18
discharges them by a cutoff argument from \(K_{\mathrm{AnnConvReg}}^+\) and
\(K_{\mathrm{ScaleV4L^1Win}}^+\).

S2 excludes the vorticity-controlled rough-core subtype:
\[
K_{\mathrm{RepBridge}}^+
\wedge
K_{L^3\mathrm{Bd}}^+
\wedge
K_{\mathrm{VortL^2Win}}^+
\Longrightarrow
K_{\mathrm{RoughCoreBlk}}^+.
\]
Thus any represented bounded-critical-norm rough-core survivor must now carry
\(K_{\mathrm{VortL^2Win}}^-\) or one of the modulation/Caccioppoli defects
listed above.
This is not the primary rough-core closure. The primary closure remains the
C6/T13 Caccioppoli bridge, which proves \(K_{\mathrm{WinH1}}^+\) directly from
bounded critical \(L^3\), pressure reconstruction, bounded modulation, and
Caccioppoli regularity.

7. **Cost-adapter defect.** The PDE hypotheses of the compact theorem are present and Theorem A'' gives infinite localized PDE cost, but that divergence has not been compiled into the framework-level `BarrierTypeII` certificate. In the default Navier-Stokes Type II backend this is implemented by the identity cost
\[
\mathfrak C_{\mathrm{II}}^{NS}=\tilde{\mathfrak D}_{R_0},
\]
so \(K_{\mathrm{CostBridge}}^+\) is automatic once measurability, nonnegativity, local finite-interval integrability, and the backend registration are accepted. If a different generic framework cost is used, the remaining datum is the comparison witness \(K_{\mathrm{CostCompare}}^+\).

8. **Promotion/monotonicity defect.** The fixed-cutoff formal `UP-TypeII`
application still requires the finite monotonicity-error package
\(K_{\mathrm{FiniteMonoErr}}^+\). The moving-cutoff replacement reduces the
annular part to \(K_{\mathrm{MoveAnnErr}}^+\), but C16 shows that the
scale-negative part has an exhaustive alternative:
\[
K_{\mathrm{ScaleNegL^1}}^+
\quad\text{or}\quad
K_{\mathrm{ScaleCollapseDrift}}^-.
\]
The positive alternative follows from bounded moving localized \(L^2\) mass
and finite negative repaired-gauge drift. Genuine scale collapse with a
nonvanishing localized \(L^2\) core forces the obstruction, so it must be
handled either by the C17 scale-collapse cost bridge, by the S3
generalized self-similar reduction route, or retained as a residual Type II
class. If \(K_{\mathrm{ScaleCollapseCostBridge}}^+\) or
\(K_{\mathrm{AbsScaleCostBridge}}^+\) is supplied, this obstruction emits
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) directly. S3 instead reduces the
obstruction to autonomous compactness, stationary-limit, and parameter-rigidity
payloads; only when those payloads are supplied does it emit
\(K_{\mathrm{ScaleCollapseBlk}}^+\).

9. **Multibubble ledger.** After the technical bridge payloads, all closeable
non-bubble strata, and the S5 regular cascade discharge are supplied, the S4/S6
multibubble ledger is:
\[
\lambda_1(t)\ll\lambda_2(t)\to0
\]
at one physical blowup point only through failure of terminal profile
completeness or terminal nonlinear decoupling, or at two or more physical
concentration centers only through failure of terminal camera decoupling. S6
proves that if both positive decoupling payloads are supplied, the multibubble
residue is ruled out by single-camera S3 reductions. S7 proves both payloads
from \(K_{\mathrm{NLProfDec},NS3D}^+\), and S8 proves
\(K_{\mathrm{NLProfDec},NS3D}^+\) in terminal cameras. Thus the remaining
multibubble problem is an upstream terminal profile-completeness,
exterior-discard, representation, Caccioppoli, or S3-rigidity defect, not a
separate singularity class.

The finite-cost PDE contrapositive is sharper after the bridge defects are
discharged. If \(K_{\mathrm{RepBridge}}^+\),
\(K_{\mathrm{StratCritPacket}}^+\), and \(K_{\mathrm{CostBridge}}^+\) are
available at this node, then every finite-cost non-suppressed candidate is
either radiative/noncompact or rough-core. Equivalently, the good-window
closure says that a represented positive-finite-critical-mass branch with
uniform \(L^3\)-tightness and uniform local \(L^2_\tau H^1_y\) bounds on unit
windows cannot have finite total localized renormalization cost.

The fully PDE-intrinsic Type II classification still depends on the analytic content of the C1 payload: concentration extraction, scale routing, profile completeness, and no lost Type II branch for the declared backend. C1 packages those requirements as \(K_{\mathrm{ExhPayload}}^+\Rightarrow K_{\mathrm{TypeIIExhaust}}^+\), but the payload itself remains the backend completeness obligation. With that payload in place, the list above is exhaustive for declared Type II candidates in the backend; without it, the candidate is classified by one of the C1 exhaustion defects.

The following are not separate survivor classes:

- Pressure-tail failure: T1 proves pressure-tail control from global \(L^3\).
- Failure of the T1.5 pressure hypotheses under \(L^\infty_\tau L^3_y\) and local \(L^2_\tau H^1_y\) control: T1.4 verifies those hypotheses.
- Failure of local \(L^2_\tau L^2_y\) pressure control modulo constants under the same upstream assumptions: T1.5 proves this pressure estimate.
- Failure of \(H^{-1}\)-time regularity as an isolated obstruction: it is relevant only in the Aubin-Lions route and depends on the upstream local \(H^1\), pressure, and modulation estimates.
- Gauge degeneracy as an unnamed class: inside the declared backend C2.R discharges the repaired-gauge solve. Outside that backend it is recorded explicitly as \(K_{\mathrm{GaugeSolve}}^-\), not as an unnamed Type II mechanism.
- Modulation-matrix inverse failure after the repaired-gauge nondegeneracy
  payload is imported: C7 discharges it through
  \(K_{\mathrm{ModMatrixPayload}}^+\Rightarrow K_{\mathrm{ModMatrixInv}}^+\).
- Singular-weight convective integration-by-parts failure as an independent
  scale-force class: T18 discharges it from annular convective regularity and
  weighted \(L^4\)-window control.
- Vorticity-controlled rough core: S2 proves that local vorticity-window
  control implies \(K_{\mathrm{WinH1}}^+\). S2 also records that the 3D
  enstrophy equation does not self-close because of vortex stretching, so the
  remaining rough-core closure is C6/T13 plus the C7 modulation/Caccioppoli
  payloads.
- Non-single-profile behavior as an unnamed final class: S4 names the genuine
  non-single-profile residue as multibubble concentration. S5 removes regular
  strict same-point cascades, S6 reduces multibubbles to camera payloads, S7
  reduces those payloads to \(K_{\mathrm{NLProfDec},NS3D}^+\), and S8 proves
  \(K_{\mathrm{NLProfDec},NS3D}^+\) in terminal active cameras from profile
  completeness, exterior-regular discard, and the repaired-gauge/Caccioppoli
  bridge. Remaining
  non-single-profile failures are upstream profile, representation,
  Caccioppoli, or S3 defects.

These are the targets of C1--C18, T8--T18, and S2--S14 in [type2_roadmap.md](type2_roadmap.md). The compact single-core barrier is assembled. The current endpoint is the C18 final Type-II-specific `UP-TypeII` theorem for the declared terminal NS3D backend, with S12 replacing the global `CatLib` placeholder by the standard critical profile-decomposition theorem, U3b supplying local stratified critical-mass bookkeeping, S13 discharging bounded terminal sequence input from the stratified packet route, and S14 discharging terminal sequence routing.

## Practical reading order

For the fastest orientation, read:

1. This README.
2. [type2_roadmap.md](type2_roadmap.md) for theorem status and open structural tasks.
3. [hypostructure_sieve_typeII_certificates.md](hypostructure_sieve_typeII_certificates.md) for the sieve-certificate interface and bridge ledger.
4. [compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md) for the full proof chain.
5. [required_new_scale_gauge_theorems.md](required_new_scale_gauge_theorems.md) for the repaired gauge.
6. [cluster_next_easiest_theorems.md](cluster_next_easiest_theorems.md), [cluster_t7_L3_compactness_upgrade.md](cluster_t7_L3_compactness_upgrade.md), and [cluster_structural_t8_t11.md](cluster_structural_t8_t11.md) for theorem-cluster closures.
7. [cluster_t15_t16_integrated_modulation.md](cluster_t15_t16_integrated_modulation.md), [cluster_t18_scale_force_decomposition.md](cluster_t18_scale_force_decomposition.md), and [cluster_s2_rough_core_vorticity_exclusion.md](cluster_s2_rough_core_vorticity_exclusion.md) for the integrated modulation-force and vorticity-controlled rough-core routes.
8. [cluster_s4_multibubble_residue_classification.md](cluster_s4_multibubble_residue_classification.md), [cluster_s5_multiscale_cascade_exclusion.md](cluster_s5_multiscale_cascade_exclusion.md), [cluster_s6_multibubble_camera_reduction.md](cluster_s6_multibubble_camera_reduction.md), [cluster_s7_decoupling_payload_discharge.md](cluster_s7_decoupling_payload_discharge.md), and [cluster_s8_terminal_nonlinear_profile_decoupling.md](cluster_s8_terminal_nonlinear_profile_decoupling.md) for the reduction to multibubble residue, exclusion of regular same-point cascades, camera reduction of multibubbles, discharge of the decoupling payloads from \(K_{\mathrm{NLProfDec},NS3D}^+\), and proof of \(K_{\mathrm{NLProfDec},NS3D}^+\) in terminal cameras.
9. [cluster_s9_scattering_branch_discharge.md](cluster_s9_scattering_branch_discharge.md), [cluster_s10_exterior_regular_discard.md](cluster_s10_exterior_regular_discard.md), [cluster_s11_caccioppoli_regularity_discharge.md](cluster_s11_caccioppoli_regularity_discharge.md), and [cluster_u5a_bare_data_caccioppoli.md](cluster_u5a_bare_data_caccioppoli.md) for the small-data scattering branch, exterior-regular discard, and Caccioppoli regularity certificates used by S8 and the rough-core bridge.
