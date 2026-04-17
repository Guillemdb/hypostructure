# Pending targets for the compact Type II exclusion program

The items below list the theorem-level steps required in [compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md) to complete the conditional route.

## DAG-local assumption convention

The roadmap is a directed acyclic certificate ledger.  Every theorem is read
at the node where it is used.  Upstream certificates already emitted on that
path are safe assumptions at the current node, and no downstream theorem
routes back to reopen an earlier branch.  If a live certificate fails, the
output is a named downstream defect stratum.  It is not interpreted as a
failure to prove the upstream route from arbitrary Navier-Stokes data.

In particular, once \(K_{\mathrm{ScattBranch}}^-\) has been emitted, later
nodes do not reconsider the scattering branch.  U2a is similarly local to the
represented node: it proves automatic representation identities once the AC
chart and AC repaired-gauge path are the live inputs at that node.

## Progress status

- The declared NS3D repaired-gauge Type II backend contract is fixed in
  [ns3d_repaired_gauge_backend_contract.md](/home/guillem/hypostructure/docs/source/type_2/ns3d_repaired_gauge_backend_contract.md).
  This is the local contract under which C1--C4 are explicit inside the declared backend.
- T1--T6 are now packaged as a theorem-cluster note:
  [cluster_next_easiest_theorems.md](/home/guillem/hypostructure/docs/source/type_2/cluster_next_easiest_theorems.md)
  - This is an implementation note for the cluster statement layer, not a full closure of external analytic sublemmas.
  - It records how each theorem is reduced to explicit hypotheses already used in the master note.
- The preferred final compactness closure is now Theorem A'' in
  [compact_typeII_master_note_repaired_gauge.md](/home/guillem/hypostructure/docs/source/type_2/compact_typeII_master_note_repaired_gauge.md),
  which uses good-window selection instead of endpoint trace compactness.
- The C-series bridge stack is implemented through C18. It now includes
  declared-backend Type II exhaustion, repaired-gauge representation and
  explicit C2.R representation-payload discharge, critical \(L^3\)-normalization, cost compilation, two-bucket finite-cost
  classification, radiative and rough-core blocker routes, NS-valid Type II
  promotion, formal `UP-TypeII` application criteria, explicit NS3D
  application checklists, moving monotonicity, the scale-negative drift
  dichotomy, a scale-collapse barrier fallback, and the final NS3D
  `UP-TypeII` assembly theorem.
- S3 now adds a scale-collapse generalized self-similar reduction alternative
  to the C17 cost bridge: scale-collapse drift is blocked only after the
  compactness, autonomous-modulation, stationary-limit, and Liouville-rigidity
  payloads for the selected parameter regime are supplied.
- The modulation-force stack is implemented through T18. Pointwise forcing is
  reduced by T14 and the pointwise scale-force decomposition; the non-circular
  good-window route is reduced by T15--T18 to integrated translation forcing,
  annular convective regularity, weighted \(L^4\)-window control, weighted
  diffusion control, and weighted pressure control.
- S2 is implemented as a rough-core status theorem plus a vorticity-controlled
  subtype exclusion. It records that the 3D vorticity/enstrophy route does not
  self-close because of vortex stretching, and proves that
  \(K_{\mathrm{VortL^2Win}}^+\) gives \(K_{\mathrm{WinH1}}^+\) on represented
  bounded-critical-norm divergence-free branches.
- S4 is implemented as the current NS3D endpoint theorem: after technical
  bridge payloads, exterior-regular far-radiation discard, rough-core
  Caccioppoli closure, and either the C17 scale-collapse cost bridge or the S3
  compactness/modulation/stationary-limit/Liouville route are supplied, the
  only remaining Type II residue is multibubble concentration.
- S5 is implemented as the same-point multiscale cascade exclusion theorem. It
  proves critical \(L^3\)-mass exhaustion of cascades, discharges the regular
  nested-cascade inner expansion and perturbative S3 robustness estimates, and
  rules out regular strict same-point cascades. It also proves that the
  nonlinear no-splitting payload \(K_{\mathrm{SamePointNLDec}}^+\) rules out
  every same-point multibubble cascade. The remaining same-point case is
  exactly \(K_{\mathrm{SamePointNLDec}}^-\).
- S6 is implemented as the multibubble camera-reduction theorem. It proves
  that comparable same-point profiles are compound profiles, separated
  physical-point profiles are locally invisible in another point's camera, and
  every multibubble candidate is ruled out under
  \(K_{\mathrm{SamePointNLDec}}^+\),
  \(K_{\mathrm{MultiPointCamDec}}^+\), and
  \(K_{\mathrm{S3NRSPayload}}^+\).
- S7 is implemented as the decoupling-payload discharge theorem. It proves
  \(K_{\mathrm{SamePointNLDec}}^+\) and \(K_{\mathrm{MultiPointCamDec}}^+\)
  from the single nonlinear profile-evolution theorem
  \(K_{\mathrm{NLProfDec},NS3D}^+\), using local perturbation stability,
  static \(L^3\) decoupling, and pressure kernel-tail estimates.
- S8 is implemented as the terminal nonlinear profile-decoupling theorem. It
  proves \(K_{\mathrm{NLProfDec},NS3D}^+\) for terminal active cameras from
  \(K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+\), removal of the scattering branch,
  exterior-regular discard, repaired-gauge representation, and Caccioppoli
  regularity. The terminal quantifier is part of the theorem: arbitrary
  nonterminal cameras can still see smaller same-point bubbles.
- S9--S11 discharge three auxiliary certificates used by S8 and the rough-core
  bridge: \(K_{\mathrm{ScattBranch}}^-\) from small-data critical \(L^3\)
  theory, \(K_{\mathrm{ExtRegDiscard}}^+\) from single-point exterior
  regularity, and \(K_{\mathrm{CaccioppoliReg}}^+\) from physical suitability
  plus the declared repaired-gauge representation via U5a.
- S12 discharges the terminal profile-completeness slot from the standard
  critical Navier-Stokes profile-decomposition and nonlinear stability theorem:
  \(K_{\mathrm{TermCritProfThm},NS3D}^+\Rightarrow
  K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+\). This is the concrete
  Type-II-local replacement for the handwavy global `CatLib` placeholder. The
  small-data factor is now the explicit Kato \(L^3\) theorem, and the profile
  factor is the accepted critical NS profile-decomposition/nonlinear stability
  theorem in the terminal-window form needed by S8.
- S13 discharges the bounded critical terminal-sequence input in S12 from
  the stratified finite critical packet and terminal sequence routing:
  \(K_{\mathrm{StratCritPacket}}^+\wedge K_{\mathrm{TermSeqFromOrbit}}^+
  \Rightarrow K_{\mathrm{BoundedCritTermSeq}}^+\). Failure is classified as
  \(K_{\mathrm{TermSeqRoute}}^-\), \(K_{L^3\mathrm{Dom}}^-\),
  \(K_{L^3\mathrm{Inf}}^-\), or \(K_{L^3\mathrm{Zero}}^-\). After S14 is
  imported, the C18 terminal route uses
  \(K_{\mathrm{StratCritPacket}}^+\wedge K_{\mathrm{RepBridge}}^+\).
- S14 discharges terminal sequence routing from repaired-gauge representation:
  terminal-camera construction is built into the declared S8 terminal backend,
  and \(K_{\mathrm{RepBridge}}^+\Rightarrow
  K_{\mathrm{TermSeqFromOrbit}}^+\). Failure inside the declared terminal
  backend is \(K_{\mathrm{RepBridge}}^-\); malformed terminal cameras are
  outside the declared backend, not a Type II singularity class.
- U2a is implemented in
  [cluster_u2a_repaired_gauge_representation_pieces.md](/home/guillem/hypostructure/docs/source/type_2/cluster_u2a_repaired_gauge_representation_pieces.md).
  It proves the automatic repaired-gauge representation pieces from an AC raw
  chart and an AC repaired-gauge path: final chart identity, pressure
  reconstruction modulo constants, modulation coefficient formulas, AC gauge
  regularity, and critical \(L^3\)-norm invariance. The remaining U2 representation inputs are
  chart extraction, repaired-gauge root solvability, AC root selection,
  and terminal admissibility of the final scale-time chart.

The low-hanging bare-data interface upgrades now completed are:

1. \(K_{\mathrm{TermCamConstruct}}^+\) is definitional in the declared S8
   terminal backend.
2. \(K_{\mathrm{TermSeqFromOrbit}}^+\) is discharged from
   \(K_{\mathrm{RepBridge}}^+\) by S14.
3. \(K_{\mathrm{SmallDataStab}_{L^3}}^+\) is emitted by the explicit Kato
   small-data theorem in S12.
4. \(K_{\mathrm{ExtRegDiscard}}^+\) includes the positive-distance profile
   localization lemma in S10.
5. \(K_{\mathrm{CriticalNSProfDecomp}}^+\) is now stated as the precise
   imported critical profile-decomposition/nonlinear stability theorem needed
   by the terminal profile route.
6. U3a is implemented in
   [cluster_u3a_zero_critical_mass_exclusion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_u3a_zero_critical_mass_exclusion.md):
   retained Type II cores emit subsequential nontriviality
   \(K_{\mathrm{CoreSubseqNontriv}}^+\), and
   \(K_{\mathrm{CorePersist}}^+\wedge K_{\mathrm{CoreSubseqNontriv}}^+
   \Rightarrow \neg K_{L^3\mathrm{Zero}}^-\). Thus zero mass is removed only
   after the explicit no-return/persistence payload is supplied.
7. U2a is implemented in
   [cluster_u2a_repaired_gauge_representation_pieces.md](/home/guillem/hypostructure/docs/source/type_2/cluster_u2a_repaired_gauge_representation_pieces.md):
   pressure pullback, modulation formulas, final AC chart identities,
   critical \(L^3\)-invariance, and gauge-regularity inheritance are automatic
   after an AC chart and AC repaired-gauge path are supplied.
8. U3b is implemented in
   [cluster_u3b_critical_mass_completion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_u3b_critical_mass_completion.md):
   it proves local positive finite critical mass for every retained active
   terminal stratum from the exhaustive state-space partition and terminal
   profile completeness. It also proves finite active-packet bounds from
   terminal profile mass decoupling. No global \(L^3\) estimate is used.

At the current DAG frontier, the remaining live downstream strata/payload
nodes are:

\[
K_{\mathrm{ClassComplete}}^+,\quad
K_{\mathrm{RepBridge}}^+
\text{ reduced by U2a to chart extraction and repaired-gauge solve},\quad
K_{\mathrm{StateStratExh}}^+ and K_{\mathrm{StratCritMass}}^+,\quad
K_{\mathrm{C6Route}}^+,\quad
K_{\mathrm{CaccioppoliReg}}^+,\quad
K_{\mathrm{S3NRSPayload}}^+,
\]

plus the imported critical profile theorem
\(K_{\mathrm{CriticalNSProfDecomp}}^+\). If the target conclusion is the
formal hypostructure suppression certificate rather than a PDE-level blocked
Type II certificate, one also needs
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\).

## Exclusion certificate ledger

The roadmap tracks Type II exclusion by explicit certificate packages. A
stratum is considered excluded only when the certificates in the middle column
are supplied.

| Type II stratum | Certificates required for exclusion | Theorem proving the exclusion |
|---|---|---|
| Represented compact single-core branch | \(K_{\mathrm{RepBridge}}^+\) (discharged by C2.R from \(K_{\mathrm{RepDischarge},NS3D}^+\)), \(K_{\mathrm{StratCritPacket}}^+\), \(K_{L^3\mathrm{Tight}}^+\), \(K_{\mathrm{WinH1}}^+\), pressure reconstruction, bounded modulation, and good-window compactness | Theorem A'' in [compact_typeII_master_note_repaired_gauge.md](/home/guillem/hypostructure/docs/source/type_2/compact_typeII_master_note_repaired_gauge.md), with T1--T7 supplied by [cluster_next_easiest_theorems.md](/home/guillem/hypostructure/docs/source/type_2/cluster_next_easiest_theorems.md) and [cluster_t7_L3_compactness_upgrade.md](/home/guillem/hypostructure/docs/source/type_2/cluster_t7_L3_compactness_upgrade.md) |
| Finite-cost non-suppressed branch classification | \(K_{\mathrm{ClassComplete}}^+\), \(K_{\mathrm{CostBridge}}^+\), represented positive finite critical mass | C4 in [cluster_c4_cost_bridge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c4_cost_bridge.md) and C5 in [cluster_c5_two_bucket_classification.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c5_two_bucket_classification.md) |
| Small-data scattering profile branch | Small critical profile norm \(\|\phi\|_3\le\varepsilon_{\mathrm{sd}}\), Kato critical mild stability, and the scattering ledger | S9 in [cluster_s9_scattering_branch_discharge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s9_scattering_branch_discharge.md), proving \(K_{\mathrm{ScattBranch}}^-\) |
| Non-bubble radiative/noncompact branch | \(K_{\mathrm{RadBlk}}^+\equiv K_{L^3\mathrm{Tight}}^+\), supplied by a C6 route | C6 in [cluster_c6_radiative_noncompact_exclusion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c6_radiative_noncompact_exclusion.md) |
| Physically exterior far radiation | \(K_{\mathrm{SinglePointBlowup}}^+\wedge K_{\mathrm{ExtRegDiscard}}^+\) | S4 in [cluster_s4_multibubble_residue_classification.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s4_multibubble_residue_classification.md), with \(K_{\mathrm{ExtRegDiscard}}^+\) proved in S10 [cluster_s10_exterior_regular_discard.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s10_exterior_regular_discard.md) |
| Rough-core branch | \(K_{\mathrm{RepBridge}}^+\), \(K_{L^3\mathrm{Bd}}^+\), \(K_{\mathrm{ModMatrixInv}}^+\), \(K_{\mathrm{ModForceBd}}^+\), \(K_{\mathrm{CaccioppoliReg}}^+\), pressure reconstruction | C6/T13 in [cluster_c6_windowed_h1_bridge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c6_windowed_h1_bridge.md) and C7 in [cluster_c7_win_h1_certificate_composition.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c7_win_h1_certificate_composition.md), with \(K_{\mathrm{CaccioppoliReg}}^+\) proved in U5a [cluster_u5a_bare_data_caccioppoli.md](/home/guillem/hypostructure/docs/source/type_2/cluster_u5a_bare_data_caccioppoli.md), strengthening S11 [cluster_s11_caccioppoli_regularity_discharge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s11_caccioppoli_regularity_discharge.md) |
| Vorticity-controlled rough-core subtype | \(K_{\mathrm{RepBridge}}^+\wedge K_{L^3\mathrm{Bd}}^+\wedge K_{\mathrm{VortL^2Win}}^+\) | S2 in [cluster_s2_rough_core_vorticity_exclusion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s2_rough_core_vorticity_exclusion.md) |
| Scale-collapse drift | C17 cost bridge \(K_{\mathrm{ScaleCollapseCostBridge}}^+\) or \(K_{\mathrm{AbsScaleCostBridge}}^+\), or S3 compactness/autonomous/stationary/Liouville payload \(K_{\mathrm{S3NRSPayload}}^+\) | C16 in [cluster_c16_scale_negative_drift_dichotomy.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c16_scale_negative_drift_dichotomy.md), C17 in [cluster_c17_scale_collapse_barrier.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c17_scale_collapse_barrier.md), and S3 in [cluster_s3_scale_collapse_attractor_stratification.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s3_scale_collapse_attractor_stratification.md) |
| Regular strict same-point cascade | S5 profile decoupling, active-bubble mass floor, inner-representation, perturbative S3 robustness, and \(K_{\mathrm{S3NRSPayload}}^+\) | S5 in [cluster_s5_multiscale_cascade_exclusion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s5_multiscale_cascade_exclusion.md) |
| General same-point and multipoint multibubble | \(K_{\mathrm{TechTypeII}}^{\mathrm{blk},+}\), \(K_{\mathrm{StratCritPacket}}^+\), \(K_{\mathrm{SmallDataStab}_{L^3}}^+\), \(K_{\mathrm{CriticalNSProfDecomp}}^+\), \(K_{\mathrm{ScattBranch}}^-\), \(K_{\mathrm{ExtRegDiscard}}^+\), \(K_{\mathrm{RepBridge}}^+\), \(K_{\mathrm{CaccioppoliReg}}^+\), \(K_{\mathrm{S3NRSPayload}}^+\) | S6 in [cluster_s6_multibubble_camera_reduction.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s6_multibubble_camera_reduction.md), S7 in [cluster_s7_decoupling_payload_discharge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s7_decoupling_payload_discharge.md), S8 in [cluster_s8_terminal_nonlinear_profile_decoupling.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s8_terminal_nonlinear_profile_decoupling.md), S12 in [cluster_s12_terminal_profile_completeness.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s12_terminal_profile_completeness.md), S13 in [cluster_s13_bounded_critical_terminal_sequences.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s13_bounded_critical_terminal_sequences.md), and S14 in [cluster_s14_terminal_sequence_routing.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s14_terminal_sequence_routing.md), with \(K_{\mathrm{ScattBranch}}^-\), \(K_{\mathrm{ExtRegDiscard}}^+\), and \(K_{\mathrm{CaccioppoliReg}}^+\) proved in S9--S11 and strengthened by U5a |
| Post-UP Type II suppression from a blocked certificate | \(K_{\mathrm{NS\text{-}UPTypeII}}^+\), \(K_{\mathrm{FormalUPTypeIIApp},NS3D}^+\), emitted \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\), and localized monotonicity/finite-error certificates | C10--C14 in [cluster_c10_ns_up_typeII_promotion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c10_ns_up_typeII_promotion.md), [cluster_c11_generic_up_typeII_payload_discharge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c11_generic_up_typeII_payload_discharge.md), [cluster_c12_ns_localized_monotonicity_translation.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c12_ns_localized_monotonicity_translation.md), [cluster_c13_formal_up_typeII_application.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c13_formal_up_typeII_application.md), and [cluster_c14_explicit_up_typeII_ns3d_checklist.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c14_explicit_up_typeII_ns3d_checklist.md) |
| Terminal sequence routing | \(K_{\mathrm{RepBridge}}^+\) inside the declared S8 terminal backend | S14 in [cluster_s14_terminal_sequence_routing.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s14_terminal_sequence_routing.md) |
| Bounded terminal critical sequences | \(K_{\mathrm{StratCritPacket}}^+\wedge K_{\mathrm{RepBridge}}^+\) | S13 in [cluster_s13_bounded_critical_terminal_sequences.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s13_bounded_critical_terminal_sequences.md), with routing supplied by S14 |
| Terminal profile completeness | \(K_{\mathrm{StratCritPacket}}^+\), \(K_{\mathrm{RepBridge}}^+\), \(K_{\mathrm{SmallDataStab}_{L^3}}^+\), and accepted critical NS profile decomposition \(K_{\mathrm{CriticalNSProfDecomp}}^+\) | S12 in [cluster_s12_terminal_profile_completeness.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s12_terminal_profile_completeness.md), with boundedness supplied by S13 and routing by S14 |
| Final declared NS3D Type II suppression | \(K_{\mathrm{UPTypeII},NS3D}^{\mathrm{term}}\), i.e. \(K_{\mathrm{ClassComplete}}^+\), \(K_{\mathrm{TechTypeII}}^{\mathrm{blk},+}\), \(K_{\mathrm{C6Route}}^+\), \(K_{\mathrm{StratCritPacket}}^+\), \(K_{\mathrm{SmallDataStab}_{L^3}}^+\), \(K_{\mathrm{CriticalNSProfDecomp}}^+\), emitted \(K_{\mathrm{ScattBranch}}^-\), exterior discard, repaired-gauge representation, Caccioppoli regularity, S3 scale-collapse rigidity, and \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) | C18 in [cluster_c18_final_up_typeII_ns3d.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c18_final_up_typeII_ns3d.md) |

Any missing certificate in this ledger is retained as a named defect in the
residual classification. No excluded stratum is used without its listed proof
certificate.

## Remaining downstream-stratum program

C18 proves Type II exclusion inside the declared terminal NS3D repaired-gauge
backend. To close the current DAG frontier, each live downstream defect stratum
must either be killed by a certificate theorem or retained as an explicit
terminal obstruction.  No item below reopens an upstream branch.

### U1. Bare-data Type II backend entry

Target implication:
\[
\text{actual suitable NS3D Type II blowup}
\Longrightarrow
K_{\mathrm{ClassComplete}}^+.
\]

Expanded target:
\[
\text{actual Type II blowup}
\Longrightarrow
K_{\mathrm{TypeIIExhaust}}^+
\wedge
K_{\mathrm{CmuExtract}}^+
\wedge
K_{\mathrm{ScaleRoute}}^+
\wedge
K_{\mathrm{ProfComplete}}^+
\wedge
K_{\mathrm{LostTypeII}}^+.
\]

Role: prove that no genuine Type II branch is lost before the repaired-gauge
terminal backend is reached. This is the first global completeness theorem:
it turns the C1 declared-backend exhaustion theorem into a statement about
arbitrary suitable weak Type II blowup branches.

Main defects if U1 fails:
\[
K_{\mathrm{CmuExtract}}^-,
\quad
K_{\mathrm{ScaleRoute}}^-,
\quad
K_{\mathrm{ProfComplete}}^-,
\quad
K_{\mathrm{LostTypeII}}^-.
\]

Status: **missing**. This is a backend-completeness theorem, not a local
estimate.

### U2. Bare-data repaired-gauge representation

Target implication:
\[
\text{actual extracted Type II branch}
\Longrightarrow
K_{\mathrm{RepBridge}}^+.
\]

Expanded target:
\[
\text{profile branch}
\Longrightarrow
(V,P,a,b)
\text{ solves the repaired-gauge renormalized NS system}
\]
with admissible AC chart maps, pressure reconstruction, AC repaired-gauge solve, and modulation coefficients compatible with the C2/C2.R contract.

Role: this now discharges more than representation. By S14, once the branch is
in the declared terminal backend,
\[
K_{\mathrm{RepBridge}}^+
\Longrightarrow
K_{\mathrm{TermSeqFromOrbit}}^+.
\]
Thus U2 also removes terminal sequence routing as an independent issue.

Main diagnostics if U2 fails outside the declared backend:
\[
K_{\mathrm{Chart}}^-,
\quad
K_{\mathrm{GaugeSolve}}^-.
\]
U2a refines these diagnostics further.  Once an AC raw chart and an AC
repaired-gauge path are supplied, pressure reconstruction, modulation
coefficient realization, AC gauge regularity, final chart identities, and
critical \(L^3\)-invariance are
theorems.  The remaining genuine representation inputs are chart extraction,
repaired-gauge root solvability, AC root selection, terminal admissibility of
the final scale-time chart.

Status: **DAG-local automatic pieces implemented by C2/C2.R and [cluster_u2a_repaired_gauge_representation_pieces.md](/home/guillem/hypostructure/docs/source/type_2/cluster_u2a_repaired_gauge_representation_pieces.md)**. The remaining U2 outputs are downstream representation strata: chart extraction, repaired-gauge root solvability, AC root selection, and terminal admissibility.

### U3. Stratified critical \(L^3\)-mass from state-space partition

Target implication:
\[
\text{terminal state-space partition}
\Longrightarrow
K_{\mathrm{StratCritMass}}^+.
\]

Expanded target:
\[
\forall \mathfrak s\in\mathcal S_{\mathrm{act}},
\qquad
0<\|\Phi_{\mathfrak s}\|_{L^3}<\infty.
\]

Role: supplies the positive finite critical-mass bookkeeping used by S12--S14
and C18 on each retained active terminal stratum. It is not a global estimate
on the full solution.

Main defects if U3 fails:
\[
K_{L^3\mathrm{Dom}}^-,
\quad
K_{L^3\mathrm{Zero}}^-,
\quad
K_{L^3\mathrm{Inf}}^-.
\]

Required subtheorems:

1. Nonzero concentration lower bound:
   \[
   \text{actual Type II concentration}
   \Longrightarrow
   \limsup_{\tau\to\infty}\|V(\tau)\|_3>0.
   \]
2. Zero-mass exclusion:
   \[
   K_{L^3\mathrm{Zero}}^-
   \Longrightarrow
   \text{no nontrivial Type II core}.
   \]
3. Infinite critical-mass routing:
   \[
   K_{L^3\mathrm{Inf}}^-
   \Longrightarrow
   \text{accepted barrier defect or outside-Type-II classification}.
   \]

Status: **stratified local discharge implemented by U3b**. The terminal
backend no longer requires a global \(L^3\) bound for the full solution. It
requires an exhaustive terminal state-space partition and positive finite
critical mass on each retained active stratum.

Implemented subresult U3a:

\[
K_{\mathrm{CorePersist}}^+
\wedge
K_{\mathrm{CoreSubseqNontriv}}^+
\Longrightarrow
\neg K_{L^3\mathrm{Zero}}^-.
\]

See
[cluster_u3a_zero_critical_mass_exclusion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_u3a_zero_critical_mass_exclusion.md).
After U3a, the remaining U3 work is:

1. prove the core persistence/no-return payload
   \(K_{\mathrm{CorePersist}}^+\), or classify its failure as an upstream
   oscillatory/recurrence defect;
2. prove \(K_{\mathrm{RawOrbL3},NS3D}^+\), which gives
   \(K_{L^3\mathrm{Dom}}^+\) by U3b;
3. prove the upper finite critical-mass bound \(K_{L^3\mathrm{Bd}}^+\), or
   route failure through \(K_{L^3\mathrm{InfRoute}}^+\);
4. decide whether \(K_{L^3\mathrm{Inf}}^-\) is accepted as a direct barrier
   defect or classified outside the finite-critical terminal backend.

Implemented subresult U3b:

\[
K_{\mathrm{StateStratExh}}^+
\wedge
K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+
\Longrightarrow
K_{\mathrm{StratCritMass}}^+.
\]

For bounded terminal packets, U3b also emits
\[
K_{\mathrm{StratCritPacket}}^+(\eta,M,J)
\]
from profile mass decoupling and the active-profile threshold. This is the
local stratified replacement for global \(K_{\mathrm{StratCritPacket}}^+\) in the
terminal backend.

See
[cluster_u3b_critical_mass_completion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_u3b_critical_mass_completion.md).


### U4. No-radiation/tightness discharge

Target implication:
\[
\text{actual represented Type II branch}
\Longrightarrow
K_{\mathrm{C6Route}}^+
\Longrightarrow
K_{\mathrm{RadBlk}}^+
\equiv
K_{L^3\mathrm{Tight}}^+.
\]

Equivalent target:
\[
\forall\varepsilon>0\ \exists R<\infty:
\sup_{\tau\gg1}
\int_{|y|>R}|V(y,\tau)|^3\,dy
<\varepsilon,
\]
or a proof that failure of this tightness is already one of the named
S4--S8 profile/radiation defects that the terminal backend blocks.

Role: removes non-bubble radiation and supplies the compactness side of the
single-core barrier. This is one of the main remaining analytic obstacles.

Main defects if U4 fails:
\[
K_{L^3\mathrm{Tight}}^-,
\quad
K_{\mathrm{RadBlk}}^-,
\quad
K_{\mathrm{ExtRegDiscard}}^-,
\quad
K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^-.
\]

Status: **missing**. This is the highest-value analytic theorem after U2/U3.

### U5. Bare-data Caccioppoli/windowed \(H^1\) route

Target implication:
\[
\text{actual represented suitable NS3D Type II branch}
\Longrightarrow
K_{\mathrm{CaccioppoliReg}}^+,
\]
and then, with the C7 composition inputs,
\[
K_{\mathrm{RepBridge}}^+
\wedge
K_{L^3\mathrm{Bd}}^+
\wedge
K_{\mathrm{ModMatrixInv}}^+
\wedge
K_{\mathrm{ModForceBd}}^+
\wedge
K_{\mathrm{CaccioppoliReg}}^+
\Longrightarrow
K_{\mathrm{WinH1}}^+.
\]

Role: removes rough-core Type II branches by producing the windowed local
\(H^1\) control needed by the compact barrier.

Main defects if U5 fails after U5a:
\[
K_{\mathrm{Suitability}}^-,
\quad
K_{\mathrm{ModMatrixInv}}^-,
\quad
K_{\mathrm{ModForceBd}}^-,
\quad
K_{L^3\mathrm{Bd}}^-.
\]

Status: **U5a implemented in [cluster_u5a_bare_data_caccioppoli.md](/home/guillem/hypostructure/docs/source/type_2/cluster_u5a_bare_data_caccioppoli.md)**. U5a proves \(K_{\mathrm{CaccioppoliReg}}^+\) from compact-cylinder physical suitability, the declared Type II route, and the declared repaired-gauge backend. Pressure reconstruction and AC gauge regularity are discharged by C2.R and are no longer independent U5 defects. U5a is deliberately local: it does not assume global \(L^3\), global tightness, bounded modulation, finite energy at infinity, global compactness, or uniform windowed \(H^1\). C7/C6 still require the separate bounded critical-norm and modulation-control inputs to upgrade Caccioppoli regularity to \(K_{\mathrm{WinH1}}^+\).

### U6. Scale-collapse rigidity or accepted scale-collapse cost

Target implication, rigidity route:
\[
\text{actual scale-collapse Type II branch}
\Longrightarrow
K_{\mathrm{S3NRSPayload}}^+.
\]

Alternative target, cost route:
\[
\text{actual scale-collapse Type II branch}
\Longrightarrow
K_{\mathrm{ScaleCollapseCostBridge}}^+
\quad\text{or}\quad
K_{\mathrm{AbsScaleCostBridge}}^+.
\]

Role: removes the scale-collapse drift branch either by the S3
compactness/autonomous/stationary/Liouville route or by direct acceptance of
the scale-collapse cost by the `BarrierTypeII` evaluator.

Required rigidity subtheorems:

1. negative-drift window extraction;
2. compactness/tightness of the selected windows;
3. autonomous modulation limit;
4. nonzero stationary or invariant-measure extraction;
5. parameter reduction to the NRŠ backward-self-similar class or another
   covered Liouville regime;
6. Liouville rigidity for the extracted limit.

Status: **structured by S3/C16/C17, but the full payload is still missing**.

### U7. Single-core exterior localization or multipoint backend completion

Target implication:
\[
\text{actual Type II branch}
\Longrightarrow
K_{\mathrm{SinglePointBlowup}}^+
\quad\text{or}\quad
\text{multipoint terminal backend route}.
\]

Role: S10 proves
\[
K_{\mathrm{SinglePointBlowup}}^+
\Longrightarrow
K_{\mathrm{ExtRegDiscard}}^+.
\]
At this downstream localization node, either the branch is localized to one
terminal core after profile splitting, or every additional physical point is
already routed into the S6/S8 terminal profile machinery.

Main defects if U7 fails:
\[
K_{\mathrm{SinglePointBlowup}}^-,
\quad
K_{\mathrm{MultiPointRoute}}^-,
\quad
K_{\mathrm{ExtRegDiscard}}^-.
\]

Status: **partially covered by S4/S6/S10; the live downstream localization
strata are \(K_{\mathrm{SinglePointBlowup}}^-\),
\(K_{\mathrm{MultiPointRoute}}^-\), and \(K_{\mathrm{ExtRegDiscard}}^-\)**.

### U8. Critical NS profile theorem citation or internal proof

Target implication:
\[
\text{bounded terminal critical sequences}
\Longrightarrow
K_{\mathrm{CriticalNSProfDecomp}}^+.
\]

Role: S12 now states the exact theorem needed: critical \(L^3_\sigma\)
profile decomposition, Kato-small heat remainder, nonlinear profile
stability on compact terminal windows, and hidden terminal-window mass
extraction. To make the document fully self-contained, this theorem must be
proved internally. For a rigorous paper-style proof, it can instead be cited
as an external theorem with exact references and matching function spaces.

Status: **precisely stated but imported**.

### U9. Formal NS `UP-TypeII` promotion

Target implication:
\[
\text{NS3D blocked Type II certificate}
\Longrightarrow
K_{\mathrm{NS\text{-}UPTypeII}}^+.
\]

Equivalently, discharge the C10--C14 application package:
\[
K_{\mathrm{FormalUPTypeIIApp},NS3D}^+.
\]

Role: needed only if the desired final output is the hypostructure
suppression certificate
\[
K_{\mathrm{SC}_\lambda}^{\sim}.
\]
If the target is a PDE-level blocked Type II exclusion, U9 is optional.

Status: **structured by C10--C14/C15--C17; still requires the localized
monotonicity or moving-cutoff finite-error payloads for the exact theorem
variant used**.

### U10. Final bare-data Type II exclusion assembly

Target theorem:
\[
\text{actual suitable NS3D Type II blowup}
\Longrightarrow
K_{\mathrm{UPTypeII},NS3D}^{\mathrm{term}}.
\]

Then C18 gives
\[
\forall\omega\in\mathcal U_{\mathrm{II}}^{NS},
\qquad
K_{\mathrm{SC}_\lambda}^{\sim}(\omega),
\]
or, if U9 is omitted, the PDE-level blocked Type II certificate
\[
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
\]

Status: **not implemented**. This is the final assembly after U1--U8, plus U9
if formal UP suppression is required.

## Tier 1: Technical closures

### T1. Uniform local pressure-tail bound
\[
\sup_{\tau}\sup_{x\in B_R}\int_{|z|>2R}\frac{|V(z,\tau)|^2}{|x-z|^5}\,dz<\infty.
\]

- Role: close the local pressure-tail estimate and give uniform local \(L^{3/2}\) pressure control modulo constants.
- Difficulty: Technical.
- Dependencies: local pressure representation and critical-norm tightness.
- Status: **implemented (theorem form, with full dyadic proof)**

### T1.4. Verification of the T1.5 pressure hypotheses
Prove that
\[
L^\infty_\tau L^3_y(\mathbb R^3)\cap L^2_\tau H^1_y(B_{4R})
\]
implies both
\[
V\in L^4_\tau L^4_y(B_{4R})
\]
and
\[
\mathcal H_R\in L^2_\tau.
\]

- Role: remove the apparent extra assumptions in T1.5.
- Difficulty: Technical.
- Dependencies: local Sobolev interpolation and dyadic far-field summation from global \(L^3\).
- Status: **implemented (theorem form, with interpolation and dyadic proof)**

### T1.5. Local \(L^2\) pressure estimate modulo constants
Prove that local \(L^4_{t,x}\) velocity control plus a fourth-order far-field pressure tail gives
\[
P-c_R\in L^2_\tau L^2_y(B_R).
\]

- Role: provide the pressure input required by T6.
- Difficulty: Technical.
- Dependencies: T1.4, Calderon-Zygmund for the local part, and harmonic far-field kernel estimates.
- Status: **implemented (T1.4 verifies the hypotheses from \(L^\infty_\tau L^3_y\) and local \(L^2_\tau H^1_y\); T1.5 gives the pressure estimate)**
- Note: \(L^\infty_\tau L^{3/2}_y\) pressure control alone is not enough for the Hilbert \(H^{-1}\) estimate in T6; T1.5 is retained.

### T2. Translation-block invertibility for the centering gauge
Prove uniform invertibility of the \((3\times 3)\) translation block of the modulation matrix.

- Role: provide one block of full modulation nondegeneracy.
- Difficulty: Moderate; reduced to core-mass dominance with annular error.
- Dependencies: centering block formulae.
- Status: **implemented (theorem form, with explicit Neumann-series proof)**

### T3. Cross-term control for the repaired modulation matrix
Prove uniform control of scale–translation and translation–scale mixed entries.

- Role: enable the Schur complement argument for full nondegeneracy.
- Difficulty: Moderate.
- Dependencies: \(DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)]=p\Theta_0>0\) and mixed-term estimates.
- Status: **implemented (theorem form, with annulus identity and Schur compatibility estimate)**

## Tier 2: Medium-level structural PDE-theorem inputs

### T4. Full repaired-gauge nondegeneracy
Prove uniform invertibility of the full \((4\times 4)\) modulation matrix along the compact renormalized orbit.

- Role: produce bounded modulation parameters \((a(\tau),b(\tau))\).
- Difficulty: Medium; combines Tier 1 inputs and Schur complement.
- Dependencies: T2 and T3.
- Status: **implemented (reduction to Schur complement + bounded cross-term ansatz)**

### T5. Boundedness of \((a(\tau),b(\tau))\)
Prove
\[
\sup_\tau(|a(\tau)|+|b(\tau)|)<\infty
\]
from repaired gauge identities and forcing control.

- Role: make the renormalized PDE estimates uniform in time.
- Difficulty: Medium.
- Dependencies: T4.
- Status: **implemented (linear-system reduction)**

### T6. Unconditional local \((L^2_\tau H^{-1}_{\mathrm{loc}})\) time regularity
Prove
\[
\partial_\tau V\in L^2_{\mathrm{loc}}(\tau;H^{-1}_{\mathrm{loc}}(y))
\]
from the renormalized NS equation.

- Role: provide the \(H^{-1}\) time-regularity input once local \(L^2_\tau L^2_y\) pressure control modulo constants is available.
- Difficulty: Medium; PDE-term estimates become standard once bounded modulation and the stronger pressure hypothesis are available.
- Dependencies: T1.4, T1.5, and T5.
- Status: **implemented once \(L^\infty_\tau L^3_y\), local \(L^2_\tau H^1_y\), and bounded modulation are available**

### T7. Strong local \(L^3\)-compactness upgrade
Prove strong local convergence of sampled renormalized states in \(L^3_{\mathrm{loc}}\), not merely \(L^2_{\mathrm{loc}}\).

- Role: match the compact low-dissipation contradiction theorem requirements.
- Difficulty: Medium; spacetime compactness is standard, but fixed sampled-time compactness needs an extra trace/compact-orbit input.
- Dependencies: T6, local \(L^6\)-type regularity, and either sampled-time strong \(L^2_{\mathrm{loc}}\)-compactness or the T7.4 good-time selection mechanism.
- Status: **implemented as a rigorous conditional theorem, including a good-time replacement for sampled compactness, in cluster_t7_L3_compactness_upgrade.md**

### T7.5. Good-window final barrier closure
Use finite total cost to produce vanishing-cost windows, select common good times using local \(L^2_\tau H^1_y\) bounds, and close the final contradiction without endpoint trace compactness.

- Role: make the final compactness step depend on good windows rather than arbitrary sampled endpoints.
- Difficulty: Low; measure selection plus Rellich compactness.
- Dependencies: T7.4, normalization, tightness, and uniform local \(L^2_\tau H^1_y\) bounds on unit windows.
- Status: **implemented as Theorem A'' in the master note**

### T7.6. Contrapositive classification of finite-cost candidates
Use Theorem A'' to show any finite-cost normalized Type II candidate must either lose global \(L^3\)-tightness or fail local windowed \(H^1\) control.

- Role: narrow the surviving finite-cost Type II scenarios to radiative/noncompact candidates or rough-core candidates.
- Difficulty: Low; direct contrapositive of Theorem A''.
- Dependencies: T7.5.
- Status: **implemented as Corollary 16 in the master note**

### T7.7. Hypostructure sieve-certificate integration
Link the compact Type II proof stack to the Hypostructure framework and the
Navier-Stokes dataset certificate trail without invoking the final global
regularity conclusion.

- Role: identify which certificates from the sieve can be imported into the Type II proof, isolate the bridge certificates needed to turn them into stronger Type II exclusion hypotheses, and mark that NS3D post-UP Type II suppression requires the explicit \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) bridge.
- Difficulty: Low as documentation; medium/high for the bridge proofs it exposes.
- Dependencies: Theorem A'', Corollary 16, `BarrierTypeII`, the C9 NS-valid Type II promotion bridge, and the Navier-Stokes dataset certificate table.
- Status: **implemented as [hypostructure_sieve_typeII_certificates.md](/home/guillem/hypostructure/docs/source/type_2/hypostructure_sieve_typeII_certificates.md)**

## Tier 3: Structural upgrades

## Chapter C: Classification-completeness program

This chapter turns the current certificate-level Type II classification into a
PDE-intrinsic classification of all potentially remaining Type II candidates in
the declared Navier-Stokes backend. It is intentionally separate from global
regularity: the target is to exhaust and eliminate the Type II branch, not to
close the full Lock/continuation route.

### C1. Type II branch exhaustion
Prove that every Type II candidate in the declared Navier-Stokes backend is
detected by the sieve branch
\[
K_{C_\mu}^+\wedge K_{\mathrm{SC}_\lambda}^-
\]
and enters the profile package \(K_{\mathrm{Prof}_{NS}}^+\).

- Role: eliminate the possibility of an unrepresented Type II singularity outside the sieve profile/germ formalism.
- Difficulty: High.
- Dependencies: concentration compactness, profile extraction modulo Navier-Stokes symmetries, and compatibility with the dataset backend.
- Output certificate: \(K_{\mathrm{TypeIIExhaust}}^+\).
- Status: **implemented and instantiated for the NS3D dataset in [cluster_c1_typeII_branch_exhaustion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c1_typeII_branch_exhaustion.md)**. The theorem proves \(K_{\mathrm{TypeIIExhaust}}^+\) from \(K_{\mathrm{ExhPayload}}^+\), classifies failures as \(K_{\mathrm{CmuExtract}}^-\), \(K_{\mathrm{ScaleRoute}}^-\), \(K_{\mathrm{ProfComplete}}^-\), or \(K_{\mathrm{LostTypeII}}^-\), and adds the concrete NS3D payload \(K_{\mathrm{ExhPayload},NS3D}^+\) from the dataset's \(K_{C_\mu}^+\), \(K_{\mathrm{SC}_\lambda}^-\), and \(K_{\mathrm{Prof}_{NS}}^+\) outputs.

### C2. Representation bridge
Prove that the sieve concentration/profile data gives a repaired-gauge
renormalized orbit
\[
(V,P,a,b)
\]
with center, scale, renormalized time, and gauge compatibility.

- Role: convert \(K_{C_\mu}^+\wedge K_{\mathrm{Prof}_{NS}}^+\) into the PDE object used by Theorem A''.
- Difficulty: High.
- Dependencies: repaired gauge construction, concentration profile regularity, pressure reconstruction.
- Output certificate: \(K_{\mathrm{RepBridge}}^+\).
- Status: **implemented and refined by [cluster_c2_representation_bridge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c2_representation_bridge.md), [cluster_c2_repbridge_payload_discharge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c2_repbridge_payload_discharge.md), and U2a [cluster_u2a_repaired_gauge_representation_pieces.md](/home/guillem/hypostructure/docs/source/type_2/cluster_u2a_repaired_gauge_representation_pieces.md)**. C2 proves \(K_{\mathrm{RepBridge}}^+\) from \(K_{\mathrm{RawOrb}}^+\), \(K_{\mathrm{GaugeReal}}^+\), \(K_{\mathrm{PressureRep}}^+\), and \(K_{\mathrm{ModParams}}^+\). C2.R discharges that high-level row from \(K_{\mathrm{Chart},NS3D}^+\) and \(K_{\mathrm{GaugeSolve},NS3D}^+\), including the necessary time reparametrization for scale-gauge changes. U2a proves the automatic subpieces from an AC chart and AC repaired-gauge path: pressure pullback, final modulation coefficients, final chart identity, AC gauge regularity, and critical \(L^3\)-invariance. Missing genuine subpayloads are chart extraction, repaired-gauge root solvability, AC root selection, and terminal admissibility.

### C3. Critical \(L^3\)-normalization bridge
Prove that every represented Type II candidate can be placed on the normalized
critical branch
\[
\|V(\tau)\|_{L^3(\mathbb R^3)}=1
\]
or else produces a separate non-Type-II/degenerate certificate.

- Role: remove normalization failure as an independent survivor mechanism.
- Difficulty: Medium.
- Dependencies: critical norm selection, scale gauge compatibility, nontriviality of the Type II profile.
- Output certificate: \(K_{\mathrm{StratCritPacket}}^+\).
- Status: **implemented in [cluster_c3_l3_normalization_bridge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c3_l3_normalization_bridge.md)** as a rigorous nonzero finite critical-mass bridge. Exact unit normalization is replaced by \(0<\eta\le\|V(\tau)\|_3\le M<\infty\), and failures are classified by \(K_{L^3\mathrm{Zero}}^-\), \(K_{L^3\mathrm{Inf}}^-\), or \(K_{L^3\mathrm{Dom}}^-\). Full removal of these defects from genuine Type II candidates is deferred to C2/C1 representation exhaustion.

### C4. PDE-to-framework cost bridge
Prove that the infinite localized PDE renormalization cost from Theorem A''
compiles into the framework's blocked Type II barrier:
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty
\Longrightarrow
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
\]

- Role: let the compact Type II PDE theorem trigger the blocked `BarrierTypeII` certificate; post-promotion suppression additionally needs \(K_{\mathrm{NS\text{-}UPTypeII}}^+\).
- Difficulty: Medium.
- Dependencies: comparison between \(\tilde{\mathfrak D}_{R_0}\) and the framework renormalization cost, monotonicity/barrier conventions in `BarrierTypeII`.
- Output certificate: \(K_{\mathrm{CostBridge}}^+\).
- Status: **implemented in [cluster_c4_cost_bridge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c4_cost_bridge.md)**. The default Navier-Stokes `BarrierTypeII` evaluator is defined with identity cost \(\mathfrak C_{\mathrm{II}}^{NS}=\tilde{\mathfrak D}_{R_0}\), so C4 is automatic in the default backend once the PDE cost is measurable, nonnegative, and locally integrable on finite \(\tau\)-intervals. If a different generic model cost is retained, the remaining obligation is the comparison witness \(K_{\mathrm{CostCompare}}^+\).

### C5. Two-bucket finite-cost classification
Combine C1--C4 with Corollary 16 / Theorem A'' to prove that every finite-cost,
non-suppressed Type II candidate is either radiative/noncompact or rough-core.

- Role: remove representation, normalization, and cost-adapter defects from the survivor list.
- Difficulty: Low after C1--C4.
- Dependencies: C1, C2, C3, C4, Theorem A'', and \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) for the promoted suppressed output.
- Output certificate: \(K_{\mathrm{ClassComplete}}^+\).
- Status: **implemented in [cluster_c5_two_bucket_classification.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c5_two_bucket_classification.md)**. The theorem defines \(K_{\mathrm{ClassComplete}}^+\) as the C1--C4 bridge package, adds total tightness/windowed-\(H^1\) evaluators, proves the ordered suppressed-or-two-bucket classification under the NS-valid promotion bridge, and derives the finite-cost non-suppressed corollary: the candidate emits either \(K_{L^3\mathrm{Tight}}^-\) or \(K_{\mathrm{WinH1}}^-\), i.e. it is radiative/noncompact or rough-core.

### C6. Radiative/noncompact branch exclusion
Prove that the structural sieve certificates rule out failure of global
critical \(L^3\)-tightness, or identify the exact additional no-radiation
backend needed.

- Role: eliminate the radiative/noncompact Type II bucket.
- Difficulty: High.
- Dependencies: T8--T10, no-radiation, no-splitting, and possibly a positive scattering/mixing certificate replacing \(K_{\mathrm{TB}_\rho}^{\mathrm{inc}}\).
- Output certificate: \(K_{\mathrm{RadBlk}}^+\) or \(K_{L^3\mathrm{Tight}}^+\).
- Status: **implemented as strengthened no-radiation certificate routes in [cluster_c6_radiative_noncompact_exclusion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c6_radiative_noncompact_exclusion.md)**. The note proves \(K_{\mathrm{RadBlk}}^+\equiv K_{L^3\mathrm{Tight}}^+\) from global \(L^3\)-compactness, compact-core/vanishing-remainder, no-splitting/no-radiation profiles, finite-library \(L^3\)-approximation, tame finite-description compact realization, or a-posteriori discharge of \(K_{L^3\mathrm{Tight}}^{\mathrm{inc}}\) through \(K_{\mathrm{C6Route}}^+\). It shows that C5 plus C6 leaves only rough-core finite-cost survivors.

### C7. Rough-core branch exclusion
Prove that local high-frequency or gradient concentration in a bounded
renormalized core is impossible under the available energy, pressure,
modulation, and sieve certificates.

- Role: eliminate the rough-core Type II bucket.
- Difficulty: High.
- Dependencies: local Caccioppoli, pressure control, repaired modulation, and possibly positive stiffness or oscillation certificates replacing \(K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}\) or \(K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}\).
- Output certificate: \(K_{\mathrm{RoughCoreBlk}}^+\) or \(K_{\mathrm{WinH1}}^+\).
- Status: **implemented at certificate level in [cluster_c7_win_h1_certificate_composition.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c7_win_h1_certificate_composition.md), using the PDE bridge [cluster_c6_windowed_h1_bridge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c6_windowed_h1_bridge.md). The theorem proves \(K_{\mathrm{RoughCoreBlk}}^+\) from \(K_{\mathrm{RepBridge}}^+\), \(K_{L^3\mathrm{Bd}}^+\), \(K_{\mathrm{ModMatrixInv}}^+\), \(K_{\mathrm{ModForceBd}}^+\), and \(K_{\mathrm{CaccioppoliReg}}^+\); \(K_{L^3\mathrm{Bd}}^-\) is discharged by [cluster_c7_l3bd_defect_discharge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c7_l3bd_defect_discharge.md), and \(K_{\mathrm{ModMatrixInv}}^-\) is discharged by [cluster_c7_modmatrix_inverse_discharge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c7_modmatrix_inverse_discharge.md) from the repaired-gauge nondegeneracy payload. T14 conditionally reduces pointwise \(K_{\mathrm{ModForceBd}}^+\) when independent local translation-force hypotheses are available. The pointwise scale-force side is decomposed in [cluster_c7_scale_force_bound.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c7_scale_force_bound.md), which proves the nonlinear row from \(K_{\mathrm{ScaleL4Mom}}^+\). T15--T17 in [cluster_t15_t16_integrated_modulation.md](/home/guillem/hypostructure/docs/source/type_2/cluster_t15_t16_integrated_modulation.md) give the integrated good-window replacement using \(K_{\mathrm{TransForceL^1Win}}^+\) and \(K_{\mathrm{ScaleForceL^1Win}}^+\). T18 in [cluster_t18_scale_force_decomposition.md](/home/guillem/hypostructure/docs/source/type_2/cluster_t18_scale_force_decomposition.md) is the integrated analogue, decomposing the scale-force defect into weighted diffusion, annular convective regularity, weighted \(L^4\)-convective, and weighted pressure pieces; the singular-weight boundary terms are discharged by the T18 cutoff argument. U5a proves \(K_{\mathrm{CaccioppoliReg}}^+\) from physical suitability and the declared repaired-gauge representation; pressure reconstruction and AC gauge regularity are discharged by C2.R. Remaining nontrivial rough-core defects are \(K_{\mathrm{ModForceBd}}^-\), bounded-modulation failure, or failure of physical suitability, with integrated forcing defects available for selected-time arguments.**

### C8. Type II branch exclusion theorem
Combine classification completeness with radiative and rough-core suppression:
\[
K_{\mathrm{ClassComplete}}^+
\wedge
K_{\mathrm{RadBlk}}^+
\wedge
K_{\mathrm{RoughCoreBlk}}^+
\wedge
K_{\mathrm{NS\text{-}UPTypeII}}^+
\Longrightarrow
\text{no admissible unresolved Type II branch in the declared backend}.
\]

- Role: final Type II-specific theorem, still short of global regularity.
- Difficulty: Low after C1--C7.
- Dependencies: C1--C7 and C10's NS-valid promotion bridge.
- Output: Type II branch exclusion in the declared Navier-Stokes hypostructure backend.
- Status: **implemented in [cluster_c8_typeII_branch_exclusion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c8_typeII_branch_exclusion.md)**. The theorem proves that \(K_{\mathrm{ClassComplete}}^+\wedge K_{\mathrm{RadBlk}}^+\wedge K_{\mathrm{RoughCoreBlk}}^+\wedge K_{\mathrm{NS\text{-}UPTypeII}}^+\) forces every declared Type II candidate to emit \(K_{\mathrm{SC}_\lambda}^{\sim}\), so no admissible unresolved Type II branch remains in the declared backend.

### C9. Energy-cost bridge
Relate the unweighted renormalized Type II barrier cost to the physical
Navier-Stokes energy budget.

- Role: prevent the overclaim that every infinite unweighted renormalization cost forces infinite initial energy; identify the scale-weighted cost actually controlled by physical energy.
- Difficulty: Medium.
- Dependencies: repaired-gauge representation, physical energy inequality \(K_{D_E}^+\), and weighted \(a_+\)-cost control.
- Output certificates: \(K_{\mathrm{ScaleDecCost}}^+\), \(K_{\mathrm{PhysCostInf}}^-\), and the scale-weighted energy-cost bridge.
- Status: **implemented in [cluster_c9_energy_cost_bridge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c9_energy_cost_bridge.md)**

### C10. NS3D Type II promotion bridge
Make the promotion
\[
K_{\mathrm{SC}_\lambda}^-
\wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}
\Longrightarrow
K_{\mathrm{SC}_\lambda}^{\sim}
\]
explicit for the declared Navier-Stokes repaired-gauge backend.

- Role: prevent the overclaim that the generic heat-model `UP-TypeII` proof-object applies automatically to 3D Navier-Stokes.
- Difficulty: Low as a certificate-interface theorem; high for the remaining analytic discharge of \(K_{\mathrm{PromotionSound}}^+\).
- Dependencies: C1 route admissibility, C4 `BarrierTypeII` blocked output, C9's distinction between blocked and promoted Type II, and either \(K_{\mathrm{GenericUPTypeIIAdmiss}}^+\) for the formal theorem {prf:ref}`mt-up-type-ii` or an NS-specific replacement theorem.
- Output certificates: \(K_{\mathrm{NSPromPayload}}^+\), \(K_{\mathrm{NS\text{-}UPTypeII}}^+\), \(K_{\mathrm{GenericUPTypeIIAdmiss}}^+\), and the ordered promotion defects \(K_{\mathrm{PromRoute}}^-\), \(K_{\mathrm{BarrierAdmiss}}^-\), \(K_{\mathrm{ScaleRouteStable}}^-\), \(K_{\mathrm{NoBarrierLeak}}^-\), \(K_{\mathrm{PromotionSound}}^-\).
- Status: **implemented in [cluster_c10_ns_up_typeII_promotion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c10_ns_up_typeII_promotion.md)**. C10 proves the pointwise and uniform promotion theorem conditional on the explicit NS3D promotion payload, defines \(K_{\mathrm{GenericUPTypeIIAdmiss}}^+\) as the exact payload that permits invoking the formal `UP-TypeII` theorem, and records that without \(K_{\mathrm{PromotionSound}}^+\) the rigorous output is blocked Type II, not post-UP suppression.

### C11. Generic `UP-TypeII` admissibility payload discharge
Discharge the C10 generic route payload:
\[
K_{\mathrm{GenericUPTypeIIAdmiss}}^+.
\]

- Role: identify exactly what is needed before the formal theorem {prf:ref}`mt-up-type-ii` can be used for an NS3D repaired-gauge branch.
- Difficulty: Medium as a certificate compiler; high for the localized monotonicity witness.
- Dependencies: C1 route, C2 representation, C4 cost bridge, C10 promotion target, dataset \(K_{\mathrm{Auto}}^+\) and \(K_{D_E}^+\), plus \(K_{\mathrm{NSLocMonoTrans}}^+\).
- Output certificates: \(K_{\mathrm{GenericUPTypeIIAdmiss}}^+\), \(K_{\mathrm{GenUPPayload},NS3D}^+\), and ordered defects including \(K_{\mathrm{NSLocMonoTrans}}^-\).
- Status: **implemented in [cluster_c11_generic_up_typeII_payload_discharge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c11_generic_up_typeII_payload_discharge.md)**. C11 proves that the current C-series stack discharges every generic-UP payload component except the genuinely analytic localized monotonicity translation \(K_{\mathrm{NSLocMonoTrans}}^+\). With that witness, C11 emits \(K_{\mathrm{GenericUPTypeIIAdmiss}}^+\), so C10 may apply the formal `UP-TypeII` theorem.

### C12. NS3D localized monotonicity translation
Prove the localized monotonicity witness
\[
K_{\mathrm{NSLocMonoTrans}}^+
\]
needed by C11.

- Role: turn the repaired-gauge local energy identity into the monotonicity input required by {prf:ref}`mt-up-type-ii`.
- Difficulty: High for finite-tail estimates; medium for the corrected-monotonicity compiler.
- Dependencies: C2 repaired-gauge representation, C4 cost convention, Lemma 4 local energy identity in the master note, and finite tail control of the explicit monotonicity-error density.
- Output certificates: \(K_{\mathrm{NSLocMonoTrans}}^+\), \(K_{\mathrm{FiniteMonoErr}}^+\), and subcertificates \(K_{\mathrm{ViscCutErr}}^+\), \(K_{\mathrm{ConvFluxErr}}^+\), \(K_{\mathrm{PressureFluxErr}}^+\), \(K_{\mathrm{CenterDriftErr}}^+\), \(K_{\mathrm{ScaleNegErr}}^+\), \(K_{\mathrm{ScaleCutErr}}^+\).
- Status: **implemented in [cluster_c12_ns_localized_monotonicity_translation.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c12_ns_localized_monotonicity_translation.md)**. C12 proves a corrected monotonicity formula
  \[
  \frac{d}{d\tau}\mathcal E_{R_0}^{\mathrm{corr}}
  +\frac12\tilde{\mathfrak D}_{R_0}\le0
  \]
  from \(K_{\mathrm{FiniteMonoErr}}^+\). It reduces the remaining gap to finite tail integrability of the explicit error density.

### C13. Formal `UP-TypeII` application theorem
Compose C10--C12 into the exact theorem licensing the formal upgrade
{prf:ref}`mt-up-type-ii` for NS3D.

- Role: answer precisely when the generic formal `UP-TypeII` theorem may be applied to a declared NS3D repaired-gauge Type II branch.
- Difficulty: Low after C10--C12; all difficulty is isolated in \(K_{\mathrm{FiniteMonoErr}}^+\).
- Dependencies: \(K_{\mathrm{Auto}}^+\), \(K_{\mathrm{TypeIIRoute}}^+\), \(K_{D_E}^+\), \(K_{\mathrm{RepBridge}}^+\), \(K_{\mathrm{CostBridge}}^+\), \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\), \(K_{\mathrm{NSLocEnergyId}}^+\), and \(K_{\mathrm{FiniteMonoErr}}^+\).
- Output certificate: \(K_{\mathrm{FormalUPTypeIIApp},NS3D}^+\Rightarrow K_{\mathrm{SC}_\lambda}^{\sim}\).
- Status: **implemented in [cluster_c13_formal_up_typeII_application.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c13_formal_up_typeII_application.md)**. C13 proves that the formal theorem {prf:ref}`mt-up-type-ii` applies exactly under the application package. The blocked certificate \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) must actually be emitted and \(K_{\mathrm{NSLocEnergyId}}^+\) must be supplied; after those are present, the remaining analytic obstruction on the current C-series route is \(K_{\mathrm{FiniteMonoErr}}^+\).

### C14. Explicit NS3D `UP-TypeII` checklist
State every certificate needed to apply formal `UP-TypeII` to NS3D.

- Role: provide direct and expanded application packages so there is no hidden payload behind "apply `UP-TypeII`."
- Difficulty: Low as a compiler; the only hard analytic subpayload remains the finite monotonicity-error tail.
- Dependencies: C1--C13.
- Output certificates: \(K_{\mathrm{UPTypeIIReady},NS3D}^{\mathrm{dir}}\), \(K_{\mathrm{UPTypeIIReady},NS3D}^{\mathrm{exp}}\), and the six finite-tail subcertificates for \(K_{\mathrm{FiniteMonoErr}}^+\).
- Status: **implemented in [cluster_c14_explicit_up_typeII_ns3d_checklist.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c14_explicit_up_typeII_ns3d_checklist.md)**. C14 states that formal {prf:ref}`mt-up-type-ii` is licensed exactly under the direct package with an emitted \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\), or under the expanded compact-barrier package that produces the blocked certificate.

### C15. Moving-cutoff monotonicity replacement
Replace the fixed-annulus finite-error certificate by a moving-cutoff
summability package.

- Role: make tightness/no-radiation usable for monotonicity errors by moving the cutoff annulus outward.
- Difficulty: Medium as a compiler; high for the scale-negative drift tail.
- Dependencies: C12, C14, summable moving-annulus estimates, \(K_{\mathrm{MovingCostBridge}}^+\), and \(K_{\mathrm{ScaleNegL^1}}^+\).
- Output certificates: \(K_{\mathrm{MovingFiniteMonoErr}}^+\), \(K_{\mathrm{MoveAnnErr}}^+\), \(K_{\mathrm{SummTightSched}}^+\), \(K_{\mathrm{MovingCostBridge}}^+\), and \(K_{\mathrm{ScaleNegL^1}}^+\).
- Status: **implemented in [cluster_c15_moving_cutoff_monotonicity.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c15_moving_cutoff_monotonicity.md)**. C15 proves corrected monotonicity with a moving cutoff and reduces the replacement for \(K_{\mathrm{FiniteMonoErr}}^+\) to summable annular errors, moving-cost admissibility, and the scale-negative \(L^1\) drift certificate.

### C16. Scale-negative drift dichotomy
Classify the remaining scale-negative drift term in the C15 moving-cutoff
route.

- Role: discharge \(K_{\mathrm{ScaleNegL^1}}^+\) under finite negative
  repaired-gauge drift, or route failure into an explicit
  \(K_{\mathrm{ScaleCollapseDrift}}^-\) survivor.
- Difficulty: Low as a certificate dichotomy; high if one wants to eliminate
  the scale-collapse survivor by PDE rigidity.
- Dependencies: C15 and the master-note scale convention
  \(a=d_\tau\log\lambda\).
- Output certificates: \(K_{\mathrm{NegDriftL^1}}^+\),
  \(K_{\mathrm{CoreL^2Bd}}^+\), \(K_{\mathrm{CoreL^2Floor}}^+\),
  \(K_{\mathrm{ScaleNegL^1}}^+\), and
  \(K_{\mathrm{ScaleCollapseDrift}}^-\).
- Status: **implemented in [cluster_c16_scale_negative_drift_dichotomy.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c16_scale_negative_drift_dichotomy.md)**. C16 proves \(K_{\mathrm{CoreL^2Bd}}^+\wedge K_{\mathrm{NegDriftL^1}}^+\Rightarrow K_{\mathrm{ScaleNegL^1}}^+\), proves the exact finite/infinite alternative for the weighted negative-drift integral, and shows that genuine scale collapse \(\lambda(\tau)\to0\) with a nonvanishing localized \(L^2\) core forces \(K_{\mathrm{ScaleCollapseDrift}}^-\).

### C17. Scale-collapse drift barrier
Register the C16 obstruction as a direct Type II barrier branch.

- Role: prevent the scale-collapse obstruction from remaining merely a failed
  monotonicity-error estimate. If the backend accepts the scale-collapse cost
  or the absolute scale-drift cost, then
  \(K_{\mathrm{ScaleCollapseDrift}}^-\) emits the blocked Type II certificate.
- Difficulty: Low as a cost-bridge theorem; medium/high only for deciding
  whether the declared backend should accept the enlarged cost family.
- Dependencies: C16, C10, C14/C15, and the `BarrierTypeII` cost-registration
  convention.
- Output certificates: \(K_{\mathrm{ScaleCollapseCostBridge}}^+\),
  \(K_{\mathrm{AbsScaleCostBridge}}^+\),
  \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\), and, under NS-valid promotion,
  \(K_{\mathrm{SC}_\lambda}^{\sim}\).
- Status: **implemented in [cluster_c17_scale_collapse_barrier.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c17_scale_collapse_barrier.md)**. C17 proves that \(K_{\mathrm{ScaleCollapseDrift}}^-\) emits \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) once either the scale-collapse cost or absolute scale-drift cost is registered with `BarrierTypeII`, and it composes this with C10's \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) to emit \(K_{\mathrm{SC}_\lambda}^{\sim}\).

### C18. Final NS3D `UP-TypeII` theorem
Assemble the C-series classification stack, S-series scale-collapse and
multibubble closures, rough-core blocker, and C10 promotion bridge.

- Role: state the final Type-II-specific theorem for the declared terminal
  NS3D repaired-gauge backend.
- Difficulty: Low as an assembly theorem; all analytic difficulty is in the
  payloads it imports.
- Dependencies: C1--C8, C10, S3, S4--S8, repaired-gauge representation,
  Caccioppoli regularity, terminal profile completeness, emitted \(K_{\mathrm{ScattBranch}}^-\),
  and exterior-regular discard.
- Output certificates:
  \(K_{\mathrm{UPTypeII},NS3D}^{\mathrm{final}}\),
  \(K_{\mathrm{UPTypeII},NS3D}^{\mathrm{term}}\), and universal
  \(K_{\mathrm{SC}_\lambda}^{\sim}\) for all declared Type II candidates.
- Status: **implemented in [cluster_c18_final_up_typeII_ns3d.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c18_final_up_typeII_ns3d.md)**. C18 proves that \(K_{\mathrm{ClassComplete}}^+\wedge K_{\mathrm{RadBlk}}^+\wedge K_{\mathrm{RoughCoreBlk}}^+\wedge K_{\mathrm{NS\text{-}UPTypeII}}^+\) suppresses every declared Type II candidate via C8, and then expands \(K_{\mathrm{RadBlk}}^+\) and \(K_{\mathrm{RoughCoreBlk}}^+\) using the terminal S4--S8, S3, C7, representation, and Caccioppoli payloads.

### S3. Scale-collapse generalized self-similar reduction
Reduce the scale-collapse drift survivor to an autonomous generalized
self-similar or stationary Liouville problem.

- Role: provide a PDE state-space alternative to the C17 cost-registration
  barrier. S3 does not claim that C16 alone gives a self-similar attractor:
  infinite weighted negative drift may be thin or oscillatory. Instead it
  isolates the extra payloads needed to pass to an autonomous reduced limit,
  extract a stationary profile, and apply parameter-specific rigidity.
- Difficulty: High. The compactness step is local Caccioppoli/Aubin--Lions;
  the hard steps are autonomous modulation, stationary omega-limit extraction,
  and the Liouville theorem for the resulting parameter pair
  \((a_\infty,b_\infty)\).
- Dependencies: C10, C16, \(K_{\mathrm{ModBd}}^+\),
  \(K_{\mathrm{WinH1}}^+\), the repaired-gauge equation, pressure compactness,
  and self-similar or stationary Liouville inputs.
- Output certificates: \(K_{\mathrm{S3ScaleCollapse}}^+\),
  \(K_{\mathrm{S3Compact}}^+\), \(K_{\mathrm{S3ModLim}}^+\),
  \(K_{\mathrm{S3StatLim}}^+\), \(K_{\mathrm{S3Rig}}^+\),
  \(K_{\mathrm{ScaleCollapseBlk}}^+\), and, under NS-valid promotion,
  \(K_{\mathrm{SC}_\lambda}^{\sim}\).
- Status: **implemented in [cluster_s3_scale_collapse_attractor_stratification.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s3_scale_collapse_attractor_stratification.md)**. S3 proves the conditional route
  \[
    K_{\mathrm{S3ScaleCollapse}}^+
    \wedge K_{\mathrm{S3Compact}}^+
    \wedge K_{\mathrm{S3ModLim}}^+
    \wedge K_{\mathrm{S3StatLim}}^+
    \wedge K_{\mathrm{S3Rig}}^+
    \Longrightarrow K_{\mathrm{ScaleCollapseBlk}}^+.
  \]
  The remaining S3 defects are thin/oscillatory drift, failed compactness,
  failed autonomous modulation, failed stationary omega-limit extraction, and
  uncovered generalized self-similar or stationary \(L^3\)-Liouville regimes.

### T8. Compact branch \(\Rightarrow\) global critical precompactness
Show that membership in the compact Type II branch implies
\[
\{V(\tau)\}_{\tau\ge\tau_0}\ \text{is precompact in a global critical norm (e.g. }L^3).
\]

- Role: turn orbit-level structure into global tightness.
- Difficulty: High; stronger than local compactness and profile-level control.
- Dependencies: concentration-compactness framework for the branch.
- Status: **conditional structural form implemented in [cluster_structural_t8_t11.md](/home/guillem/hypostructure/docs/source/type_2/cluster_structural_t8_t11.md); the bridge from Type II branch membership to global \(L^3\)-precompactness remains part of T9--T11/C1--C7**

### T9. No-radiation
Exclude nontrivial scattering remainders in the compact Type II decomposition.

- Role: eliminate compact core plus dispersive tail alternatives.
- Difficulty: High.
- Dependencies: T8.
- Status: **conditional no-radiation criteria implemented in [cluster_structural_t8_t11.md](/home/guillem/hypostructure/docs/source/type_2/cluster_structural_t8_t11.md); deriving the required no-radiation certificate from Navier-Stokes Type II branch data remains structural**

### T10. No-splitting / single-profile saturation
Show a Type II object cannot contain multiple nontrivial profiles.

- Role: prevent multi-bubble escape and support global precompactness.
- Difficulty: High.
- Dependencies: T8, T9, rigidity principles.
- Status: **conditional no-splitting and single-profile saturation criteria implemented in [cluster_structural_t8_t11.md](/home/guillem/hypostructure/docs/source/type_2/cluster_structural_t8_t11.md); deriving the required profile decomposition, mass decoupling, and saturation from Navier-Stokes Type II branch data remains structural**

## Tier 4: Deep rigidity/classification phase

### T11. Full irreducibility of the Type II branch
Show every Type II object on \(K_{C_\mu}^+\) is represented by a single compact gauge-fixed renormalized orbit with no nonvanishing remainder.

- Role: internalize the barrier mechanism at full structural level.
- Difficulty: Very high; encompasses T9–T10 and the compactness upgrade.
- Dependencies: T8–T10.

### T12. Direct zero-cost limit classification
Classify all compact zero-cost renormalized limit objects and exclude nontrivial possibilities.

- Role: bypasses several intermediate compactness/no-splitting stages via direct classification.
- Difficulty: Very high; core PDE object-classification problem.
- Dependencies: T11 and additional rigidity input.

## Current proof strategy

The roadmap should be read as a classification pipeline.

1. C1 routes a declared Type II candidate into the concentration/scale/profile
   backend. Failure here is a backend exhaustion defect.
2. C2--C4 and T1--T7 build the repaired-gauge orbit, normalize the critical
   \(L^3\) mass, prove the compact single-core estimates, and compile Theorem
   A'' into the `BarrierTypeII` certificate.
3. C5 identifies the only finite-cost failures of the compact single-core
   barrier: radiative/noncompact behavior \(K_{L^3\mathrm{Tight}}^-\) or
   rough-core behavior \(K_{\mathrm{WinH1}}^-\).
4. C6 and S4 remove non-bubble radiation. Far radiation is exterior-regular
   under the single-point blowup and exterior-discard payloads; the remaining
   radiative residue is multibubble.
5. S5 removes regular strict same-point cascades, escaping-offset local
   interactions, and all same-point cascades under
   \(K_{\mathrm{SamePointNLDec}}^+\). It leaves only
   \(K_{\mathrm{SamePointNLDec}}^-\), plus the multi-point residue.
6. S6 reduces separated multi-point concentration to single-camera S3
   contradictions under \(K_{\mathrm{MultiPointCamDec}}^+\). It leaves only
   \(K_{\mathrm{MultiPointCamDec}}^-\) as the separated-point multibubble
   interaction defect.
7. C6/T13, C7, T14--T18, and S2 remove or localize rough-core behavior. The
   main closure is Caccioppoli/windowed \(H^1\); S2 is a vorticity-controlled
   subtype result.
8. C16--C17 and S3 route scale-collapse drift. C17 closes it by cost
   registration; S3 closes it only under its compactness, autonomous-modulation,
   stationary-limit, and parameter-rigidity payloads.
9. C10--C14 state the extra promotion payload needed to turn a blocked Type II
   outcome into post-UP suppression.
10. S7 discharges the same-point and separated-point decoupling payloads from
    \(K_{\mathrm{NLProfDec},NS3D}^+\). S8 proves
    \(K_{\mathrm{NLProfDec},NS3D}^+\) in terminal active cameras from terminal
    profile completeness, emitted \(K_{\mathrm{ScattBranch}}^-\), exterior-regular discard,
    repaired-gauge representation, and Caccioppoli regularity. S12 discharges
    the profile-completeness slot from the critical NS profile theorem. After
    these routes are supplied, any remaining multibubble failure is an upstream
    bounded-critical terminal-sequence/profile-theorem defect,
    exterior-discard, representation, Caccioppoli, or S3-rigidity defect, not
    a separate singularity mechanism.

So the current strategy is not to claim all Type II singularities are already
excluded. It is to show that every declared Type II candidate is either a named
technical bridge defect, a blocked non-bubble stratum, an upstream terminal
profile-completeness defect, or an S3 rigidity defect.

## Current residual classification

After the implemented T1--T18, C1--C18, and S2--S12 stack, the remaining
ledger entries split into technical payload defects, promotion outputs, and
rigidity payloads. Multibubble is no longer an independent singularity
configuration inside the terminal-complete backend, because S8 supplies the
terminal nonlinear profile decoupling used by S7.

1. **Backend exhaustion defects.** The declared Type II candidate is not
   certified to enter the C1 concentration/scale/profile route:
   \[
   K_{\mathrm{CmuExtract}}^-,
   \quad
   K_{\mathrm{ScaleRoute}}^-,
   \quad
   K_{\mathrm{ProfComplete}}^-,
   \quad
   K_{\mathrm{LostTypeII}}^-.
   \]
2. **Representation bridge.** Inside the declared backend, C2.R emits
   \(K_{\mathrm{RepBridge}}^+\) for every routed candidate. The only
   outside-contract representation diagnostics are:
   \[
   K_{\mathrm{Chart}}^-,
   \quad
   K_{\mathrm{GaugeSolve}}^-.
   \]
   Pressure representation and final modulation parameters are consequences
   of these two inputs, not independent defects.
   U2a refines the row: pressure reconstruction, modulation coefficients,
   final chart identity, AC gauge regularity, and critical \(L^3\)-invariance
   are automatic once an AC
   chart and AC repaired-gauge path are present. The genuine remaining
   subpayloads are chart extraction, repaired-gauge root solvability, AC root
   selection, and terminal admissibility.
3. **Critical-mass defects.** The represented orbit is not on the positive
   finite critical \(L^3\)-mass branch:
   \[
   K_{L^3\mathrm{Dom}}^-,
   \quad
   K_{L^3\mathrm{Inf}}^-,
   \quad
   K_{L^3\mathrm{Zero}}^-.
   \]
4. **Radiative/noncompact Type II.** The branch fails global critical
   \(L^3\)-tightness. This includes genuine radiation tails, outward escaping
   profiles, multi-profile splitting, cascade behavior, and failure of global
   critical precompactness. S4 discharges far radiation under the single-point
   blowup and exterior-regular discard payloads. S5 removes regular strict
   same-point cascades and closes all same-point cascades under
   \(K_{\mathrm{SamePointNLDec}}^+\). S6 reduces the multibubble part to
   camera-decoupling payloads, S7 reduces those payloads to
   \(K_{\mathrm{NLProfDec},NS3D}^+\), and S8 proves that payload in terminal
   cameras from \(K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+\) and the
   repaired-gauge/Caccioppoli bridge. S12 supplies
   \(K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+\) from
   \(K_{\mathrm{TermCritProfThm},NS3D}^+\). The remaining radiative defects are
   bounded-critical terminal-sequence/profile-theorem failure,
   exterior-discard, representation, Caccioppoli, or S3-rigidity defects.
5. **Rough-core Type II.** The branch is tight but fails local windowed
   \(H^1\) control. After C3 and repaired-gauge nondegeneracy are imported,
   U5a discharges the Caccioppoli regularity input from physical suitability
   and the declared repaired-gauge representation; C2.R supplies pressure
   reconstruction and AC gauge/modulation. The remaining pointwise rough-core
   defects are
   \[
   K_{\mathrm{ModForceBd}}^-,
   \]
   The integrated good-window route refines the force side into
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
   S2 removes the vorticity-controlled subtype:
   \[
   K_{\mathrm{RepBridge}}^+
   \wedge
   K_{L^3\mathrm{Bd}}^+
   \wedge
   K_{\mathrm{VortL^2Win}}^+
   \Longrightarrow
   K_{\mathrm{RoughCoreBlk}}^+.
   \]
   Thus any remaining rough-core branch must include a local vorticity-window
   defect, a modulation-force defect, bounded-modulation failure, or failure
   of physical suitability.
   The actual rough-core closure is still the direct C6/T13 Caccioppoli route;
   S2 is a subtype exclusion and diagnostic reduction, not an autonomous
   vorticity-based proof of rough-core exclusion.
6. **Promotion/monotonicity defects.** The PDE barrier may emit the blocked
   Type II certificate without licensing the formal NS3D `UP-TypeII`
   promotion. The fixed-cutoff promotion-side analytic payload is the finite
   monotonicity-error package \(K_{\mathrm{FiniteMonoErr}}^+\), equivalently
   the six finite-tail subcertificates in C14. The moving-cutoff replacement
   reduces the annular part to \(K_{\mathrm{MoveAnnErr}}^+\) and isolates the
   scale-negative alternative
   \[
   K_{\mathrm{ScaleNegL^1}}^+
   \quad\text{or}\quad
   K_{\mathrm{ScaleCollapseDrift}}^-.
   \]
   C16 proves that finite negative repaired-gauge drift with bounded localized
   \(L^2\) mass gives the positive certificate, while genuine scale collapse
   with a nonvanishing localized \(L^2\) core forces the obstruction. C17
   turns that obstruction into a blocked Type II branch if the backend supplies
   \(K_{\mathrm{ScaleCollapseCostBridge}}^+\) or
   \(K_{\mathrm{AbsScaleCostBridge}}^+\). S3 alternatively reduces the same
   obstruction to autonomous compactness, stationary-limit, and Liouville
   rigidity payloads; otherwise the remaining S3 defects are thin drift,
   failed modulation convergence, failed stationary extraction, or an uncovered
   generalized self-similar/stationary Liouville regime.
7. **Multibubble ledger.** After the technical bridge payloads and closeable
   strata are supplied, S5 removes regular strict same-point cascades, S6
   reduces multibubbles to same-point and separated-point decoupling payloads,
   S7 discharges those payloads from \(K_{\mathrm{NLProfDec},NS3D}^+\), and
   S8 proves \(K_{\mathrm{NLProfDec},NS3D}^+\) for terminal active cameras.
   S12 discharges terminal profile completeness from
   \(K_{\mathrm{TermCritProfThm},NS3D}^+\). A multibubble failure can therefore
   occur only through upstream failure of the bounded-critical
   terminal-sequence/profile-theorem input, \(K_{\mathrm{ExtRegDiscard}}^+\),
   repaired-gauge representation, Caccioppoli regularity, or the S3 rigidity
   payload. The scattering branch has already been removed by
   \(K_{\mathrm{ScattBranch}}^-\) before this node.

Consequently, once \(K_{\mathrm{TechTypeII}}^{\mathrm{blk},+}\),
\(K_{\mathrm{SinglePointBlowup}}^+\),
\(K_{\mathrm{ExtRegDiscard}}^+\), rough-core Caccioppoli closure, and either
C17 scale-collapse cost registration or the S3 compactness/modulation/
stationary-limit/Liouville route are supplied, and once the S5 cascade payloads
\(K_{\mathrm{L^3ProfDec},NS3D}^+\), \(K_{\mathrm{ProfStab},NS3D}^+\),
\(K_{\mathrm{InnerRepBridge}}^+\), and \(K_{\mathrm{S3PertRob}}^+\) are
supplied, and if \(K_{\mathrm{NLProfDec},NS3D}^+\) is supplied, no
multibubble singularity configuration remains. S8 supplies
\(K_{\mathrm{NLProfDec},NS3D}^+\) in terminal active cameras from terminal
profile completeness. S14 routes terminal sequences from the repaired-gauge
orbit, S13 supplies bounded terminal critical sequences from the C3
\(L^3\)-normalization branch, and S12 supplies terminal profile completeness
from the critical NS profile theorem and small-data stability. Thus
multibubble failure is not a separate remaining singularity configuration once
\(K_{\mathrm{RepBridge}}^+\), \(K_{\mathrm{StratCritPacket}}^+\),
\(K_{\mathrm{SmallDataStab}_{L^3}}^+\),
\(K_{\mathrm{CriticalNSProfDecomp}}^+\), and the S8 auxiliary payloads are
supplied. Terminal-camera construction is part of the declared S8 terminal
backend and is no longer an independent hypothesis.

The following are no longer independent survivor buckets: pressure-tail
failure, T1.5 pressure-hypothesis failure under the stated upstream estimates,
local \(L^2\) pressure failure modulo constants, repaired-gauge matrix inverse
failure after \(K_{\mathrm{ModMatrixPayload}}^+\), and singular-weight
convective boundary terms after T18.

## Suggested execution order

The old T/S/C technical stack is implemented through the declared terminal
backend. The next work should target the live downstream stratum-discharge
theorems in this order.

1. U3b: prove \(L^3\)-domain well-definedness on retained represented
   branches, or classify \(K_{L^3\mathrm{Dom}}^-\) as an upstream
   representation defect.
2. U3c: route \(K_{L^3\mathrm{Inf}}^-\) either to an accepted barrier defect
   or to an outside-Type-II classification.
3. U7a: finish the single-core exterior localization statement, using S10's
   positive-distance profile lemma.
4. U4a: prove the first no-radiation/tightness route, preferably
   "compact core plus vanishing remainder" or "profile-completeness plus
   exterior discard implies tightness."
5. U6a: choose between the S3 rigidity route and the C17 cost-acceptance route
   for scale-collapse, then prove the selected payload.
6. U1a: assemble backend entry once U2--U7 make the declared terminal backend
   exhaustive enough for actual Type II branches.
7. U8: replace the imported critical profile theorem by exact citations, or
   add a self-contained proof appendix.
8. U10: assemble the bare-data Type II exclusion theorem.
9. U9: discharge formal NS `UP-TypeII` promotion if the desired conclusion is
   \(K_{\mathrm{SC}_\lambda}^{\sim}\) rather than the PDE-level blocked
   certificate.

## Implemented technical closure cluster

- T1 (pressure-tail control),
- T1.4 (verification of T1.5 hypotheses),
- T1.5 (local \(L^2\) pressure estimate modulo constants),
- T2 (translation-block invertibility),
- T3 (cross-term control),
- T4 (repaired-gauge nondegeneracy),
- T5 (bounded \((a,b)\)),
- T6 (time regularity).

This cluster is implemented in
[cluster_next_easiest_theorems.md](/home/guillem/hypostructure/docs/source/type_2/cluster_next_easiest_theorems.md).

After this cluster, the principal remaining obstacle is the bubble/no-splitting structural program of T10--T12, with S5 removing the regular same-point cascade subcase and T8--T9 supplying the compactness and no-radiation inputs.

## Implemented structural and certificate clusters

- T7 (strong \(L^3_{\mathrm{loc}}\)-compactness upgrade):
  [cluster_t7_L3_compactness_upgrade.md](/home/guillem/hypostructure/docs/source/type_2/cluster_t7_L3_compactness_upgrade.md)
- T7.5 (good-window final barrier closure):
  Theorem A'' in [compact_typeII_master_note_repaired_gauge.md](/home/guillem/hypostructure/docs/source/type_2/compact_typeII_master_note_repaired_gauge.md)
- T7.6 (contrapositive classification):
  Corollary 16 in [compact_typeII_master_note_repaired_gauge.md](/home/guillem/hypostructure/docs/source/type_2/compact_typeII_master_note_repaired_gauge.md)
- T8 (compact branch \(\Rightarrow\) global critical precompactness):
  [cluster_structural_t8_t11.md](/home/guillem/hypostructure/docs/source/type_2/cluster_structural_t8_t11.md)
- T7.7 (hypostructure sieve-certificate integration):
  [hypostructure_sieve_typeII_certificates.md](/home/guillem/hypostructure/docs/source/type_2/hypostructure_sieve_typeII_certificates.md)
- T14--T18 (pointwise and integrated modulation-force reductions):
  [cluster_t14_modulation_force_bound.md](/home/guillem/hypostructure/docs/source/type_2/cluster_t14_modulation_force_bound.md),
  [cluster_t15_t16_integrated_modulation.md](/home/guillem/hypostructure/docs/source/type_2/cluster_t15_t16_integrated_modulation.md),
  [cluster_t18_scale_force_decomposition.md](/home/guillem/hypostructure/docs/source/type_2/cluster_t18_scale_force_decomposition.md)
- S2 (rough-core status and vorticity-controlled subtype exclusion):
  [cluster_s2_rough_core_vorticity_exclusion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s2_rough_core_vorticity_exclusion.md)
- S4 (reduction to multibubble residue):
  [cluster_s4_multibubble_residue_classification.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s4_multibubble_residue_classification.md)
- S5 (regular same-point cascade exclusion):
  [cluster_s5_multiscale_cascade_exclusion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s5_multiscale_cascade_exclusion.md)
- S6 (multibubble camera reduction):
  [cluster_s6_multibubble_camera_reduction.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s6_multibubble_camera_reduction.md)
- S7 (decoupling payload discharge):
  [cluster_s7_decoupling_payload_discharge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s7_decoupling_payload_discharge.md)
- S8 (terminal nonlinear profile decoupling):
  [cluster_s8_terminal_nonlinear_profile_decoupling.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s8_terminal_nonlinear_profile_decoupling.md)
- S9 (scattering branch discharge):
  [cluster_s9_scattering_branch_discharge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s9_scattering_branch_discharge.md)
- S10 (exterior-regular discard):
  [cluster_s10_exterior_regular_discard.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s10_exterior_regular_discard.md)
- S11 (Caccioppoli regularity discharge):
  [cluster_s11_caccioppoli_regularity_discharge.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s11_caccioppoli_regularity_discharge.md)
- C1--C18 plus S3 (declared Type II backend, bridge, classification,
  promotion, formal-application stack, moving monotonicity, scale-drift
  dichotomy, scale-collapse barrier, final `UP-TypeII` assembly, and
  scale-collapse generalized self-similar reduction):
  [cluster_c1_typeII_branch_exhaustion.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c1_typeII_branch_exhaustion.md)
  through
  [cluster_c18_final_up_typeII_ns3d.md](/home/guillem/hypostructure/docs/source/type_2/cluster_c18_final_up_typeII_ns3d.md), plus
  [cluster_s3_scale_collapse_attractor_stratification.md](/home/guillem/hypostructure/docs/source/type_2/cluster_s3_scale_collapse_attractor_stratification.md)
