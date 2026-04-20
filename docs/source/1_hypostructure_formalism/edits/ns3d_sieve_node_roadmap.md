# Navier--Stokes 3D Sieve Node Roadmap

This roadmap maps the existing sieve nodes to the proof architecture for the
three-dimensional incompressible Navier--Stokes equations. Its purpose is to
prepare the node-by-node replacement of the generic PDE language by an
NS3D-specific instantiation.

The roadmap does not yet rewrite the node templates. It records where each
node should obtain its PDE objects, estimates, compactness statements,
rigidity inputs, and exclusion theorems.

## Proof Sources

The NS3D instantiation should be read against the following source documents.

- `docs/source/type_1/unconditional_typeI_chain/paper0_local_concentration_entry_typeII.tex`
  gives the local concentration entry mechanism, Caffarelli--Kohn--Nirenberg
  epsilon regularity input, positive local concentration, and the local
  Type I/Type II dichotomy.
- `docs/source/type_1/unconditional_typeI_chain/paperI_typeI_blowup_limits_residual_criterion.tex`
  gives the Type I blow-up extraction, pressure gauges, ancient suitable
  limits, nontriviality, classical Liouville branches, and the residual
  criterion.
- `docs/source/type_1/unconditional_typeI_chain/paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex`
  gives the uniformly tight ancient dynamics branch, compact hulls,
  invariant measures, stationary limits, endpoint $L^3$ closure, and the
  tight Liouville criterion.
- `docs/source/type_1/unconditional_typeI_chain/paperIII_structured_ancient_branch_exclusions.tex`
  gives structured ancient branch exclusions, including axisymmetric
  consequences, weighted-vorticity Lyapunov rigidity, Burgers-vortex
  rigidity, and localized scalar criteria.
- `docs/source/type_1/unconditional_typeI_chain/residual_branch.tex`
  gives the residual branch: terminal local concentration, noncompact and
  nonconcentrating alternatives, finite separated profile families, atomic
  profile reduction, endpoint Liouville closure, and final residual closure.
- `docs/source/type_2/papers/paper1_compact_repaired_gauge_typeII.tex`
  gives the compact single-core Type II criterion and the zero-dissipation
  compact branch.
- `docs/source/type_2/papers/paper2_renormalized_typeII_repaired_gauges.tex`
  gives the renormalized Type II coordinates, repaired gauges, modulation
  equations, local pressure decomposition, and stability of the localized
  energy inequality.
- `docs/source/type_2/papers/paper3_critical_profile_decompositions_radiation_multibubbles.tex`
  gives active frames, critical profile decompositions, radiation,
  multibubble alternatives, same-point cascade reduction, and terminal
  decoupling.
- `docs/source/type_2/papers/paper4_local_regularity_rough_core_exclusion.tex`
  gives compact-cylinder Caccioppoli estimates and rough-core exclusion.
- `docs/source/type_2/papers/paper5_scale_collapse_alternatives.tex`
  gives scale-collapse alternatives, finite-cost exclusions, autonomous
  windows, and stationary omega-limit stratification.
- `docs/source/type_2/papers/paper6_pde_typeII_exclusion_assembly.tex`
  gives the Type II assembly: representation, single-core exclusion,
  multibubble/cascade exclusion, rough-core reduction, scale-cost exclusion,
  scale stratification, local state-space decomposition, and local Type II
  exclusion.

## Global Reading Order

1. Read `paper0_local_concentration_entry_typeII.tex` for the entry nodes:
   finite energy, local concentration, CKN regularity, positive concentration,
   profile entry, and the local Type I/Type II dichotomy.
2. Read `paperI_typeI_blowup_limits_residual_criterion.tex` for the Type I
   extraction pipeline, the ancient suitable limit, the pressure gauge package,
   nontriviality, classical Liouville classes, and the residual criterion.
3. Read `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex`
   for tight ancient dynamics, compact hulls, invariant measures, stationary
   limits, and endpoint $L^3$ closure.
4. Read `paperIII_structured_ancient_branch_exclusions.tex` for symmetry,
   relative-equilibrium, Lyapunov, and structured ancient branch exclusions.
5. Read `residual_branch.tex` for terminal residual branches, finite separated
   profile families, atomic reduction, and final residual closure.
6. Read the Type II paper series in order for the Type II side of the sieve:
   compact single core, repaired gauges, radiation and multibubbles, rough
   core, scale collapse, and final assembly.

## Node Mapping Table

The table records the first source to read when instantiating each execution
node and the pre-entry interface. A single node may depend on several sources;
the primary source provides the main theorem, while secondary sources provide
definitions, estimates, or compatibility checks.

The estimate, failure, and refinement boxes attached to a node are covered by
that node's row. They should be instantiated as the local estimate to prove,
the NS3D failure mode if the estimate fails, and the refinement or extraction
lemma that restores the forward route.

| Node | NS3D meaning | Primary proof source | Closure obligation for the node |
|---|---|---|---|
| `H0` | NS3D thin interface: equation, domain, solution class, pressure convention, data class, local cylinders, critical quantities, scaling, and target regularity criterion. | `paper0_local_concentration_entry_typeII.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper6_pde_typeII_exclusion_assembly.tex` | Declare the incompressible 3D Navier--Stokes system, suitable weak or Leray--Hopf solution class, local energy inequality, pressure representative, CKN quantities, scaling conventions, and the singular-point criterion that allows entry into `D_E`. |
| `D_E` | Local finite-energy and local-energy-inequality entry for a suitable weak solution or Leray--Hopf solution. | `paper0_local_concentration_entry_typeII.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex` | Verify the local energy class, local energy inequality, admissible pressure, and finite local kinetic energy on the analysis cylinder. Use the no-escape and local energy/dissipation estimates from Paper I when the node is entered from a blow-up sequence. |
| `Rec_N` | Selection of singular times, centers, and scales for the local blow-up analysis. | `paper0_local_concentration_entry_typeII.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex` | Produce a valid local concentration sequence or a Type I blow-up sequence. The output must include centers $x_n$, times $t_n$, scales $r_n$, and cylinders on which the scale-invariant quantities are controlled or positively concentrated. |
| `C_mu` | Certified local concentration or compactness witness. | `paper0_local_concentration_entry_typeII.tex` | Use CKN epsilon regularity and the positive local concentration lemma to decide whether a nontrivial concentration measure/profile is present. If concentration vanishes, close the branch by the no-concentration regularity theorem. If concentration persists, enter the profile audit. |
| `PS0` | Local singularity-entry check for the profile sieve. | `paper0_local_concentration_entry_typeII.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex` | Verify that the selected point is a terminal singular point in the local CKN sense and that the rescaled sequence is admissible for subsequent profile extraction. |
| `PS1` | Positive local concentration check. | `paper0_local_concentration_entry_typeII.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex` | Prove a positive lower bound for a critical local quantity such as $C(r)+D(r)$, or for the local velocity concentration used in the Type I extraction. |
| `PS2` | Center selection and localization check. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper2_renormalized_typeII_repaired_gauges.tex` | Choose centers so that the concentration packet is nontrivial and stable under parabolic rescaling. For Type II nodes, this becomes the preliminary frame and repaired gauge center selection. |
| `PS3` | Scale selection and Type I/Type II rate classification. | `paper0_local_concentration_entry_typeII.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper5_scale_collapse_alternatives.tex` | Decide whether the selected concentration sequence satisfies the Type I rate bound or the local Type II alternative. For Type II, record whether scale collapse, absolute scale drift, or finite-cost behavior remains. |
| `PS4` | Pressure gauge and normalization check. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper2_renormalized_typeII_repaired_gauges.tex` | Construct pressure representatives compatible with the local energy inequality and compactness topology. In the Type II branch, prove the repaired gauge and pressure decomposition statements. |
| `PS5` | Renormalized Navier--Stokes equation check. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper2_renormalized_typeII_repaired_gauges.tex` | Derive the rescaled or renormalized Navier--Stokes system, including the transformed pressure, drift/modulation terms, and localized error terms that must vanish or be controlled. |
| `PS6` | Compactness and limiting profile check. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex`; `paper1_compact_repaired_gauge_typeII.tex`; `paper3_critical_profile_decompositions_radiation_multibubbles.tex` | Extract an ancient suitable limit, a compact single-core Type II profile, or a critical profile decomposition. Record the topology of convergence and the inherited equation. |
| `PS7` | Suitability and admissibility inheritance check. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper2_renormalized_typeII_repaired_gauges.tex`; `paper4_local_regularity_rough_core_exclusion.tex` | Prove that the limiting object still satisfies the correct local energy inequality, pressure relation, and admissibility conditions after rescaling, gauge repair, or localization. |
| `PS8` | Activity and nontriviality check. | `paper0_local_concentration_entry_typeII.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper3_critical_profile_decompositions_radiation_multibubbles.tex` | Exclude the zero profile by persistence of concentration or by the nonzero ancient-limit theorem. In Type II, identify at least one active frame or retained profile. |
| `PS9` | Type I ancient-profile branch. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex`; `paperIII_structured_ancient_branch_exclusions.tex`; `residual_branch.tex` | Route the Type I profile into classical small/stationary/tight/structured/residual alternatives and close each by the corresponding Liouville or residual theorem. |
| `PS10` | Type II local branch. | `paper0_local_concentration_entry_typeII.tex`; `paper6_pde_typeII_exclusion_assembly.tex` | Enter the local Type II state-space decomposition. The node must verify the positive local Type II concentration sequence and select the Type II architecture used by Papers 1--6. |
| `PS11` | Scale-cascade or scale-collapse branch. | `paper3_critical_profile_decompositions_radiation_multibubbles.tex`; `paper5_scale_collapse_alternatives.tex`; `paper6_pde_typeII_exclusion_assembly.tex` | Decide whether unresolved concentration is caused by same-point scale cascades, absolute scale drift, or finite-cost scale collapse. Close by the multibubble/cascade and scale-cost theorems. |
| `PS12` | Stationary ancient-profile branch. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex`; `paperIII_structured_ancient_branch_exclusions.tex` | Identify stationary or asymptotically stationary ancient limits and apply stationary $L^3$, tight stationary-limit, or structured stationary rigidity theorems. |
| `PS13` | Compact-orbit or compact-hull branch. | `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex` | Construct the trajectory hull of the ancient solution, prove compactness in the local smooth topology, and use minimal invariant sets or invariant measures to force stationarity or contradiction. |
| `PS14` | Terminal heteroclinic or terminal residual branch. | `residual_branch.tex`; `paper3_critical_profile_decompositions_radiation_multibubbles.tex`; `paper6_pde_typeII_exclusion_assembly.tex` | Analyze terminal noncompact or nonconcentrating alternatives, terminal decoupling, and virial terminal states. Prove that any retained terminal profile either becomes atomic or is excluded. |
| `PS15` | Uniform tightness branch. | `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex`; `residual_branch.tex` | Verify uniform $L^3$ tightness or its failure. If tightness holds, apply the tight Liouville theorem. If not, pass to radiation, separated-profile, or residual nodes. |
| `PS16` | Radiation or escaping-profile branch. | `paper3_critical_profile_decompositions_radiation_multibubbles.tex`; `residual_branch.tex`; `paper6_pde_typeII_exclusion_assembly.tex` | Detect energy or critical norm escaping every compact concentration frame. Use escaping-frame decay, radiative-stratum discharge, or Type II state decomposition to remove the branch. |
| `PS17` | Rough-core branch. | `paper4_local_regularity_rough_core_exclusion.tex`; `paper6_pde_typeII_exclusion_assembly.tex`; `residual_branch.tex` | Verify whether concentration persists without compact-cylinder control. Apply compact-cylinder Caccioppoli estimates and rough-core exclusion, or route to the residual rough-core alternative. |
| `PS18` | Multicenter or multibubble branch. | `paper3_critical_profile_decompositions_radiation_multibubbles.tex`; `residual_branch.tex`; `paper6_pde_typeII_exclusion_assembly.tex` | Identify multiple separated active cores, bubbles, or retained profiles. Prove decoupling and reduce finite separated families to single-profile or excluded multibubble alternatives. |
| `PS19` | Finite packet branch. | `paper3_critical_profile_decompositions_radiation_multibubbles.tex`; `residual_branch.tex` | Prove finite active count, parameter separation, and packet decoupling. The node should record whether the finite packet is reducible to atomic profiles or excluded by no-multi-profile arguments. |
| `PS20` | Terminal decoupling branch. | `paper3_critical_profile_decompositions_radiation_multibubbles.tex`; `residual_branch.tex` | Establish terminal profile-evolution decoupling. Show that same-point cascades, separated-point profiles, and tail limits have disjoint contributions or are forced into an excluded atomic branch. |
| `PS21` | Small-data or perturbative branch. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper1_compact_repaired_gauge_typeII.tex` | Check whether the limiting profile lies in a small critical class or a zero-dissipation compact branch. Close by small Liouville, epsilon regularity, or zero-dissipation contradiction. |
| `PS22` | Stationary critical-norm branch. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex` | Check whether the ancient profile is stationary with controlled critical norm. Apply the stationary $L^3$ class exclusion, endpoint ancient Liouville theorem, or stationary self-similar rigidity. |
| `PS23` | Symmetry-class branch. | `paperIII_structured_ancient_branch_exclusions.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex` | Determine whether the limiting profile has an exact or limiting symmetry, such as axisymmetry or a preserved geometric structure. Apply the relevant structured Liouville consequence. |
| `PS24` | Relative equilibrium or coherent-structure branch. | `paperIII_structured_ancient_branch_exclusions.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex` | Check for traveling, rotating, self-similar, Burgers-vortex-type, or other coherent structures after normalization. Use structured rigidity or no-generic-symmetry results to close the branch. |
| `PS25` | Degenerate structured direction branch. | `paperIII_structured_ancient_branch_exclusions.tex`; `residual_branch.tex` | Detect a nontrivial degenerate direction in the structured ancient dynamics that is not closed by small, stationary, tight, or symmetry reductions. The node should point to weighted-vorticity, scalar-flattening, or residual closure lemmas depending on the detected defect. |
| `PS26` | Symmetry action and normalization branch. | `paperIII_structured_ancient_branch_exclusions.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex` | Verify that the selected normalization is compatible with the symmetry action and does not hide a translation, dilation, rotation, or pressure-gauge defect. |
| `PS27` | Symmetry-breaking stability branch. | `paperIII_structured_ancient_branch_exclusions.tex` | Prove that small departures from the structured class either decay by a Lyapunov or scalar criterion, or produce a quantified instability routed to the residual branch. |
| `PS28` | Transition-action or finite-cost branch. | `paper5_scale_collapse_alternatives.tex`; `paper6_pde_typeII_exclusion_assembly.tex` | Decide whether scale transitions have finite or infinite cost in the modulated Type II dynamics. Apply the cost exclusion or autonomous-window reduction. |
| `PS29` | Lyapunov, monotonicity, or statistical-rigidity branch. | `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex`; `paperIII_structured_ancient_branch_exclusions.tex`; `paper5_scale_collapse_alternatives.tex` | Identify a monotone, coercive, statistical, or virial functional that forces equilibrium, stationarity, or contradiction along the terminal dynamics. |
| `PS30` | Defect audit branch. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex`; `paper2_renormalized_typeII_repaired_gauges.tex`; `paper4_local_regularity_rough_core_exclusion.tex`; `paper3_critical_profile_decompositions_radiation_multibubbles.tex`; `residual_branch.tex` | Check for measure defects, pressure defects, Reynolds or stress defects, boundary/cutoff defects, frequency defects, and unknown residual defects. Each defect must be either absorbed by a local estimate, repaired by gauge/compactness/decoupling, or routed as an unresolved residual obstruction. |
| `PS31` | Endpoint hypothesis verification. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex`; `paperIII_structured_ancient_branch_exclusions.tex`; `paper6_pde_typeII_exclusion_assembly.tex`; `residual_branch.tex` | Verify the exact assumptions of the endpoint theorem selected by the branch: CKN regularity, Type I Liouville, tight ancient Liouville, structured rigidity, Type II state-space exclusion, or residual closure. |
| `PS32` | Endpoint exclusion theorem application. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex`; `paperIII_structured_ancient_branch_exclusions.tex`; `paper6_pde_typeII_exclusion_assembly.tex`; `residual_branch.tex` | Apply the endpoint theorem and record the contradiction: zero profile against positive concentration, regularity against singularity, stationarity against nonzero normalized dynamics, or excluded terminal Type II state. |
| `PS33` | Realization or admissible counterexample check. | `residual_branch.tex`; `paper6_pde_typeII_exclusion_assembly.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex` | Decide whether the remaining branch corresponds to an admissible NS3D singular profile not covered by the current exclusions. If such a branch is not realizable, close it; if realizable, record the precise missing theorem. |
| `PS34` | Residual complement branch. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `residual_branch.tex`; `paper6_pde_typeII_exclusion_assembly.tex` | Collect all branches not closed by the named compact, tight, structured, Type II, or perturbative exclusions. Apply the residual closure theorem, terminal local concentration theorem, or local Type II state-space decomposition. |
| `PS35` | Case-decomposition completeness check. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper6_pde_typeII_exclusion_assembly.tex`; `residual_branch.tex` | Prove that the union of Type I, Type II, compact, radiation, rough-core, multibubble, scale-collapse, terminal, and residual alternatives exhausts every admissible profile branch. |
| `Bound_partial` | Boundary or physical-domain compatibility check. | `paper0_local_concentration_entry_typeII.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper4_local_regularity_rough_core_exclusion.tex` | For the whole-space or interior local NS3D problem this node should record the verified no-physical-boundary conclusion. If boundary or artificial cutoff boundaries are present, verify trace, flux, and cutoff errors through local energy and Caccioppoli estimates. |
| `Bound_B` | Forcing, lower-order, or cutoff-source compatibility check. | `paper2_renormalized_typeII_repaired_gauges.tex`; `paper4_local_regularity_rough_core_exclusion.tex`; `paper5_scale_collapse_alternatives.tex` | For unforced NS3D, physical forcing is absent. The active obligation is to control localization, modulation, pressure, and cutoff-source terms introduced by renormalization or compact-cylinder estimates. |
| `Bound_Sigma` | Sufficiency of input data and selected analysis objects. | `paper0_local_concentration_entry_typeII.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper6_pde_typeII_exclusion_assembly.tex` | Verify that the solution class, singular point, selected cylinders, pressure representatives, scales, centers, and compactness topology are sufficient to enter the final exclusion pipeline. |
| `GC_T` | Global compatibility of the local contradiction with the target regularity statement. | `paper0_local_concentration_entry_typeII.tex`; `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper6_pde_typeII_exclusion_assembly.tex` | Ensure that excluding every local singular profile implies the desired local regularity conclusion. This node aligns the profile-level contradiction with the CKN criterion and the Type I/Type II dichotomy. |
| `FinalExcl` | Final NS3D singularity exclusion. | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex`; `paperIII_structured_ancient_branch_exclusions.tex`; `residual_branch.tex`; `paper6_pde_typeII_exclusion_assembly.tex` | Combine Type I closure, tight/structured ancient exclusions, residual closure, and local Type II exclusion. The final output is that no admissible local singular profile remains. |

## Local Case-Analysis Aliases

The current document also contains local case-analysis diagrams. Those diagrams
are not separate execution-node templates unless their atomic checks have an
assigned first-class node number. For the NS3D instantiation, use the following
aliases to read those diagrams against the numbered execution nodes.

| Local diagram atom | First-class node or node group | NS3D proof source |
|---|---|---|
| Type I scale law | `PS9` | `paperI_typeI_blowup_limits_residual_criterion.tex` |
| Type II scale law | `PS10` | `paper0_local_concentration_entry_typeII.tex`; `paper6_pde_typeII_exclusion_assembly.tex` |
| Scale cascade | `PS11` | `paper3_critical_profile_decompositions_radiation_multibubbles.tex`; `paper5_scale_collapse_alternatives.tex` |
| Scale residual | `PS34` and `PS35` | `paperI_typeI_blowup_limits_residual_criterion.tex`; `residual_branch.tex`; `paper6_pde_typeII_exclusion_assembly.tex` |
| Stationary orbit | `PS12` and `PS22` | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex` |
| Compact orbit or hull | `PS13` | `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex` |
| Terminal orbit | `PS14` | `residual_branch.tex`; `paper3_critical_profile_decompositions_radiation_multibubbles.tex`; `paper6_pde_typeII_exclusion_assembly.tex` |
| Tight localization | `PS15` | `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex` |
| Radiation or exterior escape | `PS16` | `paper3_critical_profile_decompositions_radiation_multibubbles.tex`; `residual_branch.tex` |
| Rough localization core | `PS17` | `paper4_local_regularity_rough_core_exclusion.tex`; `paper6_pde_typeII_exclusion_assembly.tex` |
| Multicenter packet | `PS18` | `paper3_critical_profile_decompositions_radiation_multibubbles.tex`; `residual_branch.tex` |
| Finite packet | `PS19` | `paper3_critical_profile_decompositions_radiation_multibubbles.tex`; `residual_branch.tex` |
| Terminal decoupling | `PS20` | `paper3_critical_profile_decompositions_radiation_multibubbles.tex`; `residual_branch.tex` |
| Small branch | `PS21` | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper1_compact_repaired_gauge_typeII.tex` |
| Symmetry branch | `PS23` and `PS26` | `paperIII_structured_ancient_branch_exclusions.tex` |
| Relative-equilibrium branch | `PS24` | `paperIII_structured_ancient_branch_exclusions.tex` |
| Degenerate structured branch | `PS25` | `paperIII_structured_ancient_branch_exclusions.tex`; `residual_branch.tex` |
| Symmetry-breaking branch | `PS27` | `paperIII_structured_ancient_branch_exclusions.tex` |
| Finite transition action | `PS28` | `paper5_scale_collapse_alternatives.tex` |
| Lyapunov or statistical rigidity | `PS29` | `paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex`; `paperIII_structured_ancient_branch_exclusions.tex` |
| Measure defect | `PS30` | `paper0_local_concentration_entry_typeII.tex`; `residual_branch.tex` |
| Stress or Reynolds defect | `PS30` | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper3_critical_profile_decompositions_radiation_multibubbles.tex` |
| Pressure or multiplier defect | `PS30` | `paperI_typeI_blowup_limits_residual_criterion.tex`; `paper2_renormalized_typeII_repaired_gauges.tex`; `paper4_local_regularity_rough_core_exclusion.tex` |
| Boundary, cutoff, or trace defect | `PS30`, `Bound_partial`, and `Bound_B` | `paper2_renormalized_typeII_repaired_gauges.tex`; `paper4_local_regularity_rough_core_exclusion.tex` |
| Frequency or scale-frequency defect | `PS30` and `PS11` | `paper3_critical_profile_decompositions_radiation_multibubbles.tex`; `paper5_scale_collapse_alternatives.tex` |
| Unknown defect channel | `PS30` and `PS34` | `residual_branch.tex`; `paper6_pde_typeII_exclusion_assembly.tex` |
| Endpoint hypothesis match | `PS31` | all endpoint sources listed for `PS31` |
| Endpoint exclusion | `PS32` | all endpoint sources listed for `PS32` |
| Realization or attainability | `PS33` | `residual_branch.tex`; `paper6_pde_typeII_exclusion_assembly.tex` |
| Residual complement and case completion | `PS34` and `PS35` | `paperI_typeI_blowup_limits_residual_criterion.tex`; `residual_branch.tex`; `paper6_pde_typeII_exclusion_assembly.tex` |

## Branch-Level Architecture

### Entry And Dichotomy

The entry branch is controlled by `paper0_local_concentration_entry_typeII.tex`.
Nodes `D_E`, `Rec_N`, `C_mu`, and `PS0`--`PS3` should be instantiated using
suitable weak solutions, local energy inequalities, CKN scale-invariant
quantities, positive local concentration, and the local Type I/Type II
dichotomy.

The essential local implication is:

$$
\text{terminal singularity}
\quad\Longrightarrow\quad
\text{positive local concentration}
\quad\Longrightarrow\quad
\text{Type I branch or Type II branch}.
$$

### Type I Branch

The Type I branch begins at `PS9` and is mainly governed by
`paperI_typeI_blowup_limits_residual_criterion.tex`. The extracted ancient
solution must be suitable, nonzero, centered, and Type I bounded. The branch
then routes through small, stationary, tight, structured, or residual
alternatives.

The tight branch is closed through
`paperII_uniformly_tight_ancient_dynamics_equilibrium_rigidity.tex`. The
structured branch is closed through
`paperIII_structured_ancient_branch_exclusions.tex`. The remaining alternatives
are handled by `residual_branch.tex`.

### Type II Branch

The Type II branch begins at `PS10`. It is assembled by the Type II paper
series:

- Paper 1 closes the compact single-core branch.
- Paper 2 supplies repaired gauges, modulation, pressure decomposition, and
  localized suitability.
- Paper 3 handles radiation, critical profile decompositions, multibubbles,
  same-point cascades, and terminal decoupling.
- Paper 4 excludes rough-core behavior through compact-cylinder Caccioppoli
  estimates.
- Paper 5 handles scale collapse, finite-cost alternatives, and autonomous
  terminal windows.
- Paper 6 assembles these alternatives into the local Type II exclusion.

### Residual Branch

The residual branch collects any profile not closed by the compact, tight,
structured, perturbative, Type II, or explicit endpoint exclusions. It should
be instantiated from `residual_branch.tex`, especially for terminal local
concentration, noncompact/nonconcentrating alternatives, finite separated
profile families, atomic profile reduction, and final residual closure.

### Boundary And Source Nodes

For the whole-space or interior local NS3D problem, `Bound_partial` and
`Bound_B` are usually not physical boundary nodes. They should be interpreted
as compatibility checks for localization, cutoffs, pressure gauges, artificial
compact-cylinder boundaries, and modulation errors. If no physical boundary or
external forcing is present, the node should record a verified scope conclusion and pass
forward, while recording the cutoff and pressure estimates needed by later
nodes.

## Audit Questions For Node Instantiation

When each generic node template is replaced by an NS3D-specific template, the
following questions should be answered explicitly.

1. What exact NS3D object enters the node: suitable weak solution, Leray--Hopf
   solution, rescaled sequence, ancient suitable limit, Type II profile,
   active frame, terminal profile family, or residual profile?
2. Which local cylinder, scale, center, and time interval are used?
3. Which pressure representative or gauge has been fixed?
4. Which scale-invariant quantities are known to be bounded, small, positive,
   tight, or decoupled?
5. Which theorem from the mapped source proves the check, the local estimate,
   the compactness step, the defect absorption, or the exclusion?
6. What contradiction is produced: CKN regularity, zero profile versus positive
   concentration, violation of a Liouville theorem, scale-cost exclusion,
   rough-core exclusion, multibubble exclusion, or residual closure?
