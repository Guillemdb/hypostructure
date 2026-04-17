# ArXiv packaging plan: traditional PDE papers first, certificate formalization last

This document explains how to package the Type II exclusion proof in
`docs/source/type_2` as a sequence of traditional PDE papers.  The guiding
principle is:

```{math}
\text{Papers I--V: PDE theorems with explicit analytic assumptions}
\qquad
\text{Paper VI: certificate formalization and assembly}.
```

The first five papers should avoid certificate language as much as possible.
They should be written in standard PDE style: hypotheses, estimates,
compactness statements, profile decompositions, rigidity alternatives, and
contradictions.  The final paper then translates those PDE theorems into the
Hypostructure/certificate DAG and proves that the assumptions match the formal
nodes used by the Type II sieve.

This split makes the project reviewable by PDE referees before asking them to
accept the certificate framework.  The framework becomes a bookkeeping and
assembly layer, not the language in which the analytic estimates are proved.

## Executive structure

The recommended paper sequence is:

1. **Paper I: Compact repaired-gauge Type II barrier.**
2. **Paper II: Renormalized Type II limits and repaired-gauge representation.**
3. **Paper III: Terminal profile decompositions and exclusion of radiation and multibubbles.**
4. **Paper IV: Local regularity and exclusion of rough-core Type II behavior.**
5. **Paper V: Scale-collapse alternatives and self-similar rigidity.**
6. **Paper VI: Certificate formalization and final Type II assembly.**

Papers I--V should be readable as standalone PDE papers.  Paper VI should be
readable as a formalization paper that imports the PDE results and shows how
they compose.

## Common PDE setup for Papers I--V

Each PDE paper should use the same core setup.

Let `u` be a suitable weak solution of the 3D incompressible Navier-Stokes
equations on a spacetime cylinder ending at a candidate singular point
`(x_*,T_*)`:

```{math}
\partial_t u + u\cdot\nabla u + \nabla p = \Delta u,
\qquad
\nabla\cdot u=0.
```

Let a Type II blowup scenario mean that the natural Type I scale-invariant
bounds are not the active mechanism, and that one studies rescaled flows around
`(x_*,T_*)` using scales `lambda_n -> 0`, centers `x_n -> x_*`, and times
`t_n -> T_*`.

The rescaled fields are

```{math}
u_n(y,s)=\lambda_n u(x_n+\lambda_n y, t_n+\lambda_n^2 s),
\qquad
q_n(y,s)=\lambda_n^2 p(x_n+\lambda_n y, t_n+\lambda_n^2 s).
```

The repaired-gauge renormalized variable is denoted by `V(y,tau)`.  The exact
chart may include translation, scale, and time reparametrization parameters.
For the PDE papers, the chart should be stated directly as an analytic
hypothesis or construction, not as a certificate.

The common critical space is `L^3(R^3)`, with local energy and pressure spaces
specified cylinder by cylinder.

## Paper I: Compact repaired-gauge Type II barrier

### Proposed title

**A repaired-gauge compactness barrier for Type II Navier-Stokes blowup**

### PDE purpose

This paper proves the core analytic contradiction.  It assumes one already has
a repaired-gauge renormalized orbit with compactness/tightness and local energy
control.  Under those assumptions, finite total renormalization cost is
impossible.

This is the analytic heart of the project.  It should be entirely traditional:
gauge identities, pressure estimates, local compactness, low-dissipation
rigidity, contradiction.

### Source material

Primary files:

- [compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md)
- [required_new_scale_gauge_theorems.md](required_new_scale_gauge_theorems.md)
- [cluster_next_easiest_theorems.md](cluster_next_easiest_theorems.md)
- [cluster_t7_L3_compactness_upgrade.md](cluster_t7_L3_compactness_upgrade.md)
- [abstract_and_ns_theorems.md](abstract_and_ns_theorems.md)

Supporting files:

- [cluster_t14_modulation_force_bound.md](cluster_t14_modulation_force_bound.md)
- [cluster_c7_scale_force_bound.md](cluster_c7_scale_force_bound.md)
- [cluster_t15_t16_integrated_modulation.md](cluster_t15_t16_integrated_modulation.md)
- [cluster_t18_scale_force_decomposition.md](cluster_t18_scale_force_decomposition.md)

### Main PDE assumptions

State the theorem for a repaired-gauge renormalized solution `V` satisfying:

1. `V` solves the repaired-gauge renormalized Navier-Stokes equation on
   `[tau_0,infty)`.
2. `V` is divergence-free and locally suitable in renormalized cylinders.
3. Positive finite critical mass:

```{math}
0<\eta\le \|V(\tau)\|_{L^3(\mathbb R^3)}\le M<\infty
\qquad\text{for all }\tau\ge\tau_0.
```

4. Uniform critical tightness:

```{math}
\forall\varepsilon>0\ \exists R_\varepsilon<\infty:
\sup_{\tau\ge\tau_0}\int_{|y|>R_\varepsilon}|V(y,\tau)|^3dy<\varepsilon.
```

5. Local windowed energy control:

```{math}
\sup_{\tau\ge\tau_0}\int_{\tau}^{\tau+1}\int_{B_R}|\nabla V|^2dy d\sigma<\infty
\qquad\text{for every }R<\infty.
```

6. Pressure reconstruction with the standard Calderon-Zygmund local/tail split.
7. Repaired-gauge nondegeneracy and bounded modulation parameters.
8. Finite renormalization cost:

```{math}
\int_{\tau_0}^{\infty}\mathfrak D_{R_0}(V(\tau))d\tau<\infty
```

for the chosen localized dissipation/cost functional.

### Main theorem

```{prf:theorem} Compact repaired-gauge Type II barrier
Under the assumptions above, no such nonzero compact repaired-gauge
renormalized orbit can have finite total renormalization cost.  Equivalently,
finite cost forces a low-dissipation sequence, the compactness hypotheses
produce a nonzero compact limit, and the low-dissipation rigidity theorem
forces that limit to vanish or become forbidden by the gauge normalization.
```

A sharper theorem can be stated as a contrapositive:

```{math}
\text{nonzero compact repaired-gauge Type II orbit}
\quad\Longrightarrow\quad
\int_{\tau_0}^{\infty}\mathfrak D_{R_0}(V(\tau))d\tau=\infty.
```

### Internal sections

1. Introduction and statement of the compact barrier.
2. Repaired scale-translation gauge.
3. Renormalized Navier-Stokes equation.
4. Pressure reconstruction and tail estimates.
5. Local pressure estimates modulo constants.
6. Repaired-gauge modulation matrix and nondegeneracy.
7. Bounded modulation parameters.
8. Local time regularity and Aubin-Lions compactness.
9. Good-window compactness avoiding endpoint trace compactness.
10. Low-cost sample extraction.
11. Low-dissipation rigidity.
12. Barrier contradiction.

### What this paper proves

It proves the compact finite-cost contradiction under explicit PDE hypotheses.
It does not prove that an arbitrary Type II solution satisfies those hypotheses.
That is handled by Papers II--V.

## Paper II: Renormalized Type II limits and repaired-gauge representation

### Proposed title

**Renormalized Type II limits and repaired gauges for Navier-Stokes blowup**

### PDE purpose

This paper constructs the analytic objects assumed in Paper I from Type II
blowup sequences, as far as the construction is local and representation-based.
It should avoid certificates and instead state concrete extraction and
representation theorems.

### Source material

Primary files:

- [ns3d_repaired_gauge_backend_contract.md](ns3d_repaired_gauge_backend_contract.md)
- [cluster_c1_typeII_branch_exhaustion.md](cluster_c1_typeII_branch_exhaustion.md)
- [cluster_c2_representation_bridge.md](cluster_c2_representation_bridge.md)
- [cluster_c2_repbridge_payload_discharge.md](cluster_c2_repbridge_payload_discharge.md)
- [cluster_u2a_repaired_gauge_representation_pieces.md](cluster_u2a_repaired_gauge_representation_pieces.md)
- [cluster_c3_l3_normalization_bridge.md](cluster_c3_l3_normalization_bridge.md)
- [cluster_u3a_zero_critical_mass_exclusion.md](cluster_u3a_zero_critical_mass_exclusion.md)
- [cluster_u3b_critical_mass_completion.md](cluster_u3b_critical_mass_completion.md)
- [cluster_c4_cost_bridge.md](cluster_c4_cost_bridge.md)
- [cluster_c5_two_bucket_classification.md](cluster_c5_two_bucket_classification.md)

### Main PDE assumptions

Work with a suitable weak solution `u` and a candidate singular point
`(x_*,T_*)`.  Assume:

1. There exists a Type II blowup sequence `(x_n,t_n,lambda_n)` with
   `lambda_n -> 0`.
2. The rescaled sequence is bounded in the critical topology needed for profile
   extraction, usually `L^3` or a closely related critical space.
3. The sequence has nontrivial critical concentration:

```{math}
\liminf_{n\to\infty}\|u_n(\cdot,0)\|_{L^3(B_R)}>0
```

for some fixed `R`, after choosing the blowup camera.

4. The repaired gauge equations are solvable along the extracted branch, with
   absolutely continuous scale and translation parameters.
5. The pressure can be pulled back through the chart, modulo harmless constants.
6. The chosen renormalization cost agrees with the localized PDE cost used in
   Paper I.

### Main theorem A: repaired-gauge representation

```{prf:theorem} Repaired-gauge representation of a Type II branch
Let `u` be a suitable weak Navier-Stokes solution and let a Type II blowup
sequence satisfy the extraction, nontriviality, and gauge-solvability
assumptions above.  Then a subsequence admits a repaired-gauge renormalized
representation `V(y,tau)` satisfying the renormalized Navier-Stokes equation,
pressure reconstruction modulo constants, and the modulation coefficient
identities used in Paper I.
```

### Main theorem B: stratified critical mass

```{prf:theorem} Positive finite critical mass on retained terminal strata
Suppose the terminal state space of the extracted branch is partitioned into
scattering, exterior, radiative, rough-core, multibubble, and retained compact
profile strata.  Assume retained active strata are represented by terminal
profiles in `L^3_sigma(R^3)` and that active profiles below the small-data
threshold are removed.  Then every retained active stratum has positive finite
critical `L^3` mass.  If the terminal sequence is `L^3`-bounded, the number of
active retained strata is finite and each active mass is bounded above and
below.
```

This theorem is the traditional PDE version of U3b.  It should be stated in
terms of profiles and norms, not certificates.

### Main theorem C: two-bucket reduction

```{prf:theorem} Two-bucket reduction for represented finite-cost branches
Let a represented finite-cost Type II branch satisfy the representation,
positive finite critical mass, and cost-compatibility hypotheses.  If the
compact barrier of Paper I does not already rule out the branch, then at least
one of the following fails:

1. uniform critical tightness;
2. uniform local windowed `H^1` control.

Thus every represented finite-cost survivor is either radiative/noncompact or
rough-core.
```

### Internal sections

1. Type II blowup sequences and rescaled solutions.
2. Critical concentration extraction.
3. Repaired scale-translation gauge.
4. Absolutely continuous gauge paths.
5. Pressure pullback.
6. Modulation coefficient identities.
7. Critical norm normalization versus positive finite mass.
8. Terminal state-space partition.
9. Positive finite mass for retained active strata.
10. Cost compatibility with the compact barrier.
11. Two-bucket reduction.

### What this paper proves

It proves that, after explicit extraction and gauge assumptions, a Type II
branch can be put into the analytic form required by Paper I, and that failure
of the compact barrier reduces to radiation/tightness failure or rough-core
`H^1` failure.

### What this paper does not prove

It does not prove no radiation, no multibubble, no rough core, or no
scale-collapse.  Those are Papers III--V.

## Paper III: Terminal profile decompositions, radiation, and multibubbles

### Proposed title

**Terminal profile decompositions and the radiative/multibubble alternatives
for Type II Navier-Stokes blowup**

### PDE purpose

This paper handles loss of critical tightness and multibubble splitting.  It
should be written as a concentration-compactness/profile-decomposition paper.
The main point is to prove that radiative and multibubble branches can be
eliminated under standard terminal profile hypotheses: small-data scattering,
exterior regularity, nonlinear profile decomposition, and decoupling.

### Source material

Primary files:

- [cluster_c6_radiative_noncompact_exclusion.md](cluster_c6_radiative_noncompact_exclusion.md)
- [cluster_s4_multibubble_residue_classification.md](cluster_s4_multibubble_residue_classification.md)
- [cluster_s5_multiscale_cascade_exclusion.md](cluster_s5_multiscale_cascade_exclusion.md)
- [cluster_s6_multibubble_camera_reduction.md](cluster_s6_multibubble_camera_reduction.md)
- [cluster_s7_decoupling_payload_discharge.md](cluster_s7_decoupling_payload_discharge.md)
- [cluster_s8_terminal_nonlinear_profile_decoupling.md](cluster_s8_terminal_nonlinear_profile_decoupling.md)
- [cluster_s9_scattering_branch_discharge.md](cluster_s9_scattering_branch_discharge.md)
- [cluster_s10_exterior_regular_discard.md](cluster_s10_exterior_regular_discard.md)
- [cluster_s12_terminal_profile_completeness.md](cluster_s12_terminal_profile_completeness.md)
- [cluster_s13_bounded_critical_terminal_sequences.md](cluster_s13_bounded_critical_terminal_sequences.md)
- [cluster_s14_terminal_sequence_routing.md](cluster_s14_terminal_sequence_routing.md)

### Main PDE assumptions

Let `(u_n)` be a local terminal compactness sequence in `L^3_sigma(R^3)` arising
from the represented Type II branch.  Assume:

1. A local suitable compactness theorem exists for `(u_n)`.
2. The decomposition has asymptotic `L^3` decoupling for active profiles and
   remainders.
3. Profiles with `L^3` norm below the Kato small-data threshold scatter and do
   not contribute to the singular core.
4. Profiles whose physical support remains a positive distance from the
   singular point are regular exterior profiles and can be discarded from the
   core analysis.
5. The nonlinear profile evolution/stability theorem applies to the finite
   active packet.
6. Terminal cameras are compatible with the repaired-gauge representation from
   Paper II.

### Main theorem A: small-data and exterior profile removal

```{prf:theorem} Removal of scattering and exterior terminal profiles
Under the assumptions above, small critical profiles generate global
perturbative Kato solutions and cannot carry the Type II singular core.
Profiles remaining exterior to the singular point are regular on a neighborhood
of the core and can be discarded from the core blowup ledger.
```

### Main theorem B: no radiative critical tail

```{prf:theorem} Radiative tail exclusion under terminal profile decoupling
Assume the terminal critical sequence has a nonlinear profile decomposition
with vanishing critical remainder and no active secondary profile escaping the
core.  Then the represented branch is uniformly critically tight.  Therefore
loss of critical tightness is possible only through failure of one of the
profile-decomposition, decoupling, small-data, exterior-discard, or terminal
camera assumptions.
```

### Main theorem C: multibubble exclusion

```{prf:theorem} Multibubble exclusion under nonlinear profile decoupling
Assume active terminal bubbles satisfy asymptotic `L^3` decoupling and nonlinear
profile stability in their terminal cameras.  Then separated-point multibubble
configurations and same-point finite cascades cannot persist as independent
Type II core mechanisms.  Same-point cascades are either regular perturbative
outer bubbles or violate the nonlinear no-splitting/decoupling assumptions.
```

### Internal sections

1. Terminal critical sequences.
2. Linear profile decomposition in `L^3`.
3. Nonlinear Navier-Stokes profiles.
4. Small-data Kato scattering profiles.
5. Exterior regular profiles.
6. Decoupling of `L^3` mass.
7. Radiative tails and tightness.
8. Same-point multiscale cascades.
9. Separated-point multipoint bubbles.
10. Terminal cameras and nonlinear stability.
11. Radiative/multibubble exclusion theorem.

### What this paper proves

It proves that radiation and multibubble behavior are not remaining Type II
mechanisms once the terminal profile decomposition and decoupling assumptions
hold.

### What this paper does not prove

It does not prove the rough-core local `H^1` estimate or scale-collapse
rigidity.

## Paper IV: Rough-core exclusion by local regularity

### Proposed title

**Local Caccioppoli regularity and rough-core exclusion for Type II
Navier-Stokes blowup**

### PDE purpose

This paper handles the second bucket from Paper II: a branch that is critically
tight but may retain persistent local high-frequency activity or gradient
concentration in a bounded renormalized core.  The goal is to prove that such a
rough core is excluded once local energy, pressure, bounded critical mass, and
modulation-force assumptions are present.

### Source material

Primary files:

- [cluster_c6_windowed_h1_bridge.md](cluster_c6_windowed_h1_bridge.md)
- [cluster_c7_win_h1_certificate_composition.md](cluster_c7_win_h1_certificate_composition.md)
- [cluster_c7_l3bd_defect_discharge.md](cluster_c7_l3bd_defect_discharge.md)
- [cluster_c7_modmatrix_inverse_discharge.md](cluster_c7_modmatrix_inverse_discharge.md)
- [cluster_s11_caccioppoli_regularity_discharge.md](cluster_s11_caccioppoli_regularity_discharge.md)
- [cluster_u5a_bare_data_caccioppoli.md](cluster_u5a_bare_data_caccioppoli.md)
- [cluster_s2_rough_core_vorticity_exclusion.md](cluster_s2_rough_core_vorticity_exclusion.md)
- [cluster_t14_modulation_force_bound.md](cluster_t14_modulation_force_bound.md)
- [cluster_c7_scale_force_bound.md](cluster_c7_scale_force_bound.md)
- [cluster_t15_t16_integrated_modulation.md](cluster_t15_t16_integrated_modulation.md)
- [cluster_t18_scale_force_decomposition.md](cluster_t18_scale_force_decomposition.md)

### Main PDE assumptions

Let `V` be a represented repaired-gauge branch from Paper II.  Assume:

1. `V` is suitable on every compact renormalized cylinder.
2. Local pressure reconstruction holds.
3. The retained active terminal packet has bounded critical `L^3` mass.
4. The repaired-gauge modulation matrix is invertible with controlled inverse.
5. The modulation forcing terms are bounded either pointwise or in the
   integrated good-window sense.
6. The local energy inequality can be tested with compactly supported
   renormalized cutoffs.

### Main theorem A: compact-cylinder Caccioppoli estimate

```{prf:theorem} Renormalized Caccioppoli estimate
Under suitability, pressure reconstruction, and compact-cylinder localization,
`V` satisfies the local Caccioppoli estimate

\[
\sup_{\tau\in I'}\int_{B_R}|V(\tau)|^2
+
\int_{I'}\int_{B_R}|\nabla V|^2
\le C(R,I',I)\,\mathcal E_{loc}(V,p;B_{2R}\times I),
\]

where the right-hand side is controlled by the local energy, critical norm,
pressure, and modulation terms on the larger cylinder.
```

### Main theorem B: windowed local `H^1` control

```{prf:theorem} Rough-core windowed `H^1` bound
Assume the compact-cylinder Caccioppoli estimate, bounded active critical
packet, pressure reconstruction, bounded modulation, and modulation-force
control.  Then for every `R<infty`,

\[
\sup_{\tau\ge\tau_0}\int_{\tau}^{\tau+1}\int_{B_R}|\nabla V|^2dy d\sigma<\infty.
\]

Consequently a critically tight represented branch cannot remain a rough-core
Type II survivor.
```

### Main theorem C: vorticity-controlled subtype

```{prf:theorem} Vorticity-window rough-core subtype
If, in addition, the branch has local vorticity-window control, then the
windowed local `H^1` bound follows directly from the vorticity identity and
local pressure/modulation control.  This gives a simpler exclusion theorem for
vorticity-controlled rough-core candidates.
```

### Internal sections

1. Rough-core alternative from the two-bucket reduction.
2. Renormalized local energy inequality.
3. Compact-cylinder Caccioppoli estimate.
4. Pressure terms and pressure-tail decomposition.
5. Bounded critical packet and local interpolation.
6. Modulation matrix inverse.
7. Translation and scale forcing.
8. Integrated good-window forcing.
9. Vorticity-controlled subtype.
10. Rough-core exclusion theorem.

### What this paper proves

It proves that critically tight represented branches with the stated local PDE
controls emit the windowed local `H^1` bound needed by Paper I.  Hence the
rough-core bucket is closed under explicit local assumptions.

### What this paper does not prove

It does not prove the terminal profile decomposition assumptions of Paper III
or the scale-collapse rigidity assumptions of Paper V.

## Paper V: Scale-collapse alternatives and self-similar rigidity

### Proposed title

**Scale-collapse alternatives for Type II Navier-Stokes blowup**

### PDE purpose

This paper handles branches where the repaired scale drifts in a way not
covered by the compact fixed-window barrier alone.  The point is to split scale
collapse into explicit PDE alternatives: a cost-barrier route or a generalized
self-similar reduction route.

### Source material

Primary files:

- [cluster_s3_scale_collapse_attractor_stratification.md](cluster_s3_scale_collapse_attractor_stratification.md)
- [cluster_c16_scale_negative_drift_dichotomy.md](cluster_c16_scale_negative_drift_dichotomy.md)
- [cluster_c17_scale_collapse_barrier.md](cluster_c17_scale_collapse_barrier.md)

Supporting files:

- [cluster_c4_cost_bridge.md](cluster_c4_cost_bridge.md)
- [cluster_c12_ns_localized_monotonicity_translation.md](cluster_c12_ns_localized_monotonicity_translation.md)

### Main PDE assumptions

Let `V` be a represented branch with scale parameter `lambda(tau)` or logarithmic
scale `ell(tau)`.  Assume:

1. The scale drift is negative on the relevant regime, or satisfies an average
   negative-drift condition.
2. The localized monotonicity formula holds with finite error, or an absolute
   scale-cost comparison is available.
3. In the self-similar route, the branch has compactness sufficient to extract
   an autonomous reduced limit.
4. The modulation coefficients converge or average to autonomous parameters.
5. The omega-limit is stationary for the reduced equation.
6. A parameter-specific Liouville/no-nontrivial-recurrent-profile theorem holds
   for the reduced stationary or self-similar equation.

### Main theorem A: scale-negative drift dichotomy

```{prf:theorem} Scale-negative drift dichotomy
A represented branch with persistent negative scale drift either pays infinite
localized scale cost or admits a renormalized subsequence entering an
autonomous scale-collapse regime.
```

### Main theorem B: cost-barrier exclusion

```{prf:theorem} Scale-collapse cost barrier
If the localized monotonicity formula has finite error and the scale-collapse
cost comparison holds, then persistent scale collapse forces infinite
renormalization cost.  Therefore finite-cost Type II branches cannot realize
this scale-collapse route.
```

### Main theorem C: generalized self-similar reduction

```{prf:theorem} Self-similar reduction and rigidity
Assume the autonomous scale-collapse compactness and stationary-limit
hypotheses.  Then a finite-cost scale-collapse branch yields a nonzero solution
of a reduced stationary or generalized self-similar Navier-Stokes profile
equation.  If the corresponding Liouville/rigidity theorem rules out such
profiles in the parameter range, the scale-collapse branch is impossible.
```

### Internal sections

1. Scale parameter and repaired-gauge drift.
2. Negative drift regimes.
3. Localized monotonicity with finite error.
4. Absolute scale-cost comparison.
5. Cost-barrier exclusion.
6. Thick versus thin scale-collapse regimes.
7. Autonomous modulation extraction.
8. Omega-limit and stationary reduced equation.
9. Liouville/rigidity hypotheses.
10. Scale-collapse exclusion theorem and uncovered regimes.

### What this paper proves

It proves that scale collapse is either blocked by cost or reduced to explicit
self-similar/rigidity assumptions.  Where the rigidity theorem is known, the
scale-collapse branch is excluded.

### What this paper does not prove

It should not pretend that all possible self-similar parameter regimes are
ruled out unless the relevant Liouville theorem is actually proved or cited.
Uncovered regimes should be stated clearly.

## Paper VI: Certificate formalization and final assembly

### Proposed title

**A certificate formalization of Type II exclusion for the three-dimensional
Navier-Stokes equations**

### Purpose

This is the only paper that should foreground the Hypostructure framework,
certificates, and `UP-TypeII` formalization.  It imports the PDE theorems from
Papers I--V and proves that their assumptions and conclusions match the formal
DAG nodes used in `docs/source/type_2`.

The goal is not to prove new PDE estimates.  The goal is to show that the PDE
results compose without hidden circularity.

### Source material

Primary files:

- [README.md](README.md)
- [type2_roadmap.md](type2_roadmap.md)
- [hypostructure_sieve_typeII_certificates.md](hypostructure_sieve_typeII_certificates.md)
- [cluster_c8_full_typeII_exclusion_assembly.md](cluster_c8_full_typeII_exclusion_assembly.md)
- [cluster_c9_up_typeII_interface.md](cluster_c9_up_typeII_interface.md)
- [cluster_c10_ns_up_typeII_promotion.md](cluster_c10_ns_up_typeII_promotion.md)
- [cluster_c11_generic_up_typeII_payload_discharge.md](cluster_c11_generic_up_typeII_payload_discharge.md)
- [cluster_c12_ns_localized_monotonicity_translation.md](cluster_c12_ns_localized_monotonicity_translation.md)
- [cluster_c13_formal_up_typeII_application.md](cluster_c13_formal_up_typeII_application.md)
- [cluster_c14_explicit_up_typeII_ns3d_checklist.md](cluster_c14_explicit_up_typeII_ns3d_checklist.md)
- [cluster_c15_up_typeII_application_closure.md](cluster_c15_up_typeII_application_closure.md)
- [cluster_c18_final_up_typeII_ns3d.md](cluster_c18_final_up_typeII_ns3d.md)

External framework files:

- [../dataset/navier_stokes_3d.md](../dataset/navier_stokes_3d.md)
- [../sketches/gmt/up-type-ii.md](../sketches/gmt/up-type-ii.md)

### Main formalization theorem

```{prf:theorem} Certificate assembly of the PDE Type II exclusion stack
Assume the PDE hypotheses and conclusions of Papers I--V hold in their stated
forms.  Then the corresponding certificates in the Hypostructure Type II DAG
are emitted.  Consequently, every terminal NS3D Type II candidate inside the
declared repaired-gauge backend is routed either to a blocked Type II conclusion
or to a named non-emitted analytic input from Papers I--V.
```

If Papers III--V close all their stated branches, the final conclusion becomes:

```{math}
\forall \omega\in\mathcal U^{NS}_{II},
\qquad
K_{\mathrm{SC}_\lambda}^{\sim}(\omega).
```

### Internal sections

1. The Hypostructure Type II DAG.
2. Translation dictionary from PDE assumptions to certificates.
3. Paper I certificates: compact barrier.
4. Paper II certificates: representation, stratification, and two-bucket
   reduction.
5. Paper III certificates: radiation and multibubble blockers.
6. Paper IV certificates: rough-core blocker.
7. Paper V certificates: scale-collapse blocker or named rigidity defect.
8. `UP-TypeII` metatheorem and NS3D applicability checklist.
9. Final assembly theorem.
10. Non-emitted analytic inputs and exact remaining defect strata.

### Translation table

| PDE result | Certificate emitted in Paper VI |
|---|---|
| Paper I compact repaired-gauge barrier | `K_{SC_lambda}^{blk}` on compact represented branches |
| Paper II representation theorem | `K_{RepBridge}^+` |
| Paper II terminal state-space stratification | `K_{StateStratExh}^+` |
| Paper II positive finite active mass | `K_{StratCritMass}^+` and `K_{StratCritPacket}^+` |
| Paper II two-bucket reduction | C5 finite-cost classification |
| Paper III small-data scattering removal | `K_{ScattBranch}^-` |
| Paper III exterior regular discard | `K_{ExtRegDiscard}^+` |
| Paper III terminal nonlinear profile decomposition | `K_{NLProfDec,NS3D}^+` |
| Paper III radiative/multibubble exclusion | `K_{RadBlk}^+` and multibubble blocker payloads |
| Paper IV Caccioppoli estimate | `K_{CaccioppoliReg}^+` |
| Paper IV windowed `H^1` estimate | `K_{WinH1}^+` |
| Paper IV rough-core exclusion | `K_{RoughCoreBlk}^+` |
| Paper V scale-collapse cost barrier | `K_{ScaleCollapseCostBridge}^+` or `K_{AbsScaleCostBridge}^+` |
| Paper V self-similar rigidity | `K_{S3NRSPayload}^+` |
| Final formal assembly | `K_{SC_lambda}^{sim}` |

### What this paper proves

It proves that the PDE papers compose into the formal terminal Type II
exclusion theorem.  It also identifies the exact analytic assumption that is
responsible for any remaining open branch.

### What this paper should not do

It should not prove the PDE estimates again.  It should not hide analytic
assumptions behind certificates.  Every certificate must point back to a PDE
theorem from Papers I--V or to a clearly stated imported theorem.

## Dependency graph

```{mermaid}
flowchart TD
  P1["Paper I: compact repaired-gauge barrier"] --> P6["Paper VI: certificate assembly"]
  P2["Paper II: representation and two-bucket reduction"] --> P3["Paper III: radiation / multibubble"]
  P2 --> P4["Paper IV: rough core"]
  P2 --> P5["Paper V: scale collapse"]
  P2 --> P6
  P3 --> P6
  P4 --> P6
  P5 --> P6
```

The PDE logic before formalization is:

```{mermaid}
flowchart TD
  A["Suitable NS solution with Type II blowup sequence"] --> B["Paper II: extract repaired-gauge branch"]
  B --> C["Paper II: positive finite active critical mass"]
  B --> D["Paper I: compact barrier"]
  C --> D
  D --> E["If finite cost survives: tightness fails or H1-window fails"]
  E --> F["Paper III: radiation / multibubble exclusion"]
  E --> G["Paper IV: rough-core exclusion"]
  B --> H["Paper V: scale-collapse alternatives"]
  F --> I["No radiative/multibubble survivor under profile assumptions"]
  G --> J["No rough-core survivor under local regularity assumptions"]
  H --> K["No scale-collapse survivor under cost or rigidity assumptions"]
  I --> L["PDE Type II exclusion package"]
  J --> L
  K --> L
  L --> M["Paper VI: certificate formalization"]
```

## Suggested abstracts

### Paper I abstract

We prove a compactness barrier for Type II blowup in the three-dimensional
Navier-Stokes equations using a repaired scale-translation gauge.  The theorem
applies to repaired-gauge renormalized orbits with positive finite critical
mass, uniform critical tightness, local Caccioppoli control, pressure
reconstruction, bounded modulation, and finite renormalization cost.  The proof
combines pressure-tail estimates, repaired-gauge nondegeneracy, good-window
compactness, and low-dissipation rigidity.  It shows that a nonzero compact
renormalized Type II orbit must have infinite total renormalization cost.

### Paper II abstract

We construct repaired-gauge renormalized branches from Type II Navier-Stokes
blowup sequences under explicit extraction and gauge-solvability assumptions.
The construction gives the renormalized equation, pressure pullback, modulation
coefficient identities, positive finite critical mass on retained terminal
strata, and compatibility with the compactness barrier.  As a consequence,
every represented finite-cost survivor is reduced to one of two alternatives:
loss of critical tightness or failure of local windowed `H^1` control.

### Paper III abstract

We study terminal profile decompositions for Type II Navier-Stokes blowup
sequences.  Under critical profile decomposition, small-data Kato stability,
exterior regularity, terminal camera compatibility, and nonlinear profile
stability, we eliminate scattering profiles, exterior profiles, radiative
critical tails, same-point finite cascades, and separated multipoint
multibubbles.  The result is a traditional concentration-compactness theorem
closing the radiative/multibubble alternatives under explicit terminal profile
assumptions.

### Paper IV abstract

We prove a rough-core exclusion theorem for represented repaired-gauge Type II
Navier-Stokes branches.  A rough core is a critically tight branch with
persistent local high-frequency activity or gradient concentration in a bounded
renormalized region.  We show that suitability, pressure reconstruction,
bounded active critical mass, repaired-gauge modulation invertibility, and
pointwise or integrated modulation-force control imply uniform local windowed
`H^1` bounds.  Thus the rough-core alternative from the two-bucket reduction is
closed under explicit local regularity assumptions.

### Paper V abstract

We analyze scale-collapse alternatives for represented Type II Navier-Stokes
branches.  Persistent negative scale drift is shown to lead either to an
infinite-cost mechanism through localized monotonicity or to an autonomous
renormalized limit satisfying a generalized self-similar stationary equation.
When the corresponding Liouville or rigidity theorem holds, the scale-collapse
branch is excluded.  The theorem isolates any uncovered parameter ranges as
explicit self-similar rigidity assumptions.

### Paper VI abstract

We formalize the preceding PDE Type II exclusion results in the Hypostructure
certificate framework.  Each analytic assumption and theorem from Papers I--V
is translated into a certificate node, and the resulting directed acyclic proof
graph is checked for circularity.  The final assembly theorem states that
terminal NS3D Type II candidates in the declared repaired-gauge backend are
routed either to a blocked Type II conclusion or to an explicitly named
non-emitted analytic input.

## How to keep Papers I--V independent of certificates

Use the following rule in Papers I--V:

- Do not write `K_{...}` notation in theorem statements.
- Do not refer to C1, C2, S8, U3b, or other node names in main text.
- Translate each node into a PDE assumption or theorem.
- Put any certificate correspondence in a short appendix titled “Relation to
  the formalization paper,” or omit it until Paper VI.

For example, instead of writing:

```{math}
K_{\mathrm{RepBridge}}^+\wedge K_{\mathrm{StratCritPacket}}^+\Rightarrow K_{\mathrm{WinH1}}^+,
```

Paper IV should write:

```{prf:theorem}
Let `V` be a repaired-gauge renormalized solution with bounded active critical
mass, pressure reconstruction, invertible modulation matrix, compact-cylinder
suitability, and bounded modulation forcing.  Then `V` satisfies uniform local
windowed `H^1` bounds.
```

Paper VI then states that this theorem emits the certificate
`K_{WinH1}^+`.

## Review strategy

The PDE papers should be reviewed in this order:

1. Paper I: check the barrier contradiction.
2. Paper II: check extraction and repaired-gauge representation.
3. Paper IV: check local regularity and rough-core closure.
4. Paper III: check terminal profile decomposition and radiation/multibubble
   closure.
5. Paper V: check scale-collapse alternatives and rigidity assumptions.
6. Paper VI: check formal composition.

Paper I and Paper IV are closest to standard local PDE estimates.  Paper III is
closest to concentration compactness.  Paper V is the most sensitive because it
depends on the exact rigidity/Liouville theorem available for the reduced
self-similar equation.  Paper VI should be short and formal once the PDE papers
are stable.

## Minimal version

If the goal is fewer papers, use four papers:

1. **Compact barrier and repaired gauges**: merge Papers I and II.
2. **Terminal profiles and radiation/multibubbles**: Paper III.
3. **Rough core and scale collapse**: merge Papers IV and V.
4. **Certificate formalization and final assembly**: Paper VI.

The six-paper version is better for review because it separates compactness,
representation, profiles, local regularity, scale collapse, and formalization.

## Final recommendation

Use the six-paper structure.  The key correction is that certificates should
not drive the exposition of the analytic papers.  Papers I--V should look like
ordinary PDE papers with explicit assumptions.  Paper VI is where the
certificate language belongs: it imports the PDE theorems, builds the formal
translation table, checks the DAG, and states the final terminal Type II
assembly.
