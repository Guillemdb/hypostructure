# Burgers 1D Ground-Truth Lean Roadmap

Date: 2026-04-17

## Purpose

This document specifies what remains before the Lean development proves the
classical one-dimensional viscous Burgers global regularity theorem purely from
mathlib-formalized local PDE certificates assembled by the hypostructure
machinery.

The current state is not enough for that claim.

Current honest status:

```text
Yes:
  Lean checks the hypostructure certificate route and proof-object shape.

No:
  Lean does not yet prove classical Burgers global regularity purely from
  mathlib-formalized local PDE certificates.

Additional proved restricted instance:
  Lean now proves the full hypostructure template route for the zero
  equilibrium: zero initial datum, zero Burgers curve, zero heat curve,
  local certificate bundle, final route certificate, global H1, and
  positive-time smoothness. This is a no-axiom/no-sorry ground-truth sanity
  check for the framework, not the arbitrary-data Burgers theorem.

Additional proved nonzero family:
  Lean now proves the same template-shaped route for every constant
  equilibrium `u0 = m`, including nonzero constants. This uses the strengthened
  weak-test language with compact-time and periodic-space cancellation laws.
```

## Implementation Status: 2026-04-16

The migrated ground-truth path now has a compiled interface, certificate-run,
and conditional final theorem under:

```text
Hypostructure/Backends/Burgers1D/Parameters.lean
Hypostructure/Backends/Burgers1D/Torus.lean
Hypostructure/Backends/Burgers1D/GroundTruthState.lean
Hypostructure/Backends/Burgers1D/GroundTruthPDE.lean
Hypostructure/Backends/Burgers1D/GroundTruthWindows.lean
Hypostructure/Backends/Burgers1D/GroundTruthLocalAnalysis.lean
Hypostructure/Backends/Burgers1D/GroundTruthCertificates.lean
Hypostructure/Backends/Burgers1D/GroundTruthRun.lean
Hypostructure/Backends/Burgers1D/GroundTruthHeat.lean
Hypostructure/Backends/Burgers1D/GroundTruthColeHopf.lean
Hypostructure/Backends/Burgers1D/GroundTruthLock.lean
Hypostructure/Backends/Burgers1D/GroundTruthUpgrade.lean
Hypostructure/Backends/Burgers1D/GroundTruthFinal.lean
Hypostructure/Backends/Burgers1D/GroundTruthAudit.lean
Hypostructure/Backends/Burgers1D/GroundTruthZeroEquilibrium.lean
Hypostructure/Backends/Burgers1D/GroundTruthConstantEquilibrium.lean
Hypostructure/Literature/Burgers/Periodic1D.lean
Hypostructure/Literature/ColeHopf/PeriodicBurgers1D.lean
Hypostructure/Literature/Heat/Periodic1D.lean
Hypostructure/Framework/Route.lean
Hypostructure/Framework/Upgrade.lean
hypostructure_reusable_framework.md
```

`Hypostructure.lean` now imports the constant-equilibrium and zero-equilibrium
ground-truth routes; the other ground-truth files above are pulled in
transitively. This keeps the public Lean surface aligned with the template route
instead of also importing old examples and scaffold backends.

Implemented:

- Phase 1: `PeriodicH1State` is a concrete periodic carrier over
  `BurgersTorus`, with a value field, weak-derivative field, `Memℒp` witnesses
  for both fields, and a weak-derivative integration-by-parts specification.
- Phase 1: constants, addition, scalar multiplication, negation, subtraction,
  mean, energy, dissipation, mean-zero projection, and the mean-zero plus
  constant decomposition are implemented and proved for `PeriodicH1State`.
- Phase 2: the new PDE-facing path defines `BurgersSolutionCurve`,
  `InitialCondition`, `PeriodicBoundary`, `WeakBurgersResidual`,
  `SolvesViscousBurgersWeak`, `UniqueWeakBurgersSolution`,
  `GlobalH1Solution`, `SmoothAtPositiveTime`, and
  `BurgersGroundTruthGlobalRegularityStatement`.
- Phase 2: `SmoothCompactTimePeriodicSpaceTest` now carries the cancellation
  laws for constant heat and Burgers residuals. These are the finite-window
  abstraction of compact time support and periodic spatial derivative
  cancellation, and they are what make nonzero constant weak solutions
  provable in Lean.
- Phase 2: no package-level evolution class remains on the ground-truth path.
  The route keeps only the concrete weak PDE predicates and the global target
  statement. Backend construction is represented by explicit remaining permits
  until each permit is replaced by a proved theorem.
- Phase 2 cleanup: `BurgersSolutionCurve` and `PeriodicHeatCurve` no longer
  contain an unused `timeRegularity : Prop` field. Regularity appears only in
  concrete predicates such as `GlobalH1Solution`, `SmoothAtPositiveTime`,
  `HeatGlobalH1`, and `HeatSmoothAtPositiveTime`.
- Locality refactor step 1: `GroundTruthWindows.lean` now defines the reusable
  window layer used by the next local certificate refactors: `TimeWindow`
  membership/subwindow lemmas, `BurgersWindow`, `HeatWindow`,
  `RouteLocalBadGermWindow`, and `CertifiedBurgersLocalWindow`. `GroundTruthHeat.lean`
  also now exposes `CertifiedHeatWindow` plus heat residual and boundary
  restriction lemmas on heat windows. `GroundTruthCertificates.lean` maps the
  existing `BurgersBadGerm`/`TimeSpaceCylinder` data into the route-local
  bad-germ window language.
- Phase 3: finite-window energy inequalities and local energy framework
  certificates are implemented. The smooth snapshot energy identity was moved
  into `GroundTruthLocalAnalysis.lean` so the ground-truth path no longer
  imports the old `Analysis.lean` scaffold.
- Phase 4: local Poincare, local mean-sector preservation, local dissipative
  window, and mean-sector decomposition certificates are implemented against
  the new carrier.
- Phase 5: local bad-germ data, local admissibility, and local capacity
  certificates are implemented without asserting that the global singular set
  is empty.
- Phase 6-10 migration boundary: `GroundTruthRun.lean` records the
  document-shaped sieve route, including the negative scaling certificate
  `K_SC_lambda^-`, the blocked Type II branch `K_SC_lambda^blk`, the effective
  continuation certificate `K_SC_lambda^~`, the Lock certificate
  `K_Cat_Hom^blk`, and the final regularity certificate
  `K_Reg_Burgers1D^+`.
- Phase 6-10 cleanup: the Burgers `RunValidity` instance no longer fills route
  fields with `True`. Required trace execution, boundary handling, Lock
  execution, preservation lemmas, and upgrade non-circularity are proved from
  the recorded execution trace, certificate chain, and explicit non-membership
  checks.
- Reusable framework extraction: Hypostructure.Framework.Route now contains the
  generic TraceBackedRoute, TraceBackedRouteProof, route-validity theorem, and
  TraceBackedTargetClaim. The Burgers route instantiates this reusable layer
  instead of defining route validity manually.
- Reusable upgrade extraction: Hypostructure.Framework.Upgrade now contains the
  generic E1-E13 Lock tactic identifiers, LockTacticDossier, and
  LocalToTargetUpgradeDossier plus CertifiedUpgradeDossier. It also contains
  `APosterioriLocalization`, `APosterioriLocalizationDossier`, and the generic
  `APosterioriLocalization.target` metatheorem. The Burgers route instantiates
  E5 as its Lock tactic, routes the final theorem through the combined generic
  upgrade dossier, and instantiates the reusable a-posteriori theorem for its
  decomposed missing-regularity-to-raw-failure-window and raw-failure-window
  nonextendability literature facts.
- Rigor boundary extraction: Hypostructure.Framework.Rigor records the three
  proof-boundary layers used by the Burgers route: framework-proved logic
  (`F`), reusable literature facts (`L`), and problem-specific math (`P`).
  It now also provides `AxiomBoundaryAudit`, a reusable checked object for
  theorem checkpoints whose named custom boundary should be exactly the
  reusable literature boundary and contain no problem-specific assumptions.
  `GroundTruthAudit.lean` instantiates this for
  `burgers_groundTruth_dataset_theorem_from_axioms`.
  `GroundTruthFinal.lean` now exposes `burgersGroundTruthRigorBoundary`, plus
  separate literature and problem-specific boundary lists.
- Core bundle factory: `GroundTruthCoreFactory.lean` now provides
  `BurgersGroundTruthCoreCertificateFactory.toBundle` and an axiom-free
  `burgersCanonicalCoreCertificateBundle` built from the canonical zero local
- Windowed Burgers local input boundary: `GroundTruthWindows.lean` now defines
  `SolvesViscousBurgersWeakOnWindow`, `WeakBurgersResidualOnWindow`, and
  `CertifiedBurgersLocalWindow`. `GroundTruthCoreFactory.lean` introduces
  `BurgersCorePDELocalWindowInputsFor nu u0 W`, which contains only a certified
  Burgers window and windowed local estimates. The core bundle itself now
  consumes `solutionCertifiedOnWindow`, so there is no remaining
  window-to-global adapter axiom in the local certificate path.
- Phase 6: `GroundTruthHeat.lean` now separates the heat construction source
  from the exported local certificate. `LocalHeatWindowCertificate`
  exposes only windowed heat residual, forward-time local uniqueness, positive-time
  smoothing on the window, finite-window energy/dissipation contraction, and
  heat bad-germ exclusion. The heat-side forbidden predicate is now concrete:
  `HeatForbiddenBadGerm` means a route-local heat germ contains a positive time
  at which the certified heat curve is not smooth. The certificate excludes it
  by applying its windowed smoothing theorem to supported germs. The underlying
  `CertifiedHeatWindow` now carries only the local window, initial profile,
  heat curve, local initial condition on the window, and local `H¹` boundary on
  the window; it no longer bundles a global `SolvesPeriodicHeatWeak`. The old global-shaped
  `LocalHeatSmoothingAndUniqueness` interface has been replaced by the reduced
  `PeriodicHeatSemigroupBackend`, which is only a construction source for local
  window certificates and no longer exposes unused semigroup/global fields to
  the proof route.
- Literature heat extraction: the remaining standard periodic heat boundary is
  no longer a global semigroup axiom or a full local-certificate axiom. The
  `L²` Fourier reconstruction from the evolved coefficients is now proved in
  `Hypostructure.Literature.Heat.Periodic1D` as
  `periodicHeat1D_l2ReconstructionTheory_literature`: Lean constructs the
  reconstructed value and weak-derivative `L²` objects and proves coefficient
  recovery, Fourier-series convergence in `L²`, and Parseval identities for
  those `L²` reconstructions. Lean also proves the Fourier/Sobolev `H¹`
  reconstruction package `periodicHeat1D_fourierH1ReconstructionTheory_literature`:
  the reconstructed value/derivative pair satisfies the Fourier weak-derivative
  relation, Parseval identities, and finite-time contractions. Lean now also
  proves `periodicHeat1D_positiveTimeCoefficientSmoothingTheory_literature`:
  integer Gaussian summability, polynomially weighted heat-multiplier
  summability, and polynomially weighted evolved value/derivative mode-energy
  summability for every positive time. Lean also assembles
  `periodicHeat1D_positiveTimeSmoothFourierRepresentativeTheory_literature`,
  a positive-time Fourier representative carrying the reconstructed
  Fourier-`H¹` state, evolved coefficient identities, Parseval/contraction, and
  all polynomial weighted summability facts. It also proves weighted `ℓ¹`
  summability of the evolved value/derivative coefficients, constructs
  continuous complex Fourier-series slices on the torus, and proves the full
  complex spatial-derivative Fourier tower with termwise classical
  differentiability. Lean now also proves
  `periodicHeat1D_positiveTimeSmoothPeriodicH1RepresentativeTheory_literature`:
  taking real parts of that derivative tower gives a concrete positive-time
  `PeriodicH1State`, and the recursive weak-derivative chain follows from a
  reusable integration-by-parts bridge from classical lifted derivatives. Lean
  now also proves `periodicHeat1D_fourierH1CandidateIdentification_literature`:
  positive-time Fourier coefficients of the explicit Lean-defined candidate
  heat curve are recovered from the formal heat-evolved coefficients by the
  continuous Fourier representative and Hilbert-basis coefficient recovery.
  Lean derives the zero-time coefficient case and the energy/Parseval match
  with the Fourier-`H¹` reconstruction from coefficient identities plus
  Parseval. The former monolithic heat upgrade now has one remaining reusable
  heat PDE-semantics fact:
  `periodicHeat1D_weakSolutionValueFourierEvolution_literature`, stating that
  every weak periodic heat solution has the standard forward-time value Fourier
  coefficient evolution on each finite local window. Lean then proves
  weak-derivative coefficient evolution from the value statement using the
  carrier's weak-derivative Fourier identity, and proves canonical
  forward-time windowed uniqueness using `periodicH1State_ext_fourierCoeff`.
  Lean now constructs
  `periodicHeat1D_fourierH1IntegratedClassicalHeat_literature` without custom
  axioms: termwise Fourier differentiation gives the positive-time derivative,
  the local spatial heat balance follows from the weak-derivative tower, the
  nonpositive-time branch follows from test support, and interval-integral
  congruence gives the finite-window integrated heat certificate. That
  certificate is converted into the windowed weak residual by
  `HeatWindowResidual.of_integratedClassical`. Positive-time smoothing is now derived in Lean by
  `periodicHeat1D_fourierH1WindowSmoothing_fromContinuousCurve` from the
  positive-time identification with the proved smooth real Fourier
  representative. Lean proves the assembler
  `periodicHeat1D_localWindowCertificate_literature`, which directly supplies a
  `LocalHeatWindowCertificate` for the exact datum and route window. Heat
  bad-germ exclusion is already discharged by
  `periodicHeat1D_badGermExclusion_fromSmoothing`, so it is not a separate
  axiom. A full heat semigroup may still be proved later as a construction
  source, but the final theorem no longer depends on a semigroup-shaped
  literature assumption.
- Heat Fourier multiplier layer: `Hypostructure.Literature.Heat.Periodic1D`
  now defines the concrete mode frequency `(2πn)^2`, heat exponent
  `-ν(2πn)^2t`, and multiplier `exp(-ν(2πn)^2t)`. Lean proves nonnegative
  frequency, strict positivity for nonzero modes, zero-time identity,
  multiplier positivity, positive-time contraction `≤ 1`, strict nonzero-mode
  damping for positive time, absolute-value and square contraction, and
  time-addition composition. The bundled
  `periodicHeat1D_fourierModeTheory_literature` is a definition, not an axiom.
  Lean also proves the coefficient-level package
  `periodicHeat1D_fourierCoefficientTheory_literature`: evolved Fourier
  coefficient initial values, coefficient semigroup law, derivative
  compatibility, all-mode value/derivative summability preservation, and all-
  mode `ℓ²` contraction for the value and weak-derivative coefficient families.
  To support this, the analysis module now exposes full all-mode value and
  derivative Parseval/summability theorems in addition to the mean-zero Poincare
  projections. Lean also proves the Hilbert-space reconstruction package
  `periodicHeat1D_l2ReconstructionTheory_literature`: the evolved value and
  derivative coefficients define `ℓ²` vectors, mathlib's Fourier Hilbert basis
  reconstructs `L²` objects from them, and Lean verifies coefficient recovery,
  `L²` Fourier-series convergence, and Parseval for those reconstructed
  objects. Lean also proves
  `periodicHeat1D_fourierH1ReconstructionTheory_literature`, which packages the
  Fourier weak-derivative relation, value/derivative Parseval identities, and
  finite-time value/derivative contractions on the reconstructed Fourier-`H¹`
  carrier. Lean also proves
  `periodicHeat1D_positiveTimeCoefficientSmoothingTheory_literature`, the
  coefficient-level positive-time smoothing theorem: for every positive time
  and every polynomial frequency weight, the heat multiplier and the evolved
  value/derivative mode energies are summable. Lean also proves
  `periodicHeat1D_positiveTimeSmoothFourierRepresentativeTheory_literature`,
  which packages those smoothing estimates together with the reconstructed
  Fourier-`H¹` state, evolved coefficient identities, Parseval identities, and
  finite-time contractions. This package now also includes weighted `ℓ¹`
  summability of the evolved value/derivative coefficients and continuous
  complex Fourier-series slices with pointwise convergence. Lean also proves
  `periodicHeat1D_localContraction_fromFourierReconstruction`, deriving the
  local heat contraction fields from reconstruction Parseval identities plus
  coefficient-level contraction, and
  `periodicHeat1D_localTheory_fromFourierReconstruction`, assembling the local
  heat theory from the narrowed reconstruction package. The older
  `periodicHeat1D_localTheory_literature` name is now a Lean definition, not an
  axiom.
- Phase 7: `GroundTruthColeHopf.lean` defines the concrete Cole-Hopf transform
  interface, inverse laws, certified-window chart validity, window mapping,
  inverse transfer, residual-transfer statements, uniqueness transfer,
  route-local bad-germ transport, the framework `K_ColeHopf^+` certificate,
  and `PeriodicColeHopfBackend.windowBridge` as the construction-source-to-
  local-certificate restriction theorem. Bad-germ transport is now derived by
  Lean from H¹ preservation of the transform instead of being a backend axiom
  field.
- Literature Cole-Hopf extraction: the reusable periodic Burgers/KPZ-style
  Cole-Hopf backend now lives in
  `Hypostructure.Literature.ColeHopf.PeriodicBurgers1D` as the bundled reusable
  theory package `PeriodicBurgers1DColeHopfTheory`. Its single literature axiom
  `periodicBurgers1D_coleHopfTheory_literature` exports transform data, chart
  validity, window mapping, inverse window mapping, residual transfer in both
  directions, uniqueness transfer, and the theorem that arbitrary periodic `H¹`
  Burgers profiles lie in the transform PDE domain. Lean proves the old split
  names as projections and assembles `periodicBurgers1D_coleHopfBackend_literature`
  from the package, so Burgers-specific code consumes only the resulting local
  bridge and reusable literature-domain theorem.
- Literature Burgers continuation extraction: reusable one-dimensional periodic
  viscous Burgers continuation/localization now lives in
  `Hypostructure.Literature.Burgers.Periodic1D` as the bundled theory package
  `PeriodicBurgers1DContinuationTheory`, exposed by the single literature axiom
  `periodicBurgers1D_continuationTheory_literature`. It exports local
  uniqueness on certified windows, missing-regularity-to-raw-failure-window
  localization, and raw-failure-window nonextendability for the canonical
  finite-`H¹` obstruction criterion. The old split names are Lean projections.
  The final Burgers route proves `burgers_localUniquenessOnOverlapsFromLiterature`,
  adapting the windowed theorem to the overlap-uniqueness record consumed by the
  local continuation chain; `GroundTruthUpgrade.lean` no longer imports Burgers
  literature facts.
- Phase 8: `GroundTruthCertificates.lean` now implements the documented
  finite bad-pattern library: the singleton finite-time `H¹` blow-up template,
  `K_Germ^+`, `K_init^+`, and `K_CatLib^+` certificate objects, and a
  completeness theorem classifying every route-local finite-time `H¹` bad germ
  through that library. `GroundTruthLock.lean` consumes this completeness
  theorem before applying the local Cole-Hopf/heat E5 contradiction.
- Phase 9: `GroundTruthUpgrade.lean` names the final local-to-global analytic
  theorem object, the generic local continuation/localization chain interface,
  and its framework analytic-regularity certificate. Literature-backed default
  adapters live in `GroundTruthFinal.lean`, so the reusable upgrade layer has no
  Burgers literature import cycle.
- Phase 10 migration boundary: `GroundTruthFinal.lean` exposes the migrated
  dataset claim over `BurgersGroundTruthBackendPermits`. This deliberately
  prevents the final ground-truth path from using any old packaged
  global regularity shortcut.
- Proved restricted instance: `GroundTruthZeroEquilibrium.lean` constructs the
  zero periodic H1 state, the zero Burgers curve, the zero heat curve, proves
  the Burgers and heat weak residuals by computation, instantiates every core
  local certificate for that equilibrium, and exposes
  `burgers_zeroEquilibrium_template_theorem` with no axioms or sorries.
- Proved restricted family: `GroundTruthConstantEquilibrium.lean` proves the
  same template-shaped statement for every constant equilibrium `m : ℝ`,
  including the nonzero Burgers transport term. It proves the constant Burgers
  weak residual, constant heat weak residual, global H1, positive-time
  smoothness, and all local certificates for the constant profile.
- Restricted heat backend: `GroundTruthConstantEquilibrium.lean` implements the
  constant-sector heat flow and proves zero-time, additivity, fixed constants,
  mean preservation, energy contraction, dissipation contraction, and the
  exported `constantLocalHeatWindowCertificate` for the singleton local window.
- Restricted bridge: `GroundTruthConstantEquilibrium.lean` also implements the
  identity Cole-Hopf transform on the constant sector as a
  `PeriodicColeHopfBackend`, restricts it to a `LocalColeHopfWindowBridge`, and
  proves the local bridge, heat certificate, Lock theorem, and analytic-upgrade
  input for every constant equilibrium. The heat-to-Burgers direction
  constructs the proved constant Burgers solution, so this is a genuine
  restricted bridge rather than a placeholder proposition.

Compilation checks run after the replacements:

```text
lake build Hypostructure.Backends.Burgers1D.GroundTruthState
lake build Hypostructure.Backends.Burgers1D.GroundTruthPDE
lake build Hypostructure.Backends.Burgers1D.GroundTruthWindows
lake build Hypostructure.Backends.Burgers1D.GroundTruthLocalAnalysis
lake build Hypostructure.Backends.Burgers1D.GroundTruthCertificates
lake build Hypostructure.Backends.Burgers1D.GroundTruthRun
lake build Hypostructure.Backends.Burgers1D.GroundTruthHeat
lake build Hypostructure.Backends.Burgers1D.GroundTruthColeHopf
lake build Hypostructure.Backends.Burgers1D.GroundTruthLock
lake build Hypostructure.Backends.Burgers1D.GroundTruthUpgrade
lake build Hypostructure.Backends.Burgers1D.GroundTruthFinal
lake build Hypostructure.Backends.Burgers1D.GroundTruthZeroEquilibrium
lake build Hypostructure.Backends.Burgers1D.GroundTruthConstantEquilibrium
lake build Hypostructure
```

The focused checks and the public `Hypostructure` target passed after removing
the obsolete Burgers scaffold files.

Important caveat:

The reusable ground-truth interface, local certificate layer, concrete heat and
Cole-Hopf certificate languages, finite-library Lock theorem, named
analytic-upgrade boundary, zero-equilibrium template theorem, and all-constant
equilibrium template theorem are implemented. They do not yet prove classical
global Burgers regularity for arbitrary initial data. The remaining hard
analytic work is to instantiate the nontrivial heat local-window certificate,
Cole-Hopf transform, decomposed PDE-local core analytic inputs, and a-posteriori
failure-localization theorem with mathlib-backed constructions.

Current locality status:

The heat side has been refactored: `K_HeatSmooth^+` is now backed by
`LocalHeatWindowCertificate`, and any all-time heat semigroup theorem must be
restricted to a certified window before entering the certificate chain. The
Cole-Hopf side has also been refactored: `K_ColeHopf^+` is now backed by
`LocalColeHopfWindowBridge`, with chart validity, window mapping, inverse
transfer, residual transfer, uniqueness transfer, and bad-germ transport. The
remaining gap is no longer the certificate boundary; it is the analytic
instantiation for arbitrary nonconstant data. The only place where global
Burgers regularity should be assembled is `BurgersAnalyticUpgradeTheorem`.

Required separation of roles:

| Component | Allowed payload | Forbidden payload |
|---|---|---|
| `K_HeatSmooth^+` | windowed heat residual, forward-time local heat uniqueness, local smoothing, heat-side bad-germ exclusion | Burgers global existence, Burgers global uniqueness, Burgers global smoothness |
| `K_ColeHopf^+` | windowed chart validity, local residual transfer, inverse transfer, bad-germ transport | any theorem equivalent to `BurgersGroundTruthGlobalRegularityStatement` |
| `K_Cat_Hom^blk` | contradiction for a route-local bad morphism using E5 | a global empty-singularity theorem |
| `K_Reg_Burgers1D^+` | final analytic assembly of the local chain | use as input to any earlier certificate |

The target is to remove every hardcoded or package-supplied global regularity
shortcut. The final theorem must be assembled from local certificates through
the sieve, Lock, and analytic upgrade.

## Template Document vs Current Lean Proof

The template implementation document and `docs/source/dataset/burgers_1d.md`
describe a proof route:

- define the thin problem interface;
- verify local certificates for energy, compactness, scaling, capacity,
  stiffness, topology, tameness, ergodicity, representation, oscillation, and
  boundary behavior;
- block the bad morphism using the Lock;
- use a local analytic bridge, here Cole-Hopf plus heat smoothing, to upgrade
  the blocked route into global Burgers regularity.

The migrated Lean path now mirrors the route shape, but not all mathematical
backend proofs are in mathlib yet:

- The route shape is implemented by `GroundTruthRun.lean`. It records the
  document's negative scaling branch, Type II block, structural exclusion, and
  final regularity certificate.
- The local core language is implemented by `GroundTruthState.lean`,
  `GroundTruthPDE.lean`, and `GroundTruthCertificates.lean`. These replace the
  old abstract `BurgersState` package with a concrete periodic H1 carrier,
  concrete weak PDE predicates, and local certificate data.
- The final theorem boundary is implemented by `GroundTruthFinal.lean`, but it
  is conditional on `BurgersGroundTruthBackendPermits`. Those fields now carry
  concrete heat, Cole-Hopf, and analytic-upgrade theorem objects instead of
  arbitrary heat/Cole-Hopf/Lock propositions.
- The zero-equilibrium theorem is implemented by
  `GroundTruthZeroEquilibrium.lean`. It proves an end-to-end template instance
  for `u0 = 0` by constructing the actual zero solution and proving the local
  residual and certificate checks directly. This validates the certificate/run
  architecture but does not instantiate the nonzero arbitrary-data backend
  permits.

The old Burgers route has been removed from the current backend tree. The only
remaining Burgers path is the migrated route through `GroundTruthRun.lean` and
`GroundTruthFinal.lean`. It does not use the old model instance as semantic
evidence, and it exposes the exact missing math as backend permits.

To make the Lean proof fully match the template document,
`BurgersGroundTruthBackendPermits` must be constructible internally from
mathlib-backed analysis. After that, `burgers_groundTruth_dataset_theorem`
should take only the PDE parameters, initial data, local hypotheses required by
the template, and the constructed local certificates. It should no longer take
an external backend theorem package.

## Target Theorem

The final migrated theorem should have two connected outputs:

1. The classical PDE conclusion:

   ```lean
   BurgersGroundTruthGlobalRegularityStatement nu
   ```

   This replaces the old scaffold-facing `BurgersGlobalRegularityStatement`
   on the ground-truth path. It means the actual periodic viscous Burgers
   theorem:

   - initial data `u0` belongs to a real periodic `H^1(T)` carrier;
   - there exists a global solution curve;
   - the curve satisfies the actual PDE
     `u_t + u * u_x = nu * u_xx`;
   - the solution is unique in the chosen weak/classical class;
   - the solution remains in `H^1` globally;
   - the solution is smooth for every positive time.

2. The hypostructure conclusion:

   ```lean
   "K_Reg_Burgers1D^+" ∈ burgersGroundTruthFinalCertificateChain.certificates
   ```

   with the final certificate derived only from local certificate premises.

The intended logical shape is:

```lean
localEnergyCertificate nu
  -> localPoincareCertificate nu
  -> localMeanSectorCertificate nu
  -> localBadGermCapacityCertificate nu
  -> localColeHopfCertificate nu
  -> localHeatSmoothingCertificate nu
  -> lockBlocksBurgersBadGerms nu
  -> analyticUpgradeFromLocalCertificates nu
  -> BurgersGroundTruthGlobalRegularityStatement nu
```

The final theorem must not use a pre-existing
`BurgersGroundTruthGlobalRegularityStatement` field as an input.

## Removed Hardcoded or Too-Weak Pieces

The following pieces were removed from the current Burgers backend surface. They
are documented here as historical failure modes that must not be reintroduced.

### 1. `BurgersPDEEvolutionPackage` is too permissive

Removed problem:

```lean
class BurgersPDEEvolutionPackage (nu : BurgersParameters)
    extends BurgersEvolution nu, BurgersEvolutionRegularity nu where
  solvesViscousBurgers : Prop
  solvesViscousBurgers_holds : solvesViscousBurgers
  periodicBoundary : Prop
  periodicBoundary_holds : periodicBoundary
  stateSpaceH1Like : Prop
  stateSpaceH1Like_holds : stateSpaceH1Like
```

This lets any backend choose arbitrary propositions for the PDE and regularity
fields. A ground-truth proof needs concrete predicates, not arbitrary `Prop`
slots.

Current replacement: the ground-truth path directly states
`SolvesViscousBurgersWeak`, `GlobalH1Solution`, `UniqueWeakBurgersSolution`,
`SmoothAtPositiveTime`, and `BurgersGroundTruthGlobalRegularityStatement`.
There is no package class on the current route.

### 2. `BurgersEvolutionRegularity` can currently be trivial

Removed problem:

```lean
class BurgersEvolutionRegularity (nu : BurgersParameters) [BurgersEvolution nu] where
  globalH1 : BurgersState -> Prop
  unique : BurgersState -> Prop
  smoothPositiveTime : BurgersState -> Prop
  globalH1_holds : forall u0, globalH1 u0
  unique_holds : forall u0, unique u0
  smoothPositiveTime_holds : forall u0, smoothPositiveTime u0
```

In model/scaffold instances these predicates can be set to `True`. That is
structurally valid Lean, but it is not a proof of Burgers.

Current replacement: this class is absent from the remaining Burgers backend.
The concrete predicates live in `GroundTruthPDE.lean`.

### 3. The analytic upgrade currently has a shortcut

Removed problem:

```lean
class BurgersAnalyticUpgradePackage ... where
  analyticUpgrade :
    germSmallness ->
    universalBadInitialized ->
    catLibraryComplete ->
    BurgersBridgeInvariantStatement nu ->
    BurgersGlobalRegularityStatement nu
```

The default instance currently proves this by using the packaged evolution
regularity theorem. That is the hardcoded global-result-shaped dependency that
must be removed.

Required replacement:

Current replacement: `GroundTruthUpgrade.lean` now implements the final
analytic upgrade theorem as a genuine contradiction argument from a named
a-posteriori localization principle:

```lean
def BurgersAposterioriBadMorphismLocalization
    (nu : BurgersParameters)
    (input : BurgersAnalyticUpgradeInput nu) : Prop :=
  APosterioriLocalization
    PeriodicH1State.IsPeriodicH1
    (BurgersRegularityWitness nu)
    (BurgersBadMorphismExists nu input.bundle input.heat input.coleHopf)

theorem burgersAnalyticUpgrade_fromAposterioriLocalization
    (nu : BurgersParameters)
    (input : BurgersAnalyticUpgradeInput nu)
    (hlocalize : BurgersAposterioriBadMorphismLocalization nu input) :
    BurgersGroundTruthGlobalRegularityStatement nu
```

The proof assumes a missing regularity witness, localizes that failure to a bad
morphism by `hlocalize`, and contradicts `input.lock` by applying the reusable
framework theorem `APosterioriLocalization.target`. This is the intended
hypostructure upgrade shape. Lean now derives this localization theorem from a
decomposed local continuation/localization chain: certified local evolution,
literature-backed local uniqueness, the canonical obstruction criterion, a
maximal nonextendable window, and explicit bridge-transfer facts. The old monolithic
`burgers_axiom_failureLocalizesToWitness` boundary has been removed. The
canonical continuation criterion is fully defined in Lean: a window is
extendable exactly when it has no Burgers-side finite `H¹` obstruction data.
That obstruction data now contains the local germ, finite-time `H¹` evidence,
support in the Burgers window, positive obstruction time, explicit
nonregularity of the germ profile, inclusion of the Burgers obstruction window
in the certified heat window, and equality between the certified heat value and
the Cole-Hopf image at the obstruction time. The previous
chart/support/smoothness bridge axioms have been narrowed further: the
Cole-Hopf PDE-domain condition is now derived from the finite `H¹` obstruction
certificate plus the reusable literature theorem that periodic `H¹` Burgers
profiles lie in the transform PDE domain. Chart membership, heat-image support,
smoothness reflection, and heat-side forbiddenness are derived in Lean from
those narrower inputs plus the Cole-Hopf/heat interfaces.

The desired fully discharged theorem still has this local-certificate shape:

```lean
theorem analyticUpgradeFromLocalCertificates
    (nu : BurgersParameters)
    (hEnergy : LocalEnergyIdentity nu)
    (hPoincare : LocalPoincareCoercivity nu)
    (hMean : LocalMeanSectorPreservation nu)
    (hGerm : LocalBadGermCapacity nu)
    (hHeat : LocalHeatWindowCertificate nu)
    (hColeHopf : LocalColeHopfWindowBridge nu hHeat)
    (hLock : LockBlocksBurgersBadGerms nu) :
   BurgersGroundTruthGlobalRegularityStatement nu
```

No premise may be equivalent to the desired global regularity theorem.

### 4. `RealScaffold.lean` was structural, not the real PDE

Removed problem:

- `realBurgersFlow` is defined by Cole-Hopf/heat conjugacy;
- `RealBurgersDynamicsStatement` states that same conjugacy;
- it does not state the PDE residual
  `u_t + u * u_x = nu * u_xx`.

Current replacement: `RealScaffold.lean` was deleted from the Burgers backend.
The final theorem imports the ground-truth backend only.

## Required New Lean Modules

The exact file names can change, but the ground-truth path should be separated
from the model/scaffold path.

Implemented files:

```text
Hypostructure/Backends/Burgers1D/Parameters.lean
Hypostructure/Backends/Burgers1D/Torus.lean
Hypostructure/Backends/Burgers1D/GroundTruthState.lean
Hypostructure/Backends/Burgers1D/GroundTruthPDE.lean
Hypostructure/Backends/Burgers1D/GroundTruthWindows.lean
Hypostructure/Backends/Burgers1D/GroundTruthLocalAnalysis.lean
Hypostructure/Backends/Burgers1D/GroundTruthCertificates.lean
Hypostructure/Backends/Burgers1D/GroundTruthRun.lean
Hypostructure/Backends/Burgers1D/GroundTruthHeat.lean
Hypostructure/Backends/Burgers1D/GroundTruthColeHopf.lean
Hypostructure/Backends/Burgers1D/GroundTruthLock.lean
Hypostructure/Backends/Burgers1D/GroundTruthUpgrade.lean
Hypostructure/Backends/Burgers1D/GroundTruthFinal.lean
Hypostructure/Backends/Burgers1D/GroundTruthZeroEquilibrium.lean
Hypostructure/Backends/Burgers1D/GroundTruthConstantEquilibrium.lean
Hypostructure/Framework/Route.lean
Hypostructure/Framework/Upgrade.lean
```

No old Burgers scaffold files remain in the backend tree. The final
ground-truth claim lives in `GroundTruthFinal.lean`, but it is still conditional
on explicit backend theorem objects. The finished version should instantiate
those theorem objects with proved mathlib constructions, not add a parallel
scaffold route.

The currently proved nonconditional theorem is the restricted zero-equilibrium
instance in `GroundTruthZeroEquilibrium.lean`. It is intentionally separate from
`GroundTruthFinal.lean` because proving the zero curve solves Burgers is not the
same theorem as proving global regularity for arbitrary periodic H1 data.

## Phase 1: Real Periodic H1 Carrier

### Goal

Define the actual state space used by the theorem.

### Required definitions

Implement a periodic `H^1` carrier over `T = R / Z`, preferably using
`UnitAddCircle` or the existing `BurgersTorus`.

Required structure:

```lean
structure PeriodicH1State where
  value : T -> Real
  weakDeriv : T -> Real
  value_memL2 : MemLp value 2 volume
  deriv_memL2 : MemLp weakDeriv 2 volume
  weakDeriv_spec : IsWeakDerivativeOnTorus value weakDeriv
```

If mathlib lacks the exact weak derivative predicate needed on `UnitAddCircle`,
define the minimal local predicate using test functions and integration by
parts:

```lean
def IsWeakDerivativeOnTorus (u du : T -> Real) : Prop :=
  forall phi : SmoothPeriodicTestFunction,
    integral (fun x => u x * phi.deriv x)
      = - integral (fun x => du x * phi x)
```

Current Lean status: `SmoothPeriodicTestFunction` is no longer an untied pair
of continuous maps. It now carries a real-line lift, lifted derivative, periodic
compatibility laws with the torus maps, and a `HasDerivAt` certificate tying
`deriv` to `value`. Continuity and periodicity of both lifted functions are now
derived from the continuous torus maps plus quotient compatibility, rather than
stored as fields. The periodic boundary cancellation theorem
`SmoothPeriodicTestFunction.integral_deriv_zero` is derived from these facts,
mathlib's interval-integral representation of `UnitAddCircle`, and the
fundamental theorem of calculus. It is no longer an input field of the test
function interface.

### Required operations

- `mean : PeriodicH1State -> Real`
- `constantState : Real -> PeriodicH1State`
- `meanZeroPart : PeriodicH1State -> PeriodicH1State`
- `energy : PeriodicH1State -> Real`
- `dissipation : BurgersParameters -> PeriodicH1State -> Real`
- addition, scalar multiplication, negation, subtraction where needed
- proof that `u = meanZeroPart u + constantState (mean u)`
- proof that `mean (meanZeroPart u) = 0`

### Acceptance criteria

- No final theorem depends on the old pair
  `BurgersProfile × BurgersDerivative` unless it is proven equivalent to
  `PeriodicH1State`.
- The derivative field is tied to the value by `weakDeriv_spec`.
- All local certificate statements are phrased using the new carrier.

## Phase 2: Concrete PDE and Solution Predicates

### Goal

Define the actual Burgers equation in Lean.

### Required definitions

Define time-dependent solution curves:

```lean
structure BurgersSolutionCurve (nu : BurgersParameters) where
  eval : NNReal -> PeriodicH1State
  timeRegularity : Prop
```

Define a weak PDE residual:

```lean
def SolvesViscousBurgersWeak
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (u : BurgersSolutionCurve nu) : Prop :=
  InitialCondition u u0
    /\ PeriodicBoundary u
    /\ WeakBurgersResidual nu u
```

The weak residual should be an integrated identity against smooth periodic
test functions. A typical target shape is:

```lean
def WeakBurgersResidual
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu) : Prop :=
  forall phi : SmoothCompactTimePeriodicSpaceTest,
    timeSpaceIntegral
      (fun t x =>
        - u(t,x) * partial_t phi(t,x)
        - (u(t,x)^2 / 2) * partial_x phi(t,x)
        + nu.viscosity * partial_x u(t,x) * partial_x phi(t,x))
      = 0
```

The exact sign convention can vary, but it must be fixed and used
consistently.

### Required theorem statements

- `InitialCondition`
- `PeriodicBoundary`
- `WeakBurgersResidual`
- `UniqueWeakBurgersSolution`
- `GlobalH1Solution`
- `SmoothAtPositiveTime`
- strengthened `BurgersGroundTruthGlobalRegularityStatement`

### Acceptance criteria

- The theorem statement mentions the actual PDE residual.
- `solvesViscousBurgers` cannot be proved by `rfl` unless the residual is
  definitionally the PDE residual.
- No theorem in the final dependency cone uses `True` as the meaning of
  regularity or uniqueness.

## Phase 3: Local Energy Certificate

### Goal

Prove the energy identity/inequality as a local certificate, not as a global
regularity theorem.

### Required theorem

```lean
theorem localEnergyIdentity
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu)
    (hsol : SolvesViscousBurgersWeak nu u0 u) :
    LocalEnergyIdentity nu u
```

For smooth solutions, prove:

```lean
d/dt E(v(t)) + nu * ||partial_x v(t)||_L2^2 = 0
```

For weak solutions, prove the inequality version:

```lean
E(v(t2)) + ∫ t in t1..t2, dissipation(v(t)) <= E(v(t1))
```

The hypostructure node should only use a finite-window form.

### Acceptance criteria

- Certificate `K_D_E^+` contains the local identity or finite-window inequality.
- It does not state global existence.
- It does not state global smoothness.

## Phase 4: Local Poincare and Mean-Sector Certificates

### Goal

Prove the local coercivity and sector decomposition needed by stiffness,
topology, and tameness nodes.

### Required theorems

- `mean_constantState`
- `mean_meanZeroPart`
- `decompose_mean_zero_plus_constant`
- `localMeanSectorPreservation`
- `periodicPoincareMeanZero`
- `localDissipativeWindow`

Expected shapes:

```lean
theorem periodicPoincareMeanZero
    (u : PeriodicH1State)
    (hmean : mean u = 0) :
    l2NormSq u.value <= C * l2NormSq u.weakDeriv
```

```lean
theorem localMeanSectorPreservation
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu)
    (hsol : SolvesViscousBurgersWeak nu u0 u) :
    forall t in certifiedWindow, mean (u.eval t) = mean u0
```

### Acceptance criteria

- `K_LS_sigma^+`, `K_TB_pi^+`, `K_TB_O^+`, and `K_TB_rho^+` are backed by
  local statements.
- No node proves or assumes `Sigma = empty`.
- No node proves or assumes global convergence.

Current Lean status: the invalid proof of Poincare from an independently chosen
test derivative has been removed. `periodicH1_poincare_meanZero_literature` is
now a framework wrapper around the proved Lean theorem
`periodicH1_poincare_spectralGap_literature`. That theorem is assembled from
the bundled reusable `PeriodicH1FourierTheory` boundary, which packages value
Parseval with the zero mode removed, value-mode summability, derivative-mode
summability, derivative Parseval lower bound, and the weak-derivative Fourier
coefficient identity. Lean already proves the first-frequency gap and per-mode
energy control from the coefficient identity.

## Phase 5: Local Bad-Germ Capacity Certificate

### Goal

Replace any global singular-set claim with a local bad-germ support statement.

### Required definitions

```lean
structure BurgersBadGerm where
  centerTime : NNReal
  centerSpace : T
  localWindow : TimeSpaceCylinder
  profile : ...
```

```lean
def LocalBadGermCapacity (g : BurgersBadGerm) : Prop := ...
```

### Required theorem

```lean
theorem localBadGermCapacityCertificate
    (g : BurgersBadGerm)
    (hlocal : LocallyAdmissibleBurgersBadGerm g) :
    LocalBadGermCapacity g
```

This theorem should say the local support is capacity-small or otherwise
inadmissible for the route. It should not say no singularity exists globally.

### Acceptance criteria

- `K_Cap_H^+` is a statement about local bad-germ support.
- The final emptiness/no-blow-up conclusion appears only after the Lock and
  analytic upgrade.

## Phase 6: Periodic Heat Semigroup Backend

### Goal

Build the heat side used by the Cole-Hopf route.

### Required construction

Construct the requested finite-window heat certificate on the periodic carrier.
Possible implementation paths:

- Fourier series on `UnitAddCircle`;
- heat kernel on the torus;
- a proved abstract semigroup as an internal construction source, restricted to
  `periodicHeat1D_localWindowCertificate_literature` before entering the final
  Burgers route.

### Required certificates

- local heat zero-time law on a certified window;
- local heat additivity/restriction law for composable certified windows;
- local mean preservation;
- constants fixed;
- finite-window energy contraction;
- finite-window dissipation contraction;
- forward-time local heat uniqueness on a certified window;
- local heat smoothing for positive time inside the certified window;
- exclusion of heat-side bad-germ windows needed by E5.

### Implemented interface refactor

The old global-shaped `LocalHeatSmoothingAndUniqueness` record has been split
into a windowed certificate and a separate semigroup construction source:

```lean
structure HeatWindow where
  t0 : ℝ
  t1 : ℝ
  ordered : t0 ≤ t1

structure LocalHeatWindowCertificate (nu : BurgersParameters) where
  certified : CertifiedHeatWindow nu
  residual : HeatWindowResidual nu certified
  unique : HeatWindowUniqueness nu certified
  smooth : HeatWindowSmoothing certified
  energy_contraction : HeatWindowEnergyContraction certified
  dissipation_contraction : HeatWindowDissipationContraction certified
  excludes_heat_bad_germs : ...

structure PeriodicHeatSemigroupBackend (nu : BurgersParameters) where
  -- construction source, restricted to a LocalHeatWindowCertificate before any
  -- fact is exported as K_HeatSmooth^+; it contains only the fields consumed by
  -- the local route: flow/curve compatibility, weak heat solving, uniqueness,
  -- positive-time smoothing, and window-restrictable contractions.
  ...
```

`CertifiedHeatWindow` itself is local: it no longer contains a global weak heat
solution package. Global heat theory may still be used to construct one, but the
certificate boundary stores only the certified window, curve, initial datum, and
window-local boundary/initial facts.

A full periodic heat semigroup may still be proved as a backend theorem, but
`K_HeatSmooth^+` should expose only the local/windowed facts consumed by the
sieve and Lock. If a global semigroup theorem is used internally to prove those
local facts, it must be treated as a heat backend theorem, not as Burgers global
regularity and not as a premise equivalent to the Burgers target.

### Required theorem

```lean
def periodicHeat1D_l2ReconstructionTheory_literature : ...
def periodicHeat1D_fourierH1ReconstructionTheory_literature : ...
def periodicHeat1D_positiveTimeCoefficientSmoothingTheory_literature : ...
def periodicHeat1D_positiveTimeSmoothFourierRepresentativeTheory_literature : ...
def periodicHeat1D_positiveTimeSmoothPeriodicH1RepresentativeTheory_literature : ...

def periodicHeat1D_fourierH1CandidateHeatCurve : ...
theorem periodicHeat1D_candidateValueCoeff_fromPositiveTimeIdentification : ...
theorem periodicHeat1D_candidateDerivativeCoeff_fromPositiveTimeIdentification : ...
def periodicHeat1D_fourierH1CandidateIdentification_literature : ...
def periodicHeat1D_fourierH1ContinuousCurve_literature : ...
def periodicHeat1D_fourierH1IntegratedClassicalHeat_literature : ...
theorem periodicHeat1D_fourierH1WindowResidual_literature : ...
structure PeriodicHeat1DWeakSolutionValueFourierEvolutionFor : ...
structure PeriodicHeat1DWeakSolutionFourierEvolutionFor : ...
def periodicHeat1D_fourierH1CandidateHeatCurve_fourierEvolution : ...
axiom periodicHeat1D_weakSolutionValueFourierEvolution_literature : ... -- arbitrary weak solution value-coefficient evolution
theorem periodicHeat1D_weakSolutionFourierEvolution_literature : ... -- derived full coefficient evolution
theorem periodicHeat1D_fourierH1WindowUniqueness_fromWeakSolutionFourierEvolution : ...
theorem periodicHeat1D_fourierH1WindowUniqueness_literature : ... -- derived forward-time uniqueness
def periodicHeat1D_fourierH1WindowSmoothing_fromContinuousCurve : ...
def periodicHeat1D_fourierH1WindowUpgrade_literature : ...
def periodicHeat1D_fourierReconstruction_literature : ...

def periodicHeat1D_localTheory_literature : ...

def periodicHeat1D_localExistenceWindow_literature : ...
def periodicHeat1D_localResidual_literature : ...
def periodicHeat1D_localUniqueness_literature : ...
def periodicHeat1D_localSmoothing_literature : ...
def periodicHeat1D_localContraction_literature : ...
theorem periodicHeat1D_badGermExclusion_fromSmoothing : ...
def periodicHeat1D_localWindowCertificate_literature : ...
```

Current Lean status: the interface and the windowed literature boundary are
implemented. `PeriodicHeatSemigroupBackend.windowCertificate` remains available
as a construction-source restriction theorem, but the final Burgers route now
imports only the assembled `periodicHeat1D_localWindowCertificate_literature`.
The scalar Fourier heat multiplier layer and coefficient-level `ℓ²`
contraction/summability layer are now proved in Lean, and so is the `L²`
Fourier reconstruction layer. Lean constructs the value and weak-derivative
`L²` reconstructions, proves their Fourier coefficients are exactly the evolved
coefficients, proves their Fourier series converge in `L²`, and proves Parseval
for those reconstructed `L²` objects. Lean also constructs the Fourier/Sobolev
`H¹` state from those objects and proves the Fourier weak-derivative relation,
value/derivative Parseval identities, and finite-time contractions on that
carrier. Lean also proves the coefficient-level positive-time smoothing layer:
polynomially weighted heat multipliers and evolved value/derivative mode
energies are summable for every positive time. Lean also proves the
positive-time smooth Fourier representative package that combines those
summability facts with the reconstructed Fourier-`H¹` state, evolved
coefficient identities, Parseval, and contractions. That package now also
proves weighted `ℓ¹` coefficient summability and constructs continuous complex
Fourier-series slices with pointwise convergence, a full complex
spatial-derivative tower, and termwise classical differentiability of that
tower. Lean also proves the real positive-time `PeriodicH1State`
representative and its recursive weak-derivative smoothness chain by taking
real parts and applying the generic lifted-derivative-to-weak-derivative
bridge. Lean also proves real-valued Fourier conjugate symmetry and that the
heat multiplier preserves it for evolved value and weak-derivative
coefficients. Lean now also proves positive-time candidate coefficient
identification for the explicit heat curve from the continuous Fourier
representative and Hilbert-basis coefficient recovery. The remaining math is
now the PDE-semantics upgrade for the arbitrary-data continuous `H¹` heat curve
on each certified local window, split into two explicit facts: weak residual
and uniqueness at the concrete window/PDE level. Zero-time coefficient
identification, Parseval/energy identification, and positive-time smoothing are
derived in Lean from coefficient identities. The older `periodicHeat1D_fourierH1WindowUpgrade_literature` and
`periodicHeat1D_fourierReconstruction_literature` names are now Lean
compatibility definitions, not axioms.

### Acceptance criteria

- `K_HeatSmooth^+` is backed by this theorem.
- The theorem does not import Burgers global regularity.
- The exported certificate is local/windowed. It must not require the final
  Burgers global theorem, a global empty-singularity theorem, or a global
  Burgers continuation statement.
- Any all-time heat semigroup theorem is hidden behind local restriction lemmas
  before it reaches `K_HeatSmooth^+`.

Current partial proof: `GroundTruthZeroEquilibrium.lean` proves its restricted
heat facts directly. `GroundTruthConstantEquilibrium.lean` now packages the
constant heat facts as `constantLocalHeatWindowCertificate`, so the constant
route exercises the same local heat certificate boundary used by the general
framework. The arbitrary-data finite-window heat theorem remains unimplemented;
if it is proved via an all-time semigroup, that semigroup must enter the final
route only through `periodicHeat1D_localWindowCertificate_literature` and
`LocalHeatWindowCertificate`.

## Phase 7: Cole-Hopf Bridge Certificate

### Goal

Prove the bridge used by E5 in the Lock.

### Required definitions

- Cole-Hopf transform on the mean-zero or mean-adjusted sector;
- inverse Cole-Hopf;
- positivity/nonvanishing condition for the heat variable;
- moving-frame or mean-sector correction for nonzero mean;
- compatibility with periodicity;
- certified Burgers windows and heat windows;
- chart-validity predicates saying the logarithm/positive heat chart is valid
  on the window;
- local residual-transfer predicates between a Burgers window and its heat-side
  image;
- local bad-germ transport predicates between Burgers bad-germ windows and
  heat bad-germ windows.

### Implemented interface refactor

The old `LocalColeHopfBridge` has been changed from all-state/all-curve
transfer to a windowed bridge. The implemented shape is:

```lean
structure LocalColeHopfWindowBridge
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) where
  transform : ColeHopfTransform
  chart_valid : ...
  maps_window : ...
  inverse_maps_window : ...
  burgers_to_heat_residual_on_window : ...
  heat_to_burgers_residual_on_window : ...
  uniqueness_transfer_on_window : ...
  transports_bad_germs : ...
```

The existing global-shaped `ColeHopfBurgersToHeatResidualTransfer`,
`ColeHopfHeatToBurgersResidualTransfer`, and `ColeHopfUniquenessTransfer` can
remain as backend convenience theorems only if they are not exposed as the
certificate payload. Before reaching `K_ColeHopf^+`, they must be restricted to
the concrete certified windows and bad-germ charts emitted by the route.

### Required theorem

```lean
theorem localColeHopfWindowBridge
    (nu : BurgersParameters) :
    (H : LocalHeatWindowCertificate nu) →
      LocalColeHopfWindowBridge nu H
```

The theorem should include:

- transform maps certified Burgers local windows to heat local windows;
- inverse maps positive heat windows back to Burgers windows;
- transform and inverse agree on the relevant sector;
- Burgers weak residual transfers to heat residual;
- heat residual transfers back to Burgers residual;
- uniqueness transfers through the transform;
- bad-germ transport: every route-relevant Burgers bad-germ window maps to a
  heat bad-germ window excluded by `K_HeatSmooth^+`.

### Acceptance criteria

- `K_ColeHopf^+` is not an identity-transform scaffold.
- The Lock uses this bridge to map bad Burgers germs into heat-side bad germs.
- The theorem is local/windowed at the certificate boundary.
- It does not prove or assume global Burgers existence, global Burgers
  uniqueness, global Burgers smoothness, or global emptiness of the singular
  set.

Current partial proof: the framework language is implemented. The constant
equilibrium route now constructs a `PeriodicColeHopfBackend`, restricts it to a
`LocalColeHopfWindowBridge`, proves the bridge statement, and uses it with the
local heat certificate to prove the finite-library Lock theorem. The backend no
longer assumes bad-germ transport separately: `windowBridge` derives it from
H¹ preservation of the transform. A genuine arbitrary-data Cole-Hopf bridge is
still required for the classical theorem.

## Phase 8: Lock From Local Certificates

### Goal

Prove that the bad morphism is blocked using local certificates.

### Required definitions

```lean
def BurgersBadMorphismExists (nu : BurgersParameters) : Prop := ...

def LockBlocksBurgersBadGerms (nu : BurgersParameters) : Prop :=
  not (BurgersBadMorphismExists nu)
```

### Required theorem

```lean
theorem lockBlocksBadGermsFromLocalCertificates
    (nu : BurgersParameters)
    (bundle : BurgersGroundTruthCoreCertificateBundle nu)
    (hHeat : LocalHeatWindowCertificate nu)
    (hCole : LocalColeHopfWindowBridge nu hHeat) :
    LockBlocksBurgersBadGerms nu bundle hHeat hCole
```

### Required proof shape

The Lock proof must be a contradiction on a local bad morphism candidate:

1. take a route-relevant finite-time `H¹` Burgers bad-germ candidate;
2. classify it through `bundle.badPatternLibrary.complete`;
3. derive route-local admissibility from the matched bad-pattern object;
4. use the local Cole-Hopf bridge to transport that window to the heat side;
5. use the local heat certificate to exclude the heat-side bad-germ window;
6. conclude that the original Burgers bad morphism cannot exist.

No step is allowed to cite the final Burgers global theorem. The Lock proves a
structural exclusion certificate, not global regularity itself.

### Acceptance criteria

- `K_Cat_Hom^blk` is proved from local certificates.
- The proof does not mention `BurgersGroundTruthGlobalRegularityStatement` or
  any old scaffold global regularity theorem.
- The proof does not use a global empty singular set theorem.
- The proof consumes `K_ColeHopf^+` and `K_HeatSmooth^+` only through their
  windowed bad-germ transport/exclusion fields.

## Phase 9: Analytic Upgrade From Local Certificates

### Goal

Replace the current hardcoded upgrade package with a real theorem.

### Required theorem

```lean
theorem analyticUpgradeFromLocalCertificates
    (nu : BurgersParameters)
    (hEnergy : LocalEnergyIdentity nu)
    (hPoincare : LocalPoincareCoercivity nu)
    (hMean : LocalMeanSectorPreservation nu)
    (hCapacity : LocalBadGermCapacity nu)
    (hHeat : LocalHeatWindowCertificate nu)
    (hCole : LocalColeHopfWindowBridge nu hHeat)
    (hLock : LockBlocksBurgersBadGerms nu) :
    BurgersGroundTruthGlobalRegularityStatement nu
```

This is where global regularity is assembled. It is allowed to conclude the
global theorem here because all inputs are local certificates plus Lock
exclusion.

### Required proof shape

The analytic upgrade should be the only theorem that turns local certificates
into the classical global statement. Its proof should proceed by the standard
hypostructure contradiction route:

1. assume the classical target fails for the chosen initial datum;
2. localize the failure to a finite-time route-relevant bad-germ window using
   the bad-pattern library and local compactness/representation certificates;
3. use the sieve certificate chain to show the localized failure lies in the
   Lock target class;
4. apply `K_Cat_Hom^blk` to rule out that localized failure;
5. construct the global solution/regularity witness from the remaining local
   continuation certificates and heat/Cole-Hopf backend restrictions.

This theorem may invoke backend heat and Cole-Hopf theorems, but only after
they have been converted into local certificate payloads. It must not take a
field whose statement is already `BurgersGroundTruthGlobalRegularityStatement`.

### Acceptance criteria

- `BurgersAnalyticUpgradePackage` is either removed from the final path or its
  only instance is this theorem.
- The proof of `BurgersGroundTruthGlobalRegularityStatement` does not call a
  package field that already states global regularity.
- The dependency cone contains local certificates, heat/Cole-Hopf bridge, Lock,
  and the upgrade theorem.
- `K_ColeHopf^+` and `K_HeatSmooth^+` appear as local/windowed inputs. Their
  statements are strictly weaker than the final Burgers theorem.

## Phase 10: Ground-Truth Final Theorem

### Goal

Expose the final theorem that the user-facing proof should cite.

### Required final theorem

Current migrated shape:

```lean
def burgers_groundTruth_dataset_theorem
    (nu : BurgersParameters)
    (bundle : BurgersGroundTruthCoreCertificateBundle nu)
    (permits : BurgersGroundTruthBackendPermits nu) :
    BurgersGroundTruthDatasetClaim nu
```

Required finished shape:

```lean
def burgers_groundTruth_dataset_theorem
    (nu : BurgersParameters)
    (hnu : 0 < nu.viscosity) :
    BurgersGroundTruthDatasetClaim nu := by
  -- construct local certificates
  -- run/verify sieve certificate route
  -- prove Lock
  -- apply analytic upgrade
```

The final claim should include:

- concrete state-space theorem;
- concrete PDE solution theorem;
- global regularity theorem;
- proof that `K_Reg_Burgers1D^+` is reached by the hypostructure run.

### Acceptance criteria

- `GroundTruthFinal.lean` builds.
- It imports no model/scaffold theorem as semantic evidence.
- It does not import `ModelInstance.lean`.
- It does not use `burgersModelPDEEvolutionPackageData`.
- It does not use the old `BurgersEvolutionRegularity` trivial fields.

## Required Import Discipline

The ground-truth final path must not depend on:

- `Hypostructure.Backends.Burgers1D.ModelInstance`
- model flow regularity proofs;
- scaffold-only `RealBurgersDynamicsStatement` as the PDE theorem;
- any theorem whose proof is just `trivial` because a regularity predicate was
  defined as `True`.

Allowed transitional imports:

- core hypostructure engine;
- certificate structures;
- mathlib analysis modules;
- locally proven heat/Cole-Hopf/Burgers certificate modules.

## Required CI/Search Checks

The following checks should pass for the ground-truth path:

```bash
rg "\bsorry\b|\baxiom\b" lean/Hypostructure/Backends/Burgers1D/GroundTruth*.lean

rg "ModelInstance|modelFlow|burgersModel|modelColeHopf|modelPeriodicHeat" \
  lean/Hypostructure/Backends/Burgers1D/GroundTruth*.lean

rg "globalH1 := fun .* True|unique := fun .* True|smoothPositiveTime := fun .* True" \
  lean/Hypostructure/Backends/Burgers1D/GroundTruth*.lean

cd lean && lake build Hypostructure.Backends.Burgers1D.GroundTruthFinal
```

The broader repository may still contain non-Burgers examples, but the Burgers
backend tree no longer contains the old scaffold route. The ground-truth theorem
must continue not to depend on any reintroduced scaffold module.

## Implementation Order

Recommended sequence from the current migrated state:

1. Completed: introduce explicit certified window types: `BurgersWindow`,
   `HeatWindow`, route-local bad-germ windows, and restriction lemmas for
   Burgers and heat solution curves on those windows.
2. Completed: refactor `LocalHeatSmoothingAndUniqueness` into a local
   `LocalHeatWindowCertificate`. Keep any all-time heat semigroup theorem as a
   backend construction theorem, and export only windowed heat facts to the
   certificate chain.
3. Completed: refactor `LocalColeHopfBridge` into
   `LocalColeHopfWindowBridge`, with explicit chart-validity, window mapping,
   inverse transfer, residual transfer on certified windows, uniqueness
   transfer, and bad-germ transport fields. Add
   `PeriodicColeHopfBackend.windowBridge` as the backend construction source.
4. In progress: instantiate the periodic heat backend on `PeriodicH1State` and prove the
   requested local window certificate from mathlib-backed Fourier/semigroup
   analysis. The mode-level heat multiplier theory, coefficient-level `ℓ²`
   contraction/summability theory, `L²` Fourier reconstruction theory, and
   Fourier/Sobolev `H¹` reconstruction theory are proved in Lean. The
   positive-time coefficient smoothing layer is also proved in Lean: polynomial
   frequency weights remain summable after positive heat time for the multiplier
   and for evolved value/derivative mode energies. The positive-time smooth
   Fourier representative package is now also proved in Lean, assembling the
   reconstructed Fourier-`H¹` state with evolved coefficient identities,
   Parseval/contraction facts, and all polynomial weighted summability facts.
   It also proves weighted `ℓ¹` coefficient summability and constructs
   continuous complex Fourier-series slices, the full complex spatial-derivative
   tower, and termwise classical differentiability. The real-valued
   positive-time `PeriodicH1State` representative theorem is now also proved in
   Lean, including the recursive weak-derivative smoothness chain. Lean also
   proves positive-time and zero-time candidate coefficient matching and the
   candidate energy/Parseval match from coefficient identities plus Parseval.
   The next missing heat work is split into the integrated classical heat
   certificate feeding the weak-residual theorem and the windowed uniqueness
   theorem for the reconstructed Fourier-`H¹` object. The final route now
   exposes only those two heat PDE-semantics facts as heat axioms; local
   existence, residual, uniqueness, smoothing, contraction, and certificate
   exports are Lean projections/assemblers. An
   arbitrary-data heat semigroup is useful as a construction theorem but is no
   longer in the theorem axiom boundary. Heat bad-germ exclusion is already
   proved from smoothing by
   `periodicHeat1D_badGermExclusion_fromSmoothing`.
5. In progress: instantiate the concrete periodic Cole-Hopf transform and inverse on the
   ground-truth carrier, including the mean-sector/moving-frame correction and
   positivity chart needed for the local bridge.
   The backend-source interface and constant-sector bridge are proved. The
   arbitrary-data Cole-Hopf boundary is now one reusable theory package,
   `periodicBurgers1D_coleHopfTheory_literature`; its transform, chart, window,
   residual-transfer, uniqueness-transfer, and H¹-domain exports are Lean
   projections from that package. The nonconstant periodic Cole-Hopf construction
   remains to be formalized inside that package.
6. Completed framework-side and decomposed into literature boundaries:
   `GroundTruthCoreFactory.lean` provides an axiom-free canonical zero core
   bundle, a reusable `toBundle` constructor, and the pointwise windowed
   `BurgersCorePDELocalWindowInputsFor` interface. The arbitrary-data local
   well-posedness/local-estimate boundary now lives in
   `Hypostructure.Literature.Burgers.Periodic1D` as one reusable local theory
   source, `periodicBurgers1D_localWindowTheory_literature`. Lean proves the
   Poincare/coercivity wrapper `periodicH1_poincare_meanZero_literature` from
   the Lean-assembled spectral-gap theorem
   `periodicH1_poincare_spectralGap_literature`, which now depends only on the
   reusable `PeriodicH1FourierTheory` package discharged from mathlib Fourier
   analysis and the carrier's weak-derivative interface. The former Fourier
   literature boundary `periodicH1_fourierTheory_literature` is now a Lean
   definition, not an axiom. Lean proves the local existence, residual,
   energy, mean-preservation, and dissipative projections from the single
   Burgers local theory source, and then proves the assemblers
   `periodicBurgers1D_meanPoincare_literature` and
   `periodicBurgers1D_localWindowInputs_literature`, which produce the exact
   core package consumed by the route. The old global-shaped adapter has been
   removed: the current core bundle consumes a certified Burgers window and
   `SolvesViscousBurgersWeakOnWindow`.
7. Completed for the current dataset: `LockBlocksBurgersBadGerms` now uses the
   full documented finite-time `H¹` bad-pattern library and the `K_Germ^+`,
   `K_init^+`, `K_CatLib^+` certificates. Extend this only if the dataset adds
   more bad-pattern kinds beyond finite-time `H¹` blow-up.
8. Implemented with an explicit decomposed axiom boundary and reusable framework theorem:
   `burgersAnalyticUpgrade_fromAposterioriLocalization` instantiates
   `APosterioriLocalization.target` to prove
   `BurgersGroundTruthGlobalRegularityStatement` from the local certificate
   bundle, local heat certificate, local Cole-Hopf bridge, Lock theorem, and
   the a-posteriori localization principle. Lean derives this principle from
   `BurgersLocalContinuationLocalizationChain.localizedFailureWitness` and
   `BurgersLocalizedFailureWitness.toBadMorphismCandidate`. The remaining work
   is to discharge the smaller local PDE facts listed below from compactness,
   representation, bad-pattern completeness, continuation analysis, and local
   Cole-Hopf/heat obstruction transfer.
9. Remove `BurgersGroundTruthBackendPermits` from
   `burgers_groundTruth_dataset_theorem` once the backend theorem objects are
   constructed internally.
10. Run the ground-truth import/search checks and keep the migrated route as the
   only documented theorem path.

## Definition of Done

The work is complete only when:

- `GroundTruthFinal.lean` proves the classical Burgers theorem;
- the proof route emits local certificates matching `docs/source/dataset/burgers_1d.md`;
- the final global theorem is assembled only at the analytic upgrade step;
- no final theorem depends on a hardcoded global regularity package field;
- no final theorem depends on model/scaffold flow data;
- no `sorry` or `axiom` appears in the ground-truth dependency cone;
- the proof object still verifies that the hypostructure template route reaches
  `K_Reg_Burgers1D^+`.

## Current Practical Milestone

The next code milestone is no longer to create the heat/Cole-Hopf/Lock/upgrade
interfaces or a no-axiom sanity instance. Those exist and compile, and the
zero-equilibrium plus all-constant routes prove template-shaped restricted
theorems. The constant route now also constructs the local heat certificate,
local Cole-Hopf bridge, Lock theorem, and analytic-upgrade input through the
same reusable framework objects. The next milestone is to extend the first
nontrivial theorem object beyond constant curves with real mathlib-backed
analysis.

Recommended next target:

```text
mathlib-backed periodic heat backend on PeriodicH1State
```

The certified window language, local heat certificate, local Cole-Hopf bridge,
construction-source restriction records, finite-library Lock, and constant
route replay are now implemented. The final analytic upgrade is now pointwise:
`BurgersPointwiseLocalEstimateProvider` supplies both a local hypostructure
package and a local continuation/localization chain for the exact admissible
initial datum `u0`. `APosterioriLocalization.point` turns that pointwise chain
plus the Lock into a regularity witness for the same `u0`, and
`burgers_globalRegularity_from_pointwise_local_estimates` assembles the global
theorem by iterating over all `u0`. The temporary closed theorem is available
as `burgers_groundTruth_dataset_theorem_from_axioms`.

Current axiom audit:

```text
#print axioms burgers_globalRegularity_from_pointwise_local_estimates
  custom axioms: none
  Lean foundations: propext, Classical.choice, Quot.sound

#print axioms burgers_groundTruth_dataset_theorem_from_axioms
  custom axioms: exactly the named boundary below
  Lean foundations: propext, Classical.choice, Quot.sound
```

The kernel-level equality between the declared boundary and the theorem's
actual custom axioms is now checked by the Lake script target:

```bash
lake run auditBurgersAxioms
```

The script runs Lean, prints the theorem axioms, extracts the custom
`Hypostructure.*` constants, drops only the known Lean foundations, and fails if
the result differs from `burgersGroundTruthAxiomBoundary`.

The current named axiom boundary is:

```text
periodicBurgers1D_localWindowTheory_literature
periodicHeat1D_weakSolutionValueFourierEvolution_literature
periodicBurgers1D_coleHopfTheory_literature
periodicBurgers1D_continuationTheory_literature
```

The current rigor split is:

```text
F: APosterioriLocalization.point
F: burgers_globalRegularity_from_pointwise_local_estimates
F: periodicHeat1D_fourierModeTheory_literature
F: periodicHeat1D_fourierCoefficientTheory_literature
F: periodicHeat1D_l2ReconstructionTheory_literature
F: periodicHeat1D_fourierH1ReconstructionTheory_literature
F: periodicHeat1D_positiveTimeCoefficientSmoothingTheory_literature
F: periodicHeat1D_positiveTimeSmoothFourierRepresentativeTheory_literature
F: periodicHeat1D_positiveTimeSmoothPeriodicH1RepresentativeTheory_literature
F: periodicH1_weakDerivative_fourierCoeff_allModes_literature
F: periodicHeat1D_l2FourierSeries_fourierCoeff
F: periodicHeat1D_continuousSpatialDerivativeFourierSeries_fourierCoeff
F: periodicHeat1D_realSpatialDerivativeFourierSeries_fourierCoeff
F: realContinuousMap_fourierCoeff_neg_eq_star
F: complexContinuousMap_ext_fourierCoeff
F: realContinuousMap_ext_fourierCoeff
F: periodicH1State_ext_fourierCoeff
F: periodicHeat1D_evolvedValueFourierCoeff_neg_eq_star
F: periodicHeat1D_evolvedDerivativeFourierCoeff_neg_eq_star
F: periodicHeat1D_modeMultiplier_hasDerivAt
F: periodicHeat1D_evolvedTimeDerivativeFourierCoeff_eq_viscosity_secondSpatial
F: periodicHeat1D_modeMultiplier_add_time
F: periodicHeat1D_modeMultiplier_le_one
F: periodicH1_value_parseval_literature
F: periodicH1_derivative_parseval_literature
F: periodicHeat1D_localTheory_fromSemigroupBackend
F: periodicHeat1D_localContraction_fromFourierReconstruction
F: periodicHeat1D_localTheory_fromFourierReconstruction
F: periodicHeat1D_localWindowCertificate_literature
F: periodicHeat1D_badGermExclusion_fromSmoothing
F: periodicHeat1D_fourierH1CandidateHeatCurve
F: periodicHeat1D_candidateValueCoeff_fromPositiveTimeIdentification
F: periodicHeat1D_candidateDerivativeCoeff_fromPositiveTimeIdentification
F: PeriodicHeat1DWeakSolutionFourierEvolutionFor
F: PeriodicHeat1DWeakSolutionValueFourierEvolutionFor
F: periodicHeat1D_fourierH1CandidateHeatCurve_fourierEvolution
F: periodicHeat1D_fourierH1CandidateIdentification_literature
F: periodicHeat1D_fourierH1ContinuousCurve_literature
F: IntegratedClassicalHeatWindowCertificate
F: HeatWindowResidual.of_integratedClassical
F: periodicHeat1D_fourierH1IntegratedClassicalHeat_literature
F: periodicHeat1D_fourierH1WindowResidual_literature
L: periodicHeat1D_weakSolutionValueFourierEvolution_literature
F: periodicHeat1D_weakSolutionFourierEvolution_fromValueEvolution
F: periodicHeat1D_weakSolutionFourierEvolution_literature
F: periodicHeat1D_fourierH1WindowUniqueness_fromWeakSolutionFourierEvolution
F: periodicHeat1D_fourierH1WindowUniqueness_literature
F: periodicHeat1D_fourierH1WindowSmoothing_fromContinuousCurve
F: periodicHeat1D_fourierH1WindowUpgrade_literature
F: periodicHeat1D_fourierReconstruction_literature
F: periodicHeat1D_localTheory_literature
F: periodicHeat1D_localExistenceWindow_literature
F: periodicHeat1D_localResidual_literature
F: periodicHeat1D_localUniqueness_literature
F: periodicHeat1D_localSmoothing_literature
F: periodicHeat1D_localContraction_literature
F: periodicBurgers1D_coleHopfBackend_literature
L: periodicBurgers1D_coleHopfTheory_literature
F: periodicBurgers1D_coleHopfTransform_literature
F: periodicBurgers1D_coleHopfChartValid_literature
F: periodicBurgers1D_coleHopfMapsWindow_literature
F: periodicBurgers1D_coleHopfInverseMapsWindow_literature
F: periodicBurgers1D_coleHopfBurgersToHeatResidual_literature
F: periodicBurgers1D_coleHopfHeatToBurgersResidual_literature
F: periodicBurgers1D_coleHopfUniquenessTransfer_literature
F: periodicBurgers1D_h1InBurgersPDEDomain_literature
F: periodicBurgers1D_localWindowInputs_literature
L: periodicBurgers1D_localWindowTheory_literature
F: periodicBurgers1D_localExistenceWindow_literature
F: periodicBurgers1D_localResidual_literature
F: periodicBurgers1D_localEnergy_literature
F: periodicBurgers1D_meanPoincare_literature
F: periodicBurgers1D_poincare_literature
F: periodicH1_poincare_meanZero_literature
F: periodicH1_poincare_spectralGap_literature
F: periodicH1_firstFrequencyGap
F: periodicH1_valueModeEnergy_le_derivativeModeEnergy
F: periodicH1_fourierTheory_literature
F: periodicH1_value_parseval_meanZero_literature
F: periodicH1_value_nonzeroModeEnergy_summable_literature
F: periodicH1_derivative_nonzeroModeEnergy_summable_literature
F: periodicH1_derivative_parseval_lowerBound_literature
F: periodicH1_weakDerivative_fourierCoeff_literature
F: periodicBurgers1D_meanPreservation_literature
F: periodicBurgers1D_localDissipative_literature
L: periodicBurgers1D_continuationTheory_literature
F: periodicBurgers1D_localUniquenessOnWindow_literature
F: periodicBurgers1D_missingRegularityProducesRawFailureWindow_literature
F: periodicBurgers1D_rawFailureWindow_nonextendable_literature
F: burgers_localUniquenessOnOverlapsFromLiterature
F: BurgersRawFailureWindow.finiteH1Obstruction_of_not_extendable
F: periodicBurgers1D_missingRegularityProducesObstructionWindow_fromContinuationLiterature
F: BurgersFiniteH1BadGermObstruction.coleHopfDomainWitness_from_finiteH1
F: BurgersFiniteH1BadGermObstruction.obstructionWindowSupportedInHeatWindow
F: BurgersFiniteH1BadGermObstruction.heatImageMatchesAtCenter
```

The next phase is to discharge the reusable literature facts with mathlib-backed
analysis modules. The heat boundary has already been narrowed past scalar
Fourier estimates, coefficient `ℓ²` estimates, `L²` Hilbert-basis
reconstruction, Fourier/Sobolev `H¹` reconstruction, and coefficient-level
positive-time smoothing with arbitrary polynomial frequency weights. Lean also
now has the positive-time Fourier representative package collecting the
reconstructed Fourier-`H¹` state, evolved coefficients, Parseval/contraction,
all polynomial weighted summability facts, weighted `ℓ¹` coefficient
summability, continuous complex Fourier-series slices, the complex
spatial-derivative tower, and termwise differentiability. It also has the
real-valued positive-time `PeriodicH1State` representative and recursive
weak-derivative smoothness proof. Lean also proves real Fourier conjugate
symmetry, preservation of that symmetry by the heat multiplier, zero-time
candidate coefficient matching, and candidate energy/Parseval matching from
coefficient identities. The explicit candidate heat curve is now a Lean
definition, and its positive-time candidate coefficient identification theorem
is proved in Lean. The weak residual is now a Lean theorem from a reusable
integrated-classical heat interface. The remaining heat package is the single
value-coefficient-evolution literature fact
`periodicHeat1D_weakSolutionValueFourierEvolution_literature`; full
weak-derivative coefficient evolution is derived from that fact plus the
`PeriodicH1State` weak-derivative Fourier identity, and canonical windowed
uniqueness is derived from full coefficient evolution plus Fourier
extensionality for the concrete `PeriodicH1State` carrier. Concrete-window
smoothing is derived from
the candidate's positive-time representative. The local
heat theory, contraction facts, decomposed local-window projections, and
`periodicHeat1D_localWindowCertificate_literature` are assembled in Lean from
that fact plus the proved coefficient-level, `L²`, and Fourier-`H¹`
reconstruction/smoothing facts.
The Cole-Hopf boundary has
also been narrowed to the single reusable package
`periodicBurgers1D_coleHopfTheory_literature`, with its old component exports
proved as projections. The Burgers continuation/localization boundary has also
been narrowed to the single reusable package
`periodicBurgers1D_continuationTheory_literature`, with local uniqueness,
missing-regularity localization, and raw-window nonextendability exported as
projections. The problem-specific boundary is now empty: all remaining custom
assumptions are reusable periodic Burgers local theory, periodic Burgers
continuation theory, periodic `H¹` Fourier, heat, or Cole-Hopf literature facts.
The largest Burgers-side literature fact left is the single arbitrary-data local
Burgers window theory. Lean then extracts the
structured finite `H¹` obstruction and assembles the localized obstruction
window. Local uniqueness follows from the reusable periodic Burgers literature
theorem. The Cole-Hopf PDE-domain condition for finite `H¹` obstructions follows
from the literature Cole-Hopf `H¹`-domain theorem, while the localized Burgers
obstruction data carries the Burgers-window-to-heat-window inclusion and center
image matching facts as bookkeeping invariants. The framework projects those
facts by theorem rather than assuming separate bridge axioms. The framework
already derives nonextendable-window construction, chart membership, heat-image
support, smoothness reflection, and heat-forbiddenness from those narrower
facts.

`GroundTruthAudit.lean` records the current trust-boundary checkpoint as
`burgersGroundTruthAxiomAudit`. Lean proves
`burgersGroundTruthAudit_no_problem_specific`, showing that the current
problem-specific assumption boundary is empty, and
`burgersGroundTruthAudit_named_axiom_is_literature`, showing that every named
custom assumption in `burgersGroundTruthAxiomBoundary` is classified as reusable
literature. The complementary theorem
`burgersGroundTruthAudit_literature_is_named_axiom` shows the audit does not
omit any declared literature boundary item. Kernel-level verification is now
automated by `lake run auditBurgersAxioms`, which compares the actual
`#print axioms burgers_groundTruth_dataset_theorem_from_axioms` output against
`burgersGroundTruthAxiomBoundary`.
