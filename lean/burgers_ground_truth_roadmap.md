# Burgers 1D Ground-Truth Lean Roadmap

Date: 2026-04-16

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
Hypostructure/Backends/Burgers1D/GroundTruthZeroEquilibrium.lean
Hypostructure/Backends/Burgers1D/GroundTruthConstantEquilibrium.lean
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
  missing-regularity-witness localization axiom.
- Rigor boundary extraction: Hypostructure.Framework.Rigor records the three
  proof-boundary layers used by the Burgers route: framework-proved logic
  (`F`), reusable literature facts (`L`), and problem-specific math (`P`).
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
  exposes only windowed heat residual, local uniqueness, positive-time
  smoothing on the window, finite-window energy/dissipation contraction, and
  heat bad-germ exclusion, with a derived contradiction theorem for
  capacity-failing supported heat bad germs. The old global-shaped
  `LocalHeatSmoothingAndUniqueness` interface has been replaced by the reduced
  `PeriodicHeatSemigroupBackend`, which is only a construction source for local
  window certificates and no longer exposes unused semigroup/global fields to
  the proof route.
- Literature heat extraction: the remaining standard periodic heat backend is
  no longer a Burgers-specific axiom. It lives in
  `Hypostructure.Literature.Heat.Periodic1D` as
  `periodicHeat1D_semigroupBackend_literature`, together with the derived local
  window certificate constructor. Burgers imports this literature fact and
  restricts it to the route window.
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
  `Hypostructure.Literature.ColeHopf.PeriodicBurgers1D` as
  `periodicBurgers1D_coleHopfBackend_literature`, with a derived local bridge
  theorem. Burgers-specific code consumes only the resulting local bridge.
- Phase 8: `GroundTruthCertificates.lean` now implements the documented
  finite bad-pattern library: the singleton finite-time `H¹` blow-up template,
  `K_Germ^+`, `K_init^+`, and `K_CatLib^+` certificate objects, and a
  completeness theorem classifying every route-local finite-time `H¹` bad germ
  through that library. `GroundTruthLock.lean` consumes this completeness
  theorem before applying the local Cole-Hopf/heat E5 contradiction.
- Phase 9: `GroundTruthUpgrade.lean` names the final local-to-global analytic
  theorem object and its framework analytic-regularity certificate.
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
analytic work is to instantiate the nontrivial heat semigroup, Cole-Hopf
transform, PDE-local core analytic inputs, and a-posteriori failure-localization
theorem with mathlib-backed constructions.

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
| `K_HeatSmooth^+` | windowed heat residual, local heat uniqueness, local smoothing, heat-side bad-germ exclusion | Burgers global existence, Burgers global uniqueness, Burgers global smoothness |
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
local uniqueness, local continuation, maximal nonextendable window, finite `H¹`
bad-germ extraction, finite-library classification, and local Cole-Hopf/heat
obstruction transfer. The old monolithic
`burgers_axiom_failureLocalizesToWitness` boundary has been removed.

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

Construct a heat semigroup on the periodic carrier or on a compatible heat
carrier. Possible implementation paths:

- Fourier series on `UnitAddCircle`;
- heat kernel on the torus;
- abstract semigroup only if all properties are proven from mathlib
  constructions, not assumed as package fields.

### Required certificates

- local heat zero-time law on a certified window;
- local heat additivity/restriction law for composable certified windows;
- local mean preservation;
- constants fixed;
- finite-window energy contraction;
- finite-window dissipation contraction;
- local heat uniqueness on a certified window;
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

A full periodic heat semigroup may still be proved as a backend theorem, but
`K_HeatSmooth^+` should expose only the local/windowed facts consumed by the
sieve and Lock. If a global semigroup theorem is used internally to prove those
local facts, it must be treated as a heat backend theorem, not as Burgers global
regularity and not as a premise equivalent to the Burgers target.

### Required theorem

```lean
theorem localHeatWindowCertificate
    (nu : BurgersParameters) :
    LocalHeatWindowCertificate nu
```

Current Lean status: the interface and restriction theorem
`PeriodicHeatSemigroupBackend.windowCertificate` are implemented. The remaining
math is to instantiate `PeriodicHeatSemigroupBackend` from mathlib-backed
periodic heat analysis for arbitrary `PeriodicH1State` data.

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
framework. The arbitrary-data heat semigroup remains unimplemented, but when it
is proved it will now enter the framework only through
`PeriodicHeatSemigroupBackend.windowCertificate` and
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
4. In progress: instantiate the periodic heat backend on `PeriodicH1State` and prove its
   local window certificate from mathlib-backed Fourier/semigroup analysis.
   The local certificate interface is implemented, and the constant-sector
   window certificate is proved; the arbitrary-data heat semigroup is still the
   missing analytic construction.
5. In progress: instantiate the concrete periodic Cole-Hopf transform and inverse on the
   ground-truth carrier, including the mean-sector/moving-frame correction and
   positivity chart needed for the local bridge.
   The backend-source interface and constant-sector bridge are proved; the
   nonconstant periodic Cole-Hopf chart and residual transfer remain missing.
6. Partially completed with a narrower axiom boundary:
   `GroundTruthCoreFactory.lean` provides an axiom-free canonical zero core
   bundle, a reusable `toBundle` constructor, and the pointwise windowed
   `BurgersCorePDELocalWindowInputsFor` boundary. The remaining explicit axiom
   `burgers_axiom_corePDELocalWindowInputsFor` covers only the certified local
   Burgers window, local residual, local energy, Poincare, mean-sector, and
   dissipative-window facts for the exact admissible initial datum `u0` being
   proved. The old global-shaped adapter has been removed: the current core
   bundle consumes a certified Burgers window and `SolvesViscousBurgersWeakOnWindow`.
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
  custom axioms: exactly the named boundary below, plus the two literature facts
  Lean foundations: propext, Classical.choice, Quot.sound
```

The current named axiom boundary is:

```text
burgers_axiom_corePDELocalWindowInputsFor
periodicHeat1D_semigroupBackend_literature
periodicBurgers1D_coleHopfBackend_literature
burgers_axiom_localUniquenessOnOverlaps
burgers_axiom_localContinuationCriterion
burgers_axiom_missingRegularityProducesMaximalNonextendableWindow
burgers_axiom_nonextendableWindowProducesFiniteH1Obstruction
burgers_axiom_classifiedObstructionInColeHopfChart
burgers_axiom_classifiedObstructionHeatImageSupported
burgers_axiom_classifiedObstructionHeatImageCapacityFails
```

The current rigor split is:

```text
F: APosterioriLocalization.point
F: burgers_globalRegularity_from_pointwise_local_estimates
L: periodicHeat1D_semigroupBackend_literature
L: periodicBurgers1D_coleHopfBackend_literature
P: burgers_axiom_corePDELocalWindowInputsFor
P: burgers_axiom_localUniquenessOnOverlaps
P: burgers_axiom_localContinuationCriterion
P: burgers_axiom_missingRegularityProducesMaximalNonextendableWindow
P: burgers_axiom_nonextendableWindowProducesFiniteH1Obstruction
P: burgers_axiom_classifiedObstructionInColeHopfChart
P: burgers_axiom_classifiedObstructionHeatImageSupported
P: burgers_axiom_classifiedObstructionHeatImageCapacityFails
```

The next phase is to discharge the reusable literature facts with mathlib-backed
analysis modules, or replace the heat semigroup literature boundary by an
equivalent local-window heat theorem. Separately, discharge the Burgers-specific
windowed local PDE certificate factory and discharge the decomposed local
continuation/localization chain showing that any missing regularity witness
produces a route-local bad morphism classified by the finite-time `H¹` library.
