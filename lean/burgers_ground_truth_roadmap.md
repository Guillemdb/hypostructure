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
```

## Implementation Status: 2026-04-16

The migrated ground-truth path now has a compiled interface, certificate-run,
and conditional final theorem under:

```text
Hypostructure/Backends/Burgers1D/GroundTruthState.lean
Hypostructure/Backends/Burgers1D/GroundTruthPDE.lean
Hypostructure/Backends/Burgers1D/GroundTruthCertificates.lean
Hypostructure/Backends/Burgers1D/GroundTruthRun.lean
Hypostructure/Backends/Burgers1D/GroundTruthFinal.lean
```

These files are imported by `Hypostructure.lean`, so the new path is part of
the public Lean surface.

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
- Phase 2: `BurgersGroundTruthEvolutionPackage` replaces the permissive old
  package on the ground-truth path. Its fields must prove the concrete weak PDE
  residual, global `H¹`, uniqueness, and positive-time smoothing predicates.
- Phase 3: finite-window energy inequalities and local energy framework
  certificates are implemented. The existing smooth snapshot energy identity is
  exposed as a genuine local analytic theorem.
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
- Phase 10 migration boundary: `GroundTruthFinal.lean` exposes the migrated
  dataset claim over `BurgersGroundTruthBackendPermits`. This deliberately
  prevents the final ground-truth path from using the old scaffold's packaged
  global regularity shortcut.

Compilation checks run after the replacements:

```text
lake build Hypostructure.Backends.Burgers1D.GroundTruthState
lake build Hypostructure.Backends.Burgers1D.GroundTruthPDE
lake build Hypostructure.Backends.Burgers1D.GroundTruthCertificates
lake build Hypostructure.Backends.Burgers1D.GroundTruthRun
lake build Hypostructure.Backends.Burgers1D.GroundTruthFinal
```

All five focused checks passed.

Important caveat:

The first five phases are implemented as the reusable ground-truth interface and
local certificate layer. They do not yet prove classical global Burgers
regularity. In particular, the remaining hard analytic work is still in Phases
6-10: periodic heat backend, Cole-Hopf bridge, Lock theorem from local
certificates, analytic upgrade, and final dataset theorem.

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
  is conditional on `BurgersGroundTruthBackendPermits`. Those permits name the
  remaining backend theorems explicitly instead of hiding them behind the old
  scaffold.

The important distinction is that the Lean proof now has two routes:

- Legacy route: `Run.lean`, `Final.lean`, `PDEEvolution.lean`,
  `ModelInstance.lean`, and related scaffold modules. This route checks a
  proof-object shape, but it can still rely on package-level regularity fields
  that are too weak for a ground-truth Burgers theorem.
- Migrated route: `GroundTruthRun.lean` and `GroundTruthFinal.lean`. This route
  is the one that should be developed until it matches the template document.
  It does not use the old model instance as semantic evidence, and it exposes
  the exact missing math as backend permits.

To make the Lean proof fully match the template document, every field of
`BurgersGroundTruthBackendPermits` must be replaced by a theorem proved from
the local certificate bundle and mathlib-backed analysis. After that,
`burgers_groundTruth_dataset_theorem` should take only the PDE parameters,
initial data, local hypotheses required by the template, and the constructed
local certificates. It should no longer take a generic permit package.

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

## Current Hardcoded or Too-Weak Pieces

### 1. `BurgersPDEEvolutionPackage` is too permissive

Current problem:

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

Required replacement:

```lean
class BurgersPDEEvolutionPackage (nu : BurgersParameters) where
  State : Type
  IsPeriodicH1 : State -> Prop
  flow : NNReal -> State -> State
  solutionCurve : State -> BurgersSolutionCurve nu
  solves :
    forall u0, IsPeriodicH1 u0 ->
      SolvesViscousBurgersWeak nu u0 (solutionCurve u0)
  globalH1 :
    forall u0 t, IsPeriodicH1 u0 ->
      IsPeriodicH1 (flow t u0)
  unique :
    forall u0, IsPeriodicH1 u0 ->
      UniqueWeakBurgersSolution nu u0 (solutionCurve u0)
  smoothPositiveTime :
    forall u0 t, IsPeriodicH1 u0 -> 0 < (t : Real) ->
      SmoothAtPositiveTime (solutionCurve u0) t
```

The exact names can change. The key requirement is that these fields force the
proof to mention the actual PDE and the actual state carrier.

### 2. `BurgersEvolutionRegularity` can currently be trivial

Current problem:

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

Required replacement:

- remove or quarantine this class from the ground-truth path;
- define concrete predicates for global `H^1`, uniqueness, and smoothing;
- ensure those predicates are proved from local certificates and analytic
  constructions.

### 3. The analytic upgrade currently has a shortcut

Current problem:

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

The analytic upgrade must be a theorem whose premises are concrete local
certificates:

```lean
theorem analyticUpgradeFromLocalCertificates
    (nu : BurgersParameters)
    (hEnergy : LocalEnergyIdentity nu)
    (hPoincare : LocalPoincareCoercivity nu)
    (hMean : LocalMeanSectorPreservation nu)
    (hGerm : LocalBadGermCapacity nu)
    (hColeHopf : LocalColeHopfBridge nu)
    (hHeat : LocalHeatSmoothingAndUniqueness nu)
    (hLock : LockBlocksBurgersBadGerms nu) :
    BurgersGroundTruthGlobalRegularityStatement nu
```

No premise may be equivalent to the desired global regularity theorem.

### 4. `RealScaffold.lean` is still structural, not the real PDE

Current problem:

- `realBurgersFlow` is defined by Cole-Hopf/heat conjugacy;
- `RealBurgersDynamicsStatement` states that same conjugacy;
- it does not state the PDE residual
  `u_t + u * u_x = nu * u_xx`.

Required replacement:

- keep `RealScaffold.lean` as engineering scaffolding only;
- introduce a new ground-truth backend path that defines the actual weak PDE;
- ensure the final theorem imports the ground-truth backend, not the scaffold
  as semantic evidence.

## Required New Lean Modules

The exact file names can change, but the ground-truth path should be separated
from the model/scaffold path.

Implemented files:

```text
Hypostructure/Backends/Burgers1D/GroundTruthState.lean
Hypostructure/Backends/Burgers1D/GroundTruthPDE.lean
Hypostructure/Backends/Burgers1D/GroundTruthCertificates.lean
Hypostructure/Backends/Burgers1D/GroundTruthRun.lean
Hypostructure/Backends/Burgers1D/GroundTruthFinal.lean
```

Remaining recommended files:

```text
Hypostructure/Backends/Burgers1D/GroundTruthHeat.lean
Hypostructure/Backends/Burgers1D/GroundTruthColeHopf.lean
Hypostructure/Backends/Burgers1D/GroundTruthUpgrade.lean
```

The final ground-truth claim lives in `GroundTruthFinal.lean` now, but it is
still conditional on explicit backend permits. The finished version should
move those permits into `GroundTruthHeat.lean`, `GroundTruthColeHopf.lean`,
and `GroundTruthUpgrade.lean` as proved theorems.

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

- `heat_zero`
- `heat_add`
- mean preservation
- constants fixed
- finite-window energy contraction
- finite-window dissipation contraction
- heat uniqueness
- heat smoothing for positive time

### Required theorem

```lean
theorem localHeatSmoothingAndUniqueness
    (nu : BurgersParameters) :
    LocalHeatSmoothingAndUniqueness nu
```

### Acceptance criteria

- `K_HeatSmooth^+` is backed by this theorem.
- The theorem does not import Burgers global regularity.
- Heat smoothing is allowed as a backend local/semigroup theorem because it is
  not the Burgers target; it becomes part of the local bridge package.

## Phase 7: Cole-Hopf Bridge Certificate

### Goal

Prove the bridge used by E5 in the Lock.

### Required definitions

- Cole-Hopf transform on the mean-zero or mean-adjusted sector;
- inverse Cole-Hopf;
- positivity/nonvanishing condition for the heat variable;
- moving-frame or mean-sector correction for nonzero mean;
- compatibility with periodicity.

### Required theorem

```lean
theorem localColeHopfBridge
    (nu : BurgersParameters) :
    LocalColeHopfBridge nu
```

The theorem should include:

- transform maps certified Burgers local windows to heat local windows;
- inverse maps positive heat windows back to Burgers windows;
- transform and inverse agree on the relevant sector;
- Burgers weak residual transfers to heat residual;
- heat residual transfers back to Burgers residual;
- uniqueness transfers through the transform.

### Acceptance criteria

- `K_ColeHopf^+` is not an identity-transform scaffold.
- The Lock uses this bridge to map bad Burgers germs into heat-side bad germs.
- The theorem is local/windowed where possible.

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
    (hGerm : LocalBadGermCapacity nu)
    (hInit : UniversalBadObjectInitialized nu)
    (hLib : BadPatternLibraryComplete nu)
    (hCole : LocalColeHopfBridge nu)
    (hHeat : LocalHeatSmoothingAndUniqueness nu) :
    LockBlocksBurgersBadGerms nu
```

### Acceptance criteria

- `K_Cat_Hom^blk` is proved from local certificates.
- The proof does not mention `BurgersGroundTruthGlobalRegularityStatement` or
  any old scaffold global regularity theorem.
- The proof does not use a global empty singular set theorem.

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
    (hCole : LocalColeHopfBridge nu)
    (hHeat : LocalHeatSmoothingAndUniqueness nu)
    (hLock : LockBlocksBurgersBadGerms nu) :
    BurgersGroundTruthGlobalRegularityStatement nu
```

This is where global regularity is assembled. It is allowed to conclude the
global theorem here because all inputs are local certificates plus Lock
exclusion.

### Acceptance criteria

- `BurgersAnalyticUpgradePackage` is either removed from the final path or its
  only instance is this theorem.
- The proof of `BurgersGroundTruthGlobalRegularityStatement` does not call a
  package field that already states global regularity.
- The dependency cone contains local certificates, heat/Cole-Hopf bridge, Lock,
  and the upgrade theorem.

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

The broader tree may still contain scaffolds, but the ground-truth theorem must
not depend on them.

## Implementation Order

Recommended sequence from the current migrated state:

1. Keep `GroundTruthState.lean`, `GroundTruthPDE.lean`,
   `GroundTruthCertificates.lean`, `GroundTruthRun.lean`, and
   `GroundTruthFinal.lean` as the only ground-truth public route.
2. Build `GroundTruthHeat.lean` and prove the periodic heat semigroup
   certificates on the same carrier.
3. Build `GroundTruthColeHopf.lean` and prove the local Cole-Hopf bridge and
   inverse bridge.
4. Prove compactness and representation certificates against
   `PeriodicH1State`, replacing the current route-level permits for
   `K_C_mu^+` and `K_RepDesc_K^+`.
5. Prove `lockBlocksBadGerms` from local capacity, local Cole-Hopf, and local
   heat smoothing, without mentioning global Burgers regularity.
6. Prove `analyticUpgradeFromLocalCertificates` from the local certificate
   bundle and the Lock theorem.
7. Replace `BurgersGroundTruthBackendPermits` in
   `burgers_groundTruth_dataset_theorem` with the concrete proved backend
   theorems.
8. Run the ground-truth import/search checks and then make the migrated route
   the documented theorem path.

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

The next code milestone is no longer to create the ground-truth path. That
path exists and compiles. The next milestone is to replace the first explicit
backend permit with a proved theorem.

Recommended next target:

```text
Hypostructure/Backends/Burgers1D/GroundTruthHeat.lean
```

This should introduce the periodic heat carrier/semigroup interface and prove
the local heat certificates needed by `localHeatSmoothingAndUniqueness`. Once
that compiles, the corresponding field should be removed from
`BurgersGroundTruthBackendPermits` or replaced by a concrete theorem argument
with the exact heat statement.
