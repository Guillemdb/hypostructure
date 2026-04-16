import Hypostructure.Backends.Burgers1D.GroundTruthRun

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Framework

noncomputable section

/-- A reusable constructor input for the framework-side Burgers core bundle.
This is not an analytic theorem: it packages already-proved local certificate
records into the route object consumed by the Lock and upgrade layers. -/
structure BurgersGroundTruthCoreCertificateFactory
    (nu : BurgersParameters) where
  u0 : PeriodicH1State
  window : BurgersWindow
  window_contains_zero : window.Contains 0
  solution : BurgersSolutionCurve nu
  solutionCertifiedOnWindow : SolvesViscousBurgersWeakOnWindow nu u0 solution window
  energy : LocalEnergyCertificateData nu solution
  compactness : LocalCompactnessCertificateData
  poincare : LocalPoincareCertificateData
  meanSector : LocalMeanSectorCertificateData nu u0 solution
  dissipativeWindow : LocalDissipativeWindowCertificateData nu solution
  representation : LocalRepresentationCertificateData
  badGermCapacity : LocalBadGermCapacityCertificateData
  badPatternLibrary : BurgersBadPatternLibraryCertificateData

def BurgersGroundTruthCoreCertificateFactory.toBundle
    {nu : BurgersParameters}
    (F : BurgersGroundTruthCoreCertificateFactory nu) :
    BurgersGroundTruthCoreCertificateBundle nu where
  u0 := F.u0
  window := F.window
  window_contains_zero := F.window_contains_zero
  solution := F.solution
  solutionCertifiedOnWindow := F.solutionCertifiedOnWindow
  energy := F.energy
  compactness := F.compactness
  poincare := F.poincare
  meanSector := F.meanSector
  dissipativeWindow := F.dissipativeWindow
  representation := F.representation
  badGermCapacity := F.badGermCapacity
  badPatternLibrary := F.badPatternLibrary

theorem BurgersGroundTruthCoreCertificateFactory.toBundle_sound
    {nu : BurgersParameters}
    (F : BurgersGroundTruthCoreCertificateFactory nu) :
    F.toBundle.allCertificatesSound :=
  BurgersGroundTruthCoreCertificateBundle.localCertificatesSound F.toBundle

/-- The fully assembled arbitrary-data local analytic input package consumed by
the core bundle factory. The current final boundary uses the narrower
`BurgersCorePDELocalAnalyticInputs` structure and constructs this package by
adding the 0-truncated framework certificates below. -/
abbrev BurgersCoreLocalAnalyticInputs :=
  BurgersGroundTruthCoreCertificateFactory

def BurgersCoreLocalAnalyticInputs.toCoreBundle
    {nu : BurgersParameters}
    (I : BurgersCoreLocalAnalyticInputs nu) :
    BurgersGroundTruthCoreCertificateBundle nu :=
  I.toBundle

theorem BurgersCoreLocalAnalyticInputs.toCoreBundle_sound
    {nu : BurgersParameters}
    (I : BurgersCoreLocalAnalyticInputs nu) :
    I.toCoreBundle.allCertificatesSound :=
  I.toBundle_sound

theorem BurgersGroundTruthCoreCertificateBundle.u0_isPeriodicH1
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) :
    PeriodicH1State.IsPeriodicH1 B.u0 := by
  have hboundary : PeriodicH1State.IsPeriodicH1 (B.solution.eval 0) :=
    B.solutionCertifiedOnWindow.2.1 0 B.window_contains_zero
  rw [B.solutionCertifiedOnWindow.1 B.window_contains_zero] at hboundary
  exact hboundary

def identityCoreRepresentationDictionary : LocalRepresentationDictionary where
  Code := PeriodicH1State
  encode := id
  decode := id
  faithful := by intro u; rfl

def coreBadGermForState (u : PeriodicH1State) : BurgersBadGerm where
  centerTime := 0
  centerSpace := 0
  localWindow :=
    { centerTime := 0
      radius := 1
      radius_pos := by norm_num
      centerSpace := 0 }
  profile := u

/-- The genuinely analytic part of the core Burgers route. Everything omitted
from this structure is constructed below by finite 0-truncated framework
machinery: singleton compactness, identity representation, local bad-germ
capacity, and the finite bad-pattern library. -/
structure BurgersCorePDELocalAnalyticInputs
    (nu : BurgersParameters) where
  u0 : PeriodicH1State
  window : BurgersWindow
  window_contains_zero : window.Contains 0
  solution : BurgersSolutionCurve nu
  solutionCertifiedOnWindow : SolvesViscousBurgersWeakOnWindow nu u0 solution window
  energy : LocalEnergyCertificateData nu solution
  poincare : LocalPoincareCoercivity u0
  meanSector : LocalMeanSectorCertificateData nu u0 solution
  dissipativeWindow : LocalDissipativeWindowCertificateData nu solution

/-- Pointwise local PDE estimates for the exact initial datum being proved.
This is the shape needed for the classical arbitrary-data theorem: each `u0`
gets its own local certificate package before the hypostructure Lock and
a-posteriori upgrade are applied. -/
structure BurgersCorePDELocalAnalyticInputsFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State) where
  window : BurgersWindow
  window_contains_zero : window.Contains 0
  solution : BurgersSolutionCurve nu
  solutionCertifiedOnWindow : SolvesViscousBurgersWeakOnWindow nu u0 solution window
  energy : LocalEnergyCertificateData nu solution
  poincare : LocalPoincareCoercivity u0
  meanSector : LocalMeanSectorCertificateData nu u0 solution
  dissipativeWindow : LocalDissipativeWindowCertificateData nu solution

/-- Genuinely windowed Burgers local data for the exact initial datum being
proved. This is the public problem-specific local-input boundary: it contains a
certified local Burgers window, windowed weak residual data, and windowed local
estimates only. It deliberately does not contain `SolvesViscousBurgersWeak`,
`PeriodicBoundary`, or `WeakBurgersResidual` on arbitrary times/windows. -/
structure BurgersCorePDELocalWindowInputsFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : BurgersWindow) where
  certified : CertifiedBurgersLocalWindow nu
  initial_eq : certified.initial = u0
  window_eq : certified.window = W
  window_contains_zero : W.Contains 0
  energy : LocalEnergyCertificateData nu certified.curve
  energy_window_eq : energy.window = W.time
  poincare : LocalPoincareCoercivity u0
  meanSector : LocalMeanSectorCertificateData nu u0 certified.curve
  mean_window_eq : meanSector.window = W.time
  dissipativeWindow : LocalDissipativeWindowCertificateData nu certified.curve
  dissipative_window_eq : dissipativeWindow.window = W.time

def BurgersCorePDELocalWindowInputsFor.curve
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {W : BurgersWindow}
    (I : BurgersCorePDELocalWindowInputsFor nu u0 W) :
    BurgersSolutionCurve nu :=
  I.certified.curve

theorem BurgersCorePDELocalWindowInputsFor.solvesOnWindow
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {W : BurgersWindow}
    (I : BurgersCorePDELocalWindowInputsFor nu u0 W) :
    SolvesViscousBurgersWeakOnWindow nu u0 I.curve W := by
  cases I with
  | mk certified initial_eq window_eq window_contains_zero energy energy_window_eq poincare
      meanSector mean_window_eq dissipativeWindow dissipative_window_eq =>
      subst u0
      subst W
      exact certified.solvesOnWindow

theorem BurgersCorePDELocalWindowInputsFor.residual_on_window
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {W : BurgersWindow}
    (I : BurgersCorePDELocalWindowInputsFor nu u0 W)
    (φ : SmoothCompactTimePeriodicSpaceTest)
    (hφ : φ.window = W.time) :
    timeSpaceIntegralOn W.time
      (burgersWeakResidualIntegrand nu I.curve φ) = 0 := by
  have hsol := I.solvesOnWindow
  exact hsol.2.2 φ hφ

def BurgersCorePDELocalWindowInputsFor.toAnalyticInputsFor
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {W : BurgersWindow}
    (I : BurgersCorePDELocalWindowInputsFor nu u0 W) :
    BurgersCorePDELocalAnalyticInputsFor nu u0 where
  window := W
  solution := I.curve
  window_contains_zero := I.window_contains_zero
  solutionCertifiedOnWindow := I.solvesOnWindow
  energy := I.energy
  poincare := I.poincare
  meanSector := I.meanSector
  dissipativeWindow := I.dissipativeWindow

def BurgersCorePDELocalAnalyticInputsFor.toInputs
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    (I : BurgersCorePDELocalAnalyticInputsFor nu u0) :
    BurgersCorePDELocalAnalyticInputs nu where
  u0 := u0
  window := I.window
  window_contains_zero := I.window_contains_zero
  solution := I.solution
  solutionCertifiedOnWindow := I.solutionCertifiedOnWindow
  energy := I.energy
  poincare := I.poincare
  meanSector := I.meanSector
  dissipativeWindow := I.dissipativeWindow

theorem BurgersCorePDELocalAnalyticInputsFor.toInputs_u0
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    (I : BurgersCorePDELocalAnalyticInputsFor nu u0) :
    I.toInputs.u0 = u0 :=
  rfl

theorem BurgersCorePDELocalAnalyticInputs.u0_isPeriodicH1
    {nu : BurgersParameters}
    (I : BurgersCorePDELocalAnalyticInputs nu) :
    PeriodicH1State.IsPeriodicH1 I.u0 := by
  have hboundary : PeriodicH1State.IsPeriodicH1 (I.solution.eval 0) :=
    I.solutionCertifiedOnWindow.2.1 0 I.window_contains_zero
  rw [I.solutionCertifiedOnWindow.1 I.window_contains_zero] at hboundary
  exact hboundary

def BurgersCorePDELocalAnalyticInputs.compactnessData
    {nu : BurgersParameters}
    (I : BurgersCorePDELocalAnalyticInputs nu) :
    LocalCompactnessCertificateData where
  state := I.u0
  profileSet := {I.u0}
  contains_state := by simp
  finite_profileSet := Set.finite_singleton I.u0

def BurgersCorePDELocalAnalyticInputs.poincareData
    {nu : BurgersParameters}
    (I : BurgersCorePDELocalAnalyticInputs nu) :
    LocalPoincareCertificateData where
  state := I.u0
  certified := I.poincare

def BurgersCorePDELocalAnalyticInputs.representationData
    {nu : BurgersParameters}
    (_I : BurgersCorePDELocalAnalyticInputs nu) :
    LocalRepresentationCertificateData where
  dictionary := identityCoreRepresentationDictionary

def BurgersCorePDELocalAnalyticInputs.badGermCapacityData
    {nu : BurgersParameters}
    (I : BurgersCorePDELocalAnalyticInputs nu) :
    LocalBadGermCapacityCertificateData where
  germ := coreBadGermForState I.u0
  admissible := I.u0_isPeriodicH1

def BurgersCorePDELocalAnalyticInputs.toLocalAnalyticInputs
    {nu : BurgersParameters}
    (I : BurgersCorePDELocalAnalyticInputs nu) :
    BurgersCoreLocalAnalyticInputs nu where
  u0 := I.u0
  window := I.window
  window_contains_zero := I.window_contains_zero
  solution := I.solution
  solutionCertifiedOnWindow := I.solutionCertifiedOnWindow
  energy := I.energy
  compactness := I.compactnessData
  poincare := I.poincareData
  meanSector := I.meanSector
  dissipativeWindow := I.dissipativeWindow
  representation := I.representationData
  badGermCapacity := I.badGermCapacityData
  badPatternLibrary := burgersFiniteTimeH1BadPatternLibraryCertificateData

theorem BurgersCorePDELocalAnalyticInputs.toLocalAnalyticInputs_sound
    {nu : BurgersParameters}
    (I : BurgersCorePDELocalAnalyticInputs nu) :
    I.toLocalAnalyticInputs.toCoreBundle.allCertificatesSound :=
  I.toLocalAnalyticInputs.toCoreBundle_sound

def canonicalZeroCoreState : PeriodicH1State :=
  PeriodicH1State.constantState 0

theorem canonicalZeroCoreState_isPeriodicH1 :
    PeriodicH1State.IsPeriodicH1 canonicalZeroCoreState :=
  PeriodicH1State.isPeriodicH1 canonicalZeroCoreState

theorem canonicalZeroCoreState_mean :
    PeriodicH1State.mean canonicalZeroCoreState = 0 := by
  simp [canonicalZeroCoreState, PeriodicH1State.mean_constantState]

theorem canonicalZeroCoreState_energy :
    PeriodicH1State.energy canonicalZeroCoreState = 0 := by
  simp [canonicalZeroCoreState, PeriodicH1State.energy,
    PeriodicH1State.constantState, PeriodicH1State.constantProfile]

theorem canonicalZeroCoreState_derivativeEnergy :
    PeriodicH1State.derivativeEnergy canonicalZeroCoreState = 0 := by
  simp [canonicalZeroCoreState, PeriodicH1State.derivativeEnergy,
    PeriodicH1State.constantState, PeriodicH1State.zeroDerivative]

theorem canonicalZeroCoreState_dissipation
    (nu : BurgersParameters) :
    PeriodicH1State.dissipation nu canonicalZeroCoreState = 0 := by
  simp [PeriodicH1State.dissipation_eq_viscosity_mul_derivativeEnergy,
    canonicalZeroCoreState_derivativeEnergy]

def canonicalZeroCoreBurgersCurve
    (nu : BurgersParameters) : BurgersSolutionCurve nu where
  eval := fun _ => canonicalZeroCoreState

theorem canonicalZeroCoreBurgersCurve_initial
    (nu : BurgersParameters) :
    InitialCondition canonicalZeroCoreState
      (canonicalZeroCoreBurgersCurve nu) := by
  rfl

theorem canonicalZeroCoreBurgersCurve_boundary
    (nu : BurgersParameters) :
    PeriodicBoundary (canonicalZeroCoreBurgersCurve nu) := by
  intro _t
  exact canonicalZeroCoreState_isPeriodicH1

theorem canonicalZeroCoreBurgersCurve_weakResidual
    (nu : BurgersParameters) :
    WeakBurgersResidual nu (canonicalZeroCoreBurgersCurve nu) := by
  intro φ
  simp [WeakBurgersResidual, timeSpaceIntegralOn, burgersWeakResidualIntegrand,
    canonicalZeroCoreBurgersCurve, canonicalZeroCoreState,
    PeriodicH1State.constantState, PeriodicH1State.constantProfile,
    PeriodicH1State.zeroDerivative]

theorem canonicalZeroCoreBurgersCurve_solves
    (nu : BurgersParameters) :
    SolvesViscousBurgersWeak nu canonicalZeroCoreState
      (canonicalZeroCoreBurgersCurve nu) := by
  exact ⟨
    canonicalZeroCoreBurgersCurve_initial nu,
    canonicalZeroCoreBurgersCurve_boundary nu,
    canonicalZeroCoreBurgersCurve_weakResidual nu
  ⟩

def canonicalZeroCoreTimeWindow : TimeWindow where
  t0 := 0
  t1 := 0
  ordered := le_rfl

theorem canonicalZeroCoreLocalEnergy
    (nu : BurgersParameters) :
    LocalEnergyIdentity nu
      (canonicalZeroCoreBurgersCurve nu)
      canonicalZeroCoreTimeWindow := by
  simp [LocalEnergyIdentity, FiniteWindowEnergyInequality, timeIntegralOn,
    canonicalZeroCoreTimeWindow, canonicalZeroCoreBurgersCurve,
    canonicalZeroCoreState_energy, canonicalZeroCoreState_dissipation]

theorem canonicalZeroCoreLocalPoincare :
    LocalPoincareCoercivity canonicalZeroCoreState := by
  intro _hmean
  simp [LocalPoincareCoercivity, canonicalZeroCoreState_energy,
    canonicalZeroCoreState_derivativeEnergy]

theorem canonicalZeroCoreLocalMeanSector
    (nu : BurgersParameters) :
    LocalMeanSectorPreservation nu canonicalZeroCoreState
      (canonicalZeroCoreBurgersCurve nu)
      canonicalZeroCoreTimeWindow := by
  intro _BW _hW _hsol _t _ht0 _ht1
  simp [canonicalZeroCoreBurgersCurve, canonicalZeroCoreState_mean]

theorem canonicalZeroCoreLocalDissipativeWindow
    (nu : BurgersParameters) :
    LocalDissipativeWindow nu
      (canonicalZeroCoreBurgersCurve nu)
      canonicalZeroCoreTimeWindow := by
  intro _t _ht0 _ht1
  simp [canonicalZeroCoreBurgersCurve, canonicalZeroCoreState_energy]

def canonicalZeroCoreBadGerm : BurgersBadGerm where
  centerTime := 0
  centerSpace := 0
  localWindow :=
    { centerTime := 0
      radius := 1
      radius_pos := by norm_num
      centerSpace := 0 }
  profile := canonicalZeroCoreState

/-- A fully proved framework-side core bundle for the canonical zero route. It
is intentionally only a sanity instance; the final arbitrary-data theorem keeps
the narrower `BurgersCorePDELocalAnalyticInputs` package explicit until the
nonconstant PDE-local certificate factory is proved. -/
def burgersCanonicalCoreCertificateFactory
    (nu : BurgersParameters) :
    BurgersGroundTruthCoreCertificateFactory nu where
  u0 := canonicalZeroCoreState
  window :=
    { time := canonicalZeroCoreTimeWindow }
  window_contains_zero := by
    simp [BurgersWindow.Contains, TimeWindow.Contains, canonicalZeroCoreTimeWindow]
  solution := canonicalZeroCoreBurgersCurve nu
  solutionCertifiedOnWindow :=
    (canonicalZeroCoreBurgersCurve_solves nu).on_burgersWindow
      { time := canonicalZeroCoreTimeWindow }
  energy :=
    { window := canonicalZeroCoreTimeWindow
      certified := canonicalZeroCoreLocalEnergy nu }
  compactness :=
    { state := canonicalZeroCoreState
      profileSet := {canonicalZeroCoreState}
      contains_state := by simp
      finite_profileSet := Set.finite_singleton canonicalZeroCoreState }
  poincare :=
    { state := canonicalZeroCoreState
      certified := canonicalZeroCoreLocalPoincare }
  meanSector :=
    { window := canonicalZeroCoreTimeWindow
      certified := canonicalZeroCoreLocalMeanSector nu }
  dissipativeWindow :=
    { window := canonicalZeroCoreTimeWindow
      certified := canonicalZeroCoreLocalDissipativeWindow nu }
  representation :=
    { dictionary := identityCoreRepresentationDictionary }
  badGermCapacity :=
    { germ := canonicalZeroCoreBadGerm
      admissible := canonicalZeroCoreState_isPeriodicH1 }
  badPatternLibrary := burgersFiniteTimeH1BadPatternLibraryCertificateData

def burgersCanonicalCoreCertificateBundle
    (nu : BurgersParameters) :
    BurgersGroundTruthCoreCertificateBundle nu :=
  (burgersCanonicalCoreCertificateFactory nu).toBundle

theorem burgersCanonicalCoreCertificateBundle_sound
    (nu : BurgersParameters) :
    (burgersCanonicalCoreCertificateBundle nu).allCertificatesSound :=
  (burgersCanonicalCoreCertificateFactory nu).toBundle_sound

end

end Hypostructure.Backends.Burgers1D
