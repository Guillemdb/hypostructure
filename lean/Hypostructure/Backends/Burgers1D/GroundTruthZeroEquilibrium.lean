import Hypostructure.Backends.Burgers1D.GroundTruthFinal

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Framework

noncomputable section

def zeroState : PeriodicH1State :=
  PeriodicH1State.constantState 0

theorem zeroState_isPeriodicH1 :
    PeriodicH1State.IsPeriodicH1 zeroState :=
  PeriodicH1State.isPeriodicH1 zeroState

theorem zeroState_mean :
    PeriodicH1State.mean zeroState = 0 := by
  simp [zeroState, PeriodicH1State.mean_constantState]

theorem zeroState_energy :
    PeriodicH1State.energy zeroState = 0 := by
  simp [zeroState, PeriodicH1State.energy, PeriodicH1State.constantState,
    PeriodicH1State.constantProfile]

theorem zeroState_derivativeEnergy :
    PeriodicH1State.derivativeEnergy zeroState = 0 := by
  simp [zeroState, PeriodicH1State.derivativeEnergy, PeriodicH1State.constantState,
    PeriodicH1State.zeroDerivative]

theorem zeroState_dissipation
    (nu : BurgersParameters) :
    PeriodicH1State.dissipation nu zeroState = 0 := by
  simp [PeriodicH1State.dissipation_eq_viscosity_mul_derivativeEnergy,
    zeroState_derivativeEnergy]

theorem zeroState_hasWeakDerivativeOrder
    (k : ℕ) :
    HasWeakDerivativeOrder k zeroState.value := by
  induction k with
  | zero =>
      exact zeroState.value_memL2
  | succ k ih =>
      refine ⟨zeroState.weakDeriv, zeroState.weakDeriv_spec, ?_⟩
      simpa [zeroState, PeriodicH1State.constantState, PeriodicH1State.zeroDerivative]
        using ih

theorem zeroState_smooth :
    SmoothPeriodicState zeroState := by
  intro k
  exact zeroState_hasWeakDerivativeOrder k

def zeroBurgersCurve (nu : BurgersParameters) : BurgersSolutionCurve nu where
  eval := fun _ => zeroState

theorem zeroBurgersCurve_initial
    (nu : BurgersParameters) :
    InitialCondition zeroState (zeroBurgersCurve nu) := by
  rfl

theorem zeroBurgersCurve_boundary
    (nu : BurgersParameters) :
    PeriodicBoundary (zeroBurgersCurve nu) := by
  intro _t
  exact zeroState_isPeriodicH1

theorem zeroBurgersCurve_weakResidual
    (nu : BurgersParameters) :
    WeakBurgersResidual nu (zeroBurgersCurve nu) := by
  intro φ
  simp [WeakBurgersResidual, timeSpaceIntegralOn, burgersWeakResidualIntegrand,
    zeroBurgersCurve, zeroState, PeriodicH1State.constantState,
    PeriodicH1State.constantProfile, PeriodicH1State.zeroDerivative]

theorem zeroBurgersCurve_solves
    (nu : BurgersParameters) :
    SolvesViscousBurgersWeak nu zeroState (zeroBurgersCurve nu) := by
  exact ⟨
    zeroBurgersCurve_initial nu,
    zeroBurgersCurve_boundary nu,
    zeroBurgersCurve_weakResidual nu
  ⟩

theorem zeroBurgersCurve_globalH1
    (nu : BurgersParameters) :
    GlobalH1Solution (zeroBurgersCurve nu) := by
  intro _t
  exact zeroState_isPeriodicH1

theorem zeroBurgersCurve_smoothPositive
    (nu : BurgersParameters)
    (t : ℝ)
    (_ht : 0 < t) :
    SmoothAtPositiveTime (zeroBurgersCurve nu) t :=
  zeroState_smooth

def zeroHeatCurve (nu : BurgersParameters) : PeriodicHeatCurve nu where
  eval := fun _ => zeroState

theorem zeroHeatCurve_initial
    (nu : BurgersParameters) :
    HeatInitialCondition zeroState (zeroHeatCurve nu) := by
  rfl

theorem zeroHeatCurve_boundary
    (nu : BurgersParameters) :
    HeatPeriodicBoundary (zeroHeatCurve nu) := by
  intro _t
  exact zeroState_isPeriodicH1

theorem zeroHeatCurve_weakResidual
    (nu : BurgersParameters) :
    WeakHeatResidual nu (zeroHeatCurve nu) := by
  intro φ
  simp [WeakHeatResidual, timeSpaceIntegralOn, heatWeakResidualIntegrand,
    zeroHeatCurve, zeroState, PeriodicH1State.constantState,
    PeriodicH1State.constantProfile, PeriodicH1State.zeroDerivative]

theorem zeroHeatCurve_solves
    (nu : BurgersParameters) :
    SolvesPeriodicHeatWeak nu zeroState (zeroHeatCurve nu) := by
  exact ⟨
    zeroHeatCurve_initial nu,
    zeroHeatCurve_boundary nu,
    zeroHeatCurve_weakResidual nu
  ⟩

theorem zeroHeatCurve_globalH1
    (nu : BurgersParameters) :
    HeatGlobalH1 (zeroHeatCurve nu) := by
  intro _t
  exact zeroState_isPeriodicH1

theorem zeroHeatCurve_smoothPositive
    (nu : BurgersParameters)
    (t : ℝ)
    (_ht : 0 < t) :
    HeatSmoothAtPositiveTime (zeroHeatCurve nu) t :=
  zeroState_smooth

def zeroTimeWindow : TimeWindow where
  t0 := 0
  t1 := 0
  ordered := le_rfl

theorem zeroLocalEnergy
    (nu : BurgersParameters) :
    LocalEnergyIdentity nu (zeroBurgersCurve nu) zeroTimeWindow := by
  simp [LocalEnergyIdentity, FiniteWindowEnergyInequality, timeIntegralOn,
    zeroTimeWindow, zeroBurgersCurve, zeroState_energy, zeroState_dissipation]

theorem zeroLocalPoincare :
    LocalPoincareCoercivity zeroState := by
  intro _hmean
  simp [LocalPoincareCoercivity, zeroState_energy, zeroState_derivativeEnergy]

theorem zeroLocalMeanSector
    (nu : BurgersParameters) :
    LocalMeanSectorPreservation nu zeroState (zeroBurgersCurve nu) zeroTimeWindow := by
  intro _BW _hW _hsol t _ht0 _ht1
  simp [zeroBurgersCurve, zeroState_mean]

theorem zeroLocalDissipativeWindow
    (nu : BurgersParameters) :
    LocalDissipativeWindow nu (zeroBurgersCurve nu) zeroTimeWindow := by
  intro _t _ht0 _ht1
  simp [zeroBurgersCurve, zeroState_energy]

def zeroBadGerm : BurgersBadGerm where
  centerTime := 0
  centerSpace := 0
  localWindow :=
    { centerTime := 0
      radius := 1
      radius_pos := by norm_num
      centerSpace := 0 }
  profile := zeroState

def identityRepresentationDictionary : LocalRepresentationDictionary where
  Code := PeriodicH1State
  encode := id
  decode := id
  faithful := by intro u; rfl

def zeroCoreCertificateBundle
    (nu : BurgersParameters) :
    BurgersGroundTruthCoreCertificateBundle nu where
  u0 := zeroState
  window := { time := zeroTimeWindow }
  window_contains_zero := by
    simp [BurgersWindow.Contains, TimeWindow.Contains, zeroTimeWindow]
  solution := zeroBurgersCurve nu
  solutionCertifiedOnWindow :=
    (zeroBurgersCurve_solves nu).on_burgersWindow
      { time := zeroTimeWindow }
  energy :=
    { window := zeroTimeWindow
      certified := zeroLocalEnergy nu }
  compactness :=
    { state := zeroState
      profileSet := {zeroState}
      contains_state := by simp
      finite_profileSet := Set.finite_singleton zeroState }
  poincare :=
    { state := zeroState
      certified := zeroLocalPoincare }
  meanSector :=
    { window := zeroTimeWindow
      certified := zeroLocalMeanSector nu }
  dissipativeWindow :=
    { window := zeroTimeWindow
      certified := zeroLocalDissipativeWindow nu }
  representation :=
    { dictionary := identityRepresentationDictionary }
  badGermCapacity :=
    { germ := zeroBadGerm
      admissible := zeroState_isPeriodicH1 }
  badPatternLibrary := burgersFiniteTimeH1BadPatternLibraryCertificateData

theorem zeroCoreCertificateBundle_sound
    (nu : BurgersParameters) :
    (zeroCoreCertificateBundle nu).allCertificatesSound :=
  BurgersGroundTruthCoreCertificateBundle.localCertificatesSound
    (zeroCoreCertificateBundle nu)

/-- The fully proved equilibrium part of the Burgers route. This is not the
classical arbitrary-data theorem: it is the fully internal template instance
for the zero equilibrium. -/
structure BurgersZeroEquilibriumTemplateClaim
    (nu : BurgersParameters) where
  finalCertificateRecorded :
    burgersGroundTruthFinalCertificateChain.designatedGoal ∈
      burgersGroundTruthFinalCertificateChain.certificates
  routeValidity :
    burgersGroundTruthRunValidity.meetsTemplateCompletionCriteria
  solves : SolvesViscousBurgersWeak nu zeroState (zeroBurgersCurve nu)
  globalH1 : GlobalH1Solution (zeroBurgersCurve nu)
  smoothPositiveTime : ∀ t : ℝ, 0 < t → SmoothAtPositiveTime (zeroBurgersCurve nu) t
  localCertificatesSound :
    (zeroCoreCertificateBundle nu).allCertificatesSound

def burgers_zeroEquilibrium_template_theorem
    (nu : BurgersParameters) :
    BurgersZeroEquilibriumTemplateClaim nu where
  finalCertificateRecorded := burgers_groundTruth_final_certificate_recorded
  routeValidity := burgersGroundTruthRunValidity_holds
  solves := zeroBurgersCurve_solves nu
  globalH1 := zeroBurgersCurve_globalH1 nu
  smoothPositiveTime := zeroBurgersCurve_smoothPositive nu
  localCertificatesSound := zeroCoreCertificateBundle_sound nu

end

end Hypostructure.Backends.Burgers1D
