import Hypostructure.Backends.Burgers1D.GroundTruthFinal

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Framework

noncomputable section

def constantEquilibriumState (m : ℝ) : PeriodicH1State :=
  PeriodicH1State.constantState m

theorem constantEquilibriumState_isPeriodicH1
    (m : ℝ) :
    PeriodicH1State.IsPeriodicH1 (constantEquilibriumState m) :=
  PeriodicH1State.isPeriodicH1 (constantEquilibriumState m)

theorem constantEquilibriumState_mean
    (m : ℝ) :
    PeriodicH1State.mean (constantEquilibriumState m) = m := by
  simp [constantEquilibriumState, PeriodicH1State.mean_constantState]

theorem constantEquilibriumState_derivativeEnergy
    (m : ℝ) :
    PeriodicH1State.derivativeEnergy (constantEquilibriumState m) = 0 := by
  simp [constantEquilibriumState, PeriodicH1State.derivativeEnergy,
    PeriodicH1State.constantState, PeriodicH1State.zeroDerivative]

theorem constantEquilibriumState_dissipation
    (nu : BurgersParameters)
    (m : ℝ) :
    PeriodicH1State.dissipation nu (constantEquilibriumState m) = 0 := by
  simp [PeriodicH1State.dissipation_eq_viscosity_mul_derivativeEnergy,
    constantEquilibriumState_derivativeEnergy]

theorem zeroDerivative_hasWeakDerivativeOrder
    (k : ℕ) :
    HasWeakDerivativeOrder k PeriodicH1State.zeroDerivative := by
  induction k with
  | zero =>
      exact continuousMap_memℒp_two PeriodicH1State.zeroDerivative
  | succ k ih =>
      exact ⟨
        PeriodicH1State.zeroDerivative,
        by simpa using PeriodicH1State.zero_isWeakDerivative_constant 0,
        ih
      ⟩

theorem constantEquilibriumState_hasWeakDerivativeOrder
    (m : ℝ)
    (k : ℕ) :
    HasWeakDerivativeOrder k (constantEquilibriumState m).value := by
  cases k with
  | zero =>
      exact (constantEquilibriumState m).value_memL2
  | succ k =>
      exact ⟨
        (constantEquilibriumState m).weakDeriv,
        (constantEquilibriumState m).weakDeriv_spec,
        by
          simpa [constantEquilibriumState, PeriodicH1State.constantState,
            PeriodicH1State.zeroDerivative] using
              zeroDerivative_hasWeakDerivativeOrder k
      ⟩

theorem constantEquilibriumState_smooth
    (m : ℝ) :
    SmoothPeriodicState (constantEquilibriumState m) := by
  intro k
  exact constantEquilibriumState_hasWeakDerivativeOrder m k

def constantBurgersCurve
    (nu : BurgersParameters)
    (m : ℝ) : BurgersSolutionCurve nu where
  eval := fun _ => constantEquilibriumState m

theorem constantBurgersCurve_initial
    (nu : BurgersParameters)
    (m : ℝ) :
    InitialCondition (constantEquilibriumState m) (constantBurgersCurve nu m) := by
  rfl

theorem constantBurgersCurve_boundary
    (nu : BurgersParameters)
    (m : ℝ) :
    PeriodicBoundary (constantBurgersCurve nu m) := by
  intro _t
  exact constantEquilibriumState_isPeriodicH1 m

theorem constantBurgersCurve_weakResidual
    (nu : BurgersParameters)
    (m : ℝ) :
    WeakBurgersResidual nu (constantBurgersCurve nu m) := by
  intro φ
  convert φ.constant_burgers_cancellation m using 4
  simp [burgersWeakResidualIntegrand, constantBurgersCurve,
    constantEquilibriumState, PeriodicH1State.constantState,
    PeriodicH1State.constantProfile, PeriodicH1State.zeroDerivative,
    mul_assoc]

theorem constantBurgersCurve_solves
    (nu : BurgersParameters)
    (m : ℝ) :
    SolvesViscousBurgersWeak nu
      (constantEquilibriumState m)
      (constantBurgersCurve nu m) := by
  exact ⟨
    constantBurgersCurve_initial nu m,
    constantBurgersCurve_boundary nu m,
    constantBurgersCurve_weakResidual nu m
  ⟩

theorem constantBurgersCurve_globalH1
    (nu : BurgersParameters)
    (m : ℝ) :
    GlobalH1Solution (constantBurgersCurve nu m) := by
  intro _t
  exact constantEquilibriumState_isPeriodicH1 m

theorem constantBurgersCurve_smoothPositive
    (nu : BurgersParameters)
    (m t : ℝ)
    (_ht : 0 < t) :
    SmoothAtPositiveTime (constantBurgersCurve nu m) t :=
  constantEquilibriumState_smooth m

def constantHeatCurve
    (nu : BurgersParameters)
    (m : ℝ) : PeriodicHeatCurve nu where
  eval := fun _ => constantEquilibriumState m

theorem constantHeatCurve_initial
    (nu : BurgersParameters)
    (m : ℝ) :
    HeatInitialCondition (constantEquilibriumState m) (constantHeatCurve nu m) := by
  rfl

theorem constantHeatCurve_boundary
    (nu : BurgersParameters)
    (m : ℝ) :
    HeatPeriodicBoundary (constantHeatCurve nu m) := by
  intro _t
  exact constantEquilibriumState_isPeriodicH1 m

theorem constantHeatCurve_weakResidual
    (nu : BurgersParameters)
    (m : ℝ) :
    WeakHeatResidual nu (constantHeatCurve nu m) := by
  intro φ
  convert φ.constant_heat_cancellation m using 4
  simp [heatWeakResidualIntegrand, constantHeatCurve,
    constantEquilibriumState, PeriodicH1State.constantState,
    PeriodicH1State.constantProfile, PeriodicH1State.zeroDerivative,
    mul_assoc]

theorem constantHeatCurve_solves
    (nu : BurgersParameters)
    (m : ℝ) :
    SolvesPeriodicHeatWeak nu (constantEquilibriumState m) (constantHeatCurve nu m) := by
  exact ⟨
    constantHeatCurve_initial nu m,
    constantHeatCurve_boundary nu m,
    constantHeatCurve_weakResidual nu m
  ⟩

theorem constantHeatCurve_globalH1
    (nu : BurgersParameters)
    (m : ℝ) :
    HeatGlobalH1 (constantHeatCurve nu m) := by
  intro _t
  exact constantEquilibriumState_isPeriodicH1 m

theorem constantHeatCurve_smoothPositive
    (nu : BurgersParameters)
    (m t : ℝ)
    (_ht : 0 < t) :
    HeatSmoothAtPositiveTime (constantHeatCurve nu m) t :=
  constantEquilibriumState_smooth m

def constantHeatFlow
    (_nu : BurgersParameters)
    (_t : ℝ)
    (m : ℝ) : PeriodicH1State :=
  constantEquilibriumState m

theorem constantHeatFlow_zero
    (nu : BurgersParameters)
    (m : ℝ) :
    constantHeatFlow nu 0 m = constantEquilibriumState m :=
  rfl

theorem constantHeatFlow_add
    (nu : BurgersParameters)
    (s t m : ℝ) :
    constantHeatFlow nu (s + t) m =
      constantHeatFlow nu s m :=
  rfl

theorem constantHeatFlow_fixed
    (nu : BurgersParameters)
    (t m : ℝ) :
    constantHeatFlow nu t m = PeriodicH1State.constantState m :=
  rfl

theorem constantHeatFlow_mean_preserving
    (nu : BurgersParameters)
    (t m : ℝ) :
    PeriodicH1State.mean (constantHeatFlow nu t m) =
      PeriodicH1State.mean (constantEquilibriumState m) := by
  simp [constantHeatFlow, constantEquilibriumState_mean]

theorem constantHeatFlow_energy_contraction
    (nu : BurgersParameters)
    (t m : ℝ)
    (_ht : 0 ≤ t) :
    PeriodicH1State.energy (constantHeatFlow nu t m) ≤
      PeriodicH1State.energy (constantEquilibriumState m) := by
  rfl

theorem constantHeatFlow_dissipation_contraction
    (nu : BurgersParameters)
    (t m : ℝ)
    (_ht : 0 ≤ t) :
    PeriodicH1State.derivativeEnergy (constantHeatFlow nu t m) ≤
      PeriodicH1State.derivativeEnergy (constantEquilibriumState m) := by
  rfl

def constantEquilibriumTimeWindow : TimeWindow where
  t0 := 0
  t1 := 0
  ordered := le_rfl

def constantHeatWindow : HeatWindow where
  time := constantEquilibriumTimeWindow

def constantCertifiedHeatWindow
    (nu : BurgersParameters)
    (m : ℝ) : CertifiedHeatWindow nu where
  window := constantHeatWindow
  initial := constantEquilibriumState m
  curve := constantHeatCurve nu m
  initial_on_window := by
    intro _h0
    exact (constantHeatCurve_solves nu m).1
  boundary_on_window := by
    exact (constantHeatCurve_solves nu m).restrictsToWindow constantHeatWindow

theorem constantHeatWindow_residual
    (nu : BurgersParameters)
    (m : ℝ) :
    HeatWindowResidual nu (constantCertifiedHeatWindow nu m) := by
  intro φ hφ
  exact (constantHeatCurve_weakResidual nu m).on_heatWindow
    constantHeatWindow φ hφ

theorem constantHeatWindow_uniqueness
    (nu : BurgersParameters)
    (m : ℝ) :
    HeatWindowUniqueness nu (constantCertifiedHeatWindow nu m) := by
  intro w hw t ht _ht_nonneg
  have ht0 : t = 0 := le_antisymm ht.2 ht.1
  subst t
  simpa [constantCertifiedHeatWindow, constantHeatCurve]
    using hw.1

theorem constantHeatWindow_smoothing
    (nu : BurgersParameters)
    (m : ℝ) :
    HeatWindowSmoothing (constantCertifiedHeatWindow nu m) := by
  intro t ht htpos
  have ht0 : t = 0 := le_antisymm ht.2 ht.1
  rw [ht0] at htpos
  exact (False.elim (lt_irrefl (0 : ℝ) htpos))

theorem constantHeatWindow_energy_contraction
    (nu : BurgersParameters)
    (m : ℝ) :
    HeatWindowEnergyContraction (constantCertifiedHeatWindow nu m) := by
  intro _t _ht _ht_nonneg
  rfl

theorem constantHeatWindow_dissipation_contraction
    (nu : BurgersParameters)
    (m : ℝ) :
    HeatWindowDissipationContraction (constantCertifiedHeatWindow nu m) := by
  intro _t _ht _ht_nonneg
  rfl

theorem constantHeatWindow_excludes_bad_germs
    (nu : BurgersParameters)
    (m : ℝ) :
    HeatBadGermWindowExclusion
      (constantCertifiedHeatWindow nu m).window
      (HeatForbiddenBadGerm (constantCertifiedHeatWindow nu m)) :=
  HeatForbiddenBadGerm.excluded (constantHeatWindow_smoothing nu m)

def constantLocalHeatWindowCertificate
    (nu : BurgersParameters)
    (m : ℝ) : LocalHeatWindowCertificate nu where
  certified := constantCertifiedHeatWindow nu m
  residual := constantHeatWindow_residual nu m
  unique := constantHeatWindow_uniqueness nu m
  smooth := constantHeatWindow_smoothing nu m
  energy_contraction := constantHeatWindow_energy_contraction nu m
  dissipation_contraction := constantHeatWindow_dissipation_contraction nu m
  excludes_heat_bad_germs := constantHeatWindow_excludes_bad_germs nu m

theorem constantLocalEnergy
    (nu : BurgersParameters)
    (m : ℝ) :
    LocalEnergyIdentity nu (constantBurgersCurve nu m) constantEquilibriumTimeWindow := by
  simp [LocalEnergyIdentity, FiniteWindowEnergyInequality, timeIntegralOn,
    constantEquilibriumTimeWindow, constantBurgersCurve,
    constantEquilibriumState_dissipation]

theorem constantLocalPoincare
    (m : ℝ) :
    LocalPoincareCoercivity (constantEquilibriumState m) := by
  intro hmean
  have hm : m = 0 := by
    simpa [constantEquilibriumState_mean] using hmean
  subst m
  simp [LocalPoincareCoercivity, constantEquilibriumState,
    PeriodicH1State.energy, PeriodicH1State.derivativeEnergy,
    PeriodicH1State.constantState, PeriodicH1State.constantProfile,
    PeriodicH1State.zeroDerivative]

theorem constantLocalMeanSector
    (nu : BurgersParameters)
    (m : ℝ) :
    LocalMeanSectorPreservation nu
      (constantEquilibriumState m)
      (constantBurgersCurve nu m)
      constantEquilibriumTimeWindow := by
  intro _BW _hW _hsol t _ht0 _ht1
  simp [constantBurgersCurve, constantEquilibriumState_mean]

theorem constantLocalDissipativeWindow
    (nu : BurgersParameters)
    (m : ℝ) :
    LocalDissipativeWindow nu (constantBurgersCurve nu m) constantEquilibriumTimeWindow := by
  intro _t _ht0 _ht1
  simp [constantBurgersCurve]

def constantBadGerm
    (m : ℝ) : BurgersBadGerm where
  centerTime := 0
  centerSpace := 0
  localWindow :=
    { centerTime := 0
      radius := 1
      radius_pos := by norm_num
      centerSpace := 0 }
  profile := constantEquilibriumState m

def identityPeriodicH1RepresentationDictionary : LocalRepresentationDictionary where
  Code := PeriodicH1State
  encode := id
  decode := id
  faithful := by intro u; rfl

def constantCoreCertificateBundle
    (nu : BurgersParameters)
    (m : ℝ) :
    BurgersGroundTruthCoreCertificateBundle nu where
  u0 := constantEquilibriumState m
  window := { time := constantEquilibriumTimeWindow }
  window_contains_zero := by
    simp [BurgersWindow.Contains, TimeWindow.Contains, constantEquilibriumTimeWindow]
  solution := constantBurgersCurve nu m
  solutionCertifiedOnWindow :=
    (constantBurgersCurve_solves nu m).on_burgersWindow
      { time := constantEquilibriumTimeWindow }
  energy :=
    { window := constantEquilibriumTimeWindow
      certified := constantLocalEnergy nu m }
  compactness :=
    { state := constantEquilibriumState m
      profileSet := {constantEquilibriumState m}
      contains_state := by simp
      finite_profileSet := Set.finite_singleton (constantEquilibriumState m) }
  poincare :=
    { state := constantEquilibriumState m
      certified := constantLocalPoincare m }
  meanSector :=
    { window := constantEquilibriumTimeWindow
      certified := constantLocalMeanSector nu m }
  dissipativeWindow :=
    { window := constantEquilibriumTimeWindow
      certified := constantLocalDissipativeWindow nu m }
  representation :=
    { dictionary := identityPeriodicH1RepresentationDictionary }
  badGermCapacity :=
    { germ := constantBadGerm m
      admissible := constantEquilibriumState_isPeriodicH1 m }
  badPatternLibrary := burgersFiniteTimeH1BadPatternLibraryCertificateData

theorem constantCoreCertificateBundle_sound
    (nu : BurgersParameters)
    (m : ℝ) :
    (constantCoreCertificateBundle nu m).allCertificatesSound :=
  BurgersGroundTruthCoreCertificateBundle.localCertificatesSound
    (constantCoreCertificateBundle nu m)

def IsConstantPeriodicH1State (u : PeriodicH1State) : Prop :=
  ∃ m : ℝ, u = constantEquilibriumState m

def constantSectorColeHopfTransform : ColeHopfTransform where
  toHeat := id
  fromHeat := id
  burgersSector := IsConstantPeriodicH1State
  burgersPDEDomain := IsConstantPeriodicH1State
  heatSector := IsConstantPeriodicH1State
  toHeat_preservesH1 := by
    intro u hu
    exact hu
  fromHeat_preservesH1 := by
    intro theta htheta
    exact htheta
  burgersPDEDomain_in_sector := by
    intro u hdomain
    exact hdomain
  toHeat_sector := by
    intro u hu
    exact hu
  fromHeat_sector := by
    intro theta htheta
    exact htheta
  left_inverse := by
    intro _u _hu
    rfl
  right_inverse := by
    intro _theta _htheta
    rfl
  smoothness_reflects_from_heat_image := by
    intro u theta _hdomain htheta hsmooth
    simpa [htheta]
      using hsmooth

def constantSectorColeHopfBackend
    (nu : BurgersParameters) : PeriodicColeHopfBackend nu where
  transform := constantSectorColeHopfTransform
  chart_valid := by
    intro H
    intro C _hwindow hsector hinit
    simpa [hinit] using hsector
  maps_window := by
    intro H
    intro C hwindow _hsector hinit
    exact ⟨hwindow, hinit⟩
  inverse_maps_window := by
    intro H
    intro C _hwindow hsector
    rcases hsector with ⟨m, hm⟩
    refine ⟨
      { window := { time := C.window.time }
        initial := constantEquilibriumState m
        curve := constantBurgersCurve nu m
        solvesOnWindow :=
          (constantBurgersCurve_solves nu m).on_burgersWindow
            { time := C.window.time } },
      ?_, ?_
    ⟩
    · rfl
    · exact hm.symm
  burgers_to_heat_residual_on_window := by
    intro H
    intro C _hwindow _hsector hinit
    simpa [hinit] using H.residual
  heat_to_burgers_residual_on_window := by
    intro H
    intro C _hwindow hsector
    rcases hsector with ⟨m, hm⟩
    refine ⟨
      { window := { time := C.window.time }
        initial := constantEquilibriumState m
        curve := constantBurgersCurve nu m
        solvesOnWindow :=
          (constantBurgersCurve_solves nu m).on_burgersWindow
            { time := C.window.time } },
      ?_, ?_
    ⟩
    · rfl
    · exact hm.symm
  uniqueness_transfer_on_window := by
    intro H
    intro _C _hwindow _hsector _hinit
    exact H.unique

def constantSectorColeHopfBridge
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) :
    LocalColeHopfWindowBridge nu H :=
  (constantSectorColeHopfBackend nu).windowBridge H

theorem constantSectorColeHopfBridge_sound
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) :
    (constantSectorColeHopfBridge nu H).bridgeStatement :=
  PeriodicColeHopfBackend.windowBridge_sound
    (constantSectorColeHopfBackend nu) H

def constantLocalColeHopfWindowBridge
    (nu : BurgersParameters)
    (m : ℝ) :
    LocalColeHopfWindowBridge nu (constantLocalHeatWindowCertificate nu m) :=
  constantSectorColeHopfBridge nu (constantLocalHeatWindowCertificate nu m)

theorem constantLocalHeatWindowCertificate_sound
    (nu : BurgersParameters)
    (m : ℝ) :
    (constantLocalHeatWindowCertificate nu m).certificateStatement :=
  localHeatSmoothingFrameworkCertificate_sound
    nu (constantLocalHeatWindowCertificate nu m)

theorem constantLocalColeHopfWindowBridge_sound
    (nu : BurgersParameters)
    (m : ℝ) :
    (constantLocalColeHopfWindowBridge nu m).bridgeStatement :=
  constantSectorColeHopfBridge_sound
    nu (constantLocalHeatWindowCertificate nu m)

theorem constantLockBlocksBadGerms
    (nu : BurgersParameters)
    (m : ℝ) :
    LockBlocksBurgersBadGerms
      nu
      (constantCoreCertificateBundle nu m)
      (constantLocalHeatWindowCertificate nu m)
      (constantLocalColeHopfWindowBridge nu m) :=
  lockBlocksBadGermsFromLocalCertificates
    nu
    (constantCoreCertificateBundle nu m)
    (constantLocalHeatWindowCertificate nu m)
    (constantLocalColeHopfWindowBridge nu m)

def constantAnalyticUpgradeInput
    (nu : BurgersParameters)
    (m : ℝ) : BurgersAnalyticUpgradeInput nu :=
  BurgersAnalyticUpgradeInput.fromLocalCertificates
    nu
    (constantCoreCertificateBundle nu m)
    (constantLocalHeatWindowCertificate nu m)
    (constantLocalColeHopfWindowBridge nu m)

structure BurgersConstantEquilibriumTemplateClaim
    (nu : BurgersParameters)
    (m : ℝ) where
  finalCertificateRecorded :
    burgersGroundTruthFinalCertificateChain.designatedGoal ∈
      burgersGroundTruthFinalCertificateChain.certificates
  routeValidity :
    burgersGroundTruthRunValidity.meetsTemplateCompletionCriteria
  burgersSolves :
    SolvesViscousBurgersWeak nu
      (constantEquilibriumState m)
      (constantBurgersCurve nu m)
  burgersGlobalH1 : GlobalH1Solution (constantBurgersCurve nu m)
  burgersSmoothPositiveTime :
    ∀ t : ℝ, 0 < t → SmoothAtPositiveTime (constantBurgersCurve nu m) t
  heatSolves :
    SolvesPeriodicHeatWeak nu
      (constantEquilibriumState m)
      (constantHeatCurve nu m)
  heatGlobalH1 : HeatGlobalH1 (constantHeatCurve nu m)
  heatSmoothPositiveTime :
    ∀ t : ℝ, 0 < t → HeatSmoothAtPositiveTime (constantHeatCurve nu m) t
  localCertificatesSound :
    (constantCoreCertificateBundle nu m).allCertificatesSound

def burgers_constantEquilibrium_template_theorem
    (nu : BurgersParameters)
    (m : ℝ) :
    BurgersConstantEquilibriumTemplateClaim nu m where
  finalCertificateRecorded := burgers_groundTruth_final_certificate_recorded
  routeValidity := burgersGroundTruthRunValidity_holds
  burgersSolves := constantBurgersCurve_solves nu m
  burgersGlobalH1 := constantBurgersCurve_globalH1 nu m
  burgersSmoothPositiveTime := constantBurgersCurve_smoothPositive nu m
  heatSolves := constantHeatCurve_solves nu m
  heatGlobalH1 := constantHeatCurve_globalH1 nu m
  heatSmoothPositiveTime := constantHeatCurve_smoothPositive nu m
  localCertificatesSound := constantCoreCertificateBundle_sound nu m

end

end Hypostructure.Backends.Burgers1D
