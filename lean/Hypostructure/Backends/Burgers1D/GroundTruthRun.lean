import Hypostructure.Backends.Burgers1D.GroundTruthCertificates
import Hypostructure.Framework.Route

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Framework

noncomputable section

/-- The local certificate payloads that have already been migrated to the
ground-truth Burgers language. This bundle covers the first five implementation
requirements: concrete state/PDE language, local energy, mean-sector/Poincare,
local dissipative windows, and local bad-germ capacity. -/
structure BurgersGroundTruthCoreCertificateBundle
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

def BurgersGroundTruthCoreCertificateBundle.energyCertificate
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) :
    EnergyCertificate :=
  localEnergyFrameworkCertificate nu B.solution B.energy

def BurgersGroundTruthCoreCertificateBundle.poincareCertificate
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) :
    StiffnessCertificate :=
  localPoincareFrameworkCertificate B.poincare

def BurgersGroundTruthCoreCertificateBundle.compactnessCertificate
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) :
    CompactnessCertificate :=
  localCompactnessFrameworkCertificate B.compactness

def BurgersGroundTruthCoreCertificateBundle.meanSectorCertificate
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) :
    TopologyCertificate :=
  localMeanSectorFrameworkCertificate nu B.u0 B.solution B.meanSector

def BurgersGroundTruthCoreCertificateBundle.dissipativeWindowCertificate
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) :
    MixingCertificate :=
  localDissipativeWindowFrameworkCertificate nu B.solution B.dissipativeWindow

def BurgersGroundTruthCoreCertificateBundle.representationCertificate
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) :
    RepresentationCertificate :=
  localRepresentationFrameworkCertificate B.representation

def BurgersGroundTruthCoreCertificateBundle.badGermCapacityCertificate
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) :
    CapacityCertificate :=
  localBadGermCapacityFrameworkCertificate B.badGermCapacity

def BurgersGroundTruthCoreCertificateBundle.germCertificate
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) :
    GermCertificate :=
  burgersGermFrameworkCertificate B.badPatternLibrary

def BurgersGroundTruthCoreCertificateBundle.initialityCertificate
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) :
    InitialityCertificate :=
  burgersInitialityFrameworkCertificate B.badPatternLibrary

def BurgersGroundTruthCoreCertificateBundle.catLibCertificate
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) :
    CatLibCertificate :=
  burgersCatLibFrameworkCertificate B.badPatternLibrary

def BurgersGroundTruthCoreCertificateBundle.allCertificatesSound
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) : Prop :=
  B.energyCertificate.meaning ∧
    B.compactnessCertificate.meaning ∧
    B.poincareCertificate.meaning ∧
    B.meanSectorCertificate.meaning ∧
    B.dissipativeWindowCertificate.meaning ∧
    B.representationCertificate.meaning ∧
    B.badGermCapacityCertificate.meaning ∧
    B.germCertificate.meaning ∧
    B.initialityCertificate.meaning ∧
    B.catLibCertificate.meaning

theorem BurgersGroundTruthCoreCertificateBundle.localCertificatesSound
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) :
    B.allCertificatesSound := by
  exact ⟨
    localEnergyFrameworkCertificate_sound nu B.solution B.energy,
    localCompactnessFrameworkCertificate_sound B.compactness,
    localPoincareFrameworkCertificate_sound B.poincare,
    localMeanSectorFrameworkCertificate_sound nu B.u0 B.solution B.meanSector,
    localDissipativeWindowFrameworkCertificate_sound nu B.solution B.dissipativeWindow,
    localRepresentationFrameworkCertificate_sound B.representation,
    localBadGermCapacityFrameworkCertificate_sound B.badGermCapacity,
    burgersGermFrameworkCertificate_sound B.badPatternLibrary,
    burgersInitialityFrameworkCertificate_sound B.badPatternLibrary,
    burgersCatLibFrameworkCertificate_sound B.badPatternLibrary
  ⟩

/-- The documented Lock package for the ground-truth route. -/
def burgersGroundTruthLockPackage : LockCertificatePackage where
  tacticName := "E5 Cole-Hopf functional bridge"
  requiredCertificates :=
    [ "K_Germ^+"
    , "K_init^+"
    , "K_CatLib^+"
    , "K_ColeHopf^+"
    , "K_HeatSmooth^+"
    ]
  blockedCertificate := "K_Cat_Hom^blk"

/-- The document's blocked scaling promotion, separated from the final analytic
upgrade. -/
def burgersGroundTruthScalingPromotion : UpgradeRule where
  name := "Burgers1D blocked TypeII scaling continuation"
  premises := [ "K_SC_lambda^-", "K_SC_lambda^blk" ]
  conclusion := "K_SC_lambda^~"
  nonCircular := "K_SC_lambda^~" ∉ [ "K_SC_lambda^-", "K_SC_lambda^blk" ]

/-- The final analytic promotion in the document. This is just the route rule;
the theorem that discharges it lives in `GroundTruthFinal`. -/
def burgersGroundTruthAnalyticUpgradeRule : UpgradeRule where
  name := "Burgers1D ground-truth analytic regularity upgrade"
  premises :=
    [ "K_StructReg_Burgers1D^+"
    , "K_ColeHopf^+"
    , "K_HeatSmooth^+"
    ]
  conclusion := "K_Reg_Burgers1D^+"
  nonCircular :=
    "K_Reg_Burgers1D^+" ∉
      [ "K_StructReg_Burgers1D^+", "K_ColeHopf^+", "K_HeatSmooth^+" ]

/-- Route trace matching `docs/source/dataset/burgers_1d.md`: Node 4 is a
negative scaling certificate, then BarrierTypeII blocks that branch, producing
an effective continuation certificate. -/
def burgersGroundTruthExecutionTrace : ExecutionTrace :=
  [ { index := 1, location := .gate .energyCheck, certificateName := "K_D_E^+",
      outcome := .yes, summary := "ground-truth finite-window energy certificate" }
  , { index := 2, location := .gate .zenoCheck, certificateName := "K_Rec_N^+",
      outcome := .yes, summary := "empty local repair-event list" }
  , { index := 3, location := .gate .compactCheck, certificateName := "K_C_mu^+",
      outcome := .yes, summary := "compactness modulo torus translation, still to be fully mathlib-grounded" }
  , { index := 4, location := .gate .scaleCheck, certificateName := "K_SC_lambda^-",
      outcome := .noWitness, summary := "template-native negative scaling branch" }
  , { index := 5, location := .barrier .barrierTypeII, certificateName := "K_SC_lambda^blk",
      outcome := .blocked, summary := "TypeII branch blocked by the local dissipative/Cole-Hopf route" }
  , { index := 6, location := .derived "Blocked Scaling Promotion", certificateName := "K_SC_lambda^~",
      outcome := .derived, summary := "effective continuation after blocked TypeII scaling branch" }
  , { index := 7, location := .gate .paramCheck, certificateName := "K_SC_dc^+",
      outcome := .yes, summary := "stable parameters (nu, mean)" }
  , { index := 8, location := .gate .geomCheck, certificateName := "K_Cap_H^+",
      outcome := .yes, summary := "local bad-germ capacity bound" }
  , { index := 9, location := .gate .stiffnessCheck, certificateName := "K_LS_sigma^+",
      outcome := .yes, summary := "local Poincare coercivity" }
  , { index := 10, location := .gate .topoCheck, certificateName := "K_TB_pi^+",
      outcome := .yes, summary := "local mean-sector preservation" }
  , { index := 11, location := .gate .tameCheck, certificateName := "K_TB_O^+",
      outcome := .yes, summary := "mean-sector decomposition and local chart discipline" }
  , { index := 12, location := .gate .ergoCheck, certificateName := "K_TB_rho^+",
      outcome := .yes, summary := "local dissipative window" }
  , { index := 13, location := .gate .complexCheck, certificateName := "K_RepDesc_K^+",
      outcome := .yes, summary := "representation dictionary permit, still to be fully Fourier-grounded" }
  , { index := 14, location := .gate .oscillateCheck, certificateName := "K_GC_nabla^+",
      outcome := .yes, summary := "local gradient/Cole-Hopf compatibility permit" }
  , { index := 15, location := .gate .boundaryCheck, certificateName := "K_Bound_partial^-",
      outcome := .noWitness, summary := "closed-system torus branch" }
  , { index := 16, location := .gate .overloadCheck, certificateName := "N/A",
      outcome := .notApplicable, summary := "open-system only" }
  , { index := 17, location := .gate .starveCheck, certificateName := "N/A",
      outcome := .notApplicable, summary := "open-system only" }
  , { index := 18, location := .gate .alignCheck, certificateName := "N/A",
      outcome := .notApplicable, summary := "open-system only" }
  , { index := 19, location := .barrier .barrierExclusion, certificateName := "K_Cat_Hom^blk",
      outcome := .blocked, summary := "Lock blocked by local Cole-Hopf and heat permits" }
  , { index := 20, location := .derived "Structural Exclusion", certificateName := "K_StructReg_Burgers1D^+",
      outcome := .derived, summary := "bad morphism excluded by the local certificate route" }
  , { index := 21, location := .derived "Analytic Upgrade", certificateName := "K_Reg_Burgers1D^+",
      outcome := .derived, summary := "global regularity certificate, conditional on the ground-truth analytic upgrade theorem" }
  ]

def burgersGroundTruthRequiredTraceCertificateNames : List String :=
  [ "K_D_E^+"
  , "K_Rec_N^+"
  , "K_C_mu^+"
  , "K_SC_lambda^-"
  , "K_SC_lambda^blk"
  , "K_SC_lambda^~"
  , "K_SC_dc^+"
  , "K_Cap_H^+"
  , "K_LS_sigma^+"
  , "K_TB_pi^+"
  , "K_TB_O^+"
  , "K_TB_rho^+"
  , "K_RepDesc_K^+"
  , "K_GC_nabla^+"
  , "K_Bound_partial^-"
  , "N/A"
  , "K_Cat_Hom^blk"
  , "K_StructReg_Burgers1D^+"
  , "K_Reg_Burgers1D^+"
  ]

def burgersGroundTruthAnalyticCertificateNames : List String :=
  [ "K_ColeHopf^+"
  , "K_HeatSmooth^+"
  , "K_StructReg_Burgers1D^+"
  ]

def burgersGroundTruthPreservationCertificateNames : List String :=
  [ "K_TB_pi^+"
  , "K_TB_O^+"
  , "K_TB_rho^+"
  ]

def burgersGroundTruthUpgradeRules : List UpgradeRule :=
  [ burgersGroundTruthScalingPromotion
  , burgersGroundTruthAnalyticUpgradeRule
  ]

def burgersGroundTruthObligationLedger : ObligationLedger := []

theorem burgersGroundTruth_goalConeEmpty :
    ObligationLedger.goalConeEmpty burgersGroundTruthObligationLedger :=
  emptyLedger_goalConeEmpty

def burgersGroundTruthFinalCertificateChain : FinalCertificateChain where
  certificates :=
    [ "K_D_E^+"
    , "K_Rec_N^+"
    , "K_C_mu^+"
    , "K_SC_lambda^-"
    , "K_SC_lambda^blk"
    , "K_SC_lambda^~"
    , "K_SC_dc^+"
    , "K_Cap_H^+"
    , "K_LS_sigma^+"
    , "K_TB_pi^+"
    , "K_TB_O^+"
    , "K_TB_rho^+"
    , "K_RepDesc_K^+"
    , "K_GC_nabla^+"
    , "K_Bound_partial^-"
    , "K_Germ^+"
    , "K_init^+"
    , "K_CatLib^+"
    , "K_ColeHopf^+"
    , "K_HeatSmooth^+"
    , "K_Cat_Hom^blk"
    , "K_StructReg_Burgers1D^+"
    , "K_Reg_Burgers1D^+"
    ]
  designatedGoal := "K_Reg_Burgers1D^+"
  containsGoal := by simp

def burgersGroundTruthRoute : TraceBackedRoute where
  trace := burgersGroundTruthExecutionTrace
  requiredTraceCertificates := burgersGroundTruthRequiredTraceCertificateNames
  boundaryCertificate := "K_Bound_partial^-"
  lockCertificate := "K_Cat_Hom^blk"
  preservationCertificates := burgersGroundTruthPreservationCertificateNames
  analyticCertificates := burgersGroundTruthAnalyticCertificateNames
  upgrades := burgersGroundTruthUpgradeRules
  obligationLedger := burgersGroundTruthObligationLedger
  lockPackage := burgersGroundTruthLockPackage
  finalChain := burgersGroundTruthFinalCertificateChain

def burgersGroundTruthRunValidity : RunValidity :=
  burgersGroundTruthRoute.runValidity

theorem burgersGroundTruthTrace_contains_scalingNegative :
    burgersGroundTruthExecutionTrace.contains "K_SC_lambda^-" := by
  simp [ExecutionTrace.contains, ExecutionTrace.names, burgersGroundTruthExecutionTrace]

theorem burgersGroundTruthTrace_contains_typeIIBlocked :
    burgersGroundTruthExecutionTrace.contains "K_SC_lambda^blk" := by
  simp [ExecutionTrace.contains, ExecutionTrace.names, burgersGroundTruthExecutionTrace]

theorem burgersGroundTruthTrace_contains_goal :
    burgersGroundTruthExecutionTrace.contains "K_Reg_Burgers1D^+" := by
  simp [ExecutionTrace.contains, ExecutionTrace.names, burgersGroundTruthExecutionTrace]

theorem burgersGroundTruthFinalChain_contains_goal :
    burgersGroundTruthFinalCertificateChain.designatedGoal ∈
      burgersGroundTruthFinalCertificateChain.certificates :=
  burgersGroundTruthFinalCertificateChain.containsGoal

def burgersGroundTruthRouteProof : TraceBackedRouteProof burgersGroundTruthRoute where
  requiredTraceNodesExecuted := by
    intro cert hcert
    simp [burgersGroundTruthRoute, TraceBackedRoute.requiredTraceNodesExecuted,
      burgersGroundTruthRequiredTraceCertificateNames, ExecutionTrace.contains,
      ExecutionTrace.names, burgersGroundTruthExecutionTrace] at hcert ⊢
    exact hcert
  boundaryHandled := by
    simp [burgersGroundTruthRoute, TraceBackedRoute.boundaryHandled,
      ExecutionTrace.contains, ExecutionTrace.names, burgersGroundTruthExecutionTrace]
  lockExecuted := by
    simp [burgersGroundTruthRoute, TraceBackedRoute.lockExecuted,
      ExecutionTrace.contains, ExecutionTrace.names, burgersGroundTruthExecutionTrace]
  upgradesCompleted := by
    intro rule hrule
    simp [burgersGroundTruthRoute, TraceBackedRoute.upgradesCompleted,
      burgersGroundTruthUpgradeRules, burgersGroundTruthScalingPromotion,
      burgersGroundTruthAnalyticUpgradeRule] at hrule ⊢
    rcases hrule with rfl | rfl <;> trivial
  goalConeEmpty := burgersGroundTruth_goalConeEmpty
  designatedGoalReached := burgersGroundTruthFinalChain_contains_goal
  lockCompletenessPresent := by
    intro cert hcert
    simp [burgersGroundTruthRoute, TraceBackedRoute.lockCompletenessPresent,
      burgersGroundTruthLockPackage, burgersGroundTruthFinalCertificateChain] at hcert ⊢
    rcases hcert with rfl | rfl | rfl | rfl | rfl
    all_goals simp [burgersGroundTruthFinalCertificateChain]
  analyticPermitPresent := by
    intro cert hcert
    simp [burgersGroundTruthRoute, TraceBackedRoute.analyticPermitPresent,
      burgersGroundTruthAnalyticCertificateNames,
      burgersGroundTruthFinalCertificateChain] at hcert ⊢
    rcases hcert with rfl | rfl | rfl
    all_goals simp [burgersGroundTruthFinalCertificateChain]
  preservationLemmasPresent := by
    intro cert hcert
    simp [burgersGroundTruthRoute, TraceBackedRoute.preservationLemmasPresent,
      burgersGroundTruthPreservationCertificateNames, ExecutionTrace.contains,
      ExecutionTrace.names, burgersGroundTruthExecutionTrace] at hcert ⊢
    rcases hcert with rfl | rfl | rfl
    all_goals simp [burgersGroundTruthExecutionTrace, ExecutionTrace.contains,
      ExecutionTrace.names]

theorem burgersGroundTruthRunValidity_holds :
    burgersGroundTruthRunValidity.meetsTemplateCompletionCriteria := by
  exact burgersGroundTruthRouteProof.runValidity_holds

end

end Hypostructure.Backends.Burgers1D
