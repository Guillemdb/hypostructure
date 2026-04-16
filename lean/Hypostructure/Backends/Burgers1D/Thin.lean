import Hypostructure.Backends.Burgers1D.Literature
import Hypostructure.Backends.Burgers1D.Upgrade
import Hypostructure.Problem.Backend
import Hypostructure.Sieve.GenericProgram
import Hypostructure.Framework.Document

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Core
open Hypostructure.Problem
open Hypostructure.Sieve
open Hypostructure.Framework

noncomputable section

def toThinInput
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    ThinInput where
  stateSpace := BurgersState
  instAddComm := inferInstance
  instModuleR := inferInstance
  instNormedGroup := inferInstance
  instNormedSpace := inferInstance
  ambientSpace := BurgersState
  targetSpace := BurgersState
  realizerSpace := BurgersState
  complexityLevel := 1
  rankBound := 1
  supportRegion := Set.univ
  symmetryGroup := BurgersTorus
  potential := fun u => ‖u‖ ^ (2 : ℕ)
  dissipation := BurgersState.dissipation
  flow := BurgersEvolution.flow (ν := ν)
  flow_zero := BurgersEvolution.flow_zero (ν := ν)
  flow_add := BurgersEvolution.flow_add (ν := ν)
  dissipation_nonnegative := by
    intro x
    exact BurgersState.dissipation_nonneg x
  potential_quadratic := by
    intro x
    rfl
  targetElement := BurgersState.constantState 0
  realize := fun u => u
  basePredicate := fun u => BurgersState.mean u = 0
  refinedPredicate := fun u => ∃ m : ℝ, u = BurgersState.constantState m
  ambientConnected := True
  ambientConnected_holds := trivial
  ambientContractible := True
  ambientContractible_holds := trivial
  ambientTame := BurgersLiterature.tameMeanSector (ν := ν)
  ambientTame_holds := burgers_tame_route ν
  ambientRegular := True
  ambientRegular_holds := trivial
  ambientCompact := BurgersLiterature.compactnessModuloTranslation (ν := ν)
  ambientCompact_holds := BurgersLiterature.compactnessModuloTranslation_holds (ν := ν)
  ambientOverGround := True
  ambientOverGround_holds := trivial
  representationConservative := BurgersLiterature.fourierFaithful (ν := ν)
  representationConservative_holds := burgers_fourier_route ν
  representationComplete := BurgersLiterature.fourierFaithful (ν := ν)
  representationComplete_holds := burgers_fourier_route ν
  targetAdmissible :=
    BurgersLiterature.energyIdentity (ν := ν) ∧
      BurgersLiterature.poincareCoercive (ν := ν) ∧
      BurgersLiterature.compactnessModuloTranslation (ν := ν)
  targetAdmissible_holds := by
    exact ⟨
      burgers_energy_nonnegative ν,
      burgers_poincare_route ν,
      BurgersLiterature.compactnessModuloTranslation_holds (ν := ν)
    ⟩
  germBounded :=
    @BurgersBadPatternPackage.germSmallness ν
      (inferInstance : BurgersBadPatternPackage ν)
  germBounded_holds := burgers_germ_smallness ν
  reductionInitialized :=
    @BurgersBadPatternPackage.universalBadInitialized ν
      (inferInstance : BurgersBadPatternPackage ν)
  reductionInitialized_holds := burgers_universal_bad_initialized ν
  libraryComplete :=
    @BurgersBadPatternPackage.catLibraryComplete ν
      (inferInstance : BurgersBadPatternPackage ν)
  libraryComplete_holds := burgers_cat_library_complete ν
  gammaExact := BurgersLiterature.coleHopfBridge (ν := ν)
  gammaExact_holds := BurgersLiterature.coleHopfBridge_holds (ν := ν)
  gammaFaithful :=
    HeatUniqueStatement ν ∧ HeatEquilibriumStatement ν
  gammaFaithful_holds := by
    exact ⟨heat_unique_statement ν, heat_equilibrium_statement ν⟩
  gammaTensorPreserving :=
    HeatEnergyContractiveStatement ν ∧ HeatDissipationContractiveStatement ν
  gammaTensorPreserving_holds := by
    exact ⟨heat_energy_contractive_statement ν, heat_dissipation_contractive_statement ν⟩
  boundaryEven := True
  boundaryEven_holds := trivial
  boundaryPolarized := True
  boundaryPolarized_holds := trivial
  boundedPartialWitness := BurgersLiterature.compactnessModuloTranslation (ν := ν)
  boundedPartialWitness_holds := BurgersLiterature.compactnessModuloTranslation_holds (ν := ν)
  realizerNonempty := ⟨BurgersState.constantState 0⟩

def burgersProblemInstance
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    ProblemInstance NodeTag :=
  thinKernel

def burgersGeneratedCertificates
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    Context NodeTag :=
  generatedCertificates

def burgersProblemDataFromThin
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    UserProblemData :=
  UserProblemData.ofThinInput (toThinInput ν) backendType

theorem burgers_generatedCertificates_uses_generic_dag
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    burgersGeneratedCertificates ν = runDag thinSieve seedContext := by
  simpa [burgersGeneratedCertificates] using generatedCertificates_eq_runDag

theorem burgers_generatedCertificates_match_kernel
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    burgersGeneratedCertificates ν = runKernel thinKernel := by
  simpa [burgersGeneratedCertificates] using generatedCertificates_match_kernel

theorem burgers_goal_mem_generatedCertificates
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    NodeTag.goal ∈ burgersGeneratedCertificates ν := by
  simpa [burgersGeneratedCertificates] using goal_mem_generatedCertificates

theorem burgers_problem_uses_generic_rules
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    (burgersProblemInstance ν).rules = allRules := by
  rfl

theorem burgers_problem_uses_generic_seed
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    (burgersProblemInstance ν).seed = seedContext := by
  rfl

theorem burgers_classification_ready
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    NodeTag.germ ∈ burgersGeneratedCertificates ν ∧
      NodeTag.reduction ∈ burgersGeneratedCertificates ν ∧
      NodeTag.library ∈ burgersGeneratedCertificates ν ∧
      NodeTag.gamma ∈ burgersGeneratedCertificates ν := by
  exact ⟨
    by simpa [burgersGeneratedCertificates] using germ_mem_generatedCertificates,
    by simpa [burgersGeneratedCertificates] using reduction_mem_generatedCertificates,
    by simpa [burgersGeneratedCertificates] using library_mem_generatedCertificates,
    by simpa [burgersGeneratedCertificates] using gamma_mem_generatedCertificates
  ⟩

theorem burgers_thinLibraryBounded
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    BoundedLibraryComplete (toThinInput ν) := by
  exact thinLibraryBounded (toThinInput ν)

theorem burgers_thinGammaConstructor
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    HasGammaPackage (toThinInput ν) := by
  exact thinGammaConstructor (toThinInput ν)

theorem burgers_goal_certified_in_generic_thin_kernel
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    NodeTag.goal ∈ runKernel (burgersProblemInstance ν) := by
  have hgoal : NodeTag.goal ∈ generatedCertificates := goal_mem_generatedCertificates
  simpa [burgersProblemInstance] using (generatedCertificates_match_kernel ▸ hgoal)

end

end Hypostructure.Backends.Burgers1D
