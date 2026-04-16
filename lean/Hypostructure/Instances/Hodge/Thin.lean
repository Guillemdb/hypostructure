import Hypostructure.Problem.Thin
import Hypostructure.Problem.Backend
import Hypostructure.Sieve.GenericProgram
import HypoHodge.Algebraic.VerifiedThinInput

namespace Hypostructure.Instances.Hodge

open Hypostructure.Core
open Hypostructure.Problem
open Hypostructure.Sieve
open HypoHodge.Algebraic

def toThinInput (I : VerifiedHodgeThinInput) : ThinInput where
  stateSpace := I.V
  instAddComm := I.instAddComm
  instModuleR := I.instModuleR
  instNormedGroup := I.instNormedGroup
  instNormedSpace := I.instNormedSpace
  ambientSpace := I.varietyType
  targetSpace := I.cohomologyType
  realizerSpace := I.cycleType
  complexityLevel := I.p
  rankBound := I.Qrank
  supportRegion := I.hodgeSubset
  symmetryGroup := I.symmetry
  potential := I.potential
  dissipation := I.dissipation
  flow := I.flow
  flow_zero := by
    intro x
    simpa using I.flow_id 0 x
  flow_add := by
    intro s t x
    rw [I.flow_id, I.flow_id, I.flow_id]
  dissipation_nonnegative := by
    intro x
    simpa [I.dissipation_zero x]
  potential_quadratic := I.potential_quad
  targetElement := I.hodgeClass
  realize := I.cycleClass
  basePredicate := I.rationalClass
  refinedPredicate := I.hodgePredicate
  ambientConnected := I.connected
  ambientConnected_holds := I.connected_holds
  ambientContractible := I.contractible
  ambientContractible_holds := I.contractible_holds
  ambientTame := I.tame
  ambientTame_holds := I.tame_holds
  ambientRegular := I.varietySmooth
  ambientRegular_holds := I.varietySmooth_holds
  ambientCompact := I.varietyProjective
  ambientCompact_holds := I.varietyProjective_holds
  ambientOverGround := I.varietyOverComplex
  ambientOverGround_holds := I.varietyOverComplex_holds
  representationConservative := I.representationConservative
  representationConservative_holds := I.representationConservative_holds
  representationComplete := I.representationComplete
  representationComplete_holds := I.representationComplete_holds
  targetAdmissible := I.targetAdmissible
  targetAdmissible_holds := I.targetAdmissible_holds
  germBounded := I.germBounded
  germBounded_holds := I.germBounded_holds
  reductionInitialized := I.initialReduction
  reductionInitialized_holds := I.initialReduction_holds
  libraryComplete := I.categoryLibraryComplete
  libraryComplete_holds := I.categoryLibraryComplete_holds
  gammaExact := I.gammaExact
  gammaExact_holds := I.gammaExact_holds
  gammaFaithful := I.gammaFaithful
  gammaFaithful_holds := I.gammaFaithful_holds
  gammaTensorPreserving := I.gammaTensorPres
  gammaTensorPreserving_holds := I.gammaTensorPres_holds
  boundaryEven := I.targetWeightEven
  boundaryEven_holds := I.targetWeightEven_holds
  boundaryPolarized := I.targetPolarizable
  boundaryPolarized_holds := I.targetPolarizable_holds
  boundedPartialWitness := I.boundedPartialWitness
  boundedPartialWitness_holds := I.boundedPartialWitness_holds
  realizerNonempty := I.cycleCarrierNonempty

def hodgeProblemInstance (_I : VerifiedHodgeThinInput) : ProblemInstance NodeTag :=
  thinKernel

def hodgeGeneratedCertificates (_I : VerifiedHodgeThinInput) : Context NodeTag :=
  generatedCertificates

theorem hodge_generatedCertificates_uses_generic_dag (I : VerifiedHodgeThinInput) :
    hodgeGeneratedCertificates I = runDag thinSieve seedContext := by
  simpa [hodgeGeneratedCertificates] using generatedCertificates_eq_runDag

theorem hodge_goal_mem_generatedCertificates (I : VerifiedHodgeThinInput) :
    NodeTag.goal ∈ hodgeGeneratedCertificates I := by
  simpa [hodgeGeneratedCertificates] using goal_mem_generatedCertificates

theorem hodge_problem_uses_generic_rules (I : VerifiedHodgeThinInput) :
    (hodgeProblemInstance I).rules = allRules := by
  rfl

theorem hodge_problem_uses_generic_seed (I : VerifiedHodgeThinInput) :
    (hodgeProblemInstance I).seed = seedContext := by
  rfl

theorem hodge_target_certified_iff (I : VerifiedHodgeThinInput) :
    TargetCertified (toThinInput I) ↔ TargetHodgeClass I := by
  rfl

theorem hodge_target_realized_iff (I : VerifiedHodgeThinInput) :
    TargetRealized (toThinInput I) ↔ TargetRealizedByCycle I := by
  rfl

theorem hodge_target_solved_iff (I : VerifiedHodgeThinInput) :
    TargetSolved (toThinInput I) ↔ HodgeConjectureTarget I := by
  rfl

end Hypostructure.Instances.Hodge
