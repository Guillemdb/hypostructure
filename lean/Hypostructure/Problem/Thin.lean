import Hypostructure.Core.Context
import Hypostructure.Sieve.Node

namespace Hypostructure.Problem

open Hypostructure.Core
open Hypostructure.Sieve

structure ThinInput where
  stateSpace            : Type
  instAddComm           : AddCommGroup stateSpace
  instModuleR           : Module ℝ stateSpace
  instNormedGroup       : NormedAddCommGroup stateSpace
  instNormedSpace       : NormedSpace ℝ stateSpace

  ambientSpace          : Type
  targetSpace           : Type
  realizerSpace         : Type

  complexityLevel       : ℕ
  rankBound             : ℕ
  supportRegion         : Set stateSpace
  symmetryGroup         : Type

  potential             : stateSpace → ℝ
  dissipation           : stateSpace → ℝ
  flow                  : ℝ → stateSpace → stateSpace

  flow_zero             : ∀ x, flow 0 x = x
  flow_add             : ∀ s t x, flow (s + t) x = flow s (flow t x)
  dissipation_nonnegative : ∀ x, 0 ≤ dissipation x
  potential_quadratic   : ∀ x, potential x = ‖x‖ ^ (2 : ℕ)

  targetElement         : targetSpace
  realize               : realizerSpace → targetSpace
  basePredicate         : targetSpace → Prop
  refinedPredicate      : targetSpace → Prop

  ambientConnected      : Prop
  ambientConnected_holds : ambientConnected
  ambientContractible   : Prop
  ambientContractible_holds : ambientContractible
  ambientTame           : Prop
  ambientTame_holds     : ambientTame
  ambientRegular        : Prop
  ambientRegular_holds  : ambientRegular
  ambientCompact        : Prop
  ambientCompact_holds  : ambientCompact
  ambientOverGround     : Prop
  ambientOverGround_holds : ambientOverGround
  representationConservative : Prop
  representationConservative_holds : representationConservative
  representationComplete : Prop
  representationComplete_holds : representationComplete
  targetAdmissible      : Prop
  targetAdmissible_holds : targetAdmissible
  germBounded           : Prop
  germBounded_holds     : germBounded
  reductionInitialized  : Prop
  reductionInitialized_holds : reductionInitialized
  libraryComplete       : Prop
  libraryComplete_holds : libraryComplete
  gammaExact            : Prop
  gammaExact_holds      : gammaExact
  gammaFaithful         : Prop
  gammaFaithful_holds   : gammaFaithful
  gammaTensorPreserving : Prop
  gammaTensorPreserving_holds : gammaTensorPreserving
  boundaryEven          : Prop
  boundaryEven_holds    : boundaryEven
  boundaryPolarized     : Prop
  boundaryPolarized_holds : boundaryPolarized
  boundedPartialWitness : Prop
  boundedPartialWitness_holds : boundedPartialWitness
  realizerNonempty      : Nonempty realizerSpace

attribute [instance] ThinInput.instAddComm
attribute [instance] ThinInput.instModuleR
attribute [instance] ThinInput.instNormedGroup
attribute [instance] ThinInput.instNormedSpace

def TargetCertified (I : ThinInput) : Prop :=
  I.basePredicate I.targetElement ∧ I.refinedPredicate I.targetElement

def TargetRealized (I : ThinInput) : Prop :=
  ∃ Z : I.realizerSpace, I.realize Z = I.targetElement

def TargetSolved (I : ThinInput) : Prop :=
  TargetCertified I → TargetRealized I

structure AmbientCertificate (I : ThinInput) : Prop where
  regular : I.ambientRegular
  compact : I.ambientCompact
  overGround : I.ambientOverGround
  connected : I.ambientConnected

structure RepresentationConservativeCertificate (I : ThinInput) : Prop where
  conservative : I.representationConservative

structure RepresentationCompleteCertificate (I : ThinInput) : Prop where
  complete : I.representationComplete

structure DefectEnergyCertificate (I : ThinInput) : Prop where
  nonnegativeDissipation : ∀ x, 0 ≤ I.dissipation x

structure RecurrenceCertificate (I : ThinInput) : Prop where
  flowZero : ∀ x, I.flow 0 x = x
  flowAdd : ∀ s t x, I.flow (s + t) x = I.flow s (I.flow t x)

structure WitnessMultiplicityCertificate (I : ThinInput) : Prop where
  nonemptyRealizer : Nonempty I.realizerSpace

structure TameCertificate (I : ThinInput) : Prop where
  tame : I.ambientTame

structure PartialCompactnessCertificate (I : ThinInput) : Prop where
  connected : I.ambientConnected

structure CapacityCertificate (I : ThinInput) : Prop where
  contractible : I.ambientContractible

structure LinearStabilityCertificate (I : ThinInput) : Prop where
  quadraticPotential : ∀ x, I.potential x = ‖x‖ ^ (2 : ℕ)

structure BoundaryEvenCertificate (I : ThinInput) : Prop where
  even : I.boundaryEven

structure BoundaryPolarizedCertificate (I : ThinInput) : Prop where
  polarized : I.boundaryPolarized

structure BoundedPartialCertificate (I : ThinInput) : Prop where
  boundedPartial : I.boundedPartialWitness

structure LibraryCertificate (I : ThinInput) : Prop where
  complete : I.libraryComplete

structure ReductionCertificate (I : ThinInput) : Prop where
  initialized : I.reductionInitialized
  library : LibraryCertificate I

structure GermCertificate (I : ThinInput) : Prop where
  bounded : I.germBounded
  reduction : ReductionCertificate I

structure GammaCertificate (I : ThinInput) : Prop where
  exact : I.gammaExact
  faithful : I.gammaFaithful
  tensorPreserving : I.gammaTensorPreserving

structure AdmissibilityCertificate (I : ThinInput) : Prop where
  defect : DefectEnergyCertificate I
  recurrence : RecurrenceCertificate I
  multiplicity : WitnessMultiplicityCertificate I
  tame : TameCertificate I
  partialCompactness : PartialCompactnessCertificate I
  capacity : CapacityCertificate I
  stability : LinearStabilityCertificate I
  boundaryEven : BoundaryEvenCertificate I
  boundaryPolarized : BoundaryPolarizedCertificate I
  boundedPartial : BoundedPartialCertificate I
  admissible : I.targetAdmissible

structure AnchorCertificate (I : ThinInput) : Prop where
  ambient : AmbientCertificate I
  repConservative : RepresentationConservativeCertificate I
  repComplete : RepresentationCompleteCertificate I
  gamma : GammaCertificate I
  germ : GermCertificate I

structure UnresolvedGoalObstruction (I : ThinInput) : Prop where
  targetCertified : TargetCertified I
  noRealizer : ¬ TargetRealized I

structure PromotionGap (I : ThinInput) : Prop where
  blockedByPromotion : ¬ I.gammaTensorPreserving

def ambientCertificate (I : ThinInput) : AmbientCertificate I :=
  ⟨I.ambientRegular_holds, I.ambientCompact_holds, I.ambientOverGround_holds, I.ambientConnected_holds⟩

def representationConservativeCertificate (I : ThinInput) :
    RepresentationConservativeCertificate I :=
  ⟨I.representationConservative_holds⟩

def representationCompleteCertificate (I : ThinInput) :
    RepresentationCompleteCertificate I :=
  ⟨I.representationComplete_holds⟩

def defectEnergyCertificate (I : ThinInput) : DefectEnergyCertificate I :=
  ⟨I.dissipation_nonnegative⟩

def recurrenceCertificate (I : ThinInput) : RecurrenceCertificate I :=
  ⟨I.flow_zero, I.flow_add⟩

def witnessMultiplicityCertificate (I : ThinInput) :
    WitnessMultiplicityCertificate I :=
  ⟨I.realizerNonempty⟩

def tameCertificate (I : ThinInput) : TameCertificate I :=
  ⟨I.ambientTame_holds⟩

def partialCompactnessCertificate (I : ThinInput) :
    PartialCompactnessCertificate I :=
  ⟨I.ambientConnected_holds⟩

def capacityCertificate (I : ThinInput) : CapacityCertificate I :=
  ⟨I.ambientContractible_holds⟩

def linearStabilityCertificate (I : ThinInput) :
    LinearStabilityCertificate I :=
  ⟨I.potential_quadratic⟩

def boundaryEvenCertificate (I : ThinInput) :
    BoundaryEvenCertificate I :=
  ⟨I.boundaryEven_holds⟩

def boundaryPolarizedCertificate (I : ThinInput) :
    BoundaryPolarizedCertificate I :=
  ⟨I.boundaryPolarized_holds⟩

def boundedPartialCertificate (I : ThinInput) :
    BoundedPartialCertificate I :=
  ⟨I.boundedPartialWitness_holds⟩

def libraryCertificate (I : ThinInput) :
    LibraryCertificate I :=
  ⟨I.libraryComplete_holds⟩

def reductionCertificate (I : ThinInput) :
    ReductionCertificate I :=
  ⟨I.reductionInitialized_holds, libraryCertificate I⟩

def germCertificate (I : ThinInput) : GermCertificate I :=
  ⟨I.germBounded_holds, reductionCertificate I⟩

def gammaCertificate (I : ThinInput) : GammaCertificate I :=
  ⟨I.gammaExact_holds, I.gammaFaithful_holds, I.gammaTensorPreserving_holds⟩

def admissibilityCertificate (I : ThinInput) : AdmissibilityCertificate I :=
  ⟨ defectEnergyCertificate I,
    recurrenceCertificate I,
    witnessMultiplicityCertificate I,
    tameCertificate I,
    partialCompactnessCertificate I,
    capacityCertificate I,
    linearStabilityCertificate I,
    boundaryEvenCertificate I,
    boundaryPolarizedCertificate I,
    boundedPartialCertificate I,
    I.targetAdmissible_holds ⟩

def anchorCertificate (I : ThinInput) : AnchorCertificate I :=
  ⟨ ambientCertificate I,
    representationConservativeCertificate I,
    representationCompleteCertificate I,
    gammaCertificate I,
    germCertificate I ⟩

def seedContext : Context NodeTag :=
  ({ NodeTag.anchor, NodeTag.defect, NodeTag.recurrence, NodeTag.multiplicity, NodeTag.tame,
     NodeTag.partialCompact, NodeTag.capacity, NodeTag.stability,
     NodeTag.boundaryEven, NodeTag.boundaryPolarized, NodeTag.boundedPartial } : Finset NodeTag)

end Hypostructure.Problem
