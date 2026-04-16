import HypoHodge.Core.ProofComplete

namespace HypoHodge.Algebraic
open HypoHodge.Core

structure VerifiedHodgeThinInput where
  V                : Type
  instAddComm      : AddCommGroup V
  instModuleR      : Module ℝ V
  instNormedGroup  : NormedAddCommGroup V
  instNormedSpace  : NormedSpace ℝ V
  instFiniteDim    : FiniteDimensional ℝ V

  varietyType      : Type
  cohomologyType   : Type
  cycleType        : Type

  p                : ℕ
  Qrank            : ℕ
  hodgeSubset      : Set V
  symmetry         : Type

  potential        : V → ℝ
  dissipation      : V → ℝ
  flow             : ℝ → V → V

  flow_id          : ∀ t x, flow t x = x
  dissipation_zero : ∀ x, dissipation x = 0
  potential_quad   : ∀ x, potential x = ‖x‖ ^ (2 : ℕ)

  hodgeClass       : cohomologyType
  cycleClass       : cycleType → cohomologyType
  rationalClass    : cohomologyType → Prop
  hodgePredicate   : cohomologyType → Prop

  connected        : Prop
  connected_holds  : connected
  contractible     : Prop
  contractible_holds : contractible
  tame             : Prop
  tame_holds       : tame
  varietySmooth    : Prop
  varietySmooth_holds : varietySmooth
  varietyProjective : Prop
  varietyProjective_holds : varietyProjective
  varietyOverComplex : Prop
  varietyOverComplex_holds : varietyOverComplex
  representationConservative : Prop
  representationConservative_holds : representationConservative
  representationComplete : Prop
  representationComplete_holds : representationComplete
  targetAdmissible : Prop
  targetAdmissible_holds : targetAdmissible
  germBounded : Prop
  germBounded_holds : germBounded
  initialReduction : Prop
  initialReduction_holds : initialReduction
  categoryLibraryComplete : Prop
  categoryLibraryComplete_holds : categoryLibraryComplete
  gammaExact : Prop
  gammaExact_holds : gammaExact
  gammaFaithful : Prop
  gammaFaithful_holds : gammaFaithful
  gammaTensorPres : Prop
  gammaTensorPres_holds : gammaTensorPres
  targetWeightEven : Prop
  targetWeightEven_holds : targetWeightEven
  targetPolarizable : Prop
  targetPolarizable_holds : targetPolarizable
  boundedPartialWitness : Prop
  boundedPartialWitness_holds : boundedPartialWitness
  cycleCarrierNonempty : Nonempty cycleType

attribute [instance] VerifiedHodgeThinInput.instAddComm
attribute [instance] VerifiedHodgeThinInput.instModuleR
attribute [instance] VerifiedHodgeThinInput.instNormedGroup
attribute [instance] VerifiedHodgeThinInput.instNormedSpace
attribute [instance] VerifiedHodgeThinInput.instFiniteDim

def TargetHodgeClass (I : VerifiedHodgeThinInput) : Prop :=
  I.rationalClass I.hodgeClass ∧ I.hodgePredicate I.hodgeClass

def TargetRealizedByCycle (I : VerifiedHodgeThinInput) : Prop :=
  ∃ Z : I.cycleType, I.cycleClass Z = I.hodgeClass

def HodgeConjectureTarget (I : VerifiedHodgeThinInput) : Prop :=
  TargetHodgeClass I → TargetRealizedByCycle I

structure AmbientCertificate (I : VerifiedHodgeThinInput) : Prop where
  smooth : I.varietySmooth
  projective : I.varietyProjective
  overComplex : I.varietyOverComplex
  connected : I.connected

structure RepresentationConservativeCertificate (I : VerifiedHodgeThinInput) : Prop where
  conservative : I.representationConservative

structure RepresentationCompleteCertificate (I : VerifiedHodgeThinInput) : Prop where
  complete : I.representationComplete

structure DefectEnergyCertificate (I : VerifiedHodgeThinInput) : Prop where
  zeroDissipation : ∀ x, I.dissipation x = 0

structure RecurrenceCertificate (I : VerifiedHodgeThinInput) : Prop where
  trivialFlow : ∀ t x, I.flow t x = x

structure CycleMultiplicityCertificate (I : VerifiedHodgeThinInput) : Prop where
  nonemptyCycleCarrier : Nonempty I.cycleType

structure TameLambdaCertificate (I : VerifiedHodgeThinInput) : Prop where
  tame : I.tame

structure PartialCompactnessCertificate (I : VerifiedHodgeThinInput) : Prop where
  connected : I.connected

structure CapacityCertificate (I : VerifiedHodgeThinInput) : Prop where
  contractible : I.contractible

structure LinearStabilityCertificate (I : VerifiedHodgeThinInput) : Prop where
  quadraticPotential : ∀ x, I.potential x = ‖x‖ ^ (2 : ℕ)

structure ThinBoundaryPiCertificate (I : VerifiedHodgeThinInput) : Prop where
  weightEven : I.targetWeightEven

structure ThinBoundaryOCertificate (I : VerifiedHodgeThinInput) : Prop where
  polarizable : I.targetPolarizable

structure BoundedPartialCertificate (I : VerifiedHodgeThinInput) : Prop where
  boundedPartial : I.boundedPartialWitness

structure CategoryLibraryCertificate (I : VerifiedHodgeThinInput) : Prop where
  complete : I.categoryLibraryComplete

structure InitialReductionCertificate (I : VerifiedHodgeThinInput) : Prop where
  initialized : I.initialReduction
  library : CategoryLibraryCertificate I

structure GermBoundedCertificate (I : VerifiedHodgeThinInput) : Prop where
  bounded : I.germBounded
  initial : InitialReductionCertificate I

structure GammaPackageCertificate (I : VerifiedHodgeThinInput) : Prop where
  exact : I.gammaExact
  faithful : I.gammaFaithful
  tensorPres : I.gammaTensorPres

structure AdmissibilityCertificate (I : VerifiedHodgeThinInput) : Prop where
  defect : DefectEnergyCertificate I
  recurrence : RecurrenceCertificate I
  cycleMultiplicity : CycleMultiplicityCertificate I
  tameLambda : TameLambdaCertificate I
  partialCompactness : PartialCompactnessCertificate I
  capacity : CapacityCertificate I
  linearStability : LinearStabilityCertificate I
  thinBoundaryPi : ThinBoundaryPiCertificate I
  thinBoundaryO : ThinBoundaryOCertificate I
  boundedPartial : BoundedPartialCertificate I
  admissible : I.targetAdmissible

structure AdjunctionCertificate (I : VerifiedHodgeThinInput) : Prop where
  ambient : AmbientCertificate I
  repCon : RepresentationConservativeCertificate I
  repComp : RepresentationCompleteCertificate I
  gamma : GammaPackageCertificate I
  germ : GermBoundedCertificate I

structure UnresolvedLockObstruction (I : VerifiedHodgeThinInput) : Prop where
  targetHodge : TargetHodgeClass I
  noCycleLift : ¬ TargetRealizedByCycle I

structure PromotionGap (I : VerifiedHodgeThinInput) : Prop where
  blockedByPromotion : ¬ I.gammaTensorPres

def ambientCertificate (I : VerifiedHodgeThinInput) : AmbientCertificate I :=
  ⟨I.varietySmooth_holds, I.varietyProjective_holds, I.varietyOverComplex_holds, I.connected_holds⟩

def representationConservativeCertificate (I : VerifiedHodgeThinInput) :
    RepresentationConservativeCertificate I :=
  ⟨I.representationConservative_holds⟩

def representationCompleteCertificate (I : VerifiedHodgeThinInput) :
    RepresentationCompleteCertificate I :=
  ⟨I.representationComplete_holds⟩

def defectEnergyCertificate (I : VerifiedHodgeThinInput) : DefectEnergyCertificate I :=
  ⟨I.dissipation_zero⟩

def recurrenceCertificate (I : VerifiedHodgeThinInput) : RecurrenceCertificate I :=
  ⟨I.flow_id⟩

def cycleMultiplicityCertificate (I : VerifiedHodgeThinInput) :
    CycleMultiplicityCertificate I :=
  ⟨I.cycleCarrierNonempty⟩

def tameLambdaCertificate (I : VerifiedHodgeThinInput) : TameLambdaCertificate I :=
  ⟨I.tame_holds⟩

def partialCompactnessCertificate (I : VerifiedHodgeThinInput) :
    PartialCompactnessCertificate I :=
  ⟨I.connected_holds⟩

def capacityCertificate (I : VerifiedHodgeThinInput) : CapacityCertificate I :=
  ⟨I.contractible_holds⟩

def linearStabilityCertificate (I : VerifiedHodgeThinInput) :
    LinearStabilityCertificate I :=
  ⟨I.potential_quad⟩

def thinBoundaryPiCertificate (I : VerifiedHodgeThinInput) :
    ThinBoundaryPiCertificate I :=
  ⟨I.targetWeightEven_holds⟩

def thinBoundaryOCertificate (I : VerifiedHodgeThinInput) :
    ThinBoundaryOCertificate I :=
  ⟨I.targetPolarizable_holds⟩

def boundedPartialCertificate (I : VerifiedHodgeThinInput) :
    BoundedPartialCertificate I :=
  ⟨I.boundedPartialWitness_holds⟩

def categoryLibraryCertificate (I : VerifiedHodgeThinInput) :
    CategoryLibraryCertificate I :=
  ⟨I.categoryLibraryComplete_holds⟩

def initialReductionCertificate (I : VerifiedHodgeThinInput) :
    InitialReductionCertificate I :=
  ⟨I.initialReduction_holds, categoryLibraryCertificate I⟩

def germBoundedCertificate (I : VerifiedHodgeThinInput) : GermBoundedCertificate I :=
  ⟨I.germBounded_holds, initialReductionCertificate I⟩

def gammaPackageCertificate (I : VerifiedHodgeThinInput) : GammaPackageCertificate I :=
  ⟨I.gammaExact_holds, I.gammaFaithful_holds, I.gammaTensorPres_holds⟩

def admissibilityCertificate (I : VerifiedHodgeThinInput) : AdmissibilityCertificate I :=
  ⟨ defectEnergyCertificate I,
    recurrenceCertificate I,
    cycleMultiplicityCertificate I,
    tameLambdaCertificate I,
    partialCompactnessCertificate I,
    capacityCertificate I,
    linearStabilityCertificate I,
    thinBoundaryPiCertificate I,
    thinBoundaryOCertificate I,
    boundedPartialCertificate I,
    I.targetAdmissible_holds ⟩

def adjunctionCertificate (I : VerifiedHodgeThinInput) : AdjunctionCertificate I :=
  ⟨ ambientCertificate I,
    representationConservativeCertificate I,
    representationCompleteCertificate I,
    gammaPackageCertificate I,
    germBoundedCertificate I ⟩

def gamma0 (I : VerifiedHodgeThinInput) : Context :=
  ({ CertTag.adj, CertTag.dE, CertTag.recN, CertTag.cMu, CertTag.scLambda,
     CertTag.scPartialC, CertTag.capH, CertTag.lsSigma,
     CertTag.tbPi, CertTag.tbO, CertTag.boundPartial } : Finset CertTag)

end HypoHodge.Algebraic
