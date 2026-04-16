import HypoHodge.Algebraic.VerifiedThinInput

namespace HypoHodge.Imported
open HypoHodge.Algebraic

structure IncreasingFiltration (A : Type _) where
  level : ℤ → Set A
  mono : ∀ {i j : ℤ}, i ≤ j → level i ⊆ level j

structure DecreasingFiltration (A : Type _) where
  level : ℤ → Set A
  mono : ∀ {i j : ℤ}, i ≤ j → level j ⊆ level i

def IncreasingFiltration.Mem {A : Type _} (W : IncreasingFiltration A) (n : ℤ) (x : A) : Prop :=
  x ∈ W.level n

def DecreasingFiltration.Mem {A : Type _} (F : DecreasingFiltration A) (n : ℤ) (x : A) : Prop :=
  x ∈ F.level n

structure PolarizationData (A : Type _) where
  pairing : A → A → ℤ
  bilinear : Prop
  nondegenerate : Prop
  positiveOnPrimitive : Prop

structure ComparisonIsomorphism (A B : Type _) where
  toFun : A → B
  invFun : B → A
  left_inv : Function.LeftInverse invFun toFun
  right_inv : Function.RightInverse invFun toFun

structure HodgeFunctorialityData (M : Type _) (A : Type _) where
  map : M → A → A
  respectsIdentity : Prop
  respectsComposition : Prop
  preservesWeightFiltration : Prop
  preservesHodgeFiltration : Prop
  preservesPolarization : Prop

structure MixedHodgeStructure (A : Type _) where
  weightFiltration : IncreasingFiltration A
  hodgeFiltration : DecreasingFiltration A
  rationalLattice : Set A
  weightExhaustive : Prop
  weightSeparated : Prop
  hodgeExhaustive : Prop
  hodgeSeparated : Prop
  functorial : Prop

structure PureHodgeStructure (A : Type _) extends MixedHodgeStructure A where
  weight : ℤ
  pure : Prop
  polarization : PolarizationData A

def PureHodgeStructure.toMixed {A : Type _} (H : PureHodgeStructure A) : MixedHodgeStructure A :=
  H.toMixedHodgeStructure

structure AcceptedLiteratureTheorems (A : Type _) where
  deligneMixedHodge : Prop
  smoothProjectivePurity : Prop
  bettiDeRhamComparison : Prop
  cycleClassFunctoriality : Prop
  hardLefschetz : Prop
  hodgeRiemannBilinear : Prop
  lefschetz11 : Prop
  semisimplicityPolarizablePurePieces : Prop

structure ClassicalOrigin (I : VerifiedHodgeThinInput) where
  ambient : AmbientCertificate I
  cycleMultiplicity : CycleMultiplicityCertificate I
  classicalTheorems : AcceptedLiteratureTheorems I.cohomologyType

structure HodgeBridgePremises (I : VerifiedHodgeThinInput) where
  classical : ClassicalOrigin I
  ambient : AmbientCertificate I
  thinBoundaryPi : ThinBoundaryPiCertificate I
  tameLambda : TameLambdaCertificate I
  defect : DefectEnergyCertificate I

structure GammaContextPremises (I : VerifiedHodgeThinInput) where
  ambient : AmbientCertificate I
  repCon : RepresentationConservativeCertificate I
  repComp : RepresentationCompleteCertificate I
  catLib : CategoryLibraryCertificate I

structure ProducesMHS (I : VerifiedHodgeThinInput) where
  mixedStructure : MixedHodgeStructure I.cohomologyType
  pureStructure : PureHodgeStructure I.cohomologyType
  functoriality : HodgeFunctorialityData I.symmetry I.cohomologyType
  comparison : ComparisonIsomorphism I.cohomologyType I.cohomologyType
  polarization : PolarizationData I.cohomologyType
  literature : AcceptedLiteratureTheorems I.cohomologyType
  targetRational : I.rationalClass I.hodgeClass
  targetHodge : I.hodgePredicate I.hodgeClass
  targetInWeightFiltration : IncreasingFiltration.Mem mixedStructure.weightFiltration 0 I.hodgeClass
  targetInHodgeFiltration : DecreasingFiltration.Mem mixedStructure.hodgeFiltration 0 I.hodgeClass
  pureRefinesMixed : pureStructure.toMixed = mixedStructure
  comparisonRespectsFiltrations : Prop
  comparisonRespectsRationalStructure : Prop
  cycleClassCompatibleWithComparison : Prop
  polarizable : I.targetPolarizable
  weightEven : I.targetWeightEven

structure ProducesGamma (I : VerifiedHodgeThinInput) where
  exact : I.gammaExact
  faithful : I.gammaFaithful
  tensorPres : I.gammaTensorPres

structure TannakianBridgePremises (I : VerifiedHodgeThinInput) where
  admissibility : AdmissibilityCertificate I
  gamma : ProducesGamma I
  catLib : CategoryLibraryCertificate I

structure ProducesTann (I : VerifiedHodgeThinInput) where
  liftedCycle : I.cycleType
  realizesTarget :
    ProducesMHS I →
    CycleMultiplicityCertificate I →
    LinearStabilityCertificate I →
    ThinBoundaryOCertificate I →
    CategoryLibraryCertificate I →
    InitialReductionCertificate I →
    AdmissibilityCertificate I →
    I.cycleClass liftedCycle = I.hodgeClass

class ImportedHodgeAxioms (I : VerifiedHodgeThinInput) where
  classicalExtraction :
    ClassicalOrigin I
  hodgeBridgeSound :
    HodgeBridgePremises I → ProducesMHS I
  tannakianContextSound :
    GammaContextPremises I → ProducesGamma I
  tannakianBridgeSound :
    TannakianBridgePremises I → ProducesTann I

theorem ProducesMHS.targetHodgeClass
    {I : VerifiedHodgeThinInput}
    (hMHS : ProducesMHS I) :
    TargetHodgeClass I := by
  exact ⟨hMHS.targetRational, hMHS.targetHodge⟩

theorem ProducesMHS.targetInWeightZero
    {I : VerifiedHodgeThinInput}
    (hMHS : ProducesMHS I) :
    I.hodgeClass ∈ hMHS.mixedStructure.weightFiltration.level 0 := by
  exact hMHS.targetInWeightFiltration

theorem ProducesMHS.targetInHodgeZero
    {I : VerifiedHodgeThinInput}
    (hMHS : ProducesMHS I) :
    I.hodgeClass ∈ hMHS.mixedStructure.hodgeFiltration.level 0 := by
  exact hMHS.targetInHodgeFiltration

def ProducesMHS.hasPolarization
    {I : VerifiedHodgeThinInput}
    (hMHS : ProducesMHS I) :
    Prop :=
  hMHS.polarization.nondegenerate

theorem ProducesMHS.hasComparisonIsomorphism
    {I : VerifiedHodgeThinInput}
    (hMHS : ProducesMHS I) :
    Function.LeftInverse hMHS.comparison.invFun hMHS.comparison.toFun ∧
      Function.RightInverse hMHS.comparison.invFun hMHS.comparison.toFun := by
  exact ⟨hMHS.comparison.left_inv, hMHS.comparison.right_inv⟩

def ProducesMHS.importsHardLefschetz
    {I : VerifiedHodgeThinInput}
    (hMHS : ProducesMHS I) :
    Prop :=
  hMHS.literature.hardLefschetz

def ProducesMHS.importsLefschetz11
    {I : VerifiedHodgeThinInput}
    (hMHS : ProducesMHS I) :
    Prop :=
  hMHS.literature.lefschetz11

end HypoHodge.Imported
