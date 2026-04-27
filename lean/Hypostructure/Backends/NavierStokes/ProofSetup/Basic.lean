import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic

namespace Hypostructure.Backends.NavierStokes.ProofSetup

structure SpacetimePoint where
  label : String
  deriving DecidableEq, Repr

abbrev TerminalSingularSet := Set SpacetimePoint

structure PressureGaugeQuotient where
  representativeName : String := "unspecified-pressure-gauge"
  moduloTimeFunctions : Prop := True
  localEnergyCompatible : Prop := True
  deriving Repr

structure TemporalCutoffConvention where
  solutionName : String := "ambient-solution"
  admissibleEndpointTruncations : Prop := True
  deriving Repr

structure SuitableWeakSolution where
  name : String
  terminalTime : ℝ := 0
  pressureGauge : PressureGaugeQuotient := {}
  distributionalEquation : Prop := True
  localEnergyInequality : Prop := True
  deriving Repr

structure TerminalSingularPoint where
  point : SpacetimePoint
  terminalTime : ℝ := 0
  deriving Repr

structure ParabolicCylinder where
  center : SpacetimePoint
  radius : ℝ
  deriving Repr

structure CriticalQuantityC where
  value : ℝ
  deriving Repr

structure CriticalQuantityD where
  value : ℝ
  deriving Repr

structure KineticQuantityA where
  value : ℝ
  deriving Repr

structure DissipationQuantityE where
  value : ℝ
  deriving Repr

structure CriticalQuantityBundle where
  C : CriticalQuantityC
  D : CriticalQuantityD
  A : KineticQuantityA := { value := 0 }
  E : DissipationQuantityE := { value := 0 }
  deriving Repr

def CriticalQuantityBundle.totalCD (q : CriticalQuantityBundle) : ℝ :=
  q.C.value + q.D.value

def RegularAt (_ : SuitableWeakSolution) (_ : TerminalSingularPoint) : Prop := True

def SingularAt (_ : SuitableWeakSolution) (_ : TerminalSingularPoint) : Prop := True

structure SingularityWitness where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  singular : SingularAt solution singularPoint := True
  deriving Repr

structure RegularityCertificate where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  regular : RegularAt solution singularPoint := True
  deriving Repr

def VanishingCriticalQuantities
    (_ : SuitableWeakSolution)
    (_ : TerminalSingularPoint) : Prop := True

structure VanishingCriticalQuantitiesWitness where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  vanishing : VanishingCriticalQuantities solution singularPoint := True
  deriving Repr

def LocalPointwiseTypeIEnvelope
    (_ : SuitableWeakSolution)
    (_ : TerminalSingularPoint) : Prop := True

structure LocalTypeIEnvelopeWitness where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  pointwiseEnvelope : LocalPointwiseTypeIEnvelope solution singularPoint := True
  localBoundData : Prop := True
  deriving Repr

structure FailedLocalTypeIEnvelopeWitness where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  failure : Prop := True
  deriving Repr

def VelocityNoEscape
    (_ : SuitableWeakSolution)
    (_ : TerminalSingularPoint) : Prop := True

structure VelocityNoEscapeCertificate where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  noEscape : VelocityNoEscape solution singularPoint := True
  retainedOffTerminalSlice : Prop := True
  deriving Repr

structure FiniteEnergyWitness where
  solution : SuitableWeakSolution
  finiteEnergy : Prop := True
  deriving Repr

structure GlobalTypeIEnvelopeWitness where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  boundConstant : ℝ := 0
  envelope : Prop := True
  deriving Repr

structure TerminalCompactnessWitness where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  compactness : Prop := True
  deriving Repr

structure LocalEnergyTransferData where
  sourceName : String := "local-energy-transfer"
  transferred : Prop := True
  deriving Repr

structure PressureTransferData where
  gauge : PressureGaugeQuotient := {}
  transferred : Prop := True
  deriving Repr

structure ScaleWindowData where
  selectedRadii : Nat → ℝ := fun _ => 0
  shrinking : Prop := True
  windowControl : Prop := True
  deriving Repr

structure PressureAtlasData where
  gauge : PressureGaugeQuotient := {}
  compatibility : Prop := True
  deriving Repr

structure RescaledSolution where
  source : SuitableWeakSolution
  cylinder : ParabolicCylinder
  suitable : Prop := True
  scalingCompatible : Prop := True
  deriving Repr

structure CompactL3Witness where
  radius : ℝ := 0
  timeInterval : ℝ × ℝ := (0, 0)
  lowerBound : ℝ := 0
  witness : Prop := True
  deriving Repr

structure InvariantLocalNonvanishingData where
  radius : ℝ := 0
  lowerBound : ℝ := 0
  timeTranslationInvariant : Prop := True
  deriving Repr

structure ResidualAncestryData where
  realizedByOriginalBranch : Prop := True
  compactMassRetained : Prop := True
  pressureGaugeCompatible : Prop := True
  deriving Repr

structure CenteredVariables where
  logarithmicTime : Prop := True
  centeredCoordinates : Prop := True
  deriving Repr

structure PositiveConcentrationSequence where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  radii : Nat → ℝ
  criticalValues : Nat → CriticalQuantityBundle := fun _ =>
    { C := { value := 0 }, D := { value := 0 } }
  radiiShrink : Prop := True
  etaCD : ℝ
  etaV : ℝ
  criticalMassLowerBound : Prop := True
  velocityMassLowerBound : Prop := True
  deriving Repr

structure FiniteEnergyTypeIEnvelope where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  finiteEnergy : FiniteEnergyWitness
  typeIEnvelope : GlobalTypeIEnvelopeWitness
  preferredNoEscape : Option VelocityNoEscapeCertificate := none
  deriving Repr

structure AdmissibleLocalTypeISequence where
  concentration : PositiveConcentrationSequence
  localEnvelope : LocalTypeIEnvelopeWitness
  terminalCompactness : TerminalCompactnessWitness
  noEscape : VelocityNoEscapeCertificate
  admissibility : Prop := True
  compatibility : Prop := True
  deriving Repr

structure TypeIIExportData where
  concentration : PositiveConcentrationSequence
  failedLocalTypeIEnvelope : FailedLocalTypeIEnvelopeWitness
  localEnergy : LocalEnergyTransferData
  pressure : PressureTransferData
  scaleWindows : ScaleWindowData
  compatibility : Prop := True
  deriving Repr

structure BlowupSequence where
  entry : AdmissibleLocalTypeISequence
  rescaledSolutions : Nat → RescaledSolution
  localBounds : Prop := True
  pressureAtlas : PressureAtlasData
  deriving Repr

structure AncientSuitableTypeILimit where
  source : BlowupSequence
  pressureGauge : PressureGaugeQuotient := {}
  suitable : Prop := True
  typeIBound : Prop := True
  nonzero : Prop := True
  deriving Repr

structure CenteredAncientProfile where
  source : AncientSuitableTypeILimit
  variables : CenteredVariables := {}
  pressureGauge : PressureGaugeQuotient := {}
  centeredEquation : Prop := True
  smooth : Prop := True
  classicalTypeIBound : Prop := True
  deriving Repr

structure NormalizedSereginLimit where
  profile : CenteredAncientProfile
  normalized : Prop := True
  compactL3Witness : CompactL3Witness
  deriving Repr

structure InvariantLocalNonvanishing where
  limit : NormalizedSereginLimit
  data : InvariantLocalNonvanishingData
  deriving Repr

inductive GeneratedClassTag where
  | small
  | stationary
  | tight
  | structured
  | remaining
  deriving DecidableEq, Repr

structure GeneratedSereginState where
  limit : NormalizedSereginLimit
  nonvanishing : InvariantLocalNonvanishing
  tag : GeneratedClassTag := .remaining
  mild : Prop := True
  bounded : Prop := True
  centered : Prop := True
  deriving Repr

structure RawGeneratedSereginStateSpace where
  root : NormalizedSereginLimit
  states : List GeneratedSereginState
  closedUnderTimeTranslations : Prop := True
  closedUnderLocalSmoothLimits : Prop := True
  deriving Repr

abbrev GeneratedSereginClass := RawGeneratedSereginStateSpace

structure MildStratum where
  carrier : RawGeneratedSereginStateSpace
  states : List GeneratedSereginState
  projectedMildFormulation : Prop := True
  deriving Repr

def SmallClassPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .small

def StationaryClassPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .stationary

def TightClassPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .tight

def StructureDecayClassPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .structured

abbrev StructureDecayPredicate := StructureDecayClassPredicate

def RemainingClassPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .remaining

def MildStratumPredicate (state : GeneratedSereginState) : Prop :=
  state.mild

def AxisymmetricNoSwirlPredicate (_ : GeneratedSereginState) : Prop := True

def PointwiseScaleInvariantPredicate (_ : GeneratedSereginState) : Prop := True

def FiniteSwirlPredicate (_ : GeneratedSereginState) : Prop := True

def PeriodicSwirlPredicate (_ : GeneratedSereginState) : Prop := True

def WeightedVorticityPredicate (_ : GeneratedSereginState) : Prop := True

structure ResidualExportData where
  generatedClass : GeneratedSereginClass
  chosenState : GeneratedSereginState
  inRemaining : RemainingClassPredicate chosenState
  ancestry : ResidualAncestryData
  pressureCompatibility : PressureTransferData
  compactMassCarryover : Prop := True
  deriving Repr

end Hypostructure.Backends.NavierStokes.ProofSetup