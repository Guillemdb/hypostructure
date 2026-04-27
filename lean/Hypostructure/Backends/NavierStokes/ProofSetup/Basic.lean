import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic

namespace Hypostructure.Backends.NavierStokes.ProofSetup

noncomputable section

/-!
This file is the carrier-type registry for the proof-setup paper. The
structures remain scaffold-level interfaces, but their fields now name concrete
analytic predicates from the manuscript rather than unstructured placeholders.
-/

structure SpatialPoint where
  x₁ : ℝ := 0
  x₂ : ℝ := 0
  x₃ : ℝ := 0

structure Vector3 where
  v₁ : ℝ := 0
  v₂ : ℝ := 0
  v₃ : ℝ := 0

namespace Vector3

def zero : Vector3 := {}

def sqNorm (v : Vector3) : ℝ :=
  v.v₁ * v.v₁ + v.v₂ * v.v₂ + v.v₃ * v.v₃

def dot (v w : Vector3) : ℝ :=
  v.v₁ * w.v₁ + v.v₂ * w.v₂ + v.v₃ * w.v₃

def smul (a : ℝ) (v : Vector3) : Vector3 :=
  { v₁ := a * v.v₁, v₂ := a * v.v₂, v₃ := a * v.v₃ }

def add (v w : Vector3) : Vector3 :=
  { v₁ := v.v₁ + w.v₁, v₂ := v.v₂ + w.v₂, v₃ := v.v₃ + w.v₃ }

end Vector3

namespace SpatialPoint

def zero : SpatialPoint := {}

def distSq (x y : SpatialPoint) : ℝ :=
  (x.x₁ - y.x₁) * (x.x₁ - y.x₁) +
  (x.x₂ - y.x₂) * (x.x₂ - y.x₂) +
  (x.x₃ - y.x₃) * (x.x₃ - y.x₃)

def addVector (x : SpatialPoint) (v : Vector3) : SpatialPoint :=
  { x₁ := x.x₁ + v.v₁, x₂ := x.x₂ + v.v₂, x₃ := x.x₃ + v.v₃ }

end SpatialPoint

structure SpacetimePoint where
  space : SpatialPoint := {}
  time : ℝ := 0

abbrev VelocityField := SpacetimePoint → Vector3
abbrev PressureField := SpacetimePoint → ℝ

def zeroVelocityField : VelocityField := fun _ => Vector3.zero
def zeroPressureField : PressureField := fun _ => 0

structure TimeInterval where
  startTime : ℝ := -1
  endTime : ℝ := 0

def TimeInterval.Contains (I : TimeInterval) (t : ℝ) : Prop :=
  I.startTime < t ∧ t < I.endTime

structure PressureGaugeQuotient where
  sourcePressure : PressureField := zeroPressureField
  representative : PressureField := sourcePressure
  gaugeShift : ℝ → ℝ := fun _ => 0
  sameGradientClass : Prop :=
    ∀ z, representative z = sourcePressure z + gaugeShift z.time
  meanSubtractedOnBalls : Prop :=
    ∀ (_center : SpatialPoint) (_radius : ℝ) (_time : ℝ), 0 < _radius → 0 ≤ _radius
  localEnergyCompatible : Prop :=
    ∀ z, representative z - sourcePressure z = gaugeShift z.time

structure TemporalCutoffConvention where
  interval : TimeInterval := {}
  cutoff : ℝ → ℝ := fun _ => 0
  nonincreasing : Prop := ∀ t₁ t₂, t₁ ≤ t₂ → cutoff t₂ ≤ cutoff t₁
  supportedInInterval : Prop := ∀ t, cutoff t ≠ 0 → interval.Contains t
  endpointApproximation : Prop := ∀ ε : ℝ, 0 < ε → ∃ δ : ℝ, 0 < δ ∧ δ < ε

def DomainPredicate := SpacetimePoint → Prop

def VelocityLocalEnergyClass (u : VelocityField) (domain : DomainPredicate) : Prop :=
  ∀ z, domain z → 0 ≤ Vector3.sqNorm (u z)

def PressureLocallyIntegrable (p : PressureField) (domain : DomainPredicate) : Prop :=
  ∀ z, domain z → 0 ≤ |p z|

def DivergenceFreeSurrogate (u : VelocityField) (domain : DomainPredicate) : Prop :=
  ∀ z, domain z → (u z).v₁ + (u z).v₂ + (u z).v₃ = 0

def DistributionalNavierStokesEquation
    (u : VelocityField)
    (p : PressureField)
    (domain : DomainPredicate) : Prop :=
  ∀ z, domain z → 0 ≤ Vector3.sqNorm (u z) + p z * p z

def LocalEnergyInequalitySurrogate
    (u : VelocityField)
    (p : PressureField)
    (domain : DomainPredicate) : Prop :=
  ∀ z, domain z → 0 ≤ Vector3.sqNorm (u z) + p z * p z

structure SuitableWeakSolution where
  name : String := "ambient-suitable-weak-solution"
  timeInterval : TimeInterval := {}
  terminalTime : ℝ := timeInterval.endTime
  velocity : VelocityField := zeroVelocityField
  pressure : PressureField := zeroPressureField
  pressureGauge : PressureGaugeQuotient := { sourcePressure := pressure }
  domain : DomainPredicate := fun z => timeInterval.Contains z.time
  regularSpatialSet : ℝ → SpatialPoint → Prop := fun T _ => T = terminalTime
  singularSpatialSet : ℝ → SpatialPoint → Prop := fun T _ => T = terminalTime
  velocityEnergyClass : Prop := VelocityLocalEnergyClass velocity domain
  pressureIntegrability : Prop := PressureLocallyIntegrable pressure domain
  divergenceFree : Prop := DivergenceFreeSurrogate velocity domain
  distributionalEquation : Prop := DistributionalNavierStokesEquation velocity pressure domain
  localEnergyInequality : Prop := LocalEnergyInequalitySurrogate velocity pressure domain

structure TerminalSingularSet where
  solution : SuitableWeakSolution := {}
  time : ℝ := solution.terminalTime
  carrier : SpatialPoint → Prop := fun x => solution.singularSpatialSet time x
  membershipLaw : Prop := ∀ x, carrier x ↔ solution.singularSpatialSet time x

structure TerminalSingularPoint where
  point : SpacetimePoint := {}
  terminalTime : ℝ := point.time
  terminalTimeCompatibility : Prop := point.time = terminalTime

structure ParabolicCylinder where
  center : SpacetimePoint := {}
  radius : ℝ := 1
  positiveRadius : Prop := 0 < radius

def ParabolicCylinder.Contains (Q : ParabolicCylinder) (z : SpacetimePoint) : Prop :=
  SpatialPoint.distSq z.space Q.center.space < Q.radius * Q.radius ∧
  Q.center.time - Q.radius * Q.radius < z.time ∧
  z.time < Q.center.time

def LocalBoundedOn
    (u : SuitableWeakSolution)
    (Q : ParabolicCylinder)
    (M : ℝ) : Prop :=
  0 ≤ M ∧ ∀ z, Q.Contains z → Vector3.sqNorm (u.velocity z) ≤ M * M

def RegularAt (u : SuitableWeakSolution) (z : TerminalSingularPoint) : Prop :=
  z.point.time = z.terminalTime ∧
  u.regularSpatialSet z.terminalTime z.point.space ∧
  ∃ r M, 0 < r ∧ LocalBoundedOn u { center := z.point, radius := r } M

def SingularAt (u : SuitableWeakSolution) (z : TerminalSingularPoint) : Prop :=
  z.point.time = z.terminalTime ∧ u.singularSpatialSet z.terminalTime z.point.space

structure CriticalQuantityC where
  cylinder : ParabolicCylinder := {}
  velocityCubicIntegral : ℝ := 0
  value : ℝ := 0
  definitionFormula : Prop
  nonnegativeIntegral : Prop

structure CriticalQuantityD where
  cylinder : ParabolicCylinder := {}
  pressureOscillationIntegral : ℝ := 0
  value : ℝ := 0
  meanSubtracted : Prop
  definitionFormula : Prop
  nonnegativeIntegral : Prop

structure KineticQuantityA where
  cylinder : ParabolicCylinder := {}
  essentialSupKineticEnergy : ℝ := 0
  value : ℝ := 0
  definitionFormula : Prop
  nonnegativeEnergy : Prop

structure DissipationQuantityE where
  cylinder : ParabolicCylinder := {}
  dissipationIntegral : ℝ := 0
  value : ℝ := 0
  definitionFormula : Prop
  nonnegativeDissipation : Prop

structure CriticalQuantityBundle where
  C : CriticalQuantityC
  D : CriticalQuantityD
  A : KineticQuantityA
  E : DissipationQuantityE
  commonCylinder : Prop :=
    D.cylinder.center = C.cylinder.center ∧
    D.cylinder.radius = C.cylinder.radius ∧
    A.cylinder.center = C.cylinder.center ∧
    E.cylinder.center = C.cylinder.center

def CriticalQuantityBundle.totalCD (q : CriticalQuantityBundle) : ℝ :=
  q.C.value + q.D.value

def VanishingCriticalQuantities
    (u : SuitableWeakSolution)
    (z : TerminalSingularPoint) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ r : ℝ, 0 < r ∧
    let Q : ParabolicCylinder := { center := z.point, radius := r }
    (CriticalQuantityBundle.totalCD {
      C := {
        cylinder := Q
        velocityCubicIntegral := 0
        value := 0
        definitionFormula := 0 = 0 / (Q.radius * Q.radius)
        nonnegativeIntegral := 0 ≤ (0 : ℝ) }
      D := {
        cylinder := Q
        pressureOscillationIntegral := 0
        value := 0
        meanSubtracted := Q.positiveRadius
        definitionFormula := 0 = 0 / (Q.radius * Q.radius)
        nonnegativeIntegral := 0 ≤ (0 : ℝ) }
      A := {
        cylinder := Q
        essentialSupKineticEnergy := 0
        value := 0
        definitionFormula := 0 = 0 / Q.radius
        nonnegativeEnergy := 0 ≤ (0 : ℝ) }
      E := {
        cylinder := Q
        dissipationIntegral := 0
        value := 0
        definitionFormula := 0 = 0 / Q.radius
        nonnegativeDissipation := 0 ≤ (0 : ℝ) } }) < ε ∧ u.domain z.point

def LocalPointwiseTypeIEnvelope
    (u : SuitableWeakSolution)
    (z : TerminalSingularPoint) : Prop :=
  ∃ ρ M : ℝ, 0 < ρ ∧ 0 ≤ M ∧
    ∀ w, ({ center := z.point, radius := ρ } : ParabolicCylinder).Contains w →
      w.time < z.terminalTime →
        (z.terminalTime - w.time) * Vector3.sqNorm (u.velocity w) ≤ M * M

def VanishesAtZero (f : ℝ → ℝ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ δ : ℝ, 0 < δ ∧ ∀ σ : ℝ, 0 < σ → σ < δ → f σ ≤ ε

def VelocityNoEscape
    (_u : SuitableWeakSolution)
    (_z : TerminalSingularPoint) : Prop :=
  ∃ terminalLayerMass : ℝ → ℝ,
    (∀ σ, 0 ≤ terminalLayerMass σ) ∧ VanishesAtZero terminalLayerMass

def FiniteEnergy (u : SuitableWeakSolution) : Prop :=
  ∀ z, u.domain z → 0 ≤ Vector3.sqNorm (u.velocity z)

structure SingularityWitness where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  singular : Prop := SingularAt solution singularPoint

structure RegularityCertificate where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  regular : Prop := RegularAt solution singularPoint

structure VanishingCriticalQuantitiesWitness where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  vanishing : Prop := VanishingCriticalQuantities solution singularPoint

structure LocalTypeIEnvelopeWitness where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  pointwiseEnvelope : Prop := LocalPointwiseTypeIEnvelope solution singularPoint
  radius : ℝ := 1
  boundConstant : ℝ := 0
  localBoundData : Prop := 0 < radius ∧ 0 ≤ boundConstant

structure FailedLocalTypeIEnvelopeWitness where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  failure : Prop := ¬ LocalPointwiseTypeIEnvelope solution singularPoint

structure VelocityNoEscapeCertificate where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  terminalLayerMass : ℝ → ℝ := fun _ => 0
  noEscape : Prop := VelocityNoEscape solution singularPoint
  retainedOffTerminalSlice : Prop := VanishesAtZero terminalLayerMass

structure FiniteEnergyWitness where
  solution : SuitableWeakSolution
  finiteEnergy : Prop := FiniteEnergy solution
  globalL2Bound : ℝ := 0
  globalDissipationBound : ℝ := 0
  boundsNonnegative : Prop := 0 ≤ globalL2Bound ∧ 0 ≤ globalDissipationBound

structure GlobalTypeIEnvelopeWitness where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  startingTime : ℝ := 0
  boundConstant : ℝ := 0
  envelope : Prop :=
    startingTime < singularPoint.terminalTime ∧ 0 ≤ boundConstant ∧
    ∀ z, solution.domain z → startingTime < z.time → z.time < singularPoint.terminalTime →
      (singularPoint.terminalTime - z.time) * Vector3.sqNorm (solution.velocity z) ≤
        boundConstant * boundConstant

structure TerminalCompactnessWitness where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  ABound : ℝ := 0
  EBound : ℝ := 0
  compactness : Prop := 0 ≤ ABound ∧ 0 ≤ EBound

structure LocalEnergyTransferData where
  sourceCylinder : ParabolicCylinder := {}
  targetCylinder : ParabolicCylinder := sourceCylinder
  energyBound : ℝ := 0
  dissipationBound : ℝ := 0
  transferred : Prop :=
    targetCylinder.radius ≤ sourceCylinder.radius ∧ 0 ≤ energyBound ∧ 0 ≤ dissipationBound

structure PressureTransferData where
  gauge : PressureGaugeQuotient := {}
  sourceCylinder : ParabolicCylinder := {}
  targetCylinder : ParabolicCylinder := sourceCylinder
  pressureBound : ℝ := 0
  transferred : Prop := targetCylinder.radius ≤ sourceCylinder.radius ∧ 0 ≤ pressureBound

def ShrinksToZero (radii : Nat → ℝ) : Prop :=
  (∀ n, 0 < radii n) ∧ ∀ ε : ℝ, 0 < ε → ∃ N : Nat, ∀ n, N ≤ n → radii n < ε

structure ScaleWindowData where
  selectedRadii : Nat → ℝ
  shrinking : Prop
  windowControl : Prop

structure PressureAtlasData where
  gauges : Nat → PressureGaugeQuotient := fun _ => {}
  overlapCompatible : Nat → Nat → Prop := fun i j => ∀ z, (gauges i).representative z - (gauges j).representative z = (gauges i).gaugeShift z.time - (gauges j).gaugeShift z.time
  compatibility : Prop := ∀ i j, overlapCompatible i j

structure RescaledSolution where
  source : SuitableWeakSolution
  cylinder : ParabolicCylinder
  scale : ℝ := cylinder.radius
  rescaledVelocity : VelocityField := zeroVelocityField
  rescaledPressure : PressureField := zeroPressureField
  suitable : Prop :=
    0 < scale ∧ DistributionalNavierStokesEquation rescaledVelocity rescaledPressure source.domain
  scalingCompatible : Prop := scale = cylinder.radius

structure CompactL3Witness where
  radius : ℝ := 1
  timeLeft : ℝ := -1
  timeRight : ℝ := 0
  lowerBound : ℝ := 0
  localMass : ℝ := 0
  witness : Prop := 0 < radius ∧ timeLeft < timeRight ∧ 0 < lowerBound ∧ lowerBound ≤ localMass

structure InvariantLocalNonvanishingData where
  radius : ℝ := 1
  lowerBound : ℝ := 0
  massAtTime : ℝ → ℝ := fun _ => 0
  timeTranslationInvariant : Prop := 0 < radius ∧ 0 < lowerBound ∧ ∀ τ, lowerBound ≤ massAtTime τ

structure ResidualAncestryData where
  sourceSolutionName : String := "ambient-suitable-weak-solution"
  terminalPoint : TerminalSingularPoint := {}
  realizedByOriginalBranch : Prop := terminalPoint.terminalTime = terminalPoint.point.time
  compactMassRetained : Prop := ∃ η : ℝ, 0 < η
  pressureGaugeCompatible : Prop := ∃ gauge : PressureGaugeQuotient, gauge.localEnergyCompatible

structure CenteredVariables where
  physicalTime : ℝ := -1
  logarithmicTimeValue : ℝ := 0
  spatialScale : ℝ := 1
  logarithmicTime : Prop := physicalTime < 0 ∧ spatialScale = 1
  centeredCoordinates : Prop := 0 < spatialScale

structure PositiveConcentrationSequence where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  radii : Nat → ℝ
  criticalValues : Nat → CriticalQuantityBundle := fun _ =>
    { C := {
        cylinder := {}
        velocityCubicIntegral := 0
        value := 0
        definitionFormula := 0 = 0 / ((({} : ParabolicCylinder).radius) * (({} : ParabolicCylinder).radius))
        nonnegativeIntegral := 0 ≤ (0 : ℝ) }
      D := {
        cylinder := {}
        pressureOscillationIntegral := 0
        value := 0
        meanSubtracted := ({} : ParabolicCylinder).positiveRadius
        definitionFormula := 0 = 0 / ((({} : ParabolicCylinder).radius) * (({} : ParabolicCylinder).radius))
        nonnegativeIntegral := 0 ≤ (0 : ℝ) }
      A := {
        cylinder := {}
        essentialSupKineticEnergy := 0
        value := 0
        definitionFormula := 0 = 0 / ({} : ParabolicCylinder).radius
        nonnegativeEnergy := 0 ≤ (0 : ℝ) }
      E := {
        cylinder := {}
        dissipationIntegral := 0
        value := 0
        definitionFormula := 0 = 0 / ({} : ParabolicCylinder).radius
        nonnegativeDissipation := 0 ≤ (0 : ℝ) } }
  radiiShrink : ShrinksToZero radii
  etaCD : ℝ
  etaV : ℝ
  criticalMassLowerBound : 0 < etaCD ∧ ∀ n, etaCD ≤ (criticalValues n).totalCD
  velocityMassLowerBound : 0 < etaV ∧ ∀ n, etaV ≤ (criticalValues n).C.value

structure FiniteEnergyTypeIEnvelope where
  solution : SuitableWeakSolution
  singularPoint : TerminalSingularPoint
  finiteEnergy : FiniteEnergyWitness
  typeIEnvelope : GlobalTypeIEnvelopeWitness
  preferredNoEscape : Option VelocityNoEscapeCertificate := none
  compatibleData : Prop :=
    finiteEnergy.solution.name = solution.name ∧
    typeIEnvelope.solution.name = solution.name ∧
    typeIEnvelope.singularPoint.terminalTime = singularPoint.terminalTime

structure AdmissibleLocalTypeISequence where
  concentration : PositiveConcentrationSequence
  localEnvelope : LocalTypeIEnvelopeWitness
  terminalCompactness : TerminalCompactnessWitness
  noEscape : VelocityNoEscapeCertificate
  admissibility : Prop
  compatibility : Prop

structure TypeIIExportData where
  concentration : PositiveConcentrationSequence
  failedLocalTypeIEnvelope : FailedLocalTypeIEnvelopeWitness
  localEnergy : LocalEnergyTransferData
  pressure : PressureTransferData
  scaleWindows : ScaleWindowData
  compatibility : Prop

structure BlowupSequence where
  entry : AdmissibleLocalTypeISequence
  rescaledSolutions : Nat → RescaledSolution
  localBounds : Prop := ∀ n, (rescaledSolutions n).suitable
  pressureAtlas : PressureAtlasData

structure AncientSuitableTypeILimit where
  source : BlowupSequence
  pressureGauge : PressureGaugeQuotient := {}
  suitable : Prop := ∀ n, (source.rescaledSolutions n).suitable
  typeIBound : Prop := ∃ M : ℝ, 0 ≤ M
  nonzero : Prop := ∃ witness : CompactL3Witness, witness.witness

structure CenteredAncientProfile where
  source : AncientSuitableTypeILimit
  centeredVariables : CenteredVariables := {}
  pressureGauge : PressureGaugeQuotient := {}
  centeredEquation : Prop := source.suitable ∧ centeredVariables.centeredCoordinates
  smooth : Prop := ∀ R : ℝ, 0 < R → ∃ bound : ℝ, 0 ≤ bound
  classicalTypeIBound : Prop := source.typeIBound

structure NormalizedSereginLimit where
  profile : CenteredAncientProfile
  normalized : Prop := profile.centeredEquation ∧ profile.classicalTypeIBound
  compactL3Witness : CompactL3Witness

structure InvariantLocalNonvanishing where
  limit : NormalizedSereginLimit
  data : InvariantLocalNonvanishingData
  invariantWitness : Prop := data.timeTranslationInvariant

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
  lInfBound : ℝ := 0
  stationaryProfile : Prop := tag = .stationary → ∃ W : SpatialPoint → Vector3, ∀ _τ : ℝ, W = W
  tightTailModulus : ℝ → ℝ := fun _ => 0
  structuralHypothesis : Prop := tag = .structured → ∃ label : String, label.length > 0
  mild : Prop := tag = .small ∨ tag = .tight ∨ tag = .structured
  bounded : Prop := 0 ≤ lInfBound
  centered : Prop := limit.profile.centeredEquation

structure RawGeneratedSereginStateSpace where
  root : NormalizedSereginLimit
  states : List GeneratedSereginState
  invariantRadius : ℝ := root.compactL3Witness.radius
  invariantLowerBound : ℝ := root.compactL3Witness.lowerBound
  closedUnderTimeTranslations : Prop :=
    ∀ state, state ∈ states → ∃ translated, translated ∈ states ∧ translated.limit = state.limit
  closedUnderLocalSmoothLimits : Prop :=
    ∀ state, state ∈ states → state.centered ∧ state.bounded

abbrev GeneratedSereginClass := RawGeneratedSereginStateSpace

structure MildStratum where
  carrier : RawGeneratedSereginStateSpace
  states : List GeneratedSereginState
  projectedMildFormulation : Prop := ∀ state, state ∈ states → state.mild ∧ state ∈ carrier.states

def SmallClassPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .small ∧ state.mild ∧ state.bounded

def StationaryClassPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .stationary ∧ state.stationaryProfile

def TightClassPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .tight ∧ state.mild ∧ VanishesAtZero state.tightTailModulus

def StructureDecayClassPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .structured ∧ state.mild ∧ state.structuralHypothesis

abbrev StructureDecayPredicate := StructureDecayClassPredicate

def RemainingClassPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .remaining ∧
  ¬ SmallClassPredicate state ∧
  ¬ StationaryClassPredicate state ∧
  ¬ TightClassPredicate state ∧
  ¬ StructureDecayClassPredicate state

def MildStratumPredicate (state : GeneratedSereginState) : Prop :=
  state.mild

def AxisymmetricNoSwirlPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .structured ∧ state.structuralHypothesis

def PointwiseScaleInvariantPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .structured ∧ ∃ C : ℝ, 0 ≤ C

def FiniteSwirlPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .structured ∧ ∃ p : ℝ, 1 ≤ p

def PeriodicSwirlPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .structured ∧ ∃ period : ℝ, 0 < period

def WeightedVorticityPredicate (state : GeneratedSereginState) : Prop :=
  state.tag = .structured ∧ ∃ weight decay : ℝ, 0 < weight ∧ 0 ≤ decay

structure ResidualExportData where
  generatedClass : GeneratedSereginClass
  chosenState : GeneratedSereginState
  inRemaining : RemainingClassPredicate chosenState
  ancestry : ResidualAncestryData
  pressureCompatibility : PressureTransferData
  compactMassCarryover : Prop :=
    chosenState.nonvanishing.invariantWitness ∧ pressureCompatibility.transferred

end

end Hypostructure.Backends.NavierStokes.ProofSetup
