import Hypostructure.Sieve.FiniteDag
import Hypostructure.Backends.NavierStokes.ProofSetup.FinalAssembly
import Hypostructure.Backends.NavierStokes.ProofSetup.TypeIIInterface
import Hypostructure.Backends.NavierStokes.ResidualBranch.FinalClosure

namespace Hypostructure.Backends.NavierStokes.ProofSetup

set_option maxRecDepth 20000
set_option maxHeartbeats 1000000

open Hypostructure.Core
open Hypostructure.Sieve

/-!
A reusable finite-DAG view of the proof-setup architecture.

The human-readable `DependencyStep` metadata is converted into `Rule`s from the
core/sieve framework, and the ranked list of steps supplies the acyclicity proof
needed to package the architecture as a `FiniteSieve`.
-/

inductive ProofSetupObjectId where
  | spatialPoint
  | velocityVector
  | spacetimePoint
  | timeInterval
  | velocityField
  | pressureField
  | domainPredicate
  | suitableWeakSolution
  | terminalSingularPoint
  | terminalSingularSet
  | pressureGaugeQuotient
  | temporalCutoffConvention
  | parabolicCylinder
  | rescaledSolution
  | criticalQuantityC
  | criticalQuantityD
  | kineticQuantityA
  | dissipationQuantityE
  | criticalQuantityBundle
  | vanishingCriticalWitness
  | singularityWitness
  | regularityCertificate
  | positiveConcentrationSequence
  | finiteEnergyWitness
  | globalTypeIEnvelopeWitness
  | finiteEnergyTypeIEnvelope
  | localTypeIEnvelopeWitness
  | failedLocalTypeIEnvelopeWitness
  | velocityNoEscapeCertificate
  | terminalCompactnessWitness
  | localEnergyTransferData
  | pressureTransferData
  | scaleWindowData
  | pressureAtlasData
  | admissibleLocalTypeISequence
  | unifiedTypeIEntryData
  | blowupSequence
  | blowupCompactnessDossier
  | ancientSuitableTypeILimit
  | centeredVariables
  | centeredAncientProfile
  | compactL3Witness
  | normalizedSereginLimit
  | invariantLocalNonvanishingData
  | invariantLocalNonvanishing
  | generatedSereginState
  | rawGeneratedSereginStateSpace
  | mildStratum
  | classTag
  | smallClassPredicate
  | stationaryClassPredicate
  | tightClassPredicate
  | structureDecayPredicate
  | remainingClassPredicate
  | residualAncestryData
  | residualExportData
  | setupResidualInterfaceContract
  | residualImportedObject
  | residualAlternativeState
  | residualCompanionClosure
  | typeIIExportData
  | typeIIInterfaceContract
  | typeIIExternalClosure
  | residualClosureHypothesis
  | regularAtConclusion
  deriving DecidableEq, Repr, Fintype

namespace ProofSetupObjectId

def label : ProofSetupObjectId → String
  | .spatialPoint => "Spatial point"
  | .velocityVector => "Velocity vector"
  | .spacetimePoint => "Spacetime point"
  | .timeInterval => "Time interval"
  | .velocityField => "Velocity field"
  | .pressureField => "Pressure field"
  | .domainPredicate => "Spacetime domain predicate"
  | .suitableWeakSolution => "Suitable weak solution"
  | .terminalSingularPoint => "Terminal singular point"
  | .terminalSingularSet => "Terminal singular set"
  | .pressureGaugeQuotient => "Pressure gauge quotient"
  | .temporalCutoffConvention => "Temporal cutoff convention"
  | .parabolicCylinder => "Parabolic cylinder"
  | .rescaledSolution => "Rescaled solution"
  | .criticalQuantityC => "Critical quantity C"
  | .criticalQuantityD => "Critical quantity D"
  | .kineticQuantityA => "Kinetic quantity A"
  | .dissipationQuantityE => "Dissipation quantity E"
  | .criticalQuantityBundle => "Critical quantity bundle"
  | .vanishingCriticalWitness => "Vanishing critical quantities witness"
  | .singularityWitness => "Singularity witness"
  | .regularityCertificate => "Regularity certificate"
  | .positiveConcentrationSequence => "Positive concentration sequence"
  | .finiteEnergyWitness => "Finite energy witness"
  | .globalTypeIEnvelopeWitness => "Global Type I envelope witness"
  | .finiteEnergyTypeIEnvelope => "Finite-energy Type I envelope"
  | .localTypeIEnvelopeWitness => "Local Type I envelope witness"
  | .failedLocalTypeIEnvelopeWitness => "Failed local Type I envelope witness"
  | .velocityNoEscapeCertificate => "Velocity no-escape certificate"
  | .terminalCompactnessWitness => "Terminal compactness witness"
  | .localEnergyTransferData => "Local energy transfer data"
  | .pressureTransferData => "Pressure transfer data"
  | .scaleWindowData => "Scale window data"
  | .pressureAtlasData => "Pressure atlas data"
  | .admissibleLocalTypeISequence => "Admissible local Type I sequence"
  | .unifiedTypeIEntryData => "Unified Type I entry data"
  | .blowupSequence => "Blow-up sequence"
  | .blowupCompactnessDossier => "Blow-up compactness dossier"
  | .ancientSuitableTypeILimit => "Ancient suitable Type I limit"
  | .centeredVariables => "Centered variables"
  | .centeredAncientProfile => "Centered ancient profile"
  | .compactL3Witness => "Compact L3 witness"
  | .normalizedSereginLimit => "Normalized Seregin limit"
  | .invariantLocalNonvanishingData => "Invariant local nonvanishing data"
  | .invariantLocalNonvanishing => "Invariant local nonvanishing"
  | .generatedSereginState => "Generated Seregin state"
  | .rawGeneratedSereginStateSpace => "Raw generated Seregin state space"
  | .mildStratum => "Mild stratum"
  | .classTag => "Generated class tag"
  | .smallClassPredicate => "Small class predicate"
  | .stationaryClassPredicate => "Stationary class predicate"
  | .tightClassPredicate => "Tight class predicate"
  | .structureDecayPredicate => "Structure-decay predicate"
  | .remainingClassPredicate => "Remaining class predicate"
  | .residualAncestryData => "Residual ancestry data"
  | .residualExportData => "Residual export data"
  | .setupResidualInterfaceContract => "Setup-residual interface contract"
  | .residualImportedObject => "Residual imported setup object"
  | .residualAlternativeState => "Residual alternative state"
  | .residualCompanionClosure => "Residual companion closure"
  | .typeIIExportData => "Type II export data"
  | .typeIIInterfaceContract => "Type II interface contract"
  | .typeIIExternalClosure => "External Type II closure"
  | .residualClosureHypothesis => "Residual closure hypothesis"
  | .regularAtConclusion => "Regular-at conclusion"

def homeModule : ProofSetupObjectId → String
  | .spatialPoint
  | .velocityVector
  | .spacetimePoint
  | .timeInterval
  | .velocityField
  | .pressureField
  | .domainPredicate
  | .suitableWeakSolution
  | .terminalSingularPoint
  | .terminalSingularSet
  | .pressureGaugeQuotient
  | .temporalCutoffConvention
  | .parabolicCylinder
  | .rescaledSolution
  | .criticalQuantityC
  | .criticalQuantityD
  | .kineticQuantityA
  | .dissipationQuantityE
  | .criticalQuantityBundle
  | .vanishingCriticalWitness
  | .singularityWitness
  | .regularityCertificate
  | .positiveConcentrationSequence
  | .finiteEnergyWitness
  | .globalTypeIEnvelopeWitness
  | .finiteEnergyTypeIEnvelope
  | .localTypeIEnvelopeWitness
  | .failedLocalTypeIEnvelopeWitness
  | .velocityNoEscapeCertificate
  | .terminalCompactnessWitness
  | .localEnergyTransferData
  | .pressureTransferData
  | .scaleWindowData
  | .pressureAtlasData
  | .admissibleLocalTypeISequence
  | .blowupSequence
  | .ancientSuitableTypeILimit
  | .centeredVariables
  | .centeredAncientProfile
  | .compactL3Witness
  | .normalizedSereginLimit
  | .invariantLocalNonvanishingData
  | .invariantLocalNonvanishing
  | .generatedSereginState
  | .rawGeneratedSereginStateSpace
  | .mildStratum
  | .classTag
  | .smallClassPredicate
  | .stationaryClassPredicate
  | .tightClassPredicate
  | .structureDecayPredicate
  | .remainingClassPredicate
  | .residualAncestryData
  | .residualExportData => "ProofSetup.Basic"
  | .unifiedTypeIEntryData => "ProofSetup.EntryData"
  | .blowupCompactnessDossier => "ProofSetup.BlowupCompactness"
  | .setupResidualInterfaceContract
  | .residualClosureHypothesis => "ProofSetup.ResidualInterface"
  | .typeIIExportData
  | .typeIIInterfaceContract => "ProofSetup.TypeIIInterface"
  | .residualImportedObject
  | .residualAlternativeState
  | .residualCompanionClosure => "ResidualBranch"
  | .typeIIExternalClosure => "type_II_regularity external package"
  | .regularAtConclusion => "ProofSetup.FinalAssembly"

def carrier : ProofSetupObjectId → Type
  | .spatialPoint => SpatialPoint
  | .velocityVector => Vector3
  | .spacetimePoint => SpacetimePoint
  | .timeInterval => TimeInterval
  | .velocityField => VelocityField
  | .pressureField => PressureField
  | .domainPredicate => DomainPredicate
  | .suitableWeakSolution => SuitableWeakSolution
  | .terminalSingularPoint => TerminalSingularPoint
  | .terminalSingularSet => TerminalSingularSet
  | .pressureGaugeQuotient => PressureGaugeQuotient
  | .temporalCutoffConvention => TemporalCutoffConvention
  | .parabolicCylinder => ParabolicCylinder
  | .rescaledSolution => RescaledSolution
  | .criticalQuantityC => CriticalQuantityC
  | .criticalQuantityD => CriticalQuantityD
  | .kineticQuantityA => KineticQuantityA
  | .dissipationQuantityE => DissipationQuantityE
  | .criticalQuantityBundle => CriticalQuantityBundle
  | .vanishingCriticalWitness => VanishingCriticalQuantitiesWitness
  | .singularityWitness => SingularityWitness
  | .regularityCertificate => RegularityCertificate
  | .positiveConcentrationSequence => PositiveConcentrationSequence
  | .finiteEnergyWitness => FiniteEnergyWitness
  | .globalTypeIEnvelopeWitness => GlobalTypeIEnvelopeWitness
  | .finiteEnergyTypeIEnvelope => FiniteEnergyTypeIEnvelope
  | .localTypeIEnvelopeWitness => LocalTypeIEnvelopeWitness
  | .failedLocalTypeIEnvelopeWitness => FailedLocalTypeIEnvelopeWitness
  | .velocityNoEscapeCertificate => VelocityNoEscapeCertificate
  | .terminalCompactnessWitness => TerminalCompactnessWitness
  | .localEnergyTransferData => LocalEnergyTransferData
  | .pressureTransferData => PressureTransferData
  | .scaleWindowData => ScaleWindowData
  | .pressureAtlasData => PressureAtlasData
  | .admissibleLocalTypeISequence => AdmissibleLocalTypeISequence
  | .unifiedTypeIEntryData => UnifiedTypeIEntryData
  | .blowupSequence => BlowupSequence
  | .blowupCompactnessDossier => BlowupCompactnessDossier
  | .ancientSuitableTypeILimit => AncientSuitableTypeILimit
  | .centeredVariables => CenteredVariables
  | .centeredAncientProfile => CenteredAncientProfile
  | .compactL3Witness => CompactL3Witness
  | .normalizedSereginLimit => NormalizedSereginLimit
  | .invariantLocalNonvanishingData => InvariantLocalNonvanishingData
  | .invariantLocalNonvanishing => InvariantLocalNonvanishing
  | .generatedSereginState => GeneratedSereginState
  | .rawGeneratedSereginStateSpace => RawGeneratedSereginStateSpace
  | .mildStratum => MildStratum
  | .classTag => GeneratedClassTag
  | .smallClassPredicate => GeneratedSereginState → Prop
  | .stationaryClassPredicate => GeneratedSereginState → Prop
  | .tightClassPredicate => GeneratedSereginState → Prop
  | .structureDecayPredicate => GeneratedSereginState → Prop
  | .remainingClassPredicate => GeneratedSereginState → Prop
  | .residualAncestryData => ResidualAncestryData
  | .residualExportData => ResidualExportData
  | .setupResidualInterfaceContract => SetupResidualInterfaceContract
  | .residualImportedObject =>
      Hypostructure.Backends.NavierStokes.ResidualBranch.ImportedSetupResidualObject
  | .residualAlternativeState =>
      Hypostructure.Backends.NavierStokes.ResidualBranch.ResidualAlternativeState
  | .residualCompanionClosure => ResidualClosureHypothesis
  | .typeIIExportData => TypeIIExportData
  | .typeIIInterfaceContract => TypeIIInterfaceContract
  | .typeIIExternalClosure => Prop
  | .residualClosureHypothesis => ResidualClosureHypothesis
  | .regularAtConclusion => Prop

def rank : ProofSetupObjectId → Nat
  | .spatialPoint
  | .velocityVector
  | .spacetimePoint
  | .timeInterval
  | .velocityField
  | .pressureField
  | .domainPredicate => 0
  | .suitableWeakSolution
  | .terminalSingularPoint
  | .terminalSingularSet
  | .pressureGaugeQuotient
  | .temporalCutoffConvention
  | .parabolicCylinder => 1
  | .rescaledSolution
  | .criticalQuantityC
  | .criticalQuantityD
  | .kineticQuantityA
  | .dissipationQuantityE
  | .criticalQuantityBundle => 2
  | .vanishingCriticalWitness
  | .singularityWitness
  | .regularityCertificate
  | .positiveConcentrationSequence => 3
  | .finiteEnergyWitness
  | .globalTypeIEnvelopeWitness
  | .finiteEnergyTypeIEnvelope
  | .localTypeIEnvelopeWitness
  | .failedLocalTypeIEnvelopeWitness
  | .localEnergyTransferData
  | .pressureTransferData
  | .scaleWindowData
  | .pressureAtlasData => 4
  | .velocityNoEscapeCertificate
  | .terminalCompactnessWitness => 5
  | .admissibleLocalTypeISequence => 6
  | .typeIIExportData
  | .typeIIInterfaceContract => 5
  | .unifiedTypeIEntryData
  | .blowupSequence => 7
  | .blowupCompactnessDossier => 8
  | .ancientSuitableTypeILimit => 9
  | .centeredVariables
  | .centeredAncientProfile
  | .compactL3Witness
  | .normalizedSereginLimit => 10
  | .invariantLocalNonvanishingData
  | .invariantLocalNonvanishing => 11
  | .generatedSereginState
  | .rawGeneratedSereginStateSpace
  | .mildStratum => 12
  | .classTag
  | .smallClassPredicate
  | .stationaryClassPredicate
  | .tightClassPredicate
  | .structureDecayPredicate
  | .remainingClassPredicate => 13
  | .residualAncestryData
  | .residualExportData
  | .setupResidualInterfaceContract => 14
  | .residualImportedObject
  | .residualAlternativeState => 15
  | .residualCompanionClosure
  | .typeIIExternalClosure
  | .residualClosureHypothesis => 16
  | .regularAtConclusion => 17

end ProofSetupObjectId

structure DependencyStep (κ : Type _) where
  name : String
  theoremLabels : List String
  inputs : List κ
  outputs : List κ
  deriving Repr

namespace DependencyStep

def toRules
    {κ : Type _}
    [DecidableEq κ]
    (kind : RuleKind := .backend)
    (step : DependencyStep κ) : RuleSet κ :=
  step.outputs.map fun output =>
    { kind := kind
      premises := step.inputs.toFinset
      conclusion := output }

end DependencyStep

structure RankedDependencyStep
    (κ : Type _)
    [DecidableEq κ]
    (rank : κ → Nat) where
  step : DependencyStep κ
  kind : RuleKind := .backend
  rankOk : ∀ ⦃input output : κ⦄,
    input ∈ step.inputs → output ∈ step.outputs → rank input < rank output

namespace RankedDependencyStep

def toRules
    {κ : Type _}
    [DecidableEq κ]
    {rank : κ → Nat}
    (step : RankedDependencyStep κ rank) : RuleSet κ :=
  step.step.toRules step.kind

theorem mem_toRules_rank_lt
    {κ : Type _}
    [DecidableEq κ]
    {rank : κ → Nat}
    (step : RankedDependencyStep κ rank)
    {r : Rule κ}
    (hr : r ∈ step.toRules)
    {input : κ}
    (hinput : input ∈ r.premises) :
    rank input < rank r.conclusion := by
  unfold toRules DependencyStep.toRules at hr
  rcases List.mem_map.mp hr with ⟨output, houtput, hEq⟩
  rw [← hEq] at hinput ⊢
  exact step.rankOk (by simpa using hinput) houtput

end RankedDependencyStep

def rankedDependencyStepsToRules
    {κ : Type _}
    [DecidableEq κ]
    {rank : κ → Nat}
    (steps : List (RankedDependencyStep κ rank)) : RuleSet κ :=
  steps.bind RankedDependencyStep.toRules

theorem rankedDependencyStepsToRules_acyclic
    {κ : Type _}
    [DecidableEq κ]
    {rank : κ → Nat}
    (steps : List (RankedDependencyStep κ rank))
    {r : Rule κ}
    (hr : r ∈ rankedDependencyStepsToRules steps)
    {input : κ}
    (hinput : input ∈ r.premises) :
    rank input < rank r.conclusion := by
  unfold rankedDependencyStepsToRules at hr
  rw [List.mem_bind] at hr
  rcases hr with ⟨step, _hstep, hrule⟩
  exact step.mem_toRules_rank_lt hrule hinput

abbrev ProofSetupRankedStep :=
  RankedDependencyStep ProofSetupObjectId ProofSetupObjectId.rank

macro "rank_step" : tactic =>
  `(tactic| decide)

def proofSetupRankedDependencySteps : List ProofSetupRankedStep :=
  [ { step :=
        { name := "analytic primitives"
          theoremLabels := ["doc:p0:def:suitable-weak-solution", "p1:eq:NS"]
          inputs := []
          outputs := [.spatialPoint, .velocityVector, .spacetimePoint, .timeInterval,
            .velocityField, .pressureField, .domainPredicate] }
      rankOk := by intro input output hinput _houtput; cases hinput }
  , { step :=
        { name := "ambient suitable solution and singular point"
          theoremLabels := ["doc:p0:def:suitable-weak-solution", "doc:p0:def:singular-set-at-time", "p0:lem:temporal-cutoff", "p1:def:terminal-singular", "p1:def:suitable"]
          inputs := [.spatialPoint, .velocityVector, .spacetimePoint, .timeInterval,
            .velocityField, .pressureField, .domainPredicate]
          outputs := [.suitableWeakSolution, .terminalSingularPoint, .terminalSingularSet,
            .pressureGaugeQuotient, .temporalCutoffConvention, .parabolicCylinder] }
      rankOk := by rank_step }
  , { step :=
        { name := "scale-invariant package"
          theoremLabels := ["doc:p0:prop:parabolic-rescaling-invariance", "doc:p0:def:scale-invariant-local-quantities", "p0:prop:critical-scaling", "p1:lem:scaling"]
          inputs := [.suitableWeakSolution, .terminalSingularPoint, .parabolicCylinder]
          outputs := [.rescaledSolution, .criticalQuantityC, .criticalQuantityD,
            .kineticQuantityA, .dissipationQuantityE, .criticalQuantityBundle] }
      rankOk := by rank_step }
  , { step :=
        { name := "vanishing branch regularity"
          theoremLabels := ["doc:p0:def:vanishing-local-scale-invariant-quantities", "p0:thm:ckn", "p0:thm:velocity-ckn", "p0:thm:no-concentration", "p0:cor:escape"]
          inputs := [.suitableWeakSolution, .terminalSingularPoint, .criticalQuantityBundle]
          outputs := [.vanishingCriticalWitness, .regularityCertificate] }
      rankOk := by rank_step }
  , { step :=
        { name := "positive concentration"
          theoremLabels := ["p0:lem:positive-concentration", "p0:def:positive-concentration-sequence", "p0:thm:singular-positive", "p0:cor:velocity-concentration"]
          inputs := [.suitableWeakSolution, .terminalSingularPoint, .criticalQuantityBundle]
          outputs := [.singularityWitness, .positiveConcentrationSequence] }
      rankOk := by rank_step }
  , { step :=
        { name := "finite energy and envelope data"
          theoremLabels := ["p0:def:finite-energy-typeI-paper0", "p1:def:suitable-leray-hopf", "p1:def:admissible-type-I"]
          inputs := [.suitableWeakSolution, .terminalSingularPoint, .positiveConcentrationSequence]
          outputs := [.finiteEnergyWitness, .globalTypeIEnvelopeWitness, .finiteEnergyTypeIEnvelope,
            .localTypeIEnvelopeWitness, .failedLocalTypeIEnvelopeWitness, .localEnergyTransferData,
            .pressureTransferData, .scaleWindowData, .pressureAtlasData] }
      rankOk := by rank_step }
  , { step :=
        { name := "no-escape and terminal compactness"
          theoremLabels := ["p0:lem:uloc-typeI-propagation", "p0:prop:auto-terminal-A", "p0:thm:auto-velocity-no-escape", "p0:thm:local-weak-serrin-typeI"]
          inputs := [.finiteEnergyTypeIEnvelope, .localTypeIEnvelopeWitness, .positiveConcentrationSequence]
          outputs := [.velocityNoEscapeCertificate, .terminalCompactnessWitness] }
      rankOk := by rank_step }
  , { step :=
        { name := "local Type I admissibility"
          theoremLabels := ["p0:def:paper0-admissible-local-typeI-sequence", "p0:cor:paper0-local-typeI-admissible", "p0:lem:paper0-local-typeI-into-terminal-dichotomy"]
          inputs := [.positiveConcentrationSequence, .localTypeIEnvelopeWitness,
            .velocityNoEscapeCertificate, .terminalCompactnessWitness]
          outputs := [.admissibleLocalTypeISequence] }
      rankOk := by rank_step }
  , { step :=
        { name := "local Type II export"
          theoremLabels := ["p0:def:typeII-alternative", "p0:thm:typeI-typeII-dichotomy", "p0:prop:concentration-package"]
          inputs := [.positiveConcentrationSequence, .failedLocalTypeIEnvelopeWitness,
            .localEnergyTransferData, .pressureTransferData, .scaleWindowData]
          outputs := [.typeIIExportData, .typeIIInterfaceContract] }
      rankOk := by rank_step }
  , { step :=
        { name := "unified Type I entry"
          theoremLabels := ["p1:def:admissible-type-I", "p0:lem:local-seregin-extraction"]
          inputs := [.admissibleLocalTypeISequence, .finiteEnergyTypeIEnvelope]
          outputs := [.unifiedTypeIEntryData, .blowupSequence] }
      rankOk := by rank_step }
  , { step :=
        { name := "blow-up compactness dossier"
          theoremLabels := ["p1:lem:inherited-bound", "p1:lem:pressure", "p1:lem:pressure-atlas", "p1:prop:compactness"]
          inputs := [.unifiedTypeIEntryData, .blowupSequence, .pressureAtlasData]
          outputs := [.blowupCompactnessDossier] }
      rankOk := by rank_step }
  , { step :=
        { name := "ancient limit"
          theoremLabels := ["p1:lem:limit-equation", "p1:lem:suitability", "p1:lem:limit-bound", "p1:prop:ancient-limit", "p1:prop:nonzero"]
          inputs := [.blowupCompactnessDossier]
          outputs := [.ancientSuitableTypeILimit] }
      rankOk := by rank_step }
  , { step :=
        { name := "centered profile and raw extraction"
          theoremLabels := ["p1:lem:centered-pullback-identities", "p1:lem:renormalized-equation", "p1:def:seregin-limit", "p1:cor:raw-extracted-nonzero"]
          inputs := [.ancientSuitableTypeILimit]
          outputs := [.centeredVariables, .centeredAncientProfile, .compactL3Witness,
            .normalizedSereginLimit] }
      rankOk := by rank_step }
  , { step :=
        { name := "invariant local nonvanishing"
          theoremLabels := ["p1:lem:compact-to-invariant-nonzero"]
          inputs := [.normalizedSereginLimit, .compactL3Witness]
          outputs := [.invariantLocalNonvanishingData, .invariantLocalNonvanishing] }
      rankOk := by rank_step }
  , { step :=
        { name := "generated state space and mild stratum"
          theoremLabels := ["p1:def:raw-generated-seregin-space", "p1:def:seregin-collection", "p1:def:mild-stratum", "p1:lem:raw-generated-closure-stability", "p1:lem:mild-stratum-stability"]
          inputs := [.normalizedSereginLimit, .invariantLocalNonvanishing]
          outputs := [.generatedSereginState, .rawGeneratedSereginStateSpace, .mildStratum] }
      rankOk := by rank_step }
  , { step :=
        { name := "class exhaustion"
          theoremLabels := ["p1:def:small-class", "p1:def:stationary-class", "p1:def:tight-class", "p1:def:known-structure-decay-class", "p1:def:remaining-class", "p1:prop:exhaustion"]
          inputs := [.generatedSereginState, .rawGeneratedSereginStateSpace, .mildStratum]
          outputs := [.classTag, .smallClassPredicate, .stationaryClassPredicate,
            .tightClassPredicate, .structureDecayPredicate, .remainingClassPredicate] }
      rankOk := by rank_step }
  , { step :=
        { name := "known class closures"
          theoremLabels := ["p1:thm:classical-classes", "p1:thm:tight-liouville", "p1:thm:known-structure-decay-liouville", "p1:prop:known-class-exclusion"]
          inputs := [.smallClassPredicate, .stationaryClassPredicate, .tightClassPredicate,
            .structureDecayPredicate, .invariantLocalNonvanishing]
          outputs := [.regularAtConclusion] }
      rankOk := by rank_step }
  , { step :=
        { name := "residual export"
          theoremLabels := ["p1:hyp:no-remainder", "p1:cor:remainder-rigidity", "p1:prop:remainder-equivalence"]
          inputs := [.rawGeneratedSereginStateSpace, .generatedSereginState,
            .remainingClassPredicate]
          outputs := [.residualAncestryData, .residualExportData, .setupResidualInterfaceContract] }
      rankOk := by rank_step }
  , { step :=
        { name := "residual companion import and routing"
          theoremLabels := ["thm:imported-setup-results", "hyp:base-seregin-hypotheses"]
          inputs := [.residualExportData, .setupResidualInterfaceContract]
          outputs := [.residualImportedObject, .residualAlternativeState] }
      rankOk := by rank_step }
  , { step :=
        { name := "residual companion closure feedback"
          theoremLabels := ["thm:paperIV-residual-closure", "cor:setup-residual-hypothesis-proof"]
          inputs := [.residualImportedObject, .residualAlternativeState]
          outputs := [.residualCompanionClosure, .residualClosureHypothesis] }
      rankOk := by rank_step }
  , { step :=
        { name := "external Type II closure"
          theoremLabels := ["paper6:thm:paper0", "thm:c1-typeII-branch-exhaustion", "paper6:thm:physical-typeII-covered-entry", "paper6a:thm:typeII-analytic-data-exhaustive"]
          inputs := [.typeIIExportData, .typeIIInterfaceContract]
          outputs := [.typeIIExternalClosure] }
      rankOk := by rank_step }
  , { step :=
        { name := "final assembly"
          theoremLabels := ["p1:thm:final-assembly", "p0:prop:local-typeI-entry-final-assembly", "p0:cor:paper0-local-typeII-after-local-typeI-reduction"]
          inputs := [.residualClosureHypothesis, .typeIIInterfaceContract, .typeIIExternalClosure]
          outputs := [.regularAtConclusion] }
      rankOk := by rank_step } ]

def proofSetupDependencySteps : List (DependencyStep ProofSetupObjectId) :=
  proofSetupRankedDependencySteps.map (fun step => step.step)

def proofSetupRules : RuleSet ProofSetupObjectId :=
  rankedDependencyStepsToRules proofSetupRankedDependencySteps

theorem proofSetupRules_acyclic
    {r : Rule ProofSetupObjectId}
    (hr : r ∈ proofSetupRules)
    {input : ProofSetupObjectId}
    (hinput : input ∈ r.premises) :
    ProofSetupObjectId.rank input < ProofSetupObjectId.rank r.conclusion := by
  exact rankedDependencyStepsToRules_acyclic proofSetupRankedDependencySteps hr hinput

def proofSetupFiniteSieve : FiniteSieve ProofSetupObjectId where
  rules := proofSetupRules
  rank := ProofSetupObjectId.rank
  acyclic := by
    intro r hr input hinput
    exact proofSetupRules_acyclic hr hinput

def proofSetupAnchorContext : Context ProofSetupObjectId :=
  ([ .spatialPoint, .velocityVector, .spacetimePoint, .timeInterval,
     .velocityField, .pressureField, .domainPredicate ] : List ProofSetupObjectId).toFinset

def proofSetupRunFromAnchors : Context ProofSetupObjectId :=
  runDag proofSetupFiniteSieve proofSetupAnchorContext

theorem proofSetupEdgeRankStrict
    {a b : ProofSetupObjectId}
    (h : edgeRelation proofSetupFiniteSieve a b) :
    ProofSetupObjectId.rank a < ProofSetupObjectId.rank b :=
  edge_rank_lt proofSetupFiniteSieve h

theorem proofSetupAnchorsSurviveRun :
    proofSetupAnchorContext ⊆ proofSetupRunFromAnchors := by
  exact subset_runDag proofSetupFiniteSieve proofSetupAnchorContext

end Hypostructure.Backends.NavierStokes.ProofSetup
