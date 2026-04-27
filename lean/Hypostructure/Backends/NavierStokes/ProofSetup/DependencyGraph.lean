import Hypostructure.Backends.NavierStokes.ProofSetup.FinalAssembly
import Hypostructure.Backends.NavierStokes.ProofSetup.TypeIIInterface
import Hypostructure.Backends.NavierStokes.ResidualBranch.FinalClosure

namespace Hypostructure.Backends.NavierStokes.ProofSetup

inductive ProofSetupObjectId where
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
  | finiteEnergyTypeIEnvelope
  | localTypeIEnvelopeWitness
  | failedLocalTypeIEnvelopeWitness
  | velocityNoEscapeCertificate
  | terminalCompactnessWitness
  | localEnergyTransferData
  | pressureTransferData
  | scaleWindowData
  | pressureAtlasData
  | blowupSequence
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
    | residualImportedObject
    | residualAlternativeState
    | residualCompanionClosure
  | typeIIExportData
  | typeIIInterfaceContract
    | typeIIExternalClosure
  | residualClosureHypothesis
  | regularAtConclusion
  deriving DecidableEq, Repr, Fintype

def ProofSetupObjectId.label : ProofSetupObjectId → String
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
  | .finiteEnergyTypeIEnvelope => "Finite-energy Type I envelope"
  | .localTypeIEnvelopeWitness => "Local Type I envelope witness"
  | .failedLocalTypeIEnvelopeWitness => "Failed local Type I envelope witness"
  | .velocityNoEscapeCertificate => "Velocity no-escape certificate"
  | .terminalCompactnessWitness => "Terminal compactness witness"
  | .localEnergyTransferData => "Local energy transfer data"
  | .pressureTransferData => "Pressure transfer data"
  | .scaleWindowData => "Scale window data"
  | .pressureAtlasData => "Pressure atlas data"
  | .blowupSequence => "Blow-up sequence"
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
    | .residualImportedObject => "Residual imported setup object"
    | .residualAlternativeState => "Residual alternative state"
    | .residualCompanionClosure => "Residual companion closure"
  | .typeIIExportData => "Type II export data"
  | .typeIIInterfaceContract => "Type II interface contract"
    | .typeIIExternalClosure => "External Type II closure"
  | .residualClosureHypothesis => "Residual closure hypothesis"
  | .regularAtConclusion => "Regular-at conclusion"

def ProofSetupObjectId.homeModule : ProofSetupObjectId → String
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
  | .finiteEnergyTypeIEnvelope
  | .localTypeIEnvelopeWitness
  | .failedLocalTypeIEnvelopeWitness
  | .velocityNoEscapeCertificate
  | .terminalCompactnessWitness
  | .localEnergyTransferData
  | .pressureTransferData
  | .scaleWindowData
  | .pressureAtlasData
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
    | .residualImportedObject
    | .residualAlternativeState
    | .residualCompanionClosure => "ResidualBranch"
  | .typeIIExportData
  | .typeIIInterfaceContract => "ProofSetup.TypeIIInterface"
    | .typeIIExternalClosure => "type_II_regularity external package"
  | .residualClosureHypothesis => "ProofSetup.ResidualInterface"
  | .regularAtConclusion => "ProofSetup.FinalAssembly"

def ProofSetupObjectId.carrier : ProofSetupObjectId → Type
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
  | .finiteEnergyTypeIEnvelope => FiniteEnergyTypeIEnvelope
  | .localTypeIEnvelopeWitness => LocalTypeIEnvelopeWitness
  | .failedLocalTypeIEnvelopeWitness => FailedLocalTypeIEnvelopeWitness
  | .velocityNoEscapeCertificate => VelocityNoEscapeCertificate
  | .terminalCompactnessWitness => TerminalCompactnessWitness
  | .localEnergyTransferData => LocalEnergyTransferData
  | .pressureTransferData => PressureTransferData
  | .scaleWindowData => ScaleWindowData
  | .pressureAtlasData => PressureAtlasData
  | .blowupSequence => BlowupSequence
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

structure DependencyStep where
  name : String
  theoremLabels : List String
  inputs : List ProofSetupObjectId
  outputs : List ProofSetupObjectId
  deriving Repr

def proofSetupDependencySteps : List DependencyStep :=
  [ { name := "ambient input"
      theoremLabels := ["doc:p0:def:suitable-weak-solution", "doc:p0:def:singular-set-at-time", "p0:lem:temporal-cutoff"]
      inputs := []
      outputs := [.suitableWeakSolution, .terminalSingularPoint, .terminalSingularSet, .pressureGaugeQuotient, .temporalCutoffConvention] }
  , { name := "scale-invariant package"
      theoremLabels := ["doc:p0:prop:parabolic-rescaling-invariance", "doc:p0:def:scale-invariant-local-quantities", "p0:prop:critical-scaling"]
      inputs := [.suitableWeakSolution, .terminalSingularPoint]
      outputs := [.parabolicCylinder, .rescaledSolution, .criticalQuantityC, .criticalQuantityD, .kineticQuantityA, .dissipationQuantityE, .criticalQuantityBundle] }
  , { name := "vanishing branch regularity"
      theoremLabels := ["p0:thm:ckn", "p0:thm:velocity-ckn", "p0:thm:no-concentration", "p0:cor:escape"]
      inputs := [.suitableWeakSolution, .terminalSingularPoint, .criticalQuantityBundle]
      outputs := [.vanishingCriticalWitness, .regularityCertificate] }
  , { name := "positive concentration"
      theoremLabels := ["p0:lem:positive-concentration", "p0:thm:singular-positive", "p0:cor:velocity-concentration"]
      inputs := [.suitableWeakSolution, .terminalSingularPoint, .criticalQuantityBundle]
      outputs := [.singularityWitness, .positiveConcentrationSequence] }
  , { name := "finite-energy type I and no-escape"
      theoremLabels := ["p0:def:finite-energy-typeI-paper0", "p0:lem:uloc-typeI-propagation", "p0:prop:auto-terminal-A", "p0:thm:auto-velocity-no-escape"]
      inputs := [.suitableWeakSolution, .terminalSingularPoint, .positiveConcentrationSequence]
      outputs := [.finiteEnergyWitness, .finiteEnergyTypeIEnvelope, .velocityNoEscapeCertificate, .terminalCompactnessWitness] }
  , { name := "local dichotomy and Type II export"
      theoremLabels := ["p0:thm:typeI-typeII-dichotomy", "p0:thm:local-weak-serrin-typeI", "p0:cor:paper0-local-typeI-admissible"]
      inputs := [.positiveConcentrationSequence, .localTypeIEnvelopeWitness, .failedLocalTypeIEnvelopeWitness, .velocityNoEscapeCertificate, .terminalCompactnessWitness, .localEnergyTransferData, .pressureTransferData, .scaleWindowData]
      outputs := [.typeIIExportData, .typeIIInterfaceContract] }
  , { name := "Type I admissible entry"
      theoremLabels := ["p0:def:paper0-admissible-local-typeI-sequence", "p0:lem:local-seregin-extraction"]
      inputs := [.positiveConcentrationSequence, .localTypeIEnvelopeWitness, .velocityNoEscapeCertificate, .terminalCompactnessWitness]
      outputs := [.blowupSequence] }
  , { name := "blow-up compactness and ancient limit"
      theoremLabels := ["p1:prop:compactness", "p1:prop:ancient-limit", "p1:prop:nonzero"]
      inputs := [.blowupSequence, .pressureAtlasData]
      outputs := [.ancientSuitableTypeILimit] }
  , { name := "centered profile and raw extraction"
      theoremLabels := ["p1:lem:centered-pullback-identities", "p1:lem:renormalized-equation", "p1:def:seregin-limit"]
      inputs := [.ancientSuitableTypeILimit]
      outputs := [.centeredVariables, .centeredAncientProfile, .compactL3Witness, .normalizedSereginLimit] }
  , { name := "invariant local nonvanishing"
      theoremLabels := ["p1:lem:compact-to-invariant-nonzero"]
      inputs := [.normalizedSereginLimit, .compactL3Witness]
      outputs := [.invariantLocalNonvanishingData, .invariantLocalNonvanishing] }
  , { name := "generated state space and mild stratum"
      theoremLabels := ["p1:def:raw-generated-seregin-space", "p1:def:mild-stratum", "p1:lem:raw-generated-closure-stability", "p1:lem:mild-stratum-stability"]
      inputs := [.normalizedSereginLimit, .invariantLocalNonvanishing]
      outputs := [.generatedSereginState, .rawGeneratedSereginStateSpace, .mildStratum] }
  , { name := "class exhaustion"
      theoremLabels := ["p1:def:small-class", "p1:def:stationary-class", "p1:def:tight-class", "p1:def:known-structure-decay-class", "p1:def:remaining-class", "p1:prop:exhaustion"]
      inputs := [.generatedSereginState, .rawGeneratedSereginStateSpace]
      outputs := [.classTag, .smallClassPredicate, .stationaryClassPredicate, .tightClassPredicate, .structureDecayPredicate, .remainingClassPredicate] }
  , { name := "known class closures"
      theoremLabels := ["p1:thm:classical-classes", "p1:thm:tight-liouville", "p1:thm:known-structure-decay-liouville"]
      inputs := [.smallClassPredicate, .stationaryClassPredicate, .tightClassPredicate, .structureDecayPredicate, .invariantLocalNonvanishing]
      outputs := [.regularAtConclusion] }
  , { name := "residual export"
      theoremLabels := ["p1:hyp:no-remainder", "p1:prop:remainder-equivalence"]
      inputs := [.rawGeneratedSereginStateSpace, .generatedSereginState, .remainingClassPredicate]
      outputs := [.residualAncestryData, .residualExportData, .residualClosureHypothesis] }
  , { name := "residual companion import and routing"
      theoremLabels := ["thm:imported-setup-results", "hyp:base-seregin-hypotheses"]
      inputs := [.residualExportData]
      outputs := [.residualImportedObject, .residualAlternativeState] }
  , { name := "residual companion closure feedback"
      theoremLabels := ["thm:paperIV-residual-closure", "cor:setup-residual-hypothesis-proof"]
      inputs := [.residualImportedObject, .residualAlternativeState]
      outputs := [.residualCompanionClosure, .residualClosureHypothesis] }
  , { name := "external Type II closure"
      theoremLabels := ["paper6:thm:paper0", "thm:c1-typeII-branch-exhaustion", "paper6:thm:physical-typeII-covered-entry", "paper6a:thm:typeII-analytic-data-exhaustive"]
      inputs := [.typeIIExportData, .typeIIInterfaceContract]
      outputs := [.typeIIExternalClosure] }
  , { name := "final assembly"
      theoremLabels := ["p1:thm:final-assembly"]
      inputs := [.residualClosureHypothesis, .typeIIInterfaceContract, .typeIIExternalClosure]
      outputs := [.regularAtConclusion] } ]

end Hypostructure.Backends.NavierStokes.ProofSetup