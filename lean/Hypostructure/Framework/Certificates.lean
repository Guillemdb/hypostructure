import Hypostructure.Framework.Document

namespace Hypostructure.Framework

universe u v

structure PositiveCertificate (α : Type u) where
  node : GateNode
  payload : α
  meaning : Prop

structure WitnessNoCertificate (α : Type u) where
  node : GateNode
  payload : α
  witnessFound : Prop

structure InconclusiveNoCertificate (α : Type u) where
  node : GateNode
  payload : α
  missing : Finset NodeTag
  trace : List GateNode

structure BlockedCertificate (α : Type u) where
  barrier : BarrierNode
  payload : α
  continuation : GateNode

structure BreachedCertificate (α : Type u) where
  barrier : BarrierNode
  payload : α
  mode : ModeId
  surgery : SurgeryId
  reentry : GateNode

structure BenignCertificate (α : Type u) where
  barrier : BarrierNode
  payload : α
  continuation : GateNode

structure PathologicalCertificate (α : Type u) where
  barrier : BarrierNode
  payload : α
  mode : ModeId
  surgery : SurgeryId
  reentry : GateNode

structure StagnationCertificate (α : Type u) where
  barrier : BarrierNode
  payload : α
  restoration : GateNode

structure EnergyPayload where
  heightName : String
  dissipationName : String
  boundStatement : Prop

abbrev EnergyCertificate := PositiveCertificate EnergyPayload

structure RecoveryPayload where
  badSetName : String
  recoveryMapDeclared : Bool
  eventCount : ℕ

abbrev RecoveryCertificate := PositiveCertificate RecoveryPayload

structure CompactnessPayload where
  symmetryGroupName : String
  quotientName : String
  profileName : String
  compactnessStatement : Prop

abbrev CompactnessCertificate := PositiveCertificate CompactnessPayload

structure ScalingPayload where
  alpha : ℤ
  beta : ℤ
  route : String

abbrev ScalingCertificate := PositiveCertificate ScalingPayload

structure ParameterPayload where
  parameterSpace : String
  referencePoint : String
  stableStatement : Prop

abbrev ParameterCertificate := PositiveCertificate ParameterPayload

structure CapacityPayload where
  singularSetName : String
  capacityValue : ℝ
  negligible : Prop

abbrev CapacityCertificate := PositiveCertificate CapacityPayload

structure StiffnessPayload where
  gapConstant : ℝ
  exponent : ℝ
  coercivityStatement : Prop

abbrev StiffnessCertificate := PositiveCertificate StiffnessPayload

structure TopologyPayload where
  invariantName : String
  sectorStatement : Prop

abbrev TopologyCertificate := PositiveCertificate TopologyPayload

structure TamenessPayload where
  structureName : String
  stratificationBound : ℕ
  tameStatement : Prop

abbrev TamenessCertificate := PositiveCertificate TamenessPayload

structure MixingPayload where
  invariantMeasureName : String
  mixingTimeFinite : Prop
  convergenceStatement : Prop

abbrev MixingCertificate := PositiveCertificate MixingPayload

structure RepresentationPayload where
  languageName : String
  dictionaryName : String
  faithfulStatement : Prop

abbrev RepresentationCertificate := PositiveCertificate RepresentationPayload

structure GradientPayload where
  metricName : String
  vectorFieldName : String
  monotonicityStatement : Prop

abbrev GradientCertificate := PositiveCertificate GradientPayload

structure BoundaryClosedPayload where
  reason : String

abbrev BoundaryClosedCertificate := WitnessNoCertificate BoundaryClosedPayload

structure GermPayload where
  libraryName : String
  smallnessWitness : Prop

abbrev GermCertificate := PositiveCertificate GermPayload

structure InitialityPayload where
  universalBadName : String
  initialityWitness : Prop

abbrev InitialityCertificate := PositiveCertificate InitialityPayload

structure CatLibPayload where
  libraryName : String
  completenessWitness : Prop

abbrev CatLibCertificate := PositiveCertificate CatLibPayload

structure ColeHopfPayload where
  transformName : String
  targetSemigroup : String
  bridgeStatement : Prop

abbrev ColeHopfCertificate := PositiveCertificate ColeHopfPayload

structure HeatSmoothPayload where
  semigroupName : String
  smoothingStatement : Prop
  uniquenessStatement : Prop

abbrev HeatSmoothCertificate := PositiveCertificate HeatSmoothPayload

structure StructRegPayload where
  backendName : String
  obstructionEmpty : Prop

abbrev StructRegCertificate := PositiveCertificate StructRegPayload

structure AnalyticRegularityPayload where
  backendName : String
  targetClaim : Prop

abbrev AnalyticRegularityCertificate := PositiveCertificate AnalyticRegularityPayload

structure LyapunovPayload where
  functionName : String
  minimumSetName : String
  monotoneStatement : Prop

abbrev LyapunovCertificate := PositiveCertificate LyapunovPayload

structure JacobiPayload where
  metricName : String
  distanceName : String
  comparisonStatement : Prop

abbrev JacobiCertificate := PositiveCertificate JacobiPayload

structure HamiltonJacobiPayload where
  functionName : String
  gradientName : String
  relationStatement : Prop

abbrev HamiltonJacobiCertificate := PositiveCertificate HamiltonJacobiPayload

structure BarrierSatPayload where
  energyBounded : Prop
  driftBounded : Prop

abbrev BarrierSatBlockedCertificate := BlockedCertificate BarrierSatPayload
abbrev BarrierSatBreachedCertificate := BreachedCertificate BarrierSatPayload

structure BarrierCausalPayload where
  depthIntegralInfinite : Prop

abbrev BarrierCausalBlockedCertificate := BlockedCertificate BarrierCausalPayload
abbrev BarrierCausalBreachedCertificate := BreachedCertificate BarrierCausalPayload

structure BarrierScatPayload where
  interactionFinite : Prop

abbrev BarrierScatBenignCertificate := BenignCertificate BarrierScatPayload
abbrev BarrierScatPathologicalCertificate := PathologicalCertificate BarrierScatPayload

structure BarrierTypeIIPayload where
  renormCostInfinite : Prop

abbrev BarrierTypeIIBlockedCertificate := BlockedCertificate BarrierTypeIIPayload
abbrev BarrierTypeIIBreachedCertificate := BreachedCertificate BarrierTypeIIPayload

structure BarrierVacPayload where
  thermalBarrierStable : Prop

abbrev BarrierVacBlockedCertificate := BlockedCertificate BarrierVacPayload
abbrev BarrierVacBreachedCertificate := BreachedCertificate BarrierVacPayload

structure BarrierCapPayload where
  zeroCapacity : Prop

abbrev BarrierCapBlockedCertificate := BlockedCertificate BarrierCapPayload
abbrev BarrierCapBreachedCertificate := BreachedCertificate BarrierCapPayload

structure BarrierGapPayload where
  spectralGapPositive : Prop

abbrev BarrierGapBlockedCertificate := BlockedCertificate BarrierGapPayload
abbrev BarrierGapStagnationCertificate := StagnationCertificate BarrierGapPayload

structure BarrierActionPayload where
  actionGapProtected : Prop

abbrev BarrierActionBlockedCertificate := BlockedCertificate BarrierActionPayload
abbrev BarrierActionBreachedCertificate := BreachedCertificate BarrierActionPayload

structure BarrierOminPayload where
  definable : Prop

abbrev BarrierOminBlockedCertificate := BlockedCertificate BarrierOminPayload
abbrev BarrierOminBreachedCertificate := BreachedCertificate BarrierOminPayload

structure BarrierMixPayload where
  finiteMixingTime : Prop

abbrev BarrierMixBlockedCertificate := BlockedCertificate BarrierMixPayload
abbrev BarrierMixBreachedCertificate := BreachedCertificate BarrierMixPayload

structure BarrierEpiPayload where
  boundedApproxComplexity : Prop

abbrev BarrierEpiBlockedCertificate := BlockedCertificate BarrierEpiPayload
abbrev BarrierEpiBreachedCertificate := BreachedCertificate BarrierEpiPayload

structure BarrierFreqPayload where
  finiteOscillationEnergy : Prop

abbrev BarrierFreqBlockedCertificate := BlockedCertificate BarrierFreqPayload
abbrev BarrierFreqBreachedCertificate := BreachedCertificate BarrierFreqPayload

structure BarrierBodePayload where
  finiteBodeIntegral : Prop

abbrev BarrierBodeBlockedCertificate := BlockedCertificate BarrierBodePayload
abbrev BarrierBodeBreachedCertificate := BreachedCertificate BarrierBodePayload

structure BarrierInputPayload where
  positiveReserve : Prop

abbrev BarrierInputBlockedCertificate := BlockedCertificate BarrierInputPayload
abbrev BarrierInputBreachedCertificate := BreachedCertificate BarrierInputPayload

structure BarrierVarietyPayload where
  requisiteVariety : Prop

abbrev BarrierVarietyBlockedCertificate := BlockedCertificate BarrierVarietyPayload
abbrev BarrierVarietyBreachedCertificate := BreachedCertificate BarrierVarietyPayload

structure BarrierExclusionPayload where
  tacticName : String
  obstructionEmpty : Prop

abbrev BarrierExclusionBlockedCertificate := BlockedCertificate BarrierExclusionPayload
abbrev BarrierExclusionBreachedCertificate := BreachedCertificate BarrierExclusionPayload

end Hypostructure.Framework
