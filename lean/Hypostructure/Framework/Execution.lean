import Hypostructure.Framework.Barriers

namespace Hypostructure.Framework

inductive OutcomeTag
  | yes
  | noWitness
  | noInconclusive
  | blocked
  | breached
  | benign
  | pathological
  | stagnation
  | derived
  | notApplicable
  deriving DecidableEq, Repr, Fintype

inductive TraceLocation
  | gate (node : GateNode)
  | barrier (barrier : BarrierNode)
  | derived (name : String)
  deriving DecidableEq, Repr

structure CertificateTraceEntry where
  index : Nat
  location : TraceLocation
  certificateName : String
  outcome : OutcomeTag
  summary : String

abbrev ExecutionTrace := List CertificateTraceEntry

def ExecutionTrace.names (tr : ExecutionTrace) : List String :=
  tr.map (·.certificateName)

def ExecutionTrace.contains (tr : ExecutionTrace) (name : String) : Prop :=
  name ∈ tr.names

structure UpgradeRule where
  name : String
  premises : List String
  conclusion : String
  nonCircular : Prop

structure LockCertificatePackage where
  tacticName : String
  requiredCertificates : List String
  blockedCertificate : String

structure ObligationEntry where
  id : String
  location : TraceLocation
  certificateName : String
  obligation : String
  inGoalCone : Bool

abbrev ObligationLedger := List ObligationEntry

def ObligationLedger.goalConeOpen (L : ObligationLedger) : Prop :=
  ∃ e ∈ L, e.inGoalCone = true

def ObligationLedger.goalConeEmpty (L : ObligationLedger) : Prop :=
  ¬ L.goalConeOpen

structure FinalCertificateChain where
  certificates : List String
  designatedGoal : String
  containsGoal : designatedGoal ∈ certificates

structure RunValidity where
  allCoreNodesExecuted : Prop
  boundaryHandled : Prop
  lockExecuted : Prop
  upgradeCompleted : Prop
  goalConeEmpty : Prop
  designatedGoalReached : Prop := True
  lockCompletenessPresent : Prop := True
  analyticPermitPresent : Prop := True
  preservationLemmasPresent : Prop := True

def RunValidity.meetsTemplateCompletionCriteria (V : RunValidity) : Prop :=
  V.allCoreNodesExecuted ∧
    V.boundaryHandled ∧
    V.lockExecuted ∧
    V.upgradeCompleted ∧
    V.goalConeEmpty ∧
    V.designatedGoalReached ∧
    V.lockCompletenessPresent ∧
    V.analyticPermitPresent ∧
    V.preservationLemmasPresent

theorem RunValidity.templateCriteria_intro
    {V : RunValidity}
    (hcore : V.allCoreNodesExecuted)
    (hboundary : V.boundaryHandled)
    (hlock : V.lockExecuted)
    (hupgrade : V.upgradeCompleted)
    (hcone : V.goalConeEmpty)
    (hgoal : V.designatedGoalReached)
    (hcomplete : V.lockCompletenessPresent)
    (hanalytic : V.analyticPermitPresent)
    (hpres : V.preservationLemmasPresent) :
    V.meetsTemplateCompletionCriteria := by
  exact ⟨hcore, hboundary, hlock, hupgrade, hcone, hgoal, hcomplete, hanalytic, hpres⟩

theorem emptyLedger_goalConeEmpty :
    ObligationLedger.goalConeEmpty [] := by
  simp [ObligationLedger.goalConeEmpty, ObligationLedger.goalConeOpen]

theorem finalChain_contains
    (C : FinalCertificateChain)
    : C.designatedGoal ∈ C.certificates :=
  C.containsGoal

end Hypostructure.Framework
