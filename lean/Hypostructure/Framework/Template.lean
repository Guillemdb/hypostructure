import Hypostructure.Framework.Execution

namespace Hypostructure.Framework

structure ProofMetadata where
  problemName : String
  problemSlug : String
  systemType : BackendType
  targetClaim : String
  frameworkVersion : String := "Hypostructure v1.0"
  date : String

structure LabelNamingConventions where
  definitionPrefix : String
  theoremPrefix : String
  lemmaPrefix : String
  remarkPrefix : String
  proofPrefix : String
  sketchPrefix : String

def defaultLabelNamingConventions (slug : String) : LabelNamingConventions where
  definitionPrefix := s!"def-{slug}-"
  theoremPrefix := s!"thm-{slug}-"
  lemmaPrefix := s!"lem-{slug}-"
  remarkPrefix := s!"rem-{slug}-"
  proofPrefix := s!"proof-{slug}-"
  sketchPrefix := s!"sketch-{slug}-"

structure AutomationWitnessSection where
  typeWitness : Prop
  automationGuarantee : Prop
  factoriesEnabled : List String

structure AbstractSection where
  approach : String
  result : String

structure TheoremStatementSection where
  stateSpace : String
  dynamics : String
  initialData : String
  claim : String
  notationTable : List (String × String)

structure ThinObjectSection where
  arena : String
  potential : String
  cost : String
  invariance : String

structure InterfaceChecklist where
  energyCheck : Prop
  zenoCheck : Prop
  compactCheck : Prop
  scaleCheck : Prop
  paramCheck : Prop
  geomCheck : Prop
  stiffnessCheck : Prop
  topoCheck : Prop
  tameCheck : Prop
  ergoCheck : Prop
  complexCheck : Prop
  oscillateCheck : Prop
  boundaryCheck : Prop
  overloadCheck : Prop
  starveCheck : Prop
  alignCheck : Prop
  lockCheck : Prop
  germPackage : Prop
  initialityPackage : Prop
  catLibPackage : Prop

def InterfaceChecklist.coreNodesReady (C : InterfaceChecklist) : Prop :=
  C.energyCheck ∧ C.zenoCheck ∧ C.compactCheck ∧ C.scaleCheck ∧ C.paramCheck ∧
    C.geomCheck ∧ C.stiffnessCheck ∧ C.topoCheck ∧ C.tameCheck ∧
    C.ergoCheck ∧ C.complexCheck ∧ C.oscillateCheck

def InterfaceChecklist.lockReady (C : InterfaceChecklist) : Prop :=
  C.lockCheck ∧ C.germPackage ∧ C.initialityPackage ∧ C.catLibPackage

structure LockMechanismSection where
  lockPackage : LockCertificatePackage
  primaryTactic : String
  attemptedTactics : List String
  preservationLemmasNeeded : List String

structure FinalVerdictSection where
  status : String
  designatedGoal : String
  goalConeEmpty : Prop
  singularitySetDescription : String

structure StructuralSieveProofObject where
  metadata : ProofMetadata
  labels : LabelNamingConventions
  automation : AutomationWitnessSection
  abstract : AbstractSection
  theoremStatement : TheoremStatementSection
  thinObjects : ThinObjectSection
  checklist : InterfaceChecklist
  trace : ExecutionTrace
  lockMechanism : LockMechanismSection
  upgrades : List UpgradeRule
  obligationLedger : ObligationLedger
  finalChain : FinalCertificateChain
  validity : RunValidity
  finalVerdict : FinalVerdictSection

def StructuralSieveProofObject.templateAligned
    (P : StructuralSieveProofObject) : Prop :=
  P.checklist.coreNodesReady ∧
    P.checklist.lockReady ∧
    P.validity.meetsTemplateCompletionCriteria ∧
    P.finalChain.designatedGoal ∈ P.finalChain.certificates ∧
    P.finalVerdict.goalConeEmpty

theorem StructuralSieveProofObject.goalRecorded
    (P : StructuralSieveProofObject) :
    P.finalVerdict.designatedGoal = P.finalChain.designatedGoal → P.finalChain.designatedGoal ∈ P.finalChain.certificates := by
  intro _h
  exact P.finalChain.containsGoal

theorem StructuralSieveProofObject.templateAligned_of_fields
    (P : StructuralSieveProofObject)
    (hcore : P.checklist.coreNodesReady)
    (hlock : P.checklist.lockReady)
    (hvalid : P.validity.meetsTemplateCompletionCriteria)
    (hgoal : P.finalVerdict.goalConeEmpty) :
    P.templateAligned := by
  exact ⟨hcore, hlock, hvalid, P.finalChain.containsGoal, hgoal⟩

end Hypostructure.Framework
