import Hypostructure.Problem.SingularityClassification
import Hypostructure.Sieve.GenericProgram

namespace Hypostructure.Framework

open Hypostructure.Problem
open Hypostructure.Sieve

universe u v

inductive BackendType
  | parabolic
  | dispersive
  | metricGF
  | Markov
  | algorithmic
  deriving DecidableEq, Repr, Fintype

inductive GateNode
  | energyCheck
  | zenoCheck
  | compactCheck
  | scaleCheck
  | paramCheck
  | geomCheck
  | stiffnessCheck
  | bifurcateCheck
  | symCheck
  | checkSSB
  | checkTB
  | topoCheck
  | tameCheck
  | ergoCheck
  | complexCheck
  | oscillateCheck
  | boundaryCheck
  | overloadCheck
  | starveCheck
  | alignCheck
  | lock
  deriving DecidableEq, Repr, Fintype

inductive BarrierNode
  | barrierSat
  | barrierCausal
  | barrierScat
  | barrierTypeII
  | barrierVac
  | barrierCap
  | barrierGap
  | barrierAction
  | barrierOmin
  | barrierMix
  | barrierEpi
  | barrierFreq
  | barrierBode
  | barrierInput
  | barrierVariety
  | barrierExclusion
  deriving DecidableEq, Repr, Fintype

inductive ModeId
  | cE
  | cC
  | cD
  | sE
  | sC
  | sD
  | tE
  | tC
  | tD
  | dC
  | dE
  | bE
  | bD
  | bC
  deriving DecidableEq, Repr, Fintype

inductive SurgeryId
  | surgCE
  | surgCC
  | surgCDAlt
  | surgSE
  | surgSC
  | surgCD
  | surgSD
  | surgSCRest
  | surgTERest
  | surgTE
  | surgTC
  | surgTD
  | surgDC
  | surgDE
  | surgBE
  | surgBD
  | surgBC
  deriving DecidableEq, Repr, Fintype

structure UserProblemData where
  State : Type u
  Symmetry : Type v
  height : State → ℝ
  dissipation : State → ℝ
  recovery : Option (State → ℝ) := none
  capacity : Option (Set State → ℝ) := none
  sectorLabel : Option (State → ℕ) := none
  dictionary : Option (State → Finset ℕ) := none
  backend : BackendType

structure WitnessFailureCertificate (State : Type u) where
  node : GateNode
  predicate : State → Prop
  witness : State
  valid : predicate witness

structure InconclusiveCertificate where
  node : GateNode
  missing : Finset NodeTag
  trace : List GateNode

inductive TypedNoCertificate (State : Type u) (Profile : Type v)
  | witness (C : WitnessFailureCertificate State)
  | inconclusive (C : InconclusiveCertificate)
  | profileWild (W : WildProfileWitness Profile)
  | profileInconclusive (O : InconclusiveProfileWitness Profile)

structure SingularityInterfaces (State : Type u) (Profile : Type v) where
  extract : State → Profile
  locus : State → Set State

inductive ProfileResolution (Profile : Type v)
  | library (L : FiniteProfileLibrary Profile) (profile : Profile)
  | stratified (S : TameProfileStratification Profile) (profile : Profile)
  | wild (W : WildProfileWitness Profile)
  | inconclusive (O : InconclusiveProfileWitness Profile)

noncomputable def resolveProfile
    {State : Type u}
    {Profile : Type v}
    [HasProfileClassification Profile]
    (I : SingularityInterfaces State Profile)
    (x : State) :
    ProfileResolution Profile :=
  match profileClassification Profile with
  | .library L => .library L (I.extract x)
  | .stratified S => .stratified S (I.extract x)
  | .wild W => .wild W
  | .inconclusive O => .inconclusive O

theorem resolveProfile_exhaustive
    {State : Type u}
    {Profile : Type v}
    [HasProfileClassification Profile]
    (I : SingularityInterfaces State Profile)
    (x : State) :
    (∃ L : FiniteProfileLibrary Profile, resolveProfile I x = .library L (I.extract x)) ∨
      (∃ S : TameProfileStratification Profile,
        resolveProfile I x = .stratified S (I.extract x)) ∨
      (∃ W : WildProfileWitness Profile, resolveProfile I x = .wild W) ∨
      ∃ O : InconclusiveProfileWitness Profile, resolveProfile I x = .inconclusive O := by
  cases h : profileClassification Profile with
  | library L =>
      exact Or.inl ⟨L, by simp [resolveProfile, h]⟩
  | stratified S =>
      exact Or.inr <| Or.inl ⟨S, by simp [resolveProfile, h]⟩
  | wild W =>
      exact Or.inr <| Or.inr <| Or.inl ⟨W, by simp [resolveProfile, h]⟩
  | inconclusive O =>
      exact Or.inr <| Or.inr <| Or.inr ⟨O, by simp [resolveProfile, h]⟩

structure SurgeryData (State : Type u) (Profile : Type v) where
  locus : Set State
  profile : Profile

def surgeryDataOf
    {State : Type u}
    {Profile : Type v}
    (I : SingularityInterfaces State Profile)
    (x : State) :
    SurgeryData State Profile :=
  ⟨I.locus x, I.extract x⟩

class HasSemanticSingularity (State : Type u) where
  singular : State → Prop

def singularityFree
    {State : Type u}
    [HasSemanticSingularity State]
    (x : State) :
    Prop :=
  ¬ HasSemanticSingularity.singular x

inductive ProgressKind
  | boundedResource
  | wellFoundedDecrease
  deriving DecidableEq, Repr, Fintype

structure ProgressCertificate (State : Type u) where
  kind : ProgressKind
  measure : State → ℕ
  before : State
  after : State
  progress :
    match kind with
    | .boundedResource => measure after ≤ measure before
    | .wellFoundedDecrease => measure after < measure before

inductive SurgeryAdmissibilityOutcome (State : Type u) (Profile : Type v)
  | admissible (data : SurgeryData State Profile)
  | admissibleEq (data : SurgeryData State Profile) (equivalenceMove : State → State → Prop)
  | horizon (data : SurgeryData State Profile) (certificate : InconclusiveCertificate)

class HasSurgeryAdmissibility (State : Type u) (Profile : Type v) where
  admissibility : SurgeryData State Profile → SurgeryAdmissibilityOutcome State Profile

structure SurgeryStep (State : Type u) where
  id : SurgeryId
  targetMode : ModeId
  before : State
  after : State
  reentryTarget : GateNode
  progress : ProgressCertificate State

structure SurgeryOperator (State : Type u) (Profile : Type v) where
  onAdmissible : SurgeryData State Profile → SurgeryStep State
  onAdmissibleEq :
    (data : SurgeryData State Profile) →
      (equivalenceMove : State → State → Prop) →
      SurgeryStep State

inductive AutomaticSurgeryOutcome (State : Type u)
  | performed (step : SurgeryStep State)
  | blocked (certificate : InconclusiveCertificate)

def automaticSurgery
    {State : Type u}
    {Profile : Type v}
    [HasSurgeryAdmissibility State Profile]
    (op : SurgeryOperator State Profile)
    (data : SurgeryData State Profile) :
    AutomaticSurgeryOutcome State :=
  match HasSurgeryAdmissibility.admissibility data with
  | .admissible d => .performed (op.onAdmissible d)
  | .admissibleEq d e => .performed (op.onAdmissibleEq d e)
  | .horizon _ c => .blocked c

theorem automaticSurgery_exhaustive
    {State : Type u}
    {Profile : Type v}
    [HasSurgeryAdmissibility State Profile]
    (op : SurgeryOperator State Profile)
    (data : SurgeryData State Profile) :
    (∃ step : SurgeryStep State, automaticSurgery op data = .performed step) ∨
      ∃ cert : InconclusiveCertificate, automaticSurgery op data = .blocked cert := by
  cases h : HasSurgeryAdmissibility.admissibility data with
  | admissible d =>
      exact Or.inl ⟨op.onAdmissible d, by simp [automaticSurgery, h]⟩
  | admissibleEq d e =>
      exact Or.inl ⟨op.onAdmissibleEq d e, by simp [automaticSurgery, h]⟩
  | horizon d cert =>
      exact Or.inr ⟨cert, by simp [automaticSurgery, h]⟩

structure RoutingPolicy where
  onWitness : GateNode → ModeId
  onProfileWild : ModeId
  onProfileInconclusive : ModeId
  onAdmissibilityHorizon : ModeId

inductive KernelTerminalVerdict
    (State : Type u)
    (Profile : Type v)
    [HasSemanticSingularity State]
    (x : State)
  | victory
      (goalDetected : NodeTag.goal ∈ generatedCertificates)
      (regular : singularityFree x)
  | mode
      (route : ModeId)
      (certificate : TypedNoCertificate State Profile)
  | surgery
      (step : SurgeryStep State)

structure CompiledSieve
    (State : Type u)
    (Profile : Type v)
    [HasProfileClassification Profile]
    [HasSemanticSingularity State]
    [HasSurgeryAdmissibility State Profile] where
  backend : BackendType
  interfaces : SingularityInterfaces State Profile
  routing : RoutingPolicy
  surgery : SurgeryOperator State Profile

noncomputable def terminalVerdict
    {State : Type u}
    {Profile : Type v}
    [HasProfileClassification Profile]
    [HasSemanticSingularity State]
    [HasSurgeryAdmissibility State Profile]
    (C : CompiledSieve State Profile)
    (x : State) :
    KernelTerminalVerdict State Profile x := by
  by_cases hs : HasSemanticSingularity.singular x
  · cases hprof : resolveProfile C.interfaces x with
    | library L p =>
        let data : SurgeryData State Profile := ⟨C.interfaces.locus x, p⟩
        cases hadm : HasSurgeryAdmissibility.admissibility data with
        | admissible d =>
            exact .surgery (C.surgery.onAdmissible d)
        | admissibleEq d e =>
            exact .surgery (C.surgery.onAdmissibleEq d e)
        | horizon d cert =>
            exact .mode C.routing.onAdmissibilityHorizon (.inconclusive cert)
    | stratified S p =>
        let data : SurgeryData State Profile := ⟨C.interfaces.locus x, p⟩
        cases hadm : HasSurgeryAdmissibility.admissibility data with
        | admissible d =>
            exact .surgery (C.surgery.onAdmissible d)
        | admissibleEq d e =>
            exact .surgery (C.surgery.onAdmissibleEq d e)
        | horizon d cert =>
            exact .mode C.routing.onAdmissibilityHorizon (.inconclusive cert)
    | wild W =>
        exact .mode C.routing.onProfileWild (.profileWild W)
    | inconclusive O =>
        exact .mode C.routing.onProfileInconclusive (.profileInconclusive O)
  · exact .victory goal_mem_generatedCertificates hs

theorem terminalVerdict_exhaustive
    {State : Type u}
    {Profile : Type v}
    [HasProfileClassification Profile]
    [HasSemanticSingularity State]
    [HasSurgeryAdmissibility State Profile]
    (C : CompiledSieve State Profile)
    (x : State) :
    (∃ hgoal hreg, terminalVerdict C x = .victory hgoal hreg) ∨
      (∃ m cert, terminalVerdict C x = .mode m cert) ∨
      ∃ step, terminalVerdict C x = .surgery step := by
  cases h : terminalVerdict C x with
  | victory hgoal hreg =>
      exact Or.inl ⟨hgoal, hreg, rfl⟩
  | mode m cert =>
      exact Or.inr <| Or.inl ⟨m, cert, rfl⟩
  | surgery step =>
      exact Or.inr <| Or.inr ⟨step, rfl⟩

structure AutomationGuarantee
    (State : Type u)
    (Profile : Type v)
    [HasProfileClassification Profile]
    [HasSemanticSingularity State]
    [HasSurgeryAdmissibility State Profile] where
  compiled : CompiledSieve State Profile
  profileResolutionAutomatic :
    ∀ x : State, Nonempty (ProfileResolution Profile)
  terminalRoutingAutomatic :
    ∀ x : State, Nonempty (KernelTerminalVerdict State Profile x)

def compiledSieveAutomation
    {State : Type u}
    {Profile : Type v}
    [HasProfileClassification Profile]
    [HasSemanticSingularity State]
    [HasSurgeryAdmissibility State Profile]
    (C : CompiledSieve State Profile) :
    AutomationGuarantee State Profile where
  compiled := C
  profileResolutionAutomatic := fun x => ⟨resolveProfile C.interfaces x⟩
  terminalRoutingAutomatic := fun x => ⟨terminalVerdict C x⟩

structure GateEvaluatorFactory (D : UserProblemData) where
  supports : GateNode → Prop

structure BarrierFactory (D : UserProblemData) where
  supports : BarrierNode → Prop

structure SurgerySchemaFactory (D : UserProblemData) where
  supports : SurgeryId → Prop

structure EquivalenceTransportFactory (D : UserProblemData) where
  transports : GateNode → GateNode → Prop

structure LockBackendFactory (D : UserProblemData) where
  supportsReconstruction : Prop
  supportsExclusion : Prop

structure FactoryBundle (D : UserProblemData) where
  gates : GateEvaluatorFactory D
  barriers : BarrierFactory D
  surgeries : SurgerySchemaFactory D
  transports : EquivalenceTransportFactory D
  lockBackend : LockBackendFactory D

theorem factInstantiation_output_trichotomy
    {State : Type u}
    {Profile : Type v}
    [HasProfileClassification Profile]
    [HasSemanticSingularity State]
    [HasSurgeryAdmissibility State Profile]
    (C : CompiledSieve State Profile)
    (x : State) :
    (∃ hgoal hreg, terminalVerdict C x = .victory hgoal hreg) ∨
      (∃ m cert, terminalVerdict C x = .mode m cert) ∨
      ∃ step, terminalVerdict C x = .surgery step :=
  terminalVerdict_exhaustive C x

theorem factInstantiation_goal_certificate
    {State : Type u}
    {Profile : Type v}
    [HasProfileClassification Profile]
    [HasSemanticSingularity State]
    [HasSurgeryAdmissibility State Profile]
    (C : CompiledSieve State Profile)
    {x : State}
    (h : singularityFree x) :
    ∃ hgoal, terminalVerdict C x = .victory hgoal h := by
  have hs : ¬ HasSemanticSingularity.singular x := h
  refine ⟨goal_mem_generatedCertificates, ?_⟩
  simp [terminalVerdict, hs, singularityFree] 

noncomputable def UserProblemData.ofThinInput (I : ThinInput) (backend : BackendType) : UserProblemData where
  State := I.stateSpace
  Symmetry := I.symmetryGroup
  height := I.potential
  dissipation := I.dissipation
  recovery := some fun x => ‖x‖
  capacity := some fun s => by
    classical
    exact if s.Nonempty then 1 else 0
  sectorLabel := some fun _ => 0
  dictionary := some fun _ => {}
  backend := backend
