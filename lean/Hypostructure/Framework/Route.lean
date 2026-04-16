import Hypostructure.Framework.Execution

namespace Hypostructure.Framework

/-- A reusable, problem-independent description of a hypostructure proof route.
It records only framework data: the executed trace, the certificates that must
appear, the Lock package, the upgrade rules, the final chain, and the open
obligation ledger. It deliberately contains no PDE/problem-specific theorem. -/
structure TraceBackedRoute where
  trace : ExecutionTrace
  requiredTraceCertificates : List String
  boundaryCertificate : String
  lockCertificate : String
  preservationCertificates : List String
  analyticCertificates : List String
  upgrades : List UpgradeRule
  obligationLedger : ObligationLedger
  lockPackage : LockCertificatePackage
  finalChain : FinalCertificateChain

def TraceBackedRoute.requiredTraceNodesExecuted
    (R : TraceBackedRoute) : Prop :=
  ∀ cert ∈ R.requiredTraceCertificates, R.trace.contains cert

def TraceBackedRoute.boundaryHandled
    (R : TraceBackedRoute) : Prop :=
  R.trace.contains R.boundaryCertificate

def TraceBackedRoute.lockExecuted
    (R : TraceBackedRoute) : Prop :=
  R.trace.contains R.lockCertificate

def TraceBackedRoute.upgradesCompleted
    (R : TraceBackedRoute) : Prop :=
  ∀ rule ∈ R.upgrades, rule.nonCircular

def TraceBackedRoute.goalConeEmpty
    (R : TraceBackedRoute) : Prop :=
  ObligationLedger.goalConeEmpty R.obligationLedger

def TraceBackedRoute.designatedGoalReached
    (R : TraceBackedRoute) : Prop :=
  R.finalChain.designatedGoal ∈ R.finalChain.certificates

def TraceBackedRoute.lockCompletenessPresent
    (R : TraceBackedRoute) : Prop :=
  ∀ cert ∈ R.lockPackage.requiredCertificates,
    cert ∈ R.finalChain.certificates

def TraceBackedRoute.analyticPermitPresent
    (R : TraceBackedRoute) : Prop :=
  ∀ cert ∈ R.analyticCertificates,
    cert ∈ R.finalChain.certificates

def TraceBackedRoute.preservationLemmasPresent
    (R : TraceBackedRoute) : Prop :=
  ∀ cert ∈ R.preservationCertificates, R.trace.contains cert

def TraceBackedRoute.runValidity
    (R : TraceBackedRoute) : RunValidity where
  allCoreNodesExecuted := R.requiredTraceNodesExecuted
  boundaryHandled := R.boundaryHandled
  lockExecuted := R.lockExecuted
  upgradeCompleted := R.upgradesCompleted
  goalConeEmpty := R.goalConeEmpty
  designatedGoalReached := R.designatedGoalReached
  lockCompletenessPresent := R.lockCompletenessPresent
  analyticPermitPresent := R.analyticPermitPresent
  preservationLemmasPresent := R.preservationLemmasPresent

/-- The generic route verifier: once a backend proves the trace contains the
required certificates and the ledger has no open goal-cone obligations, the
framework constructs the standard `RunValidity` theorem. -/
theorem TraceBackedRoute.runValidity_holds
    (R : TraceBackedRoute)
    (hrequired : R.requiredTraceNodesExecuted)
    (hboundary : R.boundaryHandled)
    (hlock : R.lockExecuted)
    (hupgrades : R.upgradesCompleted)
    (hgoalCone : R.goalConeEmpty)
    (hdesignated : R.designatedGoalReached)
    (hlockComplete : R.lockCompletenessPresent)
    (hanalytic : R.analyticPermitPresent)
    (hpreservation : R.preservationLemmasPresent) :
    R.runValidity.meetsTemplateCompletionCriteria := by
  exact RunValidity.templateCriteria_intro
    hrequired hboundary hlock hupgrades hgoalCone hdesignated
    hlockComplete hanalytic hpreservation

theorem TraceBackedRoute.finalCertificateRecorded
    (R : TraceBackedRoute) :
    R.finalChain.designatedGoal ∈ R.finalChain.certificates :=
  R.finalChain.containsGoal

structure TraceBackedRouteProof
    (R : TraceBackedRoute) where
  requiredTraceNodesExecuted : R.requiredTraceNodesExecuted
  boundaryHandled : R.boundaryHandled
  lockExecuted : R.lockExecuted
  upgradesCompleted : R.upgradesCompleted
  goalConeEmpty : R.goalConeEmpty
  designatedGoalReached : R.designatedGoalReached
  lockCompletenessPresent : R.lockCompletenessPresent
  analyticPermitPresent : R.analyticPermitPresent
  preservationLemmasPresent : R.preservationLemmasPresent

def TraceBackedRouteProof.runValidity_holds
    {R : TraceBackedRoute}
    (P : TraceBackedRouteProof R) :
    R.runValidity.meetsTemplateCompletionCriteria :=
  R.runValidity_holds
    P.requiredTraceNodesExecuted
    P.boundaryHandled
    P.lockExecuted
    P.upgradesCompleted
    P.goalConeEmpty
    P.designatedGoalReached
    P.lockCompletenessPresent
    P.analyticPermitPresent
    P.preservationLemmasPresent

/-- Reusable final wrapper for any backend target. The target remains entirely
problem-specific; the certificate-recording and route-validity parts are pure
hypostructure framework obligations. -/
structure TraceBackedTargetClaim
    (R : TraceBackedRoute)
    (Target : Prop) where
  finalCertificateRecorded :
    R.finalChain.designatedGoal ∈ R.finalChain.certificates
  routeValidity : R.runValidity.meetsTemplateCompletionCriteria
  target : Target

def TraceBackedRouteProof.targetClaim
    {R : TraceBackedRoute}
    (P : TraceBackedRouteProof R)
    {Target : Prop}
    (hTarget : Target) :
    TraceBackedTargetClaim R Target where
  finalCertificateRecorded := R.finalCertificateRecorded
  routeValidity := P.runValidity_holds
  target := hTarget

end Hypostructure.Framework
