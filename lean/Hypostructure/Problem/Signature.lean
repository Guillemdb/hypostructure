import Hypostructure.Core.ProofComplete

namespace Hypostructure.Problem

open Hypostructure.Core

structure ProblemProfile (κ : Type _) where
  isPending : κ → Bool
  goal      : κ

structure ProblemInstance
    (κ : Type _)
    [DecidableEq κ]
    [Fintype κ]
    extends ProblemProfile κ where
  seed : Context κ
  rules : RuleSet κ
  deps : RuleSet κ := rules

def runKernel
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (I : ProblemInstance κ) :
    Context κ :=
  closure I.rules I.seed

def kernelComplete
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (I : ProblemInstance κ) :
    Prop :=
  ProofComplete I.isPending I.rules I.deps (runKernel I) I.goal

structure CertificatePayload (κ : Type _) where
  cert    : κ
  meaning : Prop

structure SemanticBridge
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (I : ProblemInstance κ)
    (State : Type _) where
  certMeaning : κ → State → Prop
  target      : State → Prop
  seedState   : State
  goalSound   : I.goal ∈ runKernel I → target seedState

end Hypostructure.Problem
