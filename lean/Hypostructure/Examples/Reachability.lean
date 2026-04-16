import Hypostructure.Problem.Signature
import Hypostructure.Sieve.FiniteDag

namespace Hypostructure.Examples

open Hypostructure.Core
open Hypostructure.Problem
open Hypostructure.Sieve

inductive ReachTag
  | seed
  | lifted
  | target
  | unresolved
  deriving DecidableEq, Repr, Fintype

def ReachTag.isPending : ReachTag → Bool
  | .unresolved => true
  | _ => false

def reachRules : RuleSet ReachTag :=
  [ { kind := RuleKind.backend
      premises := ({ ReachTag.seed } : Finset ReachTag)
      conclusion := ReachTag.lifted }
  , { kind := RuleKind.bridge
      premises := ({ ReachTag.lifted } : Finset ReachTag)
      conclusion := ReachTag.target } ]

def reachProblem : ProblemInstance ReachTag where
  isPending := ReachTag.isPending
  goal := ReachTag.target
  seed := ({ ReachTag.seed } : Finset ReachTag)
  rules := reachRules

def reachSieve : FiniteSieve ReachTag where
  rules := reachRules
  rank
    | .seed => 0
    | .lifted => 1
    | .target => 2
    | .unresolved => 3
  acyclic := by
    intro r hr k hk
    simp [reachRules] at hr
    rcases hr with rfl | rfl
    · simp at hk
      subst k
      decide
    · simp at hk
      subst k
      decide

theorem reach_goal_mem :
    ReachTag.target ∈ runKernel reachProblem := by
  simp [runKernel, reachProblem, reachRules, Hypostructure.Core.closure,
    Hypostructure.Core.closureN, Hypostructure.Core.step,
    Hypostructure.Core.fireRule, Hypostructure.Core.Rule.enabled]

theorem reach_problem_complete :
    kernelComplete reachProblem := by
  apply proofComplete_of_goal_mem_and_disjoint
    (isPending := ReachTag.isPending)
    (rules := reachProblem.rules)
    (deps := reachProblem.deps)
  · simpa [reachProblem] using reach_goal_mem
  · simp [runKernel, reachProblem, reachRules, Hypostructure.Core.closure,
      Hypostructure.Core.closureN, Hypostructure.Core.step,
      Hypostructure.Core.fireRule, Hypostructure.Core.Rule.enabled,
      Hypostructure.Core.obligations, ReachTag.isPending, Hypostructure.Core.goalCone,
      Hypostructure.Core.goalConeN, Hypostructure.Core.goalExpandN, Hypostructure.Core.backStep]

theorem reach_truncated_matches_closure :
    truncatedRun reachSieve reachProblem.seed = runKernel reachProblem := by
  rfl

end Hypostructure.Examples
