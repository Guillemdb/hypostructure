import Hypostructure.Problem.Signature
import Hypostructure.Problem.Thin
import Hypostructure.Problem.Backend
import Hypostructure.Sieve.FiniteDag

namespace Hypostructure.Sieve

open Hypostructure.Core
open Hypostructure.Problem

def backendBase : Context NodeTag :=
  ({ NodeTag.ambient, NodeTag.repConservative, NodeTag.repComplete, NodeTag.admissible,
     NodeTag.germ, NodeTag.reduction, NodeTag.library, NodeTag.gamma } : Finset NodeTag)

def backendRules : RuleSet NodeTag :=
  [ { kind := RuleKind.backend, premises := ({ NodeTag.anchor } : Finset NodeTag), conclusion := NodeTag.ambient }
  , { kind := RuleKind.backend, premises := ({ NodeTag.anchor } : Finset NodeTag), conclusion := NodeTag.repConservative }
  , { kind := RuleKind.backend, premises := ({ NodeTag.anchor } : Finset NodeTag), conclusion := NodeTag.repComplete }
  , { kind := RuleKind.backend,
      premises := ({ NodeTag.anchor, NodeTag.defect, NodeTag.recurrence, NodeTag.multiplicity,
                     NodeTag.tame, NodeTag.partialCompact, NodeTag.capacity, NodeTag.stability,
                     NodeTag.boundaryEven, NodeTag.boundaryPolarized, NodeTag.boundedPartial } : Finset NodeTag),
      conclusion := NodeTag.admissible }
  , { kind := RuleKind.backend, premises := ({ NodeTag.anchor } : Finset NodeTag), conclusion := NodeTag.germ }
  , { kind := RuleKind.backend, premises := ({ NodeTag.germ } : Finset NodeTag), conclusion := NodeTag.reduction }
  , { kind := RuleKind.backend, premises := ({ NodeTag.reduction } : Finset NodeTag), conclusion := NodeTag.library }
  , { kind := RuleKind.backend, premises := ({ NodeTag.anchor } : Finset NodeTag), conclusion := NodeTag.gamma }
  ]

def permitPrimary : Rule NodeTag :=
  { kind := RuleKind.bridge
    premises := ({ NodeTag.ambient, NodeTag.boundaryEven, NodeTag.tame, NodeTag.defect } : Finset NodeTag)
    conclusion := NodeTag.primaryBridge }

def permitSecondary : Rule NodeTag :=
  { kind := RuleKind.bridge
    premises := ({ NodeTag.admissible, NodeTag.gamma, NodeTag.library } : Finset NodeTag)
    conclusion := NodeTag.secondaryBridge }

def permitGoal : Rule NodeTag :=
  { kind := RuleKind.bridge
    premises := ({ NodeTag.primaryBridge, NodeTag.secondaryBridge, NodeTag.multiplicity, NodeTag.stability,
                   NodeTag.boundaryPolarized, NodeTag.library, NodeTag.reduction, NodeTag.admissible } : Finset NodeTag)
    conclusion := NodeTag.goal }

def bridgeRules : RuleSet NodeTag := [permitPrimary, permitSecondary, permitGoal]
def promotionRules : RuleSet NodeTag := []
def allRules : RuleSet NodeTag := backendRules ++ bridgeRules ++ promotionRules
def deps : RuleSet NodeTag := allRules

def thinKernel : ProblemInstance NodeTag where
  isPending := NodeTag.isPending
  goal := NodeTag.goal
  seed := seedContext
  rules := allRules
  deps := deps

def thinSieve : FiniteSieve NodeTag where
  rules := allRules
  rank
    | .anchor => 0
    | .defect => 0
    | .recurrence => 0
    | .multiplicity => 0
    | .tame => 0
    | .partialCompact => 0
    | .capacity => 0
    | .stability => 0
    | .boundaryEven => 0
    | .boundaryPolarized => 0
    | .boundedPartial => 0
    | .ambient => 1
    | .repConservative => 1
    | .repComplete => 1
    | .admissible => 1
    | .germ => 1
    | .reduction => 2
    | .library => 3
    | .gamma => 1
    | .primaryBridge => 4
    | .secondaryBridge => 4
    | .goal => 5
    | .goalInc => 6
    | .promoInc => 6
  acyclic := by
    intro r hr k hk
    simp [allRules, backendRules, bridgeRules, promotionRules] at hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · simp at hk
      subst k
      decide
    · simp at hk
      subst k
      decide
    · simp at hk
      subst k
      decide
    · simp at hk
      rcases hk with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide
    · simp at hk
      subst k
      decide
    · simp at hk
      subst k
      decide
    · simp at hk
      subst k
      decide
    · simp at hk
      subst k
      decide
    · simp [permitPrimary] at hk
      rcases hk with rfl | rfl | rfl | rfl <;> decide
    · simp [permitSecondary] at hk
      rcases hk with rfl | rfl | rfl <;> decide
    · simp [permitGoal] at hk
      rcases hk with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

def expectedAfterRank0 : Context NodeTag :=
  seedContext

def afterRank0 : Context NodeTag :=
  executeRank thinSieve 0 seedContext

def expectedAfterRank1 : Context NodeTag :=
  ({ NodeTag.anchor, NodeTag.defect, NodeTag.recurrence, NodeTag.multiplicity, NodeTag.tame,
     NodeTag.partialCompact, NodeTag.capacity, NodeTag.stability,
     NodeTag.boundaryEven, NodeTag.boundaryPolarized, NodeTag.boundedPartial,
     NodeTag.ambient, NodeTag.repConservative, NodeTag.repComplete,
     NodeTag.admissible, NodeTag.germ, NodeTag.gamma } : Finset NodeTag)

def afterRank1 : Context NodeTag :=
  executeRank thinSieve 1 afterRank0

def expectedAfterRank2 : Context NodeTag :=
  insert NodeTag.reduction expectedAfterRank1

def afterRank2 : Context NodeTag :=
  executeRank thinSieve 2 afterRank1

def expectedAfterRank3 : Context NodeTag :=
  insert NodeTag.library expectedAfterRank2

def afterRank3 : Context NodeTag :=
  executeRank thinSieve 3 afterRank2

def expectedAfterRank4 : Context NodeTag :=
  insert NodeTag.secondaryBridge (insert NodeTag.primaryBridge expectedAfterRank3)

def afterRank4 : Context NodeTag :=
  executeRank thinSieve 4 afterRank3

def expectedAfterRank5 : Context NodeTag :=
  insert NodeTag.goal expectedAfterRank4

def afterRank5 : Context NodeTag :=
  executeRank thinSieve 5 afterRank4

def generatedCertificates : Context NodeTag :=
  executeRank thinSieve 6 afterRank5

def productiveNodes : Context NodeTag :=
  ({ NodeTag.anchor, NodeTag.defect, NodeTag.recurrence, NodeTag.multiplicity, NodeTag.tame,
     NodeTag.partialCompact, NodeTag.capacity, NodeTag.stability,
     NodeTag.boundaryEven, NodeTag.boundaryPolarized, NodeTag.boundedPartial,
     NodeTag.ambient, NodeTag.repConservative, NodeTag.repComplete,
     NodeTag.admissible, NodeTag.germ, NodeTag.reduction, NodeTag.library, NodeTag.gamma,
     NodeTag.primaryBridge, NodeTag.secondaryBridge, NodeTag.goal } : Finset NodeTag)

theorem afterRank0_eq_expected :
    afterRank0 = expectedAfterRank0 := by
  native_decide

theorem afterRank1_eq_expected :
    afterRank1 = expectedAfterRank1 := by
  native_decide

theorem afterRank2_eq_expected :
    afterRank2 = expectedAfterRank2 := by
  native_decide

theorem afterRank3_eq_expected :
    afterRank3 = expectedAfterRank3 := by
  native_decide

theorem afterRank4_eq_expected :
    afterRank4 = expectedAfterRank4 := by
  native_decide

theorem afterRank5_eq_expected :
    afterRank5 = expectedAfterRank5 := by
  native_decide

theorem generatedCertificates_eq_afterRank5 :
    generatedCertificates = afterRank5 := by
  native_decide

theorem generatedCertificates_eq_productiveNodes :
    generatedCertificates = productiveNodes := by
  native_decide

theorem mem_generatedCertificates_iff
    (k : NodeTag) :
    k ∈ generatedCertificates ↔ k ≠ NodeTag.goalInc ∧ k ≠ NodeTag.promoInc := by
  cases k <;> native_decide

theorem generatedCertificates_eq_runDag :
    generatedCertificates = runDag thinSieve seedContext := by
  native_decide

theorem generatedCertificates_match_kernel :
    generatedCertificates = runKernel thinKernel := by
  native_decide

theorem goal_mem_generatedCertificates :
    NodeTag.goal ∈ generatedCertificates := by
  native_decide

theorem germ_mem_generatedCertificates :
    NodeTag.germ ∈ generatedCertificates := by
  simpa [mem_generatedCertificates_iff]

theorem reduction_mem_generatedCertificates :
    NodeTag.reduction ∈ generatedCertificates := by
  simpa [mem_generatedCertificates_iff]

theorem library_mem_generatedCertificates :
    NodeTag.library ∈ generatedCertificates := by
  simpa [mem_generatedCertificates_iff]

theorem gamma_mem_generatedCertificates :
    NodeTag.gamma ∈ generatedCertificates := by
  simpa [mem_generatedCertificates_iff]

theorem goalInc_not_mem_generatedCertificates :
    NodeTag.goalInc ∉ generatedCertificates := by
  native_decide

theorem promoInc_not_mem_generatedCertificates :
    NodeTag.promoInc ∉ generatedCertificates := by
  native_decide

theorem truncated_sieve_matches_kernel :
    truncatedRun thinSieve seedContext = runKernel thinKernel := by
  rfl

end Hypostructure.Sieve
