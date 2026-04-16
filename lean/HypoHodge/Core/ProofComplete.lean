import HypoHodge.Core.ObligationLedger
import Hypostructure.Core.ProofComplete

namespace HypoHodge.Core

def ProofComplete (rules deps : RuleSet) (Γ : Context) (goal : CertTag) : Prop :=
  Hypostructure.Core.ProofComplete CertTag.isInc rules deps Γ goal

theorem proofComplete_iff
    (rules deps : RuleSet) (Γ : Context) (goal : CertTag) :
    ProofComplete rules deps Γ goal ↔
      goal ∈ Γ ∧ Disjoint (obligations Γ) (goalCone deps goal) := by
  simpa [ProofComplete, obligations] using
    (Hypostructure.Core.proofComplete_iff CertTag.isInc rules deps Γ goal)

theorem proofComplete_of_goal_mem_and_disjoint
    (rules deps : RuleSet) (Γ : Context) (goal : CertTag)
    (hgoal : goal ∈ Γ)
    (hdisj : Disjoint (obligations Γ) (goalCone deps goal)) :
    ProofComplete rules deps Γ goal := by
  exact Hypostructure.Core.proofComplete_of_goal_mem_and_disjoint
    CertTag.isInc rules deps Γ goal hgoal hdisj

theorem not_proofComplete_of_goal_missing
    (rules deps : RuleSet) (Γ : Context) (goal : CertTag)
    (hgoal : goal ∉ Γ) :
    ¬ ProofComplete rules deps Γ goal := by
  exact Hypostructure.Core.not_proofComplete_of_goal_missing
    CertTag.isInc rules deps Γ goal hgoal

theorem not_proofComplete_of_inc_in_goalCone
    (rules deps : RuleSet) (Γ : Context) (goal k : CertTag)
    (hinc : k ∈ obligations Γ)
    (hcone : k ∈ goalCone deps goal) :
    ¬ ProofComplete rules deps Γ goal := by
  exact Hypostructure.Core.not_proofComplete_of_inc_in_goalCone
    CertTag.isInc rules deps Γ goal k hinc hcone

end HypoHodge.Core
