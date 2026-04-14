import HypoHodge.Core.ObligationLedger

namespace HypoHodge.Core

def ProofComplete (rules deps : RuleSet) (Γ : Context) (goal : CertTag) : Prop :=
  goal ∈ Γ ∧ Disjoint (obligations Γ) (goalCone deps goal)

theorem proofComplete_iff
    (rules deps : RuleSet) (Γ : Context) (goal : CertTag) :
    ProofComplete rules deps Γ goal ↔
      goal ∈ Γ ∧ Disjoint (obligations Γ) (goalCone deps goal) := by
  rfl

theorem proofComplete_of_goal_mem_and_disjoint
    (rules deps : RuleSet) (Γ : Context) (goal : CertTag)
    (hgoal : goal ∈ Γ)
    (hdisj : Disjoint (obligations Γ) (goalCone deps goal)) :
    ProofComplete rules deps Γ goal := by
  exact ⟨hgoal, hdisj⟩

theorem not_proofComplete_of_goal_missing
    (rules deps : RuleSet) (Γ : Context) (goal : CertTag)
    (hgoal : goal ∉ Γ) :
    ¬ ProofComplete rules deps Γ goal := by
  intro h
  exact hgoal h.1

theorem not_proofComplete_of_inc_in_goalCone
    (rules deps : RuleSet) (Γ : Context) (goal k : CertTag)
    (hinc : k ∈ obligations Γ)
    (hcone : k ∈ goalCone deps goal) :
    ¬ ProofComplete rules deps Γ goal := by
  intro h
  exact Finset.disjoint_left.mp h.2 k hinc hcone

end HypoHodge.Core
