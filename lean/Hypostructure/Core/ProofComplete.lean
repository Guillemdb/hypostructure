import Hypostructure.Core.ObligationLedger

namespace Hypostructure.Core

def ProofComplete
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (isPending : κ → Bool)
    (rules deps : RuleSet κ)
    (Γ : Context κ)
    (goal : κ) : Prop :=
  goal ∈ Γ ∧ Disjoint (obligations isPending Γ) (goalCone deps goal)

theorem proofComplete_iff
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (isPending : κ → Bool)
    (rules deps : RuleSet κ)
    (Γ : Context κ)
    (goal : κ) :
    ProofComplete isPending rules deps Γ goal ↔
      goal ∈ Γ ∧ Disjoint (obligations isPending Γ) (goalCone deps goal) := by
  rfl

theorem proofComplete_of_goal_mem_and_disjoint
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (isPending : κ → Bool)
    (rules deps : RuleSet κ)
    (Γ : Context κ)
    (goal : κ)
    (hgoal : goal ∈ Γ)
    (hdisj : Disjoint (obligations isPending Γ) (goalCone deps goal)) :
    ProofComplete isPending rules deps Γ goal := by
  exact ⟨hgoal, hdisj⟩

theorem not_proofComplete_of_goal_missing
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (isPending : κ → Bool)
    (rules deps : RuleSet κ)
    (Γ : Context κ)
    (goal : κ)
    (hgoal : goal ∉ Γ) :
    ¬ ProofComplete isPending rules deps Γ goal := by
  intro h
  exact hgoal h.1

theorem not_proofComplete_of_inc_in_goalCone
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (isPending : κ → Bool)
    (rules deps : RuleSet κ)
    (Γ : Context κ)
    (goal k : κ)
    (hinc : k ∈ obligations isPending Γ)
    (hcone : k ∈ goalCone deps goal) :
    ¬ ProofComplete isPending rules deps Γ goal := by
  intro h
  exact Finset.disjoint_left.mp h.2 hinc hcone

end Hypostructure.Core
