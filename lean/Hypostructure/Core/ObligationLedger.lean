import Hypostructure.Core.GoalCone

namespace Hypostructure.Core

def obligations
    {κ : Type _}
    [DecidableEq κ]
    (isPending : κ → Bool)
    (Γ : Context κ) :
    Context κ :=
  Γ.filter (fun k => isPending k = true)

theorem obligations_subset
    {κ : Type _}
    [DecidableEq κ]
    (isPending : κ → Bool)
    (Γ : Context κ) :
    obligations isPending Γ ⊆ Γ := by
  intro k hk
  exact (Finset.mem_filter.mp hk).1

theorem mem_obligations_iff
    {κ : Type _}
    [DecidableEq κ]
    (isPending : κ → Bool)
    (Γ : Context κ)
    (k : κ) :
    k ∈ obligations isPending Γ ↔ k ∈ Γ ∧ isPending k = true := by
  simp [obligations]

end Hypostructure.Core
