import Mathlib

namespace Hypostructure.Core

abbrev Context (κ : Type _) := Finset κ

theorem context_ext
    {κ : Type _}
    [DecidableEq κ]
    (Γ Δ : Context κ)
    (h : ∀ k, k ∈ Γ ↔ k ∈ Δ) :
    Γ = Δ := by
  exact Finset.ext h

end Hypostructure.Core
