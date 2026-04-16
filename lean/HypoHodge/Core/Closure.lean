import HypoHodge.Core.Rule
import Hypostructure.Core.Closure

namespace HypoHodge.Core

def closureN (rules : RuleSet) : ℕ → Context → Context
  | n, Γ => Hypostructure.Core.closureN rules n Γ

def closure (rules : RuleSet) (Γ : Context) : Context :=
  Hypostructure.Core.closure rules Γ

theorem subset_closureN (rules : RuleSet) (n : ℕ) (Γ : Context) :
    Γ ⊆ closureN rules n Γ := by
  simpa [closureN] using Hypostructure.Core.subset_closureN rules n Γ

theorem monotone_closureN (rules : RuleSet) (n : ℕ) :
    Monotone (closureN rules n) := by
  simpa [closureN] using Hypostructure.Core.monotone_closureN rules n

theorem subset_closure (rules : RuleSet) (Γ : Context) :
    Γ ⊆ closure rules Γ := by
  simpa [closure] using Hypostructure.Core.subset_closure rules Γ

theorem monotone_closure (rules : RuleSet) :
    Monotone (closure rules) := by
  simpa [closure] using Hypostructure.Core.monotone_closure rules

theorem closure_fixed (rules : RuleSet) (Γ : Context) :
    step rules (closure rules Γ) = closure rules Γ := by
  simpa [closure] using Hypostructure.Core.closure_fixed rules Γ

theorem closure_least_fixed
    (rules : RuleSet) (Γ Δ : Context)
    (hΓ : Γ ⊆ Δ)
    (hΔ : step rules Δ = Δ) :
    closure rules Γ ⊆ Δ := by
  simpa [closure] using Hypostructure.Core.closure_least_fixed rules Γ Δ hΓ hΔ

theorem closure_idempotent (rules : RuleSet) (Γ : Context) :
    closure rules (closure rules Γ) = closure rules Γ := by
  simpa [closure] using Hypostructure.Core.closure_idempotent rules Γ

end HypoHodge.Core
