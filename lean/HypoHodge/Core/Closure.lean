import HypoHodge.Core.Rule

namespace HypoHodge.Core

def closureN (rules : RuleSet) (n : ℕ) (Γ : Context) : Context :=
  Nat.iterate (step rules) n Γ

def closure (rules : RuleSet) (Γ : Context) : Context :=
  closureN rules allTags.card Γ

axiom subset_closureN (rules : RuleSet) (n : ℕ) (Γ : Context) :
    Γ ⊆ closureN rules n Γ

axiom monotone_closureN (rules : RuleSet) (n : ℕ) :
    Monotone (closureN rules n)

theorem subset_closure (rules : RuleSet) (Γ : Context) :
    Γ ⊆ closure rules Γ := by
  simpa [closure] using subset_closureN rules allTags.card Γ

theorem monotone_closure (rules : RuleSet) :
    Monotone (closure rules) := by
  simpa [closure] using monotone_closureN rules allTags.card

axiom closure_fixed (rules : RuleSet) (Γ : Context) :
    step rules (closure rules Γ) = closure rules Γ

axiom closure_idempotent (rules : RuleSet) (Γ : Context) :
    closure rules (closure rules Γ) = closure rules Γ

axiom closure_least_fixed
    (rules : RuleSet) (Γ Δ : Context)
    (hΓ : Γ ⊆ Δ)
    (hΔ : step rules Δ = Δ) :
    closure rules Γ ⊆ Δ

end HypoHodge.Core
