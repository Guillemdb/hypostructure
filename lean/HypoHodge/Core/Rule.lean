import HypoHodge.Core.Context

namespace HypoHodge.Core

inductive RuleKind
  | backend
  | bridge
  | promotion
  | incUpgrade
  deriving DecidableEq, Repr

structure Rule where
  kind       : RuleKind
  premises   : Finset CertTag
  conclusion : CertTag
  deriving Repr

abbrev RuleSet := List Rule

def Rule.enabled (r : Rule) (Γ : Context) : Prop :=
  r.premises ⊆ Γ

def fireRule (r : Rule) (Γ : Context) : Context :=
  if h : r.enabled Γ then insert r.conclusion Γ else Γ

def step : RuleSet → Context → Context
  | [], Γ => Γ
  | r :: rs, Γ => step rs (fireRule r Γ)

theorem enabled_iff_subset (r : Rule) (Γ : Context) :
    r.enabled Γ ↔ r.premises ⊆ Γ := by
  rfl

theorem fireRule_eq_insert_of_enabled
    (r : Rule) (Γ : Context)
    (h : r.enabled Γ) :
    fireRule r Γ = insert r.conclusion Γ := by
  simp [fireRule, h]

theorem fireRule_eq_self_of_disabled
    (r : Rule) (Γ : Context)
    (h : ¬ r.enabled Γ) :
    fireRule r Γ = Γ := by
  simp [fireRule, h]

theorem subset_fireRule (r : Rule) (Γ : Context) :
    Γ ⊆ fireRule r Γ := by
  intro k hk
  by_cases h : r.enabled Γ
  · simp [fireRule, h, hk]
  · simp [fireRule, h, hk]

theorem monotone_fireRule (r : Rule) :
    Monotone (fireRule r)

theorem subset_step (rules : RuleSet) (Γ : Context) :
    Γ ⊆ step rules Γ

theorem monotone_step (rules : RuleSet) :
    Monotone (step rules)

theorem monotone_fireRule (r : Rule) :
    Monotone (fireRule r) := by
  intro Γ Δ hΓΔ
  by_cases hΓ : r.enabled Γ
  · have hΔ : r.enabled Δ := by
      intro k hk
      exact hΓΔ (hΓ hk)
    intro k hk
    rw [fireRule_eq_insert_of_enabled _ _ hΓ] at hk
    rw [fireRule_eq_insert_of_enabled _ _ hΔ]
    simp at hk ⊢
    exact hk.elim Or.inl (fun hkΓ => Or.inr (hΓΔ hkΓ))
  · intro k hk
    rw [fireRule_eq_self_of_disabled _ _ hΓ] at hk
    exact subset_fireRule r Δ (hΓΔ hk)

theorem subset_step (rules : RuleSet) (Γ : Context) :
    Γ ⊆ step rules Γ := by
  induction rules generalizing Γ with
  | nil =>
      intro k hk
      simpa [step] using hk
  | cons r rs ih =>
      intro k hk
      simpa [step] using ih (fireRule r Γ) (subset_fireRule r Γ hk)

theorem monotone_step (rules : RuleSet) :
    Monotone (step rules) := by
  induction rules with
  | nil =>
      intro Γ Δ hΓΔ k hk
      simpa [step] using hΓΔ hk
  | cons r rs ih =>
      intro Γ Δ hΓΔ
      simpa [step] using ih (monotone_fireRule r hΓΔ)

end HypoHodge.Core
