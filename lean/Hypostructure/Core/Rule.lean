import Hypostructure.Core.Context

namespace Hypostructure.Core

inductive RuleKind
  | backend
  | bridge
  | promotion
  | incUpgrade
  deriving DecidableEq

structure Rule (κ : Type _) where
  kind       : RuleKind
  premises   : Finset κ
  conclusion : κ

abbrev RuleSet (κ : Type _) := List (Rule κ)

def Rule.enabled
    {κ : Type _}
    [DecidableEq κ]
    (r : Rule κ)
    (Γ : Context κ) : Prop :=
  r.premises ⊆ Γ

instance instDecidableEnabled
    {κ : Type _}
    [DecidableEq κ]
    (r : Rule κ)
    (Γ : Context κ) :
    Decidable (r.enabled Γ) := by
  classical
  unfold Rule.enabled
  infer_instance

def fireRule
    {κ : Type _}
    [DecidableEq κ]
    (r : Rule κ)
    (Γ : Context κ) : Context κ :=
  if h : r.enabled Γ then insert r.conclusion Γ else Γ

def step
    {κ : Type _}
    [DecidableEq κ] :
    RuleSet κ → Context κ → Context κ
  | [], Γ => Γ
  | r :: rs, Γ => step rs (fireRule r Γ)

theorem enabled_iff_subset
    {κ : Type _}
    [DecidableEq κ]
    (r : Rule κ)
    (Γ : Context κ) :
    r.enabled Γ ↔ r.premises ⊆ Γ := by
  rfl

theorem fireRule_eq_insert_of_enabled
    {κ : Type _}
    [DecidableEq κ]
    (r : Rule κ)
    (Γ : Context κ)
    (h : r.enabled Γ) :
    fireRule r Γ = insert r.conclusion Γ := by
  simp [fireRule, h]

theorem fireRule_eq_self_of_disabled
    {κ : Type _}
    [DecidableEq κ]
    (r : Rule κ)
    (Γ : Context κ)
    (h : ¬ r.enabled Γ) :
    fireRule r Γ = Γ := by
  simp [fireRule, h]

theorem subset_fireRule
    {κ : Type _}
    [DecidableEq κ]
    (r : Rule κ)
    (Γ : Context κ) :
    Γ ⊆ fireRule r Γ := by
  intro k hk
  by_cases h : r.enabled Γ
  · simp [fireRule, h, hk]
  · simp [fireRule, h, hk]

theorem monotone_fireRule
    {κ : Type _}
    [DecidableEq κ]
    (r : Rule κ) :
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

theorem subset_step
    {κ : Type _}
    [DecidableEq κ]
    (rules : RuleSet κ)
    (Γ : Context κ) :
    Γ ⊆ step rules Γ := by
  induction rules generalizing Γ with
  | nil =>
      intro k hk
      simpa [step] using hk
  | cons r rs ih =>
      intro k hk
      simpa [step] using ih (fireRule r Γ) (subset_fireRule r Γ hk)

theorem monotone_step
    {κ : Type _}
    [DecidableEq κ]
    (rules : RuleSet κ) :
    Monotone (step rules) := by
  induction rules with
  | nil =>
      intro Γ Δ hΓΔ k hk
      simpa [step] using hΓΔ hk
  | cons r rs ih =>
      intro Γ Δ hΓΔ
      simpa [step] using ih (monotone_fireRule r hΓΔ)

theorem step_append
    {κ : Type _}
    [DecidableEq κ]
    (rules₁ rules₂ : RuleSet κ)
    (Γ : Context κ) :
    step (rules₁ ++ rules₂) Γ = step rules₂ (step rules₁ Γ) := by
  induction rules₁ generalizing Γ with
  | nil =>
      simp [step]
  | cons r rs ih =>
      simp [step, ih]

end Hypostructure.Core
