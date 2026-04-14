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

axiom monotone_fireRule (r : Rule) :
    Monotone (fireRule r)

axiom subset_step (rules : RuleSet) (Γ : Context) :
    Γ ⊆ step rules Γ

axiom monotone_step (rules : RuleSet) :
    Monotone (step rules)

end HypoHodge.Core
