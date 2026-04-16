import Hypostructure.Core.Rule

namespace Hypostructure.Core

def closureN
    {κ : Type _}
    [DecidableEq κ] :
    RuleSet κ → ℕ → Context κ → Context κ
  | _, 0, Γ => Γ
  | rules, n + 1, Γ => step rules (closureN rules n Γ)

def closure
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (rules : RuleSet κ)
    (Γ : Context κ) :
    Context κ :=
  closureN rules (Fintype.card κ) Γ

private theorem subset_univ
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (Γ : Context κ) :
    Γ ⊆ (Finset.univ : Finset κ) := by
  intro k hk
  exact Finset.mem_univ k

private theorem card_le_univ
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (Γ : Context κ) :
    Γ.card ≤ Fintype.card κ := by
  simpa using Finset.card_le_card (subset_univ Γ)

theorem subset_closureN
    {κ : Type _}
    [DecidableEq κ]
    (rules : RuleSet κ)
    (n : ℕ)
    (Γ : Context κ) :
    Γ ⊆ closureN rules n Γ := by
  induction n generalizing Γ with
  | zero =>
      intro k hk
      simpa [closureN] using hk
  | succ n ih =>
      exact subset_trans (ih Γ) (subset_step rules (closureN rules n Γ))

theorem monotone_closureN
    {κ : Type _}
    [DecidableEq κ]
    (rules : RuleSet κ)
    (n : ℕ) :
    Monotone (closureN rules n) := by
  induction n with
  | zero =>
      intro Γ Δ hΓΔ
      simpa [closureN] using hΓΔ
  | succ n ih =>
      intro Γ Δ hΓΔ
      exact monotone_step rules (ih hΓΔ)

theorem subset_closure
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (rules : RuleSet κ)
    (Γ : Context κ) :
    Γ ⊆ closure rules Γ := by
  simpa [closure] using subset_closureN rules (Fintype.card κ) Γ

theorem monotone_closure
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (rules : RuleSet κ) :
    Monotone (closure rules) := by
  simpa [closure] using monotone_closureN rules (Fintype.card κ)

private theorem closureN_card_strict_of_ne
    {κ : Type _}
    [DecidableEq κ]
    (rules : RuleSet κ)
    (n : ℕ)
    (Γ : Context κ)
    (hne : closureN rules n Γ ≠ closureN rules (n + 1) Γ) :
    (closureN rules n Γ).card < (closureN rules (n + 1) Γ).card := by
  have hsubset : closureN rules n Γ ⊆ closureN rules (n + 1) Γ :=
    subset_step rules (closureN rules n Γ)
  exact Finset.card_lt_card (lt_of_le_of_ne hsubset hne)

private theorem closureN_strict_chain_if_not_fixed
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (rules : RuleSet κ)
    (Γ : Context κ)
    (hnot : step rules (closure rules Γ) ≠ closure rules Γ) :
    ∀ m ≤ Fintype.card κ, closureN rules m Γ ≠ closureN rules (m + 1) Γ := by
  intro m hm
  intro hEq
  have hFixAtM : ∀ t, closureN rules (m + t) Γ = closureN rules (m + t + 1) Γ := by
    intro t
    induction t with
    | zero =>
        simpa using hEq
    | succ t ih =>
        simpa [closureN, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using congrArg (step rules) ih
  have hTop : closureN rules (Fintype.card κ) Γ = closureN rules (Fintype.card κ + 1) Γ := by
    have hm' : m + (Fintype.card κ - m) = Fintype.card κ := Nat.add_sub_of_le hm
    simpa [hm', Nat.add_assoc] using hFixAtM (Fintype.card κ - m)
  exact hnot (by simpa [closure] using hTop.symm)

private theorem card_lower_bound_of_strict_chain
    {κ : Type _}
    [DecidableEq κ]
    (rules : RuleSet κ)
    (Γ : Context κ) :
    ∀ n,
      (∀ m < n, closureN rules m Γ ≠ closureN rules (m + 1) Γ) →
      n ≤ (closureN rules n Γ).card := by
  intro n
  induction n with
  | zero =>
      intro h
      simp
  | succ n ih =>
      intro h
      have hprev : ∀ m < n, closureN rules m Γ ≠ closureN rules (m + 1) Γ := by
        intro m hm
        exact h m (lt_trans hm (Nat.lt_succ_self n))
      have hih : n ≤ (closureN rules n Γ).card := ih hprev
      have hlt :
          (closureN rules n Γ).card < (closureN rules (n + 1) Γ).card :=
        closureN_card_strict_of_ne rules n Γ (h n (Nat.lt_succ_self n))
      exact le_trans (Nat.succ_le_succ hih) (Nat.succ_le_of_lt hlt)

theorem closure_fixed
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (rules : RuleSet κ)
    (Γ : Context κ) :
    step rules (closure rules Γ) = closure rules Γ := by
  by_contra hnot
  have hneq : ∀ m < Fintype.card κ + 1, closureN rules m Γ ≠ closureN rules (m + 1) Γ := by
    intro m hm
    exact closureN_strict_chain_if_not_fixed rules Γ hnot m (Nat.lt_succ_iff.mp hm)
  have hlower : Fintype.card κ + 1 ≤ (closureN rules (Fintype.card κ + 1) Γ).card :=
    card_lower_bound_of_strict_chain rules Γ (Fintype.card κ + 1) hneq
  have hupper : (closureN rules (Fintype.card κ + 1) Γ).card ≤ Fintype.card κ := by
    simpa using card_le_univ (closureN rules (Fintype.card κ + 1) Γ)
  exact Nat.not_succ_le_self (Fintype.card κ) (le_trans hlower hupper)

theorem step_subset_closure
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (rules : RuleSet κ)
    (Γ : Context κ) :
    step rules Γ ⊆ closure rules Γ := by
  have hstep : step rules Γ ⊆ step rules (closure rules Γ) :=
    monotone_step rules (subset_closure rules Γ)
  simpa [closure_fixed rules Γ] using hstep

theorem closure_least_fixed
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (rules : RuleSet κ)
    (Γ Δ : Context κ)
    (hΓ : Γ ⊆ Δ)
    (hΔ : step rules Δ = Δ) :
    closure rules Γ ⊆ Δ := by
  have hstay : ∀ n, closureN rules n Δ = Δ := by
    intro n
    induction n with
    | zero =>
        rfl
    | succ n ih =>
        simpa [closureN, ih, hΔ]
  have hmono := monotone_closureN rules (Fintype.card κ) hΓ
  simpa [closure, hstay (Fintype.card κ)] using hmono

theorem closure_idempotent
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (rules : RuleSet κ)
    (Γ : Context κ) :
    closure rules (closure rules Γ) = closure rules Γ := by
  apply Finset.Subset.antisymm
  · exact closure_least_fixed rules (closure rules Γ) (closure rules Γ)
      (by intro k hk; exact hk)
      (closure_fixed rules Γ)
  · exact subset_closure rules (closure rules Γ)

end Hypostructure.Core
