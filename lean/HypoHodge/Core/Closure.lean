import HypoHodge.Core.Rule

namespace HypoHodge.Core

def closureN (rules : RuleSet) (n : ℕ) (Γ : Context) : Context :=
  Nat.iterate (step rules) n Γ

def closure (rules : RuleSet) (Γ : Context) : Context :=
  closureN rules allTags.card Γ

private theorem subset_allTags (Γ : Context) : Γ ⊆ allTags := by
  intro k hk
  exact mem_allTags k

private theorem card_le_allTags (Γ : Context) : Γ.card ≤ allTags.card := by
  exact Finset.card_le_card (subset_allTags Γ)

private theorem iterate_card_lt_of_ne
    (f : Context → Context)
    (hext : ∀ Γ, Γ ⊆ f Γ)
    (Γ : Context)
    (n : ℕ)
    (hne : Nat.iterate f n Γ ≠ Nat.iterate f (n + 1) Γ) :
    (Nat.iterate f n Γ).card < (Nat.iterate f (n + 1) Γ).card := by
  have hsubset : Nat.iterate f n Γ ⊆ f (Nat.iterate f n Γ) := hext _
  have hlt : Nat.iterate f n Γ < f (Nat.iterate f n Γ) := by
    apply lt_of_le_of_ne hsubset
    simpa [Function.iterate_succ_apply] using hne
  simpa [Function.iterate_succ_apply] using Finset.card_lt_card hlt

private theorem iterate_card_lower_bound
    (f : Context → Context)
    (hext : ∀ Γ, Γ ⊆ f Γ)
    (Γ : Context) :
    ∀ n,
      (∀ m < n, Nat.iterate f m Γ ≠ Nat.iterate f (m + 1) Γ) →
      n ≤ (Nat.iterate f n Γ).card := by
  intro n
  induction n with
  | zero =>
      intro _
      simp
  | succ n ih =>
      intro hneq
      have hprev : ∀ m < n, Nat.iterate f m Γ ≠ Nat.iterate f (m + 1) Γ := by
        intro m hm
        exact hneq m (lt_trans hm (Nat.lt_succ_self n))
      have hih : n ≤ (Nat.iterate f n Γ).card := ih hprev
      have hlt :
          (Nat.iterate f n Γ).card < (Nat.iterate f (n + 1) Γ).card :=
        iterate_card_lt_of_ne f hext Γ n (hneq n (Nat.lt_succ_self n))
      exact le_trans (Nat.succ_le_succ hih) (Nat.succ_le_of_lt hlt)

private theorem iterate_eq_propagates
    (f : Context → Context)
    {Γ : Context}
    {m : ℕ}
    (h : Nat.iterate f m Γ = Nat.iterate f (m + 1) Γ) :
    ∀ t, Nat.iterate f (m + t) Γ = Nat.iterate f (m + t + 1) Γ := by
  intro t
  induction t with
  | zero =>
      simpa using h
  | succ t ih =>
      have hcongr :
          f (Nat.iterate f (m + t) Γ) = f (Nat.iterate f (m + t + 1) Γ) :=
        congrArg f ih
      simpa [Function.iterate_succ_apply, Nat.add_assoc] using hcongr

private theorem iterate_fixed_at_card
    (f : Context → Context)
    (hext : ∀ Γ, Γ ⊆ f Γ)
    (Γ : Context) :
    f (Nat.iterate f allTags.card Γ) = Nat.iterate f allTags.card Γ := by
  let N := allTags.card
  by_contra hfix
  have hN :
      Nat.iterate f N Γ ≠ Nat.iterate f (N + 1) Γ := by
    intro heq
    apply hfix
    simpa [N, Function.iterate_succ_apply] using heq.symm
  have hneqAll :
      ∀ m < N + 1, Nat.iterate f m Γ ≠ Nat.iterate f (m + 1) Γ := by
    intro m hm
    by_contra hmeq
    have hprop := iterate_eq_propagates f hmeq (N - m)
    have hmle : m ≤ N := Nat.lt_succ_iff.mp hm
    have hNfix : Nat.iterate f N Γ = Nat.iterate f (N + 1) Γ := by
      simpa [N, Nat.sub_add_cancel hmle, Nat.add_assoc] using hprop
    exact hN hNfix
  have hlower : N + 1 ≤ (Nat.iterate f (N + 1) Γ).card :=
    iterate_card_lower_bound f hext Γ (N + 1) hneqAll
  have hupper : (Nat.iterate f (N + 1) Γ).card ≤ N := by
    simpa [N] using card_le_allTags (Nat.iterate f (N + 1) Γ)
  exact Nat.not_succ_le_self N (le_trans hlower hupper)

private theorem iterate_eq_self_of_fixed
    (f : Context → Context)
    {Γ : Context}
    (h : f Γ = Γ) :
    ∀ n, Nat.iterate f n Γ = Γ := by
  intro n
  induction n with
  | zero =>
      rfl
  | succ n ih =>
      simpa [Function.iterate_succ_apply, h, ih]

theorem subset_closureN (rules : RuleSet) (n : ℕ) (Γ : Context) :
    Γ ⊆ closureN rules n Γ

theorem monotone_closureN (rules : RuleSet) (n : ℕ) :
    Monotone (closureN rules n)

theorem subset_closureN (rules : RuleSet) (n : ℕ) (Γ : Context) :
    Γ ⊆ closureN rules n Γ := by
  induction n generalizing Γ with
  | zero =>
      intro k hk
      simpa [closureN] using hk
  | succ n ih =>
      exact subset_trans (ih Γ)
        (by
          simpa [closureN, Function.iterate_succ_apply] using
            subset_step rules (closureN rules n Γ))

theorem monotone_closureN (rules : RuleSet) (n : ℕ) :
    Monotone (closureN rules n) := by
  induction n with
  | zero =>
      intro Γ Δ hΓΔ
      simpa [closureN] using hΓΔ
  | succ n ih =>
      intro Γ Δ hΓΔ
      simpa [closureN, Function.iterate_succ_apply] using
        monotone_step rules (ih hΓΔ)

theorem subset_closure (rules : RuleSet) (Γ : Context) :
    Γ ⊆ closure rules Γ := by
  simpa [closure] using subset_closureN rules allTags.card Γ

theorem monotone_closure (rules : RuleSet) :
    Monotone (closure rules) := by
  simpa [closure] using monotone_closureN rules allTags.card

theorem closure_fixed (rules : RuleSet) (Γ : Context) :
    step rules (closure rules Γ) = closure rules Γ

theorem closure_idempotent (rules : RuleSet) (Γ : Context) :
    closure rules (closure rules Γ) = closure rules Γ

theorem closure_least_fixed
    (rules : RuleSet) (Γ Δ : Context)
    (hΓ : Γ ⊆ Δ)
    (hΔ : step rules Δ = Δ) :
    closure rules Γ ⊆ Δ

theorem closure_fixed (rules : RuleSet) (Γ : Context) :
    step rules (closure rules Γ) = closure rules Γ := by
  simpa [closure] using iterate_fixed_at_card (step rules) (subset_step rules) Γ

theorem closure_least_fixed
    (rules : RuleSet) (Γ Δ : Context)
    (hΓ : Γ ⊆ Δ)
    (hΔ : step rules Δ = Δ) :
    closure rules Γ ⊆ Δ := by
  have hmono : closureN rules allTags.card Γ ⊆ closureN rules allTags.card Δ :=
    monotone_closureN rules allTags.card hΓ
  have hfixed : closureN rules allTags.card Δ = Δ :=
    iterate_eq_self_of_fixed (step rules) hΔ allTags.card
  simpa [closure, hfixed] using hmono

theorem closure_idempotent (rules : RuleSet) (Γ : Context) :
    closure rules (closure rules Γ) = closure rules Γ := by
  apply Finset.Subset.antisymm
  · exact closure_least_fixed rules (closure rules Γ) (closure rules Γ)
      (by intro k hk; exact hk)
      (closure_fixed rules Γ)
  · exact subset_closure rules (closure rules Γ)

end HypoHodge.Core
