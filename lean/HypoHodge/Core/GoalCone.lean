import HypoHodge.Core.Closure

namespace HypoHodge.Core

def backStep (deps : RuleSet) (S : Context) : Context :=
  deps.foldl
    (fun acc r => if r.conclusion ∈ acc then acc ∪ r.premises else acc)
    S

def goalConeN (deps : RuleSet) (n : ℕ) (goal : CertTag) : Context :=
  Nat.iterate (backStep deps) n ({goal} : Finset CertTag)

def goalCone (deps : RuleSet) (goal : CertTag) : Context :=
  goalConeN deps allTags.card goal

private theorem subset_allTags (Γ : Context) : Γ ⊆ allTags := by
  intro k hk
  exact mem_allTags k

private theorem card_le_allTags (Γ : Context) : Γ.card ≤ allTags.card := by
  exact Finset.card_le_card (subset_allTags Γ)

private theorem subset_backStep (deps : RuleSet) (S : Context) :
    S ⊆ backStep deps S := by
  induction deps generalizing S with
  | nil =>
      intro k hk
      simpa [backStep] using hk
  | cons r rs ih =>
      have hseed :
          S ⊆ (if r.conclusion ∈ S then S ∪ r.premises else S) := by
        by_cases h : r.conclusion ∈ S
        · intro k hk
          simp [h, hk]
        · intro k hk
          simp [h, hk]
      exact subset_trans hseed
        (by
          simpa [backStep] using
            ih (if r.conclusion ∈ S then S ∪ r.premises else S))

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

private theorem subset_iterate_of_extensive
    (f : Context → Context)
    (hext : ∀ Γ, Γ ⊆ f Γ)
    (n : ℕ)
    (Γ : Context) :
    Γ ⊆ Nat.iterate f n Γ := by
  induction n generalizing Γ with
  | zero =>
      intro k hk
      simpa using hk
  | succ n ih =>
      exact subset_trans (ih Γ)
        (by
          simpa [Function.iterate_succ_apply] using hext (Nat.iterate f n Γ))

private theorem premises_subset_backStep_of_conclusion_mem
    (deps : RuleSet)
    (S : Context)
    (r : Rule)
    (hr : r ∈ deps)
    (hconc : r.conclusion ∈ S) :
    r.premises ⊆ backStep deps S := by
  induction deps generalizing S with
  | nil =>
      cases hr
  | cons r' rs ih =>
      simp at hr
      cases hr with
      | inl hEq =>
          subst hEq
          have hseed :
              r.premises ⊆ (if r.conclusion ∈ S then S ∪ r.premises else S) := by
            intro k hk
            simp [hconc, hk]
          exact subset_trans hseed
            (by
              simpa [backStep] using
                subset_backStep rs (if r.conclusion ∈ S then S ∪ r.premises else S))
      | inr hrs =>
          have hseed :
              S ⊆ (if r'.conclusion ∈ S then S ∪ r'.premises else S) := by
            by_cases h : r'.conclusion ∈ S
            · intro k hk
              simp [h, hk]
            · intro k hk
              simp [h, hk]
          have hconc' :
              r.conclusion ∈ (if r'.conclusion ∈ S then S ∪ r'.premises else S) :=
            hseed hconc
          simpa [backStep] using
            ih (if r'.conclusion ∈ S then S ∪ r'.premises else S) hrs hconc'

theorem goal_mem_goalCone (deps : RuleSet) (goal : CertTag) :
    goal ∈ goalCone deps goal

theorem backStep_monotone (deps : RuleSet) :
    Monotone (backStep deps)

theorem goalConeN_monotone (deps : RuleSet) (n : ℕ) :
    Monotone (fun S : Context => Nat.iterate (backStep deps) n S)

theorem goalCone_fixed (deps : RuleSet) (goal : CertTag) :
    backStep deps (goalCone deps goal) = goalCone deps goal

theorem premises_in_goalCone_of_conclusion
    (deps : RuleSet) (goal : CertTag) (r : Rule)
    (hr : r ∈ deps)
    (hconc : r.conclusion ∈ goalCone deps goal) :
    r.premises ⊆ goalCone deps goal

theorem goal_mem_goalCone (deps : RuleSet) (goal : CertTag) :
    goal ∈ goalCone deps goal := by
  have hsub :
      ({goal} : Finset CertTag) ⊆ goalCone deps goal := by
    simpa [goalCone, goalConeN] using
      subset_iterate_of_extensive (backStep deps) (subset_backStep deps) allTags.card
        ({goal} : Finset CertTag)
  exact hsub (by simp)

theorem backStep_monotone (deps : RuleSet) :
    Monotone (backStep deps) := by
  induction deps with
  | nil =>
      intro S T hST
      simpa [backStep] using hST
  | cons r rs ih =>
      intro S T hST
      have hacc :
          (if r.conclusion ∈ S then S ∪ r.premises else S) ⊆
          (if r.conclusion ∈ T then T ∪ r.premises else T) := by
        by_cases hS : r.conclusion ∈ S
        · have hT : r.conclusion ∈ T := hST hS
          intro k hk
          simp [hS, hT] at hk ⊢
          exact hk.elim (fun hkS => Or.inl (hST hkS)) Or.inr
        · by_cases hT : r.conclusion ∈ T
          · intro k hk
            simp [hS, hT] at hk ⊢
            exact Or.inl (hST hk)
          · intro k hk
            simp [hS, hT] at hk ⊢
            exact hST hk
      simpa [backStep] using ih hacc

theorem goalConeN_monotone (deps : RuleSet) (n : ℕ) :
    Monotone (fun S : Context => Nat.iterate (backStep deps) n S) := by
  induction n with
  | zero =>
      intro S T hST
      simpa using hST
  | succ n ih =>
      intro S T hST
      simpa [Function.iterate_succ_apply] using
        backStep_monotone deps (ih hST)

theorem goalCone_fixed (deps : RuleSet) (goal : CertTag) :
    backStep deps (goalCone deps goal) = goalCone deps goal := by
  simpa [goalCone, goalConeN] using
    iterate_fixed_at_card (backStep deps) (subset_backStep deps) ({goal} : Finset CertTag)

theorem premises_in_goalCone_of_conclusion
    (deps : RuleSet) (goal : CertTag) (r : Rule)
    (hr : r ∈ deps)
    (hconc : r.conclusion ∈ goalCone deps goal) :
    r.premises ⊆ goalCone deps goal := by
  have hsub :
      r.premises ⊆ backStep deps (goalCone deps goal) :=
    premises_subset_backStep_of_conclusion_mem deps (goalCone deps goal) r hr hconc
  simpa [goalCone_fixed deps goal] using hsub

end HypoHodge.Core
