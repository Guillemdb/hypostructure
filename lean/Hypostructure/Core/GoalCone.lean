import Hypostructure.Core.Closure

namespace Hypostructure.Core

def backStep
    {κ : Type _}
    [DecidableEq κ]
    (deps : RuleSet κ)
    (S : Context κ) :
    Context κ :=
  deps.foldl
    (fun acc r => if r.conclusion ∈ acc then acc ∪ r.premises else acc)
    S

def goalExpandN
    {κ : Type _}
    [DecidableEq κ] :
    RuleSet κ → ℕ → Context κ → Context κ
  | deps, 0, S => S
  | deps, n + 1, S => backStep deps (goalExpandN deps n S)

def goalConeN
    {κ : Type _}
    [DecidableEq κ]
    (deps : RuleSet κ)
    (n : ℕ)
    (goal : κ) :
    Context κ :=
  goalExpandN deps n ({goal} : Finset κ)

def goalCone
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (deps : RuleSet κ)
    (goal : κ) :
    Context κ :=
  goalConeN deps (Fintype.card κ) goal

private theorem card_le_univ
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (Γ : Context κ) :
    Γ.card ≤ Fintype.card κ := by
  simpa using Finset.card_le_card (by intro k hk; simp)

private theorem subset_backStep
    {κ : Type _}
    [DecidableEq κ]
    (deps : RuleSet κ)
    (S : Context κ) :
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

theorem backStep_monotone
    {κ : Type _}
    [DecidableEq κ]
    (deps : RuleSet κ) :
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

theorem goalConeN_monotone
    {κ : Type _}
    [DecidableEq κ]
    (deps : RuleSet κ)
    (n : ℕ) :
    Monotone (goalExpandN deps n) := by
  induction n with
  | zero =>
      intro S T hST
      simpa [goalExpandN] using hST
  | succ n ih =>
      intro S T hST
      exact backStep_monotone deps (ih hST)

private theorem subset_goalExpandN
    {κ : Type _}
    [DecidableEq κ]
    (deps : RuleSet κ)
    (n : ℕ)
    (S : Context κ) :
    S ⊆ goalExpandN deps n S := by
  induction n generalizing S with
  | zero =>
      intro k hk
      simpa [goalExpandN] using hk
  | succ n ih =>
      exact subset_trans (ih S) (subset_backStep deps (goalExpandN deps n S))

private theorem goalExpandN_card_strict_of_ne
    {κ : Type _}
    [DecidableEq κ]
    (deps : RuleSet κ)
    (n : ℕ)
    (S : Context κ)
    (hne : goalExpandN deps n S ≠ goalExpandN deps (n + 1) S) :
    (goalExpandN deps n S).card < (goalExpandN deps (n + 1) S).card := by
  have hsubset : goalExpandN deps n S ⊆ goalExpandN deps (n + 1) S :=
    subset_backStep deps (goalExpandN deps n S)
  exact Finset.card_lt_card (lt_of_le_of_ne hsubset hne)

private theorem goalExpandN_stable_from
    {κ : Type _}
    [DecidableEq κ]
    (deps : RuleSet κ)
    (S : Context κ)
    {m : ℕ}
    (h : goalExpandN deps m S = goalExpandN deps (m + 1) S) :
    ∀ t, goalExpandN deps (m + t) S = goalExpandN deps (m + t + 1) S := by
  intro t
  induction t with
  | zero =>
      simpa using h
  | succ t ih =>
      simpa [goalExpandN, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using
        congrArg (backStep deps) ih

private theorem strict_chain_card_lower_bound
    {κ : Type _}
    [DecidableEq κ]
    (deps : RuleSet κ)
    (S : Context κ) :
    ∀ n,
      (∀ m < n, goalExpandN deps m S ≠ goalExpandN deps (m + 1) S) →
      n ≤ (goalExpandN deps n S).card := by
  intro n
  induction n with
  | zero =>
      intro h
      simp
  | succ n ih =>
      intro h
      have hprev : ∀ m < n, goalExpandN deps m S ≠ goalExpandN deps (m + 1) S := by
        intro m hm
        exact h m (lt_trans hm (Nat.lt_succ_self n))
      have hih : n ≤ (goalExpandN deps n S).card := ih hprev
      have hlt :
          (goalExpandN deps n S).card < (goalExpandN deps (n + 1) S).card :=
        goalExpandN_card_strict_of_ne deps n S (h n (Nat.lt_succ_self n))
      exact le_trans (Nat.succ_le_succ hih) (Nat.succ_le_of_lt hlt)

theorem goal_mem_goalCone
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (deps : RuleSet κ)
    (goal : κ) :
    goal ∈ goalCone deps goal := by
  have hsub : ({goal} : Finset κ) ⊆ goalCone deps goal := by
    simpa [goalCone, goalConeN] using subset_goalExpandN deps (Fintype.card κ) ({goal} : Finset κ)
  exact hsub (by simp)

theorem goalCone_fixed
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (deps : RuleSet κ)
    (goal : κ) :
    backStep deps (goalCone deps goal) = goalCone deps goal := by
  let S : Context κ := ({goal} : Finset κ)
  by_contra hnot
  have hneq : ∀ m < Fintype.card κ + 1, goalExpandN deps m S ≠ goalExpandN deps (m + 1) S := by
    intro m hm
    intro hEq
    have hprop := goalExpandN_stable_from deps S hEq (Fintype.card κ - m)
    have hmle : m ≤ Fintype.card κ := Nat.lt_succ_iff.mp hm
    have hTop : goalExpandN deps (Fintype.card κ) S = goalExpandN deps (Fintype.card κ + 1) S := by
      simpa [Nat.add_assoc, Nat.add_sub_of_le hmle] using hprop
    exact hnot (by simpa [goalCone, goalConeN, S] using hTop.symm)
  have hlower : Fintype.card κ + 1 ≤ (goalExpandN deps (Fintype.card κ + 1) S).card :=
    strict_chain_card_lower_bound deps S (Fintype.card κ + 1) hneq
  have hupper : (goalExpandN deps (Fintype.card κ + 1) S).card ≤ Fintype.card κ := by
    simpa using card_le_univ (goalExpandN deps (Fintype.card κ + 1) S)
  exact Nat.not_succ_le_self (Fintype.card κ) (le_trans hlower hupper)

private theorem premises_subset_backStep_of_conclusion_mem
    {κ : Type _}
    [DecidableEq κ]
    (deps : RuleSet κ)
    (S : Context κ)
    (r : Rule κ)
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

theorem premises_in_goalCone_of_conclusion
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (deps : RuleSet κ)
    (goal : κ)
    (r : Rule κ)
    (hr : r ∈ deps)
    (hconc : r.conclusion ∈ goalCone deps goal) :
    r.premises ⊆ goalCone deps goal := by
  have hsub :
      r.premises ⊆ backStep deps (goalCone deps goal) :=
    premises_subset_backStep_of_conclusion_mem deps (goalCone deps goal) r hr hconc
  simpa [goalCone_fixed deps goal] using hsub

end Hypostructure.Core
