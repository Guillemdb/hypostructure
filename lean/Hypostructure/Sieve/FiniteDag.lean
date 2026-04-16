import Hypostructure.Core.Closure

namespace Hypostructure.Sieve

open Hypostructure.Core

structure FiniteSieve
    (κ : Type _)
    [DecidableEq κ]
    [Fintype κ] where
  rules : RuleSet κ
  rank : κ → ℕ
  acyclic :
    ∀ ⦃r : Rule κ⦄, r ∈ rules → ∀ ⦃k : κ⦄, k ∈ r.premises → rank k < rank r.conclusion

def edgeRelation
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ) :
    κ → κ → Prop :=
  fun a b => ∃ r ∈ S.rules, a ∈ r.premises ∧ r.conclusion = b

noncomputable def incoming
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    (b : κ) :
    Finset κ := by
  classical
  exact Finset.univ.filter (fun a => decide (edgeRelation S a b))

theorem incoming_spec
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    (a b : κ) :
    a ∈ incoming S b ↔ edgeRelation S a b := by
  classical
  simp [incoming, edgeRelation]

theorem edge_rank_lt
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    {a b : κ}
    (h : edgeRelation S a b) :
    S.rank a < S.rank b := by
  rcases h with ⟨r, hr, ha, rfl⟩
  exact S.acyclic hr ha

def dagStep
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    (Γ : Context κ) :
    Context κ :=
  step S.rules Γ

def maxRank
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ) :
    Nat :=
  (Finset.univ : Finset κ).sup S.rank

def rulesAtRank
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    (n : Nat) :
    RuleSet κ :=
  S.rules.filter (fun r => S.rank r.conclusion = n)

def executeRank
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    (n : Nat)
    (Γ : Context κ) :
    Context κ :=
  step (rulesAtRank S n) Γ

def executeByRankN
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ) :
    Nat → Context κ → Context κ
  | 0, Γ => Γ
  | n + 1, Γ => executeRank S n (executeByRankN S n Γ)

def runDag
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    (Γ : Context κ) :
    Context κ :=
  executeByRankN S (maxRank S + 1) Γ

theorem subset_executeRank
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    (n : Nat)
    (Γ : Context κ) :
    Γ ⊆ executeRank S n Γ := by
  simpa [executeRank] using subset_step (rulesAtRank S n) Γ

theorem subset_executeByRankN
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    (n : Nat)
    (Γ : Context κ) :
    Γ ⊆ executeByRankN S n Γ := by
  induction n generalizing Γ with
  | zero =>
      intro k hk
      simpa [executeByRankN] using hk
  | succ n ih =>
      exact subset_trans (ih Γ) (subset_executeRank S n (executeByRankN S n Γ))

theorem subset_runDag
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    (Γ : Context κ) :
    Γ ⊆ runDag S Γ := by
  simpa [runDag] using subset_executeByRankN S (maxRank S + 1) Γ

def truncatedRun
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    (Γ : Context κ) :
    Context κ :=
  closure S.rules Γ

theorem dagStep_subset_truncatedRun
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    (Γ : Context κ) :
    dagStep S Γ ⊆ truncatedRun S Γ := by
  simpa [dagStep, truncatedRun] using step_subset_closure S.rules Γ

theorem truncatedRun_eq_closure
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    (Γ : Context κ) :
    truncatedRun S Γ = closure S.rules Γ := by
  rfl

theorem runDag_contains_truncatedRun_seed
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    (Γ : Context κ) :
    Γ ⊆ runDag S Γ := by
  exact subset_runDag S Γ

theorem closure_fixed_on_truncatedRun
    {κ : Type _}
    [DecidableEq κ]
    [Fintype κ]
    (S : FiniteSieve κ)
    (Γ : Context κ) :
    step S.rules (truncatedRun S Γ) = truncatedRun S Γ := by
  simpa [truncatedRun] using closure_fixed S.rules Γ

end Hypostructure.Sieve
