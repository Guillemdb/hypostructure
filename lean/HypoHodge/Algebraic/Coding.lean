import HypoHodge.Algebraic.BadGerm

namespace HypoHodge.Algebraic
open Classical

noncomputable section

structure GermCode (n p : ℕ) where
  rankCode   : Fin (n + 1)
  witnessTag : Nat
  nonzeroTag : Bool
  badTag     : Bool
  minTag     : Bool
  deriving DecidableEq, Repr

noncomputable def encodeBadAlgGerm (A : BadAlgGerm n p) : GermCode n p where
  rankCode := ⟨A.rankBound, Nat.lt_succ_of_le A.h_rank⟩
  witnessTag := A.witness.tag
  nonzeroTag := decide A.witness.nonzero
  badTag := decide A.bad
  minTag := decide A.minimal

private theorem prop_eq_of_decide_eq {P Q : Prop} [Decidable P] [Decidable Q]
    (h : decide P = decide Q) :
    P = Q := by
  apply propext
  by_cases hP : P <;> by_cases hQ : Q <;> simp [hP, hQ] at h ⊢

theorem encodeBadAlgGerm_injective (n p : ℕ) :
    Function.Injective (encodeBadAlgGerm : BadAlgGerm n p → GermCode n p) := by
  intro A B h
  cases A with
  | mk rankA hRankA witnessA badA minA =>
      cases B with
      | mk rankB hRankB witnessB badB minB =>
          cases witnessA with
          | mk tagA nonzeroA =>
              cases witnessB with
              | mk tagB nonzeroB =>
                  have hRank : rankA = rankB := by
                    have h' := congrArg (fun c => c.rankCode.val) h
                    simpa [encodeBadAlgGerm] using h'
                  have hTag : tagA = tagB := by
                    have h' := congrArg GermCode.witnessTag h
                    simpa [encodeBadAlgGerm] using h'
                  have hNonzero : nonzeroA = nonzeroB := by
                    have h' := congrArg GermCode.nonzeroTag h
                    exact prop_eq_of_decide_eq (by simpa [encodeBadAlgGerm] using h')
                  have hBad : badA = badB := by
                    have h' := congrArg GermCode.badTag h
                    exact prop_eq_of_decide_eq (by simpa [encodeBadAlgGerm] using h')
                  have hMin : minA = minB := by
                    have h' := congrArg GermCode.minTag h
                    exact prop_eq_of_decide_eq (by simpa [encodeBadAlgGerm] using h')
                  subst hRank
                  subst hTag
                  subst hNonzero
                  subst hBad
                  subst hMin
                  have hProof : hRankA = hRankB := Subsingleton.elim _ _
                  subst hProof
                  rfl

instance instEncodableGermCode : Encodable (GermCode n p) := by
  infer_instance

noncomputable instance instEncodableBadAlgGerm : Encodable (BadAlgGerm n p) :=
  Encodable.ofInjective encodeBadAlgGerm (encodeBadAlgGerm_injective n p)

theorem boundedGermSmallness (n p : ℕ) :
    Encodable (BadAlgGerm n p) := by
  infer_instance

end

end HypoHodge.Algebraic
