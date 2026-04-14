import HypoHodge.Algebraic.BadGerm

namespace HypoHodge.Algebraic
open Classical

noncomputable section

structure GermCode (n p : ℕ) where
  rankCode   : Fin (n + 1)
  witnessTag : Nat
  badTag     : Bool
  minTag     : Bool
  deriving DecidableEq, Repr

noncomputable def encodeBadAlgGerm (A : BadAlgGerm n p) : GermCode n p where
  rankCode := ⟨A.rankBound, Nat.lt_succ_of_le A.h_rank⟩
  witnessTag := A.witness.tag
  badTag := decide A.bad
  minTag := decide A.minimal

axiom encodeBadAlgGerm_injective (n p : ℕ) :
    Function.Injective (encodeBadAlgGerm : BadAlgGerm n p → GermCode n p)

instance instEncodableGermCode : Encodable (GermCode n p) := by
  infer_instance

noncomputable instance instEncodableBadAlgGerm : Encodable (BadAlgGerm n p) :=
  Encodable.ofInjective encodeBadAlgGerm (encodeBadAlgGerm_injective n p)

theorem boundedGermSmallness (n p : ℕ) :
    Encodable (BadAlgGerm n p) := by
  infer_instance

end

end HypoHodge.Algebraic
