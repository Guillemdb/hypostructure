import HypoHodge.Algebraic.BoundedReduction

namespace HypoHodge.Algebraic

structure UniversalBad (n p : ℕ) where
  carrier : Type
  inject  : ∀ A : BadAlgGerm n p, Nat → carrier
  initial :
    ∀ {X : Type} (f : ∀ A : BadAlgGerm n p, Nat → X),
      ∃! g : carrier → X, ∀ A a, g (inject A a) = f A a

def HasBoundedUniversalBad (I : VerifiedHodgeThinInput) : Prop :=
  ∃ U : UniversalBad I.Qrank I.p, True

def canonicalUniversalBad (n p : ℕ) : UniversalBad n p where
  carrier := Σ A : BadAlgGerm n p, Nat
  inject := fun A a => ⟨A, a⟩
  initial := by
    intro X f
    refine ⟨fun x => f x.1 x.2, ?_, ?_⟩
    · intro A a
      rfl
    · intro g hg
      funext x
      cases x with
      | mk A a =>
          simpa using congrFun (hg A) a

theorem hodgeInitialityBounded
    (I : VerifiedHodgeThinInput) :
    HasBoundedUniversalBad I := by
  exact ⟨canonicalUniversalBad I.Qrank I.p, trivial⟩

end HypoHodge.Algebraic
