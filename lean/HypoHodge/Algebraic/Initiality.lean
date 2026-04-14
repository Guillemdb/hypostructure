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

axiom hodgeInitialityBounded
    (I : VerifiedHodgeThinInput) :
    HasBoundedUniversalBad I

end HypoHodge.Algebraic
