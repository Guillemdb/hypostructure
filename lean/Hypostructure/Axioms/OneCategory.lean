import Mathlib

open CategoryTheory

namespace Hypostructure.Axioms

structure OneCategoricalSieve
    (C : Type _)
    [Category C]
    (X : C) where
  arrows : ∀ ⦃Y : C⦄, (Y ⟶ X) → Prop
  downward_closed :
    ∀ ⦃Y Z : C⦄, (f : Y ⟶ X) → arrows f → (g : Z ⟶ Y) → arrows (g ≫ f)

class OneCategoryAxioms
    (C : Type _)
    [Category C] where
  hasFiniteCovers : Prop
  hasPullbackStableSieves : Prop
  hasFiniteLimits : Prop
  transportAlongIsomorphism : Prop

end Hypostructure.Axioms
