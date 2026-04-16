import HypoHodge.Algebraic.Initiality
import HypoHodge.Core.Closure

namespace HypoHodge.Algebraic
open HypoHodge.Core

def boundedBadLibrary (_I : VerifiedHodgeThinInput) : Context :=
  ({ CertTag.catLib } : Finset CertTag)

def BoundedCatLibComplete (I : VerifiedHodgeThinInput) : Prop :=
  CategoryLibraryCertificate I ∧ StaticClassifiable I ∧ HasBoundedUniversalBad I

theorem hodgeCatLibBounded
    (I : VerifiedHodgeThinInput) :
    BoundedCatLibComplete I := by
  exact ⟨categoryLibraryCertificate I, hodgeClassifiableStatic I, hodgeInitialityBounded I⟩

theorem emit_catLib_from_bounded_completeness
    (I : VerifiedHodgeThinInput)
    (h : BoundedCatLibComplete I) :
    CertTag.catLib ∈ closure [] (insert CertTag.catLib (gamma0 I)) := by
  have _ : CategoryLibraryCertificate I := h.1
  exact subset_closure ([] : RuleSet) (insert CertTag.catLib (gamma0 I)) (by simp)

end HypoHodge.Algebraic
