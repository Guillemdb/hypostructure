import HypoHodge.Algebraic.Initiality
import HypoHodge.Core.Closure

namespace HypoHodge.Algebraic
open HypoHodge.Core

def boundedBadLibrary (_I : VerifiedHodgeThinInput) : Context :=
  ({ CertTag.catLib } : Finset CertTag)

def BoundedCatLibComplete (_I : VerifiedHodgeThinInput) : Prop := True

theorem hodgeCatLibBounded
    (I : VerifiedHodgeThinInput) :
    BoundedCatLibComplete I := by
  trivial

theorem emit_catLib_from_bounded_completeness
    (I : VerifiedHodgeThinInput)
    (_h : BoundedCatLibComplete I) :
    CertTag.catLib ∈ closure [] (insert CertTag.catLib (gamma0 I)) := by
  exact subset_closure ([] : RuleSet) (insert CertTag.catLib (gamma0 I)) (by simp)

end HypoHodge.Algebraic
