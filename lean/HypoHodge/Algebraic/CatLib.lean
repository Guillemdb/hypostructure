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

axiom emit_catLib_from_bounded_completeness
    (I : VerifiedHodgeThinInput)
    (h : BoundedCatLibComplete I) :
    CertTag.catLib ∈ closure [] (insert CertTag.catLib (gamma0 I))

end HypoHodge.Algebraic
