import HypoHodge.Hodge.Run

namespace HypoHodge.Hodge
open HypoHodge.Algebraic
open HypoHodge.Imported

def blockedGoal_semantics
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    TargetRealizedByCycle I :=
  emit_catHomBlk_semantics I

end HypoHodge.Hodge
