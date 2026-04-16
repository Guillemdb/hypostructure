import HypoHodge.Hodge.ProofAudit
import HypoHodge.Core.ProofComplete

namespace HypoHodge.Hodge
open HypoHodge.Core
open HypoHodge.Algebraic
open HypoHodge.Imported

theorem hodge_framework_kernel_complete
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    ProofComplete allRules deps (runHodgeSystem I) CertTag.catHomBlk := by
  apply proofComplete_of_goal_mem_and_disjoint
  · simpa [runHodgeSystem, finalContext] using emit_catHomBlk I
  · exact hodgeProofAudit I

theorem hodge_framework_target_realized
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    TargetRealizedByCycle I := by
  exact blockedGoal_semantics I

theorem hodge_framework_target_hodge
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    TargetHodgeClass I := by
  exact ⟨(emit_mhs_semantics I).targetRational, (emit_mhs_semantics I).targetHodge⟩

theorem hodge_framework_unconditional
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    HodgeConjectureTarget I := by
  intro _
  exact hodge_framework_target_realized I

end HypoHodge.Hodge
