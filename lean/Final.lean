import HypoHodge.Hodge.ProofAudit
import HypoHodge.Core.ProofComplete

namespace HypoHodge.Hodge
open HypoHodge.Core
open HypoHodge.Algebraic
open HypoHodge.Imported

theorem hodge_framework_unconditional
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    ProofComplete allRules deps (runHodgeSystem I) CertTag.catHomBlk := by
  apply proofComplete_of_goal_mem_and_disjoint
  · simpa [runHodgeSystem, finalContext] using emit_catHomBlk I
  · exact hodgeProofAudit I

end HypoHodge.Hodge
