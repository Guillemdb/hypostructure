import HypoHodge.Hodge.Permits
import HypoHodge.Algebraic.BackendAutoclose
import HypoHodge.Core.Closure

namespace HypoHodge.Hodge
open HypoHodge.Core
open HypoHodge.Algebraic
open HypoHodge.Imported

def initialContext (I : VerifiedHodgeThinInput) : Context :=
  gamma0 I

def contextAfterBackend (I : VerifiedHodgeThinInput) : Context :=
  closure backendRules (initialContext I)

def contextAfterMhs (I : VerifiedHodgeThinInput) : Context :=
  closure (backendRules ++ [permitHdg]) (initialContext I)

def contextAfterTann (I : VerifiedHodgeThinInput) : Context :=
  closure (backendRules ++ [permitHdg, permitTann]) (initialContext I)

def finalContext (I : VerifiedHodgeThinInput) : Context :=
  closure allRules (initialContext I)

def runHodgeSystem (I : VerifiedHodgeThinInput) : Context :=
  finalContext I

theorem emit_backend_context
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    backendBase I ⊆ contextAfterBackend I := by
  simpa [contextAfterBackend, initialContext] using hodgeBackendAutoclose I

theorem emit_mhs
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.mhs ∈ contextAfterMhs I := by
  have hstepToClosure :
      step (backendRules ++ [permitHdg]) (initialContext I) ⊆ contextAfterMhs I := by
    have hmono :
        step (backendRules ++ [permitHdg]) (initialContext I) ⊆
          step (backendRules ++ [permitHdg]) (closure (backendRules ++ [permitHdg]) (initialContext I)) :=
      monotone_step (backendRules ++ [permitHdg])
        (subset_closure (backendRules ++ [permitHdg]) (initialContext I))
    simpa [contextAfterMhs, closure_fixed (backendRules ++ [permitHdg]) (initialContext I)] using hmono
  have hmhs : CertTag.mhs ∈ step (backendRules ++ [permitHdg]) (initialContext I) := by
    simp [initialContext, backendRules, permitHdg, step, fireRule, Rule.enabled, gamma0]
  exact hstepToClosure hmhs

theorem emit_tann
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.tann ∈ contextAfterTann I := by
  have hstepToClosure :
      step (backendRules ++ [permitHdg, permitTann]) (initialContext I) ⊆ contextAfterTann I := by
    have hmono :
        step (backendRules ++ [permitHdg, permitTann]) (initialContext I) ⊆
          step (backendRules ++ [permitHdg, permitTann])
            (closure (backendRules ++ [permitHdg, permitTann]) (initialContext I)) :=
      monotone_step (backendRules ++ [permitHdg, permitTann])
        (subset_closure (backendRules ++ [permitHdg, permitTann]) (initialContext I))
    simpa [contextAfterTann, closure_fixed (backendRules ++ [permitHdg, permitTann]) (initialContext I)] using hmono
  have htann : CertTag.tann ∈ step (backendRules ++ [permitHdg, permitTann]) (initialContext I) := by
    simp [initialContext, backendRules, permitHdg, permitTann, step, fireRule, Rule.enabled, gamma0]
  exact hstepToClosure htann

theorem emit_catHomBlk
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.catHomBlk ∈ finalContext I := by
  have hstepToClosure :
      step allRules (initialContext I) ⊆ finalContext I := by
    have hmono :
        step allRules (initialContext I) ⊆
          step allRules (closure allRules (initialContext I)) :=
      monotone_step allRules (subset_closure allRules (initialContext I))
    simpa [finalContext, closure_fixed allRules (initialContext I)] using hmono
  have hblk : CertTag.catHomBlk ∈ step allRules (initialContext I) := by
    simp [allRules, bridgeRules, promotionRules, backendRules,
      permitHdg, permitTann, permitLock, initialContext, step, fireRule, Rule.enabled, gamma0]
  exact hstepToClosure hblk

end HypoHodge.Hodge
