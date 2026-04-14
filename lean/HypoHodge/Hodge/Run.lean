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

axiom emit_mhs
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.mhs ∈ contextAfterMhs I

axiom emit_tann
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.tann ∈ contextAfterTann I

axiom emit_catHomBlk
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.catHomBlk ∈ finalContext I

end HypoHodge.Hodge
