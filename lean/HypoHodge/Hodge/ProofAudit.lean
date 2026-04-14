import HypoHodge.Hodge.Run
import HypoHodge.Algebraic.LocalCertificates
import HypoHodge.Core.ObligationLedger

namespace HypoHodge.Hodge
open HypoHodge.Core
open HypoHodge.Algebraic
open HypoHodge.Imported

theorem no_local_inc
    (I : VerifiedHodgeThinInput) :
    Disjoint (obligations (gamma0 I)) (goalCone deps CertTag.catHomBlk) := by
  simp [gamma0_no_local_inc I]

theorem no_backend_inc
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.germ ∉ obligations (finalContext I) ∧
    CertTag.init ∉ obligations (finalContext I) ∧
    CertTag.catLib ∉ obligations (finalContext I) := by
  simp [obligations, CertTag.isInc]

axiom no_lock_inc
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.catHomInc ∉ finalContext I

axiom no_promo_inc
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.promoInc ∉ finalContext I

theorem hodgeProofAudit
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    Disjoint
      (obligations (finalContext I))
      (goalCone deps CertTag.catHomBlk) := by
  have hEmpty : obligations (finalContext I) = ∅ := by
    apply Finset.ext
    intro k
    cases k <;>
      simp [obligations, CertTag.isInc, no_lock_inc (I := I), no_promo_inc (I := I)]
  simp [hEmpty]

end HypoHodge.Hodge
