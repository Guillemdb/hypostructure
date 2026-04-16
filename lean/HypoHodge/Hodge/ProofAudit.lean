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

theorem no_lock_inc
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.catHomInc ∉ finalContext I := by
  rw [finalContext, tannStepContext, mhsStepContext, backendStepContext, initialContext]
  simp [gamma0]

theorem no_promo_inc
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.promoInc ∉ finalContext I := by
  rw [finalContext, tannStepContext, mhsStepContext, backendStepContext, initialContext]
  simp [gamma0]

theorem hodgeProofAudit
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    Disjoint
      (obligations (finalContext I))
      (goalCone deps CertTag.catHomBlk) := by
  have hEmpty : obligations (finalContext I) = ∅ := by
    apply Finset.eq_empty_iff_forall_not_mem.mpr
    intro k hk
    rcases (mem_obligations_iff (finalContext I) k).1 hk with ⟨hkCtx, hkInc⟩
    cases k <;> simp [CertTag.isInc] at hkInc
    · exact no_lock_inc (I := I) hkCtx
    · exact no_promo_inc (I := I) hkCtx
  simp [hEmpty]

theorem no_semantic_lock_gap
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    ¬ UnresolvedLockObstruction I := by
  intro hGap
  exact hGap.noCycleLift (emit_catHomBlk_semantics I)

theorem no_semantic_promotion_gap
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    ¬ PromotionGap I := by
  intro hGap
  exact hGap.blockedByPromotion (hodgeGammaCertificate I).tensorPres

theorem hodgeSemanticAudit
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    ¬ UnresolvedLockObstruction I ∧ ¬ PromotionGap I := by
  exact ⟨no_semantic_lock_gap I, no_semantic_promotion_gap I⟩

end HypoHodge.Hodge
