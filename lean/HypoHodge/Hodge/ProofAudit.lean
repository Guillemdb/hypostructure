import HypoHodge.Hodge.Run
import HypoHodge.Algebraic.LocalCertificates
import HypoHodge.Core.ObligationLedger

namespace HypoHodge.Hodge
open HypoHodge.Core
open HypoHodge.Algebraic
open HypoHodge.Imported

private theorem not_mem_fireRule_of_not_mem_of_ne_conclusion
    (r : Rule) (Γ : Context) {k : CertTag}
    (hΓ : k ∉ Γ)
    (hconc : r.conclusion ≠ k) :
    k ∉ fireRule r Γ := by
  by_cases h : r.enabled Γ
  · rw [fireRule_eq_insert_of_enabled _ _ h]
    simp [hΓ, hconc]
  · rw [fireRule_eq_self_of_disabled _ _ h]
    exact hΓ

private theorem not_mem_step_of_not_mem_of_no_conclusion
    (rules : RuleSet) (Γ : Context) {k : CertTag}
    (hΓ : k ∉ Γ)
    (hrules : ∀ r ∈ rules, r.conclusion ≠ k) :
    k ∉ step rules Γ := by
  induction rules generalizing Γ with
  | nil =>
      simpa [step] using hΓ
  | cons r rs ih =>
      have hfire : k ∉ fireRule r Γ :=
        not_mem_fireRule_of_not_mem_of_ne_conclusion r Γ hΓ (hrules r (by simp))
      have hrs : ∀ r' ∈ rs, r'.conclusion ≠ k := by
        intro r' hr'
        exact hrules r' (by simp [hr'])
      simpa [step] using ih (fireRule r Γ) hfire hrs

private theorem not_mem_closureN_of_not_mem_of_no_conclusion
    (rules : RuleSet) (n : ℕ) (Γ : Context) {k : CertTag}
    (hΓ : k ∉ Γ)
    (hrules : ∀ r ∈ rules, r.conclusion ≠ k) :
    k ∉ closureN rules n Γ := by
  induction n generalizing Γ with
  | zero =>
      simpa [closureN] using hΓ
  | succ n ih =>
      have hstep : k ∉ step rules Γ :=
        not_mem_step_of_not_mem_of_no_conclusion rules Γ hΓ hrules
      simpa [closureN, Nat.iterate_succ] using ih (step rules Γ) hstep hrules

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
  have hinit : CertTag.catHomInc ∉ initialContext I := by
    simp [initialContext, gamma0]
  have hrules : ∀ r ∈ allRules, r.conclusion ≠ CertTag.catHomInc := by
    intro r hr
    simp [allRules, bridgeRules, promotionRules, HypoHodge.Algebraic.backendRules,
      permitHdg, permitTann, permitLock] at hr ⊢
  simpa [finalContext, initialContext, closure] using
    not_mem_closureN_of_not_mem_of_no_conclusion allRules allTags.card (initialContext I) hinit hrules

theorem no_promo_inc
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.promoInc ∉ finalContext I := by
  have hinit : CertTag.promoInc ∉ initialContext I := by
    simp [initialContext, gamma0]
  have hrules : ∀ r ∈ allRules, r.conclusion ≠ CertTag.promoInc := by
    intro r hr
    simp [allRules, bridgeRules, promotionRules, HypoHodge.Algebraic.backendRules,
      permitHdg, permitTann, permitLock] at hr ⊢
  simpa [finalContext, initialContext, closure] using
    not_mem_closureN_of_not_mem_of_no_conclusion allRules allTags.card (initialContext I) hinit hrules

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

end HypoHodge.Hodge
