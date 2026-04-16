import HypoHodge.Imported.Boundary
import HypoHodge.Core.ObligationLedger

namespace HypoHodge.Algebraic
open HypoHodge.Core

theorem emit_adj (I : VerifiedHodgeThinInput) :
    CertTag.adj ∈ gamma0 I := by
  simp [gamma0]

theorem adj_semantics (I : VerifiedHodgeThinInput) :
    AdjunctionCertificate I := by
  exact adjunctionCertificate I

theorem emit_dE (I : VerifiedHodgeThinInput) :
    CertTag.dE ∈ gamma0 I := by
  simp [gamma0]

theorem dE_semantics (I : VerifiedHodgeThinInput) :
    DefectEnergyCertificate I := by
  exact defectEnergyCertificate I

theorem emit_recN (I : VerifiedHodgeThinInput) :
    CertTag.recN ∈ gamma0 I := by
  simp [gamma0]

theorem recN_semantics (I : VerifiedHodgeThinInput) :
    RecurrenceCertificate I := by
  exact recurrenceCertificate I

theorem emit_cMu (I : VerifiedHodgeThinInput) :
    CertTag.cMu ∈ gamma0 I := by
  simp [gamma0]

theorem cMu_semantics (I : VerifiedHodgeThinInput) :
    CycleMultiplicityCertificate I := by
  exact cycleMultiplicityCertificate I

theorem emit_scLambda (I : VerifiedHodgeThinInput) :
    CertTag.scLambda ∈ gamma0 I := by
  simp [gamma0]

theorem scLambda_semantics (I : VerifiedHodgeThinInput) :
    TameLambdaCertificate I := by
  exact tameLambdaCertificate I

theorem emit_scPartialC (I : VerifiedHodgeThinInput) :
    CertTag.scPartialC ∈ gamma0 I := by
  simp [gamma0]

theorem scPartialC_semantics (I : VerifiedHodgeThinInput) :
    PartialCompactnessCertificate I := by
  exact partialCompactnessCertificate I

theorem emit_capH (I : VerifiedHodgeThinInput) :
    CertTag.capH ∈ gamma0 I := by
  simp [gamma0]

theorem capH_semantics (I : VerifiedHodgeThinInput) :
    CapacityCertificate I := by
  exact capacityCertificate I

theorem emit_lsSigma (I : VerifiedHodgeThinInput) :
    CertTag.lsSigma ∈ gamma0 I := by
  simp [gamma0]

theorem lsSigma_semantics (I : VerifiedHodgeThinInput) :
    LinearStabilityCertificate I := by
  exact linearStabilityCertificate I

theorem emit_tbPi (I : VerifiedHodgeThinInput) :
    CertTag.tbPi ∈ gamma0 I := by
  simp [gamma0]

theorem tbPi_semantics (I : VerifiedHodgeThinInput) :
    ThinBoundaryPiCertificate I := by
  exact thinBoundaryPiCertificate I

theorem emit_tbO (I : VerifiedHodgeThinInput) :
    CertTag.tbO ∈ gamma0 I := by
  simp [gamma0]

theorem tbO_semantics (I : VerifiedHodgeThinInput) :
    ThinBoundaryOCertificate I := by
  exact thinBoundaryOCertificate I

theorem emit_boundPartial (I : VerifiedHodgeThinInput) :
    CertTag.boundPartial ∈ gamma0 I := by
  simp [gamma0]

theorem boundPartial_semantics (I : VerifiedHodgeThinInput) :
    BoundedPartialCertificate I := by
  exact boundedPartialCertificate I

theorem gamma0_complete (I : VerifiedHodgeThinInput) :
    ({ CertTag.adj, CertTag.dE, CertTag.recN, CertTag.cMu, CertTag.scLambda,
       CertTag.scPartialC, CertTag.capH, CertTag.lsSigma,
       CertTag.tbPi, CertTag.tbO, CertTag.boundPartial } : Finset CertTag) ⊆ gamma0 I := by
  intro k hk
  simpa [gamma0] using hk

theorem gamma0_no_local_inc (I : VerifiedHodgeThinInput) :
    obligations (gamma0 I) = ∅ := by
  apply Finset.eq_empty_iff_forall_not_mem.mpr
  intro k hk
  rcases (mem_obligations_iff (gamma0 I) k).1 hk with ⟨hkGamma, hkInc⟩
  cases k <;> simp [gamma0, CertTag.isInc] at hkGamma hkInc

end HypoHodge.Algebraic
