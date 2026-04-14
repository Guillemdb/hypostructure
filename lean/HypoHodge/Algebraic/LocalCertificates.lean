import HypoHodge.Imported.Boundary
import HypoHodge.Core.ObligationLedger

namespace HypoHodge.Algebraic
open HypoHodge.Core

theorem emit_adj (I : VerifiedHodgeThinInput) :
    CertTag.adj ∈ gamma0 I := by
  simp [gamma0]

theorem emit_dE (I : VerifiedHodgeThinInput) :
    CertTag.dE ∈ gamma0 I := by
  simp [gamma0]

theorem emit_recN (I : VerifiedHodgeThinInput) :
    CertTag.recN ∈ gamma0 I := by
  simp [gamma0]

theorem emit_cMu (I : VerifiedHodgeThinInput) :
    CertTag.cMu ∈ gamma0 I := by
  simp [gamma0]

theorem emit_scLambda (I : VerifiedHodgeThinInput) :
    CertTag.scLambda ∈ gamma0 I := by
  simp [gamma0]

theorem emit_scPartialC (I : VerifiedHodgeThinInput) :
    CertTag.scPartialC ∈ gamma0 I := by
  simp [gamma0]

theorem emit_capH (I : VerifiedHodgeThinInput) :
    CertTag.capH ∈ gamma0 I := by
  simp [gamma0]

theorem emit_lsSigma (I : VerifiedHodgeThinInput) :
    CertTag.lsSigma ∈ gamma0 I := by
  simp [gamma0]

theorem emit_tbPi (I : VerifiedHodgeThinInput) :
    CertTag.tbPi ∈ gamma0 I := by
  simp [gamma0]

theorem emit_tbO (I : VerifiedHodgeThinInput) :
    CertTag.tbO ∈ gamma0 I := by
  simp [gamma0]

theorem emit_boundPartial (I : VerifiedHodgeThinInput) :
    CertTag.boundPartial ∈ gamma0 I := by
  simp [gamma0]

theorem gamma0_complete (I : VerifiedHodgeThinInput) :
    ({ CertTag.adj, CertTag.dE, CertTag.recN, CertTag.cMu, CertTag.scLambda,
       CertTag.scPartialC, CertTag.capH, CertTag.lsSigma,
       CertTag.tbPi, CertTag.tbO, CertTag.boundPartial } : Finset CertTag) ⊆ gamma0 I := by
  intro k hk
  simpa [gamma0] using hk

theorem gamma0_no_local_inc (I : VerifiedHodgeThinInput) :
    obligations (gamma0 I) = ∅ := by
  apply Finset.ext
  intro k
  simp [obligations, gamma0, CertTag.isInc]

end HypoHodge.Algebraic
