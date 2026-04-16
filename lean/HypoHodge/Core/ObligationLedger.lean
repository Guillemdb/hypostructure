import HypoHodge.Core.GoalCone
import Hypostructure.Core.ObligationLedger

namespace HypoHodge.Core

def obligations (Γ : Context) : Context :=
  Hypostructure.Core.obligations CertTag.isInc Γ

theorem obligations_subset (Γ : Context) :
    obligations Γ ⊆ Γ := by
  simpa [obligations] using Hypostructure.Core.obligations_subset CertTag.isInc Γ

theorem mem_obligations_iff (Γ : Context) (k : CertTag) :
    k ∈ obligations Γ ↔ k ∈ Γ ∧ CertTag.isInc k = true := by
  simpa [obligations] using Hypostructure.Core.mem_obligations_iff CertTag.isInc Γ k

end HypoHodge.Core
