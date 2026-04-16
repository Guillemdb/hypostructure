import HypoHodge.Core.CertTag
import Hypostructure.Core.Context

namespace HypoHodge.Core

abbrev Context := Hypostructure.Core.Context CertTag

theorem context_ext (Γ Δ : Context) (h : ∀ k, k ∈ Γ ↔ k ∈ Δ) : Γ = Δ := by
  exact Hypostructure.Core.context_ext Γ Δ h

end HypoHodge.Core
