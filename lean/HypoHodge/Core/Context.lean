import HypoHodge.Core.CertTag

namespace HypoHodge.Core

abbrev Context := Finset CertTag

theorem context_ext (Γ Δ : Context) (h : ∀ k, k ∈ Γ ↔ k ∈ Δ) : Γ = Δ := by
  exact Finset.ext h

end HypoHodge.Core
