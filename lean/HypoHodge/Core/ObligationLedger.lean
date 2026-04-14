import HypoHodge.Core.GoalCone

namespace HypoHodge.Core

def obligations (Γ : Context) : Context :=
  Γ.filter (fun k => CertTag.isInc k = true)

theorem obligations_subset (Γ : Context) :
    obligations Γ ⊆ Γ := by
  intro k hk
  exact (Finset.mem_filter.mp hk).1

theorem mem_obligations_iff (Γ : Context) (k : CertTag) :
    k ∈ obligations Γ ↔ k ∈ Γ ∧ CertTag.isInc k = true := by
  simp [obligations]

end HypoHodge.Core
