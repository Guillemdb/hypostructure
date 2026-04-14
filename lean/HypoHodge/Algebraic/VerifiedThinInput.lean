import HypoHodge.Core.ProofComplete

namespace HypoHodge.Algebraic
open HypoHodge.Core

structure VerifiedHodgeThinInput where
  V                : Type
  instAddComm      : AddCommGroup V
  instModuleR      : Module ℝ V
  instNormedGroup  : NormedAddCommGroup V
  instNormedSpace  : NormedSpace ℝ V
  instFiniteDim    : FiniteDimensional ℝ V

  p                : ℕ
  Qrank            : ℕ
  hodgeSubset      : Set V
  symmetry         : Type

  potential        : V → ℝ
  dissipation      : V → ℝ
  flow             : ℝ → V → V

  flow_id          : ∀ t x, flow t x = x
  dissipation_zero : ∀ x, dissipation x = 0
  potential_quad   : ∀ x, potential x = ‖x‖ ^ (2 : ℕ)

  connected        : Prop
  contractible     : Prop
  tame             : Prop

attribute [instance] VerifiedHodgeThinInput.instAddComm
attribute [instance] VerifiedHodgeThinInput.instModuleR
attribute [instance] VerifiedHodgeThinInput.instNormedGroup
attribute [instance] VerifiedHodgeThinInput.instNormedSpace
attribute [instance] VerifiedHodgeThinInput.instFiniteDim

def gamma0 (I : VerifiedHodgeThinInput) : Context :=
  ({ CertTag.adj, CertTag.dE, CertTag.recN, CertTag.cMu, CertTag.scLambda,
     CertTag.scPartialC, CertTag.capH, CertTag.lsSigma,
     CertTag.tbPi, CertTag.tbO, CertTag.boundPartial } : Finset CertTag)

end HypoHodge.Algebraic
