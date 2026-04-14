import HypoHodge.Hodge.Final

namespace HypoHodge.Examples
open HypoHodge.Algebraic
open HypoHodge.Imported
open HypoHodge.Hodge
open HypoHodge.Core

noncomputable def toyInput : VerifiedHodgeThinInput where
  V := ℝ
  instAddComm := inferInstance
  instModuleR := inferInstance
  instNormedGroup := inferInstance
  instNormedSpace := inferInstance
  instFiniteDim := inferInstance
  p := 1
  Qrank := 1
  hodgeSubset := Set.univ
  symmetry := PUnit
  potential := fun x => ‖x‖ ^ (2 : ℕ)
  dissipation := fun _ => 0
  flow := fun _ x => x
  flow_id := by intro t x; rfl
  dissipation_zero := by intro x; rfl
  potential_quad := by intro x; rfl
  connected := True
  contractible := True
  tame := True

instance : ImportedHodgeAxioms toyInput where
  classicalExtraction := ⟨⟩
  hodgeBridgeSound := by intro _; exact ⟨⟩
  tannakianContextSound := by intro _; exact ⟨⟩
  tannakianBridgeSound := by intro _; exact ⟨⟩

example : ProofComplete allRules deps (runHodgeSystem toyInput) CertTag.catHomBlk :=
  hodge_framework_unconditional toyInput

end HypoHodge.Examples
