import HypoHodge.Hodge.Final
import Hypostructure.Instances.Hodge.Thin

namespace Hypostructure.Instances.Hodge

open HypoHodge.Algebraic
open HypoHodge.Imported

abbrev HodgeInput := VerifiedHodgeThinInput
abbrev HodgeAxioms := ImportedHodgeAxioms
abbrev GenericHodgeInput := Hypostructure.Problem.ThinInput

theorem kernel_complete
    (I : HodgeInput)
    [HodgeAxioms I] :
    HypoHodge.Core.ProofComplete
      HypoHodge.Hodge.allRules
      HypoHodge.Hodge.deps
      (HypoHodge.Hodge.runHodgeSystem I)
      HypoHodge.Core.CertTag.catHomBlk := by
  exact HypoHodge.Hodge.hodge_framework_kernel_complete I

theorem target_realized
    (I : HodgeInput)
    [HodgeAxioms I] :
    TargetRealizedByCycle I := by
  exact HypoHodge.Hodge.hodge_framework_target_realized I

theorem target_hodge
    (I : HodgeInput)
    [HodgeAxioms I] :
    TargetHodgeClass I := by
  exact HypoHodge.Hodge.hodge_framework_target_hodge I

theorem unconditional
    (I : HodgeInput)
    [HodgeAxioms I] :
    HodgeConjectureTarget I := by
  exact HypoHodge.Hodge.hodge_framework_unconditional I

end Hypostructure.Instances.Hodge
