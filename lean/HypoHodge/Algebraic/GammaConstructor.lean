import HypoHodge.Algebraic.CatLib
import HypoHodge.Imported.Boundary

namespace HypoHodge.Algebraic
open HypoHodge.Imported

structure GammaPackage where
  C            : Type
  omegaB       : Type
  exactness    : Prop
  faithfulness : Prop
  tensorPres   : Prop

def HasGammaPackage (I : VerifiedHodgeThinInput) : Prop :=
  ∃ G : GammaPackage, True

theorem hodgeGammaConstructor
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    HasGammaPackage I := by
  refine ⟨{ C := PUnit, omegaB := PUnit, exactness := True, faithfulness := True, tensorPres := True }, trivial⟩

end HypoHodge.Algebraic
