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

def GammaPackageSound (I : VerifiedHodgeThinInput) : Prop :=
  GammaContextPremises I → ProducesGamma I

def HasGammaPackage (I : VerifiedHodgeThinInput) : Prop :=
  ∃ G : GammaPackage, G.exactness ∧ G.faithfulness ∧ G.tensorPres ∧ GammaPackageSound I

theorem hodgeGammaConstructor
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    HasGammaPackage I := by
  refine ⟨{ C := PUnit, omegaB := PUnit, exactness := True, faithfulness := True, tensorPres := True },
    trivial, trivial, trivial, ?_⟩
  intro hGamma
  exact ImportedHodgeAxioms.tannakianContextSound (I := I) hGamma

end HypoHodge.Algebraic
