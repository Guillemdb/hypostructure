import HypoHodge.Algebraic.CatLib
import HypoHodge.Imported.Boundary

namespace HypoHodge.Algebraic
open HypoHodge.Imported

structure GammaPackage where
  sourceCategory : Type
  fiberFunctor : Type
  exactness : Prop
  faithfulness : Prop
  tensorPreservation : Prop

def GammaPackageSound (I : VerifiedHodgeThinInput) : Prop :=
  ∃ G : GammaPackage,
    G.exactness ∧ G.faithfulness ∧ G.tensorPreservation ∧ Nonempty (ProducesGamma I)

def HasGammaPackage (I : VerifiedHodgeThinInput) : Prop :=
  ∃ G : GammaPackage,
    G.exactness ∧ G.faithfulness ∧ G.tensorPreservation ∧ GammaPackageSound I

def gammaContextPremises (I : VerifiedHodgeThinInput) : GammaContextPremises I :=
  ⟨ambientCertificate I,
    representationConservativeCertificate I,
    representationCompleteCertificate I,
    categoryLibraryCertificate I⟩

theorem hodgeGammaConstructor
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    HasGammaPackage I := by
  let hGamma : ProducesGamma I :=
    ImportedHodgeAxioms.tannakianContextSound (I := I) (gammaContextPremises I)
  refine ⟨{ sourceCategory := I.varietyType,
            fiberFunctor := I.cohomologyType,
            exactness := I.gammaExact,
            faithfulness := I.gammaFaithful,
            tensorPreservation := I.gammaTensorPres },
    hGamma.exact,
    hGamma.faithful,
    hGamma.tensorPres,
    ?_⟩
  exact ⟨{ sourceCategory := I.varietyType,
           fiberFunctor := I.cohomologyType,
           exactness := I.gammaExact,
           faithfulness := I.gammaFaithful,
           tensorPreservation := I.gammaTensorPres },
    hGamma.exact,
    hGamma.faithful,
    hGamma.tensorPres,
    ⟨hGamma⟩⟩

end HypoHodge.Algebraic
