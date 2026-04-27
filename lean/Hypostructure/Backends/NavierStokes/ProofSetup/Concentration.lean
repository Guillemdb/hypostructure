import Hypostructure.Backends.NavierStokes.ProofSetup.ScaleInvariant
import Hypostructure.Backends.NavierStokes.ProofSetup.Axioms

namespace Hypostructure.Backends.NavierStokes.ProofSetup

axiom vanishingConcentrationImpliesRegular
    (w : VanishingCriticalQuantitiesWitness) :
    RegularityCertificate

axiom singularityYieldsPositiveConcentration
    (w : SingularityWitness) :
    PositiveConcentrationSequence

def PositiveConcentrationLowerBounds
    (seq : PositiveConcentrationSequence) : Prop :=
  (0 < seq.etaCD ∧ ∀ n, seq.etaCD ≤ (seq.criticalValues n).totalCD) ∧
    (0 < seq.etaV ∧ ∀ n, seq.etaV ≤ (seq.criticalValues n).C.value)

theorem positiveConcentrationCarriesLowerBounds
    (seq : PositiveConcentrationSequence) :
    PositiveConcentrationLowerBounds seq := by
  exact ⟨seq.criticalMassLowerBound, seq.velocityMassLowerBound⟩

theorem singularityYieldsPositiveConcentration_lower_bounds
    (w : SingularityWitness) :
    let seq := singularityYieldsPositiveConcentration w
    PositiveConcentrationLowerBounds seq := by
  simpa using positiveConcentrationCarriesLowerBounds (singularityYieldsPositiveConcentration w)

end Hypostructure.Backends.NavierStokes.ProofSetup