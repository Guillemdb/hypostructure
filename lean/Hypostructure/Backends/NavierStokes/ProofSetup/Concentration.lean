import Hypostructure.Backends.NavierStokes.ProofSetup.ScaleInvariant
import Hypostructure.Backends.NavierStokes.ProofSetup.Axioms

namespace Hypostructure.Backends.NavierStokes.ProofSetup

axiom vanishingConcentrationImpliesRegular
    (w : VanishingCriticalQuantitiesWitness) :
    RegularityCertificate

axiom singularityYieldsPositiveConcentration
    (w : SingularityWitness) :
    PositiveConcentrationSequence

theorem positiveConcentrationCarriesLowerBounds
    (seq : PositiveConcentrationSequence) :
    seq.criticalMassLowerBound ∧ seq.velocityMassLowerBound := by
  exact ⟨seq.criticalMassLowerBound, seq.velocityMassLowerBound⟩

theorem singularityYieldsPositiveConcentration_lower_bounds
    (w : SingularityWitness) :
    let seq := singularityYieldsPositiveConcentration w
    seq.criticalMassLowerBound ∧ seq.velocityMassLowerBound := by
  simpa using positiveConcentrationCarriesLowerBounds (singularityYieldsPositiveConcentration w)

end Hypostructure.Backends.NavierStokes.ProofSetup