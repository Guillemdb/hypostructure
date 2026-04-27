import Hypostructure.Backends.NavierStokes.ProofSetup.ClassRouting
import Hypostructure.Backends.NavierStokes.ProofSetup.Axioms

namespace Hypostructure.Backends.NavierStokes.ProofSetup

def StructuredWitness (state : GeneratedSereginState) : Prop :=
  AxisymmetricNoSwirlPredicate state ∨
    PointwiseScaleInvariantPredicate state ∨
      FiniteSwirlPredicate state ∨
        PeriodicSwirlPredicate state ∨ WeightedVorticityPredicate state

axiom structuredWitnessOfClass
    (state : GeneratedSereginState) :
    StructureDecayClassPredicate state → StructuredWitness state

theorem structureDecayClassExcluded
    (state : GeneratedSereginState)
    (hstate : StructureDecayClassPredicate state) :
    False := by
  rcases structuredWitnessOfClass state hstate with hAxis | hRest
  · exact axisymmetricNoSwirlLiouville state hAxis
  · rcases hRest with hPoint | hRest
    · exact pointwiseScaleInvariantLiouville state hPoint
    · rcases hRest with hFinite | hRest
      · exact finiteSwirlLiouville state hFinite
      · rcases hRest with hPeriodic | hWeighted
        · exact periodicSwirlLiouville state hPeriodic
        · exact weightedVorticityDecayLiouville state hWeighted

end Hypostructure.Backends.NavierStokes.ProofSetup