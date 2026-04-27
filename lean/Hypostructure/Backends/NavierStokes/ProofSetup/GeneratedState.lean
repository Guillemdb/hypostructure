import Hypostructure.Backends.NavierStokes.ProofSetup.CenteredProfile

namespace Hypostructure.Backends.NavierStokes.ProofSetup

axiom generatedStateStable
    (G : GeneratedSereginClass) :
    True

axiom mildStratumStable
    (S : MildStratum) :
    True

theorem mildStatesAreGenerated
    (S : MildStratum) :
    S.states ⊆ S.carrier.states := by
  intro state hstate
  exact hstate

end Hypostructure.Backends.NavierStokes.ProofSetup