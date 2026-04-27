import Hypostructure.Backends.NavierStokes.ProofSetup.CenteredProfile

namespace Hypostructure.Backends.NavierStokes.ProofSetup

def GeneratedStateStability (G : GeneratedSereginClass) : Prop :=
    G.closedUnderTimeTranslations ∧ G.closedUnderLocalSmoothLimits

def MildStratumStability (S : MildStratum) : Prop :=
    S.projectedMildFormulation ∧ ∀ state, state ∈ S.states → state ∈ S.carrier.states

axiom generatedStateStable
    (G : GeneratedSereginClass) :
        GeneratedStateStability G

axiom mildStratumStable
    (S : MildStratum) :
        MildStratumStability S

theorem mildStatesAreGenerated
    (S : MildStratum) :
    S.states ⊆ S.carrier.states := by
        exact fun state hstate => ((mildStratumStable S).2) state hstate

end Hypostructure.Backends.NavierStokes.ProofSetup