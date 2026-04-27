import Hypostructure.Backends.NavierStokes.ProofSetup.Basic

namespace Hypostructure.Backends.NavierStokes.ProofSetup

def singularSetAtTime (_ : SuitableWeakSolution) : TerminalSingularSet := Set.univ

axiom temporalCutoffConvention
    (u : SuitableWeakSolution) :
    TemporalCutoffConvention

end Hypostructure.Backends.NavierStokes.ProofSetup