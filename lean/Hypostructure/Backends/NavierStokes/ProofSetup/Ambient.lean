import Hypostructure.Backends.NavierStokes.ProofSetup.Basic

namespace Hypostructure.Backends.NavierStokes.ProofSetup

def singularSetAtTime (u : SuitableWeakSolution) : TerminalSingularSet :=
        { solution := u,
            time := u.terminalTime,
            carrier := fun x => u.singularSpatialSet u.terminalTime x,
            membershipLaw := ∀ x, u.singularSpatialSet u.terminalTime x ↔
                u.singularSpatialSet u.terminalTime x }

def temporalCutoffConvention
        (u : SuitableWeakSolution) :
        TemporalCutoffConvention :=
        { interval := u.timeInterval,
            cutoff := fun _ => 0 }

end Hypostructure.Backends.NavierStokes.ProofSetup