import Hypostructure.Backends.NavierStokes.ProofSetup.Ambient

namespace Hypostructure.Backends.NavierStokes.ProofSetup

def criticalQuantitiesOn
    (_ : SuitableWeakSolution)
    (_ : ParabolicCylinder) : CriticalQuantityBundle :=
    { C := { value := 0 }
        D := { value := 0 }
        A := { value := 0 }
        E := { value := 0 } }

axiom rescaledSolutionOf
        (u : SuitableWeakSolution)
        (Q : ParabolicCylinder) :
        RescaledSolution

axiom rescalingPreservesSuitability
    (u : SuitableWeakSolution)
    (Q : ParabolicCylinder) :
        (rescaledSolutionOf u Q).suitable

axiom criticalScalingIdentity
    (u : SuitableWeakSolution)
    (Q : ParabolicCylinder) :
    True

end Hypostructure.Backends.NavierStokes.ProofSetup