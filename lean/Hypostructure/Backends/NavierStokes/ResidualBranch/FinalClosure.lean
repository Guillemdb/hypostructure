import Hypostructure.Backends.NavierStokes.ResidualBranch.Routing

namespace Hypostructure.Backends.NavierStokes.ResidualBranch

open Hypostructure.Backends.NavierStokes.ProofSetup

axiom axisymmetricAlternativeClosed
    (obj : ImportedSetupResidualObject) : False

axiom rotationalAlternativeClosed
    (obj : ImportedSetupResidualObject) : False

axiom degenerateStationaryHullClosed
    (obj : ImportedSetupResidualObject) : False

axiom affineParasiticAlternativeClosed
    (obj : ImportedSetupResidualObject) : False

axiom criticalTailAlternativeClosed
    (obj : ImportedSetupResidualObject) : False

axiom genericTerminalAlternativeClosed
    (obj : ImportedSetupResidualObject) : False

def companionResidualClosureHypothesis : ResidualClosureHypothesis :=
    { closesRemainingExport := fun residual => by
            let obj := importSetupResidualObject residual
            cases routeImportedResidualObject obj with
            | axisymmetric obj => exact axisymmetricAlternativeClosed obj
            | rotational obj => exact rotationalAlternativeClosed obj
            | degenerateStationaryHull obj => exact degenerateStationaryHullClosed obj
            | affineParasitic obj => exact affineParasiticAlternativeClosed obj
            | criticalTail obj => exact criticalTailAlternativeClosed obj
            | genericTerminal obj => exact genericTerminalAlternativeClosed obj }

end Hypostructure.Backends.NavierStokes.ResidualBranch