import Hypostructure.Backends.NavierStokes.ResidualBranch.ImportFromSetup

namespace Hypostructure.Backends.NavierStokes.ResidualBranch

inductive ResidualAlternative where
  | axisymmetric (obj : ImportedSetupResidualObject)
  | rotational (obj : ImportedSetupResidualObject)
  | degenerateStationaryHull (obj : ImportedSetupResidualObject)
  | affineParasitic (obj : ImportedSetupResidualObject)
  | criticalTail (obj : ImportedSetupResidualObject)
  | genericTerminal (obj : ImportedSetupResidualObject)

axiom routeImportedResidualObject
    (obj : ImportedSetupResidualObject) :
    ResidualAlternative

end Hypostructure.Backends.NavierStokes.ResidualBranch