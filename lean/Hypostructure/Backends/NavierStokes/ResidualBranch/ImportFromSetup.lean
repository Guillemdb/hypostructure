import Hypostructure.Backends.NavierStokes.ResidualBranch.Basic

namespace Hypostructure.Backends.NavierStokes.ResidualBranch

open Hypostructure.Backends.NavierStokes.ProofSetup

def importSetupResidualObject
    (residual : ResidualExportData) : ImportedSetupResidualObject :=
  { contract := setupResidualInterfaceOfExport residual }

end Hypostructure.Backends.NavierStokes.ResidualBranch