import Hypostructure.Backends.NavierStokes.ResidualBranch.Basic

namespace Hypostructure.Backends.NavierStokes.ResidualBranch

open Hypostructure.Backends.NavierStokes.ProofSetup

def importSetupResidualObject
    (export : ResidualExportData) : ImportedSetupResidualObject :=
  { contract := setupResidualInterfaceOfExport export }

end Hypostructure.Backends.NavierStokes.ResidualBranch