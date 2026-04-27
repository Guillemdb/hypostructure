import Hypostructure.Backends.NavierStokes.ProofSetup.ClassicalClosure
import Hypostructure.Backends.NavierStokes.ProofSetup.TightClosure
import Hypostructure.Backends.NavierStokes.ProofSetup.StructuredClosure
import Hypostructure.Backends.NavierStokes.ProofSetup.ResidualInterface
import Hypostructure.Backends.NavierStokes.ProofSetup.EntryData

namespace Hypostructure.Backends.NavierStokes.ProofSetup

axiom localTypeIRegularUnderResidualClosure
    (entry : UnifiedTypeIEntryData)
    (_hresidual : ResidualClosureHypothesis) :
    RegularAt entry.solution entry.singularPoint

end Hypostructure.Backends.NavierStokes.ProofSetup