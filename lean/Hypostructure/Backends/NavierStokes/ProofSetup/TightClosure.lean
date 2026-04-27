import Hypostructure.Backends.NavierStokes.ProofSetup.ClassRouting
import Hypostructure.Backends.NavierStokes.ProofSetup.Axioms

namespace Hypostructure.Backends.NavierStokes.ProofSetup

theorem tightClassExcluded
    (state : GeneratedSereginState) :
    TightClassPredicate state → False :=
  endpointAncientLiouville state

end Hypostructure.Backends.NavierStokes.ProofSetup