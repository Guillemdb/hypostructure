import Hypostructure.Backends.NavierStokes.ProofSetup.ClassRouting
import Hypostructure.Backends.NavierStokes.ProofSetup.Axioms

namespace Hypostructure.Backends.NavierStokes.ProofSetup

axiom smallClassExcluded
    (state : GeneratedSereginState) :
    SmallClassPredicate state → False

theorem stationaryClassExcluded
    (state : GeneratedSereginState) :
    StationaryClassPredicate state → False :=
  stationarySelfSimilarRigidity state

end Hypostructure.Backends.NavierStokes.ProofSetup