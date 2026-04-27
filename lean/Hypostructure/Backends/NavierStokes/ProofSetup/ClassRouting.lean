import Hypostructure.Backends.NavierStokes.ProofSetup.GeneratedState

namespace Hypostructure.Backends.NavierStokes.ProofSetup

inductive GeneratedClassCase (state : GeneratedSereginState) where
  | small (h : SmallClassPredicate state)
  | stationary (h : StationaryClassPredicate state)
  | tight (h : TightClassPredicate state)
  | structured (h : StructureDecayClassPredicate state)
  | remaining (h : RemainingClassPredicate state)

axiom generatedClassExhaustive
    (G : GeneratedSereginClass)
    (state : GeneratedSereginState) :
    state ∈ G.states → GeneratedClassCase state

end Hypostructure.Backends.NavierStokes.ProofSetup