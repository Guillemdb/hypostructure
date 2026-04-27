import Hypostructure.Backends.NavierStokes.ProofSetup.ClassRouting

namespace Hypostructure.Backends.NavierStokes.ProofSetup

structure SetupResidualInterfaceContract where
  export : ResidualExportData
  retainsCompactMass : Prop := export.compactMassCarryover
  retainsPressureCompatibility : Prop := export.pressureCompatibility.transferred
  retainsAncestry : Prop := export.ancestry.realizedByOriginalBranch
  deriving Repr

structure ResidualClosureHypothesis where
  closesRemainingExport :
    ∀ export : ResidualExportData, False

axiom remainingClassEquivalentToResidualExport
    (G : GeneratedSereginClass)
    (state : GeneratedSereginState)
    (hmem : state ∈ G.states) :
    RemainingClassPredicate state ↔
      ∃ export : ResidualExportData,
        export.generatedClass = G ∧ export.chosenState = state

def setupResidualInterfaceOfExport
    (export : ResidualExportData) : SetupResidualInterfaceContract :=
  { export := export }

end Hypostructure.Backends.NavierStokes.ProofSetup