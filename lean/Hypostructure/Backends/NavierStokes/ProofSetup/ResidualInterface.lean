import Hypostructure.Backends.NavierStokes.ProofSetup.ClassRouting

namespace Hypostructure.Backends.NavierStokes.ProofSetup

structure SetupResidualInterfaceContract where
  residualExport : ResidualExportData
  retainsCompactMass : Prop := residualExport.compactMassCarryover
  retainsPressureCompatibility : Prop := residualExport.pressureCompatibility.transferred
  retainsAncestry : Prop := residualExport.ancestry.realizedByOriginalBranch

structure ResidualClosureHypothesis where
  closesRemainingExport :
    ∀ residual : ResidualExportData, False

axiom remainingClassEquivalentToResidualExport
    (G : GeneratedSereginClass)
    (state : GeneratedSereginState)
    (hmem : state ∈ G.states) :
    RemainingClassPredicate state ↔
      ∃ residual : ResidualExportData,
        residual.generatedClass = G ∧ residual.chosenState = state

def setupResidualInterfaceOfExport
    (residual : ResidualExportData) : SetupResidualInterfaceContract :=
  { residualExport := residual }

end Hypostructure.Backends.NavierStokes.ProofSetup