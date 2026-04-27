import Hypostructure.Backends.NavierStokes.ProofSetup.ResidualInterface

namespace Hypostructure.Backends.NavierStokes.ResidualBranch

open Hypostructure.Backends.NavierStokes.ProofSetup

structure ImportedSetupResidualObject where
  contract : SetupResidualInterfaceContract
  sameNormalizedCenteredObject : Prop :=
    contract.residualExport.chosenState.centered ∧
      contract.residualExport.chosenState.nonvanishing.invariantWitness
  retainedCompactMass : Prop := contract.retainsCompactMass
  pressureGaugeCompatible : Prop := contract.retainsPressureCompatibility
  validTerminalAncestry : Prop := contract.retainsAncestry

inductive ResidualAlternativeTag where
  | axisymmetric
  | rotational
  | degenerateStationaryHull
  | affineParasitic
  | criticalTail
  | genericTerminal
  deriving DecidableEq, Repr

structure ResidualAlternativeState where
  imported : ImportedSetupResidualObject
  tag : ResidualAlternativeTag

end Hypostructure.Backends.NavierStokes.ResidualBranch