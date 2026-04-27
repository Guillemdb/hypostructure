import Hypostructure.Backends.NavierStokes.ProofSetup.ResidualInterface

namespace Hypostructure.Backends.NavierStokes.ResidualBranch

open Hypostructure.Backends.NavierStokes.ProofSetup

structure ImportedSetupResidualObject where
  contract : SetupResidualInterfaceContract
  sameNormalizedCenteredObject : Prop := True
  retainedCompactMass : Prop := True
  pressureGaugeCompatible : Prop := True
  validTerminalAncestry : Prop := True
  deriving Repr

inductive ResidualAlternativeTag where
  | axisymmetric
  | rotational
  | degenerateStationaryHull
  | affineParasitic
  | criticalTail
  | genericTerminal
  deriving DecidableEq, Repr, Fintype

structure ResidualAlternativeState where
  imported : ImportedSetupResidualObject
  tag : ResidualAlternativeTag
  deriving Repr

end Hypostructure.Backends.NavierStokes.ResidualBranch