import Hypostructure.Backends.NavierStokes.ProofSetup.BlowupCompactness

namespace Hypostructure.Backends.NavierStokes.ProofSetup

axiom centeredProfileExists
    (D : BlowupCompactnessDossier) :
    CenteredAncientProfile

axiom rawSereginLimitExists
    (D : BlowupCompactnessDossier) :
    NormalizedSereginLimit

axiom invariantNonvanishingOfRawLimit
    (D : BlowupCompactnessDossier) :
    InvariantLocalNonvanishing

theorem compactL3WitnessOfRawLimit
    (D : BlowupCompactnessDossier) :
    CompactL3Witness :=
  (rawSereginLimitExists D).compactL3Witness

end Hypostructure.Backends.NavierStokes.ProofSetup