import Hypostructure.Backends.NavierStokes.ProofSetup.EntryData

namespace Hypostructure.Backends.NavierStokes.ProofSetup

structure BlowupCompactnessDossier where
  entry : UnifiedTypeIEntryData
  blowup : BlowupSequence
  inheritedBounds : Prop := blowup.localBounds ∧ entry.localEntry.admissibility
  pressureAtlas : PressureAtlasData
  compactness : Prop := pressureAtlas.compatibility ∧ inheritedBounds

axiom compactnessOfBlowupSequence
    (entry : UnifiedTypeIEntryData) :
    BlowupCompactnessDossier

axiom ancientLimitExists
    (D : BlowupCompactnessDossier) :
    AncientSuitableTypeILimit

axiom ancientLimitNonzero
    (D : BlowupCompactnessDossier) :
    (ancientLimitExists D).nonzero

end Hypostructure.Backends.NavierStokes.ProofSetup