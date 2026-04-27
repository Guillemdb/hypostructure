import Hypostructure.Backends.NavierStokes.ProofSetup.Dichotomy

namespace Hypostructure.Backends.NavierStokes.ProofSetup

structure UnifiedTypeIEntryData where
  localEntry : AdmissibleLocalTypeISequence
  globalEnvelope : Option FiniteEnergyTypeIEnvelope := none
  deriving Repr

def UnifiedTypeIEntryData.solution
    (entry : UnifiedTypeIEntryData) : SuitableWeakSolution :=
  entry.localEntry.concentration.solution

def UnifiedTypeIEntryData.singularPoint
    (entry : UnifiedTypeIEntryData) : TerminalSingularPoint :=
  entry.localEntry.concentration.singularPoint

def UnifiedTypeIEntryData.noEscape
    (entry : UnifiedTypeIEntryData) : VelocityNoEscapeCertificate :=
  entry.localEntry.noEscape

axiom ofLocalAdmissible
    (entry : AdmissibleLocalTypeISequence) :
    UnifiedTypeIEntryData

def ofGlobalEnvelope
    (H : FiniteEnergyTypeIEnvelope)
    (seq : PositiveConcentrationSequence)
    (env : LocalTypeIEnvelopeWitness)
    (compact : TerminalCompactnessWitness)
    (noEscape : VelocityNoEscapeCertificate) :
    UnifiedTypeIEntryData :=
  { localEntry := certifyLocalTypeIEntry seq env compact noEscape
    globalEnvelope := some H }

end Hypostructure.Backends.NavierStokes.ProofSetup