import Hypostructure.Backends.NavierStokes.ProofSetup.NoEscape
import Hypostructure.Backends.NavierStokes.ProofSetup.Axioms

namespace Hypostructure.Backends.NavierStokes.ProofSetup

def certifyLocalTypeIEntry
    (seq : PositiveConcentrationSequence)
    (env : LocalTypeIEnvelopeWitness)
    (compact : TerminalCompactnessWitness)
    (noEscape : VelocityNoEscapeCertificate) :
    AdmissibleLocalTypeISequence :=
  { concentration := seq
    localEnvelope := env
    terminalCompactness := compact
    noEscape := noEscape }

def exportTypeIIBranch
    (seq : PositiveConcentrationSequence)
    (failure : FailedLocalTypeIEnvelopeWitness)
    (energy : LocalEnergyTransferData)
    (pressure : PressureTransferData)
    (windows : ScaleWindowData) :
    TypeIIExportData :=
  { concentration := seq
    failedLocalTypeIEnvelope := failure
    localEnergy := energy
    pressure := pressure
    scaleWindows := windows }

inductive LocalBlowupAlternative where
  | typeI (entry : AdmissibleLocalTypeISequence)
  | typeII (data : TypeIIExportData)
  deriving Repr

def typeIAlternativeOfCertifiedEntry
    (entry : AdmissibleLocalTypeISequence) :
    LocalBlowupAlternative :=
  .typeI entry

def typeIIAlternativeOfExport
    (data : TypeIIExportData) :
    LocalBlowupAlternative :=
  .typeII data

axiom localTypeITypeIIDichotomy
    (seq : PositiveConcentrationSequence) :
    LocalBlowupAlternative

axiom localPointwiseTypeIProducesCompactness
    (seq : PositiveConcentrationSequence)
    (env : LocalTypeIEnvelopeWitness) :
    TerminalCompactnessWitness

theorem localPointwiseTypeIIsAdmissible
    (seq : PositiveConcentrationSequence)
    (env : LocalTypeIEnvelopeWitness)
    (noEscape : VelocityNoEscapeCertificate) :
    AdmissibleLocalTypeISequence :=
  certifyLocalTypeIEntry seq env (localPointwiseTypeIProducesCompactness seq env) noEscape

end Hypostructure.Backends.NavierStokes.ProofSetup