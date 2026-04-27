import Hypostructure.Backends.NavierStokes.ProofSetup.NoEscape
import Hypostructure.Backends.NavierStokes.ProofSetup.Axioms

namespace Hypostructure.Backends.NavierStokes.ProofSetup

def certifyLocalTypeIEntry
    (seq : PositiveConcentrationSequence)
    (env : LocalTypeIEnvelopeWitness)
    (compact : TerminalCompactnessWitness)
    (noEscape : VelocityNoEscapeCertificate) :
    AdmissibleLocalTypeISequence :=
  { concentration := seq,
    localEnvelope := env,
    terminalCompactness := compact,
    noEscape := noEscape,
    admissibility :=
      (0 < seq.etaV ∧ ∀ n, seq.etaV ≤ (seq.criticalValues n).C.value) ∧
        noEscape.noEscape ∧ compact.compactness,
    compatibility :=
      env.solution.name = seq.solution.name ∧
        noEscape.singularPoint.terminalTime = seq.singularPoint.terminalTime }

def exportTypeIIBranch
    (seq : PositiveConcentrationSequence)
    (failure : FailedLocalTypeIEnvelopeWitness)
    (energy : LocalEnergyTransferData)
    (pressure : PressureTransferData)
    (windows : ScaleWindowData) :
    TypeIIExportData :=
  { concentration := seq,
    failedLocalTypeIEnvelope := failure,
    localEnergy := energy,
    pressure := pressure,
    scaleWindows := windows,
    compatibility :=
      (0 < seq.etaCD ∧ ∀ n, seq.etaCD ≤ (seq.criticalValues n).totalCD) ∧
        failure.failure ∧ energy.transferred ∧ pressure.transferred ∧ windows.shrinking }

inductive LocalBlowupAlternative where
  | typeI (entry : AdmissibleLocalTypeISequence)
  | typeII (data : TypeIIExportData)

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

noncomputable def localPointwiseTypeIIsAdmissible
    (seq : PositiveConcentrationSequence)
    (env : LocalTypeIEnvelopeWitness)
    (noEscape : VelocityNoEscapeCertificate) :
    AdmissibleLocalTypeISequence :=
  certifyLocalTypeIEntry seq env (localPointwiseTypeIProducesCompactness seq env) noEscape

end Hypostructure.Backends.NavierStokes.ProofSetup