import Hypostructure.Backends.NavierStokes.ProofSetup.Concentration

namespace Hypostructure.Backends.NavierStokes.ProofSetup

axiom globalTypeIImpliesVelocityNoEscape
    (H : FiniteEnergyTypeIEnvelope) :
    VelocityNoEscapeCertificate

end Hypostructure.Backends.NavierStokes.ProofSetup