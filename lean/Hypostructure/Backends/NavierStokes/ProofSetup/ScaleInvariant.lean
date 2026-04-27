import Hypostructure.Backends.NavierStokes.ProofSetup.Ambient

namespace Hypostructure.Backends.NavierStokes.ProofSetup

def criticalQuantitiesOn
        (_ : SuitableWeakSolution)
        (Q : ParabolicCylinder) : CriticalQuantityBundle :=
        { C := {
                cylinder := Q,
                velocityCubicIntegral := 0,
                value := 0,
                definitionFormula := 0 = 0 / (Q.radius * Q.radius),
                nonnegativeIntegral := 0 ≤ (0 : ℝ) },
            D := {
                cylinder := Q,
                pressureOscillationIntegral := 0,
                value := 0,
                meanSubtracted := Q.positiveRadius,
                definitionFormula := 0 = 0 / (Q.radius * Q.radius),
                nonnegativeIntegral := 0 ≤ (0 : ℝ) },
            A := {
                cylinder := Q,
                essentialSupKineticEnergy := 0,
                value := 0,
                definitionFormula := 0 = 0 / Q.radius,
                nonnegativeEnergy := 0 ≤ (0 : ℝ) },
            E := {
                cylinder := Q,
                dissipationIntegral := 0,
                value := 0,
                definitionFormula := 0 = 0 / Q.radius,
                nonnegativeDissipation := 0 ≤ (0 : ℝ) } }

def rescaledSolutionOf
        (u : SuitableWeakSolution)
        (Q : ParabolicCylinder) :
        RescaledSolution :=
        { source := u,
            cylinder := Q }

axiom rescalingPreservesSuitability
    (u : SuitableWeakSolution)
    (Q : ParabolicCylinder) :
        (rescaledSolutionOf u Q).suitable

def CriticalScalingIdentity
    (u : SuitableWeakSolution)
    (Q : ParabolicCylinder) : Prop :=
  let q := criticalQuantitiesOn u Q
  q.C.cylinder = Q ∧ q.D.cylinder = Q ∧ q.A.cylinder = Q ∧ q.E.cylinder = Q

axiom criticalScalingIdentity
    (u : SuitableWeakSolution)
    (Q : ParabolicCylinder) :
    CriticalScalingIdentity u Q

end Hypostructure.Backends.NavierStokes.ProofSetup