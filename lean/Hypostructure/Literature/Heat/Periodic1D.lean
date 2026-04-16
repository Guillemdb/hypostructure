import Hypostructure.Backends.Burgers1D.GroundTruthHeat

namespace Hypostructure.Literature.Heat.Periodic1D

open Hypostructure.Backends.Burgers1D

noncomputable section

/-- Literature boundary for the one-dimensional periodic heat equation on the
current `PeriodicH1State` carrier. This is reusable analytic infrastructure,
not Burgers-specific nonlinear regularity: it packages the standard linear heat
existence, uniqueness, contraction, and positive-time smoothing theory.

TODO: replace this axiom by a mathlib-backed Fourier/semigroup construction. -/
axiom periodicHeat1D_semigroupBackend_literature :
  ∀ nu : BurgersParameters, PeriodicHeatSemigroupBackend nu

def periodicHeat1D_localWindowCertificate
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    LocalHeatWindowCertificate nu :=
  (periodicHeat1D_semigroupBackend_literature nu).windowCertificate
    u0 W hu0

theorem periodicHeat1D_localWindowCertificate_sound
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    (periodicHeat1D_localWindowCertificate nu u0 W hu0).certificateStatement :=
  localHeatSmoothingFrameworkCertificate_sound
    nu (periodicHeat1D_localWindowCertificate nu u0 W hu0)

end

end Hypostructure.Literature.Heat.Periodic1D
