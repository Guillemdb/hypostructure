import Hypostructure.Backends.Burgers1D.GroundTruthColeHopf

namespace Hypostructure.Literature.ColeHopf.PeriodicBurgers1D

open Hypostructure.Backends.Burgers1D

noncomputable section

/-- Literature boundary for the periodic one-dimensional Cole-Hopf transform
used by Burgers/KPZ-type arguments. This is not part of the hypostructure
framework, but it is reusable analytic infrastructure for Burgers-like PDE
backends.

TODO: replace this axiom by a mathlib-backed construction of the transform,
inverse chart, residual transfer, and uniqueness transfer. -/
axiom periodicBurgers1D_coleHopfBackend_literature :
  ∀ nu : BurgersParameters, PeriodicColeHopfBackend nu

def periodicBurgers1D_localColeHopfWindowBridge
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) :
    LocalColeHopfWindowBridge nu H :=
  (periodicBurgers1D_coleHopfBackend_literature nu).windowBridge H

theorem periodicBurgers1D_localColeHopfWindowBridge_sound
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) :
    (periodicBurgers1D_localColeHopfWindowBridge nu H).bridgeStatement :=
  PeriodicColeHopfBackend.windowBridge_sound
    (periodicBurgers1D_coleHopfBackend_literature nu) H

end

end Hypostructure.Literature.ColeHopf.PeriodicBurgers1D
