import Hypostructure.Backends.Burgers1D.GroundTruthRun
import Hypostructure.Backends.Burgers1D.GroundTruthColeHopf

namespace Hypostructure.Backends.Burgers1D

noncomputable section

/-- A route-relevant bad morphism candidate. This is the 0-truncated,
framework-checkable Lock target: the candidate is a finite-time `H¹` blow-up
bad-germ window, and the certified bad-pattern library classifies it before E5
transports it to the heat side. -/
structure BurgersBadMorphismCandidate
    (nu : BurgersParameters)
    (bundle : BurgersGroundTruthCoreCertificateBundle nu)
    (H : LocalHeatWindowCertificate nu)
    (B : LocalColeHopfWindowBridge nu H) where
  germ : BurgersBadGerm
  finite_time_h1_blowup : FiniteTimeH1BlowUpBadGerm germ
  heat_available : H.certificateStatement
  bridge_available : B.bridgeStatement
  library_complete : BurgersBadPatternLibrary.Complete bundle.badPatternLibrary.library
  burgers_chart : B.transform.burgersSector germ.profile
  heat_image_supported :
    (ColeHopfHeatBadGermImage B.transform germ.routeWindow).SupportedInHeatWindow
      H.certified.window
  heat_image_capacity_fails :
    ¬ (ColeHopfHeatBadGermImage B.transform germ.routeWindow).LocalCapacity

def BurgersBadMorphismExists
    (nu : BurgersParameters)
    (bundle : BurgersGroundTruthCoreCertificateBundle nu)
    (H : LocalHeatWindowCertificate nu)
    (B : LocalColeHopfWindowBridge nu H) : Prop :=
  Nonempty (BurgersBadMorphismCandidate nu bundle H B)

def LockBlocksBurgersBadGerms
    (nu : BurgersParameters)
    (bundle : BurgersGroundTruthCoreCertificateBundle nu)
    (H : LocalHeatWindowCertificate nu)
    (B : LocalColeHopfWindowBridge nu H) : Prop :=
  ¬ BurgersBadMorphismExists nu bundle H B

theorem lockBlocksBadGermsFromLocalCertificates
    (nu : BurgersParameters)
    (bundle : BurgersGroundTruthCoreCertificateBundle nu)
    (H : LocalHeatWindowCertificate nu)
    (B : LocalColeHopfWindowBridge nu H) :
    LockBlocksBurgersBadGerms nu bundle H B := by
  intro hbad
  rcases hbad with ⟨candidate⟩
  rcases candidate.library_complete candidate.finite_time_h1_blowup with
    ⟨pattern, _hpattern_mem, hpattern_accepts⟩
  have hroute_admissible : candidate.germ.routeWindow.Admissible :=
    pattern.accepts_routeWindow hpattern_accepts
  have htransport :=
    B.transports_bad_germs
      candidate.germ.routeWindow
      candidate.heat_image_supported
      hroute_admissible
      candidate.burgers_chart
  exact H.no_capacity_failing_heat_bad_germ
    (ColeHopfHeatBadGermImage B.transform candidate.germ.routeWindow)
    htransport.1
    htransport.2
    candidate.heat_image_capacity_fails

end

end Hypostructure.Backends.Burgers1D
