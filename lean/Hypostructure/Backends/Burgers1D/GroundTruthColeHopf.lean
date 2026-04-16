import Hypostructure.Backends.Burgers1D.GroundTruthHeat

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Framework

noncomputable section

/-- Concrete Cole-Hopf transform data on the ground-truth carrier. The actual
formula will be supplied by the backend construction; the framework only needs
the transform, inverse, sectors, and round-trip laws. -/
structure ColeHopfTransform where
  toHeat : PeriodicH1State → PeriodicH1State
  fromHeat : PeriodicH1State → PeriodicH1State
  burgersSector : PeriodicH1State → Prop
  heatSector : PeriodicH1State → Prop
  toHeat_preservesH1 : ∀ u : PeriodicH1State,
    PeriodicH1State.IsPeriodicH1 u →
      PeriodicH1State.IsPeriodicH1 (toHeat u)
  fromHeat_preservesH1 : ∀ theta : PeriodicH1State,
    PeriodicH1State.IsPeriodicH1 theta →
      PeriodicH1State.IsPeriodicH1 (fromHeat theta)
  toHeat_sector : ∀ u : PeriodicH1State,
    burgersSector u → heatSector (toHeat u)
  fromHeat_sector : ∀ theta : PeriodicH1State,
    heatSector theta → burgersSector (fromHeat theta)
  left_inverse : ∀ u : PeriodicH1State,
    burgersSector u → fromHeat (toHeat u) = u
  right_inverse : ∀ theta : PeriodicH1State,
    heatSector theta → toHeat (fromHeat theta) = theta

def ColeHopfBurgersToHeatResidualTransfer
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu)
    (T : ColeHopfTransform) : Prop :=
  ∀ C : CertifiedBurgersLocalWindow nu,
    C.window.time = H.certified.window.time →
      T.burgersSector C.initial →
        H.certified.initial = T.toHeat C.initial →
          SolvesPeriodicHeatWeak nu (T.toHeat C.initial) H.certified.curve

def ColeHopfHeatToBurgersResidualTransfer
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu)
    (T : ColeHopfTransform) : Prop :=
  ∀ C : CertifiedHeatWindow nu,
    C.window.time = H.certified.window.time →
      T.heatSector C.initial →
        ∃ B : CertifiedBurgersLocalWindow nu,
          B.window.time = C.window.time ∧
            B.initial = T.fromHeat C.initial

def ColeHopfUniquenessTransfer
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu)
    (T : ColeHopfTransform) : Prop :=
  ∀ C : CertifiedBurgersLocalWindow nu,
    C.window.time = H.certified.window.time →
      T.burgersSector C.initial →
        H.certified.initial = T.toHeat C.initial →
          HeatWindowUniqueness nu H.certified

def ColeHopfWindowChartValid
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu)
    (T : ColeHopfTransform) : Prop :=
  ∀ C : CertifiedBurgersLocalWindow nu,
    C.window.time = H.certified.window.time →
      T.burgersSector C.initial →
        H.certified.initial = T.toHeat C.initial →
          T.heatSector H.certified.initial

def ColeHopfMapsBurgersWindow
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu)
    (T : ColeHopfTransform) : Prop :=
  ∀ C : CertifiedBurgersLocalWindow nu,
    C.window.time = H.certified.window.time →
      T.burgersSector C.initial →
        H.certified.initial = T.toHeat C.initial →
          C.window.time = H.certified.window.time ∧
            H.certified.initial = T.toHeat C.initial

def ColeHopfInverseMapsHeatWindow
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu)
    (T : ColeHopfTransform) : Prop :=
  ∀ C : CertifiedHeatWindow nu,
    C.window.time = H.certified.window.time →
      T.heatSector C.initial →
        ∃ B : CertifiedBurgersLocalWindow nu,
          B.window.time = C.window.time ∧
            B.initial = T.fromHeat C.initial

def ColeHopfHeatBadGermImage
    (T : ColeHopfTransform)
    (G : RouteLocalBadGermWindow) : RouteLocalBadGermWindow where
  time := G.time
  centerSpace := G.centerSpace
  profile := T.toHeat G.profile

theorem ColeHopfHeatBadGermImage.supportedInHeatWindow
    (T : ColeHopfTransform)
    {G : RouteLocalBadGermWindow}
    {W : HeatWindow}
    (hsupport : G.SupportedInHeatWindow W) :
    (ColeHopfHeatBadGermImage T G).SupportedInHeatWindow W :=
  hsupport

theorem ColeHopfHeatBadGermImage.admissible
    (T : ColeHopfTransform)
    (G : RouteLocalBadGermWindow)
    (hadm : G.Admissible) :
    (ColeHopfHeatBadGermImage T G).Admissible :=
  T.toHeat_preservesH1 G.profile hadm

def ColeHopfTransportsBadGerms
    (H : LocalHeatWindowCertificate nu)
    (T : ColeHopfTransform) : Prop :=
  ∀ G : RouteLocalBadGermWindow,
    G.SupportedInHeatWindow H.certified.window →
      G.Admissible →
        T.burgersSector G.profile →
          (ColeHopfHeatBadGermImage T G).SupportedInHeatWindow H.certified.window ∧
            (ColeHopfHeatBadGermImage T G).Admissible

theorem ColeHopfTransform.transportsBadGerms
    {nu : BurgersParameters}
    (T : ColeHopfTransform)
    (H : LocalHeatWindowCertificate nu) :
    ColeHopfTransportsBadGerms H T := by
  intro G hsupport hadm _hsector
  exact ⟨
    ColeHopfHeatBadGermImage.supportedInHeatWindow T hsupport,
    ColeHopfHeatBadGermImage.admissible T G hadm
  ⟩

/-- The local Cole-Hopf bridge required by the Lock. It exposes only certified
window data: chart validity, window mapping, inverse mapping, residual transfer,
uniqueness transfer, and bad-germ transport. -/
structure LocalColeHopfWindowBridge
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) where
  transform : ColeHopfTransform
  chart_valid : ColeHopfWindowChartValid nu H transform
  maps_window : ColeHopfMapsBurgersWindow nu H transform
  inverse_maps_window : ColeHopfInverseMapsHeatWindow nu H transform
  burgers_to_heat_residual_on_window :
    ColeHopfBurgersToHeatResidualTransfer nu H transform
  heat_to_burgers_residual_on_window :
    ColeHopfHeatToBurgersResidualTransfer nu H transform
  uniqueness_transfer_on_window : ColeHopfUniquenessTransfer nu H transform
  transports_bad_germs : ColeHopfTransportsBadGerms H transform

def LocalColeHopfWindowBridge.bridgeStatement
    {nu : BurgersParameters}
    {H : LocalHeatWindowCertificate nu}
    (B : LocalColeHopfWindowBridge nu H) : Prop :=
  ColeHopfWindowChartValid nu H B.transform ∧
    ColeHopfMapsBurgersWindow nu H B.transform ∧
    ColeHopfInverseMapsHeatWindow nu H B.transform ∧
    ColeHopfBurgersToHeatResidualTransfer nu H B.transform ∧
    ColeHopfHeatToBurgersResidualTransfer nu H B.transform ∧
    ColeHopfUniquenessTransfer nu H B.transform ∧
    ColeHopfTransportsBadGerms H B.transform

def localColeHopfFrameworkCertificate
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu)
    (B : LocalColeHopfWindowBridge nu H) :
    ColeHopfCertificate :=
  { node := .lock
    payload :=
      { transformName := "ground-truth periodic Cole-Hopf transform"
        targetSemigroup := "ground-truth periodic heat semigroup"
        bridgeStatement := B.bridgeStatement }
    meaning := B.bridgeStatement }

theorem localColeHopfFrameworkCertificate_sound
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu)
    (B : LocalColeHopfWindowBridge nu H) :
    (localColeHopfFrameworkCertificate nu H B).meaning := by
  exact ⟨
    B.chart_valid,
    B.maps_window,
    B.inverse_maps_window,
    B.burgers_to_heat_residual_on_window,
    B.heat_to_burgers_residual_on_window,
    B.uniqueness_transfer_on_window,
    B.transports_bad_germs
  ⟩

/-- All-window Cole-Hopf backend data. This is a construction source for local
bridge certificates, not the exported `K_ColeHopf^+` payload. A concrete
periodic Cole-Hopf implementation should prove these fields from its transform
formula and then restrict them with `windowBridge`. -/
structure PeriodicColeHopfBackend
    (nu : BurgersParameters) where
  transform : ColeHopfTransform
  chart_valid : ∀ H : LocalHeatWindowCertificate nu,
    ColeHopfWindowChartValid nu H transform
  maps_window : ∀ H : LocalHeatWindowCertificate nu,
    ColeHopfMapsBurgersWindow nu H transform
  inverse_maps_window : ∀ H : LocalHeatWindowCertificate nu,
    ColeHopfInverseMapsHeatWindow nu H transform
  burgers_to_heat_residual_on_window : ∀ H : LocalHeatWindowCertificate nu,
    ColeHopfBurgersToHeatResidualTransfer nu H transform
  heat_to_burgers_residual_on_window : ∀ H : LocalHeatWindowCertificate nu,
    ColeHopfHeatToBurgersResidualTransfer nu H transform
  uniqueness_transfer_on_window : ∀ H : LocalHeatWindowCertificate nu,
    ColeHopfUniquenessTransfer nu H transform

def PeriodicColeHopfBackend.windowBridge
    {nu : BurgersParameters}
    (B : PeriodicColeHopfBackend nu)
    (H : LocalHeatWindowCertificate nu) :
    LocalColeHopfWindowBridge nu H where
  transform := B.transform
  chart_valid := B.chart_valid H
  maps_window := B.maps_window H
  inverse_maps_window := B.inverse_maps_window H
  burgers_to_heat_residual_on_window :=
    B.burgers_to_heat_residual_on_window H
  heat_to_burgers_residual_on_window :=
    B.heat_to_burgers_residual_on_window H
  uniqueness_transfer_on_window := B.uniqueness_transfer_on_window H
  transports_bad_germs := B.transform.transportsBadGerms H

theorem PeriodicColeHopfBackend.windowBridge_sound
    {nu : BurgersParameters}
    (B : PeriodicColeHopfBackend nu)
    (H : LocalHeatWindowCertificate nu) :
    (B.windowBridge H).bridgeStatement :=
  localColeHopfFrameworkCertificate_sound nu H (B.windowBridge H)

theorem coleHopf_toHeat_preservesH1
    {T : ColeHopfTransform}
    {u : PeriodicH1State}
    (hu : PeriodicH1State.IsPeriodicH1 u) :
    PeriodicH1State.IsPeriodicH1 (T.toHeat u) :=
  T.toHeat_preservesH1 u hu

theorem coleHopf_fromHeat_preservesH1
    {T : ColeHopfTransform}
    {theta : PeriodicH1State}
    (htheta : PeriodicH1State.IsPeriodicH1 theta) :
    PeriodicH1State.IsPeriodicH1 (T.fromHeat theta) :=
  T.fromHeat_preservesH1 theta htheta

end

end Hypostructure.Backends.Burgers1D
