import Hypostructure.Backends.Burgers1D.GroundTruthColeHopf

namespace Hypostructure.Literature.ColeHopf.PeriodicBurgers1D

open Hypostructure.Backends.Burgers1D

noncomputable section

/-- Reusable periodic Cole-Hopf theory package for the ground-truth Burgers
route.

The previous boundary exposed the transform, chart validity, window transport,
two residual-transfer directions, uniqueness transfer, and `H¹`-domain theorem
as separate axioms. Bundling them records that these are one reusable
Cole-Hopf backend theorem for the fixed viscosity parameters. -/
structure PeriodicBurgers1DColeHopfTheory
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
  h1_in_burgersPDEDomain : ∀ u : PeriodicH1State,
    PeriodicH1State.IsPeriodicH1 u →
      transform.burgersPDEDomain u

/-- Single reusable Cole-Hopf literature boundary for the periodic Burgers
route. -/
axiom periodicBurgers1D_coleHopfTheory_literature :
  ∀ nu : BurgersParameters, PeriodicBurgers1DColeHopfTheory nu

def periodicBurgers1D_coleHopfTransform_literature
    (nu : BurgersParameters) : ColeHopfTransform :=
  (periodicBurgers1D_coleHopfTheory_literature nu).transform

theorem periodicBurgers1D_coleHopfChartValid_literature
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) :
    ColeHopfWindowChartValid nu H
      (periodicBurgers1D_coleHopfTransform_literature nu) :=
  (periodicBurgers1D_coleHopfTheory_literature nu).chart_valid H

theorem periodicBurgers1D_coleHopfMapsWindow_literature
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) :
    ColeHopfMapsBurgersWindow nu H
      (periodicBurgers1D_coleHopfTransform_literature nu) :=
  (periodicBurgers1D_coleHopfTheory_literature nu).maps_window H

theorem periodicBurgers1D_coleHopfInverseMapsWindow_literature
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) :
    ColeHopfInverseMapsHeatWindow nu H
      (periodicBurgers1D_coleHopfTransform_literature nu) :=
  (periodicBurgers1D_coleHopfTheory_literature nu).inverse_maps_window H

theorem periodicBurgers1D_coleHopfBurgersToHeatResidual_literature
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) :
    ColeHopfBurgersToHeatResidualTransfer nu H
      (periodicBurgers1D_coleHopfTransform_literature nu) :=
  (periodicBurgers1D_coleHopfTheory_literature nu).burgers_to_heat_residual_on_window H

theorem periodicBurgers1D_coleHopfHeatToBurgersResidual_literature
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) :
    ColeHopfHeatToBurgersResidualTransfer nu H
      (periodicBurgers1D_coleHopfTransform_literature nu) :=
  (periodicBurgers1D_coleHopfTheory_literature nu).heat_to_burgers_residual_on_window H

theorem periodicBurgers1D_coleHopfUniquenessTransfer_literature
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) :
    ColeHopfUniquenessTransfer nu H
      (periodicBurgers1D_coleHopfTransform_literature nu) :=
  (periodicBurgers1D_coleHopfTheory_literature nu).uniqueness_transfer_on_window H

theorem periodicBurgers1D_h1InBurgersPDEDomain_literature
    (nu : BurgersParameters)
    (u : PeriodicH1State)
    (hu : PeriodicH1State.IsPeriodicH1 u) :
    (periodicBurgers1D_coleHopfTransform_literature nu).burgersPDEDomain u :=
  (periodicBurgers1D_coleHopfTheory_literature nu).h1_in_burgersPDEDomain u hu

def periodicBurgers1D_coleHopfBackend_literature
    (nu : BurgersParameters) :
    PeriodicColeHopfH1DomainBackend nu where
  backend :=
    { transform := periodicBurgers1D_coleHopfTransform_literature nu
      chart_valid := periodicBurgers1D_coleHopfChartValid_literature nu
      maps_window := periodicBurgers1D_coleHopfMapsWindow_literature nu
      inverse_maps_window :=
        periodicBurgers1D_coleHopfInverseMapsWindow_literature nu
      burgers_to_heat_residual_on_window :=
        periodicBurgers1D_coleHopfBurgersToHeatResidual_literature nu
      heat_to_burgers_residual_on_window :=
        periodicBurgers1D_coleHopfHeatToBurgersResidual_literature nu
      uniqueness_transfer_on_window :=
        periodicBurgers1D_coleHopfUniquenessTransfer_literature nu }
  h1_in_burgersPDEDomain :=
    periodicBurgers1D_h1InBurgersPDEDomain_literature nu

def periodicBurgers1D_coleHopfBackend
    (nu : BurgersParameters) : PeriodicColeHopfBackend nu :=
  (periodicBurgers1D_coleHopfBackend_literature nu).backend

theorem periodicBurgers1D_h1_in_burgersPDEDomain_literature
    (nu : BurgersParameters)
    (u : PeriodicH1State)
    (hu : PeriodicH1State.IsPeriodicH1 u) :
    (periodicBurgers1D_coleHopfBackend nu).transform.burgersPDEDomain u :=
  (periodicBurgers1D_coleHopfBackend_literature nu).h1_in_burgersPDEDomain u hu

def periodicBurgers1D_localColeHopfWindowBridge
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) :
    LocalColeHopfWindowBridge nu H :=
  (periodicBurgers1D_coleHopfBackend nu).windowBridge H

theorem periodicBurgers1D_localColeHopfWindowBridge_sound
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) :
  (periodicBurgers1D_localColeHopfWindowBridge nu H).bridgeStatement :=
  PeriodicColeHopfBackend.windowBridge_sound
    (periodicBurgers1D_coleHopfBackend nu) H

end

end Hypostructure.Literature.ColeHopf.PeriodicBurgers1D
