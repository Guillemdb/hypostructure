import Hypostructure.Backends.Burgers1D.GroundTruthCoreFactory
import Hypostructure.Backends.Burgers1D.GroundTruthUpgrade
import Hypostructure.Framework.Rigor
import Hypostructure.Literature.ColeHopf.PeriodicBurgers1D
import Hypostructure.Literature.Heat.Periodic1D

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Framework

noncomputable section

/-- Explicit remaining backend permits for the ground-truth route. These are
not proved yet by the current backend. Supplying this structure is exactly the
remaining work needed to turn the migrated route into the full theorem in
`docs/source/dataset/burgers_1d.md`. -/
structure BurgersGroundTruthBackendPermits
    (nu : BurgersParameters) where
  heat : LocalHeatWindowCertificate nu
  coleHopf : LocalColeHopfWindowBridge nu heat

/-- TODO: construct the arbitrary-data windowed PDE-local analytic input
package. This is the genuinely local Burgers input boundary: it contains only a
certified Burgers window and windowed local estimates for the exact admissible
initial datum. -/
axiom burgers_axiom_corePDELocalWindowInputsFor :
  ∀ (nu : BurgersParameters)
    (u0 : PeriodicH1State),
    PeriodicH1State.IsPeriodicH1 u0 →
      Σ W : BurgersWindow, BurgersCorePDELocalWindowInputsFor nu u0 W

def burgers_corePDELocalWindowInputsFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    Σ W : BurgersWindow, BurgersCorePDELocalWindowInputsFor nu u0 W :=
  burgers_axiom_corePDELocalWindowInputsFor nu u0 hu0

def burgers_corePDELocalAnalyticInputsFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    BurgersCorePDELocalAnalyticInputsFor nu u0 :=
  let localInput := burgers_corePDELocalWindowInputsFor nu u0 hu0
  localInput.2.toAnalyticInputsFor

def burgers_coreLocalAnalyticInputsFromPDEFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    BurgersCoreLocalAnalyticInputs nu :=
  (burgers_corePDELocalAnalyticInputsFor nu u0 hu0).toInputs.toLocalAnalyticInputs

def burgers_coreCertificateBundleFromLocalAnalyticInputsFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    BurgersGroundTruthCoreCertificateBundle nu :=
  (burgers_coreLocalAnalyticInputsFromPDEFor nu u0 hu0).toCoreBundle

theorem burgers_coreCertificateBundleFromLocalAnalyticInputsFor_u0
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    (burgers_coreCertificateBundleFromLocalAnalyticInputsFor
      nu u0 hu0).u0 = u0 :=
  rfl

theorem burgers_coreCertificateBundleFromLocalAnalyticInputsFor_sound
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    (burgers_coreCertificateBundleFromLocalAnalyticInputsFor
      nu u0 hu0).allCertificatesSound :=
  (burgers_coreLocalAnalyticInputsFromPDEFor nu u0 hu0).toCoreBundle_sound

def burgers_periodicHeatSemigroupBackendFromLiterature
    (nu : BurgersParameters) :
    PeriodicHeatSemigroupBackend nu :=
  Hypostructure.Literature.Heat.Periodic1D.periodicHeat1D_semigroupBackend_literature nu

def BurgersGroundTruthCoreCertificateBundle.heatWindow
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) : HeatWindow where
  time := B.energy.window

def burgers_localHeatWindowCertificateFromBackend
    (nu : BurgersParameters)
    (bundle : BurgersGroundTruthCoreCertificateBundle nu)
    (backend : PeriodicHeatSemigroupBackend nu) :
    LocalHeatWindowCertificate nu :=
  backend.windowCertificate
    bundle.u0
    bundle.heatWindow
    bundle.u0_isPeriodicH1

def burgers_periodicColeHopfBackendFromLiterature
    (nu : BurgersParameters) :
    PeriodicColeHopfBackend nu :=
  Hypostructure.Literature.ColeHopf.PeriodicBurgers1D.periodicBurgers1D_coleHopfBackend_literature nu

def burgers_localColeHopfWindowBridgeFromBackend
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu)
    (backend : PeriodicColeHopfBackend nu) :
    LocalColeHopfWindowBridge nu H :=
  backend.windowBridge H

def BurgersGroundTruthBackendPermits.fromAxioms
    (nu : BurgersParameters)
    (bundle : BurgersGroundTruthCoreCertificateBundle nu) :
    BurgersGroundTruthBackendPermits nu :=
  let H : LocalHeatWindowCertificate nu :=
    burgers_localHeatWindowCertificateFromBackend
      nu bundle (burgers_periodicHeatSemigroupBackendFromLiterature nu)
  { heat := H
    coleHopf :=
      burgers_localColeHopfWindowBridgeFromBackend
        nu H (burgers_periodicColeHopfBackendFromLiterature nu) }

/-- User-facing migrated theorem claim: the documented final certificate is
recorded, and the classical ground-truth PDE theorem follows from the explicit
ground-truth backend permits. -/
structure BurgersGroundTruthDatasetClaim
    (nu : BurgersParameters) where
  finalCertificateRecorded :
    burgersGroundTruthFinalCertificateChain.designatedGoal ∈
      burgersGroundTruthFinalCertificateChain.certificates
  routeValidity :
    burgersGroundTruthRunValidity.meetsTemplateCompletionCriteria
  globalRegularity :
    BurgersGroundTruthGlobalRegularityStatement nu

theorem burgers_groundTruth_final_certificate_recorded :
    burgersGroundTruthFinalCertificateChain.designatedGoal ∈
      burgersGroundTruthFinalCertificateChain.certificates :=
  burgersGroundTruthFinalChain_contains_goal

theorem burgers_groundTruth_global_regularized
    (nu : BurgersParameters)
    (provider : BurgersPointwiseLocalEstimateProvider nu) :
    BurgersGroundTruthGlobalRegularityStatement nu :=
  burgers_globalRegularity_from_pointwise_local_estimates
    nu provider.localInput provider.localChain

def burgers_groundTruth_dataset_template_claim
    (nu : BurgersParameters)
    (provider : BurgersPointwiseLocalEstimateProvider nu) :
    TraceBackedTargetClaim
      burgersGroundTruthRoute
      (BurgersGroundTruthGlobalRegularityStatement nu) :=
  burgersGroundTruthRouteProof.targetClaim
    (burgers_groundTruth_global_regularized nu provider)

def burgers_groundTruth_dataset_theorem
    (nu : BurgersParameters)
    (provider : BurgersPointwiseLocalEstimateProvider nu) :
    BurgersGroundTruthDatasetClaim nu where
  finalCertificateRecorded := burgers_groundTruth_final_certificate_recorded
  routeValidity := burgersGroundTruthRunValidity_holds
  globalRegularity := burgers_groundTruth_global_regularized nu provider

def burgers_pointwiseAnalyticUpgradeInputFromAxioms
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    BurgersAnalyticUpgradeInputFor nu u0 :=
  let bundle :=
    burgers_coreCertificateBundleFromLocalAnalyticInputsFor nu u0 hu0
  let permits := BurgersGroundTruthBackendPermits.fromAxioms nu bundle
  BurgersAnalyticUpgradeInputFor.fromLocalCertificates
    nu u0 bundle
    (burgers_coreCertificateBundleFromLocalAnalyticInputsFor_u0 nu u0 hu0)
    permits.heat permits.coleHopf

def burgers_pointwiseLocalEstimateProviderFromAxioms
    (nu : BurgersParameters) :
    BurgersPointwiseLocalEstimateProvider nu where
  localInput := burgers_pointwiseAnalyticUpgradeInputFromAxioms nu
  localChain := fun u0 hu0 =>
    burgers_localContinuationLocalizationChainFromAxioms
      nu u0 (burgers_pointwiseAnalyticUpgradeInputFromAxioms nu u0 hu0)

/-- Closed end-to-end theorem with every remaining non-framework analytic
piece exposed as a named axiom. This is the temporary checkpoint theorem: once
the axioms below are discharged, this becomes the desired ground-truth theorem. -/
def burgers_groundTruth_dataset_theorem_from_axioms
    (nu : BurgersParameters) :
    BurgersGroundTruthDatasetClaim nu :=
  burgers_groundTruth_dataset_theorem
    nu
    (burgers_pointwiseLocalEstimateProviderFromAxioms nu)

def burgersGroundTruthAxiomBoundary : List String :=
  [ "burgers_axiom_corePDELocalWindowInputsFor"
  , "periodicHeat1D_semigroupBackend_literature"
  , "periodicBurgers1D_coleHopfBackend_literature"
  , "burgers_axiom_localUniquenessOnOverlaps"
  , "burgers_axiom_localContinuationCriterion"
  , "burgers_axiom_missingRegularityProducesMaximalNonextendableWindow"
  , "burgers_axiom_nonextendableWindowProducesFiniteH1Obstruction"
  , "burgers_axiom_classifiedObstructionInColeHopfChart"
  , "burgers_axiom_classifiedObstructionHeatImageSupported"
  , "burgers_axiom_classifiedObstructionHeatImageCapacityFails"
  ]

def burgersGroundTruthRigorBoundary : List RigorBoundaryItem :=
  [ { name := "APosterioriLocalization.point"
      layer := .framework
      description := "Reusable hypostructure proof: Lock plus pointwise failure-localization implies the target witness for the same object." }
  , { name := "burgers_globalRegularity_from_pointwise_local_estimates"
      layer := .framework
      description := "Framework assembly theorem turning pointwise local packages and local continuation/localization chains into the classical globally quantified Burgers target." }
  , { name := "periodicHeat1D_semigroupBackend_literature"
      layer := .literature
      description := "Reusable standard 1D periodic heat existence, uniqueness, smoothing, and contraction package." }
  , { name := "periodicBurgers1D_coleHopfBackend_literature"
      layer := .literature
      description := "Reusable Cole-Hopf transform/bridge package for periodic 1D Burgers-like equations." }
  , { name := "burgers_axiom_corePDELocalWindowInputsFor"
      layer := .problem
      description := "Problem-specific pointwise windowed Burgers PDE certificate factory for each admissible initial datum." }
  , { name := "burgers_axiom_localUniquenessOnOverlaps"
      layer := .problem
      description := "Problem-specific local uniqueness theorem on certified Burgers windows." }
  , { name := "burgers_axiom_localContinuationCriterion"
      layer := .problem
      description := "Problem-specific local continuation criterion: no local bad obstruction implies window extendability." }
  , { name := "burgers_axiom_missingRegularityProducesMaximalNonextendableWindow"
      layer := .problem
      description := "Problem-specific maximal-window principle: missing regularity yields a finite nonextendable local window." }
  , { name := "burgers_axiom_nonextendableWindowProducesFiniteH1Obstruction"
      layer := .problem
      description := "Problem-specific extraction of a finite-time H1 bad germ from a nonextendable local window." }
  , { name := "burgers_axiom_classifiedObstructionInColeHopfChart"
      layer := .problem
      description := "Problem-specific local chart theorem placing classified Burgers obstructions in the Cole-Hopf sector." }
  , { name := "burgers_axiom_classifiedObstructionHeatImageSupported"
      layer := .problem
      description := "Problem-specific local support theorem for the heat image of a classified Burgers obstruction." }
  , { name := "burgers_axiom_classifiedObstructionHeatImageCapacityFails"
      layer := .problem
      description := "Problem-specific local transfer theorem: the heat image of a classified obstruction is capacity-failing." }
  ]

def burgersGroundTruthLiteratureBoundary : List String :=
  [ "periodicHeat1D_semigroupBackend_literature"
  , "periodicBurgers1D_coleHopfBackend_literature"
  ]

def burgersGroundTruthProblemSpecificBoundary : List String :=
  [ "burgers_axiom_corePDELocalWindowInputsFor"
  , "burgers_axiom_localUniquenessOnOverlaps"
  , "burgers_axiom_localContinuationCriterion"
  , "burgers_axiom_missingRegularityProducesMaximalNonextendableWindow"
  , "burgers_axiom_nonextendableWindowProducesFiniteH1Obstruction"
  , "burgers_axiom_classifiedObstructionInColeHopfChart"
  , "burgers_axiom_classifiedObstructionHeatImageSupported"
  , "burgers_axiom_classifiedObstructionHeatImageCapacityFails"
  ]

/-- The migrated route deliberately no longer proves global regularity from any
old scaffold evolution package. It needs the concrete backend permits above. -/
def burgersGroundTruthRemainingBackendWork : List String :=
  [ "Discharge burgers_axiom_corePDELocalWindowInputsFor by constructing the pointwise arbitrary-data windowed PDE certificate factory for each admissible u0."
  , "Discharge periodicBurgers1D_coleHopfBackend_literature by constructing the concrete periodic Cole-Hopf transform and inverse on nonconstant data."
  , "Discharge periodicHeat1D_semigroupBackend_literature by constructing a periodic heat semigroup on the whole PeriodicH1State carrier, or by replacing it with an equivalent local-window heat literature theorem."
  , "Extend the finite-time H1 bad-pattern library only if the dataset is expanded beyond the current documented finite-time H1 blow-up template."
  , "Discharge the local continuation/localization chain axioms: local uniqueness, continuation criterion, maximal nonextendable window, finite H1 germ extraction, and local Cole-Hopf/heat obstruction transfer."
  ]

end

end Hypostructure.Backends.Burgers1D
