import Hypostructure.Backends.Burgers1D.GroundTruthCoreFactory
import Hypostructure.Backends.Burgers1D.GroundTruthUpgrade
import Hypostructure.Framework.Rigor
import Hypostructure.Literature.Burgers.Periodic1D
import Hypostructure.Literature.ColeHopf.PeriodicBurgers1D
import Hypostructure.Literature.Heat.Periodic1D

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Framework

noncomputable section

set_option maxHeartbeats 4000000

/-- Explicit remaining backend permits for the ground-truth route. These are
not proved yet by the current backend. Supplying this structure is exactly the
remaining work needed to turn the migrated route into the full theorem in
`docs/source/dataset/burgers_1d.md`. -/
structure BurgersGroundTruthBackendPermits
    (nu : BurgersParameters) where
  heat : LocalHeatWindowCertificate nu
  coleHopf : LocalColeHopfWindowBridge nu heat
  coleHopf_h1_in_burgersPDEDomain : ∀ u : PeriodicH1State,
    PeriodicH1State.IsPeriodicH1 u →
      coleHopf.transform.burgersPDEDomain u

def burgers_corePDELocalWindowInputsFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    Σ W : BurgersWindow, BurgersCorePDELocalWindowInputsFor nu u0 W :=
  Hypostructure.Literature.Burgers.Periodic1D.periodicBurgers1D_localWindowInputs_literature
    nu u0 hu0

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

def BurgersGroundTruthCoreCertificateBundle.heatWindow
    {nu : BurgersParameters}
    (B : BurgersGroundTruthCoreCertificateBundle nu) : HeatWindow where
  time := B.energy.window

def burgers_localHeatWindowCertificateFromLiterature
    (nu : BurgersParameters)
    (bundle : BurgersGroundTruthCoreCertificateBundle nu) :
    LocalHeatWindowCertificate nu :=
  Hypostructure.Literature.Heat.Periodic1D.periodicHeat1D_localWindowCertificate
    nu bundle.u0 bundle.heatWindow
    bundle.u0_isPeriodicH1

def burgers_periodicColeHopfBackendFromLiterature
    (nu : BurgersParameters) :
    PeriodicColeHopfBackend nu :=
  Hypostructure.Literature.ColeHopf.PeriodicBurgers1D.periodicBurgers1D_coleHopfBackend nu

def burgers_periodicColeHopfH1DomainBackendFromLiterature
    (nu : BurgersParameters) :
    PeriodicColeHopfH1DomainBackend nu :=
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
    burgers_localHeatWindowCertificateFromLiterature nu bundle
  let CH : PeriodicColeHopfH1DomainBackend nu :=
    burgers_periodicColeHopfH1DomainBackendFromLiterature nu
  let B : LocalColeHopfWindowBridge nu H :=
    burgers_localColeHopfWindowBridgeFromBackend nu H CH.backend
  { heat := H
    coleHopf := B
    coleHopf_h1_in_burgersPDEDomain := by
      intro u hu
      exact CH.h1_in_burgersPDEDomain u hu }

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
    permits.heat permits.coleHopf permits.coleHopf_h1_in_burgersPDEDomain

def burgers_localUniquenessOnOverlapsFromLiterature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0) :
    BurgersCertifiedLocalEvolutionFor nu u0 input →
      BurgersLocalUniquenessOnOverlaps nu u0 input := by
  intro localEvolution
  refine ⟨?_⟩
  intro v hv t ht
  exact Hypostructure.Literature.Burgers.Periodic1D.periodicBurgers1D_localUniquenessOnWindow_literature
    nu u0 input.input.bundle.solution v input.input.bundle.window
    localEvolution.solves_on_window hv t ht

def periodicBurgers1D_missingRegularityProducesObstructionWindow_fromContinuationLiterature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (localEvolution : BurgersCertifiedLocalEvolutionFor nu u0 input)
    (localUniqueness : BurgersLocalUniquenessOnOverlaps nu u0 input) :
    PeriodicH1State.IsPeriodicH1 u0 →
      ¬ BurgersRegularityWitness nu u0 →
        BurgersLocalizedObstructionWindow nu u0 input := by
  intro hu0 hmissing
  let rawWindow :=
    Hypostructure.Literature.Burgers.Periodic1D.periodicBurgers1D_missingRegularityProducesRawFailureWindow_literature
      nu u0 input localEvolution localUniqueness hu0 hmissing
  exact rawWindow.toLocalizedObstructionWindow
    (Hypostructure.Literature.Burgers.Periodic1D.periodicBurgers1D_rawFailureWindow_nonextendable_literature
      nu u0 input localEvolution localUniqueness rawWindow)

def burgers_missingRegularityProducesMaximalNonextendableWindow_fromObstructionWindow
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (localEvolution : BurgersCertifiedLocalEvolutionFor nu u0 input)
    (localUniqueness : BurgersLocalUniquenessOnOverlaps nu u0 input) :
    PeriodicH1State.IsPeriodicH1 u0 →
      ¬ BurgersRegularityWitness nu u0 →
        BurgersMaximalNonextendableWindow
          nu u0 input (burgers_canonicalLocalContinuationCriterion nu u0 input) := by
  intro hu0 hmissing
  exact (periodicBurgers1D_missingRegularityProducesObstructionWindow_fromContinuationLiterature
    nu u0 input localEvolution localUniqueness hu0 hmissing).toMaximalNonextendableWindow

def burgers_localContinuationLocalizationChainFromAxioms
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0) :
    BurgersLocalContinuationLocalizationChain nu u0 input :=
  let localEvolution := input.certifiedLocalEvolution
  let localUniqueness :=
    burgers_localUniquenessOnOverlapsFromLiterature nu u0 input localEvolution
  let continuation :=
    burgers_canonicalLocalContinuationCriterion nu u0 input
  { localEvolution := localEvolution
    localUniqueness := localUniqueness
    continuation := continuation
    maximalFailure :=
      burgers_missingRegularityProducesMaximalNonextendableWindow_fromObstructionWindow
        nu u0 input localEvolution localUniqueness
    obstructionInColeHopfChart :=
      burgers_obstructionInColeHopfChart_fromDomain nu u0 input continuation
    heatImageSupported :=
      burgers_obstructionHeatImageSupported_fromWindowSupport
        nu u0 input continuation
    heatImageForbidden :=
      burgers_obstructionTransfersToHeatForbidden_fromSmoothnessReflection
        nu u0 input continuation }

theorem burgers_pointwise_badMorphismLocalization_fromLocalContinuationChainAxioms
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0) :
    BurgersAposterioriBadMorphismLocalizationFor nu u0 input :=
  burgers_pointwise_badMorphismLocalization_fromLocalContinuationChain
    nu u0 input (burgers_localContinuationLocalizationChainFromAxioms nu u0 input)

theorem burgersPointwiseRegularity_fromLocalContinuationChainAxioms
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (input : BurgersAnalyticUpgradeInputFor nu u0) :
    BurgersRegularityWitness nu u0 :=
  burgersPointwiseRegularity_fromAposterioriLocalization
    nu u0 hu0 input
    (burgers_pointwise_badMorphismLocalization_fromLocalContinuationChainAxioms
      nu u0 input)

def burgersAnalyticPointwiseUpgradeTheoremFromAposterioriAxiom
    (nu : BurgersParameters) :
    BurgersAnalyticPointwiseUpgradeTheorem nu where
  upgrade := burgersPointwiseRegularity_fromLocalContinuationChainAxioms nu

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
  [ "periodicBurgers1D_localWindowTheory_literature"
  , "periodicHeat1D_weakSolutionValueFourierEvolution_literature"
  , "periodicBurgers1D_coleHopfTheory_literature"
  , "periodicBurgers1D_continuationTheory_literature"
  ]

def burgersGroundTruthRigorBoundary : List RigorBoundaryItem :=
  [ { name := "APosterioriLocalization.point"
      layer := .framework
      description := "Reusable hypostructure proof: Lock plus pointwise failure-localization implies the target witness for the same object." }
  , { name := "burgers_globalRegularity_from_pointwise_local_estimates"
      layer := .framework
      description := "Framework assembly theorem turning pointwise local packages and local continuation/localization chains into the classical globally quantified Burgers target." }
  , { name := "periodicHeat1D_fourierModeTheory_literature"
      layer := .framework
      description := "Mathlib-backed periodic heat Fourier multiplier package; Lean proves zero-time identity, positivity, positive-time contraction, square contraction, and time-addition composition for exp(-nu(2*pi*n)^2*t)." }
  , { name := "periodicHeat1D_fourierCoefficientTheory_literature"
      layer := .framework
      description := "Mathlib-backed coefficient-level heat package; Lean proves evolved Fourier coefficient initial values, semigroup law, derivative compatibility, all-mode summability preservation, and value/derivative l2 contraction." }
  , { name := "periodicHeat1D_l2ReconstructionTheory_literature"
      layer := .framework
      description := "Mathlib-backed Hilbert-space reconstruction package; Lean constructs the L2 value and weak-derivative reconstructions from the evolved Fourier coefficients and proves coefficient recovery, L2 Fourier-series convergence, and Parseval identities for those reconstructed L2 objects." }
  , { name := "periodicHeat1D_fourierH1ReconstructionTheory_literature"
      layer := .framework
      description := "Mathlib-backed Fourier-Sobolev H1 reconstruction package; Lean proves the reconstructed L2 value/derivative pair satisfies the Fourier weak-derivative relation, Parseval identities, and finite-time value/derivative contractions on the Fourier-H1 carrier." }
  , { name := "periodicHeat1D_positiveTimeCoefficientSmoothingTheory_literature"
      layer := .framework
      description := "Mathlib-backed positive-time coefficient smoothing package; Lean proves integer Gaussian summability, polynomially weighted heat-multiplier summability, and polynomially weighted evolved value/derivative mode-energy summability for every positive time." }
  , { name := "periodicHeat1D_positiveTimeSmoothFourierRepresentativeTheory_literature"
      layer := .framework
      description := "Mathlib-backed positive-time Fourier representative package; Lean assembles the reconstructed Fourier-H1 state, evolved coefficient identities, Parseval/contraction identities, arbitrary polynomial weighted value/derivative summability, weighted l1 coefficient summability, continuous complex Fourier-series slices, and the full complex spatial-derivative Fourier tower with termwise classical differentiability for every positive time." }
  , { name := "periodicHeat1D_positiveTimeSmoothPeriodicH1RepresentativeTheory_literature"
      layer := .framework
      description := "Mathlib-backed positive-time real PeriodicH1State representative package; Lean takes real parts of the complex Fourier derivative tower, proves the recursive weak-derivative chain by integration by parts from classical lifted derivatives, and packages positive-time smoothness without adding a heat PDE axiom." }
  , { name := "periodicH1_weakDerivative_fourierCoeff_allModes_literature"
      layer := .framework
      description := "Lean theorem extending the periodic H1 weak-derivative Fourier coefficient identity to every integer mode, including the zero mode needed by the heat representative." }
  , { name := "periodicHeat1D_l2FourierSeries_fourierCoeff"
      layer := .framework
      description := "Mathlib Hilbert-basis theorem recovering each Fourier coefficient from an L2 Fourier series represented by HasSum over the periodic Fourier basis." }
  , { name := "periodicHeat1D_continuousSpatialDerivativeFourierSeries_fourierCoeff"
      layer := .framework
      description := "Lean theorem recovering the Fourier coefficients of every positive-time continuous complex spatial-derivative Fourier series slice from its absolutely summable coefficient series." }
  , { name := "periodicHeat1D_realSpatialDerivativeFourierSeries_fourierCoeff"
      layer := .framework
      description := "Lean theorem proving that taking the real part of the positive-time complex heat Fourier series preserves the intended real-valued Fourier coefficients." }
  , { name := "realContinuousMap_fourierCoeff_neg_eq_star"
      layer := .framework
      description := "Lean theorem proving conjugate symmetry of mathlib Fourier coefficients for real-valued continuous periodic functions." }
  , { name := "complexContinuousMap_ext_fourierCoeff"
      layer := .framework
      description := "Mathlib-backed extensionality theorem: continuous complex functions on the unit torus are equal when all Fourier coefficients agree." }
  , { name := "realContinuousMap_ext_fourierCoeff"
      layer := .framework
      description := "Lean theorem deriving real continuous torus-map extensionality from complex Fourier coefficient extensionality." }
  , { name := "periodicH1State_ext_fourierCoeff"
      layer := .framework
      description := "Lean theorem deriving equality of concrete PeriodicH1State values from equality of value and weak-derivative Fourier coefficients." }
  , { name := "periodicHeat1D_evolvedValueFourierCoeff_neg_eq_star"
      layer := .framework
      description := "Lean theorem proving the heat multiplier preserves real-valued conjugate symmetry for evolved value Fourier coefficients." }
  , { name := "periodicHeat1D_evolvedDerivativeFourierCoeff_neg_eq_star"
      layer := .framework
      description := "Lean theorem proving the heat multiplier preserves real-valued conjugate symmetry for evolved weak-derivative Fourier coefficients." }
  , { name := "periodicHeat1D_modeMultiplier_hasDerivAt"
      layer := .framework
      description := "Lean theorem computing the time derivative of the scalar heat Fourier multiplier." }
  , { name := "periodicHeat1D_evolvedTimeDerivativeFourierCoeff_eq_viscosity_secondSpatial"
      layer := .framework
      description := "Lean coefficient-level heat equation: the formal time-derivative Fourier coefficient equals viscosity times the second spatial Fourier coefficient." }
  , { name := "periodicH1_value_parseval_literature"
      layer := .framework
      description := "Full all-mode Parseval identity for the periodic H1 value field, exposed for the heat coefficient contraction theorem." }
  , { name := "periodicH1_derivative_parseval_literature"
      layer := .framework
      description := "Full all-mode Parseval identity for the periodic H1 weak-derivative field, exposed for the heat coefficient contraction theorem." }
  , { name := "periodicHeat1D_localTheory_fromSemigroupBackend"
      layer := .framework
      description := "Framework theorem restricting any proved periodic heat construction source to the exact local-window theory shape consumed by the route." }
  , { name := "periodicHeat1D_localContraction_fromFourierReconstruction"
      layer := .framework
      description := "Lean theorem deriving windowed heat energy and derivative-energy contraction from Fourier reconstruction Parseval identities plus the proved coefficient-level l2 contractions." }
  , { name := "periodicHeat1D_localTheory_fromFourierReconstruction"
      layer := .framework
      description := "Lean assembler turning the Fourier reconstruction package into the local periodic heat theory consumed by the route." }
  , { name := "periodicHeat1D_localWindowCertificate_literature"
      layer := .framework
      description := "Lean assembler from decomposed periodic heat local-window facts into the exact heat certificate consumed by the route." }
  , { name := "periodicHeat1D_fourierH1CandidateHeatCurve"
      layer := .framework
      description := "Lean definition of the explicit heat curve candidate: it is the initial datum at nonpositive times and the proved smooth real Fourier PeriodicH1State representative at positive times." }
  , { name := "periodicHeat1D_candidateValueCoeff_fromPositiveTimeIdentification"
      layer := .framework
      description := "Lean theorem extending positive-time candidate coefficient identification to all nonnegative times by deriving the zero-time case from the candidate definition and the heat multiplier zero-time law." }
  , { name := "periodicHeat1D_candidateDerivativeCoeff_fromPositiveTimeIdentification"
      layer := .framework
      description := "Lean theorem extending positive-time weak-derivative candidate coefficient identification to all nonnegative times by deriving the zero-time case from the candidate definition and the heat multiplier zero-time law." }
  , { name := "PeriodicHeat1DWeakSolutionFourierEvolutionFor"
      layer := .framework
      description := "Reusable local heat interface: a weak heat solution has the expected forward-time value and weak-derivative Fourier coefficient evolution on the requested finite window." }
  , { name := "PeriodicHeat1DWeakSolutionValueFourierEvolutionFor"
      layer := .framework
      description := "Reusable local heat interface isolating the remaining PDE boundary to value Fourier coefficient evolution only; derivative coefficient evolution is derived from the H1 weak-derivative identity." }
  , { name := "periodicHeat1D_fourierH1CandidateHeatCurve_fourierEvolution"
      layer := .framework
      description := "Lean theorem proving the explicit canonical heat candidate satisfies the coefficient-evolution interface on every local window." }
  , { name := "periodicHeat1D_fourierH1CandidateIdentification_literature"
      layer := .framework
      description := "Lean theorem proving that at positive times the explicit candidate curve has the heat-evolved value/derivative Fourier coefficients, using the continuous Fourier representative and Hilbert-basis coefficient recovery." }
  , { name := "periodicHeat1D_fourierH1ContinuousCurve_literature"
      layer := .framework
      description := "Lean assembler turning the explicit candidate curve plus candidate-identification theorem into the continuous-curve interface consumed by the local heat window upgrade." }
  , { name := "IntegratedClassicalHeatWindowCertificate"
      layer := .framework
      description := "Reusable heat interface separating classical/integrated heat equation data from the final weak residual exported to certificates." }
  , { name := "HeatWindowResidual.of_integratedClassical"
      layer := .framework
      description := "Reusable theorem deriving a certified windowed weak heat residual from integrated classical heat data plus the test function time integration-by-parts rule." }
  , { name := "periodicHeat1D_fourierH1IntegratedClassicalHeat_literature"
      layer := .framework
      description := "Lean definition/proof of the integrated heat certificate for the concrete Fourier heat candidate: positive-time time differentiation is proved termwise, positive-time spatial heat balance is proved from the weak-derivative chain, nonpositive times are discharged by test support, and the finite-window integrated balance follows by interval-integral congruence." }
  , { name := "periodicHeat1D_fourierH1WindowResidual_literature"
      layer := .framework
      description := "Lean theorem deriving the windowed weak heat residual from periodicHeat1D_fourierH1IntegratedClassicalHeat_literature via HeatWindowResidual.of_integratedClassical." }
  , { name := "periodicHeat1D_weakSolutionValueFourierEvolution_literature"
      layer := .literature
      description := "Remaining periodic heat PDE theorem: every weak periodic heat solution has the standard forward-time value Fourier coefficient evolution on each finite local window." }
  , { name := "periodicHeat1D_weakSolutionFourierEvolution_fromValueEvolution"
      layer := .framework
      description := "Lean theorem deriving full value-plus-weak-derivative coefficient evolution from value coefficient evolution and the PeriodicH1State weak-derivative Fourier identity." }
  , { name := "periodicHeat1D_weakSolutionFourierEvolution_literature"
      layer := .framework
      description := "Lean theorem packaging full weak-solution coefficient evolution from the narrower value-coefficient literature boundary." }
  , { name := "periodicHeat1D_fourierH1WindowUniqueness_fromWeakSolutionFourierEvolution"
      layer := .framework
      description := "Lean theorem deriving forward-time windowed heat uniqueness from weak-solution Fourier coefficient evolution plus PeriodicH1State Fourier extensionality." }
  , { name := "periodicHeat1D_fourierH1WindowUniqueness_literature"
      layer := .framework
      description := "Lean theorem packaging canonical forward-time windowed heat uniqueness from the narrower weak-solution coefficient-evolution boundary." }
  , { name := "periodicHeat1D_fourierH1WindowSmoothing_fromContinuousCurve"
      layer := .framework
      description := "Lean theorem deriving certified-window positive-time heat smoothing from the continuous-curve positive-time identification with the proved smooth real Fourier PeriodicH1State representative." }
  , { name := "periodicHeat1D_fourierH1WindowUpgrade_literature"
      layer := .framework
      description := "Lean assembler from the strengthened continuous representative theorem plus residual and uniqueness into the former monolithic Fourier-H1 certified-window upgrade interface; smoothing is derived in Lean from the representative identification." }
  , { name := "periodicHeat1D_fourierReconstruction_literature"
      layer := .framework
      description := "Lean compatibility shim deriving the older evolved-coefficient reconstruction interface from the narrower Fourier-H1 window upgrade plus the proved Fourier-Sobolev reconstruction theory." }
  , { name := "periodicHeat1D_localTheory_literature"
      layer := .framework
      description := "Lean-defined local periodic heat theory assembled from Fourier reconstruction plus proved coefficient-level contraction; no longer an axiom." }
  , { name := "periodicHeat1D_localExistenceWindow_literature"
      layer := .framework
      description := "Projection from the reusable periodic heat local theory to the local heat existence window." }
  , { name := "periodicHeat1D_localResidual_literature"
      layer := .framework
      description := "Projection from the reusable periodic heat local theory to the windowed heat residual theorem." }
  , { name := "periodicHeat1D_localUniqueness_literature"
      layer := .framework
      description := "Projection from the reusable periodic heat local theory to finite-window heat uniqueness." }
  , { name := "periodicHeat1D_localSmoothing_literature"
      layer := .framework
      description := "Projection from the reusable periodic heat local theory to positive-time smoothing on the window." }
  , { name := "periodicHeat1D_localContraction_literature"
      layer := .framework
      description := "Projection from the reusable periodic heat local theory to finite-window energy and dissipation contraction." }
  , { name := "periodicHeat1D_badGermExclusion_fromSmoothing"
      layer := .framework
      description := "Framework theorem: the concrete heat forbidden-germ predicate is excluded by the local smoothing theorem, so no separate heat bad-germ axiom is needed." }
  , { name := "periodicBurgers1D_coleHopfBackend_literature"
      layer := .framework
      description := "Lean assembler from the bundled periodic Cole-Hopf theory package into the backend consumed by the Burgers route." }
  , { name := "periodicBurgers1D_coleHopfTheory_literature"
      layer := .literature
      description := "Single reusable periodic Cole-Hopf theory package exporting the transform, chart validity, window transport, residual-transfer facts, uniqueness transfer, and H1-domain theorem." }
  , { name := "periodicBurgers1D_coleHopfTransform_literature"
      layer := .framework
      description := "Projection from the bundled periodic Cole-Hopf theory package to the concrete periodic Cole-Hopf transform and pointwise algebraic/regularity laws." }
  , { name := "periodicBurgers1D_coleHopfChartValid_literature"
      layer := .framework
      description := "Projection from the bundled periodic Cole-Hopf theory package to local chart validity on certified windows." }
  , { name := "periodicBurgers1D_coleHopfMapsWindow_literature"
      layer := .framework
      description := "Projection from the bundled periodic Cole-Hopf theory package to the theorem that the transform maps certified Burgers windows into the certified heat window." }
  , { name := "periodicBurgers1D_coleHopfInverseMapsWindow_literature"
      layer := .framework
      description := "Projection from the bundled periodic Cole-Hopf theory package to inverse-window transport from heat data back to Burgers data." }
  , { name := "periodicBurgers1D_coleHopfBurgersToHeatResidual_literature"
      layer := .framework
      description := "Projection from the bundled periodic Cole-Hopf theory package to local residual transfer from Burgers to heat on certified windows." }
  , { name := "periodicBurgers1D_coleHopfHeatToBurgersResidual_literature"
      layer := .framework
      description := "Projection from the bundled periodic Cole-Hopf theory package to local residual transfer from heat back to Burgers on certified windows." }
  , { name := "periodicBurgers1D_coleHopfUniquenessTransfer_literature"
      layer := .framework
      description := "Projection from the bundled periodic Cole-Hopf theory package to the transfer theorem connecting heat uniqueness to the Cole-Hopf Burgers route." }
  , { name := "periodicBurgers1D_h1InBurgersPDEDomain_literature"
      layer := .framework
      description := "Projection from the bundled periodic Cole-Hopf theory package to the theorem that arbitrary periodic H1 Burgers profiles lie in the Cole-Hopf PDE domain." }
  , { name := "periodicBurgers1D_continuationTheory_literature"
      layer := .literature
      description := "Single reusable periodic Burgers continuation/localization package exporting local uniqueness on certified windows, missing-regularity-to-raw-failure-window localization, and nonextendability of that raw failure window for the canonical finite-H1 obstruction criterion." }
  , { name := "periodicBurgers1D_localUniquenessOnWindow_literature"
      layer := .framework
      description := "Projection from the bundled periodic Burgers continuation theory to standard local uniqueness for one-dimensional periodic viscous Burgers weak solutions on a certified finite window." }
  , { name := "periodicBurgers1D_localWindowInputs_literature"
      layer := .framework
      description := "Lean assembler from the single periodic Burgers local theory source plus reusable Poincare into the exact windowed core certificate package consumed by the route." }
  , { name := "periodicBurgers1D_localWindowTheory_literature"
      layer := .literature
      description := "Single reusable finite-window periodic Burgers local theory source for the requested admissible datum; it exports local existence, weak residual, energy, mean preservation, and dissipative estimates on the same certified window." }
  , { name := "periodicBurgers1D_localExistenceWindow_literature"
      layer := .framework
      description := "Projection from the reusable periodic Burgers local theory to its local existence window." }
  , { name := "periodicBurgers1D_localResidual_literature"
      layer := .framework
      description := "Projection from the reusable periodic Burgers local theory to the weak residual theorem on its certified local window." }
  , { name := "periodicBurgers1D_localEnergy_literature"
      layer := .framework
      description := "Projection from the reusable periodic Burgers local theory to its finite-window local energy identity/estimate." }
  , { name := "periodicBurgers1D_meanPoincare_literature"
      layer := .framework
      description := "Lean assembler combining current-carrier local Poincare/coercivity with the mean-preservation projection from the periodic Burgers local theory." }
  , { name := "periodicBurgers1D_poincare_literature"
      layer := .framework
      description := "Burgers adapter for the current-carrier periodic H1 Poincare/coercivity theorem." }
  , { name := "periodicH1_poincare_meanZero_literature"
      layer := .framework
      description := "Framework wrapper exposing the local Poincare/coercivity predicate consumed by the certificate layer from the proved spectral-gap theorem." }
  , { name := "periodicH1_poincare_spectralGap_literature"
      layer := .framework
      description := "Lean theorem assembling periodic Poincare from the bundled periodic H1 Fourier theory package plus the proved first-frequency gap." }
  , { name := "periodicH1_firstFrequencyGap"
      layer := .framework
      description := "Lean arithmetic theorem proving the first nonzero periodic frequency multiplier has squared norm at least one." }
  , { name := "periodicH1_valueModeEnergy_le_derivativeModeEnergy"
      layer := .framework
      description := "Lean theorem deriving per-mode value-energy control from the weak-derivative Fourier coefficient identity and first-frequency gap." }
  , { name := "periodicH1_fourierTheory_literature"
      layer := .framework
      description := "Mathlib-backed reusable Fourier-analysis theory package for the periodic H1 carrier; Lean proves mean-zero value Parseval, value and derivative mode summability, the derivative Parseval lower bound, and the weak-derivative Fourier multiplier identity." }
  , { name := "periodicH1_value_parseval_meanZero_literature"
      layer := .framework
      description := "Projection from the bundled periodic H1 Fourier theory to value Parseval with the zero mode removed by the mean-zero hypothesis." }
  , { name := "periodicH1_value_nonzeroModeEnergy_summable_literature"
      layer := .framework
      description := "Projection from the bundled periodic H1 Fourier theory to summability of the nonzero value-mode energy series." }
  , { name := "periodicH1_derivative_nonzeroModeEnergy_summable_literature"
      layer := .framework
      description := "Projection from the bundled periodic H1 Fourier theory to summability of the nonzero derivative-mode energy series." }
  , { name := "periodicH1_derivative_parseval_lowerBound_literature"
      layer := .framework
      description := "Projection from the bundled periodic H1 Fourier theory to the derivative Parseval lower bound controlling nonzero derivative Fourier modes by derivative energy." }
  , { name := "periodicH1_weakDerivative_fourierCoeff_literature"
      layer := .framework
      description := "Projection from the bundled periodic H1 Fourier theory to the weak-derivative Fourier multiplier identity for nonzero modes." }
  , { name := "periodicBurgers1D_meanPreservation_literature"
      layer := .framework
      description := "Projection from the reusable periodic Burgers local theory to local mean preservation on its certified window." }
  , { name := "periodicBurgers1D_localDissipative_literature"
      layer := .framework
      description := "Projection from the reusable periodic Burgers local theory to its local dissipative-window estimate." }
  , { name := "burgers_localUniquenessOnOverlapsFromLiterature"
      layer := .framework
      description := "Framework adapter: the reusable windowed Burgers literature uniqueness theorem supplies the overlap-uniqueness record consumed by the local continuation chain." }
  , { name := "burgers_canonicalLocalContinuationCriterion"
      layer := .framework
      description := "Canonical framework-side criterion defining local extendability as absence of Burgers-side finite H1 obstruction data; no PDE theorem is hidden here." }
  , { name := "periodicBurgers1D_missingRegularityProducesRawFailureWindow_literature"
      layer := .framework
      description := "Projection from the bundled periodic Burgers continuation theory to the a-posteriori fact that missing regularity yields a raw finite local failure window with no obstruction data bundled into it." }
  , { name := "periodicBurgers1D_rawFailureWindow_nonextendable_literature"
      layer := .framework
      description := "Projection from the bundled periodic Burgers continuation theory to the theorem that the raw failure window is nonextendable for the canonical finite-H1-obstruction continuation criterion." }
  , { name := "BurgersRawFailureWindow.finiteH1Obstruction_of_not_extendable"
      layer := .framework
      description := "Framework theorem: nonextendability for the canonical criterion yields structured finite H1 obstruction data by classical logic." }
  , { name := "periodicBurgers1D_missingRegularityProducesObstructionWindow_fromContinuationLiterature"
      layer := .framework
      description := "Framework assembler: raw failure-window localization plus local nonextendability produces the localized obstruction window consumed by the route." }
  , { name := "burgers_missingRegularityProducesMaximalNonextendableWindow_fromObstructionWindow"
      layer := .framework
      description := "Reusable constructor: a localized obstruction window gives a nonextendable window for the canonical obstruction criterion." }
  , { name := "BurgersFiniteH1BadGermObstruction.coleHopfDomainWitness_from_finiteH1"
      layer := .framework
      description := "Framework theorem: the finite H1 obstruction certificate plus the arbitrary-data Cole-Hopf H1-domain theorem gives Cole-Hopf PDE-domain membership." }
  , { name := "BurgersFiniteH1BadGermObstruction.coleHopfChartWitness_of_domain"
      layer := .framework
      description := "Reusable bridge theorem: the Cole-Hopf backend maps its explicit PDE-domain predicate into chart membership." }
  , { name := "BurgersFiniteH1BadGermObstruction.obstructionWindowSupportedInHeatWindow"
      layer := .framework
      description := "Framework projection theorem: finite obstruction data directly exports that its Burgers obstruction window is contained in the certified local heat window." }
  , { name := "BurgersFiniteH1BadGermObstruction.heatImageSupported_of_windowSupport"
      layer := .framework
      description := "Reusable support-transport theorem: Burgers-germ support plus Burgers-window-to-heat-window inclusion gives heat-image support." }
  , { name := "BurgersFiniteH1BadGermObstruction.heatImageMatchesAtCenter"
      layer := .framework
      description := "Framework projection theorem: finite obstruction data directly exports that the heat curve equals the Cole-Hopf image at the obstruction center time." }
  , { name := "BurgersFiniteH1BadGermObstruction.coleHopfSmoothnessReflection_of_domain_and_centerMatch"
      layer := .framework
      description := "Reusable bridge theorem: Cole-Hopf backend smoothness reflection plus center image matching yields obstruction-level smoothness reflection." }
  , { name := "BurgersFiniteH1BadGermObstruction.heatImageForbidden_of_smoothnessReflection"
      layer := .framework
      description := "Reusable local theorem: Burgers-side nonregularity plus Cole-Hopf smoothness reflection turns the heat image into a local heat forbidden bad germ." }
  ]

def burgersGroundTruthLiteratureBoundary : List String :=
  [ "periodicBurgers1D_coleHopfTheory_literature"
  , "periodicBurgers1D_localWindowTheory_literature"
  , "periodicHeat1D_weakSolutionValueFourierEvolution_literature"
  , "periodicBurgers1D_continuationTheory_literature"
  ]

def burgersGroundTruthProblemSpecificBoundary : List String :=
  []

/-- The migrated route deliberately no longer proves global regularity from any
old scaffold evolution package. It needs the concrete backend permits above. -/
def burgersGroundTruthRemainingBackendWork : List String :=
  [ "Discharge periodicBurgers1D_localWindowTheory_literature by constructing the requested finite-window periodic Burgers local theory for each admissible u0. Its projections supply local existence, weak residual, energy, mean preservation, and dissipative estimates. The periodic H1 Fourier package is now discharged in Lean from mathlib Fourier analysis plus the carrier's weak-derivative interface."
  , "Discharge the split periodic Cole-Hopf literature facts: transform, chart validity, window maps, inverse window maps, two residual-transfer directions, uniqueness transfer, and the H1-domain theorem."
  , "Discharge the remaining periodic heat PDE-semantics fact: periodicHeat1D_weakSolutionValueFourierEvolution_literature, stating that every weak periodic heat solution has the standard forward-time value Fourier coefficient evolution on each finite local window. Full weak-derivative coefficient evolution is no longer an axiom: periodicHeat1D_weakSolutionFourierEvolution_literature is a Lean theorem from value evolution plus the PeriodicH1State weak-derivative Fourier identity. Canonical windowed heat uniqueness is also no longer an axiom: periodicHeat1D_fourierH1WindowUniqueness_literature is a Lean theorem from coefficient evolution plus periodicH1State_ext_fourierCoeff. The heat curve itself is no longer constructed by axiom: periodicHeat1D_fourierH1CandidateHeatCurve is a Lean definition using u0 at nonpositive times and the proved smooth real Fourier representative at positive times. Lean proves the positive-time candidate coefficient identities, derives the zero-time coefficient identities, derives the energy/Parseval match with the Fourier-H1 reconstruction from coefficient matching plus Parseval, proves the positive-time time derivative by termwise Fourier differentiation, proves the local spatial heat balance from the weak-derivative tower, discharges nonpositive times by test support, and assembles periodicHeat1D_fourierH1IntegratedClassicalHeat_literature without custom axioms. Positive-time smoothing on the certified curve is no longer a separate axiom: periodicHeat1D_fourierH1WindowSmoothing_fromContinuousCurve derives it in Lean from the continuous-curve positive-time identification. The former monolithic periodicHeat1D_fourierH1WindowUpgrade_literature is now a Lean assembler. The scalar Fourier heat multiplier layer, coefficient-level l2 contraction/summability layer, L2 Hilbert reconstruction layer, Fourier-H1 derivative/Parseval/contraction layer, positive-time coefficient smoothing layer, weighted l1 coefficient layer, continuous complex Fourier-series derivative tower, real-valued positive-time PeriodicH1State smoothing package, local contraction derivation, integrated heat residual derivation, derivative-evolution derivation, canonical uniqueness derivation, and local certificate packaging are now proved in Lean. Heat bad-germ exclusion is already derived from smoothing by periodicHeat1D_badGermExclusion_fromSmoothing."
  , "Extend the finite-time H1 bad-pattern library only if the dataset is expanded beyond the current documented finite-time H1 blow-up template."
  , "Discharge periodicBurgers1D_continuationTheory_literature by formalizing the reusable periodic Burgers continuation/localization package: local uniqueness on certified windows, missing-regularity-to-raw-failure-window localization, and raw-failure-window nonextendability for the canonical finite-H1-obstruction criterion. The Cole-Hopf PDE-domain condition for finite H1 obstructions follows from the literature Cole-Hopf H1-domain theorem, and the Burgers-window-to-heat-window inclusion and center image matching facts are carried by the finite obstruction data and projected by framework theorems."
  ]

end

end Hypostructure.Backends.Burgers1D
