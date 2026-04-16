import Hypostructure.Backends.Burgers1D.GroundTruthRun

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Framework

noncomputable section

/-- Explicit remaining backend permits for the ground-truth route. These are
not instantiated by the current scaffold. Supplying this structure is exactly
the remaining work needed to turn the migrated route into the full theorem in
`docs/source/dataset/burgers_1d.md`. -/
structure BurgersGroundTruthBackendPermits
    (nu : BurgersParameters) where
  localColeHopfBridge : Prop
  localColeHopfBridge_holds : localColeHopfBridge
  localHeatSmoothingAndUniqueness : Prop
  localHeatSmoothingAndUniqueness_holds : localHeatSmoothingAndUniqueness
  lockBlocksBadGerms : Prop
  lockBlocksBadGerms_holds : lockBlocksBadGerms
  analyticUpgrade :
    BurgersGroundTruthCoreCertificateBundle nu →
      localColeHopfBridge →
      localHeatSmoothingAndUniqueness →
      lockBlocksBadGerms →
      BurgersGroundTruthGlobalRegularityStatement nu

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
    (bundle : BurgersGroundTruthCoreCertificateBundle nu)
    (permits : BurgersGroundTruthBackendPermits nu) :
    BurgersGroundTruthGlobalRegularityStatement nu :=
  permits.analyticUpgrade
    bundle
    permits.localColeHopfBridge_holds
    permits.localHeatSmoothingAndUniqueness_holds
    permits.lockBlocksBadGerms_holds

def burgers_groundTruth_dataset_theorem
    (nu : BurgersParameters)
    (bundle : BurgersGroundTruthCoreCertificateBundle nu)
    (permits : BurgersGroundTruthBackendPermits nu) :
    BurgersGroundTruthDatasetClaim nu where
  finalCertificateRecorded := burgers_groundTruth_final_certificate_recorded
  routeValidity := burgersGroundTruthRunValidity_holds
  globalRegularity := burgers_groundTruth_global_regularized nu bundle permits

/-- The migrated route deliberately no longer proves global regularity from the
old `BurgersPDEEvolutionPackage`. It needs the concrete backend permits above. -/
def burgersGroundTruthRemainingBackendWork : List String :=
  [ "Prove the compactness and representation certificates against PeriodicH1State, not the old BurgersState scaffold."
  , "Replace the package-level localColeHopfBridge permit with a concrete periodic Cole-Hopf transform on the ground-truth carrier."
  , "Replace the package-level localHeatSmoothingAndUniqueness permit with a concrete periodic heat semigroup theorem on the ground-truth carrier."
  , "Prove lockBlocksBadGerms from local bad-germ capacity, Cole-Hopf, and heat smoothing, without using global Burgers regularity."
  , "Prove analyticUpgrade from the local certificate bundle plus the Lock/backend permits to BurgersGroundTruthGlobalRegularityStatement."
  ]

end

end Hypostructure.Backends.Burgers1D
