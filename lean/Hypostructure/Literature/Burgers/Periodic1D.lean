import Hypostructure.Backends.Burgers1D.GroundTruthCoreFactory
import Hypostructure.Backends.Burgers1D.GroundTruthUpgrade
import Hypostructure.Literature.Analysis.PeriodicPoincare1D

namespace Hypostructure.Literature.Burgers.Periodic1D

open Hypostructure.Backends.Burgers1D

noncomputable section

/-- Literature-supplied local existence window before the weak residual and
local estimates are attached. This keeps local existence separate from the
energy, mean, Poincare, and dissipative estimates used by the framework route. -/
structure PeriodicBurgers1DLocalExistenceWindow
    (nu : BurgersParameters)
    (u0 : PeriodicH1State) where
  window : BurgersWindow
  window_contains_zero : window.Contains 0
  curve : BurgersSolutionCurve nu
  initial_on_window : BurgersInitialConditionOnWindow u0 curve window
  boundary_on_window : BurgersPeriodicBoundaryOnWindow curve window

def PeriodicBurgers1DLocalExistenceWindow.toCertified
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    (D : PeriodicBurgers1DLocalExistenceWindow nu u0)
    (residual : WeakBurgersResidualOnWindow nu D.curve D.window) :
    CertifiedBurgersLocalWindow nu where
  window := D.window
  initial := u0
  curve := D.curve
  solvesOnWindow := ⟨D.initial_on_window, D.boundary_on_window, residual⟩

structure PeriodicBurgers1DLocalEnergyEstimate
    (nu : BurgersParameters)
    {u0 : PeriodicH1State}
    (D : PeriodicBurgers1DLocalExistenceWindow nu u0) where
  energy : LocalEnergyCertificateData nu D.curve
  window_eq : energy.window = D.window.time

structure PeriodicBurgers1DMeanPoincareEstimate
    (nu : BurgersParameters)
    {u0 : PeriodicH1State}
    (D : PeriodicBurgers1DLocalExistenceWindow nu u0) where
  poincare : LocalPoincareCoercivity u0
  meanSector : LocalMeanSectorCertificateData nu u0 D.curve
  mean_window_eq : meanSector.window = D.window.time

structure PeriodicBurgers1DMeanPreservationEstimate
    (nu : BurgersParameters)
    {u0 : PeriodicH1State}
    (D : PeriodicBurgers1DLocalExistenceWindow nu u0) where
  meanSector : LocalMeanSectorCertificateData nu u0 D.curve
  window_eq : meanSector.window = D.window.time

structure PeriodicBurgers1DLocalDissipativeEstimate
    (nu : BurgersParameters)
    {u0 : PeriodicH1State}
    (D : PeriodicBurgers1DLocalExistenceWindow nu u0) where
  dissipativeWindow : LocalDissipativeWindowCertificateData nu D.curve
  window_eq : dissipativeWindow.window = D.window.time

/-- Reusable local periodic Burgers theory package for the exact admissible
datum. This is the single literature source for the local Burgers PDE package:
local existence, the weak residual, and the finite-window energy, mean, and
dissipative estimates are supplied for the same certified local window. -/
structure PeriodicBurgers1DLocalWindowTheoryFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State) where
  existence : PeriodicBurgers1DLocalExistenceWindow nu u0
  residual : WeakBurgersResidualOnWindow nu existence.curve existence.window
  energy : PeriodicBurgers1DLocalEnergyEstimate nu existence
  meanPreservation : PeriodicBurgers1DMeanPreservationEstimate nu existence
  dissipative : PeriodicBurgers1DLocalDissipativeEstimate nu existence

/-- Literature boundary for arbitrary-data finite-window periodic Burgers
local theory. All framework-facing local Burgers facts below are Lean
projections from this single source theorem. -/
axiom periodicBurgers1D_localWindowTheory_literature :
  ∀ (nu : BurgersParameters)
    (u0 : PeriodicH1State),
    PeriodicH1State.IsPeriodicH1 u0 →
      PeriodicBurgers1DLocalWindowTheoryFor nu u0

/-- Framework projection from the reusable local Burgers theory to its
existence window. -/
def periodicBurgers1D_localExistenceWindow_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicBurgers1DLocalExistenceWindow nu u0 :=
  (periodicBurgers1D_localWindowTheory_literature nu u0 hu0).existence

/-- Framework projection from the reusable local Burgers theory to the weak
residual on the same local window. -/
def periodicBurgers1D_localResidual_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    WeakBurgersResidualOnWindow nu
      (periodicBurgers1D_localExistenceWindow_literature nu u0 hu0).curve
      (periodicBurgers1D_localExistenceWindow_literature nu u0 hu0).window :=
  (periodicBurgers1D_localWindowTheory_literature nu u0 hu0).residual

/-- Framework projection from the reusable local Burgers theory to the
finite-window local energy identity/estimate. -/
def periodicBurgers1D_localEnergy_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicBurgers1DLocalEnergyEstimate nu
      (periodicBurgers1D_localExistenceWindow_literature nu u0 hu0) :=
  (periodicBurgers1D_localWindowTheory_literature nu u0 hu0).energy

/-- Burgers adapter for reusable periodic Poincare/coercivity. The actual
analytic input lives in `Hypostructure.Literature.Analysis.PeriodicPoincare1D`;
this theorem only specializes it to the exact Burgers datum. -/
def periodicBurgers1D_poincare_literature
    (_nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (_D : PeriodicBurgers1DLocalExistenceWindow _nu u0) :
    LocalPoincareCoercivity u0 :=
  Hypostructure.Literature.Analysis.PeriodicPoincare1D.periodicH1_poincare_meanZero_literature
    u0 hu0

/-- Framework projection from the reusable local Burgers theory to local mean
preservation on the same finite window as the local existence data. -/
def periodicBurgers1D_meanPreservation_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicBurgers1DMeanPreservationEstimate nu
      (periodicBurgers1D_localExistenceWindow_literature nu u0 hu0) :=
  (periodicBurgers1D_localWindowTheory_literature nu u0 hu0).meanPreservation

/-- Assembler for the mean/Poincare package consumed by the local core route.
Poincare is imported from reusable periodic analysis, while mean preservation is
projected from the local Burgers theory source. Their combination is framework
bookkeeping. -/
def periodicBurgers1D_meanPoincare_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicBurgers1DMeanPoincareEstimate nu
      (periodicBurgers1D_localExistenceWindow_literature nu u0 hu0) :=
  let D := periodicBurgers1D_localExistenceWindow_literature nu u0 hu0
  let meanPreservation := periodicBurgers1D_meanPreservation_literature nu u0 hu0
  { poincare := periodicBurgers1D_poincare_literature nu u0 hu0 D
    meanSector := meanPreservation.meanSector
    mean_window_eq := meanPreservation.window_eq }

/-- Framework projection from the reusable local Burgers theory to the local
dissipative window estimate. -/
def periodicBurgers1D_localDissipative_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicBurgers1DLocalDissipativeEstimate nu
      (periodicBurgers1D_localExistenceWindow_literature nu u0 hu0) :=
  (periodicBurgers1D_localWindowTheory_literature nu u0 hu0).dissipative

/-- Assembler from the single local Burgers theory source plus reusable
Poincare into the exact local input package consumed by the hypostructure core
route. This definition is proved in Lean; the literature assumptions above are
the only analytic boundary. -/
def periodicBurgers1D_localWindowInputs_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    Σ W : BurgersWindow, BurgersCorePDELocalWindowInputsFor nu u0 W :=
  let D := periodicBurgers1D_localExistenceWindow_literature nu u0 hu0
  let residual := periodicBurgers1D_localResidual_literature nu u0 hu0
  let energy := periodicBurgers1D_localEnergy_literature nu u0 hu0
  let meanPoincare := periodicBurgers1D_meanPoincare_literature nu u0 hu0
  let dissipative := periodicBurgers1D_localDissipative_literature nu u0 hu0
  ⟨D.window,
    { certified := D.toCertified residual
      initial_eq := rfl
      window_eq := rfl
      window_contains_zero := D.window_contains_zero
      energy := energy.energy
      energy_window_eq := energy.window_eq
      poincare := meanPoincare.poincare
      meanSector := meanPoincare.meanSector
      mean_window_eq := meanPoincare.mean_window_eq
      dissipativeWindow := dissipative.dissipativeWindow
      dissipative_window_eq := dissipative.window_eq }
  ⟩

/-- Reusable continuation/localization package for one-dimensional periodic
viscous Burgers.

This groups the a-posteriori PDE facts used by the hypostructure upgrade:
local uniqueness on certified windows, localization of missing regularity to a
raw finite failure window, and nonextendability of that raw window for the
canonical finite-`H¹` obstruction criterion. These are reusable Burgers
continuation facts, not hypostructure framework logic. -/
structure PeriodicBurgers1DContinuationTheory where
  localUniquenessOnWindow :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (u v : BurgersSolutionCurve nu)
      (W : BurgersWindow),
      SolvesViscousBurgersWeakOnWindow nu u0 u W →
        SolvesViscousBurgersWeakOnWindow nu u0 v W →
          ∀ t : ℝ, W.Contains t → v.eval t = u.eval t
  missingRegularityProducesRawFailureWindow :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (input : BurgersAnalyticUpgradeInputFor nu u0)
      (_localEvolution : BurgersCertifiedLocalEvolutionFor nu u0 input)
      (_localUniqueness : BurgersLocalUniquenessOnOverlaps nu u0 input),
      PeriodicH1State.IsPeriodicH1 u0 →
        ¬ BurgersRegularityWitness nu u0 →
          BurgersRawFailureWindow nu u0 input
  rawFailureWindow_nonextendable :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (input : BurgersAnalyticUpgradeInputFor nu u0)
      (_localEvolution : BurgersCertifiedLocalEvolutionFor nu u0 input)
      (_localUniqueness : BurgersLocalUniquenessOnOverlaps nu u0 input)
      (W : BurgersRawFailureWindow nu u0 input),
      ¬ (burgers_canonicalLocalContinuationCriterion nu u0 input).extendableAt W.window

/-- Single reusable literature boundary for periodic Burgers continuation and
a-posteriori localization. -/
axiom periodicBurgers1D_continuationTheory_literature :
  PeriodicBurgers1DContinuationTheory

/-- Projection from the bundled continuation theory to local uniqueness of
one-dimensional periodic viscous Burgers solutions on certified finite time
windows. -/
theorem periodicBurgers1D_localUniquenessOnWindow_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (u v : BurgersSolutionCurve nu)
    (W : BurgersWindow)
    (hu : SolvesViscousBurgersWeakOnWindow nu u0 u W)
    (hv : SolvesViscousBurgersWeakOnWindow nu u0 v W)
    (t : ℝ)
    (ht : W.Contains t) :
    v.eval t = u.eval t :=
  periodicBurgers1D_continuationTheory_literature.localUniquenessOnWindow
    nu u0 u v W hu hv t ht

/-- Projection from the bundled continuation theory to the localization theorem:
if a periodic `H¹` datum lacks the regularity witness, standard 1D Burgers
continuation theory supplies a raw finite failure window for the same locally
certified evolution. -/
def periodicBurgers1D_missingRegularityProducesRawFailureWindow_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (localEvolution : BurgersCertifiedLocalEvolutionFor nu u0 input)
    (localUniqueness : BurgersLocalUniquenessOnOverlaps nu u0 input)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (hmissing : ¬ BurgersRegularityWitness nu u0) :
    BurgersRawFailureWindow nu u0 input :=
  periodicBurgers1D_continuationTheory_literature.missingRegularityProducesRawFailureWindow
    nu u0 input localEvolution localUniqueness hu0 hmissing

/-- Projection from the bundled continuation theory to the nonextendability
criterion for the raw failure window. -/
theorem periodicBurgers1D_rawFailureWindow_nonextendable_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (localEvolution : BurgersCertifiedLocalEvolutionFor nu u0 input)
    (localUniqueness : BurgersLocalUniquenessOnOverlaps nu u0 input)
    (W : BurgersRawFailureWindow nu u0 input) :
    ¬ (burgers_canonicalLocalContinuationCriterion nu u0 input).extendableAt W.window :=
  periodicBurgers1D_continuationTheory_literature.rawFailureWindow_nonextendable
    nu u0 input localEvolution localUniqueness W

end

end Hypostructure.Literature.Burgers.Periodic1D
