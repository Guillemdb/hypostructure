import Hypostructure.Backends.Burgers1D.GroundTruthLock
import Hypostructure.Framework.Upgrade

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Framework

noncomputable section

/-- All local data consumed by the final analytic upgrade. This object is the
template boundary where local certificates plus the Lock are allowed to produce
the global Burgers regularity statement. -/
structure BurgersAnalyticUpgradeInput
    (nu : BurgersParameters) where
  bundle : BurgersGroundTruthCoreCertificateBundle nu
  heat : LocalHeatWindowCertificate nu
  coleHopf : LocalColeHopfWindowBridge nu heat
  lock : LockBlocksBurgersBadGerms nu bundle heat coleHopf

/-- Pointwise local estimate package for the exact initial datum being proved.
The equality prevents the final theorem from using a fixed or trivial local
route to prove regularity for unrelated initial data. -/
structure BurgersAnalyticUpgradeInputFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State) where
  input : BurgersAnalyticUpgradeInput nu
  initial_eq : input.bundle.u0 = u0
  coleHopf_h1_in_burgersPDEDomain : ∀ u : PeriodicH1State,
    PeriodicH1State.IsPeriodicH1 u →
      input.coleHopf.transform.burgersPDEDomain u

/-- The remaining analytic theorem to be supplied by the backend. It is named
as a theorem-shaped object rather than an unstructured function field. -/
structure BurgersAnalyticUpgradeTheorem
    (nu : BurgersParameters) where
  upgrade : ∀ _input : BurgersAnalyticUpgradeInput nu,
    BurgersGroundTruthGlobalRegularityStatement nu

def BurgersRegularityWitness
    (nu : BurgersParameters)
    (u0 : PeriodicH1State) : Prop :=
  ∃ u : BurgersSolutionCurve nu,
    SolvesViscousBurgersWeak nu u0 u ∧
      GlobalH1Solution u ∧
      UniqueWeakBurgersSolution nu u0 u ∧
      ∀ t : ℝ, 0 < t → SmoothAtPositiveTime u t

/-- A-posteriori localization principle for the final upgrade: if a classical
regularity witness is missing for some admissible initial datum, the failure
produces a route-local bad morphism into the certified hypostructure. This is
the precise analytic gap left to discharge from PDE compactness/continuation
machinery. -/
def BurgersAposterioriBadMorphismLocalization
    (nu : BurgersParameters)
    (input : BurgersAnalyticUpgradeInput nu) : Prop :=
  APosterioriLocalization
    PeriodicH1State.IsPeriodicH1
    (BurgersRegularityWitness nu)
    (BurgersBadMorphismExists nu input.bundle input.heat input.coleHopf)

/-- Pointwise a-posteriori localization for the local route attached to the
same initial datum `u0`. This is the form used to assemble global regularity
from local estimates. -/
def BurgersAposterioriBadMorphismLocalizationFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0) : Prop :=
  PeriodicH1State.IsPeriodicH1 u0 →
    ¬ BurgersRegularityWitness nu u0 →
      BurgersBadMorphismExists
        nu input.input.bundle input.input.heat input.input.coleHopf

/-- Localized data extracted from a missing regularity witness. This is the
backend-specific output of the a-posteriori compactness/continuation analysis;
the framework then turns it into a Lock bad morphism. -/
structure BurgersLocalizedFailureWitness
    (nu : BurgersParameters)
    (input : BurgersAnalyticUpgradeInput nu)
    (u0 : PeriodicH1State) where
  germ : BurgersBadGerm
  finite_time_h1_blowup : FiniteTimeH1BlowUpBadGerm germ
  burgers_chart : input.coleHopf.transform.burgersSector germ.profile
  heat_image_supported :
    (ColeHopfHeatBadGermImage input.coleHopf.transform germ.routeWindow).SupportedInHeatWindow
      input.heat.certified.window
  heat_image_forbidden :
    input.heat.forbidden_bad_germ
      (ColeHopfHeatBadGermImage input.coleHopf.transform germ.routeWindow)

def BurgersLocalizedFailureWitness.toBadMorphismCandidate
    {nu : BurgersParameters}
    {input : BurgersAnalyticUpgradeInput nu}
    {u0 : PeriodicH1State}
    (W : BurgersLocalizedFailureWitness nu input u0) :
    BurgersBadMorphismCandidate nu input.bundle input.heat input.coleHopf where
  germ := W.germ
  finite_time_h1_blowup := W.finite_time_h1_blowup
  heat_available := localHeatSmoothingFrameworkCertificate_sound nu input.heat
  bridge_available :=
    localColeHopfFrameworkCertificate_sound nu input.heat input.coleHopf
  library_complete := input.bundle.badPatternLibrary.complete
  burgers_chart := W.burgers_chart
  heat_image_supported := W.heat_image_supported
  heat_image_forbidden := W.heat_image_forbidden

/-- The certified local evolution data already present in the hypostructure
input for the exact datum being proved. This is fully local/windowed: it records
only the certified initial window, weak residual on that window, and local
energy/mean/dissipation certificates. -/
structure BurgersCertifiedLocalEvolutionFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0) where
  window_contains_zero : input.input.bundle.window.Contains 0
  solves_on_window :
    SolvesViscousBurgersWeakOnWindow
      nu u0 input.input.bundle.solution input.input.bundle.window
  energy_on_window :
    LocalEnergyIdentity
      nu input.input.bundle.solution input.input.bundle.energy.window
  mean_on_window :
    LocalMeanSectorPreservation
      nu u0 input.input.bundle.solution input.input.bundle.meanSector.window
  dissipative_on_window :
    LocalDissipativeWindow
      nu input.input.bundle.solution input.input.bundle.dissipativeWindow.window

def BurgersAnalyticUpgradeInputFor.certifiedLocalEvolution
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    (input : BurgersAnalyticUpgradeInputFor nu u0) :
    BurgersCertifiedLocalEvolutionFor nu u0 input := by
  rcases input with ⟨upgradeInput, initial_eq⟩
  subst u0
  exact
    { window_contains_zero := upgradeInput.bundle.window_contains_zero
      solves_on_window := upgradeInput.bundle.solutionCertifiedOnWindow
      energy_on_window := upgradeInput.bundle.energy.certified
      mean_on_window := upgradeInput.bundle.meanSector.certified
      dissipative_on_window := upgradeInput.bundle.dissipativeWindow.certified }

/-- Local uniqueness on the certified window. This is the first analytic PDE
input after local existence: two windowed weak Burgers evolutions with the same
initial datum agree on the overlap represented by the certified window. -/
structure BurgersLocalUniquenessOnOverlaps
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0) where
  unique_on_certified_window :
    ∀ v : BurgersSolutionCurve nu,
      SolvesViscousBurgersWeakOnWindow nu u0 v input.input.bundle.window →
        ∀ t : ℝ, input.input.bundle.window.Contains t →
          v.eval t = input.input.bundle.solution.eval t

/-- A Burgers window is supported in a heat window when every time certified on
the Burgers side lies in the heat-side certified window. -/
def BurgersWindow.SupportedInHeatWindow
    (BW : BurgersWindow)
    (HW : HeatWindow) : Prop :=
  ∀ t : ℝ, BW.Contains t → HW.Contains t

/-- A bad germ is supported in a Burgers window when every time in its local
route support lies inside that certified Burgers window. -/
def BurgersBadGerm.SupportedInBurgersWindow
    (g : BurgersBadGerm)
    (W : BurgersWindow) : Prop :=
  ∀ t : ℝ, g.routeWindow.ContainsTime t → W.Contains t

/-- Local data carried by a bad continuation window. This is the finite `H¹`
obstruction data that the local continuation theorem must expose: a local
Burgers bad germ, finite-time `H¹` bad-pattern evidence, support in the bad
continuation window, the bookkeeping facts linking that window and center time
to the certified heat window, and positivity/nonregularity of the obstruction
time. It deliberately does not include Cole-Hopf domain or chart membership;
those remain separate bridge-transfer facts. -/
structure BurgersWindowFiniteH1ObstructionData
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (W : BurgersWindow) where
  germ : BurgersBadGerm
  finite_time_h1_blowup : FiniteTimeH1BlowUpBadGerm germ
  supported_in_window : germ.SupportedInBurgersWindow W
  window_supported_in_heat : W.SupportedInHeatWindow input.input.heat.certified.window
  heat_image_matches_at_center :
    input.input.heat.certified.curve.eval germ.centerTime =
      input.input.coleHopf.transform.toHeat germ.profile
  positive_center_time : 0 < germ.centerTime
  center_nonregular : ¬ SmoothPeriodicState germ.profile

/-- Local continuation criterion. The framework only needs the local logical
shape: a certified window is extendable unless it carries a structured finite
`H¹` Burgers-side local obstruction. The analytic backend supplies the concrete
`extendableAt` and `badAt` predicates plus the extraction theorem from badness
to finite obstruction data. -/
structure BurgersLocalContinuationCriterion
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0) where
  extendableAt : BurgersWindow → Prop
  badAt : BurgersWindow → Prop
  bad_obstruction_data : ∀ W : BurgersWindow,
    badAt W → BurgersWindowFiniteH1ObstructionData nu u0 input W
  no_bad_implies_extendable : ∀ W : BurgersWindow,
    ¬ badAt W → extendableAt W

/-- Canonical local continuation criterion used by the framework layer. It does
not assert any PDE continuation theorem by itself: it defines extendability as
absence of a Burgers-side finite `H¹` obstruction. The remaining analytic
content is the separate maximal-failure/localization theorem proving that a
missing regularity witness produces a nonextendable window for this criterion. -/
def burgers_canonicalLocalContinuationCriterion
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0) :
    BurgersLocalContinuationCriterion nu u0 input where
  extendableAt := fun W => ¬ Nonempty (BurgersWindowFiniteH1ObstructionData nu u0 input W)
  badAt := fun W => Nonempty (BurgersWindowFiniteH1ObstructionData nu u0 input W)
  bad_obstruction_data := by
    intro _W hbad
    exact Classical.choice hbad
  no_bad_implies_extendable := by
    intro _W hno_bad
    exact hno_bad

/-- Direct PDE-local output of a missing regularity witness for the canonical
criterion: a finite local window carrying an actual Burgers-side obstruction.
This is narrower than assuming a ready-made maximal nonextendable object. -/
structure BurgersLocalizedObstructionWindow
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0) where
  window : BurgersWindow
  contains_initial : window.Contains 0
  finite_terminal : Prop
  finite_terminal_holds : finite_terminal
  obstruction : Nonempty (BurgersWindowFiniteH1ObstructionData nu u0 input window)

/-- Raw output of the a-posteriori failure localization step before applying
the canonical continuation criterion. This deliberately contains only the local
finite window metadata; it does not itself contain finite `H¹` obstruction data.
The separate local continuation theorem below is what turns this raw failure
window into a nonextendable window, and then Lean extracts the structured
obstruction from the canonical criterion. -/
structure BurgersRawFailureWindow
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (_input : BurgersAnalyticUpgradeInputFor nu u0) where
  window : BurgersWindow
  contains_initial : window.Contains 0
  finite_terminal : Prop
  finite_terminal_holds : finite_terminal

theorem BurgersLocalContinuationCriterion.bad_of_not_extendable
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    (C : BurgersLocalContinuationCriterion nu u0 input)
    {W : BurgersWindow}
    (hnot : ¬ C.extendableAt W) :
    C.badAt W := by
  by_contra hbad
  exact hnot (C.no_bad_implies_extendable W hbad)

/-- Maximal local development that cannot be extended by the local continuation
criterion. This is a local object: it records a certified Burgers window and the
failure of local extendability there, not a global bound. -/
structure BurgersMaximalNonextendableWindow
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (C : BurgersLocalContinuationCriterion nu u0 input) where
  window : BurgersWindow
  contains_initial : window.Contains 0
  finite_terminal : Prop
  finite_terminal_holds : finite_terminal
  not_extendable : ¬ C.extendableAt window

/-- Local continuation obstruction extracted from a maximal nonextendable
window by the continuation criterion. -/
structure BurgersLocalContinuationObstruction
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (C : BurgersLocalContinuationCriterion nu u0 input) where
  window : BurgersWindow
  contains_initial : window.Contains 0
  finite_terminal : Prop
  finite_terminal_holds : finite_terminal
  bad_on_window : C.badAt window

def BurgersMaximalNonextendableWindow.toLocalContinuationObstruction
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (M : BurgersMaximalNonextendableWindow nu u0 input C) :
    BurgersLocalContinuationObstruction nu u0 input C where
  window := M.window
  contains_initial := M.contains_initial
  finite_terminal := M.finite_terminal
  finite_terminal_holds := M.finite_terminal_holds
  bad_on_window := C.bad_of_not_extendable M.not_extendable

def BurgersLocalizedObstructionWindow.toMaximalNonextendableWindow
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    (W : BurgersLocalizedObstructionWindow nu u0 input) :
    BurgersMaximalNonextendableWindow
      nu u0 input (burgers_canonicalLocalContinuationCriterion nu u0 input) where
  window := W.window
  contains_initial := W.contains_initial
  finite_terminal := W.finite_terminal
  finite_terminal_holds := W.finite_terminal_holds
  not_extendable := by
    intro hextendable
    exact hextendable W.obstruction

theorem BurgersRawFailureWindow.finiteH1Obstruction_of_not_extendable
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    (W : BurgersRawFailureWindow nu u0 input)
    (hnot_extendable :
      ¬ (burgers_canonicalLocalContinuationCriterion nu u0 input).extendableAt W.window) :
    Nonempty (BurgersWindowFiniteH1ObstructionData nu u0 input W.window) := by
  by_contra hno_obstruction
  exact hnot_extendable hno_obstruction

def BurgersRawFailureWindow.toLocalizedObstructionWindow
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    (W : BurgersRawFailureWindow nu u0 input)
    (hnot_extendable :
      ¬ (burgers_canonicalLocalContinuationCriterion nu u0 input).extendableAt W.window) :
    BurgersLocalizedObstructionWindow nu u0 input where
  window := W.window
  contains_initial := W.contains_initial
  finite_terminal := W.finite_terminal
  finite_terminal_holds := W.finite_terminal_holds
  obstruction := W.finiteH1Obstruction_of_not_extendable hnot_extendable

/-- Finite `H¹` bad germ obtained from a local continuation obstruction. This is
the object classified by the finite bad-pattern library before the Lock is
applied. -/
structure BurgersFiniteH1BadGermObstruction
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (C : BurgersLocalContinuationCriterion nu u0 input) where
  local_obstruction : BurgersLocalContinuationObstruction nu u0 input C
  germ : BurgersBadGerm
  finite_time_h1_blowup : FiniteTimeH1BlowUpBadGerm germ
  supported_in_window : germ.SupportedInBurgersWindow local_obstruction.window
  window_supported_in_heat :
    local_obstruction.window.SupportedInHeatWindow input.input.heat.certified.window
  heat_image_matches_at_center :
    input.input.heat.certified.curve.eval germ.centerTime =
      input.input.coleHopf.transform.toHeat germ.profile
  positive_center_time : 0 < germ.centerTime
  center_nonregular : ¬ SmoothPeriodicState germ.profile

def BurgersLocalContinuationObstruction.toFiniteH1Obstruction
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersLocalContinuationObstruction nu u0 input C) :
    BurgersFiniteH1BadGermObstruction nu u0 input C :=
  let data := C.bad_obstruction_data O.window O.bad_on_window
  { local_obstruction := O
    germ := data.germ
    finite_time_h1_blowup := data.finite_time_h1_blowup
    supported_in_window := data.supported_in_window
    window_supported_in_heat := data.window_supported_in_heat
    heat_image_matches_at_center := data.heat_image_matches_at_center
    positive_center_time := data.positive_center_time
    center_nonregular := data.center_nonregular }

/-- The finite bad-pattern library classifies the extracted local obstruction.
This is a proposition, not a chosen data object: the route only needs the fact
that the finite library covers the obstruction before the local Lock transfer is
applied. -/
def BurgersFiniteH1BadGermObstruction.Classified
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C) :
    Prop :=
  ∃ P : BurgersBadPattern,
    P ∈ input.input.bundle.badPatternLibrary.library.patterns ∧
      P.Accepts O.germ

theorem BurgersFiniteH1BadGermObstruction.classified
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C) :
    O.Classified :=
  input.input.bundle.badPatternLibrary.complete O.finite_time_h1_blowup

def BurgersFiniteH1BadGermObstruction.ColeHopfChartWitness
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C) : Prop :=
  input.input.coleHopf.transform.burgersSector O.germ.profile

def BurgersFiniteH1BadGermObstruction.ColeHopfDomainWitness
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C) : Prop :=
  input.input.coleHopf.transform.burgersPDEDomain O.germ.profile

theorem BurgersFiniteH1BadGermObstruction.coleHopfDomainWitness_from_finiteH1
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C) :
    O.ColeHopfDomainWitness :=
  input.coleHopf_h1_in_burgersPDEDomain
    O.germ.profile
    O.finite_time_h1_blowup.admissible

theorem BurgersFiniteH1BadGermObstruction.coleHopfChartWitness_of_domain
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C)
    (hdomain : O.ColeHopfDomainWitness) :
    O.ColeHopfChartWitness :=
  input.input.coleHopf.transform.burgersPDEDomain_in_sector O.germ.profile hdomain

def BurgersFiniteH1BadGermObstruction.HeatImageSupported
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C) : Prop :=
  (ColeHopfHeatBadGermImage
      input.input.coleHopf.transform O.germ.routeWindow).SupportedInHeatWindow
    input.input.heat.certified.window

def BurgersFiniteH1BadGermObstruction.ObstructionWindowSupportedInHeatWindow
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C) : Prop :=
  O.local_obstruction.window.SupportedInHeatWindow input.input.heat.certified.window

theorem BurgersFiniteH1BadGermObstruction.heatImageSupported_of_windowSupport
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C)
    (hwindow : O.ObstructionWindowSupportedInHeatWindow) :
    O.HeatImageSupported := by
  intro t ht
  exact hwindow t (O.supported_in_window t ht)

theorem BurgersFiniteH1BadGermObstruction.obstructionWindowSupportedInHeatWindow
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C) :
    O.ObstructionWindowSupportedInHeatWindow :=
  O.window_supported_in_heat

def BurgersFiniteH1BadGermObstruction.HeatImageForbidden
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C) : Prop :=
  input.input.heat.forbidden_bad_germ
    (ColeHopfHeatBadGermImage
      input.input.coleHopf.transform O.germ.routeWindow)

/-- The narrow analytic bridge needed to turn a Burgers-side nonregularity
obstruction into a heat forbidden germ: if the heat image were smooth at the
obstruction time, then the Burgers profile would be smooth. -/
def BurgersFiniteH1BadGermObstruction.ColeHopfSmoothnessReflection
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C) : Prop :=
  HeatSmoothAtPositiveTime input.input.heat.certified.curve O.germ.centerTime →
    SmoothPeriodicState O.germ.profile

def BurgersFiniteH1BadGermObstruction.HeatImageMatchesAtCenter
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C) : Prop :=
  input.input.heat.certified.curve.eval O.germ.centerTime =
    input.input.coleHopf.transform.toHeat O.germ.profile

theorem BurgersFiniteH1BadGermObstruction.heatImageMatchesAtCenter
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C) :
    O.HeatImageMatchesAtCenter :=
  O.heat_image_matches_at_center

theorem BurgersFiniteH1BadGermObstruction.coleHopfSmoothnessReflection_of_domain_and_centerMatch
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C)
    (hdomain : O.ColeHopfDomainWitness)
    (hmatch : O.HeatImageMatchesAtCenter) :
    O.ColeHopfSmoothnessReflection := by
  intro hheat_smooth
  exact input.input.coleHopf.transform.smoothness_reflects_from_heat_image
    O.germ.profile
    (input.input.heat.certified.curve.eval O.germ.centerTime)
    hdomain
    hmatch
    hheat_smooth

theorem BurgersFiniteH1BadGermObstruction.heatImageForbidden_of_smoothnessReflection
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C)
    (hreflection : O.ColeHopfSmoothnessReflection) :
    O.HeatImageForbidden := by
  refine ⟨O.germ.centerTime, ?_, O.positive_center_time, ?_⟩
  · exact O.finite_time_h1_blowup.2
  · intro hheat_smooth
    exact O.center_nonregular (hreflection hheat_smooth)

theorem BurgersFiniteH1BadGermObstruction.excluded_by_local_heat_bridge
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    {C : BurgersLocalContinuationCriterion nu u0 input}
    (O : BurgersFiniteH1BadGermObstruction nu u0 input C)
    (hclassified : O.Classified)
    (hchart : O.ColeHopfChartWitness)
    (hsupport : O.HeatImageSupported)
    (hforbidden : O.HeatImageForbidden) :
    False := by
  rcases hclassified with ⟨pattern, _hpattern_mem, hpattern_accepts⟩
  have hroute_admissible : O.germ.routeWindow.Admissible :=
    pattern.accepts_routeWindow hpattern_accepts
  have htransport :=
    input.input.coleHopf.transports_bad_germs
      O.germ.routeWindow
      hsupport
      hroute_admissible
      hchart
  exact input.input.heat.no_forbidden_heat_bad_germ
    (ColeHopfHeatBadGermImage input.input.coleHopf.transform O.germ.routeWindow)
    htransport.1
    htransport.2
    hforbidden

/-- The local continuation/localization chain replacing the previous monolithic
failure-localization axiom. Continuation produces only a Burgers-side finite
`H¹` obstruction; separate bridge-transfer fields then place the classified
obstruction in the Cole-Hopf chart, support its heat image in the certified heat
window, and mark that heat image as forbidden by local heat smoothing. -/
structure BurgersLocalContinuationLocalizationChain
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0) where
  localEvolution : BurgersCertifiedLocalEvolutionFor nu u0 input
  localUniqueness : BurgersLocalUniquenessOnOverlaps nu u0 input
  continuation : BurgersLocalContinuationCriterion nu u0 input
  maximalFailure :
    PeriodicH1State.IsPeriodicH1 u0 →
      ¬ BurgersRegularityWitness nu u0 →
        BurgersMaximalNonextendableWindow nu u0 input continuation
  obstructionInColeHopfChart :
    ∀ O : BurgersFiniteH1BadGermObstruction nu u0 input continuation,
      O.Classified → O.ColeHopfChartWitness
  heatImageSupported :
    ∀ O : BurgersFiniteH1BadGermObstruction nu u0 input continuation,
      O.Classified → O.HeatImageSupported
  heatImageForbidden :
    ∀ O : BurgersFiniteH1BadGermObstruction nu u0 input continuation,
      O.Classified → O.HeatImageForbidden

def BurgersLocalContinuationLocalizationChain.localizedFailureWitness
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {input : BurgersAnalyticUpgradeInputFor nu u0}
    (chain : BurgersLocalContinuationLocalizationChain nu u0 input)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (hmissing : ¬ BurgersRegularityWitness nu u0) :
    BurgersLocalizedFailureWitness nu input.input u0 :=
  let maximal := chain.maximalFailure hu0 hmissing
  let localObstruction := maximal.toLocalContinuationObstruction
  let finiteObstruction := localObstruction.toFiniteH1Obstruction
  let hclassified := finiteObstruction.classified
  { germ := finiteObstruction.germ
    finite_time_h1_blowup := finiteObstruction.finite_time_h1_blowup
    burgers_chart := chain.obstructionInColeHopfChart finiteObstruction hclassified
    heat_image_supported := chain.heatImageSupported finiteObstruction hclassified
    heat_image_forbidden := chain.heatImageForbidden finiteObstruction hclassified }

theorem burgers_obstructionInColeHopfChart_fromDomain
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (continuation : BurgersLocalContinuationCriterion nu u0 input) :
    ∀ O : BurgersFiniteH1BadGermObstruction nu u0 input continuation,
      O.Classified → O.ColeHopfChartWitness := by
  intro O _hclassified
  exact O.coleHopfChartWitness_of_domain
    O.coleHopfDomainWitness_from_finiteH1

theorem burgers_obstructionHeatImageSupported_fromWindowSupport
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (continuation : BurgersLocalContinuationCriterion nu u0 input) :
    ∀ O : BurgersFiniteH1BadGermObstruction nu u0 input continuation,
      O.Classified → O.HeatImageSupported := by
  intro O _hclassified
  exact O.heatImageSupported_of_windowSupport
    O.obstructionWindowSupportedInHeatWindow

theorem burgers_obstructionHeatSmoothnessReflects_fromDomainAndCenterMatch
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (continuation : BurgersLocalContinuationCriterion nu u0 input) :
    ∀ O : BurgersFiniteH1BadGermObstruction nu u0 input continuation,
      O.Classified → O.ColeHopfSmoothnessReflection := by
  intro O _hclassified
  exact O.coleHopfSmoothnessReflection_of_domain_and_centerMatch
    O.coleHopfDomainWitness_from_finiteH1
    O.heatImageMatchesAtCenter

theorem burgers_obstructionTransfersToHeatForbidden_fromSmoothnessReflection
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (continuation : BurgersLocalContinuationCriterion nu u0 input) :
    ∀ O : BurgersFiniteH1BadGermObstruction nu u0 input continuation,
      O.Classified → O.HeatImageForbidden := by
  intro O hclassified
  exact O.heatImageForbidden_of_smoothnessReflection
    (burgers_obstructionHeatSmoothnessReflects_fromDomainAndCenterMatch
      nu u0 input continuation O hclassified)

def burgers_localizedFailureWitness_fromLocalContinuationChain
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (chain : BurgersLocalContinuationLocalizationChain nu u0 input) :
    PeriodicH1State.IsPeriodicH1 u0 →
      ¬ BurgersRegularityWitness nu u0 →
        BurgersLocalizedFailureWitness nu input.input u0 :=
  chain.localizedFailureWitness

theorem burgers_pointwise_badMorphismLocalization_fromLocalContinuationChain
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (chain : BurgersLocalContinuationLocalizationChain nu u0 input) :
    BurgersAposterioriBadMorphismLocalizationFor nu u0 input := by
  intro hu0 hmissing
  exact ⟨
    ((burgers_localizedFailureWitness_fromLocalContinuationChain
      nu u0 input chain hu0 hmissing).toBadMorphismCandidate)
  ⟩

theorem burgersPointwiseRegularity_fromAposterioriLocalization
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (hlocalize : BurgersAposterioriBadMorphismLocalizationFor nu u0 input) :
    BurgersRegularityWitness nu u0 :=
  APosterioriLocalization.point input.input.lock hlocalize hu0

theorem burgersPointwiseRegularity_fromLocalContinuationChain
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (input : BurgersAnalyticUpgradeInputFor nu u0)
    (chain : BurgersLocalContinuationLocalizationChain nu u0 input) :
    BurgersRegularityWitness nu u0 :=
  burgersPointwiseRegularity_fromAposterioriLocalization
    nu u0 hu0 input
    (burgers_pointwise_badMorphismLocalization_fromLocalContinuationChain
      nu u0 input chain)

/-- A provider of local hypostructure estimates for every admissible initial
datum. This is the formal boundary for the claim "global regularity follows
from local estimates": the global theorem below only iterates this provider and
applies the pointwise Lock/a-posteriori upgrade. -/
structure BurgersPointwiseLocalEstimateProvider
    (nu : BurgersParameters) where
  localInput : ∀ u0 : PeriodicH1State,
    PeriodicH1State.IsPeriodicH1 u0 →
      BurgersAnalyticUpgradeInputFor nu u0
  localChain : ∀ (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0),
      BurgersLocalContinuationLocalizationChain nu u0 (localInput u0 hu0)

theorem BurgersPointwiseLocalEstimateProvider.globalRegularity
    {nu : BurgersParameters}
    (P : BurgersPointwiseLocalEstimateProvider nu) :
    BurgersGroundTruthGlobalRegularityStatement nu := by
  intro u0 hu0
  exact burgersPointwiseRegularity_fromLocalContinuationChain
    nu u0 hu0 (P.localInput u0 hu0) (P.localChain u0 hu0)

theorem burgers_globalRegularity_from_pointwise_local_estimates
    (nu : BurgersParameters)
    (localInput : ∀ u0 : PeriodicH1State,
      PeriodicH1State.IsPeriodicH1 u0 →
        BurgersAnalyticUpgradeInputFor nu u0)
    (localChain : ∀ (u0 : PeriodicH1State)
      (hu0 : PeriodicH1State.IsPeriodicH1 u0),
        BurgersLocalContinuationLocalizationChain nu u0 (localInput u0 hu0)) :
    BurgersGroundTruthGlobalRegularityStatement nu :=
  (BurgersPointwiseLocalEstimateProvider.mk localInput localChain).globalRegularity

structure BurgersAnalyticPointwiseUpgradeTheorem
    (nu : BurgersParameters) where
  upgrade : ∀ u0 : PeriodicH1State,
    PeriodicH1State.IsPeriodicH1 u0 →
      BurgersAnalyticUpgradeInputFor nu u0 →
        BurgersRegularityWitness nu u0

def burgersAnalyticUpgradeTheoremFromPointwiseProvider
    (nu : BurgersParameters)
    (provider : BurgersPointwiseLocalEstimateProvider nu) :
    BurgersAnalyticUpgradeTheorem nu where
  upgrade := fun _input => provider.globalRegularity

theorem burgersAnalyticUpgrade_fromAposterioriLocalization
    (nu : BurgersParameters)
    (input : BurgersAnalyticUpgradeInput nu)
    (hlocalize : BurgersAposterioriBadMorphismLocalization nu input) :
    BurgersGroundTruthGlobalRegularityStatement nu :=
  APosterioriLocalization.target input.lock hlocalize

def burgersAposterioriLocalizationDossier
    (nu : BurgersParameters)
    (input : BurgersAnalyticUpgradeInput nu)
    (hlocalize : BurgersAposterioriBadMorphismLocalization nu input) :
    APosterioriLocalizationDossier PeriodicH1State where
  name := "Burgers1D a-posteriori failure-to-bad-morphism upgrade"
  admissible := PeriodicH1State.IsPeriodicH1
  witness := BurgersRegularityWitness nu
  badMorphism := BurgersBadMorphismExists nu input.bundle input.heat input.coleHopf
  lockBlocks := input.lock
  localizes := hlocalize

theorem burgersAnalyticUpgrade_fromAposterioriDossier
    (nu : BurgersParameters)
    (D : APosterioriLocalizationDossier PeriodicH1State)
    (hadmissible : D.admissible = PeriodicH1State.IsPeriodicH1)
    (hwitness : D.witness = BurgersRegularityWitness nu) :
    BurgersGroundTruthGlobalRegularityStatement nu := by
  intro u0 hu0
  have htarget := D.target
  have hw : D.witness u0 := htarget u0 (by
    rw [hadmissible]
    exact hu0)
  rw [hwitness] at hw
  exact hw

def analyticRegularityFrameworkCertificate
    (nu : BurgersParameters)
    (_input : BurgersAnalyticUpgradeInput nu)
    (_upgradeThm : BurgersAnalyticUpgradeTheorem nu) :
    AnalyticRegularityCertificate :=
  { node := .lock
    payload :=
      { backendName := "Burgers1D ground-truth local-to-global analytic upgrade"
        targetClaim := BurgersGroundTruthGlobalRegularityStatement nu }
    meaning := BurgersGroundTruthGlobalRegularityStatement nu }

theorem analyticRegularityFrameworkCertificate_sound
    (nu : BurgersParameters)
    (input : BurgersAnalyticUpgradeInput nu)
    (upgradeThm : BurgersAnalyticUpgradeTheorem nu) :
    (analyticRegularityFrameworkCertificate nu input upgradeThm).meaning :=
  upgradeThm.upgrade input

def BurgersAnalyticUpgradeInput.fromLocalCertificates
    (nu : BurgersParameters)
    (bundle : BurgersGroundTruthCoreCertificateBundle nu)
    (H : LocalHeatWindowCertificate nu)
    (B : LocalColeHopfWindowBridge nu H) :
    BurgersAnalyticUpgradeInput nu where
  bundle := bundle
  heat := H
  coleHopf := B
  lock := lockBlocksBadGermsFromLocalCertificates nu bundle H B

def BurgersAnalyticUpgradeInputFor.fromLocalCertificates
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (bundle : BurgersGroundTruthCoreCertificateBundle nu)
    (hinit : bundle.u0 = u0)
    (H : LocalHeatWindowCertificate nu)
    (B : LocalColeHopfWindowBridge nu H)
    (h1_domain : ∀ u : PeriodicH1State,
      PeriodicH1State.IsPeriodicH1 u → B.transform.burgersPDEDomain u) :
    BurgersAnalyticUpgradeInputFor nu u0 where
  input := BurgersAnalyticUpgradeInput.fromLocalCertificates nu bundle H B
  initial_eq := hinit
  coleHopf_h1_in_burgersPDEDomain := h1_domain

/-- Burgers uses the E5 functional tactic for the Lock: local Cole-Hopf and
heat certificates rule out the represented bad morphism. -/
def burgersE5LockTacticCertificate
    (nu : BurgersParameters)
    (input : BurgersAnalyticUpgradeInput nu) :
    LockTacticCertificate where
  tactic := .E5_functional
  certificateName := "K_Cat_Hom^blk"
  requiredCertificates := burgersGroundTruthLockPackage.requiredCertificates
  preservationCertificates := burgersGroundTruthPreservationCertificateNames
  obstruction :=
    LockBlocksBurgersBadGerms nu input.bundle input.heat input.coleHopf

def burgersE5LockTacticDossier
    (nu : BurgersParameters)
    (input : BurgersAnalyticUpgradeInput nu) :
    LockTacticDossier burgersGroundTruthRoute where
  certificate := burgersE5LockTacticCertificate nu input
  tacticCertificatePresent := by
    simp [burgersE5LockTacticCertificate, burgersGroundTruthRoute,
      burgersGroundTruthFinalCertificateChain]
  requiredCertificatesPresent := by
    intro cert hcert
    simp [burgersE5LockTacticCertificate, burgersGroundTruthLockPackage,
      burgersGroundTruthRoute, burgersGroundTruthFinalCertificateChain] at hcert ⊢
    rcases hcert with rfl | rfl | rfl | rfl | rfl
    all_goals simp [burgersGroundTruthFinalCertificateChain]
  preservationCertificatesPresent := by
    intro cert hcert
    simp [burgersE5LockTacticCertificate, burgersGroundTruthPreservationCertificateNames,
      burgersGroundTruthRoute, burgersGroundTruthFinalCertificateChain] at hcert ⊢
    rcases hcert with rfl | rfl | rfl
    all_goals simp [burgersGroundTruthFinalCertificateChain]
  obstructionProof := input.lock
  blockedCertificatePresent := by
    simp [burgersGroundTruthRoute, burgersGroundTruthFinalCertificateChain]

def burgersAnalyticLocalCertificateNames : List String :=
  [ "K_D_E^+"
  , "K_C_mu^+"
  , "K_Cap_H^+"
  , "K_LS_sigma^+"
  , "K_TB_pi^+"
  , "K_TB_O^+"
  , "K_TB_rho^+"
  , "K_RepDesc_K^+"
  ]

def burgersAnalyticBridgeCertificateNames : List String :=
  [ "K_ColeHopf^+"
  , "K_HeatSmooth^+"
  ]

/-- Burgers instantiation of the reusable local/bridge/Lock-to-target upgrade
dossier. The framework checks certificate accounting and non-circularity; the
backend still supplies the mathematical upgrade theorem. -/
def burgersAnalyticUpgradeDossier
    (nu : BurgersParameters)
    (input : BurgersAnalyticUpgradeInput nu)
    (upgradeThm : BurgersAnalyticUpgradeTheorem nu) :
    LocalToTargetUpgradeDossier
      burgersGroundTruthRoute
      (BurgersGroundTruthGlobalRegularityStatement nu) where
  name := "Burgers1D local-to-global analytic upgrade"
  targetCertificate := "K_Reg_Burgers1D^+"
  structuralCertificate := "K_StructReg_Burgers1D^+"
  localCertificateNames := burgersAnalyticLocalCertificateNames
  bridgeCertificateNames := burgersAnalyticBridgeCertificateNames
  lockCertificatePresent := by
    simp [burgersGroundTruthRoute, burgersGroundTruthFinalCertificateChain]
  structuralCertificatePresent := by
    simp [burgersGroundTruthRoute, burgersGroundTruthFinalCertificateChain]
  localCertificatesPresent := by
    intro cert hcert
    simp [burgersGroundTruthRoute, burgersGroundTruthFinalCertificateChain,
      burgersAnalyticLocalCertificateNames] at hcert ⊢
    rcases hcert with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals simp [burgersGroundTruthFinalCertificateChain]
  bridgeCertificatesPresent := by
    intro cert hcert
    simp [burgersGroundTruthRoute, burgersGroundTruthFinalCertificateChain,
      burgersAnalyticBridgeCertificateNames] at hcert ⊢
    rcases hcert with rfl | rfl
    all_goals simp [burgersGroundTruthFinalCertificateChain]
  targetCertificatePresent := by
    simp [burgersGroundTruthRoute, burgersGroundTruthFinalCertificateChain]
  nonCircular := by
    simp [burgersGroundTruthRoute, burgersAnalyticLocalCertificateNames,
      burgersAnalyticBridgeCertificateNames]
  upgrade := analyticRegularityFrameworkCertificate_sound nu input upgradeThm

def burgersCertifiedAnalyticUpgradeDossier
    (nu : BurgersParameters)
    (input : BurgersAnalyticUpgradeInput nu)
    (upgradeThm : BurgersAnalyticUpgradeTheorem nu) :
    CertifiedUpgradeDossier
      burgersGroundTruthRoute
      (BurgersGroundTruthGlobalRegularityStatement nu) where
  lock := burgersE5LockTacticDossier nu input
  upgrade := burgersAnalyticUpgradeDossier nu input upgradeThm

end

end Hypostructure.Backends.Burgers1D
