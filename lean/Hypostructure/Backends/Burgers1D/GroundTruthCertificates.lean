import Hypostructure.Backends.Burgers1D.GroundTruthWindows
import Hypostructure.Backends.Burgers1D.GroundTruthLocalAnalysis
import Hypostructure.Framework.Certificates

namespace Hypostructure.Backends.Burgers1D

open MeasureTheory
open Hypostructure.Framework

noncomputable section

namespace PeriodicH1State

noncomputable def derivativeEnergy (u : PeriodicH1State) : ℝ :=
  ∫ x : BurgersTorus, (u.weakDeriv x) ^ (2 : ℕ)

theorem energy_nonneg (u : PeriodicH1State) :
    0 ≤ energy u := by
  unfold energy
  exact integral_nonneg fun _x => sq_nonneg _

theorem derivativeEnergy_nonneg (u : PeriodicH1State) :
    0 ≤ derivativeEnergy u := by
  unfold derivativeEnergy
  exact integral_nonneg fun _x => sq_nonneg _

theorem dissipation_eq_viscosity_mul_derivativeEnergy
    (nu : BurgersParameters)
    (u : PeriodicH1State) :
    dissipation nu u = nu.viscosity * derivativeEnergy u := by
  rfl

theorem dissipation_nonneg
    (nu : BurgersParameters)
    (u : PeriodicH1State) :
    0 ≤ dissipation nu u := by
  rw [dissipation_eq_viscosity_mul_derivativeEnergy]
  exact mul_nonneg (le_of_lt nu.viscosity_pos) (derivativeEnergy_nonneg u)

end PeriodicH1State

def timeIntegralOn
    (W : TimeWindow)
    (F : ℝ → ℝ) : ℝ :=
  ∫ t in W.t0..W.t1, F t

/-- The local finite-window energy inequality used by `K_D_E^+`. This is a
windowed statement only; it does not assert global existence or global
smoothness. -/
def FiniteWindowEnergyInequality
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu)
    (W : TimeWindow) : Prop :=
  PeriodicH1State.energy (u.eval W.t1) +
      timeIntegralOn W (fun t => PeriodicH1State.dissipation nu (u.eval t)) ≤
    PeriodicH1State.energy (u.eval W.t0)

def LocalEnergyIdentity
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu)
    (W : TimeWindow) : Prop :=
  FiniteWindowEnergyInequality nu u W

/-- Snapshot-level smooth energy identity already available in mathlib-style
calculus form. The windowed flow theorem is still future work, but this theorem
is a genuine local analytic identity, not a global regularity assumption. -/
def SmoothSnapshotEnergyIdentity
    (nu : BurgersParameters) : Prop :=
  ∀ s : SmoothPeriodicSnapshot nu,
    ∫ x in (0 : ℝ)..1, s.v x * s.vt x =
      -nu.viscosity * ∫ x in (0 : ℝ)..1, (s.vx x) ^ (2 : ℕ)

theorem smoothSnapshotEnergyIdentity
    (nu : BurgersParameters) :
    SmoothSnapshotEnergyIdentity nu :=
  SmoothPeriodicSnapshot.energy_pairing_identity

structure LocalEnergyCertificateData
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu) where
  window : TimeWindow
  certified : LocalEnergyIdentity nu u window

def localEnergyFrameworkCertificate
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu)
    (cert : LocalEnergyCertificateData nu u) :
    EnergyCertificate :=
  { node := .energyCheck
    payload :=
      { heightName := "ground-truth periodic H1 L2 energy"
        dissipationName := "nu * ||weakDeriv||_L2^2 on a finite window"
        boundStatement := LocalEnergyIdentity nu u cert.window }
    meaning := LocalEnergyIdentity nu u cert.window }

theorem localEnergyFrameworkCertificate_sound
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu)
    (cert : LocalEnergyCertificateData nu u) :
    (localEnergyFrameworkCertificate nu u cert).meaning :=
  cert.certified

/-- The 0-truncated compactness certificate used by the template route. It is a
finite local profile library containing the certified state; no global compact
embedding theorem is hidden here. -/
structure LocalCompactnessCertificateData where
  state : PeriodicH1State
  profileSet : Set PeriodicH1State
  contains_state : state ∈ profileSet
  finite_profileSet : profileSet.Finite

def LocalCompactnessModuloTranslation
    (cert : LocalCompactnessCertificateData) : Prop :=
  cert.state ∈ cert.profileSet ∧ cert.profileSet.Finite

def localCompactnessFrameworkCertificate
    (cert : LocalCompactnessCertificateData) :
    CompactnessCertificate :=
  { node := .compactCheck
    payload :=
      { symmetryGroupName := "periodic torus translations"
        quotientName := "0-truncated finite local profile quotient"
        profileName := "ground-truth periodic H1 profile"
        compactnessStatement := LocalCompactnessModuloTranslation cert }
    meaning := LocalCompactnessModuloTranslation cert }

theorem localCompactnessFrameworkCertificate_sound
    (cert : LocalCompactnessCertificateData) :
    (localCompactnessFrameworkCertificate cert).meaning := by
  exact ⟨cert.contains_state, cert.finite_profileSet⟩

/-- A concrete representation dictionary for the 0-truncated backend. The
faithfulness law is the only framework property needed by the sieve. -/
structure LocalRepresentationDictionary where
  Code : Type
  encode : PeriodicH1State → Code
  decode : Code → PeriodicH1State
  faithful : ∀ u : PeriodicH1State, decode (encode u) = u

structure LocalRepresentationCertificateData where
  dictionary : LocalRepresentationDictionary

def LocalRepresentationFaithful
    (cert : LocalRepresentationCertificateData) : Prop :=
  ∀ u : PeriodicH1State,
    cert.dictionary.decode (cert.dictionary.encode u) = u

def localRepresentationFrameworkCertificate
    (cert : LocalRepresentationCertificateData) :
    RepresentationCertificate :=
  { node := .complexCheck
    payload :=
      { languageName := "0-truncated periodic H1 code"
        dictionaryName := "ground-truth finite local representation dictionary"
        faithfulStatement := LocalRepresentationFaithful cert }
    meaning := LocalRepresentationFaithful cert }

theorem localRepresentationFrameworkCertificate_sound
    (cert : LocalRepresentationCertificateData) :
    (localRepresentationFrameworkCertificate cert).meaning :=
  cert.dictionary.faithful

/-- Local Poincare/coercivity statement on a single certified state. -/
def LocalPoincareCoercivity
    (u : PeriodicH1State) : Prop :=
  PeriodicH1State.mean u = 0 →
    PeriodicH1State.energy u ≤ PeriodicH1State.derivativeEnergy u

/-- Local mean-sector preservation on a finite time window. -/
def LocalMeanSectorPreservation
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (u : BurgersSolutionCurve nu)
    (W : TimeWindow) : Prop :=
  ∀ BW : BurgersWindow,
    BW.time = W →
      SolvesViscousBurgersWeakOnWindow nu u0 u BW →
    ∀ t : ℝ, W.t0 ≤ t → t ≤ W.t1 →
      PeriodicH1State.mean (u.eval t) = PeriodicH1State.mean u0

/-- Local dissipative window statement used by the mixing/tameness route. -/
def LocalDissipativeWindow
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu)
    (W : TimeWindow) : Prop :=
  ∀ t : ℝ, W.t0 ≤ t → t ≤ W.t1 →
    PeriodicH1State.energy (u.eval t) ≤ PeriodicH1State.energy (u.eval W.t0)

def LocalMeanSectorDecomposition : Prop :=
  ∀ u : PeriodicH1State,
    ∃ v : PeriodicH1State, ∃ m : ℝ,
      PeriodicH1State.mean v = 0 ∧
        PeriodicH1State.add v (PeriodicH1State.constantState m) = u

theorem mean_constantState
    (m : ℝ) :
    PeriodicH1State.mean (PeriodicH1State.constantState m) = m :=
  PeriodicH1State.mean_constantState m

theorem mean_meanZeroPart
    (u : PeriodicH1State) :
    PeriodicH1State.mean (PeriodicH1State.meanZeroPart u) = 0 :=
  PeriodicH1State.mean_meanZeroPart u

theorem decompose_mean_zero_plus_constant
    (u : PeriodicH1State) :
    PeriodicH1State.add
        (PeriodicH1State.meanZeroPart u)
        (PeriodicH1State.constantState (PeriodicH1State.mean u)) = u :=
  PeriodicH1State.decompose_mean_zero_plus_constant u

theorem localMeanSectorDecomposition :
    LocalMeanSectorDecomposition := by
  intro u
  exact ⟨
    PeriodicH1State.meanZeroPart u,
    PeriodicH1State.mean u,
    PeriodicH1State.mean_meanZeroPart u,
    PeriodicH1State.decompose_mean_zero_plus_constant u
  ⟩

structure LocalPoincareCertificateData where
  state : PeriodicH1State
  certified : LocalPoincareCoercivity state

structure LocalMeanSectorCertificateData
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (u : BurgersSolutionCurve nu) where
  window : TimeWindow
  certified : LocalMeanSectorPreservation nu u0 u window

structure LocalDissipativeWindowCertificateData
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu) where
  window : TimeWindow
  certified : LocalDissipativeWindow nu u window

def localPoincareFrameworkCertificate
    (cert : LocalPoincareCertificateData) :
    StiffnessCertificate :=
  { node := .stiffnessCheck
    payload :=
      { gapConstant := 1
        exponent := (1 : ℝ) / 2
        coercivityStatement := LocalPoincareCoercivity cert.state }
    meaning := LocalPoincareCoercivity cert.state }

theorem localPoincareFrameworkCertificate_sound
    (cert : LocalPoincareCertificateData) :
    (localPoincareFrameworkCertificate cert).meaning :=
  cert.certified

def localMeanSectorFrameworkCertificate
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (u : BurgersSolutionCurve nu)
    (cert : LocalMeanSectorCertificateData nu u0 u) :
    TopologyCertificate :=
  { node := .topoCheck
    payload :=
      { invariantName := "mean on a certified finite window"
        sectorStatement := LocalMeanSectorPreservation nu u0 u cert.window }
    meaning := LocalMeanSectorPreservation nu u0 u cert.window }

theorem localMeanSectorFrameworkCertificate_sound
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (u : BurgersSolutionCurve nu)
    (cert : LocalMeanSectorCertificateData nu u0 u) :
    (localMeanSectorFrameworkCertificate nu u0 u cert).meaning :=
  cert.certified

def localTamenessFrameworkCertificate :
    TamenessCertificate :=
  { node := .tameCheck
    payload :=
      { structureName := "ground-truth mean-zero plus constant sector split"
        stratificationBound := 1
        tameStatement := LocalMeanSectorDecomposition }
    meaning := LocalMeanSectorDecomposition }

theorem localTamenessFrameworkCertificate_sound :
    localTamenessFrameworkCertificate.meaning :=
  localMeanSectorDecomposition

def localDissipativeWindowFrameworkCertificate
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu)
    (cert : LocalDissipativeWindowCertificateData nu u) :
    MixingCertificate :=
  { node := .ergoCheck
    payload :=
      { invariantMeasureName := "finite-window H1 energy sector"
        mixingTimeFinite := LocalDissipativeWindow nu u cert.window
        convergenceStatement := LocalDissipativeWindow nu u cert.window }
    meaning := LocalDissipativeWindow nu u cert.window }

theorem localDissipativeWindowFrameworkCertificate_sound
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu)
    (cert : LocalDissipativeWindowCertificateData nu u) :
    (localDissipativeWindowFrameworkCertificate nu u cert).meaning :=
  cert.certified

/-- Local cylinder supporting a candidate bad germ. No global singular set is
mentioned here. -/
structure TimeSpaceCylinder where
  centerTime : ℝ
  radius : ℝ
  radius_pos : 0 < radius
  centerSpace : BurgersTorus

namespace TimeSpaceCylinder

def timeWindow (C : TimeSpaceCylinder) : TimeWindow where
  t0 := C.centerTime - C.radius
  t1 := C.centerTime + C.radius
  ordered := by
    calc
      C.centerTime - C.radius ≤ C.centerTime :=
        sub_le_self C.centerTime (le_of_lt C.radius_pos)
      _ ≤ C.centerTime + C.radius :=
        le_add_of_nonneg_right (le_of_lt C.radius_pos)

theorem centerTime_mem_timeWindow
    (C : TimeSpaceCylinder) :
    C.timeWindow.Contains C.centerTime :=
  ⟨sub_le_self C.centerTime (le_of_lt C.radius_pos),
    le_add_of_nonneg_right (le_of_lt C.radius_pos)⟩

end TimeSpaceCylinder

structure BurgersBadGerm where
  centerTime : ℝ
  centerSpace : BurgersTorus
  localWindow : TimeSpaceCylinder
  profile : PeriodicH1State

namespace BurgersBadGerm

def routeWindow (g : BurgersBadGerm) : RouteLocalBadGermWindow where
  time := g.localWindow.timeWindow
  centerSpace := g.centerSpace
  profile := g.profile

theorem centerTime_mem_routeWindow
    (g : BurgersBadGerm)
    (hcenter : g.centerTime = g.localWindow.centerTime) :
    g.routeWindow.ContainsTime g.centerTime :=
  hcenter.symm ▸ g.localWindow.centerTime_mem_timeWindow

theorem localWindow_centerTime_mem_routeWindow
    (g : BurgersBadGerm) :
    g.routeWindow.ContainsTime g.localWindow.centerTime :=
  g.localWindow.centerTime_mem_timeWindow

end BurgersBadGerm

def LocallyAdmissibleBurgersBadGerm
    (g : BurgersBadGerm) : Prop :=
  PeriodicH1State.IsPeriodicH1 g.profile

theorem LocallyAdmissibleBurgersBadGerm.routeWindow
    {g : BurgersBadGerm}
    (h : LocallyAdmissibleBurgersBadGerm g) :
    g.routeWindow.Admissible :=
  h

/-- Local bad-germ capacity means the germ has a finite local energy bound.
This is deliberately local: it does not assert that the global singular set is
empty. -/
def LocalBadGermCapacity
    (g : BurgersBadGerm) : Prop :=
  ∃ C : ℝ, 0 ≤ C ∧ PeriodicH1State.energy g.profile ≤ C

def RouteLocalBadGermWindow.LocalCapacity
    (W : RouteLocalBadGermWindow) : Prop :=
  ∃ C : ℝ, 0 ≤ C ∧ PeriodicH1State.energy W.profile ≤ C

theorem LocalBadGermCapacity.routeWindow
    {g : BurgersBadGerm}
    (h : LocalBadGermCapacity g) :
    g.routeWindow.LocalCapacity :=
  h

/-- The Burgers bad-pattern taxonomy currently needed by
`docs/source/dataset/burgers_1d.md`. The document's certified library has one
minimal class: finite-time loss of `H¹` regularity, represented by a local bad
germ. More constructors can be added here without changing the Lock API. -/
inductive BurgersBadPatternKind where
  | finiteTimeH1BlowUp
  deriving DecidableEq, Repr

/-- The local predicate represented by the finite-time `H¹` blow-up bad
pattern. This is still a local, 0-truncated object: it says that the candidate
germ is an admissible periodic `H¹` local profile and that its recorded center
time lies in its certified local support. -/
def FiniteTimeH1BlowUpBadGerm
    (g : BurgersBadGerm) : Prop :=
  LocallyAdmissibleBurgersBadGerm g ∧
    g.routeWindow.ContainsTime g.centerTime

theorem FiniteTimeH1BlowUpBadGerm.admissible
    {g : BurgersBadGerm}
    (h : FiniteTimeH1BlowUpBadGerm g) :
    LocallyAdmissibleBurgersBadGerm g :=
  h.1

theorem FiniteTimeH1BlowUpBadGerm.routeWindow
    {g : BurgersBadGerm}
    (h : FiniteTimeH1BlowUpBadGerm g) :
    g.routeWindow.Admissible :=
  h.admissible.routeWindow

/-- A finite-library bad pattern is a template predicate on local bad germs,
with the framework facts needed by the Lock. -/
structure BurgersBadPattern where
  kind : BurgersBadPatternKind
  name : String
  accepts : BurgersBadGerm → Prop
  accepts_finiteTimeH1 : ∀ {g : BurgersBadGerm},
    accepts g → FiniteTimeH1BlowUpBadGerm g
  local_capacity : ∀ {g : BurgersBadGerm},
    accepts g → LocalBadGermCapacity g

namespace BurgersBadPattern

def Accepts (P : BurgersBadPattern) (g : BurgersBadGerm) : Prop :=
  P.accepts g

theorem accepts_admissible
    {P : BurgersBadPattern}
    {g : BurgersBadGerm}
    (h : P.Accepts g) :
    LocallyAdmissibleBurgersBadGerm g :=
  (P.accepts_finiteTimeH1 h).admissible

theorem accepts_routeWindow
    {P : BurgersBadPattern}
    {g : BurgersBadGerm}
    (h : P.Accepts g) :
    g.routeWindow.Admissible :=
  (P.accepts_finiteTimeH1 h).routeWindow

end BurgersBadPattern

theorem localBadGermCapacityCertificate
    (g : BurgersBadGerm)
    (_hlocal : LocallyAdmissibleBurgersBadGerm g) :
    LocalBadGermCapacity g := by
  exact ⟨
    PeriodicH1State.energy g.profile,
    PeriodicH1State.energy_nonneg g.profile,
    le_rfl
  ⟩

structure LocalBadGermCapacityCertificateData where
  germ : BurgersBadGerm
  admissible : LocallyAdmissibleBurgersBadGerm germ

/-- The canonical bad-pattern object for the Burgers dataset. It classifies
exactly the route-local finite-time `H¹` blow-up germs. -/
def finiteTimeH1BlowUpBadPattern : BurgersBadPattern where
  kind := .finiteTimeH1BlowUp
  name := "finite-time H1 blow-up bad pattern"
  accepts := FiniteTimeH1BlowUpBadGerm
  accepts_finiteTimeH1 := by
    intro g h
    exact h
  local_capacity := by
    intro g h
    exact localBadGermCapacityCertificate g h.admissible

/-- A finite bad-pattern library for the Lock. Finiteness is represented by the
concrete list of pattern objects. -/
structure BurgersBadPatternLibrary where
  patterns : List BurgersBadPattern

namespace BurgersBadPatternLibrary

def Contains (L : BurgersBadPatternLibrary) (P : BurgersBadPattern) : Prop :=
  P ∈ L.patterns

def Complete (L : BurgersBadPatternLibrary) : Prop :=
  ∀ {g : BurgersBadGerm},
    FiniteTimeH1BlowUpBadGerm g →
      ∃ P : BurgersBadPattern, P ∈ L.patterns ∧ P.Accepts g

end BurgersBadPatternLibrary

/-- The small germ package committed to by the Burgers route. In the
0-truncated implementation, smallness means the relevant germ class is named by
a finite list of pattern templates. -/
def BurgersClassifiableGermPackageSmall
    (L : BurgersBadPatternLibrary) : Prop :=
  ∃ P : BurgersBadPattern,
    P ∈ L.patterns ∧ P.kind = .finiteTimeH1BlowUp

/-- The universal bad object is initialized when the library contains a
finite-time `H¹` pattern that accepts every local finite-time `H¹` blow-up
germ. -/
def UniversalFiniteTimeH1BadObjectInitialized
    (L : BurgersBadPatternLibrary) : Prop :=
  ∃ P : BurgersBadPattern,
    P ∈ L.patterns ∧
      P.kind = .finiteTimeH1BlowUp ∧
        ∀ g : BurgersBadGerm,
          FiniteTimeH1BlowUpBadGerm g → P.Accepts g

def burgersFiniteTimeH1BadPatternLibrary : BurgersBadPatternLibrary where
  patterns := [finiteTimeH1BlowUpBadPattern]

theorem burgersFiniteTimeH1BadPatternLibrary_small :
    BurgersClassifiableGermPackageSmall
      burgersFiniteTimeH1BadPatternLibrary := by
  exact ⟨finiteTimeH1BlowUpBadPattern, by simp [burgersFiniteTimeH1BadPatternLibrary], rfl⟩

theorem burgersFiniteTimeH1BadPatternLibrary_initialized :
    UniversalFiniteTimeH1BadObjectInitialized
      burgersFiniteTimeH1BadPatternLibrary := by
  refine ⟨finiteTimeH1BlowUpBadPattern, ?_, rfl, ?_⟩
  · simp [burgersFiniteTimeH1BadPatternLibrary]
  · intro g hg
    exact hg

theorem burgersFiniteTimeH1BadPatternLibrary_complete :
    BurgersBadPatternLibrary.Complete
      burgersFiniteTimeH1BadPatternLibrary := by
  intro g hg
  exact ⟨
    finiteTimeH1BlowUpBadPattern,
    by simp [burgersFiniteTimeH1BadPatternLibrary],
    hg
  ⟩

structure BurgersBadPatternLibraryCertificateData where
  library : BurgersBadPatternLibrary
  germ_small : BurgersClassifiableGermPackageSmall library
  universal_initialized : UniversalFiniteTimeH1BadObjectInitialized library
  complete : BurgersBadPatternLibrary.Complete library

def burgersFiniteTimeH1BadPatternLibraryCertificateData :
    BurgersBadPatternLibraryCertificateData where
  library := burgersFiniteTimeH1BadPatternLibrary
  germ_small := burgersFiniteTimeH1BadPatternLibrary_small
  universal_initialized := burgersFiniteTimeH1BadPatternLibrary_initialized
  complete := burgersFiniteTimeH1BadPatternLibrary_complete

def burgersGermFrameworkCertificate
    (cert : BurgersBadPatternLibraryCertificateData) :
    GermCertificate :=
  { node := .lock
    payload :=
      { libraryName := "Burgers finite-time H1 bad-germ package"
        smallnessWitness := BurgersClassifiableGermPackageSmall cert.library }
    meaning := BurgersClassifiableGermPackageSmall cert.library }

theorem burgersGermFrameworkCertificate_sound
    (cert : BurgersBadPatternLibraryCertificateData) :
    (burgersGermFrameworkCertificate cert).meaning :=
  cert.germ_small

def burgersInitialityFrameworkCertificate
    (cert : BurgersBadPatternLibraryCertificateData) :
    InitialityCertificate :=
  { node := .lock
    payload :=
      { universalBadName := "universal finite-time H1 Burgers bad object"
        initialityWitness :=
          UniversalFiniteTimeH1BadObjectInitialized cert.library }
    meaning := UniversalFiniteTimeH1BadObjectInitialized cert.library }

theorem burgersInitialityFrameworkCertificate_sound
    (cert : BurgersBadPatternLibraryCertificateData) :
    (burgersInitialityFrameworkCertificate cert).meaning :=
  cert.universal_initialized

def burgersCatLibFrameworkCertificate
    (cert : BurgersBadPatternLibraryCertificateData) :
    CatLibCertificate :=
  { node := .lock
    payload :=
      { libraryName := "Burgers finite bad-pattern library"
        completenessWitness := BurgersBadPatternLibrary.Complete cert.library }
    meaning := BurgersBadPatternLibrary.Complete cert.library }

theorem burgersCatLibFrameworkCertificate_sound
    (cert : BurgersBadPatternLibraryCertificateData) :
    (burgersCatLibFrameworkCertificate cert).meaning :=
  cert.complete

def localBadGermCapacityFrameworkCertificate
    (cert : LocalBadGermCapacityCertificateData) :
    CapacityCertificate :=
  { node := .geomCheck
    payload :=
      { singularSetName := "local Burgers bad-germ support"
        capacityValue := PeriodicH1State.energy cert.germ.profile
        negligible := LocalBadGermCapacity cert.germ }
    meaning := LocalBadGermCapacity cert.germ }

theorem localBadGermCapacityFrameworkCertificate_sound
    (cert : LocalBadGermCapacityCertificateData) :
    (localBadGermCapacityFrameworkCertificate cert).meaning :=
  localBadGermCapacityCertificate cert.germ cert.admissible

end

end Hypostructure.Backends.Burgers1D
