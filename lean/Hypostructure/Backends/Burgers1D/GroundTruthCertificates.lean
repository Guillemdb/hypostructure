import Hypostructure.Backends.Burgers1D.GroundTruthPDE
import Hypostructure.Backends.Burgers1D.Analysis
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
  SolvesViscousBurgersWeak nu u0 u →
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

structure BurgersBadGerm where
  centerTime : ℝ
  centerSpace : BurgersTorus
  localWindow : TimeSpaceCylinder
  profile : PeriodicH1State

def LocallyAdmissibleBurgersBadGerm
    (g : BurgersBadGerm) : Prop :=
  PeriodicH1State.IsPeriodicH1 g.profile

/-- Local bad-germ capacity means the germ has a finite local energy bound.
This is deliberately local: it does not assert that the global singular set is
empty. -/
def LocalBadGermCapacity
    (g : BurgersBadGerm) : Prop :=
  ∃ C : ℝ, 0 ≤ C ∧ PeriodicH1State.energy g.profile ≤ C

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
