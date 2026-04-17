import Hypostructure.Backends.Burgers1D.GroundTruthCertificates

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Framework

noncomputable section

/-- A heat solution curve on the same periodic `H¹` carrier used by the Burgers
backend. This is the local heat-side object used by the Cole-Hopf bridge. -/
structure PeriodicHeatCurve (nu : BurgersParameters) where
  eval : ℝ → PeriodicH1State

def HeatInitialCondition
    {nu : BurgersParameters}
    (u0 : PeriodicH1State)
    (v : PeriodicHeatCurve nu) : Prop :=
  v.eval 0 = u0

def HeatPeriodicBoundary
    {nu : BurgersParameters}
    (v : PeriodicHeatCurve nu) : Prop :=
  ∀ t : ℝ, PeriodicH1State.IsPeriodicH1 (v.eval t)

def heatWeakResidualIntegrand
    (nu : BurgersParameters)
    (v : PeriodicHeatCurve nu)
    (φ : SmoothCompactTimePeriodicSpaceTest)
    (t : ℝ)
    (x : BurgersTorus) : ℝ :=
  -((v.eval t).value x) * φ.timeDeriv t x +
    nu.viscosity * (v.eval t).weakDeriv x * φ.spaceDeriv t x

/-- Weak residual for the periodic heat equation `theta_t = nu theta_xx`. -/
def WeakHeatResidual
    (nu : BurgersParameters)
    (v : PeriodicHeatCurve nu) : Prop :=
  ∀ φ : SmoothCompactTimePeriodicSpaceTest,
    timeSpaceIntegralOn φ.window
      (heatWeakResidualIntegrand nu v φ) = 0

def SolvesPeriodicHeatWeak
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (v : PeriodicHeatCurve nu) : Prop :=
  HeatInitialCondition u0 v ∧
    HeatPeriodicBoundary v ∧
    WeakHeatResidual nu v

/-- Heat-side restriction of a curve to a certified local window. -/
def PeriodicHeatCurve.RestrictsToWindow
    {nu : BurgersParameters}
    (v : PeriodicHeatCurve nu)
    (W : HeatWindow) : Prop :=
  ∀ t : ℝ, W.Contains t → PeriodicH1State.IsPeriodicH1 (v.eval t)

theorem HeatPeriodicBoundary.restrictsToWindow
    {nu : BurgersParameters}
    {v : PeriodicHeatCurve nu}
    (hboundary : HeatPeriodicBoundary v)
    (W : HeatWindow) :
    v.RestrictsToWindow W := by
  intro t _ht
  exact hboundary t

theorem SolvesPeriodicHeatWeak.restrictsToWindow
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {v : PeriodicHeatCurve nu}
    (hsol : SolvesPeriodicHeatWeak nu u0 v)
    (W : HeatWindow) :
    v.RestrictsToWindow W :=
  hsol.2.1.restrictsToWindow W

theorem WeakHeatResidual.on_timeWindow
    {nu : BurgersParameters}
    {v : PeriodicHeatCurve nu}
    (hres : WeakHeatResidual nu v)
    (W : TimeWindow)
    (φ : SmoothCompactTimePeriodicSpaceTest)
    (hφ : φ.window = W) :
    timeSpaceIntegralOn W
      (heatWeakResidualIntegrand nu v φ) = 0 := by
  simpa [hφ] using hres φ

theorem WeakHeatResidual.on_heatWindow
    {nu : BurgersParameters}
    {v : PeriodicHeatCurve nu}
    (hres : WeakHeatResidual nu v)
    (W : HeatWindow)
    (φ : SmoothCompactTimePeriodicSpaceTest)
    (hφ : φ.window = W.time) :
    timeSpaceIntegralOn W.time
      (heatWeakResidualIntegrand nu v φ) = 0 :=
  hres.on_timeWindow W.time φ hφ

/-- A heat solution restricted to a certified heat window. -/
structure CertifiedHeatWindow
    (nu : BurgersParameters) where
  window : HeatWindow
  initial : PeriodicH1State
  curve : PeriodicHeatCurve nu
  initial_on_window : window.Contains 0 → HeatInitialCondition initial curve
  boundary_on_window : curve.RestrictsToWindow window

namespace CertifiedHeatWindow

theorem restricts_to_window
    {nu : BurgersParameters}
    (C : CertifiedHeatWindow nu) :
    C.curve.RestrictsToWindow C.window :=
  C.boundary_on_window

end CertifiedHeatWindow

def UniquePeriodicHeatSolution
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (v : PeriodicHeatCurve nu) : Prop :=
  ∀ w : PeriodicHeatCurve nu,
    SolvesPeriodicHeatWeak nu u0 w →
      ∀ t : ℝ, w.eval t = v.eval t

def HeatGlobalH1
    {nu : BurgersParameters}
    (v : PeriodicHeatCurve nu) : Prop :=
  ∀ t : ℝ, PeriodicH1State.IsPeriodicH1 (v.eval t)

def HeatSmoothAtPositiveTime
    {nu : BurgersParameters}
    (v : PeriodicHeatCurve nu)
    (t : ℝ) : Prop :=
  SmoothPeriodicState (v.eval t)

def HeatEnergyContraction
    (_nu : BurgersParameters)
    (flow : ℝ → PeriodicH1State → PeriodicH1State) : Prop :=
  ∀ u0 : PeriodicH1State, ∀ t : ℝ, 0 ≤ t →
    PeriodicH1State.energy (flow t u0) ≤ PeriodicH1State.energy u0

def HeatDissipationContraction
    (_nu : BurgersParameters)
    (flow : ℝ → PeriodicH1State → PeriodicH1State) : Prop :=
  ∀ u0 : PeriodicH1State, ∀ t : ℝ, 0 ≤ t →
    PeriodicH1State.derivativeEnergy (flow t u0) ≤
      PeriodicH1State.derivativeEnergy u0

/-- Windowed heat residual exported to the hypostructure certificate chain. -/
def HeatWindowResidual
    (nu : BurgersParameters)
    (C : CertifiedHeatWindow nu) : Prop :=
  ∀ φ : SmoothCompactTimePeriodicSpaceTest,
    φ.window = C.window.time →
      timeSpaceIntegralOn C.window.time
        (heatWeakResidualIntegrand nu C.curve φ) = 0

/-- Windowed uniqueness: another heat solution agrees with the certified one on
the certified window only. -/
def HeatWindowUniqueness
    (nu : BurgersParameters)
    (C : CertifiedHeatWindow nu) : Prop :=
  ∀ w : PeriodicHeatCurve nu,
    SolvesPeriodicHeatWeak nu C.initial w →
      ∀ t : ℝ, C.window.Contains t → w.eval t = C.curve.eval t

def HeatWindowSmoothing
    {nu : BurgersParameters}
    (C : CertifiedHeatWindow nu) : Prop :=
  ∀ t : ℝ, C.window.Contains t → 0 < t →
    HeatSmoothAtPositiveTime C.curve t

def HeatWindowEnergyContraction
    {nu : BurgersParameters}
    (C : CertifiedHeatWindow nu) : Prop :=
  ∀ t : ℝ, C.window.Contains t → 0 ≤ t →
    PeriodicH1State.energy (C.curve.eval t) ≤
      PeriodicH1State.energy C.initial

def HeatWindowDissipationContraction
    {nu : BurgersParameters}
    (C : CertifiedHeatWindow nu) : Prop :=
  ∀ t : ℝ, C.window.Contains t → 0 ≤ t →
    PeriodicH1State.derivativeEnergy (C.curve.eval t) ≤
      PeriodicH1State.derivativeEnergy C.initial

def RouteLocalBadGermWindow.SupportedInHeatWindow
    (G : RouteLocalBadGermWindow)
    (W : HeatWindow) : Prop :=
  ∀ t : ℝ, G.ContainsTime t → W.Contains t

def HeatBadGermWindowExclusion
    (W : HeatWindow)
    (Forbidden : RouteLocalBadGermWindow → Prop) : Prop :=
  ∀ G : RouteLocalBadGermWindow,
    G.SupportedInHeatWindow W → G.Admissible → Forbidden G → False

/-- Local heat-side bad germ ruled out by a heat window certificate: the germ
contains a positive time at which the certified heat curve is not smooth. The
support hypothesis in `HeatBadGermWindowExclusion` then places that time inside
the certified heat window, where `HeatWindowSmoothing` applies. -/
def HeatForbiddenBadGerm
    {nu : BurgersParameters}
    (C : CertifiedHeatWindow nu)
    (G : RouteLocalBadGermWindow) : Prop :=
  ∃ t : ℝ, G.ContainsTime t ∧ 0 < t ∧ ¬ HeatSmoothAtPositiveTime C.curve t

theorem HeatForbiddenBadGerm.excluded
    {nu : BurgersParameters}
    {C : CertifiedHeatWindow nu}
    (hsmooth : HeatWindowSmoothing C) :
    HeatBadGermWindowExclusion C.window (HeatForbiddenBadGerm C) := by
  intro G hsupport _hadm hforbidden
  rcases hforbidden with ⟨t, htG, htpos, hnot_smooth⟩
  exact hnot_smooth (hsmooth t (hsupport t htG) htpos)

theorem routeLocalBadGermWindow_localCapacity
    (G : RouteLocalBadGermWindow)
    (_hG : G.Admissible) :
    G.LocalCapacity :=
  ⟨PeriodicH1State.energy G.profile,
    PeriodicH1State.energy_nonneg G.profile,
    le_rfl⟩

/-- The exported local heat certificate required by the template. It exposes
only windowed residual, uniqueness, smoothing, contraction, and heat bad-germ
exclusion facts. -/
structure LocalHeatWindowCertificate
    (nu : BurgersParameters) where
  certified : CertifiedHeatWindow nu
  residual : HeatWindowResidual nu certified
  unique : HeatWindowUniqueness nu certified
  smooth : HeatWindowSmoothing certified
  energy_contraction : HeatWindowEnergyContraction certified
  dissipation_contraction : HeatWindowDissipationContraction certified
  excludes_heat_bad_germs :
    HeatBadGermWindowExclusion certified.window (HeatForbiddenBadGerm certified)

def LocalHeatWindowCertificate.forbidden_bad_germ
    {nu : BurgersParameters}
    (H : LocalHeatWindowCertificate nu) :
    RouteLocalBadGermWindow → Prop :=
  HeatForbiddenBadGerm H.certified

def LocalHeatWindowCertificate.residualStatement
    {nu : BurgersParameters}
    (H : LocalHeatWindowCertificate nu) : Prop :=
  HeatWindowResidual nu H.certified

def LocalHeatWindowCertificate.smoothingStatement
    {nu : BurgersParameters}
    (H : LocalHeatWindowCertificate nu) : Prop :=
  HeatWindowSmoothing H.certified

def LocalHeatWindowCertificate.uniquenessStatement
    {nu : BurgersParameters}
    (H : LocalHeatWindowCertificate nu) : Prop :=
  HeatWindowUniqueness nu H.certified

def LocalHeatWindowCertificate.contractionStatement
    {nu : BurgersParameters}
    (H : LocalHeatWindowCertificate nu) : Prop :=
  HeatWindowEnergyContraction H.certified ∧
    HeatWindowDissipationContraction H.certified

def LocalHeatWindowCertificate.badGermExclusionStatement
    {nu : BurgersParameters}
    (H : LocalHeatWindowCertificate nu) : Prop :=
  HeatBadGermWindowExclusion H.certified.window H.forbidden_bad_germ

def LocalHeatWindowCertificate.certificateStatement
    {nu : BurgersParameters}
    (H : LocalHeatWindowCertificate nu) : Prop :=
  H.residualStatement ∧
    H.smoothingStatement ∧
    H.uniquenessStatement ∧
    H.contractionStatement ∧
    H.badGermExclusionStatement

def localHeatSmoothingFrameworkCertificate
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) :
    HeatSmoothCertificate :=
  { node := .lock
    payload :=
      { semigroupName := "ground-truth periodic heat semigroup restricted to a certified window"
        smoothingStatement := H.smoothingStatement
        uniquenessStatement := H.uniquenessStatement }
    meaning := H.certificateStatement }

theorem localHeatSmoothingFrameworkCertificate_sound
    (nu : BurgersParameters)
    (H : LocalHeatWindowCertificate nu) :
    (localHeatSmoothingFrameworkCertificate nu H).meaning := by
  exact ⟨
    H.residual,
    H.smooth,
    H.unique,
    ⟨H.energy_contraction, H.dissipation_contraction⟩,
    H.excludes_heat_bad_germs
  ⟩

theorem localHeat_window_residual
    {nu : BurgersParameters}
    (H : LocalHeatWindowCertificate nu) :
    HeatWindowResidual nu H.certified :=
  H.residual

theorem localHeat_window_uniqueness
    {nu : BurgersParameters}
    (H : LocalHeatWindowCertificate nu) :
    HeatWindowUniqueness nu H.certified :=
  H.unique

theorem localHeat_window_smoothing
    {nu : BurgersParameters}
    (H : LocalHeatWindowCertificate nu) :
    HeatWindowSmoothing H.certified :=
  H.smooth

theorem LocalHeatWindowCertificate.no_forbidden_heat_bad_germ
    {nu : BurgersParameters}
    (H : LocalHeatWindowCertificate nu)
    (G : RouteLocalBadGermWindow)
    (hsupport : G.SupportedInHeatWindow H.certified.window)
    (hadm : G.Admissible)
    (hforbidden : H.forbidden_bad_germ G) :
    False :=
  H.excludes_heat_bad_germs G hsupport hadm hforbidden

/-- All-time heat semigroup backend data. This is a construction source for
local certificates, not the exported `K_HeatSmooth^+` payload. -/
structure PeriodicHeatSemigroupBackend
    (nu : BurgersParameters) where
  flow : ℝ → PeriodicH1State → PeriodicH1State
  curve : PeriodicH1State → PeriodicHeatCurve nu
  curve_eval : ∀ u0 t, (curve u0).eval t = flow t u0
  solves : ∀ u0 : PeriodicH1State,
    PeriodicH1State.IsPeriodicH1 u0 →
      SolvesPeriodicHeatWeak nu u0 (curve u0)
  energy_contraction : HeatEnergyContraction nu flow
  dissipation_contraction : HeatDissipationContraction nu flow
  unique : ∀ u0 : PeriodicH1State,
    PeriodicH1State.IsPeriodicH1 u0 →
      UniquePeriodicHeatSolution nu u0 (curve u0)
  smooth_positive_time : ∀ u0 : PeriodicH1State, ∀ t : ℝ,
    PeriodicH1State.IsPeriodicH1 u0 → 0 < t →
      HeatSmoothAtPositiveTime (curve u0) t

def PeriodicHeatSemigroupBackend.windowCertificate
    {nu : BurgersParameters}
    (H : PeriodicHeatSemigroupBackend nu)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    LocalHeatWindowCertificate nu where
  certified :=
    { window := W
      initial := u0
      curve := H.curve u0
      initial_on_window := by
        intro _h0
        exact (H.solves u0 hu0).1
      boundary_on_window := by
        exact (H.solves u0 hu0).2.1.restrictsToWindow W }
  residual := by
    intro φ hφ
    exact (H.solves u0 hu0).2.2.on_heatWindow W φ hφ
  unique := by
    intro w hw t _ht
    exact H.unique u0 hu0 w hw t
  smooth := by
    intro t _ht htpos
    exact H.smooth_positive_time u0 t hu0 htpos
  energy_contraction := by
    intro t _ht ht_nonneg
    simpa [H.curve_eval] using H.energy_contraction u0 t ht_nonneg
  dissipation_contraction := by
    intro t _ht ht_nonneg
    simpa [H.curve_eval] using H.dissipation_contraction u0 t ht_nonneg
  excludes_heat_bad_germs := by
    exact HeatForbiddenBadGerm.excluded (by
      intro t _ht htpos
      exact H.smooth_positive_time u0 t hu0 htpos)

theorem localHeat_solves
    {nu : BurgersParameters}
    (H : PeriodicHeatSemigroupBackend nu)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    SolvesPeriodicHeatWeak nu u0 (H.curve u0) :=
  H.solves u0 hu0

theorem localHeat_globalH1
    {nu : BurgersParameters}
    (H : PeriodicHeatSemigroupBackend nu)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicH1State.IsPeriodicH1 (H.flow t u0) :=
  by
    have hboundary : PeriodicH1State.IsPeriodicH1 ((H.curve u0).eval t) :=
      (H.solves u0 hu0).2.1 t
    simpa [H.curve_eval u0 t] using hboundary

end

end Hypostructure.Backends.Burgers1D
