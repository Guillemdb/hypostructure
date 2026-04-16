import Hypostructure.Backends.Burgers1D.GroundTruthPDE

namespace Hypostructure.Backends.Burgers1D

noncomputable section

namespace TimeWindow

/-- Membership in a finite time window. -/
def Contains (W : TimeWindow) (t : ℝ) : Prop :=
  W.t0 ≤ t ∧ t ≤ W.t1

/-- Inclusion of finite time windows. -/
def Subwindow (V W : TimeWindow) : Prop :=
  W.t0 ≤ V.t0 ∧ V.t1 ≤ W.t1

theorem left_mem (W : TimeWindow) : W.Contains W.t0 :=
  ⟨le_rfl, W.ordered⟩

theorem right_mem (W : TimeWindow) : W.Contains W.t1 :=
  ⟨W.ordered, le_rfl⟩

theorem contains_of_subwindow
    {V W : TimeWindow}
    (hsub : V.Subwindow W)
    {t : ℝ}
    (ht : V.Contains t) :
    W.Contains t :=
  ⟨le_trans hsub.1 ht.1, le_trans ht.2 hsub.2⟩

def singleton (t : ℝ) : TimeWindow where
  t0 := t
  t1 := t
  ordered := le_rfl

theorem singleton_contains
    (t : ℝ) :
    (singleton t).Contains t :=
  ⟨le_rfl, le_rfl⟩

end TimeWindow

/-- A Burgers-side certified time window. This wrapper keeps Burgers and heat
windowed certificates distinct even when they use the same time interval. -/
structure BurgersWindow where
  time : TimeWindow

namespace BurgersWindow

def Contains (W : BurgersWindow) (t : ℝ) : Prop :=
  W.time.Contains t

def Subwindow (V W : BurgersWindow) : Prop :=
  V.time.Subwindow W.time

theorem contains_of_subwindow
    {V W : BurgersWindow}
    (hsub : V.Subwindow W)
    {t : ℝ}
    (ht : V.Contains t) :
    W.Contains t :=
  TimeWindow.contains_of_subwindow hsub ht

end BurgersWindow

/-- A heat-side certified time window. -/
structure HeatWindow where
  time : TimeWindow

namespace HeatWindow

def Contains (W : HeatWindow) (t : ℝ) : Prop :=
  W.time.Contains t

def Subwindow (V W : HeatWindow) : Prop :=
  V.time.Subwindow W.time

def ofBurgersWindow (W : BurgersWindow) : HeatWindow where
  time := W.time

theorem contains_of_subwindow
    {V W : HeatWindow}
    (hsub : V.Subwindow W)
    {t : ℝ}
    (ht : V.Contains t) :
    W.Contains t :=
  TimeWindow.contains_of_subwindow hsub ht

theorem contains_of_burgersWindow
    (W : BurgersWindow)
    {t : ℝ}
    (ht : W.Contains t) :
    (ofBurgersWindow W).Contains t :=
  ht

end HeatWindow

/-- A route-local bad-germ window. It records only local support and profile
data; it does not assert global existence or global absence of singularities. -/
structure RouteLocalBadGermWindow where
  time : TimeWindow
  centerSpace : BurgersTorus
  profile : PeriodicH1State

namespace RouteLocalBadGermWindow

def ContainsTime (W : RouteLocalBadGermWindow) (t : ℝ) : Prop :=
  W.time.Contains t

def Admissible (W : RouteLocalBadGermWindow) : Prop :=
  PeriodicH1State.IsPeriodicH1 W.profile

end RouteLocalBadGermWindow

def BurgersInitialConditionOnWindow
    {nu : BurgersParameters}
    (u0 : PeriodicH1State)
    (u : BurgersSolutionCurve nu)
    (W : BurgersWindow) : Prop :=
  W.Contains 0 → InitialCondition u0 u

def BurgersPeriodicBoundaryOnWindow
    {nu : BurgersParameters}
    (u : BurgersSolutionCurve nu)
    (W : BurgersWindow) : Prop :=
  ∀ t : ℝ, W.Contains t → PeriodicH1State.IsPeriodicH1 (u.eval t)

def WeakBurgersResidualOnWindow
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu)
    (W : BurgersWindow) : Prop :=
  ∀ φ : SmoothCompactTimePeriodicSpaceTest,
    φ.window = W.time →
      timeSpaceIntegralOn W.time
        (burgersWeakResidualIntegrand nu u φ) = 0

/-- Windowed weak Burgers solution predicate. Unlike
`SolvesViscousBurgersWeak`, this is restricted to a certified Burgers window and
does not assert boundary or residual facts on arbitrary times/windows. -/
def SolvesViscousBurgersWeakOnWindow
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (u : BurgersSolutionCurve nu)
    (W : BurgersWindow) : Prop :=
  BurgersInitialConditionOnWindow u0 u W ∧
    BurgersPeriodicBoundaryOnWindow u W ∧
    WeakBurgersResidualOnWindow nu u W

structure CertifiedBurgersLocalWindow
    (nu : BurgersParameters) where
  window : BurgersWindow
  initial : PeriodicH1State
  curve : BurgersSolutionCurve nu
  solvesOnWindow : SolvesViscousBurgersWeakOnWindow nu initial curve window

namespace CertifiedBurgersLocalWindow

theorem boundary_on_window
    {nu : BurgersParameters}
    (C : CertifiedBurgersLocalWindow nu) :
    BurgersPeriodicBoundaryOnWindow C.curve C.window :=
  C.solvesOnWindow.2.1

theorem residual_on_window
    {nu : BurgersParameters}
    (C : CertifiedBurgersLocalWindow nu)
    (φ : SmoothCompactTimePeriodicSpaceTest)
    (hφ : φ.window = C.window.time) :
    timeSpaceIntegralOn C.window.time
      (burgersWeakResidualIntegrand nu C.curve φ) = 0 :=
  C.solvesOnWindow.2.2 φ hφ

end CertifiedBurgersLocalWindow

def BurgersSolutionCurve.RestrictsToWindow
    {nu : BurgersParameters}
    (u : BurgersSolutionCurve nu)
    (W : BurgersWindow) : Prop :=
  ∀ t : ℝ, W.Contains t → PeriodicH1State.IsPeriodicH1 (u.eval t)

theorem PeriodicBoundary.restrictsToWindow
    {nu : BurgersParameters}
    {u : BurgersSolutionCurve nu}
    (hboundary : PeriodicBoundary u)
    (W : BurgersWindow) :
    u.RestrictsToWindow W := by
  intro t _ht
  exact hboundary t

theorem SolvesViscousBurgersWeak.restrictsToWindow
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {u : BurgersSolutionCurve nu}
    (hsol : SolvesViscousBurgersWeak nu u0 u)
    (W : BurgersWindow) :
    u.RestrictsToWindow W :=
  hsol.2.1.restrictsToWindow W

theorem InitialCondition.on_burgersWindow
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {u : BurgersSolutionCurve nu}
    (hinit : InitialCondition u0 u)
    (W : BurgersWindow) :
    BurgersInitialConditionOnWindow u0 u W :=
  fun _ => hinit

theorem PeriodicBoundary.on_burgersWindow
    {nu : BurgersParameters}
    {u : BurgersSolutionCurve nu}
    (hboundary : PeriodicBoundary u)
    (W : BurgersWindow) :
    BurgersPeriodicBoundaryOnWindow u W := by
  intro t _ht
  exact hboundary t

theorem WeakBurgersResidual.on_timeWindow
    {nu : BurgersParameters}
    {u : BurgersSolutionCurve nu}
    (hres : WeakBurgersResidual nu u)
    (W : TimeWindow)
    (φ : SmoothCompactTimePeriodicSpaceTest)
    (hφ : φ.window = W) :
    timeSpaceIntegralOn W
      (burgersWeakResidualIntegrand nu u φ) = 0 := by
  simpa [hφ] using hres φ

theorem WeakBurgersResidual.on_burgersWindow
    {nu : BurgersParameters}
    {u : BurgersSolutionCurve nu}
    (hres : WeakBurgersResidual nu u)
    (W : BurgersWindow)
    (φ : SmoothCompactTimePeriodicSpaceTest)
    (hφ : φ.window = W.time) :
    timeSpaceIntegralOn W.time
      (burgersWeakResidualIntegrand nu u φ) = 0 :=
  hres.on_timeWindow W.time φ hφ

theorem SolvesViscousBurgersWeak.on_burgersWindow
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {u : BurgersSolutionCurve nu}
    (hsol : SolvesViscousBurgersWeak nu u0 u)
    (W : BurgersWindow) :
    SolvesViscousBurgersWeakOnWindow nu u0 u W :=
  ⟨
    hsol.1.on_burgersWindow W,
    hsol.2.1.on_burgersWindow W,
    hsol.2.2.on_burgersWindow W
  ⟩

end

end Hypostructure.Backends.Burgers1D
