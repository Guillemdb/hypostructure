import Hypostructure.Backends.Burgers1D.GroundTruthState

import Mathlib.MeasureTheory.Integral.IntervalIntegral

namespace Hypostructure.Backends.Burgers1D

open MeasureTheory

open scoped ENNReal

noncomputable section

/-- A finite time window for local PDE certificates. -/
structure TimeWindow where
  t0 : ℝ
  t1 : ℝ
  ordered : t0 ≤ t1

/-- The time-space integral used by weak Burgers residuals and finite-window
local certificates. -/
def timeSpaceIntegralOn
    (W : TimeWindow)
    (F : ℝ → BurgersTorus → ℝ) : ℝ :=
  ∫ t in W.t0..W.t1, ∫ x : BurgersTorus, F t x

/-- Smooth compactly time-windowed test data with periodic spatial component.
The compact support is represented by the finite `window`; later phases can
refine this with a stronger support predicate without changing the residual
shape. -/
structure SmoothCompactTimePeriodicSpaceTest where
  window : TimeWindow
  value : ℝ → ContinuousMap BurgersTorus ℝ
  timeDeriv : ℝ → ContinuousMap BurgersTorus ℝ
  spaceDeriv : ℝ → ContinuousMap BurgersTorus ℝ
  /-- For each time, the spatial slice is a genuine smooth periodic test
  function. This ties `spaceDeriv` to `value` instead of allowing an arbitrary
  independently chosen spatial derivative. -/
  spaceTest : ℝ → SmoothPeriodicTestFunction
  value_eq_spaceTest : ∀ t : ℝ, value t = (spaceTest t).value
  spaceDeriv_eq_spaceTest : ∀ t : ℝ, spaceDeriv t = (spaceTest t).deriv
  /-- The time derivative field is a genuine pointwise time derivative of the
  test value. -/
  time_hasDerivAt : ∀ t : ℝ, ∀ x : BurgersTorus,
    HasDerivAt (fun s : ℝ => value s x) (timeDeriv t x) t
  /-- The test is supported away from nonpositive time. This is the weak-form
  convention used by the heat certificate at the initial endpoint: all
  integrations are over finite windows, but test data contributes only where
  the positive-time heat representative is differentiable. -/
  nonpositive_value_zero : ∀ t : ℝ, t ≤ 0 → value t = 0
  nonpositive_timeDeriv_zero : ∀ t : ℝ, t ≤ 0 → timeDeriv t = 0
  nonpositive_spaceDeriv_zero : ∀ t : ℝ, t ≤ 0 → spaceDeriv t = 0
  /-- The finite-window time integration-by-parts rule for compactly supported
  tests. Concrete test constructors must prove the endpoint/support condition
  needed for this identity; residual proofs may then use it generically. -/
  timeSpace_product_integration_by_parts :
    ∀ (a da : ℝ → ContinuousMap BurgersTorus ℝ),
      (∀ t : ℝ, ∀ x : BurgersTorus,
        HasDerivAt (fun s : ℝ => a s x) (da t x) t) →
        timeSpaceIntegralOn window
          (fun t x => da t x * value t x + a t x * timeDeriv t x) = 0
  /-- Positive-time version of the product integration-by-parts rule. This is
  the one used for heat curves with rough initial data, where the curve need not
  be classically differentiable at `t = 0` but tests vanish for `t ≤ 0`. -/
  timeSpace_product_integration_by_parts_positive :
    ∀ (a da : ℝ → ContinuousMap BurgersTorus ℝ),
      (∀ t : ℝ, 0 < t → ∀ x : BurgersTorus,
        HasDerivAt (fun s : ℝ => a s x) (da t x) t) →
        timeSpaceIntegralOn window
          (fun t x => da t x * value t x + a t x * timeDeriv t x) = 0
  constant_heat_cancellation : ∀ m : ℝ,
    timeSpaceIntegralOn window
      (fun t x => -m * timeDeriv t x) = 0
  constant_burgers_cancellation : ∀ m : ℝ,
    timeSpaceIntegralOn window
      (fun t x => -m * timeDeriv t x -
        ((m ^ (2 : ℕ)) / 2) * spaceDeriv t x) = 0

/-- A Burgers solution curve on the ground-truth periodic `H¹` carrier. -/
structure BurgersSolutionCurve (nu : BurgersParameters) where
  eval : ℝ → PeriodicH1State

def InitialCondition
    {nu : BurgersParameters}
    (u0 : PeriodicH1State)
    (u : BurgersSolutionCurve nu) : Prop :=
  u.eval 0 = u0

def PeriodicBoundary
    {nu : BurgersParameters}
    (u : BurgersSolutionCurve nu) : Prop :=
  ∀ t : ℝ, PeriodicH1State.IsPeriodicH1 (u.eval t)

def burgersWeakResidualIntegrand
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu)
    (φ : SmoothCompactTimePeriodicSpaceTest)
    (t : ℝ)
    (x : BurgersTorus) : ℝ :=
  -((u.eval t).value x) * φ.timeDeriv t x -
    (((u.eval t).value x) ^ (2 : ℕ) / 2) * φ.spaceDeriv t x +
    nu.viscosity * (u.eval t).weakDeriv x * φ.spaceDeriv t x

/-- Weak residual for `u_t + u u_x = nu u_xx`, integrated by parts against
periodic spatial test functions on finite time windows. -/
def WeakBurgersResidual
    (nu : BurgersParameters)
    (u : BurgersSolutionCurve nu) : Prop :=
  ∀ φ : SmoothCompactTimePeriodicSpaceTest,
    timeSpaceIntegralOn φ.window
      (burgersWeakResidualIntegrand nu u φ) = 0

def SolvesViscousBurgersWeak
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (u : BurgersSolutionCurve nu) : Prop :=
  InitialCondition u0 u ∧
    PeriodicBoundary u ∧
    WeakBurgersResidual nu u

def UniqueWeakBurgersSolution
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (u : BurgersSolutionCurve nu) : Prop :=
  ∀ v : BurgersSolutionCurve nu,
    SolvesViscousBurgersWeak nu u0 v →
      ∀ t : ℝ, v.eval t = u.eval t

def GlobalH1Solution
    {nu : BurgersParameters}
    (u : BurgersSolutionCurve nu) : Prop :=
  ∀ t : ℝ, PeriodicH1State.IsPeriodicH1 (u.eval t)

/-- Recursive weak-derivative chain used as the positive-time smoothness
predicate. It is intentionally concrete: every derivative order supplies the
next weak derivative and its own weak-derivative identity. -/
def HasWeakDerivativeOrder :
    ℕ → ContinuousMap BurgersTorus ℝ → Prop
  | 0, f =>
      Memℒp (fun x : BurgersTorus => f x) (2 : ℝ≥0∞) volume
  | n + 1, f =>
      ∃ df : ContinuousMap BurgersTorus ℝ,
        IsWeakDerivativeOnTorus f df ∧ HasWeakDerivativeOrder n df

def SmoothPeriodicState (u : PeriodicH1State) : Prop :=
  ∀ k : ℕ, HasWeakDerivativeOrder k u.value

def SmoothAtPositiveTime
    {nu : BurgersParameters}
    (u : BurgersSolutionCurve nu)
    (t : ℝ) : Prop :=
  SmoothPeriodicState (u.eval t)

/-- The ground-truth global regularity target. Unlike the scaffolded theorem,
this quantifies over the concrete periodic `H¹` carrier and requires the actual
weak Burgers residual, global `H¹`, uniqueness, and positive-time smoothness. -/
def BurgersGroundTruthGlobalRegularityStatement
    (nu : BurgersParameters) : Prop :=
  ∀ u0 : PeriodicH1State,
    PeriodicH1State.IsPeriodicH1 u0 →
      ∃ u : BurgersSolutionCurve nu,
        SolvesViscousBurgersWeak nu u0 u ∧
          GlobalH1Solution u ∧
          UniqueWeakBurgersSolution nu u0 u ∧
          ∀ t : ℝ, 0 < t → SmoothAtPositiveTime u t

end

end Hypostructure.Backends.Burgers1D
