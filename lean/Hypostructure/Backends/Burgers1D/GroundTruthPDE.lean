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

/-- A Burgers solution curve on the ground-truth periodic `H¹` carrier. -/
structure BurgersSolutionCurve (nu : BurgersParameters) where
  eval : ℝ → PeriodicH1State
  timeRegularity : Prop

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

/-- A concrete replacement for the old permissive PDE package. The fields now
force any backend to provide a flow, solution curves, and proofs of the actual
PDE-facing predicates above. -/
class BurgersGroundTruthEvolutionPackage
    (nu : BurgersParameters) where
  flow : ℝ → PeriodicH1State → PeriodicH1State
  solutionCurve : PeriodicH1State → BurgersSolutionCurve nu
  solutionCurve_eval :
    ∀ u0 t, (solutionCurve u0).eval t = flow t u0
  solves :
    ∀ u0, PeriodicH1State.IsPeriodicH1 u0 →
      SolvesViscousBurgersWeak nu u0 (solutionCurve u0)
  globalH1 :
    ∀ u0 t, PeriodicH1State.IsPeriodicH1 u0 →
      PeriodicH1State.IsPeriodicH1 (flow t u0)
  unique :
    ∀ u0, PeriodicH1State.IsPeriodicH1 u0 →
      UniqueWeakBurgersSolution nu u0 (solutionCurve u0)
  smoothPositiveTime :
    ∀ u0 t, PeriodicH1State.IsPeriodicH1 u0 → 0 < t →
      SmoothAtPositiveTime (solutionCurve u0) t

theorem burgers_groundTruth_globalRegularity_from_package
    (nu : BurgersParameters)
    [BurgersGroundTruthEvolutionPackage nu] :
    BurgersGroundTruthGlobalRegularityStatement nu := by
  intro u0 hu0
  refine ⟨BurgersGroundTruthEvolutionPackage.solutionCurve (nu := nu) u0, ?_, ?_, ?_, ?_⟩
  · exact BurgersGroundTruthEvolutionPackage.solves (nu := nu) u0 hu0
  · intro t
    rw [BurgersGroundTruthEvolutionPackage.solutionCurve_eval (nu := nu) u0 t]
    exact BurgersGroundTruthEvolutionPackage.globalH1 (nu := nu) u0 t hu0
  · exact BurgersGroundTruthEvolutionPackage.unique (nu := nu) u0 hu0
  · intro t ht
    exact BurgersGroundTruthEvolutionPackage.smoothPositiveTime (nu := nu) u0 t hu0 ht

end

end Hypostructure.Backends.Burgers1D
