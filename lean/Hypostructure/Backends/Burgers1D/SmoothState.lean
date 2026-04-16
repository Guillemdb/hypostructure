import Hypostructure.Backends.Burgers1D.Analysis

open scoped Topology
open Set MeasureTheory intervalIntegral

namespace Hypostructure.Backends.Burgers1D

/-- Smooth periodic profiles on the universal cover `ℝ`, with first and second derivative data.
This is the natural pre-quotient state space for the genuine periodic heat/Cole-Hopf route. -/
structure SmoothPeriodicState where
  val : ℝ → ℝ
  periodic : Function.Periodic val 1
  val_cont : Continuous val
  deriv : ℝ → ℝ
  second : ℝ → ℝ
  deriv_cont : Continuous deriv
  second_cont : Continuous second
  hasDeriv_val : ∀ x : ℝ, HasDerivAt val (deriv x) x
  hasDeriv_deriv : ∀ x : ℝ, HasDerivAt deriv (second x) x

namespace SmoothPeriodicState

noncomputable def mean (u : SmoothPeriodicState) : ℝ :=
  ∫ x in (0 : ℝ)..1, u.val x

noncomputable def meanZeroEnergy (u : SmoothPeriodicState) : ℝ :=
  ∫ x in (0 : ℝ)..1, (u.val x - mean u) ^ (2 : ℕ)

noncomputable def dissipation (u : SmoothPeriodicState) : ℝ :=
  ∫ x in (0 : ℝ)..1, (u.deriv x) ^ (2 : ℕ)

def meanZero (u : SmoothPeriodicState) : Prop :=
  mean u = 0

def constant (m : ℝ) : SmoothPeriodicState where
  val := fun _ => m
  periodic := by
    intro x
    rfl
  val_cont := continuous_const
  deriv := fun _ => 0
  second := fun _ => 0
  deriv_cont := continuous_const
  second_cont := continuous_const
  hasDeriv_val := by
    intro x
    simpa using (hasDerivAt_const x m)
  hasDeriv_deriv := by
    intro x
    simpa using (hasDerivAt_const x (0 : ℝ))

theorem deriv_periodic (u : SmoothPeriodicState) :
    Function.Periodic u.deriv 1 :=
  periodic_deriv u.periodic u.hasDeriv_val

theorem second_periodic (u : SmoothPeriodicState) :
    Function.Periodic u.second 1 :=
  periodic_deriv (deriv_periodic u) u.hasDeriv_deriv

theorem mean_constant (m : ℝ) :
    mean (constant m) = m := by
  simp [mean, constant]

theorem meanZeroEnergy_nonneg (u : SmoothPeriodicState) : 0 ≤ meanZeroEnergy u := by
  unfold meanZeroEnergy
  exact intervalIntegral.integral_nonneg zero_le_one fun _ _ => sq_nonneg _

theorem dissipation_nonneg (u : SmoothPeriodicState) : 0 ≤ dissipation u := by
  unfold dissipation
  exact intervalIntegral.integral_nonneg zero_le_one fun _ _ => sq_nonneg _

end SmoothPeriodicState

end Hypostructure.Backends.Burgers1D
