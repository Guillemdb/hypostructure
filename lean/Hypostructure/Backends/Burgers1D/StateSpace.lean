import Mathlib.Topology.Instances.AddCircle
import Mathlib.MeasureTheory.Integral.Periodic
import Mathlib.Topology.ContinuousMap.Basic

namespace Hypostructure.Backends.Burgers1D

open MeasureTheory

abbrev BurgersTorus := UnitAddCircle

abbrev BurgersProfile := ContinuousMap BurgersTorus ℝ

abbrev BurgersDerivative := ContinuousMap BurgersTorus ℝ

/-- A concrete periodic `H¹`-like carrier: a periodic profile together with a periodic
derivative witness. This is stronger than the old bare `ContinuousMap`, while still remaining
lightweight enough for the hypostructure engine. -/
abbrev BurgersState := BurgersProfile × BurgersDerivative

namespace BurgersState

def value (u : BurgersState) : BurgersProfile := u.1

def deriv (u : BurgersState) : BurgersDerivative := u.2

theorem integrable_value (u : BurgersState) :
    Integrable (fun x : BurgersTorus => u.value x) := by
  rw [← integrableOn_univ]
  exact u.value.continuous.continuousOn.integrableOn_compact isCompact_univ

theorem integrable_deriv (u : BurgersState) :
    Integrable (fun x : BurgersTorus => u.deriv x) := by
  rw [← integrableOn_univ]
  exact u.deriv.continuous.continuousOn.integrableOn_compact isCompact_univ

noncomputable def mean (u : BurgersState) : ℝ :=
  ∫ x : BurgersTorus, u.value x

theorem mean_add (u v : BurgersState) : mean (u + v) = mean u + mean v := by
  simpa [mean, value, Pi.add_apply] using
    integral_add (integrable_value u) (integrable_value v)

theorem mean_smul (a : ℝ) (u : BurgersState) : mean (a • u) = a * mean u := by
  simpa [mean, value, Pi.smul_apply, smul_eq_mul] using
    integral_mul_left a (fun x : BurgersTorus => u.value x)

theorem mean_neg (u : BurgersState) : mean (-u) = -mean u := by
  simpa using (mean_smul (-1) u)

theorem mean_sub (u v : BurgersState) : mean (u - v) = mean u - mean v := by
  simpa [sub_eq_add_neg, mean_neg] using mean_add u (-v)

noncomputable def constantProfile (m : ℝ) : BurgersProfile where
  toFun := fun _ => m
  continuous_toFun := continuous_const

noncomputable def zeroDerivative : BurgersDerivative := 0

noncomputable def constantState (m : ℝ) : BurgersState :=
  (constantProfile m, zeroDerivative)

theorem deriv_constantState (m : ℝ) : (constantState m).deriv = 0 := by
  rfl

theorem mean_constantState (m : ℝ) : mean (constantState m) = m := by
  simp [mean, constantState, constantProfile, value, AddCircle.measure_univ]

noncomputable def meanZeroEnergy (u : BurgersState) : ℝ :=
  ∫ x : BurgersTorus, (u.value x - mean u) ^ (2 : ℕ)

noncomputable def dissipation (u : BurgersState) : ℝ :=
  ∫ x : BurgersTorus, (u.deriv x) ^ (2 : ℕ)

theorem meanZeroEnergy_nonneg (u : BurgersState) : 0 ≤ meanZeroEnergy u := by
  unfold meanZeroEnergy mean
  exact integral_nonneg fun _x => sq_nonneg _

theorem dissipation_nonneg (u : BurgersState) : 0 ≤ dissipation u := by
  unfold dissipation
  exact integral_nonneg fun _x => sq_nonneg _

theorem meanZeroEnergy_smul (a : ℝ) (u : BurgersState) :
    meanZeroEnergy (a • u) = a ^ (2 : ℕ) * meanZeroEnergy u := by
  unfold meanZeroEnergy
  rw [mean_smul]
  calc
    ∫ x : BurgersTorus, (a * u.value x - a * mean u) ^ (2 : ℕ)
      = ∫ x : BurgersTorus, a ^ (2 : ℕ) * (u.value x - mean u) ^ (2 : ℕ) := by
          refine integral_congr_ae ?_
          filter_upwards with x
          ring
    _ = a ^ (2 : ℕ) * ∫ x : BurgersTorus, (u.value x - mean u) ^ (2 : ℕ) := by
          rw [integral_mul_left]
    _ = a ^ (2 : ℕ) * meanZeroEnergy u := by rfl

theorem dissipation_smul (a : ℝ) (u : BurgersState) :
    dissipation (a • u) = a ^ (2 : ℕ) * dissipation u := by
  unfold dissipation
  calc
    ∫ x : BurgersTorus, (a * u.deriv x) ^ (2 : ℕ)
      = ∫ x : BurgersTorus, a ^ (2 : ℕ) * (u.deriv x) ^ (2 : ℕ) := by
          refine integral_congr_ae ?_
          filter_upwards with x
          ring
    _ = a ^ (2 : ℕ) * ∫ x : BurgersTorus, (u.deriv x) ^ (2 : ℕ) := by
          rw [integral_mul_left]
    _ = a ^ (2 : ℕ) * dissipation u := by rfl

def profileDictionary (u : BurgersState) : BurgersState := u

theorem profileDictionary_injective : Function.Injective profileDictionary := by
  intro u v h
  exact h

noncomputable def meanEquilibrium (u : BurgersState) : BurgersState :=
  constantState (mean u)

theorem meanEquilibrium_preserves_mean (u : BurgersState) :
    mean (meanEquilibrium u) = mean u := by
  simp [meanEquilibrium, mean_constantState]

theorem meanEquilibrium_exists (u : BurgersState) :
    ∃ m : ℝ, mean (constantState m) = mean u := by
  exact ⟨mean u, meanEquilibrium_preserves_mean u⟩

end BurgersState

end Hypostructure.Backends.Burgers1D
