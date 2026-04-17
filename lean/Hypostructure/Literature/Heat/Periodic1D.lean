import Hypostructure.Backends.Burgers1D.GroundTruthHeat
import Hypostructure.Literature.Analysis.PeriodicPoincare1D
import Mathlib.Analysis.Calculus.SmoothSeries
import Mathlib.NumberTheory.ModularForms.JacobiTheta.TwoVariable
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic

namespace Hypostructure.Literature.Heat.Periodic1D

open Hypostructure.Backends.Burgers1D
open Hypostructure.Literature.Analysis.PeriodicPoincare1D
open MeasureTheory
open Filter
open scoped ENNReal
open scoped Topology

noncomputable section

/-- Fourier frequency of the `n`th periodic heat mode on the unit torus. -/
def periodicHeat1D_modeFrequency (n : ℤ) : ℝ :=
  (2 * Real.pi * (n : ℝ)) ^ (2 : ℕ)

/-- Exponent of the periodic heat multiplier on one Fourier mode. -/
def periodicHeat1D_modeExponent
    (nu : BurgersParameters)
    (t : ℝ)
    (n : ℤ) : ℝ :=
  -(nu.viscosity * periodicHeat1D_modeFrequency n * t)

/-- The scalar Fourier multiplier for the one-dimensional periodic heat flow. -/
def periodicHeat1D_modeMultiplier
    (nu : BurgersParameters)
    (t : ℝ)
    (n : ℤ) : ℝ :=
  Real.exp (periodicHeat1D_modeExponent nu t n)

theorem periodicHeat1D_modeFrequency_nonneg
    (n : ℤ) :
    0 ≤ periodicHeat1D_modeFrequency n := by
  unfold periodicHeat1D_modeFrequency
  exact sq_nonneg _

theorem periodicHeat1D_modeFrequency_pos
    (n : PeriodicH1NonzeroMode) :
    0 < periodicHeat1D_modeFrequency n.1 := by
  have hn : (n.1 : ℝ) ≠ 0 := by exact_mod_cast n.2
  have hmul : 2 * Real.pi * (n.1 : ℝ) ≠ 0 := by
    exact mul_ne_zero (mul_ne_zero (by norm_num) (ne_of_gt Real.pi_pos)) hn
  simpa [periodicHeat1D_modeFrequency] using (sq_pos_of_ne_zero hmul)

theorem periodicHeat1D_modeExponent_zero_time
    (nu : BurgersParameters)
    (n : ℤ) :
    periodicHeat1D_modeExponent nu 0 n = 0 := by
  simp [periodicHeat1D_modeExponent]

theorem periodicHeat1D_modeMultiplier_zero_time
    (nu : BurgersParameters)
    (n : ℤ) :
    periodicHeat1D_modeMultiplier nu 0 n = 1 := by
  simp [periodicHeat1D_modeMultiplier, periodicHeat1D_modeExponent_zero_time]

theorem periodicHeat1D_modeExponent_nonpos
    (nu : BurgersParameters)
    (t : ℝ)
    (n : ℤ)
    (ht : 0 ≤ t) :
    periodicHeat1D_modeExponent nu t n ≤ 0 := by
  unfold periodicHeat1D_modeExponent
  have hnu : 0 ≤ nu.viscosity := le_of_lt nu.viscosity_pos
  have hfreq : 0 ≤ periodicHeat1D_modeFrequency n :=
    periodicHeat1D_modeFrequency_nonneg n
  have hprod : 0 ≤ nu.viscosity * periodicHeat1D_modeFrequency n * t :=
    mul_nonneg (mul_nonneg hnu hfreq) ht
  exact neg_nonpos.mpr hprod

theorem periodicHeat1D_modeMultiplier_nonneg
    (nu : BurgersParameters)
    (t : ℝ)
    (n : ℤ) :
    0 ≤ periodicHeat1D_modeMultiplier nu t n := by
  exact Real.exp_nonneg _

theorem periodicHeat1D_modeMultiplier_le_one
    (nu : BurgersParameters)
    (t : ℝ)
    (n : ℤ)
    (ht : 0 ≤ t) :
    periodicHeat1D_modeMultiplier nu t n ≤ 1 := by
  exact Real.exp_le_one_iff.mpr
    (periodicHeat1D_modeExponent_nonpos nu t n ht)

theorem periodicHeat1D_modeExponent_neg
    (nu : BurgersParameters)
    (t : ℝ)
    (n : PeriodicH1NonzeroMode)
    (ht : 0 < t) :
    periodicHeat1D_modeExponent nu t n.1 < 0 := by
  unfold periodicHeat1D_modeExponent
  have hprod : 0 < nu.viscosity * periodicHeat1D_modeFrequency n.1 * t :=
    mul_pos (mul_pos nu.viscosity_pos (periodicHeat1D_modeFrequency_pos n)) ht
  exact neg_lt_zero.mpr hprod

theorem periodicHeat1D_modeMultiplier_lt_one
    (nu : BurgersParameters)
    (t : ℝ)
    (n : PeriodicH1NonzeroMode)
    (ht : 0 < t) :
    periodicHeat1D_modeMultiplier nu t n.1 < 1 := by
  exact Real.exp_lt_one_iff.mpr
    (periodicHeat1D_modeExponent_neg nu t n ht)

theorem periodicHeat1D_modeMultiplier_abs_le_one
    (nu : BurgersParameters)
    (t : ℝ)
    (n : ℤ)
    (ht : 0 ≤ t) :
    |periodicHeat1D_modeMultiplier nu t n| ≤ 1 := by
  rw [abs_of_nonneg (periodicHeat1D_modeMultiplier_nonneg nu t n)]
  exact periodicHeat1D_modeMultiplier_le_one nu t n ht

theorem periodicHeat1D_modeMultiplier_sq_le_one
    (nu : BurgersParameters)
    (t : ℝ)
    (n : ℤ)
    (ht : 0 ≤ t) :
    (periodicHeat1D_modeMultiplier nu t n) ^ (2 : ℕ) ≤ 1 := by
  have hnonneg := periodicHeat1D_modeMultiplier_nonneg nu t n
  have hle := periodicHeat1D_modeMultiplier_le_one nu t n ht
  nlinarith [sq_nonneg (periodicHeat1D_modeMultiplier nu t n - 1)]

theorem periodicHeat1D_modeExponent_add_time
    (nu : BurgersParameters)
    (t s : ℝ)
    (n : ℤ) :
    periodicHeat1D_modeExponent nu (t + s) n =
      periodicHeat1D_modeExponent nu t n + periodicHeat1D_modeExponent nu s n := by
  unfold periodicHeat1D_modeExponent
  ring

theorem periodicHeat1D_modeMultiplier_add_time
    (nu : BurgersParameters)
    (t s : ℝ)
    (n : ℤ) :
    periodicHeat1D_modeMultiplier nu (t + s) n =
      periodicHeat1D_modeMultiplier nu t n *
        periodicHeat1D_modeMultiplier nu s n := by
  unfold periodicHeat1D_modeMultiplier
  rw [periodicHeat1D_modeExponent_add_time, Real.exp_add]

theorem periodicHeat1D_modeMultiplier_mul_neg_time
    (nu : BurgersParameters)
    (t : ℝ)
    (n : ℤ) :
    periodicHeat1D_modeMultiplier nu t n *
      periodicHeat1D_modeMultiplier nu (-t) n = 1 := by
  have h := periodicHeat1D_modeMultiplier_add_time nu t (-t) n
  have hzero : t + -t = 0 := by ring
  rw [hzero, periodicHeat1D_modeMultiplier_zero_time] at h
  exact h.symm

theorem periodicHeat1D_modeMultiplier_complex_mul_neg_time
    (nu : BurgersParameters)
    (t : ℝ)
    (n : ℤ) :
    (periodicHeat1D_modeMultiplier nu t n : ℂ) *
      (periodicHeat1D_modeMultiplier nu (-t) n : ℂ) = 1 := by
  have hreal := periodicHeat1D_modeMultiplier_mul_neg_time nu t n
  exact_mod_cast hreal

theorem periodicHeat1D_modeFrequency_neg
    (n : ℤ) :
    periodicHeat1D_modeFrequency (-n) = periodicHeat1D_modeFrequency n := by
  simp [periodicHeat1D_modeFrequency]

theorem periodicHeat1D_modeMultiplier_neg
    (nu : BurgersParameters)
    (t : ℝ)
    (n : ℤ) :
    periodicHeat1D_modeMultiplier nu t (-n) =
      periodicHeat1D_modeMultiplier nu t n := by
  simp [periodicHeat1D_modeMultiplier, periodicHeat1D_modeExponent,
    periodicHeat1D_modeFrequency]

theorem periodicHeat1D_modeExponent_hasDerivAt
    (nu : BurgersParameters)
    (n : ℤ)
    (t : ℝ) :
    HasDerivAt (fun s : ℝ => periodicHeat1D_modeExponent nu s n)
      (-(nu.viscosity * periodicHeat1D_modeFrequency n)) t := by
  unfold periodicHeat1D_modeExponent
  simpa using
    (hasDerivAt_const
      (x := t) (c := -(nu.viscosity * periodicHeat1D_modeFrequency n))).mul
      (hasDerivAt_id t)

theorem periodicHeat1D_modeMultiplier_hasDerivAt
    (nu : BurgersParameters)
    (n : ℤ)
    (t : ℝ) :
    HasDerivAt (fun s : ℝ => periodicHeat1D_modeMultiplier nu s n)
      (-(nu.viscosity * periodicHeat1D_modeFrequency n) *
        periodicHeat1D_modeMultiplier nu t n) t := by
  have hExp := periodicHeat1D_modeExponent_hasDerivAt nu n t
  simpa [periodicHeat1D_modeMultiplier, mul_comm] using hExp.exp

theorem periodicHeat1D_modeMultiplier_anti_mono_time
    (nu : BurgersParameters)
    (n : ℤ)
    {s t : ℝ}
    (hst : s ≤ t) :
    periodicHeat1D_modeMultiplier nu t n ≤
      periodicHeat1D_modeMultiplier nu s n := by
  unfold periodicHeat1D_modeMultiplier periodicHeat1D_modeExponent
  apply Real.exp_le_exp.mpr
  have hnonneg : 0 ≤ nu.viscosity * periodicHeat1D_modeFrequency n := by
    exact mul_nonneg (le_of_lt nu.viscosity_pos)
      (periodicHeat1D_modeFrequency_nonneg n)
  nlinarith

/-- Mathlib-dischargeable Fourier multiplier layer for the periodic heat
equation. This is not yet the full heat semigroup construction on `H¹`; it is
the reusable mode-level theory needed for the contraction and semigroup parts
of that construction. -/
structure PeriodicHeat1DFourierModeTheory where
  frequency_nonneg : ∀ n : ℤ, 0 ≤ periodicHeat1D_modeFrequency n
  frequency_pos_nonzero :
    ∀ n : PeriodicH1NonzeroMode, 0 < periodicHeat1D_modeFrequency n.1
  multiplier_zero_time :
    ∀ (nu : BurgersParameters) (n : ℤ),
      periodicHeat1D_modeMultiplier nu 0 n = 1
  multiplier_nonneg :
    ∀ (nu : BurgersParameters) (t : ℝ) (n : ℤ),
      0 ≤ periodicHeat1D_modeMultiplier nu t n
  multiplier_le_one :
    ∀ (nu : BurgersParameters) (t : ℝ) (n : ℤ),
      0 ≤ t → periodicHeat1D_modeMultiplier nu t n ≤ 1
  multiplier_lt_one_nonzero :
    ∀ (nu : BurgersParameters) (t : ℝ) (n : PeriodicH1NonzeroMode),
      0 < t → periodicHeat1D_modeMultiplier nu t n.1 < 1
  multiplier_abs_le_one :
    ∀ (nu : BurgersParameters) (t : ℝ) (n : ℤ),
      0 ≤ t → |periodicHeat1D_modeMultiplier nu t n| ≤ 1
  multiplier_sq_le_one :
    ∀ (nu : BurgersParameters) (t : ℝ) (n : ℤ),
      0 ≤ t → (periodicHeat1D_modeMultiplier nu t n) ^ (2 : ℕ) ≤ 1
  multiplier_add_time :
    ∀ (nu : BurgersParameters) (t s : ℝ) (n : ℤ),
      periodicHeat1D_modeMultiplier nu (t + s) n =
        periodicHeat1D_modeMultiplier nu t n *
          periodicHeat1D_modeMultiplier nu s n

/-- Proved mode-level periodic heat theory. The remaining heat literature
boundary starts only after this point: constructing the H¹ heat curve and
proving its weak residual, uniqueness, smoothing, and windowed contractions. -/
def periodicHeat1D_fourierModeTheory_literature :
    PeriodicHeat1DFourierModeTheory where
  frequency_nonneg := periodicHeat1D_modeFrequency_nonneg
  frequency_pos_nonzero := periodicHeat1D_modeFrequency_pos
  multiplier_zero_time := periodicHeat1D_modeMultiplier_zero_time
  multiplier_nonneg := periodicHeat1D_modeMultiplier_nonneg
  multiplier_le_one := periodicHeat1D_modeMultiplier_le_one
  multiplier_lt_one_nonzero := periodicHeat1D_modeMultiplier_lt_one
  multiplier_abs_le_one := periodicHeat1D_modeMultiplier_abs_le_one
  multiplier_sq_le_one := periodicHeat1D_modeMultiplier_sq_le_one
  multiplier_add_time := periodicHeat1D_modeMultiplier_add_time

/-- Formal heat evolution of the value Fourier coefficient of mode `n`. -/
def periodicHeat1D_evolvedValueFourierCoeff
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (n : ℤ) : ℂ :=
  (periodicHeat1D_modeMultiplier nu t n : ℂ) *
    periodicH1_valueFourierCoeff u0 n

/-- Formal heat evolution of the weak-derivative Fourier coefficient of mode
`n`. This is the derivative-side coefficient family used for the coefficient
contraction theorem; reconstruction of a state with these coefficients is the
remaining heat backend work. -/
def periodicHeat1D_evolvedDerivativeFourierCoeff
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (n : ℤ) : ℂ :=
  (periodicHeat1D_modeMultiplier nu t n : ℂ) *
    periodicH1_derivativeFourierCoeff u0 n

def periodicHeat1D_evolvedValueModeEnergy
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (n : ℤ) : ℝ :=
  ‖periodicHeat1D_evolvedValueFourierCoeff nu u0 t n‖ ^ (2 : ℕ)

def periodicHeat1D_evolvedDerivativeModeEnergy
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (n : ℤ) : ℝ :=
  ‖periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n‖ ^ (2 : ℕ)

theorem periodicHeat1D_realScalar_complex_norm_sq
    (m : ℝ)
    (z : ℂ) :
    ‖((m : ℂ) * z)‖ ^ (2 : ℕ) =
      m ^ (2 : ℕ) * ‖z‖ ^ (2 : ℕ) := by
  have h := Complex.normSq_mul (m : ℂ) z
  rw [Complex.normSq_eq_norm_sq ((m : ℂ) * z)] at h
  rw [Complex.normSq_eq_norm_sq z] at h
  rw [Complex.normSq_ofReal] at h
  simpa [pow_two] using h

theorem periodicHeat1D_evolvedValueFourierCoeff_zero_time
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (n : ℤ) :
    periodicHeat1D_evolvedValueFourierCoeff nu u0 0 n =
      periodicH1_valueFourierCoeff u0 n := by
  simp [periodicHeat1D_evolvedValueFourierCoeff,
    periodicHeat1D_modeMultiplier_zero_time]

theorem periodicHeat1D_evolvedDerivativeFourierCoeff_zero_time
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (n : ℤ) :
    periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 0 n =
      periodicH1_derivativeFourierCoeff u0 n := by
  simp [periodicHeat1D_evolvedDerivativeFourierCoeff,
    periodicHeat1D_modeMultiplier_zero_time]

theorem periodicHeat1D_evolvedValueFourierCoeff_neg_eq_star
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (n : ℤ) :
    periodicHeat1D_evolvedValueFourierCoeff nu u0 t (-n) =
      star (periodicHeat1D_evolvedValueFourierCoeff nu u0 t n) := by
  calc
    periodicHeat1D_evolvedValueFourierCoeff nu u0 t (-n)
        = (periodicHeat1D_modeMultiplier nu t n : ℂ) *
            star (periodicH1_valueFourierCoeff u0 n) := by
          simp [periodicHeat1D_evolvedValueFourierCoeff,
            periodicHeat1D_modeMultiplier_neg,
            periodicH1_valueFourierCoeff_neg_eq_star]
    _ = star (periodicHeat1D_evolvedValueFourierCoeff nu u0 t n) := by
          simp [periodicHeat1D_evolvedValueFourierCoeff]

theorem periodicHeat1D_evolvedDerivativeFourierCoeff_neg_eq_star
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (n : ℤ) :
    periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t (-n) =
      star (periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n) := by
  calc
    periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t (-n)
        = (periodicHeat1D_modeMultiplier nu t n : ℂ) *
            star (periodicH1_derivativeFourierCoeff u0 n) := by
          simp [periodicHeat1D_evolvedDerivativeFourierCoeff,
            periodicHeat1D_modeMultiplier_neg,
            periodicH1_derivativeFourierCoeff_neg_eq_star]
    _ = star (periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n) := by
          simp [periodicHeat1D_evolvedDerivativeFourierCoeff]

theorem periodicHeat1D_evolvedValueFourierCoeff_add_time
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t s : ℝ)
    (n : ℤ) :
    periodicHeat1D_evolvedValueFourierCoeff nu u0 (t + s) n =
      (periodicHeat1D_modeMultiplier nu t n : ℂ) *
        periodicHeat1D_evolvedValueFourierCoeff nu u0 s n := by
  simp [periodicHeat1D_evolvedValueFourierCoeff,
    periodicHeat1D_modeMultiplier_add_time, mul_assoc]

theorem periodicHeat1D_evolvedDerivativeFourierCoeff_add_time
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t s : ℝ)
    (n : ℤ) :
    periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 (t + s) n =
      (periodicHeat1D_modeMultiplier nu t n : ℂ) *
        periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 s n := by
  simp [periodicHeat1D_evolvedDerivativeFourierCoeff,
    periodicHeat1D_modeMultiplier_add_time, mul_assoc]

theorem periodicHeat1D_evolvedDerivativeFourierCoeff_compatible
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (n : PeriodicH1NonzeroMode) :
    periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n.1 =
      (2 * Real.pi * Complex.I * (n.1 : ℂ)) *
        periodicHeat1D_evolvedValueFourierCoeff nu u0 t n.1 := by
  dsimp [periodicHeat1D_evolvedDerivativeFourierCoeff,
    periodicHeat1D_evolvedValueFourierCoeff]
  rw [periodicH1_weakDerivative_fourierCoeff_literature u0 hu0 n]
  ring

theorem periodicHeat1D_evolvedDerivativeFourierCoeff_compatible_allModes
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (n : ℤ) :
    periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n =
      (2 * Real.pi * Complex.I * (n : ℂ)) *
        periodicHeat1D_evolvedValueFourierCoeff nu u0 t n := by
  dsimp [periodicHeat1D_evolvedDerivativeFourierCoeff,
    periodicHeat1D_evolvedValueFourierCoeff]
  rw [periodicH1_weakDerivative_fourierCoeff_allModes_literature u0 hu0 n]
  ring

theorem periodicHeat1D_evolvedValueModeEnergy_nonneg
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (n : ℤ) :
    0 ≤ periodicHeat1D_evolvedValueModeEnergy nu u0 t n := by
  exact sq_nonneg _

theorem periodicHeat1D_evolvedDerivativeModeEnergy_nonneg
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (n : ℤ) :
    0 ≤ periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n := by
  exact sq_nonneg _

theorem periodicHeat1D_evolvedValueModeEnergy_le_initial
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (n : ℤ)
    (ht : 0 ≤ t) :
    periodicHeat1D_evolvedValueModeEnergy nu u0 t n ≤
      periodicH1_valueModeEnergy u0 n := by
  dsimp [periodicHeat1D_evolvedValueModeEnergy,
    periodicHeat1D_evolvedValueFourierCoeff, periodicH1_valueModeEnergy]
  change
    ‖((periodicHeat1D_modeMultiplier nu t n : ℂ) *
        periodicH1_valueFourierCoeff u0 n)‖ ^ (2 : ℕ) ≤
      ‖periodicH1_valueFourierCoeff u0 n‖ ^ (2 : ℕ)
  rw [periodicHeat1D_realScalar_complex_norm_sq]
  have hm_sq := periodicHeat1D_modeMultiplier_sq_le_one nu t n ht
  nlinarith [sq_nonneg (Complex.abs (periodicH1_valueFourierCoeff u0 n))]

theorem periodicHeat1D_evolvedDerivativeModeEnergy_le_initial
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (n : ℤ)
    (ht : 0 ≤ t) :
    periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n ≤
      periodicH1_derivativeModeEnergy u0 n := by
  dsimp [periodicHeat1D_evolvedDerivativeModeEnergy,
    periodicHeat1D_evolvedDerivativeFourierCoeff,
    periodicH1_derivativeModeEnergy]
  change
    ‖((periodicHeat1D_modeMultiplier nu t n : ℂ) *
        periodicH1_derivativeFourierCoeff u0 n)‖ ^ (2 : ℕ) ≤
      ‖periodicH1_derivativeFourierCoeff u0 n‖ ^ (2 : ℕ)
  rw [periodicHeat1D_realScalar_complex_norm_sq]
  have hm_sq := periodicHeat1D_modeMultiplier_sq_le_one nu t n ht
  nlinarith [sq_nonneg (Complex.abs (periodicH1_derivativeFourierCoeff u0 n))]

theorem periodicHeat1D_evolvedValueModeEnergy_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) :
    Summable (periodicHeat1D_evolvedValueModeEnergy nu u0 t) :=
  Summable.of_nonneg_of_le
    (periodicHeat1D_evolvedValueModeEnergy_nonneg nu u0 t)
    (fun n => periodicHeat1D_evolvedValueModeEnergy_le_initial nu u0 t n ht)
    (periodicH1_valueModeEnergy_summable_literature u0 hu0)

theorem periodicHeat1D_evolvedDerivativeModeEnergy_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) :
    Summable (periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t) :=
  Summable.of_nonneg_of_le
    (periodicHeat1D_evolvedDerivativeModeEnergy_nonneg nu u0 t)
    (fun n => periodicHeat1D_evolvedDerivativeModeEnergy_le_initial nu u0 t n ht)
    (periodicH1_derivativeModeEnergy_summable_literature u0 hu0)

theorem periodicHeat1D_evolvedValueModeEnergy_tsum_le_initial
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) :
    (∑' n : ℤ, periodicHeat1D_evolvedValueModeEnergy nu u0 t n) ≤
      PeriodicH1State.energy u0 := by
  have hle :
      (∑' n : ℤ, periodicHeat1D_evolvedValueModeEnergy nu u0 t n) ≤
        ∑' n : ℤ, periodicH1_valueModeEnergy u0 n :=
    tsum_le_tsum
      (fun n => periodicHeat1D_evolvedValueModeEnergy_le_initial nu u0 t n ht)
      (periodicHeat1D_evolvedValueModeEnergy_summable nu u0 hu0 t ht)
      (periodicH1_valueModeEnergy_summable_literature u0 hu0)
  calc
    (∑' n : ℤ, periodicHeat1D_evolvedValueModeEnergy nu u0 t n)
        ≤ ∑' n : ℤ, periodicH1_valueModeEnergy u0 n := hle
    _ = PeriodicH1State.energy u0 :=
        periodicH1_value_parseval_literature u0 hu0

theorem periodicHeat1D_evolvedDerivativeModeEnergy_tsum_le_initial
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) :
    (∑' n : ℤ, periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n) ≤
      PeriodicH1State.derivativeEnergy u0 := by
  have hle :
      (∑' n : ℤ, periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n) ≤
        ∑' n : ℤ, periodicH1_derivativeModeEnergy u0 n :=
    tsum_le_tsum
      (fun n => periodicHeat1D_evolvedDerivativeModeEnergy_le_initial nu u0 t n ht)
      (periodicHeat1D_evolvedDerivativeModeEnergy_summable nu u0 hu0 t ht)
      (periodicH1_derivativeModeEnergy_summable_literature u0 hu0)
  calc
    (∑' n : ℤ, periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n)
        ≤ ∑' n : ℤ, periodicH1_derivativeModeEnergy u0 n := hle
    _ = PeriodicH1State.derivativeEnergy u0 :=
        periodicH1_derivative_parseval_literature u0 hu0

/-- Coefficient-level periodic heat theory. This is the part of the heat
backend that is now fully proved in Lean: the formal Fourier coefficients have
the correct initial value, semigroup law, derivative compatibility, summability,
and ℓ² contractions. Later sections prove `L²` and Fourier/Sobolev `H¹`
reconstruction before the remaining certified-window/PDE upgrade boundary. -/
structure PeriodicHeat1DFourierCoefficientTheory where
  value_zero_time :
    ∀ (nu : BurgersParameters) (u0 : PeriodicH1State) (n : ℤ),
      periodicHeat1D_evolvedValueFourierCoeff nu u0 0 n =
        periodicH1_valueFourierCoeff u0 n
  derivative_zero_time :
    ∀ (nu : BurgersParameters) (u0 : PeriodicH1State) (n : ℤ),
      periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 0 n =
        periodicH1_derivativeFourierCoeff u0 n
  value_add_time :
    ∀ (nu : BurgersParameters) (u0 : PeriodicH1State) (t s : ℝ) (n : ℤ),
      periodicHeat1D_evolvedValueFourierCoeff nu u0 (t + s) n =
        (periodicHeat1D_modeMultiplier nu t n : ℂ) *
          periodicHeat1D_evolvedValueFourierCoeff nu u0 s n
  derivative_add_time :
    ∀ (nu : BurgersParameters) (u0 : PeriodicH1State) (t s : ℝ) (n : ℤ),
      periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 (t + s) n =
        (periodicHeat1D_modeMultiplier nu t n : ℂ) *
          periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 s n
  derivative_compatible :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ)
      (n : PeriodicH1NonzeroMode),
      periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n.1 =
        (2 * Real.pi * Complex.I * (n.1 : ℂ)) *
          periodicHeat1D_evolvedValueFourierCoeff nu u0 t n.1
  value_summable :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ),
      0 ≤ t → Summable (periodicHeat1D_evolvedValueModeEnergy nu u0 t)
  derivative_summable :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ),
      0 ≤ t → Summable (periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t)
  value_contraction :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ),
      0 ≤ t →
        (∑' n : ℤ, periodicHeat1D_evolvedValueModeEnergy nu u0 t n) ≤
          PeriodicH1State.energy u0
  derivative_contraction :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ),
      0 ≤ t →
        (∑' n : ℤ, periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n) ≤
          PeriodicH1State.derivativeEnergy u0

def periodicHeat1D_fourierCoefficientTheory_literature :
    PeriodicHeat1DFourierCoefficientTheory where
  value_zero_time := periodicHeat1D_evolvedValueFourierCoeff_zero_time
  derivative_zero_time := periodicHeat1D_evolvedDerivativeFourierCoeff_zero_time
  value_add_time := periodicHeat1D_evolvedValueFourierCoeff_add_time
  derivative_add_time := periodicHeat1D_evolvedDerivativeFourierCoeff_add_time
  derivative_compatible := periodicHeat1D_evolvedDerivativeFourierCoeff_compatible
  value_summable := periodicHeat1D_evolvedValueModeEnergy_summable
  derivative_summable := periodicHeat1D_evolvedDerivativeModeEnergy_summable
  value_contraction := periodicHeat1D_evolvedValueModeEnergy_tsum_le_initial
  derivative_contraction := periodicHeat1D_evolvedDerivativeModeEnergy_tsum_le_initial

/-- The `ℓ²` coefficient carrier used by mathlib's Fourier Hilbert basis on
the unit torus. This is the first reconstruction target below the stronger
continuous `H¹` carrier used by the Burgers backend. -/
abbrev PeriodicHeat1DFourierCoeffL2 := lp (fun _ : ℤ => ℂ) (2 : ℝ≥0∞)

/-- The `L²` heat state reconstructed directly from Fourier coefficients. -/
abbrev PeriodicHeat1DL2State := Lp ℂ 2 (AddCircle.haarAddCircle (T := (1 : ℝ)))

/-- Coefficient recovery for an `L²` Fourier series over mathlib's periodic
Hilbert basis.

This is the reusable Hilbert-space fact behind the later continuous-series
coefficient recovery: if an `L²` state is the Hilbert-basis sum with
coefficients `a`, then its `m`th Fourier coefficient is exactly `a m`. -/
theorem periodicHeat1D_l2FourierSeries_fourierCoeff
    (a : ℤ → ℂ)
    (f : PeriodicHeat1DL2State)
    (h : HasSum
      (fun n : ℤ => a n • (fourierLp (T := (1 : ℝ)) 2 n)) f)
    (m : ℤ) :
    fourierCoeff f m = a m := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  have hinner : HasSum
      (fun n : ℤ =>
        inner (𝕜 := ℂ) (fourierLp (T := (1 : ℝ)) 2 m)
          (a n • (fourierLp (T := (1 : ℝ)) 2 n)))
      (inner (𝕜 := ℂ) (fourierLp (T := (1 : ℝ)) 2 m) f) := by
    exact (innerSL ℂ (fourierLp (T := (1 : ℝ)) 2 m)).hasSum h
  have hterm :
      (fun n : ℤ =>
        inner (𝕜 := ℂ) (fourierLp (T := (1 : ℝ)) 2 m)
          (a n • (fourierLp (T := (1 : ℝ)) 2 n))) =
        fun n : ℤ => if n = m then a m else 0 := by
    funext n
    calc
      inner (𝕜 := ℂ) (fourierLp (T := (1 : ℝ)) 2 m)
          (a n • (fourierLp (T := (1 : ℝ)) 2 n))
          = a n * inner (𝕜 := ℂ) (fourierLp (T := (1 : ℝ)) 2 m)
              (fourierLp (T := (1 : ℝ)) 2 n) := by
            rw [inner_smul_right]
      _ = a n * (if m = n then 1 else 0) := by
            have horth := orthonormal_fourier (T := (1 : ℝ))
            rw [orthonormal_iff_ite.mp horth m n]
      _ = (if n = m then a m else 0) := by
            by_cases hnm : n = m
            · subst hnm
              simp
            · have hmn : m ≠ n := fun h => hnm h.symm
              simp [hnm, hmn]
  have hsingle : HasSum (fun n : ℤ => if n = m then a m else 0) (a m) := by
    simpa [eq_comm] using hasSum_ite_eq m (a m)
  have hcoord : inner (𝕜 := ℂ) (fourierLp (T := (1 : ℝ)) 2 m) f = a m := by
    exact (hinner.congr_fun (by intro n; exact (congrFun hterm n).symm)).unique
      hsingle
  have hB : (fourierBasis (T := (1 : ℝ)) m : PeriodicHeat1DL2State) =
      fourierLp (T := (1 : ℝ)) 2 m := by
    rw [← coe_fourierBasis (T := (1 : ℝ))]
  calc
    fourierCoeff f m = fourierBasis.repr f m := by
      exact (fourierBasis_repr f m).symm
    _ = inner (𝕜 := ℂ) (fourierBasis (T := (1 : ℝ)) m) f := by
      rw [HilbertBasis.repr_apply_apply]
    _ = inner (𝕜 := ℂ) (fourierLp (T := (1 : ℝ)) 2 m) f := by
      rw [hB]
    _ = a m := hcoord

/-- Fourier/Sobolev `H¹` carrier for the periodic heat backend.

This carrier is weaker than the concrete `PeriodicH1State`: it records the
value and weak derivative as `L²` objects and requires the derivative relation
only at Fourier-coefficient level. That is exactly the regularity provided by
the mathlib Hilbert-basis reconstruction before choosing continuous
representatives or proving PDE semantics on a certified window. -/
structure PeriodicFourierH1State where
  valueL2 : PeriodicHeat1DL2State
  derivativeL2 : PeriodicHeat1DL2State
  derivative_coeff :
    ∀ n : PeriodicH1NonzeroMode,
      fourierCoeff derivativeL2 n.1 =
        (2 * Real.pi * Complex.I * (n.1 : ℂ)) *
          fourierCoeff valueL2 n.1

namespace PeriodicFourierH1State

/-- Value-side `L²` energy of a Fourier/Sobolev `H¹` state. -/
def energy (u : PeriodicFourierH1State) : ℝ :=
  ∫ x : BurgersTorus, ‖(u.valueL2 x : ℂ)‖ ^ (2 : ℕ)
    ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))

/-- Derivative-side `L²` energy of a Fourier/Sobolev `H¹` state. -/
def derivativeEnergy (u : PeriodicFourierH1State) : ℝ :=
  ∫ x : BurgersTorus, ‖(u.derivativeL2 x : ℂ)‖ ^ (2 : ℕ)
    ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))

end PeriodicFourierH1State

/-- The value-side evolved Fourier coefficients form an `ℓ²` vector. -/
def periodicHeat1D_evolvedValueFourierCoeff_l2
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) : PeriodicHeat1DFourierCoeffL2 :=
  ⟨fun n : ℤ => periodicHeat1D_evolvedValueFourierCoeff nu u0 t n, by
    apply memℓp_gen
    simpa [periodicHeat1D_evolvedValueModeEnergy] using
      periodicHeat1D_evolvedValueModeEnergy_summable nu u0 hu0 t ht⟩

/-- The derivative-side evolved Fourier coefficients form an `ℓ²` vector. -/
def periodicHeat1D_evolvedDerivativeFourierCoeff_l2
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) : PeriodicHeat1DFourierCoeffL2 :=
  ⟨fun n : ℤ => periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n, by
    apply memℓp_gen
    simpa [periodicHeat1D_evolvedDerivativeModeEnergy] using
      periodicHeat1D_evolvedDerivativeModeEnergy_summable nu u0 hu0 t ht⟩

/-- `L²` heat reconstruction of the value component by the Fourier Hilbert
basis. This is fully mathlib-backed, but it is intentionally weaker than the
backend's continuous `H¹` curve. -/
def periodicHeat1D_reconstructedValueL2
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) : PeriodicHeat1DL2State := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  exact fourierBasis.repr.symm
    (periodicHeat1D_evolvedValueFourierCoeff_l2 nu u0 hu0 t ht)

/-- `L²` heat reconstruction of the weak-derivative component by the Fourier
Hilbert basis. -/
def periodicHeat1D_reconstructedDerivativeL2
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) : PeriodicHeat1DL2State := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  exact fourierBasis.repr.symm
    (periodicHeat1D_evolvedDerivativeFourierCoeff_l2 nu u0 hu0 t ht)

theorem periodicHeat1D_reconstructedValueL2_fourierCoeff
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t)
    (n : ℤ) :
    fourierCoeff (periodicHeat1D_reconstructedValueL2 nu u0 hu0 t ht) n =
      periodicHeat1D_evolvedValueFourierCoeff nu u0 t n := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  have hrepr := fourierBasis_repr
    (periodicHeat1D_reconstructedValueL2 nu u0 hu0 t ht) n
  rw [← hrepr]
  simp [periodicHeat1D_reconstructedValueL2,
    periodicHeat1D_evolvedValueFourierCoeff_l2]

theorem periodicHeat1D_reconstructedDerivativeL2_fourierCoeff
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t)
    (n : ℤ) :
    fourierCoeff (periodicHeat1D_reconstructedDerivativeL2 nu u0 hu0 t ht) n =
      periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  have hrepr := fourierBasis_repr
    (periodicHeat1D_reconstructedDerivativeL2 nu u0 hu0 t ht) n
  rw [← hrepr]
  simp [periodicHeat1D_reconstructedDerivativeL2,
    periodicHeat1D_evolvedDerivativeFourierCoeff_l2]

theorem periodicHeat1D_reconstructedValueL2_hasSum
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) :
    HasSum
      (fun n : ℤ =>
        periodicHeat1D_evolvedValueFourierCoeff nu u0 t n • fourierLp 2 n)
      (periodicHeat1D_reconstructedValueL2 nu u0 hu0 t ht) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  simpa [periodicHeat1D_reconstructedValueL2_fourierCoeff] using
    hasSum_fourier_series_L2
      (periodicHeat1D_reconstructedValueL2 nu u0 hu0 t ht)

theorem periodicHeat1D_reconstructedDerivativeL2_hasSum
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) :
    HasSum
      (fun n : ℤ =>
        periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n • fourierLp 2 n)
      (periodicHeat1D_reconstructedDerivativeL2 nu u0 hu0 t ht) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  simpa [periodicHeat1D_reconstructedDerivativeL2_fourierCoeff] using
    hasSum_fourier_series_L2
      (periodicHeat1D_reconstructedDerivativeL2 nu u0 hu0 t ht)

theorem periodicHeat1D_reconstructedValueL2_parseval
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) :
    (∑' n : ℤ, periodicHeat1D_evolvedValueModeEnergy nu u0 t n) =
      ∫ x : BurgersTorus,
        ‖((periodicHeat1D_reconstructedValueL2 nu u0 hu0 t ht :
            PeriodicHeat1DL2State) x : ℂ)‖ ^ (2 : ℕ)
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  have h := tsum_sq_fourierCoeff
    (periodicHeat1D_reconstructedValueL2 nu u0 hu0 t ht)
  rw [← h]
  apply tsum_congr
  intro n
  simp [periodicHeat1D_evolvedValueModeEnergy,
    periodicHeat1D_reconstructedValueL2_fourierCoeff]

theorem periodicHeat1D_reconstructedDerivativeL2_parseval
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) :
    (∑' n : ℤ, periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n) =
      ∫ x : BurgersTorus,
        ‖((periodicHeat1D_reconstructedDerivativeL2 nu u0 hu0 t ht :
            PeriodicHeat1DL2State) x : ℂ)‖ ^ (2 : ℕ)
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  have h := tsum_sq_fourierCoeff
    (periodicHeat1D_reconstructedDerivativeL2 nu u0 hu0 t ht)
  rw [← h]
  apply tsum_congr
  intro n
  simp [periodicHeat1D_evolvedDerivativeModeEnergy,
    periodicHeat1D_reconstructedDerivativeL2_fourierCoeff]

/-- The proved `L²` value and derivative reconstructions form a Fourier/Sobolev
`H¹` state. This is the non-continuous, coefficient-level heat reconstruction
available directly from mathlib Fourier analysis. -/
def periodicHeat1D_reconstructedFourierH1
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) : PeriodicFourierH1State where
  valueL2 := periodicHeat1D_reconstructedValueL2 nu u0 hu0 t ht
  derivativeL2 := periodicHeat1D_reconstructedDerivativeL2 nu u0 hu0 t ht
  derivative_coeff := by
    intro n
    rw [periodicHeat1D_reconstructedDerivativeL2_fourierCoeff,
      periodicHeat1D_reconstructedValueL2_fourierCoeff]
    exact periodicHeat1D_evolvedDerivativeFourierCoeff_compatible
      nu u0 hu0 t n

theorem periodicHeat1D_reconstructedFourierH1_derivative_coeff
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t)
    (n : PeriodicH1NonzeroMode) :
    fourierCoeff
      (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht).derivativeL2 n.1 =
      (2 * Real.pi * Complex.I * (n.1 : ℂ)) *
        fourierCoeff
          (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht).valueL2 n.1 :=
  (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht).derivative_coeff n

theorem periodicHeat1D_reconstructedFourierH1_value_parseval
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) :
    PeriodicFourierH1State.energy
      (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht) =
        ∑' n : ℤ, periodicHeat1D_evolvedValueModeEnergy nu u0 t n := by
  simpa [PeriodicFourierH1State.energy,
    periodicHeat1D_reconstructedFourierH1] using
    (periodicHeat1D_reconstructedValueL2_parseval nu u0 hu0 t ht).symm

theorem periodicHeat1D_reconstructedFourierH1_derivative_parseval
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) :
    PeriodicFourierH1State.derivativeEnergy
      (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht) =
        ∑' n : ℤ, periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n := by
  simpa [PeriodicFourierH1State.derivativeEnergy,
    periodicHeat1D_reconstructedFourierH1] using
    (periodicHeat1D_reconstructedDerivativeL2_parseval nu u0 hu0 t ht).symm

theorem periodicHeat1D_reconstructedFourierH1_energy_le_initial
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) :
    PeriodicFourierH1State.energy
      (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht) ≤
        PeriodicH1State.energy u0 := by
  calc
    PeriodicFourierH1State.energy
        (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht)
        = ∑' n : ℤ, periodicHeat1D_evolvedValueModeEnergy nu u0 t n :=
          periodicHeat1D_reconstructedFourierH1_value_parseval nu u0 hu0 t ht
    _ ≤ PeriodicH1State.energy u0 :=
          periodicHeat1D_evolvedValueModeEnergy_tsum_le_initial
            nu u0 hu0 t ht

theorem periodicHeat1D_reconstructedFourierH1_derivativeEnergy_le_initial
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 ≤ t) :
    PeriodicFourierH1State.derivativeEnergy
      (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht) ≤
        PeriodicH1State.derivativeEnergy u0 := by
  calc
    PeriodicFourierH1State.derivativeEnergy
        (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht)
        = ∑' n : ℤ,
            periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n :=
          periodicHeat1D_reconstructedFourierH1_derivative_parseval
            nu u0 hu0 t ht
    _ ≤ PeriodicH1State.derivativeEnergy u0 :=
          periodicHeat1D_evolvedDerivativeModeEnergy_tsum_le_initial
            nu u0 hu0 t ht

/-- Fully proved `L²` reconstruction layer for the periodic heat coefficients.
The stronger remaining boundary is not Fourier reconstruction in `L²`; it is
the upgrade from this Hilbert-space object to the continuous `H¹` window with
weak heat residual, uniqueness, and smoothing used by the backend. -/
structure PeriodicHeat1DL2ReconstructionTheory where
  value_coeff :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ)
      (ht : 0 ≤ t),
      ∀ n : ℤ,
        fourierCoeff
          (periodicHeat1D_reconstructedValueL2 nu u0 _hu0 t ht) n =
          periodicHeat1D_evolvedValueFourierCoeff nu u0 t n
  derivative_coeff :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ)
      (ht : 0 ≤ t),
      ∀ n : ℤ,
        fourierCoeff
          (periodicHeat1D_reconstructedDerivativeL2 nu u0 _hu0 t ht) n =
          periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n
  value_hasSum :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ)
      (ht : 0 ≤ t),
      HasSum
        (fun n : ℤ =>
          periodicHeat1D_evolvedValueFourierCoeff nu u0 t n • fourierLp 2 n)
        (periodicHeat1D_reconstructedValueL2 nu u0 _hu0 t ht)
  derivative_hasSum :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ)
      (ht : 0 ≤ t),
      HasSum
        (fun n : ℤ =>
          periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n • fourierLp 2 n)
        (periodicHeat1D_reconstructedDerivativeL2 nu u0 _hu0 t ht)
  value_parseval :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ)
      (ht : 0 ≤ t),
      (∑' n : ℤ, periodicHeat1D_evolvedValueModeEnergy nu u0 t n) =
        ∫ x : BurgersTorus,
          ‖((periodicHeat1D_reconstructedValueL2 nu u0 _hu0 t ht :
              PeriodicHeat1DL2State) x : ℂ)‖ ^ (2 : ℕ)
          ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))
  derivative_parseval :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ)
      (ht : 0 ≤ t),
      (∑' n : ℤ, periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n) =
        ∫ x : BurgersTorus,
          ‖((periodicHeat1D_reconstructedDerivativeL2 nu u0 _hu0 t ht :
              PeriodicHeat1DL2State) x : ℂ)‖ ^ (2 : ℕ)
          ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))

def periodicHeat1D_l2ReconstructionTheory_literature :
    PeriodicHeat1DL2ReconstructionTheory where
  value_coeff := periodicHeat1D_reconstructedValueL2_fourierCoeff
  derivative_coeff := periodicHeat1D_reconstructedDerivativeL2_fourierCoeff
  value_hasSum := periodicHeat1D_reconstructedValueL2_hasSum
  derivative_hasSum := periodicHeat1D_reconstructedDerivativeL2_hasSum
  value_parseval := periodicHeat1D_reconstructedValueL2_parseval
  derivative_parseval := periodicHeat1D_reconstructedDerivativeL2_parseval

/-- Fully proved Fourier/Sobolev `H¹` reconstruction layer for the heat
coefficients. This packages the derivative relation, Parseval identities, and
finite-time contractions on the reconstructed Fourier-`H¹` carrier. -/
structure PeriodicHeat1DFourierH1ReconstructionTheory where
  derivative_coeff :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ)
      (ht : 0 ≤ t)
      (n : PeriodicH1NonzeroMode),
      fourierCoeff
        (periodicHeat1D_reconstructedFourierH1 nu u0 _hu0 t ht).derivativeL2 n.1 =
        (2 * Real.pi * Complex.I * (n.1 : ℂ)) *
          fourierCoeff
            (periodicHeat1D_reconstructedFourierH1 nu u0 _hu0 t ht).valueL2 n.1
  value_parseval :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ)
      (ht : 0 ≤ t),
      PeriodicFourierH1State.energy
        (periodicHeat1D_reconstructedFourierH1 nu u0 _hu0 t ht) =
          ∑' n : ℤ, periodicHeat1D_evolvedValueModeEnergy nu u0 t n
  derivative_parseval :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ)
      (ht : 0 ≤ t),
      PeriodicFourierH1State.derivativeEnergy
        (periodicHeat1D_reconstructedFourierH1 nu u0 _hu0 t ht) =
          ∑' n : ℤ, periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n
  value_contraction :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ)
      (ht : 0 ≤ t),
      PeriodicFourierH1State.energy
        (periodicHeat1D_reconstructedFourierH1 nu u0 _hu0 t ht) ≤
          PeriodicH1State.energy u0
  derivative_contraction :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ)
      (ht : 0 ≤ t),
      PeriodicFourierH1State.derivativeEnergy
        (periodicHeat1D_reconstructedFourierH1 nu u0 _hu0 t ht) ≤
          PeriodicH1State.derivativeEnergy u0

def periodicHeat1D_fourierH1ReconstructionTheory_literature :
    PeriodicHeat1DFourierH1ReconstructionTheory where
  derivative_coeff := periodicHeat1D_reconstructedFourierH1_derivative_coeff
  value_parseval := periodicHeat1D_reconstructedFourierH1_value_parseval
  derivative_parseval := periodicHeat1D_reconstructedFourierH1_derivative_parseval
  value_contraction := periodicHeat1D_reconstructedFourierH1_energy_le_initial
  derivative_contraction := periodicHeat1D_reconstructedFourierH1_derivativeEnergy_le_initial

/-- General integer Gaussian summability used by the positive-time heat
smoothing layer. This is a direct mathlib theorem via the Jacobi theta
summability bound, specialized to the one-dimensional integer lattice. -/
theorem periodicHeat1D_intPolynomialGaussian_summable
    (k : ℕ)
    {r : ℝ}
    (hr : 0 < r) :
    Summable
      (fun n : ℤ =>
        ((n.natAbs : ℝ) ^ k) * Real.exp (-r * ((n : ℝ) ^ (2 : ℕ)))) := by
  have hT : 0 < r / Real.pi := div_pos hr Real.pi_pos
  have h := summable_pow_mul_jacobiTheta₂_term_bound 0 hT k
  convert h using 1
  · ext n
    simp only [Nat.cast_natAbs]
    congr 1
    ring_nf
    field_simp [Real.pi_ne_zero]

theorem periodicHeat1D_modeMultiplier_polynomial_summable
    (nu : BurgersParameters)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ) :
    Summable
      (fun n : ℤ =>
        ((n.natAbs : ℝ) ^ k) * periodicHeat1D_modeMultiplier nu t n) := by
  let r : ℝ := nu.viscosity * t * (2 * Real.pi) ^ (2 : ℕ)
  have hr : 0 < r := by
    dsimp [r]
    exact mul_pos (mul_pos nu.viscosity_pos ht)
      (sq_pos_of_ne_zero (mul_ne_zero (by norm_num) Real.pi_ne_zero))
  have hgauss := periodicHeat1D_intPolynomialGaussian_summable k hr
  convert hgauss using 1
  · ext n
    congr 1
    simp [periodicHeat1D_modeMultiplier, periodicHeat1D_modeExponent,
      periodicHeat1D_modeFrequency, r]
    ring

def periodicHeat1D_weightedHeatMultiplierEnergy
    (nu : BurgersParameters)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) : ℝ :=
  ((n.natAbs : ℝ) ^ k) *
    (periodicHeat1D_modeMultiplier nu t n) ^ (2 : ℕ)

theorem periodicHeat1D_weightedHeatMultiplierEnergy_nonneg
    (nu : BurgersParameters)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) :
    0 ≤ periodicHeat1D_weightedHeatMultiplierEnergy nu t k n := by
  exact mul_nonneg (pow_nonneg (Nat.cast_nonneg _) _)
    (sq_nonneg _)

theorem periodicHeat1D_weightedHeatMultiplierEnergy_summable
    (nu : BurgersParameters)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ) :
    Summable (periodicHeat1D_weightedHeatMultiplierEnergy nu t k) := by
  have h2t : 0 < t + t := add_pos ht ht
  have h :=
    periodicHeat1D_modeMultiplier_polynomial_summable nu (t + t) h2t k
  convert h using 1
  · ext n
    simp [periodicHeat1D_weightedHeatMultiplierEnergy,
      periodicHeat1D_modeMultiplier_add_time, pow_two]

theorem periodicHeat1D_summable_mul_of_summable_nonneg
    {ι : Type*}
    {f g : ι → ℝ}
    (hf : Summable f)
    (hf_nonneg : ∀ i, 0 ≤ f i)
    (hg : Summable g)
    (hg_nonneg : ∀ i, 0 ≤ g i) :
    Summable (fun i => f i * g i) := by
  refine Summable.of_nonneg_of_le ?_ ?_ (hf.mul_right (∑' i, g i))
  · intro i
    exact mul_nonneg (hf_nonneg i) (hg_nonneg i)
  · intro i
    have hg_bound : g i ≤ ∑' i, g i :=
      le_tsum hg i (by intro j _hj; exact hg_nonneg j)
    exact mul_le_mul_of_nonneg_left hg_bound (hf_nonneg i)

def periodicHeat1D_weightedEvolvedValueModeEnergy
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) : ℝ :=
  ((n.natAbs : ℝ) ^ k) *
    periodicHeat1D_evolvedValueModeEnergy nu u0 t n

def periodicHeat1D_weightedEvolvedDerivativeModeEnergy
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) : ℝ :=
  ((n.natAbs : ℝ) ^ k) *
    periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n

theorem periodicHeat1D_weightedEvolvedValueModeEnergy_nonneg
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) :
    0 ≤ periodicHeat1D_weightedEvolvedValueModeEnergy nu u0 t k n := by
  exact mul_nonneg (pow_nonneg (Nat.cast_nonneg _) _)
    (periodicHeat1D_evolvedValueModeEnergy_nonneg nu u0 t n)

theorem periodicHeat1D_weightedEvolvedDerivativeModeEnergy_nonneg
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) :
    0 ≤ periodicHeat1D_weightedEvolvedDerivativeModeEnergy nu u0 t k n := by
  exact mul_nonneg (pow_nonneg (Nat.cast_nonneg _) _)
    (periodicHeat1D_evolvedDerivativeModeEnergy_nonneg nu u0 t n)

theorem periodicH1_valueModeEnergy_nonneg
    (u : PeriodicH1State)
    (n : ℤ) :
    0 ≤ periodicH1_valueModeEnergy u n := by
  exact sq_nonneg _

theorem periodicH1_derivativeModeEnergy_nonneg
    (u : PeriodicH1State)
    (n : ℤ) :
    0 ≤ periodicH1_derivativeModeEnergy u n := by
  exact sq_nonneg _

theorem periodicHeat1D_weightedEvolvedValueModeEnergy_eq_multiplier_mul_initial
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) :
    periodicHeat1D_weightedEvolvedValueModeEnergy nu u0 t k n =
      periodicHeat1D_weightedHeatMultiplierEnergy nu t k n *
        periodicH1_valueModeEnergy u0 n := by
  simp [periodicHeat1D_weightedEvolvedValueModeEnergy,
    periodicHeat1D_weightedHeatMultiplierEnergy,
    periodicHeat1D_evolvedValueModeEnergy,
    periodicHeat1D_evolvedValueFourierCoeff,
    periodicH1_valueModeEnergy,
    periodicHeat1D_realScalar_complex_norm_sq]
  rw [abs_of_nonneg (periodicHeat1D_modeMultiplier_nonneg nu t n)]
  ring

theorem periodicHeat1D_weightedEvolvedDerivativeModeEnergy_eq_multiplier_mul_initial
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) :
    periodicHeat1D_weightedEvolvedDerivativeModeEnergy nu u0 t k n =
      periodicHeat1D_weightedHeatMultiplierEnergy nu t k n *
        periodicH1_derivativeModeEnergy u0 n := by
  simp [periodicHeat1D_weightedEvolvedDerivativeModeEnergy,
    periodicHeat1D_weightedHeatMultiplierEnergy,
    periodicHeat1D_evolvedDerivativeModeEnergy,
    periodicHeat1D_evolvedDerivativeFourierCoeff,
    periodicH1_derivativeModeEnergy,
    periodicHeat1D_realScalar_complex_norm_sq]
  rw [abs_of_nonneg (periodicHeat1D_modeMultiplier_nonneg nu t n)]
  ring

theorem periodicHeat1D_weightedEvolvedValueModeEnergy_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ) :
    Summable (periodicHeat1D_weightedEvolvedValueModeEnergy nu u0 t k) := by
  have hprod :
      Summable
        (fun n : ℤ =>
          periodicHeat1D_weightedHeatMultiplierEnergy nu t k n *
            periodicH1_valueModeEnergy u0 n) :=
    periodicHeat1D_summable_mul_of_summable_nonneg
      (periodicHeat1D_weightedHeatMultiplierEnergy_summable nu t ht k)
      (periodicHeat1D_weightedHeatMultiplierEnergy_nonneg nu t k)
      (periodicH1_valueModeEnergy_summable_literature u0 hu0)
      (periodicH1_valueModeEnergy_nonneg u0)
  convert hprod using 1
  · ext n
    exact periodicHeat1D_weightedEvolvedValueModeEnergy_eq_multiplier_mul_initial
      nu u0 t k n

theorem periodicHeat1D_weightedEvolvedDerivativeModeEnergy_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ) :
    Summable (periodicHeat1D_weightedEvolvedDerivativeModeEnergy nu u0 t k) := by
  have hprod :
      Summable
        (fun n : ℤ =>
          periodicHeat1D_weightedHeatMultiplierEnergy nu t k n *
            periodicH1_derivativeModeEnergy u0 n) :=
    periodicHeat1D_summable_mul_of_summable_nonneg
      (periodicHeat1D_weightedHeatMultiplierEnergy_summable nu t ht k)
      (periodicHeat1D_weightedHeatMultiplierEnergy_nonneg nu t k)
      (periodicH1_derivativeModeEnergy_summable_literature u0 hu0)
      (periodicH1_derivativeModeEnergy_nonneg u0)
  convert hprod using 1
  · ext n
    exact periodicHeat1D_weightedEvolvedDerivativeModeEnergy_eq_multiplier_mul_initial
      nu u0 t k n

def periodicHeat1D_weightedEvolvedValueFourierCoeffNorm
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) : ℝ :=
  ((n.natAbs : ℝ) ^ k) *
    ‖periodicHeat1D_evolvedValueFourierCoeff nu u0 t n‖

def periodicHeat1D_weightedEvolvedDerivativeFourierCoeffNorm
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) : ℝ :=
  ((n.natAbs : ℝ) ^ k) *
    ‖periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n‖

theorem periodicHeat1D_weightedEvolvedValueFourierCoeffNorm_nonneg
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) :
    0 ≤ periodicHeat1D_weightedEvolvedValueFourierCoeffNorm nu u0 t k n := by
  exact mul_nonneg (pow_nonneg (Nat.cast_nonneg _) _) (norm_nonneg _)

theorem periodicHeat1D_weightedEvolvedDerivativeFourierCoeffNorm_nonneg
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) :
    0 ≤ periodicHeat1D_weightedEvolvedDerivativeFourierCoeffNorm nu u0 t k n := by
  exact mul_nonneg (pow_nonneg (Nat.cast_nonneg _) _) (norm_nonneg _)

theorem periodicHeat1D_weightedEvolvedValueFourierCoeffNorm_le_quadratic
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) :
    periodicHeat1D_weightedEvolvedValueFourierCoeffNorm nu u0 t k n ≤
      (periodicHeat1D_weightedHeatMultiplierEnergy nu t (2 * k) n +
        periodicH1_valueModeEnergy u0 n) / 2 := by
  unfold periodicHeat1D_weightedEvolvedValueFourierCoeffNorm
    periodicHeat1D_weightedHeatMultiplierEnergy periodicH1_valueModeEnergy
  dsimp [periodicHeat1D_evolvedValueFourierCoeff]
  rw [Complex.abs.map_mul, Complex.abs_ofReal,
    abs_of_nonneg (periodicHeat1D_modeMultiplier_nonneg nu t n)]
  rw [show (n.natAbs : ℝ) ^ (2 * k) =
      ((n.natAbs : ℝ) ^ k) ^ (2 : ℕ) by
    rw [← pow_mul]
    congr 1
    exact Nat.mul_comm 2 k]
  set w : ℝ := (n.natAbs : ℝ) ^ k
  set m : ℝ := periodicHeat1D_modeMultiplier nu t n
  set a : ℝ := ‖periodicH1_valueFourierCoeff u0 n‖
  change w * (m * a) ≤ (w ^ (2 : ℕ) * m ^ (2 : ℕ) + a ^ (2 : ℕ)) / 2
  have hineq : 2 * (w * m) * a ≤ (w * m) ^ (2 : ℕ) + a ^ (2 : ℕ) := by
    nlinarith [sq_nonneg (w * m - a)]
  nlinarith [hineq]

theorem periodicHeat1D_weightedEvolvedDerivativeFourierCoeffNorm_le_quadratic
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) :
    periodicHeat1D_weightedEvolvedDerivativeFourierCoeffNorm nu u0 t k n ≤
      (periodicHeat1D_weightedHeatMultiplierEnergy nu t (2 * k) n +
        periodicH1_derivativeModeEnergy u0 n) / 2 := by
  unfold periodicHeat1D_weightedEvolvedDerivativeFourierCoeffNorm
    periodicHeat1D_weightedHeatMultiplierEnergy periodicH1_derivativeModeEnergy
  dsimp [periodicHeat1D_evolvedDerivativeFourierCoeff]
  rw [Complex.abs.map_mul, Complex.abs_ofReal,
    abs_of_nonneg (periodicHeat1D_modeMultiplier_nonneg nu t n)]
  rw [show (n.natAbs : ℝ) ^ (2 * k) =
      ((n.natAbs : ℝ) ^ k) ^ (2 : ℕ) by
    rw [← pow_mul]
    congr 1
    exact Nat.mul_comm 2 k]
  set w : ℝ := (n.natAbs : ℝ) ^ k
  set m : ℝ := periodicHeat1D_modeMultiplier nu t n
  set a : ℝ := ‖periodicH1_derivativeFourierCoeff u0 n‖
  change w * (m * a) ≤ (w ^ (2 : ℕ) * m ^ (2 : ℕ) + a ^ (2 : ℕ)) / 2
  have hineq : 2 * (w * m) * a ≤ (w * m) ^ (2 : ℕ) + a ^ (2 : ℕ) := by
    nlinarith [sq_nonneg (w * m - a)]
  nlinarith [hineq]

theorem periodicHeat1D_weightedEvolvedValueFourierCoeffNorm_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ) :
    Summable (periodicHeat1D_weightedEvolvedValueFourierCoeffNorm nu u0 t k) := by
  have hmajor :
      Summable
        (fun n : ℤ =>
          (periodicHeat1D_weightedHeatMultiplierEnergy nu t (2 * k) n +
            periodicH1_valueModeEnergy u0 n) / 2) := by
    simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using
      ((periodicHeat1D_weightedHeatMultiplierEnergy_summable nu t ht (2 * k)).add
        (periodicH1_valueModeEnergy_summable_literature u0 hu0)).mul_right (2 : ℝ)⁻¹
  exact Summable.of_nonneg_of_le
    (periodicHeat1D_weightedEvolvedValueFourierCoeffNorm_nonneg nu u0 t k)
    (periodicHeat1D_weightedEvolvedValueFourierCoeffNorm_le_quadratic nu u0 t k)
    hmajor

theorem periodicHeat1D_weightedEvolvedDerivativeFourierCoeffNorm_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ) :
    Summable (periodicHeat1D_weightedEvolvedDerivativeFourierCoeffNorm nu u0 t k) := by
  have hmajor :
      Summable
        (fun n : ℤ =>
          (periodicHeat1D_weightedHeatMultiplierEnergy nu t (2 * k) n +
            periodicH1_derivativeModeEnergy u0 n) / 2) := by
    simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using
      ((periodicHeat1D_weightedHeatMultiplierEnergy_summable nu t ht (2 * k)).add
        (periodicH1_derivativeModeEnergy_summable_literature u0 hu0)).mul_right (2 : ℝ)⁻¹
  exact Summable.of_nonneg_of_le
    (periodicHeat1D_weightedEvolvedDerivativeFourierCoeffNorm_nonneg nu u0 t k)
    (periodicHeat1D_weightedEvolvedDerivativeFourierCoeffNorm_le_quadratic nu u0 t k)
    hmajor

theorem periodicHeat1D_evolvedValueFourierCoeff_norm_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) :
    Summable (fun n : ℤ => ‖periodicHeat1D_evolvedValueFourierCoeff nu u0 t n‖) := by
  convert periodicHeat1D_weightedEvolvedValueFourierCoeffNorm_summable
      nu u0 hu0 t ht 0 using 1
  ext n
  simp [periodicHeat1D_weightedEvolvedValueFourierCoeffNorm]

theorem periodicHeat1D_evolvedDerivativeFourierCoeff_norm_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) :
    Summable (fun n : ℤ => ‖periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n‖) := by
  convert periodicHeat1D_weightedEvolvedDerivativeFourierCoeffNorm_summable
      nu u0 hu0 t ht 0 using 1
  ext n
  simp [periodicHeat1D_weightedEvolvedDerivativeFourierCoeffNorm]

theorem periodicHeat1D_evolvedValueFourierCoeff_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) :
    Summable (periodicHeat1D_evolvedValueFourierCoeff nu u0 t) :=
  (periodicHeat1D_evolvedValueFourierCoeff_norm_summable
    nu u0 hu0 t ht).of_norm

theorem periodicHeat1D_evolvedDerivativeFourierCoeff_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) :
    Summable (periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t) :=
  (periodicHeat1D_evolvedDerivativeFourierCoeff_norm_summable
    nu u0 hu0 t ht).of_norm

theorem periodicHeat1D_valueFourierContinuousTerm_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) :
    Summable
      (fun n : ℤ =>
        periodicHeat1D_evolvedValueFourierCoeff nu u0 t n •
          (fourier n : C(BurgersTorus, ℂ))) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  refine Summable.of_norm_bounded
    (fun n : ℤ => ‖periodicHeat1D_evolvedValueFourierCoeff nu u0 t n‖)
    (periodicHeat1D_evolvedValueFourierCoeff_norm_summable nu u0 hu0 t ht)
    ?_
  intro n
  rw [norm_smul, fourier_norm]
  simp

theorem periodicHeat1D_derivativeFourierContinuousTerm_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) :
    Summable
      (fun n : ℤ =>
        periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n •
          (fourier n : C(BurgersTorus, ℂ))) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  refine Summable.of_norm_bounded
    (fun n : ℤ => ‖periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n‖)
    (periodicHeat1D_evolvedDerivativeFourierCoeff_norm_summable nu u0 hu0 t ht)
    ?_
  intro n
  rw [norm_smul, fourier_norm]
  simp

def periodicHeat1D_continuousValueFourierSeries
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ) : C(BurgersTorus, ℂ) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  exact ∑' n : ℤ,
    periodicHeat1D_evolvedValueFourierCoeff nu u0 t n •
      (fourier n : C(BurgersTorus, ℂ))

def periodicHeat1D_continuousDerivativeFourierSeries
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ) : C(BurgersTorus, ℂ) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  exact ∑' n : ℤ,
    periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n •
      (fourier n : C(BurgersTorus, ℂ))

theorem periodicHeat1D_continuousValueFourierSeries_hasSum
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) :
    HasSum
      (fun n : ℤ =>
        periodicHeat1D_evolvedValueFourierCoeff nu u0 t n •
          (fourier n : C(BurgersTorus, ℂ)))
      (periodicHeat1D_continuousValueFourierSeries nu u0 t) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  simpa [periodicHeat1D_continuousValueFourierSeries] using
    (periodicHeat1D_valueFourierContinuousTerm_summable
      nu u0 hu0 t ht).hasSum

theorem periodicHeat1D_continuousDerivativeFourierSeries_hasSum
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) :
    HasSum
      (fun n : ℤ =>
        periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n •
          (fourier n : C(BurgersTorus, ℂ)))
      (periodicHeat1D_continuousDerivativeFourierSeries nu u0 t) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  simpa [periodicHeat1D_continuousDerivativeFourierSeries] using
    (periodicHeat1D_derivativeFourierContinuousTerm_summable
      nu u0 hu0 t ht).hasSum

theorem periodicHeat1D_continuousValueFourierSeries_pointwise_hasSum
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (x : BurgersTorus) :
    HasSum
      (fun n : ℤ =>
        periodicHeat1D_evolvedValueFourierCoeff nu u0 t n * fourier n x)
      (periodicHeat1D_continuousValueFourierSeries nu u0 t x) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  simpa [smul_eq_mul] using
    (ContinuousMap.evalCLM ℂ x).hasSum
      (periodicHeat1D_continuousValueFourierSeries_hasSum nu u0 hu0 t ht)

theorem periodicHeat1D_continuousDerivativeFourierSeries_pointwise_hasSum
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (x : BurgersTorus) :
    HasSum
      (fun n : ℤ =>
        periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n * fourier n x)
      (periodicHeat1D_continuousDerivativeFourierSeries nu u0 t x) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  simpa [smul_eq_mul] using
    (ContinuousMap.evalCLM ℂ x).hasSum
      (periodicHeat1D_continuousDerivativeFourierSeries_hasSum nu u0 hu0 t ht)

def periodicHeat1D_spatialDerivativeFourierMultiplier
    (k : ℕ)
    (n : ℤ) : ℂ :=
  (2 * Real.pi * Complex.I * (n : ℂ)) ^ k

def periodicHeat1D_evolvedSpatialDerivativeFourierCoeff
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) : ℂ :=
  periodicHeat1D_spatialDerivativeFourierMultiplier k n *
    periodicHeat1D_evolvedValueFourierCoeff nu u0 t n

theorem periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_neg_eq_star
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ)
    (n : ℤ) :
    periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k (-n) =
      star (periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k n) := by
  simp [periodicHeat1D_evolvedSpatialDerivativeFourierCoeff,
    periodicHeat1D_spatialDerivativeFourierMultiplier,
    periodicHeat1D_evolvedValueFourierCoeff_neg_eq_star]

theorem periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_one
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (n : ℤ) :
    periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t 1 n =
      periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n := by
  rw [periodicHeat1D_evolvedDerivativeFourierCoeff_compatible_allModes
    nu u0 hu0 t n]
  simp [periodicHeat1D_evolvedSpatialDerivativeFourierCoeff,
    periodicHeat1D_spatialDerivativeFourierMultiplier]

theorem periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_zero
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (n : ℤ) :
    periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t 0 n =
      periodicHeat1D_evolvedValueFourierCoeff nu u0 t n := by
  simp [periodicHeat1D_evolvedSpatialDerivativeFourierCoeff,
    periodicHeat1D_spatialDerivativeFourierMultiplier]

def periodicHeat1D_evolvedTimeDerivativeFourierCoeff
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (n : ℤ) : ℂ :=
  (((-(nu.viscosity * periodicHeat1D_modeFrequency n) : ℝ) : ℂ) *
    periodicHeat1D_evolvedValueFourierCoeff nu u0 t n)

theorem periodicHeat1D_evolvedValueFourierCoeff_hasDerivAt_time
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (n : ℤ)
    (t : ℝ) :
    HasDerivAt (fun s : ℝ => periodicHeat1D_evolvedValueFourierCoeff nu u0 s n)
      (periodicHeat1D_evolvedTimeDerivativeFourierCoeff nu u0 t n) t := by
  have hm := periodicHeat1D_modeMultiplier_hasDerivAt nu n t
  have hc :
      HasDerivAt
        (fun s : ℝ => ((periodicHeat1D_modeMultiplier nu s n : ℝ) : ℂ))
        (((-(nu.viscosity * periodicHeat1D_modeFrequency n) *
          periodicHeat1D_modeMultiplier nu t n : ℝ) : ℂ)) t := by
    exact hm.ofReal_comp
  simpa [periodicHeat1D_evolvedValueFourierCoeff,
    periodicHeat1D_evolvedTimeDerivativeFourierCoeff,
    mul_assoc, mul_left_comm, mul_comm] using
    hc.mul_const (periodicH1_valueFourierCoeff u0 n)

theorem periodicHeat1D_spatialDerivativeFourierMultiplier_two
    (n : ℤ) :
    periodicHeat1D_spatialDerivativeFourierMultiplier 2 n =
      ((-(periodicHeat1D_modeFrequency n) : ℝ) : ℂ) := by
  unfold periodicHeat1D_spatialDerivativeFourierMultiplier
    periodicHeat1D_modeFrequency
  calc
    (2 * ↑Real.pi * Complex.I * ↑n) ^ 2
        = (2 * ↑Real.pi * ↑n : ℂ) ^ 2 * Complex.I ^ 2 := by ring
    _ = -((2 * ↑Real.pi * ↑n : ℂ) ^ 2) := by
      rw [Complex.I_sq]
      ring
    _ = ((-((2 * Real.pi * (n : ℝ)) ^ 2) : ℝ) : ℂ) := by
      norm_num

theorem periodicHeat1D_evolvedTimeDerivativeFourierCoeff_eq_viscosity_secondSpatial
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (n : ℤ) :
    periodicHeat1D_evolvedTimeDerivativeFourierCoeff nu u0 t n =
      (nu.viscosity : ℂ) *
        periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t 2 n := by
  have hmult := periodicHeat1D_spatialDerivativeFourierMultiplier_two n
  simp [periodicHeat1D_evolvedTimeDerivativeFourierCoeff,
    periodicHeat1D_evolvedSpatialDerivativeFourierCoeff, hmult]
  ring

theorem periodicHeat1D_evolvedTimeDerivativeFourierCoeff_norm_le_spatialAt
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    {eps t : ℝ}
    (n : ℤ)
    (heps_t : eps ≤ t) :
    ‖periodicHeat1D_evolvedTimeDerivativeFourierCoeff nu u0 t n‖ ≤
      nu.viscosity *
        ‖periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 eps 2 n‖ := by
  have hmul := periodicHeat1D_modeMultiplier_anti_mono_time
    nu n heps_t
  have hmul_nonneg_t := periodicHeat1D_modeMultiplier_nonneg nu t n
  have hmul_nonneg_eps := periodicHeat1D_modeMultiplier_nonneg nu eps n
  simp [periodicHeat1D_evolvedTimeDerivativeFourierCoeff,
    periodicHeat1D_evolvedSpatialDerivativeFourierCoeff,
    periodicHeat1D_spatialDerivativeFourierMultiplier_two,
    periodicHeat1D_evolvedValueFourierCoeff, norm_mul,
    Complex.normSq_eq_norm_sq, Complex.normSq_ofReal]
  rw [abs_of_nonneg (le_of_lt nu.viscosity_pos),
    abs_of_nonneg (periodicHeat1D_modeFrequency_nonneg n),
    abs_of_nonneg hmul_nonneg_t, abs_of_nonneg hmul_nonneg_eps]
  ring_nf
  have hcabs : 0 ≤ Complex.abs (periodicH1_valueFourierCoeff u0 n) :=
    Complex.abs.nonneg _
  have hfactor_nonneg :
      0 ≤ nu.viscosity * periodicHeat1D_modeFrequency n := by
    exact mul_nonneg (le_of_lt nu.viscosity_pos)
      (periodicHeat1D_modeFrequency_nonneg n)
  have hmul_c :
      periodicHeat1D_modeMultiplier nu t n *
          Complex.abs (periodicH1_valueFourierCoeff u0 n) ≤
        periodicHeat1D_modeMultiplier nu eps n *
          Complex.abs (periodicH1_valueFourierCoeff u0 n) := by
    exact mul_le_mul_of_nonneg_right hmul hcabs
  have hfinal := mul_le_mul_of_nonneg_left hmul_c hfactor_nonneg
  nlinarith

theorem periodicHeat1D_spatialDerivativeFourierMultiplier_norm
    (k : ℕ)
    (n : ℤ) :
    ‖periodicHeat1D_spatialDerivativeFourierMultiplier k n‖ =
      (2 * Real.pi) ^ k * (n.natAbs : ℝ) ^ k := by
  unfold periodicHeat1D_spatialDerivativeFourierMultiplier
  rw [norm_pow]
  have hbase : ‖(2 * Real.pi * Complex.I * (n : ℂ))‖ =
      (2 * Real.pi) * (n.natAbs : ℝ) := by
    rw [norm_mul, norm_mul, norm_mul]
    have h2 : ‖(2 : ℂ)‖ = 2 := by norm_num
    have hpi : ‖(Real.pi : ℂ)‖ = Real.pi := by
      simp [Complex.normSq_eq_norm_sq, Complex.normSq_ofReal,
        abs_of_nonneg Real.pi_pos.le]
    have hI : ‖Complex.I‖ = 1 := by simp
    have hn : ‖(n : ℂ)‖ = (n.natAbs : ℝ) := by
      rw [show ‖(n : ℂ)‖ = |(n : ℝ)| by
        simp [Complex.normSq_eq_norm_sq, Complex.normSq_ofReal]]
      simpa using (Int.cast_natAbs (R := ℝ) (n := n)).symm
    rw [h2, hpi, hI, hn]
    ring
  rw [hbase, mul_pow]

theorem periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_norm_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ) :
    Summable
      (fun n : ℤ =>
        ‖periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k n‖) := by
  have hbase :=
    periodicHeat1D_weightedEvolvedValueFourierCoeffNorm_summable
      nu u0 hu0 t ht k
  have hmul :
      Summable
        (fun n : ℤ =>
          (2 * Real.pi) ^ k *
            periodicHeat1D_weightedEvolvedValueFourierCoeffNorm nu u0 t k n) :=
    hbase.mul_left ((2 * Real.pi) ^ k)
  convert hmul using 1
  ext n
  unfold periodicHeat1D_evolvedSpatialDerivativeFourierCoeff
    periodicHeat1D_weightedEvolvedValueFourierCoeffNorm
  rw [norm_mul, periodicHeat1D_spatialDerivativeFourierMultiplier_norm]
  ring

theorem periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ) :
    Summable (periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k) :=
  (periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_norm_summable
    nu u0 hu0 t ht k).of_norm

theorem periodicHeat1D_spatialDerivativeFourierContinuousTerm_summable
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ) :
    Summable
      (fun n : ℤ =>
        periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k n •
          (fourier n : C(BurgersTorus, ℂ))) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  refine Summable.of_norm_bounded
    (fun n : ℤ =>
      ‖periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k n‖)
    (periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_norm_summable
      nu u0 hu0 t ht k)
    ?_
  intro n
  rw [norm_smul, fourier_norm]
  simp

def periodicHeat1D_continuousSpatialDerivativeFourierSeries
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ) : C(BurgersTorus, ℂ) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  exact ∑' n : ℤ,
    periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k n •
      (fourier n : C(BurgersTorus, ℂ))

theorem periodicHeat1D_continuousSpatialDerivativeFourierSeries_hasSum
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ) :
    HasSum
      (fun n : ℤ =>
        periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k n •
          (fourier n : C(BurgersTorus, ℂ)))
      (periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t k) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  simpa [periodicHeat1D_continuousSpatialDerivativeFourierSeries] using
    (periodicHeat1D_spatialDerivativeFourierContinuousTerm_summable
      nu u0 hu0 t ht k).hasSum

theorem periodicHeat1D_continuousSpatialDerivativeFourierSeries_pointwise_hasSum
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ)
    (x : BurgersTorus) :
    HasSum
      (fun n : ℤ =>
        periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k n *
          fourier n x)
      (periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t k x) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  simpa [smul_eq_mul] using
    (ContinuousMap.evalCLM ℂ x).hasSum
      (periodicHeat1D_continuousSpatialDerivativeFourierSeries_hasSum
        nu u0 hu0 t ht k)

theorem periodicHeat1D_continuousSpatialDerivativeFourierSeries_fourierCoeff
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ)
    (n : ℤ) :
    fourierCoeff
        (periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t k) n =
      periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k n := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  let F := periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t k
  let a := periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k
  have hC : HasSum (fun m : ℤ => a m • (fourier m : C(BurgersTorus, ℂ))) F := by
    simpa [F, a] using
      periodicHeat1D_continuousSpatialDerivativeFourierSeries_hasSum
        nu u0 hu0 t ht k
  have hLpRaw : HasSum
      (fun m : ℤ =>
        (ContinuousMap.toLp (E := ℂ) 2 AddCircle.haarAddCircle ℂ)
          (a m • (fourier m : C(BurgersTorus, ℂ))))
      ((ContinuousMap.toLp (E := ℂ) 2 AddCircle.haarAddCircle ℂ) F) := by
    exact (ContinuousMap.toLp (E := ℂ) 2 AddCircle.haarAddCircle ℂ).hasSum hC
  have hLp : HasSum
      (fun m : ℤ => a m • (fourierLp (T := (1 : ℝ)) 2 m))
      ((ContinuousMap.toLp (E := ℂ) 2 AddCircle.haarAddCircle ℂ) F) := by
    refine hLpRaw.congr_fun ?_
    intro m
    simp [fourierLp]
  calc
    fourierCoeff F n =
        fourierCoeff
          ((ContinuousMap.toLp (E := ℂ) 2 AddCircle.haarAddCircle ℂ) F) n := by
          exact (fourierCoeff_toLp F n).symm
    _ = a n :=
          periodicHeat1D_l2FourierSeries_fourierCoeff a
            ((ContinuousMap.toLp (E := ℂ) 2 AddCircle.haarAddCircle ℂ) F)
            hLp n

theorem periodicHeat1D_continuousSpatialDerivativeFourierSeries_ofReal_re
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ)
    (x : BurgersTorus) :
    ((periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t k x).re : ℂ) =
      periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t k x := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  let F := periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t k
  let a := periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k
  let term : ℤ → ℂ := fun n => a n * fourier n x
  have hcoeff_sym : ∀ n : ℤ, a (-n) = star (a n) := by
    intro n
    exact periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_neg_eq_star
      nu u0 t k n
  have hsum : HasSum term (F x) := by
    simpa [term, F, a] using
      periodicHeat1D_continuousSpatialDerivativeFourierSeries_pointwise_hasSum
        nu u0 hu0 t ht k x
  have hneg : HasSum (fun n : ℤ => term (-n)) (F x) := by
    have hinj : Function.Injective (fun n : ℤ => -n) := by
      intro m n h
      exact neg_injective h
    have hzero : ∀ y : ℤ, y ∉ Set.range (fun n : ℤ => -n) → term y = 0 := by
      intro y hy
      exfalso
      exact hy ⟨-y, by simp⟩
    exact (hinj.hasSum_iff (f := term) (g := fun n : ℤ => -n) (a := F x)
      hzero).2 hsum
  have hterm_star : (fun n : ℤ => term (-n)) = fun n : ℤ => star (term n) := by
    funext n
    simp [term, hcoeff_sym n, fourier_neg]
  have hstar : HasSum (fun n : ℤ => term (-n)) (star (F x)) := by
    exact hsum.star.congr_fun (by intro n; exact congrFun hterm_star n)
  have hfixed : star (F x) = F x := hstar.unique hneg
  apply Complex.ext
  · simp [F]
  · have him : -(F x).im = (F x).im := by
      have h := congrArg Complex.im hfixed
      simpa using h
    norm_num at him ⊢
    linarith

theorem periodicHeat1D_continuousSpatialDerivativeFourierSeries_lift_hasDerivAt
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ)
    (x : ℝ) :
    HasDerivAt
      (fun y : ℝ =>
        periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t k
          (y : UnitAddCircle))
      (periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t (k + 1)
        (x : UnitAddCircle))
      x := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  let coeff := periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t
  have hu : Summable (fun n : ℤ => ‖coeff (k + 1) n‖) :=
    periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_norm_summable
      nu u0 hu0 t ht (k + 1)
  have hg0 : Summable
      (fun n : ℤ => coeff k n * fourier n ((0 : ℝ) : UnitAddCircle)) := by
    refine Summable.of_norm_bounded (fun n : ℤ => ‖coeff k n‖)
      (periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_norm_summable
        nu u0 hu0 t ht k) ?_
    intro n
    rw [norm_mul]
    have hfourier_norm : ‖fourier n ((0 : ℝ) : UnitAddCircle)‖ = 1 := by
      rw [fourier_coe_apply, Complex.norm_eq_abs, Complex.abs_exp]
      simp
    rw [hfourier_norm]
    simp
  have hterm : ∀ n : ℤ, ∀ y : ℝ,
      HasDerivAt (fun z : ℝ => coeff k n * fourier n (z : UnitAddCircle))
        (coeff (k + 1) n * fourier n (y : UnitAddCircle)) y := by
    intro n y
    have hfourier := hasDerivAt_fourier (T := (1 : ℝ)) n y
    have hmul := hfourier.const_mul (coeff k n)
    convert hmul using 1
    dsimp [coeff, periodicHeat1D_evolvedSpatialDerivativeFourierCoeff,
      periodicHeat1D_spatialDerivativeFourierMultiplier]
    ring
  have hbound : ∀ n : ℤ, ∀ y : ℝ,
      ‖coeff (k + 1) n * fourier n (y : UnitAddCircle)‖ ≤
        ‖coeff (k + 1) n‖ := by
    intro n y
    rw [norm_mul]
    have hfourier_norm : ‖fourier n (y : UnitAddCircle)‖ = 1 := by
      rw [fourier_coe_apply, Complex.norm_eq_abs, Complex.abs_exp]
      simp
    rw [hfourier_norm]
    simp
  have hraw :
      HasDerivAt
        (fun y : ℝ => ∑' n : ℤ, coeff k n * fourier n (y : UnitAddCircle))
        (∑' n : ℤ, coeff (k + 1) n * fourier n (x : UnitAddCircle)) x := by
    exact hasDerivAt_tsum hu hterm hbound hg0 x
  convert hraw using 1
  · ext y
    exact ((periodicHeat1D_continuousSpatialDerivativeFourierSeries_pointwise_hasSum
      nu u0 hu0 t ht k (y : UnitAddCircle)).tsum_eq).symm
  · exact ((periodicHeat1D_continuousSpatialDerivativeFourierSeries_pointwise_hasSum
      nu u0 hu0 t ht (k + 1) (x : UnitAddCircle)).tsum_eq).symm

theorem periodicHeat1D_continuousValueFourierSeries_time_hasDerivAt
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (x : BurgersTorus) :
    HasDerivAt
      (fun s : ℝ =>
        periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 s 0 x)
      ((nu.viscosity : ℂ) *
        periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t 2 x)
      t := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  let eps : ℝ := t / 2
  have heps_pos : 0 < eps := by
    dsimp [eps]
    linarith
  have ht_mem : t ∈ Set.Ioi eps := by
    dsimp [eps]
    exact half_lt_self ht
  let u : ℤ → ℝ := fun n : ℤ =>
    nu.viscosity *
      ‖periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 eps 2 n‖
  have hu : Summable u := by
    dsimp [u]
    exact (periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_norm_summable
      nu u0 hu0 eps heps_pos 2).mul_left nu.viscosity
  let g : ℤ → ℝ → ℂ := fun n s =>
    periodicHeat1D_evolvedValueFourierCoeff nu u0 s n * fourier n x
  let g' : ℤ → ℝ → ℂ := fun n s =>
    periodicHeat1D_evolvedTimeDerivativeFourierCoeff nu u0 s n * fourier n x
  have hg : ∀ n : ℤ, ∀ y : ℝ, y ∈ Set.Ioi eps →
      HasDerivAt (g n) (g' n y) y := by
    intro n y _hy
    dsimp [g, g']
    exact (periodicHeat1D_evolvedValueFourierCoeff_hasDerivAt_time
      nu u0 n y).mul_const (fourier n x)
  have hg' : ∀ n : ℤ, ∀ y : ℝ, y ∈ Set.Ioi eps → ‖g' n y‖ ≤ u n := by
    intro n y hy
    dsimp [g', u]
    change ‖periodicHeat1D_evolvedTimeDerivativeFourierCoeff nu u0 y n *
        fourier n x‖ ≤
      nu.viscosity *
        ‖periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 eps 2 n‖
    rw [norm_mul]
    have hfourier_norm : ‖fourier n x‖ = 1 := by
      simp [fourier, Complex.norm_eq_abs, Complex.abs_exp]
    rw [hfourier_norm, mul_one]
    exact periodicHeat1D_evolvedTimeDerivativeFourierCoeff_norm_le_spatialAt
      nu u0 n (le_of_lt hy)
  have hg0 : Summable (fun n : ℤ => g n t) := by
    dsimp [g]
    exact (periodicHeat1D_continuousValueFourierSeries_pointwise_hasSum
      nu u0 hu0 t ht x).summable
  have hraw :
      HasDerivAt (fun z : ℝ => ∑' n : ℤ, g n z)
        (∑' n : ℤ, g' n t) t := by
    exact hasDerivAt_tsum_of_isPreconnected
      (u := u)
      (t := Set.Ioi eps)
      (y₀ := t)
      (y := t)
      hu isOpen_Ioi (convex_Ioi eps).isPreconnected hg hg' ht_mem hg0 ht_mem
  have hsource :
      (fun z : ℝ =>
        periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 z 0 x) =ᶠ[𝓝 t]
      (fun z : ℝ => ∑' n : ℤ, g n z) := by
    have hnhds : Set.Ioi eps ∈ 𝓝 t := isOpen_Ioi.mem_nhds ht_mem
    filter_upwards [hnhds] with z hz
    have hzpos : 0 < z := lt_trans heps_pos hz
    calc
      periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 z 0 x
          = ∑' n : ℤ,
              periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 z 0 n *
                fourier n x := by
              exact ((periodicHeat1D_continuousSpatialDerivativeFourierSeries_pointwise_hasSum
                nu u0 hu0 z hzpos 0 x).tsum_eq).symm
      _ = ∑' n : ℤ, g n z := by
              apply tsum_congr
              intro n
              dsimp [g]
              rw [periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_zero]
  have htarget :
      (∑' n : ℤ, g' n t) =
        (nu.viscosity : ℂ) *
          periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t 2 x := by
    have hsp :=
      periodicHeat1D_continuousSpatialDerivativeFourierSeries_pointwise_hasSum
        nu u0 hu0 t ht 2 x
    have htime : HasSum (fun n : ℤ => g' n t)
        ((nu.viscosity : ℂ) *
          periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t 2 x) := by
      refine (hsp.const_smul (nu.viscosity : ℂ)).congr_fun ?_
      intro n
      dsimp [g']
      rw [periodicHeat1D_evolvedTimeDerivativeFourierCoeff_eq_viscosity_secondSpatial]
      ring
    exact htime.tsum_eq
  exact (hraw.congr_of_eventuallyEq hsource).congr_deriv htarget

def periodicHeat1D_realSpatialDerivativeFourierSeries
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (k : ℕ) : C(BurgersTorus, ℝ) where
  toFun x :=
    (periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t k x).re
  continuous_toFun := Complex.continuous_re.comp
    (periodicHeat1D_continuousSpatialDerivativeFourierSeries
      nu u0 t k).continuous

/-- Real-valued positive-time heat derivative represented by the Fourier heat
equation `u_t = ν u_xx`. -/
def periodicHeat1D_realTimeDerivativeFourierSeries
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ) : C(BurgersTorus, ℝ) :=
  nu.viscosity • periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t 2

theorem periodicHeat1D_realTimeDerivativeFourierSeries_apply
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (t : ℝ)
    (x : BurgersTorus) :
    periodicHeat1D_realTimeDerivativeFourierSeries nu u0 t x =
      nu.viscosity *
        periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t 2 x := by
  rfl

theorem periodicHeat1D_realSpatialDerivativeFourierSeries_fourierCoeff
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ)
    (n : ℤ) :
    fourierCoeff
        (fun x : BurgersTorus =>
          (periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t k x : ℂ)) n =
      periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k n := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  calc
    fourierCoeff
        (fun x : BurgersTorus =>
          (periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t k x : ℂ)) n
        = fourierCoeff
            (periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t k) n := by
          unfold fourierCoeff
          refine integral_congr_ae ?_
          filter_upwards with x
          change (fourier (-n)) x •
              (((periodicHeat1D_continuousSpatialDerivativeFourierSeries
                nu u0 t k) x).re : ℂ) =
            (fourier (-n)) x •
              (periodicHeat1D_continuousSpatialDerivativeFourierSeries
                nu u0 t k) x
          rw [periodicHeat1D_continuousSpatialDerivativeFourierSeries_ofReal_re
            nu u0 hu0 t ht k x]
    _ = periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k n :=
          periodicHeat1D_continuousSpatialDerivativeFourierSeries_fourierCoeff
            nu u0 hu0 t ht k n

theorem periodicHeat1D_realSpatialDerivativeFourierSeries_lift_hasDerivAt
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ)
    (x : ℝ) :
    HasDerivAt
      (fun y : ℝ =>
        periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t k
          (y : UnitAddCircle))
      (periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t (k + 1)
        (x : UnitAddCircle))
      x := by
  have hc :=
    periodicHeat1D_continuousSpatialDerivativeFourierSeries_lift_hasDerivAt
      nu u0 hu0 t ht k x
  have hcomp :=
    (Complex.reCLM : ℂ →L[ℝ] ℝ).hasFDerivAt.comp_hasDerivAt x hc
  simpa [Function.comp_def, periodicHeat1D_realSpatialDerivativeFourierSeries,
    RCLike.reCLM_apply] using hcomp

theorem periodicHeat1D_realValueFourierSeries_time_hasDerivAt
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (x : BurgersTorus) :
    HasDerivAt
      (fun s : ℝ => periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 s 0 x)
      (periodicHeat1D_realTimeDerivativeFourierSeries nu u0 t x)
      t := by
  have hc :=
    periodicHeat1D_continuousValueFourierSeries_time_hasDerivAt
      nu u0 hu0 t ht x
  have hr :=
    (Complex.reCLM : ℂ →L[ℝ] ℝ).hasFDerivAt.comp_hasDerivAt t hc
  simpa [Function.comp_def, periodicHeat1D_realSpatialDerivativeFourierSeries,
    periodicHeat1D_realTimeDerivativeFourierSeries, RCLike.reCLM_apply,
    ContinuousMap.smul_apply, smul_eq_mul] using hr

theorem periodicHeat1D_realSpatialDerivative_isWeakDerivativeOnTorus
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k : ℕ) :
    IsWeakDerivativeOnTorus
      (periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t k)
      (periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t (k + 1)) := by
  refine isWeakDerivativeOnTorus_of_lift_hasDerivAt
    (periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t k)
    (periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t (k + 1))
    (fun y : ℝ =>
      periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t k
        (y : UnitAddCircle))
    (fun y : ℝ =>
      periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t (k + 1)
        (y : UnitAddCircle)) ?_ ?_ ?_
  · intro x
    rfl
  · intro x
    rfl
  · intro x
    exact periodicHeat1D_realSpatialDerivativeFourierSeries_lift_hasDerivAt
      nu u0 hu0 t ht k x

theorem periodicHeat1D_realSpatialDerivative_hasWeakDerivativeOrder
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (k order : ℕ) :
    HasWeakDerivativeOrder order
      (periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t k) := by
  induction order generalizing k with
  | zero =>
      exact continuousMap_memℒp_two
        (periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t k)
  | succ order ih =>
      exact ⟨periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t (k + 1),
        periodicHeat1D_realSpatialDerivative_isWeakDerivativeOnTorus
          nu u0 hu0 t ht k,
        ih (k + 1)⟩

def periodicHeat1D_positiveTimeSmoothPeriodicH1State
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) : PeriodicH1State where
  value := periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t 0
  weakDeriv := periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t 1
  value_memL2 := continuousMap_memℒp_two
    (periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t 0)
  deriv_memL2 := continuousMap_memℒp_two
    (periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t 1)
  weakDeriv_spec := by
    simpa using
      periodicHeat1D_realSpatialDerivative_isWeakDerivativeOnTorus
        nu u0 hu0 t ht 0

theorem periodicHeat1D_positiveTimeSmoothPeriodicH1State_smooth
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) :
    SmoothPeriodicState
      (periodicHeat1D_positiveTimeSmoothPeriodicH1State nu u0 hu0 t ht) := by
  intro order
  simpa [periodicHeat1D_positiveTimeSmoothPeriodicH1State] using
    periodicHeat1D_realSpatialDerivative_hasWeakDerivativeOrder
      nu u0 hu0 t ht 0 order

/-- Positive-time concrete real `PeriodicH1State` representative obtained by
taking real parts of the proved complex Fourier derivative tower. This package
does not yet identify the representative with the local heat curve at time
zero or prove the windowed heat residual/uniqueness facts; it supplies the
real-valued smooth state needed for the remaining heat-window upgrade. -/
structure PeriodicHeat1DPositiveTimeSmoothPeriodicH1RepresentativeFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) where
  state : PeriodicH1State
  real_spatial_derivative : ℕ → C(BurgersTorus, ℝ)
  real_spatial_derivative_eq :
    ∀ k : ℕ,
      real_spatial_derivative k =
        periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t k
  value_eq : state.value = real_spatial_derivative 0
  weakDeriv_eq : state.weakDeriv = real_spatial_derivative 1
  weak_derivative_chain :
    ∀ k : ℕ,
      IsWeakDerivativeOnTorus
        (real_spatial_derivative k)
        (real_spatial_derivative (k + 1))
  smooth : SmoothPeriodicState state

def periodicHeat1D_positiveTimeSmoothPeriodicH1Representative
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) :
    PeriodicHeat1DPositiveTimeSmoothPeriodicH1RepresentativeFor
      nu u0 hu0 t ht where
  state := periodicHeat1D_positiveTimeSmoothPeriodicH1State nu u0 hu0 t ht
  real_spatial_derivative := periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t
  real_spatial_derivative_eq := by
    intro k
    rfl
  value_eq := rfl
  weakDeriv_eq := rfl
  weak_derivative_chain := by
    intro k
    exact periodicHeat1D_realSpatialDerivative_isWeakDerivativeOnTorus
      nu u0 hu0 t ht k
  smooth := periodicHeat1D_positiveTimeSmoothPeriodicH1State_smooth
    nu u0 hu0 t ht

structure PeriodicHeat1DPositiveTimeSmoothPeriodicH1RepresentativeTheory where
  representative :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ)
      (ht : 0 < t),
      PeriodicHeat1DPositiveTimeSmoothPeriodicH1RepresentativeFor
        nu u0 hu0 t ht

def periodicHeat1D_positiveTimeSmoothPeriodicH1RepresentativeTheory_literature :
    PeriodicHeat1DPositiveTimeSmoothPeriodicH1RepresentativeTheory where
  representative := periodicHeat1D_positiveTimeSmoothPeriodicH1Representative

/-- Fully proved positive-time coefficient smoothing for periodic heat. It is
still below the continuous-window PDE boundary, but it proves the key Fourier
fact used by smoothing: after any positive time, the evolved value and
weak-derivative mode energies remain summable after multiplication by any fixed
polynomial frequency weight. -/
structure PeriodicHeat1DPositiveTimeCoefficientSmoothingTheory where
  gaussian_kernel :
    ∀ (k : ℕ) {r : ℝ}, 0 < r →
      Summable
        (fun n : ℤ =>
          ((n.natAbs : ℝ) ^ k) *
            Real.exp (-r * ((n : ℝ) ^ (2 : ℕ))))
  multiplier_kernel :
    ∀ (nu : BurgersParameters) (t : ℝ), 0 < t → ∀ k : ℕ,
      Summable
        (fun n : ℤ =>
          ((n.natAbs : ℝ) ^ k) *
            periodicHeat1D_modeMultiplier nu t n)
  multiplier_energy :
    ∀ (nu : BurgersParameters) (t : ℝ), 0 < t → ∀ k : ℕ,
      Summable (periodicHeat1D_weightedHeatMultiplierEnergy nu t k)
  value_energy :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ),
      0 < t → ∀ k : ℕ,
        Summable (periodicHeat1D_weightedEvolvedValueModeEnergy nu u0 t k)
  derivative_energy :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (_hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ),
      0 < t → ∀ k : ℕ,
        Summable
          (periodicHeat1D_weightedEvolvedDerivativeModeEnergy nu u0 t k)

def periodicHeat1D_positiveTimeCoefficientSmoothingTheory_literature :
    PeriodicHeat1DPositiveTimeCoefficientSmoothingTheory where
  gaussian_kernel := periodicHeat1D_intPolynomialGaussian_summable
  multiplier_kernel := periodicHeat1D_modeMultiplier_polynomial_summable
  multiplier_energy := periodicHeat1D_weightedHeatMultiplierEnergy_summable
  value_energy := periodicHeat1D_weightedEvolvedValueModeEnergy_summable
  derivative_energy :=
    periodicHeat1D_weightedEvolvedDerivativeModeEnergy_summable

/-- Positive-time Fourier-side heat representative.

This is the strongest currently proved object below the continuous
`PeriodicH1State` heat curve. It is a reconstructed Fourier/Sobolev `H¹` state
whose coefficients are exactly the heat-evolved coefficients and whose evolved
value/derivative mode energies have arbitrary polynomial frequency weights.
The remaining analytic step is to turn this Fourier-side representative into a
concrete continuous/smooth `PeriodicH1State` representative on a local window. -/
structure PeriodicHeat1DPositiveTimeSmoothFourierRepresentativeFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) where
  state : PeriodicFourierH1State
  value_coeff :
    ∀ n : ℤ,
      fourierCoeff state.valueL2 n =
        periodicHeat1D_evolvedValueFourierCoeff nu u0 t n
  derivative_coeff :
    ∀ n : ℤ,
      fourierCoeff state.derivativeL2 n =
        periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n
  weak_derivative_coeff :
    ∀ n : PeriodicH1NonzeroMode,
      fourierCoeff state.derivativeL2 n.1 =
        (2 * Real.pi * Complex.I * (n.1 : ℂ)) *
          fourierCoeff state.valueL2 n.1
  value_parseval :
    PeriodicFourierH1State.energy state =
      ∑' n : ℤ, periodicHeat1D_evolvedValueModeEnergy nu u0 t n
  derivative_parseval :
    PeriodicFourierH1State.derivativeEnergy state =
      ∑' n : ℤ, periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n
  value_contraction :
    PeriodicFourierH1State.energy state ≤ PeriodicH1State.energy u0
  derivative_contraction :
    PeriodicFourierH1State.derivativeEnergy state ≤
      PeriodicH1State.derivativeEnergy u0
  value_weighted_energy_summable :
    ∀ k : ℕ,
      Summable (periodicHeat1D_weightedEvolvedValueModeEnergy nu u0 t k)
  derivative_weighted_energy_summable :
    ∀ k : ℕ,
      Summable
        (periodicHeat1D_weightedEvolvedDerivativeModeEnergy nu u0 t k)
  value_coeff_norm_summable :
    ∀ k : ℕ,
      Summable
        (periodicHeat1D_weightedEvolvedValueFourierCoeffNorm nu u0 t k)
  derivative_coeff_norm_summable :
    ∀ k : ℕ,
      Summable
        (periodicHeat1D_weightedEvolvedDerivativeFourierCoeffNorm nu u0 t k)
  continuous_value : C(BurgersTorus, ℂ)
  continuous_derivative : C(BurgersTorus, ℂ)
  continuous_value_hasSum :
    HasSum
      (fun n : ℤ =>
        periodicHeat1D_evolvedValueFourierCoeff nu u0 t n •
          (fourier n : C(BurgersTorus, ℂ)))
      continuous_value
  continuous_derivative_hasSum :
    HasSum
      (fun n : ℤ =>
        periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n •
          (fourier n : C(BurgersTorus, ℂ)))
      continuous_derivative
  continuous_value_pointwise_hasSum :
    ∀ x : BurgersTorus,
      HasSum
        (fun n : ℤ =>
          periodicHeat1D_evolvedValueFourierCoeff nu u0 t n * fourier n x)
        (continuous_value x)
  continuous_derivative_pointwise_hasSum :
    ∀ x : BurgersTorus,
      HasSum
        (fun n : ℤ =>
          periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n * fourier n x)
        (continuous_derivative x)
  complex_spatial_derivative : ℕ → C(BurgersTorus, ℂ)
  complex_spatial_derivative_coeff_norm_summable :
    ∀ k : ℕ,
      Summable
        (fun n : ℤ =>
          ‖periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k n‖)
  complex_spatial_derivative_hasSum :
    ∀ k : ℕ,
      HasSum
        (fun n : ℤ =>
          periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k n •
            (fourier n : C(BurgersTorus, ℂ)))
        (complex_spatial_derivative k)
  complex_spatial_derivative_pointwise_hasSum :
    ∀ k : ℕ, ∀ x : BurgersTorus,
      HasSum
        (fun n : ℤ =>
          periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t k n *
            fourier n x)
        (complex_spatial_derivative k x)

def periodicHeat1D_positiveTimeSmoothFourierRepresentative
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) :
    PeriodicHeat1DPositiveTimeSmoothFourierRepresentativeFor
      nu u0 hu0 t ht where
  state := periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t (le_of_lt ht)
  value_coeff := by
    intro n
    simpa [periodicHeat1D_reconstructedFourierH1] using
      periodicHeat1D_reconstructedValueL2_fourierCoeff
        nu u0 hu0 t (le_of_lt ht) n
  derivative_coeff := by
    intro n
    simpa [periodicHeat1D_reconstructedFourierH1] using
      periodicHeat1D_reconstructedDerivativeL2_fourierCoeff
        nu u0 hu0 t (le_of_lt ht) n
  weak_derivative_coeff := by
    intro n
    exact periodicHeat1D_reconstructedFourierH1_derivative_coeff
      nu u0 hu0 t (le_of_lt ht) n
  value_parseval :=
    periodicHeat1D_reconstructedFourierH1_value_parseval
      nu u0 hu0 t (le_of_lt ht)
  derivative_parseval :=
    periodicHeat1D_reconstructedFourierH1_derivative_parseval
      nu u0 hu0 t (le_of_lt ht)
  value_contraction :=
    periodicHeat1D_reconstructedFourierH1_energy_le_initial
      nu u0 hu0 t (le_of_lt ht)
  derivative_contraction :=
    periodicHeat1D_reconstructedFourierH1_derivativeEnergy_le_initial
      nu u0 hu0 t (le_of_lt ht)
  value_weighted_energy_summable := by
    intro k
    exact periodicHeat1D_weightedEvolvedValueModeEnergy_summable
      nu u0 hu0 t ht k
  derivative_weighted_energy_summable := by
    intro k
    exact periodicHeat1D_weightedEvolvedDerivativeModeEnergy_summable
      nu u0 hu0 t ht k
  value_coeff_norm_summable := by
    intro k
    exact periodicHeat1D_weightedEvolvedValueFourierCoeffNorm_summable
      nu u0 hu0 t ht k
  derivative_coeff_norm_summable := by
    intro k
    exact periodicHeat1D_weightedEvolvedDerivativeFourierCoeffNorm_summable
      nu u0 hu0 t ht k
  continuous_value := periodicHeat1D_continuousValueFourierSeries nu u0 t
  continuous_derivative := periodicHeat1D_continuousDerivativeFourierSeries nu u0 t
  continuous_value_hasSum :=
    periodicHeat1D_continuousValueFourierSeries_hasSum nu u0 hu0 t ht
  continuous_derivative_hasSum :=
    periodicHeat1D_continuousDerivativeFourierSeries_hasSum nu u0 hu0 t ht
  continuous_value_pointwise_hasSum := by
    intro x
    exact periodicHeat1D_continuousValueFourierSeries_pointwise_hasSum
      nu u0 hu0 t ht x
  continuous_derivative_pointwise_hasSum := by
    intro x
    exact periodicHeat1D_continuousDerivativeFourierSeries_pointwise_hasSum
      nu u0 hu0 t ht x
  complex_spatial_derivative :=
    periodicHeat1D_continuousSpatialDerivativeFourierSeries nu u0 t
  complex_spatial_derivative_coeff_norm_summable := by
    intro k
    exact periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_norm_summable
      nu u0 hu0 t ht k
  complex_spatial_derivative_hasSum := by
    intro k
    exact periodicHeat1D_continuousSpatialDerivativeFourierSeries_hasSum
      nu u0 hu0 t ht k
  complex_spatial_derivative_pointwise_hasSum := by
    intro k x
    exact periodicHeat1D_continuousSpatialDerivativeFourierSeries_pointwise_hasSum
      nu u0 hu0 t ht k x

/-- Fully proved positive-time Fourier representative package. This is the
precise interface to feed into the next mathlib smooth-series reconstruction
step; it does not assume any continuous heat curve, weak PDE residual, or
windowed uniqueness theorem. -/
structure PeriodicHeat1DPositiveTimeSmoothFourierRepresentativeTheory where
  representative :
    ∀ (nu : BurgersParameters)
      (u0 : PeriodicH1State)
      (hu0 : PeriodicH1State.IsPeriodicH1 u0)
      (t : ℝ)
      (ht : 0 < t),
      PeriodicHeat1DPositiveTimeSmoothFourierRepresentativeFor
        nu u0 hu0 t ht

def periodicHeat1D_positiveTimeSmoothFourierRepresentativeTheory_literature :
    PeriodicHeat1DPositiveTimeSmoothFourierRepresentativeTheory where
  representative := periodicHeat1D_positiveTimeSmoothFourierRepresentative

/-- Literature-supplied local heat solution on one datum and one finite window,
before residual, uniqueness, smoothing, and contraction estimates are attached.
The final Burgers route consumes only the assembled local certificate below; any
all-time heat semigroup can still be used later as a construction source, but it
is no longer part of the theorem's exposed axiom boundary. -/
structure PeriodicHeat1DLocalExistenceWindowFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow) where
  certified : CertifiedHeatWindow nu
  initial_eq : certified.initial = u0
  window_eq : certified.window = W

structure PeriodicHeat1DLocalContractionFor
    {nu : BurgersParameters}
    (C : CertifiedHeatWindow nu) where
  energy : HeatWindowEnergyContraction C
  dissipation : HeatWindowDissipationContraction C

structure PeriodicHeat1DLocalTheoryFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow) where
  existence : PeriodicHeat1DLocalExistenceWindowFor nu u0 W
  residual : HeatWindowResidual nu existence.certified
  unique : HeatWindowUniqueness nu existence.certified
  smooth : HeatWindowSmoothing existence.certified
  contraction : PeriodicHeat1DLocalContractionFor existence.certified

/-- Fourier-reconstruction boundary for the local periodic heat equation.

Everything before this structure is proved in Lean at coefficient level. This
record isolates the remaining analytic reconstruction/PDE semantics: a
certified local heat curve must be reconstructed from the formal Fourier heat
coefficients, must satisfy the weak heat residual, uniqueness, and positive-time
smoothing, and must expose Parseval identities tying its reconstructed energy
back to those coefficients. The local energy and derivative-energy contractions
are then proved below from the already-discharged coefficient estimates. -/
structure PeriodicHeat1DFourierReconstructionFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow) where
  certified : CertifiedHeatWindow nu
  initial_eq : certified.initial = u0
  window_eq : certified.window = W
  value_coeff :
    ∀ t : ℝ, W.Contains t → 0 ≤ t → ∀ n : ℤ,
      periodicH1_valueFourierCoeff (certified.curve.eval t) n =
        periodicHeat1D_evolvedValueFourierCoeff nu u0 t n
  derivative_coeff :
    ∀ t : ℝ, W.Contains t → 0 ≤ t → ∀ n : ℤ,
      periodicH1_derivativeFourierCoeff (certified.curve.eval t) n =
        periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n
  value_parseval :
    ∀ t : ℝ, W.Contains t → 0 ≤ t →
      PeriodicH1State.energy (certified.curve.eval t) =
        ∑' n : ℤ, periodicHeat1D_evolvedValueModeEnergy nu u0 t n
  derivative_parseval :
    ∀ t : ℝ, W.Contains t → 0 ≤ t →
      PeriodicH1State.derivativeEnergy (certified.curve.eval t) =
        ∑' n : ℤ, periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n
  residual : HeatWindowResidual nu certified
  unique : HeatWindowUniqueness nu certified
  smooth : HeatWindowSmoothing certified

/-- Continuous-curve representative boundary for the proved Fourier/Sobolev
`H¹` heat reconstruction.

This is deliberately not a PDE semantics package. It only says that the
Fourier-`H¹` reconstruction can be represented by a concrete heat curve valued
in the current `PeriodicH1State` carrier, and that the representative has the
same Fourier coefficients and energies as the reconstructed Fourier-`H¹`
object on the requested local window. -/
structure PeriodicHeat1DFourierH1ContinuousCurveFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) where
  curve : PeriodicHeatCurve nu
  initial_eq : curve.eval 0 = u0
  value_fourierH1_coeff :
    ∀ t : ℝ, W.Contains t → (ht : 0 ≤ t) → ∀ n : ℤ,
      periodicH1_valueFourierCoeff (curve.eval t) n =
        fourierCoeff
          (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht).valueL2 n
  derivative_fourierH1_coeff :
    ∀ t : ℝ, W.Contains t → (ht : 0 ≤ t) → ∀ n : ℤ,
      periodicH1_derivativeFourierCoeff (curve.eval t) n =
        fourierCoeff
          (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht).derivativeL2 n
  value_fourierH1_parseval :
    ∀ t : ℝ, W.Contains t → (ht : 0 ≤ t) →
      PeriodicH1State.energy (curve.eval t) =
        PeriodicFourierH1State.energy
          (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht)
  derivative_fourierH1_parseval :
    ∀ t : ℝ, W.Contains t → (ht : 0 ≤ t) →
      PeriodicH1State.derivativeEnergy (curve.eval t) =
        PeriodicFourierH1State.derivativeEnergy
          (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht)
  positive_time_eval :
    ∀ t : ℝ, W.Contains t → (ht : 0 < t) →
      curve.eval t =
        periodicHeat1D_positiveTimeSmoothPeriodicH1State nu u0 hu0 t ht

namespace PeriodicHeat1DFourierH1ContinuousCurveFor

def certified
    {nu : BurgersParameters}
    {u0 : PeriodicH1State}
    {W : HeatWindow}
    {hu0 : PeriodicH1State.IsPeriodicH1 u0}
    (R : PeriodicHeat1DFourierH1ContinuousCurveFor nu u0 W hu0) :
    CertifiedHeatWindow nu where
  window := W
  initial := u0
  curve := R.curve
  initial_on_window := by
    intro _h0
    exact R.initial_eq
  boundary_on_window := by
    intro t _ht
    exact PeriodicH1State.isPeriodicH1 (R.curve.eval t)

end PeriodicHeat1DFourierH1ContinuousCurveFor

/-- Narrow remaining heat upgrade boundary.

At this point the heat flow has already been reconstructed in the
Fourier/Sobolev `H¹` carrier. This record asks only for the analytic upgrade
from that carrier to the concrete certified local heat window used by the
route: a continuous `PeriodicH1State` curve whose Fourier data and energies
match the reconstructed Fourier-`H¹` state, plus the weak heat residual,
uniqueness, and smoothing facts. -/
structure PeriodicHeat1DFourierH1WindowUpgradeFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) where
  certified : CertifiedHeatWindow nu
  initial_eq : certified.initial = u0
  window_eq : certified.window = W
  value_fourierH1_coeff :
    ∀ t : ℝ, W.Contains t → (ht : 0 ≤ t) → ∀ n : ℤ,
      periodicH1_valueFourierCoeff (certified.curve.eval t) n =
        fourierCoeff
          (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht).valueL2 n
  derivative_fourierH1_coeff :
    ∀ t : ℝ, W.Contains t → (ht : 0 ≤ t) → ∀ n : ℤ,
      periodicH1_derivativeFourierCoeff (certified.curve.eval t) n =
        fourierCoeff
          (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht).derivativeL2 n
  value_fourierH1_parseval :
    ∀ t : ℝ, W.Contains t → (ht : 0 ≤ t) →
      PeriodicH1State.energy (certified.curve.eval t) =
        PeriodicFourierH1State.energy
          (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht)
  derivative_fourierH1_parseval :
    ∀ t : ℝ, W.Contains t → (ht : 0 ≤ t) →
      PeriodicH1State.derivativeEnergy (certified.curve.eval t) =
        PeriodicFourierH1State.derivativeEnergy
          (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht)
  residual : HeatWindowResidual nu certified
  unique : HeatWindowUniqueness nu certified
  smooth : HeatWindowSmoothing certified

def periodicHeat1D_fourierH1WindowUpgrade_fromContinuousCurve
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (R : PeriodicHeat1DFourierH1ContinuousCurveFor nu u0 W hu0)
    (hresidual : HeatWindowResidual nu R.certified)
    (hunique : HeatWindowUniqueness nu R.certified)
    (hsmooth : HeatWindowSmoothing R.certified) :
    PeriodicHeat1DFourierH1WindowUpgradeFor nu u0 W hu0 where
  certified := R.certified
  initial_eq := rfl
  window_eq := rfl
  value_fourierH1_coeff := by
    intro t htW ht n
    exact R.value_fourierH1_coeff t htW ht n
  derivative_fourierH1_coeff := by
    intro t htW ht n
    exact R.derivative_fourierH1_coeff t htW ht n
  value_fourierH1_parseval := by
    intro t htW ht
    exact R.value_fourierH1_parseval t htW ht
  derivative_fourierH1_parseval := by
    intro t htW ht
    exact R.derivative_fourierH1_parseval t htW ht
  residual := hresidual
  unique := hunique
  smooth := hsmooth

def periodicHeat1D_fourierReconstruction_fromFourierH1WindowUpgrade
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (U : PeriodicHeat1DFourierH1WindowUpgradeFor nu u0 W hu0) :
    PeriodicHeat1DFourierReconstructionFor nu u0 W where
  certified := U.certified
  initial_eq := U.initial_eq
  window_eq := U.window_eq
  value_coeff := by
    intro t htW ht n
    calc
      periodicH1_valueFourierCoeff (U.certified.curve.eval t) n
          = fourierCoeff
              (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht).valueL2 n :=
            U.value_fourierH1_coeff t htW ht n
      _ = periodicHeat1D_evolvedValueFourierCoeff nu u0 t n := by
            simpa [periodicHeat1D_reconstructedFourierH1] using
              periodicHeat1D_reconstructedValueL2_fourierCoeff
                nu u0 hu0 t ht n
  derivative_coeff := by
    intro t htW ht n
    calc
      periodicH1_derivativeFourierCoeff (U.certified.curve.eval t) n
          = fourierCoeff
              (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht).derivativeL2 n :=
            U.derivative_fourierH1_coeff t htW ht n
      _ = periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n := by
            simpa [periodicHeat1D_reconstructedFourierH1] using
              periodicHeat1D_reconstructedDerivativeL2_fourierCoeff
                nu u0 hu0 t ht n
  value_parseval := by
    intro t htW ht
    calc
      PeriodicH1State.energy (U.certified.curve.eval t)
          = PeriodicFourierH1State.energy
              (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht) :=
            U.value_fourierH1_parseval t htW ht
      _ = ∑' n : ℤ, periodicHeat1D_evolvedValueModeEnergy nu u0 t n :=
            periodicHeat1D_reconstructedFourierH1_value_parseval
              nu u0 hu0 t ht
  derivative_parseval := by
    intro t htW ht
    calc
      PeriodicH1State.derivativeEnergy (U.certified.curve.eval t)
          = PeriodicFourierH1State.derivativeEnergy
              (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht) :=
            U.derivative_fourierH1_parseval t htW ht
      _ = ∑' n : ℤ, periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n :=
            periodicHeat1D_reconstructedFourierH1_derivative_parseval
              nu u0 hu0 t ht
  residual := U.residual
  unique := U.unique
  smooth := U.smooth

def periodicHeat1D_localContraction_fromFourierReconstruction
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (R : PeriodicHeat1DFourierReconstructionFor nu u0 W) :
    PeriodicHeat1DLocalContractionFor R.certified where
  energy := by
    intro t ht ht_nonneg
    have htW : W.Contains t := by
      simpa [R.window_eq] using ht
    calc
      PeriodicH1State.energy (R.certified.curve.eval t)
          = ∑' n : ℤ, periodicHeat1D_evolvedValueModeEnergy nu u0 t n :=
            R.value_parseval t htW ht_nonneg
      _ ≤ PeriodicH1State.energy R.certified.initial := by
            simpa [R.initial_eq] using
              periodicHeat1D_evolvedValueModeEnergy_tsum_le_initial
                nu u0 hu0 t ht_nonneg
  dissipation := by
    intro t ht ht_nonneg
    have htW : W.Contains t := by
      simpa [R.window_eq] using ht
    calc
      PeriodicH1State.derivativeEnergy (R.certified.curve.eval t)
          = ∑' n : ℤ, periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n :=
            R.derivative_parseval t htW ht_nonneg
      _ ≤ PeriodicH1State.derivativeEnergy R.certified.initial := by
            simpa [R.initial_eq] using
              periodicHeat1D_evolvedDerivativeModeEnergy_tsum_le_initial
                nu u0 hu0 t ht_nonneg

def periodicHeat1D_localTheory_fromFourierReconstruction
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (R : PeriodicHeat1DFourierReconstructionFor nu u0 W) :
    PeriodicHeat1DLocalTheoryFor nu u0 W :=
  { existence :=
      { certified := R.certified
        initial_eq := R.initial_eq
        window_eq := R.window_eq }
    residual := R.residual
    unique := R.unique
    smooth := R.smooth
    contraction :=
      periodicHeat1D_localContraction_fromFourierReconstruction
        nu u0 W hu0 R }

/-- Restrict any proved periodic heat construction source to the local heat
theory shape consumed by the hypostructure route. This theorem is framework
bookkeeping: it proves that a backend construction source exports exactly the
same window-local facts as the literature boundary below. -/
def periodicHeat1D_localTheory_fromSemigroupBackend
    {nu : BurgersParameters}
    (H : PeriodicHeatSemigroupBackend nu)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicHeat1DLocalTheoryFor nu u0 W :=
  let C := H.windowCertificate u0 W hu0
  { existence :=
      { certified := C.certified
        initial_eq := by rfl
        window_eq := by rfl }
    residual := C.residual
    unique := C.unique
    smooth := C.smooth
    contraction :=
      { energy := C.energy_contraction
        dissipation := C.dissipation_contraction } }

/-- Explicit local heat-curve candidate used by the Fourier heat package.

At positive times it is the proved smooth real Fourier representative. At
nonpositive times it is the initial datum. The `PeriodicHeatCurve` carrier has
no continuity field, so the remaining identification work is only coefficient,
energy, residual, and uniqueness semantics for this explicit curve. -/
def periodicHeat1D_fourierH1CandidateHeatCurve
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) : PeriodicHeatCurve nu where
  eval := fun t =>
    if ht : 0 < t then
      periodicHeat1D_positiveTimeSmoothPeriodicH1State nu u0 hu0 t ht
    else
      u0

theorem periodicHeat1D_fourierH1CandidateHeatCurve_initial
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    HeatInitialCondition u0
      (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) := by
  simp [HeatInitialCondition, periodicHeat1D_fourierH1CandidateHeatCurve]

theorem periodicHeat1D_fourierH1CandidateHeatCurve_boundary
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    HeatPeriodicBoundary
      (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) := by
  intro t
  by_cases ht : 0 < t
  · simp [periodicHeat1D_fourierH1CandidateHeatCurve, ht]
    exact PeriodicH1State.isPeriodicH1
      (periodicHeat1D_positiveTimeSmoothPeriodicH1State nu u0 hu0 t ht)
  · simp [periodicHeat1D_fourierH1CandidateHeatCurve, ht, hu0]

theorem periodicHeat1D_fourierH1CandidateHeatCurve_positive_time_eval
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t) :
    (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t =
      periodicHeat1D_positiveTimeSmoothPeriodicH1State nu u0 hu0 t ht := by
  simp [periodicHeat1D_fourierH1CandidateHeatCurve, ht]

/-- Canonical positive-time derivative of the explicit Fourier heat candidate.
It is zero at nonpositive times because the weak test interface used below only
requires positive-time differentiability. -/
def periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (_hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    ℝ → C(BurgersTorus, ℝ) :=
  fun t =>
    if 0 < t then
      periodicHeat1D_realTimeDerivativeFourierSeries nu u0 t
    else
      0

theorem periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv_positive_eval
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (x : BurgersTorus) :
    periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x =
      periodicHeat1D_realTimeDerivativeFourierSeries nu u0 t x := by
  simp [periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv, ht]

theorem periodicHeat1D_fourierH1CandidateHeatCurve_time_hasDerivAt_positive
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (t : ℝ)
    (ht : 0 < t)
    (x : BurgersTorus) :
    HasDerivAt
      (fun s : ℝ =>
        ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval s).value x)
      (periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x)
      t := by
  have hlocal :
      (fun s : ℝ =>
        ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval s).value x)
        =ᶠ[𝓝 t]
      (fun s : ℝ =>
        periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 s 0 x) := by
    have hnhds : Set.Ioi (0 : ℝ) ∈ 𝓝 t := isOpen_Ioi.mem_nhds ht
    filter_upwards [hnhds] with s hs
    have hpos : 0 < s := hs
    simp [periodicHeat1D_fourierH1CandidateHeatCurve, hpos,
      periodicHeat1D_positiveTimeSmoothPeriodicH1State]
  have hderiv :=
    periodicHeat1D_realValueFourierSeries_time_hasDerivAt
      nu u0 hu0 t ht x
  have htarget :
      periodicHeat1D_realTimeDerivativeFourierSeries nu u0 t x =
        periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x := by
    simp [periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv, ht]
  exact (hderiv.congr_of_eventuallyEq hlocal).congr_deriv htarget

/-- At a positive time, the explicit Fourier heat candidate satisfies the
spatial part of the weak heat identity. This is a local-in-time statement: it
uses only the positive-time Fourier representative and its weak derivative
chain, not any global heat semigroup theorem. -/
theorem periodicHeat1D_fourierH1CandidateHeatCurve_heat_balance_inner_positive
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (φ : SmoothCompactTimePeriodicSpaceTest)
    (t : ℝ)
    (ht : 0 < t) :
    (∫ x : BurgersTorus,
        heatWeakResidualIntegrand nu
          (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) φ t x) =
      -∫ x : BurgersTorus,
        (periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x *
            φ.value t x +
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t).value x *
            φ.timeDeriv t x) := by
  let f0 : C(BurgersTorus, ℝ) :=
    periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t 0
  let f1 : C(BurgersTorus, ℝ) :=
    periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t 1
  let f2 : C(BurgersTorus, ℝ) :=
    periodicHeat1D_realSpatialDerivativeFourierSeries nu u0 t 2
  let A : ℝ := ∫ x : BurgersTorus, f0 x * φ.timeDeriv t x
  let B : ℝ := ∫ x : BurgersTorus, f1 x * φ.spaceDeriv t x
  let C : ℝ := ∫ x : BurgersTorus, f2 x * φ.value t x
  have hweak : B = -C := by
    have h := periodicHeat1D_realSpatialDerivative_isWeakDerivativeOnTorus
      nu u0 hu0 t ht 1 (φ.spaceTest t)
    simpa [B, C, f1, f2, φ.spaceDeriv_eq_spaceTest t,
      φ.value_eq_spaceTest t] using h
  have hleft :
      (∫ x : BurgersTorus,
          heatWeakResidualIntegrand nu
            (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) φ t x) =
        -A + nu.viscosity * B := by
    have hcongr :
        (fun x : BurgersTorus =>
          heatWeakResidualIntegrand nu
            (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) φ t x)
          =ᵐ[volume]
        (fun x : BurgersTorus =>
          (-1 : ℝ) * (f0 x * φ.timeDeriv t x) +
            nu.viscosity * (f1 x * φ.spaceDeriv t x)) := by
      exact Filter.Eventually.of_forall fun x => by
        simp [heatWeakResidualIntegrand,
          periodicHeat1D_fourierH1CandidateHeatCurve_positive_time_eval
            nu u0 hu0 t ht,
          periodicHeat1D_positiveTimeSmoothPeriodicH1State, f0, f1,
          mul_assoc]
    calc
      (∫ x : BurgersTorus,
          heatWeakResidualIntegrand nu
            (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) φ t x)
          = ∫ x : BurgersTorus,
              ((-1 : ℝ) * (f0 x * φ.timeDeriv t x) +
                nu.viscosity * (f1 x * φ.spaceDeriv t x)) :=
            MeasureTheory.integral_congr_ae hcongr
      _ = (∫ x : BurgersTorus, (-1 : ℝ) * (f0 x * φ.timeDeriv t x)) +
            ∫ x : BurgersTorus,
              nu.viscosity * (f1 x * φ.spaceDeriv t x) := by
            refine MeasureTheory.integral_add ?_ ?_
            · exact (continuousMap_mul_integrable f0 (φ.timeDeriv t)).const_mul
                (-1 : ℝ)
            · exact (continuousMap_mul_integrable f1 (φ.spaceDeriv t)).const_mul
                nu.viscosity
      _ = -A + nu.viscosity * B := by
            simp [A, B,
              MeasureTheory.integral_mul_left, MeasureTheory.integral_neg]
  have hright :
      -∫ x : BurgersTorus,
        (periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x *
            φ.value t x +
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t).value x *
            φ.timeDeriv t x) =
        -(nu.viscosity * C + A) := by
    have hcongr :
        (fun x : BurgersTorus =>
          periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x *
              φ.value t x +
            ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t).value x *
              φ.timeDeriv t x)
          =ᵐ[volume]
        (fun x : BurgersTorus =>
          nu.viscosity * (f2 x * φ.value t x) +
            f0 x * φ.timeDeriv t x) := by
      exact Filter.Eventually.of_forall fun x => by
        simp [periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv,
          periodicHeat1D_realTimeDerivativeFourierSeries,
          periodicHeat1D_fourierH1CandidateHeatCurve_positive_time_eval
            nu u0 hu0 t ht,
          periodicHeat1D_positiveTimeSmoothPeriodicH1State, ht, f0, f2,
          mul_assoc]
    calc
      -∫ x : BurgersTorus,
        (periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x *
            φ.value t x +
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t).value x *
            φ.timeDeriv t x)
          = -∫ x : BurgersTorus,
              (nu.viscosity * (f2 x * φ.value t x) +
                f0 x * φ.timeDeriv t x) := by
            rw [MeasureTheory.integral_congr_ae hcongr]
      _ = -((∫ x : BurgersTorus,
              nu.viscosity * (f2 x * φ.value t x)) +
            ∫ x : BurgersTorus, f0 x * φ.timeDeriv t x) := by
            rw [MeasureTheory.integral_add]
            · exact (continuousMap_mul_integrable f2 (φ.value t)).const_mul
                nu.viscosity
            · exact continuousMap_mul_integrable f0 (φ.timeDeriv t)
      _ = -(nu.viscosity * C + A) := by
            simp [A, C, MeasureTheory.integral_mul_left]
  rw [hleft, hright, hweak]
  ring

/-- The same inner heat balance at nonpositive time. Here no heat regularity is
used: the test interface states that value, time derivative, and spatial
derivative vanish for `t ≤ 0`. -/
theorem periodicHeat1D_fourierH1CandidateHeatCurve_heat_balance_inner_nonpositive
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (φ : SmoothCompactTimePeriodicSpaceTest)
    (t : ℝ)
    (ht : t ≤ 0) :
    (∫ x : BurgersTorus,
        heatWeakResidualIntegrand nu
          (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) φ t x) =
      -∫ x : BurgersTorus,
        (periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x *
            φ.value t x +
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t).value x *
            φ.timeDeriv t x) := by
  have htime := φ.nonpositive_timeDeriv_zero t ht
  have hspace := φ.nonpositive_spaceDeriv_zero t ht
  have hvalue := φ.nonpositive_value_zero t ht
  have hleft_congr :
      (fun x : BurgersTorus =>
        heatWeakResidualIntegrand nu
          (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) φ t x)
        =ᵐ[volume]
      (fun _x : BurgersTorus => (0 : ℝ)) := by
    exact Filter.Eventually.of_forall fun x => by
      have htime_x : φ.timeDeriv t x = 0 := by rw [htime]; rfl
      have hspace_x : φ.spaceDeriv t x = 0 := by rw [hspace]; rfl
      simp [heatWeakResidualIntegrand, htime_x, hspace_x]
  have hright_congr :
      (fun x : BurgersTorus =>
        periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x *
            φ.value t x +
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t).value x *
            φ.timeDeriv t x)
        =ᵐ[volume]
      (fun _x : BurgersTorus => (0 : ℝ)) := by
    exact Filter.Eventually.of_forall fun x => by
      have htime_x : φ.timeDeriv t x = 0 := by rw [htime]; rfl
      have hvalue_x : φ.value t x = 0 := by rw [hvalue]; rfl
      simp [htime_x, hvalue_x]
  calc
    (∫ x : BurgersTorus,
        heatWeakResidualIntegrand nu
          (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) φ t x)
        = ∫ _x : BurgersTorus, (0 : ℝ) :=
          MeasureTheory.integral_congr_ae hleft_congr
    _ = 0 := by simp
    _ = -∫ _x : BurgersTorus, (0 : ℝ) := by simp
    _ = -∫ x : BurgersTorus,
        (periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x *
            φ.value t x +
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t).value x *
            φ.timeDeriv t x) := by
          rw [MeasureTheory.integral_congr_ae hright_congr]

theorem periodicHeat1D_fourierH1CandidateHeatCurve_heat_balance_inner
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (φ : SmoothCompactTimePeriodicSpaceTest)
    (t : ℝ) :
    (∫ x : BurgersTorus,
        heatWeakResidualIntegrand nu
          (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) φ t x) =
      -∫ x : BurgersTorus,
        (periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x *
            φ.value t x +
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t).value x *
            φ.timeDeriv t x) := by
  by_cases ht : 0 < t
  · exact periodicHeat1D_fourierH1CandidateHeatCurve_heat_balance_inner_positive
      nu u0 hu0 φ t ht
  · exact periodicHeat1D_fourierH1CandidateHeatCurve_heat_balance_inner_nonpositive
      nu u0 hu0 φ t (le_of_not_gt ht)

theorem periodicHeat1D_fourierH1CandidateHeatCurve_integrated_heat_balance
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (φ : SmoothCompactTimePeriodicSpaceTest) :
    timeSpaceIntegralOn W.time
        (heatWeakResidualIntegrand nu
          (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) φ) =
      -timeSpaceIntegralOn W.time
        (fun t x =>
          periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x *
              φ.value t x +
            ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t).value x *
              φ.timeDeriv t x) := by
  unfold timeSpaceIntegralOn
  calc
    (∫ t in W.time.t0..W.time.t1,
        ∫ x : BurgersTorus,
          heatWeakResidualIntegrand nu
            (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) φ t x)
        = ∫ t in W.time.t0..W.time.t1,
            -(∫ x : BurgersTorus,
              (periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x *
                  φ.value t x +
                ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t).value x *
                  φ.timeDeriv t x)) := by
          refine intervalIntegral.integral_congr_ae ?_
          exact Filter.Eventually.of_forall fun t _ht =>
            periodicHeat1D_fourierH1CandidateHeatCurve_heat_balance_inner
              nu u0 hu0 φ t
    _ = -∫ t in W.time.t0..W.time.t1,
            ∫ x : BurgersTorus,
              (periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x *
                  φ.value t x +
                ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t).value x *
                  φ.timeDeriv t x) := by
          rw [intervalIntegral.integral_neg]

/-- Remaining representative identification boundary for the explicit heat
curve candidate. It no longer constructs a curve or asserts smoothing; it only
identifies the candidate's positive-time Fourier coefficients with the formal
heat-evolved coefficients on the requested local window. The zero-time
coefficient identities and all energy matching with the proved
Fourier/Sobolev `H¹` reconstruction are derived below, so they are not part of
the boundary. -/
structure PeriodicHeat1DFourierH1CandidateIdentificationFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) where
  value_coeff :
    ∀ t : ℝ, W.Contains t → 0 < t → ∀ n : ℤ,
      periodicH1_valueFourierCoeff
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t) n =
        periodicHeat1D_evolvedValueFourierCoeff nu u0 t n
  derivative_coeff :
    ∀ t : ℝ, W.Contains t → 0 < t → ∀ n : ℤ,
      periodicH1_derivativeFourierCoeff
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t) n =
        periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n

/-- Integrating-factor conservation for the value Fourier modes of a weak
periodic heat solution on a local window.

This is the local scalar-mode shape of the remaining heat PDE boundary. It is
independent of the named initial datum: the right-hand side is the coefficient
at time zero of the same weak solution. The initial-condition field of
`SolvesPeriodicHeatWeak` is used below to convert this into heat-semigroup
evolution from `u0`. -/
structure PeriodicHeat1DWeakSolutionValueModeConservationFor
    (nu : BurgersParameters)
    (W : HeatWindow)
    (v : PeriodicHeatCurve nu) where
  conserved :
    ∀ t : ℝ, W.Contains t → 0 ≤ t → ∀ n : ℤ,
      (periodicHeat1D_modeMultiplier nu (-t) n : ℂ) *
          periodicH1_valueFourierCoeff (v.eval t) n =
        periodicH1_valueFourierCoeff (v.eval 0) n

/-- Value Fourier-coefficient evolution interface for arbitrary weak periodic
heat solutions on a local window.

This is the narrow heat PDE uniqueness boundary used by the backend. It does
not assert equality to the canonical curve directly, and it does not assume
weak-derivative coefficient evolution. The derivative-side evolution is derived
below from this value statement plus the `PeriodicH1State` weak-derivative
Fourier identity. This interface is now derived from integrating-factor
conservation rather than assumed directly. -/
structure PeriodicHeat1DWeakSolutionValueFourierEvolutionFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (v : PeriodicHeatCurve nu) where
  value_coeff :
    ∀ t : ℝ, W.Contains t → 0 ≤ t → ∀ n : ℤ,
      periodicH1_valueFourierCoeff (v.eval t) n =
        periodicHeat1D_evolvedValueFourierCoeff nu u0 t n

/-- Full value-and-weak-derivative Fourier-coefficient evolution interface.

This is no longer the axiom boundary. It is kept as the convenient downstream
interface used by uniqueness and reconstruction bookkeeping. -/
structure PeriodicHeat1DWeakSolutionFourierEvolutionFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (v : PeriodicHeatCurve nu) where
  value_coeff :
    ∀ t : ℝ, W.Contains t → 0 ≤ t → ∀ n : ℤ,
      periodicH1_valueFourierCoeff (v.eval t) n =
        periodicHeat1D_evolvedValueFourierCoeff nu u0 t n
  derivative_coeff :
    ∀ t : ℝ, W.Contains t → 0 ≤ t → ∀ n : ℤ,
      periodicH1_derivativeFourierCoeff (v.eval t) n =
        periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n

theorem periodicHeat1D_candidateValueCoeff_fromPositiveTimeIdentification
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (I : PeriodicHeat1DFourierH1CandidateIdentificationFor nu u0 W hu0)
    (t : ℝ)
    (htW : W.Contains t)
    (ht : 0 ≤ t)
    (n : ℤ) :
    periodicH1_valueFourierCoeff
        ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t) n =
      periodicHeat1D_evolvedValueFourierCoeff nu u0 t n := by
  by_cases hpos : 0 < t
  · exact I.value_coeff t htW hpos n
  · have ht0 : t = 0 := le_antisymm (not_lt.mp hpos) ht
    subst t
    simp [periodicHeat1D_fourierH1CandidateHeatCurve,
      periodicHeat1D_evolvedValueFourierCoeff_zero_time]

theorem periodicHeat1D_candidateDerivativeCoeff_fromPositiveTimeIdentification
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (I : PeriodicHeat1DFourierH1CandidateIdentificationFor nu u0 W hu0)
    (t : ℝ)
    (htW : W.Contains t)
    (ht : 0 ≤ t)
    (n : ℤ) :
    periodicH1_derivativeFourierCoeff
        ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t) n =
      periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n := by
  by_cases hpos : 0 < t
  · exact I.derivative_coeff t htW hpos n
  · have ht0 : t = 0 := le_antisymm (not_lt.mp hpos) ht
    subst t
    simp [periodicHeat1D_fourierH1CandidateHeatCurve,
      periodicHeat1D_evolvedDerivativeFourierCoeff_zero_time]

def periodicHeat1D_fourierH1CandidateHeatCurve_fourierEvolution_fromIdentification
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (I : PeriodicHeat1DFourierH1CandidateIdentificationFor nu u0 W hu0) :
    PeriodicHeat1DWeakSolutionFourierEvolutionFor
      nu u0 W (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) where
  value_coeff := by
    intro t htW ht n
    exact periodicHeat1D_candidateValueCoeff_fromPositiveTimeIdentification
      nu u0 W hu0 I t htW ht n
  derivative_coeff := by
    intro t htW ht n
    exact periodicHeat1D_candidateDerivativeCoeff_fromPositiveTimeIdentification
      nu u0 W hu0 I t htW ht n

def periodicHeat1D_fourierH1ContinuousCurve_fromCandidateIdentification
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (I : PeriodicHeat1DFourierH1CandidateIdentificationFor nu u0 W hu0) :
    PeriodicHeat1DFourierH1ContinuousCurveFor nu u0 W hu0 where
  curve := periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0
  initial_eq := periodicHeat1D_fourierH1CandidateHeatCurve_initial nu u0 hu0
  value_fourierH1_coeff := by
    intro t htW ht n
    calc
      periodicH1_valueFourierCoeff
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t) n
          = periodicHeat1D_evolvedValueFourierCoeff nu u0 t n :=
            periodicHeat1D_candidateValueCoeff_fromPositiveTimeIdentification
              nu u0 W hu0 I t htW ht n
      _ = fourierCoeff
            (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht).valueL2 n := by
            exact (periodicHeat1D_reconstructedValueL2_fourierCoeff
              nu u0 hu0 t ht n).symm
  derivative_fourierH1_coeff := by
    intro t htW ht n
    calc
      periodicH1_derivativeFourierCoeff
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t) n
          = periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n :=
            periodicHeat1D_candidateDerivativeCoeff_fromPositiveTimeIdentification
              nu u0 W hu0 I t htW ht n
      _ = fourierCoeff
            (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht).derivativeL2 n := by
            exact (periodicHeat1D_reconstructedDerivativeL2_fourierCoeff
              nu u0 hu0 t ht n).symm
  value_fourierH1_parseval := by
    intro t htW ht
    let C := (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t
    have hC : PeriodicH1State.IsPeriodicH1 C := by
      dsimp [C]
      exact periodicHeat1D_fourierH1CandidateHeatCurve_boundary nu u0 hu0 t
    calc
      PeriodicH1State.energy
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t)
          = ∑' n : ℤ, periodicH1_valueModeEnergy C n := by
            simpa [C] using (periodicH1_value_parseval_literature C hC).symm
      _ = ∑' n : ℤ, periodicHeat1D_evolvedValueModeEnergy nu u0 t n := by
            apply tsum_congr
            intro n
            have hcoeff :=
              periodicHeat1D_candidateValueCoeff_fromPositiveTimeIdentification
                nu u0 W hu0 I t htW ht n
            simpa [C, periodicH1_valueModeEnergy,
              periodicHeat1D_evolvedValueModeEnergy] using
              congrArg (fun z : ℂ => ‖z‖ ^ (2 : ℕ)) hcoeff
      _ = PeriodicFourierH1State.energy
            (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht) := by
            exact (periodicHeat1D_reconstructedFourierH1_value_parseval
              nu u0 hu0 t ht).symm
  derivative_fourierH1_parseval := by
    intro t htW ht
    let C := (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t
    have hC : PeriodicH1State.IsPeriodicH1 C := by
      dsimp [C]
      exact periodicHeat1D_fourierH1CandidateHeatCurve_boundary nu u0 hu0 t
    calc
      PeriodicH1State.derivativeEnergy
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t)
          = ∑' n : ℤ, periodicH1_derivativeModeEnergy C n := by
            simpa [C] using (periodicH1_derivative_parseval_literature C hC).symm
      _ = ∑' n : ℤ, periodicHeat1D_evolvedDerivativeModeEnergy nu u0 t n := by
            apply tsum_congr
            intro n
            have hcoeff :=
              periodicHeat1D_candidateDerivativeCoeff_fromPositiveTimeIdentification
                nu u0 W hu0 I t htW ht n
            simpa [C, periodicH1_derivativeModeEnergy,
              periodicHeat1D_evolvedDerivativeModeEnergy] using
              congrArg (fun z : ℂ => ‖z‖ ^ (2 : ℕ)) hcoeff
      _ = PeriodicFourierH1State.derivativeEnergy
            (periodicHeat1D_reconstructedFourierH1 nu u0 hu0 t ht) := by
            exact (periodicHeat1D_reconstructedFourierH1_derivative_parseval
              nu u0 hu0 t ht).symm
  positive_time_eval := by
    intro t _htW ht
    exact periodicHeat1D_fourierH1CandidateHeatCurve_positive_time_eval
      nu u0 hu0 t ht

/-- Positive-time coefficient identification for the explicit heat candidate.

This is no longer a literature boundary: the candidate is the real part of the
proved continuous Fourier series at positive time, and its `L²` Fourier
coefficients are recovered from the Hilbert-basis series. -/
def periodicHeat1D_fourierH1CandidateIdentification_literature :
  ∀ (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0),
      PeriodicHeat1DFourierH1CandidateIdentificationFor nu u0 W hu0 := by
  intro nu u0 W hu0
  refine
    { value_coeff := ?_
      derivative_coeff := ?_ }
  · intro t _htW ht n
    calc
      periodicH1_valueFourierCoeff
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t) n
          = periodicH1_valueFourierCoeff
              (periodicHeat1D_positiveTimeSmoothPeriodicH1State nu u0 hu0 t ht) n := by
              rw [periodicHeat1D_fourierH1CandidateHeatCurve_positive_time_eval
                nu u0 hu0 t ht]
      _ = periodicHeat1D_evolvedValueFourierCoeff nu u0 t n := by
              have h0 :
                  periodicH1_valueFourierCoeff
                      (periodicHeat1D_positiveTimeSmoothPeriodicH1State
                        nu u0 hu0 t ht) n =
                    periodicHeat1D_evolvedSpatialDerivativeFourierCoeff
                      nu u0 t 0 n := by
                simpa [periodicH1_valueFourierCoeff,
                  periodicHeat1D_positiveTimeSmoothPeriodicH1State] using
                  periodicHeat1D_realSpatialDerivativeFourierSeries_fourierCoeff
                    nu u0 hu0 t ht 0 n
              exact h0.trans
                (periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_zero
                  nu u0 t n)
  · intro t _htW ht n
    calc
      periodicH1_derivativeFourierCoeff
          ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t) n
          = periodicH1_derivativeFourierCoeff
              (periodicHeat1D_positiveTimeSmoothPeriodicH1State nu u0 hu0 t ht) n := by
              rw [periodicHeat1D_fourierH1CandidateHeatCurve_positive_time_eval
                nu u0 hu0 t ht]
      _ = periodicHeat1D_evolvedSpatialDerivativeFourierCoeff nu u0 t 1 n := by
              simpa [periodicH1_derivativeFourierCoeff,
                periodicHeat1D_positiveTimeSmoothPeriodicH1State] using
                periodicHeat1D_realSpatialDerivativeFourierSeries_fourierCoeff
                  nu u0 hu0 t ht 1 n
      _ = periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n :=
              periodicHeat1D_evolvedSpatialDerivativeFourierCoeff_one
                nu u0 hu0 t n

def periodicHeat1D_fourierH1CandidateHeatCurve_fourierEvolution
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicHeat1DWeakSolutionFourierEvolutionFor
      nu u0 W (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) :=
  periodicHeat1D_fourierH1CandidateHeatCurve_fourierEvolution_fromIdentification
    nu u0 W hu0
    (periodicHeat1D_fourierH1CandidateIdentification_literature nu u0 W hu0)

def periodicHeat1D_fourierH1ContinuousCurve_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicHeat1DFourierH1ContinuousCurveFor nu u0 W hu0 :=
  periodicHeat1D_fourierH1ContinuousCurve_fromCandidateIdentification
    nu u0 W hu0
    (periodicHeat1D_fourierH1CandidateIdentification_literature nu u0 W hu0)

/-- Integrated classical heat certificate for the concrete Fourier candidate.
This is no longer a literature axiom: the positive-time derivative is proved by
termwise Fourier differentiation, and the integrated weak heat balance is
assembled from the local spatial weak-derivative identity plus finite-window
time integration. -/
def periodicHeat1D_fourierH1IntegratedClassicalHeat_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    IntegratedClassicalHeatWindowCertificate nu
      (periodicHeat1D_fourierH1ContinuousCurve_literature nu u0 W hu0).certified := by
  refine
    { timeDeriv := periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0
      time_hasDerivAt_positive := ?_
      integrated_heat_balance := ?_ }
  · intro t ht x
    change HasDerivAt
      (fun s : ℝ =>
        ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval s).value x)
      (periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x) t
    exact periodicHeat1D_fourierH1CandidateHeatCurve_time_hasDerivAt_positive
      nu u0 hu0 t ht x
  · intro φ _hφ
    change timeSpaceIntegralOn W.time
        (heatWeakResidualIntegrand nu
          (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0) φ) =
      -timeSpaceIntegralOn W.time
        (fun t x =>
          periodicHeat1D_fourierH1CandidateHeatCurveTimeDeriv nu u0 hu0 t x *
              φ.value t x +
            ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t).value x *
              φ.timeDeriv t x)
    exact periodicHeat1D_fourierH1CandidateHeatCurve_integrated_heat_balance
      nu u0 W hu0 φ

theorem periodicHeat1D_fourierH1WindowResidual_literature :
  ∀ (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0),
      HeatWindowResidual nu
        (periodicHeat1D_fourierH1ContinuousCurve_literature nu u0 W hu0).certified := by
  intro nu u0 W hu0
  exact HeatWindowResidual.of_integratedClassical
    (periodicHeat1D_fourierH1IntegratedClassicalHeat_literature
      nu u0 W hu0)

/-- Remaining heat PDE boundary: arbitrary weak heat solutions have the
expected value Fourier-coefficient evolution on every finite forward-time
window.

This is strictly narrower than windowed uniqueness for the certified curve and
also narrower than full `H¹` coefficient evolution: weak-derivative coefficient
evolution is proved below from this value statement and the carrier's weak
derivative Fourier identity. -/
axiom periodicHeat1D_weakSolutionValueFourierEvolution_literature :
  ∀ (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (w : PeriodicHeatCurve nu),
      SolvesPeriodicHeatWeak nu u0 w →
        PeriodicHeat1DWeakSolutionValueFourierEvolutionFor nu u0 W w

def periodicHeat1D_weakSolutionFourierEvolution_fromValueEvolution
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (Hvalue : ∀ w : PeriodicHeatCurve nu,
      SolvesPeriodicHeatWeak nu u0 w →
        PeriodicHeat1DWeakSolutionValueFourierEvolutionFor nu u0 W w) :
    ∀ w : PeriodicHeatCurve nu,
      SolvesPeriodicHeatWeak nu u0 w →
        PeriodicHeat1DWeakSolutionFourierEvolutionFor nu u0 W w
  | w, hsol =>
      let Hv := Hvalue w hsol
      { value_coeff := by
          intro t htW ht n
          exact Hv.value_coeff t htW ht n
        derivative_coeff := by
          intro t htW ht n
          calc
            periodicH1_derivativeFourierCoeff (w.eval t) n
                = (2 * Real.pi * Complex.I * (n : ℂ)) *
                    periodicH1_valueFourierCoeff (w.eval t) n :=
                  periodicH1_weakDerivative_fourierCoeff_allModes_literature
                    (w.eval t) (hsol.2.1 t) n
            _ = (2 * Real.pi * Complex.I * (n : ℂ)) *
                  periodicHeat1D_evolvedValueFourierCoeff nu u0 t n := by
                  rw [Hv.value_coeff t htW ht n]
            _ = periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n := by
                  exact
                    (periodicHeat1D_evolvedDerivativeFourierCoeff_compatible_allModes
                      nu u0 (PeriodicH1State.isPeriodicH1 u0) t n).symm }

/-- Full coefficient evolution is a Lean consequence of the value-coefficient
literature boundary plus the concrete `PeriodicH1State` weak-derivative
identity. -/
def periodicHeat1D_weakSolutionFourierEvolution_literature :
  ∀ (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (w : PeriodicHeatCurve nu),
      SolvesPeriodicHeatWeak nu u0 w →
        PeriodicHeat1DWeakSolutionFourierEvolutionFor nu u0 W w := by
  intro nu u0 W
  exact periodicHeat1D_weakSolutionFourierEvolution_fromValueEvolution
    nu u0 W
    (fun w hw =>
      periodicHeat1D_weakSolutionValueFourierEvolution_literature
        nu u0 W w hw)

theorem periodicHeat1D_fourierH1WindowUniqueness_fromWeakSolutionFourierEvolution
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (Hweak : ∀ w : PeriodicHeatCurve nu,
      SolvesPeriodicHeatWeak nu u0 w →
        PeriodicHeat1DWeakSolutionFourierEvolutionFor nu u0 W w) :
    HeatWindowUniqueness nu
      (periodicHeat1D_fourierH1ContinuousCurve_literature nu u0 W hu0).certified := by
  intro w hw t htW ht_nonneg
  have hw_u0 : SolvesPeriodicHeatWeak nu u0 w := by
    simpa [periodicHeat1D_fourierH1ContinuousCurve_literature,
      periodicHeat1D_fourierH1ContinuousCurve_fromCandidateIdentification,
      PeriodicHeat1DFourierH1ContinuousCurveFor.certified] using hw
  have htW' : W.Contains t := by
    simpa [periodicHeat1D_fourierH1ContinuousCurve_literature,
      periodicHeat1D_fourierH1ContinuousCurve_fromCandidateIdentification,
      PeriodicHeat1DFourierH1ContinuousCurveFor.certified] using htW
  change w.eval t = (periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t
  have Hw := Hweak w hw_u0
  have Hc := periodicHeat1D_fourierH1CandidateHeatCurve_fourierEvolution
    nu u0 W hu0
  exact periodicH1State_ext_fourierCoeff
    (by
      intro n
      calc
        periodicH1_valueFourierCoeff (w.eval t) n
            = periodicHeat1D_evolvedValueFourierCoeff nu u0 t n :=
              Hw.value_coeff t htW' ht_nonneg n
        _ = periodicH1_valueFourierCoeff
              ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t) n :=
              (Hc.value_coeff t htW' ht_nonneg n).symm)
    (by
      intro n
      calc
        periodicH1_derivativeFourierCoeff (w.eval t) n
            = periodicHeat1D_evolvedDerivativeFourierCoeff nu u0 t n :=
              Hw.derivative_coeff t htW' ht_nonneg n
        _ = periodicH1_derivativeFourierCoeff
              ((periodicHeat1D_fourierH1CandidateHeatCurve nu u0 hu0).eval t) n :=
              (Hc.derivative_coeff t htW' ht_nonneg n).symm)

/-- Forward-time windowed uniqueness for the canonical Fourier-`H¹`
continuous representative. This is now a theorem from the narrower
coefficient-evolution boundary. -/
theorem periodicHeat1D_fourierH1WindowUniqueness_literature :
  ∀ (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0),
      HeatWindowUniqueness nu
        (periodicHeat1D_fourierH1ContinuousCurve_literature nu u0 W hu0).certified := by
  intro nu u0 W hu0
  exact periodicHeat1D_fourierH1WindowUniqueness_fromWeakSolutionFourierEvolution
    nu u0 W hu0
    (fun w hw =>
      periodicHeat1D_weakSolutionFourierEvolution_literature
        nu u0 W w hw)

/-- Positive-time smoothing is now derived from the continuous-curve
identification with the proved real Fourier `PeriodicH1State` representative,
not assumed as a separate heat PDE axiom. -/
theorem periodicHeat1D_fourierH1WindowSmoothing_fromContinuousCurve
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (R : PeriodicHeat1DFourierH1ContinuousCurveFor nu u0 W hu0) :
    HeatWindowSmoothing R.certified := by
  intro t htW htpos
  have htW' : W.Contains t := by
    simpa [PeriodicHeat1DFourierH1ContinuousCurveFor.certified] using htW
  dsimp [HeatSmoothAtPositiveTime]
  change SmoothPeriodicState (R.curve.eval t)
  rw [R.positive_time_eval t htW' htpos]
  exact periodicHeat1D_positiveTimeSmoothPeriodicH1State_smooth
    nu u0 hu0 t htpos

/-- The formerly monolithic heat upgrade is now a Lean assembler from the
continuous representative theorem plus the residual and uniqueness PDE
semantics. Smoothing is derived from positive-time representative
identification, not assumed here. -/
def periodicHeat1D_fourierH1WindowUpgrade_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicHeat1DFourierH1WindowUpgradeFor nu u0 W hu0 :=
  let R := periodicHeat1D_fourierH1ContinuousCurve_literature nu u0 W hu0
  periodicHeat1D_fourierH1WindowUpgrade_fromContinuousCurve
    nu u0 W hu0 R
    (periodicHeat1D_fourierH1WindowResidual_literature nu u0 W hu0)
    (periodicHeat1D_fourierH1WindowUniqueness_literature nu u0 W hu0)
    (periodicHeat1D_fourierH1WindowSmoothing_fromContinuousCurve nu u0 W hu0 R)

/-- Compatibility shim for the older reconstruction interface. It is no longer
an axiom: Lean derives coefficient/Parseval identities against the evolved heat
coefficients from the narrower Fourier-`H¹` window upgrade plus the proved
Fourier/Sobolev reconstruction theory. -/
def periodicHeat1D_fourierReconstruction_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicHeat1DFourierReconstructionFor nu u0 W :=
  periodicHeat1D_fourierReconstruction_fromFourierH1WindowUpgrade
    nu u0 W hu0
    (periodicHeat1D_fourierH1WindowUpgrade_literature nu u0 W hu0)

/-- Local heat theory assembled from Fourier reconstruction plus the proved
coefficient-level contraction package. This is no longer an axiom. -/
def periodicHeat1D_localTheory_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicHeat1DLocalTheoryFor nu u0 W :=
  periodicHeat1D_localTheory_fromFourierReconstruction
    nu u0 W hu0
    (periodicHeat1D_fourierReconstruction_literature nu u0 W hu0)

def periodicHeat1D_localExistenceWindow_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicHeat1DLocalExistenceWindowFor nu u0 W :=
  (periodicHeat1D_localTheory_literature nu u0 W hu0).existence

def periodicHeat1D_localResidual_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    HeatWindowResidual nu
      (periodicHeat1D_localExistenceWindow_literature nu u0 W hu0).certified :=
  (periodicHeat1D_localTheory_literature nu u0 W hu0).residual

def periodicHeat1D_localUniqueness_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    HeatWindowUniqueness nu
      (periodicHeat1D_localExistenceWindow_literature nu u0 W hu0).certified :=
  (periodicHeat1D_localTheory_literature nu u0 W hu0).unique

def periodicHeat1D_localSmoothing_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    HeatWindowSmoothing
      (periodicHeat1D_localExistenceWindow_literature nu u0 W hu0).certified :=
  (periodicHeat1D_localTheory_literature nu u0 W hu0).smooth

def periodicHeat1D_localContraction_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicHeat1DLocalContractionFor
      (periodicHeat1D_localExistenceWindow_literature nu u0 W hu0).certified :=
  (periodicHeat1D_localTheory_literature nu u0 W hu0).contraction

/-- Heat bad-germ exclusion is not a separate analytic assumption: by
definition of the heat forbidden predicate, local smoothing on the same window
excludes supported forbidden heat bad germs. -/
theorem periodicHeat1D_badGermExclusion_fromSmoothing
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    HeatBadGermWindowExclusion
      (periodicHeat1D_localExistenceWindow_literature nu u0 W hu0).certified.window
      (HeatForbiddenBadGerm
        (periodicHeat1D_localExistenceWindow_literature nu u0 W hu0).certified) :=
  HeatForbiddenBadGerm.excluded
    (periodicHeat1D_localSmoothing_literature nu u0 W hu0)

/-- Assembled heat certificate for one datum and one finite window. -/
structure PeriodicHeat1DLocalWindowCertificateFor
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow) where
  certificate : LocalHeatWindowCertificate nu
  initial_eq : certificate.certified.initial = u0
  window_eq : certificate.certified.window = W

/-- Assembler from decomposed periodic heat literature facts into the exact
local heat certificate consumed by the hypostructure route. This constructor is
proved in Lean; the literature assumptions are the existence/residual/
uniqueness/smoothing/contraction facts above. -/
def periodicHeat1D_localWindowCertificate_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicHeat1DLocalWindowCertificateFor nu u0 W :=
  let D := periodicHeat1D_localExistenceWindow_literature nu u0 W hu0
  let contraction := periodicHeat1D_localContraction_literature nu u0 W hu0
  { certificate :=
      { certified := D.certified
        residual := periodicHeat1D_localResidual_literature nu u0 W hu0
        unique := periodicHeat1D_localUniqueness_literature nu u0 W hu0
        smooth := periodicHeat1D_localSmoothing_literature nu u0 W hu0
        energy_contraction := contraction.energy
        dissipation_contraction := contraction.dissipation
        excludes_heat_bad_germs :=
          periodicHeat1D_badGermExclusion_fromSmoothing nu u0 W hu0 }
    initial_eq := D.initial_eq
    window_eq := D.window_eq }

def periodicHeat1D_localWindowCertificate
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    LocalHeatWindowCertificate nu :=
  (periodicHeat1D_localWindowCertificate_literature nu u0 W hu0).certificate

theorem periodicHeat1D_localWindowCertificate_initial_eq
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    (periodicHeat1D_localWindowCertificate nu u0 W hu0).certified.initial = u0 :=
  (periodicHeat1D_localWindowCertificate_literature nu u0 W hu0).initial_eq

theorem periodicHeat1D_localWindowCertificate_window_eq
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    (periodicHeat1D_localWindowCertificate nu u0 W hu0).certified.window = W :=
  (periodicHeat1D_localWindowCertificate_literature nu u0 W hu0).window_eq

theorem periodicHeat1D_localWindowCertificate_sound
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    (periodicHeat1D_localWindowCertificate nu u0 W hu0).certificateStatement :=
  localHeatSmoothingFrameworkCertificate_sound
    nu (periodicHeat1D_localWindowCertificate nu u0 W hu0)

end

end Hypostructure.Literature.Heat.Periodic1D
