import Hypostructure.Backends.Burgers1D.GroundTruthHeat
import Hypostructure.Literature.Analysis.PeriodicPoincare1D
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic

namespace Hypostructure.Literature.Heat.Periodic1D

open Hypostructure.Backends.Burgers1D
open Hypostructure.Literature.Analysis.PeriodicPoincare1D
open MeasureTheory
open scoped ENNReal

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

/-- Remaining heat representative boundary: produce a concrete continuous
`PeriodicH1State` curve representing the proved Fourier/Sobolev `H¹` heat
reconstruction on the requested local window. -/
axiom periodicHeat1D_fourierH1ContinuousCurve_literature :
  ∀ (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0),
      PeriodicHeat1DFourierH1ContinuousCurveFor nu u0 W hu0

/-- Remaining heat PDE boundary: the Fourier-`H¹` continuous representative
satisfies the weak heat residual on the certified local window. -/
axiom periodicHeat1D_fourierH1WindowResidual_literature :
  ∀ (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (R : PeriodicHeat1DFourierH1ContinuousCurveFor nu u0 W hu0),
      HeatWindowResidual nu R.certified

/-- Remaining heat PDE boundary: windowed uniqueness for the Fourier-`H¹`
continuous representative. -/
axiom periodicHeat1D_fourierH1WindowUniqueness_literature :
  ∀ (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (R : PeriodicHeat1DFourierH1ContinuousCurveFor nu u0 W hu0),
      HeatWindowUniqueness nu R.certified

/-- Remaining heat PDE boundary: positive-time smoothing for the Fourier-`H¹`
continuous representative on the certified local window. -/
axiom periodicHeat1D_fourierH1WindowSmoothing_literature :
  ∀ (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0)
    (R : PeriodicHeat1DFourierH1ContinuousCurveFor nu u0 W hu0),
      HeatWindowSmoothing R.certified

/-- The formerly monolithic heat upgrade is now a Lean assembler from the
continuous representative theorem plus the three windowed PDE semantics
theorems. -/
def periodicHeat1D_fourierH1WindowUpgrade_literature
    (nu : BurgersParameters)
    (u0 : PeriodicH1State)
    (W : HeatWindow)
    (hu0 : PeriodicH1State.IsPeriodicH1 u0) :
    PeriodicHeat1DFourierH1WindowUpgradeFor nu u0 W hu0 :=
  let R := periodicHeat1D_fourierH1ContinuousCurve_literature nu u0 W hu0
  periodicHeat1D_fourierH1WindowUpgrade_fromContinuousCurve
    nu u0 W hu0 R
    (periodicHeat1D_fourierH1WindowResidual_literature nu u0 W hu0 R)
    (periodicHeat1D_fourierH1WindowUniqueness_literature nu u0 W hu0 R)
    (periodicHeat1D_fourierH1WindowSmoothing_literature nu u0 W hu0 R)

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
