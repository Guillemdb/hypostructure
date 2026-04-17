import Hypostructure.Backends.Burgers1D.GroundTruthCertificates

import Mathlib.Analysis.Fourier.AddCircle

namespace Hypostructure.Literature.Analysis.PeriodicPoincare1D

open Hypostructure.Backends.Burgers1D
open MeasureTheory
open scoped ENNReal

noncomputable section

abbrev PeriodicH1NonzeroMode := {n : ℤ // n ≠ 0}

noncomputable def periodicH1_valueFourierCoeff
    (u : PeriodicH1State)
    (n : ℤ) : ℂ :=
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  fourierCoeff (fun x : BurgersTorus => (u.value x : ℂ)) n

noncomputable def periodicH1_derivativeFourierCoeff
    (u : PeriodicH1State)
    (n : ℤ) : ℂ :=
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  fourierCoeff (fun x : BurgersTorus => (u.weakDeriv x : ℂ)) n

noncomputable def periodicH1_valueNonzeroModeEnergy
    (u : PeriodicH1State)
    (n : PeriodicH1NonzeroMode) : ℝ :=
  ‖periodicH1_valueFourierCoeff u n.1‖ ^ (2 : ℕ)

noncomputable def periodicH1_derivativeNonzeroModeEnergy
    (u : PeriodicH1State)
    (n : PeriodicH1NonzeroMode) : ℝ :=
  ‖periodicH1_derivativeFourierCoeff u n.1‖ ^ (2 : ℕ)

noncomputable def periodicH1_valueModeEnergy
    (u : PeriodicH1State)
    (n : ℤ) : ℝ :=
  ‖periodicH1_valueFourierCoeff u n‖ ^ (2 : ℕ)

noncomputable def periodicH1_derivativeModeEnergy
    (u : PeriodicH1State)
    (n : ℤ) : ℝ :=
  ‖periodicH1_derivativeFourierCoeff u n‖ ^ (2 : ℕ)

private theorem periodicH1_haar_eq_volume :
    (volume : Measure BurgersTorus) = AddCircle.haarAddCircle := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  rw [AddCircle.volume_eq_smul_haarAddCircle (T := (1 : ℝ))]
  simp

private theorem periodicH1_complexContinuous_integrable
    (f : BurgersTorus → ℂ)
    (hf : Continuous f) :
    Integrable f (AddCircle.haarAddCircle (T := (1 : ℝ))) := by
  rw [← integrableOn_univ]
  exact hf.continuousOn.integrableOn_compact
    (μ := AddCircle.haarAddCircle (T := (1 : ℝ))) isCompact_univ

private noncomputable def periodicH1_valueComplexMap
    (u : PeriodicH1State) : C(BurgersTorus, ℂ) where
  toFun x := (u.value x : ℂ)
  continuous_toFun := Complex.continuous_ofReal.comp u.value.continuous

private noncomputable def periodicH1_derivativeComplexMap
    (u : PeriodicH1State) : C(BurgersTorus, ℂ) where
  toFun x := (u.weakDeriv x : ℂ)
  continuous_toFun := Complex.continuous_ofReal.comp u.weakDeriv.continuous

private theorem periodicH1_value_allModes_summable
    (u : PeriodicH1State) :
    Summable (fun n : ℤ => ‖periodicH1_valueFourierCoeff u n‖ ^ (2 : ℕ)) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  let f : Lp ℂ 2 AddCircle.haarAddCircle :=
    (ContinuousMap.toLp (E := ℂ) 2 AddCircle.haarAddCircle ℂ)
      (periodicH1_valueComplexMap u)
  have hs :=
    (lp.memℓp (fourierBasis.repr f)).summable
      (by norm_num : 0 < (2 : ℝ≥0∞).toReal)
  refine hs.congr ?_
  intro n
  rw [fourierBasis_repr, fourierCoeff_toLp]
  norm_num [periodicH1_valueFourierCoeff, periodicH1_valueComplexMap]

private theorem periodicH1_derivative_allModes_summable
    (u : PeriodicH1State) :
    Summable (fun n : ℤ => ‖periodicH1_derivativeFourierCoeff u n‖ ^ (2 : ℕ)) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  let f : Lp ℂ 2 AddCircle.haarAddCircle :=
    (ContinuousMap.toLp (E := ℂ) 2 AddCircle.haarAddCircle ℂ)
      (periodicH1_derivativeComplexMap u)
  have hs :=
    (lp.memℓp (fourierBasis.repr f)).summable
      (by norm_num : 0 < (2 : ℝ≥0∞).toReal)
  refine hs.congr ?_
  intro n
  rw [fourierBasis_repr, fourierCoeff_toLp]
  norm_num [periodicH1_derivativeFourierCoeff, periodicH1_derivativeComplexMap]

private theorem periodicH1_value_allModes_parseval
    (u : PeriodicH1State) :
    (∑' n : ℤ, ‖periodicH1_valueFourierCoeff u n‖ ^ (2 : ℕ)) =
      PeriodicH1State.energy u := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  have h :=
    tsum_sq_fourierCoeff
      ((ContinuousMap.toLp (E := ℂ) 2 AddCircle.haarAddCircle ℂ)
        (periodicH1_valueComplexMap u))
  simp_rw [fourierCoeff_toLp] at h
  have hrhs :
      (∫ t : BurgersTorus,
        ‖(((ContinuousMap.toLp (E := ℂ) 2 AddCircle.haarAddCircle ℂ)
          (periodicH1_valueComplexMap u) :
            Lp ℂ 2 AddCircle.haarAddCircle) t : ℂ)‖ ^ (2 : ℕ)
        ∂AddCircle.haarAddCircle) =
        PeriodicH1State.energy u := by
    dsimp [PeriodicH1State.energy]
    rw [periodicH1_haar_eq_volume]
    refine integral_congr_ae ?_
    have h_ae :=
      ContinuousMap.coeFn_toAEEqFun
        AddCircle.haarAddCircle (periodicH1_valueComplexMap u)
    filter_upwards [h_ae] with t ht
    rw [ht]
    simp [periodicH1_valueComplexMap]
  rw [← hrhs]
  exact h

private theorem periodicH1_derivative_allModes_parseval
    (u : PeriodicH1State) :
    (∑' n : ℤ, ‖periodicH1_derivativeFourierCoeff u n‖ ^ (2 : ℕ)) =
      PeriodicH1State.derivativeEnergy u := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  have h :=
    tsum_sq_fourierCoeff
      ((ContinuousMap.toLp (E := ℂ) 2 AddCircle.haarAddCircle ℂ)
        (periodicH1_derivativeComplexMap u))
  simp_rw [fourierCoeff_toLp] at h
  have hrhs :
      (∫ t : BurgersTorus,
        ‖(((ContinuousMap.toLp (E := ℂ) 2 AddCircle.haarAddCircle ℂ)
          (periodicH1_derivativeComplexMap u) :
            Lp ℂ 2 AddCircle.haarAddCircle) t : ℂ)‖ ^ (2 : ℕ)
        ∂AddCircle.haarAddCircle) =
        PeriodicH1State.derivativeEnergy u := by
    dsimp [PeriodicH1State.derivativeEnergy]
    rw [periodicH1_haar_eq_volume]
    refine integral_congr_ae ?_
    have h_ae :=
      ContinuousMap.coeFn_toAEEqFun
        AddCircle.haarAddCircle (periodicH1_derivativeComplexMap u)
    filter_upwards [h_ae] with t ht
    rw [ht]
    simp [periodicH1_derivativeComplexMap]
  rw [← hrhs]
  exact h

private theorem periodicH1_value_zeroCoeff
    (u : PeriodicH1State) :
    periodicH1_valueFourierCoeff u 0 = (PeriodicH1State.mean u : ℂ) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  dsimp [periodicH1_valueFourierCoeff, PeriodicH1State.mean, fourierCoeff]
  simp_rw [zero_zsmul, AddCircle.toCircle_zero]
  simp only [Circle.coe_one, one_mul]
  calc
    ∫ t : BurgersTorus, (u.value t : ℂ)
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))
        = (((∫ t : BurgersTorus, u.value t
            ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))) : ℝ) : ℂ) := by
          simpa using
            (integral_ofReal
              (μ := AddCircle.haarAddCircle (T := (1 : ℝ))) (𝕜 := ℂ)
              (f := fun t : BurgersTorus => u.value t))
    _ = (((∫ x : BurgersTorus, u.value x) : ℝ) : ℂ) := by
          congr 1
          rw [periodicH1_haar_eq_volume]

private theorem periodicH1_value_allModes_split
    (u : PeriodicH1State) :
    ∑' n : ℤ, ‖periodicH1_valueFourierCoeff u n‖ ^ (2 : ℕ) =
      ‖periodicH1_valueFourierCoeff u 0‖ ^ (2 : ℕ) +
        ∑' n : PeriodicH1NonzeroMode, periodicH1_valueNonzeroModeEnergy u n := by
  let f : ℤ → ℝ := fun n => ‖periodicH1_valueFourierCoeff u n‖ ^ (2 : ℕ)
  have hs : Summable f := periodicH1_value_allModes_summable u
  have hite := tsum_eq_add_tsum_ite hs 0
  have hind :
      (∑' n : ℤ, (if n = 0 then 0 else f n)) =
        ∑' n : PeriodicH1NonzeroMode, periodicH1_valueNonzeroModeEnergy u n := by
    calc
      (∑' n : ℤ, (if n = 0 then 0 else f n))
          = ∑' n : ℤ, ({n : ℤ | n ≠ 0}.indicator f n) := by
            apply tsum_congr
            intro n
            by_cases h : n = 0 <;> simp [h]
      _ = ∑' n : {n : ℤ // n ≠ 0}, f n.1 := by
            exact (tsum_subtype ({n : ℤ | n ≠ 0}) f).symm
      _ = ∑' n : PeriodicH1NonzeroMode, periodicH1_valueNonzeroModeEnergy u n := by
            rfl
  rw [hite, hind]

private theorem periodicH1_derivative_allModes_split
    (u : PeriodicH1State) :
    ∑' n : ℤ, ‖periodicH1_derivativeFourierCoeff u n‖ ^ (2 : ℕ) =
      ‖periodicH1_derivativeFourierCoeff u 0‖ ^ (2 : ℕ) +
        ∑' n : PeriodicH1NonzeroMode,
          periodicH1_derivativeNonzeroModeEnergy u n := by
  let f : ℤ → ℝ := fun n => ‖periodicH1_derivativeFourierCoeff u n‖ ^ (2 : ℕ)
  have hs : Summable f := periodicH1_derivative_allModes_summable u
  have hite := tsum_eq_add_tsum_ite hs 0
  have hind :
      (∑' n : ℤ, (if n = 0 then 0 else f n)) =
        ∑' n : PeriodicH1NonzeroMode,
          periodicH1_derivativeNonzeroModeEnergy u n := by
    calc
      (∑' n : ℤ, (if n = 0 then 0 else f n))
          = ∑' n : ℤ, ({n : ℤ | n ≠ 0}.indicator f n) := by
            apply tsum_congr
            intro n
            by_cases h : n = 0 <;> simp [h]
      _ = ∑' n : {n : ℤ // n ≠ 0}, f n.1 := by
            exact (tsum_subtype ({n : ℤ | n ≠ 0}) f).symm
      _ = ∑' n : PeriodicH1NonzeroMode,
            periodicH1_derivativeNonzeroModeEnergy u n := by
            rfl
  rw [hite, hind]

private noncomputable def periodicH1_fourierRealTest
    (n : ℤ) : SmoothPeriodicTestFunction where
  value :=
    { toFun := fun x : BurgersTorus => ((fourier (-n) x : ℂ).re)
      continuous_toFun := Complex.continuous_re.comp (fourier (-n)).continuous }
  deriv :=
    { toFun := fun x : BurgersTorus =>
        -(((2 * Real.pi * Complex.I * (n : ℂ)) * fourier (-n) x : ℂ).re)
      continuous_toFun :=
        (Complex.continuous_re.comp
          (continuous_const.mul (fourier (-n)).continuous)).neg }
  lift := fun x : ℝ => ((fourier (-n) (x : BurgersTorus) : ℂ).re)
  liftDeriv := fun x : ℝ =>
    -(((2 * Real.pi * Complex.I * (n : ℂ)) *
      fourier (-n) (x : BurgersTorus) : ℂ).re)
  value_lift := by intro x; rfl
  deriv_lift := by intro x; rfl
  hasDerivAt_lift := by
    intro x
    letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
    have h := hasDerivAt_fourier_neg (T := (1 : ℝ)) n x
    simpa [Function.comp_def, neg_mul] using
      ((Complex.reCLM.hasFDerivAt.comp x h.hasFDerivAt).hasDerivAt)

private noncomputable def periodicH1_fourierImagTest
    (n : ℤ) : SmoothPeriodicTestFunction where
  value :=
    { toFun := fun x : BurgersTorus => ((fourier (-n) x : ℂ).im)
      continuous_toFun := Complex.continuous_im.comp (fourier (-n)).continuous }
  deriv :=
    { toFun := fun x : BurgersTorus =>
        -(((2 * Real.pi * Complex.I * (n : ℂ)) * fourier (-n) x : ℂ).im)
      continuous_toFun :=
        (Complex.continuous_im.comp
          (continuous_const.mul (fourier (-n)).continuous)).neg }
  lift := fun x : ℝ => ((fourier (-n) (x : BurgersTorus) : ℂ).im)
  liftDeriv := fun x : ℝ =>
    -(((2 * Real.pi * Complex.I * (n : ℂ)) *
      fourier (-n) (x : BurgersTorus) : ℂ).im)
  value_lift := by intro x; rfl
  deriv_lift := by intro x; rfl
  hasDerivAt_lift := by
    intro x
    letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
    have h := hasDerivAt_fourier_neg (T := (1 : ℝ)) n x
    simpa [Function.comp_def, neg_mul] using
      ((Complex.imCLM.hasFDerivAt.comp x h.hasFDerivAt).hasDerivAt)

private theorem periodicH1_weakDerivative_fourierCoeff_realPart
    (u : PeriodicH1State)
    (hu : PeriodicH1State.IsPeriodicH1 u)
    (n : ℤ) :
    ∫ x : BurgersTorus, u.weakDeriv x * (fourier (-n) x : ℂ).re
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) =
      ∫ x : BurgersTorus,
        u.value x * (((2 * Real.pi * Complex.I * (n : ℂ)) *
          fourier (-n) x : ℂ).re)
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  have h := hu.2.2 (periodicH1_fourierRealTest n)
  rw [periodicH1_haar_eq_volume] at h
  change (∫ x : BurgersTorus,
      u.value x *
        (-(((2 * Real.pi * Complex.I * (n : ℂ)) *
          fourier (-n) x : ℂ).re))
      ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))) =
    -∫ x : BurgersTorus, u.weakDeriv x * (fourier (-n) x : ℂ).re
      ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) at h
  simp_rw [mul_neg] at h
  rw [integral_neg] at h
  linarith

private theorem periodicH1_weakDerivative_fourierCoeff_imagPart
    (u : PeriodicH1State)
    (hu : PeriodicH1State.IsPeriodicH1 u)
    (n : ℤ) :
    ∫ x : BurgersTorus, u.weakDeriv x * (fourier (-n) x : ℂ).im
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) =
      ∫ x : BurgersTorus,
        u.value x * (((2 * Real.pi * Complex.I * (n : ℂ)) *
          fourier (-n) x : ℂ).im)
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  have h := hu.2.2 (periodicH1_fourierImagTest n)
  rw [periodicH1_haar_eq_volume] at h
  change (∫ x : BurgersTorus,
      u.value x *
        (-(((2 * Real.pi * Complex.I * (n : ℂ)) *
          fourier (-n) x : ℂ).im))
      ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))) =
    -∫ x : BurgersTorus, u.weakDeriv x * (fourier (-n) x : ℂ).im
      ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) at h
  simp_rw [mul_neg] at h
  rw [integral_neg] at h
  linarith

private theorem periodicH1_derivativeCoeff_re
    (u : PeriodicH1State)
    (n : ℤ) :
    (periodicH1_derivativeFourierCoeff u n).re =
      ∫ x : BurgersTorus, u.weakDeriv x * (fourier (-n) x : ℂ).re
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  unfold periodicH1_derivativeFourierCoeff
  unfold fourierCoeff
  change
    (∫ t : BurgersTorus, (fourier (-n) t : ℂ) *
      (u.weakDeriv t : ℂ) ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))).re =
    ∫ x : BurgersTorus, u.weakDeriv x * (fourier (-n) x : ℂ).re
      ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))
  have hint :
      Integrable
        (fun t : BurgersTorus =>
          (fourier (-n) t : ℂ) * (u.weakDeriv t : ℂ))
        (AddCircle.haarAddCircle (T := (1 : ℝ))) := by
    refine periodicH1_complexContinuous_integrable _ ?_
    exact (fourier (-n)).continuous.mul
      (Complex.continuous_ofReal.comp u.weakDeriv.continuous)
  calc
    (∫ t : BurgersTorus, (fourier (-n) t : ℂ) *
      (u.weakDeriv t : ℂ) ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))).re
        = ∫ x : BurgersTorus,
            (((fourier (-n) x : ℂ) * (u.weakDeriv x : ℂ)).re)
            ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
          exact (integral_re hint).symm
    _ = ∫ x : BurgersTorus, u.weakDeriv x * (fourier (-n) x : ℂ).re
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
          refine integral_congr_ae ?_
          filter_upwards with x
          simp [Complex.mul_re, mul_comm, mul_left_comm, mul_assoc]

private theorem periodicH1_derivativeCoeff_im
    (u : PeriodicH1State)
    (n : ℤ) :
    (periodicH1_derivativeFourierCoeff u n).im =
      ∫ x : BurgersTorus, u.weakDeriv x * (fourier (-n) x : ℂ).im
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  unfold periodicH1_derivativeFourierCoeff
  unfold fourierCoeff
  change
    (∫ t : BurgersTorus, (fourier (-n) t : ℂ) *
      (u.weakDeriv t : ℂ) ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))).im =
    ∫ x : BurgersTorus, u.weakDeriv x * (fourier (-n) x : ℂ).im
      ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))
  have hint :
      Integrable
        (fun t : BurgersTorus =>
          (fourier (-n) t : ℂ) * (u.weakDeriv t : ℂ))
        (AddCircle.haarAddCircle (T := (1 : ℝ))) := by
    refine periodicH1_complexContinuous_integrable _ ?_
    exact (fourier (-n)).continuous.mul
      (Complex.continuous_ofReal.comp u.weakDeriv.continuous)
  calc
    (∫ t : BurgersTorus, (fourier (-n) t : ℂ) *
      (u.weakDeriv t : ℂ) ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))).im
        = ∫ x : BurgersTorus,
            (((fourier (-n) x : ℂ) * (u.weakDeriv x : ℂ)).im)
            ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
          exact (integral_im hint).symm
    _ = ∫ x : BurgersTorus, u.weakDeriv x * (fourier (-n) x : ℂ).im
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
          refine integral_congr_ae ?_
          filter_upwards with x
          simp [Complex.mul_im, mul_comm, mul_left_comm, mul_assoc]

private theorem periodicH1_multiplierValueCoeff_re
    (u : PeriodicH1State)
    (n : ℤ) :
    ((2 * Real.pi * Complex.I * (n : ℂ)) *
      periodicH1_valueFourierCoeff u n).re =
      ∫ x : BurgersTorus,
        u.value x * (((2 * Real.pi * Complex.I * (n : ℂ)) *
          fourier (-n) x : ℂ).re)
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  unfold periodicH1_valueFourierCoeff
  unfold fourierCoeff
  change
    ((2 * Real.pi * Complex.I * (n : ℂ)) •
      (∫ t : BurgersTorus, (fourier (-n) t : ℂ) *
        (u.value t : ℂ) ∂(AddCircle.haarAddCircle (T := (1 : ℝ))))).re =
      ∫ x : BurgersTorus,
        u.value x * (((2 * Real.pi * Complex.I * (n : ℂ)) *
          fourier (-n) x : ℂ).re)
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))
  rw [← MeasureTheory.integral_smul]
  simp only [smul_eq_mul]
  have hint :
      Integrable
        (fun t : BurgersTorus =>
          (2 * Real.pi * Complex.I * (n : ℂ)) *
            ((fourier (-n) t : ℂ) * (u.value t : ℂ)))
        (AddCircle.haarAddCircle (T := (1 : ℝ))) := by
    refine periodicH1_complexContinuous_integrable _ ?_
    exact continuous_const.mul
      ((fourier (-n)).continuous.mul
        (Complex.continuous_ofReal.comp u.value.continuous))
  calc
    (∫ t : BurgersTorus,
      (2 * Real.pi * Complex.I * (n : ℂ)) *
        ((fourier (-n) t : ℂ) * (u.value t : ℂ))
      ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))).re
        = ∫ x : BurgersTorus,
            (((2 * Real.pi * Complex.I * (n : ℂ)) *
              ((fourier (-n) x : ℂ) * (u.value x : ℂ))).re)
            ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
          exact (integral_re hint).symm
    _ = ∫ x : BurgersTorus,
          u.value x * (((2 * Real.pi * Complex.I * (n : ℂ)) *
            fourier (-n) x : ℂ).re)
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
          refine integral_congr_ae ?_
          filter_upwards with x
          simp [Complex.mul_re, Complex.mul_im, mul_comm, mul_left_comm, mul_assoc]

private theorem periodicH1_multiplierValueCoeff_im
    (u : PeriodicH1State)
    (n : ℤ) :
    ((2 * Real.pi * Complex.I * (n : ℂ)) *
      periodicH1_valueFourierCoeff u n).im =
      ∫ x : BurgersTorus,
        u.value x * (((2 * Real.pi * Complex.I * (n : ℂ)) *
          fourier (-n) x : ℂ).im)
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
  letI : Fact (0 < (1 : ℝ)) := ⟨by norm_num⟩
  unfold periodicH1_valueFourierCoeff
  unfold fourierCoeff
  change
    ((2 * Real.pi * Complex.I * (n : ℂ)) •
      (∫ t : BurgersTorus, (fourier (-n) t : ℂ) *
        (u.value t : ℂ) ∂(AddCircle.haarAddCircle (T := (1 : ℝ))))).im =
      ∫ x : BurgersTorus,
        u.value x * (((2 * Real.pi * Complex.I * (n : ℂ)) *
          fourier (-n) x : ℂ).im)
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))
  rw [← MeasureTheory.integral_smul]
  simp only [smul_eq_mul]
  have hint :
      Integrable
        (fun t : BurgersTorus =>
          (2 * Real.pi * Complex.I * (n : ℂ)) *
            ((fourier (-n) t : ℂ) * (u.value t : ℂ)))
        (AddCircle.haarAddCircle (T := (1 : ℝ))) := by
    refine periodicH1_complexContinuous_integrable _ ?_
    exact continuous_const.mul
      ((fourier (-n)).continuous.mul
        (Complex.continuous_ofReal.comp u.value.continuous))
  calc
    (∫ t : BurgersTorus,
      (2 * Real.pi * Complex.I * (n : ℂ)) *
        ((fourier (-n) t : ℂ) * (u.value t : ℂ))
      ∂(AddCircle.haarAddCircle (T := (1 : ℝ)))).im
        = ∫ x : BurgersTorus,
            (((2 * Real.pi * Complex.I * (n : ℂ)) *
              ((fourier (-n) x : ℂ) * (u.value x : ℂ))).im)
            ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
          exact (integral_im hint).symm
    _ = ∫ x : BurgersTorus,
          u.value x * (((2 * Real.pi * Complex.I * (n : ℂ)) *
            fourier (-n) x : ℂ).im)
        ∂(AddCircle.haarAddCircle (T := (1 : ℝ))) := by
          refine integral_congr_ae ?_
          filter_upwards with x
          simp [Complex.mul_re, Complex.mul_im, mul_comm, mul_left_comm, mul_assoc]

/-- Reusable Fourier-analysis package needed to prove the one-dimensional
periodic `H¹` Poincare/coercivity theorem on the ground-truth carrier.

The previous boundary exposed these as five independent axioms. Bundling them
records that they are one reusable Fourier theory package: Parseval for the
mean-zero value field, summability for value and derivative modes, a derivative
Parseval lower bound, and the weak-derivative Fourier multiplier identity. -/
structure PeriodicH1FourierTheory where
  value_parseval_meanZero :
    ∀ u : PeriodicH1State,
      PeriodicH1State.IsPeriodicH1 u →
        PeriodicH1State.mean u = 0 →
          PeriodicH1State.energy u =
            ∑' n : PeriodicH1NonzeroMode,
              periodicH1_valueNonzeroModeEnergy u n
  value_nonzeroModeEnergy_summable :
    ∀ u : PeriodicH1State,
      PeriodicH1State.IsPeriodicH1 u →
        PeriodicH1State.mean u = 0 →
          Summable (periodicH1_valueNonzeroModeEnergy u)
  derivative_nonzeroModeEnergy_summable :
    ∀ u : PeriodicH1State,
      PeriodicH1State.IsPeriodicH1 u →
        Summable (periodicH1_derivativeNonzeroModeEnergy u)
  derivative_parseval_lowerBound :
    ∀ u : PeriodicH1State,
      PeriodicH1State.IsPeriodicH1 u →
        (∑' n : PeriodicH1NonzeroMode,
          periodicH1_derivativeNonzeroModeEnergy u n) ≤
          PeriodicH1State.derivativeEnergy u
  weakDerivative_fourierCoeff :
    ∀ u : PeriodicH1State,
      PeriodicH1State.IsPeriodicH1 u →
        ∀ n : PeriodicH1NonzeroMode,
          periodicH1_derivativeFourierCoeff u n.1 =
            (2 * Real.pi * Complex.I * (n.1 : ℂ)) *
              periodicH1_valueFourierCoeff u n.1

/-- Reusable Fourier package for the periodic `H¹` carrier, discharged from
mathlib Fourier analysis plus the carrier's weak-derivative interface. -/
def periodicH1_fourierTheory_literature : PeriodicH1FourierTheory where
  value_parseval_meanZero := by
    intro u _hu hmean
    have hall := periodicH1_value_allModes_parseval u
    have hsplit := periodicH1_value_allModes_split u
    have hzero : ‖periodicH1_valueFourierCoeff u 0‖ ^ (2 : ℕ) = 0 := by
      rw [periodicH1_value_zeroCoeff u, hmean]
      simp
    calc
      PeriodicH1State.energy u
          = ∑' n : ℤ, ‖periodicH1_valueFourierCoeff u n‖ ^ (2 : ℕ) := hall.symm
      _ = ‖periodicH1_valueFourierCoeff u 0‖ ^ (2 : ℕ) +
            ∑' n : PeriodicH1NonzeroMode,
              periodicH1_valueNonzeroModeEnergy u n := hsplit
      _ = ∑' n : PeriodicH1NonzeroMode,
            periodicH1_valueNonzeroModeEnergy u n := by rw [hzero, zero_add]
  value_nonzeroModeEnergy_summable := by
    intro u _hu _hmean
    simpa [periodicH1_valueNonzeroModeEnergy]
      using (periodicH1_value_allModes_summable u).comp_injective
        (show Function.Injective (fun n : PeriodicH1NonzeroMode => (n : ℤ)) from
          Subtype.val_injective)
  derivative_nonzeroModeEnergy_summable := by
    intro u _hu
    simpa [periodicH1_derivativeNonzeroModeEnergy]
      using (periodicH1_derivative_allModes_summable u).comp_injective
        (show Function.Injective (fun n : PeriodicH1NonzeroMode => (n : ℤ)) from
          Subtype.val_injective)
  derivative_parseval_lowerBound := by
    intro u _hu
    have hall := periodicH1_derivative_allModes_parseval u
    have hsplit := periodicH1_derivative_allModes_split u
    have hzero_nonneg :
        0 ≤ ‖periodicH1_derivativeFourierCoeff u 0‖ ^ (2 : ℕ) := by
      exact sq_nonneg _
    calc
      (∑' n : PeriodicH1NonzeroMode,
        periodicH1_derivativeNonzeroModeEnergy u n)
          ≤ ‖periodicH1_derivativeFourierCoeff u 0‖ ^ (2 : ℕ) +
              ∑' n : PeriodicH1NonzeroMode,
                periodicH1_derivativeNonzeroModeEnergy u n := by
            nlinarith [hzero_nonneg]
      _ = ∑' n : ℤ, ‖periodicH1_derivativeFourierCoeff u n‖ ^ (2 : ℕ) := by
            exact hsplit.symm
      _ = PeriodicH1State.derivativeEnergy u := hall
  weakDerivative_fourierCoeff := by
    intro u hu n
    apply Complex.ext
    · rw [periodicH1_derivativeCoeff_re, periodicH1_multiplierValueCoeff_re]
      exact periodicH1_weakDerivative_fourierCoeff_realPart u hu n.1
    · rw [periodicH1_derivativeCoeff_im, periodicH1_multiplierValueCoeff_im]
      exact periodicH1_weakDerivative_fourierCoeff_imagPart u hu n.1

/-- Full all-mode value Fourier summability on the periodic `H¹` carrier.
The Poincare theorem only needs the mean-zero/nonzero projection, but the heat
coefficient backend needs the all-mode statement because the zero mode is
preserved by the heat flow. -/
theorem periodicH1_valueModeEnergy_summable_literature
    (u : PeriodicH1State)
    (_hu : PeriodicH1State.IsPeriodicH1 u) :
    Summable (periodicH1_valueModeEnergy u) := by
  simpa [periodicH1_valueModeEnergy]
    using periodicH1_value_allModes_summable u

/-- Full all-mode weak-derivative Fourier summability on the periodic `H¹`
carrier, exposed for the heat coefficient contraction theorem. -/
theorem periodicH1_derivativeModeEnergy_summable_literature
    (u : PeriodicH1State)
    (_hu : PeriodicH1State.IsPeriodicH1 u) :
    Summable (periodicH1_derivativeModeEnergy u) := by
  simpa [periodicH1_derivativeModeEnergy]
    using periodicH1_derivative_allModes_summable u

/-- Full Parseval identity for the value field. -/
theorem periodicH1_value_parseval_literature
    (u : PeriodicH1State)
    (_hu : PeriodicH1State.IsPeriodicH1 u) :
    (∑' n : ℤ, periodicH1_valueModeEnergy u n) =
      PeriodicH1State.energy u := by
  simpa [periodicH1_valueModeEnergy]
    using periodicH1_value_allModes_parseval u

/-- Full Parseval identity for the weak-derivative field. -/
theorem periodicH1_derivative_parseval_literature
    (u : PeriodicH1State)
    (_hu : PeriodicH1State.IsPeriodicH1 u) :
    (∑' n : ℤ, periodicH1_derivativeModeEnergy u n) =
      PeriodicH1State.derivativeEnergy u := by
  simpa [periodicH1_derivativeModeEnergy]
    using periodicH1_derivative_allModes_parseval u

/-- Fourier Parseval boundary for the value field after the zero mode is
removed by the mean-zero hypothesis. This is now a projection from the bundled
Fourier theory, not an independent axiom. -/
theorem periodicH1_value_parseval_meanZero_literature
    (u : PeriodicH1State)
    (hu : PeriodicH1State.IsPeriodicH1 u)
    (hmean : PeriodicH1State.mean u = 0) :
    PeriodicH1State.energy u =
      ∑' n : PeriodicH1NonzeroMode,
        periodicH1_valueNonzeroModeEnergy u n :=
  periodicH1_fourierTheory_literature.value_parseval_meanZero u hu hmean

/-- Summability of the nonzero value-mode energy series used by Parseval. This
is now a projection from the bundled Fourier theory. -/
theorem periodicH1_value_nonzeroModeEnergy_summable_literature
    (u : PeriodicH1State)
    (hu : PeriodicH1State.IsPeriodicH1 u)
    (hmean : PeriodicH1State.mean u = 0) :
    Summable (periodicH1_valueNonzeroModeEnergy u) :=
  periodicH1_fourierTheory_literature.value_nonzeroModeEnergy_summable u hu hmean

/-- Summability of the derivative-mode energy series used by Parseval. This is
now a projection from the bundled Fourier theory. -/
theorem periodicH1_derivative_nonzeroModeEnergy_summable_literature
    (u : PeriodicH1State)
    (hu : PeriodicH1State.IsPeriodicH1 u) :
    Summable (periodicH1_derivativeNonzeroModeEnergy u) :=
  periodicH1_fourierTheory_literature.derivative_nonzeroModeEnergy_summable u hu

/-- Parseval/lower-bound boundary for the weak derivative. Only the nonzero
modes are needed for Poincare, so this is stated as a lower bound rather than a
full derivative Parseval equality. This is now a projection from the bundled
Fourier theory. -/
theorem periodicH1_derivative_parseval_lowerBound_literature
    (u : PeriodicH1State)
    (hu : PeriodicH1State.IsPeriodicH1 u) :
    (∑' n : PeriodicH1NonzeroMode,
      periodicH1_derivativeNonzeroModeEnergy u n) ≤
      PeriodicH1State.derivativeEnergy u :=
  periodicH1_fourierTheory_literature.derivative_parseval_lowerBound u hu

/-- Fourier coefficient identity for the corrected weak-derivative interface.
This is the analytic bridge from the integration-by-parts definition to the
spectral derivative multiplier. This is now a projection from the bundled
Fourier theory. -/
theorem periodicH1_weakDerivative_fourierCoeff_literature
    (u : PeriodicH1State)
    (hu : PeriodicH1State.IsPeriodicH1 u)
    (n : PeriodicH1NonzeroMode) :
    periodicH1_derivativeFourierCoeff u n.1 =
      (2 * Real.pi * Complex.I * (n.1 : ℂ)) *
        periodicH1_valueFourierCoeff u n.1 :=
  periodicH1_fourierTheory_literature.weakDerivative_fourierCoeff u hu n

theorem periodicH1_firstFrequencyGap
    (n : PeriodicH1NonzeroMode) :
    (1 : ℝ) ≤ ‖(2 * Real.pi * Complex.I * (n.1 : ℂ))‖ ^ (2 : ℕ) := by
  rw [← Complex.normSq_eq_norm_sq]
  rw [map_mul, map_mul, map_mul]
  simp [Complex.normSq_ofReal, Complex.normSq_intCast]
  have hpi : (2 : ℝ) ≤ Real.pi := Real.two_le_pi
  have hn1 : (1 : ℝ) ≤ |(n.1 : ℝ)| := by
    rw [← Int.cast_abs]
    exact_mod_cast Int.one_le_abs n.2
  have hn_sq : (1 : ℝ) ≤ (n.1 : ℝ) * (n.1 : ℝ) := by
    have hs : (1 : ℝ) ^ (2 : ℕ) ≤ |(n.1 : ℝ)| ^ (2 : ℕ) := by
      exact sq_le_sq.mpr (by simpa using hn1)
    rw [_root_.sq_abs (n.1 : ℝ)] at hs
    simpa [pow_two] using hs
  nlinarith

theorem periodicH1_valueModeEnergy_le_derivativeModeEnergy
    (u : PeriodicH1State)
    (hu : PeriodicH1State.IsPeriodicH1 u)
    (n : PeriodicH1NonzeroMode) :
    periodicH1_valueNonzeroModeEnergy u n ≤
      periodicH1_derivativeNonzeroModeEnergy u n := by
  dsimp [periodicH1_valueNonzeroModeEnergy, periodicH1_derivativeNonzeroModeEnergy]
  have hgap := periodicH1_firstFrequencyGap n
  have hcoeff := periodicH1_weakDerivative_fourierCoeff_literature u hu n
  rw [hcoeff, Complex.abs.map_mul, mul_pow]
  have hgapAbs :
      (1 : ℝ) ≤ Complex.abs (2 * Real.pi * Complex.I * (n.1 : ℂ)) ^ (2 : ℕ) := by
    simpa using hgap
  nlinarith [sq_nonneg (Complex.abs (periodicH1_valueFourierCoeff u n.1))]

/-- Spectral-gap/Poincare theorem assembled in Lean from the explicit Fourier
component facts above. -/
theorem periodicH1_poincare_spectralGap_literature
    (u : PeriodicH1State)
    (hu : PeriodicH1State.IsPeriodicH1 u)
    (hmean : PeriodicH1State.mean u = 0) :
    PeriodicH1State.energy u ≤ PeriodicH1State.derivativeEnergy u := by
  rw [periodicH1_value_parseval_meanZero_literature u hu hmean]
  exact (tsum_le_tsum
    (fun n => periodicH1_valueModeEnergy_le_derivativeModeEnergy u hu n)
    (periodicH1_value_nonzeroModeEnergy_summable_literature u hu hmean)
    (periodicH1_derivative_nonzeroModeEnergy_summable_literature u hu)).trans
      (periodicH1_derivative_parseval_lowerBound_literature u hu)

/-- Framework wrapper exposing the local Poincare/coercivity predicate used by
the hypostructure certificate layer. The only analytic boundary is the narrow
spectral-gap theorem above. -/
theorem periodicH1_poincare_meanZero_literature
    (u : PeriodicH1State)
    (hu : PeriodicH1State.IsPeriodicH1 u) :
    LocalPoincareCoercivity u := by
  intro hmean
  exact periodicH1_poincare_spectralGap_literature u hu hmean

end

end Hypostructure.Literature.Analysis.PeriodicPoincare1D
