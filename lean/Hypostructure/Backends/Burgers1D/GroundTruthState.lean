import Hypostructure.Backends.Burgers1D.Parameters
import Hypostructure.Backends.Burgers1D.Torus

import Mathlib.Analysis.Normed.Group.Bounded
import Mathlib.MeasureTheory.Function.LpSeminorm.Basic

namespace Hypostructure.Backends.Burgers1D

open MeasureTheory

open scoped ENNReal

noncomputable section

/-- Smooth periodic test functions, represented at the level needed for the weak
derivative identity on the one-dimensional torus. The field
`integral_deriv_zero` is the periodic boundary cancellation used by constants. -/
structure SmoothPeriodicTestFunction where
  value : ContinuousMap BurgersTorus ℝ
  deriv : ContinuousMap BurgersTorus ℝ
  integral_deriv_zero : ∫ x : BurgersTorus, deriv x = 0

namespace SmoothPeriodicTestFunction

theorem integrable_value (φ : SmoothPeriodicTestFunction) :
    Integrable (fun x : BurgersTorus => φ.value x) := by
  rw [← integrableOn_univ]
  exact φ.value.continuous.continuousOn.integrableOn_compact isCompact_univ

theorem integrable_deriv (φ : SmoothPeriodicTestFunction) :
    Integrable (fun x : BurgersTorus => φ.deriv x) := by
  rw [← integrableOn_univ]
  exact φ.deriv.continuous.continuousOn.integrableOn_compact isCompact_univ

end SmoothPeriodicTestFunction

theorem continuousMap_integrable (f : ContinuousMap BurgersTorus ℝ) :
    Integrable (fun x : BurgersTorus => f x) := by
  rw [← integrableOn_univ]
  exact f.continuous.continuousOn.integrableOn_compact isCompact_univ

theorem continuousMap_mul_integrable
    (f g : ContinuousMap BurgersTorus ℝ) :
    Integrable (fun x : BurgersTorus => f x * g x) := by
  rw [← integrableOn_univ]
  exact (f.continuous.mul g.continuous).continuousOn.integrableOn_compact isCompact_univ

theorem continuousMap_memℒp_two
    (f : ContinuousMap BurgersTorus ℝ) :
    Memℒp (fun x : BurgersTorus => f x) (2 : ℝ≥0∞) volume := by
  rcases isCompact_univ.exists_bound_of_continuousOn
      (s := (Set.univ : Set BurgersTorus)) f.continuous.continuousOn with
    ⟨C, hC⟩
  exact Memℒp.of_bound
    (Continuous.aestronglyMeasurable f.continuous)
    C
    (Filter.Eventually.of_forall fun x => by
      simpa using hC x trivial)

/-- Weak derivative on the torus, encoded by integration by parts against
periodic test functions. -/
def IsWeakDerivativeOnTorus
    (u du : ContinuousMap BurgersTorus ℝ) : Prop :=
  ∀ φ : SmoothPeriodicTestFunction,
    ∫ x : BurgersTorus, u x * φ.deriv x =
      -∫ x : BurgersTorus, du x * φ.value x

/-- The ground-truth periodic `H¹` carrier for the Burgers backend. It is a
periodic value, a periodic weak derivative, L² witnesses for both, and the
weak-derivative identity tying the two fields together. -/
structure PeriodicH1State where
  value : ContinuousMap BurgersTorus ℝ
  weakDeriv : ContinuousMap BurgersTorus ℝ
  value_memL2 : Memℒp (fun x : BurgersTorus => value x) (2 : ℝ≥0∞) volume
  deriv_memL2 : Memℒp (fun x : BurgersTorus => weakDeriv x) (2 : ℝ≥0∞) volume
  weakDeriv_spec : IsWeakDerivativeOnTorus value weakDeriv

namespace PeriodicH1State

@[ext]
theorem ext
    {u v : PeriodicH1State}
    (hvalue : u.value = v.value)
    (hderiv : u.weakDeriv = v.weakDeriv) :
    u = v := by
  cases u
  cases v
  cases hvalue
  cases hderiv
  simp

def IsPeriodicH1 (u : PeriodicH1State) : Prop :=
  Memℒp (fun x : BurgersTorus => u.value x) (2 : ℝ≥0∞) volume ∧
    Memℒp (fun x : BurgersTorus => u.weakDeriv x) (2 : ℝ≥0∞) volume ∧
    IsWeakDerivativeOnTorus u.value u.weakDeriv

theorem isPeriodicH1 (u : PeriodicH1State) : IsPeriodicH1 u := by
  exact ⟨u.value_memL2, u.deriv_memL2, u.weakDeriv_spec⟩

theorem value_integrable (u : PeriodicH1State) :
    Integrable (fun x : BurgersTorus => u.value x) :=
  continuousMap_integrable u.value

theorem weakDeriv_integrable (u : PeriodicH1State) :
    Integrable (fun x : BurgersTorus => u.weakDeriv x) :=
  continuousMap_integrable u.weakDeriv

noncomputable def mean (u : PeriodicH1State) : ℝ :=
  ∫ x : BurgersTorus, u.value x

noncomputable def energy (u : PeriodicH1State) : ℝ :=
  ∫ x : BurgersTorus, (u.value x) ^ (2 : ℕ)

noncomputable def dissipation (nu : BurgersParameters) (u : PeriodicH1State) : ℝ :=
  nu.viscosity * ∫ x : BurgersTorus, (u.weakDeriv x) ^ (2 : ℕ)

noncomputable def constantProfile (m : ℝ) : ContinuousMap BurgersTorus ℝ where
  toFun := fun _ => m
  continuous_toFun := continuous_const

noncomputable def zeroDerivative : ContinuousMap BurgersTorus ℝ := 0

theorem zero_isWeakDerivative_constant (m : ℝ) :
    IsWeakDerivativeOnTorus (constantProfile m) zeroDerivative := by
  intro φ
  calc
    ∫ x : BurgersTorus, constantProfile m x * φ.deriv x
        = ∫ x : BurgersTorus, m * φ.deriv x := by rfl
    _ = m * ∫ x : BurgersTorus, φ.deriv x := by
          rw [integral_mul_left]
    _ = -∫ x : BurgersTorus, zeroDerivative x * φ.value x := by
          simp [φ.integral_deriv_zero, zeroDerivative]

noncomputable def constantState (m : ℝ) : PeriodicH1State where
  value := constantProfile m
  weakDeriv := zeroDerivative
  value_memL2 := continuousMap_memℒp_two (constantProfile m)
  deriv_memL2 := continuousMap_memℒp_two zeroDerivative
  weakDeriv_spec := zero_isWeakDerivative_constant m

theorem mean_constantState (m : ℝ) :
    mean (constantState m) = m := by
  simp [mean, constantState, constantProfile, AddCircle.measure_univ]

theorem add_spec (u v : PeriodicH1State) :
    IsWeakDerivativeOnTorus (u.value + v.value) (u.weakDeriv + v.weakDeriv) := by
  intro φ
  have hu := u.weakDeriv_spec φ
  have hv := v.weakDeriv_spec φ
  have hleft :
      ∫ x : BurgersTorus, (u.value + v.value) x * φ.deriv x =
        (∫ x : BurgersTorus, u.value x * φ.deriv x) +
          ∫ x : BurgersTorus, v.value x * φ.deriv x := by
    calc
      ∫ x : BurgersTorus, (u.value + v.value) x * φ.deriv x
          = ∫ x : BurgersTorus,
              (u.value x * φ.deriv x + v.value x * φ.deriv x) := by
                refine integral_congr_ae ?_
                filter_upwards with x
                simp [mul_add, add_mul, mul_comm, mul_left_comm, mul_assoc]
      _ = (∫ x : BurgersTorus, u.value x * φ.deriv x) +
            ∫ x : BurgersTorus, v.value x * φ.deriv x := by
              rw [integral_add]
              · exact continuousMap_mul_integrable u.value φ.deriv
              · exact continuousMap_mul_integrable v.value φ.deriv
  have hright :
      ∫ x : BurgersTorus, (u.weakDeriv + v.weakDeriv) x * φ.value x =
        (∫ x : BurgersTorus, u.weakDeriv x * φ.value x) +
          ∫ x : BurgersTorus, v.weakDeriv x * φ.value x := by
    calc
      ∫ x : BurgersTorus, (u.weakDeriv + v.weakDeriv) x * φ.value x
          = ∫ x : BurgersTorus,
              (u.weakDeriv x * φ.value x + v.weakDeriv x * φ.value x) := by
                refine integral_congr_ae ?_
                filter_upwards with x
                simp [mul_add, add_mul, mul_comm, mul_left_comm, mul_assoc]
      _ = (∫ x : BurgersTorus, u.weakDeriv x * φ.value x) +
            ∫ x : BurgersTorus, v.weakDeriv x * φ.value x := by
              rw [integral_add]
              · exact continuousMap_mul_integrable u.weakDeriv φ.value
              · exact continuousMap_mul_integrable v.weakDeriv φ.value
  linarith

noncomputable def add (u v : PeriodicH1State) : PeriodicH1State where
  value := u.value + v.value
  weakDeriv := u.weakDeriv + v.weakDeriv
  value_memL2 := continuousMap_memℒp_two (u.value + v.value)
  deriv_memL2 := continuousMap_memℒp_two (u.weakDeriv + v.weakDeriv)
  weakDeriv_spec := add_spec u v

theorem smul_spec (a : ℝ) (u : PeriodicH1State) :
    IsWeakDerivativeOnTorus (a • u.value) (a • u.weakDeriv) := by
  intro φ
  have hu := u.weakDeriv_spec φ
  calc
    ∫ x : BurgersTorus, (a • u.value) x * φ.deriv x
        = ∫ x : BurgersTorus, a * (u.value x * φ.deriv x) := by
              refine integral_congr_ae ?_
              filter_upwards with x
              simp [smul_eq_mul, mul_comm, mul_left_comm, mul_assoc]
    _ = a * ∫ x : BurgersTorus, u.value x * φ.deriv x := by
          rw [integral_mul_left]
    _ = -∫ x : BurgersTorus, (a • u.weakDeriv) x * φ.value x := by
          calc
            a * (∫ x : BurgersTorus, u.value x * φ.deriv x)
                = a * (-(∫ x : BurgersTorus, u.weakDeriv x * φ.value x)) := by rw [hu]
            _ = -(∫ x : BurgersTorus, a * (u.weakDeriv x * φ.value x)) := by
                  rw [integral_mul_left]
                  ring
            _ = -∫ x : BurgersTorus, (a • u.weakDeriv) x * φ.value x := by
                  congr 1
                  refine integral_congr_ae ?_
                  filter_upwards with x
                  simp [smul_eq_mul, mul_comm, mul_left_comm, mul_assoc]

noncomputable def smul (a : ℝ) (u : PeriodicH1State) : PeriodicH1State where
  value := a • u.value
  weakDeriv := a • u.weakDeriv
  value_memL2 := continuousMap_memℒp_two (a • u.value)
  deriv_memL2 := continuousMap_memℒp_two (a • u.weakDeriv)
  weakDeriv_spec := smul_spec a u

noncomputable def neg (u : PeriodicH1State) : PeriodicH1State :=
  smul (-1) u

noncomputable def sub (u v : PeriodicH1State) : PeriodicH1State :=
  add u (neg v)

theorem mean_add (u v : PeriodicH1State) :
    mean (add u v) = mean u + mean v := by
  simpa [mean, add] using
    integral_add (continuousMap_integrable u.value) (continuousMap_integrable v.value)

theorem mean_smul (a : ℝ) (u : PeriodicH1State) :
    mean (smul a u) = a * mean u := by
  simpa [mean, smul, smul_eq_mul] using
    integral_mul_left a (fun x : BurgersTorus => u.value x)

theorem mean_neg (u : PeriodicH1State) :
    mean (neg u) = -mean u := by
  simpa [neg] using mean_smul (-1) u

theorem mean_sub (u v : PeriodicH1State) :
    mean (sub u v) = mean u - mean v := by
  rw [sub, mean_add, mean_neg]
  ring

noncomputable def meanZeroPart (u : PeriodicH1State) : PeriodicH1State :=
  sub u (constantState (mean u))

theorem mean_meanZeroPart (u : PeriodicH1State) :
    mean (meanZeroPart u) = 0 := by
  rw [meanZeroPart, mean_sub, mean_constantState]
  ring

theorem decompose_mean_zero_plus_constant (u : PeriodicH1State) :
    add (meanZeroPart u) (constantState (mean u)) = u := by
  apply PeriodicH1State.ext
  · ext x
    simp [meanZeroPart, sub, add, neg, smul, constantState, constantProfile]
  · ext x
    simp [meanZeroPart, sub, add, neg, smul, constantState, zeroDerivative]

end PeriodicH1State

end

end Hypostructure.Backends.Burgers1D
