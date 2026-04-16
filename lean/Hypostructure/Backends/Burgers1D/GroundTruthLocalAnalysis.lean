import Hypostructure.Backends.Burgers1D.Parameters

import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.MeasureTheory.Integral.FundThmCalculus
import Mathlib.MeasureTheory.Integral.Periodic

open scoped Topology
open Set MeasureTheory intervalIntegral

namespace Hypostructure.Backends.Burgers1D

theorem periodic_hasDerivAt_eq
    {f f' : ℝ → ℝ}
    (hper : Function.Periodic f 1)
    (hderiv : ∀ x : ℝ, HasDerivAt f (f' x) x) :
    ∀ x : ℝ, f' (x + 1) = f' x := by
  intro x
  have hshift : HasDerivAt (fun y : ℝ => f (y + 1)) (f' (x + 1)) x :=
    by simpa [one_mul] using (hderiv (x + 1)).comp x <| hasDerivAt_id x |>.add_const 1
  have hsame : HasDerivAt (fun y : ℝ => f (y + 1)) (f' x) x := by
    have hfun : (fun y : ℝ => f (y + 1)) = f := by
      funext y
      simpa [Function.Periodic] using hper y
    simpa [hfun] using hderiv x
  exact (hsame.unique hshift).symm

theorem periodic_deriv
    {f f' : ℝ → ℝ}
    (hper : Function.Periodic f 1)
    (hderiv : ∀ x : ℝ, HasDerivAt f (f' x) x) :
    Function.Periodic f' 1 := by
  intro x
  exact periodic_hasDerivAt_eq hper hderiv x

theorem periodic_intervalIntegral_deriv_eq_zero
    {f f' : ℝ → ℝ}
    (hper : Function.Periodic f 1)
    (hderiv : ∀ x ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt f (f' x) x)
    (hint : IntervalIntegrable f' volume 0 1) :
    ∫ x in (0 : ℝ)..1, f' x = 0 := by
  have hFTC :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt (a := (0 : ℝ)) (b := 1) hderiv hint
  have h01 : f 1 = f 0 := by simpa using hper 0
  simpa [h01] using hFTC

theorem periodic_intervalIntegral_sq_mul_deriv_eq_zero
    {f f' : ℝ → ℝ}
    (hper : Function.Periodic f 1)
    (hderiv : ∀ x : ℝ, HasDerivAt f (f' x) x)
    (hint : IntervalIntegrable (fun x => (f x) ^ 2 * f' x) volume 0 1) :
    ∫ x in (0 : ℝ)..1, (f x) ^ 2 * f' x = 0 := by
  let g : ℝ → ℝ := fun x => (f x) ^ 3 / 3
  have hderiv_g : ∀ x ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt g ((f x) ^ 2 * f' x) x := by
    intro x _hx
    dsimp [g]
    simpa [pow_two, pow_succ, mul_assoc, mul_left_comm, mul_comm, div_eq_mul_inv] using
      ((hderiv x).pow 3).div_const (3 : ℝ)
  have hFTC :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt (a := (0 : ℝ)) (b := 1) hderiv_g hint
  have hgper : Function.Periodic g 1 := by
    intro x
    dsimp [g]
    rw [hper x]
  have hg01 : g 1 = g 0 := by simpa using hgper 0
  simpa [hg01] using hFTC

/-- Smooth one-time Burgers snapshot used only for the local energy identity.
This is local analytic data, not a global solution package. -/
structure SmoothPeriodicSnapshot (ν : BurgersParameters) where
  v : ℝ → ℝ
  periodic : Function.Periodic v 1
  v_cont : Continuous v
  vx : ℝ → ℝ
  vxx : ℝ → ℝ
  vt : ℝ → ℝ
  vx_cont : Continuous vx
  vxx_cont : Continuous vxx
  vt_cont : Continuous vt
  hasDeriv_v : ∀ x : ℝ, HasDerivAt v (vx x) x
  hasDeriv_vx : ∀ x : ℝ, HasDerivAt vx (vxx x) x
  pde : ∀ x : ℝ, vt x + v x * vx x = ν.viscosity * vxx x

theorem SmoothPeriodicSnapshot.vx_periodic
    {ν : BurgersParameters}
    (s : SmoothPeriodicSnapshot ν) :
    Function.Periodic s.vx 1 :=
  periodic_deriv s.periodic s.hasDeriv_v

theorem SmoothPeriodicSnapshot.transport_integrable
    {ν : BurgersParameters}
    (s : SmoothPeriodicSnapshot ν) :
    IntervalIntegrable (fun x => (s.v x) ^ 2 * s.vx x) volume 0 1 := by
  simpa using ((s.v_cont.pow 2).mul s.vx_cont).intervalIntegrable (a := (0 : ℝ)) (b := 1)

theorem SmoothPeriodicSnapshot.diffusion_pairing_integrable
    {ν : BurgersParameters}
    (s : SmoothPeriodicSnapshot ν) :
    IntervalIntegrable (fun x => s.v x * s.vxx x) volume 0 1 := by
  simpa using (s.v_cont.mul s.vxx_cont).intervalIntegrable (a := (0 : ℝ)) (b := 1)

theorem SmoothPeriodicSnapshot.energy_pairing_integrable
    {ν : BurgersParameters}
    (s : SmoothPeriodicSnapshot ν) :
    IntervalIntegrable (fun x => s.v x * s.vt x) volume 0 1 := by
  simpa using (s.v_cont.mul s.vt_cont).intervalIntegrable (a := (0 : ℝ)) (b := 1)

theorem SmoothPeriodicSnapshot.vx_sq_integrable
    {ν : BurgersParameters}
    (s : SmoothPeriodicSnapshot ν) :
    IntervalIntegrable (fun x => (s.vx x) ^ 2) volume 0 1 := by
  simpa using (s.vx_cont.pow 2).intervalIntegrable (a := (0 : ℝ)) (b := 1)

theorem SmoothPeriodicSnapshot.transport_term_zero
    {ν : BurgersParameters}
    (s : SmoothPeriodicSnapshot ν) :
    ∫ x in (0 : ℝ)..1, (s.v x) ^ 2 * s.vx x = 0 :=
  periodic_intervalIntegral_sq_mul_deriv_eq_zero
    s.periodic s.hasDeriv_v s.transport_integrable

theorem SmoothPeriodicSnapshot.diffusion_integration_by_parts
    {ν : BurgersParameters}
    (s : SmoothPeriodicSnapshot ν) :
    ∫ x in (0 : ℝ)..1, s.v x * s.vxx x =
      - ∫ x in (0 : ℝ)..1, (s.vx x) ^ 2 := by
  let g : ℝ → ℝ := fun x => s.v x * s.vx x
  have hderiv_g :
      ∀ x ∈ Set.uIcc (0 : ℝ) 1,
        HasDerivAt g ((s.vx x) ^ 2 + s.v x * s.vxx x) x := by
    intro x _hx
    dsimp [g]
    simpa [pow_two, mul_assoc, mul_left_comm, mul_comm, add_comm, add_left_comm, add_assoc] using
      (s.hasDeriv_v x).mul (s.hasDeriv_vx x)
  have hgper : Function.Periodic g 1 := by
    intro x
    dsimp [g]
    rw [s.periodic x, s.vx_periodic x]
  have hint :
      IntervalIntegrable (fun x => (s.vx x) ^ 2 + s.v x * s.vxx x) volume 0 1 := by
    exact s.vx_sq_integrable.add s.diffusion_pairing_integrable
  have hzero :=
    periodic_intervalIntegral_deriv_eq_zero hgper hderiv_g hint
  have hadd :
      ∫ x in (0 : ℝ)..1, ((s.vx x) ^ 2 + s.v x * s.vxx x) =
        (∫ x in (0 : ℝ)..1, (s.vx x) ^ 2) + ∫ x in (0 : ℝ)..1, s.v x * s.vxx x := by
    rw [intervalIntegral.integral_add]
    exact s.vx_sq_integrable
    exact s.diffusion_pairing_integrable
  linarith [show
    (∫ x in (0 : ℝ)..1, ((s.vx x) ^ 2 + s.v x * s.vxx x)) = 0 from hzero, hadd]

theorem SmoothPeriodicSnapshot.energy_pairing_identity
    {ν : BurgersParameters}
    (s : SmoothPeriodicSnapshot ν) :
    ∫ x in (0 : ℝ)..1, s.v x * s.vt x =
      -ν.viscosity * ∫ x in (0 : ℝ)..1, (s.vx x) ^ 2 := by
  have hpde_int :
      ∫ x in (0 : ℝ)..1, (s.v x * s.vt x + (s.v x) ^ 2 * s.vx x) =
        ∫ x in (0 : ℝ)..1, ν.viscosity * (s.v x * s.vxx x) := by
    apply intervalIntegral.integral_congr_ae
    filter_upwards with x
    intro _hx
    have := congrArg (fun z => s.v x * z) (s.pde x)
    ring_nf at this ⊢
    exact this
  have hleft :
      ∫ x in (0 : ℝ)..1, (s.v x * s.vt x + (s.v x) ^ 2 * s.vx x) =
        (∫ x in (0 : ℝ)..1, s.v x * s.vt x) +
        ∫ x in (0 : ℝ)..1, (s.v x) ^ 2 * s.vx x := by
    rw [intervalIntegral.integral_add]
    exact s.energy_pairing_integrable
    exact s.transport_integrable
  have hright :
      ∫ x in (0 : ℝ)..1, ν.viscosity * (s.v x * s.vxx x) =
        ν.viscosity * ∫ x in (0 : ℝ)..1, s.v x * s.vxx x := by
    rw [intervalIntegral.integral_const_mul]
  have hmain :
      (∫ x in (0 : ℝ)..1, s.v x * s.vt x) +
        ∫ x in (0 : ℝ)..1, (s.v x) ^ 2 * s.vx x =
        ν.viscosity * ∫ x in (0 : ℝ)..1, s.v x * s.vxx x := by
    calc
      (∫ x in (0 : ℝ)..1, s.v x * s.vt x) + ∫ x in (0 : ℝ)..1, (s.v x) ^ 2 * s.vx x
          = ∫ x in (0 : ℝ)..1, (s.v x * s.vt x + (s.v x) ^ 2 * s.vx x) := by
              rw [← hleft]
      _ = ∫ x in (0 : ℝ)..1, ν.viscosity * (s.v x * s.vxx x) := hpde_int
      _ = ν.viscosity * ∫ x in (0 : ℝ)..1, s.v x * s.vxx x := hright
  rw [s.transport_term_zero, add_zero, s.diffusion_integration_by_parts] at hmain
  nlinarith [hmain]

end Hypostructure.Backends.Burgers1D
