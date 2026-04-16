import Hypostructure.Backends.Burgers1D.Basic

import Mathlib.MeasureTheory.Integral.FundThmCalculus
import Mathlib.MeasureTheory.Integral.Periodic
import Mathlib.Topology.ContinuousMap.Bounded
import Mathlib.Topology.MetricSpace.Equicontinuity

open scoped Topology
open Set MeasureTheory intervalIntegral

namespace Hypostructure.Backends.Burgers1D

theorem burgers_diffusion_dominated_scaling_exact : (1 : ℤ) < 3 := by
  decide

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

structure MeanZeroPeriodicC1 where
  f : ℝ → ℝ
  periodic : Function.Periodic f 1
  f_cont : Continuous f
  deriv : ℝ → ℝ
  deriv_cont : Continuous deriv
  hasDerivAt : ∀ x : ℝ, HasDerivAt f (deriv x) x
  mean_zero : ∫ x in (0 : ℝ)..1, f x = 0

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
    filter_upwards with x hx
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

theorem MeanZeroPeriodicC1.deriv_integrable
    (p : MeanZeroPeriodicC1) :
    IntervalIntegrable p.deriv volume 0 1 := by
  simpa using p.deriv_cont.intervalIntegrable (a := (0 : ℝ)) (b := 1)

theorem MeanZeroPeriodicC1.abs_deriv_integrable
    (p : MeanZeroPeriodicC1) :
    IntervalIntegrable (fun x => |p.deriv x|) volume 0 1 := by
  simpa using p.deriv_cont.norm.intervalIntegrable (a := (0 : ℝ)) (b := 1)

theorem MeanZeroPeriodicC1.exists_zero
    (p : MeanZeroPeriodicC1) :
    ∃ c ∈ Set.Icc (0 : ℝ) 1, p.f c = 0 := by
  obtain ⟨xmin, hxmin_mem, hxmin⟩ :=
    isCompact_Icc.exists_isMinOn (nonempty_Icc.2 zero_le_one) p.f_cont.continuousOn
  obtain ⟨xmax, hxmax_mem, hxmax⟩ :=
    isCompact_Icc.exists_isMaxOn (nonempty_Icc.2 zero_le_one) p.f_cont.continuousOn
  have hmin_nonpos : p.f xmin ≤ 0 := by
    by_contra hpos
    have hpositive : ∀ x ∈ Set.Icc (0 : ℝ) 1, 0 < p.f x := by
      intro x hx
      exact lt_of_lt_of_le (lt_of_not_ge hpos) (hxmin hx)
    have hpos_int :
        0 < ∫ x in (0 : ℝ)..1, p.f x := by
      exact intervalIntegral.intervalIntegral_pos_of_pos_on
        (p.f_cont.intervalIntegrable (a := (0 : ℝ)) (b := 1))
        (fun x hx => hpositive x ⟨le_of_lt hx.1, le_of_lt hx.2⟩)
        zero_lt_one
    linarith [p.mean_zero, hpos_int]
  have hmax_nonneg : 0 ≤ p.f xmax := by
    by_contra hneg
    have hneg' : p.f xmax < 0 := lt_of_not_ge hneg
    have hpositive_neg : ∀ x ∈ Set.Icc (0 : ℝ) 1, 0 < -p.f x := by
      intro x hx
      exact neg_pos.2 <| lt_of_le_of_lt (hxmax hx) hneg'
    have hpos_int :
        0 < ∫ x in (0 : ℝ)..1, -p.f x := by
      exact intervalIntegral.intervalIntegral_pos_of_pos_on
        ((p.f_cont.neg).intervalIntegrable (a := (0 : ℝ)) (b := 1))
        (fun x hx => hpositive_neg x ⟨le_of_lt hx.1, le_of_lt hx.2⟩)
        zero_lt_one
    have : 0 < -∫ x in (0 : ℝ)..1, p.f x := by
      simpa [intervalIntegral.integral_neg] using hpos_int
    linarith [p.mean_zero, this]
  have hzero_mem : (0 : ℝ) ∈ Set.uIcc (p.f xmin) (p.f xmax) := by
    simpa [Set.mem_uIcc] using Or.inl (And.intro hmin_nonpos hmax_nonneg)
  have himage :=
    intermediate_value_uIcc (a := xmin) (b := xmax) (f := p.f) p.f_cont.continuousOn hzero_mem
  rcases himage with ⟨c, hc, hc0⟩
  have hc01 : c ∈ Set.Icc (0 : ℝ) 1 := by
    exact Set.uIcc_subset_Icc hxmin_mem hxmax_mem hc
  exact ⟨c, hc01, hc0⟩

theorem MeanZeroPeriodicC1.abs_bound
    (p : MeanZeroPeriodicC1)
    {x : ℝ}
    (hx : x ∈ Set.Icc (0 : ℝ) 1) :
    |p.f x| ≤ ∫ y in (0 : ℝ)..1, |p.deriv y| := by
  obtain ⟨c, hc01, hc0⟩ := p.exists_zero
  have hFTC :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt
      (a := c) (b := x)
      (fun y _hy => p.hasDerivAt y)
      (p.deriv_cont.intervalIntegrable (a := c) (b := x))
  have hfx : p.f x = ∫ y in c..x, p.deriv y := by
    simpa [hc0] using hFTC.symm
  have hc01u : c ∈ Set.uIcc (0 : ℝ) 1 := by
    simpa [Set.uIcc, min_eq_left zero_le_one, max_eq_right zero_le_one] using hc01
  have hxu : x ∈ Set.uIcc (0 : ℝ) 1 := by
    simpa [Set.uIcc, min_eq_left zero_le_one, max_eq_right zero_le_one] using hx
  have hsub_uIoc : Set.uIoc c x ⊆ Set.uIoc (0 : ℝ) 1 :=
    Set.uIoc_subset_uIoc_of_uIcc_subset_uIcc <| Set.uIcc_subset_uIcc hc01u hxu
  let Icx : ℝ := ∫ y in c..x, |p.deriv y|
  let I01 : ℝ := ∫ y in (0 : ℝ)..1, |p.deriv y|
  have hnorm :
      |∫ y in c..x, p.deriv y| ≤ |Icx| := by
    simpa [Real.norm_eq_abs] using
      (intervalIntegral.norm_integral_le_abs_integral_norm (f := p.deriv) (a := c) (b := x))
  have hmono :
      |Icx| ≤ |I01| := by
    exact intervalIntegral.abs_integral_mono_interval
      (μ := volume) (f := fun y => |p.deriv y|) hsub_uIoc
      (Filter.Eventually.of_forall fun y => abs_nonneg _)
      p.abs_deriv_integrable
  have hnonneg :
      0 ≤ I01 := by
    exact intervalIntegral.integral_nonneg zero_le_one fun _ _ => abs_nonneg _
  calc
    |p.f x| = |∫ y in c..x, p.deriv y| := by simpa [hfx]
    _ ≤ |Icx| := hnorm
    _ ≤ |I01| := hmono
    _ = I01 := abs_of_nonneg hnonneg

theorem MeanZeroPeriodicC1.poincare_coercive_l1
    (p : MeanZeroPeriodicC1) :
    ∫ x in (0 : ℝ)..1, (p.f x) ^ 2 ≤
      (∫ y in (0 : ℝ)..1, |p.deriv y|) ^ 2 := by
  have hpoint :
      ∀ x ∈ Set.Icc (0 : ℝ) 1,
        (p.f x) ^ 2 ≤ (∫ y in (0 : ℝ)..1, |p.deriv y|) ^ 2 := by
    intro x hx
    have h := MeanZeroPeriodicC1.abs_bound p hx
    let I : ℝ := ∫ y in (0 : ℝ)..1, |p.deriv y|
    have habs : |p.f x| ≤ |I| := by
      have hnonneg :
          0 ≤ I := by
        exact intervalIntegral.integral_nonneg zero_le_one fun _ _ => abs_nonneg _
      simpa [abs_of_nonneg hnonneg] using h
    have hsq := mul_self_le_mul_self (abs_nonneg _) habs
    simpa [sq, abs_mul_abs_self] using hsq
  have hfi :
      IntervalIntegrable (fun x => (p.f x) ^ 2) volume 0 1 := by
    simpa using (p.f_cont.pow 2).intervalIntegrable (a := (0 : ℝ)) (b := 1)
  have hconst :
      IntervalIntegrable (fun _x : ℝ => (∫ y in (0 : ℝ)..1, |p.deriv y|) ^ 2) volume 0 1 := by
    simpa using
      (continuous_const : Continuous (fun _x : ℝ => (∫ y in (0 : ℝ)..1, |p.deriv y|) ^ 2))
        .intervalIntegrable (a := (0 : ℝ)) (b := 1)
  calc
    ∫ x in (0 : ℝ)..1, (p.f x) ^ 2
        ≤ ∫ x in (0 : ℝ)..1, (∫ y in (0 : ℝ)..1, |p.deriv y|) ^ 2 := by
            exact intervalIntegral.integral_mono_on
              (a := (0 : ℝ)) (b := 1) (μ := volume) zero_le_one hfi hconst hpoint
    _ = (∫ y in (0 : ℝ)..1, |p.deriv y|) ^ 2 := by
          simp

abbrev UnitInterval := ↥(Set.Icc (0 : ℝ) 1)

abbrev UnitIntervalMap := BoundedContinuousFunction UnitInterval ℝ

def boundedLipschitzFamily (R : ℝ) (L : NNReal) : Set UnitIntervalMap :=
  { f | (∀ x, |f x| ≤ R) ∧ LipschitzWith L f }

theorem boundedLipschitzFamily_closure_isCompact
    {R : ℝ}
    (hR : 0 ≤ R)
    (L : NNReal) :
    IsCompact (closure (boundedLipschitzFamily R L)) := by
  let s : Set ℝ := Set.Icc (-R) R
  have hs : IsCompact s := isCompact_Icc
  refine BoundedContinuousFunction.arzela_ascoli s hs _ ?_ ?_
  · intro f x hf
    exact (abs_le.mp (hf.1 x))
  · refine Metric.equicontinuous_of_continuity_modulus (fun t : ℝ => (L : ℝ) * t) ?_
      ((↑) : ↥(boundedLipschitzFamily R L) → UnitInterval → ℝ) ?_
    · simpa using (continuous_const.mul continuous_id).tendsto (0 : ℝ)
    · intro x y f
      simpa [UnitIntervalMap] using f.2.2.dist_le_mul x y

end Hypostructure.Backends.Burgers1D
