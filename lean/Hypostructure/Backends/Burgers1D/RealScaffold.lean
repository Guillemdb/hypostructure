import Mathlib.Algebra.Periodic
import Hypostructure.Backends.Burgers1D.Final
import Hypostructure.Backends.Burgers1D.SmoothState

namespace Hypostructure.Backends.Burgers1D

namespace SmoothPeriodicState

noncomputable def descendedValue (u : SmoothPeriodicState) : BurgersProfile where
  toFun := u.periodic.lift
  continuous_toFun := by
    simpa [Function.Periodic.lift] using
      u.val_cont.quotient_lift (fun a b h => by
        have hq : ((a : BurgersTorus) : UnitAddCircle) = (b : BurgersTorus) :=
          Quotient.sound h
        calc
          u.val a = u.periodic.lift (a : BurgersTorus) := by
            simpa using (Function.Periodic.lift_coe (h := u.periodic) a).symm
          _ = u.periodic.lift (b : BurgersTorus) := by rw [hq]
          _ = u.val b := by
            simpa using Function.Periodic.lift_coe (h := u.periodic) b)

noncomputable def descendedDeriv (u : SmoothPeriodicState) : BurgersDerivative where
  toFun := (deriv_periodic u).lift
  continuous_toFun := by
    simpa [Function.Periodic.lift] using
      u.deriv_cont.quotient_lift (fun a b h => by
        have hq : ((a : BurgersTorus) : UnitAddCircle) = (b : BurgersTorus) :=
          Quotient.sound h
        calc
          u.deriv a = (deriv_periodic u).lift (a : BurgersTorus) := by
            simpa using (Function.Periodic.lift_coe (h := deriv_periodic u) a).symm
          _ = (deriv_periodic u).lift (b : BurgersTorus) := by rw [hq]
          _ = u.deriv b := by
            simpa using Function.Periodic.lift_coe (h := deriv_periodic u) b)

noncomputable def toBurgersState (u : SmoothPeriodicState) : BurgersState :=
  (descendedValue u, descendedDeriv u)

@[simp] theorem descendedValue_coe_apply (u : SmoothPeriodicState) (x : ℝ) :
    descendedValue u (x : BurgersTorus) = u.val x := by
  simpa [descendedValue] using Function.Periodic.lift_coe (h := u.periodic) x

@[simp] theorem descendedDeriv_coe_apply (u : SmoothPeriodicState) (x : ℝ) :
    descendedDeriv u (x : BurgersTorus) = u.deriv x := by
  simpa [descendedDeriv] using Function.Periodic.lift_coe (h := deriv_periodic u) x

@[simp] theorem toBurgersState_value_coe_apply (u : SmoothPeriodicState) (x : ℝ) :
    BurgersState.value (toBurgersState u) (x : BurgersTorus) = u.val x := by
  simpa [toBurgersState, BurgersState.value] using descendedValue_coe_apply u x

@[simp] theorem toBurgersState_deriv_coe_apply (u : SmoothPeriodicState) (x : ℝ) :
    BurgersState.deriv (toBurgersState u) (x : BurgersTorus) = u.deriv x := by
  simpa [toBurgersState, BurgersState.deriv] using descendedDeriv_coe_apply u x

theorem toBurgersState_constant (m : ℝ) :
    toBurgersState (constant m) = BurgersState.constantState m := by
  apply Prod.ext
  · ext x
    induction x using QuotientAddGroup.induction_on
    simp [toBurgersState, descendedValue, BurgersState.constantState,
      BurgersState.constantProfile, Function.Periodic.lift_coe, constant]
  · ext x
    induction x using QuotientAddGroup.induction_on
    simp [toBurgersState, descendedDeriv, BurgersState.constantState,
      BurgersState.zeroDerivative, Function.Periodic.lift_coe, constant]

theorem mean_toBurgersState (u : SmoothPeriodicState) :
    BurgersState.mean (toBurgersState u) = mean u := by
  calc
    BurgersState.mean (toBurgersState u)
      = ∫ x in (0 : ℝ)..((0 : ℝ) + 1), u.val x := by
          rw [BurgersState.mean, toBurgersState, BurgersState.value,
            ← UnitAddCircle.intervalIntegral_preimage (E := ℝ) (t := 0)
            (f := descendedValue u)]
          exact intervalIntegral.integral_congr_ae
            (μ := MeasureTheory.volume) (a := (0 : ℝ)) (b := ((0 : ℝ) + 1))
            (f := fun x : ℝ => descendedValue u x) (g := u.val)
            (Filter.Eventually.of_forall fun x _hx => by
              simpa using descendedValue_coe_apply u x)
    _ = mean u := by
          simp [mean]

theorem dissipation_toBurgersState (u : SmoothPeriodicState) :
    BurgersState.dissipation (toBurgersState u) = dissipation u := by
  calc
    BurgersState.dissipation (toBurgersState u)
      = ∫ x in (0 : ℝ)..((0 : ℝ) + 1), (u.deriv x) ^ (2 : ℕ) := by
          rw [BurgersState.dissipation, toBurgersState, BurgersState.deriv,
            ← UnitAddCircle.intervalIntegral_preimage (E := ℝ) (t := 0)
            (f := fun x : BurgersTorus => (descendedDeriv u x) ^ (2 : ℕ))]
          exact intervalIntegral.integral_congr_ae
            (μ := MeasureTheory.volume) (a := (0 : ℝ)) (b := ((0 : ℝ) + 1))
            (f := fun x : ℝ => (descendedDeriv u x) ^ (2 : ℕ))
            (g := fun x : ℝ => (u.deriv x) ^ (2 : ℕ))
            (Filter.Eventually.of_forall fun x _hx => by
              simpa using congrArg (fun y : ℝ => y ^ (2 : ℕ)) (descendedDeriv_coe_apply u x))
    _ = dissipation u := by
          simp [dissipation]

theorem meanZeroEnergy_toBurgersState (u : SmoothPeriodicState) :
    BurgersState.meanZeroEnergy (toBurgersState u) = meanZeroEnergy u := by
  calc
    BurgersState.meanZeroEnergy (toBurgersState u)
      = ∫ x in (0 : ℝ)..((0 : ℝ) + 1), (u.val x - mean u) ^ (2 : ℕ) := by
          rw [BurgersState.meanZeroEnergy, mean_toBurgersState,
            ← UnitAddCircle.intervalIntegral_preimage (E := ℝ) (t := 0)
              (f := fun x : BurgersTorus =>
                (BurgersState.value (toBurgersState u) x - mean u) ^ (2 : ℕ))]
          exact intervalIntegral.integral_congr_ae
            (μ := MeasureTheory.volume) (a := (0 : ℝ)) (b := ((0 : ℝ) + 1))
            (f := fun x : ℝ => (BurgersState.value (toBurgersState u) x - mean u) ^ (2 : ℕ))
            (g := fun x : ℝ => (u.val x - mean u) ^ (2 : ℕ))
            (Filter.Eventually.of_forall fun x _hx => by
              simp)
    _ = meanZeroEnergy u := by
          simp [meanZeroEnergy]

end SmoothPeriodicState

namespace BurgersState

def Realizable (u : BurgersState) : Prop :=
  ∃ s : SmoothPeriodicState, SmoothPeriodicState.toBurgersState s = u

theorem realizable_toBurgersState (u : SmoothPeriodicState) :
    Realizable (SmoothPeriodicState.toBurgersState u) :=
  ⟨u, rfl⟩

theorem realizable_constantState (m : ℝ) : Realizable (constantState m) := by
  exact ⟨SmoothPeriodicState.constant m, SmoothPeriodicState.toBurgersState_constant m⟩

end BurgersState

/-- Smooth periodic heat construction before extension to the theorem carrier. -/
class SmoothHeatConstruction (ν : BurgersParameters) where
  flow : ℝ → SmoothPeriodicState → SmoothPeriodicState
  flow_zero : ∀ u : SmoothPeriodicState, flow 0 u = u
  flow_add :
    ∀ s t : ℝ, ∀ u : SmoothPeriodicState, flow (s + t) u = flow s (flow t u)
  mean_preserving :
    ∀ t : ℝ, ∀ u : SmoothPeriodicState,
      SmoothPeriodicState.mean (flow t u) = SmoothPeriodicState.mean u
  fixes_constant_states :
    ∀ t : ℝ, ∀ m : ℝ, flow t (SmoothPeriodicState.constant m) = SmoothPeriodicState.constant m
  contracts_meanZeroEnergy :
    ∀ t : ℝ, 0 ≤ t → ∀ u : SmoothPeriodicState,
      SmoothPeriodicState.meanZeroEnergy (flow t u) ≤ SmoothPeriodicState.meanZeroEnergy u
  contracts_dissipation :
    ∀ t : ℝ, 0 ≤ t → ∀ u : SmoothPeriodicState,
      SmoothPeriodicState.dissipation (flow t u) ≤ SmoothPeriodicState.dissipation u
  descended_wellDefined :
    ∀ t : ℝ, ∀ {u v : SmoothPeriodicState},
      SmoothPeriodicState.toBurgersState u = SmoothPeriodicState.toBurgersState v →
        SmoothPeriodicState.toBurgersState (flow t u) =
          SmoothPeriodicState.toBurgersState (flow t v)
  injective_time_one_descended :
    ∀ {u v : SmoothPeriodicState},
      SmoothPeriodicState.toBurgersState (flow 1 u) =
        SmoothPeriodicState.toBurgersState (flow 1 v) →
      SmoothPeriodicState.toBurgersState u = SmoothPeriodicState.toBurgersState v

/-- Smooth Cole-Hopf construction before transporting back to the theorem carrier. -/
class SmoothColeHopfConstruction
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν] where
  transform : SmoothPeriodicState → SmoothPeriodicState
  inverse : SmoothPeriodicState → SmoothPeriodicState
  left_inv : ∀ u : SmoothPeriodicState, inverse (transform u) = u
  right_inv : ∀ w : SmoothPeriodicState, transform (inverse w) = w
  mean_preserving :
    ∀ u : SmoothPeriodicState,
      SmoothPeriodicState.mean (transform u) = SmoothPeriodicState.mean u
  preserves_constant_states :
    ∀ m : ℝ, transform (SmoothPeriodicState.constant m) = SmoothPeriodicState.constant m
  injective : Function.Injective transform
  intertwines_heat :
    ∀ t : ℝ, ∀ u : SmoothPeriodicState,
      transform (SmoothHeatConstruction.flow (ν := ν) t u) =
        SmoothHeatConstruction.flow (ν := ν) t (transform u)
  descended_wellDefined :
    ∀ {u v : SmoothPeriodicState},
      SmoothPeriodicState.toBurgersState u = SmoothPeriodicState.toBurgersState v →
        SmoothPeriodicState.toBurgersState (transform u) =
          SmoothPeriodicState.toBurgersState (transform v)
  inverse_descended_wellDefined :
    ∀ {u v : SmoothPeriodicState},
      SmoothPeriodicState.toBurgersState u = SmoothPeriodicState.toBurgersState v →
        SmoothPeriodicState.toBurgersState (inverse u) =
          SmoothPeriodicState.toBurgersState (inverse v)
  injective_descended :
    ∀ {u v : SmoothPeriodicState},
      SmoothPeriodicState.toBurgersState (transform u) =
        SmoothPeriodicState.toBurgersState (transform v) →
      SmoothPeriodicState.toBurgersState u = SmoothPeriodicState.toBurgersState v

def SmoothBridgeInvariantStatement
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] : Prop :=
  (∀ m : ℝ,
      SmoothColeHopfConstruction.transform (ν := ν) (SmoothPeriodicState.constant m) =
        SmoothPeriodicState.constant m) ∧
    Function.Injective (SmoothColeHopfConstruction.transform (ν := ν)) ∧
    (∀ t : ℝ, 0 ≤ t → ∀ u : SmoothPeriodicState,
      SmoothPeriodicState.meanZeroEnergy (SmoothHeatConstruction.flow (ν := ν) t u) ≤
        SmoothPeriodicState.meanZeroEnergy u) ∧
    (∀ t : ℝ, 0 ≤ t → ∀ u : SmoothPeriodicState,
      SmoothPeriodicState.dissipation (SmoothHeatConstruction.flow (ν := ν) t u) ≤
        SmoothPeriodicState.dissipation u) ∧
    (∀ t : ℝ, ∀ u : SmoothPeriodicState,
      SmoothColeHopfConstruction.transform (ν := ν)
          (SmoothHeatConstruction.flow (ν := ν) t u) =
        SmoothHeatConstruction.flow (ν := ν) t
          (SmoothColeHopfConstruction.transform (ν := ν) u))

theorem smooth_bridge_invariants
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    SmoothBridgeInvariantStatement ν := by
  exact ⟨
    SmoothColeHopfConstruction.preserves_constant_states (ν := ν),
    SmoothColeHopfConstruction.injective (ν := ν),
    SmoothHeatConstruction.contracts_meanZeroEnergy (ν := ν),
    SmoothHeatConstruction.contracts_dissipation (ν := ν),
    SmoothColeHopfConstruction.intertwines_heat (ν := ν)
  ⟩

noncomputable def burgersHeatFlow
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν] :
    ℝ → BurgersState → BurgersState := by
  classical
  exact fun t u =>
    if h : BurgersState.Realizable u then
      SmoothPeriodicState.toBurgersState
        (SmoothHeatConstruction.flow (ν := ν) t (Classical.choose h))
    else
      u

theorem burgersHeatFlow_of_realizable
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    (t : ℝ) {u : BurgersState}
    (h : BurgersState.Realizable u) :
    burgersHeatFlow ν t u =
      SmoothPeriodicState.toBurgersState
        (SmoothHeatConstruction.flow (ν := ν) t (Classical.choose h)) := by
  classical
  simp [burgersHeatFlow, h]

theorem burgersHeatFlow_of_not_realizable
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    (t : ℝ) {u : BurgersState}
    (h : ¬ BurgersState.Realizable u) :
    burgersHeatFlow ν t u = u := by
  classical
  simp [burgersHeatFlow, h]

theorem burgersHeatFlow_toBurgersState
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    (t : ℝ) (u : SmoothPeriodicState) :
    burgersHeatFlow ν t (SmoothPeriodicState.toBurgersState u) =
      SmoothPeriodicState.toBurgersState (SmoothHeatConstruction.flow (ν := ν) t u) := by
  let h : BurgersState.Realizable (SmoothPeriodicState.toBurgersState u) :=
    BurgersState.realizable_toBurgersState u
  rw [burgersHeatFlow_of_realizable ν t h]
  exact SmoothHeatConstruction.descended_wellDefined (ν := ν) t (Classical.choose_spec h)

theorem burgersHeatFlow_realizable
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    (t : ℝ) {u : BurgersState}
    (h : BurgersState.Realizable u) :
    BurgersState.Realizable (burgersHeatFlow ν t u) := by
  rcases h with ⟨su, rfl⟩
  simpa [burgersHeatFlow_toBurgersState] using
    BurgersState.realizable_toBurgersState (SmoothHeatConstruction.flow (ν := ν) t su)

theorem burgersHeatFlow_zero
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν] :
    ∀ u : BurgersState, burgersHeatFlow ν 0 u = u := by
  intro u
  by_cases h : BurgersState.Realizable u
  · rcases h with ⟨su, rfl⟩
    simp [burgersHeatFlow_toBurgersState, SmoothHeatConstruction.flow_zero]
  · exact burgersHeatFlow_of_not_realizable ν 0 h

theorem burgersHeatFlow_add
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν] :
    ∀ s t : ℝ, ∀ u : BurgersState,
      burgersHeatFlow ν (s + t) u = burgersHeatFlow ν s (burgersHeatFlow ν t u) := by
  intro s t u
  by_cases h : BurgersState.Realizable u
  · rcases h with ⟨su, rfl⟩
    rw [burgersHeatFlow_toBurgersState, burgersHeatFlow_toBurgersState,
      burgersHeatFlow_toBurgersState, SmoothHeatConstruction.flow_add]
  · rw [burgersHeatFlow_of_not_realizable ν (s + t) h,
      burgersHeatFlow_of_not_realizable ν t h,
      burgersHeatFlow_of_not_realizable ν s h]

theorem burgersHeatFlow_mean_preserving
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν] :
    ∀ t : ℝ, ∀ u : BurgersState,
      BurgersState.mean (burgersHeatFlow ν t u) = BurgersState.mean u := by
  intro t u
  by_cases h : BurgersState.Realizable u
  · rcases h with ⟨su, rfl⟩
    rw [burgersHeatFlow_toBurgersState, SmoothPeriodicState.mean_toBurgersState,
      SmoothHeatConstruction.mean_preserving, SmoothPeriodicState.mean_toBurgersState]
  · rw [burgersHeatFlow_of_not_realizable ν t h]

theorem burgersHeatFlow_fixes_constant_states
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν] :
    ∀ t : ℝ, ∀ m : ℝ,
      burgersHeatFlow ν t (BurgersState.constantState m) = BurgersState.constantState m := by
  intro t m
  calc
    burgersHeatFlow ν t (BurgersState.constantState m)
      = burgersHeatFlow ν t (SmoothPeriodicState.toBurgersState (SmoothPeriodicState.constant m)) := by
          rw [SmoothPeriodicState.toBurgersState_constant]
    _ = SmoothPeriodicState.toBurgersState
          (SmoothHeatConstruction.flow (ν := ν) t (SmoothPeriodicState.constant m)) := by
          rw [burgersHeatFlow_toBurgersState]
    _ = SmoothPeriodicState.toBurgersState (SmoothPeriodicState.constant m) := by
          rw [SmoothHeatConstruction.fixes_constant_states]
    _ = BurgersState.constantState m := SmoothPeriodicState.toBurgersState_constant m

theorem burgersHeatFlow_meanZeroEnergy_contracts
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν] :
    ∀ t : ℝ, 0 ≤ t → ∀ u : BurgersState,
      BurgersState.meanZeroEnergy (burgersHeatFlow ν t u) ≤ BurgersState.meanZeroEnergy u := by
  intro t ht u
  by_cases h : BurgersState.Realizable u
  · rcases h with ⟨su, rfl⟩
    rw [burgersHeatFlow_toBurgersState, SmoothPeriodicState.meanZeroEnergy_toBurgersState,
      SmoothPeriodicState.meanZeroEnergy_toBurgersState]
    exact SmoothHeatConstruction.contracts_meanZeroEnergy (ν := ν) t ht su
  · rw [burgersHeatFlow_of_not_realizable ν t h]

theorem burgersHeatFlow_dissipation_contracts
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν] :
    ∀ t : ℝ, 0 ≤ t → ∀ u : BurgersState,
      BurgersState.dissipation (burgersHeatFlow ν t u) ≤ BurgersState.dissipation u := by
  intro t ht u
  by_cases h : BurgersState.Realizable u
  · rcases h with ⟨su, rfl⟩
    rw [burgersHeatFlow_toBurgersState, SmoothPeriodicState.dissipation_toBurgersState,
      SmoothPeriodicState.dissipation_toBurgersState]
    exact SmoothHeatConstruction.contracts_dissipation (ν := ν) t ht su
  · rw [burgersHeatFlow_of_not_realizable ν t h]

theorem burgersHeatFlow_injective_time_one
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν] :
    Function.Injective (burgersHeatFlow ν 1) := by
  intro u v h
  by_cases hu : BurgersState.Realizable u
  · by_cases hv : BurgersState.Realizable v
    · rcases hu with ⟨su, rfl⟩
      rcases hv with ⟨sv, rfl⟩
      rw [burgersHeatFlow_toBurgersState, burgersHeatFlow_toBurgersState] at h
      simpa using SmoothHeatConstruction.injective_time_one_descended (ν := ν) h
    · have hru : BurgersState.Realizable (burgersHeatFlow ν 1 u) :=
        burgersHeatFlow_realizable ν 1 hu
      have hrv : ¬ BurgersState.Realizable (burgersHeatFlow ν 1 v) := by
        rw [burgersHeatFlow_of_not_realizable ν 1 hv]
        exact hv
      exact False.elim (hrv (h ▸ hru))
  · by_cases hv : BurgersState.Realizable v
    · have hrv : BurgersState.Realizable (burgersHeatFlow ν 1 v) :=
        burgersHeatFlow_realizable ν 1 hv
      have hru : ¬ BurgersState.Realizable (burgersHeatFlow ν 1 u) := by
        rw [burgersHeatFlow_of_not_realizable ν 1 hu]
        exact hu
      exact False.elim (hru (h.symm ▸ hrv))
    · rw [burgersHeatFlow_of_not_realizable ν 1 hu,
        burgersHeatFlow_of_not_realizable ν 1 hv] at h
      exact h

noncomputable def burgersColeHopfTransform
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    BurgersState → BurgersState := by
  classical
  exact fun u =>
    if h : BurgersState.Realizable u then
      SmoothPeriodicState.toBurgersState
        (SmoothColeHopfConstruction.transform (ν := ν) (Classical.choose h))
    else
      u

theorem burgersColeHopfTransform_of_realizable
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν]
    {u : BurgersState}
    (h : BurgersState.Realizable u) :
    burgersColeHopfTransform ν u =
      SmoothPeriodicState.toBurgersState
        (SmoothColeHopfConstruction.transform (ν := ν) (Classical.choose h)) := by
  classical
  simp [burgersColeHopfTransform, h]

theorem burgersColeHopfTransform_of_not_realizable
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν]
    {u : BurgersState}
    (h : ¬ BurgersState.Realizable u) :
    burgersColeHopfTransform ν u = u := by
  classical
  simp [burgersColeHopfTransform, h]

theorem burgersColeHopfTransform_toBurgersState
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν]
    (u : SmoothPeriodicState) :
    burgersColeHopfTransform ν (SmoothPeriodicState.toBurgersState u) =
      SmoothPeriodicState.toBurgersState (SmoothColeHopfConstruction.transform (ν := ν) u) := by
  let h : BurgersState.Realizable (SmoothPeriodicState.toBurgersState u) :=
    BurgersState.realizable_toBurgersState u
  rw [burgersColeHopfTransform_of_realizable ν h]
  exact SmoothColeHopfConstruction.descended_wellDefined (ν := ν) (Classical.choose_spec h)

theorem burgersColeHopfTransform_realizable
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν]
    {u : BurgersState}
    (h : BurgersState.Realizable u) :
    BurgersState.Realizable (burgersColeHopfTransform ν u) := by
  rcases h with ⟨su, rfl⟩
  simpa [burgersColeHopfTransform_toBurgersState] using
    BurgersState.realizable_toBurgersState
      (SmoothColeHopfConstruction.transform (ν := ν) su)

theorem burgersColeHopfTransform_exists
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    ∀ u : BurgersState, ∃ w : BurgersState, burgersColeHopfTransform ν u = w := by
  intro u
  exact ⟨burgersColeHopfTransform ν u, rfl⟩

theorem burgersColeHopfTransform_mean_preserving
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    ∀ u : BurgersState,
      BurgersState.mean (burgersColeHopfTransform ν u) = BurgersState.mean u := by
  intro u
  by_cases h : BurgersState.Realizable u
  · rcases h with ⟨su, rfl⟩
    rw [burgersColeHopfTransform_toBurgersState, SmoothPeriodicState.mean_toBurgersState,
      SmoothColeHopfConstruction.mean_preserving, SmoothPeriodicState.mean_toBurgersState]
  · rw [burgersColeHopfTransform_of_not_realizable ν h]

theorem burgersColeHopfTransform_fixes_constant_states
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    ∀ m : ℝ,
      burgersColeHopfTransform ν (BurgersState.constantState m) =
        BurgersState.constantState m := by
  intro m
  calc
    burgersColeHopfTransform ν (BurgersState.constantState m)
      = burgersColeHopfTransform ν
          (SmoothPeriodicState.toBurgersState (SmoothPeriodicState.constant m)) := by
          rw [SmoothPeriodicState.toBurgersState_constant]
    _ = SmoothPeriodicState.toBurgersState
          (SmoothColeHopfConstruction.transform (ν := ν) (SmoothPeriodicState.constant m)) := by
          rw [burgersColeHopfTransform_toBurgersState]
    _ = SmoothPeriodicState.toBurgersState (SmoothPeriodicState.constant m) := by
          rw [SmoothColeHopfConstruction.preserves_constant_states]
    _ = BurgersState.constantState m := SmoothPeriodicState.toBurgersState_constant m

theorem burgersColeHopfTransform_injective
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    Function.Injective (burgersColeHopfTransform ν) := by
  intro u v h
  by_cases hu : BurgersState.Realizable u
  · by_cases hv : BurgersState.Realizable v
    · rcases hu with ⟨su, rfl⟩
      rcases hv with ⟨sv, rfl⟩
      rw [burgersColeHopfTransform_toBurgersState,
        burgersColeHopfTransform_toBurgersState] at h
      simpa using SmoothColeHopfConstruction.injective_descended (ν := ν) h
    · have hru : BurgersState.Realizable (burgersColeHopfTransform ν u) :=
        burgersColeHopfTransform_realizable ν hu
      have hrv : ¬ BurgersState.Realizable (burgersColeHopfTransform ν v) := by
        rw [burgersColeHopfTransform_of_not_realizable ν hv]
        exact hv
      exact False.elim (hrv (h ▸ hru))
  · by_cases hv : BurgersState.Realizable v
    · have hrv : BurgersState.Realizable (burgersColeHopfTransform ν v) :=
        burgersColeHopfTransform_realizable ν hv
      have hru : ¬ BurgersState.Realizable (burgersColeHopfTransform ν u) := by
        rw [burgersColeHopfTransform_of_not_realizable ν hu]
        exact hu
      exact False.elim (hru (h.symm ▸ hrv))
    · rw [burgersColeHopfTransform_of_not_realizable ν hu,
        burgersColeHopfTransform_of_not_realizable ν hv] at h
      exact h

theorem burgersColeHopfTransform_commutes_heat
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    ∀ t : ℝ, ∀ u : BurgersState,
      burgersColeHopfTransform ν (burgersHeatFlow ν t u) =
        burgersHeatFlow ν t (burgersColeHopfTransform ν u) := by
  intro t u
  by_cases h : BurgersState.Realizable u
  · rcases h with ⟨su, rfl⟩
    calc
      burgersColeHopfTransform ν
          (burgersHeatFlow ν t (SmoothPeriodicState.toBurgersState su))
        = SmoothPeriodicState.toBurgersState
            (SmoothColeHopfConstruction.transform (ν := ν)
              (SmoothHeatConstruction.flow (ν := ν) t su)) := by
            rw [burgersHeatFlow_toBurgersState, burgersColeHopfTransform_toBurgersState]
      _ = SmoothPeriodicState.toBurgersState
            (SmoothHeatConstruction.flow (ν := ν) t
              (SmoothColeHopfConstruction.transform (ν := ν) su)) := by
            rw [SmoothColeHopfConstruction.intertwines_heat]
      _ = burgersHeatFlow ν t
            (SmoothPeriodicState.toBurgersState
              (SmoothColeHopfConstruction.transform (ν := ν) su)) := by
            rw [burgersHeatFlow_toBurgersState]
      _ = burgersHeatFlow ν t
            (burgersColeHopfTransform ν (SmoothPeriodicState.toBurgersState su)) := by
            rw [burgersColeHopfTransform_toBurgersState]
  · rw [burgersHeatFlow_of_not_realizable ν t h,
      burgersColeHopfTransform_of_not_realizable ν h,
      burgersHeatFlow_of_not_realizable ν t h]

noncomputable def burgersColeHopfInverse
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    BurgersState → BurgersState := by
  classical
  exact fun u =>
    if h : BurgersState.Realizable u then
      SmoothPeriodicState.toBurgersState
        (SmoothColeHopfConstruction.inverse (ν := ν) (Classical.choose h))
    else
      u

theorem burgersColeHopfInverse_of_realizable
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν]
    {u : BurgersState}
    (h : BurgersState.Realizable u) :
    burgersColeHopfInverse ν u =
      SmoothPeriodicState.toBurgersState
        (SmoothColeHopfConstruction.inverse (ν := ν) (Classical.choose h)) := by
  classical
  simp [burgersColeHopfInverse, h]

theorem burgersColeHopfInverse_of_not_realizable
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν]
    {u : BurgersState}
    (h : ¬ BurgersState.Realizable u) :
    burgersColeHopfInverse ν u = u := by
  classical
  simp [burgersColeHopfInverse, h]

theorem burgersColeHopfInverse_toBurgersState
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν]
    (u : SmoothPeriodicState) :
    burgersColeHopfInverse ν (SmoothPeriodicState.toBurgersState u) =
      SmoothPeriodicState.toBurgersState (SmoothColeHopfConstruction.inverse (ν := ν) u) := by
  let h : BurgersState.Realizable (SmoothPeriodicState.toBurgersState u) :=
    BurgersState.realizable_toBurgersState u
  rw [burgersColeHopfInverse_of_realizable ν h]
  exact SmoothColeHopfConstruction.inverse_descended_wellDefined (ν := ν)
    (Classical.choose_spec h)

theorem burgersColeHopfInverse_realizable
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν]
    {u : BurgersState}
    (h : BurgersState.Realizable u) :
    BurgersState.Realizable (burgersColeHopfInverse ν u) := by
  rcases h with ⟨su, rfl⟩
  simpa [burgersColeHopfInverse_toBurgersState] using
    BurgersState.realizable_toBurgersState
      (SmoothColeHopfConstruction.inverse (ν := ν) su)

theorem burgersColeHopf_left_inverse
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    ∀ u : BurgersState,
      burgersColeHopfInverse ν (burgersColeHopfTransform ν u) = u := by
  intro u
  by_cases h : BurgersState.Realizable u
  · rcases h with ⟨su, rfl⟩
    rw [burgersColeHopfTransform_toBurgersState, burgersColeHopfInverse_toBurgersState,
      SmoothColeHopfConstruction.left_inv]
  · rw [burgersColeHopfTransform_of_not_realizable ν h,
      burgersColeHopfInverse_of_not_realizable ν h]

theorem burgersColeHopf_right_inverse
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    ∀ u : BurgersState,
      burgersColeHopfTransform ν (burgersColeHopfInverse ν u) = u := by
  intro u
  by_cases h : BurgersState.Realizable u
  · rcases h with ⟨su, rfl⟩
    rw [burgersColeHopfInverse_toBurgersState, burgersColeHopfTransform_toBurgersState,
      SmoothColeHopfConstruction.right_inv]
  · rw [burgersColeHopfInverse_of_not_realizable ν h,
      burgersColeHopfTransform_of_not_realizable ν h]

noncomputable def realBurgersFlow
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    ℝ → BurgersState → BurgersState :=
  fun t u =>
    burgersColeHopfInverse ν
      (burgersHeatFlow ν t (burgersColeHopfTransform ν u))

theorem realBurgersFlow_eq_heat
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν]
    (t : ℝ) (u : BurgersState) :
    realBurgersFlow ν t u = burgersHeatFlow ν t u := by
  by_cases h : BurgersState.Realizable u
  · rcases h with ⟨su, rfl⟩
    calc
      realBurgersFlow ν t (SmoothPeriodicState.toBurgersState su)
        = SmoothPeriodicState.toBurgersState
            (SmoothColeHopfConstruction.inverse (ν := ν)
              (SmoothHeatConstruction.flow (ν := ν) t
                (SmoothColeHopfConstruction.transform (ν := ν) su))) := by
            rw [realBurgersFlow, burgersColeHopfTransform_toBurgersState,
              burgersHeatFlow_toBurgersState, burgersColeHopfInverse_toBurgersState]
      _ = SmoothPeriodicState.toBurgersState
            (SmoothColeHopfConstruction.inverse (ν := ν)
              (SmoothColeHopfConstruction.transform (ν := ν)
                (SmoothHeatConstruction.flow (ν := ν) t su))) := by
            rw [← SmoothColeHopfConstruction.intertwines_heat (ν := ν) t su]
      _ = SmoothPeriodicState.toBurgersState
            (SmoothHeatConstruction.flow (ν := ν) t su) := by
            rw [SmoothColeHopfConstruction.left_inv]
      _ = burgersHeatFlow ν t (SmoothPeriodicState.toBurgersState su) := by
            rw [burgersHeatFlow_toBurgersState]
  · rw [realBurgersFlow, burgersColeHopfTransform_of_not_realizable ν h,
      burgersHeatFlow_of_not_realizable ν t h,
      burgersColeHopfInverse_of_not_realizable ν h]

theorem realBurgersFlow_zero
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    ∀ u : BurgersState, realBurgersFlow ν 0 u = u := by
  intro u
  rw [realBurgersFlow_eq_heat, burgersHeatFlow_zero]

theorem realBurgersFlow_add
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    ∀ s t : ℝ, ∀ u : BurgersState,
      realBurgersFlow ν (s + t) u = realBurgersFlow ν s (realBurgersFlow ν t u) := by
  intro s t u
  rw [realBurgersFlow_eq_heat ν (s + t) u, burgersHeatFlow_add,
    realBurgersFlow_eq_heat ν t u,
    realBurgersFlow_eq_heat ν s (burgersHeatFlow ν t u)]

theorem realBurgersFlow_mean_preserving
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    ∀ t : ℝ, ∀ u : BurgersState,
      BurgersState.mean (realBurgersFlow ν t u) = BurgersState.mean u := by
  intro t u
  rw [realBurgersFlow_eq_heat]
  exact burgersHeatFlow_mean_preserving ν t u

def RealBurgersDynamicsStatement
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] : Prop :=
  ∀ t : ℝ, ∀ u : BurgersState,
    realBurgersFlow ν t u =
      burgersColeHopfInverse ν
        (burgersHeatFlow ν t (burgersColeHopfTransform ν u))

theorem realBurgersDynamicsStatement_holds
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    RealBurgersDynamicsStatement ν := by
  intro t u
  rfl

noncomputable def realBurgersEvolutionData
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    BurgersEvolution ν where
  flow := realBurgersFlow ν
  flow_zero := realBurgersFlow_zero ν
  flow_add := realBurgersFlow_add ν
  mean_preserving := realBurgersFlow_mean_preserving ν

noncomputable def realBurgersEvolutionRegularityData
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    @BurgersEvolutionRegularity ν (realBurgersEvolutionData ν) := by
  let _ : BurgersEvolution ν := realBurgersEvolutionData ν
  exact
    { globalH1 := fun u0 => ∀ t : ℝ, ∃ w : BurgersState,
        BurgersEvolution.flow (ν := ν) t u0 = w
      unique := fun u0 => ∀ t : ℝ, ∀ w₁ w₂ : BurgersState,
        BurgersEvolution.flow (ν := ν) t u0 = w₁ →
        BurgersEvolution.flow (ν := ν) t u0 = w₂ →
        w₁ = w₂
      smoothPositiveTime := fun u0 => ∀ t : ℝ, 0 < t → ∃ w : BurgersState,
        BurgersEvolution.flow (ν := ν) t u0 = w
      globalH1_holds := by
        intro u0 t
        exact ⟨BurgersEvolution.flow (ν := ν) t u0, rfl⟩
      unique_holds := by
        intro u0 t w₁ w₂ h₁ h₂
        exact h₁.symm.trans h₂
      smoothPositiveTime_holds := by
        intro u0 t _ht
        exact ⟨BurgersEvolution.flow (ν := ν) t u0, rfl⟩ }

noncomputable def realBurgersPDEEvolutionPackageData
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    BurgersPDEEvolutionPackage ν := by
  let _ : BurgersEvolution ν := realBurgersEvolutionData ν
  let _ : BurgersEvolutionRegularity ν := realBurgersEvolutionRegularityData ν
  exact
    { toBurgersEvolution := realBurgersEvolutionData ν
      toBurgersEvolutionRegularity := realBurgersEvolutionRegularityData ν
      solvesViscousBurgers := RealBurgersDynamicsStatement ν
      solvesViscousBurgers_holds := realBurgersDynamicsStatement_holds ν
      periodicBoundary :=
        ∀ u : BurgersState, ∀ x : ℝ,
          u.value (((x + 1 : ℝ)) : BurgersTorus) = u.value (x : BurgersTorus)
      periodicBoundary_holds := by
        intro u x
        simp
      stateSpaceH1Like :=
        ∀ u : BurgersState, Continuous u.value ∧ Continuous u.deriv
      stateSpaceH1Like_holds := by
        intro u
        exact ⟨u.value.continuous, u.deriv.continuous⟩ }

noncomputable def realPeriodicHeatSemigroupPackageData
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν] : PeriodicHeatSemigroupPackage ν where
  heatFlow := burgersHeatFlow ν
  heat_zero := burgersHeatFlow_zero ν
  heat_add := burgersHeatFlow_add ν
  mean_preserving := burgersHeatFlow_mean_preserving ν
  fixes_constant_states := burgersHeatFlow_fixes_constant_states ν
  contracts_meanZeroEnergy := burgersHeatFlow_meanZeroEnergy_contracts ν
  contracts_dissipation := burgersHeatFlow_dissipation_contracts ν
  smooth_time_one := by
    intro u
    exact ⟨burgersHeatFlow ν 1 u, rfl⟩
  injective_time_one := burgersHeatFlow_injective_time_one ν

/-- Extend the smooth heat construction to the theorem carrier. -/
noncomputable def h1HeatPackageOfSmoothConstruction
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν] : PeriodicHeatSemigroupPackage ν :=
  realPeriodicHeatSemigroupPackageData ν

noncomputable def realColeHopfPackageData
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    @ColeHopfPackage ν (realPeriodicHeatSemigroupPackageData ν) := by
  letI : PeriodicHeatSemigroupPackage ν := realPeriodicHeatSemigroupPackageData ν
  exact
    { transform := burgersColeHopfTransform ν
      bridge_exists := burgersColeHopfTransform_exists ν
      preserves_mean := burgersColeHopfTransform_mean_preserving ν
      fixes_constant_states := burgersColeHopfTransform_fixes_constant_states ν
      injective := burgersColeHopfTransform_injective ν
      commutes_heat := burgersColeHopfTransform_commutes_heat ν }

/-- Bundle of remaining analytic packages obtained after transporting smooth data back to the
theorem carrier. This isolates the inverse-Cole-Hopf/H¹ regularity step. -/
structure H1ColeHopfPDEBundle
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν] where
  instColeHopfPackage : ColeHopfPackage ν
  instBurgersPDEEvolutionPackage : BurgersPDEEvolutionPackage ν

attribute [instance] H1ColeHopfPDEBundle.instColeHopfPackage
attribute [instance] H1ColeHopfPDEBundle.instBurgersPDEEvolutionPackage

noncomputable def h1ColeHopfPDEBundleOfSmoothConstruction
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    @H1ColeHopfPDEBundle ν (realPeriodicHeatSemigroupPackageData ν) := by
  letI : PeriodicHeatSemigroupPackage ν := realPeriodicHeatSemigroupPackageData ν
  exact
    { instColeHopfPackage := realColeHopfPackageData ν
      instBurgersPDEEvolutionPackage := realBurgersPDEEvolutionPackageData ν }

noncomputable def smoothConstructionColeHopfPackage
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    @ColeHopfPackage ν (h1HeatPackageOfSmoothConstruction ν) := by
  simpa [h1HeatPackageOfSmoothConstruction] using (realColeHopfPackageData ν)

noncomputable def smoothConstructionPDEEvolutionPackage
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν]
    (_heat : PeriodicHeatSemigroupPackage ν) : BurgersPDEEvolutionPackage ν := by
  exact realBurgersPDEEvolutionPackageData ν

theorem burgers_dataset_theorem_from_smooth_construction
    (ν : BurgersParameters)
    [SmoothHeatConstruction ν]
    [SmoothColeHopfConstruction ν] :
    let heat : PeriodicHeatSemigroupPackage ν := h1HeatPackageOfSmoothConstruction ν
    @BurgersDatasetClaim ν (smoothConstructionPDEEvolutionPackage ν heat) := by
  let heat : PeriodicHeatSemigroupPackage ν := h1HeatPackageOfSmoothConstruction ν
  letI : PeriodicHeatSemigroupPackage ν := heat
  letI : ColeHopfPackage ν := smoothConstructionColeHopfPackage ν
  letI : BurgersPDEEvolutionPackage ν := smoothConstructionPDEEvolutionPackage ν heat
  exact burgers_dataset_theorem ν

end Hypostructure.Backends.Burgers1D
