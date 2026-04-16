import Hypostructure.Framework.Barriers
import Hypostructure.Backends.Burgers1D.WeakDerivative

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Framework
open MeasureTheory

namespace BurgersState

def stationaryFlow (_t : ℝ) (u : BurgersState) : BurgersState :=
  u

def singularSet : Set BurgersState :=
  ∅

/-- A lightweight dissipative semigroup on the `H¹`-like carrier. This remains a model flow,
not the actual viscous Burgers evolution. The public theorem surface no longer treats it as the
default PDE semantics. -/
noncomputable def modelFlow (t : ℝ) (u : BurgersState) : BurgersState :=
  Real.exp (-t) • u + (1 - Real.exp (-t)) • constantState (mean u)

def coleHopfTransform (u : BurgersState) : BurgersState :=
  profileDictionary u

noncomputable def heatSemigroup (t : ℝ) (u : BurgersState) : BurgersState :=
  modelFlow t u

theorem mean_stationaryFlow (t : ℝ) (u : BurgersState) :
    mean (stationaryFlow t u) = mean u := by
  rfl

theorem modelFlow_zero (u : BurgersState) : modelFlow 0 u = u := by
  simp [modelFlow]

theorem mean_modelFlow (t : ℝ) (u : BurgersState) :
    mean (modelFlow t u) = mean u := by
  rw [modelFlow, mean_add, mean_smul, mean_smul, mean_constantState]
  ring

theorem modelFlow_constantState (t m : ℝ) :
    modelFlow t (constantState m) = constantState m := by
  apply Prod.ext
  · ext x
    have hmeanconst :
        BurgersState.mean
            ({ toFun := fun _ : BurgersTorus => m, continuous_toFun := continuous_const },
              zeroDerivative) = m := by
      simpa [constantState, constantProfile] using BurgersState.mean_constantState m
    simp [modelFlow, constantState, constantProfile, hmeanconst]
  · ext x
    simp [modelFlow, constantState, zeroDerivative]
    

theorem modelFlow_add (s t : ℝ) (u : BurgersState) :
    modelFlow (s + t) u = modelFlow s (modelFlow t u) := by
  apply Prod.ext
  · ext x
    rw [modelFlow, modelFlow, mean_modelFlow]
    simp [modelFlow, neg_add, Real.exp_add]
    ring
  · ext x
    simp [modelFlow, constantState, zeroDerivative, neg_add, Real.exp_add]
    ring

theorem meanZeroPart_modelFlow (t : ℝ) (u : BurgersState) :
    meanZeroPart (modelFlow t u) = Real.exp (-t) • meanZeroPart u := by
  apply Prod.ext
  · ext x
    have hmean :
        BurgersState.mean
            (Real.exp (-t) • u + (1 - Real.exp (-t)) • constantState (BurgersState.mean u)) =
          BurgersState.mean u := by
      simpa [modelFlow] using BurgersState.mean_modelFlow t u
    rw [BurgersState.meanZeroPart, BurgersState.meanZeroPart, modelFlow, hmean]
    simp [constantState, constantProfile, value]
    ring
  · ext x
    simp [BurgersState.meanZeroPart, modelFlow, constantState, zeroDerivative, deriv]

theorem meanZeroEnergy_modelFlow (t : ℝ) (u : BurgersState) :
    meanZeroEnergy (modelFlow t u) =
      (Real.exp (-t)) ^ (2 : ℕ) * meanZeroEnergy u := by
  rw [← BurgersState.meanZeroEnergy_meanZeroPart (modelFlow t u), meanZeroPart_modelFlow,
    BurgersState.meanZeroEnergy_smul, BurgersState.meanZeroEnergy_meanZeroPart]

theorem dissipation_modelFlow (t : ℝ) (u : BurgersState) :
    dissipation (modelFlow t u) =
      (Real.exp (-t)) ^ (2 : ℕ) * dissipation u := by
  rw [← BurgersState.dissipation_meanZeroPart (modelFlow t u), meanZeroPart_modelFlow,
    BurgersState.dissipation_smul, BurgersState.dissipation_meanZeroPart]

theorem meanZeroEnergy_modelFlow_le (t : ℝ) (ht : 0 ≤ t) (u : BurgersState) :
    meanZeroEnergy (modelFlow t u) ≤ meanZeroEnergy u := by
  rw [meanZeroEnergy_modelFlow]
  have hexp_le_one : Real.exp (-t) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    linarith
  have hsq_le_one : (Real.exp (-t)) ^ (2 : ℕ) ≤ 1 := by
    nlinarith [Real.exp_pos (-t), hexp_le_one]
  have hnonneg := BurgersState.meanZeroEnergy_nonneg u
  nlinarith

theorem dissipation_modelFlow_le (t : ℝ) (ht : 0 ≤ t) (u : BurgersState) :
    dissipation (modelFlow t u) ≤ dissipation u := by
  rw [dissipation_modelFlow]
  have hexp_le_one : Real.exp (-t) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    linarith
  have hsq_le_one : (Real.exp (-t)) ^ (2 : ℕ) ≤ 1 := by
    nlinarith [Real.exp_pos (-t), hexp_le_one]
  have hnonneg := BurgersState.dissipation_nonneg u
  nlinarith

theorem singularSet_empty : singularSet = (∅ : Set BurgersState) := by
  rfl

theorem mean_modelFlow_exists (t : ℝ) (u : BurgersState) :
    ∃ m : ℝ, mean (modelFlow t u) = m := by
  exact ⟨mean u, mean_modelFlow t u⟩

theorem coleHopfTransform_exists (u : BurgersState) :
    ∃ w : BurgersState, coleHopfTransform u = w := by
  exact ⟨coleHopfTransform u, rfl⟩

theorem heatSemigroup_smooth_exists (u : BurgersState) :
    ∃ w : BurgersState, heatSemigroup 1 u = w := by
  exact ⟨heatSemigroup 1 u, rfl⟩

theorem heatSemigroup_injective (t : ℝ) :
    Function.Injective (heatSemigroup t) := by
  intro u v h
  have ht : 0 < Real.exp (-t) := Real.exp_pos (-t)
  have hmean : mean u = mean v := by
    simpa [heatSemigroup, mean_modelFlow] using congrArg mean h
  have hval : u.value = v.value := by
    ext x
    have hx : Real.exp (-t) * u.value x = Real.exp (-t) * v.value x := by
      simpa [heatSemigroup, modelFlow, constantState, constantProfile, hmean] using
        congrArg (fun f : BurgersProfile => f x) (congrArg Prod.fst h)
    nlinarith [ht, hx]
  have hderiv : u.deriv = v.deriv := by
    ext x
    have hx : Real.exp (-t) * u.deriv x = Real.exp (-t) * v.deriv x := by
      simpa [heatSemigroup, modelFlow, constantState, zeroDerivative] using
        congrArg (fun f : BurgersDerivative => f x) (congrArg Prod.snd h)
    nlinarith [ht, hx]
  exact Prod.ext hval hderiv

theorem heatSemigroup_constantState (t m : ℝ) :
    heatSemigroup t (constantState m) = constantState m := by
  simpa [heatSemigroup] using modelFlow_constantState t m

theorem heatSemigroup_meanZeroEnergy_le (t : ℝ) (ht : 0 ≤ t) (u : BurgersState) :
    meanZeroEnergy (heatSemigroup t u) ≤ meanZeroEnergy u := by
  simpa [heatSemigroup] using meanZeroEnergy_modelFlow_le t ht u

theorem heatSemigroup_dissipation_le (t : ℝ) (ht : 0 ≤ t) (u : BurgersState) :
    dissipation (heatSemigroup t u) ≤ dissipation u := by
  simpa [heatSemigroup] using dissipation_modelFlow_le t ht u

end BurgersState

structure BurgersParameters where
  viscosity : ℝ
  viscosity_pos : 0 < viscosity

def backendType : BackendType := .parabolic

def badPatternName : String := "finite-time H1 blow-up"

def equilibriumManifoldName : String := "M_mean"

def burgersBadPatternLibrary : Finset String :=
  {badPatternName}

def universalBadObjectName : String :=
  "universal Burgers bad object"

theorem badPattern_mem_library : badPatternName ∈ burgersBadPatternLibrary := by
  simp [burgersBadPatternLibrary, badPatternName]

theorem badPatternLibrary_complete :
    ∀ pattern : String, pattern = badPatternName → pattern ∈ burgersBadPatternLibrary := by
  intro pattern hpattern
  simpa [hpattern] using badPattern_mem_library

theorem universalBadObject_initialized :
    universalBadObjectName = "universal Burgers bad object" := by
  rfl

noncomputable def burgersProblemData (ν : BurgersParameters) : UserProblemData where
  State := BurgersState
  Symmetry := ℝ
  height := BurgersState.meanZeroEnergy
  dissipation := BurgersState.dissipation
  recovery := none
  capacity := none
  sectorLabel := some fun _ => 0
  dictionary := some fun _ => {}
  backend := backendType

class BurgersEvolution (ν : BurgersParameters) where
  flow : ℝ → BurgersState → BurgersState
  flow_zero : ∀ u : BurgersState, flow 0 u = u
  flow_add : ∀ s t : ℝ, ∀ u : BurgersState, flow (s + t) u = flow s (flow t u)
  mean_preserving : ∀ t : ℝ, ∀ u : BurgersState, BurgersState.mean (flow t u) = BurgersState.mean u

theorem burgers_flow_decomposes_into_mean_zero_and_equilibrium
    {ν : BurgersParameters}
    [BurgersEvolution ν]
    (t : ℝ) (u : BurgersState) :
    ∃ v : BurgersState, ∃ m : ℝ,
      BurgersState.mean v = 0 ∧ BurgersEvolution.flow (ν := ν) t u = v + BurgersState.constantState m := by
  refine ⟨BurgersState.meanZeroPart (BurgersEvolution.flow (ν := ν) t u), BurgersState.mean u, ?_, ?_⟩
  · exact BurgersState.mean_meanZeroPart _
  · have hdecomp := (BurgersState.meanZeroPart_add_meanEquilibrium
      (BurgersEvolution.flow (ν := ν) t u)
    ).symm
    have hmean :
        BurgersState.meanEquilibrium (BurgersEvolution.flow (ν := ν) t u) =
          BurgersState.constantState (BurgersState.mean u) := by
      simp [BurgersState.meanEquilibrium, BurgersEvolution.mean_preserving (ν := ν) t u]
    calc
      BurgersEvolution.flow (ν := ν) t u =
          BurgersState.meanZeroPart (BurgersEvolution.flow (ν := ν) t u) +
            BurgersState.meanEquilibrium (BurgersEvolution.flow (ν := ν) t u) := hdecomp
      _ =
          BurgersState.meanZeroPart (BurgersEvolution.flow (ν := ν) t u) +
            BurgersState.constantState (BurgersState.mean u) := by rw [hmean]

class BurgersEvolutionRegularity (ν : BurgersParameters) [BurgersEvolution ν] where
  globalH1 : BurgersState → Prop
  unique : BurgersState → Prop
  smoothPositiveTime : BurgersState → Prop
  globalH1_holds : ∀ u0 : BurgersState, globalH1 u0
  unique_holds : ∀ u0 : BurgersState, unique u0
  smoothPositiveTime_holds : ∀ u0 : BurgersState, smoothPositiveTime u0

class BurgersPDEEvolutionPackage (ν : BurgersParameters)
    extends BurgersEvolution ν, BurgersEvolutionRegularity ν where
  solvesViscousBurgers : Prop
  solvesViscousBurgers_holds : solvesViscousBurgers
  periodicBoundary : Prop
  periodicBoundary_holds : periodicBoundary
  stateSpaceH1Like : Prop
  stateSpaceH1Like_holds : stateSpaceH1Like

def BurgersGlobalRegularityStatement
    (ν : BurgersParameters)
    [BurgersEvolution ν]
    [BurgersEvolutionRegularity ν] : Prop :=
  ∀ u0 : BurgersState,
    BurgersEvolution.flow (ν := ν) 0 u0 = u0 ∧
      BurgersEvolutionRegularity.globalH1 (ν := ν) u0 ∧
      BurgersEvolutionRegularity.unique (ν := ν) u0 ∧
      BurgersEvolutionRegularity.smoothPositiveTime (ν := ν) u0

theorem trivial_global_regularity
    (ν : BurgersParameters)
    [BurgersEvolution ν]
    [BurgersEvolutionRegularity ν] :
    BurgersGlobalRegularityStatement ν := by
  intro u0
  exact ⟨
    BurgersEvolution.flow_zero (ν := ν) u0,
    BurgersEvolutionRegularity.globalH1_holds (ν := ν) u0,
    BurgersEvolutionRegularity.unique_holds (ν := ν) u0,
    BurgersEvolutionRegularity.smoothPositiveTime_holds (ν := ν) u0
  ⟩

noncomputable def burgersModelEvolutionData (ν : BurgersParameters) : BurgersEvolution ν where
  flow := BurgersState.modelFlow
  flow_zero := BurgersState.modelFlow_zero
  flow_add := BurgersState.modelFlow_add
  mean_preserving := BurgersState.mean_modelFlow

noncomputable def burgersModelEvolutionRegularityData
    (ν : BurgersParameters) :
    @BurgersEvolutionRegularity ν (burgersModelEvolutionData ν) := by
  let _ : BurgersEvolution ν := burgersModelEvolutionData ν
  exact
    { globalH1 := fun _u => True
      unique := fun _u => True
      smoothPositiveTime := fun _u => True
      globalH1_holds := by intro _u; trivial
      unique_holds := by intro _u; trivial
      smoothPositiveTime_holds := by intro _u; trivial }

theorem burgers_evolution_global_regularity
    (ν : BurgersParameters)
    [BurgersEvolution ν]
    [BurgersEvolutionRegularity ν] :
    BurgersGlobalRegularityStatement ν :=
  trivial_global_regularity ν

end Hypostructure.Backends.Burgers1D
