import Hypostructure.Backends.Burgers1D.Basic

namespace Hypostructure.Backends.Burgers1D

/-- Explicit package for the periodic heat semigroup side of the Burgers backend. -/
class PeriodicHeatSemigroupPackage (ν : BurgersParameters) where
  heatFlow : ℝ → BurgersState → BurgersState
  heat_zero : ∀ u : BurgersState, heatFlow 0 u = u
  heat_add : ∀ s t : ℝ, ∀ u : BurgersState, heatFlow (s + t) u = heatFlow s (heatFlow t u)
  mean_preserving : ∀ t : ℝ, ∀ u : BurgersState, BurgersState.mean (heatFlow t u) = BurgersState.mean u
  fixes_constant_states :
    ∀ t : ℝ, ∀ m : ℝ, heatFlow t (BurgersState.constantState m) = BurgersState.constantState m
  contracts_meanZeroEnergy :
    ∀ t : ℝ, 0 ≤ t → ∀ u : BurgersState,
      BurgersState.meanZeroEnergy (heatFlow t u) ≤ BurgersState.meanZeroEnergy u
  contracts_dissipation :
    ∀ t : ℝ, 0 ≤ t → ∀ u : BurgersState,
      BurgersState.dissipation (heatFlow t u) ≤ BurgersState.dissipation u
  smooth_time_one : ∀ u : BurgersState, ∃ w : BurgersState, heatFlow 1 u = w
  injective_time_one : Function.Injective (heatFlow 1)

def HeatSmoothStatement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν] : Prop :=
  ∀ u : BurgersState, ∃ w : BurgersState,
    PeriodicHeatSemigroupPackage.heatFlow (ν := ν) 1 u = w

def HeatUniqueStatement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν] : Prop :=
  Function.Injective (PeriodicHeatSemigroupPackage.heatFlow (ν := ν) 1)

def HeatEquilibriumStatement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν] : Prop :=
  ∀ t : ℝ, ∀ m : ℝ,
    PeriodicHeatSemigroupPackage.heatFlow (ν := ν) t (BurgersState.constantState m) =
      BurgersState.constantState m

def HeatEnergyContractiveStatement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν] : Prop :=
  ∀ t : ℝ, 0 ≤ t → ∀ u : BurgersState,
    BurgersState.meanZeroEnergy (PeriodicHeatSemigroupPackage.heatFlow (ν := ν) t u) ≤
      BurgersState.meanZeroEnergy u

def HeatDissipationContractiveStatement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν] : Prop :=
  ∀ t : ℝ, 0 ≤ t → ∀ u : BurgersState,
    BurgersState.dissipation (PeriodicHeatSemigroupPackage.heatFlow (ν := ν) t u) ≤
      BurgersState.dissipation u

def HeatPackageSummary
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν] : Prop :=
  HeatSmoothStatement ν ∧
    HeatUniqueStatement ν ∧
    HeatEquilibriumStatement ν ∧
    HeatEnergyContractiveStatement ν ∧
    HeatDissipationContractiveStatement ν

theorem heat_smooth_statement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν] :
    HeatSmoothStatement ν :=
  PeriodicHeatSemigroupPackage.smooth_time_one (ν := ν)

theorem heat_unique_statement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν] :
    HeatUniqueStatement ν :=
  PeriodicHeatSemigroupPackage.injective_time_one (ν := ν)

theorem heat_equilibrium_statement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν] :
    HeatEquilibriumStatement ν :=
  PeriodicHeatSemigroupPackage.fixes_constant_states (ν := ν)

theorem heat_energy_contractive_statement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν] :
    HeatEnergyContractiveStatement ν :=
  PeriodicHeatSemigroupPackage.contracts_meanZeroEnergy (ν := ν)

theorem heat_dissipation_contractive_statement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν] :
    HeatDissipationContractiveStatement ν :=
  PeriodicHeatSemigroupPackage.contracts_dissipation (ν := ν)

theorem heat_package_summary
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν] :
    HeatPackageSummary ν := by
  exact ⟨
    heat_smooth_statement ν,
    heat_unique_statement ν,
    heat_equilibrium_statement ν,
    heat_energy_contractive_statement ν,
    heat_dissipation_contractive_statement ν
  ⟩

/-- Current model heat package built from the lightweight semigroup in `Basic`. This is explicit
data, not a default backend instance. -/
noncomputable def modelPeriodicHeatSemigroupPackageData
    (ν : BurgersParameters) : PeriodicHeatSemigroupPackage ν where
  heatFlow := BurgersState.heatSemigroup
  heat_zero := by
    intro u
    simpa [BurgersState.heatSemigroup] using BurgersState.modelFlow_zero u
  heat_add := by
    intro s t u
    simpa [BurgersState.heatSemigroup] using BurgersState.modelFlow_add s t u
  mean_preserving := by
    intro t u
    simpa [BurgersState.heatSemigroup] using BurgersState.mean_modelFlow t u
  fixes_constant_states := by
    intro t m
    simpa [BurgersState.heatSemigroup] using BurgersState.heatSemigroup_constantState t m
  contracts_meanZeroEnergy := by
    intro t ht u
    simpa [BurgersState.heatSemigroup] using BurgersState.heatSemigroup_meanZeroEnergy_le t ht u
  contracts_dissipation := by
    intro t ht u
    simpa [BurgersState.heatSemigroup] using BurgersState.heatSemigroup_dissipation_le t ht u
  smooth_time_one := BurgersState.heatSemigroup_smooth_exists
  injective_time_one := BurgersState.heatSemigroup_injective 1

end Hypostructure.Backends.Burgers1D
