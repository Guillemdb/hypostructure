import Hypostructure.Backends.Burgers1D.Basic

namespace Hypostructure.Backends.Burgers1D

/-- The PDE-facing state-space commitment used by the dataset statement. -/
def BurgersStateSpaceStatement
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν] : Prop :=
  BurgersPDEEvolutionPackage.stateSpaceH1Like (ν := ν)

/-- The periodic boundary commitment used by the dataset statement. -/
def BurgersPeriodicBoundaryStatement
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν] : Prop :=
  BurgersPDEEvolutionPackage.periodicBoundary (ν := ν)

/-- The PDE-solving commitment used by the dataset statement. -/
def BurgersDynamicsStatement
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν] : Prop :=
  BurgersPDEEvolutionPackage.solvesViscousBurgers (ν := ν)

/-- Mean-zero sector decomposition along the chosen Burgers evolution. -/
def BurgersMeanZeroSectorInvariant
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν] : Prop :=
  ∀ t : ℝ, ∀ u : BurgersState, ∃ v : BurgersState, ∃ m : ℝ,
    BurgersState.mean v = 0 ∧
      BurgersEvolution.flow (ν := ν) t u = v + BurgersState.constantState m

theorem burgers_stateSpace_statement
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν] :
    BurgersStateSpaceStatement ν :=
  BurgersPDEEvolutionPackage.stateSpaceH1Like_holds (ν := ν)

theorem burgers_periodic_boundary_statement
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν] :
    BurgersPeriodicBoundaryStatement ν :=
  BurgersPDEEvolutionPackage.periodicBoundary_holds (ν := ν)

theorem burgers_dynamics_statement
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν] :
    BurgersDynamicsStatement ν :=
  BurgersPDEEvolutionPackage.solvesViscousBurgers_holds (ν := ν)

theorem burgers_mean_zero_sector_invariant
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν] :
    BurgersMeanZeroSectorInvariant ν :=
  burgers_flow_decomposes_into_mean_zero_and_equilibrium

/-- A reusable summary of the PDE package obligations that the Burgers backend now exposes. -/
structure BurgersPDEPackageSummary
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν] where
  stateSpace : BurgersStateSpaceStatement ν
  periodicBoundary : BurgersPeriodicBoundaryStatement ν
  dynamics : BurgersDynamicsStatement ν
  meanZeroSector : BurgersMeanZeroSectorInvariant ν
  regularity : BurgersGlobalRegularityStatement ν

def burgersPDEPackageSummary
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν] :
    BurgersPDEPackageSummary ν where
  stateSpace := burgers_stateSpace_statement ν
  periodicBoundary := burgers_periodic_boundary_statement ν
  dynamics := burgers_dynamics_statement ν
  meanZeroSector := burgers_mean_zero_sector_invariant ν
  regularity := burgers_evolution_global_regularity ν

/-- Dataset-level theorem statement exposed by the explicit PDE package. -/
def BurgersDatasetClaim
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν] : Prop :=
  BurgersStateSpaceStatement ν ∧
    BurgersPeriodicBoundaryStatement ν ∧
    BurgersDynamicsStatement ν ∧
    BurgersMeanZeroSectorInvariant ν ∧
    BurgersGlobalRegularityStatement ν

theorem burgers_dataset_claim_from_package
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν] :
    BurgersDatasetClaim ν := by
  exact ⟨
    burgers_stateSpace_statement ν,
    burgers_periodic_boundary_statement ν,
    burgers_dynamics_statement ν,
    burgers_mean_zero_sector_invariant ν,
    burgers_evolution_global_regularity ν
  ⟩

end Hypostructure.Backends.Burgers1D
