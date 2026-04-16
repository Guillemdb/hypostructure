import Hypostructure.Backends.Burgers1D.BadPatterns
import Hypostructure.Backends.Burgers1D.ColeHopf

namespace Hypostructure.Backends.Burgers1D

def BurgersBridgeInvariantStatement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] : Prop :=
  ColeHopfMeanStatement ν ∧
    ColeHopfEquilibriumStatement ν ∧
    ColeHopfFaithfulStatement ν ∧
    ColeHopfHeatIntertwiningStatement ν ∧
    HeatEquilibriumStatement ν ∧
    HeatEnergyContractiveStatement ν ∧
    HeatDissipationContractiveStatement ν

def BurgersLockBlockedStatement
    (ν : BurgersParameters)
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] : Prop :=
  BurgersBadPatternPackage.germSmallness (ν := ν) ∧
    BurgersBadPatternPackage.universalBadInitialized (ν := ν) ∧
    BurgersBadPatternPackage.catLibraryComplete (ν := ν) ∧
    ColeHopfHeatRoute ν

def BurgersStructuralExclusionStatement
    (ν : BurgersParameters)
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] : Prop :=
  BurgersBadPatternPackage.germSmallness (ν := ν) ∧
    BurgersBadPatternPackage.universalBadInitialized (ν := ν) ∧
    BurgersBadPatternPackage.catLibraryComplete (ν := ν) ∧
    BurgersBridgeInvariantStatement ν

theorem burgers_bridge_invariants
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] :
    BurgersBridgeInvariantStatement ν := by
  exact ⟨
    coleHopf_mean_statement ν,
    coleHopf_equilibrium_statement ν,
    coleHopf_faithful_statement ν,
    coleHopf_heat_intertwining_statement ν,
    heat_equilibrium_statement ν,
    heat_energy_contractive_statement ν,
    heat_dissipation_contractive_statement ν
  ⟩

theorem burgers_lock_blocked
    (ν : BurgersParameters)
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] :
    BurgersLockBlockedStatement ν := by
  exact ⟨
    burgers_germ_smallness ν,
    burgers_universal_bad_initialized ν,
    burgers_cat_library_complete ν,
    coleHopf_heat_route ν
  ⟩

theorem burgers_structural_exclusion_from_lock
    (ν : BurgersParameters)
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] :
    BurgersStructuralExclusionStatement ν := by
  rcases burgers_lock_blocked ν with ⟨hgerm, hinit, hlib, hroute⟩
  exact ⟨hgerm, hinit, hlib, burgers_bridge_invariants ν⟩

end Hypostructure.Backends.Burgers1D
