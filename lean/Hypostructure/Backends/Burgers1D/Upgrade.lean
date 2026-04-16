import Hypostructure.Backends.Burgers1D.Lock
import Hypostructure.Backends.Burgers1D.PDEEvolution

namespace Hypostructure.Backends.Burgers1D

def BurgersAnalyticUpgradeInputStatement
    (ν : BurgersParameters)
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] : Prop :=
  BurgersBadPatternPackage.germSmallness (ν := ν) ∧
    BurgersBadPatternPackage.universalBadInitialized (ν := ν) ∧
    BurgersBadPatternPackage.catLibraryComplete (ν := ν) ∧
    BurgersBridgeInvariantStatement ν

class BurgersAnalyticUpgradePackage
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] where
  analyticUpgrade :
    BurgersBadPatternPackage.germSmallness (ν := ν) →
      BurgersBadPatternPackage.universalBadInitialized (ν := ν) →
      BurgersBadPatternPackage.catLibraryComplete (ν := ν) →
      BurgersBridgeInvariantStatement ν →
      BurgersGlobalRegularityStatement ν

instance burgersAnalyticUpgradePackage
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] :
    BurgersAnalyticUpgradePackage ν where
  analyticUpgrade := by
    intro _hgerm _hinit _hlib _hbridge
    exact burgers_evolution_global_regularity ν

theorem burgers_analytic_upgrade_inputs
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersAnalyticUpgradePackage ν] :
    BurgersAnalyticUpgradeInputStatement ν → BurgersGlobalRegularityStatement ν := by
  intro hinput
  rcases hinput with ⟨hgerm, hinit, hlib, hbridge⟩
  exact BurgersAnalyticUpgradePackage.analyticUpgrade (ν := ν) hgerm hinit hlib hbridge

theorem burgers_analytic_upgrade_from_package
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersAnalyticUpgradePackage ν] :
    BurgersStructuralExclusionStatement ν → BurgersGlobalRegularityStatement ν := by
  intro hstruct
  rcases hstruct with ⟨hgerm, hinit, hlib, hbridge⟩
  exact BurgersAnalyticUpgradePackage.analyticUpgrade (ν := ν) hgerm hinit hlib hbridge

end Hypostructure.Backends.Burgers1D
