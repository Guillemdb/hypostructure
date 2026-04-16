import Hypostructure.Backends.Burgers1D.Run
import Hypostructure.Backends.Burgers1D.PDEEvolution

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Framework

theorem burgers_structural_exclusion
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    BurgersStructuralExclusionStatement ν :=
  burgers_structural_exclusion_from_lock ν

theorem burgers_analytic_regularity
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν]
    [BurgersAnalyticUpgradePackage ν] :
    BurgersGlobalRegularityStatement ν := by
  exact burgers_analytic_upgrade_from_package ν (burgers_structural_exclusion ν)

theorem burgers_dataset_theorem
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν]
    [BurgersAnalyticUpgradePackage ν] :
    BurgersDatasetClaim ν := by
  exact ⟨
    burgers_stateSpace_statement ν,
    burgers_periodic_boundary_statement ν,
    burgers_dynamics_statement ν,
    burgers_mean_zero_sector_invariant ν,
    burgers_analytic_regularity ν
  ⟩

theorem burgers_final_certificate_sound
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν]
    [BurgersAnalyticUpgradePackage ν] :
    (burgersPositiveRoute ν).analyticReg.payload.targetClaim := by
  exact burgers_analytic_upgrade_inputs ν

def burgersRemainingMathlibWork : List String :=
  [ "Identify the current periodic profile-plus-derivative carrier with a genuine Sobolev H1(T) model and connect its derivative witness to mathlib weak-derivative infrastructure."
  , "Replace the current model semigroup by the actual periodic viscous Burgers evolution on that H1(T) carrier."
  , "Prove that the chosen evolution solves u_t + u u_x = nu u_xx with periodic boundary conditions."
  , "Upgrade the current snapshot-level energy/coercivity lemmas into the exact mean-zero energy identity for the actual Burgers flow."
  , "Formalize the periodic heat semigroup on T with the smoothing and uniqueness statements used by the dataset."
  , "Formalize the Cole-Hopf transform on the periodic mean-zero sector and prove conjugacy between Burgers and heat."
  , "Prove the Lock obstruction theorem: Burgers bad morphisms induce impossible heat bad morphisms."
  , "Replace the current model regularity package by a genuine analytic upgrade theorem derived from structural exclusion + Cole-Hopf + heat smoothing."
  ]

end Hypostructure.Backends.Burgers1D
