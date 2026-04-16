import Hypostructure.Backends.Burgers1D.Final

namespace Hypostructure.Backends.Burgers1D

/-- Explicit concrete PDE-facing package for the current Burgers model flow.
This removes abstract package dependence for the model backend, but it is still a model
instantiation, not the actual viscous Burgers PDE package from the dataset. -/
noncomputable def burgersModelPDEEvolutionPackageData
    (ν : BurgersParameters) : BurgersPDEEvolutionPackage ν := by
  let _ : BurgersEvolution ν := burgersModelEvolutionData ν
  let _ : BurgersEvolutionRegularity ν := burgersModelEvolutionRegularityData ν
  exact
    { toBurgersEvolution := burgersModelEvolutionData ν
      toBurgersEvolutionRegularity := burgersModelEvolutionRegularityData ν
      solvesViscousBurgers :=
        ∀ t : ℝ, ∀ u : BurgersState,
          BurgersEvolution.flow (ν := ν) t u = BurgersState.modelFlow t u
      solvesViscousBurgers_holds := by
        intro t u
        rfl
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

theorem burgers_model_analytic_regularity
    (ν : BurgersParameters) :
    let _ : BurgersPDEEvolutionPackage ν := burgersModelPDEEvolutionPackageData ν
    let _ : PeriodicHeatSemigroupPackage ν := modelPeriodicHeatSemigroupPackageData ν
    let _ : ColeHopfPackage ν := modelColeHopfPackageData ν
    BurgersGlobalRegularityStatement ν := by
  let _ : BurgersPDEEvolutionPackage ν := burgersModelPDEEvolutionPackageData ν
  let _ : PeriodicHeatSemigroupPackage ν := modelPeriodicHeatSemigroupPackageData ν
  let _ : ColeHopfPackage ν := modelColeHopfPackageData ν
  exact burgers_analytic_regularity ν

theorem burgers_model_dataset_theorem
    (ν : BurgersParameters) :
    let _ : BurgersPDEEvolutionPackage ν := burgersModelPDEEvolutionPackageData ν
    let _ : PeriodicHeatSemigroupPackage ν := modelPeriodicHeatSemigroupPackageData ν
    let _ : ColeHopfPackage ν := modelColeHopfPackageData ν
    BurgersDatasetClaim ν := by
  let _ : BurgersPDEEvolutionPackage ν := burgersModelPDEEvolutionPackageData ν
  let _ : PeriodicHeatSemigroupPackage ν := modelPeriodicHeatSemigroupPackageData ν
  let _ : ColeHopfPackage ν := modelColeHopfPackageData ν
  exact burgers_dataset_theorem ν

end Hypostructure.Backends.Burgers1D
