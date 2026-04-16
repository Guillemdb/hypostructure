import Hypostructure.Backends.Burgers1D.HeatKernel

namespace Hypostructure.Backends.Burgers1D

/-- Explicit package for the Cole-Hopf bridge used by the Burgers Lock route. -/
class ColeHopfPackage (ν : BurgersParameters) [PeriodicHeatSemigroupPackage ν] where
  transform : BurgersState → BurgersState
  bridge_exists : ∀ u : BurgersState, ∃ w : BurgersState, transform u = w
  preserves_mean :
    ∀ u : BurgersState, BurgersState.mean (transform u) = BurgersState.mean u
  fixes_constant_states :
    ∀ m : ℝ, transform (BurgersState.constantState m) = BurgersState.constantState m
  injective : Function.Injective transform
  commutes_heat :
    ∀ t : ℝ, ∀ u : BurgersState,
      transform (PeriodicHeatSemigroupPackage.heatFlow (ν := ν) t u) =
        PeriodicHeatSemigroupPackage.heatFlow (ν := ν) t (transform u)

def ColeHopfExistsStatement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] : Prop :=
  ∀ u : BurgersState, ∃ w : BurgersState,
    ColeHopfPackage.transform (ν := ν) u = w

def ColeHopfMeanStatement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] : Prop :=
  ∀ u : BurgersState,
    BurgersState.mean (ColeHopfPackage.transform (ν := ν) u) = BurgersState.mean u

def ColeHopfEquilibriumStatement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] : Prop :=
  ∀ m : ℝ,
    ColeHopfPackage.transform (ν := ν) (BurgersState.constantState m) =
      BurgersState.constantState m

def ColeHopfFaithfulStatement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] : Prop :=
  Function.Injective (ColeHopfPackage.transform (ν := ν))

def ColeHopfHeatIntertwiningStatement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] : Prop :=
  ∀ t : ℝ, ∀ u : BurgersState,
    ColeHopfPackage.transform (ν := ν)
        (PeriodicHeatSemigroupPackage.heatFlow (ν := ν) t u) =
      PeriodicHeatSemigroupPackage.heatFlow (ν := ν) t
        (ColeHopfPackage.transform (ν := ν) u)

def ColeHopfBridgeStatement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] : Prop :=
  ColeHopfExistsStatement ν ∧
    ColeHopfMeanStatement ν ∧
    ColeHopfEquilibriumStatement ν ∧
    ColeHopfFaithfulStatement ν ∧
    ColeHopfHeatIntertwiningStatement ν

def ColeHopfHeatRoute
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] : Prop :=
  ColeHopfBridgeStatement ν ∧ HeatPackageSummary ν

theorem coleHopf_exists_statement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] :
    ColeHopfExistsStatement ν :=
  ColeHopfPackage.bridge_exists (ν := ν)

theorem coleHopf_mean_statement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] :
    ColeHopfMeanStatement ν :=
  ColeHopfPackage.preserves_mean (ν := ν)

theorem coleHopf_equilibrium_statement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] :
    ColeHopfEquilibriumStatement ν :=
  ColeHopfPackage.fixes_constant_states (ν := ν)

theorem coleHopf_faithful_statement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] :
    ColeHopfFaithfulStatement ν :=
  ColeHopfPackage.injective (ν := ν)

theorem coleHopf_heat_intertwining_statement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] :
    ColeHopfHeatIntertwiningStatement ν :=
  ColeHopfPackage.commutes_heat (ν := ν)

theorem coleHopf_bridge_statement
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] :
    ColeHopfBridgeStatement ν := by
  exact ⟨
    coleHopf_exists_statement ν,
    coleHopf_mean_statement ν,
    coleHopf_equilibrium_statement ν,
    coleHopf_faithful_statement ν,
    coleHopf_heat_intertwining_statement ν
  ⟩

theorem coleHopf_heat_route
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] :
    ColeHopfHeatRoute ν := by
  exact ⟨coleHopf_bridge_statement ν, heat_package_summary ν⟩

/-- Current model Cole-Hopf package built from the lightweight transform in `Basic`.
This is explicit data, not a default backend instance. -/
def modelColeHopfPackageData
    (ν : BurgersParameters)
    [PeriodicHeatSemigroupPackage ν] : ColeHopfPackage ν where
  transform := BurgersState.coleHopfTransform
  bridge_exists := BurgersState.coleHopfTransform_exists
  preserves_mean := by
    intro u
    simp [BurgersState.coleHopfTransform, BurgersState.profileDictionary]
  fixes_constant_states := by
    intro m
    simp [BurgersState.coleHopfTransform, BurgersState.profileDictionary]
  injective := by
    intro u v h
    simpa [BurgersState.coleHopfTransform, BurgersState.profileDictionary] using h
  commutes_heat := by
    intro t u
    simp [BurgersState.coleHopfTransform, BurgersState.profileDictionary]

end Hypostructure.Backends.Burgers1D
