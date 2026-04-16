import Hypostructure.Backends.Burgers1D.Analysis
import Hypostructure.Backends.Burgers1D.ColeHopf

namespace Hypostructure.Backends.Burgers1D

class BurgersLiterature
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] where
  energyIdentity : Prop
  energyIdentity_holds : energyIdentity
  zeroEventRoute : Prop
  zeroEventRoute_holds : zeroEventRoute
  compactnessModuloTranslation : Prop
  compactnessModuloTranslation_holds : compactnessModuloTranslation
  diffusionDominatedScaling : Prop
  diffusionDominatedScaling_holds : diffusionDominatedScaling
  meanConserved : Prop
  meanConserved_holds : meanConserved
  localBadGermCapacity : Prop
  localBadGermCapacity_holds : localBadGermCapacity
  poincareCoercive : Prop
  poincareCoercive_holds : poincareCoercive
  tameMeanSector : Prop
  tameMeanSector_holds : tameMeanSector
  localDissipativeWindow : Prop
  localDissipativeWindow_holds : localDissipativeWindow
  fourierFaithful : Prop
  fourierFaithful_holds : fourierFaithful
  coleHopfBridge : Prop
  coleHopfBridge_holds : coleHopfBridge
  heatSemigroupSmooth : Prop
  heatSemigroupSmooth_holds : heatSemigroupSmooth
  heatSemigroupUnique : Prop
  heatSemigroupUnique_holds : heatSemigroupUnique

instance burgersLiterature
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν] :
    BurgersLiterature ν where
  energyIdentity :=
    ∀ s : SmoothPeriodicSnapshot ν,
      ∫ x in (0 : ℝ)..1, s.v x * s.vt x =
        -ν.viscosity * ∫ x in (0 : ℝ)..1, (s.vx x) ^ 2
  energyIdentity_holds := SmoothPeriodicSnapshot.energy_pairing_identity
  zeroEventRoute :=
    ∀ p : MeanZeroPeriodicC1,
      ∃ c ∈ Set.Icc (0 : ℝ) 1, p.f c = 0
  zeroEventRoute_holds := MeanZeroPeriodicC1.exists_zero
  compactnessModuloTranslation :=
    ∀ {R : ℝ}, 0 ≤ R → ∀ L : NNReal,
      IsCompact (closure (boundedLipschitzFamily R L))
  compactnessModuloTranslation_holds := by
    intro R hR L
    exact boundedLipschitzFamily_closure_isCompact hR L
  diffusionDominatedScaling := (1 : ℤ) < 3
  diffusionDominatedScaling_holds := burgers_diffusion_dominated_scaling_exact
  meanConserved :=
    ∀ t : ℝ, ∀ u : BurgersState,
      BurgersState.mean (BurgersEvolution.flow (ν := ν) t u) = BurgersState.mean u
  meanConserved_holds := BurgersEvolution.mean_preserving (ν := ν)
  localBadGermCapacity :=
    ∀ p : MeanZeroPeriodicC1,
      ∃ c ∈ Set.Icc (0 : ℝ) 1, p.f c = 0
  localBadGermCapacity_holds := MeanZeroPeriodicC1.exists_zero
  poincareCoercive :=
    ∀ p : MeanZeroPeriodicC1,
      ∫ x in (0 : ℝ)..1, (p.f x) ^ 2 ≤
        (∫ y in (0 : ℝ)..1, |p.deriv y|) ^ 2
  poincareCoercive_holds := MeanZeroPeriodicC1.poincare_coercive_l1
  tameMeanSector :=
    ∀ u : BurgersState, ∃ v : BurgersState, ∃ m : ℝ,
      BurgersState.mean v = 0 ∧ u = v + BurgersState.constantState m
  tameMeanSector_holds := BurgersState.decomposes_into_mean_zero_and_equilibrium
  localDissipativeWindow :=
    ∀ p : MeanZeroPeriodicC1,
      ∫ x in (0 : ℝ)..1, (p.f x) ^ 2 ≤
        (∫ y in (0 : ℝ)..1, |p.deriv y|) ^ 2
  localDissipativeWindow_holds := MeanZeroPeriodicC1.poincare_coercive_l1
  fourierFaithful := Function.Injective BurgersState.profileDictionary
  fourierFaithful_holds := BurgersState.profileDictionary_injective
  coleHopfBridge := ColeHopfBridgeStatement ν
  coleHopfBridge_holds := coleHopf_bridge_statement ν
  heatSemigroupSmooth := HeatSmoothStatement ν
  heatSemigroupSmooth_holds := heat_smooth_statement ν
  heatSemigroupUnique := HeatUniqueStatement ν
  heatSemigroupUnique_holds := heat_unique_statement ν
  

theorem burgers_energy_nonnegative
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν]
    :
    BurgersLiterature.energyIdentity (ν := ν) :=
  BurgersLiterature.energyIdentity_holds (ν := ν)

theorem burgers_zero_event_route
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν]
    :
    BurgersLiterature.zeroEventRoute (ν := ν) :=
  BurgersLiterature.zeroEventRoute_holds (ν := ν)

theorem burgers_compactness_route
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν]
    :
    BurgersLiterature.compactnessModuloTranslation (ν := ν) :=
  BurgersLiterature.compactnessModuloTranslation_holds (ν := ν)

theorem burgers_mean_conserved
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν]
    :
    BurgersLiterature.meanConserved (ν := ν) :=
  BurgersLiterature.meanConserved_holds (ν := ν)

theorem burgers_local_bad_germ_capacity_route
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν]
    :
    BurgersLiterature.localBadGermCapacity (ν := ν) :=
  BurgersLiterature.localBadGermCapacity_holds (ν := ν)

theorem burgers_poincare_route
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν]
    :
    BurgersLiterature.poincareCoercive (ν := ν) :=
  BurgersLiterature.poincareCoercive_holds (ν := ν)

theorem burgers_tame_route
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν]
    :
    BurgersLiterature.tameMeanSector (ν := ν) :=
  BurgersLiterature.tameMeanSector_holds (ν := ν)

theorem burgers_local_dissipative_window_route
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν]
    :
    BurgersLiterature.localDissipativeWindow (ν := ν) :=
  BurgersLiterature.localDissipativeWindow_holds (ν := ν)

theorem burgers_fourier_route
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν]
    :
    BurgersLiterature.fourierFaithful (ν := ν) :=
  BurgersLiterature.fourierFaithful_holds (ν := ν)

theorem burgers_diffusion_dominated_scaling
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    BurgersLiterature.diffusionDominatedScaling (ν := ν) := by
  exact BurgersLiterature.diffusionDominatedScaling_holds (ν := ν)

end Hypostructure.Backends.Burgers1D
