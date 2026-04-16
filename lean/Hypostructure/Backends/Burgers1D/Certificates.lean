import Hypostructure.Backends.Burgers1D.Literature
import Hypostructure.Backends.Burgers1D.Upgrade

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Framework

noncomputable section

def energyCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    EnergyCertificate :=
  { node := .energyCheck
    payload :=
      { heightName := "mean-zero L2 energy"
        dissipationName := "nu * ||v_x||^2"
        boundStatement := BurgersLiterature.energyIdentity (ν := ν) }
    meaning := BurgersLiterature.energyIdentity (ν := ν) }

def recoveryCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    RecoveryCertificate :=
  { node := .zenoCheck
    payload :=
      { badSetName := badPatternName
        recoveryMapDeclared := false
        eventCount := 0 }
    meaning := BurgersLiterature.zeroEventRoute (ν := ν) }

def compactnessCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    CompactnessCertificate :=
  { node := .compactCheck
    payload :=
      { symmetryGroupName := "translations of T"
        quotientName := "periodic profiles modulo translation"
        profileName := equilibriumManifoldName
        compactnessStatement := BurgersLiterature.compactnessModuloTranslation (ν := ν) }
    meaning := BurgersLiterature.compactnessModuloTranslation (ν := ν) }

def scalingCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    ScalingCertificate :=
  { node := .scaleCheck
    payload := { alpha := 1, beta := 3, route := "diffusion-dominated local scaling" }
    meaning := BurgersLiterature.diffusionDominatedScaling (ν := ν) }

def parameterCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    ParameterCertificate :=
  { node := .paramCheck
    payload :=
      { parameterSpace := "{(nu, mean)}"
        referencePoint := "(nu, mean(u0))"
        stableStatement := BurgersLiterature.meanConserved (ν := ν) }
    meaning := BurgersLiterature.meanConserved (ν := ν) }

def capacityCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    CapacityCertificate :=
  { node := .geomCheck
    payload :=
      { singularSetName := "local bad-germ support"
        capacityValue := 0
        negligible := BurgersLiterature.localBadGermCapacity (ν := ν) }
    meaning := BurgersLiterature.localBadGermCapacity (ν := ν) }

def stiffnessCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    StiffnessCertificate :=
  { node := .stiffnessCheck
    payload :=
      { gapConstant := ν.viscosity
        exponent := (1 : ℝ) / 2
        coercivityStatement := BurgersLiterature.poincareCoercive (ν := ν) }
    meaning := BurgersLiterature.poincareCoercive (ν := ν) }

def topologyCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    TopologyCertificate :=
  { node := .topoCheck
    payload :=
      { invariantName := "mean"
        sectorStatement := BurgersLiterature.meanConserved (ν := ν) }
    meaning := BurgersLiterature.meanConserved (ν := ν) }

def tamenessCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    TamenessCertificate :=
  { node := .tameCheck
    payload :=
      { structureName := "semialgebraic/real-analytic mean-sector structure"
        stratificationBound := 1
        tameStatement := BurgersLiterature.tameMeanSector (ν := ν) }
    meaning := BurgersLiterature.tameMeanSector (ν := ν) }

def mixingCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    MixingCertificate :=
  { node := .ergoCheck
    payload :=
      { invariantMeasureName := "mean-sector local dissipative window"
        mixingTimeFinite := BurgersLiterature.localDissipativeWindow (ν := ν)
        convergenceStatement := BurgersLiterature.localDissipativeWindow (ν := ν) }
    meaning := BurgersLiterature.localDissipativeWindow (ν := ν) }

def representationCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    RepresentationCertificate :=
  { node := .complexCheck
    payload :=
      { languageName := "continuous periodic profiles"
        dictionaryName := "periodic profile dictionary"
        faithfulStatement := BurgersLiterature.fourierFaithful (ν := ν) }
    meaning := BurgersLiterature.fourierFaithful (ν := ν) }

def gradientCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    GradientCertificate :=
  { node := .oscillateCheck
    payload :=
      { metricName := "mean-zero L2 metric"
        vectorFieldName := "viscous Burgers vector field"
        monotonicityStatement := BurgersLiterature.energyIdentity (ν := ν) }
    meaning := BurgersLiterature.energyIdentity (ν := ν) }

def boundaryClosedCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    BoundaryClosedCertificate :=
  { node := .boundaryCheck
    payload := { reason := "periodic torus: closed-system branch" }
    witnessFound := True }

def germCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    GermCertificate :=
  { node := .compactCheck
    payload := { libraryName := "Burgers blow-up germ package", smallnessWitness := True }
    meaning :=
      @BurgersBadPatternPackage.germSmallness ν
        (inferInstance : BurgersBadPatternPackage ν) }

def initialityCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    InitialityCertificate :=
  { node := .lock
    payload := { universalBadName := "universal Burgers bad object", initialityWitness := True }
    meaning :=
      @BurgersBadPatternPackage.universalBadInitialized ν
        (inferInstance : BurgersBadPatternPackage ν) }

def catLibCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    CatLibCertificate :=
  { node := .lock
    payload := { libraryName := "finite Burgers bad-pattern library", completenessWitness := True }
    meaning :=
      @BurgersBadPatternPackage.catLibraryComplete ν
        (inferInstance : BurgersBadPatternPackage ν) }

def coleHopfCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    ColeHopfCertificate :=
  { node := .oscillateCheck
    payload :=
      { transformName := "Cole-Hopf"
        targetSemigroup := "positive heat semigroup"
        bridgeStatement := BurgersBridgeInvariantStatement ν }
    meaning := BurgersBridgeInvariantStatement ν }

def heatSmoothCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    HeatSmoothCertificate :=
  { node := .lock
    payload :=
      { semigroupName := "heat semigroup"
        smoothingStatement :=
          HeatEnergyContractiveStatement ν ∧ HeatDissipationContractiveStatement ν
        uniquenessStatement :=
          HeatUniqueStatement ν ∧ HeatEquilibriumStatement ν }
    meaning :=
      (HeatEnergyContractiveStatement ν ∧ HeatDissipationContractiveStatement ν) ∧
      (HeatUniqueStatement ν ∧ HeatEquilibriumStatement ν) }

def structRegCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    StructRegCertificate :=
  { node := .lock
    payload :=
      { backendName := "Burgers1D"
        obstructionEmpty :=
          @BurgersAnalyticUpgradeInputStatement ν
            (inferInstance : BurgersBadPatternPackage ν)
            (inferInstance : PeriodicHeatSemigroupPackage ν)
            (inferInstance : ColeHopfPackage ν) }
    meaning :=
      @BurgersStructuralExclusionStatement ν
        (inferInstance : BurgersBadPatternPackage ν)
        (inferInstance : PeriodicHeatSemigroupPackage ν)
        (inferInstance : ColeHopfPackage ν) }

def analyticRegularityCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    AnalyticRegularityCertificate :=
  { node := .lock
    payload :=
      { backendName := "Burgers1D"
        targetClaim :=
          @BurgersAnalyticUpgradeInputStatement ν
            (inferInstance : BurgersBadPatternPackage ν)
            (inferInstance : PeriodicHeatSemigroupPackage ν)
            (inferInstance : ColeHopfPackage ν) →
          BurgersGlobalRegularityStatement ν }
    meaning :=
      @BurgersAnalyticUpgradeInputStatement ν
        (inferInstance : BurgersBadPatternPackage ν)
        (inferInstance : PeriodicHeatSemigroupPackage ν)
        (inferInstance : ColeHopfPackage ν) →
      BurgersGlobalRegularityStatement ν }

def lyapunovCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    LyapunovCertificate :=
  { node := .stiffnessCheck
    payload :=
      { functionName := "L(u)=1/2 ||u-mean(u)||^2"
        minimumSetName := equilibriumManifoldName
        monotoneStatement := BurgersLiterature.energyIdentity (ν := ν) }
    meaning := BurgersLiterature.energyIdentity (ν := ν) }

def jacobiCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    JacobiCertificate :=
  { node := .stiffnessCheck
    payload :=
      { metricName := "g_D"
        distanceName := "dist_gD"
        comparisonStatement := BurgersLiterature.poincareCoercive (ν := ν) }
    meaning := BurgersLiterature.poincareCoercive (ν := ν) }

def hamiltonJacobiCertificate
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν] :
    HamiltonJacobiCertificate :=
  { node := .stiffnessCheck
    payload :=
      { functionName := "L"
        gradientName := "grad L"
        relationStatement := BurgersLiterature.energyIdentity (ν := ν) }
    meaning := BurgersLiterature.energyIdentity (ν := ν) }

end

end Hypostructure.Backends.Burgers1D
