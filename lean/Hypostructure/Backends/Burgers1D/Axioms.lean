import Hypostructure.Backends.Burgers1D.Final

namespace Hypostructure.Backends.Burgers1D

theorem burgers_analytic_regularity_axiomatic
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν]
    [BurgersAnalyticUpgradePackage ν] :
    BurgersGlobalRegularityStatement ν :=
  burgers_analytic_regularity ν

theorem burgers_final_certificate_sound_axiomatic
    (ν : BurgersParameters)
    [BurgersPDEEvolutionPackage ν]
    [BurgersBadPatternPackage ν]
    [PeriodicHeatSemigroupPackage ν]
    [ColeHopfPackage ν]
    [BurgersLiterature ν]
    [BurgersAnalyticUpgradePackage ν] :
    (burgersPositiveRoute ν).analyticReg.payload.targetClaim :=
  burgers_final_certificate_sound ν

end Hypostructure.Backends.Burgers1D
