import Hypostructure.Backends.NavierStokes.ProofSetup.Basic
import Hypostructure.Framework.Rigor

namespace Hypostructure.Backends.NavierStokes.ProofSetup

open Hypostructure.Framework

def importedBoundary : List RigorBoundaryItem :=
  [ { name := "AX-CKN"
      layer := .literature
      description := "Caffarelli-Kohn-Nirenberg epsilon-regularity interface" }
  , { name := "AX-VEL-EPS"
      layer := .literature
      description := "Velocity epsilon-regularity interface of Gustafson-Kang-Tsai type" }
  , { name := "AX-AB-WS"
      layer := .literature
      description := "Local weak-Serrin Type I bridge interface" }
  , { name := "AX-AB-END"
      layer := .literature
      description := "Endpoint ancient Liouville interface" }
  , { name := "AX-NRS"
      layer := .literature
      description := "Stationary self-similar rigidity interface" }
  , { name := "AX-KNSS-NS"
      layer := .literature
      description := "Axisymmetric no-swirl rigidity interface" }
  , { name := "AX-KNSS-PW"
      layer := .literature
      description := "Pointwise scale-invariant axisymmetric rigidity interface" }
  , { name := "AX-LZZ"
      layer := .literature
      description := "Finite-swirl rigidity interface" }
  , { name := "AX-LRZ"
      layer := .literature
      description := "Periodic-swirl rigidity interface" }
  , { name := "AX-GW"
      layer := .literature
      description := "Weighted-vorticity perturbative decay interface" } ]

axiom cknEpsilonRegularity
    (u : SuitableWeakSolution)
    (z : TerminalSingularPoint)
    (q : CriticalQuantityBundle) :
    q.totalCD < 1 → RegularAt u z

axiom velocityEpsilonRegularity
    (u : SuitableWeakSolution)
    (z : TerminalSingularPoint)
    (q : CriticalQuantityBundle) :
    q.C.value < 1 → RegularAt u z

axiom localWeakSerrinTypeIBridge
    (u : SuitableWeakSolution)
    (z : TerminalSingularPoint) :
    LocalPointwiseTypeIEnvelope u z → VelocityNoEscape u z

axiom endpointAncientLiouville
    (state : GeneratedSereginState) :
    TightClassPredicate state → False

axiom stationarySelfSimilarRigidity
    (state : GeneratedSereginState) :
    StationaryClassPredicate state → False

axiom axisymmetricNoSwirlLiouville
    (state : GeneratedSereginState) :
    AxisymmetricNoSwirlPredicate state → False

axiom pointwiseScaleInvariantLiouville
    (state : GeneratedSereginState) :
    PointwiseScaleInvariantPredicate state → False

axiom finiteSwirlLiouville
    (state : GeneratedSereginState) :
    FiniteSwirlPredicate state → False

axiom periodicSwirlLiouville
    (state : GeneratedSereginState) :
    PeriodicSwirlPredicate state → False

axiom weightedVorticityDecayLiouville
    (state : GeneratedSereginState) :
    WeightedVorticityPredicate state → False

end Hypostructure.Backends.NavierStokes.ProofSetup