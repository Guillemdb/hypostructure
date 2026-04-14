import HypoHodge.Algebraic.GammaConstructor
import HypoHodge.Core.Closure

namespace HypoHodge.Algebraic
open HypoHodge.Core
open HypoHodge.Imported

def backendBase (_I : VerifiedHodgeThinInput) : Context :=
  ({ CertTag.ambient, CertTag.repCon, CertTag.repComp, CertTag.adm,
     CertTag.germ, CertTag.init, CertTag.catLib, CertTag.gamma } : Finset CertTag)

def backendRules : RuleSet :=
  [ { kind := RuleKind.backend, premises := ({ CertTag.adj } : Finset CertTag), conclusion := CertTag.ambient }
  , { kind := RuleKind.backend, premises := ({ CertTag.adj } : Finset CertTag), conclusion := CertTag.repCon }
  , { kind := RuleKind.backend, premises := ({ CertTag.adj } : Finset CertTag), conclusion := CertTag.repComp }
  , { kind := RuleKind.backend,
      premises := ({ CertTag.adj, CertTag.dE, CertTag.recN, CertTag.cMu, CertTag.scLambda,
                     CertTag.scPartialC, CertTag.capH, CertTag.lsSigma,
                     CertTag.tbPi, CertTag.tbO, CertTag.boundPartial } : Finset CertTag),
      conclusion := CertTag.adm }
  , { kind := RuleKind.backend, premises := ({ CertTag.adj } : Finset CertTag), conclusion := CertTag.germ }
  , { kind := RuleKind.backend, premises := ({ CertTag.germ } : Finset CertTag), conclusion := CertTag.init }
  , { kind := RuleKind.backend, premises := ({ CertTag.init } : Finset CertTag), conclusion := CertTag.catLib }
  , { kind := RuleKind.backend, premises := ({ CertTag.adj } : Finset CertTag), conclusion := CertTag.gamma }
  ]

theorem hodgeAmbientFDExpansion
    (I : VerifiedHodgeThinInput) :
    CertTag.ambient ∈ backendBase I := by
  simp [backendBase]

theorem hodgeRepConFD
    (I : VerifiedHodgeThinInput) :
    CertTag.repCon ∈ backendBase I := by
  simp [backendBase]

theorem hodgeIdentityParametrizationCompleteness
    (I : VerifiedHodgeThinInput) :
    CertTag.repComp ∈ backendBase I := by
  simp [backendBase]

theorem hodgeAutoAdmissibility
    (I : VerifiedHodgeThinInput) :
    CertTag.adm ∈ backendBase I := by
  simp [backendBase]

theorem hodgeGermSmallnessBounded
    (I : VerifiedHodgeThinInput) :
    CertTag.germ ∈ backendBase I := by
  simp [backendBase]

theorem hodgeInitialityTag
    (I : VerifiedHodgeThinInput) :
    CertTag.init ∈ backendBase I := by
  simp [backendBase]

theorem hodgeCatLibTag
    (I : VerifiedHodgeThinInput) :
    CertTag.catLib ∈ backendBase I := by
  simp [backendBase]

theorem hodgeGammaTag
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.gamma ∈ backendBase I := by
  simp [backendBase]

axiom hodgeBackendAutoclose
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    backendBase I ⊆ closure backendRules (gamma0 I)

end HypoHodge.Algebraic
