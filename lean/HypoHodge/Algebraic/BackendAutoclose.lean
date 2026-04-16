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

theorem hodgeAmbientCertificate
    (I : VerifiedHodgeThinInput) :
    AmbientCertificate I := by
  exact ambientCertificate I

theorem hodgeRepConFD
    (I : VerifiedHodgeThinInput) :
    CertTag.repCon ∈ backendBase I := by
  simp [backendBase]

theorem hodgeRepConCertificate
    (I : VerifiedHodgeThinInput) :
    RepresentationConservativeCertificate I := by
  exact representationConservativeCertificate I

theorem hodgeIdentityParametrizationCompleteness
    (I : VerifiedHodgeThinInput) :
    CertTag.repComp ∈ backendBase I := by
  simp [backendBase]

theorem hodgeRepCompCertificate
    (I : VerifiedHodgeThinInput) :
    RepresentationCompleteCertificate I := by
  exact representationCompleteCertificate I

theorem hodgeAutoAdmissibility
    (I : VerifiedHodgeThinInput) :
    CertTag.adm ∈ backendBase I := by
  simp [backendBase]

theorem hodgeAdmissibilityCertificate
    (I : VerifiedHodgeThinInput) :
    AdmissibilityCertificate I := by
  exact admissibilityCertificate I

theorem hodgeGermSmallnessBounded
    (I : VerifiedHodgeThinInput) :
    CertTag.germ ∈ backendBase I := by
  simp [backendBase]

theorem hodgeGermCertificate
    (I : VerifiedHodgeThinInput) :
    GermBoundedCertificate I := by
  exact germBoundedCertificate I

theorem hodgeInitialityTag
    (I : VerifiedHodgeThinInput) :
    CertTag.init ∈ backendBase I := by
  simp [backendBase]

theorem hodgeInitialityCertificate
    (I : VerifiedHodgeThinInput) :
    InitialReductionCertificate I := by
  exact initialReductionCertificate I

theorem hodgeCatLibTag
    (I : VerifiedHodgeThinInput) :
    CertTag.catLib ∈ backendBase I := by
  simp [backendBase]

theorem hodgeCatLibCertificate
    (I : VerifiedHodgeThinInput) :
    CategoryLibraryCertificate I := by
  exact (hodgeCatLibBounded I).1

theorem hodgeGammaTag
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.gamma ∈ backendBase I := by
  simp [backendBase]

def hodgeGammaCertificate
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    ProducesGamma I := by
  exact ImportedHodgeAxioms.tannakianContextSound (I := I) (gammaContextPremises I)

theorem step_subset_closure
    (rules : RuleSet) (Γ : Context) :
    step rules Γ ⊆ closure rules Γ := by
  have hstep : step rules Γ ⊆ step rules (closure rules Γ) :=
    monotone_step rules (subset_closure rules Γ)
  simpa [closure_fixed rules Γ] using hstep

theorem backend_step_eq
    (I : VerifiedHodgeThinInput) :
    step backendRules (gamma0 I) =
      insert CertTag.gamma
        (insert CertTag.catLib
          (insert CertTag.init
            (insert CertTag.germ
              (insert CertTag.adm
                (insert CertTag.repComp
                  (insert CertTag.repCon
                    (insert CertTag.ambient (gamma0 I)))))))) := by
  have hγ :
      gamma0 I =
        ({ CertTag.adj, CertTag.dE, CertTag.recN, CertTag.cMu, CertTag.scLambda,
           CertTag.scPartialC, CertTag.capH, CertTag.lsSigma,
           CertTag.tbPi, CertTag.tbO, CertTag.boundPartial } : Finset CertTag) := by
    rfl
  rw [hγ]
  native_decide

private theorem backendBase_subset_step
    (I : VerifiedHodgeThinInput) :
    backendBase I ⊆ step backendRules (gamma0 I) := by
  intro k hk
  rw [backend_step_eq]
  simp [backendBase] at hk
  rcases hk with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp [gamma0]

theorem hodgeBackendAutoclose
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    backendBase I ⊆ closure backendRules (gamma0 I) := by
  exact subset_trans (backendBase_subset_step I) (step_subset_closure backendRules (gamma0 I))

end HypoHodge.Algebraic
