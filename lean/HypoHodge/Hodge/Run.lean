import HypoHodge.Hodge.Permits
import HypoHodge.Algebraic.BackendAutoclose
import HypoHodge.Core.Closure

set_option linter.constructorNameAsVariable false
set_option maxRecDepth 20000
set_option maxHeartbeats 800000

namespace HypoHodge.Hodge
open HypoHodge.Core
open HypoHodge.Algebraic
open HypoHodge.Imported

def initialContext (I : VerifiedHodgeThinInput) : Context :=
  gamma0 I

def contextAfterBackend (I : VerifiedHodgeThinInput) : Context :=
  closure backendRules (initialContext I)

def contextAfterMhs (I : VerifiedHodgeThinInput) : Context :=
  closure [permitHdg] (contextAfterBackend I)

def contextAfterTann (I : VerifiedHodgeThinInput) : Context :=
  closure [permitTann] (contextAfterMhs I)

def finalContext (I : VerifiedHodgeThinInput) : Context :=
  closure [permitLock] (contextAfterTann I)

def runHodgeSystem (I : VerifiedHodgeThinInput) : Context :=
  finalContext I

private theorem backend_tag
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I]
    {k : CertTag}
    (hk : k ∈ backendBase I) :
    k ∈ contextAfterBackend I := by
  change k ∈ closure backendRules (gamma0 I)
  exact hodgeBackendAutoclose I hk

private theorem local_tag
    (I : VerifiedHodgeThinInput)
    {k : CertTag}
    (hk : k ∈ initialContext I) :
    k ∈ contextAfterBackend I := by
  change k ∈ closure backendRules (initialContext I)
  exact (subset_closure backendRules (initialContext I)) hk

theorem emit_backend_context
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    backendBase I ⊆ contextAfterBackend I := by
  intro k hk
  exact backend_tag I hk

private theorem permitHdg_enabled
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    permitHdg.enabled (contextAfterBackend I) := by
  intro k hk
  simp [permitHdg] at hk
  rcases hk with rfl | rfl | rfl | rfl
  · exact backend_tag I (by simp [backendBase])
  · exact local_tag I (by simp [initialContext, gamma0])
  · exact local_tag I (by simp [initialContext, gamma0])
  · exact local_tag I (by simp [initialContext, gamma0])

theorem emit_mhs
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.mhs ∈ contextAfterMhs I := by
  have hStep : step [permitHdg] (contextAfterBackend I) ⊆ contextAfterMhs I :=
    step_subset_closure [permitHdg] (contextAfterBackend I)
  have hmhs : CertTag.mhs ∈ step [permitHdg] (contextAfterBackend I) := by
    change CertTag.mhs ∈ fireRule permitHdg (contextAfterBackend I)
    rw [fireRule_eq_insert_of_enabled _ _ (permitHdg_enabled I)]
    simpa
  exact hStep hmhs

def emit_mhs_semantics
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    ProducesMHS I := by
  exact ImportedHodgeAxioms.hodgeBridgeSound (I := I)
    ⟨ImportedHodgeAxioms.classicalExtraction (I := I),
      ambientCertificate I,
      thinBoundaryPiCertificate I,
      tameLambdaCertificate I,
      defectEnergyCertificate I⟩

private theorem mhs_lifts_backend
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I]
    {k : CertTag}
    (hk : k ∈ contextAfterBackend I) :
    k ∈ contextAfterMhs I := by
  change k ∈ closure [permitHdg] (contextAfterBackend I)
  exact (subset_closure [permitHdg] (contextAfterBackend I)) hk

private theorem permitTann_enabled
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    permitTann.enabled (contextAfterMhs I) := by
  intro k hk
  simp [permitTann] at hk
  rcases hk with rfl | rfl | rfl
  · exact mhs_lifts_backend I (backend_tag I (by simp [backendBase]))
  · exact mhs_lifts_backend I (backend_tag I (by simp [backendBase]))
  · exact mhs_lifts_backend I (backend_tag I (by simp [backendBase]))

theorem emit_tann
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.tann ∈ contextAfterTann I := by
  have hStep : step [permitTann] (contextAfterMhs I) ⊆ contextAfterTann I :=
    step_subset_closure [permitTann] (contextAfterMhs I)
  have htann : CertTag.tann ∈ step [permitTann] (contextAfterMhs I) := by
    change CertTag.tann ∈ fireRule permitTann (contextAfterMhs I)
    rw [fireRule_eq_insert_of_enabled _ _ (permitTann_enabled I)]
    simpa
  exact hStep htann

def emit_tann_semantics
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    ProducesTann I := by
  exact ImportedHodgeAxioms.tannakianBridgeSound (I := I)
    ⟨admissibilityCertificate I,
      hodgeGammaCertificate I,
      categoryLibraryCertificate I⟩

private theorem tann_lifts_mhs
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I]
    {k : CertTag}
    (hk : k ∈ contextAfterMhs I) :
    k ∈ contextAfterTann I := by
  change k ∈ closure [permitTann] (contextAfterMhs I)
  exact (subset_closure [permitTann] (contextAfterMhs I)) hk

private theorem permitLock_enabled
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    permitLock.enabled (contextAfterTann I) := by
  intro k hk
  simp [permitLock] at hk
  rcases hk with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact tann_lifts_mhs I (emit_mhs I)
  · exact emit_tann I
  · exact tann_lifts_mhs I (mhs_lifts_backend I (local_tag I (by simp [initialContext, gamma0])))
  · exact tann_lifts_mhs I (mhs_lifts_backend I (local_tag I (by simp [initialContext, gamma0])))
  · exact tann_lifts_mhs I (mhs_lifts_backend I (local_tag I (by simp [initialContext, gamma0])))
  · exact tann_lifts_mhs I (mhs_lifts_backend I (backend_tag I (by simp [backendBase])))
  · exact tann_lifts_mhs I (mhs_lifts_backend I (backend_tag I (by simp [backendBase])))
  · exact tann_lifts_mhs I (mhs_lifts_backend I (backend_tag I (by simp [backendBase])))

theorem emit_catHomBlk
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.catHomBlk ∈ finalContext I := by
  have hStep : step [permitLock] (contextAfterTann I) ⊆ finalContext I :=
    step_subset_closure [permitLock] (contextAfterTann I)
  have hblk : CertTag.catHomBlk ∈ step [permitLock] (contextAfterTann I) := by
    change CertTag.catHomBlk ∈ fireRule permitLock (contextAfterTann I)
    rw [fireRule_eq_insert_of_enabled _ _ (permitLock_enabled I)]
    simpa
  exact hStep hblk

theorem kernel_derives_catHomBlk
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.catHomBlk ∈ runHodgeSystem I := by
  exact emit_catHomBlk I

def emit_catHomBlk_semantics
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    TargetRealizedByCycle I := by
  refine ⟨(emit_tann_semantics I).liftedCycle, ?_⟩
  exact (emit_tann_semantics I).realizesTarget
    (emit_mhs_semantics I)
    (cycleMultiplicityCertificate I)
    (linearStabilityCertificate I)
    (thinBoundaryOCertificate I)
    (categoryLibraryCertificate I)
    (initialReductionCertificate I)
    (admissibilityCertificate I)

end HypoHodge.Hodge
