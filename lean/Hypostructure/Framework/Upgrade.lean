import Hypostructure.Framework.Route

namespace Hypostructure.Framework

/-- Framework-level names for the documented Lock exclusion tactics. These
identifiers are reusable: a backend chooses one tactic and then supplies the
semantic obstruction proof required by that tactic. -/
inductive ExclusionTacticId
  | E1_dimension
  | E2_invariant
  | E3_positivity
  | E4_integrality
  | E5_functional
  | E6_causal
  | E7_thermodynamic
  | E8_holographic
  | E9_ergodic
  | E10_definability
  | E11_galoisMonodromy
  | E12_algebraicCompressibility
  | E13_algorithmicCompleteness
  deriving DecidableEq, Repr, Fintype

def ExclusionTacticId.label : ExclusionTacticId → String
  | .E1_dimension => "E1 Dimension"
  | .E2_invariant => "E2 Invariant"
  | .E3_positivity => "E3 Positivity"
  | .E4_integrality => "E4 Integrality"
  | .E5_functional => "E5 Functional"
  | .E6_causal => "E6 Causal"
  | .E7_thermodynamic => "E7 Thermodynamic"
  | .E8_holographic => "E8 Holographic"
  | .E9_ergodic => "E9 Ergodic"
  | .E10_definability => "E10 Definability"
  | .E11_galoisMonodromy => "E11 Galois-Monodromy"
  | .E12_algebraicCompressibility => "E12 Algebraic Compressibility"
  | .E13_algorithmicCompleteness => "E13 Algorithmic Completeness"

/-- Certificate-name accounting shared by Lock tactic and analytic-upgrade
dossiers. The meanings of the certificates stay backend-specific. -/
def CertificatesPresent
    (C : FinalCertificateChain)
    (names : List String) : Prop :=
  ∀ name ∈ names, name ∈ C.certificates

/-- A reusable certificate for one Lock tactic. The obstruction proposition is
left as a backend parameter, because the mathematical content of E1-E13 depends
on the problem domain. -/
structure LockTacticCertificate where
  tactic : ExclusionTacticId
  certificateName : String
  requiredCertificates : List String
  preservationCertificates : List String := []
  obstruction : Prop

/-- A verified Lock tactic dossier for a concrete route. This proves that the
chosen tactic certificate and all required support certificates are present,
and that the route records the blocked Lock certificate. -/
structure LockTacticDossier
    (R : TraceBackedRoute) where
  certificate : LockTacticCertificate
  tacticCertificatePresent :
    certificate.certificateName ∈ R.finalChain.certificates
  requiredCertificatesPresent :
    CertificatesPresent R.finalChain certificate.requiredCertificates
  preservationCertificatesPresent :
    CertificatesPresent R.finalChain certificate.preservationCertificates
  obstructionProof : certificate.obstruction
  blockedCertificatePresent :
    R.lockCertificate ∈ R.finalChain.certificates

def LockTacticDossier.lockBlocked
    {R : TraceBackedRoute}
    (_D : LockTacticDossier R) : Prop :=
  R.lockCertificate ∈ R.finalChain.certificates

theorem LockTacticDossier.lockBlocked_holds
    {R : TraceBackedRoute}
    (D : LockTacticDossier R) :
    D.lockBlocked :=
  D.blockedCertificatePresent

/-- Reusable shape of the downstream analytic upgrade. The framework records
the local certificates, bridge certificates, structural Lock certificate, and
target certificate. The actual target theorem remains backend-proved. -/
structure LocalToTargetUpgradeDossier
    (R : TraceBackedRoute)
    (Target : Prop) where
  name : String
  targetCertificate : String
  structuralCertificate : String
  localCertificateNames : List String
  bridgeCertificateNames : List String
  lockCertificatePresent :
    R.lockCertificate ∈ R.finalChain.certificates
  structuralCertificatePresent :
    structuralCertificate ∈ R.finalChain.certificates
  localCertificatesPresent :
    CertificatesPresent R.finalChain localCertificateNames
  bridgeCertificatesPresent :
    CertificatesPresent R.finalChain bridgeCertificateNames
  targetCertificatePresent :
    targetCertificate ∈ R.finalChain.certificates
  nonCircular :
    targetCertificate ∉
      localCertificateNames ++ bridgeCertificateNames ++
        [R.lockCertificate, structuralCertificate]
  upgrade : Target

theorem LocalToTargetUpgradeDossier.target
    {R : TraceBackedRoute}
    {Target : Prop}
    (D : LocalToTargetUpgradeDossier R Target) :
    Target :=
  D.upgrade

/-- Reusable pointwise target shape for a posteriori upgrades. A backend picks
an object type `α`, an admissibility predicate, and the witness it ultimately
wants for every admissible object. -/
def PointwiseTarget
    {α : Type u}
    (Admissible Witness : α → Prop) : Prop :=
  ∀ x : α, Admissible x → Witness x

/-- Reusable a-posteriori localization principle. It says that any missing
target witness for an admissible object can be localized to a bad morphism in
the Lock target class. -/
def APosterioriLocalization
    {α : Type u}
    (Admissible Witness : α → Prop)
    (BadMorphism : Prop) : Prop :=
  ∀ x : α, Admissible x → ¬ Witness x → BadMorphism

/-- General a-posteriori upgrade metatheorem. If every missing witness
localizes to a bad morphism and the Lock blocks bad morphisms, then every
admissible object has the desired witness. -/
theorem APosterioriLocalization.target
    {α : Type u}
    {Admissible Witness : α → Prop}
    {BadMorphism : Prop}
    (lockBlocks : ¬ BadMorphism)
    (localizes : APosterioriLocalization Admissible Witness BadMorphism) :
    PointwiseTarget Admissible Witness := by
  classical
  intro x hx
  by_contra hmissing
  exact lockBlocks (localizes x hx hmissing)

/-- Pointwise form of the a-posteriori upgrade. This is the local-estimates
version used by backends that build a fresh local certificate package for the
specific object currently being proved regular. -/
theorem APosterioriLocalization.point
    {α : Type u}
    {Admissible Witness : α → Prop}
    {BadMorphism : Prop}
    {x : α}
    (lockBlocks : ¬ BadMorphism)
    (localizesAt : Admissible x → ¬ Witness x → BadMorphism)
    (hx : Admissible x) :
    Witness x := by
  classical
  by_contra hmissing
  exact lockBlocks (localizesAt hx hmissing)

/-- Dossier form of the reusable a-posteriori upgrade, useful for backends that
want to package the Lock obstruction and localization theorem together before
passing the result into a target-specific final claim. -/
structure APosterioriLocalizationDossier
    (α : Type u) where
  name : String
  admissible : α → Prop
  witness : α → Prop
  badMorphism : Prop
  lockBlocks : ¬ badMorphism
  localizes : APosterioriLocalization admissible witness badMorphism

theorem APosterioriLocalizationDossier.target
    {α : Type u}
    (D : APosterioriLocalizationDossier α) :
    PointwiseTarget D.admissible D.witness :=
  APosterioriLocalization.target D.lockBlocks D.localizes

/-- Prop-level variant for final goals that are not naturally pointwise. -/
def APosterioriTargetLocalization
    (Target BadMorphism : Prop) : Prop :=
  ¬ Target → BadMorphism

theorem APosterioriTargetLocalization.target
    {Target BadMorphism : Prop}
    (lockBlocks : ¬ BadMorphism)
    (localizes : APosterioriTargetLocalization Target BadMorphism) :
    Target := by
  classical
  by_contra hmissing
  exact lockBlocks (localizes hmissing)

def TraceBackedRouteProof.targetClaimOfUpgrade
    {R : TraceBackedRoute}
    (P : TraceBackedRouteProof R)
    {Target : Prop}
    (D : LocalToTargetUpgradeDossier R Target) :
    TraceBackedTargetClaim R Target :=
  P.targetClaim D.target

/-- A complete reusable upgrade package: one E1-E13 Lock tactic blocks the bad
route, then the local/bridge/Lock dossier upgrades to the backend target. -/
structure CertifiedUpgradeDossier
    (R : TraceBackedRoute)
    (Target : Prop) where
  lock : LockTacticDossier R
  upgrade : LocalToTargetUpgradeDossier R Target

theorem CertifiedUpgradeDossier.target
    {R : TraceBackedRoute}
    {Target : Prop}
    (D : CertifiedUpgradeDossier R Target) :
    Target :=
  D.upgrade.target

def TraceBackedRouteProof.targetClaimOfCertifiedUpgrade
    {R : TraceBackedRoute}
    (P : TraceBackedRouteProof R)
    {Target : Prop}
    (D : CertifiedUpgradeDossier R Target) :
    TraceBackedTargetClaim R Target :=
  P.targetClaim D.target

end Hypostructure.Framework
