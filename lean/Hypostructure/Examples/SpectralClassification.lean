import Hypostructure.Problem.SingularityClassification
import Hypostructure.Framework.Document
import Hypostructure.Sieve.GenericProgram
import Mathlib.LinearAlgebra.Charpoly.ToMatrix
import Mathlib.LinearAlgebra.Matrix.HermitianFunctionalCalculus

namespace Hypostructure.Examples

open Matrix
open Hypostructure.Framework
open Hypostructure.Problem
open Hypostructure.Sieve

section

variable {n : Type*} [Fintype n] [DecidableEq n]

abbrev SpectralProfile (n : Type*) [Fintype n] [DecidableEq n] :=
  { A : Matrix n n ℂ // A.IsHermitian }

abbrev SpectralPolynomialClass := Polynomial ℂ

def SpectralUnitaryEquivalent (A B : SpectralProfile n) : Prop :=
  ∃ U : Matrix.unitaryGroup n ℂ, B.1 = (U : Matrix n n ℂ) * A.1 * star (U : Matrix n n ℂ)

abbrev SpectralRankClass (n : Type*) [Fintype n] [DecidableEq n] :=
  Fin (Fintype.card n + 1)

noncomputable def spectralRankClass (A : SpectralProfile n) : SpectralRankClass n :=
  ⟨Fintype.card { i : n // A.2.eigenvalues i ≠ 0 },
    Nat.lt_succ_of_le (Fintype.card_subtype_le fun i : n => A.2.eigenvalues i ≠ 0)⟩

abbrev SpectralParameters (k : SpectralRankClass n) :=
  { f : n → ℝ // Fintype.card { i : n // f i ≠ 0 } = k.val }

noncomputable def spectralCoordinates (A : SpectralProfile n) :
    SpectralParameters (spectralRankClass A) :=
  ⟨A.2.eigenvalues, by
    simp [SpectralParameters, spectralRankClass]⟩

noncomputable def spectralTameStratification :
    TameProfileStratification (SpectralProfile n) where
  Stratum := SpectralRankClass n
  instFintypeStratum := inferInstance
  instDecidableEqStratum := inferInstance
  Param := SpectralParameters
  dim := fun k => k.val
  assign := spectralRankClass
  coord := spectralCoordinates

noncomputable def spectralNormalForm (A : SpectralProfile n) : Matrix n n ℂ :=
  diagonal (fun i => RCLike.ofReal (A.2.eigenvalues i))

noncomputable def spectralPolynomialClass (A : SpectralProfile n) : SpectralPolynomialClass :=
  A.1.charpoly

noncomputable instance spectralHasProfileClassification :
    HasProfileClassification (SpectralProfile n) where
  classification := .stratified spectralTameStratification

def spectralRankTable : Finset (SpectralRankClass n) :=
  Finset.univ

theorem spectral_profileClassification_exhaustive :
    (∃ L : FiniteProfileLibrary (SpectralProfile n),
        profileClassification (SpectralProfile n) = .library L) ∨
      (∃ S : TameProfileStratification (SpectralProfile n),
        profileClassification (SpectralProfile n) = .stratified S) ∨
      (∃ W : WildProfileWitness (SpectralProfile n),
        profileClassification (SpectralProfile n) = .wild W) ∨
      ∃ O : InconclusiveProfileWitness (SpectralProfile n),
        profileClassification (SpectralProfile n) = .inconclusive O :=
    by
      exact Or.inr <| Or.inl ⟨spectralTameStratification, rfl⟩

theorem spectral_profileClassification_is_stratified :
    profileClassification (SpectralProfile n) = .stratified spectralTameStratification := by
  rfl

theorem spectral_coordinates_live_in_assigned_stratum
    (A : SpectralProfile n) :
    (spectralTameStratification.coord A).1 = A.2.eigenvalues := by
  rfl

theorem spectral_coordinates_count_nonzero
    (A : SpectralProfile n) :
    Fintype.card { i : n // (spectralTameStratification.coord A).1 i ≠ 0 } =
      (spectralTameStratification.assign A).val := by
  simpa [spectralTameStratification] using (spectralCoordinates A).2

theorem spectralRankClass_mem_table
    (A : SpectralProfile n) :
    spectralRankClass A ∈ spectralRankTable (n := n) := by
  simp [spectralRankTable]

theorem spectralRankClassifies
    (A : SpectralProfile n) :
    A.1.rank = (spectralRankClass A).val := by
  simpa [spectralRankClass] using A.2.rank_eq_card_non_zero_eigs

theorem spectral_profileClassification_refines_to_rank_table
    (A : SpectralProfile n) :
    ∃ k ∈ spectralRankTable (n := n), A.1.rank = k.val := by
  refine ⟨spectralRankClass A, spectralRankClass_mem_table A, spectralRankClassifies A⟩

theorem spectral_normal_form_diagonalization
    (A : SpectralProfile n) :
    A.1 =
      (A.2.eigenvectorUnitary : Matrix n n ℂ) * spectralNormalForm A *
        star (A.2.eigenvectorUnitary : Matrix n n ℂ) := by
  simpa [spectralNormalForm] using A.2.spectral_theorem

theorem spectral_profileClassification_refines_to_normal_form
    (A : SpectralProfile n) :
    ∃ U : Matrix n n ℂ,
      A.1 = U * spectralNormalForm A * star U := by
  refine ⟨A.2.eigenvectorUnitary, ?_⟩
  simpa using spectral_normal_form_diagonalization A

theorem spectral_spectrum_exhaustive
    (A : SpectralProfile n) :
    spectrum ℝ A.1 = Set.range A.2.eigenvalues := by
  simpa using A.2.eigenvalues_eq_spectrum_real

theorem matrix_charpoly_unitary_conj
    (U : Matrix.unitaryGroup n ℂ) (A : Matrix n n ℂ) :
    ((((U : Matrix n n ℂ) * A) * star (U : Matrix n n ℂ))).charpoly = A.charpoly := by
  let Uc : Matrix n n (Polynomial ℂ) :=
    (Polynomial.C : ℂ →+* Polynomial ℂ).mapMatrix (U : Matrix n n ℂ)
  let UcStar : Matrix n n (Polynomial ℂ) :=
    (Polynomial.C : ℂ →+* Polynomial ℂ).mapMatrix (star (U : Matrix n n ℂ))
  have hUU : Uc * UcStar = 1 := by
    dsimp [Uc, UcStar]
    rw [← Matrix.map_mul]
    simpa using
      congrArg ((Polynomial.C : ℂ →+* Polynomial ℂ).mapMatrix) (unitary.coe_mul_star_self U)
  let CA : Matrix n n (Polynomial ℂ) := (Polynomial.C : ℂ →+* Polynomial ℂ).mapMatrix A
  calc
    ((((U : Matrix n n ℂ) * A) * star (U : Matrix n n ℂ))).charpoly
        = det (scalar n Polynomial.X - Uc * CA * UcStar) := by
            simp [Matrix.charpoly, Matrix.charmatrix, Uc, UcStar, CA, Matrix.map_mul, mul_assoc]
    _ = det (scalar n Polynomial.X * Uc * UcStar - Uc * CA * UcStar) := by
          rw [Matrix.mul_assoc ((scalar n) Polynomial.X), hUU, Matrix.mul_one]
    _ = det (Uc * scalar n Polynomial.X * UcStar - Uc * CA * UcStar) := by
          rw [scalar_commute _ (fun p => Polynomial.commute_X p)]
    _ = det (Uc * (scalar n Polynomial.X - CA) * UcStar) := by
          rw [← Matrix.sub_mul, ← Matrix.mul_sub]
    _ = det Uc * det (scalar n Polynomial.X - CA) *
          det UcStar := by
          rw [det_mul, det_mul]
    _ = det Uc * det UcStar * det (scalar n Polynomial.X - CA) := by ring
    _ = det (scalar n Polynomial.X - CA) := by
          rw [← det_mul, hUU, det_one, one_mul]
    _ = A.charpoly := rfl

theorem spectral_charpoly_eq_normal_form
    (A : SpectralProfile n) :
    A.1.charpoly = (spectralNormalForm A).charpoly := by
  have h := matrix_charpoly_unitary_conj (A.2.eigenvectorUnitary) (spectralNormalForm A)
  rw [← spectral_normal_form_diagonalization A] at h
  exact h

theorem spectral_polynomialClass_eq_charpoly
    (A : SpectralProfile n) :
    spectralPolynomialClass A = A.1.charpoly := by
  rfl

theorem spectral_normal_form_has_same_polynomialClass
    (A : SpectralProfile n) :
    (spectralNormalForm A).charpoly = spectralPolynomialClass A := by
  simpa [spectralPolynomialClass] using (spectral_charpoly_eq_normal_form A).symm

theorem spectral_coordinates_determine_normal_form
    {A B : SpectralProfile n}
    (h : (spectralTameStratification.coord A).1 = (spectralTameStratification.coord B).1) :
    spectralNormalForm A = spectralNormalForm B := by
  ext i j
  by_cases hij : i = j
  · subst hij
    have hi := congrFun h i
    simpa [spectralTameStratification, spectralCoordinates, spectralNormalForm]
      using congrArg RCLike.ofReal hi
  · simp [spectralNormalForm, hij]

theorem spectral_coordinates_determine_polynomialClass
    {A B : SpectralProfile n}
    (h : (spectralTameStratification.coord A).1 = (spectralTameStratification.coord B).1) :
    spectralPolynomialClass A = spectralPolynomialClass B := by
  calc
    spectralPolynomialClass A = (spectralNormalForm A).charpoly := by
      exact (spectral_normal_form_has_same_polynomialClass A).symm
    _ = (spectralNormalForm B).charpoly := by
      exact congrArg Matrix.charpoly (spectral_coordinates_determine_normal_form h)
    _ = spectralPolynomialClass B := by
      exact spectral_normal_form_has_same_polynomialClass B

theorem spectral_normal_form_complete
    {A B : SpectralProfile n}
    (h : spectralNormalForm A = spectralNormalForm B) :
    SpectralUnitaryEquivalent A B := by
  let U : Matrix.unitaryGroup n ℂ := B.2.eigenvectorUnitary * star (A.2.eigenvectorUnitary)
  refine ⟨U, ?_⟩
  calc
    B.1 = (B.2.eigenvectorUnitary : Matrix n n ℂ) * spectralNormalForm B *
          star (B.2.eigenvectorUnitary : Matrix n n ℂ) := by
            simpa using spectral_normal_form_diagonalization B
    _ = (B.2.eigenvectorUnitary : Matrix n n ℂ) * spectralNormalForm A *
          star (B.2.eigenvectorUnitary : Matrix n n ℂ) := by rw [h]
    _ = (U : Matrix n n ℂ) * A.1 * star (U : Matrix n n ℂ) := by
          let UA : Matrix.unitaryGroup n ℂ := A.2.eigenvectorUnitary
          let UB : Matrix.unitaryGroup n ℂ := B.2.eigenvectorUnitary
          have hAeq :
              (U : Matrix n n ℂ) * A.1 * star (U : Matrix n n ℂ) =
                (U : Matrix n n ℂ) * ((UA : Matrix n n ℂ) * spectralNormalForm A *
                  star (UA : Matrix n n ℂ)) * star (U : Matrix n n ℂ) := by
            rw [spectral_normal_form_diagonalization A]
          have hUA :
              star (UA : Matrix n n ℂ) * (UA : Matrix n n ℂ) = 1 :=
            unitary.coe_star_mul_self UA
          have hTail :
              star (UA : Matrix n n ℂ) *
                ((UA : Matrix n n ℂ) * star (UB : Matrix n n ℂ)) =
              star (UB : Matrix n n ℂ) := by
            rw [← mul_assoc, hUA, one_mul]
          have hInner :
              star (UA : Matrix n n ℂ) *
                ((UA : Matrix n n ℂ) *
                  (spectralNormalForm A *
                    (star (UA : Matrix n n ℂ) *
                      ((UA : Matrix n n ℂ) * star (UB : Matrix n n ℂ))))) =
              spectralNormalForm A * star (UB : Matrix n n ℂ) := by
            rw [← mul_assoc, hUA, one_mul, hTail]
          symm
          calc
            (U : Matrix n n ℂ) * A.1 * star (U : Matrix n n ℂ)
                = (U : Matrix n n ℂ) * ((UA : Matrix n n ℂ) * spectralNormalForm A *
                    star (UA : Matrix n n ℂ)) * star (U : Matrix n n ℂ) := hAeq
            _ = (UB : Matrix n n ℂ) * spectralNormalForm A * star (UB : Matrix n n ℂ) := by
                  dsimp [U, UA, UB]
                  calc
                    ((UB : Matrix n n ℂ) * star (UA : Matrix n n ℂ)) *
                        ((UA : Matrix n n ℂ) * spectralNormalForm A * star (UA : Matrix n n ℂ)) *
                        star ((UB : Matrix.unitaryGroup n ℂ) * star (UA : Matrix.unitaryGroup n ℂ) :
                          Matrix.unitaryGroup n ℂ)
                        = (UB : Matrix n n ℂ) *
                            (star (UA : Matrix n n ℂ) *
                              ((UA : Matrix n n ℂ) *
                                (spectralNormalForm A *
                                  (star (UA : Matrix n n ℂ) *
                                    ((UA : Matrix n n ℂ) * star (UB : Matrix n n ℂ)))))) := by
                                      simp [Matrix.star_mul, mul_assoc]
                    _ = (UB : Matrix n n ℂ) * (spectralNormalForm A * star (UB : Matrix n n ℂ)) := by
                          rw [hInner]
                    _ = (UB : Matrix n n ℂ) * spectralNormalForm A * star (UB : Matrix n n ℂ) := by
                          simp [mul_assoc]

theorem spectral_coordinates_complete
    {A B : SpectralProfile n}
    (h : (spectralTameStratification.coord A).1 = (spectralTameStratification.coord B).1) :
    SpectralUnitaryEquivalent A B :=
  spectral_normal_form_complete (spectral_coordinates_determine_normal_form h)

theorem spectral_unitaryEquivalent_preserves_polynomialClass
    {A B : SpectralProfile n}
    (h : SpectralUnitaryEquivalent A B) :
    spectralPolynomialClass A = spectralPolynomialClass B := by
  rcases h with ⟨U, hB⟩
  have hchar : B.1.charpoly = A.1.charpoly := by
    rw [hB]
    exact matrix_charpoly_unitary_conj U A.1
  simpa [spectralPolynomialClass] using hchar.symm

theorem spectral_profileClassification_refines_to_polynomialClass
    (A : SpectralProfile n) :
    ∃ p : SpectralPolynomialClass, p = spectralPolynomialClass A := by
  exact ⟨spectralPolynomialClass A, rfl⟩

theorem spectral_backend_is_classification_ready_from_generic_sieve :
    NodeTag.germ ∈ generatedCertificates ∧
      NodeTag.reduction ∈ generatedCertificates ∧
      NodeTag.library ∈ generatedCertificates ∧
      NodeTag.gamma ∈ generatedCertificates := by
  exact ⟨germ_mem_generatedCertificates, reduction_mem_generatedCertificates,
    library_mem_generatedCertificates, gamma_mem_generatedCertificates⟩

theorem spectral_sieve_supports_stratified_classification :
    NodeTag.germ ∈ generatedCertificates ∧
      NodeTag.library ∈ generatedCertificates ∧
      profileClassification (SpectralProfile n) = .stratified spectralTameStratification := by
  exact ⟨germ_mem_generatedCertificates, library_mem_generatedCertificates,
    spectral_profileClassification_is_stratified⟩

def spectralInterfaces : SingularityInterfaces (SpectralProfile n) (SpectralProfile n) where
  extract := fun A => A
  locus := fun A => { B | B = A }

instance spectralHasSemanticSingularity :
    HasSemanticSingularity (SpectralProfile n) where
  singular := fun A => A.1.rank < Fintype.card n

instance spectralHasSurgeryAdmissibility :
    HasSurgeryAdmissibility (SpectralProfile n) (SpectralProfile n) where
  admissibility := fun data => .admissible data

@[simp] theorem spectral_admissibility_is_admissible
    (data : SurgeryData (SpectralProfile n) (SpectralProfile n)) :
    HasSurgeryAdmissibility.admissibility data = .admissible data := rfl

def spectralRoutingPolicy : RoutingPolicy where
  onWitness := fun _ => .dC
  onProfileWild := .tC
  onProfileInconclusive := .dC
  onAdmissibilityHorizon := .dE

def spectralIdentityProgress
    (A : SpectralProfile n) :
    ProgressCertificate (SpectralProfile n) where
  kind := .boundedResource
  measure := fun _ => 0
  before := A
  after := A
  progress := by simp

def spectralSurgeryOperator :
    SurgeryOperator (SpectralProfile n) (SpectralProfile n) where
  onAdmissible := fun data =>
    { id := .surgCDAlt
      targetMode := .cD
      before := data.profile
      after := data.profile
      reentryTarget := .tameCheck
      progress := spectralIdentityProgress data.profile }
  onAdmissibleEq := fun data _ =>
    { id := .surgTC
      targetMode := .tC
      before := data.profile
      after := data.profile
      reentryTarget := .ergoCheck
      progress := spectralIdentityProgress data.profile }

noncomputable def spectralCompiledSieve :
    CompiledSieve (SpectralProfile n) (SpectralProfile n) where
  backend := .metricGF
  interfaces := spectralInterfaces
  routing := spectralRoutingPolicy
  surgery := spectralSurgeryOperator

noncomputable def spectralAutomationGuarantee :
    AutomationGuarantee (SpectralProfile n) (SpectralProfile n) :=
  compiledSieveAutomation spectralCompiledSieve

theorem spectral_terminalVerdict_exhaustive
    (A : SpectralProfile n) :
    (∃ hgoal hreg, terminalVerdict spectralCompiledSieve A = .victory hgoal hreg) ∨
      (∃ m cert, terminalVerdict spectralCompiledSieve A = .mode m cert) ∨
      ∃ step, terminalVerdict spectralCompiledSieve A = .surgery step :=
  factInstantiation_output_trichotomy spectralCompiledSieve A

theorem spectral_regular_profiles_hit_victory
    (A : SpectralProfile n)
    (h : singularityFree A) :
    ∃ hgoal, terminalVerdict spectralCompiledSieve A = .victory hgoal h :=
  factInstantiation_goal_certificate spectralCompiledSieve h

theorem spectral_singular_profiles_route_to_surgery
    (A : SpectralProfile n)
    (h : HasSemanticSingularity.singular A) :
    ∃ step, terminalVerdict spectralCompiledSieve A = .surgery step := by
  let data : SurgeryData (SpectralProfile n) (SpectralProfile n) :=
    ⟨spectralInterfaces.locus A, A⟩
  refine ⟨spectralSurgeryOperator.onAdmissible data, ?_⟩
  simp [terminalVerdict, h, spectralCompiledSieve, spectralInterfaces, spectralSurgeryOperator,
    spectralRoutingPolicy, data, resolveProfile, profileClassification]

end

end Hypostructure.Examples
