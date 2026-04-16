import Hypostructure.Problem.Thin
import Hypostructure.Core.Closure

namespace Hypostructure.Problem

open Hypostructure.Core
open Hypostructure.Sieve
open Classical

noncomputable section

structure Witness where
  tag : Nat
  nonzero : Prop

structure BadGerm (n c : ℕ) where
  rankBound : ℕ
  h_rank : rankBound ≤ n
  witness : Witness
  bad : Prop
  minimal : Prop

structure WitnessHom {n c : ℕ} (A B : BadGerm n c) where
  mapTag : Nat → Nat
  mapWitness : mapTag A.witness.tag = B.witness.tag
  preservesBad : A.bad → B.bad
  preservesNonzero : A.witness.nonzero → B.witness.nonzero

namespace WitnessHom

def id (A : BadGerm n c) : WitnessHom A A where
  mapTag := fun x => x
  mapWitness := rfl
  preservesBad := fun h => h
  preservesNonzero := fun h => h

def comp {A B C : BadGerm n c} (f : WitnessHom A B) (g : WitnessHom B C) :
    WitnessHom A C where
  mapTag := g.mapTag ∘ f.mapTag
  mapWitness := by
    calc
      (g.mapTag ∘ f.mapTag) A.witness.tag = g.mapTag (f.mapTag A.witness.tag) := rfl
      _ = g.mapTag B.witness.tag := by rw [f.mapWitness]
      _ = C.witness.tag := g.mapWitness
  preservesBad := fun hA => g.preservesBad (f.preservesBad hA)
  preservesNonzero := fun hA => g.preservesNonzero (f.preservesNonzero hA)

end WitnessHom

structure GermCode (n c : ℕ) where
  rankCode : Fin (n + 1)
  witnessTag : Nat
  nonzeroTag : Bool
  badTag : Bool
  minTag : Bool
  deriving DecidableEq, Repr

def encodeGermCodeTuple (g : GermCode n c) : Fin (n + 1) × Nat × Bool × Bool × Bool :=
  (g.rankCode, g.witnessTag, g.nonzeroTag, g.badTag, g.minTag)

theorem encodeGermCodeTuple_injective (n c : ℕ) :
    Function.Injective (encodeGermCodeTuple : GermCode n c → Fin (n + 1) × Nat × Bool × Bool × Bool) := by
  intro A B h
  cases A
  cases B
  cases h
  rfl

noncomputable def encodeBadGerm (A : BadGerm n c) : GermCode n c where
  rankCode := ⟨A.rankBound, Nat.lt_succ_of_le A.h_rank⟩
  witnessTag := A.witness.tag
  nonzeroTag := decide A.witness.nonzero
  badTag := decide A.bad
  minTag := decide A.minimal

private theorem prop_eq_of_decide_eq {P Q : Prop} [Decidable P] [Decidable Q]
    (h : decide P = decide Q) :
    P = Q := by
  apply propext
  by_cases hP : P <;> by_cases hQ : Q <;> simp [hP, hQ] at h ⊢

theorem encodeBadGerm_injective (n c : ℕ) :
    Function.Injective (encodeBadGerm : BadGerm n c → GermCode n c) := by
  intro A B h
  cases A with
  | mk rankA hRankA witnessA badA minA =>
      cases B with
      | mk rankB hRankB witnessB badB minB =>
          cases witnessA with
          | mk tagA nonzeroA =>
              cases witnessB with
              | mk tagB nonzeroB =>
                  have hRank : rankA = rankB := by
                    have h' := congrArg (fun g => g.rankCode.val) h
                    simpa [encodeBadGerm] using h'
                  have hTag : tagA = tagB := by
                    have h' := congrArg GermCode.witnessTag h
                    simpa [encodeBadGerm] using h'
                  have hNonzero : nonzeroA = nonzeroB := by
                    have h' := congrArg GermCode.nonzeroTag h
                    exact prop_eq_of_decide_eq (by simpa [encodeBadGerm] using h')
                  have hBad : badA = badB := by
                    have h' := congrArg GermCode.badTag h
                    exact prop_eq_of_decide_eq (by simpa [encodeBadGerm] using h')
                  have hMin : minA = minB := by
                    have h' := congrArg GermCode.minTag h
                    exact prop_eq_of_decide_eq (by simpa [encodeBadGerm] using h')
                  subst hRank
                  subst hTag
                  subst hNonzero
                  subst hBad
                  subst hMin
                  have hProof : hRankA = hRankB := Subsingleton.elim _ _
                  subst hProof
                  rfl

noncomputable instance instEncodableGermCode : Encodable (GermCode n c) :=
  Encodable.ofInj encodeGermCodeTuple (encodeGermCodeTuple_injective n c)

noncomputable instance instEncodableBadGerm : Encodable (BadGerm n c) :=
  Encodable.ofInj encodeBadGerm (encodeBadGerm_injective n c)

def StaticClassifiable (I : ThinInput) : Prop :=
  Nonempty (Encodable (BadGerm I.rankBound I.complexityLevel)) ∧ I.germBounded

def BoundedReductionRealized (I : ThinInput) : Prop :=
  ∀ {A B : BadGerm I.rankBound I.complexityLevel}, WitnessHom A B →
    ∃ C : BadGerm I.rankBound I.complexityLevel, C.rankBound ≤ I.rankBound ∧ I.germBounded

theorem thinBoundedReduction_rankBounded
    (I : ThinInput)
    {A B : BadGerm I.rankBound I.complexityLevel}
    (_f : WitnessHom A B) :
    ∃ C : BadGerm I.rankBound I.complexityLevel, C.rankBound ≤ I.rankBound ∧ I.germBounded := by
  exact ⟨B, B.h_rank, I.germBounded_holds⟩

theorem thinBoundedReductionRealized
    (I : ThinInput) :
    BoundedReductionRealized I := by
  intro A B f
  exact thinBoundedReduction_rankBounded I f

theorem thinClassifiableStatic
    (I : ThinInput) :
    StaticClassifiable I := by
  exact ⟨⟨instEncodableBadGerm (n := I.rankBound) (c := I.complexityLevel)⟩, I.germBounded_holds⟩

structure UniversalBad (n c : ℕ) where
  carrier : Type
  inject : ∀ A : BadGerm n c, Nat → carrier
  initial :
    ∀ {X : Type} (f : ∀ A : BadGerm n c, Nat → X),
      ∃! g : carrier → X, ∀ A a, g (inject A a) = f A a

def HasBoundedUniversalBad (I : ThinInput) : Prop :=
  Nonempty (UniversalBad I.rankBound I.complexityLevel) ∧ I.reductionInitialized

def canonicalUniversalBad (n c : ℕ) : UniversalBad n c where
  carrier := Σ A : BadGerm n c, Nat
  inject := fun A a => ⟨A, a⟩
  initial := by
    intro X f
    refine ⟨fun x => f x.1 x.2, ?_, ?_⟩
    · intro A a
      rfl
    · intro g hg
      funext x
      cases x with
      | mk A a =>
          simpa using hg A a

theorem thinInitialityBounded
    (I : ThinInput) :
    HasBoundedUniversalBad I := by
  exact ⟨⟨canonicalUniversalBad I.rankBound I.complexityLevel⟩, I.reductionInitialized_holds⟩

def BoundedLibraryComplete (I : ThinInput) : Prop :=
  LibraryCertificate I ∧ StaticClassifiable I ∧ HasBoundedUniversalBad I

theorem thinLibraryBounded
    (I : ThinInput) :
    BoundedLibraryComplete I := by
  exact ⟨libraryCertificate I, thinClassifiableStatic I, thinInitialityBounded I⟩

theorem emit_library_from_bounded_completeness
    (I : ThinInput)
    (h : BoundedLibraryComplete I) :
    NodeTag.library ∈ closure [] (insert NodeTag.library seedContext) := by
  have _ : LibraryCertificate I := h.1
  exact subset_closure ([] : RuleSet NodeTag) (insert NodeTag.library seedContext) (by simp)

structure GammaPackage (I : ThinInput) where
  sourceSpace : Type
  fiberSpace : Type
  exactness : I.gammaExact
  faithfulness : I.gammaFaithful
  tensorPreservation : I.gammaTensorPreserving

def HasGammaPackage (I : ThinInput) : Prop :=
  Nonempty (GammaPackage I)

def gammaPackage (I : ThinInput) : GammaPackage I where
  sourceSpace := I.ambientSpace
  fiberSpace := I.targetSpace
  exactness := I.gammaExact_holds
  faithfulness := I.gammaFaithful_holds
  tensorPreservation := I.gammaTensorPreserving_holds

theorem thinGammaConstructor (I : ThinInput) : HasGammaPackage I := by
  exact ⟨gammaPackage I⟩

end

end Hypostructure.Problem
