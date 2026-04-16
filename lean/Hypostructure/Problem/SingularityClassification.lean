import Hypostructure.Problem.Backend

namespace Hypostructure.Problem

noncomputable section

structure FiniteProfileLibrary (α : Type _) where
  Class : Type _
  instFintypeClass : Fintype Class
  instDecidableEqClass : DecidableEq Class
  classify : α → Class

attribute [instance] FiniteProfileLibrary.instFintypeClass
attribute [instance] FiniteProfileLibrary.instDecidableEqClass

structure TameProfileStratification (α : Type _) where
  Stratum : Type _
  instFintypeStratum : Fintype Stratum
  instDecidableEqStratum : DecidableEq Stratum
  Param : Stratum → Type _
  dim : Stratum → ℕ
  assign : α → Stratum
  coord : (a : α) → Param (assign a)

attribute [instance] TameProfileStratification.instFintypeStratum
attribute [instance] TameProfileStratification.instDecidableEqStratum

inductive WildProfileMode
  | strangeAttractor
  | turbulentCascade
  | undecidableStructure
  | nondefinableEscape
  deriving DecidableEq, Repr

structure WildProfileWitness (α : Type _) where
  mode : WildProfileMode
  obstruction : α → Prop
  profile : α
  exhibits : obstruction profile

inductive InconclusiveProfileMode
  | missingLibrary
  | missingDefinability
  | missingRigidity
  | missingCompactness
  | missingBridge
  deriving DecidableEq, Repr

structure InconclusiveProfileWitness (α : Type _) where
  mode : InconclusiveProfileMode
  obligation : α → Prop
  profile : α
  pending : obligation profile

inductive ProfileClass (α : Type _)
  | library (L : FiniteProfileLibrary α)
  | stratified (S : TameProfileStratification α)
  | wild (W : WildProfileWitness α)
  | inconclusive (O : InconclusiveProfileWitness α)

class HasProfileClassification (α : Type _) where
  classification : ProfileClass α

def profileClassification (α : Type _) [HasProfileClassification α] : ProfileClass α :=
  HasProfileClassification.classification

theorem profileClassification_exhaustive
    (α : Type _)
    [HasProfileClassification α] :
    (∃ L : FiniteProfileLibrary α, profileClassification α = .library L) ∨
      (∃ S : TameProfileStratification α, profileClassification α = .stratified S) ∨
      (∃ W : WildProfileWitness α, profileClassification α = .wild W) ∨
      ∃ O : InconclusiveProfileWitness α, profileClassification α = .inconclusive O := by
  cases h : profileClassification α with
  | library L =>
      exact Or.inl ⟨L, by simpa [h]⟩
  | stratified S =>
      exact Or.inr <| Or.inl ⟨S, by simpa [h]⟩
  | wild W =>
      exact Or.inr <| Or.inr <| Or.inl ⟨W, by simpa [h]⟩
  | inconclusive O =>
      exact Or.inr <| Or.inr <| Or.inr ⟨O, by simpa [h]⟩

abbrev GermProfile (I : ThinInput) := BadGerm I.rankBound I.complexityLevel

theorem germProfileClassification_exhaustive
    (I : ThinInput)
    [HasProfileClassification (GermProfile I)] :
    (∃ L : FiniteProfileLibrary (GermProfile I),
        profileClassification (GermProfile I) = .library L) ∨
      (∃ S : TameProfileStratification (GermProfile I),
        profileClassification (GermProfile I) = .stratified S) ∨
      (∃ W : WildProfileWitness (GermProfile I),
        profileClassification (GermProfile I) = .wild W) ∨
      ∃ O : InconclusiveProfileWitness (GermProfile I),
        profileClassification (GermProfile I) = .inconclusive O :=
  profileClassification_exhaustive (α := GermProfile I)

end

end Hypostructure.Problem
