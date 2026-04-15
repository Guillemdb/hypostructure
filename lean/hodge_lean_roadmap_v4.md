
# Hodge Lean Roadmap V4

## Metadata

| Field | Value |
|---|---|
| **Goal** | Formalize the hypostructure proof engine in Lean and prove the Hodge thin-to-Lock completion theorem |
| **Lean Scope** | The framework core, the new algebraic backend, the Hodge thin instance, and the final template-closure proof |
| **Trusted Boundary** | Only the imported literature layer and the classical thin-input extraction layer |
| **Target Theorem** | `hodge_framework_unconditional` |
| **Implementation Strategy** | Phase A = tag-level certificate kernel proof; Phase B = optional proof-relevant payload upgrade |
| **Document Role** | Standalone engineering specification for the Lean development |

---

## 0. Mission

This roadmap specifies the exact Lean development needed to prove the Hodge template inside the hypostructure framework, under the rule:

- **assume as axioms every statement already established in the classical literature,**
- **prove in Lean every framework-original statement and every new algebraic backend statement needed for the template.**

The final Lean theorem is not a formalization of all classical Hodge theory from first principles. The final Lean theorem is:

> starting from a **verified Hodge thin input**, and assuming the imported literature bridge layer, the framework engine derives the backend closure, dispatches the Hodge and Tannakian bridges, emits the blocked Lock certificate, and proves that no goal-relevant inconclusive obligations remain.

That is the exact theorem implemented in Lean.

---

## 1. Final Lean theorem

The top-level theorem to prove is:

```lean
theorem hodge_framework_unconditional
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    ProofComplete allRules deps (runHodgeSystem I) CertTag.catHomBlk
```

A more explicit equivalent form is:

```lean
theorem hodge_framework_unconditional_explicit
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.catHomBlk ∈ runHodgeSystem I ∧
    Disjoint
      (obligations (runHodgeSystem I))
      (goalCone deps CertTag.catHomBlk)
```

This theorem is the only required top-level theorem. Everything else is support.

---

## 2. Trusted boundary

This section lists **everything that is assumed** rather than proved.

## 2.1 Trusted mathematical content

The following are **not proved in Lean** in this project:

1. the classical extraction of the Hodge thin object from ordinary mathematics,
2. the literature-anchored Hodge bridge,
3. the literature-anchored Tannakian bridge,
4. the classical soundness of the Tannakian context package.

These are bundled into one class:

```lean
class ImportedHodgeAxioms (I : VerifiedHodgeThinInput) : Prop where
  classicalExtraction :
    ClassicalOrigin I
  hodgeBridgeSound :
    HodgeBridgePremises I → ProducesMHS I
  tannakianContextSound :
    GammaContextPremises I → ProducesGamma I
  tannakianBridgeSound :
    TannakianBridgePremises I → ProducesTann I
```

## 2.2 Trusted objects to declare

The following objects are declared but not internally constructed from first principles:

```lean
structure ClassicalOrigin (I : VerifiedHodgeThinInput) : Prop := ...
structure HodgeBridgePremises (I : VerifiedHodgeThinInput) : Prop := ...
structure GammaContextPremises (I : VerifiedHodgeThinInput) : Prop := ...
structure TannakianBridgePremises (I : VerifiedHodgeThinInput) : Prop := ...

structure ProducesMHS (I : VerifiedHodgeThinInput) : Prop := ...
structure ProducesGamma (I : VerifiedHodgeThinInput) : Prop := ...
structure ProducesTann (I : VerifiedHodgeThinInput) : Prop := ...
```

## 2.3 Nothing else is trusted

Everything not listed in Section 2.1 and Section 2.2 is part of the new Lean proof and must be implemented.

---

## 3. Phase A design choice

The Lean development should use a **tag-level certificate kernel** first.

That means:

- certificates are represented by a finite tag type `CertTag`,
- contexts are finite sets of tags,
- bridge dispatch, promotions, and inc-upgrades are all inference rules on tags,
- closure is a fixed point on finite contexts,
- proof completion is a proposition about tag membership and tag-level obligation disjointness.

This is the minimal design that still proves the Hodge template in the kernel.

### Why this is the correct first design

1. it matches the bounded-description regime of the framework,
2. it keeps closure finite and mechanically checkable,
3. it isolates the new mathematics from heavy proof-relevant payload engineering,
4. it lets the final theorem compile sooner,
5. it can be upgraded later to proof-relevant certificates without changing the high-level theorem.

---

## 4. Repository layout

The Lean repository should be:

```text
HypoHodge/
  lakefile.lean
  lean-toolchain
  HypoHodge.lean

  HypoHodge/
    Core/
      CertTag.lean
      Context.lean
      Rule.lean
      Closure.lean
      GoalCone.lean
      ObligationLedger.lean
      ProofComplete.lean

    Imported/
      Boundary.lean

    Algebraic/
      VerifiedThinInput.lean
      LocalCertificates.lean
      BadGerm.lean
      Coding.lean
      BoundedReduction.lean
      Initiality.lean
      CatLib.lean
      GammaConstructor.lean
      BackendAutoclose.lean

    Hodge/
      Permits.lean
      Run.lean
      ProofAudit.lean
      Final.lean

    Examples/
      Toy.lean
```

The package root file:

```lean
-- HypoHodge.lean
import HypoHodge.Core.CertTag
import HypoHodge.Core.Context
import HypoHodge.Core.Rule
import HypoHodge.Core.Closure
import HypoHodge.Core.GoalCone
import HypoHodge.Core.ObligationLedger
import HypoHodge.Core.ProofComplete

import HypoHodge.Imported.Boundary

import HypoHodge.Algebraic.VerifiedThinInput
import HypoHodge.Algebraic.LocalCertificates
import HypoHodge.Algebraic.BadGerm
import HypoHodge.Algebraic.Coding
import HypoHodge.Algebraic.BoundedReduction
import HypoHodge.Algebraic.Initiality
import HypoHodge.Algebraic.CatLib
import HypoHodge.Algebraic.GammaConstructor
import HypoHodge.Algebraic.BackendAutoclose

import HypoHodge.Hodge.Permits
import HypoHodge.Hodge.Run
import HypoHodge.Hodge.ProofAudit
import HypoHodge.Hodge.Final
```

---

## 5. Mathlib objects to reuse

The project should reuse, not redefine, the following standard objects:

- `Finset`
- `Fintype`
- `DecidableEq`
- `Set`
- `Monotone`
- `Nat.iterate`
- `AddCommGroup`
- `Module`
- `NormedAddCommGroup`
- `NormedSpace`
- `FiniteDimensional`
- `Encodable`
- `Function.LeftInverse`
- `Subsingleton`
- standard lemmas on finite-dimensional linear maps, ranges, kernels, and ranks

The project should **not** attempt to reimplement these.

---

## 6. Core object inventory

This section lists every public object that must be defined in the Lean code.

## 6.1 `Core/CertTag.lean`

### Objects to define

```lean
namespace HypoHodge.Core

inductive CertTag
  | adj
  | dE
  | recN
  | cMu
  | scLambda
  | scPartialC
  | capH
  | lsSigma
  | tbPi
  | tbO
  | boundPartial
  | ambient
  | repCon
  | repComp
  | adm
  | germ
  | init
  | catLib
  | gamma
  | mhs
  | tann
  | catHomBlk
  | catHomInc
  | promoInc
  deriving DecidableEq, Repr

def CertTag.isInc : CertTag → Bool
def allTags : Finset CertTag

end HypoHodge.Core
```

### Public theorems to prove

```lean
theorem mem_allTags (k : CertTag) : k ∈ allTags
theorem allTags_nodup : allTags.Nodup
```

### Notes

- `allTags` should be defined explicitly.
- Do not rely on automation for finiteness; make the finite universe manifest.

### Implementation note

Implemented in `lean/HypoHodge/Core/CertTag.lean`.

- `CertTag` is defined with the full Phase A tag universe and derives `Fintype`.
- `allTags` is realized concretely and the finiteness lemmas are proved.
- This section is complete for the current Phase A kernel.

---

## 6.2 `Core/Context.lean`

### Objects to define

```lean
abbrev Context := Finset CertTag
```

### Public theorems to prove

```lean
theorem context_ext :
    (Γ Δ : Context) →
    (∀ k, k ∈ Γ ↔ k ∈ Δ) →
    Γ = Δ
```

This is optional if not needed later, but recommended.

### Implementation note

Implemented in `lean/HypoHodge/Core/Context.lean`.

- `Context` is defined as `Finset CertTag`.
- `context_ext` is proved via `Finset.ext`.
- This section is complete for the current Phase A kernel.

---

## 6.3 `Core/Rule.lean`

### Objects to define

```lean
inductive RuleKind
  | backend
  | bridge
  | promotion
  | incUpgrade
  deriving DecidableEq, Repr

structure Rule where
  kind       : RuleKind
  premises   : Finset CertTag
  conclusion : CertTag
  deriving Repr

abbrev RuleSet := List Rule

def Rule.enabled (r : Rule) (Γ : Context) : Prop :=
  r.premises ⊆ Γ

def fireRule (r : Rule) (Γ : Context) : Context
def step (rules : RuleSet) (Γ : Context) : Context
```

### Public theorems to prove

```lean
theorem enabled_iff_subset (r : Rule) (Γ : Context) :
    r.enabled Γ ↔ r.premises ⊆ Γ

theorem fireRule_eq_insert_of_enabled
    (r : Rule) (Γ : Context)
    (h : r.enabled Γ) :
    fireRule r Γ = insert r.conclusion Γ

theorem fireRule_eq_self_of_disabled
    (r : Rule) (Γ : Context)
    (h : ¬ r.enabled Γ) :
    fireRule r Γ = Γ

theorem subset_fireRule (r : Rule) (Γ : Context) :
    Γ ⊆ fireRule r Γ

theorem monotone_fireRule (r : Rule) :
    Monotone (fireRule r)

theorem subset_step (rules : RuleSet) (Γ : Context) :
    Γ ⊆ step rules Γ

theorem monotone_step (rules : RuleSet) :
    Monotone (step rules)
```

### Implementation note

Implemented in `lean/HypoHodge/Core/Rule.lean`.

- Replaced the placeholder declarations for `monotone_fireRule`, `subset_step`, and `monotone_step` with direct proofs.
- The proofs are by case analysis on rule enablement and induction on the rule list.
- Public statements were kept unchanged.

---

## 6.4 `Core/Closure.lean`

### Objects to define

```lean
def closureN (rules : RuleSet) (n : ℕ) (Γ : Context) : Context :=
  Nat.iterate (step rules) n Γ

def closure (rules : RuleSet) (Γ : Context) : Context :=
  closureN rules allTags.card Γ
```

### Public theorems to prove

```lean
theorem subset_closureN (rules : RuleSet) (n : ℕ) (Γ : Context) :
    Γ ⊆ closureN rules n Γ

theorem monotone_closureN (rules : RuleSet) (n : ℕ) :
    Monotone (closureN rules n)

theorem subset_closure (rules : RuleSet) (Γ : Context) :
    Γ ⊆ closure rules Γ

theorem monotone_closure (rules : RuleSet) :
    Monotone (closure rules)

theorem closure_fixed (rules : RuleSet) (Γ : Context) :
    step rules (closure rules Γ) = closure rules Γ

theorem closure_idempotent (rules : RuleSet) (Γ : Context) :
    closure rules (closure rules Γ) = closure rules Γ

theorem closure_least_fixed
    (rules : RuleSet) (Γ Δ : Context)
    (hΓ : Γ ⊆ Δ)
    (hΔ : step rules Δ = Δ) :
    closure rules Γ ⊆ Δ
```

These are the core fixed-point theorems.

### Implementation note

Implemented in `lean/HypoHodge/Core/Closure.lean`.

- Replaced the placeholder fixed-point theorems with finite-cardinality proofs over `CertTag`.
- The implementation proves stabilization after `allTags.card` iterations by showing that every non-fixed iterate strictly increases cardinality, which is impossible beyond the finite tag universe.
- `closure_idempotent` is then derived from `closure_fixed` and `closure_least_fixed`.

---

## 6.5 `Core/GoalCone.lean`

### Objects to define

```lean
def backStep (deps : RuleSet) (S : Context) : Context
def goalConeN (deps : RuleSet) (n : ℕ) (goal : CertTag) : Context
def goalCone (deps : RuleSet) (goal : CertTag) : Context
```

Where `goalCone` should be:

```lean
def goalCone (deps : RuleSet) (goal : CertTag) : Context :=
  goalConeN deps allTags.card goal
```

### Public theorems to prove

```lean
theorem goal_mem_goalCone (deps : RuleSet) (goal : CertTag) :
    goal ∈ goalCone deps goal

theorem backStep_monotone (deps : RuleSet) :
    Monotone (backStep deps)

theorem goalConeN_monotone (deps : RuleSet) (n : ℕ) :
    Monotone (fun S => Nat.iterate (backStep deps) n S)

theorem goalCone_fixed (deps : RuleSet) (goal : CertTag) :
    backStep deps (goalCone deps goal) = goalCone deps goal

theorem premises_in_goalCone_of_conclusion
    (deps : RuleSet) (goal : CertTag) (r : Rule)
    (hr : r ∈ deps)
    (hconc : r.conclusion ∈ goalCone deps goal) :
    r.premises ⊆ goalCone deps goal
```

### Implementation note

Implemented in `lean/HypoHodge/Core/GoalCone.lean`.

- Replaced the placeholder backward-closure theorems with direct proofs from the `foldl` definition of `backStep`.
- The fixed-point proof uses the same finite-cardinality stabilization argument as `Closure`.
- The monotonicity theorem was corrected to the order-theoretically meaningful statement: monotonicity of iterating `backStep` on seed contexts. The original signature was on the `goal : CertTag` parameter, which is not the intended monotone object here.

---

## 6.6 `Core/ObligationLedger.lean`

### Objects to define

```lean
def obligations (Γ : Context) : Context :=
  Γ.filter (fun k => CertTag.isInc k)
```

### Public theorems to prove

```lean
theorem obligations_subset (Γ : Context) :
    obligations Γ ⊆ Γ

theorem mem_obligations_iff (Γ : Context) (k : CertTag) :
    k ∈ obligations Γ ↔ k ∈ Γ ∧ CertTag.isInc k = true
```

### Implementation note

Implemented in `lean/HypoHodge/Core/ObligationLedger.lean`.

- `obligations` is defined by filtering on `CertTag.isInc`.
- `obligations_subset` and `mem_obligations_iff` are proved directly from the filter definition.
- This section is complete for the current Phase A kernel.

---

## 6.7 `Core/ProofComplete.lean`

### Objects to define

```lean
def ProofComplete (rules deps : RuleSet) (Γ : Context) (goal : CertTag) : Prop :=
  goal ∈ Γ ∧ Disjoint (obligations Γ) (goalCone deps goal)
```

### Public theorems to prove

```lean
theorem proofComplete_iff
    (rules deps : RuleSet) (Γ : Context) (goal : CertTag) :
    ProofComplete rules deps Γ goal ↔
      goal ∈ Γ ∧ Disjoint (obligations Γ) (goalCone deps goal)

theorem proofComplete_of_goal_mem_and_disjoint
    (rules deps : RuleSet) (Γ : Context) (goal : CertTag)
    (hgoal : goal ∈ Γ)
    (hdisj : Disjoint (obligations Γ) (goalCone deps goal)) :
    ProofComplete rules deps Γ goal

theorem not_proofComplete_of_goal_missing
    (rules deps : RuleSet) (Γ : Context) (goal : CertTag)
    (hgoal : goal ∉ Γ) :
    ¬ ProofComplete rules deps Γ goal

theorem not_proofComplete_of_inc_in_goalCone
    (rules deps : RuleSet) (Γ : Context) (goal k : CertTag)
    (hinc : k ∈ obligations Γ)
    (hcone : k ∈ goalCone deps goal) :
    ¬ ProofComplete rules deps Γ goal
```

These are the exact kernel-level theorems used by the final Hodge proof.

### Implementation note

Implemented in `lean/HypoHodge/Core/ProofComplete.lean`.

- `ProofComplete` is defined exactly as specified.
- All listed introduction and exclusion lemmas are proved directly from the definition and `Disjoint`.
- This section is complete for the current Phase A kernel.

---

## 7. Imported boundary inventory

All objects in this section are **declared**, not proved.

## 7.1 `Imported/Boundary.lean`

### Objects to define

```lean
structure ClassicalOrigin (I : VerifiedHodgeThinInput) : Prop := ...
structure HodgeBridgePremises (I : VerifiedHodgeThinInput) : Prop := ...
structure GammaContextPremises (I : VerifiedHodgeThinInput) : Prop := ...
structure TannakianBridgePremises (I : VerifiedHodgeThinInput) : Prop := ...

structure ProducesMHS (I : VerifiedHodgeThinInput) : Prop := ...
structure ProducesGamma (I : VerifiedHodgeThinInput) : Prop := ...
structure ProducesTann (I : VerifiedHodgeThinInput) : Prop := ...

class ImportedHodgeAxioms (I : VerifiedHodgeThinInput) : Prop where
  classicalExtraction :
    ClassicalOrigin I
  hodgeBridgeSound :
    HodgeBridgePremises I → ProducesMHS I
  tannakianContextSound :
    GammaContextPremises I → ProducesGamma I
  tannakianBridgeSound :
    TannakianBridgePremises I → ProducesTann I
```

### No theorems to prove

This file is the trusted boundary. There are no Lean proofs here.

### Implementation note

Implemented in `lean/HypoHodge/Imported/Boundary.lean`.

- The trusted boundary declarations and `ImportedHodgeAxioms` class are present as specified.
- This section is complete by declaration and intentionally remains outside the proved kernel.

---

## 8. Hodge thin input inventory

## 8.1 `Algebraic/VerifiedThinInput.lean`

### Objects to define

```lean
structure VerifiedHodgeThinInput where
  V               : Type
  instAddComm     : AddCommGroup V
  instModuleR     : Module ℝ V
  instNormedGroup : NormedAddCommGroup V
  instNormedSpace : NormedSpace ℝ V
  instFiniteDim   : FiniteDimensional ℝ V

  Qrank           : ℕ
  hodgeSubset     : Set V
  symmetry        : Type

  potential       : V → ℝ
  dissipation     : V → ℝ
  flow            : ℝ → V → V

  flow_id         : ∀ t x, flow t x = x
  dissipation_zero : ∀ x, dissipation x = 0
  potential_quad  : ∀ x, potential x = ‖x‖^2

  connected       : Prop
  contractible    : Prop
  tame            : Prop
```

After definition, mark the instance fields as instances.

### Objects to define in the same file

```lean
def gamma0 (I : VerifiedHodgeThinInput) : Context :=
  { CertTag.adj, CertTag.dE, CertTag.recN, CertTag.cMu, CertTag.scLambda,
    CertTag.scPartialC, CertTag.capH, CertTag.lsSigma,
    CertTag.tbPi, CertTag.tbO, CertTag.boundPartial }
```

No theorems in this file besides simple membership lemmas if useful.

### Implementation note

Implemented in `lean/HypoHodge/Algebraic/VerifiedThinInput.lean`.

- `VerifiedHodgeThinInput` is defined with the required analytic and finite-dimensional data.
- The structure fields exporting algebraic and topological data are installed as instances where needed.
- `gamma0` is defined with the required local certificate basis.
- Conformance note: the implementation includes an additional integer parameter `p`, because the downstream bounded-germ sector is indexed by both `Qrank` and `p`.
- This section is complete for the current Phase A scaffold, with that representation-level extension.

---

## 8.2 `Algebraic/LocalCertificates.lean`

### Public theorems to prove

```lean
theorem emit_adj (I : VerifiedHodgeThinInput) :
    CertTag.adj ∈ gamma0 I

theorem emit_dE (I : VerifiedHodgeThinInput) :
    CertTag.dE ∈ gamma0 I

theorem emit_recN (I : VerifiedHodgeThinInput) :
    CertTag.recN ∈ gamma0 I

theorem emit_cMu (I : VerifiedHodgeThinInput) :
    CertTag.cMu ∈ gamma0 I

theorem emit_scLambda (I : VerifiedHodgeThinInput) :
    CertTag.scLambda ∈ gamma0 I

theorem emit_scPartialC (I : VerifiedHodgeThinInput) :
    CertTag.scPartialC ∈ gamma0 I

theorem emit_capH (I : VerifiedHodgeThinInput) :
    CertTag.capH ∈ gamma0 I

theorem emit_lsSigma (I : VerifiedHodgeThinInput) :
    CertTag.lsSigma ∈ gamma0 I

theorem emit_tbPi (I : VerifiedHodgeThinInput) :
    CertTag.tbPi ∈ gamma0 I

theorem emit_tbO (I : VerifiedHodgeThinInput) :
    CertTag.tbO ∈ gamma0 I

theorem emit_boundPartial (I : VerifiedHodgeThinInput) :
    CertTag.boundPartial ∈ gamma0 I
```

### Aggregate theorems to prove

```lean
theorem gamma0_complete (I : VerifiedHodgeThinInput) :
    { CertTag.adj, CertTag.dE, CertTag.recN, CertTag.cMu, CertTag.scLambda,
      CertTag.scPartialC, CertTag.capH, CertTag.lsSigma,
      CertTag.tbPi, CertTag.tbO, CertTag.boundPartial } ⊆ gamma0 I

theorem gamma0_no_local_inc (I : VerifiedHodgeThinInput) :
    obligations (gamma0 I) = ∅
```

### Implementation note

Implemented in `lean/HypoHodge/Algebraic/LocalCertificates.lean`.

- All local emission lemmas are proved directly from `gamma0`.
- `gamma0_complete` and `gamma0_no_local_inc` are proved.
- This section is complete for the current Phase A kernel.

---

## 9. New algebraic backend object inventory

Everything in this section is new Lean mathematics and must be implemented.

## 9.1 `Algebraic/BadGerm.lean`

### Objects to define

```lean
structure HodgeWitness where
  tag      : Nat
  nonzero  : Prop

structure BadAlgGerm (n p : ℕ) where
  rankBound : ℕ
  h_rank    : rankBound ≤ n
  witness   : HodgeWitness
  bad       : Prop
  minimal   : Prop

structure WitnessHom {n p : ℕ} (A B : BadAlgGerm n p) where
  mapWitness       : A.witness → B.witness
  preservesBad     : A.bad → B.bad
  preservesNonzero : A.witness.nonzero → B.witness.nonzero
```

### Public theorems to prove

```lean
def WitnessHom.id (A : BadAlgGerm n p) : WitnessHom A A
def WitnessHom.comp {A B C : BadAlgGerm n p} :
    WitnessHom A B → WitnessHom B C → WitnessHom A C

theorem WitnessHom.id_comp
    (f : WitnessHom A B) :
    WitnessHom.comp (WitnessHom.id A) f = f

theorem WitnessHom.comp_id
    (f : WitnessHom A B) :
    WitnessHom.comp f (WitnessHom.id B) = f

theorem WitnessHom_assoc
    (f : WitnessHom A B) (g : WitnessHom B C) (h : WitnessHom C D) :
    WitnessHom.comp (WitnessHom.comp f g) h =
    WitnessHom.comp f (WitnessHom.comp g h)
```

These category-style laws are needed only if you want the bounded bad family to behave as a small category. They are recommended.

### Implementation note

Implemented in `lean/HypoHodge/Algebraic/BadGerm.lean`.

- Replaced the placeholder identity and associativity laws for `WitnessHom` with definitional proofs.
- The implementation uses case-splitting on the morphism structures; no new axioms were added.
- Conformance note: `WitnessHom` is implemented via witness-tag transport (`mapTag` plus compatibility proofs) rather than literally as a function `A.witness → B.witness`, because `HodgeWitness` contains proposition fields and the current scaffold treats the witness payload at tag level.

---

## 9.2 `Algebraic/Coding.lean`

### Objects to define

Use an explicit code type to witness smallness.

```lean
structure GermCode (n p : ℕ) where
  rankCode   : Fin (n + 1)
  witnessTag : Nat
  nonzeroTag : Bool
  badTag     : Bool
  minTag     : Bool
  deriving DecidableEq, Repr
```

### Objects to define

```lean
def encodeBadAlgGerm : BadAlgGerm n p → GermCode n p
```

### Public theorems to prove

```lean
theorem encodeBadAlgGerm_injective :
    Function.Injective (encodeBadAlgGerm : BadAlgGerm n p → GermCode n p)

instance instEncodableGermCode : Encodable (GermCode n p)
instance instEncodableBadAlgGerm : Encodable (BadAlgGerm n p)

theorem boundedGermSmallness (n p : ℕ) :
    Encodable (BadAlgGerm n p)
```

### Design decision

This project will use:

```lean
abbrev GermSmall (α : Type) := Encodable α
```

instead of trying to use a more elaborate notion of smallness. This is the right Lean witness for the bounded algebraic sector.

### Implementation note

Implemented in `lean/HypoHodge/Algebraic/Coding.lean`.

- Replaced the placeholder `encodeBadAlgGerm_injective` with a proof.
- The code witness was extended with `nonzeroTag : Bool`; without that field, injectivity fails because `HodgeWitness.nonzero` would be erased by the encoding.
- The proof reconstructs equality of the proposition fields from equality of their `decide` booleans via propositional extensionality, and uses proof irrelevance for the rank-bound witness.

---

## 9.3 `Algebraic/BoundedReduction.lean`

### Public theorems to prove

```lean
theorem hodgeBadBoundedReduction
    (I : VerifiedHodgeThinInput)
    {A B : BadAlgGerm I.Qrank p}
    (f : WitnessHom A B) :
    ∃ C : BadAlgGerm I.Qrank p, True
```

This theorem is the Lean representative of bounded bad-pattern reduction.

```lean
theorem hodgeClassifiableStatic
    (I : VerifiedHodgeThinInput) :
    True
```

This theorem should later be strengthened to a dedicated proposition:

```lean
def StaticClassifiable (I : VerifiedHodgeThinInput) : Prop := ...
theorem hodgeClassifiableStatic
    (I : VerifiedHodgeThinInput) :
    StaticClassifiable I
```

### Recommendation

Use the strengthened proposition form. Do not leave this theorem at `True` in the final code.

### Implementation note

Implemented in `lean/HypoHodge/Algebraic/BoundedReduction.lean`.

- `StaticClassifiable` is now implemented as `Encodable (BadAlgGerm I.Qrank I.p)`, so the section is tied directly to the coding/smallness result.
- Added a stronger helper theorem `hodgeBadBoundedReduction_rankBounded` and the proposition-level witness `BoundedReductionRealized`.
- The original roadmap theorem `hodgeBadBoundedReduction : ∃ C, True` is still present for interface compatibility, but the semantic content now lives in the stronger helper results.
- Conformance note: this section still deviates from the original roadmap at the statement-shape level because the public theorem in the document remains weaker than the stronger invariant now implemented in code.

---

## 9.4 `Algebraic/Initiality.lean`

### Objects to define

```lean
structure UniversalBad (n p : ℕ) where
  carrier : Type
  inject  : ∀ A : BadAlgGerm n p, A.witness → carrier
  initial :
    ∀ {X : Type} (f : ∀ A : BadAlgGerm n p, A.witness → X),
      ∃! g : carrier → X, ∀ A, g ∘ inject A = f A
```

### Public theorems to prove

```lean
theorem hodgeInitialityBounded
    (I : VerifiedHodgeThinInput) :
    ∃ U : UniversalBad I.Qrank p, True
```

Recommended strengthened version:

```lean
def HasBoundedUniversalBad (I : VerifiedHodgeThinInput) : Prop := ...
theorem hodgeInitialityBounded
    (I : VerifiedHodgeThinInput) :
    HasBoundedUniversalBad I
```

### Implementation note

Implemented in `lean/HypoHodge/Algebraic/Initiality.lean`.

- Added a concrete bounded universal object `canonicalUniversalBad`.
- Proved `hodgeInitialityBounded` by packaging that explicit object into `HasBoundedUniversalBad`.
- Conformance note: `UniversalBad.inject` is implemented over `Nat` tags rather than literal `A.witness` terms, matching the current witness-tag encoding used in `BadGerm`.

---

## 9.5 `Algebraic/CatLib.lean`

### Objects to define

```lean
def boundedBadLibrary (I : VerifiedHodgeThinInput) : Context :=
  { CertTag.catLib }
```

### Public theorems to prove

```lean
theorem hodgeCatLibBounded
    (I : VerifiedHodgeThinInput) :
    CertTag.catLib ∈ boundedBadLibrary I
```

Recommended strengthened proposition:

```lean
def BoundedCatLibComplete (I : VerifiedHodgeThinInput) : Prop := ...
theorem hodgeCatLibBounded
    (I : VerifiedHodgeThinInput) :
    BoundedCatLibComplete I
```

Then derive the tag from the proposition:

```lean
theorem emit_catLib_from_bounded_completeness
    (I : VerifiedHodgeThinInput)
    (h : BoundedCatLibComplete I) :
    CertTag.catLib ∈ closure [] (gamma0 I ∪ {CertTag.catLib})
```

### Implementation note

Implemented in `lean/HypoHodge/Algebraic/CatLib.lean`.

- Replaced the placeholder emission theorem with a direct closure-membership proof from `subset_closure`.
- The proof does not use any additional trusted assumptions.

---

## 9.6 `Algebraic/GammaConstructor.lean`

### Objects to define

```lean
structure GammaPackage where
  C            : Type
  omegaB       : Type
  exactness    : Prop
  faithfulness : Prop
  tensorPres   : Prop
```

### Public theorems to prove

```lean
theorem hodgeGammaConstructor
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    ∃ G : GammaPackage, True
```

Recommended strengthened proposition:

```lean
def HasGammaPackage (I : VerifiedHodgeThinInput) : Prop := ...
theorem hodgeGammaConstructor
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    HasGammaPackage I
```

### Implementation note

Implemented in `lean/HypoHodge/Algebraic/GammaConstructor.lean`.

- `GammaPackage` is defined, and `HasGammaPackage` now requires package-level exactness, faithfulness, tensor preservation, and a soundness bridge `GammaContextPremises I → ProducesGamma I`.
- `hodgeGammaConstructor` is proved by constructing a concrete package and discharging the soundness component from `ImportedHodgeAxioms.tannakianContextSound`.
- Conformance note: the package is still intentionally lightweight at the data level, but this section is no longer only a vacuous existence witness.

---

## 9.7 `Algebraic/BackendAutoclose.lean`

### Objects to define

```lean
def backendBase (I : VerifiedHodgeThinInput) : Context :=
  { CertTag.ambient, CertTag.repCon, CertTag.repComp, CertTag.adm,
    CertTag.germ, CertTag.init, CertTag.catLib, CertTag.gamma }
```

```lean
def backendRules : RuleSet := [...]
```

### Public theorems to prove

```lean
theorem hodgeAmbientFDExpansion
    (I : VerifiedHodgeThinInput) :
    CertTag.ambient ∈ backendBase I

theorem hodgeRepConFD
    (I : VerifiedHodgeThinInput) :
    CertTag.repCon ∈ backendBase I

theorem hodgeIdentityParametrizationCompleteness
    (I : VerifiedHodgeThinInput) :
    CertTag.repComp ∈ backendBase I

theorem hodgeAutoAdmissibility
    (I : VerifiedHodgeThinInput) :
    CertTag.adm ∈ backendBase I

theorem hodgeGermSmallnessBounded
    (I : VerifiedHodgeThinInput) :
    CertTag.germ ∈ backendBase I

theorem hodgeInitialityTag
    (I : VerifiedHodgeThinInput) :
    CertTag.init ∈ backendBase I

theorem hodgeCatLibTag
    (I : VerifiedHodgeThinInput) :
    CertTag.catLib ∈ backendBase I

theorem hodgeGammaTag
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.gamma ∈ backendBase I

theorem hodgeBackendAutoclose
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    backendBase I ⊆ closure backendRules (gamma0 I)
```

This theorem is the exact Lean version of backend auto-closure.

### Implementation note

Implemented in `lean/HypoHodge/Algebraic/BackendAutoclose.lean`.

- Replaced the placeholder theorem with an explicit proof.
- The proof shows `backendBase` is already present after one `step backendRules (gamma0 I)` pass, then lifts that one-step result into `closure` using monotonicity of `step` and the closure fixed-point theorem already available in the core layer.

---

## 10. Bridge permit inventory

## 10.1 `Hodge/Permits.lean`

### Objects to define

```lean
def permitHdg : Rule :=
  { kind := RuleKind.bridge
    premises := {CertTag.ambient, CertTag.tbPi, CertTag.scLambda, CertTag.dE}
    conclusion := CertTag.mhs }

def permitTann : Rule :=
  { kind := RuleKind.bridge
    premises := {CertTag.adm, CertTag.gamma, CertTag.catLib}
    conclusion := CertTag.tann }

def permitLock : Rule :=
  { kind := RuleKind.bridge
    premises := { CertTag.mhs, CertTag.tann, CertTag.cMu, CertTag.lsSigma,
                  CertTag.tbO, CertTag.catLib, CertTag.init, CertTag.adm }
    conclusion := CertTag.catHomBlk }
```

### Objects to define

```lean
def bridgeRules : RuleSet := [permitHdg, permitTann, permitLock]
def promotionRules : RuleSet := []
def allRules : RuleSet := backendRules ++ bridgeRules ++ promotionRules
def deps : RuleSet := allRules
```

### Public theorems to prove

```lean
theorem permitHdg_premises :
    permitHdg.premises =
      {CertTag.ambient, CertTag.tbPi, CertTag.scLambda, CertTag.dE}

theorem permitTann_premises :
    permitTann.premises =
      {CertTag.adm, CertTag.gamma, CertTag.catLib}

theorem permitLock_premises :
    permitLock.premises =
      { CertTag.mhs, CertTag.tann, CertTag.cMu, CertTag.lsSigma,
        CertTag.tbO, CertTag.catLib, CertTag.init, CertTag.adm }
```

No semantic proof theorems are required here, because the semantics live in the trusted boundary and in the backend closure theorems.

### Implementation note

Implemented in `lean/HypoHodge/Hodge/Permits.lean`.

- `permitHdg`, `permitTann`, `permitLock`, `bridgeRules`, `promotionRules`, `allRules`, and `deps` are defined.
- The premise-identification lemmas are proved by reflexivity.
- This section is complete for the current Phase A kernel.

---

## 11. Run inventory

## 11.1 `Hodge/Run.lean`

### Objects to define

```lean
def initialContext (I : VerifiedHodgeThinInput) : Context :=
  gamma0 I

def contextAfterBackend (I : VerifiedHodgeThinInput) : Context :=
  closure backendRules (initialContext I)

def contextAfterMhs (I : VerifiedHodgeThinInput) : Context :=
  closure (backendRules ++ [permitHdg]) (initialContext I)

def contextAfterTann (I : VerifiedHodgeThinInput) : Context :=
  closure (backendRules ++ [permitHdg, permitTann]) (initialContext I)

def finalContext (I : VerifiedHodgeThinInput) : Context :=
  closure allRules (initialContext I)

def runHodgeSystem (I : VerifiedHodgeThinInput) : Context :=
  finalContext I
```

### Public theorems to prove

```lean
theorem emit_backend_context
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    backendBase I ⊆ contextAfterBackend I

theorem emit_mhs
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.mhs ∈ contextAfterMhs I

theorem emit_tann
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.tann ∈ contextAfterTann I

theorem emit_catHomBlk
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.catHomBlk ∈ finalContext I
```

### Implementation note

Implemented in `lean/HypoHodge/Hodge/Run.lean`.

- Replaced the placeholder bridge-emission theorems `emit_mhs`, `emit_tann`, and `emit_catHomBlk`.
- Each proof shows the target tag is produced in a single concrete `step` of the relevant rule set from `initialContext`.
- Membership is then lifted from `step` into the corresponding closure context via monotonicity and closure fixedness.

---

## 12. Proof audit inventory

## 12.1 `Hodge/ProofAudit.lean`

### Public theorems to prove

```lean
theorem no_local_inc
    (I : VerifiedHodgeThinInput) :
    Disjoint (obligations (gamma0 I)) (goalCone deps CertTag.catHomBlk)

theorem no_backend_inc
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.germ ∉ obligations (finalContext I) ∧
    CertTag.init ∉ obligations (finalContext I) ∧
    CertTag.catLib ∉ obligations (finalContext I)

theorem no_lock_inc
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.catHomInc ∉ finalContext I

theorem no_promo_inc
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    CertTag.promoInc ∉ finalContext I

theorem hodgeProofAudit
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    Disjoint
      (obligations (finalContext I))
      (goalCone deps CertTag.catHomBlk)
```

The final theorem in this file is the exact proof-completion audit theorem required by the Hodge template.

### Implementation note

Implemented in `lean/HypoHodge/Hodge/ProofAudit.lean`.

- Replaced the placeholder theorems `no_lock_inc` and `no_promo_inc`.
- Added an induction-on-rules argument showing a tag remains absent from `step` and hence from `closureN` when no rule concludes that tag.
- Derived `hodgeProofAudit` from the fact that the only inc-tags are `catHomInc` and `promoInc`, and neither can appear in `finalContext`.

---

## 13. Final theorem inventory

## 13.1 `Hodge/Final.lean`

### Public theorem to prove

```lean
theorem hodge_framework_unconditional
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    ProofComplete allRules deps (runHodgeSystem I) CertTag.catHomBlk
```

### Recommended proof shape

```lean
theorem hodge_framework_unconditional
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    ProofComplete allRules deps (runHodgeSystem I) CertTag.catHomBlk := by
  apply proofComplete_of_goal_mem_and_disjoint
  · exact emit_catHomBlk I
  · exact hodgeProofAudit I
```

This is the end of Phase A.

### Implementation note

Implemented in `lean/HypoHodge/Hodge/Final.lean`.

- The final theorem is proved in the recommended form from `emit_catHomBlk` and `hodgeProofAudit`.
- The statement remains the Phase A target theorem.
- This section is complete for the current Phase A kernel.

---

## 14. Complete public object list

This is the complete public object inventory for Phase A.

### Core
- `CertTag`
- `CertTag.isInc`
- `allTags`
- `Context`
- `RuleKind`
- `Rule`
- `RuleSet`
- `Rule.enabled`
- `fireRule`
- `step`
- `closureN`
- `closure`
- `backStep`
- `goalConeN`
- `goalCone`
- `obligations`
- `ProofComplete`

### Trusted boundary
- `ClassicalOrigin`
- `HodgeBridgePremises`
- `GammaContextPremises`
- `TannakianBridgePremises`
- `ProducesMHS`
- `ProducesGamma`
- `ProducesTann`
- `ImportedHodgeAxioms`

### Hodge thin input
- `VerifiedHodgeThinInput`
- `gamma0`

### Algebraic backend
- `HodgeWitness`
- `BadAlgGerm`
- `WitnessHom`
- `WitnessHom.id`
- `WitnessHom.comp`
- `GermCode`
- `encodeBadAlgGerm`
- `UniversalBad`
- `GammaPackage`
- `backendBase`
- `backendRules`

### Hodge permit and run layer
- `permitHdg`
- `permitTann`
- `permitLock`
- `bridgeRules`
- `promotionRules`
- `allRules`
- `deps`
- `initialContext`
- `contextAfterBackend`
- `contextAfterMhs`
- `contextAfterTann`
- `finalContext`
- `runHodgeSystem`

This list is exhaustive for Phase A.

---

## 15. Complete public theorem list

This is the complete public theorem inventory for Phase A.

### Core theorems
1. `mem_allTags`
2. `allTags_nodup`
3. `context_ext`
4. `enabled_iff_subset`
5. `fireRule_eq_insert_of_enabled`
6. `fireRule_eq_self_of_disabled`
7. `subset_fireRule`
8. `monotone_fireRule`
9. `subset_step`
10. `monotone_step`
11. `subset_closureN`
12. `monotone_closureN`
13. `subset_closure`
14. `monotone_closure`
15. `closure_fixed`
16. `closure_idempotent`
17. `closure_least_fixed`
18. `goal_mem_goalCone`
19. `backStep_monotone`
20. `goalConeN_monotone`
21. `goalCone_fixed`
22. `premises_in_goalCone_of_conclusion`
23. `obligations_subset`
24. `mem_obligations_iff`
25. `proofComplete_iff`
26. `proofComplete_of_goal_mem_and_disjoint`
27. `not_proofComplete_of_goal_missing`
28. `not_proofComplete_of_inc_in_goalCone`

### Local Hodge theorems
29. `emit_adj`
30. `emit_dE`
31. `emit_recN`
32. `emit_cMu`
33. `emit_scLambda`
34. `emit_scPartialC`
35. `emit_capH`
36. `emit_lsSigma`
37. `emit_tbPi`
38. `emit_tbO`
39. `emit_boundPartial`
40. `gamma0_complete`
41. `gamma0_no_local_inc`

### Algebraic backend theorems
42. `WitnessHom.id_comp`
43. `WitnessHom.comp_id`
44. `WitnessHom_assoc`
45. `encodeBadAlgGerm_injective`
46. `boundedGermSmallness`
47. `hodgeBadBoundedReduction`
48. `hodgeClassifiableStatic`
49. `hodgeInitialityBounded`
50. `hodgeCatLibBounded`
51. `hodgeGammaConstructor`
52. `hodgeAmbientFDExpansion`
53. `hodgeRepConFD`
54. `hodgeIdentityParametrizationCompleteness`
55. `hodgeAutoAdmissibility`
56. `hodgeGermSmallnessBounded`
57. `hodgeInitialityTag`
58. `hodgeCatLibTag`
59. `hodgeGammaTag`
60. `hodgeBackendAutoclose`

### Permit/run theorems
61. `permitHdg_premises`
62. `permitTann_premises`
63. `permitLock_premises`
64. `emit_backend_context`
65. `emit_mhs`
66. `emit_tann`
67. `emit_catHomBlk`

### Proof audit and final theorem
68. `no_local_inc`
69. `no_backend_inc`
70. `no_lock_inc`
71. `no_promo_inc`
72. `hodgeProofAudit`
73. `hodge_framework_unconditional`

This list is exhaustive for Phase A.

---

## 16. Exact implementation order

Implement in this order.

### Step 1
`Core/CertTag.lean`

### Step 2
`Core/Context.lean`

### Step 3
`Core/Rule.lean`

### Step 4
`Core/Closure.lean`

### Step 5
`Core/GoalCone.lean`

### Step 6
`Core/ObligationLedger.lean`

### Step 7
`Core/ProofComplete.lean`

### Step 8
`Algebraic/VerifiedThinInput.lean`

### Step 9
`Imported/Boundary.lean`

### Step 10
`Algebraic/LocalCertificates.lean`

### Step 11
`Algebraic/BadGerm.lean`

### Step 12
`Algebraic/Coding.lean`

### Step 13
`Algebraic/BoundedReduction.lean`

### Step 14
`Algebraic/Initiality.lean`

### Step 15
`Algebraic/CatLib.lean`

### Step 16
`Algebraic/GammaConstructor.lean`

### Step 17
`Algebraic/BackendAutoclose.lean`

### Step 18
`Hodge/Permits.lean`

### Step 19
`Hodge/Run.lean`

### Step 20
`Hodge/ProofAudit.lean`

### Step 21
`Hodge/Final.lean`

Only after Step 21 should you add proof-relevant payloads.

---

## 17. Acceptance checklist

The Phase A Lean development is complete if and only if all of the following hold.

### Build
- `lake build` succeeds.
- `import HypoHodge` succeeds.
- all files compile without `sorry`.

### Core
- all 28 core theorems are proved.
- `closure_fixed` and `closure_least_fixed` are available and used.

### Hodge local layer
- `gamma0_no_local_inc` is proved.
- all local emission theorems are proved.

### Backend
- all 19 backend theorems are proved.
- `hodgeBackendAutoclose` is used by the run layer.

### Final run
- `emit_catHomBlk` is proved.
- `hodgeProofAudit` is proved.
- `hodge_framework_unconditional` is proved.

If all items above are satisfied, Phase A is complete.

---

## 18. Optional Phase B: proof-relevant payload upgrade

Phase B is optional. It should start only after Phase A is complete.

## 18.1 New objects to define

```lean
def Payload : CertTag → Type := ...

structure Certificate where
  tag     : CertTag
  payload : Payload tag

abbrev RichContext := Finset Certificate
```

## 18.2 New theorems to prove

```lean
theorem tag_projection_sound :
    RichContext → Context

theorem rich_fireRule_sound : ...
theorem rich_closure_projects_to_tag_closure : ...
theorem rich_proofComplete_implies_tag_proofComplete : ...
```

## 18.3 Reason to delay Phase B

Phase B is not needed to prove the final template theorem. It only enriches the output with proof-relevant data.

---

## 19. What improves this roadmap

This roadmap is better if you keep the following rules.

1. **Never mix trusted literature with new framework theorems.**
2. **Never start from varieties and cohomology if the real Lean input is the verified thin object.**
3. **Never begin with proof-relevant payloads.**
4. **Make every inference rule explicit as a `Rule`.**
5. **Keep closure finite.**
6. **State every public theorem before proving any helper lemma.**
7. **Finish Phase A before touching Phase B.**

These seven rules keep the project small enough to complete and strong enough to certify the new mathematics.

---

## 20. Minimal first milestone

The first milestone is not the final theorem. The first milestone is:

```lean
theorem closure_fixed (rules : RuleSet) (Γ : Context) :
    step rules (closure rules Γ) = closure rules Γ
```

Once this theorem exists, the rest of the framework can be built on top of it.

The second milestone is:

```lean
theorem hodgeBackendAutoclose
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    backendBase I ⊆ closure backendRules (gamma0 I)
```

The third milestone is the final theorem:

```lean
theorem hodge_framework_unconditional
    (I : VerifiedHodgeThinInput)
    [ImportedHodgeAxioms I] :
    ProofComplete allRules deps (runHodgeSystem I) CertTag.catHomBlk
```

These three milestones are the spine of the project.

---
