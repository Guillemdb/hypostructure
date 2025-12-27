---
title: "RESOLVE-Expansion - Complexity Theory Translation"
---

# RESOLVE-Expansion: Type Elaboration and Inference

## Overview

This document provides a complete complexity-theoretic translation of the RESOLVE-Expansion theorem (Thin-to-Full Expansion) from the hypostructure framework. The translation establishes a formal correspondence between the expansion of thin objects to full kernel objects and **type elaboration** in programming language theory, where minimal type annotations expand to full type derivations via inference.

**Original Theorem Reference:** {prf:ref}`mt-resolve-expansion`

---

## Hypostructure Context

The RESOLVE-Expansion theorem states that given thin objects $(\mathcal{X}^{\text{thin}}, \Phi^{\text{thin}}, \mathfrak{D}^{\text{thin}}, G^{\text{thin}})$, the Framework automatically constructs:

1. **Topological Structure:** SectorMap, Dictionary
2. **Singularity Detection:** Bad sets $\mathcal{X}_{\text{bad}}$, singular support $\Sigma$
3. **Profile Classification:** ProfileExtractor, canonical library
4. **Surgery Construction:** SurgeryOperator, admissibility predicates

The key insight: users provide only 10 primitive components, and the framework derives the full ~30-component kernel object specification.

---

## Complexity Theory Statement

**Theorem (Type Elaboration Completeness).**
Let $\Gamma$ be a typing context and $e$ be a term with **partial type annotations** (thin specification). Let $\mathcal{I}$ be a type inference system satisfying:

1. **Principality:** For typeable terms, there exists a most general type scheme
2. **Decidability:** Type inference terminates on all inputs
3. **Soundness:** Inferred types are valid in the full type system

Then the **elaboration function** $\mathcal{E}$ automatically constructs:

$$\mathcal{E}: (\Gamma, e^{\text{thin}}) \mapsto (e^{\text{full}}, \tau, \Pi)$$

where:
- $e^{\text{full}}$ is the fully-annotated term (explicit type applications, coercions)
- $\tau$ is the principal type of $e$
- $\Pi$ is the typing derivation (proof term)

**Guarantee:** If the thin specification satisfies basic well-formedness (scoping, arity), then elaboration produces a valid fully-typed term.

**Complexity:**
- **Hindley-Milner:** $O(n \cdot \alpha(n))$ where $\alpha$ is inverse Ackermann (nearly linear)
- **System F with inference:** Undecidable in general (Wells 1999)
- **Bidirectional with annotations:** $O(n^2)$ to $O(n^3)$ depending on constraint complexity
- **Dependent types with holes:** Semi-decidable; termination depends on user hints

---

## Terminology Translation Table

| Hypostructure Concept | Type Theory Equivalent | Formal Correspondence |
|-----------------------|------------------------|----------------------|
| Thin object $\mathcal{X}^{\text{thin}}$ | Partially-annotated term $e^{\text{thin}}$ | Minimal specification with inference holes |
| Full kernel object $\mathcal{H}^{\text{full}}$ | Fully-elaborated term $e^{\text{full}}$ | All types explicit, all coercions inserted |
| Space $\mathcal{X}$ | Base type / carrier set | The domain of values |
| Metric $d$ | Subtyping distance | Coercion cost between types |
| Measure $\mu$ | Type inhabitance | Density of values in a type |
| Energy $\Phi$ | Type complexity measure | Size of type derivation |
| Scaling $\alpha$ | Type polymorphism degree | Number of quantified variables |
| Dissipation $\mathfrak{D}$ | Constraint solving progress | Reduction in unification problem size |
| Symmetry $G$ | Type equivalence / isomorphism | $\alpha$-equivalence, type isomorphism |
| SectorMap | Module structure | Namespace/scope resolution |
| Dictionary | Type environment $\Gamma$ | Mapping from variables to types |
| ProfileExtractor | Principal type algorithm | Algorithm $\mathcal{W}$ or similar |
| Bad set $\mathcal{X}_{\text{bad}}$ | Untypeable terms | Terms with no valid typing |
| Singular support $\Sigma$ | Type errors | Locations where inference fails |
| SurgeryOperator | Type hole filling | Elaboration of implicit arguments |
| Admissibility predicate | Well-typedness check | Decidable type checking |
| Consistency hypotheses | Well-scoped, well-formed | Lexical correctness of input |
| Expansion function | Elaboration $\mathcal{E}$ | Thin $\mapsto$ Full transformation |
| Concentration-compactness | Constraint propagation | Local decisions propagate globally |

---

## Type-Theoretic Framework

### The Elaboration Problem

**Definition (Thin Specification).**
A thin specification consists of:

1. **Term skeleton** $e$: The syntactic structure with type holes `_`
2. **Sparse annotations** $A$: User-provided type annotations at key positions
3. **Context** $\Gamma$: Variable bindings from enclosing scope

**Definition (Full Elaboration).**
A full elaboration consists of:

1. **Explicit term** $e^{\text{full}}$: All implicit arguments made explicit
2. **Type assignment** $\tau$: Complete type for the term
3. **Derivation** $\Pi$: Proof tree witnessing $\Gamma \vdash e^{\text{full}} : \tau$
4. **Coercions** $C$: Inserted subtyping witnesses
5. **Dictionary** $\Delta$: Instance resolutions (for type classes/implicits)

### Hindley-Milner Type Inference

**System HM** is the canonical example of thin-to-full expansion:

**Thin Input:**
```
let id = \x -> x
let compose = \f -> \g -> \x -> f (g x)
```

**Full Output:**
```
let id : forall a. a -> a = /\a -> \(x : a) -> x
let compose : forall a b c. (b -> c) -> (a -> b) -> (a -> c)
            = /\a -> /\b -> /\c -> \(f : b -> c) -> \(g : a -> b) -> \(x : a) -> f (g x)
```

**Algorithm W** (Damas-Milner 1982) performs this expansion:

1. **Generate constraints:** Traverse term, emit type equations
2. **Solve constraints:** Unification algorithm produces most general unifier
3. **Generalize:** Quantify over free type variables
4. **Elaborate:** Insert type abstractions and applications

---

## Proof Sketch

### Setup: The Elaboration Correspondence

We establish the correspondence:

| Hypostructure | Type Theory |
|---------------|-------------|
| $(\mathcal{X}, d, \mu)$ | Type universe $(U, \leq, \text{size})$ |
| $(\Phi, \nabla, \alpha)$ | Constraint system $(C, \text{solve}, \text{rank})$ |
| $(\mathfrak{D}, \beta)$ | Unification $(U, \text{mgu})$ |
| $(G, \rho, \mathcal{S})$ | Type equivalence $(\cong, \text{rename}, \text{inst})$ |

**Expansion Function:**
$$\text{Expand}: (\Gamma, e^{\text{thin}}) \mapsto (\Gamma', e^{\text{full}}, \tau, \Pi)$$

### Step 1: Topological Construction (Context Building)

**Claim:** Given term skeleton $e$ and context $\Gamma$, compute the extended context $\Gamma'$ and scope structure.

**Construction (SectorMap = Module Resolution):**

For each subterm of $e$:
1. **Free variables:** Look up in $\Gamma$ (the Dictionary)
2. **Bound variables:** Extend context with fresh type variables
3. **Nested scopes:** Maintain stack of local contexts

**Formally:**
$$\text{SectorMap}(e) = \pi_0(\text{FV}(e)) \cup \text{BV}(e)$$

where $\pi_0$ denotes connected components (modules/namespaces).

**Dictionary Construction:**
$$\text{Dictionary} = \Gamma \cup \{x : \alpha_x \mid x \in \text{BV}(e), \alpha_x \text{ fresh}\}$$

**Correspondence to Hypostructure:**
- $\pi_0(\mathcal{X})$ (path components) $\leftrightarrow$ Module/namespace structure
- Dimension extraction $\leftrightarrow$ Type arity/kind inference

### Step 2: Constraint Generation (Energy Functional)

**Claim:** Given $(e, \Gamma')$, generate constraint set $C$ representing type equations.

**Algorithm (Constraint Harvesting):**

```
Generate(e, Gamma) -> (tau, C):
  match e with
  | x         -> (Gamma(x), {})
  | \x -> e'  -> let (tau', C') = Generate(e', Gamma[x:alpha])
                 in (alpha -> tau', C')
  | e1 e2     -> let (tau1, C1) = Generate(e1, Gamma)
                 let (tau2, C2) = Generate(e2, Gamma)
                 in (beta, C1 ++ C2 ++ {tau1 = tau2 -> beta})
  | let x = e1 in e2 ->
                 let (tau1, C1) = Generate(e1, Gamma)
                 let sigma = Generalize(Solve(C1), tau1)
                 let (tau2, C2) = Generate(e2, Gamma[x:sigma])
                 in (tau2, C2)
```

**Energy Interpretation:**
- $\Phi(e) = |C|$ (constraint set size)
- $\nabla \Phi$ corresponds to constraint propagation direction
- Scaling $\alpha$ = polymorphism rank (number of $\forall$ quantifiers)

**Correspondence to Hypostructure:**
- Energy $\Phi$ bounded $\leftrightarrow$ Finite constraint set
- Lower semicontinuity $\leftrightarrow$ Monotonic constraint growth

### Step 3: Constraint Solving (Profile Extraction)

**Claim:** Unification extracts the principal type (canonical profile).

**Unification Algorithm (Robinson 1965, Martelli-Montanari 1982):**

```
Unify(C) -> Substitution:
  match C with
  | {}                    -> id
  | {tau = tau} ++ C'     -> Unify(C')
  | {alpha = tau} ++ C'   -> if alpha in FV(tau) then FAIL (occurs check)
                             else [alpha := tau] . Unify(C'[alpha := tau])
  | {tau1 -> tau2 = tau3 -> tau4} ++ C'
                          -> Unify({tau1 = tau3, tau2 = tau4} ++ C')
  | {C1 = C2} ++ C'       -> FAIL (rigid mismatch)
```

**Profile Extraction Correspondence:**

| Hypostructure | Type Inference |
|---------------|----------------|
| Scaling group $\mathcal{S}$ | Type instantiation |
| Profile $V = \lim \lambda^{-\alpha} x$ | Principal type $\sigma = \forall \bar{\alpha}. \tau$ |
| Concentration-compactness | Occurs check / constraint propagation |
| Moduli space $\mathcal{X} // G$ | Types modulo $\alpha$-equivalence |

**Principal Type Theorem (Hindley 1969, Milner 1978):**

For HM-typeable terms, Algorithm W computes the principal (most general) type:
$$\forall \sigma'. (\Gamma \vdash e : \sigma') \Rightarrow \sigma \sqsubseteq \sigma'$$

where $\sqsubseteq$ denotes instantiation ordering.

**Correspondence:** The principal type is the "canonical profile" extracted by the scaling limit. All other valid types are instances (specializations) of the principal type.

### Step 4: Elaboration (Surgery Construction)

**Claim:** Given solution $\theta = \text{Unify}(C)$ and type $\tau$, construct the fully-elaborated term.

**Elaboration as Surgery:**

The elaboration inserts:
1. **Type abstractions** $\Lambda \alpha. e$ for generalized variables
2. **Type applications** $e[\tau]$ for instantiations
3. **Coercions** for subtyping witnesses
4. **Dictionary passing** for type class instances

**Surgery Operator Correspondence:**

$$\text{Surgery}(e^{\text{thin}}) = \text{colim}\left(e|_{\text{annotated}} \leftarrow e|_{\text{boundary}} \to \text{inferred}\right)$$

This is the pushout that glues user annotations to inferred types along their interface.

**Implicit Argument Synthesis:**

In systems with implicit arguments (Agda, Idris, Coq), elaboration fills holes:

```
Thin:  f _ x
Full:  f {A = Nat} {B = Bool} x
```

The Framework derives implicit arguments from:
- Unification constraints
- Instance search (type classes)
- Tactic invocation (in dependent types)

### Step 5: Validation

**Claim:** Elaborated terms are well-typed.

**Type Checking (Decidable for Elaborated Terms):**

After elaboration, type checking is:
- **Syntax-directed:** Each term form has unique typing rule
- **Compositional:** Subterm types determine term type
- **Efficient:** Linear in term size

**Soundness Theorem:**
$$\text{Elaborate}(\Gamma, e^{\text{thin}}) = (\Gamma', e^{\text{full}}, \tau, \Pi) \Rightarrow \Gamma' \vdash e^{\text{full}} : \tau$$

**Correspondence to Hypostructure:**
- Consistency hypotheses $\leftrightarrow$ Well-scoped input
- Valid Kernel Objects $\leftrightarrow$ Well-typed elaborated term
- Interface satisfaction $\leftrightarrow$ Type checking success

---

## Bidirectional Type Checking

Bidirectional type checking provides an efficient algorithm for elaboration that closely mirrors the thin-to-full expansion.

### The Bidirectional Discipline

**Two Modes:**

1. **Checking mode** $\Gamma \vdash e \Leftarrow \tau$: Given expected type, check term
2. **Inference mode** $\Gamma \vdash e \Rightarrow \tau$: Synthesize type from term

**Key Rules:**

```
                    Gamma |- e => tau'    tau' = tau
Check-Infer:       ----------------------------------------
                           Gamma |- e <= tau

                    Gamma, x:tau1 |- e <= tau2
Lam-Check:         --------------------------------
                    Gamma |- \x -> e <= tau1 -> tau2

                    Gamma |- e1 => tau1 -> tau2    Gamma |- e2 <= tau1
App-Infer:         ------------------------------------------------
                              Gamma |- e1 e2 => tau2

                    Gamma |- e <= tau
Annot-Infer:       -------------------------
                    Gamma |- (e : tau) => tau
```

### Correspondence to Thin Objects

| Thin Object Component | Bidirectional Mode | Role |
|-----------------------|-------------------|------|
| User annotation | Checking mode | Provides expected type |
| Inferred component | Inference mode | Synthesizes type |
| Interface boundary | Mode switch | Annotation insertion point |
| Consistency check | Subsumption | $\tau' \sqsubseteq \tau$ verification |

**Minimal Annotation Principle:**

Bidirectional systems require annotations only at:
1. **Top-level bindings:** `f : tau`
2. **Polymorphic instantiation:** `f @tau`
3. **Ambiguous constructs:** Empty lists, numeric literals

This corresponds to the "10 primitive components" in the hypostructure thin objects.

### Spine-Local Type Inference

**Observation:** In application spines $f e_1 e_2 \ldots e_n$, type information flows:
- **Left-to-right:** From function to arguments (checking)
- **Right-to-left:** From arguments to result (synthesis)

**Spine-local inference** (Dunfield & Krishnaswami 2013) exploits this bidirectional flow:

```
Gamma |- f => forall a. tau1 -> tau2
Gamma |- e <= tau1[a := ?]           -- solve ?a from e
-----------------------------------------
Gamma |- f e => tau2[a := solution]
```

**Correspondence:** This is analogous to the concentration-compactness profile extraction: local information (argument types) determines global structure (instantiation).

---

## Dependent Type Elaboration

In dependently-typed systems, elaboration is more complex but follows the same pattern.

### The Elaboration Problem in Dependent Types

**Input (Thin):**
```agda
id : (A : Type) -> A -> A
id _ x = x

compose : (A B C : Type) -> (B -> C) -> (A -> B) -> A -> C
compose _ _ _ f g x = f (g x)
```

**Output (Full):**
```agda
id : (A : Type) -> A -> A
id A x = x

compose : (A B C : Type) -> (B -> C) -> (A -> B) -> A -> C
compose A B C f g x = f (g x)
```

### Implicit Arguments

**Thin specification with implicits:**
```agda
id : {A : Type} -> A -> A
id x = x

-- Usage (thin): id 42
-- Usage (full): id {Nat} 42
```

**Elaboration fills implicits via unification:**

1. Generate metavariable $?A$ for implicit argument
2. From `42 : Nat`, constrain $?A = \text{Nat}$
3. Solve and substitute: `id {Nat} 42`

### Instance Arguments (Type Classes)

**Thin:**
```agda
show : {A : Type} -> {{Show A}} -> A -> String
show x = ...

-- Usage: show 42
```

**Full (after instance resolution):**
```agda
show {Nat} {{showNat}} 42
```

**Instance Search as Profile Classification:**

The instance search algorithm:
1. Collects candidate instances from context
2. Filters by type matching
3. Selects unique instance or reports ambiguity

This corresponds to the ProfileExtractor selecting from the canonical library.

### Holes and Tactics

**Thin (with holes):**
```agda
lemma : (n : Nat) -> n + 0 == n
lemma n = {!!}
```

**Elaboration (interactive or automatic):**
```agda
lemma : (n : Nat) -> n + 0 == n
lemma zero = refl
lemma (suc n) = cong suc (lemma n)
```

**Tactics as Surgery Operators:**

| Tactic | Surgery Analogue |
|--------|------------------|
| `auto` | Automatic surgery construction |
| `rewrite` | Local excision and replacement |
| `induction` | Profile decomposition |
| `apply` | Profile matching from library |

---

## Connections to Classical Results

### 1. Hindley-Milner Type Inference (Damas & Milner 1982)

**Theorem (Principal Types):** In System HM, every typeable term has a principal type computable in nearly linear time.

**Connection to RESOLVE-Expansion:**

| Hypostructure | HM Type Inference |
|---------------|-------------------|
| Thin objects | Unannotated term |
| Full kernel | Fully-typed term |
| Expand function | Algorithm W |
| Consistency | Well-scoped |
| Profile extraction | Principal type |
| Canonical library | Polymorphic type schemes |

**Complexity:** $O(n \cdot \alpha(n))$ where $\alpha$ is inverse Ackermann function.

### 2. Wells' Undecidability (1999)

**Theorem:** Type inference for System F (second-order lambda calculus) is undecidable.

**Connection:** This corresponds to the distinction between:
- **Good types** (HM): Automatic expansion succeeds
- **Bad types** (System F): Expansion may require user hints

**Resolution via Annotations:**

Bidirectional checking restores decidability by requiring annotations at polymorphic instantiations:
```
f : forall a. a -> a
f @Int 42    -- annotation needed
```

### 3. Local Type Inference (Pierce & Turner 2000)

**Principle:** Infer types locally; propagate bidirectionally.

**Key insight:** Type information flows:
- **Downward** (checking): From context to subterms
- **Upward** (synthesis): From subterms to context

**Connection to Hypostructure:**

The bidirectional flow corresponds to:
- Energy dissipation (checking): Known structure constrains subproblems
- Profile extraction (synthesis): Local information reveals global structure

### 4. Colored Local Type Inference (Odersky et al. 2001)

**Extension:** Track inference direction via "colors" on type constructors.

**Application:** Scala's type inference uses colored constraints to:
- Propagate expected types into lambda bodies
- Synthesize result types from expressions
- Handle variance correctly in generics

### 5. Complete and Easy Bidirectional Type Checking (Dunfield & Krishnaswami 2013, 2019)

**Theorem:** Bidirectional type checking for higher-rank polymorphism is:
- **Complete:** All HM-typeable terms are accepted
- **Sound:** All accepted terms are well-typed
- **Efficient:** Polynomial time with minimal annotations

**Connection to RESOLVE-Expansion:**

The "complete and easy" algorithm is the computational realization of thin-to-full expansion:
- Minimal annotations (thin) suffice for full elaboration
- Elaboration is deterministic and efficient
- No backtracking required (unlike Prolog-style inference)

---

## Certificate Construction

The elaboration process produces explicit certificates:

**Elaboration Certificate $K_{\text{Elab}}$:**

```
K_Elab := (
    thin_term        : e with holes/implicits,
    full_term        : e' with all types explicit,
    principal_type   : sigma = forall alpha_bar. tau,
    derivation       : Pi :: Gamma |- e' : sigma,
    constraint_log   : sequence of unification steps,
    instance_dict    : resolution of type class instances,
    coercions        : inserted subtyping witnesses
)
```

**Verification (Type Checking):**

Given $K_{\text{Elab}}$, verify:
1. $\Gamma \vdash e' : \sigma$ (derivation is valid)
2. $e' \sim e$ (elaboration preserves semantics)
3. $\sigma$ is principal (most general)

**Complexity:** Verification is linear in derivation size, always decidable.

---

## Quantitative Bounds

### Constraint Size

For term $e$ of size $n$:
- **Hindley-Milner:** $|C| = O(n)$ constraints
- **Higher-rank:** $|C| = O(n^2)$ constraints
- **Dependent types:** $|C|$ unbounded (may require user input)

### Unification Complexity

- **First-order unification:** $O(n \cdot \alpha(n))$ (nearly linear)
- **Higher-order unification:** Undecidable
- **Pattern unification:** $O(n^2)$ (Miller 1991)

### Elaboration Time

| System | Inference | Checking |
|--------|-----------|----------|
| HM | $O(n \cdot \alpha(n))$ | $O(n)$ |
| Bidirectional | $O(n^2)$ | $O(n)$ |
| Dependent (decidable fragment) | $O(n^3)$ | $O(n^2)$ |

---

## Worked Example: Let-Polymorphism

**Thin Input:**
```ml
let compose = fun f -> fun g -> fun x -> f (g x) in
let id = fun x -> x in
compose id id 42
```

**Elaboration Trace:**

1. **Generate constraints for `compose`:**
   - $f : \alpha_f$, $g : \alpha_g$, $x : \alpha_x$
   - $g\ x : \beta_1$ implies $\alpha_g = \alpha_x \to \beta_1$
   - $f\ (g\ x) : \beta_2$ implies $\alpha_f = \beta_1 \to \beta_2$
   - `compose` $: \alpha_f \to \alpha_g \to \alpha_x \to \beta_2$

2. **Solve:**
   - Substitution: $\{\alpha_g = \alpha_x \to \beta_1, \alpha_f = \beta_1 \to \beta_2\}$
   - Type: $(\beta_1 \to \beta_2) \to (\alpha_x \to \beta_1) \to \alpha_x \to \beta_2$

3. **Generalize:**
   - Principal type: $\forall a\ b\ c.\ (b \to c) \to (a \to b) \to a \to c$

4. **Elaborate:**
   ```ml
   let compose : forall a b c. (b -> c) -> (a -> b) -> a -> c =
     /\a -> /\b -> /\c ->
       fun (f : b -> c) -> fun (g : a -> b) -> fun (x : a) -> f (g x) in
   let id : forall a. a -> a = /\a -> fun (x : a) -> x in
   compose [Int] [Int] [Int] (id [Int]) (id [Int]) 42
   ```

**Correspondence:**
- Thin: 3 lines, no type annotations
- Full: Explicit type abstractions, applications, annotations
- Expansion ratio: ~3x in syntax, but semantically equivalent

---

## Summary

The RESOLVE-Expansion theorem, translated to type theory, states:

**Minimal type annotations (thin objects) expand to complete type derivations (full kernel objects) via elaboration algorithms.**

This principle:
1. **Justifies implicit programming:** Users need only provide key annotations
2. **Guarantees completeness:** Typeable terms always elaborate successfully
3. **Provides principal types:** Elaboration finds the most general solution
4. **Enables efficient checking:** Elaborated terms check in linear time

The translation illuminates deep connections:

| Hypostructure | Type Theory |
|---------------|-------------|
| Concentration-compactness | Constraint propagation |
| Profile extraction | Principal type inference |
| Surgery construction | Implicit argument synthesis |
| Canonical library | Instance/type class resolution |
| Thin-to-full expansion | Elaboration |

**Key Insight:** Just as the hypostructure framework reduces user burden from ~30 components to 10 primitives, type inference reduces annotation burden from every subterm to a sparse set of key positions. Both achieve this via algorithms that propagate local information to derive global structure.

---

## Literature

1. **Damas, L. & Milner, R. (1982).** "Principal Type-Schemes for Functional Programs." POPL. *Foundational HM type inference.*

2. **Hindley, J. R. (1969).** "The Principal Type-Scheme of an Object in Combinatory Logic." *Transactions AMS.* *Principal types for combinators.*

3. **Milner, R. (1978).** "A Theory of Type Polymorphism in Programming." *JCSS.* *ML type system.*

4. **Robinson, J. A. (1965).** "A Machine-Oriented Logic Based on the Resolution Principle." *JACM.* *Unification algorithm.*

5. **Wells, J. B. (1999).** "Typability and Type Checking in System F are Equivalent and Undecidable." *Annals of Pure and Applied Logic.* *System F undecidability.*

6. **Pierce, B. C. & Turner, D. N. (2000).** "Local Type Inference." *TOPLAS.* *Bidirectional foundations.*

7. **Dunfield, J. & Krishnaswami, N. (2013).** "Complete and Easy Bidirectional Typechecking for Higher-Rank Polymorphism." *ICFP.* *Modern bidirectional.*

8. **Dunfield, J. & Krishnaswami, N. (2019).** "Bidirectional Typing." *arXiv.* *Comprehensive survey.*

9. **Odersky, M. et al. (2001).** "Colored Local Type Inference." *POPL.* *Scala's inference.*

10. **Miller, D. (1991).** "A Logic Programming Language with Lambda-Abstraction, Function Variables, and Simple Unification." *JLC.* *Pattern unification.*

11. **Norell, U. (2007).** "Towards a Practical Programming Language Based on Dependent Type Theory." *PhD Thesis.* *Agda elaboration.*

12. **Brady, E. (2013).** "Idris, a General-Purpose Dependently Typed Programming Language." *JFP.* *Idris elaboration.*

13. **Coq Development Team (2021).** *The Coq Proof Assistant Reference Manual.* *Coq elaboration and tactics.*

14. **Lions, P.-L. (1984).** "The Concentration-Compactness Principle in the Calculus of Variations." *Annales IHP.* *Concentration-compactness (hypostructure source).*
