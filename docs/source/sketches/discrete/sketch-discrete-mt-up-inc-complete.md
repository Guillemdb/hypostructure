---
title: "UP-IncComplete - Complexity Theory Translation"
---

# UP-IncComplete: Premise Completion via Auxiliary Lemmas

## Overview

This document provides a complete complexity-theoretic translation of the UP-IncComplete theorem (Inconclusive Discharge by Missing-Premise Completion) from the hypostructure framework. The translation establishes a formal correspondence between inconclusive certificates that discharge via premise completion, and proof completion techniques that complete partial proofs through auxiliary lemma synthesis or axiom injection.

**Original Theorem Reference:** {prf:ref}`mt-up-inc-complete`

---

## Original Theorem (Hypostructure Context)

**Context:** A node returns $K_P^{\mathrm{inc}} = (\mathsf{obligation}, \mathsf{missing}, \mathsf{code}, \mathsf{trace})$ where $\mathsf{missing}$ specifies the certificate types that would enable decision.

**Hypotheses:** For each $m \in \mathsf{missing}$, the context $\Gamma$ contains a certificate $K_m^+$ such that:
$$\bigwedge_{m \in \mathsf{missing}} K_m^+ \Rightarrow \mathsf{obligation}$$

**Statement:** The inconclusive permit upgrades immediately to YES:
$$K_P^{\mathrm{inc}} \wedge \bigwedge_{m \in \mathsf{missing}} K_m^+ \Rightarrow K_P^+$$

**Certificate Logic:**
$$\mathsf{Obl}(\Gamma) \setminus \{(\mathsf{id}_P, \ldots)\} \cup \{K_P^+\}$$

**Core Insight:** Inconclusive verdicts are not failures; they are requests for additional premises. When those premises become available, the inconclusive certificate immediately upgrades to a positive certificate.

---

## Complexity Theory Statement

**Theorem (UP-IncComplete, Proof-Theoretic Form).**
Let $\mathcal{P} = (\Gamma, \varphi, \pi_{\text{partial}})$ be a partial proof context with:
- Axiom/lemma context $\Gamma$
- Goal formula $\varphi$
- Partial proof $\pi_{\text{partial}}$ with holes (missing subproofs)

Suppose $\pi_{\text{partial}}$ has **hole annotations** specifying the types of missing subproofs:
$$\mathsf{holes}(\pi_{\text{partial}}) = \{(h_1, \psi_1), (h_2, \psi_2), \ldots, (h_k, \psi_k)\}$$

where each $(h_i, \psi_i)$ indicates hole $h_i$ requires a proof of $\psi_i$.

**Statement (Proof Completion):**
If the context $\Gamma$ is extended with lemmas $\{L_1, \ldots, L_k\}$ such that:
$$\forall i. \Gamma \vdash L_i : \psi_i$$

Then the partial proof completes:
$$\pi_{\text{partial}}[h_1 \mapsto L_1, \ldots, h_k \mapsto L_k] : \varphi$$

**Corollary (Lemma Injection).**
For any incomplete proof, the following are equivalent:
1. The proof can be completed by lemma injection
2. All hole obligations are derivable from the extended context
3. The inconclusive certificate upgrades to positive

---

## Terminology Translation Table

| Hypostructure Concept | Complexity/Proof Theory Analog | Formal Correspondence |
|-----------------------|--------------------------------|------------------------|
| Inconclusive certificate $K_P^{\mathrm{inc}}$ | Partial proof with holes | Incomplete derivation tree |
| Missing set $\mathsf{missing}$ | Hole obligations $\{(h_i, \psi_i)\}$ | Subgoals requiring proofs |
| Obligation $\mathsf{obligation}$ | Goal formula $\varphi$ | Top-level theorem statement |
| Certificate type | Lemma type / proposition | What must be proved |
| Context $\Gamma$ | Axiom/lemma context | Available assumptions |
| Prerequisite certificate $K_m^+$ | Auxiliary lemma $L_m$ | Proof of subgoal |
| Immediate upgrade | Proof completion | Filling holes with lemmas |
| Blocked certificate $K^{\mathrm{blk}}$ | Unprovable subgoal | No available lemma |
| Code $\mathsf{code}$ | Proof term structure | Derivation skeleton |
| Trace $\mathsf{trace}$ | Proof search trace | How incomplete proof was constructed |
| Obligation ledger $\mathsf{Obl}(\Gamma)$ | Proof obligation tracker | Outstanding subgoals |
| Certificate upgrade $K^{\mathrm{inc}} \Rightarrow K^+$ | Proof completion | All holes filled |
| Interface permit | Exported lemma | Usable theorem |

---

## Proof-Theoretic Framework

### Partial Proofs and Holes

**Definition (Partial Proof).**
A partial proof $\pi_{\text{partial}}$ is a derivation tree where some leaves are labeled with **holes** rather than axioms or assumptions:

```
        ?h1 : A    B ⊢ B
        ─────────────────
              A ∧ B
              ─────── (∧-E1)
                 A
              ────────────── (→-I)
              A ∧ B → A
```

Here `?h1 : A` is a hole requiring a proof of $A$.

**Definition (Hole Context).**
The hole context of a partial proof extracts all obligations:
$$\mathsf{holes}(\pi) = \{(h, \psi) : h \text{ is a hole in } \pi \text{ with type } \psi\}$$

**Definition (Proof Completion).**
Given partial proof $\pi$ and substitution $\sigma : \text{holes} \to \text{proofs}$:
$$\pi[\sigma] = \pi[h_1 \mapsto \sigma(h_1), \ldots, h_k \mapsto \sigma(h_k)]$$

The completion is valid iff each $\sigma(h_i) : \psi_i$ where $(h_i, \psi_i) \in \mathsf{holes}(\pi)$.

### Correspondence to Inconclusive Certificates

| Hypostructure | Proof Theory |
|---------------|--------------|
| $K_P^{\mathrm{inc}} = (\mathsf{obl}, \mathsf{miss}, \mathsf{code}, \mathsf{trace})$ | $(\varphi, \mathsf{holes}(\pi), \pi_{\text{partial}}, \text{trace})$ |
| $\mathsf{missing} = \{m_1, \ldots, m_k\}$ | $\mathsf{holes} = \{(h_1, \psi_1), \ldots, (h_k, \psi_k)\}$ |
| $K_{m_i}^+$ in context | Lemma $L_i : \psi_i$ available |
| $K_P^{\mathrm{inc}} \wedge \bigwedge K_{m_i}^+ \Rightarrow K_P^+$ | $\pi[\sigma]$ is complete proof |

---

## Proof Sketch

### Step 1: Incomplete Proof Recognition

**Claim:** A proof attempt may fail to complete not due to invalidity, but due to missing auxiliary lemmas.

**Setup:** Consider proving $\varphi$ in context $\Gamma$. The proof search proceeds:

```
ProofSearch(Gamma, phi):
  attempt := try_tactics(Gamma, phi)

  if attempt = Complete(proof):
    return K_phi^+ = (Gamma, phi, proof)

  elif attempt = Stuck(partial_proof, missing_subgoals):
    // Cannot proceed, but not refuted
    return K_phi^inc = (phi, missing_subgoals, partial_proof, trace)

  elif attempt = Refuted(counterexample):
    return K_phi^- = (phi, counterexample)
```

**Key Distinction:**
- **Complete ($K^+$):** All subgoals discharged
- **Refuted ($K^-$):** Counterexample found
- **Inconclusive ($K^{\text{inc}}$):** Subgoals remain but no refutation

**Certificate Produced:** $K_\varphi^{\text{inc}} = (\varphi, \mathsf{missing}, \pi_{\text{partial}}, \text{trace})$

---

### Step 2: Missing Premise Identification

**Claim:** The $\mathsf{missing}$ set precisely identifies what additional facts would complete the proof.

**Algorithm (Hole Extraction):**
```
ExtractMissing(partial_proof):
  missing := {}

  for each hole h in partial_proof:
    psi := type_of_hole(h)
    context_at_h := local_assumptions(h)

    missing := missing ∪ {(h, psi, context_at_h)}

  return missing
```

**Properties of Missing Set:**
1. **Completeness:** If all $\psi_i$ are provable, then $\varphi$ is provable
2. **Minimality:** Each $\psi_i$ is necessary (removing any leaves proof incomplete)
3. **Typed:** Each hole has a specific type constraint

**Example:**
```
Goal: Prove "A ∧ B → C" given only "A → C" in context

Partial proof:
    ?h1 : B → C    A → C ⊢ A → C
    ─────────────────────────────
    (A → C) ∧ (B → C)
    ────────────────────────────── (∧-E2)
    B → C
    ... → C

Missing: {(h1, B → C)}
```

**Correspondence:** $\mathsf{missing} = \{(h_i, \psi_i)\}$ specifies exactly what certificates enable decision.

---

### Step 3: Context Extension and Lemma Injection

**Claim:** When missing premises become available, the inconclusive certificate upgrades.

**Definition (Context Extension).**
An extension of $\Gamma$ is $\Gamma' = \Gamma \cup \{L_1 : \psi_1, \ldots, L_k : \psi_k\}$ where each $L_i$ is a new lemma.

**Algorithm (Lemma Injection):**
```
InjectLemmas(K_inc, Gamma_extended):
  (phi, missing, partial, trace) := K_inc

  for each (h, psi) in missing:
    if exists L in Gamma_extended such that Gamma ⊢ L : psi:
      sigma[h] := L
    else:
      return K_inc  // Still inconclusive

  complete_proof := partial[sigma]

  // Verify completion
  if type_check(Gamma_extended, complete_proof, phi):
    return K_phi^+ = (Gamma_extended, phi, complete_proof)
  else:
    return K_phi^- = ("Completion failed", ...)
```

**Upgrade Rule:**
$$\frac{K_\varphi^{\text{inc}} \quad \forall (h_i, \psi_i) \in \mathsf{missing}. \exists L_i \in \Gamma'. \Gamma \vdash L_i : \psi_i}{K_\varphi^+}$$

**Certificate Produced:** $K_\varphi^+ = (\Gamma', \varphi, \pi[\sigma])$

---

### Step 4: Auxiliary Axiom Systems

**Claim:** Lemma injection is equivalent to working in an extended axiom system.

**Definition (Axiom Extension).**
Given base theory $T$ and additional axioms $A = \{A_1, \ldots, A_k\}$:
$$T + A = T \cup A$$

**Conservativity:** If $T + A \vdash \varphi$ and $\varphi$ is in the language of $T$, then $T + A$ is a **conservative extension** if $T \vdash \varphi$ implies the same.

**Non-Conservativity:** Lemma injection may be non-conservative when lemmas add new information.

**Correspondence:**

| Certificate Upgrade | Axiom Extension |
|--------------------|-----------------|
| $K^{\text{inc}} \to K^+$ | $T + A \vdash \varphi$ |
| Missing premises | Required axioms $A$ |
| Upgrade validity | Consistency of $T + A$ |
| Soundness | No false theorems derivable |

---

### Step 5: Proof Assistant Implementation

**Claim:** Modern proof assistants implement UP-IncComplete via interactive hole-filling.

**Coq-Style Proof State:**
```coq
Theorem example : A /\ B -> C.
Proof.
  intro H.
  destruct H as [Ha Hb].
  (* Proof state: Ha : A, Hb : B |- C *)
  (* Two possible continuations: *)

  (* Option 1: User provides lemma *)
  apply (lemma_A_to_C Ha).  (* Fills hole with imported lemma *)

  (* Option 2: Hole remains, proof is incomplete *)
  (* State: K^inc with missing = [{?Goal : C}] *)
Admitted.  (* Marks as inconclusive *)
```

**Lean-Style Elaboration:**
```lean
theorem example : A ∧ B → C := by
  intro ⟨ha, hb⟩
  -- Elaborator creates: ?m : C
  -- If solved: K^+
  -- If unsolved: K^inc with missing = [(?m, C)]
  sorry  -- Inconclusive marker
```

**Agda-Style Holes:**
```agda
example : A ∧ B → C
example (a , b) = {! !}  -- Hole of type C
-- Agda reports: ?0 : C
-- This is K^inc with missing = [(?0, C)]
```

**Implementation Pattern:**

| Proof Assistant | Inconclusive Marker | Hole Syntax | Upgrade Mechanism |
|-----------------|--------------------|--------------|--------------------|
| Coq | `Admitted` | `_` or `?Goal` | `apply lemma` |
| Lean | `sorry` | `_` or `?m` | `exact lemma` |
| Agda | `{! !}` | `?` | Hole filling |
| Isabelle | `sorry` | `?thesis` | `using lemma` |

---

## Connections to Proof Assistants

### 1. Elaboration with Metavariables

**Definition (Metavariable).**
A metavariable $?m$ is a placeholder for an unknown term/proof. The elaborator tracks:
$$\text{MetaContext} = \{(?m_1, \Gamma_1 \vdash \tau_1), \ldots, (?m_k, \Gamma_k \vdash \tau_k)\}$$

**Correspondence to UP-IncComplete:**

| Elaboration | Hypostructure |
|-------------|---------------|
| Metavariable $?m$ | Missing certificate $m$ |
| $\Gamma_i \vdash \tau_i$ | Certificate type specification |
| Instantiation $?m := t$ | Certificate arrival $K_m^+$ |
| Fully instantiated | Upgraded to $K^+$ |

**Algorithm (Unification-Based Completion):**
```
Elaborate(term, expected_type):
  (elaborated, metas) := infer_with_holes(term)

  for each ?m in metas:
    solution := unify_or_search(?m.type, expected_type, context)
    if solution found:
      instantiate(?m, solution)
    else:
      mark_as_hole(?m)

  if all metas instantiated:
    return K^+ = (term, elaborated)
  else:
    return K^inc = (term, remaining_metas, partial_elaboration)
```

### 2. Tactic-Based Proof Search

**Definition (Tactic).**
A tactic transforms a proof goal into zero or more subgoals:
$$\text{tactic} : \text{Goal} \to \text{List}(\text{Goal}) + \text{Proof}$$

**Correspondence:**

| Tactic Result | Certificate Type |
|---------------|------------------|
| `Proof p` | $K^+$ (complete) |
| `Goals [g1, ..., gk]` | $K^{\text{inc}}$ with $\mathsf{missing} = \{g_1, \ldots, g_k\}$ |
| `Fail` | $K^-$ (refuted) |

**Example (Coq `auto` Tactic):**
```coq
Goal A -> B -> A /\ B.
Proof.
  auto.  (* Succeeds: K^+ *)
Qed.

Goal A -> C.  (* C unrelated to A *)
Proof.
  auto.  (* Fails to complete: returns K^inc or tries other tactics *)
```

### 3. Lemma Databases and Hint Systems

**Definition (Hint Database).**
A hint database $\mathcal{H}$ is a collection of lemmas available for automatic proof search:
$$\mathcal{H} = \{(L_1, \text{priority}_1), \ldots, (L_n, \text{priority}_n)\}$$

**UP-IncComplete with Hints:**
```
AutoWithHints(K_inc, hints):
  (phi, missing, partial, trace) := K_inc

  for each (h, psi) in missing:
    for each (L, priority) in sorted(hints, by priority):
      if matches(L.type, psi):
        sigma[h] := L
        break

  if all holes filled:
    return K^+ = complete_proof(partial, sigma)
  else:
    // Report what's still missing
    still_missing := {(h, psi) : sigma[h] undefined}
    return K^inc' = (phi, still_missing, partial[sigma], trace')
```

**Coq Hint Commands:**
```coq
Hint Resolve lemma_A_implies_C : mydb.
Hint Constructors and or : mydb.

(* Now auto with mydb can complete more proofs *)
```

---

## Lemma Synthesis

### 1. Automatic Lemma Generation

**Problem:** Given $K^{\text{inc}}$ with $\mathsf{missing} = \{(h_1, \psi_1), \ldots, (h_k, \psi_k)\}$, can we automatically synthesize lemmas $L_i : \psi_i$?

**Approaches:**

| Technique | Description | Applicability |
|-----------|-------------|---------------|
| Enumerative search | Try all terms up to size $n$ | Small types |
| SMT solving | Encode as satisfiability | Decidable theories |
| Machine learning | Neural-guided synthesis | Large search spaces |
| Superposition | First-order theorem proving | Universal formulas |
| Induction | Recursive function synthesis | Inductive types |

### 2. SMT-Based Lemma Synthesis

**Algorithm (SMT Lemma Finder):**
```
SynthesizeLemma(context, psi):
  // Encode the goal as SMT constraint
  smt_query := encode(context ⊢ ?L : psi)

  result := SMT_Solve(smt_query)

  if result = SAT(model):
    L := extract_term(model)
    return Some(L)
  elif result = UNSAT:
    return None  // No lemma exists
  else:
    return Unknown  // Timeout
```

**Example:**
```
Context: x : Int, y : Int, H : x > 0
Goal: ?L : x + y > y

SMT Query:
  (declare-const x Int)
  (declare-const y Int)
  (assert (> x 0))
  (assert (not (> (+ x y) y)))
  (check-sat)

Result: UNSAT (goal is valid)
Synthesized Lemma: refl (x + y > y follows from x > 0)
```

### 3. Inductive Lemma Synthesis

**Problem:** Synthesize lemmas involving recursive definitions.

**Algorithm (Inductive Synthesis):**
```
SynthesizeInductiveLemma(context, psi, inductive_type):
  // Generate candidate recursive structure
  for each constructor C of inductive_type:
    // Try to prove psi by structural induction
    base_case := try_prove(psi[C_base])
    inductive_case := try_prove(
      forall x. psi[x] -> psi[C_rec(x)]
    )

    if base_case and inductive_case:
      return induction_proof(base_case, inductive_case)

  return None
```

**Example:**
```
Goal: ?L : length (xs ++ ys) = length xs + length ys

Inductive synthesis over xs:
  Base: length ([] ++ ys) = length [] + length ys
        = length ys = 0 + length ys  ✓

  Step: Assume length (xs ++ ys) = length xs + length ys
        Show length ((x::xs) ++ ys) = length (x::xs) + length ys
        = 1 + length (xs ++ ys) = 1 + length xs + length ys  ✓

Synthesized: list_length_app (automatic induction)
```

---

## Certificate Payload Structure

The complete proof completion certificate:

```
K_IncComplete^+ := {
  original_goal: phi,

  partial_proof: {
    structure: pi_partial,
    holes: [(h1, psi1), ..., (hk, psik)],
    trace: proof_search_trace
  },

  completions: {
    (h1, L1, derivation1),
    ...,
    (hk, Lk, derivationk)
  },

  complete_proof: {
    term: pi_partial[h1 := L1, ..., hk := Lk],
    type_check: OK,
    context: Gamma_extended
  },

  provenance: {
    lemma_sources: [
      (L1, "library", "Lemma.foo"),
      (L2, "synthesized", "SMT"),
      (L3, "user_hint", "hint_db")
    ]
  }
}
```

---

## Quantitative Analysis

### Hole Complexity

**Definition (Hole Complexity).**
For partial proof $\pi$ with holes $H = \{h_1, \ldots, h_k\}$:
$$\text{HoleComplexity}(\pi) = \sum_{i=1}^{k} \text{difficulty}(\psi_i)$$

where $\text{difficulty}(\psi)$ measures proof search complexity for $\psi$.

**Difficulty Metrics:**

| Formula Class | Difficulty | Synthesis Method |
|---------------|------------|------------------|
| Propositional | Low | SAT solving |
| Quantifier-free FOL | Medium | SMT |
| First-order | High | Superposition/resolution |
| Higher-order | Very high | Interactive/ML-guided |
| Undecidable | Infinite | Requires user insight |

### Upgrade Probability

**Definition (Upgrade Probability).**
Given library $\mathcal{L}$ and $K^{\text{inc}}$ with missing $M$:
$$P(\text{upgrade}) = \prod_{(h, \psi) \in M} P(\exists L \in \mathcal{L}. L : \psi)$$

**Empirical Observation:** Proof assistants with large libraries (e.g., Mathlib for Lean) have higher upgrade probabilities for standard mathematical goals.

---

## Connections to Classical Results

### 1. Cut Elimination and Lemma Injection

**Theorem (Cut Elimination, Gentzen 1935).**
If $\Gamma \vdash \varphi$ has a proof with cuts, then it has a cut-free proof.

**Connection to UP-IncComplete:**

| Cut Rule | Lemma Injection |
|----------|-----------------|
| $\frac{\Gamma \vdash A \quad \Gamma, A \vdash B}{\Gamma \vdash B}$ | $\frac{K^{\text{inc}}[\text{needs } A] \quad K_A^+}{K_B^+}$ |
| Cut formula $A$ | Missing premise |
| Cut elimination | Proof normalization (removes lemma indirection) |

**Interpretation:** Lemma injection is the computational dual of cut introduction. Where cut elimination removes intermediate lemmas, lemma injection adds them to complete proofs.

### 2. Craig Interpolation

**Theorem (Craig Interpolation).**
If $\vdash A \to C$, then there exists $B$ (the interpolant) such that:
- $\vdash A \to B$
- $\vdash B \to C$
- $B$ uses only vocabulary common to $A$ and $C$

**Connection to UP-IncComplete:**

| Interpolation | Proof Completion |
|---------------|------------------|
| $A$ (antecedent) | Current context $\Gamma$ |
| $C$ (consequent) | Goal $\varphi$ |
| $B$ (interpolant) | Missing lemma $\psi$ |
| Finding $B$ | Synthesizing missing premise |

**Algorithm Insight:** Craig interpolation can synthesize the "right" missing lemma that bridges the gap between context and goal.

### 3. Herbrand's Theorem

**Theorem (Herbrand 1930).**
A first-order formula $\exists \vec{x}. \varphi(\vec{x})$ is provable iff there exist ground terms $\vec{t}_1, \ldots, \vec{t}_k$ such that:
$$\varphi(\vec{t}_1) \vee \cdots \vee \varphi(\vec{t}_k)$$
is a propositional tautology.

**Connection to UP-IncComplete:**

| Herbrand | Proof Completion |
|----------|------------------|
| Ground instances $\vec{t}_i$ | Witness terms for holes |
| Herbrand disjunction | Disjunction of filled holes |
| Finding witnesses | Synthesizing lemma instantiations |

### 4. Resolution Completeness

**Theorem (Robinson 1965).**
Resolution is refutation-complete for first-order logic.

**Connection to UP-IncComplete:**
- Resolution produces **refutation proofs** when the goal is unsatisfiable
- Failed resolution indicates either satisfiability or incompleteness
- UP-IncComplete handles the incomplete case by requesting missing clauses (lemmas)

---

## Implementation in Proof Assistants

### Coq: Obligation System

```coq
(* Program mode with obligations *)
Program Definition safe_div (n m : nat) (H : m <> 0) : nat :=
  n / m.
Next Obligation.
  (* Proof obligation: m <> 0 -> ... *)
  (* This is K^inc with one hole *)
  (* User fills the hole: *)
  assumption.
Qed.
```

### Lean 4: Synthetic Holes

```lean
-- Using `_` creates a synthetic metavariable
theorem example : ∀ n, n + 0 = n := by
  intro n
  -- Goal: n + 0 = n
  -- Lean's `simp` can complete this
  simp [Nat.add_zero]  -- Injects the lemma Nat.add_zero
```

### Agda: Interactive Hole Filling

```agda
-- Agda shows hole type in goal buffer
example : (n : Nat) -> n + 0 ≡ n
example n = {! !}
-- Goal: n + 0 ≡ n
-- User types C-c C-a (auto) to attempt synthesis
-- Or manually fills with: +-identityʳ n
```

### Isabelle: Sledgehammer

```isabelle
lemma example: "length (xs @ ys) = length xs + length ys"
  (* sledgehammer attempts to find a proof *)
  (* If it finds one, it suggests: *)
  by (simp add: length_append)
  (* This injects the missing lemma length_append *)
```

---

## Summary

The UP-IncComplete theorem, translated to complexity/proof theory, establishes **Proof Completion via Lemma Injection**:

1. **Fundamental Correspondence:**
   - Inconclusive certificate $K^{\text{inc}}$ $\leftrightarrow$ Partial proof with holes
   - Missing set $\mathsf{missing}$ $\leftrightarrow$ Proof obligations $\{(h_i, \psi_i)\}$
   - Certificate upgrade $\leftrightarrow$ All holes filled with valid lemmas

2. **Main Result:** An inconclusive proof becomes complete when:
   $$\forall (h_i, \psi_i) \in \mathsf{missing}. \exists L_i \in \Gamma'. \Gamma \vdash L_i : \psi_i$$

3. **Proof Assistant Implementation:**
   - Metavariables and holes track missing premises
   - Tactic systems incrementally fill holes
   - Hint databases and automation inject lemmas
   - Lemma synthesis attempts automatic completion

4. **Certificate Structure:**
   $$K_P^{\mathrm{inc}} \wedge \bigwedge_{m \in \mathsf{missing}} K_m^+ \Rightarrow K_P^+$$

   translates to:

   $$\pi_{\text{partial}}[\mathsf{holes}] \wedge \bigwedge_i (L_i : \psi_i) \Rightarrow \pi_{\text{complete}} : \varphi$$

5. **Key Insight:** Inconclusive verdicts are not failures but **structured requests for help**. The $\mathsf{missing}$ set precisely identifies what additional lemmas would complete the proof, enabling:
   - Interactive proof development (user fills holes)
   - Automated lemma synthesis (SMT, enumeration, ML)
   - Library-assisted completion (hint databases, sledgehammer)
   - Incremental proof construction (tactics)

This translation reveals that proof completion is a form of **certificate upgrade**: partial proofs, viewed as inconclusive certificates, become complete proofs when their missing premises become available through context extension.

---

## Literature

1. **Gentzen, G. (1935).** "Untersuchungen uber das logische Schliessen." Math. Zeitschrift. *Cut elimination.*

2. **Craig, W. (1957).** "Three Uses of the Herbrand-Gentzen Theorem." JSL. *Interpolation theorem.*

3. **Robinson, J. A. (1965).** "A Machine-Oriented Logic Based on the Resolution Principle." JACM. *Resolution completeness.*

4. **Miller, D. & Nadathur, G. (2012).** *Programming with Higher-Order Logic.* Cambridge. *Higher-order logic programming.*

5. **Paulson, L. C. & Blanchette, J. C. (2015).** "Three Years of Experience with Sledgehammer." ITP. *Automated reasoning in Isabelle.*

6. **de Moura, L. & Ullrich, S. (2021).** "The Lean 4 Theorem Prover and Programming Language." CADE. *Lean elaboration.*

7. **Brady, E. (2013).** "Idris, a General Purpose Dependently Typed Programming Language." JFP. *Elaboration with holes.*

8. **Norell, U. (2009).** "Dependently Typed Programming in Agda." AFP. *Interactive hole-filling.*

9. **Czajka, L. & Kaliszyk, C. (2018).** "Hammer for Coq: Automation for Dependent Type Theory." JAR. *CoqHammer.*

10. **First, E., Rabe, M., Ringer, T., & Brun, Y. (2023).** "Baldur: Whole-Proof Generation and Repair with Large Language Models." FSE. *ML-based proof synthesis.*

11. **Yang, K. & Deng, J. (2019).** "Learning to Prove Theorems via Interacting with Proof Assistants." ICML. *Neural theorem proving.*

12. **Polu, S. & Sutskever, I. (2020).** "Generative Language Modeling for Automated Theorem Proving." arXiv. *GPT-f for theorem proving.*
