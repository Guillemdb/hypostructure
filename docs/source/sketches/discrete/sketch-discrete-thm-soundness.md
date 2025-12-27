---
title: "Soundness - Complexity Theory Translation"
---

# THM-SOUNDNESS: Proof System Soundness

## Overview

This document provides a complete complexity-theoretic translation of the Soundness theorem from the hypostructure framework. The theorem establishes that every transition in a sieve run is certificate-justified, ensuring the logical coherence of the verification process. In complexity theory terms, this corresponds to the fundamental property of proof system soundness: every derivation step is valid and no unsound inferences can occur.

**Original Theorem Reference:** {prf:ref}`thm-soundness`

---

## Complexity Theory Statement

**Theorem (THM-SOUNDNESS, Computational Form).**
Let $\mathcal{P} = (\Sigma, \mathcal{R}, \vdash)$ be a proof system where:
- $\Sigma$ is the set of formulas (statements/judgments)
- $\mathcal{R}$ is the set of inference rules
- $\vdash$ is the derivability relation

A proof system $\mathcal{P}$ is **sound** if for every derivation $\pi: \Gamma \vdash \varphi$:
1. **Certificate Production:** Each inference step in $\pi$ produces a valid proof witness $K$ for its conclusion
2. **Premise-Conclusion Validity:** If $K$ witnesses that premises $\Gamma_i$ hold, then $K$ implies the precondition for deriving the conclusion $\varphi$
3. **Context Accumulation:** The proof witness $K$ is added to the derivation context, enabling subsequent inference steps

**Formal Statement.** If the proof system transitions from judgment $J_1$ to judgment $J_2$ via rule $R$ with proof witness $K_R$, then:
$$K_R \Rightarrow \mathrm{Pre}(J_2)$$

where $\mathrm{Pre}(J_2)$ is the precondition required for $J_2$ to be derivable.

**Corollary (No False Derivations).** A sound proof system cannot derive falsehood from true premises:
$$\Gamma \vdash_\mathcal{P} \varphi \Rightarrow \Gamma \models \varphi$$

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Sieve run | Proof/derivation | Sequence of inference steps |
| Node $N_i$ | Inference rule application | Single derivation step |
| Transition $N_1 \to N_2$ | Rule application | $\frac{\Gamma_1}{\Gamma_2}$ |
| Outcome $o$ | Derivation outcome | Success/failure of rule |
| Certificate $K_o$ | Proof witness | Evidence for conclusion |
| Precondition $\mathrm{Pre}(N)$ | Side conditions | Requirements for rule application |
| Context $\Gamma$ | Proof context | Accumulated assumptions/lemmas |
| Edge validity | Inference validity | Premises imply conclusion |
| Certificate-justified | Valid proof step | Step has valid justification |
| Node evaluation function | Rule application function | Maps context to conclusion |
| Outcome alphabet $\mathcal{O}_N$ | Judgment space | Possible conclusions |
| Certificate type $\mathcal{K}_N$ | Witness type | Kind of proof evidence |

---

## Proof System Framework

### Definition (Proof System)

A **proof system** $\mathcal{P}$ consists of:

1. **Formula Language $\Sigma$:** The set of well-formed formulas/judgments that can be expressed
2. **Inference Rules $\mathcal{R}$:** A finite set of rules of the form:
   $$\frac{\varphi_1 \quad \varphi_2 \quad \cdots \quad \varphi_n}{\psi} \; [R]$$
   where $\varphi_i$ are premises and $\psi$ is the conclusion
3. **Derivability Relation $\vdash$:** The smallest relation closed under the rules in $\mathcal{R}$

### Definition (Proof Witness)

A **proof witness** (certificate) $K$ for a derivation step is a tuple:
$$K = (\text{rule}, \text{premises}, \text{conclusion}, \text{justification})$$

where:
- `rule` identifies the inference rule applied
- `premises` are the antecedent judgments
- `conclusion` is the consequent judgment
- `justification` provides evidence that premises imply conclusion

### Definition (Soundness)

A proof system $\mathcal{P}$ is **sound** with respect to a semantics $\models$ if:
$$\Gamma \vdash_\mathcal{P} \varphi \Rightarrow \Gamma \models \varphi$$

That is, every derivable statement is semantically valid.

---

## Proof Sketch

### Setup: Proof Systems as Transition Systems

**Translation.** We model proof derivations as transitions in a state machine:
- **States:** Proof contexts $\Gamma$ (accumulated judgments)
- **Transitions:** Inference rule applications
- **Certificates:** Proof witnesses justifying each step

**Correspondence to Hypostructure:**

| Sieve Component | Proof System Component |
|-----------------|------------------------|
| Node $N$ | Inference rule $R$ |
| State $x \in X$ | Proof context $\Gamma$ |
| Evaluation $\mathrm{eval}_N(x, \Gamma)$ | Rule application $R(\Gamma)$ |
| Outcome $o$ | Derivation result (success/failure) |
| Certificate $K_o$ | Proof witness for derived judgment |
| Edge $N_1 \xrightarrow{o} N_2$ | Sequential rule composition |

### Step 1: Rule Application Produces Certificates (By Construction)

**Claim.** Each inference rule application produces a valid proof witness.

**Proof.** By the definition of node evaluation (Definition {prf:ref}`def-node-evaluation`), each node $N$ has an evaluation function:
$$\mathrm{eval}_N : X \times \Gamma \to \mathcal{O}_N \times \mathcal{K}_N \times X \times \Gamma$$

This maps $(x, \Gamma) \mapsto (o, K_o, x', \Gamma')$ where:
- $o$ is the outcome (corresponding to the inference result)
- $K_o \in \mathcal{K}_N$ is the certificate witnessing outcome $o$
- $\Gamma' = \Gamma \cup \{K_o\}$ extends the context

**Proof System Interpretation.** For an inference rule $R$:
$$\frac{\varphi_1 \quad \cdots \quad \varphi_n}{\psi}$$

The evaluation function:
1. Checks that $\varphi_1, \ldots, \varphi_n \in \Gamma$ (premises in context)
2. If satisfied, produces outcome "derivable" with certificate $K_\psi$
3. Adds $\psi$ and $K_\psi$ to the context

Hence every rule application produces a certificate by construction. $\square$

### Step 2: Certificate Implies Precondition (Edge Validity)

**Claim.** The certificate produced by one rule implies the precondition for the next.

**Proof.** By the definition of edge validity (Definition {prf:ref}`def-edge-validity`), an edge $N_1 \xrightarrow{o} N_2$ is valid if and only if:
$$K_o \Rightarrow \mathrm{Pre}(N_2)$$

**Proof System Interpretation.** Consider a derivation fragment:
$$\frac{\Gamma_1 \vdash \varphi}{\Gamma_2 \vdash \psi}$$

where $\Gamma_2 = \Gamma_1 \cup \{\varphi\}$. The certificate $K_\varphi$ produced by deriving $\varphi$ establishes that $\varphi$ holds in the context. The precondition for deriving $\psi$ includes having $\varphi$ available:
$$\mathrm{Pre}(\psi) = \{\varphi \in \Gamma\}$$

Since $K_\varphi$ witnesses $\varphi \in \Gamma$, we have $K_\varphi \Rightarrow \mathrm{Pre}(\psi)$. $\square$

### Step 3: Context Accumulation (Monotonic Growth)

**Claim.** Certificates are added to and never removed from the context during a derivation.

**Proof.** By the definition of context (Definition {prf:ref}`def-context`), the context $\Gamma$ grows monotonically:
$$\Gamma' = \Gamma \cup \{K_o\}$$

New certificates are added at each step, and existing certificates are preserved.

**Proof System Interpretation.** This corresponds to the standard property of proof systems that derived lemmas remain available:
- If $\Gamma \vdash \varphi$ is derived at step $i$, then $\varphi \in \Gamma_j$ for all $j \geq i$
- The proof context accumulates all derived judgments
- No "forgetting" or retraction occurs during a proof

This ensures that later inference steps can reference earlier derived results. $\square$

### Step 4: Induction on Derivation Length

**Theorem (Soundness by Induction).**
Every transition in a proof derivation is certificate-justified.

**Proof by Induction on Derivation Length.**

**Base Case (Length 0):** The initial context $\Gamma_0$ contains only axioms. Each axiom $A$ comes with an axiomatic certificate $K_A$ witnessing its validity. No transitions have occurred, so the property holds vacuously.

**Inductive Case (Length $n+1$):** Assume all transitions in a derivation of length $n$ are certificate-justified. Consider extending the derivation by one step using rule $R$:
$$\frac{\Gamma_n \vdash \varphi_1 \quad \cdots \quad \Gamma_n \vdash \varphi_k}{\Gamma_{n+1} \vdash \psi}$$

By the inductive hypothesis, each $\varphi_i$ has a valid certificate $K_{\varphi_i} \in \Gamma_n$.

By Step 1, the rule application produces a certificate $K_\psi$ for $\psi$.

By Step 2, $K_\psi$ satisfies $K_\psi \Rightarrow \mathrm{Pre}(\text{next step})$.

By Step 3, $K_\psi$ is added to $\Gamma_{n+1} = \Gamma_n \cup \{K_\psi\}$.

Hence the $(n+1)$-th transition is certificate-justified. $\square$

---

## Certificate Construction

The soundness proof is constructive. For each derivation step, we can explicitly construct the certificate:

**Inference Certificate $K_R$:**
$$K_R = \left(\text{rule\_id}, \{\text{premise\_witnesses}\}, \text{conclusion}, \text{validity\_proof}\right)$$

where:
- `rule_id` identifies which inference rule was applied
- `premise_witnesses` = $\{K_{\varphi_1}, \ldots, K_{\varphi_k}\}$ are certificates for premises
- `conclusion` = $\psi$ is the derived judgment
- `validity_proof` demonstrates that the rule application is correct

**Derivation Certificate (Full Proof):**
$$\mathcal{C}_\pi = \left(\Gamma_0, \langle R_1, \ldots, R_n \rangle, \langle K_1, \ldots, K_n \rangle, \varphi_{\text{final}}\right)$$

where:
- $\Gamma_0$ is the initial context (axioms/assumptions)
- $\langle R_1, \ldots, R_n \rangle$ is the sequence of rules applied
- $\langle K_1, \ldots, K_n \rangle$ is the sequence of certificates produced
- $\varphi_{\text{final}}$ is the final derived judgment

**Verification Algorithm.** Given $\mathcal{C}_\pi$, verify soundness by:
1. Check each $K_i$ is produced by valid rule application $R_i$
2. Check $K_i \Rightarrow \mathrm{Pre}(R_{i+1})$ for each $i < n$
3. Check $K_n$ witnesses $\varphi_{\text{final}}$

---

## Connections to Classical Results

### 1. Proof System Soundness (Foundational Logic)

**Classical Statement.** A proof system is sound if it cannot derive false statements from true premises.

**Connection to THM-SOUNDNESS.** The hypostructure soundness theorem is a generalization:
- Classical soundness: $\Gamma \vdash \varphi \Rightarrow \Gamma \models \varphi$
- Certificate soundness: Each step has explicit justification

The certificate mechanism makes the soundness proof constructive and verifiable.

| Classical Soundness | THM-SOUNDNESS |
|--------------------|---------------|
| Derivability implies truth | Each transition is justified |
| Induction on proof length | Induction on sieve trace |
| Rule validity | Edge validity |
| Semantic entailment | Certificate implication |

### 2. Type Safety (Progress + Preservation)

**Classical Statement (Wright-Felleisen 1994).** A type system is safe if:
- **Progress:** Well-typed terms can take a step or are values
- **Preservation:** Typing is preserved under reduction

**Connection to THM-SOUNDNESS.** The soundness theorem ensures:
- **Progress Analog:** Each node produces an outcome (computation proceeds)
- **Preservation Analog:** Certificates propagate through transitions (invariants maintained)

| Type Safety | THM-SOUNDNESS |
|-------------|---------------|
| Progress: term steps or is value | Node evaluation produces outcome |
| Preservation: typing preserved | Certificate implies next precondition |
| Well-typed programs don't get stuck | Sieve runs don't produce unjustified transitions |
| Subject reduction | Context extension |

**Explicit Correspondence:**
- Type judgment $\Gamma \vdash e : \tau$ $\leftrightarrow$ Certificate $K$ in context $\Gamma$
- Reduction $e \to e'$ $\leftrightarrow$ Transition $N_1 \to N_2$
- Preservation $\Gamma \vdash e' : \tau$ $\leftrightarrow$ $K$ implies $\mathrm{Pre}(N_2)$

### 3. Certified Compilation (Leroy 2009)

**Classical Statement.** A compiler is certified if it preserves program semantics:
$$\llbracket \text{source} \rrbracket = \llbracket \text{target} \rrbracket$$

**Connection to THM-SOUNDNESS.** Certified compilation is a special case of soundness:
- Source program = Initial sieve state
- Compilation passes = Sieve transitions
- Correctness certificates = Transition certificates

| Certified Compilation | THM-SOUNDNESS |
|----------------------|---------------|
| Compiler correctness | Sieve soundness |
| Semantic preservation | Certificate propagation |
| Proof-carrying code | Certificate context $\Gamma$ |
| Verified optimization | Certificate-justified surgery |

**CompCert Structure.** The CompCert verified compiler (Leroy) produces:
1. Compiled code
2. Proof that compilation preserves semantics

This matches the sieve structure:
1. Final state (computation result)
2. Certificate context $\Gamma_{\text{final}}$ (proof of correctness)

### 4. Proof-Carrying Code (Necula 1997)

**Classical Statement.** Proof-carrying code attaches machine-checkable proofs to programs, allowing untrusted code to be verified before execution.

**Connection to THM-SOUNDNESS.** The certificate context $\Gamma$ is exactly a proof-carrying code certificate:
- Code = Sieve trace (sequence of transitions)
- Proof = Certificate context (sequence of witnesses)
- Verification = Checking $K_i \Rightarrow \mathrm{Pre}(\text{next})$

| Proof-Carrying Code | THM-SOUNDNESS |
|--------------------|---------------|
| Untrusted code | External input/program |
| Safety policy | Sieve predicates |
| Proof certificate | Context $\Gamma$ |
| Proof checker | Certificate verification |
| Type safety | Transition validity |

### 5. Hoare Logic (Program Verification)

**Classical Statement.** Hoare triples $\{P\} \, C \, \{Q\}$ specify that if precondition $P$ holds before executing $C$, then postcondition $Q$ holds after.

**Connection to THM-SOUNDNESS.** Each sieve transition corresponds to a Hoare triple:
$$\{\mathrm{Pre}(N_1)\} \; N_1 \; \{\mathrm{Pre}(N_2)\}$$

The certificate $K_o$ witnesses that the postcondition implies the next precondition.

| Hoare Logic | THM-SOUNDNESS |
|-------------|---------------|
| Precondition $P$ | $\mathrm{Pre}(N_1)$ |
| Command $C$ | Node evaluation |
| Postcondition $Q$ | Certificate $K_o$ |
| Sequential composition | Edge validity |
| Weakest precondition | Certificate implication |

---

## Quantitative Refinements

### Proof Complexity Bounds

**Certificate Size.** The size of a soundness certificate is bounded by:
$$|K| \leq O(|\pi|)$$

where $|\pi|$ is the length of the derivation. Each step adds one certificate.

**Verification Time.** Verifying a soundness certificate takes:
$$T_{\text{verify}} = O(|\pi| \cdot T_{\text{check}})$$

where $T_{\text{check}}$ is the time to verify a single certificate implication.

### Relationship to Proof Complexity Classes

| Proof System | Certificate Verification | Corresponding Complexity |
|--------------|-------------------------|--------------------------|
| Propositional resolution | Polynomial | NP (via SAT) |
| Frege systems | Polynomial | coNP (via TAUT) |
| Extended Frege | Polynomial | P/poly |
| Bounded arithmetic | Polynomial | PH |

The soundness theorem ensures that certificate verification is efficient relative to the proof system.

---

## Algorithmic Implications

### Certificate Verification Algorithm

**Input:** Sieve trace $\pi = (N_1, o_1, K_1) \to (N_2, o_2, K_2) \to \cdots \to (N_n, o_n, K_n)$

**Output:** Boolean indicating whether all transitions are certificate-justified

```
function VerifySoundness(trace):
    context := InitialContext()

    for each (N_i, o_i, K_i) in trace:
        // Step 1: Verify certificate was produced by node
        if not ValidCertificate(N_i, K_i):
            return UNSOUND

        // Step 2: Verify certificate implies next precondition
        if i < n:
            if not Implies(K_i, Precondition(N_{i+1})):
                return UNSOUND

        // Step 3: Add certificate to context
        context := context + {K_i}

    return SOUND
```

**Complexity:** $O(n \cdot T_{\text{implication}})$ where $n$ is trace length and $T_{\text{implication}}$ is the cost of checking certificate implication.

### Certificate Extraction

From a successful sieve run, extract the full soundness certificate:

```
function ExtractCertificate(sieve_run):
    certificates := []

    for each transition in sieve_run:
        (N_1, N_2, outcome, K) := transition

        cert := {
            source: N_1,
            target: N_2,
            outcome: outcome,
            witness: K,
            implication_proof: ProveImplication(K, Precondition(N_2))
        }

        certificates.append(cert)

    return SoundnessCertificate(certificates)
```

---

## Summary

The THM-SOUNDNESS theorem, translated to complexity theory, establishes that:

1. **Every inference step is justified:** Each transition in a proof derivation produces a valid certificate witnessing the conclusion.

2. **Premises imply conclusions:** Certificates for premises logically imply the preconditions for deriving conclusions, ensuring no invalid inferences.

3. **No unsound derivations:** The combination of certificate production and implication checking guarantees that false statements cannot be derived from true premises.

4. **Constructive verification:** The proof is constructive, yielding an efficient algorithm for verifying that a derivation is sound.

This translation reveals that the hypostructure soundness theorem is a generalization of fundamental results in proof theory, type theory, and program verification. The certificate mechanism provides a uniform framework for:
- Proof system soundness (logic)
- Type safety (programming languages)
- Certified compilation (compilers)
- Proof-carrying code (security)

The key insight is that **certificates are first-class objects** that accumulate during computation, providing a complete audit trail of the verification process.
