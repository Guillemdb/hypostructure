---
title: "UP-Surgery - Complexity Theory Translation"
---

# UP-Surgery: Proof System Simulation and Extended Frege

## Overview

This document provides a complete complexity-theoretic translation of the UP-Surgery metatheorem (Surgery Promotion) from the hypostructure framework. The theorem establishes that when a valid surgery operator is applied to resolve a singularity, the flow continues on the modified Hypostructure as a generalized (surgery/weak) solution. In proof complexity terms, this corresponds to **Proof System Simulation**: a weaker proof system (with "gaps" or "singularities") can be extended via simulation to a stronger system that achieves the desired result.

**Original Theorem Reference:** {prf:ref}`mt-up-surgery`

**Central Translation:** Successful surgery promotes to relaxed YES outcome $\longleftrightarrow$ **Proof Simulation**: Incomplete proof + cut-elimination $\to$ extended Frege proof.

---

## Complexity Theory Statement

**Theorem (Proof System Simulation with Extension, Computational Form).**
Let $\mathcal{P}$ be a proof system and $\pi$ be an incomplete or blocked proof. There exists a **proof simulation** such that:

**Input**: Incomplete proof $\pi$ with gap + admissibility certificate (gap is "surgically removable")

**Output**:
- Extended proof $\pi'$ in a stronger system $\mathcal{P}'$ establishing the same theorem
- Simulation overhead: $|\pi'| \leq p(|\pi|)$ for polynomial $p$
- Certificate that $\pi'$ is valid in $\mathcal{P}'$

**Guarantees**:
1. **Proof continuation**: The extended proof $\pi'$ proves the same theorem as intended by $\pi$
2. **Simulation bound**: Proof size bounded by polynomial in original size (p-simulation)
3. **Certificate production**: Validity witness for the extended proof
4. **Progress**: Each simulation step produces a strictly more complete proof

**Formal Statement.** Let $\mathcal{P}$ be a proof system with proof $\pi$ that fails at some step (barrier breach). If the failure is "admissible" (surgically removable), then:

1. **Simulation Exists:** There exists an extended proof $\pi': \Gamma \vdash \Delta$ in system $\mathcal{P}'$

2. **p-Simulation Bound:** The extended proof satisfies:
   $$|\pi'| \leq p(|\pi|)$$
   for some polynomial $p$ (Cook-Reckhow p-simulation)

3. **Generalized Solution:** The combined proof (original steps + extended steps) constitutes a valid proof in the generalized sense (surgery/weak)

4. **Re-entry to Proof Hierarchy:** The extended system $\mathcal{P}'$ is related to $\mathcal{P}$ by a known simulation relationship in the proof complexity hierarchy

---

## Terminology Translation Table

| Hypostructure Concept | Proof Complexity Analog | Formal Correspondence |
|-----------------------|-------------------------|------------------------|
| State space $\mathcal{X}$ | Proof space $\mathcal{P}$ | Space of all proofs in system $\mathcal{P}$ |
| Semiflow $S_t$ | Proof derivation | Sequential application of inference rules |
| Cohomological height $\Phi$ | Proof length/complexity | Size of proof tree, number of steps |
| Singularity $(t^*, x^*)$ | Proof gap/failure | Step where derivation cannot continue |
| Modal diagnosis $M$ | Failure type | Missing lemma, invalid rule application, etc. |
| Surgery operator $\mathcal{O}_S$ | System extension/simulation | Moving to stronger proof system |
| Modified Hypostructure $\mathcal{H}'$ | Extended proof system $\mathcal{P}'$ | Frege $\to$ Extended Frege, etc. |
| Canonical library $\mathcal{L}_T$ | Standard proof patterns | Resolution, Frege rules, extension axioms |
| Capacity bound $\varepsilon_{\text{adm}}$ | Simulation overhead | Polynomial blowup in proof size |
| Height decrease $\delta_S$ | Complexity progress | Steps remaining to complete proof |
| Re-entry certificate $K^{\mathrm{re}}$ | Simulation certificate | Proof that $\mathcal{P}' \geq_p \mathcal{P}$ |
| Generalized solution | Extended/weak proof | Proof valid in extended system |
| Canonical neighborhood | Local proof structure | Standard inference patterns |
| Gluing interface $\partial$ | Formula/sequent interface | Shared formulas between proof steps |
| Diffeomorphic outcomes | Equivalent proofs | Proofs with same logical content |
| Bordism category $\mathbf{Bord}_n$ | Proof system hierarchy | Cook-Reckhow ordering $\leq_p$ |

---

## Proof System Simulation as Surgery

### The Cook-Reckhow Framework

**Definition (Proof System).** A **proof system** for a language $L$ is a polynomial-time computable function $f: \{0,1\}^* \to \{0,1\}^* \cup \{\bot\}$ such that:
$$x \in L \iff \exists \pi. f(\pi) = x$$

The string $\pi$ is a **proof** of $x$.

**Definition (p-Simulation).** Proof system $\mathcal{P}'$ **p-simulates** proof system $\mathcal{P}$ (written $\mathcal{P}' \geq_p \mathcal{P}$) if there exists a polynomial $p$ such that:
$$\forall \tau \in L. \forall \pi \in \mathcal{P}. f_{\mathcal{P}}(\pi) = \tau \Rightarrow \exists \pi' \in \mathcal{P}'. f_{\mathcal{P}'}(\pi') = \tau \wedge |\pi'| \leq p(|\pi|)$$

**Surgery Interpretation:** When a proof $\pi$ in system $\mathcal{P}$ encounters a "singularity" (gap or failure), surgery corresponds to:
1. Recognizing the failure type (modal diagnosis)
2. Moving to a stronger system $\mathcal{P}'$ that p-simulates $\mathcal{P}$
3. Completing the proof in $\mathcal{P}'$ with polynomial overhead
4. The combined proof (surgery solution) is valid in $\mathcal{P}'$

### The Proof Complexity Hierarchy

**Definition (Standard Hierarchy).** The proof complexity hierarchy includes:

$$\text{Resolution} \leq_p \text{Frege} \leq_p \text{Extended Frege} \leq_p \text{Frege + $\forall$-cuts}$$

**Correspondence to Surgery Levels:**

| Surgery Type | Proof System | Singularity Resolved |
|--------------|--------------|---------------------|
| No surgery needed | Original system $\mathcal{P}$ | No gap in proof |
| Local surgery | Frege (propositional) | Propositional gap |
| Global surgery | Extended Frege | Lemma introduction needed |
| Transfinite surgery | Frege + quantifiers | Induction/quantifier gap |

**Extended Frege as Surgery Target:**

Extended Frege (EF) extends Frege by allowing the introduction of new propositional variables (extension axioms):
$$p \leftrightarrow \phi$$

This corresponds to surgery: the "singularity" (complex subformula $\phi$) is "excised" and replaced by a new atom $p$, with the equivalence serving as the "cap."

---

## Proof Sketch: Surgery Promotion = Proof Extension

### Setup: The Blocked Proof

**Given Data:**
- A proof attempt $\pi$ in system $\mathcal{P}$ for theorem $\tau$
- A "gap" at step $k$: the derivation cannot continue using rules of $\mathcal{P}$
- Modal diagnosis $M$: the type of failure (e.g., "missing lemma," "cut needed")

**Surgery Conditions:**
1. **Admissibility**: The gap belongs to a "canonical" type that can be resolved
2. **Capacity bound**: The gap can be resolved with polynomial overhead
3. **Progress**: Resolving the gap brings us closer to completing the proof

---

### Step 1: Canonical Neighborhood Theorem (Local Proof Structure)

**Claim.** Near any proof gap in $\pi$, the local structure is $\varepsilon$-close to one of finitely many canonical patterns.

**Proof (Complexity Version).**

**Step 1.1 (Gap Classification).** Any proof gap falls into one of the following categories:

| Gap Type | Local Structure | Canonical Pattern |
|----------|-----------------|-------------------|
| Missing lemma | $\Gamma \vdash \Delta$ where $\Delta$ requires auxiliary result | Lemma introduction |
| Cut needed | Two branches share formula not in conclusion | Cut rule |
| Extension needed | Complex subformula repeated | Extension axiom |
| Induction gap | Property requires case analysis | Induction schema |

**Step 1.2 (Finite Classification).** The classification is **finite** because:
- Propositional connectives are finite: $\{\land, \lor, \to, \neg\}$
- Each connective has bounded introduction/elimination patterns
- The depth of the gap formula is bounded by proof size

This corresponds to Perelman's canonical neighborhood theorem: near high-curvature points, geometry is standard.

**Step 1.3 (Standard Resolution).** Each canonical gap type has a standard resolution:
- **Round sphere** (extinction) $\longleftrightarrow$ **Trivial lemma** (immediate derivation)
- **Round cylinder** (neck pinch) $\longleftrightarrow$ **Cut formula** (shared intermediate)
- **Bryant soliton** (asymptotic model) $\longleftrightarrow$ **Extension axiom** (abbreviation)

**Certificate:** $K_{\text{can}}^+ = (\text{gap type}, \text{canonical pattern}, \text{resolution method})$ $\square$

---

### Step 2: Non-Collapsing Estimates (Simulation Bounds)

**Claim.** The proof extension satisfies polynomial bounds, preventing "collapse" to trivial or exponentially long proofs.

**Proof (Complexity Version).**

**Step 2.1 (Perelman's Monotonicity $\to$ Proof Length Monotonicity).** Define the **proof complexity measure**:
$$\Phi(\pi) := |\pi| + \sum_{i} \text{depth}(\phi_i)$$
where the sum is over all formulas in the proof.

**Step 2.2 (Surgery Does Not Increase Complexity Dramatically).** When moving from $\mathcal{P}$ to $\mathcal{P}'$:

- **Resolution $\to$ Frege**: At most polynomial blowup (well-known)
- **Frege $\to$ Extended Frege**: No blowup (EF can simulate Frege with same size)
- **Extended Frege + Extension Axioms**: Polynomial blowup per extension

**Step 2.3 (Non-Collapsing in Terms of Formulas).** The extension satisfies:
$$|\pi'| \leq p(|\pi|, |\phi_{\text{gap}}|)$$

This prevents:
- **Collapse to triviality**: The proof must still derive the theorem
- **Exponential blowup**: The simulation is polynomial

This corresponds to $\kappa$-non-collapsing: volume (proof size) cannot decrease arbitrarily relative to curvature (formula complexity). $\square$

---

### Step 3: Finite Surgery Count (Bounded Extensions)

**Claim.** The number of extensions needed is bounded by a function of the initial proof structure.

**Proof (Complexity Version).**

**Step 3.1 (Energy Budget).** Define the **extension budget**:
$$E_0 := \text{number of distinct complex subformulas in } \pi$$

**Step 3.2 (Extension Decreases Budget).** Each extension axiom $p \leftrightarrow \phi$:
- Introduces one new variable $p$
- Eliminates one complex subformula $\phi$ from consideration
- Strictly decreases the complexity of remaining gaps

**Step 3.3 (Finiteness).** The number of extensions is bounded by:
$$N_{\text{ext}} \leq |\text{subformulas}(\tau)| \leq |\tau|$$

where $\tau$ is the theorem being proved.

**Step 3.4 (No Accumulation).** Extensions cannot "accumulate" at a single point because:
- Each extension introduces a fresh variable
- Variables must eventually be eliminated
- The theorem $\tau$ has fixed size

This corresponds to Perelman's finite surgery theorem: surgeries cannot accumulate in finite time. $\square$

---

### Step 4: Generalized Solution (Extended Proof)

**Claim.** The combined structure (original proof steps + extensions) constitutes a valid proof in the extended system.

**Proof (Complexity Version).**

**Step 4.1 (Definition of Surgery Solution).** A **proof with surgery** is a sequence:
$$\pi = (\pi_0, E_1, \pi_1, E_2, \pi_2, \ldots, E_k, \pi_k)$$

where:
- Each $\pi_i$ is a partial proof in the base system
- Each $E_i$ is an extension (surgery)
- The final $\pi_k$ derives the target theorem

**Step 4.2 (Weak/Generalized Validity).** The surgery proof is valid if:
1. Each $\pi_i$ follows the rules of the system
2. Each $E_i$ is a valid extension (admissible surgery)
3. The composition derives the theorem

**Step 4.3 (Energy Inequality).** The extended proof satisfies:
$$\Phi(\pi') \leq \Phi(\pi_0) + \sum_i \Delta\Phi_{E_i}$$

where $\Delta\Phi_{E_i}$ is the complexity cost of extension $E_i$. This is the analog of:
$$\mathcal{W}(g(t_2)) \leq \mathcal{W}(g(t_1)) + \sum_{k: t_1 < t_k \leq t_2} \Delta \mathcal{W}_k$$

**Step 4.4 (Functoriality).** The surgery operation is **functorial** in the proof system hierarchy:
- If $\mathcal{P}_1 \leq_p \mathcal{P}_2$ and surgery takes $\mathcal{P}_1 \to \mathcal{P}_1'$, then $\mathcal{P}_1' \leq_p \mathcal{P}_2'$
- Composition of surgeries corresponds to composition of simulations

This corresponds to functoriality in $\mathbf{Bord}_n$: surgery is composable. $\square$

---

### Step 5: Re-Entry Certificate (Simulation Validity)

**Claim.** The surgery produces a certificate enabling continuation of the proof verification.

**Proof (Complexity Version).**

**Step 5.1 (Certificate Components).** Construct:
$$K^{\mathrm{re}} = (K^{\mathrm{re}}_{\text{ext}}, K^{\mathrm{re}}_{\text{sim}}, K^{\mathrm{re}}_{\text{valid}}, K^{\mathrm{re}}_{\text{route}})$$

1. **Extension certificate** $K^{\mathrm{re}}_{\text{ext}}$:
   - Extension axiom: $p \leftrightarrow \phi$
   - Syntactic validity: $p$ fresh, $\phi$ well-formed
   - Scope: where $p$ can be used

2. **Simulation certificate** $K^{\mathrm{re}}_{\text{sim}}$:
   - Source system: $\mathcal{P}$
   - Target system: $\mathcal{P}'$ (Extended Frege)
   - Simulation bound: $|\pi'| \leq p(|\pi|)$

3. **Validity certificate** $K^{\mathrm{re}}_{\text{valid}}$:
   - Extended proof $\pi'$ is valid in $\mathcal{P}'$
   - Same theorem derived: $f_{\mathcal{P}'}(\pi') = \tau$

4. **Routing certificate** $K^{\mathrm{re}}_{\text{route}}$:
   - Current position in proof
   - Remaining gaps to resolve
   - Path to completion

**Step 5.2 (Precondition Satisfaction).**
- **Pre(Extension)**: Gap is of canonical type $\checkmark$
- **Pre(Simulation)**: $\mathcal{P}' \geq_p \mathcal{P}$ $\checkmark$
- **Pre(Validity)**: All rules correctly applied $\checkmark$
- **Pre(Termination)**: Remaining gaps bounded $\checkmark$

**Step 5.3 (Implication Logic).**
$$K^{\mathrm{re}} \Rightarrow \text{Pre}(\text{proof completion})$$

The certificate enables verification to continue in the extended system. $\square$

---

## Connections to Proof Complexity

### 1. Cook-Reckhow Theorem (1979)

**Classical Result.** NP = coNP if and only if there exists a polynomially bounded proof system for TAUT.

**Connection to UP-Surgery:**
- **Singularity** = Proof that cannot be completed in current system
- **Surgery** = Moving to a stronger proof system
- **Generalized solution** = Proof in the extended system
- **p-simulation** = Polynomial overhead bound

**Cook-Reckhow Hierarchy:**
$$\text{Resolution} <_p \text{Cutting Planes} <_p \text{Frege} \leq_p \text{Extended Frege}$$

Surgery corresponds to "climbing" this hierarchy when the current system is insufficient.

### 2. Extended Frege and Extension Axioms

**Definition.** Extended Frege (EF) allows introduction of extension axioms:
$$p \leftrightarrow \phi$$

where $p$ is a fresh variable and $\phi$ is any formula.

**Surgery Correspondence:**

| Geometric Surgery | Proof Extension |
|-------------------|-----------------|
| Excise singular region | Identify complex subformula $\phi$ |
| Glue in standard cap | Introduce abbreviation $p$ |
| Match at boundary | Extension axiom $p \leftrightarrow \phi$ |
| Continue flow | Continue proof with $p$ |

**Extension as "Lemma Introduction":** The extension axiom is like introducing a lemma mid-proof:
$$\text{"Let } p \text{ denote } \phi \text{. Then..."} $$

This allows exponentially shorter proofs of some tautologies (Haken 1985).

### 3. Proof Complexity and Circuit Lower Bounds

**Connection.** Proof complexity is related to circuit complexity:

| Proof System | Circuit Class | Simulation |
|--------------|---------------|------------|
| Resolution | Constant-depth circuits | Resolution simulates AC$^0$-Frege |
| Frege | NC$^1$ circuits | Frege captures NC$^1$ |
| Extended Frege | P/poly | EF may capture P/poly |

**Surgery and Circuit Depth:**
- **Singularity** = Depth limitation in circuit
- **Surgery** = Depth reduction transformation
- **Extended system** = Higher depth allowed

### 4. Interpolation and Feasible Interpolation

**Craig Interpolation.** If $A \vdash B$, there exists $C$ (the interpolant) using only shared vocabulary, with $A \vdash C$ and $C \vdash B$.

**Feasible Interpolation.** The interpolant $C$ can be computed in polynomial time from a proof of $A \to B$.

**Surgery Connection:**
- **Cut formula** = Interpolant between proof parts
- **Surgery excision** = Identifying the interface
- **Capping** = Constructing the interpolant
- **Feasibility** = Polynomial simulation bound

### 5. Automatability and Proof Search

**Definition.** A proof system is **automatizable** if there is an algorithm that, given $\tau \in \text{TAUT}$, finds a proof in time polynomial in the shortest proof.

**Connection to Surgery:**
- **Admissibility check** = Deciding if surgery applies
- **Canonical library** = Standard proof patterns for automation
- **Surgery algorithm** = Proof search with extensions

**Result (Bonet et al. 2000).** Extended Frege is not automatizable unless the polynomial hierarchy collapses.

Surgery interpretation: Finding the "right" extensions is computationally hard.

---

## Certificate Construction

**Surgery Simulation Certificate:**

```
K_SurgerySim = {
    mode: "Proof_Extension",
    mechanism: "System_Simulation",

    singularity: {
        gap_location: step k in proof pi,
        gap_type: M in {missing_lemma, cut_needed, extension_needed},
        formula: phi (formula causing gap),
        admissible: true (canonical type)
    },

    excision: {
        blocked_derivation: pi[1..k],
        interface: Gamma |- Delta (sequent at gap),
        capacity: |phi| (formula complexity)
    },

    capping: {
        pattern: gap_type,
        extension_axiom: p <-> phi,
        from_library: L_T[gap_type]
    },

    pushout: {
        extended_proof: pi',
        system: EF (Extended Frege),
        simulation: P <=_p P'
    },

    complexity_control: {
        pre_size: |pi|,
        post_size: |pi'| <= p(|pi|),
        overhead: polynomial
    },

    progress: {
        measure: (remaining_gaps, complexity),
        well_founded: lexicographic,
        bound: |tau| extensions max
    },

    certificate: {
        valid_in_EF: true,
        same_theorem: tau,
        simulation_witness: K_sim
    }
}
```

---

## Quantitative Summary

| Property | Bound |
|----------|-------|
| Extension count | $O(\|\tau\|)$ |
| Simulation overhead | Polynomial $p(\|\pi\|)$ |
| Certificate verification | Polynomial in $\|\pi'\|$ |
| Gap classification | Finite (canonical types) |

### Proof System Comparison

| System | Surgery Level | Simulation Overhead |
|--------|---------------|---------------------|
| Resolution | None | - |
| Frege | Local | None (same system) |
| Extended Frege | Global | Polynomial |
| Frege + Induction | Transfinite | Ordinal-bounded |

### Surgery-Simulation Correspondence

| Surgery Property | Simulation Property |
|------------------|---------------------|
| Canonical neighborhood | Standard inference pattern |
| Non-collapsing | Polynomial bound |
| Finite surgeries | Finite extensions |
| Generalized solution | Extended proof |
| Re-entry certificate | Simulation witness |

---

## Extended Connections

### 1. Natural Proofs and Barriers

**Razborov-Rudich Natural Proofs (1997).** Natural proof methods cannot prove superpolynomial circuit lower bounds if one-way functions exist.

**Surgery Barrier Analogy:**
- **Natural property** = Admissible singularity pattern
- **Barrier** = Obstruction to surgery
- **Horizon (inadmissible)** = Proof method cannot handle certain structures

**Connection:** Just as natural proofs face barriers, proof surgery faces limits when gaps are not "canonical."

### 2. Bounded Arithmetic and Propositional Translations

**Bounded Arithmetic.** Theories $S^i_2$ (Buss) capture polynomial-time reasoning.

**Propositional Translation.** Proofs in $S^i_2$ translate to propositional proofs.

| Arithmetic Theory | Proof System | Surgery Level |
|-------------------|--------------|---------------|
| $S^1_2$ | Extended Frege | Global surgery |
| $S^2_2$ | Frege + Extension | Transfinite surgery |
| $T^i_2$ | Higher systems | Extended surgery |

### 3. Proof Complexity and Cryptography

**Connection.** Proof complexity lower bounds imply cryptographic hardness:
- Short proofs $\to$ Easy breaking
- Long proofs $\to$ Secure cryptography

**Surgery Interpretation:**
- **Admissible surgery** = Efficient attack
- **Inadmissible (horizon)** = Cryptographic hardness
- **p-simulation** = Attack simulation

### 4. SAT Solving and CDCL

**Modern SAT Solvers.** Use Conflict-Driven Clause Learning (CDCL), which corresponds to proof search in Resolution.

**Surgery in CDCL:**

| CDCL Component | Surgery Analog |
|----------------|----------------|
| Conflict | Singularity detection |
| Learned clause | Surgery product |
| Backtracking | Excision |
| Unit propagation | Cap derivation |
| Restart | Global surgery |

### 5. Intuitionistic Logic and Realizability

**Intuitionistic Proofs.** Every proof has computational content (BHK interpretation).

**Surgery Connection:**
- **Classical proof** = May have "singularities" (non-constructive steps)
- **Surgery** = Replace non-constructive with constructive
- **Extended system** = Classical logic with realizability

---

## Conclusion

The UP-Surgery theorem translates to proof complexity as **Proof System Simulation with Extension**:

1. **Surgery = System Extension:** Moving from a weaker proof system to a stronger one that can handle the "gap."

2. **Energy = Proof Complexity:** The measure (proof length, formula depth) that must be controlled during extension.

3. **Canonical Neighborhood = Standard Patterns:** Finite classification of gap types enabling systematic resolution.

4. **Generalized Solution = Extended Proof:** The combined proof valid in the stronger system.

5. **p-Simulation = Surgery Bound:** Polynomial overhead ensures surgery is "efficient."

**Physical Interpretation (Computational Analogue):**

- **Singularity** = Proof gap where current system is insufficient
- **Surgery** = Extending to stronger proof system
- **Energy decrease** = Progress toward proof completion
- **Re-entry** = Continuation with extended resources

**The Surgery Simulation Certificate:**

$$K_{\text{SurgerySim}}^+ = \begin{cases}
\pi' & \text{extended proof in } \mathcal{P}' \\
\mathcal{P}' \geq_p \mathcal{P} & \text{simulation relationship} \\
|\pi'| \leq p(|\pi|) & \text{polynomial bound} \\
f_{\mathcal{P}'}(\pi') = \tau & \text{same theorem}
\end{cases}$$

**Key Insight:** The hypostructure UP-Surgery theorem captures the fundamental proof complexity phenomenon: when a proof system is insufficient, we can "surgically" extend it to a stronger system while maintaining polynomial bounds. This is the proof-theoretic analog of Perelman's surgery for Ricci flow: controlled intervention that allows continuation through singularities.

The correspondence reveals that:
- **Hamilton/Perelman surgery** = **Cook-Reckhow simulation**
- **Canonical neighborhoods** = **Standard proof patterns**
- **Non-collapsing** = **Polynomial bounds**
- **Finite surgery time** = **Bounded extensions**

---

## Literature

1. **Cook, S. A. & Reckhow, R. A. (1979).** "The Relative Efficiency of Propositional Proof Systems." *Journal of Symbolic Logic.* *Foundation of proof complexity.*

2. **Krajicek, J. (1995).** *Bounded Arithmetic, Propositional Logic, and Complexity Theory.* Cambridge University Press. *Connections between bounded arithmetic and proof systems.*

3. **Buss, S. R. (1998).** "An Introduction to Proof Theory." In *Handbook of Proof Theory.* Elsevier. *Comprehensive treatment of proof theory.*

4. **Bonet, M. L., Pitassi, T., & Raz, R. (2000).** "On Interpolation and Automatization for Frege Systems." *SIAM Journal on Computing.* *Hardness of automatization.*

5. **Haken, A. (1985).** "The Intractability of Resolution." *Theoretical Computer Science.* *Exponential lower bounds for resolution.*

6. **Razborov, A. A. & Rudich, S. (1997).** "Natural Proofs." *Journal of Computer and System Sciences.* *Barriers to circuit lower bounds.*

7. **Perelman, G. (2003).** "Ricci Flow with Surgery on Three-Manifolds." *arXiv.* *Original surgery construction (geometric).*

8. **Hamilton, R. S. (1997).** "Four-Manifolds with Positive Isotropic Curvature." *Communications in Analysis and Geometry.* *Surgery program for Ricci flow.*

9. **Kleiner, B. & Lott, J. (2008).** "Notes on Perelman's Papers." *Geometry & Topology.* *Detailed verification of Perelman's work.*

10. **Beame, P. & Pitassi, T. (1996).** "Propositional Proof Complexity: Past, Present, and Future." *Bulletin of the EATCS.* *Survey of proof complexity.*

11. **Pudlak, P. (1998).** "The Lengths of Proofs." In *Handbook of Proof Theory.* Elsevier. *Proof length and complexity.*

12. **Krajicek, J. (2019).** *Proof Complexity.* Cambridge University Press. *Modern treatment of the field.*
