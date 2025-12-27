---
title: "FACT-SoftWP - Complexity Theory Translation"
---

# FACT-SoftWP: Local-to-Global Consistency Compilation

## Overview

This document provides a complete complexity-theoretic translation of the FACT-SoftWP theorem (Soft→WP Compilation) from the hypostructure framework. The translation establishes a formal correspondence between template-based well-posedness derivation and local-to-global consistency reductions in computational complexity.

**Original Theorem Reference:** {prf:ref}`mt-fact-soft-wp`

**Core Insight:** Local consistency constraints (soft interface certificates) compose to yield global consistency (well-posedness), analogous to how locally verifiable proofs can be compiled into globally valid certificates through compositional verification.

---

## Complexity Theory Statement

**Theorem (Local-to-Global Consistency Compilation).**
Let $\mathcal{V} = (\mathcal{C}, \mathcal{T}, \text{Match}, \text{Instantiate})$ be a compositional verification system where:
- $\mathcal{C} = \{C_1, \ldots, C_k\}$ is a set of local consistency certificates
- $\mathcal{T} = \{T_1, \ldots, T_m\}$ is a template library of global consistency proofs
- $\text{Match}: \mathcal{C}^* \to \mathcal{T} \cup \{\bot\}$ maps certificate signatures to templates
- $\text{Instantiate}: \mathcal{T} \times \mathcal{C}^* \to \mathcal{G}$ produces global certificates

If the local certificates $\mathcal{C}$ satisfy:
1. **Interface Completeness:** Each $C_i$ certifies a decidable local property
2. **Compositional Closure:** The conjunction $\bigwedge_i C_i$ determines a unique template match
3. **Template Coverage:** For "good" problem classes, some template $T_j \in \mathcal{T}$ matches

Then the global consistency certificate $G = \text{Instantiate}(\text{Match}(\mathcal{C}), \mathcal{C})$ is:
- **Sound:** $G$ proves global consistency if and only if the system is globally consistent
- **Complete for Good Classes:** All problems in the good class yield $G \neq \bot$
- **Efficiently Constructible:** $|G| = \text{poly}(|\mathcal{C}|)$ and construction is polynomial-time

**Corollary (Composable Proof Systems).**
The local-to-global compilation corresponds to the composability property of interactive proofs: locally checkable certificates can be composed into globally valid proofs without re-verification of components.

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Soft certificate $K^+$ | Local consistency proof | Witness for decidable predicate |
| Template database $\mathcal{T}$ | Proof template library | Parameterized verification schemes |
| Template matching | Signature-based dispatch | Pattern matching on certificate structure |
| Well-posedness $K_{\mathrm{WP}}^+$ | Global consistency certificate | Complete proof of correctness |
| Good types | Efficiently verifiable problem classes | Problems with polynomial certificates |
| Automation Guarantee | Completeness for good classes | Template coverage theorem |
| Signature extraction | Certificate fingerprinting | Hash/feature extraction from local proofs |
| Theorem instantiation | Template parameter filling | Substitution in proof schema |
| Continuation criterion | Progress measure | Ranking function for termination |
| Inconclusive certificate $K^{\mathrm{inc}}$ | "Unknown" output | Honest failure for hard instances |
| Energy certificate $K_{D_E}^+$ | Resource bound proof | Proof of polynomial resource usage |
| Boundary certificate $K_{\mathrm{Bound}}^+$ | Interface specification | I/O format and constraint proof |
| Scaling certificate $K_{\mathrm{SC}}^+$ | Complexity scaling proof | Proof of sublinear/polynomial growth |
| Representation certificate $K_{\mathrm{Rep}}^+$ | Finite description proof | Proof of bounded description complexity |
| Substrate certificate $K_{\mathcal{H}_0}^+$ | Transition system proof | Proof of valid state-space structure |
| PDE template (parabolic, wave, etc.) | Complexity class (P, NP, PSPACE) | Structural problem classification |
| Strichartz estimates | Dispersive computation bounds | Spacetime tradeoffs in verification |
| Energy-dissipation inequality | Resource monotonicity | Proof that resources decrease monotonically |
| Critical regularity $s_c$ | Minimum witness size | Optimal certificate complexity |
| Blowup criterion | Termination failure condition | Condition for non-halting behavior |

---

## Logical Framework

### Compositional Verification Systems

**Definition (Local Certificate).** A local certificate $C$ for property $P$ on component $i$ is a tuple:
$$C = (\text{witness}, \text{property}, \text{component\_id}, \text{verification\_time})$$

such that $\text{Verify}(C) = 1$ implies $P(i)$ holds, and $|\text{witness}| = \text{poly}(|i|)$.

**Definition (Template).** A proof template $T$ is a parameterized proof scheme:
$$T = (\text{signature}, \text{preconditions}, \text{theorem}, \text{instantiation\_map})$$

where:
- $\text{signature}$ identifies the template via certificate features
- $\text{preconditions}$ lists required local certificates
- $\text{theorem}$ is the global consistency result
- $\text{instantiation\_map}$ fills parameters from certificates

**Definition (Global Certificate).** The global certificate $G$ produced by compilation:
$$G = (\text{template\_id}, \text{local\_certificates}, \text{instantiated\_proof}, \text{validity\_bound})$$

### Connection to Probabilistically Checkable Proofs

The FACT-SoftWP compilation mirrors PCP composition:

| PCP Concept | FACT-SoftWP Analog |
|-------------|---------------------|
| Local verifier | Soft certificate checker |
| Proof oracle | Template database |
| Query complexity | Number of local certificates needed |
| Soundness | Template matching correctness |
| Completeness | Good type coverage |
| Composition theorem | Soft→WP compilation |

**PCP Composition Principle:** If a proof can be locally verified with $q$ queries each with soundness $s$, then composition yields a globally sound proof with soundness $s^q$.

**FACT-SoftWP Analog:** If each soft certificate $C_i$ is locally verifiable with soundness 1 (deterministic), then their composition via template matching yields a globally sound well-posedness certificate.

---

## Proof Sketch

### Setup: Compositional Verification Framework

**Definition (Verification System).**
A compositional verification system is a tuple $\mathcal{V} = (\mathcal{C}, \mathcal{T}, \text{Match}, \text{Instantiate})$ where:

- $\mathcal{C}$ is the space of local certificates
- $\mathcal{T}$ is the template library
- $\text{Match}: 2^{\mathcal{C}} \to \mathcal{T} \cup \{\bot\}$ performs signature-based dispatch
- $\text{Instantiate}: \mathcal{T} \times 2^{\mathcal{C}} \to \mathcal{G}$ produces global certificates

**Definition (Good Problem Class).**
A problem class $\mathcal{P}$ is "good" for verification system $\mathcal{V}$ if:
1. Every instance $x \in \mathcal{P}$ admits local certificates $\mathcal{C}(x) \subseteq \mathcal{C}$
2. For every $x \in \mathcal{P}$, $\text{Match}(\mathcal{C}(x)) \neq \bot$
3. The compilation $\text{Instantiate}(\text{Match}(\mathcal{C}(x)), \mathcal{C}(x))$ is polynomial-time

**Correspondence to Hypostructure.** The good types $T \in \{T_{\text{parabolic}}, T_{\text{dispersive}}, T_{\text{hyperbolic}}\}$ satisfying the Automation Guarantee correspond to problem classes in P, BPP, or other tractable complexity classes.

---

### Step 1: Signature Extraction (Feature Fingerprinting)

**Claim.** Given local certificates $\{C_1, \ldots, C_k\}$, extract a signature $\Sigma$ that uniquely identifies the applicable template.

**Construction.** Define the signature extractor:
$$\Sigma: \mathcal{C}^* \to \text{Sig}(\mathcal{T})$$

The signature consists of:

1. **Resource Profile** (from $C_{D_E}$ - Energy Certificate):
   - Extract resource bound $B$ and monotonicity type
   - Classify: polynomial ($B = n^{O(1)}$), exponential ($B = 2^{O(n)}$), etc.
   - Determine if resources are strictly decreasing (dissipative) or conserved

2. **Interface Structure** (from $C_{\mathrm{Bound}}$ - Boundary Certificate):
   - Extract I/O specification and constraint type
   - Classify: closed system (no I/O), open system (specified interfaces)
   - Verify interface regularity (well-defined data formats)

3. **Scaling Behavior** (from $C_{\mathrm{SC}}$ - Scaling Certificate):
   - Extract scaling exponents $(\alpha, \beta)$
   - Compute scaling gap $\Delta = \alpha - \beta$
   - Classify: subcritical ($\Delta > 0$), critical ($\Delta = 0$), supercritical ($\Delta < 0$)

4. **Description Complexity** (from $C_{\mathrm{Rep}}$ - Representation Certificate):
   - Extract Kolmogorov complexity bound $K$
   - Verify finite representability
   - Classify by description length

**Template Matching Algorithm:**

```
function MATCH_TEMPLATE(Sigma):
    resource_profile <- Sigma.resource_type
    interface_type <- Sigma.interface_structure
    scaling_gap <- Sigma.scaling_behavior
    description <- Sigma.representation

    if resource_profile = "dissipative" and interface_type = "bounded":
        if scaling_gap >= 0:
            return T_polynomial  # Corresponds to T_para

    if resource_profile = "conservative" and interface_type = "dispersive":
        if scaling_gap >= 0:
            return T_spacetime  # Corresponds to T_wave/T_NLS

    if description = "finite_state":
        return T_finite  # Corresponds to T_hyp

    # No template matched
    return TEMPLATE_MISS
```

**Complexity.** Signature extraction is $O(|\mathcal{C}|)$ - linear scan of local certificates.

---

### Step 2: Template Instantiation (Parameterized Proof Filling)

**Claim.** Given matched template $T_*$ and local certificates $\mathcal{C}$, instantiate a complete global proof.

**Case Analysis by Template Type:**

#### Case 2.1: Polynomial Resource Template ($T_{\text{polynomial}}$)

**Template Signature:** Dissipative resource profile + bounded interface

**Corresponding Complexity Class:** Problems solvable in polynomial time with polynomial-size witnesses.

**Instantiation:**

1. **Resource Bound Extraction:** From $C_{D_E}$, extract $B(n) = n^c$ for some constant $c$
2. **Monotonicity Verification:** Check that resource decreases: $R(t+1) \leq R(t)$
3. **Termination Bound:** By Discrete Lyapunov (cf. KRNL-Consistency), termination in $\leq B(n)$ steps
4. **Global Certificate Construction:**
   $$G_{\text{poly}} = (\text{bound} = n^c, \text{witness\_size} = \text{poly}(n), \text{verification} = O(n^c))$$

**Complexity Interpretation:** This corresponds to the energy method for parabolic PDE - resources (energy) dissipate monotonically, guaranteeing termination.

#### Case 2.2: Spacetime Tradeoff Template ($T_{\text{spacetime}}$)

**Template Signature:** Conservative resource + dispersive interface

**Corresponding Complexity Class:** Problems with time-space tradeoffs (e.g., branching programs, streaming algorithms).

**Instantiation:**

1. **Conservation Law Extraction:** From $C_{D_E}$, verify $R(t) = R(0)$ (resource conservation)
2. **Dispersive Bound Extraction:** From $C_{\mathrm{SC}}$, extract dispersive decay rate
3. **Spacetime Product Bound:** Total work bounded by space $\times$ time tradeoff
4. **Global Certificate Construction:**
   $$G_{\text{st}} = (\text{space} = S, \text{time} = T, S \cdot T = \text{poly}(n))$$

**Complexity Interpretation:** This corresponds to Strichartz estimates - the spacetime norm is controlled even though individual norms may grow.

#### Case 2.3: Finite State Template ($T_{\text{finite}}$)

**Template Signature:** Finite description in $C_{\mathrm{Rep}}$

**Corresponding Complexity Class:** Regular languages, finite automata, bounded-width computation.

**Instantiation:**

1. **State Space Extraction:** From $C_{\mathrm{Rep}}$, extract finite state description
2. **Transition Verification:** Verify transitions are well-defined
3. **Termination by Finiteness:** Bounded reachability in finite graph
4. **Global Certificate Construction:**
   $$G_{\text{fin}} = (\text{states} = Q, |Q| < \infty, \text{verification} = O(|Q|^2))$$

**Complexity Interpretation:** This corresponds to symmetric hyperbolic systems - finite propagation speed ensures bounded computation.

---

### Step 3: Composition Correctness (Soundness of Compilation)

**Lemma (Local-to-Global Soundness).** If all local certificates $C_i$ are valid and template $T$ is correct, then the global certificate $G$ is valid.

**Proof (Compositional Induction):**

1. **Base Case (Local Validity):**
   Each $C_i$ certifies a local property $P_i$. By assumption, $\text{Verify}(C_i) = 1$ implies $P_i$ holds.

2. **Template Correctness:**
   Template $T$ encodes the theorem: $\bigwedge_i P_i \Rightarrow Q$ where $Q$ is global consistency.
   This is a meta-theorem validated once (e.g., from {cite}`CazenaveSemilinear03`, {cite}`Tao06`).

3. **Composition:**
   Since all $P_i$ hold (by local validity) and $\bigwedge_i P_i \Rightarrow Q$ (by template correctness), we have $Q$.

4. **Certificate Validity:**
   The global certificate $G$ packages the proof of $Q$:
   - Template ID: Identifies which meta-theorem was used
   - Local certificates: Provide the precondition proofs
   - Instantiation: Fills parameters correctly
   - Validity bound: Derived from template quantitative estimates

**Soundness.** $\text{Verify}(G) = 1$ if and only if global consistency holds. $\square$

---

### Step 4: Completeness for Good Classes

**Theorem (Template Coverage).** For every good problem class $\mathcal{P}$, there exists a template $T \in \mathcal{T}$ such that all instances $x \in \mathcal{P}$ match $T$.

**Proof (Classification Argument):**

Good problem classes are characterized by structural properties that determine template matching:

1. **Resource Structure:**
   - Dissipative → Polynomial template
   - Conservative → Spacetime template
   - Finite → Finite state template

2. **Scaling Behavior:**
   - Subcritical ($\Delta > 0$): Perturbation theory applies, standard templates work
   - Critical ($\Delta = 0$): Threshold cases, specialized templates
   - Supercritical ($\Delta < 0$): No general template (corresponds to undecidable cases)

3. **Interface Regularity:**
   - Well-posed interfaces admit template matching
   - Ill-posed interfaces emit $K^{\mathrm{inc}}$ (inconclusive)

**Automation Guarantee:** For good types, the template database $\mathcal{T}$ is complete:
$$\forall x \in \mathcal{P}_{\text{good}}.\ \text{Match}(\mathcal{C}(x)) \in \mathcal{T}$$

This mirrors the Automation Guarantee in hypostructure: good types have guaranteed template coverage.

---

### Step 5: Inconclusive Handling (Honest Failure)

**Definition (Inconclusive Certificate).** When no template matches, emit:
$$K^{\mathrm{inc}} = (\texttt{TEMPLATE\_MISS}, \Sigma, \text{manual\_hook})$$

**Properties:**
1. **Soundness Preserved:** $K^{\mathrm{inc}}$ makes no claim about global consistency
2. **Informative:** Includes signature $\Sigma$ for debugging
3. **Extensible:** Manual hook allows user-provided proofs

**Complexity Interpretation:** This corresponds to problems outside the good class:
- Undecidable problems: No template exists
- Hard problems (NP-complete, PSPACE-complete): Templates may exist but with exponential instantiation
- Novel structures: May require template library extension

**User Actions on $K^{\mathrm{inc}}$:**
1. Provide manual global proof (converted to $K^+$)
2. Extend template library with new pattern
3. Reformulate problem to fit existing templates

---

## Certificate Construction

The compilation produces explicit certificates:

**Local Certificate Package:**
$$\mathcal{C} = (C_{D_E}, C_{\mathrm{Bound}}, C_{\mathrm{SC}}, C_{\mathrm{Rep}}, C_{\mathcal{H}_0})$$

where each component certifies:
- $C_{D_E}$: Resource bounds and monotonicity
- $C_{\mathrm{Bound}}$: Interface specifications
- $C_{\mathrm{SC}}$: Scaling behavior
- $C_{\mathrm{Rep}}$: Finite representability
- $C_{\mathcal{H}_0}$: Valid transition structure

**Global Certificate Structure:**
$$G = (\text{template\_id}, \text{theorem\_ref}, \text{critical\_parameter}, \text{continuation\_criterion})$$

**Explicit Certificate Tuple:**
```
G := (
    template_id          : T_poly | T_spacetime | T_finite,
    theorem_reference    : citation to meta-theorem,
    critical_parameter   : minimum witness/resource parameter,
    continuation_bound   : progress measure for termination,
    local_certificates   : [C_1, ..., C_k],
    instantiation_map    : parameter assignments from C_i
)
```

**Verification Algorithm:**
```
function VERIFY_GLOBAL(G):
    # Step 1: Verify local certificates
    for C_i in G.local_certificates:
        if not VERIFY_LOCAL(C_i):
            return REJECT

    # Step 2: Verify template match
    Sigma <- EXTRACT_SIGNATURE(G.local_certificates)
    if MATCH_TEMPLATE(Sigma) != G.template_id:
        return REJECT

    # Step 3: Verify instantiation
    expected_params <- COMPUTE_PARAMS(G.template_id, G.local_certificates)
    if G.instantiation_map != expected_params:
        return REJECT

    # Step 4: Verify continuation bound is computable
    if not COMPUTABLE(G.continuation_bound):
        return REJECT

    return ACCEPT
```

**Complexity of Verification:** $O(|\mathcal{C}| + |T|)$ - linear in certificate size plus template lookup.

---

## Quantitative Refinements

### Certificate Size Bounds

**Lemma (Certificate Compactness).** For good problem classes:
$$|G| = O\left(\sum_i |C_i| + |\text{template}|\right) = \text{poly}(n)$$

**Proof:** Each local certificate has polynomial size (by definition of good class). Template size is constant (finite library). Instantiation map is polynomial. $\square$

### Compilation Time Bounds

**Lemma (Efficient Compilation).** The compilation $\mathcal{C} \to G$ runs in time:
$$T_{\text{compile}} = O(|\mathcal{C}| \cdot \log|\mathcal{T}|)$$

**Proof:** Signature extraction is linear. Template matching is logarithmic (binary search on sorted templates). Instantiation is linear. $\square$

### Continuation Parameter Bounds

For each template type, the continuation parameter provides quantitative guarantees:

| Template | Continuation Parameter | Bound |
|----------|------------------------|-------|
| $T_{\text{poly}}$ | Resource remaining | $\leq B(n) = n^c$ |
| $T_{\text{spacetime}}$ | Spacetime product | $\leq S \cdot T = \text{poly}(n)$ |
| $T_{\text{finite}}$ | States visited | $\leq |Q|^2$ |

---

## Connections to Classical Results

### 1. PCP Theorem and Composition

**Theorem (PCP Composition, Arora-Safra).** Probabilistically checkable proofs compose: if $L$ has a PCP verifier with $q$ queries and soundness $s$, then composition yields a PCP with $q' = q$ queries and soundness $s' = s$.

**Connection to FACT-SoftWP:**

| PCP Composition | FACT-SoftWP Compilation |
|-----------------|-------------------------|
| Local verifier queries | Soft certificate verification |
| Proof composition | Template instantiation |
| Soundness preservation | Sound global certificate |
| Completeness | Template coverage for good types |

**Interpretation:** FACT-SoftWP is a deterministic analog of PCP composition for structured verification.

### 2. Interactive Proof Composition

**Theorem (IP Composition).** Interactive proofs compose: if $L_1, L_2 \in \text{IP}$, then $L_1 \cap L_2, L_1 \cup L_2 \in \text{IP}$.

**Connection to FACT-SoftWP:** The local certificates $C_i$ are like individual IP transcripts. The template instantiation composes them into a single global proof.

### 3. Proof Complexity and Certificate Compactness

**Frege Systems:** In proof complexity, Frege systems allow local proof steps to compose into global proofs. The template library $\mathcal{T}$ is analogous to a set of inference rules.

**Extended Frege:** Templates with auxiliary variables correspond to Extended Frege, which can prove some theorems exponentially more efficiently.

**Connection:** Good types correspond to problems with polynomial-size Frege proofs. Template matching identifies the appropriate proof system.

### 4. Descriptive Complexity Connection

**Theorem (Fagin 1974).** NP = $\exists$SO (existential second-order logic).

**Connection to FACT-SoftWP:**

| Descriptive Complexity | FACT-SoftWP |
|------------------------|-------------|
| $\exists$SO formula | Template specification |
| Model checking | Template matching |
| Witness structure | Local certificates |
| Formula complexity | Template library size |

For good types, the corresponding logic is decidable (FO, LFP, etc.).

### 5. Composable Security Proofs (UC Framework)

**Universal Composability:** In cryptography, UC-secure protocols compose: if $\pi_1, \pi_2$ are UC-secure, so is $\pi_1 \| \pi_2$.

**Connection to FACT-SoftWP:**

| UC Framework | FACT-SoftWP |
|--------------|-------------|
| Ideal functionality | Global consistency property |
| Protocol | Local certificates |
| Composition theorem | Template instantiation |
| Simulator | Instantiation map |

**Interpretation:** FACT-SoftWP provides a "universal composability" theorem for verification: locally verified components compose into globally verified systems.

---

## Worked Example: Polynomial-Time Verification Compilation

**Problem:** Verify that a computation halts in polynomial time.

**Local Certificates:**
1. $C_{D_E}$: Resource bound $R(n) = n^3$
2. $C_{\mathrm{Bound}}$: Input/output specification (well-formed)
3. $C_{\mathrm{SC}}$: Scaling exponent $\alpha = 3$ (polynomial)
4. $C_{\mathrm{Rep}}$: Finite-state transition system (TM description)
5. $C_{\mathcal{H}_0}$: Valid TM semantics

**Signature Extraction:**
$$\Sigma = (\text{dissipative}, \text{bounded\_IO}, \Delta = 3 > 0, \text{finite})$$

**Template Match:** $T_{\text{polynomial}}$

**Instantiation:**
- Bound parameter: $c = 3$
- Continuation criterion: Steps remaining $\leq n^3 - t$
- Theorem reference: Polynomial-time halting

**Global Certificate:**
```
G = (
    template_id        : T_poly,
    theorem_ref        : "PTIME-Halting",
    critical_param     : c = 3,
    continuation       : {steps_remaining > 0},
    local_certs        : [C_DE, C_Bound, C_SC, C_Rep, C_H0],
    instantiation      : {B = n^3, monotone = strict}
)
```

**Verification:** Check all local certificates, verify template match, confirm instantiation. Accept. $\square$

---

## Worked Example: Spacetime Verification Compilation

**Problem:** Verify a streaming algorithm with space-time tradeoff.

**Local Certificates:**
1. $C_{D_E}$: Space bound $S = \sqrt{n}$, time bound $T = \sqrt{n}$
2. $C_{\mathrm{Bound}}$: Stream interface (one-pass)
3. $C_{\mathrm{SC}}$: Conservation law (space $\times$ time constant)
4. $C_{\mathrm{Rep}}$: Finite memory states
5. $C_{\mathcal{H}_0}$: Valid streaming semantics

**Signature Extraction:**
$$\Sigma = (\text{conservative}, \text{streaming}, \Delta = 0, \text{finite})$$

**Template Match:** $T_{\text{spacetime}}$

**Instantiation:**
- Space parameter: $S = \sqrt{n}$
- Time parameter: $T = \sqrt{n}$
- Product bound: $S \cdot T = n$

**Global Certificate:**
```
G = (
    template_id        : T_spacetime,
    theorem_ref        : "Streaming-Bound",
    critical_param     : ST = n,
    continuation       : {bits_remaining > 0},
    local_certs        : [C_DE, C_Bound, C_SC, C_Rep, C_H0],
    instantiation      : {S = sqrt(n), T = sqrt(n)}
)
```

---

## Theoretical Implications

### Local-to-Global Paradigm

FACT-SoftWP exemplifies a fundamental principle in complexity theory:

**Local Verification Suffices:** For structured problems, verifying local consistency is sufficient to guarantee global consistency.

This appears in:
- **Constraint Satisfaction:** Local consistency propagation
- **Graph Algorithms:** Local-ratio technique
- **Distributed Computing:** Local checkability
- **Proof Systems:** Local verification (PCP)

### Template Libraries as Proof Strategies

The template library $\mathcal{T}$ can be viewed as:
1. A set of **proof strategies** for different problem structures
2. A **knowledge base** of verification techniques
3. A **compiler** from local properties to global guarantees

Extending $\mathcal{T}$ corresponds to developing new verification techniques.

### Limits of Composability

The inconclusive certificate $K^{\mathrm{inc}}$ captures limits:
- **Undecidable problems:** No template can exist
- **NP-complete problems:** Templates exist but require exponential instantiation
- **Novel structures:** Require new templates

---

## Summary

The FACT-SoftWP theorem, translated to complexity theory, states:

**Local consistency certificates compose into global consistency proofs via template matching, yielding polynomial-size global certificates for good problem classes.**

This principle:
1. Provides a framework for compositional verification
2. Connects to PCP composition and interactive proof theory
3. Explains how local properties imply global correctness
4. Offers constructive certificates for verification

The translation illuminates a deep connection between:
- Well-posedness theory (dynamical systems)
- Compositional verification (complexity theory)
- Local-to-global principles (proof theory)
- Template-based reasoning (knowledge representation)

**Key Insight:** Just as soft interface certificates (local energy bounds, boundary conditions, scaling laws) compose to yield well-posedness, local computational certificates compose to yield global consistency guarantees through template-based compilation.
