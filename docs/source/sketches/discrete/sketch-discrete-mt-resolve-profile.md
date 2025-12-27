---
title: "RESOLVE-Profile - Complexity Theory Translation"
---

# RESOLVE-Profile: Kernelization Trichotomy

## Overview

This document provides a complete complexity-theoretic translation of the RESOLVE-Profile theorem (Profile Classification Trichotomy) from the hypostructure framework. The translation establishes a formal correspondence between profile classification outcomes and the **Kernelization Trichotomy** in parameterized complexity theory, where problems classify as having polynomial kernels, XP algorithms, or no known tractable structure.

**Original Theorem Reference:** {prf:ref}`mt-resolve-profile`

---

## Hypostructure Context

The RESOLVE-Profile theorem (Profile Classification Trichotomy) states that at the Profile node, after CompactCheck passes, the framework produces exactly one of three certificates:

1. **Case 1: Finite library membership** ($K_{\text{lib}}$) - The limiting profile $V$ belongs to a finite, pre-classified library $\mathcal{L}$ of canonical profiles.

2. **Case 2: Tame stratification** ($K_{\text{strat}}$) - Profiles are parameterized in a definable (o-minimal) family $\mathcal{F}$ with finite stratification. Classification is tractable though not finite.

3. **Case 3: Classification Failure** ($K_{\mathrm{prof}}^-$) - Either NO-wild (profile exhibits wildness witness) or NO-inconclusive (methods exhausted without resolution).

The key insight: extracted limiting profiles either belong to a classifiable family enabling further analysis, or demonstrate inherent classification complexity.

**Literature:** Lions 1984, Kenig-Merle 2006 (Concentration-Compactness Principle)

---

## Complexity Theory Statement

**Theorem (Kernelization Trichotomy).**
Let $(L, k)$ be a parameterized problem in NP. The problem classifies into exactly one of three structural categories:

1. **Polynomial Kernel (Finite Library):** The problem admits a kernel of size $f(k)$ for polynomial $f$. The hard core is bounded and belongs to a classifiable family.

2. **XP/FPT with Tame Structure:** The problem admits an algorithm running in time $O(n^{g(k)})$ or $O(f(k) \cdot n^c)$, where complexity depends on definable parameter families with bounded stratification.

3. **W-hard / No Polynomial Kernel:** The problem is W[t]-hard for some $t \geq 1$, or provably has no polynomial kernel under standard complexity assumptions. Classification of hard instances is intractable.

**Formal Statement:** For parameterized problem $(L, k) \in \text{NP}$, define:
- $K_L(n, k) := \min\{s : (L, k) \text{ admits kernel of size } s\}$
- $\mathcal{C}_L := \{\text{hard cores of instances at parameter } k\}$ (the "profile library")

Then exactly one of the following holds:

| Outcome | Kernel Structure | Problem Class |
|---------|------------------|---------------|
| **Polynomial Kernel** | $K_L(n, k) = O(k^c)$ | Finite classifiable core family |
| **XP/Tame** | $K_L(n, k) = O(n^{g(k)})$ | Definable stratified family |
| **W-hard/No Kernel** | $K_L(n, k) \geq n^{1-\epsilon}$ (conditional) | Wild or unclassifiable |

---

## Terminology Translation Table

| Hypostructure Term | Complexity Theory Equivalent |
|--------------------|------------------------------|
| Limiting profile $V$ | Hard kernel core $\kappa(x, k)$ |
| Profile node (after CompactCheck) | Post-kernelization phase |
| Finite library $\mathcal{L}$ | Polynomial kernel catalog |
| Canonical profiles | Standard hard instances (clique, vertex cover cores) |
| Tame stratification | XP parameter hierarchy |
| Definable family $\mathcal{F}$ | Parameterized complexity class (FPT, XP, W[1], ...) |
| O-minimal structure | Bounded alternation / stratified hardness |
| Classification failure (NO-wild) | W[t]-hardness, kernelization lower bound |
| Classification failure (NO-inconclusive) | Open problem (neither FPT nor W-hard proven) |
| Wildness witness | W-hardness reduction |
| Chaotic attractor | Self-reducible hard core (SAT-like structure) |
| Scaling group $G$ | Parameter rescaling symmetries |
| CompactCheck | Kernelization preprocessing |
| Concentration-compactness | Reduction rule exhaustion |
| Profile decomposition | Sunflower lemma / crown decomposition |
| Energy escape | Parameter blowup under reductions |
| Moduli space $\mathcal{M}_{\text{prof}}$ | Parameterized problem zoo |

---

## Parameterized Complexity Background

### The W-Hierarchy

**Definition.** The W-hierarchy classifies parameterized problems by circuit depth:

$$\text{FPT} \subseteq \text{W}[1] \subseteq \text{W}[2] \subseteq \cdots \subseteq \text{W}[P] \subseteq \text{XP}$$

- **FPT (Fixed-Parameter Tractable):** Solvable in time $f(k) \cdot n^{O(1)}$
- **W[1]:** Complete problem is CLIQUE (find $k$-clique in graph)
- **W[2]:** Complete problem is DOMINATING-SET (find $k$-dominating set)
- **W[P]:** Complete problem is CIRCUIT-SAT parameterized by number of true gates
- **XP:** Solvable in time $n^{f(k)}$ (parameter appears in exponent)

**Intuition:** W[t]-complete problems have "hard cores" requiring $n^{\Omega(k)}$ time, analogous to profiles with uncontrolled energy concentration.

### Kernelization

**Definition.** A kernelization for $(L, k)$ is a polynomial-time algorithm producing:
$$(x, k) \mapsto (x', k')$$

where:
- $|x'| + k' \leq f(k)$ for computable $f$
- $(x, k) \in L \Leftrightarrow (x', k') \in L$

The output $(x', k')$ is the **kernel** - the irreducible hard core.

**Key Theorem (Bodlaender et al. 2009, Fortnow-Santhanam 2011):**
$$\text{FPT} = \text{problems with kernelization}$$

Moreover, kernel size characterizes tractability:
- Polynomial kernel $\Rightarrow$ efficient classification
- No polynomial kernel $\Rightarrow$ W-hardness or compositional hardness

---

## Proof Sketch

### Setup: The Profile-Kernel Correspondence

We establish the correspondence between profile classification and kernel classification:

| Hypostructure | Parameterized Complexity |
|---------------|--------------------------|
| Singularity sequence $(t_n, x_n) \to (T_*, x_*)$ | Instance sequence $(x_n, k_n)$ with growing difficulty |
| Compactness modulo symmetry | Reduction rules modulo isomorphism |
| Limiting profile $V = \lim g_n \cdot u(t_n)$ | Kernel core $\kappa = \lim \text{Reduce}(x_n)$ |
| Profile library $\mathcal{L}_T$ | Kernel catalog $\mathcal{K}$ |
| Definable family $\mathcal{F}_T$ | Parameterized complexity class |

**Profile Extraction = Kernelization:**

Given instance $(x, k)$, the kernelization process mirrors profile extraction:

1. **Apply reduction rules** (scaling/symmetry): $x \to x_1 \to x_2 \to \cdots \to \kappa(x)$
2. **Extract invariant core** (profile): The kernel $\kappa(x)$ is the irreducible hard core
3. **Classify core** (library lookup): Check if $\kappa \in \mathcal{K}$ (polynomial kernel catalog)

---

### Step 1: Instance Sequence and Reduction Exhaustion

**Setup:** Consider a sequence of instances $(x_n, k)$ for fixed parameter $k$ with $|x_n| \to \infty$.

**Reduction Phase (CompactCheck Analogue):**

Apply polynomial-time reduction rules exhaustively:

```
Reduce(x, k):
  while applicable reduction rule rho_i exists:
    (x, k) := rho_i(x, k)
  return (x, k)  -- the kernel
```

**Reduction Rules (Scaling Symmetries):**

| Problem | Reduction Rule | Analogue |
|---------|---------------|----------|
| Vertex Cover | Remove isolated vertices | Remove dispersive modes |
| k-Path | Contract degree-1 vertices | Quotient by stabilizer |
| Dominating Set | Crown reduction | O-minimal cell decomposition |
| SAT | Unit propagation | Energy dissipation |

**Fixed Point (Profile):**

The kernel $\kappa(x, k) = \text{Reduce}(x, k)$ is the fixed point under all reduction rules. This is the computational analogue of the limiting profile $V$.

---

### Step 2: Trichotomy Classification

After reduction exhaustion, classify the kernel $\kappa(x, k)$:

---

#### Case 1: Polynomial Kernel (Finite Library)

**Criterion:** $|\kappa(x, k)| \leq p(k)$ for polynomial $p$.

**Interpretation:** The hard core has size bounded by a polynomial in the parameter alone. This means:
- The kernel catalog $\mathcal{K}_k = \{\kappa : |\kappa| \leq p(k)\}$ is finite for each $k$
- Classification reduces to lookup in a finite table
- FPT algorithm: Kernelize + brute-force search on bounded core

**Certificate Produced:**
$$K_{\text{poly-kernel}} = (\kappa, |\kappa| \leq p(k), \mathcal{K}_k, \text{FPT algorithm})$$

**Examples:**
- **Vertex Cover:** Kernel of size $2k$ (Nemhauser-Trotter)
- **Feedback Vertex Set:** Kernel of size $O(k^2)$
- **Point Line Cover:** Kernel of size $O(k^2)$

**Connection to $K_{\text{lib}}$:**

The polynomial kernel catalog $\mathcal{K}_k$ corresponds to the finite library $\mathcal{L}$ of canonical profiles. Each kernel type represents a "standard singularity" with known properties:

| Profile Type | Kernel Type |
|--------------|-------------|
| Cylinder (Ricci flow) | Star graph (Vertex Cover) |
| Sphere (MCF) | Clique core (Clique Cover) |
| Soliton (NLS) | Path core (k-Path) |
| Ground state | Minimum dominating structure |

---

#### Case 2: XP with Tame Stratification

**Criterion:** $|\kappa(x, k)| = O(n^{g(k)})$ but problem is in XP with structured parameterization.

**Interpretation:** The hard core size depends on instance size, but with controlled (definable) dependence on parameter. The parameter hierarchy stratifies complexity:

$$\mathcal{C}(k) := \{(x, k) : \text{kernel size} = \Theta(n^{c_k})\}$$

forms a definable family with finite stratification by $k$.

**Certificate Produced:**
$$K_{\text{XP}} = (\kappa, |\kappa| = O(n^{g(k)}), \text{XP algorithm}, \text{stratification data})$$

**Examples:**
- **k-Clique:** In XP via $O(n^k)$ brute force, W[1]-complete
- **k-Dominating Set:** In XP via $O(n^k)$ brute force, W[2]-complete
- **Bandwidth:** In XP, exact complexity stratified by parameter

**Connection to $K_{\text{strat}}$:**

The XP stratification corresponds to the tame family $\mathcal{F}$ with o-minimal structure:

- Each stratum $\mathcal{C}(k)$ is a definable set in the parameter-instance space
- The stratification has finite depth (number of distinct complexity behaviors)
- Classification is tractable: identify stratum, then apply stratum-specific algorithm

**O-Minimal Analogy:**

The W-hierarchy levels form an o-minimal structure in the following sense:
- **Cells:** Problems at each W[t] level
- **Definable sets:** Parameterized reductions preserve level
- **Finite stratification:** W[1] $\subset$ W[2] $\subset$ ... has finite height per problem

---

#### Case 3: Classification Failure (W-Hardness or No Polynomial Kernel)

**Case 3a: NO-wild (W-hardness):**

**Criterion:** $(L, k)$ is W[t]-complete for some $t \geq 1$.

**Wildness Witness:** A parameterized reduction from a W[t]-complete problem:
$$\text{W}[t]\text{-complete} \leq_{\text{fpt}} (L, k)$$

**Interpretation:** The hard core exhibits "chaotic" structure - self-similarity under parameter scaling. Like turbulent cascades or chaotic attractors, W-hard problems have:
- Self-reducibility: Large instances embed smaller instances
- Scale invariance: Hardness preserved under parameter transformation
- No polynomial kernel (under standard assumptions)

**Certificate Produced:**
$$K_{\text{wild}} = (\text{W}[t]\text{-hard}, \text{reduction witness}, \text{no poly kernel conditional})$$

**Examples:**
- **Clique:** W[1]-complete via reduction from Independent Set
- **Dominating Set:** W[2]-complete via reduction from Set Cover
- **SAT (parameterized by clauses):** W[2]-complete

**Connection to $K_{\mathrm{prof}}^{\mathrm{wild}}$:**

The W-hardness reduction is the computational analogue of:
- Positive Lyapunov exponent (chaotic dynamics)
- Turbulent cascade (energy spreads to all scales)
- Undecidable structure (reduction encodes halting-like problems)

---

**Case 3b: NO-inconclusive:**

**Criterion:** Neither polynomial kernel nor W-hardness proven.

**Interpretation:** Classification methods exhausted without definitive result. The problem occupies the "grey zone" of parameterized complexity.

**Certificate Produced:**
$$K_{\text{inconclusive}} = (\text{unknown}, \text{methods exhausted}, \text{current bounds})$$

**Examples:**
- **Graph Isomorphism (parameterized by treewidth):** Neither FPT nor W[1]-hard proven
- **Integer Factoring (parameterized by number of prime factors):** Complexity open
- **Minimum Fill-In:** Polynomial kernel status unresolved for some parameterizations

**Connection to $K_{\mathrm{prof}}^{\mathrm{inc}}$:**

This corresponds to:
- Profile classification requiring more structure theory
- Insufficient algebraic/geometric constraints to decide
- Open problems in the field

---

### Step 3: The Trichotomy is Exhaustive

**Theorem (Kernelization Trichotomy):**
Every parameterized problem in NP falls into exactly one of:
1. Admits polynomial kernel (FPT with bounded core)
2. In XP with stratified parameter dependence
3. No polynomial kernel (W-hard or compositionally hard)

**Proof Sketch:**

The trichotomy follows from the kernelization-FPT equivalence and W-hierarchy structure:

1. **Polynomial kernel $\Rightarrow$ FPT:** Kernelize then brute-force on $O(k^c)$ core
2. **FPT without poly kernel $\Rightarrow$ XP:** Time $f(k) \cdot n^c$ implies $n^{O(1)}$ for fixed $k$
3. **Not FPT $\Rightarrow$ W-hard or open:** By definition of W-hierarchy

**Composition Theorem (Bodlaender-Downey-Fellows-Hermelin 2009):**

If $(L, k)$ admits polynomial kernel and has an OR-composition:
$$\bigvee_{i=1}^t (x_i, k) \in L \Leftrightarrow ((x_1, \ldots, x_t), k') \in L$$
with $k' = \text{poly}(k, \log t)$, then coNP $\subseteq$ NP/poly.

**Implication:** Under standard assumptions, compositional problems cannot have polynomial kernels, forcing Case 3.

---

## Certificate Construction

For each trichotomy outcome, we produce explicit certificates:

**Case 1: Polynomial Kernel (Finite Library)**
```
K_poly = {
  outcome: "Polynomial_Kernel",
  kernel_size: p(k),
  catalog: K_k = {kappa_1, kappa_2, ..., kappa_m},
  fpt_algorithm: A_fpt,
  time_bound: f(k) * n^c,
  evidence: {
    kernelization: pi_kernel,
    catalog_membership: kappa in K_k,
    profile_type: "finite_library"
  },
  literature: "Downey-Fellows 1999, Cygan et al. 2015"
}
```

**Case 2: XP / Tame Stratification**
```
K_xp = {
  outcome: "XP_Stratification",
  kernel_size: n^{g(k)},
  stratification: {C(1), C(2), ..., C(k)},
  xp_algorithm: A_xp,
  time_bound: n^{f(k)},
  w_level: t (if W[t]-complete),
  evidence: {
    stratum_identification: c_k,
    definable_family: F,
    profile_type: "tame_family"
  },
  literature: "Flum-Grohe 2006"
}
```

**Case 3a: W-Hardness (Wild)**
```
K_wild = {
  outcome: "W_Hard",
  w_level: t,
  wildness_witness: {
    reduction: "W[t]-complete <=_fpt L",
    source_problem: P_complete,
    reduction_function: rho,
    parameter_bound: k' <= g(k)
  },
  kernel_lower_bound: "n^{1-eps} unless coNP in NP/poly",
  evidence: {
    self_similarity: "reduction embeds instances",
    chaos_indicator: "positive Lyapunov exponent analogue",
    profile_type: "wild"
  },
  literature: "Downey-Fellows 1995"
}
```

**Case 3b: Inconclusive**
```
K_inconclusive = {
  outcome: "Inconclusive",
  status: "classification_open",
  current_bounds: {
    upper: "XP via n^{f(k)}",
    lower: "no W[1]-hardness known"
  },
  evidence: {
    methods_tried: [kernelization, reductions, algebraic],
    profile_type: "inconclusive"
  },
  literature: "current research frontier"
}
```

---

## Connections to Parameterized Complexity Hierarchy

### 1. FPT and Polynomial Kernelization

**Theorem (Cai-Chen-Downey-Fellows 1997):**
$(L, k) \in \text{FPT} \Leftrightarrow (L, k)$ admits a kernelization.

**Connection to Profile Library:**

- FPT $\Leftrightarrow$ finite profile library (bounded core catalog)
- Polynomial kernel $\Leftrightarrow$ profiles have bounded "energy" $\Phi(V) \leq p(k)$
- Kernelization algorithm $\Leftrightarrow$ ProfileExtractor in hypostructure

**Key Results:**
- Vertex Cover: $2k$ kernel (optimal up to constant)
- Feedback Vertex Set: $O(k^2)$ kernel
- k-Path: $O(k \log k)$ kernel via color-coding

### 2. Kernelization Lower Bounds

**Theorem (Fortnow-Santhanam 2011):**
If SAT parameterized by number of variables has polynomial kernel, then coNP $\subseteq$ NP/poly.

**Theorem (Bodlaender-Thomassé-Yeo 2011):**
Many natural problems (k-Path, k-Cycle, etc.) require kernels of size $\Omega(k^{2-\epsilon})$ unless coNP $\subseteq$ NP/poly.

**Connection to Wild Profiles:**

Kernelization lower bounds correspond to:
- Profile energy that cannot be reduced below threshold
- Self-similar structure preventing classification
- Compositional hardness (OR-composition implies no poly kernel)

### 3. The W-Hierarchy as Stratification

**Structure:**
$$\text{FPT} = \text{W}[0] \subsetneq \text{W}[1] \subsetneq \text{W}[2] \subsetneq \cdots \subsetneq \text{W}[P] \subsetneq \text{XP}$$

(Separations conditional on FPT $\neq$ W[1])

**Complete Problems at Each Level:**

| Level | Complete Problem | Profile Analogue |
|-------|-----------------|------------------|
| W[1] | k-Clique, k-Independent Set | First-order concentration |
| W[2] | k-Dominating Set, k-Set Cover | Second-order concentration |
| W[t] | Weighted Weft-t Circuit SAT | t-th order concentration |
| W[P] | Circuit SAT (bounded true gates) | Polynomial-depth singularity |

**Connection to O-Minimal Stratification:**

The W-hierarchy provides an o-minimal-like stratification:
- Each W[t] level is a "stratum" with uniform complexity behavior
- The weft parameter measures "oscillation depth" (circuit alternation)
- Problems within a level have comparable hardness (complete problems equivalent)

### 4. Turing Kernelization and Adaptive Profiles

**Definition:** A Turing kernelization for $(L, k)$ is an algorithm using oracle access to a kernel oracle, where each query has size $\leq f(k)$.

**Theorem (Binkele-Raible et al. 2012):**
Some problems admit Turing kernels but not standard kernels.

**Connection to Profile Decomposition:**

Turing kernelization corresponds to:
- Profile decomposition into multiple bubbles $V^{(1)}, V^{(2)}, \ldots$
- Each bubble has bounded size (oracle query)
- Interaction between bubbles requires adaptive computation

This mirrors the Bahouri-Gerard decomposition in concentration-compactness.

### 5. Cross-Composition and Profile Wildness

**Definition:** An OR-cross-composition from $L$ to $(L', k)$ produces:
$$\bigvee_{i=1}^t x_i \in L \Leftrightarrow (x', k') \in L'$$
with $k' = \text{poly}(\max_i |x_i|, \log t)$.

**Theorem:** If $(L', k)$ has polynomial kernel and OR-cross-composition from NP-complete $L$, then coNP $\subseteq$ NP/poly.

**Connection to Wild Profiles:**

Cross-composition is the computational analogue of:
- Turbulent cascade: Information from many scales combines
- Energy pile-up: Cannot reduce below threshold without polynomial collapse
- Wildness witness: The composition encodes NP-completeness

---

## Quantitative Bounds

### Kernel Size Hierarchy

| Problem | Kernel Size | Classification |
|---------|-------------|----------------|
| Vertex Cover | $2k$ | Poly kernel (optimal) |
| Feedback Vertex Set | $O(k^2)$ | Poly kernel |
| k-Path | $O(k^{1.5})$ | Poly kernel |
| k-Dominating Set (general) | No poly kernel | W[2]-complete |
| k-Clique | No poly kernel | W[1]-complete |
| Treewidth | No poly kernel | FPT, no poly kernel |

### Time Complexity by Class

| Class | Time Bound | Kernel Status |
|-------|------------|---------------|
| FPT | $f(k) \cdot n^{O(1)}$ | Admits kernel |
| Poly-kernel FPT | $f(k) \cdot n^{O(1)}$, $K = O(k^c)$ | Polynomial kernel |
| FPT no-poly | $f(k) \cdot n^{O(1)}$, $K = \omega(k^c)$ | Super-poly kernel |
| XP | $n^{f(k)}$ | May or may not have kernel |
| W[t]-complete | $n^{\Omega(k)}$ unless FPT=W[1] | No poly kernel (conditional) |

### Profile Energy Thresholds

Translating to hypostructure energy $\Phi$:

| Profile Type | Energy Bound | Kernel Analogue |
|--------------|--------------|-----------------|
| Dispersive | $\Phi(V) = 0$ | Trivial kernel (problem in P) |
| Subcritical | $\Phi(V) \leq \Phi_c$ | Polynomial kernel |
| Critical | $\Phi(V) = \Phi_c$ | Borderline (FPT but no poly kernel) |
| Supercritical | $\Phi(V) > \Phi_c$ | W-hard, no poly kernel |
| Wild | $\Phi(V) = \infty$ | No kernel exists |

---

## Worked Example: Vertex Cover Kernelization

**Problem:** Given graph $G = (V, E)$ and parameter $k$, decide if $G$ has vertex cover of size $\leq k$.

**Thin Input:** $(G, k)$ with $|V| = n$, $|E| = m$

**Kernelization (Profile Extraction):**

1. **Rule 1 (Isolated vertices):** Remove vertices with degree 0
   - *Analogue:* Remove dispersive modes with zero energy

2. **Rule 2 (Pendant vertices):** For degree-1 vertex $v$ with neighbor $u$: include $u$ in cover, delete $u$ and its neighbors, set $k := k - 1$
   - *Analogue:* Quotient by symmetry stabilizer

3. **Rule 3 (High-degree vertices):** If $\deg(v) > k$, include $v$ in cover (else need $> k$ vertices to cover $v$'s edges)
   - *Analogue:* Concentration triggers inclusion

4. **Rule 4 (Nemhauser-Trotter):** Solve LP relaxation; include vertices with $x_v = 1$ in solution, exclude vertices with $x_v = 0$, keep "half-integral" vertices
   - *Analogue:* Crown decomposition = profile decomposition

**Kernel Size:** At most $2k$ vertices remain after exhaustive application of rules.

**Certificate:**
```
K_vc = {
  outcome: "Polynomial_Kernel",
  kernel_size: 2k,
  kernel_graph: G' with |V'| <= 2k,
  reduction_sequence: [rule_1, rule_2, rule_3, rule_4],
  profile_type: "finite_library",
  library: {star_k, matching_k, clique_k, ...},
  fpt_algorithm: "brute_force on 2^{2k} subsets",
  time_bound: O(1.2738^k + kn)
}
```

**Profile Library for Vertex Cover:**

The kernel structures form a finite catalog for each $k$:
- Star graphs $S_j$ with $j \leq k$ leaves
- Matching graphs $M_j$ with $j \leq k$ edges
- Crown structures
- Combinations thereof

Each kernel type has known covering structure, enabling efficient classification.

---

## Worked Example: Clique (W[1]-Complete)

**Problem:** Given graph $G = (V, E)$ and parameter $k$, decide if $G$ contains $k$-clique.

**Classification Attempt:**

1. **Kernelization:** No polynomial kernel (unless coNP $\subseteq$ NP/poly)
2. **Best algorithm:** $O(n^{k/3})$ via matrix multiplication, $O(1.62^k \cdot n^2)$ via fast matrix multiplication
3. **W[1]-hardness:** Reduction from INDEPENDENT-SET

**Wildness Witness (W[1]-Reduction):**

The reduction from k-INDEPENDENT-SET to k-CLIQUE via complement graph:
$$G \text{ has independent set of size } k \Leftrightarrow \bar{G} \text{ has clique of size } k$$

This is an FPT reduction (parameter preserved, polynomial time).

**Self-Similarity (Chaotic Structure):**

The clique structure is self-similar:
- A $k$-clique contains $\binom{k}{j}$ many $j$-cliques for each $j < k$
- Hardness cascades: Finding large cliques requires finding smaller cliques
- No polynomial kernelization can break this self-reference

**Certificate:**
```
K_clique = {
  outcome: "W_Hard",
  w_level: 1,
  wildness_witness: {
    reduction: "k-INDEPENDENT-SET <=_fpt k-CLIQUE",
    transformation: "graph complement",
    parameter_bound: k' = k
  },
  kernel_lower_bound: "n^{k-eps} unless ETH fails",
  self_similarity: "k-clique contains (k choose j) j-cliques",
  profile_type: "wild"
}
```

---

## Summary

The RESOLVE-Profile theorem, translated to parameterized complexity, states:

**Every parameterized problem classifies into exactly one of: polynomial kernel (finite library), XP with tame stratification, or W-hard/no polynomial kernel (wild or inconclusive).**

This trichotomy is **exhaustive** and **rigid**:

1. **Polynomial Kernel (Finite Library):** The hard core is bounded by $O(k^c)$. The kernel catalog is finite for each parameter value. Classification reduces to catalog lookup. Problems are FPT with efficient algorithms.

2. **XP / Tame Stratification:** The hard core size depends on instance size but with controlled parameter dependence. The W-hierarchy provides definable stratification. Classification is tractable within each stratum.

3. **W-Hard / No Kernel (Wild):** The hard core exhibits self-similar structure. No polynomial kernelization exists under standard assumptions. Classification is intractable.

**Key Correspondences:**

| Hypostructure | Parameterized Complexity |
|---------------|--------------------------|
| Limiting profile $V$ | Kernel core $\kappa$ |
| Finite library $\mathcal{L}$ | Polynomial kernel catalog |
| Tame family $\mathcal{F}$ | W-hierarchy stratification |
| Wild profile | W[t]-hardness |
| ProfileExtractor | Kernelization algorithm |
| O-minimal structure | Bounded weft/depth |
| Concentration-compactness | Reduction rule exhaustion |

**Physical Interpretation:**

- **Poly kernel:** Singularity has bounded energy, classifiable into standard forms
- **XP/Tame:** Singularity has parameter-dependent energy, stratified behavior
- **W-hard:** Singularity exhibits turbulent cascade, self-similar at all scales

---

## Literature

1. **Downey, R. G. & Fellows, M. R. (1999).** *Parameterized Complexity.* Springer. *Foundational text on FPT and W-hierarchy.*

2. **Flum, J. & Grohe, M. (2006).** *Parameterized Complexity Theory.* Springer. *Comprehensive treatment including W-hierarchy.*

3. **Cygan, M. et al. (2015).** *Parameterized Algorithms.* Springer. *Modern techniques including kernelization.*

4. **Bodlaender, H. L. et al. (2009).** "Kernelization Lower Bounds by Cross-Composition." *SIAM J. Discrete Math.* *Compositional hardness.*

5. **Fortnow, L. & Santhanam, R. (2011).** "Infeasibility of Instance Compression and Succinct PCPs for NP." *JCSS.* *Kernel lower bounds.*

6. **Lokshtanov, D., Marx, D., & Saurabh, S. (2011).** "Lower Bounds Based on the Exponential Time Hypothesis." *Bulletin EATCS.* *ETH-based lower bounds.*

7. **Dell, H. & van Melkebeek, D. (2014).** "Satisfiability Allows No Nontrivial Sparsification Unless the Polynomial-Time Hierarchy Collapses." *JACM.* *Sparsification limits.*

8. **Lions, P.-L. (1984).** "The Concentration-Compactness Principle in the Calculus of Variations." *Annales IHP.* *Original concentration-compactness (hypostructure source).*

9. **Kenig, C. E. & Merle, F. (2006).** "Global Well-Posedness, Scattering and Blow-Up for the Energy-Critical NLS." *Inventiones Math.* *Profile classification in PDE.*

10. **van den Dries, L. (1998).** *Tame Topology and O-Minimal Structures.* Cambridge. *O-minimal structures and stratification.*

11. **Chen, Y. et al. (2006).** "On Miniaturized Problems in Parameterized Complexity Theory." *Theoretical Computer Science.* *W-hierarchy structure.*

12. **Kratsch, S. (2014).** "Recent Developments in Kernelization." *Bulletin EATCS.* *Survey of kernelization techniques.*
