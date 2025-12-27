---
title: "LOCK-Periodic - Complexity Theory Translation"
---

# LOCK-Periodic: The Periodic Law as Strategy Tables

## Overview

This document provides a complete complexity-theoretic translation of the LOCK-Periodic metatheorem (The Periodic Law, mt-lock-periodic) from the hypostructure framework. The theorem establishes that the proof strategy for any dynamical system is uniquely determined by its location in the 8x21 Periodic Table (Family x Stratum).

In complexity theory, this corresponds to **Strategy Tables** and **Algorithm Portfolios**: problem classification determines solution approach. Just as the position of an element in the chemical periodic table predicts its properties and reactions, the position of a computational problem in a classification matrix determines which algorithmic techniques apply.

**Original Theorem Reference:** {prf:ref}`mt-lock-periodic`

---

## Complexity Theory Statement

**Theorem (LOCK-Periodic, Strategy Table Form).**
Let $\mathcal{C}$ be a classification of computational problems into:
- **8 Families** (problem types based on structural properties)
- **21 Strata** (algorithmic phases/decision points)

Then:
1. **Row Determination:** The problem family determines the *class* of applicable algorithms
2. **Column Determination:** The stratum determines the *phase* and *technique* within that class
3. **Unique Strategy:** The (Family, Stratum) pair uniquely determines the optimal solution approach

**Formal Statement.** For any computational problem $P$:

$$\text{Strategy}(P) = \mathcal{S}(\text{Family}(P), \text{Stratum}(P))$$

where $\mathcal{S}: \{1, \ldots, 8\} \times \{1, \ldots, 21\} \to \text{Algorithms}$ is the **Strategy Table**.

**Corollary (Algorithm Selection).** Given a problem classification:
1. Compute $(\text{Family}, \text{Stratum})$ from problem features
2. Look up the strategy in the 8x21 matrix
3. Apply the prescribed algorithm with guaranteed applicability

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Equivalent | Formal Correspondence |
|-----------------------|------------------------------|------------------------|
| 8x21 Periodic Table | Strategy Table / Algorithm Portfolio | Problem-to-solution mapping |
| Family (Row) | Problem Class | Structural classification (P, NP, PSPACE, etc.) |
| Stratum (Column) | Decision Phase / Feature | Algorithmic phase or problem feature |
| Certificate $K_N$ | Feature Vector Component | Problem characteristic at phase $N$ |
| Structural DNA | Feature Signature | Complete problem characterization |
| Family I (Stable) | P / Easy Problems | A-priori polynomial algorithms |
| Family II (Relaxed) | Subexponential / FPT | Dispersive/parameterized methods |
| Family III (Gauged) | Reduction-Based | Solve via equivalence to known problem |
| Family IV (Resurrected) | Branch-and-Bound / Surgery | Decomposition and recombination |
| Family V (Synthetic) | Augmentation Methods | Extend problem to richer structure |
| Family VI (Forbidden) | Lower Bound / Barrier | Categorical impossibility |
| Family VII (Singular) | Definite Hardness | Proven NP-hard, PSPACE-hard, etc. |
| Family VIII (Horizon) | Undecidability / Unknown | Epistemic limit (undecidable or open) |
| Stiffness Subtree (7a-7d) | Refinement Cascade | Detailed technique selection |
| Node 17 (Lock) | Final Classification | Termination with definite answer |
| Metatheorem Selection | Algorithm Selection | Choosing technique from portfolio |
| Slot (Family, Stratum) | Algorithm Class | Unique strategy identifier |

---

## The Eight Families: Complexity Class Mapping

### Family I: The Stable (Polynomial-Time Solvable)

**Hypostructure:** Certificate chain of $K^+$ (immediate satisfaction).

**Complexity Equivalent:** Problems in **P** or with polynomial-time algorithms.

**Algorithmic Strategy:** Direct algorithms via a-priori estimates and induction.

| Problem Class | Example | Technique |
|--------------|---------|-----------|
| Sorting | Comparison sort | Divide-and-conquer |
| Graph connectivity | DFS/BFS | Linear-time traversal |
| Linear programming | Simplex/Interior point | Optimization |
| Matching | Bipartite matching | Augmenting paths |

**Certificate:** $K^+ = (\text{poly-time algorithm}, \text{correctness proof}, O(n^k))$

---

### Family II: The Relaxed (Subexponential / FPT / Scattering)

**Hypostructure:** Neutral certificates $K^\circ$ at boundary.

**Complexity Equivalent:** **FPT** (Fixed-Parameter Tractable), subexponential algorithms, or problems with "scattering" structure.

**Algorithmic Strategy:** Parameterized algorithms, dispersive methods, or branch with decay.

| Problem Class | Example | Technique |
|--------------|---------|-----------|
| FPT problems | Vertex Cover (param $k$) | Bounded search tree |
| Subexponential | Planar graph problems | Baker's technique |
| Scattering | Random instances | Probabilistic analysis |
| Decay structure | Sparse systems | Locality exploitation |

**Certificate:** $K^\circ = (\text{parameter } k, \text{FPT algorithm}, O(f(k) \cdot n^c))$

---

### Family III: The Gauged (Reduction-Based / Equivalent)

**Hypostructure:** Equivalence certificates $K^{\sim}$.

**Complexity Equivalent:** Problems solvable via **reduction** to known solved problems.

**Algorithmic Strategy:** Find equivalence, transform, solve, translate back.

| Problem Class | Example | Technique |
|--------------|---------|-----------|
| NP-intermediate | Graph isomorphism | Reduce to group theory |
| Linear algebra | System solving | Gaussian elimination |
| Parsing | CFG parsing | CYK via matrix multiply |
| Optimization | LP relaxation | Reduce to LP then round |

**Certificate:** $K^{\sim} = (\text{reduction } f, \text{solved problem } Q, f: P \leq_p Q)$

---

### Family IV: The Resurrected (Surgery / Branch-and-Bound)

**Hypostructure:** Re-entry certificates $K^{\mathrm{re}}$ via surgery.

**Complexity Equivalent:** Problems solvable by **decomposition**, **branch-and-bound**, or **divide-and-conquer** with careful recombination.

**Algorithmic Strategy:** Identify singularities (hard subproblems), solve separately, recombine.

| Problem Class | Example | Technique |
|--------------|---------|-----------|
| Optimization | TSP (small instances) | Branch-and-bound |
| Constraint solving | SAT | DPLL/CDCL with restarts |
| Game solving | Minimax | Alpha-beta pruning |
| Decomposable | Tree decomposition | Dynamic programming on tree |

**Certificate:** $K^{\mathrm{re}} = (\text{decomposition}, \text{subproblem solutions}, \text{recombination})$

---

### Family V: The Synthetic (Augmentation / Extension)

**Hypostructure:** Extension certificates $K^{\mathrm{ext}}$.

**Complexity Equivalent:** Problems requiring **auxiliary structure** not in original formulation.

**Algorithmic Strategy:** Extend to richer structure, solve in extended space, project back.

| Problem Class | Example | Technique |
|--------------|---------|-----------|
| Semidefinite | Max-Cut approximation | SDP relaxation |
| Algebraic | Polynomial identity testing | Extension to larger field |
| Randomized | BPP algorithms | Add random bits |
| Interactive | IP protocols | Add interaction rounds |

**Certificate:** $K^{\mathrm{ext}} = (\text{extension } \iota: P \hookrightarrow \tilde{P}, \text{solution in } \tilde{P}, \text{projection})$

---

### Family VI: The Forbidden (Barrier / Lower Bound)

**Hypostructure:** Blocked certificates $K^{\mathrm{blk}}$.

**Complexity Equivalent:** Problems with **proven lower bounds** or **barrier arguments**.

**Algorithmic Strategy:** Prove impossibility; use barriers to establish limits.

| Problem Class | Example | Technique |
|--------------|---------|-----------|
| Circuit lower bounds | Parity not in $\text{AC}^0$ | Switching lemma |
| Communication | Set disjointness | Information theory |
| Query complexity | Sorting lower bound | Adversary argument |
| Algebraic | Degree lower bounds | Polynomial method |

**Certificate:** $K^{\mathrm{blk}} = (\text{barrier type}, \text{lower bound}, \Omega(f(n)))$

---

### Family VII: The Singular (Proven Hard)

**Hypostructure:** Morphism certificates $K^{\mathrm{morph}}$ (bad pattern embeds).

**Complexity Equivalent:** Problems with **definitive hardness proofs** (NP-complete, PSPACE-complete, etc.).

**Algorithmic Strategy:** Accept hardness; use approximation, heuristics, or restricted inputs.

| Problem Class | Example | Technique |
|--------------|---------|-----------|
| NP-complete | 3-SAT | Approximation / heuristics |
| PSPACE-complete | QBF | Space-efficient algorithms |
| EXPTIME-complete | Generalized chess | Optimal play algorithms |
| #P-complete | Permanent | FPRAS (if exists) |

**Certificate:** $K^{\mathrm{morph}} = (\text{completeness proof}, C\text{-complete}, \text{reduction chain})$

---

### Family VIII: The Horizon (Undecidable / Unknown)

**Hypostructure:** Incompleteness certificates $K^{\mathrm{inc}}$.

**Complexity Equivalent:** **Undecidable** problems or problems at the epistemic horizon.

**Algorithmic Strategy:** Recognize limits; use semi-decision procedures or approximations.

| Problem Class | Example | Technique |
|--------------|---------|-----------|
| Undecidable | Halting problem | Enumerate halting programs |
| Unknown complexity | Graph isomorphism | Best known algorithms |
| Independence | Continuum hypothesis | Relative consistency |
| Epistemic | P vs NP | Research frontier |

**Certificate:** $K^{\mathrm{inc}} = (\text{undecidability proof}, \text{reduction from } H, \text{or OPEN})$

---

## The Twenty-One Strata: Algorithmic Phases

The 21 strata correspond to **decision points** in algorithm selection, analogous to the 21 nodes of the Structural Sieve. In complexity theory, these map to algorithmic phases or problem features:

### Primary Strata (1-7, 8-17)

| Stratum | Hypostructure Node | Complexity Equivalent | Feature Tested |
|---------|-------------------|----------------------|----------------|
| 1 | EnergyCheck ($D_E$) | Resource Bound | Is the problem finite/bounded? |
| 2 | ZenoCheck ($\text{Rec}_N$) | Termination | Does the algorithm terminate? |
| 3 | CompactCheck ($C_\mu$) | Concentration/Dispersion | Does structure concentrate or scatter? |
| 4 | ScaleCheck ($\text{SC}_\lambda$) | Scaling Behavior | How does complexity scale? |
| 5 | MonotoneCheck ($\text{Mon}$) | Monotonicity | Are there monotone properties? |
| 6 | CapacityCheck ($\text{Cap}_H$) | Capacity Bound | Are singular sets removable? |
| 7 | StiffnessCheck ($\text{Stiff}$) | Primary Structure | Is there rigid structure? |
| 8 | SurgeryCheck | Decomposability | Can the problem be surgically decomposed? |
| 9 | TopologyCheck | Topological Properties | Does topology constrain solutions? |
| 10 | CobordismCheck | Cobordism Structure | Are there cobordism relations? |
| 11 | DictionaryCheck | Translation | Can we translate to known problems? |
| 12 | AlgebraCheck | Algebraic Structure | Is there algebraic structure to exploit? |
| 13 | AnalyticCheck | Analytic Properties | Are analytic methods applicable? |
| 14 | GeometricCheck | Geometric Structure | Does geometry help? |
| 15 | CategoricalCheck | Categorical Properties | Are there functorial relations? |
| 16 | BarrierCheck | Barrier Arguments | Do barriers apply? |
| 17 | LockCheck | Final Classification | Definite conclusion |

### Stiffness Subtree (7a-7d): Refinement Cascade

| Stratum | Hypostructure Node | Complexity Equivalent | Refinement |
|---------|-------------------|----------------------|------------|
| 7a | BifurcationCheck | Branching Analysis | Detect and analyze branching structure |
| 7b | SymmetryCheck | Hidden Symmetry | Exploit symmetry for speedup |
| 7c | PhaseCheck | Phase Transition | Identify phase transition behavior |
| 7d | TunnelingCheck | Barrier Tunneling | Handle barriers via tunneling/continuation |

---

## Strategy Table: The 8x21 Matrix

The complete **Strategy Table** assigns an algorithm class to each (Family, Stratum) pair. Here is a representative excerpt:

### Families I-II (Easy/FPT) x Strata 1-7

| Family / Stratum | 1: Resource | 2: Termination | 3: Concentrate | 4: Scale | 5: Monotone | 6: Capacity | 7: Structure |
|-----------------|-------------|----------------|----------------|----------|-------------|-------------|--------------|
| I (Stable/P) | Bound check | Loop analysis | Divide-conquer | Poly bound | Greedy | Linear scan | Induction |
| II (Relaxed/FPT) | Param bound | FPT term | Kernelization | $f(k)n^c$ | Param greedy | FPT scan | Bounded search |

### Families III-IV (Reduction/Surgery) x Strata 1-7

| Family / Stratum | 1: Resource | 2: Termination | 3: Concentrate | 4: Scale | 5: Monotone | 6: Capacity | 7: Structure |
|-----------------|-------------|----------------|----------------|----------|-------------|-------------|--------------|
| III (Gauged) | Reduce bound | Reduce term | Reduce struct | Scale map | Reduce mono | Reduce cap | Equivalence |
| IV (Resurrected) | Branch bound | B&B term | Decompose | Branch scale | B&B prune | Subproblem | Surgery |

### Families V-VI (Extension/Barrier) x Strata 1-7

| Family / Stratum | 1: Resource | 2: Termination | 3: Concentrate | 4: Scale | 5: Monotone | 6: Capacity | 7: Structure |
|-----------------|-------------|----------------|----------------|----------|-------------|-------------|--------------|
| V (Synthetic) | Extend bound | SDP term | Relax | Lift scale | Extend mono | Relax cap | Augment |
| VI (Forbidden) | Lower bound | Non-term | Spread | Lower scale | Anti-mono | Zero cap | Barrier |

### Families VII-VIII (Hard/Undecidable) x Strata 1-7

| Family / Stratum | 1: Resource | 2: Termination | 3: Concentrate | 4: Scale | 5: Monotone | 6: Capacity | 7: Structure |
|-----------------|-------------|----------------|----------------|----------|-------------|-------------|--------------|
| VII (Singular) | Exp bound | Exp term | Hard core | Exp scale | Hardness | Hard cap | Complete |
| VIII (Horizon) | Unbound | Undecidable | Unknown | Unbounded | Unknown | Infinite | Undecidable |

---

## Proof Sketch

### Step 1: Problem Classification via Feature Extraction

**Claim:** Every computational problem admits a unique (Family, Stratum) classification.

**Procedure:**

1. **Extract Feature Vector:** Compute the signature $\sigma(P) = (f_1, f_2, \ldots, f_{21})$ where each $f_i$ encodes the problem's behavior at stratum $i$.

2. **Determine Family:** The family is determined by the "dominant" feature:
   $$\text{Family}(P) = \max_{i \in \{1, \ldots, 21\}} \text{type}(f_i)$$
   where types are ordered: $K^+ < K^\circ < K^{\sim} < K^{\mathrm{re}} < K^{\mathrm{ext}} < K^{\mathrm{blk}} < K^{\mathrm{morph}} < K^{\mathrm{inc}}$

3. **Determine Stratum:** The stratum is the first position where the dominant type appears:
   $$\text{Stratum}(P) = \min\{i : \text{type}(f_i) = \text{Family}(P)\}$$

**Certificate:** $K_{\text{classify}} = (\sigma(P), \text{Family}(P), \text{Stratum}(P))$

---

### Step 2: Strategy Uniqueness via Completeness

**Claim:** The (Family, Stratum) pair uniquely determines the algorithmic strategy.

**Proof:**

1. **Completeness:** By construction, the 8 families partition all problem types by their structural obstruction level. The 21 strata partition the decision phases. Together, they cover all possibilities.

2. **Disjointness:** Each (Family, Stratum) pair corresponds to a distinct structural configuration. A problem cannot simultaneously be in Family I (polynomial) and Family VII (NP-complete).

3. **Strategy Assignment:** For each of the 168 slots, there is a canonical algorithm class:
   $$\mathcal{S}: \{1, \ldots, 8\} \times \{1, \ldots, 21\} \to \text{AlgorithmClasses}$$

**Certificate:** $K_{\text{unique}} = (\text{(Family, Stratum)}, \mathcal{S}(\text{Family}, \text{Stratum}))$

---

### Step 3: Metatheorem Selection

**Claim:** The slot determines which meta-algorithms apply.

**Correspondence:**

| Family | Applicable Meta-Algorithms |
|--------|---------------------------|
| I (Stable) | Polynomial-time paradigms (greedy, DP, divide-conquer) |
| II (Relaxed) | FPT toolkit (kernelization, bounded search trees, color-coding) |
| III (Gauged) | Reduction framework (Karp reductions, Turing reductions) |
| IV (Resurrected) | Branch-and-bound, DPLL/CDCL, decomposition methods |
| V (Synthetic) | SDP relaxation, LP relaxation, randomization |
| VI (Forbidden) | Lower bound techniques (adversary, information-theoretic) |
| VII (Singular) | Approximation algorithms, heuristics, SAT solvers |
| VIII (Horizon) | Semi-decision procedures, enumeration, oracles |

---

### Step 4: Connections to Algorithm Selection and AutoML

**Algorithm Selection Problem (ASP):**

Given:
- Instance space $\mathcal{I}$
- Algorithm portfolio $\mathcal{A} = \{A_1, \ldots, A_m\}$
- Performance metric $p: \mathcal{I} \times \mathcal{A} \to \mathbb{R}$

Find: Selector $s: \mathcal{I} \to \mathcal{A}$ maximizing expected performance.

**LOCK-Periodic as ASP:**

The 8x21 Strategy Table is a **solved instance** of the Algorithm Selection Problem:
- $\mathcal{I}$ = all computational problems
- Feature extraction = computing (Family, Stratum)
- $\mathcal{A}$ = the 168 algorithm classes in the Strategy Table
- $s$ = lookup in the 8x21 matrix

**Meta-Learning Connection:**

Algorithm selection via meta-learning uses:
1. **Features:** Problem characteristics (size, structure, density, etc.)
2. **Meta-model:** Trained to predict best algorithm from features
3. **Portfolio:** Collection of algorithms with complementary strengths

The Strategy Table is the **oracle meta-model** that perfectly predicts algorithm class from structural features.

---

## Certificate Construction

**Periodic Classification Certificate:**

```
K_Periodic = {
  mode: "Strategy_Table_Classification",
  mechanism: "Feature_Extraction_Lookup",

  problem: {
    name: P,
    description: "computational problem instance"
  },

  feature_extraction: {
    signature: [f_1, f_2, ..., f_21],
    dominant_type: K_max,
    first_occurrence: stratum_i
  },

  classification: {
    family: F in {I, II, ..., VIII},
    stratum: S in {1, 2, ..., 17, 7a, 7b, 7c, 7d},
    slot: (F, S)
  },

  strategy_assignment: {
    algorithm_class: A[F, S],
    meta_algorithms: [MA_1, ..., MA_k],
    expected_complexity: O(f(n))
  },

  decidability: {
    classification_time: polynomial,
    strategy_lookup: O(1),
    guaranteed_applicability: true
  }
}
```

**Algorithm Portfolio Certificate:**

```
K_Portfolio = {
  mode: "Algorithm_Portfolio",
  mechanism: "Strategy_Table",

  portfolio: {
    families: 8,
    strata: 21,
    total_slots: 168,
    algorithms_per_slot: >= 1
  },

  coverage: {
    completeness: "all problems covered",
    disjointness: "slots mutually exclusive",
    decidability: "classification computable"
  },

  performance: {
    per_slot_guarantee: "applicable algorithm exists",
    optimality: "slot-optimal algorithm selected"
  }
}
```

---

## Connections to Algorithm Selection

### 1. Rice's Framework (1976)

**Rice's Algorithm Selection Problem:**

Given instances with features, select the best algorithm from a portfolio.

**Connection:** The 8x21 Strategy Table is a **feature-based selector**:
- Features = (Family, Stratum)
- Selection = matrix lookup
- Optimality = by construction

### 2. SATzilla and Algorithm Portfolios

**SATzilla (Xu et al., 2008):** Portfolio-based SAT solver that selects algorithms based on instance features.

**Connection:** The Strategy Table generalizes this:
- Structural features determine Family (problem type)
- Detailed features determine Stratum (technique selection)
- The 168 slots correspond to specialized solvers

### 3. AutoML and Meta-Learning

**AutoML:** Automatically select and configure ML algorithms.

**Connection:** The Strategy Table is the **theoretical foundation**:
- Problem type (Family) determines algorithm family
- Problem phase (Stratum) determines configuration
- The table provides the **oracle** that AutoML systems approximate

### 4. No Free Lunch and Specialization

**No Free Lunch Theorem:** No algorithm is best on all problems.

**Connection:** The Strategy Table **organizes specialization**:
- Each slot has its optimal algorithm class
- Universal algorithms (Family I) work when structure permits
- Specialized algorithms (Families II-VIII) when structure demands

---

## Quantitative Bounds

| Property | Bound |
|----------|-------|
| Number of Families | 8 (complete classification) |
| Number of Strata | 21 (17 primary + 4 subsidiary) |
| Total Slots | 168 = 8 x 21 |
| Classification Time | $O(\text{poly}(n))$ for feature extraction |
| Lookup Time | $O(1)$ for strategy selection |
| Completeness | Every problem maps to exactly one slot |
| Soundness | Strategy guaranteed applicable to its slot |

---

## Algorithmic Applications

### 1. Problem Triage

**Application:** Quickly classify a new problem to determine approach.

**Procedure:**
1. Extract features at each of 21 strata
2. Identify dominant feature type (Family)
3. Identify first occurrence (Stratum)
4. Look up strategy in table

### 2. Algorithm Portfolio Construction

**Application:** Build a portfolio covering all problem types.

**Procedure:**
1. For each of 168 slots, include one algorithm
2. Ensure coverage (at least one algorithm per slot)
3. Optimize for expected performance across distribution

### 3. Automated Theorem Proving

**Application:** Select proof strategy based on theorem classification.

**Procedure:**
1. Classify the theorem by structure (Family)
2. Identify the proof phase (Stratum)
3. Apply the corresponding proof technique

### 4. Compiler Optimization

**Application:** Select optimization strategy based on code structure.

**Procedure:**
1. Analyze code features (loop structure, data dependencies)
2. Classify into (Family, Stratum)
3. Apply corresponding optimization technique

---

## Summary

The LOCK-Periodic metatheorem, translated to complexity theory, establishes:

1. **Strategy Tables:** The 8x21 classification matrix provides a complete mapping from problem types to solution strategies. Every computational problem has a unique (Family, Stratum) classification.

2. **Family Determines Class:** The 8 families correspond to fundamental complexity classes:
   - I: Polynomial (P)
   - II: Parameterized (FPT)
   - III: Reduction-based (equivalence)
   - IV: Decomposition (branch-and-bound)
   - V: Extension (SDP, randomization)
   - VI: Barrier (lower bounds)
   - VII: Complete (NP-hard, etc.)
   - VIII: Horizon (undecidable)

3. **Stratum Determines Phase:** The 21 strata encode decision points where algorithmic choices are made, from resource bounds (Stratum 1) to final classification (Stratum 17).

4. **Unique Strategy:** The (Family, Stratum) pair uniquely determines the applicable algorithm class. This is the complexity-theoretic analog of the chemical periodic table: position determines properties.

5. **Algorithm Selection:** The Strategy Table solves the algorithm selection problem theoretically. Practical systems (SATzilla, AutoML) approximate this oracle.

**The Complexity-Theoretic Insight:**

The LOCK-Periodic theorem reveals that **problem classification is algorithm selection**: by classifying a problem into the 8x21 matrix, we automatically determine the correct algorithmic approach. This transforms algorithm selection from an empirical art into a systematic science.

The 168 slots of the Strategy Table constitute a **complete taxonomy** of computational problems, with each slot specifying:
- Which algorithmic paradigm applies
- Which complexity bounds hold
- Which meta-algorithms are available
- Which related problems share structure

This is the realization of Hilbert's program for computation: a systematic classification of problems where location determines method.

---

## Literature

**Algorithm Selection:**
- Rice, J. R. (1976). "The Algorithm Selection Problem." Advances in Computers. *Foundational framework.*
- Leyton-Brown, K. et al. (2003). "Boosting as a Metaphor for Algorithm Design." CP. *Portfolio methods.*
- Xu, L. et al. (2008). "SATzilla: Portfolio-based Algorithm Selection for SAT." JAIR. *Practical ASP.*
- Kotthoff, L. (2016). "Algorithm Selection for Combinatorial Search Problems: A Survey." AI Magazine. *Comprehensive survey.*

**Meta-Learning and AutoML:**
- Brazdil, P. et al. (2008). *Metalearning: Applications to Data Mining.* Springer. *Meta-learning foundations.*
- Hutter, F. et al. (2019). *Automated Machine Learning.* Springer. *AutoML handbook.*
- Vanschoren, J. (2019). "Meta-Learning." *AutoML Book.* *Meta-learning chapter.*

**Complexity Classification:**
- Papadimitriou, C. H. (1994). *Computational Complexity.* Addison-Wesley. *Complexity classes.*
- Arora, S. & Barak, B. (2009). *Computational Complexity: A Modern Approach.* Cambridge. *Modern treatment.*
- Downey, R. G. & Fellows, M. R. (2013). *Fundamentals of Parameterized Complexity.* Springer. *FPT classification.*

**Algorithm Design Paradigms:**
- Cormen, T. H. et al. (2009). *Introduction to Algorithms.* MIT Press. *Algorithmic paradigms.*
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design.* Pearson. *Design techniques.*
- Williamson, D. P. & Shmoys, D. B. (2011). *The Design of Approximation Algorithms.* Cambridge. *Approximation.*

**No Free Lunch:**
- Wolpert, D. H. & Macready, W. G. (1997). "No Free Lunch Theorems for Optimization." IEEE Trans. Evol. Comp. *NFL theorems.*
- Igel, C. & Toussaint, M. (2005). "A No-Free-Lunch Theorem for Non-Uniform Distributions." J. Math. Model. Algorithms. *NFL extensions.*

**SAT Solving and Portfolios:**
- Gomes, C. P. & Selman, B. (2001). "Algorithm Portfolios." Artificial Intelligence. *Portfolio theory.*
- Biere, A. et al. (2009). *Handbook of Satisfiability.* IOS Press. *SAT solving.*
