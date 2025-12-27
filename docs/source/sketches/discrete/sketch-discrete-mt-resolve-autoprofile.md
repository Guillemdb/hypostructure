---
title: "RESOLVE-AutoProfile - Complexity Theory Translation"
---

# RESOLVE-AutoProfile: Automatic Algorithm Selection

## Overview

This document provides a complete complexity-theoretic translation of the RESOLVE-AutoProfile theorem (Automatic Profile Classification) from the hypostructure framework. The translation establishes a formal correspondence between the Framework's multi-mechanism OR-schema for profile classification and **algorithm selection** in combinatorial optimization, where a meta-algorithm chooses the best approach from a portfolio of solvers based on instance features.

**Original Theorem Reference:** {prf:ref}`mt-resolve-auto-profile`

**Core Insight:** The Structural Sieve automatically selects among multiple classification mechanisms (CC+Rigidity, Attractor+Morse, Tame+LS, Lock/Hom-Exclusion) based on which soft interfaces are available. This mirrors portfolio solving and automatic algorithm configuration: given a problem instance, the meta-algorithm selects the most appropriate solver from a portfolio without user intervention.

---

## Hypostructure Context

The RESOLVE-AutoProfile theorem states that for any Hypostructure $\mathcal{H} = (\mathcal{X}, \Phi, \mathfrak{D}, G)$ satisfying the Automation Guarantee, the Profile Classification Trichotomy is **automatically computed** by the Sieve via one of four mechanisms:

**Mechanism A: CC+Rigidity** (Best for NLS, NLW, dispersive PDEs)
- Uses concentration-compactness and rigidity arguments
- Soft conditions: $K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{Mon}_\phi}^+ \wedge K_{\mathrm{Rep}_K}^+$

**Mechanism B: Attractor+Morse** (Best for reaction-diffusion, MCF)
- Uses global attractor theory and Morse decomposition
- Soft conditions: $K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{TB}_\pi}^+$

**Mechanism C: Tame+LS** (Best for algebraic/polynomial systems)
- Uses o-minimal definability and cell decomposition
- Soft conditions: $K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{TB}_O}^+$

**Mechanism D: Lock/Hom-Exclusion** (Best for categorical systems)
- Uses categorical obstruction for regularity
- Soft conditions: $K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$

**Unified Output:** All mechanisms produce the same certificate type $K_{\mathrm{prof}}^+$, with a route tag indicating provenance. Downstream theorems depend only on the certificate, never on which mechanism produced it.

---

## Complexity Theory Statement

**Theorem (Automatic Algorithm Selection).**
Let $\mathcal{P}$ be a problem class with:
- Instance space $I$
- Algorithm portfolio $\mathcal{A} = \{A_1, A_2, \ldots, A_m\}$
- Feature extraction function $\phi: I \to \mathbb{R}^d$
- Performance metric $c: I \times \mathcal{A} \to \mathbb{R}_{\geq 0}$ (runtime, solution quality)

A **portfolio solver** $\Pi$ with selection function $S: \mathbb{R}^d \to \mathcal{A}$ automatically chooses the best algorithm:

$$\Pi(x) = S(\phi(x))(x)$$

**Goal:** Minimize expected cost over instance distribution $\mathcal{D}$:

$$\min_S \mathbb{E}_{x \sim \mathcal{D}}[c(x, S(\phi(x)))]$$

**Guarantee (Oracle Bound):** For portfolio $\mathcal{A}$ and selector $S$:

$$\mathbb{E}[c(x, \Pi(x))] \leq \min_{A \in \mathcal{A}} \mathbb{E}[c(x, A)] + \epsilon_{\text{selection}}$$

where $\epsilon_{\text{selection}}$ is the selection overhead (feature extraction + dispatch).

**Complexity:**
- **Feature extraction:** $O(n)$ to $O(n^2)$ depending on features
- **Selection:** $O(m \cdot d)$ for linear selector, $O(\log m)$ for decision tree
- **Dispatch overhead:** Constant per instance

---

## Terminology Translation Table

| Hypostructure Concept | Algorithm Selection Equivalent | Formal Correspondence |
|-----------------------|-------------------------------|----------------------|
| Profile classification | Problem solving | The computational task |
| Mechanism A/B/C/D | Solver $A_1, A_2, \ldots, A_m$ | Algorithm portfolio |
| Soft interfaces $K_{\mathcal{P}}^+$ | Instance features $\phi(x)$ | Structural properties enabling algorithms |
| Dispatcher logic | Selector $S(\phi(x))$ | Chooses algorithm from portfolio |
| Route tag | Algorithm ID | Records which solver was used |
| Unified certificate $K_{\mathrm{prof}}^+$ | Solution certificate | Output independent of solver |
| Downstream independence | Certificate abstraction | Consumers see only solution, not solver |
| Automation Guarantee | Completeness | At least one algorithm applies |
| OR-schema | Portfolio disjunction | $A_1 \lor A_2 \lor \cdots \lor A_m$ |
| Profile library $\mathcal{L}_T$ | Solution cache | Known solutions for lookup |
| Tame family $\mathcal{F}_T$ | Structured instance class | Tractable subproblem |
| Wildness witness | Hardness certificate | Evidence of intractability |
| Soft-to-Backend Compilation | Algorithm preprocessing | Derives internal data structures |

---

## Algorithm Portfolio Framework

### The Algorithm Selection Problem

**Definition (Algorithm Selection Problem - ASP).**
Given:
1. **Instance space** $I$: The set of all possible problem instances
2. **Algorithm set** $\mathcal{A} = \{A_1, \ldots, A_m\}$: Available solvers
3. **Performance measure** $c: I \times \mathcal{A} \to \mathbb{R}$: Cost (runtime, quality)
4. **Instance distribution** $\mathcal{D}$: Probability over instances

Find selector $S^*: I \to \mathcal{A}$ minimizing expected cost:

$$S^* = \arg\min_S \mathbb{E}_{x \sim \mathcal{D}}[c(x, S(x))]$$

**Definition (Virtual Best Solver - VBS).**
The oracle selector choosing the best algorithm per instance:

$$\text{VBS}(x) = \arg\min_{A \in \mathcal{A}} c(x, A)$$

The VBS performance is the lower bound:

$$c_{\text{VBS}} = \mathbb{E}_{x \sim \mathcal{D}}[\min_{A \in \mathcal{A}} c(x, A)]$$

**Definition (Single Best Solver - SBS).**
The best single algorithm across all instances:

$$\text{SBS} = \arg\min_A \mathbb{E}_{x \sim \mathcal{D}}[c(x, A)]$$

**Performance Gap:** The selection benefit is:

$$\Delta = c_{\text{SBS}} - c_{\text{VBS}} \geq 0$$

When $\Delta$ is large, algorithm selection provides significant value.

### Mechanism-Solver Correspondence

| Hypostructure Mechanism | SAT Portfolio Analog | Characteristic |
|------------------------|---------------------|----------------|
| **A: CC+Rigidity** | CDCL solver (MiniSat) | Conflict-driven, good for structured |
| **B: Attractor+Morse** | Local search (WalkSAT) | Gradient-based, good for random |
| **C: Tame+LS** | Algebraic solver (PolyBoRi) | Exploits polynomial structure |
| **D: Lock/Hom-Exclusion** | Preprocessing (SatELite) | Eliminates subproblems |

---

## Proof Sketch

### Setup: The Portfolio Correspondence

We establish the correspondence:

| Hypostructure | Algorithm Selection |
|---------------|---------------------|
| Hypostructure $\mathcal{H}$ | Problem instance $x \in I$ |
| Soft interfaces | Feature vector $\phi(x)$ |
| Mechanism $M \in \{A, B, C, D\}$ | Solver $A \in \mathcal{A}$ |
| Dispatcher | Selector $S$ |
| Certificate $K_{\mathrm{prof}}^+$ | Solution with proof |

**Selection Function:**

$$S: \text{SoftInterfaces} \to \{\text{MechA}, \text{MechB}, \text{MechC}, \text{MechD}\}$$

### Step 1: Feature Extraction (Soft Interface Evaluation)

**Claim.** The soft interfaces form a feature vector characterizing the problem instance.

**Construction (Feature Extraction):**

For hypostructure $\mathcal{H}$, extract features:

```
phi(H) = (
    has_monotonicity : K_Mon^+ present -> 1, else 0
    has_finite_top   : K_TB_pi^+ present -> 1, else 0
    has_o_minimal    : K_TB_O^+ present -> 1, else 0
    has_lock_block   : K_Cat_Hom^blk present -> 1, else 0
    energy_bound     : value of D_E
    scaling_exp      : value of alpha from SC_lambda
    lojasiewicz_exp  : value of theta from LS_sigma
)
```

**Correspondence:**

| Soft Interface | Feature | Solver Preference |
|----------------|---------|-------------------|
| $K_{\mathrm{Mon}_\phi}^+$ | Monotonicity | CDCL (propagation) |
| $K_{\mathrm{TB}_\pi}^+$ | Finite topology | Local search (basin structure) |
| $K_{\mathrm{TB}_O}^+$ | O-minimal | Algebraic solver (cell decomposition) |
| $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | Categorical | Preprocessing (eliminate) |

**SAT Feature Analog:**

In SAT portfolio solvers, instance features include:
- Clause-to-variable ratio
- Variable degree distribution
- Backbone size estimates
- Community structure (modularity)
- Horn clause fraction

These features predict which solver will perform best, analogous to soft interfaces predicting which mechanism applies.

### Step 2: Dispatcher Logic (Algorithm Selection)

**Claim.** The Sieve's dispatcher implements a decision-tree-style selector.

**Dispatcher Algorithm:**

```
function DISPATCH(soft_interfaces):
    if has_monotonicity(soft_interfaces) and has_rep_K(soft_interfaces):
        return MechA_CC_Rigidity
    elif has_finite_topology(soft_interfaces):
        return MechB_Attractor_Morse
    elif has_o_minimal(soft_interfaces):
        return MechC_Tame_LS
    elif has_lock_block(soft_interfaces):
        return MechD_Lock_Exclusion
    else:
        return FAIL_inconclusive
```

**Decision Tree Formulation:**

The dispatcher is a depth-4 decision tree:

```
           [has_Mon + has_Rep?]
              /          \
            YES           NO
            /              \
        MechA        [has_TB_pi?]
                      /       \
                    YES        NO
                    /            \
                MechB      [has_TB_O?]
                            /      \
                          YES       NO
                          /          \
                      MechC    [has_Lock?]
                                /      \
                              YES       NO
                              /          \
                          MechD       FAIL
```

**Complexity:** Selection takes $O(d)$ where $d$ is the number of soft interfaces (constant, $d \leq 10$).

**Portfolio Solver Analog:**

| Selection Method | Hypostructure | Complexity |
|------------------|---------------|------------|
| Static schedule | Fixed mechanism order | $O(1)$ selection |
| Decision tree | Dispatcher logic | $O(d)$ selection |
| Regression model | Learned feature weights | $O(d)$ selection |
| Neural selector | Deep feature mapping | $O(d^2)$ selection |

### Step 3: Mechanism Independence (Solver Orthogonality)

**Claim.** The mechanisms operate independently; no mechanism depends on another's output.

**Parallel Alternatives Structure:**

The OR-schema has structure:

$$\text{Goal} \Leftarrow (\text{MechA} \lor \text{MechB} \lor \text{MechC} \lor \text{MechD})$$

Each mechanism $M_i$ is a **complete solver** for instances where its preconditions hold:

| Mechanism | Complete When | Independent Proof |
|-----------|---------------|-------------------|
| A | $K_{\mathrm{Mon}_\phi}^+ \wedge K_{\mathrm{Rep}_K}^+$ | Lions-CC + Kenig-Merle |
| B | $K_{\mathrm{TB}_\pi}^+$ | Temam-Hale + Conley |
| C | $K_{\mathrm{TB}_O}^+$ | van den Dries + Lojasiewicz |
| D | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | Categorical obstruction |

**No Circular Dependencies:**

Mechanism proofs are modular:
- Each mechanism uses only its soft interfaces
- No mechanism calls another mechanism
- Backend permits are derived independently within each mechanism

**Portfolio Analog:** In SAT portfolios, solvers are independent processes:
- CDCL and local search use different data structures
- No shared state during solving
- Only the first solution is accepted

### Step 4: Dispatcher Completeness (Coverage Guarantee)

**Claim.** For hypostructures satisfying the Automation Guarantee, at least one mechanism applies.

**Automation Guarantee (Definition):**

A type $T$ satisfies the Automation Guarantee if:

$$K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \Rightarrow \exists M \in \{A,B,C,D\}.\ M \text{ applies}$$

**Proof (Completeness via Type Classification):**

For "good" types (parabolic, dispersive, hyperbolic):

1. **Parabolic (heat, Ricci, MCF):** Finite topology → Mechanism B applies
2. **Dispersive (NLS, NLW):** Monotonicity (virial/Morawetz) → Mechanism A applies
3. **Algebraic (polynomial ODEs):** O-minimal definable → Mechanism C applies
4. **Categorical (homotopy types):** Lock obstruction → Mechanism D applies

**Coverage Theorem:**

$$\text{GoodType}(T) \Rightarrow \text{Coverage}(\{A,B,C,D\})$$

**Portfolio Completeness Analog:**

In SAT portfolio solving, completeness means:
- For every satisfiable instance, at least one solver finds a solution
- For every unsatisfiable instance, at least one solver proves UNSAT
- The portfolio is **complementary**: solvers cover each other's weaknesses

### Step 5: Downstream Independence (Certificate Abstraction)

**Claim.** All downstream theorems consume only $K_{\mathrm{prof}}^+$, never the mechanism tag.

**Certificate Interface:**

```
type ProfileCertificate = {
    profile    : V,
    family     : L_T or F_T,
    route_tag  : CC-Rig | Attr-Morse | Tame-LS | Lock-Excl,
    class_data : mechanism-specific data
}

// Downstream consumers pattern-match on certificate type, not provenance
function use_certificate(cert: ProfileCertificate):
    match cert.family with
    | L_T -> apply_library_surgery(cert.profile)
    | F_T -> apply_tame_surgery(cert.profile, cert.class_data)
```

**Abstraction Barrier:**

The route tag is for **logging/debugging only**. Semantic content is:
- Profile $V$: The classified object
- Family $\mathcal{L}_T$ or $\mathcal{F}_T$: Library or tame classification
- Classification data: Parameters for surgery

**Portfolio Analog:**

In SAT solving, the output is a satisfying assignment or UNSAT proof:
- Downstream uses (e.g., model checking, planning) consume only the assignment
- Which solver produced it is irrelevant
- Proof format is standardized (DRAT, LRAT) regardless of solver

---

## Connections to SAT Portfolio Solving

### 1. SATzilla (Xu et al. 2008)

**Architecture:**
1. **Feature extraction:** 48 syntactic features (clause/variable statistics)
2. **Selector:** Ridge regression predicting runtime per solver
3. **Portfolio:** 7 CDCL and local search solvers

**Mechanism Correspondence:**

| SATzilla Component | Hypostructure Component |
|--------------------|------------------------|
| Feature extraction | Soft interface evaluation |
| Runtime predictor | Mechanism applicability test |
| Solver portfolio | $\{$MechA, MechB, MechC, MechD$\}$ |
| Winner-take-all | First-success dispatcher |

**Performance:**
- SATzilla outperforms every individual solver
- Selection overhead: ~1 second (negligible for hard instances)
- Gap closed: $\sim$90% of VBS performance

### 2. Algorithm Selection via Machine Learning

**General Framework (Rice 1976):**

```
         +---------+     +----------+     +---------+
Instance |         |     |          |     |         | Solution
-------->| Feature |---->| Selector |---->| Solver  |-------->
         | Extract |     | S(phi)   |     | A_i     |
         +---------+     +----------+     +---------+
```

**Learning Approaches:**

| Method | Hypostructure Analog | Complexity |
|--------|---------------------|------------|
| Decision tree | Dispatcher logic | $O(\log m)$ selection |
| Random forest | Ensemble of dispatchers | $O(k \log m)$ selection |
| Neural network | Learned soft interface | $O(d^2)$ selection |
| Pairwise ranking | Mechanism preference | $O(m^2)$ training |

### 3. Parallel Portfolio Solving

**Strategy:** Run all solvers in parallel, take first solution.

**Implementation:**

```
function PARALLEL_PORTFOLIO(instance x):
    for A in portfolio do in parallel:
        result_A <- A(x)
        if result_A.success:
            cancel_all()
            return result_A
```

**Hypostructure Analog:**

The dispatcher could be parallelized:
- Try all mechanisms concurrently
- Accept first successful certificate
- Cancel remaining mechanism evaluations

**Trade-off:**
- Serial: $O(\text{cheapest mechanism})$ on success
- Parallel: $O(\text{fastest mechanism})$ but higher resource use

---

## Connections to AutoML

### 1. Automatic Algorithm Configuration (ParamILS, SMAC)

**Problem:** Find optimal hyperparameters for an algorithm.

**Hypostructure Analog:** The soft interfaces act as "hyperparameters" selecting mechanism behavior:

| AutoML Hyperparameter | Hypostructure Parameter |
|----------------------|------------------------|
| Learning rate | Scaling exponent $\alpha$ |
| Regularization | Dissipation coefficient |
| Architecture choice | Mechanism selection |
| Ensemble method | OR-schema combination |

### 2. Neural Architecture Search (NAS)

**Problem:** Automatically design neural network architecture.

**Correspondence to Mechanism Selection:**

| NAS Component | Hypostructure Component |
|---------------|------------------------|
| Search space | $\{$MechA, MechB, MechC, MechD$\}$ |
| Performance estimator | Soft interface evaluator |
| Search strategy | Dispatcher logic |
| Weight sharing | Shared soft core |

**DARTS-style Differentiable Selection:**

Continuous relaxation of mechanism selection:

$$\text{Output} = \sum_{M \in \{A,B,C,D\}} \alpha_M \cdot M(\mathcal{H})$$

where $\alpha_M = \text{softmax}(\theta_M)$ are learned weights.

### 3. Meta-Learning for Algorithm Selection

**MAML-style Approach:**

Learn initial selector that quickly adapts to new problem types:

1. **Meta-training:** Train selector on diverse hypostructure types
2. **Adaptation:** Fine-tune on specific type with few examples
3. **Deployment:** Apply adapted selector to new instances

**Hypostructure Analog:**

The Automation Guarantee is a **meta-theorem**:
- Proven once for the framework
- Instantiated automatically for each type
- No per-type learning required (purely logic-based)

---

## Certificate Construction

The AutoProfile theorem produces explicit certificates:

**AutoProfile Certificate $K_{\mathrm{prof}}^+$:**

```
K_prof+ := (
    profile          : V (scaling limit),
    library_member   : V in L_T or V in F_T \ L_T,
    route_tag        : CC-Rig | Attr-Morse | Tame-LS | Lock-Excl,
    classification   : Case 1 (library) | Case 2 (tame) | Case 3 (wild/inc),
    mechanism_data   : {
        A: { concentration: K_CC, rigidity: K_rig },
        B: { attractor: K_attr, morse: K_morse },
        C: { cell_id: i, stratum_dim: d_i },
        D: { lock_cert: K_lock, regularity: vacuous }
    }[route_tag]
)
```

**Dispatch Certificate:**

```
K_dispatch := (
    soft_interfaces  : { D_E, C_mu, SC_lambda, LS_sigma, ... },
    mechanisms_tried : [A?, B?, C?, D?],
    selected         : route_tag,
    selection_time   : O(d) steps
)
```

**Verification:**

1. Check soft interfaces are valid
2. Check mechanism preconditions match route tag
3. Verify profile classification within mechanism
4. Confirm downstream independence (route tag not used semantically)

---

## Quantitative Bounds

### Selection Overhead

For portfolio with $m$ mechanisms and $d$ soft interfaces:

| Operation | Complexity |
|-----------|------------|
| Feature extraction | $O(d)$ |
| Decision tree selection | $O(\log m)$ |
| Mechanism dispatch | $O(1)$ |
| Total selection overhead | $O(d + \log m)$ |

For the hypostructure framework: $m = 4$, $d \approx 10$, so overhead is $O(1)$.

### Mechanism Runtime

| Mechanism | Runtime Bound | Bottleneck |
|-----------|---------------|------------|
| A: CC+Rigidity | $O(n^2)$ | Profile decomposition |
| B: Attractor+Morse | $O(n^3)$ | Morse decomposition |
| C: Tame+LS | $O(n^{O(d)})$ | Cell decomposition |
| D: Lock/Hom-Exclusion | $O(n)$ | Lock check (categorical) |

### Portfolio Speedup

**Theorem (Portfolio Speedup).**
Let $T_i(x)$ be the runtime of mechanism $i$ on instance $x$. The portfolio achieves:

$$T_{\text{portfolio}}(x) \leq \min_i T_i(x) + O(d)$$

For instances where mechanisms have complementary performance:

$$\mathbb{E}[T_{\text{portfolio}}] \leq \min_i \mathbb{E}[T_i] \cdot (1 - \delta)$$

where $\delta > 0$ depends on portfolio diversity.

---

## Worked Example: Profile Classification of NLS

**Problem:** Classify blow-up profiles for energy-critical NLS:

$$i\partial_t u + \Delta u = |u|^{4/(n-2)} u$$

**Soft Interfaces Available:**
- $K_{D_E}^+$: Energy bound $E(u) \leq E_c$
- $K_{C_\mu}^+$: Concentration-compactness applies
- $K_{\mathrm{SC}_\lambda}^+$: Critical scaling $\alpha = 2/n$
- $K_{\mathrm{LS}_\sigma}^+$: Lojasiewicz for soliton manifold
- $K_{\mathrm{Mon}_\phi}^+$: Morawetz monotonicity
- $K_{\mathrm{Rep}_K}^+$: Representation by Kenig-Merle profiles

**Dispatcher Evaluation:**

```
DISPATCH(NLS_soft_interfaces):
    has_monotonicity(K_Mon+) = TRUE  (Morawetz)
    has_rep_K(K_Rep+)        = TRUE  (Kenig-Merle)
    -> SELECT MechA (CC+Rigidity)
```

**Mechanism A Execution:**

1. **Step A1:** Derive $K_{\mathrm{WP}}^+$ via template matching (Strichartz estimates)
2. **Step A2:** Profile decomposition via concentration-compactness
3. **Step A3:** Kenig-Merle minimal counterexample extraction
4. **Step A4:** Rigidity via Morawetz + Lojasiewicz closure
5. **Step A5:** Emit certificate

**Output Certificate:**

```
K_prof+ := (
    profile    : W (ground state soliton),
    family     : L_T (finite library),
    route_tag  : CC-Rig,
    class_data : {
        energy_threshold : E(W),
        scattering_below : TRUE,
        blowup_above     : TRUE (via Glassey)
    }
)
```

**Alternative (if monotonicity fails):**

If NLS has no Morawetz identity (e.g., non-radial, magnetic):
- Mechanism A fails at precondition check
- Try Mechanism B: Check for attractor → Fails (dispersive)
- Try Mechanism C: Check for o-minimal → May apply if polynomial
- Try Mechanism D: Check for Lock obstruction → May apply

---

## Advantages of the OR-Schema

### 1. Robustness via Redundancy

The multi-mechanism approach provides:
- **Fault tolerance:** If one mechanism fails, others may succeed
- **Coverage:** Different mechanisms handle different problem structures
- **Graceful degradation:** Fallback to simpler mechanisms

### 2. Extensibility

Adding new mechanisms is modular:
1. Define new soft interface requirements
2. Implement mechanism logic
3. Add to dispatcher (extend decision tree)
4. Downstream unchanged (certificate abstraction)

### 3. Interpretability

Route tags provide explanations:
- "Profile classified via CC+Rigidity" → PDE community understands
- "Profile classified via Tame+LS" → O-minimal community understands
- Mechanism selection justifies which theory applies

---

## Summary

The RESOLVE-AutoProfile theorem, translated to complexity theory, states:

**A meta-algorithm automatically selects the best classification mechanism from a portfolio based on structural features, achieving performance close to the virtual best solver.**

This principle:

1. **Abstracts mechanism choice:** Users need not know which mechanism applies
2. **Guarantees coverage:** The Automation Guarantee ensures completeness
3. **Preserves downstream independence:** Consumers see only the certificate
4. **Enables extensibility:** New mechanisms integrate without breaking existing code

The translation illuminates deep connections:

| Hypostructure | Algorithm Selection |
|---------------|---------------------|
| Soft interfaces | Instance features |
| Mechanism portfolio | Algorithm portfolio |
| Dispatcher logic | Selector function |
| OR-schema | Disjunctive completion |
| Unified certificate | Solution abstraction |
| Route tag | Algorithm provenance |

**Key Insight:** Just as SAT portfolio solvers outperform individual solvers by selecting the right algorithm per instance, the hypostructure framework achieves broad applicability by selecting the right mechanism per problem type. The **OR-schema** is the logical formulation of portfolio solving: success via ANY mechanism is success overall.

---

## Literature

1. **Xu, L., Hutter, F., Hoos, H., & Leyton-Brown, K. (2008).** "SATzilla: Portfolio-based Algorithm Selection for SAT." *JAIR.* *Foundational SAT portfolio work.*

2. **Rice, J. R. (1976).** "The Algorithm Selection Problem." *Advances in Computers.* *Original ASP formulation.*

3. **Kotthoff, L. (2016).** "Algorithm Selection for Combinatorial Search Problems: A Survey." *AI Magazine.* *Comprehensive ASP survey.*

4. **Hutter, F., Hoos, H., & Leyton-Brown, K. (2011).** "Sequential Model-based Optimization for General Algorithm Configuration." *LION.* *SMAC algorithm configuration.*

5. **Biere, A., Heule, M., van Maaren, H., & Walsh, T. (eds.) (2009).** *Handbook of Satisfiability.* IOS Press. *SAT solving reference.*

6. **Hoos, H. & Stutzle, T. (2005).** *Stochastic Local Search: Foundations and Applications.* Morgan Kaufmann. *Local search foundations.*

7. **Feurer, M., Klein, A., Eggensperger, K., Springenberg, J., Blum, M., & Hutter, F. (2015).** "Efficient and Robust Automated Machine Learning." *NeurIPS.* *Auto-sklearn.*

8. **Zoph, B. & Le, Q. (2017).** "Neural Architecture Search with Reinforcement Learning." *ICLR.* *NAS foundations.*

9. **Finn, C., Abbeel, P., & Levine, S. (2017).** "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." *ICML.* *MAML meta-learning.*

10. **Liu, H., Simonyan, K., & Yang, Y. (2019).** "DARTS: Differentiable Architecture Search." *ICLR.* *Differentiable selection.*

11. **Lions, P.-L. (1984).** "The Concentration-Compactness Principle." *Annales IHP.* *CC foundations (hypostructure source).*

12. **Kenig, C. E. & Merle, F. (2006).** "Global Well-Posedness, Scattering and Blow-Up." *Inventiones.* *Rigidity theorem (Mechanism A source).*

13. **Temam, R. (1997).** *Infinite-Dimensional Dynamical Systems in Mechanics and Physics.* Springer. *Attractor theory (Mechanism B source).*

14. **van den Dries, L. (1998).** *Tame Topology and O-minimal Structures.* Cambridge. *O-minimal theory (Mechanism C source).*
