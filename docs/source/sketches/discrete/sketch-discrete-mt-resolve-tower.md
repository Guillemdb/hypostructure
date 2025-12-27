---
title: "RESOLVE-Tower - Complexity Theory Translation"
---

# RESOLVE-Tower: Local-to-Global Lifting of Complexity Bounds

## Overview

This document provides a complete complexity-theoretic translation of the RESOLVE-Tower theorem (Soft Local Tower Globalization) from the hypostructure framework. The translation establishes a formal correspondence between local obstruction analysis lifting to global tower structure and **local-to-global principles** in complexity theory, where local consistency guarantees propagate to global solution existence.

**Original Theorem Reference:** {prf:ref}`mt-resolve-tower`

**Core Insight:** In tower hypostructures, local data at each scale determines global asymptotic behavior. The Lefschetz theorem in algebraic geometry provides the template: local cohomological information lifts to global topological structure. In complexity theory, this manifests as **lifting theorems** that connect local constraint satisfaction to global solution bounds.

---

## Hypostructure Context

The RESOLVE-Tower theorem addresses **tower hypostructures**---multiscale systems where:

1. **Scale Index:** $t \in \mathbb{N}$ or $t \in \mathbb{R}_+$ indexes resolution levels
2. **State Spaces:** $X_t$ is the state space at level $t$
3. **Transition Maps:** $S_{t \to s}: X_t \to X_s$ (for $s < t$) are scale-compatible
4. **Height Functional:** $\Phi(t)$ measures energy at level $t$
5. **Dissipation:** $\mathfrak{D}(t)$ tracks energy loss at level $t$

**Tower-Specific Interface Permits:**

| Permit | Name | Complexity Interpretation |
|--------|------|---------------------------|
| $C_\mu^{\mathrm{tower}}$ | SliceCompact | Local problem size bounded |
| $D_E^{\mathrm{tower}}$ | SubcritDissip | Weighted sum of local costs finite |
| $\mathrm{SC}_\lambda^{\mathrm{tower}}$ | ScaleCohere | Local contributions sum to global |
| $\mathrm{Rep}_K^{\mathrm{tower}}$ | LocalRecon | Global determined by local invariants |

**Theorem Conclusion:** Given certified permits, the tower admits:
1. Globally consistent asymptotic structure $X_\infty = \varprojlim X_t$
2. Asymptotic behavior completely determined by local invariants
3. No supercritical growth or uncontrolled accumulation

---

## Complexity Theory Statement

**Theorem (Lifting Theorem for Complexity Bounds).**
Let $\mathcal{P} = (V, C, w)$ be a hierarchical constraint satisfaction problem where:
- $V = \bigcup_{t=0}^T V_t$ is a layered variable set (tower structure)
- $C = \bigcup_t C_t$ is a layered constraint set
- $w: C \to \mathbb{R}_+$ assigns weights to constraints

The **lifting conditions** are:

1. **Local Boundedness ($C_\mu^{\mathrm{tower}}$):** For each level $t$, the local CSP $(V_t, C_t)$ has solution space of bounded size
2. **Subcritical Aggregation ($D_E^{\mathrm{tower}}$):** $\sum_t \alpha^t \cdot \text{cost}(t) < \infty$ for some $\alpha < 1$
3. **Scale Coherence ($\mathrm{SC}_\lambda^{\mathrm{tower}}$):** Local solutions extend coherently: $\text{cost}(t_2) - \text{cost}(t_1) = \sum_{u=t_1}^{t_2-1} L(u) + O(1)$
4. **Local Reconstruction ($\mathrm{Rep}_K^{\mathrm{tower}}$):** Global cost determined by local invariants: $\text{OPT} = F(\{I_\alpha\}_\alpha) + O(1)$

**Conclusion (Global Lifting):**

**(1)** The hierarchical CSP admits a **globally consistent solution**:
$$\sigma_\infty = \lim_{t \to \infty} \sigma_t \quad \text{where } \sigma_t \text{ solves } (V_{\leq t}, C_{\leq t})$$

**(2)** The **optimal global cost** is determined by local invariants:
$$\text{OPT} = F(\{I_\alpha(\infty)\}_\alpha) + O(1)$$

**(3)** **No complexity explosion:** Supercritical growth at any level contradicts the subcritical aggregation condition.

**Complexity:**
- **Local consistency checking:** $O(|C_t| \cdot d^k)$ per level (for arity-$k$ constraints, domain size $d$)
- **Global lifting:** $O(T \cdot \text{local})$ where $T$ is tower height
- **Verification:** Polynomial in certificate size

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Equivalent | Formal Correspondence |
|-----------------------|----------------------|----------------------|
| Tower hypostructure $\mathbb{H}$ | Hierarchical CSP | Layered constraint problem |
| Scale index $t$ | Refinement level | Resolution/hierarchy depth |
| State space $X_t$ | Local solution space | Assignments at level $t$ |
| Transition map $S_{t \to s}$ | Coarsening/projection | Local-to-global restriction |
| Height $\Phi(t)$ | Accumulated cost | Sum of constraint violations |
| Dissipation $\mathfrak{D}(t)$ | Local improvement | Cost reduction at level $t$ |
| $C_\mu^{\mathrm{tower}}$ (SliceCompact) | Bounded local CSP | $|X_t| \leq B$ per level |
| $D_E^{\mathrm{tower}}$ (SubcritDissip) | Summable weights | $\sum_t \alpha^t \cdot w_t < \infty$ |
| $\mathrm{SC}_\lambda^{\mathrm{tower}}$ (ScaleCohere) | Telescoping costs | Local contributions sum |
| $\mathrm{Rep}_K^{\mathrm{tower}}$ (LocalRecon) | Sheaf condition | Global from local data |
| $X_\infty = \varprojlim X_t$ | Global solution | Inverse limit / consistent extension |
| Supercritical growth | Complexity explosion | Exponential blowup |
| Frozen dynamics | Convergence | Fixed point reached |
| Local invariants $\{I_\alpha(t)\}$ | Local certificates | Witnesses for local consistency |
| Certificate $K_{\mathrm{Global}}^+$ | Global solution certificate | Proof of satisfiability |
| Profile concentration | Constraint propagation | Local decisions propagate |
| Defect inheritance | Error propagation | Local errors lift globally |

---

## Local-Global Principles in Complexity Theory

### The Sheaf-Theoretic Viewpoint

**Definition (Presheaf of Solutions).** A hierarchical CSP defines a presheaf:
$$\mathcal{F}: \text{Levels}^{\text{op}} \to \textbf{Set}$$
$$\mathcal{F}(t) = \{\sigma: V_t \to D \mid \sigma \text{ satisfies } C_t\}$$

with restriction maps:
$$\rho_{t,s}: \mathcal{F}(t) \to \mathcal{F}(s) \quad \text{for } s < t$$

**Definition (Sheaf Condition).** The presheaf is a **sheaf** if local solutions patch uniquely:
$$\mathcal{F}(\text{global}) = \lim_{\leftarrow} \mathcal{F}(t)$$

**Connection to RESOLVE-Tower:** The $\mathrm{Rep}_K^{\mathrm{tower}}$ permit ensures the sheaf condition: local invariants determine global structure.

### Constraint Propagation as Cohomology

**Observation:** Obstructions to lifting local solutions globally are measured by **sheaf cohomology**:
$$H^1(\mathcal{F}) = \text{obstruction to global section}$$

**Vanishing Theorem:** If $H^1(\mathcal{F}) = 0$, then every local section extends globally.

**RESOLVE-Tower Translation:** The tower permits ensure vanishing of obstructions:
- $C_\mu^{\mathrm{tower}}$: Local sections exist (bounded solution space)
- $D_E^{\mathrm{tower}}$: Obstructions decay (summable weights)
- $\mathrm{SC}_\lambda^{\mathrm{tower}}$: Gluing is consistent (scale coherence)
- $\mathrm{Rep}_K^{\mathrm{tower}}$: Global determined locally (sheaf condition)

---

## Proof Sketch

### Setup: The Lifting Correspondence

We establish the correspondence between hypostructure components and hierarchical CSPs:

| Hypostructure | Hierarchical CSP |
|---------------|------------------|
| State space $X_t$ | Local assignments $\sigma_t: V_t \to D$ |
| Transition $S_{t \to s}$ | Restriction $\sigma_t|_{V_s}$ |
| Height $\Phi(t)$ | Accumulated cost $\sum_{u \leq t} w_u \cdot \text{viol}(\sigma_u)$ |
| Dissipation $\mathfrak{D}(t)$ | Improvement $\Phi(t-1) - \Phi(t)$ |
| Limit $X_\infty$ | Global solution $\sigma: V \to D$ |

### Step 1: Local Existence (Compactness)

**Claim.** The $C_\mu^{\mathrm{tower}}$ permit guarantees local solutions exist at each level.

**Construction.** For each level $t$:
- The local CSP $(V_t, C_t)$ has bounded solution space: $|\mathcal{F}(t)| \leq B_t$
- Compactness modulo symmetries ensures non-empty fibers

**Complexity Interpretation:** Each local problem is tractable:
$$\text{Time}(\text{solve } t) = O(|C_t| \cdot d^{|V_t|}) \leq O(|C_t| \cdot B_t)$$

**Certificate:** For each $t$, produce local solution $\sigma_t$ and witness $K_t^+$.

### Step 2: Finite Total Cost (Dissipation Bound)

**Claim.** The $D_E^{\mathrm{tower}}$ permit bounds total cost across all levels.

**Proof.** By subcritical dissipation:
$$\sum_t w(t) \cdot \mathfrak{D}(t) < \infty \quad \text{where } w(t) \sim e^{-\alpha t}$$

**Implication for Complexity:**
$$\sum_t \text{cost}(t) = \sum_t \alpha^t \cdot \text{local\_cost}(t) < \infty$$

This bounds the **total work** across all levels:
$$\text{Total\_Work} = \sum_t \text{Time}(t) = O\left(\sum_t \alpha^t \cdot |C_t| \cdot B_t\right) < \infty$$

**Key Insight:** Exponential decay weight $\alpha^t$ absorbs potential exponential growth in problem size, ensuring convergent total complexity.

### Step 3: Consistent Extension (Scale Coherence)

**Claim.** The $\mathrm{SC}_\lambda^{\mathrm{tower}}$ permit ensures local solutions patch coherently.

**Proof.** By scale coherence:
$$\Phi(t_2) - \Phi(t_1) = \sum_{u=t_1}^{t_2-1} L(u) + O(1)$$

where each $L(u)$ is a local contribution determined by level $u$ data.

**Telescoping Property:**
$$\Phi(\infty) - \Phi(0) = \sum_{u=0}^{\infty} L(u) + O(1)$$

The sum converges absolutely: $|L(u)| \leq C \cdot \mathfrak{D}(u)$ and $\sum_u \mathfrak{D}(u) < \infty$.

**Complexity Interpretation:** The global optimum cost is:
$$\text{OPT} = \text{OPT}_0 + \sum_{u=0}^{T} \Delta_u + O(1)$$

where $\Delta_u$ is the local improvement at level $u$. The $O(1)$ error from boundary/interface terms is uniformly bounded.

### Step 4: Global Reconstruction (Local Determination)

**Claim.** The $\mathrm{Rep}_K^{\mathrm{tower}}$ permit ensures global behavior is determined by local invariants.

**Proof.** By soft local reconstruction:
$$\Phi(t) = F(\{I_\alpha(t)\}_\alpha) + O(1)$$

for local invariants $\{I_\alpha(t)\}$. Taking $t \to \infty$:
- Local invariants stabilize: $I_\alpha(t) \to I_\alpha(\infty)$
- Global cost is determined: $\Phi(\infty) = F(\{I_\alpha(\infty)\}_\alpha) + O(1)$

**Complexity Interpretation:** The global optimum is **computable from local data**:
$$\text{OPT} = F(\text{local\_invariants}) + O(1)$$

This is the **local-global principle**: understanding local structure suffices for global bounds.

### Step 5: Exclusion of Complexity Explosion

**Claim.** Supercritical growth at any level contradicts the permits.

**Proof by Contradiction.** Suppose supercritical growth at level $t_0$:
$$\Phi(t_0 + n) - \Phi(t_0) \gtrsim n^\gamma \quad \text{for some } \gamma > 0$$

By scale coherence, this growth reflects in local contributions:
$$\sum_{u=t_0}^{t_0+n} L(u) \gtrsim n^\gamma$$

But then:
$$\sum_t w(t) \mathfrak{D}(t) \geq \sum_{u=t_0}^{\infty} e^{-\alpha u} \cdot u^{\gamma - 1} = \infty$$

This contradicts the $D_E^{\mathrm{tower}}$ permit. $\square$

**Complexity Interpretation:** If local problems explode in size, the weighted sum diverges. The permits **prevent complexity explosions** by construction.

---

## The Arc Consistency Paradigm

### Local Consistency Algorithms

**Definition (Arc Consistency).** A CSP is **arc consistent** if for every constraint $c$ involving variables $x, y$:
$$\forall a \in D_x.\ \exists b \in D_y.\ (a, b) \text{ satisfies } c$$

**AC-3 Algorithm:**
```
function AC3(CSP):
    queue <- all arcs (constraints)
    while queue not empty:
        (x, y, c) <- dequeue
        if REVISE(x, y, c):
            if D_x is empty: return INCONSISTENT
            add all arcs involving x to queue
    return CONSISTENT

function REVISE(x, y, c):
    revised <- false
    for each a in D_x:
        if no b in D_y satisfies c(a, b):
            remove a from D_x
            revised <- true
    return revised
```

**Complexity:** $O(ed^3)$ where $e$ is number of constraints, $d$ is domain size.

### Generalization: $k$-Consistency

**Definition ($k$-Consistency).** A CSP is $k$-consistent if every consistent assignment to $k-1$ variables extends to any $k$-th variable.

| Consistency Level | Description | Complexity |
|-------------------|-------------|------------|
| Node consistency | Unary constraints satisfied | $O(nd)$ |
| Arc consistency | Binary constraints arc-consistent | $O(ed^3)$ |
| Path consistency | 3-consistency | $O(n^3 d^3)$ |
| Strong $k$-consistency | All $j$-consistent for $j \leq k$ | $O(n^k d^k)$ |

### Tower Perspective on Consistency Hierarchies

**Observation:** The $k$-consistency hierarchy forms a **tower**:

- Level $t = 1$: Node consistency (local)
- Level $t = 2$: Arc consistency
- Level $t = 3$: Path consistency
- ...
- Level $t = n$: Global consistency

**RESOLVE-Tower Translation:**

| Tower Component | Consistency Hierarchy |
|-----------------|----------------------|
| $X_t$ | Solutions at $t$-consistency level |
| $S_{t \to s}$ | Projection/restriction |
| $\Phi(t)$ | Number of values/tuples pruned |
| $\mathfrak{D}(t)$ | Pruning at level $t$ |

**Key Insight:** The permits characterize when local consistency **implies** global consistency:

**Theorem (Local-Global CSP).** If strong $(k, k-1)$-consistency implies global consistency (e.g., for tree-structured CSPs with $k=2$), then the tower collapses:
$$X_\infty = X_k$$

---

## Connections to Classical Lifting Theorems

### 1. The Lefschetz Hyperplane Theorem

**Theorem (Lefschetz).** Let $X$ be a smooth projective variety and $H \subset X$ a hyperplane section. Then:
$$H^i(X) \cong H^i(H) \quad \text{for } i < \dim(X) - 1$$

**Connection to RESOLVE-Tower:**

| Lefschetz | Tower Globalization |
|-----------|---------------------|
| Variety $X$ | Global solution space |
| Hyperplane $H$ | Level-$t$ slice |
| Cohomology $H^i$ | Obstruction groups |
| Isomorphism | Local determines global |

**Complexity Interpretation:** Local topological information (hyperplane section) determines global topology. Similarly, local consistency at bounded levels determines global satisfiability.

### 2. Grothendieck's Descent Theory

**Principle:** Local data satisfying cocycle conditions glue to global objects.

**Descent Datum:** $(F_i, \phi_{ij})$ where:
- $F_i$ is local data on patch $U_i$
- $\phi_{ij}: F_i|_{U_{ij}} \to F_j|_{U_{ij}}$ are transition maps
- Cocycle: $\phi_{jk} \circ \phi_{ij} = \phi_{ik}$

**Connection to RESOLVE-Tower:**

| Descent | Tower Structure |
|---------|-----------------|
| Patches $U_i$ | Levels $X_t$ |
| Local data $F_i$ | Solutions $\sigma_t$ |
| Transitions $\phi_{ij}$ | Scale maps $S_{t \to s}$ |
| Cocycle condition | Scale coherence |
| Descended object | Global solution $\sigma_\infty$ |

### 3. Constraint Propagation as Descent

**Observation:** Arc consistency propagation is a form of descent:

1. **Local Data:** Domains $D_x$ for each variable
2. **Consistency:** Constraints ensure compatibility
3. **Propagation:** Reduces domains until fixed point
4. **Global Solution:** Non-empty domains imply solution exists (for certain structures)

**Structural Tractability:** CSPs with bounded treewidth admit polynomial-time algorithms because local consistency propagates globally.

### 4. Proof Complexity and Lifting

**Resolution Refutations:** A resolution proof can be viewed as a tower:
- Level $t$: Clauses derivable in $t$ resolution steps
- Transition: Subsumption/simplification
- Height: Clause width or size

**Width-Based Lower Bounds (Ben-Sasson & Wigderson 2001):**
$$\text{Space}(\pi) \geq \text{Width}(\pi) - O(1)$$

**Connection:** Resolution width is a local-to-global barrier. The permits in RESOLVE-Tower characterize when width-bounded proofs exist.

---

## Applications to Specific Problem Classes

### 1. Hierarchical Satisfiability

**Problem:** SAT instances with hierarchical variable structure:
$$V = V_1 \cup V_2 \cup \ldots \cup V_T$$

where clauses at level $t$ involve only $\bigcup_{s \leq t} V_s$.

**Lifting Theorem Application:**
- $C_\mu^{\mathrm{tower}}$: Each level has bounded number of clauses
- $D_E^{\mathrm{tower}}$: Clause weights decay: $w_t \sim \alpha^t$
- $\mathrm{SC}_\lambda^{\mathrm{tower}}$: Satisfying level $t$ extends to level $t+1$ with bounded effort
- $\mathrm{Rep}_K^{\mathrm{tower}}$: Satisfiability determined by level-wise invariants

**Result:** If permits are satisfied, SAT is solvable in time:
$$O\left(\sum_t \alpha^t \cdot 2^{|V_t|}\right)$$

which converges if $|V_t|$ grows slower than $\log(1/\alpha) \cdot t$.

### 2. Multi-Resolution Optimization

**Problem:** Optimization over a multi-resolution representation:
$$\min_x f(x) \quad \text{where } x = (x_1, x_2, \ldots, x_T) \text{ at resolutions } 1, 2, \ldots, T$$

**Multigrid Method as Tower:**

| Multigrid | Tower Structure |
|-----------|-----------------|
| Fine grid $x_T$ | Top level $X_T$ |
| Coarse grid $x_1$ | Base level $X_1$ |
| Restriction $R$ | $S_{t \to t-1}$ |
| Prolongation $P$ | Inverse transition |
| V-cycle | Tower traversal |

**Convergence via RESOLVE-Tower:**
- Smoothing property $\leftrightarrow$ $D_E^{\mathrm{tower}}$
- Approximation property $\leftrightarrow$ $\mathrm{SC}_\lambda^{\mathrm{tower}}$
- Error bounds $\leftrightarrow$ $\mathrm{Rep}_K^{\mathrm{tower}}$

### 3. Hierarchical Planning

**Problem:** Planning with hierarchical task decomposition.

**HTN Structure:**
- Level 0: High-level goals
- Level $t$: Methods decomposing level $t-1$ tasks
- Level $T$: Primitive actions

**Lifting Theorem Application:**
- Local consistency: Methods applicable at each level
- Global solution: Valid plan achieving all goals

**Complexity:** Hierarchical planning is EXPTIME-complete in general, but RESOLVE-Tower characterizes tractable fragments.

---

## Quantitative Bounds

### Tower Complexity Classes

**Definition.** A tower CSP is in class $\text{TOWER}[\alpha, B, L]$ if:
- Subcritical weight: $w(t) \sim \alpha^t$
- Level bound: $|X_t| \leq B$
- Coherence: $|L(u)| \leq L$ for all $u$

**Theorem (Tower Tractability).** Problems in $\text{TOWER}[\alpha, B, L]$ are solvable in time:
$$O\left(\frac{1}{1-\alpha} \cdot B \cdot L \cdot n\right)$$

where $n$ is total input size.

**Proof Sketch:** Total work is:
$$\sum_{t=0}^T \text{Time}(t) \leq \sum_{t=0}^\infty \alpha^t \cdot B \cdot L = \frac{B \cdot L}{1 - \alpha}$$

### Certificate Size

**Lemma (Certificate Compactness).** The global certificate $K_{\mathrm{Global}}^+$ has size:
$$|K_{\mathrm{Global}}^+| = O\left(\sum_t |K_t^+|\right) = O\left(\sum_t \alpha^t \cdot B\right) = O\left(\frac{B}{1-\alpha}\right)$$

### Verification Complexity

**Theorem (Efficient Verification).** Given certificate $K_{\mathrm{Global}}^+$, verification runs in time:
$$O(|K_{\mathrm{Global}}^+|) = O\left(\frac{B}{1-\alpha}\right)$$

---

## Worked Example: Resolution Width in CNF

**Problem:** Prove unsatisfiability of a CNF formula $F$.

**Tower Structure:**
- Level $t$: Clauses of width $\leq t$
- Transition: Width reduction via resolution
- Height: Total number of clauses at each width

**Permits Check:**

1. **$C_\mu^{\mathrm{tower}}$ (SliceCompact):** Number of width-$t$ clauses bounded:
   $$|\{C \in F : |C| = t\}| \leq \binom{n}{t} \cdot 2^t$$

2. **$D_E^{\mathrm{tower}}$ (SubcritDissip):** Resolution decreases width or increases clauses bounded by input:
   $$\text{work at width } t \leq \text{poly}(n, m) \cdot t$$

3. **$\mathrm{SC}_\lambda^{\mathrm{tower}}$ (ScaleCohere):** Resolution preserves unsatisfiability:
   $$F \text{ unsat} \iff F \cup \{C\} \text{ unsat for any } C \text{ derived}$$

4. **$\mathrm{Rep}_K^{\mathrm{tower}}$ (LocalRecon):** Width-$k$ refutations certify unsatisfiability:
   $$\exists \text{ width-}k \text{ refutation} \Rightarrow F \text{ unsat}$$

**Lifting Result:** If $F$ has a width-$k$ refutation, the proof has size:
$$\text{Size} \leq n^{O(k)}$$

**Interpretation:** The RESOLVE-Tower framework captures the connection between resolution width and space/size in proof complexity.

---

## Connections to Algebraic Complexity

### The GCT Program

**Geometric Complexity Theory (GCT)** uses algebraic geometry to attack $\mathbf{P}$ vs $\mathbf{NP}$.

**Tower Structure in GCT:**
- Levels: Degree-$d$ polynomials
- Transition: Polynomial interpolation
- Lifting: Local properties of polynomials lift to global complexity bounds

**Mulmuley's Paradigm:**
$$\text{Local geometric properties} \xrightarrow{\text{lift}} \text{Global computational lower bounds}$$

**Connection to RESOLVE-Tower:** The interface permits characterize when local algebraic structure (degree bounds, coefficient sparsity) determines global computational complexity.

### Polynomial Identity Testing

**Problem:** Given arithmetic circuit $C$, decide if $C \equiv 0$.

**Hierarchical Structure:**
- Level $t$: Depth-$t$ sub-circuits
- Local property: Sub-circuit outputs
- Global: $C(x) = 0$ for all $x$

**Schwartz-Zippel via Tower:**
- Local check: Evaluate at random point
- Lifting: Low-degree polynomials determined by few evaluations
- Global: Zero polynomial iff zero on all points

---

## Summary

The RESOLVE-Tower theorem, translated to complexity theory, states:

**Local consistency and bounded local complexity lift to global solution existence via the local-global principle.**

This principle:

1. **Unifies CSP Tractability:** Structural restrictions (treewidth, consistency levels) are tower conditions
2. **Explains Multigrid Convergence:** Multi-resolution methods satisfy tower permits
3. **Connects to Proof Complexity:** Resolution width/space tradeoffs follow tower structure
4. **Generalizes Descent:** Grothendieck's descent and sheaf theory provide the abstract framework

The translation illuminates deep connections:

| Hypostructure | Complexity Theory |
|---------------|-------------------|
| Tower permits | Local consistency conditions |
| $X_\infty = \varprojlim X_t$ | Global solution from local |
| Subcritical dissipation | Summable complexity |
| Scale coherence | Telescoping cost bounds |
| Local reconstruction | Sheaf condition |
| Supercritical exclusion | Complexity explosion prevention |

**Key Insight:** Just as the Lefschetz theorem derives global topology from local hyperplane data, the RESOLVE-Tower framework derives global computational tractability from local consistency bounds. The permits precisely characterize when "local is enough"---the fundamental principle underlying tractable combinatorial optimization.

---

## Literature

1. **Apt, K. R. (2003).** *Principles of Constraint Programming.* Cambridge University Press. *CSP foundations.*

2. **Ben-Sasson, E. & Wigderson, A. (2001).** "Short Proofs are Narrow - Resolution Made Simple." *JACM.* *Width-space tradeoffs.*

3. **Beeri, C. et al. (1983).** "On the Desirability of Acyclic Database Schemes." *JACM.* *Acyclic CSPs.*

4. **Dechter, R. (2003).** *Constraint Processing.* Morgan Kaufmann. *Constraint propagation algorithms.*

5. **Freuder, E. C. (1982).** "A Sufficient Condition for Backtrack-Free Search." *JACM.* *Width and tractability.*

6. **Grothendieck, A. (1961).** "Techniques de descente et théorèmes d'existence en géométrie algébrique." *Séminaire Bourbaki.* *Descent theory.*

7. **Grohe, M. (2007).** "The Complexity of Homomorphism and Constraint Satisfaction Problems Seen from the Other Side." *JACM.* *CSP dichotomy.*

8. **Mackworth, A. K. (1977).** "Consistency in Networks of Relations." *Artificial Intelligence.* *Arc consistency.*

9. **Marx, D. (2010).** "Tractable Hypergraph Properties for Constraint Satisfaction and Conjunctive Queries." *JACM.* *Structural tractability.*

10. **Mulmuley, K. (2012).** "The GCT Program Toward the P vs. NP Problem." *Communications ACM.* *Geometric complexity theory.*

11. **Hackbusch, W. (1985).** *Multi-Grid Methods and Applications.* Springer. *Multigrid convergence.*

12. **Atiyah, M. & Macdonald, I. (1969).** *Introduction to Commutative Algebra.* Addison-Wesley. *Local-global principles.*

13. **Ivanyos, G. et al. (2018).** "On the Complexity of the Orbit Problem." *Journal of the ACM.* *Algebraic complexity.*

14. **Lions, P.-L. (1984).** "The Concentration-Compactness Principle in the Calculus of Variations." *Annales IHP.* *Tower limits in analysis.*
