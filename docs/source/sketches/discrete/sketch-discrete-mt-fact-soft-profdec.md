---
title: "FACT-SoftProfDec - Complexity Theory Translation"
---

# FACT-SoftProfDec: The Decomposition Lemma

## Overview

This document provides a complete complexity-theoretic translation of the FACT-SoftProfDec theorem (Soft-to-Profile Decomposition Compilation) from the hypostructure framework. The translation establishes a formal correspondence between concentration-compactness profile decomposition in PDEs and decomposition theorems in complexity theory, revealing deep connections to kernelization, modular decomposition, and tree decomposition.

**Original Theorem Reference:** {prf:ref}`mt-fact-soft-profdec`

**Core Translation:** Profile decomposition derived from soft interfaces corresponds to the **Decomposition Lemma**: hard instances decompose into canonical components via concentration-compactness.

---

## Complexity Theory Statement

**Theorem (Decomposition Lemma, Computational Form).**
Let $(L, k)$ be a parameterized problem with instance $x$ of size $n$. Under appropriate structural conditions (kernelizability, bounded parameter), the instance admits a canonical decomposition:

$$x = \bigoplus_{j=1}^{J} g^{(j)} \cdot \kappa^{(j)} \oplus r$$

where:
1. **Canonical Components $\kappa^{(j)}$:** Irreducible kernel components with bounded size $|\kappa^{(j)}| \leq f(k)$
2. **Symmetry Actions $g^{(j)}$:** Automorphisms/transformations from the problem's symmetry group $G = \text{Aut}(L)$
3. **Residual $r$:** Polynomial-time solvable remainder with $|r| \to 0$ under reductions
4. **Orthogonality:** Components are independent: $\text{interact}(\kappa^{(j)}, \kappa^{(k)}) = 0$ for $j \neq k$
5. **Finite Decomposition:** $J \leq \lfloor E / \delta_0 \rfloor$ where $E$ is instance complexity and $\delta_0$ is the minimum component size

**Corollary (Graph Decomposition Correspondence).**
For graph problems, the decomposition specializes to:
- **Tree Decomposition:** Graphs decompose into bags with bounded treewidth $\text{tw}(G) \leq f(k)$
- **Modular Decomposition:** Graphs decompose into prime modules with bounded size
- **Branch Decomposition:** Hypergraphs decompose into components with bounded branchwidth

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Energy-bounded sequence $(u_n)$ | Problem instance $x$ with bounded parameter $k$ | $\sup_n \Phi(u_n) \leq E$ $\leftrightarrow$ $|x| + k \leq B$ |
| Profile $V^{(j)}$ | Canonical kernel component $\kappa^{(j)}$ | Irreducible subproblem extracted by reduction |
| Symmetry parameters $g_n^{(j)} = (\lambda_n, x_n)$ | Automorphism/embedding $g^{(j)} \in \text{Aut}(L)$ | Variable renaming, structural isomorphism |
| Orthogonality: $d_G(g^{(j)}, g^{(k)}) \to \infty$ | Independence: $\text{interact}(\kappa^{(j)}, \kappa^{(k)}) = 0$ | No shared variables/constraints |
| Remainder $w_n^{(J)} \rightharpoonup 0$ | Residual $r$ is P-solvable | Polynomial-time algorithm for remainder |
| Energy decoupling: $\Phi(u_n) = \sum_j \Phi(V^{(j)}) + o(1)$ | Complexity additivity: $\text{Comp}(x) = \sum_j \text{Comp}(\kappa^{(j)}) + O(1)$ | Pythagorean decomposition of hardness |
| Concentration certificate $K_{C_\mu}^+$ | Kernel extraction succeeds | Component emerges from reduction rules |
| Scaling control $K_{\mathrm{SC}_\lambda}^+$ | Parameter-bounded kernelization | $|\kappa^{(j)}| \leq f(k)$ for computable $f$ |
| Representation certificate $K_{\mathrm{Rep}_K}^+$ | Bounded representation theorem | Finite decomposition exists |
| Lions' dichotomy | Compress-or-disperse dichotomy | Either kernel concentrates or instance disperses |
| Bahouri-Gerard iteration | Iterative reduction rule application | Sequential component extraction |
| Profile list $\{V^{(j)}\}_{j=1}^J$ | Component list $\{\kappa^{(j)}\}_{j=1}^J$ | Finite set of irreducible kernels |
| Critical regularity $s_c$ | Problem-specific parameter | Treewidth, modular width, etc. |
| Symmetry group $G = \mathbb{R}^+ \times \mathbb{R}^d$ | Automorphism group $\text{Aut}(L)$ | Problem symmetries (relabeling, isomorphism) |

---

## Proof Sketch: Concentration-Compactness as Decomposition

### Setup: Parameterized Decomposition Framework

**Definition (Decomposable Problem).**
A parameterized problem $(L, k)$ is decomposable if instances admit a representation:
$$x = \bigoplus_{j=1}^{J} g^{(j)} \cdot \kappa^{(j)} \oplus r$$

where $\bigoplus$ denotes composition (disjoint union for graphs, conjunction for formulas, etc.).

**Definition (Canonical Component).**
A component $\kappa$ is canonical if:
1. **Irreducibility:** No reduction rule applies to $\kappa$
2. **Boundedness:** $|\kappa| \leq f(k)$ for computable $f$
3. **Non-triviality:** $|\kappa| \geq \delta_0 > 0$ (carries genuine complexity)

**Definition (Orthogonal Components).**
Components $\kappa^{(j)}$ and $\kappa^{(k)}$ are orthogonal if:
$$\text{var}(\kappa^{(j)}) \cap \text{var}(\kappa^{(k)}) = \emptyset$$

for SAT-like problems, or more generally:
$$\text{interface}(\kappa^{(j)}, \kappa^{(k)}) = \emptyset$$

---

### Step 1: Lions' Dichotomy = Kernelization Dichotomy

**Claim (Computational Lions' Dichotomy).**
For any parameterized instance $(x, k)$ with bounded complexity, exactly one of the following holds:

**(V) Vanishing/Dispersion:** The instance is polynomial-time solvable without parameterization:
$$\lim_{\text{reductions}} |x| = 0 \quad \text{or} \quad x \in P$$

**(C) Concentration:** There exist reduction sequences and symmetry operations such that a non-trivial kernel emerges:
$$\exists g.\ g^{-1} \cdot \rho^*(x) \rightharpoonup \kappa \neq \emptyset$$

where $\rho^*$ denotes exhaustive reduction rule application.

**Proof (Computational Analogue of Lions):**

**Phase 1: Concentration Function.**

Define the concentration function for instance $x$:
$$Q_R(x) := \max_{\text{subinstance } y \subseteq x, |y| \leq R} \text{hardness}(y)$$

This measures the maximum hardness concentrated in any bounded-size subinstance.

**Phase 2: Dichotomy Check.**

- If $\lim_{R \to \infty} Q_R(x) = 0$: The instance disperses. All hardness is spread across the instance with no concentration. This corresponds to Mode D.D (global polynomial solvability).

- If $\exists \delta > 0, R_0.\ Q_{R_0}(x) \geq \delta$: Hardness concentrates in a bounded region. Extract the concentrating subinstance as the first kernel component $\kappa^{(1)}$.

**Correspondence to Lions:**

| Lions' Concentration-Compactness | Computational Dichotomy |
|----------------------------------|-------------------------|
| Vanishing: $\|u_n\|_{L^{p^*}(B_R)} \to 0$ | Dispersion: all subinstances in P |
| Concentration: $\|v_n - V\| \to 0$ | Kernel extraction: $\kappa$ emerges |
| Rescaling: $\lambda_n^{-\gamma} u_n(\lambda_n^{-1}(\cdot - x_n))$ | Symmetry: $g^{-1} \cdot x$ (relabeling/isomorphism) |

---

### Step 2: Bahouri-Gerard Iteration = Iterative Decomposition

**Claim (Iterative Decomposition Algorithm).**
The decomposition proceeds by iterative extraction:

```
Algorithm: BAHOURI_GERARD_DECOMPOSITION(x, k)
Input: Instance x, parameter k
Output: Decomposition {(g^(j), kappa^(j))}_{j=1}^J, residual r

1. Initialize: j := 0, w^(0) := x
2. While w^(j) has concentration (hardness >= delta_0):
   a. Apply Lions' dichotomy to w^(j)
   b. If CONCENTRATION:
      - Extract symmetry g^(j+1) and kernel kappa^(j+1)
      - Set w^(j+1) := w^(j) - g^(j+1) . kappa^(j+1)
      - j := j + 1
   c. If DISPERSION:
      - Set r := w^(j)
      - Break
3. Return {(g^(j), kappa^(j))}_{j=1}^J, r := w^(J)
```

**Termination Proof (Finite Profiles):**

Each extracted component carries hardness $\text{Comp}(\kappa^{(j)}) \geq \delta_0 > 0$.

By complexity decoupling:
$$\text{Comp}(x) \geq \sum_{j=1}^{J} \text{Comp}(\kappa^{(j)}) \geq J \cdot \delta_0$$

Therefore:
$$J \leq \frac{\text{Comp}(x)}{\delta_0} \leq \frac{E}{\delta_0} < \infty$$

**Correspondence to Bahouri-Gerard:**

| Bahouri-Gerard PDE | Computational Decomposition |
|--------------------|----------------------------|
| $u_n = \sum_j g_n^{(j)} \cdot V^{(j)} + w_n^{(J)}$ | $x = \bigoplus_j g^{(j)} \cdot \kappa^{(j)} \oplus r$ |
| $\Phi(u_n) = \sum_j \Phi(V^{(j)}) + \Phi(w_n^{(J)}) + o(1)$ | $\text{Comp}(x) = \sum_j \text{Comp}(\kappa^{(j)}) + O(1)$ |
| $J \leq E/\delta_0$ | $J \leq \text{Comp}(x)/\delta_0$ |

---

### Step 3: Orthogonality = Component Independence

**Claim (Parameter Divergence = Variable Disjointness).**
The extracted components satisfy orthogonality:
$$\text{interface}(\kappa^{(j)}, \kappa^{(k)}) = \emptyset \quad \text{for } j \neq k$$

**Proof (Separation Argument):**

Suppose components $\kappa^{(j)}$ and $\kappa^{(k)}$ share a variable/constraint. Then the interaction:
$$\text{interact}(\kappa^{(j)}, \kappa^{(k)}) \neq 0$$

would imply that extracting $\kappa^{(k)}$ from the remainder $w^{(j-1)} - g^{(j)} \cdot \kappa^{(j)}$ would still "see" part of $\kappa^{(j)}$, contradicting the weak limit extraction (the kernel is the unique limit modulo symmetry).

**Divergence Metric:**

Define the computational analogue of parameter divergence:
$$d_{\text{Aut}}(g^{(j)}, g^{(k)}) := |\text{var}(g^{(j)} \cdot \kappa^{(j)}) \triangle \text{var}(g^{(k)} \cdot \kappa^{(k)})|$$

For orthogonal components:
$$d_{\text{Aut}}(g^{(j)}, g^{(k)}) = |\text{var}(\kappa^{(j)})| + |\text{var}(\kappa^{(k)})|$$

(complete separation).

---

### Step 4: Energy Decoupling = Complexity Additivity

**Theorem (Pythagorean Theorem for Computational Complexity).**
For orthogonal components:
$$\text{Comp}(x) = \sum_{j=1}^{J} \text{Comp}(\kappa^{(j)}) + \text{Comp}(r) + o(1)$$

**Proof Sketch:**

**Case 1: Additive Complexity (SAT-like).**

For CNF formulas with disjoint variable sets:
$$\text{SAT}(\phi_1 \wedge \phi_2) = \text{SAT}(\phi_1) \wedge \text{SAT}(\phi_2)$$

The complexity (e.g., resolution width) adds:
$$\text{width}(\phi_1 \wedge \phi_2) = \max(\text{width}(\phi_1), \text{width}(\phi_2))$$

For disjoint formulas, this becomes additive in a suitable measure.

**Case 2: Graph Problems (Treewidth-like).**

For graphs with disjoint vertex sets:
$$\text{tw}(G_1 \sqcup G_2) = \max(\text{tw}(G_1), \text{tw}(G_2))$$

Energy decoupling corresponds to the fact that treewidth over disjoint components is the maximum, not the sum, but the solution time is multiplicative/additive.

**Correspondence to Brezis-Lieb:**

| Brezis-Lieb Lemma | Computational Analogue |
|-------------------|------------------------|
| $\|u_n\|^p = \|V\|^p + \|u_n - V\|^p + o(1)$ | $\text{Comp}(x) = \text{Comp}(\kappa) + \text{Comp}(x \ominus \kappa) + O(1)$ |
| Nonlinear functional splitting | Hardness measure splitting |

---

### Step 5: Remainder Vanishing = Residual Tractability

**Claim (Dispersive Remainder).**
After extracting $J$ components, the residual $r = w^{(J)}$ satisfies:
$$r \in P \quad \text{(polynomial-time solvable)}$$

**Proof:**

By Lions' dichotomy, if the residual had concentration $\text{Comp}(r) \geq \delta_0$, we could extract another component $\kappa^{(J+1)}$, contradicting termination.

Therefore:
$$\text{Comp}(r) < \delta_0$$

For problems with a hardness threshold (e.g., NP-hardness requires certain structure), $\text{Comp}(r) < \delta_0$ implies $r \in P$.

**Mode D.D Interpretation:**

The residual being in P corresponds to Mode D.D (global existence/dispersion) in the hypostructure:
- **PDE:** Solution exists globally in time, scatters to linear behavior
- **Complexity:** Instance is globally tractable, reduces to polynomial algorithms

---

## Connections to Graph Decomposition Theorems

### 1. Tree Decomposition and Treewidth

**Theorem (Robertson-Seymour, 1984).**
Every graph $G$ admits a tree decomposition $(T, \{B_t\}_{t \in V(T)})$ where:
- $T$ is a tree
- $B_t \subseteq V(G)$ are bags covering all vertices and edges
- For each $v \in V(G)$, the set $\{t : v \in B_t\}$ induces a connected subtree

The **treewidth** $\text{tw}(G)$ is the minimum $\max_t |B_t| - 1$ over all tree decompositions.

**Connection to Profile Decomposition:**

| Profile Decomposition | Tree Decomposition |
|-----------------------|-------------------|
| Profile $V^{(j)}$ | Bag $B_t$ |
| Symmetry parameter $g^{(j)} = (\lambda, x)$ | Tree node $t$ |
| Orthogonality: $d_G(g^{(j)}, g^{(k)}) \to \infty$ | Tree distance: $d_T(t, s)$ large |
| Energy per profile: $\Phi(V^{(j)}) \geq \delta_0$ | Bag size: $|B_t| \leq \text{tw}(G) + 1$ |
| Remainder $w_n \to 0$ | Separator properties of tree edges |

**Decomposition Lemma for Treewidth:**

For graphs with bounded treewidth $\text{tw}(G) \leq k$:
- **Components:** Bags $B_t$ of size at most $k+1$
- **Orthogonality:** Tree structure ensures bags overlap only at separators
- **Finite decomposition:** $|V(T)| \leq |V(G)|$
- **Tractability:** Many NP-hard problems are FPT parameterized by treewidth

**Certificate:**
$$K_{\text{TreeDec}}^+ = (T, \{B_t\}, \text{tw}(G) \leq k)$$

### 2. Modular Decomposition

**Definition (Module).**
A module $M \subseteq V(G)$ is a set such that for all $v \notin M$:
$$N(v) \cap M \in \{\emptyset, M\}$$

Every vertex outside $M$ is either adjacent to all of $M$ or none of $M$.

**Theorem (Modular Decomposition, Gallai 1967).**
Every graph admits a unique modular decomposition tree where:
- Leaves are single vertices (trivial modules)
- Internal nodes are labeled **series** (complete join), **parallel** (disjoint union), or **prime**
- Prime modules have no non-trivial submodules

**Connection to Profile Decomposition:**

| Profile Decomposition | Modular Decomposition |
|-----------------------|-----------------------|
| Profile $V^{(j)}$ | Module $M_j$ |
| Orthogonality | Modules are disjoint or nested |
| Remainder | Quotient graph (prime structure) |
| Energy decoupling | Recursive structure of decomposition tree |

**Decomposition Lemma for Modules:**

An instance $(G, k)$ decomposes into modules:
$$G = \text{compose}(\{M_j\}_{j=1}^J, Q)$$

where $Q$ is the quotient graph (prime structure). If $Q$ has bounded size $|Q| \leq f(k)$, the decomposition witnesses tractability.

**Certificate:**
$$K_{\text{ModDec}}^+ = (\{M_j\}_{j=1}^J, Q, |Q| \leq f(k))$$

### 3. Branch Decomposition and Branchwidth

**Definition (Branch Decomposition).**
A branch decomposition of hypergraph $H$ is a tree $T$ with:
- Leaves in bijection with hyperedges
- Internal nodes have degree 3
- For each edge $e \in E(T)$, define $\text{mid}(e)$ as vertices appearing on both sides

The **branchwidth** is $\min \max_{e \in E(T)} |\text{mid}(e)|$.

**Connection to Profile Decomposition:**

Branch decomposition is the hypergraph analogue of tree decomposition, directly corresponding to profile decomposition for constraint satisfaction problems.

| Profile Decomposition | Branch Decomposition |
|-----------------------|----------------------|
| Profile energy $\Phi(V^{(j)})$ | Middle set size $|\text{mid}(e)|$ |
| Orthogonality condition | Tree edge separation |
| Finite profiles $J < \infty$ | Tree has $|E(H)|$ leaves |

### 4. Rank Decomposition and Rank-Width

**Definition (Rank-Width).**
For graph $G$, the rank-width $\text{rw}(G)$ is defined via the cut-rank function:
$$\text{cutrk}_G(A) := \text{rank}_{\mathbb{F}_2}(M[A, V \setminus A])$$

where $M$ is the adjacency matrix.

**Connection:** Rank-width generalizes treewidth to dense graphs. The profile decomposition for rank-width:
- **Profiles:** Cuts with bounded rank
- **Orthogonality:** Cuts are nested (submodular structure)
- **Energy:** Cut-rank function

---

## Certificate Construction

### Profile Decomposition Certificate (Complete)

```
K_{ProfDec}^+ = {
  profiles: {kappa^(j)}_{j=1}^J,
  symmetries: {g^(j)}_{j=1}^J,
  orthogonality: {
    forall j != k: interface(kappa^(j), kappa^(k)) = empty,
    divergence_proof: d_Aut(g^(j), g^(k)) >= separation_threshold
  },
  remainder: {
    r: residual instance,
    tractability: proof that r in P,
    vanishing_rate: Comp(r) < delta_0
  },
  bounds: {
    J <= E / delta_0,
    |kappa^(j)| <= f(k) for all j,
    sum_j Comp(kappa^(j)) <= E
  },
  metadata: {
    critical_parameter: k,
    symmetry_group: Aut(L),
    decomposition_type: "tree" | "modular" | "branch"
  }
}
```

### Specialized Certificates

**Tree Decomposition Certificate:**
```
K_{TreeDec}^+ = {
  tree: T,
  bags: {B_t}_{t in V(T)},
  treewidth: tw(G) = max_t |B_t| - 1,
  verification: {
    coverage: union_t B_t = V(G),
    edge_coverage: forall (u,v) in E(G), exists t: {u,v} subset B_t,
    connectivity: forall v, {t : v in B_t} is connected in T
  }
}
```

**Modular Decomposition Certificate:**
```
K_{ModDec}^+ = {
  modules: {M_j}_{j=1}^J,
  quotient: Q (prime graph),
  decomposition_tree: D,
  verification: {
    partition: modules partition V(G),
    module_property: forall v not in M_j, N(v) cap M_j in {empty, M_j},
    primality: Q has no non-trivial modules
  }
}
```

---

## Quantitative Bounds

### Number of Components

**Profile Bound:** $J \leq E / \delta_0$ where:
- $E = \text{Comp}(x)$ is the instance complexity (e.g., size, parameter)
- $\delta_0$ is the minimum component hardness (problem-specific)

**Specific Examples:**
- **Vertex Cover:** $J \leq k$ components (each covers at least one edge)
- **SAT:** $J \leq n$ components (each involves at least one variable)
- **Treewidth-$k$ graphs:** $J \leq n$ bags of size at most $k+1$

### Decomposition Time

**Polynomial-Time Decomposition:**

| Decomposition Type | Time Complexity | Reference |
|--------------------|-----------------|-----------|
| Tree decomposition (approx.) | $O(n \cdot 2^{3k} \cdot k)$ | Bodlaender et al. 2016 |
| Modular decomposition | $O(n + m)$ | Tedder et al. 2008 |
| Branch decomposition | $O(n^3)$ for bounded branchwidth | Robertson-Seymour |
| Rank decomposition | $O(n^3)$ for bounded rank-width | Oum 2005 |

### Component Size Bounds (Kernelization)

| Problem | Component Size | Decomposition Type |
|---------|----------------|-------------------|
| Vertex Cover | $2k$ | Crown decomposition |
| Feedback Vertex Set | $O(k^2)$ | Iterative compression |
| Dominating Set (planar) | $O(k)$ | Protrusion replacement |
| Treewidth-$k$ problems | Bags of size $k+1$ | Tree decomposition |

---

## Algorithmic Implications

### FPT via Decomposition

**Meta-Algorithm:**

```
Algorithm: SOLVE_VIA_DECOMPOSITION(x, k)

1. Compute decomposition: {kappa^(j)}, r := DECOMPOSE(x, k)

2. Solve components independently:
   for j = 1 to J:
     sol^(j) := BRUTE_FORCE(kappa^(j))  // O(2^{f(k)}) time

3. Solve residual:
   sol_r := POLY_ALGORITHM(r)  // O(n^c) time

4. Combine solutions:
   return COMBINE({sol^(j)}, sol_r)

Time: O(n^c + J * 2^{f(k)}) = O(n^c + (E/delta_0) * 2^{f(k)})
     = O(n^c * h(k)) for FPT
```

### Dynamic Programming on Decompositions

For many problems, the decomposition enables efficient DP:

**Tree Decomposition DP:**
$$\text{OPT}[t, S] = \text{best solution in subtree } T_t \text{ with interface } S \subseteq B_t$$

Time: $O(n \cdot 2^{\text{tw}(G)} \cdot \text{tw}(G))$

**Modular Decomposition DP:**
$$\text{OPT}[M] = \text{best solution for module } M \text{ given children solutions}$$

Time: Depends on quotient graph complexity.

---

## Connections to Classical Results

### 1. Robertson-Seymour Graph Minor Theorem

**Statement:** Every minor-closed graph family $\mathcal{F}$ is characterized by a finite set of excluded minors.

**Connection:** The Graph Minor Theorem implies that graphs in $\mathcal{F}$ have bounded treewidth (up to a finite number of exceptions), enabling the decomposition lemma.

### 2. Courcelle's Theorem

**Statement:** Every MSO-definable problem is linear-time solvable on graphs of bounded treewidth.

**Connection:** Courcelle's theorem is the algorithmic payoff of tree decomposition. Once the decomposition certificate $K_{\text{TreeDec}}^+$ is established, any MSO-expressible property can be checked in linear time.

### 3. Erdos-Rado Sunflower Lemma

**Statement:** Any family of $k! \cdot (p-1)^k + 1$ sets of size $k$ contains a sunflower with $p$ petals.

**Connection:** Sunflower extraction is a decomposition step: the core of the sunflower is a canonical component $\kappa$, and the petals are orthogonal remainder structures.

### 4. Crown Decomposition for Vertex Cover

**Statement:** Every graph with $> k^2$ vertices either contains a crown or has a kernel of size $\leq 2k$.

**Connection:** Crown decomposition instantiates the profile decomposition:
- **Crown $(C, H, M)$:** Reducible profile (can be extracted)
- **No crown:** Concentration achieved, kernel bounded

---

## Summary

The FACT-SoftProfDec theorem, translated to complexity theory, establishes the **Decomposition Lemma**:

1. **Concentration-Compactness = Kernelization Dichotomy:** Instances either concentrate into bounded kernels or disperse into polynomial-time solvability.

2. **Bahouri-Gerard Iteration = Iterative Reduction:** Components are extracted sequentially via reduction rules, with each component carrying bounded hardness.

3. **Orthogonality = Independence:** Extracted components are variable-disjoint, enabling parallel solution.

4. **Energy Decoupling = Complexity Additivity:** Total hardness decomposes as sum of component hardnesses plus tractable remainder.

5. **Finite Termination = Bounded Profile Count:** The number of components is bounded by $E / \delta_0$.

**Certificate:**
$$K_{\mathrm{ProfDec}_{s_c,G}}^+ = (\{\kappa^{(j)}\}_{j=1}^J, \{g^{(j)}\}_{j=1}^J, \mathsf{orthogonality}, \mathsf{remainder\_tractability})$$

This certificate enables FPT algorithms via decomposition-based dynamic programming, connecting the hypostructure framework to the algorithmic meta-theorems of parameterized complexity.

---

## Literature

1. **Lions, P.-L. (1984).** "The Concentration-Compactness Principle in the Calculus of Variations." *Annales IHP*. *Original concentration-compactness.*

2. **Bahouri, H. & Gerard, P. (1999).** "High Frequency Approximation of Solutions to Critical Nonlinear Wave Equations." *American Journal of Mathematics*. *Profile decomposition for dispersive PDEs.*

3. **Robertson, N. & Seymour, P. D. (1984-2004).** "Graph Minors I-XXIII." *JCTB*. *Tree decomposition and graph minor theory.*

4. **Bodlaender, H. L. (1996).** "A Linear-Time Algorithm for Finding Tree-Decompositions of Small Treewidth." *SICOMP*. *Efficient tree decomposition.*

5. **Courcelle, B. (1990).** "The Monadic Second-Order Logic of Graphs I." *Information and Computation*. *MSO decidability on bounded treewidth.*

6. **Gallai, T. (1967).** "Transitiv orientierbare Graphen." *Acta Mathematica Hungarica*. *Modular decomposition.*

7. **Tedder, M. et al. (2008).** "Simpler Linear-Time Modular Decomposition via Recursive Factorizing Permutations." *ICALP*. *Linear-time modular decomposition.*

8. **Cygan, M. et al. (2015).** *Parameterized Algorithms.* Springer. *Modern FPT and kernelization.*

9. **Downey, R. G. & Fellows, M. R. (1999).** *Parameterized Complexity.* Springer. *Foundations of parameterized complexity.*

10. **Oum, S. & Seymour, P. (2006).** "Approximating Clique-Width and Branch-Width." *JCTB*. *Rank-width and branch-width.*
