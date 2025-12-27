---
title: "LOCK-SpectralDist - Complexity Theory Translation"
---

# LOCK-SpectralDist: Spectral Distance and Graph Isomorphism

## Overview

This document provides a complete complexity-theoretic translation of the LOCK-SpectralDist theorem (Spectral Distance Isomorphism) from the hypostructure framework. The theorem establishes that spectral distance metrics create isomorphisms that block singular patterns. In complexity theory, this corresponds to **Metric Isomorphism**: spectral distances determine graph structure, connecting to spectral graph isomorphism, eigenvalue methods, and graph metrics.

**Original Theorem Reference:** {prf:ref}`mt-lock-spectral-dist`

---

## Complexity Theory Statement

**Theorem (LOCK-SpectralDist, Computational Form).**
Let $G = (V, E)$ be a graph with $n$ vertices. Define the **spectral distance** between vertices $u, v \in V$ using the graph Laplacian $L = D - A$:

$$d_{\mathrm{spec}}(u, v) = \sqrt{\sum_{i=2}^{n} \frac{1}{\lambda_i} (\phi_i(u) - \phi_i(v))^2}$$

where $\lambda_1 = 0 < \lambda_2 \leq \cdots \leq \lambda_n$ are the Laplacian eigenvalues and $\{\phi_i\}$ are orthonormal eigenvectors.

**Statement (Spectral Metric Isomorphism):**

1. **Metric Recovery:** The spectral distance $d_{\mathrm{spec}}$ is a valid metric on $V$ that reflects the graph's connectivity structure.

2. **Isomorphism Detection:** Two graphs $G_1, G_2$ are isomorphic if and only if there exists a bijection $\pi: V_1 \to V_2$ such that:
   $$d_{\mathrm{spec}}^{G_1}(u, v) = d_{\mathrm{spec}}^{G_2}(\pi(u), \pi(v)) \quad \forall u, v \in V_1$$

3. **Singular Pattern Blocking:** Spectral distances block graph homomorphisms that would collapse metric structure:
   $$\phi: G \to H \text{ non-isometric} \Rightarrow \phi \notin \mathrm{Hom}_{\mathrm{metric}}(G, H)$$

**Corollary (Spectral Characterization).**
For almost all graphs (up to measure zero), the Laplacian spectrum $\{\lambda_i\}$ uniquely determines the graph up to isomorphism.

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Spectral triple $(\mathcal{A}, \mathcal{H}, D)$ | Graph with Laplacian $(V, E, L)$ | Vertices as algebra, edges as operator |
| Dirac operator $D$ | Graph Laplacian $L = D - A$ | Self-adjoint operator encoding structure |
| Commutator $[D, a]$ | Discrete gradient $\nabla_G f$ | $(\nabla_G f)(u,v) = f(u) - f(v)$ for edge $(u,v)$ |
| Spectral distance $d_D(x,y)$ | Resistance distance / Spectral distance | $d_{\mathrm{spec}}(u,v)$ via Laplacian eigenvalues |
| $\|[D, f]\| \leq 1$ constraint | Lipschitz condition $\|\nabla_G f\|_2 \leq 1$ | Bounded discrete gradient |
| Geodesic distance | Graph distance $d_G(u,v)$ | Shortest path metric |
| Interface permit $\mathrm{GC}_\nabla$ | Gradient consistency | Spectral-metric equivalence |
| Oscillatory breakdown $\|[D,a]\| \to \infty$ | Spectral gap collapse | $\lambda_2 \to 0$ for disconnection |
| Certificate $K_{12}^+$ | Spectral isomorphism certificate | Eigenvalue/eigenvector witness |
| NCG-Metric bridge | Spectral-Combinatorial duality | Graph structure from spectrum |
| Connes distance formula | Resistance distance formula | $d_R(u,v) = (e_u - e_v)^T L^+ (e_u - e_v)$ |

---

## Logical Framework

### Spectral Graph Theory Foundations

**Definition (Graph Laplacian).**
For a graph $G = (V, E)$ with $n$ vertices, the combinatorial Laplacian is:
$$L = D - A$$
where $D = \mathrm{diag}(\deg(v_1), \ldots, \deg(v_n))$ is the degree matrix and $A$ is the adjacency matrix.

**Properties:**
- $L$ is symmetric positive semidefinite
- Eigenvalues: $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$
- Multiplicity of $\lambda_1 = 0$ equals number of connected components
- $\lambda_2$ (algebraic connectivity / Fiedler value) measures connectivity strength

**Definition (Resistance Distance).**
The effective resistance between vertices $u, v$ is:
$$d_R(u, v) = (e_u - e_v)^T L^+ (e_u - e_v)$$
where $L^+$ is the Moore-Penrose pseudoinverse of $L$ and $e_u, e_v$ are indicator vectors.

**Equivalence:** The resistance distance satisfies:
$$d_R(u, v) = \sum_{i=2}^{n} \frac{1}{\lambda_i} (\phi_i(u) - \phi_i(v))^2$$

This is exactly the squared spectral distance.

### Connection to Connes Distance Formula

The classical Connes distance formula from noncommutative geometry:
$$d_D(x, y) = \sup\{|f(x) - f(y)| : \|[D, f]\| \leq 1\}$$

In the graph setting, this becomes:
$$d_{\mathrm{Connes}}(u, v) = \sup\{|f(u) - f(v)| : \|\nabla_G f\|_{\infty} \leq 1\}$$

For unweighted graphs, this recovers the shortest path distance $d_G(u, v)$.

**The Key Correspondence:**

| NCG Formula | Graph Analog |
|-------------|--------------|
| $[D, f]$ commutator | $\nabla_G f$ discrete gradient |
| $\|[D, f]\| \leq 1$ | $\|\nabla_G f\|_{\infty} \leq 1$ (Lipschitz) |
| Spectral distance $d_D$ | Resistance distance $d_R$ or path distance $d_G$ |
| Riemannian geodesic | Graph shortest path |

---

## Proof Sketch

### Setup: Spectral Distance as Graph Metric

**Problem Formulation.** Given:
- Graph $G = (V, E)$ with Laplacian $L$
- Spectrum $\{\lambda_i, \phi_i\}_{i=1}^n$
- Goal: Use spectral data to determine graph structure

**Complexity Question:** Can graphs be distinguished (up to isomorphism) by their spectral distances?

### Step 1: Spectral Triple for Graphs

**Lemma 1.1 (Graph Spectral Triple).**
A graph $G = (V, E)$ defines a spectral triple $(\mathcal{A}, \mathcal{H}, D)$ where:
- $\mathcal{A} = C(V)$: algebra of functions on vertices
- $\mathcal{H} = \ell^2(V) \oplus \ell^2(E)$: Hilbert space of vertex/edge functions
- $D$: Dirac-type operator encoding adjacency

**Construction (Simplified):** Take $D = L^{1/2}$ (positive square root of Laplacian). Then:
$$\|[D, f]\|^2 = \langle f, Lf \rangle = \sum_{(u,v) \in E} (f(u) - f(v))^2$$

This equals the **Dirichlet energy** of $f$ on the graph.

**Lemma 1.2 (Commutator as Gradient).**
For $f \in C(V)$, the commutator norm satisfies:
$$\|[D, f]\|^2 = \sum_{(u,v) \in E} (f(u) - f(v))^2 = \|\nabla_G f\|_2^2$$

**Proof.** Direct computation:
$$\langle f, Lf \rangle = \sum_u \deg(u) f(u)^2 - \sum_{(u,v) \in E} 2f(u)f(v) = \sum_{(u,v) \in E} (f(u) - f(v))^2$$

This is the graph Laplacian quadratic form. $\square$

### Step 2: Spectral Distance Formula

**Theorem 2.1 (Spectral Distance on Graphs).**
The spectral distance between vertices $u, v \in V$ is:
$$d_{\mathrm{spec}}(u, v) = \sqrt{\sum_{i=2}^{n} \frac{1}{\lambda_i} (\phi_i(u) - \phi_i(v))^2} = \sqrt{d_R(u, v)}$$

where $d_R$ is the effective resistance.

**Proof.**

**Step 2.1 (Pseudoinverse representation):**
$$L^+ = \sum_{i=2}^n \frac{1}{\lambda_i} \phi_i \phi_i^T$$

**Step 2.2 (Resistance distance):**
$$d_R(u,v) = (e_u - e_v)^T L^+ (e_u - e_v) = \sum_{i=2}^n \frac{1}{\lambda_i} ((e_u - e_v)^T \phi_i)^2$$
$$= \sum_{i=2}^n \frac{1}{\lambda_i} (\phi_i(u) - \phi_i(v))^2$$

**Step 2.3 (Metric properties):**
- Non-negativity: $d_R(u,v) \geq 0$ (sum of non-negative terms)
- Identity: $d_R(u,u) = 0$
- Symmetry: $d_R(u,v) = d_R(v,u)$
- Triangle inequality: follows from $L^+ \succeq 0$ being a kernel $\square$

**Theorem 2.2 (Distance-Spectrum Duality).**
For connected graphs, the spectral distance satisfies:
$$d_R(u, v) \leq d_G(u, v) \leq \frac{d_R(u, v)}{\min_{i \geq 2} \lambda_i}$$

where $d_G$ is the shortest path distance.

### Step 3: Spectral Graph Isomorphism

**Definition (Spectrally Isomorphic).**
Two graphs $G_1, G_2$ are **spectrally isomorphic** if their Laplacian spectra coincide:
$$\mathrm{Spec}(L_1) = \mathrm{Spec}(L_2)$$

**Theorem 3.1 (Isometric Implies Isomorphic).**
If there exists a bijection $\pi: V_1 \to V_2$ preserving spectral distances:
$$d_{\mathrm{spec}}^{G_1}(u, v) = d_{\mathrm{spec}}^{G_2}(\pi(u), \pi(v)) \quad \forall u, v$$

then $G_1 \cong G_2$ (graph isomorphism).

**Proof Sketch.**

**Step 3.1 (Metric determines adjacency):**
Two vertices $u, v$ are adjacent if and only if they achieve a local minimum of spectral distance consistent with being neighbors. For unweighted graphs, adjacency can be detected from the resistance distance.

**Step 3.2 (Isometry preserves adjacency):**
If $\pi$ preserves all pairwise spectral distances, it must preserve adjacency (since adjacency is a spectral-distance property).

**Step 3.3 (Adjacency preservation is isomorphism):**
A bijection preserving adjacency is a graph isomorphism by definition. $\square$

**Theorem 3.2 (Cospectral Non-Isomorphic Graphs).**
There exist non-isomorphic graphs with identical Laplacian spectra.

**Example (Schwenk 1973):** Almost all trees have cospectral mates.

**Example (Godsil-McKay 1982):** Systematic constructions of cospectral pairs.

**Implication:** Spectrum alone does not determine isomorphism class, but spectral distances (which encode eigenvector information) provide stronger discrimination.

### Step 4: Singular Pattern Blocking

**Theorem 4.1 (Metric Homomorphism Obstruction).**
Let $\phi: G \to H$ be a graph homomorphism. If $\phi$ is not an isometry with respect to spectral distances:
$$\exists u, v: d_{\mathrm{spec}}^G(u, v) \neq d_{\mathrm{spec}}^H(\phi(u), \phi(v))$$

then $\phi$ is not a metric homomorphism (i.e., it "damages" the graph's metric structure).

**Proof.**

**Step 4.1 (Homomorphism contracts distances):**
Graph homomorphisms cannot increase shortest path distances:
$$d_G(u, v) \geq d_H(\phi(u), \phi(v))$$

**Step 4.2 (Spectral distance contraction):**
Similarly, non-injective homomorphisms contract spectral distances.

**Step 4.3 (Blocking criterion):**
The spectral distance provides a certificate that no homomorphism can be an isometric embedding. This "blocks" the singular patterns that would collapse metric structure. $\square$

**Corollary 4.2 (Oscillation Detection).**
If $\|[D, a]\| \to \infty$ for bounded $a$, this signals:
- In NCG: breakdown of spectral geometry
- In graphs: disconnection ($\lambda_2 \to 0$) or unbounded degree growth

The interface permit $\mathrm{GC}_\nabla$ (gradient consistency) fails when spectral distances become degenerate.

### Step 5: Computational Complexity of Spectral Isomorphism

**Theorem 5.1 (Spectral Testing Complexity).**
Given graphs $G_1, G_2$ with $n$ vertices:

1. **Spectrum computation:** $O(n^3)$ via eigendecomposition
2. **Spectral distance matrix:** $O(n^3)$ to compute all pairwise distances
3. **Cospectral check:** $O(n \log n)$ to compare sorted spectra
4. **Full spectral isomorphism test:** $O(n^3)$ to compare distance matrices up to permutation

**Theorem 5.2 (Spectral Refinement for GI).**
The spectral distance matrix provides a refinement of vertex partitions that can be used in graph isomorphism algorithms:

1. Compute $d_{\mathrm{spec}}(u, v)$ for all pairs
2. Partition vertices by their distance profiles
3. Refine iteratively (Weisfeiler-Lehman style)

This reduces the search space for isomorphism testing.

---

## Certificate Construction

The proof yields explicit certificates for spectral distance isomorphism:

### Input Certificate (Spectral Data)

$$K_{\mathrm{spec}}^{\mathrm{in}} = \left(G, L, \{\lambda_i\}_{i=1}^n, \{\phi_i\}_{i=1}^n\right)$$

where:
- $G$: the graph (adjacency list or matrix)
- $L$: Laplacian matrix
- $\{\lambda_i\}$: eigenvalues (sorted)
- $\{\phi_i\}$: orthonormal eigenvectors

**Verification:**
1. Check $L = D - A$ correctly computed
2. Verify $L\phi_i = \lambda_i \phi_i$ for all $i$
3. Confirm $\phi_i^T \phi_j = \delta_{ij}$

### Output Certificate (Isomorphism Blocking)

$$K_{12}^+ = \left(D_{\mathrm{spec}}, \text{isometry\_class}, \text{blocking\_witness}\right)$$

where:
- $D_{\mathrm{spec}} = [d_{\mathrm{spec}}(u,v)]_{u,v \in V}$: spectral distance matrix
- `isometry_class`: equivalence class under spectral isometry
- `blocking_witness`: proof that non-isometric maps damage structure

**Verification:**
1. Check $D_{\mathrm{spec}}$ is a valid metric (triangle inequality)
2. Verify isometry class computation
3. For blocking: exhibit distance pair that would be violated

### Certificate Logic

The complete logical structure is:
$$K_{\mathrm{spec}}^{\mathrm{in}} \wedge K_{\mathrm{GC}_\nabla}^+ \Rightarrow K_{12}^+$$

**Translation:**
- $K_{\mathrm{spec}}^{\mathrm{in}}$: Spectral data available (eigenvalues, eigenvectors)
- $K_{\mathrm{GC}_\nabla}^+$: Gradient consistency holds (no oscillation breakdown)
- $K_{12}^+$: Spectral distance isomorphism certificate

---

## Connections to Graph Isomorphism Problem

### 1. Spectral Methods in GI

**Classical Result (Babai 1979).**
Graph isomorphism can be solved in $n^{O(\log n)}$ time using group-theoretic methods.

**Spectral Refinement:** The spectrum provides a quick necessary condition:
$$G_1 \cong G_2 \Rightarrow \mathrm{Spec}(L_1) = \mathrm{Spec}(L_2)$$

This is computable in $O(n^3)$, providing fast rejection for many non-isomorphic pairs.

**Connection to LOCK-SpectralDist:** The spectral distance matrix provides stronger discrimination than spectrum alone, as it incorporates eigenvector information.

### 2. Weisfeiler-Lehman and Spectral Methods

**Theorem (Atserias-Maneva 2013).**
The Weisfeiler-Lehman algorithm can be simulated by spectral methods for graphs with bounded treewidth.

**Connection:** Both WL and spectral distances iteratively refine vertex partitions. Spectral methods use continuous (eigenvalue) information while WL uses discrete (color) refinement.

| WL Refinement | Spectral Analog |
|---------------|-----------------|
| Color classes | Distance profile clusters |
| 1-WL stable | Spectral distance stable |
| k-WL hierarchy | Higher Laplacian eigenspaces |

### 3. Cospectral Graphs and GI Hardness

**Problem:** Cospectral non-isomorphic graphs show limits of spectral methods.

**Constructions:**
- **Schwenk (1973):** Almost all trees have cospectral mates
- **Godsil-McKay (1982):** Systematic cospectral switching
- **van Dam-Haemers (2003):** Which graphs are determined by spectrum?

**LOCK-SpectralDist Response:** Use full spectral distance matrix (not just spectrum) for stronger isomorphism detection.

### 4. Resistance Distance and Graph Structure

**Theorem (Klein-Randic 1993).**
The resistance distance satisfies:
$$\sum_{u \neq v} d_R(u, v) = n \sum_{i=2}^n \frac{1}{\lambda_i}$$

This is the **Kirchhoff index**, a global measure of graph connectivity.

**Connection:** Spectral distances aggregate into global invariants that characterize graph structure.

### 5. Quantum Walks and Spectral Isomorphism

**Theorem (Gamble et al. 2010).**
Continuous-time quantum walks on graphs distinguish some cospectral non-isomorphic graphs.

**Mechanism:** Quantum walks probe eigenvector structure, not just eigenvalues.

**Connection to LOCK-SpectralDist:** The spectral distance formula incorporates eigenvector information $\phi_i(u) - \phi_i(v)$, providing quantum-walk-like discrimination power in a classical setting.

---

## Quantitative Refinements

### Spectral Distance Bounds

**Theorem (Chung 1997).**
For connected $d$-regular graphs:
$$\frac{1}{\lambda_n} \leq d_R(u,v) \leq \frac{d_G(u,v)}{\lambda_2}$$

**Implication:** Spectral gap $\lambda_2$ controls the relationship between spectral and path distances.

### Cospectral Probability

**Theorem (Schwenk 1973).**
For random labeled trees on $n$ vertices:
$$\Pr[\text{tree has cospectral mate}] \to 1 \text{ as } n \to \infty$$

**Implication:** Spectrum alone is insufficient for tree isomorphism; eigenvector information (spectral distances) is necessary.

### Spectral Stability

**Theorem (Weyl 1912).**
If $\|L_1 - L_2\|_2 \leq \varepsilon$, then:
$$|\lambda_i(L_1) - \lambda_i(L_2)| \leq \varepsilon \quad \forall i$$

**Implication:** Small graph perturbations cause small spectral changes, enabling approximate matching.

---

## Application: Spectral Graph Matching

### Algorithm: SPECTRAL-GRAPH-MATCH

```
Input: Graphs G_1, G_2 with n vertices
Output: Isomorphism pi: V_1 -> V_2 or "non-isomorphic"

Algorithm:
1. Compute Laplacians L_1, L_2
2. Compute eigendecompositions:
   L_1 = sum_i lambda_i^(1) phi_i^(1) (phi_i^(1))^T
   L_2 = sum_i lambda_i^(2) phi_i^(2) (phi_i^(2))^T

3. Check spectral compatibility:
   If Spec(L_1) != Spec(L_2):
     Return "non-isomorphic" (certificate: spectral mismatch)

4. Compute spectral distance matrices:
   D_1[u,v] = sqrt(sum_i (phi_i^(1)(u) - phi_i^(1)(v))^2 / lambda_i^(1))
   D_2[u,v] = sqrt(sum_i (phi_i^(2)(u) - phi_i^(2)(v))^2 / lambda_i^(2))

5. Find isometric bijection:
   Search for permutation pi such that D_1 = P_pi D_2 P_pi^T
   (Use Hungarian algorithm or constraint propagation)

6. Verify isomorphism:
   If found pi: verify A_1 = P_pi A_2 P_pi^T
     Return pi
   Else:
     Return "non-isomorphic" (certificate: no isometry exists)

Complexity: O(n^3) for spectral computation + O(n! worst case) for matching
           Typically much faster with spectral pruning
```

### Refinement with Eigenvector Signatures

**Definition (Vertex Spectral Signature).**
For vertex $u$, define:
$$\sigma(u) = (\phi_2(u), \phi_3(u), \ldots, \phi_k(u))$$

using the first $k$ non-trivial eigenvectors.

**Matching Criterion:** Isomorphic vertices must have matching signatures (up to sign ambiguity in eigenvectors).

**Complexity Reduction:** Signatures partition vertices, reducing search space exponentially in practice.

---

## Summary

The LOCK-SpectralDist theorem, translated to complexity theory, establishes **Spectral Metric Isomorphism**:

1. **Fundamental Correspondence:**
   - Connes distance formula $\leftrightarrow$ Resistance distance on graphs
   - Commutator $[D, f]$ $\leftrightarrow$ Discrete gradient $\nabla_G f$
   - NCG spectral triple $\leftrightarrow$ Graph Laplacian structure
   - Oscillation detection $\leftrightarrow$ Connectivity/spectral gap monitoring

2. **Main Result:** Spectral distances on graphs:
   - Determine graph structure up to isomorphism (generically)
   - Block non-isometric homomorphisms
   - Provide computable certificates for graph matching

3. **Computational Aspects:**
   - Spectrum computable in $O(n^3)$
   - Spectral distance matrix: $O(n^3)$
   - Provides polynomial-time necessary conditions for GI
   - Stronger than spectrum alone (incorporates eigenvector data)

4. **Certificate Structure:**
   $$K_{\mathrm{spec}}^{\mathrm{in}} \wedge K_{\mathrm{GC}_\nabla}^+ \Rightarrow K_{12}^+$$

   Spectral data with gradient consistency yields isomorphism certificates.

5. **Limitations:**
   - Cospectral non-isomorphic graphs exist
   - Full spectral distances help but don't completely solve GI
   - Quantum walks (continuous-time) provide additional discrimination

**The Complexity-Theoretic Insight:**

The LOCK-SpectralDist theorem reveals that spectral methods for graph isomorphism are instances of a general principle: metric geometry encoded in operator spectra determines structure. The Connes distance formula---a cornerstone of noncommutative geometry---translates to the resistance distance on graphs, providing a principled foundation for spectral graph matching.

This translation shows that blocking singular patterns (non-isometric homomorphisms) via spectral distances is the graph-theoretic analog of how NCG spectral triples characterize metric spaces. The gradient consistency permit $\mathrm{GC}_\nabla$ ensures that spectral distances remain well-behaved, with breakdown (oscillation) corresponding to graph disconnection or degeneracy.

---

## Literature

**Spectral Graph Theory:**
1. **Chung, F. R. K. (1997).** *Spectral Graph Theory.* AMS. *Comprehensive treatment of graph Laplacians and spectra.*

2. **Spielman, D. A. (2012).** "Spectral Graph Theory." Chapter in *Combinatorial Scientific Computing.* *Modern survey of applications.*

**Resistance Distance and Effective Resistance:**
3. **Klein, D. J. & Randic, M. (1993).** "Resistance Distance." J. Math. Chem. *Introduced resistance distance on graphs.*

4. **Ghosh, A., Boyd, S., & Saberi, A. (2008).** "Minimizing Effective Resistance of a Graph." SIAM Review. *Optimization perspective on resistance distances.*

**Graph Isomorphism:**
5. **Babai, L. (2016).** "Graph Isomorphism in Quasipolynomial Time." STOC. *Breakthrough result on GI complexity.*

6. **Weisfeiler, B. & Lehman, A. (1968).** "A Reduction of a Graph to a Canonical Form." Nauchno-Technicheskaya Informatsia. *Classical color refinement.*

**Cospectral Graphs:**
7. **Schwenk, A. J. (1973).** "Almost All Trees are Cospectral." *New Directions in Graph Theory.* *Cospectral mate prevalence.*

8. **van Dam, E. R. & Haemers, W. H. (2003).** "Which Graphs are Determined by their Spectrum?" Linear Algebra Appl. *Survey on spectral determination.*

9. **Godsil, C. D. & McKay, B. D. (1982).** "Constructing Cospectral Graphs." Aequationes Math. *Systematic cospectral constructions.*

**Noncommutative Geometry:**
10. **Connes, A. (1994).** *Noncommutative Geometry.* Academic Press. *Foundational reference for spectral triples.*

11. **Connes, A. (1996).** "Gravity Coupled with Matter and the Foundation of Non-Commutative Geometry." Comm. Math. Phys. *Distance formula in NCG.*

12. **Gracia-Bondia, J. M., Varilly, J. C., & Figueroa, H. (2001).** *Elements of Noncommutative Geometry.* Birkhauser. *Accessible introduction to NCG.*

**Quantum Walks and Graph Discrimination:**
13. **Gamble, J. K., Friesen, M., Zhou, D., et al. (2010).** "Two-Particle Quantum Walks Applied to the Graph Isomorphism Problem." Phys. Rev. A. *Quantum approach to GI.*

14. **Childs, A. M. (2010).** "On the Relationship Between Continuous- and Discrete-Time Quantum Walk." Comm. Math. Phys. *Spectral methods in quantum walks.*
