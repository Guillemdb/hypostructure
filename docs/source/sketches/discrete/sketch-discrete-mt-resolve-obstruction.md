---
title: "RESOLVE-Obstruction - Complexity Theory Translation"
---

# RESOLVE-Obstruction: Obstruction Capacity Collapse

## Overview

This document provides a complete complexity-theoretic translation of the RESOLVE-Obstruction theorem (Obstruction Capacity Collapse) from the hypostructure framework. The translation establishes a formal correspondence between the finiteness of obstruction sectors (analogous to Tate-Shafarevich groups in arithmetic geometry) and the **finite obstruction property** in complexity theory, most notably exemplified by the Robertson-Seymour Graph Minor Theorem.

**Original Theorem Reference:** {prf:ref}`mt-resolve-obstruction`

---

## Hypostructure Context

The RESOLVE-Obstruction theorem states that under appropriate capacity bounds, obstruction sectors must be finite. In arithmetic geometry, this corresponds to finiteness of the Tate-Shafarevich group $\text{III}(E/K)$; in the hypostructure framework, it guarantees that "bad configurations" cannot accumulate infinitely.

**Key Permits:**
- $K_{\mathrm{TB}+\mathrm{LS}}^{\mathcal{O}+}$: Non-degenerate duality pairing on obstruction sector
- $K_{C+\mathrm{Cap}}^{\mathcal{O}+}$: Compact sublevel sets under obstruction height
- $K_{\mathrm{SC}_\lambda}^{\mathcal{O}+}$: Subcritical accumulation of obstructions
- $K_{D_E}^{\mathcal{O}+}$: Subcritical obstruction dissipation

**Conclusion:** The obstruction sector $\mathcal{O}$ is finite-dimensional, with no runaway obstruction modes.

---

## Complexity Theory Statement

**Theorem (Finite Obstruction Principle).**
Let $\mathcal{P}$ be a hereditary property of finite structures (graphs, matroids, etc.) closed under taking minors/substructures. Then:

1. **Finite Characterization:** There exists a finite set of forbidden minors $\mathcal{F} = \{H_1, \ldots, H_k\}$ such that:
$$G \in \mathcal{P} \iff \forall H \in \mathcal{F}: H \not\preceq G$$
where $\preceq$ denotes the minor (or appropriate substructure) relation.

2. **Obstruction Finiteness:** The set $\mathcal{F}$ of minimal forbidden structures is finite.

3. **Algorithmic Consequence:** Membership in $\mathcal{P}$ is fixed-parameter tractable with parameter $|H|$ for each $H \in \mathcal{F}$.

**Paradigmatic Example (Robertson-Seymour, 1983-2004):**
For any minor-closed family $\mathcal{G}$ of graphs, the set of forbidden minors is finite. This resolved a 70-year conjecture (Wagner's Conjecture) and established that graph minor testing is in polynomial time.

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Equivalent | Formal Correspondence |
|-----------------------|------------------------------|------------------------|
| Obstruction sector $\mathcal{O}$ | Forbidden minor set $\mathcal{F}$ | Minimal witnesses to non-membership |
| Obstruction $x \in \mathcal{O}$ | Forbidden minor $H \in \mathcal{F}$ | Minimal structure violating property |
| Obstruction height $H_{\mathcal{O}}(x)$ | Size/complexity of forbidden structure | $|V(H)|$ or tree-width of $H$ |
| Capacity bound $\mathrm{Cap}_H^{\mathcal{O}}$ | Well-quasi-ordering (WQO) | No infinite antichains |
| Subcritical accumulation $\mathrm{SC}_\lambda^{\mathcal{O}}$ | Finite antichain property | Bounded obstruction count |
| Non-degenerate pairing $\langle\cdot,\cdot\rangle_{\mathcal{O}}$ | Robertson-Seymour structure theorem | Graph structure determines minor behavior |
| Compact sublevel sets | Finite graphs of bounded tree-width | Bounded complexity implies finiteness |
| Tate-Shafarevich finiteness | Robertson-Seymour finiteness | $|\text{III}| < \infty$ $\leftrightarrow$ $|\mathcal{F}| < \infty$ |
| Runaway obstruction mode | Infinite antichain | Would violate WQO |
| Obstruction dissipation $\mathfrak{D}_{\mathcal{O}}$ | Minor relation is a WQO | Infinite sequences have comparable elements |
| Structural detectability | Polynomial-time minor testing | Obstructions are algorithmically recognizable |
| Cartan's Theorems A/B | Kuratowski/Wagner theorems | Finite forbidden configurations |

---

## Logical Framework

### Well-Quasi-Orderings (WQO)

**Definition.** A quasi-order $(Q, \leq)$ is a **well-quasi-order** if:
1. $Q$ contains no infinite strictly descending chains
2. $Q$ contains no infinite antichains

Equivalently: every infinite sequence $q_1, q_2, q_3, \ldots$ contains $q_i \leq q_j$ for some $i < j$.

**Theorem (Higman 1952).** The subsequence embedding on finite words over a finite alphabet is a WQO.

**Theorem (Robertson-Seymour 2004).** The minor relation on finite graphs is a WQO.

### Forbidden Minor Characterization

**Definition.** For a quasi-order $(Q, \leq)$ and a downward-closed set $\mathcal{P} \subseteq Q$, define:
$$\mathrm{Obs}(\mathcal{P}) = \min(Q \setminus \mathcal{P})$$
the set of minimal elements not in $\mathcal{P}$.

**Lemma (WQO Obstruction Finiteness).** If $(Q, \leq)$ is a WQO and $\mathcal{P}$ is downward-closed, then $\mathrm{Obs}(\mathcal{P})$ is finite.

**Proof.** Elements of $\mathrm{Obs}(\mathcal{P})$ are pairwise incomparable (if $x \leq y$ with $x, y \in \mathrm{Obs}(\mathcal{P})$ and $x \neq y$, then $x \in \mathcal{P}$ by minimality of $y$, contradicting $x \in \mathrm{Obs}(\mathcal{P})$). By WQO, no infinite antichain exists, so $|\mathrm{Obs}(\mathcal{P})| < \infty$. $\square$

---

## Proof Sketch

### Setup: The Obstruction Correspondence

We establish the correspondence between hypostructure obstruction theory and complexity-theoretic forbidden structures:

| Hypostructure | Complexity Theory |
|---------------|-------------------|
| Hypostructure $\mathbb{H}$ | Structure class with quasi-order |
| Obstruction sector $\mathcal{O} \subset \mathcal{X}$ | Forbidden minor set $\mathcal{F}$ |
| Height functional $H_{\mathcal{O}}$ | Size function $|H|$ on structures |
| Pairing $\langle\cdot,\cdot\rangle_{\mathcal{O}}$ | Embedding/minor relation |
| Capacity bounds | WQO property |

### Step 1: Finiteness at Each Scale (Bounded Tree-Width)

**Claim:** For each bound $B$, the set of forbidden minors of size at most $B$ is finite.

**Proof (Complexity Version).**

Fix tree-width bound $k$. The class of graphs with tree-width $\leq k$ is characterized by a finite set of forbidden minors. Specifically:

- **Tree-width 1:** Forbidden minor $K_3$ (graphs with tw $\leq 1$ are forests)
- **Tree-width 2:** Forbidden minor $K_4$ (graphs with tw $\leq 2$ are series-parallel)
- **Tree-width $k$:** Finite set $\mathcal{F}_k$ of forbidden minors

**Grid Minor Theorem (Robertson-Seymour):** Every graph of tree-width $\geq f(k)$ contains the $k \times k$ grid as a minor. This provides the "compact sublevel set" property: bounded tree-width implies bounded structure.

**Correspondence to Hypostructure.** The capacity bound $K_{C+\mathrm{Cap}}^{\mathcal{O}+}$ states that sublevel sets $\{x : H_{\mathcal{O}}(x) \leq B\}$ are finite/compact. In graph theory, this translates to: graphs of bounded tree-width form a "compact" (well-structured, efficiently describable) class.

### Step 2: Uniform Bound on Obstruction Count (WQO Argument)

**Claim:** The total number of minimal obstructions is finite.

**Proof (Complexity Version).**

The Robertson-Seymour theorem proves that the graph minor relation is a WQO. For any minor-closed family $\mathcal{G}$:

1. Define $\mathcal{O} = \min(\overline{\mathcal{G}})$ (minimal non-members)
2. Elements of $\mathcal{O}$ form an antichain (pairwise incomparable under $\preceq$)
3. By WQO, $\mathcal{O}$ is finite

**Correspondence to Hypostructure.** The subcritical accumulation condition $K_{\mathrm{SC}_\lambda}^{\mathcal{O}+}$ states:
$$\sum_t w(t) \sum_{x \in \mathcal{O}_t} H_{\mathcal{O}}(x) < \infty$$

In graph-theoretic terms, if we weight obstructions by their size $|H|$, the WQO property ensures that the "weighted count" of distinct obstructions cannot diverge:
- Each obstruction $H$ contributes weight $|H|$
- Obstructions are pairwise incomparable (antichain)
- Finite antichain $\Rightarrow$ finite weighted sum

### Step 3: Global Finiteness (Robertson-Seymour Conclusion)

**Claim:** The complete obstruction set is finite-dimensional.

**Proof (Complexity Version).**

Combining Steps 1 and 2:

1. **Finite at each size:** For each $n$, there are finitely many forbidden minors on $n$ vertices (up to isomorphism)
2. **Bounded total count:** The antichain property limits the total number
3. **Global finiteness:** $|\mathcal{F}| < \infty$

**Explicit Examples:**

| Property $\mathcal{P}$ | Forbidden Minors $\mathcal{F}$ | $|\mathcal{F}|$ |
|------------------------|-------------------------------|-----------------|
| Forests | $\{K_3\}$ (triangle) | 1 |
| Outerplanar | $\{K_4, K_{2,3}\}$ | 2 |
| Series-parallel | $\{K_4\}$ | 1 |
| Planar | $\{K_5, K_{3,3}\}$ (Kuratowski) | 2 |
| Linklessly embeddable | Petersen family | 7 |
| Knotlessly embeddable | Heawood family | $>250$ |
| Tree-width $\leq k$ | $\mathcal{F}_k$ | Finite (unknown for $k \geq 4$) |
| Graphs embeddable in $\Sigma_g$ | $\mathcal{F}_g$ | Finite (grows with genus) |

### Step 4: No Runaway Modes (Antichain Boundedness)

**Claim:** There is no infinite sequence of increasingly complex obstructions.

**Proof (Complexity Version).**

Suppose $(H_n)_{n \geq 1}$ is an infinite sequence of forbidden minors with $|H_n| \to \infty$. Since each $H_n$ is a forbidden minor:
- $H_n \notin \mathcal{G}$ (not in the good class)
- $H_n$ is minimal: every proper minor of $H_n$ is in $\mathcal{G}$

If the $H_n$ are pairwise incomparable (an antichain), this contradicts the WQO property. Therefore, for some $i < j$: $H_i \preceq H_j$.

But then $H_i$ is a proper minor of $H_j$, so $H_i \in \mathcal{G}$ by minimality of $H_j$. This contradicts $H_i \in \mathcal{F}$.

**Conclusion:** No infinite sequence of forbidden minors exists. $\square$

**Correspondence to Hypostructure.** The dissipation bound $K_{D_E}^{\mathcal{O}+}$ ensures:
$$\mathfrak{D}_{\mathcal{O}}(x_n) \leq C \cdot H_{\mathcal{O}}(x_n)^{1-\delta}$$

This sublinear growth prevents accumulation of large obstructions. In graph theory, the WQO property plays this role: the minor relation "dissipates" potential infinite sequences by forcing comparabilities.

### Step 5: Structural Detectability (Algorithmic Membership)

**Claim:** Obstructions are algorithmically detectable.

**Proof (Complexity Version).**

**Theorem (Robertson-Seymour 1995).** For any fixed graph $H$, determining whether $H$ is a minor of $G$ is decidable in $O(n^3)$ time (where $n = |V(G)|$).

**Corollary.** For any minor-closed family $\mathcal{G}$ with forbidden minors $\mathcal{F} = \{H_1, \ldots, H_k\}$:
$$G \in \mathcal{G} \iff \bigwedge_{i=1}^{k} (H_i \not\preceq G)$$

This is decidable in $O(k \cdot n^3)$ time.

**Correspondence to Hypostructure.** The non-degeneracy condition $K_{\mathrm{TB}+\mathrm{LS}}^{\mathcal{O}+}$ ensures that obstructions are structurally identifiable. In the complexity setting, the Robertson-Seymour algorithm provides this detectability: each forbidden minor is "paired" with graphs that contain it via the minor testing algorithm.

---

## Certificate Construction

The proof produces explicit certificates analogous to the hypostructure obstruction certificates:

**Obstruction Certificate $K_{\mathrm{Obs}}^{\mathrm{finite}}$:**

```
K_Obs^finite := (
    obstruction_set:    F = {H_1, ..., H_k}         -- Finite forbidden minors
    height_function:    |H_i| for each H_i          -- Size of each obstruction
    antichain_witness:  proof that H_i incomparable -- WQO-derived finiteness
    detection_oracle:   O(n^3) minor testing        -- Algorithmic membership
)
```

**Example Certificate (Planarity):**

```
K_Obs^finite := (
    obstruction_set:    {K_5, K_{3,3}}
    height_function:    |K_5| = 5, |K_{3,3}| = 6
    antichain_witness:  K_5 and K_{3,3} are minor-incomparable
    detection_oracle:   Hopcroft-Tarjan planarity testing O(n)
)
```

**Usage in Algorithmic Verification:**

Given a graph $G$, to verify $G \in \mathcal{G}$:
1. For each $H \in \mathcal{F}$: run minor testing algorithm
2. If any test succeeds ($H \preceq G$): reject
3. If all tests fail: accept

**Complexity:** $O(|\mathcal{F}| \cdot n^3)$, polynomial for fixed property.

---

## Connections to Classical Results

### 1. Robertson-Seymour Graph Minor Theorem (1983-2004)

**Theorem (Graph Minor Theorem).** The set of finite graphs under the minor relation forms a well-quasi-order.

**Corollary.** Every minor-closed family of graphs has a finite forbidden minor characterization.

**Connection to RESOLVE-Obstruction:**

| RESOLVE-Obstruction | Robertson-Seymour |
|---------------------|-------------------|
| Obstruction sector $\mathcal{O}$ | Forbidden minors $\mathcal{F}$ |
| Finite-dimensionality | $|\mathcal{F}| < \infty$ |
| Capacity bound | WQO property |
| No runaway modes | No infinite antichains |
| Structural detectability | $O(n^3)$ minor testing |

**Scale of the Achievement:** The Robertson-Seymour proof spans 20 papers and over 500 pages, developing the Graph Structure Theorem as the key technical tool. This corresponds to the "non-degenerate pairing" in the hypostructure: the structure theorem provides deep understanding of how minors interact with graph structure.

### 2. Kuratowski's Theorem (1930)

**Theorem (Kuratowski).** A graph is planar if and only if it contains no subdivision of $K_5$ or $K_{3,3}$.

**Wagner's Reformulation (1937).** A graph is planar if and only if it has no $K_5$ or $K_{3,3}$ minor.

**Connection:** Kuratowski's theorem is the prototypical finite obstruction result. It predates the Robertson-Seymour theorem by 50+ years and handles the specific case of planarity with $|\mathcal{F}| = 2$.

**Cartan A/B Analogy:** Just as Cartan's Theorems A and B characterize coherent sheaf cohomology through finite conditions, Kuratowski's theorem characterizes planarity through finitely many forbidden structures.

### 3. Tate-Shafarevich Group Finiteness

**Conjecture (Tate, Shafarevich).** For an elliptic curve $E/K$ over a number field, the Tate-Shafarevich group $\text{III}(E/K)$ is finite.

**Partial Results:**
- **Kolyvagin (1990):** $\text{III}(E/\mathbb{Q})$ is finite when $E$ has analytic rank 0 or 1
- **Rubin (1991):** Finiteness for CM elliptic curves under certain conditions

**Connection to RESOLVE-Obstruction:**

| Tate-Shafarevich | Graph Minors |
|------------------|--------------|
| $\text{III}(E/K)$ elements | Forbidden minors |
| Cohomological obstruction | Embedding obstruction |
| Descent via height | Size bound on minors |
| Cassels-Tate pairing | Minor relation structure |
| Finiteness conjecture | Robertson-Seymour theorem |

The RESOLVE-Obstruction theorem abstracts the common pattern: both settings feature "obstructions" (to rational points / to property membership) that are controlled by structural bounds (height functions / tree-width) leading to finiteness.

### 4. Kruskal's Tree Theorem (1960)

**Theorem (Kruskal).** Finite trees under the topological minor (homeomorphic embedding) relation form a WQO.

**Connection:** Kruskal's theorem was a major precursor to Robertson-Seymour. It establishes finite obstruction sets for tree-like structures, which extend to:
- Series-parallel graphs
- Graphs of bounded path-width
- Tree decomposable structures

### 5. Higman's Lemma (1952)

**Theorem (Higman).** For any finite alphabet $\Sigma$, the set $\Sigma^*$ of finite words under subsequence embedding is a WQO.

**Consequence:** Any language closed under taking subsequences has a finite set of forbidden patterns.

**Connection:** Higman's lemma is the combinatorial foundation underlying many WQO results. The proof technique (minimal bad sequence argument) directly corresponds to the "no runaway modes" step in RESOLVE-Obstruction.

### 6. Fellows-Langston Framework

**Theorem (Fellows-Langston 1988).** For any minor-closed graph property $\mathcal{G}$:
1. $\mathcal{G}$ has finite forbidden minors (by Robertson-Seymour)
2. Membership in $\mathcal{G}$ is decidable in $O(n^3)$ time
3. Many $\mathcal{G}$-problems are fixed-parameter tractable

**Non-Constructivity Caveat:** The Robertson-Seymour theorem is non-constructive: it proves $|\mathcal{F}| < \infty$ without providing $\mathcal{F}$ explicitly. For many properties, the forbidden minors are unknown.

**Connection:** This highlights a gap between the hypostructure certificate $K_{\mathrm{Obs}}^{\mathrm{finite}}$ and algorithmic practicality. The certificate guarantees finiteness but may not be effectively computable.

---

## Quantitative Bounds

### Size of Forbidden Minor Sets

For a minor-closed family $\mathcal{G}$, let $f(\mathcal{G}) = |\mathcal{F}|$ denote the number of forbidden minors.

**Known Bounds:**

| Property | $f(\mathcal{G})$ | Largest Forbidden Minor |
|----------|------------------|------------------------|
| Forests | 1 | $K_3$ (3 vertices) |
| Outerplanar | 2 | $K_4$ (4 vertices) |
| Planar | 2 | $K_{3,3}$ (6 vertices) |
| $\Sigma_1$-embeddable (torus) | $>17{,}000$ | Unknown |
| Linkless embedding | 7 | Petersen family (~10 vertices) |
| Knotless embedding | $>250$ | Unknown |
| Tree-width $\leq 3$ | 4 | Prism, $K_5$, etc. |
| Tree-width $\leq k$ | $\leq 2^{2^{O(k^5)}}$ | Doubly exponential in $k$ |

### Tree-Width Bounds on Obstructions

**Theorem (Robertson-Seymour).** The forbidden minors for tree-width $\leq k$ have tree-width exactly $k+1$.

**Theorem (Lagergren 1998).** The forbidden minors for tree-width $\leq k$ have at most $2^{2^{O(k^5)}}$ vertices.

### Algorithmic Complexity

**Minor Testing:**
- General graphs: $O(n^3)$ (Robertson-Seymour)
- Planar graphs: $O(n)$ (Hopcroft-Tarjan for planarity)
- Bounded tree-width: $O(n)$ (Courcelle's theorem + tree decomposition)

**Property Membership:**
$$T(\mathcal{G}, n) = O(|\mathcal{F}| \cdot n^3)$$

For fixed $\mathcal{G}$, this is polynomial. The constant $|\mathcal{F}|$ may be astronomical but is independent of input size.

---

## The Finite Obstruction Meta-Theorem

**Meta-Theorem (Finite Obstruction Principle).**
Let $(Q, \preceq)$ be a well-quasi-ordered set of finite structures. For any hereditary property $\mathcal{P} \subseteq Q$:

1. **Finite Characterization:** $\mathcal{P}$ is characterized by finitely many forbidden minimal structures

2. **Algorithmic Membership:** If $\preceq$-testing is decidable, then $\mathcal{P}$-membership is decidable

3. **Certificate Structure:**
$$K_{\mathrm{Obs}}^{\mathrm{finite}} = (\mathcal{F}, |\cdot|, \text{antichain}, \text{oracle})$$

**Instances:**

| WQO Setting | Relation $\preceq$ | Applications |
|-------------|-------------------|--------------|
| Finite words | Subsequence | Regex, pattern avoidance |
| Finite trees | Topological minor | Parse trees, XML schemas |
| Finite graphs | Graph minor | Network properties |
| Matroids | Matroid minor | Linear algebra, codes |
| Permutations | Pattern containment | Sorting, stack-sorting |

---

## Non-Constructivity and Algorithmic Gaps

### The Non-Constructivity Problem

**Observation.** The Robertson-Seymour theorem proves $|\mathcal{F}| < \infty$ without providing an algorithm to compute $\mathcal{F}$.

**Consequence:** For many minor-closed properties:
- We know forbidden minors exist
- We do not know what they are
- The $O(n^3)$ algorithm is not truly practical

**Example:** For graphs embeddable on the torus ($\Sigma_1$), we know:
- $|\mathcal{F}|$ is finite (Robertson-Seymour)
- $|\mathcal{F}| > 17{,}000$ (partial enumeration)
- The complete list is unknown

### Effective vs. Non-Effective Finiteness

The RESOLVE-Obstruction theorem guarantees:
- **Effective finiteness** when the obstruction sector is explicitly computable
- **Non-effective finiteness** when only existence is proven

**Correspondence:**

| Hypostructure Certificate | Algorithmic Status |
|---------------------------|-------------------|
| $K_{\mathrm{Obs}}^{\mathrm{finite}}$ with explicit $\mathcal{O}$ | Constructive finite obstruction |
| $K_{\mathrm{Obs}}^{\mathrm{finite}}$ existence proof only | Non-constructive finiteness |

### Computability of Obstruction Sets

**Question:** Given a minor-closed property $\mathcal{G}$ (specified by a decision algorithm), can we compute $\mathcal{F}$?

**Answer:** Surprisingly, yes (but impractically):

**Theorem (Fellows-Langston 1988).** There exists an algorithm that, given oracle access to $\mathcal{G}$-membership, computes the forbidden minor set $\mathcal{F}$.

**Proof Sketch:** Enumerate all graphs by size. For each graph $G$:
1. Test if $G \in \mathcal{G}$
2. If no, check if all proper minors are in $\mathcal{G}$
3. If yes to both, add $G$ to $\mathcal{F}$
4. By WQO, this terminates

**Complexity:** Astronomically large; not practical.

---

## Summary

The RESOLVE-Obstruction theorem, translated to complexity theory, establishes:

1. **Finite Forbidden Structures:** Hereditary properties closed under minors/substructures have finite characterizations by forbidden minimal elements.

2. **WQO Foundation:** The well-quasi-ordering property (no infinite antichains) is the structural foundation ensuring obstruction finiteness.

3. **Algorithmic Tractability:** Finite obstruction sets yield polynomial-time membership algorithms (given the obstruction set).

4. **Non-Constructivity Gap:** Finiteness proofs may not yield explicit obstructions, creating a gap between theoretical tractability and practical algorithms.

**The Central Correspondence:**

| RESOLVE-Obstruction | Finite Obstruction Principle |
|---------------------|------------------------------|
| Obstruction sector finite | Forbidden minor set finite |
| Capacity bound | WQO property |
| No runaway modes | No infinite antichains |
| Height function | Structure size |
| Pairing non-degeneracy | Robertson-Seymour structure theorem |
| Cartan A/B analogy | Kuratowski/Wagner theorems |
| Tate-Shafarevich finiteness | Graph Minor Theorem |

**Key Insight:** The hypostructure framework reveals that the Robertson-Seymour theorem and Tate-Shafarevich finiteness share a common structural pattern: both concern "obstructions" (forbidden minors / cohomological elements) that are controlled by "capacity bounds" (WQO / height functions) leading to finiteness. The RESOLVE-Obstruction theorem abstracts this pattern, providing a unified framework for understanding finite obstruction phenomena across mathematics and computer science.

---

## Literature

1. **Robertson, N. & Seymour, P. D. (1983-2004).** "Graph Minors I-XXIII." *Journal of Combinatorial Theory Series B.* *The complete Graph Minor Theorem proof.*

2. **Kuratowski, K. (1930).** "Sur le probleme des courbes gauches en topologie." *Fundamenta Mathematicae.* *Forbidden subdivision characterization of planarity.*

3. **Wagner, K. (1937).** "Uber eine Eigenschaft der ebenen Komplexe." *Mathematische Annalen.* *Minor reformulation of Kuratowski's theorem.*

4. **Kruskal, J. B. (1960).** "Well-quasi-ordering, the Tree Theorem, and Vazsonyi's conjecture." *Transactions AMS.* *WQO for finite trees.*

5. **Higman, G. (1952).** "Ordering by divisibility in abstract algebras." *Proceedings London Math Soc.* *WQO for finite words.*

6. **Fellows, M. R. & Langston, M. A. (1988).** "Nonconstructive tools for proving polynomial-time decidability." *JACM.* *Algorithmic implications of Robertson-Seymour.*

7. **Kolyvagin, V. A. (1990).** "Euler systems." *The Grothendieck Festschrift.* *Finiteness of Tate-Shafarevich groups.*

8. **Rubin, K. (1991).** "The main conjectures of Iwasawa theory for imaginary quadratic fields." *Inventiones mathematicae.* *Tate-Shafarevich finiteness results.*

9. **Diestel, R. (2017).** *Graph Theory (5th edition).* Springer. *Modern treatment including Graph Minor Theorem.*

10. **Lovasz, L. (2006).** "Graph minor theory." *Bulletin AMS.* *Survey of Robertson-Seymour theory.*

11. **Downey, R. G. & Fellows, M. R. (1999).** *Parameterized Complexity.* Springer. *FPT consequences of finite obstruction sets.*

12. **Cartan, H. & Serre, J.-P. (1953).** "Un theoreme de finitude concernant les varietes analytiques compactes." *C. R. Acad. Sci. Paris.* *Cartan's Theorems A/B (hypostructure source).*
