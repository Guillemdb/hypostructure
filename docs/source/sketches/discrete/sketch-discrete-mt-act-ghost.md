---
title: "ACT-Ghost - Complexity Theory Translation"
---

# ACT-Ghost: Gauge Redundancy and Canonical Form Computation

## Overview

This document provides a complete complexity-theoretic translation of the ACT-Ghost metatheorem (Derived Extension / BRST) from the hypostructure framework. The theorem establishes that ghost field extension via BRST cohomology resolves gauge redundancy singularities by introducing auxiliary variables that cancel divergent determinants. In computational terms, this corresponds to **Gauge Redundancy Resolution**: quotient structures and canonical forms eliminate computational redundancy arising from symmetry.

**Original Theorem Reference:** {prf:ref}`mt-act-ghost`

**Central Translation:** Ghost field extension (BRST cohomology) resolves gauge redundancy singularities $\longleftrightarrow$ **Canonical Form Computation**: Equivalence classes and gauge fixing eliminate redundant computation.

---

## Complexity Theory Statement

**Theorem (Canonical Form and Redundancy Elimination, Computational Form).**
Let $\mathcal{X}$ be a computational space with symmetry group $G$ acting on it. The naive enumeration $\sum_{x \in \mathcal{X}} f(x)$ overcounts by the orbit size $|G \cdot x|$. Introduce **canonical form computation** to obtain:

$$\sum_{x \in \mathcal{X}/G} f([x]) = \frac{1}{|G|} \sum_{x \in \mathcal{X}} f(x) \cdot w(x)$$

where $w(x)$ is a weight function (the computational analogue of Faddeev-Popov determinant).

**Input**: Space $\mathcal{X}$ with $G$-action + admissibility certificate (group action is well-defined)

**Output**:
- Canonical representative function $\text{canon}: \mathcal{X} \to \mathcal{X}/G$
- Orbit-counting weight $w(x) = |\text{Stab}(x)|^{-1}$
- Certificate that computation is consistent across equivalence classes

**Guarantees**:
1. **Well-definedness**: Functions on $\mathcal{X}/G$ are independent of representative choice
2. **Complexity control**: Canonical form computation bounded by orbit structure
3. **Certificate production**: Witness that quotient is correctly computed
4. **Redundancy elimination**: Each equivalence class counted exactly once

**Formal Statement.** Let $G$ be a finite group acting on finite set $\mathcal{X}$. For any $G$-invariant function $f: \mathcal{X} \to R$:

1. **Quotient Exists:** The orbit space $\mathcal{X}/G$ is well-defined with $|\mathcal{X}/G| = |\mathcal{X}|/|G|$ (for free actions)

2. **Canonical Form Complexity:** Computing a canonical representative satisfies:
   $$T(\text{canon}(x)) \leq O(|G| \cdot T(\text{compare}))$$
   where $T(\text{compare})$ is the time to compare two elements

3. **Gauge Fixing:** Choosing a section $s: \mathcal{X}/G \to \mathcal{X}$ (gauge-fixing) enables:
   $$\sum_{[x] \in \mathcal{X}/G} f(s([x])) = \sum_{x \in \mathcal{X}} f(x) \cdot \delta_{\text{gauge}}(x)$$
   where $\delta_{\text{gauge}}(x) = 1$ iff $x = s([x])$

4. **BRST Cohomology Analogue:** Physical observables (well-defined functions) are exactly $G$-invariant functions:
   $$\mathcal{O}_{\text{phys}} = \{f : \mathcal{X} \to R \mid f(g \cdot x) = f(x) \; \forall g \in G\}$$

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Equivalent | Formal Correspondence |
|-----------------------|------------------------------|------------------------|
| Connection space $\mathcal{A}$ | Configuration space $\mathcal{X}$ | Space of all problem instances |
| Gauge group $\mathcal{G}$ | Symmetry group $G$ | Automorphism group $\text{Aut}(\mathcal{X})$ |
| Gauge orbit $\mathcal{G} \cdot A$ | Equivalence class $[x] = G \cdot x$ | Orbit under group action |
| Infinite orbit volume | Redundant enumeration | Overcounting symmetric configurations |
| Ghost fields $(c, \bar{c})$ | Auxiliary variables | Orbit representatives, canonical forms |
| Opposite statistics | Cancellation mechanism | Inclusion-exclusion, Mobius inversion |
| Faddeev-Popov determinant | Orbit stabilizer weight | $|\text{Stab}(x)|^{-1}$ |
| Gauge-fixing function $F(A) = 0$ | Canonical form selection | Lexicographically smallest representative |
| BRST charge $s$ | Coboundary operator | $\delta: C^k(G) \to C^{k+1}(G)$ |
| BRST cohomology $H^*_s$ | Invariant functions | $H^0(G, \mathcal{F}) = \mathcal{F}^G$ |
| Nilpotent $s^2 = 0$ | $\delta^2 = 0$ | Cochain complex property |
| Physical observables | $G$-invariant computations | Functions constant on orbits |
| Stiffness restoration | Unique representative | Well-defined quotient |
| Capacity cancellation | Orbit size correction | Burnside's lemma application |
| Path integral $\int \mathcal{D}A$ | Summation $\sum_{x \in \mathcal{X}}$ | Enumeration over configurations |
| Regularized path integral | Orbit-weighted sum | $\sum_{[x]} f([x])$ |
| Certificate $(s, H^*_s, c, \bar{c})$ | Certificate $(\text{canon}, G, \text{Stab})$ | Canonical form algorithm + stabilizer data |

---

## Gauge Redundancy as Computational Overcounting

### The Categorical Framework

**Definition (Gauge Redundancy as Symmetry).** In computational problems, a **gauge redundancy** arises when multiple inputs represent the same abstract object:

- **Graph isomorphism**: Two adjacency matrices represent the same graph
- **Polynomial representation**: $x^2 - 1$ and $(x-1)(x+1)$ represent the same polynomial
- **Linear algebra**: Row-equivalent matrices represent the same row space
- **SAT**: Permuting variable names gives equivalent formulas

**Observation (Redundancy = Computational Detour):** Naive algorithms that process each representation separately waste computation:
- **Naive enumeration**: $O(|\mathcal{X}|)$ work
- **Quotient enumeration**: $O(|\mathcal{X}/G|) = O(|\mathcal{X}|/|G|)$ work (for free actions)

The ratio $|G|$ represents the **divergent factor** that must be cancelled.

### Faddeev-Popov as Orbit Counting

**Definition (Computational Faddeev-Popov).** The gauge-fixing procedure corresponds to:

1. **Gauge choice**: Select canonical representative function $\text{canon}: \mathcal{X} \to \mathcal{X}$
2. **Indicator function**: $\delta_{\text{gauge}}(x) = \mathbf{1}[x = \text{canon}([x])]$
3. **Determinant/Weight**: $w(x) = |\{g \in G : g \cdot x = x\}|^{-1} = |\text{Stab}(x)|^{-1}$

**Faddeev-Popov Identity (Computational Form):**
$$\sum_{x \in \mathcal{X}} f(x) = |G| \cdot \sum_{[x] \in \mathcal{X}/G} f(\text{canon}([x]))$$

when $f$ is $G$-invariant. For non-free actions:
$$\sum_{x \in \mathcal{X}} f(x) = \sum_{[x] \in \mathcal{X}/G} |G \cdot x| \cdot f(\text{canon}([x]))$$

### Ghost Fields as Auxiliary Computation

**Definition (Computational Ghost).** In the BRST formalism, ghost fields $(c, \bar{c})$ are Grassmann (anticommuting) variables. The computational analogue is **signed inclusion-exclusion counting**:

- **Bosonic contribution** (positive): Direct enumeration
- **Fermionic/ghost contribution** (negative): Subtraction of overcounted orbits
- **Net effect**: Exact orbit counting via cancellation

**Example (Burnside's Lemma as BRST):**

Burnside's lemma:
$$|\mathcal{X}/G| = \frac{1}{|G|} \sum_{g \in G} |\mathcal{X}^g|$$

where $\mathcal{X}^g = \{x : g \cdot x = x\}$ is the fixed-point set.

This corresponds to:
- **Ghost insertion**: Sum over group elements $g$
- **Determinant**: Fixed-point count $|\mathcal{X}^g|$
- **Normalization**: Division by $|G|$ (gauge orbit volume)

---

## Proof Sketch: BRST = Canonical Form Computation

### Setup: Quotient Computation Model

**Definitions (Equivalence and Canonical Forms):**

1. **Equivalence relation**: $x \sim y$ iff $\exists g \in G: g \cdot x = y$
2. **Equivalence class**: $[x] = \{g \cdot x : g \in G\}$
3. **Canonical form**: Function $\text{canon}: \mathcal{X} \to \mathcal{X}$ with $\text{canon}(x) \sim x$ and $\text{canon}(x) = \text{canon}(y)$ iff $x \sim y$
4. **Gauge fixing**: Section $s: \mathcal{X}/G \to \mathcal{X}$ with $\pi \circ s = \text{id}$ where $\pi: \mathcal{X} \to \mathcal{X}/G$

**Complexity Measures:**

| Measure | Definition | Role |
|---------|------------|------|
| Orbit size $|G \cdot x|$ | Size of equivalence class | Gauge orbit volume |
| Stabilizer size $|\text{Stab}(x)|$ | $|\{g : g \cdot x = x\}|$ | Faddeev-Popov weight |
| Canonical form time | $T(\text{canon}(x))$ | Gauge-fixing cost |
| Isomorphism test time | $T(x \sim y)$ | Equivalence checking |

---

### Step 1: Gauge Orbit Volume = Overcounting Factor

**Claim.** Naive enumeration overcounts by the orbit size.

**Proof.**

For a $G$-invariant function $f$:
$$\sum_{x \in \mathcal{X}} f(x) = \sum_{[x] \in \mathcal{X}/G} \sum_{y \in [x]} f(y) = \sum_{[x] \in \mathcal{X}/G} |G \cdot x| \cdot f([x])$$

For free actions ($\text{Stab}(x) = \{e\}$ for all $x$):
$$\sum_{x \in \mathcal{X}} f(x) = |G| \cdot \sum_{[x] \in \mathcal{X}/G} f([x])$$

The factor $|G|$ is the **infinite volume** that causes the path integral to diverge.

**Computational Interpretation:**
- Naive graph counting: $n!$ adjacency matrices per graph
- Naive polynomial counting: Multiple coefficient representations
- Redundancy factor: Group order $|G|$ $\square$

---

### Step 2: Ghost Fields = Inclusion-Exclusion Correction

**Claim.** Ghost field contributions implement Mobius inversion / inclusion-exclusion.

**Proof.**

Consider the poset of subgroups $H \leq G$. The Mobius function $\mu(H, G)$ on this poset satisfies:
$$\sum_{H \leq K \leq G} \mu(H, K) = \delta_{H,G}$$

**BRST Translation:**

The BRST differential $s$ acts on the group cohomology complex:
$$C^0 \xrightarrow{s} C^1 \xrightarrow{s} C^2 \xrightarrow{s} \cdots$$

where $C^k = \text{Maps}(G^k, R)$.

The cohomology $H^0(G, R) = R^G$ consists of $G$-invariant elements---the "physical observables."

**Ghost Contribution:**
- **Ghost $c$** (degree $+1$): Inserts group element dependence
- **Antighost $\bar{c}$** (degree $-1$): Removes group element dependence
- **Net effect**: Projection onto $G$-invariant subspace via alternating sum

This is exactly inclusion-exclusion: positive terms for even degree, negative for odd. $\square$

---

### Step 3: Gauge Fixing = Canonical Representative Selection

**Claim.** Gauge-fixing function $F(A) = 0$ corresponds to canonical form algorithm.

**Proof.**

**Standard Gauge Choices (Computational Analogues):**

| Physics Gauge | Computational Canonical Form |
|---------------|------------------------------|
| Lorenz gauge $\partial_\mu A^\mu = 0$ | Reduced row echelon form |
| Coulomb gauge $\nabla \cdot \mathbf{A} = 0$ | Lexicographically smallest permutation |
| Axial gauge $A_3 = 0$ | Standard form with leading coefficient 1 |
| Temporal gauge $A_0 = 0$ | Monic polynomial representation |

**Canonical Form Algorithm Structure:**

```
CanonicalForm(x, G):
    Input: Element x, symmetry group G
    Output: Canonical representative canon(x)

    1. Compute orbit: O = {g · x : g ∈ G}
    2. Apply ordering: Sort O by comparison function
    3. Return: min(O) under ordering
```

**Complexity:**
$$T(\text{canon}) = O(|G| \cdot T(\text{action}) + |G| \log|G| \cdot T(\text{compare}))$$

For efficient groups (polynomial orbit enumeration):
$$T(\text{canon}) = O(\text{poly}(n))$$

**Key Property:** The gauge-fixing function $F$ is non-degenerate iff the canonical form is well-defined:
$$F(A) = 0 \text{ has unique solution per orbit} \iff \text{canon is injective on } \mathcal{X}/G$$ $\square$

---

### Step 4: BRST Cohomology = Invariant Functions

**Claim.** Physical observables $H^0_s(X_{\text{BRST}})$ correspond to $G$-invariant functions.

**Proof.**

**BRST Complex (Computational Version):**

Define the cochain complex:
$$C^0(\mathcal{X}) \xrightarrow{\delta} C^1(\mathcal{X} \times G) \xrightarrow{\delta} C^2(\mathcal{X} \times G^2) \xrightarrow{\delta} \cdots$$

where:
- $C^0(\mathcal{X}) = \{f: \mathcal{X} \to R\}$ (functions on configurations)
- $(\delta f)(x, g) = f(g \cdot x) - f(x)$ (coboundary)

**Cohomology:**
$$H^0 = \ker(\delta) = \{f : f(g \cdot x) = f(x) \; \forall g\} = \mathcal{O}^G$$

The zeroth cohomology is exactly the space of $G$-invariant functions.

**Physical Observable = Well-Defined on Quotient:**

A function $f: \mathcal{X} \to R$ descends to $\bar{f}: \mathcal{X}/G \to R$ iff $f \in H^0$:
$$f(x) = f(y) \text{ whenever } [x] = [y]$$

**Certificate Verification:**
- Check: $f(g \cdot x) = f(x)$ for generators of $G$
- Time: $O(|\text{gen}(G)| \cdot T(f))$
- This is the computational analogue of "BRST-closed" verification $\square$

---

### Step 5: Stiffness Restoration = Well-Defined Quotient

**Claim.** The Hessian $\nabla^2 \Phi_{\text{tot}}$ becoming non-degenerate corresponds to the quotient map being a covering map.

**Proof.**

**Degeneracy in Naive Computation:**

Without gauge-fixing, the "Hessian" of the counting functional is degenerate:
$$\text{rank}(\nabla^2 N) < \dim(\mathcal{X})$$

because directions along gauge orbits contribute zero second derivative.

**Restoration via Gauge-Fixing:**

After gauge-fixing:
$$\nabla^2 N_{\text{gauge-fixed}} = \nabla^2 N|_{\text{transverse}} + M_{\text{FP}}$$

where $M_{\text{FP}}$ (Faddeev-Popov operator) contributes along gauge directions.

**Computational Analogue:**

The quotient map $\pi: \mathcal{X} \to \mathcal{X}/G$ is a covering map (locally invertible) when:
1. Action is free (or effectively free for computational purposes)
2. Canonical form is efficiently computable
3. Equivalence testing is decidable

This ensures that optimization/search over $\mathcal{X}/G$ is well-defined. $\square$

---

## Connections to Canonical Form Computation

### 1. Graph Canonization (Nauty/Bliss)

**Classical Result.** Given a graph $G$ on $n$ vertices, compute a canonical labeling in quasi-polynomial time.

**Connection to ACT-Ghost:**
- **Gauge group**: $S_n$ (vertex permutations)
- **Configuration space**: Adjacency matrices
- **Gauge fixing**: Canonical labeling algorithm
- **Ghost fields**: Automorphism group computation (required for weight)

**McKay's Algorithm Structure:**
```
GraphCanon(G):
    1. Compute automorphism group Aut(G)     [Ghost field computation]
    2. Build search tree with pruning        [Gauge fixing]
    3. Return lexicographically smallest     [Canonical representative]
```

**Complexity:**
- Worst case: $2^{O(\sqrt{n \log n})}$ (Babai 2016)
- Practical: Often polynomial via symmetry exploitation

### 2. String Isomorphism and Word Problems

**Classical Result.** Given strings $s, t$, determine if $s \sim t$ under rotation/reflection.

**Connection to ACT-Ghost:**
- **Gauge group**: Cyclic group $\mathbb{Z}_n$ (rotations) or dihedral $D_n$
- **Canonical form**: Lexicographically smallest rotation (Booth's algorithm, $O(n)$)
- **Ghost contribution**: String period structure

**Canonical Rotation Algorithm:**
```
CanonicalRotation(s):
    1. Construct s·s (doubled string)
    2. Find lexicographically minimal substring of length n
    3. Return this rotation
```

### 3. Polynomial Canonical Forms

**Classical Result.** Represent polynomials in canonical form (monic, reduced, etc.).

| Polynomial Ring | Canonical Form | Gauge Group |
|-----------------|----------------|-------------|
| $\mathbb{Z}[x]$ | Monic or primitive | $\{1, -1\}$ |
| $\mathbb{Q}[x]$ | Monic with integer coefficients | $\mathbb{Q}^*$ scaling |
| $\mathbb{F}_q[x]$ | Monic | $\mathbb{F}_q^*$ |
| Multivariate | Groebner basis | Ideal equivalence |

**Groebner Basis as Gauge Fixing:**
- **Configuration**: Polynomial ideal $I$
- **Gauge group**: Representation equivalence (same ideal)
- **Canonical form**: Reduced Groebner basis (unique for fixed term order)
- **Ghost/determinant**: S-polynomial reduction (ensures termination)

### 4. Linear Algebra Canonical Forms

| Matrix Space | Canonical Form | Gauge Group |
|--------------|----------------|-------------|
| $M_n(\mathbb{F})$ | Jordan/Rational form | Similarity $PMP^{-1}$ |
| Symmetric matrices | Diagonal | Orthogonal $O^T M O$ |
| Integer matrices | Smith normal form | $GL_n(\mathbb{Z})$ equivalence |
| Row space | RREF | Row operations |

**Jordan Form as BRST Fixed Point:**
- **BRST charge**: Conjugation action $s(M) = [X, M]$ for $X \in \mathfrak{gl}_n$
- **Cohomology**: Matrices commuting with a fixed Jordan form
- **Physical observables**: Spectral invariants (eigenvalues, characteristic polynomial)

---

## Connections to Isomorphism Problems

### The Isomorphism Hierarchy

| Problem | Group $G$ | Complexity | Canonical Form |
|---------|----------|------------|----------------|
| String equality | Trivial | $O(n)$ | Identity |
| String rotation | $\mathbb{Z}_n$ | $O(n)$ | Booth's algorithm |
| Permutation conjugacy | $S_n$ | $O(n)$ | Cycle type |
| Matrix similarity | $GL_n$ | $O(n^\omega)$ | Rational/Jordan form |
| Graph isomorphism | $S_n$ (restricted) | Quasi-poly | Nauty/Traces |
| Code equivalence | Monomial | NP-hard in general | Systematic form |
| Group isomorphism | Varies | In NP | Presentation reduction |

### Weisfeiler-Leman as Iterative Gauge Fixing

The $k$-dimensional Weisfeiler-Leman algorithm iteratively refines vertex colorings:
$$\chi^{(t+1)}(v_1, \ldots, v_k) = \text{hash}\left(\chi^{(t)}(v_1, \ldots, v_k), \{\!\{\chi^{(t)}(\ldots)\}\!\}\right)$$

**BRST Interpretation:**
- **Initial coloring**: Trivial gauge (all vertices equivalent)
- **Refinement step**: Partial gauge fixing (coarser quotient)
- **Fixed point**: Canonical coloring (finest quotient detectable by WL)
- **Failure modes**: Non-isomorphic graphs with same WL-coloring (CFI graphs)

**Ghost Field Analogue:**
The multiset $\{\!\{\ldots\}\!\}$ of neighbor colorings acts as a "determinant" that distinguishes orbits.

---

## Certificate Construction

**Gauge Redundancy Resolution Certificate:**

```
K_Ghost = {
    mode: "Derived_Extension",
    mechanism: "Canonical_Form",

    gauge_structure: {
        space: X (configuration space),
        group: G (symmetry group),
        action: (g, x) ↦ g · x,
        orbit_structure: {X/G, π: X → X/G}
    },

    ghost_fields: {
        c: orbit_indicator (selects canonical representative),
        c_bar: stabilizer_weight (|Stab(x)|^{-1}),
        nilpotent: δ² = 0 (coboundary property)
    },

    gauge_fixing: {
        function: canon: X → X,
        uniqueness: canon(x) = canon(y) ⟺ x ~ y,
        section: s: X/G → X with π ∘ s = id,
        complexity: T(canon) ≤ O(poly(|x|))
    },

    BRST_cohomology: {
        H^0: G-invariant functions,
        physical_observables: {f : f(g·x) = f(x)},
        projection: P_G = (1/|G|) Σ_{g∈G} ρ(g)
    },

    stiffness_restoration: {
        pre: Hessian degenerate along orbits,
        post: Quotient Hessian non-degenerate,
        FP_determinant: det(M_FP) = |Stab(x)|
    },

    capacity_cancellation: {
        naive_count: |X|,
        corrected_count: |X/G|,
        burnside: (1/|G|) Σ_g |X^g|
    },

    certificate: {
        canonical_form: canon(x),
        equivalence_witness: g with g · x = y (if x ~ y),
        non_equivalence: distinct canon values,
        verification: O(|gen(G)| · T(action))
    }
}
```

---

## Quantitative Summary

| Property | Bound |
|----------|-------|
| Orbit enumeration | $O(\|G\| \cdot T(\text{action}))$ |
| Canonical form (naive) | $O(\|G\| \cdot T(\text{compare}))$ |
| Canonical form (graph) | $2^{O(\sqrt{n \log n})}$ (Babai) |
| Canonical form (string rotation) | $O(n)$ (Booth) |
| Invariant verification | $O(\|\text{gen}(G)\| \cdot T(f))$ |
| Orbit counting (Burnside) | $O(\|G\| \cdot T(\text{fixed-point}))$ |

### Complexity Class Connections

| Gauge Structure | Canonical Form Complexity | Notes |
|-----------------|---------------------------|-------|
| Trivial group | $O(1)$ (identity) | No redundancy |
| Cyclic $\mathbb{Z}_n$ | $O(n)$ | String rotation |
| Symmetric $S_n$ | Quasi-polynomial | Graph isomorphism |
| $GL_n(\mathbb{F}_q)$ | $O(n^\omega)$ | Matrix forms |
| Infinite group | Undecidable in general | Word problem |

---

## Extended Connections

### 1. Polya Enumeration as Gauge-Fixed Counting

**Polya's Theorem:** Count distinct objects under group action:
$$|F/G| = \frac{1}{|G|} \sum_{g \in G} |F^g|$$

This is exactly the BRST partition function:
$$Z = \int \mathcal{D}A \mathcal{D}c \mathcal{D}\bar{c} \, e^{-S_{\text{tot}}} \longleftrightarrow |F/G| = \frac{1}{|G|} \sum_g |\text{Fix}(g)|$$

### 2. Constraint Satisfaction and Symmetry Breaking

In CSP solving, symmetry breaking adds constraints to prune symmetric solutions:
- **Lex-leader**: Require solution to be lexicographically smallest in orbit
- **SBDD**: Symmetry-breaking during search via dominance detection

This is computational gauge-fixing: selecting one representative per equivalence class.

### 3. Database Query Optimization

Query equivalence under schema automorphisms:
- **Gauge group**: Attribute permutations preserving functional dependencies
- **Canonical form**: Query rewriting to normal form
- **Ghost contribution**: Chase procedure for constraint propagation

### 4. Code Equivalence in Coding Theory

Two linear codes are equivalent if related by monomial transformation:
- **Gauge group**: Monomial group $\text{Mon}_n(\mathbb{F}_q)$
- **Canonical form**: Systematic form or reduced echelon
- **Invariants**: Weight distribution (BRST-closed observable)

---

## Conclusion

The ACT-Ghost theorem translates to complexity theory as **Canonical Form Computation and Gauge Redundancy Elimination**:

1. **Ghost Fields = Auxiliary Computation:** Orbit representatives and stabilizer weights.

2. **BRST Cohomology = Invariant Functions:** Well-defined operations on equivalence classes.

3. **Gauge Fixing = Canonical Form:** Selecting unique representatives eliminates redundancy.

4. **Stiffness Restoration = Well-Defined Quotient:** The quotient space admits efficient computation.

5. **Capacity Cancellation = Orbit Counting:** Burnside/Polya enumeration corrects for overcounting.

**Physical Interpretation (Computational Analogue):**

- **Gauge orbit** = Equivalence class under symmetry
- **Ghost fields** = Canonical form algorithm + stabilizer data
- **BRST charge** = Coboundary operator testing invariance
- **Physical observable** = Function constant on orbits (isomorphism invariant)

**The Ghost Extension Certificate:**

$$K_{\text{Ghost}}^+ = \begin{cases}
\text{canon}: \mathcal{X} \to \mathcal{X} & \text{canonical form function} \\
G, \text{Stab}(x) & \text{group and stabilizer data} \\
H^0 = \mathcal{O}^G & \text{invariant functions (observables)} \\
|\mathcal{X}/G| = |\mathcal{X}|/|G| & \text{orbit count (free action)}
\end{cases}$$

This translation reveals that the hypostructure ACT-Ghost theorem is a generalization of fundamental results in computational group theory: **canonical form computation** (gauge fixing) via **symmetry exploitation** (BRST cohomology) with **orbit counting** (Faddeev-Popov/Burnside) and **redundancy elimination** (quotient construction).

---

## Literature

1. **Becchi, C., Rouet, A., Stora, R. (1976).** "Renormalization of Gauge Theories." *Annals of Physics* 98:287-321. *Original BRST construction.*

2. **Tyutin, I.V. (1975).** "Gauge Invariance in Field Theory." Preprint FIAN-39. *Independent BRST discovery.*

3. **Faddeev, L.D., Popov, V.N. (1967).** "Feynman Diagrams for the Yang-Mills Field." *Physics Letters B* 25:29-30. *Faddeev-Popov ghosts.*

4. **Henneaux, M., Teitelboim, C. (1992).** *Quantization of Gauge Systems.* Princeton University Press. *Comprehensive BRST treatment.*

5. **McKay, B.D. (1981).** "Practical Graph Isomorphism." *Congressus Numerantium* 30:45-87. *Nauty algorithm.*

6. **Babai, L. (2016).** "Graph Isomorphism in Quasipolynomial Time." STOC. *Breakthrough GI algorithm.*

7. **Booth, K.S. (1980).** "Lexicographically Least Circular Substrings." *Information Processing Letters* 10:240-242. *Canonical string rotation.*

8. **Polya, G. (1937).** "Kombinatorische Anzahlbestimmungen fur Gruppen, Graphen und chemische Verbindungen." *Acta Mathematica* 68:145-254. *Polya enumeration theorem.*

9. **Burnside, W. (1897).** *Theory of Groups of Finite Order.* Cambridge University Press. *Burnside's lemma.*

10. **Weisfeiler, B., Leman, A. (1968).** "A Reduction of a Graph to a Canonical Form and an Algebra Arising During this Reduction." *Nauchno-Technicheskaya Informatsia* 2(9):12-16. *WL algorithm.*
