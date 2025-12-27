---
title: "KRNL-Subsystem - Complexity Theory Translation"
---

# KRNL-Subsystem: Subsystem Inheritance

## Overview

This document provides a complete complexity-theoretic translation of the KRNL-Subsystem theorem (Subsystem Inheritance) from the hypostructure framework. The translation establishes that tractability is a hereditary property: if a general computational problem is tractable, then every natural restriction to a sub-problem or promise version inherits tractability. This principle connects to fundamental results on closure under restrictions, promise problems, and parameterized complexity.

**Original Theorem Reference:** {prf:ref}`mt-krnl-subsystem`

---

## Complexity Theory Statement

**Theorem (KRNL-Subsystem, Computational Form).**
Let $\mathcal{P}$ be a computational problem (decision problem, search problem, or optimization problem) with input class $\mathcal{I}$. Let $\mathcal{Q} \subseteq \mathcal{P}$ be a **restriction** of $\mathcal{P}$ to a sub-class of inputs $\mathcal{I}' \subseteq \mathcal{I}$, where:

1. **Invariance:** If $x \in \mathcal{I}'$ and $\mathcal{A}$ is an algorithm for $\mathcal{P}$, then $\mathcal{A}(x)$ remains within the computational structure of $\mathcal{Q}$
2. **Structure Inheritance:** The restriction $\mathcal{Q}$ inherits the problem structure from $\mathcal{P}$ (objective function, constraints, solution space)

**Statement:** Tractability is hereditary. If the general problem $\mathcal{P}$ is tractable (solvable in polynomial time), then every restriction $\mathcal{Q} \subseteq \mathcal{P}$ is also tractable.

$$\mathcal{P} \in \mathrm{P} \wedge (\mathcal{Q} \subseteq \mathcal{P} \text{ restriction}) \Rightarrow \mathcal{Q} \in \mathrm{P}$$

**Contrapositive (Hardness Lifting):** If a restriction $\mathcal{Q}$ is NP-hard, then the general problem $\mathcal{P}$ is NP-hard.

$$\mathcal{Q} \text{ is NP-hard} \wedge (\mathcal{Q} \subseteq \mathcal{P}) \Rightarrow \mathcal{P} \text{ is NP-hard}$$

**Corollary (Promise Problem Inheritance).**
Let $(L_{\mathrm{YES}}, L_{\mathrm{NO}})$ be a promise problem where instances are guaranteed to satisfy a promise $\Pi$. If the unrestricted problem $L$ (without the promise) is in P, then the promise problem is in P.

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Equivalent | Formal Correspondence |
|-----------------------|------------------------------|------------------------|
| Parent system $\mathcal{H}$ | Full problem / unrestricted input class | Problem $\mathcal{P}$ on input domain $\mathcal{I}$ |
| Invariant subsystem $\mathcal{S} \subset \mathcal{H}$ | Restricted problem / promise problem | $\mathcal{Q} \subseteq \mathcal{P}$ on $\mathcal{I}' \subseteq \mathcal{I}$ |
| Lock Blocked $K_{\text{Lock}}^{\mathrm{blk}}$ | Tractability (P membership) | $\mathcal{P} \in \mathrm{P}$ |
| Singularity | Hardness (NP-completeness) | $\mathcal{P}$ is NP-hard |
| Lock certificate | Proof of tractability | Polynomial-time algorithm |
| Invariance under semiflow | Closure under computation | Restriction preserved by algorithm |
| Morphism $\mathcal{B}_{\text{univ}} \to \mathcal{H}$ | Reduction from SAT | SAT $\leq_p \mathcal{P}$ |
| Morphism composition | Reduction composition | Transitivity of $\leq_p$ |
| Inclusion $\iota: \mathcal{S} \hookrightarrow \mathcal{H}$ | Trivial reduction | $\mathcal{Q} \leq_p \mathcal{P}$ via identity |
| Energy functional $\Phi$ | Computational resource | Time/space complexity |
| Dissipation $\mathfrak{D}$ | Resource consumption rate | Steps per input bit |
| Fenichel persistence | Restriction preserves structure | Sub-problem inherits algorithm |
| Normally hyperbolic manifold | Well-structured input class | Inputs with bounded parameters |
| Symmetry group $G$ | Problem automorphisms | Permutations preserving solutions |

---

## Proof Sketch

### Setup: Computational Problem Restrictions

**Definition (Problem Restriction).**
Let $\mathcal{P} = (L, \mathcal{I}, \mathcal{S})$ be a computational problem where:
- $L \subseteq \Sigma^*$ is the language (YES-instances)
- $\mathcal{I} \subseteq \Sigma^*$ is the input domain
- $\mathcal{S}: \mathcal{I} \to 2^{\Sigma^*}$ maps inputs to solution spaces

A **restriction** $\mathcal{Q} = (L', \mathcal{I}', \mathcal{S}')$ is defined by:
- $\mathcal{I}' \subseteq \mathcal{I}$ (restricted input domain)
- $L' = L \cap \mathcal{I}'$ (induced language)
- $\mathcal{S}' = \mathcal{S}|_{\mathcal{I}'}$ (induced solution space)

**Definition (Promise Problem).**
A promise problem is a pair $(L_{\mathrm{YES}}, L_{\mathrm{NO}})$ with $L_{\mathrm{YES}} \cap L_{\mathrm{NO}} = \emptyset$. The **promise** is $\Pi = L_{\mathrm{YES}} \cup L_{\mathrm{NO}}$. An algorithm for the promise problem need only be correct on inputs satisfying $\Pi$.

**Invariance Condition:** A restriction $\mathcal{Q} \subseteq \mathcal{P}$ is **invariant** if for any algorithm $\mathcal{A}$ that solves $\mathcal{P}$:
- If $x \in \mathcal{I}'$, then $\mathcal{A}(x)$ gives the correct answer for $\mathcal{Q}$
- The computational path of $\mathcal{A}$ on $\mathcal{I}'$ does not exit the structure of $\mathcal{Q}$

---

### Step 1: Categorical Obstruction Argument (Reduction Composition)

**Goal:** Show that any hardness in the restriction implies hardness in the full problem.

#### Step 1.1: Hardness as Morphism

In complexity theory, NP-hardness corresponds to the existence of a polynomial-time reduction from SAT:
$$\text{SAT} \leq_p \mathcal{Q}$$

This is the computational analogue of a "singularity morphism" $\mathcal{B}_{\text{univ}} \to \mathcal{S}$ in the hypostructure framework. The reduction $f: \Sigma^* \to \mathcal{I}'$ maps SAT instances to instances of $\mathcal{Q}$, preserving satisfiability.

#### Step 1.2: Inclusion as Trivial Reduction

Since $\mathcal{Q}$ is a restriction of $\mathcal{P}$ with $\mathcal{I}' \subseteq \mathcal{I}$, there is a trivial reduction:
$$\mathcal{Q} \leq_p \mathcal{P} \quad \text{via the identity function}$$

For any $x \in \mathcal{I}'$: $x \in L'$ iff $x \in L$ (since $L' = L \cap \mathcal{I}'$). The identity function is computable in $O(n)$ time.

This corresponds to the inclusion morphism $\iota: \mathcal{S} \hookrightarrow \mathcal{H}$ in the hypostructure setting.

#### Step 1.3: Composition Yields Hardness of Full Problem

If $\mathcal{Q}$ were NP-hard (admitted a singularity), we could compose reductions:
$$\text{SAT} \xrightarrow{f} \mathcal{Q} \xrightarrow{\text{id}} \mathcal{P}$$

The composition $\text{id} \circ f = f$ is a polynomial-time reduction:
$$\text{SAT} \leq_p \mathcal{P}$$

Hence $\mathcal{P}$ would also be NP-hard.

**Contrapositive:** If $\mathcal{P}$ is NOT NP-hard (i.e., $\mathcal{P} \in \mathrm{P}$ assuming P $\neq$ NP), then $\mathcal{Q}$ cannot be NP-hard.

**Conclusion of Step 1:** By the transitivity of polynomial-time reductions, tractability of the parent problem implies tractability of any restriction.

---

### Step 2: Algorithmic Inheritance (Fenichel Analogue)

**Goal:** Provide a constructive proof that polynomial-time algorithms restrict to polynomial-time algorithms on sub-problems.

#### Step 2.1: Algorithm Restriction Theorem

**Theorem (Algorithm Inheritance).** Let $\mathcal{A}$ be a polynomial-time algorithm for $\mathcal{P}$ running in time $T(n) = O(n^k)$. Then the restriction $\mathcal{A}|_{\mathcal{I}'}$ solves $\mathcal{Q}$ in time $O(n^k)$.

**Proof:**
1. **Correctness:** For $x \in \mathcal{I}'$, we have $x \in L'$ iff $x \in L$ (by definition of restriction). Since $\mathcal{A}$ correctly decides $L$, it correctly decides $L'$ on inputs from $\mathcal{I}'$.

2. **Complexity:** The running time $T_{\mathcal{A}}(x)$ for $x \in \mathcal{I}'$ satisfies:
   $$T_{\mathcal{A}}(x) \leq T_{\mathcal{A}}^{\max}(|x|) = O(|x|^k)$$
   The restriction does not increase running time.

3. **Termination:** If $\mathcal{A}$ terminates on all inputs in $\mathcal{I}$, it terminates on all inputs in $\mathcal{I}' \subseteq \mathcal{I}$.

$\square$

This is the computational analogue of Fenichel's Invariant Manifold Theorem: regularity properties (polynomial time) persist under restriction to invariant subsets (input sub-classes).

#### Step 2.2: Resource Inheritance

The resource consumption of $\mathcal{A}$ on the restriction satisfies:

**Time:** $T_{\mathcal{Q}}(n) \leq T_{\mathcal{P}}(n)$

**Space:** $S_{\mathcal{Q}}(n) \leq S_{\mathcal{P}}(n)$

**Circuit Depth:** $D_{\mathcal{Q}}(n) \leq D_{\mathcal{P}}(n)$

All complexity measures are inherited from the parent problem. This corresponds to the energy-dissipation inheritance in the hypostructure:
$$\Phi|_{\mathcal{S}}(x) = \Phi(x) \quad \text{for } x \in \mathcal{S}$$

#### Step 2.3: Normal Hyperbolicity = Well-Structured Restrictions

Fenichel's theorem requires **normal hyperbolicity** of the invariant manifold. In complexity theory, this corresponds to **well-structured** restrictions:

**Definition (Well-Structured Restriction).**
A restriction $\mathcal{Q} \subseteq \mathcal{P}$ is well-structured if:
1. Membership in $\mathcal{I}'$ is polynomial-time decidable
2. The restriction is defined by a finite set of polynomial-time checkable constraints
3. The parameter of the restriction (if parameterized) is polynomially bounded

**Examples of Well-Structured Restrictions:**
- **Planar graphs:** $\mathcal{I}' = \{G : G \text{ is planar}\}$
- **Bounded-degree graphs:** $\mathcal{I}' = \{G : \Delta(G) \leq d\}$
- **Bounded treewidth:** $\mathcal{I}' = \{G : \text{tw}(G) \leq k\}$
- **Special cases of CSPs:** $\mathcal{I}' = \{(V, C) : \text{all constraints are Horn clauses}\}$

**When Fenichel Applies:** For well-structured restrictions, the inheritance is automatic. The polynomial-time algorithm for $\mathcal{P}$ directly yields a polynomial-time algorithm for $\mathcal{Q}$.

**When Fenichel Does NOT Apply:** For pathological restrictions (e.g., $\mathcal{I}' = L$ itself, making the restriction trivial), or restrictions with undecidable membership, the categorical argument (Step 1) must be used instead.

---

### Step 3: Verification of Complexity Class Closure

**Goal:** Verify that standard complexity classes are closed under restriction.

#### Axiom (P-Closure): Polynomial Time is Closed Under Restriction

**Theorem.** If $\mathcal{P} \in \mathrm{P}$ and $\mathcal{Q} \subseteq \mathcal{P}$ is a restriction, then $\mathcal{Q} \in \mathrm{P}$.

**Proof.** Let $\mathcal{A}$ be a polynomial-time algorithm for $\mathcal{P}$. Define $\mathcal{A}'$ for $\mathcal{Q}$:
$$\mathcal{A}'(x) = \begin{cases} \mathcal{A}(x) & \text{if } x \in \mathcal{I}' \\ \text{reject} & \text{otherwise} \end{cases}$$

If membership in $\mathcal{I}'$ is polynomial-time decidable (well-structured restriction):
- Checking $x \in \mathcal{I}'$: $O(n^{k_1})$ time
- Running $\mathcal{A}(x)$: $O(n^{k_2})$ time
- Total: $O(n^{\max(k_1, k_2)})$ = polynomial time

If membership in $\mathcal{I}'$ is trivial (no checking needed, algorithm works on all inputs):
- Simply run $\mathcal{A}(x)$: $O(n^{k_2})$ time

In both cases, $\mathcal{Q} \in \mathrm{P}$. $\square$

#### Axiom (FPT-Closure): FPT is Closed Under Parameter-Preserving Restriction

**Theorem.** If $(\mathcal{P}, k) \in \mathrm{FPT}$ and $(\mathcal{Q}, k) \subseteq (\mathcal{P}, k)$ is a parameter-preserving restriction, then $(\mathcal{Q}, k) \in \mathrm{FPT}$.

**Proof.** Let $\mathcal{A}$ solve $(\mathcal{P}, k)$ in time $f(k) \cdot n^c$. The restriction inherits this bound:
$$T_{\mathcal{Q}}(n, k) \leq T_{\mathcal{P}}(n, k) = O(f(k) \cdot n^c)$$

Hence $(\mathcal{Q}, k) \in \mathrm{FPT}$. $\square$

#### Axiom (Promise-Closure): Promise Problems Inherit Tractability

**Theorem.** If the unrestricted problem $L$ is in P, then any promise problem $(L_{\mathrm{YES}}, L_{\mathrm{NO}})$ with $L_{\mathrm{YES}} \subseteq L$ and $L_{\mathrm{NO}} \subseteq \overline{L}$ is in P.

**Proof.** The polynomial-time algorithm for $L$ works on the promise problem:
- On $x \in L_{\mathrm{YES}}$: accepts (since $L_{\mathrm{YES}} \subseteq L$)
- On $x \in L_{\mathrm{NO}}$: rejects (since $L_{\mathrm{NO}} \subseteq \overline{L}$)

The promise is never violated, and the algorithm runs in polynomial time. $\square$

---

### Step 4: Conclusion

We have established via three independent arguments that tractability is hereditary:

1. **Categorical Argument (Step 1):** Any reduction to the restriction composes with the trivial reduction to yield a reduction to the full problem. By transitivity of $\leq_p$, hardness of the restriction implies hardness of the full problem. Contrapositively, tractability of the full problem implies tractability of all restrictions.

2. **Algorithmic Argument (Step 2):** Polynomial-time algorithms restrict to polynomial-time algorithms on sub-problems. This is the computational Fenichel theorem: algorithms inherit efficiency under restriction to invariant input classes.

3. **Closure Verification (Step 3):** The complexity classes P, FPT, and promise-P are all closed under natural restrictions. The axioms of computational complexity theory confirm the inheritance principle.

Therefore, the **Subsystem Inheritance Principle** holds:
$$\boxed{\mathcal{P} \in \mathrm{P} \wedge (\mathcal{Q} \subseteq \mathcal{P} \text{ restriction}) \Rightarrow \mathcal{Q} \in \mathrm{P}}$$

---

## Certificate Construction

For each restriction $\mathcal{Q} \subseteq \mathcal{P}$, we construct explicit certificates:

**Tractability Certificate $K_{\mathrm{Tract}}$:**
```
K_Tract = {
  parent_problem: P,
  parent_certificate: {
    algorithm: A,
    time_bound: O(n^k),
    correctness_proof: pi_P
  },
  restriction: Q,
  inheritance_proof: {
    reduction: "Q <=_p P via identity",
    time_overhead: O(n),
    restriction_check: "I' membership in O(n^j)"
  },
  derived_certificate: {
    algorithm: A|_{I'},
    time_bound: O(n^max(k,j)),
    correctness: "inherited from P"
  }
}
```

**Hardness Non-Certificate (Obstruction):**
```
K_Hard_Obstruction = {
  claim: "Q is NP-hard",
  obstruction: {
    parent_tractability: "P in P",
    composition: "SAT <=_p Q <=_p P",
    contradiction: "would imply P = NP"
  },
  conclusion: "Q cannot be NP-hard"
}
```

---

## Connections to Classical Results

### 1. Promise Problems (Goldreich 2006)

**Definition.** A promise problem is a pair $(L_{\mathrm{YES}}, L_{\mathrm{NO}})$ of disjoint languages. An algorithm solves the promise problem if it:
- Accepts all $x \in L_{\mathrm{YES}}$
- Rejects all $x \in L_{\mathrm{NO}}$
- May behave arbitrarily on $x \notin L_{\mathrm{YES}} \cup L_{\mathrm{NO}}$

**Connection to KRNL-Subsystem:** Promise problems are restrictions of decision problems. The promise $\Pi = L_{\mathrm{YES}} \cup L_{\mathrm{NO}}$ defines the invariant subsystem. The KRNL-Subsystem theorem implies:

- If the unrestricted problem (no promise) is in P, the promise version is in P
- Promise problems cannot be harder than their unrestricted versions
- Tractability certificates for the full problem yield certificates for the promise version

**Key Examples:**
- **UNIQUE-SAT** (promise: at most one solution): No harder than SAT
- **GAP-MAX-CUT** (promise: large gap between YES/NO): Inherited from MAX-CUT
- **Approximate optimization:** Promise versions inherit tractability bounds

### 2. Closure Under Restrictions (Papadimitriou 1994)

**Definition.** A complexity class $\mathcal{C}$ is **closed under restrictions** if:
$$L \in \mathcal{C} \text{ and } L' = L \cap \mathcal{I}' \Rightarrow L' \in \mathcal{C}$$
whenever $\mathcal{I}'$ is a "natural" input class (polynomial-time recognizable).

**Connection to KRNL-Subsystem:** The KRNL-Subsystem theorem establishes that:

| Complexity Class | Closed Under Restriction? | Condition |
|------------------|---------------------------|-----------|
| P | Yes | No condition |
| NP | Yes | No condition |
| FPT | Yes | Parameter-preserving |
| XP | Yes | Parameter-preserving |
| PSPACE | Yes | No condition |
| L (logspace) | Yes | No condition |

**Closure Hierarchy:**
$$\text{L} \subseteq \text{NL} \subseteq \text{P} \subseteq \text{NP} \subseteq \text{PSPACE} \subseteq \text{EXP}$$

Each class is closed under restriction, with tractability flowing downward through the restriction relation.

### 3. Dichotomy Theorems and Special Cases

**Schaefer's Dichotomy (1978):** For Boolean CSPs, every constraint language $\Gamma$ yields a problem that is either in P or NP-complete.

**Connection to KRNL-Subsystem:** Tractable special cases of NP-complete problems are restrictions:
- **2-SAT** $\subseteq$ **SAT**: 2-SAT is the restriction to clauses of size $\leq 2$
- **Horn-SAT** $\subseteq$ **SAT**: Horn-SAT restricts to Horn clauses
- **XOR-SAT** $\subseteq$ **SAT**: XOR-SAT restricts to XOR clauses

By KRNL-Subsystem, if SAT were in P, all these restrictions would be in P. The contrapositive: the tractability of 2-SAT, Horn-SAT, and XOR-SAT does NOT imply SAT is in P, because these are restrictions, not generalizations.

**Inheritance Direction:**
$$\text{General problem tractable} \Rightarrow \text{Restriction tractable}$$
$$\text{Restriction tractable} \not\Rightarrow \text{General problem tractable}$$

### 4. Parameterized Complexity and Kernelization

**Fixed-Parameter Tractability (FPT):** A parameterized problem $(L, k)$ is in FPT if it is solvable in time $f(k) \cdot n^{O(1)}$ for some computable $f$.

**Connection to KRNL-Subsystem:** Parameter restrictions are subsystems:
- **Vertex Cover** with parameter $k$ (solution size): Restriction of general Vertex Cover
- **k-Path**: Restriction of Hamiltonian Path to short paths
- **Bounded treewidth**: Restriction to structured graphs

The KRNL-Subsystem theorem implies:
- If a problem is FPT for general $k$, it is FPT for any fixed $k_0$
- Parameter restrictions inherit the FPT algorithm
- Kernelization bounds are preserved under restriction

**Kernelization Inheritance:**
$$\text{Kernel}(\mathcal{Q}) \leq \text{Kernel}(\mathcal{P})$$

If $\mathcal{P}$ admits a polynomial kernel, so does every restriction $\mathcal{Q}$.

### 5. Graph Classes and Hereditary Properties

**Definition.** A graph class $\mathcal{G}$ is **hereditary** if it is closed under induced subgraphs: $G \in \mathcal{G}$ and $H \subseteq_{\text{ind}} G$ implies $H \in \mathcal{G}$.

**Examples:**
- **Chordal graphs** (forbidden: induced $C_k$ for $k \geq 4$)
- **Perfect graphs** (closed under complementation and induced subgraphs)
- **Planar graphs** (closed under minors, hence under induced subgraphs)

**Connection to KRNL-Subsystem:** For problems on hereditary graph classes:

| General Problem | Hereditary Restriction | Tractability Inheritance |
|-----------------|------------------------|--------------------------|
| 3-COLORING (NP-c) | 3-COLORING on chordal | P (inherited structure) |
| MAX-CLIQUE (NP-c) | MAX-CLIQUE on perfect | P (inherited structure) |
| HAMILTONIAN (NP-c) | HAMILTONIAN on $K_n$ | P (trivial on complete) |

The KRNL-Subsystem theorem explains why restricting to tractable graph classes yields tractable problems: the restriction inherits the tractability of the class-specific algorithm.

---

## Quantitative Bounds

### Complexity Inheritance

For a restriction $\mathcal{Q} \subseteq \mathcal{P}$ with $\mathcal{P}$ solvable in time $T_{\mathcal{P}}(n)$:

**Time Bound:**
$$T_{\mathcal{Q}}(n) \leq T_{\mathcal{P}}(n) + T_{\text{check}}(n)$$

where $T_{\text{check}}(n)$ is the time to verify $x \in \mathcal{I}'$.

**For Well-Structured Restrictions:**
- $T_{\text{check}}(n) = O(n^j)$ for polynomial-time checkable restrictions
- $T_{\mathcal{Q}}(n) = O(T_{\mathcal{P}}(n))$ when check is dominated

**For Trivial Restrictions:**
- $T_{\text{check}}(n) = 0$ (no check needed)
- $T_{\mathcal{Q}}(n) = T_{\mathcal{P}}(n)$ exactly

### FPT Inheritance

For parameterized problems with $(\mathcal{P}, k) \in \mathrm{FPT}$ solvable in time $f(k) \cdot n^c$:

**Parameter-Preserving Restriction:**
$$T_{(\mathcal{Q}, k)}(n) \leq f(k) \cdot n^c$$

**Parameter-Bounded Restriction** (restricting to $k \leq k_0$):
$$T_{(\mathcal{Q}, k_0)}(n) = O(n^c) \in \mathrm{P}$$

### Approximation Inheritance

For optimization problems with approximation ratio $\alpha$:

**PTAS Inheritance:** If $\mathcal{P}$ has a PTAS (polynomial-time approximation scheme), then any restriction $\mathcal{Q}$ has a PTAS with the same approximation guarantee.

**Inapproximability Lifting:** If $\mathcal{Q}$ has no $\alpha$-approximation (assuming P $\neq$ NP), then $\mathcal{P}$ has no $\alpha$-approximation.

---

## Summary

The KRNL-Subsystem theorem, translated to complexity theory, establishes the fundamental principle of **tractability inheritance**:

1. **Restrictions Inherit Tractability:** If a general problem $\mathcal{P}$ is solvable in polynomial time, every restriction $\mathcal{Q} \subseteq \mathcal{P}$ is also solvable in polynomial time. The polynomial-time algorithm for $\mathcal{P}$ directly yields a polynomial-time algorithm for $\mathcal{Q}$.

2. **Promise Problems Inherit Tractability:** Promise problems cannot be harder than their unrestricted versions. If the full decision problem is in P, every promise version is in P.

3. **Hardness Lifts to General Problems:** Contrapositively, if a restriction is NP-hard, the general problem must be NP-hard. Hardness propagates upward through the restriction hierarchy.

4. **Complexity Classes are Closed Under Restriction:** The standard complexity classes (P, NP, FPT, PSPACE) are all closed under natural restrictions. This closure property is the complexity-theoretic manifestation of the Subsystem Inheritance Principle.

**Physical Interpretation (Computational Analogue):**

- **Parent System $\mathcal{H}$:** The general problem with full input domain
- **Invariant Subsystem $\mathcal{S}$:** The restricted problem with constrained inputs
- **Lock Blocked (Regularity):** The problem is tractable (P membership)
- **Singularity (Hardness):** The problem is NP-complete

Just as a globally regular dynamical system cannot develop singularities in any invariant subsystem, a tractable computational problem cannot have intractable restrictions. The tractability "flows" from the general to the specific, while hardness "lifts" from the specific to the general.

**The Subsystem Inheritance Certificate:**
$$K_{\text{Subsystem}} = \begin{cases}
K_{\text{Inherited}}(\mathcal{Q}) & \text{if } \mathcal{P} \in \mathrm{P} \\
K_{\text{Hardness-Lift}}(\mathcal{P}) & \text{if } \mathcal{Q} \text{ is NP-hard}
\end{cases}$$

---

## Literature

1. **Fenichel, N. (1971).** "Persistence and Smoothness of Invariant Manifolds for Flows." *Indiana Univ. Math. J.* 21:193-226. *Invariant manifold persistence.*

2. **Hirsch, M. W., Pugh, C. C., & Shub, M. (1977).** *Invariant Manifolds.* Lecture Notes in Mathematics 583, Springer. *Infinite-dimensional invariant manifold theory.*

3. **Wiggins, S. (1994).** *Normally Hyperbolic Invariant Manifolds in Dynamical Systems.* Springer. *Applications to Hamiltonian and dissipative systems.*

4. **Goldreich, O. (2006).** "On Promise Problems: A Survey." *Theoretical Computer Science* 3895:254-290. *Promise problems and computational complexity.*

5. **Papadimitriou, C. H. (1994).** *Computational Complexity.* Addison-Wesley. *Foundations of complexity theory.*

6. **Schaefer, T. J. (1978).** "The Complexity of Satisfiability Problems." *STOC 1978*, 216-226. *Boolean CSP dichotomy theorem.*

7. **Downey, R. G. & Fellows, M. R. (1999).** *Parameterized Complexity.* Springer. *FPT and kernelization theory.*

8. **Cygan, M. et al. (2015).** *Parameterized Algorithms.* Springer. *Modern parameterized complexity.*

9. **Garey, M. R. & Johnson, D. S. (1979).** *Computers and Intractability: A Guide to the Theory of NP-Completeness.* Freeman. *Classical NP-completeness reference.*

10. **Arora, S. & Barak, B. (2009).** *Computational Complexity: A Modern Approach.* Cambridge University Press. *Modern complexity theory.*
