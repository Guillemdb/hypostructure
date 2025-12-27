---
title: "LOCK-Schematic - Complexity Theory Translation"
---

# LOCK-Schematic: Sum-of-Squares Certificates and Proof Complexity

## Overview

This document provides a complete complexity-theoretic translation of the LOCK-Schematic theorem (Semialgebraic Exclusion) from the hypostructure framework. The theorem establishes that schematic locks---polynomial certificate constructions via Stengle's Positivstellensatz---provide algebraic witnesses for excluding bad patterns from safe regions.

In complexity theory, this corresponds to **Sum-of-Squares (SOS) Proof Complexity**: the study of how polynomial identity certificates witness infeasibility of constraint systems. The SOS hierarchy provides a systematic framework for certifying that polynomial systems have no solutions, with degree bounds controlling certificate complexity.

**Original Theorem Reference:** {prf:ref}`mt-lock-schematic`

---

## Complexity Theory Statement

**Theorem (LOCK-Schematic, SOS Certificate Form).**
Let $\mathcal{R} = \mathbb{R}[x_1, \ldots, x_n]$ be a polynomial ring over $n$ variables. Consider:

- **Safe region:** $S = \{x \in \mathbb{R}^n \mid g_1(x) \geq 0, \ldots, g_k(x) \geq 0\}$ (semialgebraic set defined by polynomial inequalities)
- **Bad pattern region:** $B \subseteq \mathbb{R}^n$ (states violating safety constraints)

**Statement (Positivstellensatz Certificate):**
The regions are disjoint, $S \cap B = \emptyset$, if and only if there exist **sum-of-squares polynomials** $\{p_\alpha\}_{\alpha \in \{0,1\}^k} \subseteq \Sigma[x]$ such that:

$$-1 = p_0 + \sum_{i} p_i g_i + \sum_{i<j} p_{ij} g_i g_j + \cdots + p_{1\ldots k} g_1 \cdots g_k$$

where $\Sigma[x] = \{\sum_j f_j^2 : f_j \in \mathbb{R}[x]\}$ denotes the cone of sum-of-squares polynomials.

**Corollary (SOS Degree Bound).**
If such a certificate exists with SOS polynomials of degree at most $2d$, then:
1. The infeasibility can be verified by semidefinite programming (SDP) of size $\binom{n+d}{d}$.
2. The certificate has **SOS degree** $d$, measuring proof complexity.

**Corollary (Nullstellensatz Refutation).**
When constraints include equalities $h_j(x) = 0$, the Nullstellensatz provides:
$$1 = \sum_i p_i g_i + \sum_j q_j h_j$$
for SOS $p_i$ and arbitrary polynomials $q_j$, certifying $S \cap B = \emptyset$ over $\mathbb{C}$.

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Schematic lock | Sum-of-Squares certificate | $\{p_\alpha\}$ witnessing infeasibility |
| Polynomial identity | Algebraic circuit identity testing | Verifying $-1 = \sum_\alpha p_\alpha \prod_{i \in \alpha} g_i$ |
| Degree bound | SOS degree | Maximum degree of $p_\alpha$ polynomials |
| Schematic sieve | Nullstellensatz/Positivstellensatz refutation | Algebraic proof of emptiness |
| Algebraic variety | Solution set of polynomial system | $V(I) = \{x : f(x) = 0 \; \forall f \in I\}$ |
| Safe region $S$ | Feasible region of polynomial constraints | Semialgebraic set $\{g_i(x) \geq 0\}$ |
| Bad pattern $B$ | Infeasible region to exclude | Unsafe states in constraint system |
| SOS witness $\{p_\alpha\}$ | SDP feasibility certificate | Gram matrix decomposition |
| Permit certificate $K_{\text{SOS}}^+$ | Proof complexity certificate | Degree-bounded refutation witness |
| Capacity bound $K_{\mathrm{Cap}_H}^+$ | Dimension bound on variety | $\dim(V) \leq d$ |
| Lojasiewicz gradient $K_{\mathrm{LS}_\sigma}^+$ | Error bound / regularity | $\|x - V\| \leq C \cdot \|g(x)\|^\theta$ |
| Subcritical scaling $K_{\mathrm{SC}_\lambda}^+$ | Polynomial growth bound | Degree vs. size trade-off |
| Topological bound $K_{\mathrm{TB}_\pi}^+$ | Betti number bound | $\sum_i b_i(V) \leq B$ |
| SDP feasibility | Proof verification | Polynomial-time certificate checking |

---

## Logical Framework

### Sum-of-Squares and Semidefinite Programming

**Definition (Sum-of-Squares Polynomial).**
A polynomial $p \in \mathbb{R}[x_1, \ldots, x_n]$ is a **sum of squares (SOS)** if:
$$p(x) = \sum_{j=1}^r f_j(x)^2$$
for some polynomials $f_1, \ldots, f_r \in \mathbb{R}[x]$.

**Characterization (Gram Matrix).**
A polynomial $p$ of degree $2d$ is SOS if and only if there exists a positive semidefinite matrix $Q \succeq 0$ such that:
$$p(x) = [x]_d^T Q [x]_d$$
where $[x]_d = (1, x_1, \ldots, x_n, x_1^2, x_1 x_2, \ldots, x_1^d, \ldots, x_n^d)^T$ is the vector of monomials up to degree $d$.

**Complexity:** Testing whether $p$ is SOS is decidable via SDP in time polynomial in $\binom{n+d}{d}$.

### Positivstellensatz and Nullstellensatz

**Theorem (Stengle's Positivstellensatz, 1974).**
Let $g_1, \ldots, g_k \in \mathbb{R}[x]$. The semialgebraic set $S = \{x : g_i(x) \geq 0 \; \forall i\}$ is empty if and only if there exist SOS polynomials $\{p_\alpha\}_{\alpha \in \{0,1\}^k}$ such that:
$$-1 = \sum_{\alpha \in \{0,1\}^k} p_\alpha \cdot \prod_{i \in \alpha} g_i$$

**Theorem (Hilbert's Nullstellensatz).**
For polynomials $f_1, \ldots, f_m \in \mathbb{C}[x]$, the variety $V(f_1, \ldots, f_m) = \emptyset$ if and only if there exist $q_1, \ldots, q_m \in \mathbb{C}[x]$ such that:
$$1 = \sum_{j=1}^m q_j f_j$$

**Key Distinction:**
- Nullstellensatz: equalities over $\mathbb{C}$, arbitrary polynomial multipliers
- Positivstellensatz: inequalities over $\mathbb{R}$, SOS polynomial multipliers

### SOS Hierarchy and Lasserre Relaxation

**Definition (SOS Degree-$d$ Proof).**
An **SOS degree-$d$ refutation** of $\{g_i(x) \geq 0\}$ is a certificate where all $p_\alpha$ have degree at most $2d$.

**The Lasserre/SOS Hierarchy:**
For optimization problem $\min\{c^T x : x \in S\}$:
- Level $d$: Search for degree-$2d$ SOS certificates
- Relaxation: SDP of size $O(n^d)$
- Completeness: Exact for $d \geq $ Positivstellensatz degree

**Theorem (Lasserre 2001).**
The SOS hierarchy provides a systematic sequence of SDP relaxations with:
1. Increasing tightness at each level
2. Finite convergence to exact optimum
3. Polynomial-time solvability at each fixed level

---

## Proof Sketch

### Setup: Polynomial Constraint System

**Problem Formulation.** Given:
- Structural invariants as polynomial variables: $x_1 = \Phi$ (potential), $x_2 = \mathfrak{D}$ (dimension), $x_3 = \text{Gap}$ (spectral gap), etc.
- Safe region from permit certificates:
  - $g_{\text{SC}}(x) := \beta - \alpha - \varepsilon \geq 0$ (subcritical scaling)
  - $g_{\text{Cap}}(x) := C\mathfrak{D} - \text{Cap}_H(\text{Supp}) \geq 0$ (capacity bound)
  - $g_{\text{LS}}(x) := \|\nabla\Phi\|^2 - C_{\text{LS}}^2 |\Phi - \Phi_{\min}|^{2\theta} \geq 0$ (Lojasiewicz gradient)
  - $g_{\text{TB}}(x) := c^2 - \|\nabla\Pi\|^2 \geq 0$ (topological bound)
- Bad pattern region $B$: states leading to singularity

**Goal:** Construct SOS certificate that $S \cap B = \emptyset$.

### Step 1: Real Algebraic Geometry Foundation

**Lemma 1.1 (Semialgebraic Sets).**
The safe region $S$ and bad pattern region $B$ are semialgebraic sets: finite Boolean combinations of polynomial inequalities.

**Proof.**
Each permit certificate defines a polynomial inequality:
- $K_{\mathrm{SC}_\lambda}^+$: $g_{\text{SC}}(x) = \beta(x) - \alpha(x) - \varepsilon \geq 0$
- $K_{\mathrm{Cap}_H}^+$: $g_{\text{Cap}}(x) = C \cdot \mathfrak{D}(x) - \text{Cap}(x) \geq 0$

The intersection of finitely many such inequalities is semialgebraic. $\square$

**Lemma 1.2 (Positivstellensatz Applicability).**
For semialgebraic $S, B$, the emptiness $S \cap B = \emptyset$ admits an algebraic certificate via Stengle's Positivstellensatz.

**Key Insight:** The Nullstellensatz handles equalities over algebraically closed fields ($\mathbb{C}$), but permit certificates assert *inequalities* over $\mathbb{R}$. The Positivstellensatz is the correct tool.

### Step 2: Bad Pattern Encoding

**Definition (Bad Pattern Semialgebraic Encoding).**
A bad pattern $B_i$ (singularity type $i$) is encoded as:
$$B_i = \{x \in \mathbb{R}^n \mid h_1(x) \geq 0, \ldots, h_m(x) \geq 0, f(x) = 0\}$$
representing states that lead to failure mode $i$.

**Example Encodings:**
- **Geometric collapse (C.D):** $h(x) = -\text{Gap}(x) + \delta$ (spectral gap below threshold)
- **Stiffness breakdown (S.D):** $h(x) = \varepsilon - \|\nabla\Phi\|^2 / |\Phi - \Phi_{\min}|^{2\theta}$ (Lojasiewicz violated)
- **Capacity overflow:** $h(x) = \text{Cap}(x) - C \cdot \mathfrak{D}(x)$ (capacity exceeds bound)

### Step 3: Infeasibility Certificate Construction

**Theorem 3.1 (SOS Infeasibility Certificate).**
If $S \cap B = \emptyset$, then there exist SOS polynomials $\{p_\alpha\}$ such that:
$$-1 = p_0 + \sum_{i=1}^k p_i g_i + \sum_{1 \leq i < j \leq k} p_{ij} g_i g_j + \cdots + p_{1\ldots k} \prod_{i=1}^k g_i$$

**Proof (Stengle's Positivstellensatz).**

**Step 3.1 (Preordering).**
Define the preordering generated by $g_1, \ldots, g_k$:
$$T(g_1, \ldots, g_k) = \left\{\sum_{\alpha \in \{0,1\}^k} p_\alpha \prod_{i \in \alpha} g_i : p_\alpha \in \Sigma[x]\right\}$$

**Step 3.2 (Archimedean property).**
If $S$ is compact (bounded), then $T$ is Archimedean: there exists $N$ such that $N - \sum x_i^2 \in T$.

**Step 3.3 (Positivstellensatz application).**
By Stengle's theorem, $S \cap B = \emptyset$ implies $-1 \in T + I(B)$ where $I(B)$ is the ideal generated by equalities defining $B$.

**Step 3.4 (Certificate extraction).**
Rewrite the membership identity to obtain explicit $\{p_\alpha\}$. $\square$

### Step 4: SOS Computation via Semidefinite Programming

**Algorithm (SOS Certificate Search).**
Given constraint polynomials $g_1, \ldots, g_k$ and degree bound $d$:

1. **Parameterize SOS polynomials:** For each $\alpha \in \{0,1\}^k$, let:
   $$p_\alpha(x) = [x]_{d_\alpha}^T Q_\alpha [x]_{d_\alpha}$$
   where $Q_\alpha \succeq 0$ and $d_\alpha$ is chosen so $\deg(p_\alpha \prod_{i \in \alpha} g_i) \leq 2d$.

2. **Form identity constraint:** The equation
   $$-1 = \sum_\alpha p_\alpha \prod_{i \in \alpha} g_i$$
   yields linear constraints on entries of $\{Q_\alpha\}$ by matching coefficients.

3. **Solve SDP feasibility:**
   $$\text{find } \{Q_\alpha\}_{\alpha \in \{0,1\}^k} \text{ such that } Q_\alpha \succeq 0, \; \text{coefficient constraints satisfied}$$

4. **Certificate output:** If feasible, extract $\{p_\alpha\}$ from $\{Q_\alpha\}$.

**Complexity:** The SDP has $2^k$ matrix variables, each of size $\binom{n+d}{d}$. For fixed $k$ and $d$, this is polynomial in $n$.

### Step 5: Certificate Assembly

**Definition (SOS Certificate).**
The output certificate is:
$$K_{\text{SOS}}^+ = \left(\{p_\alpha\}_{\alpha \in \{0,1\}^k}, \{g_i\}_{i=1}^k, \{Q_\alpha\}_\alpha\right)$$

where:
- $\{p_\alpha\}$: SOS polynomials witnessing the Positivstellensatz identity
- $\{g_i\}$: Permit constraint polynomials defining $S$
- $\{Q_\alpha\}$: SDP Gram matrices certifying each $p_\alpha$ is SOS

**Verification (Polynomial Time):**
1. Check $Q_\alpha \succeq 0$ for all $\alpha$ (eigenvalue test)
2. Verify $p_\alpha(x) = [x]^T Q_\alpha [x]$ (matrix multiplication)
3. Confirm identity $-1 = \sum_\alpha p_\alpha \prod_{i \in \alpha} g_i$ (coefficient comparison)

---

## Certificate Construction

### Input Certificates (Permit Constraints)

The schematic lock requires permits as polynomial constraints:

$$K_{\mathrm{Cap}_H}^+ : g_{\text{Cap}}(x) = C\mathfrak{D} - \text{Cap}_H(\text{Supp}) \geq 0$$

$$K_{\mathrm{LS}_\sigma}^+ : g_{\text{LS}}(x) = \|\nabla\Phi\|^2 - C_{\text{LS}}^2 |\Phi - \Phi_{\min}|^{2\theta} \geq 0$$

$$K_{\mathrm{SC}_\lambda}^+ : g_{\text{SC}}(x) = \beta - \alpha - \varepsilon \geq 0$$

$$K_{\mathrm{TB}_\pi}^+ : g_{\text{TB}}(x) = c^2 - \|\nabla\Pi\|^2 \geq 0$$

### Output Certificate (SOS Witness)

$$K_{\text{SOS}}^+ = \left(\{p_\alpha\}_\alpha, \{g_i\}_i, \{Q_\alpha\}_\alpha, d_{\text{SOS}}\right)$$

where:
- $\{p_\alpha\}$: SOS polynomials in the Positivstellensatz identity
- $\{g_i\}$: Constraint polynomials from permits
- $\{Q_\alpha\}$: Gram matrices (SDP witness)
- $d_{\text{SOS}}$: Maximum SOS degree (proof complexity measure)

### Certificate Logic

The complete logical structure:
$$K_{\mathrm{Cap}_H}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{TB}_\pi}^+ \Rightarrow K_{\text{SOS}}^+$$

**Translation:**
- Input permits define safe region $S$ via polynomial inequalities
- SOS certificate proves $S \cap B = \emptyset$ algebraically
- SDP computation provides verifiable witness

---

## Connections to Classical Results

### 1. SOS Hierarchy and Proof Complexity

**Theorem (Grigoriev 2001, Schoenebeck 2008).**
There exist constraint systems requiring SOS degree $\Omega(n)$ to refute, despite being unsatisfiable.

**Example (Knapsack):** Random 3-XOR instances require linear SOS degree.

**Connection to LOCK-Schematic:** The SOS degree $d_{\text{SOS}}$ in the certificate measures proof complexity. High-degree requirements indicate "hard" infeasibility proofs.

### 2. Grigoriev-Razborov Lower Bounds

**Theorem (Grigoriev-Razborov 2001).**
Refuting the Pigeonhole Principle $\text{PHP}_{n+1}^n$ (encoding $n+1$ pigeons into $n$ holes) requires SOS degree $\Omega(n)$.

**Proof Sketch:**
1. Encode PHP as polynomial constraints: $\sum_j x_{ij} = 1$ (each pigeon in some hole), $x_{ij} x_{kj} = 0$ for $i \neq k$ (no hole collision).
2. Show any low-degree SOS refutation implies a low-degree "pseudo-expectation" operator.
3. Construct such an operator consistent with constraints, contradicting refutation existence.

**Implications for LOCK-Schematic:**
- Not all infeasibility proofs have efficient (low-degree) SOS certificates
- The degree bound $d_{\text{SOS}}$ captures intrinsic proof complexity
- Some "bad pattern exclusions" may require high-degree certificates

### 3. Lasserre Hierarchy and Optimization

**Theorem (Lasserre 2001).**
For polynomial optimization $\min\{f(x) : x \in S\}$ over compact semialgebraic $S$:
- Level-$d$ Lasserre relaxation is an SDP of size $O(n^d)$
- Optimal value converges to true minimum as $d \to \infty$
- Rate depends on algebraic degree of optimum

**Connection to LOCK-Schematic:**
- Lasserre hierarchy systematizes SOS certificate search
- Each level corresponds to bounded SOS degree
- Hierarchy provides algorithmic path to certificate computation

### 4. Real Algebraic Geometry and Complexity

**Theorem (Basu-Pollack-Roy 1996).**
Deciding emptiness of semialgebraic sets is in $\text{PSPACE}$ and $\text{EXPTIME}$-hard in general.

**Theorem (Renegar 1992).**
For fixed number of variables $n$, semialgebraic decision is polynomial in constraint complexity.

**Connection to LOCK-Schematic:**
- General certificate existence is decidable but expensive
- Fixed-parameter tractability when $n$ is bounded
- SOS provides practical algorithms via SDP relaxation

### 5. Nullstellensatz Proof Complexity

**Theorem (Beame et al. 1996).**
Nullstellensatz refutation degree for PHP is $\Omega(n)$ over any field.

**Comparison with Positivstellensatz:**
| Property | Nullstellensatz | Positivstellensatz |
|----------|-----------------|-------------------|
| Field | Any (typically $\mathbb{C}$) | $\mathbb{R}$ only |
| Constraints | Equalities | Inequalities |
| Multipliers | Arbitrary polynomials | SOS polynomials |
| Computation | Linear algebra | Semidefinite programming |
| Degree bounds | Well-studied | Active research |

### 6. SOS and Approximation Algorithms

**Theorem (Barak-Steurer 2014).**
Constant-level SOS relaxations achieve best-known approximations for:
- Constraint satisfaction problems (CSPs)
- Unique Games variants
- Planted clique (subexponential algorithms)

**Connection to LOCK-Schematic:**
- SOS certificates not only prove infeasibility but guide optimization
- Low-degree certificates indicate tractable structure
- Links to approximation algorithms and hardness

---

## Quantitative Refinements

### SOS Degree Bounds

**Theorem (Effective Positivstellensatz).**
If $S = \{g_i(x) \geq 0\} \cap B = \emptyset$ and constraints have degree at most $D$, then there exists a Positivstellensatz certificate with:
$$\deg(p_\alpha) \leq 2^{2^{O(n)}} \cdot D^{O(n)}$$

**Remark:** The doubly-exponential bound is worst-case. Structured instances often admit much lower degree.

### Computational Complexity of Certificate Search

| Degree $d$ | SDP Size | Verification | Certificate Size |
|------------|----------|--------------|-----------------|
| $O(1)$ | $\text{poly}(n)$ | $\text{poly}(n)$ | $\text{poly}(n)$ |
| $O(\log n)$ | $n^{O(\log n)}$ | $n^{O(\log n)}$ | $n^{O(\log n)}$ |
| $O(n)$ | $2^{O(n \log n)}$ | $2^{O(n \log n)}$ | $2^{O(n \log n)}$ |

### Lower Bounds Summary

| Problem | SOS Degree Lower Bound | Reference |
|---------|----------------------|-----------|
| PHP$_{n+1}^n$ | $\Omega(n)$ | Grigoriev-Razborov 2001 |
| Random 3-XOR | $\Omega(n)$ | Schoenebeck 2008 |
| Random 3-SAT | $\Omega(n)$ | Grigoriev 2001 |
| Planted Clique | $\Omega(\sqrt{n})$ | Barak et al. 2019 |
| Knapsack | $\Omega(n)$ | Grigoriev 2001 |

---

## Application: Verifiable Constraint Satisfaction

### Algorithm: SOS-CERTIFICATE-SEARCH

```
Input: Constraint polynomials g_1, ..., g_k defining safe region S
       Bad pattern polynomial description B
       Degree bound d

Output: SOS certificate K_SOS^+ or "degree insufficient"

Algorithm:
1. Enumerate SOS polynomial templates:
   For each alpha in {0,1}^k:
     Set degree(p_alpha) = 2d - sum_{i in alpha} deg(g_i)
     Create Gram matrix variable Q_alpha of size binom(n + d_alpha, d_alpha)

2. Form Positivstellensatz identity:
   Expand -1 = sum_alpha p_alpha prod_{i in alpha} g_i
   Match coefficients to obtain linear constraints on Q_alpha entries

3. Solve SDP feasibility:
   Find {Q_alpha} such that:
     - Q_alpha >= 0 for all alpha (positive semidefinite)
     - Linear coefficient constraints satisfied

4. If feasible:
     Extract p_alpha from Q_alpha
     Return K_SOS^+ = ({p_alpha}, {g_i}, {Q_alpha}, d)
   Else:
     Return "degree insufficient, try d+1"

Complexity: O(2^k * binom(n+d,d)^3) for SDP solving
```

### Verification Protocol

```
Input: Certificate K_SOS^+ = ({p_alpha}, {g_i}, {Q_alpha}, d)

Verification:
1. For each alpha:
   a. Check Q_alpha is symmetric
   b. Compute eigenvalues of Q_alpha
   c. Verify all eigenvalues >= 0 (Q_alpha >= 0)
   d. Verify p_alpha(x) = [x]^T Q_alpha [x]

2. Expand Positivstellensatz identity:
   Compute P(x) = sum_alpha p_alpha(x) * prod_{i in alpha} g_i(x)

3. Verify P(x) = -1:
   Check all coefficients of P except constant term are 0
   Check constant term equals -1

Output: ACCEPT if all checks pass, REJECT otherwise

Complexity: O(2^k * binom(n+d,d)^2) for verification
```

---

## Summary

The LOCK-Schematic theorem, translated to complexity theory, establishes **Sum-of-Squares Proof Complexity for Semialgebraic Exclusion**:

1. **Fundamental Correspondence:**
   - Schematic lock $\leftrightarrow$ SOS certificate
   - Polynomial identity $\leftrightarrow$ Algebraic circuit identity
   - Degree bound $\leftrightarrow$ SOS degree (proof complexity)
   - Schematic sieve $\leftrightarrow$ Nullstellensatz/Positivstellensatz refutation
   - Safe region $S$ $\leftrightarrow$ Feasible semialgebraic set
   - Bad pattern $B$ $\leftrightarrow$ Infeasible region to exclude

2. **Main Result:** The exclusion $S \cap B = \emptyset$ admits a constructive SOS certificate:
   - Computed via semidefinite programming
   - Verified in polynomial time (given degree bound)
   - Degree captures proof complexity

3. **Connection to Proof Complexity:**
   - Grigoriev-Razborov lower bounds: some exclusions require high SOS degree
   - Lasserre hierarchy provides systematic certificate search
   - SOS degree stratifies proof complexity landscape

4. **Certificate Structure:**
   $$K_{\mathrm{Cap}_H}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{TB}_\pi}^+ \Rightarrow K_{\text{SOS}}^+$$

   Permit constraints (polynomial inequalities) combine to yield SOS infeasibility witness.

5. **Computational Aspects:**
   - Certificate search: SDP of size $O(n^d)$ for degree-$d$ proof
   - Verification: polynomial in certificate size
   - Lower bounds: $\Omega(n)$ degree for Pigeonhole, random CSPs

**The Complexity-Theoretic Insight:**

The LOCK-Schematic theorem reveals that algebraic exclusion certificates in the hypostructure framework correspond to Sum-of-Squares proofs in computational complexity. The Positivstellensatz---the real algebraic analog of Hilbert's Nullstellensatz---provides the mathematical foundation, while SDP-based algorithms make certificate construction practical.

The SOS degree measures proof complexity: low-degree certificates indicate tractable exclusion proofs, while Grigoriev-Razborov style lower bounds identify intrinsically hard instances. This stratification by degree parallels the classification of proof systems in propositional proof complexity, with SOS occupying a powerful position in the hierarchy (stronger than bounded-depth Frege, weaker than general Frege).

For complexity theorists, LOCK-Schematic offers a geometric perspective on SOS: permit certificates define a safe polytope in structural-invariant space, and SOS refutations algebraically certify that bad patterns lie outside this polytope. The interplay between polynomial constraints, semidefinite programming, and proof complexity illuminates both the power and limitations of algebraic reasoning in computation.

---

## Literature

**Sum-of-Squares and Positivstellensatz:**

1. **Stengle, G. (1974).** "A Nullstellensatz and a Positivstellensatz in Semialgebraic Geometry." Mathematische Annalen. *Original Positivstellensatz theorem.*

2. **Parrilo, P. A. (2003).** "Semidefinite Programming Relaxations for Semialgebraic Problems." Mathematical Programming. *SOS computation via SDP.*

3. **Lasserre, J. B. (2001).** "Global Optimization with Polynomials and the Problem of Moments." SIAM J. Optimization. *Lasserre hierarchy for polynomial optimization.*

4. **Blekherman, G., Parrilo, P., & Thomas, R. (2012).** *Semidefinite Optimization and Convex Algebraic Geometry.* SIAM. *Comprehensive treatment of SOS and SDP.*

**SOS Lower Bounds and Proof Complexity:**

5. **Grigoriev, D. (2001).** "Linear Lower Bound on Degrees of Positivstellensatz Calculus Proofs for the Parity." Theoretical Computer Science. *Early SOS degree lower bounds.*

6. **Grigoriev, D. & Razborov, A. (2001).** "Exponential Lower Bounds for Depth 3 Arithmetic Circuits in Algebras of Functions over Finite Fields." Applicable Algebra in Engineering. *Foundational lower bound techniques.*

7. **Schoenebeck, G. (2008).** "Linear Level Lasserre Lower Bounds for Certain k-CSPs." FOCS. *SOS lower bounds for random CSPs.*

8. **Barak, B. & Steurer, D. (2014).** "Sum-of-Squares Proofs and the Quest Toward Optimal Algorithms." ICM Proceedings. *Survey of SOS in algorithms.*

**Nullstellensatz and Algebraic Proof Complexity:**

9. **Beame, P., Impagliazzo, R., Krajicek, J., Pitassi, T., & Pudlak, P. (1996).** "Lower Bounds on Hilbert's Nullstellensatz and Propositional Proofs." Proc. London Math. Soc. *Nullstellensatz proof complexity.*

10. **Buss, S. R. (1998).** "Lower Bounds on Nullstellensatz Proofs via Designs." In *Proof Complexity and Feasible Arithmetics.* *Design-based lower bound techniques.*

**Real Algebraic Geometry:**

11. **Basu, S., Pollack, R., & Roy, M.-F. (2006).** *Algorithms in Real Algebraic Geometry.* Springer. *Comprehensive algorithmic treatment.*

12. **Renegar, J. (1992).** "On the Computational Complexity and Geometry of the First-Order Theory of the Reals." J. Symbolic Computation. *Complexity of real algebraic decision.*

**Applications and Extensions:**

13. **Barak, B., Hopkins, S., Kelner, J., Kothari, P., Moitra, A., & Potechin, A. (2019).** "A Nearly Tight Sum-of-Squares Lower Bound for the Planted Clique Problem." SIAM J. Computing. *State-of-art SOS lower bounds.*

14. **O'Donnell, R. (2017).** "SOS Is Not Obviously Automatizable, Even Approximately." ITCS. *Computational hardness of SOS optimization.*

15. **Raghavendra, P. & Weitz, B. (2017).** "On the Bit Complexity of Sum-of-Squares Proofs." ICALP. *Numerical precision in SOS computation.*
