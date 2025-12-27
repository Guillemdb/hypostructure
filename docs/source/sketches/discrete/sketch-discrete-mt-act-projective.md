---
title: "ACT-Projective - Complexity Theory Translation"
---

# ACT-Projective: Compactification and Projective Closure

## Overview

This document provides a complete complexity-theoretic translation of the ACT-Projective metatheorem (Projective Extension) from the hypostructure framework. The theorem establishes that collapsed constraint sets can be repaired via slack variables and relaxation, restoring positive capacity. In algebraic complexity terms, this corresponds to **Compactification**: extending problems via projective closure to handle boundary and degenerate cases.

**Original Theorem Reference:** {prf:ref}`mt-act-projective`

**Central Translation:** Slack variable relaxation restores capacity to collapsed constraint sets $\longleftrightarrow$ **Projective Closure**: Homogenization and compactification extend computations to handle points at infinity and boundary degeneracies.

---

## Complexity Theory Statement

**Theorem (Projective Extension for Algebraic Computation).**
Let $\mathcal{C}$ be an algebraic computation over affine space $\mathbb{A}^n$ that degenerates or becomes singular at boundary configurations. There exists a **projective extension** that:

**Input**: Affine problem $\mathcal{C}$ + degeneracy certificate (capacity collapse detected)

**Output**:
- Projective problem $\tilde{\mathcal{C}}$ over $\mathbb{P}^n$
- Desingularization via homogenization
- Certificate that boundary behavior is controlled

**Guarantees**:
1. **Capacity restoration**: $\text{Cap}(\tilde{\mathcal{C}}) > 0$ even where affine problem collapses
2. **Convergence**: Solutions in projective space limit to affine solutions
3. **Uniform bounds**: Complexity controlled by degree, independent of boundary distance
4. **Certificate production**: Validity witness for extended computation

**Formal Statement.** Let $f \in k[x_1, \ldots, x_n]$ be a polynomial of degree $d$ defining an affine variety $V = V(f) \subset \mathbb{A}^n$. The homogenization:

$$\tilde{f}(x_0, x_1, \ldots, x_n) = x_0^d \cdot f\left(\frac{x_1}{x_0}, \ldots, \frac{x_n}{x_0}\right)$$

defines a projective variety $\tilde{V} = V(\tilde{f}) \subset \mathbb{P}^n$ satisfying:

1. **Extension:** $\tilde{V} \cap \{x_0 \neq 0\} \cong V$ (affine part recovered)

2. **Compactness:** $\tilde{V}$ is a compact variety (closed in Zariski topology)

3. **Degree Preservation:** $\deg(\tilde{V}) = \deg(V) = d$

4. **Bezout at Infinity:** Intersections are counted including points at infinity:
   $$|\tilde{V}_1 \cap \tilde{V}_2| = d_1 \cdot d_2$$
   (counting multiplicities)

**Corollary (Bounded Arithmetic via Homogenization).**
For algebraic computations with potential blowup at boundaries, homogenization provides uniform complexity bounds:
$$\text{Complexity}(\tilde{\mathcal{C}}) \leq \text{poly}(n, d)$$
independent of proximity to singular loci.

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Constraint set $K = \{g_i(x) \leq 0\}$ | Affine variety $V \subset \mathbb{A}^n$ | Zero set of polynomial system |
| Collapsed capacity $\text{Cap}(K) = 0$ | Degenerate/boundary configuration | Points at infinity, singular loci |
| Slack variable $s_i \geq 0$ | Homogenizing variable $x_0$ | Additional coordinate for projective space |
| Relaxed set $K_\varepsilon$ | Projective variety $\tilde{V} \subset \mathbb{P}^n$ | Compactified space including infinity |
| $\text{Cap}(K_\varepsilon) > 0$ | Non-degenerate projective variety | Compact, finite intersection number |
| Hausdorff convergence $K_\varepsilon \to K$ | Dehomogenization $x_0 \to 1$ | Recovery of affine variety |
| Logarithmic barrier $-\mu \sum \log s_i$ | Projective distance function | Metric on projective space |
| Central path $\nabla f_\mu = 0$ | Newton iteration in projective space | Path-following in $\mathbb{P}^n$ |
| Interior point method | Projective elimination | Solving via homogenized system |
| Certificate $(\varepsilon, s^*, x^*)$ | Projective solution + degree | Bezout number, intersection multiplicity |
| Volume restoration | Dimension count | $\dim(\tilde{V}) = \dim(V)$ |
| Barrier function | Fubini-Study metric | Standard metric on $\mathbb{P}^n$ |
| Convergence $\varepsilon \to 0$ | Limiting to affine chart | $\tilde{V} \cap \{x_0 = 1\} = V$ |
| Constraint relaxation | Projective desingularization | Blowing up singular points |

---

## Connections to Projective Geometry in Complexity

### 1. Homogenization and Degree Control

**Definition (Homogenization).** For $f \in k[x_1, \ldots, x_n]$ of degree $d$, the homogenization is:

$$f^h(x_0, x_1, \ldots, x_n) = x_0^d \cdot f\left(\frac{x_1}{x_0}, \ldots, \frac{x_n}{x_0}\right)$$

where $f^h$ is homogeneous of degree $d$.

**Example.** For $f = x^2 + y - 1$ (degree 2):
$$f^h = x^2 + yz - z^2$$
The affine hyperbola becomes a projective conic.

**Complexity Benefit:** Homogeneous polynomials have uniform behavior:
- No "escape to infinity" during evaluation
- Bezout's theorem applies exactly
- Resultants have predictable degree

### 2. Bezout's Theorem as Capacity Bound

**Theorem (Bezout).** For hypersurfaces $V_1, \ldots, V_n \subset \mathbb{P}^n$ of degrees $d_1, \ldots, d_n$:

$$|V_1 \cap \cdots \cap V_n| = \prod_{i=1}^n d_i$$

counting multiplicities and including points at infinity.

**Affine Failure:** In $\mathbb{A}^n$, the intersection count can be less than expected:
- Lines may be parallel (intersect at infinity)
- Varieties may escape to infinity
- Boundary effects reduce intersection count

**Projective Repair:** Compactification ensures the full intersection count is realized:
- Parallel lines intersect at a point at infinity
- All intersections accounted for
- Capacity (intersection number) is preserved

**Correspondence to ACT-Projective:**
| Hypostructure | Projective Geometry |
|---------------|---------------------|
| $\text{Cap}(K) = 0$ | Missing intersections at infinity |
| Slack variables | Homogenizing coordinate $x_0$ |
| $\text{Cap}(K_\varepsilon) > 0$ | Full Bezout number achieved |
| $\varepsilon \to 0$ limit | Dehomogenization to affine chart |

### 3. Bounded Arithmetic and Projective Coordinates

**Connection to Bounded Arithmetic:**

In bounded arithmetic (e.g., $S^1_2$, $T^1_2$), computations are restricted to polynomial-length witnesses. Projective coordinates provide natural bounds:

| Bounded Arithmetic | Projective Geometry |
|--------------------|---------------------|
| Polynomial-length witnesses | Homogeneous coordinates |
| Bit-complexity bounds | Degree bounds on $x_i/x_0$ |
| Induction on formula complexity | Induction on projective degree |
| Witnessing theorems | Bezout bounds on solutions |

**Theorem (Projective Bit Complexity).** For a projective variety $V \subset \mathbb{P}^n$ of degree $d$ defined over $\mathbb{Q}$, any rational point $[a_0 : \cdots : a_n] \in V(\mathbb{Q})$ with $\gcd(a_i) = 1$ satisfies:

$$\max_i |a_i| \leq H(V)^{d^{O(n)}}$$

where $H(V)$ is the height of $V$.

**Complexity Implication:** Projective coordinates provide uniform bit-length bounds, enabling bounded arithmetic reasoning about algebraic solutions.

### 4. Projective Elimination Theory

**Definition (Projective Resultant).** For homogeneous polynomials $f_0, \ldots, f_n \in k[x_0, \ldots, x_n]$, the resultant $\text{Res}(f_0, \ldots, f_n)$ satisfies:

$$\text{Res}(f_0, \ldots, f_n) = 0 \iff V(f_0) \cap \cdots \cap V(f_n) \neq \emptyset \text{ in } \mathbb{P}^n$$

**Complexity Control:** The resultant has degree bounded by:
$$\deg(\text{Res}) \leq \prod_{i=0}^n d_i$$

This provides polynomial-time algorithms for:
1. Deciding if projective system has solutions
2. Computing intersection multiplicities
3. Eliminating variables with degree control

**Affine Failure vs. Projective Success:**

| Problem | Affine | Projective |
|---------|--------|------------|
| Parallel lines intersect? | No | Yes (at infinity) |
| Resultant well-defined? | May degenerate | Always well-defined |
| Bezout sharp? | Lower bound only | Exact equality |
| Elimination terminates? | May diverge | Always terminates |

---

## Proof Sketch: Capacity Restoration via Compactification

### Setup: The Affine-Projective Correspondence

**Definitions:**
1. **Affine space:** $\mathbb{A}^n = \{(x_1, \ldots, x_n) : x_i \in k\}$
2. **Projective space:** $\mathbb{P}^n = (\mathbb{A}^{n+1} \setminus \{0\})/k^*$
3. **Standard inclusion:** $\mathbb{A}^n \hookrightarrow \mathbb{P}^n$ via $(x_1, \ldots, x_n) \mapsto [1 : x_1 : \cdots : x_n]$
4. **Hyperplane at infinity:** $H_\infty = \{[0 : x_1 : \cdots : x_n]\} \cong \mathbb{P}^{n-1}$

**Complexity Measures:**

| Measure | Affine | Projective |
|---------|--------|------------|
| Degree | $\deg(f)$ | $\deg(f^h)$ (same) |
| Dimension | $\dim(V)$ | $\dim(\tilde{V})$ (same) |
| Intersection | May be empty | Bezout number exact |
| Closure | Open subset | Compact variety |

---

### Step 1: Slack Introduction = Homogenization

**Claim.** Adding slack variables corresponds to homogenizing polynomials.

**Proof.**

The hypostructure statement: Replace $g_i(x) \leq 0$ with $g_i(x) - s_i \leq 0$ and $s_i \geq 0$.

In algebraic geometry: For affine polynomial $f(x_1, \ldots, x_n)$ of degree $d$, homogenize:
$$f^h(x_0, x_1, \ldots, x_n) = x_0^d \cdot f(x_1/x_0, \ldots, x_n/x_0)$$

**The slack variable $x_0$ enables:**
1. **Extension:** When $x_0 \neq 0$, recover affine variety via $x_i/x_0$
2. **Infinity:** When $x_0 = 0$, capture behavior at infinity
3. **Compactness:** Projective variety is closed (compact in classical topology over $\mathbb{C}$)

**Example.** The constraint $x^2 + y^2 \leq 1$ collapses as $(x,y) \to \infty$.

Homogenization: $x^2 + y^2 - z^2 = 0$ defines a conic in $\mathbb{P}^2$.

The points at infinity $[1 : \pm i : 0]$ (over $\mathbb{C}$) complete the variety. $\square$

---

### Step 2: Capacity Restoration = Compactness

**Claim.** Projective closure restores positive capacity (finite intersection count).

**Proof.**

*Step 2.1 (Affine Degeneration):*
In affine space, two lines $L_1 = \{y = 2x + 1\}$ and $L_2 = \{y = 2x - 1\}$ are parallel:
$$|L_1 \cap L_2| = 0$$

This corresponds to $\text{Cap}(K) = 0$: the constraint system is empty.

*Step 2.2 (Projective Restoration):*
Homogenize:
- $L_1^h: y - 2x - z = 0$
- $L_2^h: y - 2x + z = 0$

The intersection includes the point at infinity $[1 : 2 : 0]$:
$$|\tilde{L}_1 \cap \tilde{L}_2| = 1$$

Bezout's theorem is satisfied: $1 \times 1 = 1$.

*Step 2.3 (General Principle):*
For varieties $V_1, \ldots, V_k \subset \mathbb{A}^n$ with $\bigcap V_i = \emptyset$, the projective closures $\tilde{V}_i \subset \mathbb{P}^n$ satisfy:
$$\bigcap_{i=1}^k \tilde{V}_i \supseteq \bigcap_{i=1}^k \tilde{V}_i \cap H_\infty \neq \emptyset$$
(generically, by dimension counting).

**Certificate:** $\text{Cap}(K_\varepsilon) = |\tilde{V}_1 \cap \cdots \cap \tilde{V}_k| = \prod d_i > 0$. $\square$

---

### Step 3: Barrier Function = Fubini-Study Metric

**Claim.** The logarithmic barrier corresponds to the projective distance function.

**Proof.**

*Step 3.1 (Hypostructure Barrier):*
The logarithmic barrier for slack variables:
$$f_\mu(x, s) = f(x) - \mu \sum_i \log s_i$$

forces $s_i > 0$ (staying in the interior).

*Step 3.2 (Projective Metric):*
The Fubini-Study metric on $\mathbb{P}^n$ is:
$$d_{FS}([z], [w]) = \arccos\left(\frac{|\langle z, w \rangle|}{\|z\| \|w\|}\right)$$

This metric makes $\mathbb{P}^n$ compact with finite diameter $\pi/2$.

*Step 3.3 (Correspondence):*

| Barrier Method | Projective Geometry |
|----------------|---------------------|
| $\log s_i$ penalty | Distance to hyperplane $\{x_i = 0\}$ |
| $\mu \to 0$ | Approaching boundary of affine chart |
| Central path | Geodesic in $\mathbb{P}^n$ |
| Interior point | Point in affine chart $\{x_0 \neq 0\}$ |
| Boundary | Hyperplane at infinity $H_\infty$ |

*Step 3.4 (Algorithmic Use):*
Newton's method in projective space:
1. Lift affine problem to $\mathbb{P}^n$
2. Follow geodesic (central path) toward solution
3. Project back to affine chart

Complexity: Polynomial in degree and dimension. $\square$

---

### Step 4: Convergence = Dehomogenization

**Claim.** As $\varepsilon \to 0$, projective solutions converge to affine solutions.

**Proof.**

*Step 4.1 (Relaxation Parameter):*
In ACT-Projective, $K_\varepsilon \to K$ as $\varepsilon \to 0$ in Hausdorff distance.

*Step 4.2 (Projective Limit):*
For a family of points $[1 : x_1(\varepsilon) : \cdots : x_n(\varepsilon)] \in \tilde{V}$:

If $\|x(\varepsilon)\| \to \infty$, the limit is a point at infinity $[0 : x_1 : \cdots : x_n]$.

If $\|x(\varepsilon)\|$ stays bounded, the limit is an affine point $(x_1, \ldots, x_n) \in V$.

*Step 4.3 (Convergence Guarantee):*
By compactness of $\mathbb{P}^n$, every sequence in $\tilde{V}$ has a convergent subsequence. The limit exists (possibly at infinity).

**Affine Recovery:** Solutions with $x_0 \neq 0$ correspond exactly to affine solutions.

**Complexity:** Convergence rate is polynomial in $1/\varepsilon$ and degree $d$. $\square$

---

### Step 5: Certificate Production

**Claim.** The projective extension produces a validity certificate.

**Proof.**

The certificate $K_{\text{SurgCD}}$ contains:

$$K_{\text{SurgCD}} = \begin{cases}
\tilde{V} \subset \mathbb{P}^n & \text{(projective variety)} \\
\deg(\tilde{V}) = d & \text{(degree preserved)} \\
\dim(\tilde{V}) = k & \text{(dimension computed)} \\
|\tilde{V} \cap \tilde{W}| = d \cdot e & \text{(Bezout number)} \\
\text{affine chart: } V = \tilde{V} \cap \{x_0 = 1\} & \text{(recovery)}
\end{cases}$$

**Verification:**
1. Homogeneity: Check $\tilde{f}(\lambda x_0, \ldots, \lambda x_n) = \lambda^d \tilde{f}(x_0, \ldots, x_n)$
2. Bezout: Verify intersection count matches degree product
3. Affine recovery: Dehomogenize and check containment

Verification complexity: $O(\text{poly}(n, d))$. $\square$

---

## Projective Geometry in Algebraic Complexity

### 1. Projective Algebraic Circuits

**Definition (Projective Circuit).** A projective algebraic circuit computes a homogeneous polynomial. All intermediate values are homogeneous of specified degree.

**Advantages:**
- Uniform degree tracking
- No division by zero issues
- Clean intersection theory

**Example.** The determinant is naturally homogeneous:
$$\det(X) = \det(\lambda X) / \lambda^n$$

Circuit for det uses only $+$ and $\times$ on matrix entries (no division needed).

### 2. Grassmannians and Subspace Computation

**Definition.** The Grassmannian $\text{Gr}(k, n)$ parametrizes $k$-dimensional subspaces of $k^n$.

**Projective Embedding:** Plucker embedding $\text{Gr}(k,n) \hookrightarrow \mathbb{P}^{\binom{n}{k}-1}$.

**Complexity Application:** Linear algebra problems become questions about varieties in projective space:
- Rank computation: Position on Grassmannian
- Eigenvalue problems: Intersection with hyperplanes
- Matrix completion: Extending points to full variety

### 3. Resultants and Elimination

**Theorem (Resultant Degree Bound).** The resultant of $n+1$ homogeneous polynomials in $n+1$ variables has degree:
$$\deg(\text{Res}) = \prod_{i=0}^n d_i / d_j$$

for each variable $j$ (multigraded resultant).

**Algorithm:**
1. Homogenize input polynomials
2. Compute resultant via Bezout matrix or Sylvester matrix
3. Extract conditions on coefficients for common solution

**Complexity:** $O(d^{O(n)})$ where $d = \max_i d_i$.

### 4. Tropical Geometry and Valuation

**Connection to Projective Space:**
Tropical varieties are "shadows" of algebraic varieties under valuation:
$$\text{val}: k^* \to \mathbb{R}$$

**Projective Tropicalization:** For $V \subset \mathbb{P}^n$:
$$\text{Trop}(V) = \overline{\text{val}(V \cap (\mathbb{C}^*)^n)} \subset \mathbb{R}^n / \mathbb{R} \cdot \mathbf{1}$$

**Complexity Application:**
- Tropical computations are piecewise linear
- Provide combinatorial approximations to algebraic problems
- Newton polytopes encode tropical varieties

---

## Certificate Construction

**Projective Extension Certificate:**

```
K_SurgCD := (
    mode:                "Constraint_Relaxation"
    mechanism:           "Projective_Closure"

    affine_problem: {
        variety:         V = V(f_1, ..., f_m) subset A^n
        dimension:       dim(V) = k
        singular_locus:  Sing(V) or boundary at infinity
        capacity:        Cap(V) = 0 (degenerate)
    }

    homogenization: {
        variables:       (x_0, x_1, ..., x_n)
        polynomials:     f_i^h = x_0^{d_i} f_i(x_1/x_0, ..., x_n/x_0)
        degrees:         deg(f_i^h) = d_i
        projective:      V_tilde = V(f_1^h, ..., f_m^h) subset P^n
    }

    capacity_restoration: {
        compactness:     "V_tilde is closed in P^n"
        bezout_number:   prod(d_i) (intersection count)
        infinity_points: V_tilde cap H_infinity
        proof:           "Bezout_theorem"
    }

    convergence: {
        affine_chart:    V = V_tilde cap {x_0 = 1}
        hausdorff_limit: "V_eps -> V as eps -> 0"
        rate:            "poly(1/eps, d)"
    }

    certificate: {
        degree:          d = max(d_i)
        dimension:       k = dim(V_tilde)
        intersection:    prod(d_i)
        recovery:        "dehomogenize to obtain affine solutions"
    }

    literature: {
        projective_geometry: "Hartshorne77, GriffithsHarris78"
        bezout_theorem:      "Fulton84"
        elimination_theory:  "GKZ94, Sturmfels02"
        interior_point:      "NesterovNemirovskii94"
    }
)
```

---

## Quantitative Summary

| Property | Affine | Projective |
|----------|--------|------------|
| Intersection count | $\leq \prod d_i$ | $= \prod d_i$ (exact) |
| Degree bound | $\deg(f)$ | $\deg(f^h) = \deg(f)$ |
| Resultant degree | May be undefined | $\prod d_i$ |
| Bit complexity | Unbounded near infinity | $H^{d^{O(n)}}$ |
| Algorithm termination | May diverge | Always terminates |
| Solution count | Lower bound | Exact (with multiplicity) |

### Complexity Bounds

| Algorithm | Affine Complexity | Projective Complexity |
|-----------|------------------|----------------------|
| Polynomial solving | $O(d^n)$ expected | $O(d^n)$ worst-case |
| Resultant computation | May fail | $O(d^{2n})$ |
| Intersection count | Heuristic | Exact in $O(d^n)$ |
| Elimination | Degree explosion | Controlled by $\prod d_i$ |
| GCD computation | Division issues | Clean via resultant |

---

## Extended Connections

### 1. Compactification in Optimization

**Interior Point Methods:**

The central path follows:
$$x(\mu) = \arg\min_x \left\{ c^T x - \mu \sum_i \log(b_i - A_i^T x) \right\}$$

As $\mu \to 0$, the path approaches the optimal vertex.

**Projective Interpretation:**
- Feasible region = affine variety
- Barrier = distance to boundary
- Central path = geodesic in projective metric
- Optimal solution = limiting point (possibly at infinity)

### 2. Compactification in Algebraic Statistics

**Log-Linear Models:**
Statistical models parametrized by:
$$p_\theta(x) \propto \exp\left(\sum_i \theta_i f_i(x)\right)$$

**Projective Closure:**
The variety of marginal distributions lives in projective space. Boundary corresponds to degenerate distributions (some $p_\theta(x) = 0$).

**Maximum Likelihood:**
MLE may lie at infinity (non-existent in affine space). Projective closure provides extended MLE.

### 3. Compactification in Algebraic Complexity

**Geometric Complexity Theory (GCT):**
Mulmuley's approach to VP vs VNP uses compactification:
- Orbit closures in projective space
- Representation-theoretic obstructions
- Boundaries encode complexity lower bounds

**Permanent vs Determinant:**
The orbit closure $\overline{GL_n \cdot \det}$ in projective space captures the algebraic complexity of determinant. Permanent lies outside this closure iff VP $\neq$ VNP.

### 4. Bounded Arithmetic and Projective Proofs

**Paris-Wilkie Theorem:**
Bounded arithmetic $I\Delta_0$ cannot prove totality of super-polynomial functions.

**Projective Interpretation:**
- Bounded formulas = affine constraints
- Proof = path in solution space
- Totality = existence of solution for all inputs
- Projective closure = adding "points at infinity" (non-standard models)

**Certificate:**
Proofs in bounded arithmetic correspond to algebraic certificates with degree bounds - exactly the projective degree of the solution variety.

---

## Summary

The ACT-Projective theorem translates to complexity theory as **Compactification via Projective Closure**:

1. **Surgery = Homogenization:** Adding slack variables corresponds to introducing homogeneous coordinates.

2. **Capacity Restoration = Compactness:** Projective closure ensures all intersections are counted (Bezout's theorem).

3. **Convergence = Dehomogenization:** Affine solutions are recovered as $x_0 \to 1$.

4. **Barrier = Fubini-Study Metric:** Logarithmic barrier corresponds to projective distance.

5. **Certificate = Degree Bound:** Bezout number provides the complexity certificate.

**Physical Interpretation (Computational Analogue):**

- **Collapsed constraint** = Degenerate affine configuration
- **Slack variables** = Homogenizing coordinate $x_0$
- **Relaxed region** = Projective variety $\tilde{V}$
- **Capacity > 0** = Bezout number is positive
- **Convergence** = Projective limit to affine chart

**The Projective Extension Certificate:**

$$K_{\text{SurgCD}}^+ = \begin{cases}
\tilde{V} \subset \mathbb{P}^n & \text{projective variety} \\
\deg(\tilde{V}) = d & \text{degree preserved} \\
|\tilde{V} \cap \tilde{W}| = d \cdot e & \text{Bezout bound} \\
V = \tilde{V} \cap \{x_0 = 1\} & \text{affine recovery}
\end{cases}$$

This translation reveals that the hypostructure ACT-Projective theorem is a generalization of **projective compactification**: extending affine problems to projective space where boundary degeneracies are resolved. Just as slack variables restore feasibility to collapsed constraint sets, homogenization restores intersection count by including points at infinity.

The key insight: **Compactification provides uniform complexity bounds** by eliminating "escape to infinity." Projective coordinates bound bit-complexity, projective degree bounds intersection count, and projective algorithms terminate where affine ones might diverge.

---

## Literature

1. **Hartshorne, R. (1977).** *Algebraic Geometry.* Springer. *Standard reference for projective varieties.*

2. **Griffiths, P. & Harris, J. (1978).** *Principles of Algebraic Geometry.* Wiley. *Analytic and algebraic perspectives.*

3. **Fulton, W. (1984).** *Intersection Theory.* Springer. *Rigorous treatment of Bezout's theorem.*

4. **Gel'fand, I. M., Kapranov, M. M., & Zelevinsky, A. V. (1994).** *Discriminants, Resultants, and Multidimensional Determinants.* Birkhauser. *Elimination theory.*

5. **Sturmfels, B. (2002).** *Solving Systems of Polynomial Equations.* AMS. *Computational algebraic geometry.*

6. **Nesterov, Y. & Nemirovskii, A. (1994).** *Interior-Point Polynomial Algorithms in Convex Programming.* SIAM. *Barrier methods and central path.*

7. **Boyd, S. & Vandenberghe, L. (2004).** *Convex Optimization.* Cambridge. *Interior point methods and slack variables.*

8. **Rockafellar, R. T. (1970).** *Convex Analysis.* Princeton. *Foundations of constrained optimization.*

9. **Ben-Tal, A. & Nemirovski, A. (2001).** *Lectures on Modern Convex Optimization.* SIAM. *Robust optimization and relaxation.*

10. **Cox, D., Little, J., & O'Shea, D. (2007).** *Ideals, Varieties, and Algorithms* (3rd ed.). Springer. *Computational introduction to algebraic geometry.*

11. **Mulmuley, K. & Sohoni, M. (2001).** "Geometric Complexity Theory I." *SIAM J. Computing.* *GCT and orbit closure.*

12. **Shub, M. & Smale, S. (1993).** "Complexity of Bezout's Theorem I-V." *Various venues.* *Algebraic complexity of polynomial solving.*

13. **Giusti, M. & Heintz, J. (1991).** "La determination des points isoles et de la dimension d'une variete algebrique." *Computational complexity of variety computation.*

14. **Canny, J. (1988).** "Some Algebraic and Geometric Computations in PSPACE." *STOC.* *PSPACE algorithms for algebraic problems.*

15. **Basu, S., Pollack, R., & Roy, M.-F. (2006).** *Algorithms in Real Algebraic Geometry* (2nd ed.). Springer. *Comprehensive treatment of computational real algebraic geometry.*
