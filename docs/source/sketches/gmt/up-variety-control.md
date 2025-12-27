# UP-VarietyControl: Variety-Control Theorem — GMT Translation

## Original Statement (Hypostructure)

The variety-control theorem shows that algebraic constraints on singular sets provide effective control on their geometry and complexity.

## GMT Setting

**Algebraic Variety:** $V = \{x : P_1(x) = \cdots = P_k(x) = 0\}$

**Singular Set Control:** $\Sigma \subset V$ implies $\Sigma$ inherits algebraic structure

**Complexity:** Bounded by degree of defining polynomials

## GMT Statement

**Theorem (Variety-Control).** If singular set $\Sigma$ is contained in algebraic variety $V$ of degree $d$:

1. **Dimension Bound:** $\dim(\Sigma) \leq \dim(V)$

2. **Complexity Bound:** Number of components $\leq C(n, d)$

3. **Volume Bound:** $\mathcal{H}^{\dim V}(\Sigma \cap B_1) \leq C(n, d)$

4. **Stratification:** $\Sigma$ has semialgebraic stratification

## Proof Sketch

### Step 1: Algebraic Variety Structure

**Definition:** An algebraic variety $V \subset \mathbb{R}^n$ is:
$$V = \{x \in \mathbb{R}^n : P_1(x) = \cdots = P_k(x) = 0\}$$

for polynomials $P_i$ of degree $\leq d$.

**Degree:** $\deg(V) = $ maximum of intersection number with generic lines.

**Reference:** Harris, J. (1992). *Algebraic Geometry: A First Course*. Springer.

### Step 2: Dimension from Degree

**Bézout's Theorem:** For $V_1, V_2 \subset \mathbb{P}^n$:
$$\deg(V_1 \cap V_2) \leq \deg(V_1) \cdot \deg(V_2)$$

**Dimension Formula:**
$$\dim(V_1 \cap V_2) \geq \dim(V_1) + \dim(V_2) - n$$

**Reference:** Fulton, W. (1984). *Intersection Theory*. Springer.

### Step 3: Component Counting

**Theorem (Oleinik-Petrovskii-Milnor-Thom):** Number of connected components of real algebraic variety of degree $d$ in $\mathbb{R}^n$:
$$b_0(V) \leq d(2d - 1)^{n-1}$$

**Reference:** Milnor, J. (1964). On the Betti numbers of real varieties. *Proc. AMS*, 15, 275-280.

**Refinement (Basu et al.):** More precise bounds available.

**Reference:** Basu, S., Pollack, R., Roy, M.-F. (2006). *Algorithms in Real Algebraic Geometry*. Springer.

### Step 4: Volume Bounds

**Crofton Formula:** For $k$-dimensional variety $V$:
$$\mathcal{H}^k(V \cap B_R) \leq C(k) \cdot \deg(V) \cdot R^k$$

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer.

**Algebraic Volume:** Volume of algebraic variety in unit ball bounded by degree.

### Step 5: Semialgebraic Stratification

**Definition:** A set is **semialgebraic** if defined by polynomial equalities and inequalities.

**Stratification (Łojasiewicz):** Every semialgebraic set has Whitney stratification:
$$V = \bigsqcup_{i=1}^N S_i$$

with $N \leq C(n, d)$.

**Reference:** Łojasiewicz, S. (1965). Ensembles semi-analytiques. IHES.

### Step 6: Singular Set Containment

**Application:** If $\text{sing}(T) \subset V$ for algebraic $V$:
- $\dim(\text{sing}(T)) \leq \dim(V)$
- Number of strata $\leq C(n, \deg(V))$
- Volume in $B_1$ bounded by $C \cdot \deg(V)$

### Step 7: Effective Regularity

**Theorem (Quantitative Regularity):** If $T \in \mathbf{I}_k(M)$ with:
$$\text{sing}(T) \subset V, \quad \deg(V) = d$$

then:
$$\mathcal{H}^{k-2}(\text{sing}(T)) \leq C(n, k, d) \cdot \mathbf{M}(T)$$

*Proof:* Algebraic containment + GMT regularity theory.

### Step 8: Algebraic Blow-Up Limits

**Theorem:** Tangent cones of algebraic varieties are algebraic.

**Consequence:** Blow-up limits of $\text{sing}(T) \subset V$ are contained in algebraic cones.

**Classification:** Only finitely many algebraic cone types of given degree.

### Step 9: Control via Hilbert Function

**Hilbert Function:** $H_V(d) = \dim(\mathbb{C}[x_1, \ldots, x_n]_d / I(V)_d)$

**Hilbert Polynomial:** For $d \gg 0$:
$$H_V(d) = \frac{\deg(V)}{(\dim V)!} d^{\dim V} + \text{lower terms}$$

**Reference:** Eisenbud, D. (1995). *Commutative Algebra with a View Toward Algebraic Geometry*. Springer.

**Complexity Control:** Hilbert function encodes complexity of $V$.

### Step 10: Compilation Theorem

**Theorem (Variety-Control):**

1. **Dimension:** $\Sigma \subset V \implies \dim(\Sigma) \leq \dim(V)$

2. **Components:** Number bounded by $C(n, d)$

3. **Volume:** $\mathcal{H}^{\dim V}(\Sigma \cap B_1) \leq C(n, d)$

4. **Stratification:** Semialgebraic with controlled complexity

**Applications:**
- Effective singular set bounds
- Classification of blow-up limits
- Complexity analysis of regularity

## Key GMT Inequalities Used

1. **Bézout:**
   $$\deg(V_1 \cap V_2) \leq \deg(V_1) \deg(V_2)$$

2. **Component Count:**
   $$b_0(V) \leq d(2d-1)^{n-1}$$

3. **Volume Bound:**
   $$\mathcal{H}^k(V \cap B_R) \leq C \deg(V) R^k$$

4. **Stratification Count:**
   $$|\{S_i\}| \leq C(n, d)$$

## Literature References

- Harris, J. (1992). *Algebraic Geometry: A First Course*. Springer.
- Fulton, W. (1984). *Intersection Theory*. Springer.
- Milnor, J. (1964). On the Betti numbers of real varieties. *Proc. AMS*, 15.
- Basu, S., Pollack, R., Roy, M.-F. (2006). *Algorithms in Real Algebraic Geometry*. Springer.
- Łojasiewicz, S. (1965). Ensembles semi-analytiques. IHES.
- Eisenbud, D. (1995). *Commutative Algebra*. Springer.
