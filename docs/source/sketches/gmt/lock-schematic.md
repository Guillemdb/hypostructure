# LOCK-Schematic: Semialgebraic Exclusion Lock — GMT Translation

## Original Statement (Hypostructure)

The semialgebraic exclusion lock shows that semialgebraic constraints on configuration space create effective barriers, excluding regions definable by polynomial inequalities.

## GMT Setting

**Semialgebraic Set:** Defined by finite Boolean combinations of polynomial inequalities

**Exclusion:** Flow cannot enter certain semialgebraic regions

**O-minimal Structure:** Tame topology ensuring finiteness properties

## GMT Statement

**Theorem (Semialgebraic Exclusion Lock).** For flow on algebraic varieties:

1. **Semialgebraic Constraint:** $\mathcal{C} = \{x : P_1(x) \geq 0, \ldots, P_k(x) \geq 0\}$

2. **Flow Exclusion:** If $T_0 \notin \mathcal{C}$ and $\mathcal{C}$ is flow-invariant, then $T_t \notin \mathcal{C}$

3. **Lock:** Polynomial barriers cannot be crossed by polynomial flows

## Proof Sketch

### Step 1: Semialgebraic Sets

**Definition:** $S \subset \mathbb{R}^n$ is semialgebraic if:
$$S = \bigcup_{i=1}^p \bigcap_{j=1}^{q_i} \{x : f_{ij}(x) \star_{ij} 0\}$$

where $f_{ij} \in \mathbb{R}[x_1, \ldots, x_n]$ and $\star_{ij} \in \{>, <, =\}$.

**Reference:** Bochnak, J., Coste, M., Roy, M.-F. (1998). *Real Algebraic Geometry*. Springer.

### Step 2: Tarski-Seidenberg Principle

**Theorem:** Projection of semialgebraic set is semialgebraic:
$$\pi: \mathbb{R}^{n+1} \to \mathbb{R}^n$$

**Quantifier Elimination:** First-order formulas over reals equivalent to quantifier-free.

**Reference:** Tarski, A. (1951). *A Decision Method for Elementary Algebra and Geometry*. RAND.

### Step 3: O-minimal Structures

**Definition:** An o-minimal structure on $\mathbb{R}$ is collection of definable sets satisfying:
- Boolean closure
- Projection closure
- Definable sets in $\mathbb{R}$ are finite unions of points and intervals

**Reference:** van den Dries, L. (1998). *Tame Topology and O-minimal Structures*. Cambridge.

**Semialgebraic is O-minimal:** $\mathbb{R}_{\text{alg}}$ (semialgebraic) is o-minimal.

### Step 4: Cell Decomposition

**Theorem:** Every semialgebraic set has cell decomposition:
$$S = \bigsqcup_{i=1}^N C_i$$

where $C_i$ are cells (homeomorphic to $\mathbb{R}^{d_i}$).

**Reference:** Collins, G. E. (1975). Quantifier elimination for real closed fields by cylindrical algebraic decomposition. *Lecture Notes in Computer Science*, 33.

**Finiteness:** Finitely many cells of bounded complexity.

### Step 5: Flow on Algebraic Varieties

**Polynomial Flow:** For polynomial vector field $V(x) = (P_1(x), \ldots, P_n(x))$:
$$\frac{dx}{dt} = V(x)$$

**Semialgebraic Solutions:** Integral curves are semialgebraic (Khovanskii theory).

**Reference:** Khovanskii, A. G. (1991). *Fewnomials*. AMS.

### Step 6: Invariant Semialgebraic Sets

**Definition:** $S$ is flow-invariant if $\varphi_t(S) \subset S$ for all $t \geq 0$.

**Criterion:** $S$ invariant iff $V(x) \cdot \nabla \chi_S(x) \leq 0$ on $\partial S$ (inward pointing).

**Lyapunov-like:** $P(x) \geq 0$ defines invariant set if $\frac{d}{dt}P(x(t)) \leq 0$ when $P = 0$.

### Step 7: Exclusion by Barrier Function

**Barrier Function:** $B: \mathbb{R}^n \to \mathbb{R}$ polynomial with:
- $B(x) \leq 0$ on allowed region
- $\dot{B}(x) \leq 0$ along flow when $B(x) = 0$

**Exclusion:** If $B(x_0) < 0$, then $B(x_t) < 0$ for all $t \geq 0$.

**Reference:** Prajna, S., Jadbabaie, A. (2004). Safety verification of hybrid systems using barrier certificates. *HSCC 2004*.

### Step 8: Effective Bounds

**Theorem (Gabrielov-Vorobjov):** Complexity of semialgebraic sets bounded by:
- Number of polynomials
- Degrees of polynomials
- Dimension

**Reference:** Gabrielov, A., Vorobjov, N. (2004). Complexity of computations with Pfaffian and Noetherian functions. *Normal Forms, Bifurcations and Finiteness Problems*.

**Effective Lock:** Barriers have bounded complexity.

### Step 9: GMT Application

**Singular Set Constraint:** For $T \in \mathbf{I}_k(M)$:
$$\text{sing}(T) \subset V$$

where $V$ is algebraic variety.

**Semialgebraic Control:** If $V$ is defined by degree $d$ polynomials, singular set has:
- At most $C(n,d)$ components
- Volume bounded by $C(n,d)$

### Step 10: Compilation Theorem

**Theorem (Semialgebraic Exclusion Lock):**

1. **Semialgebraic:** Barrier regions defined by polynomial inequalities

2. **Tarski-Seidenberg:** Semialgebraic closure under projection

3. **Flow Invariance:** Polynomial flows preserve semialgebraic barriers

4. **Lock:** Initial conditions outside barrier remain outside

**Applications:**
- Effective singular set control
- Algebraic flow analysis
- Decidability of reachability

## Key GMT Inequalities Used

1. **Cell Complexity:**
   $$\#\text{cells}(S) \leq C(n, d, k)$$

2. **Barrier Invariance:**
   $$B(x_0) < 0, \dot{B}|_{B=0} \leq 0 \implies B(x_t) < 0$$

3. **Bézout Bound:**
   $$\#(V_1 \cap V_2) \leq \deg(V_1) \cdot \deg(V_2)$$

4. **Component Bound:**
   $$b_0(V) \leq d(2d-1)^{n-1}$$

## Literature References

- Bochnak, J., Coste, M., Roy, M.-F. (1998). *Real Algebraic Geometry*. Springer.
- Tarski, A. (1951). *Decision Method for Elementary Algebra*. RAND.
- van den Dries, L. (1998). *Tame Topology and O-minimal Structures*. Cambridge.
- Collins, G. E. (1975). Cylindrical algebraic decomposition. *LNCS*, 33.
- Khovanskii, A. G. (1991). *Fewnomials*. AMS.
- Prajna, S., Jadbabaie, A. (2004). Barrier certificates. *HSCC 2004*.
