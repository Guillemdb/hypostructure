# UP-Saturation: Saturation Promotion — GMT Translation

## Original Statement (Hypostructure)

The saturation promotion upgrades a local certificate to a global one by showing that the certificate's validity extends to all scales and locations.

## GMT Setting

**Local Certificate:** $K^+_{\text{loc}}$ — certificate valid in neighborhood

**Global Certificate:** $K^+_{\text{glob}}$ — certificate valid everywhere

**Saturation:** Process of extending local to global

## GMT Statement

**Theorem (Saturation Promotion).** If local certificate $K^+_{\text{loc}}(x, r)$ holds for all $(x, r) \in M \times (0, R)$, and:

1. **Consistency:** Local certificates are compatible on overlaps

2. **Scale Coherence:** Certificates are stable under rescaling

3. **Compactness:** The certificate predicate is closed under limits

Then the global certificate $K^+_{\text{glob}}$ holds.

## Proof Sketch

### Step 1: Local-to-Global Framework

**Local Predicate:** $P(x, r)$ — property holds in $B_r(x)$

**Saturation Condition:** $P(x, r)$ for all $(x, r)$ with $r < R$

**Global Property:** $P_{\text{glob}}$ — property holds on all of $M$

**Reference:** Heinonen, J. (2001). *Lectures on Analysis on Metric Spaces*. Springer.

### Step 2: Cover Argument

**Open Cover:** $\{B_r(x) : x \in M, r < R\}$ covers $M$

**Compactness:** If $M$ is compact, extract finite subcover:
$$M \subset \bigcup_{i=1}^N B_{r_i}(x_i)$$

**Local-to-Global:** If $P$ holds on each $B_{r_i}(x_i)$ and is compatible on overlaps, $P$ holds globally.

### Step 3: Partition of Unity

**Partition of Unity:** Functions $\{\rho_i\}$ with:
- $\rho_i \geq 0$, $\text{supp}(\rho_i) \subset B_{r_i}(x_i)$
- $\sum_i \rho_i = 1$ on $M$

**Gluing:** Define global object:
$$T_{\text{glob}} = \sum_i \rho_i \cdot T_{\text{loc}}^{(i)}$$

**Consistency:** Overlap compatibility ensures well-definedness.

**Reference:** Warner, F. W. (1983). *Foundations of Differentiable Manifolds and Lie Groups*. Springer.

### Step 4: Scale Coherence Verification

**Scale-Invariant Property:** $P$ is **scale-coherent** if:
$$P(x, r) \iff P(x, \lambda r) \text{ for } \lambda \in (\lambda_0, \lambda_1)$$

**Monotonicity:** Often $P(x, r) \Rightarrow P(x, r')$ for $r' < r$.

**Extension:** Scale coherence allows extending to all scales $r \in (0, \infty)$.

### Step 5: Compactness Closure

**Limit Stability:** The property $P$ is **closed under limits** if:
$$P(x_n, r_n) \text{ and } (x_n, r_n) \to (x, r) \implies P(x, r)$$

**Consequence:** The set $\{(x, r) : P(x, r)\}$ is closed.

**Saturation:** If closed set contains dense subset, it's everything.

### Step 6: GMT Saturation Examples

**Example 1: ε-Regularity Saturation**

*Local:* $T$ is $\varepsilon$-regular in $B_r(x)$ (small tilt excess)

*Saturation:* If $\varepsilon$-regular everywhere locally, $T$ is globally regular.

**Reference:** Allard, W. K. (1972). On the first variation of a varifold. *Ann. of Math.*, 95.

**Example 2: Monotonicity Saturation**

*Local:* Density ratio $\Theta(T, x, r)$ is monotone in $r$

*Saturation:* Monotonicity at all points implies global density control.

**Reference:** Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.

**Example 3: Curvature Saturation**

*Local:* $|Rm| \leq \kappa$ in $B_r(x)$

*Saturation:* Local curvature bounds saturate to global.

### Step 7: Propagation Argument

**Propagation Lemma:** If $P(x_0, R)$ and $P$ propagates:
$$P(x, r) \land d(x, x') < r/2 \implies P(x', r/2)$$

then $P$ holds in expanding regions.

**Iteration:** Starting from $P(x_0, R)$:
1. $P$ holds in $B_R(x_0)$
2. For $x \in B_{R/2}(x_0)$, $P(x, R/2)$ holds
3. Expand to $B_{3R/2}(x_0)$
4. Iterate to cover all of $M$

### Step 8: Obstruction Analysis

**Failure Modes:** Saturation can fail if:
1. **Incompatibility:** Local certificates conflict on overlaps
2. **Scale Breakdown:** Property fails at some scale
3. **Limit Failure:** Property not closed under limits

**Detection:** Check each failure mode algorithmically.

### Step 9: Quantitative Saturation

**Uniform Bounds:** If local certificate has uniform constants:
$$P(x, r) \text{ with constants } C_1, \ldots, C_k$$

then global certificate has:
$$P_{\text{glob}} \text{ with constants } C_1', \ldots, C_k'$$

where $C_i' = f_i(C_1, \ldots, C_k, \text{cover data})$.

### Step 10: Compilation Theorem

**Theorem (Saturation Promotion):**

1. **Local-to-Global:** Compatible local certificates extend to global

2. **Scale Coherence:** Scale-invariant properties extend to all scales

3. **Compactness Closure:** Limit-closed properties saturate

4. **Quantitative:** Constants in global certificate depend on local constants

**Constructive Content:**
- Algorithm to check compatibility
- Algorithm to extend local to global
- Explicit dependence of global constants on local constants

## Key GMT Inequalities Used

1. **Cover Bound:**
   $$M \subset \bigcup_{i=1}^N B_{r_i}(x_i)$$

2. **Overlap Compatibility:**
   $$P(x, r) \land P(x', r') \land B_r(x) \cap B_{r'}(x') \neq \emptyset \implies \text{compatible}$$

3. **Limit Closure:**
   $$P(x_n, r_n) \to P(x, r)$$

4. **Propagation:**
   $$P(x, r) \implies P(x', r/2) \text{ for } d(x, x') < r/2$$

## Literature References

- Heinonen, J. (2001). *Lectures on Analysis on Metric Spaces*. Springer.
- Warner, F. W. (1983). *Foundations of Differentiable Manifolds*. Springer.
- Allard, W. K. (1972). On the first variation of a varifold. *Ann. of Math.*, 95.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
- Federer, H. (1969). *Geometric Measure Theory*. Springer.
