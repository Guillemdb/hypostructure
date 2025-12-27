# UP-Capacity: Capacity Promotion — GMT Translation

## Original Statement (Hypostructure)

The capacity promotion upgrades capacity bounds from local to global, ensuring that exceptional sets have controlled size throughout the domain.

## GMT Setting

**Capacity:** $\text{Cap}_{p,k}(E)$ — $(p, k)$-Sobolev capacity

**Local Bound:** $\text{Cap}_{p,k}(E \cap B_r(x)) \leq \varepsilon r^{n-pk}$

**Global Bound:** $\text{Cap}_{p,k}(E) \leq C$

## GMT Statement

**Theorem (Capacity Promotion).** If local capacity bounds hold:
$$\text{Cap}_{1,2}(\Sigma \cap B_r(x)) \leq \varepsilon r^{n-2}$$

for all $(x, r)$, then:

1. **Global Bound:** $\text{Cap}_{1,2}(\Sigma) \leq C(\varepsilon, n) \cdot \mathcal{H}^{n-2}(\Sigma)$

2. **Removability:** If $\dim_{\mathcal{H}}(\Sigma) \leq n - 2$, then $\text{Cap}_{1,2}(\Sigma) = 0$

3. **Hausdorff Equivalence:** Capacity and Hausdorff content are equivalent

## Proof Sketch

### Step 1: Capacity Definition

**Sobolev Capacity (Adams-Hedberg, 1996):**
$$\text{Cap}_{1,2}(E) := \inf\left\{\int_{\mathbb{R}^n} |\nabla u|^2 + u^2 \, dx : u \geq 1 \text{ on } E, u \in W^{1,2}\right\}$$

**Reference:** Adams, D., Hedberg, L. (1996). *Function Spaces and Potential Theory*. Springer.

**Alternative:** Using Riesz potential:
$$\text{Cap}_{1,2}(E) = \inf\left\{\int |f|^2 : I_1 f \geq 1 \text{ on } E\right\}$$

where $I_\alpha f(x) = \int |x-y|^{-(n-\alpha)} f(y) \, dy$.

### Step 2: Local-to-Global Aggregation

**Covering Argument:** Cover $\Sigma$ by balls:
$$\Sigma \subset \bigcup_{i=1}^N B_{r_i}(x_i)$$

**Subadditivity:** Capacity is subadditive:
$$\text{Cap}_{1,2}(\Sigma) \leq \sum_{i=1}^N \text{Cap}_{1,2}(\Sigma \cap B_{r_i}(x_i))$$

**Local Bound Application:**
$$\text{Cap}_{1,2}(\Sigma) \leq \sum_{i=1}^N \varepsilon r_i^{n-2}$$

### Step 3: Vitali Covering

**Vitali Covering Lemma:** For any covering $\{B_{r_i}(x_i)\}$, there exists disjoint subcollection $\{B_{r_{i_k}}(x_{i_k})\}$ with:
$$\Sigma \subset \bigcup_k B_{5r_{i_k}}(x_{i_k})$$

**Reference:** Mattila, P. (1995). *Geometry of Sets and Measures in Euclidean Spaces*. Cambridge.

**Capacity Bound:**
$$\text{Cap}_{1,2}(\Sigma) \leq C(n) \sum_k \varepsilon (5r_{i_k})^{n-2} = C(n) \varepsilon \mathcal{H}^{n-2}_\delta(\Sigma)$$

### Step 4: Capacity-Hausdorff Relation

**Frostman's Lemma (1935):** For $E \subset \mathbb{R}^n$:
$$\mathcal{H}^s(E) > 0 \iff \exists \mu \text{ with } \mu(B_r(x)) \leq r^s$$

**Reference:** Frostman, O. (1935). Potentiel d'équilibre et capacité des ensembles. *Lund Univ. Thesis*.

**Capacity-Dimension:**
$$\dim_{\mathcal{H}}(E) \leq s \implies \text{Cap}_{1, 2}(E) = 0 \text{ if } s < n - 2$$

**Reference:** Adams, D., Hedberg, L. (1996). *Function Spaces and Potential Theory*. Springer. [Chapter 5]

### Step 5: Dimensional Threshold

**Critical Dimension:** For $(1, 2)$-capacity:
$$\text{Cap}_{1,2}(E) = 0 \iff \dim_{\mathcal{H}}(E) \leq n - 2$$

**Proof (⇐):** If $\dim_{\mathcal{H}}(E) < n - 2$:
$$\mathcal{H}^{n-2}(E) = 0 \implies \text{Cap}_{1,2}(E) = 0$$

by capacity-Hausdorff comparison.

**Proof (⇒):** If $\dim_{\mathcal{H}}(E) > n - 2$:
$$\mathcal{H}^{n-2+\delta}(E) > 0 \implies \text{Cap}_{1,2}(E) > 0$$

by Frostman.

### Step 6: Removability

**Removable Sets:** A set $E$ is **removable** for $W^{1,2}$ if:
$$u \in W^{1,2}(M \setminus E) \implies u \text{ extends to } W^{1,2}(M)$$

**Theorem (Removability):** $E$ is removable iff $\text{Cap}_{1,2}(E) = 0$.

*Proof:* Zero capacity means test functions can "jump" across $E$ with zero Sobolev cost.

### Step 7: Quantitative Promotion

**Theorem:** If for all $r \leq R$:
$$\text{Cap}_{1,2}(\Sigma \cap B_r(x)) \leq \varepsilon r^{n-2}$$

then:
$$\text{Cap}_{1,2}(\Sigma \cap B_R(x_0)) \leq C(n) \varepsilon R^{n-2}$$

**Constant:** $C(n) = 5^{n-2} \cdot C_{\text{Vitali}}(n)$

### Step 8: Capacity and PDEs

**Application to Regularity:** If singular set $\Sigma$ has:
$$\text{Cap}_{1,2}(\Sigma) = 0$$

then solutions of elliptic PDEs extend across $\Sigma$:
$$-\Delta u = f \text{ on } M \setminus \Sigma \implies -\Delta u = f \text{ on } M$$

**Reference:** Kilpeläinen, T. (1994). Weighted Sobolev spaces and capacity. *Ann. Acad. Sci. Fenn. Math.*, 19.

### Step 9: Higher Order Capacities

**$(k, p)$-Capacity:**
$$\text{Cap}_{k,p}(E) := \inf\left\{\|u\|_{W^{k,p}}^p : u \geq 1 \text{ on } E\right\}$$

**Critical Dimension:**
$$\text{Cap}_{k,p}(E) = 0 \iff \dim_{\mathcal{H}}(E) \leq n - kp$$

**Promotion:** Local $(k, p)$-capacity bounds promote to global similarly.

### Step 10: Compilation Theorem

**Theorem (Capacity Promotion):**

1. **Local-to-Global:** Local capacity bounds aggregate to global

2. **Hausdorff Equivalence:** $\text{Cap}_{1,2} \asymp \mathcal{H}^{n-2}$-content for $(n-2)$-sets

3. **Removability:** Zero-capacity sets are removable

4. **Dimensional:** $\dim_{\mathcal{H}} \leq n - 2 \iff \text{Cap}_{1,2} = 0$

**Applications:**
- Singular set bounds in regularity theory
- Removable singularities for PDEs
- Capacity estimates in GMT

## Key GMT Inequalities Used

1. **Subadditivity:**
   $$\text{Cap}(E_1 \cup E_2) \leq \text{Cap}(E_1) + \text{Cap}(E_2)$$

2. **Scaling:**
   $$\text{Cap}_{1,2}(\lambda E) = \lambda^{n-2} \text{Cap}_{1,2}(E)$$

3. **Dimension Threshold:**
   $$\dim_{\mathcal{H}}(E) \leq n-2 \implies \text{Cap}_{1,2}(E) = 0$$

4. **Local Bound:**
   $$\text{Cap}_{1,2}(E \cap B_r) \leq C r^{n-2}$$

## Literature References

- Adams, D., Hedberg, L. (1996). *Function Spaces and Potential Theory*. Springer.
- Mattila, P. (1995). *Geometry of Sets and Measures*. Cambridge.
- Frostman, O. (1935). Potentiel d'équilibre et capacité. *Lund Thesis*.
- Maz'ya, V. G. (1985). *Sobolev Spaces*. Springer.
- Kilpeläinen, T. (1994). Weighted Sobolev spaces and capacity. *Ann. Acad. Sci. Fenn. Math.*, 19.
- Federer, H. (1969). *Geometric Measure Theory*. Springer.
