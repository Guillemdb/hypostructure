# UP-Censorship: Causal Censor Promotion — GMT Translation

## Original Statement (Hypostructure)

The causal censor promotion establishes that certain information cannot propagate beyond barriers, preventing leakage of singular behavior into regular regions.

## GMT Setting

**Barrier:** $B \subset M$ — hypersurface separating regions

**Causal Structure:** Information propagation respecting barrier

**Censorship:** Singular behavior behind barrier doesn't affect regular region

## GMT Statement

**Theorem (Causal Censor Promotion).** Let $M = M_{\text{reg}} \sqcup B \sqcup M_{\text{sing}}$ be a decomposition with barrier $B$. If:

1. **Barrier Condition:** $\text{Cap}_{1,2}(B) > 0$ or $B$ has positive codimension

2. **Flow Respect:** The gradient flow $\varphi_t$ respects the barrier

3. **No Tunneling:** No trajectory crosses $B$ from $M_{\text{sing}}$ to $M_{\text{reg}}$

Then singularities in $M_{\text{sing}}$ do not affect the regular flow in $M_{\text{reg}}$.

## Proof Sketch

### Step 1: Barrier as Separating Hypersurface

**Hypersurface Definition:** $B = f^{-1}(0)$ for smooth $f: M \to \mathbb{R}$ with $\nabla f \neq 0$ on $B$.

**Separation:** $M_{\text{reg}} = \{f > 0\}$, $M_{\text{sing}} = \{f < 0\}$

**Reference:** Lee, J. M. (2012). *Introduction to Smooth Manifolds*. Springer.

### Step 2: Capacity Barrier

**Sobolev Capacity:** If $\text{Cap}_{1,2}(B) > 0$:
$$\int_M |\nabla u|^2 \, d\mu \geq c > 0$$

for any $u$ crossing the barrier (i.e., $u|_{M_{\text{reg}}} > 0$, $u|_{M_{\text{sing}}} < 0$).

**Energy Cost:** Crossing the barrier requires positive energy.

**Reference:** Adams, D., Hedberg, L. (1996). *Function Spaces and Potential Theory*. Springer.

### Step 3: Maximum Principle

**Parabolic Maximum Principle:** For heat-type flows:
$$\partial_t u \leq \Delta u \implies \max_{M \times [0,T]} u = \max_{M \times \{0\} \cup \partial M \times [0,T]} u$$

**Reference:** Evans, L. C. (2010). *Partial Differential Equations*. AMS.

**Application:** If $u(x, 0) \leq 0$ in $M_{\text{sing}}$ and barrier $B$ has $u = 0$, then $u$ remains $\leq 0$ in $M_{\text{sing}}$ for all time.

**Consequence:** Perturbations in $M_{\text{sing}}$ don't propagate to $M_{\text{reg}}$.

### Step 4: Domain of Dependence

**Finite Speed:** For hyperbolic systems, information propagates at finite speed $c$.

**Domain of Dependence:**
$$D(x, t) = \{(y, s) : s \leq t, |y - x| \leq c(t - s)\}$$

**Barrier Respect:** If $D(x, t) \cap M_{\text{sing}} = \emptyset$ for $x \in M_{\text{reg}}$:
$$u(x, t) \text{ depends only on initial data in } M_{\text{reg}}$$

**Reference:** John, F. (1982). *Partial Differential Equations*. Springer.

### Step 5: Censorship in Black Hole Physics

**Cosmic Censorship (Penrose, 1969):** Singularities are hidden behind event horizons.

**Reference:** Penrose, R. (1969). Gravitational collapse: the role of general relativity. *Riv. Nuovo Cimento*, 1, 252-276.

**GMT Analogue:** Geometric singularities (tangent cones) are "censored" from affecting regular evolution:
- The singular set $\Sigma = \text{sing}(T)$ is the "singularity"
- The barrier $B = \partial B_\varepsilon(\Sigma)$ is the "horizon"
- Regular evolution in $M \setminus B_\varepsilon(\Sigma)$ is unaffected

### Step 6: Flow Barrier Condition

**Barrier Preservation:** The flow $\varphi_t$ **preserves** barrier $B$ if:
$$\varphi_t(B) \subset B \quad \text{or} \quad \varphi_t(M_{\text{reg}}) \subset M_{\text{reg}}$$

**Sufficient Condition:** $\nabla \Phi \cdot \nabla f \geq 0$ on $B$ (gradient points into $M_{\text{sing}}$).

**Consequence:** Flow from $M_{\text{reg}}$ stays in $M_{\text{reg}}$.

### Step 7: No Tunneling Condition

**Tunneling:** A trajectory $\gamma: [0, T] \to M$ "tunnels" if:
$$\gamma(0) \in M_{\text{sing}}, \quad \gamma(T) \in M_{\text{reg}}$$

**No Tunneling Condition:** Such trajectories don't exist for the gradient flow.

*Proof:* By energy monotonicity:
- $\Phi(\gamma(T)) \leq \Phi(\gamma(0))$
- If $M_{\text{sing}}$ has higher energy, tunneling to lower-energy $M_{\text{reg}}$ is forbidden

### Step 8: Quantitative Censorship

**Decay Rate:** Influence of $M_{\text{sing}}$ on $M_{\text{reg}}$ decays:
$$|u_{\text{reg}}(x, t) - u_{\text{reg}}^0(x, t)| \leq C e^{-\alpha d(x, B)}$$

where:
- $u_{\text{reg}}^0$ is solution ignoring $M_{\text{sing}}$
- $\alpha > 0$ is censorship rate
- $d(x, B)$ is distance to barrier

### Step 9: Singular Set Censorship

**Theorem (Singular Censorship):** Under soft permits, the singular set $\Sigma$ doesn't affect regular evolution:

1. **Codimension:** $\dim(\Sigma) \leq k - 2$
2. **Capacity:** $\text{Cap}_{1,2}(\Sigma) = 0$
3. **Removability:** Solutions in $M \setminus \Sigma$ extend uniquely across $\Sigma$

*Proof:* By capacity-dimension relation (Federer, 1969):
$$\dim(\Sigma) \leq k - 2 \implies \text{Cap}_{1,2}(\Sigma) = 0$$

Zero-capacity sets are removable for Sobolev functions.

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer.

### Step 10: Compilation Theorem

**Theorem (Causal Censor Promotion):**

1. **Barrier Separation:** Positive-capacity barriers separate regions

2. **No Tunneling:** Gradient flow respects barriers

3. **Singular Censorship:** Codimension-2 singular sets don't affect evolution

4. **Decay:** Influence across barrier decays exponentially

**Applications:**
- Singularities in MCF don't affect distant regular flow
- Ricci flow singularities are topologically localized
- Harmonic map singularities are isolated

## Key GMT Inequalities Used

1. **Capacity Barrier:**
   $$\text{Cap}_{1,2}(B) > 0 \implies \text{energy cost to cross}$$

2. **Maximum Principle:**
   $$\partial_t u \leq \Delta u \implies \max u \text{ on boundary}$$

3. **Domain of Dependence:**
   $$|D(x, t)| \leq c \cdot t^n$$

4. **Capacity-Dimension:**
   $$\dim(\Sigma) \leq n - 2 \implies \text{Cap}_{1,2}(\Sigma) = 0$$

## Literature References

- Adams, D., Hedberg, L. (1996). *Function Spaces and Potential Theory*. Springer.
- Evans, L. C. (2010). *Partial Differential Equations*. AMS.
- John, F. (1982). *Partial Differential Equations*. Springer.
- Penrose, R. (1969). Gravitational collapse. *Riv. Nuovo Cimento*, 1.
- Federer, H. (1969). *Geometric Measure Theory*. Springer.
- Lee, J. M. (2012). *Introduction to Smooth Manifolds*. Springer.
