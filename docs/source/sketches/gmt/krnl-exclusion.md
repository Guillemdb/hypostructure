# KRNL-Exclusion: Principle of Structural Exclusion — GMT Translation

## Original Statement (Hypostructure)

If the Hom-set from the universal bad pattern to a system's hypostructure is empty, then no singularity can exist. This categorical obstruction provides a global regularity guarantee.

## GMT Setting

**Ambient Space:** $(M^n, g)$ — complete Riemannian manifold of dimension $n$

**Currents:** $\mathbf{I}_k(M)$ — space of integral $k$-currents with finite mass

**Varifolds:** $\mathbf{V}_k(M)$ — space of $k$-dimensional varifolds

**Bad Pattern:** $\Sigma_{\text{bad}} \in \mathbf{I}_{n-2}(M)$ — universal singularity model (e.g., the Simons cone)

**Admissible Class:** $\mathcal{A} \subset \mathbf{I}_n(M)$ — currents satisfying structural bounds (mass, first variation, etc.)

## GMT Statement

**Theorem (Structural Exclusion via Current Non-Embedding).** Let $(M^n, g)$ be a complete Riemannian manifold and $\mathcal{A} \subset \mathbf{I}_n(M)$ an admissible class of integral currents satisfying:

- **(Mass Bound)** $\mathbf{M}(T) \leq \Lambda$ for all $T \in \mathcal{A}$
- **(First Variation Bound)** $\|\delta T\|(M) \leq \Lambda$ for all $T \in \mathcal{A}$
- **(Density Bounds)** $\Theta^n(T, x) \in [\theta_{\min}, \theta_{\max}]$ for $\|T\|$-a.e. $x$

Let $\mathcal{G} = \{[\Sigma_i]\}_{i \in I}$ be the germ set of singularity models — a finite collection of $(n-2)$-dimensional cones in $\mathbb{R}^n$ representing possible tangent cones at singular points.

Define the **universal bad pattern** as the disjoint union current:
$$\mathbf{B}_{\text{univ}} := \bigsqcup_{i \in I} \llbracket C_i \rrbracket \in \mathbf{I}_{n-2}(\mathbb{R}^n)$$

**Exclusion Criterion:** If for all $T \in \mathcal{A}$ and all $x \in \text{sing}(T)$:
$$\text{VarTan}(T, x) \cap \mathcal{G} = \emptyset$$

(no tangent cone at any singular point belongs to the germ set), then $\text{sing}(T) = \emptyset$ — every $T \in \mathcal{A}$ is regular.

## Proof Sketch

### Step 1: Stratification of the Singular Set

By Federer's dimension reduction principle, the singular set of any $T \in \mathcal{A}$ admits a stratification:
$$\text{sing}(T) = S_0 \cup S_1 \cup \cdots \cup S_{n-2}$$

where $\dim_{\mathcal{H}}(S_k) \leq k$ and points in $S_k \setminus S_{k-1}$ have tangent cones that split off an $(n-k)$-dimensional factor.

**Key GMT Fact (Federer):** For integral currents with bounded first variation:
$$\mathcal{H}^{n-2}(\text{sing}(T)) < \infty$$

The singular set is at most $(n-2)$-rectifiable.

### Step 2: Tangent Cone Extraction via Blow-Up

For $x_0 \in \text{sing}(T)$, define the **blow-up sequence**:
$$T_{x_0, r_j} := (\eta_{x_0, r_j})_\# T$$

where $\eta_{x_0, r}(y) = (y - x_0)/r$ is the dilation centered at $x_0$, and $r_j \to 0$.

By the monotonicity formula for currents with bounded mean curvature:
$$\frac{\|T\|(B_r(x_0))}{r^n} \leq e^{C\Lambda r} \cdot \frac{\|T\|(B_s(x_0))}{s^n} \quad \text{for } 0 < r < s$$

the density $\Theta^n(T, x_0) = \lim_{r \to 0} \frac{\|T\|(B_r(x_0))}{\omega_n r^n}$ exists.

**Compactness (Federer-Fleming):** The sequence $\{T_{x_0, r_j}\}$ has uniformly bounded mass in any ball. By the compactness theorem for integral currents, there exists a subsequence converging to a **tangent cone**:
$$T_{x_0, r_{j_k}} \to C \in \mathbf{I}_n(\mathbb{R}^n)$$

in the flat topology, where $C$ is a cone (invariant under positive dilations).

### Step 3: Classification of Tangent Cones (Germ Set Construction)

The tangent cone $C$ at a singular point must satisfy:
- $C$ is an integral current in $\mathbb{R}^n$
- $C$ is a cone: $(\eta_{0, \lambda})_\# C = C$ for all $\lambda > 0$
- $\partial C = 0$ (if $\partial T = 0$) or $\partial C$ is the tangent cone of $\partial T$
- $\mathbf{M}(C \llcorner B_1(0)) = \Theta^n(T, x_0) \cdot \omega_n$

**Germ Set Finiteness:** By subcriticality assumptions (analogous to $K_{\text{SC}_\lambda}^+$), the germ set $\mathcal{G}$ is finite:

$$|\mathcal{G}| \leq N(\Lambda, n, \theta_{\min}, \theta_{\max})$$

This follows from the Łojasiewicz-type uniqueness of tangent cones under energy bounds. Specifically, the **isolation theorem** (Simon, Allard): if $C_1, C_2$ are two distinct tangent cones with the same density, then $\text{dist}(C_1, C_2) \geq \varepsilon_0(\Lambda, n) > 0$ in the flat metric on $B_1(0)$.

### Step 4: Morphism Obstruction as Current Non-Embedding

The categorical condition $\text{Hom}(\mathbb{H}_{\text{bad}}, \mathbb{H}(T)) = \emptyset$ translates to: there exists no **linking current** $L \in \mathbf{I}_{n-1}(M)$ such that:
$$\partial L = \llbracket \Sigma_{\text{bad}} \rrbracket - (\iota_T)_\# \llbracket \Sigma_T \rrbracket$$

where $\Sigma_T$ is any singular stratum of $T$ and $\iota_T: \Sigma_T \hookrightarrow M$ is the inclusion.

**Homological Interpretation:** The obstruction is cohomological. Let $[\Sigma_{\text{bad}}] \in H_{n-2}(M; \mathbb{Z})$ be the homology class of the universal bad pattern. The condition $\text{Hom} = \emptyset$ means:
$$[\Sigma_{\text{bad}}] \neq [\iota_T(\Sigma_T)]$$

for any singular stratum $\Sigma_T \subset \text{sing}(T)$.

### Step 5: Capacity Bounds and Rectifiability

The exclusion is enforced by **capacity estimates**. Define the Sobolev $(1,2)$-capacity:
$$\text{Cap}_{1,2}(E) := \inf \left\{ \int_M |\nabla u|^2 \, dV_g : u \geq 1 \text{ on } E, \, u \in H^1(M) \right\}$$

**Key Estimate:** For any tangent cone $C \in \mathcal{G}$:
$$\text{Cap}_{1,2}(C \cap B_1(0)) \geq c_0(n) > 0$$

(non-zero capacity for $(n-2)$-dimensional cones in $\mathbb{R}^n$).

**Exclusion Mechanism:** If $\text{VarTan}(T, x) \cap \mathcal{G} = \emptyset$ for all $x \in \text{sing}(T)$, then at every singular point, the tangent cone has **zero capacity**:
$$\text{Cap}_{1,2}(C_x \cap B_1(0)) = 0$$

But cones of dimension $\leq n-3$ have zero $(1,2)$-capacity in $\mathbb{R}^n$. This forces:
$$\dim(\text{sing}(T)) \leq n - 3$$

### Step 6: Dimension Improvement and Regularity

By Federer's dimension reduction, if $\dim(\text{sing}(T)) \leq n - 3$, we can apply the **Allard regularity theorem**:

**Allard's $\varepsilon$-Regularity:** There exists $\varepsilon = \varepsilon(n, \Lambda) > 0$ such that if $T \in \mathbf{I}_n(M)$ with $\|\delta T\|(B_r(x)) \leq \varepsilon r^{n-1}$ and:
$$\frac{\|T\|(B_r(x))}{\omega_n r^n} \leq 1 + \varepsilon$$

then $T$ is represented by a $C^{1,\alpha}$ graph in $B_{r/2}(x)$.

Since singularities can only occur where the density ratio exceeds $1 + \varepsilon$, and we've excluded all tangent cones in $\mathcal{G}$, the remaining "bad" tangent cones have:
- Either dimension $\leq n - 3$ (removable by capacity)
- Or density $\leq 1 + \varepsilon$ (regular by Allard)

**Conclusion:** $\text{sing}(T) = \emptyset$, completing the exclusion proof.

## Key GMT Inequalities Used

1. **Monotonicity Formula for Currents:**
   $$\frac{d}{dr}\left( e^{Cr} \frac{\|T\|(B_r(x))}{r^n} \right) \geq 0$$

2. **Federer-Fleming Compactness:**
   $$\mathbf{M}(T_j) + \mathbf{M}(\partial T_j) \leq \Lambda \implies T_{j_k} \to T_\infty \text{ in flat norm}$$

3. **Allard's Regularity Estimate:**
   $$\|T\|(B_r(x)) \leq (1 + \varepsilon) \omega_n r^n, \, \|\delta T\|(B_r(x)) \leq \varepsilon r^{n-1} \implies \text{reg}(T) \cap B_{r/2}(x) \neq \emptyset$$

4. **Capacity-Dimension Relation:**
   $$\dim_{\mathcal{H}}(E) \leq n - 2 \implies \text{Cap}_{1,2}(E) = 0$$

## Literature References

- Federer, H. (1969). *Geometric Measure Theory*. Springer. [Chapter 4: Homological Integration Theory]
- Allard, W. K. (1972). On the first variation of a varifold. *Annals of Mathematics*, 95, 417-491.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. Proceedings of the CMA, ANU. [Chapters 6-7]
- De Lellis, C., Spadaro, E. (2016). Regularity of area-minimizing currents III: Blow-up. *Annals of Mathematics*, 183, 577-617.
- White, B. (1997). Stratification of minimal surfaces, mean curvature flows, and harmonic maps. *Journal für die reine und angewandte Mathematik*, 488, 1-35.
