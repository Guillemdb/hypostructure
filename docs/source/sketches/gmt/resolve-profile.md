# RESOLVE-Profile: Profile Classification Trichotomy — GMT Translation

## Original Statement (Hypostructure)

After concentration, profiles classify into exactly one of three types: finite library membership, tame stratification, or classification failure.

## GMT Setting

**Blow-Up Sequence:** $\{T_j\} \subset \mathbf{I}_k(\mathbb{R}^n)$ — rescaled currents at concentration point

**Profile:** $T_\infty \in \mathbf{I}_k(\mathbb{R}^n)$ — limit current (tangent cone)

**Canonical Library:** $\mathcal{L} = \{C_1, \ldots, C_N\}$ — finite list of classified cones

**Tame Family:** $\mathcal{F}$ — definable family in an o-minimal structure

## GMT Statement

**Theorem (Profile Trichotomy).** Let $\{T_j\}$ be a blow-up sequence with bounded mass converging to a profile $T_\infty$. Then exactly one holds:

**Case 1 (Library):** $T_\infty \in \mathcal{L}$ — profile belongs to finite canonical list

**Case 2 (Tame):** $T_\infty \in \mathcal{F} \setminus \mathcal{L}$ — profile belongs to definable family with finite stratification

**Case 3 (Wild):** $T_\infty \notin \mathcal{F}$ — profile exhibits wildness (chaotic, undecidable, or turbulent structure)

## Proof Sketch

### Step 1: Profile Extraction via Compactness

**Federer-Fleming Compactness (1960):** Let $\{T_j\} \subset \mathbf{I}_k(\mathbb{R}^n)$ satisfy:
$$\sup_j (\mathbf{M}(T_j) + \mathbf{M}(\partial T_j)) \leq \Lambda$$

Then there exists a subsequence $T_{j_l}$ and $T_\infty \in \mathbf{I}_k(\mathbb{R}^n)$ such that:
$$T_{j_l} \to T_\infty \quad \text{in flat norm on compact sets}$$

**Reference:** Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72, 458-520.

### Step 2: Tangent Cone Classification

**Definition:** A current $C \in \mathbf{I}_k(\mathbb{R}^n)$ is a **cone** if:
$$(\eta_{0,\lambda})_\# C = C \quad \text{for all } \lambda > 0$$

**Dimension Stratification (White, 1997):** The singular set of a cone $C$ satisfies:
$$\text{sing}(C) = S_0 \cup S_1 \cup \cdots \cup S_{k-2}$$

where $\dim_{\mathcal{H}}(S_j) \leq j$.

**Reference:** White, B. (1997). Stratification of minimal surfaces. *J. reine angew. Math.*, 488, 1-35.

### Step 3: Canonical Library Construction

**Library Definition:** The canonical library $\mathcal{L}$ consists of cones $C$ such that:
1. $C$ is area-minimizing (or stationary)
2. $C$ is isolated in the moduli space (discrete automorphism group)
3. $\mathbf{M}(C \cap B_1) \leq \Lambda_0$ for fixed threshold

**Finiteness (Simon, 1983):** For area-minimizing cones in $\mathbb{R}^n$:
$$|\mathcal{L}| \leq N(n, \Lambda_0)$$

The bound follows from:
- Dimension reduction: each cone has lower-dimensional singular set
- Compactness: energy bound limits complexity
- Isolation: Łojasiewicz-type uniqueness separates cones

**Reference:** Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.

### Step 4: Case 1 — Library Membership

**Membership Test:** For profile $T_\infty$, check:
$$d_{\text{flat}}(T_\infty, C_i) < \varepsilon_{\text{lib}} \quad \text{for some } C_i \in \mathcal{L}$$

**Uniqueness (Allard, 1972):** If $T_\infty$ is $\varepsilon$-close to a unique cone $C_i$ with multiplicity 1, then:
$$T_\infty = C_i$$

by Allard's regularity and uniqueness theorem.

**Reference:** Allard, W. K. (1972). On the first variation of a varifold. *Ann. of Math.*, 95, 417-491.

### Step 5: Case 2 — Tame Family (O-Minimal)

**O-Minimal Structure (van den Dries, 1998):** An o-minimal expansion of $(\mathbb{R}, <, +, \cdot)$ is a collection of definable sets satisfying:
1. Boolean closure
2. Projection (Tarski-Seidenberg)
3. One-dimensional definable sets are finite unions of points and intervals

**Tame Cone Families:** A family $\mathcal{F}$ of cones is **tame** if:
$$\mathcal{F} = \{C_\theta : \theta \in \Theta\}$$

where $\Theta \subset \mathbb{R}^m$ is definable in an o-minimal structure and $\theta \mapsto C_\theta$ is definable.

**Stratification (Kurdyka-Parusiński, 1994):** Every definable family admits a Whitney stratification:
$$\mathcal{F} = \bigsqcup_{i=1}^N S_i$$

where each $S_i$ is a smooth manifold.

**Reference:**
- van den Dries, L. (1998). *Tame Topology and O-minimal Structures*. Cambridge.
- Kurdyka, K. (1988). Points réguliers d'un sous-analytique. *Ann. Inst. Fourier*, 38, 133-156.

### Step 6: Case 3 — Wild Profiles

**Wildness Criteria:** A profile $T_\infty$ is **wild** if:
1. **Fractal dimension:** $\dim_{\mathcal{H}}(\text{sing}(T_\infty))$ is non-integer
2. **Positive Lyapunov exponent:** The blow-up dynamics are chaotic
3. **Undecidable structure:** The profile encodes a computationally undecidable problem

**Example (Smale's Horseshoe):** The Smale horseshoe produces a Cantor set of homoclinic tangencies with:
$$\dim_{\mathcal{H}}(\text{sing}) = \frac{\log 2}{\log 3}$$

Such profiles cannot belong to an o-minimal family.

**Reference:** Smale, S. (1967). Differentiable dynamical systems. *Bull. AMS*, 73, 747-817.

### Step 7: Trichotomy Exhaustiveness

**Mutual Exclusion:**
- Case 1: $T_\infty$ is isolated, finite automorphisms
- Case 2: $T_\infty$ belongs to continuous family, tame
- Case 3: $T_\infty$ exhibits genuine complexity

**Exhaustiveness:** Every cone either:
- Is isolated (Case 1)
- Belongs to a tame continuous family (Case 2)
- Exhibits wild behavior (Case 3)

This trichotomy is exhaustive by the structure theory of o-minimal sets.

**Certificate Production:**
- Case 1: $(T_\infty, \mathcal{L}, T_\infty = C_i)$
- Case 2: $(T_\infty, \mathcal{F}, \text{stratification data})$
- Case 3: $(T_\infty, \text{wildness witness})$

## Key GMT Inequalities Used

1. **Federer-Fleming Compactness:**
   $$\sup_j \mathbf{M}(T_j) \leq \Lambda \implies T_{j_k} \to T_\infty$$

2. **Allard's Uniqueness:**
   $$d_{\text{flat}}(T, C) < \varepsilon \text{ and } \Theta(T, 0) = \Theta(C, 0) \implies T = C$$

3. **O-Minimal Cell Decomposition:**
   $$\text{Definable } X \subset \mathbb{R}^n = \bigsqcup_i C_i \text{ (cells)}$$

4. **Łojasiewicz Isolation:**
   $$|\nabla \Phi|(x) \geq c |\Phi(x) - \Phi(x_0)|^{1-\theta} \implies x_0 \text{ isolated critical point}$$

## Literature References

- Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72, 458-520.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
- White, B. (1997). Stratification of minimal surfaces. *J. reine angew. Math.*, 488, 1-35.
- van den Dries, L. (1998). *Tame Topology and O-minimal Structures*. Cambridge.
- De Lellis, C., Spadaro, E. (2016). Regularity of area-minimizing currents I-III. *Ann. of Math.*, *GAFA*.
- Lions, P.-L. (1984). Concentration-compactness. *Ann. Inst. H. Poincaré*.
- Merle, F., Zaag, H. (1998). Optimal estimates for blowup rate and behavior. *Duke Math. J.*, 94, 293-319.
