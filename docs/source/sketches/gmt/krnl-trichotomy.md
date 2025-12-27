# KRNL-Trichotomy: Structural Resolution — GMT Translation

## Original Statement (Hypostructure)

Every trajectory with finite breakdown time classifies into exactly one of three outcomes: Global Existence (dispersion), Global Regularity (concentration with permits satisfied), or Genuine Singularity (concentration with permits violated).

## GMT Setting

**Ambient Space:** $(\mathbb{R}^n, |\cdot|, \mathcal{L}^n)$ — Euclidean space with Lebesgue measure

**Current Space:** $\mathbf{I}_k(\mathbb{R}^n)$ — integral $k$-currents

**Varifold Space:** $\mathbf{V}_k(\mathbb{R}^n)$ — $k$-varifolds with first variation in $L^1_{\text{loc}}$

**Scaling Group:** $G = \mathbb{R}^+ \times \mathbb{R}^n$ acting by $(λ, x_0) \cdot T = (η_{x_0, λ})_\# T$

**Energy Functional:** $\mathbf{M}(T) = \|T\|(\mathbb{R}^n)$ — total mass

## GMT Statement

**Theorem (Trichotomy for Blow-Up Sequences).** Let $\{T_j\}_{j=1}^\infty \subset \mathbf{I}_k(\mathbb{R}^n)$ be a sequence of integral currents with:
- **(Energy Bound)** $\sup_j \mathbf{M}(T_j) \leq \Lambda < \infty$
- **(Boundary Control)** $\sup_j \mathbf{M}(\partial T_j) \leq \Lambda$

Then exactly one of the following holds:

**Case D.D (Dispersion):** The mass disperses to infinity:
$$\forall R > 0: \quad \limsup_{j \to \infty} \|T_j\|(B_R(0)) = 0$$

**Case Reg (Global Regularity):** There exist $(\lambda_j, x_j) \in G$ and $T_\infty \in \mathbf{I}_k(\mathbb{R}^n)$ with $T_\infty$ regular (smooth embedded submanifold) such that:
$$(\eta_{x_j, \lambda_j})_\# T_j \to T_\infty \quad \text{in flat norm}$$

**Case C.E (Genuine Singularity):** There exist $(\lambda_j, x_j) \in G$ and $T_\infty \in \mathbf{I}_k(\mathbb{R}^n)$ with $\text{sing}(T_\infty) \neq \emptyset$ such that:
$$(\eta_{x_j, \lambda_j})_\# T_j \to T_\infty \quad \text{in flat norm}$$

and $T_\infty$ is a singular profile (cone, soliton, or other classifiable singularity type).

## Proof Sketch

### Step 1: Concentration Function and Dichotomy

Define the **concentration function** for the sequence $\{T_j\}$:
$$Q(R) := \limsup_{j \to \infty} \sup_{x \in \mathbb{R}^n} \|T_j\|(B_R(x))$$

**Dichotomy Lemma (Lions-type):** Either:
- (Vanishing) $\lim_{R \to \infty} Q(R) = 0$, or
- (Concentration) $\exists R_0 > 0, \delta > 0$ such that $Q(R_0) \geq \delta$

*Proof:* $Q(R)$ is non-decreasing in $R$. If $Q(\infty) := \lim_{R \to \infty} Q(R) = 0$, we have vanishing. Otherwise, the intermediate value property gives concentration.

**Vanishing $\Rightarrow$ Case D.D:** If $Q(\infty) = 0$, then for any fixed $R$, $\|T_j\|(B_R(0)) \to 0$. This is dispersion.

### Step 2: Concentration Implies Profile Extraction

Assume concentration: $\exists R_0, \delta$ with $Q(R_0) \geq \delta$. Choose sequences $(x_j, r_j)$ realizing the concentration:
$$\|T_j\|(B_{R_0}(x_j)) \geq \delta/2$$

**Scale Selection:** Define the concentration scale:
$$\lambda_j := \sup \left\{ r > 0 : \sup_{x} \|T_j\|(B_r(x)) \leq \delta/2 \right\}$$

By definition, $\lambda_j > 0$ and there exists $y_j$ with $\|T_j\|(B_{\lambda_j}(y_j)) = \delta/2$.

**Rescaled Sequence:** Define $\tilde{T}_j := (\eta_{y_j, \lambda_j})_\# T_j$. Then:
$$\|\tilde{T}_j\|(B_1(0)) = \delta/2, \quad \mathbf{M}(\tilde{T}_j) \leq \Lambda$$

### Step 3: Compactness and Limit Extraction

**Federer-Fleming Compactness:** The sequence $\{\tilde{T}_j\}$ has uniformly bounded mass and boundary mass. There exists a subsequence (still denoted $\tilde{T}_j$) and $T_\infty \in \mathbf{I}_k(\mathbb{R}^n)$ such that:
$$\tilde{T}_j \to T_\infty \quad \text{in flat norm on compact sets}$$

By lower semicontinuity of mass:
$$\|T_\infty\|(B_1(0)) \geq \liminf_{j} \|\tilde{T}_j\|(B_1(0)) = \delta/2 > 0$$

Hence $T_\infty \neq 0$ — this is the **non-trivial profile**.

### Step 4: Regularity vs. Singularity Classification

The limit $T_\infty$ is classified by its singular set:

**Regularity Check (Allard):** At each point $x \in \text{spt}(T_\infty)$, compute the density:
$$\Theta^k(T_\infty, x) = \lim_{r \to 0} \frac{\|T_\infty\|(B_r(x))}{\omega_k r^k}$$

**$\varepsilon$-Regularity Criterion:** If $\Theta^k(T_\infty, x) < 1 + \varepsilon_0$ and the first variation satisfies the tilt-excess bound:
$$\int_{B_r(x)} |H_{T_\infty}|^2 \, d\|T_\infty\| \leq \varepsilon_0 r^{k-2}$$

then $x \in \text{reg}(T_\infty)$.

**Singular Set Characterization:** By Federer's dimension reduction:
$$\text{sing}(T_\infty) = \{x : \Theta^k(T_\infty, x) \geq 1 + \varepsilon_0\}$$

is a closed set with $\mathcal{H}^{k-2}(\text{sing}(T_\infty)) < \infty$.

### Step 5: Case Distinction

**Case Reg (Regular Profile):** If $\text{sing}(T_\infty) = \emptyset$, then $T_\infty$ is represented by a smooth $k$-dimensional embedded submanifold $\Sigma \subset \mathbb{R}^n$. By the regularity theory for minimal surfaces (or mean curvature flow profiles), this submanifold is complete and has controlled geometry.

**Case C.E (Singular Profile):** If $\text{sing}(T_\infty) \neq \emptyset$, the profile is genuinely singular. The tangent cone at each $x_0 \in \text{sing}(T_\infty)$ belongs to a finite germ set $\mathcal{G}$ by energy bounds.

**Profile Library Membership:** The singular profile $T_\infty$ is characterized by:
1. Its tangent cones at singular points
2. Its behavior at infinity (asymptotic cone)
3. Its mass/energy $\mathbf{M}(T_\infty)$

This data determines a unique element of the **canonical profile library** $\mathcal{L}$.

### Step 6: Exhaustiveness of Trichotomy

**Mutual Exclusion:** The three cases are mutually exclusive:
- D.D: $T_\infty = 0$ (vacuously regular with empty support)
- Reg: $T_\infty \neq 0$, $\text{sing}(T_\infty) = \emptyset$
- C.E: $T_\infty \neq 0$, $\text{sing}(T_\infty) \neq \emptyset$

**Exhaustiveness:** By Steps 1-4, every bounded-mass sequence either disperses (D.D) or concentrates with a non-trivial limit $T_\infty$. The limit is either regular (Reg) or singular (C.E).

**No Fourth Case:** The possibility of "weak limit exists but is zero despite concentration" is excluded by the normalization $\|\tilde{T}_j\|(B_1(0)) = \delta/2$, which guarantees $\|T_\infty\|(B_1(0)) \geq \delta/2 > 0$ by lower semicontinuity.

## Key GMT Inequalities Used

1. **Lions Concentration-Compactness Dichotomy:**
   $$Q(R) \to 0 \text{ (vanishing)} \quad \text{or} \quad Q(R_0) \geq \delta \text{ (concentration)}$$

2. **Federer-Fleming Compactness:**
   $$\mathbf{M}(T_j) + \mathbf{M}(\partial T_j) \leq \Lambda \implies \exists T_\infty: T_j \to T_\infty$$

3. **Mass Lower Semicontinuity:**
   $$\|T_\infty\|(U) \leq \liminf_{j \to \infty} \|T_j\|(U)$$

4. **Allard's Regularity Threshold:**
   $$\Theta^k(T, x) < 1 + \varepsilon_0 \text{ and } \int |H|^2 \leq \varepsilon_0 \implies x \in \text{reg}(T)$$

5. **Federer's Dimension Bound:**
   $$\dim_{\mathcal{H}}(\text{sing}(T)) \leq k - 2$$

## Literature References

- Lions, P.-L. (1984). The concentration-compactness principle in the calculus of variations. *Ann. Inst. H. Poincaré Anal. Non Linéaire*, 1, 109-145, 223-283.
- Bahouri, H., Gérard, P. (1999). High frequency approximation of solutions to critical nonlinear wave equations. *Amer. J. Math.*, 121(1), 131-175.
- Federer, H. (1969). *Geometric Measure Theory*. Springer.
- Allard, W. K. (1972). On the first variation of a varifold. *Annals of Mathematics*, 95, 417-491.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
- Kenig, C., Merle, F. (2006). Global well-posedness, scattering and blow-up for the energy-critical focusing non-linear wave equation. *Acta Math.*, 201(2), 147-212.
