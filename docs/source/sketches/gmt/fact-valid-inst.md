# FACT-ValidInst: Valid Instantiation — GMT Translation

## Original Statement (Hypostructure)

A valid instantiation maps abstract permits to concrete geometric/analytic data satisfying all required bounds. The instantiation witnesses the realizability of the hypostructure.

## GMT Setting

**Abstract Permit:** $\Pi = (K_{D_E}, K_{C_\mu}, K_{\text{SC}_\lambda}, K_{\text{LS}_\sigma})$ — soft certificates

**Instantiation:** $\mathcal{I}: \Pi \to (\mathbf{I}_k(M), \Phi, \mathfrak{D}, \mathcal{L})$ — concrete data

**Valid:** $\mathcal{I}(\Pi)$ satisfies all bounds specified by $\Pi$

## GMT Statement

**Theorem (Valid Instantiation).** An instantiation $\mathcal{I}$ is **valid** if and only if:

1. **Energy Dissipation:** $K_{D_E}^+ \Rightarrow$ the current $T = \mathcal{I}(K_{D_E})$ satisfies:
$$\frac{d}{dt}\Phi(T_t) \leq -\mathfrak{D}(T_t) \leq 0$$

2. **Compactness:** $K_{C_\mu}^+ \Rightarrow$ the sequence $\{T_j\}$ has convergent subsequence in flat norm

3. **Scale Coherence:** $K_{\text{SC}_\lambda}^+ \Rightarrow$ blow-up limits are homogeneous:
$$(\eta_{0,\lambda})_\# T_\infty = T_\infty \quad \forall \lambda > 0$$

4. **Stiffness:** $K_{\text{LS}_\sigma}^+ \Rightarrow$ Łojasiewicz-Simon inequality holds:
$$|\nabla \Phi|(T) \geq c |\Phi(T) - \Phi_*|^{1-\theta}$$

## Proof Sketch

### Step 1: Dissipation Instantiation

**Energy Functional:** Define $\Phi: \mathbf{I}_k(M) \to \mathbb{R}$ as:
- **Mass:** $\Phi(T) = \mathbf{M}(T)$
- **Dirichlet:** $\Phi(T) = \int_M |d u_T|^2$ where $u_T$ is associated function
- **Willmore:** $\Phi(T) = \int_{\text{spt}(T)} |H|^2 d\mathcal{H}^k$ for surfaces

**Dissipation (Ambrosio-Gigli-Savaré, 2008):** The metric derivative:
$$|\partial \Phi|(T) := \limsup_{S \to T} \frac{[\Phi(T) - \Phi(S)]^+}{d(T, S)}$$

**Reference:** Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows in Metric Spaces*. Birkhäuser.

**Valid Instantiation:** $K_{D_E}^+$ requires:
$$\mathfrak{D}(T) := |\partial \Phi|^2(T) \text{ and } \frac{d}{dt}\Phi(T_t) = -\mathfrak{D}(T_t)$$

### Step 2: Compactness Instantiation

**Federer-Fleming (1960):** For $\{T_j\} \subset \mathbf{I}_k(M)$ with:
$$\sup_j (\mathbf{M}(T_j) + \mathbf{M}(\partial T_j)) \leq \Lambda$$

there exists subsequence $T_{j_l} \to T_\infty$ in flat norm.

**Reference:** Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72, 458-520.

**Valid Instantiation:** $K_{C_\mu}^+$ requires:
- Mass bound: $\mathbf{M}(T_j) \leq \Lambda$
- Boundary mass bound: $\mathbf{M}(\partial T_j) \leq \Lambda'$
- Tightness: $\text{spt}(T_j) \subset K$ for compact $K$

### Step 3: Scale Coherence Instantiation

**Tangent Cones (Simon, 1983):** At $x \in \text{sing}(T)$, define blow-up:
$$T_{x,\lambda} := (\eta_{x,\lambda})_\# T$$

where $\eta_{x,\lambda}(y) = (y - x)/\lambda$.

**Reference:** Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.

**Cone Property:** A tangent cone $C = \lim_{\lambda_j \to 0} T_{x,\lambda_j}$ satisfies:
$$(\eta_{0,\mu})_\# C = C \quad \forall \mu > 0$$

**Valid Instantiation:** $K_{\text{SC}_\lambda}^+$ requires that blow-up limits are cones (scale-invariant).

### Step 4: Stiffness Instantiation

**Łojasiewicz-Simon (1983):** For analytic energy $\Phi$ near critical point $T_*$:
$$|\nabla \Phi|(T) \geq c |\Phi(T) - \Phi(T_*)|^{1-\theta}$$

for some $\theta \in (0, 1/2]$.

**Reference:** Simon, L. (1983). Asymptotics for a class of nonlinear evolution equations. *Ann. of Math.*, 118, 525-571.

**Exponent Range (Haraux-Jendoubi, 2015):**
- Analytic case: $\theta \in (0, 1/2]$
- $C^\infty$ case: may fail (counterexamples exist)
- Algebraic case: $\theta$ computable from degree

**Reference:** Haraux, A., Jendoubi, M. A. (2015). *The Convergence Problem for Dissipative Autonomous Systems*. Springer.

**Valid Instantiation:** $K_{\text{LS}_\sigma}^+$ requires:
- Energy is analytic (or definable in o-minimal structure)
- Exponent $\theta$ is bounded: $\theta \geq \theta_0 > 0$
- Constant $c > 0$ is explicit

### Step 5: Consistency Conditions

**Inter-Certificate Compatibility:** Valid instantiation requires:

1. **$K_{D_E}^+ \land K_{C_\mu}^+$:** Dissipation implies tightness along flow
2. **$K_{C_\mu}^+ \land K_{\text{SC}_\lambda}^+$:** Compactness at all scales
3. **$K_{\text{SC}_\lambda}^+ \land K_{\text{LS}_\sigma}^+$:** Homogeneous limits with Łojasiewicz

**Proposition:** If $\mathcal{I}$ satisfies each certificate individually, it satisfies them jointly.

*Proof:* The certificates are independent conditions that can be verified separately.

### Step 6: Witness Construction

**Existence of Valid Instantiation:** Given abstract permits $\Pi$ arising from a well-posed geometric problem, there exists valid instantiation.

**Examples:**

1. **Minimal Surfaces:**
   - $K_{D_E}^+$: Area functional $\Phi(T) = \mathbf{M}(T)$
   - $K_{C_\mu}^+$: Federer-Fleming compactness
   - $K_{\text{SC}_\lambda}^+$: Tangent cones at singularities
   - $K_{\text{LS}_\sigma}^+$: Simon's Łojasiewicz (1983)

2. **Mean Curvature Flow:**
   - $K_{D_E}^+$: $\partial_t \mathbf{M} = -\int |H|^2$
   - $K_{C_\mu}^+$: Brakke compactness (1978)
   - $K_{\text{SC}_\lambda}^+$: Huisken monotonicity (1990)
   - $K_{\text{LS}_\sigma}^+$: Schulze's Łojasiewicz for MCF (2014)

**Reference:**
- Brakke, K. (1978). *The Motion of a Surface by Its Mean Curvature*. Princeton.
- Huisken, G. (1990). Asymptotic behavior for singularities of MCF. *J. Diff. Geom.*, 31.
- Schulze, F. (2014). Uniqueness of compact tangent flows in MCF. *J. reine angew. Math.*, 690.

### Step 7: Uniqueness of Instantiation

**Proposition:** Valid instantiation is unique up to:
1. Isometry of ambient space
2. Rescaling of energy
3. Gauge equivalence

*Proof:* The certificates determine:
- Energy up to additive/multiplicative constant
- Compactness up to choice of topology
- Blow-up limits uniquely (by tangent cone uniqueness theorems)

**Tangent Cone Uniqueness (Simon, 1993):** At isolated singularities, tangent cones are unique.

**Reference:** Simon, L. (1993). Cylindrical tangent cones and the singular set of minimal submanifolds. *J. Diff. Geom.*, 38, 585-652.

## Key GMT Inequalities Used

1. **Energy-Dissipation:**
   $$\frac{d}{dt}\Phi(T_t) = -|\partial\Phi|^2(T_t)$$

2. **Federer-Fleming Compactness:**
   $$\sup_j \mathbf{M}(T_j) < \infty \implies T_{j_k} \to T_\infty$$

3. **Cone Homogeneity:**
   $$(\eta_{0,\lambda})_\# C = C$$

4. **Łojasiewicz-Simon:**
   $$|\nabla\Phi|(T) \geq c|\Phi(T) - \Phi_*|^{1-\theta}$$

## Literature References

- Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72.
- Simon, L. (1983). Asymptotics for nonlinear evolution equations. *Ann. of Math.*, 118.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
- Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows*. Birkhäuser.
- Brakke, K. (1978). *The Motion of a Surface by Its Mean Curvature*. Princeton.
- Haraux, A., Jendoubi, M. A. (2015). *The Convergence Problem*. Springer.
