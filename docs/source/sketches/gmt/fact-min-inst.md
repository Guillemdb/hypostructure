# FACT-MinInst: Minimal Instantiation — GMT Translation

## Original Statement (Hypostructure)

Among all valid instantiations, there exists a minimal one that uses the weakest assumptions sufficient to derive the required conclusions.

## GMT Setting

**Instantiation Space:** $\mathcal{I}(\Pi)$ — all valid instantiations of permits $\Pi$

**Ordering:** $\mathcal{I}_1 \leq \mathcal{I}_2$ if $\mathcal{I}_1$ uses weaker assumptions

**Minimal:** $\mathcal{I}_{\min}$ — instantiation with no strictly weaker valid instantiation

## GMT Statement

**Theorem (Minimal Instantiation).** For any permit system $\Pi$, there exists a minimal valid instantiation $\mathcal{I}_{\min}$ characterized by:

1. **Minimal Energy Class:** $\Phi \in \text{Lip}(X) \cap \text{BV}(X)$ (not necessarily smooth)

2. **Minimal Compactness:** Exactly the tightness required by energy bounds

3. **Minimal Scale Structure:** Only scales appearing in blow-up sequences

4. **Minimal Stiffness:** Łojasiewicz exponent $\theta = \theta_{\max}$ (largest admissible)

## Proof Sketch

### Step 1: Ordering on Instantiations

**Definition:** For instantiations $\mathcal{I}_1, \mathcal{I}_2$ of the same permits $\Pi$:
$$\mathcal{I}_1 \leq \mathcal{I}_2 \iff \text{every bound in } \mathcal{I}_1 \text{ is implied by bounds in } \mathcal{I}_2$$

**Concrete Orderings:**
- Energy: $\Phi_1 \leq \Phi_2$ if $\Phi_1(T) \leq \Phi_2(T)$ for all $T$
- Compactness: $K_1 \leq K_2$ if $K_1 \subset K_2$ (larger compact set)
- Stiffness: $\theta_1 \leq \theta_2$ if $\theta_1 \geq \theta_2$ (larger exponent = weaker condition)

**Poset Structure:** $(\mathcal{I}(\Pi), \leq)$ is a partially ordered set.

### Step 2: Existence via Zorn's Lemma

**Chain Completeness:** Let $\{\mathcal{I}_\alpha\}_{\alpha \in A}$ be a chain. Define:
$$\mathcal{I}_{\inf}(T) := \inf_\alpha \mathcal{I}_\alpha(T)$$

**Lemma:** $\mathcal{I}_{\inf}$ is a valid instantiation.

*Proof:* Each certificate is preserved under infima:
- $K_{D_E}^+$: Dissipation inequality passes to limits
- $K_{C_\mu}^+$: Intersection of compact sets is compact
- $K_{\text{SC}_\lambda}^+$: Scale coherence is preserved
- $K_{\text{LS}_\sigma}^+$: Łojasiewicz with $\theta_{\sup} = \sup_\alpha \theta_\alpha$

**Zorn's Lemma:** Every chain has a lower bound, so minimal elements exist.

**Reference:** Kelley, J. L. (1955). *General Topology*. Van Nostrand.

### Step 3: Minimal Energy Functional

**BV Sufficiency (Ambrosio-Fusco-Pallara, 2000):** For gradient flows, the energy need only be in $\text{BV}$:
$$\Phi \in \text{BV}(X, d, \mu) \iff \text{total variation } |D\Phi|(X) < \infty$$

**Reference:** Ambrosio, L., Fusco, N., Pallara, D. (2000). *Functions of Bounded Variation and Free Discontinuity Problems*. Oxford.

**Minimal Energy:** The minimal instantiation uses:
$$\Phi_{\min}(T) := \inf\{\Phi(T) : \Phi \text{ satisfies } K_{D_E}^+\}$$

**Characterization:** $\Phi_{\min}$ is the largest function satisfying the dissipation bound.

### Step 4: Minimal Compactness

**Prokhorov's Theorem (1956):** A family $\{\mu_j\}$ of measures is tight iff precompact in weak topology.

**Reference:** Prokhorov, Y. V. (1956). Convergence of random processes and limit theorems in probability theory. *Theory Probab. Appl.*, 1, 157-214.

**Minimal Tightness:** The minimal compact set containing supports:
$$K_{\min} := \overline{\bigcup_j \text{spt}(\mu_j)}^{\text{ess}}$$

where essential closure removes null sets.

**Minimal Mass Bound:**
$$\Lambda_{\min} := \liminf_{j \to \infty} \mathbf{M}(T_j)$$

### Step 5: Minimal Scale Structure

**Essential Scales:** The set of scales appearing in blow-up:
$$\mathcal{S}_{\min} := \{\lambda > 0 : \exists x, \, T_{x,\lambda} \not\approx T_{x,1}\}$$

**Discrete vs Continuous:** By monotonicity formulas (Almgren, 1979):
- If $\mathcal{S}_{\min}$ is discrete: isolated singularities
- If $\mathcal{S}_{\min}$ is continuous: continuous family of singularities

**Reference:** Almgren, F. J. (1979). Dirichlet's problem for multiple valued functions. *Arch. Rational Mech. Anal.*, 72, 275-369.

**Minimal Scale Coherence:** Use only scales in $\mathcal{S}_{\min}$.

### Step 6: Optimal Łojasiewicz Exponent

**Definition:** The optimal Łojasiewicz exponent is:
$$\theta_{\text{opt}} := \sup\{\theta : \text{LS inequality holds with exponent } \theta\}$$

**Upper Bound (Łojasiewicz, 1965):** For real-analytic $\Phi$:
$$\theta_{\text{opt}} \leq 1/2$$

**Lower Bound (Simon, 1983):** For area-minimizing currents:
$$\theta_{\text{opt}} \geq 1/(2n+2)$$

**Reference:**
- Łojasiewicz, S. (1965). Ensembles semi-analytiques. IHES preprint.
- Simon, L. (1983). Asymptotics for nonlinear evolution equations. *Ann. of Math.*, 118.

**Minimal Stiffness:** Use $\theta = \theta_{\text{opt}}$ (largest exponent, weakest assumption).

### Step 7: Uniqueness of Minimal Instantiation

**Theorem:** The minimal instantiation is unique up to equivalence.

*Proof:* Suppose $\mathcal{I}_1, \mathcal{I}_2$ are both minimal. By minimality:
- $\mathcal{I}_1 \leq \mathcal{I}_2$ implies $\mathcal{I}_1 = \mathcal{I}_2$ (else $\mathcal{I}_2$ not minimal)
- $\mathcal{I}_2 \leq \mathcal{I}_1$ implies $\mathcal{I}_2 = \mathcal{I}_1$ (else $\mathcal{I}_1$ not minimal)

If incomparable, define $\mathcal{I}_3 := \mathcal{I}_1 \wedge \mathcal{I}_2$ (meet). If $\mathcal{I}_3$ is valid, contradicts minimality. If not, proves necessity of both.

### Step 8: Constructive Characterization

**Algorithm for Minimal Instantiation:**

1. **Energy:** Start with $\Phi = \mathbf{M}$ (mass). Verify $K_{D_E}^+$. If fails, enlarge.

2. **Compactness:** Start with $K = \text{spt}(T_0)$. Enlarge until $K_{C_\mu}^+$ holds.

3. **Scales:** Compute blow-up at each singular point. Record necessary scales.

4. **Stiffness:** Compute $\theta_{\text{opt}}$ via:
   $$\theta_{\text{opt}} = \lim_{T \to T_*} \frac{\log |\nabla \Phi|(T)}{\log |\Phi(T) - \Phi_*|}$$

**Output:** $\mathcal{I}_{\min} = (\Phi_{\min}, K_{\min}, \mathcal{S}_{\min}, \theta_{\text{opt}})$

### Step 9: Minimal Instantiation for Specific Problems

**Example 1: Area-Minimizing Currents**
- $\Phi_{\min} = \mathbf{M}$ (mass)
- $K_{\min} = \overline{\text{spt}(T)}$
- $\theta_{\text{opt}} = 1/2$ (analytic boundary)

**Example 2: Mean Curvature Flow**
- $\Phi_{\min} = \mathbf{M}$ with $\partial_t \mathbf{M} = -\int |H|^2$
- $K_{\min}$ determined by parabolic maximum principle
- $\theta_{\text{opt}}$ given by Schulze (2014)

**Reference:** Schulze, F. (2014). Uniqueness of compact tangent flows. *J. reine angew. Math.*, 690.

## Key GMT Inequalities Used

1. **BV Energy:**
   $$|D\Phi|(X) < \infty$$

2. **Prokhorov Tightness:**
   $$\forall \varepsilon > 0, \exists K_\varepsilon : \mu_j(X \setminus K_\varepsilon) < \varepsilon$$

3. **Optimal LS Exponent:**
   $$|\nabla\Phi| \geq c|\Phi - \Phi_*|^{1-\theta_{\text{opt}}}$$

4. **Monotonicity:**
   $$r \mapsto \frac{\mathbf{M}(T \cap B_r)}{r^k} \text{ is monotone}$$

## Literature References

- Łojasiewicz, S. (1965). Ensembles semi-analytiques. IHES.
- Simon, L. (1983). Asymptotics for nonlinear evolution equations. *Ann. of Math.*, 118.
- Prokhorov, Y. V. (1956). Convergence of random processes. *Theory Probab. Appl.*, 1.
- Ambrosio, L., Fusco, N., Pallara, D. (2000). *Functions of Bounded Variation*. Oxford.
- Almgren, F. J. (1979). Dirichlet's problem for multiple valued functions. *Arch. Rational Mech. Anal.*, 72.
- Schulze, F. (2014). Uniqueness of compact tangent flows. *J. reine angew. Math.*, 690.
