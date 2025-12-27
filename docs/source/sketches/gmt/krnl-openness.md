# KRNL-Openness: Openness of Regularity — GMT Translation

## Original Statement (Hypostructure)

The set of globally regular hypostructures is open in the parameter topology. If a system is regular with strict gaps, nearby systems are also regular.

## GMT Setting

**Family of Currents:** $\{T_\theta\}_{\theta \in \Theta} \subset \mathbf{I}_k(M)$ — parametrized family of integral currents

**Parameter Space:** $\Theta \subset \mathbb{R}^m$ — open subset

**Regularity:** $T_\theta$ is regular if $\text{sing}(T_\theta) = \emptyset$

**Singular Set:** $\text{sing}(T) = \{x \in \text{spt}(T) : T \text{ is not } C^{1,\alpha} \text{ near } x\}$

**Gap Condition:** Density gap $\Theta^k(T, x) \geq 1 + \varepsilon_0$ at singular points

## GMT Statement

**Theorem (Openness of Regularity for Currents).** Let $\{T_\theta\}_{\theta \in \Theta}$ be a smooth family of integral $k$-currents in a Riemannian manifold $(M^n, g)$ satisfying:

1. **(Uniform Bounds)** $\sup_{\theta \in \Theta} (\mathbf{M}(T_\theta) + \mathbf{M}(\partial T_\theta)) \leq \Lambda$

2. **(Continuity)** $\theta \mapsto T_\theta$ is continuous in the flat norm topology

3. **(Strict Gap at $\theta_0$)** At $\theta_0$, the current $T_{\theta_0}$ satisfies:
   - $\text{sing}(T_{\theta_0}) = \emptyset$ (regular)
   - $\sup_{x \in \text{spt}(T_{\theta_0})} (\Theta^k(T_{\theta_0}, x) - 1) \leq 1 - \delta$ for some $\delta > 0$

Then there exists a neighborhood $U \ni \theta_0$ in $\Theta$ such that $T_\theta$ is regular for all $\theta \in U$.

## Proof Sketch

### Step 1: Allard's Regularity Theorem and $\varepsilon$-Regularity

**Allard's Theorem:** There exist $\varepsilon_A = \varepsilon_A(n, k, \Lambda) > 0$ and $\alpha \in (0, 1)$ such that if $V \in \mathbf{V}_k(M)$ is a varifold with:
1. $\Theta^k(V, x_0) < 1 + \varepsilon_A$ for all $x \in B_r(x_0)$
2. $\|\delta V\|(B_r(x_0)) \leq \varepsilon_A r^{k-1}$

then $V \llcorner B_{r/2}(x_0)$ is represented by a $C^{1,\alpha}$ embedded submanifold.

**Translation to Currents:** For integral currents, the density $\Theta^k(T, x)$ is always $\geq 1$ on $\text{spt}(T)$. The condition $\Theta^k < 1 + \varepsilon_A$ implies regularity.

### Step 2: Density Lower Semicontinuity

**Flat Convergence and Density:** If $T_j \to T$ in flat norm and $\mathbf{M}(T_j)$ is uniformly bounded, then for any $x \in M$:
$$\Theta^k(T, x) \leq \liminf_{j \to \infty} \Theta^k(T_j, x)$$

**Lower Semicontinuity of Singular Set:** The condition $\Theta^k(T, x) \geq 1 + \varepsilon$ is open. Hence:
$$\text{sing}(T) \subset \liminf_{j \to \infty} \text{sing}(T_j)$$

(singular points can only disappear, not appear, in the limit).

### Step 3: Continuity of Density in Parameters

**Density Function:** Define $\rho_\theta(x) := \Theta^k(T_\theta, x)$ for $x \in M$.

**Claim:** If $\theta \mapsto T_\theta$ is continuous in flat norm, then $\theta \mapsto \rho_\theta(x)$ is upper semicontinuous for each $x$.

*Proof:* For any $\varepsilon > 0$, there exists $r_0 > 0$ such that:
$$\left| \frac{\|T_\theta\|(B_r(x))}{\omega_k r^k} - \rho_\theta(x) \right| < \varepsilon \quad \text{for } r < r_0$$

By flat continuity, $\theta \mapsto \|T_\theta\|(B_r(x))$ is continuous for fixed $r$. Hence $\rho_\theta(x)$ varies continuously from above.

### Step 4: Gap Preservation

**At $\theta_0$:** By assumption, $\sup_x \rho_{\theta_0}(x) \leq 1 + (1 - \delta)$ for some $\delta > 0$.

**Near $\theta_0$:** By upper semicontinuity of $\rho_\theta$ in $\theta$, for $\theta$ near $\theta_0$:
$$\sup_x \rho_\theta(x) \leq \sup_x \rho_{\theta_0}(x) + \frac{\delta}{2} < 1 + \varepsilon_A$$

if $\delta$ is small enough compared to the Allard threshold $\varepsilon_A$.

**Regularity Inheritance:** By Allard's theorem, $T_\theta$ is regular for $\theta$ in a neighborhood of $\theta_0$.

### Step 5: Quantitative Neighborhood Size

**Modulus of Continuity:** Let $\omega(\cdot)$ be the modulus of continuity of $\theta \mapsto T_\theta$ in flat norm:
$$\mathbb{F}(T_\theta - T_{\theta_0}) \leq \omega(|\theta - \theta_0|)$$

**Stability Estimate:** The density perturbation satisfies:
$$|\rho_\theta(x) - \rho_{\theta_0}(x)| \leq C \cdot \omega(|\theta - \theta_0|)^{1/k} \cdot r^{-k}$$

for $r$ the scale at which density is measured.

**Neighborhood Size:** Choose $U = B_{\eta}(\theta_0)$ where $\eta > 0$ satisfies:
$$C \cdot \omega(\eta)^{1/k} \cdot R^{-k} < \frac{\delta}{2}$$

for a fixed scale $R$ depending on the geometry of $\text{spt}(T_{\theta_0})$.

### Step 6: Varifold Formulation

**Varifold Version:** The openness theorem extends to varifolds $V_\theta \in \mathbf{V}_k(M)$ under:
1. $\theta \mapsto V_\theta$ continuous in the varifold topology
2. Uniform bounds on $\|V_\theta\|(M)$ and $\|\delta V_\theta\|(M)$
3. Strict density gap at $\theta_0$

**First Variation Bound:** The first variation satisfies:
$$\delta V_\theta(X) = -\int \text{div}_V X \, d\|V_\theta\|$$

Continuity in $\theta$ implies the first variation bound propagates to nearby parameters.

**Regularity Conclusion:** By the varifold regularity theorem (Allard), the regularity set:
$$\mathcal{R} := \{\theta \in \Theta : \text{sing}(V_\theta) = \emptyset\}$$

is open.

## Key GMT Inequalities Used

1. **Allard's $\varepsilon$-Regularity:**
   $$\Theta^k(V, x) < 1 + \varepsilon_A, \, \|\delta V\| \leq \varepsilon_A r^{k-1} \implies \text{reg}(V) \cap B_{r/2}(x) = \text{spt}(V) \cap B_{r/2}(x)$$

2. **Density Lower Semicontinuity:**
   $$T_j \xrightarrow{\mathbb{F}} T \implies \Theta^k(T, x) \leq \liminf_j \Theta^k(T_j, x)$$

3. **Monotonicity Formula:**
   $$\frac{\|V\|(B_r(x))}{r^k} \leq e^{Cr} \frac{\|V\|(B_s(x))}{s^k}$$

4. **Flat-Mass Comparison:**
   $$\mathbb{F}(T_1 - T_2) \leq C(\mathbf{M}(T_1) + \mathbf{M}(T_2))$$

## Literature References

- Allard, W. K. (1972). On the first variation of a varifold. *Annals of Mathematics*, 95, 417-491.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
- White, B. (2005). A local regularity theorem for mean curvature flow. *Annals of Mathematics*, 161, 1487-1519.
- Smale, S. (1967). Differentiable dynamical systems. *Bull. AMS*, 73, 747-817.
- Palis, J., de Melo, W. (1982). *Geometric Theory of Dynamical Systems*. Springer.
- Robinson, C. (1999). *Dynamical Systems: Stability, Symbolic Dynamics, and Chaos*. 2nd ed. CRC Press.
