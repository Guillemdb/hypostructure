# KRNL-MetricAction: Extended Action Reconstruction — GMT Translation

## Original Statement (Hypostructure)

The reconstruction theorems extend from Riemannian manifolds to general metric spaces via the metric slope and the Ambrosio-Gigli-Savaré theory of gradient flows.

## GMT Setting

**Ambient Space:** $(X, d)$ — complete separable metric space

**Energy Functional:** $\Phi: X \to (-\infty, +\infty]$ — proper lower semicontinuous

**Metric Slope:** $|\partial \Phi|(x) := \limsup_{y \to x} \frac{[\Phi(x) - \Phi(y)]^+}{d(x, y)}$ — local descent rate

**Metric Derivative:** For a curve $\gamma: (a, b) \to X$: $|\dot{\gamma}|(t) := \lim_{h \to 0} \frac{d(\gamma(t+h), \gamma(t))}{|h|}$

**Critical Set:** $\mathcal{M} := \{x \in X : |\partial \Phi|(x) = 0\}$

## GMT Statement

**Theorem (Metric Action Reconstruction).** Let $(X, d)$ be a complete geodesic metric space and $\Phi: X \to [0, \infty]$ proper l.s.c. satisfying:

1. **(EDI)** The gradient flow $(S_t)$ satisfies the Energy-Dissipation Inequality:
   $$\Phi(S_t x) + \int_0^t |\partial \Phi|^2(S_s x) \, ds \leq \Phi(x)$$

2. **(Geodesic Convexity)** $\Phi$ is $\lambda$-geodesically convex for some $\lambda \in \mathbb{R}$:
   $$\Phi(\gamma_s) \leq (1-s)\Phi(\gamma_0) + s\Phi(\gamma_1) - \frac{\lambda}{2}s(1-s)d(\gamma_0, \gamma_1)^2$$

3. **(Stiffness)** Metric Łojasiewicz inequality: $|\partial \Phi|(x) \geq C|\Phi(x) - \Phi_{\min}|^{1-\theta}$

Then the Lyapunov functional satisfies:
$$\mathcal{L}(x) = \Phi_{\min} + \inf_{\gamma: \mathcal{M} \to x} \int_0^1 |\partial \Phi|(\gamma(s)) \cdot |\dot{\gamma}|(s) \, ds$$

## Proof Sketch

### Step 1: Absolutely Continuous Curves and Metric Derivative

**Definition:** A curve $\gamma: [0, T] \to X$ is **absolutely continuous** if there exists $m \in L^1([0, T])$ such that:
$$d(\gamma(s), \gamma(t)) \leq \int_s^t m(r) \, dr \quad \text{for all } 0 \leq s \leq t \leq T$$

**Metric Derivative:** For a.c. curves, the metric derivative exists a.e.:
$$|\dot{\gamma}|(t) = \lim_{h \to 0} \frac{d(\gamma(t+h), \gamma(t))}{|h|} \quad \text{for a.e. } t$$

and $|\dot{\gamma}| \in L^1([0, T])$ with:
$$d(\gamma(s), \gamma(t)) \leq \int_s^t |\dot{\gamma}|(r) \, dr$$

### Step 2: Curves of Maximal Slope

**Definition:** An a.c. curve $\gamma: [0, T] \to X$ is a **curve of maximal slope** for $\Phi$ if:
$$\frac{d}{dt} \Phi(\gamma(t)) = -|\dot{\gamma}|(t) \cdot |\partial \Phi|(\gamma(t)) = -\frac{1}{2}|\dot{\gamma}|^2(t) - \frac{1}{2}|\partial \Phi|^2(\gamma(t))$$

for a.e. $t \in [0, T]$.

**Equivalently (Young's inequality form):**
$$|\dot{\gamma}|(t) = |\partial \Phi|(\gamma(t)) \quad \text{a.e.}$$

**Theorem (AGS):** If $\Phi$ is $\lambda$-geodesically convex with $\lambda > -\infty$, then for each $x_0 \in \overline{D(\Phi)}$, there exists a unique curve of maximal slope $\gamma$ with $\gamma(0) = x_0$, and this curve is the gradient flow $S_t x_0$.

### Step 3: Energy-Dissipation Equality

For curves of maximal slope, the **Energy-Dissipation Identity** (EDI becomes EDE):
$$\Phi(\gamma(0)) - \Phi(\gamma(T)) = \frac{1}{2}\int_0^T |\dot{\gamma}|^2 \, dt + \frac{1}{2}\int_0^T |\partial \Phi|^2(\gamma) \, dt = \int_0^T |\partial \Phi|^2(\gamma) \, dt$$

The last equality uses $|\dot{\gamma}| = |\partial \Phi|(\gamma)$.

**Consequence:** The dissipation integral equals the energy drop:
$$\int_0^T |\partial \Phi|^2(S_s x) \, ds = \Phi(x) - \Phi(S_T x)$$

### Step 4: Action Functional in Metric Spaces

Define the **action functional** for curves $\gamma: [0, 1] \to X$:
$$\mathcal{A}[\gamma] := \int_0^1 |\partial \Phi|(\gamma(s)) \cdot |\dot{\gamma}|(s) \, ds$$

**Optimal Curves:** The gradient flow (reparametrized to constant speed) minimizes the action among all curves connecting $x$ to $\mathcal{M}$.

*Proof:* For any curve $\tilde{\gamma}: [0, 1] \to X$ from $x = \tilde{\gamma}(0)$ to $y \in \mathcal{M}$:
$$\mathcal{A}[\tilde{\gamma}] = \int_0^1 |\partial \Phi|(\tilde{\gamma}) \cdot |\dot{\tilde{\gamma}}| \, ds \geq \int_0^1 \frac{d}{ds}(-\Phi(\tilde{\gamma}(s))) \, ds = \Phi(x) - \Phi(y)$$

The inequality is the chain rule bound: $-\frac{d}{ds}\Phi(\tilde{\gamma}) \leq |\partial \Phi|(\tilde{\gamma}) \cdot |\dot{\tilde{\gamma}}|$.

Equality holds when the chain rule is saturated, i.e., when $\tilde{\gamma}$ moves in the direction of steepest descent with matching speed — the gradient flow.

### Step 5: Wasserstein Space Application

**Setting:** $X = (\mathcal{P}_2(\mathbb{R}^n), W_2)$ — Wasserstein space of probability measures

**Energy:** $\Phi(\mu) = \int_{\mathbb{R}^n} F(\rho(x)) \, dx$ where $\mu = \rho \, dx$ — internal energy functional

**Metric Slope:** $|\partial \Phi|(\mu) = \|\nabla \frac{\delta \Phi}{\delta \mu}\|_{L^2(\mu)} = \|F'(\rho)\|_{\dot{H}^1(\mu)}$

**Gradient Flow:** The Wasserstein gradient flow of $\Phi$ is the **Fokker-Planck equation**:
$$\partial_t \rho = \nabla \cdot (\rho \nabla F'(\rho))$$

**Action Reconstruction:** The Lyapunov functional is:
$$\mathcal{L}(\mu) = \Phi_{\min} + W_2(\mu, \mu_{\infty})_{\text{weighted}}$$

where the weighted Wasserstein distance uses the metric $|\partial \Phi|$ as a conformal factor.

### Step 6: Discrete Metric Spaces

**Setting:** $X = V$ — vertices of a graph with combinatorial distance

**Energy:** $\Phi: V \to \mathbb{R}$ — function on vertices

**Metric Slope:** For $x \in V$:
$$|\partial \Phi|(x) = \max_{y \sim x} [\Phi(x) - \Phi(y)]^+$$

(maximum over neighbors)

**Gradient Flow:** Discrete gradient flow:
$$x_{n+1} = \arg\min_{y \sim x_n} \Phi(y)$$

**Action on Paths:** For a path $\gamma = (x_0, x_1, \ldots, x_N)$:
$$\mathcal{A}[\gamma] = \sum_{i=0}^{N-1} |\partial \Phi|(x_i) \cdot d(x_i, x_{i+1}) = \sum_{i=0}^{N-1} |\partial \Phi|(x_i)$$

(assuming unit edge lengths)

**Reconstruction:**
$$\mathcal{L}(x) = \Phi_{\min} + \min_{\text{paths } \gamma: \mathcal{M} \to x} \mathcal{A}[\gamma]$$

This equals the **Dijkstra shortest path** in the graph with edge weights $|\partial \Phi|$.

## Key GMT Inequalities Used

1. **Metric Chain Rule:**
   $$\frac{d}{dt}\Phi(\gamma(t)) \geq -|\partial \Phi|(\gamma(t)) \cdot |\dot{\gamma}|(t)$$

2. **AGS Energy-Dissipation Identity:**
   $$\Phi(\gamma_0) - \Phi(\gamma_T) = \int_0^T |\partial \Phi|^2(\gamma(t)) \, dt$$

3. **Geodesic Convexity Implication:**
   $$\Phi \text{ is } \lambda\text{-convex} \implies \text{EVI}_\lambda \implies \text{existence and uniqueness of gradient flow}$$

4. **Benamou-Brenier Formula (Wasserstein):**
   $$W_2(\mu_0, \mu_1)^2 = \inf_{\rho, v} \int_0^1 \int_{\mathbb{R}^n} |v|^2 \rho \, dx \, dt$$

## Literature References

- Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures*. 2nd ed. Birkhäuser.
- Jordan, R., Kinderlehrer, D., Otto, F. (1998). The variational formulation of the Fokker-Planck equation. *SIAM J. Math. Anal.*, 29(1), 1-17.
- Maas, J. (2011). Gradient flows of the entropy for finite Markov chains. *J. Funct. Anal.*, 261(8), 2250-2292.
- Mielke, A. (2016). On evolutionary Γ-convergence for gradient systems. *Macroscopic and Large Scale Phenomena*. Springer.
- Erbar, M., Maas, J. (2012). Ricci curvature of finite Markov chains via convexity of the entropy. *Arch. Ration. Mech. Anal.*, 206(3), 997-1038.
