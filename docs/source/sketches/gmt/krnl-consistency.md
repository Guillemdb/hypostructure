# KRNL-Consistency: The Fixed-Point Principle — GMT Translation

## Original Statement (Hypostructure)

In the categorical framework, a structural flow datum $\mathcal{S}$ with strict dissipation satisfies: the hypostructure axioms hold on all finite-energy trajectories if and only if every finite-energy trajectory is asymptotically self-consistent, if and only if the only persistent states are fixed points of the evolution operator.

## GMT Setting

**Ambient Space:** $(X, d, \mu)$ — complete separable metric space with $\sigma$-finite Radon measure $\mu$

**Function Class:** $\Phi: X \to [0, \infty]$ — proper lower semicontinuous functional (height/energy)

**Gradient Flow:** $(S_t)_{t \geq 0}$ — metric gradient flow of $\Phi$ in the sense of curves of maximal slope

**Dissipation:** $|\partial \Phi|: X \to [0, \infty]$ — metric slope (local Lipschitz constant of $\Phi$)

## GMT Statement

**Theorem (Fixed-Point Principle).** Let $(X, d)$ be a complete metric space and $\Phi: X \to [0, \infty]$ a proper lower semicontinuous functional. Let $(S_t)_{t \geq 0}$ be the gradient flow of $\Phi$ satisfying the Energy-Dissipation Inequality:

$$\Phi(S_t x) + \int_0^t |\partial \Phi|^2(S_s x) \, ds \leq \Phi(x) \quad \forall x \in D(\Phi), \, t > 0$$

Assume **strict dissipation**: $|\partial \Phi|(x) > 0$ for all $x \notin \text{Crit}(\Phi)$. Then the following are equivalent:

1. **(Regularity)** For all $x$ with $\Phi(x) < \infty$, the trajectory $t \mapsto S_t x$ exists for all $t \geq 0$ and satisfies the gradient flow equation in the metric sense.

2. **(Asymptotic Self-Consistency)** For all $x$ with $\Phi(x) < \infty$, there exists $x_\infty \in X$ such that $d(S_t x, x_\infty) \to 0$ as $t \to \infty$.

3. **(Fixed-Point Characterization)** The $\omega$-limit set of any finite-energy trajectory is contained in $\text{Crit}(\Phi) = \{y : |\partial \Phi|(y) = 0\}$.

## Proof Sketch

### Step 1: Energy Monotonicity and Compactness Preparation

By the Energy-Dissipation Inequality, $t \mapsto \Phi(S_t x)$ is non-increasing. Define the **dissipation measure**:
$$\mathfrak{D}_x := \int_0^\infty |\partial \Phi|^2(S_s x) \, ds \leq \Phi(x) - \liminf_{t \to \infty} \Phi(S_t x) < \infty$$

This finite total dissipation is the GMT analogue of the categorical certificate $K_{D_E}^+$.

### Step 2: Rectifiability of Gradient Flow Curves

The trajectory $\gamma_x: [0, \infty) \to X$ defined by $\gamma_x(t) = S_t x$ is a **curve of maximal slope** in the sense of Ambrosio-Gigli-Savaré. By the metric Brenier-Benamou formula:

$$\text{Length}(\gamma_x|_{[0,T]}) = \int_0^T |\dot{\gamma}_x|(t) \, dt = \int_0^T |\partial \Phi|(\gamma_x(t)) \, dt$$

The equality $|\dot{\gamma}| = |\partial \Phi| \circ \gamma$ holds $\mathcal{L}^1$-a.e. by the definition of curves of maximal slope.

**Rectifiability Consequence:** The image $\gamma_x([0, \infty))$ is a rectifiable curve in $(X, d)$ with:
$$\mathcal{H}^1(\gamma_x([0, \infty))) \leq \int_0^\infty |\partial \Phi|(S_s x) \, ds \leq \sqrt{T} \cdot \sqrt{\mathfrak{D}_x}$$

by Cauchy-Schwarz. For $T \to \infty$, the curve has finite length if and only if $\mathfrak{D}_x < \infty$.

### Step 3: Concentration and Tangent Measures

Consider a sequence $t_n \to \infty$. Define the **occupation measures**:
$$\mu_n := \frac{1}{t_n} \int_0^{t_n} \delta_{S_s x} \, ds \in \mathcal{P}(X)$$

By the finite dissipation condition, the measures $\mu_n$ are tight (supported on sublevel sets of $\Phi$). By Prokhorov's theorem, extract a subsequence $\mu_{n_k} \rightharpoonup \mu_\infty$ weakly.

**Claim:** $\text{supp}(\mu_\infty) \subseteq \text{Crit}(\Phi)$.

*Proof of Claim:* Suppose $y \in \text{supp}(\mu_\infty)$ with $|\partial \Phi|(y) > 0$. By lower semicontinuity of $|\partial \Phi|$, there exists $\varepsilon > 0$ and $r > 0$ such that $|\partial \Phi| \geq \varepsilon$ on $B_r(y)$. Then:
$$\liminf_{n \to \infty} \mu_n(B_r(y)) > 0 \implies \int_0^\infty |\partial \Phi|^2(S_s x) \, ds \geq \varepsilon^2 \cdot \liminf_n t_n \cdot \mu_n(B_r(y)) = \infty$$

contradicting $\mathfrak{D}_x < \infty$. Hence $|\partial \Phi|(y) = 0$.

### Step 4: Convergence to Critical Points

We now prove $(1) \Rightarrow (2)$. The $\omega$-limit set is defined as:
$$\omega(x) := \bigcap_{T > 0} \overline{\{S_t x : t \geq T\}}$$

By Step 3, $\omega(x) \subseteq \text{Crit}(\Phi)$. We need to show $\omega(x)$ is a singleton.

**Łojasiewicz-Simon Argument in Metric Spaces:** Assume the **metric Łojasiewicz inequality**: there exist $\theta \in (0, 1]$, $C > 0$, and $\rho > 0$ such that for all $y \in B_\rho(x_*)$ where $x_* \in \text{Crit}(\Phi)$:
$$|\partial \Phi|(y) \geq C \cdot |\Phi(y) - \Phi(x_*)|^{1-\theta}$$

Define the **Łojasiewicz length functional**:
$$\mathcal{L}(y) := |\Phi(y) - \Phi(x_*)|^\theta$$

Then for $y = S_t x$ near $x_*$:
$$\frac{d}{dt} \mathcal{L}(S_t x) = \theta |\Phi(S_t x) - \Phi(x_*)|^{\theta - 1} \cdot \frac{d}{dt}\Phi(S_t x) \leq -\theta C^{-1} |\partial \Phi|(S_t x)^{1/(1-\theta)} \cdot |\partial \Phi|(S_t x)$$

This gives finite arc-length convergence: $\int_T^\infty d(S_t x, S_{t+h} x)/h \, dt < \infty$.

### Step 5: Fixed-Point Characterization $(2) \Leftrightarrow (3)$

$(2) \Rightarrow (3)$: If $S_t x \to x_\infty$, then $x_\infty \in \omega(x)$. By continuity of the flow, $S_s x_\infty = x_\infty$ for all $s \geq 0$. Differentiating at $s = 0$ (in the metric sense): $|\partial \Phi|(x_\infty) = 0$.

$(3) \Rightarrow (2)$: If $\omega(x) \subseteq \text{Crit}(\Phi)$ and $\omega(x)$ is non-empty (guaranteed by compactness of sublevel sets), connectivity of $\omega(x)$ plus strict dissipation implies $\omega(x)$ is a singleton.

### Step 6: Global Regularity $(3) \Rightarrow (1)$

Suppose the flow breaks down at finite time $T_* < \infty$ for some initial condition $x_0$. Define the **concentration current**:
$$T := \int_0^{T_*} |\partial \Phi|(S_s x_0) \cdot \llbracket S_s x_0 \rrbracket \, ds$$

where $\llbracket y \rrbracket$ denotes the 0-current (Dirac mass) at $y$. The mass $\mathbf{M}(T) = \int_0^{T_*} |\partial \Phi|(S_s x_0) \, ds$ must be infinite for breakdown to occur (otherwise the curve has finite length and admits a limit).

By the fixed-point hypothesis (3), infinite mass is impossible for finite-energy data. Hence $T_* = \infty$.

## Key GMT Inequalities Used

1. **Energy-Dissipation Identity** (Ambrosio-Gigli-Savaré):
   $$\Phi(S_t x) + \frac{1}{2}\int_0^t |\dot{\gamma}|^2 \, ds + \frac{1}{2}\int_0^t |\partial \Phi|^2(\gamma(s)) \, ds = \Phi(x)$$

2. **Metric Łojasiewicz Inequality**:
   $$|\partial \Phi|(y) \geq C |\Phi(y) - \Phi_{\min}|^{1-\theta}$$

3. **Cauchy-Schwarz for Arc-Length**:
   $$\int_0^T |\partial \Phi| \, dt \leq \sqrt{T} \cdot \left(\int_0^T |\partial \Phi|^2 \, dt\right)^{1/2}$$

## Literature References

- Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures*. Birkhäuser. [Chapters 2-4]
- Simon, L. (1983). Asymptotics for a class of non-linear evolution equations. *Annals of Mathematics*, 118, 525-571.
- Kurdyka, K. (1998). On gradients of functions definable in o-minimal structures. *Annales de l'Institut Fourier*, 48(3), 769-783.
- Bolte, J., Daniilidis, A., Lewis, A. (2007). The Łojasiewicz inequality for nonsmooth subanalytic functions. *Mathematische Annalen*, 337(4), 933-950.
