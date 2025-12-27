---
title: "FACT-Soft-Morse - AI/RL/ML Translation"
---

# FACT-Soft-Morse: Loss Landscape Topology and Mode Connectivity

## Original Statement (Hypostructure)

**[FACT-SoftMorse] Soft->MorseDecomp Compilation.** Morse/gradient-like decomposition is derived from attractor existence + soft interfaces.

**Hypotheses:**
$$K_{\mathrm{Attr}}^+ \wedge K_{D_E}^+ \wedge K_{\mathrm{LS}_\sigma}^+$$

**Produces:**
$$K_{\mathrm{MorseDecomp}}^+$$

**Mechanism:**
1. **Attractor Exists:** From $K_{\mathrm{Attr}}^+$ (compact, invariant, attracting)
2. **Lyapunov Function:** $D_E^+$ certifies $\Phi$ decreases along trajectories (dissipation)
3. **Gradient-like Structure:** If $\Phi$ is strictly decreasing except at equilibria, apply gradient-like backend
4. **Lojasiewicz Prevents Cycles:** $\mathrm{LS}_\sigma^+$ ensures trajectories cannot oscillate indefinitely

**Certificate Emitted:**
$$K_{\mathrm{MorseDecomp}}^+ = (\mathsf{gradient\_like}, \mathcal{E}, \{W^u(\xi)\}, \mathsf{no\_periodic})$$

## AI/RL/ML Statement

**Theorem (Loss Landscape Morse Decomposition).** Let $\mathcal{L}: \Theta \to \mathbb{R}$ be a loss function over parameter space $\Theta \subseteq \mathbb{R}^n$ for a neural network or RL agent. Assume:

1. **Global Attractor (Bounded Optimization):** There exists a compact set $\mathcal{A} \subseteq \Theta$ such that all gradient descent trajectories $\theta(t)$ eventually enter and remain in $\mathcal{A}$:
   $$\lim_{t \to \infty} \mathrm{dist}(\theta(t), \mathcal{A}) = 0$$

2. **Strict Dissipation (Loss Decrease):** The loss strictly decreases along gradient flow except at critical points:
   $$\frac{d\mathcal{L}}{dt} = -\|\nabla \mathcal{L}\|^2 < 0 \quad \text{unless } \nabla \mathcal{L} = 0$$

3. **Lojasiewicz Gradient Inequality:** Near each critical point $\theta^* \in \mathrm{Crit}(\mathcal{L})$, there exist $C > 0$, $\theta \in (0, 1/2]$, and $\delta > 0$ such that:
   $$\|\nabla \mathcal{L}(\theta)\| \geq C \cdot |\mathcal{L}(\theta) - \mathcal{L}(\theta^*)|^{1-\theta}$$

**Then:**

**(A) Gradient-like Structure:** The loss landscape is gradient-like: all gradient descent trajectories flow monotonically from higher to lower loss.

**(B) Critical Point Classification:** The critical point set $\mathrm{Crit}(\mathcal{L}) = \{\theta : \nabla \mathcal{L}(\theta) = 0\}$ is discrete (no accumulation points). Each critical point is classified by its **Morse index** (number of negative Hessian eigenvalues):
- **Index 0:** Local minimum (stable)
- **Index k > 0:** Saddle point with k unstable directions
- **Index n:** Local maximum (fully unstable)

**(C) No Periodic Orbits:** There are no limit cycles in the gradient flow. Every trajectory converges to a single critical point.

**(D) Morse Decomposition:** The attractor decomposes as:
$$\mathcal{A} = \bigcup_{\theta^* \in \mathrm{Crit}(\mathcal{L})} W^u(\theta^*)$$
where $W^u(\theta^*)$ is the unstable manifold (all points whose gradient flow converges to $\theta^*$).

**Certificate:**
$$K_{\mathrm{LossLandscape}}^+ = (\mathsf{gradient\_like}, \mathrm{Crit}(\mathcal{L}), \{\text{Morse indices}\}, \mathsf{no\_cycles})$$

## Terminology Translation Table

| Hypostructure | AI/RL/ML |
|---------------|----------|
| State space $\mathcal{X}$ | Parameter space $\Theta \subseteq \mathbb{R}^n$ |
| Semiflow $S_t$ | Gradient descent flow $\theta(t)$ |
| Height functional $\Phi$ | Loss function $\mathcal{L}(\theta)$ |
| Value function $V(s)$ | Negative loss $-\mathcal{L}$ or value estimate $V_\phi(s)$ |
| Dissipation $\mathfrak{D}$ | Policy gradient $\nabla_\theta J(\pi_\theta)$ |
| Equilibrium set $\mathcal{E}$ | Critical points $\mathrm{Crit}(\mathcal{L}) = \{\theta : \nabla \mathcal{L} = 0\}$ |
| Global attractor $\mathcal{A}$ | Reachable parameter region under training |
| Morse function | Loss function with non-degenerate critical points |
| Morse index | Number of negative Hessian eigenvalues |
| Unstable manifold $W^u(\xi)$ | Basin of convergence to minimum $\theta^*$ |
| Łojasiewicz inequality | Polyak-Łojasiewicz (PL) condition |
| Gradient-like structure | Monotonic loss decrease during training |
| No periodic orbits | No training oscillations / mode cycling |
| Morse decomposition | Loss landscape connected components |
| Connecting orbit | Training trajectory between critical points |
| Certificate $K_{\mathrm{MorseDecomp}}^+$ | Convergence guarantee for optimizer |

## Proof Sketch

### Setup: Loss Landscapes and Gradient Flow

**Definition (Gradient Flow).** The gradient descent dynamics on loss $\mathcal{L}: \Theta \to \mathbb{R}$ are:
$$\frac{d\theta}{dt} = -\nabla \mathcal{L}(\theta)$$

In discrete time with learning rate $\eta$:
$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$$

**Definition (Morse Function).** A smooth function $\mathcal{L}: \Theta \to \mathbb{R}$ is a **Morse function** if all critical points are non-degenerate:
$$\nabla \mathcal{L}(\theta^*) = 0 \implies \det(\nabla^2 \mathcal{L}(\theta^*)) \neq 0$$

The **Morse index** of a critical point $\theta^*$ is the number of negative eigenvalues of the Hessian $\nabla^2 \mathcal{L}(\theta^*)$.

**Critical Point Classification:**
- **Morse index 0:** All eigenvalues positive, local minimum
- **Morse index k:** k negative eigenvalues, saddle point with k unstable directions
- **Morse index n:** All eigenvalues negative, local maximum

### Step 1: Attractor Confinement (Bounded Training)

**Claim:** All gradient descent trajectories are eventually confined to a compact attractor $\mathcal{A}$.

**Proof (Weight Decay Regularization):** Consider the regularized loss:
$$\tilde{\mathcal{L}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|^2$$

For $\|\theta\|$ large, the regularization term dominates:
$$\nabla \tilde{\mathcal{L}}(\theta) \cdot \theta \approx \lambda \|\theta\|^2 > 0$$

This pushes trajectories inward. The sublevel set $\{\theta : \tilde{\mathcal{L}}(\theta) \leq c\}$ is compact for any $c$, and trajectories cannot escape due to strict loss decrease.

**Alternative (Gradient Clipping):** With gradient clipping at norm $G$:
$$\theta_{t+1} = \theta_t - \eta \cdot \min(1, G/\|\nabla \mathcal{L}\|) \cdot \nabla \mathcal{L}$$

Combined with loss boundedness $\mathcal{L} \geq 0$, trajectories remain in a bounded region.

**Conclusion:** The attractor $\mathcal{A} = \bigcap_{t \geq 0} \overline{\{\theta(s) : s \geq t\}}$ exists and is compact.

### Step 2: Gradient-like Structure (Lyapunov Property)

**Claim:** The loss function $\mathcal{L}$ is a strict Lyapunov function for the gradient flow.

**Proof:** Along any gradient descent trajectory:
$$\frac{d\mathcal{L}}{dt} = \nabla \mathcal{L} \cdot \frac{d\theta}{dt} = -\|\nabla \mathcal{L}\|^2 \leq 0$$

with equality if and only if $\nabla \mathcal{L}(\theta) = 0$ (critical point).

**Strict Decrease:** For $\theta \notin \mathrm{Crit}(\mathcal{L})$:
$$\mathcal{L}(\theta(t_2)) < \mathcal{L}(\theta(t_1)) \quad \text{for all } t_2 > t_1$$

**Implication:** Gradient descent is a gradient-like system. Trajectories flow monotonically from higher to lower loss values.

### Step 3: No Periodic Orbits (No Training Oscillations)

**Claim:** There are no periodic orbits in the gradient flow.

**Proof by contradiction:** Suppose $\theta(t)$ is a non-constant periodic orbit with period $T > 0$:
$$\theta(t + T) = \theta(t) \quad \text{for all } t$$

By the Lyapunov property:
$$\mathcal{L}(\theta(T)) = \mathcal{L}(\theta(0)) - \int_0^T \|\nabla \mathcal{L}(\theta(s))\|^2 ds$$

Since the orbit is non-constant, $\nabla \mathcal{L} \neq 0$ somewhere on it, so:
$$\int_0^T \|\nabla \mathcal{L}(\theta(s))\|^2 ds > 0$$

But periodicity requires $\mathcal{L}(\theta(T)) = \mathcal{L}(\theta(0))$. Contradiction.

**RL Interpretation:** In policy gradient methods, this means:
- No indefinite oscillation between different policy modes
- Training converges to a stationary point
- Value function estimates stabilize

### Step 4: Convergence via Łojasiewicz Inequality

**Claim:** Every trajectory converges to a single critical point (not a set of them).

**Łojasiewicz-Simon Inequality:** Near a critical point $\theta^*$:
$$\|\nabla \mathcal{L}(\theta)\| \geq C \cdot |\mathcal{L}(\theta) - \mathcal{L}(\theta^*)|^{1-\theta}$$

for some $\theta \in (0, 1/2]$ (Łojasiewicz exponent).

**Convergence Rate Derivation:** Let $E(t) = \mathcal{L}(\theta(t)) - \mathcal{L}(\theta^*)$. Then:
$$\frac{dE}{dt} = -\|\nabla \mathcal{L}\|^2 \leq -C^2 E^{2(1-\theta)}$$

Integrating this differential inequality:
- For $\theta = 1/2$ (analytic losses): Exponential convergence $E(t) \leq E(0) e^{-C^2 t}$
- For $\theta < 1/2$: Polynomial convergence $E(t) \leq C' t^{-\frac{1-\theta}{1-2\theta}}$

**Finite-Length Trajectories:** The Łojasiewicz inequality implies:
$$\int_0^\infty \|\dot{\theta}(t)\| dt < \infty$$

Trajectories have finite arc length, so they must converge to a limit point (not oscillate or cycle).

**Single Limit Point:** The omega-limit set $\omega(\theta_0) = \{\theta : \exists t_n \to \infty, \theta(t_n) \to \theta\}$ must be a single critical point, not a connected set of them.

### Step 5: Morse Decomposition (Loss Landscape Topology)

**Definition (Unstable Manifold).** For a critical point $\theta^*$:
$$W^u(\theta^*) = \{\theta \in \mathcal{A} : \lim_{t \to \infty} \theta(t) = \theta^*\}$$

This is the **basin of attraction** of $\theta^*$ under gradient descent.

**Attractor Decomposition:** By Step 4, every point in $\mathcal{A}$ converges to some critical point:
$$\mathcal{A} = \bigsqcup_{\theta^* \in \mathrm{Crit}(\mathcal{L})} W^u(\theta^*)$$

This is a disjoint union (each trajectory has a unique limit).

**Morse Index Structure:**
- **Minima (index 0):** Stable attractors, $W^u(\theta^*)$ has positive measure
- **Saddles (index k):** Unstable, $W^u(\theta^*)$ has measure zero (codimension k)
- **Maxima (index n):** Fully unstable, $W^u(\theta^*) = \{\theta^*\}$

**Mode Connectivity:** Two minima $\theta_1^*, \theta_2^*$ are connected if there exists a path through a saddle $\theta^s$:
$$\theta_1^* \leftarrow \theta^s \rightarrow \theta_2^*$$

The Morse decomposition reveals all such connections.

### Step 6: Certificate Construction

**Output Certificate:**
$$K_{\mathrm{LossLandscape}}^+ = (\mathsf{gradient\_like}, \mathrm{Crit}(\mathcal{L}), \{m_i\}_{i=1}^N, \mathsf{convergence\_rates})$$

**Components:**

1. **Gradient-like witness:** The loss function $\mathcal{L}$ itself, with $\frac{d\mathcal{L}}{dt} \leq 0$

2. **Critical points:** $\mathrm{Crit}(\mathcal{L}) = \{\theta_1^*, \ldots, \theta_N^*\}$ ordered by loss value

3. **Morse indices:** $m_i = \#\{\lambda < 0 : \lambda \in \mathrm{spec}(\nabla^2 \mathcal{L}(\theta_i^*))\}$

4. **Convergence rates:** For each basin, the Łojasiewicz exponent $\theta_i$ and constant $C_i$

5. **No-cycle guarantee:** Proof that $\int_0^\infty \|\dot{\theta}\| dt < \infty$

## Connections to AI/ML Literature

### Loss Landscape Analysis

**Hessian Eigenspectrum (Sagun et al., 2017; Ghorbani et al., 2019).** The Hessian $\nabla^2 \mathcal{L}(\theta)$ has a characteristic spectrum:
- **Bulk:** Most eigenvalues near zero (flat directions)
- **Outliers:** Few large positive eigenvalues (sharp directions)
- **Near saddles:** Some negative eigenvalues

**Morse Index Interpretation:** The Morse index counts negative eigenvalues. High-dimensional landscapes have many saddle points with high Morse index, but these are unstable and avoided by gradient descent with noise.

**Connection to FACT-Soft-Morse:** The theorem guarantees that near each critical point, the Hessian eigenspectrum determines convergence behavior and basin geometry.

**References:**
- Sagun, L., Evci, U., Guney, V. U., Dauphin, Y., Bottou, L. (2017). Empirical analysis of the Hessian of over-parameterized neural networks. *ICLR Workshop*.
- Ghorbani, B., Krishnan, S., Xiao, Y. (2019). An investigation into neural net optimization via Hessian eigenvalue density. *ICML*.

### Saddle Point Escape

**Strict Saddle Property (Ge et al., 2015; Jin et al., 2017).** A function $\mathcal{L}$ is **strict saddle** if at every saddle point, the Hessian has at least one strictly negative eigenvalue:
$$\theta^* \text{ saddle} \implies \lambda_{\min}(\nabla^2 \mathcal{L}(\theta^*)) < -\gamma$$

**Escape Time Bounds:** With gradient descent + noise (e.g., SGD), saddle escape time is:
$$T_{\mathrm{escape}} = O\left(\frac{\log(d)}{\eta \gamma}\right)$$

where $d$ is dimension and $\gamma$ is the spectral gap.

**Morse Decomposition Interpretation:** Saddle points have Morse index $\geq 1$, so their unstable manifolds have measure zero. Almost all trajectories escape to minima.

**References:**
- Ge, R., Huang, F., Jin, C., Yuan, Y. (2015). Escaping from saddle points: Online stochastic gradient for tensor decomposition. *COLT*.
- Jin, C., Ge, R., Netrapalli, P., Kakade, S. M., Jordan, M. I. (2017). How to escape saddle points efficiently. *ICML*.
- Lee, J. D., Simchowitz, M., Jordan, M. I., Recht, B. (2016). Gradient descent only converges to minimizers. *COLT*.

### Mode Connectivity

**Linear Mode Connectivity (Garipov et al., 2018; Draxler et al., 2018).** Different trained networks (local minima) are often connected by paths of low loss:
$$\mathcal{L}(\alpha \theta_1 + (1-\alpha) \theta_2) \approx \mathcal{L}(\theta_1) \approx \mathcal{L}(\theta_2)$$

**Lottery Ticket Connectivity (Frankle et al., 2020).** Paths exist through the loss landscape connecting different solutions.

**Morse Theory Interpretation:** Mode connectivity follows from the Morse decomposition:
- If two minima $\theta_1^*, \theta_2^*$ are connected via a saddle $\theta^s$, there is a gradient descent path
- The number of saddles between minima (connection structure) is topological

**FACT-Soft-Morse Contribution:** The theorem proves that:
1. All critical points are isolated (no degenerate continua)
2. The connection structure is determined by the Morse index ordering
3. Paths exist whenever the loss difference exceeds the saddle barrier

**References:**
- Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D. P., Wilson, A. G. (2018). Loss surfaces, mode connectivity, and fast ensembling of DNNs. *NeurIPS*.
- Draxler, F., Veschgini, K., Salmhofer, M., Hamprecht, F. (2018). Essentially no barriers in neural network energy landscape. *ICML*.
- Frankle, J., Dziugaite, G. K., Roy, D. M., Carbin, M. (2020). Linear mode connectivity and the lottery ticket hypothesis. *ICML*.

### Polyak-Łojasiewicz Condition in Deep Learning

**PL Condition (Polyak, 1963).** A function satisfies the PL condition if:
$$\frac{1}{2}\|\nabla \mathcal{L}(\theta)\|^2 \geq \mu (\mathcal{L}(\theta) - \mathcal{L}^*)$$

This is the special case $\theta = 1/2$ of the Łojasiewicz inequality.

**Over-parameterized Networks (Liu et al., 2022).** Wide neural networks satisfy PL in a neighborhood of initialization, ensuring:
- Linear convergence: $\mathcal{L}(\theta_t) - \mathcal{L}^* \leq (1 - \eta \mu)^t (\mathcal{L}(\theta_0) - \mathcal{L}^*)$
- Global convergence to zero training loss

**Connection to Łojasiewicz-Simon:** The PL condition is a uniform version of the Łojasiewicz inequality with $\theta = 1/2$. FACT-Soft-Morse extends this to:
- Non-uniform exponents $\theta \in (0, 1/2]$ near different critical points
- Polynomial convergence rates when $\theta < 1/2$

**References:**
- Polyak, B. T. (1963). Gradient methods for minimizing functionals. *Zhurnal Vychislitel'noi Matematiki i Matematicheskoi Fiziki*.
- Liu, C., Zhu, L., Belkin, M. (2022). Loss landscapes and optimization in over-parameterized non-linear systems and neural networks. *Applied and Computational Harmonic Analysis*.
- Karimi, H., Nutini, J., Schmidt, M. (2016). Linear convergence of gradient and proximal-gradient methods under the Polyak-Łojasiewicz condition. *ECML-PKDD*.

### Value Function Convergence in RL

**Policy Gradient as Gradient Descent.** Policy gradient methods optimize:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

The gradient flow is:
$$\dot{\theta} = \nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)]$$

**Convergence to Stationary Policies:** Under standard assumptions:
- $J(\theta)$ is smooth (continuous policy, bounded rewards)
- Gradient descent on $-J(\theta)$ is a gradient-like system
- FACT-Soft-Morse implies convergence to a local optimum

**No Policy Cycling:** The no-periodic-orbit guarantee means:
- Policy gradient does not oscillate between different policies
- Value function estimates stabilize
- Behavioral policy converges

**References:**
- Sutton, R. S., McAllester, D., Singh, S., Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. *NeurIPS*.
- Agarwal, A., Kakade, S. M., Lee, J. D., Mahajan, G. (2021). On the theory of policy gradient methods: Optimality, approximation, and distribution shift. *JMLR*.

## Implementation Notes

### Hessian Eigenvalue Computation

**Power Iteration for Top Eigenvalues:**
```python
def top_hessian_eigenvalue(loss_fn, params, num_iters=100):
    """Compute largest Hessian eigenvalue via power iteration."""
    v = torch.randn_like(params)
    v = v / v.norm()

    for _ in range(num_iters):
        # Hessian-vector product via autodiff
        grad = torch.autograd.grad(loss_fn(params), params, create_graph=True)
        Hv = torch.autograd.grad(grad, params, grad_outputs=v)

        eigenvalue = (v * Hv).sum()
        v = Hv / Hv.norm()

    return eigenvalue.item()
```

**Negative Curvature Detection (Morse Index):**
```python
def count_negative_eigenvalues(hessian, threshold=-1e-5):
    """Count negative eigenvalues (Morse index)."""
    eigenvalues = torch.linalg.eigvalsh(hessian)
    return (eigenvalues < threshold).sum().item()
```

### Łojasiewicz Exponent Estimation

**Empirical Exponent Estimation:**
```python
def estimate_lojasiewicz_exponent(loss_history, grad_norm_history):
    """Estimate Lojasiewicz exponent from training logs."""
    # Near critical point: |grad| >= C * |loss - loss*|^(1-theta)
    # Log scale: log|grad| >= log(C) + (1-theta) * log|loss - loss*|

    loss_star = loss_history[-1]  # Final loss as proxy for L*
    log_loss_diff = np.log(np.abs(loss_history[:-1] - loss_star) + 1e-10)
    log_grad = np.log(grad_norm_history[:-1] + 1e-10)

    # Linear regression: slope = 1 - theta
    slope, intercept = np.polyfit(log_loss_diff, log_grad, 1)
    theta = 1 - slope

    return np.clip(theta, 0.01, 0.5)  # Lojasiewicz exponent in (0, 1/2]
```

### Mode Connectivity Verification

**Path Finding Between Minima:**
```python
def find_connecting_path(theta1, theta2, loss_fn, num_points=10):
    """Find low-loss path between two minima."""
    # Linear interpolation baseline
    alphas = np.linspace(0, 1, num_points)
    linear_path = [alpha * theta2 + (1 - alpha) * theta1 for alpha in alphas]
    linear_losses = [loss_fn(p).item() for p in linear_path]

    # Bezier curve optimization for lower barrier
    control_point = nn.Parameter((theta1 + theta2) / 2)
    optimizer = torch.optim.Adam([control_point], lr=0.01)

    for _ in range(100):
        path_losses = []
        for alpha in alphas:
            # Quadratic Bezier
            p = (1-alpha)**2 * theta1 + 2*(1-alpha)*alpha * control_point + alpha**2 * theta2
            path_losses.append(loss_fn(p))

        barrier = max(path_losses)
        barrier.backward()
        optimizer.step()
        optimizer.zero_grad()

    return max(linear_losses), barrier.item()  # Linear vs optimized barrier
```

### Convergence Rate Monitoring

**Tracking Convergence Type:**
```python
def classify_convergence(loss_history, window=100):
    """Classify convergence as exponential or polynomial."""
    recent = loss_history[-window:]
    t = np.arange(len(recent))

    # Exponential fit: L(t) = A * exp(-lambda * t)
    log_loss = np.log(np.array(recent) + 1e-10)
    exp_slope, exp_intercept = np.polyfit(t, log_loss, 1)
    exp_rate = -exp_slope

    # Polynomial fit: L(t) = C * t^(-alpha)
    log_t = np.log(t + 1)
    poly_slope, poly_intercept = np.polyfit(log_t, log_loss, 1)
    poly_rate = -poly_slope

    # Compare fits
    exp_residual = np.mean((log_loss - (exp_intercept + exp_slope * t))**2)
    poly_residual = np.mean((log_loss - (poly_intercept + poly_slope * log_t))**2)

    if exp_residual < poly_residual:
        return "exponential", exp_rate  # Lojasiewicz exponent = 1/2
    else:
        # Polynomial: rate = (1-theta)/(1-2*theta) => theta = (rate-1)/(2*rate-1)
        theta = (poly_rate - 1) / (2 * poly_rate - 1) if poly_rate > 0.5 else 0.5
        return "polynomial", theta
```

### Saddle Point Detection and Escape

**Negative Curvature Direction:**
```python
def escape_saddle(params, loss_fn, lr=0.01, noise_scale=0.1):
    """Escape saddle point by following negative curvature."""
    grad = torch.autograd.grad(loss_fn(params), params)[0]

    if grad.norm() < 1e-6:  # Near critical point
        # Check for negative curvature
        eigenvalue = top_hessian_eigenvalue(loss_fn, params)
        if eigenvalue < -1e-4:  # Saddle detected
            # Add noise in negative curvature direction
            noise = torch.randn_like(params) * noise_scale
            params = params + noise

    return params - lr * grad
```

## Literature References

### Morse Theory and Dynamical Systems
- Milnor, J. (1963). *Morse Theory*. Princeton University Press.
- Conley, C. (1978). *Isolated Invariant Sets and the Morse Index*. CBMS Regional Conference Series.
- Smale, S. (1961). On gradient dynamical systems. *Annals of Mathematics*, 74(1), 199-206.

### Łojasiewicz Inequality
- Łojasiewicz, S. (1963). Une propriete topologique des sous-ensembles analytiques reels. *Colloques Internationaux du CNRS*, 117, 87-89.
- Simon, L. (1983). Asymptotics for a class of nonlinear evolution equations, with applications to geometric problems. *Annals of Mathematics*, 118(3), 525-571.
- Kurdyka, K. (1998). On gradients of functions definable in o-minimal structures. *Annales de l'Institut Fourier*, 48(3), 769-783.

### Loss Landscape Analysis
- Li, H., Xu, Z., Taylor, G., Studer, C., Goldstein, T. (2018). Visualizing the loss landscape of neural nets. *NeurIPS*.
- Fort, S., Jastrzebski, S. (2019). Large scale structure of neural network loss landscapes. *NeurIPS*.
- Choromanska, A., Henaff, M., Mathieu, M., Arous, G. B., LeCun, Y. (2015). The loss surfaces of multilayer networks. *AISTATS*.

### Saddle Point Dynamics
- Dauphin, Y. N., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., Bengio, Y. (2014). Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. *NeurIPS*.
- Lee, J. D., Simchowitz, M., Jordan, M. I., Recht, B. (2016). Gradient descent only converges to minimizers. *COLT*.
- Panageas, I., Piliouras, G. (2017). Gradient descent only converges to minimizers: Non-isolated critical points and invariant regions. *ITCS*.

### Mode Connectivity
- Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D. P., Wilson, A. G. (2018). Loss surfaces, mode connectivity, and fast ensembling of DNNs. *NeurIPS*.
- Draxler, F., Veschgini, K., Salmhofer, M., Hamprecht, F. (2018). Essentially no barriers in neural network energy landscape. *ICML*.
- Nguyen, Q. (2019). On connected sublevel sets in deep learning. *ICML*.

### Convergence Theory
- Polyak, B. T. (1963). Gradient methods for minimizing functionals. *Zh. Vychisl. Mat. Mat. Fiz.*, 3(4), 643-653.
- Karimi, H., Nutini, J., Schmidt, M. (2016). Linear convergence of gradient and proximal-gradient methods under the Polyak-Łojasiewicz condition. *ECML-PKDD*.
- Liu, C., Zhu, L., Belkin, M. (2022). Loss landscapes and optimization in over-parameterized non-linear systems and neural networks. *Applied and Computational Harmonic Analysis*.

### Reinforcement Learning
- Sutton, R. S., Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Agarwal, A., Kakade, S. M., Lee, J. D., Mahajan, G. (2021). On the theory of policy gradient methods: Optimality, approximation, and distribution shift. *JMLR*.
- Mei, S., Xiao, T., Szepesvari, C., Schuurmans, D. (2020). On the global convergence rates of softmax policy gradient methods. *ICML*.
