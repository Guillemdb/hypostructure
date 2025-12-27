---
title: "KRNL-MetricAction - AI/RL/ML Translation"
---

# KRNL-MetricAction: Extended Action Reconstruction

## Original Hypostructure Statement

**Reference:** {prf:ref}`mt-krnl-metric-action`

Under interface permit $\mathrm{GC}'_\nabla$ (dissipation-slope equality), the reconstruction theorems extend to general metric spaces. The Lyapunov functional satisfies:

$$\mathcal{L}(x) = \Phi_{\min} + \inf_{\gamma: M \to x} \int_0^1 |\partial \Phi|(\gamma(s)) \cdot |\dot{\gamma}|(s) \, ds$$

where $|\dot{\gamma}|$ denotes the metric derivative and the infimum ranges over all absolutely continuous curves from the safe manifold $M$ to $x$.

---

## AI/RL/ML Statement

**The framework extends to reinforcement learning and machine learning with sample/query complexity as the action integral.**

Let $\mathcal{S}$ be a state space, $\Pi$ the space of policies $\pi: \mathcal{S} \to \Delta(\mathcal{A})$, and $V^\pi: \mathcal{S} \to \mathbb{R}$ the value function under policy $\pi$. Define:

- **Value function** $V: \mathcal{S} \to \mathbb{R}$ measuring expected cumulative reward (height $\Phi$)
- **Policy metric** $d: \Pi \times \Pi \to \mathbb{R}_{\geq 0}$ measuring distance in policy space (KL divergence, Wasserstein, etc.)
- **Policy gradient slope** $|\partial V|(\pi) := \limsup_{\pi' \to \pi} \frac{[V^{\pi'}(s_0) - V^\pi(s_0)]^+}{d(\pi, \pi')}$

Then the **sample complexity** for learning an $\epsilon$-optimal policy from initial policy $\pi_0$ is:

$$N_{\mathrm{samples}}(\pi_0 \to \pi^*) = N_{\min} + \inf_{\gamma: \pi_0 \to \Pi^*} \int_0^1 |\partial V|(\gamma(s)) \cdot |\dot{\gamma}|(s) \, ds$$

where the infimum is over all learning trajectories $\gamma$ from initial policy $\pi_0$ to the set of optimal policies $\Pi^*$.

**Interpretation:** The sample complexity to reach optimality equals the integral of "local improvement rate" times "policy update speed" along the optimal learning path.

---

## Terminology Translation Table

| Hypostructure | AI/RL/ML | Interpretation |
|---------------|----------|----------------|
| State space $\mathcal{X}$ | Policy space $\Pi$ | Space of all policies $\pi: \mathcal{S} \to \Delta(\mathcal{A})$ |
| Height function $\Phi$ | Value function $V^\pi(s_0)$ | Expected cumulative reward from initial state |
| Metric $d$ | KL divergence / Wasserstein distance | Distance between policies or state distributions |
| Metric slope $|\partial\Phi|$ | Policy gradient magnitude $\|\nabla_\theta J(\theta)\|$ | Local improvement rate per unit policy change |
| Metric derivative $|\dot{\gamma}|$ | Learning rate / policy update magnitude | Rate of policy change during training |
| Action integral $\int |\partial\Phi| \cdot |\dot{\gamma}|$ | Sample complexity | Total samples needed along learning trajectory |
| Safe manifold $M$ | Optimal policy set $\Pi^*$ | Policies achieving maximum value |
| Dissipation $\mathfrak{D}$ | Variance of policy gradient / exploration cost | Rate of "using up" information from samples |
| Lyapunov $\mathcal{L}(x)$ | Regret-to-go / suboptimality gap | Distance to optimal performance |
| Gradient flow | Policy gradient descent / Natural policy gradient | Optimization trajectory in policy space |
| Geodesic | Optimal learning trajectory | Path minimizing sample complexity |
| Energy-Dissipation Identity | Variance-bias tradeoff | Relationship between updates and progress |
| Wasserstein space $\mathcal{P}_2$ | State occupancy distribution space | Space of distributions $d^\pi(s)$ over states |

---

## Proof Sketch

### Setup: Policy Space as Metric Space

We work with the following structures from RL theory:

1. **State space** $\mathcal{S}$, action space $\mathcal{A}$, transition dynamics $P(s'|s,a)$
2. **Policy space** $\Pi = \{\pi_\theta : \theta \in \Theta\}$ parametrized by $\theta \in \mathbb{R}^d$
3. **Value function** $V^\pi(s) = \mathbb{E}^\pi\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s\right]$
4. **Objective** $J(\theta) = \mathbb{E}_{s_0 \sim \mu_0}[V^{\pi_\theta}(s_0)]$

**Definition (Policy Metric Space).** The *policy metric space* $(\Pi, d)$ uses one of:

- **KL divergence:** $d_{\mathrm{KL}}(\pi, \pi') = \mathbb{E}_{s \sim d^\pi}\left[\mathrm{KL}(\pi(\cdot|s) \| \pi'(\cdot|s))\right]$
- **Wasserstein distance:** $d_{W_2}(d^\pi, d^{\pi'})$ on state occupancy measures
- **Fisher metric:** $d_F(\theta, \theta') = \sqrt{(\theta - \theta')^T \mathcal{F}(\theta) (\theta - \theta')}$

The **policy gradient slope** at $\pi_\theta$ is:
$$|\partial V|(\pi_\theta) := \|\nabla_\theta J(\theta)\|_{\mathcal{F}^{-1}} = \sqrt{\nabla_\theta J^T \mathcal{F}^{-1} \nabla_\theta J}$$

where $\mathcal{F}(\theta)$ is the Fisher information matrix.

---

### Step 1: Policy Gradient as Metric Slope

**Lemma (Policy Gradient Theorem).** For a differentiable policy $\pi_\theta$:
$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta}\left[Q^{\pi_\theta}(s,a) \nabla_\theta \log \pi_\theta(a|s)\right]$$

**Interpretation as Metric Slope:**

The policy gradient $\nabla_\theta J$ measures the rate of value improvement per unit change in parameters:
$$|\partial V|(\pi_\theta) = \lim_{\delta\theta \to 0} \frac{J(\theta + \delta\theta) - J(\theta)}{\|\delta\theta\|_\mathcal{F}}$$

This exactly matches the hypostructure metric slope $|\partial\Phi|$ when:
- Height $\Phi \leftrightarrow$ Value $J(\theta)$
- Metric $d \leftrightarrow$ Fisher distance on $\Theta$

**Natural Policy Gradient (NPG).** The natural gradient $\tilde{\nabla}J = \mathcal{F}^{-1}\nabla J$ is the steepest ascent direction in Fisher geometry:
$$\tilde{\nabla}_\theta J = \arg\max_{\|v\|_\mathcal{F} \leq 1} \langle \nabla J, v \rangle$$

---

### Step 2: Learning Trajectory as Absolutely Continuous Curve

A **learning trajectory** is a path $\gamma: [0, T] \to \Pi$ in policy space generated by gradient updates:
$$\dot{\theta}(t) = \eta(t) \cdot \tilde{\nabla}_\theta J(\theta(t))$$

**Metric Derivative:** The metric derivative measures the "speed" of policy updates:
$$|\dot{\gamma}|(t) = \lim_{h \to 0} \frac{d(\pi_{\theta(t+h)}, \pi_{\theta(t)})}{|h|}$$

For continuous-time gradient flow with Fisher metric:
$$|\dot{\gamma}|(t) = \|\dot{\theta}(t)\|_\mathcal{F} = \eta(t) \cdot |\partial V|(\pi_{\theta(t)})$$

**Absolutely Continuous Learning:** A learning trajectory is *absolutely continuous* if:
$$d(\pi_{\theta(s)}, \pi_{\theta(t)}) \leq \int_s^t |\dot{\gamma}|(u) \, du$$

This holds for gradient descent with bounded learning rate.

---

### Step 3: Action Integral as Sample Complexity

**Theorem (Action-Sample Complexity Identity).** For a learning trajectory $\gamma: [0, T] \to \Pi$:

$$\mathrm{Action}(\gamma) := \int_0^T |\partial V|(\gamma(t)) \cdot |\dot{\gamma}|(t) \, dt \propto N_{\mathrm{samples}}(\gamma)$$

**Derivation:**

*Step 3.1 (Sample-per-update bound).* To estimate $\nabla_\theta J$ with variance $\sigma^2$, we need:
$$n(t) \geq \frac{\mathrm{Var}[\hat{\nabla}J]}{|\partial V|^2(\theta(t))} = \frac{\sigma^2}{|\partial V|^2(\theta(t))}$$
samples per gradient step.

*Step 3.2 (Updates-per-distance).* To move distance $\delta$ in policy space with step size $\eta$:
$$k \approx \frac{\delta}{\eta \cdot |\partial V|(\theta)}$$
updates required.

*Step 3.3 (Total samples).* Total samples along trajectory:
$$N_{\mathrm{samples}} = \int_0^T n(t) \cdot \frac{|\dot{\gamma}|(t)}{\eta(t) \cdot |\partial V|(\theta(t))} \, dt$$

For optimal learning rate $\eta(t) \propto |\partial V|(\theta(t))$ (adaptive):
$$N_{\mathrm{samples}} \propto \int_0^T |\partial V|(\gamma(t)) \cdot |\dot{\gamma}|(t) \, dt = \mathrm{Action}(\gamma)$$

---

### Step 4: Optimal Learning Path and Sample Complexity Lower Bound

**Theorem (Sample Complexity as Lyapunov).** The minimum sample complexity to learn an $\epsilon$-optimal policy from $\pi_0$ is:

$$N^*(\pi_0) = N_{\min} + \inf_{\gamma: \pi_0 \to \Pi^*_\epsilon} \int_0^1 |\partial V|(\gamma(s)) \cdot |\dot{\gamma}|(s) \, ds$$

where $\Pi^*_\epsilon = \{\pi : J(\pi) \geq J^* - \epsilon\}$.

**Properties:**

1. **Optimality at convergence:** $N^*(\pi^*) = N_{\min}$ for optimal policies

2. **Monotonicity:** $N^*$ decreases along any valid learning trajectory

3. **Bellman-like recursion:** For intermediate policy $\pi'$:
   $$N^*(\pi_0) = N(\pi_0 \to \pi') + N^*(\pi')$$

4. **Lipschitz bound:** $|N^*(\pi) - N^*(\pi')| \leq C \cdot d(\pi, \pi')$

**Lower Bound Interpretation:** The action integral provides a fundamental lower bound:
$$N_{\mathrm{samples}} \geq \int_{\gamma^*} |\partial V| \cdot |\dot{\gamma}| \, ds$$

No learning algorithm can use fewer samples than the action along the geodesic.

---

### Step 5: Wasserstein Gradient Flow in RL

**Setting:** State occupancy measures $d^\pi \in \mathcal{P}(\mathcal{S})$ with Wasserstein metric $W_2$.

**Energy Functional:** Negative expected reward:
$$\Phi(d^\pi) = -\mathbb{E}_{s \sim d^\pi}[r(s)]$$

**Gradient Flow:** The Wasserstein gradient flow of $\Phi$ corresponds to:
$$\partial_t d^\pi = -\nabla \cdot (d^\pi \nabla \frac{\delta \Phi}{\delta d^\pi}) = \nabla \cdot (d^\pi \nabla r)$$

which pushes probability mass toward high-reward states.

**Action Reconstruction:** The sample complexity in terms of distributions:
$$N^*(d^{\pi_0}) = \inf_{\gamma} \int_0^1 |\partial \Phi|(d^{\gamma(s)}) \cdot W_2(d^{\gamma(s)}, d^{\gamma(s+ds)}) \, ds$$

This connects to optimal transport theory: the sample complexity measures the "cost" of transporting the state distribution from initial to optimal.

---

### Step 6: Discrete Policy Space (Bandits and Tabular RL)

**Setting:** Finite action space $|\mathcal{A}| = K$, tabular policies $\pi \in \Delta(\mathcal{A})$.

**Metric:** KL divergence $d(\pi, \pi') = \mathrm{KL}(\pi \| \pi')$.

**Metric Slope:**
$$|\partial V|(\pi) = \max_{a: \pi(a) > 0} \frac{Q(a) - V}{\sqrt{\pi(a)}}$$

**Gradient Flow:** Softmax policy gradient / mirror descent:
$$\dot{\pi}(a) = \pi(a)(Q(a) - V)$$

**Sample Complexity (Bandits):** For $K$-armed bandits with gap $\Delta$:
$$N^* = \sum_{a: \Delta_a > 0} \frac{1}{\Delta_a^2} = \int_{\text{path}} |\partial V| \cdot |\dot{\pi}| \, dt$$

The action integral recovers the instance-dependent regret bound.

---

## Connections to Classical Results

### PAC-RL Sample Complexity Bounds

**Classical Result (Kakade, 2003):** For tabular MDPs with $|\mathcal{S}|$ states, $|\mathcal{A}|$ actions, horizon $H$:
$$N_{\mathrm{PAC}} = \tilde{O}\left(\frac{|\mathcal{S}||\mathcal{A}|H^3}{\epsilon^2}\right)$$

**Action Integral Interpretation:** The PAC bound corresponds to:
$$\mathrm{Action} = \int_0^1 |\partial V|(\gamma) \cdot |\dot{\gamma}| \, ds = \int_0^1 \frac{H}{\sqrt{|\mathcal{S}||\mathcal{A}|}} \cdot \sqrt{|\mathcal{S}||\mathcal{A}|} \cdot H \, ds$$

The factors:
- $|\partial V| \sim H/\sqrt{|\mathcal{S}||\mathcal{A}|}$: gradient magnitude scales with horizon and inversely with state-action count
- $|\dot{\gamma}| \sim \sqrt{|\mathcal{S}||\mathcal{A}|} \cdot H$: policy updates scale with problem size
- Path length $\sim H$: learning takes $H$ effective epochs

### Information-Theoretic Lower Bounds

**Classical Result (Lattimore-Szepesvari):** For bandits with $K$ arms:
$$N_{\mathrm{lower}} = \Omega\left(\sum_{a=1}^K \frac{1}{\mathrm{KL}(p_a \| p^*)}\right)$$

**Metric Action Interpretation:** The KL divergence in the denominator is exactly the metric slope:
$$|\partial V|(\pi) = \sqrt{\sum_a \pi(a) (\nabla_a \log \pi)^2 \cdot Q(a)^2} \approx \sqrt{\mathrm{KL}(\pi \| \pi^*)}$$

The information-theoretic lower bound equals the action along the geodesic from uniform to optimal.

### Natural Policy Gradient Convergence

**Classical Result (Kakade, 2001):** Natural policy gradient converges as:
$$J^* - J(\theta_t) \leq \frac{C}{t}$$

**Proof via Action Integral:**

The NPG update $\theta_{t+1} = \theta_t + \eta \mathcal{F}^{-1}\nabla J$ satisfies:
$$|\dot{\gamma}|(t) = \eta |\partial V|(\theta_t)$$

Total action:
$$\mathrm{Action}(0 \to T) = \int_0^T \eta |\partial V|^2(\theta_t) \, dt = \eta \int_0^T |\partial V|^2 \, dt$$

By the Energy-Dissipation Identity:
$$J(\theta_T) - J(\theta_0) = \int_0^T |\partial V|^2 \, dt$$

Hence sample complexity $N = \mathrm{Action} = \eta(J^* - J(\theta_0))$, giving $J^* - J(\theta_T) = O(1/N)$.

### Regret Bounds and Lyapunov

**Classical Result:** Regret of UCB/Thompson Sampling:
$$R(T) = \sum_{t=1}^T (V^* - V^{\pi_t}) = O(\sqrt{KT \log T})$$

**Lyapunov Interpretation:** Define $\mathcal{L}(\pi_t) = \mathbb{E}[\text{regret-to-go}]$. Then:
$$\mathcal{L}(\pi_0) = \sum_{t=0}^T (V^* - V^{\pi_t}) = \int_{\gamma} |\partial V| \cdot |\dot{\gamma}| \, ds$$

The regret equals the action integral along the learning trajectory.

---

## Implementation Notes

### Sample Complexity Estimation

To estimate the action integral numerically during training:

```python
def estimate_action_integral(trajectory, value_fn, policy_metric):
    """
    Estimate action integral along learning trajectory.

    Args:
        trajectory: List of (policy_params, num_samples) tuples
        value_fn: Function mapping params -> expected return
        policy_metric: Function mapping (params1, params2) -> distance

    Returns:
        Cumulative action (sample complexity proxy)
    """
    total_action = 0.0
    for i in range(len(trajectory) - 1):
        theta_t, n_t = trajectory[i]
        theta_next, n_next = trajectory[i + 1]

        # Metric slope: |nabla V| / sqrt(Fisher)
        grad_V = estimate_policy_gradient(theta_t, n_t)
        fisher = estimate_fisher_matrix(theta_t, n_t)
        metric_slope = np.sqrt(grad_V @ np.linalg.solve(fisher, grad_V))

        # Metric derivative: policy distance / time
        metric_deriv = policy_metric(theta_t, theta_next)

        # Action increment
        total_action += metric_slope * metric_deriv

    return total_action
```

### Optimal Learning Rate from Action Principle

The action integral suggests an adaptive learning rate:

```python
def optimal_learning_rate(theta, grad, fisher, target_action_per_step):
    """
    Compute learning rate to achieve target action per step.

    The action per step is |grad|_F * |theta_update|_F = eta * |grad|_F^2
    So eta = target_action / |grad|_F^2
    """
    natural_grad_norm_sq = grad @ np.linalg.solve(fisher, grad)
    eta = target_action_per_step / natural_grad_norm_sq
    return min(eta, max_lr)  # Clip for stability
```

### KL Divergence as Policy Metric

For policy gradient methods with softmax policies:

```python
def policy_kl_divergence(pi1, pi2, states):
    """
    Compute average KL divergence between policies.

    d_KL(pi1, pi2) = E_s[KL(pi1(.|s) || pi2(.|s))]
    """
    kl_sum = 0.0
    for s in states:
        kl_sum += kl_divergence(pi1.get_action_probs(s),
                                 pi2.get_action_probs(s))
    return kl_sum / len(states)
```

### Wasserstein Distance on State Distributions

For distributional RL:

```python
def wasserstein_state_distance(pi1, pi2, env, n_samples=1000):
    """
    Estimate W_2 distance between state occupancy measures.
    """
    states1 = collect_state_samples(env, pi1, n_samples)
    states2 = collect_state_samples(env, pi2, n_samples)
    return ot.emd2(states1, states2, cost_matrix)
```

---

## Literature

### Foundational RL Theory

- Kakade, S. (2001). A natural policy gradient. *NeurIPS*.
- Kakade, S. (2003). On the sample complexity of reinforcement learning. *PhD thesis, UCL*.
- Lattimore, T., Szepesvari, C. (2020). *Bandit Algorithms*. Cambridge University Press.

### Information-Theoretic Bounds

- Russo, D., Van Roy, B. (2016). An information-theoretic analysis of Thompson Sampling. *JMLR*.
- Agarwal, A., et al. (2020). Optimality and approximation with policy gradient methods in Markov decision processes. *COLT*.
- Jin, C., et al. (2018). Is Q-learning provably efficient? *NeurIPS*.

### Natural Gradient and Fisher Information

- Amari, S. (1998). Natural gradient works efficiently in learning. *Neural Computation*.
- Peters, J., Schaal, S. (2008). Natural actor-critic. *Neurocomputing*.
- Schulman, J., et al. (2015). Trust region policy optimization. *ICML*.

### Wasserstein and Optimal Transport in RL

- Bellemare, M.G., et al. (2017). A distributional perspective on reinforcement learning. *ICML*.
- Zhang, R., et al. (2018). A unified view of entropy-regularized Markov decision processes. *arXiv*.
- Richemond, P.H., Maginnis, B. (2017). On Wasserstein reinforcement learning and the Fokker-Planck equation. *arXiv*.

### Metric Gradient Flows

- Ambrosio, L., Gigli, N., Savare, G. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures*. Birkhauser.
- Jordan, R., Kinderlehrer, D., Otto, F. (1998). The variational formulation of the Fokker-Planck equation. *SIAM J. Math. Anal.*.
- Maas, J. (2011). Gradient flows of the entropy for finite Markov chains. *J. Funct. Anal.*.
- Mielke, A. (2011). A gradient structure for reaction-diffusion systems and for energy-drift-diffusion systems. *Nonlinearity*.

---

## Summary

The KRNL-MetricAction theorem translates to AI/RL/ML as follows:

1. **Metric spaces** $\to$ **Policy spaces** with KL/Fisher/Wasserstein geometry

2. **Metric slope** $|\partial \Phi|$ $\to$ **Policy gradient magnitude** $\|\nabla_\theta J\|_\mathcal{F}$

3. **Action integral** $\to$ **Sample complexity** (total samples along learning trajectory)

4. **Lyapunov** $\mathcal{L}$ $\to$ **Regret-to-go** / suboptimality gap

5. **Infimum over paths** $\to$ **Optimal learning algorithm** (minimizing sample complexity)

6. **Geodesic** $\to$ **Optimal learning trajectory** (natural policy gradient path)

7. **Energy-Dissipation Identity** $\to$ **Sample-value tradeoff** (samples used = improvement achieved)

The key insight is that sample complexity in RL satisfies the same variational principle as action in physics: the total "cost" (samples) to reach the goal (optimal policy) equals the path integral of "local cost rate" (policy gradient magnitude) times "speed" (policy update rate). Optimal algorithms follow geodesics in policy space, minimizing this action integral.
