---
title: "KRNL-Jacobi - AI/RL/ML Translation"
---

# KRNL-Jacobi: Action Reconstruction

## Original Theorem

**[KRNL-Jacobi] Action Reconstruction** (Theorem {prf:ref}`mt-krnl-jacobi`)

Given a hypostructure satisfying interface permits $D_E$ (dissipation-energy inequality), $\mathrm{LS}_\sigma$ (linear stability), and $\mathrm{GC}_\nabla$ (gradient consistency), the canonical Lyapunov functional equals the geodesic distance in the Jacobi metric:

$$\mathcal{L}(x) = \Phi_{\min} + \mathrm{dist}_{g_{\mathfrak{D}}}(x, M)$$

where $g_{\mathfrak{D}} := \mathfrak{D} \cdot g$ is the conformal scaling of the base metric by the dissipation rate.

**Explicit Formula:**
$$\mathcal{L}(x) = \Phi_{\min} + \inf_{\gamma: x \to M} \int_0^1 \sqrt{\mathfrak{D}(\gamma(s))} \cdot \|\dot{\gamma}(s)\|_g \, ds$$

---

## AI/RL/ML Statement

**Theorem (Value Function as Policy-Weighted Return):**

Let $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ be a Markov Decision Process where:
- $\mathcal{S}$ is the state space
- $\mathcal{A}$ is the action space
- $P(s'|s,a)$ is the transition kernel
- $r(s,a)$ is the reward function
- $\gamma \in [0,1)$ is the discount factor

Let $\pi: \mathcal{S} \to \Delta(\mathcal{A})$ be a policy and $\mathcal{G} \subseteq \mathcal{S}$ be the set of goal states. Then the **optimal value function** from any state $s \in \mathcal{S}$ is:

$$V^*(s) = \sup_\pi \mathbb{E}_\pi\left[\sum_{t=0}^{\tau} \gamma^t r(s_t, a_t) \mid s_0 = s\right]$$

where $\tau = \inf\{t : s_t \in \mathcal{G}\}$ is the first hitting time of the goal set.

**Jacobi-Bellman Formulation:** Under gradient consistency (policy matches value gradient), the value function satisfies:

$$V^*(s) = \mathrm{dist}_{\pi}(s, \mathcal{G}) := \sup_{\tau: s \to \mathcal{G}} \int_0^T \pi(\tau(t)) \cdot \|\dot{\tau}(t)\| \, dt$$

where the supremum is over all trajectories $\tau$ connecting $s$ to the goal set $\mathcal{G}$.

**Key Insight:** The value function is the "reward-weighted geodesic distance" from the current state to the goal, where the policy $\pi$ conformally scales the state space metric---high-policy-density regions appear "closer" in value space.

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Equivalent | Interpretation |
|----------------------|---------------------|----------------|
| State space $\mathcal{X}$ | State space $\mathcal{S}$ | Set of all possible agent configurations |
| Height function $\Phi(x)$ | Value function $V(s)$ | Expected cumulative reward from state $s$ |
| Dissipation rate $\mathfrak{D}(x)$ | Policy density $\pi(a\|s)$ | Action selection probability at state $s$ |
| Jacobi metric $g_{\mathfrak{D}} = \mathfrak{D} \cdot g$ | Policy-weighted metric | High-policy regions have shorter distances |
| Conformal scaling by $\mathfrak{D}$ | Reward shaping / advantage weighting | Policy modulates effective state-space geometry |
| Geodesic in Jacobi metric | Optimal trajectory | Reward-maximizing path to goal |
| Safe manifold $M$ | Goal states $\mathcal{G}$ | Terminal states with maximum value |
| Lyapunov functional $\mathcal{L}(x)$ | Value function $V^*(s)$ | Optimal expected return |
| Gradient flow $\dot{u} = -\nabla \Phi$ | Greedy policy $\pi^*(s) = \arg\max_a Q^*(s,a)$ | Follow value gradient |
| Boundary condition $\mathcal{L}\|_M = \Phi_{\min}$ | $V^*(s) = 0$ for terminal states | Zero future reward at goal |
| Gradient consistency $\mathrm{GC}_\nabla$ | Bellman consistency | $V^* = \mathcal{T}V^*$ where $\mathcal{T}$ is Bellman operator |
| Action functional $\int \sqrt{\mathfrak{D}} \|\dot{\gamma}\|$ | Cumulative return $\sum \gamma^t r_t$ | Total reward along trajectory |
| Action minimization | Return maximization | Optimal control objective |
| Jacobi field | Policy gradient / trajectory sensitivity | How trajectories change under policy perturbation |

---

## Proof Sketch

### Setup: MDP as Dynamical System

**Definition (MDP Dynamics):**
An MDP defines a controlled dynamical system on state space $\mathcal{S}$:
- **Deterministic case:** $s_{t+1} = f(s_t, a_t)$ where $a_t \sim \pi(\cdot|s_t)$
- **Stochastic case:** $s_{t+1} \sim P(\cdot|s_t, a_t)$

**Value Function as Height:**
The value function $V^\pi(s) = \mathbb{E}_\pi[\sum_{t=0}^\infty \gamma^t r_t | s_0 = s]$ serves as the "height" function $\Phi$:
- High-value states are "peaks" (abundant future reward)
- Low-value states are "valleys" (limited future reward)
- Goal states are "sea level" ($V = 0$ for absorbing terminal states)

**Policy as Dissipation:**
The policy $\pi(a|s)$ determines the rate of "flow" through state space:
- High $\pi(a|s)$ means rapid transitions in direction $a$
- The "dissipation" $\mathfrak{D}(s) \sim \mathbb{E}_{a \sim \pi}[A(s,a)^2]$ where $A$ is advantage

---

### Step 1: Bellman Equation as Hamilton-Jacobi

**Continuous-Time Bellman (HJB Equation):**
For a continuous-time MDP with dynamics $\dot{s} = f(s, a)$ and running reward $r(s,a)$:

$$\max_a \left[ r(s, a) + \nabla V(s) \cdot f(s, a) \right] = 0$$

This is the **Hamilton-Jacobi-Bellman (HJB) equation**.

**Discrete-Time Bellman:**
$$V^*(s) = \max_a \left[ r(s,a) + \gamma \mathbb{E}_{s' \sim P}[V^*(s')] \right]$$

**Connection to KRNL-Jacobi:**
The hypostructure Hamilton-Jacobi equation $\|\nabla_g \mathcal{L}\|_g^2 = \mathfrak{D}$ translates to:

$$\|\nabla V^*(s)\|^2 \propto \pi^*(a|s) \cdot \|f(s,a)\|^2$$

The squared gradient of value equals the policy-weighted squared velocity---this is **gradient consistency** in RL language.

---

### Step 2: Policy-Weighted Metric (Jacobi Metric)

**Definition (Policy Metric):**
Define the policy-induced Riemannian metric on state space:

$$g_\pi(s) := \mathbb{E}_{a \sim \pi(\cdot|s)}[\nabla_s \log \pi(a|s) \otimes \nabla_s \log \pi(a|s)]$$

This is the **Fisher information metric** of the policy.

**Conformal Scaling Interpretation:**
When policy probability correlates with advantage:
$$\pi(a|s) \propto \exp(\beta A(s,a))$$

the policy metric conformally scales the base metric by the exponentiated advantage:
$$g_\pi \approx e^{2\beta A} \cdot g_{\text{base}}$$

**Jacobi Distance in RL:**
The "Jacobi distance" between states becomes:

$$d_\pi(s, s') = \inf_{\tau: s \to s'} \int_0^1 \sqrt{\pi(\tau(t))} \cdot \|\dot{\tau}(t)\| \, dt$$

High-policy-probability regions have shorter Jacobi distance---the policy "curves" the state space to make optimal paths shorter.

---

### Step 3: Value Function as Geodesic Distance

**Theorem (Value = Jacobi Distance to Goal):**
Under Bellman consistency (policy matches value gradient), the optimal value function equals the Jacobi distance to the goal set:

$$V^*(s) = V^*_{\max} - d_\pi(s, \mathcal{G})$$

where $V^*_{\max}$ is the maximum achievable value (at goal states).

**Proof Outline:**

*Step 3a (Gradient Consistency).* The optimal policy satisfies:
$$\pi^*(a|s) \propto \exp(\beta Q^*(s,a)) \propto \exp(\beta (r(s,a) + \gamma V^*(s')))$$

For continuous dynamics: $\|\dot{s}\|^2 = \|f(s, a^*)\|^2$ where $a^* = \arg\max Q^*$.

*Step 3b (Jacobi Length = Return).* For trajectory $\tau = (s_0, s_1, \ldots, s_T)$ under policy $\pi$:
$$L_\pi(\tau) = \sum_{t=0}^{T-1} \sqrt{\pi(a_t|s_t)} \cdot \|s_{t+1} - s_t\|$$

Under gradient consistency: $\sqrt{\pi(a|s)} \cdot \|f(s,a)\| \propto r(s,a)$

Hence: $L_\pi(\tau) \propto \sum_t r(s_t, a_t) = R(\tau)$

*Step 3c (Optimal Path = Maximum Return).* The supremum of $L_\pi$ over paths equals the supremum of returns---achieved by following the optimal policy.

---

### Step 4: Bellman Backup as Variational Principle

**Bellman Operator:**
$$(\mathcal{T}V)(s) = \max_a \left[ r(s,a) + \gamma \mathbb{E}[V(s')] \right]$$

**Variational Characterization:**
$$V^*(s) = \sup_\pi \inf_{\tau: s \to \mathcal{G}} \int_\tau r(\tau(t), \pi(\tau(t))) \, dt$$

This is the RL analog of the calculus of variations: find the path (and policy) that extremizes cumulative reward.

**Connection to Action Principle:**
In classical mechanics, the **action** $S = \int L \, dt$ is minimized by physical trajectories.
In RL, the **return** $R = \sum \gamma^t r_t$ is maximized by optimal policies.

The Jacobi metric encodes how the policy "tilts" the landscape:
- Maupertuis' principle: $\delta \int \sqrt{2(E-V)} \, ds = 0$ (fixed energy)
- RL analog: $\delta \int \sqrt{\pi(a|s)} \, ds = 0$ (optimal policy)

---

### Step 5: Jacobi Fields as Policy Gradients

**Definition (Jacobi Field in RL):**
A Jacobi field $J(t)$ along an optimal trajectory $\tau^*$ measures how nearby trajectories deviate under policy perturbation.

**Policy Gradient Connection:**
$$\nabla_\theta V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^\pi(s_t, a_t)\right]$$

The policy gradient is an integral of "Jacobi-like" deviations weighted by Q-values.

**Second Variation (Natural Gradient):**
The Fisher information metric $F_\theta = \mathbb{E}[\nabla \log \pi \otimes \nabla \log \pi]$ determines:
- Curvature of the policy manifold
- Natural gradient direction: $\tilde{\nabla}_\theta V = F_\theta^{-1} \nabla_\theta V$
- Stability of policy updates (trust region)

**Jacobi Equation in RL:**
The Jacobi equation $\ddot{J} + R(J, \dot{\tau})\dot{\tau} = 0$ (where $R$ is curvature) translates to:
- How Q-value Hessian affects trajectory sensitivity
- Conjugate points = policy bifurcations = mode collapse

---

## Connections to Classical RL Results

### Bellman Equation and Dynamic Programming

**Bellman (1957):** The principle of optimality: an optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision.

**Connection:** KRNL-Jacobi's geodesic reconstruction is the geometric analog:
- Optimal paths are geodesics in the Jacobi metric
- Subpaths of geodesics are geodesics
- Geodesic distance satisfies the triangle inequality (dynamic programming)

### Policy Gradient Methods

**REINFORCE (Williams, 1992):**
$$\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) \cdot R]$$

**Connection:** The gradient of the return with respect to policy parameters is analogous to the gradient of the Lyapunov functional. The policy gradient theorem shows how to "flow" in parameter space to increase value.

### Natural Policy Gradient

**Kakade (2001), Amari (1998):**
Natural gradient uses Fisher information to account for the geometry of the policy manifold:
$$\theta_{t+1} = \theta_t + \alpha F_\theta^{-1} \nabla_\theta J$$

**Connection:** The Fisher metric is the RL analog of the Jacobi metric. Natural policy gradient is "geodesic optimization" in policy space---just as gradient flow is geodesic in the Jacobi metric of state space.

### Trust Region Methods (TRPO, PPO)

**Schulman et al. (2015, 2017):**
Constrain policy updates to stay within a trust region:
$$\max_\theta J(\theta) \quad \text{s.t.} \quad D_{KL}(\pi_\theta \| \pi_{\theta_{\text{old}}}) \leq \delta$$

**Connection:** The KL divergence constraint defines a "ball" in the policy-induced metric. Trust region optimization is geodesic approximation---small steps in the Jacobi metric direction.

### Soft Actor-Critic and Maximum Entropy RL

**Haarnoja et al. (2018):**
$$J(\pi) = \mathbb{E}_\pi\left[\sum_t \gamma^t (r_t + \alpha H(\pi(\cdot|s_t)))\right]$$

**Connection:** Adding entropy $H(\pi)$ to the objective smooths the value landscape, analogous to adding regularization to the Lyapunov functional. The entropy term modifies the effective "dissipation" rate.

### Temporal Difference Learning

**Sutton (1988):**
$$V(s_t) \leftarrow V(s_t) + \alpha [r_t + \gamma V(s_{t+1}) - V(s_t)]$$

**Connection:** TD learning approximates the solution to the Hamilton-Jacobi equation via stochastic gradient descent. The TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ measures "Bellman residual"---how far the current value estimate is from satisfying the HJB equation.

---

## Implementation Notes

### Computing the Policy-Weighted Metric

**Practical Approximation:**
For deep RL with neural network policies $\pi_\theta(a|s)$:

1. **Fisher Information Estimation:**
   ```python
   # Monte Carlo estimate of Fisher matrix
   log_probs = policy.log_prob(states, actions)
   grads = torch.autograd.grad(log_probs.sum(), policy.parameters())
   fisher = outer_product_average(grads, grads)
   ```

2. **Natural Gradient via Conjugate Gradient:**
   ```python
   # Solve F @ x = g for natural gradient direction
   def fisher_vector_product(v):
       return fisher @ v + damping * v
   natural_grad = conjugate_gradient(fisher_vector_product, policy_grad)
   ```

### Value Function as Distance

**Distance-Based Value Networks:**
Instead of learning $V(s)$ directly, learn:
$$V(s) = V_{\max} - d_\theta(s, \mathcal{G})$$

where $d_\theta$ is a learned distance function to goal states.

**Advantages:**
- Naturally enforces $V(s) = V_{\max}$ for $s \in \mathcal{G}$
- Triangle inequality provides structural regularization
- Interpretable as "how far from goal"

### Geodesic Planning

**Trajectory Optimization:**
Instead of optimizing actions directly, optimize trajectories in the Jacobi metric:

```python
def jacobi_trajectory_cost(trajectory, policy):
    """Compute Jacobi length of trajectory under policy."""
    cost = 0
    for t in range(len(trajectory) - 1):
        s, s_next = trajectory[t], trajectory[t+1]
        # Policy weight at state s
        pi_weight = policy.prob(s, infer_action(s, s_next))
        # Metric distance
        ds = torch.norm(s_next - s)
        # Jacobi length element
        cost += torch.sqrt(pi_weight) * ds
    return cost
```

### Stability via Lyapunov Certificates

**Neural Lyapunov Functions:**
Train a network $\mathcal{L}_\theta(s)$ satisfying:
1. $\mathcal{L}_\theta(s) \geq 0$ with equality iff $s \in \mathcal{G}$
2. $\mathcal{L}_\theta(s') < \mathcal{L}_\theta(s)$ along policy trajectories

**Loss Function:**
```python
def lyapunov_loss(lyap_net, policy, states, next_states):
    L_s = lyap_net(states)
    L_next = lyap_net(next_states)
    # Decrease condition
    decrease_loss = F.relu(L_next - L_s + margin)
    # Boundary condition (goal states)
    boundary_loss = lyap_net(goal_states).pow(2)
    return decrease_loss.mean() + boundary_loss.mean()
```

---

## Literature

### Reinforcement Learning Foundations
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. 2nd ed. MIT Press.
- Bertsekas, D. P. (2019). *Reinforcement Learning and Optimal Control*. Athena Scientific.

### Policy Gradient and Natural Gradient
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8, 229-256.
- Kakade, S. (2001). A natural policy gradient. *NIPS*.
- Amari, S. (1998). Natural gradient works efficiently in learning. *Neural Computation*, 10(2), 251-276.

### Hamilton-Jacobi-Bellman in Control
- Fleming, W. H., & Soner, H. M. (2006). *Controlled Markov Processes and Viscosity Solutions*. 2nd ed. Springer.
- Bardi, M., & Capuzzo-Dolcetta, I. (1997). *Optimal Control and Viscosity Solutions of Hamilton-Jacobi-Bellman Equations*. Birkhauser.

### Optimal Transport and Gradient Flows
- Ambrosio, L., Gigli, N., & Savare, G. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures*. Birkhauser.
- Villani, C. (2009). *Optimal Transport: Old and New*. Springer.

### Geometric Methods in RL
- Todorov, E. (2009). Efficient computation of optimal actions. *PNAS*, 106(28), 11478-11483.
- Levine, S. (2018). Reinforcement learning and control as probabilistic inference. *arXiv:1805.00909*.
- Nachum, O., et al. (2017). Bridging the gap between value and policy based reinforcement learning. *NIPS*.

### Lyapunov Methods in RL
- Berkenkamp, F., et al. (2017). Safe model-based reinforcement learning with stability guarantees. *NIPS*.
- Chang, Y. C., et al. (2019). Neural Lyapunov control. *NeurIPS*.
- Richards, S. M., et al. (2018). The Lyapunov neural network: Adaptive stability certification for safe learning of dynamical systems. *CoRL*.

### Trust Region and Geometry
- Schulman, J., et al. (2015). Trust region policy optimization. *ICML*.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Haarnoja, T., et al. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *ICML*.

---

## Summary

The KRNL-Jacobi theorem provides a geometric foundation for understanding value functions in reinforcement learning:

| Hypostructure | AI/RL/ML |
|--------------|----------|
| Jacobi metric $g_{\mathfrak{D}} = \mathfrak{D} \cdot g$ | Policy-weighted state metric |
| Geodesic distance to safe manifold | Value function (distance to goal) |
| Lyapunov functional $\mathcal{L}(x)$ | Optimal value $V^*(s)$ |
| Gradient flow | Greedy/optimal policy execution |
| Hamilton-Jacobi equation | Bellman optimality equation |
| Conformal scaling by dissipation | Policy shapes effective geometry |
| Jacobi field | Policy gradient / trajectory sensitivity |
| Action minimization | Return maximization |

**Core Message:** The optimal value function is the "policy-weighted geodesic distance" from the current state to the goal. The policy conformally scales the state space, making high-probability-action regions "closer" in value terms. This geometric perspective:

1. Explains why **natural policy gradient** (using Fisher metric) outperforms vanilla gradient
2. Justifies **trust region methods** as geodesic approximations
3. Connects **Bellman consistency** to the Hamilton-Jacobi equation from optimal control
4. Provides a principled foundation for **Lyapunov-based safe RL**

The action principle in physics (minimize action) becomes the return principle in RL (maximize return), both unified through the language of geodesics in conformally-scaled metrics.
