---
title: "KRNL-Lyapunov - AI/RL/ML Translation"
---

# KRNL-Lyapunov: Value Function as Lyapunov Stability Certificate

## Original Statement (Hypostructure)

Given a hypostructure with validated interface permits for dissipation ($D_E$ with $C=0$), compactness ($C_\mu$), and local stiffness ($\mathrm{LS}_\sigma$), there exists a canonical Lyapunov functional $\mathcal{L}: \mathcal{X} \to \mathbb{R} \cup \{\infty\}$ with the following properties:

1. **Monotonicity:** Along any trajectory, $t \mapsto \mathcal{L}(u(t))$ is nonincreasing and strictly decreasing whenever $u(t) \notin M$.
2. **Stability:** $\mathcal{L}$ attains its minimum precisely on $M$.
3. **Height Equivalence:** $\mathcal{L}(x) - \mathcal{L}_{\min} \asymp (\Phi(x) - \Phi_{\min})$ on energy sublevels.
4. **Uniqueness:** Any other Lyapunov functional $\Psi$ with these properties satisfies $\Psi = f \circ \mathcal{L}$ for some monotone $f$.

**Explicit Construction (Value Function):**
$$\mathcal{L}(x) := \inf\left\{\Phi(y) + \mathcal{C}(x \to y) : y \in M\right\}$$

where the infimal cost is:
$$\mathcal{C}(x \to y) := \inf\left\{\int_0^T \mathfrak{D}(S_s x) \, ds : S_T x = y, T < \infty\right\}$$

## AI/RL/ML Setting

**State Space:** $\mathcal{S}$ -- state space of the environment (continuous or discrete)

**Action Space:** $\mathcal{A}$ -- available actions for the agent

**Policy:** $\pi: \mathcal{S} \to \Delta(\mathcal{A})$ -- stochastic policy mapping states to action distributions

**Value Function:** $V^\pi: \mathcal{S} \to \mathbb{R}$ -- expected cumulative reward under policy $\pi$

**Goal Region:** $\mathcal{G} \subseteq \mathcal{S}$ -- target states (safe set, goal set, equilibrium)

**Dynamics:** $f: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$ or $P(s'|s,a)$ -- deterministic or stochastic transition

**Cost Function:** $c: \mathcal{S} \times \mathcal{A} \to \mathbb{R}_{\geq 0}$ -- instantaneous cost (negative reward)

## AI/RL/ML Statement

**Theorem (Value Function as Control Lyapunov Function).** Let $(\mathcal{S}, \mathcal{A}, f, c, \mathcal{G})$ be a control system with:

1. **(Dissipation / Policy Improvement)** There exists a policy $\pi$ such that following $\pi$ decreases cost-to-go:
   $$V^\pi(s') \leq V^\pi(s) - c(s, a) \quad \text{for } s' = f(s, \pi(s))$$

2. **(Compactness / Bounded State Space)** The reachable state space under bounded cost has compact sublevel sets, or the neural network approximation has bounded capacity.

3. **(Local Stiffness / Goal Stability)** Near the goal region $\mathcal{G}$, the Bellman residual satisfies a Lojasiewicz-type inequality:
   $$\|\nabla V(s)\| \geq C \cdot |V(s) - V^*|^{1-\theta}$$

Then the **optimal value function** $V^*(s)$ serves as a **Control Lyapunov Function (CLF)** with:

**(A) Monotonicity:** $V^*(s_{t+1}) \leq V^*(s_t)$ along optimal trajectories, with strict decrease outside $\mathcal{G}$

**(B) Stability Certificate:** $V^*(s) = 0 \iff s \in \mathcal{G}$

**(C) Height Equivalence:** $V^*(s) \asymp d(s, \mathcal{G})^2$ near the goal (quadratic lower bound)

**(D) Uniqueness:** Any other CLF $L$ satisfying (A)-(C) is a monotone transformation of $V^*$

**Explicit Construction (Bellman Value Function):**
$$V^*(s) := \inf_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^{T-1} c(s_t, a_t) \mid s_0 = s, s_T \in \mathcal{G} \right]$$

## Terminology Translation Table

| Hypostructure | AI/RL/ML |
|---------------|----------|
| State space $\mathcal{X}$ | State space $\mathcal{S}$ |
| Flow $S_t$ | Policy rollout $s_{t+1} = f(s_t, \pi(s_t))$ |
| Height functional $\Phi$ | Value function $V(s)$ |
| Lyapunov functional $\mathcal{L}$ | Control Lyapunov Function (CLF) |
| Safe manifold $M$ | Goal region $\mathcal{G}$ (or safe set) |
| Dissipation $\mathfrak{D}$ | Policy $\pi(a|s)$ (dissipation via action selection) |
| Energy decrease | Bellman residual decrease |
| Transport cost $\mathcal{C}(x \to y)$ | Cumulative cost / negative return |
| Monotonicity of $\mathcal{L}$ | TD error $\leq 0$ (value decreases along trajectory) |
| Height equivalence | Value $\asymp$ distance-to-goal |
| Interface permit $D_E^+$ | Bounded cost per step |
| Interface permit $C_\mu^+$ | Compact state space / bounded network |
| Interface permit $\mathrm{LS}_\sigma^+$ | Goal reachability / policy convergence |
| Gradient flow | Policy gradient descent |
| Łojasiewicz inequality | Polyak-Lojasiewicz condition for convergence |

## Proof Sketch

### Setup: Value Functions and Lyapunov Stability

The central insight is that the **optimal value function** in reinforcement learning naturally serves as a **Lyapunov function** for the closed-loop system under the optimal policy. This connection bridges control theory's stability analysis with RL's value-based methods.

**Key Definitions:**

- **Value Function:** $V^\pi(s) = \mathbb{E}_\pi[\sum_{t=0}^\infty \gamma^t r_t | s_0 = s]$ (discounted) or cost-to-go (undiscounted)
- **Bellman Equation:** $V^*(s) = \min_a \{c(s,a) + V^*(f(s,a))\}$ (deterministic) or with expectation (stochastic)
- **Control Lyapunov Function:** $L: \mathcal{S} \to \mathbb{R}_{\geq 0}$ such that $\exists \pi: L(f(s,\pi(s))) < L(s)$ for $s \notin \mathcal{G}$

**Lyapunov Stability Theorem (Classical):** If there exists a CLF $L$ with:
1. $L(s) > 0$ for $s \notin \mathcal{G}$, $L(s) = 0$ for $s \in \mathcal{G}$
2. $L(s') < L(s)$ for the next state under some policy

Then the system is asymptotically stable to $\mathcal{G}$.

### Step 1: Value Function Satisfies Bellman Optimality

**Claim:** The optimal value function $V^*$ satisfies the Bellman optimality equation.

**Bellman Optimality Principle:** For any state $s$:
$$V^*(s) = \min_{a \in \mathcal{A}} \left\{ c(s, a) + V^*(f(s, a)) \right\}$$

with boundary condition $V^*(s) = 0$ for $s \in \mathcal{G}$.

**Verification:** This follows from the principle of optimality: an optimal trajectory from $s$ consists of an optimal first action followed by an optimal continuation.

**Value Iteration Convergence:** The sequence $V_{k+1}(s) = \min_a \{c(s,a) + V_k(f(s,a))\}$ converges to $V^*$ under standard conditions (contraction in sup-norm for discounted case, or finite horizon).

### Step 2: V* Decreases Along Optimal Trajectories (Lyapunov Property)

**Claim:** $V^*(s_{t+1}) \leq V^*(s_t) - c(s_t, a_t^*)$ along optimal trajectories.

**Proof:** By the Bellman equation:
$$V^*(s_t) = c(s_t, a_t^*) + V^*(s_{t+1})$$

Rearranging:
$$V^*(s_{t+1}) = V^*(s_t) - c(s_t, a_t^*)$$

Since $c \geq 0$, we have $V^*(s_{t+1}) \leq V^*(s_t)$.

**Strict Decrease:** If $s_t \notin \mathcal{G}$ and $c(s_t, a) > 0$ for all actions, then $V^*(s_{t+1}) < V^*(s_t)$.

**This is the Lyapunov monotonicity property:** The value function strictly decreases along trajectories until the goal is reached.

### Step 3: Goal Characterization (Minimum at Equilibrium)

**Claim:** $V^*(s) = 0 \iff s \in \mathcal{G}$.

**Proof of $\Leftarrow$:** If $s \in \mathcal{G}$, no further cost is incurred: $V^*(s) = 0$ by definition.

**Proof of $\Rightarrow$:** If $V^*(s) = 0$ but $s \notin \mathcal{G}$, then any action incurs positive cost: $c(s, a) > 0$. But:
$$V^*(s) = c(s, a^*) + V^*(f(s, a^*)) \geq c(s, a^*) > 0$$

Contradiction. Hence $V^*(s) = 0 \implies s \in \mathcal{G}$.

**Stability Certificate:** The goal region $\mathcal{G}$ is precisely the zero-level set of $V^*$, making $V^*$ a valid Lyapunov certificate for asymptotic stability to $\mathcal{G}$.

### Step 4: Height Equivalence (Value Bounds Distance to Goal)

**Upper Bound:** Let $d(s, \mathcal{G}) = \inf_{g \in \mathcal{G}} \|s - g\|$ be the distance to goal. For Lipschitz dynamics and bounded cost:
$$V^*(s) \leq C_1 \cdot d(s, \mathcal{G})$$

(proportional to minimum path length at maximum cost rate).

**Lower Bound (Łojasiewicz-type):** Under the stiffness condition, near the goal:
$$V^*(s) \geq c_0 \cdot d(s, \mathcal{G})^{2\theta}$$

For $\theta = 1/2$ (quadratic growth), this gives $V^*(s) \asymp d(s, \mathcal{G})^2$.

**Interpretation:** The value function is equivalent to a distance measure -- it provides both upper and lower bounds on the "difficulty" of reaching the goal.

### Step 5: Uniqueness of Canonical Value Function

**Claim:** Any CLF $L$ satisfying (A)-(C) is a monotone transformation of $V^*$.

**Proof:**

*Level Set Correspondence:* Both $V^*$ and $L$ decrease strictly along optimal trajectories. Each trajectory passes through each level set exactly once.

*Monotone Bijection:* For states $s, s'$ with $V^*(s) < V^*(s')$, the optimal trajectory from $s'$ must pass through a state with value equal to $V^*(s)$. By strict decrease of $L$ along this trajectory, $L(s) \leq L(s')$.

*Construction:* Define $f: \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$ by $f(V^*(s)) := L(s)$. This is well-defined (constant on level sets) and monotone, giving $L = f \circ V^*$.

## Connections to Classical Results

### Control Lyapunov Functions (Sontag, 1983; Artstein, 1983)

A **Control Lyapunov Function (CLF)** for a system $\dot{x} = f(x, u)$ is a smooth function $V: \mathbb{R}^n \to \mathbb{R}_{\geq 0}$ satisfying:

$$\inf_{u \in \mathcal{U}} \left\{ \nabla V(x) \cdot f(x, u) \right\} < 0 \quad \forall x \neq 0$$

**Artstein's Theorem:** A system is asymptotically stabilizable iff it admits a CLF.

**Connection to KRNL-Lyapunov:** The optimal value function $V^*$ is precisely a CLF -- the Bellman equation guarantees the existence of a control (optimal action) that decreases $V^*$.

**Reference:**
- Sontag, E. D. (1983). A Lyapunov-like characterization of asymptotic controllability. *SIAM J. Control Optim.*, 21(3), 462-471.
- Artstein, Z. (1983). Stabilization with relaxed controls. *Nonlinear Anal.*, 7(11), 1163-1173.

### Hamilton-Jacobi-Bellman Equation (Optimal Control)

The continuous-time analogue of the Bellman equation is the **Hamilton-Jacobi-Bellman (HJB) PDE**:

$$0 = \min_{u \in \mathcal{U}} \left\{ c(x, u) + \nabla V(x) \cdot f(x, u) \right\}$$

This is the viscosity solution characterization of the value function.

**Connection:** The hypostructure's Hamilton-Jacobi formulation (KRNL-Hamilton-Jacobi) is the PDE version of KRNL-Lyapunov, with the value function $V$ solving the HJB equation.

**Reference:** Fleming, W. H., & Soner, H. M. (2006). *Controlled Markov Processes and Viscosity Solutions* (2nd ed.). Springer.

### Safe Reinforcement Learning (Berkenkamp et al., 2017)

Safe RL uses Lyapunov functions to **certify safety** during learning:

**Lyapunov-based Safe RL:** Learn a policy $\pi$ and Lyapunov function $L$ jointly such that:
1. $L(s') \leq L(s) - \epsilon$ for $s' \sim P(\cdot|s, \pi(s))$ (decrease condition)
2. $\{s : L(s) \leq c\}$ is safe (sublevel sets avoid unsafe regions)

**Region of Attraction:** The sublevel set $\{s : L(s) \leq c\}$ is a certified **region of attraction** -- any trajectory starting there remains safe and converges to the goal.

**Key Insight:** KRNL-Lyapunov provides the theoretical foundation: the value function is the canonical Lyapunov function, and safe RL algorithms approximate it.

**Reference:**
- Berkenkamp, F., Turchetta, M., Schoellig, A., & Krause, A. (2017). Safe model-based reinforcement learning with stability guarantees. *NeurIPS 2017*.
- Chow, Y., Nachum, O., Duenez-Guzman, E., & Ghavamzadeh, M. (2018). A Lyapunov-based approach to safe reinforcement learning. *NeurIPS 2018*.

### Neural Lyapunov Functions (Chang et al., 2019)

Neural networks can **learn Lyapunov functions** directly:

**Neural Lyapunov Approach:**
1. Parameterize $V_\theta(s)$ as a neural network
2. Train to satisfy Lyapunov conditions:
   - $V_\theta(s) > 0$ for $s \neq 0$
   - $\dot{V}_\theta(s) = \nabla V_\theta \cdot f(s, \pi(s)) < 0$
3. Use SMT solvers or sampling to verify conditions

**Loss Function (Lyapunov Risk):**
$$\mathcal{L}(\theta) = \mathbb{E}_s\left[ \max(0, \dot{V}_\theta(s) + \epsilon) + \max(0, -V_\theta(s) + \delta \|s\|^2) \right]$$

**Connection to KRNL-Lyapunov:** The "canonical" Lyapunov functional from KRNL-Lyapunov is the optimal target for neural Lyapunov learning. The uniqueness property ensures all valid neural Lyapunov functions are monotone transformations of $V^*$.

**Reference:**
- Chang, Y. C., Roohi, N., & Gao, S. (2019). Neural Lyapunov control. *NeurIPS 2019*.
- Richards, S. M., Berkenkamp, F., & Krause, A. (2018). The Lyapunov neural network: Adaptive stability certification for safe learning of dynamical systems. *CoRL 2018*.

### Temporal Difference Learning and Bellman Residual

**TD Error as Lyapunov Decrease:** The temporal difference error:
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

measures how much the value function "should" decrease. For the optimal value function, $\delta_t = 0$ (Bellman optimality).

**Bellman Residual Minimization:** Training minimizes:
$$\mathcal{L}(\theta) = \mathbb{E}\left[ (V_\theta(s) - r - \gamma V_{\theta'}(s'))^2 \right]$$

This is analogous to minimizing the Lyapunov derivative's violation.

**Reference:** Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

## Implementation Notes

### Neural Lyapunov Function Architecture

**Input-Convex Neural Networks (ICNNs):** Ensure $V(s)$ is convex by construction:
- All weights non-negative in hidden layers
- Convex activation functions (ReLU, softplus)
- Guarantees unique minimum and convex sublevel sets

**Positive-Definite Parameterization:**
$$V_\theta(s) = \|g_\theta(s)\|^2 + \epsilon \|s\|^2$$

ensures $V(s) > 0$ for $s \neq 0$ and $V(0) = 0$ (if $g_\theta(0) = 0$).

### Lyapunov Constraint Enforcement

**Lagrangian Relaxation:** Add Lyapunov constraint to policy loss:
$$\mathcal{L}_\pi = \mathbb{E}[c(s,a)] + \lambda \mathbb{E}[\max(0, V(s') - V(s) + \alpha)]$$

**Projection Methods:** Project policy updates to satisfy $\dot{V} < 0$:
$$\pi' = \arg\min_\pi \|\pi - \pi_{\text{new}}\|^2 \quad \text{s.t.} \quad \dot{V}^\pi(s) < 0$$

### Verification of Neural Lyapunov Functions

**Sampling-Based:** Check $\dot{V}(s) < 0$ at many sampled states.

**SMT-Based (dReal):** Formally verify Lyapunov conditions using satisfiability modulo theories:
$$\forall s \in \mathcal{S}: V(s) > 0 \land \dot{V}(s) < 0$$

**Interval Bound Propagation:** Compute certified bounds on $\dot{V}$ over regions.

**Reference:**
- Dai, H., Landry, B., Yang, L., Pavone, M., & Tedrake, R. (2021). Lyapunov-stable neural-network control. *RSS 2021*.
- Abate, A., Ahmed, D., Giacobbe, M., & Peruffo, A. (2021). Formal synthesis of Lyapunov neural networks. *IEEE Control Systems Letters*.

### Connection to Actor-Critic Methods

**Critic as Lyapunov:** In actor-critic RL, the critic $V_\phi(s)$ learns the value function. Under optimal training, $V_\phi \to V^*$, which is the canonical Lyapunov function.

**Safe Actor-Critic:** Constrain actor updates to decrease the critic:
$$\pi_{\theta'} = \arg\max_\pi \mathbb{E}_{a \sim \pi}[Q(s,a)] \quad \text{s.t.} \quad V(s') \leq V(s) - \epsilon$$

## Certificate Construction

The proof yields an explicit certificate for safe RL:

**Certificate:** $K_V^+ = (V^*, \mathcal{G}, \pi^*, \text{Lyapunov Proof})$

**Components:**

1. **Value Function $V^*$:**
   - Neural network weights $\theta$ such that $V_\theta \approx V^*$
   - Or tabular representation for finite MDPs

2. **Goal Region $\mathcal{G}$:**
   - Explicit description: $\mathcal{G} = \{s : \|s - s_{\text{goal}}\| < \epsilon\}$
   - Verification: $s \in \mathcal{G} \iff V^*(s) \approx 0$

3. **Optimal Policy $\pi^*$:**
   - $\pi^*(s) = \arg\min_a \{c(s,a) + V^*(f(s,a))\}$
   - Or learned policy network $\pi_\phi(a|s)$

4. **Lyapunov Decrease Certificate:**
   - For sampled/verified states: $V^*(s') < V^*(s) - \alpha c(s,a)$
   - Region of attraction: $\{s : V^*(s) \leq c\}$ remains invariant

5. **Bellman Optimality Verification:**
   - TD error $\approx 0$ for trained value function
   - Or formal verification via SMT/interval methods

## Literature References

### Control-Lyapunov Functions
- Sontag, E. D. (1983). A Lyapunov-like characterization of asymptotic controllability. *SIAM J. Control Optim.*, 21(3), 462-471.
- Artstein, Z. (1983). Stabilization with relaxed controls. *Nonlinear Anal.*, 7(11), 1163-1173.
- Khalil, H. K. (2002). *Nonlinear Systems* (3rd ed.). Prentice Hall, Ch. 4 (Lyapunov Stability).

### Safe Reinforcement Learning
- Berkenkamp, F., Turchetta, M., Schoellig, A., & Krause, A. (2017). Safe model-based reinforcement learning with stability guarantees. *NeurIPS 2017*.
- Chow, Y., Nachum, O., Duenez-Guzman, E., & Ghavamzadeh, M. (2018). A Lyapunov-based approach to safe reinforcement learning. *NeurIPS 2018*.
- Garcia, J., & Fernandez, F. (2015). A comprehensive survey on safe reinforcement learning. *JMLR*, 16(1), 1437-1480.

### Neural Lyapunov Functions
- Chang, Y. C., Roohi, N., & Gao, S. (2019). Neural Lyapunov control. *NeurIPS 2019*.
- Richards, S. M., Berkenkamp, F., & Krause, A. (2018). The Lyapunov neural network. *CoRL 2018*.
- Dai, H., Landry, B., Yang, L., Pavone, M., & Tedrake, R. (2021). Lyapunov-stable neural-network control. *RSS 2021*.
- Abate, A., Ahmed, D., Giacobbe, M., & Peruffo, A. (2021). Formal synthesis of Lyapunov neural networks. *IEEE Control Systems Letters*.

### Optimal Control and HJB
- Fleming, W. H., & Soner, H. M. (2006). *Controlled Markov Processes and Viscosity Solutions* (2nd ed.). Springer.
- Bertsekas, D. P. (2019). *Reinforcement Learning and Optimal Control*. Athena Scientific.

### Reinforcement Learning Foundations
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Szepesvari, C. (2010). *Algorithms for Reinforcement Learning*. Morgan & Claypool.
