# FACT-Soft-WP: Reward Shaping from Soft Specifications

## Overview

This document provides a complete AI/RL/ML translation of the FACT-Soft-WP theorem (Soft-to-WP Compilation) from the hypostructure framework. The translation establishes a formal correspondence between automatic well-posedness derivation from soft interfaces and reward shaping / potential-based shaping in reinforcement learning.

**Original Theorem Reference:** {prf:ref}`mt-fact-soft-wp`

---

## AI/RL/ML Statement

### Original Statement (Hypostructure)

For good types $T$ satisfying the Automation Guarantee, critical well-posedness is derived from soft interfaces:

**Soft Hypotheses:**
$$K_{\mathcal{H}_0}^+ \wedge K_{D_E}^+ \wedge K_{\mathrm{Bound}}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{Rep}_K}^+$$

**Produces:**
$$K_{\mathrm{WP}_{s_c}}^+$$

The mechanism uses template matching: the system signature is compared against known templates (semilinear parabolic, wave, Schrodinger, symmetric hyperbolic), and the corresponding well-posedness theorem is automatically instantiated.

---

## AI/RL/ML Formulation

### Setup

Consider a reinforcement learning problem where:

- **State space:** $\mathcal{S}$ (environment states)
- **Action space:** $\mathcal{A}$ (agent actions)
- **Value function:** $V^\pi(s)$ or $Q^\pi(s, a)$ (height/energy analog)
- **Policy:** $\pi(a|s)$ (dissipation analog)
- **Goal specification:** $G \subseteq \mathcal{S}$ (target states)
- **Soft interface certificates:** High-level problem specifications

The "weakest precondition" problem asks: What initial conditions guarantee goal achievement? The "soft WP" translation provides automatic reward shaping that guides learning toward the goal.

### Statement (AI/RL/ML Version)

**Theorem (Automatic Reward Shaping from Goal Specifications).**
Let $(\mathcal{S}, \mathcal{A}, P, G)$ be a goal-conditioned RL problem with soft specifications:

1. **Substrate Certificate ($K_{\mathcal{H}_0}^+$):** Well-defined transition dynamics $P(s'|s, a)$
2. **Value Certificate ($K_{D_E}^+$):** Bounded value function with Lyapunov-like decrease $V(s') \leq V(s) - \delta$ toward goal
3. **Boundary Certificate ($K_{\mathrm{Bound}}^+$):** Goal specification $G$ and boundary conditions
4. **Scaling Certificate ($K_{\mathrm{SC}_\lambda}^+$):** Discount structure $\gamma$ and temporal scaling
5. **Representation Certificate ($K_{\mathrm{Rep}_K}^+$):** Finite MDP description (tabular or feature-based)

Then there exists an **automatic reward shaping function** $F: \mathcal{S} \times \mathcal{S} \to \mathbb{R}$ such that:

$$R'(s, a, s') = R(s, a, s') + F(s, s')$$

where:
- $F(s, s') = \gamma \Phi(s') - \Phi(s)$ is potential-based (Ng et al. form)
- $\Phi: \mathcal{S} \to \mathbb{R}$ is the shaping potential derived from soft specifications
- The shaped reward $R'$ preserves optimal policy while accelerating learning

**Certificate Emitted:**
$$K_{\mathrm{WP}}^+ = (\mathsf{template\_ID}, \mathsf{shaping\_potential}, \mathsf{goal\_specification}, \mathsf{continuation\_criterion})$$

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Weakest precondition (WP) | Initial state requirements for goal achievement | $\{s_0 : V^*(s_0) \geq V_{\min}\}$ |
| Soft WP | Reward shaping / potential-based shaping | $F(s, s') = \gamma\Phi(s') - \Phi(s)$ |
| Height functional $\Phi$ | Value function $V(s)$ or negative cost-to-go | $\Phi(s) \leftrightarrow V^*(s)$ |
| Dissipation $\mathfrak{D}$ | Policy improvement / TD error | $\mathfrak{D} \leftrightarrow \|V^{\pi_{k+1}} - V^{\pi_k}\|$ |
| Energy-dissipation inequality | Bellman consistency / contraction | $V(s') \leq V(s) - \mathfrak{D}(s)$ |
| Template matching | Problem classification | Tabular / linear / neural MDP |
| Soft certificates | High-level specifications | Goal, discount, dynamics structure |
| Critical regularity $s_c$ | Feature dimension / representation capacity | $\dim(\phi(s))$ |
| Continuation criterion | Goal reachability condition | $V^*(s_0) > V_{\min}$ for goal achievement |
| Signature extraction | Problem featurization | MDP structure analysis |
| Template database | Known solvable problem classes | Tabular, linear, factored MDPs |
| $K_{\mathrm{WP}}^{\mathrm{inc}}$ (inconclusive) | Unknown problem class | No efficient algorithm known |
| Blowup criterion | Goal unreachability | $V^*(s_0) = -\infty$ (impossible goal) |
| Subcriticality $\alpha > \beta$ | Sufficient exploration / coverage | Sample complexity polynomial in $|\mathcal{S}|$ |
| Supercriticality $\alpha < \beta$ | Exploration hardness | Exponential sample complexity |

---

## Proof Sketch

### Step 1: Signature Extraction = Problem Classification

**Claim:** From soft certificates, extract a problem signature that identifies the MDP class.

**Construction:**

Given soft certificates $(K_{\mathcal{H}_0}^+, K_{D_E}^+, K_{\mathrm{Bound}}^+, K_{\mathrm{SC}_\lambda}^+, K_{\mathrm{Rep}_K}^+)$, extract:

1. **Dynamics Type (from $K_{\mathcal{H}_0}^+$):**
   - Deterministic: $P(s'|s, a) \in \{0, 1\}$
   - Stochastic: $P(s'|s, a) \in (0, 1)$
   - Ergodic: All states reachable from all states
   - Episodic: Terminal states exist

2. **Value Structure (from $K_{D_E}^+$):**
   - Monotonic: $V(s') \leq V(s)$ along trajectories (goal-directed)
   - Conservative: $\sum_t r_t$ bounded (finite-horizon)
   - Unbounded: Possible divergence (requires discounting)

3. **Goal Type (from $K_{\mathrm{Bound}}^+$):**
   - Reachability: $G = \{s_{\text{goal}}\}$ single target
   - Avoidance: $G = \mathcal{S} \setminus \{s_{\text{bad}}\}$ safety constraint
   - Temporal: LTL/CTL specification

4. **Scaling (from $K_{\mathrm{SC}_\lambda}^+$):**
   - Discount factor $\gamma \in [0, 1)$
   - Horizon $H$ (finite or infinite)

5. **Representation (from $K_{\mathrm{Rep}_K}^+$):**
   - Tabular: $|\mathcal{S}|, |\mathcal{A}|$ finite and small
   - Linear: $Q(s, a) = \phi(s, a)^\top \theta$
   - Neural: Deep function approximation

**Template Matching Algorithm:**

```python
def match_template(soft_certificates):
    dynamics = extract_dynamics_type(K_H0)
    value_structure = extract_value_structure(K_DE)
    goal_type = extract_goal_type(K_Bound)
    scaling = extract_scaling(K_SC)
    representation = extract_representation(K_Rep)

    # Template 1: Tabular MDP (analogous to parabolic)
    if representation == "tabular" and goal_type == "reachability":
        return Template.TABULAR_REACHABILITY

    # Template 2: Linear MDP (analogous to wave/dispersive)
    if representation == "linear" and value_structure == "monotonic":
        return Template.LINEAR_GOAL_CONDITIONED

    # Template 3: Factored MDP (analogous to symmetric hyperbolic)
    if representation == "factored" and dynamics == "deterministic":
        return Template.FACTORED_DETERMINISTIC

    # Template 4: Continuous control (analogous to NLS)
    if representation == "neural" and scaling.gamma < 1:
        return Template.CONTINUOUS_DISCOUNTED

    # No template matched
    return Template.INCONCLUSIVE
```

### Step 2: Potential-Based Shaping Derivation

**Claim:** For each matched template, derive the optimal shaping potential $\Phi(s)$.

**Template 1: Tabular Reachability (Parabolic Analog)**

For tabular MDPs with goal $G$, the shaping potential is the optimal value function:

$$\Phi(s) = V^*(s) = \max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r(s_t, a_t) \mid s_0 = s\right]$$

**Derivation:** By value iteration:
$$V_{k+1}(s) = \max_a \left[R(s, a) + \gamma \sum_{s'} P(s'|s, a) V_k(s')\right]$$

Convergence: $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$ (Bellman contraction).

**Shaping Function:**
$$F(s, s') = \gamma V^*(s') - V^*(s)$$

By Ng et al. (1999), this preserves optimal policies:
$$\pi^*_{R'} = \pi^*_R$$

**Template 2: Linear Goal-Conditioned (Wave Analog)**

For linear MDPs with feature map $\phi: \mathcal{S} \times \mathcal{A} \to \mathbb{R}^d$:
$$Q^*(s, a) = \phi(s, a)^\top \theta^*$$

**Shaping Potential:**
$$\Phi(s) = \max_a \phi(s, a)^\top \theta^*$$

The features $\phi$ play the role of Strichartz exponents, controlling sample complexity.

**Template 3: Factored MDP (Hyperbolic Analog)**

For factored MDPs with state $s = (s_1, \ldots, s_k)$ and independent dynamics:
$$P(s'|s, a) = \prod_{i=1}^k P_i(s'_i | s_{\text{pa}(i)}, a)$$

**Shaping Potential:**
$$\Phi(s) = \sum_{i=1}^k \Phi_i(s_i)$$

Decomposability enables efficient computation (analogous to Friedrichs energy method).

**Template 4: Continuous Control (NLS Analog)**

For continuous state spaces with neural approximation:
$$\Phi_\theta(s) = \text{NeuralNet}_\theta(s)$$

**Training Objective:**
$$\min_\theta \mathbb{E}_{s \sim \mathcal{D}}\left[(V^*(s) - \Phi_\theta(s))^2\right]$$

Regularization prevents overfitting (analogous to critical regularity constraints).

### Step 3: Certificate Construction

**Claim:** Assemble the shaping certificate with all required components.

**Certificate Structure:**
```python
K_WP = {
    'template_id': matched_template,
    'shaping_potential': Phi,  # Function S -> R
    'goal_specification': G,   # Target state set
    'continuation_criterion': {
        'reachability': V_star(s_0) > V_min,
        'sample_complexity': m(epsilon, delta),
        'convergence_rate': gamma
    },
    'literature': ['Ng1999', 'Sutton2018']
}
```

**Quantitative Bounds:**

| Template | Sample Complexity | Convergence Rate |
|----------|-------------------|------------------|
| Tabular | $O\left(\frac{|\mathcal{S}||\mathcal{A}|}{\epsilon^2(1-\gamma)^3}\right)$ | $O(\gamma^k)$ |
| Linear | $O\left(\frac{d}{\epsilon^2(1-\gamma)^2}\right)$ | $O(\gamma^k)$ |
| Factored | $O\left(\frac{k \cdot |\mathcal{S}_{\max}|}{\epsilon^2}\right)$ | $O(\gamma^k)$ |
| Neural | Problem-dependent | $O(1/\sqrt{k})$ (SGD) |

### Step 4: Inconclusive Case Handling

**Claim:** When no template matches, emit inconclusive certificate without false claims.

**Inconclusive Certificate:**
```python
K_WP_inc = {
    'status': 'INCONCLUSIVE',
    'failure_code': 'TEMPLATE_MISS',
    'signature': extracted_signature,
    'manual_override_hook': allow_custom_shaping,
    'message': "No efficient algorithm known. User may provide custom shaping."
}
```

**Examples of Inconclusive Cases:**
1. **POMDP:** Partial observability breaks Markov assumption
2. **Adversarial MDP:** No fixed optimal policy exists
3. **Non-ergodic:** Some states permanently unreachable
4. **Cryptographic:** Goal requires solving hard problem

**Soundness:** Emitting $K^{\text{inc}}$ means "unknown," not "impossible."

### Step 5: Continuation Criterion = Goal Reachability

**Claim:** The continuation criterion monitors whether the goal remains achievable.

**Definition:**
$$\text{cont}(s, t) \equiv V^*(s) > V_{\min}$$

**Interpretation:**
- $V^*(s) > V_{\min}$: Goal achievable from state $s$
- $V^*(s) = -\infty$: Goal unreachable (analogous to blowup)
- $V^*(s) < V_{\min}$: Below threshold, trigger intervention

**Monitoring Algorithm:**
```python
def monitor_continuation(trajectory, V_star, V_min):
    for s in trajectory:
        if V_star(s) < V_min:
            return ReachabilityFailure(s, V_star(s))
    return Reachable()
```

**Intervention (Surgery Analog):**
When continuation fails, options include:
1. **Goal relaxation:** Expand $G$ to include nearby states
2. **Reset:** Return to high-value initial state
3. **Subgoal insertion:** Introduce intermediate goals

---

## Connections to Classical Results

### 1. Potential-Based Reward Shaping (Ng et al. 1999)

**Theorem (Ng, Harada, Russell).** Let $\Phi: \mathcal{S} \to \mathbb{R}$ be any bounded real-valued function. Define the shaping reward:
$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

Then for any MDP $M = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$ and shaped MDP $M' = (\mathcal{S}, \mathcal{A}, P, R + F, \gamma)$:
$$\pi^*_{M'} = \pi^*_M$$

**Connection to FACT-Soft-WP:**

| Hypostructure | Reward Shaping |
|---------------|----------------|
| Soft certificates | Problem specifications |
| Template matching | MDP classification |
| WP derivation | Shaping potential derivation |
| $K_{\mathrm{WP}}^+$ certificate | Shaped reward $R' = R + F$ |
| Continuation criterion | Goal reachability |
| Blowup | Unreachable goal |

The FACT-Soft-WP theorem provides the mathematical foundation for automatic shaping potential derivation from high-level specifications.

### 2. Goal-Conditioned Reinforcement Learning

**Framework (Schaul et al. 2015, Andrychowicz et al. 2017).**
Learn a universal value function:
$$V(s, g) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r(s_t, g) \mid s_0 = s\right]$$

where $g \in \mathcal{G}$ is the goal specification.

**Connection to Soft WP:**
- Soft certificates specify the goal $g$
- Template matching identifies goal type (reachability, avoidance, temporal)
- Shaping potential is $\Phi(s) = V^*(s, g)$

**Hindsight Experience Replay (HER):**
Failed trajectories relabeled with achieved goals:
$$\{(s_t, a_t, s_{t+1}, g) : g = s_T\}$$

This corresponds to surgery in the hypostructure: when original goal unreachable, replace with achievable subgoal.

### 3. Inverse Reinforcement Learning

**Problem (Ng & Russell 2000, Abbeel & Ng 2004).**
Given expert demonstrations $\{(s_t, a_t)\}_{t=0}^T$, recover reward function $R(s, a)$.

**Connection to FACT-Soft-WP:**
- Expert demonstrations provide implicit soft certificates
- Template matching identifies reward structure
- Derived reward $R$ is the shaping potential gradient:
  $$R(s, a) \propto \nabla_a \Phi(s, a)$$

**Maximum Entropy IRL:**
$$\pi^*(a|s) = \frac{\exp(Q^*(s, a)/\alpha)}{\sum_{a'} \exp(Q^*(s, a')/\alpha)}$$

The temperature $\alpha$ corresponds to the dissipation rate $\mathfrak{D}$.

### 4. Hierarchical Reinforcement Learning

**Options Framework (Sutton et al. 1999).**
Options $o = (I_o, \pi_o, \beta_o)$ decompose tasks into subtasks:
- $I_o$: Initiation set (precondition)
- $\pi_o$: Option policy
- $\beta_o$: Termination condition (postcondition)

**Connection to Weakest Precondition:**
- $I_o = \text{wp}[\pi_o](G_o)$ where $G_o$ is option goal
- Template matching identifies option structure
- Shaping potential decomposes: $\Phi(s) = \sum_o \Phi_o(s)$

**MAXQ Decomposition (Dietterich 2000):**
$$Q(s, a) = V(s, a) + C(s, a)$$
- $V(s, a)$: Value of completing subtask
- $C(s, a)$: Completion function (context)

This mirrors the profile decomposition in concentration-compactness.

### 5. Lyapunov-Based Policy Synthesis

**Control Lyapunov Function (CLF):**
A function $V: \mathcal{S} \to \mathbb{R}_{\geq 0}$ is a CLF if:
$$\forall s \neq s_{\text{goal}}, \exists a: V(f(s, a)) < V(s)$$

**Connection to Energy-Dissipation:**

| CLF Property | Hypostructure | RL |
|--------------|---------------|-----|
| $V(s) \geq 0$ | $\Phi(x) \geq 0$ | $V^*(s) \leq V_{\max}$ |
| $V(s_{\text{goal}}) = 0$ | Equilibrium | Goal state |
| $V$ decreases | Dissipation | Policy improvement |

**Barrier Functions:**
$$B(s) \to \infty \text{ as } s \to \partial\mathcal{S}_{\text{safe}}$$

Barrier certificates ensure safety, analogous to boundary conditions in PDE well-posedness.

---

## Implementation Notes

### Practical Reward Shaping

**Algorithm: Automatic Shaping from Specifications**

```python
class SoftWPCompiler:
    """
    Compile soft specifications into reward shaping.
    Implements FACT-Soft-WP for RL.
    """

    def __init__(self, template_database):
        self.templates = template_database

    def extract_signature(self, env, goal_spec):
        """
        Extract problem signature from environment and goal.
        Analogous to Step 1 of FACT-Soft-WP proof.
        """
        signature = {
            'state_space': self._analyze_state_space(env),
            'action_space': self._analyze_action_space(env),
            'dynamics': self._analyze_dynamics(env),
            'goal_type': self._classify_goal(goal_spec),
            'discount': env.discount_factor
        }
        return signature

    def match_template(self, signature):
        """
        Match signature against template database.
        Returns (template_id, shaping_method) or INCONCLUSIVE.
        """
        for template in self.templates:
            if template.matches(signature):
                return template
        return InconclusiveTemplate(signature)

    def derive_potential(self, template, env, goal_spec):
        """
        Derive shaping potential from template.
        Analogous to Step 2 of FACT-Soft-WP proof.
        """
        if template.id == 'TABULAR_REACHABILITY':
            # Compute optimal value function via value iteration
            return self._value_iteration(env, goal_spec)

        elif template.id == 'LINEAR_GOAL_CONDITIONED':
            # Compute linear value function
            return self._linear_programming(env, goal_spec)

        elif template.id == 'FACTORED_DETERMINISTIC':
            # Decomposed computation
            return self._factored_vi(env, goal_spec)

        elif template.id == 'CONTINUOUS_DISCOUNTED':
            # Neural approximation
            return self._fitted_value_iteration(env, goal_spec)

        else:
            return None  # Inconclusive

    def construct_shaping(self, potential, gamma):
        """
        Construct Ng et al. shaping function.
        F(s, s') = gamma * Phi(s') - Phi(s)
        """
        def shaping_reward(s, s_prime):
            return gamma * potential(s_prime) - potential(s)
        return shaping_reward

    def compile(self, env, goal_spec):
        """
        Main compilation: soft specs -> shaping reward.
        Returns K_WP certificate.
        """
        # Step 1: Extract signature
        signature = self.extract_signature(env, goal_spec)

        # Step 2: Template matching
        template = self.match_template(signature)

        if template.is_inconclusive():
            return self._emit_inconclusive(template, signature)

        # Step 3: Derive potential
        potential = self.derive_potential(template, env, goal_spec)

        # Step 4: Construct shaping
        shaping = self.construct_shaping(potential, env.gamma)

        # Step 5: Construct certificate
        return WPCertificate(
            template_id=template.id,
            shaping_potential=potential,
            shaping_reward=shaping,
            goal_spec=goal_spec,
            continuation_criterion=self._derive_continuation(potential, goal_spec)
        )
```

### Continuation Monitoring

```python
class ContinuationMonitor:
    """
    Monitor goal reachability during training.
    Analogous to blowup detection in PDE.
    """

    def __init__(self, potential, v_min, goal_spec):
        self.potential = potential
        self.v_min = v_min
        self.goal_spec = goal_spec

    def check(self, state):
        """Check if goal remains reachable from state."""
        v = self.potential(state)

        if v < self.v_min:
            return UnreachableWarning(state, v, self.v_min)

        if self.goal_spec.is_terminal(state):
            return GoalReached(state)

        return Reachable(state, v)

    def suggest_intervention(self, failure):
        """Suggest intervention when continuation fails (surgery analog)."""
        if failure.v < -1e10:
            # Hopeless: reset to initial state
            return ResetIntervention()

        elif failure.v < self.v_min:
            # Suboptimal: insert subgoal
            subgoal = self._find_nearest_reachable(failure.state)
            return SubgoalIntervention(subgoal)

        else:
            # Continue with caution
            return ContinueWithMonitoring()
```

### Template Database

```python
# Standard templates (analogous to PDE well-posedness templates)

TEMPLATE_DATABASE = [
    Template(
        id='TABULAR_REACHABILITY',
        signature_pattern={
            'state_space': 'discrete',
            'action_space': 'discrete',
            'size': lambda s, a: s * a < 1e6,
            'goal_type': 'reachability'
        },
        method='value_iteration',
        sample_complexity='O(|S||A| / (eps^2 * (1-gamma)^3))',
        literature='Sutton & Barto 2018'
    ),

    Template(
        id='LINEAR_GOAL_CONDITIONED',
        signature_pattern={
            'representation': 'linear',
            'dynamics': 'linear_gaussian',
            'goal_type': 'reachability'
        },
        method='lspi',  # Least-Squares Policy Iteration
        sample_complexity='O(d / (eps^2 * (1-gamma)^2))',
        literature='Lagoudakis & Parr 2003'
    ),

    Template(
        id='FACTORED_DETERMINISTIC',
        signature_pattern={
            'structure': 'factored',
            'dynamics': 'deterministic',
            'dependencies': 'sparse'
        },
        method='factored_vi',
        sample_complexity='O(k * |S_max| / eps^2)',
        literature='Boutilier et al. 1999'
    ),

    Template(
        id='CONTINUOUS_DISCOUNTED',
        signature_pattern={
            'state_space': 'continuous',
            'action_space': 'continuous',
            'discount': lambda g: g < 1.0
        },
        method='fitted_q_iteration',
        sample_complexity='problem_dependent',
        literature='Ernst et al. 2005'
    )
]
```

### Integration with Policy Gradient

```python
def shaped_policy_gradient(env, policy, potential, gamma, n_episodes):
    """
    Policy gradient with automatic shaping.
    The shaping accelerates learning without changing optimal policy.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for episode in range(n_episodes):
        trajectory = collect_trajectory(env, policy)

        # Compute shaped returns
        shaped_returns = []
        for t, (s, a, r, s_next) in enumerate(trajectory):
            # Ng et al. shaping
            shaping = gamma * potential(s_next) - potential(s)
            shaped_r = r + shaping
            shaped_returns.append(shaped_r)

        # Compute advantages
        advantages = compute_gae(shaped_returns, potential, gamma)

        # Policy gradient update
        loss = -torch.mean(advantages * policy.log_prob(trajectory))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Monitor continuation (goal reachability)
        if not check_continuation(trajectory, potential):
            print(f"Warning: Goal may be unreachable at episode {episode}")
            # Consider intervention (subgoal, reset, etc.)
```

---

## Literature

1. **Ng, A. Y., Harada, D., & Russell, S. (1999).** "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping." *ICML*. *Foundational paper on potential-based reward shaping.*

2. **Sutton, R. S. & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. *Comprehensive RL textbook covering value functions and policy optimization.*

3. **Schaul, T., Horgan, D., Gregor, K., & Silver, D. (2015).** "Universal Value Function Approximators." *ICML*. *Goal-conditioned RL with universal value functions.*

4. **Andrychowicz, M. et al. (2017).** "Hindsight Experience Replay." *NeurIPS*. *Goal relabeling for sample-efficient goal-conditioned learning.*

5. **Abbeel, P. & Ng, A. Y. (2004).** "Apprenticeship Learning via Inverse Reinforcement Learning." *ICML*. *Learning reward functions from demonstrations.*

6. **Sutton, R. S., Precup, D., & Singh, S. (1999).** "Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning." *Artificial Intelligence*. *Options framework for hierarchical RL.*

7. **Dietterich, T. G. (2000).** "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition." *JAIR*. *Value function decomposition for hierarchical tasks.*

8. **Lagoudakis, M. G. & Parr, R. (2003).** "Least-Squares Policy Iteration." *JMLR*. *Linear function approximation for policy iteration.*

9. **Boutilier, C., Dearden, R., & Goldszmidt, M. (2000).** "Stochastic Dynamic Programming with Factored Representations." *Artificial Intelligence*. *Exploiting structure in MDPs.*

10. **Ernst, D., Geurts, P., & Wehenkel, L. (2005).** "Tree-Based Batch Mode Reinforcement Learning." *JMLR*. *Fitted Q-iteration for continuous control.*

11. **Berkenkamp, F., Turchetta, M., Schoellig, A., & Krause, A. (2017).** "Safe Model-Based Reinforcement Learning with Stability Guarantees." *NeurIPS*. *Lyapunov-based safe RL.*

12. **Ames, A. D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath, K., & Tabuada, P. (2019).** "Control Barrier Functions: Theory and Applications." *ECC*. *Barrier certificates for safety.*

---

## Summary

The FACT-Soft-WP theorem, translated to AI/RL/ML, establishes that:

1. **Automatic Shaping from Specifications:** High-level problem specifications (goal, dynamics, discount) can be automatically compiled into reward shaping functions that accelerate learning while preserving optimal policies.

2. **Template Matching = Problem Classification:** Just as PDE well-posedness uses template matching (parabolic, wave, Schrodinger), RL problems can be classified (tabular, linear, factored, neural) to select appropriate algorithms.

3. **Potential-Based Shaping = Soft WP:** The Ng et al. shaping function $F(s, s') = \gamma\Phi(s') - \Phi(s)$ is the RL analog of the soft weakest precondition, providing guidance toward the goal without changing optimal behavior.

4. **Continuation = Goal Reachability:** The continuation criterion in WP certificates corresponds to goal reachability in RL. When $V^*(s) < V_{\min}$, the goal may be unreachable (analogous to blowup), triggering intervention strategies.

5. **Inconclusive Cases:** When no template matches (e.g., POMDPs, adversarial settings), the framework honestly reports "inconclusive" rather than making false claims, allowing manual specification of shaping.

This translation reveals that the hypostructure's FACT-Soft-WP provides the mathematical foundation for automatic reward engineering in reinforcement learning, unifying goal-conditioned RL, hierarchical RL, and inverse RL under a single framework of "soft specification compilation."
