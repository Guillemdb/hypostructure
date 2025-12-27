# FACT-SoftAttr: Multi-Agent Equilibrium Existence

## AI/RL/ML Statement

### Original Statement (Hypostructure)
*Reference: mt-fact-soft-attr*

**[FACT-SoftAttr] Soft→Attr Compilation:** Global attractor existence is derived from soft interfaces for dissipative systems. Given soft hypotheses $K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{TB}_\pi}^+$, the theorem produces the attractor certificate $K_{\mathrm{Attr}}^+$.

---

## AI/RL/ML Formulation

### Setup

Consider a multi-agent learning system where:

- **State space:** Joint policy space $\Pi = \Pi_1 \times \Pi_2 \times \cdots \times \Pi_n$ for $n$ agents
- **Height/Energy (Value function):** Potential function $\Phi(\pi) = -\sum_i V^{\pi_i}(s_0)$ or Nash gap $\mathcal{G}(\pi) = \sum_i \max_{\pi_i'} [V^{\pi_i'}(s) - V^{\pi_i}(s)]$
- **Dissipation (Policy):** Learning dynamics $\pi_{t+1} = \mathcal{T}(\pi_t)$ (e.g., gradient descent, fictitious play, replicator dynamics)
- **Soft attractor:** Nash equilibrium, correlated equilibrium, or PPAD fixed point
- **Attraction basin:** Convergence region of the learning algorithm

The theorem establishes that dissipative multi-agent learning systems with compact strategy spaces converge to equilibrium sets.

### Statement (AI/RL/ML Version)

**Theorem (Multi-Agent Equilibrium Existence).** Let $\mathcal{G} = (N, \{\Pi_i\}, \{u_i\})$ be a multi-agent game with $n$ players, strategy spaces $\Pi_i$, and utility functions $u_i$. Let $\mathcal{T}: \Pi \to \Pi$ be a learning dynamics operator. Suppose:

1. **Dissipation ($D_E^+$):** There exists a potential function $\Phi: \Pi \to \mathbb{R}_{\geq 0}$ such that
   $$\Phi(\mathcal{T}(\pi)) + \int_0^1 \mathfrak{D}(\mathcal{T}_s(\pi)) \, ds \leq \Phi(\pi)$$
   where $\mathfrak{D} \geq 0$ is the improvement rate (gradient magnitude, regret reduction, etc.)

2. **Compactness ($C_\mu^+$):** The strategy space $\Pi$ is compact, or sublevel sets $\{\pi : \Phi(\pi) \leq R\}$ are precompact modulo symmetries (permutation of equivalent agents)

3. **Continuity ($\mathrm{TB}_\pi^+$):** The learning operator $\mathcal{T}$ is continuous (or at least upper hemicontinuous for best-response dynamics)

**Then:** There exists a global attractor $\mathcal{A} \subset \Pi$ satisfying:
- **Compactness:** $\mathcal{A}$ is compact
- **Invariance:** $\mathcal{T}(\mathcal{A}) = \mathcal{A}$ (equilibrium set is stable under learning)
- **Attraction:** For all bounded $B \subset \Pi$: $\lim_{t \to \infty} \mathrm{dist}(\mathcal{T}^t(B), \mathcal{A}) = 0$

**Corollary (Nash Equilibrium Containment):** The attractor $\mathcal{A}$ contains all Nash equilibria of $\mathcal{G}$. For potential games, $\mathcal{A}$ equals the set of Nash equilibria.

---

## Terminology Translation Table

| Hypostructure Term | AI/RL/ML Equivalent |
|--------------------|---------------------|
| Semigroup $S_t: \mathcal{X} \to \mathcal{X}$ | Learning dynamics $\mathcal{T}^t: \Pi \to \Pi$ (policy iteration, gradient descent, replicator) |
| Energy functional $\Phi: \mathcal{X} \to \mathbb{R}_{\geq 0}$ | Potential function, Nash gap $\mathcal{G}(\pi)$, or negative social welfare $-\sum_i u_i(\pi)$ |
| Dissipation density $\mathfrak{D}(x)$ | Regret reduction rate, gradient magnitude $\|\nabla_\pi \mathcal{L}\|$, exploitability decrease |
| Dissipation certificate $K_{D_E}^+$ | No-regret learning guarantee, potential game structure, or gradient dominance |
| Compactness $K_{C_\mu}^+$ | Compact strategy spaces (simplex $\Delta^n$, bounded action sets) |
| Continuity $K_{\mathrm{TB}_\pi}^+$ | Continuous best-response, Lipschitz policy gradients |
| Global attractor $\mathcal{A}$ | Equilibrium set (Nash, correlated, coarse correlated, or mean-field equilibrium) |
| Absorbing set $B_R$ | Bounded rationality region, $\epsilon$-Nash neighborhood |
| Forward invariance | Stability of equilibrium set under learning perturbations |
| Backward invariance | Every equilibrium can be reached from some initialization |
| Omega-limit set $\omega(x)$ | Long-run behavior of learning trajectory, limit cycle or fixed point |
| Symmetry group $G$ | Agent permutation symmetry, type symmetry in anonymous games |
| Finite-energy state | Bounded regret, finite potential |
| Asymptotic compactness | Convergence of learning trajectories (no divergence) |
| Profile decomposition | Equilibrium selection, hierarchical game decomposition |

---

## Proof Sketch

### Step 1: Absorbing Set via Regret/Potential Bounds

**Claim:** There exists a bounded set $B_R \subset \Pi$ that absorbs all learning trajectories.

**Proof (Potential Game Case):**

For a potential game with potential $\Phi: \Pi \to \mathbb{R}$, any improvement step satisfies:
$$\Phi(\pi_{t+1}) \leq \Phi(\pi_t) - \eta \cdot \text{improvement}_t$$

Since $\Phi$ is bounded below (on compact $\Pi$), the trajectory enters a region where improvements are small:
$$B_R := \{\pi : \Phi(\pi) \leq \Phi(\pi_0)\}$$

is forward-invariant and absorbing.

**Proof (No-Regret Learning Case):**

For no-regret learners, the average regret satisfies:
$$\frac{1}{T} \sum_{t=1}^T \text{Regret}_t(\pi_t) \leq O(1/\sqrt{T}) \to 0$$

This implies trajectories are eventually absorbed into an $\epsilon$-equilibrium region:
$$B_\epsilon := \{\pi : \text{Exploitability}(\pi) \leq \epsilon\}$$

### Step 2: Asymptotic Compactness via Strategy Space Compactness

**Claim:** Learning trajectories have convergent subsequences.

**Proof:**

By compactness of $\Pi$ (or precompactness of sublevel sets), any sequence $\{\pi_t\}_{t \geq 0}$ has a convergent subsequence $\pi_{t_k} \to \pi^*$.

For continuous dynamics $\mathcal{T}$, the omega-limit set:
$$\omega(\pi_0) := \bigcap_{s \geq 0} \overline{\{\pi_t : t \geq s\}}$$
is non-empty and compact.

**Connection to PPAD:** The existence of fixed points in continuous maps on compact convex sets is guaranteed by Brouwer's theorem. This underlies the PPAD complexity class for computing Nash equilibria.

### Step 3: Attractor Construction

**Definition:** The global attractor is:
$$\mathcal{A} := \omega(B_R) = \bigcap_{s \geq 0} \overline{\bigcup_{t \geq s} \mathcal{T}^t(B_R)}$$

**Properties:**

1. **Compactness:** $\mathcal{A} \subset B_R$ which is compact.

2. **Forward Invariance ($\mathcal{T}(\mathcal{A}) \subset \mathcal{A}$):** If $\pi^* \in \mathcal{A}$, then $\pi^* = \lim_{k} \pi_{t_k}$ for some trajectory. By continuity, $\mathcal{T}(\pi^*) = \lim_k \mathcal{T}(\pi_{t_k}) = \lim_k \pi_{t_k+1} \in \mathcal{A}$.

3. **Backward Invariance ($\mathcal{A} \subset \mathcal{T}(\mathcal{A})$):** For $\pi^* \in \mathcal{A}$, there exist $\pi_{t_k} \to \pi^*$ with $t_k \to \infty$. The sequence $\{\pi_{t_k-1}\}$ has a convergent subsequence (by compactness) to some $\pi' \in \mathcal{A}$, and $\mathcal{T}(\pi') = \pi^*$.

4. **Attraction:** By asymptotic compactness, all trajectories converge to $\mathcal{A}$.

### Step 4: Equilibrium Characterization

**Key Observation (Dissipation Vanishes on Attractor):**

On the attractor, the dissipation must vanish:
$$\mathfrak{D}(\pi) = 0 \quad \text{for all } \pi \in \mathcal{A}$$

**Proof:** If $\mathfrak{D}(\pi) > 0$ for some $\pi \in \mathcal{A}$, then $\Phi(\mathcal{T}(\pi)) < \Phi(\pi)$. By invariance, $\mathcal{T}(\pi) \in \mathcal{A}$. Iterating, $\Phi(\mathcal{T}^k(\pi)) < \Phi(\pi) - k\delta \to -\infty$, contradicting $\Phi \geq 0$.

**Consequence:** For learning dynamics where $\mathfrak{D}(\pi) = 0$ implies equilibrium (e.g., gradient descent on potential, no-regret dynamics), the attractor consists of equilibria:
$$\mathcal{A} \subseteq \{\pi : \pi \text{ is a Nash equilibrium}\}$$

---

## Connections

### 1. PPAD Complexity and Fixed-Point Computation

**PPAD (Polynomial Parity Arguments on Directed graphs):** The complexity class containing problems equivalent to finding Brouwer fixed points.

**Connection to FACT-SoftAttr:**
- The theorem guarantees *existence* of fixed points (attractor)
- PPAD hardness shows *finding* these fixed points is computationally hard in general
- The attractor $\mathcal{A}$ is the "target" that PPAD algorithms seek

**Key Results:**
- Nash equilibrium in 2-player games is PPAD-complete (Daskalakis-Goldberg-Papadimitriou 2009)
- Gradient descent on smooth games converges to $\mathcal{A}$ but may take exponential time (Hirsch-Papadimitriou-Vavasis 1989)

| FACT-SoftAttr Concept | PPAD Analog |
|----------------------|-------------|
| Attractor existence | Brouwer fixed point theorem |
| Compactness $K_{C_\mu}^+$ | Convex compact domain |
| Continuity $K_{\mathrm{TB}_\pi}^+$ | Continuous displacement function |
| Dissipation $K_{D_E}^+$ | Path-following (homotopy) methods |

### 2. Nash Equilibrium Theory

**Nash's Theorem (1950):** Every finite game has a mixed-strategy Nash equilibrium.

**Connection to FACT-SoftAttr:**
- Mixed strategies live in the simplex $\Delta^n$ (compact)
- Best-response is upper hemicontinuous
- Nash equilibria are exactly the fixed points of best-response dynamics

**Attractor Interpretation:**
$$\mathcal{A} = \{\pi \in \Pi : \pi_i \in BR_i(\pi_{-i}) \text{ for all } i\}$$

**Refinements:**
- Strict equilibria are isolated points in $\mathcal{A}$
- Evolutionarily stable strategies are asymptotically stable within $\mathcal{A}$
- Trembling-hand perfect equilibria are robust to perturbations

### 3. Multi-Agent Reinforcement Learning (MARL)

**Challenge:** In MARL, agents learn simultaneously, creating non-stationary environments.

**FACT-SoftAttr Perspective:**
- Joint policy space $\Pi = \Pi_1 \times \cdots \times \Pi_n$
- Each agent's learning is a component of the semigroup $\mathcal{T}$
- Convergence to $\mathcal{A}$ means agents reach a stable joint policy

**Convergence Results:**
- **Self-play in zero-sum games:** Converges to minimax equilibrium (Brown 1951)
- **Fictitious play in potential games:** Converges to Nash (Monderer-Shapley 1996)
- **Policy gradient in Markov games:** Converges under certain conditions (Mazumdar et al. 2020)

**Algorithm Templates:**

```python
def multi_agent_learning(game, T_max, learning_rate):
    """
    Generic multi-agent learning with attractor convergence.

    Args:
        game: Multi-agent game (N, Pi_i, u_i)
        T_max: Maximum iterations
        learning_rate: Step size eta

    Returns:
        Joint policy in attractor A
    """
    # Initialize in compact strategy space
    pi = [initialize_policy(Pi_i) for i in range(game.n_agents)]

    for t in range(T_max):
        # Compute gradients/best-responses for each agent
        gradients = [compute_gradient(game, pi, i) for i in range(game.n_agents)]

        # Update (dissipation step)
        for i in range(game.n_agents):
            pi[i] = project_to_simplex(pi[i] + learning_rate * gradients[i])

        # Check convergence (exploitability / Nash gap)
        if exploitability(game, pi) < epsilon:
            break  # Reached attractor

    return pi
```

### 4. Mean-Field Games

**Setting:** Large population games where individual impact vanishes.

**Mean-Field Limit:**
- State: Population distribution $\mu \in \mathcal{P}(\mathcal{S})$
- Dynamics: Fokker-Planck equation for population evolution
- Equilibrium: Mean-field Nash equilibrium (fixed point of McKean-Vlasov dynamics)

**FACT-SoftAttr Application:**
- Population distribution space is compact (weak topology on probability measures)
- Mean-field dynamics are continuous (under regularity conditions)
- Dissipation: Free energy $\Phi(\mu) = \mathbb{E}[u] + \tau \text{Entropy}(\mu)$ decreases

**Theorem (Mean-Field Attractor):** Under monotonicity conditions (Lasry-Lions 2007), the mean-field dynamics have a unique global attractor $\mathcal{A} = \{\mu^*\}$ consisting of the mean-field equilibrium.

**Key Papers:**
- Lasry-Lions (2007): Mean field games
- Cardaliaguet et al. (2019): Master equation and mean field games
- Carmona-Delarue (2018): Probabilistic theory of mean field games

---

## Implementation Notes

### Verifying Soft Interface Certificates

**Certificate $K_{D_E}^+$ (Dissipation):**

```python
def verify_dissipation(game, dynamics, pi, delta_t=0.01):
    """
    Verify that Phi(T(pi)) <= Phi(pi) - integral of dissipation.
    """
    # Compute potential/Nash gap before
    Phi_before = compute_potential(game, pi)

    # Apply dynamics
    pi_next = dynamics(pi)

    # Compute potential after
    Phi_after = compute_potential(game, pi_next)

    # Estimate dissipation integral
    dissipation = compute_exploitability(game, pi)

    # Verify inequality
    return Phi_after + delta_t * dissipation <= Phi_before + 1e-6
```

**Certificate $K_{C_\mu}^+$ (Compactness):**

```python
def verify_compactness(strategy_space):
    """
    Verify strategy space is compact.
    """
    if isinstance(strategy_space, Simplex):
        return True  # Simplex is always compact
    elif isinstance(strategy_space, BoundedBox):
        return strategy_space.is_closed()
    elif isinstance(strategy_space, FunctionSpace):
        # Need RKHS or finite-dimensional approximation
        return strategy_space.dimension < float('inf')
    return False
```

**Certificate $K_{\mathrm{TB}_\pi}^+$ (Continuity):**

```python
def verify_continuity(dynamics, pi, epsilon=1e-4):
    """
    Verify dynamics are Lipschitz continuous.
    """
    pi_perturbed = pi + epsilon * np.random.randn(*pi.shape)
    pi_perturbed = project_to_simplex(pi_perturbed)

    output_orig = dynamics(pi)
    output_pert = dynamics(pi_perturbed)

    # Check Lipschitz bound
    input_dist = np.linalg.norm(pi - pi_perturbed)
    output_dist = np.linalg.norm(output_orig - output_pert)

    lipschitz_const = output_dist / (input_dist + 1e-10)
    return lipschitz_const < 100  # Reasonable Lipschitz constant
```

### Practical Attractor Computation

**Method 1: Long-Run Simulation**

```python
def compute_attractor_simulation(game, dynamics, n_samples=100, T=10000):
    """
    Estimate attractor via long-run simulation from random starts.
    """
    attractor_samples = []

    for _ in range(n_samples):
        pi = random_initialization(game)
        for t in range(T):
            pi = dynamics(pi)
        attractor_samples.append(pi)

    # Cluster to identify attractor structure
    attractor = cluster_policies(attractor_samples)
    return attractor
```

**Method 2: Fixed-Point Iteration**

```python
def compute_attractor_fixedpoint(game, dynamics, pi_init, tol=1e-6, max_iter=10000):
    """
    Compute attractor via fixed-point iteration with convergence check.
    """
    pi = pi_init
    for t in range(max_iter):
        pi_next = dynamics(pi)

        # Check for fixed point (in attractor)
        if np.linalg.norm(pi_next - pi) < tol:
            return pi_next, True

        pi = pi_next

    return pi, False  # May be in limit cycle within attractor
```

### Monitoring Convergence to Attractor

```python
class AttractorMonitor:
    """Monitor convergence to global attractor in multi-agent learning."""

    def __init__(self, game):
        self.game = game
        self.history = []

    def log(self, pi, t):
        exploitability = compute_exploitability(self.game, pi)
        potential = compute_potential(self.game, pi)
        self.history.append({
            't': t,
            'exploitability': exploitability,
            'potential': potential,
            'policy': pi.copy()
        })

    def check_absorption(self, R):
        """Check if trajectory is absorbed into B_R."""
        if len(self.history) < 10:
            return False
        recent = [h['potential'] for h in self.history[-10:]]
        return all(p <= R for p in recent)

    def estimate_attractor_distance(self, attractor_samples):
        """Estimate distance to known attractor samples."""
        current = self.history[-1]['policy']
        distances = [np.linalg.norm(current - a) for a in attractor_samples]
        return min(distances)
```

---

## Literature

1. **Nash, J. (1950).** "Equilibrium Points in N-Person Games." *PNAS*. *Existence of Nash equilibrium via Brouwer fixed point.*

2. **Daskalakis, C., Goldberg, P. W., & Papadimitriou, C. H. (2009).** "The Complexity of Computing a Nash Equilibrium." *SIAM J. Comput.* *PPAD-completeness of Nash equilibrium.*

3. **Monderer, D. & Shapley, L. S. (1996).** "Potential Games." *Games and Economic Behavior.* *Convergence of learning in potential games.*

4. **Lasry, J.-M. & Lions, P.-L. (2007).** "Mean Field Games." *Japanese Journal of Mathematics.* *Mean-field game theory and equilibrium existence.*

5. **Cesa-Bianchi, N. & Lugosi, G. (2006).** *Prediction, Learning, and Games.* Cambridge. *No-regret learning and convergence to correlated equilibrium.*

6. **Fudenberg, D. & Levine, D. K. (1998).** *The Theory of Learning in Games.* MIT Press. *Convergence of learning dynamics in games.*

7. **Rosen, J. B. (1965).** "Existence and Uniqueness of Equilibrium Points for Concave N-Person Games." *Econometrica.* *Uniqueness conditions for Nash equilibria.*

8. **Hofbauer, J. & Sigmund, K. (1998).** *Evolutionary Games and Population Dynamics.* Cambridge. *Replicator dynamics and evolutionary stable strategies.*

9. **Mazumdar, E., Ratliff, L. J., & Sastry, S. S. (2020).** "On Gradient-Based Learning in Continuous Games." *SIAM J. Math. Data Sci.* *Convergence of policy gradient in games.*

10. **Papadimitriou, C. H. (1994).** "On the Complexity of the Parity Argument and Other Inefficient Proofs of Existence." *JCSS.* *PPAD complexity class.*

11. **Temam, R. (1997).** *Infinite-Dimensional Dynamical Systems in Mechanics and Physics* (2nd ed.). Springer. *Global attractor theory (mathematical foundation).*

12. **Cardaliaguet, P., Delarue, F., Lasry, J.-M., & Lions, P.-L. (2019).** *The Master Equation and the Convergence Problem in Mean Field Games.* Princeton. *Rigorous mean-field limit.*

---

## Summary

The FACT-SoftAttr theorem, translated to AI/RL/ML, establishes that:

1. **Multi-agent learning systems with dissipation converge:** When learning dynamics reduce a potential function (Nash gap, regret), trajectories are absorbed into bounded regions and converge to an attractor.

2. **Equilibria form a global attractor:** The attractor $\mathcal{A}$ is compact, invariant under learning dynamics, and attracts all bounded initializations. For games with proper dissipation structure, $\mathcal{A}$ contains (or equals) the Nash equilibria.

3. **Compactness is essential:** The compactness of strategy spaces (mixed strategies on simplex, bounded policy parameters) enables the existence of the attractor via Brouwer-type fixed-point arguments.

4. **PPAD connection:** While FACT-SoftAttr guarantees existence of equilibria, computing them is PPAD-hard in general. The theorem characterizes *what* exists; PPAD complexity characterizes *how hard* it is to find.

5. **Mean-field games extend the framework:** In large-population limits, the attractor becomes a mean-field equilibrium, with population distributions replacing individual policies.

This translation reveals that the hypostructure's global attractor construction provides the mathematical foundation for equilibrium existence in game theory and multi-agent reinforcement learning, unifying Nash equilibrium theory, no-regret learning, and mean-field games under a single dynamical-systems perspective.
