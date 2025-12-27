---
title: "KRNL-Equivariance - AI/RL/ML Translation"
---

# KRNL-Equivariance: Equivariant Learning and Symmetry Preservation

## Original Theorem (Hypostructure Context)

The KRNL-Equivariance theorem establishes that when a system distribution is invariant under a symmetry group G, and the parametrization respects this symmetry, then:
1. Risk minimizers lie in complete G-orbits
2. Gradient flow preserves G-orbits
3. Learned hypostructures inherit all symmetries of the input distribution

**Core insight:** Symmetry in the data distribution forces symmetry in optimal learned representations.

---

## AI/RL/ML Statement

**Theorem (Equivariant Learning Principle):** Let $G$ be a compact symmetry group acting on state space $\mathcal{S}$ and policy/network parameter space $\Theta$. Suppose:

1. **(G-Invariant Environment):** The state distribution $\mu(s)$ and reward function $R(s,a)$ are $G$-invariant:
   $$\mu(g \cdot s) = \mu(s), \quad R(g \cdot s, g \cdot a) = R(s, a)$$

2. **(Equivariant Architecture):** The policy network $\pi_\theta$ satisfies:
   $$\pi_{g \cdot \theta}(g \cdot a | g \cdot s) = \pi_\theta(a | s)$$

3. **(Value Equivariance):** The value function transforms correctly:
   $$V^{\pi}(g \cdot s) = V^{\pi}(s)$$

Then:
- **Optimal policies form G-orbits:** If $\theta^*$ is optimal, then $g \cdot \theta^*$ is optimal for all $g \in G$
- **Training preserves equivariance:** If $\theta_0$ is $G$-equivariant, then $\theta_t$ remains $G$-equivariant throughout gradient-based training
- **Learned representations inherit symmetry:** Features, value estimates, and certificates respect $G$-invariance

**Informal:** For symmetric MDPs, equivariant networks are optimal without loss of representational power.

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Equivalent |
|----------------------|---------------------|
| Compact Lie group $G$ | Symmetry group (rotations $SO(n)$, translations $\mathbb{R}^d$, permutations $S_n$) |
| System distribution $\mathcal{S}$ | State/observation distribution $\mu(s)$ |
| $G$-covariant distribution | Symmetric environment/MDP |
| Parameter space $\Theta$ | Neural network weights |
| Equivariant parametrization $\mathcal{H}_\Theta$ | Equivariant neural network: $f(g \cdot x) = g \cdot f(x)$ |
| Risk functional $R(\Theta)$ | Expected loss / negative expected return $\mathbb{E}[-R(\tau)]$ |
| Gradient flow $\dot{\Theta} = -\nabla R$ | Gradient descent / policy gradient |
| Risk minimizer $\widehat{\Theta}$ | Optimal policy parameters $\theta^*$ |
| Hypostructure $\mathcal{H}_\Theta(S)$ | Learned representation / feature map |
| Certificate $K_{A,S}^{(\Theta)}$ | Value function $V(s)$, advantage $A(s,a)$, Q-values |
| Height $\Phi$ | Value function $V(s)$ (Lyapunov-like quantity) |
| Dissipation $\mathfrak{D}$ | Policy $\pi(a|s)$ (action selection mechanism) |
| $G$-orbit $G \cdot \Theta$ | Equivalence class of weights under symmetry transforms |
| Haar measure $\mu_G$ | Uniform distribution over group elements |
| Defect-level equivariance | Layer-wise equivariance constraints |

---

## Proof Sketch

### Setup: Equivariant Reinforcement Learning

**Definition (G-Symmetric MDP):** An MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$ is $G$-symmetric if:
- State space $\mathcal{S}$ admits $G$-action: $G \times \mathcal{S} \to \mathcal{S}$
- Action space $\mathcal{A}$ admits compatible $G$-action
- Dynamics are equivariant: $P(g \cdot s' | g \cdot s, g \cdot a) = P(s' | s, a)$
- Rewards are invariant: $R(g \cdot s, g \cdot a) = R(s, a)$

**Definition (Equivariant Policy Network):** A neural network $f_\theta: \mathcal{S} \to \mathcal{A}$ is $G$-equivariant if:
$$f_\theta(g \cdot s) = g \cdot f_\theta(s) \quad \forall g \in G, s \in \mathcal{S}$$

For stochastic policies $\pi_\theta(a|s)$:
$$\pi_\theta(g \cdot a | g \cdot s) = \pi_\theta(a | s)$$

**Definition (G-Invariant Value Function):** The value function is $G$-invariant:
$$V^\pi(g \cdot s) = V^\pi(s) \quad \forall g \in G$$

---

### Step 1: Expected Return Invariance

**Claim:** For $G$-symmetric MDPs, the expected return $J(\theta)$ is $G$-invariant.

**Proof:**

Define the expected return:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^\infty \gamma^t R(s_t, a_t)\right]$$

For any $g \in G$, consider the transformed policy $\pi_{g \cdot \theta}$. By the equivariant parametrization assumption:
$$\pi_{g \cdot \theta}(a | s) = \pi_\theta(g^{-1} \cdot a | g^{-1} \cdot s)$$

The trajectory distribution under $\pi_{g \cdot \theta}$ starting from $g \cdot s_0$ is related to trajectories under $\pi_\theta$ starting from $s_0$ by:
$$\tau^{g \cdot \theta} \sim (g \cdot s_0, g \cdot a_0, g \cdot s_1, \ldots)$$

By $G$-invariance of rewards:
$$R(g \cdot s_t, g \cdot a_t) = R(s_t, a_t)$$

By $G$-invariance of the initial state distribution $\mu_0(g \cdot s) = \mu_0(s)$:
$$J(g \cdot \theta) = \mathbb{E}_{s_0 \sim \mu_0}\mathbb{E}_{\tau^{g \cdot \theta}}\left[\sum_{t=0}^\infty \gamma^t R(s_t, a_t)\right] = J(\theta)$$

**RL interpretation:** Rotating or translating the policy parameters does not change expected return in symmetric environments.

---

### Step 2: Policy Gradient Equivariance

**Claim:** The policy gradient is $G$-equivariant: $\nabla_\theta J(g \cdot \theta) = g \cdot \nabla_\theta J(\theta)$.

**Proof:**

By the chain rule applied to $J(g \cdot \theta) = J(\theta)$:
$$\nabla_\theta J(g \cdot \theta) \cdot \frac{\partial (g \cdot \theta)}{\partial \theta} = \nabla_\theta J(\theta)$$

For linear group actions where $g \cdot \theta = \rho(g) \theta$ with representation $\rho$:
$$\nabla J(g \cdot \theta) = \rho(g)^{-T} \nabla J(\theta)$$

For orthogonal representations (rotations, permutations): $\rho(g)^{-T} = \rho(g)$, so:
$$\nabla J(g \cdot \theta) = g \cdot \nabla J(\theta)$$

**Consequence for Policy Gradient Methods:**

Standard policy gradient update:
$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)$$

If $\theta_0 = g \cdot \theta_0$ (initialized symmetrically), then:
$$\theta_{t+1} = g \cdot \theta_t + \alpha \cdot g \cdot \nabla J(\theta_t) = g \cdot (\theta_t + \alpha \nabla J(\theta_t)) = g \cdot \theta_{t+1}$$

**PPO/TRPO interpretation:** Trust region methods with symmetric objectives preserve equivariance throughout optimization.

---

### Step 3: Optimal Policies Form Symmetry Orbits

**Claim:** The set of optimal policies is a union of complete $G$-orbits.

**Proof:**

Let $\theta^*$ be an optimal policy achieving maximum return:
$$J(\theta^*) = \max_\theta J(\theta)$$

By Step 1 (return invariance):
$$J(g \cdot \theta^*) = J(\theta^*) = \max_\theta J(\theta)$$

Therefore, every element of the orbit $G \cdot \theta^* = \{g \cdot \theta^* : g \in G\}$ is also optimal.

**Practical implication:** Multiple optimal policies exist related by symmetry transforms. Training may converge to any element of this orbit.

---

### Step 4: Value Function Symmetry Inheritance

**Claim:** For optimal equivariant policies, $V^*(s)$ is $G$-invariant.

**Proof:**

The Bellman optimality equation:
$$V^*(s) = \max_a \left[R(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s,a)}[V^*(s')]\right]$$

Apply $g$ to both sides. By $G$-invariance of $R$ and $P$:
\begin{align*}
V^*(g \cdot s) &= \max_{a} \left[R(g \cdot s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|g \cdot s, a)}[V^*(s')]\right] \\
&= \max_{a} \left[R(g \cdot s, g \cdot (g^{-1} \cdot a)) + \gamma \mathbb{E}_{g \cdot s'' \sim P(\cdot|g \cdot s, g \cdot (g^{-1} \cdot a))}[V^*(g \cdot s'')]\right]
\end{align*}

By equivariance of dynamics $P(g \cdot s' | g \cdot s, g \cdot a) = P(s' | s, a)$ and invariance of $R$:
$$V^*(g \cdot s) = \max_{a'} \left[R(s, a') + \gamma \mathbb{E}_{s'' \sim P(\cdot|s, a')}[V^*(g \cdot s'')]\right]$$

By induction on the Bellman backup (value iteration), $V^*(g \cdot s) = V^*(s)$.

**Deep RL interpretation:** Value networks should be designed as $G$-invariant functions to avoid learning redundant representations.

---

### Step 5: Equivariant Network Representation

**Claim:** Equivariant layers preserve symmetry through the network.

**Construction (G-Equivariant Layer):**

For a layer $\ell$ with input $f_\ell \in \mathbb{R}^{n_\ell}$ and output $f_{\ell+1} \in \mathbb{R}^{n_{\ell+1}}$:

$$f_{\ell+1} = \sigma(W_\ell f_\ell + b_\ell)$$

The layer is $G$-equivariant if:
$$\rho_{\ell+1}(g) f_{\ell+1}(x) = f_{\ell+1}(\rho_\ell(g) \cdot x)$$

This requires the weight matrix $W_\ell$ to satisfy:
$$\rho_{\ell+1}(g) W_\ell = W_\ell \rho_\ell(g) \quad \forall g \in G$$

**Solution space:** $W_\ell \in \text{Hom}_G(\rho_\ell, \rho_{\ell+1})$, the space of $G$-equivariant linear maps.

**Dimension reduction:** For large groups $G$, the equivariant weight space is much smaller than the full weight space, providing:
- Reduced sample complexity
- Built-in generalization across symmetry transforms
- Computational efficiency

---

## Connections to Classical Results

### Cohen & Welling: Group Equivariant CNNs (2016)

The seminal G-CNN paper formalizes equivariance for convolutional networks:

**G-Convolution:**
$$[f \star_G \psi](g) = \sum_{h \in G} f(h) \psi(g^{-1}h)$$

**Connection to KRNL-Equivariance:**
- Hypothesis (H2) (equivariant parametrization) is exactly the G-CNN equivariance constraint
- The risk invariance (Step 1) corresponds to augmentation-free data efficiency
- Gradient equivariance (Step 2) ensures training preserves symmetry

**Key result:** G-CNNs achieve state-of-the-art with fewer parameters by exploiting symmetry.

### Geometric Deep Learning (Bronstein et al., 2021)

The "5G" framework unifies:
- **Grids** (CNNs): Translation equivariance $G = \mathbb{Z}^2$
- **Groups** (G-CNNs): General group equivariance
- **Graphs** (GNNs): Permutation equivariance $G = S_n$
- **Geodesics** (Manifold NNs): Gauge equivariance
- **Gauges** (Gauge NNs): Local symmetry

**Connection to KRNL-Equivariance:**
| GDL Concept | KRNL-Equivariance Analog |
|-------------|-------------------------|
| Symmetry group $G$ | Group action on system distribution |
| Feature field $f: X \to V$ | Hypostructure assignment $\mathcal{H}_\Theta$ |
| Equivariant map | Equivariant parametrization (H2) |
| Message passing | Certificate propagation |
| Pooling (invariant) | Risk functional aggregation |

### Kondor & Trivedi: Generalization of Equivariance (2018)

Extended equivariant networks to arbitrary compact groups using harmonic analysis:

**Fourier-theoretic construction:**
$$W = \bigoplus_{\rho \in \hat{G}} W_\rho \otimes I_{d_\rho}$$

where $\hat{G}$ is the set of irreducible representations.

**Connection to KRNL-Equivariance:**
- Step 5 (quantitative stability) bounds error under approximate equivariance
- Harmonic analysis provides the "canonical" decomposition of parameter space

### Noether's Theorem and RL

**Classical Noether:** Continuous symmetries imply conserved quantities.

**RL Analog:**
- Symmetry group $G$ acting on MDP
- Conserved quantity: Value function $V(s)$ along $G$-orbits
- Infinitesimal version: $\langle \nabla V, \xi_s \rangle = 0$ for $\xi \in \mathfrak{g}$ (Lie algebra)

**KRNL-Equivariance Step 6** (infinitesimal equivariance) is precisely the RL version of Noether's theorem: gradients are perpendicular to symmetry directions.

---

## Implementation Notes

### Practical Equivariant Architectures

**1. Steerable CNNs (for $SO(2)$, $O(2)$, $SO(3)$):**
```
Architecture:
- Input: Image/3D data with rotational symmetry
- Layers: Steerable convolutions with harmonic filters
- Output: Rotation-invariant predictions

Use case: Molecular property prediction, medical imaging
```

**2. Graph Neural Networks (for $S_n$):**
```
Architecture:
- Input: Graph with node features
- Layers: Permutation-equivariant message passing
- Aggregation: Permutation-invariant pooling (sum, mean, max)
- Output: Graph-level predictions

Use case: Molecular dynamics, social networks
```

**3. Equivariant Transformers:**
```
Architecture:
- Input: Set/sequence with symmetry
- Attention: G-equivariant attention weights
- MLP: Equivariant feedforward layers

Use case: Point clouds, multi-agent RL
```

### Equivariant Policy Networks for RL

**For robotics with $SE(3)$ symmetry:**
```python
# Pseudocode for SE(3)-equivariant policy
class EquivariantPolicy(nn.Module):
    def __init__(self):
        # Equivariant encoder: observations -> G-equivariant features
        self.encoder = SE3EquivariantEncoder()
        # Invariant value head: features -> scalar V(s)
        self.value_head = InvariantMLP()
        # Equivariant policy head: features -> action distribution
        self.policy_head = SE3EquivariantPolicy()

    def forward(self, obs):
        features = self.encoder(obs)  # Equivariant features
        value = self.value_head(features)  # Invariant value
        action_dist = self.policy_head(features)  # Equivariant policy
        return action_dist, value
```

**For multi-agent with permutation symmetry:**
```python
# Pseudocode for permutation-equivariant multi-agent
class PermEquivariantMARL(nn.Module):
    def __init__(self, n_agents):
        self.gnn = EquivariantGNN()  # Message passing
        self.policy_head = PermEquivariantHead()

    def forward(self, agent_obs):
        # agent_obs: [n_agents, obs_dim]
        features = self.gnn(agent_obs)  # Equivariant
        actions = self.policy_head(features)  # Per-agent actions
        return actions  # Permutes correctly with agent ordering
```

### Training Considerations

**1. Weight Initialization:**
- Initialize within equivariant subspace
- Use $G$-averaged random initialization:
  $$W_0 = \frac{1}{|G|} \sum_{g \in G} \rho(g)^{-1} W_{\text{random}} \rho(g)$$

**2. Data Augmentation vs. Equivariance:**
- Data augmentation: Approximate equivariance via sampling
- Equivariant architecture: Exact equivariance by construction
- Trade-off: Augmentation is flexible but sample-inefficient; equivariance is rigid but data-efficient

**3. Symmetry Breaking:**
- Some tasks require breaking symmetry (e.g., choosing a reference frame)
- Use symmetry-breaking layers at output only
- Maintain equivariance in feature extraction

### Verification Algorithm

**Algorithm VerifyEquivariance($\pi_\theta$, $\mathcal{S}$, $G$, $\epsilon$):**

```
Input:
- Policy π_θ with parameters θ
- State distribution S
- Symmetry group G
- Tolerance ε > 0

Output:
- Certificate K_Equiv if verified, or FAIL

Procedure:

1. Test Value Invariance:
   - Sample N states s_i ~ S
   - For each s_i and random g_j ~ G:
     - Compute V(s_i) and V(g_j · s_i)
     - Verify |V(s_i) - V(g_j · s_i)| < ε
   - If violation: return FAIL

2. Test Policy Equivariance:
   - Sample N states s_i ~ S
   - For each s_i and random g_j ~ G:
     - Compute π(a | s_i) and π(g_j · a | g_j · s_i)
     - Verify KL(π(· | s_i) || π(g_j · · | g_j · s_i)) < ε
   - If violation: return FAIL

3. Test Gradient Equivariance:
   - Compute ∇J(θ) and ∇J(g · θ) for random g
   - Verify ||∇J(g · θ) - g · ∇J(θ)|| < ε
   - If violation: return FAIL

4. Return Certificate K_Equiv with test statistics
```

---

## Literature

### Foundational

- **Noether (1918):** "Invariante Variationsprobleme" - Symmetries and conservation laws
- **Weyl (1946):** "The Classical Groups" - Representation theory for compact groups

### Group Equivariant Deep Learning

- **Cohen & Welling (2016):** "Group Equivariant Convolutional Networks" (ICML) - Foundational G-CNN paper
- **Kondor & Trivedi (2018):** "On the Generalization of Equivariance and Convolution in Neural Networks" (ICML) - Harmonic analysis approach
- **Weiler et al. (2018):** "3D Steerable CNNs" (NeurIPS) - SE(3) equivariance for 3D data
- **Finzi et al. (2020):** "Generalizing Convolutional Neural Networks for Equivariance to Lie Groups" (ICML) - Continuous group equivariance

### Geometric Deep Learning

- **Bronstein et al. (2021):** "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges" - Comprehensive survey
- **Satorras et al. (2021):** "E(n) Equivariant Graph Neural Networks" (ICML) - Efficient E(n)-equivariant GNNs
- **Batzner et al. (2022):** "E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials" (Nature Communications)

### Equivariant Reinforcement Learning

- **van der Pol et al. (2020):** "MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning" (NeurIPS) - Equivariant RL theory
- **Wang et al. (2020):** "Policy Learning in SE(3) Action Spaces" (CoRL) - SE(3) equivariance for robotics
- **Zhao et al. (2022):** "Integrating Symmetry into Deep Dynamics Models for Improved Generalization" (ICLR)
- **Simm et al. (2020):** "Symmetry-Aware Actor-Critic for 3D Molecular Design" (ICLR)

### Applications

- **Fuchs et al. (2020):** "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks" (NeurIPS)
- **Hutchinson et al. (2021):** "LieTransformer: Equivariant Self-Attention for Lie Groups" (ICML)
- **Kofinas et al. (2021):** "Roto-translation equivariant convolutional networks" (Medical Image Analysis)

---

## Summary

The KRNL-Equivariance theorem, translated to AI/RL/ML, establishes:

**For environments with symmetry group $G$, equivariant neural networks achieve optimal performance without loss of representational capacity.**

Key implications for practice:

1. **Sample efficiency:** Equivariant networks require fewer samples by sharing weights across symmetry transforms
2. **Generalization:** Symmetry is built-in, not learned, ensuring perfect generalization across $G$-orbits
3. **Computational efficiency:** Reduced parameter count from equivariant constraints
4. **Training stability:** Gradient flow preserves symmetry structure throughout optimization

The theorem bridges:
- **PDE/Physics:** Noether's theorem (symmetries $\Leftrightarrow$ conservation)
- **Category theory:** Functorial equivariance
- **Deep learning:** Group equivariant architectures
- **RL:** Symmetric MDPs and policy optimization

**Certificate $K_{\text{SV08}}^+$ (Symmetry Preservation):** Verifiable through value invariance tests, policy equivariance checks, and gradient structure analysis.
