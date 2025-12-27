---
title: "UP-Absorbing - AI/RL/ML Translation"
---

# UP-Absorbing: Absorbing States in Training Dynamics

## Overview

The absorbing states theorem establishes conditions under which training dynamics reach absorbing configurations—states from which the system cannot escape via gradient updates. These include optimal minima, degenerate solutions, and training traps.

**Original Theorem Reference:** {prf:ref}`mt-up-absorbing`

---

## AI/RL/ML Statement

**Theorem (Absorbing States, ML Form).**
A parameter configuration $\theta^*$ is absorbing if:

1. **Gradient Absorption:** $\nabla\mathcal{L}(\theta^*) = 0$
2. **Stability:** All eigenvalues of $H(\theta^*) \geq 0$
3. **Basin Attraction:** $\exists \epsilon > 0$ s.t. $\|\theta - \theta^*\| < \epsilon \implies \theta_t \to \theta^*$

**Types of Absorbing States:**

| Type | Gradient | Hessian | Quality |
|------|----------|---------|---------|
| Global minimum | 0 | $\succ 0$ | Optimal |
| Local minimum | 0 | $\succeq 0$ | Suboptimal |
| Saddle point | 0 | Indefinite | Unstable |
| Degenerate | 0 | Singular | Problematic |

**Escape Condition:** From saddle $\theta_s$:
$$P(\text{escape}) = 1 \text{ under SGD noise}$$

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Absorbing state | Stationary point | $\nabla\mathcal{L} = 0$ |
| Global attractor | Global minimum | $\theta^* = \arg\min\mathcal{L}$ |
| Local attractor | Local minimum | Local minimizer |
| Saddle | Saddle point | $\lambda_{\min}(H) < 0$ |
| Basin of attraction | Convergence region | Where GD converges to $\theta^*$ |
| Escape dynamics | Saddle escape | SGD noise helps escape |
| Degenerate attractor | Mode collapse | Multiple equivalent solutions |

---

## Absorbing State Classification

### By Hessian Structure

| State | $\lambda_{\min}(H)$ | $\lambda_{\max}(H)$ | Stability |
|-------|---------------------|---------------------|-----------|
| Strict local min | $> 0$ | $< \infty$ | Stable |
| Weak local min | $= 0$ | $< \infty$ | Marginally stable |
| Saddle | $< 0$ | $> 0$ | Unstable |
| Maximum | $< 0$ | $< 0$ | Repelling |

### Training Traps

| Trap | Cause | Escape |
|------|-------|--------|
| Sharp minimum | High curvature | Noise, large LR |
| Flat region | Near-zero gradient | Momentum |
| Saddle | Negative curvature | SGD noise |
| Plateau | Loss stagnation | LR schedule |

---

## Proof Sketch

### Step 1: Stationary Point Characterization

**Claim:** Training converges to stationary points.

**Gradient Descent:**
$$\theta_{t+1} = \theta_t - \eta\nabla\mathcal{L}(\theta_t)$$

**Fixed Point:**
$$\theta^* = \theta^* - \eta\nabla\mathcal{L}(\theta^*) \iff \nabla\mathcal{L}(\theta^*) = 0$$

**Convergence:** For $L$-smooth, $\mu$-strongly convex:
$$\|\theta_t - \theta^*\| \leq (1 - \mu\eta)^t \|\theta_0 - \theta^*\|$$

**Reference:** Nesterov, Y. (2004). *Introductory Lectures on Convex Optimization*. Springer.

### Step 2: Local Minimum Stability

**Claim:** Local minima are absorbing under GD.

**Lyapunov Function:** $V(\theta) = \mathcal{L}(\theta) - \mathcal{L}(\theta^*)$

**Decrease:**
$$\frac{d}{dt}V(\theta_t) = -\|\nabla\mathcal{L}\|^2 \leq 0$$

**Basin:** Set $B_\epsilon = \{\theta : V(\theta) < \epsilon\}$ is invariant.

### Step 3: Saddle Point Instability

**Claim:** Saddles are not absorbing under stochastic dynamics.

**Escape Probability:** For SGD with noise $\sigma$:
$$P(\text{escape from saddle}) = 1 - e^{-ct}$$

for some rate $c > 0$ depending on $\sigma$ and $|\lambda_{\min}|$.

**Mechanism:** Noise explores negative curvature directions.

**Reference:** Ge, R., et al. (2015). Escaping from saddle points. *COLT*.

### Step 4: Mode Collapse as Absorbing

**Claim:** GAN mode collapse is an absorbing state.

**Collapsed State:**
$$G(z) = x_0 \quad \forall z$$

**Gradient:** If $D$ saturated, $\nabla_G\mathcal{L} \approx 0$.

**Absorbing:** Generator stuck at single mode.

**Escape:** Minibatch discrimination, spectral normalization.

**Reference:** Salimans, T., et al. (2016). Improved techniques for GANs. *NeurIPS*.

### Step 5: Policy Collapse in RL

**Claim:** Deterministic policies are absorbing.

**Collapsed Policy:**
$$\pi(a|s) = \delta_{a_0}(a)$$

**Entropy:** $H(\pi) = 0$.

**Absorbing:** No exploration $\implies$ no learning.

**Escape:** Entropy regularization:
$$\mathcal{L}_{\text{reg}} = \mathcal{L} - \alpha H(\pi)$$

**Reference:** Haarnoja, T., et al. (2018). Soft actor-critic. *ICML*.

### Step 6: Dead Neurons

**Claim:** Dead ReLU neurons are absorbing.

**Dead State:**
$$\text{ReLU}(W^T x + b) = 0 \quad \forall x \in \mathcal{D}$$

**Gradient:**
$$\nabla_W\mathcal{L} = 0 \quad \text{(no gradient flows)}$$

**Absorbing:** Neuron never activates, never updates.

**Prevention:** Leaky ReLU, careful initialization.

**Reference:** Lu, L., et al. (2019). Dying ReLU and initialization. *AAAI*.

### Step 7: Representation Collapse

**Claim:** Collapsed representations are absorbing.

**Collapse:**
$$f_\theta(x) = c \quad \forall x$$

**Gradient:** If all outputs identical, loss gradient may vanish.

**Self-Supervised:** Contrastive loss prevents this:
$$\mathcal{L} = -\log\frac{\exp(\text{sim}(z_i, z_j))}{\sum_k \exp(\text{sim}(z_i, z_k))}$$

**Reference:** Chen, T., et al. (2020). SimCLR. *ICML*.

### Step 8: Basin of Attraction Size

**Claim:** Basin size determines practical convergence.

**Local Basin:** For minimum $\theta^*$:
$$B(\theta^*) = \{\theta_0 : \lim_{t\to\infty}\theta_t = \theta^*\}$$

**Wide Minima:** Flat minima have larger basins.

**Generalization:** Flat minima $\implies$ better generalization.

**Reference:** Keskar, N.S., et al. (2017). On large-batch training. *ICLR*.

### Step 9: Global vs Local Absorption

**Claim:** Global minima attract from almost everywhere.

**PL Condition:** If $\|\nabla\mathcal{L}\|^2 \geq 2\mu(\mathcal{L} - \mathcal{L}^*)$:
$$\theta_t \to \theta^* \quad \text{from any } \theta_0$$

**Neural Networks:** Overparameterized networks often satisfy PL locally.

**Reference:** Karimi, H., et al. (2016). Linear convergence under PL. *JMLR*.

### Step 10: Compilation Theorem

**Theorem (Absorbing States):**

1. **Stationary:** $\nabla\mathcal{L}(\theta^*) = 0$
2. **Stable:** $H(\theta^*) \succeq 0$ for local minima
3. **Unstable:** Saddles escaped by SGD noise
4. **Degenerate:** Mode/representation collapse

**Absorption Certificate:**
$$K_{\text{abs}} = \begin{cases}
\theta^* & \text{absorbing point} \\
H(\theta^*) & \text{Hessian (stability)} \\
B(\theta^*) & \text{basin size}
\end{cases}$$

**Applications:**
- Convergence analysis
- Training diagnostics
- Mode collapse detection
- Saddle escape strategies

---

## Key AI/ML Techniques Used

1. **Stationary Point:**
   $$\nabla\mathcal{L}(\theta^*) = 0$$

2. **Hessian Analysis:**
   $$H = \nabla^2\mathcal{L}$$

3. **Escape Rate:**
   $$P(\text{escape}) = 1 - e^{-ct}$$

4. **Entropy Regularization:**
   $$\mathcal{L}_{\text{reg}} = \mathcal{L} - \alpha H(\pi)$$

---

## Literature References

- Nesterov, Y. (2004). *Introductory Lectures on Convex Optimization*. Springer.
- Ge, R., et al. (2015). Escaping from saddle points. *COLT*.
- Salimans, T., et al. (2016). Improved techniques for GANs. *NeurIPS*.
- Haarnoja, T., et al. (2018). Soft actor-critic. *ICML*.
- Keskar, N.S., et al. (2017). On large-batch training. *ICLR*.

