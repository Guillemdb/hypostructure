---
title: "UP-Scattering - AI/RL/ML Translation"
---

# UP-Scattering: Scattering in Optimization Landscapes

## Overview

The scattering theorem establishes how optimization trajectories disperse across the loss landscape. Scattering quantifies sensitivity to initial conditions and characterizes the diversity of reachable solutions from different starting points.

**Original Theorem Reference:** {prf:ref}`mt-up-scattering`

---

## AI/RL/ML Statement

**Theorem (Optimization Scattering, ML Form).**
For gradient descent trajectories $\theta_t^{(i)}$ starting from points $\theta_0^{(i)}$:

1. **Scattering Rate:** The trajectory separation grows as:
   $$\|\theta_t^{(1)} - \theta_t^{(2)}\| \leq e^{\lambda_{max} t} \|\theta_0^{(1)} - \theta_0^{(2)}\|$$
   where $\lambda_{max}$ is the maximum Lyapunov exponent.

2. **Convergence Scattering:** Near minima:
   $$\|\theta_\infty^{(1)} - \theta_\infty^{(2)}\| \leq C \cdot \|\theta_0^{(1)} - \theta_0^{(2)}\|^\alpha$$

3. **Basin Boundary:** Different basins lead to different attractors:
   $$\theta_0^{(1)} \in B(\theta^*_1), \theta_0^{(2)} \in B(\theta^*_2) \implies \theta_\infty^{(1)} \neq \theta_\infty^{(2)}$$

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Scattering | Trajectory divergence | Sensitivity to init |
| Lyapunov exponent | Instability rate | $\lambda = \lim \log\|\delta\theta\|/t$ |
| Attractor basin | Convergence region | $B(\theta^*)$ |
| Sensitive dependence | Chaos | $\lambda > 0$ |
| Asymptotic scattering | Final positions | $\theta_\infty$ distribution |
| Scattering cross-section | Basin measure | $|B(\theta^*)|$ |

---

## Scattering Analysis

### Scattering Regimes

| Regime | Lyapunov | Behavior |
|--------|----------|----------|
| Contracting | $\lambda < 0$ | Trajectories converge |
| Neutral | $\lambda = 0$ | Constant separation |
| Expanding | $\lambda > 0$ | Chaotic divergence |
| Mixed | Varies | Direction-dependent |

### Factors Affecting Scattering

| Factor | Effect on Scattering |
|--------|---------------------|
| Learning rate | Higher $\eta$ → more scattering |
| Loss curvature | Sharp → less scattering near minimum |
| Initialization | Wider init → more basins explored |
| Noise (SGD) | Induces stochastic scattering |

---

## Proof Sketch

### Step 1: Linearized Dynamics

**Claim:** Local scattering determined by Jacobian.

**Gradient Flow:**
$$\dot{\theta} = -\nabla\mathcal{L}(\theta)$$

**Perturbation:**
$$\dot{\delta\theta} = -H(\theta) \cdot \delta\theta$$

where $H = \nabla^2\mathcal{L}$ is Hessian.

**Growth/Decay:** Eigenvalues determine scattering.

**Reference:** Strogatz, S. (2015). *Nonlinear Dynamics and Chaos*. Westview.

### Step 2: Lyapunov Exponents

**Claim:** Lyapunov exponents quantify scattering rate.

**Definition:**
$$\lambda_i = \lim_{t \to \infty} \frac{1}{t} \log \frac{\|\delta\theta_i(t)\|}{\|\delta\theta_i(0)\|}$$

**Positive $\lambda$:** Exponential separation (chaos).
**Negative $\lambda$:** Exponential contraction (stability).

**Maximum:** $\lambda_{max}$ determines dominant behavior.

**Reference:** Oseledets, V. (1968). Multiplicative ergodic theorem. *Trans. Moscow Math. Soc.*.

### Step 3: Basin of Attraction

**Claim:** Scattering partitions initial conditions into basins.

**Basin:**
$$B(\theta^*) = \{\theta_0 : \lim_{t \to \infty} \theta_t = \theta^*\}$$

**Partition:** $\bigcup_i B(\theta^*_i) = \Theta$ (up to measure zero).

**Scattering:** Different basins $\implies$ different final points.

### Step 4: SGD Scattering

**Claim:** SGD noise induces stochastic scattering.

**SGD Update:**
$$\theta_{t+1} = \theta_t - \eta\nabla\mathcal{L}_B(\theta_t)$$

**Noise:** $\nabla\mathcal{L}_B - \nabla\mathcal{L}$ is zero-mean, bounded variance.

**Diffusion:** Trajectories spread around deterministic path.

**Basin Hopping:** Noise can push across basin boundaries.

**Reference:** Kleinberg, R., et al. (2018). Alternative view of SGD. *ICML*.

### Step 5: Loss Landscape Geometry

**Claim:** Landscape geometry determines scattering patterns.

**Saddle Points:** Slow dynamics near saddles increase scattering time.

**Valleys:** Trajectories funnel together in valleys.

**Ridges:** Trajectories diverge on ridges.

**Reference:** Li, H., et al. (2018). Visualizing the loss landscape. *NeurIPS*.

### Step 6: Mode Connectivity and Scattering

**Claim:** Connected modes reduce asymptotic scattering.

**Mode Connectivity:** If minima $\theta^*_1, \theta^*_2$ connected by low-loss path:
$$\text{Effective basins merge}$$

**Scattering:** Starting points in either basin reach equivalent solutions.

**Reference:** Draxler, F., et al. (2018). No barriers in neural network energy landscape. *ICML*.

### Step 7: Ensemble Diversity from Scattering

**Claim:** Scattering creates diverse ensemble members.

**Different Inits:** $\theta_0^{(i)}$ scatter to different $\theta_\infty^{(i)}$.

**Ensemble:**
$$f_{ensemble}(x) = \frac{1}{M}\sum_i f_{\theta_\infty^{(i)}}(x)$$

**Diversity:** Scattering $\implies$ diverse predictions $\implies$ better ensemble.

**Reference:** Lakshminarayanan, B., et al. (2017). Simple and scalable predictive uncertainty. *NeurIPS*.

### Step 8: Learning Rate and Scattering

**Claim:** Learning rate controls scattering magnitude.

**Discrete Dynamics:**
$$\theta_{t+1} = \theta_t - \eta\nabla\mathcal{L}$$

**Effective Lyapunov:**
$$\lambda_{eff} = \log|1 - \eta\lambda_H|$$

**Large $\eta$:** Can cause divergence ($|1 - \eta\lambda_H| > 1$).

**Reference:** Wu, L., et al. (2018). How SGD selects the global minima. *NeurIPS*.

### Step 9: Sharpness and Local Scattering

**Claim:** Sharp minima have less local scattering.

**Sharpness:** $\lambda_{max}(H)$ at minimum.

**Local Scattering:** $\lambda_{local} = -\eta\lambda_{max}(H)$ (negative = contraction).

**Sharp $\implies$:** Faster contraction, less local scattering.

**Flat $\implies$:** Slower contraction, more wandering.

**Reference:** Keskar, N., et al. (2017). On large-batch training. *ICLR*.

### Step 10: Compilation Theorem

**Theorem (Optimization Scattering):**

1. **Lyapunov:** $\lambda > 0$ implies chaotic scattering
2. **Basins:** Initial conditions partition into attractor basins
3. **SGD Noise:** Induces stochastic basin hopping
4. **Sharpness:** Controls local contraction rate

**Scattering Certificate:**
$$K_{scatter} = \begin{cases}
\lambda_{max} & \text{Lyapunov exponent} \\
|B(\theta^*)| & \text{basin measure} \\
\text{Var}(\theta_\infty) & \text{solution diversity} \\
\eta\lambda_H & \text{effective contraction}
\end{cases}$$

**Applications:**
- Ensemble methods
- Initialization strategies
- Optimizer design
- Uncertainty quantification

---

## Key AI/ML Techniques Used

1. **Lyapunov Exponent:**
   $$\lambda = \lim_{t \to \infty} \frac{1}{t}\log\|\delta\theta(t)\|$$

2. **Basin Definition:**
   $$B(\theta^*) = \{\theta_0 : \theta_t \to \theta^*\}$$

3. **Ensemble Diversity:**
   $$D = \mathbb{E}[\|f_{\theta^{(i)}} - f_{\theta^{(j)}}\|^2]$$

4. **Effective Contraction:**
   $$r = |1 - \eta\lambda_H|$$

---

## Literature References

- Strogatz, S. (2015). *Nonlinear Dynamics and Chaos*. Westview.
- Li, H., et al. (2018). Visualizing the loss landscape. *NeurIPS*.
- Draxler, F., et al. (2018). No barriers in NN energy landscape. *ICML*.
- Lakshminarayanan, B., et al. (2017). Predictive uncertainty. *NeurIPS*.
- Keskar, N., et al. (2017). On large-batch training. *ICLR*.

