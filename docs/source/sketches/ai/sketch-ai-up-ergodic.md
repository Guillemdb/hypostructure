---
title: "UP-Ergodic - AI/RL/ML Translation"
---

# UP-Ergodic: Ergodicity in Training Dynamics

## Overview

The ergodic theorem establishes when training dynamics explore the full parameter space versus remaining trapped in local regions. Ergodic training ensures representative sampling of loss landscape, enabling proper convergence and generalization.

**Original Theorem Reference:** {prf:ref}`mt-up-ergodic`

---

## AI/RL/ML Statement

**Theorem (Training Ergodicity, ML Form).**
For stochastic training dynamics on parameter space $\Theta$:

1. **Ergodic Condition:** Training is ergodic if:
   $$\lim_{T \to \infty} \frac{1}{T}\int_0^T f(\theta_t) dt = \int_\Theta f(\theta) d\mu(\theta)$$
   for all observables $f$ and unique invariant measure $\mu$.

2. **Mixing Time:** Time to approach invariant distribution:
   $$\|\mathbb{P}(\theta_t \in \cdot) - \mu\|_{TV} \leq Ce^{-t/\tau_{mix}}$$

3. **Exploration-Exploitation:** Ergodicity requires:
   $$\text{noise level} \geq \Omega(\text{barrier height})$$

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Ergodicity | Full exploration | Time avg = space avg |
| Invariant measure | Stationary distribution | Gibbs/Boltzmann |
| Mixing time | Convergence rate | Steps to stationarity |
| Phase space | Parameter space | $\Theta \subset \mathbb{R}^d$ |
| Recurrence | Revisiting regions | Return to neighborhoods |
| Quasi-ergodicity | Metastability | Trapped in local basins |

---

## Ergodicity in Training

### Training Regimes

| Regime | Ergodic? | Consequence |
|--------|----------|-------------|
| Pure GD | No | Converges to single minimum |
| SGD | Partial | Explores local basin |
| SGLD | Yes (with conditions) | Explores according to $e^{-\mathcal{L}/T}$ |
| Simulated annealing | Asymptotically | Temperature schedule critical |

### Factors Affecting Ergodicity

| Factor | Effect on Ergodicity |
|--------|---------------------|
| Learning rate | Higher $\implies$ more exploration |
| Batch size | Smaller $\implies$ more noise |
| Temperature | Higher $\implies$ more ergodic |
| Barriers | Higher $\implies$ less ergodic |

---

## Proof Sketch

### Step 1: Langevin Dynamics for Training

**Claim:** SGLD implements Langevin dynamics.

**Stochastic Gradient Langevin Dynamics:**
$$\theta_{t+1} = \theta_t - \eta\nabla\mathcal{L}(\theta_t) + \sqrt{2\eta T}\xi_t$$

where $\xi_t \sim \mathcal{N}(0, I)$.

**Stationary Distribution:**
$$\pi(\theta) \propto e^{-\mathcal{L}(\theta)/T}$$

**Reference:** Welling, M., Teh, Y. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML*.

### Step 2: Mixing Time Analysis

**Claim:** Mixing time depends on landscape geometry.

**Log-Sobolev Inequality:**
$$\text{Ent}_\pi(f^2) \leq \frac{2}{\rho} \mathbb{E}_\pi[\|\nabla f\|^2]$$

**Mixing Time Bound:**
$$\tau_{mix} \leq O(1/\rho)$$

**Dependence:** $\rho$ relates to curvature and temperature.

**Reference:** Raginsky, M., et al. (2017). Non-convex learning via SGLD. *COLT*.

### Step 3: Barrier Crossing

**Claim:** Barriers limit ergodicity timescale.

**Kramers' Rate:** Escape time from local minimum:
$$\tau_{escape} \sim e^{\Delta \mathcal{L}/T}$$

**High Barriers:** Long escape times break practical ergodicity.

**Low Temperature:** Effectively non-ergodic.

**Reference:** Kramers, H. (1940). Brownian motion in a field of force. *Physica*.

### Step 4: SGD as Implicit Langevin

**Claim:** SGD noise provides implicit temperature.

**Gradient Noise:**
$$\nabla\mathcal{L}_B(\theta) = \nabla\mathcal{L}(\theta) + \epsilon$$

with $\text{Cov}(\epsilon) \approx \Sigma/|B|$.

**Effective Temperature:**
$$T_{eff} \approx \eta \cdot \text{tr}(\Sigma)/(2|B|)$$

**Partial Ergodicity:** SGD explores basins but may not cross barriers.

**Reference:** Mandt, S., et al. (2017). Stochastic gradient descent as approximate Bayesian inference. *JMLR*.

### Step 5: Mode Connectivity and Ergodicity

**Claim:** Connected modes enable ergodic exploration.

**Mode Connectivity:** If modes $\theta_1, \theta_2$ connected by low-loss path:
$$\max_{t \in [0,1]} \mathcal{L}(\gamma(t)) \leq \mathcal{L}(\theta_1) + \epsilon$$

**Ergodicity:** Connected modes are ergodically accessible.

**Disconnected:** High-barrier separation prevents ergodic mixing.

**Reference:** Draxler, F., et al. (2018). Essentially no barriers in neural network energy landscape. *ICML*.

### Step 6: Cyclical Learning Rates

**Claim:** Cyclical schedules enhance ergodicity.

**Schedule:**
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\pi t/T_{cycle}))$$

**Effect:** Periodic high learning rates enable barrier crossing.

**Snapshot Ensembles:** Capture diverse modes during cycling.

**Reference:** Loshchilov, I., Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *ICLR*.

### Step 7: Replica Exchange

**Claim:** Parallel tempering achieves ergodicity.

**Multiple Chains:** Run at temperatures $T_1 < T_2 < \cdots < T_K$.

**Exchange:** Swap configurations between chains.

**Hot Chain:** Ergodically explores, communicates to cold chain.

**Reference:** Swendsen, R., Wang, J. (1986). Replica Monte Carlo simulation. *PRL*.

### Step 8: Ergodicity Breaking in Deep Learning

**Claim:** Deep networks can break ergodicity.

**Over-parameterization:** Many equivalent minima, dynamics get trapped.

**Feature Learning:** Early layers freeze, breaking ergodicity.

**Layer-wise:** Different layers have different effective temperatures.

### Step 9: Practical Ergodicity Measures

**Claim:** Ergodicity can be diagnosed practically.

**Metrics:**
- Autocorrelation time: How quickly observables decorrelate
- Effective sample size: Independent samples from trajectory
- Basin hopping: Frequency of crossing loss thresholds

**Non-ergodic Symptoms:**
- Training stuck in local minimum
- Poor mode coverage
- Sensitivity to initialization

### Step 10: Compilation Theorem

**Theorem (Training Ergodicity):**

1. **SGLD Ergodicity:** With sufficient temperature, converges to Gibbs measure
2. **Mixing Time:** $\tau_{mix} \sim e^{\Delta\mathcal{L}/T}$ for barriers
3. **SGD Partial:** Explores basins, may not cross barriers
4. **Enhancement:** Cyclical rates, replica exchange improve ergodicity

**Ergodicity Certificate:**
$$K_{erg} = \begin{cases}
\pi(\theta) & \text{invariant measure} \\
\tau_{mix} & \text{mixing time} \\
\Delta\mathcal{L} & \text{barrier heights} \\
T_{eff} & \text{effective temperature}
\end{cases}$$

**Applications:**
- Bayesian neural networks
- Uncertainty quantification
- Global optimization
- Mode discovery

---

## Key AI/ML Techniques Used

1. **Langevin Dynamics:**
   $$d\theta = -\nabla\mathcal{L}dt + \sqrt{2T}dW$$

2. **Mixing Time:**
   $$\tau_{mix} = O(1/\rho)$$

3. **Effective Temperature:**
   $$T_{eff} = \eta\cdot\text{tr}(\Sigma)/(2|B|)$$

4. **Kramers Rate:**
   $$\tau_{escape} \sim e^{\Delta\mathcal{L}/T}$$

---

## Literature References

- Welling, M., Teh, Y. (2011). SGLD. *ICML*.
- Raginsky, M., et al. (2017). Non-convex learning via SGLD. *COLT*.
- Mandt, S., et al. (2017). SGD as approximate Bayesian inference. *JMLR*.
- Draxler, F., et al. (2018). No barriers in NN landscape. *ICML*.
- Loshchilov, I., Hutter, F. (2017). SGDR. *ICLR*.

