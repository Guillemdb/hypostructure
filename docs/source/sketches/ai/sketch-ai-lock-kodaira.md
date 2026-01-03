---
title: "LOCK-Kodaira - AI/RL/ML Translation"
---

# LOCK-Kodaira: Deformation Rigidity Lock

## Overview

The deformation rigidity lock shows that when the "cohomology" of the learned representation vanishes, the model becomes rigid—no local deformations are possible without increasing loss. This explains training plateaus and representation stability.

**Original Theorem Reference:** {prf:ref}`mt-lock-kodaira`

---

## AI/RL/ML Statement

**Theorem (Deformation Rigidity Lock, ML Form).**
For trained model $f_\theta$:

1. **Tangent Space:** Local deformations of parameters classified by gradient structure

2. **Obstruction:** Second-order curvature (Hessian) obstructs deformations

3. **Stiffness:** If gradient is zero and Hessian is positive definite, model is locally rigid

4. **Lock:** At strict local minima, no infinitesimal deformations decrease loss

**Corollary (Training Plateau).**
When $\nabla \mathcal{L} \approx 0$ but loss is not at global minimum, training is stuck at a rigid configuration—a plateau or saddle requires perturbation to escape.

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Kodaira-Spencer map | Gradient structure | $\nabla_\theta \mathcal{L}$ |
| $H^1(X, TX)$ | Descent directions | Negative curvature directions |
| Obstruction $H^2$ | Second-order barrier | Hessian structure |
| Rigidity | Local minimum | $\nabla \mathcal{L} = 0$, $H > 0$ |
| Deformation | Parameter perturbation | $\theta \mapsto \theta + \delta\theta$ |
| Kuranishi space | Loss landscape basin | Local structure around minimum |
| Moduli dimension | Flat directions | Null space of Hessian |

---

## Rigidity in Neural Network Training

### Local Analysis at Critical Points

**Definition.** At critical point $\theta^*$ with $\nabla \mathcal{L}(\theta^*) = 0$:
- **Minimum:** Hessian $H$ positive definite
- **Saddle:** Hessian indefinite
- **Maximum:** Hessian negative definite

**Rigidity:** At strict minimum, all deformations increase loss.

### Connection to Training Dynamics

| ML Property | Hypostructure Property |
|-------------|------------------------|
| Zero gradient | KS map vanishing |
| Positive Hessian | No obstructions |
| Flat directions | Moduli space |
| Saddle escape | Obstruction resolution |

---

## Proof Sketch

### Step 1: Taylor Expansion

**Local Structure.** Around critical point $\theta^*$:
$$\mathcal{L}(\theta^* + \delta\theta) = \mathcal{L}(\theta^*) + \frac{1}{2} \delta\theta^T H \delta\theta + O(\|\delta\theta\|^3)$$

**Obstruction:** Deformation $\delta\theta$ increases loss if $\delta\theta^T H \delta\theta > 0$.

**Reference:** Dauphin, Y., et al. (2014). Identifying and attacking the saddle point problem. *NeurIPS*.

### Step 2: Hessian Structure

**Eigenvalue Analysis.** Hessian $H = V \Lambda V^T$ with eigenvalues $\lambda_i$:
- $\lambda_i > 0$: Stable direction (rigid)
- $\lambda_i < 0$: Escape direction (soft)
- $\lambda_i = 0$: Flat direction (moduli)

**Rigidity Criterion:** All $\lambda_i > 0 \implies$ local minimum (rigid).

### Step 3: Saddle Points vs Minima

**Saddle Point Prevalence.** In high dimensions, critical points are typically saddles:
$$P(\text{minimum}) \propto 2^{-d}$$

**Escape:** Negative eigenvalue directions allow descent.

**Reference:** Bray, A. J., Dean, D. S. (2007). Statistics of critical points. *Phys. Rev. Lett.*, 98.

### Step 4: Flat Directions

**Definition.** Flat directions satisfy:
$$H v = 0, \quad v \neq 0$$

**Moduli Space:** The set of parameters with same loss:
$$\mathcal{M} = \{\theta: \mathcal{L}(\theta) = \mathcal{L}(\theta^*)\}$$

locally has dimension = number of zero eigenvalues.

**Reference:** Sagun, L., Bottou, L., LeCun, Y. (2017). Eigenvalues of the Hessian. *arXiv*.

### Step 5: Loss Landscape Geometry

**Curvature Analysis.** The Hessian spectrum reveals:
- Many near-zero eigenvalues (flat directions)
- Few large positive eigenvalues (important parameters)
- Some negative eigenvalues at saddles

**Reference:** Ghorbani, B., Krishnan, S., Xiao, Y. (2019). Investigation of neural network training dynamics. *ICML*.

### Step 6: Rigidity and Generalization

**Flat Minima Conjecture.** Flat minima (small Hessian eigenvalues) generalize better.

**PAC-Bayes Connection:**
$$\text{Generalization gap} \propto \sqrt{\frac{\text{tr}(H)}{n}}$$

**Reference:** Keskar, N., et al. (2017). On large-batch training. *ICLR*.

### Step 7: Training Plateaus

**Plateau Detection.** Plateau occurs when:
- $\|\nabla \mathcal{L}\| \approx 0$ (near critical point)
- Not at minimum (saddle or flat region)
- Standard GD cannot escape quickly

**Escape Methods:** Noise injection, momentum, learning rate scheduling.

**Reference:** Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.

### Step 8: Fisher Information Rigidity

**Fisher Information Matrix.**
$$F_{ij} = \mathbb{E}\left[\frac{\partial \log p(y|x;\theta)}{\partial \theta_i} \frac{\partial \log p(y|x;\theta)}{\partial \theta_j}\right]$$

**Rigidity:** Parameters with high Fisher information are "stiff"—changing them affects predictions significantly.

**Reference:** Pascanu, R., Bengio, Y. (2014). Revisiting natural gradient. *ICLR*.

### Step 9: Elastic Weight Consolidation

**EWC Regularization.** Protect important parameters:
$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{new}} + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_i^*)^2$$

**Rigidity Lock:** High Fisher → high penalty → parameters locked.

**Reference:** Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting. *PNAS*.

### Step 10: Compilation Theorem

**Theorem (Deformation Rigidity Lock):**

1. **Gradient:** $\nabla \mathcal{L} = 0$ at critical point
2. **Hessian:** Positive eigenvalues → rigid, negative → escape possible
3. **Flat:** Zero eigenvalues → moduli (equivalent solutions)
4. **Lock:** Strict minima are locally rigid

**Applications:**
- Understanding training dynamics
- Escaping saddle points
- Analyzing loss landscape
- Continual learning via EWC

---

## Key AI/ML Techniques Used

1. **Local Expansion:**
   $$\mathcal{L}(\theta + \delta) = \mathcal{L}(\theta) + \nabla \mathcal{L}^T \delta + \frac{1}{2} \delta^T H \delta$$

2. **Rigidity Criterion:**
   $$\nabla \mathcal{L} = 0, \quad H \succ 0 \implies \text{rigid}$$

3. **Saddle Index:**
   $$\text{index} = \#\{\lambda_i < 0\}$$

4. **Fisher Stiffness:**
   $$\text{stiffness}_i = F_{ii}$$

---

## Literature References

- Dauphin, Y., et al. (2014). Identifying saddle point problem. *NeurIPS*.
- Sagun, L., Bottou, L., LeCun, Y. (2017). Eigenvalues of the Hessian. *arXiv*.
- Ghorbani, B., Krishnan, S., Xiao, Y. (2019). Neural network training dynamics. *ICML*.
- Keskar, N., et al. (2017). On large-batch training. *ICLR*.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting. *PNAS*.
