---
title: "LOCK-TacticCapacity - AI/RL/ML Translation"
---

# LOCK-TacticCapacity: Capacity Lower Bounds

## Overview

The capacity lower bound lock establishes fundamental limits on what neural networks can learn given their architectural constraints. Networks below a certain capacity cannot solve tasks requiring more expressivity—they are locked out of the solution space.

**Original Theorem Reference:** {prf:ref}`lock-tactic-capacity`

---

## AI/RL/ML Statement

**Theorem (Capacity Lower Bound Lock, ML Form).**
For a learning task $\mathcal{T}$ with intrinsic complexity $C(\mathcal{T})$:

1. **Capacity Definition:** Network capacity is:
   $$\text{Cap}(f_\theta) = \min\{\dim(\theta), \text{VC-dim}(\mathcal{F}), \text{Rademacher}(\mathcal{F})\}$$

2. **Lower Bound:** To achieve error $\epsilon$ on task $\mathcal{T}$:
   $$\text{Cap}(f_\theta) \geq \frac{C(\mathcal{T})}{\epsilon^2}$$

3. **Lock:** Networks with insufficient capacity cannot learn:
   $$\text{Cap}(f_\theta) < C(\mathcal{T}) \implies \mathcal{L}(f_\theta) \geq \epsilon_{\min} > 0$$

**Corollary (No Free Lunch).**
No fixed-capacity network can learn all tasks—capacity must match task complexity.

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Capacity bound | Network capacity | Parameters, VC-dim |
| Task complexity | Intrinsic dimension | $C(\mathcal{T})$ |
| Lock condition | Underfitting regime | Cap < complexity |
| Capacity saturation | Overfitting regime | Cap >> complexity |
| Optimal capacity | Bias-variance balance | Cap ≈ complexity |

---

## Capacity Measures in Deep Learning

### VC Dimension

**Definition.** VC dimension of function class $\mathcal{F}$:
$$\text{VC-dim}(\mathcal{F}) = \max\{n : \exists x_1, \ldots, x_n \text{ shattered by } \mathcal{F}\}$$

### Connection to Network Architecture

| Architecture Property | Capacity Effect |
|-----------------------|-----------------|
| Depth $L$ | Exponential in $L$ |
| Width $w$ | Polynomial in $w$ |
| Parameters $d$ | Upper bound $O(d \log d)$ |

---

## Proof Sketch

### Step 1: VC Dimension of Neural Networks

**Theorem.** For ReLU network with $W$ weights:
$$\text{VC-dim}(\mathcal{F}_W) = O(WL \log W)$$

where $L$ is depth.

**Reference:** Bartlett, P. L., et al. (2019). Nearly-tight VC-dimension and pseudodimension bounds. *JMLR*.

### Step 2: Task Intrinsic Complexity

**Definition.** Task complexity via covering numbers:
$$C(\mathcal{T}) = \log \mathcal{N}(\epsilon, \mathcal{T}, d)$$

where $\mathcal{N}$ is the covering number of the target function class.

### Step 3: Lower Bound on Sample Complexity

**Theorem.** To learn with error $\epsilon$:
$$n \geq \frac{C(\mathcal{T})}{\epsilon^2}$$

samples are necessary.

**Capacity Implication:** Network must have capacity to distinguish $n$ examples.

**Reference:** Shalev-Shwartz, S., Ben-David, S. (2014). *Understanding Machine Learning*. Cambridge.

### Step 4: Width-Depth Tradeoffs

**Theorem.** There exist functions requiring:
- Width-$2^n$ depth-$2$ networks, OR
- Width-$O(n)$ depth-$O(n)$ networks

**Lock:** Wrong tradeoff → exponential capacity overhead.

**Reference:** Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.

### Step 5: Minimum Description Length

**Principle.** Optimal capacity minimizes:
$$\text{MDL} = \text{description}(\theta) + \text{description}(D | \theta)$$

**Lock:** Undercapacity forces high data description cost.

**Reference:** Rissanen, J. (1978). Modeling by shortest data description. *Automatica*.

### Step 6: Universal Approximation vs Efficiency

**Theorem (Cybenko 1989).** Width-$n$ networks can approximate any continuous function.

**But:** Approximation may require exponential width without sufficient depth.

**Lock:** Universal approximation ≠ efficient approximation.

### Step 7: Lottery Ticket Capacity

**Observation.** Sparse subnetworks can match full network performance.

**Implication:** Effective capacity may be much smaller than parameter count.

**Reference:** Frankle, J., Carlin, M. (2019). The lottery ticket hypothesis. *ICLR*.

### Step 8: Double Descent and Capacity

**Phenomenon.** Test error decreases, increases, then decreases again with capacity.

**Interpretation:**
- Phase 1: Underfitting (insufficient capacity)
- Phase 2: Overfitting (capacity ≈ samples)
- Phase 3: Interpolation (excess capacity regularizes)

**Reference:** Belkin, M., et al. (2019). Reconciling modern machine learning with the bias-variance trade-off. *PNAS*.

### Step 9: Capacity and Transfer Learning

**Observation.** Pretrained models have "implicit capacity" for downstream tasks.

**Lock Resolution:** Transfer learning overcomes capacity limits by importing learned features.

### Step 10: Compilation Theorem

**Theorem (Capacity Lower Bound Lock):**

1. **VC Bound:** $\text{VC-dim} = O(WL \log W)$
2. **Necessity:** $\text{Cap} \geq C(\mathcal{T})/\epsilon^2$ to achieve error $\epsilon$
3. **Lock:** Undercapacity → non-zero minimum error
4. **Resolution:** Increase depth/width, or use transfer learning

**Applications:**
- Architecture sizing
- Task feasibility analysis
- Transfer learning design
- Compression limits

---

## Key AI/ML Techniques Used

1. **VC Dimension:**
   $$\text{VC-dim}(\mathcal{F}) = \max\{n : \mathcal{F} \text{ shatters } n \text{ points}\}$$

2. **Sample Complexity:**
   $$n \geq \frac{C(\mathcal{T})}{\epsilon^2}$$

3. **Capacity Lock:**
   $$\text{Cap} < C(\mathcal{T}) \implies \mathcal{L} \geq \epsilon_{\min}$$

4. **MDL:**
   $$\text{MDL} = \text{len}(\theta) + \text{len}(D|\theta)$$

---

## Literature References

- Bartlett, P. L., et al. (2019). Nearly-tight VC-dimension and pseudodimension bounds. *JMLR*.
- Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.
- Shalev-Shwartz, S., Ben-David, S. (2014). *Understanding Machine Learning*. Cambridge.
- Belkin, M., et al. (2019). Reconciling modern machine learning with bias-variance. *PNAS*.
- Frankle, J., Carlin, M. (2019). The lottery ticket hypothesis. *ICLR*.

