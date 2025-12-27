---
title: "UP-ShadowRetro - AI/RL/ML Translation"
---

# UP-ShadowRetro: Retrospective Shadowing in Training

## Overview

The retrospective shadowing theorem establishes how current training states influence the interpretation and modification of past learning. Retrospective effects occur in replay, fine-tuning, and continual learning when later knowledge reshapes earlier representations.

**Original Theorem Reference:** {prf:ref}`mt-up-shadow-retro`

---

## AI/RL/ML Statement

**Theorem (Retrospective Shadowing, ML Form).**
For sequential training with experience replay or fine-tuning:

1. **Representation Shift:** Past inputs $x$ are reinterpreted:
   $$h^{new}(x) \neq h^{old}(x)$$ after training on new data.

2. **Shadowing Bound:** The retroactive change is bounded:
   $$\|h^{new}(x) - h^{old}(x)\| \leq C \cdot \|\Delta\theta\| \cdot L_{lip}$$

3. **Beneficial Shadowing:** If new knowledge improves representation:
   $$\mathcal{L}_{old}(h^{new}) \leq \mathcal{L}_{old}(h^{old})$$

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Retrospective shadow | Representation drift | Old inputs reinterpreted |
| Shadowing | Retroactive influence | New affects old |
| Shadow certificate | Stability measure | Change bound |
| Consistent history | Stable representations | Small drift |
| History rewriting | Catastrophic forgetting | Large drift harmful |
| Shadow alignment | Transfer benefit | New improves old |

---

## Retrospective Effects

### Types of Retrospective Shadowing

| Type | Mechanism | Effect |
|------|-----------|--------|
| Replay shadowing | Old data reprocessed | Updated representations |
| Fine-tuning shift | Pretrained model adapted | Base representations change |
| Continual drift | Sequential tasks | Early task representations evolve |
| Distillation shadow | Teacher → student | Student reinterprets teacher's knowledge |

### Beneficial vs Harmful

| Shadowing Type | When Beneficial | When Harmful |
|----------------|-----------------|--------------|
| Representation update | Better features learned | Old task performance drops |
| Knowledge integration | Synergy with new | Interference with old |
| Feature alignment | Improved transfer | Forgetting critical features |

---

## Proof Sketch

### Step 1: Representation Drift Definition

**Claim:** Training changes all representations, including for old inputs.

**Representation:**
$$h_\theta(x) = f_L \circ \cdots \circ f_1(x)$$

**After Update:**
$$h_{\theta'}(x) = h_\theta(x) + \nabla_\theta h \cdot \Delta\theta + O(\|\Delta\theta\|^2)$$

**Drift:** Even for $x$ not in current batch.

**Reference:** Ramasesh, V., et al. (2021). Effect of scale on catastrophic forgetting. *ICLR*.

### Step 2: Experience Replay Shadowing

**Claim:** Replayed experiences are processed with updated weights.

**Replay Buffer:** Store $(x, y)$ from past.

**Replay Step:** Compute $\mathcal{L}(f_\theta(x), y)$ with current $\theta$.

**Retrospective:** Past experience now generates different gradients.

**Shadow:** Current knowledge affects how past is learned from.

**Reference:** Mnih, V., et al. (2015). Human-level control through deep RL. *Nature*.

### Step 3: Fine-tuning Representation Shift

**Claim:** Fine-tuning changes pretrained representations.

**Pretrained:** $h_{pretrained}(x)$ learned on large dataset.

**Fine-tuned:** $h_{finetuned}(x)$ after task-specific training.

**Shift:**
$$\Delta h(x) = h_{finetuned}(x) - h_{pretrained}(x)$$

**Retrospective:** Pretrained representations reinterpreted for new task.

**Reference:** Howard, J., Ruder, S. (2018). Universal language model fine-tuning. *ACL*.

### Step 4: Shadowing Bound

**Claim:** Representation drift bounded by parameter change.

**Lipschitz Network:**
$$\|h_\theta(x) - h_{\theta'}(x)\| \leq L_h \cdot \|\theta - \theta'\|$$

**Local Bound:** For small $\Delta\theta$:
$$\|h_{\theta+\Delta\theta}(x) - h_\theta(x)\| \leq \|\nabla_\theta h(x)\| \cdot \|\Delta\theta\|$$

**Control:** Limit $\|\Delta\theta\|$ to control shadow.

### Step 5: Positive Transfer via Shadowing

**Claim:** Beneficial shadowing improves old task performance.

**Positive Transfer:** New task learning improves old:
$$\mathcal{L}_{old}(\theta_{new}) < \mathcal{L}_{old}(\theta_{old})$$

**Mechanism:** Shared structure in new task refines representations.

**Example:** Learning related tasks improves common features.

**Reference:** Ruder, S. (2017). An overview of multi-task learning. *arXiv*.

### Step 6: Negative Shadowing and Forgetting

**Claim:** Harmful shadowing causes forgetting.

**Catastrophic Forgetting:**
$$\mathcal{L}_{old}(\theta_{new}) \gg \mathcal{L}_{old}(\theta_{old})$$

**Mechanism:** New task overwrites critical old representations.

**Retrospective Damage:** Old inputs now misclassified.

**Reference:** McCloskey, M., Cohen, N. (1989). Catastrophic interference. *Psychology of Learning*.

### Step 7: Controlling Retrospective Effects

**Claim:** Regularization controls shadowing.

**EWC:** Penalize changes to important weights:
$$\mathcal{L}_{EWC} = \mathcal{L}_{new} + \frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_i^{old})^2$$

**Effect:** Limits retrospective changes to old-task-important parameters.

**Reference:** Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting. *PNAS*.

### Step 8: Gradient-based Shadowing Analysis

**Claim:** Gradient alignment indicates shadowing direction.

**Gradient Correlation:**
$$\cos(\nabla\mathcal{L}_{old}, \nabla\mathcal{L}_{new})$$

**Positive:** Aligned gradients $\implies$ beneficial shadowing.
**Negative:** Conflicting gradients $\implies$ harmful shadowing.

**Reference:** Lopez-Paz, D., Ranzato, M. (2017). Gradient episodic memory. *NeurIPS*.

### Step 9: Knowledge Distillation Shadowing

**Claim:** Distillation creates retrospective alignment.

**Distillation Loss:**
$$\mathcal{L}_{distill} = D_{KL}(p_{teacher} \| p_{student})$$

**Retrospective:** Student's representation shaped to match teacher.

**Shadow:** Teacher's knowledge retroactively influences student's learning.

**Reference:** Hinton, G., et al. (2015). Distilling knowledge. *NeurIPS Workshop*.

### Step 10: Compilation Theorem

**Theorem (Retrospective Shadowing):**

1. **Drift:** All representations change with training
2. **Bound:** $\|\Delta h\| \leq L \cdot \|\Delta\theta\|$
3. **Beneficial:** Related tasks improve shared features
4. **Harmful:** Unrelated tasks can cause forgetting

**Retrospective Certificate:**
$$K_{retro} = \begin{cases}
\|\Delta h\| / \|\Delta\theta\| & \text{sensitivity} \\
\cos(\nabla\mathcal{L}_{old}, \nabla\mathcal{L}_{new}) & \text{alignment} \\
\mathcal{L}_{old}(\theta_{new}) - \mathcal{L}_{old}(\theta_{old}) & \text{retrospective effect} \\
\sum_i F_i(\Delta\theta_i)^2 & \text{importance-weighted change}
\end{cases}$$

**Applications:**
- Continual learning
- Transfer learning
- Curriculum design
- Replay strategies

---

## Key AI/ML Techniques Used

1. **Representation Drift:**
   $$\Delta h = h_{\theta'}(x) - h_\theta(x)$$

2. **EWC Regularization:**
   $$\mathcal{L}_{EWC} = \mathcal{L}_{new} + \frac{\lambda}{2}\sum_i F_i\Delta\theta_i^2$$

3. **Gradient Alignment:**
   $$\cos(\nabla\mathcal{L}_{old}, \nabla\mathcal{L}_{new})$$

4. **Distillation:**
   $$\mathcal{L} = D_{KL}(p_T \| p_S)$$

---

## Literature References

- Ramasesh, V., et al. (2021). Effect of scale on catastrophic forgetting. *ICLR*.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting. *PNAS*.
- Lopez-Paz, D., Ranzato, M. (2017). Gradient episodic memory. *NeurIPS*.
- McCloskey, M., Cohen, N. (1989). Catastrophic interference.
- Hinton, G., et al. (2015). Distilling knowledge. *NeurIPS Workshop*.

