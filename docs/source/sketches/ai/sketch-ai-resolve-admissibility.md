---
title: "RESOLVE-Admissibility - AI/RL/ML Translation"
---

# RESOLVE-Admissibility: Training Modification Trichotomy

## Overview

The training modification admissibility theorem establishes that before any model modification (pruning, fine-tuning, architecture change), the framework produces exactly one of three certificates: admissible, admissible up to equivalence, or not admissible with explicit failure witness.

**Original Theorem Reference:** {prf:ref}`mt-resolve-admissibility`

---

## AI/RL/ML Statement

**Theorem (Training Modification Trichotomy, ML Form).**
For any proposed model modification $M$ (pruning, surgery, fine-tuning) with mode $\mu$ and modification data $D_M$, the framework produces exactly one certificate:

**Case 1: Admissible ($K_{\text{adm}}$)**
$$K_{\text{adm}} = (M, \theta, \text{validity proof}, K_{\text{progress}}^+)$$
The modification satisfies:
1. **Canonicity:** Modified model is in standard form
2. **Capacity Bound:** $\|\Delta\theta\| \leq \varepsilon_{\text{adm}}$
3. **Progress:** Loss improves: $\mathcal{L}(\theta') < \mathcal{L}(\theta) - \delta$

**Case 2: Admissible up to Equivalence ($K_{\text{adm}}^\sim$)**
$$K_{\text{adm}}^\sim = (K_{\text{equiv}}, K_{\text{transport}}, K_{\text{adm}}[\theta'])$$
After equivalence transformation (reparameterization, normalization), modification becomes admissible.

**Case 3: Not Admissible ($K_{\text{inadm}}$)**
$$K_{\text{inadm}} = (\text{failure reason}, \text{witness})$$
Explicit obstruction:
- **Capacity exceeded:** Modification too large
- **Performance collapse:** Loss increases unboundedly
- **Structural violation:** Architecture constraints broken

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Surgery $S$ | Model modification | Pruning, fine-tuning, distillation |
| Mode $M$ | Modification type | Pruning ratio, learning rate schedule |
| Surgery data $D_S$ | Modification parameters | Layers to prune, target model |
| Singular set $\Sigma$ | Problematic weights | Near-zero, divergent, or unstable |
| Profile $V$ | Weight distribution | Spectral profile of layers |
| Canonical library $\mathcal{L}_T$ | Standard architectures | ResNet, Transformer variants |
| Codimension bound | Layer rank bound | Effective dimension of weight matrices |
| Capacity $\text{Cap}(\Sigma)$ | Modification magnitude | $\|\theta - \theta'\|$ |
| Progress density $\Delta\Phi$ | Loss improvement | $\mathcal{L}(\theta) - \mathcal{L}(\theta')$ |
| Equivalence move | Reparameterization | BatchNorm folding, weight rescaling |

---

## Training Modification Framework

### Types of Modifications

**Definition.** Model modifications in ML include:

| Modification | Description | Capacity Cost |
|--------------|-------------|---------------|
| Pruning | Remove weights/neurons | $O(\text{sparsity})$ |
| Quantization | Reduce precision | $O(\text{bits reduced})$ |
| Fine-tuning | Update subset of weights | $O(\|\Delta\theta\|)$ |
| Distillation | Transfer to smaller model | $O(\text{size ratio})$ |
| Architecture surgery | Modify layer structure | $O(\text{layers changed})$ |

### Admissibility Criteria

**Definition.** A modification is admissible if:
1. **Bounded perturbation:** $\|\theta' - \theta\| \leq \varepsilon$
2. **Performance preservation:** $\mathcal{L}(\theta') \leq \mathcal{L}(\theta) + \delta$
3. **Structural validity:** $\theta' \in \Theta_{\text{valid}}$

---

## Proof Sketch

### Step 1: Canonicity Verification

**Claim:** Verify that the modified model is in canonical form.

**Algorithm:**
```
CanonicalCheck(theta_new):
    if is_standard_architecture(theta_new):
        return PASS
    elif can_normalize(theta_new):
        return PASS_EQUIV
    else:
        return FAIL
```

**Examples:**
- BatchNorm + Conv can be folded into single layer
- Redundant skip connections can be simplified
- Equivalent parameterizations can be standardized

**Reference:** Frankle, J., Carlin, M. (2019). The lottery ticket hypothesis. *ICLR*.

### Step 2: Capacity Bound Verification

**Claim:** Modification magnitude must be bounded.

**Bound:** For weight modification $\Delta\theta = \theta' - \theta$:
$$\|\Delta\theta\|_F \leq \varepsilon_{\text{adm}} \cdot \|\theta\|_F$$

**Pruning Bound:** For pruning mask $m \in \{0, 1\}^d$:
$$\|m \odot \theta\|_0 \leq (1 - s) \cdot d$$

where $s$ is the sparsity ratio.

**Reference:** Han, S., et al. (2015). Learning both weights and connections. *NeurIPS*.

### Step 3: Progress Guarantee

**Claim:** Each admissible modification makes measurable progress.

**Progress Measure:**
$$\Delta\mathcal{L} = \mathcal{L}(\theta) - \mathcal{L}(\theta') \geq \delta_{\min}$$

**For Pruning:** Use importance scores:
$$\text{Importance}(w) = |w| \cdot |\nabla_w \mathcal{L}|$$

Remove weights with lowest importance.

**Reference:** Molchanov, P., et al. (2017). Pruning CNNs for resource efficient inference. *ICLR*.

### Step 4: Equivalence Transformations

**Claim:** Some modifications require preprocessing to become admissible.

**Equivalence Moves:**

| Move | Description | Effect |
|------|-------------|--------|
| BatchNorm folding | Merge BN into preceding layer | Simplifies architecture |
| Weight scaling | Rescale layers uniformly | Preserves function |
| Permutation | Reorder neurons | Equivalent representation |
| Factorization | Decompose large layers | Reduces parameters |

**Example:** Before pruning, fold BatchNorm layers to avoid scale mismatch.

### Step 5: Failure Certificates

**Claim:** Inadmissible modifications produce explicit failure witnesses.

**Failure Types:**

| Failure | Witness | Interpretation |
|---------|---------|----------------|
| Capacity exceeded | $\|\Delta\theta\| > \varepsilon_{\text{adm}}$ | Too aggressive modification |
| Loss explosion | $\mathcal{L}(\theta') \gg \mathcal{L}(\theta)$ | Modification breaks model |
| Rank collapse | $\text{rank}(W') < r_{\min}$ | Representation power lost |
| Gradient vanishing | $\|\nabla\mathcal{L}\| < \epsilon$ | Untrainable configuration |

**Certificate Construction:**
```
K_inadm = {
    failure_type: "capacity_exceeded",
    witness: {
        delta_norm: ||theta' - theta||,
        threshold: eps_adm,
        ratio: delta_norm / threshold
    }
}
```

### Step 6: Pruning Admissibility

**Theorem.** Magnitude-based pruning is admissible if:
$$\sum_{w \in \text{pruned}} |w|^2 \leq \varepsilon^2 \cdot \|\theta\|^2$$

**Proof:** Taylor expansion of loss:
$$\mathcal{L}(\theta') \approx \mathcal{L}(\theta) + \nabla\mathcal{L}^T \Delta\theta + \frac{1}{2}\Delta\theta^T H \Delta\theta$$

For small-weight pruning, $\|\Delta\theta\|$ is small, bounding loss increase.

**Reference:** LeCun, Y., et al. (1990). Optimal brain damage. *NeurIPS*.

### Step 7: Fine-Tuning Admissibility

**Theorem.** Fine-tuning is admissible if learning rate satisfies:
$$\eta \leq \frac{2}{\lambda_{\max}(H)}$$

where $H$ is the Hessian of the loss.

**Progress:** With proper $\eta$:
$$\mathcal{L}(\theta_{t+1}) \leq \mathcal{L}(\theta_t) - \eta\|\nabla\mathcal{L}\|^2 + O(\eta^2)$$

**Reference:** Nesterov, Y. (2004). *Introductory Lectures on Convex Optimization*. Springer.

### Step 8: Distillation Admissibility

**Theorem.** Knowledge distillation is admissible if:
$$D_{\text{KL}}(p_{\text{teacher}} \| p_{\text{student}}) \leq \varepsilon$$

**Criterion:** Student capacity must match teacher knowledge:
$$\text{Cap}(\theta_{\text{student}}) \geq \text{Info}(\theta_{\text{teacher}})$$

where Info measures effective information content.

**Reference:** Hinton, G., et al. (2015). Distilling the knowledge in a neural network. *NeurIPS Workshop*.

### Step 9: Architecture Surgery Admissibility

**Theorem.** Layer insertion/removal is admissible if:
1. **Residual addition:** Insert identity-initialized layer
2. **Residual removal:** Remove near-identity layer
3. **Width change:** Bounded rank perturbation

**Example (Net2Net):** Insert identity layer:
$$W_{\text{new}} = I, \quad b_{\text{new}} = 0$$

preserves function exactly.

**Reference:** Chen, T., et al. (2016). Net2Net: Accelerating learning via knowledge transfer. *ICLR*.

### Step 10: Compilation Theorem

**Theorem (Training Modification Trichotomy):**

1. **Admissible:** Modification bounded, progress guaranteed
2. **Admissible$^\sim$:** After normalization, becomes admissible
3. **Inadmissible:** Explicit failure witness

**Certificate:**
$$K_{\text{Mod}} = \begin{cases}
K_{\text{adm}} & \text{if } \|\Delta\theta\| \leq \varepsilon \text{ and } \Delta\mathcal{L} \geq \delta \\
K_{\text{adm}}^\sim & \text{if admissible after equivalence} \\
K_{\text{inadm}} & \text{if no bounded modification works}
\end{cases}$$

**Applications:**
- Automated model compression
- Safe fine-tuning
- Neural architecture search
- Continual learning

---

## Key AI/ML Techniques Used

1. **Modification Bound:**
   $$\|\theta' - \theta\|_F \leq \varepsilon_{\text{adm}} \cdot \|\theta\|_F$$

2. **Progress Guarantee:**
   $$\mathcal{L}(\theta') \leq \mathcal{L}(\theta) - \delta$$

3. **Importance Score:**
   $$I(w) = |w| \cdot |\nabla_w \mathcal{L}|$$

4. **Trichotomy:**
   $$K \in \{K_{\text{adm}}, K_{\text{adm}}^\sim, K_{\text{inadm}}\}$$

---

## Literature References

- LeCun, Y., et al. (1990). Optimal brain damage. *NeurIPS*.
- Han, S., et al. (2015). Learning both weights and connections. *NeurIPS*.
- Frankle, J., Carlin, M. (2019). The lottery ticket hypothesis. *ICLR*.
- Hinton, G., et al. (2015). Distilling the knowledge. *NeurIPS Workshop*.
- Chen, T., et al. (2016). Net2Net. *ICLR*.

