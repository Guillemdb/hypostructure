---
title: "LOCK-TacticScale - AI/RL/ML Translation"
---

# LOCK-TacticScale: Scale Separation Barriers

## Overview

The scale separation lock establishes that learning tasks at different scales (spatial, temporal, semantic) require corresponding multi-scale architectures. Single-scale networks cannot efficiently capture multi-scale phenomena—they are locked into limited scale ranges.

**Original Theorem Reference:** {prf:ref}`mt-lock-tactic-scale`

---

## AI/RL/ML Statement

**Theorem (Scale Separation Lock, ML Form).**
For a learning task with features at scales $\{\lambda_1, \ldots, \lambda_k\}$:

1. **Scale Range:** Define the scale ratio:
   $$r = \frac{\lambda_{\max}}{\lambda_{\min}}$$

2. **Architecture Requirement:** Network receptive field must cover all scales:
   $$\text{RF}(f_\theta) \geq \lambda_{\max}, \quad \text{Resolution}(f_\theta) \leq \lambda_{\min}$$

3. **Lock:** Single-scale networks with fixed receptive field $\text{RF}_0$ satisfy:
   $$r > \text{RF}_0/\delta \implies \text{cannot learn all scales}$$

**Corollary (Multi-Scale Architecture).**
Tasks with high scale ratio require hierarchical architectures (pyramids, U-nets, transformers with multi-head attention at different scales).

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Scale separation | Multi-scale features | Different frequencies/sizes |
| Critical exponent | Scaling law | Performance vs scale |
| Subcritical | Fine-grained | Local patterns |
| Supercritical | Coarse-grained | Global patterns |
| Scale barrier | Receptive field limit | Max learnable scale |

---

## Multi-Scale Learning

### Receptive Field

**Definition.** Receptive field of layer $l$:
$$\text{RF}_l = 1 + \sum_{i=1}^l (k_i - 1) \cdot \prod_{j=1}^{i-1} s_j$$

where $k_i$ is kernel size and $s_j$ is stride at layer $j$.

### Connection to Scale

| Architecture Property | Scale Coverage |
|-----------------------|----------------|
| Receptive field | Maximum scale |
| Stride | Scale resolution |
| Depth | Scale range |

---

## Proof Sketch

### Step 1: Fourier Analysis of Features

**Definition.** Feature scale via Fourier decomposition:
$$f(x) = \sum_k \hat{f}_k e^{2\pi i k x / \lambda_k}$$

**Scale Range:** Features span frequencies from $1/\lambda_{\max}$ to $1/\lambda_{\min}$.

### Step 2: Receptive Field Limits

**Theorem.** A convolution with kernel size $k$ can only detect features at scale $\leq k$.

**Proof:** Kernel acts as low-pass filter; features beyond kernel size are aliased.

**Reference:** Luo, W., et al. (2016). Understanding the effective receptive field in deep CNNs. *NeurIPS*.

### Step 3: Multi-Scale Architecture Necessity

**Theorem.** To capture scales $\{\lambda_1, \ldots, \lambda_k\}$ with ratio $r = \lambda_k/\lambda_1$:
$$\text{Depth} \geq \log_s(r)$$

where $s$ is the stride factor per layer.

**Reference:** Lin, T.-Y., et al. (2017). Feature pyramid networks. *CVPR*.

### Step 4: Spatial Pyramid Pooling

**Technique.** Pool at multiple scales:
$$\text{SPP}(x) = [\text{Pool}_1(x), \text{Pool}_2(x), \ldots, \text{Pool}_k(x)]$$

**Effect:** Captures features at all pooling scales.

**Reference:** He, K., et al. (2015). Spatial pyramid pooling in deep CNNs. *TPAMI*.

### Step 5: U-Net and Multi-Scale Processing

**Architecture.** Encoder-decoder with skip connections:
- Encoder: Captures coarse scales
- Decoder: Reconstructs fine scales
- Skips: Preserve detail at each scale

**Reference:** Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net. *MICCAI*.

### Step 6: Attention as Scale Selection

**Mechanism.** Self-attention spans all positions:
$$\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d})V$$

**Scale Advantage:** Attention bypasses receptive field limits by direct global access.

**Reference:** Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.

### Step 7: Scaling Laws

**Observation.** Performance scales with model size:
$$\mathcal{L} \propto C^{-\alpha}$$

where $C$ is compute and $\alpha$ is the scaling exponent.

**Scale Interpretation:** Larger models access broader scale ranges.

**Reference:** Kaplan, J., et al. (2020). Scaling laws for neural language models. *arXiv*.

### Step 8: Temporal Multi-Scale

**Challenge.** Time series with multiple frequencies.

**Solutions:**
- Dilated convolutions (WaveNet)
- Hierarchical RNNs
- Temporal attention

**Reference:** Oord, A. van den, et al. (2016). WaveNet. *arXiv*.

### Step 9: Transfer Across Scales

**Observation.** Features learned at one scale transfer to others poorly.

**Lock:** Single-scale pretraining limits multi-scale downstream performance.

### Step 10: Compilation Theorem

**Theorem (Scale Separation Lock):**

1. **Scale Range:** $r = \lambda_{\max}/\lambda_{\min}$
2. **Receptive Field:** $\text{RF} \geq \lambda_{\max}$ required
3. **Lock:** Single-scale networks fail when $r > \text{RF}/\delta$
4. **Resolution:** Multi-scale architectures (pyramids, U-nets, attention)

**Applications:**
- Image segmentation
- Object detection
- Time series forecasting
- Multi-resolution modeling

---

## Key AI/ML Techniques Used

1. **Receptive Field:**
   $$\text{RF}_l = 1 + \sum_{i=1}^l (k_i - 1) \prod_{j<i} s_j$$

2. **Scale Ratio:**
   $$r = \lambda_{\max}/\lambda_{\min}$$

3. **Multi-Scale Pool:**
   $$\text{SPP}(x) = [\text{Pool}_1(x), \ldots, \text{Pool}_k(x)]$$

4. **Scaling Law:**
   $$\mathcal{L} \propto C^{-\alpha}$$

---

## Literature References

- Luo, W., et al. (2016). Understanding the effective receptive field. *NeurIPS*.
- Lin, T.-Y., et al. (2017). Feature pyramid networks. *CVPR*.
- Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net. *MICCAI*.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Kaplan, J., et al. (2020). Scaling laws for neural language models. *arXiv*.

