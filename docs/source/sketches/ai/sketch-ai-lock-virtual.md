---
title: "LOCK-Virtual - AI/RL/ML Translation"
---

# LOCK-Virtual: Virtual Sample Complexity

## Overview

The virtual sample complexity lock establishes that effective sample complexity can differ from actual sample count due to data redundancy, augmentation, or implicit regularization. Virtual sample counts provide corrected estimates that account for information-theoretic structure.

**Original Theorem Reference:** {prf:ref}`mt-lock-virtual`

---

## AI/RL/ML Statement

**Theorem (Virtual Sample Complexity, ML Form).**
For a learning problem with:
- Actual samples: $n$
- Data redundancy factor: $r$ (correlation between samples)
- Augmentation factor: $a$ (effective multiplier from augmentation)
- Regularization factor: $\rho$ (implicit prior strength)

The **virtual sample count** is:
$$n_{\text{vir}} = \frac{n \cdot a}{\max(1, r)} + \rho$$

And generalization bounds use $n_{\text{vir}}$ instead of $n$:
$$\text{Gap} \leq O\left(\sqrt{\frac{d}{n_{\text{vir}}}}\right)$$

**Corollary (Effective Data Efficiency).**
Data augmentation and regularization increase effective sample size without collecting more data.

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Virtual fundamental class | Virtual sample count | $n_{\text{vir}}$ |
| Obstruction | Data redundancy | Correlated samples |
| Excess dimension | Overcounting | Augmented copies |
| Signed counting | Effective samples | Information-adjusted $n$ |
| Inclusion-exclusion | Redundancy correction | Remove double-counting |
| Deformation invariance | Augmentation invariance | Same label under transform |

---

## Virtual Samples in Machine Learning

### Data Augmentation as Virtual Samples

**Mechanism.** Augmentation creates virtual copies:
$$\tilde{D} = \{(g_j \cdot x_i, y_i) : (x_i, y_i) \in D, g_j \in G\}$$

**Virtual Sample Count:** $n_{\text{aug}} = n \cdot |G|$

### Connection to Generalization

| Augmentation | Virtual Sample Effect |
|--------------|----------------------|
| Horizontal flip | $\times 2$ |
| Random crop | $\times k$ (number of crops) |
| Color jitter | $\times$ (continuous) |
| Mixup | $\times$ (interpolation space) |

---

## Proof Sketch

### Step 1: Augmentation Sample Complexity

**Theorem.** With augmentation group $G$, sample complexity reduces:
$$n_{\text{required}} \leq \frac{n_{\text{no-aug}}}{|G|}$$

**Proof:** Each sample provides $|G|$ effective training points via equivariance.

**Reference:** Chen, S., et al. (2020). Group equivariant stand-alone self-attention. *ICML*.

### Step 2: Redundancy Discount

**Definition.** Effective sample size under correlation:
$$n_{\text{eff}} = \frac{n}{1 + (n-1)\rho}$$

where $\rho$ is average pairwise correlation.

**Reference:** Kish, L. (1965). *Survey Sampling*. Wiley.

### Step 3: Information-Theoretic Virtual Samples

**Definition.** Virtual sample count via information:
$$n_{\text{vir}} = \frac{I(X; Y)}{I(x; y)}$$

where $I(X; Y)$ is total mutual information and $I(x; y)$ is per-sample.

**Interpretation:** Redundant samples contribute less information.

### Step 4: Regularization as Prior Samples

**Bayesian View.** Regularization corresponds to prior:
$$\lambda \|\theta\|^2 \leftrightarrow n_0 \text{ pseudo-samples from prior}$$

**Virtual Contribution:** Prior adds $n_0 = \lambda / \sigma^2$ virtual samples.

**Reference:** Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

### Step 5: Bootstrap and Effective Samples

**Bootstrap.** Resample with replacement to estimate variance.

**Effective Sample Size:**
$$n_{\text{eff}} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}$$

for weighted samples.

### Step 6: Mixup and Interpolated Samples

**Mixup.** Create virtual samples:
$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j$$

**Virtual Sample Space:** Continuous interpolation space.

**Reference:** Zhang, H., et al. (2018). Mixup: Beyond empirical risk minimization. *ICLR*.

### Step 7: Self-Training and Pseudo-Labels

**Mechanism.** Use model predictions as virtual labels:
$$\tilde{D} = D \cup \{(x_u, \hat{y}_u) : x_u \in D_{\text{unlabeled}}\}$$

**Virtual Samples:** Pseudo-labeled data adds virtual training signal.

**Reference:** Lee, D.-H. (2013). Pseudo-label. *ICML Workshop*.

### Step 8: Contrastive Learning Virtual Samples

**Mechanism.** Each positive pair and $K$ negatives:
$$n_{\text{contrastive}} = n \cdot K$$

**Virtual Effect:** More negatives → more effective samples.

**Reference:** He, K., et al. (2020). Momentum contrast for unsupervised learning. *CVPR*.

### Step 9: Transfer Learning and Pretrained Priors

**Virtual Samples from Pretraining:**
$$n_{\text{eff}} = n_{\text{downstream}} + \alpha \cdot n_{\text{pretrain}}$$

where $\alpha$ measures transfer quality.

**Reference:** Neyshabur, B., et al. (2020). What is being transferred in transfer learning? *NeurIPS*.

### Step 10: Compilation Theorem

**Theorem (Virtual Sample Complexity):**

1. **Augmentation:** $n_{\text{aug}} = n \cdot |G|$
2. **Redundancy:** $n_{\text{eff}} = n / (1 + (n-1)\rho)$
3. **Regularization:** Adds $n_0 = \lambda/\sigma^2$ virtual samples
4. **Bound:** Gap ≤ $O(\sqrt{d/n_{\text{vir}}})$

**Applications:**
- Data augmentation design
- Sample efficiency estimation
- Transfer learning
- Semi-supervised learning

---

## Key AI/ML Techniques Used

1. **Augmentation Factor:**
   $$n_{\text{aug}} = n \cdot |G|$$

2. **Effective Sample Size:**
   $$n_{\text{eff}} = \frac{n}{1 + (n-1)\rho}$$

3. **Prior Samples:**
   $$n_0 = \lambda / \sigma^2$$

4. **Generalization with Virtual:**
   $$\text{Gap} \leq O\left(\sqrt{\frac{d}{n_{\text{vir}}}}\right)$$

---

## Literature References

- Zhang, H., et al. (2018). Mixup: Beyond empirical risk minimization. *ICLR*.
- Chen, S., et al. (2020). Group equivariant stand-alone self-attention. *ICML*.
- He, K., et al. (2020). Momentum contrast for unsupervised learning. *CVPR*.
- Neyshabur, B., et al. (2020). What is being transferred in transfer learning? *NeurIPS*.
- Shorten, C., Khoshgoftaar, T. M. (2019). A survey on image data augmentation. *Journal of Big Data*.

