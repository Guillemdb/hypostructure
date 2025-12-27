---
title: "LOCK-Motivic - AI/RL/ML Translation"
---

# LOCK-Motivic: Architectural Invariant Lock

## Overview

The architectural invariant lock shows that fundamental structural properties of neural networks (analogous to motivic invariants) create barriers between model classes. Networks with different architectural invariants cannot be connected by continuous training—they occupy distinct regions of function space.

**Original Theorem Reference:** {prf:ref}`lock-motivic`

---

## AI/RL/ML Statement

**Theorem (Architectural Invariant Lock, ML Form).**
For neural network architectures:

1. **Invariant Class:** Each architecture $\mathcal{A}$ has invariants $I(\mathcal{A})$ (depth, width pattern, connectivity structure, symmetries)

2. **Function Space Partition:** Networks with different invariants realize disjoint function classes:
   $$I(\mathcal{A}_1) \neq I(\mathcal{A}_2) \implies \mathcal{F}_{\mathcal{A}_1} \cap \mathcal{F}_{\mathcal{A}_2} \text{ is measure-zero}$$

3. **Training Lock:** Training cannot transform one architectural class into another—invariants are preserved under gradient flow

**Corollary (No Universal Architecture).**
No single architecture can efficiently approximate all function classes. The choice of architecture locks the learnable function class.

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Motive $h(X)$ | Architecture signature | Depth, width, connectivity |
| L-function | Expressivity profile | What functions can be represented |
| Periods | Weight statistics | Spectral properties of weight matrices |
| Motivic class | Architecture class | Equivalent architectures modulo reparametrization |
| Deformation | Training dynamics | Continuous parameter updates |
| Lock | Expressivity barrier | Cannot represent certain functions |
| Grothendieck ring | Architecture taxonomy | Classification of network types |
| Hodge numbers | Layer dimensions | $h^l = \dim(\text{layer } l)$ |
| Galois group | Symmetry group | Permutation/equivariance structure |

---

## Architectural Invariants in Deep Learning

### Network Invariants

**Definition.** Key architectural invariants include:
- **Depth:** $d = \text{number of layers}$
- **Width pattern:** $\mathbf{w} = (w_1, \ldots, w_d)$
- **Connectivity:** Dense, convolutional, residual, attention
- **Symmetries:** Equivariances built into architecture

### Connection to Expressivity

| ML Property | Hypostructure Property |
|-------------|------------------------|
| Function class $\mathcal{F}_\mathcal{A}$ | Motivic realization |
| Depth separation | Locked classes |
| Width-depth tradeoff | Hodge decomposition |
| Equivariance group | Motivic Galois group |

---

## Proof Sketch

### Step 1: Neural Network Taxonomy

**Definition.** The architecture class of a network is determined by:
$$[\mathcal{A}] = (\text{depth}, \text{width pattern}, \text{connectivity}, \text{activation})$$

**Equivalence:** Two networks are equivalent if they have the same architecture class (up to width-preserving isomorphism).

**Reference:** Raghu, M., et al. (2017). On the expressive power of deep neural networks. *ICML*.

### Step 2: Depth Separation Theorem

**Theorem (Telgarsky 2016).** There exist functions computable by depth-$d$ networks of polynomial size that require exponential size for depth-$(d-1)$ networks.

**Lock Mechanism:** Depth creates an irreducible invariant—shallow networks cannot efficiently represent deep network functions.

**Proof Sketch:**
1. Construct oscillatory functions via iterated composition
2. Show shallow networks need exponential units to approximate
3. The depth invariant locks the function class

**Reference:** Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.

### Step 3: Width-Depth Tradeoffs

**Theorem (Lu et al. 2017).** For ReLU networks:
- Width-$n$ depth-$d$ networks can represent any continuous function
- But efficiency depends on matching $(n, d)$ to function structure

**Invariant Preservation:** The width-depth signature $(n, d)$ determines computational efficiency—mismatched signatures incur exponential overhead.

**Reference:** Lu, Z., et al. (2017). The expressive power of neural networks: A view from the width. *NeurIPS*.

### Step 4: Connectivity Invariants

**Definition.** Connectivity patterns create locked classes:
- **Dense:** $f(x) = W_d \sigma(\cdots W_1 x)$
- **Convolutional:** Local receptive fields, weight sharing
- **Residual:** $f(x) = x + g(x)$
- **Attention:** $\text{softmax}(QK^T)V$

**Lock:** Each connectivity class has different inductive biases. A CNN cannot efficiently learn arbitrary non-local patterns, while dense networks cannot efficiently learn translational structure.

**Reference:** Cohen, T., Welling, M. (2016). Group equivariant convolutional networks. *ICML*.

### Step 5: Symmetry/Equivariance Lock

**Definition.** Network equivariance under group $G$:
$$f(g \cdot x) = \rho(g) \cdot f(x) \quad \forall g \in G$$

**Motivic Galois Analog:** The equivariance group $G$ is an architectural invariant. Networks with different symmetry groups are locked—an $SO(3)$-equivariant network fundamentally differs from a translation-equivariant one.

**Reference:** Bronstein, M., et al. (2021). Geometric deep learning. *arXiv*.

### Step 6: Spectral Invariants

**Definition.** Spectral properties of weight matrices:
$$\sigma(W) = \text{spectrum of } W$$

**Invariant:** The spectral distribution at initialization constrains trainable functions. Networks with different spectral signatures converge to different attractors.

**Reference:** Pennington, J., Schoenholz, S., Ganguli, S. (2017). Resurrecting the sigmoid. *NeurIPS*.

### Step 7: Training Cannot Cross Locks

**Theorem.** Gradient descent on loss $\mathcal{L}$ preserves architectural invariants:
$$\frac{d}{dt} I(\theta_t) = 0$$

for invariants $I$ (depth, connectivity, equivariance).

**Proof:** Training updates parameters $\theta$, not architecture. The architecture class is fixed at initialization.

**Implication:** To change function class, one must change architecture—training alone is insufficient.

### Step 8: Neural Architecture Search

**Connection.** Neural Architecture Search (NAS) explicitly searches over architectural invariants:

$$\mathcal{A}^* = \arg\min_{\mathcal{A}} \mathcal{L}(\theta^*(\mathcal{A}))$$

**Lock Resolution:** NAS resolves locks by selecting the appropriate architecture class for the task.

**Reference:** Elsken, T., Metzen, J. H., Hutter, F. (2019). Neural architecture search: A survey. *JMLR*.

### Step 9: Universal Approximation vs Efficiency

**Theorem (Cybenko 1989, Hornik 1991).** Single hidden layer networks can approximate any continuous function.

**But (Efficiency Lock):** Universal approximation says nothing about efficiency. Depth separation shows that some functions require exponential width in shallow networks.

**Lock:** While all architectures are universal approximators, they are not universally efficient—the invariant class determines efficient function representation.

### Step 10: Compilation Theorem

**Theorem (Architectural Invariant Lock):**

1. **Invariants:** Depth, width pattern, connectivity, symmetry are preserved under training
2. **Function Partition:** Different invariant classes realize different function classes efficiently
3. **Separation:** Exponential gaps exist between classes (depth separation)
4. **Lock:** Cannot efficiently cross between classes via training

**Applications:**
- Architecture selection for specific tasks
- Understanding transfer learning limits
- Designing problem-specific networks
- Neural architecture search motivation

---

## Key AI/ML Techniques Used

1. **Depth Separation:**
   $$\text{Deep} \not\approx_{\text{poly}} \text{Shallow}$$

2. **Invariant Preservation:**
   $$\frac{d}{dt} I(\theta_t) = 0 \text{ under GD}$$

3. **Function Class:**
   $$\mathcal{F}_\mathcal{A} = \{f_\theta : \theta \in \Theta_\mathcal{A}\}$$

4. **Equivariance:**
   $$f(g \cdot x) = \rho(g) \cdot f(x)$$

---

## Literature References

- Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.
- Raghu, M., et al. (2017). On the expressive power of deep neural networks. *ICML*.
- Lu, Z., et al. (2017). The expressive power of neural networks: A view from the width. *NeurIPS*.
- Cohen, T., Welling, M. (2016). Group equivariant convolutional networks. *ICML*.
- Bronstein, M., et al. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. *arXiv*.
- Elsken, T., Metzen, J. H., Hutter, F. (2019). Neural architecture search: A survey. *JMLR*, 20.

