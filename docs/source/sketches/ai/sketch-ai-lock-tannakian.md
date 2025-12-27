---
title: "LOCK-Tannakian - AI/RL/ML Translation"
---

# LOCK-Tannakian: Symmetry Recovery Lock

## Overview

The symmetry recovery lock shows that neural networks that respect certain symmetries (equivariance) are uniquely determined by their behavior on representative inputs. The symmetry group acts as a constraint that "locks" the network architecture and reduces the parameter space.

**Original Theorem Reference:** {prf:ref}`lock-tannakian`

---

## AI/RL/ML Statement

**Theorem (Symmetry Recovery Lock, ML Form).**
For a neural network $f_\theta$ equivariant to group $G$:

1. **Equivariance Constraint:** $f_\theta(g \cdot x) = \rho(g) \cdot f_\theta(x)$ for all $g \in G$

2. **Parameter Reduction:** Equivariance reduces parameters:
   $$\dim(\Theta_{\text{equiv}}) = \dim(\Theta) / |G|$$

3. **Unique Recovery:** The equivariant network is uniquely determined by its values on a fundamental domain:
   $$f_\theta|_{\mathcal{D}} \text{ determines } f_\theta \text{ everywhere}$$

**Corollary (Symmetry as Regularization).**
Equivariant architectures have lower sample complexity by a factor of $|G|$.

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Tannakian category | Equivariant function class | $\mathcal{F}^G$ |
| Fiber functor | Evaluation map | $f \mapsto f(x)$ |
| Tensor automorphism | Symmetry action | $g \in G$ acting on inputs |
| Group reconstruction | Architecture from symmetry | Equivariant layers |
| Intertwiner space | Equivariant maps | $\text{Hom}_G(V, W)$ |
| Schur's lemma | Equivariant decomposition | Irreducible components |

---

## Equivariant Neural Networks

### Group Actions on Data

**Definition.** Group $G$ acts on input space $\mathcal{X}$:
$$\rho_{\text{in}}: G \times \mathcal{X} \to \mathcal{X}$$

**Examples:**
- Translation ($\mathbb{R}^n$): CNNs
- Rotation ($SO(3)$): Spherical CNNs
- Permutation ($S_n$): Graph neural networks

### Connection to Architecture

| Symmetry Group | Equivariant Architecture |
|----------------|-------------------------|
| Translation | Convolution |
| Rotation | Spherical harmonics |
| Permutation | Message passing |
| Scale | Wavelet/multi-scale |

---

## Proof Sketch

### Step 1: Equivariance Definition

**Definition.** Network $f_\theta$ is $G$-equivariant if:
$$f_\theta(g \cdot x) = \rho_{\text{out}}(g) \cdot f_\theta(x)$$

for all $g \in G$, $x \in \mathcal{X}$.

**Reference:** Cohen, T., Welling, M. (2016). Group equivariant CNNs. *ICML*.

### Step 2: Parameter Sharing from Symmetry

**Theorem.** Equivariant linear maps have the form:
$$W \in \text{Hom}_G(\rho_{\text{in}}, \rho_{\text{out}}) = \{W : W \rho_{\text{in}}(g) = \rho_{\text{out}}(g) W\}$$

**Dimension:** $\dim(\text{Hom}_G) \leq \dim(\text{Hom})/|G|$

**Reference:** Kondor, R., Trivedi, S. (2018). On the generalization of equivariance. *ICML*.

### Step 3: Schur's Lemma in ML

**Lemma.** For irreducible representations $\rho, \sigma$:
$$\text{Hom}_G(\rho, \sigma) = \begin{cases} \mathbb{C} & \rho \cong \sigma \\ 0 & \text{otherwise} \end{cases}$$

**Application:** Equivariant layers decompose into irreducible blocks.

### Step 4: Group Convolution

**Definition.** Group convolution:
$$(f * \kappa)(g) = \int_G f(h) \kappa(h^{-1}g) \, dh$$

**Property:** Group convolution is automatically $G$-equivariant.

**Reference:** Weiler, M., Cesa, G. (2019). General E(2)-equivariant steerable CNNs. *NeurIPS*.

### Step 5: Steerable Networks

**Definition.** Steerable feature maps transform predictably under $G$:
$$f(g \cdot x) = \rho(g) f(x)$$

**Construction:** Use group representations as filters.

**Reference:** Cohen, T., Welling, M. (2017). Steerable CNNs. *ICLR*.

### Step 6: Graph Neural Networks as Permutation Equivariant

**Architecture.** Message passing:
$$h_v^{(l+1)} = \phi\left(h_v^{(l)}, \text{Agg}_{u \in \mathcal{N}(v)} \psi(h_u^{(l)})\right)$$

**Equivariance:** Permutation of nodes permutes outputs.

**Reference:** Gilmer, J., et al. (2017). Neural message passing. *ICML*.

### Step 7: Sample Complexity Reduction

**Theorem.** For $G$-equivariant networks:
$$n_{\text{equiv}} \leq \frac{n_{\text{general}}}{|G|}$$

samples needed for same performance.

**Proof:** Each sample provides $|G|$ effective data points via symmetry.

**Reference:** Elesedy, B., Zaidi, S. (2021). Provably strict generalization benefit. *ICML*.

### Step 8: Recovery from Fundamental Domain

**Theorem.** Equivariant function on fundamental domain $\mathcal{D}$:
$$f|_\mathcal{D} \to f|_\mathcal{X}$$

via $f(g \cdot x) = \rho(g) f(x)$ for $x \in \mathcal{D}$.

**Lock:** Equivariance "locks" the function to be determined by its values on $\mathcal{D}$.

### Step 9: Approximate Symmetries

**Challenge.** Real data has approximate, not exact, symmetries.

**Solutions:**
- Relaxed equivariance
- Data augmentation
- Soft constraints

**Reference:** Finzi, M., et al. (2021). Residual pathway priors. *ICML*.

### Step 10: Compilation Theorem

**Theorem (Symmetry Recovery Lock):**

1. **Equivariance:** $f(g \cdot x) = \rho(g) f(x)$
2. **Parameter Reduction:** Factor of $|G|$
3. **Unique Recovery:** $f|_\mathcal{D}$ determines $f$
4. **Lock:** Symmetry constrains architecture

**Applications:**
- Invariant/equivariant architectures
- Sample efficiency
- Physics-informed ML
- Geometric deep learning

---

## Key AI/ML Techniques Used

1. **Equivariance:**
   $$f(g \cdot x) = \rho(g) \cdot f(x)$$

2. **Group Convolution:**
   $$(f * \kappa)(g) = \int_G f(h) \kappa(h^{-1}g) dh$$

3. **Intertwiner:**
   $$W \in \text{Hom}_G(\rho_{\text{in}}, \rho_{\text{out}})$$

4. **Sample Reduction:**
   $$n_{\text{equiv}} = n_{\text{general}}/|G|$$

---

## Literature References

- Cohen, T., Welling, M. (2016). Group equivariant CNNs. *ICML*.
- Kondor, R., Trivedi, S. (2018). On the generalization of equivariance. *ICML*.
- Weiler, M., Cesa, G. (2019). General E(2)-equivariant steerable CNNs. *NeurIPS*.
- Bronstein, M., et al. (2021). Geometric deep learning. *arXiv*.
- Elesedy, B., Zaidi, S. (2021). Provably strict generalization benefit. *ICML*.

