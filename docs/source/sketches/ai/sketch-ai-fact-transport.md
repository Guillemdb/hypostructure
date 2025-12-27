---
title: "FACT-Transport - AI/RL/ML Translation"
---

# FACT-Transport: Optimal Transport in ML

## Overview

The transport factory constructs equivalence relations and transport maps between model configurations, enabling knowledge transfer, domain adaptation, and representation alignment across different but related models.

**Original Theorem Reference:** {prf:ref}`mt-fact-transport`

---

## AI/RL/ML Statement

**Theorem (Transport Factory, ML Form).**
There exists a factory $\mathcal{F}_{\text{trans}}$ that, given:
- Models $f_{\theta_1}, f_{\theta_2}$ with $f_{\theta_1} \sim f_{\theta_2}$ (equivalence)
- Equivalence witness (alignment map)
- Structure type to transport

produces transport map $\tau_{\theta_1 \to \theta_2}$ with:

1. **Structure Preservation:** $\tau$ preserves specified structure

2. **Functoriality:** $\tau_{\theta \to \theta} = \text{id}$, compositions work

3. **Naturality:** $\tau$ commutes with training dynamics

**Corollary (Domain Adaptation Transport).**
For source domain $\mathcal{D}_S$ and target domain $\mathcal{D}_T$, optimal transport provides alignment map:
$$T^*: \mathcal{D}_S \to \mathcal{D}_T$$

minimizing transport cost while preserving label structure.

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Equivalence $T \sim S$ | Model equivalence | Same function, different params |
| Transport map $\tau$ | Weight transfer | $\theta_S \to \theta_T$ |
| Isometry group | Parameter symmetries | Permutation, scaling |
| Pushforward | Domain adaptation | $T_\#: \mathcal{D}_S \to \mathcal{D}_T$ |
| Mass transport | Distribution matching | Wasserstein distance |
| Functoriality | Composable transfers | $\tau_{A \to C} = \tau_{B \to C} \circ \tau_{A \to B}$ |
| Naturality | Training invariance | $\tau \circ \text{train} = \text{train} \circ \tau$ |
| Optimal transport | Minimum cost alignment | $\min_T \mathbb{E}[c(x, T(x))]$ |

---

## Transport in Machine Learning

### Domain Adaptation

**Definition.** Transport source distribution to target:
$$\min_T W_2(T_\# p_S, p_T)$$

where $T_\#$ is the pushforward measure.

### Connection to Transfer Learning

| ML Property | Hypostructure Property |
|-------------|------------------------|
| Weight transfer | Isometry transport |
| Domain shift | Measure transport |
| Knowledge distillation | Function transport |
| Model merging | Mass transport |

---

## Proof Sketch

### Step 1: Weight Space Symmetries

**Parameter Symmetries.** Neural networks have:
- **Permutation symmetry:** Reorder neurons in hidden layer
- **Scaling symmetry:** Rescale with inverse in next layer

**Transport via Symmetry:** If $\theta_2 = g \cdot \theta_1$ for $g \in \text{Sym}$:
$$\tau_{\theta_1 \to \theta_2} = g$$

**Reference:** Ainsworth, S. K., Hayase, J., Srinivasa, S. (2023). Git re-basin. *ICLR*.

### Step 2: Optimal Transport Basics

**Kantorovich Problem.** Find optimal coupling:
$$W_2(p, q)^2 = \inf_{\gamma \in \Pi(p, q)} \int \|x - y\|^2 d\gamma(x, y)$$

**Transport Map (Brenier, 1991).** When $p$ is absolutely continuous:
$$T = \nabla \phi$$

for convex $\phi$, and $T_\# p = q$.

**Reference:** Villani, C. (2003). *Topics in Optimal Transportation*. AMS.

### Step 3: Domain Adaptation via OT

**Distribution Matching.** Align source and target:
$$\min_T \mathbb{E}_{x \sim p_S}[c(x, T(x))] \quad \text{s.t.} \quad T_\# p_S = p_T$$

**Application:** Map source features to target domain while preserving structure.

**Reference:** Courty, N., Flamary, R., Tuia, D., Rakotomamonjy, A. (2017). Optimal transport for domain adaptation. *IEEE TPAMI*, 39(9).

### Step 4: Knowledge Distillation Transport

**Teacher-Student Transport.** Transfer knowledge:
$$\tau: f_T \to f_S$$

minimizing:
$$\mathcal{L}_{\text{distill}} = \mathbb{E}_x[\text{KL}(p_T(y|x) \| p_S(y|x))]$$

**Reference:** Hinton, G., Vinyals, O., Dean, J. (2015). Distilling knowledge. *NeurIPS Workshop*.

### Step 5: Model Merging

**Weight Averaging.** Simple transport:
$$\theta_{\text{merged}} = \frac{1}{2}(\theta_1 + \theta_2)$$

**Optimal Transport Merging.** First align, then average:
$$\theta_{\text{merged}} = \frac{1}{2}(\theta_1 + \tau_{\theta_1 \to \theta_2}^{-1}(\theta_2))$$

**Reference:** Wortsman, M., et al. (2022). Model soups. *ICML*.

### Step 6: Representation Alignment

**Procrustes Transport.** Align representations via:
$$\min_Q \|X_S Q - X_T\|_F \quad \text{s.t.} \quad Q^T Q = I$$

**Solution:** $Q = UV^T$ from SVD of $X_T^T X_S$.

**Reference:** Schönemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem. *Psychometrika*, 31(1).

### Step 7: Neural OT

**Neural Optimal Transport.** Parameterize transport map:
$$T_\psi(x) = x + v_\psi(x)$$

Train via:
$$\min_\psi W_2(T_{\psi\#} p_S, p_T)$$

**Reference:** Korotin, A., et al. (2022). Neural optimal transport. *ICLR*.

### Step 8: Factory Construction

**Transport Factory:**
```python
def transport_factory(model_source, model_target, structure_type):
    """Construct transport between models."""

    if structure_type == 'weights':
        # Permutation alignment
        perm = find_permutation_alignment(
            model_source.weights,
            model_target.weights
        )
        return PermutationTransport(perm)

    if structure_type == 'distribution':
        # Optimal transport for distributions
        ot_map = compute_optimal_transport(
            model_source.feature_distribution,
            model_target.feature_distribution
        )
        return DistributionTransport(ot_map)

    if structure_type == 'function':
        # Knowledge distillation
        return DistillationTransport(
            teacher=model_source,
            student=model_target
        )

    if structure_type == 'representation':
        # Procrustes alignment
        Q = procrustes_alignment(
            model_source.representations,
            model_target.representations
        )
        return LinearTransport(Q)

    return TRANSPORT_NOT_IMPLEMENTED
```

### Step 9: Sinkhorn Algorithm

**Efficient OT.** For discrete distributions:
$$\min_P \langle P, C \rangle - \varepsilon H(P)$$

**Sinkhorn Iteration:**
$$P^{(k+1)} = \text{diag}(a^{(k)}) K \text{diag}(b^{(k)})$$

where $K = e^{-C/\varepsilon}$ and $a, b$ are scaling vectors.

**Reference:** Cuturi, M. (2013). Sinkhorn distances. *NeurIPS*.

### Step 10: Compilation Theorem

**Theorem (Transport Factory):**

1. **Inputs:** Equivalent models, structure type
2. **Outputs:** Transport map $\tau$
3. **Guarantees:**
   - Preserves specified structure
   - Functorial (composable)
   - Computable (efficient algorithms exist)

**Applications:**
- Domain adaptation
- Transfer learning
- Model merging/ensembling
- Knowledge distillation
- Federated learning

---

## Key AI/ML Techniques Used

1. **Wasserstein Distance:**
   $$W_2(p, q)^2 = \inf_{\gamma} \int \|x-y\|^2 d\gamma$$

2. **Permutation Alignment:**
   $$\min_\pi \|\theta_1 - \pi(\theta_2)\|$$

3. **Procrustes:**
   $$Q^* = UV^T \text{ from SVD of } X_T^T X_S$$

4. **Sinkhorn:**
   $$P^* = \lim_{k \to \infty} \text{diag}(a^k) K \text{diag}(b^k)$$

---

## Literature References

- Villani, C. (2003). *Topics in Optimal Transportation*. AMS.
- Courty, N., Flamary, R., et al. (2017). Optimal transport for domain adaptation. *IEEE TPAMI*, 39(9).
- Ainsworth, S. K., Hayase, J., Srinivasa, S. (2023). Git re-basin. *ICLR*.
- Cuturi, M. (2013). Sinkhorn distances. *NeurIPS*.
- Wortsman, M., et al. (2022). Model soups. *ICML*.
- Korotin, A., et al. (2022). Neural optimal transport. *ICLR*.
- Hinton, G., Vinyals, O., Dean, J. (2015). Distilling knowledge. *NeurIPS Workshop*.
