---
title: "ACT-Projective - AI/RL/ML Translation"
---

# ACT-Projective: Projection and Embedding Principle

## Overview

The projection and embedding principle shows how to extend model configurations to completions by adding ideal points (embeddings to higher dimensions or projections to manifolds), controlling asymptotic behavior through the structure of the ambient space.

**Original Theorem Reference:** {prf:ref}`mt-act-projective`

---

## AI/RL/ML Statement

**Theorem (Projection/Embedding Extension, ML Form).**
For parameter space $\Theta \subset \mathbb{R}^d$:

1. **Embedding:** Lift $\Theta \hookrightarrow \tilde{\Theta} \subset \mathbb{R}^{D}$ with $D > d$ to overparameterized space

2. **Extension:** Loss function $\mathcal{L}$ extends to $\tilde{\mathcal{L}}$ with controlled behavior at boundary

3. **Projection:** Optimal solution in $\tilde{\Theta}$ projects back to $\Theta$ with preservation of properties

**Corollary (Overparameterization Regularization).**
Training in overparameterized space $\tilde{\Theta}$ with implicit regularization (gradient descent bias) followed by projection yields solutions with improved generalization.

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Projective space $\mathbb{P}^n$ | Extended parameter space | $\Theta \hookrightarrow \tilde{\Theta}$ |
| Hyperplane at infinity | Boundary of parameter region | Degenerate/sparse models |
| Fubini-Study metric | Parameter space geometry | Fisher information metric |
| Algebraic variety | Neural network function class | Expressible functions |
| Degree | Network complexity | Number of parameters |
| Chow's theorem | Representability | Universal approximation |
| Growth control | Regularization | Weight decay bounds |
| Moduli compactification | Architecture space | Adding degenerate architectures |

---

## Overparameterization Framework

### Embedding into Larger Space

**Definition.** For network $f_\theta$ with $d$ parameters, embed into overparameterized network $\tilde{f}_{\tilde{\theta}}$ with $D \gg d$ parameters:
$$f_\theta \hookrightarrow \tilde{f}_{\tilde{\theta}}$$

**Projection:** The solution $\tilde{\theta}^*$ projects to effective $\theta^* = P(\tilde{\theta}^*)$.

### Connection to Implicit Regularization

| ML Property | Hypostructure Property |
|-------------|------------------------|
| Overparameterization | Projective embedding |
| Gradient descent bias | Projective structure |
| Implicit regularization | Growth control |
| Minimum norm solution | Projective completion |

---

## Proof Sketch

### Step 1: Lifted Parameter Space

**Definition.** For linear network $f(x) = W_L \cdots W_1 x$, the lifted space is:
$$\tilde{\Theta} = \{(W_1, \ldots, W_L) : W_\ell \in \mathbb{R}^{d_\ell \times d_{\ell-1}}\}$$

**Overparameterization:** Many $\tilde{\theta}$ map to same function $f$.

**Reference:** Arora, S., Cohen, N., Hazan, E. (2018). On the optimization of deep networks. *ICML*.

### Step 2: Implicit Regularization

**Theorem (Gradient Descent Implicit Bias).** For linear networks with gradient descent from small initialization:
$$\tilde{\theta}^* = \arg\min_{\tilde{\theta}: \mathcal{L}(\tilde{\theta})=0} \|W_L \cdots W_1\|_*$$

where $\|\cdot\|_*$ is nuclear norm.

**Reference:** Gunasekar, S., et al. (2017). Implicit regularization in matrix factorization. *NeurIPS*.

### Step 3: Neural Network Embeddings

**Feature Embedding.** Map input $x \in \mathbb{R}^d$ to feature space:
$$\phi: x \mapsto \phi(x) \in \mathbb{R}^D$$

with $D \gg d$ (e.g., random features, kernel embedding).

**Projection:** Linear classifier in feature space:
$$f(x) = w^T \phi(x)$$

**Reference:** Rahimi, A., Recht, B. (2008). Random features for large-scale kernel machines. *NeurIPS*.

### Step 4: Kernel Methods as Projective Completion

**Reproducing Kernel Hilbert Space.** Embed to infinite-dimensional:
$$\phi: \mathcal{X} \to \mathcal{H}$$

**Kernel Trick:** Work in projected finite-dimensional subspace.

**Reference:** Schölkopf, B., Smola, A. J. (2002). *Learning with Kernels*. MIT Press.

### Step 5: ResNet as Projective Structure

**Residual Connection.** Each block:
$$h_{\ell+1} = h_\ell + f_\ell(h_\ell)$$

**Projective Interpretation:** Identity path + learned residual = projective coordinates.

**Reference:** He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning. *CVPR*.

### Step 6: Low-Rank Projection

**Matrix Factorization.** For weight $W \in \mathbb{R}^{m \times n}$:
$$W = UV^T, \quad U \in \mathbb{R}^{m \times r}, V \in \mathbb{R}^{n \times r}$$

**Projection:** Embed in $(m \times r) + (n \times r)$ space, project to rank-$r$ matrices.

**Reference:** Srebro, N., Shraibman, A. (2005). Rank, trace-norm and max-norm. *COLT*.

### Step 7: Attention as Projection

**Self-Attention.** Projections $Q, K, V$:
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

**Projective Structure:** Input projected to query/key/value spaces, combined via softmax.

**Reference:** Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.

### Step 8: Neural Tangent Kernel Embedding

**NTK Embedding.** At initialization, network behaves as kernel:
$$k(x, x') = \nabla_\theta f(x; \theta_0)^T \nabla_\theta f(x'; \theta_0)$$

**Infinite-Width Limit:** $D \to \infty$ gives projective completion (kernel limit).

**Reference:** Jacot, A., Gabriel, F., Hongler, C. (2018). Neural tangent kernel. *NeurIPS*.

### Step 9: Normalization as Projective Constraint

**Layer Normalization.** Project activations to sphere:
$$h \mapsto \frac{h - \mu}{\sigma}$$

**Projective Structure:** Scale invariance = projective equivalence.

**Reference:** Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). Layer normalization. *arXiv:1607.06450*.

### Step 10: Compilation Theorem

**Theorem (Projection/Embedding Principle):**

1. **Embedding:** $\Theta \hookrightarrow \tilde{\Theta}$ lifts to overparameterized space

2. **Regularization:** Implicit bias in $\tilde{\Theta}$ induces regularization

3. **Projection:** Optimal $\tilde{\theta}^* \mapsto \theta^*$ preserves structure

4. **Completion:** Boundary of $\tilde{\Theta}$ captures degenerate cases

**Algorithm (Projective Training):**
```python
def projective_training(model, data, projection_fn):
    """Training with projection to constrained manifold."""
    for epoch in range(num_epochs):
        for x, y in data:
            # Forward in lifted space
            loss = compute_loss(model(x), y)

            # Gradient step in lifted space
            loss.backward()
            optimizer.step()

            # Project back to constraint manifold
            with torch.no_grad():
                for param in model.parameters():
                    param.data = projection_fn(param.data)

    return model

# Example projections:
def project_to_sphere(W):
    """Project weights to unit sphere (weight normalization)."""
    return W / W.norm()

def project_to_low_rank(W, rank):
    """Project to rank-r matrices via SVD."""
    U, S, V = torch.svd(W)
    return U[:, :rank] @ torch.diag(S[:rank]) @ V[:, :rank].T

def project_to_simplex(W):
    """Project to probability simplex."""
    return torch.softmax(W, dim=-1)
```

**Applications:**
- Overparameterized neural networks
- Kernel methods and random features
- Low-rank factorization
- Weight normalization
- Attention mechanisms

---

## Key AI/ML Techniques Used

1. **Lifting:**
   $$\theta \mapsto \tilde{\theta} \in \mathbb{R}^D, \quad D \gg d$$

2. **Implicit Regularization:**
   $$\tilde{\theta}^* = \arg\min \|W\|_* \text{ s.t. } \mathcal{L} = 0$$

3. **Projection:**
   $$P: \tilde{\Theta} \to \Theta$$

4. **Kernel Embedding:**
   $$k(x, x') = \langle \phi(x), \phi(x') \rangle$$

---

## Literature References

- Arora, S., Cohen, N., Hazan, E. (2018). Optimization of deep networks. *ICML*.
- Gunasekar, S., et al. (2017). Implicit regularization in matrix factorization. *NeurIPS*.
- Rahimi, A., Recht, B. (2008). Random features. *NeurIPS*.
- Schölkopf, B., Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
- He, K., Zhang, X., et al. (2016). Deep residual learning. *CVPR*.
- Srebro, N., Shraibman, A. (2005). Rank, trace-norm and max-norm. *COLT*.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Jacot, A., Gabriel, F., Hongler, C. (2018). Neural tangent kernel. *NeurIPS*.
- Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). Layer normalization. *arXiv*.
