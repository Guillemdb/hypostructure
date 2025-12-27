---
title: "FACT-Lock - AI/RL/ML Translation"
---

# FACT-Lock: Parameter Locking Factory

## Overview

The parameter locking factory constructs mechanisms that prevent transitions to bad configurations by establishing invariants that training cannot violate. This includes frozen layers, constrained optimization, and architectural locks that block certain parameter regions.

**Original Theorem Reference:** {prf:ref}`mt-fact-lock`

---

## AI/RL/ML Statement

**Theorem (Lock Backend Factory, ML Form).**
There exists a factory $\mathcal{F}_{\text{lock}}$ that, given:
- Current configuration $\theta \in \Theta$
- Bad configurations $B \subset \Theta$
- Soft certificates (invariants, constraints)

produces lock $\text{Lock}_B$ with:

1. **Blocking:** No gradient trajectory from $\theta$ reaches $B$

2. **Certificate:** Explicit obstruction proving unreachability

3. **Verifiability:** Certificate is checkable at each training step

**Corollary (Layer Freezing Lock).**
Freezing pretrained layers $\theta_{\text{frozen}}$ creates a lock: the subspace $\{\theta: \theta_{\text{frozen}} = \theta_0^{\text{frozen}}\}$ is invariant under training, blocking access to configurations with different pretrained weights.

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Lock | Parameter constraint | $\theta \in \mathcal{C}$ enforced |
| Hom-emptiness | Unreachable configurations | No gradient path to $B$ |
| Topological obstruction | Architecture constraint | Network structure invariant |
| Capacity obstruction | Model capacity bound | $\|\theta\|_0 \leq k$ |
| Dimensional obstruction | Layer size constraint | $d_\ell$ fixed |
| Energy obstruction | Loss barrier | $\mathcal{L}(B) > \mathcal{L}(\theta) + \Delta$ |
| Certificate | Constraint proof | Verified at each step |
| Lock factory | Constraint construction | Automated lock generation |

---

## Locking Mechanisms in ML

### Layer Freezing

**Definition.** Freeze parameters by zeroing gradients:
$$\nabla_{\theta_{\text{frozen}}} \mathcal{L} \mapsto 0$$

**Lock Property:** Frozen parameters remain at initial values throughout training.

### Connection to Constrained Optimization

| ML Property | Hypostructure Property |
|-------------|------------------------|
| Frozen layers | Dimensional lock |
| Weight constraints | Capacity lock |
| Spectral bounds | Curvature lock |
| Architecture | Topological lock |

---

## Proof Sketch

### Step 1: Parameter Space Morphisms

**Morphisms in Weight Space.** A training trajectory $\gamma: [0, T] \to \Theta$ is a morphism if:
$$\gamma(t) = \theta_0 - \int_0^t \eta \nabla \mathcal{L}(\gamma(s)) ds$$

**Lock:** A set $B$ is locked if no such trajectory reaches $B$.

### Step 2: Topological Locks

**Architecture Invariant.** The network structure (layer count, widths) is fixed:
$$L(\theta) = L_0, \quad d_\ell(\theta) = d_\ell^0 \quad \forall \theta$$

**Lock:** Cannot reach configurations with different architecture.

**Reference:** He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning. *CVPR*.

### Step 3: Frozen Layer Locks

**Definition.** Freeze layers $\ell \in S$:
$$\theta_\ell^t = \theta_\ell^0 \quad \forall t, \forall \ell \in S$$

**Implementation:** Zero gradients for frozen parameters:
```python
for name, param in model.named_parameters():
    if should_freeze(name):
        param.requires_grad = False
```

**Reference:** Yosinski, J., et al. (2014). How transferable are features? *NeurIPS*.

### Step 4: Capacity Locks

**Sparsity Constraint.** Lock via sparsity:
$$\|\theta\|_0 \leq k$$

**Enforcement:** Magnitude pruning maintains sparsity lock:
$$\theta_i \mapsto 0 \text{ if } |\theta_i| < \tau$$

**Reference:** Han, S., Pool, J., et al. (2015). Learning both weights and connections. *NeurIPS*.

### Step 5: Spectral Locks

**Spectral Norm Constraint.** Lock via Lipschitz bound:
$$\|W_\ell\|_2 \leq L_\ell$$

**Enforcement:** Spectral normalization:
$$W \mapsto W / \|W\|_2 \cdot L_\ell$$

**Reference:** Miyato, T., et al. (2018). Spectral normalization for GANs. *ICLR*.

### Step 6: Energy Locks

**Loss Barrier Lock.** If $\mathcal{L}(B) > \mathcal{L}(\theta) + \Delta$:
$$\text{Gradient descent cannot reach } B$$

(since loss is non-increasing).

**Certificate:** The gap $\Delta$ and monotonicity of training.

### Step 7: Lock Factory Construction

**Lock Factory Algorithm:**

```python
def lock_factory(theta, bad_region, soft_certs):
    """Construct locks preventing access to bad region."""
    locks = []

    # Topological lock (architecture fixed)
    locks.append(ArchitectureLock(theta.architecture))

    # Frozen layer lock
    if 'frozen_layers' in soft_certs:
        locks.append(FrozenLayerLock(soft_certs['frozen_layers']))

    # Capacity lock
    if 'sparsity' in soft_certs:
        locks.append(SparsityLock(soft_certs['sparsity']))

    # Spectral lock
    if 'lipschitz' in soft_certs:
        locks.append(SpectralLock(soft_certs['lipschitz']))

    # Energy lock (check if loss separates)
    L_current = loss(theta)
    L_bad = min_loss_in_region(bad_region)
    if L_current < L_bad:
        locks.append(EnergyLock(L_current, L_bad))

    return CompositeLock(locks)
```

### Step 8: Lock Verification

**Certificate Structure:**
```python
class LockCertificate:
    lock_type: str
    invariant_current: float
    invariant_required: float
    proof_of_gap: str  # Why invariant blocks bad region
    preservation: str  # Why training preserves invariant
```

**Verification:** Check that invariant is maintained at each step.

### Step 9: Composite Locks

**Multiple Locks.** Combine for robust blocking:
- **All:** All locks must hold (intersection)
- **Any:** Any lock suffices (union)

**Example:** Frozen layers + spectral norm + sparsity = robust lock.

### Step 10: Compilation Theorem

**Theorem (Lock Factory):**

1. **Inputs:** Configuration $\theta$, bad set $B$, soft certificates
2. **Outputs:** Lock $\text{Lock}_B$ with certificate
3. **Guarantees:**
   - If invariant separates $\theta$ from $B$, lock exists
   - Certificate is verifiable during training
   - Lock is sound (no false blocks)

**Applications:**
- Transfer learning (freeze pretrained)
- Continual learning (protect old knowledge)
- Constrained optimization (enforce bounds)
- Robustness (Lipschitz constraints)

---

## Key AI/ML Techniques Used

1. **Frozen Gradients:**
   $$\nabla_{\theta_{\text{frozen}}} = 0$$

2. **Spectral Normalization:**
   $$W \mapsto W / \|W\|_2$$

3. **Sparsity Enforcement:**
   $$\|\theta\|_0 \leq k$$

4. **Loss Monotonicity:**
   $$\mathcal{L}(\theta_t) \leq \mathcal{L}(\theta_0)$$

---

## Literature References

- He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning. *CVPR*.
- Yosinski, J., et al. (2014). How transferable are features? *NeurIPS*.
- Han, S., Pool, J., et al. (2015). Learning both weights and connections. *NeurIPS*.
- Miyato, T., et al. (2018). Spectral normalization for GANs. *ICLR*.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting. *PNAS*.
