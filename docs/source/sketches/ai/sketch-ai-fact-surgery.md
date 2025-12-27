---
title: "FACT-Surgery - AI/RL/ML Translation"
---

# FACT-Surgery: Surgical Fine-tuning Factory

## Overview

The surgery schema factory constructs procedures for modifying neural networks at identified problem points—replacing failing components with standard modules from a library of known-good architectures.

**Original Theorem Reference:** {prf:ref}`mt-fact-surgery`

---

## AI/RL/ML Statement

**Theorem (Surgery Schema Factory, ML Form).**
There exists a factory $\mathcal{F}_{\text{surg}}$ that, given:
- Problematic network $f_\theta$ with identified issues $\Sigma$
- Canonical library $\mathcal{L}$ of replacement modules
- Soft certificates (performance bounds)

produces surgery schema $\mathcal{S}$ with:

1. **Profile Selection:** Replacement module $V \in \mathcal{L}$ matching problem type

2. **Scale Selection:** Scope of replacement (layer, block, connection)

3. **Operator Construction:** Cut-and-paste procedure

4. **Output:** $f_{\theta'}$ with $\text{issues}(f_{\theta'}) \subsetneq \text{issues}(f_\theta)$

**Corollary (Pruning Surgery).**
Removing dead neurons and replacing with fresh initializations is a surgery that can restore gradient flow.

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Singularity $\Sigma$ | Dead neurons, bottlenecks | $\nabla_\theta \mathcal{L} \approx 0$ |
| Surgery schema | Modification procedure | (Identify, Remove, Replace) |
| Tangent cone | Local behavior | Jacobian structure |
| Library $\mathcal{L}$ | Module library | ResBlock, Attention, etc. |
| Excision | Layer removal | Delete from computation graph |
| Replacement | Module insertion | Add new component |
| Energy drop | Performance improvement | $\mathcal{L}' < \mathcal{L}$ |
| Scale selection | Modification scope | Layer vs block vs network |

---

## Neural Network Surgery Framework

### Problem Detection

**Definition.** Identify problematic regions via:
- Dead neurons: $h_j = 0$ always
- Bottlenecks: $\text{rank}(W) \ll \min(d_{in}, d_{out})$
- Gradient vanishing: $\|\nabla_\ell\| < \varepsilon$

### Connection to Architecture Modification

| ML Property | Hypostructure Property |
|-------------|------------------------|
| Dead neuron | Singularity point |
| Skip connection | Surgery bypass |
| Knowledge distillation | Profile replacement |
| Neural architecture search | Automated surgery |

---

## Proof Sketch

### Step 1: Problem Detection

**Detection Methods:**
1. **Activation analysis:** Monitor $h_\ell$ for constant values
2. **Gradient analysis:** Track $\nabla_\ell \mathcal{L}$ magnitudes
3. **Weight analysis:** Check $\|W_\ell\|$ and condition numbers

**Reference:** Raghu, M., et al. (2017). On the expressive power of deep neural networks. *ICML*.

```python
def detect_problems(model, data):
    """Detect problematic regions in network."""
    problems = []

    for layer in model.layers:
        # Check for dead neurons
        activations = collect_activations(layer, data)
        dead = (activations.std(dim=0) < 1e-6)
        if dead.any():
            problems.append(DeadNeuronProblem(layer, dead))

        # Check for vanishing gradients
        grads = collect_gradients(layer, data)
        if grads.norm() < 1e-8:
            problems.append(VanishingGradientProblem(layer))

        # Check for bottlenecks
        if hasattr(layer, 'weight'):
            rank = matrix_rank(layer.weight)
            if rank < 0.5 * min(layer.weight.shape):
                problems.append(BottleneckProblem(layer, rank))

    return problems
```

### Step 2: Module Library

**Canonical Modules:**
1. **ResBlock:** $h' = h + f(h)$ - gradient highway
2. **Attention:** $h' = \text{Attn}(h)$ - dynamic routing
3. **BatchNorm:** $h' = \gamma \frac{h - \mu}{\sigma} + \beta$ - normalization
4. **Dropout:** $h' = h \odot m$ - regularization

**Reference:** He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning. *CVPR*.

### Step 3: Profile Matching

**Matching Algorithm:**
```python
def match_to_library(problem, library):
    """Find best replacement module for problem."""
    if isinstance(problem, VanishingGradientProblem):
        return library['ResBlock']  # Skip connection fixes gradients

    if isinstance(problem, DeadNeuronProblem):
        return library['ReluReinit']  # Reinitialize with LeakyReLU

    if isinstance(problem, BottleneckProblem):
        return library['Expander']  # Add wider layer

    return None
```

### Step 4: Scale Selection

**Scope of Surgery:**
- **Neuron-level:** Replace individual units
- **Layer-level:** Replace entire layer
- **Block-level:** Replace residual block
- **Module-level:** Replace major component (encoder, head)

**Reference:** Chen, T., Goodfellow, I., Shlens, J. (2016). Net2Net. *ICLR*.

### Step 5: Cut Operation

**Excision:** Remove problematic component from computation graph:
```python
def excise(model, problem_location):
    """Remove problematic component."""
    # Disconnect from graph
    for predecessor in problem_location.predecessors:
        predecessor.remove_successor(problem_location)
    for successor in problem_location.successors:
        successor.remove_predecessor(problem_location)

    # Store boundary connections
    boundary = {
        'input': problem_location.predecessors,
        'output': problem_location.successors
    }
    return model, boundary
```

### Step 6: Paste Operation

**Insertion:** Add replacement module:
```python
def paste(model, boundary, replacement):
    """Insert replacement module."""
    # Connect to predecessors
    for predecessor in boundary['input']:
        predecessor.add_successor(replacement)
        replacement.add_predecessor(predecessor)

    # Connect to successors
    for successor in boundary['output']:
        successor.add_predecessor(replacement)
        replacement.add_successor(successor)

    # Initialize replacement weights
    replacement.initialize(method='kaiming')

    return model
```

### Step 7: Surgery Operator

**Complete Surgery:**
```python
def surgery_operator(model, problem, library):
    """Perform complete surgery."""
    # Step 1: Match problem to replacement
    replacement = match_to_library(problem, library)

    # Step 2: Excise problem
    model, boundary = excise(model, problem.location)

    # Step 3: Paste replacement
    model = paste(model, boundary, replacement)

    # Step 4: Verify improvement
    assert verify_gradient_flow(model)
    assert verify_no_dead_neurons(model)

    return model
```

### Step 8: Factory Construction

**Surgery Factory:**
```python
def surgery_factory(model, library, soft_certs):
    """Factory for network surgery."""
    # Detect all problems
    problems = detect_problems(model, soft_certs['data'])

    surgeries = []
    for problem in problems:
        # Extract local behavior
        local_behavior = analyze_local(model, problem.location)

        # Match to library
        replacement = match_to_library(problem, library)
        if replacement is None:
            return SURGERY_FAILED  # No matching module

        # Select scale
        scale = select_scale(problem, model)

        # Construct surgery
        surgery = SurgerySchema(
            problem=problem,
            replacement=replacement,
            scale=scale
        )
        surgeries.append(surgery)

    return surgeries
```

### Step 9: Verification

**Surgery Verification:**
1. **Gradient flow:** $\|\nabla_\ell\| > 0$ for all layers
2. **No dead neurons:** $\text{Var}(h_j) > 0$ for all neurons
3. **Performance:** $\mathcal{L}(f_{\theta'}) \leq \mathcal{L}(f_\theta)$
4. **Function preservation:** $\|f_{\theta'} - f_\theta\|$ bounded (if desired)

### Step 10: Compilation Theorem

**Theorem (Surgery Factory):**

1. **Inputs:** Problematic network, library, soft certificates
2. **Outputs:** Surgery schema
3. **Guarantees:**
   - If problems match library, surgery succeeds
   - Surgery reduces problem set
   - Performance does not degrade significantly

**Applications:**
- Fixing dying ReLU with LeakyReLU
- Adding skip connections for gradient flow
- Replacing bottleneck layers
- Neural architecture search refinement

---

## Key AI/ML Techniques Used

1. **Dead Neuron Detection:**
   $$\text{Var}(h_j) < \varepsilon \implies j \text{ is dead}$$

2. **Gradient Flow Analysis:**
   $$\|\nabla_\ell\| > 0 \implies \text{healthy gradient}$$

3. **Module Replacement:**
   $$f_{\theta'} = \text{Paste}(\text{Excise}(f_\theta), V)$$

4. **Performance Bound:**
   $$\mathcal{L}(f_{\theta'}) \leq \mathcal{L}(f_\theta) + C\varepsilon$$

---

## Literature References

- He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning. *CVPR*.
- Chen, T., Goodfellow, I., Shlens, J. (2016). Net2Net. *ICLR*.
- Raghu, M., et al. (2017). Expressive power of deep neural networks. *ICML*.
- Maas, A., Hannun, A., Ng, A. (2013). Rectifier nonlinearities. *ICML Workshop*.
- Zoph, B., Le, Q. V. (2017). Neural architecture search. *ICLR*.
