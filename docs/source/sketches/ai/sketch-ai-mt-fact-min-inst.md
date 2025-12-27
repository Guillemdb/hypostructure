# FACT-MinimalInstantiation: Model Compression and Minimal Sufficient Networks

## AI/RL/ML Statement

### Original Statement (Hypostructure)

*Reference: mt-fact-min-inst*

To instantiate a Hypostructure for system $S$ using the thin object formalism, the user provides only:
1. The Space $\mathcal{X}^{\text{thin}} = (\mathcal{X}, d, \mu)$
2. The Energy $\Phi^{\text{thin}} = (F, \nabla, \alpha)$
3. The Dissipation $\mathfrak{D}^{\text{thin}} = (R, \beta)$
4. The Symmetry Group $G^{\text{thin}} = (\text{Grp}, \rho, \mathcal{S})$

The Framework automatically derives all required components (profiles, admissibility, regularization, topology, bad sets), reducing user burden from ~30 components to 10 primitive inputs.

---

## AI/RL/ML Formulation

### Setup

Consider a deep learning system where:

- **State space:** Weight space $\mathcal{W} \subset \mathbb{R}^d$ or representation space $\mathcal{Z}$
- **Height/Energy:** Value function $V(s)$ or loss function $\mathcal{L}(\theta)$
- **Dissipation:** Policy $\pi(a|s)$ or gradient dynamics $\dot{\theta} = -\nabla \mathcal{L}(\theta)$
- **Symmetry Group:** Weight permutation symmetries, scaling invariances
- **Full model:** Dense network with $d \gg 1$ parameters
- **Minimal model:** Pruned/compressed network with $k \ll d$ effective parameters

The "minimal instantiation" principle states that a complex model can be reduced to a minimal sufficient representation that preserves all essential computational capabilities.

### Statement (AI/RL/ML Version)

**Theorem (Minimal Sufficient Model).** Let $\mathcal{N}_{\text{full}} = (\mathcal{W}, \mathcal{L}, \nabla, \pi_{\text{init}})$ be a neural network with:
- Weight space $\mathcal{W} \subset \mathbb{R}^d$
- Loss function $\mathcal{L}: \mathcal{W} \to \mathbb{R}_{\geq 0}$
- Gradient operator $\nabla: \mathcal{W} \to \mathbb{R}^d$
- Initialization distribution $\pi_{\text{init}}$

Then there exists a **minimal sufficient subnetwork** $\mathcal{N}_{\text{min}} \subset \mathcal{N}_{\text{full}}$ such that:

1. **Compression:** $|\mathcal{N}_{\text{min}}| \ll |\mathcal{N}_{\text{full}}|$ (parameter count reduction)
2. **Performance Preservation:** $\mathcal{L}(\mathcal{N}_{\text{min}}) \leq \mathcal{L}(\mathcal{N}_{\text{full}}) + \epsilon$
3. **Trainability:** $\mathcal{N}_{\text{min}}$ can be trained from scratch to match $\mathcal{N}_{\text{full}}$
4. **Automatic Derivation:** The remaining structure (topology, critical points, regularization) is derived from the minimal specification

**Formal Statement:** For a network with $d$ parameters achieving loss $\mathcal{L}^*$, there exists a minimal witness $\mathcal{N}_{\text{min}}$ with:

$$|\mathcal{N}_{\text{min}}| = O\left(\frac{\text{VC}(\mathcal{N}_{\text{full}})}{\epsilon^2}\right) \ll d$$

where $\text{VC}(\cdot)$ denotes the effective VC dimension, and:

$$\mathcal{L}(\mathcal{N}_{\text{min}}) \leq \mathcal{L}^* + O(\epsilon)$$

---

## Terminology Translation Table

| Hypostructure Term | AI/RL/ML Equivalent |
|--------------------|---------------------|
| Thin object $\mathcal{X}^{\text{thin}}$ | Minimal architecture specification (layer sizes, activation functions) |
| Full object $\mathcal{X}^{\text{full}}$ | Complete trained network with all derived structures |
| Space $(\mathcal{X}, d, \mu)$ | Weight space with metric and initialization distribution |
| Height $\Phi = (F, \nabla, \alpha)$ | Loss function $\mathcal{L}$, gradients, scaling dimension |
| Dissipation $\mathfrak{D} = (R, \beta)$ | Training dynamics, learning rate, regularization |
| Symmetry Group $G$ | Weight permutation symmetries, neuron relabeling |
| Scaling subgroup $\mathcal{S}$ | Weight rescaling invariance (batch norm, weight decay) |
| Minimal instantiation | Lottery ticket hypothesis / winning ticket |
| Thin Kernel | Sparse subnetwork / pruned mask |
| Profile extraction | Feature extraction, representation learning |
| Canonical library $\mathcal{L}_T$ | Known optimal architectures (ResNet, Transformer blocks) |
| Admissibility predicate | Pruning criterion (magnitude, gradient, Hessian) |
| Surgery operator | Pruning + fine-tuning operation |
| Capacity $\text{Cap}(\Sigma)$ | Effective parameter count, model complexity |
| Bad set $\mathcal{X}_{\text{bad}}$ | Dead neurons, unstable weights, gradient vanishing regions |
| Łojasiewicz exponent $\theta$ | Convergence rate, loss landscape curvature |
| Sector map | Layer-wise structure, modular decomposition |
| Framework derivation | Knowledge distillation, architecture search |

---

## Proof Sketch

### Step 1: Existence of Minimal Witnesses (Lottery Ticket Hypothesis)

**Claim (Lottery Ticket Existence):** For any trained network $\mathcal{N}_{\text{full}}$, there exists a sparse subnetwork (winning ticket) $\mathcal{N}_{\text{min}} \subset \mathcal{N}_{\text{full}}$ that can match the full network's performance when trained in isolation.

**Construction:**

**Step 1.1 (Initial Training):** Train full network $\mathcal{N}_{\text{full}}$ from initialization $\theta_0$ to optimum $\theta^*$:
$$\theta^* = \arg\min_\theta \mathcal{L}(\theta)$$

**Step 1.2 (Mask Identification):** Identify binary mask $m \in \{0,1\}^d$ via magnitude pruning:
$$m_i = \mathbf{1}[|\theta_i^*| > \tau]$$

where $\tau$ is chosen so that $\|m\|_0 = k \ll d$.

**Step 1.3 (Rewinding):** Reset unpruned weights to their initial values:
$$\theta_{\text{min}} = m \odot \theta_0$$

**Step 1.4 (Retraining):** Train $\mathcal{N}_{\text{min}}$ with fixed mask:
$$\theta_{\text{min}}^* = \arg\min_{\theta: \text{supp}(\theta) \subseteq \text{supp}(m)} \mathcal{L}(\theta)$$

**Theorem (Frankle-Carlin 2019):** Under suitable conditions on the initialization and training dynamics:
$$\mathcal{L}(\theta_{\text{min}}^*) \leq \mathcal{L}(\theta^*) + \epsilon$$

with $\|m\|_0 / d$ as low as 1-10%.

**Connection to Hypostructure:** The mask $m$ is the "thin kernel" that specifies minimal data. The full network structure (gradients, critical points, etc.) is derived from training this minimal specification.

---

### Step 2: Automatic Structure Derivation (Thin-to-Full Expansion)

**Claim:** From the minimal specification (architecture + mask), all required training infrastructure is automatically derived.

**Step 2.1 (Topology Derivation from Architecture):**

Given thin specification:
- Layer dimensions: $(n_0, n_1, \ldots, n_L)$
- Activation functions: $(\sigma_1, \ldots, \sigma_L)$
- Connectivity pattern: mask $m$

The framework derives:
- **Weight space:** $\mathcal{W} = \{W_\ell \in \mathbb{R}^{n_\ell \times n_{\ell-1}} : \ell = 1, \ldots, L\}$
- **Forward pass:** $f_\theta(x) = \sigma_L(W_L \cdots \sigma_1(W_1 x))$
- **Gradient computation:** Backpropagation through the defined graph
- **Sector structure:** Layer-wise decomposition, skip connections

**Step 2.2 (Critical Point Structure from Loss Landscape):**

Given loss function $\mathcal{L}(\theta)$, the framework automatically computes:
- **Gradient field:** $\nabla \mathcal{L}(\theta)$
- **Hessian:** $H(\theta) = \nabla^2 \mathcal{L}(\theta)$
- **Critical points:** $\mathcal{C} = \{\theta : \nabla \mathcal{L}(\theta) = 0\}$
- **Stiffness (Łojasiewicz exponent):** From spectral gap of Hessian at minima

**Step 2.3 (Bad Set Detection):**

The singular locus (bad set) is identified as:
$$\mathcal{W}_{\text{bad}} = \{W : \text{ReLU units dead}\} \cup \{W : \|W\| \to \infty\} \cup \{W : \|\nabla \mathcal{L}\| \to \infty\}$$

Components:
- **Dead neurons:** Units with zero activation on training data
- **Exploding weights:** $\|W_\ell\|_F > \tau_{\text{explode}}$
- **Gradient pathologies:** Vanishing or exploding gradients

**Step 2.4 (Regularization Derivation):**

From dissipation parameters (learning rate $\eta$, weight decay $\lambda$), derive:
- **Implicit regularization:** SGD prefers flat minima
- **Explicit regularization:** $\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \lambda \|\theta\|^2$
- **Łojasiewicz inequality:** $\|\nabla \mathcal{L}(\theta)\| \geq C |\mathcal{L}(\theta) - \mathcal{L}^*|^{1-\theta}$

---

### Step 3: Profile Classification (Architecture Types)

**Claim:** Network architectures classify into canonical families, analogous to profile classification in the Hypostructure.

**Canonical Library for Neural Networks:**

| Architecture Type | Profile Class | Key Property |
|-------------------|---------------|--------------|
| MLPs | Generic profiles | Universal approximation |
| CNNs | Translation-equivariant | Weight sharing, local connectivity |
| ResNets | Skip-connection profiles | Gradient flow preservation |
| Transformers | Attention profiles | Global dependency modeling |
| RNNs/LSTMs | Recurrent profiles | Sequential memory |
| GNNs | Graph-equivariant | Permutation invariance |

**Symmetry-Based Classification:**

For symmetry group $G$ acting on input space:
$$\rho: G \times \mathcal{X} \to \mathcal{X}$$

The canonical architecture is the one equivariant to $G$:
$$f_\theta(g \cdot x) = g \cdot f_\theta(x) \quad \forall g \in G$$

Examples:
- $G = \text{Translations}$ $\Rightarrow$ CNNs
- $G = \text{Permutations}$ $\Rightarrow$ GNNs, DeepSets
- $G = \text{Rotations}$ $\Rightarrow$ Steerable CNNs

---

### Step 4: Admissibility and Pruning (Surgery Criterion)

**Claim:** Pruning decisions correspond to surgery admissibility in the Hypostructure.

**Pruning Admissibility Predicate:**

A weight $\theta_i$ is prunable (surgery is admissible) if:

1. **Magnitude criterion:** $|\theta_i| < \tau_{\text{mag}}$
2. **Gradient criterion:** $|\partial \mathcal{L} / \partial \theta_i| < \tau_{\text{grad}}$
3. **Hessian criterion:** $|H_{ii}| < \tau_{\text{hess}}$ (second derivative is small)
4. **Capacity criterion:** Removing $\theta_i$ does not reduce effective capacity below threshold

**Capacity Bound (Pruning Budget):**

$$\text{Cap}(\Sigma_{\text{prune}}) = \sum_{i \in \Sigma_{\text{prune}}} |\theta_i|^2 < \epsilon_{\text{adm}}$$

If the total "mass" of pruned weights is small, the network can be fine-tuned to recover performance.

**Surgery Operator (Pruning + Fine-tuning):**

The pruning surgery is:
$$\theta' = \text{Prune}(\theta, m) = m \odot \theta$$

followed by fine-tuning:
$$\theta'^* = \arg\min_{\theta': \text{supp}(\theta') \subseteq \text{supp}(m)} \mathcal{L}(\theta')$$

**Energy Control (Performance Bound):**

$$\mathcal{L}(\theta'^*) \leq \mathcal{L}(\theta) + C \cdot \text{Cap}(\Sigma_{\text{prune}})$$

This is analogous to the surgery energy bound in the Hypostructure.

---

### Step 5: User Burden Reduction Count

**Full Network Specification (Without Automation):**

To specify a complete deep learning system manually, one would need:

1. Architecture (layers, dimensions, activations)
2. Weight initialization scheme
3. Loss function and its gradient
4. Optimizer and learning rate schedule
5. Regularization terms
6. Data augmentation strategy
7. Normalization layers
8. Critical point analysis
9. Convergence criteria
10. Pruning/compression strategy
11. Quantization scheme
12. Deployment format
13. ...and many more

**Total: ~30+ design decisions**

**Minimal Specification (With Automation):**

Using modern AutoML / Neural Architecture Search:

1. **Task specification:** (input type, output type, loss)
2. **Data:** $(X_{\text{train}}, Y_{\text{train}})$
3. **Compute budget:** (FLOPS, memory, time)
4. **Performance target:** $\epsilon$

**Total: ~4-5 primitive inputs**

The framework (AutoML, NAS, pruning algorithms) derives everything else.

**Reduction Factor:** $30 \to 5$ = $6\times$ reduction (analogous to $30 \to 10$ in hypostructure)

---

## Connections to Classical Results

### 1. Lottery Ticket Hypothesis (Frankle & Carlin, 2019)

**Statement:** Dense, randomly-initialized neural networks contain sparse subnetworks (winning tickets) that, when trained in isolation, reach test accuracy comparable to the original network in a similar number of iterations.

**Connection to Minimal Instantiation:**
- **Winning ticket** = Thin Kernel
- **Full trained network** = Full Hypostructure
- **Mask identification** = Profile extraction
- **Rewinding** = Minimal witness construction
- **Matching performance** = Interface permit satisfaction

**Theorem (Lottery Ticket, Formal):** For network $f_\theta$ with $d$ parameters achieving accuracy $\alpha$ after $T$ iterations, there exists a subnetwork $f_{m \odot \theta_0}$ with $\|m\|_0 \ll d$ such that:
- After $T'$ iterations (with $T' \leq cT$), achieves accuracy $\alpha - \epsilon$
- The subnetwork $m$ is identifiable from the trained weights $\theta_T$

### 2. Knowledge Distillation (Hinton et al., 2015)

**Statement:** A small "student" network can be trained to match the performance of a large "teacher" network by learning from the teacher's soft predictions.

**Connection to Minimal Instantiation:**
- **Teacher network** = Full Hypostructure
- **Student network** = Thin object instantiation
- **Soft labels** = Derived structure (profiles, sectors)
- **Distillation loss** = Interface requirements

**Theorem (Distillation):** Let $f_T$ be a teacher with $d_T$ parameters and $f_S$ be a student with $d_S \ll d_T$. Training $f_S$ on:
$$\mathcal{L}_{\text{distill}} = (1-\alpha) \mathcal{L}_{\text{hard}}(f_S, y) + \alpha \mathcal{L}_{\text{soft}}(f_S, f_T)$$

achieves:
$$\text{Acc}(f_S) \geq \text{Acc}(f_T) - O\left(\sqrt{\frac{d_S}{n}}\right)$$

### 3. Pruning Algorithms (LeCun et al., 1990; Han et al., 2015)

**Optimal Brain Damage / Surgeon:**

Prune weights based on second-derivative information:
$$\text{Saliency}_i = \frac{1}{2} H_{ii} \theta_i^2$$

Prune weights with lowest saliency (least impact on loss).

**Connection to Minimal Instantiation:**
- **Saliency** = Capacity contribution
- **Pruning threshold** = Admissibility predicate
- **Iterative pruning** = Iterated surgery
- **Fine-tuning** = Post-surgery regularization

**Theorem (Optimal Pruning):** Removing weight $\theta_i$ changes the loss by:
$$\Delta \mathcal{L} \approx \frac{1}{2} H_{ii} \theta_i^2 + O(\theta_i^3)$$

Minimizing total loss increase under sparsity constraint is equivalent to selecting weights with smallest saliency.

### 4. Neural Architecture Search (Zoph & Le, 2017; Liu et al., 2019)

**Statement:** Optimal architectures can be automatically discovered from minimal task specification.

**Connection to Minimal Instantiation:**
- **Search space** = State space $\mathcal{X}$
- **Reward signal** = Negative loss / validation accuracy
- **Discovered architecture** = Derived full structure
- **Search algorithm** = Framework derivation mechanism

**DARTS (Differentiable Architecture Search):**

Parameterize architecture as continuous relaxation:
$$\bar{o}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o)}{\sum_{o'} \exp(\alpha_{o'})} o(x)$$

Optimize architecture parameters $\alpha$ jointly with weights $\theta$.

**Connection:** This automates the "thin-to-full expansion" by learning the optimal structure from data.

### 5. Quantization and Compression (Courbariaux et al., 2015; Jacob et al., 2018)

**Statement:** Neural networks can be compressed by reducing numerical precision of weights.

**Quantization Map:**
$$Q(w) = \text{round}(w / \Delta) \cdot \Delta$$

where $\Delta$ is the quantization step size.

**Connection to Minimal Instantiation:**
- **Full precision** = Full Hypostructure
- **Quantized weights** = Thin representation
- **Quantization error** = Approximation error $\epsilon$
- **Calibration** = Admissibility verification

**Theorem (Quantization Error):** For network $f$ with $d$ parameters quantized to $b$ bits:
$$\|f_{\text{quant}} - f\|_\infty \leq O\left(\frac{d \cdot \text{Lip}(f)}{2^b}\right)$$

### 6. Information Bottleneck (Tishby & Zaslavsky, 2015)

**Statement:** Deep learning performs optimal compression of input information relevant to the output.

**Information Bottleneck Objective:**
$$\min_{p(z|x)} I(X; Z) - \beta I(Z; Y)$$

Find minimal representation $Z$ that preserves information about $Y$.

**Connection to Minimal Instantiation:**
- **Compressed representation $Z$** = Thin kernel
- **Mutual information $I(X;Z)$** = Representation complexity (capacity)
- **Task information $I(Z;Y)$** = Performance preservation
- **Bottleneck layer** = Minimal sufficient statistic

**Theorem (Deep IB):** In deep networks, representations evolve through two phases:
1. **Fitting phase:** $I(Z; Y)$ increases (learn task)
2. **Compression phase:** $I(X; Z)$ decreases (minimize complexity)

The final representation is (approximately) the minimal sufficient statistic.

---

## Implementation Notes

### Pruning Pipeline (Minimal Instantiation in Practice)

```python
def minimal_instantiation(model_full, data, target_sparsity):
    """
    Find minimal sufficient subnetwork via pruning.

    Corresponds to FACT-MinInst: Extract thin kernel from full model.

    Args:
        model_full: Trained dense network (full Hypostructure)
        data: Training data
        target_sparsity: Fraction of weights to remove

    Returns:
        model_min: Minimal sufficient subnetwork (thin kernel)
        mask: Binary mask identifying winning ticket
    """
    # Step 1: Train full model (establish full Hypostructure)
    model_full = train(model_full, data)

    # Step 2: Identify prunable weights (admissibility predicate)
    saliency = compute_saliency(model_full)  # Magnitude, gradient, or Hessian
    threshold = np.percentile(saliency, target_sparsity * 100)
    mask = saliency > threshold

    # Step 3: Extract minimal subnetwork (thin kernel)
    model_min = apply_mask(model_full, mask)

    # Step 4: Verify admissibility (capacity check)
    capacity = compute_capacity(mask, model_full)
    if capacity > epsilon_adm:
        raise AdmissibilityViolation("Pruning too aggressive")

    # Step 5: Fine-tune (surgery + regularization)
    model_min = finetune(model_min, data)

    return model_min, mask


def compute_saliency(model, method='magnitude'):
    """
    Compute saliency scores for pruning decision.

    Analogous to admissibility predicate evaluation.
    """
    if method == 'magnitude':
        # |w| pruning (simplest)
        return {name: param.abs() for name, param in model.named_parameters()}

    elif method == 'gradient':
        # |w * dL/dw| (first-order Taylor)
        return {name: (param * param.grad).abs()
                for name, param in model.named_parameters()}

    elif method == 'hessian':
        # Fisher information / Hessian diagonal
        fisher = compute_fisher(model)
        return {name: (param ** 2 * fisher[name]).abs()
                for name, param in model.named_parameters()}
```

### Knowledge Distillation (Structure Transfer)

```python
def distill_knowledge(teacher, student, data, temperature=4.0):
    """
    Transfer knowledge from large model to small model.

    Corresponds to thin-to-full expansion: derive student structure
    from teacher's learned representations.

    Args:
        teacher: Large trained model (full Hypostructure)
        student: Small model (thin specification)
        data: Training data
        temperature: Softmax temperature for soft labels

    Returns:
        student: Trained student matching teacher performance
    """
    teacher.eval()

    for x, y in data:
        # Teacher's soft predictions (derived structure)
        with torch.no_grad():
            teacher_logits = teacher(x)
            soft_labels = F.softmax(teacher_logits / temperature, dim=-1)

        # Student predictions
        student_logits = student(x)

        # Distillation loss (interface requirement satisfaction)
        loss_soft = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            soft_labels,
            reduction='batchmean'
        ) * (temperature ** 2)

        loss_hard = F.cross_entropy(student_logits, y)

        # Combined loss
        loss = 0.7 * loss_soft + 0.3 * loss_hard

        loss.backward()
        optimizer.step()

    return student
```

### Lottery Ticket Search

```python
def find_winning_ticket(model_init, data, target_sparsity, rewinding_step=0):
    """
    Find winning ticket via iterative magnitude pruning.

    Implements the Lottery Ticket Hypothesis search procedure.

    Args:
        model_init: Initial (random) model weights
        data: Training data
        target_sparsity: Target sparsity level
        rewinding_step: Iteration to rewind weights to (0 = initial)

    Returns:
        winning_ticket: Sparse subnetwork trainable from scratch
    """
    model = copy.deepcopy(model_init)
    mask = {name: torch.ones_like(param)
            for name, param in model.named_parameters()}

    # Save initialization for rewinding
    init_weights = {name: param.clone()
                    for name, param in model_init.named_parameters()}

    current_sparsity = 0.0
    prune_rate = 0.2  # Prune 20% of remaining weights each round

    while current_sparsity < target_sparsity:
        # Train with current mask
        model = train_with_mask(model, data, mask)

        # Prune smallest magnitude weights (globally)
        all_weights = torch.cat([
            (param * mask[name]).abs().flatten()
            for name, param in model.named_parameters()
        ])
        threshold = torch.quantile(all_weights[all_weights > 0],
                                   prune_rate)

        # Update mask
        for name, param in model.named_parameters():
            mask[name] *= (param.abs() > threshold).float()

        current_sparsity = 1 - sum(m.sum() for m in mask.values()) / \
                               sum(m.numel() for m in mask.values())

        # Rewind weights to initialization
        for name, param in model.named_parameters():
            param.data = init_weights[name] * mask[name]

    return model, mask
```

### Verification Certificate

```python
def verify_minimal_instantiation(model_min, model_full, test_data, epsilon=0.01):
    """
    Verify that minimal model satisfies interface requirements.

    Produces certificate analogous to K_MinInst^+.
    """
    certificate = {
        'type': 'MinimalInstantiation',
        'status': 'pending',
        'checks': {}
    }

    # Check 1: Compression ratio
    params_full = sum(p.numel() for p in model_full.parameters())
    params_min = sum((p != 0).sum().item() for p in model_min.parameters())
    compression = 1 - params_min / params_full
    certificate['checks']['compression'] = {
        'full_params': params_full,
        'min_params': params_min,
        'compression_ratio': compression,
        'passed': compression > 0.5  # At least 50% compression
    }

    # Check 2: Performance preservation
    acc_full = evaluate(model_full, test_data)
    acc_min = evaluate(model_min, test_data)
    perf_gap = acc_full - acc_min
    certificate['checks']['performance'] = {
        'full_accuracy': acc_full,
        'min_accuracy': acc_min,
        'gap': perf_gap,
        'passed': perf_gap < epsilon
    }

    # Check 3: Trainability (can recover performance from scratch)
    model_retrain = reinitialize_and_train(model_min, test_data)
    acc_retrain = evaluate(model_retrain, test_data)
    certificate['checks']['trainability'] = {
        'retrained_accuracy': acc_retrain,
        'passed': acc_retrain >= acc_min - epsilon
    }

    # Overall status
    all_passed = all(c['passed'] for c in certificate['checks'].values())
    certificate['status'] = 'valid' if all_passed else 'failed'

    return certificate
```

---

## Literature

1. **Frankle, J. & Carlin, M. (2019).** "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." *ICLR 2019.* *Foundational work on winning tickets and minimal subnetworks.*

2. **Hinton, G., Vinyals, O., & Dean, J. (2015).** "Distilling the Knowledge in a Neural Network." *NeurIPS 2014 Workshop.* *Knowledge distillation from large to small models.*

3. **Han, S., Pool, J., Tran, J., & Dally, W.J. (2015).** "Learning both Weights and Connections for Efficient Neural Networks." *NeurIPS 2015.* *Magnitude-based pruning methodology.*

4. **LeCun, Y., Denker, J.S., & Solla, S.A. (1990).** "Optimal Brain Damage." *NeurIPS 1990.* *Second-order pruning based on Hessian information.*

5. **Hassibi, B. & Stork, D.G. (1993).** "Second Order Derivatives for Network Pruning: Optimal Brain Surgeon." *NeurIPS 1993.* *Optimal Brain Surgeon algorithm.*

6. **Liu, Z., Sun, M., Zhou, T., Huang, G., & Darrell, T. (2019).** "Rethinking the Value of Network Pruning." *ICLR 2019.* *Analysis of pruning effectiveness.*

7. **Zoph, B. & Le, Q.V. (2017).** "Neural Architecture Search with Reinforcement Learning." *ICLR 2017.* *Automated architecture discovery.*

8. **Liu, H., Simonyan, K., & Yang, Y. (2019).** "DARTS: Differentiable Architecture Search." *ICLR 2019.* *Efficient architecture search via gradient descent.*

9. **Courbariaux, M., Bengio, Y., & David, J.-P. (2015).** "BinaryConnect: Training Deep Neural Networks with Binary Weights." *NeurIPS 2015.* *Extreme quantization.*

10. **Jacob, B. et al. (2018).** "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR 2018.* *Practical quantization methods.*

11. **Tishby, N. & Zaslavsky, N. (2015).** "Deep Learning and the Information Bottleneck Principle." *IEEE ITW 2015.* *Information-theoretic view of compression.*

12. **Zhou, H. et al. (2019).** "Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask." *NeurIPS 2019.* *Analysis of lottery ticket structure.*

13. **Malach, E. et al. (2020).** "Proving the Lottery Ticket Hypothesis: Pruning is All You Need." *ICML 2020.* *Theoretical foundations of lottery tickets.*

14. **Frankle, J. et al. (2020).** "Linear Mode Connectivity and the Lottery Ticket Hypothesis." *ICML 2020.* *Geometric structure of winning tickets.*

15. **Evci, U. et al. (2020).** "Rigging the Lottery: Making All Tickets Winners." *ICML 2020.* *Dynamic sparse training without dense pretraining.*

---

## Summary

The FACT-MinimalInstantiation theorem, translated to AI/RL/ML, establishes that:

1. **Minimal sufficient models exist:** Large neural networks contain sparse subnetworks (winning tickets / thin kernels) that can match the full network's performance with dramatically fewer parameters.

2. **Structure is automatically derived:** From minimal specification (architecture + mask), all training infrastructure (gradients, critical points, regularization, optimization dynamics) is automatically constructed.

3. **Pruning corresponds to surgery:** Weight pruning decisions correspond to surgery admissibility predicates, with the pruned weight "capacity" controlling the performance degradation bound.

4. **Compression follows universal patterns:** Networks compress according to canonical patterns (lottery tickets, distillation, quantization), analogous to profile classification in the Hypostructure.

5. **User burden is dramatically reduced:** Modern AutoML and pruning algorithms reduce the specification burden from ~30 design decisions to ~5 primitive inputs, matching the 3x reduction claimed in FACT-MinInst.

This translation reveals that the hypostructure framework's Minimal Instantiation principle provides the mathematical foundation for neural network compression, unifying lottery tickets, knowledge distillation, pruning, and neural architecture search under a single theoretical framework based on minimal sufficient representations.
