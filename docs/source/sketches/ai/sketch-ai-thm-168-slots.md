---
title: "THM-168-SLOTS - AI/RL/ML Translation"
---

# THM-168-SLOTS: The 168 Training Regime Classification

## Overview

The 168 structural slots theorem establishes that all machine learning training problems can be classified into exactly 168 canonical types based on 8 problem families and 21 training phases. This classification determines optimal training strategies, architectures, and hyperparameters.

**Original Theorem Reference:** {prf:ref}`thm-168-slots`

---

## AI/RL/ML Statement

**Theorem (Training Regime Classification, ML Form).**
Every machine learning problem $\mathcal{P}$ maps to exactly one structural slot:

$$\text{Slot}(\mathcal{P}) = (\text{Family}, \text{Phase}) \in \{I, \ldots, VIII\} \times \{1, \ldots, 21\}$$

where:
- **8 Families:** Problem types (supervised, RL, generative, etc.)
- **21 Phases:** Training stages (initialization, warmup, main training, etc.)

**Total:** $8 \times 21 = 168$ canonical training regimes.

**Guarantee:** Each slot has:
- Optimal architecture pattern
- Recommended optimizer
- Hyperparameter ranges
- Expected convergence behavior

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| 8 Families (I-VIII) | 8 problem types | Supervised, RL, GAN, etc. |
| 21 Strata (nodes) | 21 training phases | Init, warmup, training stages |
| 168 structural slots | 168 training regimes | (Problem, Phase) pairs |
| Certificate type $K^+$ | Training outcome | Converged, diverged, etc. |
| Structural DNA | Training signature | Characteristic loss curve |
| Family assignment | Problem classification | What type of ML problem |
| Stratum determination | Phase identification | Which training stage |
| Periodic Table | Classification matrix | Problem × Phase table |

---

## The 8 ML Problem Families

### Classification by Objective

| Family | Name | Objective | Examples |
|--------|------|-----------|----------|
| I | Supervised Classification | $\min \mathbb{E}[\ell(f(x), y)]$ | Image classification, NLP |
| II | Supervised Regression | $\min \mathbb{E}[(f(x) - y)^2]$ | Forecasting, estimation |
| III | Reinforcement Learning | $\max \mathbb{E}[\sum_t \gamma^t r_t]$ | Games, robotics, control |
| IV | Generative Modeling | $\min D_{\text{KL}}(p_\theta \| p_{\text{data}})$ | VAE, flow, diffusion |
| V | Adversarial Learning | $\min_G \max_D V(G, D)$ | GANs, adversarial training |
| VI | Self-Supervised | $\min \mathcal{L}_{\text{pretext}}$ | Contrastive, masked prediction |
| VII | Meta-Learning | $\min \mathbb{E}_{\mathcal{T}}[\mathcal{L}_{\mathcal{T}}]$ | Few-shot, transfer |
| VIII | Multi-Task | $\min \sum_i w_i \mathcal{L}_i$ | Joint learning, MTL |

---

## The 21 Training Phases

### Phase Structure

| Phase | Name | Description |
|-------|------|-------------|
| 1-3 | Initialization | Weight init, architecture setup |
| 4-7 | Warmup | Learning rate warmup, exploration |
| 7a-7d | Stability | Gradient stabilization, normalization |
| 8-12 | Main Training | Core optimization, feature learning |
| 13-17 | Convergence | Fine-tuning, annealing, final optimization |

---

## Proof Sketch

### Step 1: Problem Family Classification

**Claim:** Every ML problem belongs to exactly one family.

**Classification Criteria:**

| Criterion | Family Determined By |
|-----------|---------------------|
| Labels available? | Supervised vs Self-supervised |
| Sequential decisions? | RL vs non-RL |
| Generative? | Generative vs Discriminative |
| Adversarial? | GAN vs non-adversarial |
| Multiple tasks? | Multi-task vs single |

**Algorithm:**
```
ClassifyFamily(problem):
    if has_labels(problem):
        if continuous_target: return Family.II  # Regression
        else: return Family.I  # Classification
    elif sequential_decisions: return Family.III  # RL
    elif generative: return Family.IV  # Generative
    elif adversarial: return Family.V  # GAN
    elif pretext_task: return Family.VI  # Self-supervised
    elif meta_objective: return Family.VII  # Meta-learning
    elif multiple_tasks: return Family.VIII  # Multi-task
```

### Step 2: Phase Identification

**Claim:** Training proceeds through identifiable phases.

**Phase Transitions:**

| Transition | Trigger | Action |
|------------|---------|--------|
| Init → Warmup | Training starts | Apply warmup schedule |
| Warmup → Main | Warmup complete | Full learning rate |
| Main → Converge | Loss plateau | Reduce LR, fine-tune |
| Converge → Done | Convergence | Stop training |

**Detection:**
$$\text{Phase} = \begin{cases}
\text{Warmup} & t < T_{\text{warmup}} \\
\text{Main} & |\Delta\mathcal{L}| > \epsilon \\
\text{Converge} & |\Delta\mathcal{L}| < \epsilon
\end{cases}$$

**Reference:** Smith, L. N. (2018). Cyclical learning rates. *WACV*.

### Step 3: Slot-Specific Strategies

**Claim:** Each slot has optimal training configuration.

**Example (Family I, Classification):**

| Phase | Learning Rate | Optimizer | Batch Size |
|-------|--------------|-----------|------------|
| 1-3 | 0 (init) | - | - |
| 4-7 | Linear warmup | SGD+momentum | Small |
| 8-12 | Constant high | Adam | Large |
| 13-17 | Cosine decay | SGD | Large |

**Example (Family III, RL):**

| Phase | Exploration | Update | Buffer |
|-------|-------------|--------|--------|
| 1-3 | Random | None | Fill |
| 4-7 | $\epsilon$-greedy high | Frequent | Growing |
| 8-12 | $\epsilon$-greedy decay | Standard | Full |
| 13-17 | Near-greedy | Fine-tune | Recent |

### Step 4: Architecture Patterns by Family

**Claim:** Optimal architecture depends on family.

| Family | Architecture Pattern |
|--------|---------------------|
| I (Classification) | Feature extractor + classifier head |
| II (Regression) | Feature extractor + regression head |
| III (RL) | Actor-critic, value network |
| IV (Generative) | Encoder-decoder, flow, diffusion |
| V (GAN) | Generator + discriminator |
| VI (Self-supervised) | Siamese, contrastive |
| VII (Meta) | Inner/outer loop structure |
| VIII (Multi-task) | Shared backbone + task heads |

### Step 5: Hyperparameter Ranges by Slot

**Claim:** Each slot has recommended hyperparameter ranges.

**Learning Rate Ranges:**

| Family | Phase 4-7 | Phase 8-12 | Phase 13-17 |
|--------|-----------|------------|-------------|
| I | 0.1-1.0 | 0.01-0.1 | 0.001-0.01 |
| III | 0.001-0.01 | 0.0001-0.001 | 0.00001 |
| IV | 0.001-0.01 | 0.0001-0.001 | 0.00001 |

**Batch Size Ranges:**

| Family | Small Data | Large Data |
|--------|------------|------------|
| I | 32-128 | 256-4096 |
| III | 32-256 | - |
| VI | 256-4096 | 4096-65536 |

### Step 6: Loss Curve Signatures

**Claim:** Each slot has characteristic loss curve.

**Signatures:**

| Family-Phase | Curve Shape |
|--------------|-------------|
| I-warmup | Steep initial drop |
| I-main | Exponential decay |
| I-converge | Plateau with oscillations |
| III-main | High variance, improving trend |
| V-all | Oscillating (adversarial) |

**Detection:** Match observed curve to known signatures for slot identification.

### Step 7: Failure Mode Classification

**Claim:** Each slot has characteristic failure modes.

| Family | Common Failures |
|--------|-----------------|
| I | Overfitting, underfitting |
| III | Sample inefficiency, instability |
| IV | Mode collapse, posterior collapse |
| V | Mode collapse, oscillation |
| VII | Negative transfer |

### Step 8: Transfer Between Slots

**Claim:** Knowledge transfers between related slots.

**Transfer Matrix:** Compatibility between families:

| From \ To | I | II | III | IV | V |
|-----------|---|----|----|----|----|
| I | ✓✓ | ✓✓ | ✓ | ✓ | ✓ |
| II | ✓✓ | ✓✓ | ✓ | ✓ | ✓ |
| III | ✓ | ✓ | ✓✓ | ✗ | ✗ |
| IV | ✓ | ✓ | ✗ | ✓✓ | ✓ |

### Step 9: Automatic Slot Detection

**Claim:** Slot can be automatically determined.

**Algorithm:**
```
DetectSlot(problem, training_state):
    family = ClassifyFamily(problem)
    phase = DetectPhase(training_state)
    return (family, phase)
```

**Features for Detection:**
- Loss value and gradient
- Weight statistics
- Activation distributions
- Training time elapsed

### Step 10: Compilation Theorem

**Theorem (168 Training Regime Classification):**

1. **Existence:** Every ML problem maps to exactly one slot
2. **Uniqueness:** Slot determined by (Family, Phase)
3. **Completeness:** All 168 slots are achievable
4. **Optimality:** Each slot has optimal strategy

**Classification Certificate:**
$$K_{168} = (\text{Family}, \text{Phase}, \text{Strategy}, \text{Expected Outcome})$$

**Applications:**
- Automatic hyperparameter selection
- Training monitoring
- Transfer learning matching
- Curriculum design

---

## Key AI/ML Techniques Used

1. **Family Classification:**
   $$\text{Family}(\mathcal{P}) \in \{I, \ldots, VIII\}$$

2. **Phase Detection:**
   $$\text{Phase} = f(\mathcal{L}_t, \nabla\mathcal{L}_t, t)$$

3. **Slot Assignment:**
   $$\text{Slot} = (\text{Family}, \text{Phase}) \in \{1, \ldots, 168\}$$

4. **Strategy Lookup:**
   $$\text{Strategy} = \text{Table}[\text{Slot}]$$

---

## Literature References

- Smith, L. N. (2018). Cyclical learning rates. *WACV*.
- Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press.
- Sutton, R., Barto, A. (2018). *Reinforcement Learning*. MIT Press.
- Finn, C., et al. (2017). MAML. *ICML*.
- Chen, T., et al. (2020). SimCLR. *ICML*.

