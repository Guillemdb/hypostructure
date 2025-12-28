---
title: "AI/RL/ML Hypostructure Translations: Complete Guide & Index"
---

# AI/RL/ML Hypostructure Translations: Complete Guide & Index

## Overview

This document provides a comprehensive guide to the AI/Reinforcement Learning/Machine Learning translations of hypostructure theorems. It serves as both a detailed dictionary of concepts and a master index for the specific theorem sketches.

Each section establishes formal correspondences between continuous dynamical systems concepts and neural network training, optimization, and learning theory.

**Purpose:** These translations serve machine learning researchers seeking to understand hypostructure through familiar optimization and learning-theoretic concepts, and dynamical systems researchers interested in the computational aspects of their theorems.

## Translation Framework Summary

A high-level dictionary of the core correspondences:

| Hypostructure | AI/RL/ML |
|---------------|----------|
| State space $\mathcal{X}$ | Parameter space $\Theta$, Policy space $\Pi$ |
| Semiflow $S_t$ | Training dynamics, Gradient descent |
| Energy functional $\Phi$ | Loss function $\mathcal{L}$, Negative value $-V$ |
| Dissipation $\mathfrak{D}$ | Gradient norm $\|\nabla\mathcal{L}\|$, TD error |
| Singularity $\Sigma$ | Mode collapse, Gradient explosion/vanishing |
| Surgery | Pruning, Knowledge distillation, Fine-tuning |
| Global attractor | Optimal policy $\pi^*$, Global minimum |
| Certificate $K$ | PAC bounds, Convergence guarantees |

---

# Part I: Core Interfaces

*Translations of foundational hypostructure objects into learning theory.*

### 1. Topos and Categories

| Hypostructure Concept | AI/ML Translation | Description |
|----------------------|-------------------|-------------|
| **Topos T** | Model space / Hypothesis class | Space of all possible models |
| **Object in T** | Neural network architecture | Specific model structure |
| **Morphism** | Transfer learning map | Knowledge transfer between models |
| **Subobject classifier Ω** | Binary classifier | {0,1}-valued prediction function |
| **Internal logic** | Inductive bias | Structural assumptions of learning |

### 2. State Spaces and Dynamics

| Hypostructure Concept | AI/ML Translation | Description |
|----------------------|-------------------|-------------|
| **State space S** | Parameter space Θ or Policy space Π | θ ∈ ℝ^p (weights) or π: S → A (policy) |
| **Configuration** | Model parameters θ | Specific weight assignment |
| **Semiflow Φₜ** | Training dynamics | θ(t+1) = θ(t) - η∇L(θ(t)) |
| **Orbit** | Training trajectory | {θ(t) : t = 0,1,2,...} |
| **Fixed point** | Critical point of loss | ∇L(θ*) = 0 |

### 3. Energy and Variational Structure

| Hypostructure Concept | AI/ML Translation | Description |
|----------------------|-------------------|-------------|
| **Energy functional E** | Loss function L(θ) | 𝔼[(f_θ(x) - y)²] + λR(θ) |
| **Dissipation Ψ** | Gradient norm ‖∇L(θ)‖² | Rate of parameter change |
| **Lyapunov function** | Training loss L(θ(t)) | Decreasing during training |
| **Energy identity** | Loss + Regularization balance | L = L_data + λL_reg |
| **Gradient system** | Gradient descent | θₜ₊₁ = θₜ - η∇L(θₜ) |

### 4. Sheaves and Localization

| Hypostructure Concept | AI/ML Translation | Description |
|----------------------|-------------------|-------------|
| **Sheaf F** | Local model ensemble | Region-specific models |
| **Stalk Fₓ** | Local prediction at x | Model behavior near datapoint x |
| **Sheaf morphism** | Model agreement on overlap | Consistency between local models |
| **Sheaf cohomology H^i** | Obstruction to global model | Cannot unify local models |
| **Čech cohomology** | Ensemble combination weights | How to merge local predictors |

### 5. Kernels and Fundamental Properties

| Hypostructure Concept | AI/ML Translation | Description |
|----------------------|-------------------|-------------|
| **Kernel (krnl)** | Optimal policy / Global minimum | θ* minimizing L(θ) |
| **Consistency** | Statistical consistency | θ̂ₙ → θ* as n → ∞ |
| **Equivariance** | Symmetry in architecture | f(g·x; θ) = g·f(x; θ) for g ∈ G |
| **Fixed point structure** | Nash equilibrium (in GANs) | Generator/discriminator balance |
| **Eigenstructure** | Principal components / Hessian spectrum | Eigenvalues of ∇²L(θ*) |

### 6. Factories and Constructions

| Hypostructure Concept | AI/ML Translation | Description |
|----------------------|-------------------|-------------|
| **Factory (fact)** | Model architecture pattern | Standard building block (ResNet, Transformer) |
| **Barrier** | Regularization / Constraint | ‖θ‖² ≤ B or dropout |
| **Gate** | Gating mechanism | LSTM gates, attention weights |
| **Stratification** | Hierarchical representation | Layers learn features of different complexity |
| **Approximation** | Model compression | Pruning, quantization, distillation |

### 7. Singularity Theory and Resolutions

| Hypostructure Concept | AI/ML Translation | Description |
|----------------------|-------------------|-------------|
| **Singularity** | Training pathology | Gradient explosion, mode collapse |
| **Concentration** | Mode collapse | Model outputs collapse to few modes |
| **Blowup** | Gradient explosion | ‖∇L(θ(t))‖ → ∞ |
| **Surgery** | Architecture modification | Add/remove layers, change activations |
| **Neck pinch** | Bottleneck layer | Dimension reduction layer |
| **Obstruction** | Fundamental limitation | No-free-lunch theorem, VC dimension |
| **Tower** | Progressive training | Curriculum learning, layer-wise training |
| **Resolution** | Fine-tuning / Distillation | Refine pre-trained model |
| **Smoothing** | Gradient clipping / Batch norm | Stabilize training dynamics |

### 8. Attractor Theory and Generalization

| Hypostructure Concept | AI/ML Translation | Description |
|----------------------|-------------------|-------------|
| **Global attractor A** | Optimal policy / Global minimum | θ* minimizing true risk |
| **Basin of attraction** | Initialization region for convergence | {θ₀ : θ(t) → θ*} |
| **Stability** | Generalization | Test loss ≈ Train loss |
| **Locking (lock)** | Representation collapse | All inputs map to same feature |
| **Entropy locking** | Maximum entropy principle | Maximize H(π) subject to constraints |
| **Capacity** | VC dimension / Rademacher complexity | Measure of model expressiveness |
| **Certificate** | Generalization bound | R(θ) ≤ R̂(θ) + C√(d/n) |
| **Verification** | Model validation | Cross-validation, holdout test |

### 9. Topos-Theoretic Structures

| Hypostructure Concept | AI/ML Translation | Description |
|----------------------|-------------------|-------------|
| **2-morphism** | Hyperparameter path | Family of models {θ_λ : λ ∈ Λ} |
| **Natural transformation** | Model interpolation | Linear path θ(t) = (1-t)θ₀ + tθ₁ |
| **Adjunction** | Encoder-decoder pair | Autoencoder structure |
| **Monad** | Recurrent structure | RNN composition Tⁿ(h₀) |
| **Comonad** | Attention mechanism | Query-key-value structure |
| **Limit** | Ensemble intersection | Bagging: agreement of all models |
| **Colimit** | Ensemble union | Boosting: combine weak learners |
| **Homology H_k(X)** | Topological data analysis | Persistent homology of data |
| **Spectrum** | Hessian eigenvalues | {λᵢ : ∇²L(θ*)vᵢ = λᵢvᵢ} |

### 10. Dualities and Convergence

| Hypostructure Concept | AI/ML Translation | Description |
|----------------------|-------------------|-------------|
| **Poincaré duality** | Encoder-decoder duality | Inverse mappings |
| **Hodge duality** | Bias-variance duality | Tradeoff decomposition |
| **Legendre duality** | Primal-dual optimization | Convex conjugate |
| **Strong convergence** | Parameter convergence | θₙ → θ* in ℓ² |
| **Weak convergence** | Distributional convergence | f_θₙ(x) ⇀ f_θ*(x) |
| **Varifold convergence** | Neural collapse | Features collapse to simplex |

---

# Part II: Passive Barriers

*Fundamental constraints, impossibility results, and bounds.*

### 1. Generalization Barriers

| Barrier Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **PAC Learning Barrier** | m ≥ (1/ε)(log(1/δ) + log\|H\|) | Sample complexity for PAC learning |
| **VC Dimension Bound** | m ≥ C(VC(H)/ε + log(1/δ)/ε) | Fundamental sample complexity |
| **Rademacher Complexity** | R(L) - R̂(L) ≤ 2R_n(H) + √(log(1/δ)/2n) | Generalization via Rademacher complexity |
| **Bias-Variance Tradeoff** | E[(f̂ - f)²] = Bias² + Variance + σ² | Irreducible decomposition |
| **No-Free-Lunch** | Avg over all f: no learner better than random | Universal learning impossible |
| **Realizable vs Agnostic Gap** | Agnostic PAC needs O(1/ε²) vs O(1/ε) | Realizable assumption helps |

### 2. Optimization Barriers

| Barrier Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Local Minima Trap** | ∇L(θ) = 0 but θ not global min | Non-convex landscape |
| **Saddle Point Barrier** | Critical point with negative eigenvalues | Hessian has mixed signs |
| **Barren Plateau** | ‖∇L‖ → 0 exponentially in depth | Quantum/deep network barrier |
| **Lipschitz Barrier** | L Lipschitz ⟹ rate O(1/√T) for SGD | Fundamental convergence rate |
| **Variance Barrier** | σ² variance ⟹ rate O(σ²/√T) | Stochastic noise floor |
| **Mini-Batch Tradeoff** | Batch size b: speedup √b, diminishing returns | Communication-computation tradeoff |
| **Learning Rate Sensitivity** | η too large ⟹ divergence, too small ⟹ slow | Goldilocks zone |

### 3. Capacity and Complexity Barriers

| Barrier Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **VC Dimension Ceiling** | VC(H) = ∞ ⟹ no uniform convergence | Infinite capacity |
| **Parameter Count** | # parameters ≈ model capacity | Proxy for expressiveness |
| **Network Width** | Width w ≥ w_min for universal approximation | Minimum width |
| **Curse of Dimensionality** | Grid in d dimensions needs exp(d) points | Exponential sample requirement |
| **Knowledge Distillation Limit** | Student cannot exceed teacher | Upper bound on compression |
| **Lottery Ticket Hypothesis** | Sparse subnetwork exists with same performance | Pruning potential |

### 4. Information-Theoretic Barriers

| Barrier Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Information Bottleneck** | Minimize I(X;Z) subject to I(Z;Y) ≥ I_min | Compression-prediction tradeoff |
| **Differential Privacy** | ε-DP ⟹ O(√(d/nε²)) generalization cost | Privacy-accuracy tradeoff |
| **Sample Complexity Lower Bound** | Ω(d/ε) samples necessary | Information-theoretic minimum |
| **Label Complexity** | Active learning saves O(log d) factor | Logarithmic improvement |
| **Transfer Learning Gap** | Source-target mismatch limits transfer | Domain divergence |

### 5. Adversarial and Robustness Barriers

| Barrier Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Adversarial Perturbation** | ‖δ‖ < ε ⟹ f(x+δ) ≠ f(x) | Small perturbation changes prediction |
| **Robustness-Accuracy Tradeoff** | Robust accuracy < standard accuracy | Fundamental tension |
| **Distribution Shift** | P_train ≠ P_test ⟹ performance drop | Covariate shift |
| **Fairness-Accuracy Tradeoff** | Demographic parity ⟹ accuracy loss | Pareto frontier |
| **Impossibility Theorem** | Cannot satisfy all fairness criteria simultaneously | Incompatible definitions |

### 6. Architectural Constraints

| Barrier Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Vanishing Gradient** | ∂L/∂θ₁ → 0 in deep networks | Backprop attenuation |
| **Exploding Gradient** | ‖∇L‖ → ∞ | Instability in training |
| **Attention Complexity** | Self-attention O(n²) in sequence length | Quadratic barrier |
| **Context Length** | Max sequence length L_max | Memory and computation limit |
| **KV Cache Memory** | O(L·d) memory for generation | Inference memory barrier |
| **LSTM/GRU Necessity** | Vanilla RNN cannot learn long dependencies | Gating required |

### 7. RL and Continual Learning Barriers

| Barrier Type | RL/ML Translation | Description |
|--------------|-------------------|-------------|
| **Regret Lower Bound** | Ω(√T) regret necessary | Fundamental exploration cost |
| **Deadly Triad** | Bootstrapping + off-policy + function approx ⟹ divergence | Instability conditions |
| **Catastrophic Forgetting** | New task ⟹ forget old tasks | Plasticity-stability dilemma |
| **Task Distribution Shift** | Train tasks ≠ test tasks | Meta-distribution shift |
| **Interpretability-Accuracy** | Complex models less interpretable | Fundamental tension |

---

# Part III: Failure Modes

*Taxonomy of training outcomes and dynamic behaviors.*

### Primary Failure Modes

| Mode | Name | Hypostructure | AI/RL/ML Interpretation | Learning Outcome |
|------|------|---------------|-------------------------|------------------|
| **D.D** | **Dispersion-Decay** | Energy disperses, global existence | Rapid convergence, convex optimization | PAC-learnable, polynomial sample complexity |
| **S.E** | **Subcritical-Equilibrium** | Subcritical scaling, bounded blowup | Curriculum learning, staged training | Curriculum learning, staged training |
| **C.D** | **Concentration-Dispersion** | Partial concentration, structural dispersion | Representation learning, autoencoders | Manifold learning with dimension reduction |
| **C.E** | **Concentration-Escape** | Genuine singularity, energy blowup | Training instability, mode collapse | Non-convex hard regime, sample complexity blowup |
| **T.E** | **Topological-Extension** | Topological completion required | Architecture surgery, NAS | Network surgery, capacity expansion |
| **S.D** | **Structural-Dispersion** | Structural rigidity forces dispersion | Symmetry-aided learning, equivariant networks | Sample-efficient learning via inductive bias |

### Secondary and Extended Modes

| Mode | Name | AI/ML Manifestation | Interpretation |
|------|------|---------------------|----------------|
| **C.C** | **Event Accumulation** | Catastrophic forgetting | Unbounded accumulation of loss on old tasks |
| **T.D** | **Glassy Freeze** | Local optima traps | Stuck in poor local minimum despite training |
| **T.C** | **Labyrinthine** | Complex architecture | High connectivity preventing simplification |
| **D.E** | **Oscillatory** | GAN instability | Generator-discriminator cycling without convergence |
| **D.C** | **Semantic Horizon** | Out-of-distribution failure | Inputs beyond training manifold leading to nonsense |
| **S.C** | **Parametric Instability** | Hyperparameter sensitivity | Phase transitions in learning rate or batch size |

### Detailed Mode Descriptions

#### Mode D.D: Dispersion-Decay (Global Convergence)
**Meaning:** Rapid and stable convergence to global optimum. The loss landscape is convex or nearly convex/PL-satisfying.
**Examples:** Linear regression, Logistic regression, SVMs, Kernel methods.
**Certificate:** $K^+_{D.D}$ with exponential convergence rate $O(e^{-\lambda t})$.

#### Mode S.E: Subcritical-Equilibrium (Curriculum Learning)
**Meaning:** Controlled optimization through curriculum or staged training. Difficulty increases progressively to avoid getting stuck.
**Examples:** Curriculum learning, Progressive GANs, Transfer learning steps.
**Certificate:** $K^+_{S.E}$ with bounded stage complexity.

#### Mode C.D: Concentration-Dispersion (Representation Learning)
**Meaning:** Mixed regime where model learns compact representations (concentration) while maintaining necessary variance (dispersion).
**Examples:** Autoencoders, VAEs, Contrastive learning (SimCLR), PCA.
**Certificate:** $K^+_{C.D}$ showing effective dimension reduction.

#### Mode C.E: Concentration-Escape (Training Instability)
**Meaning:** Genuine failure. Gradient explosion, vanishing, or mode collapse. The system cannot converge to a valid solution.
**Examples:** Deep networks without normalization (exploding gradients), GAN mode collapse, Hard exploration RL.
**Certificate:** $K^-_{C.E}$ indicating divergence or pathological collapse.

#### Mode T.E: Topological-Extension (Architecture Surgery)
**Meaning:** The current architecture is insufficient. Requires "surgery" (adding layers, pruning, skip connections) to resolve.
**Examples:** Neural Architecture Search (NAS), Pruning, Adding attention to RNNs.
**Certificate:** $K^+_{T.E}$ showing improvement after architectural modification.

#### Mode S.D: Structural-Dispersion (Symmetry-Aided Learning)
**Meaning:** Symmetries (equivariance) constrain the search space, forcing rapid convergence.
**Examples:** CNNs (translation), GNNs (permutation), Equivariant Transformers.
**Certificate:** $K^+_{S.D}$ with symmetry group $G$.

#### Mode C.C: Event Accumulation (Catastrophic Forgetting)
**Meaning:** Sequential learning fails as new tasks overwrite old ones.
**Mitigation:** Elastic Weight Consolidation (EWC), Replay buffers.

#### Mode T.D: Glassy Freeze (Local Optima Traps)
**Meaning:** Optimization gets stuck in a metastable state (poor local minimum) and cannot escape.
**Mitigation:** Stochastic weight averaging, cyclic learning rates.

#### Mode D.E: Oscillatory (Adversarial Cycles)
**Meaning:** Training cycles indefinitely (e.g., in minimax games).
**Mitigation:** Spectral normalization, Two Time-Scale Update Rule (TTUR).

#### Mode D.C: Semantic Horizon (OOD Failure)
**Meaning:** Model fails catastrophically on inputs outside the training distribution.
**Mitigation:** OOD detection, Robust training.

#### Mode S.C: Parametric Instability (Sensitivity)
**Meaning:** Tiny changes in hyperparameters (LR, batch size) cause drastic failure.
**Mitigation:** Adaptive optimizers (Adam), Normalization layers.

---

# Part IV: Active Surgeries

*Catalog of interventions, modifications, and repair techniques.*

### 1. Architecture Modification Surgeries

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Network Pruning** | Remove weights/neurons | Compression, regularization |
| **Knowledge Distillation** | Teacher-Student training | Model compression |
| **Layer Addition/Removal** | Change depth | Capacity adjustment |
| **Skip Connections** | Add residual paths | Resolve vanishing gradients |
| **Activation Surgery** | Change ReLU → GELU, etc. | Non-linearity modification |
| **Attention Mechanism** | Add self-attention | Capture long-range dependencies |

### 2. Training Procedure Surgeries

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Learning Rate Schedules** | Warmup, Cosine Annealing | stabilize and accelerate training |
| **Optimizer Switching** | SGD → Adam | Change update rule dynamics |
| **Gradient Clipping** | Bound gradient norm | Prevent explosion |
| **Batch Size Adjustment** | Increase batch size | Efficiency and stability |
| **Curriculum Learning** | Order examples by difficulty | Prevent early bad minima |
| **Data Augmentations** | Mixup, CutMix, AutoAugment | Regularization via data surgery |

### 3. Fine-Tuning and Transfer Surgeries

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Feature Extraction** | Freeze backbone, train head | Transfer learning |
| **Full Fine-Tuning** | Update all parameters | Adaptation to new task |
| **Adapter Layers** | Insert small trainable modules | Parameter-efficient transfer |
| **Domain Adaptation** | Adversarial alignment | Handle distribution shift |
| **Elastic Weight Consolidation** | Penalize changing old weights | Prevent catastrophic forgetting |
| **Memory Replay** | Re-train on old examples | Continual learning |

### 4. Regularization and Normalization Surgeries

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Dropout** | Randomly zero activations | Stochastic regularization |
| **Batch/Layer Normalization** | Normalize activations | Stabilize dynamics |
| **Weight Initialization** | Xavier, He, Orthogonal | Critical for early convergence |
| **Weight Decay** | L2 regularization | Prevent overfitting |
| **Spectral Normalization** | Constrain Lipschitz constant | Stability in GANs |

### 5. AutoML and Search Surgeries

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Neural Architecture Search** | Automate network design | Discovery of optimal topology |
| **Hyperparameter Optimization** | Bayesian Opt, Grid Search | Tuning configuration |
| **Ensembling** | Bagging, Boosting, Stacking | Combine models for robustness |
| **Mixture of Experts** | Route to specialized sub-nets | Conditional computation |

### 6. Generative and RL Surgeries

| Surgery Type | Translation | Description |
|--------------|-------------|-------------|
| **GAN Stabilization** | Gradient penalty, TTUR | Prevent mode collapse |
| **VAE Beta-Surgery** | Adjust $\beta$ in ELBO | Disentanglement control |
| **Diffusion Guidance** | Classifier guidance | Control generation content |
| **Policy Distillation** | Compress RL policy | Efficient deployment |
| **Target Networks** | Freeze Q-target | Stabilize Q-learning |
| **Entropy Regularization** | Add entropy bonus | Encourage exploration |

### 7. Transformer and Efficiency Surgeries

| Surgery Type | Translation | Description |
|--------------|-------------|-------------|
| **Sparse Attention** | Restrict attention window | Reduce O(n²) cost |
| **Positional Encoding** | RoPE, ALiBi | Handle sequence order |
| **Quantization** | FP32 → INT8 | Inference speedup |
| **Low-Rank Adaptation** | LoRA | Efficient fine-tuning |
| **Gradient Checkpointing** | Trade compute for memory | Train larger models |
| **Model Editing** | Weight patching, ablation | Direct intervention |

---

# Part V: Theorem Catalog

*Master index of all specific hypostructure theorem sketches.*

### Kernel Theorems (KRNL-*)
*Foundational properties of training dynamics.*

| File | Theorem | AI/ML Translation |
|------|---------|-------------------|
| [sketch-ai-mt-krnl-consistency.md](sketch-ai-mt-krnl-consistency.md) | KRNL-Consistency | Training convergence via gradient flow fixed points |
| [sketch-ai-mt-krnl-equivariance.md](sketch-ai-mt-krnl-equivariance.md) | KRNL-Equivariance | Group-equivariant neural networks |
| [sketch-ai-mt-krnl-exclusion.md](sketch-ai-mt-krnl-exclusion.md) | KRNL-Exclusion | No-free-lunch theorems |
| [sketch-ai-mt-krnl-hamilton-jacobi.md](sketch-ai-mt-krnl-hamilton-jacobi.md) | KRNL-HamiltonJacobi | Bellman equations in RL |
| [sketch-ai-mt-krnl-jacobi.md](sketch-ai-mt-krnl-jacobi.md) | KRNL-Jacobi | Jacobian analysis / Sensitivity |
| [sketch-ai-mt-krnl-lyapunov.md](sketch-ai-mt-krnl-lyapunov.md) | KRNL-Lyapunov | Lyapunov convergence proofs |
| [sketch-ai-mt-krnl-metric-action.md](sketch-ai-mt-krnl-metric-action.md) | KRNL-MetricAction | Natural gradient / Fisher information |
| [sketch-ai-mt-krnl-openness.md](sketch-ai-mt-krnl-openness.md) | KRNL-Openness | Algorithm robustness |
| [sketch-ai-mt-krnl-shadowing.md](sketch-ai-mt-krnl-shadowing.md) | KRNL-Shadowing | Shadowing lemma in training |
| [sketch-ai-mt-krnl-stiff-pairing.md](sketch-ai-mt-krnl-stiff-pairing.md) | KRNL-StiffPairing | Stiff dynamics conditioning |
| [sketch-ai-mt-krnl-subsystem.md](sketch-ai-mt-krnl-subsystem.md) | KRNL-Subsystem | Modular transfer |
| [sketch-ai-mt-krnl-trichotomy.md](sketch-ai-mt-krnl-trichotomy.md) | KRNL-Trichotomy | Regimes: Convergent, Cycling, Divergent |
| [sketch-ai-mt-krnl-weak-strong.md](sketch-ai-mt-krnl-weak-strong.md) | KRNL-WeakStrong | Weak-to-strong generalization |

### Sieve and Structure Theorems (THM-*)
*Computational architecture and verification.*

| File | Theorem | AI/ML Translation |
|------|---------|-------------------|
| [sketch-ai-thm-168-slots.md](sketch-ai-thm-168-slots.md) | THM-168Slots | Training regime classification |
| [sketch-ai-thm-categorical-completeness.md](sketch-ai-thm-categorical-completeness.md) | THM-CategoricalCompleteness | Architecture completeness |
| [sketch-ai-thm-closure-termination.md](sketch-ai-thm-closure-termination.md) | THM-ClosureTermination | Finite time convergence |
| [sketch-ai-thm-compactness-resolution.md](sketch-ai-thm-compactness-resolution.md) | THM-CompactnessResolution | Compactness bounds |
| [sketch-ai-thm-dag.md](sketch-ai-thm-dag.md) | THM-DAG | Feedforward structure |
| [sketch-ai-thm-epoch-termination.md](sketch-ai-thm-epoch-termination.md) | THM-EpochTermination | Epoch termination |
| [sketch-ai-thm-expansion-adjunction.md](sketch-ai-thm-expansion-adjunction.md) | THM-ExpansionAdjunction | Capacity adjunction |
| [sketch-ai-thm-finite-runs.md](sketch-ai-thm-finite-runs.md) | THM-FiniteRuns | Bounded iterations |
| [sketch-ai-thm-meta-identifiability.md](sketch-ai-thm-meta-identifiability.md) | THM-MetaIdentifiability | Meta-learning identifiability |
| [sketch-ai-thm-non-circularity.md](sketch-ai-thm-non-circularity.md) | THM-NonCircularity | Non-circular training |
| [sketch-ai-thm-soundness.md](sketch-ai-thm-soundness.md) | THM-Soundness | Optimization soundness |

### Factory Theorems (FACT-*)
*Model construction and validation.*

| File | Theorem | AI/ML Translation |
|------|---------|-------------------|
| [sketch-ai-fact-barrier.md](sketch-ai-fact-barrier.md) | FACT-Barrier | Barrier methods |
| [sketch-ai-fact-gate.md](sketch-ai-fact-gate.md) | FACT-Gate | Gating mechanisms |
| [sketch-ai-fact-instantiation.md](sketch-ai-fact-instantiation.md) | FACT-Instantiation | Initialization |
| [sketch-ai-fact-lock.md](sketch-ai-fact-lock.md) | FACT-Lock | Parameter locking |
| [sketch-ai-fact-soft-profdec.md](sketch-ai-fact-soft-profdec.md) | FACT-SoftProfDec | Curriculum/Profile decomp |
| [sketch-ai-fact-surgery.md](sketch-ai-fact-surgery.md) | FACT-Surgery | Surgical fine-tuning |
| [sketch-ai-fact-transport.md](sketch-ai-fact-transport.md) | FACT-Transport | Optimal transport |

### Resolution Theorems (RESOLVE-*)
*Training procedures and modifications.*

| File | Theorem | AI/ML Translation |
|------|---------|-------------------|
| [sketch-ai-resolve-admissibility.md](sketch-ai-resolve-admissibility.md) | RESOLVE-Admissibility | Admissible modifiers |
| [sketch-ai-resolve-auto-admit.md](sketch-ai-resolve-auto-admit.md) | RESOLVE-AutoAdmit | Auto-hyperparams |
| [sketch-ai-resolve-auto-profile.md](sketch-ai-resolve-auto-profile.md) | RESOLVE-AutoProfile | Auto-architecture |
| [sketch-ai-resolve-auto-surgery.md](sketch-ai-resolve-auto-surgery.md) | RESOLVE-AutoSurgery | Auto-pruning |
| [sketch-ai-resolve-conservation.md](sketch-ai-resolve-conservation.md) | RESOLVE-Conservation | Info conservation |
| [sketch-ai-resolve-obstruction.md](sketch-ai-resolve-obstruction.md) | RESOLVE-Obstruction | Obstruction detection |
| [sketch-ai-resolve-profile.md](sketch-ai-resolve-profile.md) | RESOLVE-Profile | Profiling |
| [sketch-ai-resolve-tower.md](sketch-ai-resolve-tower.md) | RESOLVE-Tower | Hierarchical construction |

### Promotion Theorems (UP-*)
*Local-to-global properties.*

| File | Theorem | AI/ML Translation |
|------|---------|-------------------|
| [sketch-ai-up-absorbing.md](sketch-ai-up-absorbing.md) | UP-Absorbing | Absorbing states |
| [sketch-ai-up-capacity.md](sketch-ai-up-capacity.md) | UP-Capacity | Capacity saturation |
| [sketch-ai-up-catastrophe.md](sketch-ai-up-catastrophe.md) | UP-Catastrophe | Catastrophic forgetting |
| [sketch-ai-up-ergodic.md](sketch-ai-up-ergodic.md) | UP-Ergodic | Ergodicity |
| [sketch-ai-up-lock.md](sketch-ai-up-lock.md) | UP-Lock | Locking properties |
| [sketch-ai-up-spectral.md](sketch-ai-up-spectral.md) | UP-Spectral | Spectral properties |
| [sketch-ai-up-surgery.md](sketch-ai-up-surgery.md) | UP-Surgery | Surgery bounds |
| [sketch-ai-up-type-ii.md](sketch-ai-up-type-ii.md) | UP-TypeII | Type-II singularities |

### Lock Theorems (LOCK-*)
*Impossibility results and lower bounds.*

| File | Theorem | AI/ML Translation |
|------|---------|-------------------|
| [sketch-ai-lock-entropy.md](sketch-ai-lock-entropy.md) | LOCK-Entropy | Entropy barriers |
| [sketch-ai-lock-hodge.md](sketch-ai-lock-hodge.md) | LOCK-Hodge | Feature constraints |
| [sketch-ai-lock-kodaira.md](sketch-ai-lock-kodaira.md) | LOCK-Kodaira | Vanishing gradients |
| [sketch-ai-lock-spectral-gen.md](sketch-ai-lock-spectral-gen.md) | LOCK-SpectralGen | Generalization spectral bounds |
| [sketch-ai-lock-unique-attractor.md](sketch-ai-lock-unique-attractor.md) | LOCK-UniqueAttractor | Uniqueness |

### Action Theorems (ACT-*)
*Network surgery operations.*

| File | Theorem | AI/ML Translation |
|------|---------|-------------------|
| [sketch-ai-act-align.md](sketch-ai-act-align.md) | ACT-Align | Alignment |
| [sketch-ai-act-compactify.md](sketch-ai-act-compactify.md) | ACT-Compactify | Compression |
| [sketch-ai-act-lift.md](sketch-ai-act-lift.md) | ACT-Lift | Lifting/Embedding |
| [sketch-ai-act-surgery.md](sketch-ai-act-surgery.md) | ACT-Surgery | Architecture surgery |

### Lemmas (LEM-*)

| File | Theorem | AI/ML Translation |
|------|---------|-------------------|
| [sketch-ai-lem-bridge.md](sketch-ai-lem-bridge.md) | LEM-Bridge | Theory-practice bridge |
