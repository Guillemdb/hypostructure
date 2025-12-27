# AI/ML Surgery Translations: Model Modifications and Training Interventions

## Overview

This document provides comprehensive translations of surgery operations, model modifications, and training interventions from hypostructure theory into the language of **Artificial Intelligence, Machine Learning, and Reinforcement Learning**. Surgeries represent active modifications that transform models, adjust architectures, modify training procedures, or intervene in learning dynamics to improve performance or resolve pathologies.

---

## Part I: Architecture Modification Surgeries

### 1. Network Pruning and Compression

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Weight Pruning** | Set w_ij = 0 if \|w_ij\| < τ | Remove small weights |
| **Neuron Pruning** | Remove entire neurons with small activations | Structural pruning |
| **Filter Pruning** | Remove conv filters with small norm | Channel reduction |
| **Structured Pruning** | Remove entire layers or blocks | Coarse-grained compression |
| **Magnitude-Based Pruning** | Rank by \|w\| and prune smallest | Simple criterion |
| **Gradient-Based Pruning** | Prune based on \|w · ∂L/∂w\| | Sensitivity-based |

### 2. Knowledge Distillation

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Teacher-Student Distillation** | Train small network to mimic large network | Model compression |
| **Soft Target Training** | L = αL_CE(y, ŷ) + (1-α)L_KL(ŷ_teacher, ŷ_student) | Temperature-scaled logits |
| **Feature Matching** | Match intermediate representations | Layer-wise distillation |
| **Attention Transfer** | Transfer attention maps | Spatial knowledge |
| **Self-Distillation** | Distill from own previous checkpoint | Temporal distillation |
| **Online Distillation** | Multiple students teach each other | Peer learning |

### 3. Layer and Block Modification

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Layer Addition** | Insert new layer into network | Depth increase |
| **Layer Removal** | Delete layer, connect adjacent layers | Depth reduction |
| **Skip Connection Addition** | Add u_{l+k} = u_l + f(u_l,...,u_{l+k-1}) | ResNet-style surgery |
| **Dense Connection** | Connect to all previous layers | DenseNet surgery |
| **Block Replacement** | Replace block with different architecture | Module swap |
| **Width Scaling** | Change hidden dimension d → αd | Capacity adjustment |

### 4. Activation Function Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Activation Replacement** | ReLU → GELU, Tanh → Swish, etc. | Non-linearity change |
| **Learnable Activation** | PReLU: max(0, x) + a·min(0, x) | Parameterized activation |
| **Smooth Approximation** | ReLU → Softplus(x) = log(1 + e^x) | Differentiable replacement |
| **Adaptive Activation** | Different activations per layer | Heterogeneous |
| **Gating Addition** | σ(Wx + b) ⊙ x | Multiplicative gating |
| **Attention Mechanism** | softmax(QK^T/√d)V | Self-attention surgery |

---

## Part II: Training Procedure Surgeries

### 5. Learning Rate Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Learning Rate Warmup** | η(t) = η_max · min(1, t/t_warmup) | Gradual increase |
| **Cyclical Learning Rate** | η(t) oscillates between η_min and η_max | Periodic variation |
| **Cosine Annealing** | η(t) = η_min + (η_max - η_min)·(1 + cos(πt/T))/2 | Smooth decay |
| **Learning Rate Restart** | Reset to η_max periodically | SGDR surgery |
| **Reduce on Plateau** | η ← γη when validation plateaus | Adaptive reduction |
| **Layer-Wise Learning Rates** | Different η for each layer | Discriminative fine-tuning |

### 6. Optimizer Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Optimizer Switching** | SGD → Adam or vice versa mid-training | Algorithm change |
| **Momentum Adjustment** | Change β₁ in Adam | Momentum modification |
| **Gradient Clipping** | ∇θ ← ∇θ / max(1, ‖∇θ‖/c) | Bound gradient norm |
| **Weight Decay Modification** | Change λ in L + λ‖θ‖² | Regularization strength |
| **Adaptive Gradient Clipping** | Clip based on parameter norm | AGC surgery |
| **SAM (Sharpness-Aware Minimization)** | θ ← θ - η∇L(θ + ε∇L(θ)) | Flatness-seeking |

### 7. Batch and Data Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Batch Size Increase** | B ← αB during training | Computational efficiency |
| **Curriculum Learning** | Start with easy examples, increase difficulty | Ordered data |
| **Hard Example Mining** | Focus on high-loss examples | Selective sampling |
| **Mixup** | x̃ = λx₁ + (1-λ)x₂, ỹ = λy₁ + (1-λ)y₂ | Data interpolation |
| **CutMix** | Paste patches from other images | Spatial mixing |
| **AutoAugment** | Learned data augmentation policy | Meta-learned augmentation |

---

## Part III: Fine-Tuning and Transfer Surgeries

### 8. Transfer Learning Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Feature Extraction** | Freeze early layers, train only final layers | Fixed backbone |
| **Full Fine-Tuning** | Unfreeze all layers, train on new task | End-to-end adaptation |
| **Gradual Unfreezing** | Progressively unfreeze layers | ULMFiT surgery |
| **Discriminative Fine-Tuning** | Different η for each layer group | Layer-wise rates |
| **Head Replacement** | Replace final layers, keep backbone | Task-specific head |
| **Adapter Layers** | Insert small trainable modules | Parameter-efficient |

### 9. Domain Adaptation Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Domain Adversarial Training** | Add domain classifier, reverse gradient | Confusion surgery |
| **Pseudo-Labeling** | Label target data with source model | Self-training |
| **Backtranslation** | Use target→source model to augment | Synthetic data |
| **Feature Alignment** | Minimize MMD or CORAL between domains | Distribution matching |
| **Self-Training** | Iteratively retrain on confident predictions | Bootstrapping |
| **Co-Training** | Multiple views train separate models | Multi-view surgery |

### 10. Continual Learning Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Elastic Weight Consolidation** | Add ∑ᵢ λᵢ(θᵢ - θᵢ*)² to loss | Protect important weights |
| **Progressive Neural Networks** | Add new column per task, lateral connections | Architecture expansion |
| **PackNet** | Prune network per task, non-overlapping | Sparse masking |
| **Memory Replay** | Store examples from old tasks | Experience buffer |
| **Gradient Episodic Memory** | Project gradient to not interfere with old tasks | Constrained optimization |
| **Learning Without Forgetting** | Knowledge distillation from old model | Self-distillation |

---

## Part IV: Regularization Surgeries

### 11. Dropout and Stochastic Regularization

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Dropout** | Randomly zero activations with p | Stochastic regularization |
| **DropConnect** | Randomly zero weights instead of activations | Weight dropout |
| **Stochastic Depth** | Skip entire layers randomly | Layer dropout |
| **DropBlock** | Drop contiguous regions in feature maps | Spatial dropout |
| **Dropout Scheduling** | Vary dropout rate during training | Adaptive dropout |
| **Variational Dropout** | Bayesian interpretation of dropout | Posterior inference |

### 12. Normalization Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Batch Normalization** | x̂ = (x - μ_batch)/σ_batch | Normalize activations |
| **Layer Normalization** | Normalize across features | RNN/Transformer norm |
| **Group Normalization** | Normalize within channel groups | Batch-size independent |
| **Instance Normalization** | Normalize each example independently | Style transfer |
| **Weight Normalization** | w = g · v/‖v‖ | Reparameterize weights |
| **Spectral Normalization** | W ← W/σ_max(W) | Lipschitz constraint |

### 13. Weight Initialization Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Xavier Initialization** | W ~ U(-√(6/(n_in+n_out)), √(6/(n_in+n_out))) | Variance preservation |
| **He Initialization** | W ~ N(0, 2/n_in) | ReLU-specific |
| **Orthogonal Initialization** | Initialize W as orthogonal matrix | RNN initialization |
| **Pre-Training Initialization** | Initialize from pre-trained checkpoint | Transfer starting point |
| **Lottery Ticket Rewinding** | Reset to early training checkpoint | Rewind surgery |
| **Layer-Sequential Initialization** | Initialize layer-by-layer | Greedy pre-training |

---

## Part V: Architecture Search and AutoML Surgeries

### 14. Neural Architecture Search

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Cell-Based NAS** | Search for repeatable cell structure | Modular search |
| **DARTS (Differentiable)** | Relax discrete choices to continuous | Gradient-based NAS |
| **ENAS (Parameter Sharing)** | Share weights across architectures | Efficient NAS |
| **Evolutionary NAS** | Mutate and select architectures | Genetic algorithm |
| **Random Search NAS** | Sample architectures randomly | Baseline |
| **Network Morphism** | Add units while preserving function | Function-preserving growth |

### 15. Hyperparameter Optimization Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Grid Search** | Exhaustive search over grid | Systematic search |
| **Random Search** | Sample hyperparameters randomly | More efficient than grid |
| **Bayesian Optimization** | Model p(accuracy \| hyperparams), optimize acquisition | Probabilistic |
| **Hyperband** | Successive halving with multiple brackets | Early stopping |
| **Population Based Training** | Evolve population of models | Parallel search |
| **Successive Halving** | Eliminate poorly performing configs | Resource allocation |

### 16. Model Ensemble Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Bagging** | Train on bootstrap samples | Variance reduction |
| **Boosting** | Sequential training on hard examples | Bias reduction |
| **Stacking** | Meta-model combines base models | Hierarchical ensemble |
| **Snapshot Ensembling** | Save checkpoints during training | Temporal ensemble |
| **Fast Geometric Ensembling** | Traverse multiple local minima | Cyclical learning |
| **Mixture of Experts** | Gating network routes to specialists | Conditional ensemble |

---

## Part VI: Generative Model Surgeries

### 17. GAN Architecture Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Progressive Growing** | Start low-res, add layers progressively | Stable training |
| **Self-Attention Addition** | Add attention layers to generator/discriminator | Long-range dependencies |
| **Spectral Normalization** | Normalize discriminator weights | Lipschitz constraint |
| **Two Time-Scale Update** | Different learning rates for G and D | TTUR surgery |
| **Gradient Penalty** | Add ‖∇_x D(x)‖² term | WGAN-GP |
| **Auxiliary Classifier** | Add class prediction head | AC-GAN surgery |

### 18. VAE Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **β-VAE** | Weight KL term: L = Recon + βKL | Disentanglement |
| **Conditional VAE** | Condition on label y | Controlled generation |
| **Hierarchical VAE** | Multiple latent variable layers | Structured latent |
| **VQ-VAE** | Discrete latent space via codebook | Vector quantization |
| **Normalizing Flow** | Invertible transformations for z | Flexible posterior |
| **Warm-Up** | Gradually increase β from 0 to 1 | Annealing |

### 19. Diffusion Model Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Noise Schedule Modification** | Change β_t schedule | Control diffusion |
| **Classifier Guidance** | Add ∇log p(y\|x_t) to score | Conditional generation |
| **Classifier-Free Guidance** | ε̃ = ε_uncond + w(ε_cond - ε_uncond) | No separate classifier |
| **Latent Diffusion** | Diffuse in latent space of VAE | Computational efficiency |
| **Cascaded Diffusion** | Low-res → high-res progression | Super-resolution |
| **DDIM Surgery** | Deterministic sampling | Faster generation |

---

## Part VII: Reinforcement Learning Surgeries

### 20. Policy Surgery

| Surgery Type | RL Translation | Description |
|--------------|----------------|-------------|
| **Policy Distillation** | Compress policy to smaller network | RL compression |
| **Policy Improvement** | π_{new} greedy w.r.t. Q^π_old | Policy iteration step |
| **Behavior Cloning** | Supervised learning from expert | Imitation initialization |
| **DAgger** | Aggregate expert feedback online | Interactive imitation |
| **Policy Gradient Clipping** | Clip importance ratio in PPO | Stability |
| **Trust Region** | Constrain KL(π_new \|\| π_old) ≤ δ | TRPO surgery |

### 21. Value Function Surgery

| Surgery Type | RL Translation | Description |
|--------------|----------------|-------------|
| **Target Network** | Q_target frozen for stability | Decouple update |
| **Double Q-Learning** | a* = argmax Q₁(s',a), use Q₂(s',a*) | Debiasing |
| **Dueling Architecture** | Q = V(s) + (A(s,a) - mean A) | Separate streams |
| **Distributional RL** | Learn distribution of returns | C51, QR-DQN |
| **Hindsight Experience Replay** | Relabel goals in failed trajectories | Sparse reward |
| **Prioritized Replay** | Sample transitions by TD error | Importance sampling |

### 22. Exploration Surgery

| Surgery Type | RL Translation | Description |
|--------------|----------------|-------------|
| **ε-Greedy Annealing** | ε(t) = max(ε_min, ε_max · decay^t) | Decrease exploration |
| **Boltzmann Exploration** | π(a\|s) ∝ exp(Q(s,a)/τ) | Temperature-based |
| **UCB Bonus** | Q̃(s,a) = Q(s,a) + c√(log N(s)/N(s,a)) | Optimism |
| **Curiosity-Driven** | Add intrinsic reward r_int = ‖ê - e‖ | Prediction error bonus |
| **Noisy Nets** | Add learnable noise to weights | Parameter space noise |
| **Entropy Regularization** | J = 𝔼[R] + αH(π) | Maximum entropy RL |

---

## Part VIII: Attention and Transformer Surgeries

### 23. Attention Mechanism Surgery

| Surgery Type | Transformer Translation | Description |
|--------------|-------------------------|-------------|
| **Multi-Head Attention** | h heads, each d_k dimensional | Parallel attention |
| **Sparse Attention** | Attend to subset of positions | Reduce O(n²) cost |
| **Local Attention** | Attend to window of size k | Sliding window |
| **Axial Attention** | Separate attention along axes | Factorized attention |
| **Linear Attention** | Use kernel trick for O(n) attention | Efficient approximation |
| **Cross-Attention Addition** | Attend to encoder output | Decoder surgery |

### 24. Positional Encoding Surgery

| Surgery Type | Transformer Translation | Description |
|--------------|-------------------------|-------------|
| **Absolute Positional Encoding** | PE(pos, 2i) = sin(pos/10000^{2i/d}) | Sinusoidal encoding |
| **Learned Positional Embedding** | Look up position embedding | Trainable positions |
| **Relative Positional Encoding** | Attention depends on i - j | Translation invariance |
| **Rotary Position Embedding (RoPE)** | Rotate query and key | Modern approach |
| **ALiBi** | Bias attention scores by distance | No explicit encoding |
| **Position Interpolation** | Extend to longer sequences | RoPE scaling |

### 25. Transformer Architecture Surgery

| Surgery Type | Transformer Translation | Description |
|--------------|-------------------------|-------------|
| **Pre-LN vs Post-LN** | LayerNorm before vs after attention | Training stability |
| **FFN Expansion** | d_model → d_ff → d_model | Intermediate dimension |
| **Gated FFN** | GLU, SwiGLU variants | Multiplicative gates |
| **Mixture of Experts** | Route to subset of FFN experts | Conditional computation |
| **Encoder-Decoder Separation** | Separate vs shared parameters | Architecture choice |
| **Decoder-Only Surgery** | Remove encoder | GPT-style models |

---

## Part IX: Quantization and Efficiency Surgeries

### 26. Quantization Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Post-Training Quantization** | Quantize trained FP32 model to INT8 | No retraining |
| **Quantization-Aware Training** | Simulate quantization during training | Better accuracy |
| **Binary Networks** | Weights ∈ {-1, +1} | 1-bit quantization |
| **Ternary Networks** | Weights ∈ {-1, 0, +1} | 2-bit quantization |
| **Mixed Precision** | Use FP16 for most layers, FP32 for sensitive | Hybrid approach |
| **Dynamic Quantization** | Quantize activations at runtime | Flexible |

### 27. Sparsity and Structured Pruning

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Magnitude Pruning** | Prune weights with smallest \|w\| | Simple criterion |
| **Lottery Ticket Pruning** | Find sparse subnetwork | Iterative pruning + reset |
| **Movement Pruning** | Prune based on w · Δw | Gradient-based |
| **Block Sparsity** | Enforce block-structured sparsity | Hardware-friendly |
| **N:M Sparsity** | N zeros in every M consecutive weights | Regular pattern |
| **Gradual Magnitude Pruning** | Increase sparsity over training | Annealing |

### 28. Low-Rank Factorization

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **SVD Decomposition** | W ≈ U Σ V^T, keep top-k singular values | Rank reduction |
| **Low-Rank Adaptation (LoRA)** | W ← W₀ + BA where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×d} | Parameter-efficient fine-tuning |
| **Tucker Decomposition** | Tensor factorization for conv layers | Multi-way factorization |
| **CP Decomposition** | CANDECOMP/PARAFAC for tensors | Sum of rank-1 |
| **Tensor Train** | TT-format decomposition | Compact representation |
| **Adapter Layers** | Small bottleneck modules | Low-rank modules |

---

## Part X: Debugging and Intervention Surgeries

### 29. Training Diagnostics Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Gradient Monitoring** | Track ‖∇θ‖, gradient norms per layer | Diagnose vanishing/exploding |
| **Activation Statistics** | Monitor mean, variance of activations | Detect dead neurons |
| **Weight Histograms** | Visualize weight distributions | Check initialization |
| **Learning Rate Finder** | Sweep η, plot loss vs η | Find good η |
| **Gradient Noise** | Measure variance in mini-batch gradients | Large batch insights |
| **Loss Landscape Visualization** | Plot loss in 2D slice | Understand geometry |

### 30. Repair and Recovery Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Gradient Clipping** | Prevent exploding gradients | Emergency intervention |
| **Learning Rate Reduction** | Reduce η when loss spikes | Recover from instability |
| **Checkpoint Rollback** | Revert to earlier checkpoint | Undo bad update |
| **Warmup Restart** | Reset to warmup phase | Re-stabilize |
| **Batch Norm Re-Calibration** | Recompute running statistics | Fix BN after pruning |
| **Loss Spike Filtering** | Ignore gradient if loss > threshold | Outlier rejection |

### 31. Model Editing Surgery

| Surgery Type | AI/ML Translation | Description |
|--------------|-------------------|-------------|
| **Weight Patching** | Directly modify specific weights | Manual intervention |
| **Neuron Ablation** | Zero out specific neurons | Causality analysis |
| **Activation Steering** | Add constant to activations | Control behavior |
| **Knowledge Editing** | Modify model to correct specific outputs | Precision editing |
| **Causal Tracing** | Identify critical components | Attribution |
| **Activation Patching** | Replace activations from reference run | Intervention analysis |

---

## Part XI: Federated and Distributed Learning Surgeries

### 32. Federated Learning Surgery

| Surgery Type | Federated Translation | Description |
|--------------|----------------------|-------------|
| **FedAvg** | Server averages client models | Central aggregation |
| **FedProx** | Add proximal term μ‖θ - θ_global‖² | Handle heterogeneity |
| **Personalized Federated** | Each client keeps personalized layers | Hybrid global/local |
| **Clustered Federated** | Group similar clients | Heterogeneity handling |
| **Differential Privacy** | Add noise to gradients | Privacy preservation |
| **Secure Aggregation** | Encrypted gradient aggregation | Security surgery |

---

## Conclusion

This comprehensive catalog of AI/ML surgeries establishes the complete toolkit for model modifications and training interventions:

**Architecture Surgeries** (pruning, distillation, layer modification) compress models and change capacity while preserving performance.

**Training Procedure Surgeries** (learning rate, optimizer, batch modifications) adapt the optimization process to improve convergence and generalization.

**Fine-Tuning and Transfer** (transfer learning, domain adaptation, continual learning) adapt pre-trained models to new tasks and domains.

**Regularization Surgeries** (dropout, normalization, initialization) improve generalization and training stability.

**AutoML Surgeries** (NAS, hyperparameter optimization, ensembles) automatically discover better architectures and configurations.

**Generative Model Surgeries** (GAN, VAE, diffusion modifications) improve generation quality and training stability.

**RL Surgeries** (policy, value function, exploration modifications) improve sample efficiency and training stability in sequential decision-making.

**Attention and Transformer Surgeries** (attention mechanisms, positional encoding, architecture variants) adapt transformer architectures for different tasks.

**Quantization and Efficiency** (quantization, sparsity, low-rank) reduce computational and memory costs.

**Debugging and Intervention** (diagnostics, repair, model editing) diagnose problems and make targeted fixes.

**Federated Learning** (aggregation, personalization, privacy) enable distributed training with privacy constraints.

These surgeries form the active toolkit of machine learning engineering, providing systematic interventions to improve models, adapt to new tasks, compress for deployment, and debug failures—complementing the passive learning barriers with constructive modification strategies.

---

**Cross-References:**
- [AI Index](sketch-ai-index.md) - Complete catalog of AI/ML sketches
- [AI Interfaces](sketch-ai-interfaces.md) - Core concept translations
- [AI Barriers](sketch-ai-barriers.md) - Fundamental constraints
- [AI Failure Modes](sketch-ai-failure-modes.md) - Outcome classifications
- [GMT Surgeries](../gmt/sketch-gmt-surgeries.md) - Geometric modifications
- [Complexity Surgeries](../discrete/sketch-discrete-surgeries.md) - Computational transformations
- [Arithmetic Surgeries](../arithmetic/sketch-arithmetic-surgeries.md) - Number-theoretic operations
