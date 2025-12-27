---
title: "LOCK-Periodic - AI/RL/ML Translation"
---

# LOCK-Periodic: Algorithm Selection Table

## Overview

The algorithm selection lock shows that every learning problem has a unique position in a classification matrix (analogous to the periodic table), and this position determines the appropriate training strategy. The (problem type, learning phase) pair uniquely identifies the optimal algorithmic approach.

**Original Theorem Reference:** {prf:ref}`lock-periodic`

---

## AI/RL/ML Statement

**Theorem (Algorithm Selection Lock, ML Form).**
Let $\mathcal{P}$ be the space of machine learning problems classified by:
- **8 Problem Families** (structural types based on data, task, constraints)
- **21 Learning Phases** (training stages and decision points)

Then:
1. **Family Determination:** The problem family determines the *class* of applicable algorithms
2. **Phase Determination:** The learning phase determines the *technique* within that class
3. **Unique Strategy:** The (Family, Phase) pair uniquely determines the optimal training approach

$$\text{Strategy}(\mathcal{P}) = \mathcal{S}(\text{Family}(\mathcal{P}), \text{Phase}(\mathcal{P}))$$

**Corollary (No Free Lunch Resolution).**
While no algorithm is best for all problems, the classification matrix identifies the best algorithm for each problem class—transforming algorithm selection from art to systematic science.

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| 8x21 Periodic Table | Algorithm selection matrix | Problem-to-strategy mapping |
| Family (Row) | Problem class | Supervised, unsupervised, RL, etc. |
| Stratum (Column) | Learning phase | Data prep, architecture, training, evaluation |
| Certificate $K_N$ | Phase-specific metric | Accuracy, loss, convergence rate |
| Structural DNA | Problem signature | Complete task characterization |
| Family I (Stable) | Standard supervised | SGD + cross-entropy |
| Family II (Relaxed) | Semi-supervised | Self-training, pseudo-labels |
| Family III (Gauged) | Transfer learning | Pretrain + finetune |
| Family IV (Resurrected) | Curriculum learning | Stage-wise training |
| Family V (Synthetic) | Augmentation-based | Data augmentation, mixup |
| Family VI (Forbidden) | Theoretical limit | Information-theoretic bounds |
| Family VII (Singular) | Hard instances | NP-hard optimization |
| Family VIII (Horizon) | Open problems | AGI, few-shot reasoning |

---

## The Eight Families: ML Problem Classes

### Family I: Standard Supervised Learning

**Hypostructure:** Immediate certificate satisfaction ($K^+$).

**ML Equivalent:** Well-posed supervised learning with abundant labeled data.

**Strategy:** Standard training with SGD variants.

| Problem Type | Example | Technique |
|-------------|---------|-----------|
| Classification | ImageNet | CNNs + SGD |
| Regression | House prices | Linear/neural regression |
| Sequence | Language modeling | Transformers + Adam |

**Certificate:** $K^+ = (\text{loss} < \epsilon, \text{accuracy} > 1-\delta)$

---

### Family II: Semi-Supervised/Self-Supervised

**Hypostructure:** Boundary certificates ($K^\circ$).

**ML Equivalent:** Limited labels, abundant unlabeled data.

**Strategy:** Self-training, contrastive learning, pseudo-labeling.

| Problem Type | Example | Technique |
|-------------|---------|-----------|
| SSL Classification | Limited ImageNet | FixMatch, MixMatch |
| Contrastive | Representation learning | SimCLR, MoCo |
| Masked | Language models | BERT, MAE |

**Certificate:** $K^\circ = (\text{labeled loss}, \text{unlabeled consistency})$

---

### Family III: Transfer/Meta-Learning

**Hypostructure:** Equivalence certificates ($K^{\sim}$).

**ML Equivalent:** Leverage knowledge from related tasks.

**Strategy:** Pretrain-finetune, domain adaptation, meta-learning.

| Problem Type | Example | Technique |
|-------------|---------|-----------|
| Transfer | NLP downstream | BERT finetuning |
| Domain adaptation | Sim-to-real | DANN, adversarial |
| Meta-learning | Few-shot | MAML, Prototypical |

**Certificate:** $K^{\sim} = (\text{source task}, \text{transfer gap})$

---

### Family IV: Curriculum/Incremental Learning

**Hypostructure:** Re-entry certificates ($K^{\mathrm{re}}$).

**ML Equivalent:** Progressive training, continual learning.

**Strategy:** Start easy, increase difficulty; manage forgetting.

| Problem Type | Example | Technique |
|-------------|---------|-----------|
| Curriculum | RL with sparse rewards | Self-paced learning |
| Continual | Lifelong learning | EWC, PackNet |
| Progressive | Growing networks | NetGrow |

**Certificate:** $K^{\mathrm{re}} = (\text{stage completion}, \text{no forgetting})$

---

### Family V: Augmentation/Regularization-Heavy

**Hypostructure:** Extension certificates ($K^{\mathrm{ext}}$).

**ML Equivalent:** Improve generalization via synthetic/augmented data.

**Strategy:** Heavy augmentation, regularization, ensemble methods.

| Problem Type | Example | Technique |
|-------------|---------|-----------|
| Data augmentation | Small datasets | AutoAugment |
| Regularization | Overparameterized | Dropout, weight decay |
| Ensemble | Uncertainty | MC-Dropout, deep ensembles |

**Certificate:** $K^{\mathrm{ext}} = (\text{augmentation policy}, \text{regularization strength})$

---

### Family VI: Information-Theoretic Limits

**Hypostructure:** Blocked certificates ($K^{\mathrm{blk}}$).

**ML Equivalent:** Problems with proven sample complexity or information barriers.

**Strategy:** Accept theoretical limits; optimize within bounds.

| Problem Type | Example | Technique |
|-------------|---------|-----------|
| Sample complexity | PAC bounds | Optimal sample usage |
| Capacity limits | VC dimension | Architecture tuning |
| Privacy | DP-SGD | Noise calibration |

**Certificate:** $K^{\mathrm{blk}} = (\text{lower bound}, \text{matching upper bound})$

---

### Family VII: NP-Hard/Intractable Learning

**Hypostructure:** Morphism certificates ($K^{\mathrm{morph}}$).

**ML Equivalent:** Optimization problems with proven computational hardness.

**Strategy:** Approximation, heuristics, relaxation.

| Problem Type | Example | Technique |
|-------------|---------|-----------|
| Combinatorial | Sparse coding | Greedy approximation |
| Non-convex | Deep learning | Local search + restarts |
| Adversarial | Robust training | PGD, certified defenses |

**Certificate:** $K^{\mathrm{morph}} = (\text{hardness proof}, \text{approximation ratio})$

---

### Family VIII: Open/Horizon Problems

**Hypostructure:** Incompleteness certificates ($K^{\mathrm{inc}}$).

**ML Equivalent:** Unsolved problems at the research frontier.

**Strategy:** Best current methods; acknowledge limitations.

| Problem Type | Example | Technique |
|-------------|---------|-----------|
| AGI | General reasoning | LLMs, neuro-symbolic |
| Few-shot | Novel categories | In-context learning |
| Causal | Interventional | Causal discovery |

**Certificate:** $K^{\mathrm{inc}} = (\text{OPEN}, \text{best known bounds})$

---

## The Twenty-One Phases: Learning Lifecycle

The 21 phases correspond to decision points in the ML lifecycle:

### Data Phases (1-5)

| Phase | Name | Decision |
|-------|------|----------|
| 1 | Data Quantity | Is data sufficient? |
| 2 | Data Quality | Is data clean? |
| 3 | Label Availability | Are labels available? |
| 4 | Distribution | Is distribution known? |
| 5 | Augmentation | What augmentation? |

### Architecture Phases (6-10)

| Phase | Name | Decision |
|-------|------|----------|
| 6 | Architecture Type | CNN, RNN, Transformer? |
| 7 | Depth/Width | How deep/wide? |
| 8 | Connectivity | Dense, residual, attention? |
| 9 | Initialization | How to initialize? |
| 10 | Pretrained | Use pretrained weights? |

### Training Phases (11-17)

| Phase | Name | Decision |
|-------|------|----------|
| 11 | Optimizer | SGD, Adam, LAMB? |
| 12 | Learning Rate | Schedule? Warmup? |
| 13 | Batch Size | Small or large batch? |
| 14 | Regularization | Dropout? Weight decay? |
| 15 | Early Stopping | When to stop? |
| 16 | Hyperparameter | AutoML? Grid search? |
| 17 | Convergence | Has training converged? |

### Evaluation Phases (18-21)

| Phase | Name | Decision |
|-------|------|----------|
| 18 | Validation | Holdout? Cross-val? |
| 19 | Metrics | Accuracy? F1? AUC? |
| 20 | Generalization | Does it generalize? |
| 21 | Deployment | Ready for production? |

---

## Proof Sketch

### Step 1: Problem Classification via Features

**Claim:** Every ML problem admits a unique (Family, Phase) classification.

**Procedure:**
1. Extract feature vector from problem: data size, label availability, task type, constraints
2. Determine Family from dominant structural property
3. Determine Phase from current training stage

**Reference:** Brazdil, P., et al. (2008). *Metalearning: Applications to Data Mining*. Springer.

### Step 2: Strategy Uniqueness

**Claim:** The (Family, Phase) pair uniquely determines optimal strategy.

**Proof:**
1. **Completeness:** 8 families partition all problem types
2. **Disjointness:** Families are mutually exclusive by definition
3. **Strategy Assignment:** Each of 168 cells has canonical algorithm

**Reference:** Rice, J. R. (1976). The algorithm selection problem. *Advances in Computers*.

### Step 3: AutoML as Table Lookup

**Connection:** AutoML systems approximate the selection table:
- Features → (Family, Phase) estimation
- Meta-model → Strategy prediction
- Portfolio → Coverage of table entries

**Reference:** Hutter, F., et al. (2019). *Automated Machine Learning*. Springer.

### Step 4: No Free Lunch Resolution

**Theorem (Wolpert 1997).** No algorithm is universally optimal.

**Resolution:** The selection table organizes specialization—each cell has its optimal algorithm. The NFL theorem is satisfied: no single algorithm fills the entire table.

**Reference:** Wolpert, D. H., Macready, W. G. (1997). No free lunch theorems for optimization. *IEEE Trans. Evol. Comp.*

### Step 5: Connections to Meta-Learning

**Meta-Learning:** Learn to predict which algorithm works best.

**Connection:** Meta-learning learns the selection table from experience:
- Task features → Problem classification
- Algorithm performance → Table entries
- Meta-model → Table lookup approximation

### Step 6: Phase-Dependent Strategies

**Example (Learning Rate Selection):**
- Phase 12 (LR Schedule) × Family I → Standard warmup + decay
- Phase 12 × Family III (Transfer) → Lower LR for pretrained, higher for new
- Phase 12 × Family IV (Curriculum) → Stage-dependent LR

### Step 7: Hyperparameter Recommendation

**Application:** Given problem classification, recommend hyperparameters.

**Procedure:**
1. Classify problem into (Family, Phase)
2. Look up recommended hyperparameter ranges
3. Fine-tune within recommended range

### Step 8: Architecture Selection

**Application:** Given problem classification, select architecture.

| Family | Recommended Architecture |
|--------|-------------------------|
| I (Standard) | ResNet, BERT |
| II (SSL) | SimCLR backbone |
| III (Transfer) | Pretrained + head |
| IV (Curriculum) | Progressive widening |
| V (Augmentation) | EfficientNet + augment |

### Step 9: Training Strategy Selection

**Application:** Given (Family, Phase), select training strategy.

| Example Cell | Strategy |
|--------------|----------|
| (I, 11) | SGD with momentum |
| (II, 11) | Adam with consistency loss |
| (III, 11) | AdamW with discriminative LR |
| (IV, 11) | Stage-wise optimizer |

### Step 10: Compilation Theorem

**Theorem (Algorithm Selection Lock):**

1. **Classification:** Every ML problem maps to (Family, Phase)
2. **Uniqueness:** Each cell has unique optimal strategy
3. **Coverage:** 168 cells cover all problem/phase combinations
4. **Lock:** Wrong strategy for cell incurs performance penalty

**Applications:**
- Automated algorithm selection
- Hyperparameter recommendation
- Architecture search guidance
- Training recipe generation

---

## Key AI/ML Techniques Used

1. **Problem Classification:**
   $$\sigma(\mathcal{P}) = (\text{Family}, \text{Phase})$$

2. **Strategy Lookup:**
   $$\text{Algorithm} = \mathcal{S}[\text{Family}, \text{Phase}]$$

3. **Meta-Learning:**
   $$\hat{\mathcal{S}} = \arg\min \mathbb{E}[\mathcal{L}(\mathcal{S}(f(\mathcal{P})), \mathcal{P})]$$

4. **AutoML:**
   $$\theta^* = \text{AutoML}(\mathcal{P}, \text{budget})$$

---

## Literature References

- Rice, J. R. (1976). The algorithm selection problem. *Advances in Computers*.
- Wolpert, D. H., Macready, W. G. (1997). No free lunch theorems for optimization. *IEEE Trans. Evol. Comp.*
- Brazdil, P., et al. (2008). *Metalearning: Applications to Data Mining*. Springer.
- Hutter, F., et al. (2019). *Automated Machine Learning*. Springer.
- Vanschoren, J. (2019). Meta-learning. *AutoML Book*.
- Elsken, T., Metzen, J. H., Hutter, F. (2019). Neural architecture search: A survey. *JMLR*.

