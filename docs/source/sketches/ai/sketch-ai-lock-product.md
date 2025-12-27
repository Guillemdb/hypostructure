---
title: "LOCK-Product - AI/RL/ML Translation"
---

# LOCK-Product: Training Complexity Composition

## Overview

The training complexity composition lock shows that when individual learning tasks have established complexity bounds (sample complexity, training time), their product or joint learning preserves these bounds under appropriate conditions. This underlies multi-task learning, transfer learning, and the analysis of combined optimization problems.

**Original Theorem Reference:** {prf:ref}`lock-product`

---

## AI/RL/ML Statement

**Theorem (Training Complexity Composition, ML Form).**
Let $\mathcal{T}_A$ and $\mathcal{T}_B$ be learning tasks with established complexity bounds:
- $\mathcal{T}_A$ requires $N_A$ samples and $T_A$ training steps
- $\mathcal{T}_B$ requires $N_B$ samples and $T_B$ training steps

Under appropriate "coupling" conditions (task independence, parameter sharing, or curriculum structure):

$$N(\mathcal{T}_A \otimes \mathcal{T}_B) \geq f(N_A, N_B), \quad T(\mathcal{T}_A \otimes \mathcal{T}_B) \geq g(T_A, T_B)$$

**Corollary (Multi-Task Learning Bound).**
For $k$ independent tasks trained jointly:
$$N(\mathcal{T}_1 \otimes \cdots \otimes \mathcal{T}_k) \geq \max_i N_i \text{ (best case, shared structure)}$$
$$N(\mathcal{T}_1 \otimes \cdots \otimes \mathcal{T}_k) = \sum_i N_i \text{ (worst case, no sharing)}$$

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Product system | Multi-task learning | Joint training on tasks |
| Component Lock | Task-specific bound | Sample/time complexity |
| Product Lock | Joint training bound | Combined complexity |
| Coupling term | Task relatedness | Shared representations |
| Subcritical coupling | Independent tasks | No positive transfer |
| Strong coupling | Highly related tasks | Significant positive transfer |
| Semigroup structure | Parameter sharing | Common representations |
| Energy absorbability | Information reuse | Transfer efficiency |
| Parallel repetition | Multi-head training | Same backbone, different heads |
| Gap amplification | Error reduction | Combined generalization |

---

## Task Composition in Machine Learning

### Product Task Structure

**Definition.** The product task $\mathcal{T}_A \otimes \mathcal{T}_B$:
- **Independent:** Train separate models for each task
- **Joint:** Train single model for both tasks (multi-task)
- **Sequential:** Train on $A$ first, then $B$ (transfer)

### Connection to Sample Complexity

| ML Property | Hypostructure Property |
|-------------|------------------------|
| Sample complexity | Energy bound |
| Training time | Dissipation rate |
| Task relatedness | Coupling strength |
| Transfer efficiency | Bound reduction |

---

## Proof Sketch

### Step 1: Independent Task Composition

**Claim:** For independent tasks, sample complexity adds.

**Theorem.** If tasks $\mathcal{T}_A$ and $\mathcal{T}_B$ share no structure:
$$N(\mathcal{T}_A \otimes \mathcal{T}_B) = N_A + N_B$$

**Proof:**
1. No shared representation → no transfer
2. Must learn each task from scratch
3. Sample requirements add

**Reference:** Baxter, J. (2000). A model of inductive bias learning. *JAIR*.

### Step 2: Shared Structure Composition

**Claim:** With shared structure, sample complexity can be subadditive.

**Theorem (Maurer et al. 2016).** For $k$ tasks sharing representation $\phi$:
$$N(\mathcal{T}_1 \otimes \cdots \otimes \mathcal{T}_k) \leq N_\phi + k \cdot N_{\text{head}}$$

where $N_\phi$ is complexity for shared representation.

**Proof:**
1. Learn shared representation once
2. Learn task-specific heads separately
3. Total: shared cost + per-task head cost

**Reference:** Maurer, A., Pontil, M., Romera-Paredes, B. (2016). The benefit of multitask learning. *JMLR*.

### Step 3: Transfer Learning Bound

**Claim:** Pre-training reduces sample complexity for downstream tasks.

**Theorem.** With pretrained representation $\phi$ from task $\mathcal{T}_A$:
$$N_B^{\text{transfer}} \leq N_B^{\text{scratch}} \cdot (1 - \text{transfer\_efficiency})$$

**Transfer Efficiency:** Depends on task relatedness $\rho(A, B)$.

**Reference:** Tripuraneni, N., et al. (2020). Theory of overparameterized learning. *NeurIPS*.

### Step 4: Multi-Task Learning Analysis

**Setup.** Learn $k$ tasks jointly with shared backbone:
$$f_i(x) = h_i(\phi(x)) \quad i = 1, \ldots, k$$

**Bound Composition:**
- **Backbone:** $N_\phi$ samples to learn shared $\phi$
- **Heads:** $N_{h_i}$ samples per task head
- **Total:** $N = N_\phi + \sum_i N_{h_i}$

**Subadditivity:** If $k$ is large and tasks are related:
$$\frac{N}{k} < N_{\text{single-task}}$$

**Reference:** Caruana, R. (1997). Multitask learning. *Machine Learning*.

### Step 5: Negative Transfer

**Definition.** Negative transfer occurs when joint training hurts performance:
$$\mathcal{L}_{\text{joint}} > \mathcal{L}_{\text{separate}}$$

**Cause:** Task interference—gradients for different tasks conflict.

**Lock Condition:** Negative transfer corresponds to "strong coupling" that breaches the product bound.

**Reference:** Wang, Z., et al. (2019). Characterizing and avoiding negative transfer. *CVPR*.

### Step 6: Gradient Interference Analysis

**Definition.** Task gradients interfere when:
$$\nabla_\theta \mathcal{L}_A \cdot \nabla_\theta \mathcal{L}_B < 0$$

**Product Bound Preservation:** Bound holds when:
$$\mathbb{E}[\nabla \mathcal{L}_A \cdot \nabla \mathcal{L}_B] \geq 0$$

**Mitigation:** PCGrad, GradNorm project conflicting gradients.

**Reference:** Yu, T., et al. (2020). Gradient surgery for multi-task learning. *NeurIPS*.

### Step 7: Parallel Training Repetition

**Claim:** Training same model on $k$ independent datasets amplifies generalization.

**Theorem.** For $k$-fold data augmentation/repetition:
$$\text{Generalization gap} \propto 1/\sqrt{k \cdot N}$$

**Connection:** This is the ML analog of parallel repetition—more data (repetitions) reduces error.

**Reference:** Shorten, C., Khoshgoftaar, T. M. (2019). A survey on image data augmentation. *Journal of Big Data*.

### Step 8: Continual Learning Product

**Setup.** Sequential tasks $\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_k$:
$$\text{Learn } \mathcal{T}_1 \to \mathcal{T}_2 \to \cdots \to \mathcal{T}_k$$

**Product Bound Issue:** Catastrophic forgetting violates bound preservation:
$$\mathcal{L}_i \text{ increases while learning } \mathcal{T}_j \text{ for } j > i$$

**Lock Resolution:** EWC, PackNet preserve bounds by protecting important parameters.

**Reference:** Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting. *PNAS*.

### Step 9: Ensemble Composition

**Claim:** Ensemble of models combines bounds.

**Theorem.** For $k$ independent models:
$$\text{Error}(\text{ensemble}) \leq \frac{1}{k} \sum_i \text{Error}(f_i)$$

**Diversity Bonus:** If models are diverse:
$$\text{Error}(\text{ensemble}) \ll \text{avg Error}(f_i)$$

**Reference:** Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). Simple and scalable predictive uncertainty. *NeurIPS*.

### Step 10: Compilation Theorem

**Theorem (Training Complexity Composition):**

1. **Independent:** Sample complexity adds: $N_{A \otimes B} = N_A + N_B$
2. **Shared:** Subadditive with shared structure: $N_{A \otimes B} < N_A + N_B$
3. **Transfer:** Pre-training reduces downstream: $N_B^{\text{transfer}} < N_B^{\text{scratch}}$
4. **Lock:** Negative transfer breaches bound (requires intervention)

**Applications:**
- Multi-task learning design
- Transfer learning analysis
- Continual learning protection
- Ensemble construction

---

## Key AI/ML Techniques Used

1. **Sample Complexity Addition:**
   $$N(\mathcal{T}_A \otimes \mathcal{T}_B) \geq N_A + N_B$$

2. **Transfer Efficiency:**
   $$N_B^{\text{transfer}} = N_B \cdot (1 - \eta_{\text{transfer}})$$

3. **Gradient Alignment:**
   $$\text{aligned} \iff \nabla \mathcal{L}_A \cdot \nabla \mathcal{L}_B \geq 0$$

4. **Ensemble Bound:**
   $$\text{Error}(\text{ensemble}) \leq \frac{1}{k} \sum_i \text{Error}_i$$

---

## Literature References

- Baxter, J. (2000). A model of inductive bias learning. *JAIR*.
- Caruana, R. (1997). Multitask learning. *Machine Learning*.
- Maurer, A., Pontil, M., Romera-Paredes, B. (2016). The benefit of multitask learning. *JMLR*.
- Tripuraneni, N., et al. (2020). Theory of overparameterized learning. *NeurIPS*.
- Yu, T., et al. (2020). Gradient surgery for multi-task learning. *NeurIPS*.
- Wang, Z., et al. (2019). Characterizing and avoiding negative transfer. *CVPR*.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting. *PNAS*.

