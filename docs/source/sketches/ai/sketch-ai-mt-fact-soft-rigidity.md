# FACT-Soft-Rigidity: Continual Learning via Elastic Weight Consolidation

## AI/RL/ML Statement

### Original Statement (Hypostructure)
*Reference: mt-fact-soft-rigidity*

**[FACT-SoftRigidity] Soft-to-Rigidity Compilation (Hybrid).** Rigidity is derived via monotonicity-interface producing a rigidity-check that feeds into Lock/obstruction.

**Soft Hypotheses:**
$$K_{\mathrm{Mon}_\phi}^+ \wedge K_{\mathrm{KM}}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Produces:**
$$K_{\mathrm{Rigidity}_T}^+$$

**Hybrid Mechanism (3 Steps):**

1. **(Monotonicity Check)** By $K_{\mathrm{Mon}_\phi}^+$, the almost-periodic solution $u^*$ satisfies a monotonicity identity forcing dispersion or concentration to a stationary/self-similar profile.

2. **(Lojasiewicz Closure)** By $K_{\mathrm{LS}_\sigma}^+$, near critical points the gradient bound $\|\nabla \Phi(u^*)\| \geq c|\Phi(u^*) - \Phi(V)|^{1-\theta}$ prevents oscillation, forcing convergence to equilibrium $V$.

3. **(Lock Exclusion)** By $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$, any "bad" $u^*$ (counterexample to regularity) has $\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}, \mathcal{H}) = \emptyset$. If $u^* \notin \mathcal{L}_T$ (library), it would embed a bad pattern which the Lock blocks. Therefore $u^* \in \mathcal{L}_T$.

**Key Insight:** Rigidity becomes **categorical** (Lock) rather than purely analytic. The monotonicity interface provides the analytic input; Lock provides the conclusion.

---

## AI/RL/ML Formulation

### Setup

Consider a continual learning problem where:

- **Parameter space:** Neural network weights $\theta \in \mathbb{R}^d$
- **Task sequence:** $\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_n$ (sequential tasks)
- **Height/Energy:** Loss function $L_k(\theta)$ for task $\mathcal{T}_k$
- **Dissipation:** Learning dynamics $\theta_{t+1} = \theta_t - \eta \nabla L_k(\theta_t)$
- **Value function:** $V(\theta) = \sum_{k=1}^n \mathbb{E}_{(x,y) \sim \mathcal{D}_k}[L_k(\theta; x, y)]$ (aggregate performance)
- **Rigidity:** Weight constraints preventing catastrophic forgetting

The "soft rigidity" mechanism allows adaptation to new tasks while preserving knowledge from previous tasks through importance-weighted parameter constraints.

### Statement (AI/RL/ML Version)

**Theorem (Elastic Weight Consolidation / Soft Rigidity).** Let $(\Theta, \{\mathcal{T}_k\}_{k=1}^n, L, F)$ be a continual learning system with:
- Parameter space $\Theta \subseteq \mathbb{R}^d$
- Sequential tasks $\{\mathcal{T}_k\}$
- Loss functions $\{L_k\}$
- Fisher information matrices $\{F_k\}$ measuring parameter importance

Under appropriate conditions, the **soft rigidity mechanism** prevents catastrophic forgetting while allowing adaptation:

**Soft Hypotheses (AI/RL/ML):**

| **Condition** | **AI/RL Interpretation** |
|---------------|--------------------------|
| $K_{\mathrm{Mon}_\phi}^+$ (Monotonicity) | Loss decreases during training: $L_k(\theta_{t+1}) \leq L_k(\theta_t)$ |
| $K_{\mathrm{KM}}^+$ (Critical Element) | Convergence to local minima for each task |
| $K_{\mathrm{LS}_\sigma}^+$ (Lojasiewicz) | Polyak-Lojasiewicz condition near optima |
| $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ (Lock) | Importance weights block destructive updates |

**Produces:** $K_{\mathrm{Rigidity}_T}^+$ (No Catastrophic Forgetting Certificate)

**Formal Statement:** Given:
1. **Fisher Information (Stiffness):** $F_k = \mathbb{E}\left[\nabla_\theta \log p(y|x,\theta) \nabla_\theta \log p(y|x,\theta)^\top\right]$ measures parameter importance for task $k$
2. **EWC Regularization:** $L_{\mathrm{EWC}}(\theta) = L_{n}(\theta) + \frac{\lambda}{2} \sum_{k < n} (\theta - \theta_k^*)^\top F_k (\theta - \theta_k^*)$
3. **Soft Constraint:** Parameters can move, but with resistance proportional to importance

Then learning on new task $\mathcal{T}_n$ satisfies:
$$\underbrace{L_k(\theta_n^*) - L_k(\theta_k^*)}_{\text{Forgetting on task } k} \leq \epsilon_k(\lambda, F_k)$$

where $\epsilon_k \to 0$ as $\lambda \to \infty$ (rigidity limit) or $\epsilon_k$ is controlled by the spectral properties of $F_k$.

---

## Terminology Translation Table

| Hypostructure Term | AI/RL/ML Equivalent |
|--------------------|---------------------|
| Rigidity $K_{\mathrm{Rigidity}_T}^+$ | Weight rigidity / Elastic Weight Consolidation |
| Soft rigidity | Prevent catastrophic forgetting while allowing adaptation |
| Stiffness parameter | Fisher information / importance weights |
| Monotonicity $K_{\mathrm{Mon}_\phi}^+$ | Training loss decreases monotonically |
| Almost-periodic solution $u^*$ | Weights cycling near local minimum |
| Lojasiewicz inequality $K_{\mathrm{LS}_\sigma}^+$ | Polyak-Lojasiewicz condition for convergence |
| Lock obstruction $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | Importance-weighted constraints block bad updates |
| Library $\mathcal{L}_T$ | Set of valid weight configurations |
| Critical element $u^*$ | Network weights at task convergence $\theta_k^*$ |
| Energy functional $\Phi$ | Multi-task loss $L(\theta) = \sum_k L_k(\theta)$ |
| Profile decomposition | Task-specific parameter subspaces |
| Equilibrium $V$ | Pareto-optimal multi-task solution |
| Dispersion | Knowledge spreading across parameters |
| Concentration | Knowledge localization in specific parameters |
| Bad pattern $\mathbb{H}_{\mathrm{bad}}$ | Weight update causing catastrophic forgetting |
| Interface permit | Continual learning constraint satisfaction |
| Height equivalence | Loss-regret relationship |
| Symmetry group $G$ | Parameter permutation symmetries |

---

## Proof Sketch

### Step 1: Monotonicity Check (Training Convergence)

**Claim:** Training on each task converges to a local minimum with monotonically decreasing loss.

**Setup:** For task $\mathcal{T}_k$ with loss $L_k(\theta)$ and gradient descent:
$$\theta_{t+1} = \theta_t - \eta \nabla L_k(\theta_t)$$

**Gradient Descent Convergence [Nesterov 2004]:**

For $\beta$-smooth loss functions with learning rate $\eta \leq 1/\beta$:
$$L_k(\theta_{t+1}) \leq L_k(\theta_t) - \frac{\eta}{2} \|\nabla L_k(\theta_t)\|^2$$

This is the **monotonicity condition** $K_{\mathrm{Mon}_\phi}^+$: the loss (height) strictly decreases unless at a critical point.

**Almost-Periodicity Interpretation:**

Without regularization, training on task $\mathcal{T}_n$ may oscillate near the minimum of $L_n$ while $L_k$ for $k < n$ increases -- the weights are "almost-periodic" in the joint loss landscape, cycling without converging to a good multi-task solution.

**Certificate:** $K_{\mathrm{Mon}}^+ = (\text{GD dynamics}, \beta\text{-smooth}, \eta \leq 1/\beta)$

---

### Step 2: Lojasiewicz Closure (Convergence to Equilibrium)

**Claim:** Near critical points, the Polyak-Lojasiewicz condition forces convergence rather than oscillation.

**Polyak-Lojasiewicz Condition [Polyak 1963]:**

A function $L$ satisfies PL with constant $\mu$ if:
$$\|\nabla L(\theta)\|^2 \geq 2\mu (L(\theta) - L^*)$$

This is the AI/RL analogue of the Lojasiewicz inequality:
$$\|\nabla \Phi(u)\| \geq c|\Phi(u) - \Phi(V)|^{1-\theta}$$

with exponent $\theta = 1/2$ (quadratic case).

**Convergence Theorem [Karimi et al. 2016]:**

Under PL condition with $\eta \leq 1/\beta$:
$$L(\theta_T) - L^* \leq (1 - \mu\eta)^T (L(\theta_0) - L^*)$$

The training converges **exponentially fast** to the minimum -- no oscillation or cycling.

**Multi-Task Extension:**

For EWC loss $L_{\mathrm{EWC}}(\theta) = L_n(\theta) + \frac{\lambda}{2} \sum_{k<n} (\theta - \theta_k^*)^\top F_k (\theta - \theta_k^*)$:

The quadratic regularizer ensures PL condition holds near $\theta_k^*$ for all previous tasks. The Fisher information $F_k$ acts as the **stiffness matrix** controlling convergence speed.

**Certificate:** $K_{\mathrm{LS}}^+ = (\mu\text{-PL}, \theta = 1/2, F_k \text{ positive definite})$

---

### Step 3: Lock Exclusion (Importance-Weighted Blocking)

**Claim:** The Fisher information matrix blocks updates that would cause catastrophic forgetting.

**Fisher Information as Importance (Stiffness):**

For task $\mathcal{T}_k$ with parameters $\theta_k^*$:
$$F_k = \mathbb{E}_{(x,y) \sim \mathcal{D}_k}\left[\nabla_\theta \log p_\theta(y|x)|_{\theta_k^*} \cdot \nabla_\theta \log p_\theta(y|x)|_{\theta_k^*}^\top\right]$$

The diagonal entries $(F_k)_{ii}$ measure how important parameter $\theta_i$ is for task $k$.

**Lock Mechanism:**

The EWC penalty acts as a **categorical lock** preventing bad patterns:
$$\text{Penalty}(\theta) = \frac{\lambda}{2} \sum_{k<n} \sum_i (F_k)_{ii} (\theta_i - \theta_{k,i}^*)^2$$

**High importance $(F_k)_{ii}$:** Parameter $\theta_i$ is critical for task $k$ -- strong resistance to change.

**Low importance $(F_k)_{ii}$:** Parameter $\theta_i$ is not critical -- free to adapt.

**Bad Pattern Exclusion:**

A "bad" update $\Delta\theta$ that causes forgetting on task $k$ satisfies:
$$\Delta L_k = L_k(\theta + \Delta\theta) - L_k(\theta_k^*) \approx \frac{1}{2} \Delta\theta^\top H_k \Delta\theta$$

where $H_k \approx F_k$ (Fisher approximates Hessian for log-likelihood).

The EWC penalty ensures:
$$\|\Delta\theta\|_{F_k}^2 = \Delta\theta^\top F_k \Delta\theta \geq c \cdot \Delta L_k$$

If $\Delta L_k$ is large (catastrophic forgetting), the penalty becomes prohibitive -- the **Lock blocks** the bad update.

**Library Classification:**

The set of valid weight configurations is:
$$\mathcal{L}_T = \left\{\theta : L_k(\theta) - L_k(\theta_k^*) \leq \epsilon_k \text{ for all } k\right\}$$

EWC ensures $\theta_n^* \in \mathcal{L}_T$ by construction.

**Certificate:** $K_{\mathrm{Lock}}^{\mathrm{blk}} = (F_k, \lambda, \epsilon_k\text{-tolerance})$

---

### Step 4: Rigidity Certificate (No Catastrophic Forgetting)

**Theorem (EWC Forgetting Bound) [Kirkpatrick et al. 2017]:**

For EWC with regularization strength $\lambda$ and Fisher information $F_k$:

$$L_k(\theta_n^*) - L_k(\theta_k^*) \leq \frac{\|\nabla L_n(\theta_k^*)\|^2}{2\lambda \cdot \lambda_{\min}(F_k)}$$

where $\lambda_{\min}(F_k)$ is the smallest eigenvalue of $F_k$.

**Interpretation:**

1. **Large $\lambda$:** Strong rigidity, minimal forgetting, but limited plasticity
2. **Large $\lambda_{\min}(F_k)$:** All parameters important, uniform protection
3. **Small $\|\nabla L_n(\theta_k^*)\|$:** New task compatible with old solution

**Rigidity-Plasticity Trade-off:**

The soft rigidity mechanism balances:
- **Rigidity (prevent forgetting):** $\lambda \to \infty$ freezes weights
- **Plasticity (enable learning):** $\lambda \to 0$ allows full adaptation

The optimal $\lambda^*$ satisfies:
$$\lambda^* = \arg\min_\lambda \left[ \underbrace{L_n(\theta^*(\lambda))}_{\text{New task loss}} + \underbrace{\sum_{k<n} \Delta L_k(\lambda)}_{\text{Forgetting}} \right]$$

**Final Certificate:**
$$K_{\mathrm{Rigidity}}^+ = (\{F_k\}, \lambda, \{\epsilon_k\}, \mathcal{L}_T)$$

---

## Connections to Classical Results

### 1. Elastic Weight Consolidation (Kirkpatrick et al., 2017)

**Statement:** EWC prevents catastrophic forgetting by adding a quadratic penalty anchoring weights to previous task optima, weighted by Fisher information.

**EWC Loss:**
$$L_{\mathrm{EWC}}(\theta) = L_B(\theta) + \frac{\lambda}{2} \sum_i F_i^A (\theta_i - \theta_A^*)^2$$

where:
- $L_B(\theta)$: loss on new task B
- $\theta_A^*$: optimal weights for previous task A
- $F_i^A$: Fisher information for parameter $i$ on task A

**Connection to Soft-Rigidity:**

| Hypostructure | EWC |
|---------------|-----|
| Stiffness $\sigma$ | Fisher information $F$ |
| Lock obstruction | Quadratic penalty |
| Library $\mathcal{L}_T$ | Set of Pareto-optimal solutions |
| Monotonicity | Training convergence |
| Lojasiewicz | PL condition from quadratic regularization |

**Reference:** Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.

---

### 2. Synaptic Intelligence (Zenke et al., 2017)

**Statement:** Online importance estimation by tracking parameter contributions to loss reduction during training.

**Importance Measure:**
$$\Omega_k^i = \sum_t \frac{\partial L}{\partial \theta_i} \cdot \Delta\theta_i$$

Parameters that contribute more to loss reduction are more important.

**SI Loss:**
$$L_{\mathrm{SI}}(\theta) = L_B(\theta) + c \sum_i \Omega_i (\theta_i - \theta_A^*)^2$$

**Connection:** SI computes importance **online** during training, while EWC computes Fisher information at task completion. Both implement soft rigidity with different stiffness estimation.

**Reference:** Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. *ICML*.

---

### 3. Progressive Neural Networks (Rusu et al., 2016)

**Statement:** Prevent forgetting by freezing previous task columns and adding new lateral connections.

**Architecture:**
```
Task 1: [Column 1] (frozen)
Task 2: [Column 1] -> [Column 2] (new)
Task 3: [Column 1] -> [Column 2] -> [Column 3] (new)
```

**Connection to Rigidity:**

- **Hard rigidity:** Previous columns are completely frozen ($\lambda = \infty$)
- **Soft adaptation:** Lateral connections allow knowledge transfer
- **Library:** Each column represents a task-specific solution

Progressive networks implement **absolute rigidity** on previous parameters while allowing **full plasticity** on new parameters. This is the limiting case of soft rigidity.

**Reference:** Rusu, A. A., et al. (2016). Progressive neural networks. *arXiv:1606.04671*.

---

### 4. Memory Replay Methods (Shin et al., 2017)

**Statement:** Prevent forgetting by replaying samples from previous tasks during new task training.

**Generative Replay:**
1. Train generative model $G_A$ on task A data
2. When training on task B:
   - Sample $(x, y) \sim \mathcal{D}_B$ (real data)
   - Sample $(\tilde{x}, \tilde{y}) \sim G_A$ (generated data)
   - Train on both: $L(\theta) = L_B(\theta; x, y) + L_A(\theta; \tilde{x}, \tilde{y})$

**Connection to Soft-Rigidity:**

| Mechanism | Effect |
|-----------|--------|
| Replay samples | Implicit regularization toward $\theta_A^*$ |
| Generator quality | Controls forgetting bound |
| Replay ratio | Balances rigidity vs plasticity |

Replay methods achieve soft rigidity through **data-driven regularization** rather than explicit weight constraints.

**Reference:** Shin, H., Lee, J. K., Kim, J., & Kim, J. (2017). Continual learning with deep generative replay. *NeurIPS*.

---

### 5. PackNet and Parameter Isolation (Mallya & Lazebnik, 2018)

**Statement:** Identify and freeze important parameters per task; use remaining capacity for new tasks.

**Mechanism:**
1. Train on task A, prune unimportant weights
2. Freeze remaining weights (task A subnetwork)
3. Train on task B using pruned (freed) weights
4. Repeat

**Connection:**

- **Binary rigidity:** Parameters either fully rigid or fully plastic
- **Stiffness:** Determined by pruning threshold (importance > threshold = rigid)
- **Library:** Each task has its subnetwork $\mathcal{L}_k$

PackNet implements **hard parameter isolation** rather than soft constraints.

**Reference:** Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network by iterative pruning. *CVPR*.

---

### 6. Bayesian Continual Learning (Nguyen et al., 2018)

**Statement:** Maintain posterior distribution over weights; use posterior as prior for next task.

**Variational Continual Learning:**
$$p(\theta | \mathcal{D}_1, \ldots, \mathcal{D}_n) \propto p(\mathcal{D}_n | \theta) \cdot p(\theta | \mathcal{D}_1, \ldots, \mathcal{D}_{n-1})$$

The posterior from tasks $1, \ldots, n-1$ becomes the prior for task $n$.

**Connection:**

| Bayesian | EWC/Soft-Rigidity |
|----------|-------------------|
| Posterior precision | Fisher information |
| Prior | Previous task optimum |
| KL regularization | Quadratic penalty |

EWC is a **Laplace approximation** to Bayesian continual learning:
$$\log p(\theta | \mathcal{D}_A) \approx \text{const} - \frac{1}{2}(\theta - \theta_A^*)^\top F_A (\theta - \theta_A^*)$$

**Reference:** Nguyen, C. V., Li, Y., Bui, T. D., & Turner, R. E. (2018). Variational continual learning. *ICLR*.

---

## Implementation Notes

### Fisher Information Computation

**Empirical Fisher (Practical):**
```python
def compute_fisher(model, dataloader, num_samples=1000):
    """Compute diagonal Fisher information matrix."""
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    model.eval()
    for i, (x, y) in enumerate(dataloader):
        if i >= num_samples:
            break

        model.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

        for n, p in model.named_parameters():
            fisher[n] += p.grad.data ** 2 / num_samples

    return fisher
```

**True Fisher (Theoretical):**
$$F_{ij} = \mathbb{E}_{x \sim p(x)} \mathbb{E}_{y \sim p_\theta(y|x)}\left[\frac{\partial \log p_\theta(y|x)}{\partial \theta_i} \cdot \frac{\partial \log p_\theta(y|x)}{\partial \theta_j}\right]$$

The empirical Fisher uses labels from the dataset rather than sampling from the model.

---

### EWC Training Loop

```python
class EWC:
    def __init__(self, model, lambda_ewc=1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.optpar = {}  # Optimal parameters for each task
        self.task_count = 0

    def consolidate(self, dataloader):
        """After training on a task, consolidate knowledge."""
        # Store optimal parameters
        self.optpar[self.task_count] = {
            n: p.data.clone() for n, p in self.model.named_parameters()
        }

        # Compute Fisher information
        self.fisher[self.task_count] = compute_fisher(self.model, dataloader)
        self.task_count += 1

    def penalty(self):
        """Compute EWC penalty (soft rigidity)."""
        if self.task_count == 0:
            return 0

        loss = 0
        for task_id in range(self.task_count):
            for n, p in self.model.named_parameters():
                # Quadratic penalty weighted by Fisher
                loss += (self.fisher[task_id][n] *
                        (p - self.optpar[task_id][n]) ** 2).sum()

        return self.lambda_ewc * loss / 2

    def train_step(self, x, y, optimizer):
        """Training step with EWC regularization."""
        optimizer.zero_grad()

        output = self.model(x)
        task_loss = F.cross_entropy(output, y)
        ewc_loss = self.penalty()
        total_loss = task_loss + ewc_loss

        total_loss.backward()
        optimizer.step()

        return task_loss.item(), ewc_loss.item()
```

---

### Rigidity-Plasticity Trade-off

**Hyperparameter Selection:**

The regularization strength $\lambda$ controls the trade-off:

| $\lambda$ | Behavior | Use Case |
|-----------|----------|----------|
| $\lambda = 0$ | Full plasticity, catastrophic forgetting | Single task |
| $\lambda \approx 1$ | Balanced trade-off | Standard continual learning |
| $\lambda \approx 10^3-10^4$ | Strong rigidity, limited adaptation | Many similar tasks |
| $\lambda \to \infty$ | Complete rigidity (Progressive Networks) | Disjoint tasks |

**Adaptive Lambda:**
```python
def adaptive_lambda(task_similarity, base_lambda=1000):
    """Increase rigidity for similar tasks, decrease for different."""
    # High similarity -> low lambda (shared representations useful)
    # Low similarity -> high lambda (protect previous knowledge)
    return base_lambda / (task_similarity + 0.1)
```

---

### Multi-Head vs Single-Head

**Multi-Head (Separate Outputs):**
- Each task has separate output layer
- Shared representation, task-specific classifiers
- EWC on shared layers only

**Single-Head (Shared Output):**
- Unified output for all tasks
- More challenging: must discriminate between all classes
- EWC on entire network

```python
class ContinualNetwork(nn.Module):
    def __init__(self, shared_layers, num_tasks, classes_per_task, multi_head=True):
        super().__init__()
        self.shared = shared_layers
        self.multi_head = multi_head

        if multi_head:
            self.heads = nn.ModuleList([
                nn.Linear(shared_layers[-1].out_features, classes_per_task)
                for _ in range(num_tasks)
            ])
        else:
            self.head = nn.Linear(
                shared_layers[-1].out_features,
                num_tasks * classes_per_task
            )

    def forward(self, x, task_id=None):
        features = self.shared(x)
        if self.multi_head:
            return self.heads[task_id](features)
        else:
            return self.head(features)
```

---

### Certificate Verification

**Forgetting Metric:**
```python
def compute_forgetting(model, task_dataloaders, task_performances):
    """
    Compute backward transfer / forgetting.

    Args:
        task_performances: Dict[task_id, accuracy_at_completion]

    Returns:
        Average forgetting across previous tasks
    """
    current_perf = {}
    for task_id, loader in task_dataloaders.items():
        current_perf[task_id] = evaluate(model, loader)

    forgetting = {}
    for task_id in task_performances:
        forgetting[task_id] = max(0,
            task_performances[task_id] - current_perf[task_id]
        )

    return np.mean(list(forgetting.values()))
```

**Rigidity Certificate:**
```python
K_Rigidity = {
    'mode': 'SoftRigidity',
    'mechanism': 'ElasticWeightConsolidation',
    'evidence': {
        'fisher_matrices': {task_id: F_k for task_id, F_k in fisher.items()},
        'optimal_params': {task_id: theta_k for task_id, theta_k in optpar.items()},
        'lambda': lambda_ewc,
        'forgetting_bounds': {task_id: epsilon_k for task_id, epsilon_k in bounds.items()},
        'library': 'Pareto-optimal multi-task solutions'
    },
    'certificates': {
        'monotonicity': 'Training loss decreases',
        'lojasiewicz': 'PL condition from quadratic regularization',
        'lock': 'Fisher-weighted penalty blocks catastrophic updates'
    },
    'literature': 'Kirkpatrick et al. 2017'
}
```

---

## Literature

1. **Kirkpatrick, J., et al. (2017).** "Overcoming catastrophic forgetting in neural networks." *PNAS*, 114(13), 3521-3526. *Foundational EWC paper.*

2. **Zenke, F., Poole, B., & Ganguli, S. (2017).** "Continual learning through synaptic intelligence." *ICML*. *Online importance estimation.*

3. **Rusu, A. A., et al. (2016).** "Progressive neural networks." *arXiv:1606.04671*. *Hard rigidity via architectural expansion.*

4. **Shin, H., Lee, J. K., Kim, J., & Kim, J. (2017).** "Continual learning with deep generative replay." *NeurIPS*. *Memory replay approach.*

5. **Nguyen, C. V., Li, Y., Bui, T. D., & Turner, R. E. (2018).** "Variational continual learning." *ICLR*. *Bayesian perspective on continual learning.*

6. **Mallya, A., & Lazebnik, S. (2018).** "PackNet: Adding multiple tasks to a single network by iterative pruning." *CVPR*. *Parameter isolation.*

7. **Schwarz, J., et al. (2018).** "Progress & compress: A scalable framework for continual learning." *ICML*. *Knowledge distillation for continual learning.*

8. **Li, Z., & Hoiem, D. (2017).** "Learning without forgetting." *IEEE TPAMI*. *Knowledge distillation approach.*

9. **Lopez-Paz, D., & Ranzato, M. (2017).** "Gradient episodic memory for continual learning." *NeurIPS*. *Gradient projection methods.*

10. **Chaudhry, A., et al. (2019).** "Efficient lifelong learning with A-GEM." *ICLR*. *Averaged gradient episodic memory.*

11. **Kemker, R., et al. (2018).** "Measuring catastrophic forgetting in neural networks." *AAAI*. *Forgetting metrics and benchmarks.*

12. **Parisi, G. I., et al. (2019).** "Continual lifelong learning with neural networks: A review." *Neural Networks*. *Comprehensive survey.*

13. **Kenig, C., & Merle, F. (2006).** "Global well-posedness, scattering and blow-up for the energy-critical focusing non-linear wave equation." *Acta Math.* *Original rigidity theorem (mathematical analogue).*

14. **Duyckaerts, T., Kenig, C., & Merle, F. (2011).** "Universality of blow-up profile for small radial type II blow-up solutions." *J. Eur. Math. Soc.* *Profile classification via rigidity.*
