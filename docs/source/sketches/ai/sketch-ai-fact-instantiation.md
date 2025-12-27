---
title: "FACT-Instantiation - AI/RL/ML Translation"
---

# FACT-Instantiation: Model Instantiation Metatheorem

## Overview

The instantiation metatheorem establishes that every valid abstract learning specification (architecture, objective, constraints) can be realized by concrete neural network parameters. This connects abstract expressivity requirements to trainable implementations.

**Original Theorem Reference:** {prf:ref}`mt-fact-instantiation`

---

## AI/RL/ML Statement

**Theorem (Instantiation Metatheorem, ML Form).**
For every consistent learning specification $\Pi$, there exists a valid instantiation:

1. **Existence:** $\exists \mathcal{I}: \Pi \to (\theta, \mathcal{L}, \mathcal{D})$ satisfying all requirements

2. **Uniqueness (up to equivalence):** Instantiations are unique modulo parameter symmetries

3. **Functoriality:** Specification morphisms induce instantiation morphisms

**Corollary (Universal Approximation Instantiation).**
For any continuous function $f: \mathbb{R}^d \to \mathbb{R}^m$ and $\varepsilon > 0$, there exists a neural network $f_\theta$ with $\|f - f_\theta\|_\infty < \varepsilon$.

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Abstract permit $\Pi$ | Architecture specification | $(L, d_\ell, \sigma, \text{loss})$ |
| Concrete data | Trained model | $(\theta^*, \mathcal{L}^*)$ |
| Instantiation $\mathcal{I}$ | Model initialization + training | Random init → trained params |
| Consistency | Feasibility | Specification is realizable |
| Uniqueness up to equivalence | Parameter symmetry | Weight space symmetries |
| Functoriality | Transfer learning | Spec morphism → param morphism |
| Universal instantiation | Universal approximation | Any function is approximable |
| Moduli space | Weight space orbits | $\Theta / \text{Sym}$ |

---

## Model Instantiation Framework

### Specification to Implementation

**Definition.** A learning specification $\Pi$ consists of:
- Architecture: $(L, \{d_\ell\}, \sigma)$ (depth, widths, activations)
- Objective: $\mathcal{L}: \Theta \times \mathcal{D} \to \mathbb{R}$
- Constraints: $C_i(\theta) \leq 0$
- Data: $\mathcal{D} = \{(x_i, y_i)\}$

**Instantiation:** Map $\mathcal{I}: \Pi \to \theta^*$ producing trained parameters.

### Connection to Realizability

| ML Property | Hypostructure Property |
|-------------|------------------------|
| Expressivity | Permit satisfaction |
| Trainability | Constructive existence |
| Symmetry | Gauge equivalence |
| Transfer | Functoriality |

---

## Proof Sketch

### Step 1: Specification Consistency

**Consistency Definition.** A specification $\Pi$ is **consistent** if:
- Architecture has sufficient capacity for task
- Objective is bounded below
- Constraints are satisfiable
- Data is compatible with target function

**Verification:**
1. Width bounds: $d_\ell \geq d_{\min}(\text{task})$
2. Depth bounds: $L \geq L_{\min}(\text{task})$
3. Activation: $\sigma$ is nonlinear (for expressivity)
4. Loss: $\mathcal{L} \geq 0$ (bounded below)

### Step 2: Existence via Universal Approximation

**Theorem (Cybenko, 1989).** For any $f \in C([0,1]^d)$ and $\varepsilon > 0$, there exists a single-hidden-layer network $f_\theta$ with:
$$\|f - f_\theta\|_\infty < \varepsilon$$

**Reference:** Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *MCSS*, 2(4).

**Constructive:** The proof provides explicit (though potentially large) width requirements.

### Step 3: Depth-Width Tradeoffs

**Deep Networks (Telgarsky, 2016).** There exist functions requiring exponentially many neurons in shallow networks but polynomial in deep ones.

**Reference:** Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.

**Instantiation Choice:** Depth vs width is a design choice within consistent specifications.

### Step 4: Training as Instantiation

**Gradient Descent Instantiation.** Given specification $\Pi$:
1. Initialize: $\theta_0 \sim \mathcal{N}(0, \sigma^2 I)$
2. Train: $\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$
3. Output: $\theta^* = \lim_{t \to \infty} \theta_t$

**Convergence:** Under mild conditions, gradient descent finds a minimizer.

**Reference:** Bottou, L., Curtis, F., Nocedal, J. (2018). Optimization methods. *SIAM Review*, 60(2).

### Step 5: Uniqueness up to Symmetry

**Weight Space Symmetries.** Neural networks have symmetries:
- **Permutation:** Reordering neurons in hidden layer
- **Scaling:** Rescaling weights with inverse rescaling in next layer

**Equivalence Class:** $[\theta] = \{g \cdot \theta : g \in \text{Sym}\}$

**Theorem:** Instantiations are unique up to these symmetries.

**Reference:** Hecht-Nielsen, R. (1990). On the algebraic structure of feedforward networks. *IJCNN*.

### Step 6: Functorial Transfer

**Transfer Learning.** Given specification morphism $\phi: \Pi_1 \to \Pi_2$:
$$\mathcal{I}(\phi): \mathcal{I}(\Pi_1) \to \mathcal{I}(\Pi_2)$$

**Example:** Fine-tuning pretrained model for new task.

**Reference:** Yosinski, J., et al. (2014). How transferable are features? *NeurIPS*.

### Step 7: Specific Instantiations

**Example 1: Image Classification**
- Spec: $(L=50, d_\ell=512, \sigma=\text{ReLU}, \text{loss}=\text{CE})$
- Instantiation: ResNet-50 trained on ImageNet

**Example 2: Language Modeling**
- Spec: $(L=12, d=768, \sigma=\text{GELU}, \text{loss}=\text{CE})$
- Instantiation: BERT-base trained on Wikipedia

**Example 3: Reinforcement Learning**
- Spec: $(\pi_\theta, V_\phi, \gamma=0.99)$
- Instantiation: PPO agent trained on environment

### Step 8: Moduli of Instantiations

**Moduli Space.** Define:
$$\mathcal{M}(\Pi) = \{\theta^*: \theta^* \text{ minimizes } \mathcal{L}\} / \text{Sym}$$

**Finiteness:** For most practical specifications:
$$|\mathcal{M}(\Pi)| < \infty$$

(finitely many distinct minima up to symmetry).

**Reference:** Sagun, L., et al. (2017). Empirical analysis of the Hessian. *arXiv:1706.04454*.

### Step 9: Obstruction Theory

**Non-Instantiability.** Some specifications cannot be instantiated:
1. **Capacity Obstruction:** Width/depth insufficient for target
2. **Optimization Obstruction:** Loss landscape has no accessible minimum
3. **Data Obstruction:** Insufficient or contradictory data

**Detection:** Check expressivity bounds before training.

### Step 10: Compilation Theorem

**Theorem (Instantiation):**

1. **Inputs:** Consistent specification $\Pi$
2. **Outputs:** Trained model $\mathcal{I}(\Pi) = \theta^*$
3. **Guarantees:**
   - Existence for consistent $\Pi$
   - Uniqueness up to symmetry
   - Functorial in specification morphisms

**Algorithm (Model Instantiation):**
```python
def instantiate(specification):
    """Instantiate specification as trained model."""
    # Build architecture
    model = build_architecture(
        depth=specification.depth,
        widths=specification.widths,
        activation=specification.activation
    )

    # Initialize parameters
    model.initialize(method=specification.init_method)

    # Train
    optimizer = get_optimizer(specification.optimizer)
    for epoch in range(specification.epochs):
        for batch in specification.data:
            loss = specification.loss_fn(model, batch)
            loss.backward()
            optimizer.step()

    # Verify instantiation
    assert satisfies_constraints(model, specification.constraints)

    return model
```

**Applications:**
- Neural architecture search (find specification, then instantiate)
- Transfer learning (morphism of specifications)
- Model compression (constrained instantiation)
- AutoML (automated specification + instantiation)

---

## Key AI/ML Techniques Used

1. **Universal Approximation:**
   $$\forall f \in C, \forall \varepsilon > 0: \exists f_\theta: \|f - f_\theta\| < \varepsilon$$

2. **Symmetry Equivalence:**
   $$\theta \sim \theta' \iff \exists g \in \text{Sym}: \theta' = g \cdot \theta$$

3. **Gradient Descent Convergence:**
   $$\theta_t \to \theta^* \text{ as } t \to \infty$$

4. **Transfer Functoriality:**
   $$\mathcal{I}(\Pi_1 \to \Pi_2) = \mathcal{I}(\Pi_1) \to \mathcal{I}(\Pi_2)$$

---

## Literature References

- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *MCSS*, 2(4).
- Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.
- Bottou, L., Curtis, F., Nocedal, J. (2018). Optimization methods. *SIAM Review*, 60(2).
- Hecht-Nielsen, R. (1990). Algebraic structure of feedforward networks. *IJCNN*.
- Yosinski, J., et al. (2014). How transferable are features? *NeurIPS*.
- Sagun, L., et al. (2017). Empirical analysis of the Hessian. *arXiv:1706.04454*.
