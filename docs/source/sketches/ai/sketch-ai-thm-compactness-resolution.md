---
title: "Compactness Resolution - AI/RL/ML Translation"
---

# THM-COMPACTNESS-RESOLUTION: Generalization via Finite Covering

## Overview

This document provides a complete AI/RL/ML translation of the Compactness Resolution theorem from the hypostructure framework. The theorem resolves the "Compactness Critique" by showing that generalization is decidable at runtime regardless of whether the hypothesis class is compact a priori. The translation establishes a formal correspondence with statistical learning theory, where hypothesis classes decompose into either finite covers (generalizing) or irreducible hard cores (non-generalizing).

**Original Theorem Reference:** {prf:ref}`thm-compactness-resolution`

---

## AI/RL/ML Statement

**Theorem (Compactness Resolution, ML Form).**
Let $\mathcal{H}$ be a hypothesis class with value function $V: \mathcal{S} \to \mathbb{R}$ (height) and policy $\pi: \mathcal{S} \to \Delta(\mathcal{A})$ (dissipation). At runtime, the learning algorithm executes a dichotomy check producing exactly one of:

1. **Concentration Branch (Finite Cover):** If the effective complexity concentrates ($\mathcal{N}(\mathcal{H}, \epsilon) < \infty$ for covering number $\mathcal{N}$), a **finite hypothesis cover** emerges via discretization. The generalization bound is satisfied constructively---the certificate $K_{\text{cover}}^+$ witnesses the concentration.

2. **Dispersion Branch (Uniform Convergence):** If complexity scatters uniformly ($\mathcal{R}_n(\mathcal{H}) \to 0$ for Rademacher complexity $\mathcal{R}_n$), compactness holds trivially. This triggers **Mode D.D (Dispersion/Global Generalization)**---a success state, not a failure.

**Conclusion:** Generalization is decidable regardless of whether the hypothesis class is compact *a priori*. The dichotomy is resolved at runtime, not assumed.

**Formal Statement:** For learning problem $\mathcal{P} = (\mathcal{H}, \mathcal{D}, L)$ with hypothesis class $\mathcal{H}$, data distribution $\mathcal{D}$, and loss $L$, define:

- **Covering number:** $\mathcal{N}(\mathcal{H}, \epsilon, \|\cdot\|) = \min\{|C| : \mathcal{H} \subseteq \bigcup_{h \in C} B_\epsilon(h)\}$
- **Rademacher complexity:** $\mathcal{R}_n(\mathcal{H}) = \mathbb{E}_{\sigma, S}\left[\sup_{h \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \sigma_i h(x_i)\right]$

The Compactness Resolution dichotomy:

| Outcome | Complexity Behavior | Generalization Certificate |
|---------|---------------------|---------------------------|
| **Concentration** | $\log \mathcal{N}(\mathcal{H}, \epsilon) \leq f(\epsilon^{-1})$ | Finite cover $\Rightarrow$ uniform convergence |
| **Dispersion** | $\mathcal{R}_n(\mathcal{H}) = O(n^{-1/2})$ | Rademacher bound $\Rightarrow$ generalization |
| **Hard Core** | $\mathcal{N}(\mathcal{H}, \epsilon) = \infty$ or $\mathcal{R}_n = \Omega(1)$ | No uniform convergence certificate |

---

## Terminology Translation Table

| Hypostructure Term | AI/RL/ML Equivalent |
|--------------------|---------------------|
| Energy concentration $\mu(V) > 0$ | Finite covering number: $\mathcal{N}(\mathcal{H}, \epsilon) < \infty$ |
| Energy dispersion $\mu(V) = 0$ | Vanishing Rademacher complexity: $\mathcal{R}_n \to 0$ |
| Height functional $\Phi$ | Value function $V(s)$, or negative loss $-L(\theta)$ |
| Dissipation density $\mathfrak{D}$ | Policy $\pi(a|s)$, or gradient magnitude $\|\nabla L\|$ |
| Canonical profile $V$ | Minimal $\epsilon$-cover $C_\epsilon \subset \mathcal{H}$ |
| Scaling limits | Discretization / quantization of hypothesis space |
| Profile extraction modulo $G$ | Covering modulo symmetries (equivariant covers) |
| Compactness axiom | Finite covering number / bounded VC dimension |
| Concentration-compactness dichotomy | Cover-or-disperse dichotomy |
| Certificate $K_{C_\mu}^+$ | Covering number certificate with size bound |
| Certificate $K_{C_\mu}^-$ | Rademacher bound certificate (dispersion) |
| Mode D.D (Dispersion/Global Existence) | Uniform convergence: $L_{\text{test}} \approx L_{\text{train}}$ |
| Node 3 runtime check | Generalization bound evaluation during training |
| Finite-energy profile | Finite hypothesis class / bounded capacity |
| Profile moduli space | Hypothesis equivalence classes under symmetries |

---

## Proof Sketch

### Setup: Generalization Framework

**Definition (Covering Number).** The $\epsilon$-covering number of hypothesis class $\mathcal{H}$ with respect to metric $d$ is:
$$\mathcal{N}(\mathcal{H}, \epsilon, d) = \min\left\{|C| : C \subset \mathcal{H}, \, \mathcal{H} \subseteq \bigcup_{h \in C} B_\epsilon(h)\right\}$$

**Definition (Rademacher Complexity).** The empirical Rademacher complexity of $\mathcal{H}$ on sample $S = \{x_1, \ldots, x_n\}$ is:
$$\hat{\mathcal{R}}_S(\mathcal{H}) = \mathbb{E}_\sigma\left[\sup_{h \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \sigma_i h(x_i)\right]$$
where $\sigma_i \in \{-1, +1\}$ are Rademacher random variables.

**Definition (Hard Core).** The hard core of hypothesis class $\mathcal{H}$ is the irreducible subclass $\mathcal{H}^* \subseteq \mathcal{H}$ after exhaustive application of covering/compression operations:
$$\mathcal{H} \xrightarrow{\text{cover}_1} \mathcal{H}_1 \xrightarrow{\text{cover}_2} \cdots \xrightarrow{\text{cover}_m} \mathcal{H}^*$$
where no further compression applies to $\mathcal{H}^*$.

**Covering Operations (Profile Extraction Analogue):** Each covering operation $\text{cover}_i$ is a polynomial-time transformation:
$$\text{cover}_i: \mathcal{H} \to C_\epsilon \text{ with } |C_\epsilon| < |\mathcal{H}|$$
The sequence of covers corresponds to the scaling/centering operations in concentration-compactness profile extraction.

---

### Step 1: The Dichotomy Check (Runtime Resolution)

**Claim (Generalization Dichotomy).** For any learning problem $\mathcal{P} = (\mathcal{H}, \mathcal{D}, L)$, the training phase produces exactly one of:
- **Concentration:** Finite cover $C_\epsilon$ with $|C_\epsilon| \leq f(\epsilon^{-1})$
- **Dispersion:** Certificate that $\mathcal{H}$ admits uniform convergence without finite cover

**Proof (Concentration-Compactness Analogue):**

**Phase 1: Covering Sequence (Scaling Limits)**

Apply covering operations exhaustively at decreasing $\epsilon$ scales:
$$\mathcal{H} = \mathcal{H}_0 \xrightarrow{\epsilon_1} C_1 \xrightarrow{\epsilon_2} C_2 \xrightarrow{\epsilon_3} \cdots$$

Each covering is the computational analogue of a scaling operation in profile extraction. The sequence terminates when either:
- A finite cover is achieved (concentration)
- Uniform convergence is established directly (dispersion)

**Phase 2: Complexity Classification**

After covering, measure the covering number growth:

**Case 2a (Concentration):** $\log \mathcal{N}(\mathcal{H}, \epsilon) \leq f(\epsilon^{-1})$

The hypothesis complexity concentrates into a finite cover. This corresponds to energy concentration with profile emergence in the hypostructure. The compactness axiom is satisfied constructively---the cover witnesses the concentration.

**Certificate Produced:**
$$K_{\text{cover}}^+ = (C_\epsilon, |C_\epsilon|, f(\epsilon^{-1}), \{\text{cover}_j\}_{j=1}^m)$$

**Case 2b (Dispersion):** The Rademacher complexity vanishes

If $\mathcal{R}_n(\mathcal{H}) = O(n^{-1/2})$, uniform convergence holds directly without requiring a finite cover. This is not a failure but a success: Mode D.D (global generalization).

**Certificate Produced:**
$$K_{\text{dispersion}}^- = (\mathcal{R}_n, O(n^{-1/2}), \text{generalization bound})$$

**Phase 3: Dichotomy Completeness**

The dichotomy is exhaustive because:
1. Covering operations either reduce complexity (finite cover) or don't (infinite/unbounded)
2. If no finite cover exists, either Rademacher vanishes (dispersion) or persists (hard core)
3. No fourth case exists by the fundamental theorem of statistical learning

---

### Step 2: Profile Extraction = Hypothesis Discretization

**Theorem (Cover as Limiting Profile).** The minimal $\epsilon$-cover $C_\epsilon^*$ extracted by exhaustive covering is the computational analogue of the limiting profile $V^*$ in concentration-compactness.

**Correspondence:**

| Concentration-Compactness | Covering Theory |
|---------------------------|-----------------|
| Sequence $u_n$ with bounded energy | Hypothesis class $\mathcal{H}$ |
| Symmetry group $G$ (scaling, translation) | Equivalence relation (hypothesis similarity) |
| Profile $V$ modulo $G$ | Cover element $h_c \in C_\epsilon$ modulo equivalence |
| Energy $\Phi(V^*)$ | Cover size $|C_\epsilon|$ |
| Profile decomposition $u_n = \sum_j g_n^{(j)} \cdot V^{(j)} + w_n$ | Hypothesis decomposition $h = h_c + \epsilon\text{-perturbation}$ |
| Remainder $w_n \to 0$ | Approximation error $\|h - h_c\| \leq \epsilon$ |

**Proof (Bahouri-Gerard Analogue):**

**Step 2.1 (Covering as Scaling):**

Each covering operation acts like a scaling operation:
$$\text{cover}_\epsilon: \mathcal{H} \mapsto C_\epsilon$$

with the "energy" (hypothesis complexity) preserved or decreased:
$$\log |C_\epsilon| \leq \log |\mathcal{H}|$$

**Step 2.2 (Convergence Modulo Symmetries):**

The covering sequence converges to a fixed point modulo the equivalence relation:
$$C_{\epsilon_t} \to C^* \text{ as } \epsilon_t \to 0$$

where hypotheses in the same cover element are treated as equivalent.

**Step 2.3 (Orthogonal Decomposition):**

Every hypothesis decomposes as:
$$h = h_c + r_\epsilon$$

where:
- $h_c \in C_\epsilon$ is the nearest cover element (concentrated "profile")
- $r_\epsilon$ is the approximation residual with $\|r_\epsilon\| \leq \epsilon$

This mirrors the Bahouri-Gerard profile decomposition:
$$u_n = \sum_{j=1}^J g_n^{(j)} \cdot V^{(j)} + w_n$$

---

### Step 3: Compactness Axiom = Bounded Covering Number

**Definition (Computational Compactness Axiom).** Hypothesis class $\mathcal{H}$ satisfies the compactness axiom if:
$$\forall \epsilon > 0.\ \mathcal{N}(\mathcal{H}, \epsilon) \leq f(\epsilon^{-1})$$

for some computable function $f$.

**Theorem (Compactness $\Leftrightarrow$ Uniform Convergence):**

The following are equivalent:
1. $\mathcal{H}$ satisfies the computational compactness axiom
2. $\mathcal{H}$ admits uniform convergence bounds
3. $\mathcal{H}$ is PAC-learnable via empirical risk minimization

**Proof:**

**(1) $\Rightarrow$ (2):** If all covers are bounded by $f(\epsilon^{-1})$, then by the covering number bound:
$$\Pr\left[\sup_{h \in \mathcal{H}} |L_S(h) - L_\mathcal{D}(h)| > \epsilon\right] \leq 2\mathcal{N}(\mathcal{H}, \epsilon/2) \cdot e^{-n\epsilon^2/8}$$

**(2) $\Rightarrow$ (3):** Uniform convergence implies ERM generalizes with sample complexity:
$$m(\epsilon, \delta) = O\left(\frac{\log \mathcal{N}(\mathcal{H}, \epsilon) + \log(1/\delta)}{\epsilon^2}\right)$$

**(3) $\Rightarrow$ (1):** Standard covering number theory: if $\mathcal{H}$ is PAC-learnable, it has polynomial covering numbers.

**Compactness Failure = No Uniform Convergence:**

When the compactness axiom fails:
$$\exists \epsilon_0.\ \mathcal{N}(\mathcal{H}, \epsilon_0) = \infty$$

This witnesses that no uniform convergence bound exists with finite samples.

---

### Step 4: The Resolution Mechanism

**Theorem (Runtime Compactness Resolution).** The dichotomy is resolved at training time, not assumed a priori.

**Algorithm (Sieve Node 3 for Generalization):**

```
Input: Hypothesis class H, sample S, target accuracy epsilon
Output: Concentration certificate OR Dispersion certificate

1. Compute empirical covering number:
   N_hat = estimate_covering_number(H, epsilon, S)

2. Compute Rademacher complexity:
   R_hat = estimate_rademacher(H, S)

3. Check concentration:
   if N_hat <= f(1/epsilon):
       C_epsilon = construct_cover(H, epsilon)
       return K_cover^+ (Concentration: finite cover)

4. Check dispersion:
   if R_hat <= c / sqrt(n):
       return K_dispersion^- (Dispersion: uniform convergence)

5. Hard core:
   return K_hardcore^+ (Hard core: no generalization guarantee)
```

**Runtime vs. A Priori:**

The key insight is that compactness is **checked**, not **assumed**:
- If concentration occurs (finite cover): generalization via covering bound
- If dispersion occurs (vanishing Rademacher): generalization via Rademacher bound
- Both branches lead to generalization

The third case (hard core with infinite covering number and non-vanishing Rademacher) indicates the problem genuinely requires infinite samples, but this is detected constructively.

---

## Connections to Classical Results

### 1. Covering Numbers (Metric Entropy)

**Statement (Kolmogorov-Tikhomirov 1959).** The $\epsilon$-covering number characterizes the metric entropy of a hypothesis class:
$$H(\mathcal{H}, \epsilon) = \log \mathcal{N}(\mathcal{H}, \epsilon)$$

**Connection to Compactness Resolution:**

| Lions' Concentration-Compactness | Covering Number Theory |
|----------------------------------|------------------------|
| Bounded energy sequence $u_n$ | Hypothesis class $\mathcal{H}$ |
| Concentration at points | Finite $\epsilon$-cover |
| Profile $V$ with $\Phi(V) > 0$ | Cover element with $h_c$ approximating $h$ |
| Vanishing (dispersion) | Vanishing Rademacher (uniform convergence) |
| Dichotomy: concentrate or vanish | Dichotomy: finite cover or disperse |

**Key Results:**
- **Linear Functions:** $\log \mathcal{N} = O(d \log(1/\epsilon))$ (dimension $d$)
- **Lipschitz Functions:** $\log \mathcal{N} = O((1/\epsilon)^d)$ (curse of dimensionality)
- **Neural Networks:** $\log \mathcal{N} = O(W \log(W/\epsilon))$ (parameter count $W$)

### 2. VC Dimension (Vapnik-Chervonenkis 1971)

**Statement:** For binary classification, sample complexity is:
$$m(\epsilon, \delta) = O\left(\frac{d_{\text{VC}} + \log(1/\delta)}{\epsilon^2}\right)$$

**Connection to Compactness Resolution:**

| VC Theory | Concentration-Compactness |
|-----------|---------------------------|
| VC dimension $d$ | Effective energy dimension |
| Shattering coefficient | Covering number growth rate |
| $d < \infty$ implies PAC-learnable | Finite energy implies compactness |
| Infinite VC dimension | Compactness failure |

**Covering Number Bound via VC:**
$$\mathcal{N}(\mathcal{H}, \epsilon, L^1(P_n)) \leq \left(\frac{2e}{\epsilon}\right)^{d_{\text{VC}}}$$

### 3. Rademacher Complexity (Bartlett-Mendelson 2002)

**Statement:** With probability at least $1 - \delta$:
$$L_\mathcal{D}(h) \leq L_S(h) + 2\mathcal{R}_n(\mathcal{H}) + \sqrt{\frac{\log(1/\delta)}{2n}}$$

**Connection to Compactness Resolution:**

| Rademacher Theory | Concentration-Compactness |
|-------------------|---------------------------|
| $\mathcal{R}_n(\mathcal{H})$ | Energy dispersion rate |
| $\mathcal{R}_n \to 0$ as $n \to \infty$ | Dispersion to zero |
| $\mathcal{R}_n = O(1/\sqrt{n})$ | Polynomial dispersion (Mode D.D) |
| Bounded Rademacher | Generalization certificate |

**Relationship to Covering Numbers:**
$$\mathcal{R}_n(\mathcal{H}) \leq \inf_{\alpha > 0} \left( 4\alpha + \frac{12}{\sqrt{n}} \int_\alpha^1 \sqrt{\log \mathcal{N}(\mathcal{H}, \epsilon)} \, d\epsilon \right)$$

This Dudley integral connects covering numbers (concentration) to Rademacher complexity (dispersion).

### 4. Fat-Shattering Dimension

**Statement (Alon et al. 1997).** For real-valued functions, the $\gamma$-fat-shattering dimension $\text{fat}_\gamma(\mathcal{H})$ characterizes learnability:
$$m(\epsilon, \delta) = O\left(\frac{\text{fat}_{\epsilon/8}(\mathcal{H}) + \log(1/\delta)}{\epsilon^2}\right)$$

**Connection to Compactness Resolution:**

| Fat-Shattering | Compactness Resolution |
|----------------|------------------------|
| $\text{fat}_\gamma < \infty$ | Concentration with margin $\gamma$ |
| Scale-sensitive dimension | Resolution at scale $\epsilon$ |
| Margin-based bounds | Profile extraction at multiple scales |

### 5. PAC-Bayes (McAllester 1999, Catoni 2007)

**Statement:** With prior $P$ and posterior $Q$ over hypotheses:
$$\mathbb{E}_{h \sim Q}[L_\mathcal{D}(h)] \leq \mathbb{E}_{h \sim Q}[L_S(h)] + \sqrt{\frac{\text{KL}(Q\|P) + \log(2\sqrt{n}/\delta)}{2n}}$$

**Connection to Compactness Resolution:**

| PAC-Bayes | Concentration-Compactness |
|-----------|---------------------------|
| Prior $P$ | Reference measure on profiles |
| Posterior $Q$ | Concentrated measure on learned hypothesis |
| KL divergence | Energy of profile relative to prior |
| Data-dependent bound | Runtime resolution |

---

## Implementation Notes

### Value Function Interpretation

**Height as Value Function:**
$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r(s_t, a_t) \mid s_0 = s\right]$$

**Covering Number for Value Function Classes:**
- **Tabular:** $\mathcal{N}(\mathcal{V}, \epsilon) = (R_{\max}/\epsilon)^{|\mathcal{S}|}$
- **Linear:** $\mathcal{N}(\mathcal{V}_{\text{linear}}, \epsilon) \leq (3W/\epsilon)^d$
- **Neural Networks:** $\mathcal{N}(\mathcal{V}_{\text{NN}}, \epsilon) \leq (W \cdot \text{Lip}/\epsilon)^{\text{params}}$

### Policy Interpretation

**Dissipation as Policy:**
$$\pi(a|s) \propto \exp\left(\frac{Q(s,a)}{\tau}\right)$$

**Covering the Policy Space:**

For softmax policies with temperature $\tau$:
$$\mathcal{N}(\Pi_\tau, \epsilon) \leq \left(\frac{|\mathcal{A}|}{\epsilon \tau}\right)^{|\mathcal{S}| \cdot |\mathcal{A}|}$$

Lower temperature concentrates the policy (smaller cover), higher temperature disperses (larger cover needed).

### Practical Generalization Diagnostics

```python
def check_compactness_resolution(model, train_data, val_data, epsilon):
    """
    Runtime compactness resolution check.

    Returns:
        'concentration': Finite cover exists, generalization via covering
        'dispersion': Rademacher vanishes, generalization via complexity
        'hard_core': No generalization guarantee
    """
    # Estimate covering number
    N_cover = estimate_covering_number(model.hypothesis_class, epsilon, train_data)

    # Estimate Rademacher complexity
    R_n = estimate_rademacher_complexity(model.hypothesis_class, train_data)

    # Compute generalization gap
    train_loss = compute_loss(model, train_data)
    val_loss = compute_loss(model, val_data)
    gen_gap = abs(val_loss - train_loss)

    # Concentration check
    if np.log(N_cover) <= polynomial_bound(1/epsilon):
        cover_bound = 2 * N_cover * np.exp(-len(train_data) * epsilon**2 / 8)
        return {
            'mode': 'concentration',
            'certificate': {
                'covering_number': N_cover,
                'bound': cover_bound,
                'epsilon': epsilon
            },
            'generalization': gen_gap
        }

    # Dispersion check
    n = len(train_data)
    if R_n <= c_constant / np.sqrt(n):
        rademacher_bound = 2 * R_n + np.sqrt(np.log(1/delta) / (2*n))
        return {
            'mode': 'dispersion',
            'certificate': {
                'rademacher': R_n,
                'bound': rademacher_bound,
                'sample_size': n
            },
            'generalization': gen_gap
        }

    # Hard core
    return {
        'mode': 'hard_core',
        'certificate': {
            'covering_number': N_cover,
            'rademacher': R_n,
            'warning': 'No generalization guarantee'
        },
        'generalization': gen_gap
    }
```

### Certificate Construction

**Concentration Certificate (Finite Cover):**
```python
K_concentration = {
    'mode': 'Concentration',
    'mechanism': 'Finite_Cover',
    'evidence': {
        'cover': C_epsilon,
        'cover_size': len(C_epsilon),
        'epsilon': epsilon,
        'covering_bound': f(1/epsilon),
        'generalization_bound': 2 * N * exp(-n * eps^2 / 8)
    },
    'tractability': 'PAC-learnable via ERM',
    'literature': 'Kolmogorov-Tikhomirov 1959, Dudley 1967'
}
```

**Dispersion Certificate (Rademacher Bound):**
```python
K_dispersion = {
    'mode': 'Dispersion',
    'mechanism': 'Rademacher_Convergence',
    'evidence': {
        'rademacher_complexity': R_n,
        'rate': O(1/sqrt(n)),
        'generalization_bound': 2*R_n + sqrt(log(1/delta)/(2n))
    },
    'tractability': 'Uniform convergence guaranteed',
    'note': 'NOT a failure - success state (Mode D.D)',
    'literature': 'Bartlett-Mendelson 2002, Koltchinskii 2001'
}
```

**Hard Core Certificate (No Guarantee):**
```python
K_hardcore = {
    'mode': 'Hard_Core',
    'mechanism': 'No_Uniform_Convergence',
    'evidence': {
        'covering_number': 'infinite or super-polynomial',
        'rademacher': 'non-vanishing',
        'vc_dimension': 'infinite',
        'lower_bound_witness': 'shattering set construction'
    },
    'tractability': 'Requires infinite samples or structural assumptions',
    'literature': 'Vapnik-Chervonenkis 1971, Blumer et al. 1989'
}
```

### Deep Learning Perspective

**Neural Network Covering Numbers:**

For ReLU networks with depth $L$, width $W$, and weight bound $B$:
$$\log \mathcal{N}(\mathcal{H}_{\text{NN}}, \epsilon) = O\left(L W^2 \log(L W B / \epsilon)\right)$$

**Implicit Regularization as Compactness:**

Gradient descent with early stopping implicitly constrains the hypothesis class:
- **Concentration:** Effective hypothesis class has small covering number
- **Resolution:** Early stopping decides when concentration is sufficient

**Overparameterization and Double Descent:**

| Regime | Compactness Status | Generalization |
|--------|-------------------|----------------|
| Underparameterized | Concentration (small cover) | Classical bound |
| Interpolation threshold | Hard core (critical) | Peak test error |
| Overparameterized | Dispersion (implicit regularization) | Benign overfitting |

### RL-Specific Considerations

**Value Function Generalization:**

For function approximation in RL, the compactness resolution applies to:
- $\mathcal{V} = \{V_\theta : \theta \in \Theta\}$: Value function class
- $\mathcal{Q} = \{Q_\theta : \theta \in \Theta\}$: Q-function class

**Bellman Error and Covering:**

The Bellman error can be bounded using covering numbers:
$$\|V_\theta - V^*\|_\infty \leq \text{approx}_{\epsilon}(\mathcal{V}) + O\left(\sqrt{\frac{\log \mathcal{N}(\mathcal{V}, \epsilon)}{n}}\right)$$

where $\text{approx}_\epsilon(\mathcal{V}) = \inf_{V \in \mathcal{V}} \|V - V^*\|$ is the approximation error.

---

## Literature

1. **Lions, P.-L. (1984).** "The Concentration-Compactness Principle in the Calculus of Variations. Part I: The Locally Compact Case." *Annales IHP Analyse Non Lineaire.* *Original concentration-compactness.*

2. **Lions, P.-L. (1985).** "The Concentration-Compactness Principle in the Calculus of Variations. Part II: The Limit Case." *Annales IHP Analyse Non Lineaire.* *Limit case analysis.*

3. **Vapnik, V.N. & Chervonenkis, A.Y. (1971).** "On the Uniform Convergence of Relative Frequencies of Events to Their Probabilities." *Theory of Probability and Its Applications.* *VC dimension foundations.*

4. **Kolmogorov, A.N. & Tikhomirov, V.M. (1959).** "$\epsilon$-entropy and $\epsilon$-capacity of sets in function spaces." *Uspekhi Matematicheskikh Nauk.* *Metric entropy and covering numbers.*

5. **Dudley, R.M. (1967).** "The Sizes of Compact Subsets of Hilbert Space and Continuity of Gaussian Processes." *Journal of Functional Analysis.* *Dudley entropy integral.*

6. **Bartlett, P.L. & Mendelson, S. (2002).** "Rademacher and Gaussian Complexities: Risk Bounds and Structural Results." *JMLR.* *Rademacher complexity framework.*

7. **Koltchinskii, V. & Panchenko, D. (2000).** "Rademacher Processes and Bounding the Risk of Function Learning." *High Dimensional Probability II.* *Local Rademacher complexity.*

8. **McAllester, D. (1999).** "PAC-Bayesian Model Averaging." *COLT.* *PAC-Bayes bounds.*

9. **Shalev-Shwartz, S. & Ben-David, S. (2014).** *Understanding Machine Learning: From Theory to Algorithms.* Cambridge. *Comprehensive learning theory.*

10. **Anthony, M. & Bartlett, P.L. (1999).** *Neural Network Learning: Theoretical Foundations.* Cambridge. *Covering numbers for neural networks.*

11. **Wainwright, M.J. (2019).** *High-Dimensional Statistics: A Non-Asymptotic Viewpoint.* Cambridge. *Modern concentration inequalities.*

12. **Bahouri, H. & Gerard, P. (1999).** "High Frequency Approximation of Solutions to Critical Nonlinear Wave Equations." *American Journal of Mathematics.* *Profile decomposition (mathematical analogue).*

13. **Bartlett, P.L., Foster, D.J., & Telgarsky, M. (2017).** "Spectrally-Normalized Margin Bounds for Neural Networks." *NeurIPS.* *Norm-based generalization bounds.*

14. **Neyshabur, B., Bhojanapalli, S., McAllester, D., & Srebro, N. (2017).** "Exploring Generalization in Deep Learning." *NeurIPS.* *Implicit regularization and generalization.*

15. **Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019).** "Reconciling Modern Machine Learning Practice and the Classical Bias-Variance Trade-off." *PNAS.* *Double descent and benign overfitting.*
