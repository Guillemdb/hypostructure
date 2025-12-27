# KRNL-Trichotomy: Learning Complexity Classification

## AI/RL/ML Statement

### Original Statement (Hypostructure)
*Reference: mt-krnl-trichotomy*

Every trajectory with finite breakdown time classifies into exactly one of three outcomes: Global Existence (dispersion), Global Regularity (concentration with permits satisfied), or Genuine Singularity (permits violated).

---

## AI/RL/ML Formulation

### Setup

Consider a learning problem where:

- **State space:** Environment states $\mathcal{S}$ or hypothesis class $\mathcal{H}$
- **Height/Energy:** Value function $V(s)$ or generalization error $\mathcal{E}(\theta)$
- **Dissipation:** Policy $\pi(a|s)$ or learning dynamics $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$
- **Trajectory:** Learning curve $\{(\theta_t, L(\theta_t))\}_{t=0}^T$ or policy iteration sequence $\{\pi_k\}$
- **Breakdown time:** Sample complexity bound $T_*(n, \epsilon, \delta)$

The "trichotomy" classifies learning problems by their intrinsic computational and statistical complexity.

### Statement (AI/RL/ML Version)

**Theorem (Learning Complexity Trichotomy).** Let $\mathcal{P} = (\mathcal{H}, \mathcal{D}, L)$ be a learning problem with hypothesis class $\mathcal{H}$, data distribution $\mathcal{D}$, and loss $L$. Under sample complexity and computational constraints, exactly one of three outcomes occurs:

| **Outcome** | **Learning Mode** | **Mechanism** |
|-------------|-------------------|---------------|
| **PAC-Learnable** | Dispersion (D.D) | Polynomial samples/compute suffice, generalization is uniform |
| **Structure-Dependent** | Concentration (Regularity) | Exponential worst-case but tractable with structure (sparse, low-rank, etc.) |
| **Computationally Hard** | Singularity (C.E) | Inherently hard: cryptographic, NP-hard, or statistically impossible |

**Formal Statement:** For learning problem $\mathcal{P}$ with VC dimension $d$, sample complexity $m(\epsilon, \delta)$, and computational complexity $T(n)$:

$$\text{Trichotomy}(\mathcal{P}) = \begin{cases}
\text{PAC (Easy)} & \text{if } m = O\left(\frac{d + \log(1/\delta)}{\epsilon^2}\right), \, T = \text{poly}(n) \\
\text{Structured (Intermediate)} & \text{if } m = \text{poly}(n, 1/\epsilon), \, T = f(\text{structure}) \cdot \text{poly}(n) \\
\text{Hard (Singularity)} & \text{if } m = \Omega(2^n) \text{ or } T = \Omega(2^n)
\end{cases}$$

---

## Terminology Translation Table

| Hypostructure Term | AI/RL/ML Equivalent |
|--------------------|---------------------|
| Trajectory $u(t)$ | Learning trajectory $\{(\theta_t, L(\theta_t))\}$ or policy sequence $\{\pi_k\}$ |
| Breakdown time $T_*$ | Sample complexity bound $m(\epsilon, \delta)$ or convergence time |
| Energy functional $\Phi$ | Value function $V(s)$, loss $L(\theta)$, or regret $R_T$ |
| Energy dispersion $\Phi \to 0$ | Generalization: $L_{\text{test}} \to L_{\text{train}}$, uniform convergence |
| Energy concentration $\Phi_* > 0$ | Representation bottleneck, feature concentration |
| Genuine singularity | Computational hardness (NP, crypto), statistical impossibility |
| Profile extraction | Feature learning, representation extraction |
| Limiting profile $v^*$ | Learned representation $\phi(x)$, sufficient statistic |
| Symmetry group $G$ | Equivariance group (rotations, permutations, gauge) |
| Interface permits | Structural assumptions (sparsity, smoothness, realizability) |
| Mode D.D (Dispersion-Decay) | PAC-learnable, polynomial sample complexity |
| Mode C.E (Concentration-Escape) | Computationally hard, no efficient algorithm |
| Global Regularity | Structure-exploiting algorithms (FPT, kernel methods) |
| Dissipation $D$ | Policy entropy, gradient descent dynamics |
| Semiflow $S_t$ | Learning algorithm dynamics, SGD flow |

---

## Proof Sketch

### Step 1: Sample Complexity Dichotomy (Energy Dichotomy)

**Claim:** For any learning problem $\mathcal{P}$, the sample complexity either:
- **Disperses:** $m(\epsilon, \delta) = O(\text{poly}(d, 1/\epsilon, \log(1/\delta)))$ (PAC-learnable)
- **Concentrates:** $m(\epsilon, \delta) = \omega(\text{poly}(n))$ (super-polynomial)

**Proof (Fundamental Theorem of Statistical Learning):**

By the Vapnik-Chervonenkis theorem [Vapnik-Chervonenkis 1971], for hypothesis class $\mathcal{H}$ with VC dimension $d < \infty$:

$$m(\epsilon, \delta) = O\left(\frac{d \log(1/\epsilon) + \log(1/\delta)}{\epsilon^2}\right)$$

**Case 1 (Dispersion):** If $d = O(\text{poly}(n))$, sample complexity is polynomial. The learning trajectory "disperses" uniformly across the hypothesis space, and generalization is achieved. This corresponds to Mode D.D.

$$\Pr_{S \sim \mathcal{D}^m}\left[\sup_{h \in \mathcal{H}} |L_S(h) - L_\mathcal{D}(h)| > \epsilon\right] \leq \delta$$

**Case 2 (Concentration):** If $d = \infty$ or $d = \Omega(2^n)$, sample complexity explodes. "Energy" (generalization error) concentrates in specific regions of hypothesis space. Proceed to Step 2.

**Connection to Lions' Dichotomy:** This mirrors concentration-compactness. In learning:
- **Vanishing (Dispersion):** Errors spread uniformly, no hypothesis dominates
- **Concentration:** Errors localize in specific hypothesis regions, requiring structure

---

### Step 2: Profile Extraction = Representation Learning

**Assumption:** We are in Case 2 (concentration), with $d = \infty$ or super-polynomial.

**Claim (Representation Extraction):** For structured learning problems, there exists a decomposition:

$$h(x) = g(\phi(x))$$

where:
- $\phi: \mathcal{X} \to \mathcal{Z}$ is a **learned representation** (feature map)
- $g: \mathcal{Z} \to \mathcal{Y}$ is a **simple predictor** (linear, shallow)
- $\dim(\mathcal{Z}) \ll \dim(\mathcal{X})$ captures the "hard kernel"

**Proof (Feature Learning as Profile Decomposition):**

**Step 2.1 (Representation Learning):**

For deep networks, the learned representation $\phi_\theta(x)$ extracts invariant features [Bengio et al. 2013]:

$$\phi_\theta(x) = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$$

This is analogous to profile extraction: the network finds the "limiting profile" that captures essential structure.

**Step 2.2 (Invariance under Symmetries):**

For equivariant architectures (CNNs, GNNs), the representation is invariant under group action:

$$\phi(g \cdot x) = \rho(g) \cdot \phi(x) \text{ for } g \in G$$

This mirrors profile extraction modulo symmetry group $G$ in Bahouri-Gerard.

**Step 2.3 (Sufficient Statistics):**

In the limit, $\phi(x)$ becomes a **sufficient statistic** for prediction:

$$I(Y; \phi(X)) = I(Y; X)$$

The representation captures all predictive information (data processing inequality equality).

---

### Step 3: Permit Classification = Structural Assumptions

We classify learning problems by which structural "permits" they satisfy.

**Interface Permits for Learning Problems:**

| Permit | Learning Condition | Interpretation |
|--------|-------------------|----------------|
| $\mathrm{SC}_\lambda$ (Subcriticality) | $d_{\text{VC}} = O(\text{poly}(n))$ | Finite capacity |
| $\mathrm{SC}_{\partial c}$ (Smoothness) | $L$ is $\beta$-smooth | Gradient methods work |
| $\mathrm{Cap}_H$ (Low-rank) | $\text{rank}(W^*) \leq r$ | Matrix completion tractable |
| $\mathrm{LS}_\sigma$ (Convexity) | $L$ is convex or satisfies PL | Global optima reachable |
| $\mathrm{TB}_\pi$ (Sparsity) | $\|\theta^*\|_0 \leq k$ | Compressed sensing works |
| $\mathrm{Cov}$ (Realizability) | $h^* \in \mathcal{H}$ | Target in hypothesis class |

**Trichotomy Split:**

---

#### Case 3.1: All Permits Satisfied - PAC-Learnable (Mode D.D)

**Assumption:** Learning problem $\mathcal{P}$ satisfies:
- Finite VC dimension: $d < \infty$
- Realizability: $h^* \in \mathcal{H}$
- Efficient optimization: loss is convex or satisfies PL condition

**Theorem (PAC Learning):**

If $\mathcal{H}$ has VC dimension $d$ and is efficiently searchable, then $\mathcal{P}$ is PAC-learnable with:

$$m(\epsilon, \delta) = O\left(\frac{d + \log(1/\delta)}{\epsilon^2}\right), \quad T = \text{poly}(n, d, 1/\epsilon)$$

**Proof (Valiant 1984, Kearns-Vazirani 1994):**

1. **Sample Complexity:** By VC theory, $m = O(d/\epsilon^2)$ samples suffice for uniform convergence.

2. **Computational Efficiency:** For convex losses, gradient descent finds $\epsilon$-optimal solution in $T = O(1/\epsilon^2)$ iterations.

3. **Generalization:** Rademacher complexity bound ensures test error matches train error.

**Certificate Produced:** $(\mathcal{P} \in \text{PAC}, d_{\text{VC}}, m(\epsilon, \delta), \mathcal{A}_{\text{learn}})$

---

#### Case 3.2: Some Permits Satisfied - Structure-Dependent (Regularity)

**Assumption:** Learning problem has:
- Infinite VC dimension, but structural constraints
- Sparsity: $\|\theta^*\|_0 \leq k$
- Low-rank: $\text{rank}(W^*) \leq r$
- Manifold structure: data on low-dimensional manifold

**Theorem (Structured Learning):**

With structural constraints, sample complexity becomes:

$$m(\epsilon, \delta) = O\left(\frac{k \log(n/k) + \log(1/\delta)}{\epsilon^2}\right)$$

(sparse recovery) or similar structure-dependent bounds.

**Examples:**

1. **LASSO/Compressed Sensing [Candes-Tao 2005]:**
   - Permits: Sparsity ($\|\theta^*\|_0 \leq k$), RIP condition
   - Complexity: $m = O(k \log n)$, $T = \text{poly}(n)$
   - Mode: Concentration with barriers (sparsity regularization)

2. **Matrix Completion [Candes-Recht 2009]:**
   - Permits: Low-rank ($\text{rank}(W^*) \leq r$), incoherence
   - Complexity: $m = O(nr \log n)$
   - Mode: Structure-exploiting algorithms

3. **Kernel Methods with RKHS [Scholkopf-Smola 2002]:**
   - Permits: Smoothness in RKHS, bounded norm
   - Complexity: $m = O(1/\epsilon^2)$ but $T = O(n^3)$
   - Mode: Statistical efficiency with computational cost

**Rigidity Interpretation:** Structural permits form "barriers" preventing the problem from being computationally hard. Like Kenig-Merle rigidity, structure forces tractability.

**Certificate Produced:** $(P \in \text{Structured}, \text{constraint}, m, T, \mathcal{A})$

---

#### Case 3.3: Permits Violated - Computationally Hard (Mode C.E)

**Assumption:** Learning problem violates essential permits:
- No structural constraints
- Non-convex loss landscape
- Cryptographic or NP-based construction

**Theorem (Computational Hardness):**

Learning problems can be computationally intractable even with polynomial samples:

1. **Proper Learning of DNF [Kearns-Valiant 1994]:**
   Properly learning DNF formulas is as hard as breaking RSA.

2. **Learning Sparse Parities with Noise [Blum-Kalai-Wasserman 2003]:**
   No polynomial-time algorithm unless $\text{P} = \text{NP}$.

3. **Tensor Decomposition [Hillar-Lim 2013]:**
   Computing tensor rank is NP-hard.

**Proof (Cryptographic Reduction):**

**Learning Parity with Noise (LPN):**

Given samples $(a_i, b_i = \langle a_i, s \rangle + e_i \mod 2)$ where $s$ is secret and $e_i$ is noise:
- Information-theoretically: $O(n/\epsilon)$ samples suffice
- Computationally: best known algorithms require $2^{O(n/\log n)}$ time

The "energy" (secret $s$) is concentrated in the noise-corrupted observations but computationally inaccessible.

**Hardness Catalog:**

| Violated Permit | Consequence | Hardness Source |
|-----------------|-------------|-----------------|
| $K_{\text{SC}_\lambda}^-$ (infinite VC) | Exponential samples | Statistical impossibility |
| $K_{\text{LS}_\sigma}^-$ (non-convex) | Local minima traps | Optimization hardness |
| $K_{\text{Crypto}}^-$ | Cryptographic hardness | Reduction to crypto primitives |
| $K_{\text{NP}}^-$ | NP-hard subproblem | Reduction to NP-complete |

**Certificate Produced:** $(P \in \text{Hard}, \text{reduction}, \text{violated permits})$

---

### Step 4: Rigidity = No Intermediate Regime for Well-Structured Problems

**Theorem (Learning Dichotomy for Specific Classes):**

For certain learning problem classes, there is no intermediate regime:

1. **Boolean Functions (Valiant 1984):** A concept class is either PAC-learnable (polynomial) or not learnable at all.

2. **CSP Learning (Feldman et al. 2015):** Learning constraint satisfaction problems exhibits dichotomy based on algebraic structure.

3. **Statistical Query Model (Kearns 1998):** SQ dimension determines learnability - either polynomial or exponential.

**Rigidity Mechanism:**

The algebraic/combinatorial structure of the hypothesis class determines tractability:
- **Tractable:** Linear threshold functions, decision lists, bounded-depth circuits
- **Hard:** DNF, parity functions, cryptographic primitives

No "almost learnable" problems exist in well-structured settings.

---

## Connections to Classical Results

### 1. Valiant's PAC Learning Framework (1984)

**Statement:** A concept class $\mathcal{C}$ is PAC-learnable if there exists an algorithm $\mathcal{A}$ such that for all distributions $\mathcal{D}$, all $\epsilon, \delta > 0$:

$$\Pr_{S \sim \mathcal{D}^m}[L_\mathcal{D}(\mathcal{A}(S)) \leq \epsilon] \geq 1 - \delta$$

with $m = \text{poly}(1/\epsilon, 1/\delta, n, \text{size}(c))$ and runtime $\text{poly}(m)$.

**Connection to Trichotomy:**
- **Mode D.D:** PAC-learnable concepts - errors disperse uniformly
- **Mode Regularity:** Learnable with membership queries or structure
- **Mode C.E:** Not PAC-learnable (DNF, parity with noise)

### 2. Kearns-Vazirani Computational Learning Theory (1994)

**Key Results:**
- **Cryptographic hardness of learning:** If one-way functions exist, some concept classes are not efficiently learnable
- **Representation dependence:** Learnability depends on representation (DNF vs CNF)
- **Query complexity:** Additional queries (membership, equivalence) can make hard problems tractable

**Certificate Correspondence:**
- Efficient learners $\leftrightarrow$ Dispersion certificates
- Representation barriers $\leftrightarrow$ Interface permit violations
- Query augmentation $\leftrightarrow$ Surgery operations

### 3. Statistical Learning Theory (Vapnik-Chervonenkis)

**Fundamental Theorem:** A hypothesis class $\mathcal{H}$ is learnable if and only if $d_{\text{VC}}(\mathcal{H}) < \infty$.

**Quantitative Bound:**
$$m(\epsilon, \delta) = \Theta\left(\frac{d + \log(1/\delta)}{\epsilon^2}\right)$$

**Connection:**
- Finite VC dimension $\leftrightarrow$ Subcriticality permit $\mathrm{SC}_\lambda$
- Uniform convergence $\leftrightarrow$ Energy dispersion
- Shattering $\leftrightarrow$ Concentration/singularity

### 4. Sample Complexity Hierarchy

**Hierarchy of Learning Complexities:**

$$\text{Realizable} \subset \text{Agnostic} \subset \text{Adversarial} \subset \text{Online}$$

| Regime | Sample Complexity | Mode |
|--------|-------------------|------|
| Realizable PAC | $O(d/\epsilon)$ | D.D (Dispersion) |
| Agnostic PAC | $O(d/\epsilon^2)$ | D.D (Dispersion) |
| Online Learning | $O(\sqrt{T \cdot d})$ regret | Regularity (structure-dependent) |
| Adversarial | $\Omega(n)$ in worst case | C.E (Singularity) |

### 5. Computational-Statistical Gaps

**Phenomenon:** Some problems are statistically easy but computationally hard.

**Examples:**
- **Planted Clique:** $O(\log n)$ samples suffice, but best algorithm needs $n^{1/2}$ time
- **Sparse PCA:** $O(k \log n)$ samples, but SDP relaxation needed
- **Tensor PCA:** Information-theoretically easy, computationally hard below SNR threshold

**Connection to Trichotomy:**
- Statistical tractability = Subcritical energy
- Computational hardness = Concentration without permits
- Gap = Difference between statistical and computational singularity

---

## Implementation Notes

### RL/Policy Optimization Perspective

**Value Function as Height:**
$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r(s_t, a_t) \mid s_0 = s\right]$$

**Trichotomy in RL:**
1. **D.D (Tractable MDPs):** Tabular MDPs with polynomial state/action spaces. Value iteration converges in $O(|\mathcal{S}||\mathcal{A}|/(1-\gamma)^3)$.

2. **Regularity (Structured MDPs):** Linear MDPs, factored MDPs, low-rank MDPs. Sample complexity depends on intrinsic dimension, not ambient dimension.

3. **C.E (Hard MDPs):** Partially observable MDPs (POMDPs), adversarial MDPs. PSPACE-complete in general.

### Deep Learning Perspective

**Loss Landscape Analysis:**
1. **D.D (Easy optimization):** Overparameterized networks in NTK regime. Gradient descent finds global minimum.

2. **Regularity (Structured landscape):** Networks with residual connections, normalization. Local minima are global or near-global.

3. **C.E (Hard optimization):** Deep networks with bad initialization. Spurious local minima, saddle points, vanishing gradients.

### Practical Classification Heuristics

```python
def classify_learning_problem(P):
    """
    Classify learning problem P into trichotomy.

    Returns:
        'PAC': Polynomial sample/compute complexity
        'Structured': Exponential worst-case, tractable with structure
        'Hard': Inherently hard
    """
    # Check VC dimension / capacity
    if vc_dimension(P.hypothesis_class) < poly(P.n):
        if is_convex(P.loss) or satisfies_PL(P.loss):
            return 'PAC'  # Mode D.D

    # Check structural constraints
    if has_sparsity(P) or has_low_rank(P) or has_manifold(P):
        if satisfies_RIP(P) or satisfies_incoherence(P):
            return 'Structured'  # Regularity

    # Check hardness reductions
    if reduces_to_crypto(P) or reduces_to_NP(P):
        return 'Hard'  # Mode C.E

    # Default: structure-dependent
    return 'Structured'
```

### Certificate Verification

**PAC Certificate:**
```python
K_PAC = {
    'mode': 'Dispersion',
    'mechanism': 'Uniform_Convergence',
    'evidence': {
        'vc_dimension': d,
        'sample_complexity': O(d/eps^2),
        'algorithm': 'ERM',
        'runtime': poly(n, d, 1/eps)
    },
    'literature': 'Valiant 1984, Vapnik 1998'
}
```

**Structured Certificate:**
```python
K_Structured = {
    'mode': 'Regularity',
    'mechanism': 'Structure_Exploitation',
    'evidence': {
        'structure': 'sparse/low-rank/manifold',
        'effective_dimension': k,
        'sample_complexity': O(k * log(n)),
        'algorithm': 'LASSO/NuclearNorm/ManifoldLearning',
        'permit_certificates': {
            'sparsity': 'k-sparse',
            'RIP': 'delta_2k < sqrt(2)-1',
            'incoherence': 'mu < O(1)'
        }
    },
    'literature': 'Candes-Tao 2005, Candes-Recht 2009'
}
```

**Hardness Certificate:**
```python
K_Hard = {
    'mode': 'Singularity',
    'mechanism': 'Computational_Hardness',
    'evidence': {
        'reduction': 'LPN/RSA/NP-complete',
        'violated_permits': {
            'convexity': 'non-convex',
            'VC': 'infinite or exponential',
            'crypto': 'one-way function assumption'
        },
        'lower_bound': '2^{Omega(n)}'
    },
    'literature': 'Kearns-Valiant 1994, Blum-Kalai-Wasserman 2003'
}
```

---

## Literature

1. **Valiant, L. G. (1984).** "A Theory of the Learnable." *Communications of the ACM.* *Foundational PAC learning framework.*

2. **Kearns, M. J. & Vazirani, U. V. (1994).** *An Introduction to Computational Learning Theory.* MIT Press. *Computational aspects of learning, cryptographic hardness.*

3. **Vapnik, V. N. & Chervonenkis, A. Y. (1971).** "On the Uniform Convergence of Relative Frequencies of Events to Their Probabilities." *Theory of Probability and Its Applications.* *VC dimension, uniform convergence.*

4. **Vapnik, V. N. (1998).** *Statistical Learning Theory.* Wiley. *Comprehensive treatment of statistical learning.*

5. **Blum, A., Kalai, A., & Wasserman, H. (2003).** "Noise-Tolerant Learning, the Parity Problem, and the Statistical Query Model." *JACM.* *Hardness of learning parity with noise.*

6. **Candes, E. J. & Tao, T. (2005).** "Decoding by Linear Programming." *IEEE Trans. Info. Theory.* *Compressed sensing, RIP.*

7. **Candes, E. J. & Recht, B. (2009).** "Exact Matrix Completion via Convex Optimization." *Foundations of Computational Mathematics.* *Low-rank matrix recovery.*

8. **Shalev-Shwartz, S. & Ben-David, S. (2014).** *Understanding Machine Learning: From Theory to Algorithms.* Cambridge. *Modern treatment of learning theory.*

9. **Arora, S. & Barak, B. (2009).** *Computational Complexity: A Modern Approach.* Cambridge. *Complexity theory foundations.*

10. **Feldman, V. (2017).** "A General Characterization of the Statistical Query Complexity." *COLT.* *SQ dimension and learning complexity.*

11. **Lions, P.-L. (1984).** "The Concentration-Compactness Principle in the Calculus of Variations." *Annales IHP.* *Original concentration-compactness (mathematical analogue).*

12. **Bengio, Y., Courville, A., & Vincent, P. (2013).** "Representation Learning: A Review and New Perspectives." *IEEE TPAMI.* *Deep learning as feature extraction.*
