---
title: "KRNL-Shadowing - Complexity Theory Translation"
---

# KRNL-Shadowing: Shadowing Metatheorem

## Overview

This document provides a complete complexity-theoretic translation of the KRNL-Shadowing theorem (Shadowing Metatheorem) from the hypostructure framework. The translation establishes a formal correspondence between dynamical shadowing in hyperbolic systems and the lifting of approximate solutions to exact solutions in computational problems, revealing deep connections to self-correction in proof systems and error-resilient computation.

**Original Theorem Reference:** {prf:ref}`mt-krnl-shadowing`

---

## Complexity Theory Statement

**Theorem (KRNL-Shadowing, Computational Form).**
Let $\mathcal{M} = (Q, \delta, \mathrm{Cost}, \mathrm{Spec})$ be a computational transition system with:
- Configuration space $Q$ (problem instances and partial solutions)
- Transition function $\delta: Q \to Q$ (iterative refinement operator)
- Cost function $\mathrm{Cost}: Q \to \mathbb{R}_{\geq 0}$ (solution quality measure)
- Spectral data $\mathrm{Spec}: Q \to \mathbb{R}_{>0}$ (robustness parameter at each configuration)

Suppose $\mathcal{M}$ satisfies the **robustness condition**: the linearization of $\delta$ at each configuration $q$ has spectral gap $\lambda_q = \mathrm{Spec}(q) > 0$, meaning there exists an exponential dichotomy with expansion rate $\mu \geq c\lambda_q$ in some directions and contraction rate $\mu$ in complementary directions.

**Statement (Approximate-to-Exact Lifting):**
For any $\varepsilon$-approximate solution sequence $\{y_n\}_{n=0}^N$ satisfying:
$$\|\delta(y_n) - y_{n+1}\| < \varepsilon \quad \text{for all } n$$

there exists a $\delta(\varepsilon)$-close exact solution sequence $\{x_n\}_{n=0}^N$ satisfying:
$$x_{n+1} = \delta(x_n) \quad \text{(exact iteration)}$$
$$\|x_n - y_n\| < \delta(\varepsilon) \quad \text{for all } n$$

The approximation bound satisfies $\delta(\varepsilon) = O(\varepsilon/\lambda)$ where $\lambda = \min_n \lambda_{y_n}$ is the minimum spectral gap along the approximate trajectory.

**Corollary (Self-Correction Principle).**
If a heuristic algorithm produces an approximate solution with uniformly positive robustness parameter, the approximate solution can be corrected to an exact solution via fixed-point iteration, with correction distance bounded by the approximation error divided by the robustness parameter.

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Pseudo-orbit $\{y_n\}$ | Approximate computation / heuristic solution | Sequence where each step has bounded error |
| True orbit $\{x_n\}$ | Valid computation / exact solution | Sequence satisfying exact transition relation |
| Shadowing distance $\delta$ | Error between approximate and exact | $\|x_n - y_n\| < \delta$ uniformly |
| Pseudo-orbit error $\varepsilon$ | Single-step approximation error | $\|\delta(y_n) - y_{n+1}\| < \varepsilon$ |
| Spectral gap $\lambda$ | Robustness parameter | Separation of linearization spectrum from zero |
| Exponential dichotomy | Expansion/contraction structure | Some directions amplify errors, others suppress |
| Stable bundle $E^s$ | Error-correcting directions | Components where iteration reduces error |
| Unstable bundle $E^u$ | Error-amplifying directions | Components where backward iteration reduces error |
| Dichotomy constant $C$ | Conditioning number | Bounds on projection operators |
| Dichotomy exponent $\mu$ | Convergence rate | Rate of error decay in stable/unstable directions |
| Graph transform | Fixed-point operator | Maps deviation sequences to refined sequences |
| Green's function | Error propagator | Resolves forward/backward error contributions |
| Contraction mapping | Iterative correction | Banach fixed-point iteration |
| Linearization $L_n = D\delta|_{y_n}$ | Local sensitivity matrix | Jacobian of transition at approximate solution |
| Nonlinear remainder $R_n$ | Higher-order corrections | $O(\|\xi\|^2)$ terms in Taylor expansion |
| Banach space $\ell^\infty$ | Sequence space | Bounded sequences with supremum norm |
| Certificate $K_{\text{true}}^{\delta}$ | Correctness proof | Witness that exact solution exists nearby |

---

## Logical Framework

### Approximate Computation Model

**Definition (Approximate Transition System).**
An approximate transition system is a tuple $\mathcal{M} = (Q, \delta, \|\cdot\|)$ where:
- $Q$ is a normed configuration space
- $\delta: Q \to Q$ is a (possibly implicit) exact transition
- $\|\cdot\|$ is a norm measuring distance in configuration space

**Definition ($\varepsilon$-Approximate Computation).**
A sequence $\{y_n\}_{n=0}^N$ is an $\varepsilon$-approximate computation if:
$$\|\delta(y_n) - y_{n+1}\| < \varepsilon \quad \text{for all } n \in \{0, \ldots, N-1\}$$

This captures computations with bounded per-step error, such as:
- Numerical integration with discretization error
- Heuristic algorithms with approximation guarantees
- Noisy computation with bounded noise per step

**Definition (Robustness Parameter).**
The robustness parameter $\lambda_q > 0$ at configuration $q$ measures how "stable" the local computation is. Formally, it is the spectral gap of the linearization:
$$\lambda_q = \mathrm{dist}(\sigma(D\delta|_q), \{z \in \mathbb{C} : |\mathrm{Re}(z)| = 0\})$$

where $\sigma(D\delta|_q)$ is the spectrum of the Jacobian of $\delta$ at $q$.

### Connection to Proof Systems

The shadowing theorem has a natural interpretation in proof complexity:

| Proof Systems | Shadowing |
|---------------|-----------|
| Approximate proof | Pseudo-orbit |
| Valid proof | True orbit |
| Proof verification | Checking $\delta(x_n) = x_{n+1}$ |
| Self-correction | Fixed-point iteration to repair |
| Soundness gap | Spectral gap $\lambda$ |

**Self-Correctable Proof Systems.** A proof system is self-correctable if approximate proofs can be efficiently corrected to valid proofs. The shadowing theorem provides the dynamical-systems foundation: self-correction is possible precisely when the proof verification dynamics has positive spectral gap.

---

## Proof Sketch

### Setup: Deviation Sequences and Linearization

**Problem Formulation.** Given:
- Approximate computation $\{y_n\}$ with per-step error $\varepsilon$
- Robustness parameter $\lambda > 0$ uniformly along trajectory

**Goal:** Construct exact computation $\{x_n\}$ with $x_{n+1} = \delta(x_n)$ and $\|x_n - y_n\| < \delta(\varepsilon)$.

**Deviation Sequence.** Define $\xi_n := x_n - y_n$. The exact iteration condition becomes:
$$\xi_{n+1} = \delta(y_n + \xi_n) - y_{n+1}$$

Expanding via Taylor series:
$$\xi_{n+1} = D\delta|_{y_n}(\xi_n) + \underbrace{[\delta(y_n) - y_{n+1}]}_{=: e_n} + R_n(\xi_n)$$

where:
- $L_n := D\delta|_{y_n}$ is the linearization (local sensitivity matrix)
- $e_n$ is the pseudo-orbit error with $\|e_n\| < \varepsilon$
- $R_n(\xi_n) = O(\|\xi_n\|^2)$ is the nonlinear remainder

This gives the **shadowing equation**:
$$\xi_{n+1} - L_n \xi_n = e_n + R_n(\xi_n)$$

### Step 1: Exponential Dichotomy (Robust Error Structure)

**Key Insight.** The spectral gap $\lambda > 0$ implies the linearization has exponential dichotomy: a splitting of the tangent space into stable and unstable directions.

**Definition (Exponential Dichotomy).**
The linearization $\{L_n\}$ has exponential dichotomy if there exist:
- Splitting: $\mathbb{R}^d = E^s_n \oplus E^u_n$ at each $n$
- Constants: $C \geq 1$ (conditioning) and $\mu > 0$ (rate)

Such that:
- **Stable directions:** $\|L_{n+k-1} \cdots L_n|_{E^s_n}\| \leq Ce^{-\mu k}$ for $k \geq 0$
- **Unstable directions:** $\|(L_{n+k-1} \cdots L_n)^{-1}|_{E^u_{n+k}}\| \leq Ce^{-\mu k}$ for $k \geq 0$

**Interpretation:**
- **Stable bundle $E^s$:** Errors in these directions decay under forward iteration (self-correcting)
- **Unstable bundle $E^u$:** Errors in these directions decay under backward iteration (require backward correction)

**Gap-Rate Relationship.** The dichotomy exponent $\mu$ satisfies $\mu \leq \lambda$. For normal operators, $\mu = \lambda$. For non-normal operators, $\mu$ can be smaller due to transient growth (pseudospectral effects). A sufficient condition for $\mu \geq c\lambda$ is that the operators $L_n$ are uniformly close to normal.

### Step 2: Fixed-Point Formulation (Correction Operator)

**Strategy.** Reformulate the shadowing problem as finding a fixed point of an operator on the space of deviation sequences.

**Sequence Space.** Work in the Banach space:
$$\ell^\infty = \left\{\{\xi_n\}_{n=0}^N : \|\{\xi_n\}\|_\infty := \sup_n \|\xi_n\| < \infty\right\}$$

**Green's Function Solution.** The linear equation $\xi_{n+1} - L_n\xi_n = e_n$ with bounded solutions is solved by:
$$\xi_n = \sum_{k=0}^{n-1} G_n^{n-k}(e_k) - \sum_{k=n}^{N} \tilde{G}_n^{k-n}(e_k)$$

where:
- $G_n^k = L_{n-1} \cdots L_{n-k} P^s_{n-k}$ propagates stable components forward
- $\tilde{G}_n^k = (L_n \cdots L_{n+k-1})^{-1} P^u_{n+k}$ propagates unstable components backward
- $P^s_n, P^u_n$ are projections onto stable/unstable bundles

**Exponential Decay.** By the dichotomy:
$$\|G_n^k\| \leq Ce^{-\mu k}, \quad \|\tilde{G}_n^k\| \leq Ce^{-\mu k}$$

**Fixed-Point Operator.** Define $\mathcal{T}: \ell^\infty \to \ell^\infty$ by:
$$(\mathcal{T}\xi)_n := \sum_{k=0}^{n-1} G_n^{n-k}(e_k + R_k(\xi_k)) - \sum_{k=n}^{N} \tilde{G}_n^{k-n}(e_k + R_k(\xi_k))$$

A fixed point $\xi = \mathcal{T}\xi$ solves the full shadowing equation.

### Step 3: Contraction Argument (Iterative Correction Converges)

**Goal.** Show $\mathcal{T}$ is a contraction on a ball of radius $\rho = 2C\varepsilon/\lambda$.

**Self-Mapping.** For $\xi \in B_\rho$:
$$\|(\mathcal{T}\xi)_n\| \leq \sum_{k} Ce^{-\mu|n-k|}(\varepsilon + K\|\xi_k\|^2) \leq \frac{C}{\mu}(\varepsilon + K\rho^2)$$

Using $\mu \geq c\lambda$ and choosing $\rho = 2C\varepsilon/\lambda$:
$$K\rho^2 = K \cdot \frac{4C^2\varepsilon^2}{\lambda^2} \leq \varepsilon \quad \text{when } \varepsilon \leq \frac{\lambda^2}{4C^2K}$$

This gives $\mathcal{T}: B_\rho \to B_\rho$.

**Contraction.** For $\xi, \eta \in B_\rho$:
$$\|(\mathcal{T}\xi) - (\mathcal{T}\eta)\|_\infty \leq \frac{C}{\lambda} \cdot 2K\rho \cdot \|\xi - \eta\|_\infty = \frac{4C^2K\varepsilon}{\lambda^2}\|\xi - \eta\|_\infty$$

**Smallness Condition.** The contraction factor is $\theta = 4C^2K\varepsilon/\lambda^2$. For contraction:
$$\varepsilon < \varepsilon_0 := \frac{\lambda^2}{4C^2K}$$

**Banach Fixed-Point Theorem.** For $\varepsilon < \varepsilon_0$, the operator $\mathcal{T}$ has a unique fixed point $\xi^* \in B_\rho$ with:
$$\|\xi^*\|_\infty \leq \frac{2C\varepsilon}{\lambda}$$

### Step 4: Complexity Application (Computational Interpretation)

**Exact Solution Construction.** Define:
$$x_n := y_n + \xi^*_n$$

Then $\{x_n\}$ is an exact computation:
$$x_{n+1} = \delta(x_n)$$

with shadowing bound:
$$\|x_n - y_n\| = \|\xi^*_n\| \leq \frac{2C\varepsilon}{\lambda} =: \delta(\varepsilon)$$

**Algorithmic Correction Procedure:**

```
Algorithm: SHADOW-CORRECTION
Input: Approximate sequence {y_n}, linearizations {L_n}, dichotomy data
Output: Exact sequence {x_n}

1. Initialize: xi^(0) = 0 (zero deviation)
2. For m = 1, 2, ... until convergence:
   a. Compute residuals: r_n = delta(y_n + xi^(m-1)_n) - y_{n+1} - L_n(xi^(m-1)_n)
   b. Solve linear system via Green's function:
      xi^(m) = T(xi^(m-1))
   c. Check convergence: ||xi^(m) - xi^(m-1)||_inf < tolerance
3. Output: x_n = y_n + xi^(m)_n
```

**Convergence Rate.** Geometric convergence:
$$\|\xi^{(m)} - \xi^*\|_\infty \leq \theta^m \cdot \rho$$

The number of iterations to achieve tolerance $\tau$ is:
$$m^* = O\left(\frac{\lambda^2}{\varepsilon} \log\frac{\varepsilon}{\tau\lambda}\right)$$

---

## Certificate Construction

The proof yields explicit certificates for the lifting:

### Input Certificate (Approximate Solution)

$$K_{\text{pseudo}}^\varepsilon = \left(\{y_n\}_{n=0}^N, \varepsilon, \{\|e_n\|\}_{n=0}^{N-1}\right)$$

where:
- $\{y_n\}$: the approximate solution sequence
- $\varepsilon$: bound on per-step error
- $\{\|e_n\|\}$: actual errors $e_n = \delta(y_n) - y_{n+1}$

**Verification:** Check $\|e_n\| < \varepsilon$ for all $n$.

### Robustness Certificate (Spectral Gap)

$$K_{\mathrm{LS}_\sigma}^+ = \left(\lambda, C, \mu, \{E^s_n, E^u_n\}_{n=0}^N, \{P^s_n, P^u_n\}_{n=0}^N\right)$$

where:
- $\lambda > 0$: spectral gap (minimum over trajectory)
- $C \geq 1$: dichotomy constant
- $\mu > 0$: dichotomy exponent
- $\{E^s_n, E^u_n\}$: stable/unstable bundle data
- $\{P^s_n, P^u_n\}$: projection operators

**Verification:**
1. Compute linearizations $L_n = D\delta|_{y_n}$
2. Verify spectral gap: $\min_n \mathrm{dist}(\sigma(L_n), i\mathbb{R}) \geq \lambda$
3. Construct splitting and verify dichotomy bounds

### Output Certificate (Exact Solution)

$$K_{\text{true}}^{\delta(\varepsilon)} = \left(\{x_n\}_{n=0}^N, \{\xi^*_n\}_{n=0}^N, \delta(\varepsilon)\right)$$

where:
- $\{x_n\}$: the exact solution sequence
- $\{\xi^*_n\}$: deviation sequence (correction applied)
- $\delta(\varepsilon) = 2C\varepsilon/\lambda$: shadowing bound

**Verification:**
1. Check exactness: $x_{n+1} = \delta(x_n)$ for all $n$
2. Check proximity: $\|x_n - y_n\| < \delta(\varepsilon)$ for all $n$

### Certificate Logic

The complete logical structure is:
$$K_{\mathrm{LS}_\sigma}^+ \wedge K_{\text{pseudo}}^\varepsilon \wedge [\varepsilon < \varepsilon_0] \Rightarrow K_{\text{true}}^{\delta(\varepsilon)}$$

where $\varepsilon_0 = \lambda^2/(4C^2K)$ is the smallness threshold.

**Explicit Certificate Tuple:**
$$\mathcal{C} = (\text{exact\_sequence}, \text{deviation\_witness}, \text{convergence\_proof}, \text{bound})$$

where:
- `exact_sequence` $= \{x_n\}$
- `deviation_witness` $= \{\xi^*_n\}$
- `convergence_proof` $= \{\xi^{(m)}\}_{m=0}^{m^*}$ (iteration history)
- `bound` $= \delta(\varepsilon)$

---

## Quantitative Refinements

### Shadowing Distance Scaling

**Theorem (Optimal Bound).** The shadowing distance satisfies:
$$\delta(\varepsilon) = \Theta\left(\frac{\varepsilon}{\lambda}\right)$$

This bound is:
- **Achievable:** The construction gives $\delta(\varepsilon) \leq 2C\varepsilon/\lambda$
- **Optimal:** There exist systems where $\delta(\varepsilon) \geq c\varepsilon/\lambda$ (Bowen 1975)

**Interpretation:** The ratio $\varepsilon/\lambda$ measures the "difficulty" of correction:
- Large $\lambda$ (robust system): small corrections suffice
- Small $\lambda$ (near-singular system): large corrections needed

### Smallness Threshold

**Explicit Threshold.** Contraction requires:
$$\varepsilon < \varepsilon_0 = \frac{\lambda^2}{4C^2K}$$

This threshold depends on:
- $\lambda$: spectral gap (quadratic dependence)
- $C$: conditioning of the dichotomy
- $K$: Lipschitz constant of the transition

**Robustness Analysis.** The threshold scales favorably with robustness:
- Double $\lambda$ $\Rightarrow$ quadruple the allowable error
- Better conditioning (smaller $C$) $\Rightarrow$ larger threshold

### Convergence Rate Analysis

**Iteration Count.** To achieve correction accuracy $\tau$:
$$m^* = \left\lceil \frac{\log(\rho/\tau)}{\log(1/\theta)} \right\rceil = O\left(\frac{\lambda^2}{\varepsilon} \log\frac{1}{\tau}\right)$$

**Per-Iteration Cost.** Each iteration requires:
- Computing residuals: $O(N \cdot T_\delta)$ where $T_\delta$ is cost of evaluating $\delta$
- Solving linear system: $O(N \cdot d \cdot C_{\text{split}})$ where $d$ is dimension and $C_{\text{split}}$ is splitting cost

**Total Complexity:**
$$T_{\text{total}} = O\left(N \cdot \frac{\lambda^2}{\varepsilon} \cdot (T_\delta + d \cdot C_{\text{split}}) \cdot \log\frac{1}{\tau}\right)$$

### Non-Uniform Robustness

For systems with varying spectral gap $\lambda_n = \mathrm{Spec}(y_n)$:

**Local Shadowing.** In regions where $\lambda_n \geq \lambda_{\min} > 0$:
$$\|x_n - y_n\| \leq \frac{2C\varepsilon}{\lambda_{\min}}$$

**Shadowing Breakdown.** If $\lambda_n \to 0$ (robustness degenerates):
- Shadowing distance grows: $\delta \to \infty$
- Correction may fail
- Certificate cannot be constructed

This corresponds to computational phase transitions where approximate solutions cannot be lifted.

---

## Connections to Classical Results

### 1. Anosov Shadowing Lemma (1967)

**Theorem (Anosov).** For uniformly hyperbolic flows on compact manifolds (e.g., geodesic flows on negatively curved spaces), every $\varepsilon$-pseudo-orbit is $\delta(\varepsilon)$-shadowed by a true orbit.

**Connection to KRNL-Shadowing.** Anosov's lemma is the geometric prototype:
- Negative curvature $\Leftrightarrow$ exponential dichotomy
- Uniform hyperbolicity $\Leftrightarrow$ spectral gap bounded away from zero
- Shadowing $\Leftrightarrow$ approximate-to-exact lifting

**Computational Interpretation.** Geodesic flows compute distance functions. The shadowing lemma guarantees that approximate distance computations can be corrected to exact ones when the underlying space has negative curvature (robust geometry).

### 2. Self-Correction in Proof Systems

**Theorem (Blum-Luby-Rubinfeld 1993).** For linearity testing, a function that is $\varepsilon$-close to linear can be self-corrected to a linear function with high probability.

**Connection to KRNL-Shadowing.** Self-correction shares the shadowing structure:

| Self-Correction | Shadowing |
|-----------------|-----------|
| $\varepsilon$-close to linear | $\varepsilon$-pseudo-orbit |
| Corrected linear function | True orbit |
| Random sampling | Dichotomy decomposition |
| Algebraic structure | Spectral gap |

**Generalization.** The shadowing theorem provides a dynamical-systems framework for self-correction: any computation with positive spectral gap admits self-correction with error $O(\varepsilon/\lambda)$.

### 3. Error Correction in Iterative Algorithms

**Newton's Method Analogy.** Newton iteration for $f(x) = 0$:
$$x_{n+1} = x_n - (Df|_{x_n})^{-1} f(x_n)$$

If $\{y_n\}$ is an approximate Newton sequence (e.g., with inexact derivatives), the shadowing theorem guarantees:
- If the Jacobian $Df$ is well-conditioned (spectral gap), there exists an exact Newton sequence nearby
- The correction distance is $O(\varepsilon/\lambda)$ where $\lambda$ is the condition number

**Krylov Methods.** In iterative linear solvers (CG, GMRES), accumulated floating-point errors create pseudo-orbits. The spectral gap of the iteration matrix determines whether these can be shadowed by exact solutions.

### 4. Bowen's Axiom A Extension (1975)

**Theorem (Bowen).** For Axiom A diffeomorphisms, shadowing holds on the non-wandering set with optimal bound $\delta(\varepsilon) = \Theta(\varepsilon/\lambda)$.

**Key Contribution.** Bowen proved:
- Shadowing is a **topological** consequence of hyperbolicity
- The bound $\varepsilon/\lambda$ is optimal (tight lower bound exists)
- Extension to non-compact settings (chain recurrence)

**Computational Implication.** The optimality result shows that no better lifting is possible in general: the approximation error times inverse robustness is the fundamental limit.

### 5. Palmer's Contraction Mapping Proof (1988)

**Theorem (Palmer).** The shadowing lemma admits a constructive proof via the Banach fixed-point theorem on sequence spaces.

**Algorithmic Value.** Palmer's approach (which we follow) provides:
- Explicit construction of the shadowing orbit
- Convergence rate for the iteration
- Certificate of correctness

This transforms shadowing from an existence theorem to a computational procedure.

### 6. Computer-Assisted Proofs

**Applications (Zgliczynski-Mischaikow 2001).** Rigorous numerics uses shadowing to:
- Validate long-time simulations of chaotic systems
- Prove existence of periodic orbits
- Certify global dynamics from finite computation

**Connection to KRNL-Shadowing.** The certificate construction enables computer-assisted proofs:
1. Run numerical simulation (generate pseudo-orbit)
2. Verify spectral gap along trajectory
3. Apply shadowing theorem to certify true orbit exists

---

## Extension: Infinite-Time Shadowing

### Bi-Infinite Sequences

For bi-infinite pseudo-orbits $\{y_n\}_{n \in \mathbb{Z}}$:

**Theorem (Infinite-Time Shadowing).** Under uniform hyperbolicity, every bi-infinite $\varepsilon$-pseudo-orbit is $\delta(\varepsilon)$-shadowed by a unique true orbit.

**Uniqueness.** The infinite-time case provides uniqueness (not just existence):
- Forward iteration in $E^s$ directions
- Backward iteration in $E^u$ directions
- Both must converge, pinning down a unique solution

### Periodic Orbits

**Corollary (Periodic Shadowing).** If $\{y_n\}_{n=0}^{N-1}$ is a periodic pseudo-orbit (i.e., $y_N$ is $\varepsilon$-close to $y_0$), there exists a true periodic orbit $\{x_n\}$ with period $N$ shadowing it.

**Application to Cycle Detection.** Approximate cycles in computation (e.g., detected by Floyd's algorithm) can be certified as shadows of true cycles when the spectral gap is positive.

---

## Application: Lifting Heuristic Solutions

### Framework for Heuristic-to-Exact Lifting

**Problem Setup.** Given:
- Optimization problem: find $x^*$ with $F(x^*) = 0$
- Heuristic algorithm producing approximate solution $y$ with $\|F(y)\| < \varepsilon$
- Iterative refinement $\delta(x) = x - (DF|_x)^{-1}F(x)$ (Newton-type)

**Lifting Protocol:**
1. **Generate pseudo-orbit:** Apply heuristic to get $y_0$, then iterate: $y_{n+1} \approx \delta(y_n)$ (with errors)
2. **Check robustness:** Verify $\mathrm{Spec}(y_n) \geq \lambda > 0$ (well-conditioned Jacobians)
3. **Apply shadowing:** Construct exact solution $x_n = y_n + \xi^*_n$
4. **Output:** Exact solution $x^*$ within $\delta(\varepsilon)$ of heuristic

### Example: SAT Solving

**Local Search Heuristics.** Consider WalkSAT-type algorithms:
- Configuration space: partial assignments
- Transition: flip variable to reduce unsatisfied clauses
- Approximate solutions: assignments with few conflicts

**Robustness Analysis.** The spectral gap measures:
- How much clause satisfaction changes under variable flips
- Zero gap $\Rightarrow$ stuck at local minimum
- Positive gap $\Rightarrow$ consistent improvement direction

**Shadowing Interpretation.** If local search finds an assignment with few conflicts and positive spectral gap, the shadowing theorem guarantees a satisfying assignment exists nearby (if one exists at all in that region).

---

## Summary

The KRNL-Shadowing theorem, translated to complexity theory, establishes the **Approximate-to-Exact Lifting Principle**:

1. **Fundamental Correspondence:**
   - Pseudo-orbit $\leftrightarrow$ approximate computation
   - True orbit $\leftrightarrow$ exact solution
   - Spectral gap $\leftrightarrow$ robustness parameter
   - Shadowing distance $\leftrightarrow$ correction bound

2. **Main Result:** If an approximate solution has uniformly positive robustness (spectral gap $\lambda > 0$), there exists an exact solution within distance $O(\varepsilon/\lambda)$.

3. **Algorithmic Construction:** Fixed-point iteration on sequence spaces provides an explicit correction procedure with geometric convergence.

4. **Optimality:** The bound $\delta(\varepsilon) = \Theta(\varepsilon/\lambda)$ is tight; no better lifting is possible in general.

5. **Self-Correction Foundation:** The theorem provides the dynamical-systems foundation for self-correcting algorithms and error-resilient computation.

6. **Certificate Structure:**
   $$K_{\mathrm{LS}_\sigma}^+ \wedge K_{\text{pseudo}}^\varepsilon \Rightarrow K_{\text{true}}^{\delta(\varepsilon)}$$

This translation reveals that shadowing in hyperbolic dynamics is the continuous analog of self-correction in proof systems: both rely on structural robustness (spectral gap) to lift approximate objects to exact ones.
