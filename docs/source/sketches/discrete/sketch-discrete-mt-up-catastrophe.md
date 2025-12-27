---
title: "UP-Catastrophe - Complexity Theory Translation"
---

# UP-Catastrophe: Phase Transitions and Sharp Thresholds

## Overview

This document provides a complete complexity-theoretic translation of the UP-Catastrophe theorem (Catastrophe-Stability Promotion) from the hypostructure framework. The translation establishes a formal correspondence between bifurcation catastrophes promoting degenerate stiffness to higher-order gradient domination, and sharp threshold phenomena in random combinatorial structures, with connections to random SAT phase transitions and Friedgut's theorem on coarse thresholds.

**Original Theorem Reference:** {prf:ref}`mt-up-catastrophe`

---

## Complexity Theory Statement

**Theorem (UP-Catastrophe, Computational Form).**
Let $\mathcal{P}_n(p)$ be a monotone property on random structures $G(n, p)$ (e.g., random graphs, random $k$-SAT formulas). Suppose:

1. **Degenerate First-Order Behavior:** At the critical threshold $p_c$, the "linear stiffness" vanishes:
   $$\frac{d}{dp}\Pr[\mathcal{P}_n(p)]\Big|_{p = p_c} = 0 \quad \text{or is discontinuous}$$

2. **Canonical Catastrophe Structure:** The property exhibits a normal form of order $k \geq 2$:
   $$\Pr[\mathcal{P}_n(p)] \sim F\left(\frac{p - p_c}{w(n)}\right)$$
   where $F$ is a universal scaling function and $w(n)$ is the threshold width.

3. **Higher-Order Stiffness:** The $k$-th derivative provides sharp control:
   $$\frac{d^k}{dp^k}\Pr[\mathcal{P}_n(p)]\Big|_{p = p_c} \neq 0$$

**Statement (Sharp Threshold Promotion):**
While linear behavior fails at criticality, the system exhibits:

1. **Sharp Threshold:** The property transitions from $\Pr \approx 0$ to $\Pr \approx 1$ within a window of width $w(n) = o(1)$.

2. **Polynomial Scaling:** Convergence to the threshold follows polynomial (not exponential) behavior:
   $$|\Pr[\mathcal{P}_n(p)] - \mathbf{1}_{p > p_c}| \leq C \cdot |p - p_c|^{1/(k-1)}$$

3. **Universal Exponents:** The critical exponents are determined by the catastrophe normal form.

**Corollary (Friedgut's Dichotomy).**
For monotone graph properties:
- Either the threshold is **sharp** (width $O(1/\log n)$), or
- The property has **coarse threshold** (width $\Omega(1)$) characterized by bounded complexity (local witnesses).

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Potential $V(x)$ | Free energy / log-probability functional | $V(p) = -\log \Pr[\mathcal{P}_n(p)]$ |
| Degenerate critical point $V''(x^*) = 0$ | Critical threshold $p_c$ | Linear response vanishes |
| Canonical catastrophe $V(x) = x^{k+1}/(k+1)$ | Normal form of phase transition | Universal scaling function |
| Fold catastrophe ($k=2$) | Second-order phase transition | Continuous order parameter |
| Cusp catastrophe ($k=3$) | First-order phase transition | Discontinuous jump |
| Higher-order stiffness $V^{(k+1)}(x^*) \neq 0$ | Sharp threshold condition | Finite threshold width |
| Polynomial convergence $t^{-1/(k-1)}$ | Polynomial window $w(n) = n^{-1/(k-1)}$ | Critical scaling exponent |
| Bifurcation parameter | Clause-to-variable ratio $r = m/n$ | Control parameter |
| Order parameter | Fraction of satisfying assignments | Emergent observable |
| Catastrophe manifold | Phase diagram | Boundary between SAT/UNSAT |
| Linear stiffness $\lambda_1 = 0$ | Coarse threshold | Logarithmic influence sum |
| Nonlinear stiffness | Sharp threshold | Concentrated influences |
| Gradient domination (higher order) | Threshold sharpness | $w(n) \to 0$ as $n \to \infty$ |
| Certificate $K_{\mathrm{LS}_{\partial^k V}}^+$ | Threshold width bound | Proof of sharpness |

---

## Phase Transitions in Random Structures

### The Random $k$-SAT Model

**Definition (Random $k$-SAT).**
A random $k$-SAT instance $\Phi(n, m)$ consists of:
- $n$ Boolean variables $x_1, \ldots, x_n$
- $m$ clauses, each a disjunction of $k$ uniformly random literals

The clause-to-variable ratio is $r = m/n$.

**Critical Threshold Phenomenon.**
There exists $r_k^* > 0$ such that:
$$\lim_{n \to \infty} \Pr[\Phi(n, rn) \text{ is SAT}] = \begin{cases} 1 & r < r_k^* \\ 0 & r > r_k^* \end{cases}$$

**Known Values:**
- $k = 2$: $r_2^* = 1$ (exact, linear structure)
- $k = 3$: $r_3^* \approx 4.267$ (rigorous bounds: $3.52 < r_3^* < 4.49$)
- Large $k$: $r_k^* = 2^k \ln 2 - O(k)$ (first moment threshold)

### Correspondence to Catastrophe Theory

| Catastrophe Element | Random $k$-SAT Analog |
|---------------------|----------------------|
| Control parameter $\mu$ | Clause-to-variable ratio $r$ |
| State variable $x$ | Solution density / order parameter |
| Potential $V(x; \mu)$ | Free energy $\Phi(r) = -\frac{1}{n}\log Z_n(r)$ |
| Degenerate critical point | Critical ratio $r_k^*$ |
| Catastrophe normal form | Scaling function near $r_k^*$ |
| Bifurcation set | SAT/UNSAT phase boundary |
| Catastrophe manifold | Replica-symmetric solution space |

---

## Proof Sketch

### Setup: Threshold Width as Higher-Order Stiffness

**Definition (Threshold Width).**
For a monotone property $\mathcal{P}_n$, the threshold width at level $\epsilon$ is:
$$w_\epsilon(n) := p_{1-\epsilon}(n) - p_\epsilon(n)$$
where $p_\alpha(n) = \inf\{p : \Pr[\mathcal{P}_n(p)] \geq \alpha\}$.

**Definition (Sharp Threshold).**
A property has a sharp threshold if $w_\epsilon(n) = o(1)$ for all $\epsilon > 0$.

**Definition (Coarse Threshold).**
A property has a coarse threshold if $w_\epsilon(n) = \Theta(1)$ for some $\epsilon > 0$.

### Step 1: Influences and the Russo-Margulis Formula

**Definition (Influence).**
For a Boolean function $f: \{0,1\}^n \to \{0,1\}$ and coordinate $i$:
$$I_i(f) := \Pr_{x}[f(x) \neq f(x^{\oplus i})]$$
where $x^{\oplus i}$ flips the $i$-th coordinate.

**Theorem 1.1 (Russo-Margulis Formula).**
For monotone $f$ with $\Pr_p[f = 1] = \mu(p)$:
$$\frac{d\mu}{dp} = \sum_{i=1}^n I_i^{(p)}(f)$$

where $I_i^{(p)}$ is the influence at bias $p$.

**Interpretation in Catastrophe Terms:**
- Total influence $I(f) = \sum_i I_i(f)$ corresponds to **linear stiffness**
- At criticality: $I(f)$ may diverge or vanish in specific ways
- Sharp threshold $\Leftrightarrow$ large total influence near $p_c$
- Coarse threshold $\Leftrightarrow$ bounded total influence

**Certificate Produced:** $(I(f), \{I_i(f)\}, p_c)$ = influence profile.

---

### Step 2: Friedgut's Threshold Characterization

**Theorem 2.1 (Friedgut 1999).**
Let $f: \{0,1\}^n \to \{0,1\}$ be a monotone function with threshold $p_c$. Then exactly one of the following holds:

1. **Sharp Threshold:** $I(f_{p_c}) = \omega(1)$, and
   $$w(n) = O\left(\frac{1}{\log n}\right)$$

2. **Coarse Threshold:** $I(f_{p_c}) = O(1)$, and $f$ is approximated by a junta (function of $O(1)$ coordinates).

**Proof Sketch.**

*Step 2.1 (Bourgain's Hypercontractivity Argument):*
Using the Bonami-Beckner hypercontractive inequality:
$$\|T_\rho f\|_q \leq \|f\|_p \quad \text{for } \rho \leq \sqrt{(p-1)/(q-1)}$$

where $T_\rho$ is the noise operator.

*Step 2.2 (Fourier Analysis):*
The Fourier expansion $f = \sum_{S \subseteq [n]} \hat{f}(S) \chi_S$ satisfies:
$$I_i(f) = \sum_{S \ni i} \hat{f}(S)^2$$

*Step 2.3 (Sharp Threshold from Spread Influences):*
If $\max_i I_i(f) = o(1)$ but $\sum_i I_i(f) = \omega(1)$, then no single coordinate dominates, forcing a sharp threshold.

*Step 2.4 (Coarse Threshold from Concentrated Influences):*
If a bounded set $J \subset [n]$ with $|J| = O(1)$ satisfies $\sum_{i \in J} I_i(f) = \Theta(\sum_i I_i(f))$, then $f$ is a junta, yielding coarse threshold.

**Certificate Produced:** $(\text{sharp}/\text{coarse}, J, \max_i I_i)$ = threshold type certificate.

---

### Step 3: Catastrophe Normal Forms and Critical Exponents

**Connection to Thom's Classification.**
The seven elementary catastrophes have computational analogs:

| Catastrophe | Normal Form | Phase Transition Type | Example |
|-------------|-------------|----------------------|---------|
| Fold ($A_2$) | $x^3 + \mu x$ | Continuous (2nd order) | Percolation |
| Cusp ($A_3$) | $x^4 + \mu x^2 + \nu x$ | Discontinuous (1st order) | $k$-SAT, $k \geq 3$ |
| Swallowtail ($A_4$) | $x^5 + \mu x^3 + \nu x^2 + \rho x$ | Tricritical | Random hypergraphs |
| Butterfly ($A_5$) | $x^6 + \cdots$ | Higher multicritical | Constraint satisfaction |
| Elliptic umbilic ($D_4^-$) | $x^3 - xy^2 + \cdots$ | Two order parameters | Coloring + SAT |
| Hyperbolic umbilic ($D_4^+$) | $x^3 + y^3 + \cdots$ | Coupled transitions | XOR-SAT |
| Parabolic umbilic ($D_5$) | $x^2y + y^4 + \cdots$ | Mixed criticality | NAE-SAT |

**Theorem 3.1 (Critical Exponents from Catastrophe Order).**
For a phase transition with catastrophe of order $k$:

1. **Window Width:** $w(n) \sim n^{-1/\nu}$ where $\nu = \nu(k)$
2. **Order Parameter Scaling:** $\langle m \rangle \sim (r - r_c)^\beta$ where $\beta = \beta(k)$
3. **Fluctuation Scaling:** $\chi \sim |r - r_c|^{-\gamma}$ where $\gamma = \gamma(k)$

**For Random $k$-SAT:**
- $k = 2$ (fold-like): $\beta = 1$, $\nu = 3/2$, $\gamma = 1$ (exact)
- $k \geq 3$ (cusp-like): $\beta$, $\nu$, $\gamma$ depend on replica structure

**Certificate Produced:** $(k, \nu, \beta, \gamma)$ = critical exponent tuple.

---

### Step 4: Random SAT Phase Transition Structure

**Theorem 4.1 (Random $k$-SAT Sharp Threshold, Friedgut 1999).**
For any $k \geq 2$, the satisfiability of random $k$-SAT has a sharp threshold:
$$\forall \epsilon > 0: \lim_{n \to \infty} w_\epsilon(n) = 0$$

**Proof Outline.**

*Step 4.1 (Monotonicity):*
SAT is a monotone decreasing property in clause addition. Define:
$$f_r: \{0,1\}^{\binom{N}{k}} \to \{0,1\}$$
as the indicator that the random clause set is satisfiable.

*Step 4.2 (Influence Calculation):*
Each potential clause $C$ has influence:
$$I_C(f_r) = \Pr[\Phi \text{ is SAT}] - \Pr[\Phi \cup \{C\} \text{ is SAT}]$$

The total influence is:
$$I(f_r) = \binom{N}{k} \cdot \mathbb{E}[I_C] = \Theta(n^k) \cdot \mathbb{E}[\text{marginal clause damage}]$$

*Step 4.3 (Divergence Near Threshold):*
At $r = r_k^*$:
- Expected number of solutions: $\mathbb{E}[Z] = 2^n (1 - 2^{-k})^m = 2^{n(1 - r \cdot 2^{-k}/\ln 2 + o(1))}$
- Marginal clause damage: each clause eliminates $\Theta(Z/2^k)$ solutions
- Total influence: $I(f_r) = \Theta(n) \cdot \frac{\mathbb{E}[\text{damage}]}{\mathbb{E}[Z]} = \omega(1)$

*Step 4.4 (Friedgut's Theorem Application):*
Since $I(f_{r_k^*}) = \omega(1)$ and individual influences are $O(1/n)$, Friedgut's theorem implies a sharp threshold.

**Certificate Produced:** $(r_k^*, w(n) = o(1), I(f) = \omega(1))$ = SAT threshold certificate.

---

### Step 5: Higher-Order Stiffness and Polynomial Convergence

**Catastrophe-Stiffness Correspondence.**
The key insight is that degenerate first-order behavior (flat potential) is rescued by higher-order terms:

| Hypostructure | Random Structure |
|---------------|------------------|
| $V'(x^*) = 0$ (equilibrium) | $p = p_c$ (threshold) |
| $V''(x^*) = 0$ (degenerate) | Coarse threshold if only this |
| $V^{(k+1)}(x^*) \neq 0$ | Sharp threshold via $k$-th order |

**Theorem 5.1 (Polynomial Rate from Higher-Order Stiffness).**
If the phase transition has catastrophe order $k$, then:
$$|\Pr[\mathcal{P}_n(p)] - \mathbf{1}_{p > p_c}| \leq C \cdot n^{-1/(k-1)}$$

for $p$ at distance $\Omega(n^{-1/(k-1)})$ from $p_c$.

**Proof.**
The scaling function $F(z)$ satisfies:
$$F(z) - \mathbf{1}_{z > 0} \sim |z|^{1/(k-1)}$$

for $|z| \to \infty$. With $z = n^{1/(k-1)}(p - p_c)$, the result follows.

**Comparison to Exponential Convergence:**
- **Non-degenerate case** ($k = 1$, $V'' > 0$): Exponential $e^{-cn}$
- **Fold catastrophe** ($k = 2$): $n^{-1}$ polynomial
- **Cusp catastrophe** ($k = 3$): $n^{-1/2}$ polynomial
- **Higher catastrophes**: $n^{-1/(k-1)}$ polynomial

**Certificate Produced:** $(k, 1/(k-1), \text{polynomial\_rate})$ = convergence certificate.

---

## Certificate Payload Structure

The complete catastrophe-stability certificate:

```
K_Catastrophe^+ := {
  catastrophe_order: {
    type: k (fold=2, cusp=3, ...),
    normal_form: V(x) = x^{k+1}/(k+1),
    control_parameters: (r - r_c)
  },

  threshold_structure: {
    critical_point: r_c (or p_c),
    width: w(n) = n^{-1/nu},
    sharpness: sharp (w -> 0) or coarse (w = O(1))
  },

  influence_analysis: {
    total_influence: I(f) = sum_i I_i(f),
    max_influence: max_i I_i(f),
    threshold_type: Friedgut classification
  },

  critical_exponents: {
    nu: correlation length exponent,
    beta: order parameter exponent,
    gamma: susceptibility exponent
  },

  convergence: {
    rate: polynomial t^{-1/(k-1)},
    window: (p_c - w, p_c + w),
    scaling_function: F(z)
  }
}
```

---

## Connections to Classical Results

### 1. Friedgut's Sharp Threshold Theorem (1999)

**Theorem (Friedgut).**
Every monotone graph property has a sharp threshold.

**Connection to UP-Catastrophe.**
Friedgut's theorem shows that for graph properties, the "higher-order stiffness" always exists:

| Friedgut's Proof | UP-Catastrophe |
|------------------|----------------|
| Influences spread across all edges | Higher derivative $V^{(k+1)} \neq 0$ |
| Hypercontractivity controls Fourier mass | Catastrophe normal form controls convergence |
| $\sum_i I_i = \omega(1)$ | Nonlinear stiffness positive |
| Sharp threshold width $O(1/\log n)$ | Polynomial convergence $t^{-1/(k-1)}$ |

**The UP-Catastrophe Framework:**
Graph properties have implicit higher-order stiffness because:
- No single edge can determine the property (symmetry)
- This forces influences to spread, creating large total influence
- Large total influence implies sharp threshold

### 2. Random SAT Phase Transition (Achlioptas et al. 2000s)

**Theorem (Ding-Sly-Sun 2015).**
For random $k$-SAT with $k$ large, the satisfiability threshold is:
$$r_k^* = 2^k \ln 2 - \frac{1 + \ln 2}{2} + o_k(1)$$

**Connection to UP-Catastrophe.**

| SAT Phase Transition | Catastrophe Theory |
|---------------------|-------------------|
| First moment bound ($r < r_k^*$) | Potential minimum exists |
| Second moment method | Hessian analysis |
| Threshold $r_k^*$ | Bifurcation point |
| Condensation transition | Cusp catastrophe |
| Frozen variables | Higher-order stiffness |

**The Condensation Phenomenon:**
Below $r_k^*$, solutions cluster into exponentially many "states." The transition between SAT and UNSAT exhibits cusp-like behavior:
- First-order (discontinuous) in order parameter
- Sharp threshold in probability
- Polynomial finite-size scaling

### 3. Threshold for $k$-Colorability (Achlioptas-Naor 2005)

**Theorem.**
For random graphs $G(n, p = c/n)$, the chromatic number satisfies:
$$\chi(G) \approx \frac{c}{2\ln c} \quad \text{with high probability}$$

**Connection to UP-Catastrophe.**
The colorability threshold exhibits:
- **Fold catastrophe** for $\chi = 2$ (bipartiteness): continuous transition
- **Cusp catastrophe** for $\chi \geq 3$: discontinuous, sharp threshold
- Critical exponents determined by catastrophe order

### 4. Erd\H{o}s-R\'enyi Phase Transition (1960)

**Theorem (Giant Component).**
In $G(n, p = c/n)$:
- $c < 1$: all components have size $O(\log n)$
- $c > 1$: unique giant component of size $\Theta(n)$
- $c = 1$: critical window with components of size $\Theta(n^{2/3})$

**Connection to UP-Catastrophe.**
This is the prototype for catastrophe-stiffness promotion:

| Giant Component | UP-Catastrophe |
|-----------------|----------------|
| $c = 1$ critical point | Degenerate stiffness $V'' = 0$ |
| Scaling window $|c - 1| = O(n^{-1/3})$ | Width from $k = 2$ (fold) |
| Component size $\sim n^{2/3}$ | Order parameter exponent $\beta = 1/3$ |
| Universal Airy distribution | Catastrophe scaling function |

### 5. Bollobas-Thomason Threshold (1987)

**Theorem.**
Every non-trivial monotone graph property has a threshold.

**Connection to UP-Catastrophe.**
The Bollobas-Thomason theorem is the existence half; Friedgut's theorem provides sharpness. Together:

| Result | Catastrophe Analog |
|--------|-------------------|
| Threshold exists | Critical point of potential |
| Threshold is sharp | Higher-order stiffness promotes |
| Width $\to 0$ | Polynomial convergence |

---

## Quantitative Bounds

### Threshold Width for Random SAT

For random $k$-SAT at ratio $r$:
$$w(n) = O\left(\frac{1}{n^{1/2}}\right) \quad \text{(conjectured for } k \geq 3\text{)}$$

Rigorous bounds:
$$w(n) = O\left(\frac{1}{\log n}\right) \quad \text{(Friedgut)}$$

### Influence Bounds

For satisfiability of $\Phi(n, rn)$:
$$I(\text{SAT}) = \Theta(n) \quad \text{at } r = r_k^*$$

Individual clause influence:
$$I_C = O(1) \quad \text{for any specific clause } C$$

### Critical Exponents for Random Graphs

Near the giant component threshold $c = 1$:
- Order parameter exponent: $\beta = 1$ (fraction in giant)
- Correlation length exponent: $\nu = 1/3$ (window width)
- Susceptibility exponent: $\gamma = 1$ (fluctuations)

These satisfy the hyperscaling relation:
$$2\beta + \gamma = \nu d_c$$
where $d_c = 6$ is the upper critical dimension.

---

## Algorithmic Implications

### Detecting Phase Transitions

**Algorithm: THRESHOLD-DETECT**
```
Input: Monotone property P_n, precision epsilon
Output: Threshold estimate p_c with error epsilon

1. Binary search for p such that Pr[P_n(p)] in [1/3, 2/3]
2. Estimate I(f_p) = sum of influences
3. If I(f_p) = omega(1): sharp threshold
   - Refine: p_c in [p - O(1/I(f_p)), p + O(1/I(f_p))]
4. If I(f_p) = O(1): coarse threshold
   - Identify junta coordinates
   - Threshold determined by junta structure

Complexity: O(n log(1/epsilon)) samples for sharp thresholds
```

### Hardness Near Threshold

**Theorem (Computational Hardness at Criticality).**
For random $k$-SAT at $r = r_k^*$:
- Finding a solution (if one exists) requires $2^{\Omega(n)}$ time (conjectured)
- Distinguishing SAT from UNSAT is hard (average-case complexity)
- Refutation above threshold is co-NP-hard

**Connection to Catastrophe:**
The computational hardness at criticality corresponds to the **degenerate Hessian**:
- No local algorithm can efficiently navigate the flat potential
- Global information (captured by higher-order terms) is required
- The polynomial convergence rate implies polynomial-time inapproximability

### Planted Solutions and Quiet Planting

**Algorithm Insight:**
Above the condensation threshold, solutions can be "quietly planted" without changing the distribution significantly. This corresponds to:
- **Cusp catastrophe**: multiple coexisting equilibria
- **Metastability**: planted solution is a local minimum
- **Detection hardness**: distinguishing planted from random is hard

---

## Summary

The UP-Catastrophe theorem, translated to complexity theory, establishes **Phase Transition Promotion**:

1. **Fundamental Correspondence:**
   - Degenerate critical point $V''(x^*) = 0$ $\leftrightarrow$ Critical threshold $p_c$
   - Canonical catastrophe $V = x^{k+1}/(k+1)$ $\leftrightarrow$ Normal form of phase transition
   - Higher-order stiffness $V^{(k+1)} \neq 0$ $\leftrightarrow$ Sharp threshold via Friedgut
   - Polynomial convergence $t^{-1/(k-1)}$ $\leftrightarrow$ Threshold width $w(n) = n^{-1/(k-1)}$

2. **Main Result:** If a phase transition has degenerate first-order behavior but non-degenerate higher-order behavior (catastrophe order $k$), then:
   - Sharp threshold exists with width $w(n) = o(1)$
   - Convergence is polynomial with exponent $1/(k-1)$
   - Critical exponents are universal (determined by $k$)

3. **Friedgut's Theorem as Higher-Order Stiffness:**
   - Total influence $I(f) = \omega(1)$ is the "nonlinear stiffness" condition
   - Spread of influences (no coordinate dominates) forces sharpness
   - This is the discrete analog of $V^{(k+1)} \neq 0$

4. **Certificate Structure:**
   $$K_{\mathrm{LS}_\sigma}^- \wedge K_{\mathrm{LS}_{\partial^k V}}^+ \Rightarrow K_{\mathrm{LS}_\sigma}^{\sim} \quad (\text{Polynomial Rate})$$

   Linear stiffness fails but higher-order stiffness promotes to polynomial gradient domination.

5. **Random SAT as Canonical Example:**
   - $k$-SAT threshold $r_k^*$ exhibits cusp catastrophe ($k \geq 3$)
   - Sharp threshold via Friedgut (influence analysis)
   - Polynomial finite-size scaling near threshold
   - Computational hardness at criticality from degenerate Hessian

This translation reveals that phase transitions in random structures are the discrete analog of bifurcation catastrophes: both exhibit degenerate first-order behavior rescued by higher-order terms, leading to universal scaling laws and polynomial (rather than exponential) convergence near criticality.

$$K_{\mathrm{LS}_\sigma}^- \wedge K_{\mathrm{LS}_{\partial^k V}}^+ \Rightarrow K_{\mathrm{LS}_\sigma}^{\sim}$$

translates to:

$$\text{(Coarse linear threshold)} \wedge \text{(Higher-order influence spread)} \Rightarrow \text{(Sharp polynomial threshold)}$$

---

## Literature

1. **Thom, R. (1975).** *Structural Stability and Morphogenesis.* Benjamin. *Classification of elementary catastrophes.*

2. **Arnold, V. I. (1972).** "Normal Forms of Functions Near Degenerate Critical Points." Russian Math Surveys. *Catastrophe normal forms.*

3. **Poston, T. & Stewart, I. (1978).** *Catastrophe Theory and Its Applications.* Pitman. *Applications of catastrophe theory.*

4. **Friedgut, E. (1999).** "Sharp Thresholds of Graph Properties, and the $k$-SAT Problem." JAMS. *Sharp threshold theorem and $k$-SAT.*

5. **Friedgut, E. & Kalai, G. (1996).** "Every Monotone Graph Property has a Sharp Threshold." PAMS. *General sharp threshold result.*

6. **Ding, J., Sly, A. & Sun, N. (2015).** "Proof of the Satisfiability Conjecture for Large $k$." STOC. *Random $k$-SAT threshold.*

7. **Achlioptas, D. & Peres, Y. (2004).** "The Threshold for Random $k$-SAT is $2^k\log 2 - O(k)$." JAMS. *First moment threshold.*

8. **Mezard, M. & Montanari, A. (2009).** *Information, Physics, and Computation.* Oxford. *Statistical physics of random CSPs.*

9. **Bollobas, B. (2001).** *Random Graphs.* Cambridge. *Phase transitions in random graphs.*

10. **Russo, L. (1982).** "An Approximate Zero-One Law." Zeitschrift fur Wahrscheinlichkeitstheorie. *Russo-Margulis formula.*

11. **Talagrand, M. (1994).** "On Russo's Approximate Zero-One Law." Annals of Probability. *Threshold width bounds.*

12. **Bourgain, J., Kahn, J., Kalai, G., Katznelson, Y. & Linial, N. (1992).** "The Influence of Variables in Product Spaces." Israel J. Math. *Influence and threshold.*
