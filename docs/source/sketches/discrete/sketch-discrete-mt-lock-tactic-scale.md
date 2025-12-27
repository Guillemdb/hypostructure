---
title: "LOCK-Tactic-Scale - Complexity Theory Translation"
---

# LOCK-Tactic-Scale: Type II Exclusion via Scale Hierarchy

## Original Hypostructure Statement

**Theorem (LOCK-Tactic-Scale, Type II Exclusion):** Let $\mathcal{S}$ be a hypostructure satisfying interface permits $D_E$ and $\mathrm{SC}_\lambda$ with scaling exponents $(\alpha, \beta)$ satisfying $\alpha > \beta$ (strict subcriticality). Let $x \in X$ with $\Phi(x) < \infty$ and $\mathcal{C}_*(x) < \infty$ (finite total cost). Then **no supercritical self-similar blow-up** can occur at $T_*(x)$.

More precisely: if a supercritical sequence produces a nontrivial ancient trajectory $v_\infty$, then:
$$\int_{-\infty}^0 \mathfrak{D}(v_\infty(s)) \, ds = \infty$$

**Sieve Target:** Node 4 (ScaleCheck) -- predicate $\alpha > \beta$ excludes supercritical blow-up

**Certificate Produced:** $K_4^+$ with payload $(\alpha, \beta, \alpha > \beta)$ or $K_{\text{TypeII}}^{\text{blk}}$

**Original Reference:** {prf:ref}`mt-lock-tactic-scale`

---

## Complexity Theory Statement

**Theorem (Scale Hierarchy Separation):** Let $\mathcal{C}_1$ and $\mathcal{C}_2$ be complexity classes defined by resource bounds $(T_1(n), S_1(n))$ and $(T_2(n), S_2(n))$ respectively. Suppose the scaling exponents satisfy:
- Time: $T_1(n) = O(n^\alpha)$ and $T_2(n) = \Omega(n^\beta)$
- Hierarchy gap: $\alpha < \beta$ (strict scale separation)

**Statement:** If $\alpha < \beta$ with sufficient gap (specifically, $\beta > \alpha \cdot (1 + \epsilon)$ for some $\epsilon > 0$), then $\mathcal{C}_1 \subsetneq \mathcal{C}_2$. No "scale collapse" can occur where problems in $\mathcal{C}_2$ become uniformly solvable with $\mathcal{C}_1$ resources.

More precisely: any algorithm $\mathcal{A}$ attempting to decide all of $\mathcal{C}_2$ with $\mathcal{C}_1$ resources must fail on infinitely many instances. The failure set has positive density.

**Key Insight:** The hypostructure scaling condition $\alpha > \beta$ (dissipation dominates time scaling) translates to complexity theory as resource hierarchy gaps. Just as Type II blow-up is excluded when dissipation exceeds concentration rate, complexity class collapse is excluded when resource requirements are sufficiently separated.

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Equivalent | Interpretation |
|-----------------------|------------------------------|----------------|
| Scaling exponent $\alpha$ (dissipation) | Time complexity exponent $\beta$ (requirement) | How resource demands scale with input |
| Scaling exponent $\beta$ (time) | Available time exponent $\alpha$ | Resources provided by the class |
| Subcriticality $\alpha > \beta$ | Hierarchy gap $\beta > \alpha$ | Resources insufficient for task |
| Supercritical blow-up | Complexity collapse | Lower class solves higher class |
| Type II singularity | Scale-invariant hard instances | Problems resisting compression |
| Dissipation functional $\mathfrak{D}$ | Work/computation measure | Steps required at each scale |
| Ancient trajectory $v_\infty$ | Infinite hard instance sequence | Problems at arbitrarily large scales |
| Finite total cost $\mathcal{C}_*(x) < \infty$ | Polynomial resource bound | Bounded total computation |
| Interface permit $D_E$ | Decidability condition | Problem has well-defined complexity |
| Interface permit $\mathrm{SC}_\lambda$ | Scaling uniformity | Complexity scales predictably |
| $K_4^+$ certificate | Hierarchy theorem proof | Witnesses separation |
| $K_{\text{TypeII}}^{\text{blk}}$ | Lower bound certificate | Proves impossibility of collapse |
| Rescaled time $s = \lambda_n^\beta(t - t_n)$ | Input size normalization | Measure complexity at scale $n$ |
| Scale $\lambda \to 0$ | Input size $n \to \infty$ | Asymptotic analysis |
| Self-similar solution | Scale-invariant problem family | Same structure at all input sizes |

---

## Proof Sketch (Complexity Theory Version)

### Setup: Time and Space Hierarchies

**Definitions:**

1. **Resource Bound:** For time-constructible $f: \mathbb{N} \to \mathbb{N}$, define:
   $$\text{DTIME}(f(n)) := \{L : L \text{ decided by TM in } O(f(n)) \text{ steps}\}$$

2. **Scaling Exponent:** For complexity class $\mathcal{C}$ with bound $f(n) = \Theta(n^\alpha)$, the scaling exponent is $\alpha$.

3. **Hierarchy Gap:** Classes $\mathcal{C}_1 \subseteq \text{DTIME}(n^\alpha)$ and $\mathcal{C}_2 \supseteq \text{DTIME}(n^\beta)$ have gap $\beta - \alpha$.

4. **Scale Collapse:** A hypothetical algorithm $\mathcal{A}$ achieving $\mathcal{C}_2 \subseteq \mathcal{C}_1$ (lower resources suffice for higher class).

**Resource Functional (Dissipation Analogue):**

Define the computational work at scale $n$:
$$\mathfrak{D}_\mathcal{A}(n) := \max_{|x| = n} \text{steps}(\mathcal{A}(x))$$

This measures the maximum computation required at input size $n$. The total cost integral becomes:
$$\mathcal{C}_*(\mathcal{A}) = \sum_{n=1}^{\infty} \frac{\mathfrak{D}_\mathcal{A}(n)}{n^{\alpha+1}}$$

For algorithms with scaling exponent $\alpha$, this converges. For algorithms requiring exponent $\beta > \alpha$, this diverges.

---

### Step 1: Change of Variables (Scale Normalization)

**Claim (Rescaled Complexity):** For algorithms with conjectured scaling $\lambda = n^{-1}$, the rescaled computation measure transforms as:

$$\mathfrak{D}(\mathcal{A}, n) = n^\beta \cdot \mathfrak{D}_{\text{normalized}}$$

**Proof:**

**Step 1.1 (Size Rescaling):** Define normalized input size $s = n/N$ for reference scale $N$. An algorithm with time $T(n) = n^\beta$ has:
$$T(n) = N^\beta \cdot s^\beta = N^\beta \cdot T_{\text{unit}}(s)$$

**Step 1.2 (Work Distribution):** The computational work at scale $n$ relative to scale $N$ is:
$$\frac{\mathfrak{D}(n)}{\mathfrak{D}(N)} = \left(\frac{n}{N}\right)^\beta$$

This is the complexity-theoretic analogue of the dissipation scaling in the hypostructure proof.

**Step 1.3 (Normalization):** Rescaling to unit reference:
$$\mathfrak{D}_{\text{norm}}(s) = s^{-\beta} \cdot \mathfrak{D}(s \cdot N)$$

The normalized measure is scale-invariant for self-similar problem families.

---

### Step 2: Dissipation Scaling (Resource Transformation)

**Claim (Resource Bound Transformation):** For an algorithm $\mathcal{A}$ with claimed scaling exponent $\alpha$, attempting to solve problems requiring exponent $\beta$:

$$\text{Available resources: } R(n) = O(n^\alpha)$$
$$\text{Required resources: } W(n) = \Omega(n^\beta)$$

**Proof (Incompatibility):**

**Step 2.1 (Available vs. Required):** By the interface permit $\mathrm{SC}_\lambda$, the available computation scales as:
$$R(n) = c_1 \cdot n^\alpha \cdot (1 + o(1))$$

**Step 2.2 (Demand Scaling):** By the problem's inherent difficulty, the required work scales as:
$$W(n) = c_2 \cdot n^\beta \cdot (1 + o(1))$$

**Step 2.3 (Gap Analysis):** The resource deficit is:
$$\text{Deficit}(n) = W(n) - R(n) = n^\alpha(c_2 n^{\beta-\alpha} - c_1) + o(n^\beta)$$

For $\beta > \alpha$, as $n \to \infty$:
$$\text{Deficit}(n) \to +\infty$$

The deficit grows polynomially, making collapse impossible.

---

### Step 3: Cost Transformation (Integral Divergence)

**Claim (Hierarchy Integral):** The total cost integral:
$$\mathcal{I}_{\alpha,\beta} = \int_1^N \frac{n^{\beta-1}}{n^\alpha} dn = \int_1^N n^{\beta-\alpha-1} dn$$

diverges as $N \to \infty$ when $\beta > \alpha$.

**Proof:**

**Step 3.1 (Exponent Classification):**
$$\mathcal{I}_{\alpha,\beta} = \begin{cases}
\frac{N^{\beta-\alpha} - 1}{\beta - \alpha} & \text{if } \beta \neq \alpha \\
\ln N & \text{if } \beta = \alpha
\end{cases}$$

**Step 3.2 (Divergence Condition):** The integral diverges when:
- $\beta - \alpha > 0$ (polynomial divergence): $\mathcal{I} = \Theta(N^{\beta-\alpha})$
- $\beta = \alpha$ (logarithmic divergence): $\mathcal{I} = \Theta(\ln N)$

**Step 3.3 (Convergence Condition):** The integral converges only when $\beta - \alpha < 0$, i.e., $\beta < \alpha$.

**Hierarchy Theorem Consequence:** When $\beta > \alpha$, the divergent integral proves that no algorithm with scaling $\alpha$ can solve all problems requiring scaling $\beta$. This is the complexity-theoretic Type II exclusion.

---

### Step 4: Connection to Classical Hierarchy Theorems

**Theorem (Time Hierarchy Theorem, Hartmanis-Stearns 1965):**

For time-constructible functions $f$ and $g$ with $f(n) \log f(n) = o(g(n))$:
$$\text{DTIME}(f(n)) \subsetneq \text{DTIME}(g(n))$$

**Translation to LOCK-Tactic-Scale:**

| Hypostructure | Time Hierarchy |
|---------------|----------------|
| $\alpha$ (dissipation rate) | $\log(g/f)$ (hierarchy gap) |
| $\beta$ (blow-up rate) | Time complexity exponent |
| $\alpha > \beta$ (subcriticality) | $f \log f = o(g)$ (sufficient gap) |
| Type II exclusion | Strict containment |
| Ancient trajectory | Diagonalizing language |
| $\mathfrak{D}(v_\infty) = \infty$ | Diagonalization succeeds |

**Space Hierarchy Theorem (Stearns-Hartmanis-Lewis 1965):**

For space-constructible $f$ and $g$ with $f(n) = o(g(n))$:
$$\text{DSPACE}(f(n)) \subsetneq \text{DSPACE}(g(n))$$

**Note:** Space hierarchy requires a smaller gap than time hierarchy. This corresponds to different "dissipation rates" in the hypostructure framework -- space reuse is more efficient than time reuse.

---

### Step 5: Multi-Scale Structure and Separation

**Theorem (Multi-Scale Separation):** The scale hierarchy principle extends to nested classes:

$$\text{DTIME}(n) \subsetneq \text{DTIME}(n^2) \subsetneq \text{DTIME}(n^3) \subsetneq \cdots$$

Each separation is witnessed by problems that are "self-similar" at their characteristic scale -- they resist compression to lower scales.

**Type II Interpretation:**

- **Scale $k$:** Problems solvable in $O(n^k)$ time
- **Type II at scale $k$:** Problems requiring exactly $\Theta(n^k)$ time (not $o(n^k)$)
- **Exclusion:** Problems at scale $k+1$ cannot "collapse" to scale $k$
- **Self-Similar Blow-up:** The diagonalizing language exhibits scale-invariant hardness

**Certificate Construction:**

$$K_{\text{Hierarchy}}^+ = (\alpha, \beta, \beta > \alpha \cdot (1 + \epsilon), L_{\text{diag}})$$

where $L_{\text{diag}}$ is the diagonalizing language witnessing the separation.

---

## Certificate Construction

**Mode: Scale Separation (Type II Excluded)**

```
K_ScaleSeparation = {
  mode: "TypeII_Excluded",
  mechanism: "Scale_Hierarchy",
  evidence: {
    lower_class: DTIME(n^alpha),
    upper_class: DTIME(n^beta),
    gap_condition: "beta > alpha * (1 + epsilon)",
    diagonalizing_language: L_diag,
    hierarchy_theorem: "Hartmanis-Stearns 1965"
  },
  certificate_logic: "K_SC^+ AND K_DE^+ => K_TypeII^blk",
  payload: (alpha, beta, beta > alpha)
}
```

**Mode: Complexity Class Collapse (Impossible)**

```
K_CollapseBlocked = {
  mode: "Collapse_Blocked",
  mechanism: "Integral_Divergence",
  evidence: {
    attempted_collapse: "DTIME(n^beta) subseteq DTIME(n^alpha)",
    cost_integral: "divergent for beta > alpha",
    deficit_growth: "Theta(n^{beta-alpha})",
    impossibility: "no uniform algorithm exists"
  },
  certificate_logic: "Cost divergence => collapse impossible",
  literature: "Hartmanis-Stearns 1965"
}
```

---

## Connections to Classical Results

### 1. Time Hierarchy Theorem (Hartmanis-Stearns 1965)

**Statement:** For time-constructible $f, g$ with $f(n) \log f(n) = o(g(n))$:
$$\text{DTIME}(f(n)) \subsetneq \text{DTIME}(g(n))$$

**Connection to LOCK-Tactic-Scale:**

The time hierarchy theorem is the prototypical scale separation result. The condition $f \log f = o(g)$ provides the "dissipation gap" $\alpha > \beta$ needed to exclude collapse:

- **Dissipation ($\alpha$):** The $\log f$ overhead from simulation/diagonalization
- **Scaling ($\beta$):** The polynomial growth rate of $g/f$
- **Exclusion:** When $g$ grows faster than $f \log f$, separation is guaranteed

**Proof Mechanism:** The universal simulator uses $O(f(n) \log f(n))$ time to simulate machines running in time $f(n)$. The diagonalizing language $L_d$ runs the simulator and flips the output. This language:
- Is in $\text{DTIME}(g(n))$ (uses the full time budget)
- Is not in $\text{DTIME}(f(n))$ (diagonalizes against all such machines)

**Hypostructure Reading:** The ancient trajectory $v_\infty$ is the diagonalizing language. The infinite dissipation integral $\int \mathfrak{D}(v_\infty) = \infty$ corresponds to the simulation overhead being unbounded.

### 2. Space Hierarchy Theorem (Stearns-Hartmanis-Lewis 1965)

**Statement:** For space-constructible $f, g$ with $f(n) = o(g(n))$:
$$\text{DSPACE}(f(n)) \subsetneq \text{DSPACE}(g(n))$$

**Connection:** Space hierarchy requires a smaller gap than time hierarchy because space is reusable (Savitch's theorem: $\text{NSPACE}(f) \subseteq \text{DSPACE}(f^2)$). This corresponds to lower "dissipation rate" in the hypostructure framework.

**Scale Comparison:**

| Resource | Hierarchy Gap Required | Dissipation Interpretation |
|----------|------------------------|---------------------------|
| Time | $f \log f = o(g)$ | High overhead, strong dissipation |
| Space | $f = o(g)$ | Low overhead, weak dissipation |
| Nondeterministic Space | $f = o(g)$ | Reuse enables tighter bounds |

### 3. Nondeterministic Time Hierarchy (Cook 1973)

**Statement:** For time-constructible $f, g$ with $f(n+1) = o(g(n))$:
$$\text{NTIME}(f(n)) \subsetneq \text{NTIME}(g(n))$$

**Connection:** Nondeterministic computation requires an even larger gap for separation. This reflects the "more supercritical" nature of nondeterminism -- guessing power concentrates computational energy, requiring stronger dissipation to separate.

### 4. Padding Arguments and Scale Invariance

**Theorem (Padding Lemma):** If $L \in \text{DTIME}(f(n))$, then the padded language:
$$L_{\text{pad}} = \{x \cdot 0^{g(|x|)} : x \in L\}$$
is in $\text{DTIME}(f(g^{-1}(n)))$.

**Connection to Self-Similarity:** Padding creates scale-invariant problem families:
- Padding compresses complexity at larger scales
- The padded problem has the "same structure" at all scales
- This is the complexity-theoretic analogue of self-similar solutions

**Type II Interpretation:** Self-similar solutions in PDEs (which cause Type II blow-up) correspond to padded languages in complexity:
- Both exhibit scale invariance
- Both resist "collapse" to smaller scales
- The exclusion condition ($\alpha > \beta$) prevents such invariant structures from existing

### 5. Relativized Hierarchies and Oracle Separation

**Theorem (Baker-Gill-Solovay 1975):** There exist oracles $A, B$ such that:
- $\text{P}^A = \text{NP}^A$
- $\text{P}^B \neq \text{NP}^B$

**Connection:** Oracle access can modify the "scaling exponents" of complexity classes:
- Oracle $A$ collapses the hierarchy (changes $\beta$ to match $\alpha$)
- Oracle $B$ maintains separation (preserves $\beta > \alpha$)

**Hypostructure Reading:** Oracles modify the interface permits $D_E$ and $\mathrm{SC}_\lambda$, changing the scaling relationship and potentially enabling or preventing Type II blow-up.

---

## Quantitative Bounds

### Hierarchy Gap Requirements

**Time Hierarchy:**
$$\text{Separation guaranteed if: } g(n) = \omega(f(n) \log f(n))$$

For polynomial classes with $f(n) = n^\alpha$ and $g(n) = n^\beta$:
$$\beta > \alpha + \frac{\alpha \log n}{n} = \alpha + o(1)$$

So any $\beta > \alpha$ suffices asymptotically.

**Space Hierarchy:**
$$\text{Separation guaranteed if: } g(n) = \omega(f(n))$$

For polynomial space, $\beta > \alpha$ suffices.

### Dissipation Integral Bounds

**Cost Integral for Scale Separation:**
$$\mathcal{I}_{\alpha,\beta}(N) = \int_1^N n^{\beta-\alpha-1} dn$$

| Gap $\beta - \alpha$ | Integral Growth | Separation Strength |
|---------------------|-----------------|---------------------|
| $< 0$ | Convergent | Collapse possible |
| $= 0$ | $\log N$ | Marginal separation |
| $= 1$ | $N$ | Linear separation |
| $= 2$ | $N^2$ | Quadratic separation |

### Diagonalization Efficiency

**Simulation Overhead:**

For universal simulation of time-$f(n)$ machines:
$$T_{\text{sim}}(n) = O(f(n) \log f(n))$$

For universal simulation of space-$f(n)$ machines:
$$S_{\text{sim}}(n) = O(f(n))$$

The different overheads explain why space hierarchy requires smaller gaps.

---

## Multi-Scale Certificate Summary

The LOCK-Tactic-Scale theorem, in complexity-theoretic form, establishes:

1. **Scale Hierarchy Principle:** Complexity classes form a strict hierarchy when resource bounds are sufficiently separated.

2. **Gap Condition ($\alpha > \beta$):** The dissipation rate (simulation overhead) must exceed the scaling rate (complexity growth) for separation.

3. **Type II Exclusion:** No "complexity collapse" where higher classes are uniformly solvable with lower-class resources.

4. **Self-Similar Resistance:** Scale-invariant problems (analogous to self-similar blow-up solutions) cannot exist when the gap condition holds.

5. **Certificate Production:** The diagonalizing language provides an explicit separation witness.

**The Certificate:**
$$K_{\text{Scale}} = \begin{cases}
K_4^+ = (\alpha, \beta, \alpha > \beta) & \text{if gap condition verified} \\
K_{\text{TypeII}}^{\text{blk}} & \text{if scale collapse attempted and blocked}
\end{cases}$$

---

## Physical Interpretation

The complexity-theoretic translation reveals a deep analogy:

| PDE Phenomenon | Complexity Phenomenon |
|----------------|----------------------|
| Energy dissipation rate $\alpha$ | Simulation overhead $\log f$ |
| Time scaling rate $\beta$ | Complexity growth rate |
| Type II blow-up | Complexity class collapse |
| Self-similar solution | Scale-invariant hard problems |
| Ancient trajectory | Diagonalizing language |
| Finite total cost | Polynomial resource bounds |
| Subcriticality $\alpha > \beta$ | Sufficient hierarchy gap |
| Blow-up exclusion | Separation theorem |

**The Core Insight:** Just as PDEs cannot exhibit finite-time Type II blow-up when dissipation dominates time scaling, complexity classes cannot collapse when simulation overhead dominates complexity growth. The hierarchy theorems are the complexity-theoretic manifestation of the exclusion principle.

---

## Literature

1. **Hartmanis, J., Stearns, R. E. (1965).** "On the Computational Complexity of Algorithms." *Transactions of the AMS.* *Establishes time hierarchy theorem.*

2. **Stearns, R. E., Hartmanis, J., Lewis, P. M. (1965).** "Hierarchies of Memory Limited Computations." *FOCS.* *Space hierarchy theorem.*

3. **Cook, S. A. (1973).** "A Hierarchy for Nondeterministic Time Complexity." *JCSS.* *Nondeterministic time hierarchy.*

4. **Baker, T., Gill, J., Solovay, R. (1975).** "Relativizations of the P =? NP Question." *SIAM J. Comput.* *Oracle relativization of hierarchies.*

5. **Merle, F., Zaag, H. (1998).** "Optimal Estimates for Blowup Rate and Behavior for Nonlinear Heat Equations." *Duke Math. J.* *Monotonicity formula for Type II exclusion.*

6. **Kenig, C. E., Merle, F. (2006).** "Global Well-Posedness, Scattering and Blow-Up for the Energy-Critical NLS." *Inventiones.* *Rigidity and scale separation in PDEs.*

7. **Arora, S., Barak, B. (2009).** *Computational Complexity: A Modern Approach.* Cambridge. *Standard reference for hierarchy theorems.*

8. **Tao, T. (2006).** "Nonlinear Dispersive Equations: Local and Global Analysis." *CBMS.* *Subcritical/supercritical classification.*

9. **Struwe, M. (1988).** "On the Evolution of Harmonic Maps in Higher Dimensions." *J. Differential Geom.* *Self-similar blow-up analysis.*

10. **Zheng, D. (2018).** "Hierarchies and Separations in Computational Complexity." *Survey.* *Modern treatment of hierarchy theorems.*
