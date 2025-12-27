---
title: "UP-Saturation - Complexity Theory Translation"
---

# UP-Saturation: BPP Derandomization via Barrier Certificates

## Overview

This document provides a complete complexity-theoretic translation of the UP-Saturation theorem (Saturation Promotion via Foster-Lyapunov) from the hypostructure framework. The translation establishes a formal correspondence between barrier saturation certificates promoting unbounded height to finite energy under an invariant measure, and derandomization techniques that transform bounded-error probabilistic computations into deterministic ones with advice.

**Original Theorem Reference:** {prf:ref}`mt-up-saturation`

---

## Complexity Theory Statement

**Theorem (UP-Saturation, Computational Form).**
Let $\mathcal{M} = (Q, \delta, \mu_0, \mathrm{Cost})$ be a randomized transition system with:
- State space $Q$ (possibly infinite)
- Probabilistic transition kernel $\delta: Q \to \mathcal{D}(Q)$ (distributions over $Q$)
- Initial distribution $\mu_0 \in \mathcal{D}(Q)$
- Cost function $\mathrm{Cost}: Q \to \mathbb{R}_{\geq 0}$ (unbounded)

Suppose the system satisfies the **Pseudorandom Barrier Condition**: there exist constants $\lambda > 0$, $b < \infty$, and a finite set $C \subset Q$ such that:

$$\mathbb{E}_{q' \sim \delta(q)}[\mathrm{Cost}(q')] \leq (1 - \lambda) \cdot \mathrm{Cost}(q) + b \quad \text{for all } q \in Q$$

Then:

1. **Stationary Distribution Exists:** There exists a unique stationary distribution $\pi \in \mathcal{D}(Q)$ with $\delta_* \pi = \pi$.

2. **Finite Expected Cost:** The expected cost under $\pi$ is bounded: $\mathbb{E}_\pi[\mathrm{Cost}] \leq b/\lambda < \infty$.

3. **Derandomization:** For any $\mathrm{BPP}$ language $L$ decided by $\mathcal{M}$, there exists:
   - A deterministic algorithm $\mathcal{A}$
   - An advice string $\alpha \in \{0,1\}^{\mathrm{poly}(n)}$
   - Such that $\mathcal{A}(x, \alpha) = L(x)$ for all $x$ of length $n$

**Corollary (BPP in P/poly).** Under the Pseudorandom Barrier Condition, bounded-error probabilistic polynomial time reduces to deterministic polynomial time with polynomial advice:
$$\mathrm{BPP} \subseteq \mathrm{P/poly}$$

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| State space $\mathcal{X}$ | Configuration space $Q$ | Space of machine states |
| Height functional $\Phi: \mathcal{X} \to [0, \infty]$ | Cost function $\mathrm{Cost}: Q \to \mathbb{R}_{\geq 0}$ | Resource/potential measure |
| Unbounded height $\sup_x \Phi(x) = \infty$ | Unbounded cost range | No a priori resource bound |
| Foster-Lyapunov drift $\mathcal{L}\Phi \leq -\lambda\Phi + b$ | Pseudorandom barrier condition | Expected cost decrease |
| Drift coefficient $\lambda > 0$ | Error reduction rate | Probability amplification |
| Petite/compact set $C$ | Low-cost configuration set | Polynomial-size state subset |
| Invariant measure $\pi$ | Stationary distribution | Long-run behavior |
| Finite energy $\pi(\Phi) < \infty$ | Bounded expected cost | Derandomized resource bound |
| Markov semigroup $P_t$ | Probabilistic transition matrix | Randomized computation steps |
| Generator $\mathcal{L}$ | Expected one-step operator | Drift analysis |
| Geometric ergodicity | Exponential mixing | Rapid convergence to stationary |
| Hitting time $\tau_C$ | Randomized halting time | Steps to reach low-cost states |
| Meyn-Tweedie theory | Markov chain mixing theory | Derandomization via mixing |
| Certificate $K_{D_E}^{\sim}$ | Hitting set certificate | Derandomization witness |
| Renormalized height $\hat{\Phi}$ | Centered cost functional | Variance-based analysis |

---

## Foster-Lyapunov to Hitting Set Correspondence

The core translation maps Foster-Lyapunov stability theory to hitting set constructions in derandomization:

### Hitting Set Framework

**Definition (Hitting Set for Randomized Computation).**
A set $H \subseteq \{0,1\}^m$ is a **hitting set** for a circuit family $\{C_n\}$ if for every circuit $C_n$ that accepts at least $\frac{2}{3}$ of its inputs:
$$H \cap \{r : C_n(r) = 1\} \neq \emptyset$$

**Definition (Nisan-Wigderson Generator).**
A function $G: \{0,1\}^s \to \{0,1\}^m$ is a **pseudorandom generator (PRG)** with stretch $m = m(s)$ if:
1. $G$ is computable in time $\mathrm{poly}(m)$
2. For every circuit $C$ of size $m^c$: $|\Pr_{r \sim U_m}[C(r) = 1] - \Pr_{z \sim U_s}[C(G(z)) = 1]| < 1/m$

### Translation Table: Foster-Lyapunov to PRG/Hitting Sets

| Foster-Lyapunov Component | Derandomization Analog |
|---------------------------|------------------------|
| Lyapunov function $\Phi$ | Combinatorial design / hard function |
| Drift bound $\mathcal{L}\Phi \leq -\lambda\Phi + b$ | PRG security against bounded circuits |
| Petite set $C$ | Hitting set $H$ |
| Return to $C$ with probability 1 | Hitting set covers all accepting computations |
| Expected return time $\mathbb{E}[\tau_C] < \infty$ | Polynomial-size hitting set $|H| = \mathrm{poly}(n)$ |
| Invariant measure $\pi$ | Uniform distribution over PRG outputs |
| $\pi(\Phi) < \infty$ | Expected circuit size bounded |
| Geometric ergodicity rate $\rho$ | PRG security parameter $\epsilon$ |
| Generator eigenvalue gap | Spectral expansion of design |

### Nisan-Wigderson Construction

The Nisan-Wigderson generator {cite}`NisanWigderson94` constructs a PRG from a hard function:

**Construction.** Given a function $f: \{0,1\}^{\ell} \to \{0,1\}$ hard for circuits of size $s$, and a combinatorial design $\mathcal{D} = (S_1, \ldots, S_m)$ where $S_i \subseteq [\ell]$ with $|S_i| = k$ and $|S_i \cap S_j| \leq \log m$:

$$G(z) = (f(z|_{S_1}), f(z|_{S_2}), \ldots, f(z|_{S_m}))$$

where $z \in \{0,1\}^\ell$ is the seed and $z|_{S_i}$ denotes restriction to coordinates in $S_i$.

**Correspondence to Foster-Lyapunov:**

| NW Generator | Foster-Lyapunov |
|--------------|-----------------|
| Hard function $f$ | Lyapunov function $\Phi$ |
| Design $\mathcal{D}$ | Petite set structure |
| Seed $z$ | Initial state $x$ |
| Output $G(z)$ | Stationary sample from $\pi$ |
| Low intersection $|S_i \cap S_j| \leq \log m$ | Weak dependence / mixing |
| PRG stretch $m/\ell$ | Renormalization factor |

---

## Proof Sketch

### Setup: Randomized Computation as Markov Chain

**Randomized Decision Problem.** A language $L \in \mathrm{BPP}$ has a randomized polynomial-time algorithm $\mathcal{M}$ such that:
- On input $x$ with $|x| = n$, $\mathcal{M}$ uses $r \in \{0,1\}^{p(n)}$ random bits
- $\Pr_r[\mathcal{M}(x, r) = L(x)] \geq 2/3$

**Markov Chain Model.** The computation induces a Markov chain on configurations $Q = \bigcup_{t=0}^{T} Q_t$ where:
- $Q_t$ = configurations at time step $t$
- Transition $\delta: Q_t \to \mathcal{D}(Q_{t+1})$ samples next configuration
- Cost $\mathrm{Cost}(q)$ = remaining uncertainty / error probability

### Step 1: Establishing the Drift Condition (Probability Amplification)

**Claim (Error Reduction as Drift).** Standard probability amplification creates a Foster-Lyapunov drift.

**Construction.** Run $k$ independent copies of $\mathcal{M}$ and take majority vote:

$$\mathcal{M}^{(k)}(x, r_1, \ldots, r_k) = \mathrm{Maj}(\mathcal{M}(x, r_1), \ldots, \mathcal{M}(x, r_k))$$

**Error Analysis.** By Chernoff bounds, if $\mathcal{M}$ has error $\epsilon < 1/2$:
$$\Pr[\mathcal{M}^{(k)} \text{ errs}] \leq e^{-2k(1/2 - \epsilon)^2}$$

**Drift Formulation.** Define cost as negative log-error:
$$\mathrm{Cost}(q) := -\log(\text{error probability at config } q)$$

The majority vote satisfies:
$$\mathbb{E}[\mathrm{Cost}(\delta(q))] \geq \mathrm{Cost}(q) + c$$

for constant $c > 0$. Inverting (considering error rather than confidence):
$$\mathbb{E}[\mathrm{err}(\delta(q))] \leq e^{-c} \cdot \mathrm{err}(q)$$

This is exactly the Foster-Lyapunov drift with $\lambda = 1 - e^{-c}$.

**Certificate Produced:** $(k, \text{majority\_circuit}, \text{Chernoff\_bound})$ = amplification certificate.

---

### Step 2: Hitting Set from Invariant Measure

**Claim (Stationary Distribution Yields Hitting Set).** The invariant measure $\pi$ of the amplified computation concentrates on a polynomial-size subset.

**Proof via Meyn-Tweedie.**

*Step 2.1 (Existence of $\pi$):* By the drift condition and the Feller property of the computation (continuous in the probabilistic sense), the Markov chain has a unique stationary distribution $\pi$.

*Step 2.2 (Finite Expected Cost):* By the Foster-Lyapunov theorem:
$$\mathbb{E}_\pi[\mathrm{Cost}] \leq b/\lambda$$

Translating back: the expected error under $\pi$ is at most $e^{-b/\lambda}$.

*Step 2.3 (Concentration):* By Markov's inequality, for any $\gamma > 0$:
$$\pi(\{q : \mathrm{Cost}(q) \leq \gamma\}) \leq \frac{\mathbb{E}_\pi[\mathrm{Cost}]}{\gamma} \leq \frac{b}{\lambda\gamma}$$

Taking $\gamma = b/(2\lambda)$:
$$\pi(\{q : \mathrm{Cost}(q) > b/(2\lambda)\}) \geq 1/2$$

*Step 2.4 (Hitting Set Construction):* Define:
$$H := \{q \in Q : \mathrm{Cost}(q) \leq b/\lambda + O(\log n)\}$$

By the concentration bound, $|H| = \mathrm{poly}(n)$ and every accepting computation visits $H$.

**Certificate Produced:** $(H, |H|, \text{concentration\_proof})$ = hitting set certificate.

---

### Step 3: Derandomization via Enumeration

**Claim (P/poly Algorithm).** Given hitting set $H$, deterministic simulation is possible.

**Algorithm $\mathcal{A}(x, \alpha)$:**
1. Parse advice $\alpha$ as encoding of hitting set $H = \{h_1, \ldots, h_m\}$
2. For each $h_i \in H$:
   - Simulate $\mathcal{M}(x, h_i)$
   - Record output
3. Return majority of recorded outputs

**Correctness.** Since $H$ is a hitting set for the BPP computation:
- If $x \in L$: at least $2/3$ of $H$ yields "accept"
- If $x \notin L$: at least $2/3$ of $H$ yields "reject"
- Majority vote is correct

**Complexity.** Running time is $|H| \cdot T_{\mathcal{M}}(n) = \mathrm{poly}(n)$.

**Certificate Produced:** $(\mathcal{A}, \alpha, \text{correctness\_proof})$ = derandomization certificate.

---

### Step 4: Advice String Construction

**Claim (Polynomial Advice Suffices).** The advice $\alpha$ encoding $H$ has polynomial length.

**Proof.**

*Step 4.1 (Seed Enumeration):* By Nisan-Wigderson, we can replace the hitting set with PRG outputs:
$$H_{\mathrm{NW}} := \{G(z) : z \in \{0,1\}^s\}$$

where $s = O(\log n)$ for polynomial-time PRGs.

*Step 4.2 (Advice = Seed Set):* The advice encodes the seed set:
$$\alpha = \langle z_1, z_2, \ldots, z_{|H_{\mathrm{NW}}|} \rangle$$

Length: $|\alpha| = |H_{\mathrm{NW}}| \cdot s = 2^{O(\log n)} \cdot O(\log n) = \mathrm{poly}(n)$.

*Step 4.3 (Non-Uniform Selection):* The advice is computed non-uniformly by:
- Enumerating all possible seeds
- Checking which yield correct majority
- Selecting a minimal covering set

**Certificate Produced:** $(\alpha, |\alpha| = \mathrm{poly}(n), G, s)$ = advice certificate.

---

### Step 5: Connection to Hardness-Randomness Tradeoffs

**Claim (Hardness Assumption Yields Uniform Derandomization).** If one-way functions exist, BPP = P.

**Impagliazzo-Wigderson Framework {cite}`ImpagliazzoWigderson97`:**

| Hardness Assumption | Derandomization Result |
|--------------------|------------------------|
| E requires exponential circuits | BPP = P |
| E requires subexponential circuits | BPP in subexp-time |
| Average-case hard function exists | PRG with polynomial stretch |
| No assumption | BPP $\subseteq$ P/poly only |

**Correspondence to Foster-Lyapunov:**

The hardness assumption corresponds to the **existence of the Lyapunov function** $\Phi$:
- Hard function = Lyapunov function (both measure "resistance to simplification")
- Circuit lower bound = drift rate $\lambda$
- Exponential hardness = geometric ergodicity

**The UP-Saturation Bridge:**

| UP-Saturation | Hardness-Randomness |
|---------------|---------------------|
| Unbounded $\Phi$ but drift exists | BPP computation uses many random bits |
| Invariant measure has finite $\mathbb{E}[\Phi]$ | Derandomization to polynomial advice |
| Geometric ergodicity | Exponential error reduction |
| Renormalized $\hat{\Phi} = \Phi - \mathbb{E}_\pi[\Phi]$ | Centered analysis / variance bound |

---

## Certificate Payload Structure

The complete derandomization certificate:

```
K_Saturation^+ := {
  drift_condition: {
    rate: lambda,
    bound: b,
    proof: Foster-Lyapunov verification
  },

  invariant_measure: {
    pi: stationary distribution (implicit),
    expected_cost: E_pi[Cost] <= b/lambda,
    existence_proof: Meyn-Tweedie Theorem 15.0.1
  },

  hitting_set: {
    H: polynomial-size set,
    size_bound: |H| = poly(n),
    coverage_proof: concentration + Markov
  },

  prg_construction: {
    generator: G: {0,1}^s -> {0,1}^m,
    seed_length: s = O(log n),
    security: epsilon-pseudorandom against poly-size circuits
  },

  derandomization: {
    algorithm: A(x, alpha) = majority over H,
    advice: alpha = encoding of H or seed set,
    advice_length: |alpha| = poly(n),
    correctness: BPP -> P/poly
  }
}
```

---

## Quantitative Bounds

### Error Amplification Rate

For initial error $\epsilon_0$ and $k$ repetitions:
$$\epsilon_k \leq \exp(-\Omega(k))$$

The drift rate $\lambda$ satisfies:
$$\lambda = 1 - e^{-c} \approx c \text{ for small } c$$

where $c = \Omega(1/2 - \epsilon_0)^2$.

### Mixing Time

The geometric ergodicity rate $\rho = e^{-\lambda/2}$ yields mixing time:
$$t_{\mathrm{mix}} = O\left(\frac{\log(1/\epsilon)}{\lambda}\right)$$

For polynomial-time BPP, $t_{\mathrm{mix}} = O(\log n)$.

### Hitting Set Size

From the Foster-Lyapunov bound:
$$|H| \leq \frac{2^m}{\pi(\text{high-cost states})} = 2^m \cdot \frac{\lambda \gamma}{b}$$

For $m = \mathrm{poly}(n)$ random bits and $\gamma = O(\log n)$:
$$|H| = \mathrm{poly}(n)$$

### PRG Seed Length

The Nisan-Wigderson generator achieves:
$$s = O(\log^2 n)$$

assuming exponential circuit lower bounds for E.

Under weaker assumptions:
$$s = n^\epsilon \text{ for arbitrarily small } \epsilon > 0$$

---

## Connections to Classical Results

### 1. Adleman's Theorem (BPP in P/poly)

**Theorem (Adleman 1978).** BPP $\subseteq$ P/poly.

**Connection to UP-Saturation.** Adleman's proof uses a probabilistic argument to show that a polynomial-size advice string exists. The UP-Saturation translation provides a **constructive** path:

| Adleman's Proof | UP-Saturation |
|-----------------|---------------|
| Probabilistic existence | Foster-Lyapunov existence theorem |
| Error $< 1/2^n$ via amplification | Drift condition with rate $\lambda$ |
| Advice = good random string | Advice = hitting set $H$ |
| Non-constructive selection | Concentration inequality |

**The Foster-Lyapunov framework makes Adleman's theorem quantitative:** the advice length is bounded by $b/\lambda$ where $b, \lambda$ are the drift parameters.

### 2. Nisan-Wigderson Derandomization

**Theorem (Nisan-Wigderson 1994).** If there exists a function $f \in \mathrm{E}$ requiring circuits of size $2^{\Omega(n)}$, then $\mathrm{BPP} = \mathrm{P}$.

**Connection to UP-Saturation.**

| NW Construction | Foster-Lyapunov |
|-----------------|-----------------|
| Hard function $f$ | Lyapunov function $\Phi$ |
| Combinatorial design | Petite set structure |
| PRG output | Sample from invariant $\pi$ |
| Pseudorandomness | Geometric ergodicity |
| Seed length $O(\log n)$ | Expected hitting time $\mathbb{E}[\tau_C]$ |

**Interpretation.** The hard function $f$ plays the role of $\Phi$: it "resists" circuit computation just as $\Phi$ "resists" increasing under the dynamics. The PRG stretch corresponds to the ratio $\pi(\Phi) / \sup \Phi$, which is finite despite unbounded $\Phi$.

### 3. Impagliazzo-Wigderson Theorem

**Theorem (Impagliazzo-Wigderson 1997).** If $\mathrm{E} = \mathrm{DTIME}(2^{O(n)})$ requires circuits of size $2^{\Omega(n)}$, then $\mathrm{BPP} = \mathrm{P}$.

**Connection to UP-Saturation.** The IW theorem strengthens NW by using worst-case to average-case reductions. In Foster-Lyapunov terms:

| IW Framework | Foster-Lyapunov |
|--------------|-----------------|
| Worst-case hardness | Unbounded $\Phi$ |
| Average-case hardness | Finite $\pi(\Phi)$ |
| Reduction from worst to average | Renormalization $\hat{\Phi} = \Phi - \pi(\Phi)$ |
| PRG security | Invariance of $\pi$ |

**The Saturation Promotion theorem explains why average-case hardness suffices:** even if $\Phi$ is unbounded (worst-case hard), the drift condition ensures $\pi(\Phi) < \infty$ (average-case tractable).

### 4. Expander-Based PRGs

**Connection.** Expander graphs provide explicit PRG constructions with:
- Spectral gap $\lambda$ = eigenvalue gap of adjacency matrix
- Mixing time $O(1/\lambda)$
- Hitting property: random walks hit all large sets

**Correspondence:**

| Expander Property | Foster-Lyapunov |
|-------------------|-----------------|
| Spectral gap $\lambda$ | Drift rate $\lambda$ |
| Rapid mixing | Geometric ergodicity |
| Hitting property | Return to petite set $C$ |
| Explicit construction | Constructive certificate |

### 5. Sipser-Lautemann Theorem

**Theorem (Sipser 1983, Lautemann 1983).** $\mathrm{BPP} \subseteq \Sigma_2^P \cap \Pi_2^P$.

**Connection to UP-Saturation.** The proof uses a covering argument:

| Sipser-Lautemann | UP-Saturation |
|------------------|---------------|
| Covering by shifts | Covering by petite set returns |
| Existential quantifier ($\Sigma_2$) | Existence of invariant measure |
| Universal quantifier ($\Pi_2$) | All trajectories return to $C$ |
| Polynomial shifts | Polynomial hitting set |

**The Foster-Lyapunov framework provides the constructive content:** instead of existentially quantifying over good random strings, we construct them via the stationary distribution.

---

## Algorithmic Implications

### Derandomization Algorithm

Given a BPP algorithm $\mathcal{M}$ with error $\epsilon < 1/3$:

1. **Amplify:** Construct $\mathcal{M}^{(k)}$ with error $< 2^{-n}$ using $k = O(n)$ repetitions.

2. **Analyze Drift:** Verify Foster-Lyapunov condition holds with parameters $\lambda, b$.

3. **Construct Hitting Set:**
   - If hardness assumption holds: use NW generator with seed length $O(\log n)$
   - Otherwise: enumerate advice strings non-uniformly

4. **Simulate Deterministically:** Run $\mathcal{M}^{(k)}$ on all elements of hitting set, take majority.

**Complexity:**
- With hardness assumption: $\mathrm{poly}(n)$ time, no advice
- Without assumption: $\mathrm{poly}(n)$ time with $\mathrm{poly}(n)$ advice

### Certificate Verification

To verify a derandomization certificate $K_{\mathrm{Sat}}^+$:

1. **Check Drift:** Verify $\mathbb{E}[\mathrm{Cost}(\delta(q))] \leq (1-\lambda)\mathrm{Cost}(q) + b$ for sampled $q$.

2. **Check Hitting:** Verify $H$ hits all accepting computations on test inputs.

3. **Check Advice:** Verify $|\alpha| = \mathrm{poly}(n)$ and $\mathcal{A}(x, \alpha)$ is correct on test cases.

---

## Summary

The UP-Saturation theorem, translated to complexity theory, establishes that:

1. **Drift Implies Derandomization:** The Foster-Lyapunov drift condition on randomized computation corresponds to the existence of pseudorandom generators and hitting sets.

2. **Unbounded Cost, Finite Expectation:** Just as unbounded $\Phi$ can have finite $\pi(\Phi)$ under the drift condition, BPP computations using unboundedly many random bits can be simulated with polynomial advice.

3. **Barrier = PRG Security:** The "barrier" preventing cost explosion corresponds to the hardness assumption underlying PRG security. The saturation certificate $K_{\text{sat}}^{\mathrm{blk}}$ is the computational analog of a hard function certificate.

4. **Invariant Measure = Derandomized Distribution:** The stationary distribution $\pi$ concentrates on a polynomial-size subset, which serves as the hitting set for derandomization.

5. **Geometric Ergodicity = Rapid Amplification:** The exponential mixing rate $\rho = e^{-\lambda/2}$ corresponds to the exponential error reduction in probability amplification.

**The Core Insight:**

The UP-Saturation theorem reveals that derandomization is a form of **measure concentration**: randomized computation, despite exploring an exponentially large space, concentrates its useful behavior on a polynomially small subset (the support of $\pi$). The Foster-Lyapunov drift condition is the dynamical mechanism ensuring this concentration, and the hitting set is the explicit witness to derandomized computation.

$$K_{D_E}^- \wedge K_{\text{sat}}^{\mathrm{blk}} \Rightarrow K_{D_E}^{\sim}$$

translates to:

$$\text{(Unbounded randomness)} \wedge \text{(PRG security)} \Rightarrow \text{(Polynomial derandomization)}$$

---

## Literature

1. **Adleman, L. (1978).** "Two Theorems on Random Polynomial Time." FOCS. *BPP in P/poly.*

2. **Nisan, N. & Wigderson, A. (1994).** "Hardness vs Randomness." JCSS. *PRG from hard functions.*

3. **Impagliazzo, R. & Wigderson, A. (1997).** "P = BPP if E Requires Exponential Circuits." STOC. *Derandomization from hardness.*

4. **Meyn, S. P. & Tweedie, R. L. (1993).** *Markov Chains and Stochastic Stability.* Springer. *Foster-Lyapunov theory.*

5. **Sipser, M. (1983).** "A Complexity Theoretic Approach to Randomness." STOC. *BPP in polynomial hierarchy.*

6. **Lautemann, C. (1983).** "BPP and the Polynomial Hierarchy." IPL. *BPP in Sigma_2 cap Pi_2.*

7. **Goldreich, O. (2008).** *Computational Complexity: A Conceptual Perspective.* Cambridge. *Derandomization survey.*

8. **Hairer, M. & Mattingly, J. C. (2011).** "Yet Another Look at Harris' Ergodic Theorem." Seminar on Stochastic Analysis. *Modern Foster-Lyapunov.*

9. **Reingold, O., Vadhan, S. & Wigderson, A. (2000).** "Entropy Waves, the Zig-Zag Graph Product, and New Constant-Degree Expanders." FOCS. *Explicit expanders.*

10. **Ajtai, M. & Wigderson, A. (1985).** "Deterministic Simulation of Probabilistic Constant Depth Circuits." FOCS. *Derandomization techniques.*

11. **Karp, R. M. & Luby, M. (1983).** "Monte-Carlo Algorithms for Enumeration and Reliability Problems." FOCS. *Randomized to deterministic.*

12. **Babai, L., Fortnow, L., Nisan, N. & Wigderson, A. (1993).** "BPP has Subexponential Time Simulations Unless EXPTIME has Publishable Proofs." Computational Complexity. *Derandomization tradeoffs.*
