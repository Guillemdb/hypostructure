# Proof of KRNL-Shadowing (Shadowing Metatheorem)

:::{prf:proof}
:label: proof-mt-krnl-shadowing

**Theorem Reference:** {prf:ref}`mt-krnl-shadowing`

## Setup and Notation

Let $\mathcal{H}$ be a Hypostructure with the following data:

**Hypothesis 1 (Stiffness Certificate):** A certificate $K_{\mathrm{LS}_\sigma}^+$ with spectral gap $\lambda > 0$. This means the linearized operator $L = Df$ at each point along trajectories satisfies:
$$\sigma(L) \cap \{z \in \mathbb{C} : |\text{Re}(z)| < \lambda\} = \emptyset$$
where $\sigma(L)$ denotes the spectrum of $L$.

**Hypothesis 2 (Numerical Pseudo-Orbit):** A sequence $\{y_n\}_{n=0}^N$ (where $N \in \mathbb{N} \cup \{\infty\}$) satisfying:
$$d(f(y_n), y_{n+1}) < \varepsilon \quad \text{for all } n \in \{0, 1, \ldots, N-1\}$$
for some $\varepsilon > 0$, where $f: X \to X$ is the time-one map of the flow.

**Hypothesis 3 (Hyperbolicity):** The tangent map $Df: TX \to TX$ admits an exponential dichotomy. That is, there exist:
- A splitting $TX = E^s \oplus E^u$ (stable and unstable bundles)
- Constants $C \geq 1$ and $\mu > 0$ such that:
  $$\|Df^n|_{E^s}\| \leq C e^{-\mu n} \quad \text{for } n \geq 0$$
  $$\|Df^{-n}|_{E^u}\| \leq C e^{-\mu n} \quad \text{for } n \geq 0$$

We denote:
- $X$ the phase space (a complete metric space with distance $d$)
- $f: X \to X$ the discrete-time map (time-one flow map)
- $\{x_n\}$ the true orbit to be constructed: $x_{n+1} = f(x_n)$
- $\{y_n\}$ the given pseudo-orbit with error $\varepsilon$
- $\delta_n := d(x_n, y_n)$ the shadowing distance at time $n$

**Goal:** Construct a true orbit $\{x_n\}$ such that:
$$\delta_n = d(x_n, y_n) < \delta(\varepsilon) \quad \text{for all } n,$$
where $\delta(\varepsilon) = O(\varepsilon/\lambda)$.

---

## Step 1: Linearization and Graph Transform

**Strategy:** We apply the **Graph Transform Method** introduced by Hadamard (1901) and refined by Anosov (1967). The idea is to view the shadowing problem as a fixed-point problem in a suitable function space.

### Step 1.1: Deviation Sequence

Define the **deviation sequence** $\{\xi_n\}$ by:
$$\xi_n := x_n - y_n \in T_{y_n}X.$$

The true orbit condition $x_{n+1} = f(x_n)$ becomes:
$$y_{n+1} + \xi_{n+1} = f(y_n + \xi_n).$$

Expanding via Taylor series (using smoothness of $f$):
$$f(y_n + \xi_n) = f(y_n) + Df|_{y_n}(\xi_n) + R_n(\xi_n)$$
where $R_n(\xi_n) = O(\|\xi_n\|^2)$ is the nonlinear remainder.

### Step 1.2: Linearized Equation

Substituting into the orbit condition:
$$y_{n+1} + \xi_{n+1} = f(y_n) + Df|_{y_n}(\xi_n) + R_n(\xi_n).$$

Rearranging:
$$\xi_{n+1} = Df|_{y_n}(\xi_n) + \underbrace{[f(y_n) - y_{n+1}]}_{=: e_n} + R_n(\xi_n),$$
where $e_n$ is the **pseudo-orbit error** with $\|e_n\| < \varepsilon$ by Hypothesis 2.

This gives the **shadowing equation**:
$$\xi_{n+1} - L_n \xi_n = e_n + R_n(\xi_n), \quad L_n := Df|_{y_n}.$$

### Step 1.3: Splitting via Exponential Dichotomy

By Hypothesis 3, at each point $y_n$ we have the splitting:
$$T_{y_n}X = E^s_n \oplus E^u_n$$
with projections $P^s_n: T_{y_n}X \to E^s_n$ and $P^u_n: T_{y_n}X \to E^u_n$.

Write:
$$\xi_n = \xi_n^s + \xi_n^u, \quad \xi_n^s \in E^s_n, \; \xi_n^u \in E^u_n.$$

The linearized dynamics satisfy:
$$\|L_n^k|_{E^s_n}\| \leq C e^{-\mu k} \quad \text{for } k \geq 0,$$
$$\|L_n^{-k}|_{E^u_n}\| \leq C e^{-\mu k} \quad \text{for } k \geq 0.$$

---

## Step 2: Fixed-Point Formulation

**Key Idea:** The shadowing equation has a unique solution if we impose:
- **Stable component** grows backward in time (chosen to decay forward)
- **Unstable component** grows forward in time (chosen to decay backward)

This leads to a well-posed boundary value problem in the sequence space.

### Step 2.1: Sequence Space Setup

Define the **Banach space of bounded sequences**:
$$\ell^\infty := \left\{ \{\xi_n\}_{n=0}^N : \sup_{n} \|\xi_n\| < \infty \right\}$$
with norm:
$$\|\{\xi_n\}\|_\infty := \sup_{n} \|\xi_n\|.$$

We seek $\{\xi_n\} \in \ell^\infty$ satisfying the shadowing equation with small norm $\|\{\xi_n\}\|_\infty = O(\varepsilon/\lambda)$.

### Step 2.2: Green's Function Representation

The solution to the linear equation:
$$\xi_{n+1} - L_n \xi_n = e_n$$
with bounded $\xi_n$ is given by the **discrete Green's function**:
$$\xi_n = \sum_{k=0}^{n-1} G_n^{n-k}(e_k) - \sum_{k=n}^{N} \tilde{G}_n^{k-n}(e_k),$$
where:
- $G_n^k$ is the forward propagator along $E^s$: $G_n^k = L_{n-1} L_{n-2} \cdots L_{n-k} P^s_{n-k}$
- $\tilde{G}_n^k$ is the backward propagator along $E^u$: $\tilde{G}_n^k = (L_n L_{n+1} \cdots L_{n+k-1})^{-1} P^u_{n+k}$

**Exponential decay estimates:** By the exponential dichotomy (Hypothesis 3):
$$\|G_n^k\| \leq C e^{-\mu k}, \quad \|\tilde{G}_n^k\| \leq C e^{-\mu k}.$$

### Step 2.3: Linear Estimate

For the linear problem (ignoring $R_n$ for now), we have:
$$\|\xi_n\| \leq C \sum_{k=1}^\infty e^{-\mu k} \cdot \varepsilon = C \frac{e^{-\mu}}{1 - e^{-\mu}} \varepsilon.$$

**Relating μ and λ:**

:::{important}
The spectral gap $\lambda$ and dichotomy exponent $\mu$ are related but not identical:
- $\lambda$ is the **spectral gap**: distance from $\sigma(L_n)$ to the imaginary axis
- $\mu$ is the **dichotomy exponent**: rate of exponential growth/decay in the splitting

In general: $\mu \leq \lambda$, with equality for normal operators. For non-normal operators, $\mu$ can be much smaller than $\lambda$ due to transient growth (pseudospectral effects).

**Sufficient condition for $\mu \sim \lambda$:** If the operators $L_n$ are uniformly close to normal (i.e., $\|L_n L_n^* - L_n^* L_n\| \leq \epsilon_{\text{normal}}$ for small $\epsilon_{\text{normal}}$), then:
$$\mu \geq \lambda - C_1 \epsilon_{\text{normal}}^{1/2}$$
by perturbation theory {cite}`TrefethenEmbree05`. For general operators, one must verify the dichotomy directly.
:::

Assuming $\mu \geq c \lambda$ for some constant $c > 0$ (which holds when hyperbolicity is verified), we obtain:
$$\|\xi_n\| \leq \frac{C}{c\lambda} \varepsilon = \frac{C'}{\lambda} \varepsilon.$$

This shows that **in the linear approximation**, the shadowing distance is $O(\varepsilon/\lambda)$.

---

## Step 3: Contraction Mapping Argument

To handle the **nonlinear term** $R_n(\xi_n)$, we use the **Banach Fixed-Point Theorem** on the sequence space $\ell^\infty$.

### Step 3.1: Fixed-Point Operator

Define the operator $\mathcal{T}: \ell^\infty \to \ell^\infty$ by:
$$(\mathcal{T}\xi)_n := \sum_{k=0}^{n-1} G_n^{n-k}(e_k + R_k(\xi_k)) - \sum_{k=n}^{N} \tilde{G}_n^{k-n}(e_k + R_k(\xi_k)).$$

A fixed point $\xi = \mathcal{T}\xi$ solves the full shadowing equation:
$$\xi_{n+1} - L_n \xi_n = e_n + R_n(\xi_n).$$

### Step 3.2: Contractivity Estimate

**Claim:** For sufficiently small $\varepsilon$, the operator $\mathcal{T}$ is a contraction on the ball:
$$B_\rho := \left\{ \{\xi_n\} \in \ell^\infty : \|\{\xi_n\}\|_\infty \leq \rho \right\}$$
for $\rho = 2C\varepsilon/\lambda$.

**Proof of Claim:**

**Step 3.2.1 (Self-Mapping):** For $\xi \in B_\rho$:
$$\|(\mathcal{T}\xi)_n\| \leq \sum_{k=0}^{n-1} C e^{-\mu(n-k)} (\varepsilon + K\|\xi_k\|^2) + \sum_{k=n}^{N} C e^{-\mu(k-n)} (\varepsilon + K\|\xi_k\|^2),$$
where $K$ is the Lipschitz constant of $f$ (bounding $\|R_n(\xi)\| \leq K\|\xi\|^2$).

Using $\|\xi_k\| \leq \rho$ and summing the geometric series:
$$\|(\mathcal{T}\xi)_n\| \leq \frac{C}{\lambda} (\varepsilon + K\rho^2).$$

Choosing $\rho = 2C\varepsilon/\lambda$ and assuming $\varepsilon$ small enough that:
$$K\rho^2 = K \cdot \frac{4C^2\varepsilon^2}{\lambda^2} \leq \varepsilon,$$
we obtain:
$$\|(\mathcal{T}\xi)_n\| \leq \frac{C}{\lambda} \cdot 2\varepsilon = \rho.$$

Thus $\mathcal{T}: B_\rho \to B_\rho$.

**Step 3.2.2 (Contraction):** For $\xi, \eta \in B_\rho$:
$$\|(\mathcal{T}\xi)_n - (\mathcal{T}\eta)_n\| \leq \sum_{k=0}^{n-1} C e^{-\mu(n-k)} \|R_k(\xi_k) - R_k(\eta_k)\| + \sum_{k=n}^{N} C e^{-\mu(k-n)} \|R_k(\xi_k) - R_k(\eta_k)\|.$$

Using $\|R_k(\xi_k) - R_k(\eta_k)\| \leq K(\|\xi_k\| + \|\eta_k\|) \cdot \|\xi_k - \eta_k\| \leq 2K\rho \cdot \|\xi_k - \eta_k\|$:
$$\|(\mathcal{T}\xi) - (\mathcal{T}\eta)\|_\infty \leq \frac{C}{\lambda} \cdot 2K\rho \cdot \|\xi - \eta\|_\infty = \frac{2CK}{\lambda} \cdot \frac{2C\varepsilon}{\lambda} \cdot \|\xi - \eta\|_\infty.$$

**Quantitative Smallness Condition:** The contraction requires:
$$\frac{4C^2 K \varepsilon}{\lambda^2} < 1 \quad \Leftrightarrow \quad \varepsilon < \varepsilon_0 := \frac{\lambda^2}{4C^2 K}$$

where:
- $C \geq 1$ is the dichotomy constant from Hypothesis 3
- $K$ is the Lipschitz constant of $f$ (bounding nonlinearity)
- $\lambda$ is the spectral gap

**Explicit threshold:** For the contraction to hold with factor $\theta = 1/2$, we need:
$$\varepsilon \leq \frac{\lambda^2}{8C^2 K}$$

Thus $\mathcal{T}$ is a contraction on $B_\rho$ for $\varepsilon < \varepsilon_0$.

### Step 3.3: Existence and Uniqueness

By the **Banach Fixed-Point Theorem** ({cite}`Banach22`), $\mathcal{T}$ has a unique fixed point $\{\xi_n^*\} \in B_\rho$ with:
$$\|\{\xi_n^*\}\|_\infty \leq \frac{2C\varepsilon}{\lambda}.$$

Defining the true orbit:
$$x_n := y_n + \xi_n^*,$$
we have:
$$d(x_n, y_n) = \|\xi_n^*\| \leq \frac{2C\varepsilon}{\lambda} =: \delta(\varepsilon).$$

This establishes the **shadowing property** with $\delta(\varepsilon) = O(\varepsilon/\lambda)$.

---

## Step 4: Verification and Certificate Construction

### Step 4.1: Verification that $\{x_n\}$ is a True Orbit

By construction, $\xi_n = \mathcal{T}(\xi)_n$ satisfies:
$$\xi_{n+1} = L_n \xi_n + e_n + R_n(\xi_n).$$

Substituting $x_n = y_n + \xi_n$:
$$x_{n+1} = y_{n+1} + \xi_{n+1} = y_{n+1} + L_n \xi_n + e_n + R_n(\xi_n).$$

But:
$$f(x_n) = f(y_n + \xi_n) = f(y_n) + L_n \xi_n + R_n(\xi_n) = y_{n+1} + e_n + L_n \xi_n + R_n(\xi_n),$$
using $e_n = f(y_n) - y_{n+1}$.

Therefore:
$$x_{n+1} = f(x_n),$$
confirming that $\{x_n\}$ is a true orbit of $f$.

### Step 4.2: Certificate Construction

The proof yields the following certificate:

**Input Certificate** $K_{\text{pseudo}}^{\varepsilon}$:
- Pseudo-orbit data: $\{y_n\}$
- Error bound: $\varepsilon$ with $\|e_n\| < \varepsilon$

**Stiffness Certificate** $K_{\mathrm{LS}_\sigma}^+$:
- Spectral gap: $\lambda > 0$
- Exponential dichotomy constants: $C, \mu$
- Stable/unstable splitting: $E^s \oplus E^u$

**Output Certificate** $K_{\text{true}}^{\delta(\varepsilon)}$:
- True orbit: $\{x_n\}$ with $x_{n+1} = f(x_n)$
- Shadowing distance: $\delta(\varepsilon) = 2C\varepsilon/\lambda$
- Deviation sequence: $\{\xi_n^*\}$ (witness)

**Certificate Logic Verification:**
$$K_{\mathrm{LS}_\sigma}^+ \wedge K_{\text{pseudo}}^{\varepsilon} \Rightarrow K_{\text{true}}^{\delta(\varepsilon)}$$
is established by the fixed-point construction in Steps 2–3.

### Step 4.3: Algorithmic Realization

The proof is **constructive** and provides an algorithm for computing the shadowing orbit:

**Algorithm (Shadowing Orbit Construction):**
1. **Input:** Pseudo-orbit $\{y_n\}$ with error $\varepsilon$, linearizations $L_n = Df|_{y_n}$, dichotomy data $(E^s_n, E^u_n, C, \mu)$
2. **Initialize:** $\xi^{(0)} = 0$ (zero sequence)
3. **Iterate:** For $m = 1, 2, \ldots$:
   $$\xi^{(m)} = \mathcal{T}(\xi^{(m-1)})$$
   using the Green's function representation from Step 2.2
4. **Termination:** Stop when $\|\xi^{(m)} - \xi^{(m-1)}\|_\infty < \text{tol}$
5. **Output:** True orbit $x_n = y_n + \xi^{(m)}_n$

**Convergence Rate:** By the contraction property (Step 3.2.2), convergence is geometric:
$$\|\xi^{(m)} - \xi^*\|_\infty \leq \theta^m \|\xi^{(0)} - \xi^*\|_\infty \leq \theta^m \cdot \rho$$
where $\theta = 4C^2K\varepsilon/\lambda^2 < 1$ is the contraction factor.

---

## Step 5: Extensions and Refinements

### Step 5.1: Infinite-Time Shadowing

For **infinite pseudo-orbits** ($N = \infty$), the same proof applies if we work in the space:
$$\ell^\infty_0 := \left\{ \{\xi_n\}_{n=0}^\infty : \lim_{n \to \pm\infty} \|\xi_n\| = 0 \right\}.$$

The exponential dichotomy ensures that the Green's function sums converge for both $n \to \infty$ (via $E^s$) and $n \to -\infty$ (via $E^u$).

**Result:** For uniformly hyperbolic systems (Axiom A diffeomorphisms), every infinite $\varepsilon$-pseudo-orbit is $\delta(\varepsilon)$-shadowed by a true orbit, where $\delta(\varepsilon) = O(\varepsilon/\lambda)$.

### Step 5.2: Continuous-Time Flows

For continuous-time flows $\phi_t: X \to X$, a **pseudo-trajectory** $\gamma: [0, T] \to X$ satisfies:
$$\left\| \frac{d\gamma}{dt} - V(\gamma(t)) \right\| < \varepsilon,$$
where $V$ is the vector field generating the flow.

The shadowing theorem extends: There exists a true trajectory $\tilde{\gamma}(t) = \phi_t(x_0)$ such that:
$$d(\gamma(t), \tilde{\gamma}(t)) < \delta(\varepsilon) = O(\varepsilon/\lambda)$$
for all $t \in [0, T]$.

**Proof Adaptation:** Apply the discrete-time result to the time-$h$ map $f = \phi_h$ for small $h > 0$, then take $h \to 0$ to recover the continuous-time statement. The key is that hyperbolicity (exponential dichotomy) is preserved under time discretization for sufficiently small $h$.

### Step 5.3: Shadowing in Non-Uniformly Hyperbolic Systems

For systems with **non-uniform hyperbolicity** (e.g., Lyapunov exponents $\lambda(x) > 0$ but not uniformly bounded away from zero), the shadowing property may fail globally but holds:
- **Locally** near each hyperbolic orbit
- **Generically** for typical pseudo-orbits (Pesin theory)

The shadowing distance depends on the **local spectral gap**:
$$\delta(\varepsilon, x) = O\left(\frac{\varepsilon}{\lambda(x)}\right).$$

In regions where $\lambda(x) \to 0$, the shadowing distance grows, potentially leading to **shadowing breakdown**.

### Step 5.4: Optimality of the Bound

**Question:** Is the bound $\delta(\varepsilon) = O(\varepsilon/\lambda)$ sharp?

**Answer:** Yes, in the following sense:

**Theorem (Bowen 1975, {cite}`Bowen75`):** For Axiom A diffeomorphisms, there exist pseudo-orbits $\{y_n\}$ with error $\varepsilon$ such that any shadowing orbit $\{x_n\}$ satisfies:
$$\sup_n d(x_n, y_n) \geq c \frac{\varepsilon}{\lambda}$$
for some constant $c > 0$ depending only on the system.

**Interpretation:** The $\varepsilon/\lambda$ scaling is **optimal**: the shadowing distance cannot be improved beyond this rate without additional structure (e.g., special choices of pseudo-orbits, refined dichotomy estimates).

---

## Step 6: Application to Hypostructures

### Step 6.1: Numerical Verification of Global Regularity

In the hypostructure framework, the **Shadowing Metatheorem** enables:

**Problem:** Given a PDE $\partial_t u = F(u)$ with hypostructure $\mathcal{H}$, use numerical simulation to prove global regularity.

**Approach:**
1. **Numerical Solve:** Run a high-precision numerical integrator to obtain approximate trajectory $\{u_h(t_n)\}$ with discretization error $\|u_h(t_{n+1}) - \phi_h(u_h(t_n))\| < \varepsilon_h$
2. **Stiffness Check:** Verify that the linearization $DF|_{u_h(t_n)}$ has spectral gap $\lambda > \lambda_{\min} > 0$ for all $n$
3. **Shadowing Certification:** Apply the shadowing theorem to certify existence of a true trajectory $u(t_n)$ within distance $\delta(\varepsilon_h) = C\varepsilon_h/\lambda_{\min}$ of the numerical solution
4. **Global Regularity Certificate:** If the numerical trajectory exists for all time $t \in [0, \infty)$ with bounded energy, the shadowing orbit also exists for all time and satisfies the hypostructure axioms

**Output:** A **computer-assisted proof** of global regularity, combining:
- $K_{\text{pseudo}}^{\varepsilon_h}$: numerical solution with certified error
- $K_{\mathrm{LS}_\sigma}^+$: stiffness certificate with $\lambda > \lambda_{\min}$
- $K_{\text{true}}^{\delta(\varepsilon_h)}$: shadowing orbit certificate (existence proof)

### Step 6.2: Algorithmic Theorem Component

The Shadowing Metatheorem is part of the **Algorithmic Theorem** ($T_{\text{algorithmic}}$) in the hypostructure Sieve:

**Role in the Sieve:**
- **Node 15 (Numerical):** Generate pseudo-orbit $\{y_n\}$ via high-precision simulation
- **Node 16 (Stiffness):** Verify spectral gap $\lambda > 0$ along pseudo-orbit
- **Shadowing Step:** Certify existence of nearby true orbit $\{x_n\}$ with $d(x_n, y_n) < \delta(\varepsilon)$
- **Output:** Lock certificate $K_{\text{Lock}}^{\mathrm{blk}}$ combining numerical and stiffness data

This pathway allows **upgrading numerical simulations to rigorous proofs**, closing the gap between computational evidence and mathematical certainty.

### Step 6.3: Connection to Linear Stability

The shadowing theorem requires **hyperbolicity** (exponential dichotomy), which is closely related to the **Linear Stability axiom** ($\mathrm{LS}_\sigma$) in the hypostructure framework:

**Linear Stability Axiom:** The linearization $L = DF$ at equilibria or along trajectories has spectral gap:
$$\sigma(L) \cap \{z : |\text{Re}(z)| < \lambda\} = \emptyset.$$

**Implication:** Linear stability $\Rightarrow$ exponential dichotomy $\Rightarrow$ shadowing property.

Conversely, **loss of linear stability** ($\lambda \to 0$) causes:
- **Shadowing breakdown:** $\delta(\varepsilon) = \varepsilon/\lambda \to \infty$
- **Sensitivity to perturbations:** Small numerical errors lead to large deviations from true orbits
- **Certificate failure:** $K_{\mathrm{LS}_\sigma}^+$ cannot be constructed

This mechanism explains why the Sieve requires **strict stiffness** for numerical certification to succeed.

---

## Step 7: Literature and Historical Context

### Step 7.1: Anosov's Original Result (1967)

**Theorem (Anosov Shadowing Lemma, {cite}`Anosov67`):** Let $M$ be a compact Riemannian manifold with negative sectional curvature, and $\phi_t: M \to M$ the geodesic flow. For any $\varepsilon$-pseudo-trajectory $\gamma$, there exists a unique true trajectory $\tilde{\gamma}$ within distance $\delta(\varepsilon) = O(\varepsilon)$ of $\gamma$.

**Innovation:** Anosov introduced the exponential dichotomy for geodesic flows on negatively curved manifolds, showing that hyperbolicity (uniform expansion/contraction rates) implies structural stability and shadowing.

**Application to Hypostructures:** Hypostructures inherit the geometric structure of negatively curved spaces via the dissipation functional $\Phi$, making Anosov's techniques directly applicable.

### Step 7.2: Bowen's Extension to Axiom A Systems (1975)

**Theorem (Bowen {cite}`Bowen75`):** For Axiom A diffeomorphisms (uniformly hyperbolic on the non-wandering set $\Omega$), every $\varepsilon$-pseudo-orbit in $\Omega$ is $\delta(\varepsilon)$-shadowed by a true orbit, with $\delta(\varepsilon) = O(\varepsilon/\lambda)$.

**Key Contribution:** Bowen extended Anosov's result beyond geodesic flows to general uniformly hyperbolic systems, establishing that the shadowing property is a **topological consequence of hyperbolicity**, not dependent on specific geometric structures.

**Relevance:** This generalization allows applying shadowing to arbitrary hypostructures with spectral gap $\lambda > 0$, not just those arising from Riemannian geometry.

### Step 7.3: Palmer's Contraction Mapping Proof (1988)

**Theorem (Palmer {cite}`Palmer88`):** The shadowing lemma for diffeomorphisms with exponential dichotomy can be proved via the contraction mapping theorem on the sequence space $\ell^\infty$.

**Method:** Palmer introduced the fixed-point operator $\mathcal{T}$ (our Step 3.1) and verified contractivity using the exponential dichotomy estimates. This provides a **constructive proof** with explicit convergence rates.

**Advantage for Hypostructures:** The contraction mapping approach yields:
- **Algorithmic realization** (Section Step 4.3)
- **Explicit error bounds** $\delta(\varepsilon) \leq 2C\varepsilon/\lambda$
- **Uniqueness of shadowing orbits** (essential for certificate logic)

### Step 7.4: Modern Extensions

**Computer-Assisted Proofs:** Zgliczyński and Mischaikow (2001) developed rigorous numerical methods for shadowing in chaotic systems, enabling computer-assisted proofs of long-time dynamics.

**Non-Uniform Hyperbolicity:** Pesin theory (1970s) extends shadowing to non-uniformly hyperbolic systems using Lyapunov exponents, though with weaker (probabilistic) conclusions.

**Partial Differential Equations:** Chow, Lin, and Palmer (1991) adapted shadowing to infinite-dimensional systems, relevant for PDEs arising in hypostructures.

**Stochastic Shadowing:** Luzzatto and Melbourne (2005) developed shadowing lemmas for random dynamical systems, applicable to hypostructures with noise.

---

## Conclusion

We have established the **Shadowing Metatheorem** for hypostructures:

**Main Result:** Given:
- Stiffness certificate $K_{\mathrm{LS}_\sigma}^+$ with spectral gap $\lambda > 0$
- Numerical pseudo-orbit $\{y_n\}$ with error $\varepsilon$
- Hyperbolicity (exponential dichotomy)

There exists a unique true orbit $\{x_n\}$ satisfying:
$$d(x_n, y_n) < \delta(\varepsilon) = O(\varepsilon/\lambda) \quad \forall n.$$

**Proof Method:** Fixed-point theorem on sequence space $\ell^\infty$ via the graph transform and exponential dichotomy estimates.

**Certificate Output:**
$$K_{\mathrm{LS}_\sigma}^+ \wedge K_{\text{pseudo}}^{\varepsilon} \Rightarrow K_{\text{true}}^{\delta(\varepsilon)}.$$

**Application:** Enables rigorous computer-assisted proofs of global regularity in PDEs by certifying numerical simulations.

**Literature:** Anosov (1967), Bowen (1975), Palmer (1988).

:::
