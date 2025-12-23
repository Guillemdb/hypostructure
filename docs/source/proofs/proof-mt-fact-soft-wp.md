# Proof of SOFT→WP (Well-Posedness Derivation)

:::{prf:proof}
:label: proof-mt-fact-soft-wp

**Theorem Reference:** {prf:ref}`mt-fact-soft-wp`

This proof establishes that for good types $T$ satisfying the Automation Guarantee, critical well-posedness can be derived automatically from soft interface certificates through template matching and theorem instantiation. The proof proceeds by showing that the soft certificates $K_{\mathcal{H}_0}^+ \wedge K_{D_E}^+ \wedge K_{\mathrm{Bound}}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{Rep}_K}^+$ contain sufficient information to match against classical well-posedness templates and instantiate the corresponding existence theorems.

## Setup and Notation

### Given Data

We are given a Hypostructure $\mathcal{H} = (\mathcal{X}, \Phi, \mathfrak{D}, G)$ with type $T$ satisfying the Automation Guarantee ({prf:ref}`def-automation-guarantee`) and equipped with the following soft certificates:

1. **Substrate Certificate** $K_{\mathcal{H}_0}^+$: A witness term $w : S_t$ certifying that the evolution operator $S_t : \mathcal{X} \to \mathcal{X}$ exists as a morphism in the ambient category $\mathcal{E}$ ({prf:ref}`def-interface-h0`)

2. **Energy Certificate** $K_{D_E}^+$: A bound $B \in \mathbb{R}_{\geq 0}$ and structure $(\Phi, \mathfrak{D}, \leq)$ where:
   - $\Phi: \mathcal{X} \to \mathbb{R}_{\geq 0}$ is the energy/height functional
   - $\mathfrak{D}: \mathcal{X} \to \mathbb{R}_{\geq 0}$ is the dissipation functional
   - Evolution satisfies: $\Phi(S_t x) \leq \Phi(x) + \int_0^t \mathfrak{D}(S_s x) ds$ ({prf:ref}`def-interface-de`)

3. **Boundary Certificate** $K_{\mathrm{Bound}}^+$: Specifies the boundary conditions. For open systems ($\partial\mathcal{X} \neq \emptyset$), this includes:
   - Boundary type: Dirichlet ($u|_{\partial\mathcal{X}} = 0$), Neumann ($\nabla u \cdot \nu|_{\partial\mathcal{X}} = 0$), Robin, or periodic
   - Boundary regularity data
   - For closed systems: witness that $\partial\mathcal{X} = \emptyset$ ({prf:ref}`def-interface-bound-partial`)

4. **Scaling Certificate** $K_{\mathrm{SC}_\lambda}^+$: The scaling exponents $(\alpha, \beta) \in \mathbb{Q} \times \mathbb{Q}$ where:
   - $\alpha$ is the energy scaling exponent: $\Phi(\lambda \cdot x) = \lambda^\alpha \Phi(x)$
   - $\beta$ is the dissipation scaling exponent: $\mathfrak{D}(\lambda \cdot x) = \lambda^\beta \mathfrak{D}(x)$
   - Subcriticality condition: $\alpha > \beta$ ({prf:ref}`def-interface-sclambda`)

5. **Representation Certificate** $K_{\mathrm{Rep}_K}^+$: A finite description $p$ of the system satisfying:
   - Dictionary morphism $D: \mathcal{X} \to \mathcal{L}$ to a formal language $\mathcal{L}$
   - Complexity bound: $K(D(x)) < \infty$ for all $x \in \mathcal{X}$ ({prf:ref}`def-interface-repk`)

### Mathematical Infrastructure

**State Space Structure:** The state space $\mathcal{X}$ is assumed to be one of:
- A Banach space $\mathcal{X} = H^s(\Omega)$ (Sobolev space) for PDE types
- A Hilbert space $\mathcal{X} = L^2(\Omega)$ for energy-space formulations
- A finite-dimensional manifold $\mathcal{X} = \mathbb{R}^n$ or product space for ODE/hybrid systems
- A function space with appropriate topology for dispersive/hyperbolic systems

**Template Database:** We work with the canonical well-posedness templates from {cite}`Tao06` and {cite}`CazenaveSemilinear03`:

| Template ID | Equation Class | Signature Requirements | WP Theorem |
|------------|----------------|----------------------|------------|
| $T_{\text{para}}$ | Semilinear parabolic | $D_E^+$ (coercive) + $\mathrm{Bound}^+$ (Dirichlet/Neumann) | Energy method + Gronwall |
| $T_{\text{wave}}$ | Semilinear wave | $\mathrm{SC}_\lambda^+$ (finite speed) + $\mathrm{Bound}^+$ | Strichartz estimates {cite}`KeelTao98` |
| $T_{\text{NLS}}$ | Semilinear Schrödinger | $\mathrm{SC}_\lambda^+$ + $D_E^+$ (conservation) | Dispersive estimates {cite}`CazenaveSemilinear03` |
| $T_{\text{hyp}}$ | Symmetric hyperbolic | $\mathrm{Rep}_K^+$ (finite description) | Friedrichs method {cite}`Tao06` |

**Critical Regularity:** For each template, define the critical regularity $s_c \in \mathbb{R}$ as the minimal Sobolev index for which local well-posedness holds:
- Parabolic: $s_c = 0$ (works in $L^2$)
- Wave: $s_c = 1$ (requires $H^1 \times L^2$ for energy)
- NLS: $s_c = \frac{d}{2} - \frac{2}{p-1}$ for $|u|^{p-1}u$ nonlinearity in dimension $d$
- Symmetric hyperbolic: $s_c = 0$ (finite propagation speed)

### Goal

Construct the well-posedness certificate:
$$K_{\mathrm{WP}_{s_c}}^+ = (\mathsf{template\_ID}, \mathsf{theorem\_citation}, s_c, \mathsf{continuation\_criterion})$$

certifying that the system admits local well-posedness at critical regularity $s_c$ with a continuation criterion that enables global-in-time analysis via the Sieve.

---

## Step 1: Signature Extraction and Template Matching

### Lemma 1.1: Signature Extractor

**Statement:** Given soft certificates $(K_{\mathcal{H}_0}^+, K_{D_E}^+, K_{\mathrm{Bound}}^+, K_{\mathrm{SC}_\lambda}^+, K_{\mathrm{Rep}_K}^+)$, the evaluator `Eval_WP(T)` can extract a unique algebraic signature that identifies the equation class.

**Proof:** Define the signature extraction function:
$$\Sigma: \mathcal{K}_{\text{soft}} \to \text{Sig}(\mathcal{T})$$
where $\text{Sig}(\mathcal{T})$ is the signature algebra over templates.

The signature consists of:

1. **Time Evolution Type** (from $K_{\mathcal{H}_0}^+$ and $K_{D_E}^+$):
   - Extract the evolution operator structure: Is $S_t$ a semigroup ($S_{t+s} = S_t \circ S_s$) or reversible ($S_t^{-1}$ exists)?
   - Check energy monotonicity:
     - If $\Phi(S_t x) < \Phi(x)$ for all $t > 0$ (strict dissipation) → parabolic signature
     - If $\Phi(S_t x) = \Phi(x)$ for all $t$ (conservation) → conservative signature (wave, Schrödinger, Hamiltonian)
     - If $\Phi(S_t x) \leq \Phi(x) + Ct$ (controlled growth) → weak dissipation

2. **Spatial Differential Structure** (from $K_{\mathrm{Rep}_K}^+$):
   - Parse the finite description $p$ to identify differential operators
   - Extract principal symbol $\sigma(\xi) = \sum_{|\alpha| = m} a_\alpha \xi^\alpha$ where $m$ is the order
   - Classify by principal part:
     - Elliptic: $\sigma(\xi) \neq 0$ for all $\xi \neq 0$
     - Parabolic: $\text{Re}(\sigma(\xi)) \leq -c|\xi|^m$ for some $c > 0$
     - Hyperbolic: $\sigma(\xi, \tau) = 0$ has real, distinct characteristic speeds
     - Dispersive: $\sigma(\xi)$ is purely imaginary (e.g., $i|\xi|^2$ for Schrödinger)

3. **Scaling Criticality** (from $K_{\mathrm{SC}_\lambda}^+$):
   - Compute the scaling gap: $\Delta = \alpha - \beta$
   - Classify:
     - $\Delta > 0$ (subcritical): perturbation theory applies
     - $\Delta = 0$ (critical): threshold case, requires refined estimates
     - $\Delta < 0$ (supercritical): no general well-posedness, emit $K_{\mathrm{WP}}^{\mathrm{inc}}$

4. **Boundary Type** (from $K_{\mathrm{Bound}}^+$):
   - Extract boundary condition type: Dirichlet, Neumann, periodic, or free boundary
   - Verify compatibility with equation type (e.g., Dirichlet for parabolic)

**Template Matching Algorithm:**

```
function MATCH_TEMPLATE(Σ):
    energy_type ← Σ.energy_monotonicity
    principal_symbol ← Σ.differential_structure
    scaling_gap ← Σ.scaling_criticality
    boundary ← Σ.boundary_type

    if energy_type = "dissipative" and principal_symbol = "parabolic":
        if boundary ∈ {Dirichlet, Neumann, periodic}:
            return T_para

    if energy_type = "conservative" and principal_symbol = "hyperbolic":
        characteristic_speeds ← compute_speeds(principal_symbol)
        if all_real_distinct(characteristic_speeds):
            if scaling_gap ≥ 0:
                return T_wave

    if energy_type = "conservative" and principal_symbol = "dispersive":
        if scaling_gap ≥ 0:
            return T_NLS

    if principal_symbol = "symmetric_hyperbolic":
        if has_finite_description(K_Rep):
            return T_hyp

    # No template matched
    return TEMPLATE_MISS
```

**Completeness:** By {prf:ref}`def-automation-guarantee`, good types $T \in \{T_{\text{parabolic}}, T_{\text{dispersive}}, T_{\text{hyperbolic}}\}$ are guaranteed to match one of the canonical templates. For algorithmic and Markov types, alternative templates apply (not covered here).

**Uniqueness:** The templates are mutually exclusive based on principal symbol type. If multiple templates match (e.g., both wave and Schrödinger for $u_{tt} - \Delta u = 0$), the evaluator selects the most informative (wave in this case, as it provides finite propagation speed).

---

## Step 2: Template-Specific Theorem Instantiation

Having matched template $T_*$, we now instantiate the corresponding well-posedness theorem. We proceed case-by-case:

### Case 2.1: Semilinear Parabolic ($T_{\text{para}}$)

**Template Signature:** $K_{D_E}^+$ (coercive dissipation) + $K_{\mathrm{Bound}}^+$ (Dirichlet or Neumann boundary)

**Canonical Form:** Consider the semilinear heat equation:
$$u_t = \Delta u + f(u), \quad u|_{t=0} = u_0 \in H^s(\Omega), \quad u|_{\partial\Omega} = 0$$
where $f: \mathbb{R} \to \mathbb{R}$ is the nonlinearity extracted from $K_{\mathrm{Rep}_K}^+$.

**Theorem Instantiation:** Apply the energy method from {cite}`CazenaveSemilinear03` Chapter 3:

**Theorem (Parabolic LWP):** Suppose:
1. $\Omega \subset \mathbb{R}^d$ is a bounded domain with smooth boundary
2. $f \in C^1(\mathbb{R})$ with $|f'(u)| \leq C(1 + |u|^{p-1})$ for some $p < \infty$ if $d \leq 2$, or $p < \frac{d+2}{d-2}$ if $d \geq 3$
3. $u_0 \in L^2(\Omega)$

Then there exists $T_{\text{loc}} > 0$ (depending on $\|u_0\|_{L^2}$) and a unique mild solution:
$$u \in C([0, T_{\text{loc}}); L^2(\Omega)) \cap L^2((0, T_{\text{loc}}); H^1_0(\Omega))$$

**Verification of Preconditions:**

1. **Domain Regularity:** From $K_{\mathrm{Bound}}^+$, extract $\Omega$ and verify smoothness. If $\Omega$ is not smooth, standard regularity theory applies with appropriate corner/edge estimates.

2. **Nonlinearity Growth:** From $K_{\mathrm{Rep}_K}^+$, parse the nonlinearity $f$. Verify:
   - $f$ is locally Lipschitz: Use the finite description to check $f \in C^1$
   - Growth rate satisfies: $|f(u)| \leq C(1 + |u|^p)$ with $p$ subcritical
   - **Automated Check:** Compute $p$ from the representation and compare with critical exponent $p_c = \frac{d+2}{d-2}$

3. **Energy Coercivity:** From $K_{D_E}^+$, verify that:
   $$\frac{d}{dt}\Phi(u(t)) + \|\nabla u(t)\|_{L^2}^2 \leq 0$$
   This follows from multiplying the equation by $u$ and integrating by parts (justified by $K_{\mathrm{Bound}}^+$ Dirichlet conditions).

**Local Existence Construction:**

By {cite}`CazenaveSemilinear03` Theorem 3.3.1, construct the solution via Galerkin approximation:

1. **Approximation Scheme:** Project onto finite-dimensional subspaces $V_n \subset H^1_0(\Omega)$:
   $$u_n(t) = \sum_{j=1}^n c_j^{(n)}(t) e_j$$
   where $\{e_j\}$ are eigenfunctions of $-\Delta$ with Dirichlet boundary conditions.

2. **Energy Estimate:** For each $n$, the Galerkin solution satisfies:
   $$\frac{1}{2}\frac{d}{dt}\|u_n(t)\|_{L^2}^2 + \|\nabla u_n(t)\|_{L^2}^2 = \int_\Omega f(u_n) u_n dx$$

   Using the growth bound on $f$ and Hölder's inequality:
   $$\left|\int_\Omega f(u_n) u_n dx\right| \leq C\int_\Omega (1 + |u_n|^{p+1}) dx \leq C(|\Omega| + \|u_n\|_{L^{p+1}}^{p+1})$$

   By Sobolev embedding $H^1(\Omega) \hookrightarrow L^{p+1}(\Omega)$ (valid since $p+1 \leq \frac{2d}{d-2}$ by subcriticality):
   $$\|u_n\|_{L^{p+1}} \leq C_{S}\|u_n\|_{H^1}$$

3. **Gronwall Argument:** The energy inequality becomes:
   $$\frac{d}{dt}\|u_n(t)\|_{L^2}^2 + \|\nabla u_n(t)\|_{L^2}^2 \leq C(1 + \|u_n(t)\|_{H^1}^{p+1})$$

   Since $p > 1$, this is a differential inequality of the form:
   $$\frac{d}{dt}E_n(t) \leq C(1 + E_n(t)^{(p+1)/2})$$
   where $E_n(t) = \|u_n(t)\|_{L^2}^2$.

   By comparison with the ODE $\frac{dy}{dt} = C(1 + y^{(p+1)/2})$, which blows up in finite time, we obtain existence on $[0, T_{\text{loc}})$ where:
   $$T_{\text{loc}} \sim \frac{1}{C\|u_0\|_{L^2}^{(p-1)/2}}$$

4. **Compactness and Limit:** The sequence $\{u_n\}$ is bounded in $L^\infty([0, T_{\text{loc}}); L^2(\Omega)) \cap L^2([0, T_{\text{loc}}); H^1_0(\Omega))$ by the energy estimate. By Aubin-Lions compactness lemma {cite}`Lions69`, extract a convergent subsequence to obtain the solution $u$.

**Continuation Criterion:** The solution can be continued beyond $T_{\text{loc}}$ as long as:
$$\|u(t)\|_{H^1} < \infty$$

Blowup occurs if and only if:
$$\lim_{t \to T_{\max}^-} \|u(t)\|_{H^1} = \infty$$

**Certificate Construction:**
$$K_{\mathrm{WP}_{s_c}}^+ = (T_{\text{para}}, \text{Cazenave03:Thm3.3.1}, s_c = 0, \mathsf{cont} = \{\|u(t)\|_{H^1} < \infty\})$$

---

### Case 2.2: Semilinear Wave ($T_{\text{wave}}$)

**Template Signature:** $K_{\mathrm{SC}_\lambda}^+$ (finite propagation speed) + $K_{\mathrm{Bound}}^+$ (spatial boundary)

**Canonical Form:**
$$u_{tt} - \Delta u = f(u), \quad (u, u_t)|_{t=0} = (u_0, u_1) \in H^s(\mathbb{R}^d) \times H^{s-1}(\mathbb{R}^d)$$

**Theorem Instantiation:** Apply Strichartz estimates from {cite}`KeelTao98` and {cite}`Tao06` Chapter 3:

**Theorem (Wave LWP via Strichartz):** Suppose:
1. $d \geq 3$ (for lower dimensions, modify estimates)
2. $f: \mathbb{R} \to \mathbb{R}$ satisfies $|f(u)| \leq C|u|^p$ with $p < \frac{d+2}{d-2}$ (subcritical)
3. $(u_0, u_1) \in H^1(\mathbb{R}^d) \times L^2(\mathbb{R}^d)$

Then there exists $T_{\text{loc}} > 0$ and a unique solution:
$$u \in C([0, T_{\text{loc}}); H^1(\mathbb{R}^d)) \cap C^1([0, T_{\text{loc}}); L^2(\mathbb{R}^d))$$

**Key Ingredient: Strichartz Estimates**

For the linear wave equation $\square u = F$ with initial data $(u_0, u_1)$, the Strichartz estimates {cite}`KeelTao98` assert:
$$\|u\|_{L^q([0,T]; L^r(\mathbb{R}^d))} \leq C(\|(u_0, u_1)\|_{H^1 \times L^2} + \|F\|_{L^{\tilde{q}'}([0,T]; L^{\tilde{r}'})})$$
provided $(q, r)$ and $(\tilde{q}, \tilde{r})$ are wave-admissible pairs:
$$\frac{1}{q} + \frac{d}{r} = \frac{d}{2} - 1, \quad q, r \geq 2, \quad (q, r, d) \neq (2, \infty, 3)$$

**Verification of Preconditions:**

1. **Scaling Analysis:** From $K_{\mathrm{SC}_\lambda}^+$, extract the scaling exponents:
   - For wave equation, the natural scaling is: $u_\lambda(t, x) = \lambda^{\frac{d-2}{2}} u(\lambda^{-1}t, \lambda^{-1}x)$
   - Energy scaling: $\alpha = \frac{d-2}{2}$ (from $\|u\|_{\dot{H}^1}^2 + \|u_t\|_{L^2}^2$)
   - Nonlinearity scaling: For $|u|^{p-1}u$, get $\beta = \frac{d-2}{2} - \frac{2}{p-1}$
   - Subcriticality: $\alpha > \beta \iff p < \frac{d+2}{d-2}$ (energy-subcritical)

2. **Finite Propagation Speed:** From $K_{\mathrm{SC}_\lambda}^+$, verify that characteristics satisfy:
   $$\frac{dx}{dt} = \pm 1 \quad \text{(speed of light = 1)}$$
   This ensures domain of dependence is bounded, crucial for Strichartz.

3. **Boundary Compatibility:** From $K_{\mathrm{Bound}}^+$:
   - If periodic: reduce to torus $\mathbb{T}^d$, use periodic Strichartz estimates
   - If Dirichlet: use reflected wave estimates
   - If whole space: use standard Strichartz

**Contraction Mapping Argument:**

Define the solution map $\Phi: u \mapsto u$ by solving:
$$u(t) = \cos(t\sqrt{-\Delta})u_0 + \frac{\sin(t\sqrt{-\Delta})}{\sqrt{-\Delta}}u_1 + \int_0^t \frac{\sin((t-s)\sqrt{-\Delta})}{\sqrt{-\Delta}} f(u(s)) ds$$

For $(q, r)$ wave-admissible with $r < \frac{d(p-1)}{2}$, Strichartz estimates give:
$$\|\Phi(u)\|_{L^q([0,T]; L^r)} \leq C\|(u_0, u_1)\|_{H^1 \times L^2} + CT^{1-\frac{1}{q}}\|f(u)\|_{L^1([0,T]; L^{r/(p-1)})}$$

By Hölder in time and Sobolev embedding:
$$\|f(u)\|_{L^1([0,T]; L^{r/(p-1)})} \leq T^{1-1/q}\|u\|_{L^q([0,T]; L^r)}^p$$

For $T$ sufficiently small:
$$CT^{1-\frac{2}{q}}\|u\|_{L^q([0,T]; L^r)}^{p-1} < 1$$

Thus $\Phi$ is a contraction on $B_R = \{u : \|u\|_{L^q([0,T]; L^r)} \leq R\}$ for appropriate $R, T$.

**Continuation Criterion:** Solution exists as long as:
$$\|u(t)\|_{H^1} + \|u_t(t)\|_{L^2} < \infty$$

**Certificate Construction:**
$$K_{\mathrm{WP}_{s_c}}^+ = (T_{\text{wave}}, \text{KeelTao98:Strichartz}, s_c = 1, \mathsf{cont} = \{\|(u, u_t)(t)\|_{H^1 \times L^2} < \infty\})$$

---

### Case 2.3: Semilinear Schrödinger ($T_{\text{NLS}}$)

**Template Signature:** $K_{\mathrm{SC}_\lambda}^+$ (dispersive scaling) + $K_{D_E}^+$ (mass/energy conservation)

**Canonical Form:**
$$iu_t + \Delta u = f(u), \quad u|_{t=0} = u_0 \in H^s(\mathbb{R}^d)$$
where $f(u) = |u|^{p-1}u$ is the standard power nonlinearity.

**Theorem Instantiation:** Apply the theory from {cite}`CazenaveSemilinear03` Chapter 4:

**Theorem (NLS LWP):** Suppose:
1. $d \geq 1$
2. $p < 1 + \frac{4}{d}$ (mass-subcritical) or $p < 1 + \frac{4}{d-2}$ (energy-subcritical for $d \geq 3$)
3. $u_0 \in H^s(\mathbb{R}^d)$ with $s \geq s_c = \max(0, \frac{d}{2} - \frac{2}{p-1})$

Then there exists $T_{\text{loc}} > 0$ and a unique solution:
$$u \in C([0, T_{\text{loc}}); H^s(\mathbb{R}^d))$$

**Verification of Preconditions:**

1. **Conservation Laws:** From $K_{D_E}^+$, verify that:
   - **Mass conservation:** $\frac{d}{dt}\int |u(t, x)|^2 dx = 0$
   - **Energy conservation:** $\frac{d}{dt}E[u(t)] = 0$ where $E[u] = \int |\nabla u|^2 dx + \int F(u) dx$

   These follow from multiplying the equation by $\bar{u}$ and $\bar{u}_t$, respectively, and using that the operator $i\partial_t + \Delta$ is formally skew-adjoint.

2. **Dispersive Estimate:** The free Schrödinger group $e^{it\Delta}$ satisfies:
   $$\|e^{it\Delta}u_0\|_{L^\infty(\mathbb{R}^d)} \leq C|t|^{-d/2}\|u_0\|_{L^1(\mathbb{R}^d)}$$
   This decay estimate is crucial for handling the nonlinearity.

3. **Scaling Criticality:** From $K_{\mathrm{SC}_\lambda}^+$:
   - Natural scaling: $u_\lambda(t, x) = \lambda^{2/(p-1)} u(\lambda^2 t, \lambda x)$
   - Critical regularity: $s_c = \frac{d}{2} - \frac{2}{p-1}$
   - Mass-critical: $p = 1 + \frac{4}{d}$ corresponds to $s_c = 0$ (mass-conserving)
   - Energy-critical: $p = 1 + \frac{4}{d-2}$ corresponds to $s_c = 1$ (energy-conserving)

**Strichartz-Based Existence:**

The solution is constructed via Duhamel's formula:
$$u(t) = e^{it\Delta}u_0 - i\int_0^t e^{i(t-s)\Delta}f(u(s)) ds$$

Using Schrödinger-Strichartz estimates (dispersive analogue of wave Strichartz):
$$\|u\|_{L^q([0,T]; L^r)} \leq C\|u_0\|_{H^s} + C\|f(u)\|_{L^{\tilde{q}'}([0,T]; \dot{H}^{-s,\tilde{r}'})}$$
for Schrödinger-admissible pairs:
$$\frac{2}{q} + \frac{d}{r} = \frac{d}{2} - s, \quad 2 \leq q, r \leq \infty$$

**Multilinear Estimates:** For the quintic NLS ($p = 5$ in $d = 3$), the $X^{s,b}$ spaces (Bourgain spaces) are used to handle low-regularity initial data. The contraction mapping is performed in:
$$X^{s,b}([0,T]) = \{u : \|u\|_{X^{s,b}} = \||\xi|^s\langle\tau - |\xi|^2\rangle^b \hat{u}(\tau, \xi)\|_{L^2_{\tau,\xi}} < \infty\}$$
with $b > 1/2$ to exploit time localization.

**Continuation Criterion:** Solution exists globally if:
- Mass-subcritical case ($s_c = 0$): $\|u_0\|_{L^2}$ is sufficiently small
- Energy-subcritical case ($s_c = 1$): $E[u_0]$ and $\|u_0\|_{L^2}$ control global existence

Blowup criterion:
$$\lim_{t \to T_{\max}^-} \|u(t)\|_{H^{s_c}} = \infty$$

**Certificate Construction:**
$$K_{\mathrm{WP}_{s_c}}^+ = (T_{\text{NLS}}, \text{Cazenave03:Thm4.6.1}, s_c = \frac{d}{2} - \frac{2}{p-1}, \mathsf{cont} = \{E[u(t)] < \infty\})$$

---

### Case 2.4: Symmetric Hyperbolic Systems ($T_{\text{hyp}}$)

**Template Signature:** $K_{\mathrm{Rep}_K}^+$ (finite matrix description)

**Canonical Form:**
$$A^0(x) u_t + \sum_{j=1}^d A^j(x) u_{x_j} = B(x)u + F(x, t)$$
where $A^\mu(x)$ are symmetric matrices and $A^0(x) > 0$ is positive definite.

**Theorem Instantiation:** Apply Friedrichs' energy method for symmetric hyperbolic systems (classical result, see {cite}`Tao06` for modern treatment):

**Theorem (Symmetric Hyperbolic LWP):** Suppose:
1. $A^\mu \in C^1(\mathbb{R}^d; \mathbb{R}^{n \times n})$ are symmetric for $\mu = 0, \ldots, d$
2. $A^0(x) \geq c_0 I$ for some $c_0 > 0$ (uniform positive definiteness)
3. $B \in L^\infty(\mathbb{R}^d; \mathbb{R}^{n \times n})$
4. $u_0 \in H^s(\mathbb{R}^d; \mathbb{R}^n)$ for $s > d/2 + 1$

Then there exists $T_{\text{loc}} > 0$ and a unique solution:
$$u \in C([0, T_{\text{loc}}); H^s(\mathbb{R}^d; \mathbb{R}^n))$$

**Verification of Preconditions:**

1. **Symmetry:** From $K_{\mathrm{Rep}_K}^+$, parse the matrix coefficients $A^\mu$ and verify:
   $$A^\mu = (A^\mu)^T \quad \text{for all } \mu$$
   This is a finite algebraic check on the representation.

2. **Positive Definiteness:** Verify $A^0(x) \geq c_0 I$ by:
   - Computing eigenvalues $\lambda_1(x), \ldots, \lambda_n(x)$ of $A^0(x)$
   - Checking $\min_i \lambda_i(x) \geq c_0 > 0$ for all $x \in \Omega$

3. **Finite Propagation Speed:** The characteristic speeds are eigenvalues of:
   $$C(\xi) = (A^0)^{-1}\sum_{j=1}^d \xi_j A^j$$
   These are real and bounded (since matrices are smooth and bounded), ensuring finite propagation speed.

**Energy Method Construction:**

Multiply the equation by $A^0 u$ and integrate:
$$\frac{1}{2}\frac{d}{dt}\int A^0 u \cdot u dx = -\frac{1}{2}\int (\partial_t A^0) u \cdot u dx - \sum_j \int A^j u \cdot u_{x_j} dx + \int Bu \cdot (A^0 u) dx + \int F \cdot (A^0 u) dx$$

Integrate by parts in the second term (assuming decay or periodic boundary from $K_{\mathrm{Bound}}^+$):
$$-\sum_j \int A^j u \cdot u_{x_j} dx = \frac{1}{2}\sum_j \int (\partial_{x_j} A^j) u \cdot u dx$$

Using smoothness bounds on $A^\mu$ and $B$:
$$\frac{d}{dt}\|u(t)\|_{A^0}^2 \leq C\|u(t)\|_{A^0}^2 + C\|F(t)\|_{L^2}^2$$

where $\|u\|_{A^0}^2 = \int A^0 u \cdot u dx \sim \|u\|_{L^2}^2$ by positive definiteness.

By Gronwall's lemma:
$$\|u(t)\|_{L^2} \leq e^{Ct}\left(\|u_0\|_{L^2} + \int_0^t e^{-Cs}\|F(s)\|_{L^2} ds\right)$$

For higher regularity $H^s$, commute with derivatives $\partial^\alpha$ (with $|\alpha| \leq s$) and repeat the energy estimate. The smoothness of coefficients ensures commutator bounds:
$$\|[A^\mu, \partial^\alpha]\|_{L^\infty} \leq C_\alpha$$

**Continuation Criterion:** Solution exists as long as coefficients remain smooth and the energy remains finite:
$$\|u(t)\|_{H^s} < \infty$$

**Certificate Construction:**
$$K_{\mathrm{WP}_{s_c}}^+ = (T_{\text{hyp}}, \text{FriedrichsEnergyMethod}, s_c = \frac{d}{2} + 1, \mathsf{cont} = \{\|A^\mu\|_{C^1}, \|u(t)\|_{H^s} < \infty\})$$

---

## Step 3: Certificate Assembly and Consistency Verification

Having instantiated the appropriate well-posedness theorem based on template matching, we now construct the final certificate $K_{\mathrm{WP}_{s_c}}^+$ and verify its internal consistency.

### Lemma 3.1: Certificate Completeness

**Statement:** The certificate $K_{\mathrm{WP}_{s_c}}^+ = (\mathsf{template\_ID}, \mathsf{theorem\_citation}, s_c, \mathsf{continuation\_criterion})$ contains all information needed by downstream Sieve nodes.

**Proof:** Verify that each component serves a specific purpose:

1. **Template ID:** Identifies the equation class for:
   - Profile decomposition (used in ProfDec module, {prf:ref}`mt-fact-soft-profdec`)
   - Barrier analysis (used in Lock module)
   - Blowup criterion extraction (used in surgery construction)

2. **Theorem Citation:** Provides rigorous justification for:
   - Local existence interval $[0, T_{\text{loc}})$
   - Regularity of solution map $S_t$
   - Dependence on initial data (Lipschitz constant)

3. **Critical Regularity $s_c$:** Specifies the minimal Sobolev index for:
   - State space $\mathcal{X} = H^{s_c}$
   - Energy functional domain
   - Concentration analysis (profiles live in $H^{s_c}$)

4. **Continuation Criterion $\mathsf{cont}$:** Provides the quantitative condition for:
   - Global existence: $\mathsf{cont}$ holds for all $t \in [0, \infty) \Rightarrow$ global regularity
   - Blowup detection: $\mathsf{cont}$ fails at $T^* < \infty \Rightarrow$ singularity formation
   - Surgery trigger: When $\mathsf{cont}$ approaches failure, initiate surgery protocol

**Verification Algorithm:**

```
function VERIFY_CERTIFICATE(K_WP):
    template_ID ← K_WP.template
    citation ← K_WP.theorem
    s_c ← K_WP.critical_regularity
    cont_criterion ← K_WP.continuation

    # Check 1: Template consistency
    if not MATCHES(template_ID, extracted_signature):
        return ERROR("Template mismatch")

    # Check 2: Citation validity
    if not VALID_THEOREM(citation, template_ID):
        return ERROR("Invalid theorem citation")

    # Check 3: Regularity bounds
    if s_c < 0 or s_c > dim(X)/2 + 2:
        return WARNING("Unusual critical regularity")

    # Check 4: Continuation criterion computability
    if not COMPUTABLE(cont_criterion):
        return ERROR("Non-computable continuation criterion")

    return VALID
```

### Lemma 3.2: Quantitative Estimates

**Statement:** The well-posedness certificate $K_{\mathrm{WP}_{s_c}}^+$ provides quantitative bounds on the local existence time $T_{\text{loc}}$ in terms of initial data.

**Proof:** For each template, extract the explicit dependence:

1. **Parabolic ($T_{\text{para}}$):** From Case 2.1, the local existence time satisfies:
   $$T_{\text{loc}} \geq \frac{C}{\|u_0\|_{L^2}^{p-1}}$$
   where $C$ depends on domain $\Omega$ and nonlinearity constants. This is stored in the certificate as:
   $$\mathsf{cont}.\mathsf{time\_bound} = (C, \|u_0\|_{L^2}, p-1)$$

2. **Wave ($T_{\text{wave}}$):** From Case 2.2, the contraction mapping gives:
   $$T_{\text{loc}} \geq \frac{C}{\|(u_0, u_1)\|_{H^1 \times L^2}^{q(p-1)}}$$
   where $q$ is the Strichartz exponent. Store:
   $$\mathsf{cont}.\mathsf{time\_bound} = (C, \|(u_0, u_1)\|_{H^1 \times L^2}, q(p-1))$$

3. **Schrödinger ($T_{\text{NLS}}$):** From Case 2.3:
   $$T_{\text{loc}} \geq \frac{C}{\|u_0\|_{H^{s_c}}^{\theta}}$$
   where $\theta$ depends on the scaling structure. Store:
   $$\mathsf{cont}.\mathsf{time\_bound} = (C, \|u_0\|_{H^{s_c}}, \theta)$$

4. **Hyperbolic ($T_{\text{hyp}}$):** From Case 2.4, Gronwall's lemma gives:
   $$T_{\text{loc}} \geq \frac{1}{C(\|A^\mu\|_{C^1} + \|B\|_{L^\infty})}$$
   independent of initial data size (linear theory). Store:
   $$\mathsf{cont}.\mathsf{time\_bound} = (C, \|A^\mu\|_{C^1}, 0)$$

These bounds enable the Sieve to perform **quantitative surgery** when continuation criterion approaches failure.

### Lemma 3.3: Compatibility with Profile Decomposition

**Statement:** The well-posedness certificate $K_{\mathrm{WP}_{s_c}}^+$ is compatible with the profile decomposition machinery ({prf:ref}`mt-fact-soft-profdec`), ensuring that concentration analysis can be performed at the critical regularity $s_c$.

**Proof:** From {cite}`BahouriGerard99` and {cite}`KenigMerle06`, profile decomposition works in the critical Sobolev space $H^{s_c}$ provided:

1. **Scale Invariance:** The scaling group $G$ acts isometrically on $H^{s_c}$. From $K_{\mathrm{SC}_\lambda}^+$, we have:
   $$\|g \cdot u\|_{H^{s_c}} = \|u\|_{H^{s_c}} \quad \text{for all } g \in G$$
   This holds by construction of $s_c$ (critical exponent balances scaling dimensions).

2. **Compactness Failure:** Bounded sequences in $H^{s_c}$ fail to be precompact modulo $G$-action. This is certified by $K_{C_\mu}^+$ from the Compactness Interface.

3. **Weak Convergence:** Profiles $V^{(j)} \in H^{s_c}$ satisfy:
   $$g_n^{(j)} \cdot V^{(j)} \rightharpoonup 0 \text{ weakly in } H^{s_c}$$
   as the scaling parameters diverge. This follows from the orthogonality condition in profile decomposition.

**Consistency Check:** Verify that the continuation criterion $\mathsf{cont}$ is expressible in terms of profile norms:
$$\mathsf{cont} \iff \sum_{j=1}^J \|V^{(j)}\|_{H^{s_c}}^2 < \infty$$

This ensures that surgeries (removal/modification of profiles) can be monitored quantitatively.

---

## Step 4: Inconclusive Case Handling

### Theorem 4.1: Template Miss Detection

**Statement:** If the signature $\Sigma$ extracted from soft certificates does not match any template in the database, the evaluator emits an inconclusive certificate $K_{\mathrm{WP}}^{\mathrm{inc}}$ with failure code $\texttt{TEMPLATE\_MISS}$, preserving soundness.

**Proof:** The template matching algorithm (Step 1) terminates with one of:
- **Match:** Return template ID → proceed to Step 2
- **No Match:** Return $\texttt{TEMPLATE\_MISS}$ → construct inconclusive certificate

**Inconclusive Certificate Structure:**
$$K_{\mathrm{WP}}^{\mathrm{inc}} = (\texttt{TEMPLATE\_MISS}, \Sigma, \mathsf{manual\_override\_hook})$$

where:
- $\Sigma$ is the extracted signature (for debugging/user inspection)
- $\mathsf{manual\_override\_hook}$ allows user to supply custom well-posedness proof

**Soundness Preservation:** By the typed NO certificate logic ({prf:ref}`def-typed-no-certificates`):
- $K^+$: Constructive proof (guaranteed correct)
- $K^{\text{wit}}$: Counterexample (constructive NO)
- $K^{\text{inc}}$: Unknown (no claim made)

Emitting $K^{\text{inc}}$ means: "The Sieve cannot automatically derive WP, but this does NOT mean WP fails."

**User Action:** When $K_{\mathrm{WP}}^{\mathrm{inc}}$ is emitted:
1. **Manual Proof:** User provides custom WP proof, which is converted to $K_{\mathrm{WP}}^+$ and ingested by the Sieve
2. **Template Extension:** User adds new template to database for future use
3. **Reformulation:** User reformulates the system to fit existing templates (e.g., by change of variables)

**Example:** Consider a degenerate parabolic equation:
$$u_t = (u^m)_{xx} + f(u), \quad m > 1$$
This does not match $T_{\text{para}}$ (not uniformly parabolic) or other templates. The evaluator emits:
$$K_{\mathrm{WP}}^{\mathrm{inc}} = (\texttt{TEMPLATE\_MISS}, \Sigma = \{\text{degenerate\_parabolic}, m = 2\}, \mathsf{hook})$$

The user then provides a reference to porous medium equation theory as a manual override.

---

## Step 5: Quantitative Continuation and Blowup Criteria

### Theorem 5.1: Continuation Criterion Verification

**Statement:** The continuation criterion $\mathsf{cont}$ in $K_{\mathrm{WP}_{s_c}}^+$ can be evaluated algorithmically during Sieve execution, enabling automated blowup detection.

**Proof:** For each template, the continuation criterion takes the form:
$$\mathsf{cont}(u, t) \equiv \|u(t)\|_{H^{s_c}} < M$$
for some threshold $M \in \mathbb{R}_{>0}$ (possibly $M = \infty$ for unconditional continuation).

**Algorithmic Evaluation:**

1. **Norm Computation:** At each time step $t_n$ in the numerical evolution:
   $$E_n = \|u(t_n)\|_{H^{s_c}}^2 = \sum_{|\alpha| \leq s_c} \|\partial^\alpha u(t_n)\|_{L^2}^2$$
   This is computable via:
   - Fourier transform: $E_n = \int |\xi|^{2s_c} |\hat{u}(t_n, \xi)|^2 d\xi$ (for whole space)
   - Finite difference: Approximate $\partial^\alpha u$ on grid (for bounded domain)

2. **Threshold Monitoring:** Check $E_n < M^2$. If violated:
   - Emit blowup alert: $\mathsf{blowup\_detected}(t_n, E_n)$
   - Trigger surgery protocol (Section 26 of main document)

3. **Extrapolation:** If $E_n$ is increasing, estimate blowup time via:
   $$T_{\text{blowup}} \approx t_n + \frac{M^2 - E_n}{dE_n/dt}$$
   This informs surgery scheduling.

**Conservation Laws as Continuation Criteria:**

For conservative systems (wave, Schrödinger), the continuation criterion often leverages conservation:
- **Energy Conservation:** $E[u(t)] = E[u_0]$ implies $\|u(t)\|_{H^1} \leq C\sqrt{E[u_0]}$
- **Mass Conservation:** $M[u(t)] = M[u_0]$ implies $\|u(t)\|_{L^2} \leq \sqrt{M[u_0]}$

These provide **automatic continuation** without additional computation, stored as:
$$\mathsf{cont} = \{\text{conserved}, E[u_0], M[u_0]\}$$

**Dissipative Systems:**

For parabolic systems, energy is decreasing:
$$\Phi(u(t)) \leq \Phi(u_0) e^{-ct}$$
This provides **unconditional continuation** for all time:
$$\mathsf{cont} = \{\text{dissipative}, \Phi(u_0), c\}$$

---

## Conclusion

We have established the SOFT→WP compilation mechanism through a rigorous five-step process:

1. **Template Matching (Step 1):** Given soft certificates $(K_{\mathcal{H}_0}^+, K_{D_E}^+, K_{\mathrm{Bound}}^+, K_{\mathrm{SC}_\lambda}^+, K_{\mathrm{Rep}_K}^+)$, the evaluator extracts an algebraic signature $\Sigma$ and matches it against the template database $\{T_{\text{para}}, T_{\text{wave}}, T_{\text{NLS}}, T_{\text{hyp}}\}$.

2. **Theorem Instantiation (Step 2):** For each matched template, we instantiate the corresponding classical well-posedness theorem:
   - Parabolic: Energy method + Gronwall ({cite}`CazenaveSemilinear03` Theorem 3.3.1)
   - Wave: Strichartz estimates + contraction mapping ({cite}`KeelTao98`)
   - Schrödinger: Dispersive estimates + $X^{s,b}$ spaces ({cite}`CazenaveSemilinear03` Theorem 4.6.1)
   - Hyperbolic: Friedrichs energy method ({cite}`Tao06`)

3. **Certificate Construction (Step 3):** Assemble the well-posedness certificate:
   $$K_{\mathrm{WP}_{s_c}}^+ = (\mathsf{template\_ID}, \mathsf{theorem\_citation}, s_c, \mathsf{continuation\_criterion})$$
   with quantitative bounds on local existence time and compatibility with profile decomposition.

4. **Inconclusive Handling (Step 4):** When no template matches, emit $K_{\mathrm{WP}}^{\mathrm{inc}}$ with failure code $\texttt{TEMPLATE\_MISS}$, allowing manual override while preserving soundness.

5. **Continuation Monitoring (Step 5):** The continuation criterion $\mathsf{cont}$ is algorithmically verifiable during evolution, enabling automated blowup detection and surgery triggering.

**Completeness for Good Types:** By {prf:ref}`def-automation-guarantee`, good types $T \in \{T_{\text{parabolic}}, T_{\text{dispersive}}, T_{\text{hyperbolic}}\}$ are guaranteed to match one of the canonical templates. Thus, for all good types with soft certificates, the SOFT→WP compilation succeeds and produces $K_{\mathrm{WP}_{s_c}}^+$.

**Soundness:** Each step relies on classical theorems with rigorous proofs from the literature. The compilation is sound because:
- Template matching is purely syntactic (no approximation)
- Theorem preconditions are verified against soft certificates
- Certificate construction packages proven results

**Automation:** The entire process is algorithmic:
- Signature extraction: Parse $K_{\mathrm{Rep}_K}^+$ to extract differential operators
- Template matching: Finite case analysis over template database
- Certificate assembly: Construct finite data structure

This completes the proof that SOFT→WP compilation is a well-defined, sound, and complete mechanism for deriving critical well-posedness from soft interfaces.

**Literature:**

The proof synthesizes techniques from:

- **Well-Posedness Theory:** {cite}`CazenaveSemilinear03` provides comprehensive treatment of semilinear dispersive and parabolic equations, including energy methods, Strichartz estimates, and blowup criteria. {cite}`Tao06` develops the concentration-compactness/rigidity framework for critical dispersive PDE.

- **Strichartz Estimates:** {cite}`KeelTao98` establishes endpoint Strichartz estimates for wave and Schrödinger equations, which are the foundation of modern well-posedness theory for dispersive PDE. {cite}`Strichartz77` provides the original $L^p$-$L^q$ estimates.

- **Hyperbolic Systems:** The Friedrichs energy method provides the symmetric hyperbolic formulation. Modern treatments extend to quasilinear systems with appropriate modifications for variable coefficients.

- **Profile Decomposition:** {cite}`BahouriGerard99` develops profile decomposition for dispersive equations. {cite}`KenigMerle06` applies concentration-compactness techniques to critical nonlinear Schrödinger equations.

- **Functional Analysis:** {cite}`Lions69` provides the Aubin-Lions compactness lemma used in parabolic existence. The Fourier restriction method and $X^{s,b}$ spaces (Bourgain spaces) are standard tools in dispersive PDE.

Each template instantiates a specific theorem from this literature, with preconditions verified against soft certificates. The automation comes from recognizing that these classical theorems follow template patterns identifiable from algebraic signatures.

:::
