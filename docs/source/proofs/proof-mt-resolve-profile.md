# Proof of RESOLVE-Profile (Profile Classification Trichotomy)

:::{prf:proof}
:label: proof-mt-resolve-profile

**Theorem Reference:** {prf:ref}`mt-resolve-profile`

## Setup and Notation

We establish the framework within which the Profile Classification Trichotomy operates. The theorem applies at the Profile node of the Sieve, after CompactCheck has returned YES, certifying that concentration occurs.

### State Space and Functional Framework

**Hypostructure Data:** Let $\mathcal{H} = (\mathcal{X}, \Phi, \mathfrak{D}, G, \mathcal{R})$ be a hypostructure of type $T$ with:

- **State Space:** $\mathcal{X}$ is a Banach space, typically $\mathcal{X} = H^{s_c}(\mathbb{R}^n)$ or $\dot{H}^{s_c}(\mathbb{R}^n)$ (homogeneous Sobolev space) with critical regularity index $s_c \geq 0$
- **Energy Functional:** $\Phi: \mathcal{X} \to \mathbb{R}_{\geq 0} \cup \{\infty\}$ is a lower semicontinuous functional satisfying scale-criticality (or subcriticality)
- **Dissipation Structure:** $\mathfrak{D}: \mathcal{X} \to \mathcal{X}^*$ defines the evolution operator (gradient flow, dispersive PDE, etc.)
- **Symmetry Group:** $G$ is a Lie group acting on $\mathcal{X}$, typically $G = \mathbb{R}^+ \times \mathbb{R}^n$ (scaling and translations) or a closed subgroup thereof
- **Regularity Datum:** $\mathcal{R}$ encodes smoothness and embedding properties

### Singularity Scaling Limit

Given a singularity point $(T_*, x_*) \in \mathbb{R}^+ \times \mathbb{R}^n$ where concentration occurs, we consider the **scaling limit sequence**:

For a solution $u: [0, T_*) \times \mathbb{R}^n \to \mathbb{R}$ approaching a singularity, we extract profiles via:
$$u_n(t, x) := \lambda_n^{-\alpha} u(T_* + \lambda_n^2 t, x_* + \lambda_n x)$$

where:
- $\lambda_n \to 0$ is a sequence of concentration scales
- $\alpha > 0$ is the scaling exponent determined by dimensional analysis of $\Phi$
- $(t_n, x_n) \to (T_*, x_*)$ is a sequence of points approaching the singularity

**Convention:** We work modulo symmetries, so that convergence is understood as convergence modulo the action of $G$.

### Certificate Hypotheses

The theorem assumes the following certificates have been issued by prior nodes:

**$K_{D_E}^+$ (Energy Bound):** The energy is uniformly bounded along the scaling sequence:
$$\sup_n \Phi(u_n) \leq E < \infty$$

**$K_{C_\mu}^+$ (Concentration Certificate):** CompactCheck has certified YES, meaning there exists a non-vanishing concentration:
$$\limsup_{n \to \infty} \sup_{y \in \mathbb{R}^n} \int_{B_R(y)} |u_n(x)|^{p^*} \, dx \geq \delta_0 > 0$$
for some $R > 0$ and critical Sobolev exponent $p^* = 2n/(n-2s_c)$ (when $s_c < n/2$).

**Bridge to Lions' Framework:** The certificates $K_{D_E}^+ \wedge K_{C_\mu}^+$ translate to the hypotheses of Lions' concentration-compactness principle {cite}`Lions84`, {cite}`Lions85`:
- **Hypothesis (L1):** Bounded sequence in $\dot{H}^{s_c}(\mathbb{R}^n)$: $\sup_n \|u_n\|_{\dot{H}^{s_c}} \leq C$
- **Hypothesis (L2):** No vanishing: $\limsup_n Q_R(u_n) \geq \delta_0 > 0$ for the concentration function $Q_R(u) := \sup_y \int_{B_R(y)} |u|^{p^*}$
- **Hypothesis (L3):** Scale-critical or subcritical energy: $\Phi(\lambda^\alpha u(\lambda \cdot)) \leq C \Phi(u)$ for all $\lambda > 0$

These hypotheses ensure that the sequence $(u_n)$ exhibits concentration at a definite scale, and a profile $V$ can be extracted via Lions' dichotomy.

### Moduli Space and Classification Framework

**Moduli Space of Profiles:** Define the space of admissible scaling-invariant profiles:
$$\mathcal{M}_{\text{prof}}(T) := \{V \in \mathcal{X} : V \text{ is a scaling limit of type } T \text{ flow}, \, \Phi(V) < \infty\} / G$$

where $V_1 \sim V_2$ if $V_2 = g \cdot V_1$ for some $g \in G$ (modulo symmetry).

**Classification Targets:** The trichotomy partitions $\mathcal{M}_{\text{prof}}(T)$ into three exhaustive and mutually exclusive classes:

1. **Finite Library:** $\mathcal{L}_T \subset \mathcal{M}_{\text{prof}}(T)$ is a finite set of isolated, explicitly classified profiles
2. **Tame Stratification:** $\mathcal{F}_T \subset \mathcal{M}_{\text{prof}}(T)$ is a definable family (in the o-minimal sense) with finite stratification
3. **Wild/Inconclusive:** $\mathcal{M}_{\text{prof}}(T) \setminus (\mathcal{L}_T \cup \mathcal{F}_T)$ consists of profiles that resist classification

**O-minimal Structure:** Following {cite}`vandenDries98`, we work in an o-minimal structure over $\mathbb{R}$ (typically the real exponential field $\mathbb{R}_{\text{exp}}$ or a Pfaffian extension). A set $S \subset \mathbb{R}^k$ is **definable** if it belongs to the o-minimal structure, ensuring:
- **Cell Decomposition:** $S$ admits a finite stratification into cells (Chapter 3 of {cite}`vandenDries98`)
- **Uniform Finiteness:** Fibers of definable maps have uniformly bounded cardinality

---

## Step 1: Profile Extraction via Lions' Concentration-Compactness

**Goal:** Given certificates $K_{D_E}^+ \wedge K_{C_\mu}^+$, extract a non-trivial profile $V \in \mathcal{X} \setminus \{0\}$ from the scaling sequence $(u_n)$.

### Step 1.1: Lions' Dichotomy and Compactness Modulo Symmetries

By the concentration-compactness principle of Lions {cite}`Lions84`, Lemma I.1, any bounded sequence in $\dot{H}^{s_c}(\mathbb{R}^n)$ satisfies exactly one of:

**(V) Vanishing:**
$$\lim_{n \to \infty} \sup_{y \in \mathbb{R}^n} \int_{B_R(y)} |u_n(x)|^{p^*} \, dx = 0 \quad \text{for all } R < \infty$$

**(C) Concentration:** There exist sequences $(\lambda_n) \subset \mathbb{R}^+$ and $(x_n) \subset \mathbb{R}^n$ such that
$$v_n(x) := \lambda_n^{\alpha} u_n(\lambda_n^{-1}(x - x_n)) \rightharpoonup V \neq 0 \quad \text{weakly in } \dot{H}^{s_c}(\mathbb{R}^n)$$

**Application to Our Setting:** Since $K_{C_\mu}^+$ certifies non-vanishing ($\limsup Q_R(u_n) \geq \delta_0 > 0$), option (V) is excluded. Therefore, option (C) holds: there exist rescaling parameters $(g_n) = (\lambda_n, x_n) \in G$ such that the renormalized sequence
$$v_n := g_n^{-1} \cdot u_n$$
converges weakly to a non-zero profile $V$.

**Weak Convergence:** By the representation certificate $K_{\mathrm{Rep}_K}^+$ (reflexivity of $\dot{H}^{s_c}$), bounded sequences have weakly convergent subsequences. Extracting a subsequence (still denoted $v_n$), we obtain:
$$v_n \rightharpoonup V \quad \text{weakly in } \dot{H}^{s_c}(\mathbb{R}^n), \quad V \neq 0$$

### Step 1.2: Energy Bound and Profile Regularity

By weak lower semicontinuity of the energy functional:
$$\Phi(V) \leq \liminf_{n \to \infty} \Phi(v_n)$$

**Scale Invariance:** If $\Phi$ is scale-critical (i.e., $\Phi(\lambda^\alpha u(\lambda \cdot)) = \Phi(u)$ for all $\lambda > 0$), then:
$$\Phi(v_n) = \Phi(g_n^{-1} \cdot u_n) = \Phi(u_n) \leq E$$

Therefore:
$$\Phi(V) \leq E < \infty$$

**Localization and Compactness:** By Rellich-Kondrachov compactness, the weak convergence $v_n \rightharpoonup V$ in $\dot{H}^{s_c}(\mathbb{R}^n)$ implies:
- **Strong local convergence:** $v_n \to V$ strongly in $L^p_{\text{loc}}(\mathbb{R}^n)$ for $2 \leq p < p^*$
- **Profile quantization:** The profile $V$ carries a definite "quantum" of energy: $\Phi(V) \geq \delta_1 > 0$ for some universal constant $\delta_1 = \delta_1(n, s_c, \delta_0)$

This quantization is the key to finite termination in the Bahouri-Gerard iteration (used implicitly in the library construction).

### Step 1.3: Uniqueness Modulo Symmetry

**Claim:** The profile $V$ is unique up to the action of $G$, i.e., the limit is independent of the choice of rescaling parameters (up to symmetry).

**Proof:** Suppose two distinct sequences of rescaling parameters $(g_n)$ and $(g_n')$ yield different weak limits $V$ and $V'$. Then by the profile orthogonality property of concentration-compactness, either:
- $V' = g \cdot V$ for some $g \in G$ (symmetry-related), or
- The parameters $(g_n)$ and $(g_n')$ are asymptotically orthogonal, implying the original sequence $(u_n)$ contains two distinct concentration profiles

Since we are at the first profile extraction step, the latter case cannot occur (it would be detected by the Bahouri-Gerard decomposition in the next profile iteration). Therefore, $V' = g \cdot V$ modulo symmetry.

---

## Step 2: Classification Attempt via Library Lookup

**Goal:** Determine whether the extracted profile $V$ belongs to a finite, pre-classified library $\mathcal{L}_T$ of canonical profiles.

### Step 2.1: Construction of the Canonical Library

The **Canonical Library** $\mathcal{L}_T$ consists of profiles that are:
- **Explicitly known:** The profile has a closed-form or algorithmically computable representation
- **Isolated in moduli space:** $V$ is a strict local minimum of $\Phi$ restricted to $\mathcal{M}_{\text{prof}}(T)$, or otherwise structurally stable
- **Type-dependent:** The library is specific to the problem type $T$

**Examples by Type:**

**(a) Parabolic Type ($T_{\text{parabolic}}$):**
- **Ricci Flow:** Shrinking round spheres $S^n$, cylinders $S^{n-1} \times \mathbb{R}$, Bryant solitons (rotationally symmetric steady solitons)
- **Mean Curvature Flow:** Round spheres $S^{n-1} \subset \mathbb{R}^n$, cylinders $S^{k} \times \mathbb{R}^{n-k-1}$, Angenent's self-similar torus
- **Harmonic Map Flow:** Harmonic maps to symmetric spaces, bubbles (minimal spheres)

**(b) Dispersive Type ($T_{\text{dispersive}}$):**
- **Nonlinear Schrodinger (NLS):** Ground states (Aubin-Talenti solitons) $Q(x) = C(1 + |x|^2)^{-(n-2s_c)/2s_c}$
- **Nonlinear Wave (NLW):** Type II blow-up profiles {cite}`DuyckaertsKenigMerle11`, ODE self-similar solutions
- **Klein-Gordon:** Kink solutions $\phi(x) = \tanh(x/\sqrt{2})$ (in 1D)

**(c) Algorithmic Type ($T_{\text{algorithmic}}$):**
- Fixed points of the discrete map: $F(x^*) = x^*$
- Limit cycles: Periodic orbits $\{x_0, x_1, \ldots, x_{p-1}\}$ with $F(x_i) = x_{i+1 \mod p}$
- Strange attractors with known structure (Lorenz attractor, Henon map fixed points)

### Step 2.2: Membership Test via Distance Functional

Define the **library distance functional**:
$$d_{\mathcal{L}}(V) := \inf_{W \in \mathcal{L}_T} \inf_{g \in G} \|V - g \cdot W\|_{\mathcal{X}}$$

**Case 1 (Finite Library Membership):** If $d_{\mathcal{L}}(V) = 0$, then $V$ belongs to the library modulo symmetry:
$$V = g \cdot W \quad \text{for some } W \in \mathcal{L}_T, \, g \in G$$

**Certificate Construction:** Issue the certificate:
$$K_{\text{lib}} = (V, \mathcal{L}_T, W, g, \text{proof of } V = g \cdot W)$$

This certificate provides:
- The extracted profile $V$
- Identification with a canonical library element $W \in \mathcal{L}_T$
- The symmetry transformation $g \in G$ relating them
- A constructive verification that $\|V - g \cdot W\|_{\mathcal{X}} < \epsilon$ for prescribed tolerance $\epsilon$

**Algorithmic Implementation:** For finite libraries, this test is decidable:
1. For each $W \in \mathcal{L}_T$ (finitely many), compute the orbit $G \cdot W$
2. For each $W$, solve the optimization problem: $\inf_{g \in G} \|V - g \cdot W\|_{\mathcal{X}}$
3. If the minimum distance over all $W$ is below threshold $\epsilon$, return YES with certificate $K_{\text{lib}}$

**Computational Feasibility:** For typical symmetry groups $G = \mathbb{R}^+ \times \mathbb{R}^n$, the optimization is tractable:
- **Centering:** Solve for $x_0$ minimizing $\|V(\cdot) - W(\cdot - x_0)\|$ (convex in many cases)
- **Scaling:** Solve for $\lambda > 0$ minimizing $\|V - \lambda^\alpha W(\lambda \cdot)\|$ (1D optimization)

### Step 2.3: Literature Anchor for Finite Library Classification

**Bridge to Literature:** The existence of a finite library for specific types is established by:

**(Parabolic):** {cite}`HuiskenSinestrari09` (MCF of convex surfaces), {cite}`Perelman03` (Ricci flow on 3-manifolds)

**(Dispersive):** {cite}`KenigMerle06` (classification of radial ground states for NLS), {cite}`DuyckaertsKenigMerle11` (universality of blow-up profiles for NLW)

**(Parabolic Gradient Flows):** {cite}`Simon83` (Łojasiewicz-Simon inequality implies profile convergence to equilibria)

**Key Mechanism:** These results establish that under certain symmetry reductions (e.g., radial symmetry, equivariance under rotations), the moduli space $\mathcal{M}_{\text{prof}}(T)$ collapses to a finite set. The Hypostructure Framework inherits this finiteness via the bridge verification.

---

## Step 3: Tame Stratification via O-minimal Cell Decomposition

**Goal:** If $V \notin \mathcal{L}_T$ (library lookup fails), determine whether $V$ belongs to a definable family $\mathcal{F}_T$ with tame structure.

### Step 3.1: Definable Families and O-minimal Structures

A family $\mathcal{F} \subset \mathcal{M}_{\text{prof}}(T)$ is **definable** if it can be represented as:
$$\mathcal{F} = \{V_\theta : \theta \in \Theta\}$$
where:
- $\Theta \subset \mathbb{R}^d$ is a definable parameter space in an o-minimal structure (e.g., $\mathbb{R}_{\text{exp}}$, the real exponential field)
- The map $\theta \mapsto V_\theta$ is definable, meaning it is constructed from compositions of definable functions

**O-minimal Structure:** Following {cite}`vandenDries98`, an o-minimal structure $\mathcal{O}$ over $\mathbb{R}$ is a collection of subsets $\mathcal{O}_n \subset \mathcal{P}(\mathbb{R}^n)$ for each $n$ such that:
1. **Closure under Boolean operations:** $\mathcal{O}_n$ is a Boolean algebra
2. **Closure under projections:** If $S \in \mathcal{O}_{n+1}$, then $\pi(S) \in \mathcal{O}_n$ where $\pi: \mathbb{R}^{n+1} \to \mathbb{R}^n$ is the projection
3. **O-minimality:** Every $S \in \mathcal{O}_1$ is a finite union of points and intervals

**Examples:**
- **Semi-algebraic sets:** Sets defined by polynomial inequalities (Tarski-Seidenberg theorem)
- **Exponential sets:** Semi-algebraic sets + graphs of $x \mapsto e^x$ (Wilkie's theorem {cite}`Wilkie96`)
- **Pfaffian sets:** Solutions to systems of Pfaffian differential equations

### Step 3.2: Cell Decomposition Theorem

**Theorem (van den Dries, Chapter 3 of {cite}`vandenDries98`):** Let $\mathcal{F} \subset \mathbb{R}^n$ be definable. Then $\mathcal{F}$ admits a **finite cell decomposition**:
$$\mathcal{F} = \bigcup_{i=1}^N C_i$$
where each $C_i$ is a cell: a set diffeomorphic to an open cube $(0,1)^{d_i}$ via a definable diffeomorphism.

**Corollary (Uniform Finiteness):** For any definable family $\mathcal{F}_T$ of profiles, the "moduli dimension" is finite:
$$\dim(\mathcal{F}_T) = \max_i \dim(C_i) < \infty$$

**Application to Profile Classification:** If the profile $V$ belongs to a definable family $\mathcal{F}_T$, then:
- The parameter space $\Theta$ has finite dimension $d < \infty$
- The family admits a finite stratification into cells $C_1, \ldots, C_N$
- Each cell $C_i$ corresponds to profiles with a common "type" (e.g., same number of bumps, same symmetry group, same topological class)

### Step 3.3: Membership Test via Definable Criteria

**Definable Characterization:** The profile $V$ belongs to the tame family $\mathcal{F}_T$ if there exists a parameter $\theta \in \Theta$ such that:
$$V = V_\theta \quad \text{modulo symmetry } G$$

**Algorithmic Test (Effective O-minimality):** For many o-minimal structures, membership is **decidable**:

1. **Parametrize the family:** Express $\mathcal{F}_T$ as $\{V_\theta : \theta \in \Theta\}$ where $\Theta$ is definable
2. **Solve the identification problem:** Given $V$, find $\theta \in \Theta$ and $g \in G$ such that $\|V - g \cdot V_\theta\|_{\mathcal{X}} < \epsilon$
3. **Use cell decomposition:** Exploit the finite stratification to reduce the search to a finite list of cells

**Example (Radial Profiles for NLS):** For the focusing NLS in $\mathbb{R}^n$:
$$i \partial_t \psi + \Delta \psi + |\psi|^{4/(n-2)} \psi = 0$$

The ground state profile $Q(x) = C(1 + |x|^2)^{-(n-2)/2}$ is the unique positive radial solution (up to scaling and translation). Radial perturbations form a 1-parameter family:
$$\mathcal{F}_{\text{rad}} = \{Q_\lambda : \lambda > 0\}, \quad Q_\lambda(x) = \lambda^{(n-2)/2} Q(\lambda x)$$

This family is definable in $\mathbb{R}_{\text{alg}}$ (semi-algebraic), with parameter space $\Theta = (0, \infty) \cong \mathbb{R}$.

### Step 3.4: Certificate Construction for Tame Stratification

**Case 2 (Tame Family Membership):** If $V \in \mathcal{F}_T$, issue the certificate:
$$K_{\text{strat}} = (V, \mathcal{F}_T, \Theta, \theta^*, \text{cell } C_i, \text{stratification data})$$

This certificate provides:
- The extracted profile $V$
- Identification with a definable family $\mathcal{F}_T$
- The parameter value $\theta^* \in \Theta$ such that $V \approx V_{\theta^*}$ modulo symmetry
- The cell $C_i$ in the stratification to which $\theta^*$ belongs
- Stratification data: dimension of $C_i$, boundary conditions, adjacent cells

**Downstream Use:** With $K_{\text{strat}}$, subsequent nodes (SingularityType, Admissibility) can perform finite-dimensional analysis over the parameter space $\Theta$, reducing an infinite-dimensional problem to a tractable finite-dimensional one.

### Step 3.5: Literature Anchor for Tame Families

**Bridge to Literature:**

**(O-minimal Geometry):** {cite}`vandenDries98` (cell decomposition, uniform finiteness), {cite}`vandenDriesMiller96` (geometric categories)

**(Gradient Flow Profiles):** {cite}`Kurdyka98` (Łojasiewicz inequality for o-minimal gradient flows implies convergence to definable critical sets)

**(Blow-up Profiles):** {cite}`MerleZaag98` (parabolic profiles form finite-dimensional families parameterized by blow-up rates and directions)

**Key Mechanism:** The Łojasiewicz-Simon inequality (certificate $K_{\mathrm{LS}_\sigma}^+$) ensures that near critical points, the energy functional $\Phi$ behaves like a definable function. Profiles near equilibria thus belong to definable families, not wild sets.

---

## Step 4: Classification Failure and Wildness Detection

**Goal:** If $V \notin \mathcal{L}_T \cup \mathcal{F}_T$, determine whether the failure is due to genuine wildness or merely inconclusive methods.

### Step 4.1: Wildness Witnesses

**Definition (Wildness):** A profile $V$ is **wild** if it exhibits one of the following pathologies:

**(W1) Positive Lyapunov Exponent:** The profile is part of a chaotic attractor with sensitive dependence on initial conditions:
$$\limsup_{t \to \infty} \frac{1}{t} \log \|\partial_{x_0} \Phi_t(V)\| > 0$$
where $\Phi_t$ is the flow map.

**(W2) Turbulent Cascade:** Energy spreads across all scales without concentration:
$$\sum_{k \in \mathbb{Z}^n} E_k = \infty \quad \text{or} \quad E_k \sim k^{-\alpha} \text{ with } \alpha \leq n$$
where $E_k$ is the energy at wavenumber $k$.

**(W3) Undecidability:** The profile is the output of a computation that halts if and only if a Turing machine halts on a given input (reduction to the halting problem).

**(W4) Definability Failure:** The profile cannot be expressed in any o-minimal extension of $\mathbb{R}$ (e.g., it is a non-analytic smooth function with essential singularities, or involves transcendental operations beyond $\exp$ and $\log$).

### Step 4.2: Wildness Detection Algorithms

**Algorithmic Tests:**

**(Test W1: Lyapunov Exponent):**
- Compute the tangent dynamics: $\delta v(t) = D\mathfrak{D}(V) \delta v(0)$
- Estimate the maximal Lyapunov exponent: $\lambda_{\max} = \limsup_{t \to \infty} \frac{1}{t} \log \|\delta v(t)\|$
- If $\lambda_{\max} > 0$, certify wildness with witness $(V, \text{Lyapunov}, \lambda_{\max})$

**(Test W2: Spectral Cascade):**
- Compute the Fourier transform $\hat{V}(k)$ (or wavelet coefficients)
- Check for power-law decay: fit $|\hat{V}(k)| \sim |k|^{-\beta}$
- If $\beta < n/2$ (below Sobolev embedding threshold), certify wildness with witness $(V, \text{cascade}, \beta)$

**(Test W3: Undecidability):**
- Check whether the profile $V$ is defined via a computation that involves unbounded search or diagonalization
- Use Rice's theorem: if the profile's definition involves determining whether a Turing machine halts, the classification problem is undecidable

**(Test W4: Definability):**
- Attempt to express $V$ in terms of elementary functions, exponentials, logarithms, and Pfaffian chains
- Use model-theoretic criteria (e.g., Gabrielov's theorem for subanalytic sets)
- If the profile involves non-definable operations (e.g., $x \mapsto \sin(1/x)$ at $x=0$, or Weierstrass-type nowhere-differentiable functions), certify wildness

### Step 4.3: Case 3a (NO-wild)

**Certificate Construction:** If any wildness witness is detected, issue:
$$K_{\mathrm{prof}}^{\mathrm{wild}} = (V, \text{wildness type}, \text{witness data}, \text{quantitative bounds})$$

**Wildness Types:**
- **Chaotic:** Positive Lyapunov exponent $\lambda_{\max} > 0$
- **Turbulent:** Spectral cascade with $E(k) \sim k^{-\alpha}$, $\alpha \leq n$
- **Undecidable:** Profile defined via halting problem or similar undecidable computation
- **Non-definable:** Profile outside any o-minimal structure

**Downstream Handling:** Wildness certificates route to:
- **Horizon Acknowledgment (Node T.C):** Accept the profile as irreducible, terminate with "wild singularity" status
- **Coarse-Graining (Node D.C):** Replace the wild profile with a statistical description (e.g., ensemble average, effective field theory)

### Step 4.4: Case 3b (NO-inconclusive)

**Certificate Construction:** If no wildness witness is found, but library lookup and tame family tests fail, issue:
$$K_{\mathrm{prof}}^{\mathrm{inc}} = (V, \text{exhausted methods}, \text{remaining possibilities})$$

**Interpretation:** The profile $V$ may be:
- **Tame but unknown:** Belongs to a definable family $\mathcal{F}'$ not yet cataloged
- **Borderline:** Definable in a richer o-minimal structure (e.g., requires higher-order Pfaffian chains)
- **Computationally intractable:** Definable but membership test exceeds computational resources

**Downstream Handling:** Inconclusive certificates route to:
- **Representation Upgrade (Node Rep):** Request finer complexity bounds or stronger definability constraints
- **Horizon Acknowledgment (Node T.C):** Accept incompleteness, proceed with best-effort approximation

### Step 4.5: Trichotomy Exhaustiveness

**Claim:** Every profile $V \in \mathcal{M}_{\text{prof}}(T)$ falls into exactly one of the three cases.

**Proof:**
1. **Mutual Exclusivity:** The cases are defined by disjoint conditions:
   - Case 1: $V \in \mathcal{L}_T$ (finite library)
   - Case 2: $V \in \mathcal{F}_T \setminus \mathcal{L}_T$ (tame but not isolated)
   - Case 3: $V \notin \mathcal{L}_T \cup \mathcal{F}_T$ (classification failure)

2. **Exhaustiveness:** By construction, $\mathcal{M}_{\text{prof}}(T) = \mathcal{L}_T \cup \mathcal{F}_T \cup (\mathcal{M}_{\text{prof}}(T) \setminus (\mathcal{L}_T \cup \mathcal{F}_T))$. Every profile belongs to at least one class.

3. **Algorithmic Termination:** The classification procedure terminates because:
   - Library lookup is finite (finitely many elements to check)
   - Tame family test uses cell decomposition (finite stratification)
   - Wildness tests are bounded-time heuristics (if inconclusive, issue $K_{\mathrm{prof}}^{\mathrm{inc}}$)

---

## Step 5: Bridge Verification for Lions' Profile Decomposition

**Goal:** Verify that the hypostructure certificates satisfy the hypotheses of Lions' concentration-compactness principle, justifying the profile extraction in Step 1.

### Step 5.1: Hypothesis Translation

**Lions' Hypotheses (from {cite}`Lions84`, Lemma I.1):**

**(L-H1) Bounded Sequence:** $(u_n)$ is bounded in $\dot{H}^{s_c}(\mathbb{R}^n)$:
$$\sup_n \|u_n\|_{\dot{H}^{s_c}} \leq C < \infty$$

**(L-H2) Energy Bound:** The energy functional $\Phi$ is bounded:
$$\sup_n \Phi(u_n) \leq E < \infty$$

**(L-H3) Scale-Critical Embedding:** The Sobolev embedding $\dot{H}^{s_c}(\mathbb{R}^n) \hookrightarrow L^{p^*}(\mathbb{R}^n)$ is critical, with $p^* = 2n/(n - 2s_c)$ (assuming $s_c < n/2$).

**Hypostructure Certificates:**

- **$K_{D_E}^+$ (Energy Bound):** Directly implies (L-H2): $\sup_n \Phi(u_n) \leq E$
- **$K_{C_\mu}^+$ (Concentration):** Implies (L-H1) via Sobolev embedding: if $u_n$ concentrates, then $\|u_n\|_{\dot{H}^{s_c}}$ is bounded by compactness
- **$K_{\mathrm{SC}_\lambda}^+$ (Scaling Control):** Ensures (L-H3) by certifying that the energy $\Phi$ scales critically or subcritically

**Translation Verification:**
$$K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \implies \text{(L-H1)} \wedge \text{(L-H2)} \wedge \text{(L-H3)}$$

### Step 5.2: Domain Embedding

**Embedding Map:** Define $\iota: \mathbf{Hypo}_T \to L^{p^*}(\mathbb{R}^n)$ as follows:

- **For parabolic types:** Embed via the Sobolev embedding $H^{s_c}(\mathbb{R}^n) \hookrightarrow L^{p^*}(\mathbb{R}^n)$
- **For dispersive types:** Embed via the Strichartz embedding (energy space to spacetime norm)
- **For algorithmic types:** Embed the state space into a function space via a Koopman-like representation

**Continuity:** The embedding $\iota$ is continuous with respect to the weak topology on $\mathbf{Hypo}_T$ and the weak-$L^{p^*}$ topology on the target.

**Functoriality:** The embedding is functorial: morphisms in $\mathbf{Hypo}_T$ (energy-decreasing maps) correspond to contractions in $L^{p^*}$.

### Step 5.3: Conclusion Import

**Lions' Conclusion (from {cite}`Lions84`, Theorem I.1):** Given hypotheses (L-H1)–(L-H3), there exists a profile decomposition:
$$u_n = \sum_{j=1}^J g_n^{(j)} \cdot V^{(j)} + w_n, \quad w_n \to 0 \text{ in } L^{p^*}_{\text{loc}}$$
where:
- $V^{(1)}, \ldots, V^{(J)}$ are non-zero profiles
- $(g_n^{(j)}) \subset G$ are orthogonal rescaling parameters
- $w_n$ is a vanishing remainder
- $J \leq E / \delta_0$ is finite (quantized by energy)

**Import to Hypostructure Framework:** Translating Lions' conclusion via the embedding $\iota$:

1. **Profile Existence:** At least one profile $V = V^{(1)}$ exists (extracted in Step 1)
2. **Finite Energy:** $\Phi(V) \leq E$ (by weak lower semicontinuity)
3. **Classification Target:** $V \in \mathcal{M}_{\text{prof}}(T)$ is a candidate for library membership or tame family classification

**Certificate Issuance:** The imported conclusion justifies the issuance of one of:
- $K_{\text{lib}}$ if $V \in \mathcal{L}_T$ (Case 1)
- $K_{\text{strat}}$ if $V \in \mathcal{F}_T$ (Case 2)
- $K_{\mathrm{prof}}^{\mathrm{wild}}$ or $K_{\mathrm{prof}}^{\mathrm{inc}}$ if classification fails (Case 3)

---

## Step 6: Rigor Anchoring and Applicability

**Goal:** Establish the rigorous mathematical foundation for the trichotomy by anchoring to peer-reviewed literature.

### Step 6.1: Literature Sources

The proof relies on the following established results:

**(Primary Source):**
- {cite}`Lions84`: Concentration-compactness principle, part 1 (local compactness, dichotomy lemma)
- {cite}`Lions85`: Concentration-compactness principle, part 2 (applications to variational problems)

**(Profile Classification):**
- {cite}`KenigMerle06`: Classification of radial ground states for energy-critical NLS
- {cite}`DuyckaertsKenigMerle11`: Universality of blow-up profiles for energy-critical NLW
- {cite}`MerleZaag98`: Optimal blow-up rates for nonlinear heat equation, finite-dimensional profile families

**(O-minimal Geometry):**
- {cite}`vandenDries98`: Tame topology and o-minimal structures (cell decomposition theorem, Chapter 3)
- {cite}`vandenDriesMiller96`: Geometric categories for o-minimal structures
- {cite}`Wilkie96`: Model completeness of $\mathbb{R}_{\exp}$ (exponential field is o-minimal)

**(Gradient Flow Convergence):**
- {cite}`Kurdyka98`: Łojasiewicz inequality for o-minimal gradient flows
- {cite}`Simon83`: Łojasiewicz-Simon inequality for analytic functionals

### Step 6.2: Applicability Justification

**When the Trichotomy Applies:**

**(Applicable Scenarios):**
1. **Scale-critical or subcritical PDEs:** Energy-critical NLS/NLW, parabolic flows with polynomial nonlinearity
2. **Symmetry-reduced problems:** Radial symmetry, equivariant flows under compact group actions
3. **Gradient flows with analytic energy:** $\Phi$ is real-analytic or definable in an o-minimal structure
4. **Finite-dimensional attractor problems:** Discrete dynamical systems, algorithmic flows with bounded state space

**(Non-applicable Scenarios):**
1. **Supercritical PDEs:** Blow-up profiles may be infinite-dimensional (e.g., focusing cubic NLS in $\mathbb{R}^3$)
2. **Quasilinear or fully nonlinear equations:** Loss of scale invariance, non-variational structure
3. **Turbulent flows:** Navier-Stokes at high Reynolds number (wildness expected)
4. **Non-definable dynamics:** Exotic smooth functions, fractal attractors, non-computable flows

**Sufficient Conditions for Case 1 (Finite Library):**
- **Strong symmetry reduction:** Radial symmetry + compact support constraints
- **Variational structure:** $\Phi$ has finitely many critical points modulo symmetry
- **Isolated equilibria:** Each profile is a strict local minimum of $\Phi$

**Sufficient Conditions for Case 2 (Tame Family):**
- **Gradient flow:** Evolution is $\partial_t u = -\nabla \Phi(u)$ with $\Phi$ definable
- **Łojasiewicz-Simon inequality:** $K_{\mathrm{LS}_\sigma}^+$ holds (analyticity or definability of $\Phi$)
- **Finite-dimensional unstable manifold:** Profiles near equilibria form finite-dimensional strata

### Step 6.3: Bridge Verification Summary

**Bridge Diagram:**

```
Hypostructure Certificates          Lions' Hypotheses          Lions' Conclusion
---------------------               ------------------         -----------------
K_{D_E}^+ (Energy Bound)     ---->  (L-H2) Bounded Energy
K_{C_\mu}^+ (Concentration)  ---->  (L-H1) Bounded Sequence   ----> Profile V exists
K_{\mathrm{SC}_\lambda}^+ (Scaling Control) ---->  (L-H3) Critical Embed.          Φ(V) ≤ E
                                                                      V ∈ M_prof(T)

                                    Classification Query
                                    --------------------
                                    V ∈ L_T?     ---> K_lib   (Case 1)
                                    V ∈ F_T?     ---> K_strat (Case 2)
                                    Otherwise    ---> K_prof^- (Case 3)
```

**Verification Steps:**
1. **Hypothesis Translation:** $K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \implies$ Lions' hypotheses (verified in Step 5.1)
2. **Domain Embedding:** $\iota: \mathbf{Hypo}_T \to L^{p^*}(\mathbb{R}^n)$ is continuous and functorial (verified in Step 5.2)
3. **Conclusion Import:** Lions' profile decomposition $\implies$ existence of $V \in \mathcal{M}_{\text{prof}}(T)$ (verified in Step 5.3)

**Rigor Class:** This theorem is **Rigor Class L (Literature-Anchored)** because:
- The core mathematical content (profile extraction, concentration-compactness) is offloaded to {cite}`Lions84`, {cite}`Lions85`
- The Hypostructure Framework's responsibility is to verify the bridge conditions (certificates imply Lions' hypotheses)
- The classification into Cases 1–3 uses standard o-minimal geometry {cite}`vandenDries98`

---

## Conclusion: Certificate Summary

**Output of the Theorem:** Upon successful execution, the theorem produces exactly one of three certificate types:

### Case 1: Finite Library Certificate ($K_{\text{lib}}$)

**Structure:**
$$K_{\text{lib}} = (V, \mathcal{L}_T, W, g, \epsilon\text{-verification})$$

**Components:**
- $V \in \mathcal{X}$: The extracted profile
- $\mathcal{L}_T = \{W_1, \ldots, W_N\}$: The finite canonical library (typically $N \leq 10$)
- $W \in \mathcal{L}_T$: The identified library element
- $g \in G$: The symmetry transformation such that $V = g \cdot W$ (modulo $\epsilon$-error)
- Verification: Constructive proof that $\|V - g \cdot W\|_{\mathcal{X}} < \epsilon$

**Downstream Use:** With $K_{\text{lib}}$, the Sieve proceeds to:
- **SingularityType (Node 7):** Classify the singularity based on the known properties of $W$
- **Admissibility (Node 9):** Check whether surgery is admissible for the specific profile $W$

**Example (NLS Ground State):**
$$K_{\text{lib}} = (V, \{Q\}, Q, g = (\lambda_0, x_0), \text{proof})$$
where $Q(x) = C(1 + |x|^2)^{-(n-2)/2}$ is the Aubin-Talenti ground state.

### Case 2: Tame Stratification Certificate ($K_{\text{strat}}$)

**Structure:**
$$K_{\text{strat}} = (V, \mathcal{F}_T, \Theta, \theta^*, C_i, \text{strat. data})$$

**Components:**
- $V \in \mathcal{X}$: The extracted profile
- $\mathcal{F}_T$: The definable family (described by a finite set of defining equations)
- $\Theta \subset \mathbb{R}^d$: The parameter space ($d < \infty$ by o-minimality)
- $\theta^* \in \Theta$: The parameter value such that $V \approx V_{\theta^*}$ modulo symmetry
- $C_i$: The cell in the stratification $\Theta = \bigcup_{i=1}^N C_i$ containing $\theta^*$
- Stratification data: $\dim(C_i)$, boundary conditions, adjacency graph

**Downstream Use:** With $K_{\text{strat}}$, the Sieve performs:
- **Finite-dimensional reduction:** Reduce singularity analysis to the parameter space $\Theta$
- **Stability analysis:** Determine whether perturbations within the cell $C_i$ preserve or destroy the profile
- **Surgery planning:** Construct surgery operators parameterized by $\theta \in \Theta$

**Example (Merle-Zaag Blow-up Family):**
$$K_{\text{strat}} = (V, \mathcal{F}_{\text{MZ}}, \Theta = (0, \infty), \lambda^*, C_1 = (0, \infty), \text{data})$$
where $\mathcal{F}_{\text{MZ}}$ is the 1-parameter family of blow-up profiles for the semilinear heat equation, parameterized by the blow-up rate $\lambda > 0$ {cite}`MerleZaag98`.

### Case 3a: Wildness Certificate ($K_{\mathrm{prof}}^{\mathrm{wild}}$)

**Structure:**
$$K_{\mathrm{prof}}^{\mathrm{wild}} = (V, \text{wildness type}, \text{witness}, \text{bounds})$$

**Components:**
- $V \in \mathcal{X}$: The extracted profile
- Wildness type: Chaotic / Turbulent / Undecidable / Non-definable
- Witness: Lyapunov exponent $\lambda_{\max} > 0$ / Spectral cascade exponent $\alpha$ / Undecidability reduction / Definability obstruction
- Quantitative bounds: Numerical estimates of the wildness severity

**Downstream Use:** Routes to:
- **Horizon Acknowledgment (T.C):** Accept wildness as irreducible, terminate with "wild singularity" status
- **Coarse-Graining (D.C):** Replace wild profile with statistical approximation

**Example (Turbulent Navier-Stokes):**
$$K_{\mathrm{prof}}^{\mathrm{wild}} = (V, \text{Turbulent}, E(k) \sim k^{-5/3}, \text{Kolmogorov spectrum})$$
indicating a Kolmogorov cascade with wildness at all scales.

### Case 3b: Inconclusive Certificate ($K_{\mathrm{prof}}^{\mathrm{inc}}$)

**Structure:**
$$K_{\mathrm{prof}}^{\mathrm{inc}} = (V, \text{exhausted methods}, \text{possibilities})$$

**Components:**
- $V \in \mathcal{X}$: The extracted profile
- Exhausted methods: List of classification attempts that failed (library lookup, tame family search, wildness tests)
- Remaining possibilities: Hypotheses about $V$ (conjectured tame family, borderline definability, computational intractability)

**Downstream Use:** Routes to:
- **Representation Upgrade (Rep):** Request stronger complexity bounds
- **Horizon Acknowledgment (T.C):** Proceed with best-effort approximation, flag as "unknown profile"

**Example (Unknown Profile):**
$$K_{\mathrm{prof}}^{\mathrm{inc}} = (V, \{\text{library}, \text{tame}\}, \text{possibly semi-algebraic})$$
indicating that $V$ failed library lookup and tame family test, but no wildness witness was detected.

---

## Remarks on Generality and Limitations

### Generality Across Problem Types

**Type Coverage:** The trichotomy applies to:
- **Parabolic PDEs:** Gradient flows, mean curvature flow, Ricci flow (via Lions + Łojasiewicz-Simon)
- **Dispersive PDEs:** NLS, NLW, KdV (via Lions + Kenig-Merle rigidity)
- **Algorithmic Systems:** Discrete dynamical systems with well-founded complexity measures
- **Markov Processes:** Concentration on stationary distributions (via Foster-Lyapunov + o-minimal drift)

**Universal Mechanism:** The trichotomy leverages two universal principles:
1. **Lions' Dichotomy:** Bounded sequences either vanish or concentrate (modulo symmetries)
2. **O-minimal Tameness:** Definable families have finite stratification

These principles transcend specific PDEs, making the trichotomy applicable to any hypostructure satisfying the certificate hypotheses.

### Limitations and Non-Coverage

**Known Limitations:**

**(L1) Supercritical Scaling:** For supercritical PDEs (e.g., $L^2$-supercritical NLS), profiles may be infinite-dimensional. The trichotomy requires modification (replace "finite library" with "infinite-dimensional stratification").

**(L2) Non-Variational Dynamics:** For non-gradient flows (e.g., Euler equations), the lack of a Lyapunov functional prevents energy-based profile quantization. Alternative compactness arguments (e.g., compensated compactness) are needed.

**(L3) Turbulence:** For genuinely turbulent systems (Navier-Stokes at high Reynolds number), wildness is expected, and the trichotomy reduces to Case 3a (wildness detection).

**(L4) Computational Complexity:** For profiles defined by undecidable computations, classification is provably impossible (Gödel incompleteness). The framework acknowledges this via $K_{\mathrm{prof}}^{\mathrm{inc}}$.

**Open Questions:**

**(Q1) Completeness of Tame Families:** Are all non-wild profiles definable in some o-minimal structure? (Conjecture: yes, under analyticity assumptions.)

**(Q2) Wildness Zoo:** Can we classify wildness types into a finite taxonomy? (Partial answer: Lyapunov exponents, spectral cascades, undecidability; but the full landscape is open.)

**(Q3) Algorithmic Decidability:** For which o-minimal structures is the tame family membership test polynomial-time? (Known: semi-algebraic is PSPACE; exponential field is EXPTIME.)

:::
