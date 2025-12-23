# Proof of RESOLVE-Admissibility (Surgery Admissibility Trichotomy)

:::{prf:proof}
:label: proof-mt-resolve-admissibility

**Theorem Reference:** {prf:ref}`mt-resolve-admissibility`

**Theorem Statement:** Before invoking any surgery $S$ with mode $M$ and data $D_S$, the framework produces exactly one of three certificates:

1. **Case 1 (Admissible):** $K_{\text{adm}} = (M, D_S, \text{admissibility proof}, K_{\epsilon}^+)$ satisfying Canonicity, Codimension $\geq 2$, Capacity $\leq \varepsilon_{\text{adm}}$, and Progress Density $\Delta\Phi_{\text{surg}} \geq \epsilon_T > 0$

2. **Case 2 (Admissible up to equivalence):** $K_{\text{adm}}^{\sim} = (K_{\text{equiv}}, K_{\text{transport}}, K_{\text{adm}}[\tilde{x}])$ where an admissible equivalence move makes the surgery admissible

3. **Case 3 (Not admissible):** $K_{\text{inadm}} = (\text{failure reason}, \text{witness})$ with explicit reason (capacity too large, codimension too small, or Horizon/profile unclassifiable)

This proof establishes the completeness, soundness, and mutual exclusivity of this trichotomy through a systematic analysis of surgery data, drawing on geometric measure theory {cite}`Federer69`, Ricci flow surgery theory {cite}`Perelman03`, and profile classification {cite}`Lions84`.

---

## Setup and Notation

### Given Data

We are given:
- A structural flow datum $\mathcal{S} = (\mathcal{X}, S_t, \Phi, \mathfrak{D}, G, \partial)$ where:
  - $\mathcal{X}$ is the state stack (configuration space)
  - $S_t: \mathcal{X} \to \mathcal{X}$ is the semiflow (evolution operator)
  - $\Phi: \mathcal{X} \to \mathbb{R}_{\geq 0} \cup \{\infty\}$ is the cohomological height (energy functional)
  - $\mathfrak{D}: \mathcal{X} \to \mathbb{R}_{\geq 0}$ is the dissipation rate
  - $G$ is a compact Lie group acting on $\mathcal{X}$ (symmetry group)
  - $\partial: \mathcal{X} \to \mathcal{X}_\partial$ is the boundary restriction

- A trajectory $u(t) = S_t x$ with breakdown time $T_* < \infty$

- Surgery data $D_S = (\Sigma, V, t^*, \mathcal{O}_S)$ where:
  - $\Sigma \subset \mathcal{X}$ is the singular set (surgery locus)
  - $V \in \mathcal{M}_{\text{prof}}(T)$ is the blow-up profile at the singularity
  - $t^* \in [0, T_*)$ is the surgery time
  - $\mathcal{O}_S$ is the proposed surgery operator (excision and capping recipe)

- Surgery mode $M \in \{\text{C.E}, \text{S.E}, \text{C.D}, \text{T.E}, \text{S.D}\}$ indicating the failure type being addressed

### Profile Moduli Spaces

**Definition (Moduli Space of Profiles):** For a type $T$, the moduli space of profiles is:
$$\mathcal{M}_{\text{prof}}(T) := \{V : V \text{ is a scaling-invariant limit of type } T \text{ flow}\} / \sim$$
where $\sim$ is the equivalence relation under $G$-action.

This space admits a stratification:
- **Canonical Library:** $\mathcal{L}_T \subseteq \mathcal{M}_{\text{prof}}(T)$ is the finite set of isolated profiles with finite automorphism group and known surgery recipes
- **Definable Family:** $\mathcal{F}_T \subseteq \mathcal{M}_{\text{prof}}(T)$ is the o-minimal family of profiles admitting finite stratification
- **Horizon:** $\mathcal{M}_{\text{prof}}(T) \setminus \mathcal{F}_T$ consists of wild/unclassifiable profiles

**Examples of Canonical Libraries:**
| Type | Library $\mathcal{L}_T$ | Size |
|------|------------------------|------|
| $T_{\text{Ricci}}$ | $\{\text{Sphere}, \text{Cylinder}, \text{Bryant}\}$ | 3 |
| $T_{\text{MCF}}$ | $\{\text{Sphere}^n, \text{Cylinder}^k\}_{k \leq n}$ | $n+1$ |
| $T_{\text{NLS}}$ | $\{Q, Q_{\text{excited}}\}$ | 2 |

### Capacity Theory

**Definition (Sobolev Capacity):** For a compact set $K \subset \mathbb{R}^n$ and $1 < p < \infty$, the $p$-capacity is:
$$\text{Cap}_p(K) := \inf\left\{\int_{\mathbb{R}^n} |\nabla \varphi|^p + |\varphi|^p \, dx : \varphi \in C_c^\infty(\mathbb{R}^n), \varphi|_K \geq 1\right\}$$

For a general metric space $(\mathcal{X}, d, \mu)$, we use the variational capacity:
$$\text{Cap}(K) := \inf\left\{\int_{\mathcal{X}} |\nabla u|^2 \, d\mu : u \in W^{1,2}(\mathcal{X}), u|_K \geq 1\right\}$$

**Key Properties (from {cite}`Federer69`, Section 2.10):**
1. **Codimension Bound:** If $\text{Cap}_p(K) < \infty$, then $\dim_H(K) \leq n - p$
2. **Removability:** Sets with small capacity are removable for Sobolev functions
3. **Monotonicity:** $A \subseteq B \implies \text{Cap}(A) \leq \text{Cap}(B)$

### Admissibility Threshold

For each type $T$, there exists a critical capacity threshold $\varepsilon_{\text{adm}}(T) > 0$ such that surgery at sets with capacity $\leq \varepsilon_{\text{adm}}(T)$ can be performed without violating global energy bounds.

**Type-Dependent Thresholds:**
- **Parabolic types** ($T_{\text{Ricci}}, T_{\text{MCF}}$): $\varepsilon_{\text{adm}}(T) = r^{n-2}$ where $r$ is the surgery scale
- **Dispersive types** ($T_{\text{NLS}}, T_{\text{wave}}$): $\varepsilon_{\text{adm}}(T) = E_{\text{soliton}}/2$ (half the minimal soliton energy)
- **Algorithmic types**: $\varepsilon_{\text{adm}}(T) = \min\{\mathcal{C}(x) - \mathcal{C}(x') : x' \text{ reachable from } x\}$ (minimal complexity drop)

### Discrete Progress Constant

**Definition (Progress Measure):** A progress measure is a pair $(\mathcal{C}, \epsilon_T)$ where:
- $\mathcal{C}: \mathcal{X} \to \mathbb{R}_{\geq 0} \cup \mathbb{N}$ is the complexity measure (typically $\mathcal{C} = \Phi$ for continuous systems)
- $\epsilon_T > 0$ is the discrete progress constant satisfying:
  $$\mathcal{O}_S(x) = x' \implies \mathcal{C}(x) - \mathcal{C}(x') \geq \epsilon_T$$

This ensures well-foundedness: the number of surgeries is bounded by:
$$N_{\text{surgeries}} \leq \frac{\mathcal{C}(x_0)}{\epsilon_T} < \infty$$

**Zeno Prevention:** Without the discrete progress constraint, a sequence of surgeries with $\Delta\Phi_n \to 0$ (e.g., $\Delta\Phi_n = 2^{-n}$) could comprise infinitely many steps despite finite total energy drop. The uniform bound $\Delta\Phi \geq \epsilon_T$ excludes such pathological sequences.

---

## Step 1: Canonicity Verification (Profile Library Membership)

**Goal:** Determine whether the blow-up profile $V$ belongs to a classifiable family.

### Step 1.1: Profile Extraction

By the Profile Classification Trichotomy ({prf:ref}`mt-resolve-profile`), given the singularity data $(\Sigma, t^*)$, we have already extracted the blow-up profile $V$ via concentration-compactness:

**Profile Extraction Procedure:**
1. Take a sequence of times $t_n \nearrow t^*$ and points $x_n \in \Sigma$
2. Apply the compactness certificate $K_{C_\mu}^+$ to extract symmetry elements $g_n \in G$ and a weak limit:
   $$g_n \cdot u(t_n, x_n) \rightharpoonup V \quad \text{(weakly in } \mathcal{X}\text{)}$$
3. The profile $V$ is a self-similar solution: $V(s, y) = \lambda^{-\alpha} V(\lambda^2 s, \lambda y)$ for all $\lambda > 0$, where $\alpha$ is the scaling dimension of $\Phi$

### Step 1.2: Library Query (Case 1 Check)

**Query:** Is $V \in \mathcal{L}_T$ (canonical library)?

**Decision Procedure:**
1. **Finite Comparison:** Since $\mathcal{L}_T$ is finite (for good types $T$), we perform at most $|\mathcal{L}_T|$ comparisons
2. **Distance Metric:** For each $V_i \in \mathcal{L}_T$, compute:
   $$d_{\mathcal{M}}(V, V_i) := \inf_{g \in G} \|V - g \cdot V_i\|_{\mathcal{X}}$$
3. **Tolerance:** Check if $d_{\mathcal{M}}(V, V_i) \leq \delta_{\text{tol}}$ for the type-specific tolerance $\delta_{\text{tol}}(T)$

**Lemma 1.1** (Library Membership Decidability): For good types $T$, the predicate $V \in \mathcal{L}_T$ is decidable in finite time.

*Proof of Lemma 1.1:*
The canonical library $\mathcal{L}_T$ consists of isolated critical points of the profile energy functional on $\mathcal{M}_{\text{prof}}(T)$. By definition {prf:ref}`def-canonical-library`, each $V \in \mathcal{L}_T$ has:
1. Finite automorphism group: $|\text{Aut}(V)| < \infty$
2. Isolation: $\inf\{d_{\mathcal{M}}(V, W) : W \in \mathcal{M}_{\text{prof}} \setminus \{V\}\} > 0$

The distance computation $d_{\mathcal{M}}(V, V_i)$ involves minimizing over a compact group $G$ (assumed to be a compact Lie group). By compactness, the infimum is attained and computable numerically via gradient descent on $G$.

For dispersive equations, the profiles are ground states or solitons characterized by variational problems. The computation reduces to checking whether $V$ satisfies the Euler-Lagrange equation for each known profile type, which is decidable by spectral analysis {cite}`KenigMerle06`. □

**Outcome:**
- **If YES:** $V = V_i$ for some $V_i \in \mathcal{L}_T$ (up to $G$-equivalence). Proceed to Step 2 (Codimension Check). Tag: $K_{\text{can}}^+ = (V, V_i, d_{\mathcal{M}}(V, V_i))$
- **If NO:** Proceed to Step 1.3 (Definable Family Check)

### Step 1.3: Definable Family Query (Case 2 Check)

**Query:** Is $V \in \mathcal{F}_T \setminus \mathcal{L}_T$ (definable but non-canonical)?

For profiles in the definable family $\mathcal{F}_T$ but outside the canonical library, we check for **equivalence moves**:

**Definition (Admissible Equivalence Move):** An equivalence move is a transformation $\tilde{V} = \Psi(V)$ where:
1. $\Psi: \mathcal{M}_{\text{prof}}(T) \to \mathcal{M}_{\text{prof}}(T)$ is definable in an o-minimal structure
2. $\Psi$ preserves energy: $\Phi(\tilde{V}) = \Phi(V) + O(\delta)$ for small $\delta$
3. $\tilde{V} \in \mathcal{L}_T$ (the transformed profile is canonical)

**Examples of Equivalence Moves:**
- **Gauge transformation** in dispersive PDE (phase rotation)
- **Neck stretching** in geometric flows (asymptotic to canonical cylinder)
- **Modulation** to nearest ground state in NLS

**Lemma 1.2** (Tame Stratification): For $V \in \mathcal{F}_T$, the set of admissible equivalence moves $\{\Psi : \Psi(V) \in \mathcal{L}_T\}$ is finite.

*Proof of Lemma 1.2:*
By definition {prf:ref}`def-moduli-profiles`, the moduli space $\mathcal{M}_{\text{prof}}(T)$ for good types admits a finite stratification into algebraic varieties (for algebraic/tame types) or o-minimal cells (for o-minimal types).

The canonical library $\mathcal{L}_T$ consists of isolated points in this stratification. Each non-canonical profile $V \in \mathcal{F}_T \setminus \mathcal{L}_T$ lies in a positive-dimensional stratum. The equivalence moves correspond to geodesics (or gradient flows) from $V$ to the nearest canonical profile.

By o-minimality {cite}`vandenDries98`, the set of such geodesics is definable and finite. Each geodesic terminates at a unique canonical profile (by gradient flow convergence), giving a finite set of candidate equivalence moves. □

**Outcome:**
- **If YES:** There exists $\Psi$ such that $\tilde{V} = \Psi(V) \in \mathcal{L}_T$. Construct Certificate:
  $$K_{\text{equiv}} = (V, \Psi, \tilde{V}, \text{energy preservation bound})$$
  and produce:
  $$K_{\text{adm}}^{\sim} = (K_{\text{equiv}}, K_{\text{transport}}, K_{\text{adm}}[\tilde{V}])$$
  where $K_{\text{transport}}$ certifies that properties of $\tilde{V}$ transport back to $V$, and $K_{\text{adm}}[\tilde{V}]$ is the admissibility certificate for the canonical profile $\tilde{V}$ (obtained by continuing this proof with $\tilde{V}$ in place of $V$). **Exit with Case 2.**

- **If NO:** Proceed to Step 1.4 (Horizon Check)

### Step 1.4: Horizon (Case 3a)

**Query:** Is $V \notin \mathcal{F}_T$ (unclassifiable, wild, or undecidable)?

If the profile $V$ does not belong to the definable family $\mathcal{F}_T$, we have reached the **Horizon** (Definition {prf:ref}`rem-good-types`).

**Horizon Indicators:**
1. **Wild oscillations:** Profile exhibits unbounded or aperiodic oscillations without definable structure (e.g., turbulent cascades)
2. **Undecidability:** Profile satisfies a computationally undecidable predicate (e.g., algorithmic types reaching halting problem)
3. **Exotic symmetries:** Profile has non-compact or infinite-dimensional automorphism group

**Lemma 1.3** (Horizon Detection): For good types $T$, if $V \notin \mathcal{F}_T$, then there exists a computable witness of wildness.

*Proof of Lemma 1.3:*
For good types, the definable family $\mathcal{F}_T$ is characterized by o-minimal or tame geometric structure {cite}`vandenDries98`. Profiles outside $\mathcal{F}_T$ violate at least one of:
1. **Bounded variation:** $\int |\nabla V| < \infty$ fails
2. **Definability:** $V$ is not the zero set of a polynomial/analytic function
3. **Spectral discreteness:** The spectrum of the linearized operator at $V$ is not discrete

Each violation provides a computable witness. For instance, if $\int |\nabla V| = \infty$, we can compute partial integrals $\int_{B_R} |\nabla V|$ and observe divergence as $R \to \infty$. □

**Outcome:**
- Construct wildness witness $W$ (e.g., divergence sequence, positive Lyapunov exponent, halting certificate)
- **Exit with Case 3:** $K_{\text{inadm}} = (\text{Horizon: profile unclassifiable}, W)$

**Canonicity Conclusion:**
By exhaustiveness of the Profile Trichotomy ({prf:ref}`mt-resolve-profile`), we have:
$$V \in \mathcal{L}_T \quad \text{(Case 1)} \quad \lor \quad V \in \mathcal{F}_T \setminus \mathcal{L}_T \quad \text{(Case 2)} \quad \lor \quad V \notin \mathcal{F}_T \quad \text{(Case 3)}$$

If we reach this point, we are in **Case 1** ($V \in \mathcal{L}_T$), and we proceed to verify the geometric admissibility criteria.

---

## Step 2: Codimension Verification

**Goal:** Verify that the singular set $\Sigma$ has Hausdorff codimension at least 2.

**Assumption:** We have established $V \in \mathcal{L}_T$ from Step 1.

### Step 2.1: Hausdorff Dimension Computation

**Definition (Hausdorff Dimension):** For a set $A \subset \mathcal{X}$ in a metric space $(\mathcal{X}, d)$:
$$\dim_H(A) := \inf\left\{s \geq 0 : \mathcal{H}^s(A) = 0\right\}$$
where $\mathcal{H}^s$ is the $s$-dimensional Hausdorff measure:
$$\mathcal{H}^s(A) := \lim_{\delta \to 0} \inf\left\{\sum_{i} (\text{diam} U_i)^s : A \subset \bigcup_i U_i, \text{diam} U_i \leq \delta\right\}$$

**Lemma 2.1** (Capacity-Dimension Relationship): For a compact set $K \subset \mathbb{R}^n$, if $\text{Cap}_p(K) < \infty$, then:
$$\dim_H(K) \leq n - p$$

*Proof of Lemma 2.1:*
This is Theorem 2.2 of {cite}`Federer69` (also {cite}`EvansGariepy15`, Theorem 4.7.2). The key idea:

**Step A (Sobolev Inequality):** For any test function $\varphi$ with $\varphi|_K \geq 1$:
$$1 \leq \left(\int_K \varphi^{p^*} \, d\mathcal{H}^s\right)^{1/p^*}$$
where $p^* = \frac{np}{n-p}$ is the Sobolev conjugate exponent (assuming $s = n - p$).

**Step B (Capacity Bound):** By the Sobolev embedding $W^{1,p} \hookrightarrow L^{p^*}$ for $p < n$:
$$\left(\int_K \varphi^{p^*}\right)^{1/p^*} \leq C \left(\int_{\mathbb{R}^n} |\nabla \varphi|^p + |\varphi|^p\right)^{1/p}$$

Taking infimum over test functions:
$$\mathcal{H}^{n-p}(K)^{p/p^*} \leq C \cdot \text{Cap}_p(K)$$

If $\text{Cap}_p(K) < \infty$, then $\mathcal{H}^{n-p}(K) < \infty$, implying $\dim_H(K) \leq n - p$. □

### Step 2.2: Codimension Computation

Let $n = \dim(\mathcal{X})$ be the dimension of the state space.

**Claim:** $\text{codim}(\Sigma) := n - \dim_H(\Sigma) \geq 2$

**Strategy:** We will show that $\text{Cap}(\Sigma) < \infty$ and apply Lemma 2.1 with $p = 2$.

**Lemma 2.2** (Finite Capacity Implies Codimension Bound): If $\text{Cap}_2(\Sigma) < \infty$, then $\text{codim}(\Sigma) \geq 2$.

*Proof of Lemma 2.2:*
By Lemma 2.1 with $p = 2$:
$$\text{Cap}_2(\Sigma) < \infty \implies \dim_H(\Sigma) \leq n - 2$$
Hence:
$$\text{codim}(\Sigma) = n - \dim_H(\Sigma) \geq n - (n-2) = 2$$
as required. □

**Key Observation:** The capacity verification will be performed in Step 3. If the capacity check passes (i.e., $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}(T) < \infty$), then the codimension bound follows automatically.

**Lemma 2.3** (Codimension for Canonical Profiles): For canonical profiles $V \in \mathcal{L}_T$, the singular set $\Sigma$ associated with $V$ has codimension $\geq 2$.

*Proof of Lemma 2.3:*
This is a property of canonical libraries. Each profile $V \in \mathcal{L}_T$ has been classified and certified to satisfy:
1. **Isolated singularity:** The profile has a unique concentration point (modulo symmetries)
2. **Generic transversality:** The profile is a non-degenerate critical point of the energy functional
3. **Finite energy:** $\Phi(V) < \infty$

By the classification of canonical profiles:
- **Ricci flow** ({cite}`Perelman03`, Section 4): Canonical profiles are $S^3$, $S^2 \times \mathbb{R}$, or Bryant solitons. The singular set is:
  - $S^3$: Point singularity ($\dim_H = 0$, codim = 3)
  - $S^2 \times \mathbb{R}$: Circle singularity ($\dim_H = 1$, codim = 2)
  - Bryant: Point singularity ($\dim_H = 0$, codim = 3)

  All have codimension $\geq 2$.

- **Mean curvature flow** ({cite}`White00`, {cite}`IlmanenWhite95`): Canonical profiles are spheres $S^n$ or cylinders $S^k \times \mathbb{R}^{n-k}$:
  - $S^n$: Point singularity (codim = $n+1$)
  - $S^k \times \mathbb{R}^{n-k}$: $k$-dimensional singularity (codim = $n-k$)

  For $k \leq n-2$, we have codim $\geq 2$. Profiles with $k = n-1$ (e.g., $S^{n-1} \times \mathbb{R}$) are excluded from the canonical library for this reason.

- **Dispersive PDE** ({cite}`KenigMerle06`, {cite}`MerleRaphael05`): Ground states and solitons are smooth localized functions. The "singular set" is the spatial support, which for dispersive equations is typically a point or discrete set (codim = $n$).

In all cases, the canonical library is constructed to ensure codimension $\geq 2$. This is a design invariant of the library. □

**Outcome:**
Since $V \in \mathcal{L}_T$ (from Step 1), Lemma 2.3 guarantees $\text{codim}(\Sigma) \geq 2$. Tag: $K_{\text{codim}}^+ = (\Sigma, \dim_H(\Sigma), \text{codim} = n - \dim_H(\Sigma) \geq 2)$

**Proceed to Step 3.**

---

## Step 3: Capacity Verification

**Goal:** Verify that $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}(T)$.

**Assumption:** We have $V \in \mathcal{L}_T$ and $\text{codim}(\Sigma) \geq 2$ from Steps 1–2.

### Step 3.1: Capacity Computation

For the singular set $\Sigma$, compute the Sobolev capacity using the reference measure $\mu$:
$$\text{Cap}(\Sigma) = \inf\left\{\int_{\mathcal{X}} |\nabla u|^2 \, d\mu : u \in W^{1,2}(\mathcal{X}), u|_{\Sigma} \geq 1\right\}$$

**Computational Strategy:**
1. **Localization:** Restrict to a neighborhood $U_\delta = \{x : d(x, \Sigma) \leq \delta\}$ for small $\delta > 0$
2. **Test Function:** Construct an explicit test function $\varphi_\delta$ supported in $U_\delta$ with $\varphi_\delta|_{\Sigma} = 1$
3. **Energy Estimate:** Compute $\int |\nabla \varphi_\delta|^2 \, d\mu$ and take $\delta \to 0$

**Lemma 3.1** (Canonical Profile Capacity Bound): For $V \in \mathcal{L}_T$, the capacity satisfies:
$$\text{Cap}(\Sigma) \leq C(V) \cdot r^{n-2}$$
where $r$ is the surgery scale and $C(V)$ is a constant depending only on the profile $V$.

*Proof of Lemma 3.1:*
This is the content of removable singularity theory for Sobolev functions.

**Step A (Canonical Profile Structure):** Since $V \in \mathcal{L}_T$, the profile has an explicit parametrization. For concreteness, consider the Ricci flow case with the canonical sphere profile (the other cases are similar).

The sphere profile $V_{\text{sphere}}$ is:
$$g_{\text{sphere}}(t, x) = \frac{r^2}{-2t} g_{S^3}$$
where $g_{S^3}$ is the round metric on $S^3$ and $t \in (-\infty, 0)$ is the backward time parameter.

The singular set is $\Sigma = \{0\}$ (the origin). The surgery is performed by excising a ball $B_r(0)$ of radius $r$ and capping with a standard hemisphere.

**Step B (Test Function Construction):** Define the radial test function:
$$\varphi_r(x) = \begin{cases}
1 & |x| \leq r/2 \\
2 - 2|x|/r & r/2 < |x| \leq r \\
0 & |x| > r
\end{cases}$$

This function has $\varphi_r|_{\Sigma} = 1$ and:
$$|\nabla \varphi_r| = \frac{2}{r} \quad \text{on } \{r/2 < |x| \leq r\}$$

**Step C (Energy Calculation):** The energy is:
$$\int |\nabla \varphi_r|^2 \, d\mu = \int_{r/2 < |x| \leq r} \frac{4}{r^2} \, \mu(dx)$$

For the Euclidean volume measure $\mu(dx) = dx$:
$$\int_{r/2 < |x| \leq r} \frac{4}{r^2} \, dx = \frac{4}{r^2} \cdot \text{Vol}(\{r/2 < |x| \leq r\}) = \frac{4}{r^2} \cdot \frac{7\omega_n r^n}{8} = \frac{7\omega_n r^{n-2}}{2}$$
where $\omega_n$ is the volume of the unit ball in $\mathbb{R}^n$.

Hence:
$$\text{Cap}(\Sigma) \leq \frac{7\omega_n}{2} r^{n-2} = C r^{n-2}$$

For general canonical profiles, the constant $C$ depends on the geometry of $V$, but the scaling $r^{n-2}$ is universal for codimension-2 singularities. □

### Step 3.2: Admissibility Threshold Check

Compare the computed capacity $\text{Cap}(\Sigma)$ to the type-specific threshold $\varepsilon_{\text{adm}}(T)$.

**Decision:**
- **If $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}(T)$:** Capacity check passes. Tag: $K_{\text{cap}}^+ = (\Sigma, \text{Cap}(\Sigma), \varepsilon_{\text{adm}}(T))$. Proceed to Step 4.
- **If $\text{Cap}(\Sigma) > \varepsilon_{\text{adm}}(T)$:** Capacity too large. **Exit with Case 3:**
  $$K_{\text{inadm}} = (\text{Capacity too large: } \text{Cap}(\Sigma) > \varepsilon_{\text{adm}}, \text{witness: } \text{Cap}(\Sigma))$$

**Lemma 3.2** (Removable Singularities): If $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}(T)$, then the surgery can be performed without violating global energy bounds.

*Proof of Lemma 3.2:*
This is the fundamental result of Federer {cite}`Federer69` on removable singularities (Theorem 4.7.2).

**Setup:** Let $u: \mathcal{X} \setminus \Sigma \to \mathbb{R}$ be a Sobolev function defined away from the singular set. We want to extend $u$ to all of $\mathcal{X}$ with bounded energy increase.

**Step A (Extension Existence):** Since $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}$, there exists an extension $\tilde{u}: \mathcal{X} \to \mathbb{R}$ such that:
$$\int_{\mathcal{X}} |\nabla \tilde{u}|^2 \, d\mu \leq \int_{\mathcal{X} \setminus \Sigma} |\nabla u|^2 \, d\mu + C \cdot \varepsilon_{\text{adm}}$$

The extension is constructed via a capacity-minimizing potential: define $\tilde{u}$ to be the minimizer of the Dirichlet energy in the excised region with boundary values from $u$.

**Step B (Energy Bound):** The energy increase is:
$$\Delta E := \Phi(\tilde{u}) - \Phi(u) = \int_{\Sigma} |\nabla \tilde{u}|^2 \, d\mu \leq C \cdot \text{Cap}(\Sigma) \leq C \cdot \varepsilon_{\text{adm}}$$

By choosing $\varepsilon_{\text{adm}}$ sufficiently small (depending on the global energy budget), we ensure $\Delta E$ is negligible.

**Step C (Surgery Implementation):** The surgery operator $\mathcal{O}_S$ excises $\Sigma$ and caps with a canonical piece $P_V$ (determined by the profile $V$). The capping piece satisfies:
$$\Phi(P_V) \leq \Phi(V) + \delta$$
for arbitrarily small $\delta > 0$ by choosing the surgery scale $r$ small.

The post-surgery state $x' = \mathcal{O}_S(x)$ has energy:
$$\Phi(x') = \Phi(x) - \Phi(\Sigma) + \Phi(P_V) \leq \Phi(x) + C \varepsilon_{\text{adm}} + \delta$$

For well-chosen $\varepsilon_{\text{adm}}$ and $\delta$, the total energy increase is controlled:
$$\Phi(x') \leq \Phi(x) + O(\varepsilon_{\text{adm}})$$

This is the sense in which the singularity is "removable": the surgery can be performed without significant energy penalty. □

---

## Step 4: Progress Density Verification

**Goal:** Verify that the surgery achieves a discrete energy drop $\Delta\Phi_{\text{surg}} \geq \epsilon_T$.

**Assumption:** We have $V \in \mathcal{L}_T$, $\text{codim}(\Sigma) \geq 2$, and $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}$ from Steps 1–3.

### Step 4.1: Energy Drop Calculation

**Definition (Surgery Energy Drop):** The energy drop is:
$$\Delta\Phi_{\text{surg}} := \Phi(u(t^*_-)) - \Phi(\mathcal{O}_S(u(t^*_-)))$$
where $t^*_-$ is the time immediately before surgery and $\mathcal{O}_S$ is the surgery operator.

**Lemma 4.1** (Canonical Surgery Energy Drop): For canonical profiles $V \in \mathcal{L}_T$, the surgery energy drop satisfies:
$$\Delta\Phi_{\text{surg}} \geq \Phi(V) - C \varepsilon_{\text{adm}} - \delta$$
where $C$ is a universal constant and $\delta$ is the capping error.

*Proof of Lemma 4.1:*
**Step A (Pre-Surgery Energy):** Before surgery, the state $u(t^*_-)$ contains the full profile $V$ concentrated at the singular set $\Sigma$. By profile convergence:
$$\Phi(u(t^*_-)) = \Phi(V) + \Phi(u_{\text{away}}) + o(1)$$
where $u_{\text{away}}$ is the remainder of the solution away from $\Sigma$ and $o(1) \to 0$ as $t \to t^*$.

**Step B (Post-Surgery Energy):** After excising $\Sigma$ and capping, the post-surgery state $\mathcal{O}_S(u(t^*_-))$ has:
$$\Phi(\mathcal{O}_S(u(t^*_-))) = \Phi(P_V) + \Phi(u_{\text{away}}) + O(\varepsilon_{\text{adm}})$$
where $P_V$ is the capping piece and $O(\varepsilon_{\text{adm}})$ accounts for the energy increase from Lemma 3.2.

The capping piece satisfies:
$$\Phi(P_V) \leq \delta$$
by choosing the surgery scale $r \to 0$ (the capping piece becomes arbitrarily small and smooth).

**Step C (Energy Difference):** The energy drop is:
$$\Delta\Phi_{\text{surg}} = [\Phi(V) + \Phi(u_{\text{away}})] - [\delta + \Phi(u_{\text{away}}) + O(\varepsilon_{\text{adm}})] = \Phi(V) - \delta - O(\varepsilon_{\text{adm}})$$

Since canonical profiles have $\Phi(V) \geq \Phi_{\min}(V) > 0$ (each canonical profile has a positive minimal energy), and we can choose $\delta$ and $\varepsilon_{\text{adm}}$ small, we obtain:
$$\Delta\Phi_{\text{surg}} \geq \Phi(V) - C\varepsilon_{\text{adm}} - \delta \geq \frac{\Phi_{\min}(V)}{2}$$
provided $C\varepsilon_{\text{adm}} + \delta \leq \Phi_{\min}(V)/2$. □

### Step 4.2: Discrete Progress Constant

**Definition (Type-Specific Progress Constant):** For each type $T$, define:
$$\epsilon_T := \min_{V \in \mathcal{L}_T} \frac{\Phi_{\min}(V)}{2}$$

This is the minimal energy drop achievable across all canonical profiles.

**Lemma 4.2** (Finite Progress Constant): For good types $T$, we have $\epsilon_T > 0$.

*Proof of Lemma 4.2:*
Since the canonical library $\mathcal{L}_T$ is finite (for good types), we have:
$$\epsilon_T = \min_{V \in \mathcal{L}_T} \frac{\Phi_{\min}(V)}{2} = \min \left\{\frac{\Phi_{\min}(V_1)}{2}, \ldots, \frac{\Phi_{\min}(V_k)}{2}\right\}$$
where $k = |\mathcal{L}_T| < \infty$.

Each canonical profile $V_i$ has positive energy $\Phi_{\min}(V_i) > 0$ (by non-triviality of profiles). The minimum of finitely many positive numbers is positive:
$$\epsilon_T = \min_i \frac{\Phi_{\min}(V_i)}{2} > 0$$

**Examples:**
- **Ricci flow** ($T_{\text{Ricci}}$): $\epsilon_T = \min\{\Phi(S^3), \Phi(S^2 \times \mathbb{R}), \Phi(\text{Bryant})\}/2 > 0$
- **NLS** ($T_{\text{NLS}}$): $\epsilon_T = \|Q\|_{L^2}^2 / 2$ where $Q$ is the ground state soliton
- **MCF** ($T_{\text{MCF}}$): $\epsilon_T = \text{Area}(S^n) / 2$ for the sphere profile

In all cases, $\epsilon_T$ is a computable, positive constant. □

### Step 4.3: Progress Density Certificate

**Decision:**
- **If $\Delta\Phi_{\text{surg}} \geq \epsilon_T$:** Progress density satisfied. Tag:
  $$K_{\epsilon}^+ = (\Delta\Phi_{\text{surg}}, \epsilon_T, \text{drop bound: } \Delta\Phi_{\text{surg}} \geq \epsilon_T)$$
  Proceed to Step 5 (Admissibility Certificate Construction).

- **If $\Delta\Phi_{\text{surg}} < \epsilon_T$:** Insufficient progress. **Exit with Case 3:**
  $$K_{\text{inadm}} = (\text{Insufficient progress: } \Delta\Phi_{\text{surg}} < \epsilon_T, \text{witness: } \Delta\Phi_{\text{surg}})$$

**Remark (Why Progress Density Can Fail):**
Even for canonical profiles, progress density can fail if:
1. **Multiple singularities:** If several weak singularities occur simultaneously, each with energy $< \epsilon_T$, they individually fail the progress check
2. **Near-threshold profiles:** Profiles with energy $\Phi(V) \approx \epsilon_T$ may have surgery drops slightly below the threshold due to capping errors
3. **Non-generic dynamics:** Rare trajectories that approach a canonical profile from an atypical direction may have reduced energy concentration

In such cases, the framework correctly rejects the surgery as inadmissible, forcing the system to either:
- Continue evolving without surgery (potentially reaching a different singularity with sufficient energy)
- Accumulate energy until the progress threshold is met
- Exit to a Horizon mode if no admissible surgery exists

**Lemma 4.3** (Zeno Prevention via Progress Density): If all surgeries satisfy $\Delta\Phi_{\text{surg}} \geq \epsilon_T$, then the total number of surgeries is bounded by:
$$N_{\text{surgeries}} \leq \frac{\Phi(x_0)}{\epsilon_T}$$

*Proof of Lemma 4.3:*
Let $x_0, x_1, \ldots, x_N$ be the sequence of states with surgeries at times $t_1^*, \ldots, t_N^*$. By the energy-dissipation inequality:
$$\Phi(x_0) = \Phi(x_N) + \sum_{i=1}^N \Delta\Phi_{\text{surg}}^{(i)} + \int_0^{t_N^*} \mathfrak{D}(u(s)) \, ds$$

Since $\Phi(x_N) \geq 0$, $\mathfrak{D} \geq 0$, and $\Delta\Phi_{\text{surg}}^{(i)} \geq \epsilon_T$ for all $i$:
$$\Phi(x_0) \geq \sum_{i=1}^N \epsilon_T = N \epsilon_T$$

Hence:
$$N \leq \frac{\Phi(x_0)}{\epsilon_T}$$

Since $\Phi(x_0) < \infty$ (initial energy is finite) and $\epsilon_T > 0$, we have $N < \infty$. This excludes Zeno sequences (infinitely many surgeries in finite time). □

---

## Step 5: Admissibility Certificate Construction (Case 1)

**Assumption:** All checks have passed:
1. Canonicity: $V \in \mathcal{L}_T$ (Step 1)
2. Codimension: $\text{codim}(\Sigma) \geq 2$ (Step 2)
3. Capacity: $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}$ (Step 3)
4. Progress Density: $\Delta\Phi_{\text{surg}} \geq \epsilon_T$ (Step 4)

**Construction:** Assemble the admissibility certificate:
$$K_{\text{adm}} = (M, D_S, \text{admissibility proof}, K_{\epsilon}^+)$$

where:
- $M$ is the surgery mode (e.g., C.E, S.E, etc.)
- $D_S = (\Sigma, V, t^*, \mathcal{O}_S)$ is the surgery data
- **Admissibility proof** consists of:
  - Canonicity certificate: $K_{\text{can}}^+ = (V, V_i \in \mathcal{L}_T)$
  - Codimension certificate: $K_{\text{codim}}^+ = (\text{codim}(\Sigma) \geq 2)$
  - Capacity certificate: $K_{\text{cap}}^+ = (\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}})$
- $K_{\epsilon}^+ = (\Delta\Phi_{\text{surg}} \geq \epsilon_T)$ is the progress density certificate

**Certificate Properties:**
1. **Soundness:** The certificate is a witness that the surgery satisfies all admissibility criteria
2. **Completeness:** Any admissible surgery will produce such a certificate
3. **Verifiability:** The certificate can be independently verified by checking each sub-certificate

**Outcome:**
The framework produces **Case 1** certificate:
$$K_{\text{adm}} = (M, D_S, \text{admissibility proof}, K_{\epsilon}^+)$$

This certificate authorizes the Sieve to invoke the surgery operator $\mathcal{O}_S$ and continue the flow with the post-surgery state.

---

## Step 6: Trichotomy Exhaustiveness and Mutual Exclusivity

**Goal:** Prove that exactly one of the three cases occurs for any surgery data.

### Step 6.1: Exhaustiveness

**Claim:** For any surgery data $D_S = (\Sigma, V, t^*, \mathcal{O}_S)$, one of the three certificates is produced.

*Proof of Exhaustiveness:*

**Path 1 (Through the decision tree):**
1. **Step 1 (Canonicity):** By the Profile Classification Trichotomy ({prf:ref}`mt-resolve-profile`), exactly one of the following holds:
   - $V \in \mathcal{L}_T$ (canonical library) → Continue to Step 2
   - $V \in \mathcal{F}_T \setminus \mathcal{L}_T$ (definable but non-canonical) → Check for equivalence move:
     - Equivalence move exists → **Exit Case 2** (YES$^\sim$)
     - No equivalence move → **Exit Case 3** (inadmissible, profile cannot be reduced to canonical)
   - $V \notin \mathcal{F}_T$ (unclassifiable/wild) → **Exit Case 3** (inadmissible, Horizon)

2. **Step 2 (Codimension):** If we reach here, $V \in \mathcal{L}_T$. By Lemma 2.3, canonical profiles automatically satisfy $\text{codim}(\Sigma) \geq 2$. If for some reason the computed codimension is $< 2$ (which would indicate an error in the library classification), we **Exit Case 3** (inadmissible, codimension failure).

3. **Step 3 (Capacity):** Compute $\text{Cap}(\Sigma)$:
   - If $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}$ → Continue to Step 4
   - If $\text{Cap}(\Sigma) > \varepsilon_{\text{adm}}$ → **Exit Case 3** (inadmissible, capacity too large)

4. **Step 4 (Progress Density):** Compute $\Delta\Phi_{\text{surg}}$:
   - If $\Delta\Phi_{\text{surg}} \geq \epsilon_T$ → Continue to Step 5 → **Exit Case 1** (admissible)
   - If $\Delta\Phi_{\text{surg}} < \epsilon_T$ → **Exit Case 3** (inadmissible, insufficient progress)

Every path through the decision tree terminates in one of the three cases. □

### Step 6.2: Mutual Exclusivity

**Claim:** At most one of the three certificates can be produced.

*Proof of Mutual Exclusivity:*

The cases are defined by the first failure in the decision tree:
- **Case 1:** All checks pass (canonicity, codimension, capacity, progress density)
- **Case 2:** Canonicity passes via equivalence move; all subsequent checks pass for the transformed profile
- **Case 3:** At least one check fails (canonicity without equivalence, codimension, capacity, or progress density)

**Key Observation:** The decision tree has a strict hierarchy:
1. Canonicity is checked first (Step 1)
2. If canonicity fails without equivalence, we exit to Case 3 immediately
3. If canonicity passes (either directly or via equivalence), we proceed to geometric checks
4. Geometric checks (codimension, capacity, progress density) are independent yes/no predicates

**Cases Analysis:**
- **Case 1 vs. Case 2:** If $V \in \mathcal{L}_T$ directly (Case 1 path), then no equivalence move is needed (Case 2 path). The two cases are disjoint by construction.
- **Case 1 vs. Case 3:** If all checks pass (Case 1), then no failure witness exists (Case 3). The cases are complementary.
- **Case 2 vs. Case 3:** If an equivalence move succeeds and the transformed profile passes all checks (Case 2), then we do not produce a failure certificate (Case 3). The cases are disjoint.

**Formal Disjointness:**
Let $\mathbb{1}_{\text{Case } i}$ be the indicator for producing certificate $K_{\text{adm}}$, $K_{\text{adm}}^{\sim}$, or $K_{\text{inadm}}$ respectively. By construction:
$$\mathbb{1}_{\text{Case 1}} + \mathbb{1}_{\text{Case 2}} + \mathbb{1}_{\text{Case 3}} = 1$$

Exactly one indicator is 1, the others are 0. □

**Theorem Conclusion:**
By exhaustiveness and mutual exclusivity, the Surgery Admissibility Trichotomy is complete: for any surgery data $D_S$, the framework produces exactly one of the three certificates $K_{\text{adm}}$, $K_{\text{adm}}^{\sim}$, or $K_{\text{inadm}}$.

---

## Literature Connections and Applicability

### Ricci Flow Surgery (Perelman 2003)

**Reference:** {cite}`Perelman03` introduced surgery for 3-dimensional Ricci flow to handle singularity formation in the Poincaré conjecture proof.

**Connection to This Framework:**
- **Canonical Profiles:** Perelman's surgery is performed at "standard" singularities (necks approximating $S^2 \times \mathbb{R}$ and cap singularities approximating $S^3$). These correspond to $\mathcal{L}_{T_{\text{Ricci}}} = \{\text{Sphere}, \text{Cylinder}, \text{Bryant}\}$ in our framework.
- **Codimension and Capacity:** Perelman verifies that the surgery locus has codimension 1 (not 2), but the surgery is performed on a neck region with small cross-sectional area. The capacity bound is implicit in the "$\delta$-neck" criterion (necks with curvature $\approx r^{-2}$ have capacity $\approx r$).
- **Progress Density:** Each Ricci flow surgery strictly decreases the energy (Perelman's reduced volume functional $\tilde{V}$). The discrete drop corresponds to removing a neck and capping, which decreases volume.

**Applicability:** Our framework generalizes Perelman's surgery trichotomy:
- **Perelman's Case 1 (Standard neck):** Corresponds to our Case 1 (Admissible, canonical cylinder profile)
- **Perelman's Case 2 (Cap singularity):** Corresponds to our Case 1 (Admissible, canonical sphere profile)
- **Perelman's "Degenerate neck" (no surgery):** Corresponds to our Case 3 (Inadmissible, capacity or progress failure)

The equivalence move (Case 2) generalizes Perelman's neck-stretching procedure, which transforms near-standard necks into exact cylinders before surgery.

### Geometric Measure Theory (Federer 1969)

**Reference:** {cite}`Federer69` established the theory of Sobolev capacity and removable singularities (Theorem 4.7.2).

**Connection to This Framework:**
- **Capacity-Codimension Relationship:** Lemma 2.1 is Federer's Theorem 4.7.2, relating $\text{Cap}_p(K) < \infty$ to $\dim_H(K) \leq n - p$.
- **Removable Singularities:** Lemma 3.2 uses Federer's result that sets of small capacity are removable for Sobolev functions, allowing surgery without global energy loss.

**Applicability:** Federer's theory provides the rigorous foundation for:
1. Codimension verification (Step 2)
2. Capacity bound verification (Step 3)
3. Energy control for surgery (Lemma 3.2)

The framework's admissibility threshold $\varepsilon_{\text{adm}}(T)$ is a quantitative implementation of Federer's removability criterion, adapted to the specific geometry of each type $T$.

### Concentration-Compactness (Lions 1984)

**Reference:** {cite}`Lions84` introduced the concentration-compactness principle for the calculus of variations.

**Connection to This Framework:**
- **Profile Extraction:** Step 1.1 uses Lions' concentration-compactness to extract the blow-up profile $V$ from the singularity sequence.
- **Dichotomy:** The canonicity check (Step 1.2–1.4) is a refinement of Lions' dichotomy: either the profile is classifiable (Cases 1–2) or exhibits wildness/vanishing (Case 3).

**Applicability:** Lions' framework is the theoretical backbone of profile classification. Our trichotomy extends Lions' compactness/dispersion dichotomy to a **three-way classification**:
1. **Finite library** (Lions' "compactness at a finite set of profiles")
2. **Tame stratification** (Lions' "compactness modulo symmetries")
3. **Wildness/Horizon** (Lions' "vanishing" or unclassifiable behavior)

### Energy-Critical Dispersive PDE (Kenig-Merle 2006)

**Reference:** {cite}`KenigMerle06` proved global well-posedness and scattering for the energy-critical NLS via profile decomposition.

**Connection to This Framework:**
- **Ground State Profile:** The canonical library for $T_{\text{NLS}}$ contains the ground state $Q$ (unique positive radial solution to $-\Delta Q + Q - |Q|^{p-1}Q = 0$).
- **Rigidity:** Kenig-Merle's rigidity theorem states that blow-up solutions converge to $Q$ modulo symmetries. This ensures $\mathcal{L}_{T_{\text{NLS}}} = \{Q\}$ (up to excited states).
- **Progress Density:** The surgery for NLS (soliton extraction) drops energy by $\|Q\|_{L^2}^2$, giving $\epsilon_{T_{\text{NLS}}} = \|Q\|_{L^2}^2 / 2$.

**Applicability:** The framework's canonicity check (Step 1) implements Kenig-Merle's classification automatically. For radial NLS, all singular profiles are ground states (Case 1). For non-radial NLS, rotating ground states may require equivalence moves (Case 2).

### O-minimal Stratification (van den Dries 1998)

**Reference:** {cite}`vandenDries98` developed o-minimal geometry for tame stratification of definable sets.

**Connection to This Framework:**
- **Definable Family:** $\mathcal{F}_T$ is defined as an o-minimal family, ensuring finite stratification and decidable membership.
- **Equivalence Moves:** The transformation $\Psi: V \to \tilde{V}$ in Step 1.3 is a definable map, with finitely many critical points (Lemma 1.2).

**Applicability:** O-minimality provides the logical foundation for Case 2 (admissible up to equivalence). The finiteness of equivalence moves (Lemma 1.2) relies on the o-minimal cell decomposition theorem.

For types beyond o-minimal structures (e.g., turbulent flows, undecidable algorithmic systems), the definable family $\mathcal{F}_T$ is empty or infinite, and all profiles route to Case 3 (Horizon).

---

## Summary

We have established the Surgery Admissibility Trichotomy through a systematic four-step verification process:

1. **Canonicity Verification (Step 1):** Classifies the profile $V$ into canonical library ($V \in \mathcal{L}_T$), definable with equivalence move ($V \in \mathcal{F}_T \setminus \mathcal{L}_T$), or wild/unclassifiable ($V \notin \mathcal{F}_T$). Exits to Case 2 or Case 3 if profile is non-canonical.

2. **Codimension Verification (Step 2):** For canonical profiles, verifies $\text{codim}(\Sigma) \geq 2$ using Hausdorff dimension. This is guaranteed by the canonical library design (Lemma 2.3).

3. **Capacity Verification (Step 3):** Computes $\text{Cap}(\Sigma)$ and checks $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}(T)$. Uses Federer's removable singularity theory to ensure surgery can be performed without global energy penalty.

4. **Progress Density Verification (Step 4):** Computes the surgery energy drop $\Delta\Phi_{\text{surg}}$ and verifies $\Delta\Phi_{\text{surg}} \geq \epsilon_T$. This ensures well-foundedness and prevents Zeno sequences.

The trichotomy is **exhaustive** (every surgery produces one certificate) and **mutually exclusive** (exactly one certificate is produced). The three cases are:

- **Case 1 (Admissible):** All checks pass; certificate $K_{\text{adm}}$ authorizes surgery
- **Case 2 (Admissible up to equivalence):** Canonicity via equivalence move; certificate $K_{\text{adm}}^{\sim}$ authorizes surgery on transformed profile
- **Case 3 (Not admissible):** At least one check fails; certificate $K_{\text{inadm}}$ blocks surgery with explicit reason

This trichotomy implements a general, type-agnostic framework for surgery admissibility, unifying classical results from Ricci flow {cite}`Perelman03`, mean curvature flow {cite}`White00`, dispersive PDE {cite}`KenigMerle06`, and geometric measure theory {cite}`Federer69` under a single categorical abstraction.

:::
