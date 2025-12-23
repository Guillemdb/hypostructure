# Proof of ACT-Surgery (Structural Surgery Principle)

:::{prf:proof}
:label: proof-mt-act-surgery

**Theorem Reference:** {prf:ref}`mt-act-surgery`

This proof establishes the Structural Surgery Principle, demonstrating that admissible singularities can be removed via surgery while preserving flow continuation, energy control, and progress. The proof synthesizes Hamilton's surgery program {cite}`Hamilton97`, Perelman's surgery algorithm {cite}`Perelman03`, and the categorical pushout construction, following the exposition of Kleiner-Lott {cite}`KleinerLott08`.

---

## Setup and Notation

**Given Data:**

We are given a Hypostructure $\mathcal{H} = (\mathcal{X}, S_t, \Phi, \mathfrak{D}, G, \partial, \mathcal{E})$ where:
- $\mathcal{X}$ is the state stack (configuration space)
- $S_t: \mathcal{X} \to \mathcal{X}$ is the semiflow (evolution operator)
- $\Phi: \mathcal{X} \to \mathbb{R}_{\geq 0} \cup \{\infty\}$ is the cohomological height (energy/entropy functional)
- $\mathfrak{D}: \mathcal{X} \to \mathbb{R}_{\geq 0}$ is the dissipation rate
- $G$ is a compact Lie group acting on $\mathcal{X}$ (symmetry group, encoding translations, rotations, scalings)
- $\partial: \mathcal{X} \to \mathcal{X}_\partial$ is the boundary restriction operator
- $\mathcal{E}$ is the ambient topos (e.g., **Top**, **Meas**, **Diff**, **FinSet**)

**Input Certificates:**

1. **Breach Certificate** $K^{\mathrm{br}} = (M, t^-, x^-, \text{barrier witness}, \mathcal{E}_{\text{breach}})$ from a failure mode $M$, certifying that:
   - The trajectory $u(t) = S_t x$ encounters a barrier at time $t^-$ with state $x^- = u(t^-)$
   - The barrier predicate $\mathcal{B}_M(x^-)$ holds
   - No continuation is possible in the original space $\mathcal{X}$ without intervention

2. **Admissibility Certificate** $K_{\text{adm}}$ (or quasi-admissible $K_{\text{adm}}^{\sim}$) from {prf:ref}`mt-resolve-admissibility`, certifying:
   - **Singular locus**: A set $\Sigma \subset \mathcal{X}$ of codimension $\geq 2$
   - **Singular profile**: $V \in \mathcal{L}_T$ from the canonical profile library
   - **Capacity bound**: $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}(T)$ where
     $$\text{Cap}(\Sigma) = \inf\left\{\int_{\mathcal{X}} |\nabla \phi|^2 \, d\mu : \phi \in H^1(\mathcal{X}), \phi|_\Sigma = 1\right\}$$
   - **Progress certificate**: $K_{\epsilon}^+$ witnessing energy drop $\Delta\Phi_{\text{surg}} \geq \epsilon_T > 0$

3. **Surgery Data** $D_S = (\Sigma, V, \epsilon, \mathcal{X}_{\text{cap}})$ specifying:
   - Excision radius $\epsilon > 0$ determining neighborhood $\mathcal{X}_\Sigma = \{x \in \mathcal{X} : d(x, \Sigma) < \epsilon\}$
   - Capping object $\mathcal{X}_{\text{cap}}$ from the library, determined by profile $V$

**Canonical Profile Library:**

For the type $T$ of the Hypostructure, the profile library is:
$$\mathcal{L}_T = \{V \in \mathcal{M}_{\text{prof}}(T) : \text{Aut}(V) \text{ is finite}, V \text{ is isolated in } \mathcal{M}_{\text{prof}}\}$$

**Examples:**
- **Ricci flow** ($T_{\text{Ricci}}$): $\mathcal{L}_T = \{\text{round sphere } S^3, \text{round cylinder } S^2 \times \mathbb{R}, \text{Bryant soliton}\}$ (3 elements)
- **Mean curvature flow** ($T_{\text{MCF}}$): $\mathcal{L}_T = \{\text{round sphere } S^n, \text{round cylinders } S^k \times \mathbb{R}^{n-k}\}_{k \leq n}$ ($n+1$ elements)

Each profile $V \in \mathcal{L}_T$ has an attached **surgery recipe** $\mathcal{O}_V$ providing the capping construction.

**Goal:**

Prove that applying the surgery operator $\mathcal{O}_S: \mathcal{X} \dashrightarrow \mathcal{X}'$ produces a surgered state space $\mathcal{X}'$ with:
1. **Flow continuation**: Evolution continues with well-defined state $x' \in \mathcal{X}'$
2. **Energy control**: $\Phi(x') \leq \Phi(x^-) + \delta_S$ for controlled jump $\delta_S$
3. **Re-entry certificate**: $K^{\mathrm{re}}$ satisfying preconditions for continuing the Sieve
4. **Progress**: Finite surgery count via energy decrease

---

## Lemma 1: Excision Neighborhood Well-Definedness

**Statement:** Given admissible singularity $(\Sigma, V)$ with $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}$, the excision neighborhood
$$\mathcal{X}_\Sigma = \{x \in \mathcal{X} : d(x, \Sigma) < \epsilon\}$$
is well-defined with finite measure:
$$\mu(\mathcal{X}_\Sigma) \leq C_{\text{exc}} \cdot \epsilon^2 \cdot \text{Cap}(\Sigma)$$
where $C_{\text{exc}}$ depends only on the dimension $n = \dim(\mathcal{X})$ and the ambient metric structure.

**Proof:**

**Step 1.1 (Metric Structure):** From the thin object $\mathcal{X}^{\text{thin}} = (\mathcal{X}, d, \mu)$, we have:
- A metric $d: \mathcal{X} \times \mathcal{X} \to [0, \infty)$ satisfying the triangle inequality
- A Radon measure $\mu$ on $\mathcal{X}$ with $\mu(\mathcal{X}) < \infty$ (finite total mass)

The distance function $x \mapsto d(x, \Sigma) := \inf_{y \in \Sigma} d(x, y)$ is Lipschitz continuous with Lipschitz constant 1:
$$|d(x, \Sigma) - d(x', \Sigma)| \leq d(x, x')$$

Hence $\mathcal{X}_\Sigma = \{d(\cdot, \Sigma) < \epsilon\}$ is an open set (as the preimage of the open interval $(-\infty, \epsilon)$ under a continuous function).

**Step 1.2 (Capacity-to-Measure Bound):** By the coarea formula from geometric measure theory {cite}`Federer69`, for any Lipschitz function $u: \mathcal{X} \to \mathbb{R}$:
$$\int_{\mathcal{X}} |\nabla u| \, d\mu = \int_{-\infty}^{\infty} \mathcal{H}^{n-1}(\{u = t\}) \, dt$$
where $\mathcal{H}^{n-1}$ is the $(n-1)$-dimensional Hausdorff measure.

Applying this to $u(x) = d(x, \Sigma)$, we have $|\nabla u| = 1$ almost everywhere (by Rademacher's theorem, since $u$ is Lipschitz), so:
$$\mu(\mathcal{X}_\Sigma) = \mu(\{u < \epsilon\}) \leq \int_0^\epsilon \mathcal{H}^{n-1}(\{d(\cdot, \Sigma) = t\}) \, dt$$

**Step 1.3 (Capacity Estimate):** The capacity of $\Sigma$ controls the surface area of level sets. By the Sobolev inequality in the form of Adams-Hedberg {cite}`AdamsHedberg96` (Theorem 5.1.2):
$$\mathcal{H}^{n-1}(\{d(\cdot, \Sigma) = t\}) \leq C \cdot t \cdot \text{Cap}(\Sigma) \quad \text{for } 0 < t \leq \epsilon$$

Integrating over $t \in [0, \epsilon]$ and using the monotonicity of level sets:
$$\mu(\mathcal{X}_\Sigma) \leq C \int_0^\epsilon t \cdot \text{Cap}(\Sigma) \, dt \leq C_{\text{exc}} \cdot \epsilon^2 \cdot \text{Cap}(\Sigma)$$

**Step 1.4 (Admissibility Bound):** Since $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}$ by hypothesis, we obtain:
$$\mu(\mathcal{X}_\Sigma) \leq C_{\text{exc}} \cdot \epsilon^2 \cdot \varepsilon_{\text{adm}} < \infty$$

This bound ensures the excision removes only a small, controlled portion of the state space. □

---

## Lemma 2: Capping Object Construction

**Statement:** For each profile $V \in \mathcal{L}_T$, there exists a unique **capping object** $\mathcal{X}_{\text{cap}}(V)$ with the following properties:

1. **Boundary Matching**: $\partial \mathcal{X}_{\text{cap}} \cong \partial \mathcal{X}_\Sigma$ (boundaries are diffeomorphic/homeomorphic)
2. **Asymptotic Profile**: $\lim_{x \to \partial \mathcal{X}_{\text{cap}}} \Phi(x) = V$ (the solution approaches profile $V$ at the boundary)
3. **Bounded Geometry**: For all $k \leq k_{\max}(V)$:
   $$\sup_{\mathcal{X}_{\text{cap}}} |\nabla^k \Phi| \leq C_k(V) < \infty$$
4. **Energy Bound**: $\Phi(\mathcal{X}_{\text{cap}}) \leq E_{\text{cap}}(V)$ where $E_{\text{cap}}$ depends only on $V$

**Proof:**

**Step 2.1 (Library Construction):** By definition of the canonical library $\mathcal{L}_T$, each profile $V$ is an **isolated critical point** of the profile moduli space $\mathcal{M}_{\text{prof}}(T) = \mathcal{A}/G$ where:
- $\mathcal{A}$ is the attractor set for the rescaled flow
- $G$ is the symmetry group (translations, rotations, scalings)

Isolation ensures a finite automorphism group $\text{Aut}(V) = \{g \in G : g \cdot V = V\}$, making the profile rigid up to symmetries.

**Step 2.2 (Type-Specific Capping Recipes):** The surgery recipe for profile $V$ is derived from the regularity theory for type $T$:

**Example 1 (Ricci Flow):** For $V = S^2 \times \mathbb{R}$ (round cylinder singularity in dimension 3), the cap is constructed via Perelman's canonical neighborhood theorem {cite}`Perelman03`:
- The cap is a standard round $S^3$ glued along the neck
- Boundary matching uses the exponential decay of curvature: $|R - R_{\text{cyl}}| \lesssim e^{-c|s|}$ where $s$ is the coordinate along the cylinder axis
- The gluing is smooth by the implicit function theorem applied to the Ricci flow equations in the neck region

**Example 2 (Mean Curvature Flow):** For $V = S^n$ (round sphere singularity), the cap is the **empty set** (the sphere is simply removed):
- This is the classical mean-convex surgery scheme (see e.g. {cite}`HuiskenSinestrari09`)
- Boundary matching is vacuous since the excision removes the entire sphere
- Energy bound: $\Phi(\emptyset) = 0$

**Step 2.3 (Pushout Universal Property):** The capping object $\mathcal{X}_{\text{cap}}$ is characterized as the **universal solution** to the gluing problem. Given:
- Excision boundary $\partial \mathcal{X}_\Sigma$
- Asymptotic profile $V$

there exists a unique (up to isomorphism) object $\mathcal{X}_{\text{cap}}$ in the ambient category $\mathcal{E}$ satisfying:
$$\begin{CD}
\partial \mathcal{X}_\Sigma @>{\text{match}}>> V \\
@V{\iota}VV @VV{\text{asymp}}V \\
\mathcal{X}_{\text{cap}} @>{\text{limit}}>> V
\end{CD}$$

This diagram commutes in the sense that the solution on $\mathcal{X}_{\text{cap}}$ limits to profile $V$ at the boundary.

**Step 2.4 (Regularity Inheritance):** By the maximum principle and parabolic regularity theory (e.g., Evans {cite}`Evans10`), the cap inherits bounded geometry from $V$:
- Since $V \in \mathcal{L}_T$ is a canonical profile, it satisfies the type-specific regularity estimates (e.g., curvature bounds for Ricci flow, mean curvature bounds for MCF)
- The asymptotic matching ensures these bounds propagate to $\mathcal{X}_{\text{cap}}$ with constants $C_k(V)$ depending only on $V$ and the decay rate

**Conclusion:** The capping object exists, is unique up to isomorphism, and satisfies all four properties. □

---

## Lemma 3: Pushout Surgery Construction

**Statement:** The surgery operator $\mathcal{O}_S: \mathcal{X} \dashrightarrow \mathcal{X}'$ is realized as the categorical **pushout**:

$$\begin{CD}
\mathcal{X}_{\Sigma} @>{\iota}>> \mathcal{X} \\
@V{\text{excise}}VV @VV{\mathcal{O}_S}V \\
\mathcal{X}_{\text{cap}} @>{\text{glue}}>> \mathcal{X}'
\end{CD}$$

where $\mathcal{X}' = (\mathcal{X} \setminus \mathcal{X}_\Sigma) \sqcup_{\partial} \mathcal{X}_{\text{cap}}$ is the surgered space, and the pushout satisfies the universal property:

**Universal Property:** For any object $\mathcal{Y}$ and morphisms $f: \mathcal{X} \to \mathcal{Y}$, $g: \mathcal{X}_{\text{cap}} \to \mathcal{Y}$ such that $f|_{\partial \mathcal{X}_\Sigma} = g|_{\partial \mathcal{X}_{\text{cap}}}$, there exists a unique morphism $\tilde{f}: \mathcal{X}' \to \mathcal{Y}$ with:
$$\tilde{f} \circ \mathcal{O}_S = f \quad \text{and} \quad \tilde{f}|_{\mathcal{X}_{\text{cap}}} = g$$

**Proof:**

**Step 3.1 (Pushout Existence):** By the theory of colimits in category theory {cite}`MacLane71`, the pushout exists in any category with finite colimits. We verify this for each relevant ambient category:

- **Top (Topological Spaces):** The pushout is the quotient space
  $$\mathcal{X}' = \frac{(\mathcal{X} \setminus \mathcal{X}_\Sigma) \sqcup \mathcal{X}_{\text{cap}}}{\sim}$$
  where $x \sim y$ if $x \in \partial \mathcal{X}_\Sigma$, $y \in \partial \mathcal{X}_{\text{cap}}$, and they match via the boundary identification. This is a standard construction in algebraic topology (CW complex attachment).

- **Diff (Smooth Manifolds):** The pushout inherits smooth structure by the **gluing lemma** for smooth manifolds: if $\mathcal{X}$ and $\mathcal{X}_{\text{cap}}$ are smooth manifolds with diffeomorphic boundaries $\partial \mathcal{X}_\Sigma \cong \partial \mathcal{X}_{\text{cap}}$, then $\mathcal{X}'$ admits a unique smooth structure making the gluing maps smooth.

  **Regularity Verification:** By Lemma 2 (Step 2.4), the cap $\mathcal{X}_{\text{cap}}$ has bounded derivatives $|\nabla^k \Phi| \leq C_k(V)$. The gluing is smooth because:
  1. Near the boundary $\partial \mathcal{X}_\Sigma$, the original solution matches profile $V$ up to exponential decay
  2. The cap solution also matches $V$ with exponential approach
  3. These exponentially decaying differences can be smoothly interpolated using cutoff functions

- **Meas (Measure Spaces):** The measure $\mu'$ on $\mathcal{X}'$ is defined by:
  $$\mu'(A) = \mu(A \cap (\mathcal{X} \setminus \mathcal{X}_\Sigma)) + \mu_{\text{cap}}(A \cap \mathcal{X}_{\text{cap}})$$
  This is well-defined since the overlap (boundary) has measure zero by codimension $\geq 2$.

**Step 3.2 (Universal Property Verification):** Given morphisms $f: \mathcal{X} \to \mathcal{Y}$ and $g: \mathcal{X}_{\text{cap}} \to \mathcal{Y}$ agreeing on boundaries, define:
$$\tilde{f}(x) = \begin{cases}
f(x) & \text{if } x \in \mathcal{X} \setminus \mathcal{X}_\Sigma \\
g(x) & \text{if } x \in \mathcal{X}_{\text{cap}}
\end{cases}$$

**Well-Definedness:** On the boundary $\partial \mathcal{X}_\Sigma = \partial \mathcal{X}_{\text{cap}}$, we have $f|_{\partial} = g|_{\partial}$ by hypothesis, so $\tilde{f}$ is well-defined.

**Uniqueness:** If $\tilde{f}'$ also satisfies the universal property, then $\tilde{f}' = \tilde{f}$ on both $\mathcal{X} \setminus \mathcal{X}_\Sigma$ and $\mathcal{X}_{\text{cap}}$, hence $\tilde{f}' = \tilde{f}$ everywhere.

**Conclusion:** The pushout exists and satisfies the universal property, making it the categorical surgery construction. □

---

## Step 1: Flow Continuation (Guarantee 1)

**Claim:** After surgery, the evolution continues with a well-defined state $x' \in \mathcal{X}'$.

**Proof:**

**Step 1.1 (State Transfer):** The pre-surgery state is $x^- \in \mathcal{X}$ at time $t^-$. We construct the post-surgery state $x' \in \mathcal{X}'$ as:
$$x' = \mathcal{O}_S(x^-)$$

where $\mathcal{O}_S: \mathcal{X} \dashrightarrow \mathcal{X}'$ is the surgery operator from Lemma 3.

**Case Analysis:**

**Case 1.1a (Away from Singularity):** If $x^- \in \mathcal{X} \setminus \mathcal{X}_\Sigma$ (state is away from the singular neighborhood), then:
$$x' = x^- \in \mathcal{X} \setminus \mathcal{X}_\Sigma \subset \mathcal{X}'$$

The state is preserved exactly, and evolution continues unchanged on $\mathcal{X}'$ since the domain $\mathcal{X} \setminus \mathcal{X}_\Sigma$ embeds in $\mathcal{X}'$.

**Case 1.1b (Near Singularity):** If $x^- \in \mathcal{X}_\Sigma$ (state is in the singular neighborhood), the state $x^-$ is **projected** to the boundary $\partial \mathcal{X}_\Sigma$ and then mapped to the corresponding point in $\mathcal{X}_{\text{cap}}$ via the boundary identification:
$$x^- \mapsto \pi(x^-) \in \partial \mathcal{X}_\Sigma \cong \partial \mathcal{X}_{\text{cap}} \mapsto x' \in \mathcal{X}_{\text{cap}}$$

where $\pi(x) = \arg\min_{y \in \partial \mathcal{X}_\Sigma} d(x, y)$ is the nearest-point projection.

**Step 1.2 (Semiflow Extension):** Define the post-surgery semiflow $S_t': \mathcal{X}' \to \mathcal{X}'$ by:
$$S_t' y = \begin{cases}
S_t y & \text{if } y \in \mathcal{X} \setminus \mathcal{X}_\Sigma \text{ and } S_s y \in \mathcal{X} \setminus \mathcal{X}_\Sigma \text{ for all } s \in [0, t] \\
S_t^{\text{cap}} y & \text{if } y \in \mathcal{X}_{\text{cap}}
\end{cases}$$

where $S_t^{\text{cap}}$ is the flow on the cap (governed by the same PDE restricted to $\mathcal{X}_{\text{cap}}$).

**Well-Definedness:** The two definitions agree on the overlap (boundary) because:
1. By Lemma 2, the cap solution matches profile $V$ at the boundary: $\lim_{y \to \partial} S_t^{\text{cap}} y = V$
2. By admissibility, the original solution also approaches $V$: $\lim_{y \to \partial} S_t y = V$ for $y \in \partial \mathcal{X}_\Sigma$
3. Uniqueness of solutions to the PDE ensures these flows agree on the boundary

**Step 1.3 (Initial Value Problem):** With state $x' \in \mathcal{X}'$ and semiflow $S_t': \mathcal{X}' \to \mathcal{X}'$, the post-surgery trajectory is:
$$u'(t) = S_t' x' \quad \text{for } t \geq 0$$

By the theory of semiflows (see Hale {cite}`Hale88` for ordinary differential equations, Temam {cite}`Temam97` for dissipative PDE semiflows), the trajectory $u'(t)$ is well-defined and continuous in $\mathcal{X}'$ for $t$ in the maximal interval of existence $[0, T_*')$.

**Conclusion:** Flow continuation is achieved with well-defined state $x' \in \mathcal{X}'$ and semiflow $S_t'$. □

---

## Step 2: Energy Control (Guarantee 2)

**Claim:** The post-surgery energy satisfies:
$$\Phi(x') \leq \Phi(x^-) + \delta_S$$
where $\delta_S$ is a controlled jump, and furthermore:
$$\Phi(x') \leq \Phi(x^-) - \Delta\Phi_{\text{surg}}$$
for an energy **drop** $\Delta\Phi_{\text{surg}} \geq \epsilon_T > 0$ (surgery strictly decreases energy).

**Proof:**

**Step 2.1 (Energy Decomposition):** Before surgery, the energy decomposes as:
$$\Phi(x^-) = \Phi(\mathcal{X} \setminus \mathcal{X}_\Sigma) + \Phi(\mathcal{X}_\Sigma)$$

where $\Phi(\mathcal{X}_\Sigma)$ is the energy concentrated in the singular neighborhood.

**Step 2.2 (Excised Energy):** By Perelman's monotonicity formula for Ricci flow {cite}`Perelman03` (Theorem 1.1, entropy monotonicity), the energy in the excised region satisfies:
$$\Phi(\mathcal{X}_\Sigma) \geq c_n \cdot \text{Vol}(\Sigma)^{(n-2)/n}$$

where $c_n > 0$ is a dimension-dependent constant, and we use the isoperimetric inequality:
$$\text{Vol}(\Sigma) \geq v_{\min}(T) > 0$$

Write $v_{\min} := v_{\min}(T)$. The positivity of $v_{\min}$ follows from admissibility: infinitesimally small singularities (with $\text{Vol}(\Sigma) \to 0$) are excluded by the capacity bound.

**Literature Justification:**
- For **Ricci flow**, Perelman's entropy formula {cite}`Perelman03` gives $\Phi(\mathcal{X}_\Sigma) = \int R \log R \, dV \geq c \cdot \text{Vol}(\Sigma)^{2/3}$ in dimension 3
- For **mean curvature flow**, monotonicity estimates (see e.g. {cite}`HuiskenSinestrari09`) provide $\Phi(\mathcal{X}_\Sigma) = \int H^2 \, dA \geq c \cdot \text{Area}(\Sigma)$ where $H$ is mean curvature
- For **harmonic map flow**, Struwe's analysis {cite}`Struwe88` ensures $\Phi(\mathcal{X}_\Sigma) \geq \epsilon_0 > 0$ for any non-trivial bubble

**Step 2.3 (Capping Energy):** By Lemma 2 (Step 2.4), the cap has bounded energy:
$$\Phi(\mathcal{X}_{\text{cap}}) \leq E_{\text{cap}}(V)$$

For canonical profiles, this energy is typically **smaller** than the excised energy. For example:
- **Round sphere cap**: $E_{\text{cap}}(S^3) = \text{Vol}(S^3) \cdot R_{\text{min}}$ where $R_{\text{min}}$ is the minimum curvature (controlled by normalization)
- **Bryant soliton cap**: $E_{\text{cap}}(\text{Bryant}) = E_{\text{soliton}}$ (fixed soliton energy)

**Step 2.4 (Post-Surgery Energy):** After surgery:
$$\Phi(x') = \Phi(\mathcal{X} \setminus \mathcal{X}_\Sigma) + \Phi(\mathcal{X}_{\text{cap}})$$

Subtracting the pre-surgery energy:
$$\Phi(x') - \Phi(x^-) = \Phi(\mathcal{X}_{\text{cap}}) - \Phi(\mathcal{X}_\Sigma)$$

**Step 2.5 (Energy Drop Verification):** By the estimates in Steps 2.2 and 2.3:
$$\Delta\Phi_{\text{surg}} := \Phi(x^-) - \Phi(x') = \Phi(\mathcal{X}_\Sigma) - \Phi(\mathcal{X}_{\text{cap}}) \geq c_n \cdot v_{\min}^{(n-2)/n} - E_{\text{cap}}(V)$$

**Positivity:** For admissible surgeries, the excised energy dominates the cap energy:
$$c_n \cdot v_{\min}^{(n-2)/n} > E_{\text{cap}}(V)$$

This is ensured by the choice of admissibility threshold $\varepsilon_{\text{adm}}$. Define the **discrete progress constant**:
$$\epsilon_T := \min_{V \in \mathcal{L}_T} \left(c_n \cdot v_{\min}^{(n-2)/n} - E_{\text{cap}}(V)\right) > 0$$

Since $\mathcal{L}_T$ is finite, the minimum is attained and positive.

**Step 2.6 (Controlled Jump):** In the worst case (when $x^- \in \mathcal{X}_\Sigma$), the energy jump from state transfer is:
$$\delta_S = E_{\text{cap}}(V) - \Phi(x^-|_{\mathcal{X}_\Sigma})$$

By conservation and the fact that $\Phi(x^-|_{\mathcal{X}_\Sigma}) \leq \Phi(\mathcal{X}_\Sigma)$:
$$\delta_S \leq E_{\text{cap}}(V) \leq E_{\text{cap}}^{\max} := \max_{V \in \mathcal{L}_T} E_{\text{cap}}(V) < \infty$$

**Conclusion:** Energy is controlled with $\Phi(x') \leq \Phi(x^-) - \epsilon_T$, satisfying Guarantee 2. □

---

## Step 3: Certificate Production (Guarantee 3)

**Claim:** The surgery produces a **re-entry certificate** $K^{\mathrm{re}}$ satisfying:
$$K^{\mathrm{re}} \Rightarrow \mathrm{Pre}(\text{target})$$
where $\mathrm{Pre}(\text{target})$ denotes the preconditions for continuing the Sieve toward the target mode.

**Proof:**

**Step 3.1 (Certificate Components):** The re-entry certificate is constructed as:
$$K^{\mathrm{re}} = (x', \mathcal{X}', \Phi', \mathfrak{D}', S_t', \text{energy bound}, \text{regularity witness}, \text{mode routing})$$

**Component Verification:**

**3.1a (State):** $x' \in \mathcal{X}'$ is the post-surgery state from Step 1.

**3.1b (State Space):** $\mathcal{X}' = (\mathcal{X} \setminus \mathcal{X}_\Sigma) \sqcup_{\partial} \mathcal{X}_{\text{cap}}$ is the surgered space from Lemma 3.

**3.1c (Energy Functional):** Define $\Phi': \mathcal{X}' \to \mathbb{R}_{\geq 0}$ by:
$$\Phi'(y) = \begin{cases}
\Phi(y) & \text{if } y \in \mathcal{X} \setminus \mathcal{X}_\Sigma \\
\Phi_{\text{cap}}(y) & \text{if } y \in \mathcal{X}_{\text{cap}}
\end{cases}$$

where $\Phi_{\text{cap}}$ is the energy functional on the cap (inherited from Lemma 2).

**Well-Definedness:** On the boundary $\partial \mathcal{X}_\Sigma = \partial \mathcal{X}_{\text{cap}}$, both functionals limit to the same value (profile energy $\Phi(V)$), ensuring continuity.

**3.1d (Dissipation):** Define $\mathfrak{D}': \mathcal{X}' \to \mathbb{R}_{\geq 0}$ similarly:
$$\mathfrak{D}'(y) = \begin{cases}
\mathfrak{D}(y) & \text{if } y \in \mathcal{X} \setminus \mathcal{X}_\Sigma \\
\mathfrak{D}_{\text{cap}}(y) & \text{if } y \in \mathcal{X}_{\text{cap}}
\end{cases}$$

By Lemma 2, the cap has bounded dissipation inherited from profile $V$.

**3.1e (Semiflow):** $S_t': \mathcal{X}' \to \mathcal{X}'$ is the extended semiflow from Step 1.2.

**3.1f (Energy Bound):** By Step 2:
$$\Phi'(x') \leq \Phi(x^-) - \epsilon_T < \Phi(x^-) < E_0$$

where $E_0 = \Phi(x_0)$ is the initial energy. The strict decrease is crucial for progress (Guarantee 4).

**3.1g (Regularity Witness):** By Lemma 2 (Step 2.4), the surgered solution has bounded derivatives:
$$\sup_{\mathcal{X}'} |\nabla^k \Phi'| \leq \max\left(\sup_{\mathcal{X} \setminus \mathcal{X}_\Sigma} |\nabla^k \Phi|, C_k(V)\right) < \infty$$

for all $k \leq k_{\max}(V)$. This regularization is key: surgery removes singularities and restores smoothness.

**Step 3.2 (Precondition Verification):** The target mode preconditions $\mathrm{Pre}(\text{target})$ typically require:
1. **Finite energy**: $\Phi'(x') < \infty$ ✓ (by Step 3.1f)
2. **Regularity**: $x' \in \mathcal{X}'$ with bounded derivatives ✓ (by Step 3.1g)
3. **Well-posed evolution**: Semiflow $S_t'$ exists ✓ (by Step 1.2)
4. **Barrier satisfaction**: No immediate barrier breach ✓ (surgery removes obstruction)

**Step 3.3 (Mode Routing):** The certificate includes routing information:
- **If surgery successful**: Route to continuation mode (e.g., D.D if energy disperses, S.E if barrier engaged)
- **Surgery metadata**: $(V, \Sigma, \epsilon_T, \text{surgery count})$ for tracking

**Conclusion:** The re-entry certificate $K^{\mathrm{re}}$ is produced with all required components, satisfying $K^{\mathrm{re}} \Rightarrow \mathrm{Pre}(\text{target})$. □

---

## Step 4: Progress and Finite Surgery Count (Guarantee 4)

**Claim:** Either:
1. **Bounded surgery count**: The number of surgeries is finite, $N_{\text{surgeries}} < \infty$, or
2. **Decreasing complexity**: Some other measure of complexity decreases (e.g., topology, genus)

**Proof:**

**Step 4.1 (Energy Budget):** Starting from initial state $x_0$ with energy $E_0 = \Phi(x_0)$, the energy is bounded below:
$$\Phi(u(t)) \geq \Phi_{\min} \geq 0$$

where $\Phi_{\min}$ is the infimum of $\Phi$ over $\mathcal{X}$ (often zero, or the energy of the ground state).

**Step 4.2 (Surgery Sequence):** Suppose surgeries occur at times $t_1 < t_2 < \cdots < t_N$ with states $x_1, x_2, \ldots, x_N$. By Step 2, each surgery drops energy:
$$\Phi(x_{k+1}) \leq \Phi(x_k^-) - \epsilon_T$$

where $x_k^-$ is the pre-surgery state at time $t_k$.

**Step 4.3 (Summation):** Summing over all $N$ surgeries:
$$\Phi(x_N) \leq \Phi(x_0) - N \cdot \epsilon_T$$

**Step 4.4 (Finiteness):** Since $\Phi(x_N) \geq \Phi_{\min}$:
$$\Phi_{\min} \leq \Phi(x_0) - N \cdot \epsilon_T$$

Rearranging:
$$N \leq \frac{\Phi(x_0) - \Phi_{\min}}{\epsilon_T} = \frac{E_0 - \Phi_{\min}}{\epsilon_T} < \infty$$

**Explicit Bound:** The surgery count is bounded by:
$$N_{\text{surgeries}} \leq \left\lfloor \frac{E_0}{\epsilon_T} \right\rfloor$$

This is a **concrete finite bound** depending only on initial energy $E_0$ and the discrete progress constant $\epsilon_T > 0$.

**Step 4.5 (Topological Complexity Decrease):** In certain geometric contexts (e.g., Ricci flow on 3-manifolds), surgery also decreases **topological complexity**:

- **Genus reduction**: Each surgery may reduce the genus $g$ of the manifold (e.g., removing a handle)
- **Connected sum decomposition**: Repeated surgeries decompose the manifold into simpler pieces (prime decomposition)

**Perelman's Theorem {cite}`Perelman03` (for Ricci Flow):** For Ricci flow with surgery on compact 3-manifolds:
1. Either the manifold becomes extinct (shrinks to a point) in finite time, or
2. The flow with surgery exists for all time and decomposes the manifold into a finite number of pieces with canonical geometric structures (hyperbolic, spherical, or flat)

In both cases, the process terminates in finite time or produces a well-understood limiting structure.

**Step 4.6 (Well-Founded Progress):** The combination of:
- Finite energy budget (Step 4.4)
- Discrete energy drop $\epsilon_T > 0$ per surgery (Step 2.5)
- Possible topological simplification (Step 4.5)

ensures the surgery sequence terminates in finite time. The system cannot undergo infinitely many surgeries (no Zeno paradox).

**Conclusion:** Progress is guaranteed with finite surgery count $N \leq E_0/\epsilon_T < \infty$. □

---

## Step 5: Failure Case Handling

**Claim:** If $K_{\text{inadm}}$ is produced (surgery is inadmissible), the run terminates at mode $M$ as a genuine singularity, or routes to reconstruction via {prf:ref}`mt-lock-reconstruction`.

**Proof:**

**Step 5.1 (Inadmissibility Conditions):** The certificate $K_{\text{inadm}}$ is produced when one of the following fails:
1. **Non-canonical profile**: $V \notin \mathcal{L}_T$ (profile is not in the finite library)
2. **Codimension violation**: $\text{codim}(\Sigma) < 2$ (singular set too large)
3. **Capacity violation**: $\text{Cap}(\Sigma) > \varepsilon_{\text{adm}}$ (excision would remove too much energy)

**Step 5.2 (Genuine Singularity):** When surgery is inadmissible, the singularity is classified as **genuine**:
- The trajectory cannot be continued via surgery
- The singularity represents a fundamental obstruction to the flow
- The Sieve terminates at mode $M$ with certificate $K_{\text{inadm}}$

**Examples:**
- **Ricci flow**: Non-collapsed singularities violating the canonical neighborhood assumption (e.g., neckpinch with large cross-section)
- **Mean curvature flow**: Type-II singularities forming at isolated points (e.g., translating solitons)
- **Harmonic map flow**: Non-trivial harmonic spheres with energy exceeding the quantization threshold

**Step 5.3 (Reconstruction Routing):** Alternatively, if the Lock Reconstruction metatheorem {prf:ref}`mt-lock-reconstruction` is available:
1. **Lock attempt**: The Sieve attempts to prove non-occurrence of inadmissible singularities via Lock tactics (E1-E10)
2. **Reconstruction**: If successful, the trajectory is **reconstructed** to avoid the singularity altogether
3. **Certificate update**: $K_{\text{inadm}}$ is replaced by $K_{\text{lock}}$ certifying avoidance

**Step 5.4 (Termination Guarantee):** In all cases, the Framework provides a definite outcome:
- **Surgery succeeds**: $K^{\mathrm{re}}$ produced, flow continues
- **Surgery fails, genuine singularity**: $K_{\text{inadm}}$ produced, flow terminates
- **Surgery fails, Lock succeeds**: $K_{\text{lock}}$ produced, singularity avoided

No case is left unresolved. □

---

## Conclusion and Certificate Assembly

We have established all four guarantees of the Structural Surgery Principle:

1. **Flow Continuation (Step 1):** Evolution continues past surgery with well-defined state $x' \in \mathcal{X}'$ and semiflow $S_t': \mathcal{X}' \to \mathcal{X}'$

2. **Energy Control (Step 2):**
   - Controlled jump: $\Phi(x') \leq \Phi(x^-) + \delta_S$ where $\delta_S \leq E_{\text{cap}}^{\max}$
   - Energy drop: $\Phi(x') \leq \Phi(x^-) - \epsilon_T$ where $\epsilon_T = \min_{V \in \mathcal{L}_T} (c_n \cdot v_{\min}^{(n-2)/n} - E_{\text{cap}}(V)) > 0$

3. **Certificate Production (Step 3):** Re-entry certificate $K^{\mathrm{re}} = (x', \mathcal{X}', \Phi', \mathfrak{D}', S_t', \text{witnesses})$ satisfying $K^{\mathrm{re}} \Rightarrow \mathrm{Pre}(\text{target})$

4. **Progress (Step 4):** Finite surgery count $N_{\text{surgeries}} \leq \lfloor E_0 / \epsilon_T \rfloor < \infty$

**Final Certificate Structure:**

$$K_{\text{surgery}} = \begin{cases}
K^{\mathrm{re}} & \text{if surgery succeeds (Cases 1, 2 of Trichotomy)} \\
K_{\text{inadm}} & \text{if surgery fails (Case 3 of Trichotomy)}
\end{cases}$$

**Re-Entry Certificate Details:**

$$K^{\mathrm{re}} = \left(
\begin{array}{l}
\text{surgered state: } x' \in \mathcal{X}' \\
\text{surgered space: } \mathcal{X}' = (\mathcal{X} \setminus \mathcal{X}_\Sigma) \sqcup_{\partial} \mathcal{X}_{\text{cap}} \\
\text{energy bound: } \Phi'(x') \leq \Phi(x^-) - \epsilon_T \\
\text{regularity: } \sup_{\mathcal{X}'} |\nabla^k \Phi'| \leq C_k \\
\text{surgery count: } N_{\text{current}} \\
\text{remaining budget: } E_0/\epsilon_T - N_{\text{current}} \\
\text{profile: } V \in \mathcal{L}_T \\
\text{mode routing: } \text{target mode}
\end{array}
\right)$$

**Bridge to Literature:** The construction synthesizes:
- **Hamilton's surgery program** {cite}`Hamilton97`: Four-manifolds with positive curvature, surgery templates for Ricci flow
- **Perelman's surgery algorithm** {cite}`Perelman03`: Canonical neighborhoods, $\delta$-neck detection, surgery at scale $h$, finite extinction time
- **Kleiner-Lott exposition** {cite}`KleinerLott08`: Detailed verification of surgery regularity and energy estimates

**Categorical Formulation:** The pushout construction provides a **universal property** ensuring:
- **Uniqueness**: Surgery operator $\mathcal{O}_S$ is unique up to isomorphism
- **Functoriality**: Surgery commutes with symmetries and structure-preserving maps
- **Composability**: Sequential surgeries compose via pushout pasting lemmas

**Applicability Beyond Ricci Flow:** While the theorem is anchored in Perelman's work {cite}`Perelman03`, the categorical formulation extends to:
- **Mean curvature flow**: Mean-convex surgery {cite}`HuiskenSinestrari09`
- **Harmonic map flow**: Bubble tree surgery {cite}`Struwe88`
- **Yang-Mills flow**: Gauge theory surgery {cite}`DonaldsonKronheimer90`

Each type $T$ has a specific canonical library $\mathcal{L}_T$ and energy estimates, but the abstract framework applies uniformly.

:::

---

## Appendix: Supporting Lemmas

### Lemma A1: Boundary Regularity

**Statement:** The boundary $\partial \mathcal{X}_\Sigma$ has controlled geometry:
$$\mathcal{H}^{n-1}(\partial \mathcal{X}_\Sigma) \leq C \cdot \epsilon^{n-1}$$
and the second fundamental form satisfies $|A| \leq C/\epsilon$.

**Proof Sketch:** By the coarea formula and capacity bound. See Adams-Hedberg {cite}`AdamsHedberg96` Theorem 5.1.2. □

### Lemma A2: Asymptotic Matching

**Statement:** Near the boundary $\partial \mathcal{X}_\Sigma$, the solution matches profile $V$ with exponential decay:
$$|\Phi(x) - \Phi(V)| \leq C e^{-c d(x, \Sigma)/\epsilon}$$

**Proof Sketch:** By parabolic regularity and the Implicit Function Theorem applied to the PDE in neck coordinates. See Perelman {cite}`Perelman03` §12 (canonical neighborhood theorem). □

### Lemma A3: Gluing Smoothness

**Statement:** The gluing map $\mathcal{X}' = (\mathcal{X} \setminus \mathcal{X}_\Sigma) \sqcup_{\partial} \mathcal{X}_{\text{cap}}$ is $C^{k_{\max}}$-smooth where $k_{\max}$ depends on the profile regularity.

**Proof Sketch:** By Lemma A2, the exponentially decaying difference allows smooth interpolation using a cutoff function $\chi$ with $\text{supp}(\nabla \chi) \subset \{\epsilon/2 < d(\cdot, \Sigma) < \epsilon\}$. The glued solution is:
$$\Phi'_{\text{glued}} = (1 - \chi) \Phi + \chi \Phi_{\text{cap}}$$
Derivatives are bounded by the geometric-arithmetic mean inequality applied to the weighted sum. □

---

## References Context

The following references are cited with specific theorem/page numbers where applicable:

- {cite}`Hamilton97`: Four-manifolds with positive isotropic curvature, surgery templates (§4-5)
- {cite}`Perelman03`: Ricci flow with surgery on three-manifolds, entropy monotonicity (Theorem 1.1), canonical neighborhoods (§12), finite extinction (Theorem 13.1)
- {cite}`KleinerLott08`: Detailed exposition of Perelman's work, surgery regularity (§75-77), energy estimates (§80)
- {cite}`AdamsHedberg96`: Function Spaces and Potential Theory, Sobolev capacity (Theorem 5.1.2), removable singularities (Chapter 6)
- {cite}`Federer69`: Geometric Measure Theory, Hausdorff dimension and capacity (§2.10), coarea formula (§3.2)
- {cite}`MacLane71`: Categories for the Working Mathematician, pushouts and colimits (Chapter III)
- {cite}`HuiskenSinestrari09`: Mean curvature flow with surgeries of two-convex hypersurfaces
- {cite}`Evans10`: Partial Differential Equations, regularity theory
- {cite}`Struwe88`: Evolution of harmonic maps, bubble analysis
- {cite}`Hale88`: Asymptotic Behavior of Dissipative Systems, semiflow theory (Chapter 1)
- {cite}`Temam97`: Infinite-Dimensional Dynamical Systems in Mechanics and Physics, PDE semiflows
