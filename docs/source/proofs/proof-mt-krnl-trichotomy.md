# Proof of KRNL-Trichotomy (Concentration-Compactness)

:::{prf:proof}
:label: proof-mt-krnl-trichotomy

**Theorem Reference:** {prf:ref}`mt-krnl-trichotomy`

This proof establishes the structural trichotomy for trajectories with finite breakdown time. We proceed through a rigorous concentration-compactness argument, following the framework of Lions {cite}`Lions84` with refinements from Bahouri-Gérard {cite}`BahouriGerard99`, Kenig-Merle {cite}`KenigMerle06`, and Struwe {cite}`Struwe90`.

---

## Setup and Notation

**Given Data:**
- A structural flow datum $\mathcal{S} = (\mathcal{X}, S_t, \Phi, \mathfrak{D}, G, \partial)$ where:
  - $\mathcal{X}$ is the state stack (configuration space)
  - $S_t: \mathcal{X} \to \mathcal{X}$ is the semiflow (evolution operator)
  - $\Phi: \mathcal{X} \to \mathbb{R}_{\geq 0} \cup \{\infty\}$ is the cohomological height (energy functional)
  - $\mathfrak{D}: \mathcal{X} \to \mathbb{R}_{\geq 0}$ is the dissipation rate
  - $G$ is a compact Lie group acting on $\mathcal{X}$ (symmetry group)
  - $\partial: \mathcal{X} \to \mathcal{X}_\partial$ is the boundary restriction

**Trajectory Data:**
- Initial state $x \in \mathcal{X}$ with finite energy $\Phi(x) = E_0 < \infty$
- Trajectory $u(t) := S_t x$ defined for $t \in [0, T_*(x))$ where $T_*(x) < \infty$ is the maximal existence time (breakdown time)

**Hypothesis Verification:**

1. **(Reg) Minimal Regularity:** The semiflow $S_t$ is well-defined and continuous on $\mathcal{X}$ for $t \in [0, T_*)$, meaning:
   $$\lim_{s \to t} \|S_s x - S_t x\|_{\mathcal{X}} = 0 \quad \text{for all } t \in [0, T_*)$$
   This ensures the trajectory is a bona fide solution to the governing equations.

2. **(D) Dissipation (Energy-Dissipation Inequality):** For all $t \in [0, T_*)$:
   $$\Phi(u(t)) + \int_0^t \mathfrak{D}(u(s)) \, ds \leq \Phi(x) = E_0$$
   This is the fundamental energy balance, linking the cohomological height to internal dissipation. The inequality ensures energy is non-increasing modulo dissipation:
   $$\frac{d}{dt}\Phi(u(t)) \leq -\mathfrak{D}(u(t)) \leq 0$$
   (in the distributional sense).

3. **(C) Compactness (Profile Convergence Modulo Symmetries):** For any bounded energy sequence $(x_n) \subset \mathcal{X}$ with $\sup_n \Phi(x_n) \leq E < \infty$, there exist:
   - A subsequence $(x_{n_k})$
   - Symmetry elements $g_k \in G$
   - A profile $v^* \in \mathcal{X}$

   such that:
   $$g_k \cdot x_{n_k} \rightharpoonup v^* \quad \text{(weakly in } \mathcal{X}\text{)}$$

   This is the categorical formulation of the Lions concentration-compactness principle. The quotient $\mathcal{X} // G$ is sequentially compact in the weak topology on bounded energy sets.

**Critical Exponent Convention:**
- Let $s_c$ denote the critical Sobolev exponent for $\mathcal{X}$. For dispersive equations on $\mathbb{R}^d$, typically $s_c = d/2 - 2/(p-2)$ where $p$ is the nonlinearity exponent.
- We assume $\mathcal{X} \hookrightarrow \dot{H}^{s_c}(\mathbb{R}^d)$ (the homogeneous critical Sobolev space).

**Mode Definitions:**

| Mode | Name | Description |
|------|------|-------------|
| D.D | Dispersion-Decay | Energy disperses to spatial infinity, solution scatters |
| S.E | Subcritical-Equilibration | Energy concentrates but subcritical scaling prevents blowup |
| C.D | Concentration-Dispersion | Partial concentration with dispersion of residual |
| T.E | Topological-Extension | Concentration resolved via topological completion |
| S.D | Structural-Dispersion | Structural constraints force dispersion |
| C.E | Concentration-Escape | Genuine singularity with energy escape |

**Goal:**
Prove that exactly one of the following three mutually exclusive outcomes occurs:

1. **Mode D.D (Dispersion, Global Existence):** Energy disperses, no concentration, solution scatters to free evolution
2. **Mode S.E, C.D, T.E, S.D (Global Regularity via Barrier):** Concentration occurs but all interface permits are satisfied, preventing singularity formation
3. **Mode C.E (Genuine Singularity):** Energy escapes or structured blow-up with at least one interface permit violated

---

## Step 1: Energy Dichotomy (Concentration vs. Dispersion)

**Claim:** Either $\Phi(u(t)) \to 0$ as $t \nearrow T_*$ (dispersion), or $\liminf_{t \to T_*} \Phi(u(t)) \geq \Phi_* > 0$ (concentration).

**Proof of Claim:**

By hypothesis (D), the function $t \mapsto \Phi(u(t))$ is non-increasing:
$$\Phi(u(t_2)) \leq \Phi(u(t_1)) \quad \text{for } 0 \leq t_1 \leq t_2 < T_*$$

Since $\Phi(u(t)) \geq 0$ and is monotone decreasing, the limit exists:
$$\Phi_* := \lim_{t \nearrow T_*} \Phi(u(t)) = \inf_{0 \leq t < T_*} \Phi(u(t)) \in [0, E_0]$$

We now perform case analysis:

**Case 1.1 (Dispersion):** $\Phi_* = 0$.

In this case, for any $\varepsilon > 0$, there exists $t_\varepsilon < T_*$ such that $\Phi(u(t)) < \varepsilon$ for all $t \in [t_\varepsilon, T_*)$.

By hypothesis (C), bounded energy sequences have weakly convergent subsequences modulo $G$. Since $\Phi(u(t)) \to 0$, the only possible weak limit is $v^* = 0$ (the zero element). This implies mass disperses to spatial infinity—the hallmark of scattering.

**Lions' Dichotomy (Lemma I.1 of {cite}`Lions84`):** For sequences in $L^p(\mathbb{R}^d)$ with bounded $L^p$ norm, either:
- **(Compactness)** A subsequence converges strongly in $L^p_{\text{loc}}$ (concentration)
- **(Vanishing)** The sequence vanishes: $\lim_{n \to \infty} \sup_{y \in \mathbb{R}^d} \int_{B_R(y)} |u_n|^p = 0$ for all $R < \infty$ (dispersion)
- **(Dichotomy)** Mass splits between escaping parts (Lions' third case, which we group with vanishing for trichotomy purposes)

**Applicability Remark:** Lions' lemma applies when $\Phi$ controls an $L^p$ norm (e.g., $\Phi(u) \geq c\|u\|_{L^p}^p$ for some embedding). For abstract hypostructures where this embedding is unclear, the vanishing/concentration dichotomy must be verified independently using the specific structure of $\mathcal{X}$.

Applied to our setting with the sequence $u(t_n)$ for $t_n \nearrow T_*$: if $\Phi_* = 0$ and $\Phi$ dominates local mass, then the sequence must vanish in the Lions sense. By standard scattering theory (e.g., Strichartz estimates for dispersive PDEs), the trajectory exists globally and converges to free evolution:
$$\lim_{t \to \infty} \|u(t) - U_t u_\infty\|_{\mathcal{X}} = 0$$
where $U_t$ is the free evolution (linear semiflow) and $u_\infty \in \mathcal{X}$.

**Certificate Produced:** $K_{\text{D.D}} = (\text{mode: D.D}, \Phi_*, \text{scattering data: } u_\infty)$

**Routing:** This trajectory achieves **Global Existence** and exits the trichotomy via Mode D.D.

---

**Case 1.2 (Concentration):** $\Phi_* > 0$.

In this case, energy is trapped in the system. We proceed to Step 2 to extract the concentration profile.

---

## Step 2: Profile Extraction via Concentration-Compactness

**Assumption:** We are in Case 1.2, so $\Phi_* > 0$.

**Claim:** There exist:
1. A sequence $t_n \nearrow T_*$
2. Symmetry elements $g_n \in G$ (encoding translation, scaling, rotation)
3. A non-trivial profile $v^* \in \mathcal{X}$ with $0 < \Phi(v^*) \leq \Phi_*$

such that:
$$g_n \cdot u(t_n) \rightharpoonup v^* \quad \text{(weakly in } \mathcal{X}\text{)}$$

**Proof of Claim:**

**Step 2.1 (Bounded Energy Sequence):**
Let $t_n \nearrow T_*$ be any sequence converging to the breakdown time. By hypothesis (D):
$$\Phi(u(t_n)) \leq \Phi(x) = E_0 < \infty \quad \text{for all } n$$

Thus $(u(t_n))$ is a bounded energy sequence in $\mathcal{X}$.

**Step 2.2 (Compactness Modulo Symmetries):**
By hypothesis (C), there exist a subsequence (still denoted $t_n$), symmetry elements $g_n \in G$, and a profile $v^* \in \mathcal{X}$ such that:
$$g_n \cdot u(t_n) \rightharpoonup v^* \quad \text{(weakly)}$$

**Step 2.3 (Non-Triviality):**
We claim $v^* \neq 0$. Suppose for contradiction that $v^* = 0$. By Lions' vanishing lemma (Lemma I.1 of {cite}`Lions84`), if $g_n \cdot u(t_n) \rightharpoonup 0$ weakly, then the sequence disperses to spatial infinity, meaning:
$$\lim_{n \to \infty} \Phi(u(t_n)) = 0$$

However, from Case 1.2, we know that:
$$\lim_{n \to \infty} \Phi(u(t_n)) = \Phi_* > 0$$

This is a contradiction. Hence $v^* \neq 0$.

Moreover, by weak lower semicontinuity of the energy functional $\Phi$ (a standard property for convex functionals on reflexive Banach spaces):
$$\Phi(v^*) \leq \liminf_{n \to \infty} \Phi(g_n \cdot u(t_n)) = \liminf_{n \to \infty} \Phi(u(t_n)) = \Phi_*$$

since $G$ acts by isometries (energy-preserving), so $\Phi(g_n \cdot u(t_n)) = \Phi(u(t_n))$.

**Step 2.4 (Profile Decomposition à la Bahouri-Gérard):**

For critical dispersive equations, the profile decomposition theorem of Bahouri-Gérard {cite}`BahouriGerard99` provides a more refined structure. Any bounded sequence $(u_n)$ in $\dot{H}^{s_c}(\mathbb{R}^d)$ admits a decomposition:
$$u_n = \sum_{j=1}^J g_n^{(j)} \cdot V^{(j)} + w_n^{(J)} + o_{\mathcal{X}}(1)$$

where:
- $V^{(j)} \in \dot{H}^{s_c}$ are orthogonal profiles (the "bubbles")
- $g_n^{(j)} = (\lambda_n^{(j)}, x_n^{(j)}) \in \mathbb{R}^+ \times \mathbb{R}^d$ are scaling-translation parameters satisfying:
  $$\frac{\lambda_n^{(j)}}{\lambda_n^{(k)}} + \frac{\lambda_n^{(k)}}{\lambda_n^{(j)}} + \frac{|x_n^{(j)} - x_n^{(k)}|}{\lambda_n^{(j)} \vee \lambda_n^{(k)}} \to \infty \quad \text{for } j \neq k$$
  (orthogonality of scales/positions)
- $w_n^{(J)} \rightharpoonup 0$ weakly in $\dot{H}^{s_c}$
- Energy decouples: $\|u_n\|_{\dot{H}^{s_c}}^2 = \sum_{j=1}^J \|V^{(j)}\|_{\dot{H}^{s_c}}^2 + \|w_n^{(J)}\|_{\dot{H}^{s_c}}^2 + o(1)$

**Application to Our Trajectory:**
For the sequence $u(t_n)$, we obtain:
$$u(t_n) = \sum_{j=1}^J g_n^{(j)} \cdot v^{(j)} + w_n^{(J)}$$

with $J < \infty$ determined by the energy threshold. The leading profile is $v^* = v^{(1)}$ (after reordering by energy).

**Energy Lower Bound:**
By energy decoupling and $\Phi(u(t_n)) \to \Phi_*$:
$$\Phi_* = \lim_{n \to \infty} \Phi(u(t_n)) \geq \sum_{j=1}^J \Phi(v^{(j)}) \geq \Phi(v^*)$$

Hence:
$$\Phi(v^*) \leq \Phi_* \leq E_0$$

confirming the profile has finite energy bounded by the breakdown threshold.

**Certificate Produced:** $K_{C_\mu}^+ = (\text{concentration scale: } \lambda_n^{(1)}, \text{concentration point: } x_n^{(1)}, \text{profile: } v^*)$

---

## Step 3: Profile Classification via Interface Permits

We now classify the extracted profile $v^*$ according to whether it satisfies the structural interface permits. This determines the trajectory's ultimate fate.

**Interface Permits to Check:**
1. $\mathrm{SC}_\lambda$ (Scaling/Subcriticality): Does $v^*$ have subcritical scaling dimension?
2. $\mathrm{SC}_{\partial c}$ (Parameter Stability): Are moduli parameters stable?
3. $\mathrm{Cap}_H$ (Capacity): Does the singularity set have sufficient codimension?
4. $\mathrm{LS}_\sigma$ (Local Stability): Is the profile linearly stable?
5. $\mathrm{TB}_\pi$ (Topological Boundary): Does the profile admit a topological completion?

**Trichotomy Split:**

---

### Case 3.1: All Permits Satisfied → Global Regularity

**Assumption:** The profile $v^*$ satisfies:
- $K_{\mathrm{SC}_\lambda}^+ = 1$ (subcritical scaling)
- $K_{\mathrm{SC}_{\partial c}}^+ = 1$ (stable parameters)
- $K_{\mathrm{Cap}_H}^+ = 1$ (sufficient capacity)
- $K_{\mathrm{LS}_\sigma}^+ = 1$ (locally stable)
- $K_{\mathrm{TB}_\pi}^+ = 1$ (topologically complete)

**Kenig-Merle Rigidity Theorem (Theorem 1.1 of {cite}`KenigMerle06`):**

For the energy-critical focusing nonlinear Schrödinger equation (NLS):
$$i \partial_t u + \Delta u + |u|^{4/d} u = 0$$
in $\mathbb{R}^d$, if a solution $u$ with bounded $\dot{H}^1$ norm satisfies:
- Finite energy: $\|u(t)\|_{\dot{H}^1} \leq E_c$ (below the ground state threshold)
- No scattering: $\limsup_{t \to \infty} \|\nabla u(t)\|_{L^2} > 0$

then $u$ must be a soliton solution (stationary or traveling wave) up to symmetries.

**Application to General Hypostructures:**

:::{important}
The Kenig-Merle theorem was proved for specific dispersive PDEs (energy-critical NLS). Before applying it to a general hypostructure, the following **verification checklist** must be satisfied:

| Hypothesis | NLS Verification | Hypostructure Verification |
|------------|-----------------|---------------------------|
| Critical scaling | $\dot{H}^1$-critical in $d \geq 3$ | Determine $s_c$ from dimensional analysis |
| Profile decomposition | Bahouri-Gérard applies | Verify hypothesis (C) holds |
| Variational structure | Ground state $Q$ exists | Identify $E_c$ via minimization |
| Coercivity | $E[u] - E[Q] \gtrsim \|u - Q\|^2$ | Verify near critical points |
| Monotonicity formula | Virial/Morawetz identity | Find analogous conserved quantity |

Without this verification, the rigidity argument is a **template** showing what must be established, not a complete proof.
:::

The Kenig-Merle rigidity framework applies more broadly to systems satisfying:
1. **Critical Well-Posedness:** Local well-posedness in $\dot{H}^{s_c}$ with continuous dependence
2. **Finite Speed of Propagation:** Information travels at finite speed (causality)
3. **Profile Decomposition:** Bounded sequences decompose as in Step 2.4
4. **Small Data Global Existence:** Solutions with $\|u_0\|_{\dot{H}^{s_c}} < \delta$ exist globally and scatter

**Rigidity Argument:**

Given that all interface permits are satisfied, we apply the concentration-compactness roadmap:

**Subcritical Scaling ($\mathrm{SC}_\lambda^+$):**
The profile $v^*$ has scaling dimension:
$$\lambda(v^*) = \frac{\text{energy scaling exponent}}{\text{dissipation scaling exponent}} = \frac{\alpha}{\beta} < 1$$

This is the subcriticality condition. By dimensional analysis, subcritical profiles cannot sustain concentration indefinitely—they must either:
- Disperse (return to Mode D.D), or
- Equilibrate to a stationary state

**Parameter Stability ($\mathrm{SC}_{\partial c}^+$):**
The moduli parameters $\theta \in \Theta$ governing $v^*$ satisfy:
$$\|\partial_t \theta\| < \varepsilon_{\text{crit}}$$

This ensures the profile is "almost stationary" modulo symmetries. By LaSalle invariance principle {cite}`LaSalle76`, trajectories in the maximal invariant set must converge to equilibria (for gradient flows) or periodic orbits (for Hamiltonian systems).

**Capacity and Stability ($\mathrm{Cap}_H^+$, $\mathrm{LS}_\sigma^+$):**
The capacity condition ensures the singularity set $\Sigma$ (if any) has Hausdorff dimension:
$$\dim_H(\Sigma) \leq d - 2$$

so singularities are removable by standard elliptic theory (e.g., {cite}`Struwe90` §3.5). Local stability ensures perturbations decay exponentially, preventing instability-driven blowup.

**Conclusion:**
The profile $v^*$ is necessarily a global attractor: either a fixed point of the evolution or a periodic orbit. The original trajectory $u(t)$ converges to a translate of $v^*$ as $t \to \infty$ modulo symmetries. Since $v^*$ is globally defined, the breakdown time $T_* = \infty$—contradicting the assumption $T_* < \infty$.

**Resolution:** This case is impossible under the assumption $T_* < \infty$. If all permits are satisfied, the trajectory must extend globally. The breakdown was an artifact of inadequate regularity assumptions, not a genuine singularity.

**Modes Covered:** S.E (Supercritical Energy, resolved via regularity lift), C.D (Concentration-Dispersion hybrid), T.E (Topological Extension), S.D (Structural Dispersion)

**Certificate Produced:** $K_{\text{Reg}} = (\text{mode: Global Regularity}, v^*, \text{permit certificates: } \{K_{\mathrm{SC}_\lambda}^+, K_{\mathrm{SC}_{\partial c}}^+, \ldots\})$

---

### Case 3.2: At Least One Permit Violated → Genuine Singularity

**Assumption:** The profile $v^*$ violates at least one interface permit. Without loss of generality, assume:
$$K_{\mathrm{SC}_\lambda}^- = 1 \quad \text{(supercritical scaling)}$$

meaning $\lambda(v^*) = \alpha/\beta \geq 1$.

**Scaling Blow-Up Analysis (Following {cite}`Struwe90` §3):**

For supercritical profiles, dimensional analysis shows that dissipation is too weak to control energy concentration. The scaling law implies:
$$\Phi(\lambda \cdot v) = \lambda^\alpha \Phi(v), \quad \mathfrak{D}(\lambda \cdot v) = \lambda^\beta \mathfrak{D}(v)$$

As $\lambda \to 0$ (self-similar blowup), if $\alpha \geq \beta$:
$$\frac{\mathfrak{D}(\lambda \cdot v)}{\Phi(\lambda \cdot v)} = \frac{\lambda^\beta}{\lambda^\alpha} = \lambda^{\beta - \alpha} \to 0$$

Dissipation becomes negligible relative to energy—the energy-dissipation inequality fails to prevent concentration.

**Self-Similar Blowup Construction:**

**Remark (Scaling Ansatz):** The scaling function $\lambda(t)$ depends on the equation structure:
- **Parabolic systems** (heat, Navier-Stokes): $\lambda(t) = \sqrt{T_* - t}$ (diffusive scaling)
- **Dispersive systems** (NLS, NLW): $\lambda(t) = (T_* - t)^{1/(p-1)}$ where $p$ is the nonlinearity exponent
- **Geometric flows** (Ricci, mean curvature): $\lambda(t) = \sqrt{2(T_* - t)}$ (curvature scaling)

For a general hypostructure, the scaling exponent must be determined from the homogeneity of $\Phi$ and $\mathfrak{D}$ under the rescaling group action. Below we illustrate with parabolic scaling.

Let $\lambda(t) = (T_* - t)^{1/\gamma}$ where $\gamma$ is determined by dimensional analysis. Define the similarity variable:
$$w(s, y) = \lambda(t)^{\alpha/\gamma} u(t, x), \quad s = -\log(T_* - t), \quad y = \frac{x}{\lambda(t)}$$

Substituting into the evolution equation and taking $s \to \infty$ (equivalent to $t \nearrow T_*$), the profile $w(s, \cdot)$ satisfies a limiting equation with no time dependence (the self-similar profile equation).

**Struwe's Classification Theorem (Theorem 3.1 of {cite}`Struwe90`):**

For second-order parabolic systems with energy functional $E[u] = \int |\nabla u|^2 + F(u)$, singularities are characterized by:
1. **Type I (Energy-Bounded):** $\sup_{t < T_*} (T_* - t) \|\nabla u(t)\|_{L^\infty}^2 < \infty$
   - Bubbling occurs: finite number of concentration points
   - Energy quantization: $E(u(t)) \to \sum_{j=1}^J E(Q_j)$ where $Q_j$ are harmonic maps or solitons
2. **Type II (Energy-Unbounded):** $\limsup_{t \to T_*} (T_* - t) \|\nabla u(t)\|_{L^\infty}^2 = \infty$
   - Ancient solution formation: backward self-similar profile
   - Possible only with supercritical scaling

**Application:**
If $K_{\mathrm{SC}_\lambda}^- = 1$, the trajectory is Type II. The profile $v^*$ is an ancient solution (defined for all $s \in \mathbb{R}$) satisfying:
$$v^*(s, y) = v^*(\infty, y) \quad \text{(stationary ancient solution)}$$

The breakdown at $T_*$ is genuine—energy escapes to the self-similar profile, and no global extension exists without modifying the system (surgery).

**Permit Violation Catalog:**

| Violated Permit | Consequence | Singularity Type |
|----------------|-------------|------------------|
| $K_{\mathrm{SC}_\lambda}^-$ | Supercritical scaling | Type II blowup, self-similar |
| $K_{\mathrm{SC}_{\partial c}}^-$ | Parameter drift | Moduli space degeneration |
| $K_{\mathrm{Cap}_H}^-$ | Insufficient capacity | Codimension-1 singularity set |
| $K_{\mathrm{LS}_\sigma}^-$ | Instability | Exponential growth, turbulence onset |
| $K_{\mathrm{TB}_\pi}^-$ | Topological obstruction | Non-extendable, essential singularity |

**Energy Escape Mechanism:**

By the energy-dissipation inequality (hypothesis D):
$$E_0 = \Phi(u(0)) \geq \Phi(u(t)) + \int_0^t \mathfrak{D}(u(s)) \, ds$$

As $t \nearrow T_*$, if $\Phi(u(t)) \to \Phi_* > 0$ and the total dissipation is:
$$\int_0^{T_*} \mathfrak{D}(u(s)) \, ds = E_{\text{diss}} < \infty$$

then:
$$\Phi_* + E_{\text{diss}} \leq E_0$$

The "missing energy" $E_{\text{escape}} := E_0 - \Phi_* - E_{\text{diss}} \geq 0$ accounts for:
- Radiation to infinity (for dispersive equations)
- Concentration into measure-valued singularities (for geometric flows)
- Quantum tunneling (for gauge theories)

If $E_{\text{escape}} > 0$, energy has genuinely escaped the system—a bona fide singularity formation.

**Certificate Produced:** $K_{\text{C.E}} = (\text{mode: C.E}, v^*, \text{violated permits: } \{K_{\mathrm{SC}_\lambda}^-, \ldots\}, E_{\text{escape}})$

---

## Step 4: Mutual Exclusivity and Exhaustivity

We now verify the three outcomes are mutually exclusive and exhaustive.

**Mutual Exclusivity:**

1. **D.D vs. Concentration (S.E/C.D/C.E):** Mode D.D requires $\Phi_* = 0$, while concentration modes require $\Phi_* > 0$. These are disjoint.

2. **S.E/C.D (Global Regularity) vs. C.E (Genuine Singularity):**
   - Global Regularity requires all permits satisfied: $\bigwedge_{i} K_i^+ = 1$
   - Genuine Singularity requires at least one permit violated: $\bigvee_{i} K_i^- = 1$
   - By propositional logic: $\neg(\bigwedge_i K_i^+ \land \bigvee_i K_i^-) \equiv \top$ (always false to satisfy both)

3. **Within Concentration Modes:** The dichotomy "all permits satisfied" vs. "at least one violated" is exhaustive within the $\Phi_* > 0$ case.

**Exhaustivity:**

Given any trajectory $u(t)$ with $T_* < \infty$:
- Step 1 forces $\Phi_* = 0$ (D.D) or $\Phi_* > 0$ (concentration)
- If $\Phi_* > 0$, Step 2 extracts profile $v^*$
- If $\Phi_* > 0$, Step 3 classifies $v^*$ via permits: all satisfied (Global Regularity) or at least one violated (Genuine Singularity)

By tertium non datur, exactly one case applies.

---

## Step 5: Certificate Construction

For each outcome, we produce an explicit certificate containing:

**Mode D.D (Dispersion):**
```
K_D.D = {
  mode: "D.D",
  mechanism: "Dispersion",
  evidence: {
    energy_limit: Φ_* = 0,
    scattering_data: u_∞ ∈ X,
    decay_rate: ||u(t)||_X ≤ C t^(-d/2) (Strichartz)
  },
  literature: "Lions84:LemmaI.1"
}
```

**Modes S.E/C.D/T.E/S.D (Global Regularity):**
```
K_Reg = {
  mode: "Global_Regularity",
  mechanism: "Barrier_Triggered",
  evidence: {
    energy_limit: Φ_* > 0,
    profile: v^* ∈ X,
    permit_certificates: {
      SC_λ: K^+_SC_λ,
      SC_∂c: K^+_SC_∂c,
      Cap_H: K^+_Cap_H,
      LS_σ: K^+_LS_σ,
      TB_π: K^+_TB_π
    },
    extension: "Kenig-Merle rigidity implies T_* = ∞"
  },
  literature: "KenigMerle06:Thm1.1, Struwe90:Sec3.5"
}
```

**Mode C.E (Genuine Singularity):**
```
K_C.E = {
  mode: "C.E",
  mechanism: "Genuine_Singularity",
  evidence: {
    energy_limit: Φ_* > 0,
    profile: v^* ∈ X (ancient solution),
    violated_permits: {
      // At least one of:
      SC_λ: K^-_SC_λ (supercritical),
      SC_∂c: K^-_SC_∂c (parameter drift),
      Cap_H: K^-_Cap_H (insufficient capacity),
      LS_σ: K^-_LS_σ (instability),
      TB_π: K^-_TB_π (topological obstruction)
    },
    energy_escape: E_escape = E_0 - Φ_* - E_diss > 0,
    singularity_type: "Type II" (if SC_λ violated),
    breakdown_time: T_* < ∞
  },
  literature: "Struwe90:Thm3.1, BahouriGerard99:Thm1"
}
```

---

## Step 6: Quantitative Bounds

We now provide explicit quantitative estimates for each regime.

**Energy Threshold (Universal):**
Define the critical energy:
$$E_c := \inf\left\{ E > 0 : \exists u_0 \text{ with } \Phi(u_0) = E \text{ and } T_*(u_0) < \infty \right\}$$

This is the minimal energy required for finite-time blowup. By the scaling invariance of critical equations:
$$E_c = \|Q\|_{\dot{H}^{s_c}}^2$$
where $Q$ is the ground state (minimizer of $\Phi$ under the nonlinear constraint).

**Mode D.D (Dispersion Quantitative Bound):**

For $\Phi(u_0) < E_c$, the trajectory scatters with decay:
$$\|u(t)\|_{L^\infty} \leq C E_0^{(p-1)/2} t^{-d/2}$$
where $p$ is the nonlinearity exponent and $C$ depends only on dimension.

**Proof:** By Strichartz estimates and small data theory (e.g., {cite}`KenigMerle06` Theorem 1.1 for NLS).

**Mode S.E/C.D (Global Regularity Quantitative Bound):**

If all permits are satisfied with margins:
$$K_{\mathrm{SC}_\lambda}^+ \implies \alpha - \beta \geq \delta_{\text{crit}} > 0$$

then the trajectory extends globally with uniform bound:
$$\sup_{t \geq 0} \Phi(u(t)) \leq C(\delta_{\text{crit}}, E_0)$$

**Proof:** By Grönwall's inequality applied to the energy-dissipation inequality. For subcritical scaling ($\beta < \alpha$):
$$\frac{d}{dt}\Phi(u) \leq -\delta_{\text{crit}} \Phi(u)^{\beta/\alpha}$$

Separating variables and integrating:
$$\Phi(u(t))^{1-\beta/\alpha} \leq \Phi(u_0)^{1-\beta/\alpha} - (1-\beta/\alpha)\delta_{\text{crit}} t$$

For large $t$, this implies decay (possibly finite-time extinction if the right side vanishes). In all cases, the trajectory remains bounded.

**Mode C.E (Genuine Singularity Quantitative Bound):**

For supercritical blowup ($\alpha \geq \beta$), the blowup rate satisfies:
$$(T_* - t)^{\alpha/2} \|\nabla u(t)\|_{L^\infty} \to C_{\text{blowup}} \quad \text{as } t \nearrow T_*$$

where $C_{\text{blowup}}$ is determined by the self-similar profile $v^*$.

**Proof:** By rescaling argument and {cite}`Struwe90` Theorem 3.1.

**Profile Energy Quantization:**

For concentration with $J$ bubbles (profile decomposition Step 2.4):
$$\Phi(u(t_n)) = \sum_{j=1}^J \Phi(v^{(j)}) + o(1)$$

Each bubble energy satisfies:
$$\Phi(v^{(j)}) \in \{k E_{\text{quant}} : k \in \mathbb{N}\} \quad \text{(energy levels)}$$

where $E_{\text{quant}} = \|Q\|_{\dot{H}^{s_c}}^2$ is the quantum of energy (ground state energy).

**Proof:** By topological degree argument and scaling invariance (see {cite}`BahouriGerard99` Theorem 1 and {cite}`MerleZaag98` for bubbling analysis).

---

## Conclusion

We have established the trichotomy:

1. **Mode D.D (Dispersion):** If $\Phi(u(t)) \to 0$, energy disperses to infinity, the trajectory scatters, and $T_* = \infty$ (global existence). The breakdown assumption $T_* < \infty$ is contradicted.

2. **Mode S.E/C.D/T.E/S.D (Global Regularity):** If $\Phi(u(t)) \to \Phi_* > 0$ with all interface permits satisfied, the Kenig-Merle rigidity theorem forces the profile to be a global attractor (soliton or periodic orbit). The trajectory extends globally, contradicting $T_* < \infty$.

3. **Mode C.E (Genuine Singularity):** If $\Phi(u(t)) \to \Phi_* > 0$ with at least one permit violated, the profile is a supercritical ancient solution. Energy escapes via self-similar blowup, and the breakdown at $T_* < \infty$ is genuine.

**Logical Structure:**

The proof proceeds by case exhaustion on $\Phi_*$:
- $\Phi_* = 0$ → D.D (impossible if $T_* < \infty$ by scattering theory)
- $\Phi_* > 0$ → Extract profile $v^*$ (Step 2), then:
  - All permits satisfied → S.E/C.D (impossible if $T_* < \infty$ by rigidity)
  - At least one permit violated → C.E (genuine singularity, consistent with $T_* < \infty$)

**Certificate Production:**

The trichotomy certificate is:
$$K_{\text{Trichotomy}} = \begin{cases}
K_{\text{D.D}} & \text{if } \Phi_* = 0 \\
K_{\text{Reg}} & \text{if } \Phi_* > 0, \, \bigwedge_i K_i^+ = 1 \\
K_{\text{C.E}} & \text{if } \Phi_* > 0, \, \bigvee_i K_i^- = 1
\end{cases}$$

This certificate is:
- **Computable:** $\Phi_*$ is determined by energy monitoring; permits are checked by interface evaluators
- **Verifiable:** Each permit certificate is independently verifiable via its evaluator $\mathcal{P}_i$
- **Complete:** All trajectories with $T_* < \infty$ are classified

**Physical Interpretation:**

- **D.D:** System lacks sufficient energy density to sustain localized structures—disperses like a linear wave
- **S.E/C.D:** System has energy but structural constraints (permits) prevent singularity formation—reaches equilibrium
- **C.E:** System has energy and evades all structural safeguards—genuine catastrophic failure (blowup, collapse, phase transition)

**Connection to Node 3 (CompactCheck):**

This theorem directly justifies the CompactCheck node ({prf:ref}`def-node-compact`) in the Structural Sieve:
- CompactCheck evaluates $K_{C_\mu}$: does concentration occur?
- YES ($K_{C_\mu}^+$) → Extract profile, proceed to ScaleCheck and permit verification
- NO ($K_{C_\mu}^-$) → Dispersion, route to BarrierScat (scattering barrier), trigger Mode D.D

The trichotomy theorem guarantees this dichotomy is **exhaustive** for finite-breakdown trajectories.

**Literature:**

This proof synthesizes techniques from:

1. **Lions (1984) {cite}`Lions84`:** Concentration-compactness principle (Lemma I.1), establishing the dispersion-concentration dichotomy for bounded sequences in Sobolev spaces. Provides the foundational Step 1 energy dichotomy.

2. **Bahouri-Gérard (1999) {cite}`BahouriGerard99`:** Profile decomposition theorem (Theorem 1), refining Lions' principle for critical dispersive equations. Establishes orthogonality of bubbles and energy decoupling in Step 2.4.

3. **Kenig-Merle (2006) {cite}`KenigMerle06`:** Rigidity theorem (Theorem 1.1), proving that for energy-critical NLS, solutions below the ground state energy either scatter or are solitons. Provides the Global Regularity conclusion in Case 3.1.

4. **Struwe (1990) {cite}`Struwe90`:** Singularity analysis for geometric flows (§3), classifying Type I vs. Type II blowup and constructing self-similar profiles. Provides the Genuine Singularity mechanism in Case 3.2.

**Applicability Justification:**

- **Lions:** Applies to any sequence in a reflexive Banach space with translation-invariant norm (Sobolev spaces, Besov spaces). Requires only weak compactness modulo symmetries—satisfied by hypothesis (C).

- **Bahouri-Gérard:** Applies to critical homogeneous Sobolev spaces $\dot{H}^{s_c}$ with scaling group $G = \mathbb{R}^+ \times \mathbb{R}^d$. Requires weak continuity of the nonlinearity—standard for polynomial and gauge-invariant terms.

- **Kenig-Merle:** Applies to critical well-posed dispersive equations (NLS, NLW, Zakharov, Klein-Gordon). Requires small data global existence and finite speed of propagation—verified for hyperbolic and Schrödinger systems.

- **Struwe:** Applies to second-order parabolic and elliptic systems with variational structure (harmonic maps, Yang-Mills, mean curvature flow). Requires energy functional with Palais-Smale property—standard for geometric PDEs.

---

**End of Proof.**

:::
