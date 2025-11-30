# Étude 2: Navier-Stokes Regularity via Hypostructure

## 0. Introduction

**Problem 0.1 (Navier-Stokes Millennium Problem).** Let $u_0 \in C^\infty(\mathbb{R}^3)$ be a divergence-free vector field with $|D^\alpha u_0(x)| \leq C_{\alpha,K}(1 + |x|)^{-K}$ for all $\alpha, K$. Does there exist a smooth solution $u: \mathbb{R}^3 \times [0, \infty) \to \mathbb{R}^3$ to the Navier-Stokes equations with $u(0) = u_0$?

We construct a hypostructure $\mathbb{H}_{NS} = (X, S_t, \Phi, \mathfrak{D}, G)$ and identify which axioms are verified, which remain open, and how the metatheorems constrain possible singularity formation.

---

## 1. The Navier-Stokes Equations

### 1.1 The PDE System

**Definition 1.1.1.** The incompressible Navier-Stokes equations on $\mathbb{R}^3$ are:
$$\partial_t u + (u \cdot \nabla)u = -\nabla p + \nu \Delta u$$
$$\nabla \cdot u = 0$$
where $u: \mathbb{R}^3 \times [0, T) \to \mathbb{R}^3$ is the velocity field, $p: \mathbb{R}^3 \times [0, T) \to \mathbb{R}$ is the pressure, and $\nu > 0$ is the kinematic viscosity.

**Proposition 1.1.2 (Pressure Recovery).** Given $u \in L^2_\sigma(\mathbb{R}^3) \cap L^q(\mathbb{R}^3)$ for some $q > 3$, the pressure $p$ is uniquely determined (up to constant) by:
$$-\Delta p = \partial_i \partial_j (u_i u_j) = \text{tr}(\nabla u \cdot \nabla u^T)$$
with $p(x) \to 0$ as $|x| \to \infty$. The solution is:
$$p = \sum_{i,j=1}^{3} R_i R_j (u_i u_j)$$
where $R_i := \partial_i (-\Delta)^{-1/2}$ is the $i$-th Riesz transform, satisfying $\|R_i f\|_{L^p} \leq C_p \|f\|_{L^p}$ for $1 < p < \infty$.

*Proof.* Apply $\nabla \cdot$ to the momentum equation and use $\nabla \cdot u = 0$:
$$\nabla \cdot \partial_t u = 0, \quad \nabla \cdot (\nu \Delta u) = \nu \Delta (\nabla \cdot u) = 0$$
$$\nabla \cdot \nabla p = \Delta p, \quad \nabla \cdot ((u \cdot \nabla)u) = \partial_i(u_j \partial_j u_i) = \partial_i \partial_j(u_i u_j)$$
where we used $\partial_j u_j = 0$. Inverting $-\Delta$ via the Newtonian potential and taking derivatives gives the Riesz transform representation. $\square$

**Definition 1.1.3.** The Leray projector $\mathbb{P}: L^2(\mathbb{R}^3)^3 \to L^2_\sigma(\mathbb{R}^3)$ onto divergence-free fields is:
$$\mathbb{P} = I + \nabla(-\Delta)^{-1}\nabla \cdot$$
In Fourier space: $\widehat{\mathbb{P}f}(\xi) = (I - \frac{\xi \otimes \xi}{|\xi|^2})\hat{f}(\xi)$.

**Definition 1.1.4.** The projected Navier-Stokes equation is:
$$\partial_t u = \nu \Delta u - \mathbb{P}((u \cdot \nabla)u) =: \nu \Delta u - B(u, u)$$
where $B(u, v) := \mathbb{P}((u \cdot \nabla)v)$ is the bilinear form.

### 1.2 Function Spaces

**Definition 1.2.1.** The energy space is:
$$L^2_\sigma(\mathbb{R}^3) := \overline{\{u \in C_c^\infty(\mathbb{R}^3)^3 : \nabla \cdot u = 0\}}^{L^2}$$

**Definition 1.2.2.** The homogeneous Sobolev spaces are:
$$\dot{H}^s(\mathbb{R}^3) := \{f \in \mathcal{S}'(\mathbb{R}^3) : |\xi|^s \hat{f} \in L^2(\mathbb{R}^3)\}$$
with norm $\|f\|_{\dot{H}^s} := \||\xi|^s \hat{f}\|_{L^2}$.

**Definition 1.2.3.** The critical space for Navier-Stokes is $\dot{H}^{1/2}(\mathbb{R}^3)$, characterized by scale-invariance: if $u(x, t)$ solves NS, then so does:
$$u_\lambda(x, t) := \lambda u(\lambda x, \lambda^2 t)$$
and $\|u_\lambda(\cdot, 0)\|_{\dot{H}^{1/2}} = \|u(\cdot, 0)\|_{\dot{H}^{1/2}}$.

---

## 2. The Hypostructure Data

### 2.1 State Space

**Definition 2.1.1.** The state space is:
$$X := L^2_\sigma(\mathbb{R}^3) \cap \dot{H}^{1/2}(\mathbb{R}^3)$$
with norm $\|u\|_X := \|u\|_{L^2} + \|u\|_{\dot{H}^{1/2}}$.

**Proposition 2.1.2.** $(X, \|\cdot\|_X)$ is a separable Banach space, hence Polish.

### 2.2 The Semiflow

**Definition 2.2.1.** For $u_0 \in X$, the maximal existence time is:
$$T_*(u_0) := \sup\{T > 0 : \exists \text{ mild solution } u \in C([0, T); X) \cap L^2(0, T; \dot{H}^{3/2})\}$$

**Theorem 2.2.2 (Kato [K84]).** For each $u_0 \in X$:
1. **(Local existence)** There exists $T_* = T_*(u_0) \in (0, \infty]$ and a unique function $u \in C([0, T_*); X) \cap L^2_{loc}([0, T_*); \dot{H}^{3/2})$ satisfying the integral equation:
$$u(t) = e^{\nu t \Delta} u_0 - \int_0^t e^{\nu(t-s)\Delta} B(u(s), u(s)) \, ds$$
2. **(Continuous dependence)** The map $u_0 \mapsto u(t)$ is continuous from $X$ to $C([0, T]; X)$ for $T < T_*(u_0)$.
3. **(Lower bound on existence time)** There exists $c > 0$ depending only on $\nu$ such that $T_* \geq c/\|u_0\|_{\dot{H}^{1/2}}^4$.

*Proof sketch.* Define $\Psi(u)(t) := e^{\nu t\Delta}u_0 - \int_0^t e^{\nu(t-s)\Delta}B(u,u)(s)\,ds$. Using the heat kernel estimates $\|e^{\nu t\Delta}f\|_{\dot{H}^{s+\alpha}} \leq C t^{-\alpha/2}\|f\|_{\dot{H}^s}$ and the bilinear estimate $\|B(u,v)\|_{\dot{H}^{-1/2}} \leq C\|u\|_{\dot{H}^{1/2}}\|v\|_{\dot{H}^{1/2}}$, one shows $\Psi$ is a contraction on a ball in $C([0,T]; \dot{H}^{1/2})$ for $T$ sufficiently small. $\square$

**Theorem 2.2.3 (Blow-up Criterion).** If $T_* = T_*(u_0) < \infty$, then:
$$\lim_{t \nearrow T_*} \|u(t)\|_{\dot{H}^{1/2}} = \infty$$
Equivalently, the enstrophy integral diverges: $\int_0^{T_*} \|\nabla u(t)\|_{L^2}^2 \, dt = \infty$.

**Definition 2.2.4.** The semiflow $S_t: X \to X$ is defined for $t < T_*(u_0)$ by:
$$S_t(u_0) := u(t)$$

### 2.3 Height Functional (Energy)

**Definition 2.3.1.** The kinetic energy is:
$$E(u) := \frac{1}{2}\|u\|_{L^2}^2 = \frac{1}{2}\int_{\mathbb{R}^3} |u(x)|^2 \, dx$$

**Theorem 2.3.2 (Energy Inequality).** For Leray-Hopf weak solutions:
$$E(u(t)) + \nu \int_0^t \|\nabla u(s)\|_{L^2}^2 \, ds \leq E(u_0)$$

**Definition 2.3.3.** The height functional is $\Phi := E: X \to [0, \infty)$.

### 2.4 Dissipation Functional (Enstrophy)

**Definition 2.4.1.** The enstrophy (dissipation rate) is:
$$\mathfrak{D}(u) := \nu \|\nabla u\|_{L^2}^2 = \nu \|\omega\|_{L^2}^2$$
where $\omega := \nabla \times u$ is the vorticity.

**Proposition 2.4.2.** For smooth solutions:
$$\frac{d}{dt} E(u(t)) = -\mathfrak{D}(u(t))$$

*Proof.* Multiply the Navier-Stokes equation by $u$ and integrate:
$$\int u \cdot \partial_t u = \int u \cdot (\nu \Delta u) - \int u \cdot \nabla p - \int u \cdot (u \cdot \nabla)u$$

The pressure term vanishes: $\int u \cdot \nabla p = -\int p \nabla \cdot u = 0$.

The nonlinear term vanishes: $\int u \cdot (u \cdot \nabla)u = \frac{1}{2}\int (u \cdot \nabla)|u|^2 = -\frac{1}{2}\int |u|^2 \nabla \cdot u = 0$.

The viscous term: $\int u \cdot \Delta u = -\int |\nabla u|^2 = -\nu^{-1}\mathfrak{D}(u)$. $\square$

### 2.5 Symmetry Group

**Definition 2.5.1.** The Navier-Stokes symmetry group is:
$$G := \mathbb{R}^3 \rtimes (SO(3) \times \mathbb{R}_{>0})$$
acting by:
- Translation: $(\tau_a u)(x) := u(x - a)$
- Rotation: $(R_\theta u)(x) := R_\theta u(R_\theta^{-1} x)$
- Scaling: $(\sigma_\lambda u)(x, t) := \lambda u(\lambda x, \lambda^2 t)$

**Proposition 2.5.2.** The Navier-Stokes equations are $G$-equivariant.

---

## 3. Verification of Axiom C (Compactness)

**Theorem 3.1 (Rellich-Kondrachov).** For bounded $\Omega \subset \mathbb{R}^3$:
$$H^1(\Omega) \hookrightarrow \hookrightarrow L^q(\Omega), \quad 1 \leq q < 6$$

**Theorem 3.2 (Concentration-Compactness for NS).** Let $(u_n) \subset X$ with $\sup_n E(u_n) \leq E_0$. Then there exist:
1. A subsequence (still denoted $u_n$)
2. Sequences $(x_n^j)_{j \geq 1} \subset \mathbb{R}^3$ and $(\lambda_n^j)_{j \geq 1} \subset \mathbb{R}_{>0}$
3. Profiles $(U^j)_{j \geq 1} \subset X$

such that:
$$u_n = \sum_{j=1}^J (\lambda_n^j)^{1/2} U^j((\lambda_n^j)(\cdot - x_n^j)) + w_n^J$$
where $\|w_n^J\|_{L^q} \to 0$ as $n \to \infty$ then $J \to \infty$ for $2 < q < 6$.

*Proof.* Apply the profile decomposition of Gérard [G98] adapted to the NS scaling. The critical Sobolev embedding $\dot{H}^{1/2} \hookrightarrow L^3$ fails to be compact, but concentration at isolated scales/locations is captured by the profiles. $\square$

**Proposition 3.3 (Axiom C: Partial).** On bounded subsets of $X$ with:
$$\sup_n \|u_n\|_{L^2} \leq M, \quad \sup_n \|u_n\|_{\dot{H}^1} \leq M$$
the sequence $(u_n)$ is precompact in $L^2_{loc}$.

*Proof.* The $\dot{H}^1$ bound gives compactness in $L^2_{loc}$ by Rellich-Kondrachov. $\square$

**Remark 3.4.** Full Axiom C (global precompactness in $X$) is not available due to the critical nature of $\dot{H}^{1/2}$ and non-compactness of $\mathbb{R}^3$.

---

## 4. Verification of Axiom D (Dissipation)

**Theorem 4.1 (Energy-Dissipation Identity).** For smooth solutions on $[0, T]$:
$$E(u(T)) + \int_0^T \mathfrak{D}(u(t)) \, dt = E(u(0))$$

*Proof.* Integrate Proposition 2.4.2. $\square$

**Corollary 4.2.** Axiom D holds with $C = 0$:
$$\Phi(S_t u_0) + \int_0^t \mathfrak{D}(S_s u_0) \, ds = \Phi(u_0)$$

**Corollary 4.3.** The total dissipation cost is bounded:
$$\mathcal{C}_*(u_0) := \int_0^{T_*} \mathfrak{D}(u(t)) \, dt \leq E(u_0) < \infty$$

---

## 5. Verification of Axiom SC (Scaling Structure)

**Definition 5.1.** The scaling dimensions for Navier-Stokes are:
- $[u] = -1$ (velocity scales as $\lambda^{-1}$)
- $[t] = -2$ (time scales as $\lambda^{-2}$)
- $[\nabla] = 1$
- $[E] = -1$ (energy scales as $\lambda^{-1}$ in 3D)
- $[\mathfrak{D}] = 1$ (enstrophy scales as $\lambda$)

**Proposition 5.2.** Under the scaling $u_\lambda(x, t) = \lambda u(\lambda x, \lambda^2 t)$:
$$E(u_\lambda(0)) = \lambda^{-1} E(u(0))$$
$$\int_0^{T/\lambda^2} \mathfrak{D}(u_\lambda(t)) \, dt = \lambda^{-1} \int_0^T \mathfrak{D}(u(t)) \, dt$$

*Proof.* Direct computation:
$$E(u_\lambda) = \frac{1}{2}\int |\lambda u(\lambda x)|^2 dx = \frac{\lambda^2}{2} \int |u(\lambda x)|^2 dx = \frac{\lambda^2}{2\lambda^3} \int |u(y)|^2 dy = \lambda^{-1} E(u)$$

Similarly for the dissipation integral. $\square$

**Theorem 5.3 (Criticality).** The Navier-Stokes equations are critical: $\alpha = \beta$ where:
- $\alpha = 1$ is the scaling exponent of energy
- $\beta = 1$ is the scaling exponent of dissipation cost

*Proof.* Both $E$ and $\mathcal{C}_*$ scale as $\lambda^{-1}$. $\square$

**Corollary 5.4 (Theorem 7.2 Inapplicable).** The condition $\alpha > \beta$ for Type II exclusion is not satisfied. Theorem 7.2 does not exclude Type II blow-up for Navier-Stokes.

**Remark 5.5.** This is the fundamental obstruction. For supercritical problems ($\alpha > \beta$), Type II blow-up is excluded by scaling. For critical problems ($\alpha = \beta$), both Type I and Type II remain possible a priori.

---

## 6. Critical Norms and Blow-up Criteria

### 6.1 Scaling-Invariant Norms

**Definition 6.1.1.** A norm $\|\cdot\|_Y$ is critical for NS if:
$$\|u_\lambda\|_Y = \|u\|_Y$$
for all $\lambda > 0$.

**Proposition 6.1.2.** The following norms are critical:
- $\|u\|_{L^3(\mathbb{R}^3)}$
- $\|u\|_{\dot{H}^{1/2}(\mathbb{R}^3)}$
- $\|u\|_{\dot{B}^{-1+3/p}_{p,\infty}(\mathbb{R}^3)}$ for $3 < p < \infty$
- $\|u\|_{BMO^{-1}(\mathbb{R}^3)}$

### 6.2 Blow-up Criteria

**Theorem 6.2.1 (Escauriaza-Seregin-Šverák [ESS03]).** If $T_* < \infty$, then:
$$\limsup_{t \to T_*} \|u(t)\|_{L^3(\mathbb{R}^3)} = \infty$$

**Theorem 6.2.2 (Ladyzhenskaya-Prodi-Serrin).** The solution is regular on $[0, T]$ if:
$$u \in L^p(0, T; L^q(\mathbb{R}^3)), \quad \frac{2}{p} + \frac{3}{q} = 1, \quad 3 < q \leq \infty$$

**Theorem 6.2.3 (Beale-Kato-Majda [BKM84]).** For Euler equations ($\nu = 0$), blow-up requires:
$$\int_0^{T_*} \|\omega(t)\|_{L^\infty} \, dt = \infty$$

For Navier-Stokes, this remains a blow-up criterion but is not known to be necessary.

---

## 7. Partial Verification of Axiom LS (Local Stiffness)

**Definition 7.1.** The zero solution $u \equiv 0$ is the unique equilibrium for Navier-Stokes on $\mathbb{R}^3$ with finite energy.

**Theorem 7.2 (Stability of Zero).** For $\|u_0\|_{\dot{H}^{1/2}}$ sufficiently small, the solution exists globally and:
$$\|u(t)\|_{\dot{H}^{1/2}} \leq C\|u_0\|_{\dot{H}^{1/2}} e^{-c\nu t}$$

*Proof.* Small data global existence in $\dot{H}^{1/2}$ follows from Kato's theorem with a contraction argument. The exponential decay follows from the spectral gap of the Stokes operator. $\square$

**Proposition 7.3 (Łojasiewicz Inequality at Zero).** Near $u = 0$:
$$\mathfrak{D}(u) = \nu\|\nabla u\|_{L^2}^2 \geq c\|u\|_{L^2}^2 = 2c \cdot E(u)$$
by Poincaré inequality (on bounded domains) or Hardy inequality.

**Remark 7.4.** Axiom LS holds at the equilibrium $u = 0$. The open question is whether non-zero steady states or time-periodic solutions exist that could serve as alternative attractors.

---

## 8. Partial Verification of Axiom Cap (Capacity)

**Definition 8.0 (Suitable Weak Solution).** A pair $(u, p)$ is a *suitable weak solution* on $\mathbb{R}^3 \times (0, T)$ if:
1. $u \in L^\infty(0, T; L^2) \cap L^2(0, T; \dot{H}^1)$ and $p \in L^{5/3}_{loc}(\mathbb{R}^3 \times (0, T))$
2. $(u, p)$ satisfies NS in the sense of distributions
3. The local energy inequality holds: for a.e. $t$ and all non-negative $\phi \in C_c^\infty(\mathbb{R}^3 \times (0, T))$:
$$\int |u|^2 \phi \, dx \Big|_t + 2\nu \int_0^t \int |\nabla u|^2 \phi \leq \int_0^t \int |u|^2(\partial_t \phi + \nu \Delta \phi) + \int_0^t \int (|u|^2 + 2p)(u \cdot \nabla \phi)$$

**Theorem 8.1 (Caffarelli-Kohn-Nirenberg [CKN82]).** Let $(u, p)$ be a suitable weak solution on $\mathbb{R}^3 \times (0, T)$. Define the singular set:
$$\Sigma := \{(x, t) \in \mathbb{R}^3 \times (0, T) : u \notin L^\infty(B_r(x) \times (t - r^2, t)) \text{ for all } r > 0\}$$
Then the 1-dimensional parabolic Hausdorff measure vanishes: $\mathcal{P}^1(\Sigma) = 0$.

*Proof.*
**(i) Scaled quantities.** For $(x_0, t_0) \in \mathbb{R}^3 \times (0, T)$ and $r > 0$ with $t_0 - r^2 > 0$, define:
$$A(r) := \sup_{t_0 - r^2 < t < t_0} \frac{1}{r} \int_{B_r(x_0)} |u|^2 \, dx$$
$$C(r) := \frac{1}{r^2} \int_{t_0 - r^2}^{t_0} \int_{B_r(x_0)} |u|^3 \, dx \, dt$$
$$D(r) := \frac{1}{r^2} \int_{t_0 - r^2}^{t_0} \int_{B_r(x_0)} |p - p_{B_r}|^{3/2} \, dx \, dt$$
$$E(r) := \frac{1}{r} \int_{t_0 - r^2}^{t_0} \int_{B_r(x_0)} |\nabla u|^2 \, dx \, dt$$

**(ii) Regularity criterion.** There exists $\epsilon_0 > 0$ universal such that if $\limsup_{r \to 0} (C(r) + D(r)) < \epsilon_0$, then $(x_0, t_0)$ is a regular point: $u \in L^\infty$ near $(x_0, t_0)$.

**(iii) Energy control.** From the local energy inequality and Hölder estimates:
$$C(r) + D(r) \leq C(E(r)^{3/4} A(r)^{1/4} + E(r)^{3/2})$$

**(iv) Covering argument.** Let $\Sigma_\epsilon := \{(x, t) : C(r) + D(r) \geq \epsilon_0 \text{ for all } r \leq \epsilon\}$. Cover $\Sigma_\epsilon$ by parabolic cylinders $Q_{r_i}$ with $r_i \leq \epsilon$ and $C(r_i) + D(r_i) \geq \epsilon_0$. The Vitali covering lemma gives:
$$\sum_i r_i \leq C \epsilon_0^{-1} \int_0^T \int |\nabla u|^2 \leq C \epsilon_0^{-1} E(u_0)$$

**(v) Conclusion.** $\mathcal{P}^1(\Sigma_\epsilon) \leq C \epsilon_0^{-1} E(u_0)$ independent of $\epsilon$. Since $\Sigma = \bigcap_{\epsilon > 0} \Sigma_\epsilon$ and the bound is uniform, $\mathcal{P}^1(\Sigma) = 0$. $\square$

**Corollary 8.2.** The spatial singular set at any time has Hausdorff dimension at most 1:
$$\dim_H(\Sigma_t) \leq 1$$

**Proposition 8.3 (Axiom Cap: Partial).** Singularities cannot fill positive-capacity sets. Specifically:
$$\text{Cap}_{1,2}(\Sigma_t) = 0$$

*Proof.* Sets of Hausdorff dimension $\leq 1$ in $\mathbb{R}^3$ have zero $(1,2)$-capacity. $\square$

---

## 9. The Regularity Gap

### 9.1 What Is Known

**Theorem 9.1 (Summary of Verified Axioms).**

| Axiom | Status | Reference |
|:------|:-------|:----------|
| C (Compactness) | Partial (local, with extra derivative) | Theorem 3.3 |
| D (Dissipation) | Verified | Theorem 4.1 |
| SC (Scaling) | Critical ($\alpha = \beta$) | Theorem 5.3 |
| LS (Local Stiffness) | Verified at $u = 0$ | Theorem 7.2 |
| Cap (Capacity) | Partial ($\dim \Sigma \leq 1$) | Theorem 8.1 |
| R (Recovery) | Open | — |
| TB (Topological) | N/A (contractible state space) | — |

### 9.2 What Is Missing

**Open Problem 9.2.** Verify Axiom R (Recovery) for Navier-Stokes: show that trajectories spending time in "wild" regions (high enstrophy) must dissipate proportionally.

**Conjecture 9.3 (Axiom R for NS).** There exists $c_R > 0$ such that:
$$\int_0^T \mathbf{1}_{\{\|\omega(t)\|_{L^\infty} > \Lambda\}} \, dt \leq c_R^{-1} \Lambda^{-\gamma} \int_0^T \mathfrak{D}(u(t)) \, dt$$
for some $\gamma > 0$.

**Remark 9.4.** If Conjecture 9.3 holds, combined with the CKN partial regularity, it would imply global regularity.

---

## 10. Application of Metatheorems

### 10.1 Theorem 7.1 (Structural Resolution)

**Application.** Every finite-energy trajectory either:
1. Exists globally and decays to zero
2. Blows up at finite time $T_* < \infty$

The dichotomy is established; the question is which alternative occurs.

### 10.2 Theorem 7.3 (Capacity Barrier)

**Application.** By CKN (Theorem 8.1), any blow-up occurs on a set of dimension $\leq 1$. This is the capacity barrier in action: high-dimensional blow-up sets are excluded.

**Corollary 10.1.** If blow-up occurs, it is necessarily of "sparse" type—concentrated on thin space-time filaments.

### 10.3 Theorem 9.10 (Coherence Quotient)

**Definition 10.2.** The coherence quotient for NS:
$$\mathcal{Q}(u) := \frac{\|u \otimes u - \frac{1}{3}|u|^2 I\|_{L^{3/2}}}{\|u\|_{L^3}^2}$$
measures deviation from isotropic turbulence.

**Conjecture 10.3.** Near blow-up, $\mathcal{Q}(u(t)) \to 0$ (flow becomes increasingly aligned/coherent).

### 10.4 Theorem 9.14 (Spectral Convexity)

**Application.** The energy spectrum $E(k, t) := \frac{1}{2}\int_{|\xi|=k} |\hat{u}(\xi, t)|^2 dS(\xi)$ satisfies convexity properties that constrain possible blow-up scenarios.

### 10.5 Theorem 9.90 (Hyperbolic Shadowing)

**Application.** Near the stable equilibrium $u = 0$, small perturbations decay exponentially. This is the shadowing property in the dissipative regime.

### 10.6 Theorem 9.120 (Dimensional Rigidity)

**Application.** Blow-up cannot change the "effective dimension" of the solution. Self-similar blow-up profiles must respect the 3D structure.

---

## 11. Self-Similar Blow-up Analysis

### 11.1 Self-Similar Ansatz

**Definition 11.1.** A Type I blow-up at $(0, T_*)$ has the form:
$$u(x, t) = \frac{1}{\sqrt{T_* - t}} U\left(\frac{x}{\sqrt{T_* - t}}\right)$$
where $U: \mathbb{R}^3 \to \mathbb{R}^3$ is the blow-up profile.

**Proposition 11.2.** The profile $U$ satisfies:
$$\nu \Delta U - \frac{1}{2}U - \frac{1}{2}(y \cdot \nabla)U - (U \cdot \nabla)U + \nabla P = 0$$
$$\nabla \cdot U = 0$$

### 11.2 Exclusion Results

**Theorem 11.3 (Nečas-Růžička-Šverák [NRS96]).** There is no non-trivial self-similar blow-up with $U \in L^3(\mathbb{R}^3)$.

*Proof.* Multiply the profile equation by $U$ and integrate. Use the criticality of $L^3$ and Sobolev inequalities to derive a contradiction unless $U = 0$. $\square$

**Theorem 11.4 (Tsai [T98]).** There is no non-trivial self-similar blow-up with $U \in L^p(\mathbb{R}^3)$ for any $p > 3$.

**Remark 11.5.** These results exclude "nice" self-similar blow-up but leave open singular self-similar profiles or non-self-similar blow-up.

---

## 12. Enstrophy Evolution

### 12.1 The Enstrophy Equation

**Theorem 12.1.** For smooth solutions, the enstrophy $\Omega := \frac{1}{2}\|\omega\|_{L^2}^2$ satisfies:
$$\frac{d\Omega}{dt} = -\nu\|\nabla \omega\|_{L^2}^2 + \int_{\mathbb{R}^3} \omega \cdot (\omega \cdot \nabla)u \, dx$$

*Proof.* The vorticity equation is:
$$\partial_t \omega + (u \cdot \nabla)\omega = (\omega \cdot \nabla)u + \nu \Delta \omega$$

Multiply by $\omega$ and integrate:
$$\frac{1}{2}\frac{d}{dt}\|\omega\|_{L^2}^2 = \nu \int \omega \cdot \Delta \omega + \int \omega \cdot (\omega \cdot \nabla)u$$

The transport term vanishes: $\int \omega \cdot (u \cdot \nabla)\omega = 0$. $\square$

### 12.2 The Vortex Stretching Term

**Definition 12.2.** The vortex stretching term is:
$$\mathcal{S}(\omega, u) := \int_{\mathbb{R}^3} \omega \cdot (\omega \cdot \nabla)u \, dx = \int_{\mathbb{R}^3} \omega_i \omega_j S_{ij} \, dx$$
where $S_{ij} = \frac{1}{2}(\partial_i u_j + \partial_j u_i)$ is the strain tensor.

**Proposition 12.3 (Constantin-Fefferman [CF93]).** The stretching term satisfies:
$$|\mathcal{S}(\omega, u)| \leq C\|\omega\|_{L^2}^{3/2}\|\nabla \omega\|_{L^2}^{3/2}$$

**Corollary 12.4.** By Young's inequality:
$$\frac{d\Omega}{dt} \leq -\frac{\nu}{2}\|\nabla\omega\|_{L^2}^2 + C\nu^{-3}\Omega^3$$

This shows enstrophy can grow at most doubly-exponentially in time.

---

## 13. The Critical Threshold

### 13.1 Known Conditional Results

**Theorem 13.1 (Regularity Below Critical Threshold).** There exists $\epsilon_* > 0$ such that if:
$$\|u_0\|_{\dot{H}^{1/2}} < \epsilon_*$$
then the solution exists globally and decays.

**Theorem 13.2 (Gallagher-Koch-Planchon [GKP16]).** Global regularity holds if:
$$\|u_0\|_{\dot{B}^{-1}_{\infty,\infty}} < c\nu$$
where $\dot{B}^{-1}_{\infty,\infty}$ is a critical Besov space.

### 13.2 The Gap

**Open Problem 13.3.** Does there exist $u_0 \in X$ with $\|u_0\|_{\dot{H}^{1/2}} < \infty$ such that $T_*(u_0) < \infty$?

**Remark 13.4.** The hypostructure framework identifies this as a question about:
1. Whether Axiom R holds (recovery from high-enstrophy regions)
2. Whether the capacity barrier (CKN) can be strengthened to $\dim \Sigma = -\infty$ (no singularities)

---

## 14. Conclusion

**Theorem 14.1 (Summary).** The Navier-Stokes equations form a hypostructure $\mathbb{H}_{NS}$ with:

| Component | Instantiation |
|:----------|:--------------|
| State space $X$ | $L^2_\sigma \cap \dot{H}^{1/2}$ |
| Height $\Phi$ | Kinetic energy $E(u)$ |
| Dissipation $\mathfrak{D}$ | Enstrophy $\nu\|\nabla u\|^2$ |
| Symmetry $G$ | Translations, rotations, scaling |
| Axiom D | Verified (energy equality) |
| Axiom SC | Critical ($\alpha = \beta = 1$) |
| Axiom LS | Verified at $u = 0$ |
| Axiom Cap | Partial (CKN: $\dim \Sigma \leq 1$) |

**Corollary 14.2.** By the metatheorems:
1. Any blow-up is confined to dimension $\leq 1$ (Theorem 7.3)
2. Self-similar blow-up in $L^3$ is excluded (Theorem 11.3)
3. Small data gives global regularity (Theorem 7.2 at criticality)

**Open.** Full verification of Axioms C and R, which would imply global regularity.

---

## 15. References

[BKM84] J.T. Beale, T. Kato, A. Majda. Remarks on the breakdown of smooth solutions for the 3-D Euler equations. Comm. Math. Phys. 94 (1984), 61–66.

[CF93] P. Constantin, C. Fefferman. Direction of vorticity and the problem of global regularity for the Navier-Stokes equations. Indiana Univ. Math. J. 42 (1993), 775–789.

[CKN82] L. Caffarelli, R. Kohn, L. Nirenberg. Partial regularity of suitable weak solutions of the Navier-Stokes equations. Comm. Pure Appl. Math. 35 (1982), 771–831.

[ESS03] L. Escauriaza, G. Seregin, V. Šverák. $L_{3,\infty}$-solutions of Navier-Stokes equations and backward uniqueness. Russian Math. Surveys 58 (2003), 211–250.

[G98] P. Gérard. Description du défaut de compacité de l'injection de Sobolev. ESAIM Control Optim. Calc. Var. 3 (1998), 213–233.

[GKP16] I. Gallagher, G. Koch, F. Planchon. Blow-up of critical Besov norms at a potential Navier-Stokes singularity. Comm. Math. Phys. 343 (2016), 39–82.

[K84] T. Kato. Strong $L^p$-solutions of the Navier-Stokes equation in $\mathbb{R}^m$, with applications to weak solutions. Math. Z. 187 (1984), 471–480.

[NRS96] J. Nečas, M. Růžička, V. Šverák. On Leray's self-similar solutions of the Navier-Stokes equations. Acta Math. 176 (1996), 283–294.

[T98] T.-P. Tsai. On Leray's self-similar solutions of the Navier-Stokes equations satisfying local energy estimates. Arch. Rational Mech. Anal. 143 (1998), 29–51.
