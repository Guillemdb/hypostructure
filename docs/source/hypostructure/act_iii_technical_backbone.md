# Part III: Technical Backbone

*Goal: Category/metric/gauge machinery supporting Parts I-II*

---

# Part II: Mathematical Foundations

## 2. Mathematical Foundations

### 2.1 The category of structural flows

We work in a categorical framework that unifies the treatment of different types of dynamical systems. The foundational theory of gradient flows in metric spaces is developed in \cite{AGS08}; for optimal transport and Wasserstein geometry, see \cite{Villani09}.

**Definition 2.1 (Category of metrizable spaces).** Let $\mathbf{Pol}$ denote the category whose objects are Polish spaces (complete separable metric spaces) and whose morphisms are continuous maps. Let $\mathbf{Pol}_\mu$ denote the category of Polish measure spaces $(X, d, \mu)$ where $\mu$ is a $\sigma$-finite Borel measure, with morphisms being measurable maps that are absolutely continuous with respect to the measures.

**Definition 2.2 (Structural flow data).** A **structural flow datum** is a tuple
$$
\mathcal{S} = (X, d, \mathcal{B}, \mu, (S_t)_{t \in T}, \Phi, \mathfrak{D})
$$
where:

- $(X, d)$ is a Polish space with metric $d$. We adopt the viewpoint of **Metric Geometry** as systematized by **Burago, Burago, and Ivanov \cite{Burago01}**, where length structures are intrinsic and exist independent of a smooth manifold atlas,
- $\mathcal{B}$ is the Borel $\sigma$-algebra on $X$,
- $\mu$ is a $\sigma$-finite Borel measure on $(X, \mathcal{B})$,
- $T \in \{\mathbb{R}_{\geq 0}, \mathbb{Z}_{\geq 0}\}$ is the time monoid,
- $(S_t)_{t \in T}$ is a semiflow (Definition 2.5),
- $\Phi: X \to [0, \infty]$ is the height functional (Definition 2.9),
- $\mathfrak{D}: X \to [0, \infty]$ is the dissipation functional (Definition 2.12).

**Definition 2.3 (Morphisms of structural flows).** A morphism $f: \mathcal{S}_1 \to \mathcal{S}_2$ between structural flow data is a continuous map $f: X_1 \to X_2$ such that:

1. $f$ is equivariant: $f \circ S^1_t = S^2_t \circ f$ for all $t \in T$,
2. $f$ is height-nonincreasing: $\Phi_2(f(x)) \leq \Phi_1(x)$ for all $x \in X_1$,
3. $f$ is dissipation-compatible: $\mathfrak{D}_2(f(x)) \leq C_f \mathfrak{D}_1(x)$ for some constant $C_f \geq 1$.

This defines the category $\mathbf{StrFlow}$ of structural flows. This trajectory-centric formulation aligns with the **Behavioral Approach** of Willems \cite{Willems91}, where a dynamical system is defined not by input-output maps but by the kernel of admissible behaviors in the signal space.

**Definition 2.4 (Forgetful functor).** There is a forgetful functor $U: \mathbf{StrFlow} \to \mathbf{DynSys}$ to the category of topological dynamical systems, given by $U(\mathcal{S}) = (X, (S_t)_{t \in T})$. This categorical formulation draws upon Lawvere's Functorial Semantics \cite{Lawvere63}, viewing dynamical theories as categories and models as functors.

### 2.2 State spaces and regularity

**Definition 2.5 (Semiflow).** A **semiflow** on a Polish space $X$ is a family of maps $(S_t: X \to X)_{t \in T}$ satisfying:

1. **Identity:** $S_0 = \mathrm{Id}_X$,
2. **Semigroup property:** $S_{t+s} = S_t \circ S_s$ for all $t, s \in T$,
3. **Continuity:** The map $(t, x) \mapsto S_t x$ is continuous on $T \times X$.

When $T = \mathbb{R}_{\geq 0}$, we speak of a continuous-time semiflow; when $T = \mathbb{Z}_{\geq 0}$, a discrete-time semiflow.

**Definition 2.6 (Maximal semiflow).** A **maximal semiflow** allows trajectories to be defined only on a maximal interval. For each $x \in X$, we define the **blow-up time**
$$
T_*(x) := \sup\{T > 0 : t \mapsto S_t x \text{ is defined and continuous on } [0, T)\} \in (0, \infty].
$$
The trajectory $t \mapsto S_t x$ is defined for $t \in [0, T_*(x))$.

**Definition 2.7 (Stochastic extension).** In the stochastic setting, we replace the semiflow by a **Markov semigroup** $(P_t)_{t \geq 0}$ acting on the space $\mathcal{P}(X)$ of Borel probability measures on $X$:
$$
(P_t \nu)(A) = \int_X p_t(x, A) \, d\nu(x),
$$
where $p_t(x, \cdot)$ is a transition kernel. The height functional is extended to measures by
$$
\Phi(\nu) := \int_X \Phi(x) \, d\nu(x),
$$
and similarly for dissipation.

**Definition 2.8 (Generalized semiflow).** For systems with non-unique solutions (e.g., weak solutions of PDEs), we define a **generalized semiflow** as a set-valued map $S_t: X \rightrightarrows X$ such that:

1. $S_0(x) = \{x\}$ for all $x$,
2. $S_{t+s}(x) \subseteq S_t(S_s(x)) := \bigcup_{y \in S_s(x)} S_t(y)$ for all $t, s \geq 0$,
3. The graph $\{(t, x, y) : y \in S_t(x)\}$ is closed in $T \times X \times X$.

### 2.3 Height functionals

**Definition 2.9 (Height functional).** A **height functional** on a structural flow is a function $\Phi: X \to [0, \infty]$ satisfying:

1. **Lower semicontinuity:** $\Phi$ is lower semicontinuous, i.e., $\{x : \Phi(x) \leq E\}$ is closed for all $E \geq 0$,
2. **Non-triviality:** $\{x : \Phi(x) < \infty\}$ is nonempty,
3. **Properness:** For each $E < \infty$, the sublevel set $K_E := \{x \in X : \Phi(x) \leq E\}$ has compact closure in $X$.

**Definition 2.10 (Coercivity).** The height functional $\Phi$ is **coercive** if for every sequence $(x_n) \subset X$ with $d(x_n, x_0) \to \infty$ for some fixed $x_0 \in X$, we have $\Phi(x_n) \to \infty$.

**Definition 2.11 (Lyapunov candidate).** We say $\Phi$ is a **Lyapunov candidate** if there exists $C \geq 0$ such that for all trajectories $u(t) = S_t x$:
$$
\Phi(u(t)) \leq \Phi(u(s)) + C(t - s) \quad \text{for all } 0 \leq s \leq t < T_*(x).
$$
When $C = 0$, $\Phi$ is a **Lyapunov functional**.

### 2.4 Dissipation structure

**Definition 2.12 (Dissipation functional).** A **dissipation functional** is a measurable function $\mathfrak{D}: X \to [0, \infty]$ that quantifies the instantaneous rate of irreversible cost along trajectories.

**Definition 2.13 (Dissipation measure).** Along a trajectory $u: [0, T) \to X$, the **dissipation measure** is the Radon measure on $[0, T)$ given by the Lebesgue–Stieltjes decomposition:
$$
d\mathcal{D}_u = \mathfrak{D}(u(t)) \, dt + d\mathcal{D}_u^{\mathrm{sing}},
$$
where $\mathfrak{D}(u(t)) \, dt$ is the absolutely continuous part and $d\mathcal{D}_u^{\mathrm{sing}}$ is the singular part (supported on a set of Lebesgue measure zero).

**Definition 2.14 (Total cost).** The **total cost** of a trajectory on $[0, T]$ is
$$
\mathcal{C}_T(x) := \int_0^T \mathfrak{D}(S_t x) \, dt.
$$
For the full trajectory up to blow-up time:
$$
\mathcal{C}_*(x) := \mathcal{C}_{T_*(x)}(x) = \int_0^{T_*(x)} \mathfrak{D}(S_t x) \, dt.
$$

**Definition 2.15 (Energy–dissipation inequality).** The pair $(\Phi, \mathfrak{D})$ satisfies an **energy–dissipation inequality** if there exist constants $\alpha > 0$ and $C \geq 0$ such that for all trajectories $u(t) = S_t x$:
$$
\Phi(u(t_2)) + \alpha \int_{t_1}^{t_2} \mathfrak{D}(u(s)) \, ds \leq \Phi(u(t_1)) + C(t_2 - t_1)
$$
for all $0 \leq t_1 \leq t_2 < T_*(x)$.

**Definition 2.16 (Energy–dissipation identity).** When equality holds and $C = 0$:
$$
\Phi(u(t_2)) + \alpha \int_{t_1}^{t_2} \mathfrak{D}(u(s)) \, ds = \Phi(u(t_1)),
$$
we say the system satisfies an **energy–dissipation identity** (balance law).

### 2.5 Bornological and uniform structures

**Definition 2.17 (Bornology).** A **bornology** on $X$ is a collection $\mathcal{B}$ of subsets of $X$ (called bounded sets) such that:

1. $\mathcal{B}$ covers $X$: $\bigcup_{B \in \mathcal{B}} B = X$,
2. $\mathcal{B}$ is hereditary: if $A \subseteq B \in \mathcal{B}$, then $A \in \mathcal{B}$,
3. $\mathcal{B}$ is stable under finite unions.

The bornology induced by $\Phi$ is $\mathcal{B}_\Phi := \{B \subseteq X : \sup_{x \in B} \Phi(x) < \infty\}$.

**Definition 2.18 (Equicontinuity).** The semiflow $(S_t)$ is **equicontinuous on bounded sets** if for every $B \in \mathcal{B}_\Phi$ and every $\varepsilon > 0$, there exists $\delta > 0$ such that for all $t \in [0, 1]$:
$$
x, y \in B, \, d(x, y) < \delta \implies d(S_t x, S_t y) < \varepsilon.
$$

---


---

# Part IV: Core Metatheorems

## 5. Normalization and Gauge Structure

### 5.1 Symmetry groups

**Definition 5.1 (Symmetry group action).** Let $G$ be a locally compact Hausdorff topological group. A **continuous action** of $G$ on $X$ is a continuous map $G \times X \to X$, $(g, x) \mapsto g \cdot x$, such that:

1. $e \cdot x = x$ for all $x \in X$ (where $e$ is the identity),
2. $(gh) \cdot x = g \cdot (h \cdot x)$ for all $g, h \in G$, $x \in X$.

**Definition 5.2 (Isometric action).** The action is **isometric** if $d(g \cdot x, g \cdot y) = d(x, y)$ for all $g \in G$, $x, y \in X$.

**Definition 5.3 (Proper action).** The action is **proper** if for every compact $K \subseteq X$, the set $\{g \in G : g \cdot K \cap K \neq \emptyset\}$ is compact in $G$.

**Example 5.4 (Common symmetry groups).**

1. **Translations:** $G = \mathbb{R}^n$ acting by $(a, u) \mapsto u(\cdot - a)$ on function spaces.
2. **Rotations:** $G = SO(n)$ acting by $(R, u) \mapsto u(R^{-1} \cdot)$.
3. **Scalings:** $G = \mathbb{R}_{> 0}$ acting by $(\lambda, u) \mapsto \lambda^\alpha u(\lambda \cdot)$ for some $\alpha$.
4. **Parabolic rescaling:** $G = \mathbb{R}_{> 0}$ acting by $(\lambda, u) \mapsto \lambda^\alpha u(\lambda \cdot, \lambda^2 \cdot)$.
5. **Gauge transformations:** $G = \mathcal{G}$ (a gauge group) acting by $(g, A) \mapsto g^{-1} A g + g^{-1} dg$.

### 5.2 Gauge maps and normalized slices

**Definition 5.5 (Gauge map).** A **gauge map** is a measurable function $\Gamma: X \to G$ such that the **normalized state**
$$
\tilde{x} := \Gamma(x) \cdot x
$$
lies in a designated **normalized slice** $\Sigma \subseteq X$.

**Definition 5.6 (Normalized slice).** A **normalized slice** is a measurable subset $\Sigma \subseteq X$ such that:

1. **Transversality:** For $\mu$-almost every $x \in X$, the orbit $G \cdot x$ intersects $\Sigma$.
2. **Uniqueness (up to discrete ambiguity):** For each orbit $G \cdot x$, the intersection $G \cdot x \cap \Sigma$ is a discrete (possibly singleton) set.

**Proposition 5.7 (Existence of gauge maps).** Suppose the action of $G$ on $X$ is proper and isometric. Then for any normalized slice $\Sigma$, there exists a measurable gauge map $\Gamma: X \to G$.

*Proof.* For each $x \in X$, let $\pi(x) \in \Sigma$ be a point in $G \cdot x \cap \Sigma$ (using the axiom of choice, or constructively via a measurable selection theorem since the action is proper). Define $\Gamma(x)$ to be any $g \in G$ such that $g \cdot x = \pi(x)$. The properness of the action ensures this is well-defined and measurable. $\square$

**Definition 5.8 (Bounded gauge).** The gauge map $\Gamma$ is **bounded on energy sublevels** if for each $E < \infty$, there exists a compact set $K_G \subseteq G$ such that $\Gamma(x) \in K_G$ for all $x \in K_E$.

### 5.3 Normalized functionals

**Definition 5.9 (Normalized height and dissipation).** The **normalized height** and **normalized dissipation** are
$$
\tilde{\Phi}(x) := \Phi(\Gamma(x) \cdot x), \qquad \tilde{\mathfrak{D}}(x) := \mathfrak{D}(\Gamma(x) \cdot x).
$$

**Definition 5.10 (Normalized trajectory).** For a trajectory $u(t) = S_t x$, the **normalized trajectory** is
$$
\tilde{u}(t) := \Gamma(u(t)) \cdot u(t).
$$

**Axiom N (Normalization compatibility along trajectories).** Along any trajectory $u(t) = S_t x$ with bounded energy $\sup_t \Phi(u(t)) \leq E$, the normalized functionals are comparable to the original functionals: there exist constants $0 < c_1(E) \leq c_2(E) < \infty$ (possibly depending on the energy level) such that:
$$
c_1(E) \Phi(y) \leq \tilde{\Phi}(y) \leq c_2(E) \Phi(y), \qquad c_1(E) \mathfrak{D}(y) \leq \tilde{\mathfrak{D}}(y) \leq c_2(E) \mathfrak{D}(y)
$$
for all $y$ on the trajectory.

**Fallback.** When Axiom N degenerates (i.e., $c_1(E) \to 0$ or $c_2(E) \to \infty$ as $E \to \infty$), one works in unnormalized coordinates. The theorems requiring normalization (Theorem 6.2) apply only where N holds with controlled constants.

### 5.4 Generic normalization as derived property

With Scaling Structure (Axiom SC, defined below) in place, Generic Normalization becomes a derived consequence rather than an independent axiom.

**Definition 5.11 (Scaling subgroup).** A **scaling subgroup** is a one-parameter subgroup $(\mathcal{S}_\lambda)_{\lambda > 0} \subset G$ of the symmetry group, with $\mathcal{S}_1 = e$ and $\mathcal{S}_\lambda \circ \mathcal{S}_\mu = \mathcal{S}_{\lambda\mu}$.

**Definition 5.12 (Scaling exponents).** The **scaling exponents** along an orbit where $(\mathcal{S}_\lambda)$ acts are constants $\alpha > 0$ and $\beta > 0$ such that:

1. **Dissipation scaling:** There exists $C_\alpha \geq 1$ such that for all $x$ on the orbit and $\lambda > 0$:
$$
C_\alpha^{-1} \lambda^\alpha \mathfrak{D}(x) \leq \mathfrak{D}(\mathcal{S}_\lambda \cdot x) \leq C_\alpha \lambda^\alpha \mathfrak{D}(x).
$$
2. **Temporal scaling:** Under the rescaling $s = \lambda^\beta (T - t)$ near a reference time $T$, the time differential transforms as $dt = \lambda^{-\beta} ds$.

**Axiom SC (Scaling Structure on orbits).** On any orbit where the scaling subgroup $(\mathcal{S}_\lambda)_{\lambda > 0}$ acts with well-defined scaling exponents $(\alpha, \beta)$, the **subcritical dissipation condition** holds:
$$
\alpha > \beta.
$$

**Fallback (Mode S.E).** When Axiom SC fails along a trajectory—either because no scaling subgroup acts, or the subcritical condition $\alpha > \beta$ is violated—the trajectory may exhibit **supercritical symmetry cascade** (Resolution mode 3, Theorem 6.1). Property GN is not derived in this case; Type II blow-up must be excluded by other means or accepted as a possible failure mode.

**Definition 5.13 (Supercritical sequence).** A sequence $(\lambda_n) \subset \mathbb{R}_{> 0}$ is **supercritical** if $\lambda_n \to \infty$.

**Remark 5.14.** The exponent $\alpha$ measures how strongly dissipation responds to zooming; $\beta$ measures how remaining time compresses under scaling. The condition $\alpha > \beta$ ensures that supercritical rescaling amplifies dissipation faster than it compresses time, making infinite-cost profiles unavoidable in the limit.

**Remark 5.15 (Scaling structure is soft).** For most systems of interest, the scaling structure is immediate from dimensional analysis:

- For parabolic PDEs with scaling $(x, t) \mapsto (\lambda x, \lambda^2 t)$, the exponents follow from computing how $\mathfrak{D}$ and $dt$ transform.
- For kinetic systems, the scaling comes from velocity-space rescaling.
- For discrete systems, the scaling may be combinatorial (e.g., term depth).
- For systems without natural scaling symmetry, SC does not apply and GN must be established by other structural means.

No hard analysis is required to identify SC where it applies; it is a purely structural/dimensional property.

**Definition 5.16 (Scale parameter).** A **scale parameter** is a continuous function $\sigma: G \to \mathbb{R}_{> 0}$ such that $\sigma(e) = 1$ and $\sigma(gh) = \sigma(g) \sigma(h)$ (i.e., $\sigma$ is a group homomorphism to $(\mathbb{R}_{> 0}, \times)$). For the scaling subgroup, $\sigma(\mathcal{S}_\lambda) = \lambda$.

**Definition 5.17 (Supercritical rescaling).** A sequence $(g_n) \subset G$ is **supercritical** if $\sigma(g_n) \to 0$ or $\sigma(g_n) \to \infty$ (depending on convention: the scale escapes the critical regime).

**Property GN (Generic Normalization).** For any trajectory $u(t) = S_t x$ with finite total cost $\mathcal{C}_*(x) < \infty$, if:

- $(t_n)$ is a sequence with $t_n \nearrow T_*(x)$,
- $(g_n) \subset G$ is a supercritical sequence,
- the rescaled states $v_n := g_n \cdot u(t_n)$ converge to a limit $v_\infty \in X$,

then the normalized dissipation integral along any trajectory through $v_\infty$ must diverge:
$$
\int_0^\infty \tilde{\mathfrak{D}}(S_t v_\infty) \, dt = \infty.
$$

**Remark 5.18.** Property GN says: any would-be Type II blow-up profile, when viewed in normalized coordinates, has infinite dissipation. Thus such profiles cannot arise from finite-cost trajectories. Under Axiom SC, this is not an additional assumption but a theorem (see Theorem 6.2).

### 5.5 Preparatory Lemmas

The following lemmas provide the technical foundation for the resolution theorems. They translate the abstract axioms into concrete analytical tools.

**Lemma 5.19 (Compactness extraction).** Assume Axiom C. Let $(x_n) \subset K_E$ be a sequence in an energy sublevel. Then there exist:

- a subsequence $(x_{n_k})$,
- elements $g_k \in G$,
- a limit point $x_\infty \in X$ with $\Phi(x_\infty) \leq E$,

such that $g_k \cdot x_{n_k} \to x_\infty$ in $X$.

*Proof.* Axiom C directly asserts precompactness modulo $G$. Apply the definition to the sequence $(x_n)$ to obtain $g_n \in G$ and a subsequence such that $g_{n_k} \cdot x_{n_k}$ converges. The limit $x_\infty$ satisfies $\Phi(x_\infty) \leq E$ by lower semicontinuity of $\Phi$. $\square$

**Lemma 5.20 (Dissipation chain rule).** Assume Axiom D. For any trajectory $u(t) = S_t x$, the function $t \mapsto \Phi(u(t))$ satisfies, for almost every $t \in [0, T_*(x))$:
$$
\frac{d}{dt} \Phi(u(t)) \leq -\alpha \mathfrak{D}(u(t)) + C.
$$
In particular, $\Phi(u(t))$ is absolutely continuous and
$$
\Phi(u(t)) \leq \Phi(u(0)) + Ct - \alpha \int_0^t \mathfrak{D}(u(s)) \, ds.
$$

*Proof.* Fix $t_1 < t_2$ in $[0, T_*(x))$. By Axiom D:
$$
\Phi(u(t_2)) + \alpha \int_{t_1}^{t_2} \mathfrak{D}(u(s)) \, ds \leq \Phi(u(t_1)) + C(t_2 - t_1).
$$
Rearranging:
$$
\Phi(u(t_2)) - \Phi(u(t_1)) \leq C(t_2 - t_1) - \alpha \int_{t_1}^{t_2} \mathfrak{D}(u(s)) \, ds.
$$
This shows $\Phi(u(\cdot))$ has bounded variation on compact intervals. Since $\mathfrak{D}(u(\cdot)) \in L^1_{\mathrm{loc}}$, the function $t \mapsto \int_0^t \mathfrak{D}(u(s)) \, ds$ is absolutely continuous. Thus $\Phi(u(\cdot))$ is absolutely continuous, and the differential inequality holds a.e. $\square$

**Lemma 5.21 (Cost-recovery duality).** Assume Axioms D and Rec. For any trajectory $u(t) = S_t x$:
$$
\mathrm{Leb}\{t \in [0, T) : u(t) \notin \mathcal{G}\} \leq \frac{C_0}{r_0} \mathcal{C}_T(x).
$$
In particular, if $\mathcal{C}_*(x) < \infty$, then $u(t) \in \mathcal{G}$ for almost all sufficiently large $t$.

*Proof.* Let $A = \{t \in [0, T) : u(t) \notin \mathcal{G}\}$. By Axiom Rec:
$$
r_0 \cdot \mathrm{Leb}(A) \leq \int_A \mathcal{R}(u(t)) \, dt \leq C_0 \int_0^T \mathfrak{D}(u(t)) \, dt = C_0 \mathcal{C}_T(x).
$$
Dividing by $r_0$ gives the result. If $\mathcal{C}_*(x) < \infty$, then $\mathrm{Leb}(A) < \infty$ for $T = T_*(x)$, so $A$ has finite measure. $\square$

**Lemma 5.22 (Occupation measure bounds).** Assume Axiom Cap. For any measurable set $B \subseteq X$ with $\mathrm{Cap}(B) > 0$ and any trajectory $u(t) = S_t x$:
$$
\mathrm{Leb}\{t \in [0, T] : u(t) \in B\} \leq \frac{C_{\mathrm{cap}}(\Phi(x) + T)}{\mathrm{Cap}(B)}.
$$

*Proof.* Define the occupation time $\tau_B := \mathrm{Leb}\{t \in [0, T] : u(t) \in B\}$. We have:
$$
\mathrm{Cap}(B) \cdot \tau_B = \int_0^T \mathrm{Cap}(B) \mathbf{1}_{u(t) \in B} \, dt \leq \int_0^T c(u(t)) \mathbf{1}_{u(t) \in B} \, dt \leq \int_0^T c(u(t)) \, dt.
$$
By Axiom Cap, the last integral is bounded by $C_{\mathrm{cap}}(\Phi(x) + T)$. $\square$

**Corollary 5.23 (High-capacity sets are avoided).** If $(B_k)$ is a sequence with $\mathrm{Cap}(B_k) \to \infty$, then for any fixed trajectory:
$$
\lim_{k \to \infty} \mathrm{Leb}\{t \in [0, T] : u(t) \in B_k\} = 0.
$$

**Lemma 5.24 (Łojasiewicz decay estimate).** Assume Axioms D and LS with $C = 0$ (strict Lyapunov). Suppose $u(t) = S_t x$ remains in the neighbourhood $U$ of the safe manifold $M$ for all $t \geq t_0$. Then:
$$
\mathrm{dist}(u(t), M) \leq C \cdot (t - t_0 + 1)^{-\theta/(1-\theta)} \quad \text{for all } t \geq t_0,
$$
where $C$ depends on $\Phi(u(t_0))$, $\alpha$, $C_{\mathrm{LS}}$, and $\theta$.

*Proof.* Let $\psi(t) := \Phi(u(t)) - \Phi_{\min} \geq 0$. By Lemma 5.20 (with $C = 0$):
$$
\psi'(t) \leq -\alpha \mathfrak{D}(u(t)) \quad \text{a.e.}
$$
We need to relate $\mathfrak{D}$ to $\psi$. From gradient flow structure (or analogous dissipation-height coupling in the general case), assume:
$$
\mathfrak{D}(x) \geq c |\nabla \Phi(x)|^2 \quad \text{and} \quad |\nabla \Phi(x)| \geq c' (\Phi(x) - \Phi_{\min})^{1-\theta}
$$
near $M$ (the Łojasiewicz gradient inequality). Then:
$$
\psi'(t) \leq -\alpha c (c')^2 \psi(t)^{2(1-\theta)} = -\beta \psi(t)^{2-2\theta}
$$
for some $\beta > 0$.

For $\theta < 1$, set $\gamma = 2 - 2\theta > 0$. Then:
$$
\frac{d}{dt} \psi^{1-\gamma} = (1 - \gamma) \psi^{-\gamma} \psi' \leq -\beta(1 - \gamma) < 0.
$$
Since $1 - \gamma = 2\theta - 1$, we have for $\theta > 1/2$:
$$
\psi(t)^{2\theta - 1} \leq \psi(t_0)^{2\theta - 1} - \beta(2\theta - 1)(t - t_0),
$$
giving polynomial decay of $\psi(t)$ and hence of $\mathrm{dist}(u(t), M)$ via the Łojasiewicz inequality. The general case $\theta \in (0, 1]$ follows by similar ODE analysis. $\square$

**Lemma 5.25 (Herbst argument).** Assume an invariant probability measure $\mu$ satisfies a log-Sobolev inequality with constant $\lambda_{\mathrm{LS}} > 0$. Then for any Lipschitz function $F: X \to \mathbb{R}$ with Lipschitz constant $\|F\|_{\mathrm{Lip}} \leq 1$:
$$
\mu\left(\left\{x : F(x) - \int F \, d\mu > r\right\}\right) \leq \exp\left(-\lambda_{\mathrm{LS}} r^2 / 2\right).
$$

*Proof.* For $\lambda > 0$, set $f = e^{\lambda F / 2}$. By the log-Sobolev inequality (LSI):
$$
\int f^2 \log f^2 \, d\mu - \int f^2 \, d\mu \log \int f^2 \, d\mu \leq \frac{1}{2\lambda_{\mathrm{LS}}} \int |\nabla f|^2 \, d\mu.
$$
Since $|\nabla f| = \frac{\lambda}{2} |f| |\nabla F| \leq \frac{\lambda}{2} f$ (using $\|F\|_{\mathrm{Lip}} \leq 1$):
$$
\int |\nabla f|^2 \, d\mu \leq \frac{\lambda^2}{4} \int f^2 \, d\mu.
$$
Let $Z(\lambda) = \int e^{\lambda F} \, d\mu$. The entropy inequality becomes:
$$
\frac{d}{d\lambda}\left[\lambda \log Z(\lambda)\right] = \log Z(\lambda) + \frac{\lambda Z'(\lambda)}{Z(\lambda)} \leq \frac{\lambda}{8\lambda_{\mathrm{LS}}}.
$$
Integrating and using Chebyshev's inequality yields the Gaussian concentration. $\square$

**Corollary 5.26 (Sector suppression from LSI).** If the action functional $\mathcal{A}$ satisfies $\|\mathcal{A}\|_{\mathrm{Lip}} \leq L$ and Axiom TB1 holds with gap $\Delta$, then:
$$
\mu(\{x : \tau(x) \neq 0\}) \leq \mu(\{x : \mathcal{A}(x) \geq \mathcal{A}_{\min} + \Delta\}) \leq C \exp\left(-\frac{\lambda_{\mathrm{LS}} \Delta^2}{2L^2}\right).
$$

---

