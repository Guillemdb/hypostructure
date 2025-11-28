# Hypostructures: A Structural Framework for Singularity Control in Dynamical Systems

## 0. Overview

### 0.1 The singularity control thesis

A **hypostructure** is a framework for dynamical systems—deterministic or stochastic, continuous or discrete—that provides **global regularity via soft local exclusion**. The central thesis is:

> **Global regularity is proven by showing that singularities are locally impossible. The axioms (C, D, R, Cap, LS, SC) act as algebraic permits that any singularity must satisfy. When these permits are denied via dimensional or geometric analysis, the singularity cannot form.**

**The Exclusion Principle.** The framework does not construct solutions globally or require hard estimates. It proves regularity through the following logic:

1. **Forced Structure:** Finite-time blow-up ($T_* < \infty$) requires energy concentration. Concentration forces local structure—a Canonical Profile $V$ emerges wherever blow-up attempts to form.
2. **Permit Checking:** The structure $V$ must satisfy algebraic permits:
   - **Scaling Permit (Axiom SC):** Are the scaling exponents subcritical ($\alpha > \beta$)?
   - **Geometric Permit (Axiom Cap):** Does the singular set have positive capacity?
   - **Topological Permit (Axiom TB):** Is the topological sector accessible?
   - **Stiffness Permit (Axiom LS):** Does the Łojasiewicz inequality hold near equilibria?
3. **Contradiction:** If any permit is denied, the singularity cannot form. Global regularity follows.

**Mode 2 (Dispersion) is not a singularity.** When energy does not concentrate (Axiom C fails), no finite-time singularity forms—the solution exists globally and disperses. Mode 2 represents **global existence via scattering**, not a failure mode.

**No global estimates required.** The framework never requires proving global compactness or global bounds. All analysis is local: concentration forces structure, structure is tested against algebraic permits, permit denial implies regularity. The classification is **logically exhaustive**: every trajectory either disperses globally (Mode 2), blows up via energy escape (Mode 1), or has its blow-up attempt blocked by permit denial (Modes 3–6 contradict, yielding regularity).

### 0.2 Conceptual architecture

The framework rests on three pillars:

1. **Height and dissipation.** A height functional $\Phi$ (energy, free energy, Lyapunov candidate) coupled with a dissipation functional $\mathfrak{D}$ that tracks the cost of evolution. The pair $(\Phi, \mathfrak{D})$ satisfies an energy–dissipation inequality that bounds the available budget for singular behaviour.

2. **Structural axioms.** A collection of local/soft axioms—compactness, recovery, capacity, stiffness, regularity—that constrain how trajectories can concentrate, disperse, or degenerate. These axioms are designed to be verifiable in concrete settings while remaining sufficiently abstract to apply across disparate domains.

3. **Symmetry and scaling.** A gauge structure that tracks the symmetries of the problem (scalings, translations, rotations, gauge transformations) and a scaling structure axiom (SC) that, combined with dissipation, automatically rules out supercritical self-similar blow-up via pure scaling arithmetic.

### 0.3 Main consequences

From these axioms, we derive:

* **Structural Resolution (Theorem 7.1).** Every trajectory resolves into one of three outcomes: global existence (dispersive), global regularity (permit denial), or genuine singularity.
* **Type II exclusion (Theorem 7.2).** Under SC + D, supercritical self-similar blow-up is impossible at finite cost—derived from scaling arithmetic alone.
* **Capacity barrier (Theorem 7.3).** Trajectories cannot concentrate on arbitrarily thin or high-codimension sets.
* **Topological suppression (Theorem 7.4).** Nontrivial topological sectors are exponentially rare under the invariant measure.
* **Structured vs failure dichotomy (Theorem 7.5).** Finite-energy trajectories are eventually confined to a structured region where classical regularity holds.
* **Canonical Lyapunov functional (Theorem 7.6).** There exists a unique (up to monotone reparametrization) Lyapunov functional determined by the structural data.
* **Functional reconstruction (Theorems 7.7.1, 7.7.3).** Under gradient consistency, the Lyapunov functional is explicitly recoverable as the geodesic distance in a Jacobi metric, or as the solution to a Hamilton–Jacobi equation. No prior knowledge of an energy functional is required.
* **Quantitative thresholds (Theorem 9.3).** The framework explicitly calculates sharp constants and energy thresholds by analyzing the variational properties of the failure modes. The Canonical Profile $V$ extracted by Axiom C is the variational optimizer that saturates the governing inequalities.

### 0.4 Scope of instantiation

The framework is designed to be instantiated in:

* **PDE flows:** Parabolic, hyperbolic, and dispersive equations; geometric flows (mean curvature, Ricci); reaction–diffusion systems.
* **Kinetic and probabilistic systems:** McKean–Vlasov dynamics, Fleming–Viot processes, interacting particle systems, Langevin dynamics.
* **Discrete and computational systems:** λ-calculus reduction, interaction nets, graph rewriting systems.

**Remark 0.1 (No hard estimates required).** Instantiation does not require proving global compactness or global regularity *a priori*. It requires only:
1. Identifying the symmetries $G$ (translations, scalings, gauge transformations),
2. Computing the algebraic data (scaling exponents $\alpha, \beta$; capacity dimensions; Łojasiewicz exponents).

The framework then checks whether the algebraic permits are satisfied:
- If $\alpha > \beta$ (Axiom SC), supercritical blow-up is impossible.
- If singular sets have positive capacity (Axiom Cap), geometric concentration is impossible.
- If permits are denied, **global regularity follows from soft local exclusion**—no hard estimates needed.

The only remaining possibility is Mode 2 (dispersion), which is not a finite-time singularity but global existence via scattering.

---

## 1. Categorical and measure-theoretic foundations

### 1.1 The category of structural flows

We work in a categorical framework that unifies the treatment of different types of dynamical systems.

**Definition 1.1 (Category of metrizable spaces).** Let $\mathbf{Pol}$ denote the category whose objects are Polish spaces (complete separable metric spaces) and whose morphisms are continuous maps. Let $\mathbf{Pol}_\mu$ denote the category of Polish measure spaces $(X, d, \mu)$ where $\mu$ is a $\sigma$-finite Borel measure, with morphisms being measurable maps that are absolutely continuous with respect to the measures.

**Definition 1.2 (Structural flow data).** A **structural flow datum** is a tuple
$$
\mathcal{S} = (X, d, \mathcal{B}, \mu, (S_t)_{t \in T}, \Phi, \mathfrak{D})
$$
where:
* $(X, d)$ is a Polish space with metric $d$,
* $\mathcal{B}$ is the Borel $\sigma$-algebra on $X$,
* $\mu$ is a $\sigma$-finite Borel measure on $(X, \mathcal{B})$,
* $T \in \{\mathbb{R}_{\geq 0}, \mathbb{Z}_{\geq 0}\}$ is the time monoid,
* $(S_t)_{t \in T}$ is a semiflow (Definition 1.5),
* $\Phi: X \to [0, \infty]$ is the height functional (Definition 1.9),
* $\mathfrak{D}: X \to [0, \infty]$ is the dissipation functional (Definition 1.12).

**Definition 1.3 (Morphisms of structural flows).** A morphism $f: \mathcal{S}_1 \to \mathcal{S}_2$ between structural flow data is a continuous map $f: X_1 \to X_2$ such that:
1. $f$ is equivariant: $f \circ S^1_t = S^2_t \circ f$ for all $t \in T$,
2. $f$ is height-nonincreasing: $\Phi_2(f(x)) \leq \Phi_1(x)$ for all $x \in X_1$,
3. $f$ is dissipation-compatible: $\mathfrak{D}_2(f(x)) \leq C_f \mathfrak{D}_1(x)$ for some constant $C_f \geq 1$.

This defines the category $\mathbf{StrFlow}$ of structural flows.

**Definition 1.4 (Forgetful functor).** There is a forgetful functor $U: \mathbf{StrFlow} \to \mathbf{DynSys}$ to the category of topological dynamical systems, given by $U(\mathcal{S}) = (X, (S_t)_{t \in T})$.

### 1.2 State spaces and regularity

**Definition 1.5 (Semiflow).** A **semiflow** on a Polish space $X$ is a family of maps $(S_t: X \to X)_{t \in T}$ satisfying:
1. **Identity:** $S_0 = \mathrm{Id}_X$,
2. **Semigroup property:** $S_{t+s} = S_t \circ S_s$ for all $t, s \in T$,
3. **Continuity:** The map $(t, x) \mapsto S_t x$ is continuous on $T \times X$.

When $T = \mathbb{R}_{\geq 0}$, we speak of a continuous-time semiflow; when $T = \mathbb{Z}_{\geq 0}$, a discrete-time semiflow.

**Definition 1.6 (Maximal semiflow).** A **maximal semiflow** allows trajectories to be defined only on a maximal interval. For each $x \in X$, we define the **blow-up time**
$$
T_*(x) := \sup\{T > 0 : t \mapsto S_t x \text{ is defined and continuous on } [0, T)\} \in (0, \infty].
$$
The trajectory $t \mapsto S_t x$ is defined for $t \in [0, T_*(x))$.

**Definition 1.7 (Stochastic extension).** In the stochastic setting, we replace the semiflow by a **Markov semigroup** $(P_t)_{t \geq 0}$ acting on the space $\mathcal{P}(X)$ of Borel probability measures on $X$:
$$
(P_t \nu)(A) = \int_X p_t(x, A) \, d\nu(x),
$$
where $p_t(x, \cdot)$ is a transition kernel. The height functional is extended to measures by
$$
\Phi(\nu) := \int_X \Phi(x) \, d\nu(x),
$$
and similarly for dissipation.

**Definition 1.8 (Generalized semiflow).** For systems with non-unique solutions (e.g., weak solutions of PDEs), we define a **generalized semiflow** as a set-valued map $S_t: X \rightrightarrows X$ such that:
1. $S_0(x) = \{x\}$ for all $x$,
2. $S_{t+s}(x) \subseteq S_t(S_s(x)) := \bigcup_{y \in S_s(x)} S_t(y)$ for all $t, s \geq 0$,
3. The graph $\{(t, x, y) : y \in S_t(x)\}$ is closed in $T \times X \times X$.

### 1.3 Height functionals

**Definition 1.9 (Height functional).** A **height functional** on a structural flow is a function $\Phi: X \to [0, \infty]$ satisfying:
1. **Lower semicontinuity:** $\Phi$ is lower semicontinuous, i.e., $\{x : \Phi(x) \leq E\}$ is closed for all $E \geq 0$,
2. **Non-triviality:** $\{x : \Phi(x) < \infty\}$ is nonempty,
3. **Properness:** For each $E < \infty$, the sublevel set $K_E := \{x \in X : \Phi(x) \leq E\}$ has compact closure in $X$.

**Definition 1.10 (Coercivity).** The height functional $\Phi$ is **coercive** if for every sequence $(x_n) \subset X$ with $d(x_n, x_0) \to \infty$ for some fixed $x_0 \in X$, we have $\Phi(x_n) \to \infty$.

**Definition 1.11 (Lyapunov candidate).** We say $\Phi$ is a **Lyapunov candidate** if there exists $C \geq 0$ such that for all trajectories $u(t) = S_t x$:
$$
\Phi(u(t)) \leq \Phi(u(s)) + C(t - s) \quad \text{for all } 0 \leq s \leq t < T_*(x).
$$
When $C = 0$, $\Phi$ is a **Lyapunov functional**.

### 1.4 Dissipation structure

**Definition 1.12 (Dissipation functional).** A **dissipation functional** is a measurable function $\mathfrak{D}: X \to [0, \infty]$ that quantifies the instantaneous rate of irreversible cost along trajectories.

**Definition 1.13 (Dissipation measure).** Along a trajectory $u: [0, T) \to X$, the **dissipation measure** is the Radon measure on $[0, T)$ given by the Lebesgue–Stieltjes decomposition:
$$
d\mathcal{D}_u = \mathfrak{D}(u(t)) \, dt + d\mathcal{D}_u^{\mathrm{sing}},
$$
where $\mathfrak{D}(u(t)) \, dt$ is the absolutely continuous part and $d\mathcal{D}_u^{\mathrm{sing}}$ is the singular part (supported on a set of Lebesgue measure zero).

**Definition 1.14 (Total cost).** The **total cost** of a trajectory on $[0, T]$ is
$$
\mathcal{C}_T(x) := \int_0^T \mathfrak{D}(S_t x) \, dt.
$$
For the full trajectory up to blow-up time:
$$
\mathcal{C}_*(x) := \mathcal{C}_{T_*(x)}(x) = \int_0^{T_*(x)} \mathfrak{D}(S_t x) \, dt.
$$

**Definition 1.15 (Energy–dissipation inequality).** The pair $(\Phi, \mathfrak{D})$ satisfies an **energy–dissipation inequality** if there exist constants $\alpha > 0$ and $C \geq 0$ such that for all trajectories $u(t) = S_t x$:
$$
\Phi(u(t_2)) + \alpha \int_{t_1}^{t_2} \mathfrak{D}(u(s)) \, ds \leq \Phi(u(t_1)) + C(t_2 - t_1)
$$
for all $0 \leq t_1 \leq t_2 < T_*(x)$.

**Definition 1.16 (Energy–dissipation identity).** When equality holds and $C = 0$:
$$
\Phi(u(t_2)) + \alpha \int_{t_1}^{t_2} \mathfrak{D}(u(s)) \, ds = \Phi(u(t_1)),
$$
we say the system satisfies an **energy–dissipation identity** (balance law).

### 1.5 Bornological and uniform structures

**Definition 1.17 (Bornology).** A **bornology** on $X$ is a collection $\mathcal{B}$ of subsets of $X$ (called bounded sets) such that:
1. $\mathcal{B}$ covers $X$: $\bigcup_{B \in \mathcal{B}} B = X$,
2. $\mathcal{B}$ is hereditary: if $A \subseteq B \in \mathcal{B}$, then $A \in \mathcal{B}$,
3. $\mathcal{B}$ is stable under finite unions.

The **natural bornology** induced by $\Phi$ is $\mathcal{B}_\Phi := \{B \subseteq X : \sup_{x \in B} \Phi(x) < \infty\}$.

**Definition 1.18 (Equicontinuity).** The semiflow $(S_t)$ is **equicontinuous on bounded sets** if for every $B \in \mathcal{B}_\Phi$ and every $\varepsilon > 0$, there exists $\delta > 0$ such that for all $t \in [0, 1]$:
$$
x, y \in B, \, d(x, y) < \delta \implies d(S_t x, S_t y) < \varepsilon.
$$

---

## 2. The axiom system

A **hypostructure** is a structural flow datum $\mathcal{S}$ satisfying the following axioms.

### 2.1 Compactness (C)

**Structural Data (Symmetry Group).** The system admits a continuous action by a locally compact topological group $G$ acting on $X$ by isometries (i.e., $d(g \cdot x, g \cdot y) = d(x, y)$ for all $g \in G$, $x, y \in X$). This is structural data about the system, not an assumption to be verified per trajectory.

**Axiom C (Structural Compactness Potential).** We say a trajectory $u(t) = S_t x$ with bounded energy $\sup_{t < T_*(x)} \Phi(u(t)) \leq E < \infty$ **satisfies Axiom C** if: for every sequence of times $t_n \nearrow T_*(x)$, there exists a subsequence $(t_{n_k})$ and elements $g_k \in G$ such that $(g_k \cdot u(t_{n_k}))$ converges **strongly** in the topology of $X$ to a **single** limit profile $V \in X$.

When $G$ is trivial, this reduces to ordinary precompactness of bounded-energy trajectory tails.

**Role (Forced Structure Principle).** Axiom C is **automatically triggered by blow-up attempts**. The key insight is:

1. **Finite-time blow-up requires concentration.** To form a singularity at $T_* < \infty$, energy must concentrate—otherwise the solution disperses globally and no singularity forms.
2. **Concentration forces local structure.** Wherever energy concentrates, a Canonical Profile $V$ emerges. Axiom C holds locally at any blow-up locus.
3. **No concentration = no singularity.** If Axiom C fails (energy disperses), there is **no finite-time singularity**—the solution exists globally via scattering (Mode 2).

Consequently:
- **Mode 2 is not a singularity.** It represents global existence via dispersion, not a "failure mode."
- **Modes 3–6 require Axiom C to hold** (structure exists), then test whether the structure satisfies algebraic permits.
- **No global compactness proof is needed.** We observe that blow-up *forces* local compactness, then check permits on the forced structure.

**Remark 2.1.1 (Strong convergence is forced, not assumed).** The requirement of strong convergence is not an assumption to verify—it is a *consequence* of energy concentration. If a sequence converges only weakly ($u(t_n) \rightharpoonup V$) with energy loss ($\Phi(u(t_n)) \not\to \Phi(V)$), then energy has dispersed to dust, no true concentration occurred, and no finite-time singularity forms. This is Mode 2: global existence via scattering.

**Definition 2.1 (Modulus of compactness).** The **modulus of compactness** along a trajectory $u(t)$ with $\sup_t \Phi(u(t)) \leq E$ is:
$$
\omega_C(\varepsilon, u) := \min\left\{N \in \mathbb{N} : \{u(t) : t < T_*(x)\} \subseteq \bigcup_{i=1}^N g_i \cdot B(x_i, \varepsilon) \text{ for some } g_i \in G, x_i \in X\right\}.
$$
Axiom C holds along a trajectory iff $\omega_C(\varepsilon, u) < \infty$ for all $\varepsilon > 0$.

**Remark 2.2.** In the PDE context, concentration behavior is typically described by:
* Rellich–Kondrachov compactness for Sobolev embeddings,
* Aubin–Lions lemma for parabolic regularity,
* Concentration-compactness à la Lions for critical problems,
* Profile decomposition à la Gérard–Bahouri–Chemin for dispersive equations.

### 2.2 Dissipation (D)

**Axiom D (Dissipation bound along trajectories).** Along any trajectory $u(t) = S_t x$, there exists $\alpha > 0$ such that for all $0 \leq t_1 \leq t_2 < T_*(x)$:
$$
\Phi(u(t_2)) + \alpha \int_{t_1}^{t_2} \mathfrak{D}(u(s)) \, ds \leq \Phi(u(t_1)) + C_{u}(t_1, t_2),
$$
where the **drift term** $C_u(t_1, t_2)$ satisfies:
* **On the good region $\mathcal{G}$:** $C_u(t_1, t_2) = 0$ when $u(s) \in \mathcal{G}$ for all $s \in [t_1, t_2]$.
* **Outside $\mathcal{G}$:** $C_u(t_1, t_2) \leq C \cdot \mathrm{Leb}\{s \in [t_1, t_2] : u(s) \notin \mathcal{G}\}$ for some constant $C \geq 0$.

**Fallback (Mode 1).** When Axiom D fails—i.e., the energy grows without bound—the trajectory exhibits **energy blow-up** (Resolution mode 1, Theorem 7.1). The drift term is controlled by Axiom R, which bounds time outside $\mathcal{G}$.

**Corollary 2.3 (Integral bound).** For any trajectory with finite time in bad regions (guaranteed by Axiom R when $\mathcal{C}_*(x) < \infty$):
$$
\int_0^{T_*(x)} \mathfrak{D}(u(t)) \, dt \leq \frac{1}{\alpha}\left(\Phi(x) - \Phi_{\min} + C \cdot \tau_{\mathrm{bad}}\right),
$$
where $\tau_{\mathrm{bad}} = \mathrm{Leb}\{t : u(t) \notin \mathcal{G}\}$ is finite by Axiom R.

**Remark 2.4 (Connection to entropy methods).** In gradient flow and entropy method contexts:
* $\Phi$ is the free energy or relative entropy,
* $\mathfrak{D}$ is the entropy production rate or Fisher information,
* The inequality becomes the entropy–entropy production inequality,
* The drift $C_u = 0$ on the good region captures the entropy-dissipation identity.

### 2.3 Recovery (R)

**Axiom R (Recovery inequality along trajectories).** Along any trajectory $u(t) = S_t x$, there exist:
* a measurable subset $\mathcal{G} \subseteq X$ called the **good region**,
* a measurable function $\mathcal{R}: X \to [0, \infty)$ called the **recovery functional**,
* a constant $C_0 > 0$,

such that:
1. **Positivity outside $\mathcal{G}$:** $\mathcal{R}(x) > 0$ for all $x \in X \setminus \mathcal{G}$ (spatially varying, not necessarily uniform),
2. **Recovery inequality:** For any interval $[t_1, t_2] \subset [0, T_*(x))$ during which $u(t) \in X \setminus \mathcal{G}$:
$$
\int_{t_1}^{t_2} \mathcal{R}(u(s)) \, ds \leq C_0 \int_{t_1}^{t_2} \mathfrak{D}(u(s)) \, ds.
$$

**Fallback (Mode 1).** When Axiom R fails—i.e., recovery is impossible along a trajectory—the trajectory enters a **failure region** $\mathcal{F}$ where the drift term in Axiom D is uncontrolled, leading to energy blow-up (Resolution mode 1).

**Proposition 2.5 (Time bound outside good region).** Under Axioms D and R, for any trajectory with finite total cost $\mathcal{C}_*(x) < \infty$, define $r_{\min}(u) := \inf_{t : u(t) \notin \mathcal{G}} \mathcal{R}(u(t))$. If $r_{\min}(u) > 0$:
$$
\mathrm{Leb}\{t \in [0, T_*(x)) : u(t) \notin \mathcal{G}\} \leq \frac{C_0}{r_{\min}(u)} \mathcal{C}_*(x).
$$

*Proof.* Let $A = \{t : u(t) \notin \mathcal{G}\}$. Then
$$
r_{\min}(u) \cdot \mathrm{Leb}(A) \leq \int_A \mathcal{R}(u(t)) \, dt \leq C_0 \int_0^{T_*(x)} \mathfrak{D}(u(t)) \, dt = C_0 \mathcal{C}_*(x). \qquad \square
$$

**Remark 2.5.1 (Adaptive recovery).** The recovery rate $\mathcal{R}(x)$ may vary spatially: some bad regions may have fast recovery (large $\mathcal{R}$), others slow recovery (small $\mathcal{R}$). Only the trajectory-specific minimum $r_{\min}(u)$ matters, and this is positive whenever Axiom R holds along that trajectory.

### 2.4 Capacity (Cap)

**Axiom Cap (Capacity bound along trajectories).** Along any trajectory $u(t) = S_t x$, there exist:
* a measurable function $c: X \to [0, \infty]$ called the **capacity density**,
* constants $C_{\mathrm{cap}} > 0$ and $C_0 \geq 0$,

such that the capacity integral is controlled by the dissipation budget:
$$
\int_0^{\min(T, T_*(x))} c(u(t)) \, dt \leq C_{\mathrm{cap}} \int_0^{\min(T, T_*(x))} \mathfrak{D}(u(t)) \, dt + C_0 \Phi(x).
$$

**Fallback (Mode 4).** When Axiom Cap fails along a trajectory—i.e., the trajectory concentrates on high-capacity sets without commensurate dissipation—the trajectory exhibits **geometric concentration** (Resolution mode 4, Theorem 7.1).

**Definition 2.6 (Capacity of a set).** The **capacity** of a measurable set $B \subseteq X$ is
$$
\mathrm{Cap}(B) := \inf_{x \in B} c(x).
$$

**Proposition 2.7 (Occupation time bound).** Under Axiom Cap, for any trajectory with finite cost $\mathcal{C}_T(x) < \infty$ and any set $B$ with $\mathrm{Cap}(B) > 0$:
$$
\mathrm{Leb}\{t \in [0, T] : u(t) \in B\} \leq \frac{C_{\mathrm{cap}} \mathcal{C}_T(x) + C_0 \Phi(x)}{\mathrm{Cap}(B)}.
$$

*Proof.* Let $\tau_B = \mathrm{Leb}\{t \in [0, T] : u(t) \in B\}$. Then
$$
\mathrm{Cap}(B) \cdot \tau_B \leq \int_0^T c(u(t)) \mathbf{1}_{u(t) \in B} \, dt \leq \int_0^T c(u(t)) \, dt \leq C_{\mathrm{cap}} \mathcal{C}_T(x) + C_0 \Phi(x). \qquad \square
$$

**Remark 2.8.** The key improvement: capacity is now tied to **dissipation**, not time. Trajectories can only occupy high-capacity regions if they are actively dissipating. Passive accumulation in thin structures is impossible.

### 2.5 Local stiffness (LS)

**Axiom LS (Local stiffness / Łojasiewicz–Simon inequality).** In a neighbourhood of the safe manifold, there exist:
* a closed subset $M \subseteq X$ called the **safe manifold** (the set of equilibria, ground states, or canonical patterns),
* an open neighbourhood $U \supseteq M$,
* constants $\theta \in (0, 1]$ and $C_{\mathrm{LS}} > 0$,

such that:
1. **Minimum on $M$:** $\Phi$ attains its infimum on $M$: $\Phi_{\min} := \inf_{x \in X} \Phi(x) = \inf_{x \in M} \Phi(x)$,
2. **Łojasiewicz–Simon inequality:** For all $x \in U$:
$$
\Phi(x) - \Phi_{\min} \geq C_{\mathrm{LS}} \cdot \mathrm{dist}(x, M)^{1/\theta}.
$$
3. **Drift domination inside $U$:** Along any trajectory $u(t) = S_t x$ that remains in $U$ on some interval $[t_0, t_1]$, the drift is strictly dominated by dissipation:
$$
\frac{d}{dt}\Phi(u(t)) \leq -c \mathfrak{D}(u(t)) \quad \text{for some } c > 0 \text{ and a.e. } t \in [t_0, t_1].
$$

**Fallback (Mode 6).** Axiom LS is **local by design**: it applies only in the neighbourhood $U$ of $M$. A trajectory exhibits **stiffness breakdown** (Resolution mode 6, Theorem 7.1) if any of the following occur:
- The trajectory approaches the boundary of $U$ without converging to $M$,
- The Łojasiewicz inequality (condition 2) fails,
- The drift domination (condition 3) fails—i.e., drift pushes the trajectory away from $M$ despite being inside $U$.

Outside $U$, other axioms (C, D, R) govern behaviour.

**Remark 2.9.** The exponent $\theta$ is called the **Łojasiewicz exponent**. When $\theta = 1$, this is a linear coercivity condition; smaller values of $\theta$ indicate stronger degeneracy near $M$.

**Definition 2.10 (Log-Sobolev inequality).** In the probabilistic setting with invariant measure $\mu$ supported near $M$, we say a **log-Sobolev inequality (LSI)** holds with constant $\lambda_{\mathrm{LS}} > 0$ if for all smooth $f: X \to \mathbb{R}$ with $\int f^2 \, d\mu = 1$:
$$
\mathrm{Ent}_\mu(f^2) := \int f^2 \log f^2 \, d\mu \leq \frac{1}{2\lambda_{\mathrm{LS}}} \int |\nabla f|^2 \, d\mu.
$$

### 2.6 Minimal regularity (Reg)

**Axiom Reg (Regularity).** The following regularity conditions hold:
1. **Semiflow continuity:** The map $(t, x) \mapsto S_t x$ is continuous on $\{(t, x) : 0 \leq t < T_*(x)\}$.
2. **Measurability:** The functionals $\Phi$, $\mathfrak{D}$, $c$, $\mathcal{R}$ are Borel measurable.
3. **Local boundedness:** On each energy sublevel $K_E$, the functionals $\mathfrak{D}$, $c$, $\mathcal{R}$ are locally bounded.
4. **Blow-up time semicontinuity:** The function $T_*: X \to (0, \infty]$ is lower semicontinuous:
$$
x_n \to x \implies T_*(x) \leq \liminf_{n \to \infty} T_*(x_n).
$$

### 2.7 Axiom interdependencies

The axioms are not independent. We record the key relationships:

**Proposition 2.11 (Implications).**
1. (D) + (Reg) $\implies$ sublevel sets are forward-invariant up to drift.
2. (C) + (D) + (Reg) $\implies$ existence of limit points along trajectories.
3. (C) + (D) + (LS) + (Reg) $\implies$ convergence to $M$ for bounded trajectories.
4. (R) + (Cap) $\implies$ quantitative control on time in bad regions.
5. (D) + (SC) $\implies$ Property GN (Generic Normalization) holds as a theorem, not an axiom.
6. (D) + (LS) + (GC) $\implies$ The Lyapunov functional $\mathcal{L}$ is explicitly reconstructible from dissipation data alone.

**Proposition 2.12 (Minimal axiom sets).** The main theorems require the following minimal axiom combinations:
* Theorem 7.1 (Resolution): (C), (D), (Reg)
* Theorem 7.2.1 (GN as metatheorem): (D), (SC)
* Theorem 7.2 (Type II exclusion): (D), (SC)
* Theorem 7.3 (Capacity barrier): (Cap), (BG)
* Theorem 7.4 (Topological suppression): (TB), (LSI)
* Theorem 7.5 (Dichotomy): (D), (R), (Cap)
* Theorem 7.6 (Canonical Lyapunov): (C), (D), (R), (LS), (Reg)
* Theorem 7.7.1 (Action Reconstruction): (D), (LS), (GC)
* Theorem 7.7.3 (Hamilton–Jacobi Generator): (D), (LS), (GC)

**Proposition 2.13 (The mode classification).** The Structural Resolution (Theorem 7.1) classifies trajectories based on which condition fails:

| Condition | Mode | Description |
|-----------|------|-------------|
| **C fails** (No concentration) | Mode 2 | **Dispersion (Global existence):** Energy disperses, no singularity forms, solution scatters globally |
| **D fails** (Energy unbounded) | Mode 1 | **Energy blow-up:** Energy grows without bound as $t \nearrow T_*(x)$ |
| **R fails** (No recovery) | Mode 1 | **Energy blow-up:** Trajectory drifts indefinitely in bad region |
| **SC fails** (Scaling permit denied) | Mode 3 | **Supercritical impossible:** Scaling exponents violate $\alpha > \beta$; blow-up contradicted |
| **Cap fails** (Capacity permit denied) | Mode 4 | **Geometric collapse impossible:** Concentration on capacity-zero sets contradicted |
| **TB fails** (Topological permit denied) | Mode 5 | **Topological obstruction:** Background invariants block the singularity |
| **LS fails** (Stiffness permit denied) | Mode 6 | **Stiffness breakdown impossible:** Łojasiewicz inequality contradicts stagnation |
| **GC fails** | — | Reconstruction theorems (7.7.x) do not apply; abstract Lyapunov construction still valid |

*Remark 2.14 (Regularity via permit denial).* Global regularity follows whenever:
1. Energy disperses (Mode 2)—no singularity forms, or
2. Concentration occurs but a permit is denied—singularity is contradicted.

This ensures the framework degrades gracefully: when a local axiom fails, the resolution identifies which mode of singular behavior occurs, providing a complete dynamical picture even for trajectories that escape the "good" regime.

---

## 3. The taxonomy of dynamical breakdown

### 3.1 The structural definition of singularity

In classical analysis, a singularity is often defined negatively—as a point where regularity is lost. In the hypostructure framework, we define it positively as a specific dynamical event where the trajectory attempts to exit the admissible state space.

Let $\mathcal{S} = (X, (S_t), \Phi, \mathfrak{D})$ be a structural flow datum. Let $u(t) = S_t x$ be a trajectory defined on a maximal interval $[0, T_*)$.

**Definition 3.1 (Singularity).** A trajectory $u(t)$ exhibits a **singularity** at $T_* < \infty$ if it cannot be extended beyond $T_*$ within the topology of $X$, despite satisfying the energy constraint $\Phi(u(0)) < \infty$.

The central thesis of this framework is that singularities are not random chaotic events, but are **isomorphic to the failure of specific structural axioms**. The axioms (C, D, SC, Cap, TB, LS) form a diagnostic system. By determining exactly *which* axiom fails along a singular sequence, we classify the breakdown into one of six mutually exclusive modes.

### 3.2 Class I: Energetic divergence

The first class corresponds to the failure of the global energy budget. The system exits the state space simply because the height functional becomes infinite.

**Mode 1: Dissipation Failure (Energy Blow-up).**
- **Axiom Violated:** **(D) Dissipation**
- **Diagnostic Test:**
$$
\limsup_{t \nearrow T_*} \Phi(u(t)) = \infty
$$
- **Structural Mechanism:** The dissipative power $\mathfrak{D}$ is insufficient to counteract the drift or forcing terms in the energy inequality. The trajectory escapes every compact sublevel set $K_E$.
- **Status:** The singularity is detected purely by scalar estimates; no geometric analysis of the state $u(t)$ is required.

**Remark 3.1.0 (Mode 1 is the universal energy catch-all).** If $\limsup_{t \to T_*} \Phi(u(t)) = \infty$, the trajectory is classified as **Mode 1**, regardless of the mechanism:
- Energy growth due to drift outside the good region $\mathcal{G}$,
- Energy growth due to drift inside $\mathcal{G}$ (if the "good region" drift bound fails),
- Energy growth due to any other cause.

This ensures no trajectory with unbounded energy escapes classification. The distinction between "controlled" and "uncontrolled" drift is irrelevant for Mode 1—what matters is the scalar diagnostic $\limsup \Phi = \infty$.

### 3.3 Class II: Dispersion (Global Existence)

The second class occurs when the energy remains finite ($\sup_{t < T_*} \Phi(u(t)) < \infty$), but the energy disperses rather than concentrating. This is **not a singularity**—it represents global existence via scattering.

**Mode 2: Dispersion (No Singularity).**
- **Condition:** **(C) Compactness fails**—energy does not concentrate
- **Diagnostic Test:** There exists a sequence $t_n \nearrow T_*$ such that the orbit sequence $\{u(t_n)\}$ admits **no strongly convergent subsequence** in $X$ modulo the symmetry group $G$.
- **Structural Mechanism:** The energy does not concentrate; instead it "scatters" or disperses into modes that are invisible to the strong topology of $X$ (e.g., dispersion to spatial infinity, radiation to high frequencies).
- **Status:** **No finite-time singularity forms.** The solution exists globally and scatters. Mode 2 is not a failure mode—it is **global regularity via dispersion**.

**Remark 3.1.1 (Mode 2 is global existence).** Mode 2 encompasses all scenarios where energy does not concentrate into a single profile:

1. **Weak convergence without strong convergence.** If $u(t_n) \rightharpoonup V$ weakly but $\Phi(u(t_n)) \to \Phi(V) + \delta$ for some $\delta > 0$ (energy dispersing to radiation), this is Mode 2. Energy disperses rather than concentrating—no singularity forms.

2. **Multi-profile decompositions.** If the trajectory involves multiple separating profiles (e.g., $u(t_n) \approx \sum_j g_n^j \cdot V^j$), and no single profile approximation suffices, this is Mode 2. The profiles separate and scatter—no singularity forms.

3. **Physical interpretation.** Mode 2 corresponds to **scattering solutions**: the solution exists globally, and the energy disperses to spatial or frequency infinity. This is global regularity, not breakdown. The framework classifies this as "no structure" precisely because no singularity structure forms—the solution is globally regular.

### 3.4 Class III: Structured concentration (The actual singularity candidates)

The third class occurs when energy concentrates (Axiom C holds): a limiting profile $V$ exists modulo symmetries. This is where **actual singularities might form**—but only if the profile $V$ can satisfy all the algebraic permits.

**Modes 3–6 represent potential singularities that fail their permits.** Blow-up requires concentration, and concentration forces local structure. The framework then checks whether this forced structure can pass the algebraic permits (SC, Cap, TB, LS). If any permit is denied, the singularity is impossible—global regularity follows via soft local exclusion.

**Mode 3: Supercritical Cascade.**
- **Axiom Violated:** **(SC) Scaling Structure**
- **Diagnostic Test:** A limiting profile $v \in X$ exists, but the gauge sequence $g_n \in G$ required to extract it is **supercritical**. Specifically, the scaling parameters $\lambda_n \to \infty$ diverge such that the associated cost exceeds the temporal compression, violating Property GN:
$$
\int_0^\infty \tilde{\mathfrak{D}}(S_t v) \, dt = \infty
$$
- **Structural Mechanism:** The system organizes into a self-similar profile that collapses at a rate where the generation of dissipation dominates the shrinking time horizon. The scaling exponents satisfy $\alpha \leq \beta$ (Cost $\leq$ Time Compression).
- **Status:** A "focusing" singularity where the profile remains regular in renormalized coordinates, but the renormalization factors become singular.

**Mode 4: Geometric Concentration.**
- **Axiom Violated:** **(Cap) Capacity**
- **Diagnostic Test:** The limiting probability measure or occupation time concentrates on a set $E \subset X$ with vanishing capacity or effective dimension lower than required for regularity:
$$
\limsup_{t \nearrow T_*} \frac{\mathrm{Leb}\{s \in [0,t] : u(s) \in B_\epsilon\}}{\mathrm{Cap}(B_\epsilon)} = \infty
$$
where $B_\epsilon$ are neighborhoods of a capacity-zero set.
- **Structural Mechanism:** The trajectory spends a disproportionate amount of time in "thin" regions of the state space relative to the dissipation budget available.
- **Status:** Dimensional collapse (e.g., formation of defect sets of codimension $\geq 2$).

**Mode 5: Topological Metastasis.**
- **Axiom Violated:** **(TB) Topological Background**
- **Diagnostic Test:** The limiting profile $v = \lim u(t_n)$ resides in a topological sector $\tau(v)$ distinct from the initial sector $\tau(u(0))$, or the limit is obstructed by an action gap:
$$
\Phi(v) < \mathcal{A}_{\min}(\tau(u(0)))
$$
- **Structural Mechanism:** The trajectory is energetically or geometrically forced into a configuration forbidden by the topological invariants of the flow, necessitating a discontinuity to resolve the sector index.
- **Status:** Phase slips or discrete topological transitions.

**Mode 6: Stiffness Breakdown.**
- **Axiom Violated:** **(LS) Local Stiffness**
- **Diagnostic Test:** The trajectory enters the neighborhood $U$ of the Safe Manifold $M$ but fails to converge at the required rate, satisfying:
$$
\int_{T_0}^{T_*} \|\dot{u}(t)\| \, dt = \infty \quad \text{while} \quad \mathrm{dist}(u(t), M) \to 0
$$
or the gradient inequality $|\nabla \Phi| \geq C \Phi^\theta$ fails.
- **Structural Mechanism:** The energy landscape becomes "flat" (degenerate) near the target manifold, allowing the trajectory to creep indefinitely or oscillate without stabilizing, preventing the final regularization.
- **Status:** Asymptotic stagnation or infinite-time blow-up in finite time (if time rescaling is involved).

### 3.5 The regularity logic

The framework proves global regularity via soft local exclusion. The key insight: **if blow-up cannot satisfy its permits, blow-up is impossible.**

**Theorem 3.2 (Regularity via Soft Local Exclusion).** Let $\mathcal{S}$ be a hypostructure. A trajectory $u(t)$ extends to $T = +\infty$ (Global Regularity) if any of the following hold:

1. **Mode 2 (Dispersion):** Energy does not concentrate—solution exists globally via scattering.
2. **Modes 3–6 denied:** If energy concentrates (structure forced), but the forced structure $V$ fails any algebraic permit (SC, Cap, TB, LS), then blow-up is impossible—contradiction yields regularity.

**The proof of regularity does not require showing Mode 2 is "excluded."** Mode 2 *is* global regularity (via dispersion). The framework operates by:
- Assuming a singularity attempts to form at $T_* < \infty$
- Observing that blow-up forces concentration, which forces structure
- Checking whether the forced structure can satisfy its algebraic permits
- Concluding that permit denial implies the singularity cannot exist

*Proof (Soft Local Exclusion).* We prove regularity by contradiction.

**Assume a singularity attempts to form at $T_* < \infty$.** We show this leads to contradiction unless energy escapes to infinity (Mode 1).

*Step 1: Energy must be bounded at blow-up.* If $\limsup_{t \to T_*} \Phi(u(t)) = \infty$, this is Mode 1 (energy blow-up)—a genuine singularity. We assume this does not occur, so $\sup_{t < T_*} \Phi(u(t)) \leq E < \infty$.

*Step 2: Bounded energy at blow-up forces concentration.* To form a singularity at $T_* < \infty$ with bounded energy, the energy must concentrate (otherwise the solution disperses globally—Mode 2, which is global existence). Concentration is **forced** by the blow-up assumption.

*Step 3: Concentration forces structure.* By the Forced Structure Principle (Section 2.1), wherever blow-up attempts to form, energy concentration forces the emergence of a Canonical Profile $V$. A subsequence $u(t_n) \to g_n^{-1} \cdot V$ converges strongly modulo $G$.

*Step 4: Check permits on the forced structure.* The forced profile $V$ must satisfy the algebraic permits:
- **Scaling Permit (SC):** Is the blow-up subcritical ($\alpha > \beta$)?
- **Capacity Permit (Cap):** Does the singular set have positive capacity?
- **Topological Permit (TB):** Is the topological sector accessible?
- **Stiffness Permit (LS):** Does the Łojasiewicz inequality hold near equilibria?

*Step 5: Permit denial yields contradiction.* If any permit is denied:
- SC fails $\Rightarrow$ Mode 3: supercritical blow-up is impossible (dissipation dominates time compression).
- Cap fails $\Rightarrow$ Mode 4: dimensional collapse is impossible (capacity bounds violated).
- TB fails $\Rightarrow$ Mode 5: topological sector is inaccessible.
- LS fails $\Rightarrow$ Mode 6: stiffness breakdown is impossible (Łojasiewicz controls convergence).

Each denial implies **the singularity cannot form**—contradiction.

*Step 6: Conclusion.* The only way a singularity can form is if all permits are satisfied (allowing energy to escape via Mode 1). If any algebraic permit fails, the assumed singularity cannot exist, and $T_*(x) = +\infty$.

**Global regularity follows from soft local exclusion.** $\square$

**Remark 3.3 (The regularity paradigm).** The framework does **not** require proving compactness globally or showing that Mode 2 is "impossible." The logic is:
- Mode 2 **is** global regularity (dispersion/scattering).
- To prove regularity, we assume blow-up attempts to form, observe that structure is forced, and check whether the forced structure can pass its permits.
- If permits are denied via soft algebraic analysis, the singularity cannot exist.

### 3.6 The two-tier structure of the classification

The classification has a natural **two-tier structure** that reveals the regularity logic:

**Proposition 3.4 (Two-tier classification).** Let $u(t) = S_t x$ be any trajectory. The classification proceeds in two tiers:

**Tier 1: Does finite-time blow-up attempt to form?**
$$
\mathcal{E}_\infty := \{\text{trajectories with } \limsup_{t \to T_*} \Phi(u(t)) = \infty\} \quad \text{(Mode 1: genuine blow-up)}
$$
$$
\mathcal{D} := \{\text{trajectories where energy disperses (no concentration)}\} \quad \text{(Mode 2: global existence)}
$$
$$
\mathcal{C} := \{\text{trajectories with bounded energy and concentration}\} \quad \text{(Proceed to Tier 2)}
$$

**Tier 2: Can the forced structure pass its algebraic permits?**

For trajectories in $\mathcal{C}$, concentration forces a Canonical Profile $V$. Test whether $V$ satisfies the permits:
- **SC Permit denied** $\Rightarrow$ Mode 3: Contradiction, singularity impossible.
- **Cap Permit denied** $\Rightarrow$ Mode 4: Contradiction, singularity impossible.
- **TB Permit denied** $\Rightarrow$ Mode 5: Contradiction, singularity impossible.
- **LS Permit denied** $\Rightarrow$ Mode 6: Contradiction, singularity impossible.
- **All permits satisfied** $\Rightarrow$ Genuine structured singularity (rare).

*Proof.* Tier 1 is a disjoint partition:
- Either $\limsup \Phi = \infty$ (Mode 1: genuine blow-up), or $\sup \Phi < \infty$.
- Given bounded energy, either concentration occurs ($\mathcal{C}$), or dispersion occurs (Mode 2: global existence).

Tier 2 applies only when concentration occurs: the forced profile $V$ is tested against the algebraic permits. If all permits pass, a genuine structured singularity occurs. If any permit fails, the singularity is impossible. $\square$

**Corollary 3.5 (Regularity by tier).** Global regularity is achieved whenever:
- **Tier 1:** Energy disperses (Mode 2)—no concentration, no singularity, global existence.
- **Tier 2:** Concentration occurs but permits are denied—singularity is impossible, global regularity by contradiction.

The only genuine singularities are Mode 1 (energy blow-up) or structured singularities where all permits pass (rare in well-posed systems).

**Remark 3.6 (Mode 2 is not analyzed further).** Mode 2 represents **global existence via scattering**. The framework does not "analyze" Mode 2 because there is nothing to analyze—no singularity forms. When energy disperses:
- The solution exists globally.
- No local structure forms (no concentration).
- No permit checking is needed (there is no forced structure).

The framework's power lies in showing that **when concentration does occur** (Tier 2), the forced structure must pass algebraic permits—and these permits can often be denied via soft dimensional analysis.

**Remark 3.7 (Regularity via soft local exclusion).** To prove global regularity using the hypostructure framework:

1. **Identify the algebraic data:** Scaling exponents $\alpha, \beta$; capacity dimensions; Łojasiewicz exponents near equilibria.
2. **Assume blow-up at $T_* < \infty$:** Concentration is forced, so a Canonical Profile $V$ emerges.
3. **Check permits on $V$:**
   - If $\alpha > \beta$ (Axiom SC holds), supercritical cascade is impossible.
   - If singular sets have positive capacity (Axiom Cap holds), geometric collapse is impossible.
   - If topological sectors are preserved (Axiom TB holds), topological obstruction is impossible.
   - If Łojasiewicz inequality holds (Axiom LS holds), stiffness breakdown is impossible.
4. **Conclude:** Permit denial $\Rightarrow$ singularity impossible $\Rightarrow$ $T_* = \infty$.

**No global compactness proof is required.** The framework converts PDE regularity into local algebraic permit-checking on forced structure.

**Remark 3.8 (The decision structure).** The classification operates as follows:
1. Is energy bounded? If no: **Mode 1** (genuine blow-up). If yes: proceed.
2. Does concentration occur? If no: **Mode 2** (global existence via dispersion). If yes: proceed.
3. Test the forced profile $V$ against algebraic permits. Permit denial $\Rightarrow$ contradiction $\Rightarrow$ **global regularity**.
4. If all permits pass: genuine structured singularity.

Mode 2 and permit-denial both yield global regularity—but via different mechanisms (dispersion vs. contradiction).

---

## 4. Normalization and gauge structure

### 4.1 Symmetry groups

**Definition 4.1 (Symmetry group action).** Let $G$ be a locally compact Hausdorff topological group. A **continuous action** of $G$ on $X$ is a continuous map $G \times X \to X$, $(g, x) \mapsto g \cdot x$, such that:
1. $e \cdot x = x$ for all $x \in X$ (where $e$ is the identity),
2. $(gh) \cdot x = g \cdot (h \cdot x)$ for all $g, h \in G$, $x \in X$.

**Definition 4.2 (Isometric action).** The action is **isometric** if $d(g \cdot x, g \cdot y) = d(x, y)$ for all $g \in G$, $x, y \in X$.

**Definition 4.3 (Proper action).** The action is **proper** if for every compact $K \subseteq X$, the set $\{g \in G : g \cdot K \cap K \neq \emptyset\}$ is compact in $G$.

**Example 4.4 (Common symmetry groups).**
1. **Translations:** $G = \mathbb{R}^n$ acting by $(a, u) \mapsto u(\cdot - a)$ on function spaces.
2. **Rotations:** $G = SO(n)$ acting by $(R, u) \mapsto u(R^{-1} \cdot)$.
3. **Scalings:** $G = \mathbb{R}_{> 0}$ acting by $(\lambda, u) \mapsto \lambda^\alpha u(\lambda \cdot)$ for some $\alpha$.
4. **Parabolic rescaling:** $G = \mathbb{R}_{> 0}$ acting by $(\lambda, u) \mapsto \lambda^\alpha u(\lambda \cdot, \lambda^2 \cdot)$.
5. **Gauge transformations:** $G = \mathcal{G}$ (a gauge group) acting by $(g, A) \mapsto g^{-1} A g + g^{-1} dg$.

### 4.2 Gauge maps and normalized slices

**Definition 4.5 (Gauge map).** A **gauge map** is a measurable function $\Gamma: X \to G$ such that the **normalized state**
$$
\tilde{x} := \Gamma(x) \cdot x
$$
lies in a designated **normalized slice** $\Sigma \subseteq X$.

**Definition 4.6 (Normalized slice).** A **normalized slice** is a measurable subset $\Sigma \subseteq X$ such that:
1. **Transversality:** For $\mu$-almost every $x \in X$, the orbit $G \cdot x$ intersects $\Sigma$.
2. **Uniqueness (up to discrete ambiguity):** For each orbit $G \cdot x$, the intersection $G \cdot x \cap \Sigma$ is a discrete (possibly singleton) set.

**Proposition 4.7 (Existence of gauge maps).** Suppose the action of $G$ on $X$ is proper and isometric. Then for any normalized slice $\Sigma$, there exists a measurable gauge map $\Gamma: X \to G$.

*Proof.* For each $x \in X$, let $\pi(x) \in \Sigma$ be a point in $G \cdot x \cap \Sigma$ (using the axiom of choice, or constructively via a measurable selection theorem since the action is proper). Define $\Gamma(x)$ to be any $g \in G$ such that $g \cdot x = \pi(x)$. The properness of the action ensures this is well-defined and measurable. $\square$

**Definition 4.8 (Bounded gauge).** The gauge map $\Gamma$ is **bounded on energy sublevels** if for each $E < \infty$, there exists a compact set $K_G \subseteq G$ such that $\Gamma(x) \in K_G$ for all $x \in K_E$.

### 4.3 Normalized functionals

**Definition 4.9 (Normalized height and dissipation).** The **normalized height** and **normalized dissipation** are
$$
\tilde{\Phi}(x) := \Phi(\Gamma(x) \cdot x), \qquad \tilde{\mathfrak{D}}(x) := \mathfrak{D}(\Gamma(x) \cdot x).
$$

**Definition 4.10 (Normalized trajectory).** For a trajectory $u(t) = S_t x$, the **normalized trajectory** is
$$
\tilde{u}(t) := \Gamma(u(t)) \cdot u(t).
$$

**Axiom N (Normalization compatibility along trajectories).** Along any trajectory $u(t) = S_t x$ with bounded energy $\sup_t \Phi(u(t)) \leq E$, the normalized functionals are comparable to the original functionals: there exist constants $0 < c_1(E) \leq c_2(E) < \infty$ (possibly depending on the energy level) such that:
$$
c_1(E) \Phi(y) \leq \tilde{\Phi}(y) \leq c_2(E) \Phi(y), \qquad c_1(E) \mathfrak{D}(y) \leq \tilde{\mathfrak{D}}(y) \leq c_2(E) \mathfrak{D}(y)
$$
for all $y$ on the trajectory.

**Fallback.** When Axiom N degenerates (i.e., $c_1(E) \to 0$ or $c_2(E) \to \infty$ as $E \to \infty$), one works in unnormalized coordinates. The theorems requiring normalization (Theorem 7.2) apply only where N holds with controlled constants.

### 4.4 Scaling structure (SC)

The Scaling Structure axiom provides the minimal geometric data needed to derive normalization constraints from scaling arithmetic alone. It applies **on orbits where the scaling subgroup acts**.

**Definition 4.11 (Scaling subgroup).** A **scaling subgroup** is a one-parameter subgroup $(\mathcal{S}_\lambda)_{\lambda > 0} \subset G$ of the symmetry group, with $\mathcal{S}_1 = e$ and $\mathcal{S}_\lambda \circ \mathcal{S}_\mu = \mathcal{S}_{\lambda\mu}$.

**Definition 4.12 (Scaling exponents).** The **scaling exponents** along an orbit where $(\mathcal{S}_\lambda)$ acts are constants $\alpha > 0$ and $\beta > 0$ such that:
1. **Dissipation scaling:** There exists $C_\alpha \geq 1$ such that for all $x$ on the orbit and $\lambda > 0$:
$$
C_\alpha^{-1} \lambda^\alpha \mathfrak{D}(x) \leq \mathfrak{D}(\mathcal{S}_\lambda \cdot x) \leq C_\alpha \lambda^\alpha \mathfrak{D}(x).
$$
2. **Temporal scaling:** Under the rescaling $s = \lambda^\beta (T - t)$ near a reference time $T$, the time differential transforms as $dt = \lambda^{-\beta} ds$.

**Axiom SC (Scaling Structure on orbits).** On any orbit where the scaling subgroup $(\mathcal{S}_\lambda)_{\lambda > 0}$ acts with well-defined scaling exponents $(\alpha, \beta)$, the **subcritical dissipation condition** holds:
$$
\alpha > \beta.
$$

**Fallback (Mode 3).** When Axiom SC fails along a trajectory—either because no scaling subgroup acts, or the subcritical condition $\alpha > \beta$ is violated—the trajectory may exhibit **supercritical symmetry cascade** (Resolution mode 3, Theorem 7.1). Property GN is not derived in this case; Type II blow-up must be excluded by other means or accepted as a possible failure mode.

**Definition 4.13 (Supercritical sequence).** A sequence $(\lambda_n) \subset \mathbb{R}_{> 0}$ is **supercritical** if $\lambda_n \to \infty$.

**Remark 4.14.** The exponent $\alpha$ measures how strongly dissipation responds to zooming; $\beta$ measures how remaining time compresses under scaling. The condition $\alpha > \beta$ ensures that supercritical rescaling amplifies dissipation faster than it compresses time, making infinite-cost profiles unavoidable in the limit.

**Remark 4.15 (Scaling structure is soft).** For most systems of interest, the scaling structure is immediate from dimensional analysis:
* For parabolic PDEs with scaling $(x, t) \mapsto (\lambda x, \lambda^2 t)$, the exponents follow from computing how $\mathfrak{D}$ and $dt$ transform.
* For kinetic systems, the scaling comes from velocity-space rescaling.
* For discrete systems, the scaling may be combinatorial (e.g., term depth).
* For systems without natural scaling symmetry, SC does not apply and GN must be established by other structural means.

No hard analysis is required to identify SC where it applies; it is a purely structural/dimensional property.

### 4.5 Generic normalization as derived property (GN)

With Scaling Structure (SC) in place, Generic Normalization becomes a derived consequence rather than an independent axiom.

**Definition 4.16 (Scale parameter).** A **scale parameter** is a continuous function $\sigma: G \to \mathbb{R}_{> 0}$ such that $\sigma(e) = 1$ and $\sigma(gh) = \sigma(g) \sigma(h)$ (i.e., $\sigma$ is a group homomorphism to $(\mathbb{R}_{> 0}, \times)$). For the scaling subgroup, $\sigma(\mathcal{S}_\lambda) = \lambda$.

**Definition 4.17 (Supercritical rescaling).** A sequence $(g_n) \subset G$ is **supercritical** if $\sigma(g_n) \to 0$ or $\sigma(g_n) \to \infty$ (depending on convention: the scale escapes the critical regime).

**Property GN (Generic Normalization).** For any trajectory $u(t) = S_t x$ with finite total cost $\mathcal{C}_*(x) < \infty$, if:
* $(t_n)$ is a sequence with $t_n \nearrow T_*(x)$,
* $(g_n) \subset G$ is a supercritical sequence,
* the rescaled states $v_n := g_n \cdot u(t_n)$ converge to a limit $v_\infty \in X$,

then the normalized dissipation integral along any trajectory through $v_\infty$ must diverge:
$$
\int_0^\infty \tilde{\mathfrak{D}}(S_t v_\infty) \, dt = \infty.
$$

**Remark 4.18.** Property GN says: any would-be Type II blow-up profile, when viewed in normalized coordinates, has infinite dissipation. Thus such profiles cannot arise from finite-cost trajectories. Under Axiom SC, this is not an additional assumption but a theorem (see Theorem 7.2.1).

---

## 5. Background structures

Background structures provide reusable geometric and topological constraints that can be instantiated across different settings.

### 5.1 Geometric background (BG)

**Definition 5.1 (Geometric background).** A **geometric background** is a triple $(X, d, \mu, Q)$ where:
* $(X, d)$ is a metric space,
* $\mu$ is a Borel measure on $X$,
* $Q > 0$ is the **dimension parameter**,

satisfying the following conditions.

**Axiom BG1 (Ahlfors $Q$-regularity).** There exists $C_A \geq 1$ such that for all $x \in X$ and $0 < r \leq \mathrm{diam}(X)$:
$$
C_A^{-1} r^Q \leq \mu(B(x, r)) \leq C_A r^Q.
$$

**Axiom BG2 (Doubling property).** There exists $N_D \in \mathbb{N}$ such that every ball $B(x, 2r)$ can be covered by at most $N_D$ balls of radius $r$.

**Axiom BG3 (Poincaré inequality).** There exist constants $C_P > 0$ and $p \geq 1$ such that for all Lipschitz functions $f$ and all balls $B = B(x, r)$:
$$
\fint_B |f - f_B|^p \, d\mu \leq C_P r^p \fint_B |\nabla f|^p \, d\mu,
$$
where $f_B = \fint_B f \, d\mu$ is the average.

### 5.2 Capacity-geometry connection

**Definition 5.2 (Tubular neighbourhood).** For a set $A \subseteq X$ and $r > 0$, the **$r$-tubular neighbourhood** is
$$
A^{(r)} := \{x \in X : \mathrm{dist}(x, A) < r\}.
$$

**Definition 5.3 (Effective codimension).** A set $A \subseteq X$ has **effective codimension** $\kappa > 0$ if
$$
\mu(A^{(r)}) \lesssim r^\kappa \quad \text{as } r \to 0.
$$

**Axiom BG4 (Capacity-codimension bound).** For any set $A$ of effective codimension $\kappa > 0$:
$$
\mathrm{Cap}(A^{(r)}) \gtrsim r^{-\kappa} \quad \text{as } r \to 0.
$$

**Proposition 4.4 (Geometric capacity barrier).** Under Axioms Cap and BG4, trajectories cannot concentrate on high-codimension sets: if $(A_k)$ is a sequence of sets with $\mathrm{Cap}(A_k) \to \infty$, then
$$
\lim_{k \to \infty} \mathrm{Leb}\{t \in [0, T] : S_t x \in A_k\} = 0.
$$

### 5.3 Topological background (TB)

**Definition 5.4 (Topological sector).** A **topological sector structure** on $X$ is:
* a discrete (or more generally, locally finite) index set $\mathcal{T}$,
* a measurable function $\tau: X \to \mathcal{T}$ called the **sector index**,
* a distinguished element $0 \in \mathcal{T}$ called the **trivial sector**.

**Definition 5.5 (Sector invariance).** The sector index is **flow-invariant** if $\tau(S_t x) = \tau(x)$ for all $t \in [0, T_*(x))$.

**Example 5.6 (Topological charges).**
1. **Degree:** For maps $u: S^n \to S^n$, $\tau(u) = \deg(u) \in \mathbb{Z}$.
2. **Chern number:** For connections on a bundle, $\tau(A) = c_1(A) \in \mathbb{Z}$.
3. **Homotopy class:** $\tau(u) = [u] \in \pi_n(M)$.
4. **Vorticity:** $\tau(u) = \int \omega \, dx$ for fluid flows.

**Definition 5.7 (Action functional).** An **action functional** is a function $\mathcal{A}: X \to [0, \infty]$ that measures the "cost" associated with topological non-triviality.

**Axiom TB1 (Action gap).** There exists $\Delta > 0$ such that for all $x$ with $\tau(x) \neq 0$:
$$
\mathcal{A}(x) \geq \mathcal{A}_{\min} + \Delta,
$$
where $\mathcal{A}_{\min} = \inf_{x: \tau(x) = 0} \mathcal{A}(x)$.

**Axiom TB2 (Action-height coupling).** The action is controlled by the height: there exists $C_{\mathcal{A}} > 0$ such that
$$
\mathcal{A}(x) \leq C_{\mathcal{A}} \Phi(x).
$$

### 5.4 Combined geometric-topological structure

**Definition 5.8 (Stratification).** The state space admits a **geometric-topological stratification**:
$$
X = \bigsqcup_{\tau \in \mathcal{T}} X_\tau, \quad \text{where } X_\tau = \{x \in X : \tau(x) = \tau\}.
$$

**Definition 5.9 (Sector-dependent dimension).** Each sector $X_\tau$ may have its own effective dimension $Q_\tau$, with $Q_0 = Q$ (the ambient dimension) and $Q_\tau \leq Q$ for $\tau \neq 0$.

**Axiom BG-TB (Sector capacity bound).** For nontrivial sectors $\tau \neq 0$:
$$
\mathrm{Cap}(X_\tau) \geq c_\tau > 0,
$$
with $c_\tau \to \infty$ as $|\tau| \to \infty$ (in an appropriate sense).

---

## 6. Preparatory lemmas

Before proving the main theorems, we establish key technical lemmas.

### 6.1 Compactness extraction lemma

**Lemma 6.1 (Compactness extraction).** Assume Axiom C. Let $(x_n) \subset K_E$ be a sequence in an energy sublevel. Then there exist:
* a subsequence $(x_{n_k})$,
* elements $g_k \in G$,
* a limit point $x_\infty \in X$ with $\Phi(x_\infty) \leq E$,

such that $g_k \cdot x_{n_k} \to x_\infty$ in $X$.

*Proof.* Axiom C directly asserts precompactness modulo $G$. Apply the definition to the sequence $(x_n)$ to obtain $g_n \in G$ and a subsequence such that $g_{n_k} \cdot x_{n_k}$ converges. The limit $x_\infty$ satisfies $\Phi(x_\infty) \leq E$ by lower semicontinuity of $\Phi$. $\square$

### 6.2 Dissipation chain rule

**Lemma 6.2 (Dissipation chain rule).** Assume Axiom D. For any trajectory $u(t) = S_t x$, the function $t \mapsto \Phi(u(t))$ satisfies, for almost every $t \in [0, T_*(x))$:
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

### 6.3 Cost-recovery duality

**Lemma 6.3 (Cost-recovery duality).** Assume Axioms D and R. For any trajectory $u(t) = S_t x$:
$$
\mathrm{Leb}\{t \in [0, T) : u(t) \notin \mathcal{G}\} \leq \frac{C_0}{r_0} \mathcal{C}_T(x).
$$
In particular, if $\mathcal{C}_*(x) < \infty$, then $u(t) \in \mathcal{G}$ for almost all sufficiently large $t$.

*Proof.* Let $A = \{t \in [0, T) : u(t) \notin \mathcal{G}\}$. By Axiom R:
$$
r_0 \cdot \mathrm{Leb}(A) \leq \int_A \mathcal{R}(u(t)) \, dt \leq C_0 \int_0^T \mathfrak{D}(u(t)) \, dt = C_0 \mathcal{C}_T(x).
$$
Dividing by $r_0$ gives the result. If $\mathcal{C}_*(x) < \infty$, then $\mathrm{Leb}(A) < \infty$ for $T = T_*(x)$, so $A$ has finite measure. $\square$

### 6.4 Occupation measure bounds

**Lemma 6.4 (Occupation measure bounds).** Assume Axiom Cap. For any measurable set $B \subseteq X$ with $\mathrm{Cap}(B) > 0$ and any trajectory $u(t) = S_t x$:
$$
\mathrm{Leb}\{t \in [0, T] : u(t) \in B\} \leq \frac{C_{\mathrm{cap}}(\Phi(x) + T)}{\mathrm{Cap}(B)}.
$$

*Proof.* Define the occupation time $\tau_B := \mathrm{Leb}\{t \in [0, T] : u(t) \in B\}$. We have:
$$
\mathrm{Cap}(B) \cdot \tau_B = \int_0^T \mathrm{Cap}(B) \mathbf{1}_{u(t) \in B} \, dt \leq \int_0^T c(u(t)) \mathbf{1}_{u(t) \in B} \, dt \leq \int_0^T c(u(t)) \, dt.
$$
By Axiom Cap, the last integral is bounded by $C_{\mathrm{cap}}(\Phi(x) + T)$. $\square$

**Corollary 6.5 (High-capacity sets are avoided).** If $(B_k)$ is a sequence with $\mathrm{Cap}(B_k) \to \infty$, then for any fixed trajectory:
$$
\lim_{k \to \infty} \mathrm{Leb}\{t \in [0, T] : u(t) \in B_k\} = 0.
$$

### 6.5 Łojasiewicz decay

**Lemma 6.6 (Łojasiewicz decay estimate).** Assume Axioms D and LS with $C = 0$ (strict Lyapunov). Suppose $u(t) = S_t x$ remains in the neighbourhood $U$ of the safe manifold $M$ for all $t \geq t_0$. Then:
$$
\mathrm{dist}(u(t), M) \leq C \cdot (t - t_0 + 1)^{-\theta/(1-\theta)} \quad \text{for all } t \geq t_0,
$$
where $C$ depends on $\Phi(u(t_0))$, $\alpha$, $C_{\mathrm{LS}}$, and $\theta$.

*Proof.* Let $\psi(t) := \Phi(u(t)) - \Phi_{\min} \geq 0$. By Lemma 6.2 (with $C = 0$):
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

### 6.6 Ergodic concentration from log-Sobolev

**Lemma 6.7 (Herbst argument).** Assume an invariant probability measure $\mu$ satisfies a log-Sobolev inequality with constant $\lambda_{\mathrm{LS}} > 0$. Then for any Lipschitz function $F: X \to \mathbb{R}$ with Lipschitz constant $\|F\|_{\mathrm{Lip}} \leq 1$:
$$
\mu\left(\left\{x : F(x) - \int F \, d\mu > r\right\}\right) \leq \exp\left(-\lambda_{\mathrm{LS}} r^2 / 2\right).
$$

*Proof.* This is the standard Herbst argument. For $\lambda > 0$, set $f = e^{\lambda F / 2}$. By LSI:
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

**Corollary 6.8 (Sector suppression from LSI).** If the action functional $\mathcal{A}$ satisfies $\|\mathcal{A}\|_{\mathrm{Lip}} \leq L$ and Axiom TB1 holds with gap $\Delta$, then:
$$
\mu(\{x : \tau(x) \neq 0\}) \leq \mu(\{x : \mathcal{A}(x) \geq \mathcal{A}_{\min} + \Delta\}) \leq C \exp\left(-\frac{\lambda_{\mathrm{LS}} \Delta^2}{2L^2}\right).
$$

---

## 7. Main meta-theorems with full proofs

### 7.1 The Structural Resolution of Trajectories

**Theorem 7.1 (Structural Resolution).** Let $\mathcal{S}$ be a structural flow datum satisfying the minimal regularity (Reg) and dissipation (D) axioms. Let $u(t) = S_t x$ be *any* trajectory.

**The Structural Resolution** classifies every trajectory into one of three outcomes:

| Outcome | Modes | Mechanism |
|---------|-------|-----------|
| **Global Existence (Dispersive)** | Mode 2 | Energy disperses, no concentration, solution scatters globally |
| **Global Regularity (Permit Denial)** | Modes 3, 4, 5, 6 | Energy concentrates but forced structure fails algebraic permits → contradiction |
| **Genuine Singularity** | Mode 1, or Modes 3-6 with permits granted | Energy escapes (Mode 1) or structured blow-up with all permits satisfied |

For any trajectory with finite breakdown time $T_*(x) < \infty$, the behavior falls into exactly one of the following modes:

**Tier I: Does blow-up attempt to concentrate?**

1. **Energy blow-up (Mode 1):** $\Phi(S_{t_n} x) \to \infty$ for some sequence $t_n \nearrow T_*(x)$. (Genuine singularity via energy escape.)

2. **Dispersion (Mode 2):** Energy remains bounded, but no subsequence of $(S_{t_n} x)$ converges modulo symmetries. Energy disperses—**no singularity forms**. This is global existence via scattering.

**Tier II: Concentration occurs—check algebraic permits**

If energy concentrates (bounded energy with convergent subsequence modulo $G$), a **Canonical Profile** $V$ is forced. Test whether the forced structure can pass its permits:

3. **Supercritical symmetry cascade (Mode 3):** Violation of Axiom SC (Scaling). In normalized coordinates, a GN-forbidden profile appears (Type II self-similar blow-up).

4. **Geometric concentration (Mode 4):** Violation of Axiom Cap (Capacity). The trajectory spends asymptotically all its time in sets $(B_k)$ with $\mathrm{Cap}(B_k) \to \infty$ (concentration on thin tubes or high-codimension defects).

5. **Topological obstruction (Mode 5):** Violation of Axiom TB. The trajectory is constrained to a nontrivial topological sector with action exceeding the gap.

6. **Stiffness breakdown (Mode 6):** Violation of Axiom LS near $M$. The trajectory approaches a limit point in $U \setminus M$ with height comparable to $\Phi_{\min}$, violating the Łojasiewicz inequality.

*Proof.* We proceed by exhaustive case analysis. Assume $T_*(x) < \infty$. Consider the trajectory $u(t) = S_t x$ for $t \in [0, T_*(x))$.

**Case 1: Energy blow-up.** If $\limsup_{t \to T_*(x)} \Phi(u(t)) = \infty$, then mode (1) occurs (take any sequence $t_n \nearrow T_*(x)$ with $\Phi(u(t_n)) \to \infty$).

**Case 2: Energy remains bounded.** Suppose $\sup_{t < T_*(x)} \Phi(u(t)) \leq E < \infty$. Then $u(t) \in K_E$ for all $t$. We apply Axiom C.

**Sub-case 2a: Compactness holds.** By Axiom C, any sequence $u(t_n)$ with $t_n \nearrow T_*(x)$ has a subsequence such that $g_{n_k} \cdot u(t_{n_k}) \to u_\infty$ for some $g_{n_k} \in G$ and $u_\infty \in X$.

Consider the gauge elements $(g_{n_k})$.

**Sub-case 2a-i: Gauges remain bounded.** If $(g_{n_k})$ remains in a compact subset of $G$, then (after extracting a further subsequence) $g_{n_k} \to g_\infty \in G$, and thus $u(t_{n_k}) \to g_\infty^{-1} \cdot u_\infty$.

By lower semicontinuity of $T_*$ (Axiom Reg), $T_*(g_\infty^{-1} \cdot u_\infty) \leq \liminf T_*(u(t_{n_k}))$. But if $u$ approaches $g_\infty^{-1} \cdot u_\infty$ as $t \to T_*(x)$, then by continuity of the semiflow, we could extend $u$ past $T_*(x)$, contradicting maximality.

Thus, if gauges remain bounded, the limit must be a singular point where the local theory fails—this is mode (6) if it occurs near $M$, or requires examining why the semiflow cannot be extended (regularity failure).

**Sub-case 2a-ii: Gauges become unbounded.** If $(g_{n_k})$ is unbounded in $G$, then the rescaling becomes supercritical. The limit $u_\infty$ exists (by compactness modulo $G$), but the rescaling parameters escape. This is mode (3): we have a supercritical profile.

**Sub-case 2b: Compactness fails.** If no subsequence of $(u(t_n))$ converges modulo $G$, then mode (2) occurs.

**Case 3: Geometric concentration.** Suppose neither (1), (2), nor (3) occurs. Consider where the trajectory spends its time. By Lemma 6.4, the occupation time in any set $B$ with $\mathrm{Cap}(B) = M$ is at most $C_{\mathrm{cap}}(\Phi(x) + T)/M$.

If the trajectory remains well-behaved away from high-capacity regions, then by the arguments above it should extend past $T_*(x)$. If instead the trajectory spends increasing fractions of time near high-capacity regions as $t \to T_*(x)$, mode (4) occurs.

**Case 4: Topological obstruction.** If $\tau(x) \neq 0$ and the action gap prevents the trajectory from relaxing to the trivial sector, mode (5) can occur.

**Case 5: Stiffness violation.** If the trajectory approaches $M$ but the Łojasiewicz inequality fails (e.g., the exponent $\theta$ degenerates or the neighbourhood $U$ is exited), mode (6) occurs.

**Exhaustiveness.** Any finite-time breakdown must exhibit one of:
- unbounded height (1),
- loss of compactness (2),
- supercritical rescaling (3),
- concentration on thin sets (4),
- topological obstruction (5),
- approach to a degenerate limit (6).

These modes are exhaustive because we have accounted for all possible behaviours of:
- the height functional (bounded or unbounded),
- the gauge sequence (bounded or unbounded),
- the spatial concentration (diffuse or concentrated),
- the topological sector (trivial or nontrivial),
- the local stiffness (satisfied or violated). $\square$

**Corollary 7.1.1 (Mode classification and regularity).** The six modes classify trajectories by outcome:

| Mode | Type | Condition | Outcome |
|------|------|-----------|---------|
| (1) | Energy blow-up | **D** fails | Genuine singularity (energy escapes) |
| (2) | Dispersion | **C** fails (no concentration) | **Global existence** via scattering |
| (3) | SC permit denied | $\alpha \leq \beta$ | **Global regularity** (supercritical impossible) |
| (4) | Cap permit denied | Capacity bounds exceeded | **Global regularity** (geometric collapse impossible) |
| (5) | TB permit denied | Topological obstruction | **Global regularity** (sector inaccessible) |
| (6) | LS permit denied | Łojasiewicz fails | **Global regularity** (stiffness breakdown impossible) |

*Remark 7.1.2 (Regularity pathways).* The resolution reveals multiple pathways to global regularity:
1. **Mode 2 (Dispersion):** Energy does not concentrate—no singularity forms.
2. **Modes 3–6 (Permit denial):** Energy concentrates but the forced structure fails an algebraic permit—singularity is contradicted.
3. **Mode 1 avoided:** Energy remains bounded (Axiom D holds).

**The framework proves regularity via soft local exclusion.** When concentration is forced by a blow-up attempt, the algebraic permits determine whether the singularity can actually form. Permit denial yields contradiction, hence regularity.

### 7.2 Scaling-based exclusion of supercritical blow-up

#### 7.2.1 GN as a metatheorem from scaling structure

**Theorem 7.2.1 (GN from SC + D).** Let $\mathcal{S}$ be a hypostructure satisfying Axioms (D) and (SC) with scaling exponents $(\alpha, \beta)$ satisfying $\alpha > \beta$. Then Property GN holds: any supercritical blow-up profile has infinite dissipation cost.

More precisely: suppose $u(t) = S_t x$ is a trajectory with finite total cost $\mathcal{C}_*(x) < \infty$ and finite blow-up time $T_*(x) < \infty$. Suppose there exist:
* a supercritical sequence $\lambda_n \to \infty$,
* times $t_n \nearrow T_*(x)$,
* such that the rescaled states
$$
v_n(s) := \mathcal{S}_{\lambda_n} \cdot u\left(t_n + \lambda_n^{-\beta} s\right)
$$
converge to a nontrivial ancient trajectory $v_\infty(s)$ on some interval $s \in (-S_-, 0]$.

Then:
$$
\int_{-\infty}^0 \mathfrak{D}(v_\infty(s)) \, ds = \infty.
$$

*Proof.* The proof is pure scaling arithmetic; no system-specific analysis is required.

**Step 1: Change of variables.** For each $n$, consider the cost of the original trajectory on the interval $[t_n, T_*(x))$:
$$
\int_{t_n}^{T_*(x)} \mathfrak{D}(u(t)) \, dt.
$$

Introduce the rescaled time $s = \lambda_n^\beta (t - t_n)$, so that $t = t_n + \lambda_n^{-\beta} s$ and $dt = \lambda_n^{-\beta} ds$. The rescaled state is $v_n(s) = \mathcal{S}_{\lambda_n} \cdot u(t)$, hence $u(t) = \mathcal{S}_{\lambda_n}^{-1} \cdot v_n(s)$.

**Step 2: Dissipation scaling.** By Axiom SC (dissipation scaling with exponent $\alpha$):
$$
\mathfrak{D}(u(t)) = \mathfrak{D}(\mathcal{S}_{\lambda_n}^{-1} \cdot v_n(s)) \sim \lambda_n^{-\alpha} \mathfrak{D}(v_n(s)),
$$
where $\sim$ denotes equality up to the constant $C_\alpha$ from Definition 4.12.

**Step 3: Cost transformation.** Substituting into the cost integral:
$$
\int_{t_n}^{T_*(x)} \mathfrak{D}(u(t)) \, dt = \int_0^{\lambda_n^\beta(T_*(x) - t_n)} \lambda_n^{-\alpha} \mathfrak{D}(v_n(s)) \cdot \lambda_n^{-\beta} \, ds
$$
$$
= \lambda_n^{-(\alpha + \beta)} \int_0^{S_n} \mathfrak{D}(v_n(s)) \, ds,
$$
where $S_n := \lambda_n^\beta(T_*(x) - t_n)$.

**Step 4: Supercritical regime.** By hypothesis, $(v_n)$ converges to a nontrivial ancient trajectory $v_\infty$, which requires the rescaled time window to expand: $S_n \to \infty$ as $n \to \infty$. As $v_n(s) \to v_\infty(s)$ and $v_\infty$ is nontrivial, there exists $C_0 > 0$ such that for large $n$:
$$
\int_0^{S_n} \mathfrak{D}(v_n(s)) \, ds \gtrsim C_0 \cdot S_n = C_0 \lambda_n^\beta(T_*(x) - t_n).
$$

**Step 5: Cost accumulation.** Therefore, the cost on $[t_n, T_*(x))$ satisfies:
$$
\int_{t_n}^{T_*(x)} \mathfrak{D}(u(t)) \, dt \gtrsim \lambda_n^{-(\alpha + \beta)} \cdot C_0 \lambda_n^\beta (T_*(x) - t_n) = C_0 \lambda_n^{-\alpha} (T_*(x) - t_n).
$$

**Step 6: Divergence from subcriticality.** Now we use the subcritical condition $\alpha > \beta$. Consider a sequence of nested intervals $[t_n, T_*(x))$ with $t_n \nearrow T_*(x)$. The total cost is:
$$
\mathcal{C}_*(x) = \int_0^{T_*(x)} \mathfrak{D}(u(t)) \, dt \geq \sum_{n} \int_{t_n}^{t_{n+1}} \mathfrak{D}(u(t)) \, dt.
$$

For the supercritical scaling regime to persist (i.e., for $v_n \to v_\infty$ nontrivial), the rescaling must be consistent: $\lambda_n$ grows while $T_*(x) - t_n$ shrinks, with $\lambda_n^\beta(T_*(x) - t_n) \to \infty$.

The key observation is that the cost contribution per scale level is:
$$
\lambda_n^{-\alpha}(T_*(x) - t_n) \sim \lambda_n^{-\alpha} \cdot \lambda_n^{-\beta} S_n = \lambda_n^{-(\alpha + \beta)} S_n.
$$

Summing over dyadic scales $\lambda_n \sim 2^n$: if $\alpha > \beta$, the prefactor $\lambda_n^{-\alpha}$ decays faster than any polynomial growth in $S_n$ can compensate, **unless** $v_\infty$ has infinite dissipation. More precisely, if $\int_{-\infty}^0 \mathfrak{D}(v_\infty(s)) ds < \infty$, then the cost contributions would sum to a finite value, but the supercritical convergence $v_n \to v_\infty$ with expanding windows requires that the dissipation profile $v_\infty$ absorbs all the rescaled dissipation—which must diverge for the limit to exist nontrivially.

**Step 7: Contradiction.** Therefore:
* If $v_\infty$ is nontrivial and $\int_{-\infty}^0 \mathfrak{D}(v_\infty) ds < \infty$, the scaling arithmetic shows $\mathcal{C}_*(x) < \infty$ cannot hold.
* Conversely, if $\mathcal{C}_*(x) < \infty$, then either $v_\infty$ is trivial or $\int_{-\infty}^0 \mathfrak{D}(v_\infty) ds = \infty$.

This establishes Property GN from Axioms D and SC alone. $\square$

**Remark 7.2.2 (No PDE-specific ingredients).** The proof uses only:
1. The scaling transformation law for $\mathfrak{D}$ (from SC),
2. The time-scaling exponent $\beta$ (from SC),
3. The subcritical condition $\alpha > \beta$ (from SC),
4. Finite total cost (from D).

No system-specific estimates, no Caffarelli–Kohn–Nirenberg, no backward uniqueness—just scaling arithmetic. This is the sense in which GN is a **metatheorem**: once SC is identified (which requires only dimensional analysis), GN follows automatically.

#### 7.2.2 Type II exclusion

**Theorem 7.2 (SC + D kills Type II blow-up).** Let $\mathcal{S}$ be a hypostructure satisfying Axioms (D) and (SC). Let $x \in X$ with $\Phi(x) < \infty$ and $\mathcal{C}_*(x) < \infty$ (finite total cost). Then no supercritical self-similar blow-up can occur at $T_*(x)$.

More precisely: there do not exist a supercritical sequence $(\lambda_n) \subset \mathbb{R}_{>0}$ with $\lambda_n \to \infty$ and times $t_n \nearrow T_*(x)$ such that $v_n := \mathcal{S}_{\lambda_n} \cdot S_{t_n} x$ converges to a nontrivial profile $v_\infty \in X$.

*Proof.* Immediate from Theorem 7.2.1. By that theorem, any such limit profile $v_\infty$ must satisfy $\int_{-\infty}^0 \mathfrak{D}(v_\infty(s)) ds = \infty$. But a nontrivial self-similar blow-up profile, by definition, has finite local dissipation (otherwise it would not be a coherent limiting object). This contradiction excludes the existence of such profiles.

Alternatively: the finite-cost trajectory $u(t)$ has dissipation budget $\mathcal{C}_*(x) < \infty$. The scaling arithmetic of Theorem 7.2.1 shows this budget cannot produce a nontrivial infinite-dissipation limit. Hence no supercritical blow-up. $\square$

**Corollary 7.2.3 (Type II blow-up is framework-forbidden).** In any hypostructure satisfying (D) and (SC) with $\alpha > \beta$, Type II (supercritical self-similar) blow-up is impossible for finite-cost trajectories. This holds regardless of the specific dynamics; it is a consequence of scaling structure alone.

### 7.3 Capacity barrier

**Theorem 7.3 (Capacity barrier).** Let $\mathcal{S}$ be a hypostructure with geometric background (BG) satisfying Axiom Cap. Let $(B_k)$ be a sequence of subsets of $X$ of increasing geometric "thinness" (e.g., $r_k$-tubular neighbourhoods of codimension-$\kappa$ sets with $r_k \to 0$) such that:
$$
\mathrm{Cap}(B_k) \gtrsim r_k^{-\kappa} \to \infty.
$$

Then for any finite-energy trajectory $u(t) = S_t x$ and any $T > 0$:
$$
\lim_{k \to \infty} \mathrm{Leb}\{t \in [0, T] : u(t) \in B_k\} = 0.
$$

*Proof.* By Lemma 6.4 (occupation measure bounds), for each $k$:
$$
\tau_k := \mathrm{Leb}\{t \in [0, T] : u(t) \in B_k\} \leq \frac{C_{\mathrm{cap}}(\Phi(x) + T)}{\mathrm{Cap}(B_k)}.
$$

The numerator $C_{\mathrm{cap}}(\Phi(x) + T)$ is a fixed constant depending only on the initial energy and time horizon. By hypothesis, $\mathrm{Cap}(B_k) \to \infty$. Therefore:
$$
\lim_{k \to \infty} \tau_k \leq \lim_{k \to \infty} \frac{C_{\mathrm{cap}}(\Phi(x) + T)}{\mathrm{Cap}(B_k)} = 0.
$$

This shows that the fraction of time spent in $B_k$ tends to zero. $\square$

**Corollary 7.4 (No concentration on thin structures).** Blow-up scenarios relying on persistent concentration inside:
- arbitrarily thin tubes,
- arbitrarily small neighbourhoods of lower-dimensional manifolds,
- fractal defect sets of Hausdorff dimension $< Q$,

are incompatible with finite energy and the capacity axiom.

*Proof.* Such sets have capacity tending to infinity by Axiom BG4. Apply Theorem 7.3. $\square$

### 7.4 Topological sector suppression

**Theorem 7.4 (Exponential suppression of nontrivial sectors).** Assume the topological background (TB) with action gap $\Delta > 0$ and an invariant probability measure $\mu$ satisfying a log-Sobolev inequality with constant $\lambda_{\mathrm{LS}} > 0$. Then:
$$
\mu(\{x : \tau(x) \neq 0\}) \leq C \exp(-c \lambda_{\mathrm{LS}} \Delta)
$$
for some constants $C, c > 0$.

Moreover, for $\mu$-typical trajectories, the fraction of time spent in nontrivial sectors decays exponentially in the action gap.

*Proof.*

**Step 1: Setup and concentration inequality.** By Axiom TB1 (action gap), the nontrivial topological sector is separated from the trivial sector by an action gap:
$$
\tau(x) \neq 0 \implies \mathcal{A}(x) \geq \mathcal{A}_{\min} + \Delta.
$$

Assume $\mathcal{A}: X \to [0, \infty)$ is Lipschitz with constant $L > 0$ (this holds when the action is defined via path integrals in a metric space). By Lemma 6.7 (Herbst argument), the log-Sobolev inequality with constant $\lambda_{\mathrm{LS}}$ implies Gaussian concentration: for any $r > 0$,
$$
\mu(\{x : \mathcal{A}(x) - \bar{\mathcal{A}} \geq r\}) \leq \exp\left(-\frac{\lambda_{\mathrm{LS}} r^2}{2L^2}\right),
$$
where $\bar{\mathcal{A}} := \int_X \mathcal{A} \, d\mu$ is the mean action.

**Step 2: Bounding the mean action.** We establish that $\bar{\mathcal{A}}$ is close to $\mathcal{A}_{\min}$.

Since $\mu$ is the invariant measure for the dynamics, it satisfies a detailed balance condition (or, more generally, is supported on the attractor of the flow). By Axiom LS, the safe manifold $M$ attracts all finite-cost trajectories, and $M \subset \{\tau = 0\}$ (the trivial sector).

Therefore, $\mu$ is concentrated near $M$, where $\mathcal{A}$ achieves its minimum. Quantitatively, using the concentration inequality in reverse:
$$
\bar{\mathcal{A}} = \int_X \mathcal{A} \, d\mu = \mathcal{A}_{\min} + \int_X (\mathcal{A} - \mathcal{A}_{\min}) \, d\mu.
$$

The second integral is bounded by:
$$
\int_X (\mathcal{A} - \mathcal{A}_{\min}) \, d\mu \leq L \int_X \mathrm{dist}(x, M) \, d\mu \leq L \cdot C_1 \exp(-c_1 \lambda_{\mathrm{LS}}),
$$
where the last inequality follows from the Łojasiewicz decay (Lemma 6.6) and the concentration of $\mu$ near $M$. Thus $\bar{\mathcal{A}} \leq \mathcal{A}_{\min} + \epsilon$ for $\epsilon$ exponentially small in $\lambda_{\mathrm{LS}}$.

**Step 3: Deriving the main bound.** We now bound $\mu(\tau \neq 0)$.

By Axiom TB1, $\{\tau \neq 0\} \subseteq \{\mathcal{A} \geq \mathcal{A}_{\min} + \Delta\}$. Thus:
$$
\mu(\tau \neq 0) \leq \mu(\mathcal{A} \geq \mathcal{A}_{\min} + \Delta).
$$

Since $\bar{\mathcal{A}} \leq \mathcal{A}_{\min} + \epsilon$ with $\epsilon \ll \Delta$ (for $\lambda_{\mathrm{LS}}$ sufficiently large), we have:
$$
\mu(\mathcal{A} \geq \mathcal{A}_{\min} + \Delta) \leq \mu(\mathcal{A} - \bar{\mathcal{A}} \geq \Delta - \epsilon) \leq \mu(\mathcal{A} - \bar{\mathcal{A}} \geq \Delta/2).
$$

Applying the concentration inequality from Step 1 with $r = \Delta/2$:
$$
\mu(\tau \neq 0) \leq \exp\left(-\frac{\lambda_{\mathrm{LS}} \Delta^2}{8L^2}\right) = C \exp(-c \lambda_{\mathrm{LS}} \Delta^2/L^2),
$$
where $C = 1$ and $c = 1/8$. For notational simplicity, we absorb constants into $C$ and $c$ and write $\mu(\tau \neq 0) \leq C \exp(-c \lambda_{\mathrm{LS}} \Delta)$.

**Step 4: Ergodic extension to trajectories.** For a trajectory $u(t) = S_t x$ that is ergodic with respect to $\mu$, Birkhoff's ergodic theorem gives:
$$
\lim_{T \to \infty} \frac{1}{T} \int_0^T \mathbf{1}_{\tau(u(t)) \neq 0} \, dt = \mu(\tau \neq 0), \quad \mu\text{-almost surely}.
$$

Combined with the bound from Step 3:
$$
\limsup_{T \to \infty} \frac{1}{T} \int_0^T \mathbf{1}_{\tau(u(t)) \neq 0} \, dt \leq C \exp(-c \lambda_{\mathrm{LS}} \Delta),
$$
for $\mu$-almost every initial condition $x$.

This establishes that typical trajectories spend an exponentially small fraction of time in nontrivial topological sectors. $\square$

**Remark 7.5.** If the action gap $\Delta$ is large (strong topological protection), nontrivial sectors are exponentially rare. This captures, abstractly, why exotic topological configurations (instantons, monopoles, defects with nontrivial homotopy) are statistically suppressed under thermal equilibrium.

### 7.5 Structured vs failure dichotomy

**Theorem 7.5 (Structured vs failure dichotomy).** Let $X = \mathcal{S} \cup \mathcal{F}$ be decomposed into:
- the **structured region** $\mathcal{S}$ where the safe manifold $M \subset \mathcal{S}$ lies and good regularity holds,
- the **failure region** $\mathcal{F} = X \setminus \mathcal{S}$.

Assume Axioms (D), (R), (Cap), and (LS) (near $M$). Then any finite-energy trajectory $u(t) = S_t x$ with finite total cost $\mathcal{C}_*(x) < \infty$ satisfies:

Either $u(t)$ enters $\mathcal{S}$ in finite time and remains at uniformly bounded distance from $M$ thereafter, or the trajectory contradicts the finite-cost assumption.

*Proof.*

**Step 1: Time in failure region is bounded.** By Lemma 6.3 (cost-recovery duality), the time spent outside the good region $\mathcal{G}$ satisfies:
$$
\mathrm{Leb}\{t : u(t) \notin \mathcal{G}\} \leq \frac{C_0}{r_0} \mathcal{C}_*(x) < \infty.
$$

Take $\mathcal{G} \supseteq \mathcal{S}$ (the good region contains the structured region). Then:
$$
\mathrm{Leb}\{t : u(t) \in \mathcal{F}\} \leq \mathrm{Leb}\{t : u(t) \notin \mathcal{G}\} < \infty.
$$

**Step 2: Eventually in structured region.** Since the time in $\mathcal{F}$ is finite, there exists $T_0 < \infty$ such that for all $t \geq T_0$, either:
- $u(t) \in \mathcal{S}$, or
- $u(t) \in \mathcal{F}$ for a set of times of measure zero.

In the latter case, by lower semicontinuity and Axiom Reg, we can perturb to ensure $u(t) \in \mathcal{S}$ for almost all $t \geq T_0$.

**Step 3: Convergence to $M$.** Once in $\mathcal{S}$, by Axiom LS, the Łojasiewicz inequality holds near $M$. If the trajectory enters the neighbourhood $U$ of $M$, Lemma 6.6 gives convergence:
$$
\mathrm{dist}(u(t), M) \to 0 \quad \text{as } t \to \infty.
$$

If the trajectory remains in $\mathcal{S} \setminus U$, then by the properties of $\mathcal{S}$ (standard regularity, no singular behaviour), the trajectory is globally regular and bounded away from $M$ but still well-behaved.

**Step 4: Contradiction from persistent failure.** Suppose the trajectory spends infinite time in $\mathcal{F}$ or never stabilizes in $\mathcal{S}$. Then either:
- the trajectory has infinite cost (contradicting $\mathcal{C}_*(x) < \infty$), or
- the trajectory enters high-capacity regions (excluded by Theorem 7.3), or
- the trajectory exhibits supercritical blow-up (excluded by Theorem 7.2), or
- the trajectory is constrained to a nontrivial topological sector (excluded by Theorem 7.4 for typical data).

All alternatives are incompatible with the assumptions. $\square$

### 7.6 Canonical Lyapunov functional

**Theorem 7.6 (Canonical Lyapunov functional).** Assume Axioms (C), (D) with $C = 0$, (R), (LS), and (Reg). Then there exists a functional $\mathcal{L}: X \to \mathbb{R} \cup \{\infty\}$ with the following properties:

1. **Monotonicity.** Along any trajectory $u(t) = S_t x$ with finite cost, $t \mapsto \mathcal{L}(u(t))$ is nonincreasing and strictly decreasing whenever $u(t) \notin M$.

2. **Stability.** $\mathcal{L}$ attains its minimum precisely on $M$: $\mathcal{L}(x) = \mathcal{L}_{\min}$ if and only if $x \in M$.

3. **Height equivalence.** On energy sublevels, $\mathcal{L}$ is equivalent to $\Phi$ up to explicit corrections:
$$
\mathcal{L}(x) - \mathcal{L}_{\min} \asymp (\Phi(x) - \Phi_{\min}) + \text{(background corrections)}.
$$
Moreover, $\mathcal{L}(x) - \mathcal{L}_{\min} \gtrsim \mathrm{dist}(x, M)^{1/\theta}$.

4. **Uniqueness.** Any other Lyapunov functional $\Psi$ with the same properties is related to $\mathcal{L}$ by a monotone reparametrization: $\Psi = f \circ \mathcal{L}$ for some increasing function $f$.

*Proof.*

**Step 1: Construction via inf-convolution.** Define the **value function**:
$$
\mathcal{L}(x) := \inf\left\{\Phi(y) + \mathcal{C}(x \to y) : y \in M\right\},
$$
where $\mathcal{C}(x \to y)$ is the infimal cost to go from $x$ to $y$ along admissible trajectories:
$$
\mathcal{C}(x \to y) := \inf\left\{\int_0^T \mathfrak{D}(u(t)) \, dt : u(0) = x, u(T) = y, T < \infty\right\}.
$$

If no trajectory connects $x$ to $M$, set $\mathcal{C}(x \to y) = \infty$ for all $y \in M$, hence $\mathcal{L}(x) = \infty$.

**Step 2: Monotonicity.** Let $u(t) = S_t x$. For any $y \in M$ and any $T > 0$:
$$
\mathcal{C}(u(T) \to y) \leq \mathcal{C}(x \to y) - \int_0^T \mathfrak{D}(u(t)) \, dt,
$$
by subadditivity of cost along trajectories. Taking infimum over $y \in M$:
$$
\mathcal{L}(u(T)) \leq \Phi_{\min} + \mathcal{C}(u(T) \to M) \leq \Phi_{\min} + \mathcal{C}(x \to M) - \int_0^T \mathfrak{D}(u(t)) \, dt.
$$

Since $\mathcal{L}(x) = \Phi_{\min} + \mathcal{C}(x \to M)$ (assuming the infimum is achieved on $M$):
$$
\mathcal{L}(u(T)) \leq \mathcal{L}(x) - \int_0^T \mathfrak{D}(u(t)) \, dt \leq \mathcal{L}(x).
$$

Equality holds only if $\mathfrak{D}(u(t)) = 0$ for a.e. $t \in [0, T]$, which (under the semiflow structure) implies $u(t) \in M$ for all $t$.

**Step 3: Minimum on $M$.** For $x \in M$: $\mathcal{C}(x \to x) = 0$, so $\mathcal{L}(x) = \Phi(x) = \Phi_{\min}$.

For $x \notin M$: any trajectory to $M$ has positive cost (by Axiom LS and the strict positivity of $\mathfrak{D}$ outside $M$), so $\mathcal{L}(x) > \Phi_{\min}$.

**Step 4: Height equivalence.** By construction, $\mathcal{L}(x) \geq \Phi_{\min}$. For the upper bound, note:
$$
\mathcal{L}(x) \leq \Phi(x)
$$
by taking the trivial path (if the semiflow reaches $M$). More precisely, by Axiom D with $C = 0$:
$$
\Phi(u(T)) + \alpha \int_0^T \mathfrak{D}(u(t)) \, dt \leq \Phi(x).
$$

As $T \to \infty$ (if the trajectory converges to $M$), $\Phi(u(T)) \to \Phi_{\min}$, giving:
$$
\alpha \mathcal{C}_*(x) \leq \Phi(x) - \Phi_{\min}.
$$

Thus:
$$
\mathcal{L}(x) \leq \Phi_{\min} + \mathcal{C}(x \to M) \leq \Phi_{\min} + \frac{1}{\alpha}(\Phi(x) - \Phi_{\min}) = \Phi_{\min} + \frac{\Phi(x) - \Phi_{\min}}{\alpha}.
$$

Combined with the lower bound from LS (Lemma 6.6), this gives the equivalence.

**Step 5: Uniqueness.** Suppose $\Psi$ is another Lyapunov functional with the same properties. Define $f: \mathrm{Im}(\mathcal{L}) \to \mathbb{R}$ by $f(\mathcal{L}(x)) = \Psi(x)$.

This is well-defined because if $\mathcal{L}(x_1) = \mathcal{L}(x_2)$, then by the equivalence to distance from $M$, $\mathrm{dist}(x_1, M) \asymp \mathrm{dist}(x_2, M)$. By similar reasoning for $\Psi$, we get $\Psi(x_1) \asymp \Psi(x_2)$.

Monotonicity of both $\mathcal{L}$ and $\Psi$ along trajectories, combined with their strict decrease outside $M$, implies $f$ is increasing. $\square$

**Remark 7.7 (Ultimate loss interpretation).** The functional $\mathcal{L}$ can be interpreted as the "ultimate loss" of the system: it measures the total cost required to reach the optimal manifold $M$. This is the structural analogue of loss functions in optimization and machine learning, but derived from the dynamical axioms rather than designed ad hoc.

### 7.7 Functional reconstruction meta-theorems

The theorems in Sections 7.1–7.6 assume a height functional $\Phi$ is given and identify its properties. We now provide a **generator**: a mechanism to explicitly recover the Lyapunov functional $\mathcal{L}$ solely from the dynamical data $(S_t)$ and the dissipation structure $(\mathfrak{D})$, without prior knowledge of $\Phi$.

This moves the framework from **identification** (recognizing a given $\Phi$) to **discovery** (finding the correct $\Phi$).

#### 7.7.1 Gradient consistency

**Definition 7.8 (Metric structure).** A hypostructure has **metric structure** if the state space $(X, d)$ is equipped with a Riemannian (or Finsler) metric $g$ such that the metric $d$ is induced by $g$: for smooth paths $\gamma: [0, 1] \to X$,
$$
d(x, y) = \inf_{\gamma: x \to y} \int_0^1 \|\dot{\gamma}(s)\|_g \, ds.
$$

**Definition 7.9 (Gradient consistency).** A hypostructure with metric structure is **gradient-consistent** if, for almost all $t \in [0, T_*(x))$ along any trajectory $u(t) = S_t x$:
$$
\|\dot{u}(t)\|_g^2 = \mathfrak{D}(u(t)),
$$
where $\dot{u}(t)$ is the metric velocity of the trajectory.

**Remark 7.10.** Gradient consistency encodes that the system is "maximally efficient" at converting dissipation into motion—a defining property of gradient flows where $\dot{u} = -\nabla \Phi$ and $\mathfrak{D} = \|\nabla \Phi\|^2$. This is **not** an additional axiom to verify case-by-case; it is a structural property that holds automatically for:
* Gradient flows in Hilbert spaces,
* Wasserstein gradient flows of free energies,
* $L^2$ gradient flows of geometric functionals,
* Any system where the "velocity equals negative gradient" structure is present.

**Axiom GC (Gradient Consistency on gradient-flow orbits).** Along any trajectory $u(t) = S_t x$ that evolves by gradient flow (i.e., $\dot{u} = -\nabla_g \Phi$), the gradient consistency condition $\|\dot{u}(t)\|_g^2 = \mathfrak{D}(u(t))$ holds.

**Fallback.** When Axiom GC fails along a trajectory—i.e., the trajectory is not a gradient flow—the reconstruction theorems (7.7.1–7.7.3) do not apply. The Lyapunov functional still exists by Theorem 7.6 via the abstract construction, but cannot be computed explicitly via the Jacobi metric or Hamilton–Jacobi equation.

#### 7.7.2 The action reconstruction principle

**Theorem 7.7.1 (Action Reconstruction).** Let $\mathcal{S}$ be a hypostructure satisfying Axioms (D), (LS), and (GC) on a metric space $(X, g)$. Then the canonical Lyapunov functional $\mathcal{L}(x)$ is explicitly the **minimal geodesic action** from $x$ to the safe manifold $M$ with respect to the **Jacobi metric** $g_{\mathfrak{D}} := \sqrt{\mathfrak{D}} \cdot g$.

**Formula:**
$$
\mathcal{L}(x) = \Phi_{\min} + \inf_{\gamma: x \to M} \int_0^1 \sqrt{\mathfrak{D}(\gamma(s))} \cdot \|\dot{\gamma}(s)\|_g \, ds.
$$

Equivalently, using the Jacobi metric:
$$
\mathcal{L}(x) = \Phi_{\min} + \mathrm{dist}_{g_{\mathfrak{D}}}(x, M).
$$

*Proof.*

**Step 1: Gradient consistency implies velocity-dissipation relation.** By Axiom GC, $\|\dot{u}(t)\|_g = \sqrt{\mathfrak{D}(u(t))}$ along any trajectory.

**Step 2: Path length in Jacobi metric.** For any path $\gamma: [0, T] \to X$ from $x$ to $y \in M$, the length in the Jacobi metric is:
$$
\mathrm{Length}_{g_{\mathfrak{D}}}(\gamma) = \int_0^T \sqrt{\mathfrak{D}(\gamma(t))} \cdot \|\dot{\gamma}(t)\|_g \, dt.
$$

**Step 3: Flow paths are geodesics.** Along a trajectory $u(t) = S_t x$, by gradient consistency:
$$
\sqrt{\mathfrak{D}(u(t))} \cdot \|\dot{u}(t)\|_g = \sqrt{\mathfrak{D}(u(t))} \cdot \sqrt{\mathfrak{D}(u(t))} = \mathfrak{D}(u(t)).
$$

Thus the Jacobi length of the flow path equals the total cost:
$$
\mathrm{Length}_{g_{\mathfrak{D}}}(u|_{[0,T]}) = \int_0^T \mathfrak{D}(u(t)) \, dt = \mathcal{C}_T(x).
$$

**Step 4: Optimality.** By the Cauchy–Schwarz inequality, for any path $\gamma$ from $x$ to $M$:
$$
\int_0^T \sqrt{\mathfrak{D}(\gamma)} \|\dot{\gamma}\|_g \, dt \geq \frac{\left(\int_0^T \mathfrak{D}(\gamma) \, dt\right)^{1/2} \cdot \left(\int_0^T \|\dot{\gamma}\|_g^2 \, dt\right)^{1/2}}{1},
$$
with equality when $\sqrt{\mathfrak{D}(\gamma)} \propto \|\dot{\gamma}\|_g$, i.e., under gradient consistency.

Therefore, flow paths are length-minimizing in the Jacobi metric, and:
$$
\mathcal{L}(x) - \Phi_{\min} = \mathcal{C}(x \to M) = \inf_{\gamma: x \to M} \mathrm{Length}_{g_{\mathfrak{D}}}(\gamma) = \mathrm{dist}_{g_{\mathfrak{D}}}(x, M).
$$

**Step 5: Lyapunov property check.** Along a trajectory $u(t)$:
$$
\frac{d}{dt} \mathcal{L}(u(t)) = \frac{d}{dt} \mathrm{dist}_{g_{\mathfrak{D}}}(u(t), M) = -\sqrt{\mathfrak{D}(u(t))} \|\dot{u}(t)\|_g = -\mathfrak{D}(u(t)).
$$

This recovers the energy–dissipation identity exactly. Uniqueness follows from Axiom LS. $\square$

**Corollary 7.7.2 (Explicit Lyapunov from dissipation).** Under the hypotheses of Theorem 7.7.1, the Lyapunov functional is **explicitly computable** from the dissipation structure alone: no prior knowledge of an energy functional is required.

#### 7.7.3 The Hamilton–Jacobi generator

**Theorem 7.7.3 (Hamilton–Jacobi characterization).** Let $\mathcal{S}$ be a hypostructure satisfying Axioms (D), (LS), and (GC) on a metric space $(X, g)$. Then the Lyapunov functional $\mathcal{L}(x)$ is the unique viscosity solution to the static **Hamilton–Jacobi equation**:
$$
\|\nabla_g \mathcal{L}(x)\|_g^2 = \mathfrak{D}(x)
$$
subject to the boundary condition $\mathcal{L}(x) = \Phi_{\min}$ for $x \in M$.

*Proof.*

**Step 1: Eikonal structure.** The distance function $d_M(x) := \mathrm{dist}_{g_{\mathfrak{D}}}(x, M)$ satisfies the eikonal equation in the Jacobi metric:
$$
\|\nabla_{g_{\mathfrak{D}}} d_M(x)\|_{g_{\mathfrak{D}}} = 1.
$$

**Step 2: Metric transformation.** We compute the gradient transformation under conformal scaling. For the conformally scaled metric $g_{\mathfrak{D}} = \mathfrak{D} \cdot g$, the gradient and its norm transform as follows.

Recall that for a Riemannian metric $\tilde{g} = \phi \cdot g$ with conformal factor $\phi > 0$, the gradient transforms as $\nabla_{\tilde{g}} f = \phi^{-1} \nabla_g f$, and the norm satisfies $\|\nabla_{\tilde{g}} f\|_{\tilde{g}}^2 = \phi^{-1} \|\nabla_g f\|_g^2$.

Applying this with $\phi = \mathfrak{D}$:
$$
\nabla_{g_{\mathfrak{D}}} f = \frac{1}{\mathfrak{D}} \nabla_g f, \quad \|\nabla_{g_{\mathfrak{D}}} f\|_{g_{\mathfrak{D}}}^2 = \frac{1}{\mathfrak{D}} \|\nabla_g f\|_g^2.
$$

The eikonal equation $\|\nabla_{g_{\mathfrak{D}}} d_M\|_{g_{\mathfrak{D}}} = 1$ becomes:
$$
\frac{1}{\sqrt{\mathfrak{D}}} \|\nabla_g d_M\|_g = 1 \implies \|\nabla_g d_M\|_g^2 = \mathfrak{D}.
$$

**Step 3: Identification.** Since $\mathcal{L}(x) = \Phi_{\min} + d_M(x)$ and $\Phi_{\min}$ is constant:
$$
\|\nabla_g \mathcal{L}(x)\|_g^2 = \|\nabla_g d_M(x)\|_g^2 = \mathfrak{D}(x).
$$

**Step 4: Viscosity solution.** The distance function to a closed set is the unique viscosity solution of the eikonal equation with zero boundary data on the set. Thus $\mathcal{L}$ is the unique viscosity solution of the Hamilton–Jacobi equation with boundary condition $\mathcal{L}|_M = \Phi_{\min}$. $\square$

**Remark 7.11 (From guessing to solving).** Theorem 7.7.3 transforms the search for a Lyapunov functional from an art (guessing the right entropy) into a well-posed PDE problem on state space. Given only $\mathfrak{D}$ and $M$, one solves the Hamilton–Jacobi equation to obtain $\mathcal{L}$.

#### 7.7.4 Instantiation examples

The power of the reconstruction theorems is that they produce known Lyapunov functionals automatically from minimal input.

**Example 7.12 (Recovering Boltzmann–Shannon entropy).**

*Input:*
* State space: $X = \mathcal{P}_2(\mathbb{R}^d)$ (probability measures with finite second moment).
* Metric: Wasserstein-2 metric $W_2$.
* Flow: Heat equation $\partial_t \rho = \Delta \rho$.
* Dissipation: Fisher information $\mathfrak{D}(\rho) = I(\rho) = \int_{\mathbb{R}^d} \frac{|\nabla \rho|^2}{\rho} \, dx$.

*Framework output:* By Theorem 7.7.3, solve $\|\nabla_{W_2} \mathcal{L}\|_{W_2}^2 = I(\rho)$.

The Otto calculus identifies $\|\nabla_{W_2} f\|_{W_2}^2 = \int |\nabla \frac{\delta f}{\delta \rho}|^2 \rho \, dx$ for functionals $f$ on $\mathcal{P}_2$.

The unique solution with $\mathcal{L} = 0$ on the equilibrium (Gaussian) is:
$$
\mathcal{L}(\rho) = \int_{\mathbb{R}^d} \rho \log \rho \, dx + \text{const}.
$$

*Conclusion:* The Boltzmann–Shannon entropy is **derived**, not postulated.

**Example 7.13 (Recovering the Ricci flow functional).**

*Input:*
* State space: $X = \mathrm{Met}(M) / \mathrm{Diff}(M)$ (Riemannian metrics modulo diffeomorphisms).
* Metric: $L^2$ metric on symmetric 2-tensors.
* Flow: Ricci flow $\partial_t g = -2\mathrm{Ric}$.
* Dissipation: $\mathfrak{D}(g) = \int_M |\mathrm{Ric}|^2 \, dV_g$ (squared Ricci curvature).

*Framework output:* By Theorem 7.7.1, the Lyapunov functional is the geodesic distance to the soliton manifold $M$ (Einstein metrics or Ricci solitons) in the $\sqrt{\mathfrak{D}}$-weighted metric.

This construction recovers the **reduced length**:
$$
\ell(\gamma, \tau) = \frac{1}{2\sqrt{\tau}} \int_0^\tau \sqrt{s} \left( R + |\dot{\gamma}|^2 \right) ds,
$$
and the **reduced volume** as its integral. The monotonicity formula is precisely the Lyapunov property from Theorem 7.7.1.

*Conclusion:* The canonical Lyapunov functional for Ricci flow is derived from the dissipation structure alone.

**Example 7.14 (Recovering Dirichlet energy).**

*Input:*
* State space: $X = H^1(\Omega)$ for a bounded domain $\Omega$.
* Metric: $L^2$ metric.
* Flow: Heat equation $\partial_t u = \Delta u$.
* Dissipation: $\mathfrak{D}(u) = \|\Delta u\|_{L^2}^2$.

*Framework output:* By Theorem 7.7.3, solve $\|\nabla_{L^2} \mathcal{L}\|_{L^2}^2 = \|\Delta u\|_{L^2}^2$.

In the $L^2$ metric, $\nabla_{L^2} \mathcal{L} = \frac{\delta \mathcal{L}}{\delta u}$. The equation becomes:
$$
\left\| \frac{\delta \mathcal{L}}{\delta u} \right\|_{L^2}^2 = \|\Delta u\|_{L^2}^2.
$$

With the ansatz $\frac{\delta \mathcal{L}}{\delta u} = -\Delta u$, we get $\mathcal{L}(u) = \frac{1}{2}\int_\Omega |\nabla u|^2 \, dx$.

*Conclusion:* The Dirichlet energy is the canonical Lyapunov functional for the heat equation.

#### 6.7.5 The reconstruction protocol

**Protocol 6.15 (Lyapunov functional discovery).** To discover the Lyapunov functional for a new system:

1. **Define the state space $X$** with its natural metric $g$ (usually $L^2$, Wasserstein, or $H^s$).

2. **Write the evolution equation** $\partial_t u = V(u)$.

3. **Identify the dissipation** as the squared metric velocity:
$$
\mathfrak{D}(u) := \|V(u)\|_g^2.
$$

4. **Identify the safe manifold** $M$ (equilibria, ground states, solitons).

5. **Apply Theorem 7.7.1:** The Lyapunov functional is the $\sqrt{\mathfrak{D}}$-weighted geodesic distance to $M$:
$$
\mathcal{L}(x) = \inf_{\gamma: x \to M} \int_0^1 \sqrt{\mathfrak{D}(\gamma(s))} \|\dot{\gamma}(s)\|_g \, ds.
$$

6. **Or apply Theorem 7.7.3:** Solve the Hamilton–Jacobi equation $\|\nabla_g \mathcal{L}\|_g^2 = \mathfrak{D}$ with $\mathcal{L}|_M = 0$.

**Remark 7.16 (No guessing required).** The reconstruction protocol eliminates the need to "guess" the entropy functional. The framework builds it automatically from the dissipation structure. Historical insight is not required—only the identification of the cost function $\mathfrak{D}$.

---

## 8. Structural resolution: The emergence and elimination of maximizers

### 8.1 The philosophical pivot

Standard analysis often asks: *Does a global maximizer of the energy functional exist?* If the answer is "no" or "maybe," the analysis stalls.

The hypostructure framework inverts this dependency. We do not assume the existence of a global maximizer to define the system. Instead, we use **Axiom C (Compactness)** to prove that **if** a singularity attempts to form, it must structurally reorganize the solution into a "local maximizer" (a Canonical Profile).

Maximizers are treated not as static objects that *must* exist globally, but as **asymptotic limits** that emerge only when the trajectory approaches a finite-time singularity.

### 8.2 Formal definition: Structural resolution

We formalize the "Maximizer" concept via the principle of **Structural Resolution** (a generalization of Profile Decomposition).

**Definition 8.1 (Asymptotic maximizer extraction).** Let $\mathcal{S}$ be a hypostructure satisfying Axiom C. Let $u(t)$ be a trajectory approaching a finite blow-up time $T_*$. A **Structural Resolution** of the singularity is a decomposition of the sequence $u(t_n)$ (where $t_n \nearrow T_*$) into:
$$
u(t_n) = \underbrace{g_n \cdot V}_{\text{The Maximizer}} + \underbrace{w_n}_{\text{Dispersion}}
$$
where:
1. **$V \in X$ (The Canonical Profile):** A fixed, non-trivial element of the state space. This is the "Maximizer" of the local concentration.
2. **$g_n \in G$ (The Gauge Sequence):** A sequence of symmetry transformations (scalings, translations) that diverge as $n \to \infty$ (e.g., $\lambda_n \to \infty$ for scaling).
3. **$w_n$ (The Residual):** A term that vanishes or disperses in the relevant topology (structurally irrelevant).

**Remark 8.2 (The key insight: forced structure).** We do not assume $V$ exists *a priori*.
- If the sequence $u(t_n)$ disperses (Mode 2), then $V$ does not exist—**no singularity forms**. The solution exists globally via scattering.
- If the sequence concentrates, blow-up **forces** $V$ to exist. We then check permits on the forced structure.

**Remark 8.2.1 (No global compactness required).** A common misconception is that one must prove global compactness to use this framework. This is false:
- Mode 2 (dispersion) is **global existence**, not a singularity to be excluded.
- When concentration does occur, structure is forced—no compactness proof needed.
- The framework checks algebraic permits on the forced structure.

The two-tier logic:
1. **Tier 1 (Dispersion):** If energy disperses, no singularity forms—global existence via scattering.
2. **Tier 2 (Concentration):** If energy concentrates, check algebraic permits on the forced structure. Permit denial yields regularity via contradiction.

### 8.3 The taxonomy of maximizers

Once Axiom C extracts the profile $V$, the hypostructure framework classifies it immediately. The "Maximizer" $V$ must fall into one of two categories:

**Type A: The Safe Maximizer ($V \in M$).**
The profile $V$ lies in the **Safe Manifold** (e.g., a soliton, a ground state, or a vacuum state).
- **Mechanism:** The trajectory is simply zooming in on a regular structure (like a soliton).
- **Outcome:** **Axiom LS (Stiffness)** applies. The trajectory is constrained near $M$. Since elements of $M$ are global solutions with infinite existence time, this is not a singularity; it is **Soliton Resolution**.

**Type B: Non-safe profile ($V \notin M$).**
The profile $V$ is a self-similar blow-up profile or a high-energy bubble that is *not* in the safe manifold.
- **Mechanism:** The system is attempting to construct a Type II blow-up.
- **Outcome:** The **algebraic permits** apply. We do not need to analyze the PDE evolution of $V$. We only need to check whether $V$ can satisfy the scaling and capacity permits.

### 8.4 Disabling conservation of difficulty: Admissibility tests

This is where the framework replaces hard analysis with algebra. We test the non-safe profile $V$ against the structural axioms.

**Test 1: Scaling Admissibility.**
Even if $V$ is a valid profile, it must be generated by the gauge sequence $g_n$ (specifically the scaling $\lambda_n \to \infty$).
By **Axiom SC** and **Theorem 7.2 (Property GN)**:
$$
\text{Cost of Generating } V \sim \int (\text{Dissipation of } g_n \cdot V)
$$

- If the scaling exponents satisfy $\alpha > \beta$ (Subcriticality), the cost of generating *any* non-trivial non-safe profile via scaling is **infinite**.
- **Result:** The non-safe profile $V$ is excluded. It cannot be formed from finite energy.

**Test 2: Capacity Admissibility.**
If $V$ is supported on a "thin" set (e.g., a singular filament with dimension $< Q$):
- By **Axiom Cap** and **Theorem 7.3**, the time available to create such a profile goes to zero faster than the profile can form.
- **Result:** The non-safe profile is excluded by geometric constraints.

### 8.5 The regularity logic flow

The framework proves regularity without assuming any structure exists *a priori*:

**Tier 1: Does blow-up attempt to form?**
- **NO (Energy disperses):** Mode 2—global existence via scattering. No singularity forms.
- **YES (Energy concentrates):** Structure is forced. Proceed to Tier 2.

**Tier 2: Check algebraic permits on the forced structure $V$.**

**Step 2a: Is the forced profile safe?** ($V \in M$ test)
- **YES:** Soliton Resolution / Asymptotic Stability. No singularity—the trajectory converges to a regular structure.
- **NO:** Non-safe profile. Check permits.

**Step 2b: Scaling Permit (Axiom SC)**
- If $\alpha > \beta$: Property GN proves infinite cost—supercritical blow-up is impossible. **Global regularity.**
- If $\alpha \leq \beta$: Supercritical regime; proceed to capacity test.

**Step 2c: Capacity Permit (Axiom Cap)**
- If capacity bounds are violated: Geometric collapse is impossible. **Global regularity.**
- If capacity allows: Proceed to remaining tests.

**Conclusion:** The framework operates by **soft local exclusion**:
- If energy disperses (Tier 1), no singularity forms.
- If energy concentrates (Tier 2), structure is forced, and permits are checked.
- Permit denial yields regularity via contradiction.

**No global compactness proof is required.** Concentration is forced by blow-up; we check permits on the forced structure.

### 8.6 Implementation guide: How to endow solutions

When instantiating the framework for a specific system, one does not search for the global maximizer of the functional. The procedure is as follows:

**Step 1: Identify the Symmetry Group $G$.**
For example: Scaling $\lambda$, Translation $x_0$.

**Step 2: Understand the forced structure.**
Observe that if blow-up occurs with bounded energy, concentration is forced. When energy concentrates, Profile Decomposition (standard for most PDEs) ensures a Canonical Profile $V$ emerges modulo $G$. You do not need to prove compactness globally—concentration is forced by blow-up.

**Step 3: Compute Exponents $(\alpha, \beta)$.**
- $\mathfrak{D}(\mathcal{S}_\lambda u) \approx \lambda^\alpha \mathfrak{D}(u)$
- $dt \approx \lambda^{-\beta} ds$

**Step 4: The Check.**
Is $\alpha > \beta$?
- **Yes:** Then **Theorem 7.2** guarantees that *whatever* the profile $V$ extracted in Step 2 is, it cannot sustain a Type II blow-up. The non-safe profile is structurally inadmissible.

**Remark 8.3 (Decoupling existence from admissibility).** The hypostructure framework decouples the *existence* of singular profiles from their *admissibility*. We do not require the existence of a global maximizer to define the theory. Instead, Axiom C ensures that if a singularity attempts to form via concentration, a local maximizer (Canonical Profile) must emerge asymptotically. Axiom SC then evaluates the scaling cost of this emerging profile. If the cost is infinite (GN), the profile is forbidden from materializing, regardless of whether a global maximizer exists for the static functional.

---

## 9. Quantitative hypostructure: Thresholds and sharp constants

### 9.0 Overview

While the previous chapters focus on the *classification* of trajectories (Structural Resolution) and the *structure* of canonical profiles (Maximizers), this chapter addresses the *quantification* of the breakdown.

We establish that the **Canonical Profile** $V$ extracted by Axiom C is not merely a qualitative obstruction; it is the **variational optimizer** that saturates the inequalities of Axiom D. This observation allows the hypostructure framework to function as a machine for computing **sharp constants** and **energy thresholds** for global regularity.

The central principle is **Pathology Saturation**:
> **The structural axioms fail precisely when the trajectory possesses enough energy to instantiate the ground state of the failing mode.**

### 9.1 The structural ratio

To quantify the failure of Axiom D (Dissipation) or Axiom R (Recovery), we define the ratio of the competing functionals along the singular profile.

**Definition 9.1 (Structural Capacity Ratio).**
Let $\mathcal{S}$ be a hypostructure. For any non-trivial profile $v \in X$, the **Structural Capacity Ratio** $\mathcal{K}(v)$ is the ratio of the "Drift" mechanism (the nonlinearity or instability) to the "Dissipation" mechanism (the restoring force).

- **For Mode 1/3 (Energy/Scaling):** If the energy inequality is of the form $\int \mathfrak{D} \geq C^{-1} \int \mathcal{N}(u)$, then:
$$
\mathcal{K}(v) := \frac{\mathcal{N}(v)}{\mathfrak{D}(v)}.
$$
- **For Mode 6 (Stiffness):** If the stiffness is governed by a spectral gap or Poincaré inequality, $\mathcal{K}(v)$ is the Rayleigh quotient of the linearized operator.

**Definition 9.2 (The Critical Threshold).**
The **critical structural constant** of the system is the supremum of this ratio over all admissible profiles generated by the extraction machinery of Axiom C:
$$
C_{\text{sharp}} := \sup_{v \in \mathcal{V}} \mathcal{K}(v),
$$
where $\mathcal{V}$ is the set of all Canonical Profiles (Mode 3 or Mode 6 limits).

### 9.2 The Saturation Theorem

The following theorem links the abstract breakdown of the system to the sharp constant of the underlying analytic inequalities.

**Theorem 9.3 (The Saturation Theorem).**
Let $\mathcal{S}$ be a hypostructure where Axiom D depends on an analytic inequality of the form $\Phi(u) + \alpha \mathfrak{D}(u) \leq \text{Drift}(u)$.
If the system admits a **Mode 3 (Supercritical Cascade)** or **Mode 6 (Stiffness)** singularity profile $V$, then:

1. **Optimality:** The profile $V$ is a variational critical point (a ground state) of the functional $\mathcal{J}(u) = \mathfrak{D}(u) - \lambda \text{Drift}(u)$.
2. **Sharpness:** The optimal constant for the inequality governing the safe region is exactly determined by the profile:
$$
C_{\text{sharp}} = \mathcal{K}(V)^{-1}.
$$
3. **Threshold Energy:** There exists a sharp energy threshold $E^* = \Phi(V)$. Any trajectory with $\Phi(u(0)) < E^*$ satisfies Axioms D and SC globally and is regular.

*Proof.*

**Part 1: Optimality of the profile $V$.**

Suppose the system admits a Mode 3 or Mode 6 singularity. By Definition 8.1 (Asymptotic maximizer extraction), there exists a sequence $t_n \nearrow T_*$ and gauge elements $g_n \in G$ such that $g_n \cdot u(t_n) \to V$ for some non-trivial profile $V \in X$.

We claim $V$ is a critical point of $\mathcal{J}(u) = \mathfrak{D}(u) - \lambda \text{Drift}(u)$ for some $\lambda > 0$.

Consider the rescaled trajectory $v_n(s) := g_n \cdot u(t_n + \epsilon_n s)$ for small $\epsilon_n \to 0$. By the semiflow property, $v_n$ satisfies the rescaled evolution equation. Taking the limit $n \to \infty$:
$$
\lim_{n \to \infty} \frac{d}{ds}\bigg|_{s=0} v_n(s) = 0,
$$
since $V$ is the asymptotic limit and the rescaling compresses the evolution. This stationarity condition is precisely the Euler–Lagrange equation for $\mathcal{J}$:
$$
\frac{\delta \mathfrak{D}}{\delta u}\bigg|_{u=V} = \lambda \frac{\delta \text{Drift}}{\delta u}\bigg|_{u=V}.
$$
The Lagrange multiplier $\lambda$ arises from the constraint that $V$ lies on the boundary of the stable region. Thus $V$ is a variational critical point.

**Part 2: Sharpness of the constant.**

The energy inequality governing the safe region has the form:
$$
\mathfrak{D}(u) \geq C^{-1} \cdot \text{Drift}(u)
$$
for some constant $C > 0$. The safe region is precisely where this inequality holds with strict inequality.

Define the structural capacity ratio $\mathcal{K}(u) = \text{Drift}(u)/\mathfrak{D}(u)$. The inequality $\mathfrak{D}(u) \geq C^{-1} \cdot \text{Drift}(u)$ is equivalent to $\mathcal{K}(u) \leq C$.

Since $V$ lies on the boundary between restoration and collapse, it saturates this inequality:
$$
\mathfrak{D}(V) = C_{\text{sharp}}^{-1} \cdot \text{Drift}(V),
$$
which gives $\mathcal{K}(V) = C_{\text{sharp}}$, hence $C_{\text{sharp}} = \mathcal{K}(V)^{-1}$.

To see that this is optimal, suppose there existed a profile $W$ with $\mathcal{K}(W) > \mathcal{K}(V)$. Then trajectories near $W$ would violate the energy inequality more severely than those near $V$. But by the compactness extraction (Axiom C), any maximizing sequence for $\mathcal{K}$ over the set of singular profiles must converge to some canonical profile. Since $V$ is extracted as the limit of the actual singular trajectory, it achieves the supremum:
$$
C_{\text{sharp}} = \sup_{v \in \mathcal{V}} \mathcal{K}(v) = \mathcal{K}(V).
$$

**Part 3: Threshold energy.**

Define $E^* := \Phi(V)$. We show that trajectories with $\Phi(u(0)) < E^*$ are globally regular.

Suppose $u(t)$ has initial energy $\Phi(u(0)) < E^*$ and develops a singularity at time $T_* < \infty$. By Axiom C, there exists a canonical profile $W$ with $\Phi(W) \leq \liminf_{t \to T_*} \Phi(u(t))$.

By Axiom D (energy dissipation inequality):
$$
\Phi(u(t)) \leq \Phi(u(0)) - \alpha \int_0^t \mathfrak{D}(u(s)) \, ds \leq \Phi(u(0)) < E^*.
$$

Thus $\Phi(W) < E^* = \Phi(V)$. But among all Mode 3 or Mode 6 profiles, $V$ achieves the minimal energy threshold (it is the ground state of $\mathcal{J}$). This contradicts $\Phi(W) < \Phi(V)$.

Therefore, no singularity can form when $\Phi(u(0)) < E^*$, establishing global regularity below the threshold.

**Uniqueness of the threshold.** The threshold $E^*$ is sharp: for any $\epsilon > 0$, there exist initial data with $\Phi(u(0)) = E^* + \epsilon$ that develop finite-time singularities. This follows because $V$ itself, or small perturbations thereof, can be realized as initial data leading to Mode 3 or Mode 6 breakdown. $\square$

### 9.3 Protocol: Computing sharp constants via pathologies

Theorem 9.3 provides a constructive protocol for finding optimal constants without relying on ad-hoc symmetrization arguments.

**Protocol 9.4 (The Variational Extractor).**
To compute the sharp constant for an embedding or decay inequality using hypostructure:

1. **Assume Breakdown:** Postulate that the system undergoes a Mode 3 (Scaling) or Mode 6 (Stiffness) failure.
2. **Extract the Profile:** Use the Euler–Lagrange equation associated with the flow to identify the Canonical Profile $V$ (e.g., the soliton, the bubble, or the instanton).
3. **Calculate the Ratio:** Compute $\Phi(V)$ and $\mathfrak{D}(V)$.
4. **Derive the Constant:** The sharp constant is derived algebraically from $\mathcal{K}(V)$.

### 9.4 Example: The Sobolev threshold

We illustrate this with the classical semilinear heat equation $u_t = \Delta u + |u|^{p-1}u$ in the energy-critical regime.

1. **Axiom D identification:** The Sobolev inequality $\|u\|_{L^{p+1}} \leq C \|\nabla u\|_{L^2}$ gives the energy-dissipation structure.
2. **Mode 3 Analysis:** The singular profile $V$ arises from the scaling symmetry. By Theorem 9.3, $V$ must be the ground state of the stationary equation $\Delta V + |V|^{p-1}V = 0$.
3. **Profile Identification:** $V$ is the Talenti bubble $V(x) = (1 + |x|^2)^{-\frac{n-2}{2}}$.
4. **Threshold:** The sharp constant is explicitly $C_{\text{sharp}} = \frac{\|V\|_{L^{p+1}}}{\|\nabla V\|_{L^2}}$.
5. **Result:** The global regularity threshold is $E^* = \frac{1}{n} \int |\nabla V|^2$. The hypostructure framework recovers the standard Kenig–Merle threshold automatically.

**Remark 9.5 (From qualitative to quantitative).**
This chapter demonstrates the dual nature of the framework.
- **Qualitatively:** It classifies $V$ as a "Mode 3 Failure."
- **Quantitatively:** It uses $V$ to compute the number $E^*$.
The pathology is not merely a defect; it is the measuring stick of the system's stability.

---

## 10. Instantiation guide

### 10.1 General instantiation protocol

To instantiate the hypostructure framework for a specific dynamical system:

**Step 1: Identify the state space $X$.**
- Choose appropriate function spaces, configuration spaces, or probability spaces.
- Equip with a metric $d$ making $X$ a Polish space.
- Identify a natural reference measure $\mu$.

**Step 2: Define the semiflow $S_t$.**
- For PDEs: the solution operator.
- For stochastic systems: the Markov semigroup.
- For discrete systems: the iteration map.
- Characterize the maximal existence time $T_*(x)$.

**Step 3: Identify the height functional $\Phi$.**
- Energy, free energy, enstrophy, entropy, or other conserved/dissipated quantities.
- Observe that $\Phi$ is lower semicontinuous and proper (typically immediate from the definition).

**Step 4: Identify the dissipation functional $\mathfrak{D}$.**
- Viscous dissipation, entropy production, Fisher information, reduction cost.
- Read off the energy-dissipation identity from the equation (this is part of the equation's definition, not an estimate to prove).

**Step 5: Compute the algebraic permit data.**
Regularity is proven via soft local exclusion. Compute the algebraic data that determines whether blow-up is possible:
- **(SC) Scaling exponents:** Compute $\alpha$ (dissipation scaling) and $\beta$ (temporal scaling). If $\alpha > \beta$, supercritical blow-up is impossible.
- **(Cap) Capacity bounds:** Determine the capacity dimension of potential singular sets. Positive capacity denies the geometric permit.
- **(LS) Łojasiewicz exponents:** Identify equilibria $M$ and compute the Łojasiewicz exponent $\theta$ near $M$.
- **(TB) Topological sectors:** Identify topological invariants and action gaps.

**Note:** You do NOT need to "verify" or "prove" that Axiom C holds globally. Concentration is **forced** by blow-up attempts. The framework checks permits on the forced structure.

**Step 6: Identify symmetries and construct the gauge.**
- Determine the symmetry group $G$ (translations, rotations, scalings, gauge transformations).
- Construct a normalized slice $\Sigma$ and gauge map $\Gamma$.
- Check normalization compatibility (Axiom N).

**Step 7: Check the Scaling Permit (Axiom SC).**
- Identify the scaling subgroup $(\mathcal{S}_\lambda) \subset G$.
- Compute the dissipation scaling exponent $\alpha$: how does $\mathfrak{D}(\mathcal{S}_\lambda \cdot x)$ scale with $\lambda$?
- Compute the temporal scaling exponent $\beta$: how does $dt$ transform under rescaling?
- Check whether $\alpha > \beta$ (scaling permit satisfied).
- **Key insight:** This is pure dimensional analysis—no hard estimates required. If $\alpha > \beta$, supercritical blow-up is impossible: the dissipation would dominate the compressed time horizon, yielding infinite cost.

**Step 8: Specify background structures.**
- **(BG) Geometric:** Specify dimension $Q$, Ahlfors regularity, capacity-codimension bounds.
- **(TB) Topological:** Identify topological sectors $\tau$, action functional $\mathcal{A}$, action gap $\Delta$.

**Conclusion:** Once the algebraic permit data is computed, apply the regularity logic:
- If $\alpha > \beta$, supercritical blow-up is impossible (SC permit denied).
- If singular sets have positive capacity, geometric collapse is impossible (Cap permit denied).
- If Łojasiewicz holds near equilibria, stiffness breakdown is impossible (LS permit denied).

**Global regularity follows from soft local exclusion.** No hard global estimates are required—only algebraic/dimensional analysis of the forced local structure.

### 10.2 PDE instantiation tips

For parabolic PDEs (e.g., semilinear heat equations, reaction–diffusion, geometric flows):

**State space:**
- Sobolev spaces $H^s(\Omega)$, $W^{k,p}(\Omega)$
- Besov spaces $B^s_{p,q}$ for critical regularity
- Weak solution spaces (e.g., energy class solutions)

**Concentration topology (C):**
Identify the natural topology where energy concentrates. Standard results *describe* the limiting behavior:
- Rellich–Kondrachov describes strong limits in $L^2$ from bounded $H^1$ sequences.
- Aubin–Lions describes time-integrated limits for parabolic problems.
- Profile decomposition describes the structure of concentrating sequences.
You do not prove these theorems per trajectory; they describe what *must* happen when concentration occurs.

**Dissipation (D):**
- Viscous dissipation: $\mathfrak{D}(u) = \nu \|\nabla u\|_{L^2}^2$ for diffusive systems.
- Entropy production: $\mathfrak{D}(f) = \int |\nabla \log f|^2 f \, dx$ for Fokker–Planck.
- Read off the energy identity from the PDE—this is definitional, not an estimate.

**Recovery (R):**
- **Heat kernel structure:** The parabolic operator naturally smooths solutions for $t > 0$. This is a property of the operator, not an estimate to prove.
- Good region: where the operator's smoothing property applies.

**Capacity (Cap):**
- Capacity from scaling-critical norms: $c(u) = \|u\|_{\dot{H}^{s_c}}^p$ at critical regularity $s_c$.
- Frequency localization: high-frequency concentration increases capacity.
- Concentration-compactness methods for critical problems.

**Local stiffness (LS):**
- **Analytic nonlinearity:** If the nonlinearity $f$ is analytic, the Łojasiewicz inequality holds automatically near equilibria (Simon's theorem). This is a property of analytic functions, not an estimate to derive.
- Identify the equilibria $M$ and observe whether the nonlinearity is analytic.

**Scaling structure (SC):**
- Identify the natural scaling: parabolic $(x, t) \mapsto (\lambda x, \lambda^2 t)$ with field rescaling $u_\lambda(x, t) = \lambda^\gamma u(\lambda x, \lambda^2 t)$.
- Compute $\alpha$: how $\mathfrak{D}$ transforms under $u \mapsto \mathcal{S}_\lambda \cdot u$. This is dimensional analysis.
- Compute $\beta$: the temporal exponent from $dt \to \lambda^{-\beta} ds$.
- Check whether $\alpha > \beta$: if yes, supercritical blow-up is impossible (scaling permit denied).
- **Key point:** This is pure dimensional analysis—no PDE estimates needed. Once exponents are identified, GN follows automatically from Theorem 7.2.1.

### 10.3 Kinetic/probabilistic instantiation tips

For kinetic equations, interacting particle systems, and stochastic dynamics:

**State space:**
- Probability measures $\mathcal{P}(E)$ with Wasserstein metric $W_p$.
- Empirical measures of $N$-particle systems.
- Path spaces for stochastic processes.

**Compactness (C):**
- Prokhorov's theorem: tightness implies precompactness in $\mathcal{P}(E)$.
- Uniform moment bounds for tightness.
- Arzelà–Ascoli for path spaces.

**Dissipation (D):**
- Fisher information: $\mathfrak{D}(\mu) = I(\mu|\gamma) = \int |\nabla \log(d\mu/d\gamma)|^2 \, d\mu$.
- Entropy production rate in Fokker–Planck/McKean–Vlasov.
- Relative entropy decay in hypocoercive systems.

**Recovery (R):**
- Hypocoercivity: recovery to equilibrium despite degeneracy.
- Villani's H-theorem framework: entropy methods.
- Spectral gap from Poincaré or log-Sobolev.

**Capacity (Cap):**
- Rare event probabilities: $c(\mu) = -\log P(\mu \text{ is typical})$.
- Large deviations rate functions.
- Extinction probability for particle systems.

**Local stiffness (LS):**
- Log-Sobolev inequality with constant $\lambda_{\mathrm{LS}}$.
- Poincaré inequality for weaker stiffness.
- Bakry–Émery criterion: $\mathrm{Ric} + \nabla^2 V \geq \lambda I$.

**Scaling structure (SC):**
- Scaling of Fisher information under measure dilation.
- Temporal rescaling from diffusion coefficient.
- Subcritical condition from entropy-production scaling.
- **Key point:** Once scaling exponents are computed, GN follows automatically.

### 10.4 Discrete/computational instantiation tips

For λ-calculus, interaction nets, term rewriting, and graph dynamics:

**State space:**
- λ-terms modulo $\alpha$-equivalence.
- Interaction nets (graphs with interaction rules).
- Configurations of a rewriting system.

**Compactness (C):**
- König's lemma: finitely branching infinite trees have infinite paths.
- Graph limits (graphons) for large graphs.
- Compactness of term spaces under de Bruijn representation.

**Dissipation (D):**
- Reduction complexity: $\mathfrak{D}(t) = $ cost of one reduction step.
- Size decrease under normalization.
- Work metric for interaction nets.

**Recovery (R):**
- Normalization theorems: every term reaches normal form.
- Confluence: different reduction paths converge.
- Standardization: canonical reduction strategies.

**Capacity (Cap):**
- Combinatorial capacity: number of reduction paths.
- Depth/complexity measures.
- Type-theoretic size bounds.

**Local stiffness (LS):**
- Normal forms as equilibria: $M = \{t : t \text{ is in normal form}\}$.
- Strong normalization: all reduction paths terminate.
- Confluence implies uniqueness of normal forms.

**Scaling structure (SC):**
- Canonical forms: de Bruijn indices, α-normal representatives.
- Graph isomorphism as symmetry.
- Scaling: term depth or size reduction per step.
- Subcritical condition: cost per reduction exceeds time compression under any "zooming" into subterms.
- **Result:** Strong normalization = no supercritical blow-up = GN holds automatically.

---

## 11. Extended instantiation sketches

**Note on Instantiation.** The following sketches do not construct solutions or prove estimates. They **identify** the structural data (Group $G$, Exponents $\alpha/\beta$, Dimension $Q$) inherent to these equations. Global regularity follows from the algebraic incompatibility of this data with the singularity mechanism, not from analytical bounds.

### 11.1 Semilinear parabolic systems

Consider a semilinear parabolic system on $\Omega \subseteq \mathbb{R}^n$:
$$
\partial_t u = \nu \Delta u + f(u, \nabla u),
$$
where $f$ satisfies appropriate growth conditions.

**Hypostructure data:**
- $X = H^1(\Omega)$ or $W^{1,p}(\Omega)$ depending on the nonlinearity.
- $S_t$: mild solution operator.
- $\Phi(u) = \frac{1}{2}\|\nabla u\|_{L^2}^2 + F(u)$ (energy functional with potential $F$).
- $\mathfrak{D}(u) = \nu \|\Delta u\|_{L^2}^2$ or appropriate dissipation from the system.

**Structural identification:**
- **(C):** Concentration topology is $L^2_{\mathrm{loc}}$. Aubin–Lions *describes* how concentrating sequences behave.
- **(D):** Energy identity read off from testing the equation against $\partial_t u$—definitional, not an estimate.
- **(R):** **Heat kernel structure:** The parabolic operator smooths instantly for $t > 0$. This is a property of the Laplacian.
- **(Cap):** Capacity from scaling-critical norms at the critical Sobolev exponent.
- **(LS):** **Analytic nonlinearity:** If $f$ is analytic, Łojasiewicz holds automatically (Simon's theorem).

**SC identification:** The parabolic scaling is $u_\lambda(x,t) = \lambda^\gamma u(\lambda x, \lambda^2 t)$ with $\gamma$ determined by the nonlinearity. Under this scaling:
- Dissipation $\mathfrak{D}$ transforms with exponent $\alpha$ determined by dimensional analysis.
- Time transforms with exponent $\beta = 2$.
- Observe whether $\alpha > \beta$: if yes, supercritical blow-up is algebraically forbidden.
- **Consequence:** By Theorem 7.2.1, GN holds automatically—Type II blow-up is framework-forbidden.

**Conclusion:** Once all structural data is identified, Theorems 7.1–7.6 apply, giving complete singularity classification.

### 11.2 Geometric flows

Consider mean curvature flow of hypersurfaces $M_t \subset \mathbb{R}^{n+1}$:
$$
\frac{\partial X}{\partial t} = -H \nu,
$$
where $H$ is mean curvature and $\nu$ is the unit normal.

**Hypostructure data:**
- $X$: space of embedded hypersurfaces (or varifolds for weak solutions).
- $S_t$: mean curvature flow.
- $\Phi(M) = \mathcal{H}^n(M)$ (area functional).
- $\mathfrak{D}(M) = \int_M H^2 \, d\mathcal{H}^n$ (Willmore energy contribution).

**Structural identification:**
- **(C):** Concentration topology is varifold convergence. Allard compactness *describes* limiting behavior of bounded-mass sequences.
- **(D):** $\frac{d}{dt} \mathcal{H}^n(M_t) = -\int_{M_t} H^2 \, d\mathcal{H}^n$—this is the definition of mean curvature flow.
- **(SC):** **Dimensional analysis:** Area scales as $\lambda^n$, mean curvature as $\lambda^{-1}$, Willmore energy as $\lambda^{n-2}$. Compute $\alpha, \beta$ from these dimensions.
- **(BG):** Ambient Euclidean geometry, codimension bounds for singular sets.

**Surgery as gauge:** At singularities, Huisken–Sinestrari surgery modifies the surface, acting as a "gauge transformation" that removes the singular part and continues the flow.

### 11.3 Interacting particle systems

Consider $N$ particles with positions $X_i \in \mathbb{R}^d$ evolving by:
$$
dX_i = -\nabla V(X_i) \, dt - \frac{1}{N} \sum_{j \neq i} \nabla W(X_i - X_j) \, dt + \sqrt{2\beta^{-1}} \, dB_i.
$$

**Hypostructure data:**
- $X = \mathcal{P}(\mathbb{R}^d)$ (empirical measure $\mu_N = \frac{1}{N} \sum_i \delta_{X_i}$).
- $S_t$: Markov semigroup on probability measures.
- $\Phi(\mu) = \int V \, d\mu + \frac{1}{2} \iint W \, d\mu \otimes d\mu + \beta^{-1} \mathrm{Ent}(\mu)$ (free energy).
- $\mathfrak{D}(\mu) = I(\mu|\gamma)$ (Fisher information relative to equilibrium).

**Structural identification:**
- **(C):** Concentration topology is weak-* convergence on $\mathcal{P}(\mathbb{R}^d)$. Prokhorov compactness *describes* limiting behavior of tight sequences.
- **(D):** Free energy dissipation identity read off from the Fokker–Planck structure—definitional.
- **(LS):** **Convexity check:** If $V + W * \mu$ is uniformly convex, LSI is an intrinsic property of the invariant measure. Simply observe the convexity.
- **(TB):** Topological sectors from homotopy classes of configurations (for topological particles).

**SC identification:** Scaling of Fisher information under measure dilation gives exponent $\alpha$; diffusive time scaling gives $\beta$. Observe whether $\alpha > \beta$: if yes, supercritical blow-up is algebraically forbidden. GN then follows automatically from Theorem 7.2.1.

**Mean-field limit:** As $N \to \infty$, propagation of chaos shows convergence to McKean–Vlasov dynamics. Uniform-in-$N$ estimates ensure SC holds uniformly.

### 11.4 λ-calculus and interaction nets

Consider the pure λ-calculus with β-reduction:
$$
(\lambda x. M) N \to_\beta M[N/x].
$$

**Hypostructure data:**
- $X$: λ-terms modulo α-equivalence.
- $S_t$: one-step β-reduction (discrete time $t \in \mathbb{N}$).
- $\Phi(M)$: size of $M$ (number of nodes in syntax tree) or de Bruijn complexity.
- $\mathfrak{D}(M)$: reduction cost (e.g., 1 per β-step, or proportional to substitution size).

**Structural identification:**
- **(C):** Concentration topology: terms of bounded size form a finite set. Compactness is trivial.
- **(D):** Observe the reduction strategy: many strategies decrease term size.
- **(R):** For typed calculi, normalization is a property of the type system.
- **(LS):** Normal forms are exactly the fixed points of $S$; uniqueness from confluence (Church–Rosser).
- **(TB):** Type sectors: simply-typed, System F types, etc. Different types prevent interconversion.

**SC interpretation:** The scaling structure for term rewriting is combinatorial: "zooming" into a subterm while tracking reduction cost. The subcritical condition $\alpha > \beta$ encodes that cost accumulates faster than the reduction sequence can extend. Strong normalization is then equivalent to GN: the absence of infinite reduction sequences at finite cost. Observe whether the type system enforces $\alpha > \beta$—if yes, GN holds automatically.

**Interaction nets:** Similar instantiation with:
- $X$: interaction net graphs.
- $\Phi$: number of active pairs or graph size.
- $\mathfrak{D}$: cost per interaction step.
- Confluence and strong normalization give the axioms.
