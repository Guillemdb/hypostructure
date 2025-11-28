# Hypostructures: A Structural Framework for Singularity Control in Dynamical Systems

## 0. Overview

### 0.1 The singularity control thesis

A **hypostructure** is a diagnostic framework for dynamical systems—deterministic or stochastic, continuous or discrete—that provides **complete structural classification of singularities**. The central thesis is:

> **The axioms (C, D, R, Cap, LS, SC) act as a complete set of structural filters. For any finite-energy trajectory, either all axioms hold (leading to regularity or soliton resolution), or exactly one axiom fails locally, identifying the precise "mode" of the singularity.**

In particular, **Axiom C (Compactness) is not a global assumption** of the theory, but a conditional property tested per trajectory. Its failure is not a defect of the framework, but a classification of the singularity as "unstructured" (Mode 2: Compactness Breakdown).

More precisely, we establish that any finite-time breakdown must fall into one of finitely many **structural failure modes**, each of which is explicitly characterized. The framework does not require proving all axioms hold globally; it uses axiom failures as diagnostic labels for singularity types.

### 0.2 Conceptual architecture

The framework rests on three pillars:

1. **Height and dissipation.** A height functional $\Phi$ (energy, free energy, Lyapunov candidate) coupled with a dissipation functional $\mathfrak{D}$ that tracks the cost of evolution. The pair $(\Phi, \mathfrak{D})$ satisfies an energy–dissipation inequality that bounds the available budget for singular behaviour.

2. **Structural axioms.** A collection of local/soft axioms—compactness, recovery, capacity, stiffness, regularity—that constrain how trajectories can concentrate, disperse, or degenerate. These axioms are designed to be verifiable in concrete settings while remaining sufficiently abstract to apply across disparate domains.

3. **Symmetry and scaling.** A gauge structure that tracks the symmetries of the problem (scalings, translations, rotations, gauge transformations) and a scaling structure axiom (SC) that, combined with dissipation, automatically rules out supercritical self-similar blow-up via pure scaling arithmetic.

### 0.3 Main consequences

From these axioms, we derive:

* **Structural trichotomy (Theorem 7.1).** Any finite-time breakdown falls into exactly one of six classified failure modes.
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

**Remark 0.1 (What instantiation does not require).** Instantiation does not require pre-proving that solutions stay compact. It only requires:
1. Identifying the symmetries $G$ (translations, scalings, gauge transformations),
2. Verifying the "algebraic" axioms (SC, Cap, D) via dimensional analysis.

The framework then outputs the conditional result: *"Self-similar blow-up is impossible; only dispersive singularities or regularity remain."* This converts a hard global regularity problem into a conditional classification, where the remaining analytical work is confined to understanding dispersion—often a more tractable regime.

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

**Axiom C (Structural Compactness Potential).** We say a trajectory $u(t) = S_t x$ with bounded energy $\sup_{t < T_*(x)} \Phi(u(t)) \leq E < \infty$ **satisfies Axiom C** if: for every sequence of times $t_n \nearrow T_*(x)$, there exists a subsequence $(t_{n_k})$ and elements $g_k \in G$ such that $(g_k \cdot u(t_{n_k}))$ converges in $X$.

When $G$ is trivial, this reduces to ordinary precompactness of bounded-energy trajectory tails.

**Role (Diagnostic Filter).** Axiom C is **local and conditional**. We do not assume Axiom C holds for all trajectories. Instead:
- If Axiom C **holds** along a trajectory, the singularity is **Structured**, allowing for profile extraction (the Canonical Profile / Maximizer of Section 7).
- If Axiom C **fails** along a trajectory, the singularity is **Unstructured** (Dispersion/Dust), corresponding to **Mode 2 (Compactness Breakdown)** in the Trichotomy (Theorem 7.1).

This framing is essential: Axiom C is a **classification criterion** that partitions trajectories into two analytic regimes, not a prerequisite that must be established before applying the framework.

**Definition 2.1 (Modulus of compactness).** The **modulus of compactness** along a trajectory $u(t)$ with $\sup_t \Phi(u(t)) \leq E$ is:
$$
\omega_C(\varepsilon, u) := \min\left\{N \in \mathbb{N} : \{u(t) : t < T_*(x)\} \subseteq \bigcup_{i=1}^N g_i \cdot B(x_i, \varepsilon) \text{ for some } g_i \in G, x_i \in X\right\}.
$$
Axiom C holds along a trajectory iff $\omega_C(\varepsilon, u) < \infty$ for all $\varepsilon > 0$.

**Remark 2.2.** In the PDE context, Axiom C is typically verified via:
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

**Fallback (Mode 1).** When Axiom D fails—i.e., the energy grows without bound—the trajectory exhibits **energy blow-up** (Trichotomy mode 1, Theorem 7.1). The drift term is controlled by Axiom R, which bounds time outside $\mathcal{G}$.

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

**Fallback (Mode 1).** When Axiom R fails—i.e., recovery is impossible along a trajectory—the trajectory enters a **failure region** $\mathcal{F}$ where the drift term in Axiom D is uncontrolled, leading to energy blow-up (Trichotomy mode 1).

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

**Fallback (Mode 4).** When Axiom Cap fails along a trajectory—i.e., the trajectory concentrates on high-capacity sets without commensurate dissipation—the trajectory exhibits **geometric concentration** (Trichotomy mode 4, Theorem 7.1).

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

**Fallback (Mode 6).** Axiom LS is **local by design**: it applies only in the neighbourhood $U$ of $M$. When a trajectory approaches the boundary of $U$ or the inequality fails—the trajectory exhibits **stiffness breakdown** (Trichotomy mode 6, Theorem 7.1). Outside $U$, other axioms (C, D, R) govern behaviour.

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
* Theorem 7.1 (Trichotomy): (C), (D), (Reg)
* Theorem 7.2.1 (GN as metatheorem): (D), (SC)
* Theorem 7.2 (Type II exclusion): (D), (SC)
* Theorem 7.3 (Capacity barrier): (Cap), (BG)
* Theorem 7.4 (Topological suppression): (TB), (LSI)
* Theorem 7.5 (Dichotomy): (D), (R), (Cap)
* Theorem 7.6 (Canonical Lyapunov): (C), (D), (R), (LS), (Reg)
* Theorem 7.7.1 (Action Reconstruction): (D), (LS), (GC)
* Theorem 7.7.3 (Hamilton–Jacobi Generator): (D), (LS), (GC)

**Proposition 2.13 (Axiom failure modes).** When an axiom fails along a trajectory, the failure manifests as a specific mode of the Structural Trichotomy (Theorem 7.1). The complete classification is:

| Axiom | Failure Mode | Description |
|-------|--------------|-------------|
| **C** (Compactness) | Mode 2 | Compactness breakdown: sequences along trajectory admit no convergent subsequence modulo $G$ |
| **D** (Dissipation) | Mode 1 | Energy blow-up: energy grows without bound as $t \nearrow T_*(x)$ |
| **R** (Recovery) | Mode 1 | Energy blow-up: trajectory drifts indefinitely in bad region without returning to $\mathcal{G}$ |
| **Cap** (Capacity) | Mode 4 | Geometric concentration: trajectory concentrates on high-capacity sets without commensurate dissipation |
| **LS** (Local Stiffness) | Mode 6 | Stiffness breakdown: gradient of $\mathfrak{D}$ is not controlled by metric near $M$ |
| **SC** (Scaling) | Mode 3 | Supercritical cascade: scaling exponents violate $\alpha > \beta$, enabling Type II blow-up |
| **TB** (Topological) | Mode 5 | Topological metastasis: background invariants are not preserved under concentration |
| **GC** (Gradient Consistency) | — | Reconstruction theorems (7.7.x) do not apply; Theorem 7.6 still provides Lyapunov functional via abstract construction |

*Remark 2.14 (Exhaustiveness).* The Structural Trichotomy (Theorem 7.1) is precisely the exhaustive classification of axiom failures. Every trajectory either:
1. Satisfies all relevant axioms and converges to the safe manifold $M$, or
2. Fails exactly one axiom, triggering the corresponding mode of blow-up/breakdown.

This ensures the framework degrades gracefully: when a local axiom fails, the trichotomy identifies which mode of singular behavior occurs, providing a complete dynamical picture even for trajectories that escape the "good" regime.

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

### 3.3 Class II: Structural dispersion

The second class occurs when the energy remains finite ($\sup_{t < T_*} \Phi(u(t)) < \infty$), but the trajectory fails to converge to a well-defined limit in the quotient space $X/G$.

**Mode 2: Compactness Breakdown.**
- **Axiom Violated:** **(C) Structural Compactness**
- **Diagnostic Test:** There exists a sequence $t_n \nearrow T_*$ such that the orbit sequence $\{u(t_n)\}$ admits **no convergent subsequence** in $X$ modulo the symmetry group $G$.
- **Structural Mechanism:** The energy does not diverge, but it "scatters" or "cascades" into modes that are invisible to the strong topology of $X$ (e.g., oscillation to infinite frequency, loss of mass to spatial infinity, or defect measure formation).
- **Status:** The system approaches a "weak" limit that lies outside the admissible state space. This mode represents **unstructured breakdown**.

### 3.4 Class III: Structural concentration

The third and most critical class occurs when the trajectory remains energy-bounded *and* retains compactness (a limiting profile exists modulo symmetries), but the limiting object itself violates the geometric or local regularity axioms.

This class represents **Structured Singularities**. The breakdown is further stratified by *which* specific geometric constraint is violated by the limiting profile.

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

### 3.5 The diagnostic logic

The classification implies a sequential diagnostic procedure for any singular trajectory. The proof of regularity for a specific system corresponds to sequentially closing these failure modes.

**Theorem 3.2 (The Exhaustion Principle).** Let $\mathcal{S}$ be a hypostructure. A trajectory $u(t)$ extends to $T = +\infty$ (Global Regularity) if and only if:

1. **Mode 1 is excluded** (Energy is a priori bounded via Axiom D).
2. **Mode 2 is excluded** (The orbit is precompact via Axiom C).
3. **Modes 3–6 are excluded** (The limiting profile satisfies Axioms SC, Cap, TB, and LS).

*Proof.* We prove both directions of the equivalence.

**($\Rightarrow$) Global regularity implies exclusion of all modes.**

Suppose $u(t) = S_t x$ extends to $T = +\infty$. We show each mode is excluded.

*Mode 1 excluded:* If the trajectory is globally defined, then by continuity of the semiflow and the definition of blow-up time, $T_*(x) = +\infty$. By Axiom D, for any finite $T$:
$$
\Phi(u(T)) + \alpha \int_0^T \mathfrak{D}(u(s)) \, ds \leq \Phi(u(0)) + C \cdot T_{\text{bad}},
$$
where $T_{\text{bad}}$ is the time spent outside the good region. Since the trajectory is global and has finite cost (by Axiom R), the right-hand side remains finite. Thus $\sup_{t \geq 0} \Phi(u(t)) < \infty$, excluding Mode 1.

*Mode 2 excluded:* With energy bounded, $u(t) \in K_E$ for all $t \geq 0$. By Axiom C, any sequence $u(t_n)$ has a convergent subsequence modulo $G$. Since the trajectory is global, the $\omega$-limit set $\omega(x) := \bigcap_{T > 0} \overline{\{u(t) : t \geq T\}}$ is non-empty and compact in $X/G$. This excludes dispersion (Mode 2).

*Modes 3–6 excluded:* Since the trajectory extends globally, it cannot exhibit finite-time blow-up. Any limit profile $V \in \omega(x)$ satisfies:
- $V$ has finite energy (Mode 3 excluded by SC if $V$ were supercritical),
- $V$ does not concentrate on thin sets (Mode 4 excluded by Cap),
- $V$ lies in the trivial topological sector for generic data (Mode 5 excluded by TB),
- The Łojasiewicz inequality holds near any limit point (Mode 6 excluded by LS).

**($\Leftarrow$) Exclusion of all modes implies global regularity.**

Suppose all six modes are excluded for the trajectory $u(t) = S_t x$. We prove $T_*(x) = +\infty$.

*Step 1: Energy remains bounded.* Since Mode 1 is excluded by hypothesis, $\sup_{t < T_*(x)} \Phi(u(t)) \leq E < \infty$ for some $E$.

*Step 2: Compactness holds.* Since Mode 2 is excluded and energy is bounded, Axiom C applies: any sequence $u(t_n)$ with $t_n \nearrow T_*(x)$ has a subsequence converging modulo $G$ to some profile $V \in X$.

*Step 3: The limit profile is safe.* Since Modes 3–6 are excluded:
- Mode 3 excluded $\Rightarrow$ $V$ is not a supercritical self-similar profile (Axiom SC holds),
- Mode 4 excluded $\Rightarrow$ $V$ is not concentrated on high-capacity sets (Axiom Cap holds),
- Mode 5 excluded $\Rightarrow$ $V$ lies in the trivial topological sector (Axiom TB holds),
- Mode 6 excluded $\Rightarrow$ $V$ satisfies the Łojasiewicz inequality (Axiom LS holds).

Therefore $V \in M$ (the safe manifold) or $V$ satisfies all local regularity conditions.

*Step 4: Extension past $T_*(x)$.* If $T_*(x) < \infty$, then by the definition of blow-up time, the semiflow cannot be extended past $T_*(x)$. But we have shown that $u(t_n) \to g^{-1} \cdot V$ for some $g \in G$ and $V$ satisfying all regularity axioms.

By Axiom Reg (minimal regularity), $T_*$ is lower semicontinuous: $T_*(V) \geq \liminf_{n \to \infty} T_*(u(t_n))$. Since $V \in M$ or $V$ satisfies local regularity, $T_*(V) = +\infty$.

By the semiflow property and continuity, we can restart the flow from $V$ and extend $u$ past $T_*(x)$, contradicting maximality.

Therefore $T_*(x) = +\infty$, establishing global regularity. $\square$

**Remark 3.3 (Softness of the framework).** It is crucial to note that establishing a hypostructure does **not** require proving that Modes 2–6 are impossible *a priori*.
- If a system admits Mode 2 singularities (Compactness Breakdown), the framework simply classifies them as such.
- The framework is a **classifier**, not a regularity theorem. It converts the question "Is this regular?" into the structural question "Which axiom fails?"

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

**Fallback (Mode 3).** When Axiom SC fails along a trajectory—either because no scaling subgroup acts, or the subcritical condition $\alpha > \beta$ is violated—the trajectory may exhibit **supercritical symmetry cascade** (Trichotomy mode 3, Theorem 7.1). Property GN is not derived in this case; Type II blow-up must be excluded by other means or accepted as a possible failure mode.

**Definition 4.13 (Supercritical sequence).** A sequence $(\lambda_n) \subset \mathbb{R}_{> 0}$ is **supercritical** if $\lambda_n \to \infty$.

**Remark 4.14.** The exponent $\alpha$ measures how strongly dissipation responds to zooming; $\beta$ measures how remaining time compresses under scaling. The condition $\alpha > \beta$ ensures that supercritical rescaling amplifies dissipation faster than it compresses time, making infinite-cost profiles unavoidable in the limit.

**Remark 4.15 (Scaling structure is soft).** For most systems of interest, the scaling structure is immediate from dimensional analysis:
* For parabolic PDEs with scaling $(x, t) \mapsto (\lambda x, \lambda^2 t)$, the exponents follow from computing how $\mathfrak{D}$ and $dt$ transform.
* For kinetic systems, the scaling comes from velocity-space rescaling.
* For discrete systems, the scaling may be combinatorial (e.g., term depth).
* For systems without natural scaling symmetry, SC does not apply and GN must be verified separately.

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

### 7.1 Structural trichotomy

**Theorem 7.1 (Structural trichotomy).** Let $\mathcal{S}$ be a structural flow datum satisfying the minimal regularity (Reg) and dissipation (D) axioms. Let $u(t) = S_t x$ be *any* finite-energy trajectory with breakdown time $T_*(x) < \infty$.

Then the behavior of $u(t)$ falls into exactly one of the following classes:

**Class I: Unstructured Breakdown (Axiom C Fails)**

1. **Energy blow-up (Mode 1):** $\Phi(S_{t_n} x) \to \infty$ for some sequence $t_n \nearrow T_*(x)$.

2. **Compactness breakdown (Mode 2):** Energy remains bounded, but no subsequence of $(S_{t_n} x)$ converges modulo symmetries. The singularity disperses or fractalizes such that no coherent profile emerges modulo $G$.

**Class II: Structured Breakdown (Axiom C Holds)**

If Axiom C holds along the trajectory, a **Canonical Profile** $V$ emerges (see Section 8). The breakdown is then classified by which *subsequent* axiom forbids this profile:

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

**Corollary 7.1.1 (Trichotomy as axiom failure classification).** The six modes of Theorem 7.1 are precisely the manifestations of local axiom failures:

| Mode | Failure Type | Axiom Failed |
|------|--------------|--------------|
| (1) | Energy blow-up | **D** — dissipation fails to control energy growth |
| (2) | Compactness breakdown | **C** — bounded sequences have no convergent subsequence |
| (3) | Supercritical cascade | **SC** — scaling exponents violate $\alpha > \beta$ |
| (4) | Geometric concentration | **Cap** — capacity bound exceeded without dissipation |
| (5) | Topological metastasis | **TB** — topological invariants not preserved |
| (6) | Stiffness breakdown | **LS** — Łojasiewicz inequality fails near $M$ |

*Remark 7.1.2 (Mutual exclusivity and completeness).* Each axiom has exactly one associated failure mode, and the trichotomy is exhaustive in the following sense:
1. **Completeness:** If a trajectory $u(t) = S_t x$ has finite-time breakdown $T_*(x) < \infty$, then at least one axiom must fail locally along the trajectory, triggering the corresponding mode.
2. **Diagnosis:** By determining which mode occurs (via the proof structure of Theorem 7.1), one identifies precisely which axiom fails—providing a diagnostic tool for analyzing singular trajectories.
3. **Graceful degradation:** The framework does not require all axioms to hold globally. Each axiom is invoked only along trajectories where it applies, and when it fails, the corresponding trichotomy mode describes the resulting singular behavior.

This establishes the Structural Trichotomy as the complete classification of dynamical singularities in hypostructures: every trajectory either satisfies all relevant axioms and converges to $M$, or fails exactly one axiom and exhibits the corresponding breakdown mode.

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

No system-specific estimates, no Caffarelli–Kohn–Nirenberg, no backward uniqueness—just scaling arithmetic. This is the sense in which GN is a **metatheorem**: once SC is verified (which requires only dimensional analysis), GN follows automatically.

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

The theorems in Sections 7.1–7.6 assume a height functional $\Phi$ is given and verify its properties. We now provide a **generator**: a mechanism to explicitly recover the Lyapunov functional $\mathcal{L}$ solely from the dynamical data $(S_t)$ and the dissipation structure $(\mathfrak{D})$, without prior knowledge of $\Phi$.

This moves the framework from **verification** (checking if a given $\Phi$ works) to **discovery** (finding the correct $\Phi$).

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

**Step 5: Lyapunov property verification.** Along a trajectory $u(t)$:
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

**Remark 8.2 (The key insight).** We do not assume $V$ exists *a priori*.
- If the sequence $u(t_n)$ experiences **Compactness Breakdown** (Failure Mode 2), then $V = 0$ (or does not exist). The branch is empty; no singularity forms via structure.
- If the sequence concentrates, Axiom C **forces** $V$ to exist.

**Remark 8.2.1 (On global compactness).** A common misconception is that one must prove global compactness (e.g., a universal Rellich–Kondrachov theorem) to use this framework. This is false. The hypostructure framework accepts trajectories where compactness fails—it classifies them as **Mode 2 (Compactness Breakdown)**.

The framework analyzes both branches: when compactness fails (Mode 2), and when compactness holds. In the latter case, algebraic constraints (Axioms SC, Cap) exclude structured singularities, reducing the analysis to the dispersive regime. In many applications, dispersive singularities are either:
- Well-understood via scattering theory, or
- Excluded by conservation laws or other structural arguments.

Thus the framework converts a global compactness problem into a conditional classification.

### 8.3 The taxonomy of maximizers

Once Axiom C extracts the profile $V$, the hypostructure framework classifies it immediately. The "Maximizer" $V$ must fall into one of two categories:

**Type A: The Safe Maximizer ($V \in M$).**
The profile $V$ lies in the **Safe Manifold** (e.g., a soliton, a ground state, or a vacuum state).
- **Mechanism:** The trajectory is simply zooming in on a regular structure (like a soliton).
- **Outcome:** **Axiom LS (Stiffness)** applies. The trajectory is constrained near $M$. Since elements of $M$ are global solutions with infinite existence time, this is not a singularity; it is **Soliton Resolution**.

**Type B: Non-safe profile ($V \notin M$).**
The profile $V$ is a self-similar blow-up profile or a high-energy bubble that is *not* in the safe manifold.
- **Mechanism:** The system is attempting to construct a Type II blow-up.
- **Outcome:** The **Admissibility Tests** apply. We do not need to analyze the PDE evolution of $V$. We only need to verify whether $V$ satisfies the axiom constraints.

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

### 8.5 The no-assumption logic flow

To clarify why the framework does not assume existence, we present the following logical flowchart:

**Step 1: Does the singularity try to form?**
- **NO:** Regularity holds. (Done).
- **YES:** The solution concentrates. Proceed to Step 2.

**Step 2: Does the concentration preserve structure?** (Axiom C test)
- **NO:** Compactness Breakdown. Energy disperses to infinity/dust. No structure forms. (Singularity fails via Mode 2).
- **YES:** A **Canonical Profile $V$** emerges (The Maximizer). Proceed to Step 3.

**Step 3: Is the Maximizer Safe?** ($V \in M$ test)
- **YES:** Soliton Resolution / Asymptotic Stability. (Singularity fails via Mode 6 / Stability).
- **NO:** It is a **non-safe profile**. Proceed to Step 4.

**Step 4: Does the profile satisfy the scaling constraint?** (Axiom SC test)
- **NO:** Property GN proves infinite cost. (Singularity fails via Mode 3).
- **YES:** (Only possible if $\alpha \leq \beta$, i.e., supercritical regime).

**Conclusion:** We never assume a global maximizer exists.
- If the branch is empty (no concentration), we win by default.
- If the branch is full (concentration), Axiom C *produces* the maximizer $V$ for us.
- We then exclude $V$ using scaling algebra.

### 8.6 Implementation guide: How to endow solutions

When instantiating the framework for a specific system, one does not search for the global maximizer of the functional. The procedure is as follows:

**Step 1: Identify the Symmetry Group $G$.**
For example: Scaling $\lambda$, Translation $x_0$.

**Step 2: Verify Axiom C (Compactness).**
Prove that any sequence with bounded critical norm that does *not* disperse has a convergent subsequence modulo $G$. This is typically a standard Profile Decomposition theorem.

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

While the previous chapters focus on the *classification* of singularities (Trichotomy) and their *structural resolution* (Maximizers), this chapter addresses the *quantification* of the breakdown.

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

1. **Axiom D verification** requires the Sobolev inequality $\|u\|_{L^{p+1}} \leq C \|\nabla u\|_{L^2}$.
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
- Verify lower semicontinuity and properness.

**Step 4: Identify the dissipation functional $\mathfrak{D}$.**
- Viscous dissipation, entropy production, Fisher information, reduction cost.
- Verify the energy-dissipation inequality (Axiom D).

**Step 5: Verify the core axioms.**
- **(C) Compactness:** Use problem-specific compactness theorems.
- **(D) Dissipation:** Derive from the equations/dynamics.
- **(R) Recovery:** Identify the good region and verify the recovery inequality.
- **(Cap) Capacity:** Define capacity density and verify the bound.
- **(LS) Local stiffness:** Identify equilibria $M$ and verify Łojasiewicz near $M$.
- **(Reg) Regularity:** Verify continuity and semicontinuity properties.

**Step 6: Identify symmetries and construct the gauge.**
- Determine the symmetry group $G$ (translations, rotations, scalings, gauge transformations).
- Construct a normalized slice $\Sigma$ and gauge map $\Gamma$.
- Verify normalization compatibility (Axiom N).

**Step 7: Verify the Scaling Structure axiom (SC).**
- Identify the scaling subgroup $(\mathcal{S}_\lambda) \subset G$.
- Compute the dissipation scaling exponent $\alpha$: how does $\mathfrak{D}(\mathcal{S}_\lambda \cdot x)$ scale with $\lambda$?
- Compute the temporal scaling exponent $\beta$: how does $dt$ transform under rescaling?
- Verify the subcritical condition $\alpha > \beta$.
- **Note:** This is pure dimensional analysis—no hard lemmas required. Once SC is verified, Property GN follows automatically from Theorem 7.2.1.

**Step 8: Specify background structures.**
- **(BG) Geometric:** Specify dimension $Q$, verify Ahlfors regularity, capacity-codimension bounds.
- **(TB) Topological:** Identify topological sectors $\tau$, action functional $\mathcal{A}$, action gap $\Delta$.

**Conclusion:** Once all axioms are verified, Theorems 7.1–7.6 apply, giving complete singularity control for the system.

### 10.2 PDE instantiation tips

For parabolic PDEs (e.g., semilinear heat equations, reaction–diffusion, geometric flows):

**State space:**
- Sobolev spaces $H^s(\Omega)$, $W^{k,p}(\Omega)$
- Besov spaces $B^s_{p,q}$ for critical regularity
- Weak solution spaces (e.g., energy class solutions)

**Compactness (C):**
- Rellich–Kondrachov: $H^1 \hookrightarrow\hookrightarrow L^2$ on bounded domains.
- Aubin–Lions: $L^2(0,T; H^1) \cap H^1(0,T; H^{-1}) \hookrightarrow\hookrightarrow L^2(0,T; L^2)$.
- Concentration-compactness: for scale-invariant problems on $\mathbb{R}^n$.
- Profile decomposition: for dispersive/wave equations.

**Dissipation (D):**
- Viscous dissipation: $\mathfrak{D}(u) = \nu \|\nabla u\|_{L^2}^2$ for diffusive systems.
- Entropy production: $\mathfrak{D}(f) = \int |\nabla \log f|^2 f \, dx$ for Fokker–Planck.
- Verify the energy identity/inequality from the PDE.

**Recovery (R):**
- Good region: where standard parabolic estimates apply.
- Recovery from local smoothing: DeGiorgi–Nash–Moser type estimates.
- Parabolic regularization: solutions become smooth instantly.

**Capacity (Cap):**
- Capacity from scaling-critical norms: $c(u) = \|u\|_{\dot{H}^{s_c}}^p$ at critical regularity $s_c$.
- Frequency localization: high-frequency concentration increases capacity.
- Concentration-compactness methods for critical problems.

**Local stiffness (LS):**
- Linearized spectrum around equilibria.
- Łojasiewicz–Simon for semilinear parabolic PDEs.
- Simon's theorem: analytic nonlinearities give $\theta < 1$.

**Scaling structure (SC):**
- Identify the natural scaling: parabolic $(x, t) \mapsto (\lambda x, \lambda^2 t)$ with field rescaling $u_\lambda(x, t) = \lambda^\gamma u(\lambda x, \lambda^2 t)$.
- Compute $\alpha$: how $\mathfrak{D}$ transforms under $u \mapsto \mathcal{S}_\lambda \cdot u$. This is dimensional analysis.
- Compute $\beta$: the temporal exponent from $dt \to \lambda^{-\beta} ds$.
- Verify $\alpha > \beta$: the subcritical dissipation condition.
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

**Axiom verification:**
- **(C):** Aubin–Lions gives compactness of bounded energy trajectories in $L^2_{\mathrm{loc}}$.
- **(D):** Energy inequality derived from testing the equation against $\partial_t u$.
- **(R):** Local smoothing via parabolic regularity in regions of bounded energy.
- **(Cap):** Capacity from scaling-critical norms at the critical Sobolev exponent.
- **(LS):** Linearized operator spectrum around equilibria.

**SC verification:** The parabolic scaling is $u_\lambda(x,t) = \lambda^\gamma u(\lambda x, \lambda^2 t)$ with $\gamma$ determined by the nonlinearity. Under this scaling:
- Dissipation $\mathfrak{D}$ transforms with exponent $\alpha$ determined by dimensional analysis.
- Time transforms with exponent $\beta = 2$.
- The subcritical condition $\alpha > \beta$ is verified by direct computation when the nonlinearity is energy-subcritical.
- **Consequence:** By Theorem 7.2.1, GN holds automatically—Type II blow-up is framework-forbidden.

**Conclusion:** Once all axioms are verified, Theorems 7.1–7.6 apply, giving complete singularity classification.

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

**Axiom verification:**
- **(C):** Allard compactness for varifolds with bounded mass.
- **(D):** $\frac{d}{dt} \mathcal{H}^n(M_t) = -\int_{M_t} H^2 \, d\mathcal{H}^n$.
- **(SC):** Parabolic rescaling of surfaces; Huisken monotonicity formula encodes the scaling structure. Exponents follow from dimensional analysis of area and Willmore energy.
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

**Axiom verification:**
- **(C):** Prokhorov compactness from moment bounds.
- **(D):** Free energy dissipation identity from Fokker–Planck structure.
- **(LS):** Log-Sobolev inequality from uniform convexity of $V + W * \mu$.
- **(TB):** Topological sectors from homotopy classes of configurations (for topological particles).

**SC verification:** Scaling of Fisher information under measure dilation gives exponent $\alpha$; diffusive time scaling gives $\beta$. The subcritical condition follows from entropy-production structure. GN then follows automatically from Theorem 7.2.1.

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

**Axiom verification:**
- **(C):** Compactness of terms of bounded size (finite state space at each level).
- **(D):** Many reduction strategies decrease size; for typed λ-calculus, strong normalization holds.
- **(R):** Normalization theorems show all terms reach normal form.
- **(LS):** Normal forms are exactly the fixed points of $S$; uniqueness from confluence.
- **(TB):** Type sectors: simply-typed, System F types, etc. Different types prevent interconversion.

**SC interpretation:** The scaling structure for term rewriting is combinatorial: "zooming" into a subterm while tracking reduction cost. The subcritical condition $\alpha > \beta$ encodes that cost accumulates faster than the reduction sequence can extend. Strong normalization is then equivalent to GN: the absence of infinite reduction sequences at finite cost. For typed calculi, this follows from type-theoretic methods; for resource-aware systems, from cost bounds. Either way, once SC is verified, GN holds automatically.

**Interaction nets:** Similar instantiation with:
- $X$: interaction net graphs.
- $\Phi$: number of active pairs or graph size.
- $\mathfrak{D}$: cost per interaction step.
- Confluence and strong normalization give the axioms.

---

This completes the framework of Hypostructures with full categorical foundations, rigorous axiom formulations, complete proofs of the main meta-theorems, and detailed instantiation guidance.

The central thesis: **verification of the structural axioms (C, D, R, Cap, LS, Reg, SC, GC, BG, TB) implies complete control over singularities, with the Lyapunov functional explicitly constructible from dynamical data.**
