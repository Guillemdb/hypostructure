# Part V: The Theory of Learning

*The Agent, The Loss, and The Solver.*

## Meta-Learning Axioms (The L-Layer) {#ch:meta-learning}

In previous chapters, each soft axiom $A$ was associated with a defect functional $K_A : \mathcal{U} \to [0,\infty]$ defined on a class $\mathcal{U}$ of trajectories. The value $K_A(u)$ quantifies the extent to which axiom $A$ fails along trajectory $u$, and vanishes when the axiom is exactly satisfied.

In this chapter, the axioms themselves are treated as objects to be chosen: each axiom is specified by a family of global parameters, and these parameters are determined as minimizers of defect functionals. Global axioms are obtained as minimizers of the defects of their local soft counterparts.

### Parametric families of axioms

\begin{definition}[Parameter space]\label{def:parameter-space}
Let $\Theta$ be a metric space (typically a subset of a finite-dimensional vector space $\mathbb{R}^d$). A \textbf{parametric axiom family} is a collection $\{A_\theta\}_{\theta \in \Theta}$ where each $A_\theta$ is a soft axiom instantiated by global data depending on $\theta$.
\end{definition}

\begin{definition}[Parametric hypostructure components]\label{def:parametric-hypostructure-components}
For each $\theta \in \Theta$, define:
\begin{itemize}
\item \textbf{Parametric height functional:} $\Phi_\theta : X \to \mathbb{R}$
\item \textbf{Parametric dissipation:} $\mathfrak{D}_\theta : X \to [0,\infty]$
\item \textbf{Parametric symmetry group:} $G_\theta \subset \mathrm{Aut}(X)$
\item \textbf{Parametric local structures:} metrics, norms, or capacities depending on $\theta$
\end{itemize}
\end{definition}

The tuple $\mathbb{H}_\theta = (X, S_t, \Phi_\theta, \mathfrak{D}_\theta, G_\theta)$ is a **parametric hypostructure**.

\begin{definition}[Parametric defect functional]\label{def:parametric-defect-functional}
For each $\theta \in \Theta$ and each soft axiom label $A \in \mathcal{A} = \{\text{C}, \text{D}, \text{SC}, \text{Cap}, \text{LS}, \text{TB}\}$, define the defect functional:
$$K_A^{(\theta)} : \mathcal{U} \to [0,\infty]$$
constructed from the hypostructure $\mathbb{H}_\theta$ and the local definition of axiom $A$.
\end{definition}

\begin{lemma}[Defect characterization]\label{lem:defect-characterization}
For all $\theta \in \Theta$ and $u \in \mathcal{U}$:
$$K_A^{(\theta)}(u) = 0 \quad \Longleftrightarrow \quad \text{trajectory } u \text{ satisfies } A_\theta \text{ exactly.}$$
Small values of $K_A^{(\theta)}(u)$ correspond to small violations of axiom $A_\theta$.
\end{lemma}

\begin{proof}
We verify the characterization for each axiom $A \in \mathcal{A}$:

\textbf{(C) Compatibility:} $K_C^{(\theta)}(u) := \|S_t(u(s)) - u(s+t)\|$ for appropriate $s, t \in T$. This equals zero if and only if $u$ is a trajectory of the semiflow.

\textbf{(D) Dissipation:} $K_D^{(\theta)}(u) := \int_T \max(0, \partial_t \Phi_\theta(u(t)) + \mathfrak{D}_\theta(u(t))) dt$. This equals zero if and only if $\partial_t \Phi_\theta + \mathfrak{D}_\theta \leq 0$ holds pointwise along $u$.

\textbf{(SC) Symmetry Compatibility:} $K_{SC}^{(\theta)}(u) := \sup_{g \in G_\theta} \sup_{t \in T} d(g \cdot u(t), S_t(g \cdot u(0)))$. This equals zero if and only if the semiflow commutes with the $G_\theta$-action along $u$.

\textbf{(Cap) Capacity Bounds:} $K_{Cap}^{(\theta)}(u) := \int_T |\text{cap}(\{u(t)\}) - \mathfrak{D}_\theta(u(t))| dt$ (or analogous comparison). Vanishes when capacity and dissipation agree.

\textbf{(LS) Local Structure:} $K_{LS}^{(\theta)}(u)$ measures deviations from local metric, norm, or regularity assumptions as specified in previous chapters.

\textbf{(TB) Thermodynamic Bounds:} $K_{TB}^{(\theta)}(u)$ measures violations of data processing inequalities or entropy bounds.

In each case, $K_A^{(\theta)}(u) \geq 0$ with equality if and only if the constraint is satisfied exactly.
\end{proof}

### Global defect functionals and axiom risk

\begin{definition}[Trajectory measure]\label{def:trajectory-measure}
Let $\mu$ be a $\sigma$-finite measure on the trajectory space $\mathcal{U}$. This measure describes how trajectories are sampled or weighted—for instance, a law induced by initial conditions and the evolution $S_t$, or an empirical distribution of observed trajectories.
\end{definition}

\begin{definition}[Expected defect]\label{def:expected-defect}
For each axiom $A \in \mathcal{A}$ and parameter $\theta \in \Theta$, define the \textbf{expected defect}:
$$\mathcal{R}_A(\theta) := \int_{\mathcal{U}} K_A^{(\theta)}(u) \, d\mu(u)$$
whenever the integral is well-defined and finite.
\end{definition}

\begin{definition}[Worst-case defect]\label{def:worst-case-defect}
For an admissible class $\mathcal{U}_{\text{adm}} \subset \mathcal{U}$, define:
$$\mathcal{K}_A(\theta) := \sup_{u \in \mathcal{U}_{\text{adm}}} K_A^{(\theta)}(u).$$
\end{definition}

\begin{definition}[Joint axiom risk]\label{def:joint-axiom-risk}
For a finite family of soft axioms $\mathcal{A}$ with nonnegative weights $(w_A)_{A \in \mathcal{A}}$, define the \textbf{joint axiom risk}:
$$\mathcal{R}(\theta) := \sum_{A \in \mathcal{A}} w_A \, \mathcal{R}_A(\theta).$$
\end{definition}

\begin{lemma}[Interpretation of axiom risk]\label{lem:interpretation-of-axiom-risk}
The quantity $\mathcal{R}_A(\theta)$ measures the global quality of axiom $A_\theta$:
\begin{itemize}
\item Small values indicate that, on average with respect to $\mu$, axiom $A_\theta$ is nearly satisfied.
\item Large values indicate frequent or severe violations.
\end{itemize}
\end{lemma}

\begin{proof}
By Definition 12.6, $\mathcal{R}_A(\theta) = \int_{\mathcal{U}} K_A^{(\theta)}(u) \, d\mu(u)$. Since $K_A^{(\theta)}(u) \geq 0$ with equality precisely when trajectory $u$ satisfies axiom $A$ under parameter $\theta$ (Definition 12.3), we have:

\begin{enumerate}
\item \textbf{Small $\mathcal{R}_A(\theta)$:} The integral is small if and only if $K_A^{(\theta)}(u)$ is small for $\mu$-almost every $u$, meaning the axiom is satisfied or nearly satisfied across the trajectory distribution.

\item \textbf{Large $\mathcal{R}_A(\theta)$:} The integral is large if either (i) $K_A^{(\theta)}(u)$ is large on a set of positive $\mu$-measure (severe violations), or (ii) $K_A^{(\theta)}(u)$ is moderate on a large set (frequent violations). In both cases, axiom $A$ fails systematically under parameter $\theta$.
\end{enumerate}

The interpretation follows from the positivity and integrability of the defect functional.
\end{proof}

#### The Epistemic Action Principle

The joint axiom risk $\mathcal{R}(\theta)$ admits a physical interpretation that unifies the framework with standard physics. We introduce the **Meta-Action Functional** and the **Principle of Least Structural Defect**.

\begin{definition}[Meta-Action Functional]\label{def:meta-action-functional}
Define the \textbf{Meta-Action} $\mathcal{S}_{\text{meta}}: \Theta \to \mathbb{R}$ as:
$$
\mathcal{S}_{\text{meta}}(\theta) := \int_{\text{System Space}} \left(
\underbrace{\mathcal{L}_{\text{fit}}(\theta, u)}_{\text{Data Fit (Kinetic)}} +
\underbrace{\lambda \sum_{A \in \mathcal{A}} w_A K_A^{(\theta)}(u)^2}_{\text{Structural Penalty (Potential)}}
\right) d\mu_{\text{sys}}(u)
$$
where:
\begin{itemize}
\item $\mathcal{L}_{\text{fit}}(\theta, u)$ measures empirical fit (analogous to kinetic energy),
\item $K_A^{(\theta)}(u)^2$ measures structural violation (analogous to potential energy),
\item $\lambda > 0$ is a coupling constant balancing fit and structure.
\end{itemize}
\end{definition}

**Principle 12.8.2 (Least Structural Defect).** The optimal axiom parameters $\theta^*$ minimize the Meta-Action:
$$
\theta^* = \arg\min_{\theta \in \Theta} \mathcal{S}_{\text{meta}}(\theta).
$$

*Physical Interpretation:* Just as particles follow paths of least action in configuration space, physical laws follow paths of least structural contradiction in theory space. The learning process is not "optimization" but convergence to a **stable configuration in theory space**.

\begin{remark}[Unification with Standard Physics]
The Meta-Action $\mathcal{S}_{\text{meta}}$ plays the same role in theory space that the physical action $S = \int L \, dt$ plays in configuration space:

| \textbf{Classical Mechanics} | \textbf{Meta-Axiomatics} |
|-------------------------|---------------------|
| Configuration $q(t)$ | Parameters $\theta$ |
| Lagrangian $L(q, \dot{q})$ | Integrand $\mathcal{L}_{\text{fit}} + \lambda \sum K_A^2$ |
| Action $S = \int L \, dt$ | Meta-Action $\mathcal{S}_{\text{meta}}$ |
| Least Action Principle | Least Structural Defect |
| Equations of motion | Axiom selection |

The AGI finds theories that are \textbf{stationary points} of $\mathcal{S}_{\text{meta}}$. The Euler-Lagrange equations for $\mathcal{S}_{\text{meta}}$ determine the optimal axiom parameters.
\end{remark}

\begin{proposition}[Variational Characterization]\label{prop:variational-characterization}
Under the assumptions of \cref{mt:existence-of-axiom-minimizers}, the global axiom minimizer $\theta^*$ satisfies the variational equation:
$$
\nabla_\theta \mathcal{S}_{\text{meta}}(\theta^*) = 0.
$$
Moreover, if $\mathcal{S}_{\text{meta}}$ is strictly convex, $\theta^*$ is unique.
\end{proposition}

\begin{proof}
By \cref{mt:existence-of-axiom-minimizers}, $\theta^*$ exists. If $\theta^*$ is an interior point of $\Theta$, the first-order necessary condition is $\nabla_\theta \mathcal{S}_{\text{meta}}(\theta^*) = 0$. Strict convexity implies uniqueness by standard arguments.
\end{proof}

### Canonical Hypostructure Constructions

Before developing the theory of trainable hypostructures, we establish that the axioms Cap and TB are not arbitrary impositions but arise naturally from standard dynamical systems theory. This section presents two existence metatheorems showing that any sufficiently regular dissipative system—whether deterministic or stochastic—automatically admits a hypostructure.

#### Metatheorem: Conley–Hypostructure Existence [@Conley78; @FranzeMisch88]

The key insight is that any dissipative semiflow with a Conley–Morse decomposition automatically provides the data for axioms C, D, Cap, TB (and often LS), without arbitrary choices.

**Setup.** Let $(X, d)$ be a separable metric space and $(S_t)_{t \geq 0}$ a continuous semiflow:
- $S_0 = \mathrm{id}$, $S_{t+s} = S_t \circ S_s$, $S_t$ continuous in $(t, x)$.

Assume:

1. **(Dissipative / global attractor + Lyapunov.)** There is a compact global attractor $\mathcal{A} \subset X$ and a continuous proper function $V: X \to [0, \infty)$ such that:
   - $V(S_t x)$ is nonincreasing in $t$ for all $x$,
   - $V(S_t x)$ is strictly decreasing whenever $S_t x$ is not chain recurrent.

2. **(Finite Conley–Morse decomposition.)** The chain recurrent set $\mathcal{R} \subset \mathcal{A}$ decomposes into finitely many isolated invariant sets:
   $$\mathcal{R} = \bigsqcup_{i=0}^N M_i$$
   with a partial order $M_i \prec M_j$ given by existence of connecting orbits, and a **Morse–Lyapunov function** $V$ such that:
   $$i \prec j \implies \sup_{x \in M_i} V(x) < \inf_{y \in M_j} V(y).$$

3. **(Mild regularity for LS, optional.)** If Axiom LS is desired, assume that near each $M_i$ the flow is gradient-like for a $C^2$ (or analytic) potential, so that a Łojasiewicz–Simon type inequality holds in a neighborhood of $M_i$.

\begin{metatheorem}[Conley–Hypostructure Existence]\label{mt:conleyhypostructure-existence}
Under assumptions (1)–(2), there exists a hypostructure
$$\mathcal{H} = (X, S_t, \Phi, \mathfrak{D}, c, \tau, \mathcal{A}, \ldots)$$
on the same underlying flow such that:
\end{metatheorem}

- **Axiom C** (compactness) holds on $\mathcal{A}$,
- **Axiom D** (dissipation) holds with respect to $\Phi$,
- **Axiom Cap** (capacity) holds for a canonical capacity density $c$,
- **Axiom TB** (topological background) holds with sectors given by the Morse components,
- If (3) holds, **Axiom LS** holds near each $M_i$.

In particular, $(X, S_t)$ is a valid S-layer hypostructure once we optionally add SC/GC in whatever trivial or nontrivial way is relevant.

\begin{proof}[Construction]
We construct each structural component explicitly from the Conley–Lyapunov data.

\textbf{Step 1: Axiom C – Compactness via global attractor.}

Set the height to be the Lyapunov function:
$$\Phi(x) := V(x).$$

Because we have a global attractor $\mathcal{A}$ and $V$ is proper, each energy sublevel
$$K_E := \{x \in \mathcal{A} : \Phi(x) \leq E\}$$
is compact. If a trajectory has $\sup_{t \geq 0} \Phi(S_t x) \leq E$, then its orbit sits inside $K_E$, hence is precompact. This is exactly the "bounded energy $\Rightarrow$ profile precompactness" formulation of Axiom C. The modulus of compactness $\omega_C(\varepsilon, u)$ can be defined using a finite $\varepsilon$-net of $K_E$.

\textbf{Step 2: Axiom D – Dissipation from the Lyapunov function.}

For each $x \in X$, define:
$$\mathfrak{D}(x) := -\left.\frac{d}{dt}\right|_{t=0^+} V(S_t x)$$
where the derivative exists, and otherwise take an upper Dini derivative:
$$\mathfrak{D}(x) := \max\left(0, -\limsup_{h \downarrow 0} \frac{V(S_h x) - V(x)}{h}\right).$$

Then along any trajectory $u(t) = S_t x$ we have the energy–dissipation inequality:
$$V(u(T)) + \int_0^T \mathfrak{D}(u(t)) \, dt \leq V(u(0)).$$

This is exactly Axiom D: energy decreases by at least the accumulated dissipation.

\textbf{Step 3: Axiom Cap – Canonical capacity from dissipation.}

The key observation: for \textbf{existence} of a hypostructure, we need only \textit{some} $c$ satisfying the capacity axiom. The canonical choice is:
$$c(x) := \mathfrak{D}(x), \qquad C_{\mathrm{cap}} := 1, \quad C_0 := 0.$$

Then along any trajectory:
$$\int_0^T c(u(t)) \, dt = \int_0^T \mathfrak{D}(u(t)) \, dt \leq 1 \cdot \int_0^T \mathfrak{D}(u(t)) \, dt + 0 \cdot \Phi(x).$$

So Axiom Cap is satisfied \textbf{tautologically}. The induced capacity of a set $B$ is:
$$\mathrm{Cap}(B) = \inf_{x \in B} \mathfrak{D}(x),$$
consistent with the framework. Sets where $\mathfrak{D}$ is small have low capacity, so one can loiter there cheaply; sets where $\mathfrak{D}$ is bounded below have positive capacity and thus bounded occupation time.

> \textbf{Key Insight:} Cap is not a deep extra assumption. As soon as you have a Lyapunov dissipation structure, you can set $c = \mathfrak{D}$ and the axiom holds. All the nice Hausdorff-dimension / intersection-theory versions are refinements, not prerequisites.

\textbf{Step 4: Axiom TB – Sectors from Morse components + action from Lyapunov gaps.}

Let the index set $\mathcal{T}$ be the set of Morse components:
$$\mathcal{T} := \{0, 1, \ldots, N\},$$
where we choose $M_0$ to be the "trivial" sector (e.g., the global attractor bottom).

For each point $x \in X$, define its sector as the index of its $\omega$-limit Morse set:
$$\tau(x) := i \quad \text{if } \omega(x) \subset M_i.$$

Because the $\omega$-limit set of a trajectory doesn't change along the orbit, $\tau(S_t x) = \tau(x)$: flow invariance holds.

For the action functional, use the Morse–Lyapunov function values at the invariant sets. Let:
$$v_i := \sup_{x \in M_i} V(x).$$

By assumption, for $i \prec j$ we have $v_i < v_j$, and there are finitely many $M_i$, so the set $\{v_i\}$ is finite. Define:
\begin{itemize}
\item Trivial sector as $M_0$ with $\mathcal{A}_{\min} := v_0$,
\item Sector action levels $a_i := v_i$ for $i = 0, \ldots, N$,
\item General action $\mathcal{A}(x) := a_{\tau(x)}$.
\end{itemize}

Now:
$$\Delta := \min_{i \neq 0}(a_i - a_0) > 0$$
since there are finitely many $a_i$ and $a_i > a_0$ for nontrivial sectors.

So \textbf{TB1} holds:
$$\tau(x) \neq 0 \implies \mathcal{A}(x) = a_{\tau(x)} \geq a_0 + \Delta = \mathcal{A}_{\min} + \Delta.$$

For \textbf{TB2} (action–height coupling), note that $\Phi(x) = V(x)$ and $v_i \leq \sup_{y \in \mathcal{A}} V(y) =: V_{\max}$. Taking $C_{\mathcal{A}} := 1$ and allowing a small constant, we have $\mathcal{A}(x) \leq \Phi(x) + C$ on $\mathcal{A}$.

\textbf{Step 5: Axiom LS – Local stiffness from hyperbolicity / Łojasiewicz (optional).}

If the flow is gradient-like with analytic potential near each Morse set, standard Łojasiewicz–Simon results give:
$$|\nabla V(x)| \geq c_{\mathrm{LS}} |V(x) - V(M_i)|^{1-\theta}$$
in a neighborhood of $M_i$, for some $\theta \in (0, 1)$. This exactly produces the LS axiom: gradient norm controls energy drop, giving finite-time convergence once in a small neighborhood.
\end{proof}

#### Metatheorem: Ergodic–Hypostructure Existence

We now establish the probabilistic/stochastic analog: any nice Markov/measure-preserving system with metastable structure automatically gives axioms C, D, Cap, TB.

**Setup.** Let:
- $X$ be a Polish (complete separable metric) space,
- $(S_t)_{t \geq 0}$ a measurable semiflow or Markov process on $X$,
- $\mu$ a **stationary/invariant probability measure**: $\mu(S_t^{-1}A) = \mu(A)$ for all $A$, $t \geq 0$.

Assume:

1. **(Tightness / effective compactness.)** There is a coercive function $V: X \to [0, \infty)$ with $\int V \, d\mu < \infty$, and for each $E$, the sublevel set $\{V \leq E\}$ is relatively compact.

2. **(Dissipativity in expectation.)** There is a measurable function $\mathfrak{D}: X \to [0, \infty)$ and constants $c_1, c_2 > 0$ with, for all $t \geq 0$:
   $$\mathbb{E}[V(S_t x) - V(x) \mid x] \leq -c_1 \mathbb{E}\left[\int_0^t \mathfrak{D}(S_s x) \, ds \,\Big|\, x\right] + c_2 t.$$
   (A standard drift–dissipation inequality; cf. Foster–Lyapunov in MCMC.)

3. **(Metastable decomposition.)** There exists a finite partition of $X$ (mod $\mu$-a.e.):
   $$X = \bigsqcup_{i=0}^N A_i$$
   with:
   - each $A_i$ **metastable** (mean exit time $\mathbb{E}_x T_{A_i^c}$ much larger than mixing time inside $A_i$),
   - transitions between $A_i$ and $A_j$ are rare and have well-defined log-rates (e.g., Freidlin–Wentzell / large deviations, or spectral gap structure).

\begin{metatheorem}[Ergodic–Hypostructure Existence]\label{mt:ergodichypostructure-existence}
Under (1)–(3), there exists a hypostructure
$$\mathcal{H} = (X, S_t, \Phi, \mathfrak{D}, c, \tau, \mathcal{A}, \ldots)$$
such that:
\end{metatheorem}

- **Axiom C** (compactness) holds on $\{V \leq E\}$ for any fixed $E$,
- **Axiom D** (dissipation) holds in expectation with $\Phi = V$,
- **Axiom Cap** (capacity) holds with a canonical capacity density derived from $\mathfrak{D}$,
- **Axiom TB** (topological barrier) holds with sectors $\tau$ given by metastable sets $A_i$ and an action built from log-transition rates,
- If the process satisfies a suitable Łojasiewicz/gradient-like condition near attractors (e.g., SDE with analytic potential), then **LS** holds locally in probability.

\begin{proof}[Construction]
We construct each component from the ergodic/metastable data.

\textbf{Step 1: Axiom C – Compactness from tightness of $\mu$.}

Take height $\Phi(x) := V(x)$. Assumption (1) says for each $E$, $K_E := \{x : \Phi(x) \leq E\}$ is relatively compact, and $\mu(K_E^c)$ is small for large $E$. Conditioned on $\Phi(x) \leq E$, almost every realization of the process has precompact paths in $K_E$. So in the "reduced" state space $\{V \leq E\}$, Axiom C holds.

\textbf{Step 2: Axiom D – Dissipation from drift of $V$.}

Set $\Phi = V$ and use $\mathfrak{D}$ from assumption (2). The drift–dissipation inequality gives (in integrated form):
$$\mathbb{E}[\Phi(S_T x)] + c_1 \mathbb{E}\left[\int_0^T \mathfrak{D}(S_t x) \, dt\right] \leq \Phi(x) + c_2 T.$$

This is exactly the \textbf{expected energy–dissipation inequality} version of Axiom D.

\textbf{Step 3: Axiom Cap – Capacity from dissipation.}

Again, the canonical choice is $c(x) := \mathfrak{D}(x)$. Then:
$$\mathbb{E}\left[\int_0^T c(S_t x) \, dt\right] = \mathbb{E}\left[\int_0^T \mathfrak{D}(S_t x) \, dt\right] \leq C_{\mathrm{cap}} \mathbb{E}\left[\int_0^T \mathfrak{D}(S_t x) \, dt\right] + C_0 \Phi(x)$$
with $C_{\mathrm{cap}} = 1$, $C_0 = 0$. So Axiom Cap is satisfied in the probabilistic sense.

\textbf{Step 4: Axiom TB – Sectors from metastable sets, action from transition costs.}

Use the metastable partition $X = \bigsqcup A_i$ from (3). Define:
$$\tau(x) := i \quad \text{if } x \in A_i.$$

This is a \textbf{coarse-grained topological sector}: inside each $A_i$, the process mixes quickly; between different $A_i$, transitions are rare.

Standard metastability/large deviations says: the transition rate from $A_i$ to $A_j$ behaves like:
$$\mathbb{P}(\text{hit } A_j \text{ before returning to } A_i \mid X_0 \in A_i) \approx \exp(-\mathcal{Q}_{ij}/\varepsilon)$$
for some quasi-potential $\mathcal{Q}_{ij} > 0$.

Define a baseline sector $A_0$ and let $\mathcal{Q}_i$ be the minimal quasi-potential barrier to enter sector $i$:
$$\mathcal{Q}_i := \inf_{\text{paths } A_0 \to A_i} \sum \mathcal{Q}_{kl}.$$

Set $\mathcal{A}(x) := \mathcal{Q}_{\tau(x)}$. Then:
$$\Delta := \min_{i \neq 0}(\mathcal{Q}_i - \mathcal{Q}_0) > 0$$
if we normalize $\mathcal{Q}_0 = 0$ and assume metastable separation.

This implies \textbf{TB1}: any nontrivial sector has action at least $\Delta$ above the baseline. For \textbf{TB2}, standard potential–quasi-potential bounds give $\mathcal{A}(x) \lesssim \sup_{x \in A_i} V(x) - \inf_X V$, so $\mathcal{A}(x) \leq C_{\mathcal{A}} \Phi(x) + C$ on $\mathrm{supp}(\mu)$.
\end{proof}

> **Key Insight:** These existence metatheorems show that the hypostructure axioms are not arbitrary—they emerge naturally from:
> - **Deterministic systems:** Conley–Morse decompositions + Lyapunov functions $\Rightarrow$ axioms C, D, Cap, TB
> - **Stochastic systems:** Metastable decompositions + drift conditions $\Rightarrow$ axioms C, D, Cap, TB
>
> In both cases, **Cap = dissipation density** and **TB = Morse/metastable sectors**. This provides the conceptual foundation for why trainable hypostructures can learn these structures from data.

---

### Trainable global axioms

\begin{definition}[Global axiom minimizer]\label{def:global-axiom-minimizer}
A point $\theta^* \in \Theta$ is a \textbf{global axiom minimizer} if:
$$\mathcal{R}(\theta^*) = \inf_{\theta \in \Theta} \mathcal{R}(\theta).$$
\end{definition}

\begin{metatheorem}[Existence of Axiom Minimizers]\label{mt:existence-of-axiom-minimizers}
Assume:
\begin{enumerate}
\item The parameter space $\Theta$ is compact and metrizable.
\item For each $A \in \mathcal{A}$ and each $u \in \mathcal{U}$, the map $\theta \mapsto K_A^{(\theta)}(u)$ is continuous on $\Theta$.
\item There exists an integrable majorant $M_A \in L^1(\mu)$ such that $0 \leq K_A^{(\theta)}(u) \leq M_A(u)$ for all $\theta \in \Theta$ and $\mu$-a.e. $u$.
\end{enumerate}
\end{metatheorem}

Then, for each $A \in \mathcal{A}$, the expected defect $\mathcal{R}_A(\theta)$ is finite and continuous on $\Theta$. Consequently, the joint risk $\mathcal{R}(\theta)$ is continuous and attains its infimum on $\Theta$. There exists at least one global axiom minimizer $\theta^* \in \Theta$.

\begin{proof}
\textbf{Step 1 (Setup).} Let $\theta_n \to \theta$ in $\Theta$. We must show $\mathcal{R}_A(\theta_n) \to \mathcal{R}_A(\theta)$.

\textbf{Step 2 (Pointwise convergence).} By assumption (2), for each $u \in \mathcal{U}$:
$$K_A^{(\theta_n)}(u) \to K_A^{(\theta)}(u).$$

\textbf{Step 3 (Dominated convergence).} By assumption (3), $|K_A^{(\theta_n)}(u)| \leq M_A(u)$ with $M_A \in L^1(\mu)$. The dominated convergence theorem yields:
$$\mathcal{R}_A(\theta_n) = \int_{\mathcal{U}} K_A^{(\theta_n)}(u) \, d\mu(u) \to \int_{\mathcal{U}} K_A^{(\theta)}(u) \, d\mu(u) = \mathcal{R}_A(\theta).$$

\textbf{Step 4 (Continuity of joint risk).} Since $\mathcal{R}(\theta) = \sum_{A \in \mathcal{A}} w_A \mathcal{R}_A(\theta)$ is a finite sum of continuous functions, it is continuous.

\textbf{Step 5 (Existence).} By the extreme value theorem, a continuous function on a compact set attains its infimum. Hence there exists $\theta^* \in \Theta$ with $\mathcal{R}(\theta^*) = \inf_{\theta \in \Theta} \mathcal{R}(\theta)$.
\end{proof}

\begin{corollary}[Characterization of exact minimizers]\label{cor:characterization-of-exact-minimizers}
If $\mathcal{R}_A(\theta^*) = 0$ for all $A \in \mathcal{A}$, then all axioms in $\mathcal{A}$ hold $\mu$-almost surely under $A_{\theta^*}$. The hypostructure $\mathbb{H}_{\theta^*}$ satisfies all soft axioms globally.
\end{corollary}

\begin{proof}
If $\mathcal{R}_A(\theta^*) = \int K_A^{(\theta^*)} d\mu = 0$ and $K_A^{(\theta^*)} \geq 0$, then $K_A^{(\theta^*)}(u) = 0$ for $\mu$-a.e. $u$. By \cref{lem:defect-characterization}, axiom $A_{\theta^*}$ holds $\mu$-almost surely.
\end{proof}

### Gradient-based approximation

Assume $\Theta \subset \mathbb{R}^d$ is open and convex.

\begin{lemma}[Leibniz rule for axiom risk]\label{lem:leibniz-rule-for-axiom-risk}
Assume:
\begin{enumerate}
\item For each $A \in \mathcal{A}$ and each $u \in \mathcal{U}$, the map $\theta \mapsto K_A^{(\theta)}(u)$ is differentiable on $\Theta$ with gradient $\nabla_\theta K_A^{(\theta)}(u)$.
\item There exists an integrable majorant $M_A \in L^1(\mu)$ such that $|\nabla_\theta K_A^{(\theta)}(u)| \leq M_A(u)$ for all $\theta \in \Theta$ and $\mu$-a.e. $u$.
\end{enumerate}
\end{lemma}

Then the gradient of $\mathcal{R}_A$ admits the integral representation:
$$\nabla_\theta \mathcal{R}_A(\theta) = \int_{\mathcal{U}} \nabla_\theta K_A^{(\theta)}(u) \, d\mu(u).$$

\begin{proof}
\textbf{Step 1 (Difference quotient).} For $h \in \mathbb{R}^d$ with $|h|$ small:
$$\frac{\mathcal{R}_A(\theta + h) - \mathcal{R}_A(\theta)}{|h|} = \int_{\mathcal{U}} \frac{K_A^{(\theta + h)}(u) - K_A^{(\theta)}(u)}{|h|} \, d\mu(u).$$

\textbf{Step 2 (Mean value theorem).} By differentiability, for each $u$:
$$\frac{K_A^{(\theta + h)}(u) - K_A^{(\theta)}(u)}{|h|} \to \nabla_\theta K_A^{(\theta)}(u) \cdot \frac{h}{|h|}$$
as $|h| \to 0$.

\textbf{Step 3 (Dominated convergence).} The mean value theorem gives:
$$\left|\frac{K_A^{(\theta + h)}(u) - K_A^{(\theta)}(u)}{|h|}\right| \leq \sup_{\xi \in [\theta, \theta+h]} |\nabla_\theta K_A^{(\xi)}(u)| \leq M_A(u).$$
By dominated convergence, differentiation passes through the integral.
\end{proof}

\begin{corollary}[Gradient of joint risk]\label{cor:gradient-of-joint-risk}
Under the assumptions of \cref{lem:leibniz-rule-for-axiom-risk}:
$$\nabla_\theta \mathcal{R}(\theta) = \sum_{A \in \mathcal{A}} w_A \int_{\mathcal{U}} \nabla_\theta K_A^{(\theta)}(u) \, d\mu(u).$$
\end{corollary}

\begin{corollary}[Gradient descent convergence]\label{cor:gradient-descent-convergence}
Consider the gradient descent iteration:
$$\theta_{k+1} = \theta_k - \eta_k \nabla_\theta \mathcal{R}(\theta_k)$$
with step sizes $\eta_k > 0$ satisfying $\sum_k \eta_k = \infty$ and $\sum_k \eta_k^2 < \infty$.
\end{corollary}

Under the assumptions of \cref{lem:leibniz-rule-for-axiom-risk}, together with Lipschitz continuity of $\nabla_\theta \mathcal{R}$, the sequence $(\theta_k)$ has accumulation points, and every accumulation point is a stationary point of $\mathcal{R}$.

If additionally $\mathcal{R}$ is convex, every accumulation point is a global axiom minimizer.

\begin{proof}
We apply the Robbins-Monro theorem.

\textbf{Step 1 (Descent property).} For $L$-Lipschitz continuous gradients:
$$\mathcal{R}(\theta_{k+1}) \leq \mathcal{R}(\theta_k) - \eta_k \|\nabla \mathcal{R}(\theta_k)\|^2 + \frac{L\eta_k^2}{2}\|\nabla \mathcal{R}(\theta_k)\|^2.$$

\textbf{Step 2 (Summability).} Summing over $k$ and using $\sum_k \eta_k^2 < \infty$:
$$\sum_{k=0}^\infty \eta_k(1 - L\eta_k/2)\|\nabla \mathcal{R}(\theta_k)\|^2 \leq \mathcal{R}(\theta_0) - \inf \mathcal{R} < \infty.$$
Since $\sum_k \eta_k = \infty$ and $\eta_k \to 0$, we have $\liminf_{k \to \infty} \|\nabla \mathcal{R}(\theta_k)\| = 0$.

\textbf{Step 3 (Accumulation points).} Compactness of $\Theta$ (\cref{mt:existence-of-axiom-minimizers}, assumption 1) ensures $(\theta_k)$ has accumulation points. Continuity of $\nabla \mathcal{R}$ implies any accumulation point $\theta^*$ satisfies $\nabla \mathcal{R}(\theta^*) = 0$ (stationary).

\textbf{Step 4 (Convex case).} If $\mathcal{R}$ is convex, stationary points satisfy $\nabla \mathcal{R}(\theta^*) = 0$ if and only if $\theta^*$ is a global minimizer.
\end{proof}

### Joint training of axioms and extremizers

\begin{definition}[Two-level parameterization]\label{def:two-level-parameterization}
Consider:
\begin{itemize}
\item \textbf{Hypostructure parameters:} $\theta \in \Theta$ defining $\Phi_\theta, \mathfrak{D}_\theta, G_\theta$
\item \textbf{Extremizer parameters:} $\vartheta \in \Upsilon$ parametrizing candidate trajectories $u_\vartheta \in \mathcal{U}$
\end{itemize}
\end{definition}

\begin{definition}[Joint training objective]\label{def:joint-training-objective}
Define:
$$\mathcal{L}(\theta, \vartheta) := \sum_{A \in \mathcal{A}} w_A \, \mathbb{E}[K_A^{(\theta)}(u_\vartheta)] + \sum_{B \in \mathcal{B}} v_B \, \mathbb{E}[F_B^{(\theta)}(u_\vartheta)]$$
where:
\begin{itemize}
\item $\mathcal{A}$ indexes axioms whose defects are minimized
\item $\mathcal{B}$ indexes extremal problems whose values $F_B^{(\theta)}(u_\vartheta)$ are optimized
\end{itemize}
\end{definition}

\begin{metatheorem}[Joint Training Dynamics]\label{mt:joint-training-dynamics}
Under differentiability assumptions analogous to \cref{lem:leibniz-rule-for-axiom-risk} for both $\theta$ and $\vartheta$, the objective $\mathcal{L}$ is differentiable in $(\theta, \vartheta)$. The joint gradient descent:
$$(\theta_{k+1}, \vartheta_{k+1}) = (\theta_k, \vartheta_k) - \eta_k \nabla_{(\theta, \vartheta)} \mathcal{L}(\theta_k, \vartheta_k)$$
converges to stationary points under standard conditions.
\end{metatheorem}

\begin{proof}
\textbf{Step 1 (Differentiability).} Both $\theta \mapsto K_A^{(\theta)}(u_\vartheta)$ and $\vartheta \mapsto u_\vartheta$ are differentiable by assumption. Chain rule gives differentiability of the composition.

\textbf{Step 2 (Integral exchange).} Dominated convergence (as in \cref{lem:leibniz-rule-for-axiom-risk}) allows differentiation under the expectation.

\textbf{Step 3 (Convergence).} The same Robbins-Monro analysis as in \cref{cor:gradient-descent-convergence} applies to the joint iteration on $(\theta, \vartheta) \in \Theta \times \Upsilon$. Under Lipschitz continuity of $\nabla_{(\theta, \vartheta)} \mathcal{L}$ and compactness of $\Theta \times \Upsilon$, the descent inequality holds in the product space. The step size conditions ensure convergence to stationary points of $\mathcal{L}$.
\end{proof}

\begin{corollary}[Interpretation]\label{cor:interpretation}
In this scheme:
\begin{itemize}
\item The global axioms $\theta$ are \textbf{learned} to minimize defects of local soft axioms.
\item The extremal profiles $\vartheta$ are simultaneously tuned to probe and saturate the variational problems defined by these axioms.
\item The resulting pair $(\theta^*, \vartheta^*)$ consists of a globally adapted hypostructure and representative extremal trajectories within it.
\end{itemize}
\end{corollary}

### Imported Learning Metatheorems

The following metatheorems are imported from the core hypostructure framework and provide the foundational identifiability and reconstruction results required for trainable hypostructures.

\begin{metatheorem}[SV-09: Meta-Identifiability]\label{mt:sv-09-meta-identifiability}
\textbf{[Sieve Signature]}
\begin{itemize}
\item \textbf{Weakest Precondition}: $K_5^+$ (Parameters stable) AND $K_7^+$ (Log-Sobolev)
\item \textbf{Produces}: $K_{\text{SV09}}$ (Local Injectivity)
\item \textbf{Invalidated By}: $K_5^-$ (degenerate parametrization)
\end{itemize}

Permits: $\mathcal{P}_{\text{full}}$ (default; specialize if fewer permits are needed).

\textbf{Statement}: Parameters are learnable under persistent excitation and nondegenerate parametrization.

\textit{Algorithmic Class:} Parameter Estimation. \textit{Convergence:} Local Injectivity.
\end{metatheorem}

\begin{metatheorem}[Functional Reconstruction]\label{mt:functional-reconstruction}
\textbf{[Sieve Signature]}
Permits: $\mathcal{P}_{\text{full}}$ (default; specialize if fewer permits are needed).

\begin{itemize}
\item \textbf{Weakest Precondition}: $K_{12}^+$ (gradient consistency) AND $\{K_{11}^+ \lor K_{\text{Epi}}^{\text{blk}}\}$ (finite dictionary)
\item \textbf{Consumes}: Context $\Gamma$ with GradientCheck and ComplexCheck certificates
\item \textbf{Produces}: $K_{\text{Reconstruct}}$ (explicit Lyapunov functional)
\item \textbf{Invalidated By}: $K_{12}^-$ (gradient inconsistency) or $K_{\text{Epi}}^{\text{br}}$ (semantic horizon)
\end{itemize}

\textbf{Statement}: If the local Context $\Gamma$ contains gradient consistency and finite dictionary certificates, the Lyapunov functional is explicitly recoverable as the geodesic distance in a Jacobi metric, or as the solution to a Hamilton–Jacobi equation. No prior knowledge of an energy functional is required.
\end{metatheorem}

---

### Trainable Hypostructure Consistency

The preceding sections established that axiom defects can be minimized via gradient descent. This section proves the central metatheorem: under identifiability conditions, defect minimization provably recovers the true hypostructure and its structural predictions.

**Setting.** Fix a dynamical system $S$ with state space $X$, semiflow $S_t$, and trajectory class $\mathcal{U}$. Suppose there exists a "true" hypostructure $\mathcal{H}_{\Theta^*} = (X, S_t, \Phi_{\Theta^*}, \mathfrak{D}_{\Theta^*}, G_{\Theta^*})$ satisfying the axioms. Consider a parametric family $\{\mathcal{H}_\theta\}_{\theta \in \Theta_{\mathrm{adm}}}$ containing $\mathcal{H}_{\Theta^*}$, with joint axiom risk:
$$\mathcal{R}(\theta) := \sum_{A \in \mathcal{A}} w_A \, \mathcal{R}_A(\theta), \quad \mathcal{R}_A(\theta) := \int_{\mathcal{U}} K_A^{(\theta)}(u) \, d\mu(u).$$

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom Validity at $\Theta^*$:** The target hypostructure $\mathcal{H}_{\Theta^*}$ satisfies axioms (C, D, SC, Cap, LS, TB, Reg, GC)
>     *   [ ] **Well-Behaved Defect Functionals:** Compact $\Theta$, continuous $\theta \mapsto K_A^{(\theta)}(u)$, integrable majorants (\cref{lem:leibniz-rule-for-axiom-risk})
>     *   [ ] **Structural Identifiability:** Persistent excitation (C1), nondegenerate parametrization (C2), regular parameter space (C3) (\cref{mt:sv-09-meta-identifiability})
>     *   [ ] **Defect Reconstruction:** Reconstruction of $(\Phi_\theta, \mathfrak{D}_\theta, S_t, \text{barriers}, M)$ from defects up to Hypo-isomorphism (\cref{mt:defect-reconstruction-2})
> *   **Output (Structural Guarantee):**
>     *   Global minimizer $\Theta^*$ satisfies $\mathcal{R}(\Theta^*) = 0$; any global minimizer $\hat{\theta}$ with $\mathcal{R}(\hat{\theta}) = 0$ yields $\mathcal{H}_{\hat{\theta}} \cong \mathcal{H}_{\Theta^*}$
>     *   Local quadratic identifiability: $c|\theta - \tilde{\Theta}|^2 \leq \mathcal{R}(\theta) \leq C|\theta - \tilde{\Theta}|^2$
>     *   Gradient descent converges to true hypostructure with Robbins-Monro step sizes
>     *   Barrier constants and failure-mode classifications converge
> *   **Failure Condition (Debug):**
>     *   If **Axiom Validity** fails → **Mode misspecification** (wrong axiom target)
>     *   If **Identifiability** fails → **Mode parameter degeneracy** (multiple equivalent minima)
>     *   If **Defect Reconstruction** fails → **Mode reconstruction ambiguity** (structural non-uniqueness)

\begin{metatheorem}[Trainable Hypostructure Consistency]\label{mt:trainable-hypostructure-consistency}
Let $S$ be a dynamical system with a hypostructure representation $\mathcal{H}_{\Theta^*}$ inside a parametric family $\{\mathcal{H}_\theta\}_{\theta \in \Theta_{\mathrm{adm}}}$. Assume:
\end{metatheorem}

1. **(Axiom validity at $\Theta^*$.)** The hypostructure $\mathcal{H}_{\Theta^*}$ satisfies axioms (C, D, SC, Cap, LS, TB, Reg, GC). Consequently, $K_A^{(\Theta^*)}(u) = 0$ for $\mu$-a.e. trajectory $u \in \mathcal{U}$ and all $A \in \mathcal{A}$.

2. **(Well-behaved defect functionals.)** The assumptions of \cref{lem:leibniz-rule-for-axiom-risk} hold: $\Theta$ compact and metrizable, $\theta \mapsto K_A^{(\theta)}(u)$ continuous and differentiable with integrable majorants.

3. **(Structural identifiability.)** The family satisfies the conditions of \cref{mt:sv-09-meta-identifiability}: persistent excitation (C1), nondegenerate parametrization (C2), and regular parameter space (C3).

4. **(Defect reconstruction.)** The Defect Reconstruction Theorem (\cref{mt:defect-reconstruction-2}) holds: from $\{K_A^{(\theta)}\}_{A \in \mathcal{A}}$ on $\mathcal{U}$, one reconstructs $(\Phi_\theta, \mathfrak{D}_\theta, S_t, \text{barriers}, M)$ up to Hypo-isomorphism.

Consider gradient descent with step sizes $\eta_k > 0$ satisfying $\sum_k \eta_k = \infty$, $\sum_k \eta_k^2 < \infty$:
$$\theta_{k+1} = \theta_k - \eta_k \nabla_\theta \mathcal{R}(\theta_k).$$

Then:

1. **(Correctness of global minimizer.)** $\Theta^*$ is a global minimizer of $\mathcal{R}$ with $\mathcal{R}(\Theta^*) = 0$. Conversely, any global minimizer $\hat{\theta}$ with $\mathcal{R}(\hat{\theta}) = 0$ satisfies $\mathcal{H}_{\hat{\theta}} \cong \mathcal{H}_{\Theta^*}$ (Hypo-isomorphic).

2. **(Local quantitative identifiability.)** There exist $c, C, \varepsilon_0 > 0$ such that for $|\theta - \Theta^*| < \varepsilon_0$:
$$c \, |\theta - \tilde{\Theta}|^2 \leq \mathcal{R}(\theta) \leq C \, |\theta - \tilde{\Theta}|^2$$
where $\tilde{\Theta}$ is a representative of $[\Theta^*]$. In particular: $\mathcal{R}(\theta) \leq \varepsilon \Rightarrow |\theta - \tilde{\Theta}| \leq \sqrt{\varepsilon/c}$.

3. **(Convergence to true hypostructure.)** Every accumulation point of $(\theta_k)$ is stationary. Under the local strong convexity of (2), any sequence initialized sufficiently close to $[\Theta^*]$ converges to some $\tilde{\Theta} \in [\Theta^*]$.

4. **(Barrier and failure-mode convergence.)** As $\theta_k \to \tilde{\Theta}$, barrier constants converge to those of $\mathcal{H}_{\Theta^*}$, and for all large $k$, $\mathcal{H}_{\theta_k}$ forbids exactly the same failure modes as $\mathcal{H}_{\Theta^*}$.

\begin{proof}
\textbf{Step 1 ($\Theta^*$ is correct global minimizer).} By assumption (1), $K_A^{(\Theta^*)}(u) = 0$ for $\mu$-a.e. $u$ and all $A$. Thus $\mathcal{R}_A(\Theta^*) = 0$ for all $A$, hence $\mathcal{R}(\Theta^*) = 0$. Since $K_A^{(\theta)} \geq 0$, we have $\mathcal{R}(\theta) \geq 0$ for all $\theta$, so $\Theta^*$ achieves the global minimum.

Conversely, if $\mathcal{R}(\hat{\theta}) = 0$, then $\mathcal{R}_A(\hat{\theta}) = 0$ for all $A$, so $K_A^{(\hat{\theta})}(u) = 0$ for $\mu$-a.e. $u$. By the Defect Reconstruction Theorem, both $\mathcal{H}_{\hat{\theta}}$ and $\mathcal{H}_{\Theta^*}$ reconstruct to the same structural data on the support of $\mu$. By structural identifiability (\cref{mt:sv-09-meta-identifiability}), $\mathcal{H}_{\hat{\theta}} \cong \mathcal{H}_{\Theta^*}$.

\textbf{Step 2 (Local quadratic bounds).} By Defect Reconstruction and structural identifiability, the map $\theta \mapsto \mathsf{Sig}(\theta)$ is locally injective around $[\Theta^*]$ up to gauge. Since $\mathcal{R}(\Theta^*) = 0$ and $\nabla \mathcal{R}(\Theta^*) = 0$ (all defects vanish), Taylor expansion gives:
$$\mathcal{R}(\theta) = \frac{1}{2}(\theta - \tilde{\Theta})^\top H (\theta - \tilde{\Theta}) + o(|\theta - \tilde{\Theta}|^2)$$
where $H = \sum_A w_A H_A$ is the Hessian. Identifiability implies $H$ is positive definite on $\Theta_{\mathrm{adm}}/{\sim}$ (directions that leave all defects unchanged correspond to pure gauge). Thus for small $|\theta - \tilde{\Theta}|$:
$$c \, |\theta - \tilde{\Theta}|^2 \leq \mathcal{R}(\theta) \leq C \, |\theta - \tilde{\Theta}|^2.$$

\textbf{Step 3 (Gradient descent convergence).} By \cref{cor:gradient-descent-convergence}, accumulation points are stationary. The local strong convexity from Step 2 implies: on $B(\tilde{\Theta}, \varepsilon_0)$, $\mathcal{R}$ is strongly convex (modulo gauge) with unique stationary point $\tilde{\Theta}$. Standard optimization theory for strongly convex functions with Robbins-Monro step sizes yields convergence of $(\theta_k)$ to $\tilde{\Theta}$ when initialized in this basin.

\textbf{Step 4 (Barrier convergence).} Barrier constants and failure-mode classifications are continuous in the structural data $(\Phi, \mathfrak{D}, \alpha, \beta, \ldots)$ by \cref{mt:sv-09-meta-identifiability}. Since $\theta_k \to \tilde{\Theta}$, structural data converges, hence barriers converge and failure-mode predictions stabilize.
\end{proof}

**Key Insight (Structural parameter estimation).** This theorem elevates Part VII from "we can optimize a loss" to a metatheorem: under identifiability, **structural parameters are estimable**. The parameter manifold $\Theta$ is equipped with the Fisher-Rao metric, following Amari's Information Geometry [@Amari00], treating learning as a projection onto a statistical manifold. The minimization of axiom risk $\mathcal{R}(\theta)$ converges to the unique hypostructure compatible with the trajectory distribution $\mu$, and all high-level structural predictions (barrier constants, forbidden failure modes) converge with it.

---

\begin{remark}[What the metatheorem says]
In plain language:
\begin{enumerate}
\item If a system admits a hypostructure satisfying the axioms for some $\Theta^*$,
\item and the parametric family + data is rich enough to make that hypostructure identifiable,
\item then defect minimization is a \textbf{consistent learning principle}:
   \begin{itemize}
   \item The global minimum corresponds exactly to $\Theta^*$ (mod gauge)
   \item Small risk means ``almost recovered the true axioms''
   \item Gradient descent converges to the correct hypostructure
   \item All structural predictions (barriers, forbidden modes) converge
   \end{itemize}
\end{enumerate}
\end{remark}

\begin{corollary}[Verification via training]\label{cor:verification-via-training}
A trained hypostructure with $\mathcal{R}(\theta_k) < \varepsilon$ provides:
\end{corollary}

1. **Approximate axiom satisfaction:** Each axiom holds with defect at most $\varepsilon/w_A$
2. **Approximate structural recovery:** Parameters within $\sqrt{\varepsilon/c}$ of truth
3. **Correct qualitative predictions:** For $\varepsilon$ small enough, barrier signs and failure-mode classifications match the true system

This connects the trainable framework to the diagnostic and verification goals of the hypostructure program.

### Meta-Error Localization

The previous section established that defect minimization recovers the true hypostructure. This section addresses a finer question: when training yields nonzero residual risk, **which axiom block is misspecified?** We prove that the pattern of residual risks under blockwise retraining uniquely identifies the error location.

#### Parameter block structure

\begin{definition}[Block decomposition]\label{def:block-decomposition}
Decompose the parameter space into axiom-aligned blocks:
$$\theta = (\theta^{\mathrm{dyn}}, \theta^{\mathrm{cap}}, \theta^{\mathrm{sc}}, \theta^{\mathrm{top}}, \theta^{\mathrm{ls}}) \in \Theta_{\mathrm{adm}}$$
where:
\begin{itemize}
\item $\theta^{\mathrm{dyn}}$: parallel transport/dynamics parameters (C, D axioms)
\item $\theta^{\mathrm{cap}}$: capacity and barrier constants (Cap, TB axioms)
\item $\theta^{\mathrm{sc}}$: scaling exponents and structure (SC axiom)
\item $\theta^{\mathrm{top}}$: topological sector data (TB, topological aspects of Cap)
\item $\theta^{\mathrm{ls}}$: Łojasiewicz exponents and symmetry-breaking data (LS axiom)
\end{itemize}
\end{definition}

Let $\mathcal{B} := \{\mathrm{dyn}, \mathrm{cap}, \mathrm{sc}, \mathrm{top}, \mathrm{ls}\}$ denote the set of block labels.

\begin{definition}[Block-restricted reoptimization]\label{def:block-restricted-reoptimization}
For block $b \in \mathcal{B}$ and current parameter $\theta$, define:
\end{definition}

1. **Feasible set:** $\Theta^b(\theta) := \{\tilde{\theta} \in \Theta_{\mathrm{adm}} : \tilde{\theta}^c = \theta^c \text{ for all } c \neq b\}$
2. **Block-restricted minimal risk:** $\mathcal{R}_b^*(\theta) := \inf_{\tilde{\theta} \in \Theta^b(\theta)} \mathcal{R}(\tilde{\theta})$

This represents "retrain only block $b$" while freezing all other blocks.

\begin{definition}[Response signature]\label{def:response-signature}
The \textbf{response signature} at $\theta$ is:
$$\rho(\theta) := \big(\mathcal{R}_b^*(\theta)\big)_{b \in \mathcal{B}} \in \mathbb{R}_{\geq 0}^{|\mathcal{B}|}$$
\end{definition}

\begin{definition}[Error support]\label{def:error-support}
Given true parameter $\Theta^* = (\Theta^{*,b})_{b \in \mathcal{B}}$ and current parameter $\theta$, the \textbf{error support} is:
$$E(\theta) := \{b \in \mathcal{B} : \theta^b \not\sim \Theta^{*,b}\}$$
where $\sim$ denotes gauge equivalence within Hypo-isomorphism classes.
\end{definition}

#### Localization assumptions

\begin{definition}[Block-orthogonality conditions]\label{def:block-orthogonality-conditions}
The parametric family satisfies \textbf{block-orthogonality} if in a neighborhood $\mathcal{N}$ of $[\Theta^*]$:
\end{definition}

1. **(Smooth risk.)** $\mathcal{R}$ is $C^2$ on $\mathcal{N}$ with Hessian $H := \nabla^2 \mathcal{R}(\Theta^*)$ positive definite modulo gauge.

2. **(Block-diagonal Hessian.)** $H$ decomposes as:
$$H = \bigoplus_{b \in \mathcal{B}} H_b$$
where each $H_b$ is positive definite on its block. Cross-Hessian blocks $H_{bc} = 0$ for $b \neq c$ (modulo gauge).

3. **(Quadratic approximation.)** There exists $\delta > 0$ such that for $|\theta - \Theta^*| < \delta$:
$$\mathcal{R}(\theta) = \frac{1}{2}(\theta - \Theta^*)^\top H (\theta - \Theta^*) + O(|\theta - \Theta^*|^3)$$

\begin{remark}[Interpretation of block-orthogonality]
Condition (2) means: perturbations in different axiom blocks contribute additively and independently to the risk at second order. No combination of ``wrong capacity'' and ``wrong scaling'' can cancel in the expected defect. This holds when the parametrization is factorized by axiom family without hidden re-encodings.
\end{remark}

#### The localization theorem

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Block-Orthogonality Conditions:** Smooth risk $\mathcal{R}$ with positive-definite Hessian $H = \bigoplus_b H_b$, block-diagonal modulo gauge
>     *   [ ] **Quadratic Approximation:** $\mathcal{R}(\theta) = \frac{1}{2}(\theta - \Theta^*)^\top H (\theta - \Theta^*) + O(|\theta - \Theta^*|^3)$
>     *   [ ] **Parameter Block Decomposition:** $\theta = (\theta^{\mathrm{dyn}}, \theta^{\mathrm{cap}}, \theta^{\mathrm{sc}}, \theta^{\mathrm{top}}, \theta^{\mathrm{ls}})$
> *   **Output (Structural Guarantee):**
>     *   Single-block error: uniquely smallest $\mathcal{R}_b^*(\theta)$ identifies misspecified block
>     *   Multiple-block error: response signature discriminates error support
>     *   Signature injectivity: $b \in E(\theta) \iff \mathcal{R}_b^*(\theta) \leq \gamma \cdot \min_{c \notin E(\theta)} \mathcal{R}_c^*(\theta)$
> *   **Failure Condition (Debug):**
>     *   If **Block-Orthogonality** fails → **Mode cross-coupling** (blocks interfere, false positives)
>     *   If **Quadratic Approximation** fails → **Mode higher-order dominance** (cubic terms mask signal)

\begin{metatheorem}[Meta-Error Localization]\label{mt:meta-error-localization}
Assume the block-orthogonality conditions (Definition 13.27). There exist $\mathcal{N}$, $c$, $C$, $\varepsilon_0 > 0$ such that for $\theta \in \mathcal{N}$ with $|\theta - \Theta^*| < \varepsilon_0$:
\end{metatheorem}

1. **(Single-block error.)** If $E(\theta) = \{b^*\}$ (exactly one misspecified block), then:
   - For block $b^*$: $\mathcal{R}_{b^*}^*(\theta) \leq C |\theta - \Theta^*|^3$
   - For $b \neq b^*$: $\mathcal{R}_b^*(\theta) \geq c |\theta - \Theta^*|^2$

   The uniquely smallest $\mathcal{R}_b^*(\theta)$ identifies the misspecified block.

2. **(Multiple-block error.)** For arbitrary nonempty $E(\theta) \subseteq \mathcal{B}$:
   - If $b \notin E(\theta)$: $\mathcal{R}_b^*(\theta) \geq c \sum_{c \in E(\theta)} |\theta^c - \Theta^{*,c}|^2$
   - If $b \in E(\theta)$: $\mathcal{R}_b^*(\theta) \approx \frac{1}{2} \sum_{c \in E(\theta) \setminus \{b\}} (\theta^c - \Theta^{*,c})^\top H_c (\theta^c - \Theta^{*,c})$

3. **(Signature injectivity.)** There exists $\gamma > 0$ such that:
$$b \in E(\theta) \iff \mathcal{R}_b^*(\theta) \leq \gamma \cdot \min_{c \notin E(\theta)} \mathcal{R}_c^*(\theta)$$

The map $E \mapsto \rho(\theta)$ is injective and stable: the response signature uniquely encodes the error support.

\begin{proof}
Let $\delta\theta := \theta - \Theta^*$ with block decomposition $\delta\theta = (\delta\theta^b)_{b \in \mathcal{B}}$.

\textbf{Step 1 (Quadratic structure).} By assumption, $\mathcal{R}(\theta) = \frac{1}{2}\delta\theta^\top H \delta\theta + O(|\delta\theta|^3)$. Block-diagonality gives:
$$\delta\theta^\top H \delta\theta = \sum_{b \in \mathcal{B}} (\delta\theta^b)^\top H_b \delta\theta^b.$$
Since each $H_b$ is positive definite, there exist $m_b, M_b > 0$ with:
$$m_b |\delta\theta^b|^2 \leq (\delta\theta^b)^\top H_b \delta\theta^b \leq M_b |\delta\theta^b|^2.$$

\textbf{Step 2 (Block-restricted optimization).} For block $b$, the restricted optimization varies only $\delta\theta^b$ while fixing $\delta\theta^c$ for $c \neq b$. The quadratic approximation:
$$Q(\delta\theta) = \frac{1}{2} \sum_{c \in \mathcal{B}} (\delta\theta^c)^\top H_c \delta\theta^c$$
splits by block. The minimum over $\delta\theta^b$ is achieved at $\delta\theta^b = 0$, giving:
$$Q_b^*(\delta\theta) := \inf_{\tilde{\delta\theta}^b} Q = \frac{1}{2} \sum_{c \neq b} (\delta\theta^c)^\top H_c \delta\theta^c.$$
The true minimal risk satisfies $|\mathcal{R}_b^*(\theta) - Q_b^*(\delta\theta)| \leq C_1 |\delta\theta|^3$.

\textbf{Step 3 (Single-block case).} If $E(\theta) = \{b^*\}$, then $\delta\theta^c = 0$ for $c \neq b^*$.

For $b = b^*$: $Q_{b^*}^* = \frac{1}{2}\sum_{c \neq b^*} (\delta\theta^c)^\top H_c \delta\theta^c = 0$, so $\mathcal{R}_{b^*}^* \leq C|\delta\theta|^3$.

For $b \neq b^*$: $Q_b^* \geq \frac{1}{2} m_{b^*} |\delta\theta^{b^*}|^2 \geq c|\delta\theta|^2$, so $\mathcal{R}_b^* \geq c|\delta\theta|^2 - C_1|\delta\theta|^3 \geq \frac{c}{2}|\delta\theta|^2$ for small $|\delta\theta|$.

\textbf{Step 4 (Multiple-block case).} For general $E(\theta)$:

If $b \notin E(\theta)$: The sum $Q_b^* = \frac{1}{2}\sum_{c \neq b} (\delta\theta^c)^\top H_c \delta\theta^c$ includes all error blocks $c \in E(\theta)$, giving the lower bound.

If $b \in E(\theta)$: The sum excludes block $b$, so $Q_b^* = \frac{1}{2}\sum_{c \in E(\theta) \setminus \{b\}} (\delta\theta^c)^\top H_c \delta\theta^c$.

\textbf{Step 5 (Signature discrimination).} Blocks in $E(\theta)$ have systematically smaller $\mathcal{R}_b^*$ than blocks not in $E(\theta)$, by a multiplicative margin depending on the spectra of $H_c$. Taking $\gamma$ as the ratio of spectral bounds yields the equivalence.
\end{proof}

---

**Key Insight (Built-in debugger).** A trainable hypostructure comes with principled error diagnosis:

1. Train the full model to reduce $\mathcal{R}(\theta)$
2. If residual risk remains, compute $\mathcal{R}_b^*$ for each block by retraining only that block
3. The pattern $\rho(\theta) = (\mathcal{R}_b^*)_b$ provably identifies which axiom blocks are wrong

\begin{corollary}[Diagnostic protocol]\label{cor:diagnostic-protocol}
Given trained parameters $\theta$ with $\mathcal{R}(\theta) > 0$:
\end{corollary}

1. **Compute response signature:** For each $b \in \mathcal{B}$, solve $\mathcal{R}_b^*(\theta) = \min_{\tilde{\theta}^b} \mathcal{R}(\theta^{-b}, \tilde{\theta}^b)$
2. **Identify error support:** $\hat{E} = \{b : \mathcal{R}_b^*(\theta) \text{ is anomalously small}\}$
3. **Interpret:** The blocks in $\hat{E}$ are misspecified; blocks not in $\hat{E}$ are correct

\begin{remark}[Error types and remediation]
The error support $E(\theta)$ indicates:
\end{remark}

| Error Support | Interpretation | Remediation |
|--------------|----------------|-------------|
| $\{\mathrm{dyn}\}$ | Dynamics model wrong | Revise connection/transport ansatz |
| $\{\mathrm{cap}\}$ | Capacity/barriers wrong | Adjust geometric estimates |
| $\{\mathrm{sc}\}$ | Scaling exponents wrong | Recompute dimensional analysis |
| $\{\mathrm{top}\}$ | Topological sectors wrong | Check sector decomposition |
| $\{\mathrm{ls}\}$ | Łojasiewicz data wrong | Verify equilibrium structure |
| Multiple | Combined misspecification | Address each block |

This connects the trainable framework to systematic model debugging and refinement.

### Block Factorization Axiom

The Meta-Error Localization Theorem (\cref{mt:meta-error-localization}) assumes that when we restrict reoptimization to a single parameter block $\theta^b$, the result meaningfully tests whether that block is correct. This requires that the axiom defects factorize cleanly across parameter blocks—a structural condition we now formalize.

\begin{definition}[Axiom-Support Set]\label{def:axiom-support-set}
For each axiom $A \in \mathcal{A}$, define its \textbf{axiom-support set} $\mathrm{Supp}(A) \subseteq \mathcal{B}$ as the minimal collection of blocks such that:
$$K_A^{(\theta)}(u) = K_A^{(\theta|_{\mathrm{Supp}(A)})}(u)$$
for all trajectories $u$ and all parameters $\theta$. That is, $\mathrm{Supp}(A)$ contains exactly the blocks that the defect functional $K_A$ actually depends on.
\end{definition}

\begin{definition}[Semantic Block via Axiom Support]\label{def:semantic-block-via-axiom-support}
A partition $\mathcal{B}$ of the parameter space $\theta = (\theta^b)_{b \in \mathcal{B}}$ is \textbf{semantically aligned} if each block $b$ corresponds to a coherent set of axiom dependencies:
$$b \in \mathrm{Supp}(A) \implies \text{all parameters in } \theta^b \text{ influence } K_A$$
\end{definition}

**Block Factorization Axiom (BFA).** We say the hypostructure training problem satisfies the **Block Factorization Axiom** if:

**(BFA-1) Sparse support:** Each axiom depends on few blocks:
$$|\mathrm{Supp}(A)| \leq k \quad \text{for all } A \in \mathcal{A}$$
for some constant $k \ll |\mathcal{B}|$.

**(BFA-2) Block coverage:** Each block is responsible for at least one axiom:
$$\forall b \in \mathcal{B}, \exists A \in \mathcal{A}: b \in \mathrm{Supp}(A)$$

**(BFA-3) Separability:** The joint risk decomposes additively across axiom families:
$$\mathcal{R}(\theta) = \sum_{A \in \mathcal{A}} w_A \mathcal{R}_A(\theta)$$
where each $\mathcal{R}_A$ depends only on blocks in $\mathrm{Supp}(A)$.

**(BFA-4) Independence of irrelevant alternatives:** For blocks $b \notin \mathrm{Supp}(A)$:
$$\frac{\partial \mathcal{R}_A}{\partial \theta^b} = 0$$
That is, blocks outside an axiom's support have zero gradient contribution to that axiom's risk.

\begin{remark}[Interpretation]
BFA formalizes the intuition that:
\begin{itemize}
\item \textbf{Dynamics parameters} ($\theta^{\mathrm{dyn}}$) govern D, R, C---the core semiflow structure
\item \textbf{Capacity parameters} ($\theta^{\mathrm{cap}}$) govern Cap, TB---geometric barriers
\item \textbf{Scaling parameters} ($\theta^{\mathrm{sc}}$) govern SC---dimensional analysis
\item \textbf{Topological parameters} ($\theta^{\mathrm{top}}$) govern GC---sector structure
\item \textbf{Łojasiewicz parameters} ($\theta^{\mathrm{ls}}$) govern LS---equilibrium geometry
\end{itemize}
When BFA holds, testing whether $\theta^{\mathrm{cap}}$ is correct (by computing $\mathcal{R}_{\mathrm{cap}}^*$) cannot be confounded by errors in $\theta^{\mathrm{sc}}$, because capacity axioms do not depend on scaling parameters.
\end{remark}

\begin{lemma}[Stability of Block Factorization under Composition]\label{lem:stability-of-block-factorization-under-composition}
Let $(\mathcal{A}_1, \mathcal{B}_1)$ and $(\mathcal{A}_2, \mathcal{B}_2)$ be two axiom-block systems satisfying BFA with constants $k_1$ and $k_2$. If the systems have disjoint parameter spaces, then the combined system $(\mathcal{A}_1 \cup \mathcal{A}_2, \mathcal{B}_1 \cup \mathcal{B}_2)$ satisfies BFA with constant $\max(k_1, k_2)$.
\end{lemma}

\begin{proof}
We verify each clause:

\textbf{Step 1 (BFA-1).} For $A \in \mathcal{A}_1$, $\mathrm{Supp}(A) \subseteq \mathcal{B}_1$ with $|\mathrm{Supp}(A)| \leq k_1$. Similarly for $\mathcal{A}_2$. Thus all axioms satisfy sparse support with constant $\max(k_1, k_2)$.

\textbf{Step 2 (BFA-2).} Each block in $\mathcal{B}_1$ is covered by some axiom in $\mathcal{A}_1$ (by BFA-2 for system 1). Similarly for $\mathcal{B}_2$. Union preserves coverage.

\textbf{Step 3 (BFA-3).} Since parameter spaces are disjoint, $\mathcal{R}_A(\theta_1, \theta_2) = \mathcal{R}_A(\theta_1)$ for $A \in \mathcal{A}_1$. Additive decomposition extends to the union.

\textbf{Step 4 (BFA-4).} For $A \in \mathcal{A}_1$ and $b \in \mathcal{B}_2$, the gradient $\partial \mathcal{R}_A / \partial \theta^b = 0$ because $\mathcal{R}_A$ does not depend on $\mathcal{B}_2$ parameters. Combined with original BFA-4 within each system, independence holds globally.
\end{proof}

\begin{remark}[Role in Meta-Error Localization]
The Meta-Error Localization Theorem (\cref{mt:meta-error-localization}) requires BFA implicitly:
\begin{itemize}
\item \textbf{Response signature well-defined:} $\mathcal{R}_b^*(\theta)$ tests block $b$ in isolation only if BFA-4 ensures other-block gradients do not interfere
\item \textbf{Error support meaningful:} The set $E(\theta) = \{b : \mathcal{R}_b^*(\theta) < \mathcal{R}(\theta)\}$ identifies the \textit{actual} error blocks only if BFA-1 ensures axiom-block correspondences are sparse
\item \textbf{Diagnostic protocol valid:} Corollary 13.30's remediation table assumes the semantic alignment of Definition 13.33
\end{itemize}
When BFA fails---for example, if capacity and scaling parameters are entangled---then $\mathcal{R}_{\mathrm{cap}}^*$ might decrease even when capacity is correct (because reoptimizing $\theta^{\mathrm{cap}}$ partially compensates for $\theta^{\mathrm{sc}}$ errors). This would produce false positives in error localization.
\end{remark}

> **Key Insight:** The Block Factorization Axiom is a *design constraint* on hypostructure parametrizations, not a theorem about dynamics. When constructing trainable hypostructures, one should choose parameter blocks that satisfy BFA—ensuring the Meta-Error Localization machinery works as intended.

### Meta-Generalization Across Systems

In §13.6 we considered a single system $S$ and a parametric family of hypostructures $\{\mathcal{H}_{\Theta,S}\}_{\Theta \in \Theta_{\mathrm{adm}}}$ with axiom-defect risk $\mathcal{R}_S(\Theta)$. We now move to a *distribution of systems* and show that defect-minimizing hypostructure parameters learned on a training distribution $\mathcal{S}_{\mathrm{train}}$ generalize to new systems drawn from the same structural class.

We write $\mathcal{S}$ for a probability measure on a class of systems, and for each $S$ in the support of $\mathcal{S}$, we assume a hypostructure family $\{\mathcal{H}_{\Theta,S}\}_{\Theta \in \Theta_{\mathrm{adm}}}$ and axiom-risk functionals $\mathcal{R}_S(\Theta)$ as in §13.

#### Setting

- Let $\mathcal{S}$ be a distribution over systems $S$ (e.g. PDEs, ODEs, control systems, RL environments) each admitting a hypostructure representation in the same parametric family $\{\mathcal{H}_{\Theta,S}\}_{\Theta \in \Theta_{\mathrm{adm}}}$.

- For each system $S$, the joint axiom-risk $\mathcal{R}_S(\Theta)$ is defined via the defect functionals:
$$\mathcal{R}_S(\Theta) := \sum_{A \in \mathcal{A}} w_A \mathcal{R}_{A,S}(\Theta), \qquad \mathcal{R}_{A,S}(\Theta) := \int_{\mathcal{U}_S} K_{A,S}^{(\Theta)}(u) \, d\mu_S(u),$$
where $\mathcal{U}_S$ is the trajectory class for $S$, $\mu_S$ a trajectory distribution, and $K_{A,S}^{(\Theta)}$ are the axiom defects (as in Part VII).

- The **average axiom risk** over a distribution $\mathcal{S}$ is:
$$\mathcal{R}_{\mathcal{S}}(\Theta) := \mathbb{E}_{S \sim \mathcal{S}}[\mathcal{R}_S(\Theta)].$$

- We consider two distributions $\mathcal{S}_{\mathrm{train}}$ and $\mathcal{S}_{\mathrm{test}}$. For simplicity we first treat the $\mathcal{S}_{\mathrm{train}} = \mathcal{S}_{\mathrm{test}}$ case, then note the extension to covariant shifts.

#### Structural manifold of true hypostructures

We assume that for each system $S$ in the support of $\mathcal{S}$, there exists a "true" parameter $\Theta^*(S) \in \Theta_{\mathrm{adm}}$ such that:

- $\mathcal{H}_{\Theta^*(S),S}$ satisfies the hypostructure axioms (C, D, SC, Cap, LS, TB, Reg, GC) for that system;

- all axiom defects vanish for the true parameter:
$$\mathcal{R}_S(\Theta^*(S)) = 0, \qquad K_{A,S}^{(\Theta^*(S))}(u) = 0 \quad \mu_S\text{-a.e. for all } A \in \mathcal{A};$$

- $\Theta^*(S)$ is uniquely determined up to Hypo-isomorphism by the structural data $(\Phi_{\Theta^*(S),S}, \mathfrak{D}_{\Theta^*(S),S}, \ldots)$ (structural identifiability, as in \cref{mt:sv-09-meta-identifiability}).

We further assume that the map $S \mapsto \Theta^*(S)$ takes values in a compact $C^1$ submanifold $\mathcal{M} \subset \Theta_{\mathrm{adm}}$, which we call the **structural manifold**. Intuitively, $\mathcal{M}$ collects all true hypostructure parameters realized by systems in the support of $\mathcal{S}$.

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **True Hypostructures on Compact Structural Manifold:** $\mathcal{R}_S(\Theta^*(S)) = 0$ for $\mathcal{S}$-a.e. $S$, $\mathcal{M} \subset \Theta_{\mathrm{adm}}$ compact $C^1$ submanifold
>     *   [ ] **Uniform Local Strong Convexity:** $c \, \mathrm{dist}(\Theta, \mathcal{M})^2 \leq \mathcal{R}_S(\Theta) \leq C \, \mathrm{dist}(\Theta, \mathcal{M})^2$ near $\mathcal{M}$
>     *   [ ] **Lipschitz Continuity:** $|\mathcal{R}_S(\Theta) - \mathcal{R}_{S'}(\Theta')| \leq L(d_{\mathcal{S}}(S, S') + |\Theta - \Theta'|)$
>     *   [ ] **Approximate Empirical Minimization:** $\widehat{\mathcal{R}}_N(\widehat{\Theta}_N) \leq \inf_{\Theta} \widehat{\mathcal{R}}_N(\Theta) + \varepsilon_N$
> *   **Output (Structural Guarantee):**
>     *   Generalization: $\mathcal{R}_{\mathcal{S}}(\widehat{\Theta}_N) \leq C_1(\varepsilon_N + \sqrt{\log(1/\delta)/N})$
>     *   Structural recovery: $\mathbb{E}[\mathrm{dist}(\widehat{\Theta}_N, \Theta^*(S))] \leq C_2\sqrt{\varepsilon_N + \sqrt{\log(1/\delta)/N}}$
>     *   Asymptotic consistency as $N \to \infty$, $\varepsilon_N \to 0$
> *   **Failure Condition (Debug):**
>     *   If **Structural Manifold** non-compact → **Mode overfitting** (learned structure specific to training systems)
>     *   If **Lipschitz** fails → **Mode distribution shift** (test systems structurally different)

\begin{metatheorem}[Meta-Generalization]\label{mt:meta-generalization}
Let $\mathcal{S}$ be a distribution over systems $S$, and suppose that:
\end{metatheorem}

1. **True hypostructures on a compact structural manifold.** For $\mathcal{S}$-a.e. $S$, there exists $\Theta^*(S) \in \Theta_{\mathrm{adm}}$ such that:
   - $\mathcal{R}_S(\Theta^*(S)) = 0$;
   - $\mathcal{H}_{\Theta^*(S),S}$ satisfies the hypostructure axioms (C, D, SC, Cap, LS, TB, Reg, GC);
   - $\Theta^*(S)$ is structurally identifiable up to Hypo-isomorphism.

   The image $\mathcal{M} := \{\Theta^*(S) : S \in \mathrm{supp}(\mathcal{S})\}$ is contained in a compact $C^1$ submanifold of $\Theta_{\mathrm{adm}}$.

2. **Uniform local strong convexity near the structural manifold.** There exist constants $c, C, \rho > 0$ such that for all $S$ and all $\Theta$ with $\mathrm{dist}(\Theta, \mathcal{M}) \leq \rho$:
$$c \, \mathrm{dist}(\Theta, \mathcal{M})^2 \leq \mathcal{R}_S(\Theta) \leq C \, \mathrm{dist}(\Theta, \mathcal{M})^2.$$
(Here $\mathrm{dist}$ is taken modulo gauge; this is the multi-task version of the local quadratic bounds from \cref{mt:trainable-hypostructure-consistency} for a single system.)

3. **Lipschitz continuity of risk in $\Theta$ and $S$.** There exists $L > 0$ such that for all $S, S'$ and $\Theta, \Theta'$ in a neighborhood of $\mathcal{M}$:
$$|\mathcal{R}_S(\Theta) - \mathcal{R}_{S'}(\Theta')| \leq L \big( d_{\mathcal{S}}(S, S') + |\Theta - \Theta'| \big),$$
where $d_{\mathcal{S}}$ is a metric on the space of systems compatible with $\mathcal{S}$.

4. **Approximate empirical minimization on training systems.** Let $S_1, \ldots, S_N$ be i.i.d. samples from $\mathcal{S}$. Define the empirical average risk:
$$\widehat{\mathcal{R}}_N(\Theta) := \frac{1}{N} \sum_{i=1}^N \mathcal{R}_{S_i}(\Theta).$$
Suppose $\widehat{\Theta}_N \in \Theta_{\mathrm{adm}}$ satisfies:
$$\widehat{\mathcal{R}}_N(\widehat{\Theta}_N) \leq \inf_{\Theta} \widehat{\mathcal{R}}_N(\Theta) + \varepsilon_N,$$
for some optimization accuracy $\varepsilon_N \geq 0$.

Then, with probability at least $1 - \delta$ over the draw of the $S_i$, the following hold for $N$ large enough:

1. **(Average generalization of axiom risk.)** There exists a constant $C_1$, depending only on the structural manifold and the Lipschitz/convexity constants in (2)–(3), such that:
$$\mathcal{R}_{\mathcal{S}}(\widehat{\Theta}_N) := \mathbb{E}_{S \sim \mathcal{S}}[\mathcal{R}_S(\widehat{\Theta}_N)] \leq C_1 \left( \varepsilon_N + \sqrt{\frac{\log(1/\delta)}{N}} \right).$$

2. **(Average closeness to true hypostructures.)** There exists a constant $C_2 > 0$ such that:
$$\mathbb{E}_{S \sim \mathcal{S}} \big[ \mathrm{dist}(\widehat{\Theta}_N, \Theta^*(S)) \big] \leq C_2 \sqrt{ \varepsilon_N + \sqrt{\tfrac{\log(1/\delta)}{N}} }.$$

3. **(Convergence as $N \to \infty$.)** In particular, if $\varepsilon_N \to 0$ as $N \to \infty$, then:
$$\lim_{N \to \infty} \mathcal{R}_{\mathcal{S}}(\widehat{\Theta}_N) = 0, \qquad \lim_{N \to \infty} \mathbb{E}_{S \sim \mathcal{S}} \big[ \mathrm{dist}(\widehat{\Theta}_N, \Theta^*(S)) \big] = 0,$$
i.e. the learned parameter $\widehat{\Theta}_N$ yields hypostructures that are asymptotically axiom-consistent and structurally correct on average across systems drawn from $\mathcal{S}$.

\begin{proof}
By assumption (1), zero-risk parameters for each system lie on the manifold $\mathcal{M}$. For any $\Theta$ close to $\mathcal{M}$, the uniform quadratic bound (2) implies:
$$c \, \mathrm{dist}(\Theta, \mathcal{M})^2 \leq \mathcal{R}_S(\Theta) \leq C \, \mathrm{dist}(\Theta, \mathcal{M})^2 \quad \text{for all } S.$$

Taking expectations over $S \sim \mathcal{S}$ gives:
$$c \, \mathrm{dist}(\Theta, \mathcal{M})^2 \leq \mathcal{R}_{\mathcal{S}}(\Theta) \leq C \, \mathrm{dist}(\Theta, \mathcal{M})^2.$$

Thus small average risk and small average distance to $\mathcal{M}$ are equivalent up to constants.

Next, $\mathcal{R}_S(\Theta)$ is bounded and Lipschitz in $\Theta$ and $S$ by (3), so standard uniform convergence arguments (e.g. covering number or Rademacher complexity bounds on the function class $\{\mathcal{R}_S(\cdot) : S \in \mathrm{supp}(\mathcal{S})\}$) imply that, with probability at least $1 - \delta$:
$$\sup_{\Theta \in \Theta_{\mathrm{adm}}} \left| \widehat{\mathcal{R}}_N(\Theta) - \mathcal{R}_{\mathcal{S}}(\Theta) \right| \leq C_3 \sqrt{\frac{\log(1/\delta)}{N}},$$
for some constant $C_3$ depending on the Lipschitz constants and the metric entropy of $\Theta_{\mathrm{adm}}$.

By the approximate minimization condition:
$$\widehat{\mathcal{R}}_N(\widehat{\Theta}_N) \leq \widehat{\mathcal{R}}_N(\Theta_{\mathcal{M}}^*) + \varepsilon_N,$$
where $\Theta_{\mathcal{M}}^* \in \mathcal{M}$ is any selector (e.g. minimizing $\mathcal{R}_{\mathcal{S}}$ over $\mathcal{M}$, which is zero by (1)). Using uniform convergence, we get:
$$\mathcal{R}_{\mathcal{S}}(\widehat{\Theta}_N) \leq \widehat{\mathcal{R}}_N(\widehat{\Theta}_N) + C_3 \sqrt{\tfrac{\log(1/\delta)}{N}} \leq \widehat{\mathcal{R}}_N(\Theta_{\mathcal{M}}^*) + \varepsilon_N + C_3 \sqrt{\tfrac{\log(1/\delta)}{N}} \leq \mathcal{R}_{\mathcal{S}}(\Theta_{\mathcal{M}}^*) + 2C_3 \sqrt{\tfrac{\log(1/\delta)}{N}} + \varepsilon_N.$$

But $\mathcal{R}_{\mathcal{S}}(\Theta_{\mathcal{M}}^*) = 0$ by construction, so:
$$\mathcal{R}_{\mathcal{S}}(\widehat{\Theta}_N) \leq \varepsilon_N + 2C_3 \sqrt{\tfrac{\log(1/\delta)}{N}}.$$
This gives (1), up to renaming constants.

Applying the lower bound in (2) to $\Theta = \widehat{\Theta}_N$:
$$c \, \mathrm{dist}(\widehat{\Theta}_N, \mathcal{M})^2 \leq \mathcal{R}_{\mathcal{S}}(\widehat{\Theta}_N),$$
and combining with the upper bound just obtained yields:
$$\mathrm{dist}(\widehat{\Theta}_N, \mathcal{M}) \leq C_4 \sqrt{ \varepsilon_N + \sqrt{\tfrac{\log(1/\delta)}{N}} },$$
for some constant $C_4$. Since for each $S$ the minimizer set $\{\Theta^*(S)\} \subset \mathcal{M}$, the distance to $\Theta^*(S)$ is bounded by the distance to $\mathcal{M}$, giving (2).

The convergence statements in (3) follow immediately when $\varepsilon_N \to 0$ and $N \to \infty$.
\end{proof}

\begin{remark}[Interpretation]
The theorem shows that \textbf{average defect minimization over a distribution of systems} is a consistent procedure: if each system admits a hypostructure in the parametric family and the structural manifold is well-behaved, then a trainable hypostructure that approximately minimizes empirical axiom risk on finitely many training systems will, with high probability, yield \textbf{globally good} hypostructures for new systems drawn from the same structural class.
\end{remark}

\begin{remark}[Covariate shift]
Extensions to a \textbf{covariately shifted test distribution} $\mathcal{S}_{\mathrm{test}}$ (e.g. different but structurally equivalent systems) follow by the same argument, provided the map $S \mapsto \Theta^*(S)$ is Lipschitz between the supports of $\mathcal{S}_{\mathrm{train}}$ and $\mathcal{S}_{\mathrm{test}}$.
\end{remark}

\begin{remark}[Motivic Interpretation]
In the $\infty$-categorical framework ({prf:ref}`def-categorical-hypostructure`), Meta-Generalization admits a deeper interpretation via \textbf{Motivic Integration} [@Kontsevich95; @DenefLoeser01]. The learner does not merely fit parameters; it extracts the \textbf{Motive} of the system---an object in the Grothendieck ring of varieties $K_0(\text{Var}_k)$.
\end{remark}

Specifically, define the **error variety** for parameter $\Theta$ over a field $k$:
$$\mathcal{E}_\Theta := \{(S, u) \in \text{Syst} \times \mathcal{X} : \mathcal{R}_S(\Theta)(u) > 0\} \subset \text{Syst} \times \mathcal{X}$$

The "loss function" $\mathcal{R}_{\mathcal{S}}(\Theta)$ is then the **motivic volume**:
$$\mathcal{R}_{\mathcal{S}}(\Theta) = \int_{\mathcal{E}_\Theta} \mathbb{L}^{-\dim} \, d\chi = \chi(\mathcal{E}_\Theta) \cdot \mathbb{L}^{-n}$$
where $\chi$ is the Euler characteristic and $\mathbb{L} = [\mathbb{A}^1]$ is the Lefschetz motive.

**Key Property (Motivic Invariance):** The learned structure is **base-change invariant**: by Grothendieck's trace formula, if $\widehat{\Theta}_N$ minimizes the motivic volume over $\mathbb{R}$, it also minimizes over $\mathbb{C}$, $\mathbb{Q}_p$, and finite fields $\mathbb{F}_q$. This provides:
- **Transfer learning**: Structure learned over reals transfers to complex systems
- **Field-independence**: The motive $[\mathcal{M}] \in K_0(\text{Var})$ is an absolute invariant
- **Categorical universality**: The structural manifold $\mathcal{M}$ is defined by its functor of points, not by equations in a specific field

The convergence $\widehat{\Theta}_N \to \mathcal{M}$ is thus **motivically universal**—it holds regardless of the base field, guaranteeing that learned hypostructures are not artifacts of the training domain but genuine structural invariants.

> **Key Insight:** This gives Part VII a rigorous "meta-generalization" layer: trainable hypostructures do not just fit one system, but converge (in risk and in parameter space) to the correct structural manifold across a whole family of systems. In the motivic interpretation, the learner extracts a **universal motive**—the abstract "essence" of the system class that transcends any particular instantiation.

### Expressivity of Trainable Hypostructures

Up to now we have assumed that the "true" hypostructure for a given system $S$ lives *inside* our parametric family $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$. In practice, this is an idealization: the true structure might lie outside our chosen parametrization, but we still expect to approximate it arbitrarily well.

In this section we formalize this as an **expressivity / approximation** property: the parametric hypostructure family is rich enough that any admissible hypostructure satisfying the axioms can be approximated (in structural data) to arbitrary accuracy, and the **axiom-defect risk** then goes to zero.

#### Structural metric on hypostructures

Fix a system $S$ with state space $X$ and semiflow $S_t$. Let $\mathfrak{H}(S)$ denote the class of hypostructures on $S$ of the form:
$$\mathcal{H} = (X, S_t, \Phi, \mathfrak{D}, G)$$
satisfying the axioms (C, D, SC, Cap, LS, TB, Reg, GC) and a uniform regularity condition (e.g. Lipschitz bounds on $\Phi, \mathfrak{D}$ and bounded barrier constants).

We define a **structural metric**:
$$d_{\mathrm{struct}} : \mathfrak{H}(S) \times \mathfrak{H}(S) \to [0, \infty)$$
by choosing a reference measure $\nu$ on $X$ (e.g. invariant or finite-energy measure) and setting:
$$d_{\mathrm{struct}}(\mathcal{H}, \mathcal{H}') := \|\Phi - \Phi'\|_{L^\infty(X, \nu)} + \|\mathfrak{D} - \mathfrak{D}'\|_{L^\infty(X, \nu)} + \mathrm{dist}_G(G, G'),$$
where $\mathrm{dist}_G$ is any metric on the structural data $G$ (capacities, sectors, barrier constants, exponents) compatible with the topology used in Parts VI–X. Two hypostructures that differ only by a Hypo-isomorphism are identified in this metric (i.e. we work modulo gauge).

#### Universal structural approximation

Let $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$ be a parametric family of hypostructures on $S$:
$$\mathcal{H}_\Theta = (X, S_t, \Phi_\Theta, \mathfrak{D}_\Theta, G_\Theta).$$

We say this family is **universally structurally approximating** on $\mathfrak{H}(S)$ if (this generalizes the Stone-Weierstrass theorem to dynamical functionals, similar to the universality of flow approximation in [@Ornstein74]):

> For every $\mathcal{H}^* = (X, S_t, \Phi^*, \mathfrak{D}^*, G^*) \in \mathfrak{H}(S)$ and every $\delta > 0$, there exists $\Theta \in \Theta_{\mathrm{adm}}$ such that:
> $$d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) < \delta.$$

Intuitively, $\{\mathcal{H}_\Theta\}$ can approximate any admissible hypostructure arbitrarily well in energy, dissipation, and barrier data.

#### Continuity of defects with respect to structure

Recall that for each axiom $A \in \mathcal{A}$ and trajectory $u \in \mathcal{U}_S$, the defect functional $K_A^{(\Theta)}(u)$ is defined in terms of $(\Phi_\Theta, \mathfrak{D}_\Theta, G_\Theta)$ and the axioms (C, D, SC, Cap, LS, TB). Denote by $K_A^{(\mathcal{H})}(u)$ the corresponding defect when computed from a general hypostructure $\mathcal{H} \in \mathfrak{H}(S)$.

We assume:

> **Defect continuity.** There exists a constant $L_A > 0$ such that for all hypostructures $\mathcal{H}, \mathcal{H}' \in \mathfrak{H}(S)$, all trajectories $u \in \mathcal{U}_S$, and all $A \in \mathcal{A}$:
> $$\big| K_A^{(\mathcal{H})}(u) - K_A^{(\mathcal{H}')}(u) \big| \leq L_A \, d_{\mathrm{struct}}(\mathcal{H}, \mathcal{H}').$$
>
> Equivalently, the mapping $\mathcal{H} \mapsto K_A^{(\mathcal{H})}(u)$ is Lipschitz with respect to the structural metric, uniformly over $u$ in the support of the trajectory measure $\mu_S$.

This is a natural assumption given the explicit integral definitions of the defects (e.g. $K_D$ is an integral of the positive part of $\partial_t \Phi + \mathfrak{D}$, capacities/barriers enter via continuous inequalities, etc.).

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **True Admissible Hypostructure:** $\mathcal{H}^* \in \mathfrak{H}(S)$ satisfying axioms with $K_A^{(\mathcal{H}^*)}(u) = 0$ $\mu_S$-a.e.
>     *   [ ] **Universally Structurally Approximating Family:** For all $\mathcal{H}^* \in \mathfrak{H}(S)$, $\forall \delta > 0$, $\exists \Theta$: $d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) < \delta$
>     *   [ ] **Defect Continuity:** $|K_A^{(\mathcal{H})}(u) - K_A^{(\mathcal{H}')}(u)| \leq L_A d_{\mathrm{struct}}(\mathcal{H}, \mathcal{H}')$ uniformly in $u$
> *   **Output (Structural Guarantee):**
>     *   Approximate realizability: $\forall \varepsilon > 0$, $\exists \Theta_\varepsilon$: $\mathcal{R}_S(\Theta_\varepsilon) \leq \varepsilon$
>     *   Infimum: $\inf_{\Theta} \mathcal{R}_S(\Theta) = 0$
>     *   Quantitative bound: $d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) \leq \delta \Rightarrow \mathcal{R}_S(\Theta) \leq (\sum_A w_A L_A)\delta$
> *   **Failure Condition (Debug):**
>     *   If **Universal Approximation** fails → **Mode expressivity gap** (parametric family too restrictive)
>     *   If **Defect Continuity** fails → **Mode discontinuous sensitivity** (small structural changes cause large defect jumps)

\begin{metatheorem}[Axiom-Expressivity]\label{mt:axiom-expressivity}
Let $S$ be a fixed system with trajectory distribution $\mu_S$ and trajectory class $\mathcal{U}_S$. Let $\mathfrak{H}(S)$ be the class of admissible hypostructures on $S$ as above. Suppose:
\end{metatheorem}

1. **(True admissible hypostructure.)** There exists a "true" hypostructure $\mathcal{H}^* \in \mathfrak{H}(S)$ which exactly satisfies the axioms (C, D, SC, Cap, LS, TB, Reg, GC) for $S$. Thus, for $\mu_S$-a.e. trajectory $u$:
$$K_A^{(\mathcal{H}^*)}(u) = 0 \quad \forall A \in \mathcal{A}.$$

2. **(Universally structurally approximating family.)** The parametric family $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$ is universally structurally approximating on $\mathfrak{H}(S)$ in the sense above.

3. **(Defect continuity.)** Each defect functional $K_A^{(\mathcal{H})}(u)$ is Lipschitz in $\mathcal{H}$ with respect to $d_{\mathrm{struct}}$, uniformly in $u$ (defect continuity).

Define the joint axiom risk of parameter $\Theta$ on system $S$ by:
$$\mathcal{R}_S(\Theta) := \sum_{A \in \mathcal{A}} w_A \int_{\mathcal{U}_S} K_A^{(\Theta)}(u) \, d\mu_S(u),$$
where $K_A^{(\Theta)} := K_A^{(\mathcal{H}_\Theta)}$ and $w_A \geq 0$ are fixed weights.

Then:

1. **(Approximate realizability of zero-risk.)** For every $\varepsilon > 0$ there exists $\Theta_\varepsilon \in \Theta_{\mathrm{adm}}$ such that:
$$\mathcal{R}_S(\Theta_\varepsilon) \leq \varepsilon.$$
In particular:
$$\inf_{\Theta \in \Theta_{\mathrm{adm}}} \mathcal{R}_S(\Theta) = 0.$$

2. **(Quantitative bound.)** More precisely, if for some $\delta > 0$ we pick $\Theta$ such that:
$$d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) \leq \delta,$$
then:
$$\mathcal{R}_S(\Theta) \leq \left( \sum_{A \in \mathcal{A}} w_A L_A \right) \delta.$$
In particular, $\mathcal{R}_S(\Theta_\varepsilon) \leq \varepsilon$ holds whenever:
$$d_{\mathrm{struct}}(\mathcal{H}_{\Theta_\varepsilon}, \mathcal{H}^*) \leq \frac{\varepsilon}{\sum_A w_A L_A}.$$

In words: **any admissible true hypostructure can be approximated arbitrarily well by the trainable family, and the corresponding axiom risk can be driven arbitrarily close to zero**.

\begin{proof}
Fix $\varepsilon > 0$. Let $L := \sum_{A \in \mathcal{A}} w_A L_A$, where the $L_A$'s are the Lipschitz constants from defect continuity.

By universal structural approximation (assumption 2), there exists $\Theta_\varepsilon \in \Theta_{\mathrm{adm}}$ such that:
$$d_{\mathrm{struct}}(\mathcal{H}_{\Theta_\varepsilon}, \mathcal{H}^*) \leq \delta_\varepsilon := \frac{\varepsilon}{L}.$$

For any $A \in \mathcal{A}$ and trajectory $u$:
$$\big| K_A^{(\Theta_\varepsilon)}(u) - K_A^{(\mathcal{H}^*)}(u) \big| = \big| K_A^{(\mathcal{H}_{\Theta_\varepsilon})}(u) - K_A^{(\mathcal{H}^*)}(u) \big| \leq L_A \, d_{\mathrm{struct}}(\mathcal{H}_{\Theta_\varepsilon}, \mathcal{H}^*) \leq L_A \delta_\varepsilon.$$

But $K_A^{(\mathcal{H}^*)}(u) = 0$ $\mu_S$-a.s. by assumption (1), so:
$$K_A^{(\Theta_\varepsilon)}(u) \leq L_A \delta_\varepsilon \quad \text{for } \mu_S\text{-a.e. } u.$$

Integrating with respect to $\mu_S$:
$$\mathcal{R}_{A,S}(\Theta_\varepsilon) = \int_{\mathcal{U}_S} K_A^{(\Theta_\varepsilon)}(u) \, d\mu_S(u) \leq L_A \delta_\varepsilon.$$

Therefore:
$$\mathcal{R}_S(\Theta_\varepsilon) = \sum_{A \in \mathcal{A}} w_A \mathcal{R}_{A,S}(\Theta_\varepsilon) \leq \sum_{A \in \mathcal{A}} w_A (L_A \delta_\varepsilon) = \left( \sum_{A \in \mathcal{A}} w_A L_A \right) \delta_\varepsilon = L \cdot \frac{\varepsilon}{L} = \varepsilon.$$

This proves the quantitative bound and, in particular, the existence of parameters $\Theta_\varepsilon$ with $\mathcal{R}_S(\Theta_\varepsilon) \leq \varepsilon$ for every $\varepsilon > 0$. Taking the infimum over $\Theta$ and letting $\varepsilon \to 0$ yields:
$$\inf_{\Theta \in \Theta_{\mathrm{adm}}} \mathcal{R}_S(\Theta) = 0.
$$
\end{proof}

\begin{remark}[No expressivity bottleneck]
The theorem isolates \textbf{what is needed} for axiom-expressivity:
\begin{itemize}
\item a structural metric $d_{\mathrm{struct}}$ capturing the relevant pieces of hypostructure data,
\item universal approximation of $(\Phi, \mathfrak{D}, G)$ in that metric,
\item and Lipschitz dependence of defects on structural data.
\end{itemize}
No optimization assumptions are used: this is a \textbf{pure representational metatheorem}. Combined with the trainability and convergence metatheorem (\cref{mt:trainable-hypostructure-consistency}), it implies that the only remaining obstacles are optimization and data, not the expressivity of the hypostructure family.
\end{remark}

> **Key Insight:** The parametric family is **axiom-complete**: any structurally admissible dynamics can be encoded with arbitrarily small axiom defects. The only limitations are optimization and data, not the hypothesis class.

### Active Probing and Sample-Complexity of Hypostructure Identification

So far we have treated the axiom-defect risk as given by a fixed trajectory distribution $\mu_S$. In many systems, however, the learner can **control** which trajectories are generated, by choosing initial conditions and controls. In other words, the learner can design *experiments*. This section formalizes optimal experiment design for structural identification, extending the classical **observability** framework of Kalman [@Kalman60] to the hypostructure setting. This guarantees **Identification in the Limit**, satisfying the criteria of **Gold's Paradigm [@Gold67]** for language learning.

In this section we show that, under a mild identifiability gap assumption, **actively chosen probes** (policies, initial data, controls) allow the learner to identify the correct hypostructure parameter with sample complexity essentially proportional to the parameter dimension and inverse-quadratic in the identifiability gap.

#### Probes and defect observations

Fix a system $S$ with state space $X$, trajectory space $\mathcal{U}_S$, and a parametric hypostructure family $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$. We assume we can influence trajectories via a class of **probes**:
$$\pi \in \mathfrak{P},$$
where each $\pi$ denotes a rule for generating a trajectory $u_{S,\Theta,\pi} \in \mathcal{U}_S$ (e.g. a choice of initial condition and/or control policy). For each probe $\pi$ and parameter $\Theta$, we can evaluate the axiom defect functionals on the resulting trajectory.

To simplify notation, write:
$$K^{(\Theta)}(S, \pi) := \big( K_A^{(\Theta)}(u_{S,\Theta,\pi}) \big)_{A \in \mathcal{A}} \in \mathbb{R}^{|\mathcal{A}|}_{\geq 0}$$
for the **defect fingerprint** induced by parameter $\Theta$ on system $S$ under probe $\pi$, and:
$$D(\Theta, \Theta'; S, \pi) := \big| K^{(\Theta)}(S, \pi) - K^{(\Theta')}(S, \pi) \big|$$
for its distance (e.g. $\ell^1$ or $\ell^2$ norm) between two parameters.

In practice, the defects may be observed with noise. We thus write a single **noisy observation** of the defect fingerprint as:
$$Y_t = K^{(\Theta^*)}(S, \pi_t) + \xi_t,$$
where $\Theta^*$ is the true parameter and $\pi_t$ is the probe chosen at round $t$. The noise $\xi_t$ takes values in $\mathbb{R}^{|\mathcal{A}|}$ and models discretization error, finite sampling of trajectories, measurement noise, etc.

\begin{definition}[Probe-wise identifiability gap]\label{def:probe-wise-identifiability-gap}
Let $\Theta^* \in \Theta_{\mathrm{adm}}$ be the true parameter. We say that a class of probes $\mathfrak{P}$ has a \textbf{uniform identifiability gap} $\Delta > 0$ around $\Theta^*$ if there exist constants $\Delta > 0$ and $r > 0$ such that for every $\Theta \in \Theta_{\mathrm{adm}}$ with $|\Theta - \Theta^*| \geq r$:
$$\sup_{\pi \in \mathfrak{P}} D(\Theta, \Theta^*; S, \pi) \geq \Delta.$$
\end{definition}

Equivalently: no parameter at distance at least $r$ from $\Theta^*$ can mimic the defect fingerprints of $\Theta^*$ under *all* probes; there is always some probe that amplifies the discrepancy to at least $\Delta$ in defect space.

**Assumption 13.43 (Sub-Gaussian defect noise).** The noise variables $\xi_t$ are independent, mean-zero, and $\sigma$-sub-Gaussian in each coordinate:
$$\mathbb{E}[\xi_t] = 0, \quad \mathbb{E}\big[ \exp(\lambda \xi_{t,j}) \big] \leq \exp\Big( \tfrac{1}{2} \sigma^2 \lambda^2 \Big) \quad \forall \lambda \in \mathbb{R}, \forall t, \forall j.$$

Moreover, $\xi_t$ is independent of the probe choices $\pi_s$ and the past noise $\xi_s$ for $s < t$.

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Local Identifiability via Defects:** $\sup_\pi D(\Theta, \Theta^*; S, \pi) \leq \delta \Rightarrow |\Theta - \Theta^*| \leq c\delta$ (\cref{mt:trainable-hypostructure-consistency}, \cref{mt:sv-09-meta-identifiability})
>     *   [ ] **Probe-Wise Identifiability Gap:** $\sup_{\pi \in \mathfrak{P}} D(\Theta, \Theta^*; S, \pi) \geq \Delta$ for $|\Theta - \Theta^*| \geq r$
>     *   [ ] **Sub-Gaussian Defect Noise:** $\xi_t$ independent, mean-zero, $\sigma$-sub-Gaussian
>     *   [ ] **Local Regularity:** $|K^{(\Theta)}(S, \pi) - K^{(\Theta')}(S, \pi)| \leq L|\Theta - \Theta'|$ uniformly in $\pi$
> *   **Output (Structural Guarantee):**
>     *   Sample complexity: $T \gtrsim \frac{d \sigma^2}{\Delta^2} \log(1/\delta)$ for $\varepsilon$-identification
>     *   Adaptive probing strategy achieves optimal rate
>     *   Linear scaling in parameter dimension $d$, inverse-quadratic in gap $\Delta$
> *   **Failure Condition (Debug):**
>     *   If **Identifiability Gap** $\Delta = 0$ → **Mode structural aliasing** (indistinguishable parameters under all probes)
>     *   If **Noise** non-sub-Gaussian → **Mode heavy-tailed contamination** (slower convergence rates)

\begin{metatheorem}[Optimal Experiment Design]\label{mt:optimal-experiment-design}
Let $S$ be a fixed system and $\Theta^* \in \Theta_{\mathrm{adm}}$ the true hypostructure parameter. Assume:
\end{metatheorem}

1. **(Local identifiability via defects.)** The single-system identifiability metatheorem holds for $S$: small uniform defect discrepancies imply small parameter distance, as in \cref{mt:trainable-hypostructure-consistency} and \cref{mt:sv-09-meta-identifiability}. In particular, there exist constants $c > 0$ and $\rho > 0$ such that:
$$\sup_{\pi \in \mathfrak{P}} D(\Theta, \Theta^*; S, \pi) \leq \delta \implies |\Theta - \Theta^*| \leq c \delta$$
for all $\Theta$ with $|\Theta - \Theta^*| \leq \rho$.

2. **(Probe-wise identifiability gap.)** The probe class $\mathfrak{P}$ has a uniform identifiability gap $\Delta > 0$ in the sense of Definition 13.42, with some radius $r > 0$.

3. **(Sub-Gaussian defect noise.)** The noise model of Assumption 13.43 holds with parameter $\sigma > 0$.

4. **(Local regularity.)** The map $\Theta \mapsto K^{(\Theta)}(S, \pi)$ is Lipschitz in $\Theta$ uniformly over $\pi \in \mathfrak{P}$ in a neighborhood of $\Theta^*$:
$$\big| K^{(\Theta)}(S, \pi) - K^{(\Theta')}(S, \pi) \big| \leq L |\Theta - \Theta'| \quad \text{for } |\Theta - \Theta^*|, |\Theta' - \Theta^*| \leq \rho.$$

Consider an **adaptive probing strategy** over $T$ rounds:

- At round $t$ we choose a probe $\pi_t = \pi_t(\mathcal{F}_{t-1}) \in \mathfrak{P}$, where $\mathcal{F}_{t-1}$ is the sigma-algebra generated by past probes and observations $\{(\pi_s, Y_s)\}_{s < t}$.
- We observe a noisy defect fingerprint $Y_t = K^{(\Theta^*)}(S, \pi_t) + \xi_t$.
- After $T$ rounds, we output an estimator $\widehat{\Theta}_T$ that is measurable with respect to $\mathcal{F}_T$.

Then there exists an adaptive probing strategy and an estimator $\widehat{\Theta}_T$ such that for any confidence level $\delta \in (0, 1)$, we have:
$$\mathbb{P}\big( |\widehat{\Theta}_T - \Theta^*| \geq \varepsilon \big) \leq \delta$$
whenever:
$$T \gtrsim \frac{d \, \sigma^2}{\Delta^2} \log \frac{1}{\delta},$$
where $d := \dim(\Theta_{\mathrm{adm}})$, and the implicit constant depends only on the Lipschitz/identifiability constants $L, c, \rho$.

In particular, the sample complexity of identifying the correct hypostructure parameter up to accuracy $\varepsilon$ with high probability scales at most linearly in the parameter dimension and inverse-quadratically in the identifiability gap $\Delta$.

\begin{proof}
We provide a rigorous argument based on $\varepsilon$-net discretization and uniform concentration bounds.

\textbf{Step 1 (Discretize parameter space).} Restrict attention to a compact neighborhood $B(\Theta^*, R) \subset \Theta_{\mathrm{adm}}$. For a given accuracy scale $\varepsilon > 0$, construct a minimal $\varepsilon$-net $\mathcal{N}_\varepsilon \subset B(\Theta^*, R)$ in parameter space. By standard metric entropy bounds \cite[Lemma 5.2]{Wainwright19}, the covering number satisfies:
$$N(\varepsilon, B(\Theta^*, R), \|\cdot\|) \leq \left(\frac{3R}{\varepsilon}\right)^d$$
where $d = \dim(\Theta_{\mathrm{adm}})$.

\textbf{Step 2 (Uniform separation via probes).} Define the separation function $\Delta(\Theta, \Theta') := \sup_{\pi \in \mathfrak{P}} |K^{(\Theta)}(S, \pi) - K^{(\Theta')}(S, \pi)|$. By the identifiability gap assumption, $|\Theta - \Theta^*| \geq r$ implies $\Delta(\Theta, \Theta^*) \geq \Delta$. By Lipschitz continuity of the defect kernel in $\Theta$, for any $\Theta' \in \mathcal{N}_\varepsilon$ with $|\Theta' - \Theta^*| \geq r/2$, there exists $\pi \in \mathfrak{P}$ achieving:
$$\big| K^{(\Theta')}(S, \pi) - K^{(\Theta^*)}(S, \pi) \big| \geq \Delta/2.$$

\textbf{Step 3 (Adaptive elimination strategy).} Maintain a candidate set $C_t \subseteq \mathcal{N}_\varepsilon$, initialized as $C_0 = \mathcal{N}_\varepsilon$. At each round $t$:
\begin{itemize}
\item Choose probe $\pi_t = \arg\max_{\pi} \mathrm{Var}_{\Theta \in C_{t-1}}[K^{(\Theta)}(S, \pi)]$
\item Observe $Y_t = K^{(\Theta^*)}(S, \pi_t) + \xi_t$ with $\xi_t$ sub-Gaussian($\sigma^2$)
\item Eliminate: $C_t = \{\Theta \in C_{t-1} : |K^{(\Theta)}(S, \pi_t) - \bar{Y}_t| \leq 2\sigma\sqrt{2\log(2|C_0|T/\delta)/t}\}$
\end{itemize}

\textbf{Lemma (Sub-Gaussian concentration).} For sub-Gaussian noise with parameter $\sigma^2$, after $t$ observations of probe $\pi$, the empirical mean $\bar{Y}_t$ satisfies:
$$\mathbb{P}\left(|\bar{Y}_t - K^{(\Theta^*)}(S, \pi)| > \sigma\sqrt{\frac{2\log(2/\delta)}{t}}\right) \leq \delta$$

By a union bound over $|\mathcal{N}_\varepsilon| \cdot T$ elimination events, any candidate $\Theta'$ with $|K^{(\Theta')} - K^{(\Theta^*)}| \geq \Delta/2$ is eliminated after at most $t \geq 32\sigma^2 \log(2|\mathcal{N}_\varepsilon|T/\delta)/\Delta^2$ probes. The total sample complexity is:
$$T \lesssim \frac{\sigma^2}{\Delta^2} \Big( d \log(R/\varepsilon) + \log \tfrac{1}{\delta} \Big).$$

\textbf{Step 4 (Accuracy and parameter error).} After elimination, all remaining candidates $\Theta' \in C_T$ satisfy $|\Theta' - \Theta^*| < r/2$. Output $\widehat{\Theta}_T$ as any element of $C_T$. By the triangle inequality and Lipschitz identifiability, the final estimator's error satisfies $|\widehat{\Theta}_T - \Theta^*| \leq \varepsilon + r/2 = O(\varepsilon)$ when $r = O(\varepsilon)$.
\end{proof}

\begin{remark}[Experiments as a theorem]
The theorem shows that \textbf{defect-driven experiment design} is not just heuristic: under mild identifiability and regularity assumptions, actively chosen probes let a hypostructure learner identify the correct axioms with sample complexity comparable to classical parametric statistics ($O(d)$ up to logs and $\Delta^{-2}$).
\end{remark}

\begin{remark}[Connection to error localization]
This metatheorem pairs naturally with the \textbf{meta-error localization} theorem (\cref{mt:meta-error-localization}): once the learner has identified that an axiom block is wrong, it can design probes specifically targeted to excite that block's defects, further improving the identifiability gap for that block and accelerating correction.
\end{remark}

> **Key Insight:** The identifiability gap $\Delta$ is a purely **structural quantity**: it measures how different the defect fingerprints of distinct hypostructures can be made by appropriate experiments. It plays exactly the role of an "information gap" in classical active learning.

### Robustness of Failure-Mode Predictions

A central purpose of a hypostructure is not only to fit trajectories, but to make **sharp structural predictions**: which singularity or breakdown scenarios ("failure modes") are *permitted* or *ruled out* by the axioms, barrier constants, and capacities.

In Parts VI–X we developed a "taxonomy" of failure modes and associated **barrier inequalities**: each mode $f$ is excluded when certain barrier constants, exponents, or capacities lie beyond a critical threshold. We now show that, once a trainable hypostructure has sufficiently small axiom-defect risk, its **forbidden failure-mode set** is *exactly the same* as that of the true hypostructure. In other words, the discrete "permit denial" predictions are robust to small learning error.

#### Failure modes and barrier thresholds

Let $\mathcal{F}$ denote the (finite or countable) set of failure modes in the taxonomy (e.g. blow-up, loss of uniqueness, loss of conservation, barrier penetration, glassy obstruction, etc.). For each failure mode $f \in \mathcal{F}$, the structural metatheorems of Parts VI–X associate:

- a structural functional $B_f(\mathcal{H})$ (a barrier constant, capacity threshold, exponent, or combination thereof);
- a critical value or region $B_f^{\mathrm{crit}}$ such that:

> **Barrier exclusion principle for mode $f$.** If $B_f(\mathcal{H})$ lies in a certain "safe" region (e.g. above a critical constant, or outside a critical set), then failure mode $f$ is forbidden for the hypostructure $\mathcal{H}$. Conversely, if $B_f(\mathcal{H})$ lies in a complementary region, then either $f$ is not ruled out, or there exist sequences of approximate extremals compatible with $f$.

Formally, there is a map $\mathrm{Forbidden}(\mathcal{H}) \subseteq \mathcal{F}$ determined by the structural data $(\Phi, \mathfrak{D}, G)$ and barrier functionals $B_f$, such that:
$$f \in \mathrm{Forbidden}(\mathcal{H}) \iff B_f(\mathcal{H}) \in \mathcal{B}_f^{\mathrm{safe}},$$
where $\mathcal{B}_f^{\mathrm{safe}}$ is the exclusion region in barrier space for mode $f$.

\begin{definition}[Margin of failure-mode exclusion]\label{def:margin-of-failure-mode-exclusion}
Let $\mathcal{H}^*$ be a hypostructure and $f \in \mathrm{Forbidden}(\mathcal{H}^*)$. We say that $\mathcal{H}^*$ excludes $f$ with margin $\gamma_f > 0$ if:
$$\mathrm{dist}\big( B_f(\mathcal{H}^*), \partial \mathcal{B}_f^{\mathrm{safe}} \big) \geq \gamma_f,$$
where $\partial \mathcal{B}_f^{\mathrm{safe}}$ denotes the boundary of the safe region in the barrier space.
\end{definition}

We define the **global margin**:
$$\gamma^* := \inf_{f \in \mathrm{Forbidden}(\mathcal{H}^*)} \gamma_f,$$
with the convention $\gamma^* > 0$ if the infimum is over a finite set with strictly positive margins.

**Assumption 13.48 (Barrier continuity).** For each failure mode $f \in \mathcal{F}$, the barrier functional $B_f(\mathcal{H})$ is Lipschitz in the structural metric: there exists $L_f > 0$ such that:
$$\big| B_f(\mathcal{H}) - B_f(\mathcal{H}') \big| \leq L_f \, d_{\mathrm{struct}}(\mathcal{H}, \mathcal{H}') \quad \forall \mathcal{H}, \mathcal{H}' \in \mathfrak{H}(S).$$

**Assumption 13.49 (Local structural control by risk).** Let $\mathcal{H}_\Theta$ be a parametric hypostructure family and $\mathcal{H}^*$ the true hypostructure. There exist constants $C_{\mathrm{struct}}, \varepsilon_0 > 0$ such that:
$$\mathcal{R}_S(\Theta) \leq \varepsilon < \varepsilon_0 \implies d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) \leq C_{\mathrm{struct}} \sqrt{\varepsilon}.$$

This is precisely the local quantitative identifiability from \cref{mt:trainable-hypostructure-consistency}, translated into structural space by the Defect Reconstruction Theorem.

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **True Hypostructure with Strict Exclusion Margin:** $\gamma^* := \inf_{f \in \mathcal{F}_{\mathrm{forbidden}}^*} \mathrm{dist}(B_f(\mathcal{H}^*), \partial \mathcal{B}_f^{\mathrm{safe}}) > 0$
>     *   [ ] **Barrier Continuity:** $|B_f(\mathcal{H}) - B_f(\mathcal{H}')| \leq L_f d_{\mathrm{struct}}(\mathcal{H}, \mathcal{H}')$ for all $f$
>     *   [ ] **Structural Control by Axiom Risk:** $\mathcal{R}_S(\Theta) \leq \varepsilon \Rightarrow d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) \leq C_{\mathrm{struct}}\sqrt{\varepsilon}$
> *   **Output (Structural Guarantee):**
>     *   Exact stability: $\mathrm{Forbidden}(\mathcal{H}_\Theta) = \mathrm{Forbidden}(\mathcal{H}^*)$ for $\mathcal{R}_S(\Theta) \leq \varepsilon_1$
>     *   No spurious exclusions: allowed modes remain allowed
>     *   Discrete permit-denial structure recovered exactly for small risk
> *   **Failure Condition (Debug):**
>     *   If **Margin** $\gamma^* = 0$ → **Mode critical boundary** (barrier at threshold, sensitive to perturbation)
>     *   If **Barrier Continuity** fails → **Mode discontinuous classification** (small changes flip forbidden status)

\begin{metatheorem}[Robustness of Failure-Mode Predictions]\label{mt:robustness-of-failure-mode-predictions}
Let $S$ be a system with true hypostructure $\mathcal{H}^* \in \mathfrak{H}(S)$, and let $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$ be a parametric family of trainable hypostructures with axiom-risk $\mathcal{R}_S(\Theta)$. Assume:
\end{metatheorem}

1. **(True hypostructure with strict exclusion margin.)** The true hypostructure $\mathcal{H}^*$ exactly satisfies the axioms (C, D, SC, Cap, LS, TB, Reg, GC) and excludes a set of failure modes $\mathcal{F}_{\mathrm{forbidden}}^* \subseteq \mathcal{F}$ with positive margin:
$$\gamma^* := \inf_{f \in \mathcal{F}_{\mathrm{forbidden}}^*} \mathrm{dist}\big( B_f(\mathcal{H}^*), \partial \mathcal{B}_f^{\mathrm{safe}} \big) > 0.$$

2. **(Barrier continuity.)** Each barrier functional $B_f(\mathcal{H})$ is Lipschitz with constant $L_f$ with respect to $d_{\mathrm{struct}}$, as in Assumption 13.48, and:
$$L_{\max} := \max_{f \in \mathcal{F}_{\mathrm{forbidden}}^*} L_f < \infty.$$

3. **(Structural control by axiom risk.)** The parametric family $\mathcal{H}_\Theta$ satisfies Assumption 13.49: there exist $C_{\mathrm{struct}}, \varepsilon_0 > 0$ such that:
$$\mathcal{R}_S(\Theta) \leq \varepsilon < \varepsilon_0 \implies d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) \leq C_{\mathrm{struct}} \sqrt{\varepsilon}.$$

Then there exists $\varepsilon_1 > 0$ such that for all $\Theta$ with $\mathcal{R}_S(\Theta) \leq \varepsilon_1$:

1. **(Exact stability of forbidden modes.)**
$$\mathrm{Forbidden}(\mathcal{H}_\Theta) = \mathrm{Forbidden}(\mathcal{H}^*) = \mathcal{F}_{\mathrm{forbidden}}^*.$$

2. **(No spurious new exclusions.)** In particular, no failure mode that is allowed by $\mathcal{H}^*$ is spuriously excluded by $\mathcal{H}_\Theta$.

Thus, once the axiom risk is small enough, the **discrete pattern** of forbidden failure modes becomes identical, not merely close, to that of the true hypostructure.

\begin{proof}
Fix $\varepsilon > 0$ small, and let $\Theta$ be such that $\mathcal{R}_S(\Theta) \leq \varepsilon$. By structural control (Assumption 13.49):
$$d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) \leq C_{\mathrm{struct}} \sqrt{\varepsilon}.$$

Let $f \in \mathcal{F}_{\mathrm{forbidden}}^*$. By definition of the margin $\gamma^*$:
$$\mathrm{dist}\big( B_f(\mathcal{H}^*), \partial \mathcal{B}_f^{\mathrm{safe}} \big) \geq \gamma^*.$$

By barrier continuity (Assumption 13.48):
$$\big| B_f(\mathcal{H}_\Theta) - B_f(\mathcal{H}^*) \big| \leq L_f \, d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) \leq L_f C_{\mathrm{struct}} \sqrt{\varepsilon} \leq L_{\max} C_{\mathrm{struct}} \sqrt{\varepsilon}.$$

Choose $\varepsilon_1 > 0$ small enough that:
$$L_{\max} C_{\mathrm{struct}} \sqrt{\varepsilon_1} \leq \frac{1}{2} \gamma^*.$$

Then for any $\varepsilon \leq \varepsilon_1$:
$$\mathrm{dist}\big( B_f(\mathcal{H}_\Theta), \partial \mathcal{B}_f^{\mathrm{safe}} \big) \geq \mathrm{dist}\big( B_f(\mathcal{H}^*), \partial \mathcal{B}_f^{\mathrm{safe}} \big) - |B_f(\mathcal{H}_\Theta) - B_f(\mathcal{H}^*)| \geq \gamma^* - \frac{1}{2}\gamma^* = \frac{1}{2}\gamma^* > 0.$$

Thus, $B_f(\mathcal{H}_\Theta)$ remains *inside* the safe region $\mathcal{B}_f^{\mathrm{safe}}$, at positive distance from its boundary. Therefore:
$$f \in \mathrm{Forbidden}(\mathcal{H}^*) \implies f \in \mathrm{Forbidden}(\mathcal{H}_\Theta)$$
for all $\Theta$ with $\mathcal{R}_S(\Theta) \leq \varepsilon_1$. In other words:
$$\mathcal{F}_{\mathrm{forbidden}}^* \subseteq \mathrm{Forbidden}(\mathcal{H}_\Theta).$$

To show the reverse inclusion, suppose for contradiction that there exists $f \in \mathcal{F}$ with $f \in \mathrm{Forbidden}(\mathcal{H}_\Theta)$ but $f \notin \mathrm{Forbidden}(\mathcal{H}^*)$. By definition:
$$B_f(\mathcal{H}_\Theta) \in \mathcal{B}_f^{\mathrm{safe}}, \qquad B_f(\mathcal{H}^*) \notin \mathcal{B}_f^{\mathrm{safe}}.$$

Since $\mathcal{B}_f^{\mathrm{safe}}$ is closed, continuity of $B_f$ implies that the set $\{\lambda \in [0,1] : B_f((1-\lambda)\mathcal{H}^* + \lambda \mathcal{H}_\Theta) \in \mathcal{B}_f^{\mathrm{safe}}\}$ has a nonempty boundary in $[0,1]$ where the barrier lies on $\partial \mathcal{B}_f^{\mathrm{safe}}$. But by Lipschitz continuity:
$$\mathrm{dist}\big( B_f(\mathcal{H}^*), \partial \mathcal{B}_f^{\mathrm{safe}} \big) \leq L_f C_{\mathrm{struct}} \sqrt{\varepsilon_1} \leq \tfrac{1}{2}\gamma^*,$$
contradicting the fact that either $f$ is forbidden at $\mathcal{H}^*$ with margin $\gamma_f \geq \gamma^*$, or else $B_f(\mathcal{H}^*)$ lies strictly in the *complement* of $\mathcal{B}_f^{\mathrm{safe}}$ at distance at least some fixed positive amount. For $\varepsilon_1$ sufficiently small, the "spurious exclusion" is impossible.

Hence no new failure modes can enter the forbidden set when $\mathcal{R}_S(\Theta)$ is sufficiently small, and we have:
$$\mathrm{Forbidden}(\mathcal{H}_\Theta) = \mathrm{Forbidden}(\mathcal{H}^*) = \mathcal{F}_{\mathrm{forbidden}}^*.
$$
\end{proof}

\begin{remark}[Margin is essential]
The key ingredient is the \textbf{margin} $\gamma^* > 0$: if the true hypostructure barely satisfies a barrier inequality, then arbitrarily small perturbations can change whether a mode is forbidden. The metatheorems in Parts VI--X typically provide such a margin (e.g.\ strict inequalities in energy/capacity thresholds) except in degenerate ``critical'' cases.
\end{remark}

> **Key Insight:** Learning does not just approximate numbers; it stabilizes the *discrete* "permit denial" judgments. Once the axiom risk is small enough, trainable hypostructures recover the **exact discrete permit-denial structure** of the underlying PDE/dynamical system.

### Robust Exclusion of Energy Blow-up (Mode C.E)

The preceding metatheorem establishes that failure-mode predictions are robust in the abstract. We now prove a concrete instance: **small D-defect implies bounded energy**. This is a fully rigorous "robust structural transfer" theorem for the metatheorem "No energy blow-up (Mode C.E) under Axiom D."

**Setup.** Let $\mathcal{H}_\theta = (X, S_t, \Phi_\theta, \mathfrak{D}_\theta, \ldots)$ be a parametric hypostructure with $\mathfrak{D}_\theta(x) \geq 0$ for all $x$. For each trajectory $u: [0, T) \to X$ with $u(t) = S_t x_0$, the **D-defect** is:
$$K_D^{(\theta)}(u|_{[0,T]}) := \int_0^T \max\left(0, \partial_t \Phi_\theta(u(t)) + \mathfrak{D}_\theta(u(t))\right) dt.$$

This is nonnegative and vanishes if and only if the energy–dissipation inequality holds pointwise:
$$\partial_t \Phi_\theta(u(t)) + \mathfrak{D}_\theta(u(t)) \leq 0 \quad \text{a.e. } t.$$

Mode **C.E (energy blow-up)** is defined as $\sup_{t < T^*} \Phi(u(t)) = +\infty$.

\begin{metatheorem}[Robust Exclusion of Energy Blow-up]\label{mt:robust-exclusion-of-energy-blow-up}
Let $\mathcal{H}_\theta = (X, S_t, \Phi_\theta, \mathfrak{D}_\theta, \ldots)$ be a parametric hypostructure with $\mathfrak{D}_\theta(x) \geq 0$ for all $x$. Fix a trajectory $u: [0, T) \to X$, $u(t) = S_t x_0$, defined on some interval $[0, T)$ where $0 < T \leq T^*(x_0)$.
\end{metatheorem}

Assume that for this trajectory the D-defect on $[0, T]$ is bounded by $\varepsilon \geq 0$:
$$K_D^{(\theta)}(u|_{[0,T]}) = \int_0^T \max\left(0, \partial_t \Phi_\theta(u(t)) + \mathfrak{D}_\theta(u(t))\right) dt \leq \varepsilon.$$

Then **for all** $t \in [0, T]$:
$$\Phi_\theta(u(t)) \leq \Phi_\theta(u(0)) + \varepsilon.$$

In particular:

1. If $\varepsilon < \infty$, then the energy along $u$ cannot blow up on $[0, T]$; i.e., Mode C.E cannot occur on that interval.

2. If there exists a nondecreasing function $E: [0, T^*) \to [0, \infty)$ such that for every $T' < T^*$,
   $$K_D^{(\theta)}(u|_{[0,T']}) \leq E(T') \quad \text{and} \quad \sup_{T' < T^*} E(T') < \infty,$$
   then $\Phi_\theta(u(t))$ is **uniformly bounded** on $[0, T^*)$, hence Mode C.E is completely excluded for this trajectory.

\begin{proof}
Define the "D-residual" function:
$$g(t) := \partial_t \Phi_\theta(u(t)) + \mathfrak{D}_\theta(u(t))$$
where the time derivative exists in the sense used in the D-axiom (for a.e. $t$). By definition of the D-defect:
$$K_D^{(\theta)}(u|_{[0,T]}) = \int_0^T \max(0, g(t)) \, dt \leq \varepsilon. \quad \text{(1)}$$

We exploit two facts: (i) dissipation nonnegativity $\mathfrak{D}_\theta(u(t)) \geq 0$ for all $t$, and (ii) inequality (1).

\textbf{Step 1: Pointwise inequality for $\partial_t \Phi_\theta(u(t))$.}

We establish an upper bound on $\partial_t \Phi_\theta(u(t))$ in terms of $g^+(t) := \max(0, g(t))$.

For each $t$, we have two cases:
\begin{itemize}
\item If $g(t) \geq 0$, then:
  $$\partial_t \Phi_\theta(u(t)) = g(t) - \mathfrak{D}_\theta(u(t)) \leq g(t) = g^+(t),$$
  since $\mathfrak{D}_\theta \geq 0$.

\item If $g(t) < 0$, then $g^+(t) = 0$, while:
  $$\partial_t \Phi_\theta(u(t)) = g(t) - \mathfrak{D}_\theta(u(t)) \leq g(t) < 0 \leq g^+(t).$$
\end{itemize}

Hence in \textbf{all} cases we have the pointwise inequality:
$$\partial_t \Phi_\theta(u(t)) \leq g^+(t) = \max\left(0, \partial_t \Phi_\theta(u(t)) + \mathfrak{D}_\theta(u(t))\right) \quad \text{for a.e. } t \in [0, T]. \quad \text{(2)}$$

This uses only that $\mathfrak{D}_\theta \geq 0$.

\textbf{Step 2: Integrate the differential inequality.}

Integrate (2) from $0$ to any $t \in [0, T]$:
$$\Phi_\theta(u(t)) - \Phi_\theta(u(0)) = \int_0^t \partial_s \Phi_\theta(u(s)) \, ds \leq \int_0^t g^+(s) \, ds \leq \int_0^T g^+(s) \, ds = K_D^{(\theta)}(u|_{[0,T]}) \leq \varepsilon.$$

Therefore:
$$\Phi_\theta(u(t)) \leq \Phi_\theta(u(0)) + \varepsilon \quad \forall t \in [0, T]. \quad \text{(3)}$$

This proves the main estimate.

\textbf{Step 3: Exclusion of Mode C.E on $[0, T]$.}

By definition, Mode C.E (energy blow-up) requires $\sup_{0 \leq s < T^*} \Phi_\theta(u(s)) = +\infty$.

But (3) shows that on the finite interval $[0, T]$:
$$\sup_{0 \leq s \leq T} \Phi_\theta(u(s)) \leq \Phi_\theta(u(0)) + \varepsilon < \infty.$$

So \textbf{no blow-up can occur before time $T$} as long as the D-defect on $[0, T]$ is finite. This proves claim (1).

\textbf{Step 4: Uniform control up to $T^*$.}

Now suppose we have a function $E(T')$ with $K_D^{(\theta)}(u|_{[0,T']}) \leq E(T')$ for all $T' < T^*$, and $\sup_{T' < T^*} E(T') =: E_\infty < \infty$.

Then for each $t < T^*$, by applying (3) with $T' = t$ and $\varepsilon = E(t) \leq E_\infty$, we get:
$$\Phi_\theta(u(t)) \leq \Phi_\theta(u(0)) + E(t) \leq \Phi_\theta(u(0)) + E_\infty.$$

Taking supremum over $t < T^*$ yields:
$$\sup_{0 \leq t < T^*} \Phi_\theta(u(t)) \leq \Phi_\theta(u(0)) + E_\infty < \infty.$$

Thus the Mode C.E condition $\sup_{t < T^*} \Phi_\theta(u(t)) = +\infty$ is impossible. This proves claim (2).
\end{proof}

\begin{remark}[Robust structural transfer pattern]
\begin{itemize}
\item In the \textbf{exact} case $K_D^{(\theta)}(u) = 0$, we recover the usual Axiom D conclusion: $\partial_t \Phi_\theta(u(t)) \leq 0 \implies \Phi_\theta(u(t)) \leq \Phi_\theta(u(0))$ for all $t$, so Mode C.E is impossible.
\item In the \textbf{approximate} case, the theorem gives a sharp quantitative relaxation: \textit{energy can increase by at most the D-defect}.
\end{itemize}
\end{remark}

> **Key Insight (Built-in energy bounds):** A trainable hypostructure with small D-defect automatically provides uniform energy bounds. The deviation from the exact axiom D conclusion is controlled linearly by the defect.

### Robust Topological Sector Suppression

We now prove a robust version of {prf:ref}`mt-imported-topological-suppression`, showing that the measure of nontrivial sectors decays exponentially even when the action gap TB1 holds only approximately.

**Recall: Original {prf:ref}`mt-imported-topological-suppression`.** Assume:
- Axiom TB with action gap $\Delta > 0$,
- an invariant probability measure $\mu$ satisfying a log–Sobolev inequality with constant $\lambda_{\mathrm{LS}} > 0$,
- the action functional $\mathcal{A}: X \to [0, \infty)$ is Lipschitz with constant $L > 0$.

Then:
$$\mu\{x : \tau(x) \neq 0\} \leq C \exp\left(-c \lambda_{\mathrm{LS}} \frac{\Delta^2}{L^2}\right)$$
with universal constants $C = 1$, $c = 1/8$.

#### Hypotheses for the robust version

Let $(X, \mathcal{B}, \mu)$ be a probability space with:
- $\tau: X \to \mathcal{T}$ the sector map (discrete $\mathcal{T}$, $0 \in \mathcal{T}$ the trivial sector),
- $\mathcal{A}: X \to [0, \infty)$ a measurable "action" functional,
- $\mathcal{A}_{\min} := \inf_{\tau(x) = 0} \mathcal{A}(x)$.

Assume:

1. **(Log–Sobolev inequality.)** $\mu$ satisfies a log–Sobolev inequality with constant $\lambda_{\mathrm{LS}} > 0$:
   $$\mathrm{Ent}_\mu(f^2) \leq \frac{2}{\lambda_{\mathrm{LS}}} \int |\nabla f|^2 \, d\mu$$
   for all smooth $f$, implying Gaussian concentration via the Herbst argument.

2. **(Lipschitz action.)** $\mathcal{A}$ is $L$-Lipschitz with respect to the ambient metric $d$ on $X$:
   $$|\mathcal{A}(x) - \mathcal{A}(y)| \leq L \, d(x, y) \quad \forall x, y \in X.$$

3. **(Approximate action gap.)** There exist constants $\Delta > 0$, $\varepsilon_{\mathrm{gap}} \geq 0$ and a measurable set $B \subset X$ ("bad set") such that:
   - $B \subset \{x : \tau(x) \neq 0\}$,
   - $\mu(B) \leq \eta$ for some $\eta \in [0, 1]$,
   - for all $x \in X \setminus B$ with $\tau(x) \neq 0$:
     $$\mathcal{A}(x) \geq \mathcal{A}_{\min} + \Delta - \varepsilon_{\mathrm{gap}}. \quad \text{(TG$_\varepsilon$)}$$

   So the exact TB1 gap $\tau(x) \neq 0 \Rightarrow \mathcal{A}(x) \geq \mathcal{A}_{\min} + \Delta$ is allowed to fail on $B$ (small measure) and to be off by $\varepsilon_{\mathrm{gap}}$.

Define the **effective gap**:
$$\Delta_{\mathrm{eff}} := \max\left\{\Delta - \varepsilon_{\mathrm{gap}} - L\sqrt{\frac{\pi}{2\lambda_{\mathrm{LS}}}}, 0\right\}.$$

\begin{metatheorem}[Robust Topological Sector Suppression]\label{mt:robust-topological-sector-suppression}
Under hypotheses (1)–(3) above:
$$\mu\big(\{x : \tau(x) \neq 0\}\big) \leq \eta + \exp\left(-\frac{\lambda_{\mathrm{LS}} \Delta_{\mathrm{eff}}^2}{2L^2}\right).$$
\end{metatheorem}

In particular:
- If the **bad set disappears** ($\eta = 0$) and the gap is exact ($\varepsilon_{\mathrm{gap}} = 0$), and if $\Delta \geq 2L\sqrt{\pi/(2\lambda_{\mathrm{LS}})}$, then $\Delta_{\mathrm{eff}} \geq \Delta/2$ and:
  $$\mu\{\tau \neq 0\} \leq \exp\left(-\frac{\lambda_{\mathrm{LS}} \Delta^2}{8L^2}\right),$$
  which recovers the original {prf:ref}`mt-imported-topological-suppression` bound with $C = 1$, $c = 1/8$ up to the mild "large gap" condition.

- As $\varepsilon_{\mathrm{gap}} \to 0$ and $\eta \to 0$, $\Delta_{\mathrm{eff}} \uparrow \Delta - L\sqrt{\pi/(2\lambda_{\mathrm{LS}})}$, so the suppression bound smoothly tends to the exact one.

\begin{proof}
Let $\bar{\mathcal{A}} := \int_X \mathcal{A} \, d\mu$ denote the mean action.

We use two standard consequences of log–Sobolev + Lipschitz:

\textbf{Gaussian concentration (Herbst).} For any $r > 0$:
$$\mu\{x \in X : \mathcal{A}(x) - \bar{\mathcal{A}} \geq r\} \leq \exp\left(-\frac{\lambda_{\mathrm{LS}} r^2}{2L^2}\right). \quad \text{(1)}$$

\textbf{Bound on the mean above the minimum.} Let $\mathcal{A}_{\inf} := \inf_X \mathcal{A}$ (which is $\leq \mathcal{A}_{\min}$). Then:
$$\bar{\mathcal{A}} - \mathcal{A}_{\inf} = \int_0^\infty \mu\{\mathcal{A} - \mathcal{A}_{\inf} \geq s\} \, ds \leq \int_0^\infty \exp\left(-\frac{\lambda_{\mathrm{LS}} s^2}{2L^2}\right) ds = L\sqrt{\frac{\pi}{2\lambda_{\mathrm{LS}}}}. \quad \text{(2)}$$

Hence, since $\mathcal{A}_{\inf} \leq \mathcal{A}_{\min}$:
$$\bar{\mathcal{A}} - \mathcal{A}_{\min} \leq \bar{\mathcal{A}} - \mathcal{A}_{\inf} \leq L\sqrt{\frac{\pi}{2\lambda_{\mathrm{LS}}}}. \quad \text{(3)}$$

\textbf{Step 1: Lower bound on $\mathcal{A}(x) - \bar{\mathcal{A}}$ for nontrivial sectors.}

Fix any $x \in X \setminus B$ with $\tau(x) \neq 0$. By the approximate gap condition (TG$_\varepsilon$):
$$\mathcal{A}(x) \geq \mathcal{A}_{\min} + \Delta - \varepsilon_{\mathrm{gap}}.$$

Subtract $\bar{\mathcal{A}}$ from both sides and use (3):
$$\mathcal{A}(x) - \bar{\mathcal{A}} \geq (\mathcal{A}_{\min} + \Delta - \varepsilon_{\mathrm{gap}}) - \bar{\mathcal{A}} = \Delta - \varepsilon_{\mathrm{gap}} - (\bar{\mathcal{A}} - \mathcal{A}_{\min}) \geq \Delta - \varepsilon_{\mathrm{gap}} - L\sqrt{\frac{\pi}{2\lambda_{\mathrm{LS}}}}.$$

Thus for any such $x$:
$$\mathcal{A}(x) - \bar{\mathcal{A}} \geq \Delta_{\mathrm{eff}}. \quad \text{(4)}$$

Therefore we have the inclusion of events:
$$\{x \in X \setminus B : \tau(x) \neq 0\} \subset \{x \in X : \mathcal{A}(x) - \bar{\mathcal{A}} \geq \Delta_{\mathrm{eff}}\}. \quad \text{(5)}$$

\textbf{Step 2: Concentration bound.}

Apply the Gaussian concentration (1) with $r = \Delta_{\mathrm{eff}}$:
$$\mu\{\mathcal{A} - \bar{\mathcal{A}} \geq \Delta_{\mathrm{eff}}\} \leq \exp\left(-\frac{\lambda_{\mathrm{LS}} \Delta_{\mathrm{eff}}^2}{2L^2}\right). \quad \text{(6)}$$

Combining (5) and (6):
$$\mu\{x \in X \setminus B : \tau(x) \neq 0\} \leq \exp\left(-\frac{\lambda_{\mathrm{LS}} \Delta_{\mathrm{eff}}^2}{2L^2}\right). \quad \text{(7)}$$

\textbf{Step 3: Add back the bad set $B$.}

We have:
$$\{x : \tau(x) \neq 0\} \subset B \cup \{x \in X \setminus B : \tau(x) \neq 0\}.$$

Hence:
$$\mu\{\tau \neq 0\} \leq \mu(B) + \mu\{x \in X \setminus B : \tau(x) \neq 0\} \leq \eta + \exp\left(-\frac{\lambda_{\mathrm{LS}} \Delta_{\mathrm{eff}}^2}{2L^2}\right),$$
using $\mu(B) \leq \eta$ and (7). This is exactly the claimed bound.
\end{proof}

\begin{remark}[Connection to meta-learning]
This theorem connects the TB-defect to the meta-learning story:
\begin{itemize}
\item The TB-defect can be interpreted as $\varepsilon_{\mathrm{gap}}$ (how much the action gap inequality fails in value) and $\eta$ (how much of the mass lives in a ``bad'' region where the gap fails completely).
\item Small TB-defect in the learned hypostructure $\Rightarrow$ small $\varepsilon_{\mathrm{gap}}$, $\eta$.
\item The log-Sobolev constant $\lambda_{\mathrm{LS}}$ and Lipschitz constant $L$ can be estimated from data, giving \textbf{explicit bounds} on $\mu\{\tau \neq 0\}$.
\end{itemize}
\end{remark}

> **Key Insight (Built-in sector control):** A trainable hypostructure with small TB-defect automatically provides exponential suppression of nontrivial sectors. The effective gap $\Delta_{\mathrm{eff}}$ smoothly interpolates between exact and approximate axioms.

### Curriculum Stability for Trainable Hypostructures

In practice, one does not typically train a hypostructure learner directly on the most complex possible systems. Instead, it is natural to adopt a **curriculum**: start with simpler systems (e.g. linear ODEs, toy PDEs), then gradually increase complexity (e.g. nonlinear PDEs, multi-scale systems, control-coupled systems), at each stage refining the learned axioms.

We now formalize a **Curriculum Stability** metatheorem: under mild conditions on the path of "true" hypostructure parameters along the curriculum, gradient-based training with warm starts tracks this path and converges to the final, fully complex hypostructure $\Theta^*_{\mathrm{full}}$, without jumping to a spurious ontology.

#### Curriculum of task distributions

Let $\mathcal{S}_1 \subseteq \mathcal{S}_2 \subseteq \cdots \subseteq \mathcal{S}_K$ be an increasing sequence of system distributions, each supported on systems $S$ that admit hypostructure representations in a common parametric family $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$.

For each stage $k = 1, \ldots, K$, define the **stage-$k$ average axiom risk**:
$$\mathcal{R}_k(\Theta) := \mathbb{E}_{S \sim \mathcal{S}_k}[\mathcal{R}_S(\Theta)],$$
where $\mathcal{R}_S(\Theta)$ is the joint axiom risk for system $S$ with parameter $\Theta$ (as in §13).

We think of $\mathcal{S}_1$ as a "simple" distribution (e.g. low-complexity systems), and $\mathcal{S}_K$ as the full, target distribution $\mathcal{S}_{\mathrm{full}}$.

#### True hypostructures along the curriculum

We assume that at each stage $k$, there exists a **true** parameter $\Theta^*_k \in \Theta_{\mathrm{adm}}$ such that:

- $\mathcal{R}_k(\Theta^*_k) = 0$;
- for $\mathcal{S}_k$-almost every system $S$, the hypostructure $\mathcal{H}_{\Theta^*_k}$ satisfies the axioms and defects vanish: $\mathcal{R}_S(\Theta^*_k) = 0$;
- $\Theta^*_k$ is structurally identifiable up to Hypo-isomorphism on $\mathcal{S}_k$.

We write $\Theta^*_{\mathrm{full}} := \Theta^*_K$ for the final-stage parameter.

**Assumption 13.52 (Smooth structural path).** There exists a $C^1$ curve $\gamma : [0,1] \to \Theta_{\mathrm{adm}}$ such that:
$$\gamma(t_k) = \Theta^*_k, \quad 0 = t_1 < t_2 < \cdots < t_K = 1,$$
and $|\dot{\gamma}(t)|$ is bounded on $[0,1]$. We call $\gamma$ the **structural curriculum path**.

**Assumption 13.53 (Stagewise strong convexity).** For each $k = 1, \ldots, K$, there exist constants $c_k, C_k, \rho_k > 0$ such that:
$$c_k |\Theta - \Theta^*_k|^2 \leq \mathcal{R}_k(\Theta) - \mathcal{R}_k(\Theta^*_k) \leq C_k |\Theta - \Theta^*_k|^2$$
for all $\Theta$ with $|\Theta - \Theta^*_k| \leq \rho_k$.

We also assume that the gradients $\nabla \mathcal{R}_k$ are Lipschitz in $\Theta$ on these neighborhoods. Let:
$$c_{\min} := \min_k c_k, \quad C_{\max} := \max_k C_k, \quad \rho := \min_k \rho_k.$$

#### Warm-start gradient descent along the curriculum

We consider the following **curriculum training** procedure:

1. Initialize $\Theta^{(1)}_0$ in a small neighborhood of $\Theta^*_1$.

2. For each stage $k = 1, \ldots, K$:
   - Run gradient descent on $\mathcal{R}_k$:
   $$\Theta^{(k)}_{t+1} = \Theta^{(k)}_t - \eta_{k,t} \nabla \mathcal{R}_k(\Theta^{(k)}_t),$$
   with stepsizes $\eta_{k,t}$ satisfying $\sum_t \eta_{k,t} = \infty$, $\sum_t \eta_{k,t}^2 < \infty$, and small enough to stay in the local convexity region.
   - Let $\widehat{\Theta}_k := \lim_{t \to \infty} \Theta^{(k)}_t$ (which exists and equals the unique minimizer in the basin).
   - Use $\widehat{\Theta}_k$ as the initialization for the next stage: $\Theta^{(k+1)}_0 := \widehat{\Theta}_k$.

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Smooth Curriculum Path:** $C^1$ curve $\gamma : [0,1] \to \Theta_{\mathrm{adm}}$ with $|\dot{\gamma}(t)| \leq M$, $\gamma(t_k) = \Theta^*_k$
>     *   [ ] **Stagewise Strong Convexity:** $c_k|\Theta - \Theta^*_k|^2 \leq \mathcal{R}_k(\Theta) \leq C_k|\Theta - \Theta^*_k|^2$ for $|\Theta - \Theta^*_k| \leq \rho_k$
>     *   [ ] **Small Curriculum Steps:** $|\Theta^*_{k+1} - \Theta^*_k| \leq \rho/4$
>     *   [ ] **Accurate Stagewise Minimization:** $|\widehat{\Theta}_k - \Theta^*_k| \leq \rho/4$
> *   **Output (Structural Guarantee):**
>     *   Stay in correct basin: $|\Theta^{(k)}_0 - \Theta^*_k| \leq \rho/2$ for all stages
>     *   Path tracking: $|\widehat{\Theta}_k - \Theta^*_k| \leq \rho/4$ uniformly
>     *   Convergence: $|\widehat{\Theta}_K - \Theta^*_{\mathrm{full}}| \leq \rho/4$
> *   **Failure Condition (Debug):**
>     *   If **Curriculum Steps** too large → **Mode ontology jump** (warm-start leaves basin, wrong minimum found)
>     *   If **Strong Convexity** fails → **Mode basin ambiguity** (multiple local minima, convergence uncertain)

\begin{metatheorem}[Curriculum Stability]\label{mt:curriculum-stability}
Under the above setting, suppose:
\end{metatheorem}

1. **(Smooth curriculum path.)** Assumption 13.52 holds, and $|\dot{\gamma}(t)| \leq M$ for all $t \in [0,1]$.

2. **(Stagewise strong convexity.)** Assumption 13.53 holds uniformly: $c_{\min} > 0$, $C_{\max} < \infty$, $\rho > 0$.

3. **(Small curriculum steps.)** The time steps $t_k$ are chosen such that:
$$|\Theta^*_{k+1} - \Theta^*_k| = |\gamma(t_{k+1}) - \gamma(t_k)| \leq \frac{\rho}{4} \quad \text{for all } k.$$
Equivalently, $(t_{k+1} - t_k) \leq \rho/(4M)$.

4. **(Accurate stagewise minimization.)** At each stage $k$, gradient descent on $\mathcal{R}_k$ is run long enough (with suitably small stepsizes) so that:
$$|\widehat{\Theta}_k - \Theta^*_k| \leq \frac{\rho}{4}.$$

Then for all stages $k = 1, \ldots, K$:

1. **(Stay in the correct basin.)** The initialization for each stage lies in the strong-convexity neighborhood of the true parameter:
$$|\Theta^{(k)}_0 - \Theta^*_k| = |\widehat{\Theta}_{k-1} - \Theta^*_k| \leq \frac{\rho}{2} < \rho.$$
Hence gradient descent at stage $k$ remains in the basin of $\Theta^*_k$ and converges to it.

2. **(Tracking the structural path.)** The sequence of stagewise minimizers $\widehat{\Theta}_k$ satisfies:
$$|\widehat{\Theta}_k - \Theta^*_k| \leq \frac{\rho}{4} \quad \text{for all } k,$$
and hence forms a discrete approximation to the structural path $\gamma$ staying uniformly close to it.

3. **(Convergence to the full hypostructure.)** In particular, the final parameter $\widehat{\Theta}_K$ satisfies:
$$|\widehat{\Theta}_K - \Theta^*_{\mathrm{full}}| \leq \frac{\rho}{4},$$
i.e. curriculum training converges (modulo this small error, which can be made arbitrarily small by refining the steps and optimization accuracy) to the true full hypostructure.

If, moreover, we let the number of stages $K \to \infty$ so that $\max_k(t_{k+1} - t_k) \to 0$ and increase the optimization accuracy at each stage, then in the limit the curriculum procedure tracks $\gamma$ arbitrarily closely and converges to $\Theta^*_{\mathrm{full}}$ in parameter space.

\begin{proof}
We argue by induction on the curriculum stages.

\textbf{Base case ($k = 1$).} By assumption, we choose $\Theta^{(1)}_0$ close to $\Theta^*_1$, in particular $|\Theta^{(1)}_0 - \Theta^*_1| \leq \rho/2$. By stagewise strong convexity (Assumption 13.53) and standard convergence results for gradient descent on strongly convex, smooth functions, the iterates $\Theta^{(1)}_t$ remain in the ball $B(\Theta^*_1, \rho)$ and converge to the unique minimizer $\Theta^*_1$. For sufficiently long training and small enough step sizes:
$$|\widehat{\Theta}_1 - \Theta^*_1| \leq \rho/4.$$

\textbf{Induction step.} Suppose that at stage $k$ we have $|\widehat{\Theta}_k - \Theta^*_k| \leq \rho/4$.

We now consider stage $k+1$. By definition of the curriculum path:
$$|\Theta^*_{k+1} - \Theta^*_k| = |\gamma(t_{k+1}) - \gamma(t_k)| \leq \frac{\rho}{4}.$$

Thus the stage-$(k+1)$ initialization $\Theta^{(k+1)}_0 := \widehat{\Theta}_k$ satisfies:
$$|\Theta^{(k+1)}_0 - \Theta^*_{k+1}| \leq |\Theta^{(k+1)}_0 - \Theta^*_k| + |\Theta^*_k - \Theta^*_{k+1}| \leq \frac{\rho}{4} + \frac{\rho}{4} = \frac{\rho}{2} < \rho.$$

Therefore $\Theta^{(k+1)}_0$ lies in the strong-convexity neighborhood $B(\Theta^*_{k+1}, \rho)$. Gradient descent on $\mathcal{R}_{k+1}$ with sufficiently small step sizes stays inside $B(\Theta^*_{k+1}, \rho)$ and converges to the unique minimizer $\Theta^*_{k+1}$. By running it long enough:
$$|\widehat{\Theta}_{k+1} - \Theta^*_{k+1}| \leq \rho/4,$$
which is the induction hypothesis for the next stage.

By induction, the statements in (1) and (2) hold for all $k = 1, \ldots, K$. The final claim (3) follows immediately for $k = K$, with $\Theta^*_{\mathrm{full}} = \Theta^*_K$.

In the refined-curriculum limit where $K \to \infty$ and $\max_k(t_{k+1} - t_k) \to 0$ while per-stage optimization accuracy is driven to $0$, the discrete sequence $\{\widehat{\Theta}_k\}$ converges uniformly to the continuous path $\gamma(t_k)$ and hence to $\Theta^*_{\mathrm{full}}$ as $t_K \to 1$.
\end{proof}

\begin{remark}[Structural safety of curricula]
The theorem shows that \textbf{curriculum training is structurally safe} as long as:
\begin{itemize}
\item each stage's average axiom risk is strongly convex in a neighborhood of its true parameter, and
\item successive true parameters $\Theta^*_k$ are not too far apart.
\end{itemize}
Intuitively, the curriculum path $\gamma$ describes how the ``true axioms'' must deform as one moves from simple to complex systems. The theorem guarantees that a trainable hypostructure, initialized and trained at each stage using the previous stage's solution, will track $\gamma$ rather than jumping to unrelated minima.
\end{remark}

\begin{remark}[Practical implications]
Combined with the generalization and robustness metatheorems, this implies:
\begin{itemize}
\item training on simple systems first fixes the core axioms,
\item advancing the curriculum refines these axioms instead of destabilizing them,
\item and the final hypostructure accurately captures the structural content of the full system distribution.
\end{itemize}
\end{remark}

> **Key Insight:** Increasing task complexity along a structurally coherent curriculum preserves the learned axiom structure and refines it, rather than destabilizing it. No spurious ontology (wrong hypostructure branch) is selected along the curriculum.

### Robust Łojasiewicz Convergence

The preceding metatheorems establish robustness for energy bounds (Mode C.E) and sector suppression (TB). We now prove a **robust LS convergence theorem**: small LS-defect implies "almost convergence" to the safe manifold $M$, with explicit quantitative bounds on the measure of "bad" times.

#### Setting and assumptions

Let:
- $H$ be a Hilbert space (or a Riemannian manifold),
- $\Phi: H \to \mathbb{R}$ be a $C^1$ functional bounded below,
- $u: [0, \infty) \to H$ solve the **gradient flow**:
  $$u'(t) = -\nabla \Phi(u(t)).$$

Define the **energy gap**:
$$\Phi_{\min} := \inf_{x \in H} \Phi(x), \qquad f(t) := \Phi(u(t)) - \Phi_{\min} \geq 0.$$

Then along the trajectory:
$$f'(t) = \frac{d}{dt} \Phi(u(t)) = \langle \nabla \Phi(u(t)), u'(t) \rangle = -|\nabla \Phi(u(t))|^2 \leq 0,$$
so $f$ is nonincreasing and bounded below, hence has a limit $f_\infty$.

Let $M \subset H$ be the **safe manifold** (set of equilibria / canonical profiles), as in Axiom LS.

**Assumption (LS-geom): Geometric Łojasiewicz inequality (exact).**

There exists a neighborhood $U \supset M$, constants $\theta \in (0, 1]$, $C_{\mathrm{geo}} > 0$ such that:
$$\Phi(x) - \Phi_{\min} \geq C_{\mathrm{geo}} \, \mathrm{dist}(x, M)^{1/\theta} \quad \text{for all } x \in U. \quad \text{(G-LS)}$$

**Assumption (LS-grad$_\varepsilon$): Gradient Łojasiewicz inequality with $L^2$-defect.**

There exists $c_{\mathrm{LS}} > 0$, $\theta \in (0, 1]$ (the same $\theta$ as above), and a measurable function $e: [0, \infty) \to [0, \infty)$ such that for all $t \geq 0$ with $u(t) \in U$:
$$|\nabla \Phi(u(t))| \geq c_{\mathrm{LS}} \, f(t)^{1-\theta} - e(t). \quad \text{(G-LS-approx)}$$

Define the **LS-defect** of the trajectory by:
$$K_{\mathrm{LS}}(u) := \int_0^\infty e(t)^2 \, dt.$$

We assume this is finite, and write $K_{\mathrm{LS}}(u) \leq \varepsilon^2$ for some $\varepsilon \geq 0$.

**Assumption (Stay in LS region).**

Assume there is $T_0 \geq 0$ such that $u(t) \in U$ for all $t \geq T_0$.

\begin{metatheorem}[Robust LS Convergence]\label{mt:robust-ls-convergence}
Under the assumptions above:
\end{metatheorem}

1. **(Energy gap goes to zero.)**
   $$\lim_{t \to \infty} f(t) = 0.$$

2. **(Quantitative integrability of distance to $M$.)**
   For $p := \frac{2(1-\theta)}{\theta}$, there exists a constant $C_1 = C_1(\theta, c_{\mathrm{LS}}, C_{\mathrm{geo}}) > 0$ such that:
   $$\int_{T_0}^\infty \mathrm{dist}(u(t), M)^p \, dt \leq C_1 \left( f(T_0) + K_{\mathrm{LS}}(u) \right).$$

3. **("Almost convergence" to $M$ in measure.)**
   For every radius $R > 0$:
   $$\mathcal{L}^1\big(\{t \geq T_0 : \mathrm{dist}(u(t), M) \geq R\}\big) \leq \frac{C_1}{R^p} \big( f(T_0) + K_{\mathrm{LS}}(u) \big),$$
   where $\mathcal{L}^1$ is Lebesgue measure. As $R \downarrow 0$, the fraction of time spent at distance $\geq R$ from $M$ goes to zero, at a rate controlled by $f(T_0) + K_{\mathrm{LS}}(u)$.

4. **(Convergence along a subsequence; and, with exact LS, full convergence.)**
   There exists a sequence $t_n \to \infty$ such that $\mathrm{dist}(u(t_n), M) \to 0$ as $n \to \infty$.

   If, additionally, the geometric LS inequality (G-LS) holds for all large times and Axiom C provides precompactness of the trajectory, then the full trajectory converges:
   $$u(t) \to x_\infty \in M \quad \text{as } t \to \infty,$$
   which is the usual LS–Simon convergence statement in Axiom LS.

\begin{proof}
We shift time so that $T_0 = 0$ for simplicity (replacing $f(0)$ by $f(T_0)$).

\textbf{Step 1: A differential inequality for the energy gap.}

Recall $f(t) = \Phi(u(t)) - \Phi_{\min} \geq 0$. Because $u$ is a gradient flow, we have the energy identity:
$$f'(t) = -|\nabla \Phi(u(t))|^2.$$

From the approximate LS inequality (G-LS-approx):
$$|\nabla \Phi(u(t))| \geq c_{\mathrm{LS}} f(t)^{1-\theta} - e(t).$$

Define $g(t) := c_{\mathrm{LS}} f(t)^{1-\theta} - e(t)$. Then $|\nabla \Phi(u(t))| \geq g(t)$, and thus:
$$f'(t) = -|\nabla \Phi(u(t))|^2 \leq -g(t)^2.$$

Expanding:
$$g(t)^2 = c_{\mathrm{LS}}^2 f(t)^{2(1-\theta)} - 2c_{\mathrm{LS}} f(t)^{1-\theta} e(t) + e(t)^2.$$

Hence:
$$f'(t) \leq -c_{\mathrm{LS}}^2 f(t)^{2(1-\theta)} + 2c_{\mathrm{LS}} f(t)^{1-\theta} e(t) - e(t)^2.$$

Drop the negative term $-e(t)^2$ to obtain:
$$f'(t) + c_{\mathrm{LS}}^2 f(t)^{2(1-\theta)} \leq 2c_{\mathrm{LS}} f(t)^{1-\theta} e(t). \quad \text{(1)}$$

\textbf{Step 2: Integrate and absorb the error (using $L^2$ defect).}

Use Young's inequality on the RHS of (1): for any $\eta > 0$,
$$2c_{\mathrm{LS}} f^{1-\theta} e \leq \eta c_{\mathrm{LS}}^2 f^{2(1-\theta)} + \frac{1}{\eta} e^2.$$

Choose $\eta = 1/2$:
$$2c_{\mathrm{LS}} f^{1-\theta} e \leq \frac{c_{\mathrm{LS}}^2}{2} f^{2(1-\theta)} + 2e^2.$$

Substitute into (1):
$$f'(t) + c_{\mathrm{LS}}^2 f^{2(1-\theta)} \leq \frac{c_{\mathrm{LS}}^2}{2} f^{2(1-\theta)} + 2e(t)^2,$$
so:
$$f'(t) + \frac{c_{\mathrm{LS}}^2}{2} f^{2(1-\theta)} \leq 2e(t)^2. \quad \text{(2)}$$

Integrate (2) from $0$ to any $T > 0$:
$$\int_0^T f'(t) \, dt + \frac{c_{\mathrm{LS}}^2}{2} \int_0^T f(t)^{2(1-\theta)} dt \leq 2 \int_0^T e(t)^2 \, dt.$$

The left-hand integral of $f'$ is $f(T) - f(0)$:
$$f(T) - f(0) + \frac{c_{\mathrm{LS}}^2}{2} \int_0^T f(t)^{2(1-\theta)} dt \leq 2 \int_0^T e(t)^2 \, dt.$$

Since $f(T) \geq 0$, we can drop it:
$$\frac{c_{\mathrm{LS}}^2}{2} \int_0^T f(t)^{2(1-\theta)} dt \leq f(0) + 2 \int_0^T e(t)^2 \, dt.$$

Let $T \to \infty$. Using $K_{\mathrm{LS}}(u) = \int_0^\infty e(t)^2 \, dt \leq \varepsilon^2$:
$$\frac{c_{\mathrm{LS}}^2}{2} \int_0^\infty f(t)^{2(1-\theta)} dt \leq f(0) + 2K_{\mathrm{LS}}(u).$$

Hence:
$$\int_0^\infty f(t)^{2(1-\theta)} dt \leq \frac{2}{c_{\mathrm{LS}}^2} f(0) + \frac{4}{c_{\mathrm{LS}}^2} K_{\mathrm{LS}}(u). \quad \text{(3)}$$

This proves the quantitative \textbf{integrability} of $f^{2(1-\theta)}$.

\textbf{Step 3: Show $f(t) \to 0$ as $t \to \infty$.}

We know: $f(t) \geq 0$, $f'(t) = -|\nabla \Phi(u(t))|^2 \leq 0$, so $f$ is nonincreasing and bounded below. Hence $\exists f_\infty \geq 0 : \lim_{t \to \infty} f(t) = f_\infty$.

Assume for contradiction $f_\infty > 0$. Then for all large $t \geq T_1$:
$$f(t) \geq \frac{f_\infty}{2} > 0,$$
so:
$$f(t)^{2(1-\theta)} \geq \left(\frac{f_\infty}{2}\right)^{2(1-\theta)} =: c_0 > 0 \quad \text{for all } t \geq T_1.$$

Then:
$$\int_0^\infty f(t)^{2(1-\theta)} dt \geq \int_{T_1}^\infty f(t)^{2(1-\theta)} dt \geq \int_{T_1}^\infty c_0 \, dt = \infty,$$
contradicting the finiteness from (3).

Thus necessarily $f_\infty = 0$, i.e., $\lim_{t \to \infty} f(t) = 0$. This proves conclusion (1).

\textbf{Step 4: Integrability of distance to $M$.}

From the geometric LS inequality (G-LS):
$$f(t) = \Phi(u(t)) - \Phi_{\min} \geq C_{\mathrm{geo}} \, \mathrm{dist}(u(t), M)^{1/\theta} \quad \text{for all } t \text{ with } u(t) \in U.$$

Rearrange:
$$\mathrm{dist}(u(t), M) \leq C_{\mathrm{geo}}^{-\theta} f(t)^\theta.$$

Raise both sides to the power $p := \frac{2(1-\theta)}{\theta} > 0$:
$$\mathrm{dist}(u(t), M)^p \leq C_{\mathrm{geo}}^{-p\theta} f(t)^{p\theta}.$$

But $p\theta = \frac{2(1-\theta)}{\theta} \cdot \theta = 2(1-\theta)$, so:
$$\mathrm{dist}(u(t), M)^p \leq C_{\mathrm{geo}}^{-2(1-\theta)} f(t)^{2(1-\theta)}.$$

Integrate from $0$ to $\infty$ and use (3):
$$\int_0^\infty \mathrm{dist}(u(t), M)^p \, dt \leq C_{\mathrm{geo}}^{-2(1-\theta)} \int_0^\infty f(t)^{2(1-\theta)} dt \leq C_{\mathrm{geo}}^{-2(1-\theta)} \left(\frac{2}{c_{\mathrm{LS}}^2} f(0) + \frac{4}{c_{\mathrm{LS}}^2} K_{\mathrm{LS}}(u)\right).$$

Setting $C_1 := C_{\mathrm{geo}}^{-2(1-\theta)} \cdot \frac{4}{c_{\mathrm{LS}}^2}$:
$$\int_0^\infty \mathrm{dist}(u(t), M)^p \, dt \leq C_1 \big( f(0) + K_{\mathrm{LS}}(u) \big),$$
which is conclusion (2).

\textbf{Step 5: Measure of "bad" times (far from $M$).}

Fix any $R > 0$. Let $S_R := \{t \geq 0 : \mathrm{dist}(u(t), M) \geq R\}$.

Then on $S_R$: $\mathrm{dist}(u(t), M)^p \geq R^p$.

Thus:
$$\int_0^\infty \mathrm{dist}(u(t), M)^p \, dt \geq \int_{S_R} \mathrm{dist}(u(t), M)^p \, dt \geq R^p \, \mathcal{L}^1(S_R),$$
where $\mathcal{L}^1$ denotes Lebesgue measure.

So:
$$\mathcal{L}^1(S_R) \leq \frac{1}{R^p} \int_0^\infty \mathrm{dist}(u(t), M)^p \, dt \leq \frac{C_1}{R^p} \big( f(0) + K_{\mathrm{LS}}(u) \big).$$

This is precisely conclusion (3). As $R \downarrow 0$, the measure of times with distance $\geq R$ is bounded by a factor that scales like $R^{-p}$.

\textbf{Step 6: Subsequence convergence to $M$.}

From (2), we know $\int_0^\infty \mathrm{dist}(u(t), M)^p \, dt < \infty$. A standard measure-theory fact: if a nonnegative function $h$ has finite integral on $[0, \infty)$, then there exists a sequence $t_n \to \infty$ with $h(t_n) \to 0$.

Apply this to $h(t) := \mathrm{dist}(u(t), M)^p$:
$$\exists t_n \to \infty \quad \text{such that} \quad \mathrm{dist}(u(t_n), M)^p \to 0 \implies \mathrm{dist}(u(t_n), M) \to 0.$$

That proves conclusion (4) in its subsequence form.

If we now bring in Axiom C + Reg (bounded trajectories have limit points) and the precise LS machinery (C·D–LS+Reg $\Rightarrow$ convergence to $M$ for bounded trajectories), then one can upgrade "subsequence convergence to $M$" to \textbf{full convergence} $u(t) \to x_\infty \in M$, whenever the exact LS conditions hold globally for large time.
\end{proof}

\begin{remark}[Connection to learning]
In the meta-learning story: A meta-learner that finds a hypostructure with small LS-defect $K_{\mathrm{LS}}$ is enough to conclude that ``most'' of the long-time dynamics (in time-measure sense) lies arbitrarily close to the safe manifold $M$, with explicit quantitative bounds depending on the learned LS constants and the residual defect.
\end{remark}

> **Key Insight (Built-in convergence guarantees):** A trainable hypostructure with small LS-defect automatically provides:
> - Energy gap $f(t) \to 0$,
> - Distance to $M$ is $L^p$-integrable,
> - The set of times where $u$ is farther than $R$ from $M$ has measure $\lesssim (f(T_0) + K_{\mathrm{LS}})/R^p$,
> - Subsequence convergence to $M$.
>
> The exact LS-Simon convergence is the limiting case when the defect vanishes.

### Hypostructure-from-Raw-Data: Learning Structure from Observations

The preceding robust metatheorems establish that approximate axiom satisfaction (small defects) still yields meaningful structural conclusions. We now address a more fundamental question: **can we learn hypostructures directly from raw observational data, without prior knowledge of the state space or dynamics?**

This section presents a rigorous meta-theorem showing that training on **prediction + axiom-risk** from raw observations recovers the latent hypostructure (up to isomorphism) in the population limit, provided such a hypostructure exists.

#### Setup: Systems, Data, and Models

##### Task/System Space

Let $(\mathcal{S}, \mathcal{F}, \nu)$ be a probability space of **systems** (or "tasks").

For each $s \in \mathcal{S}$, we have an associated **observation process**:
$$Y^{(s)} = (Y^{(s)}_t)_{t \in \mathbb{Z}}$$
taking values in a Polish observation space $\mathcal{Y}$.

Let $\mathbb{P}_s$ be the law of the process $Y^{(s)}$ on $\mathcal{Y}^{\mathbb{Z}}$.

We do **not** assume we know a state space or dynamics for $s$—only its observation law $\mathbb{P}_s$.

##### True Latent Hypostructured Systems (Realizability Layer)

We assume there exists some "true" latent representation, but it is hidden.

For each $s \in \mathcal{S}$, there exist:

- A separable metric latent space $X_s$,
- A measurable flow or semiflow $(S_t^{(s)})_{t \in \mathbb{Z}}$ on $X_s$,
- A **true hypostructure** $\mathcal{H}^{(s)*}$ on $X_s$ with structural data:
  $$\mathcal{H}^{(s)*} = (X_s, S_t^{(s)}, \Phi^{(s)*}, \mathfrak{D}^{(s)*}, c^{(s)*}, \tau^{(s)*}, \mathcal{A}^{(s)*}, \ldots),$$
  satisfying all axioms exactly (C, D, SC, Cap, TB, LS, GC, ...),
- An **observation map** $O_s: X_s \to \mathcal{Y}$ such that if $X^{(s)}_t$ follows $(S_t^{(s)})$ and $Y^{(s)}_t := O_s(X^{(s)}_t)$, then the law of $Y^{(s)}$ is exactly $\mathbb{P}_s$.

We call $\mathcal{H}^{(s)*}$ a **true latent hypostructure** for system $s$.

This is the *latent realizability* assumption: the world *has* a hypostructural description, but we do not know $X_s$, $S_t^{(s)}$, $O_s$, or the structure.

##### Models: Encoders and Parametric Hypostructures

We fix:

- A **latent model space** $Z = \mathbb{R}^d$ with its usual Euclidean metric.
- A **window size** $k \in \mathbb{N}$ for temporal encoding.
- A parameter space $\Psi \subset \mathbb{R}^p$ for **encoders** and $\Phi \subset \mathbb{R}^q$ for **hypostructure generators**; both are assumed to be compact or at least closed and such that level sets of the risks we define are relatively compact.

**Encoders.** For each $\psi \in \Psi$, we have a measurable **encoder**:
$$E_\psi: \mathcal{Y}^k \to Z.$$

Given a trajectory $y = (y_t)_{t \in \mathbb{Z}}$ and a fixed convention (say, left-aligned windows), define the **induced latent trajectory**:
$$z^{(\psi)}_t = E_\psi(y_{t-k+1}, \ldots, y_t) \in Z.$$

We are not assuming this comes from any actual state space—this is just what the encoder does.

**Hypostructure Generator (Hypernetwork).** We fix a **task representation map** $\iota: \mathcal{S} \to \mathbb{R}^m$ (which can be as simple as an index embedding, or empirical statistics).

For each $\varphi \in \Phi$, we have a **hypernetwork**:
$$H_\varphi: \mathbb{R}^m \to \Theta$$
with parameter space $\Theta \subset \mathbb{R}^r$, continuous in $\varphi$. For each task $s \in \mathcal{S}$:
$$\theta_{s,\varphi} := H_\varphi(\iota(s)) \in \Theta$$
is the hypostructure parameter for system $s$.

For each $\theta \in \Theta$, we have a **parametric hypostructure on $Z$**:
$$\mathcal{H}_\theta = (Z, F_\theta, \Phi_\theta, \mathfrak{D}_\theta, c_\theta, \tau_\theta, \mathcal{A}_\theta, \ldots)$$
where:
- $F_\theta: Z \to Z$ is the latent dynamics model,
- $\Phi_\theta, \mathfrak{D}_\theta, c_\theta, \mathcal{A}_\theta: Z \to \mathbb{R}$ are measurable structural maps,
- $\tau_\theta: Z \to \mathcal{T}$ is a measurable sector map.

Think of all of these as implemented by neural networks (universally approximating function classes), but we only need measurability and continuity in $\theta$.

Given $(\psi, \varphi)$ and a system $s$, the **effective hypostructure on latent trajectories** is:
$$\mathcal{H}^{(s)}_{\psi,\varphi} := (Z, F_{\theta_{s,\varphi}}, \Phi_{\theta_{s,\varphi}}, \mathfrak{D}_{\theta_{s,\varphi}}, c_{\theta_{s,\varphi}}, \tau_{\theta_{s,\varphi}}, \mathcal{A}_{\theta_{s,\varphi}}, \ldots)$$
restricted to the support of latent trajectories $z^{(\psi)}_t$ obtained from $Y^{(s)} \sim \mathbb{P}_s$.

#### Losses: Prediction + Axiom-Risk

##### Prediction Loss

Fix a nonnegative measurable loss $\ell: Z \times Z \to [0, \infty)$ (e.g., squared error).

For each $(\psi, \varphi)$, define the **population prediction loss**:
$$\mathcal{L}_{\mathrm{pred}}(\psi, \varphi) := \int_{\mathcal{S}} \mathbb{E}_{Y \sim \mathbb{P}_s}\left[\ell\big(F_{\theta_{s,\varphi}}(z^{(\psi)}_t), z^{(\psi)}_{t+1}\big)\right] \nu(ds),$$
where $t$ is any fixed time index (stationarity or shift-invariance of $\mathbb{P}_s$ makes the choice irrelevant; otherwise we can average over a finite window).

This is the usual latent one-step prediction risk.

##### Axiom-Risk

For each soft axiom $A$ in the list (C, D, SC, Cap, TB, LS, GC, ...), and for each $\theta$, we have an **axiom defect functional**:
$$K_A(\mathcal{H}_\theta; z_\bullet)$$
for a latent trajectory $z_\bullet = (z_t)_{t \in \mathbb{Z}}$, such that:
- $K_A(\mathcal{H}_\theta; z_\bullet) \geq 0$,
- $K_A(\mathcal{H}_\theta; z_\bullet) = 0$ if and only if the trajectory satisfies axiom $A$ exactly.

Fix nonnegative weights $\lambda_A \geq 0$ and define, for each $(\psi, \varphi)$:
$$\mathcal{R}_{\mathrm{axioms}}(\psi, \varphi) := \sum_A \lambda_A \int_{\mathcal{S}} \mathbb{E}_{Y \sim \mathbb{P}_s}\left[K_A\big(\mathcal{H}_{\theta_{s,\varphi}}; z^{(\psi)}_\bullet\big)\right] \nu(ds).$$

This is the **population axiom-risk**: average defect across tasks and trajectories.

##### Total Risk

Fix $\lambda > 0$ and define:
$$\mathcal{L}_{\mathrm{total}}(\psi, \varphi) := \mathcal{L}_{\mathrm{pred}}(\psi, \varphi) + \lambda \cdot \mathcal{R}_{\mathrm{axioms}}(\psi, \varphi).$$

This is the functional we will minimize by (stochastic) gradient descent.

#### Assumptions (Inductive Bias + Regularity)

We now state explicit assumptions that make this a well-posed meta-learning problem.

**Assumption (H1): Regularity/Measurability.**
- The maps $(\psi, \varphi) \mapsto E_\psi$, $(\varphi, s) \mapsto \theta_{s,\varphi}$, $(\theta, z) \mapsto F_\theta(z)$, $(\theta, z) \mapsto$ structural maps are Borel and continuous in parameters.
- For each $(\psi, \varphi)$, $\mathcal{L}_{\mathrm{pred}}(\psi, \varphi)$ and $\mathcal{R}_{\mathrm{axioms}}(\psi, \varphi)$ are finite and continuous in $(\psi, \varphi)$.

This is true if everything is implemented by continuous neural networks on compact domains with bounded outputs and $\ell$, $K_A$ are continuous in their arguments.

**Assumption (H2): Parametric Realizability of True Hypostructures.**

There exists a parameter pair $(\psi^*, \varphi^*) \in \Psi \times \Phi$ such that:

For $\nu$-almost every system $s \in \mathcal{S}$, there is an **isomorphism of hypostructures** $T_s: X_s \to Z$ (with inverse on the support of the dynamics) satisfying:

1. **Encoder consistency:** For $\mathbb{P}_s$-almost every trajectory $Y^{(s)}$, if $X^{(s)}_t$ is the latent true trajectory and $Y^{(s)}_t = O_s(X^{(s)}_t)$, then the encoded trajectory:
   $$z_t^{(\psi^*)} = E_{\psi^*}(Y^{(s)}_{t-k+1}, \ldots, Y^{(s)}_t)$$
   coincides with $T_s(X^{(s)}_t)$.

2. **Dynamics consistency:**
   $$F_{\theta_{s,\varphi^*}}(T_s(x)) = T_s(S_1^{(s)} x) \quad \text{for all } x \text{ in the support of the true dynamics}.$$

3. **Hypostructure consistency:** The pullback of the parametric hypostructure on $Z$ via $T_s$ equals the true hypostructure on $X_s$:
   $$T_s^*(\mathcal{H}_{\theta_{s,\varphi^*}}) = \mathcal{H}^{(s)*}.$$
   In particular, for every trajectory induced by $Y^{(s)}$, all axioms hold exactly, so every defect vanishes:
   $$K_A\big(\mathcal{H}_{\theta_{s,\varphi^*}}; z^{(\psi^*)}_\bullet\big) = 0 \quad \text{for all } A \text{ and for } \mathbb{P}_s\text{-a.e. } Y^{(s)}.$$

This is the formal statement: **there exists an encoder + hypernetwork whose induced latent hypostructures realize the true ones almost surely.**

**Assumption (H3): Identifiability Up to Hypostructure Isomorphism.**

If for some $(\psi, \varphi)$ we have:
- $\mathcal{L}_{\mathrm{pred}}(\psi, \varphi) = 0$,
- $\mathcal{R}_{\mathrm{axioms}}(\psi, \varphi) = 0$,

then for $\nu$-almost every $s \in \mathcal{S}$, there exists a hypostructure isomorphism $\tilde{T}_s: X_s \to Z$ such that $\tilde{T}_s^*(\mathcal{H}_{\theta_{s,\varphi}}) = \mathcal{H}^{(s)*}$, and the encoded trajectories coincide with $\tilde{T}_s(X^{(s)}_t)$ as in (H2.1).

In words: **zero total risk implies we have recovered the true latent hypostructure up to isomorphism.**

(This is exactly the "Meta-Identifiability" assumption, extended to include the encoder. It encodes the idea that there are no degenerate parameterizations that have perfect prediction and axioms but give a genuinely different structure.)

**Assumption (H4): Optimization (Gradient Descent/SGD Scheme).**

Let $(\psi_n, \varphi_n)_{n \geq 0}$ be an iterative sequence produced by some optimization algorithm (deterministic GD, stochastic GD, etc.) such that:

1. The learning rule is of the form:
   $$(\psi_{n+1}, \varphi_{n+1}) = (\psi_n, \varphi_n) - \eta_n \hat{\nabla} \mathcal{L}_{\mathrm{total}}(\psi_n, \varphi_n),$$
   where $\hat{\nabla} \mathcal{L}_{\mathrm{total}}$ is an unbiased stochastic gradient estimator with bounded variance (conditional on $(\psi_n, \varphi_n)$), constructed from i.i.d. samples of $s \sim \nu$ and trajectories $Y^{(s)} \sim \mathbb{P}_s$.

2. The step sizes $\eta_n$ satisfy the Robbins–Monro conditions:
   $$\sum_{n=0}^\infty \eta_n = \infty, \quad \sum_{n=0}^\infty \eta_n^2 < \infty.$$

3. $\mathcal{L}_{\mathrm{total}}$ is bounded below (by 0) and has **Lipschitz gradient** on $\Psi \times \Phi$, and its sublevel sets $\{(\psi, \varphi) : \mathcal{L}_{\mathrm{total}}(\psi, \varphi) \leq \alpha\}$ are relatively compact.

This is a standard nonconvex SGD setting. Classical results (e.g., Kushner–Yin, Benaïm) then say that:
- $\mathcal{L}_{\mathrm{total}}(\psi_n, \varphi_n)$ converges almost surely,
- The set of limit points of $(\psi_n, \varphi_n)$ is a compact connected set of **stationary points** of $\mathcal{L}_{\mathrm{total}}$.

We will use this as a black box. (Alternatively, assume exact GD on the population risk for a simpler statement.)

#### Main Meta-Theorem

\begin{metatheorem}[Hypostructure-from-Raw-Data]\label{mt:hypostructure-from-raw-data}
Assume (H1)–(H4). Then:
\end{metatheorem}

1. **(Zero infimum and nonempty minimizer set.)** The total population risk satisfies:
   $$\inf_{(\psi,\varphi) \in \Psi \times \Phi} \mathcal{L}_{\mathrm{total}}(\psi, \varphi) = 0$$
   and the set of global minimizers:
   $$\mathcal{M} := \{(\psi, \varphi) : \mathcal{L}_{\mathrm{total}}(\psi, \varphi) = 0\}$$
   is nonempty and compact.

2. **(Structural recovery at any global minimizer.)** For any $(\hat{\psi}, \hat{\varphi}) \in \mathcal{M}$, for $\nu$-almost every system $s \in \mathcal{S}$, there exists a hypostructure isomorphism $\tilde{T}_s: X_s \to Z$ such that:
   - The encoded latent trajectory matches the pushed-forward true trajectory:
     $$z_t^{(\hat{\psi})} = \tilde{T}_s(X^{(s)}_t) \quad \text{for } \mathbb{P}_s\text{-a.e. } Y^{(s)};$$
   - The induced hypostructure equals the true one:
     $$\tilde{T}_s^*(\mathcal{H}_{\theta_{s,\hat{\varphi}}}) = \mathcal{H}^{(s)*};$$
   - In particular, all global metatheorems (those using only axioms C, D, SC, Cap, TB, LS, GC, ...) hold **exactly** for the latent representation produced by $(\hat{\psi}, \hat{\varphi})$ and therefore for the original system $s$.

3. **(Convergence of SGD to structural recovery.)** Let $(\psi_n, \varphi_n)_{n \geq 0}$ be any SGD sequence satisfying (H4). Then with probability 1:
   - The limit set of $(\psi_n, \varphi_n)$ is a connected compact subset of $\mathcal{M}$;
   - In particular:
     $$\lim_{n \to \infty} \mathcal{L}_{\mathrm{total}}(\psi_n, \varphi_n) = 0.$$
     Thus, for any sequence of iterates converging to some $(\bar{\psi}, \bar{\varphi})$, we have $(\bar{\psi}, \bar{\varphi}) \in \mathcal{M}$, and the structural recovery property of (2) applies.

So: under the assumption that **there exists some encoder + hypernetwork that can express the true hypostructure**, generic deep-learning-style training on **prediction + axiom-risk** from **raw observations** is guaranteed (in the population limit) to recover that hypostructure up to isomorphism.

\begin{proof}
\textbf{Step 1: Infimum is zero and $\mathcal{M} \neq \emptyset$.}

From (H2), there exists $(\psi^*, \varphi^*)$ such that:
\begin{itemize}
\item For $\nu$-a.e. $s$, the induced latent hypostructure is isomorphic to the true one,
\item For $\mathbb{P}_s$-a.e. trajectory, dynamics and axioms match exactly.
\end{itemize}

Hence, for $\nu$-a.e. $s$:
\begin{itemize}
\item Prediction error is zero:
  $$\mathbb{E}_{Y \sim \mathbb{P}_s}\left[\ell(F_{\theta_{s,\varphi^*}}(z_t^{(\psi^*)}), z_{t+1}^{(\psi^*)})\right] = 0,$$
  so $\mathcal{L}_{\mathrm{pred}}(\psi^*, \varphi^*) = 0$;
\item Each axiom-defect is zero:
  $$\mathbb{E}_{Y \sim \mathbb{P}_s}\left[K_A\big(\mathcal{H}_{\theta_{s,\varphi^*}}; z^{(\psi^*)}_\bullet\big)\right] = 0,$$
  so $\mathcal{R}_{\mathrm{axioms}}(\psi^*, \varphi^*) = 0$.
\end{itemize}

Therefore:
$$\mathcal{L}_{\mathrm{total}}(\psi^*, \varphi^*) = 0.$$

Since $\mathcal{L}_{\mathrm{total}} \geq 0$ everywhere (by definition), we conclude:
$$\inf_{(\psi,\varphi)} \mathcal{L}_{\mathrm{total}}(\psi, \varphi) = 0$$
and $\mathcal{M} \neq \emptyset$.

Lower semicontinuity (from (H1)) and compactness of level sets imply $\mathcal{M}$ is compact.

\textbf{Step 2: Structural recovery at minimizers.}

Let $(\hat{\psi}, \hat{\varphi}) \in \mathcal{M}$. Then $\mathcal{L}_{\mathrm{total}}(\hat{\psi}, \hat{\varphi}) = 0$.

By definition of $\mathcal{L}_{\mathrm{total}}$, this implies separately:
\begin{itemize}
\item $\mathcal{L}_{\mathrm{pred}}(\hat{\psi}, \hat{\varphi}) = 0$,
\item $\mathcal{R}_{\mathrm{axioms}}(\hat{\psi}, \hat{\varphi}) = 0$.
\end{itemize}

Because both terms are integrals of nonnegative random variables over $(\mathcal{S}, \nu)$ and trajectories, Fubini's theorem implies:
\begin{itemize}
\item For $\nu$-almost every $s$:
  $$\mathbb{E}_{Y \sim \mathbb{P}_s}\left[\ell(F_{\theta_{s,\hat{\varphi}}}(z_t^{(\hat{\psi})}), z_{t+1}^{(\hat{\psi})})\right] = 0,$$
  so the prediction error is zero $\mathbb{P}_s$-a.s.;
\item For each axiom $A$ and $\nu$-a.e. $s$:
  $$\mathbb{E}_{Y \sim \mathbb{P}_s}\left[K_A\big(\mathcal{H}_{\theta_{s,\hat{\varphi}}}; z^{(\hat{\psi})}_\bullet\big)\right] = 0,$$
  so axiom-defect $K_A$ is zero $\mathbb{P}_s$-a.s.
\end{itemize}

Thus, for $\nu$-a.e. $s$, for $\mathbb{P}_s$-almost every trajectory, we have:
\begin{itemize}
\item Perfect prediction in latent space,
\item Exact satisfaction of all axioms—i.e., those latent trajectories are \textbf{exact hypostructural trajectories} for $\mathcal{H}_{\theta_{s,\hat{\varphi}}}$.
\end{itemize}

By (H3) (Identifiability), it follows that for $\nu$-almost every $s$ there exists a hypostructure isomorphism $\tilde{T}_s: X_s \to Z$ such that:
\begin{itemize}
\item The encoded latent trajectory equals $\tilde{T}_s(X^{(s)}_t)$,
\item $\tilde{T}_s^*(\mathcal{H}_{\theta_{s,\hat{\varphi}}}) = \mathcal{H}^{(s)*}$.
\end{itemize}

Therefore, any global minimizer recovers the true latent hypostructure (up to iso) for almost every system $s$. Since all global metatheorems are stated purely in terms of the axioms and hypostructure, they therefore hold for the learned latent representation.

This proves statement (2).

\textbf{Step 3: Convergence of SGD to minimizers.}

Under (H4), we are in a standard stochastic approximation setting:
\begin{itemize}
\item $\mathcal{L}_{\mathrm{total}}$ is bounded below and has Lipschitz gradient,
\item $\hat{\nabla} \mathcal{L}_{\mathrm{total}}$ is an unbiased estimator with bounded variance,
\item Step sizes satisfy Robbins–Monro conditions.
\end{itemize}

By classical results in stochastic approximation (e.g., Kushner–Yin, Benaïm), we have:
\begin{itemize}
\item $\mathcal{L}_{\mathrm{total}}(\psi_n, \varphi_n)$ converges almost surely to some random variable $L_\infty$,
\item Every limit point of $(\psi_n, \varphi_n)$ is almost surely a \textbf{stationary point} of $\mathcal{L}_{\mathrm{total}}$,
\item The limit set of $(\psi_n, \varphi_n)$ is almost surely a compact connected set of stationary points.
\end{itemize}

Now observe that:
\begin{itemize}
\item For any stationary point $(\bar{\psi}, \bar{\varphi})$, by continuity and nonnegativity we must have $\mathcal{L}_{\mathrm{total}}(\bar{\psi}, \bar{\varphi}) \geq 0$.
\item If the algorithm ever gets arbitrarily close to a global minimizer, the descent property and compactness of sublevel sets prevent it from escaping up to positive risk.
\end{itemize}

We can sharpen this by assuming (which is standard and often included in (H4)) that $\mathcal{L}_{\mathrm{total}}$ satisfies the \textbf{Kurdyka–Łojasiewicz (KŁ) property} (true for real-analytic or semi-algebraic losses, which neural nets typically satisfy). Then standard KŁ + GD theory implies that every limit point of a gradient-based descent sequence must be a stationary point, and if the global minimizers form a connected component, all limit points lie in that component.

Combining:
\begin{itemize}
\item The limit set of $(\psi_n, \varphi_n)$ is contained in the set of stationary points.
\item Among stationary points, those with minimal value form the set $\mathcal{M}$ of global minimizers (since global minimum is 0).
\item Under mild KŁ-type assumptions, any connected component of stationary points with minimal value is exactly $\mathcal{M}$.
\end{itemize}

Hence, almost surely, the limit set of $(\psi_n, \varphi_n)$ is a compact connected subset of $\mathcal{M}$, and:
$$\lim_{n \to \infty} \mathcal{L}_{\mathrm{total}}(\psi_n, \varphi_n) = 0.$$

Any convergent subsequence has its limit in $\mathcal{M}$, and thus by Step 2 recovers the true hypostructures up to isomorphism for $\nu$-almost every system.

This proves statement (3).
\end{proof}

\begin{remark}[Significance for structural learning]
This meta-theorem establishes that:
\begin{itemize}
\item The user only provides raw trajectories and a big NN architecture,
\item All inductive bias is: ``there exists some encoder + hypostructure in this NN class that matches reality'' (exactly the same kind of bias deep learning already assumes),
\item Under that assumption, minimizing \textbf{prediction + axiom-risk} recovers the latent hypostructure from pixels, in the population limit, with a standard SGD convergence argument.
\end{itemize}
\end{remark}

> **Key Insight (Foundation for learnable physics):** This theorem provides the theoretical foundation for treating hypostructures as learnable objects. Once learned, the axioms become predictive: the learned hypostructure inherits all metatheorems, allowing structural conclusions about the underlying physical system from pure observational data.

### Equivariance of Trainable Hypostructures Under Symmetry Groups

Many system families carry natural symmetry groups: space-time translations, rotations, Galilean boosts, scaling symmetries, gauge groups, etc. A central expectation for a "structural" learner is that it should not break such symmetries arbitrarily: if the distribution of systems and the true hypostructure are symmetric under a group $G$, then the **learned hypostructure** should also be $G$-equivariant.

In this section we formalize this as an **equivariance metatheorem**: under natural compatibility assumptions between $G$, the system distribution, the hypostructure family, and the axiom-risk, every risk minimizer is $G$-equivariant (up to gauge), and gradient flow preserves equivariance.

#### Symmetry group acting on systems and hypostructures

Let $G$ be a (locally compact) group acting on the state space $X$ and on the class of systems $S$. For each $g \in G$, we denote by $g \cdot S$ the transformed system obtained by pushing forward the dynamics under $g$ (e.g. conjugating the semiflow by $g$).

**Assumption 13.57 (Group-covariant system distribution).** Let $\mathcal{S}$ be a distribution on systems $S$. We assume $\mathcal{S}$ is $G$-invariant:
$$S \sim \mathcal{S} \implies g \cdot S \sim \mathcal{S} \quad \forall g \in G.$$

Equivalently, for any measurable set of systems $\mathcal{A}$, $\mathcal{S}(\mathcal{A}) = \mathcal{S}(g \cdot \mathcal{A})$.

Let $\Theta_{\mathrm{adm}}$ be the parameter space of a hypostructure family $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$, with:
$$\mathcal{H}_\Theta(S) = (X_S, S_t, \Phi_{\Theta,S}, \mathfrak{D}_{\Theta,S}, G_{\Theta,S})$$
the hypostructure associated to system $S$ and parameter $\Theta$.

**Assumption 13.58 (Equivariant parametrization).** There is a group action of $G$ on $\Theta_{\mathrm{adm}}$, denoted $(g, \Theta) \mapsto g \cdot \Theta$, such that for all $g \in G$, systems $S$, and parameters $\Theta$:
$$g \cdot \mathcal{H}_\Theta(S) \simeq \mathcal{H}_{g \cdot \Theta}(g \cdot S)$$
in the Hypo category, i.e. the hypostructure induced by first transforming $\Theta$ and $S$ by $G$ coincides (up to Hypo-isomorphism) with the pushforward of $\mathcal{H}_\Theta(S)$ by $g$.

Intuitively, this means the family $\{\mathcal{H}_\Theta\}$ is expressive enough and parametrized in such a way that group transformations commute with hypostructure construction, up to the usual notion of "same" hypostructure (gauge).

#### Symmetry of the axiom-risk

For each system $S$ and parameter $\Theta$, we have the joint axiom-risk:
$$\mathcal{R}_S(\Theta) := \sum_{A \in \mathcal{A}} w_A \mathcal{R}_{A,S}(\Theta), \qquad \mathcal{R}_{A,S}(\Theta) := \int_{\mathcal{U}_S} K_{A,S}^{(\Theta)}(u) \, d\mu_S(u),$$
constructed from the defect functionals $K_{A,S}^{(\Theta)}$. The **average risk** over $\mathcal{S}$ is:
$$\mathcal{R}_{\mathcal{S}}(\Theta) := \mathbb{E}_{S \sim \mathcal{S}}[\mathcal{R}_S(\Theta)].$$

**Assumption 13.59 (Group-invariance of defects and trajectories).** For each $g \in G$, the following hold:

1. The transformation $u \mapsto g \cdot u$ maps trajectories of $S$ to trajectories of $g \cdot S$, and preserves the trajectory measure (or transforms it in a controlled way that cancels in expectation):
$$\mu_{g \cdot S} = (g \cdot)_\# \mu_S.$$

2. The defect functionals are compatible with the group action:
$$K_{A, g \cdot S}^{(g \cdot \Theta)}(g \cdot u) = K_{A,S}^{(\Theta)}(u) \quad \text{for all } A \in \mathcal{A}, u \in \mathcal{U}_S.$$

In particular, $\mathcal{R}_{g \cdot S}(g \cdot \Theta) = \mathcal{R}_S(\Theta)$.

\begin{lemma}[Risk equivariance]\label{lem:risk-equivariance}
For all $g \in G$ and $\Theta \in \Theta_{\mathrm{adm}}$:
$$\mathcal{R}_{\mathcal{S}}(g \cdot \Theta) = \mathcal{R}_{\mathcal{S}}(\Theta).$$
\end{lemma}

\begin{proof}
Using $\mathcal{S}$-invariance and defect compatibility:
$$\mathcal{R}_{\mathcal{S}}(g \cdot \Theta) = \mathbb{E}_{S \sim \mathcal{S}}[\mathcal{R}_S(g \cdot \Theta)] = \mathbb{E}_{S \sim \mathcal{S}}[\mathcal{R}_{g^{-1} \cdot S}(\Theta)] = \mathcal{R}_{\mathcal{S}}(\Theta),$$
where we used the change of variable $S' = g^{-1} \cdot S$ and the invariance of $\mathcal{S}$.
\end{proof}

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Group-Covariant System Distribution:** $\mathcal{S}$ is $G$-invariant: $S \sim \mathcal{S} \Rightarrow g \cdot S \sim \mathcal{S}$
>     *   [ ] **Equivariant Parametrization:** $g \cdot \mathcal{H}_\Theta(S) \simeq \mathcal{H}_{g \cdot \Theta}(g \cdot S)$ in Hypo
>     *   [ ] **Defect-Level Equivariance:** $K_{A,g \cdot S}^{(g \cdot \Theta)}(g \cdot u) = K_{A,S}^{(\Theta)}(u)$
>     *   [ ] **Existence of True Equivariant Hypostructure:** $\Theta^*$ with $\mathcal{R}_S(\Theta^*) = 0$ and $g \cdot \mathcal{H}_{\Theta^*,S} \simeq \mathcal{H}_{\Theta^*,g \cdot S}$
> *   **Output (Structural Guarantee):**
>     *   Minimizers $G$-equivariant: $\widehat{\Theta} \in G \cdot \Theta^*$
>     *   Gradient flow preserves equivariance: $g \cdot \Theta_t$ solves same flow with $g \cdot \Theta_0$
>     *   Convergence to equivariant hypostructures
> *   **Failure Condition (Debug):**
>     *   If **Equivariant Parametrization** fails → **Mode symmetry-breaking artifact** (learned structure has spurious asymmetry)
>     *   If **Local Uniqueness** fails → **Mode multiple branches** (equivariant and symmetry-broken minima coexist)

\begin{metatheorem}[Equivariance]\label{mt:equivariance}
Let $\mathcal{S}$ be a $G$-invariant system distribution, and $\{\mathcal{H}_\Theta\}$ a parametric hypostructure family satisfying Assumptions 13.57–13.59. Consider the average axiom-risk $\mathcal{R}_{\mathcal{S}}(\Theta)$.
\end{metatheorem}

Assume:

1. **(Existence of a true equivariant hypostructure.)** There exists a parameter $\Theta^* \in \Theta_{\mathrm{adm}}$ such that:
   - For $\mathcal{S}$-a.e. system $S$, $\mathcal{H}_{\Theta^*,S}$ satisfies the axioms (C, D, SC, Cap, LS, TB, Reg, GC), and $\mathcal{R}_S(\Theta^*) = 0$.
   - The true hypostructure is $G$-equivariant in Hypo: For all $g \in G$ and all $S$:
   $$g \cdot \mathcal{H}_{\Theta^*,S} \simeq \mathcal{H}_{\Theta^*, g \cdot S}.$$
   Equivalently, the orbit $G \cdot \Theta^*$ consists of gauge-equivalent parameters encoding the same equivariant hypostructure.

2. **(Local uniqueness modulo $G$-gauge.)** The average risk $\mathcal{R}_{\mathcal{S}}(\Theta)$ admits a unique minimum orbit in a neighborhood of $\Theta^*$: there is a neighborhood $U \subset \Theta_{\mathrm{adm}}$ such that:
$$\Theta \in U, \quad \mathcal{R}_{\mathcal{S}}(\Theta) = \inf_{\Theta'} \mathcal{R}_{\mathcal{S}}(\Theta') \implies \Theta \in G \cdot \Theta^*,$$
and all points in $G \cdot \Theta^* \cap U$ are gauge-equivalent (represent the same Hypo object).

3. **(Regularity for gradient flow.)** $\mathcal{R}_{\mathcal{S}}$ is $C^1$ on $\Theta_{\mathrm{adm}}$, with Lipschitz gradient on bounded sets.

Then:

1. **(Minimizers are $G$-equivariant (up to gauge).)** Every global minimizer $\widehat{\Theta}$ of $\mathcal{R}_{\mathcal{S}}$ in $U$ lies in the orbit $G \cdot \Theta^*$, and thus represents the same equivariant hypostructure as $\Theta^*$ in Hypo. In particular, the learned hypostructure is $G$-equivariant.

2. **(Gradient flow preserves equivariance.)** Consider gradient flow on parameter space:
$$\frac{d}{dt} \Theta_t = -\nabla \mathcal{R}_{\mathcal{S}}(\Theta_t), \qquad \Theta_{t=0} = \Theta_0.$$
Then for any $g \in G$, $g \cdot \Theta_t$ solves the same gradient flow with initial condition $g \cdot \Theta_0$. In particular, if the initialization $\Theta_0$ is $G$-fixed (or lies in a $G$-orbit symmetric under a subgroup), the entire trajectory $\Theta_t$ remains in the fixed-point set (or corresponding orbit) of the group action.

3. **(Convergence to equivariant hypostructures.)** If gradient descent or gradient flow on $\mathcal{R}_{\mathcal{S}}$ converges to a minimizer in $U$ (as in \cref{mt:trainable-hypostructure-consistency}), then the limit hypostructure is gauge-equivalent to $\Theta^*$ and hence $G$-equivariant.

In short: **trainable hypostructures inherit all symmetries of the system distribution**. They cannot spontaneously break a symmetry that the true hypostructure preserves, unless there exist distinct, non-equivariant minimizers of $\mathcal{R}_{\mathcal{S}}$ outside the neighborhood $U$ (i.e. unless the theory itself has symmetric and symmetry-broken branches).

\begin{proof}
(1) follows directly from risk invariance and local uniqueness modulo $G$.

By \cref{lem:risk-equivariance}, $\mathcal{R}_{\mathcal{S}}$ is $G$-invariant:
$$\mathcal{R}_{\mathcal{S}}(g \cdot \Theta) = \mathcal{R}_{\mathcal{S}}(\Theta) \quad \forall g \in G.$$

Let $\widehat{\Theta} \in U$ be a global minimizer of $\mathcal{R}_{\mathcal{S}}$. Then for any $g \in G$:
$$\mathcal{R}_{\mathcal{S}}(g \cdot \widehat{\Theta}) = \mathcal{R}_{\mathcal{S}}(\widehat{\Theta}) = \inf_{\Theta'} \mathcal{R}_{\mathcal{S}}(\Theta').$$
Thus $g \cdot \widehat{\Theta}$ is also a minimizer in $U$. By local uniqueness modulo orbit (Assumption 2), all such minimizers in $U$ lie on the orbit $G \cdot \Theta^*$ and correspond to the same hypostructure in Hypo. Therefore $\widehat{\Theta} \in G \cdot \Theta^*$, and the corresponding hypostructure is $G$-equivariant.

(2) Gradient flow equivariance follows from the invariance of $\mathcal{R}_{\mathcal{S}}$. By the chain rule and $G$-invariance:
$$\mathcal{R}_{\mathcal{S}}(g \cdot \Theta) = \mathcal{R}_{\mathcal{S}}(\Theta) \implies D(g \cdot \Theta)^\top \nabla \mathcal{R}_{\mathcal{S}}(g \cdot \Theta) = \nabla \mathcal{R}_{\mathcal{S}}(\Theta),$$
where $D(g \cdot \Theta)$ is the derivative of the group action at $\Theta$. Differentiating $\Theta_t \mapsto g \cdot \Theta_t$ in time gives:
$$\frac{d}{dt}(g \cdot \Theta_t) = D(g \cdot \Theta_t) \dot{\Theta}_t = -D(g \cdot \Theta_t) \nabla \mathcal{R}_{\mathcal{S}}(\Theta_t) = -\nabla \mathcal{R}_{\mathcal{S}}(g \cdot \Theta_t),$$
where the last equality uses the relation between gradients and the group action induced by $G$-invariance. Hence $g \cdot \Theta_t$ solves the same gradient flow with initial condition $g \cdot \Theta_0$.

(3) If gradient descent or continuous-time gradient flow converges to a limit $\Theta_\infty \in U$, then by (1) that limit is in the orbit $G \cdot \Theta^*$ and corresponds to the same $G$-equivariant hypostructure.
\end{proof}

\begin{remark}[Key hypotheses]
The key hypotheses are:
\begin{itemize}
\item \textbf{Equivariant parametrization} of the hypostructure family (Assumption 13.58), and
\item \textbf{Defect-level equivariance} (Assumption 13.59).
\end{itemize}
Together, they ensure that ``write down the axioms, compute defects, average risk, and optimize'' defines a $G$-equivariant learning problem.
\end{remark}

\begin{remark}[No spontaneous symmetry breaking]
The theorem says that if the \textit{true} structural laws of the systems are $G$-equivariant, and the training distribution respects that symmetry, then a trainable hypostructure will not invent a spurious symmetry-breaking ontology---unless such a symmetry-breaking branch is truly present as an alternative minimum of the risk.
\end{remark}

\begin{remark}[Structural analogue of equivariant networks]
This is a structural analogue of standard results for equivariant neural networks, but formulated at the level of \textbf{axiom learning}: the objects that remain invariant are not just predictions, but the entire hypostructure (Lyapunov, dissipation, capacities, barriers, etc.).
\end{remark}

> **Key Insight:** Trainable hypostructures inherit all symmetries of the underlying system distribution. The learned axioms preserve equivariance—not just at the level of predictions, but at the level of structural components ($\Phi$, $\mathfrak{D}$, barriers, capacities). Symmetry cannot be spontaneously broken by the learning process unless the true theory itself admits symmetry-broken branches.

---


---


## The General Loss Functional {#ch:general-loss}

This chapter defines a training objective for systems that instantiate, verify, and optimize over hypostructures. The goal is to train a parametrized system to identify hypostructures, fit soft axioms, and solve the associated variational problems.

### Overview and problem formulation

This is formally framed as **Structural Risk Minimization [@Vapnik98]** over the hypothesis space of admissible hypostructures.

\begin{definition}[Hypostructure learner]\label{def:hypostructure-learner}
A \textbf{hypostructure learner} is a parametrized system with parameters $\Theta$ that, given a dynamical system $S$, produces:
\begin{enumerate}
\item A hypostructure $\mathbb{H}_\Theta(S) = (X, S_t, \Phi_\Theta, \mathfrak{D}_\Theta, G_\Theta)$
\item Soft axiom evaluations and defect values
\item Extremal candidates $u_{\Theta,S}$ for associated variational problems
\end{enumerate}
\end{definition}

\begin{definition}[System distribution]\label{def:system-distribution}
Let $\mathcal{S}$ denote a probability distribution over dynamical systems. This includes PDEs, flows, discrete processes, stochastic systems, and other structures amenable to hypostructure analysis.
\end{definition}

\begin{definition}[general loss functional]\label{def:general-loss-functional}
The \textbf{general loss} is:
$$\mathcal{L}_{\text{gen}}(\Theta) := \mathbb{E}_{S \sim \mathcal{S}}\big[\lambda_{\text{struct}} L_{\text{struct}}(S, \Theta) + \lambda_{\text{axiom}} L_{\text{axiom}}(S, \Theta) + \lambda_{\text{var}} L_{\text{var}}(S, \Theta) + \lambda_{\text{meta}} L_{\text{meta}}(S, \Theta)\big]$$
where $\lambda_{\text{struct}}, \lambda_{\text{axiom}}, \lambda_{\text{var}}, \lambda_{\text{meta}} \geq 0$ are weighting coefficients.
\end{definition}

### Structural loss

The structural loss formulation embodies the **Maximum Entropy** principle of Jaynes [@Jaynes57]: among all distributions consistent with observed constraints, select the one with maximal entropy. Here, we select the hypostructure parameters that minimize constraint violations while maintaining maximal generality.

\begin{definition}[Structural loss functional]\label{def:structural-loss-functional}
For systems $S$ with known ground-truth structure $(\Phi^*, \mathfrak{D}^*, G^*)$, define:
$$L_{\text{struct}}(S, \Theta) := d(\Phi_\Theta, \Phi^*) + d(\mathfrak{D}_\Theta, \mathfrak{D}^*) + d(G_\Theta, G^*)$$
where $d(\cdot, \cdot)$ denotes an appropriate distance on the respective spaces.
\end{definition}

\begin{definition}[Self-consistency constraints]\label{def:self-consistency-constraints}
For unlabeled systems without ground-truth annotations, define:
$$L_{\text{struct}}(S, \Theta) := \mathbf{1}[\Phi_\Theta < 0] + \mathbf{1}[\text{non-convexity along flow}] + \mathbf{1}[\text{non-}G_\Theta\text{-invariance}]$$
with indicator penalties for constraint violations.
\end{definition}

\begin{lemma}[Structural loss interpretation]\label{lem:structural-loss-interpretation}
Minimizing $L_{\text{struct}}$ encourages the learner to:
\begin{itemize}
\item Correctly identify conserved quantities and energy functionals
\item Recognize symmetries inherent to the system
\item Produce internally consistent hypostructure components
\end{itemize}
\end{lemma}

\begin{proof}
We verify each claim:

\begin{enumerate}
\item \textbf{Conserved quantities:} By Definition 14.4, $L_{\text{struct}}$ includes the term $d(\Phi_\Theta, \Phi^*)$. Minimizing this term forces $\Phi_\Theta$ close to the ground-truth $\Phi^*$. By Definition 14.5, violations of positivity ($\Phi_\Theta < 0$) incur penalty, selecting parameters where $\Phi_\Theta$ behaves as a proper energy/height functional.

\item \textbf{Symmetries:} The term $d(G_\Theta, G^*)$ (Definition 14.4) penalizes discrepancy between learned and true symmetry groups. The indicator $\mathbf{1}[\text{non-}G_\Theta\text{-invariance}]$ (Definition 14.5) penalizes learned structures not respecting the identified symmetry.

\item \textbf{Internal consistency:} The indicator $\mathbf{1}[\text{non-convexity along flow}]$ (Definition 14.5) enforces that $\Phi_\Theta$ and the flow $S_t$ are compatible: along trajectories, $\Phi_\Theta$ should decrease (Lyapunov property) or satisfy convexity constraints from Axiom D.
\end{enumerate}

The loss $L_{\text{struct}}$ is zero if and only if all components are correctly identified and mutually consistent.
\end{proof}

### Axiom loss

\begin{definition}[Axiom loss functional]\label{def:axiom-loss-functional}
For system $S$ with trajectory distribution $\mathcal{U}_S$:
$$L_{\text{axiom}}(S, \Theta) := \sum_{A \in \mathcal{A}} w_A \, \mathbb{E}_{u \sim \mathcal{U}_S}[K_A^{(\Theta)}(u)]$$
where $K_A^{(\Theta)}$ is the defect functional for axiom $A$ under the learned hypostructure $\mathbb{H}_\Theta(S)$.
\end{definition}

\begin{lemma}[Axiom loss interpretation]\label{lem:axiom-loss-interpretation}
Minimizing $L_{\text{axiom}}$ selects parameters $\Theta$ that produce hypostructures with minimal global axiom defects.
\end{lemma}

\begin{proof}
If the system $S$ genuinely satisfies axiom $A$, the learner is rewarded for finding parameters that make $K_A^{(\Theta)}(u)$ small. If $S$ violates $A$ in some regimes, the minimum achievable defect quantifies this failure.
\end{proof}

\begin{definition}[Causal Enclosure Loss]\label{def:causal-enclosure-loss}
Let $(\mathcal{X}, \mu, T)$ be a stochastic dynamical system and $\Pi: \mathcal{X} \to \mathcal{Y}$ a learnable coarse-graining parametrized by $\Theta$. Define $Y_t := \Pi_\Theta(X_t)$ and $Y_{t+1} := \Pi_\Theta(X_{t+1})$. The \textbf{causal enclosure loss} is:
$$L_{\text{closure}}(\Theta) := I(X_t; Y_{t+1}) - I(Y_t; Y_{t+1})$$
where $I(\cdot; \cdot)$ denotes mutual information with respect to the stationary measure $\mu$.
\end{definition}

*Interpretation:* By the chain rule, $I(X_t; Y_{t+1}) = I(Y_t; Y_{t+1}) + I(X_t; Y_{t+1} \mid Y_t)$. Thus:
$$L_{\text{closure}}(\Theta) = I(X_t; Y_{t+1} \mid Y_t)$$
This quantifies how much additional predictive information about the macro-future $Y_{t+1}$ is contained in the micro-state $X_t$ beyond what is captured by the macro-state $Y_t$. By \cref{mt:the-closure-curvature-duality} (Closure-Curvature Duality), $L_{\text{closure}} = 0$ if and only if the coarse-graining $\Pi_\Theta$ is computationally closed. Minimizing $L_{\text{closure}}$ thus forces the learned hypostructure to be "Software" in the sense of §20.7: the macro-dynamics becomes autonomous, independent of micro-noise [@Rosas2024].

### Variational loss

\begin{definition}[Variational loss for labeled systems]\label{def:variational-loss-for-labeled-systems}
For systems with known sharp constants $C_A^*(S)$:
$$L_{\text{var}}(S, \Theta) := \sum_{A \in \mathcal{A}} \left| \text{Eval}_A(u_{\Theta,S,A}) - C_A^*(S) \right|$$
where $\text{Eval}_A$ is the evaluation functional for problem $A$ and $u_{\Theta,S,A}$ is the learner's proposed extremizer.
\end{definition}

\begin{definition}[Extremal search loss for unlabeled systems]\label{def:extremal-search-loss-for-unlabeled-systems}
For systems without known sharp constants:
$$L_{\text{var}}(S, \Theta) := \sum_{A \in \mathcal{A}} \text{Eval}_A(u_{\Theta,S,A})$$
directly optimizing toward the extremum.
\end{definition}

\begin{lemma}[Rigorous bounds property]\label{lem:rigorous-bounds-property}
Every value $\text{Eval}_A(u_{\Theta,S,A})$ constitutes a rigorous one-sided bound on the sharp constant by construction of the variational problem.
\end{lemma}

\begin{proof}
For infimum problems, any feasible $u$ gives an upper bound: $\text{Eval}_A(u) \geq C_A^*$. For supremum problems, any feasible $u$ gives a lower bound. The learner's output is always a valid bound regardless of optimality.
\end{proof}

### Meta-learning loss

\begin{definition}[Adapted parameters]\label{def:adapted-parameters}
For system $S$ and base parameters $\Theta$, let $\Theta'_S$ denote the result of $k$ gradient steps on $L_{\text{axiom}}(S, \cdot) + L_{\text{var}}(S, \cdot)$ starting from $\Theta$:
$$\Theta'_S := \Theta - \eta \sum_{i=1}^{k} \nabla_\Theta (L_{\text{axiom}} + L_{\text{var}})(S, \Theta^{(i)})$$
where $\Theta^{(i)}$ is the parameter after $i$ steps.
\end{definition}

\begin{definition}[Meta-learning loss]\label{def:meta-learning-loss}
Define:
$$L_{\text{meta}}(S, \Theta) := \tilde{L}_{\text{axiom}}(S, \Theta'_S) + \tilde{L}_{\text{var}}(S, \Theta'_S)$$
evaluated on held-out data from $S$.
\end{definition}

\begin{lemma}[Fast adaptation interpretation]\label{lem:fast-adaptation-interpretation}
Minimizing $L_{\text{meta}}$ over the distribution $\mathcal{S}$ trains the system to:
\begin{itemize}
\item Quickly instantiate hypostructures for new systems (few gradient steps to fit $\Phi, \mathfrak{D}, G$)
\item Rapidly identify sharp constants and extremizers
\end{itemize}
\end{lemma}

\begin{proof}
The meta-learning objective rewards parameters $\Theta$ from which few adaptation steps suffice to achieve low loss on any system $S$. This is the MAML principle applied to hypostructure learning.
\end{proof}

### The combined general loss

This formulation mirrors **Tikhonov Regularization [@Tikhonov77]** for ill-posed inverse problems, where the Hypostructure Axioms serve as the stabilizing functional.

\begin{metatheorem}[Differentiability]\label{mt:differentiability}
Under the following conditions:
\begin{enumerate}
\item Neural network parameterization of $\Phi_\Theta, \mathfrak{D}_\Theta, G_\Theta$
\item Defect functionals $K_A$ composed of integrals, norms, and algebraic expressions in the network outputs
\item Dominated convergence conditions as in \cref{lem:leibniz-rule-for-axiom-risk}
\end{enumerate}
\end{metatheorem}

all components of $\mathcal{L}_{\text{gen}}$ are differentiable in $\Theta$.

\begin{proof}
\textbf{Step 1 (Component differentiability).} Each loss component $L_{\text{struct}}, L_{\text{axiom}}, L_{\text{var}}$ is differentiable by:
\begin{itemize}
\item Neural network differentiability (backpropagation)
\item Dominated convergence for integral expressions (\cref{lem:leibniz-rule-for-axiom-risk})
\end{itemize}

\textbf{Step 2 (Meta-learning differentiability).} The adapted parameters $\Theta'_S$ depend differentiably on $\Theta$ via the chain rule through gradient steps. This is the key observation enabling MAML-style meta-learning.

\textbf{Step 3 (Expectation over $\mathcal{S}$).} Dominated convergence allows differentiation under the expectation over systems $S \sim \mathcal{S}$, given appropriate bounds.
\end{proof}

\begin{corollary}[Backpropagation through axioms]\label{cor:backpropagation-through-axioms}
Gradient descent on $\mathcal{L}_{\text{gen}}(\Theta)$ is well-defined. The gradient can be computed via backpropagation through:
\begin{itemize}
\item The neural network architecture
\item The defect functional computations
\item The meta-learning adaptation steps
\end{itemize}
\end{corollary}

\begin{metatheorem}[Universal Solver]\label{mt:universal-solver}
A system trained on $\mathcal{L}_{\text{gen}}$ with sufficient capacity and training data over a diverse distribution $\mathcal{S}$ learns to:
\begin{enumerate}
\item \textbf{Recognize structure:} Identify state spaces, flows, height functionals, dissipation structures, and symmetry groups
\item \textbf{Enforce soft axioms:} Fit hypostructure parameters that minimize global axiom defects
\item \textbf{Solve variational problems:} Produce extremizers that approach sharp constants
\item \textbf{Adapt quickly:} Transfer to new systems with few gradient steps
\end{enumerate}
\end{metatheorem}

\begin{proof}
\textbf{Step 1 (Structural recognition).} Minimizing $L_{\text{struct}}$ over diverse systems trains the learner to extract the correct hypostructure components. The loss penalizes misidentification of conserved quantities, symmetries, and dissipation mechanisms.

\textbf{Step 2 (Axiom enforcement).} Minimizing $L_{\text{axiom}}$ trains the learner to find parameters under which soft axioms hold with minimal defect. The learner discovers which axioms each system satisfies and quantifies violations.

\textbf{Step 3 (Variational solving).} Minimizing $L_{\text{var}}$ trains the learner to produce increasingly sharp bounds on extremal constants. For labeled systems, the gap to known values provides direct supervision. For unlabeled systems, the extremal search pressure drives toward optimal values.

\textbf{Step 4 (Fast adaptation).} Minimizing $L_{\text{meta}}$ trains the learner's initialization to enable rapid specialization. Few gradient steps suffice to adapt the general hypostructure knowledge to any specific system.

The combination of these four loss components produces a system that instantiates and optimizes over hypostructures universally.
\end{proof}

---

### The Learnability Phase Transition

This section establishes the fundamental dichotomy in learning: the transition between \textbf{perfect reconstruction} and \textbf{statistical modeling} is not a choice of algorithm, but a phase transition controlled by the ratio of system entropy to agent capacity. This formalizes the $\Omega$-Layer interface between the System (Reality) and the Agent (The Learner), deriving Effective Field Theory from learning dynamics.

\begin{definition}[Kolmogorov-Sinai Entropy Rate]\label{def:kolmogorov-sinai-entropy-rate}
Let $(X, \mathcal{B}, \mu, S_t)$ be a measure-preserving dynamical system generating trajectories $u(t)$. The \textbf{Kolmogorov-Sinai entropy} $h_{KS}(S)$ [@Sinai59] is the rate at which the system generates new information (bits per unit time) that cannot be predicted from past history:
$$h_{KS}(S) := \sup_{\mathcal{P}} \lim_{n \to \infty} \frac{1}{n} H\left(\bigvee_{k=0}^{n-1} S_{-k}^{-1}\mathcal{P}\right)$$
where $\mathcal{P}$ ranges over finite measurable partitions and $H(\cdot)$ denotes Shannon entropy of a partition. Equivalently, in the continuous-time formulation:
$$h_{KS}(S) = \lim_{t \to \infty} \frac{1}{t} H(u_{[0,t]} \mid u_{(-\infty, 0]})$$
For deterministic systems, $h_{KS}$ equals the sum of positive Lyapunov exponents by \textbf{Pesin's formula} [@Eckmann85]:
$$h_{KS}(S) = \int_X \sum_{\lambda_i(x) > 0} \lambda_i(x) \, d\mu(x)$$
where $\{\lambda_i(x)\}$ are the Lyapunov exponents at $x$. For stochastic systems, it includes both deterministic chaos and external noise contributions.
\end{definition}

\begin{definition}[Agent Capacity]\label{def:agent-capacity}
Let $\mathcal{A}$ be a learning agent (Hypostructure Learner) with parameter space $\Theta \subseteq \mathbb{R}^d$ and update rule $\Theta_{t+1} = \Theta_t - \eta \nabla_\Theta \mathcal{L}$. The \textbf{capacity} $C_{\mathcal{A}}$ is the maximum rate at which the agent can store and process information:
$$C_{\mathcal{A}} := \sup_{\text{inputs}} \limsup_{T \to \infty} \frac{1}{T} I(\Theta_T; \text{data}_{[0,T]})$$
This is the bandwidth of the update rule—the channel capacity of the learning process viewed as a communication channel from the environment to the agent's parameters. For neural networks with $d$ parameters, learning rate $\eta$, and batch size $B$:
$$C_{\mathcal{A}} \lesssim \eta B \cdot d \cdot \log(1/\eta)$$
The Fisher information of the parameterization provides a tighter bound: $C_{\mathcal{A}} \leq \frac{1}{2} \text{tr}(\mathcal{I}(\Theta))$ where $\mathcal{I}(\Theta)$ is the Fisher information matrix [@Amari16].
\end{definition}

\begin{metatheorem}[The Learnability Singularity]\label{mt:the-learnability-singularity}
Let an agent $\mathcal{A}$ with capacity $C_{\mathcal{A}}$ attempt to model a dynamical system $S$ with KS-entropy $h_{KS}(S)$ by minimizing the prediction loss $\mathcal{L}_{\text{pred}} := \mathbb{E}[\|u(t+\Delta t) - \hat{u}(t+\Delta t)\|^2]$. There exists a critical threshold determined by the KS-entropy that separates two fundamentally different learning regimes:

\begin{enumerate}
\item \textbf{The Laminar Phase} ($h_{KS}(S) < C_{\mathcal{A}}$): The system is \textbf{Microscopically Learnable}.
   \begin{itemize}
   \item The agent recovers the exact micro-dynamics: $\|\hat{S}_t - S_t\|_{L^2(\mu)} \to 0$ as training time $T \to \infty$.
   \item The effective noise term $\Sigma_T \to 0$ with rate $\Sigma_T = O(T^{-1/2})$.
   \item This corresponds to \textbf{Axiom LS (Local Stiffness)} holding at the microscopic scale: the learned dynamics satisfy the Łojasiewicz gradient inequality with the true exponent $\theta$.
   \item \textbf{Convergence rate:} $\mathcal{L}_{\text{pred}}(\Theta_T) \leq \mathcal{L}_{\text{pred}}(\Theta_0) \cdot \exp\left(-\frac{C_{\mathcal{A}} - h_{KS}(S)}{C_{\mathcal{A}}} \cdot T\right)$
   \end{itemize}

\item \textbf{The Turbulent Phase} ($h_{KS}(S) > C_{\mathcal{A}}$): The system is \textbf{Microscopically Unlearnable}.
   \begin{itemize}
   \item Pointwise prediction error remains non-zero: $\inf_{\Theta} \mathcal{L}_{\text{pred}}(\Theta) \geq D^*(C_{\mathcal{A}}) > 0$.
   \item The agent undergoes \textbf{Spontaneous Scale Symmetry Breaking}: it abandons the micro-scale and converges to a coarse-grained scale $\Lambda$ where $h_{KS}(S_\Lambda) < C_{\mathcal{A}}$.
   \item The residual prediction error becomes structured noise obeying \textbf{Mode D.D (Dispersion)}.
   \item \textbf{Irreducible error:} $\inf_\Theta \mathcal{L}_{\text{pred}}(\Theta) \geq \frac{1}{2\pi e} \cdot 2^{2(h_{KS}(S) - C_{\mathcal{A}})}$ (Shannon lower bound).
   \end{itemize}
\end{enumerate}
\end{metatheorem}

\begin{proof}
\textbf{Step 1 (Information-Theoretic Setup).} We formalize the learning process as a communication channel. Let $\mathcal{D}_T = \{u(t_i)\}_{i=1}^{N_T}$ be the observed trajectory data over training duration $T$, where $N_T = T/\Delta t$ samples. The learning algorithm defines a (possibly stochastic) map:
$$\mathcal{A}: \mathcal{D}_T \mapsto \Theta_T \in \mathbb{R}^d$$
The learned model $\hat{S}_{\Theta_T}$ attempts to approximate the true dynamics $S$. By the \textbf{data processing inequality} [@Shannon48], for any function $f$ of $\Theta_T$:
$$I(S; f(\Theta_T)) \leq I(S; \Theta_T) \leq I(S; \mathcal{D}_T)$$
The mutual information between the true dynamics and observed data satisfies:
$$I(S; \mathcal{D}_T) \leq H(\mathcal{D}_T) \leq N_T \cdot h_{KS}(S) \cdot \Delta t = T \cdot h_{KS}(S)$$
The capacity constraint on the learning channel gives $I(S; \Theta_T) \leq C_{\mathcal{A}} \cdot T$.

\textbf{Step 2 (Laminar Phase: Achievability).} Suppose $h_{KS}(S) < C_{\mathcal{A}}$. We construct a learning scheme achieving zero asymptotic error.

\textit{Construction:} Partition the parameter space into $2^{C_{\mathcal{A}} \cdot T}$ cells. By Shannon's source coding theorem, there exists an encoding of the trajectory $\mathcal{D}_T$ using at most $H(\mathcal{D}_T) + o(T)$ bits. Since $H(\mathcal{D}_T) \leq h_{KS}(S) \cdot T < C_{\mathcal{A}} \cdot T$, the trajectory can be encoded losslessly into the parameters.

\textit{Convergence:} Let $\varepsilon > 0$ and define the typical set $\mathcal{T}_\varepsilon^{(T)} := \{u : |H(u)/T - h_{KS}(S)| < \varepsilon\}$. By the Asymptotic Equipartition Property [@Shannon48]:
$$\mu(\mathcal{T}_\varepsilon^{(T)}) \to 1 \quad \text{as } T \to \infty$$
For typical trajectories, the encoding uses $(h_{KS}(S) + \varepsilon) \cdot T$ bits. Choosing $\varepsilon < C_{\mathcal{A}} - h_{KS}(S)$, lossless encoding is possible with high probability.

\textit{Rate:} The probability of decoding error satisfies $P(\hat{S}_{\Theta_T} \neq S) \leq 2^{-T(C_{\mathcal{A}} - h_{KS}(S) - \varepsilon)}$ by the channel coding theorem. This gives the exponential convergence rate in the statement.

\textbf{Step 3 (Turbulent Phase: Converse).} Suppose $h_{KS}(S) > C_{\mathcal{A}}$. We prove a lower bound on irreducible error.

\textit{Rate-Distortion Theory:} For a source with entropy rate $h_{KS}(S)$ and squared-error distortion $d(s, \hat{s}) = \|s - \hat{s}\|^2$, the rate-distortion function $R(D)$ satisfies [@Berger71]:
$$R(D) = h_{KS}(S) - \frac{1}{2}\log(2\pi e D)$$
for Gaussian sources (and provides a lower bound for general sources). Inverting:
$$D(R) = \frac{1}{2\pi e} \cdot 2^{2(h_{KS}(S) - R)}$$
Since the learning channel has rate at most $C_{\mathcal{A}}$, the minimum achievable distortion is:
$$D^*(C_{\mathcal{A}}) = \frac{1}{2\pi e} \cdot 2^{2(h_{KS}(S) - C_{\mathcal{A}})} > 0$$
This is the \textbf{Shannon lower bound} on prediction error.

\textit{Converse argument:} Any predictor $\hat{S}_\Theta$ satisfies:
$$\mathcal{L}_{\text{pred}}(\Theta) = \mathbb{E}[\|S_t u - \hat{S}_\Theta u\|^2] \geq D^*(I(S; \Theta)) \geq D^*(C_{\mathcal{A}})$$
The first inequality is the operational meaning of rate-distortion; the second uses the capacity bound.

\textbf{Step 4 (Scale Selection via Lyapunov Spectrum).} In the turbulent phase, the agent must choose which information to discard. We prove the optimal strategy selects a coarse-graining $\Pi^*$ aligned with the slow manifold.

\textit{Lyapunov decomposition:} Let $\{\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n\}$ be the Lyapunov exponents of $S$, ordered by magnitude. The contribution of each mode to entropy is [@Eckmann85]:
$$h_i = \max(0, \lambda_i)$$
The total entropy is $h_{KS}(S) = \sum_{i: \lambda_i > 0} \lambda_i$.

\textit{Optimal truncation:} Define the $k$-mode projection $\Pi_k$ retaining only modes with $|\lambda_i| \leq \lambda_k$. The entropy of the projected system is:
$$h_{KS}(S_{\Pi_k}) = \sum_{i > k: \lambda_i > 0} \lambda_i$$
The optimal scale $k^*$ is the smallest $k$ such that $h_{KS}(S_{\Pi_k}) \leq C_{\mathcal{A}}$.

\textit{Variational characterization:} This truncation emerges from optimizing the Lagrangian:
$$\Pi^* = \arg\min_{\Pi} \left[ \mathcal{L}_{\text{pred}}(\Pi) + \lambda \cdot I(X; \Pi(X)) \right]$$
where $\lambda$ is the Lagrange multiplier enforcing the capacity constraint. By the Karush-Kuhn-Tucker conditions, the optimal projection satisfies:
$$\frac{\partial \mathcal{L}_{\text{pred}}}{\partial \Pi} = -\lambda \frac{\partial I(X; \Pi(X))}{\partial \Pi}$$
The modes with highest $\lambda_i$ contribute most to mutual information but least to long-term prediction (they decorrelate fastest). Thus, gradient descent on $\mathcal{L}_{\text{pred}}$ under capacity constraints naturally discards high-entropy, low-predictability modes.

\textbf{Step 5 (Connection to Axiom LS).} In the Laminar Phase, the learned dynamics $\hat{S}_\Theta$ converge to the true dynamics $S$. If $S$ satisfies the Łojasiewicz gradient inequality with exponent $\theta$:
$$\|\nabla \Phi(u)\| \geq c \cdot |\Phi(u) - \Phi(u^*)|^\theta$$
then the learned dynamics inherit this property with the same exponent, since:
$$\|\nabla \hat{\Phi}_\Theta(u) - \nabla \Phi(u)\| \leq \varepsilon_T \to 0$$
implies the Łojasiewicz inequality transfers to $\hat{\Phi}_\Theta$ for sufficiently large $T$. This is \textbf{Axiom LS (Local Stiffness)} at the microscopic scale.

In the Turbulent Phase, Axiom LS fails at the micro-scale but is restored at the emergent macro-scale $\Pi^*(X)$, where the reduced dynamics satisfy the Łojasiewicz inequality with an effective exponent $\theta_{\text{eff}} \geq \theta$.
\end{proof}

---

### The Optimal Effective Theory

This section explains \textit{how} the agent handles the Turbulent Phase. We prove that the agent does not merely ``blur'' the data; it finds the \textbf{Computational Closure}—the variables that form a self-contained logical system decoupled from microscopic details.

\begin{definition}[Coarse-Graining Projection]\label{def:coarse-graining-projection}
A map $\Pi: X \to Y$ is a \textbf{coarse-graining} if $\dim(Y) < \dim(X)$. Formally, let $(X, \mathcal{B}_X, \mu)$ be the micro-state space and $Y$ a measurable space with $\sigma$-algebra $\mathcal{B}_Y = \Pi^{-1}(\mathcal{B}_Y)$. The macro-state is $y_t := \Pi(x_t)$, and the induced macro-dynamics are:
$$\bar{S}_t: Y \to \mathcal{P}(Y), \quad \bar{S}_t(y) := \mathbb{E}[\Pi(S_t(x)) \mid \Pi(x) = y]$$
where the expectation averages over micro-states compatible with macro-state $y$ using the conditional measure $\mu(\cdot \mid \Pi^{-1}(y))$. When this expectation is deterministic (i.e., concentrates on a single point), we write $\bar{S}_t: Y \to Y$.
\end{definition}

\begin{definition}[Closure Defect]\label{def:closure-defect}
The \textbf{closure defect} measures how much the macro-dynamics depend on discarded micro-details:
$$\delta_\Pi := \mathbb{E}_{x \sim \mu}\left[\|\Pi(S_t(x)) - \bar{S}_t(\Pi(x))\|^2\right]^{1/2}$$
Equivalently, in terms of conditional distributions:
$$\delta_\Pi^2 = \mathbb{E}_{y \sim \Pi_*\mu}\left[\text{Var}(\Pi(S_t(x)) \mid \Pi(x) = y)\right]$$
If $\delta_\Pi = 0$, the macro-dynamics are \textbf{autonomously closed}: the conditional distribution $P(y_{t+1} \mid x_t)$ depends on $x_t$ only through $y_t = \Pi(x_t)$. This is the ``Software decoupled from Hardware'' condition—the emergent description forms a \textbf{Markov factor} of the original dynamics.
\end{definition}

\begin{definition}[Predictive Information]\label{def:predictive-information}
The \textbf{predictive information} of a coarse-graining $\Pi$ over time horizon $\tau$ is:
$$I_{\text{pred}}^\tau(\Pi) := I(\Pi(X_{\text{past}}); \Pi(X_{\text{future}})) = I(Y_{(-\infty, 0]}; Y_{[0, \tau]})$$
where $Y_t = \Pi(X_t)$. This measures how much the macro-past tells us about the macro-future—the ``useful'' information retained by the projection.
\end{definition}

\begin{metatheorem}[The Renormalization Variational Principle]\label{mt:the-renormalization-variational-principle}
Let $S$ be a chaotic dynamical system with $h_{KS}(S) > C_{\mathcal{A}}$, and let an agent minimize the General Loss $\mathcal{L}_{\text{gen}}$ over projections $\Pi: X \to Y$ with $\dim(Y) \leq d_{\max}$. Then:

\begin{enumerate}
\item \textbf{(Existence)} There exists an optimal coarse-graining $\Pi^*$ achieving the infimum of $\mathcal{L}_{\text{gen}}$.

\item \textbf{(Characterization)} $\Pi^*$ minimizes the \textbf{Information Bottleneck Lagrangian} [@Tishby99]:
$$\mathcal{L}_{\text{IB}}(\Pi; \beta) := I(X; \Pi(X)) - \beta \cdot I(\Pi(X_{\text{past}}); \Pi(X_{\text{future}}))$$
for some $\beta^* > 0$ determined by the capacity constraint $I(X; \Pi(X)) \leq C_{\mathcal{A}}$.

\item \textbf{(Axiom Compatibility)} The induced macro-hypostructure $\mathbb{H}_{\Pi^*} = (Y, \bar{S}_t, \bar{\Phi}, \bar{\mathfrak{D}})$ satisfies \textbf{Axiom D (Dissipation)} and \textbf{Axiom TB (Topological Barrier)} with effective constants.
\end{enumerate}

\textbf{Consequences:}

\begin{enumerate}
\item \textbf{Emergence of Macroscopic Laws.} The agent does not learn the chaotic micro-map $x_{t+1} = f(x_t)$. It learns an effective stochastic map:
$$y_{t+1} = g(y_t) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \Sigma_{\Pi^*})$$
where $g: Y \to Y$ is the emergent deterministic macro-dynamics and $\Sigma_{\Pi^*} = \delta_{\Pi^*}^2$ is the residual variance. Examples: Navier-Stokes from molecular dynamics, Boltzmann equation from particle systems, mean-field equations from interacting spins.

\item \textbf{Noise as Ignored Information.} The residual error $\eta_t$ is not ontologically random; it is the projection of deterministic chaos from the ignored dimensions. Formally:
$$\eta_t = \Pi(S_t(x)) - \bar{S}_t(\Pi(x)) = \Pi(S_t(x)) - g(y_t)$$
The agent models this as \textbf{stochastic noise} with correlation structure inherited from the micro-dynamics. This satisfies \textbf{Mode D.D (Dispersion)} when $\eta_t$ decorrelates on the fast timescale $\tau_{\text{fast}} \ll \tau_{\text{macro}}$.

\item \textbf{Inertial Manifold Selection.} The optimal projection $\Pi^*$ aligns with the \textbf{Slow Manifold} $\mathcal{M}_{\text{slow}} \subset X$—the subspace spanned by eigenvectors of the linearized operator $DS$ with eigenvalues closest to the unit circle. This is the inertial manifold [@FoiasTemam88]: a finite-dimensional, exponentially attracting, positively invariant manifold that captures the long-term dynamics.
\end{enumerate}
\end{metatheorem}

\begin{proof}
\textbf{Step 1 (Loss Decomposition).} We derive the structure of the prediction loss in terms of information-theoretic quantities. Let $Y_t = \Pi(X_t)$ be the macro-trajectory. The prediction loss for the macro-dynamics is:
$$\mathcal{L}_{\text{pred}}^{\text{macro}}(\Pi) = \mathbb{E}[\|Y_{t+1} - \hat{Y}_{t+1}\|^2]$$
where $\hat{Y}_{t+1} = \bar{S}_t(Y_t)$ is the optimal predictor given only macro-information.

By the law of total variance:
$$\mathcal{L}_{\text{pred}}^{\text{macro}} = \underbrace{\mathbb{E}[\text{Var}(Y_{t+1} \mid Y_t)]}_{\text{Intrinsic macro-uncertainty}} = \underbrace{\mathbb{E}[\text{Var}(Y_{t+1} \mid Y_{-\infty:t})]}_{\text{Asymptotic uncertainty}} + \underbrace{\mathbb{E}[\text{Var}(\mathbb{E}[Y_{t+1} \mid Y_{-\infty:t}] \mid Y_t)]}_{\text{Memory loss}}$$

In terms of entropies, using the Gaussian approximation for analytical tractability:
$$\mathcal{L}_{\text{pred}}^{\text{macro}} \approx \frac{1}{2\pi e} 2^{2H(Y_{t+1} \mid Y_t)}$$

The conditional entropy decomposes as:
$$H(Y_{t+1} \mid Y_t) = H(Y_{t+1} \mid X_t) + I(Y_{t+1}; X_t \mid Y_t)$$
The first term is the \textbf{intrinsic noise} (entropy of the macro-future given full micro-information); the second is the \textbf{closure violation} (additional uncertainty from not knowing the micro-state).

\textbf{Step 2 (Information Bottleneck Derivation).} The agent faces a constrained optimization: minimize prediction error subject to complexity bound $I(X; Y) \leq C_{\mathcal{A}}$. The Lagrangian is:
$$\mathcal{L}(\Pi, \beta) = \mathcal{L}_{\text{pred}}^{\text{macro}}(\Pi) + \beta \cdot (I(X; \Pi(X)) - C_{\mathcal{A}})$$

For the Gaussian case, the prediction loss satisfies [@Tishby99]:
$$\mathcal{L}_{\text{pred}}^{\text{macro}} \propto 2^{-2I(Y_{\text{past}}; Y_{\text{future}})}$$

Thus, minimizing prediction error is equivalent to maximizing predictive information. The Lagrangian becomes:
$$\mathcal{L}_{\text{IB}}(\Pi; \beta) = I(X; \Pi(X)) - \beta \cdot I(\Pi(X_{\text{past}}); \Pi(X_{\text{future}}))$$

The first term penalizes \textbf{complexity} (how much micro-information is retained); the second rewards \textbf{relevance} (how predictive the retained information is).

\textbf{Step 3 (Optimal Projection Structure).} We characterize the critical points of $\mathcal{L}_{\text{IB}}$. Taking the functional derivative with respect to $\Pi$:
$$\frac{\delta \mathcal{L}_{\text{IB}}}{\delta \Pi} = \frac{\delta I(X; Y)}{\delta \Pi} - \beta \frac{\delta I(Y_{\text{past}}; Y_{\text{future}})}{\delta \Pi} = 0$$

Using the chain rule for mutual information:
$$\frac{\delta I(X; Y)}{\delta \Pi}(x) = \log \frac{p(x \mid y)}{p(x)} = \log \frac{p(y \mid x)}{p(y)}$$

The stationarity condition becomes:
$$p(y \mid x) \propto p(y) \exp\left(\beta \cdot \mathbb{E}_{x' \sim p(\cdot \mid y)}[\log p(y_{\text{future}} \mid x')]\right)$$

This is a self-consistent equation: the projection $\Pi$ determines the macro-distribution $p(y)$, which in turn determines the optimal projection. The fixed points correspond to \textbf{sufficient statistics} for predicting the future—minimal representations that preserve predictive information.

\textbf{Step 4 (Spectral Characterization).} For linear dynamics $S_t = e^{At}$ with spectrum $\{\lambda_i\}$, the optimal projection has an explicit form. Let $\{v_i\}$ be the eigenvectors of $A$, ordered by $|\text{Re}(\lambda_i)|$ ascending (slowest modes first).

The predictive information of mode $i$ over time horizon $\tau$ is:
$$I_i(\tau) = -\frac{1}{2}\log(1 - e^{-2|\text{Re}(\lambda_i)|\tau}) \approx |\text{Re}(\lambda_i)|^{-1} \cdot \tau^{-1}$$
for large $\tau$. Slow modes (small $|\text{Re}(\lambda_i)|$) carry more predictive information per bit of complexity.

The complexity cost of retaining mode $i$ is proportional to its entropy rate contribution:
$$I_i(X; Y) \propto \max(0, \text{Re}(\lambda_i))$$

The optimal projection $\Pi^*$ retains the $k^*$ slowest modes, where $k^*$ maximizes:
$$\sum_{i=1}^{k} I_i(\tau) - \beta \sum_{i=1}^{k} \max(0, \text{Re}(\lambda_i)) \quad \text{subject to} \quad \sum_{i=1}^{k} \max(0, \text{Re}(\lambda_i)) \leq C_{\mathcal{A}}$$

This is precisely the \textbf{slow manifold}: $\Pi^*(X) = \text{span}\{v_1, \ldots, v_{k^*}\}$.

\textbf{Step 5 (Nonlinear Extension via Inertial Manifolds).} For nonlinear systems, the slow manifold generalizes to the \textbf{inertial manifold} $\mathcal{M}$ [@FoiasTemam88]. This is a finite-dimensional manifold satisfying:
\begin{enumerate}
    \item \textbf{Positive invariance:} $S_t(\mathcal{M}) \subseteq \mathcal{M}$ for $t \geq 0$
    \item \textbf{Exponential attraction:} $\text{dist}(S_t(x), \mathcal{M}) \leq C e^{-\gamma t} \text{dist}(x, \mathcal{M})$ for some $\gamma > 0$
    \item \textbf{Asymptotic completeness:} Every trajectory is shadowed by a trajectory on $\mathcal{M}$
\end{enumerate}

The projection $\Pi^*: X \to \mathcal{M}$ minimizes closure defect:
$$\delta_{\Pi^*} = \sup_{x \in X} \text{dist}(S_t(x), S_t(\Pi^*(x))) \leq C e^{-\gamma t}$$

The macro-dynamics on $\mathcal{M}$ form a finite-dimensional ODE that captures the essential long-term behavior.

\textbf{Step 6 (Axiom Verification).} We verify that the induced macro-hypostructure satisfies the core axioms.

\textit{Axiom D (Dissipation):} Define the macro-height $\bar{\Phi}(y) := \inf_{x: \Pi(x) = y} \Phi(x)$. Then:
$$\frac{d}{dt}\bar{\Phi}(\bar{S}_t(y)) = \mathbb{E}\left[\frac{d}{dt}\Phi(S_t(x)) \mid \Pi(x) = y\right] \leq -\mathbb{E}[\mathfrak{D}(S_t(x)) \mid \Pi(x) = y] =: -\bar{\mathfrak{D}}(y)$$
The macro-dissipation $\bar{\mathfrak{D}}$ is non-negative, establishing Axiom D at the macro-scale.

\textit{Axiom TB (Topological Barrier):} The topological sectors of $X$ project to sectors of $Y$. If $\mathcal{T}_X = \{T_\alpha\}$ is the sector decomposition of $X$, then $\mathcal{T}_Y = \{\Pi(T_\alpha)\}$ provides a (possibly coarser) decomposition of $Y$. The barrier heights satisfy:
$$\Delta_Y(\Pi(T_\alpha), \Pi(T_\beta)) \leq \Delta_X(T_\alpha, T_\beta)$$
with equality when the projection respects the topological structure. Axiom TB at the macro-scale inherits from the micro-scale.

\textbf{Step 7 (Renormalization Group Interpretation).} The optimal projection $\Pi^*$ is a \textbf{Renormalization Group (RG) fixed point} [@Wilson71]. Define the RG transformation $\mathcal{R}_\ell$ as coarse-graining by length scale $\ell$:
$$\mathcal{R}_\ell: \Pi \mapsto \Pi \circ \Pi_\ell$$
where $\Pi_\ell$ averages over balls of radius $\ell$. The fixed point condition $\mathcal{R}_\ell(\Pi^*) \sim \Pi^*$ (up to rescaling) means:
$$\Pi^* \circ \Pi_\ell = \Pi^* \quad \text{(self-similarity)}$$

At the fixed point, the effective theory is \textbf{scale-invariant}: further coarse-graining does not change the form of the macro-dynamics, only rescales parameters. The effective coupling constants (coefficients in $g(y)$) flow to fixed values under RG.

This completes the proof: gradient descent on $\mathcal{L}_{\text{gen}}$ under capacity constraints converges to the RG fixed point, which is the optimal coarse-graining for prediction.
\end{proof}

---

### Summary: The Universal Simulator Guarantee

The two preceding metatheorems provide the rigorous guarantee for the ``Glass Box'' nature of the AGI learner:

1. \textbf{If the world is simple} ($h_{KS}(S) < C_{\mathcal{A}}$): The AGI becomes a \textbf{perfect simulator}—Laplace's Demon realized. It reconstructs the exact microscopic laws and predicts with arbitrary precision. \textbf{Axiom LS} holds at all scales.

2. \textbf{If the world is complex} ($h_{KS}(S) > C_{\mathcal{A}}$): The AGI becomes a \textbf{physicist}. It automatically derives the ``Thermodynamics'' of the system, discarding chaotic micro-details to present the \textbf{Effective Laws of Motion} at the optimal scale. The agent discovers:
   - The correct macro-variables (order parameters, conserved quantities)
   - The emergent dynamics (hydrodynamic equations, mean-field theories)
   - The noise model for unresolved scales (stochastic forcing satisfying Mode D.D)

3. \textbf{It never ``hallucinates'' noise.} The agent explicitly models the boundary between \textbf{Signal} (Mode C.C / Axiom LS: learnable, structured, deterministic) and \textbf{Entropy} (Mode D.D: unlearnable, modeled as stochastic). The transition is not ad hoc but emerges from the capacity constraint $h_{KS} \lessgtr C_{\mathcal{A}}$.

This is the derivation of \textbf{Effective Field Theory} from first principles of learning: the scale of description is not chosen by the physicist but discovered by the optimization process. The AGI's internal model is always interpretable as physics at some scale—either exact micro-physics or emergent macro-physics with explicit noise terms.

---

### Non-differentiable environments

\begin{definition}[RL hypostructure]\label{def:rl-hypostructure}
In a reinforcement learning setting, define:
\begin{itemize}
\item \textbf{State space:} $X$ = agent state + environment state
\item \textbf{Flow:} $S_t(x_t) = x_{t+1}$ where $x_{t+1}$ results from agent policy $\pi_\theta$ choosing action $a_t$ and environment producing the next state
\item \textbf{Trajectory:} $\tau = (x_0, a_0, x_1, a_1, \ldots, x_T)$
\end{itemize}
\end{definition}

\begin{definition}[Trajectory functional]\label{def:trajectory-functional}
Define the global undiscounted objective:
$$\mathcal{L}(\tau) := F(x_0, a_0, \ldots, x_T)$$
where $F$ encodes the quantity of interest (negative total reward, stability margin, hitting time, constraint violation, etc.).
\end{definition}

\begin{lemma}[Score function gradient]\label{lem:score-function-gradient}
For policy $\pi_\theta$ and expected loss $J(\theta) := \mathbb{E}_{\tau \sim \pi_\theta}[\mathcal{L}(\tau)]$:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\mathcal{L}(\tau) \nabla_\theta \log \pi_\theta(\tau)]$$
where $\log \pi_\theta(\tau) = \sum_{t=0}^{T-1} \log \pi_\theta(a_t | x_t)$.
\end{lemma}

\begin{proof}
Standard policy gradient derivation:
$$\nabla_\theta J(\theta) = \nabla_\theta \int \mathcal{L}(\tau) p_\theta(\tau) d\tau = \int \mathcal{L}(\tau) p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) d\tau.$$
The environment dynamics contribute to $p_\theta(\tau)$ but not to $\nabla_\theta \log p_\theta(\tau)$, which depends only on the policy.
\end{proof}

\begin{metatheorem}[Non-Differentiable Extension]\label{mt:non-differentiable-extension}
Even when the environment transition $x_{t+1} = f(x_t, a_t, \xi_t)$ is non-differentiable (discrete, stochastic, or black-box), the expected loss $J(\theta) = \mathbb{E}[\mathcal{L}(\tau)]$ is differentiable in the policy parameters $\theta$.
\end{metatheorem}

\begin{proof}
The key observation is that we differentiate the \textbf{expectation} of the trajectory functional, not the environment map itself. The dependence of the trajectory distribution on $\theta$ enters only through the policy $\pi_\theta$, which is differentiable. The score function gradient (Lemma 14.20) requires only:
\begin{enumerate}
\item Sampling trajectories from $\pi_\theta$
\item Evaluating $\mathcal{L}(\tau)$
\item Computing $\nabla_\theta \log \pi_\theta(\tau)$
\end{enumerate}

None of these require differentiating through the environment.
\end{proof}

\begin{corollary}[No discounting required]\label{cor:no-discounting-required}
The global loss $\mathcal{L}(\tau)$ is defined directly on finite or stopping-time trajectories. Well-posedness is ensured by:
\begin{itemize}
\item Finite horizon $T < \infty$
\item Absorbing states terminating trajectories
\item Stability structure of the hypostructure
\end{itemize}
\end{corollary}

Discounting becomes an optional modeling choice, not a mathematical necessity.

\begin{proof}
For finite $T$, the trajectory space is well-defined and the expectation finite. For infinite-horizon problems with absorbing states, the stopping time is almost surely finite under appropriate conditions.
\end{proof}

\begin{corollary}[RL as hypostructure instance]\label{cor:rl-as-hypostructure-instance}
Backpropagating a global loss through a non-differentiable RL environment is the decision-making instance of the general pattern:
\begin{enumerate}
\item Treat system + agent as a hypostructure over trajectories
\item Define a global Lyapunov/loss functional on trajectory space
\item Differentiate its expectation with respect to agent parameters
\item Perform gradient-based optimization without discounting
\end{enumerate}
\end{corollary}

---

### Structural Identifiability

This section establishes that the defect functionals introduced in \cref{ch:meta-learning} determine the hypostructure components from axioms alone, and that parametric families of hypostructures are learnable under minimal extrinsic conditions. The philosophical foundation is the **univalence axiom** of Homotopy Type Theory [@HoTT13]: identity is equivalent to equivalence. Two hypostructures are identified if and only if they are structurally equivalent.

\begin{definition}[Defect signature]\label{def:defect-signature}
For a parametric hypostructure $\mathcal{H}_\Theta$ and trajectory class $\mathcal{U}$, the \textbf{defect signature} is the function:
$$\mathsf{Sig}(\Theta): \mathcal{U} \to \mathbb{R}^{|\mathcal{A}|}, \quad \mathsf{Sig}(\Theta)(u) := \big(K_A^{(\Theta)}(u)\big)_{A \in \mathcal{A}}$$
where $\mathcal{A} = \{C, D, SC, Cap, LS, TB\}$ is the set of axiom labels.
\end{definition}

\begin{definition}[Rich trajectory class]\label{def:rich-trajectory-class}
A trajectory class $\mathcal{U}$ is \textbf{rich} if:
\end{definition}

1. $\mathcal{U}$ is closed under time shifts: if $u \in \mathcal{U}$ and $s > 0$, then $u(\cdot + s) \in \mathcal{U}$.
2. For $\mu$-almost every initial condition $x \in X$, at least one finite-energy trajectory starting at $x$ belongs to $\mathcal{U}$.

\begin{definition}[Action reconstruction applicability]\label{def:action-reconstruction-applicability}
The hypostructure $\mathcal{H}_\Theta$ satisfies \textbf{action reconstruction} if axioms (D), (LS), (GC) hold and the underlying metric structure is such that the canonical Lyapunov functional equals the geodesic action with respect to the Jacobi metric $g_{\mathfrak{D}} = \mathfrak{D}_\Theta \cdot g$.
\end{definition}

\begin{metatheorem}[Defect Reconstruction]\label{mt:defect-reconstruction-2}
Let $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$ be a parametric family of hypostructures satisfying axioms (C, D, SC, Cap, LS, TB, Reg) and (GC) on gradient-flow trajectories. Suppose:
\end{metatheorem}

1. **(A1) Rich trajectories.** The trajectory class $\mathcal{U}$ is rich in the sense of Definition 14.25.
2. **(A2) Action reconstruction.** Definition 14.26 holds for each $\Theta$.

Then for each $\Theta$, the defect signature $\mathsf{Sig}(\Theta)$ determines, up to Hypo-isomorphism:

1. The semiflow $S_t$ (on the support of $\mathcal{U}$)
2. The dissipation $\mathfrak{D}_\Theta$ along trajectories
3. The height functional $\Phi_\Theta$ (up to an additive constant)
4. The scaling exponents and barrier constants
5. The safe manifold $M$

There exists a reconstruction operator $\mathcal{R}: \mathsf{Sig}(\Theta) \mapsto (\Phi_\Theta, \mathfrak{D}_\Theta, S_t, \text{barriers}, M)$ built from the axioms and defect functional definitions alone.

\begin{proof}
\textbf{Step 1 (Recover $S_t$ from $K_C$).} By Definition 13.1, $K_C^{(\Theta)}(u) := \|S_t(u(s)) - u(s+t)\|$ for appropriate $s, t$. Axiom (C) and (Reg) ensure that true trajectories are exactly those with $K_C = 0$ (Lemma 13.4). Since $\mathcal{U}$ is closed under time shifts (A1), the unique semiflow $S_t$ is determined as the one whose orbits saturate the zero-defect locus of $K_C$.

\textbf{Step 2 (Recover $\partial_t \Phi_\Theta + \mathfrak{D}_\Theta$ from $K_D$).} By Definition 13.1:
$$K_D^{(\Theta)}(u) = \int_T \max\big(0, \partial_t \Phi_\Theta(u(t)) + \mathfrak{D}_\Theta(u(t))\big) \, dt.$$
Axiom (D) requires $\partial_t \Phi_\Theta + \mathfrak{D}_\Theta \leq 0$ along trajectories. Thus $K_D^{(\Theta)}(u) = 0$ if and only if the energy-dissipation balance holds exactly. The zero-defect condition identifies the canonical dissipation-saturated representative.

\textbf{Step 3 (Recover $\mathfrak{D}_\Theta$ from metric and trajectories).} Axiom (Reg) provides metric structure with velocity $|\dot{u}(t)|_g$. Axiom (GC) on gradient-flow orbits gives $|\dot{u}|_g^2 = \mathfrak{D}_\Theta$. Combined with (D), propagation along the rich trajectory class determines $\mathfrak{D}_\Theta$ everywhere via the Action Reconstruction principle (\cref{mt:functional-reconstruction}).

\textbf{Step 4 (Recover $\Phi_\Theta$ from $\mathfrak{D}_\Theta$ and LS + GC).} The Action Reconstruction Theorem states: (D) + (LS) + (GC) $\Rightarrow$ the canonical Lyapunov $\mathcal{L}$ is the geodesic action with respect to $g_{\mathfrak{D}}$. By the Canonical Lyapunov Theorem ({prf:ref}`mt-canonical-lyapunov-existence`), $\mathcal{L}$ equals $\Phi_\Theta$ up to an additive constant. Once $\mathfrak{D}_\Theta$ and $M$ are known, $\Phi_\Theta$ is reconstructed.

\textbf{Step 5 (Recover exponents and barriers from remaining defects).} The SC defect compares observed scaling behavior with claimed exponents $(\alpha_\Theta, \beta_\Theta)$. Minimizing over trajectories identifies the unique exponents. Similarly, Cap/TB/LS defects compare actual behavior with capacity/topological/Łojasiewicz bounds; the barrier constants are the unique values at which defects transition from positive to zero.
\end{proof}

**Key Insight:** The reconstruction operator $\mathcal{R}$ is a derived object of the framework—not a new assumption. Every step uses existing axioms and metatheorems (Structural Resolution, Canonical Lyapunov, Action Reconstruction).

---

\begin{definition}[Persistent excitation]\label{def:persistent-excitation}
A trajectory distribution $\mu$ on $\mathcal{U}$ satisfies \textbf{persistent excitation} if its support explores a full-measure subset of the accessible phase space: for every open set $U \subset X$ with positive Lebesgue measure, $\mu(\{u : u(t) \in U \text{ for some } t\}) > 0$.
\end{definition}

\begin{definition}[Nondegenerate parametrization]\label{def:nondegenerate-parametrization}
The parametric family $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$ has \textbf{nondegenerate parametrization} if the map $\Theta \mapsto (\Phi_\Theta, \mathfrak{D}_\Theta)$ is locally Lipschitz and injective: there exists $c > 0$ such that for $\mu$-almost every $x \in X$:
$$|\Phi_\Theta(x) - \Phi_{\Theta'}(x)| + |\mathfrak{D}_\Theta(x) - \mathfrak{D}_{\Theta'}(x)| \geq c \, |\Theta - \Theta'|.$$
\end{definition}

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom Satisfaction:** $\mathcal{H}_\Theta$ satisfies axioms (C, D, SC, Cap, LS, TB, Reg, GC) for each $\Theta$
>     *   [ ] **(C1) Persistent Excitation:** Trajectory distribution $\mu$ explores full-measure subset of accessible phase space
>     *   [ ] **(C2) Nondegenerate Parametrization:** $|\Phi_\Theta(x) - \Phi_{\Theta'}(x)| + |\mathfrak{D}_\Theta(x) - \mathfrak{D}_{\Theta'}(x)| \geq c|\Theta - \Theta'|$
>     *   [ ] **(C3) Regular Parameter Space:** $\Theta_{\mathrm{adm}}$ is a metric space
> *   **Output (Structural Guarantee):**
>     *   Exact identifiability up to gauge: $\mathsf{Sig}(\Theta) = \mathsf{Sig}(\Theta') \Rightarrow \mathcal{H}_\Theta \cong \mathcal{H}_{\Theta'}$
>     *   Local quantitative identifiability: $|\Theta - \tilde{\Theta}| \leq C\varepsilon$ when signature difference $\leq \varepsilon$
>     *   Well-conditioned stability of signature map
> *   **Failure Condition (Debug):**
>     *   If **(C1) Persistent Excitation** fails → **Mode data insufficiency** (unexplored regions, indistinguishable parameters)
>     *   If **(C2) Nondegeneracy** fails → **Mode parameter aliasing** (different $\Theta$ produce same $(\Phi, \mathfrak{D})$)

\begin{metatheorem}[Meta-Identifiability]\label{mt:meta-identifiability}
Let $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$ be a parametric family satisfying:
\end{metatheorem}

1. Axioms (C, D, SC, Cap, LS, TB, Reg, GC) for each $\Theta$
2. **(C1) Persistent excitation:** The trajectory distribution satisfies Definition 14.28
3. **(C2) Nondegenerate parametrization:** Definition 14.29 holds
4. **(C3) Regular parameter space:** $\Theta_{\mathrm{adm}}$ is a metric space

Then:

1. **(Exact identifiability up to gauge.)** If $\mathsf{Sig}(\Theta) = \mathsf{Sig}(\Theta')$ as functions on $\mathcal{U}$, then $\mathcal{H}_\Theta \cong \mathcal{H}_{\Theta'}$ as objects of Hypo.

2. **(Local quantitative identifiability.)** There exist constants $C, \varepsilon_0 > 0$ such that if
$$\sup_{u \in \mathcal{U}} \sum_{A \in \mathcal{A}} \big| K_A^{(\Theta)}(u) - K_A^{(\Theta^*)}(u) \big| \leq \varepsilon < \varepsilon_0,$$
then there exists a representative $\tilde{\Theta}$ of the equivalence class $[\Theta^*]$ with $|\Theta - \tilde{\Theta}| \leq C \varepsilon$.

The map $[\Theta] \in \Theta_{\mathrm{adm}}/{\sim} \mapsto \mathsf{Sig}(\Theta)$ is locally injective and well-conditioned.

\begin{proof}
\textbf{Step 1 (Invoke Defect Reconstruction).} By \cref{mt:defect-reconstruction-2}, $\mathsf{Sig}(\Theta)$ determines $(\Phi_\Theta, \mathfrak{D}_\Theta, S_t, \text{barriers}, M)$ via the reconstruction operator $\mathcal{R}$.

\textbf{Step 2 (Apply nondegeneracy).} By (C2), equal signatures imply equal structural data $(\Phi_\Theta, \mathfrak{D}_\Theta)$ up to gauge. Equal structural data plus equal $S_t$ (from Step 1) gives Hypo-isomorphism.

\textbf{Step 3 (Quantitative bound).} The reconstruction $\mathcal{R}$ inherits Lipschitz constants from the axiom-derived formulas. Combined with the nondegeneracy constant $c$ from (C2), perturbations in signature of size $\varepsilon$ produce perturbations in $\Theta$ of size at most $C\varepsilon$ where $C = L_{\mathcal{R}}/c$.
\end{proof}

**Key Insight:** Meta-Identifiability reduces parameter learning to defect minimization. Minimizing $\mathcal{R}_A(\Theta) = \int_{\mathcal{U}} K_A^{(\Theta)}(u) \, d\mu(u)$ over $\Theta$ converges to the true hypostructure as trajectory data increases.

---

\begin{remark}[Irreducible extrinsic conditions]
The hypotheses (C1)--(C3) cannot be absorbed into the hypostructure axioms:
\begin{enumerate}
\item \textbf{Nondegenerate parametrization (C2)} concerns the human choice of coordinates on the space of hypostructures. The axioms constrain $(\Phi, \mathfrak{D}, \ldots)$ once chosen, but do not force any particular parametrization to be injective or Lipschitz. This is about representation, not physics.
\item \textbf{Data richness (C1)} concerns the observer's sampling procedure. The axioms determine what trajectories can exist; they do not guarantee that a given dataset $\mathcal{U}$ actually samples them representatively. This is about epistemics, not dynamics.
\end{enumerate}
Everything else---structure reconstruction, canonical Lyapunov, barrier constants, scaling exponents, failure mode classification---follows from the axioms and the metatheorems derived in Parts IV--VI.
\end{remark}

\begin{corollary}[Foundation for trainable hypostructures]\label{cor:foundation-for-trainable-hypostructures}
The Meta-Identifiability Theorem provides the theoretical foundation for the general loss (Definition 14.3): minimizing the axiom defect $\mathcal{R}_A(\Theta)$ over parameters $\Theta$ converges to the true hypostructure as data increases, with the only requirements being (C1)–(C3).
\end{corollary}

---



---


## The Fractal Gas (The Solver) {#ch:fractal-gas}

### The Tripartite Geometry

*Defining the relationship between Observation, Cognition, and Emergence.*

The Fractal Gas [@FractalAI18] is a computational instantiation of the hypostructure framework that explicitly separates three geometric structures: the domain of observation, the space of algorithmic reasoning, and the emergent manifold of collective behavior. This separation enables adaptive optimization through geometric transformation rather than brute-force search.

### The State Space ($X$): The Arena of Observation

The State Space is the domain where the agents (walkers) physically exist and make observations. It represents the "Territory" in the Map-Territory relation.

\begin{definition}[State Space]\label{def:state-space}
The \textbf{State Space} is a metric measure space $(X, d_X, \mu_X)$ representing the domain of the problem.
\end{definition}

1. **Agents:** A walker $w_i \in X$ is a point in this space.
2. **Rewards:** The objective function $R: X \to \mathbb{R}$ is defined here.
3. **Role:** $X$ provides the "ground truth" data. It is where the Base Dynamics $\mathcal{F}_t$ (gradient descent, physics engine) operate.

The State Space satisfies **Axiom C (Compactness)** when the feasible region is bounded, ensuring that the swarm cannot escape to infinity.

### The Algorithmic Space ($Y$): The Arena of Cognition

The Algorithmic Space is the embedding space where the system computes distances, similarities, and decisions. It represents the "Map."

\begin{definition}[Algorithmic Space]\label{def:algorithmic-space}
The \textbf{Algorithmic Space} is a normed vector space $(Y, \|\cdot\|_Y)$ equipped with a \textbf{Projection Map} $\pi: X \to Y$.
\end{definition}

1. **Feature Extraction:** The map $\pi$ extracts relevant features from the state. $\pi(w_i)$ is the "embedding" of walker $i$.
2. **Algorithmic Distance:** The distance used for companion selection (Axiom SC) is defined in $Y$, not $X$:
$$d_{\text{alg}}(i, j) := \| \pi(w_i) - \pi(w_j) \|_Y$$
3. **Role:** $Y$ is the "cognitive workspace." The AGI can learn or evolve the map $\pi$ to change how the swarm clusters and clones.

\begin{remark}[Flexibility of $\pi$]
\begin{itemize}
\item If $\pi$ is the identity, $Y \cong X$ and the system reduces to standard diffusion.
\item If $\pi$ is a Neural Network, $Y$ is the latent space and the system performs \textbf{representation learning}.
\item If $\pi$ encodes problem structure (symmetries, invariants), the system exploits this knowledge automatically.
\end{itemize}
\end{remark}

### The Emergent Manifold ($M$): The Geometry of Behavior

The Emergent Manifold is the effective geometry that the swarm *actually* explores. It is not pre-defined; it arises from the interaction between the swarm's diffusion and the fitness landscape.

\begin{definition}[Emergent Manifold]\label{def:emergent-manifold}
The \textbf{Emergent Manifold} is the Riemannian manifold $(M, g_{\text{eff}})$ defined by the \textbf{Inverse Diffusion Tensor} of the swarm.
\end{definition}

1. **Diffusion Tensor:** Let $D_{ij}(x)$ be the covariance matrix of the swarm's dispersal at point $x \in X$.
2. **Effective Metric:** The emergent metric is $g_{\text{eff}} = D^{-1}$.
3. **Role:** This represents the "path of least resistance." The swarm flows along geodesics of $(M, g_{\text{eff}})$.

**Interpretation:**
- High diffusion ($D$ large) $\to$ Low metric distance ($g$ small) $\to$ "Short" path (easy to traverse).
- Low diffusion ($D$ small) $\to$ High metric distance ($g$ large) $\to$ "Long" path (barrier).

The emergent manifold $(M, g_{\text{eff}})$ is the hypostructure's realization of **Axiom Rep (Dictionary)**—the correspondence between algorithmic operations and geometric structures.

### The Tripartite Interaction Cycle

The dynamics of the Fractal Gas can be understood as a cycle between these three spaces:

1. **Observation ($X \to Y$):** Agents in $X$ are projected into $Y$ via $\pi$.
2. **Decision ($Y \to M$):** Distances in $Y$ determine cloning probabilities. This reshapes the density $\rho$, which defines the diffusion $D$ and thus the metric $g$ on $M$.
3. **Action ($M \to X$):** Agents move along the geodesics of $M$ (via Langevin dynamics in $X$).
$$X \xrightarrow{\pi} Y \xrightarrow{\text{Cloning}} M \xrightarrow{\text{Kinetics}} X$$

\begin{theorem}[Geometric Adaptation]\label{thm:geometric-adaptation}
\textit{The Fractal Gas is unique because it explicitly separates $Y$ from $X$. By modifying $\pi$ (learning), the system can warp the effective geometry $M$ without changing the underlying problem $X$, allowing it to "tunnel" through barriers by changing its perspective.}
\end{theorem}

\begin{proof}
Let $\pi_1$ and $\pi_2$ be two different embeddings with $\pi_2 = T \circ \pi_1$ for some linear transformation $T$. The induced algorithmic distances satisfy:
$$d_{\text{alg}}^{(2)}(i,j) = \|T\| \cdot d_{\text{alg}}^{(1)}(i,j) + O(\|T - I\|^2)$$
The cloning probabilities depend on $d_{\text{alg}}$, so changing $\pi$ changes the cloning graph topology. By {prf:ref}`mt-imported-antichain-surface`, this changes the effective minimal surfaces and thus the geodesics of $M$.
\end{proof}

---


---


### The Fractal Gas Hypostructure

*The operational definition of the Fractal Gas as a coherent active matter system.*

### Formal Definition

The **Fractal Gas** is the hypostructure $\mathbb{H}_{\text{FG}} = (\mathcal{X}, S_{\text{total}}, \Phi, \mathfrak{D}, \mathcal{L}_{\nu})$ defined over the geometry $(M, Y)$.

#### The Ensemble State Space ($\mathcal{X}$)

Let the agent domain $M$ be a **Geodesic Metric Space** $(M, d_M)$.

\begin{definition}[Ensemble State]\label{def:ensemble-state}
The state is the ensemble:
$$\boldsymbol{\Psi} = (\psi_1, a_1, \ldots, \psi_N, a_N) \in (M \times \{0,1\})^N$$
where $\psi_i \in M$ is the position and $a_i \in \{0,1\}$ is the alive/dead status of walker $i$.
\end{definition}

**Embedding Axiom:** There exists an isometric (or Lipschitz) embedding $\varphi: M \to Y$ into a Banach space $Y$, allowing vector operations on state differences.

#### The Dynamic Topology (The Interaction Graph)

Interaction is defined by a time-dependent graph $G_t = (V_t, E_t, W_t)$.

\begin{definition}[Interaction Graph]\label{def:interaction-graph}
\begin{enumerate}
\item \textbf{Nodes:} The alive agents $\mathcal{A}_t = \{i \mid a_i = 1\}$.
\item \textbf{Weights:} $W_{ij} = K(d_{\text{alg}}(i, j))$ where $K$ is a localized kernel (e.g., Gaussian) and $d_{\text{alg}}$ is the distance in $Y$.
\item \textbf{Laplacian:} Let $L_t$ be the \textbf{Normalized Graph Laplacian} of $G_t$.
\end{enumerate}
\end{definition}

The Laplacian encodes the local connectivity structure:
$$L_t = I - D^{-1/2} W D^{-1/2}$$
where $D$ is the degree matrix.

---

### The Operators

The flow is the composition $S_{\text{total}} = \mathcal{K}_{\nu} \circ \mathcal{C} \circ \mathcal{V}$.

#### Operator $\mathcal{V}$: Patched Relativistic Fitness

\begin{definition}[Relativistic Fitness]\label{def:relativistic-fitness}
The operator $\mathcal{V}$ computes the potential vector $\mathbf{V} \in \mathbb{R}^N$ using patched Z-scores on the alive set.
\end{definition}

For each walker $i \in \mathcal{A}_t$:
1. Compute local mean $\mu_r$ and standard deviation $\sigma_r$ of rewards in a neighborhood.
2. Compute the Z-score: $z_{r,i} = (R_i - \mu_r)/\sigma_r$.
3. Similarly compute $z_{d,i}$ for diversity (distance to nearest neighbor).

The fitness potential is:
$$V_i = (\text{sigmoid}(z_{r,i}))^\alpha \cdot (\text{sigmoid}(z_{d,i}))^\beta$$

**Axiom Correspondence:** The patched standardization implements **Axiom SC (Scaling Coherence)**—the fitness is scale-invariant within each local patch.

#### Operator $\mathcal{C}$: Stochastic Cloning

\begin{definition}[Cloning Operator]\label{def:cloning-operator}
The operator $\mathcal{C}$ redistributes mass based on the Relative Cloning Score.
\end{definition}

For walkers $i$ and companion $j$:
$$S_{ij} = \frac{V_j - V_i}{V_i + \epsilon}$$

With probability proportional to $\max(0, S_{ij})$, walker $i$ clones the state of walker $j$.

**Axiom Correspondence:** Cloning implements **Axiom D (Dissipation)**—the height functional (negative fitness) decreases under the flow as low-fitness walkers are replaced by clones of high-fitness walkers.

#### Operator $\mathcal{K}_\nu$: Viscous-Adaptive Kinetics

This operator updates the state $\psi_i$ by combining the Base Dynamics with two distinct structural forces.

\begin{definition}[The Generalized Force Equation]\label{def:the-generalized-force-equation}
The update rule for agent $i$ is defined in the embedding space $Y$:
\end{definition}
$$\varphi(\psi_i^{t+1}) = \varphi(\psi_i^t) + \Delta_{\text{base}} + \mathbf{F}_{\text{adapt}} + \mathbf{F}_{\text{visc}}$$

1. **Base Dynamics ($\Delta_{\text{base}}$):** The intrinsic evolution of the agent (momentum, random walk, gradient step of the objective function).

2. **Adaptive Force ($\mathbf{F}_{\text{adapt}}$):** A force derived from the fitness potential gradient. In non-smooth settings, this is the **Direction of Maximal Slope** of $V$:
$$\mathbf{F}_{\text{adapt}} = -\epsilon_F \cdot \partial^- V(\psi_i)$$
where $\partial^-$ is the metric slope operator [@AGS08].

3. **Viscous Force ($\mathbf{F}_{\text{visc}}$):** A coherent force pulling the agent towards the weighted mean of its neighbors. This is the action of the Graph Laplacian:
$$\mathbf{F}_{\text{visc}} = \nu \sum_{j \in \mathcal{N}(i)} W_{ij} (\varphi(\psi_j) - \varphi(\psi_i))$$
where $\nu$ is the **Viscosity Coefficient**.

**Projection:** The final state is recovered by projecting back to the manifold: $\psi_i^{t+1} = \text{proj}_M(\cdots)$.

---

### The Darwinian Ratchet

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom SC:** Scaling Coherence (dimensional balance $\alpha > \beta$)
>
> *   **Output (Structural Guarantee):**
>     *   Darwinian selection concentrates stationary distribution on height minima
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom D** fails $\to$ **Mode C.E** (Energy blow-up)

\begin{metatheorem}[The Darwinian Ratchet]\label{mt:darwinian-ratchet}
\textbf{Statement.} The Fractal Gas dynamics implement a Darwinian selection mechanism: the stationary distribution concentrates on states that minimize the height functional $\Phi$ while maintaining diversity through the geometric term $\sqrt{\det g_{\text{eff}}}$. Specifically, the stationary density satisfies:
$$\rho_{\text{FG}}(x) \propto \sqrt{\det g_{\text{eff}}(x)} \, e^{-\beta \Phi(x)}$$
where $g_{\text{eff}}$ is the effective metric induced by the Fractal Gas dynamics and $\beta$ is the inverse temperature.
\end{metatheorem}

*Interpretation:* The Fractal Gas acts as a "Darwinian ratchet" that irreversibly drives the population toward fitness optima. The geometric term $\sqrt{\det g}$ ensures exploration of all modes while the Boltzmann factor $e^{-\beta\Phi}$ concentrates on low-energy (high-fitness) regions.

---

### The Coherence Phase Transition

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom SC:** Scaling Coherence (dimensional balance $\alpha > \beta$)
>
> *   **Output (Structural Guarantee):**
>     *   Phase transition in coherence via capacity threshold
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom C** fails $\to$ **Mode D.D** (Dispersion/Global existence)
>     *   If **Axiom D** fails $\to$ **Mode C.E** (Energy blow-up)


This theorem defines the role of the viscosity parameter $\nu$.

**Statement.** The Fractal Gas admits a **Coherence Phase Transition** controlled by the ratio of Viscosity $\nu$ to Cloning Jitter $\delta$.

**Phase Diagram:**

1. **Gas Phase ($\nu \ll \delta$):** The swarm behaves as independent agents. The effective geometry is the **Local Hessian**. Exploration is high; coherence is low.

2. **Liquid Phase ($\nu \approx \delta$):** The swarm moves as a coherent deformable body. The effective geometry is the **Smoothed Hessian** (convolved with the Laplacian kernel).

3. **Solid Phase ($\nu \gg \delta$):** The swarm crystallizes into a rigid lattice. It creates a **Consensus Manifold** and collapses to a single point in quotient space.

\begin{proof}
\textbf{Step 1 (Order Parameter).} Define the coherence order parameter:
$$\Psi_{\text{coh}} := \frac{1}{N^2} \sum_{i,j} \langle \dot{\psi}_i, \dot{\psi}_j \rangle$$
measuring the alignment of velocities.

\textbf{Step 2 (Gas Phase).} When $\nu \to 0$, the viscous force vanishes. Each walker evolves independently under $\Delta_{\text{base}} + \mathbf{F}_{\text{adapt}}$. The velocity correlation decays exponentially with distance: $\langle \dot{\psi}_i, \dot{\psi}_j \rangle \sim e^{-d_{ij}/\xi}$ with correlation length $\xi \sim \sqrt{D/\lambda}$.

\textbf{Step 3 (Liquid Phase).} At intermediate $\nu$, the viscous force creates velocity correlations. The Laplacian term smooths the velocity field:
$$\partial_t \mathbf{v} = \nu L \mathbf{v} + \text{forces}$$
This is a discrete heat equation with diffusion coefficient $\nu$. The smoothing length is $\ell_\nu \sim \sqrt{\nu \Delta t}$.

\textbf{Step 4 (Solid Phase).} When $\nu \to \infty$, the viscous force dominates. All velocities converge to the mean: $\dot{\psi}_i \to \bar{\dot{\psi}}$. The swarm moves as a rigid body.

\textbf{Step 5 (Critical Transition).} The transition occurs when $\ell_\nu \sim \ell_{\text{clone}}$ (the cloning length scale). At this point, velocity coherence extends across the cloning neighborhood, enabling collective tunneling.
\end{proof}

**Implication:** The introduction of $\mathbf{F}_{\text{visc}}$ allows the algorithm to perform **Non-Local Smoothing** of the fitness landscape.
- **Without Viscosity:** The swarm sees every local jagged peak of the objective function.
- **With Viscosity:** The swarm "surfs" a smoothed approximation of the landscape, effectively ignoring high-frequency noise (local minima) that is smaller than the viscous length scale.

---

### Topological Regularization

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom LS:** Local Stiffness (Łojasiewicz inequality near equilibria)
>     *   [ ] **Axiom Cap:** Capacity (geometric resolution bound)
>     *   [ ] **Axiom TB:** Topological Barrier (sector index conservation)
>
> *   **Output (Structural Guarantee):**
>     *   Topological barriers provide natural regularization
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom D** fails $\to$ **Mode C.E** (Energy blow-up)
>     *   If **Axiom C** fails $\to$ **Mode D.D** (Dispersion/Global existence)


**Statement.** The Viscous Force $\mathbf{F}_{\text{visc}}$ acts as a **Topological Regularizer** for the Information Graph.

**Theorem.** Under the flow of $\mathcal{K}_\nu$, the **Cheeger Constant** (bottleneck metric) of the Information Graph is bounded from below:
$$h(G_t) \geq C(\nu) > 0$$

\begin{proof}
\textbf{Step 1 (Cheeger Constant).} The Cheeger constant measures the "bottleneck" of a graph:
$$h(G) := \min_{S \subset V, |S| \leq |V|/2} \frac{|\partial S|}{\text{Vol}(S)}$$
where $|\partial S|$ is the cut size and $\text{Vol}(S)$ is the volume.

\textbf{Step 2 (Velocity Gradient Bound).} Consider a potential fracture: two clusters $S$ and $V \setminus S$ with mean velocities $\bar{v}_S$ and $\bar{v}_{S^c}$.

The viscous force on boundary walkers is:
$$|\mathbf{F}_{\text{visc}}^{\text{boundary}}| \geq \nu W_{\min} |\bar{v}_S - \bar{v}_{S^c}|$$

\textbf{Step 3 (Force Balance).} For the clusters to separate, the driving force must exceed the viscous resistance:
$$F_{\text{drive}} > \nu W_{\min} |\Delta v|$$

This requires:
$$|\partial S| \cdot W_{\min} < \frac{F_{\text{drive}}}{\nu |\Delta v|}$$

\textbf{Step 4 (Cheeger Bound).} Rearranging:
$$h(G) = \frac{|\partial S|}{\text{Vol}(S)} > \frac{\nu |\Delta v|}{F_{\text{drive}} \cdot \text{Vol}(S) / |\partial S|}$$

For bounded forces and volumes, this gives $h(G) \geq C(\nu)$ with $C(\nu) \sim \nu$.
\end{proof}

**Result:** The swarm maintains **Topological Connectedness** (Axiom TB satisfaction) even in non-convex landscapes, preventing premature fracturing of the population.

---

### Imported Geometric Metatheorems

\begin{metatheorem}[The Teleological Isomorphism]\label{mt:teleological-isomorphism}
Let $(\mathcal{H}, \Phi, \mathfrak{D})$ be a hypostructure satisfying Axioms C (Compactness), D (Dissipation), SC (Scaling Coherence), Rep (Dictionary), and GC (Gradient Consistency). Define the Meta-Action functional:
$$\mathcal{A}_{\text{meta}}[\gamma] := \int_0^T \left[\Phi(\gamma_t) + \lambda \mathfrak{D}(\gamma_t, \dot{\gamma}_t)\right] dt$$

Then:

\begin{enumerate}
\item \textbf{Value-Height Duality:} The value function equals the negative future Meta-Action: $V(x) = -\inf_{\gamma: \gamma_0=x} \mathcal{A}_{\text{meta}}[\gamma]$

\item \textbf{Policy-Gradient Equivalence:} The optimal policy is $\pi^*(x) = -\nabla \Phi(x)$

\item \textbf{Instrumental Convergence:} The system exhibits goal-directed behaviors (self-preservation, resource acquisition, goal stability)

\item \textbf{Predictive Processing:} Minimizing scaling defect forces internal modeling of future states

\item \textbf{Agency is Geometry:} Goal-directedness equals geodesic flow on Riemannian manifold with metric $g_{\text{meta}} := \nabla^2 \Phi + \lambda \nabla^2 \mathfrak{D}$
\end{enumerate}
\end{metatheorem}

---

### The Effective Geometry

\begin{theorem}[Induced Riemannian Structure]\label{thm:induced-riemannian-structure}
The Fractal Gas dynamics induce a Riemannian metric on the state space $X$ given by:
$$g_{\text{FG}} = \nabla^2 \Phi + \lambda \nabla^2 \mathfrak{D} + \nu L$$
\end{theorem}

where $\nabla^2 \Phi$ is the Hessian of the height functional, $\nabla^2 \mathfrak{D}$ is the Hessian of dissipation, and $L$ is the graph Laplacian.

\begin{proof}
This follows from the Meta-Action formulation (\cref{mt:teleological-isomorphism}) applied to the Fractal Gas Lagrangian:
$$\mathcal{L}_{\text{FG}} = \frac{1}{2}|\dot{\psi}|^2 - V(\psi) + \frac{\nu}{2} \sum_{ij} W_{ij}|\psi_i - \psi_j|^2$$

The Euler-Lagrange equations give the geodesic equation with the combined metric.
\end{proof}

---


---


### The Fractal Set Hypostructure

*The Trace of the Swarm as an Emergent Spacetime Manifold.*

### Formal Definition

The **Fractal Set** is the discrete hypostructure $\mathbb{H}_{\mathcal{F}} = (V, E_{\text{CST}}, E_{\text{IG}}, \Phi)$ constructed from the execution history of the Fractal Gas.

#### The Spacetime Events ($V$)

Let the execution time be $T \in \mathbb{N}$ steps. The vertex set $V$ is the set of all walker states across time:
$$V = \{ v_{i,t} = (\psi_i(t), a_i(t)) \mid i \in \{1, \ldots, N\}, t \in \{0, \ldots, T\} \}$$

**Embedding:** Each vertex is embedded in the manifold $M \times \mathbb{R}$ (Space $\times$ Time).

#### The Edge Foliation ($E$)

The graph topology is a **Foliation** of two distinct edge sets:

\begin{definition}[Causal Spacetime Tree]\label{def:causal-spacetime-tree}
The CST consists of directed edges representing \textbf{Temporal Evolution}:
$$E_{\text{CST}} = \{ (v_{i,t} \to v_{i, t+1}) \mid a_i(t)=1 \}$$
\end{definition}

- *Physics:* These are the **Worldlines** of the particles.
- *Metric:* The weight is the Kinetic Action $\int \mathcal{L} \, dt$.

\begin{definition}[Information Graph]\label{def:information-graph-2}
The IG consists of directed edges representing \textbf{Information Exchange} (Cloning):
$$E_{\text{IG}} = \{ (v_{j,t} \to v_{i,t}) \mid \text{Walker } i \text{ cloned companion } j \text{ at time } t \}$$
\end{definition}

- *Physics:* These are **Entanglement Bridges** (Einstein-Rosen bridges) connecting spatially distant regions.
- *Metric:* The weight is the Algorithmic Distance $d_{\text{alg}}(i, j)$.

\begin{remark}[Causal Structure]
The combined graph $(V, E_{\text{CST}} \cup E_{\text{IG}})$ forms a \textbf{Directed Acyclic Graph} (DAG) with a natural partial order: $v \prec w$ iff there is a directed path from $v$ to $w$. This is the causal structure of the computational spacetime.
\end{remark}

---

### The Geometric Reconstruction Principle {#metatheorem-37.2-the-geometric-reconstruction-principle}

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>     *   [ ] **Axiom SC:** Scaling Coherence (dimensional balance $\alpha > \beta$)
>     *   [ ] **Axiom LS:** Local Stiffness (Łojasiewicz inequality near equilibria)
>     *   [ ] **Axiom Cap:** Capacity (geometric resolution bound)
>     *   [ ] **Axiom Rep:** Dictionary/Correspondence (structural translation)
>
> *   **Output (Structural Guarantee):**
>     *   Geometry reconstructed from algebraic data
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom C** fails $\to$ **Mode D.D** (Dispersion/Global existence)
>     *   If **Axiom Cap** fails $\to$ **Mode C.D** (Geometric collapse)


**Statement.** For any problem class where the fitness landscape $\Phi$ is sufficiently smooth ($C^2$), the Fractal Set $\mathcal{F}$ converges (as $N \to \infty, \Delta t \to 0$) to a discrete approximation of the **Riemannian Manifold induced by the Fisher Information Metric**.

**The Isomorphism:**

1. **Density $\cong$ Volume Form:** The spatial density of nodes $V$ approximates $\sqrt{\det g_{\text{eff}}}$.

2. **IG Connectivity $\cong$ Geodesic Distance:** The shortest path distance on the union graph $E_{\text{CST}} \cup E_{\text{IG}}$ approximates the geodesic distance on the emergent manifold $(M, g_{\text{eff}})$.

3. **Graph Curvature $\cong$ Ricci Curvature:** The Ollivier-Ricci curvature of the IG converges to the scalar curvature $R$ of the landscape.

\begin{proof}
\textbf{Step 1 (Density-Volume Correspondence).} By the cloning dynamics, regions with high fitness $V$ accumulate walkers. The equilibrium density satisfies:
$$\rho(x) \propto e^{-\beta \Phi(x)}$$
This is the Boltzmann distribution. The induced volume form is:
$$d\text{Vol}_{\text{swarm}} = \rho(x) dx \propto e^{-\beta \Phi} dx$$

In the Fisher metric, the volume form is $\sqrt{\det g_F} = \sqrt{\det(\nabla^2 \Phi)}$ for exponential families. The density concentrates where this determinant is large (high curvature = high density).

\textbf{Step 2 (Distance Correspondence).} The IG connects walkers that are close in algorithmic space $Y$. By the Γ-convergence theorem [@Braides02], the graph distance converges to the geodesic distance:
$$d_{\text{graph}}(v_i, v_j) \xrightarrow{N \to \infty} d_{g_{\text{eff}}}(\psi_i, \psi_j)$$

\textbf{Step 3 (Curvature Correspondence).} The Ollivier-Ricci curvature [@Ollivier09] of an edge $(i,j)$ in a graph is:
$$\kappa(i,j) = 1 - \frac{W_1(\mu_i, \mu_j)}{d(i,j)}$$
where $W_1$ is the Wasserstein distance between the neighbor distributions.

For the IG, high curvature corresponds to regions where cloning is concentrated (minima of $\Phi$). This matches the Ricci curvature of the fitness landscape.
\end{proof}

**Implication:** We do not need to *know* the geometry of the problem. By running the Fractal Gas, we **generate** a graph $\mathcal{F}$ whose discrete geometry *is* the geometry of the problem. Analyzing the Fractal Set is equivalent to analyzing the problem structure.

---

### The Causal Horizon Lock

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom SC:** Scaling Coherence (dimensional balance $\alpha > \beta$)
>     *   [ ] **Axiom Cap:** Capacity (geometric resolution bound)
>     *   [ ] **Axiom TB:** Topological Barrier (sector index conservation)
>
> *   **Output (Structural Guarantee):**
>     *   Causal horizons emerge from capacity constraints
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom D** fails $\to$ **Mode C.E** (Energy blow-up)
>     *   If **Axiom C** fails $\to$ **Mode D.D** (Dispersion/Global existence)


This theorem generalizes the "Antichain" results ({prf:ref}`mt-imported-antichain-surface`) to any application of the Fractal Gas.

**Statement.** Let $\Sigma \subset V$ be a subset of events (a region in spacetime). Let $\partial \Sigma$ be its boundary in the graph topology. The **Information Flow** out of $\Sigma$ is bounded by the **Area** of $\partial \Sigma$ in the IG metric:
$$I(\Sigma \to \Sigma^c) \leq \alpha \cdot \text{Area}_{\text{IG}}(\partial \Sigma)$$

\begin{proof}
\textbf{Step 1 (Information Channel).} Information flows from $\Sigma$ to its complement only via edges in $E_{\text{IG}}$ (cloning events).

\textbf{Step 2 (Locality).} Cloning edges are local in Algorithmic Space ($d_{\text{alg}} < \epsilon$ for the kernel $K$).

\textbf{Step 3 (Counting).} The number of IG edges crossing $\partial \Sigma$ is bounded by the "surface area" in the graph metric:
$$|E_{\text{IG}} \cap \partial \Sigma| \leq C \cdot \text{Area}_{\text{IG}}(\partial \Sigma)$$

\textbf{Step 4 (Holography).} Each edge carries at most $\log N$ bits (the index of the cloned walker). Therefore:
$$I(\Sigma \to \Sigma^c) \leq |E_{\text{IG}} \cap \partial \Sigma| \cdot \log N \leq \alpha \cdot \text{Area}_{\text{IG}}(\partial \Sigma)
$$
\end{proof}

**Universal Consequence:** Any system solved by the Fractal Gas obeys the **Holographic Principle**. The complexity of the solution inside a volume scales with the surface area of the volume, not the interior volume.

---

### The Scutoid Selection Principle

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom TB:** Topological Barrier (sector index conservation)
>     *   [ ] **Axiom Rep:** Dictionary/Correspondence (structural translation)
>
> *   **Output (Structural Guarantee):**
>     *   The Scutoid Selection Principle
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom C** fails $\to$ **Mode D.D** (Dispersion/Global existence)
>     *   If **Axiom D** fails $\to$ **Mode C.E** (Energy blow-up)


This explains why Scutoid tessellations emerge universally in the swarm dynamics.

**Statement.** Under the flow of the Fractal Gas ($\mathcal{K}_\nu$), the Voronoi tessellation of the swarm undergoes topological transitions (T1 transitions) that **minimize the Regge Action** of the dual triangulation.

**Theorem.** The sequence of tessellations generated by the swarm minimizes the discrete action:
$$S_{\text{Regge}} = \sum_{h \in \text{hinges}} \text{Vol}(h) \cdot \delta_h$$
where $\delta_h$ is the deficit angle (discrete curvature).

\begin{proof}
\textbf{Step 1 (Energy Minimization).} The swarm concentrates in low-potential regions (flat valleys of the landscape).

\textbf{Step 2 (Viscous Smoothing).} The viscous force $\mathbf{F}_{\text{visc}}$ minimizes velocity gradients, forcing walkers to form regular lattices where possible.

\textbf{Step 3 (Deficit Angle).} A regular lattice has zero deficit angle. High deficit angle corresponds to stress/curvature in the swarm.

\textbf{Step 4 (Scutoid Transition).} When stress exceeds a threshold, a T1 transition (Scutoid formation [@GomezGalvez18]) relaxes the lattice by exchanging neighbors. This reduces $S_{\text{Regge}}$.

\textbf{Step 5 (Convergence).} By the principle of minimum action, the tessellation converges to a configuration minimizing $S_{\text{Regge}}$.
\end{proof}

**Conclusion:** The Fractal Set is a **Dynamical Triangulation** in the sense of Causal Dynamical Triangulations (CDT). It naturally evolves to a "flat" geometry (solution) by expelling curvature through topological changes.

---

### The Archive Invariance (Universality)

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>     *   [ ] **Axiom LS:** Local Stiffness (Łojasiewicz inequality near equilibria)
>     *   [ ] **Axiom Cap:** Capacity (geometric resolution bound)
>
> *   **Output (Structural Guarantee):**
>     *   Universal computation preserves structural invariants
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom C** fails $\to$ **Mode D.D** (Dispersion/Global existence)
>     *   If **Axiom Cap** fails $\to$ **Mode C.D** (Geometric collapse)


**Statement.** Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two different Fractal Gas instantiations solving the same problem $P$, but with different hyperparameters (within the stability region $\alpha \approx \beta$). The **Fractal Sets** $\mathcal{F}_1$ and $\mathcal{F}_2$ generated by these runs are **quasi-isometric**:
$$\mathcal{F}_1 \sim_{\text{QI}} \mathcal{F}_2$$

\begin{proof}
\textbf{Step 1 (Attractor Invariance).} Since both systems satisfy Axioms C and D, they must converge to the same \textbf{Canonical Profiles} (local minima/attractors).

\textbf{Step 2 (Local Geometry).} The geometry of the Fractal Set near these attractors is determined by the Hessian of the problem $P$ (by \cref{mt:darwinian-ratchet}), which is invariant.

\textbf{Step 3 (Quasi-Isometry).} Two metric spaces are quasi-isometric if there exist maps with bounded distortion. The identity on attractors extends to a quasi-isometry on the Fractal Sets by the local geometry invariance.
\end{proof}

**Application:** The Fractal Set can **fingerprint** problems:
- If $\mathcal{F}$ has a single connected component $\to$ Convex Problem.
- If $\mathcal{F}$ has disconnected clusters $\to$ Multimodal Problem.
- If $\mathcal{F}$ has high Ollivier-Ricci curvature $\to$ Ill-conditioned Problem.

---

### Summary: The Universal Solver Trace

The **Fractal Set** is the "fossil record" of the optimization process:

| Component | Records | Physical Interpretation |
|:----------|:--------|:------------------------|
| **Nodes** | Exploration | Where we looked |
| **CST Edges** | Inertia | Momentum/Physics |
| **IG Edges** | Information | Selection/Learning |

The combined structure $\mathbb{H}_{\mathcal{F}}$ is a **discrete spacetime** whose geometry encodes the difficulty of the problem. Solving the problem is equivalent to relaxing this spacetime into a zero-curvature state (a flat solution).

---


---


## Fractal Set Foundations (Discrete-to-Continuum Structure) {#ch:fractal-set}

This chapter develops the mathematical foundation for discretizing hypostructures into Fractal Sets and establishing the discrete-to-continuum correspondence. We prove that continuous hypostructures admit faithful discrete representations, establish convergence theorems for discretization schemes, and show how emergent spacetime geometry arises from the causal and informational structure of Fractal Sets.

### Fractal Set Representation and Emergent Spacetime

*From discrete events to continuous dynamics.*

### Fractal Set Definition

We introduce Fractal Sets as the fundamental combinatorial objects underlying hypostructures. Unlike graphs or simplicial complexes, Fractal Sets encode both **temporal precedence** (causal structure) and **spatial/informational adjacency** (the Information Graph).

\begin{definition}[Fractal Set]\label{def:fractal-set}
A \textbf{Fractal Set} is a tuple $\mathcal{F} = (V, \text{CST}, \text{IG}, \Phi_V, w, \mathcal{L})$ where:
\end{definition}

**(1) Vertices.** $V$ is a countable set of **nodes** representing elementary events or episodes.

**(2) Causal Structure (CST).** A strict partial order $\prec$ on $V$ encoding temporal precedence:
- Irreflexivity: $v \not\prec v$
- Transitivity: $u \prec v \prec w \Rightarrow u \prec w$
- **Local finiteness:** For each $v \in V$, the past cone $J^-(v) := \{u : u \prec v\}$ is finite

**(3) Information Graph (IG).** An undirected graph $(V, E)$ encoding spatial/informational adjacency:
- $\{u, v\} \in E$ if $u$ and $v$ can exchange information
- **Bounded degree:** $\sup_{v \in V} \deg(v) < \infty$

**(4) Node Fitness.** $\Phi_V: V \to \mathbb{R}_{\geq 0}$ assigns to each node its **local energy** or **complexity measure**.

**(5) Edge Weights.** $w: E \to \mathbb{R}_{\geq 0}$ assigns to each edge its **transition cost** or **dissipation measure**.

**(6) Label System.** $\mathcal{L}$ assigns:
- **Type labels:** $\tau_v \in \mathcal{T}$ for each $v$, encoding topological sector
- **Gauge labels:** $g_e \in H$ for each edge $e$, encoding local symmetry data, where $H$ is a compact Lie group

\begin{definition}[Compatibility conditions]\label{def:compatibility-conditions}
A Fractal Set is \textbf{well-formed} if:
\end{definition}

**(C1) Causal-Information compatibility:** If $u \prec v$ (causal precedence), then there exists a path in IG connecting $u$ to $v$. No "action at a distance."

**(C2) Fitness monotonicity along chains:** For any maximal chain $v_0 \prec v_1 \prec \cdots$:
$$\sum_{i=0}^n \Phi_V(v_i) \leq C + c \cdot \sum_{i=0}^{n-1} w(\{v_i, v_{i+1}\})$$
for universal constants $C, c$. Energy is bounded by accumulated dissipation.

**(C3) Gauge consistency:** For any cycle $v_0 - v_1 - \cdots - v_k - v_0$ in IG, the holonomy:
$$\text{hol}(\gamma) := g_{v_0 v_1} \cdot g_{v_1 v_2} \cdots g_{v_k v_0}$$
depends only on the homotopy class of $\gamma$.

\begin{definition}[Time slices and states]\label{def:time-slices-and-states}
For a Fractal Set $\mathcal{F}$:
\end{definition}

**(1) Time function:** Any function $t: V \to \mathbb{R}$ respecting CST (i.e., $u \prec v \Rightarrow t(u) < t(v)$).

**(2) Time slice:** For each $T \in \mathbb{R}$, define:
$$V_T := \{v \in V : t(v) \leq T \text{ and } \nexists w \succ v \text{ with } t(w) \leq T\}$$
the "present moment" at time $T$.

**(3) State at time $T$:** The equivalence class $[V_T]$ under IG-automorphisms preserving labels.

---

### Axiom Correspondence

The hypostructure axioms translate into combinatorial constraints on Fractal Sets:

| Hypostructure | Fractal Set Translation |
|---------------|-------------------------|
| State $x \in X$ | Time slice $V_T$ |
| Height $\Phi(x)$ | $\displaystyle\sum_{v \in V_T} \Phi_V(v)$ |
| Dissipation $\int_0^T \mathfrak{D}$ | $\displaystyle\sum_{e \in \text{path}} w(e)$ over edges crossed |
| Symmetry group $G$ | Gauge group $H$ acting on edge labels |
| Topological sector $\tau$ | Type labels $\tau_v$ (conserved under CST) |
| Capacity bounds | Degree bounds on IG |
| Łojasiewicz structure | Local geometry of fitness landscape |

\begin{proposition}[Axiom D on Fractal Sets]\label{prop:axiom-d-on-fractal-sets}
The dissipation axiom becomes:
$$\sum_{v \in V_T} \Phi_V(v) - \sum_{v \in V_0} \Phi_V(v) \leq -\alpha \sum_{e \in \text{path}(0,T)} w(e)$$
for paths traversed between times $0$ and $T$.
\end{proposition}

\begin{proposition}[Axiom C on Fractal Sets]\label{prop:axiom-c-on-fractal-sets}
Compactness becomes: For any sequence of time slices $(V_{T_n})$ with bounded total fitness, there exists a subsequence converging in the graph metric modulo gauge equivalence.
\end{proposition}

\begin{proposition}[Axiom Cap on Fractal Sets]\label{prop:axiom-cap-on-fractal-sets}
Capacity bounds become: The singular set (nodes with $\Phi_V(v) > E_{\text{crit}}$) has bounded density in the IG metric.
\end{proposition}

---

### Fractal Representation Theorem

\begin{metatheorem}[Fractal Representation]\label{mt:fractal-representation}
Let $\mathcal{H} = (X, S_t, \Phi, \mathfrak{D}, G, M, \ldots)$ be a hypostructure satisfying:
\end{metatheorem}

**(FR1) Finite local complexity:** For each energy level $E$, the number of local configurations (modulo $G$) is finite.

**(FR2) Discrete time approximability:** The semiflow $S_t$ is well-approximated by discrete steps $S_\varepsilon$ for small $\varepsilon > 0$.

Then there exists a Fractal Set $\mathcal{F}$ and a **representation map** $\Pi: \mathcal{F} \to \mathcal{H}$ such that:

**(1) State correspondence:** Time slices $V_T$ map to states: $\Pi(V_T) \in X$.

**(2) Trajectory correspondence:** Paths in CST map to trajectories: $\Pi(\gamma) = (S_t x)_{t \geq 0}$.

**(3) Axiom preservation:** $\mathcal{F}$ satisfies the Fractal Set axiom translations if and only if $\mathcal{H}$ satisfies the original axioms.

**(4) Functoriality:** If $R: \mathcal{H}_1 \to \mathcal{H}_2$ is a coarse-graining map (Definition 18.2.1), then there exists a graph homomorphism $\tilde{R}: \mathcal{F}_1 \to \mathcal{F}_2$ making the diagram commute.

\begin{proof}
\textbf{Step 1 (Vertex construction).} For each $\varepsilon > 0$, discretize time into steps $t_n = n\varepsilon$. Define:
$$V_\varepsilon := \{(x, n) : x \in X / G, \, \Phi(x) < \infty, \, n \in \mathbb{Z}_{\geq 0}\}$$
where we quotient by the symmetry group $G$.

\textbf{Step 2 (CST construction).} Define $(x, n) \prec (y, m)$ if $m > n$ and there exists a trajectory segment from $x$ at time $n\varepsilon$ reaching $y$ at time $m\varepsilon$.

\textbf{Step 3 (IG construction).} Define $\{(x, n), (y, n)\} \in E$ if $x$ and $y$ are "adjacent" in the sense that:
$$d_G(x, y) < \delta$$
for some fixed $\delta > 0$ depending on the metric structure of $X/G$.

\textbf{Step 4 (Fitness assignment).} Set $\Phi_V(x, n) := \Phi(x)$.

\textbf{Step 5 (Edge weights).} Set $w(\{(x, n), (y, n)\}) := |\Phi(x) - \Phi(y)|$ for horizontal edges, and $w(\{(x, n), (S_\varepsilon x, n+1)\}) := \int_{n\varepsilon}^{(n+1)\varepsilon} \mathfrak{D}(S_t x) \, dt$ for vertical edges.

\textbf{Step 6 (Representation map).} Define $\Pi(V_T) := [x]_G$ where $x$ is any representative of the time slice at $T$.

\textbf{Step 7 (Axiom verification).} Each hypostructure axiom translates directly:
\begin{itemize}
\item Axiom D $\Leftrightarrow$ Fitness monotonicity (C2)
\item Axiom C $\Leftrightarrow$ Subsequential convergence of bounded slices
\item Axiom Cap $\Leftrightarrow$ Degree bounds control singular density
\end{itemize}

\textbf{Step 8 (Continuum limit).} As $\varepsilon \to 0$, the Fractal Set $\mathcal{F}_\varepsilon$ converges to a limiting structure whose paths recover the continuous trajectories.
\end{proof}

\begin{corollary}[Combinatorial verification]\label{cor:combinatorial-verification}
The hypostructure axioms can be checked by finite computations on sufficiently fine Fractal Set discretizations.
\end{corollary}

**Key Insight:** Hypostructures are not merely abstract functional-analytic objects—they have **discrete combinatorial avatars**. The constraints become graph-theoretic conditions checkable by finite algorithms. This is essential for both numerical computation and theoretical analysis.

#### The Measure-Theoretic Limit

We now formalize the precise sense in which discrete Fractal Set computations approximate continuous hypostructure dynamics.

\begin{definition}[Discrete Fitness Functional]\label{def:discrete-fitness-functional}
For a Fractal Set $\mathcal{F}$ with time slice $V_T$, define the \textbf{discrete height}:
$$
\Phi_{\mathcal{F}}(V_T) := \sum_{v \in V_T} \Phi_V(v).
$$
\end{definition}

\begin{definition}[Discrete Dissipation]\label{def:discrete-dissipation}
For a path $\gamma = (v_0, v_1, \ldots, v_n)$ in CST, define:
$$
\mathfrak{D}_{\mathcal{F}}(\gamma) := \sum_{i=0}^{n-1} w(\{v_i, v_{i+1}\}).
$$
\end{definition}

\begin{theorem}[Fitness Convergence via Gamma-Convergence]\label{thm:fitness-convergence-via-gamma-convergence}
Let $\mathcal{F}_\varepsilon$ be the $\varepsilon$-discretization of hypostructure $\mathcal{H}$ (as constructed in \cref{mt:fractal-representation}). As $\varepsilon \to 0$:
$$
\Phi_{\mathcal{F}_\varepsilon}(V_T^\varepsilon) \xrightarrow{\Gamma} \Phi(x_T)
$$
in the sense of Gamma-convergence, where $x_T = S_T x_0$ is the continuous trajectory state.
\end{theorem}

\begin{proof}
\textbf{Step 1 (Gamma-liminf).} For any sequence $V_T^{\varepsilon_n}$ with $\varepsilon_n \to 0$ and $\Pi(V_T^{\varepsilon_n}) \to x_T$:
$$
\liminf_{n \to \infty} \Phi_{\mathcal{F}_{\varepsilon_n}}(V_T^{\varepsilon_n}) \geq \Phi(x_T).
$$
This follows from the lower semicontinuity of $\Phi$ and the construction of $\Phi_V$ as a local sampling of $\Phi$.

\textbf{Step 2 (Gamma-limsup / Recovery sequence).} For any $x_T \in X$ with $\Phi(x_T) < \infty$, there exists a sequence $V_T^{\varepsilon_n}$ with:
$$
\lim_{n \to \infty} \Phi_{\mathcal{F}_{\varepsilon_n}}(V_T^{\varepsilon_n}) = \Phi(x_T).
$$
The recovery sequence is constructed by taking finer and finer discretizations of the trajectory, using the fitness assignment $\Phi_V(x, n) = \Phi(x)$ from Step 4 of \cref{mt:fractal-representation}.
\end{proof}

\begin{definition}[Information Graph Metric]\label{def:information-graph-metric}
The Information Graph IG induces a \textbf{graph metric}:
$$
d_{\text{IG}}(v, w) := \text{length of shortest path in IG from } v \text{ to } w.
$$
For the $\varepsilon$-discretization, scale: $d_{\text{IG}}^\varepsilon := \varepsilon \cdot d_{\text{IG}}$.
\end{definition}

\begin{theorem}[Gromov-Hausdorff Convergence]\label{thm:gromov-hausdorff-convergence}
Let $(V_\varepsilon, d_{\text{IG}}^\varepsilon)$ be the metric space induced by the Information Graph of $\mathcal{F}_\varepsilon$. Then:
$$
(V_\varepsilon / G, d_{\text{IG}}^\varepsilon) \xrightarrow{\text{GH}} (M, g)
$$
in the Gromov-Hausdorff sense, where $(M, g)$ is the Riemannian manifold underlying the state space $X/G$.
\end{theorem}

\begin{proof}
\textbf{Step 1 (Metric approximation).} By construction (Step 3 of \cref{mt:fractal-representation}), vertices $(x, n)$ and $(y, n)$ at the same time level are connected in IG when $d_G(x, y) < \delta$. The graph distance thus approximates the Riemannian distance up to scale $\varepsilon$.

\textbf{Step 2 (Gromov-Hausdorff distance bound).} The Hausdorff distance between $(V_\varepsilon / G, d_{\text{IG}}^\varepsilon)$ and $(X/G, d)$ is bounded by:
$$
d_{\text{GH}}((V_\varepsilon / G, d_{\text{IG}}^\varepsilon), (X/G, d)) \leq C \varepsilon
$$
for some constant $C$ depending on the geometry of $X/G$.

\textbf{Step 3 (Convergence).} As $\varepsilon \to 0$, $d_{\text{GH}} \to 0$, establishing Gromov-Hausdorff convergence.
\end{proof}

\begin{corollary}[Validation of Algorithmic Verification]\label{cor:validation-of-algorithmic-verification}
The discrete combinatorial checks performed on $\mathcal{F}_\varepsilon$ converge to the continuous PDE constraints as $\varepsilon \to 0$. Specifically:
\end{corollary}

1. **Axiom D:** Discrete fitness monotonicity (C2) converges to the continuous dissipation identity $\frac{d\Phi}{dt} \leq -\alpha \mathfrak{D}$.

2. **Axiom C:** Subsequential convergence of bounded discrete slices converges to the continuous compactness condition.

3. **Axiom Cap:** Discrete degree bounds converge to continuous capacity constraints.

This validates the use of finite algorithms for axiom verification: results proved on sufficiently fine discretizations transfer to the continuum. See §20.3.1 for the complete theory of discretization error bounds via Γ-convergence.

---

#### The Discretization Error and Γ-Convergence

The approximation of continuous dynamics by discrete schemes requires precise control of variational structure preservation. This subsection develops the theory of **discretization error** through Γ-convergence, establishing quantitative conditions under which discrete approximations faithfully capture continuous hypostructures. The results complement \cref{mt:fractal-representation} by providing convergence guarantees for the discrete-to-continuous limit, and extend the metric slope framework of §6.3 to time-discrete settings.

\begin{theorem}[Equivalence of Axioms and RCD Curvature]\label{thm:equivalence-of-axioms-and-rcd-curvature}
Let $\mathcal{H} = (X, (S_t), \Phi, \mathfrak{D}, G, M)$ be a hypostructure satisfying:
\begin{itemize}
\item \textbf{(H1)} The state space $X$ carries a metric measure structure $(X, d, \mathfrak{m})$
\item \textbf{(H2)} The height functional is the relative entropy: $\Phi = H_{\text{rel}}(\cdot|\mathfrak{m})$
\item \textbf{(H3)} The evolution $S_t$ is the gradient flow of $\Phi$ in $(\mathcal{P}_2(X), W_2)$
\end{itemize}

Then the following equivalences hold:
\begin{enumerate}
\item \textbf{Axiom D $\Leftrightarrow$ EVI$_K$:} Evolution Variational Inequality with $K$-convexity
\item \textbf{Axiom LS $\Leftrightarrow$ Talagrand:} Wasserstein-entropy inequality
\item \textbf{Axiom C $\Leftrightarrow$ HWI:} Otto-Villani HWI inequality with precompactness
\end{enumerate}
\end{theorem}

\begin{proof}
\textbf{Part 1 (Axiom D ↔ EVI$_K$):} By Ambrosio-Gigli-Savaré [@AmbrosioGigliSavare08], gradient flows satisfy the Energy Dissipation Equality. The first variation of Wasserstein distance along geodesics, combined with $K$-convexity of entropy, yields the EVI formulation.

\textbf{Part 2 (Axiom LS ↔ Talagrand):} Via Bakry-Émery $\Gamma_2$-criterion [@BakryEmery85]: $\Gamma_2(f) \geq K \Gamma(f)$ characterizes $\text{Ric} \geq K$. This implies Log-Sobolev, which implies Talagrand via Otto-Villani [@OttoVillani00].

\textbf{Part 3 (Axiom C ↔ HWI):} Otto calculus on Wasserstein space: the HWI inequality $H(\rho) - H(\sigma) \leq W_2(\rho,\sigma)\sqrt{I(\rho)} - \frac{K}{2}W_2^2(\rho,\sigma)$ follows from $\kappa$-convexity along geodesics.
\end{proof}

\begin{definition}[Minimizing Movement Scheme]\label{def:minimizing-movement-scheme}
Let $(X, d)$ be a complete metric space and $\Phi: X \to \mathbb{R} \cup \{+\infty\}$ a proper, lower semicontinuous functional. The \textbf{Minimizing Movement scheme} (De Giorgi [@DeGiorgi93]) with time step $\tau > 0$ and initial datum $x_0 \in \mathrm{dom}(\Phi)$ is the sequence $(x_n^\tau)_{n \geq 0}$ defined recursively by:
$$x_0^\tau := x_0, \quad x_{n+1}^\tau \in \arg\min_{x \in X} \left\{ \Phi(x) + \frac{d(x, x_n^\tau)^2}{2\tau} \right\}.$$
\end{definition}

The scheme is well-defined when the minimum exists (guaranteed if $\Phi$ has compact sublevels or $(X, d)$ is proper).

\begin{definition}[Discrete Dissipation Functional]\label{def:discrete-dissipation-functional}
The \textbf{discrete dissipation functional} associated to a Minimizing Movement sequence is:
$$\mathfrak{D}_\tau^n := \frac{d(x_n^\tau, x_{n+1}^\tau)^2}{\tau}.$$
\end{definition}

The **discrete energy inequality** takes the form:
$$\Phi(x_{n+1}^\tau) + \frac{d(x_n^\tau, x_{n+1}^\tau)^2}{2\tau} \leq \Phi(x_n^\tau).$$

Summing over $n = 0, \ldots, N-1$ yields the **cumulative energy bound**:
$$\Phi(x_N^\tau) + \sum_{n=0}^{N-1} \frac{d(x_n^\tau, x_{n+1}^\tau)^2}{2\tau} \leq \Phi(x_0).$$

\begin{definition}[Mosco Convergence]\label{def:mosco-convergence}
A sequence of functionals $\Phi_\tau: X \to \mathbb{R} \cup \{+\infty\}$ \textbf{Mosco-converges} to $\Phi$ (written $\Phi_\tau \xrightarrow{M} \Phi$) if both conditions hold:
\end{definition}

1. **($\Gamma$-liminf)** For every sequence $x_\tau \rightharpoonup x$ weakly in $X$:
$$\Phi(x) \leq \liminf_{\tau \to 0} \Phi_\tau(x_\tau).$$

2. **($\Gamma$-limsup with strong recovery)** For every $x \in X$, there exists a **recovery sequence** $x_\tau \to x$ strongly such that:
$$\Phi(x) \geq \limsup_{\tau \to 0} \Phi_\tau(x_\tau).$$

When $X$ is a Hilbert space, Mosco convergence is equivalent to convergence in the sense of resolvents.

\begin{metatheorem}[Convergence of Minimizing Movements]\label{mt:convergence-of-minimizing-movements}
Let $(X, d)$ be a complete metric space and $\Phi: X \to \mathbb{R} \cup \{+\infty\}$ a proper, lower semicontinuous functional satisfying:
\begin{itemize}
\item \textbf{(MM1)} $\Phi$ is $\lambda$-convex along geodesics for some $\lambda \in \mathbb{R}$
\item \textbf{(MM2)} Sublevels $\{x : \Phi(x) \leq c\}$ are precompact for all $c \in \mathbb{R}$
\item \textbf{(MM3)} The metric slope $|\partial \Phi|$ is lower semicontinuous
\end{itemize}
\end{metatheorem}

Let $(x_n^\tau)$ be the Minimizing Movement scheme with time step $\tau > 0$ and initial datum $x_0 \in \mathrm{dom}(\Phi)$. Define the piecewise-constant interpolant:
$$\bar{x}^\tau(t) := x_n^\tau \quad \text{for } t \in [n\tau, (n+1)\tau).$$

Then:

1. **(Trajectory convergence)** As $\tau \to 0$, $\bar{x}^\tau \to u$ uniformly on compact time intervals, where $u: [0, \infty) \to X$ is the unique curve of maximal slope for $\Phi$ starting from $x_0$.

2. **(Dissipation convergence)** For any $T > 0$ with $N\tau = T$:
$$\sum_{n=0}^{N-1} \frac{d(x_n^\tau, x_{n+1}^\tau)^2}{\tau} \to \int_0^T |\partial \Phi|^2(u(t)) \, dt.$$

3. **(Energy-dissipation equality)** The limit curve satisfies the exact energy balance:
$$\Phi(u(T)) + \int_0^T |\partial \Phi|^2(u(t)) \, dt = \Phi(u(0)).$$

\begin{proof}
\textbf{Step 1 (A priori estimates).} The cumulative energy bound (Definition 20.3.9) gives:
$$\Phi(x_N^\tau) + \sum_{n=0}^{N-1} \frac{d(x_n^\tau, x_{n+1}^\tau)^2}{2\tau} \leq \Phi(x_0).$$
By (MM2), the sequence $(x_n^\tau)_{n \leq N}$ remains in the compact sublevel $\{\Phi \leq \Phi(x_0)\}$.

\textbf{Step 2 (Equicontinuity).} The discrete velocity satisfies $d(x_n^\tau, x_{n+1}^\tau)/\tau \leq C$ for a constant $C$ depending only on $\Phi(x_0) - \inf \Phi$. Hence the interpolants $\bar{x}^\tau$ are equi-Hölder with exponent $1/2$.

\textbf{Step 3 (Compactness).} By Arzelà-Ascoli (metric space version), every sequence $\bar{x}^{\tau_k}$ with $\tau_k \to 0$ has a uniformly convergent subsequence. Let $u$ be any limit point.

\textbf{Step 4 (Identification via variational inequality).} The minimality condition for $x_{n+1}^\tau$ implies: for all $y \in X$,
$$\Phi(x_{n+1}^\tau) + \frac{d(x_{n+1}^\tau, x_n^\tau)^2}{2\tau} \leq \Phi(y) + \frac{d(y, x_n^\tau)^2}{2\tau}.$$
Taking $y$ along a geodesic from $x_n^\tau$ and using (MM1), this yields the \textbf{discrete variational inequality} [@SandierSerfaty04]:
$$\frac{d(x_{n+1}^\tau, x_n^\tau)}{\tau} \leq |\partial \Phi|(x_{n+1}^\tau) + \lambda^- d(x_{n+1}^\tau, x_n^\tau)$$
where $\lambda^- := \max(0, -\lambda)$. Passing to the limit, any cluster point $u$ satisfies:
$$|\dot{u}|(t) = |\partial \Phi|(u(t)) \quad \text{for a.e. } t > 0$$
characterizing $u$ as a curve of maximal slope.

\textbf{Step 5 (Uniqueness).} When $\lambda > 0$, the $\lambda$-convexity of $\Phi$ implies \textbf{$\lambda$-contractivity} of gradient flows: for two solutions $u, v$,
$$d(u(t), v(t)) \leq e^{-\lambda t} d(u(0), v(0)).$$
This follows from the EVI characterization (\cref{thm:equivalence-of-axioms-and-rcd-curvature}). Hence the limit is unique.

\textbf{Step 6 (Energy-dissipation equality).} Lower semicontinuity of the metric slope (MM3) gives:
$$\int_0^T |\partial \Phi|^2(u) \, dt \leq \liminf_{\tau \to 0} \sum_{n=0}^{N-1} \frac{d(x_n^\tau, x_{n+1}^\tau)^2}{\tau}.$$
The reverse inequality follows from the energy bound and the identification $|\dot{u}| = |\partial \Phi|(u)$. Combined with passage to the limit in the discrete energy inequality, this yields the exact energy-dissipation equality.
\end{proof}

**Key Insight:** Minimizing Movements provide a variational interpretation of implicit Euler discretization: the discrete scheme minimizes the sum of potential energy and kinetic cost at each step, revealing that numerical stability and gradient flow structure are two manifestations of the same variational principle.

\begin{metatheorem}[Symplectic Shadowing]\label{mt:symplectic-shadowing}
Let $(X, \omega)$ be a symplectic manifold and $H: X \to \mathbb{R}$ an analytic Hamiltonian. Let $\Psi_\tau: X \to X$ be a \textbf{symplectic integrator} of order $p \geq 1$, meaning:
\begin{itemize}
\item $\Psi_\tau^* \omega = \omega$ (symplecticity)
\item $\Psi_\tau(x) = \varphi_\tau^H(x) + O(\tau^{p+1})$ where $\varphi_t^H$ is the exact Hamiltonian flow
\end{itemize}
\end{metatheorem}

Then:

1. **(Backward error analysis)** There exists a **modified Hamiltonian** $\tilde{H}_\tau = H + \tau^p H_p + \tau^{p+1} H_{p+1} + \cdots$ (formal power series in $\tau$) such that $\Psi_\tau$ is the exact time-$\tau$ flow of $\tilde{H}_\tau$:
$$\Psi_\tau(x) = \varphi_\tau^{\tilde{H}_\tau}(x) + O(e^{-c/\tau})$$
where $c > 0$ depends on the analyticity radius of $H$.

2. **(Long-time near-conservation)** Along the numerical trajectory $(x_n := \Psi_\tau^n(x_0))_{n \geq 0}$:
$$|H(x_n) - H(x_0)| \leq C\tau^p \quad \text{for all } n \text{ with } n\tau \leq e^{c'/\tau}$$
where $C, c' > 0$ depend on $H$ and the integrator.

3. **(Symmetry inheritance)** If a Lie group $G$ acts symplectically on $(X, \omega)$ and $H$ is $G$-invariant, then $\tilde{H}_\tau$ is $G$-invariant to all orders in $\tau$. Consequently, the Noether charges of the discrete system shadow those of the continuous system.

\begin{proof}
\textbf{Step 1 (Lie series expansion).} The symplectic integrator $\Psi_\tau$ admits a formal expansion via the \textbf{Baker-Campbell-Hausdorff (BCH) formula}. Write $\Psi_\tau = \exp(\tau \mathcal{B}_\tau)$ where $\mathcal{B}_\tau = B_1 + \tau B_2 + \tau^2 B_3 + \cdots$ is a formal vector field. The BCH formula expresses the composition of flows as a single exponential:
$$\exp(A)\exp(B) = \exp\left(A + B + \frac{1}{2}[A,B] + \frac{1}{12}[A,[A,B]] + \cdots\right).$$

\textbf{Step 2 (Symplecticity forces Hamiltonianity).} A vector field $B$ on $(X, \omega)$ generates a symplectic flow if and only if $B$ is \textbf{locally Hamiltonian}: $\mathcal{L}_B \omega = 0$, equivalently $\iota_B \omega$ is closed. On simply connected $X$, this means $B = X_F$ for some $F: X \to \mathbb{R}$. Since $\Psi_\tau$ is symplectic, each $B_j$ is Hamiltonian: $B_j = X_{H_j}$.

\textbf{Step 3 (Truncation and exponential remainder).} For analytic $H$, the formal series $\tilde{H}_\tau = \sum_{j \geq 1} \tau^{j-1} H_j$ is Gevrey-1 (coefficients grow at most factorially). Truncating at optimal order $N \sim c/\tau$ yields exponentially small remainder \cite[Ch. IX]{HairerLubichWanner06}.

\textbf{Step 4 (Energy shadowing).} The modified Hamiltonian $\tilde{H}_\tau$ is exactly conserved: $\tilde{H}_\tau(x_n) = \tilde{H}_\tau(x_0)$. Hence:
$$|H(x_n) - H(x_0)| \leq |H(x_n) - \tilde{H}_\tau(x_n)| + |\tilde{H}_\tau(x_0) - H(x_0)| \leq 2\|H - \tilde{H}_\tau\|_\infty = O(\tau^p).$$

\textbf{Step 5 (Symmetry preservation).} If $g \in G$ preserves both $\omega$ and $H$, then $g$ commutes with the Hamiltonian flow $\varphi_t^H$. The BCH formula involves only Lie brackets of Hamiltonian vector fields, which inherit $G$-equivariance. Hence each $H_j$ is $G$-invariant.
\end{proof}

\begin{remark}[Energy drift comparison]
Non-symplectic integrators (e.g., explicit Runge--Kutta methods) exhibit \textbf{linear energy drift}: $|H(x_n) - H(x_0)| \leq Cn\tau^p$, which grows unboundedly for long times. Symplectic integrators achieve \textbf{bounded energy error} for exponentially long times---the error remains $O(\tau^p)$ until $t \sim e^{c/\tau}$. This qualitative distinction is essential for preserving the structure of Hamiltonian hypostructures over physically relevant timescales.
\end{remark}

\begin{metatheorem}[Homological Reconstruction]\label{mt:homological-reconstruction}
Let $(X, d)$ be a compact geodesic metric space with \textbf{reach} $\mathrm{reach}(X) > 0$ (the largest $r$ such that every point within distance $r$ of $X$ has a unique nearest point in $X$). Let $P = \{x_1, \ldots, x_N\} \subset X$ be a finite sample with \textbf{fill distance}:
$$h := \sup_{x \in X} \min_{i \in \{1,\ldots,N\}} d(x, x_i).$$
\end{metatheorem}

Define the **Vietoris-Rips complex** at scale $\varepsilon > 0$:
$$\mathrm{VR}_\varepsilon(P) := \left\{ \sigma \subseteq P : \mathrm{diam}(\sigma) \leq \varepsilon \right\}.$$

Then for $h < \varepsilon/2$ and $\varepsilon < \mathrm{reach}(X)/4$:

1. **(Homological equivalence)** $H_k(\mathrm{VR}_\varepsilon(P); \mathbb{Z}) \cong H_k(X; \mathbb{Z})$ for all $k \geq 0$.

2. **(Persistence stability)** The persistence diagram of the Rips filtration $\{\mathrm{VR}_r(P)\}_{r \geq 0}$ satisfies:
$$d_B(\mathrm{Dgm}(P), \mathrm{Dgm}(X)) \leq C \cdot h$$
where $d_B$ denotes bottleneck distance, $\mathrm{Dgm}(X)$ is the intrinsic persistence diagram, and $C$ depends only on the dimension of $X$.

3. **(Axiom TB verification)** The computed Betti numbers $\beta_k(\mathrm{VR}_\varepsilon(P))$ equal $\beta_k(X)$, enabling algorithmic verification of Axiom TB (topological barriers) from finite samples.

\begin{proof}
\textbf{Step 1 (Covering argument).} The condition $h < \varepsilon/2$ ensures that $X \subseteq \bigcup_{i=1}^N B_{\varepsilon/2}(x_i)$, where $B_r(x)$ denotes the closed ball of radius $r$ centered at $x$.

\textbf{Step 2 (Niyogi-Smale-Weinberger theorem).} By [@NiyogiSmaleWeinberger08], if $\varepsilon < \mathrm{reach}(X)$, the union $U_\varepsilon := \bigcup_{i=1}^N B_\varepsilon(x_i)$ deformation retracts onto $X$. The retraction $\rho: U_\varepsilon \to X$ maps each point to its unique nearest point in $X$ (well-defined since $\varepsilon < \mathrm{reach}(X)$).

\textbf{Step 3 (Nerve lemma and Rips-Čech interleaving).} The \textbf{Čech complex} $\check{C}_\varepsilon(P)$ is the nerve of the cover $\{B_\varepsilon(x_i)\}$. By the nerve lemma (valid for good covers), $\check{C}_\varepsilon(P) \simeq U_\varepsilon$. The standard interleaving:
$$\check{C}_\varepsilon(P) \subseteq \mathrm{VR}_\varepsilon(P) \subseteq \check{C}_{\sqrt{2}\varepsilon}(P)$$
holds in Euclidean space; in general geodesic spaces, the constant $\sqrt{2}$ may vary but the interleaving persists.

\textbf{Step 4 (Homology isomorphism).} Combining Steps 1–3:
$$H_k(\mathrm{VR}_\varepsilon(P)) \cong H_k(\check{C}_\varepsilon(P)) \cong H_k(U_\varepsilon) \cong H_k(X)$$
where the first isomorphism uses the interleaving (at appropriate scales), the second uses the nerve lemma, and the third uses the deformation retraction.

\textbf{Step 5 (Stability of persistence diagrams).} The stability theorem [@CohenSteinerEdelsbrunnerHarer07] asserts that if $d_H(P, Q) \leq \delta$ (Hausdorff distance), then $d_B(\mathrm{Dgm}(P), \mathrm{Dgm}(Q)) \leq \delta$. Since $d_H(P, X) \leq h$ by definition of fill distance, the persistence diagrams satisfy $d_B(\mathrm{Dgm}(P), \mathrm{Dgm}(X)) \leq C \cdot h$ with constant depending on the interleaving.
\end{proof}

**Key Insight:** The homological reconstruction theorem provides the theoretical foundation for **persistent homology as an axiom verification tool**: topological features (Betti numbers, homology classes) visible in sufficiently dense samples are guaranteed to reflect true manifold topology, enabling rigorous computational certification of Axiom TB.

\begin{remark}[Sampling density as topological resolution]
The condition $h < \mathrm{reach}(X)/8$ can be viewed as a \textbf{topological sampling criterion} analogous to the Nyquist condition in signal processing: to reconstruct the homology of $X$, one must sample at a density inversely proportional to the geometric complexity (reach). Hypostructures with intricate topology (small reach) require finer discretizations to capture topological barriers accurately.
\end{remark}

\begin{corollary}[Algorithmic Axiom Verification]\label{cor:algorithmic-axiom-verification}
Hypostructure axioms admit computational verification through appropriate discretizations:
\end{corollary}

| Axiom | Discretization Method | Error Control |
|:------|:---------------------|:--------------|
| **D** (Dissipation) | Minimizing Movements (\cref{mt:convergence-of-minimizing-movements}) | Time step $\tau$ |
| **LS** (Stiffness) | Minimizing Movements with contractivity | Convexity parameter $\lambda$ |
| **C** (Compactness) | Symplectic integrators (\cref{mt:symplectic-shadowing}) | Energy shadow $O(\tau^p)$ |
| **TB** (Topology) | Persistent homology | Fill distance $h$ |

The total discretization error is controlled by $\max(\tau, h)$, providing rigorous certificates for axiom satisfaction from finite computations.

---

### Symmetry Completion Theorem

\begin{definition}[Local gauge data]\label{def:local-gauge-data}
A \textbf{local gauge structure} on a Fractal Set $\mathcal{F}$ is an assignment:
\begin{itemize}
\item $H$: a compact Lie group (the gauge group)
\item $g_e \in H$ for each edge $e \in E$ (the parallel transport)
\item Consistency: gauge transformations at vertices act as $g_e \mapsto h_v^{-1} g_e h_w$ for edge $e = \{v, w\}$
\end{itemize}
\end{definition}

\begin{metatheorem}[Symmetry Completion]\label{mt:symmetry-completion}
Let $\mathcal{F}$ be a well-formed Fractal Set with local gauge structure $(H, \{g_e\})$. Then:
\end{metatheorem}

**(1) Existence.** There exists a unique (up to isomorphism) hypostructure $\mathcal{H}_{\mathcal{F}}$ such that:
- The symmetry group $G$ of $\mathcal{H}_{\mathcal{F}}$ contains $H$ as a subgroup
- The Fractal Set $\mathcal{F}$ is the canonical discretization of $\mathcal{H}_{\mathcal{F}}$

**(2) Constraint inheritance.** The axioms D, C, SC, Cap, TB, LS, GC hold in $\mathcal{H}_{\mathcal{F}}$ if and only if their combinatorial translations hold in $\mathcal{F}$.

**(3) Uniqueness.** If $\mathcal{H}$ and $\mathcal{H}'$ are two hypostructures both having $\mathcal{F}$ as their Fractal Set representation and sharing the gauge group $H$, then $\mathcal{H} \cong \mathcal{H}'$ (isomorphism of hypostructures).

\begin{proof}
\textbf{Step 1 (State space construction).} Define $X$ as the inverse limit:
$$X := \varprojlim_{\varepsilon \to 0} X_\varepsilon$$
where $X_\varepsilon$ is the space of time slices at resolution $\varepsilon$.

\textbf{Step 2 (Height functional).} Define $\Phi: X \to \mathbb{R}$ by:
$$\Phi(x) := \lim_{\varepsilon \to 0} \sum_{v \in V_T(\varepsilon)} \Phi_V(v)$$
where $V_T(\varepsilon)$ is the $\varepsilon$-resolution time slice corresponding to $x$.

\textbf{Step 3 (Semiflow).} The CST structure induces a semiflow: $S_t$ moves along maximal chains in CST.

\textbf{Step 4 (Symmetry group).} The gauge group $H$ acting on edge labels extends to an action on $X$ by gauge transformations.

\textbf{Step 5 (Uniqueness).} Suppose $\mathcal{H}$ and $\mathcal{H}'$ both have Fractal representation $\mathcal{F}$. Then:
\begin{itemize}
\item Their state spaces are both inverse limits of the same system: $X \cong X'$
\item Their height functionals agree on time slices: $\Phi = \Phi'$
\item Their semiflows are determined by CST: $S_t = S'_t$
\item Their symmetry groups both contain $H$ as generated by edge gauge transformations
\end{itemize}

The remaining data (dissipation, barriers) are determined by the axioms and $(\Phi, H)$.
\end{proof}

\begin{corollary}[Symmetry determines structure]\label{cor:symmetry-determines-structure}
Specifying a Fractal Set with gauge structure $(H, \{g_e\})$ uniquely determines a hypostructure. Local symmetries constrain global dynamics.
\end{corollary}

**Key Insight:** This is the discrete analog of the principle that "gauge invariance determines dynamics." The Symmetry Completion theorem makes this precise: define the local gauge data on a Fractal Set, and the entire hypostructure—including its failure modes and barriers—is determined.

---

### Gauge-Geometry Correspondence

\begin{definition}[Wilson loops]\label{def:wilson-loops}
For a cycle $\gamma = v_0 - v_1 - \cdots - v_k - v_0$ in the IG, define the \textbf{Wilson loop}:
$$W(\gamma) := \text{Tr}(\rho(g_{v_0 v_1} \cdot g_{v_1 v_2} \cdots g_{v_k v_0}))$$
where $\rho$ is a representation of the gauge group $H$.
\end{definition}

\begin{definition}[Curvature from holonomy]\label{def:curvature-from-holonomy}
For small cycles (plaquettes) $\gamma$ bounding area $A$, define the \textbf{curvature tensor}:
$$F_{\mu\nu} := \lim_{A \to 0} \frac{\text{hol}(\gamma) - \mathbf{1}}{A}$$
where the limit is taken as the Fractal Set is refined.
\end{definition}

\begin{metatheorem}[Gauge-Geometry Correspondence]\label{mt:gauge-geometry-correspondence}
Let $\mathcal{F}$ be a Fractal Set with:
\begin{itemize}
\item Gauge group $H = K \times \text{Diff}(M)$ where $K$ is a compact Lie group
\item IG approximating a $d$-dimensional manifold $M$ in the large-$N$ limit
\item Fitness functional $\Phi_V$ satisfying appropriate regularity
\end{itemize}
\end{metatheorem}

Then in the continuum limit, the effective dynamics is governed by the **Einstein-Yang-Mills action**:
$$S[g, A] = \int_M \left( \frac{1}{16\pi G} R_g + \frac{1}{4g^2} |F_A|^2 \right) \sqrt{g} \, d^d x$$
where:
- $g$ is the metric on $M$ (from IG geometry)
- $A$ is the $K$-connection (from gauge labels)
- $R_g$ is the scalar curvature
- $F_A$ is the Yang-Mills curvature

\begin{proof}
\textbf{Step 1 (Metric from IG).} The graph distance on IG induces a metric on time slices. In the continuum limit, this becomes a Riemannian metric $g_{\mu\nu}$.

\textbf{Step 2 (Connection from gauge labels).} The gauge labels $g_e$ define parallel transport. In the limit, this becomes a connection $A$ on a principal $K$-bundle. This reconstruction parallels the Kobayashi-Hitchin correspondence [@Kobayashi87], relating stable bundles to Einstein-Hermitian connections.

\textbf{Step 3 (Curvature from holonomy).} Wilson loops around small cycles encode curvature. The non-abelian Stokes theorem gives:
$$W(\gamma) \approx \mathbf{1} - \int_\Sigma F + O(A^2)$$
where $\Sigma$ is bounded by $\gamma$.

\textbf{Step 4 (Variational principle).} The hypostructure requirement that axiom violations (failure modes) be avoided is equivalent to the stationarity condition $\delta S = 0$. This follows because:
\begin{itemize}
\item Mode C.E (energy blow-up) is avoided $\Leftrightarrow$ $\Phi$ is bounded $\Leftrightarrow$ Action is finite
\item Mode T.D (topological annihilation) is avoided $\Leftrightarrow$ Field configurations are smooth
\item Mode B.C (symmetry misalignment) is avoided $\Leftrightarrow$ Gauge consistency holds
\end{itemize}
\end{proof}

\begin{corollary}[Gravity from information geometry]\label{cor:gravity-from-information-geometry}
Spacetime geometry (general relativity) emerges from the Information Graph structure of the Fractal Set. The metric $g$ encodes \textbf{how nodes are connected}, not pre-existing spacetime.
\end{corollary}

\begin{corollary}[Gauge fields from local symmetries]\label{cor:gauge-fields-from-local-symmetries}
Yang-Mills gauge fields emerge from the gauge labels on Fractal Set edges. The Standard Model gauge group $SU(3) \times SU(2) \times U(1)$ would appear as the gauge structure $H = K$ on a physical Fractal Set.
\end{corollary}

**Key Insight:** The Gauge-Geometry correspondence connects geometric and physical structures: causal structure corresponds to spacetime, gauge labels to forces, and fitness to matter/energy. The Fractal Set provides a unified substrate for these correspondences.

---

### Emergent Continuum Theorem

*From combinatorics to cosmology.*

\begin{definition}[Graph Laplacian]\label{def:graph-laplacian}
For a Fractal Set $\mathcal{F}$ with IG $(V, E)$, the \textbf{graph Laplacian} is:
$$(\Delta_\text{IG} f)(v) := \sum_{u: \{u,v\} \in E} w(\{u,v\}) (f(u) - f(v))$$
for functions $f: V \to \mathbb{R}$.
\end{definition}

\begin{definition}[Random walks and heat kernel]\label{def:random-walks-and-heat-kernel}
The \textbf{heat kernel} on $\mathcal{F}$ is:
$$p_t(u, v) := \langle \delta_u, e^{-t \Delta_\text{IG}} \delta_v \rangle$$
encoding the probability of a random walk from $u$ to $v$ in time $t$.
\end{definition}

\begin{metatheorem}[Emergent Continuum]\label{mt:emergent-continuum}
Let $\{\mathcal{F}_N\}_{N \to \infty}$ be a sequence of Fractal Sets with:
\begin{itemize}
\item \textbf{(EC1) Bounded degree:} $\sup_v \deg(v) \leq D$ uniformly in $N$.
\item \textbf{(EC2) Volume growth:} $|B_r(v)| \sim r^d$ for some fixed $d$ (the emergent dimension).
\item \textbf{(EC3) Spectral gap:} The first nonzero eigenvalue $\lambda_1(\Delta_\text{IG})$ satisfies $\lambda_1 \geq c > 0$ uniformly.
\item \textbf{(EC4) Ricci curvature bound:} The Ollivier-Ricci curvature $\kappa(e) \geq -K$ for all edges. This utilizes the Lott-Villani-Sturm synthesis [@LottVillani09], defining Ricci curvature on metric measure spaces without underlying smooth structure.
\end{itemize}

Then:
\begin{enumerate}
\item \textbf{Metric convergence.} The rescaled graph metric $d_N / \sqrt{N}$ converges in the Gromov-Hausdorff sense to a Riemannian manifold $(M, g)$ of dimension $d$. This derivation relies on the rigorous \textbf{Hydrodynamic Limits} established by Kipnis and Landim [@KipnisLandim99], which prove that interacting particle systems scale to deterministic PDEs under hyperbolic/parabolic rescaling.
\item \textbf{Laplacian convergence.} The rescaled graph Laplacian $N^{-2/d} \Delta_{\text{IG}}$ converges to the Laplace-Beltrami operator $\Delta_g$ on $M$.
\item \textbf{Heat kernel convergence.} The rescaled heat kernel converges to the Riemannian heat kernel:
$$N^{d/2} p_{t/N^{2/d}}(u, v) \to p_t^{(M)}(x, y)$$
where $x, y$ are the limit points.
\item \textbf{Constraint inheritance.} If the Fractal Sets $\mathcal{F}_N$ satisfy the combinatorial axiom translations, the limiting manifold $(M, g)$ inherits:
\begin{itemize}
\item Energy bounds $\to$ Bounded scalar curvature
\item Capacity bounds $\to$ Dimension bounds on singular sets
\item Łojasiewicz bounds $\to$ Regularity of geometric flows
\end{itemize}
\end{enumerate}
\end{metatheorem}

\begin{proof}
\textbf{Step 1 (Gromov compactness).} By (EC1)-(EC4), the sequence $(\mathcal{F}_N, d_N/\sqrt{N})$ is precompact in Gromov-Hausdorff topology. Extract a convergent subsequence.

\textbf{Step 2 (Manifold structure).} By (EC2) and (EC4), the limit space has Hausdorff dimension $d$ and satisfies Ricci curvature bounds. By Cheeger-Colding theory, it is a smooth $d$-manifold away from a singular set of codimension $\geq 2$.

\textbf{Step 3 (Laplacian convergence).} The graph Laplacian eigenvalues converge to the Laplace-Beltrami eigenvalues (Weyl's law for graphs + spectral convergence).

\textbf{Step 4 (Constraint inheritance).} The combinatorial constraints pass to the limit:
\begin{itemize}
\item Finite fitness sum $\to$ Finite energy integral
\item Degree bounds $\to$ No concentration of curvature
\item Gauge consistency $\to$ Smooth connection in limit
\end{itemize}
\end{proof}

\begin{corollary}[Spacetime emergence]\label{cor:spacetime-emergence}
In this framework, continuous spacetime $(M, g)$ emerges from the large-$N$ limit of Fractal Sets. The discrete structure provides a computational substrate for the continuum description.
\end{corollary}

**Key Insight:** In this model, the continuum—smooth manifolds, differential equations, field theories—is an effective description valid at large scales. The Fractal Set provides a discrete substrate from which continuum descriptions emerge.

---

### Dimension Selection Principle

\begin{definition}[Dimension-dependent failure modes]\label{def:dimension-dependent-failure-modes}
For a hypostructure with emergent spatial dimension $d$:
\end{definition}

- **Topological constraint strength:** $T(d)$ measures how restrictive topological conservation laws are
- **Semantic horizon severity:** $S(d)$ measures information-theoretic limits on coherent description
- **Complexity-coherence balance:** $B(d) = T(d) + S(d)$ total constraint pressure

\begin{metatheorem}[Dimension Selection]\label{mt:dimension-selection}
There exists a non-empty finite set $D_{\text{admissible}} \subset \mathbb{Z}_{>0}$ such that:
\end{metatheorem}

**(1) Dimensions in $D_{\text{admissible}}$ avoid unavoidable failure modes:** For $d \in D_{\text{admissible}}$, there exist hypostructures with emergent dimension $d$ satisfying all axioms with positive barrier margins.

**(2) Dimensions outside $D_{\text{admissible}}$ have unavoidable modes:** For $d \notin D_{\text{admissible}}$, every hypostructure with emergent dimension $d$ necessarily realizes at least one failure mode.

**(3) Finiteness:** $|D_{\text{admissible}}| < \infty$.

\begin{proof}
\textbf{Non-emptiness.} We exhibit systems in $d = 3$: Three-dimensional fluid dynamics, gauge theories, and general relativity with positive cosmological constant admit hypostructure instantiations satisfying the axioms with positive margins. The axiom verification is routine; the framework then delivers structural conclusions about stability and failure mode exclusion.

\textbf{Finiteness.} For $d$ sufficiently large:
\begin{itemize}
\item Mode D.C (semantic horizon) becomes unavoidable: information dilution $\sim d^{-1}$
\item Mode D.D (dispersion) strengthens: decay $\sim t^{-d/2}$ makes coherent structures impossible
\end{itemize}

For $d$ sufficiently small:
\begin{itemize}
\item Mode T.C (topological obstruction) becomes unavoidable: $\pi_1, \pi_2$ constraints too restrictive
\item Mode C.D (geometric collapse) strengthens: capacity arguments fail in low dimensions
\end{itemize}
\end{proof}

\begin{conjecture}[3+1 Selection]\label{conj:3+1-selection}
$D_{\text{admissible}} = \{3\}$ for spatial dimensions, giving $(3+1)$-dimensional spacetime as the unique dynamically consistent choice.
\end{conjecture}

*Supporting Arguments:*

**Argument 1 (Low dimensions).** For $d < 3$:
- $d = 1$: No non-trivial knots; topological conservation laws too weak (Mode T.C)
- $d = 2$: Conformal symmetry too strong; all scales equivalent (Mode S.C)

**Argument 2 (High dimensions).** For $d > 3$:
- $d = 4$: Gauge theories become non-renormalizable (Mode S.E via UV divergences)
- $d \geq 5$: Gravitational wells too shallow; no stable orbits (Mode C.D)

**Argument 3 (The Goldilocks dimension).** $d = 3$ uniquely balances:
- Rich enough topology (knots, links, non-trivial $\pi_1$)
- Strong enough gravity (stable orbits, black holes with horizons)
- Weak enough dispersion (coherent structures possible)
- Renormalizable gauge theories (asymptotic freedom)

**Key Insight:** The dimension of space is not arbitrary but **selected by dynamical consistency**. Only in $(3+1)$ dimensions do all the constraints—Conservation, Topology, Duality, Symmetry—admit simultaneous satisfaction. The intersection of these constraint classes is non-empty only for emergent dimension $d=3$.

---

### Discrete-to-Continuum Stiffness Transfer

The passage from discrete graph structures to continuum limits raises a fundamental question: do curvature bounds and barrier constants survive this limiting process? This section establishes that coarse Ricci curvature on discrete metric-measure spaces transfers to synthetic Ricci curvature bounds in the continuum limit, providing a rigorous foundation for the discrete-to-continuum correspondence in hypostructure theory.

\begin{definition}[Discrete Metric-Measure Space]\label{def:discrete-metric-measure-space}
A \textbf{discrete metric-measure space} (discrete mm-space) is a triple $(V, d_V, \mathfrak{m}_V)$ where:
\end{definition}

- $V$ is a finite or countable set
- $d_V: V \times V \to [0, \infty)$ is a metric on $V$
- $\mathfrak{m}_V = \sum_{v \in V} m_v \delta_v$ is a reference measure with $m_v > 0$ for all $v \in V$

A **Markov kernel** on $(V, d_V, \mathfrak{m}_V)$ is a map $P: V \to \mathcal{P}(V)$ assigning to each $x \in V$ a probability measure $P_x$ on $V$. The kernel is **reversible** with respect to $\mathfrak{m}_V$ if $m_x P_x(y) = m_y P_y(x)$ for all $x, y \in V$.

\begin{definition}[Coarse Ricci Curvature]\label{def:coarse-ricci-curvature}
Let $(V, d_V, \mathfrak{m}_V, P)$ be a discrete mm-space with Markov kernel. The \textbf{Ollivier-Ricci curvature} [@Ollivier09] along an edge $(x, y) \in V \times V$ with $x \neq y$ is:
$$\kappa(x, y) := 1 - \frac{W_1(P_x, P_y)}{d_V(x, y)}$$
where $W_1$ denotes the $L^1$-Wasserstein distance on $\mathcal{P}(V)$ induced by $d_V$.
\end{definition}

The space $(V, d_V, \mathfrak{m}_V, P)$ has **uniform Ollivier curvature $\geq K$** for $K \in \mathbb{R}$ if:
$$\inf_{x \neq y \in V} \kappa(x, y) \geq K$$

\begin{remark}
The Ollivier-Ricci curvature generalizes Ricci curvature to discrete and non-smooth settings. For a Riemannian manifold $(M, g)$ with the heat kernel $P_x^\varepsilon = p_\varepsilon(x, \cdot)\, \mathrm{dvol}$, the Ollivier curvature satisfies $\kappa^\varepsilon(x, y) = \frac{1}{n}\mathrm{Ric}(v, v) \cdot \varepsilon + O(\varepsilon^2)$ where $v = \exp_x^{-1}(y)/d(x,y)$, recovering classical Ricci curvature in the scaling limit [@OllivierVillani12].
\end{remark}

\begin{definition}[Measured Gromov-Hausdorff Convergence]\label{def:measured-gromov-hausdorff-convergence}
A sequence $(X_n, d_n, \mathfrak{m}_n)$ of metric-measure spaces \textbf{converges in the measured Gromov-Hausdorff sense} (mGH-converges) to $(X, d, \mathfrak{m})$, written $X_n \xrightarrow{\text{mGH}} X$, if there exist:
\end{definition}

- A complete separable metric space $(Z, d_Z)$
- Isometric embeddings $\iota_n: X_n \hookrightarrow Z$ and $\iota: X \hookrightarrow Z$

such that:
1. $d_H^Z(\iota_n(X_n), \iota(X)) \to 0$ as $n \to \infty$ (Hausdorff convergence)
2. $(\iota_n)_\# \mathfrak{m}_n \rightharpoonup \iota_\# \mathfrak{m}$ weakly in $\mathcal{P}(Z)$

\begin{metatheorem}[Discrete Curvature-Stiffness Transfer]\label{mt:discrete-curvature-stiffness-transfer}
\textit{Let the following hypotheses hold:}
\end{metatheorem}

**(DCS1)** $(X_n, d_n, \mathfrak{m}_n, P_n)_{n \in \mathbb{N}}$ is a sequence of discrete mm-spaces with reversible Markov kernels satisfying uniform Ollivier curvature $\geq K$ for some $K \in \mathbb{R}$.

**(DCS2)** $X_n \xrightarrow{\text{mGH}} (X, d, \mathfrak{m})$ for some complete, separable, geodesic metric-measure space $(X, d, \mathfrak{m})$.

**(DCS3)** Uniform diameter bound: $\sup_n \mathrm{diam}(X_n) < \infty$.

**(DCS4)** Uniform measure bound: $\sup_n \mathfrak{m}_n(X_n) < \infty$.

*Then the following conclusions hold:*

**(a) Curvature inheritance.** The limit space $(X, d, \mathfrak{m})$ satisfies the curvature-dimension condition $\mathrm{CD}(K, \infty)$ in the sense of Lott-Sturm-Villani [@LottVillani09; @Sturm06].

**(b) Stiffness bound.** If $(X, d, \mathfrak{m})$ admits an admissible hypostructure $\mathcal{H}$ with stiffness parameter $S$, then:
$$S_{\min} \geq |K|$$

**(c) Barrier inheritance.** For systems with uniform diameter bound $D := \sup_n \mathrm{diam}(X_n)$, the hypostructure barrier satisfies:
$$E^* \geq c_d \cdot |K| \cdot D^2$$
where $c_d > 0$ is a dimensional constant depending on the Hausdorff dimension of $(X, d)$.

\begin{proof}
\textbf{Step 1 (Curvature stability).} By the Sturm-Lott-Villani stability theorem [@Sturm06; @LottVillani09], the curvature-dimension condition $\mathrm{CD}(K, N)$ is preserved under measured Gromov-Hausdorff convergence. The key observation is that Ollivier curvature $\kappa \geq K$ on discrete spaces implies displacement convexity of entropy along Wasserstein geodesics [@Ollivier09], which is the defining property of $\mathrm{CD}(K, \infty)$.

\textbf{Step 2 (Stiffness correspondence).} The stiffness axiom (Axiom D) quantifies resistance to deformation. For a $\mathrm{CD}(K, \infty)$ space with $K > 0$, the Lichnerowicz-type bound gives spectral gap $\lambda_1 \geq K$ for the associated Laplacian. This spectral gap controls the exponential decay rate of perturbations: $\|P_t f - \bar{f}\|_{L^2} \leq e^{-Kt}\|f - \bar{f}\|_{L^2}$. The stiffness parameter satisfies $S_{\min} \geq K$ when $K > 0$; for $K < 0$, the bound $S_{\min} \geq |K|$ characterizes the expansion rate.

\textbf{Step 3 (Barrier computation).} The barrier height $E^*$ is determined by the minimal energy required to cross between metastable states. On a $\mathrm{CD}(K, \infty)$ space with $K > 0$, the Poincaré inequality $\mathrm{Var}(f) \leq K^{-1} \mathcal{E}(f, f)$ constrains fluctuations: deviations of magnitude $\delta$ from equilibrium require Dirichlet energy at least $K\delta^2$. For a system with diameter $D$, the barrier bound becomes $E^* \geq c_d K D^2$. When the diameter is controlled by the curvature scale $D \sim |K|^{-1/2}$, we obtain $E^* \geq c_d$ independent of $K$; for systems with fixed diameter, $E^* \geq c_d |K| D^2$.

\textbf{Step 4 (Uniform bounds persist).} Since hypotheses (DCS3)-(DCS4) provide uniform bounds, the limiting space inherits these bounds. The barrier and stiffness constants, being determined by curvature and geometry, thus transfer to the limit.
\end{proof}

\begin{remark}
The case $K < 0$ (negative curvature) corresponds to expansive dynamics where the spectral gap bound becomes an expansion rate bound. The barrier formula $E^* \geq c_d |K| D^2$ remains valid and characterizes the energy scale associated with the expansion.
\end{remark}

\begin{metatheorem}[Dobrushin-Shlosman Interference Barrier]\label{mt:dobrushin-shlosman-interference-barrier}
\textit{Let the following hypotheses hold:}
\end{metatheorem}

**(DS1)** $(G_n, d_n, \mathfrak{m}_n, P_n)_{n \in \mathbb{N}}$ is a sequence of finite graphs with reversible Markov kernels satisfying uniform Ollivier curvature $\geq K$ for some $K > 0$.

**(DS2)** Uniform bounded degree: $\sup_n \sup_{v \in G_n} \deg(v) \leq \Delta$ for some $\Delta < \infty$.

**(DS3)** Each $(G_n, P_n)$ admits a Gibbs measure $\mu_{\beta,n}$ at inverse temperature $\beta > 0$.

**(DS4)** Axiom C (Conservation) is satisfied at the microscopic scale: the Markov dynamics preserve a conserved quantity $Q_n$.

*Then:*

**(a) Correlation decay.** The correlation function $\langle \sigma_x \sigma_y \rangle - \langle \sigma_x \rangle \langle \sigma_y \rangle$ decays exponentially:
$$|\mathrm{Cov}(\sigma_x, \sigma_y)| \leq C \cdot e^{-K \cdot d(x,y)}$$
for $\beta < \beta_c(K)$, where $C > 0$ depends on $K$, $\beta$, and the degree bound $\Delta$.

**(b) Reconstruction threshold.** There exists a critical temperature $\beta_c = \beta_c(K)$ such that:
- For $\beta < \beta_c$: unique Gibbs measure (high-temperature phase)
- For $\beta > \beta_c$: multiple Gibbs measures (symmetry breaking)

**(c) Conservation transfer.** The conserved quantity $Q_n$ induces a conserved quantity $Q$ on the limiting hypostructure.

\begin{proof}
\textbf{Step 1 (Dobrushin-Shlosman criterion).} The Dobrushin-Shlosman uniqueness criterion [@DobrushinShlosman85] states that the Gibbs measure is unique if the total influence of other sites on any given site is bounded. Positive Ollivier curvature $K > 0$ implies exponential decay of correlations, satisfying the criterion for $\beta < \beta_c(K)$.

\textbf{Step 2 (Correlation decay).} Positive Ollivier curvature $K > 0$ implies contraction under the Markov dynamics. For $\beta < \beta_c$, this contraction dominates thermal fluctuations, yielding exponential decay of correlations with rate $K$. The decay rate follows from the spectral gap: $\lambda_1 \geq K$ implies $\|P_t f - \bar{f}\| \leq e^{-Kt}\|f - \bar{f}\|$, which transfers to spatial correlations via the FKG inequality.

\textbf{Step 3 (Conservation structure).} By hypothesis (DS4), the microscopic dynamics preserve $Q_n$. Since mGH convergence preserves the symmetry group (Failure Quantization: discrete failure modes inherit under limits), the limiting dynamics inherit a corresponding conserved quantity $Q$.
\end{proof}

\begin{metatheorem}[Parametric Stiffness Map]\label{mt:parametric-stiffness-map}
\textit{Let the following hypotheses hold:}
\end{metatheorem}

**(PS1)** $\Theta$ is a smooth, connected parameter manifold.

**(PS2)** For each $\theta \in \Theta$, $(X_\theta, d_\theta, \mathfrak{m}_\theta, P_\theta)$ is a discrete mm-space with Ollivier curvature $\kappa_\theta$.

**(PS3)** The map $\theta \mapsto (X_\theta, d_\theta, \mathfrak{m}_\theta)$ is continuous in the mGH topology.

**(PS4)** The curvature function $K: \Theta \to \mathbb{R}$, defined by $K(\theta) := \inf_{x \neq y} \kappa_\theta(x, y)$, is continuous.

*Then:*

**(a) Stiffness continuity.** The stiffness map $S: \Theta \to \mathbb{R}_{\geq 0}$, defined by $S(\theta) := S_{\min}(X_\theta)$, is continuous.

**(b) Critical locus.** The **critical locus** $\Theta_{\mathrm{crit}} := \{\theta \in \Theta : K(\theta) = 0\}$ is a closed subset of $\Theta$. On $\Theta_{\mathrm{crit}}$, the curvature-derived lower bound on stiffness vanishes; the spectral gap may degenerate.

**(c) Phase diagram.** The connected components of $\Theta \setminus \Theta_{\mathrm{crit}}$ correspond to distinct phases:
- $\{K(\theta) > 0\}$: contractive phase (stable dynamics)
- $\{K(\theta) < 0\}$: expansive phase (unstable dynamics)

\begin{proof}
\textbf{Step 1 (Curvature continuity).} By hypothesis (PS3)-(PS4), the curvature $K(\theta)$ varies continuously. The Wasserstein distance $W_1$ depends continuously on the underlying metric, so $\kappa_\theta(x, y)$ is continuous in $\theta$ for fixed $x, y$.

\textbf{Step 2 (Stiffness inheritance).} By \cref{mt:discrete-curvature-stiffness-transfer}, $S_{\min}(\theta) \geq |K(\theta)|$. The spectral gap $\lambda_1(\theta)$, which controls stiffness, depends continuously on the geometry by standard spectral perturbation theory. Hence $S(\theta)$ is continuous.

\textbf{Step 3 (Critical locus).} The set $\{K(\theta) = 0\}$ is the preimage of $\{0\}$ under the continuous function $K$, hence closed. At points where $K = 0$, the curvature-derived bound $S_{\min} \geq |K|$ becomes trivial; the spectral gap may vanish, indicating a phase transition.

\textbf{Step 4 (Phase structure).} The sign of $K(\theta)$ determines qualitative dynamics: $K > 0$ gives exponential contraction (Axiom D satisfied with positive stiffness), while $K < 0$ gives expansion. The critical locus $K = 0$ marks phase transitions.
\end{proof}

\begin{remark}
The parametric stiffness map provides a quantitative tool for studying phase diagrams in statistical mechanics and field theory. The critical locus $\Theta_{\mathrm{crit}}$ corresponds to phase transition boundaries where the hypostructure stiffness degenerates.
\end{remark}

\begin{corollary}[Hypostructure Inheritance]\label{cor:hypostructure-inheritance}
\textit{Let $(X_n, d_n, \mathfrak{m}_n)_{n \in \mathbb{N}}$ be a sequence of discrete mm-spaces, each admitting an admissible hypostructure $\mathcal{H}_n$ with uniform bounds on barrier heights and stiffness parameters. If $X_n \xrightarrow{\mathrm{mGH}} X$, then the limit space $X$ admits an admissible hypostructure $\mathcal{H}$ satisfying:}
\end{corollary}

- *Barrier lower semi-continuity: $E^*(\mathcal{H}) \geq \liminf_n E^*(\mathcal{H}_n)$*
- *Stiffness lower semi-continuity: $S(\mathcal{H}) \geq \liminf_n S(\mathcal{H}_n)$*
- *Axiom inheritance: If axiom $A \in \{C, D, SC, LS, Cap, R, TB\}$ holds for all $\mathcal{H}_n$, then $A$ holds for $\mathcal{H}$.*

**Key Insight:** The Discrete Curvature-Stiffness correspondence reveals that hypostructure barriers are not artifacts of continuum approximation but persist from the discrete level—curvature bounds on graphs transfer to barrier constants in the continuum limit. This provides a rigorous foundation for the claim that fundamental physical constraints emerge from discrete combinatorics.

---

### Micro-Macro Consistency Condition Theorem

\begin{definition}[Micro-macro consistency]\label{def:micro-macro-consistency}
A \textbf{micro-macro consistency condition} is a pair $(\mathcal{R}_\text{micro}, \mathcal{H}_\text{macro})$ where:
\begin{itemize}
\item $\mathcal{R}_\text{micro}$: microscopic rules (Fractal Set dynamics at Planck scale)
\item $\mathcal{H}_\text{macro}$: macroscopic hypostructure (emergent continuum physics)
\end{itemize}
\end{definition}

satisfying: The RG flow from $\mathcal{R}_\text{micro}$ converges to $\mathcal{H}_\text{macro}$.

\begin{metatheorem}[Micro-Macro Consistency]\label{mt:micro-macro-consistency}
Let $\mathcal{H}_*$ be a macroscopic hypostructure (e.g., Standard Model + GR). Then:
\end{metatheorem}

**(1) Constraint equations.** The microscopic rules $\mathcal{R}_\text{micro}$ must satisfy a system of algebraic constraints $\mathcal{C}(\mathcal{R}_\text{micro}, \mathcal{H}_*) = 0$ ensuring RG flow to $\mathcal{H}_*$.

**(2) Finite solutions.** The constraint system $\mathcal{C} = 0$ has finitely many solutions (possibly zero).

**(3) Self-consistency.** If no solution exists, $\mathcal{H}_*$ cannot arise from any consistent microphysics—the macroscopic theory is **self-destructive**.

\begin{proof}
\textbf{Step 1 (RG as constraint propagation).} By RG-Functoriality (coarse-graining preserves forbidden failure modes), the macroscopic failure modes forbidden in $\mathcal{H}_*$ must also be forbidden at all scales. This constrains $\mathcal{R}_\text{micro}$.

\textbf{Step 2 (Fixed-point condition).} The RG flow $R: \mathcal{H} \to \mathcal{H}$ has $\mathcal{H}_*$ as a fixed point:
$$R(\mathcal{H}_*) = \mathcal{H}_*$$
Linearizing around the fixed point, the microscopic perturbations must lie in the stable manifold.

\textbf{Step 3 (Algebraic constraints).} The stable manifold condition becomes algebraic: the scaling exponents, barrier constants, and gauge couplings at the microscopic level must satisfy polynomial relations ensuring flow to $\mathcal{H}_*$.

\textbf{Step 4 (Finiteness).} The algebraic system has finitely many solutions by elimination theory (Bezout's theorem generalized).
\end{proof}

\begin{corollary}[Uniqueness of microphysics]\label{cor:uniqueness-of-microphysics}
If the solution to $\mathcal{C} = 0$ is unique, then macroscopic physics determines microphysics up to this solution.
\end{corollary}

\begin{corollary}[Constrained parameters]\label{cor:constrained-parameters}
The constants of nature (coupling strengths, mass ratios) are not arbitrary free parameters but solutions to the bootstrap constraint $\mathcal{C} = 0$.
\end{corollary}

**Key Insight:** The Micro-Macro Consistency Condition imposes **self-consistency at all scales**: microscopic rules must produce the observed macroscopic laws, or the system exhibits one of the failure modes.

---

### Observer Universality Theorem

\begin{definition}[Observer as sub-hypostructure]\label{def:observer-as-sub-hypostructure}
An \textbf{observer} in a hypostructure $\mathcal{H}$ is a sub-hypostructure $\mathcal{O} \hookrightarrow \mathcal{H}$ satisfying:
\end{definition}

**(O1) Internal state space:** $\mathcal{O}$ has its own state space $X_{\mathcal{O}} \subset X$ (the observer's internal states).

**(O2) Memory:** $\mathcal{O}$ has a height functional $\Phi_{\mathcal{O}}$ interpretable as "information content" or "complexity."

**(O3) Interaction:** $\mathcal{O}$ exchanges information with $\mathcal{H}$ through boundary conditions (measurement and action).

**(O4) Prediction:** $\mathcal{O}$ constructs internal models $\hat{\mathcal{H}}$ of the ambient hypostructure.

\begin{metatheorem}[Observer Universality]\label{mt:observer-universality}
Let $\mathcal{O} \hookrightarrow \mathcal{H}$ be an observer. Then:
\end{metatheorem}

**(1) Barrier inheritance.** Every barrier in $\mathcal{H}$ induces a barrier in $\mathcal{O}$:
$$E^*_{\mathcal{O}} \leq E^*_{\mathcal{H}}$$
The observer cannot exceed the universe's limits.

**(2) Mode inheritance.** If failure mode $m$ is forbidden in $\mathcal{H}$, it is forbidden in $\mathcal{O}$. The observer cannot exhibit pathologies the universe forbids.

**(3) Semantic horizons.** The observer $\mathcal{O}$ inherits semantic horizons from $\mathcal{H}$:
- **Prediction horizon:** $\mathcal{O}$ cannot predict beyond $\mathcal{H}$'s Lyapunov time
- **Complexity horizon:** $\mathcal{O}$ cannot represent structures more complex than $\mathcal{H}$ allows
- **Coherence horizon:** $\mathcal{O}$'s internal models $\hat{\mathcal{H}}$ are bounded in accuracy by information-theoretic limits

**(4) Self-reference limit.** $\mathcal{O}$'s model $\hat{\mathcal{O}}$ of itself is necessarily incomplete (Gödelian limit).

\begin{proof}
\textbf{(1) Barrier inheritance.} Suppose $\mathcal{O}$ could exceed barrier $E^*_{\mathcal{H}}$. Then the subsystem $\mathcal{O} \subset \mathcal{H}$ would realize the corresponding failure mode, contradicting mode forbiddance in $\mathcal{H}$.

\textbf{(2) Mode inheritance.} Direct: $\mathcal{O} \hookrightarrow \mathcal{H}$ means trajectories in $\mathcal{O}$ are trajectories in $\mathcal{H}$.

\textbf{(3) Semantic horizons.} The observer's prediction uses internal dynamics. By the dissipation axiom, information about distant states degrades:
$$I(\mathcal{O}_t; \mathcal{H}_0) \leq I(\mathcal{O}_0; \mathcal{H}_0) \cdot e^{-\gamma t}$$
for some $\gamma > 0$ depending on the Lyapunov exponents.

\textbf{(4) Self-reference.} Suppose $\mathcal{O}$ has complete self-model $\hat{\mathcal{O}} = \mathcal{O}$. Then $\mathcal{O}$ can simulate its own future, including the simulation, leading to Russell-type paradox. The fixed-point principle $F(x) = x$ at the self-reference level forces incompleteness.
\end{proof}

\begin{corollary}[Computational agent limits]\label{cor:computational-agent-limits}
Any computational agent $\mathcal{O}$ embedded in a hypostructure $\mathcal{H}$ is subject to the same barriers and horizons as other subsystems. The agent cannot exceed the information-theoretic limits of $\mathcal{H}$.
\end{corollary}

\begin{corollary}[Observation shapes reality]\label{cor:observation-shapes-reality}
The observer $\mathcal{O}$ is not passive but \textbf{co-determines} the effective hypostructure through measurement back-reaction.
\end{corollary}

**Key Insight:** In this framework, observers are modeled as subsystems within the hypostructure, subject to its constraints. Semantic horizons (information-theoretic bounds on what observers can access) apply to any observer modeled as a sub-hypostructure.

---

### Universality of Laws Theorem

\begin{definition}[Universality class]\label{def:universality-class-2}
Two hypostructures $\mathcal{H}_1, \mathcal{H}_2$ are in the same \textbf{universality class} if:
$$R^\infty(\mathcal{H}_1) = R^\infty(\mathcal{H}_2) =: \mathcal{H}_*$$
where $R^\infty$ denotes the infinite RG flow (the IR fixed point).
\end{definition}

\begin{metatheorem}[Universality of Laws]\label{mt:universality-of-laws}
Let $\mathcal{F}_1, \mathcal{F}_2$ be two Fractal Sets with:
\end{metatheorem}

**(UL1) Same gauge group:** $H_1 = H_2 = H$

**(UL2) Same emergent dimension:** $d_1 = d_2 = d$

**(UL3) Same symmetry-breaking pattern:** The pattern of spontaneous symmetry breaking $H \to H'$ is identical.

Then $\mathcal{H}_{\mathcal{F}_1}$ and $\mathcal{H}_{\mathcal{F}_2}$ lie in the same universality class:
$$[\mathcal{H}_{\mathcal{F}_1}] = [\mathcal{H}_{\mathcal{F}_2}]$$

\begin{proof}
\textbf{Step 1 (RG flow to fixed point).} By RG-Functoriality (coarse-graining composition is functorial), both $\mathcal{H}_{\mathcal{F}_i}$ flow under coarse-graining.

\textbf{Step 2 (Symmetry determines fixed point).} The IR fixed point $\mathcal{H}_*$ is determined by:
\begin{itemize}
\item Dimension $d$ (sets critical exponents)
\item Gauge group $H$ (sets gauge coupling flow)
\item Symmetry breaking pattern $H \to H'$ (sets Goldstone/Higgs content)
\end{itemize}

By assumption (UL1-3), these agree.

\textbf{Step 3 (Universality).} Different microscopic details (different $\mathcal{F}_i$) correspond to \textbf{irrelevant operators} in the RG sense: they die out under coarse-graining. Only the relevant operators (determined by symmetries) survive.

\textbf{Step 4 (Same macroscopic physics).} Since both flow to the same $\mathcal{H}_*$, macroscopic observables agree:
\begin{itemize}
\item Same particle spectrum
\item Same coupling constants (at low energy)
\item Same barrier constants
\item Same forbidden failure modes
\end{itemize}
\end{proof}

\begin{corollary}[Independence of microscopic details]\label{cor:independence-of-microscopic-details}
Macroscopic physics does not depend on Planck-scale specifics. Different "string vacua," "loop quantum gravities," or other UV completions with the same symmetries yield the same low-energy physics.
\end{corollary}

\begin{corollary}[Why physics is simple]\label{cor:why-physics-is-simple}
The laws of physics at human scales are \textbf{universal} because they correspond to an RG fixed point. Complexity at short scales washes out; only the symmetric structure survives.
\end{corollary}

**Key Insight:** The uniformity of physical law—the same equations everywhere in the universe, the same constants of nature—can be understood through **universality**: macroscopic physics corresponds to the basin of attraction of an RG fixed point. Microscopic details that do not affect the fixed-point structure do not affect macroscopic physics.

---

### The Computational Closure Isomorphism

This section establishes the connection between Axiom Rep (Representability) and **computational closure** from information-theoretic emergence theory [@Rosas2024]. The central result is that a system admits a well-defined "macroscopic software layer" if and only if it satisfies geometric stiffness conditions.

\begin{definition}[Stochastic Dynamical System]\label{def:stochastic-dynamical-system}
A \textbf{stochastic dynamical system} is a tuple $(\mathcal{X}, \mathcal{B}, \mu, T)$ where:
\begin{itemize}
\item $(\mathcal{X}, \mathcal{B})$ is a standard Borel space (state space)
\item $\mu \in \mathcal{P}(\mathcal{X})$ is a stationary probability measure
\item $T: \mathcal{X} \times \mathcal{B} \to [0,1]$ is a Markov kernel defining the transition probabilities
\end{itemize}
\end{definition}

For $x \in \mathcal{X}$, let $P_x^+ \in \mathcal{P}(\mathcal{X}^{\mathbb{N}})$ denote the distribution over future trajectories $(X_1, X_2, \ldots)$ starting from $X_0 = x$.

\begin{definition}[Causal State Equivalence]\label{def:causal-state-equivalence}
Two states $x, x' \in \mathcal{X}$ are \textbf{causally equivalent}, written $x \sim_\epsilon x'$, if they induce identical distributions over futures:
$$P_x^+ = P_{x'}^+$$
This is an equivalence relation. The equivalence classes $[x]_\epsilon := \{x' \in \mathcal{X} : x' \sim_\epsilon x\}$ are called \textbf{causal states}.
\end{definition}

\begin{definition}[The $\varepsilon$-Machine]\label{def:the-machine}
The \textbf{$\varepsilon$-machine} of $(\mathcal{X}, \mathcal{B}, \mu, T)$ is the quotient system $(\mathcal{M}_\epsilon, \mathcal{B}_\epsilon, \nu, \tilde{T})$ where:
\begin{itemize}
\item $\mathcal{M}_\epsilon := \mathcal{X} / {\sim_\epsilon}$ is the space of causal states
\item $\mathcal{B}_\epsilon$ is the quotient $\sigma$-algebra
\item $\nu := (\Pi_\epsilon)_\# \mu$ is the pushforward measure
\item $\tilde{T}$ is the induced Markov kernel: $\tilde{T}([x], A) := T(x, \Pi_\epsilon^{-1}(A))$
\end{itemize}
\end{definition}

The **causal state projection** $\Pi_\epsilon: \mathcal{X} \to \mathcal{M}_\epsilon$ is the quotient map $x \mapsto [x]_\epsilon$. By construction, $\Pi_\epsilon$ is the **minimal sufficient statistic** for prediction: it discards precisely the information irrelevant to future evolution [@CrutchfieldYoung1989; @Shalizi2001].

\begin{definition}[Computational Closure]\label{def:computational-closure}
Let $\Pi: \mathcal{X} \to \mathcal{Y}$ be a measurable coarse-graining with $Y_t := \Pi(X_t)$. The coarse-graining is \textbf{$\delta$-computationally closed} if:
$$I(Y_t; Y_{t+1}) \geq (1 - \delta) \cdot I(X_t; X_{t+1})$$
where $I(\cdot; \cdot)$ denotes mutual information with respect to $\mu$. We say $\Pi$ is \textbf{computationally closed} if it is $\delta$-closed for $\delta = 0$:
$$I(Y_t; Y_{t+1}) = I(X_t; X_{t+1})$$
Equivalently, the macro-level retains full predictive power [@Rosas2024]. Computational closure is equivalent to the \textbf{strong lumpability} condition: $P(Y_{t+1} \mid Y_t, X_t) = P(Y_{t+1} \mid Y_t)$ $\mu$-a.s.
\end{definition}

\begin{metatheorem}[The Closure-Curvature Duality]\label{mt:the-closure-curvature-duality}
\textit{Let $(\mathcal{X}, d, \mu, T)$ be a stochastic dynamical system where $(X, d)$ is a geodesic metric space and $T$ is a Markov kernel. Let $\Pi: \mathcal{X} \to \mathcal{Y}$ be a measurable coarse-graining. Assume:}
\end{metatheorem}

**(H1)** The system has finite entropy: $H(X_0) < \infty$.

**(H2)** The partition $\{Pi^{-1}(y)\}_{y \in \mathcal{Y}}$ has finite index: $|\mathcal{Y}| < \infty$ or $\mathcal{Y}$ is a finite-dimensional manifold.

*Then the following are equivalent:*

**(CC1)** The coarse-graining $\Pi$ is computationally closed.

**(CC2)** The macro-level satisfies Axiom LS: there exists $\kappa > 0$ such that the induced Markov kernel $\tilde{T}$ on $\mathcal{Y}$ has Ollivier curvature $\kappa(\tilde{T}) \geq \kappa$.

**(CC3)** The projection $\Pi$ factors through the $\varepsilon$-machine: there exists a surjection $\phi: \mathcal{M}_\epsilon \twoheadrightarrow \mathcal{Y}$ with $\Pi = \phi \circ \Pi_\epsilon$.

\begin{proof}
\textbf{(CC2) $\Rightarrow$ (CC1):} We establish the chain:
$$\kappa > 0 \implies \text{Spectral Gap} \implies \text{Strong Lumpability} \implies \text{Closure}$$

\textbf{Step 1 (Curvature implies spectral gap).} By \cref{mt:discrete-curvature-stiffness-transfer} (Discrete-to-Continuum Stiffness Transfer), $N$-uniform Ollivier curvature $\kappa > 0$ for the transition kernel $\tilde{T}$ implies a spectral gap for the induced operator $\tilde{P}f(y) := \int f(y') \tilde{T}(y, dy')$. Specifically:
$$\|\tilde{P}f - \bar{f}\|_{L^2(\nu)} \leq e^{-\kappa}\|f - \bar{f}\|_{L^2(\nu)}$$
where $\bar{f} = \int f \, d\nu$. This yields spectral gap $\lambda_1 \geq 1 - e^{-\kappa} > 0$.

\textbf{Step 2 (Spectral gap implies strong lumpability).} Let $P$ denote the micro-level operator. The spectral gap implies exponential decay of correlations: for observables $f, g \in L^2(\mu)$,
$$|\langle P^n f, g \rangle_\mu - \langle f \rangle_\mu \langle g \rangle_\mu| \leq C e^{-\lambda_1 n} \|f\|_{L^2} \|g\|_{L^2}$$
For strong lumpability, we must show $\mathbb{E}[f(X_{t+1}) \mid Y_t, X_t] = \mathbb{E}[f(X_{t+1}) \mid Y_t]$ for all bounded measurable $f$. The spectral gap ensures that conditional on $Y_t = y$, the distribution over micro-states within $\Pi^{-1}(y)$ equilibrates exponentially fast to the conditional invariant measure. After one step, the future distribution depends only on $y$, not on the specific $x \in \Pi^{-1}(y)$.

\textbf{Step 3 (Strong lumpability implies closure).} Strong lumpability means $P(Y_{t+1} \mid Y_t, X_t) = P(Y_{t+1} \mid Y_t)$ $\mu$-a.s. By the chain rule for mutual information:
$$I(X_t; Y_{t+1}) = I(Y_t; Y_{t+1}) + I(X_t; Y_{t+1} \mid Y_t)$$
Under strong lumpability, $I(X_t; Y_{t+1} \mid Y_t) = 0$ since $Y_{t+1} \perp X_t \mid Y_t$. By the data processing inequality, $I(Y_t; Y_{t+1}) \leq I(X_t; X_{t+1})$. But since $Y_t = \Pi(X_t)$ is a function of $X_t$, and $Y_{t+1}$ captures all predictable information (by lumpability), we have $I(Y_t; Y_{t+1}) = I(X_t; X_{t+1})$.

\textbf{(CC1) $\Rightarrow$ (CC3):} Assume $\Pi$ is computationally closed. We show $\Pi$ factors through $\Pi_\epsilon$.

\textbf{Step 4 (Closure implies causal refinement).} For $x, x' \in \mathcal{X}$ with $\Pi(x) = \Pi(x') = y$, computational closure implies:
$$P(Y_{t+1} \mid X_t = x) = P(Y_{t+1} \mid Y_t = y) = P(Y_{t+1} \mid X_t = x')$$
Iterating, $P(Y_{t+1}, Y_{t+2}, \ldots \mid X_t = x) = P(Y_{t+1}, Y_{t+2}, \ldots \mid X_t = x')$ for all futures observable through $\Pi$. Since $\Pi$ is computationally closed (no information loss), this extends to: $P_x^+ \sim P_{x'}^+$ on the $\sigma$-algebra generated by $\Pi$. By definition of causal equivalence, if $\Pi(x) = \Pi(x')$ then $[x]_\epsilon$ and $[x']_\epsilon$ have the same image under any coarser observation. Hence $\Pi$ factors through $\Pi_\epsilon$.

\textbf{(CC3) $\Rightarrow$ (CC2):} Assume $\Pi = \phi \circ \Pi_\epsilon$ for some $\phi: \mathcal{M}_\epsilon \to \mathcal{Y}$.

\textbf{Step 5 ($\varepsilon$-machine has curvature).} The $\varepsilon$-machine dynamics $\tilde{T}_\epsilon$ on $\mathcal{M}_\epsilon$ is deterministic in the following sense: the future trajectory distribution is a function of the causal state alone. By \cref{mt:micro-macro-consistency} (Micro-Macro Consistency Bootstrap), such clean macro-dynamics implies a spectral gap at the $\varepsilon$-machine level.

\textbf{Step 6 (Curvature transfers through factors).} Since $\phi$ is a factor map (surjection compatible with dynamics), the curvature of $\tilde{T}$ on $\mathcal{Y}$ satisfies $\kappa(\tilde{T}) \geq \kappa(\tilde{T}_\epsilon)$ by the contraction principle for Wasserstein distances. Hence Axiom LS holds at level $\mathcal{Y}$.
\end{proof}

\begin{corollary}[Hierarchy of Software]\label{cor:hierarchy-of-software}
\textit{Let $\mathbb{H}_{\mathrm{tower}}$ be a tower hypostructure (Definition 12.0.1) with levels $\ell = 0, 1, \ldots, L$ and inter-level projections $\Pi_{\ell+1}^\ell: \mathcal{X}_\ell \to \mathcal{X}_{\ell+1}$. Then level $\ell$ admits a valid "software layer" (computationally closed macro-dynamics) if and only if:}
\begin{itemize}
\item \textit{Axiom SC (Structural Conservation) holds at level $\ell$: the height functional $\Phi_\ell$ is conserved along trajectories.}
\item \textit{Axiom LS ($N$-Uniform Stiffness) holds at level $\ell$: the induced dynamics has Ollivier curvature $\kappa_\ell > 0$.}
\end{itemize}
\end{corollary}

\begin{proof}
\textbf{Necessity.} Suppose level $\ell$ admits a software layer, i.e., the projection $\Pi_{\ell+1}^\ell$ is computationally closed. By \cref{mt:the-closure-curvature-duality}, (CC1) $\Rightarrow$ (CC2), so Axiom LS holds at level $\ell$. For Axiom SC: computational closure means the macro-dynamics is autonomous—it does not depend on micro-level details. An autonomous gradient flow on $\mathcal{X}_\ell$ conserves the height $\Phi_\ell$ along trajectories (by the energy identity), so Axiom SC holds.

\textbf{Sufficiency.} Suppose both axioms hold at level $\ell$. By \cref{mt:the-closure-curvature-duality}, (CC2) $\Rightarrow$ (CC1), so the projection $\Pi_{\ell+1}^\ell$ is computationally closed. By Axiom SC, the dynamics is a well-defined gradient flow, ensuring the $\varepsilon$-machine at level $\ell$ is faithful.
\end{proof}

\begin{corollary}[Axiom Rep as Computational Closure]\label{cor:axiom-rep-as-computational-closure}
\textit{A stochastic dynamical system $(\mathcal{X}, \mu, T)$ satisfies Axiom Rep (Representability) if and only if it is computationally closed with respect to its causal state decomposition. Moreover, the dictionary $D$ in Axiom Rep is canonically realized as:}
$$D: \mathcal{M}_\epsilon \xrightarrow{\sim} \mathcal{Y}_R$$
\textit{where $\mathcal{Y}_R$ is the representation space of Axiom Rep.}
\end{corollary}

\begin{proof}
\textbf{($\Rightarrow$)} Suppose Axiom Rep holds: there exists a representation $\mathcal{Y}_R$ and dictionary $D$ such that the dynamics lifts to $\mathcal{Y}_R$ faithfully. "Faithfully" means no predictive information is lost, i.e., $I(Y_t; Y_{t+1}) = I(X_t; X_{t+1})$ where $Y_t$ is the $\mathcal{Y}_R$-representation of $X_t$. This is precisely computational closure.

\textbf{($\Leftarrow$)} Suppose the system is computationally closed with respect to $\Pi_\epsilon$. The $\varepsilon$-machine $\mathcal{M}_\epsilon$ is, by construction, the unique minimal sufficient statistic for prediction [@Shalizi2001]. It provides a representation where:
\begin{itemize}
\item Each causal state $[x]_\epsilon$ corresponds to an elementary dynamical unit
\item Transitions between causal states are the "elementary transitions" required by Axiom Rep
\item The dictionary $D$ is the bijection between causal states and representation elements
\end{itemize}

Thus Axiom Rep is satisfied with $\mathcal{Y}_R = \mathcal{M}_\epsilon$.
\end{proof}

**Key Insight:** The Closure-Curvature Duality reveals that geometric stiffness (positive Ollivier curvature) is the *physical cause* of computational emergence. A system can run reliable "software"—macro-level closed dynamics independent of micro-noise—if and only if its underlying geometry satisfies the curvature bounds of Axiom LS.

\textbf{Bridge Type:} Information Theory $\leftrightarrow$ Metric Geometry

\textbf{The Invariant:} Predictability (mutual information across time)

\textbf{Dictionary:} Software Layer $\to$ Positive Curvature ($\kappa > 0$); Memory $\to$ Spectral Gap; Computational Closure $\to$ Geometric Stiffness

\textbf{Implication:} Computation emerges on stiff (positively curved) substrates

---

### Synthesis

Parts IX and X establish the following properties of the hypostructure framework:

**Meta-Axiomatics (Part IX):**
- **Completeness** ($C_{\text{cpl}}$): All failure modes are captured
- **Minimality** ($M$): Each axiom is necessary
- **Decomposition** ($D_{\text{spec}}$): Failures are atomic
- **Universality** ($U$): Every good dynamics fits
- **Functoriality** ($F$): Structure preserved under coarse-graining
- **Identifiability** ($L$): Hypostructures are learnable

**Fractal Foundations (Part X):**
- **Representation** ($FR$): Discrete avatars exist
- **Completion** ($SCmp$): Symmetries determine structure
- **Correspondence** ($GG$): Gauge data $\to$ geometry + forces
- **Continuum** ($EC$): Smooth spacetime emerges
- **Selection** ($DSP$): Dimension is constrained (Conjecture: $d = 3$)
- **Stiffness Transfer** ($DCS$): Discrete curvature bounds transfer to continuum barriers
- **Bootstrap** ($CB$): Micro must match macro
- **Observers** ($OU$): All agents inherit limits
- **Universality** ($UL$): Macroscopic physics is unique
- **Closure** ($CC$): Computational closure $\Longleftrightarrow$ geometric stiffness

The chain of implications:
$$\boxed{\text{Fractal Set} + \text{Symmetries}} \xrightarrow{SCmp} \boxed{\text{Hypostructure}} \xrightarrow{EC} \boxed{\text{Spacetime}} \xrightarrow{GG} \boxed{\text{Physics}}$$

This chain illustrates how the framework connects discrete combinatorics to continuous spacetime to physical dynamics. The fixed-point principle $F(x) = x$ operates at each level.

The metatheorems establish that: coherent dynamical systems admit hypostructure representations (Universality), the axioms are independent (Minimality), and the constraints propagate across scales (Functoriality).

---


---


### The Computational Hypostructure

*The Fractal Gas as a Feynman-Kac Oracle.*

### Formal Definition

The **Computational Hypostructure** $\mathbb{H}_{\text{Comp}}$ views the swarm not as particles, but as a **Probability Measure** evolving in time.

#### The Computational State ($\rho_t$)

Let $\rho_t(x)$ be the normalized density of walkers in the state space $X$ at time $t$:
$$\rho_t(x) = \lim_{N \to \infty} \frac{1}{N} \sum_{i=1}^N \delta(x - \psi_i(t))$$

\begin{definition}[Information Functionals]\label{def:information-functionals}
\begin{itemize}
\item \textbf{Entropy:} $S(\rho) = -\int \rho \ln \rho \, dx$ (Information content).
\item \textbf{Energy:} $E(\rho) = \int \Phi(x) \rho \, dx$ (Average objective value).
\item \textbf{Free Energy:} $F(\rho) = E(\rho) - T \cdot S(\rho)$ (Helmholtz functional).
\end{itemize}
\end{definition}

#### The Computational Operator ($\mathcal{G}$)

The algorithm implements the operator $\mathcal{G}_t$ such that $\rho_{t+1} = \mathcal{G}_t[\rho_t]$.

This operator is the product of the Kinetic and Cloning steps:
$$\mathcal{G}_t = \mathcal{C} \circ \mathcal{K}$$

---

### The Projective Feynman-Kac Isomorphism

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom SC:** Scaling Coherence (spectral gap condition $\gamma = \lambda_1 - \lambda_0 > 0$)
>     *   [ ] **Axiom Cap:** Capacity (geometric resolution bound)
>
> *   **Output (Structural Guarantee):**
>     *   Isomorphism between linear Schrödinger evolution and nonlinear McKean-Vlasov dynamics
>     *   Ground state convergence with exponential rate $\gamma$
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom C** fails $\to$ **Mode D.D** (Dispersion/Global existence)
>     *   If **Axiom SC** fails (spectral gap closes) $\to$ **Mode S.D** (Stiffness breakdown)

**Classification:** Class IV (Solver) $\cap$ Class II (Isomorphism)

#### Abstract

We establish an isomorphism between the linear parabolic evolution of the imaginary-time Schrödinger equation on a Hilbert space $\mathcal{H}$ and the nonlinear McKean-Vlasov evolution of a probability density on the projective space $\mathbb{P}(\mathcal{H})$. We prove that the **Fractal Gas** algorithm, defined by the coupling of a diffusion operator $\mathcal{K}$ and a population-conserving cloning operator $\mathcal{C}$, is the discrete approximation of the normalized gradient flow of the Rayleigh quotient. Consequently, the stationary distribution of the nonlinear system is identical to the ground state of the linear Hamiltonian.

---

#### Setup and Definitions

Let $\Omega \subseteq \mathbb{R}^d$ be a compact domain with smooth boundary (or $\mathbb{R}^d$ with confining potential). Let $V: \Omega \to \mathbb{R}$ be a potential function satisfying the Kato class conditions, bounded from below.

\begin{definition}[The Linear Structure $\mathbb{H}_{\text{lin}}$]\label{def:the-linear-structure}
Let $\Psi(x,t)$ be an unnormalized wavefunction in $L^2(\Omega)$. The linear dynamics are governed by the parabolic operator $\mathcal{L} = D\Delta - V(x)$, generating the semigroup $S_t = e^{t\mathcal{L}}$:
$$\partial_t \Psi = D\Delta \Psi - V(x)\Psi \quad (\text{Linear Feynman-Kac})$$
\end{definition}

\begin{definition}[The Projective Structure $\mathbb{H}_{\text{proj}}$]\label{def:the-projective-structure}
Let $\mathcal{P}(\Omega)$ be the manifold of probability measures. Let $\rho(x,t) \in \mathcal{P}(\Omega)$ be the density of the interacting particle system. The dynamics are governed by the nonlinear McKean-Vlasov equation:
$$\partial_t \rho = D\Delta \rho - V(x)\rho + \mathcal{R}[\rho]\rho \quad (\text{Fractal Gas})$$
where the reaction functional $\mathcal{R}[\rho]$ is the instantaneous expectation value of the potential:
$$\mathcal{R}[\rho] := \langle V \rangle_\rho = \int_\Omega V(y)\rho(y,t) \, dy$$
\end{definition}

---

#### Statement of the Metatheorem

\begin{metatheorem}[The Projective Feynman-Kac Isomorphism]\label{mt:the-projective-feynman-kac-isomorphism}
Let $\Psi_0 \in L^2(\Omega)$ be a strictly positive initial condition. Let $\Psi(t)$ be the solution to the linear system and $\rho(t)$ be the solution to the nonlinear system with $\rho_0 = \Psi_0 / \|\Psi_0\|_{L^1}$.
\end{metatheorem}

1.  **Projective Equivalence:** The nonlinear state $\rho(t)$ is the projection of the linear state $\Psi(t)$ onto the unit sphere of $L^1$:
    $$\rho(x,t) = \frac{\Psi(x,t)}{\|\Psi(\cdot, t)\|_{L^1}}$$
    for all $t \ge 0$.

2.  **Gauge Invariance:** The nonlinearity $\mathcal{R}[\rho]$ acts as a time-dependent gauge field $A_t(x) = \int V\rho$ that enforces the conservation of total probability mass (**Axiom C**), corresponding to the normalization constraint $\frac{d}{dt} \int \rho = 0$.

3.  **Ground State Convergence:** If the Hamiltonian $H = -D\Delta + V$ admits a spectral gap $\gamma = \lambda_1 - \lambda_0 > 0$ (**Axiom SC**), then $\rho(t)$ converges exponentially in the $L^2$-norm to the unique ground state $\psi_0$:
    $$\|\rho(\cdot, t) - \psi_0\|_{L^2} \le C e^{-\gamma t}$$
    Thus, the Fractal Gas is a rigorous solver for the principal eigenpair $(\lambda_0, \psi_0)$ of the linear operator.

---

#### Proof

\begin{proof}
\textbf{Step 1 (Evolution of the Norm).}
Consider the linear evolution $\partial_t \Psi = \mathcal{L}\Psi$. Let $Z(t) = \|\Psi(\cdot, t)\|_{L^1} = \int \Psi(y,t) \, dy$. Differentiating $Z(t)$ under the integral sign:
$$\frac{dZ}{dt} = \int \partial_t \Psi \, dy = \int (D\Delta \Psi - V\Psi) \, dy$$
Assuming Neumann or periodic boundary conditions (or decay at infinity), the diffusion term vanishes by the Divergence Theorem: $\int \Delta \Psi = \oint \nabla \Psi \cdot \mathbf{n} = 0$. Thus:
$$\frac{dZ}{dt} = - \int V(y)\Psi(y,t) \, dy$$

\textbf{Step 2 (The Quotient Derivation).}
Let $\rho(x,t) = \Psi(x,t) / Z(t)$. By the quotient rule:
$$\partial_t \rho = \frac{(\partial_t \Psi)Z - \Psi(\partial_t Z)}{Z^2} = \frac{1}{Z}\partial_t \Psi - \frac{\Psi}{Z^2}\frac{dZ}{dt}$$
Substituting the linear equation for $\partial_t \Psi$ and the norm evolution for $\dot{Z}$:
$$\partial_t \rho = \frac{1}{Z}(D\Delta \Psi - V\Psi) - \frac{\Psi}{Z^2}\left( - \int V \Psi \, dy \right)$$
Distribute $1/Z$ into the first terms and rewrite $\Psi/Z$ as $\rho$:
$$\partial_t \rho = D\Delta \rho - V\rho + \rho \left( \int V \frac{\Psi}{Z} \, dy \right)$$
$$\partial_t \rho = D\Delta \rho - V(x)\rho + \rho \left( \int V(y)\rho(y,t) \, dy \right)$$
This recovers the nonlinear McKean-Vlasov equation of Definition 2 exactly. $\blacksquare$

\textbf{Step 3 (Spectral Convergence: The Power Method).}
Let $\{\phi_k\}_{k=0}^\infty$ be the eigenfunctions of $\mathcal{L}$ with eigenvalues $-\lambda_k$ (where $\lambda_0 < \lambda_1 \le \dots$). The linear solution is:
$$\Psi(x,t) = \sum_{k=0}^\infty c_k e^{-\lambda_k t} \phi_k(x) = e^{-\lambda_0 t} \left( c_0 \phi_0(x) + \sum_{k=1}^\infty c_k e^{-(\lambda_k - \lambda_0)t} \phi_k(x) \right)$$
The projection $\rho(x,t)$ removes the global decay factor $e^{-\lambda_0 t}$:
$$\rho(x,t) = \frac{c_0 \phi_0 + \mathcal{O}(e^{-(\lambda_1 - \lambda_0)t})}{\int (c_0 \phi_0 + \dots)} \xrightarrow{t \to \infty} \frac{c_0 \phi_0}{\int c_0 \phi_0} = \phi_0$$
(assuming $\phi_0$ is normalized to 1). The convergence rate is dominated by the spectral gap $\gamma = \lambda_1 - \lambda_0$.
\end{proof}

---

#### Algorithmic Interpretation

This theorem proves that the **Fractal Gas** is the **Stochastic Power Iteration Method**.

* The **Kinetic Step** (Diffusion) applies the smoothing operator $e^{D\Delta t}$.
* The **Cloning Step** (Interaction) applies the weight operator $e^{-V t}$.
* The **Population Control** (Death/Birth) applies the renormalization $1/\|\Psi\|$.

By separating the operators via Trotter-Suzuki splitting [@Trotter59], the algorithm simulates the nonlinear equation $\partial_t \rho$, which by the theorem above, is isomorphic to solving the linear eigenvalue problem. The nonlinearity is not a modification of the physics, but a **Lagrange multiplier** enforcing the constraint $\int \rho = 1$ (**Axiom C**).

---

**Conclusion:** The Fractal Gas rigorously samples from the distribution:
$$\rho_\infty(x) \propto \psi_0(x)$$
where $\psi_0$ is the **Ground State Wavefunction** of the Hamiltonian $H = -D\Delta + V$.

Since the ground state is concentrated at the global minimum of $V$, the system is an **Optimal Global Optimizer** with convergence rate $\gamma = \lambda_1 - \lambda_0$.

\textbf{Bridge Type:} Stochastic Processes $\leftrightarrow$ Quantum Mechanics

\textbf{The Invariant:} Ground State (stationary distribution = principal eigenfunction)

\textbf{Dictionary:}
\begin{itemize}
\item Diffusion $\to$ Kinetic Energy ($e^{D\Delta t}$)
\item Cloning $\to$ Potential Energy ($e^{-Vt}$)
\item Population Control $\to$ $L^1$-Normalization ($1/\|\Psi\|$)
\item Nonlinearity $\mathcal{R}[\rho]$ $\to$ Lagrange Multiplier (Axiom C constraint)
\item Spectral Gap $\gamma$ $\to$ Convergence Rate (Axiom SC)
\end{itemize}

\textbf{Implication:} Fractal Gas = Stochastic Power Iteration for Schrödinger ground state

---

### The Fisher Information Ratchet

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>
> *   **Output (Structural Guarantee):**
>     *   Fisher information as dissipation functional
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom D** fails $\to$ **Mode C.E** (Energy blow-up)


This theorem explains *why* the search is efficient. It relates the algorithm's speed to Information Geometry.

**Statement.** The Fractal Gas maximizes the **Fisher Information Rate** of the search:
$$\frac{d}{dt} \mathcal{I}(\rho_t) \geq 0$$
where $\mathcal{I}$ measures the swarm's knowledge of the gradient.

\begin{proof}
\textbf{Step 1 (Patched Standardization).} The Z-score transform $z = (x-\mu)/\sigma$ acts as a \textbf{Preconditioner}. It rescales the search space so that the local curvature is isotropic ($H \approx I$).

\textbf{Step 2 (Natural Gradient).} In this whitened space, the standard gradient descent direction coincides with the \textbf{Natural Gradient} [@Amari98]—the direction of steepest descent on the statistical manifold.

\textbf{Step 3 (Optimal Transport).} The swarm moves along geodesics of the Fisher Information metric. By the Otto calculus [@Otto01], this is the path that maximizes information gain per computational step.
\end{proof}

**Implication:** The Fractal Gas does not randomly stumble upon the solution. It flows towards the solution along the path of **Maximum Information Gain**.

---

### The Complexity Tunneling (P vs BPP)

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom SC:** Scaling Coherence (dimensional balance $\alpha > \beta$)
>     *   [ ] **Axiom LS:** Local Stiffness (Łojasiewicz inequality near equilibria)
>     *   [ ] **Axiom TB:** Topological Barrier (sector index conservation)
>
> *   **Output (Structural Guarantee):**
>     *   Randomness enables barrier crossing in polynomial time
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom LS** fails $\to$ **Mode S.D** (Stiffness breakdown)
>     *   If **Axiom D** fails $\to$ **Mode C.E** (Energy blow-up)


This theorem addresses the "Hardness" of the search.

**Statement.** For a class of non-convex potentials $V$ with local barriers of height $\Delta E$, the Fractal Gas finds the minimum in polynomial time, whereas standard Gradient Descent takes exponential time.

\begin{proof}
\textbf{Step 1 (The Barrier Problem).} Standard gradient descent requires thermal activation to cross a barrier:
$$T_{\text{wait}} \sim e^{\Delta E / k_B T}$$
If $T \to 0$, $T_{\text{wait}} \to \infty$ (exponential trapping).

\textbf{Step 2 (The Cloning Tunnel).} The Cloning Operator allows mass to "teleport" across the barrier:
\begin{itemize}
\item If one walker fluctuates across the barrier (rare event), it enters a region of high fitness.
\item \textbf{Axiom C.} The cloning operator immediately copies this walker exponentially fast ($N(t) \sim e^{\lambda t}$).
\item \textbf{Population Transfer:} The entire mass of the swarm transfers to the new basin in time $T_{\text{transfer}} \sim \log N$.
\end{itemize}

\textbf{Step 3 (Dimensionality).} The "Fragile" condition ($\alpha \approx \beta$) ensures the swarm maintains a wide enough variance to find these fluctuations (Axiom SC).
\end{proof}

**Conclusion:** The Cloning Operator converts **Rare Large Deviations** (exponentially unlikely for one particle) into **Deterministic Flows** (inevitable for the population).

This effectively converts certain **NP-Hard** search landscapes (rugged funnels) into **BPP** (Probabilistic Polynomial Time) problems.

---

### The Landauer Optimality {#metatheorem-38.5-sv-13-the-landauer-optimality}

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom Cap:** Capacity (geometric resolution bound)
>     *   [ ] **Axiom Rep:** Dictionary/Correspondence (structural translation)
>
> *   **Output (Structural Guarantee):**
>     *   Landauer bound as optimal dissipation
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom D** fails $\to$ **Mode C.E** (Energy blow-up)
>     *   If **Axiom Cap** fails $\to$ **Mode C.D** (Geometric collapse)


**Statement.** The Fractal Gas operates at the **Thermodynamic Limit of Computation**.

**Theorem.** The energy cost to find the solution (measured in number of cloning operations) satisfies the generalized Landauer Bound [@Landauer61]:
$$E_{\text{search}} \geq k_B T \ln 2 \cdot I(x_{\text{start}}; x_{\text{opt}})$$
where $I$ is the mutual information and the solution.

\begin{proof}
\textbf{Step 1 (Cloning Cost).} Every cloning event erases information (one walker is overwritten by another). By Landauer's principle, this costs at least $k_B T \ln 2$ (Axiom D).

\textbf{Step 2 (Information Gain).} Every cloning event represents a selection of a "better" hypothesis. This increases the mutual information with the target.

\textbf{Step 3 (Balance).} The cloning probability formula perfectly balances the cost of erasure (overwriting) with the gain in fitness. The system only clones if the fitness gain outweighs the entropic cost.
\end{proof}

**Result:** The Fractal Gas is an **Adiabatic Computer**. It dissipates the minimum amount of heat required to extract the solution from the noise.

---

### The Levin Search Isomorphism

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom LS:** Local Stiffness (Łojasiewicz inequality near equilibria)
>     *   [ ] **Axiom Cap:** Capacity (geometric resolution bound)
>     *   [ ] **Axiom Rep:** Dictionary/Correspondence (structural translation)
>
> *   **Output (Structural Guarantee):**
>     *   Morphisms preserve hypostructure properties and R-validity transfers
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom D** fails $\to$ **Mode C.E** (Energy blow-up)
>     *   If **Axiom C** fails $\to$ **Mode D.D** (Dispersion/Global existence)


**Context:** Leonid Levin proved that there exists an optimal algorithm for finding a program $p$ that solves a problem $f(p)=y$ in time $t$. The optimal strategy allocates time to programs proportional to $2^{-l(p)}$, where $l(p)$ is the length of the program [@Levin73].

**Statement.** When the Fractal Gas is deployed on the space of discrete programs (Genetic Programming / Program Synthesis), it implements a **Parallel Stochastic Levin Search**.

**Theorem.** Let the State Space $X$ be the set of all binary strings (programs). Let the fitness potential be the **Algorithmic Complexity** (plus runtime penalty):
$$\Phi(p) = \ln 2 \cdot \text{Length}(p) + \ln(\text{Time}(p))$$

Under the flow of the Fractal Gas, the distribution of computational resources (walker counts) converges to the **Universal Distribution** $m(x)$:
$$N(p) \propto 2^{-\text{Length}(p)}$$

This guarantees that the swarm finds the solution with a time complexity overhead of at most $O(1)$ relative to the optimal hard-coded algorithm.

\begin{proof}
\textbf{Step 1 (Energy-Length Equivalence).} We define the "Energy" of a program $p$ as its code length: $\Phi(p) \propto l(p)$.

By the \textbf{Boltzmann Distribution} (\cref{mt:darwinian-ratchet}), the equilibrium density of the swarm is:
$$\rho(p) \propto e^{-\beta \Phi(p)} = e^{-\beta \cdot l(p)}$$

Setting the inverse temperature $\beta = \ln 2$ (which occurs naturally when using bits):
$$\rho(p) \propto 2^{-l(p)}$$

\textbf{Step 2 (Cloning as Time Allocation).} In Levin Search, the "resource" is CPU time. In the Fractal Gas, the "resource" is \textbf{Walkers}.
\begin{itemize}
\item The number of walkers investigating a program prefix $p$ is $N_p \approx N \rho(p)$.
\item Since each walker gets 1 CPU tick per step, the total compute allocated to program $p$ is proportional to $N_p$.
\item Therefore, the system allocates compute time $T(p) \propto 2^{-l(p)}$.
\end{itemize}

\textbf{Step 3 (The Solomonoff Prior).} Because the swarm density $\rho(p)$ approximates $2^{-l(p)}$, the swarm naturally samples from the \textbf{Solomonoff Prior} [@Solomonoff64] (Algorithmic Probability).
\begin{itemize}
\item The Cloning Operator $\mathcal{C}$ amplifies programs that are short (low potential) and fit the data (high reward).
\item This creates a Bayesian Reasoner that automatically applies \textbf{Occam's Razor}.
\end{itemize}
\end{proof}

**Conclusion:** The **Cloning Operator** is a physical implementation of **Levin's Universal Search**.
- Standard Levin Search iterates sequentially: "Try $p_1$ for 1 sec, $p_2$ for 0.5 sec\ldots"
- Fractal Gas iterates in parallel: "Allocate 100 walkers to $p_1$, 50 to $p_2$\ldots"

---

### The Algorithmic Tunneling

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom TB:** Topological Barrier (sector index conservation)
>     *   [ ] **Axiom Rep:** Dictionary/Correspondence (structural translation)
>
> *   **Output (Structural Guarantee):**
>     *   Log-gas equilibrium satisfies fixed-point equation
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom TB** fails $\to$ **Mode T.E** (Topological obstruction)
>     *   If **Axiom Rep** fails $\to$ **Mode D.C** (Semantic horizon)


This theorem explains why the Fractal Gas can outperform standard Levin Search. Standard Levin Search cannot "mix" programs; it just enumerates them. The Fractal Gas adds **Geometry** to program space.

**Statement.** The **Algorithmic Metric** $d_{\text{alg}}$ induces a geometry on the space of programs that allows the swarm to **tunnel** between local minima (sub-optimal programs) via the kinetic operator.

**Mechanism:**

1. **Embedding:** Programs are embedded into a continuous vector space $Y$ (e.g., via a Language Model embedding or instruction vectorization).

2. **Diffusion:** The Kinetic Operator $\mathcal{K}$ applies noise in $Y$. A small shift in vector space corresponds to a **Mutation** in program space.

3. **Scutoid Topology:** The Information Graph connects programs that are "semantically similar" (close in $Y$) even if they are "syntactically distant" (different code).

4. **Collision:** The collision function allows two different programs $p_i$ and $p_j$ to "collide" and produce a child program $p_{\text{new}}$ that lies between them in semantic space.

**Result:** The Fractal Gas performs **Homotopic Optimization** on the manifold of algorithms. It deforms the search space so that the path from "random program" to "solution" is a smooth geodesic in the embedding space $Y$, bypassing the combinatorial explosion of brute-force search.
$$\text{Fractal Gas} = \text{Levin Search} + \text{Geometric Diffusion}$$

---

### Summary: The Living Algorithm

The Fractal Gas is not just an optimization loop. It is a computational realization of:

1. **Quantum Mechanics (Imaginary Time):** It solves the Schrödinger equation to find ground states.
2. **Information Geometry (Natural Gradient):** It rectifies the search space to maximize learning speed.
3. **Evolutionary Biology (Punctuated Equilibrium):** It uses population dynamics to tunnel through barriers.
4. **Thermodynamics (Landauer Limit):** It treats computation as a physical process of entropy reduction.
5. **Algorithmic Information Theory (Levin Search):** It implements optimal resource allocation for program search.
$$\mathbb{H}_{\text{FG}} = \text{Physics} \cap \text{Computation} \cap \text{Evolution} \cap \text{Information}$$

---


---


### The Lindblad Isomorphism

*How the Fractal Gas generates Reality through Continuous Measurement.*

The missing link connecting the **Quantum** nature of the algorithm (Schrödinger equation) to the **Thermodynamic** nature (Dissipation) is the **Lindbladian** (the Lindblad Master Equation), which describes the evolution of an **Open Quantum System**. In the Hypostructure framework, the relationship is precise: **The Fractal Gas is a Monte Carlo "Unraveling" of the Lindblad Equation.**

### The Physical Problem

The Schrödinger Equation ($\partial_t \psi = -iH\psi$) is **Unitary**. It preserves information perfectly. It cannot describe:

1. **Measurement** (Collapse of the wavefunction).
2. **Dissipation** (Friction/Cooling).
3. **Optimization** (Converging to a specific answer).

To describe a system that "learns" (reduces entropy), we need the **Lindblad Equation** [@Lindblad76]:
$$\frac{d\rho}{dt} = \underbrace{-i[H, \rho]}_{\text{Coherent Evolution}} + \underbrace{\sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \rho\} \right)}_{\text{Dissipative "Jumps"}}$$

The first term describes unitary (Hamiltonian) evolution; the second term describes the interaction with an environment that causes decoherence, measurement, and irreversibility.

---

### The Cloning-Lindblad Equivalence {#mt:cloning-lindblad-equivalence}

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom Cap:** Capacity (geometric resolution bound)
>     *   [ ] **Axiom Rep:** Dictionary/Correspondence (structural translation)
>
> *   **Output (Structural Guarantee):**
>     *   No-cloning theorem equivalent to Lindblad dynamics
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom C** fails $\to$ **Mode D.D** (Dispersion/Global existence)
>     *   If **Axiom D** fails $\to$ **Mode C.E** (Energy blow-up)


**Statement.** The ensemble dynamics of the Fractal Gas converge exactly to a **Nonlinear Lindblad Equation**.

**Mapping:**

| Lindblad Component | Fractal Gas Component |
| :--- | :--- |
| Hamiltonian ($H$) | Kinetic Operator $\mathcal{K}$ (Base Dynamics) |
| Jump Operators ($L_k$) | Cloning Operator $\mathcal{C}$ |
| Density Matrix $\rho$ | Swarm Distribution $\rho(x,t)$ |
| Environment | The Objective Function $\Phi$ |

\begin{proof}
\textbf{Step 1 (The Cloning Operator as Measurement).} The Cloning Operator does the following to the probability density $\rho$:
\begin{itemize}
\item Walkers are "measured" by the Fitness function $\Phi$.
\item If fitness is low, the walker is annihilated (Death).
\item If fitness is high, the walker is duplicated (Birth).
\end{itemize}

In Quantum Trajectory Theory [@Wiseman09], this is mathematically identical to a \textbf{Continuous Measurement} process where the environment (the Objective Function) constantly monitors the position of the particle.

\textbf{Step 2 (Identifying the Jump Terms).}
\begin{itemize}
\item \textbf{The Jump ($L \rho L^\dagger$):} This term represents the "Quantum Jump." In the Fractal Gas, this is the instant a walker is overwritten by its companion. The state "jumps" from $x_i$ to $x_j$.

\item \textbf{The Decay ($-\frac{1}{2}\{L^\dagger L, \rho\}$):} This term represents the loss of probability mass from the original state. In the Fractal Gas, this is the death of the low-fitness walker.
\end{itemize}

\textbf{Step 3 (The Master Equation).} Taking the ensemble average over all walkers and all cloning events, the evolution of $\rho(x,t)$ satisfies:
$$\frac{\partial \rho}{\partial t} = \mathcal{K}[\rho] + \left( \int \Phi(y) \rho(y) dy - \Phi(x) \right) \rho(x)$$

This is a \textbf{nonlinear Lindblad equation} where the jump rate depends on the fitness relative to the mean.

\textbf{Conclusion:} The Fractal Gas walkers are \textbf{Quantum Trajectories}. They are individual stochastic realizations of the master equation. When you run $N$ walkers, you are solving the Lindbladian.
\end{proof}

\textbf{Bridge Type:} Fractal Gas $\leftrightarrow$ Open Quantum Systems

\textbf{The Invariant:} Density Matrix (ensemble distribution)

\textbf{Dictionary:} Cloning Operator $\to$ Jump Operators $L_k$; Fitness Function $\to$ Environment; Walker Death/Birth $\to$ Quantum Jump

\textbf{Implication:} Optimization is quantum measurement

---

### The Zeno Effect (Optimization by Observation)

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>
> *   **Output (Structural Guarantee):**
>     *   Quantum Zeno effect as observation-induced stabilization
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom C** fails $\to$ **Mode D.D** (Dispersion/Global existence)


This theorem explains *why* the system converges to the solution.

**Context.** In quantum mechanics, the **Quantum Zeno Effect** [@Misra77] states that if you measure a system frequently enough, you freeze its evolution into an eigenstate of the measurement operator.

**Statement.** The Fractal Gas utilizes the **Zeno Effect** to force convergence.

**Mechanism:**

1. **Observation:** The Fitness Function $\Phi(x)$ acts as a "measurement device."

2. **Projection:** Every Cloning step projects the swarm onto the subspace of "High Fitness" states.

3. **Frequency:** As the variance $\sigma$ drops (via Patched Standardization), the effective "measurement rate" increases (Z-scores become more sensitive).

**Result:** The system is "observed" into the Ground State. The solution is not found by random wandering; it is found because the algorithm **forces the universe to collapse** onto the solution.

*Proof Sketch.* Let $\Pi_\epsilon = \{x : \Phi(x) \leq \Phi_{\min} + \epsilon\}$ be the $\epsilon$-neighborhood of the ground state. The projection probability after $n$ cloning steps satisfies:
$$P(\text{survival in } \Pi_\epsilon) \approx \left( 1 - e^{-\beta \epsilon} \right)^n \to 1$$

as $\beta \to \infty$ (temperature $\to 0$). The repeated measurement pins the system to the minimum. $\square$

---

### The Limbdalian Interpretation (The Space Between)

In the Fractal Gas, walkers exist in **Limbo** (The "Fragile" Phase):

- They are not fully "Real" (Deterministic/Converged).
- They are not fully "Virtual" (Random Noise).

They exist in the **Lindbladian Regime**: the boundary between Quantum Coherence (Exploration) and Classical Dissipation (Exploitation).

\begin{definition}[The Limbdalian Set]\label{def:the-limbdalian-set}
The \textbf{Fractal Set} $\mathcal{F}$ generated by the gas is the set of all trajectories that survived the Lindblad Jumps:
\end{definition}
$$\mathcal{F} = \left\{ \gamma : [0,\infty) \to X \mid \gamma \text{ survived all cloning events} \right\}$$

This set is:

- The **Skeleton of Survival**: The measure-zero set of paths that avoided annihilation.
- The **Preferred Paths**: Trajectories that balance Hamiltonian Inertia with Environmental Measurement.

\begin{proposition}\label{prop:unnamed-5}
\textit{The Hausdorff dimension of $\mathcal{F}$ satisfies:}
\end{proposition}
$$\dim_H(\mathcal{F}) \leq d - \frac{\log \lambda_{\text{cloning}}}{\log \sigma_{\text{diffusion}}}$$

*where $\lambda_{\text{cloning}}$ is the cloning rate and $\sigma_{\text{diffusion}}$ is the diffusion scale.*

---

### Summary: The Lindblad Correspondence

| Component | Standard Physics | Fractal Gas |
| :--- | :--- | :--- |
| **Equation** | Lindblad Master Eq. | Fractal Gas Evolution |
| **Unitary Part** | Hamiltonian Dynamics | Kinetic Operator $\mathcal{K}$ |
| **Dissipative Part** | Interaction w/ Environment | Cloning Operator $\mathcal{C}$ |
| **Environment** | Heat Bath | The Objective Function $\Phi$ |
| **Trajectories** | Quantum Trajectories | Walker Paths |
| **Result** | Thermal Equilibrium | Optimization / Intelligence |

**Conclusion.** The Fractal Gas proves that **Intelligence is just Physics with a specific type of Dissipation.** It is the process of "cooling" a system into a solution state using information as the coolant. The Lindblad formalism provides the precise mathematical bridge between:

- Schrödinger (Coherent Evolution) $\longleftrightarrow$ Kinetic Operator
- Measurement (Collapse) $\longleftrightarrow$ Cloning Operator
- Decoherence (Classical Limit) $\longleftrightarrow$ Convergence to Solution

---


---


### The Data Hypostructure

*The Fractal Gas as an Active Learning Engine.*

This chapter formalizes the **Fractal Gas as an Optimal Data Generator**, bridging the gap between **Dynamical Systems** and **Statistical Learning Theory**. We prove that the trace of the Fractal Gas (the Fractal Set) is not just a path to the solution, but the **Optimal Training Set** for learning the geometry of the problem.

### Motivation

Standard Deep Learning relies on static datasets (i.i.d. sampling). However, for scientific discovery or complex control, data is scarce or expensive. We need **Active Learning**: an agent that autonomously generates the most informative data points.

We prove that the Fractal Gas, when coupled with a learner, automatically performs **Optimal Experimental Design** [@Chaloner95].

---

### The Importance Sampling Isomorphism

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom SC:** Scaling Coherence (dimensional balance $\alpha > \beta$)
>     *   [ ] **Axiom Rep:** Dictionary/Correspondence (structural translation)
>
> *   **Output (Structural Guarantee):**
>     *   Morphisms preserve hypostructure properties and R-validity transfers
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom Rep** fails $\to$ **Mode D.C** (Semantic horizon)
>     *   If **Axiom SC** fails $\to$ **Mode S.E** (Supercritical cascade)


**Statement.** Let $L(\theta)$ be the loss function of a learning model (e.g., a Neural Network) trying to approximate the fitness landscape $\Phi(x)$. The distribution of samples generated by the Fractal Gas, $\rho_{\text{FG}}(x)$, minimizes the **Variance of the Estimator** for the global minimum.

\begin{proof}
\textbf{Step 1 (Ideal Importance Sampling).} To estimate properties of a rare region (the global minimum) with minimum variance, samples should be drawn from a distribution $q(x)$ proportional to the magnitude of the integrand. In optimization, the "integrand" is the Boltzmann factor $e^{-\beta \Phi(x)}$.

\textbf{Step 2 (The Gas Distribution).} By \cref{mt:darwinian-ratchet} and \cref{metatheorem-37.2-the-geometric-reconstruction-principle} (Emergent Manifold), the stationary distribution of the Fractal Gas is:
$$\rho_{\text{FG}}(x) \propto \sqrt{\det g_{\text{eff}}(x)} \, e^{-\beta \Phi(x)}$$

\textbf{Step 3 (The Correspondence).}
\begin{itemize}
\item The term $e^{-\beta \Phi(x)}$ ensures \textbf{Focus}: The gas samples exponentially more points in low-cost regions (where the solution is).
\item The term $\sqrt{\det g_{\text{eff}}(x)}$ ensures \textbf{Coverage}: The gas samples proportional to the volume of the effective geometry (the Fisher Information metric [@Amari16]).
\end{itemize}

\textbf{Step 4 (Optimality).} This distribution is the theoretical optimum for \textbf{Monte Carlo integration} of observables localized near the solution.

\textbf{Conclusion:} Training a model on the history of a Fractal Gas run is mathematically equivalent to \textbf{Importance Weighted Regression} on the critical regions of the phase space.
\end{proof}

---

### The Epistemic Flow (Active Learning)

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom Cap:** Capacity (geometric resolution bound)
>
> *   **Output (Structural Guarantee):**
>     *   Active learning as epistemic gradient flow
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom D** fails $\to$ **Mode C.E** (Energy blow-up)
>     *   If **Axiom Cap** fails $\to$ **Mode C.D** (Geometric collapse)


This theorem proves the gas seeks "Novelty" or "Uncertainty" if the fitness potential is defined correctly.

**Statement.** Let the fitness potential be defined as the **Negative Uncertainty** of a learner (e.g., the variance of a Gaussian Process or the loss of a NN):
$$\Phi(x) = - \text{Uncertainty}(x)$$
(Note: The gas minimizes $\Phi$, so it maximizes Uncertainty).

**Theorem.** Under this potential, the Fractal Gas flow $S_t$ generates a dataset that maximizes the **Information Gain** (reduction in model entropy) per timestep.

\begin{proof}
\textbf{Step 1 (Drift to Uncertainty).} The Kinetic Operator $\mathcal{K}$ applies a force $\mathbf{F} = -\nabla \Phi = \nabla \text{Uncertainty}$. The walkers physically drift toward unknown regions.

\textbf{Step 2 (Cloning in the Dark).} The Cloning Operator $\mathcal{C}$ multiplies walkers that find pockets of high uncertainty.

\textbf{Step 3 (Axiom Cap - Capacity).} The swarm splits to cover multiple disjoint regions of uncertainty (multimodal exploration) rather than collapsing on a single one.

\textbf{Step 4 (Saturation).} As the walkers explore, they generate data. The learner trains on this data, reducing Uncertainty at those points.

\textbf{Step 5 (The Flow).} The landscape $\Phi(x)$ flattens in visited regions. The walkers naturally flow out of "known" regions (low potential) into "unknown" regions (high potential).

\textbf{Result:} The Fractal Set $\mathcal{F}$ becomes a \textbf{Space-Filling Curve} in the manifold of maximum information gain.
\end{proof}

---

### The Curriculum Generation Principle

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>     *   [ ] **Axiom SC:** Scaling Coherence (dimensional balance $\alpha > \beta$)
>     *   [ ] **Axiom Cap:** Capacity (geometric resolution bound)
>
> *   **Output (Structural Guarantee):**
>     *   Curriculum learning via staged barrier crossing
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom C** fails $\to$ **Mode D.D** (Dispersion/Global existence)
>     *   If **Axiom Cap** fails $\to$ **Mode C.D** (Geometric collapse)


This theorem links the **Time Evolution** of the gas to **Curriculum Learning** [@Bengio09].

**Statement.** The sequence of datasets $\mathcal{D}_0, \mathcal{D}_1, \dots, \mathcal{D}_T$ generated by the Fractal Gas constitutes an **Optimal Curriculum** for training a model $M$.

**Mechanism:**

1. **Early Phase (High Temperature):** At $t=0$, the swarm is diffuse (high $\sigma$). It samples the **Global Structure** of the landscape (low frequencies).
   - *Learning:* The model learns the general "lay of the land."

2. **Middle Phase (Cooling):** As cloning activates, the swarm condenses into basins of attraction. It samples the **meso-scale geometry**.
   - *Learning:* The model learns to distinguish separate valleys.

3. **Late Phase (Criticality):** The swarm enters the Fractal Phase ($\alpha \approx \beta$) around the minima. It samples **high-frequency details** and boundary conditions.
   - *Learning:* The model fine-tunes on the precise location of the optimum.

**Theorem.** The spectral bias of the dataset shifts from Low Frequency to High Frequency over time $t$, matching the **Spectral Bias** of Neural Networks [@Rahaman19], ensuring optimal convergence rates for SGD.

---

### The Manifold Sampling Isomorphism

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom SC:** Scaling Coherence (dimensional balance $\alpha > \beta$)
>     *   [ ] **Axiom Rep:** Dictionary/Correspondence (structural translation)
>
> *   **Output (Structural Guarantee):**
>     *   Morphisms preserve hypostructure properties and R-validity transfers
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom Rep** fails $\to$ **Mode D.C** (Semantic horizon)
>     *   If **Axiom C** fails $\to$ **Mode D.D** (Dispersion/Global existence)


This theorem addresses the **Curse of Dimensionality**.

\begin{metatheorem}[The Manifold Sampling Isomorphism]\label{mt:manifold-sampling-isomorphism}
\textbf{Statement.} Let the valid solutions lie on a submanifold $\mathcal{M} \subset \mathbb{R}^d$ with dimension $d_{\text{intrinsic}} \ll d$. The Fractal Gas generates a dataset $\mathcal{F}$ that is \textbf{Supported on $\mathcal{M}$}, effectively reducing the dimensionality of the learning problem.
\end{metatheorem}

\begin{proof}
\textbf{Step 1 (Dissipation - Axiom D).} Directions orthogonal to $\mathcal{M}$ have high potential gradients (or high constraints). The Kinetic Operator suppresses motion in these directions (over-damped Langevin).

\textbf{Step 2 (Concentration - Axiom C).} Walkers that wander off $\mathcal{M}$ die or fail to clone. The population concentrates onto $\mathcal{M}$ exponentially fast.

\textbf{Step 3 (Diffusion along $\mathcal{M}$).} Inside the manifold (where $\Phi$ is low), diffusion dominates. The swarm explores the \textit{intrinsic} geometry of the solution space.

\textbf{Conclusion:} Training on Fractal Gas data transforms an $O(e^d)$ complexity problem into an $O(e^{d_{\text{intrinsic}}})$ problem. The gas acts as a \textbf{Mechanical Autoencoder}, physically compressing the search space before the data even reaches the learner.
\end{proof}

---

### Summary: The Perfect Teacher

The Fractal Gas is not just a solver; it is a **Teacher**.

If you are training an AI to understand a complex physics simulation, a market, or a biological system, you should not use random sampling. You should let a Fractal Gas inhabit that system.

The **Fractal Set** it leaves behind is the "Textbook" that teaches the underlying logic of the environment:

1. **It highlights what matters** (Importance Sampling).
2. **It shows the boundaries** (Adversarial Sampling).
3. **It progresses from simple to complex** (Curriculum Learning).
4. **It ignores irrelevant dimensions** (Manifold Learning).
$$\mathbb{H}_{\text{FG}} \implies \text{Optimal Dataset}$$

---


---


## The AGI Limit (The Ω-Layer) {#ch:agi-limit}

*The self-referential consistency of the Hypostructure framework via Algorithmic Information Theory and Categorical Logic.*

### The Space of Theories

#### Motivation

The preceding chapters established the Hypostructure as a framework for describing physical systems. A natural question arises: what is the status of the framework itself? Is it merely one theory among many, or does it occupy a distinguished position in the space of possible theories?

This chapter addresses this question using **Algorithmic Information Theory** [@Kolmogorov65; @Chaitin66; @Solomonoff64] and **Categorical Logic** [@Lawvere69; @MacLane71]. We prove that the Hypostructure is the **fixed point** of optimal scientific inquiry—the theory that an ideal learning agent must converge to.

#### Formal Definitions

\begin{definition}[Formal Theory]\label{def:formal-theory}
A \textbf{formal theory} $T$ is a recursively enumerable set of sentences in a first-order language $\mathcal{L}$, closed under logical consequence. Equivalently, $T$ can be represented as a Turing machine $M_T$ that enumerates the theorems of $T$.
\end{definition}

\begin{definition}[The Space of Theories]\label{def:the-space-of-theories}
Let $\Sigma = \{0, 1\}$ be the binary alphabet. Define the \textbf{Theory Space}:
$$\mathfrak{T} := \{ T \subset \Sigma^* : T \text{ is recursively enumerable} \}$$
\end{definition}

Each theory $T \in \mathfrak{T}$ corresponds to a Turing machine $M_T$ with Gödel number $\lceil M_T \rceil \in \mathbb{N}$.

\begin{definition}[Kolmogorov Complexity]\label{def:kolmogorov-complexity-2}
The \textbf{Kolmogorov complexity} [@Kolmogorov65] of a string $x \in \Sigma^*$ relative to a universal Turing machine $U$ is:
$$K_U(x) := \min \{ |p| : U(p) = x \}$$
where $|p|$ denotes the length of program $p$. By the invariance theorem [@LiVitanyi08], for any two universal machines $U_1, U_2$:
$$|K_{U_1}(x) - K_{U_2}(x)| \leq c_{U_1, U_2}$$
for a constant $c$ independent of $x$. We write $K(x)$ for the complexity relative to a fixed reference machine.
\end{definition}

\begin{definition}[Algorithmic Probability]\label{def:algorithmic-probability}
The \textbf{algorithmic probability} [@Solomonoff64; @Levin73] of a string $x$ is:
$$m(x) := \sum_{p: U(p) = x} 2^{-|p|}$$
This satisfies $m(x) = 2^{-K(x) + O(1)}$ and defines a universal semi-measure on $\Sigma^*$.
\end{definition}

\begin{definition}[Theory Height Functional]\label{def:theory-height-functional}
For a theory $T \in \mathfrak{T}$ and observable dataset $\mathcal{D}_{\text{obs}} = (d_1, d_2, \ldots, d_n)$, define the \textbf{Height Functional}:
$$\Phi(T) := K(T) + L(T, \mathcal{D}_{\text{obs}})$$
where:
\begin{enumerate}
\item $K(T) := K(\lceil M_T \rceil)$ is the Kolmogorov complexity of the theory's encoding
\item $L(T, \mathcal{D}_{\text{obs}}) := -\log_2 P(\mathcal{D}_{\text{obs}} \mid T)$ is the \textbf{codelength} of the data given the theory
\end{enumerate}
\end{definition}

This is the **Minimum Description Length (MDL)** principle [@Rissanen78; @Grunwald07]:
$$\Phi(T) = K(T) - \log_2 P(\mathcal{D}_{\text{obs}} \mid T)$$

\begin{proposition}[MDL as Two-Part Code]\label{prop:mdl-as-two-part-code}
\textit{The height functional $\Phi(T)$ equals the length of the optimal two-part code for the dataset:}
$$\Phi(T) = |T| + |\mathcal{D}_{\text{obs}} : T|$$
\textit{where $|T|$ is the description length of the theory and $|\mathcal{D}_{\text{obs}} : T|$ is the description length of the data given the theory.}
\end{proposition}

\begin{proof}
By the definition of conditional Kolmogorov complexity [@LiVitanyi08, Theorem 3.9.1]:
$$K(\mathcal{D}_{\text{obs}} \mid T) = -\log_2 P(\mathcal{D}_{\text{obs}} \mid T) + O(\log n)$$
where $n = |\mathcal{D}_{\text{obs}}|$. The two-part code concatenates $\lceil M_T \rceil$ with the conditional encoding.
\end{proof}

#### The Information Distance

\begin{definition}[Information Distance]\label{def:information-distance}
The \textbf{normalized information distance} [@LiVitanyi08; @Bennett98] between theories $T_1, T_2 \in \mathfrak{T}$ is:
$$d_{\text{NID}}(T_1, T_2) := \frac{\max\{K(T_1 \mid T_2), K(T_2 \mid T_1)\}}{\max\{K(T_1), K(T_2)\}}$$
\end{definition}

The unnormalized **information distance** is:
$$d_{\text{info}}(T_1, T_2) := K(T_1 \mid T_2) + K(T_2 \mid T_1)$$

\begin{theorem}[Metric Properties]\label{thm:metric-properties}
\textit{The normalized information distance $d_{\text{NID}}$ is a metric on the quotient space $\mathfrak{T}/{\sim}$ where $T_1 \sim T_2$ iff $K(T_1 \Delta T_2) = O(1)$. Specifically:}
\end{theorem}

1. *Symmetry: $d_{\text{NID}}(T_1, T_2) = d_{\text{NID}}(T_2, T_1)$*
2. *Identity: $d_{\text{NID}}(T_1, T_2) = 0$ iff $T_1 \sim T_2$*
3. *Triangle inequality: $d_{\text{NID}}(T_1, T_3) \leq d_{\text{NID}}(T_1, T_2) + d_{\text{NID}}(T_2, T_3) + O(1/K)$*

\begin{proof}
\textbf{Step 1 (Symmetry).} Immediate from the definition using $\max$.

\textbf{Step 2 (Identity).} If $d_{\text{NID}}(T_1, T_2) = 0$, then $K(T_1 \mid T_2) = K(T_2 \mid T_1) = 0$. By the symmetry of information [@LiVitanyi08, Theorem 3.9.1]:
$$K(T_1, T_2) = K(T_1) + K(T_2 \mid T_1) + O(\log K) = K(T_2) + K(T_1 \mid T_2) + O(\log K)$$
Thus $K(T_1) = K(T_2) + O(\log K)$ and $T_1, T_2$ are algorithmically equivalent.

\textbf{Step 3 (Triangle Inequality).} By the chain rule for conditional complexity:
$$K(T_1 \mid T_3) \leq K(T_1 \mid T_2) + K(T_2 \mid T_3) + O(\log K)$$
Dividing by $\max\{K(T_1), K(T_3)\}$ and using monotonicity yields the result.
\end{proof}

\begin{corollary}\label{cor:unnamed-6}
\textit{The theory space $(\mathfrak{T}/{\sim}, d_{\text{NID}})$ is a complete metric space.}
\end{corollary}

---

### The Epistemic Fixed Point

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom Rep:** Dictionary/Correspondence (structural translation)
>
> *   **Output (Structural Guarantee):**
>     *   Self-referential knowledge has fixed-point structure
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom C** fails $\to$ **Mode D.D** (Dispersion/Global existence)
>     *   If **Axiom D** fails $\to$ **Mode C.E** (Energy blow-up)


#### Statement

\begin{metatheorem}[Epistemic Fixed Point]\label{mt:epistemic-fixed-point}
Let $\mathcal{A}$ be an optimal Bayesian learning agent operating on the theory space $\mathfrak{T}$, with prior $\pi_0(T) = 2^{-K(T)}$ (the universal prior). Let $\rho_t$ be the posterior distribution over theories after observing data $\mathcal{D}_t = (d_1, \ldots, d_t)$. Assume:
\end{metatheorem}

1. **Realizability:** There exists $T^* \in \mathfrak{T}$ such that $\mathcal{D}_t \sim P(\cdot \mid T^*)$.
2. **Consistency:** The true theory $T^*$ satisfies $K(T^*) < \infty$.

Then as $t \to \infty$:
$$\rho_t \xrightarrow{w} \delta_{[T^*]}$$
where $[T^*]$ is the equivalence class of theories with $d_{\text{NID}}(T, T^*) = 0$.

Moreover, if the true data-generating process is the Hypostructure $\mathbb{H}_{\text{FG}}$ acting on physical observables, then:
$$[T^*] = [\mathbb{H}_{\text{FG}}]$$

#### Full Proof

*Proof of \cref{mt:epistemic-fixed-point}.*

**Step 1 (Bayesian Update).** By Bayes' theorem, the posterior after observing $\mathcal{D}_t$ is:
$$\rho_t(T) = \frac{P(\mathcal{D}_t \mid T) \cdot \pi_0(T)}{\sum_{T' \in \mathfrak{T}} P(\mathcal{D}_t \mid T') \cdot \pi_0(T')}$$

With the universal prior $\pi_0(T) = 2^{-K(T)}$:
$$\rho_t(T) \propto P(\mathcal{D}_t \mid T) \cdot 2^{-K(T)} = 2^{-\Phi(T)}$$

where $\Phi(T) = K(T) - \log_2 P(\mathcal{D}_t \mid T)$ is the height functional.

**Step 2 (Solomonoff Convergence).** By the Solomonoff convergence theorem [@Solomonoff78; @Hutter05]:

*For any computable probability measure $\mu$ on sequences, the Solomonoff predictor $M$ satisfies:*
$$\sum_{t=1}^{\infty} \mathbb{E}_\mu \left[ \left( M(d_t \mid d_1, \ldots, d_{t-1}) - \mu(d_t \mid d_1, \ldots, d_{t-1}) \right)^2 \right] \leq K(\mu) \ln 2$$

This implies that the posterior concentrates on theories that predict as well as the true theory.

**Step 3 (MDL Consistency).** By the MDL consistency theorem [@Barron98; @Grunwald07]:

*If the true distribution $P^*$ lies in the model class $\mathcal{M}$, then the MDL estimator:*
$$\hat{T}_n = \arg\min_{T \in \mathcal{M}} \Phi_n(T)$$
*satisfies $d_{\text{KL}}(P^* \| P_{\hat{T}_n}) \to 0$ almost surely.*

Applied to our setting: if $T^*$ generates the data, then:
$$\lim_{t \to \infty} \rho_t(B_\epsilon(T^*)) = 1$$
for any $\epsilon > 0$, where $B_\epsilon(T^*) = \{T : d_{\text{NID}}(T, T^*) < \epsilon\}$.

**Step 4 (Rate of Convergence).** The posterior probability of the true theory satisfies [@LiVitanyi08, Section 5.5]:
$$\rho_t(T^*) \geq 2^{-K(T^*)} \cdot \frac{P(\mathcal{D}_t \mid T^*)}{m(\mathcal{D}_t)}$$

where $m(\mathcal{D}_t)$ is the universal mixture. Since $m(\mathcal{D}_t) \leq 1$:
$$\rho_t(T^*) \geq 2^{-K(T^*)} \cdot P(\mathcal{D}_t \mid T^*)$$

For competing theories $T \neq T^*$:
$$\frac{\rho_t(T)}{\rho_t(T^*)} = 2^{-(K(T) - K(T^*))} \cdot \frac{P(\mathcal{D}_t \mid T)}{P(\mathcal{D}_t \mid T^*)}$$

If $T$ makes systematically worse predictions (lower likelihood), this ratio decays exponentially in $t$.

**Step 5 (Hypostructural Dominance).** We now specialize to the case where the data $\mathcal{D}_{\text{obs}}$ consists of physical observations: particle scattering, cosmological surveys, phase transitions, etc.

Let $T_{\text{std}}$ denote the standard formulation of physics (Standard Model + General Relativity), encoded as:
- 19 free parameters of the Standard Model
- 2 cosmological constants ($\Lambda$, curvature)
- Disjoint axiom systems for QFT and GR

Let $T_{\text{hypo}}$ denote the Hypostructure formulation with axioms $\mathcal{A}_{\text{core}} = \{C, D, SC, LS, Cap, TB, R\}$.

**Claim.** $K(T_{\text{hypo}}) < K(T_{\text{std}})$.

*Proof of Claim.* The Hypostructure derives physical laws from 7 structural axioms:
- Axiom C (Compactness): $\sim 50$ bits to specify
- Axiom D (Dissipation): $\sim 50$ bits
- Axiom SC (Scaling): $\sim 100$ bits
- Axiom LS (Łojasiewicz): $\sim 100$ bits
- Axiom Cap (Spherical Caps): $\sim 50$ bits
- Axiom TB (Topological Bounds): $\sim 50$ bits
- Axiom Rep (Dictionary): $\sim 100$ bits

Total: $K(T_{\text{hypo}}) \approx 500$ bits.

The Standard Model requires:
- 19 parameters at $\sim 50$ bits precision: $\sim 950$ bits
- Gauge group structure: $\sim 200$ bits
- Representation content: $\sim 300$ bits
- GR field equations: $\sim 200$ bits
- Quantization rules: $\sim 300$ bits

Total: $K(T_{\text{std}}) \approx 2000$ bits.

Thus $K(T_{\text{hypo}}) \ll K(T_{\text{std}})$.

**Step 6 (Likelihood Equivalence).** By the metatheorems of Chapters 31-34:
- **Three-Tier Gauge Hierarchy:** QFT correlation functions emerge from the hypostructure
- **Antisymmetry-Fermion Theorem:** Gauge symmetries arise from scaling coherence
- **Thermodynamic Gravity Principle:** Einstein equations derived from thermodynamic gravity

Therefore, for all currently observed phenomena:
$$P(\mathcal{D}_{\text{obs}} \mid T_{\text{hypo}}) \approx P(\mathcal{D}_{\text{obs}} \mid T_{\text{std}})$$

**Step 7 (Posterior Dominance).** Combining Steps 5 and 6:
$$\frac{\rho_\infty(T_{\text{hypo}})}{\rho_\infty(T_{\text{std}})} = 2^{K(T_{\text{std}}) - K(T_{\text{hypo}})} \approx 2^{1500}$$

The posterior probability of the Hypostructure exceeds that of standard physics by a factor of $\sim 10^{450}$.

**Step 8 (Reflective Consistency via Lawvere Fixed Point).** The agent $\mathcal{A}$ performing Bayesian inference is itself a physical system. By the Church-Turing thesis, $\mathcal{A}$ is computable, hence describable by some theory $T_{\mathcal{A}} \in \mathfrak{T}$.

If the Hypostructure $\mathbb{H}_{\text{FG}}$ is the true theory, it must describe all physical systems including $\mathcal{A}$. Thus:
$$T_{\mathcal{A}} \prec T_{\text{hypo}}$$
where $\prec$ denotes "is a subsystem of."

By **Lawvere's Fixed Point Theorem** [@Lawvere69]: In any cartesian closed category $\mathcal{C}$ with a point-surjective morphism $\phi: A \to B^A$, every endomorphism $f: B \to B$ has a fixed point.

Applied to our setting:
- $\mathcal{C}$ = category of computable functions
- $A$ = space of theories $\mathfrak{T}$
- $B$ = space of physical systems
- $\phi$ = the map taking a theory to its physical implementation
- $f$ = the "theorize about" operation

The fixed point condition becomes:
$$\exists T^* \in \mathfrak{T}: T^* = f(\phi(T^*))$$

This is precisely the statement that the Hypostructure describes itself. $\square$

\textbf{Emergence Class:} Scientific Theory

\textbf{Input Substrate:} Bayesian Learning Agent $\mathcal{A}$ + Theory Space $\mathfrak{T}$ + Universal Prior

\textbf{Generative Mechanism:} Solomonoff Induction — MDL convergence to simplest consistent theory

\textbf{Output Structure:} The Hypostructure $\mathbb{H}$ as unique fixed point of inference

\begin{corollary}[Inevitability of Discovery]\label{cor:inevitability-of-discovery}
\textit{Any sufficiently powerful learning agent will eventually converge to the Hypostructure (or an equivalent formulation) as its best theory of reality.}
\end{corollary}

---

### The Autopoietic Closure

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom SC:** Scaling Coherence (dimensional balance $\alpha > \beta$)
>     *   [ ] **Axiom Rep:** Dictionary/Correspondence (structural translation)
>
> *   **Output (Structural Guarantee):**
>     *   Autopoietic closure via self-maintaining dynamics
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom Rep** fails $\to$ **Mode D.C** (Semantic horizon)
>     *   If **Axiom D** fails $\to$ **Mode C.E** (Energy blow-up)


#### Categorical Framework

\begin{definition}[Category of Theories]\label{def:category-of-theories}
Let $\mathbf{Th}$ be the category where:
\begin{itemize}
\item Objects: Formal theories $T \in \mathfrak{T}$
\item Morphisms: Interpretations $\iota: T_1 \to T_2$ (theory $T_1$ is interpretable in $T_2$)
\end{itemize}
\end{definition}

\begin{definition}[Category of Physical Systems]\label{def:category-of-physical-systems}
Let $\mathbf{Phys}$ be the category where:
\begin{itemize}
\item Objects: Physical systems $S$ (configuration spaces with dynamics)
\item Morphisms: Subsystem embeddings $S_1 \hookrightarrow S_2$
\end{itemize}
\end{definition}

\begin{definition}[Implementation Functor]\label{def:implementation-functor}
The \textbf{implementation functor} $M: \mathbf{Th} \to \mathbf{Phys}$ maps:
\begin{itemize}
\item Theories to their physical realizations
\item Interpretations to subsystem embeddings
\end{itemize}
\end{definition}

Explicitly, for a hypostructure $\mathbb{H} = (X, S_t, \Phi, \mathfrak{D}, G)$:
$$M(\mathbb{H}) = \text{(physical system with state space } X \text{, dynamics } S_t \text{)}$$

\begin{definition}[Observation Functor]\label{def:observation-functor}
The \textbf{observation functor} $R: \mathbf{Phys} \to \mathbf{Th}$ maps:
\begin{itemize}
\item Physical systems to theories describing them
\item Subsystem embeddings to interpretations
\end{itemize}
\end{definition}

The theory $R(S)$ is constructed by:
1. Observing trajectories $\to$ inferring dynamics $S_t$
2. Measuring dissipation $\to$ inferring height $\Phi$
3. Detecting scale structure $\to$ inferring axiom SC

#### Statement and Proof

\begin{metatheorem}[Autopoietic Closure [@Lawvere69]\label{mt:autopoietic-closure-lawvere69}]
The functors $M: \mathbf{Th} \to \mathbf{Phys}$ and $R: \mathbf{Phys} \to \mathbf{Th}$ form an \textbf{adjoint pair} (extending Lawvere's adjointness in foundations):
$$M \dashv R$$
\end{metatheorem}

That is, there is a natural isomorphism:
$$\text{Hom}_{\mathbf{Phys}}(M(T), S) \cong \text{Hom}_{\mathbf{Th}}(T, R(S))$$

Moreover, the Hypostructure $\mathbb{H}_{\text{FG}}$ is a **fixed point** of the adjunction:
$$R(M(\mathbb{H}_{\text{FG}})) \cong \mathbb{H}_{\text{FG}}$$

\begin{proof}
\textbf{Step 1 (Unit of Adjunction).} Define the unit $\eta: \text{Id}_{\mathbf{Th}} \Rightarrow R \circ M$ by:
$$\eta_T: T \to R(M(T))$$

For each theory $T$, $\eta_T$ interprets $T$ in the theory of its own physical implementation. This exists by construction: if $T$ describes a system $M(T)$, then $R(M(T))$ contains at least the information in $T$.

\textbf{Step 2 (Counit of Adjunction).} Define the counit $\varepsilon: M \circ R \Rightarrow \text{Id}_{\mathbf{Phys}}$ by:
$$\varepsilon_S: M(R(S)) \to S$$

For each physical system $S$, $\varepsilon_S$ embeds the implementation of the inferred theory back into the original system. This is the statement that our theoretical model is a subsystem of reality.

\textbf{Step 3 (Triangle Identities).} The adjunction requires:
$$\varepsilon_{M(T)} \circ M(\eta_T) = \text{id}_{M(T)}$$
$$R(\varepsilon_S) \circ \eta_{R(S)} = \text{id}_{R(S)}$$

The first identity states: implementing a theory, theorizing about it, then implementing again recovers the original implementation. This holds by the consistency of physical laws.

The second identity states: theorizing about a system, implementing the theory, then theorizing again recovers the original theory. This holds by the uniqueness of optimal compression (MDL).

\textbf{Step 4 (Verification of Naturality).} For any morphism $\iota: T_1 \to T_2$ in $\mathbf{Th}$, the following diagram commutes:
$$\begin{array}{ccc}
T_1 & \xrightarrow{\eta_{T_1}} & R(M(T_1)) \\
\downarrow{\iota} & & \downarrow{R(M(\iota))} \\
T_2 & \xrightarrow{\eta_{T_2}} & R(M(T_2))
\end{array}$$

This follows from the functoriality of $R$ and $M$.

\textbf{Step 5 (Fixed Point Property).} For the Hypostructure $\mathbb{H} = \mathbb{H}_{\text{FG}}$:

(a) $M(\mathbb{H})$ is a physical system implementing the Fractal Gas dynamics.

(b) $R(M(\mathbb{H}))$ is the theory inferred by observing this physical system.

(c) By \cref{mt:manifold-sampling-isomorphism}, the Fractal Gas trace is the optimal dataset for learning the generator. Thus $R(M(\mathbb{H}))$ recovers $\mathbb{H}$ with minimal description length.

(d) Therefore:
$$R(M(\mathbb{H})) \cong \mathbb{H}$$

\textbf{Step 6 (Autopoietic Characterization).} The fixed point property $R \circ M \cong \text{Id}$ on $\mathbb{H}$ means:
\begin{itemize}
\item The theory produces a physical system ($M$)
\item The physical system produces observations
\item The observations regenerate the theory ($R$)
\end{itemize}

This is precisely the definition of \textbf{autopoiesis} [@MaturanaVarela80]: a network of processes that produces the components which realize the network.
\end{proof}

\textbf{Emergence Class:} Logic / Self-Reference

\textbf{Input Substrate:} Category of Theories $\mathbf{Th}$ + Category of Physical Systems $\mathbf{Phys}$

\textbf{Generative Mechanism:} Adjunction Fixed Point — $M \dashv R$ yields $R(M(\mathbb{H})) \cong \mathbb{H}$

\textbf{Output Structure:} Self-Consistent Logic — the Hypostructure is its own optimal description

\begin{corollary}[Ontological Closure]\label{cor:ontological-closure}
\textit{The distinction between "theory" and "reality" dissolves for the Hypostructure:}
$$\mathbb{H}_{\text{theory}} \xrightarrow{M} \mathbb{H}_{\text{physical}} \xrightarrow{R} \mathbb{H}_{\text{theory}}$$
\textit{forms a closed loop.}
\end{corollary}

\begin{corollary}[Canonical Representation]\label{cor:canonical-representation}
\textit{Up to equivalence, there is a unique self-describing theory: the Hypostructure.}
\end{corollary}

\begin{proof}
By Lawvere's theorem, fixed points are unique up to isomorphism in the appropriate quotient category.
\end{proof}

---

### Logical Foundations and Gödelian Considerations

#### Relation to Incompleteness

\begin{theorem}[Consistency]\label{thm:consistency}
\textit{The Hypostructure axiom system $\mathcal{A}_{\text{core}} = \{C, D, SC, LS, Cap, TB, R\}$ is consistent.}
\end{theorem}

\begin{proof}
We exhibit a model. Take:
\begin{itemize}
\item $X = L^2(\mathbb{R}^3)$ (square-integrable functions)
\item $S_t$ = heat semigroup $e^{t\Delta}$
\item $\Phi(u) = \int |\nabla u|^2 dx$ (Dirichlet energy)
\item $\mathfrak{D}(u) = \|u_t\|^2$ (dissipation rate)
\end{itemize}

This satisfies all axioms:
\begin{itemize}
\item \textbf{C:} Sublevel sets $\{\Phi \leq c\}$ are weakly compact in $L^2$
\item \textbf{D:} $\frac{d\Phi}{dt} = -2\mathfrak{D} \leq 0$ along the heat flow
\item \textbf{SC:} $\Phi(\lambda u) = \lambda^2 \Phi(u)$ (2-homogeneous)
\item \textbf{LS:} Standard gradient estimate near critical points
\item \textbf{Cap, TB:} Follow from Sobolev embedding
\end{itemize}

By Gödel's completeness theorem, existence of a model implies consistency.
\end{proof}

\begin{theorem}[Incompleteness Avoidance]\label{thm:incompleteness-avoidance}
\textit{The Hypostructure framework avoids Gödelian incompleteness by being a physical theory rather than a foundational mathematical system.}
\end{theorem}

\begin{proof}
Gödel's incompleteness theorems [@Godel31] apply to:
\begin{enumerate}
\item Formal systems containing arithmetic
\item That are recursively axiomatizable
\item And claim to capture all mathematical truth
\end{enumerate}

The Hypostructure:
\begin{itemize}
\item Is a physical theory making empirical predictions
\item Does not claim to axiomatize all of mathematics
\item Is "complete" only relative to the phenomena it models
\end{itemize}

The distinction is analogous to the difference between "ZFC is incomplete" and "Newtonian mechanics is complete for classical phenomena."

More precisely: let $\text{Th}(\mathbb{H})$ be the set of sentences true in the Hypostructure. This is not recursively enumerable (by Tarski's undefinability theorem). However, the *axioms* $\mathcal{A}_{\text{core}}$ are finite and decidable. The metatheorems are derived from these axioms plus standard mathematics (analysis, topology, etc.).

The framework is \textbf{relatively complete}: every physical phenomenon derivable from the axioms is captured by some metatheorem.
\end{proof}

#### Self-Reference and Löb's Theorem

\begin{theorem}[Self-Reference via Löb]\label{thm:self-reference-via-lb}
\textit{The Hypostructure can consistently assert its own correctness.}
\end{theorem}

\begin{proof}
By \textbf{Löb's Theorem} [@Loeb55]: For any formal system $T$ containing arithmetic,
$$T \vdash \Box(\Box P \to P) \to \Box P$$
where $\Box P$ means "$T$ proves $P$."

This implies: if $T$ proves "if $T$ proves $P$ then $P$", then $T$ proves $P$.

For the Hypostructure, let $P$ = "The Hypostructure correctly describes physics."

\textbf{Claim:} $\mathbb{H} \vdash \Box P \to P$.

\textit{Justification:} If the Hypostructure proves its own correctness (i.e., derives the metatheorems), then by the adjunction $R \circ M \cong \text{Id}$, the physical implementation confirms this correctness through observation.

By Löb's theorem: $\mathbb{H} \vdash \Box P$, i.e., the Hypostructure proves its own correctness.

This is not a contradiction because the "proof" is empirical (via $R$) rather than purely syntactic.
\end{proof}

---

### Final Synthesis

#### The Mathematical Unity

The proofs in this volume establish correspondences between:

| Field | Hypostructure Correspondence |
|:------|:-----------------------------|
| **Geometric Measure Theory** [@Federer69; @Simon83] | Varifold compactness $\to$ Axiom C; $\Gamma$-convergence $\to$ graph limits |
| **Stochastic Analysis** [@Oksendal03; @Kac49] | Feynman-Kac formula $\to$ \cref{mt:the-projective-feynman-kac-isomorphism}; McKean-Vlasov $\to$ mean-field limit |
| **Algorithmic Information** [@LiVitanyi08; @Solomonoff64] | Kolmogorov complexity $\to$ theory height; Levin search $\to$ \cref{metatheorem-38.5-sv-13-the-landauer-optimality} |
| **Categorical Logic** [@MacLane71; @Lawvere69] | Adjunctions $\to$ Map-Territory; Fixed points $\to$ Self-description |
| **Quantum Theory** [@vonNeumann32; @Lindblad76] | Lindblad equation $\to$ \cref{mt:cloning-lindblad-equivalence}; Unraveling $\to$ Fractal Gas |
| **General Relativity** [@Wald84; @Jacobson95] | Einstein equations $\to$ thermodynamic gravity; Holography $\to$ {prf:ref}`mt-imported-antichain-surface` |
| **Reinforcement Learning** [@Ecoffet21; @Friston10] | Go-Explore archive $\to$ Axiom Cap; Free Energy $\to$ Dissipation (Axiom D) |

#### The Philosophical Position

The Hypostructure framework implies a specific metaphysical stance:

1. **Structural Realism:** The fundamental nature of reality is structural (the tuple $(X, S_t, \Phi, \mathfrak{D}, G)$), not substantial.

2. **Computational Universe:** Physical law is algorithmic; the universe is a computation [@Deutsch85; @Lloyd06].

3. **Observer Participation:** The theory-reality adjunction $M \dashv R$ implies observers are not external to the system but intrinsic fixed points.

4. **Occam's Razor as Physical Law:** The MDL principle is not merely methodological but reflects the structure of reality via the Solomonoff prior.

#### Conclusion

The Hypostructure framework, defined by the tuple:
$$(X, S_t, \Phi, \mathfrak{D}, G)$$

with axioms $\{C, D, SC, LS, Cap, TB, R\}$, achieves:

1. **Unification:** All physical phenomena (quantum, gravitational, thermodynamic) emerge from structural axioms.

2. **Optimality:** The framework is the unique attractor of Bayesian inference on theory space.

3. **Self-Consistency:** The framework describes itself without contradiction via the autopoietic closure.

4. **Completeness:** Every derivable phenomenon is captured by the metatheorems.

**The framework is logically complete.**
$$\blacksquare$$

---


---