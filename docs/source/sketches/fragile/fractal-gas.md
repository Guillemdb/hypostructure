---
title: "Fractal Gas and Fractal Set Theorems (Permit-Adapted)"
---

# Fractal Gas and Fractal Set Theorems (Permit-Adapted)

This sketch extracts all Fractal Gas, Fractal Set, CST/IG convergence, and physics theorems from `old_docs/source/hypostructure/hypostructure.md` and rewrites them using the current thin-object + permit + node formulation. `docs/source/reference.md` contains no overlapping theorems, so all statements appear here.

## Legend: Thin Inputs and Permit Mapping

**Thin inputs (default):**
- $\mathcal{X}^{\text{thin}} = (X, d, \mathfrak{m})$
- $\Phi^{\text{thin}} = (\Phi, \nabla, \alpha)$
- $\mathfrak{D}^{\text{thin}} = (\mathfrak{D}, \beta)$
- $G^{\text{thin}} = (G, \rho, \mathcal{S})$
- $\partial^{\text{thin}} = (\mathcal{B}, \mathrm{Tr}, \mathcal{J}, \mathcal{R})$ (open systems only)

Here $(X,d,\mathfrak{m})$ is the **Thin Kernel arena** in the sense of {prf:ref}`def-thin-objects` (`docs/source/hypopermits_jb.md`). We reserve $\mu$ for *runtime* measures (empirical particle measures, trajectory measures, etc.) as in `docs/source/metalearning.md`.

**Permit legend (node IDs):**
- $D_E$ (Node 1 EnergyCheck)
- $\mathrm{Rec}_N$ (Node 2 ZenoCheck)
- $C_\mu$ (Node 3 CompactCheck)
- $\mathrm{SC}_\lambda$ (Node 4 ScaleCheck)
- $\mathrm{SC}_{\partial c}$ (Node 5 ParamCheck)
- $\mathrm{Cap}_H$ (Node 6 GeomCheck)
- $\mathrm{LS}_\sigma$ (Node 7 StiffnessCheck)
- $\mathrm{TB}_\pi$ (Node 8 TopoCheck)
- $\mathrm{TB}_O$ (Node 9 TameCheck)
- $\mathrm{TB}_\rho$ (Node 10 ErgoCheck)
- $\mathrm{Rep}_K$ (Node 11 ComplexCheck)
- $\mathrm{GC}_\nabla$ (Node 12 OscillateCheck)
- $\mathrm{Bound}_\partial$ (Node 13 BoundaryCheck)
- $\mathrm{Bound}_B$ (Node 14 OverloadCheck)
- $\mathrm{Bound}_\Sigma$ (Node 15 StarveCheck)
- $\mathrm{GC}_T$ (Node 16 AlignCheck)
- $\mathrm{Cat}_{\mathrm{Hom}}$ (Node 17 Lock)

**Axiom to permit translation (used below):**
- Axiom C -> $C_\mu$
- Axiom D -> $D_E$
- Axiom SC -> $\mathrm{SC}_\lambda$ (plus $\mathrm{SC}_{\partial c}$ when parameter stability is needed)
- Axiom Cap -> $\mathrm{Cap}_H$
- Axiom LS -> $\mathrm{LS}_\sigma$
- Axiom TB -> $\mathrm{TB}_\pi,\ \mathrm{TB}_O,\ \mathrm{TB}_\rho$ (topology/tameness/mixing layer)
- Axiom Rep -> $\mathrm{Rep}_K$
- Axiom GC -> $\mathrm{GC}_\nabla$ or $\mathrm{GC}_T$ (context noted)

## Status and Rigor Policy (Read This First)

This file is a **sketch**: it is meant to be *compatible with* the Hypostructure thin-object + permit language, but it is **not itself** a full sieve proof object with a closed certificate chain.

To keep the math honest and usable:

- **Certified** statements are those that reduce to (or are instantiated by) an explicit sieve run in a dedicated proof object (e.g. `docs/source/sketches/fragile/geometric_gas.md`).
- **Conditional** statements are standard results from analysis/probability/geometry that hold under explicit hypotheses stated in the block.
- **Heuristic** statements are analogies/interpretations. They are included to guide intuition and design choices, not as proved mathematics.
- **Default rule:** if a block does not explicitly declare a `Status`, treat it as **Heuristic**.

When a statement depends on nontrivial analytic inputs (minorization, Lyapunov drift, Hessian bounds, reach, sampling density, etc.), this sketch records them as **assumptions** rather than silently upgrading to a theorem.

**Nomenclature alignment (important):**
- In `docs/source/hypopermits_jb.md`, a **permit** is a gate/barrier predicate together with its **certificate type** (Definitions {prf:ref}`def-gate-permits`, {prf:ref}`def-barrier-permits`). In this sketch, a block’s **Permits** line should be read as its hypothesis list in the “sieve language”.
- In `docs/source/metalearning.md`, the symbol $K_A$ denotes an **axiom defect functional**. In this sketch, the symbol $K$ (when it appears) is used in the *certificate* sense (hypopermits), not as a defect functional.

## Fractal Gas Core Theorems (Solver Dynamics)

:::{prf:theorem} Geometric Adaptation (Metric Distortion Under Representation)
:label: thm:geometric-adaptation

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$, $G^{\text{thin}}$, embedding $\pi: X \to Y$.
**Permits:** $\mathrm{Rep}_K$ (N11), $\mathrm{SC}_\lambda$ (N4).

**Status:** Conditional (linear-algebraic; no solver assumptions).

**Assumptions:**
1. The algorithmic distance is computed from an embedding $\pi: X\to \mathbb{R}^n$ by
   $$
   d_{\mathrm{alg}}(x,y)=\|\pi(x)-\pi(y)\|_2
   $$
   (or any fixed Euclidean norm on the representation space).
2. Two embeddings are related by a linear map $T:\mathbb{R}^n\to\mathbb{R}^n$ via $\pi_2=T\circ \pi_1$.

**Statement:** For all $x,y\in X$,
$$
\sigma_{\min}(T)\, d_{\mathrm{alg}}^{(1)}(x,y)\ \le\ d_{\mathrm{alg}}^{(2)}(x,y)\ \le\ \|T\|\, d_{\mathrm{alg}}^{(1)}(x,y),
$$
where $\|T\|$ is the operator norm and $\sigma_{\min}(T)$ is the smallest singular value. In particular, if $T$ is invertible then $\pi_1$ and $\pi_2$ are bi-Lipschitz equivalent, and any Information Graph built from a monotone kernel of $d_{\mathrm{alg}}$ (e.g. Gaussian weights) changes only by a controlled rescaling/anisotropy of its effective neighborhood geometry.

**Remark (What “tunneling” can and cannot mean):** Changing representation can change **graph geodesics** and therefore the solver’s navigation *metric*, but it does not create new topological paths in the intrinsic space $X$; it changes the geometry used to move through $X$.
:::

:::{prf:proof}
Let $\Delta:=\pi_1(x)-\pi_1(y)\in\mathbb{R}^n$. Then $\pi_2(x)-\pi_2(y)=T\Delta$.
By definition of the operator norm and smallest singular value,
$$
\sigma_{\min}(T)\,\|\Delta\|_2\ \le\ \|T\Delta\|_2\ \le\ \|T\|\,\|\Delta\|_2.
$$
Substituting $\|\Delta\|_2=d_{\mathrm{alg}}^{(1)}(x,y)$ and $\|T\Delta\|_2=d_{\mathrm{alg}}^{(2)}(x,y)$ yields the claim.
:::

:::{prf:metatheorem} The Darwinian Ratchet (Reversible Case + Feynman–Kac Extension)
:label: mt:darwinian-ratchet

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$, $\mathfrak{D}^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $D_E$ (N1), $\mathrm{SC}_\lambda$ (N4).

**Status:** Conditional (standard reversible diffusion / QSD theory).

**Statement (A: reversible Langevin limit).** Suppose the kinetic component admits a continuum limit on a Riemannian manifold $(M,g_{\mathrm{eff}})$ with generator of (overdamped) Langevin type at inverse temperature $\beta$ and potential $\Phi$:
$$
\mathcal{L} f=\Delta_{g_{\mathrm{eff}}} f-\beta\langle \nabla_{g_{\mathrm{eff}}}\Phi,\nabla_{g_{\mathrm{eff}}} f\rangle.
$$
Then the invariant probability measure is the Gibbs law
$$
d\mu_\beta(x)=Z^{-1} e^{-\beta \Phi(x)}\,d\mathrm{Vol}_{g_{\mathrm{eff}}}(x)
\;=\;Z^{-1} e^{-\beta \Phi(x)}\sqrt{\det g_{\mathrm{eff}}(x)}\,dx
$$
in any coordinate chart $x$ on $M$.

**Statement (B: with selection/cloning).** If, instead, selection/cloning acts as a Feynman–Kac weight (or killing/respawn mechanism), the long-time normalized limit (when it exists) is generally a **principal eigenmeasure / quasi-stationary distribution** solving an eigenproblem of the form
$$
(\mathcal{L}+V)^* \nu=\lambda\,\nu
$$
(continuous time) or $\nu Q=\alpha\nu$ (discrete time). There is no closed-form Gibbs density in general; the “ratchet” behavior is controlled by the principal eigenpair and the drift/minorization constants.
:::

:::{prf:proof}
For (A), the stationary Fokker–Planck equation for $\mathcal{L}^*$ is solved by the Gibbs density with respect to the Riemannian volume form; this is the standard reversible Langevin computation on $(M,g_{\mathrm{eff}})$.

For (B), the normalized Feynman–Kac semigroup is governed by a twisted (killed/weighted) generator and its principal eigenpair. Existence/uniqueness and exponential convergence require additional assumptions (e.g. a Lyapunov drift condition and a small-set/Doeblin minorization), which are exactly the analytic inputs tracked by $D_E$ and $C_\mu$ and made explicit in the QSD block `mt:quasi-stationary-distribution-sampling`.
:::

:::{prf:principle} Coherence Phase Transition
:label: prin:coherence-transition

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$, $\mathfrak{D}^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $D_E$ (N1), $\mathrm{SC}_\lambda$ (N4).

**Status:** Heuristic-to-conditional (requires a specified continuum scaling; not a sieve-level metatheorem).

**Assumptions (for a conditional reading):**
1. A viscous mixing length $l_\nu$ is well-defined for the chosen kinetic/viscous operator over one macroscopic step, and scales as $l_\nu\sim \sqrt{\nu\,\Delta t}$ in the regime of interest.
2. A cloning/jitter correlation length $l_{\mathrm{clone}}$ is well-defined and scales as $l_{\mathrm{clone}}\sim \sigma_x$ (clone position jitter).
3. The coherence observable below is self-averaging as $N\to\infty$ (law of large numbers regime).

**Statement:** The internal coherence of the swarm is controlled by the ratio of the viscous mixing scale to the cloning correlation scale. A convenient (dimensionless) coherence observable is
$$
\bar v := \frac{1}{N}\sum_{i=1}^N v_i,\qquad
\Psi_{\mathrm{coh}} := \frac{\|\bar v\|^2}{\frac{1}{N}\sum_{i=1}^N \|v_i\|^2}\ \in[0,1],
$$
where $v_i$ are the particle velocities (or generalized “update directions” if no explicit velocities exist). Heuristically:
- **Gas:** $l_\nu \ll l_{\mathrm{clone}}$ (weak viscous synchronization) $\Rightarrow$ $\Psi_{\mathrm{coh}}\approx 0$.
- **Solid:** $l_\nu \gg l_{\mathrm{clone}}$ (strong viscous synchronization) $\Rightarrow$ $\Psi_{\mathrm{coh}}\approx 1$.
- **Liquid:** intermediate regime with partial coherence.

A phase transition (or crossover) is expected when $l_\nu$ and $l_{\mathrm{clone}}$ are comparable.
:::

:::{prf:proof}
**Step 1 (Order Parameter).**
Define $\Psi_{\mathrm{coh}}$ as the ratio “mean velocity energy / mean kinetic energy”. By Cauchy–Schwarz,
$\|\bar v\|^2 \le \frac{1}{N}\sum_i \|v_i\|^2$, hence $\Psi_{\mathrm{coh}}\in[0,1]$.
If the velocities are i.i.d. with mean $0$ (incoherent), $\|\bar v\|^2$ is small and $\Psi_{\mathrm{coh}}\approx 0$.
If the velocities are nearly identical (coherent motion), $\Psi_{\mathrm{coh}}\approx 1$.

**Step 2 (Competition of Scales).**
The dynamics are governed by two length scales:
- **Viscous Length $l_\nu$:** The distance over which momentum diffuses in time $\Delta t$. $l_\nu \sim \sqrt{\nu \Delta t}$.
- **Cloning Length $l_{\mathrm{clone}}$:** The mean separation between a parent and its clone (determined by position jitter). $l_{\mathrm{clone}} \sim \sigma_x$.

**Step 3 (Criticality).**
When $l_\nu \gg l_{\mathrm{clone}}$, viscous mixing synchronizes the swarm faster than cloning decorrelates it, leading to coherent motion.
When $l_\nu \ll l_{\mathrm{clone}}$, cloning decorrelates the swarm faster than viscosity can synchronize it, leading to incoherent motion.
A crossover is expected when $l_\nu/l_{\mathrm{clone}}$ is $O(1)$; extracting a sharp critical exponent requires a specific scaling limit and is not asserted here.
:::

:::{prf:theorem} Topological Regularization (Cheeger Bound, Conditional)
:label: thm:cheeger-bound

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$, $\mathfrak{D}^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $D_E$ (N1), $\mathrm{LS}_\sigma$ (N7), $\mathrm{Cap}_H$ (N6), $\mathrm{TB}_\pi$ (N8).

**Status:** Conditional (graph/Markov-chain mixing; not implied by viscosity alone).

**Assumption (uniform minorization / Doeblin condition):** The Information Graph induces a reversible Markov kernel $P_t$ on vertices with stationary law $\pi_t$, and there exists $\delta\in(0,1]$ such that for all times $t$ and all vertices $i$,
$$
P_t(i,\cdot)\ \ge\ \delta\,\pi_t(\cdot).
$$

**Statement:** Under this assumption, the chain has a uniform spectral gap $\lambda_1(P_t)\ge \delta$ and the Cheeger (conductance) constant is uniformly bounded below:
$$
h(G_t)\ \ge\ \frac{\lambda_1(P_t)}{2}\ \ge\ \frac{\delta}{2}\ >\ 0.
$$
In particular the graph stays connected and does not “pinch off”. (In concrete Fractal/Geometric Gas instantiations, a Doeblin $\delta$ typically comes from a **softmax floor** on companion/edge weights on a bounded diameter domain; viscosity $\nu$ affects *velocity mixing*, but does not by itself create edges.)
:::

:::{prf:proof}
Under the Doeblin condition, $P_t$ is a strict contraction in total variation with coefficient at most $1-\delta$ (Dobrushin’s argument), implying geometric ergodicity and a spectral gap bounded below by $\delta$.
For reversible chains, the (reverse) Cheeger inequality gives $h\ge \lambda_1/2$. Combining yields $h\ge \delta/2$.
:::

:::{prf:principle} Induced Local Geometry (Quadratic Form from Landscape + Graph Energy)
:label: thm:induced-riemannian-structure

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$, $\mathfrak{D}^{\text{thin}}$.
**Permits:** $D_E$ (N1), $\mathrm{LS}_\sigma$ (N7), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (becomes rigorous under uniform positive-definiteness).

**Statement:** On a compact “alive” slice where $\Phi$ and $\mathfrak{D}$ are $C^2$, the Fractal/Information-Graph constructions canonically define a **positive semidefinite quadratic form** on perturbations $\delta z$ of the swarm state that combines:
- local curvature of the landscape (via Hessians of $\Phi$ and $\mathfrak{D}$), and
- discrete Dirichlet energy from the Information Graph (via its Laplacian).

When this quadratic form is uniformly positive definite on the tangent space (e.g. near nondegenerate minima or under uniform ellipticity hypotheses), it defines a genuine Riemannian metric; otherwise it defines a sub-Riemannian/degenerate geometry.

**Do not read this as a literal tensor identity** “$g=\nabla^2\Phi+\nu L$”: Hessians and graph Laplacians live on different objects and only combine meaningfully after a concrete discretization choice (finite-dimensional tangent space, chosen coordinates, and a graph energy functional).
:::

:::{prf:proof}
This block is a design principle: a Taylor expansion of a smooth energy landscape produces Hessian quadratic forms, while graph-based “kinetic” regularization produces Dirichlet energies of the form $\sum_{i,j} w_{ij}\|\delta z_i-\delta z_j\|^2$, whose Euler–Lagrange operator is a graph Laplacian. Under uniform positive-definiteness, such quadratic forms define an inner product and therefore a Riemannian metric.
:::

## Fractal Set, CST/IG Reconstruction, and Convergence

:::{prf:principle} Geometric Reconstruction
:label: prin:geometric-reconstruction

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$, $\mathfrak{D}^{\text{thin}}$, $G^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $\mathrm{SC}_\lambda$ (N4), $\mathrm{LS}_\sigma$ (N7), $\mathrm{Cap}_H$ (N6), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (becomes rigorous under manifold-learning + graph-limit hypotheses).

**Assumptions (one typical “graph limit” setting):**
1. (**Manifold hypothesis**) There is a compact $d$-dimensional $C^2$ Riemannian manifold $(M,g)$ and an embedding $M\hookrightarrow X$ such that the relevant swarm states concentrate on $M$ in the large-$N$ regime.
2. (**Sampling**) At each scale, the swarm positions define an empirical measure $\mu_N=\frac{1}{N}\sum_{i=1}^N \delta_{x_i}$ that converges weakly to a probability measure $\mu$ on $M$ with smooth density $\rho$ w.r.t. $\mathrm{Vol}_g$, bounded away from $0$ and $\infty$ on the interior region of interest.
3. (**Graph construction**) The Information Graph uses a bandwidth/kernel regime (or $k$-NN regime) in which shortest-path distances approximate $d_g$ and the graph Laplacian converges (after rescaling) to the Laplace–Beltrami operator (e.g. Mosco/Dirichlet-form convergence).
4. (**Metric–thermodynamic link**) The thin-kernel dissipation is interpreted in the metric-measure sense (Cheeger energy / Fisher information on $(M,d_g,\mu)$ as in `docs/source/hypopermits_jb.md` §4.1), so that the continuum “thermodynamic geometry” is the one induced by $(d_g,\mu)$ rather than by an ad hoc identification $g\equiv \nabla^2\Phi$.

**Statement:** Under these assumptions, the Fractal/Information-Graph discretizations can be read as a consistent *metric-measure approximation* of $(M,d_g,\mu)$:
- **Density / volume form:** empirical degree and sampling statistics converge to $\rho\,d\mathrm{Vol}_g$ (up to the kernel normalization).
- **Distance:** the IG shortest-path metric $d_{\mathrm{IG}}$ converges to $d_g$ (Gromov–Hausdorff, under the graph regime above).
- **Dirichlet energy:** the IG quadratic form converges to the Cheeger/Dirichlet energy on $(M,g,\mu)$ (Mosco/Γ-type convergence).
- **Curvature proxies (diagnostic):** discrete curvature notions (e.g. Ollivier–Ricci) can act as *proxies* for smooth curvature only under additional scaling and regularity hypotheses; no automatic curvature-identification is claimed here.
:::

:::{prf:proof}
**Step 1 (Manifold Hypothesis).**
Assume the fitness landscape $\Phi$ and dissipation $\mathfrak{D}$ define a smooth submanifold $M \subset X$ of dimension $d \ll D$. This is the "intrinsic geometry" of the problem.

**Step 2 (Metric–thermodynamic link).**
On a metric-measure space $(M,d_g,\mu)$, the thermodynamic objects used by the hypostructure framework (entropy dissipation, Fisher information, Cheeger energy) are defined *from* $(d_g,\mu)$, not as a pointwise identification of $g$ with a Hessian of $\Phi$. In particular, when $\mu$ is of Gibbs type $\mu\propto e^{-\beta\Phi}\mathrm{Vol}_g$, Fisher information controls gradients of $\Phi$ but does not turn $\nabla^2\Phi$ into a metric tensor by itself.

**Step 3 (Graph Convergence).**
As $N \to \infty$, the graph Laplacian $L_G$ converges to the Laplace-Beltrami operator $\Delta_g$ on the manifold $(M, g)$.
The graph distance converges to the geodesic distance:
$$\lim_{N \to \infty} d_{\text{IG}}(x,y) = \inf_\gamma \int_\gamma \sqrt{g_{ij} \dot{x}^i \dot{x}^j} dt$$

**Step 4 (Curvature proxies).**
Discrete curvature notions (e.g. Ollivier–Ricci curvature) can serve as **proxies** for smooth curvature in regimes where the graph is a sufficiently fine geometric discretization of $(M,g)$ and the curvature definition is scaled appropriately. In general, curvature recovery is subtle and requires additional hypotheses (sampling density, bandwidth scaling, and curvature bounds), so this sketch treats curvature statements as conditional/diagnostic rather than automatic.
:::

:::{prf:theorem} Causal Horizon Lock (Cut-Capacity / “Area Law”, Conditional)
:label: thm:causal-horizon-lock

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$, $\mathfrak{D}^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $D_E$ (N1), $\mathrm{SC}_\lambda$ (N4), $\mathrm{Cap}_H$ (N6), $\mathrm{TB}_\pi$ (N8).

**Status:** Conditional (information-theoretic; requires a bounded per-edge channel model).

**Assumption (bounded boundary capacity):** The dynamics on the Information Graph are implemented by a local message-passing / Markov update rule where each boundary edge carries at most $C_e$ nats of information per step (channel capacity bound). Define the discrete “area” of the boundary by the cut capacity
$$
\mathrm{Area}_{\mathrm{IG}}(\partial \Sigma)\ :=\ \sum_{e\in \partial \Sigma} C_e
$$
(or, in the unweighted case, $\mathrm{Area}_{\mathrm{IG}}(\partial\Sigma)=|\partial\Sigma|$ up to a constant factor).

**Statement:** Under this assumption, the one-step information flow across the boundary satisfies the cut bound
$$
I(\Sigma\to \Sigma^c)\ \le\ \mathrm{Area}_{\mathrm{IG}}(\partial \Sigma).
$$
This is the precise mathematical content of the “area law” intuition: **boundary** degrees of freedom limit how much information can cross between inside and outside in one step.
:::

:::{prf:proof}
This is a standard cut-capacity argument (max-flow/min-cut intuition): in one update step, information that moves from $\Sigma$ to $\Sigma^c$ must pass through the boundary edges. If each boundary edge is a channel of capacity $C_e$, the total one-step transferable mutual information is at most the sum of capacities across the cut. Formal proofs use the data-processing inequality and subadditivity of mutual information under independent channels.
:::

:::{prf:principle} Scutoid Selection Principle (Heuristic Geometry)
:label: thm:scutoid-selection

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$, $\mathfrak{D}^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $D_E$ (N1), $\mathrm{TB}_\pi$ (N8), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic.

**Statement:** In 3D swarm dynamics with local neighbor exchange (cloning + companion reassignment), the induced Voronoi/Delaunay adjacency can undergo T1-like transitions. In some geometries this produces scutoid-like cell shapes. This provides a geometric analogy for “topological regularization,” but it is not asserted here as a variational minimization theorem (e.g. no claim is made that a Regge action is minimized by the algorithm).
:::

:::{prf:proof}
This block is intentionally non-rigorous: it records a geometric interpretation (neighbor exchanges can induce scutoid-like adjacency changes) without claiming a precise variational principle.
:::

:::{prf:theorem} Archive Invariance (Gromov–Hausdorff Stability, Conditional)
:label: thm:archive-invariance

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$, $\mathfrak{D}^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $\mathrm{LS}_\sigma$ (N7), $\mathrm{Cap}_H$ (N6).

**Status:** Conditional (metric-space convergence; “quasi-isometry” is not the right notion on compact spaces).

**Assumption (common compact limit):** There exists a compact metric space $(M,d)$ and scales $\varepsilon_k\downarrow 0$ such that
$$
d_{\mathrm{GH}}(\mathcal{F}_1,M)\le \varepsilon_1,\qquad d_{\mathrm{GH}}(\mathcal{F}_2,M)\le \varepsilon_2.
$$

**Statement:** Then
$$
d_{\mathrm{GH}}(\mathcal{F}_1,\mathcal{F}_2)\ \le\ \varepsilon_1+\varepsilon_2,
$$
and there exists an $(\varepsilon_1+\varepsilon_2)$-approximation map between the two archives (an $\varepsilon$-isometry in the standard GH sense). Consequently, any **stable** geometric invariant (e.g. persistent homology at scales $\gg \varepsilon_1+\varepsilon_2$) agrees between the two runs.
:::

:::{prf:proof}
The GH triangle inequality yields the bound on $d_{\mathrm{GH}}(\mathcal{F}_1,\mathcal{F}_2)$. The existence of an $\varepsilon$-approximation map is part of the definition of GH distance.
:::

### Fractal Set Foundations (Discrete-to-Continuum)

:::{prf:metatheorem} Fractal Representation
:label: mt:fractal-representation

**Thin inputs:** all thin objects.
**Permits:** $C_\mu$, $D_E$, $\mathrm{SC}_\lambda$, $\mathrm{Cap}_H$, $\mathrm{Rep}_K$, $\mathrm{TB}_\pi$.

**Status:** Heuristic-to-conditional (inverse-limit constructions are standard once a compatible projective system is specified).

**Assumptions (minimal):**
1. A compatible projective system of Information Graphs and coarse-graining maps $(G_n,\phi_{nm})$ is specified.
2. There exist embeddings $\iota_n:V(G_n)\to X$ compatible with $\phi_{nm}$, so that $\Pi((x_n)):=\lim_{n\to\infty}\iota_n(x_n)$ exists in $X$ for coherent threads $(x_n)$.

**Statement:** Under finite local complexity and discrete-time approximability, there exists a Fractal Set $\mathcal{F}$ (an inverse limit of Information Graphs) with a representation map $\Pi: \mathcal{F} \to X$ such that:
1.  **States:** Time slices $\mathcal{F}_t$ map to states $u(t) \in X$.
2.  **Dynamics:** Paths in the Causal Structure Tree map to trajectories in $X$.
3.  **Axioms:** Axiom C/D/SC at the level of the projected dynamics in $X$ translates to Compactness/Dissipation/Scaling statements on the projective system defining $\mathcal{F}$.
4.  **Commutativity:** Coarse-graining maps on $\mathcal{F}$ lift to graph homomorphisms.
:::

:::{prf:proof}
**Step 1 (Inverse Limit Construction).**
Let $\{G_n, \phi_{nm}\}$ be the projective system of Information Graphs at increasing resolution ($n \to \infty$). The Fractal Set is the inverse limit:
$$\mathcal{F} = \varprojlim (G_n, \phi_{nm}) = \{ (x_n) \in \prod G_n : \phi_{nm}(x_m) = x_n \}$$
This object is compact and totally disconnected (a Cantor-like space) but carries the "shadow" of the continuous geometry.

**Step 2 (The Representation Map).**
Define $\Pi: \mathcal{F} \to X$ by $\Pi((x_n)) = \lim_{n \to \infty} \iota_n(x_n)$, where $\iota_n$ embeds nodes of $G_n$ into the state space $X$.
Completeness of $X$ ensures the limit exists.

**Step 3 (Dynamical Equivalence).**
The shift operator on the sequence space corresponds to the time-evolution operator $S_t$ on $X$. The discrete kinetics on $G_n$ approximate $S_t$ with error $O(\tau^k)$.

**Step 4 (Axiom Translation).**
- **Compactness:** Since $\mathcal{F}$ is compact (Tychonoff theorem), its image $\Pi(\mathcal{F})$ is compact in $X$.
- **Dissipation:** Energy decrease on graphs implies Lyapunov function decrease on $X$.
:::

:::{prf:theorem} Fitness Convergence via Gamma-Convergence
:label: thm:fitness-convergence

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $D_E$ (N1).

**Status:** Conditional (standard $\Gamma$-convergence; requires an identification of discrete states with continuum states).

**Assumptions (one typical setting):**
1. There is an identification/embedding map $\iota_\varepsilon:\mathcal{F}_\varepsilon\to X$ so that sequences $(x_\varepsilon\in\mathcal{F}_\varepsilon)$ can be compared via $\iota_\varepsilon(x_\varepsilon)\to x$ in $X$.
2. The family $\{\Phi_\varepsilon\}$ is **equicoercive** with respect to this identification (sublevel sets are precompact).
3. The $\Gamma$-liminf and $\Gamma$-limsup inequalities hold with respect to $\iota_\varepsilon$.

**Statement:** Under these assumptions, the discrete functionals $\Phi_\varepsilon$ $\Gamma$-converge to $\Phi$ (in the sense above). Consequently, almost-minimizers of $\Phi_\varepsilon$ have accumulation points that minimize $\Phi$ (and minimizing values converge).
:::

:::{prf:proof}
This is the standard $\Gamma$-convergence implication:
- the liminf and limsup inequalities are the definition of $\Gamma$-convergence (relative to $\iota_\varepsilon$), and
- equicoercivity upgrades $\Gamma$-convergence to convergence of (almost-)minimizers.

The nontrivial content in applications is to verify (i) the approximation map $\iota_\varepsilon$ and (ii) the liminf/limsup bounds from the concrete discrete energy definition.
:::

:::{prf:theorem} Gromov-Hausdorff Convergence
:label: thm:gromov-hausdorff-convergence

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $G^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $\mathrm{Rep}_K$ (N11).

**Status:** Conditional (standard geometric-graph convergence results; requires sampling and scaling hypotheses).

**Assumptions (one typical setting):**
1. $(M,d_g)$ is a compact Riemannian manifold.
2. The vertex set $V_\varepsilon\subset M$ is an $\varepsilon$-net (Hausdorff distance $\le \varepsilon$).
3. The graph edges/weights are constructed from a kernel or neighborhood radius $r_\varepsilon\to 0$ in a regime that makes shortest-path distances approximate $d_g$ (e.g. dense enough to avoid “short-circuiting” and connected enough to avoid fragmentation).

**Statement:** Under such hypotheses, the metric spaces $(V_\varepsilon,d_{\mathrm{IG}}^\varepsilon)$ converge to $(M,d_g)$ in the Gromov–Hausdorff sense:
$$
(V_\varepsilon, d_{\mathrm{IG}}^\varepsilon)\xrightarrow{\mathrm{GH}} (M, d_g).
$$
:::

:::{prf:proof}
Given an $\varepsilon$-net, the natural correspondence is $(v,x)$ with $x=v\in M$. The net property controls the Hausdorff part of GH distance. The nontrivial part is to bound the distortion of shortest-path distances by comparing graph paths to manifold geodesics and vice versa; this is where the edge radius/kernel scaling hypotheses enter.
:::

:::{prf:metatheorem} Convergence of Minimizing Movements
:label: mt:convergence-minimizing-movements

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $D_E$ (N1), $\mathrm{LS}_\sigma$ (N7).

**Status:** Conditional (standard minimizing-movements theory).

**Assumptions:** $(X,d)$ is a complete metric space and $\Phi:X\to(-\infty,\infty]$ is proper, lower semicontinuous, and (geodesically) $\lambda$-convex for some $\lambda\in\mathbb{R}$ (or satisfies an alternative slope-compactness condition ensuring well-posed gradient flows).

**Statement:** The discrete “minimizing movement” scheme
$$x_{k+1} \in \mathrm{argmin}_y \left( \Phi(y) + \frac{1}{2\tau} d^2(x_k, y) \right)$$
converges (as $\tau\to 0$) to the unique curve of maximal slope (metric gradient flow) for $\Phi$. Under the standard hypotheses above, the limit satisfies the Energy–Dissipation Inequality, and under additional regularity it satisfies the Energy–Dissipation Equality.
:::

:::{prf:proof}
**Step 1 (Discrete Variational Problem).**
The update rule is implicit Euler discretization of gradient descent. It balances minimizing potential $\Phi$ with minimizing transport cost $d^2$.

**Step 2 (De Giorgi Interpolation).**
Construct a continuous trajectory $\tilde{x}_\tau(t)$ by interpolating the discrete points.
We check if the limit satisfies the weak formulation of the gradient flow.

This is a classical result of Ambrosio–Gigli–Savaré: under the stated hypotheses, the interpolated discrete solutions are precompact and any limit is the gradient flow of $\Phi$.
:::

:::{prf:metatheorem} Symplectic Shadowing
:label: mt:symplectic-shadowing

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $G^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $\mathrm{GC}_\nabla$ (N12), $\mathrm{Rep}_K$ (N11).

**Status:** Conditional (backward error analysis for symplectic integrators).

**Statement:** For sufficiently smooth (often analytic) Hamiltonians and sufficiently small step size $h$, a symplectic splitting scheme is the exact time-$h$ map of a **modified Hamiltonian**
$$
\tilde H = H + h H_1 + h^2 H_2 + \cdots
$$
up to a truncation error. As a consequence, the numerical energy error typically remains bounded and oscillatory over long times; in analytic settings one can obtain exponentially long stability times in $1/h$.
:::

:::{prf:proof}
**Step 1 (Baker-Campbell-Hausdorff).**
The splitting method (Lie-Trotter) approximates $e^{h(A+B)}$ by $e^{hA} e^{hB}$. The BCH formula gives $e^{hA} e^{hB} = e^{Z(h)}$ where $Z(h) = h(A+B) + \frac{h^2}{2}[A,B] + \dots$.
Identifying $A$ and $B$ with Liouville operators for kinetic and potential parts, the flow generated by $Z(h)$ is the exact flow of a perturbed Hamiltonian.

**Step 2 (Modified Hamiltonian).**
We explicitly construct the formal power series for $\tilde{H}$. Since the integrator is symplectic, such a Hamiltonian exists (locally).

**Step 3 (Energy Bound).**
Backward error analysis controls the difference between the numerical map and the modified Hamiltonian flow; bounded long-time energy error follows from conservation of $\tilde H$ for the modified flow and the smallness of $H-\tilde H$ at the chosen truncation order.
:::

:::{prf:metatheorem} Homological Reconstruction
:label: mt:homological-reconstruction

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{TB}_\pi$ (N8), $\mathrm{Rep}_K$ (N11).

**Status:** Conditional (computational topology; requires reach + sampling hypotheses).

**Statement (standard recovery pattern):** Let $M\subset \mathbb{R}^D$ be a compact $C^2$ submanifold with reach $\tau>0$, and let $P\subset M$ be an $\varepsilon$-sample (Hausdorff distance $\le\varepsilon$) with $\varepsilon<\tau/2$.
Then:
1. The union of balls $U_\varepsilon=\bigcup_{p\in P} B_\varepsilon(p)$ deformation retracts to $M$.
2. The Čech complex $\check C_\varepsilon(P)$ is homotopy equivalent to $U_\varepsilon$ (Nerve Lemma), hence $\check C_\varepsilon(P)\simeq M$.
3. The Vietoris–Rips and Čech filtrations are interleaved (up to a scale factor), so persistent homology of $\mathrm{VR}_r(P)$ recovers $H_\ast(M)$ at appropriate scales.

This is the rigorous content behind using IG samples to infer topological invariants: topology recovery requires **geometric sampling conditions**, not just an algorithmic run.
:::

:::{prf:proof}
Items (1)–(2) follow from the Niyogi–Smale–Weinberger theorem and the Nerve Lemma. Item (3) is the standard Čech–Vietoris–Rips interleaving: $\check C_r(P)\subseteq \mathrm{VR}_{2r}(P)\subseteq \check C_{2r}(P)$ (in Euclidean ambient space), which yields homology recovery in the persistent sense.
:::

:::{prf:metatheorem} Symmetry Completion
:label: mt:symmetry-completion

**Thin inputs:** $G^{\text{thin}}$.
**Permits:** $\mathrm{GC}_\nabla$ (N12), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (bundle/Noether inputs are standard under explicit hypotheses; “full hypostructure determination” is interpretive).

**Statement:** Given a specified symmetry group $G$ acting on local internal states and a compatible family of transition functions satisfying the cocycle condition on overlaps, the local gauge data determine (up to isomorphism) a principal $G$-bundle with a connection. If, in addition, the (continuum-limit) dynamics admit a $G$-invariant Lagrangian/Hamiltonian, Noether’s theorem yields conserved quantities/constraints. These constrain admissible hypostructure instantiations but do not by themselves uniquely determine all thin/thick objects.
:::

:::{prf:proof}
**Step 1 (Local Gauge).**
At each node $i$, the "internal state" transforms under a group $G$. Interactions are invariant under global rotation $g \in G$.

**Step 2 (Holonomy).**
Parallel transport around loops defines the holonomy group. If the curvature vanishes (flat connection), global symmetry is preserved. If not, we have a Gauge Theory.

**Step 3 (Reconstruction).**
The collection of local fibers and transition functions (connection) defines a principal $G$-bundle. The dynamics must respect this bundle structure to be well-defined. Thus, specifying the local symmetry specifies the allowed global physics.
:::

:::{prf:metatheorem} Gauge-Geometry Correspondence
:label: mt:gauge-geometry-correspondence

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $G^{\text{thin}}$.
**Permits:** $\mathrm{GC}_\nabla$ (N12), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (lattice-gauge correspondence is conditional once a gauge field is specified; “spacetime geometry emerges” is interpretive).

**Statement:** If the Information Graph is endowed with group-valued edge variables $U_{ij}\in G$ interpreted as parallel transports, then holonomies around loops encode a discrete curvature (Wilson loops / plaquette holonomy) and in suitable continuum limits recover the field-strength tensor.
$$ F_{\mu\nu} \leftrightarrow \text{Hol}(\text{plaquette}) $$
Interpreting the same data as a unified “geometry + forces” object is heuristic beyond this lattice-gauge correspondence.
:::

:::{prf:proof}
**Step 1 (Lattice Gauge Theory).**
Interpret the graph as a lattice. Link variables $U_{ij}$ are parallel transport operators.
The Wilson loop around a face (plaquette) $P_{ijkl} = U_{ij}U_{jk}U_{kl}U_{li}$ measures the flux through the face.

**Step 2 (Continuum Limit).**
For smooth fields $A_\mu$, $U_{ij} \approx e^{i \int A \cdot dx}$. The Wilson loop is $e^{i \oint A \cdot dx} \approx e^{i F_{\mu\nu} \Delta x \Delta y}$.
Thus, the discrete loop product corresponds to the field strength tensor $F_{\mu\nu}$.

**Step 3 (Unified Geometry).**
In Kaluza-Klein theory, gauge fields appear as metric components in extra dimensions. The Information Graph naturally implements a discrete version of this, where "internal" edge weights encode geometry and "internal" group labels encode forces.
:::

:::{prf:metatheorem} Emergent Continuum
:label: mt:emergent-continuum

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $\mathrm{Cap}_H$ (N6), $\mathrm{LS}_\sigma$ (N7), $\mathrm{Rep}_K$ (N11).

**Status:** Conditional (Dirichlet-form / graph-Laplacian convergence under explicit sampling/weighting hypotheses).

**Assumptions (typical manifold setting):**
1. The graphs are built from $\varepsilon_N$-samples of a compact $C^2$ Riemannian manifold $(M,g)$ with $\varepsilon_N\downarrow 0$, using a kernel-based weight scheme with bandwidth shrinking at an admissible rate.
2. The associated graph Dirichlet forms Mosco-converge to the Dirichlet form of Brownian motion on $(M,g)$.

**Statement:** With spectral gap and Laplacian convergence on the Information Graph, the rescaled graph Laplacian converges to the Laplace-Beltrami operator $\Delta_g$ on the emergent manifold. The random walk converges to Brownian motion on $(M, g)$.
:::

:::{prf:proof}
**Step 1 (Mosco Convergence).**
We prove Mosco convergence of the energy forms $\mathcal{E}_N \xrightarrow{M} \mathcal{E}$.
This requires two conditions:
1.  **Limsup:** For every $u \in H^1(M)$, there exists a sequence $u_N$ in the graph such that $\mathcal{E}_N(u_N) \to \mathcal{E}(u)$.
2.  **Liminf:** For every sequence $u_N$ converging to $u$, $\liminf \mathcal{E}_N(u_N) \geq \mathcal{E}(u)$.

**Step 2 (Semigroup Convergence).**
Mosco convergence implies strong convergence of the generated semigroups $P_t^N \to P_t$.
$P_t^N = e^{t L_N}$ is the random walk diffusion.
$P_t = e^{t \Delta_g}$ is heat diffusion on the manifold.

**Step 3 (Structure Preservation).**
Since the convergence is structural (via the form), the limiting object inherits spectral properties (eigenvalues) and functional inequalities (Poincaré, Log-Sobolev) from the discrete approximations.
:::

:::{prf:metatheorem} Dimension Selection
:label: mt:dimension-selection

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{SC}_\lambda$ (N4), $\mathrm{Cap}_H$ (N6).

**Status:** Heuristic-to-conditional (dimension notions are standard; the “selection” mechanism depends on algorithm + sampling).

**Statement:** When the IG sequence admits a scaling limit with well-defined Hausdorff and walk dimensions $(d_H,d_w)$, the spectral dimension $d_s=2d_H/d_w$ controls long-time diffusion scaling. The “dimension selection” interpretation refers to solver dynamics/sampling biasing the observed scaling exponents toward regimes where these dimensions appear stable across scales; it is not asserted as a universal theorem that the algorithm forces a particular dimension in all problems.
:::

:::{prf:proof}
**Step 1 (Pointwise Dimension).**
For each point, we estimate the local dimension $d(x) \approx \frac{\log k}{\log r_k}$.
The swarm stabilizes when $d(x)$ is uniform across the attractor (multifractal spectrum collapses).

**Step 2 (Scaling Relation).**
The walk dimension $d_w$ (diffusion speed) and Hausdorff dimension $d_H$ (volume growth) must satisfy the Einstein relation for the spectral dimension $d_s = 2 d_H / d_w$.
Stable continuum limits require $d_s$ to be integer-like (or consistent with a specific universality class).

**Step 3 (Renormalization Fixed Point).**
Consider the dimension as a running coupling under RG flow. The observed dimension is the fixed point of this flow. The Fractal Gas naturally relaxes into a geometry where the dimension is consistent with the diffusion process.
:::

:::{prf:metatheorem} Discrete Curvature-Stiffness Transfer
:label: mt:curvature-stiffness-transfer

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $\mathrm{LS}_\sigma$ (N7), $\mathrm{Cap}_H$ (N6).

**Status:** Heuristic-to-conditional (discrete curvature ⇒ functional inequalities is conditional; transfer to the continuum requires stability under the chosen convergence notion).

**Statement:** In graph settings, uniform lower bounds on an appropriate discrete curvature notion (e.g. Bakry–Émery curvature-dimension, or Ollivier-Ricci under additional regularity) imply functional inequalities such as a Poincaré inequality / spectral gap for the graph Laplacian. If, additionally, the graph sequence converges to a metric-measure limit and the curvature/Dirichlet-form bounds are stable under this convergence, the limiting space inherits the corresponding inequality (and in RCD-type settings can be interpreted as a lower Ricci-curvature bound).
:::

:::{prf:proof}
**Step 1 (Ollivier-Ricci Curvature).**
Defined by transport distance contraction:
$$ \kappa(x,y) = 1 - \frac{W_1(m_x, m_y)}{d(x,y)} $$
where $m_x, m_y$ are local neighborhoods (measures). Positive $\kappa$ means balls are closer "on average" than their centers (convergence).

**Step 2 (Bakry-Emery Condition).**
For $\Gamma$-calculus/Bakry–Émery curvature on graphs, $CD(K,\infty)$ is a checkable hypothesis implying Poincaré/log-Sobolev inequalities. For Ollivier-Ricci curvature, positive lower bounds imply certain Wasserstein contraction properties and can yield spectral-gap bounds under additional assumptions; in this sketch we treat the implication “curvature $\Rightarrow$ stiffness” as conditional on the chosen curvature notion and regularity regime. Stability under convergence is likewise conditional (e.g. Mosco convergence of Dirichlet forms and stability of the curvature-dimension condition in the relevant category, such as RCD limits).

**Step 3 (Stiffness).**
By the Lichnerowicz theorem (extended to metric measure spaces), $CD(K, \infty)$ with $K > 0$ implies a spectral gap $\lambda_1 \geq K$. This is exactly the **Stiffness** (Axiom LS) property required for exponential convergence.
:::

:::{prf:metatheorem} Dobrushin-Shlosman Interference Barrier
:label: mt:dobrushin-shlosman

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{LS}_\sigma$ (N7), $\mathrm{TB}_\rho$ (N10).

**Status:** Conditional (standard decay-of-correlations under a uniqueness regime; permits treated as hypotheses).

**Assumptions:** The induced Gibbs/Markov specification lies in a Dobrushin uniqueness (or high-temperature / strong-convexity) regime so that influence coefficients are summable and correlation length is finite.

**Statement:** Local mixing (`Stiffness`) and spectral gap prevent long-range interference. Stochastic dependencies decay exponentially with distance.
$$ \mathrm{Cov}(f(x), g(y)) \leq C e^{-d(x,y)/\xi} $$
This blocks oscillatory failures ("Goldstone modes") and ensures stability.
:::

:::{prf:proof}
**Step 1 (Dobrushin Uniqueness Condition).**
In a spin system (or particle gas), if the interaction between a site and its neighbors is weak enough (high temperature or strong external field/stiffness), the Gibbs measure is unique.

**Step 2 (Decay of Correlations).**
Under the uniqueness condition, influence propagates only locally. The correlation length $\xi$ is finite.
This is proven via coupling arguments or cluster expansion.

**Step 3 (Stability).**
Finite correlation length implies that perturbations at boundary do not affect the bulk state (no "butterfly effect" for static properties). This guarantees that the system is robust to noise and boundary conditions.
:::

:::{prf:metatheorem} Parametric Stiffness Map
:label: mt:parametric-stiffness-map

**Thin inputs:** $\Phi^{\text{thin}}$, $\mathfrak{D}^{\text{thin}}$.
**Permits:** $\mathrm{LS}_\sigma$ (N7), $D_E$ (N1).

**Status:** Heuristic-to-conditional (Bakry–Émery/Lichnerowicz in log-concave settings; requires global convexity on the region of interest).

**Statement:** The local stiffness (spectral gap) of the Fractal Gas is determined by the Hessian of the potential $\Phi$:
$$\lambda_1(x) \geq \inf \text{eig}(\nabla^2 \Phi(x))$$
Regions of high curvature in the optimization landscape correspond to "stiff" regions in the Information Graph where diffusion is suppressed and selection is strong.
:::

:::{prf:proof}
**Step 1 (Bakry-Emery Criterion).**
For a potential $V$, the generator is $L = \Delta - \nabla V \cdot \nabla$.
The Bakry-Emery curvature is bounded below by $\nabla^2 V$.

**Step 2 (Local Spectral Gap).**
By the Lichnerowicz theorem extension, if $\nabla^2 V \succeq K I$, then the local spectral gap $\lambda_1 \ge K$.
In the Fractal Gas, $V = \beta \Phi + \text{entropic terms}$.

**Step 3 (Stiffness interpretation).**
High positive curvature of $\Phi$ implies a deep local minimum. Fluctuations are suppressed by the restoring force $\nabla \Phi$. This corresponds to a "stiff" mode in the mechanical sense, or a large mass term in field theory.
:::

:::{prf:metatheorem} Micro-Macro Consistency
:label: mt:micro-macro-consistency

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $\mathrm{SC}_\lambda$ (N4), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (hydrodynamic/renormalization limits; requires a specified coarse-graining scheme).

**Statement:** The emergent dynamics are consistent across scales. Coarse-graining the microscopic random walk yields the same effective theory as simulating the macroscopic diffusion directly (Commutativity of the diagram).
$$ \mathbb{E}[\pi(\mathcal{S}_{\text{micro}}(x))] = \mathcal{S}_{\text{macro}}(\pi(x)) $$
:::

:::{prf:proof}
**Step 1 (Renormalization Group).**
Let $R_\tau$ be the coarse-graining operator (e.g., block spin or spectral truncation).
We require that the generator $L$ flows to a "renormalized" generator $L'$ such that dynamics are preserved.

**Step 2 (Hydrodynamic Limit).**
The scaling limit of the random walk converges to the diffusion equation.
The coarse-grained variables (e.g., local density) obey a hydrodynamic equation (Euler or Navier-Stokes equivalent for the graph).

**Step 3 (Commutativity).**
If the system is at a fixed point of the RG flow (as established by Dimension Selection), the functional form of the equations is scale-invariant. Thus, simulating at the micro-scale and averaging gives the same statistics as simulating at the macro scale.
:::

:::{prf:metatheorem} Observer Universality
:label: mt:observer-universality

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{TB}_O$ (N9), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic (invariance intuition; the precise “observer group” must be specified).

**Statement:** The Information Graph is intrinsic; it does not depend on the coordinate system or labeling of external states, up to isometry.
$$ \text{IG}(\pi(X)) \cong \text{IG}(X) $$
:::

:::{prf:proof}
**Step 1 (Isometry Invariance).**
The distances in the IG are defined by transition probabilities or mutual information, which are coordinate-independent quantities. $P(j|i)$ depends only on the graph topology and weights.

**Step 2 (Spectral Invariance).**
The spectrum of the Laplacian (Heat Kernel Signature) is an isometry invariant of the manifold. Two observers seeing the same graph will measure the same diffusion times and eigenvalues.

**Step 3 (Reconstruction).**
Since the geometry is reconstructed from spectral properties (see Geometric Reconstruction), any two isometric representations yield the same emergent geometry.
:::

:::{prf:metatheorem} Law Universality
:label: mt:universality-of-laws

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{SC}_\lambda$ (N4), $\mathrm{TB}_O$ (N9).

**Status:** Heuristic-to-conditional (RG/universality is standard in restricted settings; not an automatic consequence of the sieve).

**Statement:** The effective physical laws on the emergent manifold depend only on the universality class of the underlying graph (dimension, symmetries), not on microscopic details.
:::

:::{prf:proof}
**Step 1 (Universality of Random Walks).**
The Central Limit Theorem on manifolds states that long-time behavior of random walks converges to Brownian motion, regardless of the specific microscopic step distribution (provided finite variance).

**Step 2 (Local Operators).**
The effective action must be constructed from local geometric invariants (scalars constructed from curvature, gradient, etc.).
$$ S_{\text{eff}} \sim \int (c_1 |\nabla \phi|^2 + c_2 R \phi^2 + \dots) \sqrt{g} dx $$

**Step 3 (Relevance).**
Under RG flow, irrelevant operators (higher derivatives) suppressed by the scale cut-off vanish. Only relevant and marginal operators remain in the continuum limit. This dictates the form of the macroscopic laws (e.g., standard kinetic terms).
:::

:::{prf:metatheorem} Closure-Curvature Duality
:label: mt:closure-curvature-duality

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $\mathrm{Cap}_H$ (N6).

**Status:** Heuristic (analogy between compactness and geometric regularity; no equivalence theorem is claimed).

**Statement:** The "closure" of the agent's memory or state space (boundedness) induces curvature in the Information Geometry. A finite capacity channel is equivalent to a compact manifold.
:::

:::{prf:proof}
**Step 1 (Compactness).**
If the state space $X$ is compact (finite volume), the spectrum of the Laplacian is discrete.
On an infinite flat space, the spectrum is continuous.

**Step 2 (Curvature bound).**
By Bonnet-Myers theorem, if Ricci curvature is bounded below by a positive constant, the manifold is compact.
Conversely, restricting a random walk to a finite domain induces an effective positive curvature (confinement).

**Step 3 (Information Capacity).**
Finite channel capacity limits the resolution of the state space. This acts as a UV cutoff (discretization) and an IR cutoff (compactness/bounded domain). The geometry of the codebook is necessarily curved (like packing spheres on a surface).
:::

:::{prf:metatheorem} Well-Foundedness Barrier
:label: mt:well-foundedness-barrier

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{TB}_\rho$ (N10).

**Status:** Conditional (a design-time invariant once the multiscale construction is indexed by a well-founded set).

**Statement:** If the Fractal Gas / Fractal Set construction is indexed by a well-founded parameter (e.g. resolution level $n\in\mathbb{N}$ with coarse-graining maps $\phi_{nm}$ only for $n\le m$) and each definition depends only on strictly “finer” or strictly “coarser” levels, then there is no infinite regress: every dependency chain terminates at a base level (atomic inputs or the minimum/maximum resolution used).
:::

:::{prf:proof}
**Step 1 (Metric Ordering).**
We define a partial order on scales/layers based on the coarse-graining parameter $\tau$.
$\tau_{k+1} > \tau_k$.

**Step 2 (Finite Descent).**
Given a finite total capacity and a minimum resolution (uncertainty principle/bit limit), any decomposition of the state space must terminate in a finite number of steps. A strictly decreasing sequence of geometric scales in a compact space must converge to a point (or the minimal lattice scale).

**Step 3 (DAG Structure).**
The dependency graph of the definitions is a Directed Acyclic Graph. There are no circular dependencies in the definition of the state update rules.
:::

:::{prf:metatheorem} Continuum Injection
:label: mt:continuum-injection

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (conditional on a specified manifold-learning embedding and a coupling/limit theorem; “canonical” is generally too strong).

**Statement:** Under standard manifold-learning hypotheses (sampling density, reach/regularity, appropriate kernel bandwidth), there exist embeddings/injections $\iota:V(G)\hookrightarrow M$ (e.g. diffusion maps / heat-kernel embeddings) whose distortion vanishes as resolution increases, so that discrete random-walk paths on $G$ can be coupled to diffusion/geodesic processes on $(M,g)$ in the continuum limit.
$$ \iota: V(G) \hookrightarrow M $$
:::

:::{prf:proof}
**Step 1 (Embedding).**
Use the Heat Kernel embedding or Eigenmaps:
$\iota(x) = (\lambda_1 \phi_1(x), \dots, \lambda_k \phi_k(x))$.
This minimizes distortion of the diffusion metric.

**Step 2 (Shadowing).**
As $N \to \infty$ and $\epsilon \to 0$, the discrete path of a random walker $X_n$ on $G$ can be coupled to a Brownian path $B_t$ on $M$ such that $\sup | \iota(X_n) - B_{nt} | < \delta$ with high probability (Komlós-Major-Tusnády approximation equivalent).

**Step 3 (Injectivity).**
For a sufficiently high embedding dimension $k$ (Whitney embedding theorem), the map $\iota$ is an embedding (injective immersion) for generic graphs.
:::

## Algorithmic Calculus on Fractal Sets

:::{prf:metatheorem} Discrete Stokes' Theorem
:label: mt:discrete-stokes

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $\mathrm{TB}_\pi$ (N8), $\mathrm{Rep}_K$ (N11).

**Status:** Conditional (combinatorial topology; holds for oriented simplicial complexes by definition of boundary/coboundary).

**Statement:** For any discrete $k$-form $\omega$ on a simplicial complex $K$ representing the Information Graph state:
$$\langle d\omega, K \rangle = \langle \omega, \partial K \rangle$$
Flux is invariant under local remeshing (scutoid transitions) provided the cohomology class is preserved.
:::

:::{prf:proof}
**Step 1 (Chain Complex).**
Define $k$-cochains as functions on $k$-simplices. The coboundary operator $d$ is the adjoint of the boundary operator $\partial$.
$$\langle d\omega, c \rangle = \langle \omega, \partial c \rangle$$
This is the definition of the discrete exterior derivative.

**Step 2 (Local Conservation).**
The sum of fluxes out of a volume element equals the creation of "local charge" inside.
$\sum_{F \in \partial V} \text{Flux}(F) = \text{Source}(V)$.

**Step 3 (Topological Invariance).**
Pachner moves (bistellar flips) change the triangulation but preserve the manifold topology. Since $\int_M d\omega = \int_{\partial M} \omega$, and $\partial M$ is unchanged by internal remeshing, the total integral is invariant.
:::

:::{prf:metatheorem} Frostman Sampling Principle
:label: mt:frostman-sampling

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{SC}_\lambda$ (N4), $C_\mu$ (N3).

**Status:** Heuristic-to-conditional (becomes conditional once the limiting/invariant measure is known to be $s$-Frostman; this is not automatic).

**Statement:** If the empirical measures $\mu_N=\frac1N\sum_i\delta_{x_i}$ converge (along a subsequence) to a limit/invariant measure $\mu$ supported on an attractor $A$, and if $\mu$ is $s$-Frostman (upper $s$-regular),
$$ \mu(B_r(x)) \leq C r^s $$
then $s\le \dim_H(A)$ and $\mu$ controls integrals on $A$ via standard potential-theoretic estimates. The “Frostman sampling” interpretation is that, in regimes where the solver concentrates on a fractal attractor with a regular limiting law, such Frostman-type bounds can hold and justify continuum integration on $A$.
:::

:::{prf:proof}
**Step 1 (Occupancy Measure).**
Let $\mu_N = \frac{1}{N} \sum \delta_{x_i}$ be the empirical measure of particles.
The particles concentrate on the minima of $\Phi$, which form a fractal set $A$ (e.g., limit cycle or strange attractor).

**Step 2 (Scaling).**
Due to the self-similar basin structure, the mass in a ball of radius $r$ scales as $N(r) \sim r^{-s}$ (box counting).
Dividing by total mass implies $\mu(B_r) \sim r^s$.

**Step 3 (Energy Capacity).**
The "energy" of the measure $I_s(\mu) = \iint |x-y|^{-s} d\mu(x) d\mu(y)$ is finite, characterizing the set dimension. The dynamics naturally generate a measure maximizing entropy subject to constraint, often coinciding with the measure of maximal dimension.
:::

:::{prf:metatheorem} Genealogical Feynman-Kac
:label: mt:genealogical-feynman-kac

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $D_E$ (N1), $\mathrm{Rep}_K$ (N11).

**Status:** Conditional (Feynman–Kac + branching particle representation under a specified birth/death rule).

**Statement:** The path integral of a potential $\Phi$ is equivalent to the expectation over a branching particle system (the Fractal Gas).
$$ \mathbb{E}_x \left[ f(X_t) \exp\left(-\int_0^t \Phi(X_s) ds\right) \right] = \mathbb{E}_{\text{genealogy}} \left[ \sum_{i=1}^{N_t} f(x_i^t) \right] $$
:::

:::{prf:proof}
**Step 1 (Feynman-Kac Formula).**
The solution to the parabolic PDE $\partial_t u = \frac{1}{2}\Delta u - \Phi u$ is given by the expectation over Brownian paths weighted by $e^{-\int \Phi}$.

**Step 2 (Particle Interpretation).**
In the branching process, particles die at rate $\Phi^+$ and split at rate $\Phi^-$.
The expected number of particles at $x$ at time $t$ satisfies the same PDE.

**Step 3 (Mean Field Limit).**
As $N \to \infty$, the empirical density converges to the solution $u(t,x)$.
The "history" of surviving particles recovers the paths that minimize the action $\int (\frac{1}{2}\dot{x}^2 + \Phi)$.
:::

:::{prf:metatheorem} Cheeger Gradient Isomorphism
:label: mt:cheeger-gradient

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $\mathrm{Rep}_K$ (N11).

**Status:** Conditional (metric-measure / Γ-convergence statement under doubling + Poincaré + graph-limit hypotheses).

**Statement:** The discrete graph gradient $\nabla_G f$ converges to the Cheeger derivation $D f$ on the metric measure space limit.
$$ \| \nabla_G f \|_{L^2} \to \| D f \|_{L^2} $$
This justifies using graph neural networks to compute continuum derivatives.
:::

:::{prf:proof}
**Step 1 (Discrete Gradient).**
$\nabla_{ij} f = \sqrt{w_{ij}} (f(j) - f(i))$.
The Dirichlet energy is $\mathcal{E}(f) = \sum w_{ij} (f(j)-f(i))^2$.

**Step 2 (Relaxed Gradient).**
On a metric measure space $(X, d, \mathfrak{m})$, the modulus of the gradient $|\nabla f|$ is the minimal weak upper gradient.
Cheeger proved that for doubling spaces with Poincaré inequality, a differentiable structure exists.

**Step 3 (Gamma Convergence).**
The sequence of graph energies $\Gamma$-converges to the Cheeger energy on the limit space. Thus, minimizers (harmonic functions) converge.
:::

:::{prf:metatheorem} Anomalous Diffusion Principle
:label: mt:anomalous-diffusion

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{SC}_\lambda$ (N4), $D_E$ (N1).

**Status:** Heuristic-to-conditional (conditional for classes of self-similar fractals / spaces with sub-Gaussian heat-kernel bounds).

**Statement:** On fractal supports with walk dimension $d_w>2$ and sub-Gaussian heat-kernel bounds, diffusion is anomalous: the mean squared displacement scales as
$$ \langle r^2(t) \rangle \sim t^{2/d_w} $$
where $d_w > 2$ is the walk dimension. The heat kernel obeys sub-Gaussian bounds.
:::

:::{prf:proof}
**Step 1 (Spectral Dimension).**
The density of states regularizes as $\rho(\lambda) \sim \lambda^{d_s/2 - 1}$.
The return probability decays as $p_t(x,x) \sim t^{-d_s/2}$.

**Step 2 (Resistance Metric).**
The effective resistance between points at distance $r$ scales as $R(r) \sim r^{d_w - d_H}$.
Time to exit a ball $B_r$ is $T_r \sim r^{d_w}$.

**Step 3 (Sub-Gaussian Kernel).**
The heat kernel bounds are:
$$ p_t(x,y) \asymp \frac{1}{t^{d_s/2}} \exp\left( - \left(\frac{d(x,y)^{d_w}}{t}\right)^{\frac{1}{d_w-1}} \right) $$
This slower-than-Gaussian scaling reflects the "traps" and "mazes" in the fractal geometry.
:::

:::{prf:metatheorem} Spectral Decimation Principle
:label: mt:spectral-decimation

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{SC}_\lambda$ (N4), $\mathrm{Rep}_K$ (N11).

**Status:** Conditional (standard for finitely ramified self-similar graphs; requires a specified decimation scheme).

**Statement:** On self-similar graphs (like Sierpinski gaskets), the Laplacian eigenvalues satisfy a recursive relation $\lambda_{k-1} = R(\lambda_k)$.
The eigenfunctions are fractals themselves. This allows exact computation of the spectrum.
:::

:::{prf:proof}
**Step 1 (Schur Complement).**
Decompose graph nodes into "boundary" and "interior" for a cell.
Project the Laplacian onto boundary nodes: $L_{eff} = L_{bb} - L_{bi} L_{ii}^{-1} L_{ib}$.

**Step 2 (Renormalization Map).**
For self-similar graphs, $L_{eff}(\lambda) \propto L_{original}(R(\lambda))$.
The function $R(\lambda)$ is a rational map governing the spectral flow.

**Step 3 (Eigenvalues).**
The spectrum is the Julia set or attractor of the inverse map $R^{-1}$. This produces the characteristic gaps in the spectrum of fractal operators.
:::

:::{prf:metatheorem} Discrete Uniformization Principle
:label: mt:discrete-uniformization

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{TB}_\pi$ (N8), $C_\mu$ (N3).

**Status:** Conditional (circle packings / discrete conformal geometry under planar triangulation hypotheses).

**Statement:** Any planar triangulation admits a "circle packing" metric that is discretely conformally equivalent to a constant curvature surface (Spherical, Euclidean, or Hyperbolic).
This provides a canonical coordinate system for the Information Graph.
:::

:::{prf:proof}
**Step 1 (Circle Packing).**
Associate a radius $r_i$ to each vertex. Edge lengths are $l_{ij} \approx r_i + r_j$ (tangency).
The discrete conformal factor is $u_i = \log r_i$.

**Step 2 (Discrete Ricci Flow).**
Deform radii to equalize curvatures at vertices (defect angle summing to $2\pi$ or 0).
$\frac{du_i}{dt} = -K_i$.
This flow converges to a unique constant curvature metric (Chow-Luo).

**Step 3 (Approximation).**
The convergence of circle packing maps to the Riemann mapping (He-Schramm theorem) ensures that the discrete coordinates approximate the true conformal structure of the underlying manifold.
:::

:::{prf:metatheorem} Persistence Isomorphism
:label: mt:persistence-isomorphism

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{TB}_\pi$ (N8), $\mathrm{SC}_\lambda$ (N4).

**Status:** Conditional (standard persistent-homology stability for tame filtrations).

**Statement:** The persistent homology of the density sublevel sets calculates the robust topological features of the underlying manifold. The persistence diagram is stable under perturbations (Bottleneck Stability).
:::

:::{prf:proof}
**Step 1 (Filtration).**
Consider the sublevel sets $X_c = \{x : \rho(x) > c\}$.
As $c$ decreases, we get a nested sequence of spaces (filtration).

**Step 2 (Homology).**
Compute homology groups $H_k(X_c)$ across all $c$. Track birth and death of features (connected components, loops).

**Step 3 (Stability).**
The Cohen-Steiner Stability Theorem states that small changes in the function $\rho$ (in $L^\infty$ norm) lead to small changes in the persistence diagram (in bottleneck distance). Thus, the estimated topology is robust to sampling noise.
:::

:::{prf:metatheorem} Swarm Monodromy Principle
:label: mt:swarm-monodromy

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{TB}_\pi$ (N8), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic (requires a precise model of labeled-particle braiding/transport; not a generic topology-recovery theorem).

**Statement:** Heuristically, topology (holes/handles) can be probed by transporting a labeled swarm around loops and recording the induced permutations/braid data of particle clusters.
$\pi_1(M) \to S_N$.
:::

:::{prf:proof}
**Step 1 (Covering Space).**
Consider the configuration space of $N$ distinct points $C_N(M)$.
Paths in $C_N(M)$ correspond to braids.

**Step 2 (Loop Traversal).**
As the center of mass of a swarm moves around a non-contractible loop $\gamma$ in $M$, the individual particles permute positions to minimize energy/maintain spacing.
This defines a homomorphism from $\pi_1(M)$ to the Braid Group (and Symmetric Group).

**Step 3 (Reconstruction).**
By measuring the permutations for various loops, one can reconstruct the fundamental group $\pi_1(M)$ and thus the topological genus of the manifold.
:::

:::{prf:metatheorem} Particle-Field Duality
:label: mt:particle-field-duality

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $D_E$ (N1).

**Status:** Heuristic-to-conditional (empirical-measure weak convergence is standard; PDE/SPDE limits require propagation-of-chaos or mean-field hypotheses).

**Statement:** The discrete particle configuration (Lagrangian) and the continuous probability density (Eulerian) are dual representations.
Weak convergence ensures $\int f d\mu_N \to \int f \rho dx$.
:::

:::{prf:proof}
**Step 1 (Empirical Measure).**
Define mapping $\mathcal{P}: \mathbb{R}^{Nd} \to \mathcal{M}(X)$ by $X \mapsto \frac{1}{N} \sum \delta_{x_i}$.

**Step 2 (SPDE Limit).**
The evolution of the empirical measure follows a stochastic PDE (Dean-Kawasaki equation).
In the limit of large $N$ and appropriate smoothing, the noise term vanishes (Law of Large Numbers), yielding the deterministic Fokker-Planck equation.

**Step 3 (Radon-Nikodym).**
The density field $\rho(x)$ is the Radon-Nikodym derivative of the limit measure w.r.t the reference measure (volume).
:::

:::{prf:metatheorem} Cloning Transport Principle
:label: mt:cloning-transport

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $\mathrm{Rep}_K$ (N11), $D_E$ (N1).

**Status:** Heuristic-to-conditional (becomes conditional for classical multiplicative Feynman–Kac weights).

**Statement:** Reweighting/cloning of particles along a path acts as parallel transport of the normalization factor. This defines a connection on the line bundle of densities.
:::

:::{prf:proof}
**Step 1 (Multiplicative Cocycles).**
Let $W(x, y) = e^{-\beta (\Phi(y) - \Phi(x))}$ be the weight change.
Along a path $\gamma$, the cumulative weight is $W(\gamma) = \prod W(x_i, x_{i+1}) = e^{-\beta \Delta \Phi}$.

**Step 2 (Connection).**
This has the form of a parallel transport with connection 1-form $A = \beta d\Phi$.
If the potential is not single-valued (e.g., in non-conservative fields or with magnetic terms), the curvature $F = dA$ is non-zero, leading to holonomy.

**Step 3 (Bundle Section).**
The particle density is a section of a line bundle. The cloning process "transports" the density value from one tangent space to another, adjusting for local potential changes.
:::

## Fractal Gas as Computation, Quantum, and Information Engine

:::{prf:metatheorem} Projective Feynman-Kac Isomorphism
:label: mt:projective-feynman-kac

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $\mathrm{TB}_\rho$ (N10), $\mathrm{LS}_\sigma$ (N7).

**Status:** Conditional (classical Feynman–Kac normalization; the implemented pairwise selection is an approximation to this envelope).

**Statement:** The nonlinear evolution of the normalized probability density (Fractal Gas) is equivalent to the linear evolution of the unnormalized Feynman-Kac semigroup, projected onto the unit simplex.
:::

:::{prf:proof}
**Step 1 (Linear Evolution).**
Let $v_t$ boundedly solve $\partial_t v = \mathcal{L} v - \beta \Phi v$ (linear sink/source term).

**Step 2 (Projection).**
Let $u_t = \frac{v_t}{\|v_t\|_1}$.
Differentiation gives the nonlinear equation:
$\partial_t u = \mathcal{L} u - \beta \Phi u + (\beta \int \Phi u) u$.
The last term is the mean field feedback maintaining normalization.

**Step 3 (Metric Contraction).**
The map is a contraction in the Hilbert projective metric (Birkhoff's theorem), ensuring unique relaxation to the strictly positive ground state (Perron-Frobenius eigenvector).
:::

:::{prf:principle} Fisher Information Ratchet
:label: prin:fisher-information-ratchet

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $D_E$ (N1).

**Status:** Heuristic-to-conditional (information-geometry interpretation; no monotonicity law is asserted without extra hypotheses).

**Statement:** In idealized continuum limits, the Fractal Gas can be viewed as balancing:
- **Mutation/diffusion:** which typically *reduces* Fisher information and *increases* entropy (e.g. heat flow / de Bruijn identity), and
- **Selection/cloning:** which reweights toward lower $\Phi$ and tends to *concentrate* mass (often increasing Fisher information by sharpening the distribution).

This “ratchet” is an interpretation of mutation–selection/replicator dynamics as a flow on a statistical manifold (natural-gradient / Wasserstein–Fisher–Rao viewpoints) rather than a claim that $\frac{d}{dt}\mathcal{I}(\rho_t)$ has a fixed sign in general.
:::

:::{prf:proof}
**Step 1 (Fisher Information Rate).**
For the Fokker-Planck equation, the time derivative of the entropy is related to the Fisher Information (De Bruijn's identity).
$\frac{dH}{dt} = \mathcal{I}(\rho)$.

**Step 2 (Selection Pressure).**
The selection term $\mathcal{R}[\rho]\rho$ drives the distribution towards the minimum of potential $\Phi$. This sharpens the distribution, initially increasing Fisher Information (inverse variance).

**Step 3 (Geodesic Motion).**
The replicator-mutator dynamics can be viewed as a gradient flow of the KL-divergence on the Wasserstein-Fisher-Rao manifold. This path minimizes the kinetic cost of information gain.
:::

:::{prf:principle} Complexity Tunneling
:label: prin:complexity-tunneling

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $D_E$ (N1), $\mathrm{SC}_\lambda$ (N4), $\mathrm{LS}_\sigma$ (N7).

**Status:** Heuristic-to-conditional (importance-splitting intuition; not a generic worst-case complexity theorem).

**Statement:** For metastable landscapes with barrier height $\Delta E$, pure diffusion or local descent can exhibit barrier-crossing times scaling like $e^{c\,\Delta E}$ (large deviations). Cloning/selection can *in some regimes* reduce effective crossing times by implementing an importance-splitting / rare-event amplification mechanism (allocating more particles to transition regions). Quantitative speedups require explicit assumptions on the landscape (reaction coordinate, bottleneck structure) and the resampling rule; no uniform “polynomial in $\Delta E$” guarantee is asserted here.
:::

:::{prf:proof}
**Step 1 (Rare Event Sampling).**
Crossing a high barrier is a large deviation event with probability $P \sim e^{-\beta \Delta E}$.
Waiting for a single particle to cross takes time $T \sim 1/P$.

**Step 2 (Cloning Advantage).**
If particles near the barrier top are cloned, the probability mass there is artificially boosted. With an optimal cloning schedule (importance splitting), the number of clones needed grows only linearly or polynomially with barrier height.

**Step 3 (Tunneling Time).**
The effective time to transport mass across the barrier becomes $T_{eff} \sim \text{poly}(\Delta E)$. This is the computational analogue of quantum tunneling.
:::

:::{prf:metatheorem} Landauer Optimality
:label: mt:landauer-optimality

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$, $\mathfrak{D}^{\text{thin}}$.
**Permits:** $D_E$ (N1), $\mathrm{Cap}_H$ (N6), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic (Landauer is a physical bound; applying it to an abstract solver requires an explicit physical implementation model).

**Statement:** If a physical implementation of the Fractal Gas at temperature $T$ performs an information-erasing operation reducing entropy by $\Delta I$ bits, then the minimal dissipated work satisfies $W\ge k_B T\ln 2\cdot \Delta I$. Interpreting $\Delta I$ as a mutual-information gain between an initial state and an identified optimum yields the schematic bound
$$E_{\text{search}} \geq k_B T \ln 2 \cdot I(x_{\text{start}}; x_{\text{opt}})$$
Saturation requires quasi-static reversible driving and is not asserted for generic algorithmic runs.
:::

:::{prf:proof}
**Step 1 (Information Erasure).**
Selecting the optimal state $x_{\text{opt}}$ from a prior distribution reduces entropy by $\Delta S = H_{\text{prior}} - H_{\text{final}}$.
Landauer's principle requires heat dissipation $Q \geq T \Delta S$.

**Step 2 (Free Energy).**
The Fractal Gas minimizes free energy $F = U - TS$. The work done by the selection mechanism equals the change in free energy.
Efficiency is measured by proximity to this bound.

**Step 3 (Reversibility).**
If the definition of fitness changes slowly (quasi-static), the cloning/killing is reversible (detailed balance holds), and the system saturates the Landauer bound.
:::

:::{prf:metatheorem} Levin Search Isomorphism
:label: mt:levin-search

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $D_E$ (N1).

**Status:** Heuristic-to-conditional (the Gibbs-form identity is conditional once a program prior and time functional are specified; the “Levin optimality” interpretation is heuristic).

**Statement:** On a countable program space, if one defines a potential
$$
\Phi(p)=\ln 2\cdot \mathrm{Length}(p)+\ln \mathrm{Time}(p),
$$
then the formal Gibbs weight is
$$N(p) \propto 2^{-\mathrm{Length}(p)} \cdot \text{Time}(p)^{-1}$$
which matches the allocation pattern of Levin-style universal search. Making this rigorous requires specifying the reference measure on program space, the computability/approximation scheme for $\mathrm{Time}(p)$, and a normalizability regime (so $Z<\infty$).
:::

:::{prf:proof}
**Step 1 (Universal Prior).**
Solomonoff induction assigns probability $P(x) \approx 2^{-K(x)}$ to string $x$.
Levin search allocates time $t_p \propto P(p) / \text{Time}(p)$ to testing program $p$.

**Step 2 (Equilibrium).**
The stationary distribution of the Fractal Gas is $\rho \propto e^{-\Phi}$.
With $\Phi = L \ln 2 + \ln T$, we get $\rho \propto 2^{-L} T^{-1}$.

**Step 3 (Optimality).**
Levin search is optimal strictly (within a constant factor) for inversion problems. The Fractal Gas implements this allocation naturally via its energetic cost function.
:::

:::{prf:principle} Algorithmic Tunneling
:label: prin:algorithmic-tunneling

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $\mathrm{TB}_\pi$ (N8), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic (Kolmogorov complexity is uncomputable; any practical proxy is model-dependent).

**Statement:** The (uncomputable) algorithmic information distance $d_K(x,y)=K(x|y)+K(y|x)$ defines a geometry where “close” means “easily computable from one another”. A Fractal-Gas-like solver can be interpreted as diffusing in *approximations* of this geometry (program-edit graphs, compression-based distances), enabling “tunneling” between conceptually related but structurally distinct solutions when Euclidean parameter distances are misleading.
:::

:::{prf:proof}
**Step 1 (Algorithmic Geometry).**
Programs are nodes in a graph; edges are simple edits. The graph distance approximates $d_K$.

**Step 2 (Mutation).**
Random mutations in program space correspond to diffusion.
Regions connected by short programs (common subroutines) are close.

**Step 3 (Shortcuts).**
What appears as a high energetic barrier in the parameter space of a neural network (Euclidean distance) might be a short hop in program space (e.g., changing one hyperparameter or rule). The Fractal Gas explores these "wormholes."
:::

:::{prf:metatheorem} Cloning-Lindblad Equivalence
:label: mt:cloning-lindblad

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $D_E$ (N1).

**Status:** Heuristic (quantum open-system language; at best an analogy to master equations for classical interacting particle systems).

**Statement:** Heuristically, cloning/death can be viewed as coupling the system to an “environment” that implements dissipation/selection. One can write master-equation evolutions for ensembles and draw analogies to GKSL/Lindblad structure in quantum mechanics, but this sketch does not assert that Fractal Gas dynamics literally define a Lindblad evolution on a quantum density matrix.
$$ \frac{d\rho}{dt} = -i[H, \rho] + \sum (2 L_k \rho L_k^\dagger - \{L_k^\dagger L_k, \rho\}) $$
:::

:::{prf:proof}
**Step 1 (Jump Operators).**
Identify the cloning events as quantum jumps. The "jump operator" $L$ creates a copy.
The rate of jumping depends on the potential $\Phi$.

**Step 2 (Master Equation).**
Write the Master equation for the particle number distribution.
In the large $N$ limit, the density matrix $\rho$ of the single-particle state evolves according to the Lindblad form.

**Step 3 (Dissipation).**
The "environment" is the fitness landscape. It continuously measures the system (selecting for fitness), causing decoherence in the energy basis (collapse to minima).
:::

:::{prf:principle} Zeno Effect
:label: prin:zeno-effect

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $C_\mu$ (N3).

**Status:** Heuristic (quantum analogy; the cloning operator is not literally a projective measurement).

**Statement:** Frequent cloning (measurement) confines the system to a subspace (the ground state). If the cloning rate $\gamma$ is large compared to the diffusion rate, the system is "frozen" in the optimal state.
:::

:::{prf:proof}
**Step 1 (Measurement Projection).**
Each cloning event acts as a weak measurement of position (soft projection).
High cloning rate $\to$ Continuous surveillance.

**Step 2 (Survival Probability).**
The probability of transitioning out of the ground state decays as $1 - (\Delta E)^2 t^2$.
With frequent checks at interval $\tau$, the survival prob at time $T$ is $(1 - (\tau \Delta E)^2)^{T/\tau} \approx e^{-T \tau (\Delta E)^2}$.
As $\tau \to 0$, decay is suppressed.

**Step 3 (Optimization).**
This "Quantum Zeno Effect" prevents the optimizer from drifting away from a sharp minimum once found, provided the "observation" (gradient/fitness check) is frequent enough.
:::

:::{prf:principle} Importance Sampling Isomorphism
:label: prin:importance-sampling

**Thin inputs:** $\Phi^{\text{thin}}$, $\mathfrak{D}^{\text{thin}}$.
**Permits:** $\mathrm{SC}_\lambda$ (N4), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (standard importance sampling statement for *known* integrands; cloning provides an adaptive approximation).

**Statement:** For estimating an integral $\int f(x)\,dx$ with $f$ known, the zero-variance proposal is $q(x)\propto |f(x)|$. In Fractal-Gas-like interacting particle systems, resampling/cloning adaptively concentrates particles in regions with high estimated contribution to observables, which can be interpreted as learning an approximate importance distribution (e.g. Gibbs-like densities $e^{-\beta\Phi}$ in equilibrium regimes). This is an interpretation of the *variance-reduction role* of cloning, not a claim that the stationary law is exactly optimal for all observables.
:::

:::{prf:proof}
**Step 1 (Zero Variance Condition).**
The optimal proposal distribution $q(x)$ for estimating $\int f(x) dx$ is $q(x) \propto |f(x)|$.
Here we want to estimate the "ground state energy" or normalization $Z$.

**Step 2 (Self-Organized Criticality).**
The cloning dynamics naturally adjust population density until it matches the "value" of the region (proportional to contribution to the integral).
If a region is under-sampled, the high weight causes branching; if over-sampled, killing.

**Step 3 (Equilibrium).**
At steady state, the number of walkers $N(x)$ is exactly proportional to the importance weight, yielding a zero-variance estimator for observables.
:::

:::{prf:metatheorem} Epistemic Flow
:label: mt:epistemic-flow

**Thin inputs:** $\Phi^{\text{thin}} = -\mathcal{U} (Uncertainty)$.
**Permits:** $D_E$ (N1), $\mathrm{Cap}_H$ (N6).

**Status:** Heuristic-to-conditional (becomes conditional once “uncertainty” is a specified smooth functional and the induced dynamics are identified).

**Statement:** If the potential is chosen as $\Phi(x)=-\mathcal{U}(x)$ for a specified epistemic-uncertainty functional $\mathcal{U}$, then (in idealized continuum limits where the drift is $-\nabla\Phi$) the swarm drifts toward regions of high uncertainty and can be interpreted as maximizing an information-gain proxy. The “knowledge boundary” language is interpretive and depends on the statistical model used to define $\mathcal{U}$.
:::

:::{prf:proof}
**Step 1 (Information Gain).**
The expected information gain from sampling $x$ is related to the epistemic uncertainty (entropy of the posterior predictive).
$IG(x) \approx H(y|x, \mathcal{D})$.

**Step 2 (Gradient Flow).**
Dynamics $\dot{x} = -\nabla \Phi = \nabla IG$ drive agents towards high uncertainty.
Diffusion ensures exploration of multiple uncertainty modes.

**Step 3 (Space Filling).**
As regions are sampled, uncertainty decreases (potential rises). The swarm flows like a liquid filling a vessel, leveling out the uncertainty landscape (flattening the posterior).
:::

:::{prf:principle} Curriculum Generation
:label: prin:curriculum-generation

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $\mathrm{SC}_\lambda$ (N4).

**Status:** Heuristic (learning/optimization design principle).

**Statement:** The sequence of datasets $\{\mathcal{D}_t\}$ generated by the Fractal Gas constitutes an optimal curriculum. The effective temperature $T(t)$ acts as a spectral filter, admitting low-frequency (easy) patterns first and high-frequency (detail) patterns later.
:::

:::{prf:proof}
**Step 1 (Spectral Bias).**
Neural networks learn low-frequency functions faster.
The Fractal Gas at high temperature samples broadly (low frequencies).

**Step 2 (Annealing).**
As parameters converge (cooling), the "resolution" of the sampler increases (Dimension Selection).
The swarm focuses on finer details of the landscape (high frequencies).

**Step 3 (Matched Filtering).**
The distribution of training examples $P_t(x)$ evolves such that the "difficulty" of samples matches the current capacity of the learner, maximizing learning progress (the flow of gradients).
:::

:::{prf:metatheorem} Manifold Sampling Isomorphism
:label: mt:manifold-sampling

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{Rep}_K$ (N11), $\mathrm{SC}_\lambda$ (N4).

**Status:** Heuristic-to-conditional (conditional on a manifold hypothesis + a separation of normal/tangent scales).

**Statement:** If the solution set lies on a low-dimensional manifold $\mathcal{M} \subset \mathbb{R}^D$, the Fractal Gas naturally concentrates on $\mathcal{M}$, reducing the effective search dimension from $D$ to $d_{\text{intrinsic}}$.
:::

:::{prf:proof}
**Step 1 (Dimensional Collapse).**
High-energy regions are exponentially suppressed. If the potential wells are aligned with $\mathcal{M}$, the transverse fluctuations are Gaussian bounded.

**Step 2 (Tangent Space Dynamics).**
The random walk effectively occurs on the tangent bundle $T\mathcal{M}$.
The diffusion coefficient in the normal direction scales to zero (or is confined).

**Step 3 (Complexity Reduction).**
The sample complexity depends on the covering number of $\mathcal{M}$, not the volume of the ambient space. $N \sim (1/\epsilon)^d \ll (1/\epsilon)^D$.
:::

## Physics Emergence Theorems

This section is primarily **interpretative**: it records correspondences between Fractal Gas / Information-Graph language and ideas from physics. Unless a block below explicitly declares a rigorous `Status` and hypotheses, treat it as **Heuristic** and do not use it as a certificate input.

### Quantum Foundations

:::{prf:metatheorem} Hessian-Metric Isomorphism
:label: mt:hessian-metric

**Thin inputs:** $\Phi^{\text{thin}}$.
**Permits:** $\mathrm{LS}_\sigma$ (N7), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (Fisher-information identities are standard; interpreting them as “gravitational metrics” is heuristic).

**Statement:** For a Gibbs family $\rho_\theta(x)=Z(\theta)^{-1}e^{-\Phi(x;\theta)}$ satisfying standard regularity conditions (differentiate under the integral sign; finite moments), the Fisher information metric on parameter space satisfies
$$
g_{\mu\nu}(\theta)=\mathbb{E}_{\rho_\theta}\!\left[\partial_\mu \log\rho_\theta\,\partial_\nu \log\rho_\theta\right]
=\mathrm{Cov}_{\rho_\theta}\!\left(\partial_\mu \Phi,\partial_\nu \Phi\right)
=\partial_\mu\partial_\nu\log Z(\theta).
$$
In Laplace/quadratic regimes (sharp concentration), this metric is controlled by second-order curvature data of $\Phi$ near dominant modes. Any identification of $g_{\mu\nu}$ with a spacetime/gravity metric is an analogy, not a generic theorem of the framework.
:::

:::{prf:proof}
**Step 1 (Equilibrium Distribution).**
$\rho_\theta(x) = \frac{1}{Z} e^{-\Phi(x; \theta)}$.

**Step 2 (Fisher Metric).**
$\log\rho_\theta(x)=-\Phi(x;\theta)-\log Z(\theta)$, so
$$
\partial_\mu \log\rho_\theta(x)= -\partial_\mu \Phi(x;\theta)+\mathbb{E}_{\rho_\theta}[\partial_\mu \Phi(\cdot;\theta)].
$$
Hence
$$
g_{\mu\nu}=\mathbb{E}_{\rho_\theta}\!\left[(\partial_\mu \log\rho_\theta)(\partial_\nu \log\rho_\theta)\right]
=\mathrm{Cov}_{\rho_\theta}\!\left(\partial_\mu \Phi,\partial_\nu \Phi\right).
$$

**Step 3 (Equivalence).**
Under the same regularity assumptions one also has $g_{\mu\nu}=\partial_\mu\partial_\nu\log Z(\theta)$. In Laplace/quadratic approximations, $\log Z$ is controlled by local second derivatives of $\Phi$ around dominant modes, which motivates the heuristic “Hessian” language in sharply concentrated regimes.
:::

:::{prf:metatheorem} Symmetry-Gauge Correspondence
:label: mt:symmetry-gauge

**Thin inputs:** $G^{\text{thin}}$.
**Permits:** $\mathrm{GC}_\nabla$ (N12), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (standard gauge principle in field theory; mapping it to a specific solver requires a specified update rule).

**Statement:** In settings where local symmetry transformations are treated as a covariance requirement, discrete derivatives on a graph must be replaced by covariant differences involving parallel transporters/connection variables. This is the standard “gauge principle” pattern; whether a given Fractal Gas instantiation actually implements it depends on its concrete update rule.
:::

:::{prf:proof}
**Step 1 (Local Symmetry).**
Let the state $\psi(x)$ transform as $\psi'(x) = U(x) \psi(x)$.
The graph derivative $\nabla_G \psi = \psi(y) - \psi(x)$ is not covariant.

**Step 2 (Covariant Derivative).**
Introduce the parallel transporter $U_{xy} \in G$.
$D \psi = U_{xy} \psi(y) - \psi(x)$.
Under transformation, $U_{xy} \to U(x) U_{xy} U(y)^\dagger$.

**Step 3 (Constraint).**
To maintain consistent dynamics, minimizing the energy requires optimizing over these connection fields $A \sim U - I$. This implies the dynamics of gauge fields emerge from the requirement of local optimization independence.
:::

:::{prf:metatheorem} Three-Tier Gauge Hierarchy
:label: mt:three-tier-gauge

**Thin inputs:** $G^{\text{thin}}$.
**Permits:** $\mathrm{GC}_\nabla$ (N12), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic (speculative identification; no derivation of Standard Model gauge groups is claimed).

**Statement:** Heuristically, different layers of symmetry in the solver (normalization/phase-like invariances, orientation-like symmetries, and clustering/permutation structure) invite analogies to $U(1)$/$SU(2)$/$SU(3)$-type gauge organization. This is included as a correspondence map, not as a theorem that the Standard Model gauge structure is forced by the framework.
:::

:::{prf:proof}
**Step 1 (Phase).**
The wavefunction $\psi$ is complex; phase rotation $e^{i\theta}$ is a $U(1)$ symmetry.

**Step 2 (Orientation).**
On a Riemannian manifold, the tangent bundle $O(d)$ reduces to spin structure $Spin(d)$. For 3D/4D effective spacetime, this yields weak isospin-like $SU(2)$ symmetry.

**Step 3 (Clustering).**
Local permutations of indistinguishable nodes in a cluster (the "quarks" of the graph) generate $S_3$ or embedded $SU(3)$ symmetries in the strong coupling limit.
:::

:::{prf:metatheorem} Antisymmetry-Fermion Theorem
:label: mt:antisymmetry-fermion

**Thin inputs:** $G^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $\mathrm{Rep}_K$ (N11), $\mathrm{TB}_\pi$ (N8).

**Status:** Heuristic (antisymmetric structures suggest symplectic/fermionic formalisms in some representations; no “must be fermionic” implication is claimed).

**Statement:** Antisymmetric couplings ($w_{ij}=-w_{ji}$) naturally encode oriented flows/currents and can be represented using symplectic or Pfaffian/Grassmann formalisms in some path-integral constructions. This sketch uses the “fermion” analogy as an organizing intuition for antisymmetric interaction structure; it is not a theorem that antisymmetric graph weights force a fermionic QFT.
:::

:::{prf:proof}
**Step 1 (Path Integral).**
Bosonic integrals require positive definite quadratic forms ($\int e^{-xAx}$).
Antisymmetric forms vanish or are ill-defined for commuting variables.

**Step 2 (Grassmann Variables).**
For anticommuting variables $\theta$, $\int e^{-\theta^T A \theta} = \text{Pf}(A) = \sqrt{\det A}$.
This is non-zero for antisymmetric $A$.

**Step 3 (Exclusion Principle).**
The anticommutation implies $\psi(x)^2 = 0$, enforcing the Pauli exclusion principle. Directed flows in the IG (currents) naturally map to fermions.
:::

:::{prf:metatheorem} Scalar-Reward Duality (Higgs Mechanism)
:label: mt:scalar-reward-duality

**Thin inputs:** $\Phi^{\text{thin}}$, $G^{\text{thin}}$.
**Permits:** $\mathrm{LS}_\sigma$ (N7), $\mathrm{SC}_{\partial c}$ (N5).

**Status:** Heuristic (analogy to symmetry-breaking patterns; not a derived field-theory statement).

**Statement:** The potential field $\Phi$ can be viewed as a scalar “order parameter” whose minima structure induces symmetry-breaking patterns; in this analogy, “mass generation” corresponds to increased stiffness/spectral gap in certain directions once a symmetry is broken. This is a correspondence, not a claim that a Higgs mechanism is literally implemented.
:::

:::{prf:proof}
**Step 1 (Mexican Hat).**
If the potential $\Phi(\phi) = \lambda(|\phi|^2 - v^2)^2$ has a non-zero minimum $v$.
The ground state is $\phi_0 = v$.

**Step 2 (Expansion).**
Expanding around the vacuum $\phi = v + h$, the covariant derivative term $|D_\mu \phi|^2$ generates mass terms $\frac{1}{2} (gv)^2 A_\mu^2$ for the gauge field $A$.

**Step 3 (Optimization Stiffness).**
Physically, this means the solver becomes "stiff" or "massive" in certain directions—deviations cost high energy. This constrains the search to a sub-manifold.
:::

:::{prf:metatheorem} IG-Quantum Isomorphism
:label: mt:ig-quantum-isomorphism

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $\mathrm{LS}_\sigma$ (N7), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (OS reconstruction is conditional on reflection positivity and Euclidean invariance; these are not guaranteed by generic IG dynamics).

**Statement:** If an IG-based continuum limit defines Euclidean correlation functions satisfying the Osterwalder–Schrader axioms (notably reflection positivity and Euclidean invariance), then the OS reconstruction theorem yields a corresponding Lorentzian QFT. This sketch does not assert that generic Fractal Gas dynamics satisfy OS axioms; it records the conditional bridge if such axioms can be verified in a specific instantiation.
:::

:::{prf:proof}
**Step 1 (Reflection Positivity).**
The transition kernel $P_t$ is symmetric and positive definite (reversible Markov chain).
$\langle \theta f, P_t f \rangle \ge 0$. This ensures a valid Hilbert space upon quantization.

**Step 2 (Euclidean Invariance).**
The emergent manifold $M$ has $SO(d)$ symmetry locally. The limit correlations depend only on geodesic distance.

**Step 3 (Wick Rotation).**
By the reconstruction theorem, there exists a unique QFT whose Schwinger functions match the IG correlations with $t \to it$. The Fractal Gas *is* a Euclidean QFT.
:::

:::{prf:metatheorem} Spectral Action Principle
:label: mt:spectral-action-principle

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $G^{\text{thin}}$.
**Permits:** $\mathrm{SC}_\lambda$ (N4), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic (noncommutative-geometry correspondence; requires a fully specified spectral triple and scaling regime).

**Statement:** In noncommutative-geometry settings with a specified spectral triple $(\mathcal{A},\mathcal{H},D)$, one can define a spectral action $\mathrm{Tr}(f(D/\Lambda))$ whose heat-kernel expansion produces curvature invariants. The “reproduces Einstein–Hilbert + Standard Model” claim is specific to particular spectral triples and is not derived here for generic Information Graph constructions.
:::

:::{prf:proof}
**Step 1 (Dirac Operator).**
Construct the Dirac operator $D$ from the graph Laplacian and spin connection. $D^2 \approx \Delta$.

**Step 2 (Heat Kernel Expansion).**
$\mathrm{Tr}(e^{-t D^2}) \sim \frac{1}{t^{d/2}} \sum a_n t^n$.
The coefficients $a_n$ are geometric invariants: Volume, Scalar Curvature $R$, $R^2$, etc.

**Step 3 (Physical Lagrangian).**
For a cutoff function $f$, $\mathrm{Tr}(f(D/\Lambda)) \approx \Lambda^4 \text{Vol} + \Lambda^2 \int R + \dots$
The dominant terms are Cosmological Constant and Gravity. Lower terms govern gauge couplings.
:::

:::{prf:metatheorem} Geometric Diffusion Isomorphism
:label: mt:geometric-diffusion-isomorphism

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $\mathrm{Cap}_H$ (N6), $\mathrm{LS}_\sigma$ (N7), $\mathrm{Rep}_K$ (N11).

**Status:** Conditional (graph Laplacian convergence in manifold-learning regimes; heat-kernel asymptotics are classical once $(M,g)$ is identified).

**Statement:** The graph Laplacian $\Delta_G$ converges to the Laplace-Beltrami operator $\Delta_M$. The heat kernel expansion recovers the geometry:
$$\mathrm{Tr}(e^{-t\Delta}) \sim \frac{\mathrm{Vol}(M)}{(4\pi t)^{d/2}} \left(1 + \frac{t}{6} S_R + O(t^2)\right)$$
:::

:::{prf:proof}
**Step 1 (Pointwise Convergence).**
For smooth functions, $\Delta_G f \to \Delta_M f + \text{drift}$. The drift vanishes for specific weight choices (diffusion maps).

**Step 2 (Spectral Convergence).**
Eigenvalues and eigenfunctions converge appropriately (Mosco).

**Step 3 (Trace Formula).**
The short-time asymptotics of the trace of the heat kernel are determined by local geometry (Minakshisundaram-Pleijel). The first term gives dimension and volume; the second gives scalar curvature $S_R$.
:::

:::{prf:metatheorem} Spectral Distance Isomorphism
:label: mt:spectral-distance-isomorphism

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (true for commutative spectral triples under standard hypotheses; depends on how $D$ is constructed from the IG).

**Statement:** For commutative spectral triples associated to a smooth compact Riemannian manifold, the Connes spectral distance recovers the geodesic distance. Whether an IG-constructed Dirac operator achieves this in a given discretization is conditional on the spectral-triple construction and convergence regime.
$$ d_D(x,y) = \sup_{f: \|[D,f]\| \le 1} |f(x) - f(y)| = d_{\text{geo}}(x,y) $$
:::

:::{prf:proof}
**Step 1 (Gradient Constraint).**
The condition $\|[D,f]\| \le 1$ is equivalent to $|\nabla f| \le 1$ almost everywhere.

**Step 2 (Monge-Kantorovich).**
The formula is a dual formulation of the Wasserstein distance $W_1$ between delta measures.
For geodesic spaces, $W_1(\delta_x, \delta_y) = d_{\text{geo}}(x,y)$.

**Step 3 (Recovery).**
Thus, knowing the spectrum and algebra of functions allows reconstructing the metric space metric exactly.
:::

:::{prf:metatheorem} Dimension Spectrum
:label: mt:dimension-spectrum

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{SC}_\lambda$ (N4), $\mathrm{Cap}_H$ (N6).

**Status:** Heuristic-to-conditional (dimension spectrum is a noncommutative-geometry notion; relation to Hausdorff dimension is model-dependent for fractals).

**Statement:** In noncommutative geometry one defines a dimension spectrum via poles of $\zeta_D(s)=\mathrm{Tr}(|D|^{-s})$ for an appropriate Dirac-type operator. In commutative smooth settings the leading pole recovers the manifold dimension; for fractal/singular spaces the relation to Hausdorff dimension is conditional on the chosen spectral triple and regularity assumptions.
:::

:::{prf:proof}
**Step 1 (Zeta Function).**
Define $\zeta(s) = \sum \lambda_k^{-s/2}$.
For a manifold of dimension $d$, $\lambda_k \sim k^{2/d}$.
Thus the sum diverges when $s/2 \cdot 2/d = 1 \implies s=d$.

**Step 2 (Complex Poles).**
For fractals, the zeta function has complex poles $d + in p$. These correspond to the oscillatory terms in the volume growth (lacunarity) and reflect the discrete scaling symmetry.

**Step 3 (Universality).**
The dimension spectrum characterizes the universality class of the fractal substrate independent of coordinates.
:::

### Spacetime and Gravity

:::{prf:metatheorem} Scutoidal Interpolation
:label: mt:scutoidal-interpolation

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{TB}_\pi$ (N8), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (Pachner-move connectivity is conditional; the “causal foliation” interpretation requires extra structure).

**Statement:** Conditional on working with triangulations of a fixed manifold class, any two triangulations are related by a finite sequence of bistellar (Pachner) moves. Interpreting these local remeshings as “scutoid” transitions provides a discrete interpolation picture; ensuring a well-defined **causal** foliation requires additional input beyond pure topology (time-slicing, admissible moves, and compatibility with the update rule).
:::

:::{prf:proof}
**Step 1 (Pachner Moves).**
Any two triangulations of the same manifold are related by a finite sequence of Pachner moves (e.g., 2-3 flip, 1-4 flip).

**Step 2 (Spacetime Prism).**
The transition corresponds to filling the prism $M \times [0,1]$ with simplices.
A Pachner move is the cross-section of a higher-dimensional simplex (4-simplex in 3+1D).

**Step 3 (Scutoid Geometry).**
The intermediate states (frustra) are scutoids—geometric interpolants that preserve connectedness and Euler characteristic while changing local connectivity.
:::

:::{prf:metatheorem} Regge-Scutoid Dynamics
:label: mt:regge-scutoid

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $D_E$ (N1), $\mathrm{TB}_\pi$ (N8).

**Status:** Heuristic (Regge-calculus analogy; no variational principle for the implemented solver is asserted).

**Statement:** Heuristically, certain rewiring/remeshing dynamics can be compared to Regge-calculus moves that redistribute curvature concentration in a simplicial complex. This sketch does not assert that the Information Graph optimizer literally minimizes a Regge action; it records the analogy that local topology changes can relieve geometric “defects” in a discrete curvature proxy.
$$ S_{\text{Regge}} = \sum_h L_h \epsilon_h \to \min $$
:::

:::{prf:proof}
**Step 1 (Regge Action).**
For a simplicial complex, the curvature is concentrated at hinges (bones). $\epsilon_h = 2\pi - \sum \theta_i$.
The action is proportional to the integral of scalar curvature.

**Step 2 (Flip Energy).**
A bistellar flip changes the edge lengths and connectivity.
If a region has high defect (curvature), a flip can reducing the action.
Example: Flipping a diagonal in a non-convex quadrilateral relaxes tension.

**Step 3 (Relaxation).**
The Fractal Gas dynamics (diffusion + rewiring) effectively performs Metropolis-Hastings sampling of the Regge action, driving the geometry towards flatness (or constant curvature vacuum).
:::

:::{prf:metatheorem} Bio-Geometric Isomorphism
:label: mt:bio-geometric-isomorphism

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $G^{\text{thin}}$.
**Permits:** $\mathrm{Rep}_K$ (N11), $\mathrm{Cap}_H$ (N6).

**Status:** Heuristic (analogy between tissue vertex models and graph rewiring; no biological claim is proved).

**Statement:** Biological cell division and scutoid transitions correspond to geometric surgeries on the Fractal Set. The topological operations of cytokinesis are isomorphic to handle attachment or vertex splitting in the Information Graph.
$$ \text{Biology}(\text{Mitosis}) \cong \text{Geometry}(\text{Surgery}) $$
:::

:::{prf:proof}
**Step 1 (Tissue mechanics).**
Epithelial tissues minimize surface energy. Cell shape changes (T1 transitions) are driven by stress relaxation.

**Step 2 (Information Geometry).**
In the Fractal Gas, "cells" are clusters of particles (Voronoi regions).
High local error (stress) triggers cloning (cell division).

**Step 3 (Isomorphism).**
The rules for updating the graph topology during cloning (adding a node, splitting edges) are combinatorially identical to the rules for cell division in a vertex model. Thus, the morphogenesis of the swarm follows the same geometric laws as biological tissues.
:::

:::{prf:metatheorem} Antichain-Surface Correspondence
:label: mt:antichain-surface

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{TB}_\pi$ (N8), $\mathrm{Cap}_H$ (N6).

**Status:** Heuristic-to-conditional (becomes conditional under faithful-embedding/sprinkling hypotheses from causal set theory).

**Statement:** In causal set theory, under hypotheses of faithful embedding (e.g. Poisson sprinkling into a globally hyperbolic spacetime and appropriate scaling limits), maximal antichains can correspond to discrete analogues of spacelike hypersurfaces, and their cardinality can approximate spatial volume/area proxies. This sketch records that conditional correspondence, not a general theorem for arbitrary IG dynamics.
:::

:::{prf:proof}
**Step 1 (Antichains).**
An antichain is a set of elements where no two are causally related (no path exists). This represents "space" at an instant of time.

**Step 2 (Continuum Limit).**
As density $N \to \infty$, the maximum number of points in an antichain scales as the volume of a codimension-1 slice.
$V_{d-1} \sim N^{(d-1)/d}$.

**Step 3 (Foliation).**
A sequence of maximal antichains provides a discrete foliation of the spacetime, recovering the ADM formalism structure.
:::

:::{prf:principle} Holographic Bound (Area-Law Heuristic)
:label: mt:holographic-bound

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $\mathrm{Cap}_H$ (N6), $\mathrm{Rep}_K$ (N11), $C_\mu$ (N3).

**Status:** Heuristic (for a rigorous cut bound, see `thm:causal-horizon-lock`).

**Statement:** In graph-based local dynamics, boundary degrees of freedom control how much information can flow between a region and its complement. This motivates an “area law” principle: information capacity across a cut scales with a discrete boundary measure (cut size / cut capacity), not with the region’s volume.
:::

:::{prf:proof}
This is an interpretation of cut-capacity bounds for local update rules. It becomes rigorous once the update model is specified with a per-edge information bound (as in `thm:causal-horizon-lock`).
:::

:::{prf:metatheorem} Quasi-Stationary Distribution Sampling (Killed Kernels and Fleming–Viot)
:label: mt:quasi-stationary-distribution-sampling

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $D_E$ (N1).

**Status:** Conditional (standard QSD / Fleming–Viot theory).

**Setup (killing):** Let $(X_k)_{k\ge 0}$ be a Markov chain on a state space $E$ with a cemetery state $\partial$ and killing time $\tau_\partial=\inf\{k\ge 0: X_k=\partial\}$. Let $Q$ be the corresponding **sub-Markov kernel** on $E$:
$$
Q(x,A):=\mathbb{P}_x(X_1\in A,\ X_1\neq \partial),\qquad A\subseteq E.
$$

**Definition (QSD):** A probability measure $\nu$ on $E$ is a quasi-stationary distribution if there exists $\alpha\in(0,1)$ such that
$$
\nu Q=\alpha\,\nu.
$$
Equivalently, if $X_0\sim \nu$ then for all $k\ge 0$,
$$
\mathcal{L}(X_k\mid k<\tau_\partial)=\nu,\qquad \mathbb{P}(k<\tau_\partial)=\alpha^k.
$$

**Statement (existence/uniqueness and convergence):** Under standard hypotheses ensuring tightness and mixing—e.g. a Foster–Lyapunov drift condition and a small-set/Doeblin minorization on a compact set (precisely the kind of inputs tracked by $D_E$ and $C_\mu$)—a QSD exists, is unique, and the conditioned law converges to it at an exponential rate (in total variation / Wasserstein, depending on the model).

**Statement (particle approximation):** The constant-$N$ Fleming–Viot particle system (kill-at-$\partial$ + instantaneous resampling from survivors) provides an empirical-measure approximation of the QSD: as $N\to\infty$, the empirical measure converges to the nonlinear normalized semigroup, and its stationary point is the QSD $\nu$.

**Remark (what is and is not “canonical”):** QSD sampling is canonical **for the killed dynamics** $(Q,\partial)$ (up to measurable isomorphism). It does not imply a unique “diffeomorphism-invariant discretization” beyond that standard invariance.
:::

:::{prf:proof}
This is a standard theorem family in QSD theory. One route to the result is spectral: under compactness/regularity assumptions, the sub-Markov operator $Q$ has a principal eigenvalue/eigenmeasure pair $(\alpha,\nu)$, and a spectral gap gives exponential convergence of the conditioned semigroup. Another route uses Harris-type drift/minorization to prove existence/uniqueness and quantitative convergence. Fleming–Viot convergence follows from propagation-of-chaos arguments for interacting particle systems approximating the normalized semigroup.
:::

:::{prf:metatheorem} Modular-Thermal Isomorphism
:label: mt:modular-thermal

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $D_E$ (N1), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic (operator-algebraic QFT correspondence; requires a specified observable algebra and state).

**Statement:** The modular flow $\sigma_t^\phi$ of the local algebra of observables generates the time evolution. The state satisfies the KMS condition with respect to this flow, implying an intrinsic temperature (Unruh effect).
:::

:::{prf:proof}
**Step 1 (Tomita-Takesaki Theory).**
For a von Neumann algebra $\mathcal{A}$ and state $\Omega$, there exists a modular automorphism group $\sigma_t$.
$\Delta^{it} A \Delta^{-it}$.

**Step 2 (KMS Condition).**
The state $\Omega$ behaves like a thermal state $e^{-\beta H}$ with respect to the modular Hamiltonian $H = -\log \Delta$.

**Step 3 (Geometry).**
In curved spacetime (Rindler wedge), the modular flow corresponds to the boost generator. The geometric horizon induces an effective temperature proportional to surface gravity.
:::

:::{prf:metatheorem} Thermodynamic Gravity Principle
:label: mt:thermodynamic-gravity

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$, $\mathfrak{D}^{\text{thin}}$.
**Permits:** $D_E$ (N1), $\mathrm{Cap}_H$ (N6), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (Jacobson-type derivations are conditional in GR/QFT settings; mapping to IG/Fractal-Gas objects is interpretive).

**Statement:** In Jacobson’s framework (under specific semiclassical assumptions: local Rindler horizons, area–entropy relation, and Clausius relation $\delta Q=T\delta S$ for all local observers), one can derive the Einstein field equations as an equation of state. This sketch uses that conditional derivation as an analogy for “entropic/thermodynamic gravity” interpretations of solver dynamics:
$$ R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu} $$
:::

:::{prf:proof}
**Step 1 (Jacobson's Argument).**
Assume entropy is proportional to area $S \propto A$.
Heat flux is energy flux through the horizon $\delta Q = \int T_{\mu\nu} \xi^\mu d\Sigma^\nu$.

**Step 2 (Raychaudhuri Equation).**
The area change is governed by expansion $\theta$.
Geometric focussing relates curvature to matter energy.

**Step 3 (Einstein Equation).**
Matching the thermodynamic relation locally for all observers recovers the full nonlinear field equations of General Relativity. Gravity is the hydrodynamics of the Fractal Gas entanglement.
:::

:::{prf:metatheorem} Inevitability of General Relativity
:label: mt:inevitability-gr

**Thin inputs:** all thin objects.
**Permits:** $D_E$ (N1), $\mathrm{LS}_\sigma$ (N7), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (spin-2 bootstrap arguments are conditional; “emergent graviton of the IG” is interpretive).

**Statement:** In high-energy theory, under specific assumptions (Lorentz invariance, locality, unitarity, and a massless spin-2 field universally coupled to conserved stress-energy), bootstrap arguments constrain the low-energy effective theory toward GR-like dynamics. This block records that conditional line of reasoning; it does not claim that an arbitrary IG construction necessarily produces the required assumptions or a graviton sector.
:::

:::{prf:proof}
**Step 1 (Spin-2 Field).**
The fluctuations of the metric $h_{\mu\nu}$ form a spin-2 representation.

**Step 2 (Coupling).**
The field must couple to the stress-energy tensor $T_{\mu\nu}$.
Conservation $\nabla^\mu T_{\mu\nu} = 0$ requires the coupling to be universal (Equivalence Principle).

**Step 3 (Nonlinearity).**
The graviton carries energy, so it must couple to itself. This bootstrap procedure reconstructs the nonlinear Einstein-Hilbert action as the unique ghost-free extension.
:::

:::{prf:metatheorem} Virial-Cosmological Transition
:label: mt:virial-cosmological

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$, $\mathfrak{D}^{\text{thin}}$.
**Permits:** $D_E$ (N1), $\mathrm{LS}_\sigma$ (N7), $\mathrm{Cap}_H$ (N6).

**Status:** Heuristic (cosmology analogy; no physical phase transition is derived from the solver).

**Statement:** Heuristically, one can compare “bound” regimes (strong confinement/low effective temperature) to virialized equilibria and “unbound” regimes (high diffusion/weak confinement) to expansion. This is an analogy for solver behavior across energy/temperature scales, not a claim of a literal cosmological phase transition.
:::

:::{prf:proof}
**Step 1 (Virial Theorem).**
For bound systems: $2K + U = 0$. $\rho$ is stationary.
This corresponds to $\Lambda_{eff} \approx 0$ or attractive gravity.

**Step 2 (Dark Energy).**
If diffusion dominates (high entropy production), the system expands.
The acceleration $\ddot{a} > 0$ corresponds to a positive cosmological constant $\Lambda > 0$.

**Step 3 (Transition).**
The crossover occurs when the entropy of the horizon equals the entropy of the bulk. The Fractal Gas naturally tunes $\Lambda$ to be small but non-zero (entropic gravity remnant).
:::

:::{prf:metatheorem} Flow with Surgery
:label: mt:flow-with-surgery

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $D_E$ (N1), $\mathrm{Cap}_H$ (N6), $\mathrm{TB}_\pi$ (N8).

**Status:** Heuristic-to-conditional (Ricci flow with surgery is conditional on geometric hypotheses; the resampling correspondence is interpretive).

**Statement:** Conditional on being in a regime where an effective Ricci-flow description is meaningful, Ricci flow can be continued through singularities via surgery (Perelman’s theory). The analogy to Fractal Gas is that killing/resampling can remove “high-curvature/high-energy” regions and allow continued evolution; the correspondence is interpretive rather than a proved equivalence.
:::

:::{prf:proof}
**Step 1 (Singularity Formation).**
Finite time singularities occur where curvature blows up (neck pinching).
In the graph, this is a collapsing cluster or disconnected component.

**Step 2 (Surgery).**
Remove the high-curvature region and cap the holes.
In Fractal Gas: kill unstably high-energy particles and renormalization.

**Step 3 (Continuation).**
The flow is uniquely defined post-surgery (Perelman's construction). The discrete updates of the Fractal Gas naturally implement this surgery without manual intervention (topology change is automatic).
:::

:::{prf:metatheorem} Agency-Geometry Unification
:label: mt:agency-geometry

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$, $G^{\text{thin}}$.
**Permits:** $\mathrm{GC}_T$ (N16), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic-to-conditional (control–geometry dualities exist in specified settings; “dual” is interpretive without a concrete model).

**Statement:** In certain control problems, cost minimization can be reformulated as geodesic motion for an appropriate metric (e.g. Jacobi/Maupertuis metrics in conservative settings or information-geometric metrics in statistical control). This block records the conditional pattern; a specific equivalence requires an explicit dynamics/cost model and the corresponding geometric structure.
$$ \min_{\pi} J(\pi) \iff \delta \int ds = 0 $$
:::

:::{prf:proof}
**Step 1 (Control Hamiltonian).**
Optimal control defines a Hamiltonian $H(x, p)$.
Trajectories satisfy Hamilton's equations.

**Step 2 (Maupertuis Principle).**
For conservative systems (or fixed energy), the trajectories are geodesics of the Jacobi metric $g_{ij} = 2(E-V) M_{ij}$.

**Step 3 (Curvature as Constraint).**
High cost regions act as "hills" (positive curvature) that deflects the agent. The agent "perceives" the problem difficulty as geometric curvature. Constructing a solver is measuring the geometry.
:::

### Thermodynamics and Statistical Mechanics

:::{prf:metatheorem} Fisher-Hessian Isomorphism (Thermodynamics)
:label: mt:fisher-hessian-thermo

**Thin inputs:** $\Phi^{\text{thin}}$.
**Permits:** $D_E$ (N1), $\mathrm{LS}_\sigma$ (N7).

**Status:** Heuristic-to-conditional (Ruppeiner metric is standard in equilibrium thermodynamics; relating it directly to $\nabla^2\Phi$ depends on the ensemble/coordinates and Gaussian approximations).

**Statement:** In equilibrium thermodynamics, the Ruppeiner metric is defined (in entropy representation) by the Hessian
$$
g_{ij}=-\frac{\partial^2 S}{\partial E_i\,\partial E_j}.
$$
In Gaussian/Laplace regimes for Gibbs families, this metric is related to fluctuation covariances and to Hessians of appropriate thermodynamic potentials (free energies) in the chosen coordinates. The schematic identification $g_{ij}\propto \nabla^2\Phi$ should be read as an approximation valid when $\Phi$ plays the role of a quadratic effective potential in the coordinates used.
:::

:::{prf:proof}
**Step 1 (Einstein Fluctuation Formula).**
The probability of a fluctuation from equilibrium is $P(\delta x) \propto \exp(\delta S / k_B) \approx \exp(-\frac{1}{2} g_{ij} \delta x^i \delta x^j)$.

**Step 2 (Hessian).**
Expanding the potential $\Phi$ around the minimum: $\Phi \approx \Phi_0 + \frac{1}{2} \nabla^2 \Phi (\delta x)^2$.
Thus $g_{ij} \propto \nabla^2 \Phi$.

**Step 3 (Stability).**
Positive definiteness of the metric (Stiffness permit) ensures thermodynamic stability (convexity of free energy).
:::

:::{prf:metatheorem} Scalar Curvature Barrier
:label: mt:scalar-curvature-barrier

**Thin inputs:** $\Phi^{\text{thin}}$.
**Permits:** $\mathrm{LS}_\sigma$ (N7), $\mathrm{Cap}_H$ (N6).

**Status:** Heuristic-to-conditional (Ruppeiner-curvature interpretations are conditional; relating curvature bounds to solver stability is interpretive).

**Statement:** In thermodynamic geometry, scalar curvature is often interpreted as a proxy for interaction strength/correlation volume in certain model classes; near critical points it can diverge. If one is in a regime where such interpretations apply and curvature remains bounded, Gaussian/mean-field approximations are more plausible. This is a conditional diagnostic, not a universal barrier theorem for all Fractal Gas instantiations.
:::

:::{prf:proof}
**Step 1 (Interaction Length).**
Ruppeiner curvature $R$ is related to the correlation volume $\xi^d$.
$R \sim \xi^d$.

**Step 2 (Critical Point).**
At a critical point, correlation length diverges $\xi \to \infty$, so $R \to \infty$.
The breakdown of the Law of Large Numbers occurs here.

**Step 3 (Gap Condition).**
The spectral gap condition (Stiffness) bounds the correlation length $\xi < 1/\sqrt{\lambda_1}$. This implies finiteness of $R$, guaranteeing the system remains in the "gas" phase (or stable "solid" phase) away from critical boundaries.
:::

:::{prf:metatheorem} GTD Equivalence Principle
:label: mt:gtd-equivalence

**Thin inputs:** $\Phi^{\text{thin}}$.
**Permits:** $D_E$ (N1), $\mathrm{Rep}_K$ (N11).

**Status:** Heuristic (geometrothermodynamics is a specific formalism; Legendre invariance does not automatically imply representation-independence of a solver).

**Statement:** GTD proposes Legendre-invariant geometric structures on the space of equilibrium states. This block records the analogy that some thermodynamic predictions should be representation/ensemble-invariant; applying that idea to a concrete Fractal Gas requires specifying the ensemble, potential, and observables being compared.
:::

:::{prf:proof}
**Step 1 (Legendre Invariance).**
Construct a contact manifold with coordinates $(\Phi, x, \nabla \Phi)$.
A Legendre transform is a change of coordinates preserving the contact structure.

**Step 2 (Quevedo Metric).**
Define a metric $G$ on the space of equilibrium states that transforms as a tensor under Legendre maps.
$G = (\Phi - x \nabla \Phi) \nabla^2 \Phi$.

**Step 3 (Physical Meaning).**
The curvature of this metric encodes interaction independent of the control parameters used to probe it. This ensures "Observer Universality" extends to thermodynamic observation.
:::

:::{prf:metatheorem} Tikhonov Regularization
:label: mt:tikhonov-regularization

**Thin inputs:** $\Phi^{\text{thin}}$.
**Permits:** $\mathrm{SC}_{\partial c}$ (N5), $\mathrm{Cap}_H$ (N6).

**Status:** Heuristic-to-conditional (regularization improves conditioning; geometric curvature claims are model-dependent).

**Statement:** Adding a Tikhonov regularizer (weight decay) $\Phi_{reg} = \Phi + \lambda \|x\|^2$ smooths the thermodynamic geometry, preventing curvature divergence and ensuring compact level sets (Cap_H).
:::

:::{prf:proof}
**Step 1 (Hessian Modification).**
$\nabla^2 \Phi_{reg} = \nabla^2 \Phi + 2\lambda I$.
This shifts the spectrum $\text{spec}(\nabla^2 \Phi) \to \text{spec}(\nabla^2 \Phi) + 2\lambda$.

**Step 2 (Conditioning).**
Even if $\nabla^2 \Phi$ has zero eigenvalues (flat directions/Goldstone modes), the regularized Hessian is strictly positive definite. Condition number improves.

**Step 3 (Curvature Bound).**
Since the metric determinant is bounded away from zero, the scalar curvature (involving inverse metric) remains bounded. Singularities are resolved.
:::

:::{prf:metatheorem} Convex Hull Resolution
:label: mt:convex-hull-resolution

**Thin inputs:** $\Phi^{\text{thin}}$.
**Permits:** $\mathrm{Cap}_H$ (N6), $\mathrm{TB}_O$ (N9).

**Status:** Conditional (convex-envelope/Maxwell constructions in equilibrium statistical mechanics; requires an equilibrium/large-system regime).

**Statement:** Thermodynamic equilibrium is determined by the convex hull of the potential (Maxwell construction). Non-convexities in $\Phi$ (instabilities) are bridged by phase coexistence, effectively flattening the geometry to its convex envelope $\Phi^{**}$.
:::

:::{prf:proof}
**Step 1 (Legendre-Fenchel).**
The Free Energy $F(\beta)$ is the Legendre transform of $\Phi(E)$. $F$ is always concave.
Transforming back yields $\Phi^{**}$, the convex envelope.

**Step 2 (Phase Separation).**
In regions where $\Phi > \Phi^{**}$, the system separates into a mixture of pure phases (tangent points).
The effective potential sensed by macroscopic observers is $\Phi^{**}$.

**Step 3 (Stability).**
Global stability is restoring; microscopic instabilities correspond to interface formation (domain walls) which cost energy, driving the system to the convex hull solution.
:::

### Additional Physics Bounds

:::{prf:metatheorem} Holographic Power Bound
:label: mt:holographic-power-bound

**Thin inputs:** $\mathcal{X}^{\text{thin}}$.
**Permits:** $\mathrm{Cap}_H$ (N6), $\mathrm{LS}_\sigma$ (N7).

**Status:** Heuristic (physics-bound analogy; not a proved counting theorem for generic kinetic power sets).

**Statement:** The number of physical states in the Kinetic Power Set $\mathcal{P}(X)$ scales as entropy $e^S$, not as the full power set $2^{e^S}$.
Most subsets of states are physically inaccessible (energy forbidden).
:::

:::{prf:proof}
**Step 1 (Typical Set).**
By the Asymptotic Equipartition Property, the typical set $A_\epsilon^{(n)}$ has size $2^{nH}$.
The full configuration space size is $|\mathcal{X}|^n$.
If $H < \log |\mathcal{X}|$, the fraction of occupied states vanishes.

**Step 2 (Energy constraint).**
Constraint $\langle \Phi \rangle < E_{max}$ restricts the system to a thin shell in phase space.
The volume of this shell grows as $E^{dim}$, whereas the volume of the hypercube grows exponentially in dimension.

**Step 3 (Holography).**
Combining with the Holographic bound, $S \propto \text{Area}$. The accessible state space dimension is effectively lower than the bulk dimension. The "Power Set" is an illusion of non-physical kinematics.
:::

:::{prf:theorem} Trotter-Suzuki Product Formula
:label: thm:trotter-suzuki

**Thin inputs:** $\Phi^{\text{thin}}$, $\mathfrak{D}^{\text{thin}}$.
**Permits:** $\mathrm{Rep}_K$ (N11), $\mathrm{SC}_\lambda$ (N4).

**Status:** Conditional (Trotter–Kato product formula under generator/domain conditions).

**Statement:** Under standard semigroup hypotheses for generators $K$ (diffusion) and $V$ (multiplication/killing potential), the propagator for the combined operator $H = K + V$ is the limit of alternating steps:
$$e^{-t(K+V)} = \lim_{n\to\infty} (e^{-\frac{t}{n}K} e^{-\frac{t}{n}V})^n$$
This provides a mathematical justification for split-step schemes at the level of the limiting semigroup; a specific solver’s convergence still depends on how its discrete mutation/selection steps approximate $e^{-tK}$ and $e^{-tV}$.
:::

:::{prf:proof}
**Step 1 (Operator Norm).**
For bounded operators, the error is $O(t^2/n)$.
$\| e^{t(A+B)} - (e^{tA/n} e^{tB/n})^n \| \le \frac{t^2}{2n} \|[A,B]\| e^{t(\|A\|+\|B\|)}$.

**Step 2 (Unbounded Operators).**
For Laplacian $K$ and potentials $V$ bounded below, the formula holds on a core of the domain (Kato-Trotter theorem).

**Step 3 (Simulation).**
This proves that the discrete time-step algorithm converges to the continuous time physics as $\Delta t \to 0$.
:::

:::{prf:theorem} Global Convergence (Darwinian Ratchet)
:label: thm:global-convergence

**Thin inputs:** $\mathcal{X}^{\text{thin}}$, $\Phi^{\text{thin}}$.
**Permits:** $C_\mu$ (N3), $D_E$ (N1).

**Status:** Conditional (Laplace principle / simulated annealing under ergodicity + schedule hypotheses).

**Statement:** If the potential $\Phi$ has a unique global minimum $x^*$ and sublevel sets are compact, the Fractal Gas measure converges weakly to the delta measure $\delta_{x^*}$ as $\beta \to \infty$ (annealing limit).
$$ \lim_{\beta \to \infty} \rho_\beta = \delta_{x^*} $$
:::

:::{prf:proof}
**Step 1 (Large Deviation).**
The equilibrium density is $\rho_\beta(x) \propto e^{-\beta \Phi(x)}$.
The probability of being outside a neighborhood $U$ of $x^*$ is bounded by $e^{-\beta (\min_{X \setminus U} \Phi - \Phi(x^*))}$.

**Step 2 (Gap).**
Since $\Phi(x) > \Phi(x^*)$ for all $x \notin U$, the exponent is negative.
As $\beta \to \infty$, the probability vanishes exponentially.

**Step 3 (Borel-Cantelli).**
For an annealing schedule $\beta_t \sim \log t$, the system eventually settles in the global minimum basin with probability 1 (Geman-Geman).
:::

:::{prf:theorem} Spontaneous Symmetry Breaking
:label: thm:ssb

**Thin inputs:** $\Phi^{\text{thin}}$, $G^{\text{thin}}$.
**Permits:** $\mathrm{LS}_\sigma$ (N7), $\mathrm{SC}_{\partial c}$ (N5).

**Status:** Heuristic-to-conditional (SSB is sharp in thermodynamic/infinite-volume limits; finite-$N$ systems do not literally break symmetry).

**Statement:** In infinite-volume/thermodynamic limits of symmetric systems, one can have multiple extremal equilibrium states selecting a particular “vacuum” and breaking a symmetry of the Hamiltonian/potential; in that setting Goldstone modes correspond to low-cost fluctuations along the orbit of minimizers. For finite-$N$ Fractal Gas instances, symmetry breaking should be interpreted as long-lived metastable localization near one symmetry-related basin rather than literal non-invariant stationary laws.
:::

:::{prf:proof}
**Step 1 (Degeneracy).**
Symmetry implies $\Phi(g x^*) = \Phi(x^*)$. The set of minima is the manifold $M_0 \cong G/H$.

**Step 2 (Instability).**
Any perturbation breaks the symmetry explicitly. The "center of mass" of the swarm cannot remain at the unstable symmetric point (maximum of free energy in the order parameter space).

**Step 3 (Goldstone Bosons).**
The Hessian has null eigenvectors tangent to $M_0$. These are massless modes (diffusive with no drift). Transverse modes are massive (restoring force).
:::
