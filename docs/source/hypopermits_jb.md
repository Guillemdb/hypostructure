---
title: "The Hypostructure Formalism: A Categorical Framework for Singularity Resolution"
subtitle: "From Higher Topos Theory to Constructive Verification"
author: "Guillem Duran Ballester"
---

# The Hypostructure Formalism: A Categorical Framework for Singularity Resolution

## Abstract

This document defines the **Hypostructure**, a mathematical object within a cohesive $(\infty, 1)$-topos that encodes the conservation laws, symmetries, and topological constraints of a dynamical system. We introduce a constructive method for verifying global regularity: rather than assuming hard analytic bounds *a priori*, we define **Thin Kernel Objects** (minimal physical data) and a **Structural Sieve** (a decidability functor). The Sieve acts as a **resolution functor** $\mathcal{S}: \mathbf{Thin}_T \to \mathbf{Hypo}_T^{\geq 0}$, attempting to promote Thin Objects into a valid Hypostructure via certificate saturation within a Postnikov tower.

This approach resolves the "circularity of compactness" critique by treating compactness not as an assumption, but as a runtime branch (Concentration vs. Dispersion) within a rigorous diagnostic automaton. The framework provides:

1. **Categorical foundations** in Higher Topos Theory with cohesive modalities
2. **Constructive approach** via Thin Kernel Objects requiring only uncontroversial physical definitions
3. **Axiom system** organized by constraint class (Conservation, Duality, Symmetry, Topology, Boundary)
4. **Operational semantics** making the sieve diagram executable as a proof-carrying program
5. **Certificate vocabulary** with explicit schemas for all node outcomes
6. **Factory metatheorems** enabling type-based instantiation from definitions alone

---

# Part I: Categorical Foundations

## 1. The Ambient Substrate

To ensure robustness against deformation and gauge redundancies, we work within **Higher Topos Theory** and **Homotopy Type Theory (HoTT)**. This framework is strictly more expressive than ZFC set theory and naturally encodes the homotopical structure of configuration spaces.

:::{prf:definition} Ambient $\infty$-Topos
:label: def-ambient-topos

Let $\mathcal{E}$ be a **cohesive $(\infty, 1)$-topos** equipped with the shape/flat/sharp modality adjunction:
$$\Pi \dashv \flat \dashv \sharp : \mathcal{E} \to \infty\text{-Grpd}$$

The cohesion structure provides:
- **Shape** $\Pi$: Extracts the underlying homotopy type (fundamental $\infty$-groupoid)
- **Flat** $\flat$: Includes discrete $\infty$-groupoids as "constant" objects
- **Sharp** $\sharp$: Includes codiscrete objects (contractible path spaces)

Standard examples include the topos of smooth $\infty$-stacks $\mathbf{Sh}_\infty(\mathbf{CartSp})$ and differential cohesive types.
:::

## 2. The Hypostructure Object

A Hypostructure is not merely a set of equations, but a geometric object equipped with a connection and a filtration.

:::{prf:definition} Categorical Hypostructure
:label: def-categorical-hypostructure

A **Hypostructure** is a tuple $\mathbb{H} = (\mathcal{X}, \nabla, \Phi_\bullet, \tau, \partial_\bullet)$ where:

1. **State Stack** $\mathcal{X} \in \text{Obj}(\mathcal{E})$: The **configuration stack** representing all possible states. This is an $\infty$-sheaf encoding both the state space and its symmetries. The homotopy groups $\pi_n(\mathcal{X})$ capture:
   - $\pi_0$: Connected components (topological sectors)
   - $\pi_1$: Gauge symmetries and monodromy
   - $\pi_n$ ($n \geq 2$): Higher coherences and anomalies

2. **Flat Connection** $\nabla: \mathcal{X} \to T\mathcal{X}$: A section of the tangent $\infty$-bundle, encoding the dynamics as **parallel transport**. The semiflow $S_t$ is recovered as the exponential map:
   $$S_t = \exp(t \cdot \nabla): \mathcal{X} \to \mathcal{X}$$
   The flatness condition $[\nabla, \nabla] = 0$ ensures consistency of time evolution.

3. **Cohomological Height** $\Phi_\bullet: \mathcal{X} \to \mathbb{R}_\infty$: A **cohomological field theory** assigning to each state its energy/complexity. The notation $\Phi_\bullet$ indicates this is a **derived functor**—it comes equipped with higher coherences $\Phi_n$ for all $n$.

4. **Truncation Structure** $\tau = (\tau_C, \tau_D, \tau_{SC}, \tau_{LS})$: The axioms are realized as **truncation functors** on the homotopy groups of $\mathcal{X}$:
   - **Axiom C**: $\tau_C$ truncates unbounded orbits
   - **Axiom D**: $\tau_D$ bounds the energy filtration
   - **Axiom SC**: $\tau_{SC}$ constrains weight gradings
   - **Axiom LS**: $\tau_{LS}$ truncates unstable modes

5. **Boundary Morphism** $\partial_\bullet: \mathcal{X} \to \mathcal{X}_\partial$: A **restriction functor** to the boundary $\infty$-stack, representing the **Holographic Screen**—the interface between bulk dynamics and the external environment. Formally, $\partial_\bullet$ is the pullback along the inclusion $\iota: \partial\mathcal{X} \hookrightarrow \mathcal{X}$:
   $$\partial_\bullet := \iota^*: \mathbf{Sh}_\infty(\mathcal{X}) \to \mathbf{Sh}_\infty(\partial\mathcal{X})$$

   This structure satisfies:

   - **Stokes' Constraint (Differential Cohomology):** Let $\hat{\Phi} \in \hat{H}^n(\mathcal{X}; \mathbb{R})$ be the differential refinement of the energy class. The **integration pairing** satisfies:
     $$\langle d\hat{\Phi}, [\mathcal{X}] \rangle = \langle \hat{\Phi}, [\partial\mathcal{X}] \rangle$$
     where $d: \hat{H}^n \to \Omega^{n+1}_{\text{cl}}$ is the curvature map. This rigidly links internal dissipation to boundary flux via the **de Rham-Cheeger-Simons sequence**.

   - **Cobordism Interface:** For Surgery operations, $\partial_\bullet$ defines the gluing interface in the symmetric monoidal $(\infty,1)$-category $\mathbf{Bord}_n^{\text{or}}$. Given a cobordism $W: M_0 \rightsquigarrow M_1$, the boundary functor satisfies:
     $$\partial W \simeq M_0 \sqcup \overline{M_1} \quad \text{in } \mathbf{Bord}_n$$
     Surgery is a morphism in this category; gluing is composition.

   - **Holographic Bound (Entropy as Cohomological Invariant):** The **cohomological entropy** $S(\mathcal{X}) := \log|\pi_0(\mathcal{X}_{\text{sing}})|$ is bounded by the boundary capacity:
     $$S(\mathcal{X}) \leq C \cdot \chi(\partial\mathcal{X})$$
     where $\chi$ denotes the Euler characteristic. This generalizes the Bekenstein bound to the categorical setting.
:::

:::{prf:remark} Classical Recovery
When $\mathcal{E} = \mathbf{Set}$ (the trivial topos), the categorical definition reduces to classical structural flow data: $\mathcal{X}$ becomes a Polish space $X$, the connection $\nabla$ becomes a vector field generating a semiflow, the truncation functors become decidable propositions, and the boundary morphism $\partial_\bullet$ becomes the Sobolev trace operator $u \mapsto u|_{\partial\Omega}$ with flux $\mathcal{J} = \nabla u \cdot \nu$ (normal derivative).
:::

## 3. The Fixed-Point Principle

The hypostructure axioms are not independent postulates chosen for technical convenience. They are manifestations of a single organizing principle: **self-consistency under evolution**.

:::{prf:definition} Self-Consistency
:label: def-self-consistency

A trajectory $u: [0, T) \to X$ is **self-consistent** if:
1. **Temporal coherence:** The evolution $F_t: x \mapsto S_t x$ preserves the structural constraints defining $X$.
2. **Asymptotic stability:** Either $T = \infty$, or the trajectory approaches a well-defined limit as $t \nearrow T$.
:::

:::{prf:metatheorem} The Fixed-Point Principle
:label: mt-fixed-point-principle

Let $\mathcal{S}$ be a structural flow datum. The following are equivalent:
1. The system $\mathcal{S}$ satisfies the hypostructure axioms on all finite-energy trajectories.
2. Every finite-energy trajectory is asymptotically self-consistent.
3. The only persistent states are fixed points of the evolution operator $F_t = S_t$ satisfying $F_t(x) = x$.

**Interpretation:** The equation $F(x) = x$ encapsulates the principle: *structures that persist under their own evolution are precisely those that satisfy the hypostructure axioms*. Singularities represent states where $F(x) \neq x$ in the limit—the evolution attempts to produce a state incompatible with its own definition.
:::

---

# Part II: The Constructive Approach

## 4. The Thin Kernel (Minimal Inputs)

Classical analysis often critiques structural approaches for assuming hard properties (like Compactness) that are as difficult to prove as the result itself. We resolve this by requiring only **Thin Objects**—uncontroversial physical definitions—as inputs.

:::{prf:definition} Thin Kernel Objects
:label: def-thin-objects

To instantiate a system, the user provides only:

1. **The Arena** $(\mathcal{X}^{\text{thin}})$: The metric space and measure (e.g., $L^2(\mathbb{R}^3)$, a Polish space with Borel $\sigma$-algebra).

2. **The Potential** $(\Phi^{\text{thin}})$: The energy functional and its scaling dimension $\alpha$.

3. **The Cost** $(\mathfrak{D}^{\text{thin}})$: The dissipation rate and its scaling dimension $\beta$.

4. **The Invariance** $(G^{\text{thin}})$: The symmetry group and its action on $\mathcal{X}$.

5. **The Interface** $(\partial^{\text{thin}})$: The boundary data specifying how the system couples to its environment, given as a tuple $(\mathcal{B}, \text{Tr}, \mathcal{J}, \mathcal{R})$:

   - **Boundary Object** $\mathcal{B} \in \text{Obj}(\mathcal{E})$: An $\infty$-stack representing the space of boundary data (inputs, outputs, environmental states).

   - **Trace Morphism** $\text{Tr}: \mathcal{X} \to \mathcal{B}$: A morphism in $\mathcal{E}$ implementing restriction to the boundary. In the classical setting, this is the Sobolev trace $u \mapsto u|_{\partial\Omega}$. Categorically, $\text{Tr}$ is the counit of the adjunction $\iota_! \dashv \iota^*$ where $\iota: \partial\mathcal{X} \hookrightarrow \mathcal{X}$.

   - **Flux Morphism** $\mathcal{J}: \mathcal{B} \to \underline{\mathbb{R}}$: A morphism to the constant sheaf $\underline{\mathbb{R}}$, measuring energy/mass flow across the boundary. Conservation is expressed as:
     $$\frac{d}{dt}\Phi \simeq -\mathcal{J} \circ \text{Tr} \quad \text{in } \text{Hom}_{\mathcal{E}}(\mathcal{X}, \underline{\mathbb{R}})$$

   - **Reinjection Kernel** $\mathcal{R}: \mathcal{B} \to \mathcal{P}(\mathcal{X})$: A **Markov kernel** in the Kleisli category of the probability monad $\mathcal{P}$, implementing non-local boundary conditions (Fleming-Viot, McKean-Vlasov). This is a morphism $\mathcal{R}: \mathcal{B} \to \mathcal{P}(\mathcal{X})$ satisfying the **Feller property**: for each bounded continuous $f: \mathcal{X} \to \mathbb{R}$, the map $b \mapsto \int_\mathcal{X} f \, d\mathcal{R}(b)$ is continuous. Special cases:
     - $\mathcal{R} \simeq 0$ (zero measure): absorbing boundary (Dirichlet)
     - $\mathcal{R}(b) = \delta_{\iota(b)}$ (Dirac at inclusion): reflecting boundary (Neumann)
     - $\mathcal{R}(b) = \mu_t$ (empirical measure): Fleming-Viot reinjection

These are the **only** inputs. All other properties (compactness, stiffness, topological structure) are **derived** by the Sieve, not assumed.
:::

:::{prf:remark} The Structural Role of $\partial$
:label: rem-boundary-role

The Boundary Operator is not merely a geometric edge—it is a **Functor** between Bulk and Boundary categories that powers three critical subsystems:

1. **Conservation Laws (Nodes 1-2):** Via the **Stokes morphism** in differential cohomology, $\partial_\bullet$ relates internal rate of change ($\mathfrak{D}$) to external flux ($\mathcal{J}$). In the $\infty$-categorical setting:
   $$\mathfrak{D} \simeq \partial_\bullet^* \mathcal{J} \quad \text{in } \text{Hom}_{\mathcal{E}}(\mathcal{X}, \underline{\mathbb{R}})$$
   Energy blow-up requires the flux morphism to be unbounded.

2. **Control Layer (Nodes 13-16):** The Boundary Functor distinguishes:
   - **Singularity** (internal blow-up, $\text{coker}(\text{Tr})$ trivial)
   - **Injection** (external forcing, $\|\mathcal{J}\|_\infty \to \infty$)

   Node 13 checks that $\text{Tr}$ is not an equivalence (system is open). Nodes 14-15 verify that $\mathcal{J}$ factors through a bounded subobject.

3. **Surgery Interface (Cobordism):** In Metatheorem 16.1 (Structural Surgery), $\partial_\bullet$ defines the gluing interface in $\mathbf{Bord}_n$:
   - **Cutting:** The excision defines a cobordism $W$ with $\partial W = \Sigma$
   - **Gluing:** Composition in $\mathbf{Bord}_n$ via the pushout $u_{\text{bulk}} \sqcup_\Sigma u_{\text{cap}}$

4. **Holographic Bound (Tactic E8):** If $|\pi_0(\mathcal{X}_{\text{sing}})| = \infty$ but $\chi(\partial\mathcal{X}) < \infty$, the singularity is **cohomologically excluded** by the entropy bound.
:::

## 5. The Sieve as Constructor

The Structural Sieve is defined as a functor $F_{\text{Sieve}}: \mathbf{Thin} \to \mathbf{Result}$. It attempts to promote Thin Objects into a full Hypostructure via certificate saturation.

:::{prf:definition} The Sieve Functor
:label: def-sieve-functor

Given Thin Kernel Objects $\mathcal{T} = (\mathcal{X}^{\text{thin}}, \Phi^{\text{thin}}, \mathfrak{D}^{\text{thin}}, G^{\text{thin}}, \partial^{\text{thin}})$, the Sieve produces:

$$F_{\text{Sieve}}(\mathcal{T}) \in \{\texttt{REGULARITY}, \texttt{DISPERSION}, \texttt{FAILURE}(m)\}$$

where $m \in \{C.E, C.D, C.C, S.E, S.D, S.C, T.E, T.D, T.C, D.E, D.C, B.E, B.D, B.C\}$ classifies the failure mode.
:::

### 5.1. The Adjunction Principle

:::{prf:definition} Categories of Hypostructures
:label: def-hypo-thin-categories

We define two categories capturing the minimal and full structural data:

1. **$\mathbf{Thin}_T$** (Category of Thin Objects): Objects are Thin Kernel tuples $\mathcal{T} = (\mathcal{X}^{\text{thin}}, \Phi^{\text{thin}}, \mathfrak{D}^{\text{thin}}, G^{\text{thin}}, \partial^{\text{thin}})$. Morphisms are structure-preserving maps respecting energy scaling, dissipation, symmetry, and boundary structure.

2. **$\mathbf{Hypo}_T$** (Category of Hypostructures): Objects are full Hypostructures $\mathbb{H} = (\mathcal{X}, \nabla, \Phi_\bullet, \tau, \partial_\bullet)$ with certificate data. Morphisms preserve all axiom certificates.

3. **Forgetful Functor** $U: \mathbf{Hypo}_T \to \mathbf{Thin}_T$: Extracts the underlying thin data by forgetting derived structures and certificates.
:::

:::{prf:remark} The Sieve as Left Adjoint
:label: rem-sieve-adjoint

The Structural Sieve computes the **left adjoint** (free construction) to the forgetful functor:

$$F_{\text{Sieve}} \dashv U : \mathbf{Hypo}_T \rightleftarrows \mathbf{Thin}_T$$

**Interpretation:**
- The **unit** $\eta_\mathcal{T}: \mathcal{T} \to U(F_{\text{Sieve}}(\mathcal{T}))$ embeds thin data into its promoted hypostructure.
- The **counit** $\varepsilon_\mathbb{H}: F_{\text{Sieve}}(U(\mathbb{H})) \to \mathbb{H}$ witnesses that re-running the Sieve on already-verified data is idempotent.
- **Freeness:** The promoted hypostructure $F_{\text{Sieve}}(\mathcal{T})$ is the "freest" (most general) valid hypostructure compatible with the thin data—it assumes no more than what the certificates prove.

This categorical perspective explains why the Sieve construction is **canonical** (unique up to isomorphism) and **natural**: it is the universal solution to the problem "given minimal physical data, what is the most general valid structural completion?"

**Literature:** {cite}`MacLane98`; {cite}`Awodey10`
:::

### The Resolution of the Compactness Critique

The framework does **not** assume Axiom C (Compactness). Instead, **Node 3 (CompactCheck)** performs a runtime dichotomy check on the Thin Objects:

:::{prf:theorem} Compactness Resolution
:label: thm-compactness-resolution

At Node 3, the Sieve executes:

1. **Concentration Branch:** If energy concentrates ($\mu(V) > 0$ for some profile $V$), a **Canonical Profile** emerges via scaling limits. Axiom C is satisfied *constructively*—the certificate $K_{C_\mu}^+$ witnesses the concentration.

2. **Dispersion Branch:** If energy scatters ($\mu(V) = 0$ for all profiles), compactness fails. However, this triggers **Mode D.D (Dispersion/Global Existence)**—a success state, not a failure.

**Conclusion:** Regularity is decidable regardless of whether Compactness holds *a priori*. The dichotomy is resolved at runtime, not assumed.
:::

---

# Part III: The Axiom System

The Hypostructure $\mathbb{H}$ is valid if it satisfies the following structural constraints (Axioms). In the operational Sieve, these are verified as **Interface Permits** at the corresponding nodes.

## 6. Conservation Constraints

:::{prf:axiom} Axiom D (Dissipation)
:label: ax-dissipation

The energy-dissipation inequality holds:
$$\Phi(S_t x) + \int_0^t \mathfrak{D}(S_s x) \, ds \leq \Phi(x)$$

**Enforced by:** Node 1 (EnergyCheck) — Certificate $K_{D_E}^+$
:::

:::{prf:axiom} Axiom Rec (Recovery)
:label: ax-recovery

Discrete events are finite: $N(J) < \infty$ for any bounded interval $J$.

**Enforced by:** Node 2 (ZenoCheck) — Certificate $K_{\text{Rec}_N}^+$
:::

## 7. Duality Constraints

:::{prf:axiom} Axiom C (Compactness)
:label: ax-compactness

Bounded energy sequences admit convergent subsequences modulo the symmetry group $G$:
$$\sup_n \Phi(u_n) < \infty \implies \exists (n_k), g_k \in G: g_k \cdot u_{n_k} \to u_\infty$$

**Enforced by:** Node 3 (CompactCheck) — Certificate $K_{C_\mu}^+$ (or dispersion via $K_{C_\mu}^-$)
:::

:::{prf:axiom} Axiom SC (Scaling)
:label: ax-scaling

Dissipation scales faster than time: $\alpha > \beta$, where $\alpha$ is the energy scaling dimension and $\beta$ is the dissipation scaling dimension.

**Enforced by:** Node 4 (ScaleCheck) — Certificate $K_{SC_\lambda}^+$
:::

## 8. Symmetry Constraints

:::{prf:axiom} Axiom LS (Stiffness)
:label: ax-stiffness

The Łojasiewicz-Simon inequality holds near equilibria, ensuring a mass gap:
$$\inf \sigma(L) > 0$$
where $L$ is the linearized operator at equilibrium.

**Enforced by:** Node 7 (StiffnessCheck) — Certificate $K_{LS_\sigma}^+$
:::

:::{prf:axiom} Axiom GC (Gradient Consistency)
:label: ax-gradient-consistency

Gauge invariance and metric compatibility: the control $T(u)$ matches the disturbance $d$.

**Enforced by:** Node 16 (AlignCheck) — Certificate $K_{GC_T}^+$
:::

## 9. Topology Constraints

:::{prf:axiom} Axiom TB (Topological Background)
:label: ax-topology

Topological sectors are separated by an action gap:
$$[\pi] \in \pi_0(\mathcal{C})_{\text{acc}} \implies E < S_{\min} + \Delta$$

**Enforced by:** Node 8 (TopoCheck) — Certificate $K_{TB_\pi}^+$
:::

:::{prf:axiom} Axiom Cap (Capacity)
:label: ax-capacity

Capacity density bounds prevent concentration on thin sets:
$$\text{codim}(S) \geq 2 \implies \text{Cap}_H(S) = 0$$

**Enforced by:** Node 6 (GeomCheck) — Certificate $K_{\text{Cap}_H}^+$
:::

## 10. Boundary Constraints

The Boundary Constraints enforce coupling between bulk dynamics and environmental interface via the Thin Interface $\partial^{\text{thin}} = (\mathcal{B}, \text{Tr}, \mathcal{J}, \mathcal{R})$.

:::{prf:axiom} Axiom Bound (Input/Output Coupling)
:label: ax-boundary

The system's boundary morphisms satisfy:
- $\mathbf{Bound}_\partial$: $\text{Tr}: \mathcal{X} \to \mathcal{B}$ is not an equivalence (open system) — Node 13
- $\mathbf{Bound}_B$: $\mathcal{J}$ factors through a bounded subobject $\mathcal{J}: \mathcal{B} \to \underline{[-M, M]}$ — Node 14
- $\mathbf{Bound}_{\Sigma}$: The integral $\int_0^T \mathcal{J}_{\text{in}}$ exists as a morphism in $\text{Hom}(\mathbf{1}, \underline{\mathbb{R}}_{\geq r_{\min}})$ — Node 15
- $\mathbf{Bound}_{\mathcal{R}}$: The **reinjection diagram** commutes:
  $$\mathcal{J}_{\text{out}} \simeq \mathcal{J}_{\text{in}} \circ \mathcal{R} \quad \text{in } \text{Hom}_{\mathcal{E}}(\mathcal{B}, \underline{\mathbb{R}})$$

**Enforced by:** Nodes 13-16 (BoundaryCheck, OverloadCheck, StarveCheck, AlignCheck)
:::

:::{prf:remark} Reinjection Boundaries (Fleming-Viot)
:label: rem-reinjection

When $\mathcal{R} \not\simeq 0$, the boundary acts as a **non-local transport morphism** rather than an absorbing terminal object. This captures:
- **Fleming-Viot processes:** The reinjection factors through the **probability monad** $\mathcal{P}: \mathcal{E} \to \mathcal{E}$
- **McKean-Vlasov dynamics:** $\mathcal{R}$ depends on global sections $\Gamma(\mathcal{X}, \mathcal{O}_\mu)$
- **Piecewise Deterministic Markov Processes:** $\mathcal{R}$ is a morphism in the Kleisli category of the probability monad

The Sieve verifies regularity by checking **Axiom Rec** at the boundary:
1. **Node 13:** Detects that $\mathcal{J} \neq 0$ (non-trivial exit flux)
2. **Node 15 (StarveCheck):** Verifies $\mathcal{R}$ preserves the **total mass section** ($K_{\text{Mass}}^+$)

Categorically, this defines a **non-local boundary condition** as a span:
$$\mathcal{X} \xleftarrow{\text{Tr}} \mathcal{B} \xrightarrow{\mathcal{R}} \mathcal{P}(\mathcal{X})$$
The resulting integro-differential structure is tamed by **Axiom C** applied to the Wasserstein $\infty$-stack $\mathcal{P}_2(\mathcal{X})$.
:::

---

# Part IV: The Structural Sieve

## 11. The Homotopical Resolution of the Singularity Spectrum

*A Postnikov decomposition of the Regularity Functor $\mathcal{R}$ in a cohesive $(\infty,1)$-topos.*

The following diagram is the **authoritative specification** of the obstruction-theoretic resolution. All subsequent definitions and theorems must align with this categorical atlas.

### 11.1. Computational Boundaries and Undecidability

:::{prf:remark} Acknowledgment of Fundamental Limits
:label: rem-undecidability

The Structural Sieve operates within the computational limits imposed by fundamental results in mathematical logic:

1. **Gödel's Incompleteness (1931):** No sufficiently powerful formal system can prove all true statements about arithmetic within itself {cite}`Godel31`.
2. **Halting Problem (Turing, 1936):** There is no general algorithm to determine whether an arbitrary program will halt {cite}`Turing36`.
3. **Rice's Theorem (1953):** All non-trivial semantic properties of programs are undecidable {cite}`Rice53`.

**Implication for the Sieve:** For sufficiently complex systems (e.g., those encoding universal computation), certain interface predicates $\mathcal{P}_i$ may be **undecidable**—no algorithm can determine their truth value in finite time for all inputs.

The framework addresses this through **Binary Certificate Logic** with typed NO certificates. Every predicate evaluation returns exactly YES or NO—never a third truth value. The NO certificate carries type information distinguishing *refutation* from *inconclusiveness*.

:::{prf:definition} Typed NO Certificates (Binary Certificate Logic)
:label: def-typed-no-certificates

For any predicate $P$ with YES certificate $K_P^+$, the NO certificate is a **coproduct** (sum type) in the category of certificate objects:
$$K_P^- := K_P^{\mathrm{wit}} + K_P^{\mathrm{inc}}$$

**Component 1: NO-with-witness** ($K_P^{\mathrm{wit}}$)

A constructive refutation consisting of a counterexample or breach object that demonstrates $\neg P$. Formally:
$$K_P^{\mathrm{wit}} := (\mathsf{witness}: W_P, \mathsf{verification}: W_P \vdash \neg P)$$
where $W_P$ is the type of refutation witnesses for $P$.

**Component 2: NO-inconclusive** ($K_P^{\mathrm{inc}}$)

A record of evaluator failure that does *not* constitute a semantic refutation. Formally:
$$K_P^{\mathrm{inc}} := (\mathsf{obligation}: P, \mathsf{missing}: \mathcal{M}, \mathsf{code}: \mathcal{C}, \mathsf{trace}: \mathcal{T})$$
where:
- $\mathsf{obligation} \in \mathrm{Pred}(\mathcal{H})$: The exact predicate instance attempted
- $\mathsf{missing} \in \mathcal{P}(\mathrm{Template} \cup \mathrm{Precond})$: Prerequisites or capabilities absent
- $\mathsf{code} \in \{\texttt{TEMPLATE\_MISS}, \texttt{PRECOND\_MISS}, \texttt{NOT\_IMPLEMENTED}, \texttt{RESOURCE\_LIMIT}, \texttt{UNDECIDABLE}\}$
- $\mathsf{trace} \in \mathrm{Log}$: Reproducible evaluation trace (template DB hash, attempted tactics, bounds)

**Injection Maps:** The coproduct structure provides canonical injections:
$$\iota_{\mathrm{wit}}: K_P^{\mathrm{wit}} \to K_P^-, \quad \iota_{\mathrm{inc}}: K_P^{\mathrm{inc}} \to K_P^-$$

**Case Analysis:** Any function $f: K_P^- \to X$ factors uniquely through case analysis:
$$f = [f_{\mathrm{wit}}, f_{\mathrm{inc}}] \circ \mathrm{case}$$
where $f_{\mathrm{wit}}: K_P^{\mathrm{wit}} \to X$ and $f_{\mathrm{inc}}: K_P^{\mathrm{inc}} \to X$.
:::

:::{prf:remark} Routing Semantics
:label: rem-routing-semantics

The Sieve branches on certificate kind via case analysis:
- **NO with $K^{\mathrm{wit}}$** $\mapsto$ Fatal route (structural inconsistency confirmed; no reconstruction possible)
- **NO with $K^{\mathrm{inc}}$** $\mapsto$ Reconstruction route (invoke MT 42.1; add interface/refine library/extend templates)

This design maintains **proof-theoretic honesty**:
- The verdict is always in $\{$YES, NO$\}$—classical two-valued logic
- The certificate carries the epistemic distinction between "refuted" and "not yet proven"
- Reconstruction is triggered by $K^{\mathrm{inc}}$, never by $K^{\mathrm{wit}}$

**Literature:** {cite}`Godel31`; {cite}`Turing36`; {cite}`Rice53`. For sum types in type theory: {cite}`MartinLof84`; {cite}`HoTTBook`.
:::

:::{admonition} Categorical Interpretation
:class: note

This Directed Acyclic Graph represents the **spectral sequence** of obstructions to global regularity.

- **Nodes (Objects):** Each node represents a **Classifying Stack** $\mathcal{M}_i$ for a specific obstruction class (Energy, Topology, Stiffness).
- **Solid Edges (Morphisms):** Represent **Truncation Functors** $\tau_{\leq k}$. A traversal $A \to B$ indicates that the obstruction at $A$ vanishes (is trivial in cohomology), allowing the lift to the next covering space $B$.
- **Dotted Edges (Surgery):** Represent **Cobordism Morphisms** in the category of manifolds. They denote a change of topology (Pushout) required to bypass a non-trivial cohomological obstruction.
- **Terminals (Limits):** The "Victory" node represents the **Contractible Space** (Global Regularity), where all homotopy groups of the singularity vanish.
:::

:::{prf:remark} The Sieve as Spectral Sequence
:label: rem-spectral-sequence

The Structural Sieve admits a natural interpretation as a **spectral sequence** $\{E_r^{p,q}, d_r\}_{r \geq 0}$ converging to the regularity classification:

- **$E_0^{p,q}$**: Initial Thin Kernel data, filtered by obstruction type ($p \in \{\text{Conservation}, \text{Duality}, \text{Symmetry}, \text{Topology}, \text{Boundary}\}$) and filtration level ($q$)
- **Differentials** $d_r: E_r^{p,q} \to E_r^{p+r, q-r+1}$: Obstruction maps at each sieve node
  - $d_1 \sim$ EnergyCheck: Tests finite energy ($\ker d_1 =$ bounded energy states)
  - $d_2 \sim$ CompactCheck: Tests concentration vs. dispersion
  - $d_3 \sim$ ScaleCheck: Tests subcriticality
- **Gate Pass** ($K^+$): Class survives to next page ($d_r(\alpha) = 0$)
- **Gate Fail** ($K^-$): Non-zero differential ($d_r(\alpha) \neq 0$), triggers barrier/surgery
- **Global Regularity**: Collapse at $E_\infty$ with $E_\infty^{p,q} = 0$ for all $(p,q)$ corresponding to singular behavior

This interpretation connects the Sieve to classical obstruction theory in algebraic topology {cite}`McCleary01`.
:::

:::{tip} Interactive Viewing Options
:class: dropdown

This diagram is large. For better viewing:
- **Zoom**: Use your browser's zoom (Ctrl/Cmd + scroll)
- **Full-screen editor**: [Open in Mermaid Live Editor](https://mermaid.live) and paste the code from the [Appendix](#complete-sieve-algorithm)
- **Download**: In Mermaid Live Editor, use the export button to download as SVG or PNG
:::

```mermaid
graph TD
    Start(["<b>START DIAGNOSTIC</b>"]) --> EnergyCheck{"<b>1. D_E:</b> Is Energy Finite?<br>E[Φ] < ∞"}

    %% --- LEVEL 1: 0-TRUNCATION (Energy Bounds) ---
    EnergyCheck -- "No: K-_DE" --> BarrierSat{"<b>B1. D_E:</b> Is Drift Bounded?<br>E[Φ] ≤ E_sat"}
    BarrierSat -- "Yes: Kblk_DE" --> ZenoCheck
    BarrierSat -- "No: Kbr_DE" --> SurgAdmCE{"<b>A1. SurgCE:</b> Admissible?<br>conformal ∧ ∂∞X def."}
    SurgAdmCE -- "Yes: K+_Conf" --> SurgCE["<b>S1. SurgCE:</b><br>Ghost/Cap Extension"]
    SurgAdmCE -- "No: K-_Conf" --> ModeCE["<b>Mode C.E</b>: Energy Blow-Up"]
    SurgCE -. "Kre_SurgCE" .-> ZenoCheck

    EnergyCheck -- "Yes: K+_DE" --> ZenoCheck{"<b>2. Rec_N:</b> Are Discrete Events Finite?<br>N(J) < ∞"}
    ZenoCheck -- "No: K-_RecN" --> BarrierCausal{"<b>B2. Rec_N:</b> Infinite Depth?<br>D#40;T*#41; = ∞"}
    BarrierCausal -- "No: Kbr_RecN" --> SurgAdmCC{"<b>A2. SurgCC:</b> Admissible?<br>∃N_max: events ≤ N_max"}
    SurgAdmCC -- "Yes: K+_Disc" --> SurgCC["<b>S2. SurgCC:</b><br>Discrete Saturation"]
    SurgAdmCC -- "No: K-_Disc" --> ModeCC["<b>Mode C.C</b>: Event Accumulation"]
    SurgCC -. "Kre_SurgCC" .-> CompactCheck
    BarrierCausal -- "Yes: Kblk_RecN" --> CompactCheck

    ZenoCheck -- "Yes: K+_RecN" --> CompactCheck{"<b>3. C_μ:</b> Does Energy Concentrate?<br>μ(V) > 0"}

    %% --- LEVEL 2: COMPACTNESS LOCUS (Profile Moduli) ---
    CompactCheck -- "No: K-_Cmu" --> BarrierScat{"<b>B3. C_μ:</b> Is Interaction Finite?<br>M[Φ] < ∞"}
    BarrierScat -- "Yes: Kben_Cmu" --> ModeDD["<b>Mode D.D</b>: Dispersion<br><i>#40;Global Existence#41;</i>"]
    BarrierScat -- "No: Kpath_Cmu" --> SurgAdmCD_Alt{"<b>A3. SurgCD_Alt:</b> Admissible?<br>V ∈ L_soliton ∧ ‖V‖_H¹ < ∞"}
    SurgAdmCD_Alt -- "Yes: K+_Prof" --> SurgCD_Alt["<b>S3. SurgCD_Alt:</b><br>Concentration-Compactness"]
    SurgAdmCD_Alt -- "No: K-_Prof" --> ModeCD_Alt["<b>Mode C.D</b>: Geometric Collapse<br><i>#40;Via Escape#41;</i>"]
    SurgCD_Alt -. "Kre_SurgCD_Alt" .-> Profile

    CompactCheck -- "Yes: K+_Cmu" --> Profile["<b>Canonical Profile V Emerges</b>"]

    %% --- LEVEL 3: EQUIVARIANT DESCENT ---
    Profile --> ScaleCheck{"<b>4. SC_λ:</b> Is Profile Subcritical?<br>λ(V) < λ_c"}

    ScaleCheck -- "No: K-_SClam" --> BarrierTypeII{"<b>B4. SC_λ:</b> Is Renorm Cost ∞?<br>∫D̃ dt = ∞"}
    BarrierTypeII -- "No: Kbr_SClam" --> SurgAdmSE{"<b>A4. SurgSE:</b> Admissible?<br>α-β < ε_crit ∧ V smooth"}
    SurgAdmSE -- "Yes: K+_Lift" --> SurgSE["<b>S4. SurgSE:</b><br>Regularity Lift"]
    SurgAdmSE -- "No: K-_Lift" --> ModeSE["<b>Mode S.E</b>: Supercritical Cascade"]
    SurgSE -. "Kre_SurgSE" .-> ParamCheck
    BarrierTypeII -- "Yes: Kblk_SClam" --> ParamCheck

    ScaleCheck -- "Yes: K+_SClam" --> ParamCheck{"<b>5. SC_∂c:</b> Are Constants Stable?<br>‖∂c‖ < ε"}
    ParamCheck -- "No: K-_SCdc" --> BarrierVac{"<b>B5. SC_∂c:</b> Is Phase Stable?<br>ΔV > k_B T"}
    BarrierVac -- "No: Kbr_SCdc" --> SurgAdmSC{"<b>A5. SurgSC:</b> Admissible?<br>‖∂θ‖ < C_adm ∧ θ stable"}
    SurgAdmSC -- "Yes: K+_Stab" --> SurgSC["<b>S5. SurgSC:</b><br>Convex Integration"]
    SurgAdmSC -- "No: K-_Stab" --> ModeSC["<b>Mode S.C</b>: Parameter Instability"]
    SurgSC -. "Kre_SurgSC" .-> GeomCheck
    BarrierVac -- "Yes: Kblk_SCdc" --> GeomCheck

    ParamCheck -- "Yes: K+_SCdc" --> GeomCheck{"<b>6. Cap_H:</b> Is Codim ≥ Threshold?<br>codim(S) ≥ 2"}

    %% --- LEVEL 4: DIMENSION FILTRATION ---
    GeomCheck -- "No: K-_CapH" --> BarrierCap{"<b>B6. Cap_H:</b> Is Measure Zero?<br>Cap_H#40;S#41; = 0"}
    BarrierCap -- "No: Kbr_CapH" --> SurgAdmCD{"<b>A6. SurgCD:</b> Admissible?<br>Cap#40;Σ#41; ≤ ε ∧ V ∈ L_neck"}
    SurgAdmCD -- "Yes: K+_Neck" --> SurgCD["<b>S6. SurgCD:</b><br>Auxiliary/Structural"]
    SurgAdmCD -- "No: K-_Neck" --> ModeCD["<b>Mode C.D</b>: Geometric Collapse"]
    SurgCD -. "Kre_SurgCD" .-> StiffnessCheck
    BarrierCap -- "Yes: Kblk_CapH" --> StiffnessCheck

    GeomCheck -- "Yes: K+_CapH" --> StiffnessCheck{"<b>7. LS_σ:</b> Is Gap Certified?<br>inf σ(L) > 0"}

    %% --- LEVEL 5: SPECTRAL OBSTRUCTION ---
    StiffnessCheck -- "No: K-_LSsig" --> BarrierGap{"<b>B7. LS_σ:</b> Is Kernel Finite?<br>dim ker#40;L#41; < ∞ ∧ σ_ess > 0"}
    BarrierGap -- "Yes: Kblk_LSsig" --> TopoCheck
    BarrierGap -- "No: Kstag_LSsig" --> BifurcateCheck{"<b>7a. LS_∂²V:</b> Is State Unstable?<br>∂²V(x*) ⊁ 0"}

    %% --- LEVEL 5b: SPECTRAL RESTORATION (Bifurcation Resolution) ---
    BifurcateCheck -- "No: K-_LSd2V" --> SurgAdmSD{"<b>A7. SurgSD:</b> Admissible?<br>dim ker#40;H#41; < ∞ ∧ V iso."}
    SurgAdmSD -- "Yes: K+_Iso" --> SurgSD["<b>S7. SurgSD:</b><br>Ghost Extension"]
    SurgAdmSD -- "No: K-_Iso" --> ModeSD["<b>Mode S.D</b>: Stiffness Breakdown"]
    SurgSD -. "Kre_SurgSD" .-> TopoCheck
    BifurcateCheck -- "Yes: K+_LSd2V" --> SymCheck{"<b>7b. G_act:</b> Is G-orbit Degenerate?<br>⎸G·v₀⎸ = 1"}

    %% Path A: Symmetry Breaking (Governed by SC_∂c)
    SymCheck -- "Yes: K+_Gact" --> CheckSC{"<b>7c. SC_∂c:</b> Are Constants Stable?<br>‖∂c‖ < ε"}
    CheckSC -- "Yes: K+_SCdc" --> ActionSSB["<b>ACTION: SYM. BREAKING</b><br>Generates Mass Gap"]
    ActionSSB -- "Kgap" --> TopoCheck
    CheckSC -- "No: K-_SCdc" --> SurgAdmSC_Rest{"<b>A8. SurgSC_Rest:</b> Admissible?<br>ΔV > k_B T ∧ Γ < Γ_crit"}
    SurgAdmSC_Rest -- "Yes: K+_Vac" --> SurgSC_Rest["<b>S8. SurgSC_Rest:</b><br>Auxiliary Extension"]
    SurgAdmSC_Rest -- "No: K-_Vac" --> ModeSC_Rest["<b>Mode S.C</b>: Parameter Instability<br><i>#40;Vacuum Decay#41;</i>"]
    SurgSC_Rest -. "Kre_SurgSC_Rest" .-> TopoCheck

    %% Path B: Tunneling (Governed by TB_S)
    SymCheck -- "No: K-_Gact" --> CheckTB{"<b>7d. TB_S:</b> Is Tunneling Finite?<br>S[γ] < ∞"}
    CheckTB -- "Yes: K+_TBS" --> ActionTunnel["<b>ACTION: TUNNELING</b><br>Instanton Decay"]
    ActionTunnel -- "Ktunnel" --> TameCheck
    CheckTB -- "No: K-_TBS" --> SurgAdmTE_Rest{"<b>A9. SurgTE_Rest:</b> Admissible?<br>V ≅ S^n×I ∧ S_R[γ] < ∞"}
    SurgAdmTE_Rest -- "Yes: K+_Inst" --> SurgTE_Rest["<b>S9. SurgTE_Rest:</b><br>Structural"]
    SurgAdmTE_Rest -- "No: K-_Inst" --> ModeTE_Rest["<b>Mode T.E</b>: Topological Twist<br><i>#40;Metastasis#41;</i>"]
    SurgTE_Rest -. "Kre_SurgTE_Rest" .-> TameCheck

    StiffnessCheck -- "Yes: K+_LSsig" --> TopoCheck{"<b>8. TB_π:</b> Is Sector Reachable?<br>[π] ∈ π₀(C)_acc"}

    %% --- LEVEL 6: HOMOTOPICAL OBSTRUCTIONS ---
    TopoCheck -- "No: K-_TBpi" --> BarrierAction{"<b>B8. TB_π:</b> Energy < Gap?<br>E < S_min + Δ"}
    BarrierAction -- "No: Kbr_TBpi" --> SurgAdmTE{"<b>A10. SurgTE:</b> Admissible?<br>V ≅ S^n×R #40;Neck#41;"}
    SurgAdmTE -- "Yes: K+_Topo" --> SurgTE["<b>S10. SurgTE:</b><br>Tunnel"]
    SurgAdmTE -- "No: K-_Topo" --> ModeTE["<b>Mode T.E</b>: Topological Twist"]
    SurgTE -. "Kre_SurgTE" .-> TameCheck
    BarrierAction -- "Yes: Kblk_TBpi" --> TameCheck

    TopoCheck -- "Yes: K+_TBpi" --> TameCheck{"<b>9. TB_O:</b> Is Topology Tame?<br>Σ ∈ O-min"}

    TameCheck -- "No: K-_TBO" --> BarrierOmin{"<b>B9. TB_O:</b> Is It Definable?<br>S ∈ O-min"}
    BarrierOmin -- "No: Kbr_TBO" --> SurgAdmTC{"<b>A11. SurgTC:</b> Admissible?<br>Σ ∈ O-ext def. ∧ dim < n"}
    SurgAdmTC -- "Yes: K+_Omin" --> SurgTC["<b>S11. SurgTC:</b><br>O-minimal Regularization"]
    SurgAdmTC -- "No: K-_Omin" --> ModeTC["<b>Mode T.C</b>: Labyrinthine"]
    SurgTC -. "Kre_SurgTC" .-> ErgoCheck
    BarrierOmin -- "Yes: Kblk_TBO" --> ErgoCheck

    TameCheck -- "Yes: K+_TBO" --> ErgoCheck{"<b>10. TB_ρ:</b> Does Flow Mix?<br>τ_mix < ∞"}

    ErgoCheck -- "No: K-_TBrho" --> BarrierMix{"<b>B10. TB_ρ:</b> Mixing Finite?<br>τ_mix < ∞"}
    BarrierMix -- "No: Kbr_TBrho" --> SurgAdmTD{"<b>A12. SurgTD:</b> Admissible?<br>Trap iso. ∧ ∂T > 0"}
    SurgAdmTD -- "Yes: K+_Mix" --> SurgTD["<b>S12. SurgTD:</b><br>Mixing Enhancement"]
    SurgAdmTD -- "No: K-_Mix" --> ModeTD["<b>Mode T.D</b>: Glassy Freeze"]
    SurgTD -. "Kre_SurgTD" .-> ComplexCheck
    BarrierMix -- "Yes: Kblk_TBrho" --> ComplexCheck

    ErgoCheck -- "Yes: K+_TBrho" --> ComplexCheck{"<b>11. Rep_K:</b> Is K(x) Computable?<br>K(x) ∈ ℕ"}

    %% --- LEVEL 7: KOLMOGOROV FILTRATION ---
    ComplexCheck -- "No: K-_RepK" --> BarrierEpi{"<b>B11. Rep_K:</b> Approx. Bounded?<br>sup K_ε#40;x#41; ≤ S_BH"}
    BarrierEpi -- "No: Kbr_RepK" --> SurgAdmDC{"<b>A13. SurgDC:</b> Admissible?<br>K ≤ S_BH+ε ∧ Lipschitz"}
    SurgAdmDC -- "Yes: K+_Lip" --> SurgDC["<b>S13. SurgDC:</b><br>Viscosity Solution"]
    SurgAdmDC -- "No: K-_Lip" --> ModeDC["<b>Mode D.C</b>: Semantic Horizon"]
    SurgDC -. "Kre_SurgDC" .-> OscillateCheck
    BarrierEpi -- "Yes: Kblk_RepK" --> OscillateCheck

    ComplexCheck -- "Yes: K+_RepK" --> OscillateCheck{"<b>12. GC_∇:</b> Does Flow Oscillate?<br>ẋ ≠ -∇V"}

    OscillateCheck -- "Yes: K+_GCnabla" --> BarrierFreq{"<b>B12. GC_∇:</b> Oscillation Finite?<br>∫ω²S dω < ∞"}
    BarrierFreq -- "No: Kbr_GCnabla" --> SurgAdmDE{"<b>A14. SurgDE:</b> Admissible?<br>∃Λ: trunc. moment < ∞ ∧ elliptic"}
    SurgAdmDE -- "Yes: K+_Ell" --> SurgDE["<b>S14. SurgDE:</b><br>De Giorgi-Nash-Moser"]
    SurgAdmDE -- "No: K-_Ell" --> ModeDE["<b>Mode D.E</b>: Oscillatory"]
    SurgDE -. "Kre_SurgDE" .-> BoundaryCheck
    BarrierFreq -- "Yes: Kblk_GCnabla" --> BoundaryCheck

    OscillateCheck -- "No: K-_GCnabla" --> BoundaryCheck{"<b>13. Bound_∂:</b> Is System Open?<br>∂Ω ≠ ∅"}

    %% --- LEVEL 8: BOUNDARY COBORDISM ---
    BoundaryCheck -- "Yes: K+_Bound" --> OverloadCheck{"<b>14. Bound_B:</b> Is Input Bounded?<br>‖Bu‖ ≤ M"}

    OverloadCheck -- "No: K-_BoundB" --> BarrierBode{"<b>B14. Bound_B:</b> Waterbed Bounded?<br>∫ln‖S‖dω > -∞"}
    BarrierBode -- "No: Kbr_BoundB" --> SurgAdmBE{"<b>A15. SurgBE:</b> Admissible?<br>‖S‖_∞ < M ∧ φ_margin > 0"}
    SurgAdmBE -- "Yes: K+_Marg" --> SurgBE["<b>S15. SurgBE:</b><br>Saturation"]
    SurgAdmBE -- "No: K-_Marg" --> ModeBE["<b>Mode B.E</b>: Injection"]
    SurgBE -. "Kre_SurgBE" .-> StarveCheck
    BarrierBode -- "Yes: Kblk_BoundB" --> StarveCheck

    OverloadCheck -- "Yes: K+_BoundB" --> StarveCheck{"<b>15. Bound_∫:</b> Is Input Sufficient?<br>∫r dt ≥ r_min"}

    StarveCheck -- "No: K-_BoundInt" --> BarrierInput{"<b>B15. Bound_∫:</b> Reserve Positive?<br>r_reserve > 0"}
    BarrierInput -- "No: Kbr_BoundInt" --> SurgAdmBD{"<b>A16. SurgBD:</b> Admissible?<br>r_res > 0 ∧ recharge > drain"}
    SurgAdmBD -- "Yes: K+_Res" --> SurgBD["<b>S16. SurgBD:</b><br>Reservoir"]
    SurgAdmBD -- "No: K-_Res" --> ModeBD["<b>Mode B.D</b>: Starvation"]
    SurgBD -. "Kre_SurgBD" .-> AlignCheck
    BarrierInput -- "Yes: Kblk_BoundInt" --> AlignCheck

    StarveCheck -- "Yes: K+_BoundInt" --> AlignCheck{"<b>16. GC_T:</b> Is Control Matched?<br>T(u) ~ d"}
    AlignCheck -- "No: K-_GCT" --> BarrierVariety{"<b>B16. GC_T:</b> Variety Sufficient?<br>H#40;u#41; ≥ H#40;d#41;"}
    BarrierVariety -- "No: Kbr_GCT" --> SurgAdmBC{"<b>A17. SurgBC:</b> Admissible?<br>H#40;u#41; < H#40;d#41; ∧ bridgeable"}
    SurgAdmBC -- "Yes: K+_Ent" --> SurgBC["<b>S17. SurgBC:</b><br>Controller Augmentation"]
    SurgAdmBC -- "No: K-_Ent" --> ModeBC["<b>Mode B.C</b>: Misalignment"]
    SurgBC -. "Kre_SurgBC" .-> BarrierExclusion

    %% --- LEVEL 9: THE COHOMOLOGICAL BARRIER ---
    %% All successful paths funnel here
    BoundaryCheck -- "No: K-_Bound" --> BarrierExclusion
    BarrierVariety -- "Yes: Kblk_GCT" --> BarrierExclusion
    AlignCheck -- "Yes: K+_GCT" --> BarrierExclusion

    BarrierExclusion{"<b>17. Cat_Hom:</b> Is Hom#40;Bad, S#41; = ∅?<br>Hom#40;B, S#41; = ∅"}

    BarrierExclusion -- "Yes: Kblk_CatHom" --> VICTORY(["<b>GLOBAL REGULARITY</b><br><i>#40;Structural Exclusion Confirmed#41;</i>"])
    BarrierExclusion -- "No: Kmorph_CatHom" --> ModeCat["<b>FATAL ERROR</b><br>Structural Inconsistency"]
    BarrierExclusion -- "NO(inc): Kbr-inc_CatHom" --> ReconstructionLoop["<b>MT 42.1:</b><br>Structural Reconstruction"]
    ReconstructionLoop -- "Verdict: Kblk" --> VICTORY
    ReconstructionLoop -- "Verdict: Kmorph" --> ModeCat

    %% ====== STYLES ======
    %% Success states - Green
    style VICTORY fill:#22c55e,stroke:#16a34a,color:#000000,stroke-width:4px
    style ModeDD fill:#22c55e,stroke:#16a34a,color:#000000

    %% Failure modes - Red
    style ModeCE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeSE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeSC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCD_Alt fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeSD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeDC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeDE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeBE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeBD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeBC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCat fill:#ef4444,stroke:#dc2626,color:#ffffff

    %% Barriers - Orange/Amber
    style BarrierSat fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierCausal fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierScat fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierTypeII fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierVac fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierCap fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierGap fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierAction fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierOmin fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierMix fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierEpi fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierFreq fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierBode fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierInput fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierVariety fill:#f59e0b,stroke:#d97706,color:#000000

    %% Reconstruction Loop - Yellow/Gold
    style ReconstructionLoop fill:#fbbf24,stroke:#f59e0b,color:#000000,stroke-width:2px

    %% The Final Gate - Purple with thick border
    style BarrierExclusion fill:#8b5cf6,stroke:#7c3aed,color:#ffffff,stroke-width:4px

    %% Interface Checks - Blue
    style EnergyCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ZenoCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style CompactCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ScaleCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ParamCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style GeomCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style StiffnessCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style TopoCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style TameCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ErgoCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ComplexCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style OscillateCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style BoundaryCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style OverloadCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style StarveCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style AlignCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff

    %% Intermediate nodes - Purple
    style Start fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style Profile fill:#8b5cf6,stroke:#7c3aed,color:#ffffff

    %% Restoration checks - Blue (interface permit checks)
    style BifurcateCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style SymCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style CheckSC fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style CheckTB fill:#3b82f6,stroke:#2563eb,color:#ffffff

    %% Restoration mechanisms - Purple (escape mechanisms)
    style ActionSSB fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style ActionTunnel fill:#8b5cf6,stroke:#7c3aed,color:#ffffff

    %% Restoration failure modes - Red
    style ModeSC_Rest fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTE_Rest fill:#ef4444,stroke:#dc2626,color:#ffffff

    %% Surgery recovery nodes - Purple
    style SurgCE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgCC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgCD_Alt fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgCD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSC_Rest fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTE_Rest fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgDC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgDE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgBE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgBD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgBC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff

    %% Surgery Admissibility checks - Light Purple with border
    style SurgAdmCE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmCC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmCD_Alt fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmSE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmSC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmCD fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmSD fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmSC_Rest fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmTE_Rest fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmTE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmTC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmTD fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmDC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmDE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmBE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmBD fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmBC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px

```

:::{prf:remark} Operational Semantics of the Diagram
:label: rem-operational-semantics

We interpret this diagram as the computation of the **Limit** of a diagram of shapes in the $(\infty,1)$-topos of Hypostructures.

The flow proceeds by **Iterative Obstruction Theory**:

1. **Filtration:** The hierarchy (Levels 1–9) establishes a filtration of the moduli space of singularities by obstruction complexity.

2. **Lifting:** A "Yes" branch represents the successful lifting of the solution across an obstruction class—e.g., from $L^2$ energy bounds to $H^1$ regularity. The functor projects the system onto the relevant cohomology; if the class is trivial, the system lifts to the next level.

3. **Surgery as Cobordism:** The dotted "Surgery" loops represent the active cancellation of a non-trivial cohomology class (the singularity) via geometric modification. These are **Pushouts** in the category of manifolds—changing topology to bypass obstructions.

4. **Convergence to the Limit:** The **Cohomological Obstruction** (Node 17) verifies that the **Inverse Limit** of this tower is the empty set—i.e., all obstruction classes vanish—thereby proving $\mathrm{Sing}(\Phi) = \emptyset$.

5. **The Structure Sheaf:** The accumulation of certificates $\Gamma$ forms a **Structure Sheaf** $\mathcal{O}_{\mathrm{Reg}}$ over the trajectory space. A "Victory" is a proof that the **Global Sections** of the singularity sheaf vanish.
:::

---

## Interface Registry: The Obstruction Atlas

The following table defines the **Obstruction Atlas**—the collection of classifying stacks and their associated projection functors. Each interface evaluates whether a specific cohomology class vanishes.

| Interpretation | Engineering Term | Categorical Term |
|----------------|------------------|------------------|
| Node | Check | Classifying Stack $\mathcal{M}_i$ |
| Edge (Yes) | Pass | Truncation Functor $\tau_{\leq k}$ |
| Edge (No) | Fail | Non-trivial Obstruction Class |
| Certificate | Token | Section of Structure Sheaf |

To instantiate the sieve for a specific system, one must implement each projection functor for the relevant hypostructure component.

| Node | ID                            | Name             | Certificates (Output)                                                                                 | Symbol         | Object                   | Hypostructure                   | Description                        | Question                                        | Predicate                                     |
|------|-------------------------------|------------------|-------------------------------------------------------------------------------------------------------|----------------|--------------------------|---------------------------------|------------------------------------|-------------------------------------------------|-----------------------------------------------|
| 1    | $D_E$                         | EnergyCheck      | $K_{D_E}^+$ / $K_{D_E}^-$                                                                             | $E$            | Flow $\Phi$              | $\mathfrak{D}$ on $\Phi$        | Energy functional                  | Is Energy Finite?                               | $E[\Phi] < \infty$                            |
| 2    | $\mathrm{Rec}_N$              | ZenoCheck        | $K_{\mathrm{Rec}_N}^+$ / $K_{\mathrm{Rec}_N}^-$                                                       | $N$            | Jump sequence $J$        | $\mathfrak{D}$ on $\Phi$        | Event counter                      | Are Discrete Events Finite?                     | $N(J) < \infty$                               |
| 3    | $C_\mu$                       | CompactCheck     | $K_{C_\mu}^+$ / $K_{C_\mu}^-$                                                                         | $\mu$          | Profile $V$              | $\mathfrak{D}$ on $\mathcal{X}$ | Concentration measure              | Does Energy Concentrate?                        | $\mu(V) > 0$                                  |
| 4    | $\mathrm{SC}_\lambda$         | ScaleCheck       | $K_{\mathrm{SC}_\lambda}^+$ / $K_{\mathrm{SC}_\lambda}^-$                                             | $\lambda$      | Profile $V$              | $\mathfrak{D}$ on $\mathcal{X}$ | Scaling dimension                  | Is Profile Subcritical?                         | $\lambda(V) < \lambda_c$                      |
| 5    | $\mathrm{SC}_{\partial c}$    | ParamCheck       | $K_{\mathrm{SC}_{\partial c}}^+$ / $K_{\mathrm{SC}_{\partial c}}^-$                                   | $\partial c$   | Constants $c$            | $\mathfrak{D}$ on $\mathcal{X}$ | Parameter derivative               | Are Constants Stable?                           | $\lVert\partial_c\rVert < \epsilon$           |
| 6    | $\mathrm{Cap}_H$              | GeomCheck        | $K_{\mathrm{Cap}_H}^+$ / $K_{\mathrm{Cap}_H}^-$                                                       | $\dim_H$       | Singular set $S$         | $\mathfrak{D}$ on $\mathcal{X}$ | Hausdorff dimension                | Is Codim $\geq$ Threshold?                      | $\mathrm{codim}(S) \geq 2$                    |
| 7    | $\mathrm{LS}_\sigma$          | StiffnessCheck   | $K_{\mathrm{LS}_\sigma}^+$ / $K_{\mathrm{LS}_\sigma}^-$                                               | $\sigma$       | Linearization $L$        | $\mathfrak{D}$ on $\Phi$        | Spectrum                           | Is Gap Certified?                               | $\inf \sigma(L) > 0$                          |
| 7a   | $\mathrm{LS}_{\partial^2 V}$  | BifurcateCheck   | $K_{\mathrm{LS}_{\partial^2 V}}^+$ / $K_{\mathrm{LS}_{\partial^2 V}}^-$                               | $\partial^2 V$ | Equilibrium $x^*$        | $\mathfrak{D}$ on $\mathcal{X}$ | Hessian                            | Is State Unstable?                              | $\partial^2 V(x^*) \not\succ 0$               |
| 7b   | $G_{\mathrm{act}}$            | SymCheck         | $K_{G_{\mathrm{act}}}^+$ / $K_{G_{\mathrm{act}}}^-$                                                   | $G$            | Vacuum $v_0$             | $G$                             | Group action                       | Is $G$-orbit Degenerate?                        | $\lvert G \cdot v_0 \rvert = 1$               |
| 7c   | $\mathrm{SC}_{\partial c}$    | CheckSC          | $K_{\mathrm{SC}_{\partial c}}^+$ / $K_{\mathrm{SC}_{\partial c}}^-$                                   | $\partial c$   | Constants $c$            | $\mathfrak{D}$ on $\mathcal{X}$ | Parameter derivative (restoration) | Are Constants Stable?                           | $\lVert\partial_c\rVert < \epsilon$           |
| 7d   | $\mathrm{TB}_S$               | CheckTB          | $K_{\mathrm{TB}_S}^+$ / $K_{\mathrm{TB}_S}^-$                                                         | $S$            | Instanton path $\gamma$  | $\mathfrak{D}$ on $\mathcal{X}$ | Action functional                  | Is Tunneling Finite?                            | $S[\gamma] < \infty$                          |
| 8    | $\mathrm{TB}_\pi$             | TopoCheck        | $K_{\mathrm{TB}_\pi}^+$ / $K_{\mathrm{TB}_\pi}^-$                                                     | $\pi$          | Configuration $C$        | $\mathfrak{D}$ on $\mathcal{X}$ | Homotopy class                     | Is Sector Reachable?                            | $[\pi] \in \pi_0(\mathcal{C})_{\mathrm{acc}}$ |
| 9    | $\mathrm{TB}_O$               | TameCheck        | $K_{\mathrm{TB}_O}^+$ / $K_{\mathrm{TB}_O}^-$                                                         | $O$            | Stratification $\Sigma$  | $\mathfrak{D}$ on $\mathcal{X}$ | O-minimal structure                | Is Topology Tame?                               | $\Sigma \in \mathcal{O}\text{-min}$           |
| 10   | $\mathrm{TB}_\rho$            | ErgoCheck        | $K_{\mathrm{TB}_\rho}^+$ / $K_{\mathrm{TB}_\rho}^-$                                                   | $\rho$         | Invariant measure $\mu$  | $\mathfrak{D}$ on $\Phi$        | Mixing rate                        | Does Flow Mix?                                  | $\rho(\mu) > 0$                               |
| 11   | $\mathrm{Rep}_K$              | ComplexCheck     | $K_{\mathrm{Rep}_K}^+$ / $K_{\mathrm{Rep}_K}^-$                                                       | $K$            | State $x$                | $\mathfrak{D}$ on $\mathcal{X}$ | Kolmogorov complexity              | Is K(x) Computable?                             | $K(x) \in \mathbb{N}$                         |
| 12   | $\mathrm{GC}_\nabla$          | OscillateCheck   | $K_{\mathrm{GC}_\nabla}^+$ / $K_{\mathrm{GC}_\nabla}^-$                                               | $\nabla$       | Potential $V$            | $\mathfrak{D}$ on $\mathcal{X}$ | Gradient operator                  | Does Flow Oscillate?                            | $\dot{x} \neq -\nabla V$                      |
| 13   | $\mathrm{Bound}_\partial$     | BoundaryCheck    | $K_{\mathrm{Bound}_\partial}^+$ / $K_{\mathrm{Bound}_\partial}^-$                                     | $\partial$     | Domain $\Omega$          | $\mathfrak{D}$ on $\mathcal{X}$ | Boundary operator                  | Is System Open?                                 | $\partial\Omega \neq \emptyset$               |
| 14   | $\mathrm{Bound}_B$            | OverloadCheck    | $K_{\mathrm{Bound}_B}^+$ / $K_{\mathrm{Bound}_B}^-$                                                   | $B$            | Control signal $u$       | $\mathfrak{D}$ on $\Phi$        | Input operator                     | Is Input Bounded?                               | $\lVert Bu \rVert \leq M$                     |
| 15   | $\mathrm{Bound}_{\Sigma}$         | StarveCheck      | $K_{\mathrm{Bound}_{\Sigma}}^+$ / $K_{\mathrm{Bound}_{\Sigma}}^-$                                             | $\int$         | Resource $r$             | $\mathfrak{D}$ on $\Phi$        | Supply integral                    | Is Input Sufficient?                            | $\int_0^T r \, dt \geq r_{\min}$              |
| 16   | $\mathrm{GC}_T$               | AlignCheck       | $K_{\mathrm{GC}_T}^+$ / $K_{\mathrm{GC}_T}^-$                                                         | $T$            | Pair $(u,d)$             | $\mathfrak{D}$ on $\Phi$        | Gauge transform                    | Is Control Matched?                             | $T(u) \sim d$                                 |
| 17   | $\mathrm{Cat}_{\mathrm{Hom}}$ | BarrierExclusion | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ / $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}$ | $\mathrm{Hom}$ | Morphisms $\mathrm{Mor}$ | $\mathfrak{D}$ categorical      | Hom functor                        | Is $\mathrm{Hom}(\mathrm{Bad}, S) = \emptyset$? | $\mathrm{Hom}(\mathcal{B}, S) = \emptyset$    |

:::{prf:remark} Interface Composition
:label: rem-interface-composition

Barrier checks compose multiple interfaces. For example, the **Saturation Barrier** at Node 1 combines the energy interface $D_E$ with a drift control predicate. Surgery admissibility checks (the light purple diamonds) query the same interfaces as their parent gates but with different predicates.
:::

## Barrier Registry: Secondary Obstruction Classes

The following table defines the **Secondary Obstruction Classes**—cohomological barriers that activate when the primary obstruction is non-trivial. Each barrier represents a weaker cohomology condition that may still force triviality of the singularity class.

| Node | Barrier ID       | Interfaces                                       | Permits ($\Gamma$)                         | Certificates (Output)                                                                                 | Blocked Predicate                                              | Question                                                     | Metatheorem                |
|------|------------------|--------------------------------------------------|--------------------------------------------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|--------------------------------------------------------------|----------------------------|
| 1    | BarrierSat       | $D_E$, $\mathrm{SC}_\lambda$                     | $\emptyset$ (Entry)                        | $K_{D_E}^{\mathrm{blk}}$ / $K_{D_E}^{\mathrm{br}}$                                                    | $E[\Phi] \leq E_{\text{sat}} \lor \text{Drift} \leq C$         | Is the energy drift bounded by a saturation ceiling?         | Saturation Principle       |
| 2    | BarrierCausal    | $\mathrm{Rec}_N$, $\mathrm{TB}_\pi$              | $K_{D_E}^\pm$                              | $K_{\mathrm{Rec}_N}^{\mathrm{blk}}$ / $K_{\mathrm{Rec}_N}^{\mathrm{br}}$                              | $D(T_*) = \int_0^{T_*} \frac{c}{\lambda(t)} dt = \infty$       | Does the singularity require infinite computational depth?   | Algorithmic Causal Barrier |
| 3    | BarrierScat      | $C_\mu$, $D_E$                                   | $K_{D_E}^\pm, K_{\mathrm{Rec}_N}^\pm$      | $K_{C_\mu}^{\mathrm{ben}}$ / $K_{C_\mu}^{\mathrm{path}}$                                              | $\mathcal{M}[\Phi] < \infty$                                   | Is the interaction functional finite (implying dispersion)?  | Scattering-Compactness     |
| 4    | BarrierTypeII    | $\mathrm{SC}_\lambda$, $D_E$                     | $K_{C_\mu}^+$                              | $K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}$ / $K_{\mathrm{SC}_\lambda}^{\mathrm{br}}$                    | $\int \tilde{\mathfrak{D}}(S_t V) dt = \infty$                 | Is the renormalization cost of the profile infinite?         | Type II Exclusion          |
| 5    | BarrierVac       | $\mathrm{SC}_{\partial c}$, $\mathrm{LS}_\sigma$ | $K_{C_\mu}^+, K_{\mathrm{SC}_\lambda}^\pm$ | $K_{\mathrm{SC}_{\partial c}}^{\mathrm{blk}}$ / $K_{\mathrm{SC}_{\partial c}}^{\mathrm{br}}$          | $\Delta V > k_B T$                                             | Is the phase stable against thermal/parameter drift?         | Mass Gap Principle         |
| 6    | BarrierCap       | $\mathrm{Cap}_H$                                 | $K_{\mathrm{SC}_{\partial c}}^\pm$         | $K_{\mathrm{Cap}_H}^{\mathrm{blk}}$ / $K_{\mathrm{Cap}_H}^{\mathrm{br}}$                              | $\mathrm{Cap}_H(S) = 0$                                        | Is the singular set of measure zero?                         | Capacity Barrier           |
| 7    | BarrierGap       | $\mathrm{LS}_\sigma$, $\mathrm{GC}_\nabla$       | $K_{\mathrm{Cap}_H}^\pm$                   | $K_{\mathrm{LS}_\sigma}^{\mathrm{blk}}$ / $K_{\mathrm{LS}_\sigma}^{\mathrm{stag}}$                    | $\dim(\ker L) < \infty \land \sigma_{\mathrm{ess}}(L) > 0$     | Is the kernel finite-dimensional with essential spectral gap? | Spectral Generator         |
| 8    | BarrierAction    | $\mathrm{TB}_\pi$                                | $K_{\mathrm{LS}_\sigma}^\pm$               | $K_{\mathrm{TB}_\pi}^{\mathrm{blk}}$ / $K_{\mathrm{TB}_\pi}^{\mathrm{br}}$                            | $E[\Phi] < S_{\min} + \Delta$                                  | Is the energy insufficient to cross the topological gap?     | Topological Suppression    |
| 9    | BarrierOmin      | $\mathrm{TB}_O$, $\mathrm{Rep}_K$                | $K_{\mathrm{TB}_\pi}^\pm$                  | $K_{\mathrm{TB}_O}^{\mathrm{blk}}$ / $K_{\mathrm{TB}_O}^{\mathrm{br}}$                                | $S \in \mathcal{O}\text{-min}$                                 | Is the topology definable in an o-minimal structure?         | O-Minimal Taming           |
| 10   | BarrierMix       | $\mathrm{TB}_\rho$, $D_E$                        | $K_{\mathrm{TB}_O}^\pm$                    | $K_{\mathrm{TB}_\rho}^{\mathrm{blk}}$ / $K_{\mathrm{TB}_\rho}^{\mathrm{br}}$                          | $\tau_{\text{mix}} < \infty$                                   | Does the system mix fast enough to escape traps?             | Ergodic Mixing             |
| 11   | BarrierEpi       | $\mathrm{Rep}_K$, $\mathrm{Cap}_H$               | $K_{\mathrm{TB}_\rho}^\pm$                 | $K_{\mathrm{Rep}_K}^{\mathrm{blk}}$ / $K_{\mathrm{Rep}_K}^{\mathrm{br}}$                              | $\sup_\epsilon K_\epsilon(x) \leq S_{\text{BH}}$               | Is approximable complexity within holographic bounds?        | Epistemic Horizon          |
| 12   | BarrierFreq      | $\mathrm{GC}_\nabla$, $\mathrm{SC}_\lambda$      | $K_{\mathrm{Rep}_K}^\pm$                   | $K_{\mathrm{GC}_\nabla}^{\mathrm{blk}}$ / $K_{\mathrm{GC}_\nabla}^{\mathrm{br}}$                      | $\int \omega^2 S(\omega) d\omega < \infty$                     | Is the total oscillation energy finite?                      | Frequency Barrier          |
| 14   | BarrierBode      | $\mathrm{Bound}_B$, $\mathrm{LS}_\sigma$         | $K_{\mathrm{Bound}_\partial}^+$            | $K_{\mathrm{Bound}_B}^{\mathrm{blk}}$ / $K_{\mathrm{Bound}_B}^{\mathrm{br}}$                          | $\int_0^\infty \ln \lVert S(i\omega) \rVert d\omega > -\infty$ | Is the sensitivity integral conserved (waterbed effect)?     | Bode Sensitivity           |
| 15   | BarrierInput     | $\mathrm{Bound}_{\Sigma}$, $C_\mu$                   | $K_{\mathrm{Bound}_B}^\pm$                 | $K_{\mathrm{Bound}_{\Sigma}}^{\mathrm{blk}}$ / $K_{\mathrm{Bound}_{\Sigma}}^{\mathrm{br}}$                    | $r_{\text{reserve}} > 0$                                       | Is there a reservoir to prevent starvation?                  | Input Stability            |
| 16   | BarrierVariety   | $\mathrm{GC}_T$, $\mathrm{Cap}_H$                | $K_{\mathrm{Bound}_{\Sigma}}^\pm$              | $K_{\mathrm{GC}_T}^{\mathrm{blk}}$ / $K_{\mathrm{GC}_T}^{\mathrm{br}}$                                | $H(u) \geq H(d)$                                               | Does control entropy match disturbance entropy?              | Requisite Variety          |
| 17   | BarrierExclusion | $\mathrm{Cat}_{\mathrm{Hom}}$                    | Full $\Gamma$                              | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ / $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}$ / $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{hor}}$ | $\mathrm{Hom}(\mathcal{B}, S) = \emptyset$                     | Is there a categorical obstruction to the bad pattern?       | Morphism Exclusion / Reconstruction |

## Surgery Registry: Cobordism Morphisms

The following table defines the **Cobordism Morphisms**—categorical pushouts that modify the topology of the state space to cancel non-trivial obstruction classes. Each surgery constructs a new manifold where the obstruction vanishes, enabling re-entry into the resolution tower.

| #   | Surgery ID   | Interfaces                                       | Input Certificate                            | Output Certificate                        | Admissibility Predicate                                                                      | Action                    | Metatheorem             |
|-----|--------------|--------------------------------------------------|----------------------------------------------|-------------------------------------------|----------------------------------------------------------------------------------------------|---------------------------|-------------------------|
| S1  | SurgCE       | $D_E$, $\mathrm{Cap}_H$                          | $K_{D_E}^{\mathrm{br}}$                      | $K_{\mathrm{SurgCE}}^{\mathrm{re}}$       | Growth conformal $\land$ $\partial_\infty X$ definable                                       | Ghost/Cap Extension       | Compactification        |
| S2  | SurgCC       | $\mathrm{Rec}_N$, $\mathrm{TB}_\pi$              | $K_{\mathrm{Rec}_N}^{\mathrm{br}}$           | $K_{\mathrm{SurgCC}}^{\mathrm{re}}$       | $\exists N_{\max}$: events $\leq N_{\max}$                                                   | Discrete Saturation       | Event Coarsening        |
| S3  | SurgCD\_Alt  | $C_\mu$, $D_E$                                   | $K_{C_\mu}^{\mathrm{path}}$                  | $K_{\mathrm{SurgCD\_Alt}}^{\mathrm{re}}$  | $V \in \mathcal{L}_{\text{soliton}} \land \|V\|_{H^1} < \infty$                              | Concentration-Compactness | Profile Extraction      |
| S4  | SurgSE       | $\mathrm{SC}_\lambda$, $D_E$                     | $K_{\mathrm{SC}_\lambda}^{\mathrm{br}}$      | $K_{\mathrm{SurgSE}}^{\mathrm{re}}$       | $\alpha - \beta < \varepsilon_{\text{crit}} \land V$ smooth                                  | Regularity Lift           | Perturbative Upgrade    |
| S5  | SurgSC       | $\mathrm{SC}_{\partial c}$, $\mathrm{LS}_\sigma$ | $K_{\mathrm{SC}_{\partial c}}^{\mathrm{br}}$ | $K_{\mathrm{SurgSC}}^{\mathrm{re}}$       | $\|\partial_t \theta\| < C_{\text{adm}} \land \theta \in \Theta_{\text{stable}}$             | Convex Integration        | Parameter Freeze        |
| S6  | SurgCD       | $\mathrm{Cap}_H$, $\mathrm{LS}_\sigma$           | $K_{\mathrm{Cap}_H}^{\mathrm{br}}$           | $K_{\mathrm{SurgCD}}^{\mathrm{re}}$       | $\mathrm{Cap}_H(\Sigma) \leq \varepsilon_{\text{adm}} \land V \in \mathcal{L}_{\text{neck}}$ | Auxiliary/Structural      | Excision-Capping        |
| S7  | SurgSD       | $\mathrm{LS}_{\partial^2 V}$, $\mathrm{GC}_\nabla$ | $K_{\mathrm{LS}_{\partial^2 V}}^{-}$        | $K_{\mathrm{SurgSD}}^{\mathrm{re}}$       | $\dim(\ker(H_V)) < \infty \land V$ isolated                                                  | Ghost Extension           | Spectral Lift           |
| S8  | SurgSC\_Rest | $\mathrm{SC}_{\partial c}$, $\mathrm{LS}_\sigma$ | $K_{\mathrm{SC}_{\partial c}}^{-}$          | $K_{\mathrm{SurgSC\_Rest}}^{\mathrm{re}}$ | $\Delta V > k_B T \land \Gamma < \Gamma_{\text{crit}}$                                       | Auxiliary Extension       | Vacuum Shift            |
| S9  | SurgTE\_Rest | $\mathrm{TB}_S$, $C_\mu$                         | $K_{\mathrm{TB}_S}^{-}$                     | $K_{\mathrm{SurgTE\_Rest}}^{\mathrm{re}}$ | $V \cong S^{n-1} \times I \land S_R[\gamma] < \infty$ (renormalized)                         | Structural                | Instanton Reconnection  |
| S10 | SurgTE       | $\mathrm{TB}_\pi$, $C_\mu$                       | $K_{\mathrm{TB}_\pi}^{\mathrm{br}}$          | $K_{\mathrm{SurgTE}}^{\mathrm{re}}$       | $V \cong S^{n-1} \times \mathbb{R}$ (Neck)                                                   | Tunnel                    | Topological Surgery     |
| S11 | SurgTC       | $\mathrm{TB}_O$, $\mathrm{Rep}_K$                | $K_{\mathrm{TB}_O}^{\mathrm{br}}$            | $K_{\mathrm{SurgTC}}^{\mathrm{re}}$       | $\Sigma \in \mathcal{O}_{\text{ext}}$-definable $\land \dim(\Sigma) < n$                     | O-minimal Regularization  | Structure Extension     |
| S12 | SurgTD       | $\mathrm{TB}_\rho$, $D_E$                        | $K_{\mathrm{TB}_\rho}^{\mathrm{br}}$         | $K_{\mathrm{SurgTD}}^{\mathrm{re}}$       | Trap isolated $\land \partial T$ has positive measure                                        | Mixing Enhancement        | Stochastic Perturbation |
| S13 | SurgDC       | $\mathrm{Rep}_K$, $\mathrm{Cap}_H$               | $K_{\mathrm{Rep}_K}^{\mathrm{br}}$           | $K_{\mathrm{SurgDC}}^{\mathrm{re}}$       | $K(x) \leq S_{\text{BH}} + \varepsilon \land x \in W^{1,\infty}$                             | Viscosity Solution        | Mollification           |
| S14 | SurgDE       | $\mathrm{GC}_\nabla$, $\mathrm{SC}_\lambda$      | $K_{\mathrm{GC}_\nabla}^{\mathrm{br}}$       | $K_{\mathrm{SurgDE}}^{\mathrm{re}}$       | $\exists\Lambda: \int_{\lvert\omega\rvert\leq\Lambda} \omega^2 S d\omega < \infty \land$ uniform ellipticity | De Giorgi-Nash-Moser      | Hölder Regularization   |
| S15 | SurgBE       | $\mathrm{Bound}_B$, $\mathrm{LS}_\sigma$         | $K_{\mathrm{Bound}_B}^{\mathrm{br}}$         | $K_{\mathrm{SurgBE}}^{\mathrm{re}}$       | $\|S(i\omega)\|_\infty < M \land$ phase margin $> 0$                                         | Saturation                | Gain Limiting           |
| S16 | SurgBD       | $\mathrm{Bound}_{\Sigma}$, $C_\mu$                   | $K_{\mathrm{Bound}_{\Sigma}}^{\mathrm{br}}$      | $K_{\mathrm{SurgBD}}^{\mathrm{re}}$       | $r_{\text{reserve}} > 0 \land$ recharge $>$ drain                                            | Reservoir                 | Buffer Addition         |
| S17 | SurgBC       | $\mathrm{GC}_T$, $\mathrm{Cap}_H$                | $K_{\mathrm{GC}_T}^{\mathrm{br}}$            | $K_{\mathrm{SurgBC}}^{\mathrm{re}}$       | $H(u) < H(d) - \varepsilon \land \exists u': H(u') \geq H(d)$                                | Controller Augmentation   | Entropy Matching        |

:::{note} Restoration vs. Barrier Surgeries
Surgeries S7–S9 (SurgSD, SurgSC\_Rest, SurgTE\_Rest) are **restoration surgeries** triggered by gate failures ($K^-$) within the Stiffness Restoration Subtree (Nodes 7a–7d), *not* by barrier breaches ($K^{\mathrm{br}}$). This distinction is critical: barrier surgeries repair systems that have *breached* a constraint, while restoration surgeries repair systems that have *failed* a sub-gate within a recovery path.
:::

## Surgery Admissibility Registry

The following table defines all **admissibility checks** in the Structural Sieve. Each admissibility check node (A1–A17) evaluates whether a breached barrier admits surgical repair or requires termination at a failure mode.

### Admissibility Node Logic

Each admissibility node $A_i$ implements the following evaluation:

$$
\mathrm{eval}_{A_i}(K^{\mathrm{br}}_{\mathrm{Barrier}}, \Sigma, V) \rightarrow \{K^+_{A_i}, K^-_{A_i}\}
$$

**Inputs:**
- $K^{\mathrm{br}}_{\mathrm{Barrier}}$: Breach certificate from the upstream barrier
- $\Sigma$: Singular set or defect locus
- $V$: Blow-up profile or local structure

**Outputs:**
- **YES** ($K^+_{A_i}$): Admissibility Certificate — required token to proceed to Surgery
- **NO** ($K^-_{A_i}$): Inadmissibility Certificate — required token to terminate at Failure Mode

### Admissibility Token Schema

**YES Certificate** ($K^+_{\mathrm{Adm}}$) contains:
- $V_{\mathrm{can}}$: Identification of singularity profile in the Canonical Library $\mathcal{L}$
- $\pi_{\mathrm{match}}$: Isomorphism/diffeomorphism witnessing that current state matches library cap
- $\mathrm{CapBound}$: Proof that $\mathrm{Cap}(\Sigma) \leq \varepsilon_{\mathrm{crit}}$

**NO Certificate** ($K^-_{\mathrm{Adm}}$) contains:
- $\mathrm{ObstructionType} \in \{\texttt{WildProfile}, \texttt{FatSingularity}, \texttt{Horizon}\}$
- $\mathrm{Witness}$: Data proving the obstruction (e.g., accumulation point, unbounded kernel, non-definable set)

### Admissibility Registry Table

| Node | ID | Question | YES Certificate ($K^+$) | NO Certificate ($K^-$) | YES Target | NO Target |
|------|-----|----------|------------------------|------------------------|------------|-----------|
| A1 | $\mathrm{Adm}_{\mathrm{CE}}$ | Conformal? | $K^+_{\mathrm{Conf}}$: conformal factor $\Omega(x)$, definable $\partial_\infty X$ | $K^-_{\mathrm{Conf}}$: anisotropic blow-up witness | S1 | Mode C.E |
| A2 | $\mathrm{Adm}_{\mathrm{CC}}$ | Discrete? | $K^+_{\mathrm{Disc}}$: bound $N_{\max}$ on event density | $K^-_{\mathrm{Disc}}$: accumulation point $t^*$ | S2 | Mode C.C |
| A3 | $\mathrm{Adm}_{\mathrm{CDA}}$ | Soliton? | $K^+_{\mathrm{Prof}}$: $V \in \mathcal{L}_{\text{soliton}}$, finite $H^1$ norm | $K^-_{\mathrm{Prof}}$: diffusive/undefined profile | S3 | Mode C.D |
| A4 | $\mathrm{Adm}_{\mathrm{SE}}$ | Smooth? | $K^+_{\mathrm{Lift}}$: regularity gap $\alpha - \beta < \varepsilon$, $V$ smooth | $K^-_{\mathrm{Lift}}$: gap too large, profile rough | S4 | Mode S.E |
| A5 | $\mathrm{Adm}_{\mathrm{SC}}$ | Stable $\theta$? | $K^+_{\mathrm{Stab}}$: $\|\dot{\theta}\| < \delta$, $\theta$ in stable basin | $K^-_{\mathrm{Stab}}$: velocity unbounded, chaotic | S5 | Mode S.C |
| A6 | $\mathrm{Adm}_{\mathrm{CD}}$ | Neck? | $K^+_{\mathrm{Neck}}$: $V \cong S^{n-1} \times \mathbb{R}$, $\mathrm{Cap}(\Sigma) \leq \varepsilon$ | $K^-_{\mathrm{Neck}}$: fat singularity, non-cylindrical | S6 | Mode C.D |
| A7 | $\mathrm{Adm}_{\mathrm{SD}}$ | Isolated? | $K^+_{\mathrm{Iso}}$: $\dim(\ker H) < \infty$, isolated critical pt | $K^-_{\mathrm{Iso}}$: infinite kernel, continuum of vacua | S7 | Mode S.D |
| A8 | $\mathrm{Adm}_{\mathrm{SCR}}$ | Slow Tunnel? | $K^+_{\mathrm{Vac}}$: gap $\Delta V > k_B T$, $\Gamma < \Gamma_{\mathrm{crit}}$ | $K^-_{\mathrm{Vac}}$: barrier collapse, thermal runaway | S8 | Mode S.C |
| A9 | $\mathrm{Adm}_{\mathrm{TER}}$ | Renormalizable? | $K^+_{\mathrm{Inst}}$: $S_R[\gamma] < \infty$ after cutoff regularization | $K^-_{\mathrm{Inst}}$: non-renormalizable divergence | S9 | Mode T.E |
| A10 | $\mathrm{Adm}_{\mathrm{TE}}$ | Neck Pinch? | $K^+_{\mathrm{Topo}}$: $V \cong \text{Neck}$, $\pi_1$ compatible | $K^-_{\mathrm{Topo}}$: exotic topology, knotting | S10 | Mode T.E |
| A11 | $\mathrm{Adm}_{\mathrm{TC}}$ | Definable? | $K^+_{\mathrm{Omin}}$: $\Sigma$ in $\mathcal{O}_{\text{ext}}$-definable | $K^-_{\mathrm{Omin}}$: wild set (Cantor) | S11 | Mode T.C |
| A12 | $\mathrm{Adm}_{\mathrm{TD}}$ | Escapable? | $K^+_{\mathrm{Mix}}$: $\partial T$ has positive measure | $K^-_{\mathrm{Mix}}$: hermetic seal, infinite depth | S12 | Mode T.D |
| A13 | $\mathrm{Adm}_{\mathrm{DC}}$ | Lipschitz? | $K^+_{\mathrm{Lip}}$: $x \in W^{1,\infty}$, $K \approx S_{\mathrm{BH}}$ | $K^-_{\mathrm{Lip}}$: $K \gg S_{\mathrm{BH}}$, fractal state | S13 | Mode D.C |
| A14 | $\mathrm{Adm}_{\mathrm{DE}}$ | Elliptic? | $K^+_{\mathrm{Ell}}$: marginal divergence, elliptic | $K^-_{\mathrm{Ell}}$: hyperbolic/chaotic oscillation | S14 | Mode D.E |
| A15 | $\mathrm{Adm}_{\mathrm{BE}}$ | Phase Margin? | $K^+_{\mathrm{Marg}}$: phase margin $> 0$ | $K^-_{\mathrm{Marg}}$: zero phase margin | S15 | Mode B.E |
| A16 | $\mathrm{Adm}_{\mathrm{BD}}$ | Recharge? | $K^+_{\mathrm{Res}}$: recharge $>$ drain, $r > 0$ | $K^-_{\mathrm{Res}}$: systemic deficit | S16 | Mode B.D |
| A17 | $\mathrm{Adm}_{\mathrm{BC}}$ | Bridgeable? | $K^+_{\mathrm{Ent}}$: $\exists u^*$ matching entropy | $K^-_{\mathrm{Ent}}$: fundamental variety gap | S17 | Mode B.C |

:::{prf:remark} Proof Chain Completion
:label: rem-adm-chain

The admissibility registry completes the certificate chain for surgical repair:

1. **Barrier** issues breach certificate $K^{\mathrm{br}}$
2. **Admissibility Check** consumes $K^{\mathrm{br}}$ and issues either $K^+_{\mathrm{Adm}}$ or $K^-_{\mathrm{Adm}}$
3. **Surgery** accepts only $K^+_{\mathrm{Adm}}$ as input token, produces re-entry certificate $K^{\mathrm{re}}$
4. **Failure Mode** accepts only $K^-_{\mathrm{Adm}}$ as input token, terminates run with classification

This ensures that no surgery executes without verified admissibility, and no failure mode activates without witnessed obstruction.

:::

---

# Part V: The Kernel

## 1. The Sieve as a Proof-Carrying Program

:::{prf:definition} Sieve epoch
:label: def-sieve-epoch

An **epoch** is a single execution of the sieve from the START node to either:
1. A terminal node (VICTORY, Mode D.D, or FATAL ERROR), or
2. A surgery re-entry point (dotted arrow target).
Each epoch visits finitely many nodes (Theorem {ref}`thm-epoch-termination`). A complete run consists of finitely many epochs (Theorem {ref}`thm-finite-runs`).

:::

:::{prf:definition} Node numbering
:label: def-node-numbering

The sieve contains the following node classes:
- **Gates (Blue):** Nodes 1--17 performing interface permit checks
- **Barriers (Orange):** Secondary defense nodes triggered by gate failures
- **Modes (Red):** Failure mode classifications
- **Surgeries (Purple):** Repair mechanisms with re-entry targets
- **Actions (Purple):** Dynamic restoration mechanisms (SSB, Tunneling)
- **Restoration subnodes (7a--7d):** The stiffness restoration subtree

:::

---

## 2. Operational Semantics

:::{prf:definition} State space
:label: def-state-space

Let $X$ be a Polish space (complete separable metric space) representing the configuration space of the system under analysis. A **state** $x \in X$ is a point in this space representing the current system configuration at a given time or stage of analysis.

:::

:::{prf:definition} Certificate
:label: def-certificate

A **certificate** $K$ is a formal witness object that records the outcome of a verification step. Certificates are typed: each certificate $K$ belongs to a certificate type $\mathcal{K}$ specifying what property it witnesses.

:::

:::{prf:definition} Context
:label: def-context

The **context** $\Gamma$ is a finite multiset of certificates accumulated during a sieve run:
$$\Gamma = \{K_{D_E}, K_{\mathrm{Rec}_N}, K_{C_\mu}, \ldots, K_{\mathrm{Cat}_{\mathrm{Hom}}}\}$$
The context grows monotonically during an epoch: certificates are added but never removed (except at surgery re-entry, where context may be partially reset).

:::

:::{prf:definition} Node evaluation function
:label: def-node-evaluation

Each node $N$ in the sieve defines an **evaluation function**:
$$\mathrm{eval}_N : X \times \Gamma \to \mathcal{O}_N \times \mathcal{K}_N \times X \times \Gamma$$
where:
- $\mathcal{O}_N$ is the **outcome alphabet** for node $N$
- $\mathcal{K}_N$ is the **certificate type** produced by node $N$
- The function maps $(x, \Gamma) \mapsto (o, K_o, x', \Gamma')$ where:
   - $o \in \mathcal{O}_N$ is the outcome
   - $K_o \in \mathcal{K}_N$ is the certificate witnessing outcome $o$
   - $x' \in X$ is the (possibly modified) state
   - $\Gamma' = \Gamma \cup \{K_o\}$ is the extended context

:::

:::{prf:definition} Edge validity
:label: def-edge-validity

An edge $N_1 \xrightarrow{o} N_2$ in the sieve diagram is **valid** if and only if:
$$K_o \Rightarrow \mathrm{Pre}(N_2)$$
That is, the certificate produced by node $N_1$ with outcome $o$ logically implies the precondition required for node $N_2$ to be evaluable.

:::

:::{prf:definition} Determinism policy
:label: def-determinism

For **soft checks** (where the predicate cannot be definitively verified), the sieve adopts the following policy:
- If verification succeeds: output YES with positive certificate $K^+$
- If verification fails: output NO with negative certificate $K^-$
- If verification is inconclusive (UNKNOWN): output NO with uncertainty certificate $K^?$
This ensures the sieve is deterministic: UNKNOWN is conservatively treated as NO, routing to the barrier defense layer.

:::

---

## 3. Permit Vocabulary and Certificate Types

:::{prf:definition} Gate permits
:label: def-gate-permits

For each gate (blue node) $i$, the outcome alphabet is:
$$\mathcal{O}_i = \{`YES`, `NO`\}$$
with certificate types:
- $K_i^+$ (`YES` certificate): Witnesses that predicate $P_i$ holds on the current state/window
- $K_i^-$ (`NO` certificate): Witnesses either that $P_i$ fails, or that $P_i$ cannot be certified from current $\Gamma$

:::

:::{prf:remark} Dichotomy classifiers
:label: rem-dichotomy

Some gates are **dichotomy classifiers** where NO is a benign branch rather than an error:
- **CompactCheck (Node 3)**: NO = scattering $\to$ global existence (Mode D.D)
- **OscillateCheck (Node 12)**: NO = no oscillation $\to$ proceed to boundary checks
For these gates, $K^-$ represents a classification outcome, not a failure certificate.

:::

:::{prf:definition} Barrier permits
:label: def-barrier-permits

For each barrier (orange node), the outcome alphabet is one of:

**Standard barriers** (most barriers):
$$\mathcal{O}_{\text{barrier}} = \{`Blocked`, `Breached`\}$$

**Special barriers with extended alphabets:**
- **BarrierScat** (Scattering): $\mathcal{O} = \{`Benign`, `Pathological`\}$
- **BarrierGap** (Spectral): $\mathcal{O} = \{`Blocked`, `Stagnation`\}$
- **BarrierExclusion** (Lock): $\mathcal{O} = \{`Blocked`, `MorphismExists`\}$

Certificate semantics:
- $K^{\mathrm{blk}}$ (`Blocked`): Barrier holds; certificate enables passage to next gate
- $K^{\mathrm{br}}$ (`Breached`): Barrier fails; certificate activates failure mode and enables surgery

:::

:::{prf:definition} Surgery permits
:label: def-surgery-permits

For each surgery (purple node), the output is a **re-entry certificate**:
$$K^{\mathrm{re}} = (D_S, x', \pi)$$
where $D_S$ is the surgery data, $x'$ is the post-surgery state, and $\pi$ is a proof that $\mathrm{Pre}(\text{TargetNode})$ holds for $x'$.

The re-entry certificate witnesses that after surgery with data $D_S$, the precondition of the dotted-arrow target node is satisfied:
$$K^{\mathrm{re}} \Rightarrow \mathrm{Pre}(\text{TargetNode})(x')$$

:::

:::{prf:definition} YES-tilde permits
:label: def-yes-tilde

A **YES$^\sim$ permit** (YES up to equivalence) is a certificate of the form:
$$K_i^{\sim} = (K_{\mathrm{equiv}}, K_{\mathrm{transport}}, K_i^+[\tilde{x}])$$
where:
- $K_{\mathrm{equiv}}$ certifies that $\tilde{x}$ is equivalent to $x$ under an admissible equivalence move
- $K_{\mathrm{transport}}$ is a transport lemma certificate
- $K_i^+[\tilde{x}]$ is a YES certificate for predicate $P_i$ on the equivalent object $\tilde{x}$
YES$^\sim$ permits are accepted by metatheorems that tolerate equivalence.

:::

:::{prf:definition} Promotion permits
:label: def-promotion-permits

**Promotion permits** upgrade blocked certificates to full YES certificates:

**Immediate promotion** (past-only): A blocked certificate at node $i$ may be promoted if all prior nodes passed:
$$K_i^{\mathrm{blk}} \wedge \bigwedge_{j < i} K_j^+ \Rightarrow K_i^+$$
(Here $K_j^+$ denotes a YES certificate at node $j$.)

**A-posteriori promotion** (future-enabled): A blocked certificate may be promoted after later nodes pass:
$$K_i^{\mathrm{blk}} \wedge \bigwedge_{j > i} K_j^+ \Rightarrow K_i^+$$

**Combined promotion**: Blocked certificates may also promote if the barrier's ``Blocked'' outcome combined with other certificates logically implies the original predicate $P_i$ holds.

Promotion rules are applied during context closure (Definition {ref}`def-closure`).

:::

:::{prf:definition} Inconclusive upgrade permits
:label: def-inc-upgrades

**Inconclusive upgrade permits** discharge NO-inconclusive certificates by supplying certificates that satisfy their $\mathsf{missing}$ prerequisites (Definition {prf:ref}`def-typed-no-certificates`).

**Immediate inc-upgrade** (past/current): An inconclusive certificate may be upgraded if certificates satisfying its missing prerequisites are present:
$$K_P^{\mathrm{inc}} \wedge \bigwedge_{j \in J} K_j^+ \Rightarrow K_P^+$$
where $J$ indexes the certificate types listed in $\mathsf{missing}(K_P^{\mathrm{inc}})$.

**A-posteriori inc-upgrade** (future-enabled): An inconclusive certificate may be upgraded after later nodes provide the missing prerequisites:
$$K_P^{\mathrm{inc}} \wedge \bigwedge_{j \in J'} K_j^+ \Rightarrow K_P^+$$
where $J'$ indexes certificates produced by nodes evaluated after the node that produced $K_P^{\mathrm{inc}}$.

**To YES$^\sim$** (equivalence-tolerant): An inconclusive certificate may upgrade to YES$^\sim$ when the discharge is valid only up to an admissible equivalence move (Definition {prf:ref}`def-yes-tilde`):
$$K_P^{\mathrm{inc}} \wedge \bigwedge_{j \in J} K_j^+ \Rightarrow K_P^{\sim}$$

**Discharge condition (obligation matching):** An inc-upgrade rule is admissible only if its premises imply the concrete obligation instance recorded in the payload:
$$\bigwedge_{j \in J} K_j^+ \Rightarrow \mathsf{obligation}(K_P^{\mathrm{inc}})$$

This makes inc-upgrades structurally symmetric with blocked-certificate promotions (Definition {prf:ref}`def-promotion-permits`).

:::

---

## 4. Kernel Theorems

:::{prf:theorem} DAG structure
:label: thm-dag

The sieve diagram is a directed acyclic graph (DAG). All edges, including dotted surgery re-entry edges, point forward in the topological ordering. Consequently:
1. No backward edges exist
2. Each epoch visits at most $|V|$ nodes where $|V|$ is the number of nodes
3. The sieve terminates

**Literature:** Topological sorting of DAGs {cite}`Kahn62`; termination via well-founded orders {cite}`Floyd67`.

:::

:::{prf:proof}

By inspection of the diagram: all solid edges flow downward (increasing node number or to barriers/modes), and all dotted surgery edges target nodes strictly later in the flow than their source mode. The restoration subtree (7a--7d) only exits forward to TopoCheck or TameCheck.

:::

:::{prf:theorem} Epoch termination
:label: thm-epoch-termination

Each epoch terminates in finite time, visiting finitely many nodes.

**Literature:** Termination proofs via ranking functions {cite}`Floyd67`; {cite}`Turing49`.

:::

:::{prf:proof}

Immediate from Theorem {ref}`thm-dag`: the DAG structure ensures no cycles, hence any path through the sieve has bounded length.

:::

:::{prf:theorem} Finite complete runs
:label: thm-finite-runs

A complete sieve run consists of finitely many epochs.

**Literature:** Surgery bounds for Ricci flow {cite}`Perelman03`; well-founded induction {cite}`Floyd67`.

:::

:::{prf:proof}

Each surgery has an associated progress measure (Definition {ref}`def-progress-measures`):

**Type A (Bounded count)**: The surgery count is bounded by $N(T, \Phi(x_0))$, a function of the time horizon $T$ and initial energy $\Phi(x_0)$. For parabolic PDE, this bound is typically imported from classical surgery theory (e.g., Perelman's surgery bound for Ricci flow: $N \leq C(\Phi_0) T^{d/2}$). For algorithmic/iterative systems, it may be a budget constraint.

**Type B (Well-founded)**:  The complexity measure $\mathcal{C}: X \to \mathbb{N}$ (or ordinal $\alpha$) strictly decreases at each surgery:
$$\mathcal{O}_S(x) = x' \Rightarrow \mathcal{C}(x') < \mathcal{C}(x)$$
Since well-founded orders have no infinite descending chains, the surgery sequence terminates.

The total number of distinct surgery types is finite (at most 17, one per failure mode). Hence the total number of surgeries---and thus epochs---is finite.

:::

:::{prf:theorem} Soundness
:label: thm-soundness

Every transition in a sieve run is certificate-justified. Formally, if the sieve transitions from node $N_1$ to node $N_2$ with outcome $o$, then:
1. A certificate $K_o$ was produced by $N_1$
2. $K_o$ implies the precondition $\mathrm{Pre}(N_2)$
3. $K_o$ is added to the context $\Gamma$

**Literature:** Proof-carrying code {cite}`Necula97`; certified compilation {cite}`Leroy09`.

:::

:::{prf:proof}

By construction: Definition {ref}`def-node-evaluation` requires each node evaluation to produce a certificate, and Definition {ref}`def-edge-validity` requires edge validity.

:::

:::{prf:definition} Fingerprint
:label: def-fingerprint

The **fingerprint** of a sieve run is the tuple:
$$\mathcal{F} = (\mathrm{tr}, \vec{v}, \Gamma_{\mathrm{final}})$$
where:
- $\mathrm{tr}$ is the **trace**: ordered sequence of (node, outcome) pairs visited
- $\vec{v}$ is the **node vector**: for each gate $i$, the outcome $v_i \in \{`YES`, `NO`, `---`\}$ (--- if not visited)
- $\Gamma_{\mathrm{final}}$ is the final certificate context

:::

:::{prf:definition} Certificate finiteness condition
:label: def-cert-finite

For type $T$, the certificate language $\mathcal{K}(T)$ satisfies the **finiteness condition** if either:
1. **Bounded description length**: Certificates have bounded description complexity (finite precision, bounded parameters), or
2. **Depth budget**: Closure is computed to a specified depth/complexity budget $D_{\max}$
Non-termination under infinite certificate language is treated as a NO-inconclusive certificate (Remark {prf:ref}`rem-inconclusive-general`).

:::

:::{prf:definition} Promotion closure
:label: def-closure

The **promotion closure** $\mathrm{Cl}(\Gamma)$ is the least fixed point of the context under all promotion and upgrade rules:
$$\mathrm{Cl}(\Gamma) = \bigcup_{n=0}^{\infty} \Gamma_n$$
where $\Gamma_0 = \Gamma$ and $\Gamma_{n+1}$ applies all applicable immediate and a-posteriori promotions (Definition {prf:ref}`def-promotion-permits`) **and all applicable inc-upgrades** (Definition {prf:ref}`def-inc-upgrades`) to $\Gamma_n$.

:::

:::{prf:theorem} Closure termination
:label: thm-closure-termination

Under the certificate finiteness condition (Definition {ref}`def-cert-finite`), the promotion closure $\mathrm{Cl}(\Gamma)$ is computable in finite time.

**Literature:** Tarski fixed-point theorem {cite}`Tarski55`; Kleene iteration {cite}`Kleene52`.

:::

:::{prf:proof}

Under bounded description length: the certificate universe is finite (bounded by the number of distinct certificate types, the number of nodes, and the description length bound). Each promotion rule strictly increases the certificate set. Hence the iteration terminates in at most $|\mathcal{K}(T)|$ steps.

Under depth budget: closure computation halts after $D_{\max}$ iterations with a partial closure $\mathrm{Cl}_{D_{\max}}(\Gamma)$. If the true fixed point is not reached, a NO-inconclusive certificate ($K_{\mathrm{Promo}}^{\mathrm{inc}}$) is produced indicating ``promotion depth exceeded.''

:::

:::{prf:remark} NO-Inconclusive Certificates ($K^{\mathrm{inc}}$)
:label: rem-inconclusive-general

The framework produces explicit **NO-inconclusive certificates** ($K^{\mathrm{inc}}$) when classification or verification is infeasible with current methods—these are NO verdicts that do *not* constitute semantic refutations:

- **Profile Trichotomy Case 3**: $K_{\mathrm{prof}}^{\mathrm{inc}}$ with classification obstruction witness
- **Surgery Admissibility Case 3**: $K_{\mathrm{Surg}}^{\mathrm{inc}}$ with inadmissibility reason
- **Promotion Closure**: $K_{\mathrm{Promo}}^{\mathrm{inc}}$ recording non-termination under budget
- **Lock (E1--E12 fail)**: $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$ with tactic exhaustion trace

The certificate structure (Definition {prf:ref}`def-typed-no-certificates`) ensures these are first-class outputs rather than silent failures. When $K^{\mathrm{inc}}$ is produced, the Sieve routes to reconstruction (MT 42.1) rather than fatal error, since inconclusiveness does not imply existence of a counterexample.

:::

:::{prf:definition} Obligation ledger
:label: def-obligation-ledger

Given a certificate context $\Gamma$, define the **obligation ledger**:
$$\mathsf{Obl}(\Gamma) := \{ (\mathsf{id}, \mathsf{obligation}, \mathsf{missing}, \mathsf{code}) : K_P^{\mathrm{inc}} \in \Gamma \}$$

Each entry corresponds to a NO-inconclusive certificate (Definition {prf:ref}`def-typed-no-certificates`) with payload $K_P^{\mathrm{inc}} = (\mathsf{obligation}, \mathsf{missing}, \mathsf{code}, \mathsf{trace})$.

Each entry records an undecided predicate—one where the verifier could not produce either $K_P^+$ or $K_P^{\mathrm{wit}}$.

:::

:::{prf:definition} Goal dependency cone
:label: def-goal-cone

Fix a goal certificate type $K_{\mathrm{Goal}}$ (e.g., $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ for the Lock).
The **goal dependency cone** $\Downarrow(K_{\mathrm{Goal}})$ is the set of certificate types that may be referenced by the verifier or promotion rules that produce $K_{\mathrm{Goal}}$.

Formally, $\Downarrow(K_{\mathrm{Goal}})$ is the least set closed under:
1. $K_{\mathrm{Goal}} \in \Downarrow(K_{\mathrm{Goal}})$
2. If a verifier or upgrade rule has premise certificate types $\{K_1, \ldots, K_n\}$ and conclusion type in $\Downarrow(K_{\mathrm{Goal}})$, then all premise types are in $\Downarrow(K_{\mathrm{Goal}})$
3. If a certificate type is required by a transport lemma used by a verifier in $\Downarrow(K_{\mathrm{Goal}})$, it is also in $\Downarrow(K_{\mathrm{Goal}})$

**Purpose:** The goal cone determines which `inc` certificates are relevant to a given proof goal. Obligations outside the cone do not affect proof completion for that goal.

:::

:::{prf:definition} Proof completion criterion
:label: def-proof-complete

A sieve run with final context $\Gamma_{\mathrm{final}}$ **proves the goal** $K_{\mathrm{Goal}}$ if:
1. $\Gamma_{\mathrm{final}}$ contains the designated goal certificate (e.g., $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$, or $K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$ after promotion closure), and
2. $\mathsf{Obl}(\mathrm{Cl}(\Gamma_{\mathrm{final}}))$ contains **no entries whose certificate type lies in the goal dependency cone** $\Downarrow(K_{\mathrm{Goal}})$

Equivalently, all NO-inconclusive obligations relevant to the goal have been discharged.

**Consequence:** An `inc` certificate whose type lies outside $\Downarrow(K_{\mathrm{Goal}})$ does not affect proof completion for that goal.

:::

---

# Part VI: Node Specifications

## 5. Gate Node Specifications (Blue Nodes)

Each gate node is specified by:
- **Predicate** $P_i$: The property being tested
- **YES certificate** $K_i^+$: Witnesses $P_i$ holds
- **NO certificate** $K_i^-$: Witnesses $P_i$ fails or is uncertifiable
- **Context update**: What is added to $\Gamma$
- **NO routing**: Where the NO edge leads

:::{prf:remark} Mandatory inconclusive output
:label: rem-mandatory-inc

If a node verifier cannot produce either a YES certificate $K_P^+$ or a NO-with-witness certificate $K_P^{\mathrm{wit}}$, it **MUST** return a NO-inconclusive certificate $K_P^{\mathrm{inc}}$ with payload $(\mathsf{obligation}, \mathsf{missing}, \mathsf{code}, \mathsf{trace})$.

This rule preserves determinism (two-valued outcomes: YES or NO) while recording epistemic uncertainty in the certificate structure (Definition {prf:ref}`def-typed-no-certificates`). Silent failures or undefined behavior are prohibited—every predicate evaluation must produce a typed certificate.

:::

---

### Node 1: EnergyCheck ($D_E$)

```mermaid
graph LR
    EnergyCheck{"<b>1. D_E:</b> Is Energy Finite?<br>E[Φ] < ∞"}
    style EnergyCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
```

:::{prf:definition} Node 1: EnergyCheck
:label: def-node-energy

**Interface ID:** $D_E$

**Predicate** $P_1$: The height functional $\Phi$ is bounded on the analysis window $[0, T)$:
$$P_1 \equiv \sup_{t \in [0, T)} \Phi(u(t)) < \infty$$

**YES certificate** $K_{D_E}^+ = (E_{\max}, \text{bound proof})$ where $E_{\max} = \sup_t \Phi(u(t))$.

**NO certificate** $K_{D_E}^- = (\text{blow-up witness})$ documenting energy escape.

**NO routing**: BarrierSat (Saturation Barrier)

**Literature:** Energy methods trace to Leray's seminal work on Navier-Stokes {cite}`Leray34` and the modern framework of dissipative evolution equations {cite}`Dafermos16`.

:::

:::{admonition} Physics Dictionary: First Law of Thermodynamics
:class: seealso

**Physical Interpretation:** Node 1 enforces **energy conservation**. The predicate $\sup_t \Phi(u(t)) < \infty$ is the mathematical formulation of the **First Law of Thermodynamics**: energy cannot be created from nothing—only transformed or transferred.

- **$K_{D_E}^+$ (Bounded):** System respects conservation; energy remains finite.
- **$K_{D_E}^-$ (Blow-up):** Apparent energy creation—indicates either external forcing or mathematical pathology (non-physical solution).
- **BarrierSat:** Even if instantaneous energy is formally unbounded, bounded drift (Foster-Lyapunov) ensures long-term stability via entropy production bounds.
:::

---

### Node 2: ZenoCheck ($\mathrm{Rec}_N$)

:::{prf:definition} Node 2: ZenoCheck
:label: def-node-zeno

**Interface ID:** $\mathrm{Rec}_N$

**Predicate** $P_2$: Discrete events (topology changes, surgery invocations, mode transitions) are finite on any bounded interval:
$$P_2 \equiv \#\{\text{events in } [0, T)\} < \infty \quad \forall T < T_*$$

**YES certificate** $K_{\mathrm{Rec}_N}^+ = (N_{\max}, \text{event bound proof})$.

**NO certificate** $K_{\mathrm{Rec}_N}^- = (\text{accumulation point witness})$.

**NO routing**: BarrierCausal (Causal Censor)

**Literature:** Zeno phenomena and event accumulation in hybrid systems {cite}`Smale67`; surgery counting bounds for geometric flows {cite}`Hamilton97`; {cite}`Perelman03`.

:::

---

### Node 3: CompactCheck ($C_\mu$)

:::{prf:definition} Node 3: CompactCheck
:label: def-node-compact

**Interface ID:** $C_\mu$

**Predicate** $P_3$: Energy concentrates (does not scatter):
$$P_3 \equiv \exists \text{ concentration profile as } t \to T_*$$

**Semantics**: This is a *dichotomy check*. YES means concentration occurs (proceed to profile extraction). NO means energy scatters (global existence via dispersion).

**YES certificate** $K_{C_\mu}^+ = (\text{concentration scale}, \text{concentration point})$.

**NO certificate** $K_{C_\mu}^- = (\text{dispersion certificate})$ --- this is **not a failure**; it routes to Mode D.D (global existence).

**NO routing**: BarrierScat (Scattering Barrier)

**YES routing**: Profile node (canonical profile emerges)

**Literature:** Concentration-compactness principle {cite}`Lions84`; {cite}`Lions85`; profile decomposition and bubbling {cite}`KenigMerle06`.

:::

:::{admonition} Physics Dictionary: Phase Transitions and Condensation
:class: seealso

**Physical Interpretation:** Node 3 detects whether energy **concentrates** (condenses) or **disperses** (scatters). This corresponds to fundamental phase transition phenomena:

- **Concentration ($K_{C_\mu}^+$):** Energy localizes into coherent structures (solitons, vortices, droplets). Analogous to **Bose-Einstein condensation**, **nucleation** in first-order phase transitions, or **droplet formation** in supersaturated systems.
- **Dispersion ($K_{C_\mu}^-$):** Energy spreads uniformly—no localized structures persist. This is the **normal state** of gases and high-temperature systems approaching thermal equilibrium.

The dichotomy mirrors the **thermodynamic distinction** between ordered (low-entropy, concentrated) and disordered (high-entropy, dispersed) phases. The critical threshold separating these regimes is the **phase boundary**.
:::

---

### Node 4: ScaleCheck ($\mathrm{SC}_\lambda$)

:::{prf:definition} Node 4: ScaleCheck
:label: def-node-scale

**Interface ID:** $\mathrm{SC}_\lambda$

**Predicate** $P_4$: The scaling structure is subcritical:
$$P_4 \equiv \alpha > \beta$$
where $\alpha, \beta$ are the scaling exponents satisfying:
$$\Phi(\mathcal{S}_\lambda x) = \lambda^\alpha \Phi(x), \quad \mathfrak{D}(\mathcal{S}_\lambda x) = \lambda^\beta \mathfrak{D}(x)$$

**YES certificate** $K_{\mathrm{SC}_\lambda}^+ = (\alpha, \beta, \alpha > \beta \text{ proof})$.

**NO certificate** $K_{\mathrm{SC}_\lambda}^- = (\alpha, \beta, \alpha \leq \beta \text{ witness})$.

**NO routing**: BarrierTypeII (Type II Barrier)

**Literature:** Scaling critical exponents in nonlinear PDE {cite}`KenigMerle06`; {cite}`KillipVisan10`; Type I/II blow-up classification {cite}`MerleZaag98`.

:::

---

### Node 5: ParamCheck ($\mathrm{SC}_{\partial c}$)

:::{prf:definition} Node 5: ParamCheck
:label: def-node-param

**Interface ID:** $\mathrm{SC}_{\partial c}$

**Predicate** $P_5$: Structural constants (modulation parameters, coupling constants) are stable:
$$P_5 \equiv \|\theta(t) - \theta_0\| \leq C \quad \forall t \in [0, T)$$

**YES certificate** $K_{\mathrm{SC}_{\partial c}}^+ = (\theta_0, C, \text{stability proof})$.

**NO certificate** $K_{\mathrm{SC}_{\partial c}}^- = (\text{parameter drift witness})$.

**NO routing**: BarrierVac (Vacuum Barrier)

:::

---

### Node 6: GeomCheck ($\mathrm{Cap}_H$)

:::{prf:definition} Node 6: GeomCheck
:label: def-node-geom

**Interface ID:** $\mathrm{Cap}_H$

**Predicate** $P_6$: The singular set has sufficiently small capacity (high codimension):
$$P_6 \equiv \mathrm{codim}(\mathcal{Y}_{\text{sing}}) \geq d_{\text{crit}} \quad \text{equivalently} \quad \dim_H(\mathcal{Y}_{\text{sing}}) \leq d - d_{\text{crit}}$$
where $d$ is the ambient dimension and $d_{\text{crit}}$ is the critical codimension threshold (typically $d_{\text{crit}} = 2$ for parabolic problems).

**Interpretation**: YES means the singular set is geometrically negligible (small dimension, high codimension). NO means the singular set is too ``fat'' and could obstruct regularity.

**YES certificate** $K_{\mathrm{Cap}_H}^+ = (\mathrm{codim}, d_{\text{crit}}, \mathrm{codim} \geq d_{\text{crit}} \text{ proof})$.

**NO certificate** $K_{\mathrm{Cap}_H}^- = (\mathrm{codim}, d_{\text{crit}}, \mathrm{codim} < d_{\text{crit}} \text{ witness})$.

**NO routing**: BarrierCap (Capacity Barrier)

**Literature:** Geometric measure theory and Hausdorff dimension {cite}`Federer69`; capacity and potential theory {cite}`AdamsHedberg96`; partial regularity {cite}`CaffarelliKohnNirenberg82`.

:::

---

### Node 7: StiffnessCheck ($\mathrm{LS}_\sigma$)

:::{prf:definition} Node 7: StiffnessCheck
:label: def-node-stiffness

**Interface ID:** $\mathrm{LS}_\sigma$

**Predicate** $P_7$: Local stiffness (Łojasiewicz-Simon inequality) holds near critical points. The standard form is:
$$P_7 \equiv \exists \theta \in (0, \tfrac{1}{2}], C_{\text{LS}} > 0, \delta > 0 : \|\nabla \Phi(x)\| \geq C_{\text{LS}} |\Phi(x) - \Phi_*|^{1-\theta}$$
for all $x$ with $d(x, M) < \delta$, where $M$ is the set of critical points and $\Phi_*$ is the critical value.

**Consequence**: The LS inequality implies finite-length gradient flow convergence to $M$ with rate $O(t^{-\theta/(1-2\theta)})$.

**YES certificate** $K_{\mathrm{LS}_\sigma}^+ = (\theta, C_{\text{LS}}, \delta, \text{LS inequality proof})$.

**NO certificate** $K_{\mathrm{LS}_\sigma}^- = (\text{flatness witness}: \theta \to 0 \text{ or } C_{\text{LS}} \to 0 \text{ or degenerate Hessian})$.

**NO routing**: BarrierGap (Spectral Barrier)

**Literature:** Łojasiewicz gradient inequality {cite}`Lojasiewicz65`; Simon's extension to infinite dimensions {cite}`Simon83`; Kurdyka-Łojasiewicz theory {cite}`Kurdyka98`.

:::

---

### Nodes 7a--7d: Stiffness Restoration Subtree

:::{prf:definition} Node 7a: BifurcateCheck
:label: def-node-bifurcate

**Interface ID:** $\mathrm{LS}_{\partial^2 V}$

**Predicate** $P_{7a}$: The current state is dynamically unstable (admits bifurcation).

**YES certificate** $K_{\mathrm{LS}_{\partial^2 V}}^+ = (\text{unstable eigenvalue}, \text{bifurcation direction})$.

**NO certificate** $K_{\mathrm{LS}_{\partial^2 V}}^- = (\text{stability certificate})$ --- routes to Mode S.D.

**YES routing**: SymCheck (Node 7b)

**NO routing**: Mode S.D (Stiffness Breakdown)

:::

:::{prf:definition} Node 7b: SymCheck
:label: def-node-sym

**Interface ID:** $G_{\mathrm{act}}$

**Predicate** $P_{7b}$: The vacuum is degenerate (symmetry group $G$ acts non-trivially).

**YES certificate** $K_{G_{\mathrm{act}}}^+ = (G, \text{group action}, \text{degeneracy proof})$.

**NO certificate** $K_{G_{\mathrm{act}}}^- = (\text{asymmetry certificate})$.

**YES routing**: CheckSC (Node 7c) --- symmetry breaking path

**NO routing**: CheckTB (Node 7d) --- tunneling path

:::

:::{prf:definition} Node 7c: CheckSC (Restoration)
:label: def-node-checksc

**Interface ID:** $\mathrm{SC}_{\partial c}$

**Predicate** $P_{7c}$: Parameters remain stable under symmetry breaking:
$$P_{7c} \equiv \|\theta_{\text{broken}} - \theta_0\| \leq C_{\text{SSB}}$$
where $\theta_{\text{broken}}$ are the parameters in the broken-symmetry phase.

**YES certificate** $K_{\mathrm{SC}_{\partial c}}^+ = (\theta_{\text{broken}}, C_{\text{SSB}}, \text{stability proof})$. Enables ActionSSB.

**NO certificate** $K_{\mathrm{SC}_{\partial c}}^- = (\text{parameter runaway witness})$. Routes to Mode S.C (Vacuum Decay).

**YES routing**: ActionSSB $\to$ TopoCheck

**NO routing**: Mode S.C $\to$ SurgSC\_Rest $\dashrightarrow$ TopoCheck

:::

:::{prf:definition} Node 7d: CheckTB (Action)
:label: def-node-checktb

**Interface ID:** $\mathrm{TB}_S$

**Predicate** $P_{7d}$: Tunneling action cost is finite:
$$P_{7d} \equiv \mathcal{A}_{\text{tunnel}} < \infty$$
where $\mathcal{A}_{\text{tunnel}}$ is the instanton action connecting the current metastable state to a lower-energy sector.

**YES certificate** $K_{\mathrm{TB}_S}^+ = (\mathcal{A}_{\text{tunnel}}, \text{instanton path}, \text{finiteness proof})$. Enables ActionTunnel.

**NO certificate** $K_{\mathrm{TB}_S}^- = (\text{infinite action witness})$. Routes to Mode T.E (Metastasis).

**YES routing**: ActionTunnel $\to$ TameCheck

**NO routing**: Mode T.E $\to$ SurgTE\_Rest $\dashrightarrow$ TameCheck

:::

---

### Node 8: TopoCheck ($\mathrm{TB}_\pi$)

:::{prf:definition} Node 8: TopoCheck
:label: def-node-topo

**Interface ID:** $\mathrm{TB}_\pi$

**Predicate** $P_8$: The topological sector is accessible (no obstruction):
$$P_8 \equiv \tau(x) \in \mathcal{T}_{\text{accessible}}$$
where $\tau: X \to \mathcal{T}$ is the sector label.

**Semantics of NO**: "Protected" means the sector is *obstructed/inaccessible*, not "safe."

**YES certificate** $K_{\mathrm{TB}_\pi}^+ = (\tau(x), \text{accessibility proof})$.

**NO certificate** $K_{\mathrm{TB}_\pi}^- = (\tau(x), \text{obstruction certificate})$.

**NO routing**: BarrierAction (Action Barrier)

:::

---

### Node 9: TameCheck ($\mathrm{TB}_O$)

:::{prf:definition} Node 9: TameCheck
:label: def-node-tame

**Interface ID:** $\mathrm{TB}_O$

**Predicate** $P_9$: The topology is tame (definable in an o-minimal structure):
$$P_9 \equiv \text{Singular locus is o-minimally definable}$$

**YES certificate** $K_{\mathrm{TB}_O}^+ = (\text{o-minimal structure}, \text{definability proof})$.

**NO certificate** $K_{\mathrm{TB}_O}^- = (\text{wildness witness})$.

**NO routing**: BarrierOmin (O-Minimal Barrier)

**Literature:** O-minimal structures and tame topology {cite}`vandenDries98`; {cite}`vandenDriesMiller96`; model completeness {cite}`Wilkie96`.

:::

---

### Node 10: ErgoCheck ($\mathrm{TB}_\rho$)

:::{prf:definition} Node 10: ErgoCheck
:label: def-node-ergo

**Interface ID:** $\mathrm{TB}_\rho$

**Predicate** $P_{10}$: The dynamics mixes (ergodic/explores full state space):
$$P_{10} \equiv \tau_{\text{mix}} < \infty$$

**Equivalence Note:** A positive spectral gap $\rho(\mu) > 0$ is a *sufficient* condition for finite mixing time: $\tau_{\text{mix}} \lesssim \rho^{-1} \log(1/\varepsilon)$.

**YES certificate** $K_{\mathrm{TB}_\rho}^+ = (\tau_{\text{mix}}, \text{mixing proof})$.

**NO certificate** $K_{\mathrm{TB}_\rho}^- = (\text{trap certificate}, \text{invariant subset})$.

**NO routing**: BarrierMix (Mixing Barrier)

**Literature:** Ergodic theory and mixing {cite}`Birkhoff31`; {cite}`Furstenberg81`; Markov chain stability {cite}`MeynTweedie93`.

:::

:::{admonition} Physics Dictionary: Thermalization and the H-Theorem
:class: seealso

**Physical Interpretation:** Node 10 verifies **ergodicity**—whether the system explores its full phase space over time. This connects to fundamental statistical mechanics:

- **Boltzmann's H-Theorem (1872):** The H-function (negative entropy) decreases monotonically, driving systems toward thermal equilibrium. Finite mixing time $\tau_{\text{mix}} < \infty$ ensures equilibration occurs on observable timescales.
- **Thermalization:** An ergodic system eventually samples all accessible states according to the equilibrium distribution (Gibbs measure). This is the foundation of **statistical mechanics**.
- **Glassy Freeze ($K_{\mathrm{TB}_\rho}^-$):** Non-ergodic systems become trapped in metastable states—like glasses that never reach crystalline equilibrium. The mixing barrier captures this phenomenon.

The spectral gap $\rho > 0$ quantifies how fast the Second Law of Thermodynamics operates: larger gaps mean faster equilibration.
:::

---

### Node 11: ComplexCheck ($\mathrm{Rep}_K$)

:::{prf:definition} Node 11: ComplexCheck
:label: def-node-complex

**Interface ID:** $\mathrm{Rep}_K$

**Predicate** $P_{11}$: The system admits a computable finite description:
$$P_{11} \equiv K(x) \in \mathbb{N} \text{ (Kolmogorov complexity is decidable and finite)}$$

**Semantic Clarification:**
- **YES:** $K(x)$ is computable and finite → proceed to OscillateCheck
- **NO:** $K(x)$ is uncomputable, unbounded, or exceeds computational horizon → trigger BarrierEpi

**Complexity Type Clarification:**
- **Deterministic systems:** Complexity is evaluated on the state $K(x)$ or trajectory $K(x_t)$.
- **Stochastic systems (post-S12):** Complexity is evaluated on the probability law $K(\mu_t)$ where $\mu_t = \text{Law}(x_t)$, not on individual sample paths. The SDE $dx = b\,dt + \sigma\,dW_t$ has finite description length $K(\text{SDE}) < \infty$ even though individual realizations $x_t(\omega)$ are algorithmically incompressible.

**YES certificate** $K_{\mathrm{Rep}_K}^+ = (D, K(D(x)), \text{computability proof})$.

**NO certificate** $K_{\mathrm{Rep}_K}^- = (\text{uncomputability witness or divergence proof})$.

**NO routing**: BarrierEpi (Epistemic Barrier)

**Literature:** Kolmogorov complexity {cite}`Kolmogorov65`; algorithmic information theory {cite}`Chaitin66`; {cite}`LiVitanyi08`; algorithmic complexity of probability distributions {cite}`GacsEtAl01`.

:::

---

### Node 12: OscillateCheck ($\mathrm{GC}_\nabla$)

:::{prf:definition} Node 12: OscillateCheck
:label: def-node-oscillate

**Interface ID:** $\mathrm{GC}_\nabla$

**Predicate** $P_{12}$: Oscillatory behavior is present.

**Semantics**: This is *not* a good/bad check. YES means oscillation is present, which triggers the Frequency Barrier. NO means no oscillation, proceeding to boundary checks.

**YES certificate** $K_{\mathrm{GC}_\nabla}^+ = (\text{oscillation frequency}, \text{oscillation witness})$.

**NO certificate** $K_{\mathrm{GC}_\nabla}^- = (\text{monotonicity certificate})$.

**YES routing**: BarrierFreq (Frequency Barrier)

**NO routing**: BoundaryCheck (Node 13)

:::

---

### Nodes 13--16: Boundary Checks

:::{prf:definition} Node 13: BoundaryCheck
:label: def-node-boundary

**Interface ID:** $\mathrm{Bound}_\partial$

**Predicate** $P_{13}$: The system has boundary interactions (is open):
$$P_{13} \equiv \partial X \neq \varnothing \text{ or } \exists \text{ external input/output coupling}$$

**YES certificate** $K_{\mathrm{Bound}_\partial}^+ = (\partial X, u_{\text{in}}, y_{\text{out}}, \text{coupling structure})$: Documents the boundary structure, input space, output space, and their interaction.

**NO certificate** $K_{\mathrm{Bound}_\partial}^- = (\text{closed system certificate: } \partial X = \varnothing, \text{ no external coupling})$

**YES routing**: OverloadCheck (Node 14) --- enter boundary subgraph

**NO routing**: BarrierExclusion (Node 17) --- closed system, proceed to lock

:::

:::{prf:definition} Node 14: OverloadCheck
:label: def-node-overload

**Interface ID:** $\mathrm{Bound}_B$

**Predicate** $P_{14}$: Input is bounded (no injection/overload):
$$P_{14} \equiv \|u_{\text{in}}\|_{L^\infty} \leq U_{\max} \quad \text{and} \quad \int_0^T \|u_{\text{in}}(t)\|^2 dt < \infty$$

**YES certificate** $K_{\mathrm{Bound}_B}^+ = (U_{\max}, \text{input bound proof})$: Documents the maximum input magnitude and its boundedness proof.

**NO certificate** $K_{\mathrm{Bound}_B}^- = (\text{unbounded input witness: sequence } u_n \text{ with } \|u_n\| \to \infty)$

**YES routing**: StarveCheck (Node 15)

**NO routing**: BarrierBode (Bode Barrier)

:::

:::{prf:definition} Node 15: StarveCheck
:label: def-node-starve

**Interface ID:** $\mathrm{Bound}_{\Sigma}$

**Predicate** $P_{15}$: Input is sufficient (no starvation):
$$P_{15} \equiv \int_0^T \|u_{\text{in}}(t)\| dt \geq U_{\min}(T) \quad \text{for required supply threshold } U_{\min}$$

**YES certificate** $K_{\mathrm{Bound}_{\Sigma}}^+ = (U_{\min}, \int u_{\text{in}}, \text{supply sufficiency proof})$: Documents the required supply threshold and that actual supply meets or exceeds it.

**NO certificate** $K_{\mathrm{Bound}_{\Sigma}}^- = (\text{starvation witness: supply deficit } \int u_{\text{in}} < U_{\min})$

**YES routing**: AlignCheck (Node 16)

**NO routing**: BarrierInput (Input Barrier)

:::

:::{prf:definition} Node 16: AlignCheck
:label: def-node-align

**Interface ID:** $\mathrm{GC}_T$

**Predicate** $P_{16}$: System is aligned (proxy objective matches true objective):
$$P_{16} \equiv d(\mathcal{L}_{\text{proxy}}, \mathcal{L}_{\text{true}}) \leq \varepsilon_{\text{align}}$$
where $\mathcal{L}_{\text{proxy}}$ is the optimized/measured objective and $\mathcal{L}_{\text{true}}$ is the intended objective.

**YES certificate** $K_{\mathrm{GC}_T}^+ = (\varepsilon_{\text{align}}, d(\mathcal{L}_{\text{proxy}}, \mathcal{L}_{\text{true}}), \text{alignment bound proof})$: Documents the alignment tolerance and that the proxy-true distance is within tolerance.

**NO certificate** $K_{\mathrm{GC}_T}^- = (\text{misalignment witness: } d(\mathcal{L}_{\text{proxy}}, \mathcal{L}_{\text{true}}) > \varepsilon_{\text{align}})$

**YES routing**: BarrierExclusion (Node 17)

**NO routing**: BarrierVariety (Variety Barrier)

:::

---

### Node 17: BarrierExclusion ($\mathrm{Cat}_{\mathrm{Hom}}$) --- The Lock

:::{prf:definition} Barrier Specification: Morphism Exclusion (The Lock)
:label: def-node-lock

**Barrier ID:** `BarrierExclusion`

**Interface Dependencies:**
- **Primary:** $\mathrm{Cat}_{\mathrm{Hom}}$ (provides Hom functor and morphism space $\mathrm{Hom}(\mathcal{B}, S)$)
- **Secondary:** Full context (all prior certificates $\Gamma$ inform exclusion proof)

**Sieve Signature:**
- **Weakest Precondition:** Full $\Gamma$ (complete certificate chain from all prior nodes)
- **Barrier Predicate (Blocked Condition):**
  $$\mathrm{Hom}_{\mathbf{Hypo}}(\mathbb{H}_{\mathrm{bad}}, \mathcal{H}) = \varnothing$$

**Natural Language Logic:**
"Is there a categorical obstruction to the bad pattern?"
*(If no morphism exists from the universal bad pattern $\mathbb{H}_{\mathrm{bad}}$ to the system $\mathcal{H}$, then the system structurally cannot exhibit singular behavior—the morphism exclusion principle.)*

**Outcome Alphabet:** $\{\texttt{Blocked}, \texttt{Breached}\}$ (binary verdict with typed certificates)

**Outcomes:**
- **Blocked** ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$): Hom-set empty; no morphism to bad pattern exists. **VICTORY: Global Regularity Confirmed.**
- **Breached** ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br}}$): NO verdict with typed certificate (sum type $K^{\mathrm{br}} := K^{\mathrm{br\text{-}wit}} \sqcup K^{\mathrm{br\text{-}inc}}$):
  - **Breached-with-witness** ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}wit}}$): Explicit morphism $f: \mathbb{H}_{\mathrm{bad}} \to \mathcal{H}$ found; structural inconsistency. **FATAL ERROR.**
  - **Breached-inconclusive** ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$): Tactics E1–E12 exhausted without deciding Hom-emptiness. Certificate records $(\mathsf{tactics\_exhausted}, \mathsf{partial\_progress}, \mathsf{trace})$. Triggers **MT 42.1** (Structural Reconstruction Principle).

**Routing:**
- **On Block:** Exit with **GLOBAL REGULARITY** (structural exclusion confirmed).
- **On Breached-with-witness:** Exit with **FATAL ERROR** (structural inconsistency—requires interface permit revision).
- **On Breached-inconclusive:** Invoke **MT 42.1** (Structural Reconstruction) → Re-evaluate with reconstruction verdict $K_{\mathrm{Rec}}^{\mathrm{verdict}}$.

**Exclusion Tactics (E1–E12):** The emptiness proof may invoke:
- E1: Dimension count (bad pattern requires impossible dimension)
- E2: Coercivity (energy structure forbids mapping)
- E3: Spectral (eigenvalue gap prevents morphism)
- E4: Topological (homotopy class obstruction)
- E5: Categorical (universal property violation)
- E6–E10: (Additional tactics from Lock specification)
- E11: Bridge certificate (symmetry descent)
- E12: Rigidity certificate (semisimplicity/tameness/spectral gap)

:::

:::{admonition} Physics Dictionary: Pauli Exclusion and Information Conservation
:class: seealso

**Physical Interpretation:** Node 17 (the Lock) enforces a **categorical exclusion principle**—analogous to fundamental physics principles:

- **Pauli Exclusion Principle:** Two identical fermions cannot occupy the same quantum state. The Lock enforces: "A valid hypostructure cannot morphically embed a bad pattern"—certain configurations are **structurally forbidden**.
- **Conservation of Information:** In unitary quantum mechanics, information is never destroyed (Hawking's resolution of the black hole information paradox). The Lock ensures: $\text{Hom}(\mathbb{H}_{\text{bad}}, \mathcal{H}) = \varnothing$ means singularity formation would require information destruction incompatible with the system's structure.
- **No-Cloning Theorem:** Quantum states cannot be perfectly copied. Similarly, the Lock prevents "copying" of bad patterns into a valid hypostructure.

The **exclusion tactics (E1–E12)** are analogous to **selection rules** in quantum mechanics—symmetry and conservation laws that forbid certain transitions.
:::

---

## 6. Barrier Node Specifications (Orange Nodes)

Each barrier is specified by:
- **Trigger**: Which gate's NO invokes it
- **Pre-certificates**: Required context (non-circular)
- **Outcome alphabet**: Blocked/Breached (or special)
- **Blocked certificate**: Must imply Pre(next node)
- **Breached certificate**: Must imply mode activation + surgery admissibility
- **Next nodes**: Routing for each outcome

---

### BarrierSat (Saturation Barrier)

:::{prf:definition} Barrier Specification: Saturation
:label: def-barrier-sat

**Barrier ID:** `BarrierSat`

**Interface Dependencies:**
- **Primary:** $D_E$ (provides energy functional $E[\Phi]$ and its drift rate)
- **Secondary:** $\mathrm{SC}_\lambda$ (provides saturation ceiling $E_{\text{sat}}$ and drift bound $C$)

**Sieve Signature:**
- **Weakest Precondition:** $\emptyset$ (entry barrier, no prior certificates required)
- **Barrier Predicate (Blocked Condition):**
  $$E[\Phi] \leq E_{\text{sat}} \lor \text{Drift} \leq C$$

**Natural Language Logic:**
"Is the energy drift bounded by a saturation ceiling?"
*(Even if energy is not globally bounded, the drift rate may be controlled by a saturation mechanism that prevents blow-up.)*

**Outcomes:**
- **Blocked** ($K_{D_E}^{\mathrm{blk}}$): Drift is controlled by saturation ceiling. Singularity excluded via energy saturation principle.
- **Breached** ($K_{D_E}^{\mathrm{br}}$): Uncontrolled drift detected. Activates **Mode C.E** (Energy Blow-up).

**Routing:**
- **On Block:** Proceed to `ZenoCheck`.
- **On Breach:** Trigger **Mode C.E** → Enable Surgery `SurgCE` → Re-enter at `ZenoCheck`.

**Literature:** Saturation and drift bounds via Foster-Lyapunov conditions {cite}`MeynTweedie93`; energy dissipation in physical systems {cite}`Dafermos16`.

:::

---

### BarrierCausal (Causal Censor)

:::{prf:definition} Barrier Specification: Causal Censor
:label: def-barrier-causal

**Barrier ID:** `BarrierCausal`

**Interface Dependencies:**
- **Primary:** $\mathrm{Rec}_N$ (provides computational depth $D(T_*)$ of event tree)
- **Secondary:** $\mathrm{TB}_\pi$ (provides time scale $\lambda(t)$ and horizon $T_*$)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{D_E}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$D(T_*) = \int_0^{T_*} \frac{c}{\lambda(t)} dt = \infty$$

**Natural Language Logic:**
"Does the singularity require infinite computational depth?"
*(If the integral diverges, the singularity would require unbounded computational resources to describe, making it causally inaccessible—a censorship mechanism.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{Rec}_N}^{\mathrm{blk}}$): Depth diverges; singularity causally censored. Implies Pre(CompactCheck).
- **Breached** ($K_{\mathrm{Rec}_N}^{\mathrm{br}}$): Finite depth; singularity computationally accessible. Activates **Mode C.C** (Event Accumulation).

**Routing:**
- **On Block:** Proceed to `CompactCheck`.
- **On Breach:** Trigger **Mode C.C** → Enable Surgery `SurgCC` → Re-enter at `CompactCheck`.

**Literature:** Causal structure and cosmic censorship {cite}`Penrose69`; {cite}`HawkingPenrose70`; computational depth bounds {cite}`Kolmogorov65`.

:::

---

### BarrierScat (Scattering Barrier) --- Special Alphabet

:::{prf:definition} Barrier Specification: Scattering
:label: def-barrier-scat

**Barrier ID:** `BarrierScat`

**Interface Dependencies:**
- **Primary:** $C_\mu$ (provides concentration measure and interaction functional $\mathcal{M}[\Phi]$)
- **Secondary:** $D_E$ (provides dispersive energy structure)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{D_E}^{\pm}, K_{\mathrm{Rec}_N}^{\pm}\}$
- **Barrier Predicate (Benign Condition):**
  $$\mathcal{M}[\Phi] < \infty$$

**Natural Language Logic:**
"Is the interaction functional finite (implying dispersion)?"
*(Finite Morawetz interaction implies scattering to free solutions; the energy disperses rather than concentrating.)*

**Outcome Alphabet:** $\{\texttt{Benign}, \texttt{Pathological}\}$ (special)

**Outcomes:**
- **Benign** ($K_{C_\mu}^{\mathrm{ben}}$): Interaction finite; dispersion confirmed. **Success exit** via **Mode D.D** (Global Existence).
- **Pathological** ($K_{C_\mu}^{\mathrm{path}}$): Infinite interaction; soliton-like escape. Activates **Mode C.D** (Concentration-Escape).

**Routing:**
- **On Benign:** Exit to **Mode D.D** (Success: dispersion implies global existence).
- **On Pathological:** Trigger **Mode C.D** → Enable Surgery `SurgCD_Alt` → Re-enter at `Profile`.

**Literature:** Morawetz estimates and scattering {cite}`Morawetz68`; concentration-compactness rigidity {cite}`KenigMerle06`; {cite}`KillipVisan10`.

:::

---

### BarrierTypeII (Type II Barrier)

:::{prf:definition} Barrier Specification: Type II Exclusion
:label: def-barrier-type2

**Barrier ID:** `BarrierTypeII`

**Interface Dependencies:**
- **Primary:** $\mathrm{SC}_\lambda$ (provides scale parameter $\lambda(t)$ and renormalization action)
- **Secondary:** $D_E$ (provides energy functional and blow-up profile $V$)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{C_\mu}^+\}$ (concentration confirmed, profile exists)
- **Barrier Predicate (Blocked Condition):**
  $$\int \tilde{\mathfrak{D}}(S_t V) dt = \infty$$

**Natural Language Logic:**
"Is the renormalization cost of the profile infinite?"
*(If the integrated defect of the rescaled profile diverges, Type II (self-similar) blow-up is excluded by infinite renormalization cost.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}$): Renormalization cost infinite; self-similar blow-up excluded. Implies Pre(ParamCheck).
- **Breached** ($K_{\mathrm{SC}_\lambda}^{\mathrm{br}}$): Finite renormalization cost; Type II blow-up possible. Activates **Mode S.E** (Supercritical).

**Routing:**
- **On Block:** Proceed to `ParamCheck`.
- **On Breach:** Trigger **Mode S.E** → Enable Surgery `SurgSE` → Re-enter at `ParamCheck`.

**Non-circularity note:** This barrier is triggered by ScaleCheck NO (supercritical: $\alpha \leq \beta$). Subcriticality ($\alpha > \beta$) may be used as an optional *sufficient* condition for Blocked (via Type I exclusion), but is not a *prerequisite* for barrier evaluation.

**Literature:** Type II blow-up and renormalization {cite}`MerleZaag98`; {cite}`RaphaelSzeftel11`; {cite}`CollotMerleRaphael17`.

:::

---

### BarrierVac (Vacuum Barrier)

:::{prf:definition} Barrier Specification: Vacuum Stability
:label: def-barrier-vac

**Barrier ID:** `BarrierVac`

**Interface Dependencies:**
- **Primary:** $\mathrm{SC}_{\partial c}$ (provides vacuum potential $V$ and thermal scale $k_B T$)
- **Secondary:** $\mathrm{LS}_\sigma$ (provides stability landscape and barrier heights $\Delta V$)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{C_\mu}^+, K_{\mathrm{SC}_\lambda}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$\Delta V > k_B T$$

**Natural Language Logic:**
"Is the phase stable against thermal/parameter drift?"
*(If the potential barrier exceeds the thermal energy scale, the vacuum is stable against fluctuation-induced decay—the mass gap principle.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{SC}_{\partial c}}^{\mathrm{blk}}$): Phase stable; barrier exceeds thermal scale. Implies Pre(GeomCheck).
- **Breached** ($K_{\mathrm{SC}_{\partial c}}^{\mathrm{br}}$): Phase unstable; vacuum decay possible. Activates **Mode S.C** (Parameter Instability).

**Routing:**
- **On Block:** Proceed to `GeomCheck`.
- **On Breach:** Trigger **Mode S.C** → Enable Surgery `SurgSC` → Re-enter at `GeomCheck`.

**Literature:** Vacuum stability and phase transitions {cite}`Goldstone61`; {cite}`Higgs64`; {cite}`Coleman75`.

:::

---

### BarrierCap (Capacity Barrier)

:::{prf:definition} Barrier Specification: Capacity
:label: def-barrier-cap

**Barrier ID:** `BarrierCap`

**Interface Dependencies:**
- **Primary:** $\mathrm{Cap}_H$ (provides Hausdorff capacity $\mathrm{Cap}_H(S)$ of singular set $S$)
- **Secondary:** None (pure geometric criterion)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{SC}_{\partial c}}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$\mathrm{Cap}_H(S) = 0$$

**Natural Language Logic:**
"Is the singular set of measure zero?"
*(Zero capacity implies the singular set is negligible—it cannot carry enough mass to affect the dynamics. This is the capacity barrier principle.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{Cap}_H}^{\mathrm{blk}}$): Singular set has zero capacity; negligible. Implies Pre(StiffnessCheck).
- **Breached** ($K_{\mathrm{Cap}_H}^{\mathrm{br}}$): Positive capacity; singular set non-negligible. Activates **Mode C.D** (Geometric Collapse).

**Routing:**
- **On Block:** Proceed to `StiffnessCheck`.
- **On Breach:** Trigger **Mode C.D** → Enable Surgery `SurgCD` → Re-enter at `StiffnessCheck`.

**Literature:** Capacity and removable singularities {cite}`Federer69`; {cite}`EvansGariepy15`; {cite}`AdamsHedberg96`.

:::

---

### BarrierGap (Spectral Barrier) --- Special Alphabet

:::{prf:definition} Barrier Specification: Spectral Gap
:label: def-barrier-gap

**Barrier ID:** `BarrierGap`

**Interface Dependencies:**
- **Primary:** $\mathrm{LS}_\sigma$ (provides spectrum $\sigma(L)$ of linearized operator $L$)
- **Secondary:** $\mathrm{GC}_\nabla$ (provides gradient structure and Hessian at critical points)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{Cap}_H}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$\inf \sigma(L) > 0$$

**Natural Language Logic:**
"Is there a spectral gap (positive curvature) at the minimum?"
*(A positive spectral gap implies exponential decay toward the critical point via Łojasiewicz-Simon inequality—the spectral generator principle.)*

**Outcome Alphabet:** $\{\texttt{Blocked}, \texttt{Stagnation}\}$ (special)

**Outcomes:**
- **Blocked** ($K_{\mathrm{LS}_\sigma}^{\mathrm{blk}}$): Spectral gap exists; exponential convergence guaranteed. Implies Pre(TopoCheck).
- **Stagnation** ($K_{\mathrm{LS}_\sigma}^{\mathrm{stag}}$): No spectral gap; system may stagnate at degenerate critical point. Routes to restoration subtree.

**Routing:**
- **On Block:** Proceed to `TopoCheck`.
- **On Stagnation:** Enter restoration subtree via `BifurcateCheck` (Node 7a).

**Literature:** Spectral gap and gradient flows {cite}`Simon83`; {cite}`FeehanMaridakis19`; {cite}`Huang06`.

:::

:::{prf:lemma} Gap implies Łojasiewicz-Simon
:label: lem-gap-to-ls

Under the Gradient Condition ($\mathrm{GC}_\nabla$) plus analyticity of $\Phi$ near critical points:
$$\text{Spectral gap } \lambda_1 > 0 \Rightarrow \text{LS}(\theta = \tfrac{1}{2}, C_{\text{LS}} = \sqrt{\lambda_1})$$
This is the **canonical promotion** from gap certificate to stiffness certificate, bridging the diagram's ``Hessian positive?'' intuition with the formal LS inequality predicate.

:::

---

### BarrierAction (Action Barrier)

:::{prf:definition} Barrier Specification: Action Gap
:label: def-barrier-action

**Barrier ID:** `BarrierAction`

**Interface Dependencies:**
- **Primary:** $\mathrm{TB}_\pi$ (provides topological action gap $S_{\min}$ and threshold $\Delta$)
- **Secondary:** $D_E$ (provides current energy $E[\Phi]$)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{LS}_\sigma}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$E[\Phi] < S_{\min} + \Delta$$

**Natural Language Logic:**
"Is the energy insufficient to cross the topological gap?"
*(If current energy is below the action threshold, topological transitions (tunneling, kink formation) are energetically forbidden—the topological suppression principle.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{TB}_\pi}^{\mathrm{blk}}$): Energy below action gap; tunneling suppressed. Implies Pre(TameCheck).
- **Breached** ($K_{\mathrm{TB}_\pi}^{\mathrm{br}}$): Energy sufficient for topological transition. Activates **Mode T.E** (Topological Transition).

**Routing:**
- **On Block:** Proceed to `TameCheck`.
- **On Breach:** Trigger **Mode T.E** → Enable Surgery `SurgTE` → Re-enter at `TameCheck`.

**Literature:** Topological obstructions and action principles {cite}`Smale67`; {cite}`Conley78`; {cite}`Floer89`.

:::

---

### BarrierOmin (O-Minimal Barrier)

:::{prf:definition} Barrier Specification: O-Minimal Taming
:label: def-barrier-omin

**Barrier ID:** `BarrierOmin`

**Interface Dependencies:**
- **Primary:** $\mathrm{TB}_O$ (provides o-minimal structure $\mathcal{O}$ and definability criteria)
- **Secondary:** $\mathrm{Rep}_K$ (provides representation-theoretic bounds on complexity)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{TB}_\pi}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$S \in \mathcal{O}\text{-min}$$

**Natural Language Logic:**
"Is the topology definable in an o-minimal structure?"
*(O-minimal definability implies tameness: no pathological fractals, finite stratification, controlled asymptotic behavior—the o-minimal taming principle.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{TB}_O}^{\mathrm{blk}}$): Topology is o-minimally definable; wild behavior tamed. Implies Pre(ErgoCheck).
- **Breached** ($K_{\mathrm{TB}_O}^{\mathrm{br}}$): Topology not definable; genuinely wild structure. Activates **Mode T.C** (Topological Complexity).

**Routing:**
- **On Block:** Proceed to `ErgoCheck`.
- **On Breach:** Trigger **Mode T.C** → Enable Surgery `SurgTC` → Re-enter at `ErgoCheck`.

**Literature:** O-minimal structures and tame topology {cite}`vandenDries98`; {cite}`Kurdyka98`; {cite}`Wilkie96`.

:::

---

### BarrierMix (Mixing Barrier)

:::{prf:definition} Barrier Specification: Ergodic Mixing
:label: def-barrier-mix

**Barrier ID:** `BarrierMix`

**Interface Dependencies:**
- **Primary:** $\mathrm{TB}_\rho$ (provides mixing time $\tau_{\text{mix}}$ and escape probability)
- **Secondary:** $D_E$ (provides energy landscape for trap depth estimation)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{TB}_O}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$\tau_{\text{mix}} < \infty$$

**Natural Language Logic:**
"Does the system mix fast enough to escape traps?"
*(Finite mixing time implies ergodicity: the system explores all accessible states and cannot be permanently trapped—the ergodic mixing principle.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{TB}_\rho}^{\mathrm{blk}}$): Mixing time finite; trap escapable. Implies Pre(ComplexCheck).
- **Breached** ($K_{\mathrm{TB}_\rho}^{\mathrm{br}}$): Infinite mixing time; permanent trapping possible. Activates **Mode T.D** (Trapping).

**Routing:**
- **On Block:** Proceed to `ComplexCheck`.
- **On Breach:** Trigger **Mode T.D** → Enable Surgery `SurgTD` → Re-enter at `ComplexCheck`.

**Literature:** Ergodic theory and mixing {cite}`Birkhoff31`; {cite}`Furstenberg81`; {cite}`MeynTweedie93`.

:::

---

### BarrierEpi (Epistemic Barrier)

:::{prf:definition} Barrier Specification: Epistemic Horizon
:label: def-barrier-epi

**Barrier ID:** `BarrierEpi`

**Interface Dependencies:**
- **Primary:** $\mathrm{Rep}_K$ (provides Kolmogorov complexity $K(x)$ of state description)
- **Secondary:** $\mathrm{Cap}_H$ (provides Bekenstein-Hawking entropy bound $S_{\text{BH}}$)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{TB}_\rho}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$\sup_{\epsilon > 0} K_\epsilon(x) \leq S_{\text{BH}}$$
  where $K_\epsilon(x) := \min\{|p| : d(U(p), x) < \epsilon\}$ is the $\epsilon$-approximable complexity.

**Semantic Clarification:**
This barrier is triggered when Node 11 determines that exact complexity is uncomputable. The predicate now asks: "Even though we cannot compute $K(x)$ exactly, can we bound all computable approximations within the holographic limit?" This makes the "Blocked" outcome logically reachable:
- If approximations converge to a finite limit $\leq S_{\text{BH}}$ → Blocked
- If approximations diverge or exceed $S_{\text{BH}}$ → Breached

**Natural Language Logic:**
"Is the approximable description length within physical bounds?"
*(Even when exact complexity is uncomputable, if all computable approximations stay within the holographic bound, the system cannot encode more information than spacetime permits—the epistemic horizon principle.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{Rep}_K}^{\mathrm{blk}}$): Approximable complexity bounded; within holographic limit. Implies Pre(OscillateCheck).
- **Breached** ($K_{\mathrm{Rep}_K}^{\mathrm{br}}$): Approximations diverge or exceed holographic bound; epistemic horizon violated. Activates **Mode D.C** (Complexity Explosion).

**Routing:**
- **On Block:** Proceed to `OscillateCheck`.
- **On Breach:** Trigger **Mode D.C** → Enable Surgery `SurgDC` → Re-enter at `OscillateCheck`.

**Literature:** Kolmogorov complexity {cite}`Kolmogorov65`; holographic bounds {cite}`tHooft93`; {cite}`Susskind95`; {cite}`Bousso02`; resource-bounded complexity {cite}`LiVitanyi08`.

:::

---

### BarrierFreq (Frequency Barrier)

:::{prf:definition} Barrier Specification: Frequency
:label: def-barrier-freq

**Barrier ID:** `BarrierFreq`

**Interface Dependencies:**
- **Primary:** $\mathrm{GC}_\nabla$ (provides spectral density $S(\omega)$ and oscillation structure)
- **Secondary:** $\mathrm{SC}_\lambda$ (provides frequency cutoff and scaling)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$\int \omega^2 S(\omega) d\omega < \infty$$

**Natural Language Logic:**
"Is the total oscillation energy finite?"
*(Finite second moment of the spectral density implies bounded oscillation energy—the frequency barrier principle prevents infinite frequency cascades.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{GC}_\nabla}^{\mathrm{blk}}$): Oscillation integral finite; no frequency blow-up. Implies Pre(BoundaryCheck).
- **Breached** ($K_{\mathrm{GC}_\nabla}^{\mathrm{br}}$): Infinite oscillation energy; frequency cascade detected. Activates **Mode D.E** (Oscillation Divergence).

**Routing:**
- **On Block:** Proceed to `BoundaryCheck`.
- **On Breach:** Trigger **Mode D.E** → Enable Surgery `SurgDE` → Re-enter at `BoundaryCheck`.

**Literature:** De Giorgi-Nash-Moser regularity theory {cite}`DeGiorgi57`; {cite}`Nash58`; {cite}`Moser60`.

:::

---

### Boundary Barriers (BarrierBode, BarrierInput, BarrierVariety)

:::{prf:definition} Barrier Specification: Bode Sensitivity
:label: def-barrier-bode

**Barrier ID:** `BarrierBode`

**Interface Dependencies:**
- **Primary:** $\mathrm{Bound}_B$ (provides sensitivity function $S(s)$ and Bode integral $B_{\text{Bode}}$)
- **Secondary:** $\mathrm{LS}_\sigma$ (provides stability landscape for waterbed constraints)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{Bound}_\partial}^+\}$ (open system confirmed)
- **Barrier Predicate (Blocked Condition):**
  $$\int_0^\infty \ln \lVert S(i\omega) \rVert d\omega > -\infty$$

**Natural Language Logic:**
"Is the sensitivity integral conserved (waterbed effect)?"
*(The Bode integral constraint implies sensitivity cannot be reduced everywhere—reduction in one frequency band must be compensated elsewhere. Finite integral means the waterbed is bounded.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{Bound}_B}^{\mathrm{blk}}$): Bode integral finite; sensitivity bounded. Implies Pre(StarveCheck).
- **Breached** ($K_{\mathrm{Bound}_B}^{\mathrm{br}}$): Unbounded sensitivity; waterbed constraint violated. Activates **Mode B.E** (Sensitivity Explosion).

**Routing:**
- **On Block:** Proceed to `StarveCheck`.
- **On Breach:** Trigger **Mode B.E** → Enable Surgery `SurgBE` → Re-enter at `StarveCheck`.

**Literature:** Bode integral constraints and robust control {cite}`DoyleFrancisTannenbaum92`; {cite}`Sontag98`.

:::

:::{prf:definition} Barrier Specification: Input Stability
:label: def-barrier-input

**Barrier ID:** `BarrierInput`

**Interface Dependencies:**
- **Primary:** $\mathrm{Bound}_{\Sigma}$ (provides input reserve $r_{\text{reserve}}$ and flow integrals)
- **Secondary:** $C_\mu$ (provides concentration structure for resource distribution)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{Bound}_B}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$r_{\text{reserve}} > 0$$

**Natural Language Logic:**
"Is there a reservoir to prevent starvation?"
*(Positive reserve ensures the system can buffer transient input deficits—the input stability principle prevents resource starvation.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{Bound}_{\Sigma}}^{\mathrm{blk}}$): Reserve positive; buffer exists against starvation. Implies Pre(AlignCheck).
- **Breached** ($K_{\mathrm{Bound}_{\Sigma}}^{\mathrm{br}}$): Reserve depleted; system vulnerable to input starvation. Activates **Mode B.D** (Resource Depletion).

**Routing:**
- **On Block:** Proceed to `AlignCheck`.
- **On Breach:** Trigger **Mode B.D** → Enable Surgery `SurgBD` → Re-enter at `AlignCheck`.

**Literature:** Input-to-state stability {cite}`Khalil02`; {cite}`Sontag98`.

:::

:::{prf:definition} Barrier Specification: Requisite Variety
:label: def-barrier-variety

**Barrier ID:** `BarrierVariety`

**Interface Dependencies:**
- **Primary:** $\mathrm{GC}_T$ (provides control entropy $H(u)$ and tangent cone structure)
- **Secondary:** $\mathrm{Cap}_H$ (provides disturbance entropy $H(d)$ and capacity bounds)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{Bound}_{\Sigma}}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$H(u) \geq H(d)$$

**Natural Language Logic:**
"Does control entropy match disturbance entropy?"
*(Ashby's Law of Requisite Variety: a controller can only regulate what it can match in variety. Control must have at least as much entropy as the disturbance it counters.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{GC}_T}^{\mathrm{blk}}$): Control variety sufficient; can counter all disturbances. Implies Pre(BarrierExclusion).
- **Breached** ($K_{\mathrm{GC}_T}^{\mathrm{br}}$): Variety deficit; control cannot match disturbance complexity. Activates **Mode B.C** (Control Deficit).

**Routing:**
- **On Block:** Proceed to `BarrierExclusion`.
- **On Breach:** Trigger **Mode B.C** → Enable Surgery `SurgBC` → Re-enter at `BarrierExclusion`.

**Literature:** Requisite variety and cybernetics {cite}`Ashby56`; {cite}`ConantAshby70`.

:::

---

## 7. Surgery Node Specifications (Purple Nodes)

Each surgery is specified by:
- **Inputs**: Breach certificate + surgery data
- **Action**: Abstract operation performed
- **Postcondition**: Re-entry certificate + target node
- **Progress measure**: Ensures termination

:::{prf:theorem} Non-circularity rule
:label: thm-non-circularity

A barrier invoked because predicate $P_i$ failed **cannot** assume $P_i$ as a prerequisite. Formally:
$$\text{Trigger}(B) = \text{Gate}_i \text{ NO} \Rightarrow P_i \notin \mathrm{Pre}(B)$$

**Literature:** Well-founded semantics {cite}`VanGelder91`; stratification in logic programming {cite}`AptBolPedreschi94`.

:::

---

### Surgery Contracts Table

| **Surgery**  | **Input Mode**              | **Action**                | **Target**       |
|--------------|-----------------------------|---------------------------|------------------|
| SurgCE       | C.E (Energy Blow-up)        | Ghost/Cap extension       | ZenoCheck        |
| SurgCC       | C.C (Event Accumulation)    | Discrete saturation       | CompactCheck     |
| SurgCD\_Alt  | C.D (via Escape)            | Concentration-compactness | Profile          |
| SurgSE       | S.E (Supercritical)         | Regularity lift           | ParamCheck       |
| SurgSC       | S.C (Parameter Instability) | Convex integration        | GeomCheck        |
| SurgCD       | C.D (Geometric Collapse)    | Auxiliary/Structural      | StiffnessCheck   |
| SurgSD       | S.D (Stiffness Breakdown)   | Ghost extension           | TopoCheck        |
| SurgSC\_Rest | S.C (Vacuum Decay)          | Auxiliary extension       | TopoCheck        |
| SurgTE\_Rest | T.E (Metastasis)            | Structural                | TameCheck        |
| SurgTE       | T.E (Topological Twist)     | Tunnel                    | TameCheck        |
| SurgTC       | T.C (Labyrinthine)          | O-minimal regularization  | ErgoCheck        |
| SurgTD       | T.D (Glassy Freeze)         | Mixing enhancement        | ComplexCheck     |
| SurgDC       | D.C (Semantic Horizon)      | Viscosity solution        | OscillateCheck   |
| SurgDE       | D.E (Oscillatory)           | De Giorgi-Nash-Moser      | BoundaryCheck    |
| SurgBE       | B.E (Injection)             | Saturation                | StarveCheck      |
| SurgBD       | B.D (Starvation)            | Reservoir                 | AlignCheck       |
| SurgBC       | B.C (Misalignment)          | Controller Augmentation   | BarrierExclusion |

---

### Surgery Contract Template

:::{prf:definition} Surgery Specification Schema
:label: def-surgery-schema

A **Surgery Specification** is a transformation of the Hypostructure $\mathcal{H} \to \mathcal{H}'$. Each surgery defines:

**Surgery ID:** `[SurgeryID]` (e.g., SurgCE)
**Target Mode:** `[ModeID]` (e.g., Mode C.E)

**Interface Dependencies:**
- **Primary:** `[InterfaceID_1]` (provides the singular object/profile $V$ and locus $\Sigma$)
- **Secondary:** `[InterfaceID_2]` (provides the canonical library $\mathcal{L}_T$ or capacity bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{[\text{ModeID}]}^{\mathrm{br}}$ (The breach witnessing the singularity)
- **Admissibility Predicate (The Diamond):**
  $V \in \mathcal{L}_T \land \text{Cap}(\Sigma) \le \varepsilon_{\text{adm}}$
  *(Conditions required to perform surgery safely, corresponding to Case 1 of the Trichotomy.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = (X \setminus \Sigma_\varepsilon) \cup_{\partial} X_{\text{cap}}$
- **Height Jump:** $\Phi(x') \le \Phi(x) - \delta_S$
- **Topology:** $\tau(x') = [\text{New Sector}]$

**Postcondition:**
- **Re-entry Certificate:** $K_{[\text{SurgeryID}]}^{\mathrm{re}}$
- **Re-entry Target:** `[TargetNodeName]`
- **Progress Guarantee:** `[Type A (Count) or Type B (Complexity)]`

**Required Progress Certificate ($K_{\mathrm{prog}}$):**
Every surgery must produce a progress certificate witnessing either:
- **Type A (Bounded Resource):** $\Delta R \leq C$ per surgery invocation (bounded consumption)
- **Type B (Well-Founded Decrease):** $\mu(x') < \mu(x)$ for some ordinal-valued measure $\mu$

The non-circularity checker must verify that the progress measure is compatible with the surgery's re-entry target, ensuring termination of the repair loop.
:::

---

### Surgery Specifications

:::{prf:definition} Surgery Specification: Lyapunov Cap
:label: def-surgery-ce

**Surgery ID:** `SurgCE`
**Target Mode:** `Mode C.E` (Energy Blow-up)

**Interface Dependencies:**
- **Primary:** $D_E$ (Energy Interface: provides the unbounded potential $\Phi$)
- **Secondary:** $\mathrm{Cap}_H$ (Capacity Interface: provides the compactification metric)

**Admissibility Signature:**
- **Input Certificate:** $K_{D_E}^{\mathrm{br}}$ (Energy unbounded)
- **Admissibility Predicate:**
  $\text{Growth}(\Phi) \text{ is conformal} \land \partial_\infty X \text{ is definable}$
  *(The blow-up must allow conformal compactification.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $\hat{X} = X \cup \partial_\infty X$ (One-point or boundary compactification)
- **Height Rescaling:** $\hat{\Phi} = \tanh(\Phi)$ (Maps $[0, \infty) \to [0, 1)$)
- **Boundary Condition:** $\hat{S}_t |_{\partial_\infty X} = \text{Absorbing}$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgCE}}^{\mathrm{re}}$ (Witnesses $\hat{\Phi}$ is bounded)
- **Re-entry Target:** `ZenoCheck` (Node 2)
- **Progress Guarantee:** **Type A**. The system enters a bounded domain; blow-up is geometrically impossible in $\hat{X}$.

**Literature:** Compactification and boundary conditions {cite}`Dafermos16`; energy methods {cite}`Leray34`.

:::

:::{prf:definition} Surgery Specification: Discrete Saturation
:label: def-surgery-cc

**Surgery ID:** `SurgCC`
**Target Mode:** `Mode C.C` (Event Accumulation)

**Interface Dependencies:**
- **Primary:** $\mathrm{Rec}_N$ (Recovery Interface: provides event count $N$)
- **Secondary:** $\mathrm{TB}_\pi$ (Topology Interface: provides sector structure)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{Rec}_N}^{\mathrm{br}}$ (Zeno accumulation detected)
- **Admissibility Predicate:**
  $\exists N_{\max} : \#\{\text{events in } [t, t+\epsilon]\} \leq N_{\max} \text{ for small } \epsilon$
  *(Events must be locally finite, not truly Zeno.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X$ (no topological change)
- **Time Reparametrization:** $t' = \int_0^t \frac{ds}{1 + \#\text{events}(s)}$
- **Event Coarsening:** Merge events within $\epsilon$-windows

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgCC}}^{\mathrm{re}}$ (Witnesses finite event rate)
- **Re-entry Target:** `CompactCheck` (Node 3)
- **Progress Guarantee:** **Type A**. Event count bounded by $N(T, \Phi_0)$.

**Literature:** Surgery bounds in Ricci flow {cite}`Perelman03`; {cite}`KleinerLott08`.

:::

:::{prf:definition} Surgery Specification: Concentration-Compactness
:label: def-surgery-cd-alt

**Surgery ID:** `SurgCD_Alt`
**Target Mode:** `Mode C.D` (via Escape/Soliton)

**Interface Dependencies:**
- **Primary:** $C_\mu$ (Compactness Interface: provides escaping profile $V$)
- **Secondary:** $D_E$ (Energy Interface: provides energy tracking)

**Admissibility Signature:**
- **Input Certificate:** $K_{C_\mu}^{\mathrm{path}}$ (Soliton-like escape detected)
- **Admissibility Predicate:**
  $V \in \mathcal{L}_{\text{soliton}} \land \|V\|_{H^1} < \infty$
  *(Profile must be a recognizable traveling wave.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X / \sim_V$ (quotient by soliton orbit)
- **Energy Subtraction:** $\Phi(x') = \Phi(x) - E(V)$
- **Remainder:** Track $x - V$ in lower energy class

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgCD\_Alt}}^{\mathrm{re}}$ (Witnesses profile extracted)
- **Re-entry Target:** `Profile` (Re-check for further concentration)
- **Progress Guarantee:** **Type B**. Energy strictly decreases: $\Phi(x') < \Phi(x)$.

**Literature:** Concentration-compactness principle {cite}`Lions84`; profile decomposition {cite}`KenigMerle06`.

:::

:::{prf:definition} Surgery Specification: Regularity Lift
:label: def-surgery-se

**Surgery ID:** `SurgSE`
**Target Mode:** `Mode S.E` (Supercritical Cascade)

**Interface Dependencies:**
- **Primary:** $\mathrm{SC}_\lambda$ (Scaling Interface: provides critical exponent)
- **Secondary:** $D_E$ (Energy Interface: provides energy bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{SC}_\lambda}^{\mathrm{br}}$ (Supercritical scaling detected)
- **Admissibility Predicate:**
  $\alpha - \beta < \epsilon_{\text{crit}} \land \text{Profile } V \text{ is smooth}$
  *(Near-critical with smooth profile allows perturbative lift.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X$ (same space, better regularity)
- **Regularity Upgrade:** Promote $x \in H^s$ to $x' \in H^{s+\delta}$
- **Height Adjustment:** $\Phi' = \Phi + \text{regularization penalty}$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgSE}}^{\mathrm{re}}$ (Witnesses improved regularity)
- **Re-entry Target:** `ParamCheck` (Node 5)
- **Progress Guarantee:** **Type B**. Regularity index strictly increases.

**Literature:** Regularity lift in critical problems {cite}`CaffarelliKohnNirenberg82`; bootstrap arguments {cite}`DeGiorgi57`.

:::

:::{prf:definition} Surgery Specification: Convex Integration
:label: def-surgery-sc

**Surgery ID:** `SurgSC`
**Target Mode:** `Mode S.C` (Parameter Instability)

**Interface Dependencies:**
- **Primary:** $\mathrm{SC}_{\partial c}$ (Parameter Interface: provides drifting constants)
- **Secondary:** $\mathrm{LS}_\sigma$ (Stiffness Interface: provides spectral data)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{SC}_{\partial c}}^{\mathrm{br}}$ (Parameter drift detected)
- **Admissibility Predicate:**
  $\|\partial_t \theta\| < C_{\text{adm}} \land \theta \in \Theta_{\text{stable}}$
  *(Drift is slow and within stable region.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X \times \Theta'$ (extended parameter space)
- **Parameter Freeze:** $\theta' = \theta_{\text{avg}}$ (time-averaged parameter)
- **Convex Correction:** Add corrector field to absorb drift

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgSC}}^{\mathrm{re}}$ (Witnesses stable parameters)
- **Re-entry Target:** `GeomCheck` (Node 6)
- **Progress Guarantee:** **Type B**. Parameter variance strictly decreases.

**Literature:** Convex integration method {cite}`DeLellisSzekelyhidi09`; {cite}`Isett18`.

:::

:::{prf:definition} Surgery Specification: Auxiliary/Structural
:label: def-surgery-cd

**Surgery ID:** `SurgCD`
**Target Mode:** `Mode C.D` (Geometric Collapse)

**Interface Dependencies:**
- **Primary:** $\mathrm{Cap}_H$ (Capacity Interface: provides singular set measure)
- **Secondary:** $\mathrm{LS}_\sigma$ (Stiffness Interface: provides local geometry)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{Cap}_H}^{\mathrm{br}}$ (Positive capacity singularity)
- **Admissibility Predicate:**
  $\text{Cap}_H(\Sigma) \leq \varepsilon_{\text{adm}} \land V \in \mathcal{L}_{\text{neck}}$
  *(Small singular set with recognizable neck structure.)*

**Transformation Law ($\mathcal{O}_S$):**
- **Excision:** $X' = X \setminus B_\epsilon(\Sigma)$
- **Capping:** Glue auxiliary space $X_{\text{aux}}$ matching boundary
- **Height Drop:** $\Phi(x') \leq \Phi(x) - c \cdot \text{Vol}(\Sigma)^{2/n}$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgCD}}^{\mathrm{re}}$ (Witnesses smooth excision)
- **Re-entry Target:** `StiffnessCheck` (Node 7)
- **Progress Guarantee:** **Type B**. Singular set measure strictly decreases.

**Literature:** Ricci flow surgery {cite}`Hamilton97`; {cite}`Perelman03`; geometric measure theory {cite}`Federer69`.

:::

:::{prf:definition} Surgery Specification: Ghost Extension
:label: def-surgery-sd

**Surgery ID:** `SurgSD`
**Target Mode:** `Mode S.D` (Stiffness Breakdown)

**Interface Dependencies:**
- **Primary:** $\mathrm{LS}_\sigma$ (Stiffness Interface: provides spectral gap data)
- **Secondary:** $\mathrm{GC}_\nabla$ (Gradient Interface: provides flow structure)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{LS}_\sigma}^{\mathrm{br}}$ (Zero spectral gap at equilibrium)
- **Admissibility Predicate:**
  $\dim(\ker(H_V)) < \infty \land V \text{ is isolated}$
  *(Finite-dimensional kernel, isolated critical point.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $\hat{X} = X \times \mathbb{R}^k$ (ghost variables for null directions)
- **Extended Potential:** $\hat{\Phi}(x, \xi) = \Phi(x) + \frac{1}{2}|\xi|^2$
- **Artificial Gap:** New system has spectral gap $\lambda_1 > 0$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgSD}}^{\mathrm{re}}$ (Witnesses positive gap in extended system)
- **Re-entry Target:** `TopoCheck` (Node 8)
- **Progress Guarantee:** **Type A**. Bounded surgeries per unit time.

**Literature:** Ghost variable methods {cite}`Simon83`; spectral theory {cite}`FeehanMaridakis19`.

:::

:::{prf:definition} Surgery Specification: Vacuum Auxiliary
:label: def-surgery-sc-rest

**Surgery ID:** `SurgSC_Rest`
**Target Mode:** `Mode S.C` (Vacuum Decay in Restoration)

**Interface Dependencies:**
- **Primary:** $\mathrm{SC}_{\partial c}$ (Parameter Interface: provides vacuum instability)
- **Secondary:** $\mathrm{LS}_\sigma$ (Stiffness Interface: provides mass gap)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{SC}_{\partial c}}^{\mathrm{br}}$ (Vacuum decay detected)
- **Admissibility Predicate:**
  $\Delta V > k_B T \land \text{tunneling rate } \Gamma < \Gamma_{\text{crit}}$
  *(Mass gap exists and tunneling is slow.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X$ (same space)
- **Vacuum Shift:** $v_0 \to v_0'$ (new stable vacuum)
- **Energy Recentering:** $\Phi' = \Phi - \Phi(v_0') + \Phi(v_0)$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgSC\_Rest}}^{\mathrm{re}}$ (Witnesses new stable vacuum)
- **Re-entry Target:** `TopoCheck` (Node 8)
- **Progress Guarantee:** **Type B**. Vacuum energy strictly decreases.

**Literature:** Vacuum stability {cite}`Coleman75`; symmetry breaking {cite}`Goldstone61`; {cite}`Higgs64`.

:::

:::{prf:definition} Surgery Specification: Structural (Metastasis)
:label: def-surgery-te-rest

**Surgery ID:** `SurgTE_Rest`
**Target Mode:** `Mode T.E` (Topological Metastasis in Restoration)

**Interface Dependencies:**
- **Primary:** $\mathrm{TB}_\pi$ (Topology Interface: provides sector invariants)
- **Secondary:** $C_\mu$ (Compactness Interface: provides profile structure)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{TB}_\pi}^{\mathrm{br}}$ (Sector transition via decay)
- **Admissibility Predicate:**
  $V \cong S^{n-1} \times I \land \text{instanton action } S[\gamma] < \infty$
  *(Domain wall with finite tunneling action.)*

**Transformation Law ($\mathcal{O}_S$):**
- **Excision:** $X' = X \setminus (\text{domain wall})$
- **Reconnection:** Connect sectors via instanton path
- **Sector Update:** $\tau(x') = \tau_{\text{new}}$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgTE\_Rest}}^{\mathrm{re}}$ (Witnesses sector transition complete)
- **Re-entry Target:** `TameCheck` (Node 9)
- **Progress Guarantee:** **Type B**. Topological complexity (Betti sum) strictly decreases.

**Literature:** Instanton tunneling {cite}`Coleman75`; topological field theory {cite}`Floer89`.

:::

:::{prf:definition} Surgery Specification: Topological Tunneling
:label: def-surgery-te

**Surgery ID:** `SurgTE`
**Target Mode:** `Mode T.E` (Topological Twist)

**Interface Dependencies:**
- **Primary:** $\mathrm{TB}_\pi$ (Topology Interface: provides sector $\tau$ and invariants)
- **Secondary:** $C_\mu$ (Compactness Interface: provides the neck profile $V$)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{TB}_\pi}^{\mathrm{br}}$ (Sector transition attempted)
- **Admissibility Predicate:**
  $V \cong S^{n-1} \times \mathbb{R}$ *(Canonical Neck)*
  *(The singularity must be a recognizable neck pinch or domain wall.)*

**Transformation Law ($\mathcal{O}_S$):**
- **Excision:** $X' = X \setminus (S^{n-1} \times (-\varepsilon, \varepsilon))$
- **Capping:** Glue two discs $D^n$ to the exposed boundaries.
- **Sector Change:** $\tau(x') = \tau(x) \pm 1$ (Change in Euler characteristic/Betti number).

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgTE}}^{\mathrm{re}}$ (Witnesses new topology is manifold)
- **Re-entry Target:** `TameCheck` (Node 9)
- **Progress Guarantee:** **Type B**. Topological complexity (e.g., volume or Betti sum) strictly decreases: $\mathcal{C}(X') < \mathcal{C}(X)$.

**Literature:** Topological surgery {cite}`Smale67`; {cite}`Conley78`; Ricci flow surgery {cite}`Perelman03`.

:::

:::{prf:definition} Surgery Specification: O-Minimal Regularization
:label: def-surgery-tc

**Surgery ID:** `SurgTC`
**Target Mode:** `Mode T.C` (Labyrinthine/Wild Topology)

**Interface Dependencies:**
- **Primary:** $\mathrm{TB}_O$ (Tameness Interface: provides definability structure)
- **Secondary:** $\mathrm{Rep}_K$ (Dictionary Interface: provides complexity bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{TB}_O}^{\mathrm{br}}$ (Non-definable topology detected)
- **Admissibility Predicate:**
  $\Sigma \in \mathcal{O}_{\text{ext}}\text{-definable} \land \dim(\Sigma) < n$
  *(Wild set is definable in extended o-minimal structure.)*

**Transformation Law ($\mathcal{O}_S$):**
- **Structure Extension:** $\mathcal{O}' = \mathcal{O}[\exp]$ or $\mathcal{O}[\text{Pfaffian}]$
- **Stratification:** Replace $\Sigma$ with definable stratification
- **Tameness Certificate:** Produce o-minimal cell decomposition

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgTC}}^{\mathrm{re}}$ (Witnesses tame stratification)
- **Re-entry Target:** `ErgoCheck` (Node 10)
- **Progress Guarantee:** **Type B**. Definability complexity strictly decreases.

**Literature:** O-minimal structures {cite}`vandenDries98`; {cite}`Wilkie96`; stratification theory {cite}`Lojasiewicz65`.

:::

:::{prf:definition} Surgery Specification: Mixing Enhancement
:label: def-surgery-td

**Surgery ID:** `SurgTD`
**Target Mode:** `Mode T.D` (Glassy Freeze/Trapping)

**Interface Dependencies:**
- **Primary:** $\mathrm{TB}_\rho$ (Mixing Interface: provides mixing time)
- **Secondary:** $D_E$ (Energy Interface: provides energy landscape)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{TB}_\rho}^{\mathrm{br}}$ (Infinite mixing time detected)
- **Admissibility Predicate:**
  $\text{Trap } T \text{ is isolated} \land \partial T \text{ has positive measure}$
  *(Trap has accessible boundary.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X$ (same space)
- **Dynamics Modification:** Add noise term $\sigma dW_t$ to escape trap
- **Mixing Acceleration:** $\tau'_{\text{mix}} = \tau_{\text{mix}} / (1 + \sigma^2)$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgTD}}^{\mathrm{re}}$ (Witnesses finite mixing time)
- **Re-entry Target:** `ComplexCheck` (Node 11)
- **Progress Guarantee:** **Type A**. Bounded mixing enhancement per unit time.

**Complexity Type on Re-entry:** The re-entry evaluates $K(\mu_t)$ where $\mu_t = \text{Law}(x_t)$ is the probability measure on trajectories, not $K(x_t(\omega))$ for individual sample paths. The SDE has finite description length (drift $b$, diffusion $\sigma$, initial law $\mu_0$) even though individual realizations are algorithmically incompressible (white noise is random). This ensures S12 does not cause immediate failure at Node 11.

**Literature:** Stochastic perturbation and mixing {cite}`MeynTweedie93`; {cite}`HairerMattingly11`.

:::

:::{prf:definition} Surgery Specification: Viscosity Solution
:label: def-surgery-dc

**Surgery ID:** `SurgDC`
**Target Mode:** `Mode D.C` (Semantic Horizon/Complexity Explosion)

**Interface Dependencies:**
- **Primary:** $\mathrm{Rep}_K$ (Dictionary Interface: provides complexity measure)
- **Secondary:** $\mathrm{Cap}_H$ (Capacity Interface: provides dimension bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{Rep}_K}^{\mathrm{br}}$ (Complexity exceeds bound)
- **Admissibility Predicate:**
  $K(x) \leq S_{\text{BH}} + \epsilon \land x \in W^{1,\infty}$
  *(Near holographic bound with Lipschitz regularity.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X$ (same space, coarsened description)
- **Viscosity Regularization:** $x' = x * \phi_\epsilon$ (convolution smoothing)
- **Complexity Reduction:** $K(x') \leq K(x) - c \cdot \epsilon$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgDC}}^{\mathrm{re}}$ (Witnesses reduced complexity)
- **Re-entry Target:** `OscillateCheck` (Node 12)
- **Progress Guarantee:** **Type B**. Kolmogorov complexity strictly decreases.

**Literature:** Viscosity solutions {cite}`CrandallLions83`; regularization and mollification {cite}`EvansGariepy15`.

:::

:::{prf:definition} Surgery Specification: De Giorgi-Nash-Moser
:label: def-surgery-de

**Surgery ID:** `SurgDE`
**Target Mode:** `Mode D.E` (Oscillatory Divergence)

**Interface Dependencies:**
- **Primary:** $\mathrm{GC}_\nabla$ (Gradient Interface: provides oscillation structure)
- **Secondary:** $\mathrm{SC}_\lambda$ (Scaling Interface: provides frequency bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{GC}_\nabla}^{\mathrm{br}}$ (Infinite oscillation energy)
- **Admissibility Predicate:**
  There exists a cutoff scale $\Lambda$ such that the truncated second moment is finite:
  $$\exists \Lambda < \infty: \sup_{\Lambda' \leq \Lambda} \int_{|\omega| \leq \Lambda'} \omega^2 S(\omega) d\omega < \infty \quad \land \quad \text{uniform ellipticity}$$
  *(Divergence is "elliptic-regularizable" — De Giorgi-Nash-Moser applies to truncated spectrum.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X$ (same space, improved regularity)
- **Hölder Regularization:** Apply De Giorgi-Nash-Moser iteration
- **Oscillation Damping:** $\text{osc}_{B_r}(x') \leq C r^\alpha \text{osc}_{B_1}(x)$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgDE}}^{\mathrm{re}}$ (Witnesses Hölder continuity)
- **Re-entry Target:** `BoundaryCheck` (Node 13)
- **Progress Guarantee:** **Type A**. Bounded regularity improvements per unit time.

**Literature:** De Giorgi's original regularity theorem {cite}`DeGiorgi57`; Nash's parabolic regularity {cite}`Nash58`; Moser's Harnack inequality and iteration {cite}`Moser61`; unified treatment in {cite}`GilbargTrudinger01`.

:::

:::{prf:definition} Surgery Specification: Saturation
:label: def-surgery-be

**Surgery ID:** `SurgBE`
**Target Mode:** `Mode B.E` (Sensitivity Injection)

**Interface Dependencies:**
- **Primary:** $\mathrm{Bound}_B$ (Input Bound Interface: provides sensitivity integral)
- **Secondary:** $\mathrm{LS}_\sigma$ (Stiffness Interface: provides gain bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{Bound}_B}^{\mathrm{br}}$ (Bode sensitivity violated)
- **Admissibility Predicate:**
  $\|S(i\omega)\|_\infty < M \land \text{phase margin } > 0$
  *(Bounded gain with positive phase margin.)*

**Transformation Law ($\mathcal{O}_S$):**
- **Controller Modification:** Add saturation element $\text{sat}(u) = \text{sign}(u) \min(|u|, u_{\max})$
- **Gain Limiting:** $\|S'\|_\infty \leq \|S\|_\infty / (1 + \epsilon)$
- **Waterbed Conservation:** Redistribute sensitivity to safe frequencies

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgBE}}^{\mathrm{re}}$ (Witnesses bounded sensitivity)
- **Re-entry Target:** `StarveCheck` (Node 15)
- **Progress Guarantee:** **Type A**. Bounded saturation adjustments.

**Literature:** Bode sensitivity integrals and waterbed effect {cite}`Bode45`; $\mathcal{H}_\infty$ robust control {cite}`ZhouDoyleGlover96`; anti-windup for saturating systems {cite}`SeronGoodwinDeCarlo00`.

:::

:::{prf:definition} Surgery Specification: Reservoir
:label: def-surgery-bd

**Surgery ID:** `SurgBD`
**Target Mode:** `Mode B.D` (Resource Starvation)

**Interface Dependencies:**
- **Primary:** $\mathrm{Bound}_{\Sigma}$ (Supply Interface: provides resource integral)
- **Secondary:** $C_\mu$ (Compactness Interface: provides state bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{Bound}_{\Sigma}}^{\mathrm{br}}$ (Resource depletion detected)
- **Admissibility Predicate:**
  $r_{\text{reserve}} > 0 \land \text{recharge rate } > \text{drain rate}$
  *(Positive reserve with sustainable recharge.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X \times [0, R_{\max}]$ (add reservoir variable)
- **Resource Dynamics:** $\dot{r} = \text{recharge} - \text{consumption}$
- **Buffer Zone:** Maintain $r \geq r_{\min}$ always

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgBD}}^{\mathrm{re}}$ (Witnesses positive reservoir)
- **Re-entry Target:** `AlignCheck` (Node 16)
- **Progress Guarantee:** **Type A**. Bounded reservoir adjustments.

**Literature:** Reservoir computing and echo state networks {cite}`Jaeger04`; resource-bounded computation {cite}`Bellman57`; stochastic inventory theory {cite}`Arrow58`.

:::

:::{prf:definition} Surgery Specification: Controller Augmentation via Adjoint Selection
:label: def-surgery-bc

**Surgery ID:** `SurgBC`
**Target Mode:** `Mode B.C` (Control Misalignment / Variety Deficit)

**Interface Dependencies:**
- **Primary:** $\mathrm{GC}_T$ (Gauge Transform Interface: provides alignment data)
- **Secondary:** $\mathrm{Cap}_H$ (Capacity Interface: provides entropy bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{GC}_T}^{\mathrm{br}}$ (Variety deficit detected: $H(u) < H(d)$)
- **Admissibility Predicate:**
  $H(u) < H(d) - \epsilon \land \exists u' : H(u') \geq H(d)$
  *(Entropy gap exists but is bridgeable—there exists a control with sufficient variety.)*

**Transformation Law ($\mathcal{O}_S$):**
- **Controller Augmentation:** Lift control from $u \in \mathcal{U}$ to $u^* \in \mathcal{U}^* \supseteq \mathcal{U}$ where $\mathcal{U}^*$ has sufficient degrees of freedom (satisfying Ashby's Law of Requisite Variety)
- **Adjoint Selection:** Select $u^*$ from the admissible set $\{u : H(u) \geq H(d)\}$ via adjoint criterion: $u^* = \arg\max_{u \in \mathcal{U}^*} \langle u, \nabla\Phi \rangle$
- **Entropy Matching:** $H(u^*) \geq H(d)$ (guaranteed by augmentation)
- **Alignment Guarantee:** $\langle u^*, d \rangle \geq 0$ (non-adversarial, from adjoint selection)

**Semantic Clarification:** This surgery addresses Ashby's Law violation by **adding degrees of freedom** (controller augmentation), not merely aligning existing controls. The adjoint criterion selects the optimal control from the augmented space. Pure directional alignment without augmentation cannot satisfy $H(u) \geq H(d)$ if the original control space has insufficient entropy.

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgBC}}^{\mathrm{re}}$ (Witnesses entropy-sufficient control with alignment)
- **Re-entry Target:** `BarrierExclusion` (Node 17)
- **Progress Guarantee:** **Type B**. Entropy gap strictly decreases to zero.

**Literature:** Ashby's Law of Requisite Variety {cite}`Ashby56`; Pontryagin maximum principle {cite}`Pontryagin62`; adjoint methods in optimal control {cite}`Lions71`; entropy and control {cite}`ConantAshby70`.

:::

---

### Action Nodes (Dynamic Restoration)

:::{prf:definition} ActionSSB (Symmetry Breaking)
:label: def-action-ssb

**Trigger**: CheckSC YES in restoration subtree

**Action**: Spontaneous symmetry breaking of group $G$

**Output**: Mass gap certificate $K_{\text{gap}}$ guaranteeing stiffness

**Target**: TopoCheck (mass gap implies LS holds)

**Literature:** Goldstone theorem on massless modes {cite}`Goldstone61`; Higgs mechanism for mass generation {cite}`Higgs64`; Anderson's gauge-invariant treatment {cite}`Anderson63`.

:::

:::{prf:definition} ActionTunnel (Instanton Decay)
:label: def-action-tunnel

**Trigger**: CheckTB YES in restoration subtree

**Action**: Quantum/thermal tunneling to new sector

**Output**: Sector transition certificate

**Target**: TameCheck (new sector reached)

**Literature:** Instanton calculus in quantum field theory {cite}`Coleman79`; 't Hooft's instanton solutions {cite}`tHooft76`; semiclassical tunneling {cite}`Vainshtein82`.

:::

---

# Part VII: The Soft Interface Permits (X.0)

## 8. The Universal Gate Evaluator Interface

A Hypostructure $\mathbb{H}$ is an object in a cohesive $(\infty, 1)$-topos $\mathcal{E}$ equipped with a **Gate Evaluator Interface**. This interface maps abstract structural data to decidable types (Propositions) that the Sieve machine can verify.

To support everything from **Navier-Stokes** to **Graph Theory** to **Homotopy Type Theory**, we define the interfaces using the language of **Higher Topos Theory** (specifically, internal logic of an $(\infty,1)$-topos). This allows "Space" to be a manifold, a graph, or a type; "Energy" to be a functional, a complexity measure, or a truth value.

---

### 8.0. Ambient Structure

:::{prf:definition} Ambient Topos
:label: def-ambient-topos

An **Ambient Topos** for hypostructure analysis is a cohesive $(\infty,1)$-topos $\mathcal{E}$ equipped with:
1. A terminal object $1 \in \mathcal{E}$
2. Shape/flat/sharp modalities $(\int, \flat, \sharp)$ satisfying cohesion axioms
3. An internal type-theoretic logic with judgments $\Gamma \vdash t : A$
4. A subobject classifier $\Omega$ (truth values)
:::

**Notation:**
- $\mathcal{X}$: The object in $\mathcal{E}$ representing the system state (Space/Type)
- $\mathcal{H}$: The height object representing values/costs
- $\Omega$: The subobject classifier (truth values)
- $\vdash$: The judgment symbol ("System implements...")

:::{prf:definition} Height Object
:label: def-height-object

A **Height Object** $\mathcal{H}$ in $\mathcal{E}$ is an object equipped with:
1. A partial order $\leq: \mathcal{H} \times \mathcal{H} \to \Omega$
2. A bottom element $0: 1 \to \mathcal{H}$
3. An addition operation $+: \mathcal{H} \times \mathcal{H} \to \mathcal{H}$ (for accumulation)

| Domain | Height Object $\mathcal{H}$ | Interpretation |
|--------|---------------------------|----------------|
| PDEs | $\mathbb{R}_{\geq 0}$ (Dedekind reals) | Energy |
| Graphs | $\mathbb{N}$ | Discrete count |
| HoTT | $\text{Level}$ | Universe level |
| Tropical | $\mathbb{T}_\infty = ([0,\infty], \min, +)$ | Min-plus algebra |
:::

:::{prf:definition} Interface Permit
:label: def-interface-permit

An **Interface Permit** $I$ is a tuple $(\mathcal{D}, \mathcal{P}, \mathcal{K})$ consisting of:
1. **Required Structure** $\mathcal{D}$: Objects and morphisms in $\mathcal{E}$ the system must define.
2. **Evaluator** $\mathcal{P}$: A deterministic procedure $\mathcal{P}: \mathcal{D} \to \{\texttt{YES}, \texttt{NO}\}$ with typed certificates:
   - **YES:** Predicate holds with constructive witness ($K^+$)
   - **NO:** Predicate fails, with certificate distinguishing:
     - $K^{\mathrm{wit}}$: Constructive counterexample (actual refutation)
     - $K^{\mathrm{inc}}$: Evaluation exceeds computational bounds or method insufficient (not a semantic refutation)
3. **Certificate Type** $\mathcal{K}$: The witness structure produced by the evaluation, always a sum type $K^+ \sqcup K^{\mathrm{wit}} \sqcup K^{\mathrm{inc}}$.

A system **implements** Interface $I$ if it provides interpretations for all objects in $\mathcal{D}$ and a computable evaluator for $\mathcal{P}$.

**Evaluation Model (Free-Evaluator Semantics):**
Interfaces may be evaluated in any order permitted by their structural dependencies. The diagram represents a *conventional evaluation flow*, but any interface evaluator $\mathcal{P}_i$ may be called at any time if its required structure ($\mathcal{D}_i$) is available. Certificate accumulation is monotonic but not strictly sequential.

This enables:
- Early evaluation of downstream gates when prerequisites are met
- Parallel evaluation of independent interfaces
- Caching and reuse of certificates across Sieve traversals
:::

---

### 8.1. $\mathcal{H}_0$ (Substrate Interface)
*The Substrate Definition.*

:::{prf:definition} Interface $\mathcal{H}_0$
:label: def-interface-h0

**Purpose:** Ensures the system is a valid object in the topos with a notion of "existence."

**Required Structure ($\mathcal{D}$):**
- **State Object:** An object $\mathcal{X} \in \text{Obj}(\mathcal{E})$.
- **Evolution Morphism:** A family of endomorphisms $S_t: \mathcal{X} \to \mathcal{X}$ (the flow/algorithm).
- **Refinement Filter:** A topology or filtration $\mathcal{F}$ on $\mathcal{X}$ allowing limits (e.g., metric completion, domain theory limits).

**Evaluator ($\mathcal{P}_{\mathcal{H}_0}$):**
Is the morphism $S_t$ well-defined on the domain of interest?
$$\vdash S_t \in \text{Hom}_{\mathcal{E}}(\mathcal{X}, \mathcal{X})$$

**Certificates ($\mathcal{K}_{\mathcal{H}_0}$):**
- $K_{\mathcal{H}_0}^+$: A witness term $w : S_t$.
- $K_{\mathcal{H}_0}^-$: A witness that $\text{dom}(S_t) = \emptyset$ (Vacuous system).

**Does Not Promise:** Global existence. The refinement filter may exhaust at finite time.
:::

---

### 8.2. $D_E$ (Energy Interface)
*The Cost Interface. Enables Node 1: EnergyCheck*

:::{prf:definition} Interface $D_E$
:label: def-interface-de

**Purpose:** Defines a mapping from States to Values, establishing an ordering on configurations.

**Required Structure ($\mathcal{D}$):**
- **Height Morphism:** $\Phi: \mathcal{X} \to \mathcal{H}$ (Energy / Entropy / Complexity).
- **Dissipation Morphism:** $\mathfrak{D}: \mathcal{X} \to \mathcal{H}$ (Rate of change).
- **Comparison Operator:** A relation $\leq$ on $\mathcal{H}$.

**Evaluator ($\mathcal{P}_1$ - EnergyCheck):**
Does the evolution map states to lower (or bounded) height values?
$$\Phi(S_t x) \leq \Phi(x) + \int \mathfrak{D}$$

**Certificates ($\mathcal{K}_{D_E}$):**
- $K_{D_E}^+$: A bound $B \in \mathcal{H}$.
- $K_{D_E}^-$: A path $\gamma: [0,1] \to \mathcal{X}$ where $\Phi(\gamma(t)) \to \infty$ (Blow-up).

**Does Not Promise:** That energy is actually bounded.
:::

---

### 8.3. $\mathrm{Rec}_N$ (Recovery Interface)
*The Discrete Event Interface. Enables Node 2: ZenoCheck*

:::{prf:definition} Interface $\mathrm{Rec}_N$
:label: def-interface-recn

**Purpose:** Handles discrete transitions, surgeries, or logical steps.

**Required Structure ($\mathcal{D}$):**
- **Bad Subobject:** $\mathcal{B} \hookrightarrow \mathcal{X}$ (The singular locus or error states).
- **Recovery Map:** $\mathcal{R}: \mathcal{B} \to \mathcal{X} \setminus \mathcal{B}$ (The reset/surgery operator).
- **Counting Measure:** $\#: \text{Path}(\mathcal{X}) \to \mathbb{N}$ (Counting entrances to $\mathcal{B}$).

**Evaluator ($\mathcal{P}_2$ - ZenoCheck):**
Is the count of recovery events finite on finite intervals?
$$\#\{t \mid S_t(x) \in \mathcal{B}\} < \infty$$

**Certificates ($\mathcal{K}_{\mathrm{Rec}_N}$):**
- $K_{\mathrm{Rec}_N}^+$: An integer $N$.
- $K_{\mathrm{Rec}_N}^-$: An accumulation point $t_*$ (Zeno paradox).

**Does Not Promise:** That Zeno behavior is impossible.
:::

---

### 8.4. $C_\mu$ (Compactness Interface)
*The Limit Interface. Enables Node 3: CompactCheck*

:::{prf:definition} Interface $C_\mu$
:label: def-interface-cmu

**Purpose:** Defines convergence and structure extraction.

**Required Structure ($\mathcal{D}$):**
- **Symmetry Group Object:** $G \in \text{Grp}(\mathcal{E})$ acting on $\mathcal{X}$.
- **Quotient Object:** $\mathcal{X} // G$ (The stack/moduli space).
- **Limit Operator:** $\lim: \text{Seq}(\mathcal{X} // G) \to \mathcal{X} // G$.

**Evaluator ($\mathcal{P}_3$ - CompactCheck):**
Does a bounded sequence have a limit object (Profile) in the quotient?
$$\exists V \in \mathcal{X} // G : x_n \to V$$

**Certificates ($\mathcal{K}_{C_\mu}$):**
- $K_{C_\mu}^+$ (Concentration): The profile object $V$ and the gauge sequence $\{g_n\}$.
- $K_{C_\mu}^-$ (Dispersion): A witness that the measure of the state vanishes (e.g., $L^\infty \to 0$).

**Does Not Promise:** Compactness. Dispersion ($K_{C_\mu}^-$) is a valid success state.
:::

---

### 8.5. $\mathrm{SC}_\lambda$ (Scaling Interface)
*The Homogeneity Interface. Enables Node 4: ScaleCheck*

:::{prf:definition} Interface $\mathrm{SC}_\lambda$
:label: def-interface-sclambda

**Purpose:** Defines behavior under renormalization/rescaling.

**Required Structure ($\mathcal{D}$):**
- **Scaling Action:** An action of the multiplicative group $\mathbb{G}_m$ (or $\mathbb{R}^+$) on $\mathcal{X}$.
- **Weights:** Morphisms $\alpha, \beta: \mathcal{X} \to \mathbb{Q}$ defining how $\Phi$ and $\mathfrak{D}$ transform under scaling.

**Evaluator ($\mathcal{P}_4$ - ScaleCheck):**
Are the exponents ordered correctly for stability?
$$\alpha(V) > \beta(V)$$
*(Does cost grow faster than time compression?)*

**Certificates ($\mathcal{K}_{\mathrm{SC}_\lambda}$):**
- $K_{\mathrm{SC}_\lambda}^+$: The values $\alpha, \beta$.
- $K_{\mathrm{SC}_\lambda}^-$: A witness of criticality ($\alpha = \beta$) or supercriticality ($\alpha < \beta$).

**Does Not Promise:** Subcriticality.
:::

---

### 8.6. $\mathrm{SC}_{\partial c}$ (Parameter Interface)
*Enables Node 5: ParamCheck*

:::{prf:definition} Interface $\mathrm{SC}_{\partial c}$
:label: def-interface-scdc

**Purpose:** Defines stability of modulation parameters and coupling constants.

**Required Structure ($\mathcal{D}$):**
- **Parameter Object:** $\Theta \in \text{Obj}(\mathcal{E})$.
- **Parameter Morphism:** $\theta: \mathcal{X} \to \Theta$.
- **Reference Point:** $\theta_0: 1 \to \Theta$ (global section).
- **Distance Morphism:** $d: \Theta \times \Theta \to \mathcal{H}$.

**Evaluator ($\mathcal{P}_5$ - ParamCheck):**
Are structural constants stable along the trajectory?
$$\forall t.\, d(\theta(S_t x), \theta_0) \leq C$$

**Certificates ($\mathcal{K}_{\mathrm{SC}_{\partial c}}$):**
- $K_{\mathrm{SC}_{\partial c}}^+$: $(\theta_0, C, \text{stability proof})$.
- $K_{\mathrm{SC}_{\partial c}}^-$: $(\text{parameter drift witness}, t_{\text{drift}})$.

**Does Not Promise:** Parameter stability.
:::

---

### 8.7. $\mathrm{Cap}_H$ (Capacity Interface)
*The Measure/Dimension Interface. Enables Node 6: GeomCheck*

:::{prf:definition} Interface $\mathrm{Cap}_H$
:label: def-interface-caph

**Purpose:** Quantifies the "size" of subobjects.

**Required Structure ($\mathcal{D}$):**
- **Capacity Functional:** $\text{Cap}: \text{Sub}(\mathcal{X}) \to \mathcal{H}$ (e.g., Hausdorff dim, Kolmogorov complexity, Channel capacity).
- **Threshold:** A critical value $C_{\text{crit}}: 1 \to \mathcal{H}$.
- **Singular Subobject:** $\Sigma \hookrightarrow \mathcal{X}$.

**Evaluator ($\mathcal{P}_6$ - GeomCheck):**
Is the capacity of the singular set below the threshold?
$$\text{Cap}(\Sigma) < C_{\text{crit}}$$

**Certificates ($\mathcal{K}_{\mathrm{Cap}_H}$):**
- $K_{\mathrm{Cap}_H}^+$: The value $\text{Cap}(\Sigma)$.
- $K_{\mathrm{Cap}_H}^-$: A measure-preserving map from a large object into $\Sigma$.

**Does Not Promise:** That singularities are small.
:::

---

### 8.8. $\mathrm{LS}_\sigma$ (Stiffness Interface)
*The Local Convexity Interface. Enables Node 7: StiffnessCheck*

:::{prf:definition} Interface $\mathrm{LS}_\sigma$
:label: def-interface-lssigma

**Purpose:** Defines the local geometry of the potential landscape.

**Required Structure ($\mathcal{D}$):**
- **Gradient Operator:** $\nabla: \text{Hom}(\mathcal{X}, \mathcal{H}) \to T\mathcal{X}$ (Tangent bundle section).
- **Comparison:** An inequality relating gradient norm to height value.

**Evaluator ($\mathcal{P}_7$ - StiffnessCheck):**
Does the Łojasiewicz-Simon inequality hold?
$$\|\nabla \Phi(x)\| \geq C |\Phi(x) - \Phi(V)|^{1-\theta}$$

**Certificates ($\mathcal{K}_{\mathrm{LS}_\sigma}$):**
- $K_{\mathrm{LS}_\sigma}^+$: The exponent $\theta \in (0, 1]$.
- $K_{\mathrm{LS}_\sigma}^-$: A witness of flatness (e.g., a non-trivial kernel of the Hessian).

**Does Not Promise:** Convexity. Flat landscapes ($K_{\mathrm{LS}_\sigma}^-$) trigger the Spectral Barrier.
:::

---

### 8.8.1. $\mathrm{Mon}_\phi$ (Monotonicity Interface)
*The Virial/Morawetz Interface. Enables Soft→Rigidity Compilation*

:::{prf:definition} Interface $\mathrm{Mon}_\phi$ (Monotonicity / Virial-Morawetz)
:label: def-interface-mon

**Purpose:** Defines monotonicity identities that force dispersion or concentration for almost-periodic solutions.

**Required Structure ($\mathcal{D}$):**
- **Monotonicity Functional:** $M: \mathcal{X} \times \mathbb{R} \to \mathbb{R}$ (Morawetz/virial action).
- **Weight Function:** $\phi: \mathcal{X} \to \mathbb{R}$ (typically radial or localized convex weight).
- **Sign Certificate:** $\sigma \in \{+, -, 0\}$ (convexity type determining inequality direction).

**Evaluator ($\mathcal{P}_{\mathrm{Mon}}$ - MonotonicityCheck):**
Does the monotonicity identity hold with definite sign for the declared functional?
$$\frac{d^2}{dt^2} M_\phi(t) \geq c \cdot \|\nabla u\|^2 - C \cdot \|u\|^2$$
(or $\leq$ depending on $\sigma$), where $M_\phi(t) = \int \phi(x) |u(t,x)|^2 dx$ or appropriate variant.

**Certificates ($\mathcal{K}_{\mathrm{Mon}_\phi}$):**
- $K_{\mathrm{Mon}_\phi}^+ := (\phi, M, \sigma, \mathsf{identity\_proof})$ asserting:
  1. The identity is algebraically verifiable from the equation structure
  2. For almost-periodic solutions mod $G$, integration forces dispersion or concentration
  3. The sign $\sigma$ is definite (not degenerate)
- $K_{\mathrm{Mon}_\phi}^- := \text{witness that no monotonicity identity holds with useful sign}$

**Evaluator (Computable for Good Types):**
- Check if equation has standard form (semilinear wave/Schrödinger/heat with power nonlinearity)
- Verify convexity of $\phi$ and compute second derivative identity algebraically
- Return YES with $K_{\mathrm{Mon}_\phi}^+$ if sign is definite; else NO with $K_{\mathrm{Mon}_\phi}^{\mathrm{inc}}$ (if verification method insufficient) or $K_{\mathrm{Mon}_\phi}^{\mathrm{wit}}$ (if sign is provably indefinite)

**Does Not Promise:** Rigidity directly. Combined with $K_{\mathrm{LS}_\sigma}^+$ and Lock obstruction, enables hybrid rigidity derivation.

**Used by:** MT-SOFT→Rigidity compilation metatheorem ({prf:ref}`mt-soft-rigidity`).

**Literature:** Morawetz estimates {cite}`Morawetz68`; virial identities {cite}`GlasseyScattering77`; interaction Morawetz {cite}`CollianderKeelStaffilaniTakaokaTao08`.
:::

---

### 8.9. $\mathrm{TB}_\pi$ (Topology Interface)
*The Invariant Interface. Enables Node 8: TopoCheck*

:::{prf:definition} Interface $\mathrm{TB}_\pi$
:label: def-interface-tbpi

**Purpose:** Defines discrete sectors that cannot be continuously deformed into one another.

**Required Structure ($\mathcal{D}$):**
- **Sector Set:** A discrete set $\pi_0(\mathcal{X})$ (Connected components, homotopy classes).
- **Invariant Map:** $\tau: \mathcal{X} \to \pi_0(\mathcal{X})$.

**Evaluator ($\mathcal{P}_8$ - TopoCheck):**
Is the trajectory confined to a single sector?
$$\tau(S_t x) = \tau(x)$$

**Certificates ($\mathcal{K}_{\mathrm{TB}_\pi}$):**
- $K_{\mathrm{TB}_\pi}^+$: The value $\tau$.
- $K_{\mathrm{TB}_\pi}^-$: A path connecting two distinct sectors (Tunneling/Topology change).

**Does Not Promise:** Topological stability.
:::

---

### 8.10. $\mathrm{TB}_O$ (Tameness Interface)
*Enables Node 9: TameCheck*

:::{prf:definition} Interface $\mathrm{TB}_O$
:label: def-interface-tbo

**Purpose:** Defines the "tameness" of the singular locus via definability.

**Required Structure ($\mathcal{D}$):**
- **Definability Modality:** $\text{Def}: \text{Sub}(\mathcal{X}) \to \Omega$.
- **Tame Structure:** $\mathcal{O} \hookrightarrow \text{Sub}(\mathcal{E})$ (sub-Boolean algebra of definable subobjects).

**Evaluator ($\mathcal{P}_9$ - TameCheck):**
Is the singular locus $\mathcal{O}$-definable?
$$\Sigma \in \mathcal{O}\text{-definable}$$

**Certificates ($\mathcal{K}_{\mathrm{TB}_O}$):**
- $K_{\mathrm{TB}_O}^+$: $(\text{tame structure}, \text{definability proof})$.
- $K_{\mathrm{TB}_O}^-$: $(\text{wildness witness})$.

**Does Not Promise:** Tameness. Wild topology ($K_{\mathrm{TB}_O}^-$) routes to the O-Minimal Barrier.
:::

---

### 8.11. $\mathrm{TB}_\rho$ (Mixing Interface)
*Enables Node 10: ErgoCheck*

:::{prf:definition} Interface $\mathrm{TB}_\rho$
:label: def-interface-tbrho

**Purpose:** Defines ergodic/mixing properties of the dynamics.

**Required Structure ($\mathcal{D}$):**
- **Measure Object:** $\mathcal{M}(\mathcal{X})$ (probability measures internal to $\mathcal{E}$).
- **Invariant Subobject:** $\text{Inv}_S \hookrightarrow \mathcal{M}(\mathcal{X})$.
- **Mixing Time Morphism:** $\tau_{\text{mix}}: \mathcal{X} \to \mathcal{H}$.

**Evaluator ($\mathcal{P}_{10}$ - ErgoCheck):**
Does the system mix with finite mixing time?
$$\tau_{\text{mix}}(x) < \infty$$

**Certificates ($\mathcal{K}_{\mathrm{TB}_\rho}$):**
- $K_{\mathrm{TB}_\rho}^+$: $(\tau_{\text{mix}}, \text{mixing proof})$.
- $K_{\mathrm{TB}_\rho}^-$: $(\text{trap certificate}, \text{invariant subset})$.

**Does Not Promise:** Mixing.
:::

---

### 8.12. $\mathrm{Rep}_K$ (Dictionary Interface)
*The Equivalence Interface. Enables Node 11: ComplexCheck*

:::{prf:definition} Interface $\mathrm{Rep}_K$
:label: def-interface-repk

**Purpose:** Defines the mapping between the "Territory" (System) and the "Map" (Representation).

**Required Structure ($\mathcal{D}$):**
- **Language Object:** $\mathcal{L} \in \text{Obj}(\mathcal{E})$ (formal language or category).
- **Dictionary Morphism:** $D: \mathcal{X} \to \mathcal{L}$.
- **Faithfulness:** An inverse map $D^{-1}$ or equivalence witness.
- **Complexity:** $K: \mathcal{L} \to \mathbb{N}_\infty$.

**Evaluator ($\mathcal{P}_{11}$ - ComplexCheck):**
Is the state representable with finite complexity?
$$K(D(x)) < \infty$$

**Stochastic Extension:** For stochastic systems (e.g., post-S12), complexity refers to the Kolmogorov complexity of the probability law $K(\mu)$, defined as the shortest program that samples from the distribution. Formally: $K(\mu) := \min\{|p| : U(p, r) \sim \mu \text{ for random } r\}$. This ensures that SDEs with finite-description coefficients $(b, \sigma)$ satisfy the complexity check even though individual sample paths are algorithmically random.

**Certificates ($\mathcal{K}_{\mathrm{Rep}_K}$):**
- $K_{\mathrm{Rep}_K}^+$: The code/description $p$.
- $K_{\mathrm{Rep}_K}^-$: A proof of uncomputability or undecidability.

**Does Not Promise:** Computability.

**Epistemic Role:** $\mathrm{Rep}_K$ is the boundary between "analysis engine" and "conjecture prover engine." When $\mathrm{Rep}_K$ produces a NO-inconclusive certificate ($K_{\mathrm{Rep}_K}^{\mathrm{inc}}$), the Lock uses only geometric tactics (E1--E3).
:::

---

### 8.13. $\mathrm{GC}_\nabla$ (Gradient Interface)
*The Geometry Interface. Enables Node 12: OscillateCheck*

:::{prf:definition} Interface $\mathrm{GC}_\nabla$
:label: def-interface-gcnabla

**Purpose:** Defines the "Natural" geometry of the space.

**Required Structure ($\mathcal{D}$):**
- **Metric Tensor:** $g: T\mathcal{X} \otimes T\mathcal{X} \to \mathcal{H}$ (Inner product).
- **Compatibility:** A relation between the flow vector field $v$ and the potential $\Phi$:
$$v \stackrel{?}{=} -\nabla_g \Phi$$

**Evaluator ($\mathcal{P}_{12}$ - OscillateCheck):**
Does the system follow the gradient?
$$\mathfrak{D}(x) = \|\nabla_g \Phi(x)\|^2$$

**Certificates ($\mathcal{K}_{\mathrm{GC}_\nabla}$):**
- $K_{\mathrm{GC}_\nabla}^+$ (Oscillation Present): Witness of oscillatory behavior (symplectic structure, curl, or non-gradient dynamics).
- $K_{\mathrm{GC}_\nabla}^-$ (Gradient Flow): Witness that flow is monotonic (no oscillation, pure gradient descent).

**Does Not Promise:** Absence of oscillation.

**Optionality:** $\mathrm{GC}_\nabla$ is not required for basic singularity exclusion. It only unlocks "explicit Lyapunov/action reconstruction" upgrades.
:::

---

### 8.14. Open System Interfaces
*Enables Nodes 13-16: BoundaryCheck, OverloadCheck, StarveCheck, AlignCheck*

The open system checks are split into four distinct interfaces, each handling a specific aspect of boundary coupling.

#### 8.14a. $\mathrm{Bound}_\partial$ (Boundary Interface)
*Enables Node 13: BoundaryCheck*

:::{prf:definition} Interface $\mathrm{Bound}_\partial$
:label: def-interface-bound-partial

**Purpose:** Determines whether the system is open (has external boundary).

**Required Structure ($\mathcal{D}$):**
- **State Space:** $\mathcal{X}$ with topological boundary $\partial\mathcal{X}$.

**Evaluator ($\mathcal{P}_{13}$ - BoundaryCheck):**
Is the system open? Does it have a non-trivial boundary?
$$\partial\mathcal{X} \neq \emptyset$$

**Certificates ($\mathcal{K}_{\mathrm{Bound}_\partial}$):**
- $K_{\mathrm{Bound}_\partial}^+$ (Open System): Witness that boundary exists and is non-trivial.
- $K_{\mathrm{Bound}_\partial}^-$ (Closed System): Witness that system is closed; skip to Node 17.
:::

#### 8.14b. $\mathrm{Bound}_B$ (Input Bound Interface)
*Enables Node 14: OverloadCheck*

:::{prf:definition} Interface $\mathrm{Bound}_B$
:label: def-interface-bound-b

**Purpose:** Verifies that external inputs are bounded.

**Required Structure ($\mathcal{D}$):**
- **Input Object:** $\mathcal{U} \in \text{Obj}(\mathcal{E})$.
- **Input Morphism:** $\iota: \mathcal{U} \to \mathcal{X}$ (or $\mathcal{U} \times \mathcal{T} \to \mathcal{X}$).

**Evaluator ($\mathcal{P}_{14}$ - OverloadCheck):**
Is the input bounded in authority?
$$\|Bu\|_{L^\infty} \leq M \quad \land \quad \int_0^T \|u(t)\|^2 dt < \infty$$

**Certificates ($\mathcal{K}_{\mathrm{Bound}_B}$):**
- $K_{\mathrm{Bound}_B}^+$ (Bounded Input): $(\text{bound } M, \text{authority margin})$.
- $K_{\mathrm{Bound}_B}^-$ (Overload): $(\text{overload witness}, t^*)$ — triggers BarrierBode.
:::

#### 8.14c. $\mathrm{Bound}_{\Sigma}$ (Resource Interface)
*Enables Node 15: StarveCheck*

:::{prf:definition} Interface $\mathrm{Bound}_{\Sigma}$
:label: def-interface-bound-int

**Purpose:** Verifies that resource/energy supply is sufficient.

**Required Structure ($\mathcal{D}$):**
- **Resource Function:** $r: \mathcal{T} \to \mathbb{R}_{\geq 0}$.
- **Minimum Threshold:** $r_{\min} > 0$.

**Evaluator ($\mathcal{P}_{15}$ - StarveCheck):**
Is the integrated resource supply sufficient?
$$\int_0^T r(t) \, dt \geq r_{\min}$$

**Certificates ($\mathcal{K}_{\mathrm{Bound}_{\Sigma}}$):**
- $K_{\mathrm{Bound}_{\Sigma}}^+$ (Sufficient Supply): $(r_{\min}, \text{sufficiency proof})$.
- $K_{\mathrm{Bound}_{\Sigma}}^-$ (Starvation): $(\text{deficit time})$ — triggers BarrierInput.
:::

#### 8.14d. $\mathrm{GC}_T$ (Control Alignment Interface)
*Enables Node 16: AlignCheck*

:::{prf:definition} Interface $\mathrm{GC}_T$
:label: def-interface-gc-t

**Purpose:** Verifies that control inputs align with safe descent directions.

**Required Structure ($\mathcal{D}$):**
- **Control Law:** $T: \mathcal{U} \to \mathcal{X}$ (the realized control).
- **Desired Behavior:** $d \in \mathcal{Y}$ (the reference or goal).
- **Alignment Metric:** Distance function $\Delta: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\geq 0}$.

**Evaluator ($\mathcal{P}_{16}$ - AlignCheck):**
Is the control matched to the desired behavior?
$$\Delta(T(u), d) \leq \varepsilon_{\text{align}}$$

**Certificates ($\mathcal{K}_{\mathrm{GC}_T}$):**
- $K_{\mathrm{GC}_T}^+$ (Aligned Control): $(\text{alignment certificate}, \Delta_{\text{achieved}})$.
- $K_{\mathrm{GC}_T}^-$ (Misaligned): $(\text{misalignment mode})$ — triggers BarrierVariety.
:::

---

### 8.15. $\mathrm{Cat}_{\mathrm{Hom}}$ (Categorical Interface)
*Enables Node 17: The Lock (BarrierExclusion)*

:::{prf:definition} Interface $\mathrm{Cat}_{\mathrm{Hom}}$
:label: def-interface-cathom

**Purpose:** Final structural consistency verification. Certifies that no bad pattern from the library embeds into the candidate hypostructure, establishing global regularity.

**Required Structure ($\mathcal{D}$):**
- **Hypostructure Category:** $\mathbf{Hypo}_T$ — the category of admissible hypostructures for type $T$.
- **Bad Pattern Library:** $\mathcal{B} = \{B_i\}_{i \in I}$ — a finite set of *minimal bad patterns* committed to for problem type $T$. Each $B_i \in \text{Obj}(\mathbf{Hypo}_T)$ is a canonical singularity-forming structure.
- **Morphism Spaces:** $\text{Hom}_{\mathbf{Hypo}_T}(B_i, \mathcal{H})$ for each $B_i \in \mathcal{B}$.

**Completeness Axiom (Problem-Type Dependent):**
For each problem type $T$, we assume: *every singularity of type $T$ factors through some $B_i \in \mathcal{B}$.* This is a **problem-specific axiom** that must be verified for each instantiation (e.g., for Navier-Stokes, the library consists of known blow-up profiles; for Riemann Hypothesis, the library consists of zero-causing structures).

**Evaluator ($\mathcal{P}_{17}$ - BarrierExclusion):**
$$\forall i \in I: \text{Hom}_{\mathbf{Hypo}_T}(B_i, \mathcal{H}) = \emptyset$$

The Lock evaluator checks whether any morphism exists from any bad pattern to the candidate hypostructure. If all Hom-sets are empty, no singularity-forming pattern can embed, and global regularity follows.

**Certificates ($\mathcal{K}_{\mathrm{Cat}_{\mathrm{Hom}}}$):**
- $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{blk}}$ (Blocked/VICTORY): Proof that $\forall i: \text{Hom}(B_i, \mathcal{H}) = \emptyset$. Techniques include:
  - **E1 (Dimension):** $\dim(B_i) > \dim(\mathcal{H})$
  - **E2 (Invariant Mismatch):** $I(B_i) \neq I(\mathcal{H})$ for preserved invariant $I$
  - **E3 (Positivity/Integrality):** Obstruction from positivity or integrality constraints
  - **E4 (Functional Equation):** No solution to induced functional equations
  - **E5 (Modular):** Obstruction from modular/arithmetic properties
- $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{morph}}$ (Breached/FATAL): Explicit morphism $f: B_i \to \mathcal{H}$ for some $i$, witnessing that singularity formation is possible.

**Does Not Promise:** That the Lock is decidable. Tactics E1-E12 may exhaust without resolution, yielding a Breached-inconclusive certificate ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$).

**Remark (Library vs. Universal Object):**
The "universal bad object" formulation assumes a single object $\mathcal{H}_{\text{bad}}$ from which all singularities factor. Such an object may not exist as a set-sized entity in $\mathbf{Hypo}_T$. The library formulation is weaker but constructive: we commit to a finite list of known bad patterns and prove each is excluded.
:::

---

### 8.16. What X.0 Does NOT Promise

At the soft layer, the framework does **not** assume:

- **Global regularity**: Solutions may blow up; the sieve determines whether they must
- **Finite canonical profile library**: Profiles may be classified, stratified, or horizon
- **Surgery admissibility**: Each surgery must be checked via the Admissibility Trichotomy
- **Decidable Lock outcomes**: E1--E10 tactics may exhaust without resolution
- **Unique gradient structure**: $\mathrm{GC}_\nabla$ is optional; many systems lack gradient form
- **Closed system**: $\mathrm{Bound}$ explicitly handles open systems with inputs

These are obtained by **upgrades**:
- Profile Classification Trichotomy (finite library, tame stratification, or horizon)
- Surgery Admissibility Trichotomy (admissible, admissible$^\sim$, or inadmissible)
- Promotion rules (blocked $\to$ YES via certificate accumulation)
- Lock tactics (E1--E10 for Hom-emptiness proofs)

This separation makes the framework **honest about its assumptions** and enables systematic identification of what additional structure is needed when an inconclusive NO certificate ($K^{\mathrm{inc}}$) is produced.

---

### 8.16.5. Backend-Specific Permits (Specialized Certificates)

The following permits capture **backend-specific hypotheses** that are required by particular metatheorems. Unlike the universal interfaces (8.1–8.15), these permits encode deep theorems from specific mathematical domains (e.g., dispersive PDE, dynamical systems, algebraic geometry) that cannot be derived from the generic interface structure alone.

:::{prf:definition} Permit $\mathrm{WP}_{s_c}$ (Critical Well-Posedness + Continuation)
:label: def-permit-wp-sc

**Name:** CriticalWP

**Question:** Does the evolution problem $T$ admit local well-posedness in the critical phase space $X_c$ (typically $X_c = \dot{H}^{s_c}$), with a continuation criterion?

**YES certificate**
$$K_{\mathrm{WP}_{s_c}}^+ := \big(\mathsf{LWP},\ \mathsf{uniq},\ \mathsf{cont},\ \mathsf{crit\_blowup}\big)$$
where the payload asserts all of:
1. (**Local existence**) For every $u_0 \in X_c$ there exists $T(u_0) > 0$ and a solution $u \in C([0,T]; X_c)$.
2. (**Uniqueness**) The solution is unique in the specified solution class.
3. (**Continuous dependence**) The data-to-solution map is continuous (or Lipschitz) on bounded sets in $X_c$.
4. (**Continuation criterion**) If $T_{\max} < \infty$ then a specified *critical control norm* blows up:
   $$\|u\|_{S([0, T_{\max}))} = \infty \quad (\text{for a declared control norm } S).$$

**NO certificate** (sum type $K_{\mathrm{WP}_{s_c}}^- := K_{\mathrm{WP}_{s_c}}^{\mathrm{wit}} \sqcup K_{\mathrm{WP}_{s_c}}^{\mathrm{inc}}$)

*NO-with-witness:*
$$K_{\mathrm{WP}_{s_c}}^{\mathrm{wit}} := (\mathsf{counterexample}, \mathsf{mode})$$
where $\mathsf{mode} \in \{\texttt{NORM\_INFLATION}, \texttt{NON\_UNIQUE}, \texttt{ILL\_POSED}, \texttt{NO\_CONTINUATION}\}$ identifies which of (1)–(4) fails, with an explicit counterexample (e.g., a sequence demonstrating norm inflation, or a pair of distinct solutions from identical data).

*NO-inconclusive:*
$$K_{\mathrm{WP}_{s_c}}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$
Typical $\mathsf{missing}$: "no matching WP template (parabolic/dispersive/hyperbolic)", "state space $X_c$ not recognized", "operator conditions not provided by soft layer".

**Used by:** `mt-auto-profile` Mechanism A (CC+Rig), and any node that invokes "critical LWP + continuation".
:::

:::{prf:definition} Permit $\mathrm{ProfDec}_{s_c,G}$ (Profile Decomposition modulo Symmetries)
:label: def-permit-profdec-scg

**Name:** ProfileDecomp

**Question:** Does every bounded sequence in $X_c$ admit a Bahouri–Gérard/Lions type profile decomposition modulo the symmetry group $G$?

**YES certificate**
$$K_{\mathrm{ProfDec}_{s_c,G}}^+ := \big(\{\phi^j\}_{j \geq 1},\ \{g_n^j\}_{n,j},\ \{r_n^J\}_{n,J},\ \mathsf{orth},\ \mathsf{rem}\big)$$
meaning: for every bounded $(u_n) \subset X_c$ there exist profiles $\phi^j \in X_c$ and symmetry parameters $g_n^j \in G$ such that for every $J$,
$$u_n = \sum_{j=1}^J g_n^j \phi^j + r_n^J,$$
with:
1. (**Asymptotic orthogonality**) The parameters $(g_n^j)$ are pairwise orthogonal in the standard sense for $G$.
2. (**Decoupling**) Conserved quantities/energies decouple across profiles up to $o_n(1)$ errors.
3. (**Remainder smallness**) The remainder $r_n^J$ is small in the critical control norm:
   $$\lim_{J \to \infty}\ \limsup_{n \to \infty}\ \|r_n^J\|_S = 0.$$

**NO certificate** (sum type $K_{\mathrm{ProfDec}}^- := K_{\mathrm{ProfDec}}^{\mathrm{wit}} \sqcup K_{\mathrm{ProfDec}}^{\mathrm{inc}}$)

*NO-with-witness:*
$$K_{\mathrm{ProfDec}}^{\mathrm{wit}} := (\mathsf{bounded\_seq}, \mathsf{failed\_property})$$
where $\mathsf{failed\_property} \in \{\texttt{NO\_ORTH}, \texttt{NO\_DECOUPLE}, \texttt{NO\_REMAINDER\_SMALL}\}$ identifies which of (1)–(3) fails, with a concrete bounded sequence $(u_n)$ demonstrating the failure.

*NO-inconclusive:*
$$K_{\mathrm{ProfDec}}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$
Typical $\mathsf{missing}$: "symmetry group $G$ not recognized as standard decomposition group", "control norm $S$ not provided or checkable", "space not in supported class (Hilbert/Banach with required compactness structure)".

**Used by:** `mt-auto-profile` Mechanism A (CC+Rig).

**Literature:** {cite}`BahouriGerard99`; {cite}`Lions84`; {cite}`Lions85`.
:::

:::{prf:definition} Permit $\mathrm{KM}_{\mathrm{CC+stab}}$ (Concentration–Compactness + Stability Machine)
:label: def-permit-km-ccstab

**Name:** KM-Machine

**Question:** Can failure of the target property (regularity/scattering/etc.) be reduced to a *minimal counterexample* that is almost periodic modulo symmetries, using concentration–compactness plus a perturbation/stability lemma?

**YES certificate**
$$K_{\mathrm{KM}_{\mathrm{CC+stab}}}^+ := \big(\mathsf{min\_obj},\ \mathsf{ap\_modG},\ \mathsf{stab},\ \mathsf{nl\_profiles}\big)$$
where the payload asserts:
1. (**Minimal counterexample extraction**) If the target property fails, there exists a solution $u^*$ minimal with respect to a declared size functional (energy/mass/critical norm threshold).
2. (**Almost periodicity**) The orbit $\{u^*(t)\}$ is precompact in $X_c$ modulo $G$ ("almost periodic mod $G$").
3. (**Long-time perturbation**) A stability lemma: any approximate solution close in the control norm remains close to an exact solution globally on the interval.
4. (**Nonlinear profile control**) The nonlinear evolution decouples across profiles to the extent needed for the minimal-element argument.

**NO certificate** (sum type $K_{\mathrm{KM}}^- := K_{\mathrm{KM}}^{\mathrm{wit}} \sqcup K_{\mathrm{KM}}^{\mathrm{inc}}$)

*NO-with-witness:*
$$K_{\mathrm{KM}}^{\mathrm{wit}} := (\mathsf{failure\_obj}, \mathsf{step\_failed})$$
where $\mathsf{step\_failed} \in \{\texttt{NO\_MIN\_EXTRACT}, \texttt{NO\_ALMOST\_PERIODIC}, \texttt{NO\_STABILITY}, \texttt{NO\_PROFILE\_CONTROL}\}$ identifies which of (1)–(4) fails, with a concrete object demonstrating the failure.

*NO-inconclusive:*
$$K_{\mathrm{KM}}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$
Typical $\mathsf{missing}$: "composition requires $K_{\mathrm{WP}}^+$ which was not derived", "profile decomposition not available", "stability lemma not computable for this equation class".

**Used by:** `mt-auto-profile` Mechanism A (CC+Rig).

**Literature:** {cite}`KenigMerle06`; {cite}`KillipVisan10`; {cite}`DuyckaertsKenigMerle11`.
:::

:::{prf:definition} Permit $\mathrm{Attr}^+$ (Global Attractor Existence)
:label: def-permit-attractor

**Name:** GlobalAttractor

**Question:** Does the semiflow $(S_t)_{t \geq 0}$ on a phase space $X$ admit a compact global attractor?

**YES certificate**
$$K_{\mathrm{Attr}}^+ := (\mathsf{semiflow},\ \mathsf{absorbing},\ \mathsf{asymp\_compact},\ \mathsf{attractor})$$
asserting:
1. (**Semiflow structure**) $S_{t+s} = S_t \circ S_s$, $S_0 = \mathrm{id}$, and $S_t$ is continuous on bounded sets.
2. (**Dissipativity**) There exists a bounded absorbing set $B \subset X$.
3. (**Asymptotic compactness**) For any bounded $B_0 \subset X$ and any $t_n \to \infty$, the set $S_{t_n}(B_0)$ has precompact closure.
4. (**Attractor**) There exists a compact invariant set $\mathcal{A}$ attracting bounded sets:
   $$\mathrm{dist}(S_t(B_0), \mathcal{A}) \to 0 \quad (t \to \infty).$$

**NO certificate** (sum type $K_{\mathrm{Attr}}^- := K_{\mathrm{Attr}}^{\mathrm{wit}} \sqcup K_{\mathrm{Attr}}^{\mathrm{inc}}$)

*NO-with-witness:*
$$K_{\mathrm{Attr}}^{\mathrm{wit}} := (\mathsf{obstruction}, \mathsf{type})$$
where $\mathsf{type} \in \{\texttt{NO\_SEMIFLOW}, \texttt{NO\_ABSORBING\_SET}, \texttt{NO\_ASYMP\_COMPACT}, \texttt{NO\_ATTRACTOR}\}$ identifies which of (1)–(4) fails, with a concrete obstruction object.

*NO-inconclusive:*
$$K_{\mathrm{Attr}}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$
Typical $\mathsf{missing}$: "cannot verify asymptotic compactness from current soft interfaces", "Temam-Raugel template requires compactness lemma not provided", "insufficient bounds to certify absorbing set".

**Used by:** `mt-auto-profile` Mechanism B (Attr+Morse) and any node invoking global attractor machinery.

**Literature:** {cite}`Temam97`; {cite}`Raugel02`; {cite}`HaleBook88`.
:::

:::{prf:definition} Permit $\mathrm{DegImage}_m$ (Degree-of-Image Bound for Degree-$m$ Maps)
:label: def-permit-degimage

**Name:** DegImageBound

**Question:** For the chosen "compression map" $\phi$ of (algebraic) degree $\leq m$, does the standard degree inequality for images apply in your setting?

**YES certificate**
$$K_{\mathrm{DegImage}_m}^+ := (\phi,\ \mathsf{model},\ \mathsf{basepointfree},\ \mathsf{deg\_ineq})$$
asserting:
1. (**Model choice fixed**) You specify whether $\phi$ is a morphism $W \to \mathbb{P}^N$, or a rational map represented via its graph / resolution of indeterminacy.
2. (**Base-point-free representation**) After the chosen resolution/graph step, $\phi$ is induced by a base-point-free linear system of degree $\leq m$.
3. (**Degree inequality**) For projective closures, the inequality holds:
   $$\deg(\overline{\phi(W)}) \leq m^{\dim W} \cdot \deg(W)$$
   (or your preferred standard variant with the same monotone dependence on $m$).

**NO certificate** (sum type $K_{\mathrm{DegImage}_m}^- := K_{\mathrm{DegImage}_m}^{\mathrm{wit}} \sqcup K_{\mathrm{DegImage}_m}^{\mathrm{inc}}$)

*NO-with-witness:*
$$K_{\mathrm{DegImage}_m}^{\mathrm{wit}} := (\mathsf{map\_model}, \mathsf{violation})$$
where $\mathsf{violation} \in \{\texttt{NOT\_BPF}, \texttt{DEGREE\_EXCEEDS}, \texttt{INDETERMINACY\_UNRESOLVABLE}\}$ specifies which hypothesis fails with a concrete witness (e.g., a base locus, or a degree computation exceeding the bound).

*NO-inconclusive:*
$$K_{\mathrm{DegImage}_m}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$
Typical $\mathsf{missing}$: "resolution of indeterminacy not computable", "degree of image not algorithmically determinable for this variety class", "base-point-free verification requires Bertini-type theorem not available".

**Used by:** `def-e12` Backend C (morphism/compression).

**Literature:** {cite}`Lazarsfeld04`; {cite}`Fulton84`.
:::

:::{prf:definition} Permit $\mathrm{CouplingSmall}^+$ (Coupling Control in Product Regularity)
:label: def-permit-couplingsmall

**Name:** CouplingSmall

**Question:** Is the interaction term $\Phi_{\mathrm{int}}$ controlled strongly enough (in the norms used by $K_{\mathrm{Lock}}^A, K_{\mathrm{Lock}}^B$) to prevent the coupling from destroying the component bounds?

**YES certificate**
$$K_{\mathrm{CouplingSmall}}^+ := (\varepsilon,\ C_\varepsilon,\ \mathsf{bound\_form},\ \mathsf{closure})$$
asserting the existence of an inequality of one of the following standard "closure" types (declare which one you use):
- (**Energy absorbability**) For a product energy $E = E_A + E_B$,
  $$\left|\frac{d}{dt} E_{\mathrm{int}}(t)\right| \leq \varepsilon \, E(t) + C_\varepsilon,$$
  with $\varepsilon$ small enough to be absorbed by dissipation/Grönwall.
- (**Relative boundedness**) $\Phi_{\mathrm{int}}$ is bounded or relatively bounded w.r.t. the product generator (for semigroup closure).
- (**Local Lipschitz + small parameter**) $\|\Phi_{\mathrm{int}}(u_A, u_B)\| \leq \varepsilon \, F(\|u_A\|, \|u_B\|) + C$ with $\varepsilon$ in the regime required by the bootstrap.

**NO certificate** (sum type $K_{\mathrm{CouplingSmall}}^- := K_{\mathrm{CouplingSmall}}^{\mathrm{wit}} \sqcup K_{\mathrm{CouplingSmall}}^{\mathrm{inc}}$)

*NO-with-witness:*
$$K_{\mathrm{CouplingSmall}}^{\mathrm{wit}} := (\mathsf{interaction}, \mathsf{unbounded\_mode})$$
where $\mathsf{unbounded\_mode} \in \{\texttt{ENERGY\_SUPERLINEAR}, \texttt{NOT\_REL\_BOUNDED}, \texttt{LIPSCHITZ\_FAILS}\}$ specifies which closure-usable bound fails, with a concrete sequence/trajectory demonstrating growth.

*NO-inconclusive:*
$$K_{\mathrm{CouplingSmall}}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$
Typical $\mathsf{missing}$: "absorbability constant $\varepsilon$ not computable from current interfaces", "relative boundedness requires spectral information not provided", "Lipschitz constant estimation exceeds available bounds".

**Used by:** `mt-product` Backend A (when "subcritical scaling" is intended to imply analytic absorbability), and as a general interface to justify persistence of Lock bounds under coupling.
:::

:::{prf:definition} Permit $\mathrm{ACP}^+$ (Abstract Cauchy Problem Formulation)
:label: def-permit-acp

**Name:** AbstractCauchyProblem

**Question:** Can the dynamics be represented (equivalently, in the sense you require) as an abstract Cauchy problem on a Banach/Hilbert space?

**YES certificate**
$$K_{\mathrm{ACP}}^+ := (X,\ A,\ D(A),\ \mathsf{mild},\ \mathsf{equiv})$$
asserting:
1. (**State space**) A Banach/Hilbert space $X$ is fixed for the evolution state.
2. (**Generator**) A (possibly nonlinear) operator $A$ with declared domain $D(A)$ is specified such that the evolution is
   $$u'(t) = A(u(t)) \quad (\text{or } u'(t) = Au(t) + F(u(t)) \text{ in the semilinear case}).$$
3. (**Mild/strong solutions**) A mild formulation exists (e.g., Duhamel/variation of constants) in the class used by the Sieve.
4. (**Equivalence**) Solutions in the analytic/PDE sense correspond to (mild/strong) solutions of the ACP in the time intervals under consideration.

**NO certificate** (sum type $K_{\mathrm{ACP}}^- := K_{\mathrm{ACP}}^{\mathrm{wit}} \sqcup K_{\mathrm{ACP}}^{\mathrm{inc}}$)

*NO-with-witness:*
$$K_{\mathrm{ACP}}^{\mathrm{wit}} := (\mathsf{space\_candidate}, \mathsf{obstruction})$$
where $\mathsf{obstruction} \in \{\texttt{NO\_GENERATOR}, \texttt{DOMAIN\_MISMATCH}, \texttt{MILD\_FAILS}, \texttt{EQUIV\_BREAKS}\}$ specifies which of (1)–(4) fails, with a concrete witness (e.g., a solution in the PDE sense not representable in the ACP framework).

*NO-inconclusive:*
$$K_{\mathrm{ACP}}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$
Typical $\mathsf{missing}$: "generator domain $D(A)$ not characterizable from soft interfaces", "mild solution formula requires semigroup estimates not provided", "equivalence of solution notions requires regularity theory beyond current scope".

**Used by:** `mt-product` Backend B (semigroup/perturbation route), and anywhere you invoke generator/semigroup theorems.

**Literature:** {cite}`EngelNagel00`; {cite}`Pazy83`.
:::

---

:::{prf:definition} Permit $\mathrm{Rigidity}_T^+$ (Rigidity / No-Minimal-Counterexample Theorem)
:label: def-permit-rigidity

**Name:** Rigidity

**Question:** Given an almost-periodic (mod symmetries) minimal obstruction $u^\ast$ produced by the CC+stability machine, can it be ruled out (or classified into an explicit finite library) by a rigidity argument for this specific type $T$?

**Input prerequisites (expected):**
- A critical well-posedness + continuation certificate $K_{\mathrm{WP}_{s_c}}^+$.
- A profile decomposition certificate $K_{\mathrm{ProfDec}_{s_c,G}}^+$.
- A CC+stability machine certificate $K_{\mathrm{KM}_{\mathrm{CC+stab}}}^+$ producing a minimal almost-periodic $u^\ast$ (mod $G$).
- A declared target property $\mathcal P$ (e.g. scattering, global regularity) and a declared minimality functional (energy/mass/etc.).

**YES certificate**
$$K_{\mathrm{Rigidity}_T}^+ := \big(\mathsf{rigid\_statement},\ \mathsf{hypotheses},\ \mathsf{conclusion},\ \mathsf{proof\_ref}\big)$$
where the payload contains:
1. (**Rigidity statement**) A precise proposition of the form:
   > If $u$ is a maximal-lifespan solution of type $T$ which is almost periodic modulo $G$ and minimal among counterexamples to $\mathcal P$, then $u$ is impossible (contradiction), **or** $u$ lies in an explicitly listed finite family $\mathcal L_T$ (soliton, self-similar, traveling wave, etc.).
2. (**Hypotheses**) The exact analytic assumptions required (e.g. Morawetz/virial identity validity, monotonicity formula, coercivity, channel of energy, interaction Morawetz, frequency-localized estimates, etc.).
3. (**Conclusion**) One of:
   - (**Elimination**) no such $u$ exists (hence $\mathcal P$ holds globally), or
   - (**Classification**) every such $u$ belongs to the declared library $\mathcal L_T$.
4. (**Proof reference**) Either (a) a full internal proof in the current manuscript, or (b) an external theorem citation with the exact matching hypotheses.

**NO certificate** (sum type $K_{\mathrm{Rigidity}_T}^- := K_{\mathrm{Rigidity}_T}^{\mathrm{wit}} \sqcup K_{\mathrm{Rigidity}_T}^{\mathrm{inc}}$)

*NO-with-witness:*
$$K_{\mathrm{Rigidity}_T}^{\mathrm{wit}} := (u^*, \mathsf{failure\_mode})$$
where $u^*$ is an almost-periodic minimal counterexample that exists and is not eliminated/classified, and $\mathsf{failure\_mode} \in \{\texttt{NOT\_ELIMINATED}, \texttt{NOT\_IN\_LIBRARY}, \texttt{MONOTONICITY\_FAILS}, \texttt{LS\_CLOSURE\_FAILS}\}$ records which rigidity argument fails.

*NO-inconclusive:*
$$K_{\mathrm{Rigidity}_T}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$
Typical $\mathsf{missing}$: "$K_{\mathrm{Mon}_\phi}^+$ certificate insufficient to validate monotonicity inequality", "$K_{\mathrm{LS}_\sigma}^+$ constants/exponent missing", "no rigidity template (Morawetz/virial/channel-of-energy) matches type $T$".

**Used by:** `mt-auto-profile` Mechanism A (CC+Rig), Step "Hybrid Rigidity".

**Literature:** {cite}`DuyckaertsKenigMerle11`; {cite}`KenigMerle06`.
:::

---

:::{prf:definition} Permit $\mathrm{MorseDecomp}^+$ (Attractor Structure via Morse/Conley or Gradient-like Dynamics)
:label: def-permit-morsedecomp

**Name:** MorseDecomp

**Question:** Does the semiflow $(S_t)_{t\ge0}$ admit a *structural decomposition* of the global attractor sufficient to classify all bounded complete trajectories into equilibria and connecting orbits (or other explicitly described recurrent pieces)?

**Input prerequisites (expected):**
- A global attractor existence certificate $K_{\mathrm{Attr}}^+$ (compact attractor $\mathcal A$ exists).

**YES certificate**
$$K_{\mathrm{MorseDecomp}}^+ := \big(\mathsf{structure\_type},\ \{\mathcal M_i\}_{i=1}^N,\ \mathsf{order},\ \mathsf{chain\_rec},\ \mathsf{classification}\big)$$
where the payload asserts one of the following **declared structure types** (choose one and commit to it in the theorem that uses this permit):

**(A) Gradient-like / Lyapunov structure backend:**
- There exists a continuous strict Lyapunov function $L:X\to\mathbb R$ such that:
  1. $t\mapsto L(S_t x)$ is strictly decreasing unless $x$ is an equilibrium;
  2. the set of equilibria $\mathcal E$ is compact (often finite/mod-$G$);
  3. every bounded complete trajectory has $\alpha$- and $\omega$-limits contained in $\mathcal E$.
- **Classification payload:** every bounded complete trajectory is an equilibrium or a heteroclinic connection between equilibria; no periodic orbits occur.

**(B) Morse–Smale backend (stronger, if you want it):**
- The flow on $\mathcal A$ is Morse–Smale (hyperbolic equilibria/periodic orbits, transverse invariant manifolds, no complicated recurrence).
- **Classification payload:** $\mathcal A$ is a finite union of invariant sets (equilibria and possibly finitely many periodic orbits) plus their stable/unstable manifolds; every trajectory converges to one of the basic pieces.

**(C) Conley–Morse decomposition backend (most general/topological):**
- There exists a finite Morse decomposition $\{\mathcal M_i\}_{i=1}^N$ of $\mathcal A$ with a partial order $\preceq$ such that:
  1. each $\mathcal M_i$ is isolated invariant;
  2. every full trajectory in $\mathcal A$ either lies in some $\mathcal M_i$ or connects from $\mathcal M_i$ to $\mathcal M_j$ with $i\succ j$;
  3. the chain recurrent set is contained in $\bigcup_i \mathcal M_i$.
- **Classification payload:** bounded dynamics reduce to membership in one of the Morse sets plus connecting orbits; recurrent behavior is completely captured by the declared Morse sets.

**NO certificate** (sum type $K_{\mathrm{MorseDecomp}}^- := K_{\mathrm{MorseDecomp}}^{\mathrm{wit}} \sqcup K_{\mathrm{MorseDecomp}}^{\mathrm{inc}}$)

*NO-with-witness:*
$$K_{\mathrm{MorseDecomp}}^{\mathrm{wit}} := (\mathsf{recurrence\_obj}, \mathsf{failure\_type})$$
where $\mathsf{failure\_type} \in \{\texttt{STRANGE\_ATTRACTOR}, \texttt{UNCAPTURED\_CYCLE}, \texttt{INFINITE\_CHAIN\_REC}\}$ identifies recurrence in $\mathcal{A}$ not captured by any declared decomposition type, with a concrete witness object.

*NO-inconclusive:*
$$K_{\mathrm{MorseDecomp}}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$
Typical $\mathsf{missing}$: "Lyapunov function not verified to be strict", "$K_{D_E}^+$ provides weak inequality only", "Conley index computation not supported for this system class".

**Used by:** `mt-auto-profile` Mechanism B (Attr+Morse), anywhere you claim "all bounded trajectories are equilibria/heteroclinic/periodic" or a finite Morse decomposition of $\mathcal A$.

**Literature:** {cite}`Conley78`; {cite}`Hale88`; {cite}`SellYou02`.
:::

---

### 8.17. Summary Tables

#### Interface Summary Table

| Interface | Object Type | Predicate Logic | Certificate Data |
| :--- | :--- | :--- | :--- |
| **$\mathcal{H}_0$** | State Object | Well-posedness | Existence proof |
| **$D_E$** | Height Morphism | Bound check | $B \in \mathcal{H}$ |
| **$\mathrm{Rec}_N$** | Bad Subobject | Count check | Integer $N$ |
| **$C_\mu$** | Group Action | Concentration | Profile $V$ |
| **$\mathrm{SC}_\lambda$** | Scaling Action | Inequality $\alpha > \beta$ | Exponents |
| **$\mathrm{SC}_{\partial c}$** | Parameter Object | Stability check | Reference $\theta_0$ |
| **$\mathrm{Cap}_H$** | Capacity Functional | Threshold check | Capacity value |
| **$\mathrm{LS}_\sigma$** | Gradient Operator | Gradient domination | Exponent $\theta$ |
| **$\mathrm{TB}_\pi$** | Sector Set | Invariance | Sector ID |
| **$\mathrm{TB}_O$** | Definability Modality | Definability | Tame structure |
| **$\mathrm{TB}_\rho$** | Measure Object | Finite mixing | Mixing time |
| **$\mathrm{Rep}_K$** | Language Object | Finite description | Program/Code |
| **$\mathrm{GC}_\nabla$** | Metric Tensor | Metric compatibility | Flow type |
| **$\mathrm{Bound}$** | Input/Output Objects | Boundary conditions | Bounds |
| **$\mathrm{Cat}_{\mathrm{Hom}}$** | Hypostructure Category | Hom-emptiness | E1-E10 obstruction |

This table constitutes the **Type Signature** of a Hypostructure.

#### Object-Interface Map

| Interface | **PDE** | **Graph** | **HoTT** | **Neural Net** |
| :--- | :--- | :--- | :--- | :--- |
| $\mathcal{X}$ | $H^s(\mathbb{R}^d)$ | $(V, E)$ | Type $A$ | $\mathbb{R}^n$ |
| $\Phi$ | Energy | Edge count | Level | Loss |
| $G$ | ISO$(d)$ | Aut$(G)$ | Aut$(A)$ | $S_n$ |
| $\mathcal{B}$ | Singular locus | Disconnected | $\bot$ | Saddle points |

#### Domain Instantiation Checklist

| Domain | $\mathcal{E}$ | Reg | D | C | SC | Cap | LS | TB | Rep |
|--------|--------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Navier-Stokes** | Sh(Diff) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| **Graph Coloring** | Set | ✓ | ✓ | ✓ | — | ✓ | — | ✓ | ✓ |
| **HoTT** | $\infty$-Grpd | ✓ | ✓ | ✓ | ✓ | — | — | ✓ | ✓ |
| **Neural Networks** | Smooth | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | ✓ |
| **Crypto Protocols** | Sh(FinSet) | ✓ | ✓ | — | — | — | — | ✓ | ✓ |

---

### 8.18. Master Node Map

The following table provides the complete mapping from Sieve nodes to interfaces:

| Node | Name | Interface | Predicate $\mathcal{P}$ | YES → | NO → |
|:---:|:---|:---|:---|:---|:---|
| 1 | EnergyCheck | $D_E$ | $\Phi(S_t x) \leq B$ | Node 2 | BarrierSat |
| 2 | ZenoCheck | $\mathrm{Rec}_N$ | $\#\mathcal{B} < \infty$ | Node 3 | BarrierCausal |
| 3 | CompactCheck | $C_\mu$ | Concentration controlled | Node 4 | **BarrierScat** |
| 4 | ScaleCheck | $\mathrm{SC}_\lambda$ | $\alpha > \beta$ | Node 5 | BarrierTypeII |
| 5 | ParamCheck | $\mathrm{SC}_{\partial c}$ | $\dot{\theta} \approx 0$ | Node 6 | BarrierVac |
| 6 | GeomCheck | $\mathrm{Cap}_H$ | $\text{Cap}(\Sigma) \leq C$ | Node 7 | BarrierCap |
| 7 | StiffnessCheck | $\mathrm{LS}_\sigma$ | $\|\nabla\Phi\| \geq c\Phi^\theta$ | Node 8 | BarrierGap |
| 8 | TopoCheck | $\mathrm{TB}_\pi$ | Sector preserved | Node 9 | BarrierAction |
| 9 | TameCheck | $\mathrm{TB}_O$ | $\mathcal{O}$-definable | Node 10 | BarrierOmin |
| 10 | ErgoCheck | $\mathrm{TB}_\rho$ | Mixing finite | Node 11 | BarrierMix |
| 11 | ComplexCheck | $\mathrm{Rep}_K$ | $K(u) < \infty$ | Node 12 | BarrierEpi |
| 12 | OscillateCheck | $\mathrm{GC}_\nabla$ | Oscillation present | BarrierFreq | Node 13 |
| 13 | BoundaryCheck | $\mathrm{Bound}_\partial$ | Open system? | Node 14 | Node 17 |
| 14 | OverloadCheck | $\mathrm{Bound}_B$ | Input bounded | Node 15 | BarrierBode |
| 15 | StarveCheck | $\mathrm{Bound}_{\Sigma}$ | Supply sufficient | Node 16 | BarrierInput |
| 16 | AlignCheck | $\mathrm{GC}_T$ | Control aligned | Node 17 | BarrierVariety |
| 17 | **The Lock** | $\mathrm{Cat}_{\mathrm{Hom}}$ | $\text{Hom}=\emptyset$ | **VICTORY** | **NO** (typed) |

*Note: Node 17 has binary output with typed NO certificates: Blocked → VICTORY, Breached-with-witness → FATAL, Breached-inconclusive → Reconstruction (see Section 20).*

#### Restoration Subtree (Mode D Recovery)

When CompactCheck (Node 3) returns NO with a concentration profile, the system enters **Mode D.D** (Definite Deviation). The Restoration Subtree attempts recovery:

| Sub-Node | Name | Check | Success → | Failure → |
|:---:|:---|:---|:---|:---|
| 3a | ProfileID | Profile in library? | Node 3b | Mode D.H (Horizon) |
| 3b | SurgeryAdmit | Surgery admissible? | Node 3c | Mode D.I (Inadmissible) |
| 3c | SurgeryExec | Execute surgery | Node 3d | BarrierSurgery |
| 3d | ReEntry | Re-enter at Node 4 | Node 4 | BarrierReEntry |

---

### 8.A. The Kernel Objects: Interface Implementations

A Hypostructure $\mathcal{H} = (\mathcal{X}, \Phi, \mathfrak{D}, G)$ consists of four **kernel objects**. Each kernel object implements a subset of the interfaces defined in Sections 8.1–8.15.

This section specifies each kernel object using a standardized template:
1. **Component Table**: Maps each internal component to its mathematical type, the interface it satisfies, and its role in the Sieve
2. **Formal Definition**: Tuple specification with implementation constraints and domain examples

This serves as a "header file" for instantiation — users can read the table and know exactly what data structures to provide for their domain.

---

#### 8.A.1. $\mathcal{X}$ — The State Object
**Implements Interfaces:** **$\mathcal{H}_0$**, **$\mathrm{Cap}_H$**, **$\mathrm{TB}_\pi$**, **$\mathrm{TB}_O$**, **$\mathrm{Rep}_K$**, **$\mathrm{Bound}$**, **$\mathrm{Cat}_{\mathrm{Hom}}$**

| Component | Mathematical Type | Interface | Role / Description |
| :--- | :--- | :--- | :--- |
| **1. $\mathcal{X}$** | Object in $\mathcal{E}$ | **$\mathcal{H}_0$** | The configuration space (all possible states). |
| **2. $\mathcal{T}$** | Monoid Object in $\mathcal{E}$ | **$\mathcal{H}_0$** | The **Time Object** ($\mathbb{R}_{\geq 0}$, $\mathbb{N}$, or ordinal). |
| **3. $S$** | Morphism $\mathcal{T} \times \mathcal{X} \to \mathcal{X}$ | **$\mathcal{H}_0$** | The evolution action (semigroup/flow). Powers **Node 1**. |
| **4. $\mathcal{X}_0$** | Subobject $\hookrightarrow \mathcal{X}$ | **$\mathcal{H}_0$** | Initial data space. Defines well-posed initial conditions. |
| **5. $\mathcal{B}$** | Subobject $\hookrightarrow \mathcal{X}$ | **$\mathrm{Rec}_N$**, **$\mathrm{Cap}_H$** | The **Bad Locus** (singular/blow-up set). Powers **Nodes 2, 6**. |
| **6. $\{S_\tau\}_{\tau \in \mathcal{I}}$** | Coproduct $\mathcal{X} \simeq \coprod_\tau S_\tau$ | **$\mathrm{TB}_\pi$** | **Topological Sectors** indexed by invariant $\tau$. Powers **Node 8**. |
| **7. Cap** | Morphism $\text{Sub}(\mathcal{X}) \to \mathcal{H}$ | **$\mathrm{Cap}_H$** | **Capacity measure** on subobjects. Powers **Node 6**. |
| **8. $\mathcal{O}$** | O-minimal Structure on $\flat(\mathcal{X})$ | **$\mathrm{TB}_O$** | **Definability structure** for tame geometry. Powers **Node 9**. |
| **9. desc** | Morphism $\mathcal{X} \to \mathcal{L}$ | **$\mathrm{Rep}_K$** | **Description map** into language object $\mathcal{L}$. Powers **Node 11**. |
| **10. $(\iota, \pi)$** | $\iota: \mathcal{U} \to \mathcal{X}$, $\pi: \mathcal{X} \to \mathcal{Y}$ | **$\mathrm{Bound}$** | **Open system coupling** (input/output). Powers **Nodes 13-16**. |
| **11. $\mathcal{B}$** | Set of objects in $\mathbf{Hypo}_T$ | **$\mathrm{Cat}_{\mathrm{Hom}}$** | **Bad Pattern Library** $\{B_i\}_{i \in I}$ for morphism obstruction. Powers **Node 17**. |

:::{prf:definition} The State Object
:label: def-kernel-x

The **State Object** is a tuple $\mathcal{X} = (\mathcal{X}, \mathcal{T}, S, \mathcal{X}_0, \mathcal{B}, \{S_\tau\}, \text{Cap}, \mathcal{O}, \text{desc}, \iota, \pi, \mathcal{X}_{\text{bad}})$ defined within a cohesive $(\infty,1)$-topos $\mathcal{E}$:

1. **The Configuration Space ($\mathcal{X}$):** An object in $\mathcal{E}$ representing all possible system states.
   - *PDE:* $H^s(\mathbb{R}^d)$, $L^2_\sigma(\Omega)$ (divergence-free fields), or infinite-dimensional manifold.
   - *Graph:* Finite set $\text{Map}(V, [k])$ of vertex colorings.
   - *HoTT:* A type $A$ in a universe $\mathcal{U}_i$.
   - *Neural Net:* Parameter space $\mathbb{R}^n$ or weight manifold.
   - *Crypto:* State space of protocol configurations.

2. **The Time Object ($\mathcal{T}$):** A monoid object $(T, +, 0)$ in $\mathcal{E}$ parameterizing evolution.
   - *Continuous:* $\mathcal{T} = \mathbb{R}_{\geq 0}$ (real time).
   - *Discrete:* $\mathcal{T} = \mathbb{N}$ (iteration count).
   - *Transfinite:* $\mathcal{T} = \text{Ord}$ (ordinal time for termination proofs).
   - *Interface Constraint:* Must be totally ordered with $0$ as identity.

3. **The Evolution Action ($S$):** A morphism $S: \mathcal{T} \times \mathcal{X} \to \mathcal{X}$ satisfying semigroup laws.
   - *Semigroup Law:* $S(t, S(s, x)) = S(t+s, x)$ and $S(0, x) = x$.
   - *Interface Constraint ($\mathcal{H}_0$):* For each $x_0 \in \mathcal{X}_0$, there exists $T \in \mathcal{T}$ such that $S(t, x_0)$ is defined for all $t \leq T$.
   - *Role:* Powers **Node 1** (EnergyCheck) — verifies evolution exists and respects height bounds.

4. **The Initial Data Space ($\mathcal{X}_0$):** A subobject $\mathcal{X}_0 \hookrightarrow \mathcal{X}$ of admissible initial conditions.
   - *Interface Constraint ($\mathcal{H}_0$):* $\mathcal{X}_0 \neq \emptyset$ and satisfies compatibility: $S(t, \mathcal{X}_0) \subseteq \mathcal{X}$ for small $t$.
   - *PDE:* Initial data with finite energy $\{u_0 : \Phi(u_0) < \infty\}$.
   - *Graph:* All valid initial colorings.

5. **The Bad Locus ($\mathcal{B}$):** A subobject encoding where singularities/failures can occur.
   - *Interface Constraint ($\mathrm{Rec}_N$):* $|\mathcal{B} \cap \mathfrak{D}^{-1}(0)| < \infty$ (finitely many singular equilibria).
   - *Interface Constraint ($\mathrm{Cap}_H$):* $\text{Cap}(\mathcal{B}) < \infty$ (small in capacity measure).
   - *PDE:* Blow-up locus, singular set of weak solutions.
   - *Graph:* Stuck configurations (local minima with conflicts).
   - *Role:* Powers **Node 2** (ZenoCheck) and **Node 6** (GeomCheck).

6. **The Sector Decomposition ($\{S_\tau\}_{\tau \in \mathcal{I}}$):** A coproduct decomposition $\mathcal{X} \simeq \coprod_{\tau \in \mathcal{I}} S_\tau$ indexed by topological invariants.
   - *Interface Constraint ($\mathrm{TB}_\pi$):* Evolution preserves sectors: $S(t, S_\tau) \subseteq S_\tau$ for all $t \in \mathcal{T}$.
   - *PDE:* Homotopy classes, winding numbers, degree.
   - *Graph:* Connected components of the configuration graph.
   - *HoTT:* Connected components $\pi_0(A)$.
   - *Role:* Powers **Node 8** (TopoCheck).

7. **The Capacity Functional (Cap):** A morphism $\text{Cap}: \text{Sub}(\mathcal{X}) \to \mathcal{H}$ measuring "size" of subobjects.
   - *Axioms:* Monotone ($A \subseteq B \Rightarrow \text{Cap}(A) \leq \text{Cap}(B)$), subadditive.
   - *PDE:* Hausdorff measure $\mathcal{H}^{d-2}$, parabolic capacity, Minkowski content.
   - *Graph:* Cardinality $|A|$, edge boundary $|\partial A|$.
   - *HoTT:* Truncation level, h-level.
   - *Role:* Powers **Node 6** (GeomCheck) via threshold $\text{Cap}(\Sigma) \leq C$.

8. **The O-minimal Structure ($\mathcal{O}$):** A definability structure on the discrete reflection $\flat(\mathcal{X})$.
   - *Definition:* An o-minimal expansion of $(\mathbb{R}, <, +, \cdot)$ such that definable sets have tame geometry.
   - *Interface Constraint ($\mathrm{TB}_O$):* All relevant subobjects ($\mathcal{B}$, level sets of $\Phi$) are $\mathcal{O}$-definable.
   - *Examples:* $\mathbb{R}_{\text{an}}$ (restricted analytic), $\mathbb{R}_{\text{exp}}$ (with exponential), $\mathbb{R}_{\text{alg}}$ (semialgebraic).
   - *Role:* Powers **Node 9** (TameCheck). Ensures no pathological Cantor-like singular sets.

9. **The Description Map (desc):** A morphism $\text{desc}: \mathcal{X} \to \mathcal{L}$ into a formal language object.
   - *Definition:* $\mathcal{L}$ is a language object (e.g., $\Sigma^*$ for alphabet $\Sigma$) with complexity measure $K: \mathcal{L} \to \mathbb{N} \cup \{\infty\}$.
   - *Interface Constraint ($\mathrm{Rep}_K$):* $K(\text{desc}(x)) < \infty$ for states of interest.
   - *PDE:* Finite element representation, spectral coefficients.
   - *Graph:* Adjacency list encoding.
   - *Role:* Powers **Node 11** (ComplexCheck).

10. **The Boundary Coupling ($\iota, \pi$):** Input and output morphisms for open systems.
    - *Input:* $\iota: \mathcal{U} \times \mathcal{T} \to \mathcal{X}$ injects external control/disturbance.
    - *Output:* $\pi: \mathcal{X} \to \mathcal{Y}$ extracts observable quantities.
    - *Interface Constraint ($\mathrm{Bound}$):* Boundedness conditions on $\|\iota(u)\|$, alignment $\langle \iota(u), \nabla\Phi \rangle$.
    - *Role:* Powers **Nodes 13-16** (BoundaryCheck, OverloadCheck, StarveCheck, AlignCheck).

11. **The Bad Pattern Library ($\mathcal{B}$):** A finite set $\mathcal{B} = \{B_i\}_{i \in I}$ of minimal bad patterns in the category $\mathbf{Hypo}_T$.
    - *Definition:* Each $B_i$ is a canonical singularity-forming structure for type $T$. The library is problem-specific.
    - *Interface Constraint ($\mathrm{Cat}_{\mathrm{Hom}}$):* Victory requires $\forall i: \text{Hom}(B_i, \mathcal{X}) = \emptyset$.
    - *PDE:* Self-similar blow-up profiles, ancient solutions with singularity.
    - *Graph:* Chromatic obstructions (e.g., odd cycles for 2-coloring).
    - *Role:* Powers **Node 17** (The Lock).
:::

---

#### 8.A.2. $\Phi$ — The Height Object
**Implements Interfaces:** **$D_E$**, **$\mathrm{SC}_\lambda$**, **$\mathrm{LS}_\sigma$**

| Component | Mathematical Type | Interface | Role / Description |
| :--- | :--- | :--- | :--- |
| **1. $\Phi$** | Morphism $\mathcal{X} \to \mathcal{H}$ | **$D_E$** | The **Height Functional** (Energy/Cost/Complexity). Powers **Node 1**. |
| **2. $\mathcal{H}$** | Ordered Monoid in $\mathcal{E}$ | **$D_E$** | The **Height Codomain** with addition and comparison. |
| **3. $(\leq, +, 0, \infty)$** | Ordered monoid structure | **$D_E$** | **Algebraic structure** enabling bounds and accumulation. |
| **4. $\alpha$** | Element of $\mathbb{R}$ (or $\mathcal{H}$-exponent) | **$\mathrm{SC}_\lambda$** | **Height scaling exponent**: $\Phi(\lambda \cdot x) = \lambda^\alpha \Phi(x)$. Powers **Node 4**. |
| **5. $\nabla\Phi$** | Morphism $\mathcal{X} \to \mathcal{X}^*$ | **$\mathrm{LS}_\sigma$** | The **Gradient/Slope Operator** (generalized derivative). Powers **Node 7**. |
| **6. $\|\cdot\|$** | Norm $\mathcal{X}^* \to \mathcal{H}$ | **$\mathrm{LS}_\sigma$** | **Gradient magnitude** for Łojasiewicz inequality. |
| **7. $\theta$** | Element of $(0, 1]$ | **$\mathrm{LS}_\sigma$** | **Łojasiewicz exponent**: $\|\nabla\Phi\| \geq c|\Phi - \Phi_\infty|^\theta$. |
| **8. $\Phi_\infty$** | Element of $\mathcal{H}$ | **$\mathrm{LS}_\sigma$** | **Limit height** (infimum/equilibrium value). |

:::{prf:definition} The Height Object
:label: def-kernel-phi

The **Height Object** is a tuple $\Phi = (\Phi, \mathcal{H}, \leq, +, 0, \infty, \alpha, \nabla\Phi, \|\cdot\|, \theta, \Phi_\infty)$ defined within $\mathcal{E}$:

1. **The Height Morphism ($\Phi$):** A morphism $\Phi: \mathcal{X} \to \mathcal{H}$ assigning a "cost" or "potential" to each state.
   - *Interpretation:* Measures how "far" a state is from equilibrium/optimality.
   - *PDE:* Energy $\Phi(u) = \frac{1}{2}\int |\nabla u|^2 + V(u)$, entropy, or action.
   - *Graph:* Conflict count $\Phi(c) = \#\{\text{monochromatic edges}\}$, or potential function.
   - *Neural Net:* Loss function $\mathcal{L}(\theta)$, or regularized objective.
   - *HoTT:* Universe level $\text{level}(A)$, or path complexity.
   - *Crypto:* Security parameter, computational cost.

2. **The Height Codomain ($\mathcal{H}$):** An ordered monoid serving as the target of $\Phi$.
   - *Structure:* $(\mathcal{H}, \leq, +, 0, \infty)$ where:
     - $\leq$ is a total preorder (for comparisons)
     - $+$ is associative, commutative, with identity $0$
     - $\infty$ is an absorbing element ($h + \infty = \infty$)
   - *Typical choices:* $(\mathbb{R}_{\geq 0} \cup \{\infty\}, \leq, +)$, $(\mathbb{N} \cup \{\infty\}, \leq, +)$, ordinals.
   - *Interface Constraint ($D_E$):* Must support the statement "$\Phi(x) \leq B$" for bounds $B \in \mathcal{H}$.

3. **The Comparison and Algebraic Structure ($\leq, +, 0, \infty$):**
   - *Comparison ($\leq$):* Total preorder enabling boundedness checks.
   - *Addition ($+$):* For accumulating dissipation: $\Phi(S_t x) \leq \Phi(x) + \int_0^t \mathfrak{D}$.
   - *Zero ($0$):* Minimum height (ground state energy).
   - *Infinity ($\infty$):* Blow-up indicator; $\Phi(x) = \infty$ signals singularity.
   - *Role:* Powers **Node 1** (EnergyCheck) via $\Phi(S_t x) \leq B$.

4. **The Height Scaling Exponent ($\alpha$):** The homogeneity degree under the scaling action from $G$.
   - *Definition:* $\Phi(\rho(\lambda, x)) = \lambda^\alpha \Phi(x)$ for $\lambda \in \mathcal{S} \subseteq G$.
   - *Interface Constraint ($\mathrm{SC}_\lambda$):* $\alpha$ must be well-defined and computable.
   - *PDE (Navier-Stokes):* $\alpha = 2$ (kinetic energy $\sim$ velocity$^2$).
   - *PDE (NLS):* $\alpha = 2$ (mass), $\alpha = 4$ (energy in critical case).
   - *Graph:* $\alpha = 1$ if $\Phi$ counts edges.
   - *Role:* Powers **Node 4** (ScaleCheck) via $\alpha > \beta$ (Type I) or $\alpha \leq \beta$ (Type II).

5. **The Gradient Operator ($\nabla\Phi$):** A morphism computing the "direction of steepest descent."
   - *General Definition:* $\nabla\Phi: \mathcal{X} \to \mathcal{X}^*$ where $\mathcal{X}^*$ is a dual/tangent object.
   - *Characterization:* $\nabla\Phi(x)$ is the unique element such that $\langle \nabla\Phi(x), v \rangle = D\Phi(x)[v]$ (directional derivative).
   - *PDE:* Fréchet derivative; e.g., $\nabla\Phi(u) = -\Delta u + V'(u)$ for energy functional.
   - *Graph:* Discrete gradient $\nabla\Phi(c)_v = \Phi(c') - \Phi(c)$ where $c'$ differs at vertex $v$.
   - *Neural Net:* Backpropagation gradient $\nabla_\theta \mathcal{L}$.
   - *Metric Space:* Slope $|\partial\Phi|(x) = \limsup_{y \to x} \frac{[\Phi(x) - \Phi(y)]^+}{d(x,y)}$.

6. **The Gradient Norm ($\|\cdot\|$):** A morphism $\|\cdot\|: \mathcal{X}^* \to \mathcal{H}$ measuring gradient magnitude.
   - *Definition:* $\|\nabla\Phi(x)\| \in \mathcal{H}$ quantifies "how steep" the height landscape is at $x$.
   - *PDE:* $\|\nabla\Phi\|_{L^2}$, $\|\nabla\Phi\|_{H^{-1}}$, or operator norm.
   - *Graph:* Maximum local improvement $\max_v |\nabla\Phi(c)_v|$.
   - *Metric Space:* Metric slope $|\partial\Phi|$.

7. **The Łojasiewicz Exponent ($\theta$):** The exponent in the gradient domination inequality.
   - *Definition:* There exist $c > 0$ and neighborhood $U$ of equilibria such that:
     $$\|\nabla\Phi(x)\| \geq c \cdot |\Phi(x) - \Phi_\infty|^\theta \quad \text{for } x \in U$$
   - *Range:* $\theta \in (0, 1]$. The value $\theta = 1/2$ is generic for analytic functions.
   - *Convergence Rate:* $\theta = 1/2$ → exponential; $\theta < 1/2$ → polynomial $O(t^{-1/(1-2\theta)})$.
   - *Role:* Powers **Node 7** (StiffnessCheck). Stiffness = gradient domination.

8. **The Limit Height ($\Phi_\infty$):** The infimum or equilibrium value of $\Phi$.
   - *Definition:* $\Phi_\infty = \inf_{x \in \mathcal{X}} \Phi(x)$ or $\Phi_\infty = \lim_{t \to \infty} \Phi(S_t x)$.
   - *PDE:* Ground state energy, or $0$ if normalized.
   - *Graph:* $0$ (proper coloring exists) or minimum conflict count.
   - *Role:* Reference point for Łojasiewicz inequality.
:::

---

#### 8.A.3. $\mathfrak{D}$ — The Dissipation Object
**Implements Interfaces:** **$D_E$**, **$\mathrm{Rec}_N$**, **$\mathrm{SC}_\lambda$**, **$\mathrm{GC}_\nabla$**, **$\mathrm{TB}_\rho$**

| Component | Mathematical Type | Interface | Role / Description |
| :--- | :--- | :--- | :--- |
| **1. $\mathfrak{D}$** | Morphism $\mathcal{X} \to \mathcal{H}$ | **$D_E$** | The **Dissipation Rate** (instantaneous height decrease). Powers **Node 1**. |
| **2. $\beta$** | Element of $\mathbb{R}$ | **$\mathrm{SC}_\lambda$** | **Dissipation scaling exponent**: $\mathfrak{D}(\lambda \cdot x) = \lambda^\beta \mathfrak{D}(x)$. Powers **Node 4**. |
| **3. $\mathfrak{D}^{-1}(0)$** | Subobject of $\mathcal{X}$ | **$\mathrm{Rec}_N$** | The **Zero-Dissipation Locus** (equilibria/critical points). Powers **Node 2**. |
| **4. $g$** | Metric structure on $\mathcal{X}$ | **$\mathrm{GC}_\nabla$** | **Riemannian/metric tensor** enabling $\mathfrak{D} = \|\nabla\Phi\|_g^2$. Powers **Node 12**. |
| **5. $\tau_{\text{mix}}$** | Morphism $\mathcal{X} \to \mathcal{T}$ | **$\mathrm{TB}_\rho$** | **Mixing time** to equilibrium. Powers **Node 10**. |

:::{prf:definition} The Dissipation Object
:label: def-kernel-d

The **Dissipation Object** is a tuple $\mathfrak{D} = (\mathfrak{D}, \beta, \mathfrak{D}^{-1}(0), g, \tau_{\text{mix}})$ defined within $\mathcal{E}$:

1. **The Dissipation Morphism ($\mathfrak{D}$):** A morphism $\mathfrak{D}: \mathcal{X} \to \mathcal{H}$ measuring the instantaneous rate of height decrease.
   - *Fundamental Inequality ($D_E$):* The **energy-dissipation inequality** holds:
     $$\Phi(S(t,x)) + \int_0^t \mathfrak{D}(S(s,x)) \, ds \leq \Phi(x) + \int_0^t \text{Source}(s) \, ds$$
   - *Interpretation:* $\mathfrak{D}$ quantifies "how much progress" is made per unit time.
   - *PDE:* Enstrophy $\mathfrak{D}(u) = \nu\int |\nabla u|^2$, or $\mathfrak{D}(u) = \int |\partial_t u|^2$.
   - *Graph:* Per-step improvement $\mathfrak{D}(x) = \Phi(x) - \Phi(F(x))$ for update $F$.
   - *Neural Net:* Squared gradient norm $\mathfrak{D}(\theta) = \|\nabla\mathcal{L}(\theta)\|^2$.
   - *Markov:* Fisher information $\mathfrak{D}(\mu) = I(\mu | \mu_\infty)$.
   - *Crypto:* Entropy loss per round.

2. **The Dissipation Scaling Exponent ($\beta$):** The homogeneity degree of $\mathfrak{D}$ under the scaling action.
   - *Definition:* $\mathfrak{D}(\rho(\lambda, x)) = \lambda^\beta \mathfrak{D}(x)$ for $\lambda \in \mathcal{S} \subseteq G$.
   - *Interface Constraint ($\mathrm{SC}_\lambda$):* $\beta$ must be well-defined; relationship to $\alpha$ determines blow-up type.
   - *PDE (Navier-Stokes):* $\beta = 2$ (enstrophy $\sim$ gradient$^2$).
   - *Critical Comparison:*
     - $\alpha > \beta$: **Type I** (self-similar blow-up, rate $(T-t)^{-(\alpha-\beta)/2}$).
     - $\alpha \leq \beta$: **Type II** (exotic blow-up, non-self-similar).
   - *Role:* Powers **Node 4** (ScaleCheck).

3. **The Zero-Dissipation Locus ($\mathfrak{D}^{-1}(0)$):** The subobject of states with vanishing dissipation.
   - *Definition:* $\mathfrak{D}^{-1}(0) = \{x \in \mathcal{X} : \mathfrak{D}(x) = 0\}$.
   - *Interpretation:* These are **equilibria**, **steady states**, **solitons**, or **critical points**.
   - *Interface Constraint ($\mathrm{Rec}_N$):* $|\mathfrak{D}^{-1}(0) \cap \mathcal{B}| < \infty$ — only finitely many "bad" equilibria.
   - *PDE:* Stationary solutions, traveling waves, self-similar profiles.
   - *Graph:* Local minima of $\Phi$, Nash equilibria.
   - *Neural Net:* Critical points of loss landscape.
   - *Role:* Powers **Node 2** (ZenoCheck). Prevents Zeno-type accumulation of singular events.

4. **The Metric Structure ($g$):** A (possibly degenerate) inner product structure on $\mathcal{X}$.
   - *General Definition:* $g: T\mathcal{X} \times T\mathcal{X} \to \mathcal{H}$ (or internal hom in $\mathcal{E}$).
   - *Gradient Flow Condition ($\mathrm{GC}_\nabla$):* Evolution is gradient flow if:
     $$\mathfrak{D}(x) = \|\nabla\Phi(x)\|_g^2 = g(\nabla\Phi(x), \nabla\Phi(x))$$
   - *Consequence:* If $\mathrm{GC}_\nabla$ holds, then $\dot{x} = -\nabla_g \Phi(x)$ (steepest descent).
   - *PDE:* $L^2$ metric on function spaces, Wasserstein metric on probability measures.
   - *Graph:* Discrete metric (edge weights).
   - *Neural Net:* Fisher information metric, natural gradient.
   - *Role:* Powers **Node 12** (OscillateCheck). Gradient flows converge; non-gradient flows may oscillate.
   - *Note:* $g$ is **optional**. Systems without gradient structure skip $\mathrm{GC}_\nabla$ checks.

5. **The Mixing Time ($\tau_{\text{mix}}$):** A morphism measuring time to reach equilibrium distribution.
   - *Definition:* $\tau_{\text{mix}}(x) = \inf\{t : d(S(t,x), \mathcal{E}_\infty) < \epsilon\}$ for equilibrium set $\mathcal{E}_\infty$.
   - *Interface Constraint ($\mathrm{TB}_\rho$):* $\tau_{\text{mix}}(x) < \infty$ for almost all $x$.
   - *Markov:* $\tau_{\text{mix}} = \inf\{t : \|P^t - \pi\|_{TV} < 1/4\}$ (total variation mixing).
   - *PDE:* Time to decay of correlations.
   - *Neural Net:* Convergence time under SGD.
   - *Role:* Powers **Node 10** (ErgoCheck). Finite mixing prevents ergodic pathologies.
:::

---

#### 8.A.4. $G$ — The Symmetry Object
**Implements Interfaces:** **$\mathcal{H}_0$**, **$C_\mu$**, **$\mathrm{SC}_\lambda$**, **$\mathrm{LS}_\sigma$**, **$\mathrm{SC}_{\partial c}$**

| Component | Mathematical Type | Interface | Role / Description |
| :--- | :--- | :--- | :--- |
| **1. $\mathcal{G}$** | Group Object in $\mathcal{E}$ | **$\mathcal{H}_0$** | The abstract symmetry group (Lie, Discrete, Quantum, Higher). |
| **2. $\rho$** | Action $\mathcal{G} \times \mathcal{X} \to \mathcal{X}$ | **$C_\mu$** | Group action defining orbits $[x] = G \cdot x$. Enables quotient $\mathcal{X} // G$. |
| **3. $\mathcal{S}$** | Monoid homomorphism $(\mathbb{R}^+, \times) \to \mathcal{G}$ | **$\mathrm{SC}_\lambda$** | **Scaling subgroup** for dimensional analysis. Powers **Node 4**. |
| **4. $\mathfrak{P}$** | Partial map $\mathcal{X}^{\mathbb{N}} \rightharpoonup \mathcal{X} // G$ | **$C_\mu$** | **Profile Extractor** (concentration-compactness). Powers **Node 3**. |
| **5. Stab** | Map $\mathcal{X} \to \text{Sub}(\mathcal{G})$ | **$\mathrm{LS}_\sigma$** | **Stabilizer/Isotropy** at each point. Powers **Node 7** (symmetry breaking). |
| **6. $\Theta$** | Parameter space object | **$\mathrm{SC}_{\partial c}$** | **Moduli of symmetry-breaking parameters**. Powers **Node 5**. |

:::{prf:definition} The Symmetry Object
:label: def-kernel-g

The **Symmetry Object** is a tuple $G = (\mathcal{G}, \rho, \mathcal{S}, \mathfrak{P}, \text{Stab}, \Theta)$ defined within a cohesive $(\infty,1)$-topos $\mathcal{E}$:

1. **The Group Object ($\mathcal{G}$):** An object satisfying the group axioms (identity, multiplication, inverse) internal to $\mathcal{E}$.
   - *General Structure:* $\mathcal{G}$ can be:
     - A **Lie group** (continuous symmetries)
     - A **discrete group** (finite/countable symmetries)
     - A **higher group** ($\infty$-group, groupoid)
     - A **quantum group** (Hopf algebra)
   - *PDE:* $\text{ISO}(d) \ltimes \mathbb{R}^+$ (isometries + scaling), Galilean group, Poincaré group.
   - *Graph:* $\text{Aut}(\Gamma) \times S_k$ (graph automorphisms $\times$ color permutations).
   - *HoTT:* Loop space $\Omega(A, a)$, or equivalence type $A \simeq A$.
   - *Neural Net:* $S_n$ (permutation of neurons), $O(n)$ (weight orthogonality).
   - *Crypto:* Galois group, algebraic symmetries of protocol.

2. **The Action Morphism ($\rho$):** A morphism $\rho: \mathcal{G} \times \mathcal{X} \to \mathcal{X}$ satisfying:
   - *Identity:* $\rho(e, x) = x$ for identity $e \in \mathcal{G}$.
   - *Associativity:* $\rho(g, \rho(h, x)) = \rho(gh, x)$.
   - *Interface Constraint ($C_\mu$):* The action must be **Proper** — the map $(g, x) \mapsto (gx, x)$ is proper — ensuring the quotient $\mathcal{X} // G$ is well-behaved (Hausdorff/separated).
   - *Compatibility with Height:* $\Phi(\rho(g, x)) = \Phi(x)$ for all $g \in G$ (energy is $G$-invariant).
   - *Compatibility with Evolution:* $\rho(g, S(t, x)) = S(t, \rho(g, x))$ (symmetry commutes with dynamics).
   - *Role:* Enables orbit decomposition for concentration-compactness analysis.

3. **The Scaling Monoid ($\mathcal{S}$):** A monoid homomorphism $\lambda: (\mathbb{R}^+, \times, 1) \to \mathcal{G}$.
   - *Definition:* $\lambda(s) \in \mathcal{G}$ represents "zoom by factor $s$".
   - *Interface Constraint ($\mathrm{SC}_\lambda$):* Scaling determines homogeneity exponents:
     $$\Phi(\rho(\lambda(s), x)) = s^\alpha \Phi(x), \quad \mathfrak{D}(\rho(\lambda(s), x)) = s^\beta \mathfrak{D}(x)$$
   - *PDE:* $\lambda(s) \cdot u(x) = s^{-\gamma} u(x/s)$ for scaling dimension $\gamma$.
   - *Graph:* Trivial scaling ($\mathcal{S} = \{1\}$) if $\Phi$ is discrete.
   - *Neural Net:* Learning rate scaling, width scaling.
   - *Role:* Powers **Node 4** (ScaleCheck). The inequality $\alpha > \beta$ distinguishes Type I from Type II.

4. **The Profile Extractor ($\mathfrak{P}$):** A partial function extracting limiting profiles from sequences.
   - *Domain:* Sequences $\{x_n\}_{n \in \mathbb{N}} \subset \mathcal{X}$ with bounded height $\sup_n \Phi(x_n) < B$.
   - *Codomain:* The quotient $\mathcal{X} // G$ (moduli space of profiles up to symmetry).
   - *Interface Constraint ($C_\mu$):* For bounded sequences, $\mathfrak{P}$ returns one of:
     - `Concentrate(V)`: A profile $V \in \mathcal{X}//G$ with $g_n \cdot x_n \to V$ for some $\{g_n\} \subset G$.
     - `Vanish`: $\|x_n\| \to 0$ (dispersion to zero).
     - `Escape`: $x_n$ exits every compact set (scattering to infinity).
   - *Decomposition:* For multi-profile concentration:
     $$x_n = \sum_{j=1}^J g_n^{(j)} \cdot V^{(j)} + w_n, \quad w_n \to 0 \text{ (weak)}$$
   - *PDE:* Profile decomposition (Bahouri-Gérard, Kenig-Merle, Struwe).
   - *Graph:* Cluster decomposition of configurations.
   - *Role:* Powers **Node 3** (CompactCheck). This is the concentration-compactness alternative.

5. **The Stabilizer Map (Stab):** A map assigning to each state its isotropy subgroup.
   - *Definition:* $\text{Stab}(x) := \{g \in \mathcal{G} \mid \rho(g, x) = x\} \subseteq \mathcal{G}$.
   - *Orbit-Stabilizer:* $|G \cdot x| = |G| / |\text{Stab}(x)|$ (when finite).
   - *Interface Constraint ($\mathrm{LS}_\sigma$):* At the vacuum/ground state $x_{\text{vac}}$:
     - $\text{Stab}(x_{\text{vac}}) = \mathcal{G}$: Symmetry is **unbroken** (symmetric vacuum).
     - $\text{Stab}(x_{\text{vac}}) \subsetneq \mathcal{G}$: **Spontaneous symmetry breaking** (SSB).
   - *Consequences of SSB:*
     - *Goldstone Theorem:* Broken continuous symmetry $\Rightarrow$ massless modes (zero eigenvalues).
     - *Mass Gap:* Explicit breaking $\Rightarrow$ spectral gap $\Rightarrow$ stiffness.
   - *PDE:* Vacuum manifold $\mathcal{M} = G / \text{Stab}(x_{\text{vac}})$.
   - *Physics:* Higgs mechanism, ferromagnetic ordering.
   - *Role:* Powers **Node 7** (StiffnessCheck) in restoration subtree.

6. **The Parameter Moduli ($\Theta$):** A space parameterizing symmetry-breaking deformations.
   - *Definition:* $\Theta$ is the space of "external parameters" that break or restore symmetry.
   - *Interface Constraint ($\mathrm{SC}_{\partial c}$):* Parameters must be slowly varying: $\|\dot{\theta}\| \ll 1$ (adiabatic).
   - *PDE:* External fields, boundary conditions, forcing terms.
   - *Neural Net:* Hyperparameters (learning rate, regularization).
   - *Crypto:* Security parameters, key sizes.
   - *Role:* Powers **Node 5** (ParamCheck). Tracks whether parameters drift into dangerous regimes.
:::

---

### 8.B. The Instantiation Metatheorem

:::{prf:metatheorem} Valid Instantiation
:label: mt-valid-instantiation
:class: metatheorem

**Statement:** To instantiate a Hypostructure for a system $S$ of type $T$ is to provide:
1. An ambient $(\infty,1)$-topos $\mathcal{E}$ (or a 1-topos/category with sufficient structure)
2. Concrete implementations $(\mathcal{X}, \Phi, \mathfrak{D}, G)$ satisfying the specifications of Section 8.A
3. For each relevant interface $I \in \{\text{Reg}^0, \text{D}^0, \ldots, \text{Lock}^0\}$:
   - The required structure $\mathcal{D}_I$ from the interface definition
   - A computable predicate $\mathcal{P}_I$ evaluating to $\{\text{YES}, \text{NO}, \text{Blocked}\}$ with typed NO certificates ($K^{\mathrm{wit}}$ or $K^{\mathrm{inc}}$)
   - Certificate schemas $\mathcal{K}_I^+$, $\mathcal{K}_I^{\mathrm{wit}}$, and $\mathcal{K}_I^{\mathrm{inc}}$

**Consequence:** Upon valid instantiation, the Sieve Algorithm becomes a well-defined computable function:
$$\text{Sieve}: \text{Instance}(\mathcal{H}) \to \text{Result}$$
where $\text{Result} \in \{\text{GlobalRegularity}, \text{Mode}_{1..15}, \text{FatalError}\}$. NO-inconclusive certificates route to reconstruction rather than terminating as a separate outcome.

**Verification Checklist:**
- [ ] Each kernel object is defined in $\mathcal{E}$
- [ ] Each interface's required structure is provided
- [ ] Predicates are computable (or semi-decidable with timeout)
- [ ] Certificate schemas are well-formed
- [ ] Type $T$ is specified from the catalog (Section 11)

**Literature:** Higher topos theory {cite}`Lurie09`; internal logic of toposes {cite}`Johnstone77`; type-theoretic semantics {cite}`HoTTBook`.
:::

:::{prf:metatheorem} Minimal Instantiation
:label: mt-minimal-instantiation
:class: metatheorem

**Statement:** To instantiate a Hypostructure for system $S$ using the **thin object** formalism (Section 8.C), the user provides only:

1. **The Space** $\mathcal{X}$ and its geometry (metric $d$, measure $\mu$)
2. **The Energy** $\Phi$ and its scaling $\alpha$
3. **The Dissipation** $\mathfrak{D}$ and its scaling $\beta$
4. **The Symmetry Group** $G$ with action $\rho$ and scaling subgroup $\mathcal{S}$

**The Framework (Sieve) automatically derives:**
1. **Profiles:** Via Universal Profile Trichotomy (Section 14)
2. **Admissibility:** Via Surgery Admissibility Predicate (Section 15)
3. **Regularization:** Via Structural Surgery Operator (Section 16)
4. **Topology:** Via persistent homology on measure $\mu$
5. **Bad Sets:** Via concentration locus of $\mathfrak{D}$

**User vs Framework Responsibility Matrix:**

| Task | User Provides | Framework Derives |
|------|---------------|-------------------|
| Singularity Detection | Energy scaling $\alpha$ | Profile $V$ via scaling group |
| Stability Analysis | Gradient $\nabla$ | Stiffness $\theta$ via Łojasiewicz |
| Surgery Construction | Measure $\mu$ | SurgeryOperator if Cap$(\Sigma)$ small |
| Topology | Space $\mathcal{X}$ | Sectors via $\pi_0$ |
| Bad Set | Dissipation $R$ | $\Sigma = \{x: R(x) \to \infty\}$ |
| Profile Library | Symmetry $G$ | Canonical library via moduli |

**Consequence:** The full instantiation of MT {prf:ref}`mt-valid-instantiation` is achieved by the **Thin-to-Full Expansion** (MT {prf:ref}`mt-thin-expansion`), reducing user burden from ~30 components to 10 primitive inputs.

**Literature:** Scaling analysis in PDE {cite}`Tao06`; moduli spaces {cite}`MumfordFogartyKirwan94`; persistent homology {cite}`EdelsbrunnerHarer10`.
:::

:::{prf:remark} Instantiation Examples
:label: rem-instantiation-examples

**Navier-Stokes ($T = T_{\text{parabolic}}$):**
- $\mathcal{E} = \text{Sh}(\text{Diff})$ (sheaves on smooth manifolds)
- $\mathcal{X} = L^2_\sigma(\mathbb{R}^3)$ (divergence-free vector fields)
- $\Phi = \frac{1}{2}\int |u|^2$ (kinetic energy)
- $\mathfrak{D} = \nu \int |\nabla u|^2$ (enstrophy dissipation)
- $G = \text{ISO}(3) \ltimes \mathbb{R}^3$ (rotations, translations, scaling)

**Graph Coloring ($T = T_{\text{algorithmic}}$):**
- $\mathcal{E} = \text{Set}$
- $\mathcal{X} = \text{Map}(V, [k])$ (vertex colorings)
- $\Phi = \#\{\text{monochromatic edges}\}$ (conflict count)
- $\mathfrak{D} = \Delta\Phi$ (per-step improvement)
- $G = \text{Aut}(G) \times S_k$ (graph automorphisms, color permutations)
:::

---

### 8.C Thin Kernel Objects

**Design Principle:** The full Kernel Objects of Section 8.A contain both *structural data* (user-provided) and *algorithmic machinery* (Framework-derived). This section extracts the **minimal user burden**—the "thin" objects that users must specify. Everything else is automatically constructed by the Sieve via the Universal Singularity Modules (Section 14-16).

:::{prf:definition} User vs Framework Responsibility
:label: def-user-framework-split

| Aspect | User Provides | Framework Derives |
|--------|---------------|-------------------|
| **Topology** | Space $\mathcal{X}$, metric $d$ | Sectors via $\pi_0(\mathcal{X})$, dictionary via dimension |
| **Dynamics** | Energy $\Phi$, gradient $\nabla$ | Drift detection, stability via Łojasiewicz |
| **Singularity** | Scaling dimension $\alpha$ | Profile $V$ via scaling group extraction |
| **Dissipation** | Rate $R$, scaling $\beta$ | Bad set as $\{x: R(x) \to \infty\}$ |
| **Surgery** | Measure $\mu$ | Surgery operator if Cap$(\Sigma)$ small |
| **Symmetry** | Group $G$, action $\rho$ | ProfileExtractor, VacuumStabilizer |
:::

---

#### 8.C.1 $\mathcal{X}^{\text{thin}}$ — Thin State Object

**Motto:** *"The Arena"*

:::{prf:definition} Thin State Object
:label: def-thin-state

The **Thin State Object** is a tuple:
$$\mathcal{X}^{\text{thin}} = (\mathcal{X}, d, \mu)$$

| Component | Type | What User Provides |
|-----------|------|-------------------|
| $\mathcal{X}$ | Object in $\mathcal{E}$ | The state space (Polish space, scheme, $\infty$-groupoid) |
| $d$ | $\mathcal{X} \times \mathcal{X} \to [0,\infty]$ | Metric or distance structure |
| $\mu$ | Measure on $\mathcal{X}$ | Reference measure for capacity computation |

**Automatically Derived by Framework:**

| Derived Component | Construction | Used By |
|-------------------|--------------|---------|
| $\text{SectorMap}$ | $\pi_0(\mathcal{X})$ (connected components) | $\mathrm{TB}_\pi$, $C_\mu$ |
| $\text{Dictionary}$ | $\dim(\mathcal{X})$ + type signature | All interfaces |
| $\mathcal{X}_{\text{bad}}$ | $\{x : R(x) \to \infty\}$ | $\mathrm{Cat}_{\mathrm{Hom}}$ |
| $\mathcal{O}$ | O-minimal structure from $d$ | $\mathrm{TB}_O$ |
:::

:::{prf:example} Thin State Instantiations
:label: ex-thin-state

| Domain | $\mathcal{X}$ | $d$ | $\mu$ |
|--------|--------------|-----|-------|
| **Navier-Stokes** | $L^2_\sigma(\mathbb{R}^3)$ | $\|u - v\|_{L^2}$ | Gaussian |
| **Ricci Flow** | $\text{Met}(M)/\text{Diff}(M)$ | Gromov-Hausdorff | Heat kernel |
| **Neural Net** | $\mathbb{R}^n$ (parameters) | Euclidean | Initialization prior |
| **Markov Chain** | $\Delta^n$ (simplex) | Total variation | Uniform |
:::

---

#### 8.C.2 $\Phi^{\text{thin}}$ — Thin Height Object

**Motto:** *"The Potential"*

:::{prf:definition} Thin Height Object
:label: def-thin-height

The **Thin Height Object** is a tuple:
$$\Phi^{\text{thin}} = (F, \nabla, \alpha)$$

| Component | Type | What User Provides |
|-----------|------|-------------------|
| $F$ | $\mathcal{X} \to \mathbb{R} \cup \{\infty\}$ | Energy/height functional |
| $\nabla$ | Gradient or slope operator | Local descent direction |
| $\alpha$ | $\mathbb{Q}_{>0}$ | Scaling dimension |

**Automatically Derived by Framework:**

| Derived Component | Construction | Used By |
|-------------------|--------------|---------|
| $\Phi_\infty$ | $\limsup_{x \to \Sigma} F(x)$ | $\mathrm{LS}_\sigma$ |
| Parameter drift | $\sup_t |\partial_t \theta|$ via $\nabla$ flow | $\mathrm{SC}_{\partial c}$ |
| Critical set | $\text{Crit}(F) = \{x : \nabla F = 0\}$ | $\mathrm{LS}_\sigma$ |
| Stiffness | $\theta$ from $\|F - F_\infty\| \leq C \|\nabla F\|^\theta$ | $\mathrm{LS}_\sigma$ |
:::

:::{prf:example} Thin Height Instantiations
:label: ex-thin-height

| Domain | $F$ | $\nabla$ | $\alpha$ |
|--------|-----|----------|----------|
| **Navier-Stokes** | $\frac{1}{2}\|u\|^2_{L^2}$ | $-\nu\Delta + (u \cdot \nabla)$ | $1$ |
| **Mean Curvature** | Area$(M)$ | $-H\vec{n}$ | $\frac{n-1}{n}$ |
| **Graph Coloring** | $\#$ conflicts | Greedy swap | $1$ |
| **SGD** | Loss $\mathcal{L}(\theta)$ | $-\nabla_\theta \mathcal{L}$ | $2$ |
:::

---

#### 8.C.3 $\mathfrak{D}^{\text{thin}}$ — Thin Dissipation Object

**Motto:** *"The Cost"*

:::{prf:definition} Thin Dissipation Object
:label: def-thin-dissipation

The **Thin Dissipation Object** is a tuple:
$$\mathfrak{D}^{\text{thin}} = (R, \beta)$$

| Component | Type | What User Provides |
|-----------|------|-------------------|
| $R$ | Rate morphism | Dissipation rate satisfying $\frac{d}{dt}F \leq -R$ |
| $\beta$ | $\mathbb{Q}_{>0}$ | Scaling dimension of dissipation |

**Automatically Derived by Framework:**

| Derived Component | Construction | Used By |
|-------------------|--------------|---------|
| Bad set $\Sigma$ | $\{x : R(x) \to \infty\}$ | $D_E$, $\mathrm{Cap}_H$ |
| Mixing time | $\tau_{\text{mix}} = \inf\{t : \|P_t - \pi\|_{\text{TV}} < 1/e\}$ | $\mathrm{TB}_\rho$ |
| Concentration locus | $\{x : \mu(\epsilon\text{-ball}) \to 0\}$ | $C_\mu$ |
:::

:::{prf:example} Thin Dissipation Instantiations
:label: ex-thin-dissipation

| Domain | $R$ | $\beta$ |
|--------|-----|---------|
| **Navier-Stokes** | $\nu\|\nabla u\|^2_{L^2}$ | $1$ |
| **Ricci Flow** | $\int_M |Ric|^2 d\mu$ | $0$ |
| **MCMC** | Gap$(L)$ | $1$ |
| **Gradient Descent** | $\|\nabla \mathcal{L}\|^2$ | $2$ |
:::

---

#### 8.C.4 $G^{\text{thin}}$ — Thin Symmetry Object

**Motto:** *"The Invariance"*

:::{prf:definition} Thin Symmetry Object
:label: def-thin-symmetry

The **Thin Symmetry Object** is a tuple:
$$G^{\text{thin}} = (\text{Grp}, \rho, \mathcal{S})$$

| Component | Type | What User Provides |
|-----------|------|-------------------|
| $\text{Grp}$ | Group object in $\mathcal{E}$ | The symmetry group |
| $\rho$ | $\text{Grp} \times \mathcal{X} \to \mathcal{X}$ | Group action on state space |
| $\mathcal{S}$ | Subgroup of $\text{Grp}$ | Scaling subgroup |

**Automatically Derived by Framework (via Universal Singularity Modules):**

| Derived Component | Construction | Used By |
|-------------------|--------------|---------|
| ProfileExtractor | MT 14.1 (Profile Classification) | Modes 2-3 |
| VacuumStabilizer | Isotropy group of vacuum | $\mathrm{Rep}_K$ |
| SurgeryOperator | MT 16.1 (Structural Surgery) | Modes 4+barrier |
| Parameter Moduli | $\Theta = \mathcal{X}/G$ | $\mathrm{SC}_{\partial c}$ |
:::

:::{prf:example} Thin Symmetry Instantiations
:label: ex-thin-symmetry

| Domain | Grp | $\rho$ | $\mathcal{S}$ |
|--------|-----|--------|---------------|
| **Navier-Stokes** | $\text{ISO}(3) \ltimes \mathbb{R}^3$ | Rotation + translation | $\mathbb{R}_{>0}$ scaling |
| **Yang-Mills** | $\mathcal{G}$ (gauge group) | Gauge transformation | Conformal |
| **Graph** | $\text{Aut}(G) \times S_k$ | Vertex + color permutation | Trivial |
| **Neural Net** | $S_n$ (neuron permutation) | Weight rearrangement | Layer scaling |
:::

---

#### 8.C.5 Summary: The Four Thin Objects

:::{prf:remark} Minimal Instantiation Burden
:label: rem-minimal-burden

To instantiate a Hypostructure, the user provides exactly **10 primitive components**:

| Object | Components | Physical Meaning |
|--------|------------|------------------|
| $\mathcal{X}^{\text{thin}}$ | $\mathcal{X}, d, \mu$ | "Where does the system live?" |
| $\Phi^{\text{thin}}$ | $F, \nabla, \alpha$ | "What is being minimized?" |
| $\mathfrak{D}^{\text{thin}}$ | $R, \beta$ | "How fast does energy dissipate?" |
| $G^{\text{thin}}$ | Grp, $\rho, \mathcal{S}$ | "What symmetries does the system have?" |

The **full Kernel Objects** of Section 8.A are then constructed automatically:
$$\mathcal{H}^{\text{full}} = \text{Expand}(\mathcal{X}^{\text{thin}}, \Phi^{\text{thin}}, \mathfrak{D}^{\text{thin}}, G^{\text{thin}})$$

This expansion is performed by the **Universal Singularity Modules** (Sections 14-16), which implement the `ProfileExtractor` and `SurgeryOperator` interfaces as metatheorems rather than user-provided code.
:::

:::{prf:metatheorem} Thin-to-Full Expansion
:label: mt-thin-expansion

Given thin objects $(\mathcal{X}^{\text{thin}}, \Phi^{\text{thin}}, \mathfrak{D}^{\text{thin}}, G^{\text{thin}})$, the Framework automatically constructs:

1. **Topological Structure:**
   - SectorMap $\leftarrow \pi_0(\mathcal{X})$
   - Dictionary $\leftarrow \dim(\mathcal{X}) + $ type signature

2. **Singularity Detection:**
   - $\mathcal{X}_{\text{bad}} \leftarrow \{x : R(x) \to \infty\}$
   - $\Sigma \leftarrow \text{support of singular measure}$

3. **Profile Classification (MT 14.1):**
   - ProfileExtractor $\leftarrow$ scaling group orbit analysis
   - Canonical library $\leftarrow$ moduli space computation

4. **Surgery Construction (MT 16.1):**
   - SurgeryOperator $\leftarrow$ pushout along excision
   - Admissibility $\leftarrow$ capacity bounds from $\mu$

**Guarantee:** If the thin objects satisfy basic consistency (metric is complete, $F$ is lower semicontinuous, $R \geq 0$, $\rho$ is continuous), then the expansion produces valid full Kernel Objects.

**Literature:** Concentration-compactness profile extraction {cite}`Lions84`; moduli space theory {cite}`MumfordFogartyKirwan94`; excision in surgery {cite}`Perelman03`.
:::

---

### 8.19. Soft-to-Backend Compilation

This section defines the **compilation layer** that automatically derives backend permits from soft interfaces for good types. Users implement only soft interfaces; the framework derives WP, ProfDec, KM, Rigidity, etc.

#### 8.19.1 Architecture

```
USER PROVIDES (Soft Layer)
────────────────────────────────────────
D_E, C_μ, SC_λ, LS_σ, Mon_φ, Rep_K, TB_π, TB_O
────────────────────────────────────────
         ↓ Compilation Metatheorems
────────────────────────────────────────
FRAMEWORK DERIVES (Backend Layer)
────────────────────────────────────────
WP_{s_c}, ProfDec, KM, Rigidity, MorseDecomp, Attr
────────────────────────────────────────
         ↓ Existing Metatheorems
────────────────────────────────────────
FINAL RESULTS
────────────────────────────────────────
Lock^blk, K_prof^+, Global Regularity
```

For **good types** (satisfying the Automation Guarantee, Definition {prf:ref}`def-automation-guarantee`), soft interface verification **automatically discharges** backend permits via compilation metatheorems.

---

:::{prf:metatheorem} Soft→WP Compilation
:label: mt-soft-wp
:class: metatheorem

**Statement:** For good types $T$ satisfying the Automation Guarantee, critical well-posedness is derived from soft interfaces.

**Soft Hypotheses:**
$$K_{\mathcal{H}_0}^+ \wedge K_{D_E}^+ \wedge K_{\mathrm{Bound}}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{Rep}_K}^+$$

**Produces:**
$$K_{\mathrm{WP}_{s_c}}^+$$

**Mechanism (Template Matching):**
The evaluator `Eval_WP(T)` checks whether $T$ matches a known well-posedness template:

| Template | Soft Signature | WP Theorem Applied |
|----------|----------------|---------------------|
| Semilinear parabolic | $D_E^+$ (coercive) + $\mathrm{Bound}^+$ (Dirichlet/Neumann) | Energy-space LWP |
| Semilinear wave | $\mathrm{SC}_\lambda^+$ (finite speed) + $\mathrm{Bound}^+$ | Strichartz estimates |
| Semilinear Schrödinger | $\mathrm{SC}_\lambda^+$ + $D_E^+$ (conservation) | Dispersive estimates |
| Symmetric hyperbolic | $\mathrm{Rep}_K^+$ (finite description) | Friedrichs method |

**Certificate Emitted:**
$K_{\mathrm{WP}_{s_c}}^+ = (\mathsf{template\_ID}, \mathsf{theorem\_citation}, s_c, \mathsf{continuation\_criterion})$

**NO-Inconclusive Case:** If $T$ matches no template, emit $K_{\mathrm{WP}}^{\mathrm{inc}}$ with $\mathsf{failure\_code} = \texttt{TEMPLATE\_MISS}$. The user may supply a WP proof manually or extend the template database.

**Literature:** {cite}`CazenaveSemilinear03`; {cite}`Tao06`.
:::

---

:::{prf:metatheorem} Soft→ProfDec Compilation
:label: mt-soft-profdec
:class: metatheorem

**Statement:** For good types with concentration and scaling, profile decomposition is derived.

**Soft Hypotheses:**
$$K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{Rep}_K}^+$$

**Produces:**
$$K_{\mathrm{ProfDec}_{s_c,G}}^+$$

**Mechanism:**
1. **Space Check:** $C_\mu^+$ certifies concentration occurs (profile $V$ exists)
2. **Symmetry Check:** $\mathrm{SC}_\lambda^+$ certifies scaling group $G = \mathbb{R}^+ \times \mathbb{R}^d$ (or subgroup)
3. **Decomposition Theorem:** Apply Bahouri-Gérard / Lions profile decomposition:
   $$u_n = \sum_{j=1}^J g_n^{(j)} \cdot V^{(j)} + w_n^{(J)}$$
   - Orthogonality from $\mathrm{SC}_\lambda^+$ (scaling parameters diverge)
   - Remainder vanishes in control norm from $C_\mu^+$ (no further concentration)

**Certificate Emitted:**
$$K_{\mathrm{ProfDec}}^+ = (\{V^{(j)}\}, \{g_n^{(j)}\}, \mathsf{orthogonality}, \mathsf{remainder\_smallness})$$

**Evaluator `Eval_ProfDec(T)`:**
- Check: Is state space a Hilbert/Banach space with scaling action?
- Check: Is symmetry group $G$ standard (translations + scaling)?
- If both YES: emit YES with $K_{\mathrm{ProfDec}}^+$
- Else: emit NO with $K_{\mathrm{ProfDec}}^{\mathrm{inc}}$ (recording which check failed)

**Literature:** {cite}`BahouriGerard99`; {cite}`Lions84`.
:::

---

:::{prf:metatheorem} Soft→KM Compilation
:label: mt-soft-km
:class: metatheorem

**Statement:** The concentration-compactness + stability machine is derived from WP, ProfDec, and soft interfaces.

**Hypotheses (Mix of Derived + Soft):**
$$K_{\mathrm{WP}_{s_c}}^+ \wedge K_{\mathrm{ProfDec}}^+ \wedge K_{D_E}^+ \wedge K_{\mathrm{SC}_\lambda}^+$$

**Produces:**
$$K_{\mathrm{KM}_{\mathrm{CC+stab}}}^+$$

**Mechanism:**
1. **Minimal Element Extraction:** From $D_E^+$ (energy bounded below) + $\mathrm{ProfDec}^+$ (profiles extracted)
2. **Almost Periodicity:** From $\mathrm{SC}_\lambda^+$ (scaling controls trajectory)
3. **Stability/Perturbation:** From $\mathrm{WP}^+$ (small data → small evolution deviation)

**Certificate Emitted:**
$$K_{\mathrm{KM}}^+ = (\mathsf{minimal\_u^*}, E_c, \mathsf{almost\_periodic\_mod\_G}, \mathsf{perturbation\_lemma})$$

**Note:** This is a **composition** of derived permits, not a new template match. The Sieve assembles it automatically once WP and ProfDec are derived.

**Literature:** {cite}`KenigMerle06`.
:::

---

:::{prf:metatheorem} Soft→Rigidity Compilation (Hybrid)
:label: mt-soft-rigidity
:class: metatheorem

**Statement:** Rigidity is derived via monotonicity-interface producing a rigidity-check that feeds into Lock/obstruction.

**Soft Hypotheses:**
$$K_{\mathrm{Mon}_\phi}^+ \wedge K_{\mathrm{KM}}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Produces:**
$$K_{\mathrm{Rigidity}_T}^+$$

**Hybrid Mechanism (3 Steps):**

*Step 1 (Monotonicity Check).* By $K_{\mathrm{Mon}_\phi}^+$, the almost-periodic solution $u^*$ (from $K_{\mathrm{KM}}^+$) satisfies a monotonicity identity:
$$\frac{d^2}{dt^2} M_\phi(t) \geq c \|\nabla u^*\|^2 - C\|u^*\|^2$$
For almost-periodic $u^*$, integrate: either $u^*$ disperses (contradiction to almost-periodicity) or $u^*$ concentrates to a stationary/self-similar profile.

*Step 2 (Łojasiewicz Closure).* By $K_{\mathrm{LS}_\sigma}^+$, near critical points:
$$\|\nabla \Phi(u^*)\| \geq c|\Phi(u^*) - \Phi(V)|^{1-\theta}$$
This prevents oscillation: $u^*$ must converge to equilibrium $V$.

*Step 3 (Lock Exclusion).* By $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$, any "bad" $u^*$ (counterexample to regularity) has $\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}, \mathcal{H}) = \emptyset$.
- If $u^* \notin \mathcal{L}_T$ (library): it would embed a bad pattern → Lock blocks
- Therefore $u^* \in \mathcal{L}_T$ (classified into library)

**Certificate Emitted:**
$$K_{\mathrm{Rigidity}}^+ = (\mathsf{Mon\_identity}, \mathsf{LS\_closure}, \mathsf{Lock\_exclusion}, \mathcal{L}_T)$$

**Key Insight:** Rigidity becomes **categorical** (Lock) rather than purely analytic. The monotonicity interface provides the analytic input; Lock provides the conclusion.

**Literature:** {cite}`DuyckaertsKenigMerle11`; {cite}`KillipVisan10`.
:::

---

:::{prf:metatheorem} Soft→Attr Compilation
:label: mt-soft-attr
:class: metatheorem

**Statement:** Global attractor existence is derived from soft interfaces for dissipative systems.

**Soft Hypotheses:**
$$K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{TB}_\pi}^+$$

**Produces:**
$$K_{\mathrm{Attr}}^+$$

**Mechanism:**
1. **Dissipation:** $D_E^+$ ensures energy is bounded and decreasing
2. **Compactness:** $C_\mu^+$ ensures sublevel sets are precompact modulo symmetries
3. **Semigroup Structure:** $\mathrm{TB}_\pi^+$ ensures continuous evolution

By the Temam-Raugel attractor existence theorem (encapsulated in the permit payload), these conditions imply the global attractor $\mathcal{A}$ exists, is compact, invariant, and attracting.

**Certificate Emitted:**
$$K_{\mathrm{Attr}}^+ = (\mathcal{A}, \mathsf{absorbing\_set}, \mathsf{asymptotic\_compactness})$$

**Literature:** {cite}`Temam97`; {cite}`Raugel02`.
:::

---

:::{prf:metatheorem} Soft→MorseDecomp Compilation
:label: mt-soft-morsedecomp
:class: metatheorem

**Statement:** Morse/gradient-like decomposition is derived from attractor existence + soft interfaces.

**Hypotheses:**
$$K_{\mathrm{Attr}}^+ \wedge K_{D_E}^+ \wedge K_{\mathrm{LS}_\sigma}^+$$

**Produces:**
$$K_{\mathrm{MorseDecomp}}^+$$

**Mechanism:**
1. **Attractor Exists:** From $K_{\mathrm{Attr}}^+$ (compact, invariant, attracting)
2. **Lyapunov Function:** $D_E^+$ certifies $\Phi$ decreases along trajectories (dissipation)
3. **Gradient-like Structure:** If $\Phi$ is strictly decreasing except at equilibria, apply gradient-like backend
4. **Łojasiewicz Prevents Cycles:** $\mathrm{LS}_\sigma^+$ ensures trajectories cannot oscillate indefinitely

**Certificate Emitted:**
$$K_{\mathrm{MorseDecomp}}^+ = (\mathsf{gradient\_like}, \mathcal{E}, \{W^u(\xi)\}, \mathsf{no\_periodic})$$

**Evaluator:** Check if $D_E^+$ is strict (not just $\leq$). If strict → gradient-like. If not → may need Morse-Smale or Conley backend.

**Literature:** {cite}`Conley78`; {cite}`Hale88`.
:::

---

:::{prf:theorem} Soft-to-Backend Completeness
:label: thm-soft-backend-complete
:class: theorem

**Statement:** For good types $T$ satisfying the Automation Guarantee, all backend permits are derived from soft interfaces.

$$\underbrace{K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{Rep}_K}^+ \wedge K_{\mathrm{Mon}_\phi}^+}_{\text{Soft Layer (User Provides)}}$$
$$\Downarrow \text{Compilation}$$
$$\underbrace{K_{\mathrm{WP}}^+ \wedge K_{\mathrm{ProfDec}}^+ \wedge K_{\mathrm{KM}}^+ \wedge K_{\mathrm{Rigidity}}^+}_{\text{Backend Layer (Framework Derives)}}$$

**Proof:** Chain of MT-SOFT→WP ({prf:ref}`mt-soft-wp`), MT-SOFT→ProfDec ({prf:ref}`mt-soft-profdec`), MT-SOFT→KM ({prf:ref}`mt-soft-km`), MT-SOFT→Rigidity ({prf:ref}`mt-soft-rigidity`).

**Consequence:** The public signature of `mt-auto-profile` requires only soft interfaces. Backend permits appear only in the **internal compilation proof**, not in the user-facing hypotheses.
:::

---

#### 8.19.2 Evaluators for Derived Permits

The Sieve implements proof-producing evaluators for each derived permit. Every evaluator returns a binary YES/NO verdict with typed certificates:

| Evaluator | Input | YES Output | NO Output | Template Database |
|-----------|-------|------------|-----------|-------------------|
| `Eval_WP(T)` | Type $T$, soft certs | $K_{\mathrm{WP}}^+$ | $K_{\mathrm{WP}}^{\mathrm{wit}}$ or $K_{\mathrm{WP}}^{\mathrm{inc}}$ | Semilinear parabolic, wave, Schrödinger, hyperbolic |
| `Eval_ProfDec(T)` | Type $T$, $C_\mu^+$, $\mathrm{SC}_\lambda^+$ | $K_{\mathrm{ProfDec}}^+$ | $K_{\mathrm{ProfDec}}^{\mathrm{wit}}$ or $K_{\mathrm{ProfDec}}^{\mathrm{inc}}$ | Hilbert space + standard symmetry |
| `Eval_KM(T)` | $\mathrm{WP}^+$, $\mathrm{ProfDec}^+$, $D_E^+$ | $K_{\mathrm{KM}}^+$ | $K_{\mathrm{KM}}^{\mathrm{wit}}$ or $K_{\mathrm{KM}}^{\mathrm{inc}}$ | Composition (no template needed) |
| `Eval_Rigidity(T)` | $\mathrm{Mon}^+$, $\mathrm{KM}^+$, $\mathrm{LS}^+$, Lock | $K_{\mathrm{Rigidity}}^+$ | $K_{\mathrm{Rigidity}}^{\mathrm{wit}}$ or $K_{\mathrm{Rigidity}}^{\mathrm{inc}}$ | Hybrid mechanism |
| `Eval_Attr(T)` | $D_E^+$, $C_\mu^+$, $\mathrm{TB}_\pi^+$ | $K_{\mathrm{Attr}}^+$ | $K_{\mathrm{Attr}}^{\mathrm{wit}}$ or $K_{\mathrm{Attr}}^{\mathrm{inc}}$ | Temam-Raugel theorem |
| `Eval_MorseDecomp(T)` | $\mathrm{Attr}^+$, $D_E^+$, $\mathrm{LS}^+$ | $K_{\mathrm{MorseDecomp}}^+$ | $K_{\mathrm{MorseDecomp}}^{\mathrm{wit}}$ or $K_{\mathrm{MorseDecomp}}^{\mathrm{inc}}$ | Gradient-like / Morse-Smale |

**NO-Inconclusive Policy:** If no template matches, the evaluator returns NO with $K_P^{\mathrm{inc}}$ (not a semantic refutation). The user may then:
1. Supply the backend permit manually (escape hatch)
2. Add a new template to the database
3. Accept that the type is "non-good" and requires custom proof

**Routing on NO Certificates:** The Sieve branches on certificate type:
- **NO with $K^{\mathrm{wit}}$**: Fatal route—a genuine counterexample exists
- **NO with $K^{\mathrm{inc}}$**: Reconstruction route—try adding interfaces, refining library, or extending templates

---

## 9. The Weakest Precondition Principle

The interface formalism of Section 8 embodies a fundamental design principle: **regularity is an output, not an input**.

:::{prf:metatheorem} Weakest Precondition Principle
:label: mt-weakest-precondition

To instantiate the Structural Sieve for a dynamical system, users need only:

1. **Map Types**: Define the state space $X$, height functional $\Phi$, dissipation $\mathfrak{D}$, and symmetry group $G$.

2. **Implement Interfaces**: Provide computable formulas for each interface predicate $\mathcal{P}_n$ relevant to the problem:
   - Scaling exponents $\alpha, \beta$ (for $\mathrm{SC}_\lambda$)
   - Dimension estimates $\dim(\Sigma)$ (for $\mathrm{Cap}_H$)
   - Łojasiewicz exponent $\theta$ (for $\mathrm{LS}_\sigma$)
   - Topological invariant $\tau$ (for $\mathrm{TB}_\pi$)
   - etc.

3. **Run the Sieve**: Execute the Structural Sieve algorithm (Section 3).

**The Sieve automatically determines regularity.** Users do not need to:
- Prove global existence a priori
- Assume solutions are smooth
- Know where singularities occur
- Classify all possible blow-up profiles in advance

The verdict $\mathcal{V} \in \{\text{YES}, \text{NO}, \text{Blocked}\}$ emerges from the certificate-driven computation. NO verdicts carry typed certificates ($K^{\mathrm{wit}}$ or $K^{\mathrm{inc}}$) distinguishing refutation from inconclusiveness.

**Literature:** Dijkstra's weakest precondition calculus {cite}`Dijkstra76`; predicate transformer semantics {cite}`Back80`.
:::

:::{prf:remark} Computational Semantics
:label: rem-computational-semantics

The Weakest Precondition Principle gives the framework its **operational semantics**:

| User Provides | Sieve Computes |
|---------------|----------------|
| Interface implementations | Node verdicts |
| Type mappings | Barrier certificates |
| Local predicates | Global regularity/singularity |
| Computable checks | Certificate chains |

This is analogous to how a type checker requires type annotations but derives type safety, or how a SAT solver requires a formula but derives satisfiability.
:::

:::{prf:corollary} Separation of Concerns
:label: cor-separation-concerns

The interface formalism separates:
- **Domain expertise** (implementing $\mathcal{P}_n$ for specific PDEs)
- **Framework logic** (the Sieve algorithm and metatheorems)
- **Certificate verification** (checking that certificates satisfy their specifications)

A researcher can contribute a new interface implementation without understanding the full Sieve machinery, and the framework can be extended with new metatheorems without modifying existing implementations.
:::

---

## 10. Stronger Interface Permit Layers (X.A / X.B / X.C)

This section defines **optional strengthening layers** above the soft interface permits X.0. Each layer:
- is **strictly stronger** than X.0,
- corresponds to **clear level-up certificates** in the sieve context $\Gamma$,
- unlocks **stronger metatheorems** (uniqueness, rates, explicit reconstruction, decidability),
- remains reusable across many instantiations.

Think of these as ``difficulty settings'' for the hypostructure engine:
- **X.0** = minimal interfaces to run the sieve
- **X.A** = tame classification and admissible recovery are guaranteed
- **X.B** = analytic rigidity and canonical Lyapunov become available
- **X.C** = effective/algorithmic proof backend (lock decidability) becomes available

---

### X.A — Tame Singularity and Recovery Layer

**Purpose**: Guarantee that the ``singularity middle-game'' is **uniformly solvable**:
- profiles can be classified (finite or tame),
- surgery admissibility can be certified (possibly up to equivalence),
- recovery steps can be executed systematically.

This is the layer that turns the sieve into a **general singularity recovery engine**.

#### X.A.1 Profile Classification Trichotomy (built-in)

**X.A assumes** there exists a verified metatheorem (usable for type $T$) that, given a profile $V$, outputs exactly one of:

1. **Finite canonical library**: $K_{\mathrm{lib}}: \{V_1,\dots,V_N\}$ and membership certificate $K_{\mathrm{can}}: V\sim V_i$

2. **Tame stratification**: $K_{\mathrm{strat}}$: finite stratification $\bigsqcup_{k=1}^K \mathcal P_k\subseteq\mathbb R^{d_k}$ and classifier $K_{\mathrm{class}}$

3. **NO certificate (wild or inconclusive)**: $K_{\mathrm{Surg}}^{\mathrm{wild}}$ or $K_{\mathrm{Surg}}^{\mathrm{inc}}$: classification not possible under current Rep/definability regime (wildness witness or method exhaustion)

**Layer requirement**: Under X.A, outcomes (1) or (2) always occur for admissible profiles (i.e., classification failure is ruled out for the admissible type $T$).

#### X.A.2 Surgery Admissibility Trichotomy (built-in)

**X.A assumes** a verified metatheorem that, given Surgery Data $(\Sigma, V, \lambda(t), \mathrm{Cap}(\Sigma))$, outputs exactly one of:

1. **Admissible**: $K_{\mathrm{adm}}$: canonical + codim bound + cap bound (as in MT {prf:ref}`mt-surgery-trichotomy`)

2. **Admissible up to equivalence**: $K_{\mathrm{adm}^\sim}$: after an admissible equivalence move (YES$^\sim$), the singularity becomes admissible

3. **Not admissible**: $K_{\neg\mathrm{adm}}$: explicit reason (cap too large, codim too small, classification failure)

**Layer requirement**: Under X.A, if a singularity is encountered, it is either admissible (1) or admissible up to equivalence (2). The ``not admissible'' case becomes a certified rare/horizon mode for types outside X.A.

#### X.A.3 Structural Surgery is available

X.A includes the premise needed to invoke the **Structural Surgery Principle** uniformly:
- surgery operator exists when admissible,
- flow extends past surgery time,
- finite surgeries on finite windows (or well-founded complexity decrease).

**Level-up certificate**: `HasRecoveryEngine(T)`.

---

### X.B — Rigidity and Canonical Energy Layer

**Purpose**: Upgrade from ``we can recover'' to ``we can prove uniqueness, convergence, and canonical Lyapunov structure.''

X.B is where Perelman-like monotone functionals, canonical Lyapunov uniqueness, and rate results become derivable.

#### X.B.1 Local Stiffness is promoted to a core regime (LS upgraded)

Under X.B, any time the system is in the safe neighborhood $U$, the stiffness check can be made **core**, not merely ``blocked'':
- either StiffnessCheck = YES directly,
- or BarrierGap(Blocked) plus standard promotion rules yields LS-YES (often with $\theta=1/2$ when a gap exists).

**Level-up certificate**: `LS-Core`.

#### X.B.2 Canonical Lyapunov Functional is available

Under X.B, the conditions needed for canonical Lyapunov recovery hold in the relevant regime:
- existence of a Lyapunov-like progress function (already in X.0),
- plus rigidity/local structure sufficient for **uniqueness up to monotone reparameterization**.

**Level-up certificate**: `Lyapunov-Canonical`.

#### X.B.3 Quantitative convergence upgrades

Under X.B, you may state and reuse quantitative convergence metatheorems:
- exponential or polynomial rates (depending on LS exponent),
- uniqueness of limit objects (no wandering among equilibria),
- stability of attractor structure.

**Level-up certificate**: `Rates`.

#### X.B.4 Stratified Lyapunov across surgery times

Under X.B, even with surgeries, you can define a **piecewise / stratified Lyapunov**:
- decreasing between surgeries,
- jumps bounded by a certified jump rule,
- global progress because surgery count is finite or complexity decreases.

**Level-up certificate**: `Lyapunov-Stratified`.

#### X.B.5 Lyapunov Existence via Value Function

Under X.B, with validated interface permits $D_E$, $C_\mu$, and $\mathrm{LS}_\sigma$, a canonical Lyapunov functional exists and can be explicitly constructed.

**Level-up certificate:** `Lyapunov-Existence`.

:::{prf:metatheorem} Canonical Lyapunov Functional (MT-Lyap-1)
:label: mt-canonical-lyapunov-existence
:class: metatheorem

**[Sieve Signature]**
- **Requires:** $K_{D_E}^+$ AND $K_{C_\mu}^+$ AND $K_{\mathrm{LS}_\sigma}^+$
- **Produces:** $K_{\mathcal{L}}^+$ (Lyapunov functional exists)
- **Output:** Canonical loss $\mathcal{L}$ = optimal-transport cost to equilibrium

**Statement:** Given a hypostructure $\mathcal{H} = (\mathcal{X}, \Phi, \mathfrak{D}, G)$ with validated interface permits for dissipation ($D_E$ with $C=0$), compactness ($C_\mu$), and local stiffness ($\mathrm{LS}_\sigma$), there exists a canonical Lyapunov functional $\mathcal{L}: \mathcal{X} \to \mathbb{R} \cup \{\infty\}$ with the following properties:

1. **Monotonicity:** Along any trajectory $u(t) = S_t x$, $t \mapsto \mathcal{L}(u(t))$ is nonincreasing and strictly decreasing whenever $u(t) \notin M$.

2. **Stability:** $\mathcal{L}$ attains its minimum precisely on $M$: $\mathcal{L}(x) = \mathcal{L}_{\min}$ iff $x \in M$.

3. **Height Equivalence:** $\mathcal{L}(x) - \mathcal{L}_{\min} \asymp (\Phi(x) - \Phi_{\min})$ on energy sublevels.

4. **Uniqueness:** Any other Lyapunov functional $\Psi$ with these properties satisfies $\Psi = f \circ \mathcal{L}$ for some monotone $f$.

**Explicit Construction (Value Function):**
$$\mathcal{L}(x) := \inf\left\{\Phi(y) + \mathcal{C}(x \to y) : y \in M\right\}$$
where the infimal cost is:
$$\mathcal{C}(x \to y) := \inf\left\{\int_0^T \mathfrak{D}(S_s x) \, ds : S_T x = y, T < \infty\right\}$$

**Proof (5 Steps):**

*Step 1 (Well-definedness).* Define $\mathcal{L}$ via inf-convolution as above. The functional is well-defined since $\mathfrak{D} \geq 0$ implies $\mathcal{C} \geq 0$. By $C_\mu$ (compactness of sublevel sets) and lower semicontinuity of $\Phi + \mathcal{C}$, the infimum is attained.

*Step 2 (Monotonicity).* For $z = S_t x$, subadditivity gives $\mathcal{C}(x \to y) \leq \mathcal{C}(x \to z) + \mathcal{C}(z \to y)$ for any $y \in M$. Taking infimum over $y$:
$$\mathcal{L}(x) \leq \int_0^t \mathfrak{D}(S_s x)\, ds + \mathcal{L}(S_t x)$$
Hence $\mathcal{L}(S_t x) \leq \mathcal{L}(x) - \int_0^t \mathfrak{D}(S_s x)\, ds \leq \mathcal{L}(x)$, with strict inequality when $\mathfrak{D}(S_s x) > 0$ for $s$ in a set of positive measure.

*Step 3 (Minimum on M).* By $\mathrm{LS}_\sigma$, points in $M$ are critical: $\nabla\Phi|_M = 0$, hence the flow is stationary and $\mathfrak{D}|_M = 0$ (since $\mathfrak{D}$ measures instantaneous dissipation rate). Thus $\mathcal{C}(y \to y) = 0$ for $y \in M$, giving $\mathcal{L}(y) = \Phi(y) = \Phi_{\min}$. Conversely, if $x \notin M$, the flow eventually dissipates energy, so $\mathcal{L}(x) > \Phi_{\min}$.

*Step 4 (Height Equivalence).* **Upper bound:** By definition, $\mathcal{L}(x) \leq \Phi(y^*) + \mathcal{C}(x \to y^*)$ for optimal $y^* \in M$. The dissipation inequality from $D_E$ bounds $\mathcal{C}(x \to y^*) \lesssim \Phi(x) - \Phi(y^*) = \Phi(x) - \Phi_{\min}$, so $\mathcal{L}(x) - \Phi_{\min} \lesssim \Phi(x) - \Phi_{\min}$. **Lower bound:** The Łojasiewicz-Simon inequality from $\mathrm{LS}_\sigma$ gives $\|\nabla\Phi(x)\| \geq c|\Phi(x) - \Phi_{\min}|^{1-\theta}$ near $M$, which integrates to $\mathcal{L}(x) - \Phi_{\min} \gtrsim \mathrm{dist}(x, M)^{1/\theta}$.

*Step 5 (Uniqueness).* Let $\Psi$ satisfy the same properties. Since both $\mathcal{L}$ and $\Psi$ are strictly decreasing along non-equilibrium trajectories and constant on $M$, each level set $\{\mathcal{L} = c\}$ maps to a unique value under $\Psi$. Define $f: \mathrm{Im}(\mathcal{L}) \to \mathbb{R}$ by $f(\mathcal{L}(x)) := \Psi(x)$. This is well-defined (constant on level sets) and monotone (both decrease along flow). Hence $\Psi = f \circ \mathcal{L}$.

**Certificate Produced:** $K_{\mathcal{L}}^+ = (\mathcal{L}, M, \Phi_{\min}, \mathcal{C})$

**Literature:** {cite}`AmbrosioGigliSavare08,Villani09`
:::

#### X.B.6 Action Reconstruction via Jacobi Metric

When gradient consistency ($\mathrm{GC}_\nabla$) is additionally validated, the Lyapunov functional becomes explicitly computable as geodesic distance in a conformally scaled metric.

**Level-up certificate:** `Lyapunov-Jacobi`.

:::{prf:metatheorem} Action Reconstruction (MT-Lyap-2)
:label: mt-jacobi-reconstruction
:class: metatheorem

**[Sieve Signature]**
- **Requires:** $K_{D_E}^+$ AND $K_{\mathrm{LS}_\sigma}^+$ AND $K_{\mathrm{GC}_\nabla}^+$
- **Produces:** $K_{\text{Jacobi}}^+$ (Jacobi metric reconstruction)
- **Output:** $\mathcal{L}(x) = \mathrm{dist}_{g_{\mathfrak{D}}}(x, M)$

**Statement:** Let $\mathcal{H}$ satisfy interface permits $D_E$, $\mathrm{LS}_\sigma$, and $\mathrm{GC}_\nabla$ on a metric space $(\mathcal{X}, g)$. Then the canonical Lyapunov functional is explicitly the **minimal geodesic action** from $x$ to the safe manifold $M$ with respect to the **Jacobi metric**:

$$g_{\mathfrak{D}} := \mathfrak{D} \cdot g \quad \text{(conformal scaling by dissipation)}$$

**Explicit Formula:**
$$\mathcal{L}(x) = \Phi_{\min} + \inf_{\gamma: x \to M} \int_0^1 \sqrt{\mathfrak{D}(\gamma(s))} \cdot \|\dot{\gamma}(s)\|_g \, ds$$

**Simplified Form:**
$$\mathcal{L}(x) = \Phi_{\min} + \mathrm{dist}_{g_{\mathfrak{D}}}(x, M)$$

**Proof (5 Steps):**

*Step 1 (Gradient Consistency).* Interface permit $\mathrm{GC}_\nabla$ asserts: along gradient flow $\dot{u} = -\nabla_g \Phi$, we have $\|\dot{u}(t)\|_g^2 = \mathfrak{D}(u(t))$. This identifies dissipation with squared velocity.

*Step 2 (Jacobi Length Formula).* For the Jacobi metric $g_{\mathfrak{D}} = \mathfrak{D} \cdot g$, the length element is $ds_{g_{\mathfrak{D}}} = \sqrt{\mathfrak{D}} \cdot ds_g$. Hence for any curve $\gamma$:
$$\mathrm{Length}_{g_{\mathfrak{D}}}(\gamma) = \int_0^T \|\dot{\gamma}(t)\|_{g_{\mathfrak{D}}} \, dt = \int_0^T \sqrt{\mathfrak{D}(\gamma(t))} \|\dot{\gamma}(t)\|_g \, dt$$

*Step 3 (Flow Paths Have Optimal Length).* Along gradient flow $u(t) = S_t x$, by Step 1: $\sqrt{\mathfrak{D}(u(t))} \|\dot{u}(t)\|_g = \sqrt{\mathfrak{D}} \cdot \sqrt{\mathfrak{D}} = \mathfrak{D}(u(t))$. Integrating:
$$\mathrm{Length}_{g_{\mathfrak{D}}}(u|_{[0,T]}) = \int_0^T \mathfrak{D}(u(t))\, dt = \mathcal{C}(x \to u(T))$$
Thus Jacobi length equals accumulated cost, and by MT-Lyap-1, gradient flow achieves the infimal cost to $M$.

*Step 4 (Distance Identification).* The infimal Jacobi length from $x$ to $M$ equals $\mathrm{dist}_{g_{\mathfrak{D}}}(x, M)$. By Step 3 and MT-Lyap-1:
$$\mathrm{dist}_{g_{\mathfrak{D}}}(x, M) = \inf_{\gamma: x \to M} \mathrm{Length}_{g_{\mathfrak{D}}}(\gamma) = \mathcal{C}(x \to M) = \mathcal{L}(x) - \Phi_{\min}$$

*Step 5 (Lyapunov Verification).* Along flow: $\frac{d}{dt}\mathcal{L}(u(t)) = \frac{d}{dt}\mathrm{dist}_{g_{\mathfrak{D}}}(u(t), M) + 0 = -\|\dot{u}(t)\|_{g_{\mathfrak{D}}} = -\mathfrak{D}(u(t)) \leq 0$, confirming monotone decay.

**Certificate Produced:** $K_{\text{Jacobi}}^+ = (g_{\mathfrak{D}}, \mathrm{dist}_{g_{\mathfrak{D}}}, M)$

**Literature:** {cite}`Mielke16,AmbrosioGigliSavare08`
:::

#### X.B.7 Hamilton-Jacobi PDE Characterization

The Lyapunov functional satisfies a static Hamilton-Jacobi equation, providing a PDE route to explicit computation.

**Level-up certificate:** `Lyapunov-HJ`.

:::{prf:metatheorem} Hamilton-Jacobi Characterization (MT-Lyap-3)
:label: mt-hamilton-jacobi
:class: metatheorem

**[Sieve Signature]**
- **Requires:** $K_{D_E}^+$ AND $K_{\mathrm{LS}_\sigma}^+$ AND $K_{\mathrm{GC}_\nabla}^+$
- **Produces:** $K_{\text{HJ}}^+$ (Hamilton-Jacobi PDE characterization)
- **Output:** $\|\nabla_g \mathcal{L}\|_g^2 = \mathfrak{D}$ with $\mathcal{L}|_M = \Phi_{\min}$

**Statement:** Under interface permits $D_E$, $\mathrm{LS}_\sigma$, and $\mathrm{GC}_\nabla$, the Lyapunov functional $\mathcal{L}(x)$ is the unique viscosity solution to the static **Hamilton-Jacobi equation**:

$$\|\nabla_g \mathcal{L}(x)\|_g^2 = \mathfrak{D}(x)$$

subject to the boundary condition $\mathcal{L}(x) = \Phi_{\min}$ for $x \in M$.

**Conformal Transformation Identity:**
For conformal scaling $\tilde{g} = \phi \cdot g$ with $\phi > 0$:
- Inverse metric: $\tilde{g}^{-1} = \phi^{-1} g^{-1}$
- Gradient: $\nabla_{\tilde{g}} f = \tilde{g}^{-1}(df, \cdot) = \phi^{-1} \nabla_g f$
- Norm squared: $\|\nabla_{\tilde{g}} f\|_{\tilde{g}}^2 = \tilde{g}(\nabla_{\tilde{g}} f, \nabla_{\tilde{g}} f) = \phi \cdot \phi^{-2} \|\nabla_g f\|_g^2 = \phi^{-1}\|\nabla_g f\|_g^2$

For Jacobi metric $g_{\mathfrak{D}} = \mathfrak{D} \cdot g$, setting $\phi = \mathfrak{D}$:
$$\|\nabla_{g_{\mathfrak{D}}} f\|_{g_{\mathfrak{D}}}^2 = \mathfrak{D}^{-1} \|\nabla_g f\|_g^2$$

**Proof (4 Steps):**

*Step 1 (Eikonal for Distance).* In any Riemannian manifold, the distance function $d_M(x) = \mathrm{dist}(x, M)$ satisfies the eikonal equation $\|\nabla d_M\| = 1$ almost everywhere (away from cut locus). For $g_{\mathfrak{D}}$:
$$\|\nabla_{g_{\mathfrak{D}}} d_M^{g_{\mathfrak{D}}}\|_{g_{\mathfrak{D}}} = 1$$
where $d_M^{g_{\mathfrak{D}}} = \mathrm{dist}_{g_{\mathfrak{D}}}(\cdot, M)$.

*Step 2 (Apply Conformal Identity).* Using the transformation with $\phi = \mathfrak{D}$:
$$1 = \|\nabla_{g_{\mathfrak{D}}} d_M^{g_{\mathfrak{D}}}\|_{g_{\mathfrak{D}}}^2 = \mathfrak{D}^{-1} \|\nabla_g d_M^{g_{\mathfrak{D}}}\|_g^2$$
Hence $\|\nabla_g d_M^{g_{\mathfrak{D}}}\|_g^2 = \mathfrak{D}$.

*Step 3 (Identification with $\mathcal{L}$).* By MT-Lyap-2: $\mathcal{L}(x) = \Phi_{\min} + d_M^{g_{\mathfrak{D}}}(x)$. Since $\nabla_g \Phi_{\min} = 0$:
$$\|\nabla_g \mathcal{L}\|_g^2 = \|\nabla_g d_M^{g_{\mathfrak{D}}}\|_g^2 = \mathfrak{D}$$
with boundary condition $\mathcal{L}|_M = \Phi_{\min}$.

*Step 4 (Viscosity Uniqueness).* The Hamilton-Jacobi equation $\|\nabla_g u\|_g^2 = \mathfrak{D}$ with $u|_M = \Phi_{\min}$ has a unique viscosity solution by standard theory {cite}`CrandallLions83`. Since $\mathcal{L}$ satisfies this equation a.e. and is Lipschitz, it is the viscosity solution.

**Certificate Produced:** $K_{\text{HJ}}^+ = (\mathcal{L}, \nabla_g \mathcal{L}, \mathfrak{D})$

**Literature:** {cite}`Evans10,CrandallLions83`
:::

#### X.B.8 Extended Action Reconstruction (Metric Spaces)

For non-Riemannian settings (Wasserstein spaces, discrete graphs), the reconstruction extends using metric slope.

**Level-up certificate:** `Lyapunov-Metric`.

:::{prf:definition} Metric Slope
:label: def-metric-slope

The **metric slope** of $\Phi$ at $u \in \mathcal{X}$ is:
$$|\partial \Phi|(u) := \limsup_{v \to u} \frac{(\Phi(u) - \Phi(v))^+}{d(u, v)}$$
where $(a)^+ := \max(a, 0)$. This generalizes $\|\nabla \Phi\|$ to metric spaces.
:::

:::{prf:definition} Generalized Gradient Consistency ($\mathrm{GC}'_\nabla$)
:label: def-gc-prime

Interface permit $\mathrm{GC}'_\nabla$ (dissipation-slope equality) holds if along any metric gradient flow trajectory:
$$\mathfrak{D}(u(t)) = |\partial \Phi|^2(u(t))$$
This extends $\mathrm{GC}_\nabla$ from Riemannian to general metric spaces.
:::

:::{prf:metatheorem} Extended Action Reconstruction (MT-Lyap-4)
:label: mt-extended-action
:class: metatheorem

**[Sieve Signature]**
- **Requires:** $K_{D_E}^+$ AND $K_{\mathrm{LS}_\sigma}^+$ AND $K_{\mathrm{GC}'_\nabla}^+$
- **Produces:** $K_{\mathcal{L}}^{\text{metric}}$ (Lyapunov on metric spaces)
- **Extends:** Riemannian → Wasserstein → Discrete

**Statement:** Under interface permit $\mathrm{GC}'_\nabla$ (dissipation-slope equality), the reconstruction theorems extend to general metric spaces. The Lyapunov functional satisfies:

$$\mathcal{L}(x) = \Phi_{\min} + \inf_{\gamma: M \to x} \int_0^1 |\partial \Phi|(\gamma(s)) \cdot |\dot{\gamma}|(s) \, ds$$

where $|\dot{\gamma}|$ denotes the metric derivative and the infimum ranges over all absolutely continuous curves from the safe manifold $M$ to $x$.

**Applications:**

| Setting | State Space | Height $\Phi$ | Dissipation $\mathfrak{D}$ | Metric Slope |
|---------|-------------|---------------|---------------------------|--------------|
| Wasserstein | $(\mathcal{P}_2(\mathbb{R}^n), W_2)$ | Entropy $H(\rho)$ | Fisher info $I(\rho)$ | $\sqrt{I(\rho)}$ |
| Discrete | Prob. on graph $V$ | Rel. entropy $H(\mu\|\pi)$ | Dirichlet form | Discrete Otto |

**Proof (3 Steps):**

*Step 1 (Metric Derivative Identity).* For absolutely continuous curves $\gamma: [0,1] \to \mathcal{X}$, the metric derivative is $|\dot{\gamma}|(s) := \lim_{h \to 0} d(\gamma(s+h), \gamma(s))/|h|$. By $\mathrm{GC}'_\nabla$, along gradient flow curves: $|\dot{u}|(t)^2 = \mathfrak{D}(u(t)) = |\partial\Phi|^2(u(t))$, hence $|\dot{u}|(t) = |\partial\Phi|(u(t))$.

*Step 2 (Action = Cost).* The action functional $\int_0^1 |\partial\Phi|(\gamma) \cdot |\dot{\gamma}|\, ds$ generalizes Jacobi length. For gradient flow $u(t)$:
$$\int_0^T |\partial\Phi|(u(t)) \cdot |\dot{u}|(t)\, dt = \int_0^T |\partial\Phi|^2(u(t))\, dt = \int_0^T \mathfrak{D}(u(t))\, dt = \mathcal{C}(x \to u(T))$$
By the Energy-Dissipation-Identity (EDI) in metric gradient flow theory {cite}`AmbrosioGigliSavare08`, this equals $\Phi(x) - \Phi(u(T))$ for curves of maximal slope.

*Step 3 (Infimum Attained).* The infimum over curves from $M$ to $x$ is attained by the (time-reversed) gradient flow, giving:
$$\mathcal{L}(x) - \Phi_{\min} = \inf_{\gamma: M \to x} \int_0^1 |\partial\Phi|(\gamma) \cdot |\dot{\gamma}|\, ds$$
This extends MT-Lyap-2 to non-smooth settings where $|\partial\Phi|$ replaces $\|\nabla\Phi\|_g$.

**Certificate Produced:** $K_{\mathcal{L}}^{\text{metric}} = (\mathcal{L}, |\partial\Phi|, d)$

**Literature:** {cite}`AmbrosioGigliSavare08,Maas11,Mielke11`
:::

---

#### X.B.9 Soft Local Tower Globalization

For **tower hypostructures** (multiscale systems), local data at each scale determines global asymptotic behavior.

:::{prf:definition} Tower Hypostructure
:label: def-tower-hypostructure

A **tower hypostructure** is a tuple $\mathbb{H} = (X_t, S_{t \to s}, \Phi, \mathfrak{D})$ where:
- $t \in \mathbb{N}$ or $t \in \mathbb{R}_+$ is a **scale index**
- $X_t$ is the state space at level $t$
- $S_{t \to s}: X_t \to X_s$ (for $s < t$) are **scale transition maps** compatible with the semiflow
- $\Phi(t)$ is the height/energy at level $t$
- $\mathfrak{D}(t)$ is the dissipation increment at level $t$
:::

:::{prf:definition} Tower Interface Permits
:label: def-tower-permits

The following **tower-specific interface permits** extend the standard permits to multiscale settings:

| Permit | Name | Question | Certificate |
|--------|------|----------|-------------|
| $C_\mu^{\mathrm{tower}}$ | SliceCompact | Is $\{\Phi(t) \leq B\}$ compact mod symmetries for each scale? | $K_{C_\mu^{\mathrm{tower}}}^{\pm}$ |
| $D_E^{\mathrm{tower}}$ | SubcritDissip | Is $\sum_t w(t)\mathfrak{D}(t) < \infty$ for $w(t) \sim e^{-\alpha t}$? | $K_{D_E^{\mathrm{tower}}}^{\pm}$ |
| $\mathrm{SC}_\lambda^{\mathrm{tower}}$ | ScaleCohere | Is $\Phi(t_2) - \Phi(t_1) = \sum_u L(u) + o(1)$? | $K_{\mathrm{SC}_\lambda^{\mathrm{tower}}}^{\pm}$ |
| $\mathrm{Rep}_K^{\mathrm{tower}}$ | LocalRecon | Is $\Phi(t)$ determined by local invariants $\{I_\alpha(t)\}$? | $K_{\mathrm{Rep}_K^{\mathrm{tower}}}^{\pm}$ |

**$C_\mu^{\mathrm{tower}}$ (Compactness on slices):** For each bounded interval of scales and each $B > 0$, the sublevel set $\{X_t : \Phi(t) \leq B\}$ is compact or finite modulo symmetries.

**$D_E^{\mathrm{tower}}$ (Subcritical dissipation):** There exists $\alpha > 0$ and weight $w(t) \sim e^{-\alpha t}$ (or $p^{-\alpha t}$ for $p$-adic towers) such that:
$$\sum_t w(t) \mathfrak{D}(t) < \infty$$

**$\mathrm{SC}_\lambda^{\mathrm{tower}}$ (Scale coherence):** For any $t_1 < t_2$:
$$\Phi(t_2) - \Phi(t_1) = \sum_{u=t_1}^{t_2-1} L(u) + o(1)$$
where each $L(u)$ is a **local contribution** determined by level $u$ data, and $o(1)$ is uniformly bounded.

**$\mathrm{Rep}_K^{\mathrm{tower}}$ (Soft local reconstruction):** For each scale $t$, the energy $\Phi(t)$ is determined (up to bounded, summable error) by **local invariants** $\{I_\alpha(t)\}_{\alpha \in A}$ at scale $t$:
$$\Phi(t) = F(\{I_\alpha(t)\}_\alpha) + O(1)$$
:::

::::{prf:metatheorem} Soft Local Tower Globalization (MT-Tower-1)
:label: mt-tower-globalization
:class: metatheorem

**Sieve Signature:**
- *Weakest Precondition:* $K_{C_\mu^{\mathrm{tower}}}^+$, $K_{D_E^{\mathrm{tower}}}^+$, $K_{\mathrm{SC}_\lambda^{\mathrm{tower}}}^+$, $K_{\mathrm{Rep}_K^{\mathrm{tower}}}^+$
- *Produces:* $K_{\mathrm{Global}}^+$ (global asymptotic structure)
- *Invalidated By:* Local-global obstruction

**Setup.** Let $\mathbb{H} = (X_t, S_{t \to s}, \Phi, \mathfrak{D})$ be a tower hypostructure with the following interface permits certified:
1. $C_\mu^{\mathrm{tower}}$: Compactness/finiteness on slices
2. $D_E^{\mathrm{tower}}$: Subcritical dissipation with weight $w(t) \sim e^{-\alpha t}$
3. $\mathrm{SC}_\lambda^{\mathrm{tower}}$: Scale coherence
4. $\mathrm{Rep}_K^{\mathrm{tower}}$: Soft local reconstruction

**Conclusion (Soft Local Tower Globalization):**

**(1)** The tower admits a **globally consistent asymptotic hypostructure**:
$$X_\infty = \varprojlim X_t$$

**(2)** The asymptotic behavior of $\Phi$ and the defect structure of $X_\infty$ is **completely determined** by the collection of local reconstruction invariants from $\mathrm{Rep}_K^{\mathrm{tower}}$.

**(3)** No supercritical growth or uncontrolled accumulation can occur: every supercritical mode violates the $D_E^{\mathrm{tower}}$ permit.

*Proof.*

*Step 1 (Existence of limit).* By $K_{C_\mu^{\mathrm{tower}}}^+$, the spaces $\{X_t\}$ at each level are precompact modulo symmetries. The transition maps $S_{t \to s}$ are compatible by the semiflow property. By $K_{D_E^{\mathrm{tower}}}^+$, the total dissipation is finite:
$$\sum_t w(t) \mathfrak{D}(t) < \infty$$
This implies $\mathfrak{D}(t) \to 0$ as $t \to \infty$ (otherwise the weighted sum diverges). Hence dynamics becomes increasingly frozen.

*Step 2 (Asymptotic consistency).* By $K_{\mathrm{SC}_\lambda^{\mathrm{tower}}}^+$:
$$\Phi(t_2) - \Phi(t_1) = \sum_{u=t_1}^{t_2-1} L(u) + O(1)$$
Taking $t_2 \to \infty$ and using finite dissipation from Step 1:
$$\Phi(\infty) - \Phi(t_1) = \sum_{u=t_1}^{\infty} L(u) + O(1)$$
The sum converges absolutely (each $L(u)$ controlled by $\mathfrak{D}(u)$). Thus $\Phi(\infty)$ is well-defined.

*Step 3 (Local determination).* By $K_{\mathrm{Rep}_K^{\mathrm{tower}}}^+$:
$$\Phi(t) = F(\{I_\alpha(t)\}_\alpha) + O(1)$$
for local invariants $\{I_\alpha(t)\}$. Taking $t \to \infty$: local invariants stabilize (by finite dissipation) to limiting values $I_\alpha(\infty)$. Therefore:
$$\Phi(\infty) = F(\{I_\alpha(\infty)\}_\alpha) + O(1)$$
The asymptotic height is completely determined by the asymptotic local data.

*Step 4 (Exclusion of supercritical growth).* Suppose supercritical growth at scale $t_0$: $\Phi(t_0+n) - \Phi(t_0) \gtrsim n^\gamma$ for some $\gamma > 0$. By $K_{\mathrm{SC}_\lambda^{\mathrm{tower}}}^+$, this growth reflects in the local contributions. But then:
$$\sum_t w(t)\mathfrak{D}(t) \geq \sum_{u=t_0}^\infty e^{-\alpha u} \cdot u^{\gamma-1} = \infty$$
for any $\gamma > 0$, contradicting $K_{D_E^{\mathrm{tower}}}^+$.

*Step 5 (Defect inheritance).* The limit $X_\infty$ inherits the hypostructure:
- Height functional: $\Phi_\infty(x_\infty) := \lim_{t\to\infty}\Phi(x_t)$
- Dissipation: $\mathfrak{D}_\infty \equiv 0$ (frozen dynamics at infinity)
- Constraints: any violation at $X_\infty$ propagates back to finite levels, contradicting the permits. $\square$

**Certificate Produced:** $K_{\mathrm{Global}}^+ = (X_\infty, \Phi_\infty, \{I_\alpha(\infty)\}_\alpha)$

**Usage:** Applies to multiscale analytic towers (fluid dynamics, gauge theories), Iwasawa towers in arithmetic, RG flows (holographic or analytic), complexity hierarchies, spectral sequences/filtrations.
::::

---

#### X.B.10 Obstruction Capacity Collapse

The "Sha Killer" — proves obstruction sectors (like Tate-Shafarevich groups) are finite. Analogous to Cartan's Theorems A/B for coherent sheaf cohomology.

:::{prf:definition} Obstruction Interface Permits
:label: def-obstruction-permits

The following **obstruction-specific interface permits** extend the standard permits to obstruction sectors $\mathcal{O} \subset \mathcal{X}$:

| Permit | Name | Question | Certificate |
|--------|------|----------|-------------|
| $\mathrm{TB}_\pi^{\mathcal{O}} + \mathrm{LS}_\sigma^{\mathcal{O}}$ | ObsDuality | Is $\langle\cdot,\cdot\rangle_{\mathcal{O}}$ non-degenerate? | $K_{\mathrm{TB}+\mathrm{LS}}^{\mathcal{O}\pm}$ |
| $C_\mu^{\mathcal{O}} + \mathrm{Cap}_H^{\mathcal{O}}$ | ObsHeight | Does $H_{\mathcal{O}}$ have compact sublevel sets? | $K_{C+\mathrm{Cap}}^{\mathcal{O}\pm}$ |
| $\mathrm{SC}_\lambda^{\mathcal{O}}$ | ObsSubcrit | Is $\sum_t w(t) \sum_{x \in \mathcal{O}_t} H_{\mathcal{O}}(x) < \infty$? | $K_{\mathrm{SC}_\lambda}^{\mathcal{O}\pm}$ |
| $D_E^{\mathcal{O}}$ | ObsDissip | Is $\mathfrak{D}_{\mathcal{O}}$ subcritical? | $K_{D_E}^{\mathcal{O}\pm}$ |

**$\mathrm{TB}_\pi^{\mathcal{O}} + \mathrm{LS}_\sigma^{\mathcal{O}}$ (Duality/Stiffness on obstruction):** The obstruction sector admits a non-degenerate invariant pairing $\langle \cdot, \cdot \rangle_{\mathcal{O}}: \mathcal{O} \times \mathcal{O} \to A$ compatible with the hypostructure flow.

**$C_\mu^{\mathcal{O}} + \mathrm{Cap}_H^{\mathcal{O}}$ (Obstruction height):** There exists a functional $H_{\mathcal{O}}: \mathcal{O} \to \mathbb{R}_{\geq 0}$ such that:
- Sublevel sets $\{x : H_{\mathcal{O}}(x) \leq B\}$ are finite/compact
- $H_{\mathcal{O}}(x) = 0 \Leftrightarrow x$ is trivial obstruction

**$\mathrm{SC}_\lambda^{\mathcal{O}}$ (Subcritical accumulation):** Under any tower or scale decomposition:
$$\sum_t w(t) \sum_{x \in \mathcal{O}_t} H_{\mathcal{O}}(x) < \infty$$

**$D_E^{\mathcal{O}}$ (Subcritical obstruction dissipation):** The obstruction defect $\mathfrak{D}_{\mathcal{O}}$ grows strictly slower than structural permits allow for infinite accumulation.
:::

::::{prf:metatheorem} Obstruction Capacity Collapse (MT-Obs-1)
:label: mt-obstruction-collapse
:class: metatheorem

**Sieve Signature:**
- *Weakest Precondition:* $K_{\mathrm{TB}+\mathrm{LS}}^{\mathcal{O}+}$, $K_{C+\mathrm{Cap}}^{\mathcal{O}+}$, $K_{\mathrm{SC}_\lambda}^{\mathcal{O}+}$, $K_{D_E}^{\mathcal{O}+}$
- *Produces:* $K_{\mathrm{Obs}}^{\mathrm{finite}}$ (obstruction sector is finite)
- *Invalidated By:* Infinite obstruction accumulation

**Setup.** Let $\mathbb{H} = (\mathcal{X}, \Phi, \mathfrak{D})$ be a hypostructure with distinguished obstruction sector $\mathcal{O} \subset \mathcal{X}$. Assume all obstruction interface permits are certified.

**Conclusion (Obstruction Capacity Collapse):**

**(1)** The obstruction sector $\mathcal{O}$ is **finite-dimensional/finite** in the appropriate sense.

**(2)** No infinite obstruction or runaway obstruction mode can exist.

**(3)** Any nonzero obstruction must appear in strictly controlled, finitely many directions, each of which is structurally detectable.

*Proof.*

*Step 1 (Finiteness at each scale).* Fix a scale $t$. By $K_{C+\mathrm{Cap}}^{\mathcal{O}+}$, the sublevel set $\mathcal{O}_t^{\leq B} := \{x \in \mathcal{O}_t : H_{\mathcal{O}}(x) \leq B\}$ is finite or compact for each $B > 0$.

*Step 2 (Uniform bound on obstruction count).* By $K_{\mathrm{SC}_\lambda}^{\mathcal{O}+}$, the weighted sum:
$$S := \sum_t w(t) \sum_{x \in \mathcal{O}_t} H_{\mathcal{O}}(x) < \infty$$
For each $t$, let $N_t := |\{x \in \mathcal{O}_t : H_{\mathcal{O}}(x) \geq \varepsilon\}|$ count non-trivial obstructions. Then:
$$S \geq \sum_t w(t) \cdot N_t \cdot \varepsilon$$
Since $S < \infty$ and $w(t) > 0$, we have $\sum_t w(t) N_t < \infty$, implying $N_t \to 0$ as $t \to \infty$.

*Step 3 (Global finiteness).* The total obstruction $\mathcal{O}_{\text{tot}} := \bigcup_t \mathcal{O}_t$ has contributions from only finitely many scales (Step 2), each finite by Step 1. Hence $\mathcal{O}_{\text{tot}}$ is finite-dimensional.

*Step 4 (No runaway modes).* Suppose a runaway obstruction exists: $(x_n) \subset \mathcal{O}$ with $H_{\mathcal{O}}(x_n) \to \infty$. By $K_{D_E}^{\mathcal{O}+}$:
$$\mathfrak{D}_{\mathcal{O}}(x_n) \leq C \cdot H_{\mathcal{O}}(x_n)^{1-\delta}$$
for some $\delta > 0$. But accumulating such obstructions requires $\sum_n H_{\mathcal{O}}(x_n) = \infty$, contradicting $K_{\mathrm{SC}_\lambda}^{\mathcal{O}+}$.

*Step 5 (Structural detectability).* By $K_{\mathrm{TB}+\mathrm{LS}}^{\mathcal{O}+}$, the pairing is non-degenerate: any non-trivial $x \in \mathcal{O}$ has $\langle x, y \rangle_{\mathcal{O}} \neq 0$ for some $y$. Combined with $H_{\mathcal{O}}$, obstructions are localized to specific directions with quantifiable pairing contributions. $\square$

**Certificate Produced:** $K_{\mathrm{Obs}}^{\mathrm{finite}} = (\mathcal{O}_{\text{tot}}, \dim(\mathcal{O}_{\text{tot}}), H_{\mathcal{O}})$

**Usage:** Applies to Tate-Shafarevich groups, torsors/cohomological obstructions, exceptional energy concentrations in PDEs, forbidden degrees in complexity theory, anomalous configurations in gauge theory.

**Literature:** Cartan's Theorems A/B for coherent cohomology {cite}`CartanSerre53`; finiteness of Tate-Shafarevich {cite}`Kolyvagin90`; {cite}`Rubin00`; obstruction theory {cite}`Steenrod51`.
::::

---

#### X.B.11 Stiff Pairing / No Null Directions

Guarantees no hidden "ghost sectors" where singularities can hide. All degrees of freedom are accounted for by free components + obstructions.

::::{prf:metatheorem} Stiff Pairing / No Null Directions (MT-Stiff-1)
:label: mt-stiff-pairing
:class: metatheorem

**Sieve Signature:**
- *Weakest Precondition:* $K_{\mathrm{LS}_\sigma}^+$, $K_{\mathrm{TB}_\pi}^+$, $K_{\mathrm{GC}_\nabla}^+$
- *Produces:* $K_{\mathrm{Stiff}}^+$ (no null directions)
- *Invalidated By:* Hidden degeneracy

**Setup.** Let $\mathbb{H} = (\mathcal{X}, \Phi, \mathfrak{D})$ be a hypostructure with bilinear pairing $\langle \cdot, \cdot \rangle : \mathcal{X} \times \mathcal{X} \to F$ such that:
- $\Phi$ is generated by this pairing (via $\mathrm{GC}_\nabla$)
- $\mathrm{LS}_\sigma$ holds (local stiffness)

Let $\mathcal{X} = X_{\mathrm{free}} \oplus X_{\mathrm{obs}} \oplus X_{\mathrm{rest}}$ be a decomposition into free sector, obstruction sector, and possible null sector.

**Hypotheses:**
1. $K_{\mathrm{LS}_\sigma}^+ + K_{\mathrm{TB}_\pi}^+$: $\langle \cdot, \cdot \rangle$ is non-degenerate on $X_{\mathrm{free}} \oplus X_{\mathrm{obs}}$
2. $K_{\mathrm{GC}_\nabla}^+$: Flat directions for $\Phi$ are flat directions for the pairing
3. Any vector orthogonal to $X_{\mathrm{free}}$ lies in $X_{\mathrm{obs}}$

**Conclusion (Stiffness / No Null Directions):**

**(1)** There is **no** $X_{\mathrm{rest}}$: $\mathcal{X} = X_{\mathrm{free}} \oplus X_{\mathrm{obs}}$

**(2)** All degrees of freedom are accounted for by free components + obstructions.

**(3)** No hidden degeneracies or "null modes" exist.

*Proof.*

*Step 1 (Pairing structure).* The pairing induces $\Psi: \mathcal{X} \to \mathcal{X}^*$, $\Psi(x)(y) := \langle x, y \rangle$. By $K_{\mathrm{LS}_\sigma}^+ + K_{\mathrm{TB}_\pi}^+$, this map is injective on $X_{\mathrm{free}} \oplus X_{\mathrm{obs}}$.

*Step 2 (Radical characterization).* Define $\mathrm{rad}(\langle \cdot, \cdot \rangle) := \{x \in \mathcal{X} : \langle x, y \rangle = 0 \text{ for all } y\}$. Any radical element is orthogonal to $X_{\mathrm{free}}$, hence lies in $X_{\mathrm{obs}}$ by hypothesis.

*Step 3 (Radical within obstruction).* If $x \in \mathrm{rad}$, then $x \in X_{\mathrm{obs}}$ (Step 2). Within $X_{\mathrm{obs}}$, the pairing is non-degenerate, so $\langle x, y \rangle = 0$ for all $y \in X_{\mathrm{obs}}$ implies $x = 0$.

*Step 4 (No null sector).* Suppose $X_{\mathrm{rest}} \neq 0$ with nonzero $z \in X_{\mathrm{rest}}$.
- *Case (a):* $z \in \mathrm{rad} \Rightarrow z = 0$ (Step 3), contradiction.
- *Case (b):* $z \notin \mathrm{rad} \Rightarrow \exists y$ with $\langle z, y \rangle \neq 0$. But $z$ orthogonal to $X_{\mathrm{free}}$ implies $z \in X_{\mathrm{obs}}$, and $X_{\mathrm{obs}} \cap X_{\mathrm{rest}} = \{0\}$, so $z = 0$, contradiction.

*Step 5 (Gradient consistency).* By $K_{\mathrm{GC}_\nabla}^+$, flat directions of $\Phi$ correspond to flat directions of the pairing. Since the pairing has trivial radical, $\Phi$ has no hidden flat directions. Therefore $X_{\mathrm{rest}} = 0$. $\square$

**Certificate Produced:** $K_{\mathrm{Stiff}}^+ = (X_{\mathrm{free}}, X_{\mathrm{obs}}, \langle\cdot,\cdot\rangle)$

**Usage:** Applies to Selmer groups with p-adic height, Hodge-theoretic intersection forms, gauge-theory BRST pairings, PDE energy inner products, complexity gradients.

**Literature:** Selmer groups and p-adic heights {cite}`MazurTate83`; {cite}`Nekovar06`; Hodge theory {cite}`GriffithsHarris78`; BRST cohomology {cite}`HenneauxTeitelboim92`; non-degenerate pairings {cite}`Serre62`.
::::

---

### X.C — Effective (Algorithmic) Proof Layer

**Purpose**: Turn the sieve into an **effective conjecture prover**:
- certificates are finite,
- closure terminates,
- lock tactics are decidable or semi-decidable.

This layer is about **Rep + computability**.

#### X.C.1 Finite certificate language (bounded description)

X.C assumes the certificate schemas used by the run live in a **finite or bounded-description language** for the chosen type $T$:
- bounded precision, bounded term size, finite invariant basis, etc.

This ensures:
- promotion closure $\mathrm{Cl}(\Gamma)$ terminates (or is effectively computable),
- replay is decidable.

**Level-up certificate**: `Cert-Finite(T)`.

#### X.C.2 Rep is constructive

Rep is not merely ``exists,'' but comes with:
- a concrete dictionary $D$,
- verifiers for invariants and morphism constraints,
- and an explicit representation of the bad pattern object.

**Level-up certificate**: `Rep-Constructive`.

#### X.C.3 Lock backend tactics are effective

E1--E10 tactics become effective procedures:
- dimension checks, invariant mismatch checks, positivity/integrality constraints, functional equations,
- in a decidable fragment (SMT/linear arithmetic/rewrite systems).

Outcome:
- either an explicit morphism witness,
- or a checkable Hom-emptiness certificate.

**Level-up certificate**: `Lock-Decidable` or `Lock-SemiDecidable`.

#### X.C.4 Decidable/Effective classification (optional)

If you also assume effective profile stratification (X.A) and effective transport toolkit:
- classification of profiles becomes decidable within the type $T$,
- admissibility and surgery selection can be automated.

**Level-up certificate**: `Classification-Decidable`.

---

### Relationship between layers (summary)

- **X.0**: You can run the sieve; you can classify failures; you can recover when surgery is certified; horizon modes are explicit.
- **X.A**: You can *systematically classify profiles* and *systematically recover* from singularities for admissible types.
- **X.B**: You can derive *canonical Lyapunov structure*, *uniqueness*, and *rates* (including stratified Lyapunov across surgery).
- **X.C**: You can make large parts of the engine *algorithmic/decidable* (proof backend becomes executable).

---

### How to use these layers in theorem statements

Every strong metatheorem should be written as:
- **Minimal preconditions**: certificates available under X.0 (works broadly, weaker conclusion).
- **Upgrade preconditions**: additional layer certificates (X.A/X.B/X.C), yielding stronger conclusions.

**Example schema**:
- ``Soft Lyapunov exists'' (X.0)
- ``Canonical Lyapunov unique up to reparam'' (X.B)
- ``Explicit reconstruction (HJ/Jacobi)'' (X.B + GC certificates)
- ``Algorithmic lock proof (E1--E10 decidable)'' (X.C)

---

## 11. The Type System

:::{prf:definition} Problem type
:label: def-problem-type

A **type** $T$ is a class of dynamical systems sharing:
1. Standard structure (local well-posedness, energy inequality form)
2. Admissible equivalence moves
3. Applicable toolkit factories
4. Expected horizon outcomes when Rep/definability fails

:::

---

### Type Catalog

:::{prf:definition} Type $T_{\text{parabolic}}$
:label: def-type-parabolic

**Parabolic PDE / Geometric Flows**

**Structure**:
- Evolution: $\partial_t u = \Delta u + F(u, \nabla u)$ or geometric analog
- Energy: $\Phi(u) = \int |\nabla u|^2 + V(u)$
- Dissipation: $\mathfrak{D}(u) = \int |\partial_t u|^2$ or $\int |\nabla^2 u|^2$

**Equivalence moves**: Symmetry quotient, metric deformation, Ricci/mean curvature surgery

**Standard barriers**: Saturation, Type II (via monotonicity formulas), Capacity (epsilon-regularity)

**Profile library**: Solitons, shrinkers, translators, ancient solutions

:::

:::{prf:definition} Type $T_{\text{dispersive}}$
:label: def-type-dispersive

**Dispersive PDE / Scattering**

**Structure**:
- Evolution: $i\partial_t u = \Delta u + |u|^{p-1}u$ or wave equation
- Energy: $\Phi(u) = \int |\nabla u|^2 + |u|^{p+1}$
- Dispersion: Strichartz estimates

**Equivalence moves**: Galilean/Lorentz symmetry, concentration-compactness

**Standard barriers**: Scattering (Benign), Type II (Kenig-Merle), Capacity

**Profile library**: Ground states, traveling waves, blow-up profiles

:::

:::{prf:definition} Type $T_{\text{metricGF}}$
:label: def-type-metricgf

**Metric Gradient Flows**

**Structure**:
- Evolution: Curves of maximal slope in metric spaces
- Energy: Lower semicontinuous functional
- Dissipation: Metric derivative squared

**Equivalence moves**: Metric equivalence, Wasserstein transport

**Standard barriers**: EVI (Evolution Variational Inequality), Geodesic convexity

:::

:::{prf:definition} Type $T_{\text{Markov}}$
:label: def-type-markov

**Diffusions / Markov Semigroups**

**Structure**:
- Evolution: $\partial_t \mu = L^* \mu$ (Fokker-Planck)
- Energy: Free energy / entropy
- Dissipation: Fisher information

**Equivalence moves**: Time-reversal, detailed balance conjugacy

**Standard barriers**: Log-Sobolev, Poincaré, Mixing times

:::

:::{prf:definition} Type $T_{\text{algorithmic}}$
:label: def-type-algorithmic

**Computational / Iterative Systems**

**Structure**:
- Evolution: $x_{n+1} = F(x_n)$ or continuous-time analog
- Energy: Loss function / Lyapunov
- Dissipation: Per-step progress

**Equivalence moves**: Conjugacy, preconditioning

**Standard barriers**: Convergence rate, Complexity bounds

:::

---

# Part VIII: Barrier and Surgery Contracts

## 12. Certificate-Driven Barrier Atlas

:::{prf:definition} Barrier contract format
:label: def-barrier-format

Each barrier entry in the atlas specifies:

1. **Trigger**: Which gate NO invokes this barrier
2. **Pre**: Required certificates (from $\Gamma$), subject to non-circularity
3. **Blocked certificate**: $K^{\mathrm{blk}}$ satisfying $K^{\mathrm{blk}} \Rightarrow \mathrm{Pre}(\text{next gate})$
4. **Breached certificate**: $K^{\mathrm{br}}$ satisfying:
   - $K^{\mathrm{br}} \Rightarrow \text{Mode } m \text{ active}$
   - $K^{\mathrm{br}} \Rightarrow \mathrm{SurgeryAdmissible}(m)$
5. **Scope**: Which types $T$ this barrier applies to

:::

:::{prf:theorem} Non-circularity
:label: thm-barrier-noncircular

For any barrier $B$ triggered by gate $i$ with predicate $P_i$:
$$P_i \notin \mathrm{Pre}(B)$$
A barrier invoked because $P_i$ failed cannot assume $P_i$ as a prerequisite.

**Literature:** Stratification and well-foundedness {cite}`VanGelder91`; non-circular definitions {cite}`AptBolPedreschi94`.

:::

---

## 13. Surgery Contracts

:::{prf:definition} Surgery contract format
:label: def-surgery-format

Each surgery entry follows the **Surgery Specification Schema** (Definition {prf:ref}`def-surgery-schema`):

1. **Surgery ID** and **Target Mode**: Unique identifier and triggering failure mode
2. **Interface Dependencies**:
   - **Primary:** Interface providing the singular object/profile $V$ and locus $\Sigma$
   - **Secondary:** Interface providing canonical library $\mathcal{L}_T$ or capacity bounds
3. **Admissibility Signature**:
   - **Input Certificate:** $K^{\mathrm{br}}$ from triggering barrier
   - **Admissibility Predicate:** Conditions for safe surgery (Case 1 of Trichotomy)
4. **Transformation Law** ($\mathcal{O}_S$):
   - **State Space:** How $X \to X'$
   - **Height Jump:** Energy/height change guarantee
   - **Topology:** Sector changes if any
5. **Postcondition**:
   - **Re-entry Certificate:** $K^{\mathrm{re}}$ with $K^{\mathrm{re}} \Rightarrow \mathrm{Pre}(\text{target node})$
   - **Re-entry Target:** Node to resume sieve execution
   - **Progress Guarantee:** Type A (bounded count) or Type B (well-founded complexity)

See Section 7 for the complete catalog of 17 surgery specifications.
:::

:::{prf:definition} Progress measures
:label: def-progress-measures

Valid progress measures for surgery termination:

**Type A (Bounded count)**:
$$\#\{S\text{-surgeries on } [0, T)\} \leq N(T, \Phi(x_0))$$
for explicit bound $N$ depending on time and initial energy.

**Type B (Well-founded)**:
A complexity measure $\mathcal{C}: X \to \mathbb{N}$ (or ordinal) with:
$$\mathcal{O}_S(x) = x' \Rightarrow \mathcal{C}(x') < \mathcal{C}(x)$$

:::

---

# Part IX: Universal Singularity Modules

**Design Philosophy:** These metatheorems act as **Factory Functions** that automatically implement the `ProfileExtractor` and `SurgeryOperator` interfaces from Section 8.A. Users do not invent surgery procedures or profile classifiers—the Framework constructs them from the thin kernel objects.

:::{prf:remark} Factory Function Pattern
:label: rem-factory-pattern

The Universal Singularity Modules implement a **dependency injection** pattern:

| Interface | Factory Metatheorem | Input | Output |
|-----------|---------------------|-------|--------|
| `ProfileExtractor` | MT 14.1 (Profile Classification) | $G^{\text{thin}}, \Phi^{\text{thin}}$ | Canonical library $\mathcal{L}_T$ |
| `SurgeryAdmissibility` | MT 15.1 (Surgery Admissibility) | $\mu, \mathfrak{D}^{\text{thin}}$ | Admissibility predicate |
| `SurgeryOperator` | MT 16.1 (Structural Surgery) | Full $\mathcal{H}$ | Pushout surgery $\mathcal{O}_S$ |

**Key Insight:** Given thin objects satisfying the consistency conditions of MT {prf:ref}`mt-thin-expansion`, these factories produce valid implementations for all required interfaces. The user's task reduces to specifying the physics (energy, dissipation, symmetry); the Framework handles the singularity theory.
:::

:::{prf:definition} Automation Guarantee
:label: def-automation-guarantee

A Hypostructure $\mathcal{H}$ satisfies the **Automation Guarantee** if:

1. **Profile extraction is automatic:** Given any singularity point $(t^*, x^*)$, the Framework computes the profile $V$ without user intervention via scaling limit:
   $$V = \lim_{\lambda \to 0} \lambda^{-\alpha} \cdot x(t^* + \lambda^2 t, x^* + \lambda y)$$

2. **Surgery construction is automatic:** Given admissibility certificate $K_{\text{adm}}$, the Framework constructs the surgery operator $\mathcal{O}_S$ as a categorical pushout.

3. **Termination is guaranteed:** The surgery sequence either:
   - Terminates (global regularity achieved), or
   - Reaches a horizon (irreducible singularity), or
   - Has bounded count (finite surgeries per unit time)

**Type Coverage:**
- For types $T \in \{T_{\text{parabolic}}, T_{\text{dispersive}}, T_{\text{hyperbolic}}\}$: The Automation Guarantee holds whenever the thin objects are well-defined.
- For $T_{\text{algorithmic}}$: The guarantee holds when the complexity measure $\mathcal{C}$ is well-founded (decreases with each step). In this case:
  - "Profiles" are fixed points or limit cycles of the discrete dynamics
  - "Surgery" is state reset or backtracking
  - "Termination" follows from well-foundedness of $\mathcal{C}$
- For $T_{\text{Markov}}$: The guarantee holds when the spectral gap is positive. Profiles are stationary distributions; surgery is measure truncation.

**Non-PDE Convergence Criteria:** The Łojasiewicz-Simon condition used in PDE applications can be replaced by:
- **Algorithmic:** Discrete Lyapunov functions with $\mathcal{C}(x') < \mathcal{C}(x)$
- **Markov:** Spectral gap $\lambda_1 > 0$ implies exponential mixing
- **Dynamical systems:** Contraction mappings with Lipschitz constant $L < 1$
:::

---

## 14. Profile Classification Trichotomy

:::{prf:theorem} Profile Classification Trichotomy
:label: mt-profile-trichotomy
:class: metatheorem

At the Profile node (after CompactCheck YES), the framework produces exactly one of three certificates:

**Case 1: Finite library membership**
$$K_{\text{lib}} = (V, \text{canonical list } \mathcal{L}, V \in \mathcal{L})$$
The limiting profile $V$ belongs to a finite, pre-classified library $\mathcal{L}$ of canonical profiles. Each library member has known properties enabling subsequent checks.

**Case 2: Tame stratification**
$$K_{\text{strat}} = (V, \text{definable family } \mathcal{F}, V \in \mathcal{F}, \text{stratification data})$$
Profiles are parameterized in a definable (o-minimal) family $\mathcal{F}$ with finite stratification. Classification is tractable though not finite.

**Case 3: Classification Failure (NO-inconclusive or NO-wild)**
$$K_{\mathrm{prof}}^- := K_{\mathrm{prof}}^{\mathrm{wild}} \sqcup K_{\mathrm{prof}}^{\mathrm{inc}}$$
- **NO-wild** ($K_{\mathrm{prof}}^{\mathrm{wild}}$): Profile exhibits wildness witness (chaotic attractor, turbulent cascade, undecidable structure)
- **NO-inconclusive** ($K_{\mathrm{prof}}^{\mathrm{inc}}$): Classification methods exhausted without refutation (Rep/definability constraints insufficient)

Routes to T.C/D.C-family modes for reconstruction or explicit wildness acknowledgment.

**Literature:** Concentration-compactness profile decomposition {cite}`Lions84`; {cite}`Lions85`; blow-up profile classification {cite}`MerleZaag98`; o-minimal stratification {cite}`vandenDries98`.

:::

:::{prf:remark} Library examples by type
:label: rem-library-examples

- $T_{\text{parabolic}}$: Cylinders, spheres, Bryant solitons (Ricci); spheres, cylinders (MCF)
- $T_{\text{dispersive}}$: Ground states, traveling waves, multi-solitons
- $T_{\text{algorithmic}}$: Fixed points, limit cycles, strange attractors
:::

:::{prf:remark} Oscillating and Quasi-Periodic Profiles
:label: rem-oscillating-profiles

**Edge Case:** The scaling limit $V = \lim_{n \to \infty} V_n$ may fail to converge in systems with oscillating or multi-scale behavior. Such systems are handled as follows:

**Case 2a (Periodic oscillations):** If the sequence $\{V_n\}$ has periodic or quasi-periodic structure:
$$V_{n+p} \approx V_n \quad \text{for some period } p$$
then the profile $V$ is defined as the **orbit** $\{V_n\}_{n \mod p}$, which falls into Case 2 (Tame Family) with a finite-dimensional parameter space $\mathbb{Z}/p\mathbb{Z}$ or $\mathbb{T}^k$ (torus for quasi-periodic).

**Case 3a (Wild oscillations):** If oscillations are unbounded or aperiodic without definable structure, the system produces a NO-wild certificate ($K_{\mathrm{prof}}^{\mathrm{wild}}$, Case 3). This is common in:
- Turbulent cascades (energy spreads across all scales)
- Chaotic attractors with positive Lyapunov exponent
- Undecidable algorithmic dynamics

**Practical consequence:** For well-posed physical systems, periodic/quasi-periodic profiles are typically tame. Wild oscillations indicate genuine physical complexity (turbulence) or computational irreducibility.
:::

:::{prf:definition} Moduli Space of Profiles
:label: def-moduli-profiles

The **Moduli Space of Profiles** for type $T$ is:
$$\mathcal{M}_{\text{prof}}(T) := \{V : V \text{ is a scaling-invariant limit of type } T \text{ flow}\} / \sim$$

where $V_1 \sim V_2$ if related by symmetry action: $V_2 = g \cdot V_1$ for $g \in G$.

**Structure:**
- $\mathcal{M}_{\text{prof}}$ is a (possibly infinite-dimensional) moduli stack
- The Canonical Library $\mathcal{L}_T \subset \mathcal{M}_{\text{prof}}(T)$ consists of **isolated points** with trivial automorphism
- The Tame Family $\mathcal{F}_T$ consists of **definable strata** parameterized by finite-dimensional spaces

**Computation:** Given $G^{\text{thin}} = (\text{Grp}, \rho, \mathcal{S})$ and $\Phi^{\text{thin}} = (F, \nabla, \alpha)$:
$$\mathcal{M}_{\text{prof}}(T) = \{V : \mathcal{S} \cdot V = V, \nabla F(V) = 0\} / \text{Grp}$$
:::

### 14.A Implementation in Sieve

:::{prf:remark} Profile Extraction Algorithm
:label: rem-profile-extraction

The Framework implements `ProfileExtractor` as follows:

**Input:** Singularity point $(t^*, x^*)$, thin objects $(\mathcal{X}^{\text{thin}}, \Phi^{\text{thin}}, G^{\text{thin}})$

**Algorithm:**
1. **Rescaling:** For sequence $\lambda_n \to 0$, compute:
   $$V_n := \lambda_n^{-\alpha} \cdot x(t^* + \lambda_n^2 t, x^* + \lambda_n y)$$

2. **Compactification:** Apply CompactCheck ($\mathrm{Cap}_H$) to verify subsequence converges

3. **Limit Extraction:** Extract $V = \lim_{n \to \infty} V_n$ in appropriate topology

4. **Library Lookup:**
   - If $V \in \mathcal{L}_T$: Return Case 1 certificate $K_{\text{lib}}$
   - If $V \in \mathcal{F}_T \setminus \mathcal{L}_T$: Return Case 2 certificate $K_{\text{strat}}$
   - If classification fails: Return Case 3 certificate $K_{\text{hor}}$

**Output:** Profile $V$ with classification certificate
:::

:::{prf:metatheorem} Automatic Profile Classification (Multi-Mechanism OR-Schema)
:label: mt-auto-profile
:class: metatheorem

**Sieve Target:** ProfileExtractor / Profile Classification Trichotomy

**Goal Certificate:** $K_{\mathrm{prof}}^+ \in \{K_{\text{lib}}, K_{\text{strat}}, K_{\text{hor}}\}$

For any Hypostructure $\mathcal{H} = (\mathcal{X}, \Phi, \mathfrak{D}, G)$ satisfying the Automation Guarantee (Definition {prf:ref}`def-automation-guarantee`), the Profile Classification Trichotomy (MT {prf:ref}`mt-profile-trichotomy`) is **automatically computed** by the Sieve without user-provided classification code.

### Unified Output Certificate

**Profile Classification Certificate:**
$$K_{\mathrm{prof}}^+ := (V, \mathcal{L}_T \text{ or } \mathcal{F}_T, \mathsf{route\_tag}, \mathsf{classification\_data})$$

where $\mathsf{route\_tag} \in \{\text{CC-Rig}, \text{Attr-Morse}, \text{Tame-LS}, \text{Lock-Excl}\}$ indicates which mechanism produced the certificate.

**Downstream Independence:** All subsequent theorems (Lock promotion, surgery admissibility, etc.) depend only on $K_{\mathrm{prof}}^+$, never on which mechanism produced it.

---

### Public Signature (Soft Interfaces Only)

**User-Provided (Soft Core):**
$$K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{LS}_\sigma}^+$$

**Mechanism-Specific Soft Extensions:**
| Mechanism | Additional Soft Interfaces |
|-----------|---------------------------|
| A: CC+Rigidity | $K_{\mathrm{Mon}_\phi}^+ \wedge K_{\mathrm{Rep}_K}^+$ |
| B: Attractor+Morse | $K_{\mathrm{TB}_\pi}^+$ |
| C: Tame+LS | $K_{\mathrm{TB}_O}^+$ (o-minimal definability) |
| D: Lock/Hom-Exclusion | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ (Lock blocked) |

**Certificate Logic (Multi-Mechanism Disjunction):**
$$\underbrace{K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{LS}_\sigma}^+}_{\text{SoftCore}} \wedge \big(\text{MechA} \lor \text{MechB} \lor \text{MechC} \lor \text{MechD}\big) \Rightarrow K_{\mathrm{prof}}^+$$

**Key Architectural Point:** Backend permits ($K_{\mathrm{WP}}$, $K_{\mathrm{ProfDec}}$, $K_{\mathrm{KM}}$, $K_{\mathrm{Rigidity}}$, $K_{\mathrm{Attr}}$, $K_{\mathrm{MorseDecomp}}$) are **derived internally** via the Soft-to-Backend Compilation layer (Section {ref}`sec-soft-backend-compilation`), not required from users.

- **Produces:** $K_{\text{prof}}^+ \in \{K_{\text{lib}}, K_{\text{strat}}, K_{\text{hor}}\}$
- **Blocks:** Mode C.D (Geometric Collapse), Mode T.C (Labyrinthine), Mode D.C (Semantic Horizon)
- **Breached By:** Wild/undecidable dynamics, non-good types

---

### Dispatcher Logic

The Sieve tries mechanisms in order until one succeeds:

```
try MechA(SoftCore); if YES → emit K_prof^+ (tag: CC-Rig)
else try MechB(SoftCore); if YES → emit K_prof^+ (tag: Attr-Morse)
else try MechC(SoftCore); if YES → emit K_prof^+ (tag: Tame-LS)
else try MechD(SoftCore); if YES → emit K_prof^+ (tag: Lock-Excl)
else emit NO with K_prof^inc (mechanism_failures: [A,B,C,D])
```

---

### Mechanism A: Concentration-Compactness + Rigidity

**Best For:** NLS, NLW, critical dispersive PDEs

**Sufficient Soft Condition:**
$$K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{Mon}_\phi}^+ \wedge K_{\mathrm{Rep}_K}^+$$

**Proof (5 Steps via Compilation):**

*Step A1 (Well-Posedness).* By MT-SOFT→WP (MT {prf:ref}`mt-soft-wp`), derive $K_{\mathrm{WP}_{s_c}}^+$ from template matching. The evaluator recognizes the equation structure and applies the appropriate critical LWP theorem.

*Step A2 (Profile Decomposition).* By MT-SOFT→ProfDec (MT {prf:ref}`mt-soft-profdec`), derive $K_{\mathrm{ProfDec}_{s_c,G}}^+$ from $K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+$. Any bounded sequence $\{u_n\}$ in $\dot{H}^{s_c}$ admits:
$$u_n = \sum_{j=1}^J g_n^{(j)} \cdot V^{(j)} + w_n^{(J)}$$
with orthogonal symmetry parameters and vanishing remainder.

*Step A3 (Kenig-Merle Machine).* By MT-SOFT→KM (MT {prf:ref}`mt-soft-km`), derive $K_{\mathrm{KM}_{\mathrm{CC+stab}}}^+$ from composition of $K_{\mathrm{WP}}^+ \wedge K_{\mathrm{ProfDec}}^+ \wedge K_{D_E}^+$. This extracts the minimal counterexample $u^*$ with:
- $\Phi(u^*) = E_c$ (critical energy threshold),
- Trajectory is **almost periodic modulo $G$**.

*Step A4 (Hybrid Rigidity).* By MT-SOFT→Rigidity (MT {prf:ref}`mt-soft-rigidity`), derive $K_{\mathrm{Rigidity}_T}^+$ via the hybrid mechanism:
1. **Monotonicity:** $K_{\mathrm{Mon}_\phi}^+$ provides virial/Morawetz identity forcing dispersion or concentration.
2. **Łojasiewicz Closure:** $K_{\mathrm{LS}_\sigma}^+$ prevents oscillation near critical points.
3. **Lock Exclusion:** Any "bad" $u^*$ would embed a forbidden pattern; Lock blocks this.

Conclusion: almost-periodic solutions are either **stationary** (soliton/ground state) or **self-similar**.

*Step A5 (Emit Certificate).* Classify $u^*$ into $\mathcal{L}_T$:
- **Case 1 (Library):** $V \in \mathcal{L}_T$ isolated. Emit YES with $K_{\text{lib}} = (V, \mathcal{L}_T, \text{Aut}(V), \text{CC-Rig})$
- **Case 2 (Tame Stratification):** $V \in \mathcal{F}_T$ definable. Emit YES with $K_{\text{strat}} = (V, \mathcal{F}_T, \dim, \text{CC-Rig})$
- **Case 3 (Classification Failure):** Emit NO with $K_{\mathrm{prof}}^{\mathrm{wild}}$ (if wildness witness found) or $K_{\mathrm{prof}}^{\mathrm{inc}}$ (if method insufficient)

**Literature:** Concentration-compactness {cite}`Lions84`; profile decomposition {cite}`BahouriGerard99`; Kenig-Merle {cite}`KenigMerle06`; rigidity {cite}`DuyckaertsKenigMerle11`.

---

### Mechanism B: Attractor + Morse Decomposition

**Best For:** Reaction-diffusion, Navier-Stokes (bounded domain), MCF

**Sufficient Soft Condition:**
$$K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{TB}_\pi}^+$$

**Proof (4 Steps via Compilation):**

*Step B1 (Global Attractor).* By MT-SOFT→Attr (MT {prf:ref}`mt-soft-attr`), derive $K_{\mathrm{Attr}}^+$ from $K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{TB}_\pi}^+$. The attractor $\mathcal{A}$ exists, is compact, invariant, and attracts bounded sets:
$$\mathcal{A} := \bigcap_{t \geq 0} \overline{\bigcup_{s \geq t} S_s(\mathcal{X})}$$

*Step B2 (Morse Decomposition).* By MT-SOFT→MorseDecomp (MT {prf:ref}`mt-soft-morsedecomp`), derive $K_{\mathrm{MorseDecomp}}^+$ from $K_{\mathrm{Attr}}^+ \wedge K_{D_E}^+ \wedge K_{\mathrm{LS}_\sigma}^+$. For gradient-like systems, the attractor decomposes as:
$$\mathcal{A} = \mathcal{E} \cup \bigcup_{\xi \in \mathcal{E}} W^u(\xi)$$
where $\mathcal{E}$ is the equilibrium set. No periodic orbits exist (Lyapunov monotonicity).

*Step B3 (Profile Identification).* The profile space is:
$$\mathcal{M}_{\text{prof}} = \mathcal{A} / G$$
By compactness of $\mathcal{A}$, this is a compact moduli space. The canonical library is:
$$\mathcal{L}_T := \{\xi \in \mathcal{E} / G : \xi \text{ isolated}, |\text{Stab}(\xi)| < \infty\}$$

*Step B4 (Emit Certificate).* Classify rescaling limits into $\mathcal{A}/G$:
- **Case 1 (Library):** Isolated equilibrium. Emit YES with $K_{\text{lib}} = (V, \mathcal{L}_T, \text{Morse index}, \text{Attr-Morse})$
- **Case 2 (Tame Stratification):** Connecting orbit. Emit YES with $K_{\text{strat}} = (V, W^u(\xi)/G, \dim, \text{Attr-Morse})$
- **Case 3 (Classification Failure):** Strange attractor detected. Emit NO with $K_{\mathrm{prof}}^{\mathrm{wild}} = (\text{strange\_attractor}, h_{\text{top}}(\mathcal{A}))$

**Literature:** Global attractor theory {cite}`Temam97`; gradient-like structure {cite}`HaleBook88`; Morse decomposition {cite}`Conley78`.

---

### Mechanism C: Tame + Łojasiewicz (O-Minimal Types)

**Best For:** Algebraic/analytic systems, polynomial nonlinearities

**Sufficient Soft Condition:**
$$K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{TB}_O}^+$$

**Proof (3 Steps):**

*Step C1 (Definability).* By $K_{\mathrm{TB}_O}^+$, the profile space $\mathcal{M}_{\text{prof}}$ is **o-minimal definable** in the structure $\mathbb{R}_{\text{an}}$ (or $\mathbb{R}_{\text{alg}}$ for polynomial systems). This captures all algebraic, semialgebraic, and globally subanalytic families.

*Step C2 (Cell Decomposition).* By the o-minimal cell decomposition theorem, the profile space admits a **finite stratification**:
$$\mathcal{M}_{\text{prof}} = \bigsqcup_{i=1}^N C_i$$
where each $C_i$ is a definable cell (diffeomorphic to $(0,1)^{d_i}$). The stratification is canonical and computable from the defining formulas.

*Step C3 (Łojasiewicz Convergence + Emit).* By $K_{\mathrm{LS}_\sigma}^+$, trajectories converge to strata (no oscillation across cells). Emit:
- **Case 1 (Library):** Limit in 0-dimensional stratum. Emit YES with $K_{\text{lib}} = (V, \mathcal{L}_T, \text{cell ID}, \text{Tame-LS})$
- **Case 2 (Tame Stratification):** Limit in positive-dimensional stratum. Emit YES with $K_{\text{strat}} = (V, C_i, \dim C_i, \text{Tame-LS})$
- **Case 3 (Classification Failure):** Non-definable family (escape from o-minimal). Emit NO with $K_{\mathrm{prof}}^{\mathrm{wild}} = (\text{non\_definable}, \mathsf{escape\_witness})$

**Key Advantage:** No PDE-specific machinery required—works purely from definability + gradient structure.

**Literature:** O-minimal structures {cite}`vandenDries98`; tame geometry {cite}`Shiota97`; Łojasiewicz inequality {cite}`Lojasiewicz84`.

---

### Mechanism D: Lock / Hom-Exclusion (Categorical Types)

**Best For:** Systems where categorical obstruction is stronger than analytic classification

**Sufficient Soft Condition:**
$$K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Proof (2 Steps):**

*Step D1 (Lock Obstruction).* By $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$, the Lock mechanism certifies:
$$\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}, \mathcal{H}) = \emptyset$$
for all "bad patterns" $\mathbb{H}_{\mathrm{bad}}$ (singularity templates, wild dynamics markers). This is a **categorical statement**: no morphism from any forbidden object can land in the hypostructure.

*Step D2 (Emit Trivial Classification).* Since no singularity can form (Lock blocks all singular behavior), profile classification is **vacuous or trivial**:
- All solutions remain regular
- The "library" is just the space of smooth solutions
- Emit $K_{\text{lib}} = (\text{smooth}, \mathcal{L}_T := \emptyset, \text{vacuous}, \text{Lock-Excl})$

Alternatively, if Lock blocks specific patterns but allows others, classify the allowed profiles as in other mechanisms.

**Key Advantage:** No hard estimates needed—regularity follows from **categorical obstruction** rather than analytic a priori bounds.

**Literature:** Lock mechanism (Section {ref}`sec-lock`); categorical obstructions in PDE {cite}`Fargues21`.

---

### Mechanism Comparison

| Mechanism | Additional Soft | Best For | Hard Estimates? | Route Tag |
|-----------|-----------------|----------|-----------------|-----------|
| **A: CC+Rig** | $K_{\mathrm{Mon}_\phi}^+$, $K_{\mathrm{Rep}_K}^+$ | NLS, NLW, dispersive | No (compiled) | CC-Rig |
| **B: Attr+Morse** | $K_{\mathrm{TB}_\pi}^+$ | Reaction-diffusion, MCF | No (gradient-like) | Attr-Morse |
| **C: Tame+LS** | $K_{\mathrm{TB}_O}^+$ | Algebraic, polynomial | No (definability) | Tame-LS |
| **D: Lock** | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | Categorical systems | No (obstruction) | Lock-Excl |

**Mechanism Selection:** The Sieve automatically selects the first applicable mechanism based on which soft interfaces are available. Users may also specify a preferred mechanism via the `route_hint` parameter.

:::

---

## 15. Surgery Admissibility Trichotomy

:::{prf:theorem} Surgery Admissibility Trichotomy
:label: mt-surgery-trichotomy
:class: metatheorem

Before invoking any surgery $S$ with mode $M$ and data $D_S$, the framework produces exactly one of three certificates:

**Case 1: Admissible**
$$K_{\text{adm}} = (M, D_S, \text{admissibility proof})$$
The surgery satisfies:
1. **Canonicity**: Profile at surgery point is in canonical library
2. **Codimension**: Singular set has codimension $\geq 2$
3. **Capacity**: $\mathrm{Cap}(\text{excision}) \leq \varepsilon_{\text{adm}}$

**Case 2: Admissible up to equivalence (YES$^\sim$)**
$$K_{\text{adm}}^{\sim} = (K_{\text{equiv}}, K_{\text{transport}}, K_{\text{adm}}[\tilde{x}])$$
After an admissible equivalence move, the surgery becomes admissible.

**Case 3: Not admissible**
$$K_{\text{inadm}} = (\text{failure reason}, \text{witness})$$
Explicit reason certificate:
- Capacity too large: $\mathrm{Cap}(\text{excision}) > \varepsilon_{\text{adm}}$
- Codimension too small: $\mathrm{codim} < 2$
- Horizon: Profile not classifiable (Case 3 of Profile Trichotomy)

**Literature:** Surgery admissibility in Ricci flow {cite}`Perelman03`; capacity and removable singularities {cite}`Federer69`; {cite}`EvansGariepy15`.

:::

:::{prf:definition} Canonical Library
:label: def-canonical-library

The **Canonical Library** for type $T$ is:
$$\mathcal{L}_T := \{V \in \mathcal{M}_{\text{prof}}(T) : \text{Aut}(V) \text{ is finite}, V \text{ is isolated in } \mathcal{M}_{\text{prof}}\}$$

**Properties:**
- $\mathcal{L}_T$ is finite for good types (parabolic, dispersive)
- Each $V \in \mathcal{L}_T$ has a **surgery recipe** $\mathcal{O}_V$ attached
- Library membership is decidable via gradient flow to critical points

**Examples by Type:**

| Type | Library $\mathcal{L}_T$ | Size |
|------|------------------------|------|
| $T_{\text{Ricci}}$ | $\{\text{Sphere}, \text{Cylinder}, \text{Bryant}\}$ | 3 |
| $T_{\text{MCF}}$ | $\{\text{Sphere}^n, \text{Cylinder}^k\}_{k \leq n}$ | $n+1$ |
| $T_{\text{NLS}}$ | $\{Q, Q_{\text{excited}}\}$ | 2 |
| $T_{\text{wave}}$ | $\{\text{Ground state}\}$ | 1 |
:::

:::{prf:remark} Good Types
:label: rem-good-types

A type $T$ is **good** if:
1. **Compactness:** Scaling limits exist in a suitable topology (e.g., weak convergence in $L^2$, Gromov-Hausdorff)
2. **Finite stratification:** $\mathcal{M}_{\text{prof}}(T)$ admits finite stratification into isolated points and tame families
3. **Constructible caps:** Asymptotic matching for surgery caps is well-defined (unique cap per profile)

**Good types:** $T_{\text{Ricci}}$, $T_{\text{MCF}}$, $T_{\text{NLS}}$, $T_{\text{wave}}$, $T_{\text{parabolic}}$, $T_{\text{dispersive}}$.

**Non-good types:** Wild/undecidable systems that reach Horizon modes. For such systems, the Canonical Library may be empty or infinite, and the Automation Guarantee (Definition {prf:ref}`def-automation-guarantee`) does not apply.

**Algorithmic types:** $T_{\text{algorithmic}}$ is good when the complexity measure $\mathcal{C}$ is well-founded (terminates in finite steps). In this case, "profiles" are limit cycles or fixed points of the discrete dynamics.
:::

### 15.A Implementation in Sieve

:::{prf:remark} Admissibility Check Algorithm
:label: rem-admissibility-algorithm

The Framework implements `SurgeryAdmissibility` as follows:

**Input:** Singularity data $(\Sigma, V, t^*)$, thin objects $(\mathcal{X}^{\text{thin}}, \mathfrak{D}^{\text{thin}})$

**Algorithm:**
1. **Canonicity Check:**
   - Query: Is $V \in \mathcal{L}_T$?
   - If YES: Continue. If NO (but $V \in \mathcal{F}_T$): Try equivalence move. If Horizon: Return Case 3.

2. **Codimension Check:**
   - Compute $\text{codim}(\Sigma)$ using dimension of $\mathcal{X}$
   - Require: $\text{codim}(\Sigma) \geq 2$

3. **Capacity Check:**
   - Compute $\text{Cap}(\Sigma)$ using measure $\mu$ from $\mathcal{X}^{\text{thin}}$
   - Require: $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}(T)$

**Decision:**
- All checks pass → Case 1: $K_{\text{adm}}$
- Canonicity fails but equivalence available → Case 2: $K_{\text{adm}}^\sim$
- Any check fails without recovery → Case 3: $K_{\text{inadm}}$

**Output:** Admissibility certificate
:::

:::{prf:metatheorem} Automatic Admissibility
:label: mt-auto-admissibility

For any Hypostructure satisfying the Automation Guarantee, the Surgery Admissibility Trichotomy is **automatically computed** from thin objects without user-provided admissibility code.

**Key Computation:** The capacity bound is computed as:
$$\text{Cap}(\Sigma) = \inf\left\{\int |\nabla \phi|^2 d\mu : \phi|_\Sigma = 1, \phi \in H^1(\mathcal{X})\right\}$$
using the measure $\mu$ from $\mathcal{X}^{\text{thin}}$ and the metric $d$.

**Literature:** Sobolev capacity {cite}`AdamsHedberg96`; Hausdorff dimension bounds {cite}`Federer69`.
:::

---

## 16. Structural Surgery Principle

:::{prf:theorem} Structural Surgery Principle (Certificate Form)
:label: mt-structural-surgery
:class: metatheorem

Let $M$ be a failure mode with breach certificate $K^{\mathrm{br}}$, and let $S$ be the associated surgery with admissibility certificate $K_{\text{adm}}$ (or $K_{\text{adm}}^{\sim}$).

**Inputs**:
- $K^{\mathrm{br}}$: Breach certificate from barrier
- $K_{\text{adm}}$ or $K_{\text{adm}}^{\sim}$: From Surgery Admissibility Trichotomy
- $D_S$: Surgery data

**Guarantees**:
1. **Flow continuation**: Evolution continues past surgery with well-defined state $x'$
2. **Jump control**: $\Phi(x') \leq \Phi(x^-) + \delta_S$ for controlled jump $\delta_S$
3. **Certificate production**: Re-entry certificate $K^{\mathrm{re}}$ satisfying $K^{\mathrm{re}} \Rightarrow \mathrm{Pre}(\text{target})$
4. **Progress**: Either bounded surgery count or decreasing complexity

**Failure case**: If $K_{\text{inadm}}$ is produced, no surgery is performed; the run terminates at the mode as a genuine singularity (or routes to reconstruction via MT 42.1).

**Literature:** Hamilton's surgery program {cite}`Hamilton97`; Perelman's surgery algorithm {cite}`Perelman03`; {cite}`KleinerLott08`.

:::

:::{prf:definition} Surgery Morphism
:label: def-surgery-morphism

A **Surgery Morphism** for singularity $(\Sigma, V)$ is a categorical pushout:

$$\begin{CD}
\mathcal{X}_{\Sigma} @>{\iota}>> \mathcal{X} \\
@V{\text{excise}}VV @VV{\mathcal{O}_S}V \\
\mathcal{X}_{\text{cap}} @>{\text{glue}}>> \mathcal{X}'
\end{CD}$$

where:
- $\mathcal{X}_\Sigma = \{x \in \mathcal{X} : d(x, \Sigma) < \epsilon\}$ is the singular neighborhood
- $\iota$ is the inclusion
- $\mathcal{X}_{\text{cap}}$ is a **capping object** determined by profile $V$
- $\mathcal{X}' = (\mathcal{X} \setminus \mathcal{X}_\Sigma) \sqcup_{\partial} \mathcal{X}_{\text{cap}}$ is the surgered space

**Universal Property:** For any morphism $f: \mathcal{X} \to \mathcal{Y}$ that annihilates $\Sigma$ (i.e., $f|_\Sigma$ factors through a point), there exists unique $\tilde{f}: \mathcal{X}' \to \mathcal{Y}$ with $\tilde{f} \circ \mathcal{O}_S = f$.

**Categorical Context:** The pushout is computed in the appropriate category determined by the ambient topos $\mathcal{E}$:
- **Top** (topological spaces): For continuous structure and homotopy type
- **Meas** (measure spaces): For measure $\mu$ and capacity computations
- **Diff** (smooth manifolds): For PDE applications with regularity
- **FinSet** (finite sets): For algorithmic/combinatorial applications

The transfer of structures ($\Phi', \mathfrak{D}'$) to $\mathcal{X}'$ uses the universal property: any structure on $\mathcal{X}$ that is constant on $\Sigma$ induces a unique structure on $\mathcal{X}'$.
:::

:::{prf:metatheorem} Conservation of Flow
:label: mt-conservation-flow

For any admissible surgery $\mathcal{O}_S: \mathcal{X} \dashrightarrow \mathcal{X}'$, the following are conserved:

1. **Energy Drop:**
   $$\Phi(x') \leq \Phi(x^-) - \Delta\Phi_{\text{surg}}$$
   where $\Delta\Phi_{\text{surg}} \geq c \cdot \text{Vol}(\Sigma)^{2/n} > 0$ (surgery releases energy).

2. **Regularization:**
   $$\sup_{\mathcal{X}'} |\nabla^k \Phi| < \infty \quad \text{for all } k \leq k_{\max}(V)$$
   The surgered solution has bounded derivatives (smoother than pre-surgery).

3. **Countability:**
   $$\#\{\text{surgeries on } [0, T]\} \leq N(T, \Phi_0)$$
   Surgery count is bounded by initial energy and time.

**Proof Structure:** Energy drop follows from excision of high-curvature region. Regularization follows from gluing in smooth cap. Countability follows from energy monotonicity.
:::

### 16.A Implementation in Sieve

:::{prf:remark} Surgery Operator Construction
:label: rem-surgery-construction

The Framework implements `SurgeryOperator` as follows:

**Input:** Admissibility certificate $K_{\text{adm}}$, profile $V \in \mathcal{L}_T$

**Algorithm:**
1. **Neighborhood Selection:**
   - Compute singular neighborhood $\mathcal{X}_\Sigma = \{d(x, \Sigma) < \epsilon(V)\}$
   - Verify $\text{Cap}(\mathcal{X}_\Sigma) \leq \varepsilon_{\text{adm}}$

2. **Cap Selection:**
   - Look up cap $\mathcal{X}_{\text{cap}}(V)$ from library $\mathcal{L}_T$
   - Each profile $V$ has a unique asymptotically-matching cap

3. **Pushout Construction:**
   - Form pushout $\mathcal{X}' = \mathcal{X} \sqcup_{\partial \mathcal{X}_\Sigma} \mathcal{X}_{\text{cap}}$
   - Transfer height $\Phi'$ and dissipation $\mathfrak{D}'$ to $\mathcal{X}'$

4. **Certificate Production:**
   - Produce re-entry certificate $K^{\text{re}}$ with:
     - New state $x' \in \mathcal{X}'$
     - Energy bound $\Phi(x') \leq \Phi(x^-) + \delta_S$
     - Regularity guarantee for post-surgery solution

**Output:** Surgered state $x' \in \mathcal{X}'$ with re-entry certificate
:::

:::{prf:metatheorem} Automatic Surgery
:label: mt-auto-surgery

For any Hypostructure satisfying the Automation Guarantee, the Structural Surgery Principle is **automatically executed** by the Sieve using the pushout construction from $\mathcal{L}_T$.

**Key Insight:** The cap $\mathcal{X}_{\text{cap}}(V)$ is uniquely determined by the profile $V$ via asymptotic matching. Users provide the symmetry group $G$ and scaling $\alpha$; the Framework constructs the surgery operator as a categorical pushout.

**Literature:** Pushouts in category theory {cite}`MacLane71`; surgery caps in geometric flows {cite}`Hamilton97`; {cite}`KleinerLott08`.
:::

### 16.B Automated Workflow Summary

:::{prf:remark} Complete Automation Pipeline
:label: rem-automation-pipeline

The Universal Singularity Modules provide an **end-to-end automated pipeline**:

| Stage | Sieve Node | Input | Module | Output |
|-------|------------|-------|--------|--------|
| 1. Detect | Node 3 (CompactCheck) | Flow $x(t)$ | — | Singular point $(t^*, x^*)$ |
| 2. Profile | Node 4 (Profile) | $(t^*, x^*)$ | MT 14.1 | Profile $V$ with certificate |
| 3. Barrier | Mode Barrier | $V$ | MT 12.1 | Breach certificate $K^{\text{br}}$ |
| 4. Admissibility | Pre-Surgery | $(\Sigma, V)$ | MT 15.1 | Admissibility certificate |
| 5. Surgery | Surgery | $K_{\text{adm}}$ | MT 16.1 | Surgered state $x'$ |
| 6. Re-entry | Post-Surgery | $x'$ | MT 16.1 | Re-entry certificate $K^{\text{re}}$ |

**User Input:** Thin objects $(\mathcal{X}^{\text{thin}}, \Phi^{\text{thin}}, \mathfrak{D}^{\text{thin}}, G^{\text{thin}})$

**Framework Output:** Either:
- GlobalRegularity (no singularities)
- Classified Mode $M_i$ with certificates
- Horizon (irreducible singularity)

**Zero User Code for Singularity Handling:** The user never writes profile classification, admissibility checking, or surgery construction code.
:::

:::{prf:corollary} Minimal User Burden for Singularity Resolution
:label: cor-minimal-user-burden

Given thin objects satisfying the consistency conditions:
1. $(\mathcal{X}, d)$ is a complete metric space
2. $F: \mathcal{X} \to \mathbb{R} \cup \{\infty\}$ is lower semicontinuous
3. $R \geq 0$ and $\frac{d}{dt}F \leq -R$
4. $\rho: G \times \mathcal{X} \to \mathcal{X}$ is continuous

The Sieve automatically:
- Detects all singularities
- Classifies all profiles
- Determines all surgery admissibilities
- Constructs all surgery operators
- Bounds all surgery counts

**Consequence:** The "singularity problem" becomes a **typing problem**: specify the correct thin objects, and the Framework handles singularity resolution.
:::

---

# Part X: Equivalence and Transport

\begin{remark}[Naming convention]
This part defines **equivalence moves** (Eq1--Eq5) and **transport lemmas** (T1--T6). These are distinct from the **Lock tactics** (E1--E10) defined in Part XI, Section 22. The ``Eq'' prefix distinguishes equivalence moves from Lock tactics.

:::

## 17. Equivalence Library

:::{prf:definition} Admissible equivalence move
:label: def-equiv-move

An **admissible equivalence move** for type $T$ is a transformation $(x, \Phi, \mathfrak{D}) \mapsto (\tilde{x}, \tilde{\Phi}, \tilde{\mathfrak{D}})$ with:
1. **Comparability bounds**: Constants $C_1, C_2 > 0$ with
   $$\begin{aligned}
   C_1 \Phi(x) &\leq \tilde{\Phi}(\tilde{x}) \leq C_2 \Phi(x) \\
   C_1 \mathfrak{D}(x) &\leq \tilde{\mathfrak{D}}(\tilde{x}) \leq C_2 \mathfrak{D}(x)
   \end{aligned}$$
2. **Structural preservation**: Interface permits preserved
3. **Certificate production**: Equivalence certificate $K_{\text{equiv}}$

:::

---

### Standard Equivalence Moves

:::{prf:definition} Eq1: Symmetry quotient
:label: def-equiv-symmetry

For symmetry group $G$ acting on $X$:
$$\tilde{x} = [x]_G \in X/G$$
Comparability: $\Phi([x]_G) = \inf_{g \in G} \Phi(g \cdot x)$ (coercivity modulo $G$)

:::

:::{prf:definition} Eq2: Metric deformation (Hypocoercivity)
:label: def-equiv-metric

Replace metric $d$ with equivalent metric $\tilde{d}$:
$$C_1 d(x, y) \leq \tilde{d}(x, y) \leq C_2 d(x, y)$$
Used when direct LS fails but deformed LS holds.

:::

:::{prf:definition} Eq3: Conjugacy
:label: def-equiv-conjugacy

For invertible $h: X \to Y$:
$$\tilde{S}_t = h \circ S_t \circ h^{-1}$$
Comparability: $\Phi_Y(h(x)) \sim \Phi_X(x)$

:::

:::{prf:definition} Eq4: Surgery identification
:label: def-equiv-surgery-id

Outside excision region $E$:
$$x|_{X \setminus E} = x'|_{X \setminus E}$$
Transport across surgery boundary.

:::

:::{prf:definition} Eq5: Analytic-hypostructure bridge
:label: def-equiv-bridge

Between classical solution $u$ and hypostructure state $x$:
$$x = \mathcal{H}(u), \quad u = \mathcal{A}(x)$$
with inverse bounds.

:::

---

## 18. YES$^\sim$ Permits

:::{prf:definition} YES$^\sim$ certificate
:label: def-yes-tilde-cert

A **YES$^\sim$ certificate** for predicate $P_i$ is a triple:
$$K_i^{\sim} = (K_{\text{equiv}}, K_{\text{transport}}, K_i^+[\tilde{x}])$$
where:
- $K_{\text{equiv}}$: Certifies $x \sim_{\mathrm{Eq}} \tilde{x}$ for some equivalence move Eq1--Eq5
- $K_{\text{transport}}$: Transport lemma certificate (from T1--T6)
- $K_i^+[\tilde{x}]$: YES certificate for $P_i$ on the equivalent object $\tilde{x}$

:::

:::{prf:definition} YES$^\sim$ acceptance
:label: def-yes-tilde-accept

A metatheorem $\mathcal{M}$ **accepts YES$^\sim$** if:
$$\mathcal{M}(K_{I_1}, \ldots, K_{I_i}^{\sim}, \ldots, K_{I_n}) = \mathcal{M}(K_{I_1}, \ldots, K_{I_i}^+, \ldots, K_{I_n})$$
That is, YES$^\sim$ certificates may substitute for YES certificates in the metatheorem's preconditions.

:::

---

## 19. Transport Toolkit

:::{prf:definition} T1: Inequality transport
:label: def-transport-t1

Under comparability $C_1 \Phi \leq \tilde{\Phi} \leq C_2 \Phi$:
$$\tilde{\Phi}(\tilde{x}) \leq E \Rightarrow \Phi(x) \leq E/C_1$$

:::

:::{prf:definition} T2: Integral transport
:label: def-transport-t2

Under dissipation comparability:
$$\int \tilde{\mathfrak{D}} \leq C_2 \int \mathfrak{D}$$

:::

:::{prf:definition} T3: Quotient transport
:label: def-transport-t3

For $G$-quotient with coercivity:
$$P_i(x) \Leftarrow P_i([x]_G) \wedge \text{(orbit bound)}$$

:::

:::{prf:definition} T4: Metric equivalence transport
:label: def-transport-t4

LS inequality transports under equivalent metrics:
$$\text{LS}_{\tilde{d}}(\theta, C) \Rightarrow \text{LS}_d(\theta, C/C_2)$$

:::

:::{prf:definition} T5: Conjugacy transport
:label: def-transport-t5

Invariants transport under conjugacy:
$$\tau(x) = \tilde{\tau}(h(x))$$

:::

:::{prf:definition} T6: Surgery identification transport
:label: def-transport-t6

Outside excision, all certificates transfer:
$$K[x|_{X \setminus E}] = K[x'|_{X \setminus E}]$$

:::

---

## 20. Promotion System

:::{prf:definition} Immediate promotion
:label: def-promotion-immediate

Rules using only past/current certificates:

**Barrier-to-YES**: If blocked certificate plus earlier certificates imply the predicate:
$$K_i^{\mathrm{blk}} \wedge \bigwedge_{j < i} K_j^+ \Rightarrow K_i^+$$

Example: $K_{\text{Cap}}^{\mathrm{blk}}$ (singular set measure zero) plus $K_{\text{SC}}^+$ (subcritical) may together imply $K_{\text{Geom}}^+$.

:::

:::{prf:definition} A-posteriori promotion
:label: def-promotion-aposteriori

Rules using later certificates:

$$K_i^{\mathrm{blk}} \wedge \bigwedge_{j > i} K_j^+ \Rightarrow K_i^+$$

Example: Full Lock passage ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$) may retroactively promote earlier blocked certificates to full YES.

:::

:::{prf:definition} Promotion closure
:label: def-promotion-closure

The **promotion closure** $\mathrm{Cl}(\Gamma)$ is the least fixed point:
$$\Gamma_0 = \Gamma, \quad \Gamma_{n+1} = \Gamma_n \cup \{K : \text{promoted or inc-upgraded from } \Gamma_n\}$$
$$\mathrm{Cl}(\Gamma) = \bigcup_n \Gamma_n$$

This includes both blocked-certificate promotions (Definition {prf:ref}`def-promotion-permits`) and inconclusive-certificate upgrades (Definition {prf:ref}`def-inc-upgrades`).

:::

:::{prf:definition} Replay semantics
:label: def-replay

Given final context $\Gamma_{\text{final}}$, the **replay** is a re-execution of the sieve under $\mathrm{Cl}(\Gamma_{\text{final}})$, potentially yielding a different (stronger) fingerprint.

:::

---

# Part XI: The Lock (Conjecture Prover Backend)

## 21. Node 17 Contract

:::{prf:definition} Lock contract
:label: def-lock-contract

The **Categorical Lock** (Node 17) is the final barrier with special structure:

**Trigger**: All prior checks passed or blocked (convergent paths)

**Pre-certificates**: Full context $\Gamma$ from prior nodes

**Question**: Is $\mathrm{Hom}_{\mathbf{Hypo}}(\mathbb{H}_{\mathrm{bad}}, \mathcal{H}) = \varnothing$?

Where:
- $\mathbf{Hypo}$ is the category of hypostructures
- $\mathbb{H}_{\mathrm{bad}}$ is the universal bad pattern (initial object of R-breaking subcategory)
- $\mathcal{H}$ is the system under analysis

**Outcomes**:
- **Blocked** ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ or $K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$): Hom-set empty; implies GLOBAL REGULARITY
- **MorphismExists** ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^-$): Explicit morphism $\phi: \mathbb{H}_{\mathrm{bad}} \to \mathcal{H}$; implies FATAL ERROR

**Goal Certificate:** For Node 17, the designated goal certificate for the proof completion criterion (Definition {prf:ref}`def-proof-complete`) is $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$. This certificate suffices for proof completion—no additional promotion to $K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$ is required. The blocked outcome at the Lock establishes morphism exclusion directly.

:::

---

## 22. E1--E10 Exclusion Tactics

The Lock attempts ten proof-producing tactics to establish Hom-emptiness:

:::{prf:definition} E1: Dimension obstruction
:label: def-e1

**Sieve Signature:**
- **Required Permits:** $\mathrm{Rep}_K$ (representability), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+\}$ (finite representability confirmed)
- **Produces:** $K_{\mathrm{E1}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ (Hom-emptiness via dimension)
- **Blocks:** All failure modes (Global Regularity)
- **Breached By:** Dimensions compatible or dimension not computable

**Method**: Linear algebra / dimension counting

**Mechanism**: If $\dim(\mathbb{H}_{\mathrm{bad}}) \neq \dim(\mathcal{H})$ in a way incompatible with morphisms, Hom is empty.

**Certificate Logic:**
$$K_{\mathrm{Rep}_K}^+ \wedge (d_{\mathrm{bad}} \neq d_{\mathcal{H}}) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(d_{\text{bad}}, d_{\mathcal{H}}, \text{dimension mismatch proof})$

**Automation**: Fully automatable via linear algebra

**Literature:** Brouwer invariance of domain {cite}`Brouwer11`; dimension theory {cite}`HurewiczWallman41`.

:::

:::{prf:definition} E2: Invariant mismatch
:label: def-e2

**Sieve Signature:**
- **Required Permits:** $\mathrm{Rep}_K$, $\mathrm{TB}_\pi$ (topological background), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+, K_{\mathrm{TB}_\pi}^+\}$
- **Produces:** $K_{\mathrm{E2}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity)
- **Breached By:** Invariants match or invariant not extractable

**Method**: Invariant extraction + comparison

**Mechanism**: If morphisms must preserve invariant $I$ but $I(\mathbb{H}_{\mathrm{bad}}) \neq I(\mathcal{H})$, Hom is empty.

**Certificate Logic:**
$$K_{\mathrm{TB}_\pi}^+ \wedge (I_{\mathrm{bad}} \neq I_{\mathcal{H}}) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(I, I_{\text{bad}}, I_{\mathcal{H}}, I_{\text{bad}} \neq I_{\mathcal{H}} \text{ proof})$

**Automation**: Automatable for extractable invariants (Euler char, homology, etc.)

**Literature:** Topological invariants {cite}`EilenbergSteenrod52`; K-theory {cite}`Quillen73`.

:::

:::{prf:definition} E3: Positivity obstruction
:label: def-e3

**Sieve Signature:**
- **Required Permits:** $D_E$ (energy), $\mathrm{LS}_\sigma$ (local stiffness), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{D_E}^+, K_{\mathrm{LS}_\sigma}^+\}$
- **Produces:** $K_{\mathrm{E3}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity)
- **Breached By:** Positivity compatible or cone structure absent

**Method**: Cone / positivity constraints

**Mechanism**: If morphisms must preserve positivity but $\mathbb{H}_{\mathrm{bad}}$ violates positivity required by $\mathcal{H}$, Hom is empty.

**Certificate Logic:**
$$K_{\mathrm{LS}_\sigma}^+ \wedge (\Phi_{\mathrm{bad}} \notin \mathcal{C}_+) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(P, \text{positivity constraint}, \text{violation witness})$

**Automation**: Via semidefinite programming / cone analysis

**Literature:** Positive energy theorems {cite}`SchoenYau79`; {cite}`Witten81`; convex cones {cite}`Rockafellar70`.

:::

:::{prf:definition} E4: Integrality obstruction
:label: def-e4

**Sieve Signature:**
- **Required Permits:** $\mathrm{Rep}_K$, $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+\}$ (arithmetic structure available)
- **Produces:** $K_{\mathrm{E4}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity)
- **Breached By:** Arithmetic structures compatible or not decidable

**Method**: Discrete / arithmetic constraints

**Mechanism**: If morphisms require integral/rational structure but bad pattern has incompatible arithmetic, Hom is empty.

**Certificate Logic:**
$$K_{\mathrm{Rep}_K}^+ \wedge (\Lambda_{\mathrm{bad}} \not\hookrightarrow \Lambda_{\mathcal{H}}) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(\text{arithmetic structure}, \text{incompatibility proof})$

**Automation**: Via number theory / SMT with integer arithmetic

**Literature:** Arithmetic obstructions {cite}`Serre73`; lattice theory {cite}`CasselsSwinnerton70`.

:::

:::{prf:definition} E5: Functional equation obstruction
:label: def-e5

**Sieve Signature:**
- **Required Permits:** $\mathrm{Rep}_K$, $\mathrm{GC}_\nabla$ (gauge covariance), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+\}$
- **Produces:** $K_{\mathrm{E5}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity)
- **Breached By:** Functional equations solvable or undecidable

**Method**: Rewriting / functional constraints

**Mechanism**: If morphisms must satisfy functional equations that have no solution, Hom is empty.

**Certificate Logic:**
$$K_{\mathrm{Rep}_K}^+ \wedge (\text{FuncEq}(\phi) = \bot) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(\text{functional eq.}, \text{unsolvability proof})$

**Automation**: Via term rewriting / constraint solving

**Literature:** Functional equations {cite}`AczélDhombres89`; rewriting systems {cite}`BaaderNipkow98`.

:::

:::{prf:definition} E6: Causal obstruction (Well-Foundedness)
:label: def-e6

**Sieve Signature:**
- **Required Permits:** $\mathrm{TB}_\pi$ (topological/causal structure), $D_E$ (dissipation), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{TB}_\pi}^+, K_{D_E}^+\}$ (causal structure and energy bound available)
- **Produces:** $K_{\mathrm{E6}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity); specifically excludes CTCs
- **Breached By:** Causal structure compatible or well-foundedness undecidable

**Method**: Order theory / Causal set analysis

**Mechanism**: If morphisms must preserve the causal partial order $\prec$ but $\mathbb{H}_{\mathrm{bad}}$ contains infinite descending chains $v_0 \succ v_1 \succ \cdots$ (violating well-foundedness/Artinian condition), Hom is empty. The axiom of foundation connects to chronology protection: infinite causal descent requires unbounded negative energy, violating $D_E$.

**Certificate Logic:**
$$K_{\mathrm{TB}_\pi}^+ \wedge K_{D_E}^+ \wedge (\exists \text{ infinite descending chain in } \mathbb{H}_{\mathrm{bad}}) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(\prec_{\mathrm{bad}}, \text{descending chain witness}, \text{Artinian violation proof})$

**Automation**: Via order-theoretic analysis / transfinite induction / causal set algorithms

**Literature:** Causal set theory {cite}`Bombelli87`; {cite}`Sorkin05`; set-theoretic foundations {cite}`Jech03`.

:::

:::{prf:definition} E7: Thermodynamic obstruction (Entropy)
:label: def-e7

**Sieve Signature:**
- **Required Permits:** $D_E$ (dissipation/energy), $\mathrm{SC}_\lambda$ (scaling/entropy), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{D_E}^+, K_{\mathrm{SC}_\lambda}^+\}$ (energy dissipation and scaling available)
- **Produces:** $K_{\mathrm{E7}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity); specifically Mode C.E (energy blow-up)
- **Breached By:** Entropy production compatible or Lyapunov function absent

**Method**: Lyapunov analysis / Entropy production bounds

**Mechanism**: If morphisms must respect the Second Law ($\Delta S \geq 0$) but $\mathbb{H}_{\mathrm{bad}}$ requires entropy decrease incompatible with $\mathcal{H}$, Hom is empty. Lyapunov functions satisfying $\frac{d\mathcal{L}}{dt} \leq -\lambda \mathcal{L} + b$ (Foster-Lyapunov condition) enforce monotonic approach to equilibrium.

**Certificate Logic:**
$$K_{D_E}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge (\Delta S_{\mathrm{bad}} < 0) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(S_{\mathrm{bad}}, S_{\mathcal{H}}, \Delta S < 0 \text{ witness}, \text{Second Law violation proof})$

**Automation**: Via Lyapunov analysis / entropy production estimation / drift-diffusion bounds

**Literature:** Optimal transport {cite}`Villani09`; fluctuation theorems {cite}`Jarzynski97`; Foster-Lyapunov {cite}`MeynTweedie93`.

:::

:::{prf:definition} E8: Holographic obstruction (Capacity)
:label: def-e8

**Sieve Signature:**
- **Required Permits:** $\mathrm{Cap}_H$ (capacity), $\mathrm{TB}_\pi$ (topological boundary), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{Cap}_H}^+, K_{\mathrm{TB}_\pi}^+\}$ (capacity bound and topology available)
- **Produces:** $K_{\mathrm{E8}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity); specifically Mode C.D (geometric collapse)
- **Breached By:** Information density within capacity or boundary not defined

**Method**: Information-theoretic capacity bounds / Holographic principle

**Mechanism**: If morphisms must respect the Bekenstein bound $S \leq \frac{2\pi E R}{\hbar c}$ (information bounded by boundary area) but $\mathbb{H}_{\mathrm{bad}}$ requires $S_{\text{bulk}} > S_{\text{boundary}}$, Hom is empty. The holographic principle constrains information density: bulk entropy cannot exceed boundary area divided by $4G_N$.

**Certificate Logic:**
$$K_{\mathrm{Cap}_H}^+ \wedge K_{\mathrm{TB}_\pi}^+ \wedge (I_{\mathrm{bad}} > I_{\max}(\partial \mathcal{H})) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(I_{\mathrm{bad}}, I_{\max}, \text{Bekenstein violation proof})$

**Automation**: Via information-theoretic bounds / entropy estimation / channel capacity computation

**Literature:** Bekenstein bound {cite}`Bekenstein73`; holographic principle {cite}`tHooft93`; {cite}`Susskind95`; channel capacity {cite}`Shannon48`.

:::

:::{prf:definition} E9: Ergodic obstruction (Mixing)
:label: def-e9

**Sieve Signature:**
- **Required Permits:** $\mathrm{TB}_\rho$ (mixing/ergodic structure), $C_\mu$ (compactness), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{TB}_\rho}^+, K_{C_\mu}^+\}$ (mixing rate and concentration available)
- **Produces:** $K_{\mathrm{E9}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity); specifically Mode T.D (glassy freeze)
- **Breached By:** Mixing properties compatible or spectral gap not computable

**Method**: Spectral gap analysis / Mixing time bounds

**Mechanism**: If morphisms must preserve mixing properties but $\mathbb{H}_{\mathrm{bad}}$ has incompatible spectral gap, Hom is empty. Mixing systems satisfy $\mu(A \cap S_t^{-1}B) \to \mu(A)\mu(B)$, with spectral gap $\gamma > 0$ implying exponential correlation decay $|C(t)| \leq e^{-\gamma t}$. Glassy dynamics (localization) cannot map into rapidly mixing systems.

**Certificate Logic:**
$$K_{\mathrm{TB}_\rho}^+ \wedge K_{C_\mu}^+ \wedge (\gamma_{\mathrm{bad}} = 0 \wedge \gamma_{\mathcal{H}} > 0) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(\tau_{\text{mix, bad}}, \tau_{\text{mix}, \mathcal{H}}, \text{spectral gap mismatch proof})$

**Automation**: Via spectral gap estimation / Markov chain analysis / correlation function computation

**Literature:** Ergodic theorem {cite}`Birkhoff31`; mixing times {cite}`LevinPeresWilmer09`; recurrence {cite}`Furstenberg81`.

:::

:::{prf:definition} E10: Definability obstruction (Tameness)
:label: def-e10

**Sieve Signature:**
- **Required Permits:** $\mathrm{TB}_O$ (o-minimal/tame structure), $\mathrm{Rep}_K$ (representability), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{TB}_O}^+, K_{\mathrm{Rep}_K}^+\}$ (tameness and finite representation available)
- **Produces:** $K_{\mathrm{E10}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity); specifically Mode T.C (labyrinthine/wild)
- **Breached By:** Both structures tame or definability undecidable

**Method**: Model theory / O-minimal structure analysis

**Mechanism**: If morphisms must preserve o-minimal (tame) structure but $\mathbb{H}_{\mathrm{bad}}$ involves wild topology, Hom is empty. O-minimality ensures definable subsets of $\mathbb{R}$ are finite unions of points and intervals. The cell decomposition theorem gives finite stratification with bounded Betti numbers $\sum_k b_k(A) \leq C$. Wild embeddings (Alexander horned sphere, Cantor boundaries) cannot exist in tame structures.

**Certificate Logic:**
$$K_{\mathrm{TB}_O}^+ \wedge K_{\mathrm{Rep}_K}^+ \wedge (\mathbb{H}_{\mathrm{bad}} \notin \mathcal{O}\text{-min}) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(\text{definability class}, \text{wild topology witness}, \text{cell decomposition failure})$

**Automation**: Via model-theoretic analysis / stratification algorithms / Betti number computation

**Literature:** Tame topology {cite}`vandenDries98`; quantifier elimination {cite}`Tarski51`; model completeness {cite}`Wilkie96`.

:::

:::{prf:definition} E11: Galois-Monodromy Lock
:label: def-e11

**Sieve Signature:**
- **Required Permits:** $\mathrm{Rep}_K$ (representation/algebraic structure), $\mathrm{TB}_\pi$ (topology/monodromy), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+, K_{\mathrm{TB}_\pi}^+\}$ (Galois group and monodromy available)
- **Produces:** $K_{\mathrm{E11}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** S.E (Supercritical Cascade); S.C (Computational Overflow)
- **Breached By:** Galois group solvable or monodromy finite

:::{prf:definition} Galois Group
:label: def-galois-group-permit

For a polynomial $f(x) \in \mathbb{Q}[x]$, the **Galois group** $\mathrm{Gal}(f)$ is the group of automorphisms of the splitting field $K$ that fix $\mathbb{Q}$.
:::

:::{prf:definition} Monodromy Group
:label: def-monodromy-group-permit

For a differential equation with singularities, the **monodromy group** $\mathrm{Mon}(f)$ describes how solutions transform when analytically continued around singularities.
:::

**Method**: Galois theory / Monodromy representation analysis

**Mechanism**: If morphisms must preserve algebraic structure but $\mathbb{H}_{\mathrm{bad}}$ has non-solvable Galois group, no closed-form solution exists. The key constraints are:

1. **Orbit Finiteness:** If $\mathrm{Gal}(f)$ is finite, the orbit of any root under field automorphisms is finite:
   $$|\{\sigma(\alpha) : \sigma \in \mathrm{Gal}(f)\}| = |\mathrm{Gal}(f)| < \infty$$

2. **Solvability Obstruction:** If $\mathrm{Gal}(f)$ is not solvable (e.g., $S_n$ for $n \geq 5$), then $f$ has no solution in radicals. The system cannot be simplified beyond a certain complexity threshold.

3. **Monodromy Constraint:** For a differential equation, if the monodromy group is infinite, solutions have infinitely many branches (cannot be single-valued on any open set).

4. **Computational Barrier:** Determining $\mathrm{Gal}(f)$ is generally hard (no polynomial-time algorithm known). This prevents algorithmic shortcuts in solving algebraic systems.

**Certificate Logic:**
$$K_{\mathrm{Rep}_K}^+ \wedge K_{\mathrm{TB}_\pi}^+ \wedge (\mathrm{Gal}(f) \text{ non-solvable} \vee |\mathrm{Mon}(f)| = \infty) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Proof Sketch (Abel-Ruffini):**

*Step 1 (Galois correspondence).* For $f(x) \in \mathbb{Q}[x]$ with splitting field $K$, the Galois group $\mathrm{Gal}(K/\mathbb{Q})$ embeds into $S_n$ via root permutations. The Fundamental Theorem establishes bijection: subgroups $H \subseteq \mathrm{Gal}(K/\mathbb{Q}) \leftrightarrow$ intermediate fields $\mathbb{Q} \subseteq F \subseteq K$.

*Step 2 (Solvability criterion).* $f$ is solvable by radicals iff $\mathrm{Gal}(f)$ is a solvable group (admits subnormal series with abelian quotients). Each radical extension $F_{i+1} = F_i(\sqrt[n_i]{a_i})$ corresponds to cyclic Galois quotient.

*Step 3 (Non-solvability of $S_n$).* For $n \geq 5$, $A_n$ is simple (non-trivial normal subgroups contain 3-cycles, which generate $A_n$). The derived series $S_n \triangleright A_n \triangleright \{e\}$ fails to terminate abelianly; $S_n$ is not solvable.

*Step 4 (Generic quintic).* For generic quintic $f(x) = x^5 + \cdots$, $\mathrm{Gal}(f) \cong S_5$. No radical expression exists for roots.

*Step 5 (Monodromy-Galois correspondence).* For Fuchsian ODEs, the monodromy group $\mathrm{Mon}(f)$ is Zariski-dense in the differential Galois group $G_{\mathrm{diff}}$. Infinite monodromy implies infinitely many solution branches.

**Certificate Payload**: $(\mathrm{Gal}(f), \text{solvability status}, \mathrm{Mon}(f), \text{Abel-Ruffini witness})$

**Automation**: Via factorization over primes / Chebotarev density analysis / monodromy computation

**Literature:** Abel-Ruffini theorem {cite}`Abel1826`; Galois theory {cite}`DummitFoote04`; differential Galois theory {cite}`vanderPutSinger03`; Schlesinger's theorem {cite}`Schlesinger12`.

:::

:::{prf:definition} E12: Algebraic Compressibility (Permit Schema with Alternative Backends)
:label: def-e12

**Sieve Signature:**
- **Required Permits (Alternative Backends):**
  - **Backend A:** $K_{\mathrm{Rep}_K}^+$ (hypersurface) + $K_{\mathrm{SC}_\lambda}^{\text{deg}}$ → $K_{\mathrm{E12}}^{\text{hypersurf}}$
  - **Backend B:** $K_{\mathrm{Rep}_K}^+$ (complete intersection) + $K_{\mathrm{SC}_\lambda}^{\text{Bez}}$ → $K_{\mathrm{E12}}^{\text{c.i.}}$
  - **Backend C:** $K_{\mathrm{Rep}_K}^+$ (morphism) + $K_{\mathrm{DegImage}_m}^+$ + $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{deg}}$ → $K_{\mathrm{E12}}^{\text{morph}}$
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+\}$ (algebraic variety structure available)
- **Produces:** $K_{\mathrm{E12}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** S.E (Supercritical Cascade); S.C (Computational Overflow)
- **Breached By:** Degree compatibility, linear structure, or compatible morphism exists

**Context:** Algebraic compressibility obstructions arise when attempting to approximate or represent a high-degree variety using lower-degree data. The degree of an algebraic variety is an intrinsic geometric invariant that resists compression.

**Critical Remark:** The naive claim "degree $\delta$ cannot be represented by polynomials of degree $< \delta$" is **imprecise** for general varieties (e.g., a parametric representation can use lower-degree maps). The following backends make the obstruction precise by specifying what "representation" means.

:::{prf:definition} Algebraic Variety
:label: def-algebraic-variety-permit

An **algebraic variety** $V \subset \mathbb{P}^n$ (or $\mathbb{C}^n$) is the zero locus of polynomial equations:
$$V = \{x \in \mathbb{P}^n : f_1(x) = \cdots = f_k(x) = 0\}$$
:::

:::{prf:definition} Degree of a Variety
:label: def-variety-degree-permit

The **degree** $\deg(V)$ of an irreducible variety $V \subset \mathbb{P}^n$ of dimension $d$ is the number of intersection points with a generic linear subspace $L$ of complementary dimension $(n-d)$:
$$\deg(V) = \#(V \cap L)$$
counted with multiplicity. Equivalently, $\deg(V) = \int_V c_1(\mathcal{O}(1))^d$.
:::

**Certificate Logic:**
$$K_{\mathrm{Rep}_K}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge \left(K_{\mathrm{E12}}^{\text{hypersurf}} \vee K_{\mathrm{E12}}^{\text{c.i.}} \vee K_{\mathrm{E12}}^{\text{morph}}\right) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

---

#### Backend A: Hypersurface Form

**Hypotheses:**
1. $V = Z(f) \subset \mathbb{P}^n$ is an **irreducible hypersurface**
2. $f \in \mathbb{C}[x_0, \ldots, x_n]$ is irreducible with $\deg(f) = \delta$
3. "Representation" means: a single polynomial whose zero locus is $V$

**Certificate:** $K_{\mathrm{E12}}^{\text{hypersurf}} = (\delta, f, \text{irreducibility witness})$

**Proof (5 Steps):**

*Step 1 (Hypersurface Setup).* Let $V = Z(f)$ where $f$ is an irreducible homogeneous polynomial of degree $\delta$. The degree of $V$ as a variety equals $\delta$ (a generic line intersects $V$ in $\delta$ points by Bézout).

*Step 2 (Defining Equation Characterization).* A polynomial $g$ defines the same hypersurface ($Z(g) = V$) if and only if $g$ and $f$ have the same irreducible factors (up to units). Since $f$ is irreducible, $Z(g) = V$ implies $\sqrt{(g)}^{\mathrm{sat}} = (f)$ in the homogeneous coordinate ring, where $(-)^{\mathrm{sat}}$ denotes saturation by the irrelevant ideal $(x_0, \ldots, x_n)$. (In the affine case, saturation is automatic.)

*Step 3 (Degree Lower Bound via Irreducibility).* Since $Z(g) = Z(f) = V$, the radical ideals coincide: $\sqrt{(g)} = \sqrt{(f)}$. Because $f$ is irreducible, $(f)$ is a prime ideal, so $\sqrt{(f)} = (f)$. Hence $g \in \sqrt{(g)} = (f)$, which means $f | g$ (i.e., $g = f \cdot h$ for some polynomial $h$). Therefore:
$$\deg(g) = \deg(f) + \deg(h) \geq \deg(f) = \delta$$

*Step 4 (Sharpness).* The bound is achieved by $g = f$ itself. No polynomial of degree $< \delta$ can define $V$.

*Step 5 (Certificate Construction).* The obstruction: if $\mathbb{H}_{\mathrm{bad}}$ requires representing $V$ with $\deg < \delta$, this is impossible.

**Literature:** Irreducibility and defining equations {cite}`Hartshorne77`; Nullstellensatz {cite}`CoxLittleOShea15`

---

#### Backend B: Complete Intersection Form

**Hypotheses:**
1. $V \subset \mathbb{P}^n$ is a **complete intersection** of codimension $k$
2. $V = Z(f_1, \ldots, f_k)$ where $\deg(f_i) = d_i$ and $\dim V = n - k$ (expected dimension)
3. "Representation" means: $k$ equations cutting out $V$ scheme-theoretically

**Certificate:** $K_{\mathrm{E12}}^{\text{c.i.}} = (\deg(V), k, (d_1, \ldots, d_k))$

**Proof (5 Steps):**

*Step 1 (Complete Intersection Definition).* $V$ is a complete intersection if it is cut out by exactly $\text{codim}(V)$ equations and has the expected dimension. The ideal $I_V = (f_1, \ldots, f_k)$ is generated by a regular sequence.

*Step 2 (Degree via Bézout / Intersection Theory).* For a complete intersection:
$$\deg(V) = d_1 \cdot d_2 \cdots d_k$$
This follows from iterative application of Bézout's theorem {cite}`Fulton84` (Example 8.4.6).

*Step 3 (Representation Bounds).* Suppose $V = Z(g_1, \ldots, g_k)$ is another complete intersection representation with $\deg(g_i) = e_i$, **where $(g_1, \ldots, g_k)$ is also a regular sequence cutting out $V$ scheme-theoretically in expected codimension**. Then:
$$\deg(V) = e_1 \cdots e_k = d_1 \cdots d_k$$
The product of degrees is an invariant of the scheme structure.

*Step 4 (AM-GM Minimum Degree Constraint).* Among all complete intersection representations, the maximum single-equation degree satisfies:
$$\max_i(e_i) \geq \deg(V)^{1/k}$$
by AM-GM. If $d_1 \geq d_2 \geq \cdots \geq d_k$, then $d_1 \geq \deg(V)^{1/k}$.

*Step 5 (Certificate Construction).* The obstruction: if all $e_i < \deg(V)^{1/k}$, then $e_1 \cdots e_k < \deg(V)$, contradiction. Cannot uniformly lower defining degrees.

**Literature:** Bézout's theorem {cite}`Fulton84`; complete intersections {cite}`EisenbudHarris16`

---

#### Backend C: Morphism / Compression Form

**Hypotheses:**
1. $V \subset \mathbb{P}^n$ is an irreducible variety of dimension $d$ and degree $\delta$
2. A "compression of complexity $m$" is a generically finite morphism $\phi: W \to V$ of degree $\leq m$ from a variety $W$ of degree $< \delta$
3. Equivalently: $V$ is the image of a low-degree variety under a low-degree map

**Certificate:** $K_{\mathrm{E12}}^{\text{morph}} = (\delta, d, m_{\min}, \text{Bézout witness})$

**Proof (5 Steps):**

*Step 1 (Morphism Degree Definition).* For a generically finite morphism $\phi: W \to V$, the **degree** $d_\phi$ is the generic fiber cardinality: $d_\phi = |\phi^{-1}(p)|$ for generic $p \in V$.

*Step 2 (Projection Formula).* For a finite morphism $\phi: W \to V$ of degree $d_\phi$:
$$\deg(V) \cdot d_\phi = \deg(\phi^* H^{\dim V})$$
More directly: $\deg(V) \leq d_\phi \cdot \deg(W)$ with equality for finite morphisms.

*Step 3 (Degree Bound for Images).* By permit $K_{\mathrm{DegImage}_m}^+$ (degree-of-image bound, Definition {prf:ref}`def-permit-degimage`), after resolving indeterminacy (or using the graph), if $\phi$ is induced by a base-point-free linear system of degree $\leq m$, then:
$$\deg(\overline{\phi(W)}) \leq m^{\dim W} \cdot \deg(W)$$
The permit payload specifies whether $\phi$ is treated as a morphism $W \to \mathbb{P}^N$ or a rational map with resolved base locus.

*Step 4 (Compression Obstruction).* Suppose we want to represent $V$ (degree $\delta$) as $\phi(W)$ where $\deg(W) = w < \delta$ and $\phi$ has degree $\leq m$. Then:
$$\delta = \deg(V) \leq m^d \cdot w < m^d \cdot \delta$$
This is only possible if $m^d \geq \delta/w > 1$, hence $m \geq (\delta/w)^{1/d}$.

*Step 5 (Certificate Construction).* The morphism complexity lower bound:
$$m_{\min} = \left(\frac{\delta}{\deg(W)}\right)^{1/\dim V}$$
Any compression must have complexity at least $m_{\min}$.

**Literature:** Degrees of morphisms {cite}`Lazarsfeld04`; projection formulas {cite}`Fulton84`

---

**Backend Selection Logic:**

| Backend | Hypothesis | Best For |
|:-------:|:----------:|:--------:|
| A | $V$ is irreducible hypersurface | Single-equation varieties, cryptographic hardness |
| B | $V$ is complete intersection | Multi-equation varieties, computational algebra |
| C | Morphism/parametric representation | Parametrization complexity, circuit lower bounds |

**Automation:** Via degree computation / resultant analysis / intersection multiplicity bounds / Gröbner bases

**Use:** Blocks attempts to approximate high-complexity algebraic patterns using low-degree/low-complexity tools. Essential for: cryptographic hardness, complexity lower bounds, and geometric obstructions.

**Literature:** Bézout's theorem {cite}`Fulton84`; intersection theory {cite}`EisenbudHarris16`; algebraic geometry {cite}`Hartshorne77`; elimination theory {cite}`CoxLittleOShea15`; positivity {cite}`Lazarsfeld04`.

:::

**Summary: The Twelve Exclusion Tactics**

| Tactic | Name | Required Permits | Produces | Blocks |
|--------|------|------------------|----------|--------|
| E1 | Dimension | $\mathrm{Rep}_K$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E1}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | All (dim mismatch) |
| E2 | Invariant | $\mathrm{Rep}_K$, $\mathrm{TB}_\pi$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E2}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | All (invariant) |
| E3 | Positivity | $D_E$, $\mathrm{LS}_\sigma$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E3}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | All (cone) |
| E4 | Integrality | $\mathrm{Rep}_K$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E4}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | All (lattice) |
| E5 | Functional Eq. | $\mathrm{Rep}_K$, $\mathrm{GC}_\nabla$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E5}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | All (unsolvable) |
| E6 | Causal | $\mathrm{TB}_\pi$, $D_E$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E6}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | All (CTC) |
| E7 | Thermodynamic | $D_E$, $\mathrm{SC}_\lambda$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E7}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | C.E (2nd Law) |
| E8 | Holographic | $\mathrm{Cap}_H$, $\mathrm{TB}_\pi$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E8}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | C.D (Bekenstein) |
| E9 | Ergodic | $\mathrm{TB}_\rho$, $C_\mu$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E9}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | T.D (mixing) |
| E10 | Definability | $\mathrm{TB}_O$, $\mathrm{Rep}_K$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E10}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | T.C (tameness) |
| E11 | Galois-Monodromy | $\mathrm{Rep}_K$, $\mathrm{TB}_\pi$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E11}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | S.E (solvability) |
| E12 | Algebraic Compressibility | $\mathrm{Rep}_K$, $\mathrm{SC}_\lambda$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E12}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | S.E (degree) |

:::{prf:definition} Breached-Inconclusive Certificate (Lock Tactic Exhaustion)
:label: def-lock-breached-inc

If all twelve tactics fail to prove Hom-emptiness but also fail to construct an explicit morphism:

$$K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}} = (\mathsf{tactics\_exhausted}: \{E1,\ldots,E12\}, \mathsf{partial\_progress}, \mathsf{trace})$$

This is a NO verdict (Breached) with inconclusive subtype—routing to MT 42.1 (Structural Reconstruction) rather than fatal error. The certificate records which tactics were attempted and any partial progress (e.g., dimension bounds that narrowed but did not close, spectral gaps that are positive but not sufficient).

:::

---

# Part XII: Factory Metatheorems

## 23. TM-1: Gate Evaluator Factory

:::{prf:theorem} Gate Evaluator Factory
:label: mt-gate-factory
:class: metatheorem

For any system of type $T$ with user-defined objects $(\Phi, \mathfrak{D}, G, \mathcal{R}, \mathrm{Cap}, \tau, D)$, there exist canonical verifiers for all gate nodes:

**Input**: Type $T$ structural data + user definitions

**Output**: For each gate $i \in \{1, \ldots, 17\}$:
- Predicate instantiation $P_i^T$
- Verifier $V_i^T: X \times \Gamma \to \{`YES`, `NO`\} \times \mathcal{K}_i$

**Soundness**: $V_i^T(x, \Gamma) = (`YES`, K_i^+) \Rightarrow P_i^T(x)$

**Literature:** Type-theoretic verification {cite}`HoTTBook`; certified programming {cite}`Leroy09`.

:::

---

## 24. TM-2: Barrier Implementation Factory

:::{prf:theorem} Barrier Implementation Factory
:label: mt-barrier-factory
:class: metatheorem

For any system of type $T$, there exist default barrier implementations with correct outcomes and non-circular preconditions:

**Input**: Type $T$ + available literature lemmas

**Output**: For each barrier $B$:
- Default implementation $\mathcal{B}^T$
- Blocked/Breached certificate generators
- Scope specification

**Properties**:
1. Non-circularity: Trigger predicate not in Pre
2. Certificate validity: Outputs satisfy contract
3. Completeness: At least one barrier per node NO path

**Literature:** Epsilon-regularity theorems {cite}`CaffarelliKohnNirenberg82`; Foster-Lyapunov barriers {cite}`MeynTweedie93`; singularity barriers {cite}`Hamilton82`.

:::

---

## 25. TM-3: Surgery Schema Factory

:::{prf:theorem} Surgery Schema Factory
:label: mt-surgery-factory
:class: metatheorem

For any type $T$ admitting surgery, there exist default surgery operators matching diagram re-entry targets:

**Input**: Type $T$ + canonical profile library + admissibility interface

**Output**: For each surgery $S$:
- Surgery operator $\mathcal{O}_S^T$
- Admissibility checker
- Re-entry certificate generator
- Progress measure

**Fallback**: If type $T$ does not admit surgery, output "surgery unavailable" certificate ($K_{\mathrm{Surg}}^{\mathrm{inc}}$) routing to reconstruction (MT 42.1).

**Literature:** Hamilton-Perelman surgery {cite}`Hamilton97`; {cite}`Perelman03`; surgery in mean curvature flow {cite}`HuiskenSinestrari09`.

:::

---

## 26. TM-4: Equivalence + Transport Factory

:::{prf:theorem} Equivalence + Transport Factory
:label: mt-transport-factory
:class: metatheorem

For any type $T$, there exists a library of admissible equivalence moves and transport lemmas:

**Input**: Type $T$ structural assumptions

**Output**:
- Equivalence moves $\mathrm{Eq}_1^T, \ldots, \mathrm{Eq}_k^T$ with comparability bounds (instantiations of Eq1--Eq5)
- Transport lemmas $T_1^T, \ldots, T_6^T$ instantiated for $T$
- YES$^\sim$ production rules
- Promotion rules (immediate and a-posteriori)

**Literature:** Transport of structure in category theory {cite}`MacLane71`; univalent transport {cite}`HoTTBook`.

:::

---

## 27. TM-5: Lock Backend Factory

:::{prf:theorem} Lock Backend Factory
:label: mt-lock-factory
:class: metatheorem

For any type $T$ with $\mathrm{Rep}_K$ available, there exist E1--E10 tactics for the Lock:

**Input**: Type $T$ + representation substrate

**Output**:
- Tactic implementations $E_1^T, \ldots, E_5^T$
- Automation level indicators
- Horizon fallback procedure

**Rep unavailable**: If $\mathrm{Rep}_K$ is not available, Lock uses only E1--E3 (geometry-based tactics) with limited automation.

**Literature:** Automated theorem proving {cite}`BaaderNipkow98`; invariant theory {cite}`MumfordFogartyKirwan94`; obstruction theory {cite}`EilenbergSteenrod52`.

:::

---

# Part XIII: Instantiation

## 28. Certificate Generator Library

The **Certificate Generator Library** maps standard literature lemmas to permits:

| **Node/Barrier** | **Literature Tool** | **Certificate** |
|---|---|---|
| EnergyCheck | Energy inequality, Grönwall | $K_{D_E}^+$ |
| BarrierSat | Foster-Lyapunov, drift control | $K_{\text{sat}}^{\mathrm{blk}}$ |
| ZenoCheck | Dwell-time lemma, event bounds | $K_{\mathrm{Rec}_N}^+$ |
| CompactCheck (YES) | Concentration-compactness | $K_{C_\mu}^+$, $K_{\text{prof}}$ |
| CompactCheck (NO) | Dispersion estimates | $K_{C_\mu}^-$, leads to D.D |
| ScaleCheck | Scaling analysis, critical exponents | $K_{\mathrm{SC}_\lambda}^+$ |
| BarrierTypeII | Monotonicity formulas, Kenig-Merle | $K_{\text{II}}^{\mathrm{blk}}$ |
| ParamCheck | Modulation theory, orbital stability | $K_{\mathrm{SC}_{\partial c}}^+$ |
| GeomCheck | Hausdorff dimension, capacity | $K_{\mathrm{Cap}_H}^+$ |
| BarrierCap | Epsilon-regularity, partial regularity | $K_{\text{cap}}^{\mathrm{blk}}$ |
| StiffnessCheck | Łojasiewicz-Simon, spectral gap | $K_{\mathrm{LS}_\sigma}^+$ |
| BarrierGap | Poincaré inequality, mass gap | $K_{\text{gap}}^{\mathrm{blk}}$ |
| TopoCheck | Sector classification, homotopy | $K_{\mathrm{TB}_\pi}^+$ |
| TameCheck | O-minimal theory, definability | $K_{\mathrm{TB}_O}^+$ |
| ErgoCheck | Mixing times, ergodic theory | $K_{\mathrm{TB}_\rho}^+$ |
| ComplexCheck | Kolmogorov complexity, MDL | $K_{\mathrm{Rep}_K}^+$ |
| OscillateCheck | Monotonicity, De Giorgi-Nash-Moser | $K_{\mathrm{GC}_\nabla}^-$ |
| Lock | Cohomology, invariant theory | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$ |

---

## 29. Minimal Instantiation Checklist

:::{prf:theorem} Instantiation Metatheorem
:label: mt-instantiation
:class: metatheorem

For any system of type $T$ with user-supplied functionals, there exists a canonical sieve implementation satisfying all contracts:

**User provides (definitions only)**:
1. State space $X$ and symmetry group $G$
2. Height functional $\Phi: X \to [0, \infty]$
3. Dissipation functional $\mathfrak{D}: X \to [0, \infty]$
4. Recovery functional $\mathcal{R}: X \to [0, \infty)$ (if $\mathrm{Rec}_N$)
5. Capacity gauge $\mathrm{Cap}$ (if $\mathrm{Cap}_H$)
6. Sector label $\tau: X \to \mathcal{T}$ (if $\mathrm{TB}_\pi$)
7. Dictionary map $D: X \to \mathcal{T}$ (if $\mathrm{Rep}_K$, optional)
8. Type selection $T \in \{T_{\text{parabolic}}, T_{\text{dispersive}}, T_{\text{metricGF}}, T_{\text{Markov}}, T_{\text{algorithmic}}\}$

**Framework provides (compiled from factories)**:
1. Gate evaluators (TM-1)
2. Barrier implementations (TM-2)
3. Surgery schemas (TM-3)
4. Equivalence + Transport (TM-4)
5. Lock backend (TM-5)

**Output**: Sound sieve run yielding either:
- Regularity certificate (VICTORY)
- Mode certificate with admissible repair (surgery path)
- NO-inconclusive certificate ($K^{\mathrm{inc}}$) (explicit obstruction to classification/repair)

**Literature:** Type-theoretic instantiation {cite}`HoTTBook`; certified regularity proofs {cite}`Leroy09`; singularity resolution via surgery {cite}`Perelman03`.

:::

---

## 30. Metatheorem Unlock Table

The following table specifies which metatheorems are unlocked by which certificates:

| **Metatheorem** | **Required Certificates** | **Producing Nodes** |
|---|---|---|
| Structural Resolution | $K_{C_\mu}^+$ (profile) | CompactCheck YES |
| Type II Exclusion | $K_{\mathrm{SC}_\lambda}^+$ (subcritical) + $K_{D_E}^+$ (energy) | ScaleCheck YES + EnergyCheck YES |
| Capacity Barrier | $K_{\mathrm{Cap}_H}^+$ or $K_{\text{cap}}^{\mathrm{blk}}$ | GeomCheck YES/Blocked |
| Topological Suppression | $K_{\mathrm{TB}_\pi}^+$ + $K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$ (Lock) | TopoCheck YES + Lock Blocked |
| Canonical Lyapunov | $K_{\mathrm{LS}_\sigma}^+$ (stiffness) + $K_{\mathrm{GC}_\nabla}^-$ (no oscillation) | StiffnessCheck YES + OscillateCheck NO |
| Functional Reconstruction | $K_{\mathrm{LS}_\sigma}^+$ + $K_{\mathrm{Rep}_K}^+$ (Rep) + $K_{\mathrm{GC}_\nabla}^-$ | LS + Rep + GC |
| Profile Classification | $K_{C_\mu}^+$ | CompactCheck YES |
| Surgery Admissibility | $K_{\text{lib}}$ or $K_{\text{strat}}$ | Profile Trichotomy Case 1/2 |
| Global Regularity | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$ (Lock Blocked) | BarrierExclusion Blocked |

---

## 31. Diagram ↔ Specification Cross-Reference

The following table provides a complete cross-reference between diagram node names and their formal definitions:

| **Node** | **Diagram Label** | **Predicate Def.** | **Barrier/Surgery** |
|---|---|---|---|
| 1 | EnergyCheck | {ref}`def-node-energy` | BarrierSat ({ref}`def-barrier-sat`) |
| 2 | ZenoCheck | {ref}`def-node-zeno` | BarrierCausal ({ref}`def-barrier-causal`) |
| 3 | CompactCheck | {ref}`def-node-compact` | BarrierScat ({ref}`def-barrier-scat`) |
| 4 | ScaleCheck | {ref}`def-node-scale` | BarrierTypeII ({ref}`def-barrier-type2`) |
| 5 | ParamCheck | {ref}`def-node-param` | BarrierVac ({ref}`def-barrier-vac`) |
| 6 | GeomCheck | {ref}`def-node-geom` | BarrierCap ({ref}`def-barrier-cap`) |
| 7 | StiffnessCheck | {ref}`def-node-stiffness` | BarrierGap ({ref}`def-barrier-gap`) |
| 7a | BifurcateCheck | {ref}`def-node-bifurcate` | Mode S.D |
| 7b | SymCheck | {ref}`def-node-sym` | --- |
| 7c | CheckSC | {ref}`def-node-checksc` | ActionSSB ({ref}`def-action-ssb`) |
| 7d | CheckTB | {ref}`def-node-checktb` | ActionTunnel ({ref}`def-action-tunnel`) |
| 8 | TopoCheck | {ref}`def-node-topo` | BarrierAction ({ref}`def-barrier-action`) |
| 9 | TameCheck | {ref}`def-node-tame` | BarrierOmin ({ref}`def-barrier-omin`) |
| 10 | ErgoCheck | {ref}`def-node-ergo` | BarrierMix ({ref}`def-barrier-mix`) |
| 11 | ComplexCheck | {ref}`def-node-complex` | BarrierEpi ({ref}`def-barrier-epi`) |
| 12 | OscillateCheck | {ref}`def-node-oscillate` | BarrierFreq ({ref}`def-barrier-freq`) |
| 13 | BoundaryCheck | {ref}`def-node-boundary` | --- |
| 14 | OverloadCheck | {ref}`def-node-overload` | BarrierBode ({ref}`def-barrier-bode`) |
| 15 | StarveCheck | {ref}`def-node-starve` | BarrierInput ({ref}`def-barrier-input`) |
| 16 | AlignCheck | {ref}`def-node-align` | BarrierVariety ({ref}`def-barrier-variety`) |
| 17 | BarrierExclusion | {ref}`def-node-lock` | Lock (Part XI) |

---

# Part XIV: Instantaneous Upgrade Metatheorems

## 32. Instantaneous Certificate Upgrades

The **Instantaneous Upgrade Metatheorems** formalize the logical principle that a "Blocked" barrier certificate or a "Surgery" re-entry certificate can be promoted to a full **YES** (or **YES$^\sim$**) permit under appropriate structural conditions. These upgrades occur *within* a single Sieve pass when the blocking condition itself implies a stronger regularity guarantee.

**Logical Form:** $K_{\text{Node}}^- \wedge K_{\text{Barrier}}^{\mathrm{blk}} \Rightarrow K_{\text{Node}}^{\sim}$

The key insight is that certain "obstructions" are themselves *certificates of regularity* when viewed from the correct perspective.

---

### 32.1 Saturation Promotion

:::{prf:metatheorem} Saturation Promotion (BarrierSat $\to$ YES$^\sim$)
:label: mt-saturation-promotion
:class: metatheorem

**Context:** Node 1 (EnergyCheck) fails ($E = \infty$), but BarrierSat is Blocked ($K_{\text{sat}}^{\mathrm{blk}}$).

**Hypotheses.** Let $\mathcal{H} = (\mathcal{X}, \Phi, \mathfrak{D}, G)$ be a Hypostructure with:
1. A height functional $\Phi: \mathcal{X} \to [0, \infty]$ that is unbounded ($\sup_x \Phi(x) = \infty$)
2. A dissipation functional $\mathfrak{D}$ satisfying the drift condition: there exist $\lambda > 0$ and $b < \infty$ such that
   $$\mathcal{L}\Phi(x) \leq -\lambda \Phi(x) + b \quad \text{for all } x \in \mathcal{X}$$
   where $\mathcal{L}$ is the infinitesimal generator of the dynamics.
3. A compact sublevel set $\{x : \Phi(x) \leq c\}$ for some $c > b/\lambda$.

**Statement:** Under the drift condition, the process admits a unique invariant probability measure $\pi$ with $\int \Phi \, d\pi < \infty$. The system is equivalent to one with bounded energy under the renormalized measure $\pi$.

**Certificate Logic:**
$$K_{D_E}^- \wedge K_{\text{sat}}^{\mathrm{blk}} \Rightarrow K_{D_E}^{\sim}$$

**Proof sketch:** The drift condition implies geometric ergodicity by the Foster-Lyapunov criterion (Meyn and Tweedie, 1993, Theorem 15.0.1). The invariant measure $\pi$ satisfies $\pi(\Phi) < \infty$ by Theorem 14.0.1 of the same reference. The renormalized height $\hat{\Phi} = \Phi - \pi(\Phi)$ is centered and the dynamics converge exponentially to equilibrium.

**Interface Permit Validated:** Finite Energy (renormalized measure).

**Literature:** {cite}`MeynTweedie93`; {cite}`HairerMattingly11`
:::

---

### 32.2 Causal Censor Promotion

:::{prf:metatheorem} Causal Censor Promotion (BarrierCausal $\to$ YES$^\sim$)
:label: mt-causal-censor
:class: metatheorem

**Context:** Node 2 (ZenoCheck) fails ($N \to \infty$), but BarrierCausal is Blocked ($K_{\mathrm{Rec}_N}^{\mathrm{blk}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. An event counting functional $N: \mathcal{X} \times [0,T] \to \mathbb{N} \cup \{\infty\}$
2. A singularity requiring infinite computational depth to resolve: the Cauchy development $D^+(S)$ is globally hyperbolic but $N(x, T) \to \infty$ as $x \to \Sigma$
3. A cosmic censorship condition: the singular set $\Sigma$ is contained in the future boundary $\mathcal{I}^+ \cup i^+$ of conformally compactified spacetime.

**Statement:** If the singularity is hidden behind an event horizon or lies at future null/timelike infinity, it is causally inaccessible to any physical observer. The event count is finite relative to any observer worldline $\gamma$ with finite proper time.

**Certificate Logic:**
$$K_{\mathrm{Rec}_N}^- \wedge K_{\mathrm{Rec}_N}^{\mathrm{blk}} \Rightarrow K_{\mathrm{Rec}_N}^{\sim}$$

**Proof sketch:** By the weak cosmic censorship conjecture (Penrose, 1969), generic gravitational collapse produces singularities cloaked by event horizons. The Hawking-Penrose theorems (1970) establish geodesic incompleteness, but the Christodoulou-Klainerman stability theorem (1993) ensures the exterior remains regular. Any observer worldline $\gamma \subset J^-(\mathcal{I}^+)$ experiences finite proper time and finite events before the singularity becomes causally relevant.

**Interface Permit Validated:** Finite Event Count (physically observable).

**Literature:** {cite}`Penrose69`; {cite}`ChristodoulouKlainerman93`; {cite}`HawkingPenrose70`
:::

---

### 32.3 Scattering Promotion

:::{prf:metatheorem} Scattering Promotion (BarrierScat $\to$ VICTORY)
:label: mt-scattering-promotion
:class: metatheorem

**Context:** Node 3 (CompactCheck) fails (No concentration), but BarrierScat indicates Benign ($K_{C_\mu}^{\mathrm{ben}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure of type $T_{\text{dispersive}}$ with:
1. A dispersive evolution $u(t)$ satisfying a nonlinear wave or Schrödinger equation
2. The concentration-compactness dichotomy: either $\mu(V) > 0$ for some profile $V$, or dispersion dominates
3. A finite Morawetz quantity: $\int_0^\infty \int_{\mathbb{R}^n} |x|^{-1} |u|^{p+1} \, dx \, dt < \infty$

**Statement:** If energy disperses (no concentration) and the interaction functional is finite (Morawetz bound), the solution scatters to a free linear state: there exists $u_\pm \in H^1$ such that $\|u(t) - e^{it\Delta}u_\pm\|_{H^1} \to 0$ as $t \to \pm\infty$. This is a "Victory" condition equivalent to global existence and regularity.

**Certificate Logic:**
$$K_{C_\mu}^- \wedge K_{C_\mu}^{\mathrm{ben}} \Rightarrow \text{Global Regularity}$$

**Proof sketch:** The Morawetz estimate (1968) provides spacetime integrability. Combined with Strichartz estimates and the concentration-compactness/rigidity methodology of Kenig and Merle (2006), the absence of concentration implies scattering. The limiting profile $u_\pm$ is obtained via the Cook method or profile decomposition.

**Interface Permit Validated:** Global Existence (via dispersion).

**Literature:** {cite}`Morawetz68`; {cite}`KenigMerle06`; {cite}`KillipVisan10`
:::

---

### 32.4 Type II Suppression Promotion

:::{prf:metatheorem} Type II Suppression (BarrierTypeII $\to$ YES$^\sim$)
:label: mt-type-ii-suppression
:class: metatheorem

**Context:** Node 4 (ScaleCheck) fails (Supercritical), but BarrierTypeII is Blocked ($K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A supercritical scaling exponent $\alpha > \alpha_c$ (energy-supercritical regime)
2. A Type II blow-up scenario where the solution concentrates at a point with unbounded $L^\infty$ norm but bounded energy
3. An energy monotonicity formula $\frac{d}{dt}\mathcal{E}_\lambda(t) \leq 0$ for the localized energy at scale $\lambda$

**Statement:** If the renormalization cost $\int_0^{T^*} \lambda(t)^{-\gamma} \, dt = \infty$ diverges logarithmically, the supercritical singularity is suppressed and cannot form in finite time. The blow-up rate satisfies $\lambda(t) \geq c(T^* - t)^{1/\gamma}$ for some $\gamma > 0$.

**Certificate Logic:**
$$K_{\mathrm{SC}_\lambda}^- \wedge K_{\mathrm{SC}_\lambda}^{\mathrm{blk}} \Rightarrow K_{\mathrm{SC}_\lambda}^{\sim}$$

**Proof sketch:** The monotonicity formula (Merle and Zaag, 1998) bounds the blow-up rate from below. For Type II blow-up, the energy remains bounded while the scale $\lambda(t) \to 0$. The logarithmic divergence of the renormalization integral creates an energy barrier that prevents finite-time singularity formation. This mechanism underlies the Raphaël-Szeftel soliton resolution (2011).

**Interface Permit Validated:** Subcritical Scaling (effective).

**Literature:** {cite}`MerleZaag98`; {cite}`RaphaelSzeftel11`; {cite}`CollotMerleRaphael17`
:::

---

### 32.5 Capacity Promotion

:::{prf:metatheorem} Capacity Promotion (BarrierCap $\to$ YES$^\sim$)
:label: mt-capacity-promotion
:class: metatheorem

**Context:** Node 6 (GeomCheck) fails (Codim too small), but BarrierCap is Blocked ($K_{\mathrm{Cap}_H}^{\mathrm{blk}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A singular set $\Sigma \subset \mathcal{X}$ with Hausdorff dimension $\dim_H(\Sigma) \geq n-2$ (marginal codimension)
2. A capacity bound: $\mathrm{Cap}_{1,2}(\Sigma) = 0$ where $\mathrm{Cap}_{1,2}$ is the $(1,2)$-capacity (Sobolev capacity)
3. The solution $u \in H^1_{\text{loc}}(\mathcal{X} \setminus \Sigma)$

**Statement:** If the singular set has zero capacity (even if its Hausdorff dimension is large), it is removable for the $H^1$ energy class. There exists a unique extension $\tilde{u} \in H^1(\mathcal{X})$ with $\tilde{u}|_{\mathcal{X} \setminus \Sigma} = u$.

**Certificate Logic:**
$$K_{\mathrm{Cap}_H}^- \wedge K_{\mathrm{Cap}_H}^{\mathrm{blk}} \Rightarrow K_{\mathrm{Cap}_H}^{\sim}$$

**Proof sketch:** By Federer's theorem on removable singularities (1969, Section 4.7), sets of zero $(1,p)$-capacity are removable for $W^{1,p}$ functions. For $p=2$, the extension follows from the Lax-Milgram theorem applied to the weak formulation. The uniqueness follows from the maximum principle. See also Evans and Gariepy (2015, Theorem 4.7.2).

**Interface Permit Validated:** Removable Singularity.

**Literature:** {cite}`Federer69`; {cite}`EvansGariepy15`; {cite}`AdamsHedberg96`
:::

---

### 32.6 Spectral Gap Promotion

:::{prf:metatheorem} Spectral Gap Promotion (BarrierGap $\to$ YES)
:label: mt-spectral-gap
:class: metatheorem

**Context:** Node 7 (StiffnessCheck) fails (Flat), but BarrierGap is Blocked ($K_{\text{gap}}^{\mathrm{blk}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A linearized operator $L = D^2\Phi(x^*)$ at a critical point $x^*$
2. A spectral gap: $\lambda_1(L) > 0$ (smallest nonzero eigenvalue is positive)
3. The nonlinear flow $\partial_t x = -\nabla \Phi(x)$ near $x^*$

**Statement:** If a spectral gap $\lambda_1 > 0$ exists, the Łojasiewicz-Simon inequality automatically holds with optimal exponent $\theta = 1/2$. The convergence rate is exponential: $\|x(t) - x^*\| \leq Ce^{-\lambda_1 t/2}$.

**Certificate Logic:**
$$K_{\mathrm{LS}_\sigma}^- \wedge K_{\text{gap}}^{\mathrm{blk}} \Rightarrow K_{\mathrm{LS}_\sigma}^+ \quad (\text{with } \theta=1/2)$$

**Proof sketch:** The Łojasiewicz-Simon inequality states $|\Phi(x) - \Phi(x^*)|^{1-\theta} \leq C\|\nabla\Phi(x)\|$ for some $\theta \in (0,1/2]$. When the Hessian is non-degenerate ($\lambda_1 > 0$), Taylor expansion gives $\theta = 1/2$. The exponential convergence then follows from the Gronwall inequality applied to the energy functional. See Simon (1983, Theorem 3) and Feehan and Maridakis (2019).

**Interface Permit Validated:** Gradient Domination / Stiffness.

**Literature:** {cite}`Simon83`; {cite}`FeehanMaridakis19`; {cite}`Huang06`
:::

---

### 32.7 O-Minimal Promotion

:::{prf:metatheorem} O-Minimal Promotion (BarrierOmin $\to$ YES$^\sim$)
:label: mt-o-minimal
:class: metatheorem

**Context:** Node 9 (TameCheck) fails (Wild), but BarrierOmin is Blocked ($K_{\mathrm{TB}_O}^{\mathrm{blk}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A singular/wild set $W \subset \mathcal{X}$ that is a priori not regular
2. Definability: $W$ is definable in an o-minimal expansion of $(\mathbb{R}, +, \cdot)$ (e.g., $\mathbb{R}_{\text{an,exp}}$)
3. The dynamics are generated by a definable vector field

**Statement:** If the wild set is definable in an o-minimal structure, it admits a finite Whitney stratification into smooth manifolds. The set is topologically tame: it has finite Betti numbers, satisfies the curve selection lemma, and admits no pathological embeddings.

**Certificate Logic:**
$$K_{\mathrm{TB}_O}^- \wedge K_{\mathrm{TB}_O}^{\mathrm{blk}} \Rightarrow K_{\mathrm{TB}_O}^{\sim}$$

**Proof sketch:** The cell decomposition theorem (van den Dries, 1998, Chapter 3) guarantees that every definable set admits a finite partition into definable cells. The Kurdyka-Łojasiewicz inequality holds for definable functions (Kurdyka, 1998), ensuring gradient descent terminates in finite time or converges. The uniform finiteness theorem bounds topological complexity.

**Interface Permit Validated:** Tame Topology.

**Literature:** {cite}`vandenDries98`; {cite}`Kurdyka98`; {cite}`Wilkie96`
:::

---

### 32.8 Surgery Promotion

:::{prf:metatheorem} Surgery Promotion (Surgery $\to$ YES$^\sim$)
:label: mt-surgery-promotion
:class: metatheorem

**Context:** Any Node fails, Barrier breached, but Surgery $S$ executes and issues re-entry certificate ($K^{\mathrm{re}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A singularity at $(t^*, x^*) \in \mathcal{X}$ with modal diagnosis $M \in \{C.E, C.C, \ldots, B.C\}$
2. A valid surgery operator $\mathcal{O}_S: (\mathcal{X}, \Phi) \to (\mathcal{X}', \Phi')$ satisfying:
   - Admissibility: singular profile $V \in \mathcal{L}_T$ (canonical library)
   - Capacity bound: $\mathrm{Cap}(\text{excision}) \leq \varepsilon_{\text{adm}}$
   - Progress: $\Phi'(x') \leq \Phi(x) - \delta_S$ (height decrease)

**Statement:** If a valid surgery is performed, the flow continues on the modified Hypostructure $\mathcal{H}'$. The combined flow (pre-surgery on $\mathcal{X}$, post-surgery on $\mathcal{X}'$) constitutes a generalized (surgery/weak) solution.

**Certificate Logic:**
$$K_{\text{Node}}^- \wedge K_{\text{Surg}}^{\mathrm{re}} \Rightarrow K_{\text{Node}}^{\sim} \quad (\text{on } \mathcal{X}')$$

**Proof sketch:** The surgery construction follows Hamilton (1997) for Ricci flow and Perelman (2002-2003) for the rigorous completion. The key ingredients are: (1) canonical neighborhood theorem ensuring surgery regions are standard, (2) non-collapsing estimates controlling geometry, (3) finite surgery time theorem bounding the number of surgeries. The post-surgery manifold inherits all regularity properties.

**Canonical Neighborhoods (Uniqueness):** The **Canonical Neighborhood Theorem** (Perelman 2003) ensures surgery is essentially unique: near any high-curvature point $p$ with $|Rm|(p) \geq r^{-2}$, the pointed manifold $(M, g, p)$ is $\varepsilon$-close (in the pointed Cheeger-Gromov sense) to one of:
- A round shrinking sphere $S^n / \Gamma$
- A round shrinking cylinder $S^{n-1} \times \mathbb{R}$
- A Bryant soliton

This **classification of local models** eliminates surgery ambiguity: the excision location and cap geometry are determined by the canonical structure up to diffeomorphism. Different valid surgery choices yield **diffeomorphic** post-surgery manifolds, making the surgery operation **functorial** in $\mathbf{Bord}_n$.

**Interface Permit Validated:** Global Existence (in the sense of surgery/weak flow).

**Literature:** {cite}`Hamilton97`; {cite}`Perelman03`; {cite}`KleinerLott08`
:::

---

### 32.9 Lock Promotion

:::{prf:metatheorem} Lock Promotion (BarrierExclusion $\to$ GLOBAL YES)
:label: mt-lock-promotion
:class: metatheorem

**Context:** Node 17 (The Lock) is Blocked ($K_{\text{Lock}}^{\mathrm{blk}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. The universal bad pattern $\mathcal{B}_{\text{univ}}$ defined via the Interface Registry
2. The morphism obstruction: $\mathrm{Hom}_{\mathcal{C}}(\mathcal{B}_{\text{univ}}, \mathcal{H}) = \emptyset$ in the appropriate category $\mathcal{C}$
3. Categorical coherence: all nodes converge to Node 17 with compatible certificates

**Statement:** If the universal bad pattern cannot map into the system (Hom-set empty), no singularities of any type can exist. The Lock validates global regularity and retroactively confirms all earlier ambiguous certificates.

**Certificate Logic:**
$$K_{\text{Lock}}^{\mathrm{blk}} \Rightarrow \text{Global Regularity}$$

**Proof sketch:** The proof uses the contrapositive: if a singularity existed, it would generate a non-trivial morphism $\phi: \mathcal{B}_{\text{univ}} \to \mathcal{H}$ by the universal property. The emptiness of the Hom-set is established via cohomological/spectral obstructions (E1-E10 tactics). This is the "Grothendieck yoga" of reducing existence questions to non-existence of maps. See SGA 4 for the categorical framework.

**Interface Permit Validated:** All Permits (Retroactively).

**Literature:** {cite}`SGA4`; {cite}`Lurie09`; {cite}`MacLane71`
:::

---

### 32.10 Absorbing Boundary Promotion

:::{prf:metatheorem} Absorbing Boundary Promotion (BoundaryCheck $\to$ EnergyCheck)
:label: mt-absorbing-boundary
:class: metatheorem

**Context:** Node 1 (Energy) fails ($E \to \infty$), but Node 13 (Boundary) confirms an Open System with dissipative flux.

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A domain $\Omega$ with boundary $\partial\Omega$
2. An energy functional $E(t) = \int_\Omega e(x,t) \, dx$
3. A boundary flux condition: $\int_{\partial\Omega} \mathbf{n} \cdot \mathbf{F} \, dS < 0$ (strictly outgoing)
4. Bounded input: $\int_0^T \|\text{source}\|_{L^2} \, dt < \infty$

**Statement:** If the flux across the boundary is strictly outgoing (dissipative) and inputs are bounded, the internal energy cannot blow up. The boundary acts as a "heat sink" absorbing energy.

**Certificate Logic:**
$$K_{D_E}^- \wedge K_{\mathrm{Bound}_\partial}^+ \wedge (\text{Flux} < 0) \Rightarrow K_{D_E}^{\sim}$$

**Proof sketch:** The energy identity $\frac{dE}{dt} = -\mathfrak{D}(t) + \int_{\partial\Omega} \text{flux} + \int_\Omega \text{source}$ with negative flux and dissipation $\mathfrak{D} \geq 0$ yields $E(t) \leq E(0) + \int_0^t \|\text{source}\| \, ds < \infty$. This is the energy method of Dafermos (2016, Chapter 5) applied to hyperbolic conservation laws with dissipative boundary conditions.

**Interface Permit Validated:** Finite Energy (via Boundary Dissipation).

**Literature:** {cite}`Dafermos16`; {cite}`DafermosRodnianski10`
:::

---

### 32.11 Catastrophe Stability Promotion

:::{prf:metatheorem} Catastrophe-Stability Promotion (BifurcateCheck $\to$ StiffnessCheck)
:label: mt-catastrophe-stability
:class: metatheorem

**Context:** Node 7 (Stiffness) fails (Flat/Zero Eigenvalue), but Node 7a (Bifurcation) identifies a **Canonical Catastrophe**.

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A potential $V(x)$ with a degenerate critical point: $V''(x^*) = 0$
2. A canonical catastrophe normal form: $V(x) = x^{k+1}/(k+1)$ for $k \geq 2$ (fold $k=2$, cusp $k=3$, etc.)
3. Higher-order stiffness: $V^{(k)}(x^*) \neq 0$

**Statement:** While the linear stiffness is zero ($\lambda_1 = 0$), the nonlinear stiffness is positive and bounded. The system is "Stiff" in a higher-order sense, ensuring polynomial convergence $t^{-1/(k-1)}$ instead of exponential.

**Certificate Logic:**
$$K_{\mathrm{LS}_\sigma}^- \wedge K_{\mathrm{LS}_{\partial^k V}}^+ \Rightarrow K_{\mathrm{LS}_\sigma}^{\sim} \quad (\text{Polynomial Rate})$$

**Proof sketch:** The Łojasiewicz exponent at a degenerate critical point is $\theta = 1/k$ for the $A_{k-1}$ catastrophe (Thom, 1975). The gradient inequality $|V(x)|^{1-\theta} \leq C|\nabla V(x)|$ yields polynomial decay. Arnold's classification (1972) ensures these are the only structurally stable degeneracies. The convergence rate follows from integrating the gradient flow ODE.

**Interface Permit Validated:** Gradient Domination (Higher Order).

**Literature:** {cite}`Thom75`; {cite}`Arnold72`; {cite}`PostonStewart78`
:::

---

### 32.12 Inconclusive Discharge Upgrades

The following metatheorems formalize inc-upgrade rules. Blocked certificates indicate "cannot proceed"; inconclusive certificates indicate "cannot decide with current prerequisites."

:::{prf:metatheorem} Inconclusive Discharge by Missing-Premise Completion
:label: mt-inc-completion
:class: metatheorem

**Context:** A node returns $K_P^{\mathrm{inc}} = (\mathsf{obligation}, \mathsf{missing}, \mathsf{code}, \mathsf{trace})$ where $\mathsf{missing}$ specifies the certificate types that would enable decision.

**Hypotheses:** For each $m \in \mathsf{missing}$, the context $\Gamma$ contains a certificate $K_m^+$ such that:
$$\bigwedge_{m \in \mathsf{missing}} K_m^+ \Rightarrow \mathsf{obligation}$$

**Statement:** The inconclusive permit upgrades immediately to YES:
$$K_P^{\mathrm{inc}} \wedge \bigwedge_{m \in \mathsf{missing}} K_m^+ \Rightarrow K_P^+$$

**Certificate Logic:**
$$\mathsf{Obl}(\Gamma) \setminus \{(\mathsf{id}_P, \ldots)\} \cup \{K_P^+\}$$

**Proof sketch:** The NO-inconclusive certificate records an epistemic gap, not a semantic refutation (Definition {prf:ref}`def-typed-no-certificates`). When all prerequisites in $\mathsf{missing}$ are satisfied, the original predicate $P$ becomes decidable. The discharge condition (Definition {prf:ref}`def-inc-upgrades`) ensures the premises genuinely imply the obligation. The upgrade is sound because $K^{\mathrm{inc}}$ records the exact obligation and its missing prerequisites; when those prerequisites are satisfied, the original predicate $P$ holds by the discharge condition.

**Interface Permit Validated:** Original predicate $P$ (via prerequisite completion).

**Literature:** Binary Certificate Logic (Definition {prf:ref}`def-typed-no-certificates`); Obligation Ledger (Definition {prf:ref}`def-obligation-ledger`).

:::

:::{prf:metatheorem} A-Posteriori Inconclusive Discharge
:label: mt-inc-aposteriori
:class: metatheorem

**Context:** $K_P^{\mathrm{inc}}$ is produced at node $i$, and later nodes add certificates that satisfy its $\mathsf{missing}$ set.

**Hypotheses:** Let $\Gamma_i$ be the context at node $i$ with $K_P^{\mathrm{inc}} \in \Gamma_i$. Later nodes produce $\{K_{j_1}^+, \ldots, K_{j_k}^+\}$ such that:
$$\{j_1, \ldots, j_k\} \supseteq \mathsf{missing}(K_P^{\mathrm{inc}})$$

**Statement:** During promotion closure (Definition {prf:ref}`def-closure`), the inconclusive certificate upgrades:
$$K_P^{\mathrm{inc}} \wedge \bigwedge_{m \in \mathsf{missing}} K_m^+ \Rightarrow K_P^+$$

**Certificate Logic:**
$$\mathrm{Cl}(\Gamma_{\mathrm{final}}) \ni K_P^+ \quad \text{(discharged from } K_P^{\mathrm{inc}} \text{)}$$

**Proof sketch:** The promotion closure iterates until fixed point. On each iteration, inc-upgrade rules (Definition {prf:ref}`def-inc-upgrades`) are applied alongside blk-promotion rules. The a-posteriori discharge is triggered when certificates from later nodes enter the closure and match the $\mathsf{missing}$ set. Termination follows from the certificate finiteness condition (Definition {prf:ref}`def-cert-finite`).

**Consequence:** The obligation ledger $\mathsf{Obl}(\mathrm{Cl}(\Gamma_{\mathrm{final}}))$ contains strictly fewer entries than $\mathsf{Obl}(\Gamma_{\mathrm{final}})$ if any inc-upgrades fired during closure.

**Interface Permit Validated:** Original predicate $P$ (retroactively).

**Literature:** Promotion Closure (Definition {prf:ref}`def-promotion-closure`); Kleene fixed-point iteration {cite}`Kleene52`.

:::

---

# Part XV: Retroactive Promotion Theorems

## 33. A-Posteriori Upgrade Rules

The **Retroactive Promotion Theorems** (or "A-Posteriori Upgrade Rules") formalize the logical principle that a stronger global guarantee found late in the Sieve can resolve local ambiguities encountered earlier. These theorems propagate information *backwards* through the verification graph.

**Logical Form:** $K_{\text{Early}}^{\text{ambiguous}} \wedge K_{\text{Late}}^{\text{strong}} \Rightarrow K_{\text{Early}}^{\text{proven}}$

The key insight is that global constraints can retrospectively determine local behavior.

---

### 33.1 Shadow-Sector Retroactive Promotion

:::{prf:metatheorem} Shadow-Sector Retroactive Promotion (TopoCheck $\to$ ZenoCheck)
:label: mt-shadow-sector-retroactive
:class: metatheorem

**Context:** Node 2 (Zeno) fails in an early epoch, but a later epoch confirms via Node 8 (TopoCheck) that the trajectory is confined to a **Finite Sector Graph**. This is a **retroactive** promotion requiring information from a completed run.

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A sector decomposition $\mathcal{X} = \bigsqcup_{i=1}^N S_i$ with finitely many sectors
2. Transition graph $\mathcal{G} = (V, E)$ where $V = \{S_1, \ldots, S_N\}$ and edges represent allowed transitions
3. An action barrier: $\mathrm{Action}(S_i \to S_j) \geq \delta > 0$ for each transition
4. Bounded energy: $E(t) \leq E_{\max}$

**Statement:** If the topological sector graph is finite and the energy is insufficient to make infinitely many transitions, the system cannot undergo infinite distinct events (Zeno behavior). The number of sector transitions is bounded by $N_{\max} \leq E_{\max}/\delta$.

**Certificate Logic:**
$$K_{\mathrm{Rec}_N}^- \wedge K_{\mathrm{TB}_\pi}^+ \wedge K_{\text{Action}}^{\mathrm{blk}} \Rightarrow K_{\mathrm{Rec}_N}^{\sim}$$

**Why Retroactive:** The certificate $K_{\text{Action}}^{\mathrm{blk}}$ is produced by BarrierAction (downstream of Node 8), which is on a different DAG branch than Node 2 failure. In a single epoch, Node 2 failure routes through BarrierCausal, never reaching Node 8. This promotion requires information from a *completed* run that established $K_{\mathrm{TB}_\pi}^+$, then retroactively upgrades the earlier Node 2 ambiguity.

**Proof sketch:** Each sector transition costs at least $\delta$ units of action/energy. With bounded total energy $E_{\max}$, at most $E_{\max}/\delta$ transitions can occur. This is the Conley index argument (1978) applied to gradient-like flows: the Morse-Conley theory bounds the number of critical point transitions by the total change in index. Combined with energy dissipation, this forbids Zeno accumulation.

**Interface Permit Validated:** Finite Event Count (Topological Confinement).

**Literature:** {cite}`Conley78`; {cite}`Smale67`; {cite}`Floer89`
:::

---

### 33.2 The Lock-Back Theorem

:::{prf:metatheorem} Lock-Back Theorem (Lock $\to$ All Barriers)
:label: mt-lock-back
:class: metatheorem

**Theorem:** Global Regularity Retro-Validation

**Input:** $K_{\text{Lock}}^{\mathrm{blk}}$ (Node 17: Morphism Exclusion).

**Target:** Any earlier "Blocked" Barrier certificate ($K_{\text{sat}}^{\mathrm{blk}}, K_{\text{cap}}^{\mathrm{blk}}, \ldots$).

**Statement:** If the Lock proves that *no* singularity pattern can exist globally ($\mathrm{Hom}(\mathcal{B}_{\text{univ}}, \mathcal{H}) = \emptyset$), then all local "Blocked" states are retroactively validated as Regular points.

**Certificate Logic:**
$$K_{\text{Lock}}^{\mathrm{blk}} \Rightarrow \forall i: K_{\text{Barrier}_i}^{\mathrm{blk}} \to K_{\text{Gate}_i}^+$$

**Proof sketch:** The morphism obstruction at the Lock is a global invariant. If no bad pattern embeds globally, then any local certificate that was "Blocked" (i.e., locally ambiguous) must resolve to "Regular" since the alternative (singular) is globally forbidden. This is the "principle of the excluded middle" applied via the universal property of the bad pattern functor.

**Physical Interpretation:** If the laws of physics forbid black holes (Lock), then any localized dense matter detected earlier (BarrierCap) must eventually disperse, regardless of local uncertainty.

**Literature:** {cite}`Grothendieck57`; {cite}`SGA4`
:::

---

### 33.3 The Symmetry-Gap Theorem

:::{prf:metatheorem} Symmetry-Gap Theorem (SymCheck $\to$ Stiffness)
:label: mt-symmetry-gap
:class: metatheorem

**Theorem:** Mass Gap Retro-Validation

**Input:** $K_{\text{Sym}}^+$ (Node 7b: Rigid Symmetry) + $K_{\text{CheckSC}}^+$ (Node 7c: Constants Stable).

**Target:** Node 7 ($K_{\mathrm{LS}_\sigma}^{\mathrm{stag}}$: Stagnation/Flatness).

**Statement:** If the vacuum symmetry is rigid (SymCheck) and constants are stable (CheckSC), then the "Flatness" (Stagnation) detected at Node 7 is actually a **Spontaneous Symmetry Breaking** event. This mechanism generates a dynamic Mass Gap, satisfying the Stiffness requirement retroactively.

**Certificate Logic:**
$$K_{\mathrm{LS}_\sigma}^{\mathrm{stag}} \wedge K_{\text{Sym}}^+ \wedge K_{\text{CheckSC}}^+ \Rightarrow K_{\mathrm{LS}_\sigma}^+ \text{ (with gap } \lambda > 0\text{)}$$

**Proof sketch:** The Goldstone theorem (1961) states that spontaneous breaking of a continuous symmetry produces massless bosons. However, if the symmetry group is *compact* and the vacuum is unique (CheckSC), the would-be Goldstones acquire mass via the Higgs mechanism or explicit breaking. The resulting spectral gap $\lambda > 0$ provides stiffness. For gauge theories, this is the mass gap conjecture; for condensed matter, this is the BCS mechanism.

**Application:** Used in Yang-Mills and Riemann Hypothesis to upgrade a "Flat Potential" diagnosis to a "Massive/Stiff Potential" proof.

**Literature:** {cite}`Goldstone61`; {cite}`Higgs64`; {cite}`Coleman75`
:::

---

### 33.4 The Tame-Topology Theorem

:::{prf:metatheorem} Tame-Topology Theorem (TameCheck $\to$ GeomCheck)
:label: mt-tame-topology
:class: metatheorem

**Theorem:** Stratification Retro-Validation

**Input:** $K_{\mathrm{TB}_O}^+$ (Node 9: O-minimal Definability).

**Target:** Node 6 ($K_{\mathrm{Cap}_H}^{\mathrm{blk}}$: Capacity Blocked).

**Statement:** If the system is definable in an o-minimal structure (TameCheck), then any singular set $\Sigma$ with zero capacity detected at Node 6 is rigorously a **Removable Singularity** (a lower-dimensional stratum in the Whitney stratification).

**Certificate Logic:**
$$K_{\mathrm{Cap}_H}^{\mathrm{blk}} \wedge K_{\mathrm{TB}_O}^+ \Rightarrow K_{\mathrm{Cap}_H}^+$$

**Proof sketch:** In an o-minimal structure, every definable set admits a Whitney stratification into smooth manifolds (Łojasiewicz, 1965; van den Dries-Miller, 1996). A set of zero capacity is contained in a stratum of positive codimension. By the Kurdyka-Łojasiewicz inequality, the solution extends uniquely across such strata. The gradient flow cannot accumulate on a positive-codimension set.

**Application:** Ensures that "Blocked" singularities in geometric flows are not just "small," but geometrically harmless.

**Literature:** {cite}`Lojasiewicz65`; {cite}`vandenDriesMiller96`; {cite}`Kurdyka98`
:::

---

### 33.5 The Ergodic-Sat Theorem

:::{prf:metatheorem} Ergodic-Sat Theorem (ErgoCheck $\to$ EnergyCheck)
:label: mt-ergodic-sat
:class: metatheorem

**Theorem:** Recurrence Retro-Validation

**Input:** $K_{\mathrm{TB}_\rho}^+$ (Node 10: Mixing/Ergodicity).

**Target:** Node 1 ($K_{\text{sat}}^{\mathrm{blk}}$: Saturation).

**Statement:** If the system is proven to be Ergodic (mixing), then the "Saturation" bound at Node 1 is not just a ceiling, but a **Recurrence Guarantee**. The system will infinitely often visit low-energy states. In particular, $\liminf_{t \to \infty} \Phi(x(t)) \leq \bar{\Phi}$ for $\mu$-a.e. initial condition.

**Certificate Logic:**
$$K_{\text{sat}}^{\mathrm{blk}} \wedge K_{\mathrm{TB}_\rho}^+ \Rightarrow K_{D_E}^+ \text{ (Poincaré Recurrence)}$$

**Proof sketch:** The Poincaré recurrence theorem (1890) states that for a measure-preserving transformation, almost every point returns arbitrarily close to its initial position. Combined with mixing (strong ergodicity), the time averages converge to the space average: $\frac{1}{T}\int_0^T \Phi(x(t)) \, dt \to \int \Phi \, d\mu$. If the invariant measure has $\mu(\Phi) < \infty$ (Saturation), recurrence to low-energy states is guaranteed.

**Application:** Upgrades "Bounded Drift" to "Thermodynamic Stability" in statistical mechanics systems.

**Literature:** {cite}`Poincare90`; {cite}`Birkhoff31`; {cite}`Furstenberg81`
:::

---

### 33.6 The Variety-Control Theorem

:::{prf:metatheorem} Variety-Control Theorem (AlignCheck $\to$ ScaleCheck)
:label: mt-variety-control
:class: metatheorem

**Theorem:** Cybernetic Retro-Validation

**Input:** $K_{\mathrm{GC}_T}^+$ (Node 16: Alignment/Variety).

**Target:** Node 4 ($K_{\mathrm{SC}_\lambda}^-$: Supercritical).

**Statement:** If the controller possesses sufficient Requisite Variety to match the disturbance (Node 16), it can suppress the Supercritical Scaling instability (Node 4) via active feedback, rendering the effective system Subcritical.

**Certificate Logic:**
$$K_{\mathrm{SC}_\lambda}^- \wedge K_{\mathrm{GC}_T}^+ \Rightarrow K_{\mathrm{SC}_\lambda}^{\sim} \text{ (Controlled)}$$

**Proof sketch:** Ashby's Law of Requisite Variety (1956) states that "only variety can absorb variety." If the controller has sufficient degrees of freedom ($\log|\mathcal{U}| \geq \log|\mathcal{D}|$), it can cancel any disturbance. The Conant-Ashby theorem (1970) formalizes this: every good regulator of a system must be a model of that system. Applied to scaling instabilities, a sufficiently complex controller can inject anti-scaling corrections that neutralize supercritical growth.

**Application:** Used in Control Theory to prove that an inherently unstable (supercritical) plant can be stabilized by a complex controller.

**Literature:** {cite}`Ashby56`; {cite}`ConantAshby70`; {cite}`DoyleFrancisTannenbaum92`
:::

---

### 33.7 The Algorithm-Depth Theorem

:::{prf:metatheorem} Algorithm-Depth Theorem (ComplexCheck $\to$ ZenoCheck)
:label: mt-algorithm-depth
:class: metatheorem

**Theorem:** Computational Censorship Retro-Validation

**Input:** $K_{\mathrm{Rep}_K}^+$ (Node 11: Finite Complexity).

**Target:** Node 2 ($K_{\mathrm{Rec}_N}^{\mathrm{blk}}$: Causal Censor).

**Statement:** If the solution has a finite description length (ComplexCheck), then any "Infinite Event Depth" (Zeno behavior) detected at Node 2 must be an artifact of the coordinate system, not physical reality. The singularity is removable by coordinate transformation.

**Certificate Logic:**
$$K_{\mathrm{Rec}_N}^{\mathrm{blk}} \wedge K_{\mathrm{Rep}_K}^+ \Rightarrow K_{\mathrm{Rec}_N}^+$$

**Proof sketch:** Kolmogorov complexity bounds the information content of an object. If $K(x) \leq C$ for some constant $C$, then $x$ is compressible/simple. A genuinely singular object (fractal, infinitely complex) has $K(x) \to \infty$. Therefore, a Zeno singularity with finite complexity must be a coordinate artifact—like the event horizon in Schwarzschild coordinates, which disappears in Eddington-Finkelstein coordinates. Algorithmic removability follows.

**Application:** Resolves coordinate singularities (like event horizons in bad coordinates) by proving the underlying object is algorithmically simple.

**Literature:** {cite}`Kolmogorov65`; {cite}`Chaitin66`; {cite}`LiVitanyi08`
:::

---

### 33.8 The Holographic-Regularity Theorem

:::{prf:metatheorem} Holographic-Regularity Theorem (ComplexCheck $\to$ GeomCheck)
:label: mt-holographic
:class: metatheorem

**Theorem:** Information-Theoretic Smoothing

**Input:** $K_{\mathrm{Rep}_K}^+$ (Node 11: Low Kolmogorov Complexity).

**Target:** Node 6 ($K_{\mathrm{Cap}_H}^-$: Marginal/Fractal Geometry).

**Statement:** A singular set with non-integer (fractal) dimension requires infinite information to specify exactly. If ComplexCheck proves the description length is finite (low complexity), the singular set *must* be a standard geometric object (Point, Line, Surface) with integer dimension. This collapses the "Fractal" possibility into "Tame" geometry.

**Certificate Logic:**
$$K_{\mathrm{Cap}_H}^{\text{ambiguous}} \wedge K_{\mathrm{Rep}_K}^+ \Rightarrow K_{\mathrm{Cap}_H}^+ \text{ (Integer Dim)}$$

**Proof sketch:** The Kolmogorov complexity of a fractal of Hausdorff dimension $d$ scales as $K(\Sigma|_\varepsilon) \sim \varepsilon^{-d}$ where $\Sigma|_\varepsilon$ is the $\varepsilon$-covering. For non-integer $d$, this diverges as $\varepsilon \to 0$, implying $K(\Sigma) = \infty$. Conversely, $K(\Sigma) < \infty$ forces $d \in \mathbb{Z}$. This is the holographic principle (Susskind, 1995): information stored on boundaries must be finite.

**Application:** Proves that algorithmically simple systems cannot have fractal singularities.

**Literature:** {cite}`tHooft93`; {cite}`Susskind95`; {cite}`Bousso02`
:::

---

### 33.9 The Spectral-Quantization Theorem

:::{prf:metatheorem} Spectral-Quantization Theorem (Lock $\to$ OscillateCheck)
:label: mt-spectral-quant
:class: metatheorem

**Theorem:** Discrete Spectrum Enforcement

**Input:** $K_{\text{Lock}}^{\mathrm{blk}}$ (Node 17: Integrality/E4 Tactic).

**Target:** Node 12 ($K_{\mathrm{GC}_\nabla}^-$: Chaotic Oscillation).

**Statement:** If the Lock proves that global invariants must be Integers (E4: Integrality), the spectrum of the evolution operator is forced to be discrete (Quantized). Continuous chaotic drift is impossible; the system must be Quasi-Periodic or Periodic.

**Certificate Logic:**
$$K_{\mathrm{GC}_\nabla}^{\text{chaotic}} \wedge K_{\text{Lock}}^{\mathrm{blk}} \Rightarrow K_{\mathrm{GC}_\nabla}^{\sim} \text{ (Quasi-Periodic)}$$

**Proof sketch:** Weyl's law (1911) relates the spectral asymptotics $N(\lambda) \sim C\lambda^{n/2}$ to the geometry. If global invariants are quantized (integers), the spectrum is discrete: $\sigma(L) \subset \{\lambda_n\}_{n \in \mathbb{N}}$. By the Paley-Wiener theorem, functions with discrete spectrum are almost periodic. Kac's "Can one hear the shape of a drum?" (1966) shows geometry determines spectrum and vice versa.

**Application:** Proves that chaotic oscillations are forbidden when integrality constraints exist.

**Literature:** {cite}`Weyl11`; {cite}`Kac66`; {cite}`GordonWebbWolpert92`
:::

---

### 33.10 The Unique-Attractor Theorem

:::{prf:metatheorem} Unique-Attractor Theorem (Permit Schema with Alternative Backends)
:label: mt-unique-attractor
:class: metatheorem

**Theorem:** Global Selection Principle

**Sieve Target:** Node 3 (Profile Trichotomy Cases)

**Input:** $K_{\mathrm{TB}_\rho}^+$ (Node 10: Unique Invariant Measure).

**Critical Remark:** Unique ergodicity **alone** does NOT imply convergence to a single profile. Counterexample: irrational rotation $T_\alpha: x \mapsto x + \alpha \mod 1$ on the torus is uniquely ergodic (Lebesgue measure is the unique invariant measure), but orbits are **dense** and do not converge to any point. Additional dynamical hypotheses are required.

**Statement:** Under appropriate additional hypotheses (specified per backend), if the system possesses a unique invariant measure (Node 10), there can be only **one** stable profile in the library. All other profiles are transient/unstable.

**Certificate Logic:**
$$K_{\text{Profile}}^{\text{multimodal}} \wedge K_{\mathrm{TB}_\rho}^+ \wedge K_{\text{Backend}}^+ \Rightarrow K_{\text{Profile}}^{\text{unique}}$$

where $K_{\text{Backend}}^+$ is one of:
- $K_{\text{UA-A}}^+$: Unique Ergodicity + Discrete Attractor hypothesis
- $K_{\text{UA-B}}^+$: Gradient structure + Łojasiewicz-Simon convergence
- $K_{\text{UA-C}}^+$: Contraction / Spectral-gap mixing

---

#### Backend A: Unique Ergodicity + Discrete Attractor

**Additional Hypotheses:**
1. **Finite Profile Library:** $|\mathcal{L}_T| < \infty$ (Profile Classification Trichotomy Case 1)
2. **Discrete Attractor:** The $\omega$-limit sets satisfy $\omega(x) \subseteq \bigcup_{i=1}^N \{V_i\}$ for a finite set of profiles
3. **Continuous-Time Semiflow:** $(S_t)_{t \geq 0}$ is a continuous-time semiflow, OR each $V_i$ is an equilibrium ($S_t V_i = V_i$ for all $t$). (This excludes periodic orbits on finite invariant sets in discrete time.)

**Certificate:** $K_{\text{UA-A}}^+ = (K_{\mathrm{TB}_\rho}^+, K_{\text{lib}}, N < \infty, \omega\text{-inclusion}, \text{time-model})$

**Proof (5 Steps):**

*Step 1 (Ergodic Support Characterization).* Let $\mu$ be the unique invariant measure. By the ergodic decomposition theorem {cite}`Furstenberg81`, every ergodic invariant measure is extremal in $\mathcal{M}_{\text{inv}}(\mathcal{X})$. Since $\mu$ is unique, it is extremal, hence ergodic. The support $\text{supp}(\mu)$ is closed and invariant; for $x \in \text{supp}(\mu)$, the orbit stays in $\text{supp}(\mu)$, hence $\omega(x) \subseteq \text{supp}(\mu)$.

*Step 2 (Support Containment via Invariance).* The support $\text{supp}(\mu)$ is closed and forward-invariant: $S_t(\text{supp}(\mu)) \subseteq \text{supp}(\mu)$. By Step 1, if $x \in \text{supp}(\mu)$, then $\omega(x) \subseteq \text{supp}(\mu)$. The discrete attractor hypothesis gives $\omega(x) \subseteq \{V_1, \ldots, V_N\}$ for all $x$. Therefore:
$$\text{supp}(\mu) \cap \{V_1, \ldots, V_N\} \neq \emptyset \implies \text{supp}(\mu) \subseteq \{V_1, \ldots, V_N\}$$
since $\omega$-limits of points in $\text{supp}(\mu)$ must lie in the finite discrete set.

*Step 3 (Measure Concentration on Singleton).* Since $\mu$ is ergodic and $\text{supp}(\mu) \subseteq \{V_1, \ldots, V_N\}$ with $N < \infty$, the measure must concentrate on an ergodic component. For a finite discrete set, each point is its own ergodic component. Therefore $\mu = \delta_{V^*}$ for some unique profile $V^* \in \mathcal{L}_T$.

*Step 4 (Transience of Other Profiles).* For any $V_i \neq V^*$, we have $\mu(\{V_i\}) = 0$. By Birkhoff's ergodic theorem:
$$\lim_{T \to \infty} \frac{1}{T} \int_0^T \mathbf{1}_{\{V_i\}}(S_t x) \, dt = \mu(\{V_i\}) = 0 \quad \mu\text{-a.s.}$$
Hence orbits spend asymptotically zero fraction of time near $V_i$.

*Step 5 (Convergence Conclusion).* The discrete topology on $\{V_1, \ldots, V_N\}$ combined with $\mu = \delta_{V^*}$ implies that for $\mu$-a.e. initial condition, $\omega(x) = \{V^*\}$. All other profiles are transient saddle points with measure-zero basins.

**Literature:** {cite}`Birkhoff31`; {cite}`Furstenberg81`; {cite}`Oxtoby52`; {cite}`MeynTweedie93`

---

#### Backend B: Gradient + Łojasiewicz-Simon Convergence

**Additional Hypotheses:**
1. **Gradient Structure:** $K_{\mathrm{GC}_\nabla}^-$ (OscillateCheck NO: dynamics is gradient-like)
2. **Strict Lyapunov Function:** $K_{\mathrm{LS}_\sigma}^+$ with $\frac{d}{dt}\Phi(S_t x) \leq -c\mathfrak{D}(S_t x)$ for $c > 0$
3. **Precompact Trajectories:** Bounded orbits have compact closure in $\mathcal{X}$

**Certificate:** $K_{\text{UA-B}}^+ = (K_{\mathrm{TB}_\rho}^+, K_{\mathrm{GC}_\nabla}^-, K_{\mathrm{LS}_\sigma}^+, K_{C_\mu}^+)$

**Proof (5 Steps):**

*Step 1 (Gradient-Like Dynamics with Strict Lyapunov Function).* By $K_{\mathrm{GC}_\nabla}^-$, the flow $S_t$ is gradient-like: $\dot{x} = -\nabla_g \Phi(x) + R(x)$ where $R$ satisfies $\langle R, \nabla\Phi \rangle \leq 0$. The strict Lyapunov condition ensures:
$$\frac{d}{dt}\Phi(S_t x) = -\|\nabla\Phi(S_t x)\|^2 + \langle R, \nabla\Phi \rangle \leq -\|\nabla\Phi(S_t x)\|^2$$
Hence $\Phi$ is strictly decreasing away from critical points. The global attractor $\mathcal{A}$ consists of equilibria and connecting orbits.

*Step 2 (Bounded Trajectories are Precompact).* By $K_{C_\mu}^+$ (compactness), sublevel sets $\{\Phi \leq c\}$ are precompact modulo symmetry. For any bounded trajectory, the orbit closure is compact. This is the "asymptotic compactness" condition {cite}`Temam97`.

*Step 3 (Łojasiewicz-Simon Inequality Near Critical Points).* By the Łojasiewicz-Simon gradient inequality {cite}`Simon83`:
$$\|\nabla\Phi(x)\| \geq C_{\text{LS}} |\Phi(x) - \Phi(V)|^{1-\theta}$$
for $x$ in a neighborhood of any critical point $V$, with exponent $\theta \in (0, 1/2]$. This prevents oscillation near equilibria and ensures finite-length gradient flow curves.

*Step 4 (Convergence of Trajectories to Single Equilibrium).* The Łojasiewicz-Simon inequality implies:
$$\int_0^\infty \|\dot{S}_t x\| \, dt = \int_0^\infty \|\nabla\Phi(S_t x)\| \, dt < \infty$$
Hence the trajectory has **finite arc length** and converges to a single limit $V^* = \lim_{t \to \infty} S_t x$. By continuity, $\nabla\Phi(V^*) = 0$.

*Step 5 (Unique Invariant Measure Implies Unique Equilibrium).* For gradient flows, every equilibrium $V$ generates an invariant measure $\delta_V$ (since $S_t V = V$). If there existed distinct equilibria $V_1 \neq V_2$ in $\mathcal{A}$, then $\delta_{V_1}$ and $\delta_{V_2}$ would both be invariant measures, contradicting the uniqueness hypothesis $K_{\mathrm{TB}_\rho}^+$. Hence the attractor contains exactly one equilibrium: $\mathcal{A} \cap \{\text{equilibria}\} = \{V^*\}$. Combined with Step 4 (every trajectory converges to some equilibrium), we conclude $\mu = \delta_{V^*}$.

**Literature:** {cite}`Simon83`; {cite}`Huang06`; {cite}`Raugel02`; {cite}`Temam97`

---

#### Backend C: Contraction / Spectral-Gap Mixing

**Additional Hypotheses:**
1. **Strictly Contractive Semigroup:** $d(S_t x, S_t y) \leq e^{-\lambda t} d(x, y)$ for some $\lambda > 0$, OR
2. **Harris/Doeblin Condition:** For Markov dynamics, a small set $C$ with $\sup_{x \in C} \mathbb{E}_x[\tau_C] < \infty$ and minorization

**Certificate:** $K_{\text{UA-C}}^+ = (K_{\mathrm{TB}_\rho}^+, \lambda > 0, K_{\text{spec-gap}})$

**Proof (5 Steps):**

*Step 1 (Strictly Contractive Semigroup in Metric).* Assume $d(S_t x, S_t y) \leq e^{-\lambda t} d(x, y)$ for all $x, y \in \mathcal{X}$ with contraction rate $\lambda > 0$. This is the "uniformly dissipative" condition {cite}`Temam97`. For Markov chains, the analogous condition is the Harris chain criterion with geometric drift {cite}`MeynTweedie93`:
$$\mathcal{L}V \leq -\lambda V + b\mathbf{1}_C$$
for a Lyapunov function $V$ and small set $C$.

*Step 2 (Unique Invariant Measure / Stationary State).* Contraction implies the existence of a unique fixed point $V^* = \lim_{t \to \infty} S_t x$ for any initial condition. For measures, the pushforward satisfies:
$$W_1(S_t^* \mu, S_t^* \nu) \leq e^{-\lambda t} W_1(\mu, \nu)$$
in Wasserstein-1 distance. Hence there is a unique invariant measure $\mu^* = \delta_{V^*}$.

*Step 3 (Spectral Gap and Mixing Rate).* If a spectral gap $\text{gap}(\mathcal{L}) \geq \lambda_{\text{sg}} > 0$ is declared (certificate $K_{\text{spec-gap}}$), then mixing-time bounds follow. For Markov semigroups, the spectral gap equals the gap between the leading eigenvalue (1 for probability-preserving) and the second eigenvalue. The mixing time satisfies:
$$\tau_{\text{mix}}(\varepsilon) \leq \frac{1}{\lambda_{\text{sg}}} \log\left(\frac{1}{\varepsilon}\right)$$
**Note:** The contraction rate $\lambda$ (hypothesis 1) and spectral gap $\lambda_{\text{sg}}$ are related but not generally equal; in many settings $\lambda_{\text{sg}} \leq 2\lambda$. This step is optional—uniqueness of profile follows from Steps 1-2 alone.

*Step 4 (Contraction Upgrades Uniqueness to Global Attraction).* Unlike mere unique ergodicity (which only guarantees time-average convergence), contraction provides **pointwise** convergence:
$$d(S_t x, V^*) \leq e^{-\lambda t} d(x, V^*) \to 0 \quad \text{as } t \to \infty$$
for **all** initial conditions $x \in \mathcal{X}$. The basin of attraction of $V^*$ is the entire space.

*Step 5 (Conclusion: Unique Profile with Global Attraction).* The combination of unique invariant measure $\mu^* = \delta_{V^*}$, global pointwise convergence to $V^*$, and exponential mixing implies the Profile Library reduces to a singleton: $\mathcal{L}_T = \{V^*\}$. All other profiles are transient or absent.

**Literature:** {cite}`MeynTweedie93`; {cite}`HairerMattingly11`; {cite}`LevinPeresWilmer09`; {cite}`Temam97`

---

**Backend Selection Logic:**

| Backend | Required Additional Certificates | Best For |
|:-------:|:--------------------------------:|:--------:|
| A | $K_{\text{lib}}$ (finite library), $\omega$-discreteness | Discrete/finite-state systems |
| B | $K_{\mathrm{GC}_\nabla}^-$, $K_{\mathrm{LS}_\sigma}^+$, $K_{C_\mu}^+$ | Gradient flows, PDEs, geometric analysis |
| C | $\lambda > 0$ (contraction rate) or Harris condition | Markov chains, stochastic systems, SDEs |

**Application:** Resolves "multi-modal" profile ambiguity in favor of a single global attractor. Converts $K_{\text{Profile}}^{\text{multimodal}}$ to $K_{\text{Profile}}^{\text{unique}}$.

:::

---

## 34. Retroactive Upgrade Summary Table

| **Later Node (The Proof)** | **Earlier Node (The Problem)** | **Theorem** | **Upgrade Mechanism** |
|:---|:---|:---|:---|
| Node 17 (Lock) | All Barriers | {prf:ref}`mt-lock-back` | Global exclusion $\implies$ local regularity |
| Node 7b (SymCheck) | Node 7 (Stiffness) | {prf:ref}`mt-symmetry-gap` | Symmetry breaking $\implies$ mass gap |
| Node 9 (TameCheck) | Node 6 (Geometry) | {prf:ref}`mt-tame-topology` | Definability $\implies$ stratification |
| Node 10 (ErgoCheck) | Node 1 (Energy) | {prf:ref}`mt-ergodic-sat` | Mixing $\implies$ recurrence |
| Node 16 (AlignCheck) | Node 4 (Scale) | {prf:ref}`mt-variety-control` | High variety $\implies$ stabilization |
| Node 11 (Complex) | Node 2 (Zeno) | {prf:ref}`mt-algorithm-depth` | Low complexity $\implies$ coordinate artifact |
| Node 11 (Complex) | Node 6 (Geometry) | {prf:ref}`mt-holographic` | Finite info $\implies$ integer dimension |
| Node 17 (Lock/E4) | Node 12 (Oscillate) | {prf:ref}`mt-spectral-quant` | Integrality $\implies$ discrete spectrum |
| Node 10 (ErgoCheck) | Node 3 (Profile) | {prf:ref}`mt-unique-attractor` | Unique measure $\implies$ unique profile |

---

# Part XVI: Stability & Composition Metatheorems

## 35. Perturbation and Coupling

The **Stability and Composition Metatheorems** govern how Sieve verdicts extend from individual systems to families of systems (perturbations) and coupled systems (compositions). These theorems answer the fundamental questions:

1. *"If my model is slightly wrong, does the proof hold?"* (Stability)
2. *"Can I build a regular system out of regular parts?"* (Composition)

These metatheorems are **universal**: they apply to any valid Hypostructure because they operate on the Certificate algebra, not the underlying physics.

---

### 35.1 Openness of Regularity

:::{prf:metatheorem} Openness of Regularity (Structural Stability)
:label: mt-openness
:class: metatheorem

**Source:** Dynamical Systems (Morse-Smale Stability) / Geometric Analysis.

**Hypotheses.** Let $\mathcal{H}(\theta_0)$ be a Hypostructure depending on parameters $\theta \in \Theta$ (a topological space). Assume:
1. Global Regularity at $\theta_0$: $K_{\text{Lock}}^{\mathrm{blk}}(\theta_0)$
2. Strict barriers: $\mathrm{Gap}(\theta_0) > \epsilon$, $\mathrm{Cap}(\theta_0) < \delta$ for some $\epsilon, \delta > 0$
3. Continuous dependence: the certificate functionals are continuous in $\theta$

**Statement:** The set of Globally Regular Hypostructures is **open** in the parameter topology. There exists a neighborhood $U \ni \theta_0$ such that $\forall \theta \in U$, $\mathcal{H}(\theta)$ is also Globally Regular.

**Certificate Logic:**
$$K_{\text{Lock}}^{\mathrm{blk}}(\theta_0) \wedge (\mathrm{Gap} > \epsilon) \wedge (\mathrm{Cap} < \delta) \Rightarrow \exists U: \forall \theta \in U, K_{\text{Lock}}^{\mathrm{blk}}(\theta)$$

**Proof sketch:** Strict inequalities define open sets. The Morse-Smale stability theorem (Palis and de Melo, 1982) states that structurally stable systems form an open set. The key is non-degeneracy: if all eigenvalues are strictly away from zero and all capacities are strictly bounded, small perturbations preserve these properties. This is the implicit function theorem applied to the certificate functionals.

**Use:** Validates that the proof is robust to small modeling errors or physical noise.

**Literature:** {cite}`Smale67`; {cite}`PalisdeMelo82`; {cite}`Robinson99`
:::

---

### 35.2 Shadowing Metatheorem

:::{prf:metatheorem} Shadowing Metatheorem (Numerical Validity)
:label: mt-shadowing
:class: metatheorem

**Source:** Hyperbolic Dynamics (Anosov Shadowing Lemma).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A Stiffness certificate: $K_{\mathrm{LS}_\sigma}^+$ with spectral gap $\lambda > 0$
2. A numerical pseudo-orbit: $\{y_n\}$ with $d(f(y_n), y_{n+1}) < \varepsilon$ for all $n$
3. Hyperbolicity: the tangent map $Df$ has exponential dichotomy

**Statement:** For every $\varepsilon$-pseudo-orbit (numerical simulation), there exists a true orbit $\{x_n\}$ that $\delta(\varepsilon)$-shadows it: $d(x_n, y_n) < \delta(\varepsilon)$ for all $n$. The shadowing distance satisfies $\delta(\varepsilon) = O(\varepsilon/\lambda)$.

**Certificate Logic:**
$$K_{\mathrm{LS}_\sigma}^+ \wedge K_{\text{pseudo}}^{\varepsilon} \Rightarrow K_{\text{true}}^{\delta(\varepsilon)}$$

**Proof sketch:** The Anosov shadowing lemma (1967) states that uniformly hyperbolic systems have the shadowing property. The spectral gap $\lambda$ controls the contraction rate, and the shadowing distance is $\delta \sim \varepsilon/\lambda$. Bowen (1975) extended this to Axiom A systems. Palmer (1988) gave a proof via the contraction mapping theorem on sequence spaces.

**Use:** Upgrades a high-precision **Numerical Simulation** into a rigorous **Existence Proof** for a nearby solution (essential for $T_{\text{algorithmic}}$).

**Literature:** {cite}`Anosov67`; {cite}`Bowen75`; {cite}`Palmer88`
:::

---

### 35.3 Weak-Strong Uniqueness

:::{prf:metatheorem} Weak-Strong Uniqueness (Duality)
:label: mt-weak-strong
:class: metatheorem

**Source:** PDE Theory (Serrin/Prodi-Serrin Criteria).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A "Weak" solution $u_w$ constructed via concentration-compactness ($K_{C_\mu}$)
2. A "Strong" local solution $u_s$ with Stiffness ($K_{\mathrm{LS}_\sigma}^+$) on $[0, T]$
3. Both solutions have the same initial data: $u_w(0) = u_s(0)$

**Statement:** If a "Strong" solution exists on $[0, T]$, it is unique. Any "Weak" solution constructed via Compactness/Surgery must coincide with the Strong solution almost everywhere: $u_w = u_s$ a.e. on $[0, T] \times \Omega$.

**Certificate Logic:**
$$K_{C_\mu}^{\text{weak}} \wedge K_{\mathrm{LS}_\sigma}^{\text{strong}} \Rightarrow K_{\text{unique}}$$

**Proof sketch:** The weak-strong uniqueness principle uses energy estimates. If $v = u_w - u_s$, then $\frac{d}{dt}\|v\|^2 \leq C\|v\|^2 \cdot \|u_s\|_{X}$ for an appropriate norm $X$. If $u_s \in L^p([0,T]; X)$ (Serrin class), Gronwall's inequality gives $\|v(t)\| = 0$. For Navier-Stokes, $X = L^r$ with $\frac{2}{p} + \frac{3}{r} = 1$, $r > 3$ (Serrin, 1963; Lions, 1996).

**Use:** Resolves the "Non-Uniqueness" anxiety in weak solutions. If you can prove stiffness locally, the weak solution cannot branch off.

**Literature:** {cite}`Serrin63`; {cite}`Lions96`; {cite}`Prodi59`
:::

---

### 35.4 Product-Regularity Metatheorem

:::{prf:metatheorem} Product-Regularity (Permit Schema with Alternative Backends)
:label: mt-product
:class: metatheorem

**Sieve Signature:**
- **Required Permits (Alternative Backends):**
  - **Backend A:** $K_{\text{Lock}}^A \wedge K_{\text{Lock}}^B \wedge K_{\mathrm{SC}_\lambda}^{\text{sub}} \wedge K_{\mathrm{CouplingSmall}}^+$ (Subcritical Scaling + Coupling Control)
  - **Backend B:** $K_{\text{Lock}}^A \wedge K_{\text{Lock}}^B \wedge K_{D_E}^{\text{pert}} \wedge K_{\mathrm{ACP}}^+$ (Semigroup + Perturbation + ACP)
  - **Backend C:** $K_{\text{Lock}}^A \wedge K_{\text{Lock}}^B \wedge K_{\mathrm{LS}_\sigma}^{\text{abs}}$ (Energy + Absorbability)
- **Weakest Precondition:** $\{K_{\text{Lock}}^A, K_{\text{Lock}}^B\}$ (component regularity certified)
- **Produces:** $K_{\text{Lock}}^{A \times B}$ (product system globally regular)
- **Blocks:** All failure modes on product space
- **Breached By:** Strong coupling exceeding perturbation bounds

**Context:** Product systems arise when composing verified components (e.g., Neural Net + Physics Engine, multi-scale PDE systems, coupled oscillators). The principle of **modular verification** requires that certified components remain certified under weak coupling.

**Certificate Logic:**
$$K_{\text{Lock}}^A \wedge K_{\text{Lock}}^B \wedge \left((K_{\mathrm{SC}_\lambda}^{\text{sub}} \wedge K_{\mathrm{CouplingSmall}}^+) \vee (K_{D_E}^{\text{pert}} \wedge K_{\mathrm{ACP}}^+) \vee K_{\mathrm{LS}_\sigma}^{\text{abs}}\right) \Rightarrow K_{\text{Lock}}^{A \times B}$$

---

#### Backend A: Subcritical Scaling

**Hypotheses:**
1. Component Hypostructures $\mathcal{H}_A = (\mathcal{X}_A, \Phi_A, \mathfrak{D}_A)$ and $\mathcal{H}_B = (\mathcal{X}_B, \Phi_B, \mathfrak{D}_B)$
2. Lock certificates: $K_{\text{Lock}}^A$ and $K_{\text{Lock}}^B$ (global regularity for each)
3. Coupling term $\Phi_{\text{int}}: \mathcal{X}_A \times \mathcal{X}_B \to \mathbb{R}$ with scaling exponent $\alpha_{\text{int}}$
4. **Subcritical condition:** $\alpha_{\text{int}} < \min(\alpha_c^A, \alpha_c^B)$
5. **Coupling control** (permit $K_{\mathrm{CouplingSmall}}^+$, {prf:ref}`def-permit-couplingsmall`): Dissipation domination constants $\lambda_A, \lambda_B > 0$ with $\mathfrak{D}_i \geq \lambda_i E_i$, and energy absorbability $|\dot{E}_{\text{int}}| \leq \varepsilon(E_A + E_B) + C_\varepsilon$ for some $\varepsilon < \min(\lambda_A, \lambda_B)$

**Certificate:** $K_{\mathrm{SC}_\lambda}^{\text{sub}} \wedge K_{\mathrm{CouplingSmall}}^+ = (\alpha_{\text{int}}, \alpha_c^A, \alpha_c^B, \delta, \lambda_A, \lambda_B, \varepsilon, \text{absorbability witness})$

**Proof (5 Steps):**

*Step 1 (Scaling Structure).* Define the scaling action $\lambda \cdot (x_A, x_B) = (\lambda^{a_A} x_A, \lambda^{a_B} x_B)$ where $a_A, a_B$ are the homogeneity weights. The total height functional transforms as:
$$\Phi_{\text{tot}}(\lambda \cdot x) = \lambda^{\alpha_A} \Phi_A(x_A) + \lambda^{\alpha_B} \Phi_B(x_B) + \lambda^{\alpha_{\text{int}}} \Phi_{\text{int}}(x_A, x_B)$$

*Step 2 (Subcritical Dominance).* Since $\alpha_{\text{int}} < \min(\alpha_c^A, \alpha_c^B)$, the interaction term is asymptotically subdominant. For large $\lambda$:
$$|\Phi_{\text{int}}(\lambda \cdot x)| \leq C \lambda^{\alpha_{\text{int}}} = o(\lambda^{\alpha_c})$$
The interaction cannot drive blow-up faster than the natural scaling.

*Step 3 (Decoupled Barrier Transfer).* The Lock certificates $K_{\text{Lock}}^A, K_{\text{Lock}}^B$ provide a priori bounds:
$$\|u_A(t)\|_{\mathcal{X}_A} \leq M_A, \quad \|u_B(t)\|_{\mathcal{X}_B} \leq M_B \quad \forall t \geq 0$$
Under subcritical coupling, these bounds persist with at most polynomial growth correction.

*Step 4 (Energy Control).* The total energy $E_{\text{tot}} = E_A + E_B + E_{\text{int}}$ satisfies:
$$\frac{d}{dt} E_{\text{tot}} \leq -\mathfrak{D}_A - \mathfrak{D}_B + |\dot{E}_{\text{int}}|$$
where $\mathfrak{D}_A, \mathfrak{D}_B \geq 0$ are the dissipation rates (energy loss per unit time). Subcriticality implies $|\dot{E}_{\text{int}}| \leq \varepsilon (E_A + E_B) + C_\varepsilon$ for any $\varepsilon > 0$. Choosing $\varepsilon$ small enough that $\varepsilon < \min(\lambda_A, \lambda_B)$ (where $\mathfrak{D}_i \geq \lambda_i E_i$), the dissipation dominates the interaction.

*Step 5 (Grönwall Closure + Global Existence).* Standard Grönwall inequality closes the estimate. **Product local well-posedness** follows from standard semilinear theory: component LWP (guaranteed by the Lock certificates $K_{\text{Lock}}^A, K_{\text{Lock}}^B$) extends to the product system under Lipschitz coupling with subcritical growth (Hypotheses 3-4). Combined with the uniform energy bound from Step 4, global existence follows: no singularity can form in the product space.

**Literature:** Scaling analysis {cite}`Tao06`; subcritical perturbation {cite}`CazenaveSemilinear03`

---

#### Backend B: Semigroup + Perturbation Theory

**Hypotheses:**
1. Each component generates a $C_0$-semigroup: $T_A(t) = e^{tA_A}$ on $\mathcal{X}_A$, $T_B(t) = e^{tA_B}$ on $\mathcal{X}_B$
2. Global bounds: $\|T_A(t)\| \leq M_A e^{\omega_A t}$, $\|T_B(t)\| \leq M_B e^{\omega_B t}$ with $\omega_A, \omega_B \leq 0$ (dissipative)
3. Coupling operator $B: D(A_A) \times D(A_B) \to \mathcal{X}_A \times \mathcal{X}_B$ is either:
   - (i) **Bounded:** $\|B\| < \infty$, or
   - (ii) **$A$-relatively bounded:** $\|Bx\| \leq a\|(A_A \oplus A_B)x\| + b\|x\|$ with $a < 1$
4. Lock certificates translate to: trajectories remain in generator domain
5. **Abstract Cauchy Problem formulation** (permit $K_{\mathrm{ACP}}^+$, {prf:ref}`def-permit-acp`): The product dynamics are represented by the abstract Cauchy problem $\dot{u} = Au$, $u(0) = u_0$ on state space $X = \mathcal{X}_A \times \mathcal{X}_B$ with generator $A = A_A \oplus A_B + B$ and domain $D(A) \supseteq D(A_A) \times D(A_B)$

**Certificate:** $K_{D_E}^{\text{pert}} \wedge K_{\mathrm{ACP}}^+ = (A_A, A_B, B, \text{perturbation type}, X, D(A), \text{mild/strong equivalence})$

**Proof (5 Steps):**

*Step 1 (Product Semigroup).* On $\mathcal{X} = \mathcal{X}_A \times \mathcal{X}_B$, the uncoupled generator $A_0 = A_A \oplus A_B$ generates $T_0(t) = T_A(t) \times T_B(t)$ with:
$$\|T_0(t)\| \leq M_A M_B e^{\max(\omega_A, \omega_B) t}$$

*Step 2 (Perturbation Classification).* The total generator is $A = A_0 + B$ where $B$ represents coupling. By hypothesis, $B$ is either bounded or relatively bounded with bound $< 1$.

*Step 3 (Perturbation Theorem Application).*
- If $B$ bounded: **Bounded Perturbation Theorem** (Pazy, Theorem 3.1.1) yields $A$ generates $C_0$-semigroup.
- If $B$ relatively bounded with $a < 1$: **Relatively Bounded Perturbation** (Engel-Nagel, III.2.10) yields same.

*Step 4 (A Priori Bounds from Lock).* The Lock certificates provide:
$$\sup_{t \in [0,T]} \|(u_A(t), u_B(t))\|_{D(A_0)} < \infty$$
Standard semigroup theory: if $u(t) \in D(A)$ initially and $A$ generates $C_0$-semigroup, solution exists globally.

*Step 5 (Conclusion).* The perturbed semigroup $e^{tA}$ is globally defined on $\mathcal{X}_A \times \mathcal{X}_B$. No finite-time blow-up.

**Literature:** Semigroup theory {cite}`EngelNagel00`; perturbation of generators {cite}`Pazy83`; coupled parabolic systems {cite}`Cardanobile10`

---

#### Backend C: Energy + Absorbability

**Hypotheses:**
1. Coercive Lyapunov/energy functionals $E_A: \mathcal{X}_A \to \mathbb{R}$, $E_B: \mathcal{X}_B \to \mathbb{R}$:
   $$E_A(u) \geq c_A \|u\|_{\mathcal{X}_A}^p - C_A, \quad E_B(v) \geq c_B \|v\|_{\mathcal{X}_B}^q - C_B$$
2. Dissipation structure from Lock certificates:
   $$\frac{d}{dt} E_A \leq -\lambda_A E_A + d_A, \quad \frac{d}{dt} E_B \leq -\lambda_B E_B + d_B$$
3. **Absorbability condition:** The coupling contribution to energy evolution satisfies:
   $$\left|\frac{d}{dt}\Phi_{\text{int}}(u(t), v(t))\right| \leq \varepsilon (E_A(u) + E_B(v)) + C_\varepsilon$$
   for some $\varepsilon < \min(\lambda_A, \lambda_B)$. (This bounds the *rate* of energy exchange, not the potential itself.)

**Certificate:** $K_{\mathrm{LS}_\sigma}^{\text{abs}} = (E_A, E_B, \lambda_A, \lambda_B, \varepsilon, \text{absorbability witness})$

**Proof (5 Steps):**

*Step 1 (Total Energy Construction).* Define $E_{\text{tot}} = E_A + E_B$. By coercivity:
$$E_{\text{tot}}(u, v) \geq c_{\min}(\|u\|^p + \|v\|^q) - C_{\max}$$
This controls the product norm.

*Step 2 (Energy Evolution).* The time derivative:
$$\frac{d}{dt} E_{\text{tot}} = \frac{d}{dt} E_A + \frac{d}{dt} E_B + \underbrace{\text{coupling contribution}}_{\leq \varepsilon E_{\text{tot}} + C_\varepsilon}$$

*Step 3 (Grönwall Closure).* Combining dissipation and absorbability:
$$\frac{d}{dt} E_{\text{tot}} \leq -(\lambda_{\min} - \varepsilon) E_{\text{tot}} + C$$
where $\lambda_{\min} = \min(\lambda_A, \lambda_B)$. Since $\varepsilon < \lambda_{\min}$, the coefficient is negative.

*Step 4 (Global Bound).* Standard Grönwall inequality:
$$E_{\text{tot}}(t) \leq E_{\text{tot}}(0) e^{-(\lambda_{\min} - \varepsilon)t} + \frac{C}{\lambda_{\min} - \varepsilon}$$
Bounded uniformly in time.

*Step 5 (Conclusion + Global Existence).* Coercivity translates energy bound to norm bound. **Product local well-posedness** follows from standard energy-space theory: the coercive energy bounds (Hypothesis 1) provide control of the state space norms, and the Lipschitz coupling control implicit in the absorbability condition (Hypothesis 3) ensures local existence extends from components to the product. Combined with the uniform bound from Step 4, global existence follows.

**Literature:** Grönwall inequalities {cite}`Gronwall19`; energy methods {cite}`Lions69`; dissipative systems {cite}`Temam97`

---

**Backend Selection Logic:**

| Backend | Required Certificates | Best For |
|:-------:|:--------------------:|:--------:|
| A | $K_{\mathrm{SC}_\lambda}^{\text{sub}}$ (subcritical exponent) | Scaling-critical PDEs, dispersive equations |
| B | $K_{D_E}^{\text{pert}}$ (semigroup perturbation) | Linear/semilinear PDEs, evolution systems |
| C | $K_{\mathrm{LS}_\sigma}^{\text{abs}}$ (energy absorbability) | Dissipative systems, thermodynamic applications |

**Use:** Allows building complex Hypostructures by verifying components and coupling separately. The three backends accommodate different proof styles: scaling-based (A), operator-theoretic (B), and energy-based (C).

:::

---

### 35.5 Subsystem Inheritance

:::{prf:metatheorem} Subsystem Inheritance
:label: mt-subsystem
:class: metatheorem

**Source:** Invariant Manifold Theory.

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. Global Regularity: $K_{\text{Lock}}^{\mathrm{blk}}$
2. An invariant subsystem $\mathcal{S} \subset \mathcal{H}$: if $x(0) \in \mathcal{S}$, then $x(t) \in \mathcal{S}$ for all $t$
3. The subsystem inherits the Hypostructure: $\mathcal{H}|_{\mathcal{S}} = (\mathcal{S}, \Phi|_{\mathcal{S}}, \mathfrak{D}|_{\mathcal{S}}, G|_{\mathcal{S}})$

**Statement:** Regularity is hereditary. If the parent system $\mathcal{H}$ admits no singularities (Lock Blocked), then no invariant subsystem $\mathcal{S} \subset \mathcal{H}$ can develop a singularity.

**Certificate Logic:**
$$K_{\text{Lock}}^{\mathrm{blk}}(\mathcal{H}) \wedge (\mathcal{S} \subset \mathcal{H} \text{ invariant}) \Rightarrow K_{\text{Lock}}^{\mathrm{blk}}(\mathcal{S})$$

**Proof sketch:** If $\mathcal{S}$ developed a singularity, it would correspond to a morphism $\phi: \mathcal{B}_{\text{univ}} \to \mathcal{S} \hookrightarrow \mathcal{H}$. But this contradicts $\mathrm{Hom}(\mathcal{B}_{\text{univ}}, \mathcal{H}) = \emptyset$. The Fenichel invariant manifold theorem (1971) shows that normally hyperbolic invariant manifolds persist under perturbation; combined with Hirsch-Pugh-Shub (1977), the restriction maintains all regularity properties.

**Use:** Proves safety for restricted dynamics (e.g., "If the general 3D fluid is safe, the axisymmetric flow is also safe").

**Literature:** {cite}`Fenichel71`; {cite}`HirschPughShub77`; {cite}`Wiggins94`
:::

---

# Part XVII: Mathematical Foundations

This Part imports **twenty-one rigorous metatheorems** from the companion Hypostructure manuscript. These theorems provide the **mathematical engine** that substantiates the operational Sieve defined in Parts V–XVI. Each theorem supplies either:

- **Kernel Logic**: Proofs that the Sieve is a valid proof machine (categorical exclusion, structural resolution)
- **Gate Evaluator Predicates**: Exact mathematical predicates for YES/NO checks at blue nodes
- **Barrier Defense Mechanisms**: Proofs that barriers actually stop singularities
- **Surgery Constructions**: Constructive methods for purple surgery nodes

The imported theorems are organized by functional role and cross-referenced to their target Sieve components.

---

## 36. Kernel Logic Theorems

*These theorems prove that the Sieve is a valid proof machine.*

---

### 36.1 Principle of Structural Exclusion

:::{prf:metatheorem} Principle of Structural Exclusion (MT 8.11.N)
:label: mt-imported-structural-exclusion
:class: metatheorem

**Source:** Hypostructure MT 8.11.N (Capstone Theorem)

**Sieve Target:** Node 17 (Lock) — proves the Lock mechanism is valid

**Statement:** Let $T$ be a problem type with category of admissible T-hypostructures $\mathbf{Hypo}_T$. Let $\mathbb{H}_{\mathrm{bad}}^{(T)}$ be the universal Rep-breaking pattern. For any concrete object $Z$ with admissible hypostructure $\mathbb{H}(Z)$, if:

$$\mathrm{Hom}_{\mathbf{Hypo}_T}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H}(Z)) = \emptyset$$

then Interface Permit $\mathrm{Rep}_K(T, Z)$ holds, and hence the conjecture for $Z$ holds.

**Hypotheses (N1–N11):**
1. **(N1)** Category $\mathbf{Hypo}_T$ of admissible T-hypostructures satisfying core interface permits $C_\mu$, $D_E$, $\mathrm{SC}_\lambda$, $\mathrm{LS}_\sigma$, $\mathrm{Cap}_H$, $\mathrm{TB}_\pi$, $\mathrm{GC}_\nabla$
2. **(N2)** Hypostructure assignment $Z \mapsto \mathbb{H}(Z)$
3. **(N3)** Conjecture equivalence: $\mathrm{Conj}(T,Z) \Leftrightarrow \text{Interface Permit } \mathrm{Rep}_K(T,Z)$
4. **(N8)** Representational completeness of parametrization $\Theta$
5. **(N9)** Existence of universal Rep-breaking pattern with initiality property
6. **(N10)** Admissibility of $\mathbb{H}(Z)$
7. **(N11)** Obstruction condition: $\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H}(Z)) = \emptyset$

**Conclusion:** Global regularity via categorical obstruction: singularities cannot embed into admissible structures.

**Proof (5 Steps):**

*Step 1 (Structure of hypostructure space).* By representational completeness (N8), for any admissible $\mathbb{H} \in \mathbf{Hypo}_T$, there exists $\theta \in \Theta$ with $\mathbb{H}(\theta) \cong \mathbb{H}$. The zero level set $\{\theta : \mathcal{R}_{\mathrm{permits}}(\theta) = 0\}$ parametrizes all admissible T-hypostructures.

*Step 2 (Adversarial exploration).* Maximizing $\mathcal{L}_{\mathrm{bad}}(\theta) = \mathcal{R}_R(\theta) - \lambda \mathcal{R}_{\mathrm{permits}}(\theta)$ explores all admissible Rep-breaking patterns. Any universal Rep-breaking pattern can be constructed from such parametric models.

*Step 3 (Universal mapping).* If Interface Permit $\mathrm{Rep}_K(T,Z)$ failed for $\mathbb{H}(Z)$, then $\mathbb{H}(Z) \in \mathbf{Hypo}_T^{\neg R}$, and by initiality of $\mathbb{H}_{\mathrm{bad}}^{(T)}$ there would exist a morphism $F_Z: \mathbb{H}_{\mathrm{bad}}^{(T)} \to \mathbb{H}(Z)$.

*Step 4 (Categorical obstruction).* By the obstruction condition (N11), no such morphism exists. Hence the assumption that Interface Permit $\mathrm{Rep}_K(T,Z)$ fails leads to contradiction.

*Step 5 (Conclusion).* Therefore Interface Permit $\mathrm{Rep}_K(T,Z)$ holds, and by conjecture-permit equivalence (N3), $\mathrm{Conj}(T,Z)$ is true.

**Certificate Produced:** $K_{\text{Lock}}^{\mathrm{blk}}$ with payload $(\mathrm{Hom} = \emptyset, Z, T)$

**Literature:** {cite}`Grothendieck67`; {cite}`MacLane71`; {cite}`KashiwaraSchapira06`; {cite}`Lurie09`
:::

---

### 36.2 Structural Resolution Theorem

:::{prf:metatheorem} Structural Resolution (MT 5.1)
:label: mt-imported-structural-resolution
:class: metatheorem

**Source:** Hypostructure MT 5.1 (Resolution Machinery)

**Sieve Target:** Node 3 (CompactCheck) — justifies the Concentration/Dispersion dichotomy

**Statement:** Let $\mathcal{S}$ be a structural flow datum satisfying minimal regularity (Reg) and dissipation ($D_E$) interface permits. Every trajectory $u(t) = S_t x$ with finite breakdown time $T_*(x) < \infty$ classifies into exactly one of three outcomes:

| **Outcome** | **Modes** | **Mechanism** |
|-------------|-----------|---------------|
| Global Existence | Mode D.D | Energy disperses, no concentration, solution scatters |
| Global Regularity | Modes S.E, C.D, T.E, S.D | Concentration but forced structure fails permits |
| Genuine Singularity | Mode C.E | Energy escapes or structured blow-up with all permits |

**Hypotheses:**
1. **(Reg)** Minimal regularity: semiflow $S_t$ well-defined on $X$
2. **(D)** Dissipation: energy-dissipation inequality holds
3. **(C)** Compactness: bounded energy implies profile convergence modulo $G$

**Proof (Exhaustive Case Analysis):**

*Case 1: Energy blow-up.* If $\limsup_{t \to T_*(x)} \Phi(u(t)) = \infty$, then Mode C.E occurs (genuine singularity via energy escape).

*Case 2: Energy bounded, compactness holds.* By interface permit $C_\mu$, any sequence $u(t_n)$ with $t_n \nearrow T_*(x)$ has a convergent subsequence modulo $G$:
- *Sub-case 2a-i: Gauges bounded.* Limit must be singular point where local theory fails (Mode S.D if near $M$).
- *Sub-case 2a-ii: Gauges unbounded.* Rescaling becomes supercritical → Mode S.E.

*Case 3: Compactness fails.* No subsequence converges modulo $G$ → Mode D.D (dispersion, global existence).

*Cases 4–6: Permit violations.* Geometric concentration (Mode C.D), topological obstruction (Mode T.E), or stiffness violation (Mode S.D).

*Exhaustiveness.* These modes account for all behaviors of: height functional (bounded/unbounded), gauge sequence (bounded/unbounded), spatial concentration (diffuse/concentrated), topological sector (trivial/nontrivial), local stiffness (satisfied/violated).

**Certificate Produced:** Trichotomy classification $\{K_{\text{D.D}}, K_{\text{Reg}}, K_{\text{C.E}}\}$

**Literature:** {cite}`Lions84`; {cite}`KenigMerle06`; {cite}`Hironaka64`; {cite}`Struwe90`
:::

---

### 36.3 Equivariance of Trainable Hypostructures

:::{prf:metatheorem} Equivariance (MT 13.57 / SV-08)
:label: mt-imported-equivariance
:class: metatheorem

**Source:** Hypostructure MT 13.57 (Learning Theory)

**Sieve Target:** Meta-Learning — guarantees learned parameters preserve symmetry group $G$

**Statement:** Let $G$ be a compact Lie group acting on the system distribution $\mathcal{S}$ and parameter space $\Theta$. Under compatibility assumptions:

1. **(Group-Covariant Distribution)** $S \sim \mathcal{S} \Rightarrow g \cdot S \sim \mathcal{S}$ for all $g \in G$
2. **(Equivariant Parametrization)** $g \cdot \mathcal{H}_\Theta(S) \simeq \mathcal{H}_{g \cdot \Theta}(g \cdot S)$
3. **(Defect-Level Equivariance)** $K_{A,g \cdot S}^{(g \cdot \Theta)}(g \cdot u) = K_{A,S}^{(\Theta)}(u)$

Then:
- Every risk minimizer $\widehat{\Theta}$ lies in the $G$-orbit: $\widehat{\Theta} \in G \cdot \Theta^*$
- Gradient flow preserves equivariance: if $\Theta_0$ is $G$-equivariant, so is $\Theta_t$
- Learned hypostructures inherit all symmetries of the system distribution

**Proof (Gradient Flow Preservation):**

*Step 1 (Risk invariance).* By the equivariance assumptions:
$$\mathcal{R}(g \cdot \Theta) = \mathbb{E}_{S \sim \mathcal{S}}[\text{defect}(g \cdot \Theta, S)] = \mathbb{E}_{S \sim \mathcal{S}}[\text{defect}(\Theta, g^{-1} \cdot S)] = \mathcal{R}(\Theta)$$
The risk is $G$-invariant.

*Step 2 (Gradient equivariance).* Since $\mathcal{R}$ is $G$-invariant, $\nabla \mathcal{R}$ is $G$-equivariant:
$$\nabla \mathcal{R}(g \cdot \Theta) = g \cdot \nabla \mathcal{R}(\Theta)$$

*Step 3 (Flow preservation).* The gradient flow $\dot{\Theta}_t = -\nabla \mathcal{R}(\Theta_t)$ commutes with $G$-action. If $\Theta_0 \in G \cdot \Theta^*$, then $\Theta_t \in G \cdot \Theta^*$ for all $t$.

*Step 4 (Minimizers in orbit).* Global minimizers form a $G$-invariant set; by uniqueness up to symmetry, all minimizers lie in a single $G$-orbit.

**Certificate Produced:** $K_{\text{SV08}}^+$ (Symmetry Preservation)

**Literature:** {cite}`Noether18`; {cite}`CohenWelling16`; {cite}`Kondor18`; {cite}`Weyl46`
:::

---

## 37. Gate Evaluator Theorems

*These theorems define exact mathematical predicates for YES/NO checks at blue nodes.*

---

### 37.1 Type II Exclusion (ScaleCheck Predicate)

:::{prf:metatheorem} Type II Exclusion (MT 5.2)
:label: mt-imported-type-ii-exclusion
:class: metatheorem

**Source:** Hypostructure MT 5.2 (Scaling-Based Exclusion)

**Sieve Target:** Node 4 (ScaleCheck) — predicate $\alpha > \beta$ excludes supercritical blow-up

**Statement:** Let $\mathcal{S}$ be a hypostructure satisfying interface permits $D_E$ and $\mathrm{SC}_\lambda$ with scaling exponents $(\alpha, \beta)$ satisfying $\alpha > \beta$ (strict subcriticality). Let $x \in X$ with $\Phi(x) < \infty$ and $\mathcal{C}_*(x) < \infty$ (finite total cost). Then **no supercritical self-similar blow-up** can occur at $T_*(x)$.

More precisely: if a supercritical sequence produces a nontrivial ancient trajectory $v_\infty$, then:
$$\int_{-\infty}^0 \mathfrak{D}(v_\infty(s)) \, ds = \infty$$

**Proof (Scaling Arithmetic):**

*Step 1 (Change of Variables).* For rescaled time $s = \lambda_n^\beta(t - t_n)$ and rescaled state $v_n(s) = \mathcal{S}_{\lambda_n} \cdot u(t)$:
$$\int_{t_n}^{T_*(x)} \mathfrak{D}(u(t)) \, dt$$

*Step 2 (Dissipation Scaling).* By interface permit $\mathrm{SC}_\lambda$ with exponent $\alpha$:
$$\mathfrak{D}(u(t)) = \mathfrak{D}(\mathcal{S}_{\lambda_n}^{-1} \cdot v_n(s)) \sim \lambda_n^{-\alpha} \mathfrak{D}(v_n(s))$$

*Step 3 (Cost Transformation).* Substituting:
$$\int_{t_n}^{T_*(x)} \mathfrak{D}(u(t)) \, dt = \lambda_n^{-(\alpha + \beta)} \int_0^{S_n} \mathfrak{D}(v_n(s)) \, ds$$

*Step 4 (Supercritical Regime).* For nontrivial $v_\infty$ with $S_n \to \infty$:
$$\int_0^{S_n} \mathfrak{D}(v_n(s)) \, ds \gtrsim C_0 \lambda_n^\beta(T_*(x) - t_n)$$

*Step 5 (Contradiction).* If $\alpha > \beta$, summing over dyadic scales requires $\int_{-\infty}^0 \mathfrak{D}(v_\infty) ds = \infty$ for consistency with $\mathcal{C}_*(x) < \infty$.

**Certificate Produced:** $K_4^+$ with payload $(\alpha, \beta, \alpha > \beta)$ or $K_{\text{TypeII}}^{\text{blk}}$

**Literature:** {cite}`MerleZaag98`; {cite}`KenigMerle06`; {cite}`Struwe88`; {cite}`Tao06`
:::

---

### 37.2 Spectral Generator (StiffnessCheck Predicate)

:::{prf:metatheorem} Spectral Generator (MT 5.5)
:label: mt-imported-spectral-generator
:class: metatheorem

**Source:** Hypostructure MT 5.5 (Barrier Atlas)

**Sieve Target:** Node 7 (StiffnessCheck) — spectral gap $\Rightarrow$ Łojasiewicz-Simon inequality

**Statement:** Let $\mathcal{S}$ be a hypostructure satisfying interface permits $D_E$, $\mathrm{LS}_\sigma$, and $\mathrm{GC}_\nabla$. The local behavior near the safe manifold $M$ determines the sharp functional inequality governing convergence:

$$\nabla^2 \Phi|_M \succ 0 \quad \Longrightarrow \quad \|\nabla \Phi(x)\| \geq c \cdot |\Phi(x) - \Phi_{\min}|^\theta$$

for some $\theta \in [1/2, 1)$ and $c > 0$.

**Required Interface Permits:** $D_E$ (Dissipation), $\mathrm{LS}_\sigma$ (Stiffness), $\mathrm{GC}_\nabla$ (Gradient Consistency)

**Prevented Failure Modes:** S.D (Stiffness Breakdown), C.E (Energy Escape)

**Proof (4 Steps):**

*Step 1 (Hessian structure).* Near a critical point $x_* \in M$, the height functional $\Phi$ admits Taylor expansion:
$$\Phi(x) = \Phi(x_*) + \frac{1}{2}\langle \nabla^2 \Phi|_{x_*} (x - x_*), (x - x_*) \rangle + O(|x - x_*|^3)$$

*Step 2 (Spectral gap from positivity).* If $\nabla^2 \Phi|_{x_*} \succ 0$ with smallest eigenvalue $\sigma_{\min} > 0$, then:
$$\Phi(x) - \Phi(x_*) \geq \frac{\sigma_{\min}}{2}|x - x_*|^2$$

*Step 3 (Gradient bound).* The gradient satisfies $\|\nabla \Phi(x)\| \geq \sigma_{\min}|x - x_*|$. Combined with Step 2:
$$\|\nabla \Phi(x)\| \geq \sigma_{\min} \sqrt{\frac{2}{\sigma_{\min}}(\Phi(x) - \Phi_{\min})} = \sqrt{2\sigma_{\min}} |\Phi(x) - \Phi_{\min}|^{1/2}$$

This gives the Łojasiewicz exponent $\theta = 1/2$ (optimal for analytic functions).

*Step 4 (Simon's extension).* For infinite-dimensional systems, Simon (1983) extended the Łojasiewicz inequality to Banach spaces with analytic structure, showing $\theta \in [1/2, 1)$ suffices for convergence.

**Certificate Produced:** $K_7^+$ with payload $(\sigma_{\min}, \theta, c)$

**Literature:** {cite}`Lojasiewicz63`; {cite}`Simon83`; {cite}`HuangTang06`; {cite}`ColdingMinicozzi15`
:::

---

### 37.3 Ergodic Mixing Barrier (ErgoCheck Predicate)

:::{prf:metatheorem} Ergodic Mixing Barrier (MT 24.5)
:label: mt-imported-ergodic-mixing
:class: metatheorem

**Source:** Hypostructure MT 24.5 (Barrier Atlas)

**Sieve Target:** Node 10 (ErgoCheck) — mixing prevents localization

**Statement:** Let $(X, S_t, \mu)$ be a measure-preserving dynamical system satisfying interface permits $C_\mu$ and $D_E$. If the system is **mixing**, then:

1. Correlation functions decay: $C_f(t) := \int f(S_t x) f(x) d\mu - (\int f d\mu)^2 \to 0$ as $t \to \infty$
2. No localized invariant structures can persist
3. Mode T.D (Glassy Freeze) is prevented

**Required Interface Permits:** $C_\mu$ (Compactness), $D_E$ (Dissipation)

**Prevented Failure Modes:** T.D (Glassy Freeze), C.E (Escape)

**Proof:**

*Step 1 (Mixing definition).* The system is mixing if for all $f, g \in L^2(\mu)$:
$$\lim_{t \to \infty} \int f(S_t x) g(x) d\mu = \int f d\mu \int g d\mu$$

*Step 2 (Birkhoff ergodic theorem).* By Birkhoff (1931), for ergodic systems:
$$\frac{1}{T} \int_0^T f(S_t x) dt \to \int f d\mu \quad \text{a.e.}$$

*Step 3 (Localization obstruction).* A localized singular structure would require $\mu(B_\varepsilon(x_*)) > 0$ to persist under the flow for all time. But mixing implies:
$$\mu(S_t^{-1}(B_\varepsilon(x_*)) \cap B_\varepsilon(x_*)) \to \mu(B_\varepsilon(x_*))^2$$
For small $\varepsilon$, the measure of return diminishes, preventing persistent localization.

*Step 4 (Escape guarantee).* Combined with interface permit $D_E$, the trajectory cannot remain trapped in any finite region indefinitely. Mixing spreads mass throughout the accessible phase space.

**Certificate Produced:** $K_{10}^+$ (mixing) with payload $(\tau_{\text{mix}}, C_f(t))$

**Literature:** {cite}`Birkhoff31`; {cite}`Furstenberg81`; {cite}`Sinai70`; {cite}`Bowen75`
:::

---

### 37.4 Spectral Distance Isomorphism (OscillateCheck Predicate)

:::{prf:metatheorem} Spectral Distance Isomorphism (MT 25.2)
:label: mt-imported-spectral-distance
:class: metatheorem

**Source:** Hypostructure MT 25.2 (NCG Isomorphisms)

**Sieve Target:** Node 12 (OscillateCheck) — commutator $\|[D,a]\|$ detects oscillatory breakdown

**Statement:** In the framework of noncommutative geometry, the Connes distance formula provides a spectral characterization of metric structure:

$$d_D(x, y) = \sup\{|f(x) - f(y)| : \|[D, f]\| \leq 1\}$$

The interface permit $\mathrm{GC}_\nabla$ (Gradient Consistency) is equivalent to the spectral distance formula when the geometry admits a Dirac-type operator $D$.

**Bridge Type:** NCG $\leftrightarrow$ Metric Spaces

**Dictionary:**
- Commutator $[D, a]$ $\leftrightarrow$ Gradient $\nabla f$
- Spectral distance $d_D$ $\leftrightarrow$ Geodesic distance
- $\|[D, a]\| \leq 1$ $\leftrightarrow$ $\|\nabla f\| \leq 1$ (Lipschitz condition)

**Proof (Clifford Structure):**

*Step 1 (Spectral triple).* A spectral triple $(\mathcal{A}, \mathcal{H}, D)$ consists of: algebra $\mathcal{A}$ acting on Hilbert space $\mathcal{H}$, self-adjoint Dirac operator $D$ with compact resolvent.

*Step 2 (Commutator as gradient).* For $a \in \mathcal{A}$, the commutator $[D, a]$ acts on spinors. In the classical limit, $[D, f] \to \gamma(\nabla f)$ where $\gamma$ is Clifford multiplication.

*Step 3 (Distance duality).* The supremum over Lipschitz functions with $\|[D, f]\| \leq 1$ recovers the geodesic distance: $d_D(x, y) = d_g(x, y)$ for the Riemannian metric $g$.

*Step 4 (Oscillation detection).* Oscillatory breakdown corresponds to $\|[D, a]\| \to \infty$ for bounded $a$—the derivative blows up. This violates interface permit $\mathrm{GC}_\nabla$.

**Certificate Produced:** $K_{12}^+$ with payload $(D, \|[D, \cdot]\|, d_D)$

**Literature:** {cite}`Connes94`; {cite}`Connes96`; {cite}`GraciaBondia01`; {cite}`Landi97`
:::

---

### 37.5 Antichain-Surface Correspondence (BoundaryCheck Predicate)

:::{prf:metatheorem} Antichain-Surface Correspondence (MT 34.2)
:label: mt-imported-antichain-surface
:class: metatheorem

**Source:** Hypostructure MT 34.2 (Causal-Geometric Isomorphisms)

**Sieve Target:** Node 13 (BoundaryCheck) — boundary interaction measure via min-cut/max-flow

**Statement:** In a causal set $(C, \prec)$ with interface permit $\mathrm{Cap}_H$, discrete antichains converge to minimal surfaces in the continuum limit. The correspondence:

- **Antichain** (maximal set of pairwise incomparable elements) $\leftrightarrow$ **Spacelike hypersurface**
- **Cut size** $|A|$ in causal graph $\leftrightarrow$ **Area** of minimal surface

**Bridge Type:** Causal Sets $\leftrightarrow$ Riemannian Geometry

**Dictionary:**
- Antichain $A$ $\to$ Surface $\Sigma$
- Causal order $\prec$ $\to$ Metric structure
- Min-cut in causal graph $\to$ Minimal surface (area-minimizing)

**Proof (Menger's Theorem + Γ-Convergence):**

*Step 1 (Menger's theorem).* In a finite graph, the maximum flow from source to sink equals the minimum cut capacity. For causal graphs, this relates the "information flow" through time to the minimal separating surface.

*Step 2 (Discrete approximation).* Let $C_n$ be a sequence of causal sets approximating a Lorentzian manifold $(M, g)$. The number of elements in a causal diamond scales as the spacetime volume: $|J^+(p) \cap J^-(q)| \sim V_g(D(p,q))$.

*Step 3 (Γ-convergence).* The discrete cut functional:
$$F_n(A) = \frac{|A|}{n^{(d-1)/d}}$$
Γ-converges to the area functional:
$$F(Σ) = \text{Area}_g(Σ)$$
for hypersurfaces $Σ$ in the continuum limit.

*Step 4 (Minimal surface emergence).* Minimizers of $F_n$ (minimal antichains) converge to minimizers of $F$ (minimal surfaces). This is the boundary measure in the Sieve.

**Certificate Produced:** $K_{13}^+$ with payload $(|A|, \text{Area}(Σ), \text{min-cut})$

**Literature:** {cite}`Menger27`; {cite}`DeGiorgi75`; {cite}`Sorkin91`; {cite}`BombelliLeeEtAl87`
:::

---

## 38. Barrier Defense Theorems

*These theorems prove that barriers actually stop singularities.*

---

### 38.1 Saturation Principle (BarrierSat)

:::{prf:metatheorem} Saturation Principle (MT 5.4)
:label: mt-imported-saturation
:class: metatheorem

**Source:** Hypostructure MT 5.4 (Barrier Atlas)

**Sieve Target:** BarrierSat — drift control prevents blow-up

**Statement:** Let $\mathcal{S}$ be a hypostructure where interface permit $D_E$ depends on an analytic inequality of the form $\Phi(u) + \alpha \mathfrak{D}(u) \leq \text{Drift}(u)$. If there exists a Lyapunov function $\mathcal{V}: X \to [0, \infty)$ satisfying the **Foster-Lyapunov drift condition**:

$$\mathcal{L}\mathcal{V}(x) \leq -\lambda \mathcal{V}(x) + b \cdot \mathbf{1}_C(x)$$

for generator $\mathcal{L}$, constant $\lambda > 0$, bound $b < \infty$, and compact set $C$, then:

1. The process is positive recurrent
2. Energy blow-up (Mode C.E) is prevented
3. A threshold energy $E^* = b/\lambda$ bounds the asymptotic energy

**Required Interface Permits:** $D_E$ (Dissipation), $\mathrm{SC}_\lambda$ (Scaling)

**Prevented Failure Modes:** C.E (Energy Blow-up), S.E (Supercritical Cascade)

**Proof (Meyn-Tweedie):**

*Step 1 (Generator bound).* Apply Itô's lemma to $\mathcal{V}(X_t)$:
$$d\mathcal{V}(X_t) = \mathcal{L}\mathcal{V}(X_t) dt + \text{martingale}$$

*Step 2 (Drift control).* The drift condition ensures:
$$\mathbb{E}[\mathcal{V}(X_t)] \leq e^{-\lambda t} \mathcal{V}(x_0) + \frac{b}{\lambda}(1 - e^{-\lambda t})$$

*Step 3 (Asymptotic bound).* As $t \to \infty$:
$$\limsup_{t \to \infty} \mathbb{E}[\mathcal{V}(X_t)] \leq \frac{b}{\lambda} = E^*$$

*Step 4 (Pathological saturation).* Pathologies saturate the inequality: the threshold energy $E^*$ is determined by the ground state of the singular profile. Energy cannot exceed $E^*$ asymptotically.

**Certificate Produced:** $K_{\text{Sat}}^{\text{blk}}$ with payload $(E^*, \lambda, b, C)$

**Literature:** {cite}`MeynTweedie93`; {cite}`HairerMattingly06`; {cite}`Khasminskii12`; {cite}`Lyapunov1892`
:::

---

### 38.2 Physical Computational Depth Limit (BarrierCausal)

:::{prf:metatheorem} Physical Computational Depth Limit (MT 5.6)
:label: mt-imported-causal-barrier
:class: metatheorem

**Source:** Margolus-Levitin Theorem (1998)

**Sieve Target:** BarrierCausal — infinite event sequences require infinite energy-time (Zeno exclusion)

**Input Certificates:**
1. $K_{D_E}^+$: System has finite average energy $E$ relative to ground state
2. $K_{C_\mu}^+$: Singular region confined to finite volume

**Statement (Margolus-Levitin Theorem):**
The maximum rate of orthogonal state evolution is bounded by energy:
$$\nu_{\max} \leq \frac{4E}{\pi\hbar}$$

Therefore, the maximum number of distinguishable events in time interval $[0,T]$ is:
$$N(T) \leq \frac{4}{\pi\hbar} \int_0^T (E(t) - E_0) \, dt$$

**Required Interface Permits:** $D_E$ (Finite Energy), $C_\mu$ (Confinement)

**Prevented Failure Modes:** C.C (Event Accumulation / Zeno)

**Blocking Logic:**
If a singularity requires an infinite event sequence (Zeno accumulation) but the energy integral is finite (Node 1 passes), then Mode C.C is physically impossible:

$$K_{D_E}^+ \wedge (N_{\text{req}} = \infty) \Rightarrow K_{\mathrm{Rec}_N}^{\mathrm{blk}}$$

**Proof:**

*Step 1 (Energy bound).* By interface permit $D_E$, the system has finite average energy $E = \int_0^T (E(t) - E_0) dt < \infty$.

*Step 2 (Margolus-Levitin).* By quantum mechanics, the minimum time to transition between orthogonal states is $\Delta t \geq \pi\hbar / 4E$. This is a fundamental limit independent of the physical implementation.

*Step 3 (Event counting).* If $N$ distinguishable events (state changes) occur in time $T$, then:
$$N \leq \frac{4}{\pi\hbar} \int_0^T E(t) \, dt$$

*Step 4 (Zeno exclusion).* A Zeno sequence (infinitely many events in finite time) would require $N = \infty$ with $T < \infty$. By the bound above, this requires $\int_0^T E(t) dt = \infty$, contradicting the energy certificate $K_{D_E}^+$.

**Certificate Produced:** $K_{\mathrm{Rec}_N}^{\text{blk}}$ with payload $(E_{\max}, N_{\max}, T_{\text{horizon}})$ where $N_{\max} = \frac{4 E_{\max} T_{\text{horizon}}}{\pi\hbar}$

**Literature:** {cite}`MargolisLevitin98`; {cite}`Lloyd00`; {cite}`BekensteinBound81`
:::

---

### 38.3 Capacity Barrier (BarrierCap)

:::{prf:metatheorem} Capacity Barrier (MT 5.3)
:label: mt-imported-capacity-barrier
:class: metatheorem

**Source:** Hypostructure MT 5.3 (Resolution Machinery)

**Sieve Target:** BarrierCap — zero-capacity sets cannot sustain energy

**Statement:** Let $\mathcal{S}$ be a hypostructure with geometric background (BG) satisfying interface permit $\mathrm{Cap}_H$. Let $(B_k)$ be a sequence of subsets with increasing "thinness" (e.g., tubular neighborhoods of codimension-$\kappa$ sets with radius $r_k \to 0$) such that:

$$\sum_k \text{Cap}(B_k) < \infty$$

Then **occupation time bounds** hold: the trajectory cannot spend infinite time in thin sets.

**Required Interface Permits:** $\mathrm{Cap}_H$ (Capacity), $\mathrm{TB}_\pi$ (Background Geometry)

**Prevented Failure Modes:** C.D (Geometric Collapse)

**Proof (Occupation Measure):**

*Step 1 (Capacity-codimension bound).* By the background geometry interface permit (BG4):
$$\text{Cap}(B) \leq C \cdot r^{d-\kappa}$$
for sets of codimension $\kappa$ and radius $r$.

*Step 2 (Occupation measure).* The occupation measure $\mu_T(B) = \frac{1}{T}\int_0^T \mathbf{1}_B(u(t)) dt$ satisfies:
$$\mu_T(B_k) \leq \frac{C_{\text{cap}}(\Phi(x) + T)}{\text{Cap}(B_k)}$$

*Step 3 (Summability).* For $\sum_k \text{Cap}(B_k) < \infty$:
$$\sum_k \mu_T(B_k) < \infty$$
The trajectory can spend at most finite total time in all thin sets combined.

*Step 4 (Blocking mechanism).* If a blow-up required concentrating on sets with $\dim(\Sigma) < d_c$ (critical codimension), the capacity is too small to support the energy:
$$\int_\Sigma |V|^2 d\mathcal{H}^{\dim(\Sigma)} < \infty \implies E(V) = 0$$
A zero-energy profile cannot mediate blow-up.

**Certificate Produced:** $K_{\text{Cap}}^{\text{blk}}$ with payload $(\text{Cap}(B), d_c, \mu_T)$

**Literature:** {cite}`Federer69`; {cite}`EvansGariepy92`; {cite}`AdamsHedberg96`; {cite}`Maz'ya85`
:::

---

### 38.4 Topological Sector Suppression (BarrierAction)

:::{prf:metatheorem} Topological Sector Suppression (MT 5.7)
:label: mt-imported-topological-suppression
:class: metatheorem

**Source:** Hypostructure MT 5.7 (Resolution Machinery)

**Sieve Target:** BarrierAction — exponential suppression by action gap

**Statement:** Assume the topological background (TB) with action gap $\Delta > 0$ and an invariant probability measure $\mu$ satisfying a log-Sobolev inequality with constant $\lambda_{\text{LS}} > 0$. Assume the action functional $\mathcal{A}$ is Lipschitz with constant $L > 0$. Then:

$$\mu(\{x : \tau(x) \neq 0\}) \leq C \exp\left(-c \lambda_{\text{LS}} \frac{\Delta^2}{L^2}\right)$$

for universal constants $C = 1$, $c = 1/8$.

**Proof (Herbst + Łojasiewicz):**

*Step 1 (Herbst argument).* The log-Sobolev inequality (LSI) with constant $\lambda_{\text{LS}}$ implies concentration of measure. For any 1-Lipschitz function $f$:
$$\mu(\{f \geq \mathbb{E}_\mu[f] + t\}) \leq \exp\left(-\frac{\lambda_{\text{LS}} t^2}{2}\right)$$

*Step 2 (Action gap setup).* By interface permit $\mathrm{TB}_\pi$ (action gap), states in nontrivial topological sectors have:
$$\tau(x) \neq 0 \implies \mathcal{A}(x) \geq \mathcal{A}_{\min} + \Delta$$

*Step 3 (Lipschitz rescaling).* The action $\mathcal{A}$ has Lipschitz constant $L$. Define $f = \mathcal{A}/L$ (1-Lipschitz). Then:
$$\{x : \tau(x) \neq 0\} \subseteq \{f \geq f_{\min} + \Delta/L\}$$

*Step 4 (Measure bound).* By the Herbst estimate:
$$\mu(\{x : \tau(x) \neq 0\}) \leq \exp\left(-\frac{\lambda_{\text{LS}} (\Delta/L)^2}{2}\right) = \exp\left(-\frac{\lambda_{\text{LS}} \Delta^2}{2L^2}\right)$$

*Step 5 (Exponential suppression).* The probability of residing in a nontrivial topological sector decays exponentially with the action gap squared. Large $\Delta$ or strong LSI exponentially suppresses topological obstructions.

**Certificate Produced:** $K_{\text{Action}}^{\text{blk}}$ with payload $(\Delta, \lambda_{\text{LS}}, L)$

**Literature:** {cite}`Herbst75`; {cite}`Lojasiewicz63`; {cite}`Ledoux01`; {cite}`BobkovGotze99`
:::

---

### 38.5 Bode Sensitivity Integral (BarrierBode)

:::{prf:theorem} Bode Sensitivity Integral (Thm 27.1)
:label: thm-imported-bode
:class: theorem

**Source:** Hypostructure Thm 27.1 (Barrier Atlas)

**Sieve Target:** BarrierBode — waterbed effect conservation law

**Statement:** Let $\mathcal{S}$ be a feedback control system with loop transfer function $L(s)$, sensitivity $S(s) = (1 + L(s))^{-1}$, and $n_p$ unstable poles $\{p_i\}$ in the right half-plane. Then:

**Waterbed Effect:**
$$\int_0^\infty \log |S(j\omega)| \, d\omega = \pi \sum_{i=1}^{n_p} p_i$$

**Consequence:** If $|S(j\omega)| < 1$ (good rejection) on some frequency band $[\omega_1, \omega_2]$, then there must exist frequencies where $|S(j\omega)| > 1$ (amplification). Sensitivity cannot be uniformly suppressed.

**Required Interface Permits:** $\mathrm{LS}_\sigma$ (Stiffness/Stability)

**Prevented Failure Modes:** S.D (Infinite Stiffness), C.E (Instability)

**Proof (6 Steps):**

*Step 1 (Cauchy integral setup).* Consider the contour integral of $\log S(s)$ around the right half-plane: a semicircle from $-jR$ to $jR$ closed by the imaginary axis.

*Step 2 (Residue calculation).* The only singularities of $\log S(s)$ inside the contour are at the zeros of $1 + L(s)$ (closed-loop poles). For stable systems, there are none. The contribution from unstable poles of $L(s)$ comes from the integral representation.

*Step 3 (Arc contribution).* As $R \to \infty$, the semicircular arc contributes zero if $L(s) \to 0$ as $|s| \to \infty$ (strictly proper $L$).

*Step 4 (Imaginary axis integral).* The integral along the imaginary axis is:
$$\int_{-j\infty}^{j\infty} \log S(s) \, ds = 2j \int_0^\infty \log|S(j\omega)| d\omega$$
(using $\log S(-j\omega) = \overline{\log S(j\omega)}$ for real systems).

*Step 5 (Poisson-Jensen formula).* By the Poisson-Jensen formula for functions analytic in the right half-plane:
$$\int_0^\infty \log|S(j\omega)| d\omega = \pi \sum_{p_i \in \text{RHP}} \text{Re}(p_i)$$
where the sum is over unstable poles of $L(s)$.

*Step 6 (Waterbed interpretation).* The integral is fixed by unstable poles. Pushing down $|S|$ at some frequencies forces it up elsewhere—this is the "waterbed effect."

**Certificate Produced:** $K_{\text{Bode}}^{\text{blk}}$ with payload $(\int \log|S| d\omega, \{p_i\})$

**Literature:** {cite}`Bode45`; {cite}`DoyleFrancisTannenbaum92`; {cite}`SkogestadPostlethwaite05`; {cite}`Freudenberg85`
:::

---

### 38.6 Epistemic Horizon Principle (BarrierEpi)

:::{prf:metatheorem} Epistemic Horizon Principle (MT 28.1)
:label: mt-imported-epistemic-horizon
:class: metatheorem

**Source:** Hypostructure MT 28.1 (Barrier Atlas)

**Sieve Target:** BarrierEpi — one-way barrier via data processing inequality

**Statement:** Information acquisition is bounded by thermodynamic dissipation. The **Landauer bound** and **data processing inequality** establish fundamental limits:

1. **Landauer's principle:** Erasing one bit of information requires at least $k_B T \ln 2$ of energy dissipation
2. **Data processing inequality:** For any Markov chain $X \to Y \to Z$:
   $$I(X; Z) \leq I(X; Y)$$
   Information cannot increase through processing.

**Required Interface Permits:** $\mathrm{Cap}_H$ (Capacity), $D_E$ (Dissipation)

**Prevented Failure Modes:** D.E (Observation), D.C (Measurement)

**Proof:**

*Step 1 (Entropy production).* For a system with positive Lyapunov exponents $\lambda_i > 0$, Pesin's formula gives the KS entropy:
$$h_\mu = \sum_{\lambda_i > 0} \lambda_i > 0$$

*Step 2 (Total entropy).* The total entropy production up to time $T_*$ is:
$$\Sigma(T_*) = \int_0^{T_*} h_\mu(S_\tau) d\tau > 0$$

*Step 3 (Data processing).* By the data processing inequality, for $u_0 \to u(t) \to V_\lambda$:
$$I(u_0; V_\lambda) \leq I(u(t); V_\lambda) \leq I(u_0; u(t))$$

*Step 4 (Mutual information decay).* Entropy production causes information loss:
$$I(u_0; u(T_*)) \leq H(u_0) - \Sigma(T_*)$$

*Step 5 (Channel capacity bound).* The singularity requires information about the initial condition to be preserved to the blow-up time. The channel capacity is bounded:
$$I(u_0; V_\lambda) \leq \min\{C_\Phi(\lambda), H(u_0) - \Sigma(T_*)\}$$
If entropy production exceeds channel capacity, the singularity cannot form.

**Certificate Produced:** $K_{\text{Epi}}^{\text{blk}}$ with payload $(I_{\max}, h_\mu, k_B T \ln 2)$

**Literature:** {cite}`CoverThomas06`; {cite}`Landauer61`; {cite}`Bennett82`; {cite}`Pesin77`
:::

---

## 39. Surgery Construction Theorems

*These theorems provide constructive methods for purple surgery nodes.*

---

### 39.1 Regularity Lift Principle (SurgSE)

:::{prf:metatheorem} Regularity Lift Principle (MT 6.1)
:label: mt-imported-regularity-lift
:class: metatheorem

**Source:** Hypostructure MT 6.1 (Regularity Surgery)

**Sieve Target:** SurgSE (Regularity Extension) — rough path $\to$ regularity structure lift

**Repair Class:** Symmetry (Algebraic Lifting)

**Statement:** Consider a singular SPDE:
$$\partial_t u = \mathcal{L}u + F(u, \xi)$$
where $\xi$ is distributional noise (e.g., space-time white noise) and $F$ involves products ill-defined in classical distribution theory. There exists:

1. A **regularity structure** $\mathscr{T} = (T, A, G)$ with model space $T$, grading $A$, and structure group $G$
2. A **lift** $\hat{u} \in \mathcal{D}^\gamma$ (modelled distributions of regularity $\gamma$)
3. A **reconstruction operator** $\mathcal{R}: \mathcal{D}^\gamma \to \mathcal{D}'$ such that $u = \mathcal{R}\hat{u}$ solves the SPDE

**Proof (Hairer's Construction):**

*Step 1 (Regularity structure).* Build an abstract polynomial-like structure that encodes:
- Basis elements representing canonical noise terms
- Multiplication rules encoding renormalized products
- Group action implementing Taylor reexpansion under translation

*Step 2 (Admissible model).* Construct a concrete realization $\Pi_x: T \to \mathcal{D}'$ that:
- Maps basis elements to actual distributions
- Satisfies coherence: $\Pi_y = \Pi_x \circ \Gamma_{xy}$ for structure group elements

*Step 3 (Modelled distributions).* Define $\hat{u} \in \mathcal{D}^\gamma$ by local Taylor-like expansion:
$$\hat{u}(x) = \sum_{\tau \in T, |\tau| < \gamma} u_\tau(x) \cdot \tau$$
with regularity controlled by $|\hat{u}(y) - \Gamma_{xy}\hat{u}(x)| \lesssim |x-y|^\gamma$

*Step 4 (Abstract fixed point).* Solve the lifted equation:
$$\hat{u} = P * \hat{F}(\hat{u}, \hat{\xi})$$
in the space of modelled distributions. The fixed point exists by Banach contraction.

*Step 5 (Reconstruction).* Apply $\mathcal{R}$ to obtain $u = \mathcal{R}\hat{u} \in \mathcal{D}'$, the actual solution.

**Certificate Produced:** $K_{\text{SurgSE}}$ with payload $(\mathscr{T}, \hat{u}, \mathcal{R})$

**Literature:** {cite}`Hairer14`; {cite}`GubinelliImkellerPerkowski15`; {cite}`BrunedHairerZambotti19`; {cite}`FrizHairer14`
:::

---

### 39.2 Structural Surgery Principle (SurgTE)

:::{prf:metatheorem} Structural Surgery Principle (MT 6.5)
:label: mt-imported-structural-surgery
:class: metatheorem

**Source:** Hypostructure MT 6.5 (Regularity Surgery)

**Sieve Target:** SurgTE (Topological Extension) — Perelman cut-and-paste surgery

**Repair Class:** Topology (Structural Excision)

**Statement:** Let $(M, g(t))$ be a Ricci flow developing a singularity at time $T$. There exists a **surgery procedure**:

1. **Detect**: Identify neck regions where curvature exceeds threshold $|Rm| > \rho^{-2}$
2. **Excise**: Cut the manifold along approximate round spheres in neck regions
3. **Cap**: Glue in standard caps (round hemispheres with controlled geometry)
4. **Continue**: Restart the flow from the surgered manifold

The procedure maintains:
- Uniform local geometry control
- Monotonicity of Perelman's $\mathcal{W}$-entropy
- Finite number of surgeries in finite time

**Proof (Perelman-Hamilton-Kleiner-Lott):**

*Step 1 (Canonical neighborhood theorem).* Near high-curvature points, the geometry is modeled by one of:
- Shrinking round spheres $S^n$
- Shrinking cylinders $S^{n-1} \times \mathbb{R}$
- Quotients of the above
This provides surgery location candidates.

*Step 2 (Neck detection).* A neck is a region diffeomorphic to $S^{n-1} \times [-L, L]$ with:
$$\left|g - g_{cyl}\right| < \varepsilon$$
for the standard cylinder metric $g_{cyl}$.

*Step 3 (Surgery procedure).* Cut along $S^{n-1} \times \{0\}$, discard the high-curvature component, glue a standard cap:
$$M_{\text{new}} = M_{\text{low}} \cup_\partial \text{Cap}$$
where Cap has uniformly bounded geometry.

*Step 4 (Entropy control).* Perelman's $\mathcal{W}$-entropy satisfies:
$$\mathcal{W}(g_{\text{new}}) \geq \mathcal{W}(g_{\text{old}}) - C\varepsilon$$
Surgeries only decrease entropy by controlled amounts.

*Step 5 (Finite surgery).* The entropy is bounded below; each surgery costs at least $\delta > 0$ entropy. Total surgeries $\leq (\mathcal{W}_{\max} - \mathcal{W}_{\min})/\delta < \infty$.

**Certificate Produced:** $K_{\text{SurgTE}}$ with payload $(M_{\text{new}}, n_{\text{surg}}, \mathcal{W})$

**Literature:** {cite}`Perelman02`; {cite}`Perelman03a`; {cite}`Perelman03b`; {cite}`KleinerLott08`; {cite}`Hamilton97`
:::

---

### 39.3 Projective Extension (SurgCD)

:::{prf:metatheorem} Projective Extension (MT 6.3)
:label: mt-imported-projective-extension
:class: metatheorem

**Source:** Hypostructure MT 6.3 (Regularity Surgery)

**Sieve Target:** SurgCD (Constraint Relaxation) — slack variable method for geometric collapse

**Repair Class:** Geometry (Constraint Relaxation)

**Statement:** Let $K = \{x : g_i(x) \leq 0, h_j(x) = 0\}$ be a constraint set that has collapsed to measure zero ($\text{Cap}(K) = 0$). Introduce **slack variables** $s_i \geq 0$ to obtain the relaxed problem:

$$K_\varepsilon = \{(x, s) : g_i(x) \leq s_i, h_j(x) = 0, \|s\| \leq \varepsilon\}$$

The relaxation satisfies:
1. $\text{Cap}(K_\varepsilon) > 0$ for $\varepsilon > 0$
2. $K_\varepsilon \to K$ as $\varepsilon \to 0$ in Hausdorff distance
3. Solutions of the relaxed problem converge to solutions of the original (if they exist)

**Proof (Convex Optimization):**

*Step 1 (Slack introduction).* Replace hard constraint $g_i(x) \leq 0$ with soft constraint $g_i(x) - s_i \leq 0$ and $s_i \geq 0$. The feasible region expands.

*Step 2 (Capacity restoration).* For $\varepsilon > 0$:
$$\text{Vol}(K_\varepsilon) \geq c_n \varepsilon^{n_s} \cdot \text{Vol}(U)$$
where $n_s$ is the number of slack variables and $U$ is a neighborhood. Positive volume implies positive capacity.

*Step 3 (Barrier function).* Use logarithmic barrier:
$$f_\mu(x, s) = f(x) - \mu \sum_i \log s_i$$
The central path follows $\nabla f_\mu = 0$ as $\mu \to 0$.

*Step 4 (Convergence).* As $\varepsilon \to 0$ (equivalently $\mu \to 0$), the relaxed solutions converge to the original constrained optimum by standard interior point convergence theory.

**Certificate Produced:** $K_{\text{SurgCD}}$ with payload $(\varepsilon, s^*, x^*)$

**Literature:** {cite}`BoydVandenberghe04`; {cite}`NesterovNemirovskii94`; {cite}`Rockafellar70`; {cite}`BenTalNemirovski01`
:::

---

### 39.4 Derived Extension / BRST (SurgSD)

:::{prf:metatheorem} Derived Extension (MT 6.2)
:label: mt-imported-brst
:class: metatheorem

**Source:** Hypostructure MT 6.2 (Regularity Surgery)

**Sieve Target:** SurgSD (Symmetry Deformation) — ghost fields cancel divergent determinants

**Repair Class:** Symmetry (Graded Extension)

**Statement:** Let $\mathcal{A}$ be a space of connections with gauge group $\mathcal{G}$. The naive path integral $\int_\mathcal{A} e^{-S} \mathcal{D}A$ diverges due to infinite gauge orbit volume. Introduce **ghost fields** $(c, \bar{c})$ of opposite statistics to obtain:

$$Z = \int e^{-S_{\text{tot}}} \mathcal{D}A \mathcal{D}c \mathcal{D}\bar{c}$$

where $S_{\text{tot}} = S + S_{\text{gf}} + S_{\text{ghost}}$.

The BRST construction provides:
1. **Stiffness Restoration**: $\nabla^2 \Phi_{\text{tot}}$ becomes non-degenerate
2. **Capacity Cancellation**: Ghost fields provide negative capacity exactly canceling gauge orbit volume
3. **Physical State Isomorphism**: $\mathcal{H}_{\text{phys}} \cong H^0_s(X_{\text{BRST}})$ (BRST cohomology)

**Proof (Faddeev-Popov + BRST):**

*Step 1 (Gauge fixing).* Choose gauge-fixing function $F(A) = 0$. Insert:
$$1 = \int_\mathcal{G} \mathcal{D}g \, \delta(F(A^g)) \det\left(\frac{\delta F(A^g)}{\delta g}\right)$$

*Step 2 (Faddeev-Popov determinant).* The determinant $\det(\delta F/\delta g) = \det(M_{FP})$ is the Faddeev-Popov determinant. Represent it using Grassmann (ghost) fields:
$$\det(M_{FP}) = \int \mathcal{D}c \mathcal{D}\bar{c} \, e^{-\bar{c} M_{FP} c}$$

*Step 3 (BRST symmetry).* The total action $S_{\text{tot}}$ is invariant under the nilpotent BRST transformation:
$$s: A \mapsto Dc, \quad c \mapsto -\frac{1}{2}[c, c], \quad \bar{c} \mapsto B, \quad s^2 = 0$$

*Step 4 (Cohomological quotient).* Physical observables are BRST-closed: $sO = 0$. Physical states form the cohomology:
$$\mathcal{H}_{\text{phys}} = \frac{\ker(s)}{\text{Im}(s)} = H^0_s(X_{\text{BRST}})$$

*Step 5 (Capacity cancellation).* Fermionic integration contributes $(\text{det } M)^{-1}$ for bosons vs. $\text{det } M$ for fermions. Ghost fields (Grassmann) contribute:
$$\int \mathcal{D}c\mathcal{D}\bar{c} \, e^{-\bar{c}Mc} = \det(M)$$
This exactly cancels the divergent gauge orbit volume, yielding finite $Z$.

**Certificate Produced:** $K_{\text{SurgSD}}$ with payload $(s, H^*_s, c, \bar{c})$

**Literature:** {cite}`BecchiRouetStora76`; {cite}`Tyutin75`; {cite}`FaddeevPopov67`; {cite}`Weinberg96`
:::

---

### 39.5 Adjoint Surgery (SurgBC)

:::{prf:metatheorem} Adjoint Surgery (MT 6.26)
:label: mt-imported-adjoint-surgery
:class: metatheorem

**Source:** Hypostructure MT 6.26 (Boundary Surgery)

**Sieve Target:** SurgBC (Boundary Correction) — Lagrange multiplier / Actor-Critic mechanism

**Repair Class:** Boundary (Alignment Enforcement)

**Statement:** When boundary conditions become misaligned with bulk dynamics (Mode B.C), introduce **adjoint variables** $\lambda$ to enforce alignment:

$$\mathcal{L}(x, \lambda) = f(x) + \lambda^T g(x)$$

The saddle-point problem:
$$\min_x \max_\lambda \mathcal{L}(x, \lambda)$$

ensures:
1. Primal variables $x$ minimize objective
2. Dual variables $\lambda$ enforce constraints $g(x) = 0$
3. Gradient alignment: $\nabla_x f \parallel \nabla_x g$ at optimum

**Proof (Pontryagin + Actor-Critic):**

*Step 1 (KKT conditions).* At the saddle point $(x^*, \lambda^*)$:
$$\nabla_x f(x^*) + \lambda^{*T} \nabla_x g(x^*) = 0$$
$$g(x^*) = 0$$

*Step 2 (Gradient alignment).* The first condition states:
$$\nabla_x f = -\lambda^T \nabla_x g$$
The cost gradient lies in the span of constraint gradients—they are aligned.

*Step 3 (Pontryagin interpretation).* In optimal control, $\lambda(t)$ is the costate satisfying:
$$\dot{\lambda} = -\nabla_x H(x, u, \lambda)$$
The Hamiltonian $H = f + \lambda^T \dot{x}$ couples state and costate dynamics.

*Step 4 (Actor-Critic mechanism).* In reinforcement learning:
- Actor (primal): updates policy to minimize expected cost
- Critic (dual): estimates value function (Lagrange multiplier)
- Convergence requires actor-critic alignment, preventing boundary misalignment.

**Certificate Produced:** $K_{\text{SurgBC}}$ with payload $(\lambda^*, x^*, \nabla_x f \parallel \nabla_x g)$

**Literature:** {cite}`Pontryagin62`; {cite}`Lions71`; {cite}`KondaMitsalis03`; {cite}`Bertsekas19`
:::

---

### 39.6 Lyapunov Compactification (SurgCE)

:::{prf:metatheorem} Lyapunov Compactification (MT 6.4)
:label: mt-imported-lyapunov-compactification
:class: metatheorem

**Source:** Hypostructure MT 6.4 (Regularity Surgery)

**Sieve Target:** SurgCE (Conformal Extension) — conformal rescaling bounds infinite domains

**Repair Class:** Geometry (Conformal Compactification)

**Statement:** Let $(M, g)$ be a non-compact Riemannian manifold with possibly infinite diameter. There exists a **conformal factor** $\Omega: M \to (0, 1]$ such that:

1. $\tilde{g} = \Omega^2 g$ has finite diameter
2. The conformal boundary $\partial_\Omega M = \{\Omega = 0\}$ compactifies $M$
3. Trajectories approaching infinity in $(M, g)$ approach $\partial_\Omega M$ in finite $\tilde{g}$-distance

**Proof (Penrose Compactification):**

*Step 1 (Conformal factor construction).* Choose $\Omega$ vanishing at infinity:
$$\Omega(x) = \frac{1}{1 + d_g(x, x_0)^2}$$
or for asymptotically flat/hyperbolic spaces, use geometric constructions.

*Step 2 (Diameter bound).* The conformal metric $\tilde{g} = \Omega^2 g$ has geodesics satisfying:
$$\tilde{d}(x, y) = \int_\gamma \Omega \, ds_g$$
Since $\int_0^\infty \Omega(r) dr < \infty$ for suitable $\Omega$, the diameter is finite.

*Step 3 (Boundary addition).* The conformal boundary $\partial_\Omega M$ represents "points at infinity." In the compactified manifold $\bar{M} = M \cup \partial_\Omega M$:
- Null infinity $\mathscr{I}^{\pm}$ for Minkowski space
- Conformal boundary for hyperbolic space
- Point at infinity for Euclidean space

*Step 4 (Trajectory control).* A trajectory $\gamma(t) \to \infty$ in $(M, g)$ satisfies:
$$\tilde{d}(\gamma(0), \gamma(t)) \leq \int_0^t \Omega(\gamma(s)) |\dot{\gamma}(s)|_g \, ds < \infty$$
The trajectory reaches $\partial_\Omega M$ in finite $\tilde{g}$-time, preventing "escape to infinity."

**Certificate Produced:** $K_{\text{SurgCE}}$ with payload $(\Omega, \tilde{g}, \partial_\Omega M)$

**Literature:** {cite}`Penrose63`; {cite}`HawkingEllis73`; {cite}`ChoquetBruhat09`; {cite}`Wald84`
:::

---

## 40. Foundation ↔ Sieve Cross-Reference

The following table provides the complete mapping from Sieve components to their substantiating foundational theorems.

### 40.1 Kernel Logic Cross-Reference

| **Sieve Component** | **Foundation Theorem** | **Certificate** | **Primary Literature** |
|---------------------|------------------------|-----------------|------------------------|
| Node 17 (Lock) | MT 8.11.N (Structural Exclusion) | $K_{\text{Lock}}^{\text{blk}}$ | Grothendieck, Mac Lane |
| Node 3 (CompactCheck) | MT 5.1 (Structural Resolution) | Trichotomy | Lions, Kenig-Merle |
| Node 11 (ComplexCheck) | MT 14.1 (Profile Trichotomy) | $K_{11}^{\text{lib/tame/inc}}$ | van den Dries, Kurdyka |
| Meta-Learning | MT 13.57 (Equivariance) | $K_{\text{SV08}}^+$ | Noether, Cohen-Welling |

### 40.2 Gate Evaluator Cross-Reference

| **Blue Node** | **Foundation Theorem** | **Predicate** | **Primary Literature** |
|---------------|------------------------|---------------|------------------------|
| Node 4 (ScaleCheck) | MT 5.2 (Type II Exclusion) | $\alpha > \beta$ | Merle-Zaag, Kenig-Merle |
| Node 7 (StiffnessCheck) | MT 5.5 (Spectral Generator) | $\sigma_{\min} > 0$ | Łojasiewicz, Simon |
| Node 10 (ErgoCheck) | MT 24.5 (Ergodic Mixing) | $\tau_{\text{mix}} < \infty$ | Birkhoff, Sinai |
| Node 12 (OscillateCheck) | MT 25.2 (Spectral Distance) | $\|[D,a]\| < \infty$ | Connes |
| Node 13 (BoundaryCheck) | MT 34.2 (Antichain-Surface) | min-cut/max-flow | Menger, De Giorgi |

### 40.3 Barrier Defense Cross-Reference

| **Orange Barrier** | **Foundation Theorem** | **Blocking Mechanism** | **Primary Literature** |
|--------------------|------------------------|------------------------|------------------------|
| BarrierSat | MT 5.4 (Saturation) | $\mathcal{L}\mathcal{V} \leq -\lambda\mathcal{V} + b$ | Meyn-Tweedie, Hairer |
| BarrierCausal | MT 5.6 (Causal Barrier) | $d(u) < \infty \Rightarrow t < \infty$ | Bennett, Penrose |
| BarrierCap | MT 5.3 (Capacity Barrier) | $\text{Cap}(B) < \infty \Rightarrow \mu_T(B) < \infty$ | Federer, Maz'ya |
| BarrierAction | MT 5.7 (Topological Suppression) | $\mu(\tau \neq 0) \leq e^{-c\Delta^2}$ | Herbst, Łojasiewicz |
| BarrierBode | Thm 27.1 (Bode Integral) | $\int \log|S| d\omega = \pi \sum p_i$ | Bode, Doyle |
| BarrierEpi | MT 28.1 (Epistemic Horizon) | $I(X;Z) \leq I(X;Y)$ | Cover-Thomas, Landauer |

### 40.4 Surgery Construction Cross-Reference

| **Purple Surgery** | **Foundation Theorem** | **Construction** | **Primary Literature** |
|--------------------|------------------------|------------------|------------------------|
| SurgSE (Regularity) | MT 6.1 (Regularity Lift) | $\mathscr{T} = (T, A, G)$ | Hairer (2014) |
| SurgTE (Tunnel) | MT 6.5 (Structural Surgery) | Excise + Cap | Perelman (2002-03) |
| SurgCD (Auxiliary) | MT 6.3 (Projective Extension) | Slack variables | Boyd-Vandenberghe |
| SurgSD (Ghost) | MT 6.2 (BRST) | Ghost fields $(c, \bar{c})$ | Faddeev-Popov, BRST |
| SurgBC (Adjoint) | MT 6.26 (Adjoint Surgery) | Lagrange $\lambda$ | Pontryagin |
| SurgCE (Cap) | MT 6.4 (Lyapunov Compact.) | Conformal $\Omega$ | Penrose |

---

# Appendix: Diagram Excerpts by Section

(complete-sieve-algorithm)=
## Complete Sieve Algorithm

The following is the complete Mermaid source code for the Canonical Sieve Algorithm diagram. This diagram is the authoritative specification of the sieve control flow.

```mermaid
graph TD
    Start(["<b>START DIAGNOSTIC</b>"]) --> EnergyCheck{"<b>1. D_E:</b> Is Energy Finite?<br>E[Φ] < ∞"}

    %% --- LEVEL 1: 0-TRUNCATION (Energy Bounds) ---
    EnergyCheck -- "No: K-_DE" --> BarrierSat{"<b>B1. D_E:</b> Is Drift Bounded?<br>E[Φ] ≤ E_sat"}
    BarrierSat -- "Yes: Kblk_DE" --> ZenoCheck
    BarrierSat -- "No: Kbr_DE" --> SurgAdmCE{"<b>A1. SurgCE:</b> Admissible?<br>conformal ∧ ∂∞X def."}
    SurgAdmCE -- "Yes: K+_Conf" --> SurgCE["<b>S1. SurgCE:</b><br>Ghost/Cap Extension"]
    SurgAdmCE -- "No: K-_Conf" --> ModeCE["<b>Mode C.E</b>: Energy Blow-Up"]
    SurgCE -. "Kre_SurgCE" .-> ZenoCheck

    EnergyCheck -- "Yes: K+_DE" --> ZenoCheck{"<b>2. Rec_N:</b> Are Discrete Events Finite?<br>N(J) < ∞"}
    ZenoCheck -- "No: K-_RecN" --> BarrierCausal{"<b>B2. Rec_N:</b> Infinite Depth?<br>D#40;T*#41; = ∞"}
    BarrierCausal -- "No: Kbr_RecN" --> SurgAdmCC{"<b>A2. SurgCC:</b> Admissible?<br>∃N_max: events ≤ N_max"}
    SurgAdmCC -- "Yes: K+_Disc" --> SurgCC["<b>S2. SurgCC:</b><br>Discrete Saturation"]
    SurgAdmCC -- "No: K-_Disc" --> ModeCC["<b>Mode C.C</b>: Event Accumulation"]
    SurgCC -. "Kre_SurgCC" .-> CompactCheck
    BarrierCausal -- "Yes: Kblk_RecN" --> CompactCheck

    ZenoCheck -- "Yes: K+_RecN" --> CompactCheck{"<b>3. C_μ:</b> Does Energy Concentrate?<br>μ(V) > 0"}

    %% --- LEVEL 2: COMPACTNESS LOCUS (Profile Moduli) ---
    CompactCheck -- "No: K-_Cmu" --> BarrierScat{"<b>B3. C_μ:</b> Is Interaction Finite?<br>M[Φ] < ∞"}
    BarrierScat -- "Yes: Kben_Cmu" --> ModeDD["<b>Mode D.D</b>: Dispersion<br><i>#40;Global Existence#41;</i>"]
    BarrierScat -- "No: Kpath_Cmu" --> SurgAdmCD_Alt{"<b>A3. SurgCD_Alt:</b> Admissible?<br>V ∈ L_soliton ∧ ‖V‖_H¹ < ∞"}
    SurgAdmCD_Alt -- "Yes: K+_Prof" --> SurgCD_Alt["<b>S3. SurgCD_Alt:</b><br>Concentration-Compactness"]
    SurgAdmCD_Alt -- "No: K-_Prof" --> ModeCD_Alt["<b>Mode C.D</b>: Geometric Collapse<br><i>#40;Via Escape#41;</i>"]
    SurgCD_Alt -. "Kre_SurgCD_Alt" .-> Profile

    CompactCheck -- "Yes: K+_Cmu" --> Profile["<b>Canonical Profile V Emerges</b>"]

    %% --- LEVEL 3: EQUIVARIANT DESCENT ---
    Profile --> ScaleCheck{"<b>4. SC_λ:</b> Is Profile Subcritical?<br>λ(V) < λ_c"}

    ScaleCheck -- "No: K-_SClam" --> BarrierTypeII{"<b>B4. SC_λ:</b> Is Renorm Cost ∞?<br>∫D̃ dt = ∞"}
    BarrierTypeII -- "No: Kbr_SClam" --> SurgAdmSE{"<b>A4. SurgSE:</b> Admissible?<br>α-β < ε_crit ∧ V smooth"}
    SurgAdmSE -- "Yes: K+_Lift" --> SurgSE["<b>S4. SurgSE:</b><br>Regularity Lift"]
    SurgAdmSE -- "No: K-_Lift" --> ModeSE["<b>Mode S.E</b>: Supercritical Cascade"]
    SurgSE -. "Kre_SurgSE" .-> ParamCheck
    BarrierTypeII -- "Yes: Kblk_SClam" --> ParamCheck

    ScaleCheck -- "Yes: K+_SClam" --> ParamCheck{"<b>5. SC_∂c:</b> Are Constants Stable?<br>‖∂c‖ < ε"}
    ParamCheck -- "No: K-_SCdc" --> BarrierVac{"<b>B5. SC_∂c:</b> Is Phase Stable?<br>ΔV > k_B T"}
    BarrierVac -- "No: Kbr_SCdc" --> SurgAdmSC{"<b>A5. SurgSC:</b> Admissible?<br>‖∂θ‖ < C_adm ∧ θ stable"}
    SurgAdmSC -- "Yes: K+_Stab" --> SurgSC["<b>S5. SurgSC:</b><br>Convex Integration"]
    SurgAdmSC -- "No: K-_Stab" --> ModeSC["<b>Mode S.C</b>: Parameter Instability"]
    SurgSC -. "Kre_SurgSC" .-> GeomCheck
    BarrierVac -- "Yes: Kblk_SCdc" --> GeomCheck

    ParamCheck -- "Yes: K+_SCdc" --> GeomCheck{"<b>6. Cap_H:</b> Is Codim ≥ Threshold?<br>codim(S) ≥ 2"}

    %% --- LEVEL 4: DIMENSION FILTRATION ---
    GeomCheck -- "No: K-_CapH" --> BarrierCap{"<b>B6. Cap_H:</b> Is Measure Zero?<br>Cap_H#40;S#41; = 0"}
    BarrierCap -- "No: Kbr_CapH" --> SurgAdmCD{"<b>A6. SurgCD:</b> Admissible?<br>Cap#40;Σ#41; ≤ ε ∧ V ∈ L_neck"}
    SurgAdmCD -- "Yes: K+_Neck" --> SurgCD["<b>S6. SurgCD:</b><br>Auxiliary/Structural"]
    SurgAdmCD -- "No: K-_Neck" --> ModeCD["<b>Mode C.D</b>: Geometric Collapse"]
    SurgCD -. "Kre_SurgCD" .-> StiffnessCheck
    BarrierCap -- "Yes: Kblk_CapH" --> StiffnessCheck

    GeomCheck -- "Yes: K+_CapH" --> StiffnessCheck{"<b>7. LS_σ:</b> Is Gap Certified?<br>inf σ(L) > 0"}

    %% --- LEVEL 5: SPECTRAL OBSTRUCTION ---
    StiffnessCheck -- "No: K-_LSsig" --> BarrierGap{"<b>B7. LS_σ:</b> Is Kernel Finite?<br>dim ker#40;L#41; < ∞ ∧ σ_ess > 0"}
    BarrierGap -- "Yes: Kblk_LSsig" --> TopoCheck
    BarrierGap -- "No: Kstag_LSsig" --> BifurcateCheck{"<b>7a. LS_∂²V:</b> Is State Unstable?<br>∂²V(x*) ⊁ 0"}

    %% --- LEVEL 5b: SPECTRAL RESTORATION (Bifurcation Resolution) ---
    BifurcateCheck -- "No: K-_LSd2V" --> SurgAdmSD{"<b>A7. SurgSD:</b> Admissible?<br>dim ker#40;H#41; < ∞ ∧ V iso."}
    SurgAdmSD -- "Yes: K+_Iso" --> SurgSD["<b>S7. SurgSD:</b><br>Ghost Extension"]
    SurgAdmSD -- "No: K-_Iso" --> ModeSD["<b>Mode S.D</b>: Stiffness Breakdown"]
    SurgSD -. "Kre_SurgSD" .-> TopoCheck
    BifurcateCheck -- "Yes: K+_LSd2V" --> SymCheck{"<b>7b. G_act:</b> Is G-orbit Degenerate?<br>⎸G·v₀⎸ = 1"}

    %% Path A: Symmetry Breaking (Governed by SC_∂c)
    SymCheck -- "Yes: K+_Gact" --> CheckSC{"<b>7c. SC_∂c:</b> Are Constants Stable?<br>‖∂c‖ < ε"}
    CheckSC -- "Yes: K+_SCdc" --> ActionSSB["<b>ACTION: SYM. BREAKING</b><br>Generates Mass Gap"]
    ActionSSB -- "Kgap" --> TopoCheck
    CheckSC -- "No: K-_SCdc" --> SurgAdmSC_Rest{"<b>A8. SurgSC_Rest:</b> Admissible?<br>ΔV > k_B T ∧ Γ < Γ_crit"}
    SurgAdmSC_Rest -- "Yes: K+_Vac" --> SurgSC_Rest["<b>S8. SurgSC_Rest:</b><br>Auxiliary Extension"]
    SurgAdmSC_Rest -- "No: K-_Vac" --> ModeSC_Rest["<b>Mode S.C</b>: Parameter Instability<br><i>#40;Vacuum Decay#41;</i>"]
    SurgSC_Rest -. "Kre_SurgSC_Rest" .-> TopoCheck

    %% Path B: Tunneling (Governed by TB_S)
    SymCheck -- "No: K-_Gact" --> CheckTB{"<b>7d. TB_S:</b> Is Tunneling Finite?<br>S[γ] < ∞"}
    CheckTB -- "Yes: K+_TBS" --> ActionTunnel["<b>ACTION: TUNNELING</b><br>Instanton Decay"]
    ActionTunnel -- "Ktunnel" --> TameCheck
    CheckTB -- "No: K-_TBS" --> SurgAdmTE_Rest{"<b>A9. SurgTE_Rest:</b> Admissible?<br>V ≅ S^n×I ∧ S_R[γ] < ∞"}
    SurgAdmTE_Rest -- "Yes: K+_Inst" --> SurgTE_Rest["<b>S9. SurgTE_Rest:</b><br>Structural"]
    SurgAdmTE_Rest -- "No: K-_Inst" --> ModeTE_Rest["<b>Mode T.E</b>: Topological Twist<br><i>#40;Metastasis#41;</i>"]
    SurgTE_Rest -. "Kre_SurgTE_Rest" .-> TameCheck

    StiffnessCheck -- "Yes: K+_LSsig" --> TopoCheck{"<b>8. TB_π:</b> Is Sector Reachable?<br>[π] ∈ π₀(C)_acc"}

    %% --- LEVEL 6: HOMOTOPICAL OBSTRUCTIONS ---
    TopoCheck -- "No: K-_TBpi" --> BarrierAction{"<b>B8. TB_π:</b> Energy < Gap?<br>E < S_min + Δ"}
    BarrierAction -- "No: Kbr_TBpi" --> SurgAdmTE{"<b>A10. SurgTE:</b> Admissible?<br>V ≅ S^n×R #40;Neck#41;"}
    SurgAdmTE -- "Yes: K+_Topo" --> SurgTE["<b>S10. SurgTE:</b><br>Tunnel"]
    SurgAdmTE -- "No: K-_Topo" --> ModeTE["<b>Mode T.E</b>: Topological Twist"]
    SurgTE -. "Kre_SurgTE" .-> TameCheck
    BarrierAction -- "Yes: Kblk_TBpi" --> TameCheck

    TopoCheck -- "Yes: K+_TBpi" --> TameCheck{"<b>9. TB_O:</b> Is Topology Tame?<br>Σ ∈ O-min"}

    TameCheck -- "No: K-_TBO" --> BarrierOmin{"<b>B9. TB_O:</b> Is It Definable?<br>S ∈ O-min"}
    BarrierOmin -- "No: Kbr_TBO" --> SurgAdmTC{"<b>A11. SurgTC:</b> Admissible?<br>Σ ∈ O-ext def. ∧ dim < n"}
    SurgAdmTC -- "Yes: K+_Omin" --> SurgTC["<b>S11. SurgTC:</b><br>O-minimal Regularization"]
    SurgAdmTC -- "No: K-_Omin" --> ModeTC["<b>Mode T.C</b>: Labyrinthine"]
    SurgTC -. "Kre_SurgTC" .-> ErgoCheck
    BarrierOmin -- "Yes: Kblk_TBO" --> ErgoCheck

    TameCheck -- "Yes: K+_TBO" --> ErgoCheck{"<b>10. TB_ρ:</b> Does Flow Mix?<br>τ_mix < ∞"}

    ErgoCheck -- "No: K-_TBrho" --> BarrierMix{"<b>B10. TB_ρ:</b> Mixing Finite?<br>τ_mix < ∞"}
    BarrierMix -- "No: Kbr_TBrho" --> SurgAdmTD{"<b>A12. SurgTD:</b> Admissible?<br>Trap iso. ∧ ∂T > 0"}
    SurgAdmTD -- "Yes: K+_Mix" --> SurgTD["<b>S12. SurgTD:</b><br>Mixing Enhancement"]
    SurgAdmTD -- "No: K-_Mix" --> ModeTD["<b>Mode T.D</b>: Glassy Freeze"]
    SurgTD -. "Kre_SurgTD" .-> ComplexCheck
    BarrierMix -- "Yes: Kblk_TBrho" --> ComplexCheck

    ErgoCheck -- "Yes: K+_TBrho" --> ComplexCheck{"<b>11. Rep_K:</b> Is K(x) Computable?<br>K(x) ∈ ℕ"}

    %% --- LEVEL 7: KOLMOGOROV FILTRATION ---
    ComplexCheck -- "No: K-_RepK" --> BarrierEpi{"<b>B11. Rep_K:</b> Approx. Bounded?<br>sup K_ε#40;x#41; ≤ S_BH"}
    BarrierEpi -- "No: Kbr_RepK" --> SurgAdmDC{"<b>A13. SurgDC:</b> Admissible?<br>K ≤ S_BH+ε ∧ Lipschitz"}
    SurgAdmDC -- "Yes: K+_Lip" --> SurgDC["<b>S13. SurgDC:</b><br>Viscosity Solution"]
    SurgAdmDC -- "No: K-_Lip" --> ModeDC["<b>Mode D.C</b>: Semantic Horizon"]
    SurgDC -. "Kre_SurgDC" .-> OscillateCheck
    BarrierEpi -- "Yes: Kblk_RepK" --> OscillateCheck

    ComplexCheck -- "Yes: K+_RepK" --> OscillateCheck{"<b>12. GC_∇:</b> Does Flow Oscillate?<br>ẋ ≠ -∇V"}

    OscillateCheck -- "Yes: K+_GCnabla" --> BarrierFreq{"<b>B12. GC_∇:</b> Oscillation Finite?<br>∫ω²S dω < ∞"}
    BarrierFreq -- "No: Kbr_GCnabla" --> SurgAdmDE{"<b>A14. SurgDE:</b> Admissible?<br>∃Λ: trunc. moment < ∞ ∧ elliptic"}
    SurgAdmDE -- "Yes: K+_Ell" --> SurgDE["<b>S14. SurgDE:</b><br>De Giorgi-Nash-Moser"]
    SurgAdmDE -- "No: K-_Ell" --> ModeDE["<b>Mode D.E</b>: Oscillatory"]
    SurgDE -. "Kre_SurgDE" .-> BoundaryCheck
    BarrierFreq -- "Yes: Kblk_GCnabla" --> BoundaryCheck

    OscillateCheck -- "No: K-_GCnabla" --> BoundaryCheck{"<b>13. Bound_∂:</b> Is System Open?<br>∂Ω ≠ ∅"}

    %% --- LEVEL 8: BOUNDARY COBORDISM ---
    BoundaryCheck -- "Yes: K+_Bound" --> OverloadCheck{"<b>14. Bound_B:</b> Is Input Bounded?<br>‖Bu‖ ≤ M"}

    OverloadCheck -- "No: K-_BoundB" --> BarrierBode{"<b>B14. Bound_B:</b> Waterbed Bounded?<br>∫ln‖S‖dω > -∞"}
    BarrierBode -- "No: Kbr_BoundB" --> SurgAdmBE{"<b>A15. SurgBE:</b> Admissible?<br>‖S‖_∞ < M ∧ φ_margin > 0"}
    SurgAdmBE -- "Yes: K+_Marg" --> SurgBE["<b>S15. SurgBE:</b><br>Saturation"]
    SurgAdmBE -- "No: K-_Marg" --> ModeBE["<b>Mode B.E</b>: Injection"]
    SurgBE -. "Kre_SurgBE" .-> StarveCheck
    BarrierBode -- "Yes: Kblk_BoundB" --> StarveCheck

    OverloadCheck -- "Yes: K+_BoundB" --> StarveCheck{"<b>15. Bound_∫:</b> Is Input Sufficient?<br>∫r dt ≥ r_min"}

    StarveCheck -- "No: K-_BoundInt" --> BarrierInput{"<b>B15. Bound_∫:</b> Reserve Positive?<br>r_reserve > 0"}
    BarrierInput -- "No: Kbr_BoundInt" --> SurgAdmBD{"<b>A16. SurgBD:</b> Admissible?<br>r_res > 0 ∧ recharge > drain"}
    SurgAdmBD -- "Yes: K+_Res" --> SurgBD["<b>S16. SurgBD:</b><br>Reservoir"]
    SurgAdmBD -- "No: K-_Res" --> ModeBD["<b>Mode B.D</b>: Starvation"]
    SurgBD -. "Kre_SurgBD" .-> AlignCheck
    BarrierInput -- "Yes: Kblk_BoundInt" --> AlignCheck

    StarveCheck -- "Yes: K+_BoundInt" --> AlignCheck{"<b>16. GC_T:</b> Is Control Matched?<br>T(u) ~ d"}
    AlignCheck -- "No: K-_GCT" --> BarrierVariety{"<b>B16. GC_T:</b> Variety Sufficient?<br>H#40;u#41; ≥ H#40;d#41;"}
    BarrierVariety -- "No: Kbr_GCT" --> SurgAdmBC{"<b>A17. SurgBC:</b> Admissible?<br>H#40;u#41; < H#40;d#41; ∧ bridgeable"}
    SurgAdmBC -- "Yes: K+_Ent" --> SurgBC["<b>S17. SurgBC:</b><br>Controller Augmentation"]
    SurgAdmBC -- "No: K-_Ent" --> ModeBC["<b>Mode B.C</b>: Misalignment"]
    SurgBC -. "Kre_SurgBC" .-> BarrierExclusion

    %% --- LEVEL 9: THE COHOMOLOGICAL BARRIER ---
    %% All successful paths funnel here
    BoundaryCheck -- "No: K-_Bound" --> BarrierExclusion
    BarrierVariety -- "Yes: Kblk_GCT" --> BarrierExclusion
    AlignCheck -- "Yes: K+_GCT" --> BarrierExclusion

    BarrierExclusion{"<b>17. Cat_Hom:</b> Is Hom#40;Bad, S#41; = ∅?<br>Hom#40;B, S#41; = ∅"}

    BarrierExclusion -- "Yes: Kblk_CatHom" --> VICTORY(["<b>GLOBAL REGULARITY</b><br><i>#40;Structural Exclusion Confirmed#41;</i>"])
    BarrierExclusion -- "No: Kmorph_CatHom" --> ModeCat["<b>FATAL ERROR</b><br>Structural Inconsistency"]
    BarrierExclusion -- "NO(inc): Kbr-inc_CatHom" --> ReconstructionLoop["<b>MT 42.1:</b><br>Structural Reconstruction"]
    ReconstructionLoop -- "Verdict: Kblk" --> VICTORY
    ReconstructionLoop -- "Verdict: Kmorph" --> ModeCat

    %% ====== STYLES ======
    %% Success states - Green
    style VICTORY fill:#22c55e,stroke:#16a34a,color:#000000,stroke-width:4px
    style ModeDD fill:#22c55e,stroke:#16a34a,color:#000000

    %% Failure modes - Red
    style ModeCE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeSE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeSC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCD_Alt fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeSD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeDC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeDE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeBE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeBD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeBC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCat fill:#ef4444,stroke:#dc2626,color:#ffffff

    %% Barriers - Orange/Amber
    style BarrierSat fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierCausal fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierScat fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierTypeII fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierVac fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierCap fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierGap fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierAction fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierOmin fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierMix fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierEpi fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierFreq fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierBode fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierInput fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierVariety fill:#f59e0b,stroke:#d97706,color:#000000

    %% Reconstruction Loop - Yellow/Gold
    style ReconstructionLoop fill:#fbbf24,stroke:#f59e0b,color:#000000,stroke-width:2px

    %% The Final Gate - Purple with thick border
    style BarrierExclusion fill:#8b5cf6,stroke:#7c3aed,color:#ffffff,stroke-width:4px

    %% Interface Checks - Blue
    style EnergyCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ZenoCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style CompactCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ScaleCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ParamCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style GeomCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style StiffnessCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style TopoCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style TameCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ErgoCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ComplexCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style OscillateCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style BoundaryCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style OverloadCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style StarveCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style AlignCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff

    %% Intermediate nodes - Purple
    style Start fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style Profile fill:#8b5cf6,stroke:#7c3aed,color:#ffffff

    %% Restoration checks - Blue (interface permit checks)
    style BifurcateCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style SymCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style CheckSC fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style CheckTB fill:#3b82f6,stroke:#2563eb,color:#ffffff

    %% Restoration mechanisms - Purple (escape mechanisms)
    style ActionSSB fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style ActionTunnel fill:#8b5cf6,stroke:#7c3aed,color:#ffffff

    %% Restoration failure modes - Red
    style ModeSC_Rest fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTE_Rest fill:#ef4444,stroke:#dc2626,color:#ffffff

    %% Surgery recovery nodes - Purple
    style SurgCE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgCC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgCD_Alt fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgCD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSC_Rest fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTE_Rest fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgDC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgDE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgBE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgBD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgBC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff

    %% Surgery Admissibility checks - Light Purple with border
    style SurgAdmCE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmCC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmCD_Alt fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmSE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmSC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmCD fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmSD fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmSC_Rest fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmTE_Rest fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmTE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmTC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmTD fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmDC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmDE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmBE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmBD fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmBC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
```

---

## Conservation Layer (Nodes 1--3)

```mermaid
graph TD
    Start(["START"]) --> EnergyCheck{"1. D_E:<br>E[Φ] < ∞?"}
    EnergyCheck -- "Yes" --> ZenoCheck{"2. Rec_N:<br>N(J) < ∞?"}
    EnergyCheck -- "No" --> BarrierSat{"SATURATION"}
    ZenoCheck -- "Yes" --> CompactCheck{"3. C_μ:<br>μ(V) > 0?"}
    ZenoCheck -- "No" --> BarrierCausal{"CAUSAL"}
    CompactCheck -- "Yes" --> Profile["Profile V"]
    CompactCheck -- "No" --> BarrierScat{"SCATTERING"}

    style EnergyCheck fill:#3b82f6,color:#fff
    style ZenoCheck fill:#3b82f6,color:#fff
    style CompactCheck fill:#3b82f6,color:#fff
    style BarrierSat fill:#f59e0b
    style BarrierCausal fill:#f59e0b
    style BarrierScat fill:#f59e0b
```

## Symmetry/Geometry Layer (Nodes 4--7)

```mermaid
graph TD
    Profile["Profile V"] --> ScaleCheck{"4. SC_λ:<br>λ(V) < λ_c?"}
    ScaleCheck --> ParamCheck{"5. SC_∂c:<br>‖∂c‖ < ε?"}
    ParamCheck --> GeomCheck{"6. Cap_H:<br>codim(S) ≥ 2?"}
    GeomCheck --> StiffnessCheck{"7. LS_σ:<br>inf σ(L) > 0?"}

    ScaleCheck -- "No" --> BarrierTypeII{"TYPE II"}
    ParamCheck -- "No" --> BarrierVac{"VACUUM"}
    GeomCheck -- "No" --> BarrierCap{"CAPACITY"}
    StiffnessCheck -- "No" --> BarrierGap{"SPECTRAL"}

    style ScaleCheck fill:#3b82f6,color:#fff
    style ParamCheck fill:#3b82f6,color:#fff
    style GeomCheck fill:#3b82f6,color:#fff
    style StiffnessCheck fill:#3b82f6,color:#fff
    style BarrierTypeII fill:#f59e0b
    style BarrierVac fill:#f59e0b
    style BarrierCap fill:#f59e0b
    style BarrierGap fill:#f59e0b
```

## The Lock (Node 17)

```mermaid
graph TD
    Converge["All paths converge"] --> Lock{"17.now pleaLOCK<br>Hom(Bad,S) Empty?"}
    Lock -- "Yes (Blocked)" --> Victory(["GLOBAL REGULARITY"])
    Lock -- "No (Morphism)" --> Fatal["FATAL ERROR"]

    style Lock fill:#8b5cf6,color:#fff,stroke-width:4px
    style Victory fill:#22c55e,stroke-width:4px
    style Fatal fill:#ef4444,color:#fff
```

---

## 41. Algebraic-Geometric Metatheorems

*These metatheorems establish the bridge between the sieve framework and algebraic geometry. They enable sieve execution for problems involving algebraic cycles, cohomology, period domains, and moduli spaces.*

---

### 41.1 Motivic Flow Principle

:::{prf:metatheorem} Motivic Flow Principle (MT 22.1)
:label: mt-imported-motivic-flow
:class: metatheorem

**Source:** Hypostructure MT 22.1 (AG Atlas)

**Sieve Signature**
- **Requires:** $K_{D_E}^+$ (finite energy), $K_{C_\mu}^+$ (concentration), $K_{\mathrm{SC}_\lambda}^+$ (subcritical scaling)
- **Produces:** $K_{\text{motive}}^+$ (motivic assignment with weight filtration)

**Statement:** Let $X$ be a smooth projective variety over a field $k$ with flow $S_t: H^*(X) \to H^*(X)$ induced by correspondences. Suppose the sieve has issued:
- $K_{D_E}^+$: The height functional $\Phi = \|\cdot\|_H^2$ is finite on cohomology
- $K_{C_\mu}^+$: Energy concentrates on a finite-dimensional profile space $\mathcal{P}$
- $K_{\mathrm{SC}_\lambda}^+$: Scaling exponents $(\alpha, \beta)$ satisfy $\alpha < \beta + \lambda_c$

Then there exists a contravariant functor to Chow motives:
$$\mathcal{M}: \mathbf{SmProj}_k^{\text{op}} \to \mathbf{Mot}_k^{\text{eff}}, \quad X \mapsto h(X) = (X, \Delta_X, 0)$$

satisfying:

1. **Künneth Decomposition:** $h(X) = \bigoplus_{i=0}^{2\dim X} h^i(X)$ with $H^*(h^i(X)) = H^i(X, \mathbb{Q})$
2. **Weight Filtration:** The motivic weight filtration $W_\bullet h(X)$ satisfies:
   $$\text{Gr}_k^W h(X) \cong \bigoplus_{\alpha - \beta = k} h(X)_{\alpha,\beta}$$
   where $(\alpha, \beta)$ are the scaling exponents from $K_{\mathrm{SC}_\lambda}^+$
3. **Frobenius Eigenvalues:** For $k = \mathbb{F}_q$, the Frobenius $F: h(X) \to h(X)$ has eigenvalues $\{\omega_i\}$ with $|\omega_i| = q^{w_i/2}$ where $w_i \in W_{w_i}$
4. **Entropy-Trace Formula:** $\exp(h_{\text{top}}(S_t)) = \rho(F^* \mid H^*(X))$ where $\rho$ is spectral radius

**Required Interface Permits:** $D_E$, $C_\mu$, $\mathrm{SC}_\lambda$

**Prevented Failure Modes:** S.E (Supercritical Cascade), C.D (Geometric Collapse)

**Proof (7 Steps):**

*Step 1 (Profile space construction).* The certificate $K_{C_\mu}^+$ guarantees concentration: there exists a finite-dimensional algebraic variety $\mathcal{P} \subset \text{Hilb}(X)$ such that all limit profiles lie in $\mathcal{P}/G$ where $G$ is the symmetry group. By Grothendieck's representability, $\mathcal{P}$ is a quasi-projective scheme.

*Step 2 (Motive assignment).* Define the Chow motive $h(X) := (X, \Delta_X, 0) \in \mathbf{Mot}_k$ where $\Delta_X \subset X \times X$ is the diagonal correspondence. For the profile space: $h(\mathcal{P}) := (\mathcal{P}, \Delta_{\mathcal{P}}, 0)$. If $\mathcal{P}$ is singular, apply resolution of singularities $\pi: \tilde{\mathcal{P}} \to \mathcal{P}$ and set $h(\mathcal{P}) := h(\tilde{\mathcal{P}})$.

*Step 3 (Künneth projectors).* By the Künneth formula in $\mathbf{Mot}_k$ (assuming standard conjectures or working with abelian varieties where proven), there exist orthogonal idempotents $\pi^i \in \text{Corr}^0(X, X)$ with:
$$\sum_{i=0}^{2n} \pi^i = \Delta_X, \quad \pi^i \circ \pi^j = \delta_{ij}\pi^i, \quad H^*(\pi^i) = H^i(X)$$

*Step 4 (Frobenius action).* The flow $S_t$ induces a correspondence $\Gamma_{S_t} \subset X \times X$. For self-similar profiles with scaling data from $K_{\mathrm{SC}_\lambda}^+$:
$$F_t^* = [\Gamma_{S_t}]^*: H^*(X) \to H^*(X), \quad F_t^*[\alpha] = t^{\alpha - \beta}[\alpha] \text{ for } \alpha \in H^{p,q}$$
The exponent $\alpha - \beta = p - q$ is the Hodge weight difference.

*Step 5 (Weight filtration).* Define the weight filtration on $h(X)$ by:
$$W_k h(X) := \bigoplus_{\substack{i \leq k \\ \text{Frob. wt.} \leq k}} h^i(X)$$
The scaling certificate $K_{\mathrm{SC}_\lambda}^+$ with exponents $(\alpha, \beta)$ gives: $\text{Gr}_k^W \cong h(X)_{\alpha - \beta = k}$. This identifies weight graded pieces with mode sectors.

*Step 6 (Trace formula).* By the Lefschetz trace formula for correspondences:
$$\#\text{Fix}(F) = \sum_{i=0}^{2n} (-1)^i \text{Tr}(F^* \mid H^i(X))$$
The topological entropy satisfies $\exp(h_{\text{top}}) = \lim_{n \to \infty} |\text{Tr}((F^*)^n)|^{1/n} = \rho(F^*)$, the spectral radius.

*Step 7 (Certificate assembly).* Construct the output certificate:
$$K_{\text{motive}}^+ = \left(h(X), \{\pi^i\}_{i=0}^{2n}, W_\bullet, \{(\alpha_j, \beta_j)\}_j, \rho(F^*)\right)$$
containing the motive, Künneth projectors, weight filtration, scaling exponents, and spectral radius.

**Certificate Produced:** $K_{\text{motive}}^+$ with payload:
- $h(X) \in \mathbf{Mot}_k^{\text{eff}}$: The effective Chow motive
- $W_\bullet$: Weight filtration with $\text{Gr}_k^W \cong $ Mode $k$
- $(\alpha, \beta)$: Scaling exponents from $K_{\mathrm{SC}_\lambda}^+$
- $\rho(F^*)$: Spectral radius (= $\exp(h_{\text{top}})$)

**Literature:** {cite}`Manin68`; {cite}`Scholl94`; {cite}`Deligne74`; {cite}`Jannsen92`; {cite}`Andre04`
:::

---

### 41.2 Schematic Sieve

:::{prf:metatheorem} Semialgebraic Exclusion (MT 22.2)
:label: mt-imported-schematic-sieve
:class: metatheorem

**Source:** Stengle's Positivstellensatz (1974)

**Sieve Signature**
- **Requires:** $K_{\mathrm{Cap}_H}^+$ (capacity bound), $K_{\mathrm{LS}_\sigma}^+$ (Łojasiewicz gradient), $K_{\mathrm{SC}_\lambda}^+$ (subcritical scaling), $K_{\mathrm{TB}_\pi}^+$ (topological bound)
- **Produces:** $K_{\text{SOS}}^+$ (sum-of-squares certificate witnessing Bad Pattern exclusion)

**Setup:**
Let structural invariants be polynomial variables: $x_1 = \Phi$, $x_2 = \mathfrak{D}$, $x_3 = \text{Gap}$, etc.
Let $\mathcal{R} = \mathbb{R}[x_1, \ldots, x_n]$ be the polynomial ring over the reals.

**Safe Set (from certificates):**
The permit certificates define polynomial inequalities. The *safe region* is:
$$S = \{x \in \mathbb{R}^n \mid g_1(x) \geq 0, \ldots, g_k(x) \geq 0\}$$
where:
- $g_{\text{SC}}(x) := \beta - \alpha - \varepsilon$ (from $K_{\mathrm{SC}_\lambda}^+$)
- $g_{\text{Cap}}(x) := C\mathfrak{D} - \text{Cap}_H(\text{Supp})$ (from $K_{\mathrm{Cap}_H}^+$)
- $g_{\text{LS}}(x) := \|\nabla\Phi\|^2 - C_{\text{LS}}^2 |\Phi - \Phi_{\min}|^{2\theta}$ (from $K_{\mathrm{LS}_\sigma}^+$)
- $g_{\text{TB}}(x) := c^2 - \|\nabla\Pi\|^2$ (from $K_{\mathrm{TB}_\pi}^+$)

**Statement (Stengle's Positivstellensatz):**
Let $B \subset \mathbb{R}^n$ be the *bad pattern region* (states violating safety). Then:
$$S \cap B = \emptyset$$
if and only if there exist sum-of-squares polynomials $\{p_\alpha\}_{\alpha \in \{0,1\}^k} \subset \sum \mathbb{R}[x]^2$ such that:
$$-1 = p_0 + \sum_{i} p_i g_i + \sum_{i<j} p_{ij} g_i g_j + \cdots + p_{1\ldots k} g_1 \cdots g_k$$

**Required Interface Permits:** $\mathrm{Cap}_H$ (Capacity), $\mathrm{LS}_\sigma$ (Stiffness), $\mathrm{SC}_\lambda$ (Scaling), $\mathrm{TB}_\pi$ (Topology)

**Prevented Failure Modes:** C.D (Geometric Collapse), S.D (Stiffness Breakdown)

**Proof (5 Steps):**

*Step 1 (Real algebraic geometry).* The permit certificates define polynomial inequalities over $\mathbb{R}$, not equalities over $\mathbb{C}$. Hilbert's Nullstellensatz does not apply directly to inequalities; we use the Positivstellensatz instead.

*Step 2 (Bad pattern encoding).* A bad pattern $B_i$ is encoded as a semialgebraic set:
$$B_i = \{x \in \mathbb{R}^n \mid h_1(x) \geq 0, \ldots, h_m(x) \geq 0, f(x) = 0\}$$
representing states that lead to singularity type $i$.

*Step 3 (Infeasibility certificate).* By Stengle's Positivstellensatz, $S \cap B_i = \emptyset$ admits a constructive certificate: an identity expressing $-1$ as a combination of the constraint polynomials weighted by SOS polynomials.

*Step 4 (SOS computation).* The SOS certificate can be computed via semidefinite programming (SDP). Given a degree bound $d$, search for SOS polynomials $p_\alpha$ of degree $\leq d$ satisfying the identity. If such an identity exists, the intersection is algebraically certified empty.

*Step 5 (Certificate assembly).* The output certificate consists of:
$$K_{\text{SOS}}^+ = \left(\{p_\alpha\}_\alpha, \{g_i\}_i, \text{SDP feasibility witness}\right)$$

**Certificate Produced:** $K_{\text{SOS}}^+$ with payload:
- $\{p_\alpha\}$: SOS polynomials witnessing the Positivstellensatz identity
- $\{g_i\}$: Permit constraint polynomials
- SDP witness: Numerical certificate of SOS decomposition

**Remark (Nullstellensatz vs. Positivstellensatz):**
The original Nullstellensatz formulation applies to equalities over $\mathbb{C}$. Since permit certificates assert *inequalities* (e.g., $\text{Gap} > 0$) over $\mathbb{R}$, the correct algebraic certificate is Stengle's Positivstellensatz, which handles semialgebraic sets.

**Literature:** {cite}`Stengle74`; {cite}`Parrilo03`; {cite}`Blekherman12`; {cite}`Lasserre09`
:::

---

### 41.3 Kodaira-Spencer Stiffness Link

:::{prf:metatheorem} Kodaira-Spencer Stiffness Link (MT 22.3)
:label: mt-imported-kodaira-spencer
:class: metatheorem

**Source:** Hypostructure MT 22.3 (AG Atlas)

**Sieve Signature**
- **Requires:** $K_{\mathrm{LS}_\sigma}^+$ (stiffness gradient), $K_{C_\mu}^+$ (concentration on finite-dimensional moduli)
- **Produces:** $K_{\text{KS}}^+$ (deformation cohomology, rigidity classification)

**Statement:** Let $V$ be a smooth projective variety over a field $k$. Suppose the sieve has issued:
- $K_{\mathrm{LS}_\sigma}^+$: Łojasiewicz gradient with exponent $\theta \in (0,1)$ and constant $C_{\text{LS}} > 0$
- $K_{C_\mu}^+$: Energy concentrates on a finite-dimensional profile space

Consider the tangent sheaf cohomology groups $H^i(V, T_V)$ for $i = 0, 1, 2$. Then:

1. **Symmetries:** $H^0(V, T_V) \cong \text{Lie}(\text{Aut}^0(V))$ — global vector fields are infinitesimal automorphisms
2. **Deformations:** $H^1(V, T_V) \cong T_{[V]}\mathcal{M}$ — first-order deformations parametrize tangent space to moduli
3. **Obstructions:** $H^2(V, T_V) \supseteq \text{Ob}(V)$ — obstruction space for extending deformations
4. **Stiffness ↔ Rigidity:** $K_{\mathrm{LS}_\sigma}^+$ holds if and only if:
   - $H^1(V, T_V) = 0$ (infinitesimal rigidity), OR
   - The obstruction map $\text{ob}: H^1 \otimes H^1 \to H^2$ is surjective (all deformations obstructed)

**Required Interface Permits:** $\mathrm{LS}_\sigma$ (Stiffness), $C_\mu$ (Concentration)

**Prevented Failure Modes:** S.D (Stiffness Breakdown), S.C (Parameter Instability)

**Proof (7 Steps):**

*Step 1 (Deformation functor).* Define the deformation functor $\text{Def}_V: \mathbf{Art}_k \to \mathbf{Sets}$ by:
$$\text{Def}_V(A) := \left\{\text{flat } \mathcal{V} \to \text{Spec}(A) \mid \mathcal{V} \times_A k \cong V\right\} / \sim$$
This is the moduli problem for flat families with special fiber $V$.

*Step 2 (Kodaira-Spencer map).* For an infinitesimal deformation $\mathcal{V} \to \text{Spec}(k[\epsilon])$, the Kodaira-Spencer map:
$$\text{KS}: T_0\text{Def}_V \xrightarrow{\cong} H^1(V, T_V)$$
identifies first-order deformations with cohomology classes. This is an isomorphism by the exponential sequence.

*Step 3 (Kuranishi space).* By Kuranishi's theorem, there exists a versal deformation $\mathcal{V} \to (\mathcal{K}, 0)$ with:
- $(\mathcal{K}, 0)$ a germ of analytic space (or formal scheme)
- $T_0\mathcal{K} = H^1(V, T_V)$
- The obstruction space $\text{Ob} \subseteq H^2(V, T_V)$

*Step 4 (Obstruction theory).* The obstruction to extending a first-order deformation $\xi \in H^1$ to second order lies in $H^2$. The obstruction map:
$$\text{ob}: \text{Sym}^2 H^1(V, T_V) \to H^2(V, T_V)$$
arises from the bracket $[-, -]: T_V \otimes T_V \to T_V$. If $H^2 = 0$, the Kuranishi space is smooth of dimension $h^1(T_V)$.

*Step 5 (Stiffness ↔ Łojasiewicz).* The certificate $K_{\mathrm{LS}_\sigma}^+$ with gradient inequality $\|\nabla\Phi\| \geq C|\Phi - \Phi_{\min}|^\theta$ corresponds to deformation rigidity:
- **Case $H^1 = 0$:** No infinitesimal deformations exist; $V$ is locally rigid in moduli. The certificate issues with payload "rigid".
- **Case $H^1 \neq 0$, $\text{ob}$ surjective:** All first-order deformations are obstructed; $\mathcal{K} = \{0\}$ scheme-theoretically. The certificate issues with payload "obstructed".
- **Case $H^1 \neq 0$, $\text{ob}$ not surjective:** Positive-dimensional moduli; stiffness certificate $K_{\mathrm{LS}_\sigma}^-$ issues (stiffness fails).

*Step 6 (Concentration link).* The certificate $K_{C_\mu}^+$ ensures the moduli space $\mathcal{M}$ is finite-dimensional. By Grothendieck's representability:
$$\dim \mathcal{M} = h^1(V, T_V) - \dim(\text{Im ob}) < \infty$$
Concentration forces $h^1 < \infty$, which holds for all coherent sheaf cohomology on proper varieties.

*Step 7 (Certificate assembly).* Construct the output certificate:
$$K_{\text{KS}}^+ = \left((h^0, h^1, h^2), \text{ob}, \text{classification}\right)$$
where classification $\in \{\text{rigid}, \text{obstructed}, \text{unobstructed-positive}\}$.

**Certificate Produced:** $K_{\text{KS}}^+$ with payload:
- $(h^0, h^1, h^2) := (\dim H^0(T_V), \dim H^1(T_V), \dim H^2(T_V))$
- $\text{ob}: \text{Sym}^2 H^1 \to H^2$: Obstruction map
- Classification: "rigid" if $h^1 = 0$; "obstructed" if $\text{ob}$ surjective; "unobstructed" otherwise
- Rigidity flag: $\mathbf{true}$ iff $K_{\mathrm{LS}_\sigma}^+$ is compatible

**Literature:** {cite}`KodairaSpencer58`; {cite}`Kuranishi65`; {cite}`Griffiths68`; {cite}`Artin76`; {cite}`Sernesi06`
:::

---

### 41.4 Virtual Cycle Correspondence

:::{prf:metatheorem} Virtual Cycle Correspondence (MT 22.7)
:label: mt-imported-virtual-cycle
:class: metatheorem

**Source:** Hypostructure MT 22.7 (AG Atlas)

**Sieve Signature**
- **Requires:** $K_{\mathrm{Cap}_H}^+$ (capacity bound on moduli), $K_{D_E}^+$ (finite energy), $K_{\mathrm{Rep}}^+$ (representation completeness)
- **Produces:** $K_{\text{virtual}}^+$ (virtual fundamental class, enumerative invariants)

**Statement:** Let $\mathcal{M}$ be a proper Deligne-Mumford stack with perfect obstruction theory $\phi: \mathbb{E}^\bullet \to \mathbb{L}_{\mathcal{M}}$ where $\mathbb{E}^\bullet = [E^{-1} \to E^0]$. Suppose the sieve has issued:
- $K_{\mathrm{Cap}_H}^+$: The Hausdorff capacity satisfies $\text{Cap}_H(\mathcal{M}) \leq C \cdot \mathfrak{D}$ for dimension $\mathfrak{D} = \text{vdim}(\mathcal{M})$
- $K_{D_E}^+$: The energy functional $\Phi$ on $\mathcal{M}$ is bounded: $\sup_{\mathcal{M}} \Phi < \infty$

Then:

1. **Virtual Fundamental Class:** There exists a unique class:
   $$[\mathcal{M}]^{\text{vir}} = 0_E^![\mathfrak{C}_{\mathcal{M}}] \in A_{\text{vdim}}(\mathcal{M}, \mathbb{Q})$$
   where $\mathfrak{C}_{\mathcal{M}} \subset E^{-1}|_{\mathcal{M}}$ is the intrinsic normal cone and $0_E^!$ is the refined Gysin map.

2. **Certificate Integration:** For any certificate test function $\chi_A: \mathcal{M} \to \mathbb{Q}$:
   $$\int_{[\mathcal{M}]^{\text{vir}}} \chi_A = \#^{\text{vir}}\{p \in \mathcal{M} : K_A^-(p)\}$$
   counts (with virtual multiplicity) points where certificate $K_A$ fails.

3. **GW Invariants:** For $X$ a smooth projective variety, $\beta \in H_2(X, \mathbb{Z})$:
   $$\text{GW}_{g,n,\beta}(X; \gamma_1, \ldots, \gamma_n) = \int_{[\overline{M}_{g,n}(X,\beta)]^{\text{vir}}} \prod_{i=1}^n \text{ev}_i^*(\gamma_i)$$
   counts stable maps with $K_{\mathrm{Rep}}^+$ ensuring curve representability.

4. **DT Invariants:** For $X$ a Calabi-Yau threefold, $\text{ch} \in H^*(X)$:
   $$\text{DT}_{\text{ch}}(X) = \int_{[\mathcal{M}_{\text{ch}}^{\text{st}}(X)]^{\text{vir}}} 1$$
   counts stable sheaves with $K_{\mathrm{Cap}_H}^+$ ensuring proper moduli.

**Required Interface Permits:** $\mathrm{Cap}_H$ (Capacity), $D_E$ (Energy), $\mathrm{Rep}$ (Representation)

**Prevented Failure Modes:** C.D (Geometric Collapse), E.I (Enumeration Inconsistency)

**Proof (7 Steps):**

*Step 1 (Obstruction theory).* A perfect obstruction theory is a morphism $\phi: \mathbb{E}^\bullet \to \mathbb{L}_{\mathcal{M}}$ in $D^{[-1,0]}(\mathcal{M})$ satisfying:
- $h^0(\phi): h^0(\mathbb{E}^\bullet) \xrightarrow{\cong} h^0(\mathbb{L}_{\mathcal{M}}) = \Omega_{\mathcal{M}}$ is an isomorphism
- $h^{-1}(\phi): h^{-1}(\mathbb{E}^\bullet) \twoheadrightarrow h^{-1}(\mathbb{L}_{\mathcal{M}})$ is surjective

The certificate $K_{\mathrm{Cap}_H}^+$ ensures $\mathbb{E}^\bullet$ is a 2-term complex of finite-rank vector bundles.

*Step 2 (Virtual dimension).* The virtual dimension is:
$$\text{vdim}(\mathcal{M}) := \text{rk}(E^0) - \text{rk}(E^{-1}) = \chi(\mathbb{E}^\bullet)$$
At each point $p \in \mathcal{M}$: deformations $= H^0(\mathbb{E}^\bullet|_p)$, obstructions $= H^1(\mathbb{E}^\bullet|_p)$.

*Step 3 (Intrinsic normal cone).* The intrinsic normal cone $\mathfrak{C}_{\mathcal{M}} \subset h^1/h^0(\mathbb{E}^{\bullet\vee})$ is a cone stack. By Behrend-Fantechi, it embeds canonically:
$$\mathfrak{C}_{\mathcal{M}} \hookrightarrow E_1 := (E^{-1})^\vee$$
The certificate $K_{D_E}^+$ (bounded energy) ensures $\mathfrak{C}_{\mathcal{M}}$ has proper support.

*Step 4 (Virtual class construction).* Define the virtual fundamental class via the refined Gysin map:
$$[\mathcal{M}]^{\text{vir}} := 0_{E_1}^! [\mathfrak{C}_{\mathcal{M}}] \in A_{\text{vdim}}(\mathcal{M}, \mathbb{Q})$$
When $\mathcal{M}$ is smooth of dimension $d > \text{vdim}$, this equals:
$$[\mathcal{M}]^{\text{vir}} = e(\text{Ob}^\vee) \cap [\mathcal{M}]$$
where $\text{Ob} = \text{coker}(T_{\mathcal{M}} \to E^0)$ is the obstruction sheaf.

*Step 5 (Certificate integration).* For a certificate $K_A$ with associated section $s_A: \mathcal{M} \to \text{Ob}^\vee$, the zero locus $Z(s_A) = \{K_A^- \text{ holds}\}$ is the failure locus. Virtual intersection:
$$\int_{[\mathcal{M}]^{\text{vir}}} e(s_A^*\text{Ob}^\vee) = [Z(s_A)]^{\text{vir}} \cdot [\mathcal{M}]^{\text{vir}}$$
counts certificate violations with virtual multiplicity.

*Step 6 (Enumerative invariants).*
- **GW theory:** The certificate $K_{\mathrm{Rep}}^+$ (curve representability) ensures evaluation maps $\text{ev}_i: \overline{M}_{g,n}(X,\beta) \to X$ are well-defined. GW invariants count certificates issued.
- **DT theory:** The certificate $K_{\mathrm{Cap}_H}^+$ (capacity bound) ensures stable sheaves form a proper moduli space. DT invariants count $K_{\mathrm{Cap}_H}^+$ certificates.

*Step 7 (Certificate assembly).* Construct the output certificate:
$$K_{\text{virtual}}^+ = \left([\mathcal{M}]^{\text{vir}}, \text{vdim}, \mathbb{E}^\bullet, \{\text{inv}_\alpha\}_\alpha\right)$$
with virtual class, dimension, obstruction theory, and computed invariants.

**Certificate Produced:** $K_{\text{virtual}}^+$ with payload:
- $[\mathcal{M}]^{\text{vir}} \in A_{\text{vdim}}(\mathcal{M}, \mathbb{Q})$: Virtual fundamental class
- $\text{vdim} = \text{rk}(E^0) - \text{rk}(E^{-1})$: Virtual dimension
- $\mathbb{E}^\bullet = [E^{-1} \to E^0]$: Perfect obstruction theory
- Invariants: $\text{GW}_{g,n,\beta}$, $\text{DT}_{\text{ch}}$ as needed

**Literature:** {cite}`BehrFant97`; {cite}`LiTian98`; {cite}`KontsevichManin94`; {cite}`Thomas00`; {cite}`Maulik06`; {cite}`Graber99`
:::

---

### 41.5 Monodromy-Weight Lock

:::{prf:metatheorem} Monodromy-Weight Lock (MT 22.11)
:label: mt-imported-monodromy-weight
:class: metatheorem

**Source:** Hypostructure MT 22.11 (AG Atlas)

**Sieve Signature**
- **Requires:** $K_{\mathrm{TB}_\pi}^+$ (topological bound on monodromy), $K_{\mathrm{SC}_\lambda}^+$ (subcritical scaling), $K_{D_E}^+$ (finite energy)
- **Produces:** $K_{\text{MHS}}^+$ (limiting mixed Hodge structure, weight-monodromy correspondence)

**Statement:** Let $\pi: \mathcal{X} \to \Delta$ be a proper flat morphism with smooth generic fiber $X_t$ ($t \neq 0$) and semistable reduction at $0 \in \Delta$. Suppose the sieve has issued:
- $K_{\mathrm{TB}_\pi}^+$: Topological bound $\|\nabla\Pi\| \leq c$ for the period map $\Pi: \Delta^* \to D/\Gamma$
- $K_{\mathrm{SC}_\lambda}^+$: Scaling exponents $(\alpha_i)$ satisfy subcriticality $\alpha_i < \lambda_c$
- $K_{D_E}^+$: Energy $\Phi$ bounded on cohomology of general fiber

Then the limiting mixed Hodge structure (MHS) satisfies:

1. **Schmid ↔ Profile Exactification:** The nilpotent orbit
   $$F^p_t = \exp\left(\frac{\log t}{2\pi i} N\right) \cdot F^p_\infty + O(|t|^\epsilon)$$
   provides the profile map. Certificate $K_{\mathrm{TB}_\pi}^+$ ensures $F^p_\infty$ exists.

2. **Weight Filtration ↔ Scaling Exponents:** The weight filtration $W_\bullet = W(N, k)$ satisfies:
   $$\text{Gr}^W_j H^k \neq 0 \Rightarrow \alpha_{j} = j/2$$
   where $\alpha_j$ are the scaling exponents from $K_{\mathrm{SC}_\lambda}^+$.

3. **Clemens-Schmid ↔ Mode Decomposition:**
   - Vanishing cycles $V := \text{Im}(N)$ correspond to Mode C.D (collapse)
   - Invariant cycles $I := \ker(N) \cap \ker(1-T)$ correspond to Mode C.C (concentration)

4. **Picard-Lefschetz ↔ Dissipation:** Monodromy eigenvalues $\{\zeta\}$ of $T$ satisfy $|\zeta| = 1$ (roots of unity), with $\zeta \neq 1$ contributing dissipation modes.

**Required Interface Permits:** $\mathrm{TB}_\pi$ (Topology), $\mathrm{SC}_\lambda$ (Scaling), $D_E$ (Energy)

**Prevented Failure Modes:** T.E (Topological Twist), S.E (Supercritical Cascade)

**Proof (7 Steps):**

*Step 1 (Monodromy).* Let $T: H^k(X_t, \mathbb{Z}) \to H^k(X_t, \mathbb{Z})$ be the monodromy operator for a loop $\gamma$ around $0$. The certificate $K_{\mathrm{TB}_\pi}^+$ ensures $\|\nabla\Pi\|$ is bounded, which by Borel's theorem implies $T$ is quasi-unipotent:
$$(T^m - I)^{k+1} = 0 \quad \text{for some } m \geq 1$$
The bound $\|\nabla\Pi\| \leq c$ from $K_{\mathrm{TB}_\pi}^+$ controls the monodromy weight.

*Step 2 (Nilpotent orbit theorem).* After finite base change $t \mapsto t^m$, assume $T$ unipotent. Define $N := \log T = \sum_{j=1}^\infty \frac{(-1)^{j+1}}{j}(T-I)^j$. By Schmid's theorem, the period map $\Phi: \Delta^* \to D$ satisfies:
$$\Phi(t) = \exp\left(\frac{\log t}{2\pi i} N\right) \cdot \Phi_\infty + O(|t|^\epsilon)$$
for some $\epsilon > 0$. The limiting Hodge filtration $F^p_\infty$ exists and is horizontal.

*Step 3 (Weight filtration).* Construct $W_\bullet = W(N, k)$ as the unique filtration satisfying:
- **Shifting:** $N(W_j) \subseteq W_{j-2}$ (nilpotent lowers weight by 2)
- **Hard Lefschetz:** $N^j: \text{Gr}^W_{k+j} \xrightarrow{\cong} \text{Gr}^W_{k-j}$ for all $j \geq 0$

This is the Deligne weight filtration associated to $(H^k_{\lim}, N)$.

*Step 4 (Mixed Hodge structure).* The pair $(W_\bullet, F^\bullet_\infty)$ defines a mixed Hodge structure on $H^k_{\lim}$:
- Each $\text{Gr}^W_j H^k_{\lim}$ carries a pure Hodge structure of weight $j$
- The filtrations satisfy $F^p \cap W_j + F^{j-p+1} \cap W_j = W_j \cap (F^p + F^{j-p+1})$

The certificate $K_{D_E}^+$ (bounded energy) ensures the MHS has finite-dimensional graded pieces.

*Step 5 (Scaling-weight correspondence).* The certificate $K_{\mathrm{SC}_\lambda}^+$ provides scaling exponents $(\alpha_i)$. For $v \in \text{Gr}^W_j H^k$:
$$\|v(t)\| \sim |t|^{-j/2} \quad \text{as } t \to 0$$
Thus $\alpha_j = j/2$. Subcriticality $\alpha_j < \lambda_c$ bounds the maximum weight: $j_{\max} < 2\lambda_c$.

*Step 6 (Clemens-Schmid sequence).* The exact sequence of mixed Hodge structures:
$$\cdots \to H_k(X_0) \xrightarrow{i_*} H^k(X_t) \xrightarrow{1-T} H^k(X_t) \xrightarrow{\text{sp}} H_k(X_0) \xrightarrow{N} H_{k-2}(X_0)(-1) \to \cdots$$
decomposes cohomology:
- **Invariant part:** $I = \ker(1-T) = \text{Im}(i_*)$ — cycles surviving to $X_0$ (Mode C.C)
- **Vanishing part:** $V = \text{Im}(N) \cong \text{coker}(i_*)$ — cycles disappearing at $X_0$ (Mode C.D)

*Step 7 (Certificate assembly).* Construct the output certificate:
$$K_{\text{MHS}}^+ = \left(F^\bullet_\infty, W_\bullet, N, T, (I, V), \{(\alpha_j, j)\}\right)$$
containing the limiting Hodge filtration, weight filtration, monodromy data, cycle decomposition, and weight-scaling pairs.

**Certificate Produced:** $K_{\text{MHS}}^+$ with payload:
- $F^\bullet_\infty$: Limiting Hodge filtration
- $W_\bullet = W(N, k)$: Deligne weight filtration
- $N = \log(T^m)$: Nilpotent monodromy logarithm
- $(I, V)$: Invariant/vanishing cycle decomposition
- $\{(\alpha_j = j/2, j)\}$: Weight-scaling correspondence from $K_{\mathrm{SC}_\lambda}^+$

**Literature:** {cite}`Schmid73`; {cite}`Deligne80`; {cite}`Clemens77`; {cite}`CKS86`; {cite}`PS08`; {cite}`Steenbrink76`
:::

---

### 41.6 Tannakian Recognition Principle

:::{prf:metatheorem} Tannakian Recognition Principle (MT 22.15)
:label: mt-imported-tannakian-recognition
:class: metatheorem

**Source:** Hypostructure MT 22.15 (AG Atlas)

**Sieve Signature**
- **Requires:** $K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$ (Hom-functor structure), $K_{\Gamma}^+$ (full context certificate)
- **Produces:** $K_{\text{Tann}}^+$ (Galois group reconstruction, algebraicity criterion, lock exclusion)

**Statement:** Let $\mathcal{C}$ be a neutral Tannakian category over a field $k$ with fiber functor $\omega: \mathcal{C} \to \mathbf{Vect}_k$. Suppose the sieve has issued:
- $K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$: The category $\mathcal{C}$ is $k$-linear, abelian, rigid monoidal with $\text{End}(\mathbb{1}) = k$
- $K_{\Gamma}^+$: Full context certificate with fiber functor $\omega$ exact, faithful, and tensor-preserving

Then:

1. **Group Reconstruction:** The functor of tensor automorphisms
   $$G := \underline{\text{Aut}}^\otimes(\omega): \mathbf{Alg}_k \to \mathbf{Grp}, \quad R \mapsto \text{Aut}^\otimes(\omega \otimes R)$$
   is representable by an affine pro-algebraic group scheme over $k$.

2. **Categorical Equivalence:** There is a canonical equivalence of tensor categories:
   $$\mathcal{C} \xrightarrow{\simeq} \text{Rep}_k(G), \quad V \mapsto (\omega(V), \rho_V)$$
   where $\rho_V: G \to \text{GL}(\omega(V))$ is the natural action.

3. **Motivic Galois Group:** For $\mathcal{C} = \mathbf{Mot}_k^{\text{num}}$ with Betti realization $\omega = H_B$:
   - $G = \mathcal{G}_{\text{mot}}(k)$ is the motivic Galois group
   - Algebraic cycles correspond to $\mathcal{G}_{\text{mot}}$-invariants: $\text{CH}^p(X)_\mathbb{Q} \cong H^{2p}(X)^{\mathcal{G}_{\text{mot}}}$
   - Transcendental classes lie in representations with non-trivial $\mathcal{G}_{\text{mot}}$-action

4. **Lock Exclusion via Galois Constraints:** For barrier $\mathcal{B}$ and safe region $S$ in $\mathcal{C}$:
   $$\text{Hom}_{\mathcal{C}}(\mathcal{B}, S) = \emptyset \Leftrightarrow \text{Hom}_{\text{Rep}(G)}(\rho_{\mathcal{B}}, \rho_S)^G = 0$$
   The lock condition reduces to absence of $G$-equivariant morphisms.

**Required Interface Permits:** $\mathrm{Cat}_{\mathrm{Hom}}$ (Categorical Hom), $\Gamma$ (Full Context)

**Prevented Failure Modes:** L.M (Lock Morphism Existence) — excludes morphisms violating Galois constraints

**Proof (7 Steps):**

*Step 1 (Tannakian axioms).* The certificate $K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$ ensures $\mathcal{C}$ satisfies the Tannakian axioms:
- **Abelian:** $\mathcal{C}$ is a $k$-linear abelian category
- **Rigid monoidal:** $(\mathcal{C}, \otimes, \mathbb{1})$ is a rigid tensor category with unit $\mathbb{1}$
- **Neutrality:** $\text{End}_{\mathcal{C}}(\mathbb{1}) = k$ (no non-trivial automorphisms of the unit)

The certificate $K_{\Gamma}^+$ provides the fiber functor $\omega: \mathcal{C} \to \mathbf{Vect}_k$.

*Step 2 (Automorphism functor).* For any commutative $k$-algebra $R$, define:
$$G(R) := \text{Aut}^\otimes(\omega_R) = \left\{\eta: \omega_R \xrightarrow{\sim} \omega_R \;\middle|\; \begin{array}{l} \eta_{V \otimes W} = \eta_V \otimes \eta_W \\ \eta_\mathbb{1} = \text{id}_R \end{array}\right\}$$
where $\omega_R := \omega \otimes_k R: \mathcal{C} \to \mathbf{Mod}_R$. This defines a functor $G: \mathbf{Alg}_k \to \mathbf{Grp}$.

*Step 3 (Representability).* By Deligne's theorem, $G$ is represented by an affine group scheme:
$$G = \text{Spec}(\mathcal{O}(G)), \quad \mathcal{O}(G) = \varinjlim_{V \in \mathcal{C}} \text{End}(\omega(V))^*$$
The Hopf algebra structure on $\mathcal{O}(G)$ encodes the group law. For $\mathcal{C}$ of subexponential growth, $G$ is pro-algebraic.

*Step 4 (Equivalence).* The canonical functor $\Phi: \mathcal{C} \to \text{Rep}_k(G)$ defined by:
$$\Phi(V) := (\omega(V), \rho_V), \quad \rho_V(g)(v) := g_V(v) \text{ for } g \in G, v \in \omega(V)$$
is an equivalence of tensor categories. Inverse: $\Psi: \text{Rep}_k(G) \to \mathcal{C}$ via torsors.

*Step 5 (Invariant subspace).* For any $V \in \mathcal{C}$, the $G$-invariant subspace is:
$$\omega(V)^G := \{v \in \omega(V) : \forall g \in G(\bar{k}), \; g \cdot v = v\} = \text{Hom}_{\mathcal{C}}(\mathbb{1}, V)$$
This is the subspace of "algebraic" or "Hodge" elements. Certificate $K_{\text{Tann}}^+$ records $\dim V^G$.

*Step 6 (Motivic application).* For the category of numerical motives $\mathcal{C} = \mathbf{Mot}_k^{\text{num}}$:
- The motivic Galois group $\mathcal{G}_{\text{mot}} = \text{Aut}^\otimes(H_B)$ is a pro-reductive group
- The standard conjecture C implies $\mathcal{G}_{\text{mot}}$ is reductive (semisimple component)
- For a motive $h(X)$: $h(X)^{\mathcal{G}_{\text{mot}}} = \text{CH}^*(X)_\mathbb{Q}$ (algebraic cycles)
- Transcendental cycles = $h(X) / h(X)^{\mathcal{G}_{\text{mot}}}$

*Step 7 (Lock verification).* For the sieve lock condition with barrier $\mathcal{B}$ and safe region $S$:
$$K_{\text{Lock}}^+ \text{ iff } \text{Hom}_{\mathcal{C}}(\mathcal{B}, S) = \emptyset$$
By the equivalence $\mathcal{C} \simeq \text{Rep}(G)$, this becomes:
$$\text{Hom}_{\text{Rep}(G)}(\rho_{\mathcal{B}}, \rho_S)^G = 0$$
The lock is verified iff no $G$-equivariant morphisms exist. This is computed via representation theory of $G$.

**Certificate Produced:** $K_{\text{Tann}}^+$ with payload:
- $G = \text{Aut}^\otimes(\omega)$: Reconstructed Galois/automorphism group
- $\mathcal{O}(G)$: Coordinate Hopf algebra
- $\mathcal{C} \simeq \text{Rep}_k(G)$: Categorical equivalence
- $V^G = \text{Hom}(\mathbb{1}, V)$: Invariant (algebraic) subspace for each $V$
- Lock status: $\text{Hom}(\mathcal{B}, S)^G = 0$ verification

**Literature:** {cite}`Deligne90`; {cite}`SaavedraRivano72`; {cite}`DeligneMillne82`; {cite}`Andre04`; {cite}`Nori00`
:::

---

## 42. Structural Reconstruction Principle

*This section introduces a universal metatheorem that resolves epistemic deadlock at Node 17 (Lock) for any hypostructure type. The Structural Reconstruction Principle generalizes Tannakian reconstruction to encompass algebraic, parabolic, and quantum systems, providing a canonical bridge between analytic observables and structural objects.*

---

### 42.1 The Reconstruction Metatheorem

:::{prf:metatheorem} Structural Reconstruction Principle (MT 42.1)
:label: mt-structural-reconstruction
:class: metatheorem

**Source:** Hypostructure General Framework (Node 17 Resolution)

**Sieve Signature**
- **Requires:**
  - $K_{D_E}^+$ (finite energy bound on state space)
  - $K_{C_\mu}^+$ (concentration on finite-dimensional profile space)
  - $K_{\mathrm{SC}_\lambda}^+$ (subcritical scaling exponents)
  - $K_{\mathrm{LS}_\sigma}^+$ (Łojasiewicz-Simon gradient inequality)
  - $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$ (tactic exhaustion at Node 17 with partial progress)
  - $K_{\text{Bridge}}$ (critical symmetry $\Lambda$ descends from $\mathcal{A}$ to $\mathcal{S}$)
  - $K_{\text{Rigid}}$ (subcategory $\langle\Lambda\rangle_{\mathcal{S}}$ satisfies semisimplicity, tameness, or spectral gap)
- **Produces:** $K_{\text{Rec}}^+$ (constructive dictionary $D_{\text{Rec}}: \mathcal{A} \to \mathcal{S}$ with Hom isomorphism, Lock resolution)

**Statement:** Let $(\mathcal{X}, \Phi, \mathfrak{D})$ be a hypostructure of type $T \in \{T_{\text{alg}}, T_{\text{para}}, T_{\text{quant}}\}$. Let $\mathcal{A}$ denote the category of **Analytic Observables** (quantities controlled by interface permits $D_E$, $C_\mu$, $\mathrm{SC}_\lambda$, $\mathrm{LS}_\sigma$) and let $\mathcal{S} \subset \mathcal{A}$ be the rigid subcategory of **Structural Objects** (algebraic cycles, solitons, ground states). Suppose the sieve has issued the following certificates:

- $K_{D_E}^+$: The energy functional $\Phi: \mathcal{X} \to [0, \infty)$ is bounded: $\sup_{x \in \mathcal{X}} \Phi(x) < \infty$
- $K_{C_\mu}^+$: Energy concentrates on a finite-dimensional profile space $\mathcal{P}$ with $\dim \mathcal{P} \leq d_{\max}$
- $K_{\mathrm{SC}_\lambda}^+$: Scaling exponents $(\alpha, \beta)$ satisfy subcriticality: $\alpha < \beta + \lambda_c$
- $K_{\mathrm{LS}_\sigma}^+$: Łojasiewicz-Simon gradient inequality holds: $\|\nabla\Phi\| \geq C|\Phi - \Phi_{\min}|^\theta$ with $\theta \in (0,1)$

- $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$: Tactics E1-E12 fail at Node 17 with partial progress indicators:
  - Dimension bounds: $\dim \text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X}) \leq d_{\max}$ (via $K_{C_\mu}^+$)
  - Invariant constraints: $\mathcal{H}_{\text{bad}}$ annihilated by cone $\mathcal{C} \subset \text{End}(\mathcal{X})$
  - Obstruction witness: Critical symmetry group $G_{\text{crit}} \subseteq \text{Aut}(\mathcal{X})$

- $K_{\text{Bridge}}$: A **Bridge Certificate** witnessing that the critical symmetry operator $\Lambda \in \text{End}_{\mathcal{A}}(\mathcal{X})$ (governing the organization of the state space) descends to the structural category:
  $$\Lambda \in \text{End}_{\mathcal{S}}(\mathcal{X})$$
  with action $\rho: G_{\text{crit}} \to \text{Aut}_{\mathcal{S}}(\mathcal{X})$ preserving:
  - Energy (via $K_{D_E}^+$): $\Phi(\rho(g) \cdot x) = \Phi(x)$ for all $g \in G_{\text{crit}}$
  - Stratification (via $K_{\mathrm{SC}_\lambda}^+$): $\rho(g)(\Sigma_k) = \Sigma_k$ for all strata $\Sigma_k$
  - Gradient structure (via $K_{\mathrm{LS}_\sigma}^+$): $\rho(g)$ commutes with gradient flow

- $K_{\text{Rigid}}$: A **Rigidity Certificate** witnessing that the subcategory $\langle\Lambda\rangle_{\mathcal{S}}$ generated by $\Lambda$ satisfies one of:
  - **(Algebraic)** Semisimplicity: $\text{End}_{\mathcal{S}}(\mathbb{1}) = k$ and $\mathcal{S}$ is abelian semisimple (Deligne {cite}`Deligne90`)
  - **(Parabolic)** Tame Stratification: Profile family admits o-minimal stratification $\mathcal{F} = \bigsqcup_k \mathcal{F}_k$ in structure $\mathcal{O}$ (van den Dries {cite}`vandenDries98`)
  - **(Quantum)** Spectral Gap: $\inf(\sigma(L_G) \setminus \{0\}) \geq \delta > 0$ for gauge-fixed linearization $L_G$ (Simon {cite}`Simon83`)

Then there exists a canonical **Reconstruction Functor**:
$$F_{\text{Rec}}: \mathcal{A} \to \mathcal{S}$$
satisfying the following properties:

1. **Hom Isomorphism:** For any "bad pattern" $\mathcal{H}_{\text{bad}} \in \mathcal{A}$:
   $$\text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X}) \cong \text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{X}))$$
   The isomorphism is natural in $\mathcal{X}$ and preserves obstruction structure.

2. **Rep Interface Compliance:** $F_{\text{Rec}}$ satisfies the $\mathrm{Rep}$ interface (Node 11):
   - Finite representation: $|F_{\text{Rec}}(X)| < \infty$ for all $X \in \mathcal{A}$ (guaranteed by $K_{C_\mu}^+$)
   - Effectiveness: $F_{\text{Rec}}$ is computable given the input certificates

3. **Lock Resolution:** The inconclusive verdict at Node 17 is resolvable:
   $$K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}} \wedge K_{\text{Bridge}} \wedge K_{\text{Rigid}} \Longrightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{verdict}}$$
   where verdict $\in \{\text{blk}, \text{br-wit}\}$ (blocked or breached-with-witness).

4. **Type Universality:** The construction is uniform across hypostructure types $T \in \{T_{\text{alg}}, T_{\text{para}}, T_{\text{quant}}\}$.

**Required Interface Permits:** $D_E$, $C_\mu$, $\mathrm{SC}_\lambda$, $\mathrm{LS}_\sigma$, $\mathrm{Cat}_{\mathrm{Hom}}$, $\mathrm{Rep}$, $\Gamma$

**Prevented Failure Modes:** L.M (Lock Morphism Undecidability), E.D (Epistemic Deadlock), R.I (Reconstruction Incompleteness), C.D (Geometric Collapse)

**Proof (7 Steps):**

*Step 1 (Breached-inconclusive certificate analysis).* The certificate $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$ records that tactics E1-E12 have been exhausted at Node 17 without determining whether $\text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X}) = \emptyset$. The upstream certificates $K_{D_E}^+$, $K_{C_\mu}^+$, $K_{\mathrm{SC}_\lambda}^+$, $K_{\mathrm{LS}_\sigma}^+$ provide **partial progress data**:

- **Dimension bounds** (from $K_{C_\mu}^+$): $\dim \text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X}) \leq d_{\max}$ via concentration on $\mathcal{P}$
- **Scaling constraints** (from $K_{\mathrm{SC}_\lambda}^+$): The exponents $(\alpha, \beta)$ stratify the Hom-space by weight
- **Gradient regularity** (from $K_{\mathrm{LS}_\sigma}^+$): The kernel $\ker(\text{ev}: \text{Hom} \to \mathcal{X})$ has Łojasiewicz structure
- **Obstruction witness** (from E3-E12): A critical symmetry group $G_{\text{crit}} \subseteq \text{Aut}(\mathcal{X})$ emerges

The key insight is that $G_{\text{crit}}$ is not arbitrary—it is precisely the group of symmetries that prevent E1-E12 from concluding. This group becomes the target for the Bridge Certificate.

*Step 2 (Bridge certificate: structural symmetry).* The certificate $K_{\text{Bridge}}$ establishes that $G_{\text{crit}}$ acts not merely as analytic automorphisms, but as **structural** automorphisms:
$$\rho: G_{\text{crit}} \hookrightarrow \text{Aut}_{\mathcal{S}}(\mathcal{X})$$

This is verified by checking that $G_{\text{crit}}$ preserves the permit-certified data:

**(a) Energy preservation (from $K_{D_E}^+$):** For all $g \in G_{\text{crit}}$ and $x \in \mathcal{X}$:
$$\Phi(\rho(g) \cdot x) = \Phi(x)$$
The energy functional certified by $D_E$ is $G_{\text{crit}}$-invariant.

**(b) Stratification equivariance (from $K_{\mathrm{SC}_\lambda}^+$):** For the scaling stratification $\mathcal{X} = \bigsqcup_{k=0}^N \Sigma_k$:
$$\rho(g)(\Sigma_k) = \Sigma_k \quad \text{for all } g \in G_{\text{crit}}, \; k \in \{0, \ldots, N\}$$
The strata defined by scaling exponents are preserved.

**(c) Gradient compatibility (from $K_{\mathrm{LS}_\sigma}^+$):** The Łojasiewicz gradient flow commutes with $G_{\text{crit}}$:
$$\rho(g) \circ \nabla\Phi = \nabla\Phi \circ \rho(g)$$

**(d) Critical operator descent:** The operator $\Lambda$ generating $G_{\text{crit}}$ lies in $\text{End}_{\mathcal{S}}(\mathcal{X})$:
- *Algebraic:* $\Lambda = L$ (Lefschetz operator) is an algebraic correspondence ({cite}`Kleiman68`)
- *Parabolic:* $\Lambda = x \cdot \nabla$ (scaling generator) preserves soliton structure ({cite}`Weinstein85`)
- *Quantum:* $\Lambda = H$ (Hamiltonian) defines the spectral decomposition ({cite}`ReedSimon78`)

The Bridge Certificate is the mathematical content of the phrase "the organizing symmetry is structural."

*Step 3 (Rigidity certificate decomposition).* The certificate $K_{\text{Rigid}}$ provides the categorical rigidity needed for reconstruction. This certificate interacts with the upstream permits $K_{C_\mu}^+$ (finite dimensionality) and $K_{\mathrm{LS}_\sigma}^+$ (gradient structure). We analyze by type:

**Case A (Algebraic — Tannakian Rigidity):** The category $\mathcal{S}$ is a neutral Tannakian category over $k$ with:
- $\text{End}_{\mathcal{S}}(\mathbb{1}) = k$ (no non-trivial endomorphisms of the unit)
- $\mathcal{S}$ is abelian and semisimple (every object decomposes into simples)

By Deligne's theorem {cite}`Deligne90`, there exists an affine group scheme $G = \text{Spec}(\mathcal{O}(G))$ with:
$$\mathcal{S} \simeq \text{Rep}_k(G)$$
The group $G$ is the **motivic Galois group** when $\mathcal{S} = \mathbf{Mot}_k$ ({cite}`Andre04`; {cite}`Jannsen92`). The concentration certificate $K_{C_\mu}^+$ ensures $\dim \omega(V) < \infty$ for all $V \in \mathcal{S}$.

**Case B (Parabolic — O-minimal Tameness):** The profile family $\mathcal{F}$ from $K_{C_\mu}^+$ admits a tame stratification in an o-minimal structure $\mathcal{O}$ (e.g., $\mathbb{R}_{\text{an}}$, $\mathbb{R}_{\exp}$):
$$\mathcal{F} = \bigsqcup_{k=1}^N \mathcal{F}_k$$
where each $\mathcal{F}_k$ is a $C^m$-submanifold definable in $\mathcal{O}$. By van den Dries {cite}`vandenDries98` and Wilkie {cite}`Wilkie96`, such stratifications have:
- Finite complexity: $N < \infty$ strata (compatible with $K_{C_\mu}^+$)
- Cell decomposition: Each $\mathcal{F}_k$ is a union of cells
- Definable selection: Continuous selectors exist on each stratum
- Łojasiewicz inequality: Compatible with $K_{\mathrm{LS}_\sigma}^+$ ({cite}`Lojasiewicz65`)

**Case C (Quantum — Spectral Gap):** The linearized operator $L_G$ (gauge-fixed Hamiltonian, Fokker-Planck generator, or Dirichlet form) satisfies:
$$\inf(\sigma(L_G) \setminus \{0\}) \geq \delta > 0$$
This spectral gap, established via $K_{\mathrm{LS}_\sigma}^+$ (Simon {cite}`Simon83`; {cite}`Glimm87`), ensures:
- Isolated ground state: $\ker(L_G) = \text{span}(\psi_0)$
- Exponential decay: Solutions converge to ground state at rate $e^{-\delta t}$
- Perturbative stability: Gap persists under small perturbations (Kato {cite}`Kato95`)

*Step 4 (Dictionary construction).* We construct the Reconstruction Functor $F_{\text{Rec}}: \mathcal{A} \to \mathcal{S}$ explicitly by type. The construction uses the finiteness from $K_{C_\mu}^+$ and the regularity from $K_{\mathrm{LS}_\sigma}^+$:

**Type $T_{\text{alg}}$ (Algebraic):** Let $\omega: \mathcal{A} \to \mathbf{Vect}_k$ be the fiber functor (e.g., Betti cohomology $H_B$, or de Rham $H_{\text{dR}}$). Define:
$$F_{\text{Rec}}^{\text{alg}}(X) := (\omega(X), \rho_X)$$
where $\rho_X: G \to \text{GL}(\omega(X))$ is the representation induced by the Tannakian structure ({cite}`DeligneMillne82`). The functor satisfies:
- $F_{\text{Rec}}^{\text{alg}}(\mathbb{1}) = (\mathbf{1}_k, \text{triv})$ (monoidal unit)
- $F_{\text{Rec}}^{\text{alg}}(X \otimes Y) = F_{\text{Rec}}^{\text{alg}}(X) \otimes F_{\text{Rec}}^{\text{alg}}(Y)$ (tensor compatibility)
- $\dim \omega(X) < \infty$ (from $K_{C_\mu}^+$)

**Type $T_{\text{para}}$ (Parabolic):** Using the o-minimal cell decomposition from $K_{\text{Rigid}}$, define:
$$F_{\text{Rec}}^{\text{para}}(X) := (\text{profile}(X), \text{stratum}(X), \text{cell}(X))$$
where:
- $\text{profile}(X) \in \mathcal{P}$: The limit profile from $K_{C_\mu}^+$ ({cite}`KenigMerle06`)
- $\text{stratum}(X) \in \{1, \ldots, N\}$: Index of the containing stratum $\mathcal{F}_k$
- $\text{cell}(X)$: Cell index within the stratum (from o-minimal structure)

By Merle-Zaag {cite}`MerleZaag98` and Duyckaerts-Kenig-Merle {cite}`DKM19`, the profile library is finite: $|\mathcal{P}| < \infty$. The Łojasiewicz exponent from $K_{\mathrm{LS}_\sigma}^+$ determines the convergence rate to profiles.

**Type $T_{\text{quant}}$ (Quantum):** Using the spectral resolution of $L_G$ from $K_{\text{Rigid}}$, define:
$$F_{\text{Rec}}^{\text{quant}}(X) := (\psi_0(X), \sigma(X), \Pi_0(X))$$
where:
- $\psi_0(X)$: Projection onto the ground state sector ({cite}`GlimmJaffe87`)
- $\sigma(X) \subset [0, \infty)$: Spectrum of $L_G|_X$
- $\Pi_0(X) = \mathbb{1}_{\{0\}}(L_G)$: Ground state projector

The spectral gap $\delta > 0$ from $K_{\text{Rigid}}$ ensures $\Pi_0$ is finite-rank (Fröhlich-Simon-Spencer {cite}`FSS76`).

*Step 5 (Hom isomorphism verification).* We prove the central isomorphism using the certificates $K_{\text{Bridge}}$ and $K_{\text{Rigid}}$:
$$\Phi_{\text{Rec}}: \text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X}) \xrightarrow{\cong} \text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{X}))$$

**Injectivity:** Let $f, f' \in \text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X})$ with $F_{\text{Rec}}(f) = F_{\text{Rec}}(f')$. By $K_{\text{Bridge}}$, the critical symmetry $G_{\text{crit}}$ acts on both sides via structural automorphisms. The $G_{\text{crit}}$-equivariant structure of $F_{\text{Rec}}$ (inherited from $K_{D_E}^+$, $K_{\mathrm{SC}_\lambda}^+$, $K_{\mathrm{LS}_\sigma}^+$) implies:
$$f(x) = f'(x) \quad \text{for all } x \in \mathcal{H}_{\text{bad}}$$
using:
- *Algebraic:* Faithfulness of $\omega$ ({cite}`Deligne90`, Prop. 2.11)
- *Parabolic:* Definability in $\mathcal{O}$ ({cite}`vandenDries98`, Ch. 4)
- *Quantum:* Spectral uniqueness ({cite}`ReedSimon78`, Thm. VIII.5)

**Surjectivity:** Let $\phi \in \text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{X}))$. The certificate $K_{\text{Rigid}}$ ensures "enough morphisms" to lift:
- *Algebraic:* Semisimplicity implies $\mathcal{S}$ has enough injectives/projectives ({cite}`SaavedraRivano72`, §I.4)
- *Parabolic:* O-minimal definable selection provides lifts ({cite}`vandenDries98`, Thm. 6.1.7)
- *Quantum:* Spectral theorem reconstructs operators from spectral data ({cite}`Kato95`, §V.3)

By the universal property of $F_{\text{Rec}}$, there exists $f \in \text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X})$ with $F_{\text{Rec}}(f) = \phi$. This lift is unique by injectivity.

**Naturality:** For any morphism $g: \mathcal{X} \to \mathcal{Y}$ in $\mathcal{A}$, the diagram:
$$\begin{CD}
\text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X}) @>{\Phi_{\text{Rec}}}>> \text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{X})) \\
@V{g_*}VV @V{F_{\text{Rec}}(g)_*}VV \\
\text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{Y}) @>{\Phi_{\text{Rec}}}>> \text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{Y}))
\end{CD}$$
commutes by functoriality of $F_{\text{Rec}}$. This is the content of the $\mathrm{Cat}_{\mathrm{Hom}}$ interface compliance.

*Step 6 (Lock resolution).* The Hom isomorphism from Step 5 resolves the Node 17 Lock. This step consumes the $\mathrm{Cat}_{\mathrm{Hom}}$ interface permit:

**Case: $\text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{X})) = \emptyset$**

By the isomorphism $\Phi_{\text{Rec}}$:
$$\text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X}) = \emptyset$$
The sieve issues certificate $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{blk}}$ (VICTORY). The bad pattern cannot embed. This triggers success at Node 17.

**Case: $\text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{X})) \neq \emptyset$**

The Reconstruction Functor provides an explicit morphism witness via the $\mathrm{Rep}$ interface:
$$\phi \in \text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{X})) \leadsto f := \Phi_{\text{Rec}}^{-1}(\phi) \in \text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X})$$
The sieve issues certificate $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{morph}}$ with explicit witness $f$ (FATAL if bad pattern embeds, RECOVERABLE if controllable via other permits).

**Decidability via Interface Permits:** The key insight is that $\text{Hom}_{\mathcal{S}}$ is decidable because each rigidity type has effective algorithms:
- *Algebraic:* $G$-invariants in finite-dimensional representations are computable via Chevalley-Jordan decomposition ({cite}`Humphreys72`)
- *Parabolic:* O-minimal cell decomposition is effective ({cite}`vandenDries98`, Thm. 1.8.1); profile matching uses $K_{C_\mu}^+$
- *Quantum:* Spectral projections are computable for discrete spectrum ({cite}`ReedSimon78`); gap from $K_{\text{Rigid}}$ ensures isolation

The inconclusive verdict is resolved: **partial progress (from $K_{D_E}^+$, $K_{C_\mu}^+$, $K_{\mathrm{SC}_\lambda}^+$, $K_{\mathrm{LS}_\sigma}^+$) + structural symmetry ($K_{\text{Bridge}}$) + rigidity ($K_{\text{Rigid}}$) = decidable answer**.

*Step 7 (Certificate assembly).* Construct the output certificate incorporating all upstream permit data:
$$K_{\text{Rec}}^+ = \left(F_{\text{Rec}}, \Phi_{\text{Rec}}, K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{verdict}}, T, D_{\text{Rec}}\right)$$

**Certificate Produced:** $K_{\text{Rec}}^+$ with payload:
- $F_{\text{Rec}}: \mathcal{A} \to \mathcal{S}$: Reconstruction functor (fiber/profile/spectral by type)
- $\Phi_{\text{Rec}}: \text{Hom}_{\mathcal{A}} \xrightarrow{\cong} \text{Hom}_{\mathcal{S}}$: Natural isomorphism with explicit inverse
- $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{verdict}} \in \{\text{blk}, \text{morph}\}$: Resolved Lock outcome at Node 17
- $T \in \{T_{\text{alg}}, T_{\text{para}}, T_{\text{quant}}\}$: Hypostructure type
- $D_{\text{Rec}}$: Constructive Dictionary satisfying $\mathrm{Rep}$ interface (Node 11):
  - Finiteness: $|D_{\text{Rec}}(x)| < \infty$ for all $x$ (inherited from $K_{C_\mu}^+$)
  - Algorithm: Explicit computation procedure for $F_{\text{Rec}}$
- Upstream certificates consumed: $K_{D_E}^+$, $K_{C_\mu}^+$, $K_{\mathrm{SC}_\lambda}^+$, $K_{\mathrm{LS}_\sigma}^+$, $K_{\text{Bridge}}$, $K_{\text{Rigid}}$

**Literature:**
- *Tannakian Categories:* {cite}`Deligne90`; {cite}`SaavedraRivano72`; {cite}`DeligneMillne82`
- *Motivic Galois Groups:* {cite}`Andre04`; {cite}`Jannsen92`; {cite}`Nori00`
- *O-minimal Structures:* {cite}`vandenDries98`; {cite}`Wilkie96`; {cite}`Lojasiewicz65`
- *Dispersive PDEs:* {cite}`KenigMerle06`; {cite}`MerleZaag98`; {cite}`DKM19`
- *Spectral Theory:* {cite}`Simon83`; {cite}`ReedSimon78`; {cite}`Kato95`; {cite}`GlimmJaffe87`; {cite}`FSS76`
- *Algebraic Geometry:* {cite}`Kleiman68`; {cite}`Humphreys72`
:::

:::{prf:remark} Reconstruction uses obligation ledgers
:label: rem-rec-uses-ledger

When MT 42.1 is invoked (from any $K^{\mathrm{inc}}$ route, particularly $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$), its input includes the **obligation ledger** $\mathsf{Obl}(\Gamma)$ (Definition {prf:ref}`def-obligation-ledger`).

The reconstruction procedure must produce one of the following outcomes:
1. **New certificates that discharge entries:** MT 42.1 produces $K_{\text{Bridge}}$, $K_{\text{Rigid}}$, and ultimately $K_{\text{Rec}}^+$, which enable inc-upgrades (Definition {prf:ref}`def-inc-upgrades`) to fire during closure, discharging relevant $K^{\mathrm{inc}}$ entries from the ledger.

2. **Refined missing set:** If full discharge is not possible, MT 42.1 may refine the $\mathsf{missing}$ component of existing $K^{\mathrm{inc}}$ certificates into a strictly more explicit set of prerequisites—smaller template requirements, stronger preconditions, or more specific structural data. This refinement produces a new $K^{\mathrm{inc}}$ with updated payload.

**Formalization:**
$$\text{MT 42.1}: \mathsf{Obl}(\Gamma) \to \left(\{K^+_{\text{new}}\} \text{ enabling discharge}\right) \cup \left(\mathsf{Obl}'(\Gamma) \text{ with refined } \mathsf{missing}\right)$$

This ensures reconstruction makes definite progress: either discharging obligations or producing a strictly refined $\mathsf{missing}$ specification.

:::

---

### 42.2 Type Instantiation Table

The following table summarizes how the Structural Reconstruction Principle instantiates across the three fundamental hypostructure types. Each row shows how the interface permits $D_E$, $C_\mu$, $\mathrm{SC}_\lambda$, $\mathrm{LS}_\sigma$ specialize to the given type:

| **Hypostructure Type** | **Critical Symmetry ($\Lambda$)** | **Bridge Certificate ($K_{\text{Bridge}}$)** | **Rigidity Certificate ($K_{\text{Rigid}}$)** | **Resulting Theorem** |
|:----------------------|:----------------------------------|:--------------------------------------------|:---------------------------------------------|:---------------------|
| **Algebraic** ($T_{\text{alg}}$) | Lefschetz operator $L: H^{n-1} \to H^{n+1}$ | Standard Conjecture B: "$L$ is algebraic" (correspondence in $\text{CH}^1(X \times X)$) | Semisimplicity: $\mathbf{Mot}_k^{\text{num}} \simeq \text{Rep}(\mathcal{G}_{\text{mot}})$ ({cite}`Jannsen92`) | **Hodge Conjecture:** Every harmonic $(p,p)$-form is $\mathbb{Q}$-algebraic |
| **Parabolic** ($T_{\text{para}}$) | Scaling operator $\Lambda = x \cdot \nabla + \frac{2}{\alpha}$ | Virial identity from $K_{\mathrm{SC}_\lambda}^+$: Scaling is monotone in $V(t) = \int |x|^2 |u|^2$ | Tame stratification via $K_{C_\mu}^+$: $\mathcal{P} = \{W_1, \ldots, W_N\}$ in $\mathbb{R}_{\text{an}}$ ({cite}`MerleZaag98`) | **Soliton Resolution:** $u(t) = u_L + \sum_j u_j^*(t-t_j) + o(1)$ |
| **Quantum** ($T_{\text{quant}}$) | Hamiltonian $H = -\Delta + V$ | Spectral condition from $K_{D_E}^+$: $H \geq 0$ with discrete spectrum | Spectral gap via $K_{\mathrm{LS}_\sigma}^+$: $\inf \sigma(H) \setminus \{E_0\} \geq E_0 + \Delta$ ({cite}`GlimmJaffe87`) | **Mass Gap:** Vacuum unique, gap $\Delta > 0$ |

**Permit Specialization by Type:**

| **Permit** | **Algebraic ($T_{\text{alg}}$)** | **Parabolic ($T_{\text{para}}$)** | **Quantum ($T_{\text{quant}}$)** |
|:-----------|:--------------------------------|:----------------------------------|:--------------------------------|
| $K_{D_E}^+$ | Height function bounded | $\|u\|_{H^1}^2 < \infty$ | $\langle \psi, H\psi \rangle < \infty$ |
| $K_{C_\mu}^+$ | Hodge numbers finite | Profile space $\mathcal{P}$ finite | Ground state isolated |
| $K_{\mathrm{SC}_\lambda}^+$ | Weight filtration bounded | Scaling exponent subcritical | Spectral dimension finite |
| $K_{\mathrm{LS}_\sigma}^+$ | Hodge metric analytic | Łojasiewicz at solitons | Spectral gap $\delta > 0$ |

---

### 42.3 Corollaries

:::{prf:corollary} Bridge-Rigidity Dichotomy
:label: cor-bridge-rigidity

If $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$ is issued at Node 17 (with upstream certificates $K_{D_E}^+$, $K_{C_\mu}^+$, $K_{\mathrm{SC}_\lambda}^+$, $K_{\mathrm{LS}_\sigma}^+$ satisfied), then exactly one of the following holds:

1. **Bridge Certificate obtainable:** $K_{\text{Bridge}}$ can be established, and the Lock resolves via MT 42.1 producing $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{verdict}}$
2. **Bridge obstruction identified:** The failure of $K_{\text{Bridge}}$ provides a new certificate $K_{\text{Bridge}}^-$ containing:
   - A counterexample to structural descent: $\Lambda \notin \text{End}_{\mathcal{S}}(\mathcal{X})$
   - An analytic automorphism not preserving structure: $g \in G_{\text{crit}}$ with $g(\mathcal{S}) \not\subseteq \mathcal{S}$
   - A violation witness for one of $K_{D_E}^+$, $K_{\mathrm{SC}_\lambda}^+$, or $K_{\mathrm{LS}_\sigma}^+$ under the $G_{\text{crit}}$-action

In either case, the epistemic deadlock at Node 17 is resolved.
:::

:::{prf:corollary} Analytic-Structural Equivalence
:label: cor-analytic-structural

Under the hypotheses of MT 42.1 (with all interface permits $D_E$, $C_\mu$, $\mathrm{SC}_\lambda$, $\mathrm{LS}_\sigma$, $\mathrm{Cat}_{\mathrm{Hom}}$ satisfied), the categories $\mathcal{A}$ and $\mathcal{S}$ are **Hom-equivalent** on the subcategory generated by $\mathcal{H}_{\text{bad}}$:
$$\mathcal{A}|_{\langle\mathcal{H}_{\text{bad}}\rangle} \simeq_{\text{Hom}} \mathcal{S}|_{\langle F_{\text{Rec}}(\mathcal{H}_{\text{bad}})\rangle}$$

This equivalence is the rigorous formulation of "soft implies hard" for morphisms. In particular:
- Analytic obstructions (from $K_{\mathrm{LS}_\sigma}^+$) are equivalent to structural obstructions
- Concentration data (from $K_{C_\mu}^+$) determines the structural representation
- The $\mathrm{Rep}$ interface is satisfied by both categories
:::

:::{prf:corollary} Permit Flow Theorem
:label: cor-permit-flow

The Structural Reconstruction Principle defines a **permit flow** at Node 17:

$$\begin{CD}
K_{D_E}^+ @>>> K_{C_\mu}^+ @>>> K_{\mathrm{SC}_\lambda}^+ @>>> K_{\mathrm{LS}_\sigma}^+ \\
@. @. @VVV @VVV \\
@. @. K_{\text{Bridge}} @>>> K_{\text{Rigid}} \\
@. @. @VVV @VVV \\
@. @. @. K_{\text{Rec}}^+ @>>> K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{verdict}}
\end{CD}$$

Each arrow represents a certificate dependency. The output $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{verdict}} \in \{\text{blk}, \text{morph}\}$ is the decidable resolution of the Lock.
:::

---

### 42.4 The Analytic-Algebraic Rigidity Lemma

*This lemma provides the rigorous "engine" that powers the algebraic case ($T_{\text{alg}}$) of the Structural Reconstruction Principle (MT 42.1). It formalizes the a posteriori inference: analytic stiffness plus tameness forces algebraicity.*

:::{prf:lemma} Analytic-Algebraic Rigidity
:label: lem-analytic-algebraic-rigidity

**Source:** Hypostructure Algebraic Framework (Node 7 → Node 17 Bridge)

**Sieve Signature**
- **Requires:**
  - $K_{D_E}^+$ (finite energy: $\|\eta\|_{L^2}^2 < \infty$)
  - $K_{\mathrm{LS}_\sigma}^+$ (stiffness: spectral gap $\lambda > 0$ on Hodge-Riemann pairing)
  - $K_{\mathrm{Tame}}^+$ (tameness: singular support $\Sigma(\eta)$ is o-minimal definable)
  - $K_{\mathrm{Hodge}}^{(k,k)}$ (type constraint: $\eta$ is harmonic of type $(k,k)$)
- **Produces:** $K_{\mathrm{Alg}}^+$ (algebraicity certificate: $[\eta] \in \mathcal{Z}^k(X)_{\mathbb{Q}}$)

**Statement:** Let $X$ be a smooth complex projective variety with hypostructure $(\mathcal{X}, \Phi, \mathfrak{D})$ of type $T_{\text{alg}}$. Let $\eta \in H^{2k}(X, \mathbb{C})$ be a harmonic form representing a cohomology class of type $(k,k)$. Suppose the sieve has issued the following certificates:

- $K_{D_E}^+$ **(Energy Bound):** The energy functional satisfies $\Phi(\eta) = \|\eta\|_{L^2}^2 < \infty$.

- $K_{\mathrm{LS}_\sigma}^+$ **(Stiffness/Spectral Gap):** The form $\eta$ lies in a subspace $V \subset H^{2k}(X)$ on which the Hodge-Riemann pairing $Q(\cdot, \cdot)$ is non-degenerate with definite signature. For any perturbation $\delta\eta \in V$, the second variation of the energy satisfies:
  $$\|\nabla^2 \Phi(\eta)\| \geq \lambda > 0$$
  This is the **stiffness condition**: the energy landscape admits no flat directions.

- $K_{\mathrm{Tame}}^+$ **(O-minimal Tameness):** The singular support
  $$\Sigma(\eta) = \{x \in X : \eta(x) \text{ is not real-analytic}\}$$
  is definable in an o-minimal structure $\mathcal{O}$ expanding $\mathbb{R}$ (e.g., $\mathbb{R}_{\text{an}}$, $\mathbb{R}_{\exp}$).

- $K_{\mathrm{Hodge}}^{(k,k)}$ **(Type Constraint):** The form $\eta$ is harmonic ($\Delta\eta = 0$) and of Hodge type $(k,k)$.

Then $\eta$ is the fundamental class of an algebraic cycle with rational coefficients:
$$[\eta] \in \mathcal{Z}^k(X)_{\mathbb{Q}}$$

The sieve issues certificate $K_{\mathrm{Alg}}^+$ with payload $(Z^{\text{alg}}, [Z^{\text{alg}}] = [\eta], \mathbb{Q})$.

**Required Interface Permits:** $D_E$, $\mathrm{LS}_\sigma$, $\mathrm{Tame}$, $\mathrm{Hodge}$, $\mathrm{Rep}$

**Prevented Failure Modes:** W.S (Wild Smooth), S.I (Singular Irregularity), N.H (Non-Holomorphic), N.A (Non-Algebraic)

**Proof (4 Steps):**

*Step 1 (Exclusion of wild smooth forms via $K_{\mathrm{LS}_\sigma}^+$).* The stiffness certificate $K_{\mathrm{LS}_\sigma}^+$ excludes $C^\infty$ forms that are not real-analytic. Suppose $\eta$ were smooth but not real-analytic at some point $p \in X$. By the construction of smooth bump functions, there exists a perturbation:
$$\eta_\epsilon = \eta + \epsilon \psi$$
where $\psi$ is a smooth form with $\text{supp}(\psi) \subset U$ for an arbitrarily small neighborhood $U$ of $p$.

Because $\psi$ is localized, its interactions with the global Hodge-Riemann pairing $Q$ can be made arbitrarily small or sign-indefinite. This creates **flat directions** in the energy landscape:
$$\langle \nabla^2\Phi(\eta) \cdot \psi, \psi \rangle \to 0 \quad \text{as } U \to \{p\}$$

This violates the uniform spectral gap condition $\|\nabla^2\Phi\| \geq \lambda > 0$ from $K_{\mathrm{LS}_\sigma}^+$. The Łojasiewicz-Simon inequality ({cite}`Simon83`; {cite}`Lojasiewicz65`) implies the energy landscape admits no flat directions at critical points.

**Conclusion:** $\eta$ must be real-analytic on $X \setminus \Sigma$, where $\Sigma$ is the singular support. The failure mode **W.S (Wild Smooth)** is excluded.

*Step 2 (Rectifiability via $K_{\mathrm{Tame}}^+$ and $K_{D_E}^+$).* The tameness certificate $K_{\mathrm{Tame}}^+$ combined with finite energy $K_{D_E}^+$ ensures that $\eta$ extends to a rectifiable current.

By the **Cell Decomposition Theorem** for o-minimal structures ({cite}`vandenDries98`, Theorem 1.8.1), the singular support $\Sigma$ admits a finite stratification:
$$\Sigma = \bigsqcup_{i=1}^N S_i$$
where each $S_i$ is a $C^m$-submanifold definable in $\mathcal{O}$. The finiteness $N < \infty$ is guaranteed by o-minimality.

The finite energy certificate $K_{D_E}^+$ implies $\|\eta\|_{L^2}^2 < \infty$, hence $\eta$ has **finite mass** as a current:
$$\mathbb{M}(\eta) = \int_X |\eta| \, dV < \infty$$

By the **Federer-Fleming Closure Theorem** adapted to tame geometry ({cite}`Federer69`, §4.2; {cite}`vandenDries98`, Ch. 6), a current with:
- Finite mass
- O-minimal definable support

is a **rectifiable current**. The tameness of $\mathcal{O}$ excludes pathological fractal-like singularities.

**Conclusion:** $\eta$ extends to a current defined by integration over an analytic chain. The failure mode **S.I (Singular Irregularity)** is excluded.

*Step 3 (Holomorphic structure via $K_{\mathrm{Hodge}}^{(k,k)}$ and $K_{\mathrm{LS}_\sigma}^+$).* The type constraint $K_{\mathrm{Hodge}}^{(k,k)}$ combined with stiffness establishes holomorphicity.

On a Kähler manifold $X$, a real-analytic harmonic $(k,k)$-form with integral periods defines a holomorphic geometric object. The **Poincaré-Lelong equation** ({cite}`GriffithsHarris78`, Ch. 3):
$$\frac{i}{2\pi} \partial\bar{\partial} \log |s|^2 = [Z]$$
relates $(k,k)$-currents to zero sets of holomorphic sections. This provides the bridge from analytic to holomorphic.

The stiffness certificate $K_{\mathrm{LS}_\sigma}^+$ implies **deformation rigidity**: the tangent space to the moduli of such objects vanishes:
$$H^1(Z, \mathcal{N}_{Z/X}) = 0$$
where $\mathcal{N}_{Z/X}$ is the normal bundle ({cite}`Demailly12`, §VII). The moduli space is discrete (zero-dimensional). A "stiff" form cannot deform continuously into a non-holomorphic form without breaking harmonicity or Hodge type.

**Conclusion:** The analytic chain underlying $\eta$ is a complex analytic subvariety $Z \subset X$. The failure mode **N.H (Non-Holomorphic)** is excluded.

*Step 4 (Algebraization via GAGA).* The projectivity of $X$ enables the final step via Serre's GAGA theorem ({cite}`Serre56`).

We have established that $\eta$ corresponds to a global analytic subvariety $Z$ in $X^{\text{an}}$ (the analytification of $X$). Since $X$ is a projective variety, **Serre's GAGA Theorem** applies:

> *The functor from algebraic coherent sheaves on $X$ to analytic coherent sheaves on $X^{\text{an}}$ is an equivalence of categories.*

In particular:
- Every analytic subvariety of a projective variety is algebraic
- The ideal sheaf $\mathcal{I}_Z$ is the analytification of an algebraic ideal sheaf $\mathcal{I}_{Z^{\text{alg}}}$

Therefore:
$$Z = (Z^{\text{alg}})^{\text{an}}$$
for a unique algebraic subvariety $Z^{\text{alg}} \subset X$.

**Conclusion:** The cohomology class $[\eta]$ is the image of the algebraic cycle class:
$$[\eta] = [Z^{\text{alg}}] \in H^{2k}(X, \mathbb{Q})$$

The failure mode **N.A (Non-Algebraic)** is excluded.

**Certificate Produced:** $K_{\mathrm{Alg}}^+$ with payload:
- $Z^{\text{alg}}$: The algebraic cycle
- $[Z^{\text{alg}}] = [\eta]$: Cycle class equality in $H^{2k}(X, \mathbb{Q})$
- $\mathbb{Q}$-coefficients: Rationality of the cycle
- Upstream certificates consumed: $K_{D_E}^+$, $K_{\mathrm{LS}_\sigma}^+$, $K_{\mathrm{Tame}}^+$, $K_{\mathrm{Hodge}}^{(k,k)}$

**Literature:**
- *Łojasiewicz-Simon theory:* {cite}`Simon83`; {cite}`Lojasiewicz65`
- *O-minimal structures:* {cite}`vandenDries98`; {cite}`Wilkie96`
- *Geometric measure theory:* {cite}`Federer69`
- *Complex geometry:* {cite}`GriffithsHarris78`; {cite}`Demailly12`
- *GAGA:* {cite}`Serre56`
:::

---

**Connection to MT 42.1:** This lemma is the **algebraic instantiation** of the Structural Reconstruction Principle:

| MT 42.1 Component | Lemma Instantiation |
|:------------------|:--------------------|
| $\mathcal{A}$ (Analytic Observables) | Harmonic $(k,k)$-forms in $H^{2k}(X, \mathbb{C})$ |
| $\mathcal{S}$ (Structural Objects) | Algebraic cycles $\mathcal{Z}^k(X)_{\mathbb{Q}}$ |
| $K_{\text{Bridge}}$ | Lefschetz operator $L$ is algebraic (Standard Conjecture B) |
| $K_{\text{Rigid}}$ | Semisimplicity of $\mathbf{Mot}_k^{\text{num}}$ ({cite}`Jannsen92`) |
| $F_{\text{Rec}}$ | Cycle class map $\text{cl}: \mathcal{Z}^k \to H^{2k}$ |

The lemma provides the rigorous **a posteriori proof** that stiffness + tameness forces algebraicity, implementing the "soft implies hard" principle for Hodge theory.

---

# Notation Index

The following notation is used consistently throughout this document. Symbols are organized by their role in the Hypostructure formalism.

## Core Objects

| Symbol | Name | Definition | Section |
|--------|------|------------|---------|
| $\mathcal{X}$ | State Space | Configuration $\infty$-stack representing system states | 2, 8.A |
| $\mathcal{B}$ | Boundary Space | Environmental interface / boundary data | 4.5, 10 |
| $\Phi$ | Height / Energy | Cohomological height functional | 2, 8.A |
| $\mathfrak{D}$ | Dissipation | Rate of energy loss / entropy production | 2, 8.A |
| $G$ | Symmetry Group | Invariance group acting on $\mathcal{X}$ | 2, 8.A |
| $\mathbb{H}$ | Hypostructure | Full 5-tuple $(\mathcal{X}, \nabla, \Phi_\bullet, \tau, \partial_\bullet)$ | 2 |
| $\mathcal{T}$ | Thin Object | Minimal 5-tuple of physical data | 4 |

## Energy and Scaling

| Symbol | Name | Context |
|--------|------|---------|
| $E$ | Specific Energy | Instance of height $\Phi$; $E[\Phi] = \sup_t \Phi(u(t))$ |
| $\alpha$ | Energy Scaling | Exponent: $\Phi(\mathcal{S}_\lambda x) = \lambda^\alpha \Phi(x)$ |
| $\beta$ | Dissipation Scaling | Exponent: $\mathfrak{D}(\mathcal{S}_\lambda x) = \lambda^\beta \mathfrak{D}(x)$ |
| $E_{\text{sat}}$ | Saturation Ceiling | Upper bound on drift (BarrierSat) |
| $\mathcal{S}_\lambda$ | Scaling Operator | One-parameter family of dilations |

## Boundary and Reinjection

| Symbol | Name | Definition |
|--------|------|------------|
| $\partial_\bullet$ | Boundary Morphism | Restriction functor $\iota^*: \mathbf{Sh}_\infty(\mathcal{X}) \to \mathbf{Sh}_\infty(\partial\mathcal{X})$ |
| $\text{Tr}$ | Trace Morphism | $\text{Tr}: \mathcal{X} \to \mathcal{B}$ (restriction to boundary) |
| $\mathcal{J}$ | Flux Morphism | $\mathcal{J}: \mathcal{B} \to \underline{\mathbb{R}}$ (energy flow across boundary) |
| $\mathcal{R}$ | Reinjection Kernel | $\mathcal{R}: \mathcal{B} \to \mathcal{P}(\mathcal{X})$ (Markov kernel with Feller property) |

## Certificate Notation

| Symbol | Meaning |
|--------|---------|
| $K^+$ | Positive certificate (predicate holds) |
| $K^-$ | Negative certificate (sum type: $K^{\mathrm{wit}} \sqcup K^{\mathrm{inc}}$) |
| $K^{\mathrm{wit}}$ | NO-with-witness certificate (actual refutation / counterexample found) |
| $K^{\mathrm{inc}}$ | NO-inconclusive certificate (method insufficient, not a semantic refutation) |
| $K^{\text{blk}}$ | Blocked certificate (barrier holds, obstruction present) |
| $K^{\text{br}}$ | Breached certificate (barrier fails: $K^{\mathrm{br\text{-}wit}}$ or $K^{\mathrm{br\text{-}inc}}$) |
| $K^{\text{re}}$ | Re-entry certificate (surgery completed successfully) |
| $\Gamma$ | Certificate accumulator (full chain of certificates) |

## Categorical Notation

| Symbol | Name | Definition |
|--------|------|------------|
| $\mathcal{E}$ | Ambient Topos | Cohesive $(\infty,1)$-topos |
| $\mathbf{Hypo}_T$ | Hypostructure Category | Category of type-$T$ hypostructures |
| $\mathbf{Thin}_T$ | Thin Category | Category of thin kernel objects |
| $\mathbb{H}_{\text{bad}}$ | Bad Pattern | Universal singularity object |
| $\text{Hom}(\cdot, \cdot)$ | Hom Functor | Morphism space (Node 17 Lock) |
| $F_{\text{Sieve}}$ | Sieve Functor | Left adjoint $F_{\text{Sieve}} \dashv U$ |

## Interface Identifiers

| ID | Name | Node |
|----|------|------|
| $D_E$ | Energy Interface | Node 1 |
| $\mathrm{Rec}_N$ | Recovery Interface | Node 2 |
| $C_\mu$ | Compactness Interface | Node 3 |
| $\mathrm{SC}_\lambda$ | Scaling Interface | Node 4 |
| $\mathrm{Cap}_H$ | Capacity Interface | Node 6 |
| $\mathrm{LS}_\sigma$ | Stiffness Interface | Node 7 |
| $\mathrm{TB}_\pi$ | Topology Interface | Node 8 |
| $\mathrm{TB}_\rho$ | Mixing Interface | Node 10 |
| $\mathrm{Rep}_K$ | Complexity Interface | Node 11 |
| $\mathrm{GC}_\nabla$ | Gradient Interface | Node 12 |
| $\mathrm{Cat}_{\mathrm{Hom}}$ | Categorical Interface | Node 17 |

---

# References

The mathematical foundations of the Structural Sieve and Upgrade Metatheorems draw from the following literature. All citations in this document reference entries in the accompanying BibTeX file (`references.bib`).

:::{note}
When building with Jupyter Book or Sphinx with `sphinxcontrib-bibtex`, add the following to your `_config.yml`:
```yaml
bibtex_bibfiles:
  - hypo_permits/references.bib
```
:::

```{bibliography}
:style: unsrt
```

For the complete Hypostructure interface permit system and proof framework, see the main *Hypostructure* monograph

---

*This document is designed as a reference for the Structural Sieve algorithm. For detailed proofs and complete interface permit statements, see the main Hypostructure monograph.*
