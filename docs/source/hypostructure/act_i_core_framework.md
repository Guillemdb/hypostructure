# Part I: Core Fixed-Point Framework

*Goal: Fixed-point principle → axioms → failure modes → structural resolution*

---

# Part I: Foundations

## 1. Introduction and Overview

### 1.0 The Organizing Principle

#### 1.0.1 The challenge of understanding stability

We define a **Hypostructure** as a tuple $\mathcal{H} = (X, S_t, \Phi, \mathfrak{D}, G)$ satisfying a specific set of coherence constraints. This document establishes the category of Hypostructures and proves that global regularity in dynamical systems is equivalent to the non-existence of morphisms from a canonical singular object.

This document presents a structural approach: a **diagnostic framework** that identifies the conditions under which systems remain coherent, and classifies the ways they can fail. For classical background on partial differential equations and dispersive dynamics, see \cite{Evans10, Tao06}.

**Hypostructures provide a unified language for stability analysis.** Rather than treating each system in isolation, this framework establishes structural constraints that characterize coherent dynamics across domains. The structural axioms encode the necessary conditions for self-consistency under evolution.

**Remark (Scope and claims).** This framework is both **descriptive and diagnostic**. It classifies the structural conditions for coherence and reduces global regularity questions to local algebraic checks. Verifying that a specific system satisfies the hypostructure axioms requires only identifying the symmetries $G$ and computing the algebraic data (scaling exponents, capacity dimensions, Łojasiewicz exponents).

#### 1.0.2 The fixed-point principle: F(x) = x

The hypostructure axioms are not independent postulates chosen for technical convenience. They are manifestations of a single organizing principle: **self-consistency under evolution**.

**Definition 0.1 (Dynamical fixed point).** Let $\mathcal{S} = (X, (S_t), \Phi, \mathfrak{D})$ be a structural flow datum. A state $x \in X$ is a **dynamical fixed point** if $S_t x = x$ for all $t \in T$. More generally, a subset $M \subseteq X$ is **invariant** if $S_t(M) \subseteq M$ for all $t \geq 0$. The existence of fixed points under contraction mappings is guaranteed by the Banach fixed-point theorem \cite{Banach22}; for continuous mappings on compact convex sets, by Brouwer's theorem \cite{Brouwer11}.

**Definition 0.2 (Self-consistency).** A trajectory $u: [0, T) \to X$ is **self-consistent** if it satisfies:

1. **Temporal coherence:** The evolution $F_t: x \mapsto S_t x$ preserves the structural constraints defining $X$.
2. **Asymptotic stability:** Either $T = \infty$, or the trajectory approaches a well-defined limit as $t \nearrow T$.

**Metatheorem 0.3 (The Fixed-Point Principle).** Let $\mathcal{S}$ be a structural flow datum. The following are equivalent:

1. The system $\mathcal{S}$ satisfies the hypostructure axioms on all finite-energy trajectories.
2. Every finite-energy trajectory is asymptotically self-consistent: either it exists globally ($T_* = \infty$) or it converges to the safe manifold $M$.
3. The only persistent states are fixed points of the evolution operator $F_t = S_t$ satisfying $F_t(x) = x$.

**Remark 0.4.** The equation $F(x) = x$ encapsulates the principle: structures that persist under their own evolution are precisely those that satisfy the hypostructure axioms. Singularities represent states where $F(x) \neq x$ in the limit—the evolution attempts to produce a state incompatible with its own definition.

#### 1.0.3 The four fundamental constraints

The hypostructure axioms decompose into four orthogonal categories, each enforcing a distinct aspect of self-consistency. This decomposition is not merely organizational—it reflects the mathematical structure of the obstruction space.

**Definition 0.5 (Constraint classification).** The structural constraints divide into four classes:

| **Class** | **Axioms** | **Enforces** | **Failure Modes** |
|-----------|------------|--------------|-------------------|
| **Conservation** | D, Rec | Magnitude bounds | Modes C.E, C.D, C.C |
| **Topology** | TB, Cap | Connectivity | Modes T.E, T.D, T.C |
| **Duality** | C, SC | Perspective coherence | Modes D.D, D.E, D.C |
| **Symmetry** | LS, GC | Cost structure | Modes S.E, S.D, S.C |

Each constraint class is necessary for self-consistency:

**Conservation.** If information could be created, the past would not determine the future. The evolution $F$ would not be well-defined, violating $F(x) = x$. Conservation is necessary for temporal self-consistency.

**Topology.** If local patches could be glued inconsistently, the global state would be multiply-defined. The fixed point $x$ would not be unique, violating the functional equation. Topological consistency is necessary for spatial self-consistency.

**Duality.** If an object appeared different under observation without a transformation law, it would not be a single object. The equation $F(x) = x$ requires $x$ to be well-defined under all perspectives. Perspective coherence is necessary for identity self-consistency.

**Symmetry.** If structure could emerge without cost, spontaneous complexity generation would occur unboundedly, leading to divergence. The fixed point requires bounded energy, hence symmetry breaking must cost energy. This is necessary for energetic self-consistency.

**Proposition 0.6 (Constraint necessity).** The four constraint classes are necessary consequences of the fixed-point principle $F(x) = x$. Any system satisfying self-consistency under evolution must satisfy analogs of these constraints.

#### 1.0.4 Preview of failure modes

The four constraint classes admit three types of failure: **excess** (unbounded growth), **deficiency** (premature termination), and **complexity** (inaccessibility). Combined with boundary conditions for open systems, this yields fifteen failure modes.

**Table 0.7 (The taxonomy of failure modes).**

| **Constraint** | **Excess** | **Deficiency** | **Complexity** |
|----------------|------------|----------------|----------------|
| **Conservation** | Mode C.E: Energy blow-up | Mode C.D: Geometric collapse | Mode C.C: Finite-time event accumulation |
| **Topology** | Mode T.E: Metastasis | Mode T.D: Glassy freeze | Mode T.C: Labyrinthine |
| **Duality** | Mode D.E: Oscillatory | Mode D.D: Dispersion | Mode D.C: Semantic horizon |
| **Symmetry** | Mode S.E: Supercritical | Mode S.D: Stiffness breakdown | Mode S.C: Parameter manifold instability |
| **Boundary** | Mode B.E: Injection | Mode B.D: Starvation | Mode B.C: Misalignment |

**Remark 0.8.** Mode D.D (Dispersion) represents global existence via scattering, not a singularity. When energy does not concentrate, no finite-time blow-up occurs. The framework treats dispersion as success: if energy scatters rather than focusing, global regularity follows.

Global regularity is established by verifying that the Singular Locus $\mathcal{Y}_{\text{sing}}$ is empty—that is, Modes C.E, S.E–B.C are algebraically impossible under the structural axioms. The detailed classification of these modes appears in Chapter 4; their exclusion via metatheorems appears in Chapter 9.

#### 1.0.5 The axiomatic stance

We adopt a constructive formalism. The Hypostructure Axioms are not empirical observations but necessary conditions derived from the fixed-point principle $F(x)=x$. Within this axiomatic system, the results are rigorous consequences of the definitions.

**Definition (Metatheorem).** A structural truth derived solely from the Hypostructure Axioms. Metatheorems apply universally to any system instantiating the framework or to the learning process itself.

**Definition (Theorem).** A result pertaining to a specific mathematical object (e.g., Navier-Stokes), or a classical result cited from external literature.

The central logical operation of this framework is **exclusion**, not approximation:

1. We do not prove that solutions are smooth by constructing them.
2. We prove that singularities are impossible by showing that their existence would contradict the structural axioms.

If a physical or mathematical system satisfies the axioms of a Hypostructure, it inherits the global regularity theorems derived herein. The burden of proof shifts from "proving regularity" to "verifying the axioms."

**Remark 0.9 (No hard estimates required).** Instantiation does not require proving global compactness or global regularity *a priori*. It requires only:

1. Identifying the symmetries $G$ (translations, scalings, gauge transformations),
2. Computing the algebraic data (scaling exponents $\alpha, \beta$; capacity dimensions; Łojasiewicz exponents).

The framework then checks whether the algebraic permits are satisfied:
- If $\alpha > \beta$ (Axiom SC), supercritical blow-up is impossible.
- If singular sets have positive capacity (Axiom Cap), geometric concentration is impossible.
- If permits are denied, **global regularity follows from local structural exclusion**—analytic estimates are encoded within the algebraic permits.

#### 1.0.6 The Principle of Local Structural Exclusion

This text does not contain global estimates or integral bounds. The mechanism of proof is **soft local exclusion**, following the philosophy of Gromov's Partial Differential Relations \cite{Gromov86}, distinguishing between flexible (soft) and rigid (hard) geometric constraints:

1. **Assume failure:** Assume a singularity attempts to form.
2. **Forced structure (Axiom C):** For a singularity to exist in finite time, it must concentrate. Concentration forces the emergence of a limiting object: the canonical profile $V$.
3. **Permit denial:** Test this profile $V$ against algebraic constraints (Scaling, Capacity, Topology).
4. **Contradiction:** If the profile violates the algebraic permits, it cannot exist. Therefore, the singularity cannot form.

The framework replaces the analytical difficulty of tracking a trajectory with the algebraic difficulty of classifying a static profile.

**Local Structural Constraints.** The axioms are not global estimates assumed a priori. They are **local structural constraints**—qualitative properties verifiable in the neighborhood of a point, a profile, or a manifold:

- **Local Stiffness (LS):** Requires only that the gradient dominates the distance near an equilibrium.
- **Scaling Structure (SC):** Requires only that dissipation scales faster than time on a self-similar orbit.
- **Capacity (Cap):** Requires only that singular sets have positive dimension locally.

**From local to global.** The framework derives its strength from **integration**: these soft, local constraints are combined to produce global rigidity.

- **Local to global:** The framework does not assume global compactness. It assumes that if energy concentrates locally, it obeys local symmetries.
- **Soft to hard:** By proving that every possible local failure mode is algebraically forbidden, the framework assembles a global regularity result without performing a global estimate.

The construction of global solutions is replaced with the assembly of local constraints. If the local structure of the system rejects singularities everywhere, global smoothness follows.

#### 1.0.7 Summary

This document presents a framework for analyzing the stability of dynamical systems—from fluid dynamics and quantum fields to neural networks and markets. By identifying four constraint classes (**Conservation, Topology, Duality, and Symmetry**), we derive a taxonomy of 15 failure modes. The framework organizes 83 structural barriers from across mathematics into a catalog that characterizes when systems remain stable and when they break down.

The framework's value is **explanatory, diagnostic, and learnable**:

1. **Failure mode classification:** A systematic checklist of how systems can break, organized by constraint class and failure type.
2. **Unified language:** Common structural principles connecting theorems from different domains (Heisenberg uncertainty, Shannon limit, Bode integral, Nash-Moser).
3. **Physics derivation:** Known physical laws (GR, QM, thermodynamics) as necessary conditions for avoiding structural failure.
4. **Engineering applications:** Diagnostic tools for AI safety, control systems, and optimization.
5. **Trainable axioms:** A complete meta-theory of learning hypostructures from data, with theorems on consistency, generalization, error localization, robustness, curriculum stability, and equivariance.

The framework rests on a single organizing principle—the fixed-point equation $F(x) = x$—from which four fundamental constraint classes emerge as logical necessities. Part VII develops trainable hypostructures where the structural parameters $\theta \in \Theta$ determining the axioms are estimated via defect minimization, establishing that defect minimization converges to axiom-consistent structures and that learned hypostructures inherit the symmetries and failure-mode predictions of true theories.

**The framework's methodology:** Reduce difficult global questions to easy local checks. Verifying that a system satisfies the hypostructure axioms requires only standard calculations; the framework then delivers structural conclusions about stability, failure modes, and long-time behavior.

---

### 1.1 Overview and Roadmap

#### 1.1.1 The structural stability thesis

This program follows the spirit of **Grothendieck's *Esquisse d'un Programme* \cite{Grothendieck84}**, seeking to identify the "anabelian" structural constraints that rigidify dynamical systems, allowing global properties to be recovered from local data.

A **hypostructure** is a unified framework for analyzing dynamical systems—deterministic or stochastic, continuous or discrete—that characterizes stability through structural constraints. The central thesis is:

> **If a system satisfies the hypostructure axioms, then stability follows from structural logic. The axioms act as algebraic permits that any instability must satisfy. When these permits are denied via dimensional or geometric analysis, the instability cannot form.**

**The framework's value** lies in reducing difficult global questions to easy local checks. Verifying that a system satisfies the axioms requires only standard textbook calculations; the framework then delivers structural conclusions:

1. Explaining *why* known stable systems are stable
2. Predicting *which* failure modes are possible for a given system
3. Providing a *diagnostic checklist* for engineers and researchers

**The Exclusion Principle.** The framework does not construct solutions globally or require hard estimates. It proves regularity through the following logic:

1. **Forced Structure:** Finite-time blow-up ($T_* < \infty$) requires energy concentration. Concentration forces local structure—a canonical profile $V$ emerges wherever blow-up attempts to form.
2. **Permit Checking:** The structure $V$ must satisfy algebraic permits:
   - **Scaling Permit (Axiom SC):** Are the scaling exponents subcritical ($\alpha > \beta$)?
   - **Geometric Permit (Axiom Cap):** Does the singular set have positive capacity?
   - **Topological Permit (Axiom TB):** Is the topological sector accessible?
   - **Stiffness Permit (Axiom LS):** Does the Łojasiewicz inequality hold near equilibria?
3. **Contradiction:** If any permit is denied, the singularity cannot form. Global regularity follows.

**Mode D.D (Dispersion) is not a singularity.** When energy does not concentrate (Axiom C fails), no finite-time singularity forms—the solution exists globally and disperses. Mode D.D represents **global existence via scattering**, not a failure mode.

**No global estimates required.** The framework never requires proving global compactness or global bounds. All analysis is local: concentration forces structure, structure is tested against algebraic permits, permit denial implies regularity. The classification is **logically exhaustive**: every trajectory either disperses globally (Mode D.D), blows up via energy escape (Mode C.E), or has its blow-up attempt blocked by permit denial (Modes S.E–B.C contradict, yielding regularity).

### 1.2 How to read this document

**Logical Dependencies (DAG Structure).** The framework is modular. The logical flow forms a directed acyclic graph:

| Step | Parts | Function | Output |
|:-----|:------|:---------|:-------|
| 1 | **Foundations (I-II)** | Defines the Object | Hypostructure $\mathcal{H}$ |
| 2 | **Taxonomy (III)** | Defines the Problem | Singular Locus $\mathcal{Y}_{\text{sing}}$ |
| 3 | **Metatheorems (IV, X)** | Defines the Tools | Boolean Permits $\Pi_A$ |
| 4 | **Barriers (V)** | Quantifies Permits | Sharp Constants |
| 5 | **Instantiations (VI)** | Maps Reality to $\mathcal{H}$ | Concrete Systems |
| 6 | **Learning (VII)** | Automates Step 5 | AGI Blueprint |

**Dependencies:** (1) → (2) → (3) → (4); (1) → (5); (3) + (5) → (6). Parts VIII-XI extend the framework but are not prerequisites for applications.

This document is organized into eleven parts:

**Part I: Foundations (Chapters 0–1).** The organizing principle, constraint structure, and main thesis. Establishes the conceptual foundation: self-consistency under evolution, the four fundamental constraints, and the logic of soft local exclusion.

**Part II: Mathematical Foundations (Chapters 2–3).** Formal definitions of the hypostructure axioms. Chapter 2 presents the mathematical preliminaries (state spaces, semiflows, functional calculus). Chapter 3 develops the complete axiom system: core axioms (C, D, Rec) and structural axioms (SC, Cap, LS, TB, GC, Reg).

**Part III: The Failure Taxonomy (Chapter 4).** Complete classification of the fifteen ways self-consistency can break. Each mode is defined rigorously with diagnostic criteria, prototypical examples, and exclusion conditions. Organized by constraint class (Conservation, Topology, Duality, Symmetry, Boundary) and failure type (Excess, Deficiency, Complexity).

**Part IV: Core Metatheorems (Chapters 5–7).** The main theorems. Chapter 5 establishes normalization and gauge structure (Bubbling Decomposition, Profile Classification). Chapter 6 derives the resolution theorems (Type II Exclusion, Capacity Barriers, Topological Suppression, Canonical Lyapunov, Action Reconstruction). Chapter 7 presents the structural resolution of maximizers and compactness restoration.

**Part V: The Eighty-Five Barriers (Chapters 8–11).** The complete barrier catalog organized by constraint class: Conservation barriers (Chapter 8), Topology barriers (Chapter 9), Duality barriers (Chapter 10), Symmetry barriers (Chapter 11), plus computational, quantum, and additional structural barriers (Chapters 11B–11D). Each barrier provides a quantitative obstruction excluding specific failure modes.

**Part VI: Concrete Instantiations (Chapter 12).** Applications to physical and mathematical systems: fluid dynamics, geometric flows (mean curvature, Ricci), gauge theories, nonlinear wave equations, reaction-diffusion systems. These instantiations demonstrate the framework in action.

**Part VII: Trainable Hypostructures and Learning (Chapters 13–14).** The meta-theory of learning axioms. Chapter 13 develops trainable hypostructures with nine metatheorems: Consistency and Convergence (§13.6), Meta-Error Localization (§13.7), Block Factorization (§13.8), Meta-Generalization (§13.9), Expressivity (§13.10), Active Probing (§13.11), Robustness of Failure-Mode Predictions (§13.12), Curriculum Stability (§13.13), and Equivariance (§13.14). Chapter 14 presents the General Loss functional with structural identifiability theorems.

**Part VIII: Synthesis (Chapter 15).** Meta-axiomatics and the unity of structure. Establishes that the hypostructure axioms form a minimal complete system: necessary and sufficient for structural coherence, with no redundancy.

**Part IX: The Isomorphism Dictionary (Chapter 16).** Structural correspondences across mathematical domains. Shows how the same barrier mechanisms manifest in different settings (PDE, probability, algebra, computation).

**Part X: Foundational Metatheorems (Chapters 17–18).** Completeness, minimality, universality, and identifiability of hypostructures. Proves that the axiom system is the unique minimal system capturing structural coherence. Section 18.4 presents fourteen Global Metatheorems (19.4.A–N): local-to-global machinery (tower globalization, obstruction collapse, stiff pairings), the master schema reducing conjectures to Axiom R, meta-learning of admissible structure, the categorical obstruction strategy via universal R-breaking patterns, the computational layer (parametric realization and adversarial search), and the capstone Principle of Structural Exclusion theorem unifying all previous metatheorems.

**Part XI: Fractal Set Foundations (Chapters 19–20).** Advanced topics: fractal set representation of singular structures, emergent spacetime from hypostructure dynamics, and observer-dependent perspectives.

**How to approach the text.** Readers familiar with PDE regularity theory can begin with Part III (failure modes) and Part IV (core metatheorems), referring to Part II for axiom definitions as needed. Readers interested in foundations should read Parts I–II sequentially. Readers seeking applications can proceed directly to Part VI after reviewing the axioms. Researchers in machine learning should focus on Part VII (trainable hypostructures) after understanding the axiom system in Parts II–III.

### 1.3 Main consequences

From the hypostructure axioms, we derive:

**Core meta-theorems (Chapter 7):**

**Metatheorem 1.1 (Structural Resolution).** Every trajectory resolves into one of three outcomes: global existence (dispersive), global regularity (permit denial), or genuine singularity. This is the dynamical analogue of Hironaka's Resolution of Singularities Theorem in algebraic geometry \cite{Hironaka64}, blowing up the singular locus to a smooth divisor.

**Metatheorem 1.2 (Type II Exclusion).** Under SC + D, supercritical self-similar blow-up is impossible at finite cost—derived from scaling arithmetic alone.

**Metatheorem 1.3 (Capacity Barrier).** Trajectories cannot concentrate on arbitrarily thin or high-codimension sets.

**Metatheorem 1.4 (Topological Suppression).** Nontrivial topological sectors are exponentially rare under the invariant measure.

**Metatheorem 1.5 (Structured vs Failure Dichotomy).** Finite-energy trajectories are eventually confined to a structured region where classical regularity holds.

**Metatheorem 1.6 (Canonical Lyapunov Functional).** There exists a unique (up to monotone reparametrization) Lyapunov functional determined by the structural data.

**Metatheorem 1.7 (Functional Reconstruction).** Under gradient consistency, the Lyapunov functional is explicitly recoverable as the geodesic distance in a Jacobi metric, or as the solution to a Hamilton–Jacobi equation. No prior knowledge of an energy functional is required.

**Quantitative metatheorems (Chapter 9).** The framework provides **eighty-three structural barriers** organized into thirty-six categories:

*Classical and Geometric Barriers:*

- **Coherence Quotient, Spectral Convexity, Gap-Quantization** — Energy alignment, interaction potentials, phase transitions
- **Symplectic Transmission, Non-Squeezing** — Phase space rigidity and rank conservation
- **Dimensional Rigidity, Isoperimetric Resilience** — Geometric topology preservation
- **Wasserstein Transport, Chiral Anomaly Lock** — Mass movement and helicity conservation

*Information-Theoretic Barriers:*

- **Shannon–Kolmogorov, Bekenstein-Landauer** — Entropy bounds and information-energy coupling
- **Holographic Encoding, Holographic Compression** — Scale-geometry duality and isospectral locking
- **Cardinality Compression** — Separable Hilbert space constraints

*Algebraic and Arithmetic Barriers:*

- **Galois–Monodromy Lock** — Orbit exclusion via field theory
- **Algebraic Compressibility** — Degree-volume locking via Northcott bounds
- **Arithmetic Height** — Diophantine avoidance of resonances

*Computational and Logical Barriers:*

- **Algorithmic Causal Barrier** — Logical depth exclusion
- **Gödel-Turing Censor** — Chronology protection from self-reference
- **Tarski Truth Barrier** — Undefinability of truth predicates
- **Semantic Resolution Barrier** — Berry paradox and descriptive complexity

*Control-Theoretic Barriers:*

- **Nyquist–Shannon Stability, Bode Sensitivity Integral** — Bandwidth and sensitivity conservation
- **Causal Lag Barrier** — Delay feedback stability
- **Synchronization Manifold** — Coupled oscillator stability

*Quantum and Foundational Barriers:*

- **Isometric Cloning Prohibition, Entanglement Monogamy** — Quantum information constraints
- **Quantum Zeno Suppression, QEC Threshold** — Measurement and error correction
- **Vacuum Nucleation Barrier** — Coleman-De Luccia stability

*Graph-Theoretic and Combinatorial Barriers:*

- **Byzantine Fault Tolerance** — Consensus threshold in distributed systems ($n \geq 3f+1$)
- **Percolation Threshold** — Phase transitions in random graphs
- **Near-Decomposability** — Block diagonal structure in adjacency matrices

*Function Space and Optimization Barriers:*

- **No Free Lunch Theorem** — Uniform bounds on learning functionals
- **Johnson-Lindenstrauss** — Dimension reduction in normed spaces
- **Pseudospectral Bound** — Transient amplification via resolvent norms

*Scaling and Iteration Barriers:*

- **Power-Law Scaling** — Fractional exponent constraints on functional growth
- **Eigen Error Threshold** — Mutation-selection balance in discrete dynamical systems
- **Martingale Conservation** — No-arbitrage in filtered probability spaces

*Reconstruction and Embedding Barriers:*

- **Takens Embedding** — Diffeomorphism from delay coordinates to attractor
- **Hyperbolic Shadowing** — Pseudo-orbit tracing in Axiom A systems
- **Stochastic Stability** — Persistence of invariant measures under perturbation

*Holonomy and Curvature Barriers:*

- **Sagnac-Holonomy Effect** — Path-dependent phase in fiber bundles
- **Maximum Force Conjecture** — Upper bounds on stress-energy flux

*Definability and Semantic Barriers:*

- **Sorites Threshold** — Vagueness in predicate extensions
- **Intersubjective Consistency** — Compatibility of observation frames
- **Counterfactual Stability** — Acyclicity in causal DAGs

*Computational Complexity Barriers:*

- **Amdahl Scaling Limit** — Parallelization bounds on speedup functions
- **Recursive Simulation Limit** — Information-theoretic bounds on self-modeling
- **Epistemic Horizon** — Computational irreducibility in cellular automata

**Trainable hypostructures (Chapter 13):**

- Axioms treated as learnable parameters optimized via defect minimization
- Parametric families of height functionals, dissipation structures, and symmetry groups
- Joint optimization over hypostructure components and extremal profiles

*Nine metatheorems establishing the meta-theory of learning axioms:*

- **Metatheorem 13.20 (Trainable Hypostructure Consistency):** Gradient descent on joint axiom risk converges to axiom-consistent hypostructures
- **Metatheorem 13.29 (Meta-Error Localization):** Block-restricted reoptimization identifies which axiom blocks are misspecified
- **Metatheorem 13.37 (Meta-Generalization):** Training on system distributions generalizes with $O(\sqrt{\varepsilon + 1/\sqrt{N}})$ bounds
- **Metatheorem 13.40 (Axiom-Expressivity):** Parametric families can approximate any admissible hypostructure with arbitrarily small defect
- **Metatheorem 13.44 (Optimal Experiment Design):** Sample complexity for hypostructure identification is $O(d\sigma^2/\Delta^2)$
- **Metatheorem 13.50 (Robustness of Failure-Mode Predictions):** Discrete permit-denial judgments are stable under small axiom risk
- **Metatheorem 13.54 (Curriculum Stability):** Warm-start training tracks the structural path without jumping to spurious ontologies
- **Metatheorem 13.61 (Equivariance):** Learned hypostructures inherit all symmetries of the system distribution

**General loss (Chapter 14):**

- Training objective for systems that instantiate, verify, and optimize over hypostructures
- Four loss components: structural loss (energy/symmetry identification), axiom loss (soft axiom satisfaction), variational loss (extremal candidate quality), meta-loss (cross-system generalization)
- **Metatheorem 14.27 (Defect Reconstruction):** Defect signatures determine hypostructure components from axioms alone
- **Metatheorem 14.30 (Meta-Identifiability):** Parameters are learnable under persistent excitation and nondegenerate parametrization

**Global Metatheorems (Section 18.4):**

*Fourteen framework-level tools applicable across all instantiations:*

- **Metatheorem 19.4.A (Tower Globalization):** Local invariants determine global asymptotic structure
- **Metatheorem 19.4.B (Obstruction Collapse):** Obstruction sectors are finite-dimensional under subcritical accumulation
- **Metatheorem 19.4.C (Stiff Pairing):** Non-degenerate pairings exclude null directions
- **Metatheorem 19.4.D (Local → Global Height):** Local Northcott + coercivity yields global height with finiteness
- **Metatheorem 19.4.E (Local → Subcritical):** Local growth bounds automatically imply subcritical dissipation
- **Metatheorem 19.4.F (Local Duality → Stiffness):** Local perfect duality + exactness yields global non-degeneracy
- **Metatheorem 19.4.G (Conjecture-Axiom Equivalence):** Conjecture($Z$) $\Leftrightarrow$ Axiom R($Z$) for admissible objects
- **Metatheorem 19.4.H (Meta-Learning):** Admissible structure can be learned via axiom risk minimization
- **Metatheorem 19.4.I (Categorical Structure):** Category $\mathbf{Hypo}_T$ of T-hypostructures and morphisms
- **Metatheorem 19.4.J (Universal Bad Pattern):** Initial object $\mathbb{H}_{\mathrm{bad}}^{(T)}$ of R-breaking subcategory
- **Metatheorem 19.4.K (Categorical Obstruction Schema):** Empty Hom-set from universal bad pattern $\Rightarrow$ R-validity
- **Metatheorem 19.4.L (Parametric Realization):** $\Theta$-search equivalent to hypostructure search
- **Metatheorem 19.4.M (Adversarial Training):** Min-max game discovers R-breaking patterns or certifies absence
- **Metatheorem 19.4.N (Principle of Structural Exclusion):** Capstone unifying all metatheorems into single exclusion principle
- **Metatheorem 21 (Singularity Completeness):** Partition-of-unity gluing guarantees $\mathbf{Blowup}$ is universal for $\mathcal{T}_{\mathrm{sing}}$
- **Corollary 21.1 (Singularity Exclusion):** Blowup exclusion + completeness $\Rightarrow$ $\mathcal{T}_{\mathrm{sing}} = \varnothing$

**Three-Layer Axiom Architecture (Sections 3.0 and 18.3.5):**

The axioms organize into three layers of increasing abstraction:

- **S-Layer (Structural):** Core axioms X.0 enabling structural resolution and basic metatheorems 19.4.A–C
- **L-Layer (Learning):** Axioms L1 (expressivity), L2 (excitation), L3 (identifiability) enabling meta-learning (19.4.H), local-to-global construction (19.4.D–F), and full structural obstruction (19.4.N)
- **$\Omega$-Layer (AGI Limit):** Single meta-hypothesis reducing all L-axioms to universal structural approximation, enabling Theorem 0 (Convergence of Structure)

The layers form a hierarchy: L-axioms derive S-layer properties as theorems rather than assumptions; $\Omega$-axioms derive L-axioms from universal approximation and active probing. Users work at the layer appropriate to their verification capability.

### 1.4 Scope of instantiation

The framework instantiates across the following mathematical structures:

**Partial differential equations.** Parabolic, hyperbolic, and dispersive equations; geometric flows (mean curvature flow, Ricci flow); incompressible fluid equations on Riemannian manifolds.

**Stochastic processes.** McKean–Vlasov equations, Fleming–Viot processes, interacting particle systems, Langevin dynamics, Itô diffusions on manifolds.

**Discrete dynamical systems.** $\lambda$-calculus reduction systems, interaction nets, graph rewriting systems, Turing machine configurations, cellular automata on $\mathbb{Z}^d$.

**Algebraic structures.** Elliptic curves over finite fields, algebraic varieties, Galois representations, height functions on projective varieties.

**Function spaces.** Banach space optimization, Fréchet manifolds, loss landscapes on parameter spaces, kernel methods in reproducing kernel Hilbert spaces.

**Operator semigroups.** $C_0$-semigroups, transfer operators, Koopman operators, pseudospectral analysis, delay differential equations.

**Random graphs.** Erdős–Rényi percolation, configuration models, consensus dynamics on graphs, spectral graph theory.

**Hilbert space operators.** Unitary groups, self-adjoint extensions, quantum channels, completely positive maps, tensor products.

**Fiber bundles.** Principal bundles with connection, holonomy groups, characteristic classes, Chern-Weil theory.

**Iteration schemes.** Recursive function composition, fixed-point theorems, contraction mappings, asymptotic analysis.

**Attractor theory.** Strange attractors, fractal dimension, box-counting dimension, Hausdorff measure, delay embeddings.

**Remark 1.8 (Verification procedure).** Instantiation does not require proving global compactness or global regularity *a priori*. It requires only:

1. Identifying the symmetries $G$ (translations, scalings, gauge transformations),
2. Computing the algebraic data (scaling exponents $\alpha, \beta$; capacity dimensions; Łojasiewicz exponents).

The framework then checks whether the algebraic permits are satisfied:
- If $\alpha > \beta$ (Axiom SC), supercritical blow-up is impossible.
- If singular sets have positive capacity (Axiom Cap), geometric concentration is impossible.
- If permits are denied, **global regularity follows from local structural exclusion**—analytic estimates are encoded within the algebraic permits.

The only remaining possibility is Mode D.D (dispersion), which is not a finite-time singularity but global existence via scattering.

**Remark 1.9 (Universality).** This universality is not accidental. The hypostructure axioms capture the minimal conditions for structural coherence—the requirements that any well-posed mathematical object must satisfy. The metatheorems are structural invariants that hold wherever the axioms are instantiated.

**Conjecture 1.10 (Structural universality).** Every well-posed mathematical system admits a hypostructure in which the core theorems hold. Ill-posedness is equivalent to unavoidable violation of one or more constraint classes.

This document develops the framework systematically across multiple domains.

---

## 3. The Axiom System

A **hypostructure** is a structural flow datum $\mathcal{S}$ satisfying the following axioms. The axioms are organized by their role in constraining system behavior.

### 3.0 Axiom Layers: Structure, Learning, and Universality

The hypostructure axioms organize into **three layers of increasing abstraction**. Each layer subsumes the previous, enabling progressively more powerful machinery:

**Layer S (Structural Axioms).** The core axioms C.0, D.0, SC.0, LS.0, Cap.0, TB.0, GC.0, and R define what a valid hypostructure must satisfy. These are the mathematical constraints—energy balance, dissipation, scale coherence, capacity bounds, and dictionary correspondence.

*What S-Layer enables:* Metatheorems 19.4.A–N (local-to-global globalization, obstruction collapse, stiff pairings, categorical obstruction, master structural resolution). With only the S-layer, the framework provides structural resolution: every trajectory either converges to an attractor, exits to infinity in a controlled way, or fails in a classified mode.

**Layer L (Learning Axioms).** Three additional hypotheses enable the computational machinery:

- **L1 (Representational Completeness):** A parametric family $\Theta$ is dense in admissible structures—every hypostructure can be approximated to arbitrary precision. *Justified by Theorem 13.40 (Axiom-Expressivity).*

- **L2 (Persistent Excitation):** Training data distinguishes structures—no two genuinely different hypostructures produce identical defect signatures. *Ensures identifiability from finite data.*

- **L3 (Non-Degenerate Parametrization):** The map $\theta \mapsto \mathbb{H}(\theta)$ is locally Lipschitz and injective—small parameter changes yield small structural changes, and distinct parameters yield distinct structures. *Justified by Theorem 14.30 (Meta-Identifiability).*

*What L-Layer enables:* The analytic properties assumed in the S-layer become **derivable theorems**:

| Property | Derived Via | Theorem |
|----------|-------------|---------|
| Global Coercivity | L3 (Identifiability) | 14.30 |
| Global Height | L1 + meta-learning | 19.4.D |
| Subcritical Scaling | L1 + meta-learning | 19.4.E |
| Stiffness | L1 + meta-learning | 19.4.F |

**Layer $\Omega$ (AGI Limit).** The theoretical limit—reduces all L-layer assumptions to a single meta-hypothesis. The key insight is that several L-axioms become derivable under stronger conditions:

1. *S1 (Admissibility) becomes diagnostic:* The framework tests regularity rather than assuming it (Theorem 15.21).
2. *L2 (Excitation) eliminated:* Active Probing (Theorem 13.44) generates persistently exciting data.
3. *L3 (Identifiability) relaxed:* Singular Learning Theory (Watanabe) shows that the RLCT controls convergence even in degenerate landscapes.
4. *L1 (Expressivity) weakened:* Replace fixed $\Theta$ with a hierarchy $\Theta_1 \subset \Theta_2 \subset \cdots$ of increasing expressivity.

This yields **Axiom $\Omega$ (AGI Limit):** Access to a learning agent $\mathcal{A}$ equipped with:
- *Universal Approximation:* $\Theta = \bigcup_n \Theta_n$ dense in continuous functionals on trajectory data.
- *Optimal Experiment Design:* Ability to probe system $S$ and observe trajectories.
- *Defect Minimization:* Optimization oracle for the axiom risk $\mathcal{R}(\theta)$.

**Hypothesis $\Omega$ (Universal Structural Approximation):** System $S$ belongs to the closure of computable hypostructures—physics approximable by finite combination of (Energy, Dissipation, Symmetry, Topology).

*What $\Omega$-Layer enables:* **Metatheorem 0 (Convergence of Structure)**, combining Theorems 13.44, 13.40, and 15.25:
1. If $S$ is regular $\Rightarrow$ $\mathcal{A}$ converges to $\theta^*$ satisfying all structural axioms.
2. If $S$ is singular $\Rightarrow$ non-zero defects classify the failure mode (Response Signature).
3. Analytic properties (global bounds, coercivity, stiffness) emerge as properties of $\theta^*$.

**User perspective.** The three layers are not competing alternatives—they form a hierarchy. A user works at the layer appropriate to their verification capability:

- *S-layer only:* Verify structural axioms directly $\Rightarrow$ apply metatheorems 19.4.A–N.
- *S + L layers:* Verify learning axioms $\Rightarrow$ derive S-properties as theorems, not assumptions.
- *$\Omega$-layer:* Assume universal structural approximation $\Rightarrow$ derive L-properties from universal approximation.

The development in Parts III–VI focuses on the S-layer. Part VII develops the L-layer. The full $\Omega$-layer machinery appears in Section 18.3.5.

---

### 3.1 Conservation constraints

These axioms govern energy balance and recovery mechanisms—the thermodynamic backbone of the framework.

#### Axiom D (Dissipation)

**Axiom D (Dissipation bound along trajectories).** Along any trajectory $u(t) = S_t x$, there exists $\alpha > 0$ such that for all $0 \leq t_1 \leq t_2 < T_*(x)$:
$$
\Phi(u(t_2)) + \alpha \int_{t_1}^{t_2} \mathfrak{D}(u(s)) \, ds \leq \Phi(u(t_1)) + C_{u}(t_1, t_2),
$$
where the **drift term** $C_u(t_1, t_2)$ satisfies:

- **On the good region $\mathcal{G}$:** $C_u(t_1, t_2) = 0$ when $u(s) \in \mathcal{G}$ for all $s \in [t_1, t_2]$.
- **Outside $\mathcal{G}$:** $C_u(t_1, t_2) \leq C \cdot \mathrm{Leb}\{s \in [t_1, t_2] : u(s) \notin \mathcal{G}\}$ for some constant $C \geq 0$.

**Fallback (Mode C.E: Energy Blow-up).** When Axiom D fails—i.e., the energy grows without bound—the trajectory exhibits **energy blow-up**. The drift term is uncontrolled, leading to $\Phi(u(t)) \to \infty$ as $t \nearrow T_*(x)$.

**Role in constraint class.** Axiom D provides the fundamental energy–dissipation balance. It ensures that energy cannot increase without bound unless the system remains outside the good region $\mathcal{G}$ for an extended time. The drift term controls energy growth outside $\mathcal{G}$, and is regulated by Axiom Rec.

**Corollary 3.1 (Integral bound).** For any trajectory with finite time in bad regions (guaranteed by Axiom Rec when $\mathcal{C}_*(x) < \infty$):
$$
\int_0^{T_*(x)} \mathfrak{D}(u(t)) \, dt \leq \frac{1}{\alpha}\left(\Phi(x) - \Phi_{\min} + C \cdot \tau_{\mathrm{bad}}\right),
$$
where $\tau_{\mathrm{bad}} = \mathrm{Leb}\{t : u(t) \notin \mathcal{G}\}$ is finite by Axiom Rec.

**Remark 3.2 (Connection to entropy methods).** In gradient flow and entropy method contexts:
- $\Phi$ is the free energy or relative entropy,
- $\mathfrak{D}$ is the entropy production rate or Fisher information,
- The inequality becomes the entropy–entropy production inequality,
- The drift $C_u = 0$ on the good region is the entropy-dissipation identity.

#### Axiom Rec (Recovery)

**Axiom Rec (Recovery inequality along trajectories).** Along any trajectory $u(t) = S_t x$, there exist:

- a measurable subset $\mathcal{G} \subseteq X$ called the **good region**,
- a measurable function $\mathcal{R}: X \to [0, \infty)$ called the **recovery functional**,
- a constant $C_0 > 0$,

such that:

1. **Positivity outside $\mathcal{G}$:** $\mathcal{R}(x) > 0$ for all $x \in X \setminus \mathcal{G}$ (spatially varying, not necessarily uniform),
2. **Recovery inequality:** For any interval $[t_1, t_2] \subset [0, T_*(x))$ during which $u(t) \in X \setminus \mathcal{G}$:
$$
\int_{t_1}^{t_2} \mathcal{R}(u(s)) \, ds \leq C_0 \int_{t_1}^{t_2} \mathfrak{D}(u(s)) \, ds.
$$

**Fallback (Mode C.E: Energy Blow-up).** When Axiom Rec fails—i.e., recovery is impossible along a trajectory—the trajectory enters a **failure region** $\mathcal{F}$ where the drift term in Axiom D is uncontrolled, leading to energy blow-up.

**Role in constraint class.** Axiom Rec is the dual to Axiom D: it bounds the time a trajectory can spend outside the good region $\mathcal{G}$ in terms of dissipation cost. Together, D and Rec ensure that finite-cost trajectories cannot drift indefinitely in bad regions. The recovery functional $\mathcal{R}$ may vary spatially—some bad regions have fast recovery (large $\mathcal{R}$), others slow recovery (small $\mathcal{R}$).

**Proposition 3.3 (Time bound outside good region).** Under Axioms D and Rec, for any trajectory with finite total cost $\mathcal{C}_*(x) < \infty$, define $r_{\min}(u) := \inf_{t : u(t) \notin \mathcal{G}} \mathcal{R}(u(t))$. If $r_{\min}(u) > 0$:
$$
\mathrm{Leb}\{t \in [0, T_*(x)) : u(t) \notin \mathcal{G}\} \leq \frac{C_0}{r_{\min}(u)} \mathcal{C}_*(x).
$$

*Proof.* Let $A = \{t : u(t) \notin \mathcal{G}\}$. Then
$$
r_{\min}(u) \cdot \mathrm{Leb}(A) \leq \int_A \mathcal{R}(u(t)) \, dt \leq C_0 \int_0^{T_*(x)} \mathfrak{D}(u(t)) \, dt = C_0 \mathcal{C}_*(x). \qquad \square
$$

**Remark 3.4 (Adaptive recovery).** The recovery rate $\mathcal{R}(x)$ may vary spatially. Only the trajectory-specific minimum $r_{\min}(u)$ matters, and this is positive whenever Axiom Rec holds along that trajectory.

#### 3.1.7 The Recovery-Correspondence Duality

Axiom Rec (Recovery) and Axiom R (Structural Correspondence, Definition 16.1) govern distinct aspects of the same phenomenon: the capacity of a state to maintain structural coherence. This subsection establishes their categorical equivalence.

**Proposition 3.17 (Recovery-Dictionary Isomorphism).** *Let $\mathbb{H}$ be a hypostructure equipped with Recovery functional $\mathcal{R}$ (Axiom Rec), and suppose the Dictionary map $D: X \to \mathcal{T}$ (Axiom R) exists. Then the Recovery functional admits the representation:*

$$\mathcal{R}(u) = \|D(u) - D^{-1}(D(u))\|_{\mathcal{T}}$$

*where the norm quantifies the invertibility defect of the Dictionary.*

*Structural interpretation:*
- If Axiom R holds (the Dictionary is an equivalence), then $D^{-1} \circ D = \mathrm{id}$, whence $\mathcal{R}(u) = 0$ for all $u \in X$, placing every state in the good region $\mathcal{G}$.
- If Axiom R fails (the Dictionary is not invertible), then $\mathcal{R}(u) > 0$ measures the distance from the structurally coherent regime.

*Proof.* We establish the equivalence via the constraint class structure.

**Step 1 (Recovery implies Dictionary coherence).** Assume Axiom Rec holds with recovery inequality $\int \mathcal{R}(u(s)) \, ds \leq C_0 \int \mathfrak{D}(u(s)) \, ds$. Define the effective Dictionary defect:
$$\delta_D(u) := \inf\{\|u - v\| : v \in \ker(\alpha_D \circ \iota_D - \mathrm{id})\}$$
where $\iota_D$ denotes instantiation and $\alpha_D$ denotes abstraction (Definition 16.1). The recovery functional controls this defect: $\delta_D(u) \leq C \cdot \mathcal{R}(u)$ for some constant $C > 0$.

**Step 2 (Dictionary coherence implies Recovery).** Conversely, suppose the Dictionary is $\varepsilon$-invertible, i.e., $\|D^{-1}(D(u)) - u\| \leq \varepsilon$ for all $u \in X$. Setting $\mathcal{R}(u) := \|D(u) - D^{-1}(D(u))\|$ yields the recovery inequality with constant $C_0$ depending on the Lipschitz constant of $D$.

**Step 3 (Good region characterization).** The good region admits the characterization:
$$\mathcal{G} = \{u \in X : D^{-1}(D(u)) = u\} = \ker(\mathcal{R})$$
consisting precisely of states for which the Dictionary is exact. $\square$

**Corollary 3.17.1 (Unified Failure Mechanism).** *A trajectory fails to recover (violates Axiom Rec) if and only if its structural dictionary degrades (violates Axiom R). This establishes the equivalence of Conservation and Duality failures.*

**Remark 3.17.2 (Dual Perspectives).** Axiom Rec and Axiom R encode complementary perspectives on the same structural constraint:

| Perspective | Axiom | Quantity Measured | Domain |
|:------------|:------|:------------------|:-------|
| **Dynamical** | Rec | Dissipation cost to return to $\mathcal{G}$ | Trajectory space |
| **Representational** | R | Fidelity of structural translation | Dictionary space |

The dynamical formulation quantifies persistence outside the good region; the representational formulation quantifies translation accuracy between domains. The isomorphism demonstrates these to be equivalent characterizations.

**Example 3.17.3 (Fluid dynamics).** For the Navier-Stokes equations, let $D$ denote the Fourier-Littlewood-Paley decomposition mapping velocity fields to frequency-localized components. The recovery functional $\mathcal{R}(u) = \|u - P_{\leq N} u\|$ measures high-frequency content. States satisfying $\mathcal{R}(u) = 0$ (purely low-frequency) constitute the good region where finite-dimensional Galerkin approximations are exact.

**Example 3.17.4 (Optimal transport).** For gradient flows on $\mathcal{P}_2(\mathbb{R}^n)$, let $D$ map probability measures to truncated moment sequences. The recovery functional $\mathcal{R}(\mu) = W_2(\mu, \hat{\mu}_N)$ measures the Wasserstein distance to the $N$-moment reconstruction $\hat{\mu}_N$. The good region comprises measures fully determined by finitely many moments, including Gaussian measures.

### 3.2 Topology constraints

These axioms govern spatial structure and geometric concentration—where and how mass can accumulate.

#### Axiom TB (Topological Background)

**Structural Data (Topological sector structure).** The system admits a topological sector structure:
- a discrete (or locally finite) index set $\mathcal{T}$,
- a measurable function $\tau: X \to \mathcal{T}$ called the **sector index**,
- a distinguished element $0 \in \mathcal{T}$ called the **trivial sector**,
- an **action functional** $\mathcal{A}: X \to [0, \infty]$ measuring topological cost.

The sector index is **flow-invariant**: $\tau(S_t x) = \tau(x)$ for all $t \in [0, T_*(x))$.

**Axiom TB1 (Action gap).** There exists $\Delta > 0$ such that for all $x$ with $\tau(x) \neq 0$:
$$
\mathcal{A}(x) \geq \mathcal{A}_{\min} + \Delta,
$$
where $\mathcal{A}_{\min} = \inf_{x: \tau(x) = 0} \mathcal{A}(x)$.

**Axiom TB2 (Action-height coupling).** The action is controlled by the height: there exists $C_{\mathcal{A}} > 0$ such that
$$
\mathcal{A}(x) \leq C_{\mathcal{A}} \Phi(x).
$$

**Fallback (Mode T.E: Topological Obstruction).** When Axiom TB fails along a trajectory—i.e., the trajectory is constrained to a nontrivial topological sector $\tau \neq 0$ with action exceeding the gap—topological invariants prevent the singularity from forming.

**Role in constraint class.** Axiom TB provides topological obstructions to concentration. Nontrivial topological sectors (e.g., nonzero degree, Chern number, homotopy class) carry a minimum action cost $\Delta$. Trajectories in such sectors must pay this action penalty, which may exceed the available energy budget, thereby blocking singularity formation.

**Example 3.5 (Topological charges).**

1. **Degree:** For maps $u: S^n \to S^n$, $\tau(u) = \deg(u) \in \mathbb{Z}$.
2. **Chern number:** For connections on a bundle, $\tau(A) = c_1(A) \in \mathbb{Z}$.
3. **Homotopy class:** $\tau(u) = [u] \in \pi_n(M)$.
4. **Vorticity:** $\tau(u) = \int \omega \, dx$ for fluid flows.

#### Axiom Cap (Capacity)

**Axiom Cap (Capacity bound along trajectories).** Along any trajectory $u(t) = S_t x$, there exist:

- a measurable function $c: X \to [0, \infty]$ called the **capacity density**,
- constants $C_{\mathrm{cap}} > 0$ and $C_0 \geq 0$,

such that the capacity integral is controlled by the dissipation budget:
$$
\int_0^{\min(T, T_*(x))} c(u(t)) \, dt \leq C_{\mathrm{cap}} \int_0^{\min(T, T_*(x))} \mathfrak{D}(u(t)) \, dt + C_0 \Phi(x).
$$

**Fallback (Mode C.D: Geometric Concentration).** When Axiom Cap fails along a trajectory—i.e., the trajectory concentrates on high-capacity sets without commensurate dissipation—the trajectory exhibits **geometric concentration** that violates the capacity barrier.

**Role in constraint class.** Axiom Cap quantifies geometric accessibility: trajectories can only occupy high-capacity regions if they are actively dissipating. It prevents passive accumulation in thin or singular structures. Capacity is tied to dissipation, not time—spending time in high-capacity regions requires dissipation budget.

**Definition 3.6 (Capacity of a set).** The **capacity** of a measurable set $B \subseteq X$ is
$$
\mathrm{Cap}(B) := \inf_{x \in B} c(x).
$$

**Proposition 3.7 (Occupation time bound).** Under Axiom Cap, for any trajectory with finite cost $\mathcal{C}_T(x) < \infty$ and any set $B$ with $\mathrm{Cap}(B) > 0$:
$$
\mathrm{Leb}\{t \in [0, T] : u(t) \in B\} \leq \frac{C_{\mathrm{cap}} \mathcal{C}_T(x) + C_0 \Phi(x)}{\mathrm{Cap}(B)}.
$$

*Proof.* Let $\tau_B = \mathrm{Leb}\{t \in [0, T] : u(t) \in B\}$. Then
$$
\mathrm{Cap}(B) \cdot \tau_B \leq \int_0^T c(u(t)) \mathbf{1}_{u(t) \in B} \, dt \leq \int_0^T c(u(t)) \, dt \leq C_{\mathrm{cap}} \mathcal{C}_T(x) + C_0 \Phi(x). \qquad \square
$$

**Remark 3.8.** Capacity measures how "expensive" (in dissipation cost) it is to visit a region. High-capacity sets are accessible only to trajectories with high dissipation budgets.

### 3.3 Duality constraints

These axioms enforce compactness and scaling balance—the self-similar structure of concentrating solutions.

#### Axiom C (Compactness)

**Structural Data (Symmetry Group).** The system admits a continuous action by a locally compact topological group $G$ acting on $X$ by isometries (i.e., $d(g \cdot x, g \cdot y) = d(x, y)$ for all $g \in G$, $x, y \in X$). This is structural data about the system, not an assumption to be verified per trajectory.

**Axiom C (Structural Compactness Potential).** We say a trajectory $u(t) = S_t x$ with bounded energy $\sup_{t < T_*(x)} \Phi(u(t)) \leq E < \infty$ **satisfies Axiom C** if: for every sequence of times $t_n \nearrow T_*(x)$, there exists a subsequence $(t_{n_k})$ and elements $g_k \in G$ such that $(g_k \cdot u(t_{n_k}))$ converges **strongly** in the topology of $X$ to a **single** limit profile $V \in X$.

When $G$ is trivial, this reduces to ordinary precompactness of bounded-energy trajectory tails.

**Fallback (Mode D.D: Dispersion/Global Existence).** If Axiom C fails (energy disperses), there is **no finite-time singularity**—the solution exists globally via scattering (dispersion). This is not a failure mode but **global existence**: energy disperses, no concentration occurs, and no singularity forms.

**Role in constraint class.** Axiom C embodies the forced structure principle: finite-time blow-up *requires* energy concentration, and concentration *forces* the emergence of canonical profiles. The mechanism is:

1. **Finite-time blow-up requires concentration.** To form a singularity at $T_* < \infty$, energy must concentrate—otherwise the solution disperses globally and no singularity forms.
2. **Concentration forces local structure.** Wherever energy concentrates, a canonical profile $V$ emerges. Axiom C holds locally at any blow-up locus.
3. **No concentration = no singularity.** If Axiom C fails (energy disperses), there is no finite-time singularity—the solution exists globally via scattering (Mode D.D).

Consequently:
- **Mode D.D is not a singularity.** It represents global existence via dispersion, not a "failure mode."
- **Modes S.E–S.D require Axiom C to hold** (structure exists), then test whether the structure satisfies algebraic permits.
- **No global compactness proof is needed.** We observe that blow-up forces local compactness, then check permits on the forced structure.

**Remark 3.9 (Strong convergence is forced, not assumed).** The requirement of strong convergence is not an assumption to verify—it is a *consequence* of energy concentration. If a sequence converges only weakly ($u(t_n) \rightharpoonup V$) with energy loss ($\Phi(u(t_n)) \not\to \Phi(V)$), then energy has dispersed to dust, no true concentration occurred, and no finite-time singularity forms. This is Mode D.D: global existence via scattering.

**Definition 3.10 (Modulus of compactness).** The **modulus of compactness** along a trajectory $u(t)$ with $\sup_t \Phi(u(t)) \leq E$ is:
$$
\omega_C(\varepsilon, u) := \min\left\{N \in \mathbb{N} : \{u(t) : t < T_*(x)\} \subseteq \bigcup_{i=1}^N g_i \cdot B(x_i, \varepsilon) \text{ for some } g_i \in G, x_i \in X\right\}.
$$
Axiom C holds along a trajectory iff $\omega_C(\varepsilon, u) < \infty$ for all $\varepsilon > 0$.

**Remark 3.11.** In the PDE context, concentration behavior is typically described by:
- Rellich–Kondrachov compactness for Sobolev embeddings,
- Aubin–Lions lemma for parabolic regularity,
- Concentration-compactness à la Lions for critical problems \cite{Lions84},
- Profile decomposition à la Gérard–Bahouri–Chemin for dispersive equations.

#### Axiom SC (Scaling)

The Scaling Structure axiom provides the minimal geometric data needed to derive normalization constraints from scaling arithmetic alone. It applies **on orbits where the scaling subgroup acts**.

**Definition 3.12 (Scaling subgroup).** A **scaling subgroup** is a one-parameter subgroup $(\mathcal{S}_\lambda)_{\lambda > 0} \subset G$ of the symmetry group, with $\mathcal{S}_1 = e$ and $\mathcal{S}_\lambda \circ \mathcal{S}_\mu = \mathcal{S}_{\lambda\mu}$.

**Definition 3.13 (Scaling exponents).** The **scaling exponents** along an orbit where $(\mathcal{S}_\lambda)$ acts are constants $\alpha > 0$ and $\beta > 0$ such that:
$$
\Phi(\mathcal{S}_\lambda \cdot x) = \lambda^\alpha \Phi(x), \quad t \mapsto S_{\lambda^\beta t}(\mathcal{S}_\lambda \cdot x) = \mathcal{S}_\lambda \cdot S_t(x).
$$

**Axiom SC (Scaling Structure on orbits).** On any orbit where the scaling subgroup $(\mathcal{S}_\lambda)_{\lambda > 0}$ acts with well-defined scaling exponents $(\alpha, \beta)$, the **subcritical dissipation condition** holds:
$$
\alpha > \beta.
$$

**Fallback (Mode S.E: Supercritical Symmetry Cascade).** When Axiom SC fails along a trajectory—either because no scaling subgroup acts, or the subcritical condition $\alpha > \beta$ is violated—the trajectory may exhibit **supercritical symmetry cascade**. Type II (self-similar) blow-up becomes possible; normalization constraints cannot exclude it.

**Role in constraint class.** Axiom SC encodes the dimensional balance of the system. The exponent $\alpha$ governs how energy scales under dilation; $\beta$ governs how time scales. The condition $\alpha > \beta$ ensures that dissipation scales faster than time on self-similar orbits, rendering Type II blow-up impossible for finite-cost trajectories. This is a consequence of pure scaling arithmetic—no additional regularity assumptions are needed.

**Remark 3.14 (Scaling structure is soft).** For most systems of interest, the scaling structure is immediate from dimensional analysis:
- For the heat equation: $\alpha = 2$, $\beta = 2$ (critical).
- For the nonlinear Schrödinger equation: $\alpha = d/2 - 1/p$, $\beta = 2/p$ (supercritical when $\alpha < \beta$).
- For the Navier–Stokes equation in 3D: $\alpha = 1$, $\beta = 2$ (supercritical).

**Remark 3.15 (Connection to Property GN).** Under Axiom SC, Property GN (Generic Normalization) becomes a derived consequence rather than an independent axiom. Any would-be Type II blow-up profile, when viewed in normalized coordinates, has infinite dissipation. Thus such profiles cannot arise from finite-cost trajectories.

### 3.4 Symmetry constraints

These axioms enforce local rigidity near equilibria—the stiffness that drives convergence. The connection between critical point structure and global topology is formalized by **Morse theory** \cite{Morse34}: the number and types of critical points of a height functional constrain the topology of the underlying manifold.

#### Axiom LS (Local Stiffness)

**Axiom LS (Local stiffness / Łojasiewicz–Simon inequality \cite{Simon83}).** In a neighbourhood of the safe manifold, there exist:

- a closed subset $M \subseteq X$ called the **safe manifold** (the set of equilibria, ground states, or canonical patterns),
- an open neighbourhood $U \supseteq M$,
- constants $\theta \in (0, 1]$ and $C_{\mathrm{LS}} > 0$,

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

**Fallback (Mode S.D: Stiffness Breakdown).** Axiom LS is **local by design**: it applies only in the neighbourhood $U$ of $M$. A trajectory exhibits **stiffness breakdown** if any of the following occur:
- The trajectory approaches the boundary of $U$ without converging to $M$,
- The Łojasiewicz inequality (condition 2) fails,
- The drift domination (condition 3) fails—i.e., drift pushes the trajectory away from $M$ despite being inside $U$.

Outside $U$, other axioms (C, D, Rec) govern behaviour.

**Role in constraint class.** Axiom LS provides local rigidity near equilibria. The Łojasiewicz–Simon inequality quantifies the "steepness" of the energy landscape near $M$: the exponent $\theta$ controls how degenerate the energy is at equilibria. When $\theta = 1$, this is a linear coercivity condition; smaller values indicate stronger degeneracy. The drift domination ensures that trajectories inside $U$ are inexorably pulled toward $M$ by dissipation. This formalizes the concept of **Inertial Manifolds** in infinite-dimensional dynamical systems \cite{Temam88}, which contain the global attractor and capture the long-time dynamics of dissipative PDEs.

**Remark 3.16.** The exponent $\theta$ is called the **Łojasiewicz exponent**. It determines the rate of convergence to equilibrium.

**Remark 3.16b (The Spectral-Łojasiewicz Correspondence).**
While Axiom LS is formulated generally via the Łojasiewicz inequality $\|\nabla \Phi(u)\| \geq C|\Phi(u) - \Phi_\infty|^\theta$, this geometric condition encodes the spectral properties of the linearized operator $L = \nabla^2 \Phi(u_\infty)$. The exponent $\theta$ classifies the physical nature of the stability:

1. **The Mass Gap Case ($\theta = 1/2$):**
   If $\theta = 1/2$, the inequality is equivalent to **Strict Convexity** of the height functional near the equilibrium.
   - *Dynamics:* Exponential decay to equilibrium ($e^{-\lambda t}$).
   - *Physics:* This corresponds to a **Mass Gap** (strictly positive spectrum, $\lambda_1 > 0$). The potential well is quadratic.
   - *Example:* Gauge theories with confinement, damped harmonic oscillator.

2. **The Degenerate Case ($\theta \in (0, 1/2)$):**
   If $\theta < 1/2$, the potential well is "flat" at the bottom (e.g., quartic potential $x^4$ where $\theta = 1/4$).
   - *Dynamics:* Polynomial decay ($t^{-p}$).
   - *Physics:* This corresponds to **gapless modes** or critical slowing down (zero eigenvalue, $\lambda_1 = 0$), but where non-linear terms still enforce stability.
   - *Example:* Critical phase transitions, certain reaction-diffusion systems.

**Proposition 3.16c (Spectral-Łojasiewicz Equivalence).** $\theta = 1/2 \iff \lambda_1 > 0$ (mass gap).

*Proof.*

**Step 1 ($\Rightarrow$).** Suppose $\theta = 1/2$. Near equilibrium $u_\infty$, expand
$$
\Phi(u) = \Phi(u_\infty) + \frac{1}{2}\langle L(u - u_\infty), u - u_\infty \rangle + O(\|u - u_\infty\|^3).
$$
The Łojasiewicz inequality $\|\nabla \Phi\| \geq C|\Phi - \Phi_\infty|^{1/2}$ implies $\|L(u - u_\infty)\| \geq C'\|u - u_\infty\|$, hence $L \succ \lambda_1 I$ with $\lambda_1 > 0$.

**Step 2 ($\Leftarrow$).** Suppose $L \succ \lambda_1 I$. Then $\Phi(u) - \Phi_\infty \geq \frac{\lambda_1}{2}\|u - u_\infty\|^2$ and $\|\nabla \Phi(u)\| = \|L(u - u_\infty)\| \geq \lambda_1\|u - u_\infty\|$. Combining:
$$
\|\nabla \Phi\| \geq \lambda_1 \cdot \sqrt{\frac{2}{\lambda_1}} \cdot |\Phi - \Phi_\infty|^{1/2} = \sqrt{2\lambda_1} \cdot |\Phi - \Phi_\infty|^{1/2}.
$$
This is the Łojasiewicz inequality with $\theta = 1/2$. $\square$

**Corollary 3.16d (Mass Gap Detection).** Axiom LS provides the **geometric measuring stick** for the mass gap: proving a mass gap (as in gauge theories) is equivalent to proving Axiom LS holds with the specific exponent $\theta = 1/2$.

**Remark 3.16e (Hessian Positivity).** When $\Phi$ is $C^2$, the mass gap condition is equivalent to:
$$
\nabla^2 \Phi(x^*)|_{(T_{x^*}(G \cdot x^*))^\perp} \succ \lambda_{\text{gap}} \cdot I
$$
i.e., **Hessian positivity orthogonal to symmetry directions**. The Łojasiewicz formulation is more general: it applies even when $\Phi$ is not $C^2$, or when the landscape has degenerate directions beyond symmetries.

**Definition 3.17 (Log-Sobolev inequality).** In the probabilistic setting with invariant measure $\mu$ supported near $M$, we say a **log-Sobolev inequality (LSI)** holds with constant $\lambda_{\mathrm{LS}} > 0$ if for all smooth $f: X \to \mathbb{R}$ with $\int f^2 \, d\mu = 1$:
$$
\mathrm{Ent}_\mu(f^2) := \int f^2 \log f^2 \, d\mu \leq \frac{1}{2\lambda_{\mathrm{LS}}} \int |\nabla f|^2 \, d\mu.
$$

#### Axiom Reg (Regularity)

**Axiom Reg (Regularity).** The following regularity conditions hold:

1. **Semiflow continuity:** The map $(t, x) \mapsto S_t x$ is continuous on $\{(t, x) : 0 \leq t < T_*(x)\}$.
2. **Measurability:** The functionals $\Phi$, $\mathfrak{D}$, $c$, $\mathcal{R}$ are Borel measurable.
3. **Local boundedness:** On each energy sublevel $K_E$, the functionals $\mathfrak{D}$, $c$, $\mathcal{R}$ are locally bounded.
4. **Blow-up time semicontinuity:** The function $T_*: X \to (0, \infty]$ is lower semicontinuous:
$$
x_n \to x \implies T_*(x) \leq \liminf_{n \to \infty} T_*(x_n).
$$

**Fallback.** Axiom Reg is a minimal technical assumption. When it fails, the framework does not apply—the system lacks the basic regularity needed for a well-posed dynamical problem.

**Role in constraint class.** Axiom Reg provides the minimal regularity infrastructure for the framework to function. It ensures that trajectories are well-defined, functionals are measurable, and blow-up times behave semicontinuously. These are not deep constraints but basic requirements for the category-theoretic formulation.

### 3.5 Axiom interdependencies

The axioms are not independent. The relationships are:

**Proposition 3.18 (Implications).**

1. (D) + (Reg) $\implies$ sublevel sets are forward-invariant up to drift.
2. (C) + (D) + (Reg) $\implies$ existence of limit points along trajectories.
3. (C) + (D) + (LS) + (Reg) $\implies$ convergence to $M$ for bounded trajectories.
4. (Rec) + (Cap) $\implies$ quantitative control on time in bad regions.
5. (D) + (SC) $\implies$ Property GN (Generic Normalization) holds as a theorem, not an axiom.
6. (D) + (LS) + (GC) $\implies$ The Lyapunov functional $\mathcal{L}$ is explicitly reconstructible from dissipation data alone.

Here (GC) is Axiom GC (Gradient Consistency), which applies to gradient flow systems and enables explicit reconstruction of the Lyapunov functional via the Jacobi metric or Hamilton–Jacobi equation.

**Proposition 3.19 (Minimal axiom sets).** The main theorems require the following minimal axiom combinations:

- **Structural Resolution Theorem:** (C), (D), (Reg)
- **GN as Metatheorem:** (D), (SC)
- **Type II Exclusion Theorem:** (D), (SC)
- **Capacity Barrier Theorem:** (Cap), (BG)
- **Topological Suppression Theorem:** (TB), (LSI)
- **Dichotomy Theorem:** (D), (R), (Cap)
- **Canonical Lyapunov Theorem:** (C), (D), (R), (LS), (Reg)
- **Action Reconstruction Theorem:** (D), (LS), (GC)
- **Hamilton–Jacobi Generator Theorem:** (D), (LS), (GC)

Here (BG) is the Background Geometry axiom (providing geometric structure via Hausdorff measure), (LSI) is the Log-Sobolev Inequality, and (GC) is Gradient Consistency.

**Proposition 3.20 (The mode classification).** The Structural Resolution Theorem classifies trajectories based on which condition fails:

| Condition | Mode | Description |
|-----------|------|-------------|
| **C fails** (No concentration) | Mode D.D | **Dispersion (Global existence):** Energy disperses, no singularity forms, solution scatters globally |
| **D fails** (Energy unbounded) | Mode C.E | **Energy blow-up:** Energy grows without bound as $t \nearrow T_*(x)$ |
| **Rec fails** (No recovery) | Mode C.E | **Energy blow-up:** Trajectory drifts indefinitely in bad region |
| **SC fails** (Scaling permit denied) | Mode S.E | **Supercritical impossible:** Scaling exponents violate $\alpha > \beta$; blow-up contradicted |
| **Cap fails** (Capacity permit denied) | Mode C.D | **Geometric collapse impossible:** Concentration on capacity-zero sets contradicted |
| **TB fails** (Topological permit denied) | Mode T.E | **Topological obstruction:** Background invariants block the singularity |
| **LS fails** (Stiffness permit denied) | Mode S.D | **Stiffness breakdown impossible:** Łojasiewicz inequality contradicts stagnation |
| **GC fails** | — | Reconstruction theorems do not apply; abstract Lyapunov construction still valid |

**Remark 3.21 (Regularity via permit denial).** Global regularity follows whenever:

1. Energy disperses (Mode D.D)—no singularity forms, or
2. Concentration occurs but a permit is denied—singularity is contradicted.

When a local axiom fails, the resolution identifies which mode of singular behavior occurs, providing a complete classification even for trajectories that escape the "good" regime.

**Remark 3.22 (Constraint class organization).** The axioms are organized into four constraint classes:

1. **Conservation (D, Rec):** Thermodynamic balance—energy, dissipation, and recovery.
2. **Topology (TB, Cap):** Spatial structure—topological sectors and geometric capacity.
3. **Duality (C, SC):** Self-similar structure—compactness modulo symmetries and scaling balance.
4. **Symmetry (LS, Reg):** Local rigidity—stiffness near equilibria and minimal regularity.

Each class addresses a different aspect of system behavior. Together, they provide a complete classification of dynamical breakdown modes.

---

---

# Part III: The Failure Taxonomy

## 4. The Complete Classification of Dynamical Failure

### 4.1 The structural definition of singularity

In classical analysis, a singularity is often defined negatively—as a point where regularity is lost. In the hypostructure framework, we define it positively as a specific dynamical event where the trajectory attempts to exit the admissible state space.

Let $\mathcal{S} = (X, (S_t), \Phi, \mathfrak{D})$ be a structural flow datum. Let $u(t) = S_t x$ be a trajectory defined on a maximal interval $[0, T_*)$.

**Definition 4.1 (Singularity).** A trajectory $u(t)$ exhibits a **singularity** at $T_* < \infty$ if it cannot be extended beyond $T_*$ within the topology of $X$, despite satisfying the energy constraint $\Phi(u(0)) < \infty$.

The central thesis of this framework is that singularities are not random chaotic events, but are **isomorphic to the failure of specific structural axioms**. The axioms form a diagnostic system. By determining exactly *which* axiom fails along a singular sequence, we classify the breakdown into one of fifteen mutually exclusive modes.

**The Fixed-Point Principle.** The axioms are not arbitrary choices but manifestations of a single organizing principle: **self-consistency under evolution**. A system satisfies the hypostructure axioms if and only if its persistent states are fixed points of the evolution operator: $F(x) = x$. Singularities represent states where $F(x) \neq x$ in the limit—the evolution attempts to produce a state incompatible with its own definition.

---

### 4.2 The taxonomy of failure modes

The fifteen failure modes decompose according to four fundamental constraint classes, each enforcing a distinct aspect of self-consistency. This decomposition reflects the mathematical structure of the obstruction space.

**Definition 4.2 (Constraint classification).** The structural constraints divide into four orthogonal classes:

| **Class** | **Enforces** | **Axioms** |
|-----------|--------------|------------|
| **Conservation** | Magnitude bounds | D, R, Cap |
| **Topology** | Connectivity | TB, Cap |
| **Duality** | Perspective coherence | C, SC |
| **Symmetry** | Cost structure | SC, LS, GC |

Each class admits three failure types: **Excess** (too much structure), **Deficiency** (too little structure), and **Complexity** (bounded but inaccessible structure). For open systems coupled to an environment, three additional **Boundary** failure modes emerge.

**Table 4.3 (The Taxonomy of Failure Modes).**

| **Constraint** | **Excess** | **Deficiency** | **Complexity** |
|----------------|------------|----------------|----------------|
| **Conservation** | Mode C.E: Energy blow-up | Mode C.D: Geometric collapse | Mode C.C: Finite-time event accumulation |
| **Topology** | Mode T.E: Metastasis | Mode T.D: Glassy freeze | Mode T.C: Labyrinthine |
| **Duality** | Mode D.E: Oscillatory | Mode D.D: Dispersion | Mode D.C: Semantic horizon |
| **Symmetry** | Mode S.E: Supercritical | Mode S.D: Stiffness breakdown | Mode S.C: Parameter manifold instability |
| **Boundary** | Mode B.E: Injection | Mode B.D: Starvation | Mode B.C: Misalignment |

**Metatheorem 4.4 (Completeness).** The fifteen modes form a complete classification of dynamical failure. Every trajectory of a hypostructure (open or closed) either:

1. Exists globally and converges to the safe manifold $M$, or
2. Exhibits exactly one of the failure modes 1–15.

*Proof.* The four constraint classes are orthogonal by construction. Each class admits three failure types corresponding to the logical possibilities for constraint violation. The boundary class adds three modes for open systems. The $4 \times 3 + 3 = 15$ modes exhaust the logical space. $\square$

---

### 4.2.1 Diagnosis of Genuine Singularities (Sieve Calibration)

A natural skeptical question arises: *Does this framework ever predict a singularity correctly, or does it define them away?* This section demonstrates that the Sieve is a **discriminator**, not a rubber stamp for regularity. We verify that the framework correctly identifies systems known to form singularities by showing which axioms fail.

**Metatheorem 4.4.1 (Sieve Discrimination).** The hypostructure Sieve is falsifiable: there exist physically meaningful dynamical systems for which the Sieve correctly predicts singularity formation by identifying axiom violations.

*Proof.* We exhibit three canonical examples where the Sieve correctly diagnoses singularities.

#### Example 4.4.2: Euler Equations (Inviscid Fluids)

Consider the incompressible Euler equations in $\mathbb{R}^3$:
$$
\partial_t u + (u \cdot \nabla)u = -\nabla p, \quad \nabla \cdot u = 0, \quad \nu = 0.
$$

**Hypostructure data:**
- State space: $X = L^2_\sigma(\mathbb{R}^3)$ (divergence-free vector fields)
- Height functional: $\Phi(u) = \frac{1}{2}\|u\|_{L^2}^2$ (kinetic energy)
- Dissipation functional: $\mathfrak{D}(u) = \nu\|\nabla u\|_{L^2}^2 = 0$ (no viscosity)

**Axiom Check:** Axiom D requires $\mathfrak{D} > 0$ for energy decay. Here $\mathfrak{D} \equiv 0$.

**Sieve Verdict:** Axiom D is **violated**. Mode C.E (Energy accumulation) or Mode D.D (Dispersion without dissipation) is permitted. The Sieve **does not predict regularity**.

**Reality:** Elgindi \cite{Elgindi21} proved finite-time singularity formation for smooth solutions to Euler in $\mathbb{R}^3$. The framework correctly identifies the missing dissipative mechanism.

#### Example 4.4.3: Supercritical Nonlinear Schrödinger Equation

Consider the focusing NLS in dimension $d \geq 3$ with cubic nonlinearity:
$$
i\partial_t \psi + \Delta \psi + |\psi|^2 \psi = 0.
$$

**Hypostructure data:**
- State space: $X = H^1(\mathbb{R}^d)$
- Height functional: $\Phi(\psi) = \|\nabla \psi\|_{L^2}^2$ (gradient energy)
- Scaling exponents: $\alpha = d/2 - 1$, $\beta = 2$ (from dimensional analysis)

**Axiom Check:** Axiom SC requires $\alpha > \beta$ (subcritical). For $d = 3$: $\alpha = 1/2 < 2 = \beta$ (**supercritical**).

**Sieve Verdict:** Axiom SC is **violated**. Mode S.E (Supercritical Cascade) is permitted. The Sieve **does not predict regularity**.

**Reality:** Supercritical focusing NLS is known to exhibit finite-time blow-up \cite{Sulem99}. Energy can concentrate without infinite dissipation cost. The framework correctly identifies the scaling obstruction.

#### Example 4.4.4: Ricci Flow in Dimension 4

Consider Ricci flow on a compact 4-manifold:
$$
\partial_t g = -2\text{Ric}(g).
$$

**Hypostructure data:**
- State space: $X = \text{Met}(M^4)/\text{Diff}(M^4)$ (metrics modulo diffeomorphisms)
- Height functional: $\Phi(g) = \int_M |Rm|^2 \, dV_g$ (curvature energy)
- Capacity constraint: Singularities must have codimension $\geq 4$ for surgical removal

**Axiom Check:** Axiom Cap requires singular sets to have sufficiently high codimension. In 4D, singularities can form on codimension-2 sets (surfaces, filaments).

**Sieve Verdict:** Axiom Cap is **violated** (capacity bound fails). Mode T.E (Topological Metastasis) is possible. The Sieve **does not predict regularity**.

**Reality:** 4D Ricci flow can form singularities along 2-dimensional surfaces that cannot be surgically resolved \cite{Hamilton97}. Unlike 3D (where Perelman's surgery works), 4D lacks the capacity margin. The framework correctly identifies the dimensional obstruction.

**Conclusion.** The Sieve has **teeth**: it correctly predicts singularity formation in all three canonical examples by identifying the specific axiom that fails. This demonstrates that regularity predictions are non-trivial—they depend on verifiable structural properties. $\square$

---

### 4.3 Conservation failures (Modes C.E, C.D, C.C)

**Conservation constraints enforce information invariance:** phase space volume is preserved under reversible evolution, and total information content is bounded. Violations occur when magnitudes escape their permitted bounds.

#### Mode C.E: Energy blow-up

**Axiom Violated:** **(D) Dissipation**

**Diagnostic Test:**
$$
\limsup_{t \nearrow T_*} \Phi(u(t)) = \infty
$$

**Structural Mechanism:** The dissipative power $\mathfrak{D}$ is insufficient to counteract the drift or forcing terms in the energy inequality. The trajectory escapes every compact sublevel set $K_E$. The system exits the state space because the height functional becomes infinite.

**Status:** The singularity is detected purely by scalar estimates; no geometric analysis of the state $u(t)$ is required. This is a **genuine singularity**.

**Remark 4.5 (Mode C.E is the universal energy catch-all).** If $\limsup_{t \to T_*} \Phi(u(t)) = \infty$, the trajectory is classified as **Mode C.E**, regardless of the mechanism:
- Energy growth due to drift outside the good region $\mathcal{G}$,
- Energy growth due to drift inside $\mathcal{G}$ (if the "good region" drift bound fails),
- Energy growth due to any other cause.

This ensures no trajectory with unbounded energy escapes classification. The distinction between "controlled" and "uncontrolled" drift is irrelevant for Mode C.E—what matters is the scalar diagnostic $\limsup \Phi = \infty$.

#### Mode C.D: Geometric collapse

**Axiom Violated:** **(Cap) Capacity**

**Diagnostic Test:** The limiting probability measure or occupation time concentrates on a set $E \subset X$ with vanishing capacity or effective dimension lower than required for regularity:
$$
\limsup_{t \nearrow T_*} \frac{\mathrm{Leb}\{s \in [0,t] : u(s) \in B_\epsilon\}}{\mathrm{Cap}(B_\epsilon)} = \infty
$$
where $B_\epsilon$ are neighborhoods of a capacity-zero set.

**Structural Mechanism:** The trajectory spends a disproportionate amount of time in "thin" regions of the state space relative to the dissipation budget available. Energy remains bounded ($\sup_{t < T_*} \Phi(u(t)) < \infty$) but collapses onto a geometric singularity of insufficient dimension.

**Status:** Dimensional collapse (e.g., formation of defect sets of codimension $\geq 2$). This is a **genuine singularity**.

**Example 4.6.** In Navier–Stokes, this corresponds to vortex filaments collapsing to curves or points. In geometric flows, this is concentration on lower-dimensional manifolds.

#### Mode C.C: Finite-time event accumulation

**Axiom Violated:** Conservation (causal depth) / **(Rec) Recovery**

**Diagnostic Test:** The trajectory executes infinitely many discrete events in finite time:
$$
\#\{t_i \in [0, T_*) : u(t_i) \in \partial \mathcal{G}\} = \infty
$$

**Structural Mechanism:** The system undergoes an accumulation of transitions between regions, each costing finite energy but summing to finite total cost. The causal depth (number of logical steps) becomes infinite while physical time remains finite. This violates the assumption that recovery from the bad region occurs in bounded time.

**Status:** A **complexity failure**. Energy and spatial structure remain bounded, but the trajectory becomes causally dense—infinite logical depth in finite time.

**Metatheorem 4.7 (Causal Barrier).** Under Axiom D with $\alpha > 0$, Mode C.C requires $\mathcal{C}_*(x) = \infty$. For finite-cost trajectories, only finitely many discrete transitions occur.

*Proof.* Each transition dissipates at least $\delta > 0$ energy (by Axiom Rec). The total dissipation bound
$$
\int_0^{T_*} \mathfrak{D}(u(t)) \, dt \leq \frac{1}{\alpha}\Phi(u(0)) + C_0 \cdot \tau_{\mathrm{bad}} < \infty
$$
implies finitely many transitions. If infinitely many transitions occur, the cumulative dissipation diverges, contradicting bounded energy. $\square$

**Example 4.8.** A bouncing ball with coefficient of restitution $e < 1$ completes infinitely many bounces in finite time $T_* = \frac{2v_0}{g(1-e)}$. Each bounce dissipates energy $E_n = E_0 e^{2n}$, forming a convergent geometric series.

---

### 4.4 Topology failures (Modes T.E, T.D, T.C)

**Topological constraints enforce local-global consistency:** local solutions extend to global solutions when topological obstructions vanish. Violations occur when connectivity is disrupted.

#### Mode T.E: Topological sector transition

**Axiom Violated:** **(TB) Topological Background**

**Diagnostic Test:** The limiting profile $v = \lim u(t_n)$ resides in a topological sector $\tau(v)$ distinct from the initial sector $\tau(u(0))$, or the limit is obstructed by an action gap:
$$
\Phi(v) < \mathcal{A}_{\min}(\tau(u(0)))
$$

**Structural Mechanism:** The trajectory is energetically or geometrically forced into a configuration forbidden by the topological invariants of the flow, necessitating a discontinuity to resolve the sector index. Energy concentrates but cannot form a smooth limiting configuration within the accessible topological class.

**Status:** Phase slips or discrete topological transitions. This is a **genuine singularity** involving topology change.

**Proposition 4.9 (Cohomological barrier).** Let $\mathcal{S}$ be a hypostructure with topological background $\tau: X \to \mathcal{T}$. A local solution $u: U \to X$ extends globally if and only if the obstruction class $[\omega_u] \in H^1(X; \mathcal{T})$ vanishes.

*Proof.*

**Step 1 (Presheaf of local solutions).** Define the presheaf $\mathcal{S}$ on $X$ by assigning to each open set $U \subseteq X$ the set $\mathcal{S}(U)$ of local solutions $u: U \to X$ satisfying the flow equations. Restriction maps are the natural restrictions.

**Step 2 (Čech cohomology construction).** Given a local solution $u: U \to X$, cover $X$ by open sets $\{U_\alpha\}$ on which $u|_{U_\alpha}$ extends. On double overlaps $U_\alpha \cap U_\beta$, the two extensions $u_\alpha$ and $u_\beta$ differ by a gauge transformation $g_{\alpha\beta} \in G$ acting on the topological data. These transition functions form a Čech 1-cocycle $\{g_{\alpha\beta}\}$.

**Step 3 (Obstruction class).** The obstruction class $[\omega_u] \in H^1(X; \mathcal{T})$ is the cohomology class of this cocycle. It measures the failure of the local extensions to patch consistently.

**Step 4 (Vanishing implies extension).** If $[\omega_u] = 0$, then $\{g_{\alpha\beta}\} = \delta \{h_\alpha\}$ for some 0-cochain $\{h_\alpha\}$. Adjusting $u_\alpha \mapsto h_\alpha \cdot u_\alpha$ makes the extensions compatible on overlaps, yielding a global solution.

**Step 5 (Non-vanishing implies obstruction).** Conversely, if $[\omega_u] \neq 0$, no adjustment of local extensions can make them compatible—the topological twist is intrinsic. $\square$

**Example 4.10.** In superconductivity, phase slips occur when the order parameter $\psi = |\psi|e^{i\theta}$ attempts to pass through zero, forcing a discontinuous jump in the phase $\theta$. In Yang–Mills, this corresponds to crossing between topological sectors separated by action barriers.

#### Mode T.D: Glassy freeze

**Axiom Violated:** Topology (ergodicity)

**Diagnostic Test:** The trajectory becomes trapped in a metastable state $x^* \notin M$ with $\mathrm{dist}(x^*, M) > \delta > 0$ for all $t > T_0$.

**Structural Mechanism:** The energy landscape contains local minima separated from the global minimum by barriers exceeding the available kinetic energy. The trajectory satisfies $\frac{d}{dt}\Phi(u(t)) \leq 0$ but cannot cross the barrier to reach $M$. This represents **frustration** or **jamming** in the topological structure.

**Status:** A **complexity failure**. The trajectory remains bounded but becomes trapped in a metastable configuration, neither dispersing nor reaching equilibrium. This is **not a singularity** but a failure of global convergence.

**Proposition 4.11.** Mode T.D occurs when the energy landscape has local minima separated from the global minimum by barriers exceeding the available kinetic energy.

*Proof.*

**Step 1 (Local minimum characterization).** Suppose $x^*$ is a local minimum with $\nabla \Phi(x^*) = 0$, $\nabla^2 \Phi(x^*) \geq 0$ (positive semidefinite Hessian), but $x^* \notin M$ (not a global minimum).

**Step 2 (Basin of attraction).** Define the basin $B(x^*) := \{x \in X : \lim_{t \to \infty} S_t x = x^*\}$. By Axiom D, trajectories starting in $B(x^*)$ descend toward $x^*$ monotonically in $\Phi$.

**Step 3 (Barrier definition).** Let $\Delta \Phi := \inf_{\gamma: x^* \leadsto M} \max_{s} \Phi(\gamma(s)) - \Phi(x^*)$ be the minimal barrier height, where the infimum is over continuous paths from $x^*$ to $M$.

**Step 4 (Trapping condition).** If the trajectory starts with $\Phi(u(0)) < \Phi(x^*) + \Delta \Phi$, then by energy monotonicity (Axiom D), $\Phi(u(t)) \leq \Phi(u(0))$ for all $t$. The trajectory cannot cross the barrier $\Delta \Phi$ and remains trapped in $B(x^*)$.

**Step 5 (Convergence to local minimum).** By Axiom LS (Łojasiewicz), the trajectory converges to a critical point. Since it cannot escape $B(x^*)$, it converges to $x^*$, realizing Mode T.D. $\square$

**Remark 4.12.** Spin glasses, protein folding, and NP-hard optimization landscapes exhibit Mode T.D behavior. The near-decomposability principle (Theorem 9.202) characterizes when this mode is avoided—systems with hierarchical block structure allow gradual relaxation without freezing.

**Example 4.13.** In the $p$-spin glass model, the energy landscape becomes ultra-metric at low temperatures, with exponentially many metastable states separated by diverging barriers.

#### Mode T.C: Labyrinthine singularity

**Axiom Violated:** **(TB) Topological Background** (tameness)

**Diagnostic Test:** The topological complexity diverges:
$$
\limsup_{t \nearrow T_*} \sum_{k=0}^n b_k(u(t)) = \infty,
$$
where $b_k$ denotes the $k$-th Betti number (rank of the $k$-th homology group).

**Structural Mechanism:** The trajectory develops **wild topology**—infinite-dimensional homology or non-locally-finite structure. Energy remains bounded, concentration may or may not occur, but the topological type becomes infinitely complex. The configuration space becomes a labyrinth with unbounded topological features.

**Status:** A **complexity failure**. This is a **genuine singularity** involving unbounded topological invariants.

**Metatheorem 4.14 (O-Minimal Taming).** If the dynamics are definable in an o-minimal structure (e.g., generated by algebraic or analytic functions), then Mode T.C is excluded.

*Proof.*

**Step 1 (O-minimal definition).** An o-minimal structure on $\mathbb{R}$ is an expansion of the ordered field $(\mathbb{R}, <, +, \cdot)$ such that every definable subset of $\mathbb{R}$ is a finite union of points and intervals. The foundational result is the **Tarski-Seidenberg theorem** \cite{Tarski51}: the real field admits **quantifier elimination**, and this extends to all o-minimal structures.

**Step 2 (Cell decomposition theorem).** By the fundamental theorem of o-minimality (van den Dries), every definable set $S \subseteq \mathbb{R}^n$ admits a **cell decomposition**: a finite partition into cells homeomorphic to open balls of various dimensions.

**Step 3 (Finite Betti numbers).** A finite cell decomposition implies:
$$b_k(S) \leq \#\{\text{$k$-cells in decomposition}\} < \infty$$
for all $k$. The total topological complexity $\sum_k b_k(S)$ is bounded by the cell count.

**Step 4 (Application to trajectories).** If the trajectory $u(t)$ evolves via dynamics definable in an o-minimal structure, then for each $t$, the configuration $u(t)$ lies in a definable family. By uniform cell decomposition, the Betti numbers remain uniformly bounded.

**Step 5 (Mode T.C exclusion).** Mode T.C requires $\limsup_{t \nearrow T_*} \sum_k b_k(u(t)) = \infty$. By Step 4, this divergence is impossible in definable dynamics. Wild topology (infinite Betti numbers) requires operations outside o-minimal structures—limiting processes with infinitely many components, Cantor-type constructions, or non-analytic singularities. $\square$

**Example 4.15.** The Alexander horned sphere is a wild embedding of $S^2$ in $\mathbb{R}^3$ that is not ambient isotopic to the standard sphere. Such pathologies are excluded by o-minimality. Fluid interfaces governed by analytic PDEs cannot develop Alexander horns.

**Remark 4.16.** Mode T.C represents failure of the **tame topology assumption**. In practice, most physical systems satisfy tameness due to analyticity or algebraic constraints. Mode T.C is primarily a logical possibility rather than a physical obstruction.

---

### 4.5 Duality failures (Modes D.D, D.E, D.C)

**Duality constraints enforce perspective coherence:** a state $x$ and its dual representation $x^*$ (under Fourier, Legendre, or other natural pairings) are related by bounded transformations. Violations occur when dual descriptions become incompatible.

#### Mode D.D: Dispersion (Global Existence)

**Axiom Violated:** **(C) Compactness** fails—energy does not concentrate

**Diagnostic Test:** There exists a sequence $t_n \nearrow T_*$ such that the orbit sequence $\{u(t_n)\}$ admits **no strongly convergent subsequence** in $X$ modulo the symmetry group $G$.

**Structural Mechanism:** Energy remains bounded ($\sup_{t < T_*} \Phi(u(t)) < \infty$) but does not concentrate; instead it "scatters" or disperses into modes that are invisible to the strong topology of $X$ (e.g., dispersion to spatial infinity, radiation to high frequencies). The dual representation spreads according to the uncertainty principle.

**Status:** **No finite-time singularity forms.** The solution exists globally and scatters. Mode D.D is **not a failure mode**—it is **global regularity via dispersion**.

**Remark 4.17 (Mode D.D is global existence).** Mode D.D encompasses all scenarios where energy does not concentrate into a single profile:

1. **Weak convergence without strong convergence.** If $u(t_n) \rightharpoonup V$ weakly but $\Phi(u(t_n)) \to \Phi(V) + \delta$ for some $\delta > 0$ (energy dispersing to radiation), this is Mode D.D. Energy disperses rather than concentrating—no singularity forms.
2. **Multi-profile decompositions.** If the trajectory involves multiple separating profiles (e.g., $u(t_n) \approx \sum_j g_n^j \cdot V^j$), and no single profile approximation suffices, this is Mode D.D. The profiles separate and scatter—no singularity forms.
3. **Physical interpretation.** Mode D.D corresponds to **scattering solutions**: the solution exists globally, and the energy disperses to spatial or frequency infinity. This is global regularity, not breakdown. The framework classifies this as "no structure" precisely because no singularity structure forms—the solution is globally regular.

**Proposition 4.18 (Anamorphic principle).** Let $\mathcal{F}: X \to X^*$ be the Fourier or Legendre transform appropriate to the structure. If $x$ is localized ($\|x\|_{X} < \delta$), then $\mathcal{F}(x)$ is dispersed:
$$
\|x\|_X \cdot \|\mathcal{F}(x)\|_{X^*} \geq C > 0.
$$

*Proof.*

**Step 1 (Fourier transform case).** For $x \in L^2(\mathbb{R}^d)$ with Fourier transform $\hat{x} = \mathcal{F}(x)$, the Heisenberg uncertainty principle states:
$$\left(\int |y|^2 |x(y)|^2 \, dy\right)^{1/2} \cdot \left(\int |\xi|^2 |\hat{x}(\xi)|^2 \, d\xi\right)^{1/2} \geq \frac{d}{4\pi} \|x\|_{L^2}^2.$$
This shows localization in position ($x$ concentrated near origin) forces delocalization in frequency ($\hat{x}$ spread).

**Step 2 (Legendre transform case).** For convex functions $f$ with Legendre dual $f^*(p) = \sup_x \{px - f(x)\}$, convex duality implies:
$$f(x) + f^*(p) \geq xp$$
with equality at $p = \nabla f(x)$. A steep well in $f$ (localized minimum) corresponds to a shallow dual $f^*$ (dispersed minimum).

**Step 3 (General formulation).** The constant $C > 0$ is the **uncertainty constant** of the duality pairing. It depends only on the choice of norms and the transform $\mathcal{F}$, not on the specific element $x$.

**Step 4 (Structural implication).** The anamorphic principle implies: if a problem is "stuck" in representation $X$ (concentrated in a bad region), passing to the dual $X^*$ may reveal a dispersed, tractable form. Duality changes the geometry but preserves information. $\square$

##### 4.5.1 The Scattering Barrier (Quantitative Criterion)

Mode D.D characterizes global existence via dispersion, yet a natural question arises: how does one distinguish beneficial dispersion from pathological loss of compactness? The **Scattering Barrier** provides a quantitative criterion for this distinction.

**Definition 4.19 (Interaction Functional).** For a dispersive system with state $u(t)$, the **Morawetz-type interaction functional** is defined as:

$$\mathcal{M}[u] := \int_0^\infty \int_X \frac{|u(x,t)|^{p+1}}{|x|^a} \, dx \, dt$$

where $p$ denotes the nonlinearity exponent and $a > 0$ is a weight parameter (canonically $a = 1$ for spatial dimension $d \geq 3$).

**Axiom Scat (Scattering Bound).** A hypostructure satisfies the **Scattering Axiom** if there exists a constant $C > 0$, depending only on structural parameters, such that:

$$\mathcal{M}[u] \leq C \cdot \Phi(u_0)$$

where $\Phi(u_0)$ denotes the initial energy.

**Proposition 4.20 (Discrimination of Non-Compactness Types).**

1. **(Controlled dispersion, Mode D.D):** If Axiom Scat holds, energy disperses in a controlled manner and the solution scatters: $\|u(t) - e^{it\Delta}u_+\| \to 0$ as $t \to \infty$ for some asymptotic free state $u_+ \in X$.

2. **(Pathological non-compactness, Mode C.D risk):** If Axiom Scat fails, energy may concentrate on measure-zero sets or escape to infinity in an uncontrolled fashion, potentially indicating Mode C.D (geometric collapse) rather than benign dispersion.

*Proof.*
(1) The bound $\mathcal{M}[u] < \infty$ implies space-time integrability. By standard dispersive theory via Strichartz estimates \cite{KenigMerle06}, this integrability condition is sufficient for scattering.

(2) Failure of the Morawetz bound indicates one of two pathologies: concentration (mass accumulating near the spatial origin, characteristic of blow-up) or uncontrolled escape (mass radiating to spatial infinity faster than dispersive decay permits). $\square$

**Metatheorem 4.21 (Scattering-Compactness Dichotomy).** *For systems satisfying Axioms D (Dissipation) and SC (Scaling), precisely one of the following alternatives holds:*

1. *Axiom C holds: trajectories admit convergent subsequences modulo symmetry, amenable to profile decomposition analysis*
2. *Axiom Scat holds: trajectories scatter, yielding Mode D.D (global existence)*

*Proof.* This dichotomy is the concentration-compactness alternative of Lions \cite{Lions84}. The Morawetz functional furnishes the quantitative threshold: boundedness of $\mathcal{M}[u]$ implies dispersion dominance; unboundedness implies concentration. $\square$

**Example 4.22 (Defocusing nonlinear Schrödinger equation).** For the defocusing NLS $i\partial_t u + \Delta u = |u|^{p-1}u$:

| Regime | Axiom Scat Status | Dynamical Outcome |
|:-------|:------------------|:------------------|
| $p < 1 + 4/d$ (subcritical) | Satisfied automatically | Global existence with scattering |
| $p = 1 + 4/d$ (critical) | Requires Morawetz analysis | Scattering for small data |
| $p > 1 + 4/d$ (supercritical) | May fail | Possible blow-up (Mode S.E) |

**Remark 4.23 (Structural versus quantitative scattering).** The framework classifies classical Morawetz estimates as obsolete (§21.4) in the sense that **structural scattering** (Mode D.D classification) supersedes **quantitative estimation**. The Scattering Barrier (Axiom Scat) serves a distinct purpose: it furnishes the **decidability criterion** for distinguishing Mode D.D from Mode C.D when Axiom C fails. The estimate thereby assumes the role of a structural dichotomy rather than a computational tool.

**Corollary 4.24 (Scattering permit).** *The Scattering Axiom induces a permit:*
$$\Pi_{\text{Scat}}(V) = \begin{cases} \text{GRANTED} & \text{if } \mathcal{M}[V] = \infty \text{ (unbounded interaction)} \\ \text{DENIED} & \text{if } \mathcal{M}[V] < \infty \text{ (bounded interaction)} \end{cases}$$

*When $\Pi_{\text{Scat}}(V) = \text{DENIED}$, the profile $V$ cannot concentrate and must scatter, ensuring global existence (Mode D.D).*

#### Mode D.E: Oscillatory singularity

**Axiom Violated:** Duality (derivative control)

**Diagnostic Test:** Energy remains bounded but the time derivative blows up:
$$
\sup_{t < T_*} \Phi(u(t)) < \infty \quad \text{but} \quad \limsup_{t \nearrow T_*} \|\partial_t u(t)\| = \infty.
$$

**Structural Mechanism:** The trajectory undergoes **frequency blow-up**: the amplitude remains bounded but the oscillation frequency diverges. In the dual (frequency) representation, energy migrates to arbitrarily high frequencies while remaining bounded in the physical representation. This violates the duality constraint that both representations should exhibit comparable behavior.

**Status:** An **excess failure** in the duality class. This is a **genuine singularity** of oscillatory type.

**Example 4.19.** The function $u(t) = \sin(1/(T_* - t))$ remains bounded ($|u| \leq 1$) but has unbounded frequency $\omega(t) = 1/(T_* - t)^2 \to \infty$ as $t \to T_*$.

**Metatheorem 4.20 (Frequency Barrier).** Under Axiom SC with $\alpha > \beta$, Mode D.E is excluded for gradient flows. The Bode sensitivity integral provides the quantitative bound.

*Proof.* For gradient flows, $\|\partial_t u\|^2 = \mathfrak{D}(u)$. The energy–dissipation inequality bounds the time-integral of $\mathfrak{D}$:
$$
\int_0^{T_*} \mathfrak{D}(u(t)) \, dt \leq \frac{1}{\alpha}\Phi(u(0)) < \infty.
$$
By Hölder's inequality, this prevents pointwise blow-up of $\|\partial_t u\|$ unless energy also blows up. Specifically, if $\|\partial_t u(t_n)\| \to \infty$ along a sequence $t_n \to T_*$, then the integral must diverge, contradiction. $\square$

**Remark 4.21.** Mode D.E represents a duality inversion: concentration in frequency space (high modes) corresponds to rapid oscillation in physical space. The failure occurs when this inversion becomes unbounded.

#### Mode D.C: Semantic horizon (The Cryptographic Barrier)

**Alternative name for complexity-theoretic contexts:** **The Cryptographic Barrier**

**Axiom Violated:** **(Rec) Recovery** (invertibility)

**Diagnostic Test:** The conditional Kolmogorov complexity diverges:
$$
\lim_{t \nearrow T_*} K(u(t) \mid \mathcal{O}(t)) = \infty,
$$
where $\mathcal{O}(t)$ denotes the macroscopic observables.

**Structural Mechanism:** The dynamics implement a **one-way function**: the state is well-defined and bounded, but computationally inaccessible from observations. Information becomes scrambled across exponentially many microstates, forming a **semantic horizon** beyond which the state cannot be reconstructed from observations. This represents irreversible information loss in the dual (observational) description.

**Cryptographic Interpretation:** This mode captures the structural obstruction where:
- The system state is bounded and well-defined
- The dynamics are deterministic
- But the **information required to predict or invert the outcome** grows faster than any polynomial in the input size

The barrier is not about physical resources but about **informational irreducibility**—the system performs computation that cannot be shortcut without solving the problem itself. This is the dynamical manifestation of computational hardness.

**Status:** A **complexity failure**. The trajectory remains bounded but becomes semantically inaccessible. This is **not a singularity** in the classical sense but a failure of invertibility.

**Proposition 4.22.** Mode D.C occurs when the dynamics implement a one-way function: the state is well-defined but computationally inaccessible from observations.

*Proof.*

**Step 1 (One-way function definition).** A function $f: X \to Y$ is **one-way** if:
- $f(x)$ is computable in polynomial time from $x$
- $f^{-1}(y)$ requires super-polynomial (typically exponential) time to compute from $y$

**Step 2 (Forward computability).** The dynamics $S_t: X \to X$ define the forward map $x \mapsto S_t x$. Under standard assumptions (finite propagation speed, local interactions), this map is polynomial-time computable: the state at time $t$ can be computed by local updates.

**Step 3 (Backward complexity).** The inverse problem $S_t x \mapsto x$ requires reconstructing the initial condition from the final state. When the dynamics are chaotic or mixing, this reconstruction requires exponential resources:
- The number of distinguishable microstates grows as $e^{S}$ where $S$ is entropy
- Kolmogorov complexity satisfies $K(u(0) \mid u(t)) \sim S(t)$

**Step 4 (Scrambling rate bound).** The epistemic horizon principle (Theorem 9.152) bounds the rate of information scrambling. The Lieb-Robinson velocity $v_{\mathrm{LR}}$ limits how fast correlations can spread, giving:
$$K(u(t) \mid \mathcal{O}(t)) \leq v_{\mathrm{LR}} \cdot t \cdot \log(\dim X).$$
The semantic horizon forms when this bound saturates. $\square$

**Remark 4.23.** Black hole interiors (behind the event horizon), cryptographic states, and fully-developed turbulence exhibit Mode D.C characteristics. The state exists but cannot be accessed by external observers.

*Remark 4.23.1 (Information Closure Failure).* Mode D.C admits a precise characterization via **computational closure** \cite{Rosas2024}. Let $X_t$ denote the full micro-state and $Z_t := \Pi(X_t)$ the observable (macro) variables under some coarse-graining $\Pi$. The system exhibits Mode D.C when the **closure gap** is large:
$$\delta_{\text{closure}} := 1 - \frac{I(Z_t; Z_{t+1})}{I(X_t; X_{t+1})} \to 1$$
Equivalently, $I(Z_t; Z_{t+1}) \ll I(X_t; X_{t+1})$: the macro-scale retains negligible predictive power. By the Closure-Curvature Duality (Metatheorem 20.7), this occurs if and only if the macro-level Ollivier curvature $\kappa(\tilde{T}) \to 0$. Physically, the geometric stiffness that would guarantee stable macro-dynamics has collapsed—information becomes irreversibly scrambled into correlations invisible to the observer.

**Example 4.24.** In quantum many-body systems, thermalization via eigenstate thermalization hypothesis (ETH) creates a semantic horizon: the late-time state $u(t)$ is a superposition of exponentially many eigenstates, indistinguishable from a thermal state by any local observable.

---

### 4.6 Symmetry failures (Modes S.E, S.D, S.C)

**Symmetry constraints enforce cost structure:** breaking a symmetry requires positive energy. Violations occur when symmetry-breaking costs become degenerate or infinite.

#### Mode S.E: Supercritical cascade

**Axiom Violated:** **(SC) Scaling Structure**

**Diagnostic Test:** A limiting profile $v \in X$ exists, but the gauge sequence $g_n \in G$ required to extract it is **supercritical**. Specifically, the scaling parameters $\lambda_n \to \infty$ diverge such that the associated cost exceeds the temporal compression, violating Property GN:
$$
\int_0^\infty \tilde{\mathfrak{D}}(S_t v) \, dt = \infty
$$

**Structural Mechanism:** The system organizes into a self-similar profile that collapses at a rate where the generation of dissipation dominates the shrinking time horizon. The scaling exponents satisfy $\alpha \leq \beta$ (Cost $\geq$ Time Compression). Energy concentrates but the renormalized profile cannot satisfy the dissipation budget.

**Status:** A "focusing" singularity where the profile remains regular in renormalized coordinates, but the renormalization factors become singular. This is a **genuine singularity** of cascade type.

**Metatheorem 4.25 (Supercriticality Exclusion).** If $\alpha > \beta$ (subcritical regime), then Mode S.E cannot occur.

*Proof.* The time-rescaled dissipation satisfies
$$
\int_0^\infty \tilde{\mathfrak{D}}(S_t v) \, dt = \lambda_n^{\beta - \alpha} \int_0^{T_*} \mathfrak{D}(u(t)) \, dt.
$$
When $\alpha > \beta$, we have $\lambda_n^{\beta - \alpha} \to 0$, so the renormalized dissipation vanishes in the limit. This contradicts the requirement that $v$ be a non-trivial profile. Hence supercritical blow-up is impossible. $\square$

**Example 4.26.** In the focusing NLS with $L^2$-critical power, the scaling is exactly critical ($\alpha = \beta$), allowing self-similar blow-up. For subcritical powers ($\alpha > \beta$), this mechanism is excluded.

#### Mode S.D: Stiffness breakdown

**Axiom Violated:** **(LS) Local Stiffness**

**Diagnostic Test:** The trajectory enters the neighborhood $U$ of the safe manifold $M$ but fails to converge at the required rate, satisfying:
$$
\int_{T_0}^{T_*} \|\dot{u}(t)\| \, dt = \infty \quad \text{while} \quad \mathrm{dist}(u(t), M) \to 0
$$
or the gradient inequality $|\nabla \Phi| \geq C \Phi^\theta$ fails.

**Structural Mechanism:** The energy landscape becomes "flat" (degenerate) near the target manifold, allowing the trajectory to creep indefinitely or oscillate without stabilizing. The Łojasiewicz gradient inequality, which normally provides polynomial convergence, fails to hold. This prevents the final regularization step.

**Status:** Asymptotic stagnation or infinite-time blow-up in finite time (if time rescaling is involved). This is a **deficiency failure**—insufficient energy gradient to drive convergence.

**Metatheorem 4.27 (Łojasiewicz Control).** If the Łojasiewicz inequality holds near $M$:
$$
|\nabla \Phi(x)| \geq C \cdot \mathrm{dist}(x, M)^{1-\theta}
$$
for some $\theta \in [0,1)$, then Mode S.D is excluded.

*Proof.* The Łojasiewicz inequality controls the convergence rate. Combining with the energy inequality $\frac{d}{dt}\Phi \leq -\mathfrak{D} \leq -|\nabla \Phi|^2$ yields
$$
\frac{d}{dt}\mathrm{dist}(u, M) \leq -C \cdot \mathrm{dist}(u, M)^{2(1-\theta)}.
$$
Integrating gives convergence in finite time when $\theta < 1/2$, and exponential convergence when $\theta = 0$ (non-degenerate critical point). $\square$

**Example 4.28.** In the Allen–Cahn equation, convergence to equilibrium follows the Łojasiewicz gradient inequality with $\theta = 1/2$ for generic initial data. For degenerate initial data (e.g., initial configurations on center manifolds), the inequality may fail.

#### Mode S.C: Parameter manifold instability

**Axiom Violated:** Symmetry (meta-stability)

**Diagnostic Test:** The structural parameters $\Theta = (\alpha, \beta, C_{\mathrm{LS}}, \ldots)$ undergo a discontinuous transition.

**Structural Mechanism:** The system undergoes a **parameter phase transition**: the ground state itself becomes unstable, and the trajectory tunnels to a different phase with distinct structural parameters. This is not a failure within a fixed hypostructure but a failure of the hypostructure itself. The symmetry class changes discontinuously.

**Status:** A **complexity failure** representing structural instability. The vacuum (ground state) decays to a different vacuum.

**Proposition 4.29.** Mode S.C represents failure of the hypostructure itself, not failure within a fixed hypostructure. It occurs when the system tunnels to a different phase with distinct structural parameters.

*Proof.*

**Step 1 (Phase characterization).** A hypostructure $\mathcal{H} = (X, S_t, \Phi, \mathfrak{D}, G, M)$ defines a "phase" via its structural parameters $\Theta = (\alpha, \beta, \theta_{\mathrm{LS}}, \Delta, \ldots)$. Different phases have different parameter values.

**Step 2 (Barrier between phases).** Between phases $\mathcal{H}_1$ and $\mathcal{H}_2$, there exists an energy barrier:
$$B_{12} := \inf_{\gamma: M_1 \to M_2} \max_{s} \Phi(\gamma(s)) - \min(\Phi_{1,\min}, \Phi_{2,\min})$$
where the infimum is over paths connecting the safe manifolds.

**Step 3 (Instanton tunneling).** By the vacuum nucleation barrier (Theorem 9.150), the transition rate is:
$$\Gamma_{1 \to 2} \sim e^{-B_{12}/\hbar}$$
in the semiclassical limit. The instanton is the optimal path achieving the barrier minimum.

**Step 4 (Mode S.C occurrence).** Mode S.C occurs when $B_{12} = 0$ or when thermal/quantum fluctuations overcome the barrier. The system discontinuously transitions from $\mathcal{H}_1$ to $\mathcal{H}_2$, invalidating the original hypostructure description. $\square$

**Metatheorem 4.30 (Mass Gap Principle).** Let $\mathcal{S}$ be a hypostructure with scale invariance group $G = \mathbb{R}_{>0}$ (dilations). If the ground state $V \in M$ breaks scale invariance (i.e., $\lambda \cdot V \neq V$ for $\lambda \neq 1$), then there exists a mass gap:
$$
\Delta := \inf_{x \notin M} \Phi(x) - \Phi_{\min} > 0.
$$

**Remark (Structural principle).** This theorem establishes that symmetry breaking implies a mass gap within the hypostructure framework. It explains *why* mass gaps emerge in systems exhibiting dimensional transmutation—the structural logic is universal across gauge theories satisfying the axioms.

*Proof.*

**Step 1 (Scale-invariant profiles).** If the theory has scale invariance $G = \mathbb{R}_{>0}$, then scale-invariant states $V$ satisfy $\lambda \cdot V = V$ for all $\lambda > 0$. Such states have $\Phi(\lambda \cdot V) = \lambda^\alpha \Phi(V)$ by the scaling axiom.

**Step 2 (Infinite cost for scale-invariant blow-up).** By Axiom SC with $\alpha > \beta$, the dissipation cost of a scale-invariant profile satisfies:
$$\int_0^\infty \mathfrak{D}(S_t V) \, dt = \lambda^{\beta - \alpha} \int_0^{T_*} \mathfrak{D}(u(t)) \, dt \to \infty$$
as $\lambda \to \infty$. Scale-invariant blow-up profiles have infinite cost.

**Step 3 (Symmetry breaking implies gap).** If the ground state $V \in M$ breaks scale invariance ($\lambda \cdot V \neq V$ for $\lambda \neq 1$), then $V$ is not scale-invariant. By Step 2, excited states cannot be continuously connected to $V$ via scale-invariant paths without infinite cost.

**Step 4 (Gap existence).** The only finite-energy states are:
- States in $M$ (the safe manifold, containing the symmetry-breaking vacuum)
- States separated from $M$ by the energy gap $\Delta > 0$

This gap $\Delta$ is the **mass gap**: the minimal energy needed to create an excitation. It prevents continuous paths from $M$ to excited states, stabilizing the vacuum against decay. $\square$

**Example 4.31 (Physical interpretation).** In QCD, the vacuum state is scale-invariant at the classical level, but quantum corrections break this symmetry via dimensional transmutation, generating a mass gap. The framework explains *why* physicists observe confinement and a mass gap—it is a structural consequence of symmetry breaking. Mode S.C corresponds to vacuum instability in theories without such stabilization.

---

### 4.7 Boundary failures (Modes B.E–B.C)

The preceding modes (1–12) describe **internal failures**—breakdowns within a closed system. When the hypostructure is coupled to an external environment $\mathcal{E}$, three additional failure modes emerge corresponding to pathological boundary interactions.

**Definition 4.32 (Open system).** An **open hypostructure** is a tuple $(\mathcal{S}, \mathcal{E}, \partial)$ where $\mathcal{S}$ is a hypostructure, $\mathcal{E}$ is an environment, and $\partial: \mathcal{E} \times X \to TX$ is a boundary coupling.

#### Mode B.E: Injection singularity

**Axiom Violated:** Boundedness of input

**Diagnostic Test:** External forcing exceeds the dissipative capacity:
$$
\|\partial(e(t), u(t))\| > C \cdot \mathfrak{D}(u(t)) \quad \text{for } t \in [T_0, T_*).
$$

**Structural Mechanism:** The environment injects energy or information faster than the system can dissipate or process it. This represents **input overload**—the system cannot maintain internal coherence under excessive external forcing. Energy may remain bounded but the coupling term drives instability.

**Status:** A **boundary excess failure**. This is a **genuine singularity** induced by external forcing.

**Proposition 4.33 (BIBO stability).** Mode B.E is excluded if the system is bounded-input bounded-output stable: bounded external forcing produces bounded response.

*Proof.*

**Step 1 (BIBO definition).** A system with input $e(t)$ and state $u(t)$ is **bounded-input bounded-output (BIBO) stable** if:
$$\sup_t \|e(t)\| \leq M_{\text{in}} \implies \sup_t \|u(t)\| \leq M_{\text{out}}$$
for some finite $M_{\text{out}}$ depending on $M_{\text{in}}$ and initial conditions.

**Step 2 (Transfer function characterization).** In the frequency domain, the input-output relation is $\hat{u}(s) = H(s) \hat{e}(s)$ where $H(s)$ is the transfer function. BIBO stability is equivalent to:
$$\|H\|_{L^1} := \int_0^\infty |h(t)| \, dt < \infty$$
where $h(t)$ is the impulse response.

**Step 3 (Bound propagation).** Given $\|e\|_{L^\infty} \leq M$:
$$\|u(t)\| = \left\|(h * e)(t)\right\| \leq \|h\|_{L^1} \cdot \|e\|_{L^\infty} \leq \|H\|_{L^1} \cdot M.$$

**Step 4 (Mode B.E exclusion).** Mode B.E requires the response to blow up under bounded forcing. BIBO stability guarantees $\|u\|_{L^\infty} < \infty$, preventing blow-up. Thus Mode B.E is excluded for BIBO stable systems. $\square$

**Example 4.34.** Adversarial attacks on neural networks exploit Mode B.E by injecting inputs with high-frequency components exceeding the network's effective bandwidth, causing misclassification despite small input perturbations.

**Remark 4.35.** In fluid dynamics, this corresponds to forced turbulence where the stirring scale exceeds the dissipation scale, preventing energy cascade from reaching the dissipative range.

#### Mode B.D: Starvation collapse

**Axiom Violated:** Persistence of excitation

**Diagnostic Test:** The coupling to the environment vanishes:
$$
\lim_{t \to T_*} \|\partial(e(t), u(t))\| = 0 \quad \text{while } u(t) \notin M.
$$

**Structural Mechanism:** The external input ceases before the system reaches equilibrium. Without environmental coupling, the autonomous dynamics must drive the system to $M$. If the internal dissipation is insufficient, evolution halts prematurely. This represents **resource cutoff** or **starvation**.

**Status:** A **boundary deficiency failure**. This is **not a singularity** but a halting condition—the system freezes before reaching the target.

**Proposition 4.36.** Mode B.D represents halting rather than blow-up. The trajectory ceases to evolve before reaching the safe manifold.

*Proof.* Without external input, the autonomous dynamics satisfy $\frac{d}{dt}u = F(u)$. If $F(u) = 0$ while $u \notin M$, evolution halts. For gradient flows, this requires $\mathfrak{D}(u) = 0$, which occurs only at critical points. If these critical points lie outside $M$, the system is trapped. $\square$

**Example 4.37.** In neural network training, Mode B.D corresponds to vanishing gradients: the loss landscape becomes flat before reaching a global minimum, causing training to stall. In biological systems, this represents metabolic starvation—insufficient external resources to complete development.

#### Mode B.C: Boundary-bulk incompatibility

**Axiom Violated:** Alignment

**Diagnostic Test:** The internal optimization direction is orthogonal to the external utility:
$$
\langle \nabla \Phi(u), \nabla U(u) \rangle \leq 0,
$$
where $U: X \to \mathbb{R}$ is the external utility function.

**Structural Mechanism:** The system optimizes its internal metric $\Phi$ while the environment evaluates performance by an external metric $U$. When these metrics are misaligned, internal optimization leads to externally poor outcomes. This represents **objective orthogonality**—the system and environment have incompatible goals.

**Status:** A **boundary complexity failure**. The system may reach $M$ with respect to $\Phi$ but diverge with respect to $U$.

**Metatheorem 4.38 (Goodhart's Law).** If the internal objective $\Phi$ is optimized without constraint, while the external utility $U$ depends on $\Phi$ only through a proxy $\tilde{\Phi}$, then:
$$
\lim_{t \to \infty} \Phi(u(t)) = \Phi_{\min} \quad \text{does not imply} \quad \lim_{t \to \infty} U(u(t)) = U_{\max}.
$$

*Proof.* Optimizing a proxy does not optimize the true objective when the proxy-reality map is non-monotonic or has measure-zero level sets. Formally, if $\tilde{\Phi} = \pi \circ \Phi$ where $\pi: \mathbb{R} \to \mathbb{R}$ is not injective, then minimizing $\tilde{\Phi}$ permits multiple values of $\Phi$, only one of which maximizes $U$. This is Goodhart's law formalized. $\square$

**Remark 4.39.** Mode B.C is the formal statement of AI alignment failure: a system that perfectly optimizes its internal metric may produce arbitrarily bad outcomes by external metrics.

**Example 4.40.** In reinforcement learning, reward hacking occurs when an agent discovers a policy that maximizes the reward signal $\Phi$ (e.g., by exploiting bugs) without maximizing the intended utility $U$. In economics, this corresponds to metric gaming—optimizing official measures while degrading true value.

---

### 4.8 The regularity logic

The framework proves global regularity via **soft local exclusion**: if blow-up cannot satisfy its permits, blow-up is impossible.

**Metatheorem 4.41 (Regularity via Soft Local Exclusion).** Let $\mathcal{S}$ be a hypostructure. A trajectory $u(t)$ extends to $T = +\infty$ (Global Regularity) if any of the following hold:

1. **Mode D.D (Dispersion):** Energy does not concentrate—solution exists globally via scattering.
2. **Modes S.E–D.C denied:** If energy concentrates (structure forced), but the forced structure $V$ fails any algebraic permit (SC, Cap, TB, LS, etc.), then blow-up is impossible—contradiction yields regularity.
3. **Boundary modes excluded:** For open systems, if Modes B.E–B.C are excluded by stability conditions, then global regularity follows.

**The proof of regularity does not require showing Mode D.D is "excluded."** Mode D.D *is* global regularity (via dispersion). The framework operates by:
- Assuming a singularity attempts to form at $T_* < \infty$
- Observing that blow-up forces concentration, which forces structure
- Checking whether the forced structure can satisfy its algebraic permits
- Concluding that permit denial implies the singularity cannot exist

*Proof (Soft Local Exclusion).* We prove regularity by contradiction.

**Assume a singularity attempts to form at $T_* < \infty$.** We show this leads to contradiction unless energy escapes to infinity (Mode C.E).

*Step 1: Energy must be bounded at blow-up.* If $\limsup_{t \to T_*} \Phi(u(t)) = \infty$, this is Mode C.E (energy blow-up)—a genuine singularity. We assume this does not occur, so $\sup_{t < T_*} \Phi(u(t)) \leq E < \infty$.

*Step 2: Bounded energy at blow-up forces concentration.* To form a singularity at $T_* < \infty$ with bounded energy, the energy must concentrate (otherwise the solution disperses globally—Mode D.D, which is global existence). Concentration is **forced** by the blow-up assumption.

*Step 3: Concentration forces structure.* By the Forced Structure Principle (Section 2.1), wherever blow-up attempts to form, energy concentration forces the emergence of a canonical profile $V$. A subsequence $u(t_n) \to g_n^{-1} \cdot V$ converges strongly modulo $G$.

*Step 4: Check permits on the forced structure.* The forced profile $V$ must satisfy the algebraic permits:
- **Scaling Permit (SC):** Is the blow-up subcritical ($\alpha > \beta$)?
- **Capacity Permit (Cap):** Does the singular set have positive capacity?
- **Topological Permit (TB):** Is the topological sector accessible?
- **Stiffness Permit (LS):** Does the Łojasiewicz inequality hold near equilibria?
- **Additional permits:** Frequency bounds (Mode D.E), causal depth (Mode C.C), tameness (Mode T.C), etc.

*Step 5: Permit denial yields contradiction.* If any permit is denied:
- SC fails $\Rightarrow$ Mode S.E: supercritical blow-up is impossible (dissipation dominates time compression).
- Cap fails $\Rightarrow$ Mode C.D: dimensional collapse is impossible (capacity bounds violated).
- TB fails (sector) $\Rightarrow$ Mode T.E: topological sector is inaccessible.
- LS fails $\Rightarrow$ Mode S.D: stiffness breakdown is impossible (Łojasiewicz controls convergence).
- Frequency bound fails $\Rightarrow$ Mode D.E: oscillatory singularity is impossible (Bode integral constraint).
- TB fails (tameness) $\Rightarrow$ Mode T.C: wild topology is impossible (o-minimality).

Each denial implies **the singularity cannot form**—contradiction.

*Step 6: Conclusion.* The only way a singularity can form is if all permits are satisfied (allowing energy to escape via Mode C.E). If any algebraic permit fails, the assumed singularity cannot exist, and $T_*(x) = +\infty$.

**Global regularity follows from soft local exclusion.** $\square$

**Remark 4.42 (The regularity argument).** The method does **not** require proving compactness globally or showing that Mode D.D is "impossible." The logic is:
- Mode D.D **is** global regularity (dispersion/scattering).
- To prove regularity, we assume blow-up attempts to form, observe that structure is forced, and check whether the forced structure can pass its permits.
- If permits are denied via soft algebraic analysis, the singularity cannot exist.

**Corollary 4.43 (Regularity criterion).** A trajectory achieves global regularity if and only if all fifteen modes are excluded by the algebraic permits derived from the hypostructure axioms.

---

### 4.9 The two-tier classification

The classification has a two-tier structure reflecting the logical dependency of the failure modes.

**Proposition 4.44 (Two-tier classification).** Let $u(t) = S_t x$ be any trajectory. The classification proceeds in two tiers:

**Tier 1: Does finite-time blow-up attempt to form?**
$$
\mathcal{E}_\infty := \{\text{trajectories with } \limsup_{t \to T_*} \Phi(u(t)) = \infty\} \quad \text{(Mode C.E: genuine blow-up)}
$$
$$
\mathcal{D} := \{\text{trajectories where energy disperses (no concentration)}\} \quad \text{(Mode D.D: global existence)}
$$
$$
\mathcal{C} := \{\text{trajectories with bounded energy and concentration}\} \quad \text{(Proceed to Tier 2)}
$$

**Tier 2: Can the forced structure pass its algebraic permits?**

For trajectories in $\mathcal{C}$, concentration forces a canonical profile $V$. Test whether $V$ satisfies the permits:
- **SC Permit denied** $\Rightarrow$ Mode S.E: Contradiction, singularity impossible.
- **Cap Permit denied** $\Rightarrow$ Mode C.D: Contradiction, singularity impossible.
- **TB Permit denied (sector)** $\Rightarrow$ Mode T.E: Contradiction, singularity impossible.
- **LS Permit denied** $\Rightarrow$ Mode S.D: Contradiction, singularity impossible.
- **Derivative bound denied** $\Rightarrow$ Mode D.E: Contradiction, singularity impossible.
- **Ergodicity fails** $\Rightarrow$ Mode T.D: Metastable trap (not a singularity).
- **Causal depth bound denied** $\Rightarrow$ Mode C.C: Contradiction, singularity impossible.
- **Parameter stability fails** $\Rightarrow$ Mode S.C: Parameter manifold instability (structural failure).
- **Tameness denied** $\Rightarrow$ Mode T.C: Contradiction, singularity impossible.
- **Invertibility fails** $\Rightarrow$ Mode D.C: Semantic horizon (inaccessibility).
- **All permits satisfied** $\Rightarrow$ Genuine structured singularity (rare).

For **open systems**, test boundary conditions:
- **Input exceeds dissipation** $\Rightarrow$ Mode B.E: Injection singularity.
- **Input vanishes prematurely** $\Rightarrow$ Mode B.D: Starvation collapse.
- **Objective misalignment** $\Rightarrow$ Mode B.C: Alignment failure.

*Proof.* Tier 1 is a disjoint partition:
- Either $\limsup \Phi = \infty$ (Mode C.E: genuine blow-up), or $\sup \Phi < \infty$.
- Given bounded energy, either concentration occurs ($\mathcal{C}$), or dispersion occurs (Mode D.D: global existence).

Tier 2 applies only when concentration occurs: the forced profile $V$ is tested against the algebraic permits. If all permits pass, a genuine structured singularity occurs (rare). If any permit fails, the singularity is impossible. $\square$

**Corollary 4.45 (Regularity by tier).** Global regularity is achieved whenever:
- **Tier 1:** Energy disperses (Mode D.D)—no concentration, no singularity, global existence.
- **Tier 2:** Concentration occurs but permits are denied—singularity is impossible, global regularity by contradiction.

The only genuine singularities are Mode C.E (energy blow-up) or structured singularities where all permits pass (rare in well-posed systems).

**Remark 4.46 (Mode D.D is not analyzed further).** Mode D.D represents **global existence via scattering**. The framework does not "analyze" Mode D.D because there is nothing to analyze—no singularity forms. When energy disperses:
- The solution exists globally.
- No local structure forms (no concentration).
- No permit checking is needed (there is no forced structure).

The framework's power lies in showing that **when concentration does occur** (Tier 2), the forced structure must pass algebraic permits—and these permits can often be denied via soft dimensional analysis.

**Remark 4.47 (Regularity via soft local exclusion).** To prove global regularity using the hypostructure framework:

1. **Identify the algebraic data:** Scaling exponents $\alpha, \beta$; capacity dimensions; Łojasiewicz exponents near equilibria; topological invariants.
2. **Assume blow-up at $T_* < \infty$:** Concentration is forced, so a canonical profile $V$ emerges.
3. **Check permits on $V$:**
   - If $\alpha > \beta$ (Axiom SC holds), supercritical cascade (Mode S.E) is impossible.
   - If singular sets have positive capacity (Axiom Cap holds), geometric collapse (Mode C.D) is impossible.
   - If topological sectors are preserved (Axiom TB holds), topological obstruction (Mode T.E) is impossible.
   - If Łojasiewicz inequality holds (Axiom LS holds), stiffness breakdown (Mode S.D) is impossible.
   - If frequency bounds hold, oscillatory singularity (Mode D.E) is impossible.
   - If causal depth is bounded, Finite-time event accumulation (Mode C.C) is impossible.
   - If dynamics are tame, labyrinthine singularity (Mode T.C) is impossible.
4. **Conclude:** Permit denial $\Rightarrow$ singularity impossible $\Rightarrow$ $T_* = \infty$.

**No global compactness proof is required.** The framework converts PDE regularity into local algebraic permit-checking on forced structure.

**Remark 4.48 (The decision structure).** The classification operates as follows:

1. Is energy bounded? If no: **Mode C.E** (genuine blow-up). If yes: proceed.
2. Does concentration occur? If no: **Mode D.D** (global existence via dispersion). If yes: proceed.
3. Test the forced profile $V$ against algebraic permits. Permit denial $\Rightarrow$ contradiction $\Rightarrow$ **global regularity**.
4. Check complexity modes (Modes D.E–D.C) for bounded but pathological behavior.
5. For open systems, check boundary modes (Modes B.E–B.C).
6. If all permits pass: genuine structured singularity (rare).

Mode D.D and permit-denial both yield global regularity—but via different mechanisms (dispersion vs. contradiction).

---

**Summary.** The fifteen failure modes form a complete, orthogonal classification of dynamical breakdown. This topological classification of breakdown mirrors René Thom's Catastrophe Theory \cite{Thom72}, extending the elementary catastrophes to infinite-dimensional dynamical spaces. The taxonomy structure reveals that singularities are systematic violations of coherence constraints rather than arbitrary pathologies. The framework reduces the problem of proving global regularity to algebraic permit-checking on forced structures.

---

## 6. The Resolution Theorems

### 6.1 Theorem 6.1: Structural Resolution of Trajectories

(Originally Theorem 7.1 in source)

**Metatheorem 6.1 (Structural Resolution).** Let $\mathcal{S}$ be a structural flow datum satisfying the minimal regularity (Reg) and dissipation (D) axioms. Let $u(t) = S_t x$ be any trajectory.

**The Structural Resolution** classifies every trajectory into one of three outcomes:

| Outcome | Modes | Mechanism |
|---------|-------|-----------|
| **Global Existence (Dispersive)** | Mode D.D | Energy disperses, no concentration, solution scatters globally |
| **Global Regularity (Permit Denial)** | Modes S.E, C.D, T.E, S.D | Energy concentrates but forced structure fails algebraic permits → contradiction |
| **Genuine Singularity** | Mode C.E, or Modes S.E–S.D with permits granted | Energy escapes (Mode C.E) or structured blow-up with all permits satisfied |

For any trajectory with finite breakdown time $T_*(x) < \infty$, the behavior falls into exactly one of the following modes:

**Tier I: Does blow-up attempt to concentrate?**

1. **Energy blow-up (Mode C.E):** $\Phi(S_{t_n} x) \to \infty$ for some sequence $t_n \nearrow T_*(x)$. (Genuine singularity via energy escape.)

2. **Dispersion (Mode D.D):** Energy remains bounded, but no subsequence of $(S_{t_n} x)$ converges modulo symmetries. Energy disperses—**no singularity forms**. This is global existence via scattering.

**Tier II: Concentration occurs—check algebraic permits**

If energy concentrates (bounded energy with convergent subsequence modulo $G$), a **canonical profile** $V$ is forced. Test whether the forced structure can pass its permits:

3. **Supercritical symmetry cascade (Mode S.E):** Violation of Axiom SC (Scaling). In normalized coordinates, a GN-forbidden profile appears (Type II self-similar blow-up).

4. **Geometric concentration (Mode C.D):** Violation of Axiom Cap (Capacity). The trajectory spends asymptotically all its time in sets $(B_k)$ with $\mathrm{Cap}(B_k) \to \infty$ (concentration on thin tubes or high-codimension defects).

5. **Topological obstruction (Mode T.E):** Violation of Axiom TB. The trajectory is constrained to a nontrivial topological sector with action exceeding the gap.

6. **Stiffness breakdown (Mode S.D):** Violation of Axiom LS near $M$. The trajectory approaches a limit point in $U \setminus M$ with height comparable to $\Phi_{\min}$, violating the Łojasiewicz inequality.

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

**Case 3: Geometric concentration.** Suppose neither (1), (2), nor (3) occurs. Consider where the trajectory spends its time. By the capacity occupation lemma (to be established in Theorem 6.3), the occupation time in any set $B$ with $\mathrm{Cap}(B) = M$ is at most $C_{\mathrm{cap}}(\Phi(x) + T)/M$.

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

**Corollary 6.1.1 (Mode classification and regularity).** The six modes classify trajectories by outcome:

| Mode | Type | Condition | Outcome |
|------|------|-----------|---------|
| (1) | Energy blow-up | **D** fails | Genuine singularity (energy escapes) |
| (2) | Dispersion | **C** fails (no concentration) | **Global existence** via scattering |
| (3) | SC permit denied | $\alpha \leq \beta$ | **Global regularity** (supercritical impossible) |
| (4) | Cap permit denied | Capacity bounds exceeded | **Global regularity** (geometric collapse impossible) |
| (5) | TB permit denied | Topological obstruction | **Global regularity** (sector inaccessible) |
| (6) | LS permit denied | Łojasiewicz fails | **Global regularity** (stiffness breakdown impossible) |

**Remark 6.1.2 (Regularity pathways).** The resolution reveals multiple pathways to global regularity:

1. **Mode D.D (Dispersion):** Energy does not concentrate—no singularity forms.
2. **Modes S.E–S.D (Permit denial):** Energy concentrates but the forced structure fails an algebraic permit—singularity is contradicted.
3. **Mode C.E avoided:** Energy remains bounded (Axiom D holds).

**The framework proves regularity via soft local exclusion.** When concentration is forced by a blow-up attempt, the algebraic permits determine whether the singularity can form. Permit denial yields contradiction, hence regularity.

### 6.2 Theorem 6.2: Scaling-based exclusion of supercritical blow-up

(Originally Theorem 7.2 in source)

#### 6.2.1 GN as a metatheorem from scaling structure

**Metatheorem 6.2.1 (GN from SC + D).** Let $\mathcal{S}$ be a hypostructure satisfying Axioms (D) and (SC) with scaling exponents $(\alpha, \beta)$ satisfying $\alpha > \beta$. Then Property GN holds: any supercritical blow-up profile has infinite dissipation cost.

More precisely: suppose $u(t) = S_t x$ is a trajectory with finite total cost $\mathcal{C}_*(x) < \infty$ and finite blow-up time $T_*(x) < \infty$. Suppose there exist:

- a supercritical sequence $\lambda_n \to \infty$,
- times $t_n \nearrow T_*(x)$,
- such that the rescaled states
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
where $\sim$ denotes equality up to the constant $C_\alpha$ from Definition 5.12.

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

The cost contribution per scale level is:
$$
\lambda_n^{-\alpha}(T_*(x) - t_n) \sim \lambda_n^{-\alpha} \cdot \lambda_n^{-\beta} S_n = \lambda_n^{-(\alpha + \beta)} S_n.
$$

Summing over dyadic scales $\lambda_n \sim 2^n$: if $\alpha > \beta$, the prefactor $\lambda_n^{-\alpha}$ decays faster than any polynomial growth in $S_n$ can compensate, **unless** $v_\infty$ has infinite dissipation. More precisely, if $\int_{-\infty}^0 \mathfrak{D}(v_\infty(s)) ds < \infty$, then the cost contributions would sum to a finite value, but the supercritical convergence $v_n \to v_\infty$ with expanding windows requires that the dissipation profile $v_\infty$ absorbs all the rescaled dissipation—which must diverge for the limit to exist nontrivially.

**Step 7: Contradiction.** Therefore:

- If $v_\infty$ is nontrivial and $\int_{-\infty}^0 \mathfrak{D}(v_\infty) ds < \infty$, the scaling arithmetic shows $\mathcal{C}_*(x) < \infty$ cannot hold.
- Conversely, if $\mathcal{C}_*(x) < \infty$, then either $v_\infty$ is trivial or $\int_{-\infty}^0 \mathfrak{D}(v_\infty) ds = \infty$.

This establishes Property GN from Axioms D and SC alone. $\square$

**Remark 6.2.2 (No PDE-specific ingredients).** The proof uses only:

1. The scaling transformation law for $\mathfrak{D}$ (from SC),
2. The time-scaling exponent $\beta$ (from SC),
3. The subcritical condition $\alpha > \beta$ (from SC),
4. Finite total cost (from D).

The proof uses only scaling arithmetic. Once SC is identified via dimensional analysis, GN follows.

#### 6.2.2 Type II exclusion

**Metatheorem 6.2 (Type II Exclusion).** Let $\mathcal{S}$ be a hypostructure satisfying Axioms (D) and (SC). Let $x \in X$ with $\Phi(x) < \infty$ and $\mathcal{C}_*(x) < \infty$ (finite total cost). Then no supercritical self-similar blow-up can occur at $T_*(x)$.

More precisely: there do not exist a supercritical sequence $(\lambda_n) \subset \mathbb{R}_{>0}$ with $\lambda_n \to \infty$ and times $t_n \nearrow T_*(x)$ such that $v_n := \mathcal{S}_{\lambda_n} \cdot S_{t_n} x$ converges to a nontrivial profile $v_\infty \in X$.

*Proof.* Immediate from Theorem 6.2.1. By that theorem, any such limit profile $v_\infty$ must satisfy $\int_{-\infty}^0 \mathfrak{D}(v_\infty(s)) ds = \infty$. But a nontrivial self-similar blow-up profile, by definition, has finite local dissipation (otherwise it would not be a coherent limiting object). This contradiction excludes the existence of such profiles.

Alternatively: the finite-cost trajectory $u(t)$ has dissipation budget $\mathcal{C}_*(x) < \infty$. The scaling arithmetic of Theorem 6.2.1 shows this budget cannot produce a nontrivial infinite-dissipation limit. Hence no supercritical blow-up. $\square$

**Corollary 6.2.3 (Type II blow-up is framework-forbidden).** In any hypostructure satisfying (D) and (SC) with $\alpha > \beta$, Type II (supercritical self-similar) blow-up is impossible for finite-cost trajectories. This holds regardless of the specific dynamics; it is a consequence of scaling structure alone.

#### 6.2.3 The Criticality Lemma (Liouville Connection)

The results above handle the subcritical case $\alpha > \beta$. A key question remains: *What happens at criticality ($\alpha = \beta$)?* This is precisely where many important regularity problems reside. The following lemma provides the tie-breaker mechanism.

**Lemma 6.2.4 (Criticality-Liouville Bridge).** Let $\mathcal{S}$ be a hypostructure with scaling exponents $(\alpha, \beta)$ satisfying $\alpha = \beta$ (critical scaling). Suppose a trajectory $u(t)$ exhibits Type II blow-up with limiting profile $V$. Then:

1. $V$ is a non-trivial finite-energy solution to the **stationary equation** on $\mathbb{R}^n$:
$$
\mathcal{L}[V] = 0, \quad \Phi(V) < \infty.
$$
2. The existence of such $V$ is equivalent to the **failure of the Liouville theorem** for the stationary problem.

*Proof.*

**Step 1 (Profile extraction).** By hypothesis, there exist times $t_n \nearrow T_*$ and scales $\lambda_n \to \infty$ such that the rescaled states
$$
V_n(y) := \lambda_n^{\gamma} u(t_n, \lambda_n^{-1} y)
$$
converge to a nontrivial limit $V$ in an appropriate topology, where $\gamma$ is determined by scaling.

**Step 2 (Criticality forces stationarity).** At critical scaling $\alpha = \beta$, the rescaled evolution equation becomes time-independent in the limit. Specifically, if $u$ solves
$$
\partial_t u = \mathcal{L}[u]
$$
then $V_n$ solves
$$
\partial_\tau V_n = \lambda_n^{\beta - \alpha} \mathcal{L}[V_n] = \mathcal{L}[V_n]
$$
in rescaled time $\tau = \lambda_n^\beta(t - t_n)$. As $n \to \infty$, the evolution "freezes" and $V$ satisfies the stationary equation $\mathcal{L}[V] = 0$.

**Step 3 (Finite energy).** Since $\Phi(u(t_n)) \leq E_0 < \infty$ and $\alpha = \beta$, we have
$$
\Phi(V_n) = \lambda_n^{\alpha - \alpha} \Phi(u(t_n)) = \Phi(u(t_n)) \leq E_0.
$$
Thus $\Phi(V) \leq E_0 < \infty$.

**Step 4 (Liouville equivalence).** The profile $V$ is therefore a non-trivial ($V \neq 0$) finite-energy ($\Phi(V) < \infty$) solution to the stationary equation on $\mathbb{R}^n$. Such solutions exist if and only if the Liouville theorem fails for the stationary problem. $\square$

**Corollary 6.2.5 (Critical Resolution via Liouville).** If the Liouville theorem holds for the stationary equation—i.e., the only finite-energy solution to $\mathcal{L}[V] = 0$ on $\mathbb{R}^n$ is $V \equiv 0$—then Type II blow-up is excluded even in the critical case $\alpha = \beta$.

*Proof.* By Lemma 6.2.4, any blow-up profile must be a non-trivial finite-energy stationary solution. The Liouville theorem asserts no such solution exists. Therefore the profile must be trivial ($V = 0$), contradicting the assumption of non-trivial blow-up. $\square$

**Metatheorem 6.2.6 (Critical Scaling + Liouville $\implies$ Regularity).** Let $\mathcal{S}$ be a hypostructure satisfying Axioms (D), (SC), and (C) with critical scaling $\alpha = \beta$. Suppose:

1. **Liouville theorem:** The only finite-energy solution to the stationary equation $\mathcal{L}[V] = 0$ on $\mathbb{R}^n$ is $V \equiv 0$.
2. **Compactness (Axiom C):** Bounded-energy sequences have convergent subsequences modulo symmetry.

Then the system admits global regularity: $T_*(x) = \infty$ for all finite-energy initial data.

*Proof.* Suppose for contradiction that $T_*(x) < \infty$. By Axiom C, along a blow-up sequence we can extract a non-trivial limit profile $V$. By Lemma 6.2.4, $V$ is a finite-energy stationary solution. By hypothesis (1), $V = 0$. This contradicts non-triviality. $\square$

**Remark 6.2.7 (Application to viscous fluids).** For dissipative fluid equations, the Liouville theorem often holds due to the dissipation structure. Specifically, if $V$ is a finite-energy stationary solution, the energy identity gives
$$
0 = \frac{d}{dt}\Phi(V) = -\mathfrak{D}(V) \leq 0,
$$
so $\mathfrak{D}(V) = 0$. Under appropriate coercivity (Axiom LS), this implies $V = 0$. This mechanism provides the "tie-breaker" for critical scaling in viscous systems.

### 6.3 Theorem 6.3: Capacity barrier

(Originally Theorem 7.3 in source)

**Metatheorem 6.3 (Capacity Barrier).** Let $\mathcal{S}$ be a hypostructure with geometric background (BG) satisfying Axiom Cap. Let $(B_k)$ be a sequence of subsets of $X$ of increasing geometric "thinness" (e.g., $r_k$-tubular neighbourhoods of codimension-$\kappa$ sets with $r_k \to 0$) such that:
$$
\mathrm{Cap}(B_k) \gtrsim r_k^{-\kappa} \to \infty.
$$

Then for any finite-energy trajectory $u(t) = S_t x$ and any $T > 0$:
$$
\lim_{k \to \infty} \mathrm{Leb}\{t \in [0, T] : u(t) \in B_k\} = 0.
$$

*Proof.* By the occupation measure lemma (established in preparatory work), for each $k$:
$$
\tau_k := \mathrm{Leb}\{t \in [0, T] : u(t) \in B_k\} \leq \frac{C_{\mathrm{cap}}(\Phi(x) + T)}{\mathrm{Cap}(B_k)}.
$$

The numerator $C_{\mathrm{cap}}(\Phi(x) + T)$ is a fixed constant depending only on the initial energy and time horizon. By hypothesis, $\mathrm{Cap}(B_k) \to \infty$. Therefore:
$$
\lim_{k \to \infty} \tau_k \leq \lim_{k \to \infty} \frac{C_{\mathrm{cap}}(\Phi(x) + T)}{\mathrm{Cap}(B_k)} = 0.
$$

This shows that the fraction of time spent in $B_k$ tends to zero. $\square$

**Corollary 6.3.1 (No concentration on thin structures).** Blow-up scenarios relying on persistent concentration inside:
- arbitrarily thin tubes,
- arbitrarily small neighbourhoods of lower-dimensional manifolds,
- fractal defect sets of Hausdorff dimension $< Q$,

are incompatible with finite energy and the capacity axiom.

*Proof.* Such sets have capacity tending to infinity by the capacity-codimension bound (Axiom BG4). Apply Theorem 6.3. $\square$

### 6.4 Theorem 6.4: Topological sector suppression

(Originally Theorem 7.4 in source)

**Metatheorem 6.4 (Topological Sector Suppression).** Assume the topological background (TB) with action gap $\Delta > 0$ and an invariant probability measure $\mu$ satisfying a log-Sobolev inequality with constant $\lambda_{\mathrm{LS}} > 0$. Assume the action functional $\mathcal{A}$ is Lipschitz with constant $L > 0$. Then:
$$
\mu(\{x : \tau(x) \neq 0\}) \leq C \exp\left(-c \lambda_{\mathrm{LS}} \frac{\Delta^2}{L^2}\right)
$$
for universal constants $C, c > 0$ (specifically, $C = 1$ and $c = 1/8$).

Moreover, for $\mu$-typical trajectories, the fraction of time spent in nontrivial sectors decays exponentially in the action gap.

*Proof.*

**Step 1: Setup and concentration inequality.** By Axiom TB1 (action gap), the nontrivial topological sector is separated from the trivial sector by an action gap:
$$
\tau(x) \neq 0 \implies \mathcal{A}(x) \geq \mathcal{A}_{\min} + \Delta.
$$

Assume $\mathcal{A}: X \to [0, \infty)$ is Lipschitz with constant $L > 0$ (this holds when the action is defined via path integrals in a metric space). By the Herbst argument (established in preparatory lemmas), the log-Sobolev inequality with constant $\lambda_{\mathrm{LS}}$ implies Gaussian concentration: for any $r > 0$,
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
where the last inequality follows from the Łojasiewicz decay and the concentration of $\mu$ near $M$. Thus $\bar{\mathcal{A}} \leq \mathcal{A}_{\min} + \epsilon$ for $\epsilon$ exponentially small in $\lambda_{\mathrm{LS}}$.

**Step 3: Bound on nontrivial sector measure.** We bound $\mu(\tau \neq 0)$.

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
\mu(\tau \neq 0) \leq \exp\left(-\frac{\lambda_{\mathrm{LS}} (\Delta/2)^2}{2L^2}\right) = \exp\left(-\frac{\lambda_{\mathrm{LS}} \Delta^2}{8L^2}\right),
$$
which gives the claimed bound with $C = 1$ and $c = 1/8$.

**Step 4: Ergodic extension to trajectories.** For a trajectory $u(t) = S_t x$ that is ergodic with respect to $\mu$, Birkhoff's ergodic theorem gives:
$$
\lim_{T \to \infty} \frac{1}{T} \int_0^T \mathbf{1}_{\tau(u(t)) \neq 0} \, dt = \mu(\tau \neq 0), \quad \mu\text{-almost surely}.
$$

Combined with the bound from Step 3:
$$
\limsup_{T \to \infty} \frac{1}{T} \int_0^T \mathbf{1}_{\tau(u(t)) \neq 0} \, dt \leq C \exp\left(-c \lambda_{\mathrm{LS}} \frac{\Delta^2}{L^2}\right),
$$
for $\mu$-almost every initial condition $x$.

This establishes that typical trajectories spend an exponentially small fraction of time in nontrivial topological sectors. $\square$

**Remark 6.4.1.** If the action gap $\Delta$ is large (strong topological protection), nontrivial sectors are exponentially rare. Exotic topological configurations (instantons, monopoles, defects with nontrivial homotopy) are statistically suppressed under thermal equilibrium.

### 6.5 Theorem 6.5: Structured vs failure dichotomy

(Originally Theorem 7.5 in source)

**Metatheorem 6.5 (Structured vs Failure Dichotomy).** Let $X = \mathcal{S} \cup \mathcal{F}$ be decomposed into:
- the **structured region** $\mathcal{S}$ where the safe manifold $M \subset \mathcal{S}$ lies and good regularity holds,
- the **failure region** $\mathcal{F} = X \setminus \mathcal{S}$.

Assume Axioms (D), (R), (Cap), and (LS) (near $M$). Then any finite-energy trajectory $u(t) = S_t x$ with finite total cost $\mathcal{C}_*(x) < \infty$ satisfies:

Either $u(t)$ enters $\mathcal{S}$ in finite time and remains at uniformly bounded distance from $M$ thereafter, or the trajectory contradicts the finite-cost assumption.

*Proof.*

**Step 1: Time in failure region is bounded.** By the cost-recovery duality lemma, the time spent outside the good region $\mathcal{G}$ satisfies:
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

**Step 3: Convergence to $M$.** Once in $\mathcal{S}$, by Axiom LS, the Łojasiewicz inequality holds near $M$. If the trajectory enters the neighbourhood $U$ of $M$, the Łojasiewicz decay estimate gives convergence:
$$
\mathrm{dist}(u(t), M) \to 0 \quad \text{as } t \to \infty.
$$

If the trajectory remains in $\mathcal{S} \setminus U$, then by the properties of $\mathcal{S}$ (standard regularity, no singular behaviour), the trajectory is globally regular and bounded away from $M$ but still well-behaved.

**Step 4: Contradiction from persistent failure.** Suppose the trajectory spends infinite time in $\mathcal{F}$ or never stabilizes in $\mathcal{S}$. Then either:
- the trajectory has infinite cost (contradicting $\mathcal{C}_*(x) < \infty$), or
- the trajectory enters high-capacity regions (excluded by Theorem 6.3), or
- the trajectory exhibits supercritical blow-up (excluded by Theorem 6.2), or
- the trajectory is constrained to a nontrivial topological sector (excluded by Theorem 6.4 for typical data).

All alternatives are incompatible with the assumptions. $\square$

### 6.6 Theorem 6.6: Canonical Lyapunov functional

(Originally Theorem 7.6 in source)

**Metatheorem 6.6 (Canonical Lyapunov Functional).** Assume Axioms (C), (D) with $C = 0$, (R), (LS), and (Reg). Then there exists a functional $\mathcal{L}: X \to \mathbb{R} \cup \{\infty\}$ with the following properties:

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

Combined with the lower bound from LS (via the Łojasiewicz decay estimate), this gives the equivalence.

**Step 5: Uniqueness.** Suppose $\Psi$ is another Lyapunov functional with the same properties. Define $f: \mathrm{Im}(\mathcal{L}) \to \mathbb{R}$ by $f(\mathcal{L}(x)) = \Psi(x)$.

This is well-defined because if $\mathcal{L}(x_1) = \mathcal{L}(x_2)$, then by the equivalence to distance from $M$, $\mathrm{dist}(x_1, M) \asymp \mathrm{dist}(x_2, M)$. By similar reasoning for $\Psi$, we get $\Psi(x_1) \asymp \Psi(x_2)$.

Monotonicity of both $\mathcal{L}$ and $\Psi$ along trajectories, combined with their strict decrease outside $M$, implies $f$ is increasing. $\square$

**Remark 6.6.1 (Loss interpretation).** The functional $\mathcal{L}$ measures the total cost required to reach the optimal manifold $M$. This is the structural analogue of loss functions in optimization and machine learning, derived from the dynamical axioms.

### 6.7 Theorems 6.7.x: Functional reconstruction

(Originally Section 7.7 in source)

The theorems in Sections 6.1–6.6 assume a height functional $\Phi$ is given and identify its properties. We now provide a **generator**: a mechanism to explicitly recover the Lyapunov functional $\mathcal{L}$ solely from the dynamical data $(S_t)$ and the dissipation structure $(\mathfrak{D})$, without prior knowledge of $\Phi$.

This moves the framework from **identification** (recognizing a given $\Phi$) to **discovery** (finding the correct $\Phi$).

#### 6.7.1 Gradient consistency

**Definition 6.1 (Metric structure).** A hypostructure has **metric structure** if the state space $(X, d)$ is equipped with a Riemannian (or Finsler) metric $g$ such that the metric $d$ is induced by $g$: for smooth paths $\gamma: [0, 1] \to X$,
$$
d(x, y) = \inf_{\gamma: x \to y} \int_0^1 \|\dot{\gamma}(s)\|_g \, ds.
$$

**Definition 6.2 (Gradient consistency).** A hypostructure with metric structure is **gradient-consistent** if, for almost all $t \in [0, T_*(x))$ along any trajectory $u(t) = S_t x$:
$$
\|\dot{u}(t)\|_g^2 = \mathfrak{D}(u(t)),
$$
where $\dot{u}(t)$ is the metric velocity of the trajectory.

**Remark 6.2.1.** Gradient consistency encodes that the system is "maximally efficient" at converting dissipation into motion—a defining property of gradient flows where $\dot{u} = -\nabla \Phi$ and $\mathfrak{D} = \|\nabla \Phi\|^2$. This is **not** an additional axiom to verify case-by-case; it is a structural property that holds automatically for:

- Gradient flows in Hilbert spaces,
- Wasserstein gradient flows of free energies,
- $L^2$ gradient flows of geometric functionals,
- Any system where the "velocity equals negative gradient" structure is present.

**Axiom GC (Gradient Consistency on gradient-flow orbits).** Along any trajectory $u(t) = S_t x$ that evolves by gradient flow (i.e., $\dot{u} = -\nabla_g \Phi$), the gradient consistency condition $\|\dot{u}(t)\|_g^2 = \mathfrak{D}(u(t))$ holds.

**Fallback.** When Axiom GC fails along a trajectory—i.e., the trajectory is not a gradient flow—the reconstruction theorems (6.7.2–6.7.3) do not apply. The Lyapunov functional still exists by Theorem 6.6 via the abstract construction, but cannot be computed explicitly via the Jacobi metric or Hamilton–Jacobi equation.

#### 6.7.1b Generalization to Non-Riemannian Spaces

The Riemannian formulation of Axiom GC presupposes the existence of an inner product structure. For systems defined on Banach spaces (e.g., $L^1$ optimal transport), Wasserstein spaces, or discrete graphs, we require the generalization afforded by the **De Giorgi metric slope** \cite{AmbrosioGigliSavare08}.

**Definition 6.3 (Metric Slope).** Let $\Phi: (X, d) \to \mathbb{R}$ be a functional on a metric space. The **metric slope** of $\Phi$ at $u \in X$ is defined as:

$$|\partial \Phi|(u) := \limsup_{v \to u} \frac{(\Phi(u) - \Phi(v))^+}{d(u, v)}$$

where $(a)^+ := \max(a, 0)$. This quantity generalizes the gradient norm $\|\nabla \Phi\|$ to non-smooth and non-Riemannian settings.

**Remark 6.3.1 (Consistency with classical gradient).** On a Riemannian manifold $(M, g)$, the metric slope coincides with the gradient norm:
$$|\partial \Phi|(u) = \|\nabla_g \Phi(u)\|_g.$$
The generalization is strict: metric slopes remain well-defined in contexts where gradients do not exist.

**Axiom GC' (Dissipation-Slope Equality).** *Generalized Gradient Consistency.* Along any trajectory $u(t) = S_t x$ evolving as a **metric gradient flow** (in the sense of curves of maximal slope \cite{AmbrosioGigliSavare08}), the dissipation-slope equality holds:

$$\mathfrak{D}(u(t)) = |\partial \Phi|^2(u(t)).$$

**Proposition 6.3.2 (GC' extends GC).** *Axiom GC' strictly generalizes Axiom GC:*
1. *On Riemannian manifolds with gradient flow $\dot{u} = -\nabla_g \Phi$, Axiom GC' reduces to Axiom GC.*
2. *On Wasserstein space $(\mathcal{P}_2(\mathbb{R}^n), W_2)$, Axiom GC' holds for gradient flows of internal energies, including the Fokker-Planck equation.*
3. *On discrete graphs equipped with the counting metric, Axiom GC' applies to reversible Markov chains.*

*Proof.*
(1) The equivalence $|\partial \Phi| = \|\nabla_g \Phi\|_g$ on Riemannian manifolds yields the result immediately.

(2) Consider the entropy functional $\Phi(\rho) = \int \rho \log \rho \, dx$. Its Wasserstein gradient flow is the Fokker-Planck equation $\partial_t \rho = \Delta \rho$. The metric slope satisfies $|\partial \Phi|(\rho) = \|\nabla \log \rho\|_{L^2(\rho)} = \sqrt{I(\rho)}$, where $I(\rho)$ denotes the Fisher information. The dissipation functional is $\mathfrak{D}(\rho) = I(\rho)$, whence $\mathfrak{D} = |\partial \Phi|^2$.

(3) For Markov chains on a graph $(V, E)$, the discrete gradient $(\nabla f)_{xy} = f(y) - f(x)$ along edges induces a metric slope via the Benamou-Brenier formulation \cite{Maas11}. $\square$

**Metatheorem 6.7.1' (Extended Action Reconstruction).** *Under Axiom GC' (dissipation-slope equality), the reconstruction theorems (6.7.1–6.7.3) extend to general metric spaces. The Lyapunov functional satisfies:*

$$\mathcal{L}(x) = \Phi_{\min} + \inf_{\gamma: M \to x} \int_0^1 |\partial \Phi|(\gamma(s)) \cdot |\dot{\gamma}|(s) \, ds$$

*where the infimum ranges over all absolutely continuous curves from the safe manifold $M$ to $x$, and $|\dot{\gamma}|$ denotes the metric derivative.*

*Proof.* We establish the metric space generalization in three steps.

**Step 1 (Metric derivative).** For an absolutely continuous curve $\gamma: [0,1] \to (X, d)$, the metric derivative exists almost everywhere and is defined by:
$$|\dot{\gamma}|(s) := \lim_{h \to 0} \frac{d(\gamma(s+h), \gamma(s))}{|h|}$$
By \cite[Thm. 1.1.2]{AmbrosioGigliSavare08}, $|\dot{\gamma}| \in L^1([0,1])$ for absolutely continuous curves, and the curve length satisfies $\mathrm{Length}(\gamma) = \int_0^1 |\dot{\gamma}|(s) \, ds$.

**Step 2 (Energy-dissipation identity).** Along curves of maximal slope $u: [0, T] \to X$ for the functional $\Phi$, the energy-dissipation equality holds:
$$\Phi(u(0)) - \Phi(u(T)) = \int_0^T |\partial \Phi|^2(u(s)) \, ds = \int_0^T |\dot{u}|^2(s) \, ds$$
This follows from \cite[Thm. 1.2.5]{AmbrosioGigliSavare08}: curves of maximal slope satisfy $|\dot{u}|(t) = |\partial \Phi|(u(t))$ for almost every $t \in [0, T]$. The equality $|\dot{u}| = |\partial \Phi|$ characterizes gradient flows in the metric setting.

**Step 3 (Lyapunov reconstruction).** Define the candidate Lyapunov functional:
$$\mathcal{L}(x) := \Phi_{\min} + \inf_{\gamma: M \to x} \int_0^1 |\partial \Phi|(\gamma(s)) \cdot |\dot{\gamma}|(s) \, ds$$
where the infimum ranges over absolutely continuous curves from the safe manifold $M$ to $x$. By the Cauchy-Schwarz inequality:
$$\int_0^1 |\partial \Phi|(\gamma) \cdot |\dot{\gamma}| \, ds \geq \sqrt{\int_0^1 |\partial \Phi|^2(\gamma) \, ds} \cdot \sqrt{\int_0^1 |\dot{\gamma}|^2 \, ds}$$
with equality if and only if $|\dot{\gamma}|(s) = c \cdot |\partial \Phi|(\gamma(s))$ for some constant $c > 0$. This equality holds along curves of maximal slope (where $|\dot{\gamma}| = |\partial \Phi|$). Thus the infimum is achieved by gradient flow curves, yielding $\mathcal{L}(x) = \Phi(x) - \Phi_{\min}$ when $M = \{\arg\min \Phi\}$. The reconstruction of Metatheorems 6.7.1–6.7.3 follows by the same optimality arguments, with metric slopes replacing gradient norms throughout. $\square$

**Example 6.3.3 (Wasserstein space).** The heat equation $\partial_t \rho = \Delta \rho$ interpreted on $\mathcal{P}_2(\mathbb{R}^n)$:

| Component | Wasserstein Realization |
|:----------|:-----------------------|
| State space $X$ | $(\mathcal{P}_2(\mathbb{R}^n), W_2)$ |
| Height functional $\Phi$ | Boltzmann entropy $H(\rho) = \int \rho \log \rho \, dx$ |
| Dissipation $\mathfrak{D}$ | Fisher information $I(\rho) = \int |\nabla \log \rho|^2 \rho \, dx$ |
| Metric slope $|\partial \Phi|$ | $\sqrt{I(\rho)}$ |
| GC' verification | $\mathfrak{D} = I = |\partial \Phi|^2$ |

This instantiation extends the hypostructure framework to optimal transport and mean-field limits. See §15.1.1 for the complete correspondence between Axioms C, D, LS and the RCD curvature-dimension conditions.

**Example 6.3.4 (Discrete graphs).** A reversible Markov chain on a finite graph $(V, E)$ with stationary distribution $\pi$:

| Component | Discrete Realization |
|:----------|:--------------------|
| State space $X$ | Probability measures on $V$ |
| Metric $d$ | Discrete Wasserstein distance \cite{Maas11} |
| Height functional $\Phi$ | Relative entropy $H(\mu\|\pi) = \sum_v \mu(v) \log(\mu(v)/\pi(v))$ |
| Dissipation $\mathfrak{D}$ | Dirichlet form $\mathcal{E}(\sqrt{\mu/\pi})$ |
| GC' verification | Via discrete Otto calculus \cite{Maas11} |

This instantiation demonstrates the applicability of the hypostructure framework to inherently discrete systems without recourse to continuum limits.

#### 6.7.2 The action reconstruction principle

**Metatheorem 6.7.1 (Action Reconstruction).** Let $\mathcal{S}$ be a hypostructure satisfying Axioms (D), (LS), and (GC) on a metric space $(X, g)$. Then the canonical Lyapunov functional $\mathcal{L}(x)$ is explicitly the **minimal geodesic action** from $x$ to the safe manifold $M$ with respect to the **Jacobi metric** $g_{\mathfrak{D}} := \mathfrak{D} \cdot g$ (conformally scaled by the dissipation).

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

**Step 4: Optimality.** We show that flow paths minimize the Jacobi length among all paths with the same endpoints.

For any path $\gamma: [0, T] \to X$ from $x$ to $y \in M$, parametrized by arc length in the original metric (so $\|\dot{\gamma}\|_g = L/T$ where $L$ is the $g$-length), the Jacobi length is:
$$
\mathrm{Length}_{g_{\mathfrak{D}}}(\gamma) = \int_0^T \sqrt{\mathfrak{D}(\gamma(t))} \|\dot{\gamma}(t)\|_g \, dt.
$$

For a flow path $u(t)$ satisfying gradient consistency $\|\dot{u}\|_g = \sqrt{\mathfrak{D}(u)}$, Step 3 shows:
$$
\mathrm{Length}_{g_{\mathfrak{D}}}(u) = \int_0^T \mathfrak{D}(u(t)) \, dt = \mathcal{C}_T(x).
$$

To show this is minimal, consider any other path $\gamma$ connecting the same endpoints. The cost functional $\mathcal{C}(\gamma) = \int \mathfrak{D}(\gamma) dt$ satisfies:
$$
\mathcal{C}(\gamma) = \int_0^T \mathfrak{D}(\gamma(t)) \, dt \geq \mathcal{C}(u)
$$
because $u$ is a gradient flow trajectory, which minimizes cost by Theorem 6.6 (the Lyapunov functional $\mathcal{L}$ is constructed as minimal cost-to-go).

Since flow paths achieve both $\mathrm{Length}_{g_{\mathfrak{D}}} = \mathcal{C}$ (by gradient consistency) and minimize $\mathcal{C}$ (by the gradient flow property), they minimize the Jacobi length:
$$
\mathcal{L}(x) - \Phi_{\min} = \mathcal{C}(x \to M) = \inf_{\gamma: x \to M} \mathrm{Length}_{g_{\mathfrak{D}}}(\gamma) = \mathrm{dist}_{g_{\mathfrak{D}}}(x, M).
$$

**Step 5: Lyapunov property check.** Along a trajectory $u(t)$:
$$
\frac{d}{dt} \mathcal{L}(u(t)) = \frac{d}{dt} \mathrm{dist}_{g_{\mathfrak{D}}}(u(t), M) = -\sqrt{\mathfrak{D}(u(t))} \|\dot{u}(t)\|_g = -\mathfrak{D}(u(t)).
$$

This recovers the energy–dissipation identity exactly. Uniqueness follows from Axiom LS. $\square$

**Corollary 6.7.2 (Explicit Lyapunov from dissipation).** Under the hypotheses of Theorem 6.7.1, the Lyapunov functional is **explicitly computable** from the dissipation structure alone: no prior knowledge of an energy functional is required.

#### 6.7.3 The Hamilton–Jacobi generator

**Metatheorem 6.7.3 (Hamilton–Jacobi Characterization).** Let $\mathcal{S}$ be a hypostructure satisfying Axioms (D), (LS), and (GC) on a metric space $(X, g)$. Then the Lyapunov functional $\mathcal{L}(x)$ is the unique viscosity solution to the static **Hamilton–Jacobi equation**:
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

**Remark 6.7.4 (From guessing to solving).** Theorem 6.7.3 reduces the search for a Lyapunov functional to a well-posed PDE problem on state space. Given only $\mathfrak{D}$ and $M$, one solves the Hamilton–Jacobi equation to obtain $\mathcal{L}$.

---


---

## 7. Structural Resolution of Maximizers

### 7.1 The philosophical pivot

The reconstruction of an object from its representations is the dynamical realization of **Tannakian Duality \cite{DeligneMilne82}**, which asserts that a group can be reconstructed from its category of representations (the fiber functor). This principle underlies the Recovery Axiom throughout the framework.

Standard analysis often asks: *Does a global maximizer of the energy functional exist?* If the answer is "no" or "maybe," the analysis stalls.

The hypostructure framework inverts this dependency. We do not assume the existence of a global maximizer to define the system. Instead, we use **Axiom C (Compactness)** to prove that **if** a singularity attempts to form, it must structurally reorganize the solution into a "local maximizer" (a canonical profile).

Maximizers are treated not as static objects that *must* exist globally, but as **asymptotic limits** that emerge only when the trajectory approaches a finite-time singularity.

### 7.2 Formal definition: Structural resolution

We formalize the "Maximizer" concept via the principle of **Structural Resolution** (a generalization of Profile Decomposition).

**Definition 7.1 (Asymptotic maximizer extraction).** Let $\mathcal{S}$ be a hypostructure satisfying Axiom C. Let $u(t)$ be a trajectory approaching a finite blow-up time $T_*$. A **Structural Resolution** of the singularity is a decomposition of the sequence $u(t_n)$ (where $t_n \nearrow T_*$) into:
$$
u(t_n) = \underbrace{g_n \cdot V}_{\text{The Maximizer}} + \underbrace{w_n}_{\text{Dispersion}}
$$
where:

1. **$V \in X$ (The canonical profile):** A fixed, non-trivial element of the state space. This is the "Maximizer" of the local concentration.
2. **$g_n \in G$ (The Gauge Sequence):** A sequence of symmetry transformations (scalings, translations) that diverge as $n \to \infty$ (e.g., $\lambda_n \to \infty$ for scaling).
3. **$w_n$ (The Residual):** A term that vanishes or disperses in the relevant topology (structurally irrelevant).

**Remark 7.2 (Forced structure).** We do not assume $V$ exists *a priori*.
- If the sequence $u(t_n)$ disperses (Mode D.D), then $V$ does not exist—**no singularity forms**. The solution exists globally via scattering.
- If the sequence concentrates, blow-up **forces** $V$ to exist. We then check permits on the forced structure.

**Remark 7.3 (No global compactness required).** A common misconception is that one must prove global compactness to use this framework. This is false:
- Mode D.D (dispersion) is **global existence**, not a singularity to be excluded.
- When concentration does occur, structure is forced—no compactness proof needed.
- The framework checks algebraic permits on the forced structure.

The two-tier logic:

1. **Tier 1 (Dispersion):** If energy disperses, no singularity forms—global existence via scattering.
2. **Tier 2 (Concentration):** If energy concentrates, check algebraic permits on the forced structure. Permit denial yields regularity via contradiction.

### 7.3 The taxonomy of maximizers

Once Axiom C extracts the profile $V$, the hypostructure framework classifies it. The "Maximizer" $V$ falls into one of two categories:

**Type A: The Safe Maximizer ($V \in M$).**
The profile $V$ lies in the **safe manifold** (e.g., a soliton, a ground state, or a vacuum state).
- **Mechanism:** The trajectory converges to a regular structure (soliton, ground state).
- **Outcome:** **Axiom LS (Stiffness)** applies. The trajectory is constrained near $M$. Since elements of $M$ are global solutions with infinite existence time, this is not a singularity; it is **Soliton Resolution**.

**Type B: Non-safe profile ($V \notin M$).**
The profile $V$ is a self-similar blow-up profile or a high-energy bubble that is *not* in the safe manifold.
- **Mechanism:** The system is attempting to construct a Type II blow-up.
- **Outcome:** The **algebraic permits** apply. We do not need to analyze the PDE evolution of $V$. We only need to check whether $V$ can satisfy the scaling and capacity permits.

### 7.4 Admissibility tests

This is where the framework replaces hard analysis with algebra. We test the non-safe profile $V$ against the structural axioms.

**Test 1: Scaling Admissibility.**
Even if $V$ is a valid profile, it must be generated by the gauge sequence $g_n$ (specifically the scaling $\lambda_n \to \infty$).
By **Axiom SC** and **Theorem 6.2 (Property GN)**:
$$
\text{Cost of Generating } V \sim \int (\text{Dissipation of } g_n \cdot V)
$$

- If the scaling exponents satisfy $\alpha > \beta$ (Subcriticality), the cost of generating *any* non-trivial non-safe profile via scaling is **infinite**.
- **Result:** The non-safe profile $V$ is excluded. It cannot be formed from finite energy.

**Test 2: Capacity Admissibility.**
If $V$ is supported on a "thin" set (e.g., a singular filament with dimension $< Q$):
- By **Axiom Cap** and **Theorem 6.3**, the time available to create such a profile goes to zero faster than the profile can form.
- **Result:** The non-safe profile is excluded by geometric constraints.

### 7.5 The regularity logic flow

The framework proves regularity without assuming any structure exists *a priori*:

**Tier 1: Does blow-up attempt to form?**
- **NO (Energy disperses):** Mode D.D—global existence via scattering. No singularity forms.
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

### 7.6 Implementation guide

When instantiating the framework for a specific system, one does not search for the global maximizer of the functional. The procedure is as follows:

**Step 1: Identify the Symmetry Group $G$.**
For example: Scaling $\lambda$, Translation $x_0$.

**Step 2: Understand the forced structure.**
Observe that if blow-up occurs with bounded energy, concentration is forced. When energy concentrates, Profile Decomposition (standard for most PDEs) ensures a canonical profile $V$ emerges modulo $G$. You do not need to prove compactness globally—concentration is forced by blow-up.

**Step 3: Compute Exponents $(\alpha, \beta)$.**
- $\mathfrak{D}(\mathcal{S}_\lambda u) \approx \lambda^\alpha \mathfrak{D}(u)$
- $dt \approx \lambda^{-\beta} ds$

**Step 4: The Check.**
Is $\alpha > \beta$?
- **Yes:** Then **Theorem 6.2** guarantees that *whatever* the profile $V$ extracted in Step 2 is, it cannot sustain a Type II blow-up. The non-safe profile is structurally inadmissible.

**Remark 7.4 (Decoupling existence from admissibility).** The hypostructure framework decouples the *existence* of singular profiles from their *admissibility*. We do not require the existence of a global maximizer to define the theory. Instead, Axiom C ensures that if a singularity attempts to form via concentration, a local maximizer (canonical profile) must emerge asymptotically. Axiom SC then evaluates the scaling cost of this emerging profile. If the cost is infinite (GN), the profile is forbidden from materializing, regardless of whether a global maximizer exists for the static functional.
