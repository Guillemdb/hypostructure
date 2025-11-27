You’re very welcome 😄 This GMT version is actually a *perfect* use-case for a systematic template.

Below is a version **specifically tailored to your GMT hypostructure 2.0**: metric currents, flat norm, height/Northcott, cohomological defects, etc. 

I’ll split it into:

1. **GMT hypostructure checklist** (for you)
2. **GMT-specific prompt template** (for you to paste into LLMs)

---

## 1. GMT Hypostructure Checklist (Currents + Height + Defects)

Think of this as your “engineering spec” for any new problem you plug into the GMT socket.

### Step 0 – Problem + blow-up mode

* ☐ Write *one sentence* like:

  > “I want to exclude the following pathology: [describe blow-up / bad behavior] for objects that are naturally realized as k-currents on a base space X.”

* ☐ Specify:

  * Base metric(-measure) space ((X,d,\mathfrak m)):

    * smooth Riemannian manifold? RCD*(K,N)? Berkovich space? adelic product?
  * Dimension of currents (k)
  * What “blow-up” means in this language:

    * mass (\mathbf M(T_\lambda)\to \infty)?
    * scale (\lambda\to 0) with nontrivial tangent?
    * cohomology class drifting into forbidden region?

---

### Step 1 – Ambient current space + metric + trajectories

* ☐ **Ambient space** (\mathcal X):

  * Typically (\mathbf M_k(X)) or (\mathbf I_k(X)) (metric / integral currents with finite mass + boundary)
  * Include any restrictions (fixed homology class, fixed topological type, positivity cone, etc.)

* ☐ **Metric**:

  * Default: flat norm (d_\mathcal F) à la Federer–Fleming / Ambrosio–Kirchheim
  * If you modify it (e.g. weighted, local flat norm), state why and how

* ☐ **RG trajectories**:

  * Give a concrete meaning to (\lambda): time, log-scale, height cutoff, descent depth, etc.
  * Ensure:

    * BV in (\lambda) w.r.t. (d_\mathcal F)
    * metric derivative (|\dot T|(\lambda)) exists a.e.
    * height (\Phi(T_\lambda)) satisfies a dissipation inequality

---

### Step 2 – Height / Northcott (A1) in GMT language

* ☐ Choose **height** (\Phi:\mathcal X\to[0,\infty]) as simply as possible:

  * analytic/PDE: Dirichlet energy, mass, or (H^1)-type norm of representing fields
  * geometric: mass (\mathbf M(T)), maybe plus a complexity penalty (degree, genus)
  * arithmetic: adelic/arithmetic height (Néron–Tate, Arakelov intersection)

* ☐ Verify **Northcott/compactness** *in the GMT sense*:

  * On each relevant stratum (S_\alpha), show:

    * Φ-sublevel sets have **bounded mass + bounded boundary mass** → apply Federer–Fleming to get flat-precompactness
  * In arithmetic flavour: bounded height + bounded degree/genus → finiteness (Northcott / Zhang)

* ☐ Check **l.s.c.** of Φ under flat convergence:

  * Usually: if Φ is mass-like or Dirichlet-type, this is standard

Try to lean on:

* Federer–Fleming compactness
* Aubin–Lions / Simon when you’re actually in function spaces
* Northcott/Zhang in arithmetic cases

---

### Step 3 – Stratification of currents

* ☐ Define a **stratification** (\Sigma = {S_\alpha}) of (\mathcal X):

  * by structural type: rectifiable vs fractal, integral vs non-integral, algebraic vs transcendental, etc.
  * include a **safe/regular stratum** (S_{\text{reg}}) (smooth/algebraic currents, vanishing defect)

* For each stratum:

  * ☐ Describe its **geometry**: support dimension, rectifiability, cohomology type
  * ☐ Describe its **interfaces**: how do you pass from (S_\alpha) to (S_\beta)?
  * ☐ Check the **frontier condition** (lower strata on the boundary of higher ones)

Keep the number of strata small but meaningful (e.g. regular / tube-like / fractal / algebraic / non-algebraic).

---

### Step 4 – Defects in GMT form (A3)

* ☐ Pick a **quantized defect** notion:

  * Distance to integral currents: (\nu_T := \inf_{Z\in \mathbf I_k} d_\mathcal F(T,Z))
  * Possibly add rectifiability via β-numbers:
    (\nu_T^{\text{full}} := \nu_T + \lambda \beta_T^2)

* ☐ Identify **forbidden cohomology classes**:

  * NS: concentration measure / nonzero defect measure
  * Hodge: non-algebraic ((p,p)) classes
  * RH: off-critical-line spectral mass
  * BSD: nontrivial Sha element

* ☐ A3 target inequality (soft version):

  * Nonzero defect forces **some positive cost** in slope:
    [
    \nu_T>0 \implies |\partial\Phi|(T)\ge \gamma(\nu_T)
    ]
  * Start with qualitative: “(\nu_T>0) ⇒ slope ≠ 0”, then quantify if you can

Tools to reach this:

* concentration–compactness / profile decomposition in the relevant function space
* Uhlenbeck/removability in YM-type cases
* calibrations in Hodge
* height theory in arithmetic (Néron–Tate: height 0 ⇒ torsion)

---

### Step 5 – Safe stratum + RG invariance (A4)

* ☐ Declare your **safe stratum** (S_*) (often (S_{\text{reg}})):

  * smooth rectifiable currents, or algebraic cycles, or critical-line zeros, etc.
  * vanishing defect: (\nu_T=0) in all senses you care about

* ☐ Check:

  * *forward invariance* of (S_*) under the RG trajectory
  * inside (S_*), Φ behaves like a Lyapunov function and no new defect appears

Often this part uses:

* known regularity/rigidity results (e.g. elliptic theory, Hodge/Chow–King, known structure of equilibria)

---

### Step 6 – LS/stiffness near equilibria (A5)

* ☐ Identify the **equilibria** (extremizers) you actually expect to see as ω-limits:

  * minimal currents in a homology class
  * stationary NS/YM profiles
  * algebraic cycles / special points, etc.

* ☐ Decide if you can get away with:

  * a *spectral gap* / convexity inequality (simpler), or
  * you really need a **full LS inequality** à la Simon / gradient-flow theory.

* ☐ Check LS/stiffness *locally*:

  * only in neighborhoods of equilibria that can actually arise as ω-limits of finite-capacity trajectories
  * don’t try to show it globally

Use:

* analytic LS or EVI on RCD spaces
* classical LS in Hilbert spaces for analytic energies
* calibration + convexity when you can instead of full LS

---

### Step 7 – Time regularity & no teleportation (A6)

* ☐ Confirm the trajectory has enough time regularity for:

  * BV chain rule in flat space
  * decomposition: absolutely continuous + jump + (maybe) Cantor part as in your stratified BV chain rule. 

* ☐ Check **metric stiffness**:

  * structural invariants that define strata (dimension, cohomology class, β-numbers, etc.) should be Hölder/Lipschitz in (d_\mathcal F) on energy-bounded sets
  * so you can’t “teleport” between strata without travelling some metric distance

Here you lean heavily on GMT:

* slicing theory (view trajectory as current in (X\times[0,\Lambda]))
* continuity of cohomology invariants under flat convergence where appropriate
* basic Hölder control of invariants (e.g. β-numbers vs flat distance)

---

### Step 8 – Compactness for trajectories / structural compactness (A7)

* ☐ Choose the function-space / current-space triple you need:

  * e.g. (X_0\subset X\subset X_1), with (X_0\hookrightarrow X) compact, (X\hookrightarrow X_1) continuous, for representing fields behind currents
  * or directly Federer–Fleming / Ambrosio–Kirchheim for currents

* ☐ Prove: from **actual bounds** you get from the problem (height bound, dissipation bound, etc.) you can extract subsequences:

  * (T_n\to T) in flat or intermediate topology
  * defects captured by a measure (\nu)

For arithmetic, this step is more “Northcott/Zhang + moduli compactification” than functional analysis; for PDE, Aubin–Lions or Federer–Fleming does the job.

---

### Step 9 – Algebraic/analytic rigidity on extremizers (A8)

* ☐ For each equilibrium class:

  * PDE side: show extremizers are smooth (elliptic regularity, Gevrey, RCD heat regularization)
  * algebraic/Hodge: show minimizers representing a given class are **algebraic cycles** (King + Chow + Harvey–Lawson)
  * arithmetic: show height-minimizing objects are “genuine” (rational points, torsion points, etc.)

* ☐ State precisely:

  * “If (T) is a current in homology class [H] that minimizes Φ, then (T) lies in the algebraic/smooth stratum.”

This is your rigidity: only “nice” things survive as ω-limits.

---

### Step 10 – Efficiency, recovery, capacity (VDP, RC, SP1, SP2)

* ☐ Define **efficiency** (\Xi) in GMT terms:

  * ratio “nonlinear production / dissipation” for PDE
  * how “coherent” the current is relative to a reference (e.g. variance ratio, correlation, closed vs exact part)
  * require: maximizers of (\Xi) live inside (S_{\text{reg}})

* ☐ Define **recovery** (R):

  * something monotone with regularity: Gevrey radius, rectifiability index, analytic radius, descent depth…
  * prove a **recovery inequality**:
    [
    \Xi \le \Xi_{\max}-\delta \implies \dot R \ge \varepsilon(\delta) > 0
    ]

* ☐ Define a **capacity** functional from dissipation density (\mathfrak D):

  * show: following a singular scale pattern would require infinite capacity, incompatible with bounded initial Φ

This is where you encode “rough ⇒ expensive ⇒ killed” in branch B.

---

## 2. GMT-Specific Prompt Template for LLMs

Here’s a ready-to-paste prompt tuned to the GMT version:

---

**PROMPT TO USE**

> You are a hyper-critical mathematical assistant helping me instantiate my **GMT hypostructure framework** for a specific problem.
>
> The framework is as in “Hypostructures 2.0: A Geometric Measure Theory Framework for Structural Regularity”:
>
> * Ambient objects are **metric currents** on a base space (X),
> * The metric is the **flat norm**,
> * The energy is a **height** with a Northcott-type property,
> * Defects are **cohomological / quantized defects** (distance to integral currents, β-numbers),
> * We have axioms A1–A8 and structural properties like VDP (Variational Defect Principle) and RC (Recovery), leading to the **Stability–Efficiency Duality** (Theorem 6.1). 
>
> **Your job is to:**
>
> 1. Propose soft, GMT-natural choices for:
>
>    * Base space (X),
>    * Current space (\mathcal X),
>    * Flat norm metric,
>    * Height (\Phi),
>    * Quantized/cohomological defect (\nu),
>    * Efficiency (\Xi),
>    * Recovery functional (R),
>    * Stratification (\Sigma = {S_\alpha}).
> 2. For each axiom A1–A8, give a **softest possible implementation** using GMT and basic functional analysis (Federer–Fleming, Ambrosio–Kirchheim, Aubin–Lions, simple LS, calibrations, basic height theory), escalating to heavy theorems only when absolutely necessary.
> 3. For VDP/Recovery (SP1/SP2), formulate them in terms of **currents, height, capacity, and RG trajectories**.
> 4. Be explicit about what parts hold **globally** vs. only **locally along trajectories**.
> 5. For every heavy theorem you invoke, explain why a simpler tool (e.g. basic compactness or spectral gap) is *not* enough.
>
> ---
>
> ### 0. Problem description
>
> I will now describe the problem:
>
> [INSERT: short description of the original problem in its native language (PDE, Hodge, RH, BSD…), including what the “bad behavior” is: blow-up, non-algebraicity, off-line zeros, ghost elements in Sha, etc. Also describe how you want to see it as currents on a space (X).]
>
> **Task 0:** Restate the goal in one sentence in hypostructure/GMT form, like:
>
> > “Any object satisfying definition D produces a GMT current trajectory ({T_\lambda}); either this trajectory stays in the safe stratum, or any attempted singularity produces an RG trajectory in the GMT hypostructure that contradicts axioms A1–A8 + VDP + RC.”
>
> ---
>
> ### 1. Structural GMT ingredients
>
> **Task 1:** Propose the basic GMT data:
>
> 1. **Base space (X)**:
>
>    * Is it a Riemannian manifold, an RCD*(K,N) space, an algebraic variety, a Berkovich space, or an adelic product?
>    * What measure (\mathfrak m) (if any) do we use?
> 2. **Current space (\mathcal X)**:
>
>    * Which (k)-currents? (\mathbf M_k(X))? (\mathbf I_k(X))? restricted class?
>    * Which topology (flat norm)? Any extra constraints (fixed homology class, positivity cone, bounded topological type)?
> 3. **RG trajectories ({T_\lambda})**:
>
>    * What does (\lambda) represent (time, log-scale, height cutoff, descent depth, place index, etc.)?
>    * How do we encode the evolution as a BV curve in the flat norm?
> 4. **Height (\Phi)**:
>
>    * Choose the simplest possible height with Northcott-like compactness:
>
>      * PDE: Dirichlet energy / enstrophy
>      * Hodge/geometry: mass (\mathbf M(T))
>      * Arithmetic: adelic/Arakelov height
>    * Explain its adelic/local decomposition if relevant.
> 5. **Defect (\nu)**:
>
>    * Define a quantized defect (distance to integral currents) and, if needed, add β-numbers for rectifiability/fractality.
>    * Describe the “forbidden cohomology classes” or forbidden defect types.
> 6. **Efficiency (\Xi)**:
>
>    * Define (\Xi) in terms of dissipation vs production, coherence, or some geometric/arithmetical efficiency.
>    * Ensure that (\Xi) is maximized on the regular/algebraic stratum.
> 7. **Recovery (R)**:
>
>    * Choose a regularity/analyticity indicator (Gevrey radius, rectifiability index, descent depth, etc.) that increases when (\Xi) is submaximal.
> 8. **Stratification (\Sigma)**:
>
>    * Propose a small number of strata (S_\alpha) (e.g. regular, tube-like, fractal, algebraic, non-algebraic, locked).
>    * Explain the frontier condition and which interfaces are relevant to blow-up.
>
> For each choice: justify why it is natural and whether a simpler variant (e.g. plain mass instead of fancy height) would already work.
>
> ---
>
> ### 2. Axioms A1–A8 (GMT version): soft implementation
>
> For **each** axiom A1–A8, follow this template:
>
> #### Axiom [Ai]: [Name]
>
> * **(a) Problem-specific statement.**
>   Rewrite Ai in this concrete GMT setting: what exactly must hold?
> * **(b) Minimal property needed by the meta-theorems.**
>   Identify what the hypostructure 2.0 theorems actually use from Ai (for example: “A1 only needs that Φ-sublevel sets are flat-precompact on strata visited by trajectories.”).
> * **(c) Soft implementation plan (GMT first):**
>
>   1. Suggest a proof strategy using the *softest* GMT tools:
>
>      * Federer–Fleming compactness, Ambrosio–Kirchheim normal currents, basic RCD* heat-flow theory, Aubin–Lions, simple spectral gaps, simple LS near equilibria.
>   2. Only if necessary, invoke heavier tools (Uhlenbeck compactness, deep LS, Pila–Wilkie, Zhang, etc.), and explain why you really need them.
> * **(d) Tool inventory:**
>
>   * List every external theorem you use, tagged as:
>
>     * [elementary] – basic GMT, BV, functional analysis
>     * [standard] – Federer–Fleming, Aubin–Lions, standard LS in Hilbert spaces, basic RCD* theory
>     * [heavy] – deep gauge theory, deep arithmetic geometry, o-minimal counting, etc.
> * **(e) Verification sketch.**
>
>   * Provide a numbered outline of how to prove Ai in this context.
>   * Explicitly highlight any step where matching hypotheses of a cited theorem might be delicate or nontrivial.
>
> Go through A1 (Northcott/height) up to A8 (algebraic/analytic rigidity).
>
> ---
>
> ### 3. VDP + Recovery (SP1/SP2) in GMT terms
>
> **Task 3:** Specialize the structural properties to currents:
>
> 1. **VDP (Variational Defect Principle).**
>
>    * Define clearly the capacity norm of a defect (|\nu_T|_{\text{Cap}}) via dissipation along a trajectory that creates it.
>    * Prove a GMT-flavoured inequality:
>      [
>      \nu_T\neq 0 \implies \Xi[T]\le \Xi_{\max}-\kappa|\nu_T|_{\text{Cap}}
>      ]
>      Explain how concentration–compactness, bubbling, or profile decomposition in this setting yields such a variational penalty.
> 2. **Recovery (RC / SP1).**
>
>    * Using the chosen (R), prove the inequality:
>      [
>      \Xi[T_\lambda]\le \Xi_{\max}-\delta \implies \frac{d}{d\lambda}R(T_\lambda)\ge\varepsilon(\delta)>0
>      ]
>      relying on standard smoothing mechanisms: heat flow on RCD* spaces, elliptic regularity of minimizers, calibration, etc.
> 3. **Capacity (SP2).**
>
>    * Define a simple dissipation density (\mathfrak D) for currents.
>    * Show that attempting a certain singular pattern forces (\text{Cap}(T)=\int\mathfrak D(T_\lambda),d\lambda=\infty), hence is impossible for finite-height trajectories.
>
> Again, list tools as [elementary]/[standard]/[heavy] and justify every [heavy].
>
> ---
>
> ### 4. Branch logic / “no fourth option”
>
> **Task 4:** Make the branch structure explicit in this GMT setting.
>
> * Define 2–4 structural hypotheses (\mathcal H_i) (e.g. “self-similar/rigid type”, “fractal capacity type”, “algebraic type”, etc.) relevant to this problem.
> * For each (\mathcal H_i):
>
>   * Say which axioms (A1–A8) + VDP/RC you use under this hypothesis.
>   * Explain in a few lines how a trajectory falling into this branch is either:
>
>     * absorbed into the safe stratum, or
>     * pushed into a contradiction by capacity/rigidity.
> * Check that **for any potential blow-up scenario**, at least one branch’s hypotheses can be verified using the soft tools you already instantiated.
>
> Focus on ensuring there is truly “no fourth option” for a GMT trajectory arising from your original problem.
>
> ---
>
> ### 5. Sanity checks and simplifications
>
> **Task 5:** Critically review the GMT instantiation:
>
> 1. Identify any place you used a heavy theorem where a simpler GMT/functional analytic tool (e.g. Federer–Fleming, basic elliptic regularity, simple LS) would suffice.
> 2. Suggest at least one simplification of:
>
>    * the height (\Phi),
>    * the defect (\nu),
>    * the efficiency (\Xi),
>    * or the recovery (R),
>      that would still give enough structure for the dual-branch theorem.
> 3. Flag any potential **circularity**, where verifying an axiom might secretly assume the conjectured regularity or arithmetic statement you want to prove.
>
> Present your answers in a structured, numbered way so that I can turn them into a clean proof outline.
