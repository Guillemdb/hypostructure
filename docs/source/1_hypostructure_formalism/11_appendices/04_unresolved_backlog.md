# Framework Extensions and Unresolved Obligations Register

This registry records open proof obligations that are outside the current execution template and therefore cannot be discharged by existing gate/barrier/surgery machinery alone.

For framework-extension planning with exact insertion targets, use:
`09_mathematical/08_unresolved_external_extensions.md`.

The items below are only included when a dataset proof run has left a residual `K^{inc}`/`K^{\mathrm{morph}}` status in the goal dependency cone or left a required analytic bridge as an explicit external hypothesis.

## Scope

- Dataset folder reviewed: `docs/source/dataset/*.md`
- Current framework slice: `docs/source/1_hypostructure_formalism/template.md` and supporting node/interface modules
- Criteria: an issue is listed here only if the missing step is not already represented as an existing framework contract, upgrade rule, or bridge witness in the current documents.

## 1) Core PDE-Geometry Gaps

### 1.1 Riemann Hypothesis

- **Dataset file:** `docs/source/dataset/riemann_hypothesis.md`
- **Outstanding obligation:** global spectral model linking zeros of $\zeta(s)$ to a self-adjoint Hamiltonian (`OBL-1`).
- **Why template cannot close it:** the template can record `K_{D_E}^+`, `K_{\mathrm{Rec}_N}`-style diagnostics, and Lock/upgrade steps, but does not provide the external spectral construction itself.
- **Where to add:** a dedicated appendix theorem in `09_mathematical/01_theorems.md` (spectral/trace machinery) plus an explicit bridge certificate in `11_appendices/01_zfc.md` for the final import.

### 1.2 Quantum Gravity Boundary Reconstruction

- **Dataset file:** `docs/source/dataset/quantum_gravity.md`
- **Outstanding obligation:** completion of the open boundary/open-system period chain used for the declared regime.
- **Why template cannot close it:** Node 13--16 routing plus Lock execution are instantiated, but the key boundary/causality reconstruction is currently declared as non-final.
- **Where to add:** `04_nodes/03_surgery_nodes.md` (new boundary-aware reconstruction node) and supporting analytic lemma in `09_mathematical/01_theorems.md`.

### 1.3 Landau damping regularity branch

- **Dataset file:** `docs/source/dataset/landau_damping.md`
- **Outstanding obligation:** low-regularity (Sobolev) persistence gap beyond Gevrey; sector-dependent closing status remains unresolved.
- **Why template cannot close it:** the current route certifies the Gevrey branch via viscosity/analytic mechanisms while the Sobolev branch requires additional damping and mixing controls not yet represented in route-level permits.
- **Where to add:** `09_mathematical/04_taxonomy.md` (sector-tagged family classification), `08_upgrades/01_instantaneous.md` or `08_upgrades/02_retroactive.md` once a route-level promotion is proven.

## 2) Algebraic/Arithmetic Core Conjecture Gaps

### 2.1 Collatz Problem

- **Dataset file:** `docs/source/dataset/collatz.md`
- **Outstanding obligations:** `OBL-1` (global finiteness of stopping time), `OBL-3` (global exclusion certificate for bad trajectories).
- **Why template cannot close it:** the run records `K_{\mathrm{Rec}_N}^{inc}` and goal-cone placement correctly; no existing gate-level certificate gives an unconditional closure for the global recurrence claim.
- **Where to add:** `09_mathematical/04_taxonomy.md` (dependency-class support for arithmetic dynamics) with a companion theorem package in `09_mathematical/02_algebraic.md` (if a genuine recurrence theorem is available).

### 2.2 Kervaire Invariant

- **Dataset file:** `docs/source/dataset/kervaire_invariant.md`
- **Outstanding obligation:** exceptional $j=6$ residue class (`OBL-126`).
- **Why template cannot close it:** this case is already marked as unresolved in the existing route and stays outside the framework's current locked finite-obstruction closure.
- **Where to add:** `09_mathematical/02_algebraic.md` and `08_upgrades/02_retroactive.md` only if a finite-dimensional obstruction refinement is available.

### 2.3 Birch and Swinnerton-Dyer

- **Dataset file:** `docs/source/dataset/bsd_conjecture.md`
- **Outstanding obligation:** extension from low-rank cases to full analytic rank statements (`HORIZON`).
- **Why template cannot close it:** the template supports conditional tracking but does not supply missing number-theoretic descent or $L$-function rank equivalence.
- **Where to add:** `09_mathematical/02_algebraic.md` with an explicit "Global BSD completion" lemma set and corresponding bridge in `11_appendices/01_zfc.md`.

### 2.4 Langlands Program

- **Dataset file:** `docs/source/dataset/langlands.md`
- **Outstanding obligation:** full number-field correspondence (`OBL-LANG-1`).
- **Why template cannot close it:** current runs can certify the route-level formalism and record `K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{inc}}`, but the missing bridge is a domain theorem, not a template upgrade.
- **Where to add:** `09_mathematical/02_algebraic.md` and `09_mathematical/03_cross_reference.md` as a named backend completion package.

### 2.5 Yang–Mills Mass Gap

- **Dataset file:** `docs/source/dataset/yang_mills.md`
- **Outstanding obligations:** `OBL-YM-OS`, `OBL-YM-GAP`, `OBL-YM-CLUST`.
- **Why template cannot close it:** route-level certificates are present and correctly isolated as auxiliary obligations; the cluster limit/OS construction remains non-template external input.
- **Where to add:** `09_mathematical/02_algebraic.md` (quantum-classical crossover lemmas) plus `09_mathematical/05_algorithmic.md` if cluster-state obstructions are made constructive.

## 3) Complexity/Semidecidable Gaps

### 3.1 P versus NP

- **Dataset file:** `docs/source/dataset/p_vs_np.md`
- **Outstanding obligation:** certified separation at the classical arithmetic/logical level (morphism-level obstruction remains open).
- **Why template cannot close it:** the framework can represent algorithmic compression barriers and the Lock route, but a global complexity separation theorem is exactly the domain statement being tracked.
- **Where to add:** `09_mathematical/06_complexity_bridge.md` for a dedicated theorem schema and `09_mathematical/05_algorithmic.md` for any required bridge axioms to become named permits.

### 3.2 Halting Problem

- **Dataset file:** `docs/source/dataset/halting_problem.md`
- **Outstanding obligation:** axiomatically horizon-classified undecidability and its interaction profile in the framework.
- **Why template cannot close it:** the framework correctly expresses `K_{\mathrm{Cat}_{\mathrm{Hom}}}`-relative diagnostics but does not prove undecidability from internal certificates.
- **Where to add:** `09_mathematical/05_algorithmic.md` and `11_appendices/01_zfc.md` under a dedicated "undecidable core" section.

### 3.3 Stochastic Einstein–Boltzmann

- **Dataset file:** `docs/source/dataset/stochastic_einstein_boltzmann.md`
- **Outstanding obligation:** DPI/thermodynamic and censorship closure (`OBL-SEB-1`).
- **Why template cannot close it:** template diagnostics isolate the gap cleanly, but the step is currently non-mathematical unless a named entropy or censorship lemma is introduced.
- **Where to add:** `09_mathematical/01_theorems.md` (entropy-causality bridge) with optional `09_mathematical/04_taxonomy.md` annotation in a Family-IV extension class.

## 4) Resolution Rule

Each unresolved obligation above must be introduced as an explicit framework object:

1. **Definition/contract** in the relevant `09_mathematical/*` module.
2. **Permit-level witness format** in `05_interfaces/*` only if required by the sieve node.
3. **Upgrade closure entry** (if promotable) in `08_upgrades/*`.
4. **Bridge entry** in `11_appendices/01_zfc.md` only after the named certificate is established.

No file in `docs/source/dataset` should claim classical proof completion while this backlog item remains unresolved for a goal-dependent certificate.

## Last updated

Date: 2026-04-14
