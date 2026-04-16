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

### 1.1 Quantum Gravity Boundary Reconstruction

- **Dataset file:** `docs/source/dataset/quantum_gravity.md`
- **Outstanding obligation:** completion of the proper-horizon boundary/open-system period chain used for the declared regime.
- **Why template cannot close it:** Node 13--16 routing plus Lock execution are instantiated, but the key boundary/causality reconstruction is currently declared as non-final and remains a genuine horizon obstruction.
- **Where to add:** `04_nodes/03_surgery_nodes.md` (new boundary-aware reconstruction node) and supporting analytic lemma in `09_mathematical/01_theorems.md`.

## 2) Algebraic/Arithmetic Core Conjecture Gaps

### 2.1 Langlands Program

- **Dataset file:** `docs/source/dataset/langlands.md`
- **Outstanding obligation:** full number-field correspondence (`OBL-LANG-1`).
- **Why template cannot close it:** current runs can certify the route-level formalism and record `K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{inc}}`, but the missing bridge is a domain theorem, not a template upgrade.
- **Where to add:** `09_mathematical/02_algebraic.md` and `09_mathematical/03_cross_reference.md` as a named backend completion package.

### 2.2 Yang–Mills Mass Gap

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
- **Outstanding obligation:** proper-horizon undecidability and its interaction profile in the framework.
- **Why template cannot close it:** the framework correctly expresses `K_{\mathrm{Cat}_{\mathrm{Hom}}}`-relative diagnostics but does not collapse a genuine horizon branch into classical decidability.
- **Where to add:** `09_mathematical/05_algorithmic.md` and `11_appendices/01_zfc.md` under a dedicated "undecidable core" section.

## 4) Resolution Rule

Each unresolved obligation above must be introduced as an explicit framework object:

1. **Definition/contract** in the relevant `09_mathematical/*` module.
2. **Permit-level witness format** in `05_interfaces/*` only if required by the sieve node.
3. **Upgrade closure entry** (if promotable) in `08_upgrades/*`.
4. **Bridge entry** in `11_appendices/01_zfc.md` only after the named certificate is established.

No file in `docs/source/dataset` should claim classical proof completion while this backlog item remains unresolved for a goal-dependent certificate.

## Last updated

Date: 2026-04-15
