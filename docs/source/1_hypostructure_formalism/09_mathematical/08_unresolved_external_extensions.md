# External Obligations Beyond Template Closure

Date: 2026-04-14

This register tracks unresolved obligations that remain outside current template closure: each item is not discharged by the 17-node sieve gates/barriers/surgery chain plus existing upgrade and bridge rules as currently represented in the framework text.

All entries below are exact unresolved items from dataset audit traces that can still be converted into explicit framework objects (definitions/contracts/upgrade lemmas/bridge imports) by extending `09_mathematical`, `08_upgrades`, and/or `11_appendices`.

## Inclusion Rule

- Add an entry here only when the unresolved goal is outside existing contract coverage for the active route in its current template execution.
- Do not classify a classical open problem as solved here unless the exact goal cone obligation is removed.
- Each entry records:
  1. dataset file and section anchor
  2. exact unresolved obligation
  3. where to add the certificate family in the framework

## Registry Entries

| Problem | Unresolved obligation | Template limit reached | Framework insertion point |
|---------|----------------------|-----------------------|--------------------------|
| `docs/source/dataset/collatz.md` | `OBL-1` (global finiteness of stopping time) and `OBL-3` (global exclusion of bad recurrence class) | route remains with `K_{\mathrm{Rec}_N}^{inc}` and `K_{\mathrm{Cap}_H}^{inc}` in goal cone | `09_mathematical/02_algebraic.md` (arithmetic dynamics primitives), `09_mathematical/04_taxonomy.md` (arithmetic recurrence class), `11_appendices/01_zfc.md` (bridge import when completed) |
| `docs/source/dataset/riemann_hypothesis.md` | `OBL-1` (Hilbert–Pólya type self-adjoint spectral model and verified trace identity input) | Lock remains `K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}` for the spectral correspondence step | `09_mathematical/01_theorems.md` (spectral bridge), `11_appendices/01_zfc.md` (explicit import block) |
| `docs/source/dataset/kervaire_invariant.md` | `OBL-126` (remaining $j=6$ class in dimension 126) | family obstruction remains unresolved in current template dependency cone | `09_mathematical/02_algebraic.md` (equivariant stable homotopy obstruction package), `08_upgrades/02_retroactive.md` (if a finite-dimensional refinement becomes available) |
| `docs/source/dataset/bsd_conjecture.md` | `OBL-BSD-GEN` (general rank BSD) | rank $0/1$ discharged, higher rank remains in horizon branch | `09_mathematical/02_algebraic.md` (global BSD completion theorems), `11_appendices/01_zfc.md` (global rank import), `09_mathematical/04_taxonomy.md` |
| `docs/source/dataset/langlands.md` | `OBL-LANG-1` (global number-field correspondence block) | `K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}` persists at Lock | `09_mathematical/02_algebraic.md` (reciprocity/correspondence theorems), `09_mathematical/03_cross_reference.md` (bidirectional bridge bookkeeping) |
| `docs/source/dataset/yang_mills.md` | `OBL-YM-OS`, `OBL-YM-GAP`, `OBL-YM-CLUST` (OS reconstruction, non-perturbative mass-gap, clustering) | lock route is blocked by unresolved analytic/morphological input | `09_mathematical/02_algebraic.md` (quantum-classical interface), `09_mathematical/05_algorithmic.md` (constructive cluster obstruction formalization) |
| `docs/source/dataset/halting_problem.md` | Horizon undecidability branch | framework records frontier horizon certificate and does not export decidability in classical form | `09_mathematical/05_algorithmic.md` (axiomatic undecidability schema), `11_appendices/01_zfc.md` (axiom/rule statement) |
| `docs/source/dataset/quantum_gravity.md` | `OBL-QG-1` (boundary/censorship reconstruction in framework class) | classified as Period VI/HORIZON after lock boundary obligations persist | `09_mathematical/01_theorems.md` (entropy/causality bridging lemmas), `04_nodes/03_surgery_nodes.md` (new boundary-aware reconstruction node if formalized) |
| `docs/source/dataset/p_vs_np.md` | `OBL-PNP` (global separation in the classical ZFC formulation) | route records `K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}` plus residual export obligation | `09_mathematical/06_complexity_bridge.md` (hardness separation schema), `09_mathematical/05_algorithmic.md` (algorithmic certificate normalization), `11_appendices/01_zfc.md` |
| `docs/source/dataset/stochastic_einstein_boltzmann.md` | `OBL-SEB-1` (thermodynamic/DPI + censorship closure) | classified as HORIZON due unresolved obstruction branch | `09_mathematical/01_theorems.md` (entropy/causality bridge), `09_mathematical/04_taxonomy.md` (cross-family family tagging) |
| `docs/source/dataset/landau_damping.md` (sector branch) | Sobolev regularity branch remains unresolved (Gevrey branch closed) | existing execution discharges Gevrey branch and records unresolved class in goal cone for lower regularity | `09_mathematical/01_theorems.md` (low-regularity damping theorem), `08_upgrades/01_instantaneous.md` or `08_upgrades/02_retroactive.md` if promotion can be proved |

## Completion Rule

For each row above, completion requires:

1. A named framework object in the target file under `09_mathematical`.
2. Any corresponding permit/contract in `05_interfaces` if the unresolved step is reused by one of the 17 nodes.
3. Optional upgrade entry in `08_upgrades` when the step can be promoted from `inc` to `+`.
4. Explicit bridge import in `11_appendices/01_zfc.md` when external theorem usage is required.

No dataset file in `docs/source/dataset/*.md` may claim classical completion for any row above until the listed obligations are removed from the final goal cone.
