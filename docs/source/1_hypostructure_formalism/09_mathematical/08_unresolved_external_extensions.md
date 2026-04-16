# External Obligations Beyond Template Closure

Date: 2026-04-15

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
| `docs/source/dataset/langlands.md` | `OBL-LANG-1` (global number-field correspondence block) | `K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}` persists at Lock | `09_mathematical/02_algebraic.md` (reciprocity/correspondence theorems), `09_mathematical/03_cross_reference.md` (bidirectional bridge bookkeeping) |
| `docs/source/dataset/yang_mills.md` | `OBL-YM-OS`, `OBL-YM-GAP`, `OBL-YM-CLUST` (OS reconstruction, non-perturbative mass-gap, clustering) | lock route is blocked by unresolved analytic/morphological input | `09_mathematical/02_algebraic.md` (quantum-classical interface), `09_mathematical/05_algorithmic.md` (constructive cluster obstruction formalization) |
| `docs/source/dataset/halting_problem.md` | Proper horizon undecidability branch | framework records a frontier/horizon certificate and does not export decidability in classical form | `09_mathematical/05_algorithmic.md` (axiomatic undecidability schema), `11_appendices/01_zfc.md` (axiom/rule statement) |
| `docs/source/dataset/quantum_gravity.md` | `OBL-QG-1` (proper horizon boundary/censorship reconstruction in framework class) | classified as Period VI/HORIZON after lock boundary obligations persist | `09_mathematical/01_theorems.md` (entropy/causality bridging lemmas), `04_nodes/03_surgery_nodes.md` (new boundary-aware reconstruction node if formalized) |
| `docs/source/dataset/p_vs_np.md` | `OBL-PNP` (global separation in the classical ZFC formulation) | route records `K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}` plus residual export obligation | `09_mathematical/06_complexity_bridge.md` (hardness separation schema), `09_mathematical/05_algorithmic.md` (algorithmic certificate normalization), `11_appendices/01_zfc.md` |

## Completion Rule

For each row above, completion requires:

1. A named framework object in the target file under `09_mathematical`.
2. Any corresponding permit/contract in `05_interfaces` if the unresolved step is reused by one of the 17 nodes.
3. Optional upgrade entry in `08_upgrades` when the step can be promoted from `inc` to `+`.
4. Explicit bridge import in `11_appendices/01_zfc.md` when external theorem usage is required.

No dataset file in `docs/source/dataset/*.md` may claim classical completion for any row above until the listed obligations are removed from the final goal cone.
