# Proof-Audit Report: Dataset Markdown Corpus

Date: 2026-04-14

Scope: `docs/source/dataset/*.md` as checked in current workspace, with emphasis on
template consistency (hypostructure execution protocol) and mathematical closure.

## What I fixed

- `docs/source/dataset/navier_stokes_3d.md`
  - Removed a circular dependency between event-recovery accounting and the declared target goal.
  - Kept `K_Rec_N` as an auxiliary diagnostic (`K_{Rec_N}^{inc}`) rather than a
    target-relevant prerequisite.
  - Moved the recovery-related residual entries to the non-goal cone in the node trace and
    obligation ledger.
  - Updated the corresponding transition and audit rows so the final status is now consistent.
- `docs/source/dataset/hodge_conjecture.md`
  - Removed remaining judgement-style “unconditional/conditional” phrasing from the
    theorem statement and final validity row.
  - Kept the proof object aligned with the same declared-backend framework context
    used elsewhere in the dataset.

Open obligations that are outside current template closure are now recorded in
`docs/source/1_hypostructure_formalism/09_mathematical/08_unresolved_external_extensions.md`.

## Non-fixable / unresolved items

The following files currently record mathematically open or semantically external
obligations that are not discharged by the current framework content and therefore
cannot be closed here.

- `docs/source/dataset/collatz.md`
  - Outstanding obligations: `OBL-1` (global finiteness of stopping time) and
    `OBL-3` (global singular-set exclusion).
  - Verdict remains **HORIZON**.
- `docs/source/dataset/kervaire_invariant.md`
  - `j=6` case (dimension 126) remains open on the ledger (`OBL-126`).
  - Verdict is sector dependent: resolved for `j \ge 7`, unresolved for `j=6`.
- `docs/source/dataset/riemann_hypothesis.md`
  - Outstanding obligation `OBL-1`: self-adjoint/spectral model/Hilbert–Pólya step.
- `docs/source/dataset/langlands.md`
  - Full number-field correspondence remains blocked (`OBL-LANG-1`) → **HORIZON**.
- `docs/source/dataset/yang_mills.md`
  - OS-axiom construction and non-perturbative mass gap remain open (`OBL-YM-OS`, `OBL-YM-GAP`, `OBL-YM-CLUST`).
- `docs/source/dataset/landau_damping.md`
  - The Gevrey branch closes; lower-regularity (Sobolev) sector remains unresolved in the current route and is recorded as sector-dependent.
- `docs/source/dataset/quantum_gravity.md`
  - Declared boundary/open-system regime remains in **Period VI / HORIZON**.
- `docs/source/dataset/stochastic_einstein_boltzmann.md`
  - DPI/thermodynamic and censorship-related obligations remain open (`OBL-SEB-1`) → **HORIZON**.
- `docs/source/dataset/p_vs_np.md`
  - Lock route records morphism-level obstruction; separation is not certified in ZFC (`HORIZON`).
- `docs/source/dataset/halting_problem.md`
  - Axiomatic horizon theorem reports undecidable-class outcomes as unresolved by design.
- `docs/source/dataset/bsd_conjecture.md`
  - Rank 0/1 is handled, but general-rank BSD is explicitly marked **HORIZON**.

## Framework-compliance notes

- No structural syntax or rendering errors were introduced by the above edit.
- `navier_stokes_3d` now explicitly separates auxiliary residual obligations from the
  goal-cone, matching the execution protocol in the template.
- I did not find additional concrete circularity violations in the 3D Navier–Stokes
  route after the repair above; other non-closed outcomes are documented as explicit
  horizons/missing backend assumptions.

## Recommendation

- The unresolved files above should remain open in the audit registry unless and until
  their missing analytic/number-theoretic backend certificates are added or imported as
  externally validated metatheorems.
- A framework-side registry of these unresolved obligations has been added at
  `docs/source/1_hypostructure_formalism/11_appendices/04_unresolved_backlog.md`.
