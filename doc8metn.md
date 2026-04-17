# Proof Audit Notes: `docs/source/proofs`

Date: 2026-04-16

## Scope
Reviewed all files under `docs/source/proofs` for low-risk correctness and renderability issues without changing proof strategy.

## What I fixed

### 1) Structural directive hygiene
- `docs/source/proofs/proof-mt-fact-min-inst.md`
  - Fixed malformed MyST close marker at the end of an `important` directive block:
    - `::: □` → `:::`
  - This resolved an unmatched open/close imbalance for `:::` directives.

- `docs/source/proofs/proof-mt-up-type-ii.md`
  - Removed an extra top-level closing `:::` at the file end.
  - Verified the main proof block is closed before the appendix section so the appendix remains outside the proof directive.

### 2) Renderability/consistency checks run
- Re-ran a directive balance sweep over all `docs/source/proofs/*.md`.
- Result: **no unmatched opens/closes remain** after the two fixes.
- Confirmed all proof files are listed in `docs/source/proofs/proofs.md` and wired in the book config (`docs/myst.yml`) for full rendered inclusion.

## What remains to be fixed (not changed)

### A) Mathematical/semantic issues needing deeper proof-level review
- The files are long, research-style proofs with many non-trivial hypotheses (dimension restrictions, coercivity assumptions, flow-regularity assumptions, etc.).
- I did not change argument structure, hypothesis statements, or theorem logic where verification requires mathematical judgment beyond syntax/consistency.

### B) Low-priority markdown-linter edge case
- In `proof-mt-up-spectral.md`, there is a LaTeX term of the form `[...] (x - x^*)` that can resemble a markdown link pattern in naive regex scans.
  - It appears inside a display equation and currently renders in math syntax, but a strict markdown-link linter may flag it.
  - This is optional cleanup only (no proof-strategy impact); left unchanged.

## Files touched by this pass
- `docs/source/proofs/proof-mt-fact-min-inst.md`
- `docs/source/proofs/proof-mt-up-type-ii.md`
- `doc8metn.md` (new)

