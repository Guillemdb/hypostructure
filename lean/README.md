# HypoHodge Lean skeleton

This package is a concrete Lean development for the tag-level HypoHodge kernel and its
semantic Hodge-target interface.

It is intentionally split into three layers:

- `Core/`: the certificate engine and proof-completion kernel
- `Imported/`: the trusted literature boundary
- `Algebraic/` and `Hodge/`: the new algebraic backend and the Hodge template run

## Current status

The package now has:

- a complete finite certificate kernel in `Core/`,
- a semantic input layer for a distinguished Hodge problem in `Algebraic/VerifiedThinInput.lean`,
- a non-empty imported boundary in `Imported/Boundary.lean`,
- rule-soundness and closure-soundness theorems in `Hodge/Semantics.lean`,
- a kernel completion theorem and a semantic top theorem in `Hodge/Final.lean`.

The imported Hodge-theoretic layer is still abstract: accepted mathematics is represented by structured interfaces rather than a full first-principles mathlib formalization of algebraic geometry and Hodge theory.

## Build

1. Install Lean 4 and Lake.
2. From the repository root, run:
   - `lake update`
   - `lake build`

In the current workspace, `lake build` succeeds.
