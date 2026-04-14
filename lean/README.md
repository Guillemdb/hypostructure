# HypoHodge Lean skeleton

This package is a concrete Lean project skeleton for the Phase A roadmap.

It is intentionally split into three layers:

- `Core/`: the certificate engine and proof-completion kernel
- `Imported/`: the trusted literature boundary
- `Algebraic/` and `Hodge/`: the new algebraic backend and the Hodge template run

## Important note

This repository is a **stub package**. The easy finite-set/kernel lemmas are filled in where practical, but the genuinely new mathematics is left as named `axiom` placeholders so the public API is explicit from day one.

The main placeholders you are expected to replace with proofs are:

- fixed-point theorems in `Core/Closure.lean`
- backward dependency theorems in `Core/GoalCone.lean`
- the injective coding theorem in `Algebraic/Coding.lean`
- the backend auto-closure theorem in `Algebraic/BackendAutoclose.lean`
- the bridge/run emission theorems in `Hodge/Run.lean`
- the no-lock-inc / no-promo-inc theorems in `Hodge/ProofAudit.lean`

## Build

1. Install Lean 4 and Lake.
2. From the repository root, run:
   - `lake update`
   - `lake build`

Because the current execution environment does not contain `lean` or `lake`, this package was generated but not compiled here.
