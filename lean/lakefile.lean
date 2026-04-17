import Lake
open Lake DSL

package «HypoHodge» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.12.0"

lean_lib «HypoHodge» where

lean_lib «Hypostructure» where

/-- Check that the kernel-level custom axioms of the Burgers ground-truth
checkpoint exactly match the declared reusable literature boundary. -/
script auditBurgersAxioms do
  let output ← IO.Process.output {
    cmd := "python3"
    args := #["scripts/audit_burgers_axioms.py"]
  }
  IO.print output.stdout
  IO.eprint output.stderr
  return output.exitCode
