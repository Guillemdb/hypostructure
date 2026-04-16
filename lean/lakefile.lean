import Lake
open Lake DSL

package «HypoHodge» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.12.0"

lean_lib «HypoHodge» where

lean_lib «Hypostructure» where
