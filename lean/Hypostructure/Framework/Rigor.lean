namespace Hypostructure.Framework

/-- Coarse proof-boundary classification used by backend roadmaps.

`framework` means the claim is part of the reusable hypostructure engine and
must be proved in Lean. `literature` means a reusable mathematical fact that is
standard enough to share across backends, even if it is temporarily represented
by an explicit axiom while mathlib support is built. `problem` means a theorem
specific to one backend/problem instantiation. -/
inductive RigorLayer where
  | framework
  | literature
  | problem
  deriving DecidableEq, Repr

def RigorLayer.code : RigorLayer → String
  | .framework => "F"
  | .literature => "L"
  | .problem => "P"

def RigorLayer.label : RigorLayer → String
  | .framework => "hypostructure framework"
  | .literature => "reusable literature fact"
  | .problem => "problem-specific math"

structure RigorBoundaryItem where
  name : String
  layer : RigorLayer
  description : String

def RigorBoundaryItem.code (item : RigorBoundaryItem) : String :=
  item.layer.code

end Hypostructure.Framework
