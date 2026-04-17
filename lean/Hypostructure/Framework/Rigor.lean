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

def RigorBoundaryItems.namesByLayer
    (items : List RigorBoundaryItem)
    (layer : RigorLayer) : List String :=
  (items.filter fun item => item.layer = layer).map fun item => item.name

/-- Machine-checkable summary of the named non-framework trust boundary for a
theorem checkpoint.

This does not introspect Lean's kernel environment. Instead, it records the
intended theorem boundary as first-class data and proves that the named theorem
boundary coincides with the reusable literature boundary and contains no
problem-specific assumptions. The command-line `#print axioms` audit should
continue to be used as the kernel-level check that this list matches the actual
compiled theorem. -/
structure AxiomBoundaryAudit where
  finalTheorem : String
  frameworkBoundary : List String
  namedAxiomBoundary : List String
  literatureBoundary : List String
  problemSpecificBoundary : List String
  noProblemSpecificBoundary : problemSpecificBoundary = []
  namedBoundaryIsExactlyLiterature :
    ∀ name : String, name ∈ namedAxiomBoundary ↔ name ∈ literatureBoundary

namespace AxiomBoundaryAudit

theorem named_axiom_is_literature
    (audit : AxiomBoundaryAudit)
    {name : String}
    (hname : name ∈ audit.namedAxiomBoundary) :
    name ∈ audit.literatureBoundary :=
  (audit.namedBoundaryIsExactlyLiterature name).1 hname

theorem literature_is_named_axiom
    (audit : AxiomBoundaryAudit)
    {name : String}
    (hname : name ∈ audit.literatureBoundary) :
    name ∈ audit.namedAxiomBoundary :=
  (audit.namedBoundaryIsExactlyLiterature name).2 hname

theorem no_problem_specific
    (audit : AxiomBoundaryAudit) :
    audit.problemSpecificBoundary = [] :=
  audit.noProblemSpecificBoundary

end AxiomBoundaryAudit

end Hypostructure.Framework
