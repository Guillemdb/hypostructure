import Hypostructure.Backends.Burgers1D.GroundTruthFinal

namespace Hypostructure.Backends.Burgers1D

open Hypostructure.Framework

noncomputable section

/-- Framework-proved facts and assemblers recorded in the Burgers ground-truth
rigor boundary. -/
def burgersGroundTruthFrameworkBoundary : List String :=
  RigorBoundaryItems.namesByLayer burgersGroundTruthRigorBoundary .framework

/-- The current final checkpoint has no problem-specific trusted assumptions:
every named non-framework assumption is classified as reusable literature. -/
theorem burgersGroundTruth_problemSpecificBoundary_empty :
    burgersGroundTruthProblemSpecificBoundary = [] :=
  rfl

/-- The named axiom boundary for the final checkpoint is exactly the reusable
literature boundary, up to list membership/order. -/
theorem burgersGroundTruth_axiomBoundary_iff_literatureBoundary
    (name : String) :
    name ∈ burgersGroundTruthAxiomBoundary ↔
      name ∈ burgersGroundTruthLiteratureBoundary := by
  simp [burgersGroundTruthAxiomBoundary, burgersGroundTruthLiteratureBoundary]
  tauto

theorem burgersGroundTruth_named_axiom_is_literature
    {name : String}
    (hname : name ∈ burgersGroundTruthAxiomBoundary) :
    name ∈ burgersGroundTruthLiteratureBoundary :=
  (burgersGroundTruth_axiomBoundary_iff_literatureBoundary name).1 hname

theorem burgersGroundTruth_literature_is_named_axiom
    {name : String}
    (hname : name ∈ burgersGroundTruthLiteratureBoundary) :
    name ∈ burgersGroundTruthAxiomBoundary :=
  (burgersGroundTruth_axiomBoundary_iff_literatureBoundary name).2 hname

/-- First-class audit object for the current closed theorem checkpoint.

Kernel-level verification still comes from running
`#print axioms burgers_groundTruth_dataset_theorem_from_axioms`; this object is
the reusable, checked bookkeeping layer saying that the named custom boundary is
all literature and contains no problem-specific assumption. -/
def burgersGroundTruthAxiomAudit : AxiomBoundaryAudit where
  finalTheorem := "burgers_groundTruth_dataset_theorem_from_axioms"
  frameworkBoundary := burgersGroundTruthFrameworkBoundary
  namedAxiomBoundary := burgersGroundTruthAxiomBoundary
  literatureBoundary := burgersGroundTruthLiteratureBoundary
  problemSpecificBoundary := burgersGroundTruthProblemSpecificBoundary
  noProblemSpecificBoundary := burgersGroundTruth_problemSpecificBoundary_empty
  namedBoundaryIsExactlyLiterature :=
    burgersGroundTruth_axiomBoundary_iff_literatureBoundary

theorem burgersGroundTruthAudit_no_problem_specific :
    burgersGroundTruthAxiomAudit.problemSpecificBoundary = [] :=
  burgersGroundTruthAxiomAudit.no_problem_specific

theorem burgersGroundTruthAudit_named_axiom_is_literature
    {name : String}
    (hname : name ∈ burgersGroundTruthAxiomAudit.namedAxiomBoundary) :
    name ∈ burgersGroundTruthAxiomAudit.literatureBoundary :=
  burgersGroundTruthAxiomAudit.named_axiom_is_literature hname

theorem burgersGroundTruthAudit_literature_is_named_axiom
    {name : String}
    (hname : name ∈ burgersGroundTruthAxiomAudit.literatureBoundary) :
    name ∈ burgersGroundTruthAxiomAudit.namedAxiomBoundary :=
  burgersGroundTruthAxiomAudit.literature_is_named_axiom hname

end

end Hypostructure.Backends.Burgers1D
