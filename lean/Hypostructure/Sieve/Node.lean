import Mathlib

namespace Hypostructure.Sieve

inductive NodeTag
  | anchor
  | defect
  | recurrence
  | multiplicity
  | tame
  | partialCompact
  | capacity
  | stability
  | boundaryEven
  | boundaryPolarized
  | boundedPartial
  | ambient
  | repConservative
  | repComplete
  | admissible
  | germ
  | reduction
  | library
  | gamma
  | primaryBridge
  | secondaryBridge
  | goal
  | goalInc
  | promoInc
  deriving DecidableEq, Repr, Fintype

def NodeTag.isPending : NodeTag → Bool
  | .goalInc => true
  | .promoInc => true
  | _ => false

def allNodes : Finset NodeTag := Finset.univ

theorem mem_allNodes (k : NodeTag) : k ∈ allNodes := by
  simp [allNodes]

theorem allNodes_nodup : allNodes.1.Nodup := by
  simpa [allNodes] using (Finset.univ : Finset NodeTag).2

end Hypostructure.Sieve
