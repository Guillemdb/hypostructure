import HypoHodge.Core.Closure

namespace HypoHodge.Core

def backStep (deps : RuleSet) (S : Context) : Context :=
  deps.foldl
    (fun acc r => if r.conclusion ∈ acc then acc ∪ r.premises else acc)
    S

def goalConeN (deps : RuleSet) (n : ℕ) (goal : CertTag) : Context :=
  Nat.iterate (backStep deps) n ({goal} : Finset CertTag)

def goalCone (deps : RuleSet) (goal : CertTag) : Context :=
  goalConeN deps allTags.card goal

axiom goal_mem_goalCone (deps : RuleSet) (goal : CertTag) :
    goal ∈ goalCone deps goal

axiom backStep_monotone (deps : RuleSet) :
    Monotone (backStep deps)

axiom goalConeN_monotone (deps : RuleSet) (n : ℕ) :
    Monotone (goalConeN deps n)

axiom goalCone_fixed (deps : RuleSet) (goal : CertTag) :
    backStep deps (goalCone deps goal) = goalCone deps goal

axiom premises_in_goalCone_of_conclusion
    (deps : RuleSet) (goal : CertTag) (r : Rule)
    (hr : r ∈ deps)
    (hconc : r.conclusion ∈ goalCone deps goal) :
    r.premises ⊆ goalCone deps goal

end HypoHodge.Core
