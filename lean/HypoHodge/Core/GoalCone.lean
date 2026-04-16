import HypoHodge.Core.Closure
import Hypostructure.Core.GoalCone

namespace HypoHodge.Core

def backStep (deps : RuleSet) (S : Context) : Context :=
  Hypostructure.Core.backStep deps S

def goalExpandN (deps : RuleSet) : ℕ → Context → Context
  | n, S => Hypostructure.Core.goalExpandN deps n S

def goalConeN (deps : RuleSet) (n : ℕ) (goal : CertTag) : Context :=
  Hypostructure.Core.goalConeN deps n goal

def goalCone (deps : RuleSet) (goal : CertTag) : Context :=
  Hypostructure.Core.goalCone deps goal

theorem backStep_monotone (deps : RuleSet) :
    Monotone (backStep deps) := by
  simpa [backStep] using Hypostructure.Core.backStep_monotone deps

theorem goalConeN_monotone (deps : RuleSet) (n : ℕ) :
    Monotone (goalExpandN deps n) := by
  simpa [goalExpandN] using Hypostructure.Core.goalConeN_monotone deps n

theorem goal_mem_goalCone (deps : RuleSet) (goal : CertTag) :
    goal ∈ goalCone deps goal := by
  simpa [goalCone] using Hypostructure.Core.goal_mem_goalCone deps goal

theorem goalCone_fixed (deps : RuleSet) (goal : CertTag) :
    backStep deps (goalCone deps goal) = goalCone deps goal := by
  simpa [backStep, goalCone] using Hypostructure.Core.goalCone_fixed deps goal

theorem premises_in_goalCone_of_conclusion
    (deps : RuleSet) (goal : CertTag) (r : Rule)
    (hr : r ∈ deps)
    (hconc : r.conclusion ∈ goalCone deps goal) :
    r.premises ⊆ goalCone deps goal := by
  simpa [goalCone] using Hypostructure.Core.premises_in_goalCone_of_conclusion deps goal r hr hconc

end HypoHodge.Core
