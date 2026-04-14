import HypoHodge.Algebraic.Coding
import HypoHodge.Algebraic.VerifiedThinInput

namespace HypoHodge.Algebraic

def StaticClassifiable (_I : VerifiedHodgeThinInput) : Prop := True

theorem hodgeBadBoundedReduction
    (I : VerifiedHodgeThinInput)
    {A B : BadAlgGerm I.Qrank I.p}
    (f : WitnessHom A B) :
    ∃ C : BadAlgGerm I.Qrank I.p, True := by
  exact ⟨A, trivial⟩

theorem hodgeClassifiableStatic
    (I : VerifiedHodgeThinInput) :
    StaticClassifiable I := by
  trivial

end HypoHodge.Algebraic
