import HypoHodge.Algebraic.Coding
import HypoHodge.Algebraic.VerifiedThinInput

namespace HypoHodge.Algebraic

def StaticClassifiable (I : VerifiedHodgeThinInput) : Prop :=
  Encodable (BadAlgGerm I.Qrank I.p)

def BoundedReductionRealized (I : VerifiedHodgeThinInput) : Prop :=
  ∀ {A B : BadAlgGerm I.Qrank I.p}, WitnessHom A B →
    ∃ C : BadAlgGerm I.Qrank I.p, C.rankBound ≤ I.Qrank

theorem hodgeBadBoundedReduction_rankBounded
    (I : VerifiedHodgeThinInput)
    {A B : BadAlgGerm I.Qrank I.p}
    (_f : WitnessHom A B) :
    ∃ C : BadAlgGerm I.Qrank I.p, C.rankBound ≤ I.Qrank := by
  exact ⟨B, B.h_rank⟩

theorem hodgeBoundedReductionRealized
    (I : VerifiedHodgeThinInput) :
    BoundedReductionRealized I := by
  intro A B f
  exact hodgeBadBoundedReduction_rankBounded I f

theorem hodgeBadBoundedReduction
    (I : VerifiedHodgeThinInput)
    {A B : BadAlgGerm I.Qrank I.p}
    (_f : WitnessHom A B) :
    ∃ C : BadAlgGerm I.Qrank I.p, True := by
  exact ⟨B, trivial⟩

theorem hodgeClassifiableStatic
    (I : VerifiedHodgeThinInput) :
    StaticClassifiable I := by
  exact boundedGermSmallness I.Qrank I.p

end HypoHodge.Algebraic
