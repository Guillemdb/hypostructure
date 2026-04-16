import HypoHodge.Algebraic.Coding
import HypoHodge.Algebraic.VerifiedThinInput

namespace HypoHodge.Algebraic

def StaticClassifiable (I : VerifiedHodgeThinInput) : Prop :=
  Nonempty (Encodable (BadAlgGerm I.Qrank I.p)) ∧ I.germBounded

def BoundedReductionRealized (I : VerifiedHodgeThinInput) : Prop :=
  ∀ {A B : BadAlgGerm I.Qrank I.p}, WitnessHom A B →
    ∃ C : BadAlgGerm I.Qrank I.p, C.rankBound ≤ I.Qrank ∧ I.germBounded

theorem hodgeBadBoundedReduction_rankBounded
    (I : VerifiedHodgeThinInput)
    {A B : BadAlgGerm I.Qrank I.p}
    (_f : WitnessHom A B) :
    ∃ C : BadAlgGerm I.Qrank I.p, C.rankBound ≤ I.Qrank ∧ I.germBounded := by
  exact ⟨B, B.h_rank, I.germBounded_holds⟩

theorem hodgeBoundedReductionRealized
    (I : VerifiedHodgeThinInput) :
    BoundedReductionRealized I := by
  intro A B f
  exact hodgeBadBoundedReduction_rankBounded I f

theorem hodgeBadBoundedReduction
    (I : VerifiedHodgeThinInput)
    {A B : BadAlgGerm I.Qrank I.p}
    (_f : WitnessHom A B) :
    ∃ C : BadAlgGerm I.Qrank I.p, C.rankBound ≤ I.Qrank ∧ I.initialReduction := by
  exact ⟨B, B.h_rank, I.initialReduction_holds⟩

theorem hodgeClassifiableStatic
    (I : VerifiedHodgeThinInput) :
    StaticClassifiable I := by
  exact ⟨boundedGermSmallness I.Qrank I.p, I.germBounded_holds⟩

end HypoHodge.Algebraic
