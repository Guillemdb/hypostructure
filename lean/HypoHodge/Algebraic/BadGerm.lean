import HypoHodge.Algebraic.LocalCertificates

namespace HypoHodge.Algebraic

structure HodgeWitness where
  tag     : Nat
  nonzero : Prop

structure BadAlgGerm (n p : ℕ) where
  rankBound : ℕ
  h_rank    : rankBound ≤ n
  witness   : HodgeWitness
  bad       : Prop
  minimal   : Prop

structure WitnessHom {n p : ℕ} (A B : BadAlgGerm n p) where
  mapTag           : Nat → Nat
  mapWitness       : mapTag A.witness.tag = B.witness.tag
  preservesBad     : A.bad → B.bad
  preservesNonzero : A.witness.nonzero → B.witness.nonzero

namespace WitnessHom

def id (A : BadAlgGerm n p) : WitnessHom A A where
  mapTag := id
  mapWitness := rfl
  preservesBad := id
  preservesNonzero := id

def comp {A B C : BadAlgGerm n p} (f : WitnessHom A B) (g : WitnessHom B C) :
    WitnessHom A C where
  mapTag := g.mapTag ∘ f.mapTag
  mapWitness := by
    calc
      (g.mapTag ∘ f.mapTag) A.witness.tag
          = g.mapTag (f.mapTag A.witness.tag) := rfl
      _ = g.mapTag B.witness.tag := by rw [f.mapWitness]
      _ = C.witness.tag := g.mapWitness
  preservesBad := fun hA => g.preservesBad (f.preservesBad hA)
  preservesNonzero := fun hA => g.preservesNonzero (f.preservesNonzero hA)

theorem id_comp {A B : BadAlgGerm n p}
    (f : WitnessHom A B) :
    comp (id A) f = f := by
  cases f
  rfl

theorem comp_id {A B : BadAlgGerm n p}
    (f : WitnessHom A B) :
    comp f (id B) = f := by
  cases f
  rfl

theorem assoc {A B C D : BadAlgGerm n p}
    (f : WitnessHom A B) (g : WitnessHom B C) (h : WitnessHom C D) :
    comp (comp f g) h = comp f (comp g h) := by
  cases f
  cases g
  cases h
  rfl

end WitnessHom

end HypoHodge.Algebraic
