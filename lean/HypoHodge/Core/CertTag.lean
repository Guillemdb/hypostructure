import Mathlib

namespace HypoHodge.Core

inductive CertTag
  | adj
  | dE
  | recN
  | cMu
  | scLambda
  | scPartialC
  | capH
  | lsSigma
  | tbPi
  | tbO
  | boundPartial
  | ambient
  | repCon
  | repComp
  | adm
  | germ
  | init
  | catLib
  | gamma
  | mhs
  | tann
  | catHomBlk
  | catHomInc
  | promoInc
  deriving DecidableEq, Repr, Fintype

def CertTag.isInc : CertTag → Bool
  | .catHomInc => true
  | .promoInc  => true
  | _          => false

def allTags : Finset CertTag := Finset.univ

theorem mem_allTags (k : CertTag) : k ∈ allTags := by
  simp [allTags]

theorem allTags_nodup : allTags.Nodup := by
  simp [allTags]

end HypoHodge.Core
