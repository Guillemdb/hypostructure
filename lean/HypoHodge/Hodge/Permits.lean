import HypoHodge.Algebraic.BackendAutoclose

namespace HypoHodge.Hodge
open HypoHodge.Core

def permitHdg : Rule :=
  { kind := RuleKind.bridge
    premises := ({ CertTag.ambient, CertTag.tbPi, CertTag.scLambda, CertTag.dE } : Finset CertTag)
    conclusion := CertTag.mhs }

def permitTann : Rule :=
  { kind := RuleKind.bridge
    premises := ({ CertTag.adm, CertTag.gamma, CertTag.catLib } : Finset CertTag)
    conclusion := CertTag.tann }

def permitLock : Rule :=
  { kind := RuleKind.bridge
    premises := ({ CertTag.mhs, CertTag.tann, CertTag.cMu, CertTag.lsSigma,
                   CertTag.tbO, CertTag.catLib, CertTag.init, CertTag.adm } : Finset CertTag)
    conclusion := CertTag.catHomBlk }

def bridgeRules : RuleSet := [permitHdg, permitTann, permitLock]
def promotionRules : RuleSet := []
def allRules : RuleSet := HypoHodge.Algebraic.backendRules ++ bridgeRules ++ promotionRules
def deps : RuleSet := allRules

theorem permitHdg_premises :
    permitHdg.premises =
      ({ CertTag.ambient, CertTag.tbPi, CertTag.scLambda, CertTag.dE } : Finset CertTag) := by
  rfl

theorem permitTann_premises :
    permitTann.premises =
      ({ CertTag.adm, CertTag.gamma, CertTag.catLib } : Finset CertTag) := by
  rfl

theorem permitLock_premises :
    permitLock.premises =
      ({ CertTag.mhs, CertTag.tann, CertTag.cMu, CertTag.lsSigma,
         CertTag.tbO, CertTag.catLib, CertTag.init, CertTag.adm } : Finset CertTag) := by
  rfl

end HypoHodge.Hodge
