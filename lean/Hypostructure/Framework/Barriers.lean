import Hypostructure.Framework.Certificates

namespace Hypostructure.Framework

universe u

inductive BarrierOutcome (β γ δ : Type u)
  | blocked (cert : β)
  | breached (cert : γ)
  | special (cert : δ)

def barrierSatRoute
    (h : BarrierSatBlockedCertificate) :
    GateNode :=
  h.continuation

def barrierCausalRoute
    (h : BarrierCausalBlockedCertificate) :
    GateNode :=
  h.continuation

def barrierScatRoute
    (h : BarrierScatBenignCertificate) :
    GateNode :=
  h.continuation

def barrierTypeIIRoute
    (h : BarrierTypeIIBlockedCertificate) :
    GateNode :=
  h.continuation

def barrierVacRoute
    (h : BarrierVacBlockedCertificate) :
    GateNode :=
  h.continuation

def barrierCapRoute
    (h : BarrierCapBlockedCertificate) :
    GateNode :=
  h.continuation

def barrierGapRoute
    (h : BarrierGapBlockedCertificate) :
    GateNode :=
  h.continuation

def barrierGapRestoration
    (h : BarrierGapStagnationCertificate) :
    GateNode :=
  h.restoration

def barrierActionRoute
    (h : BarrierActionBlockedCertificate) :
    GateNode :=
  h.continuation

def barrierOminRoute
    (h : BarrierOminBlockedCertificate) :
    GateNode :=
  h.continuation

def barrierMixRoute
    (h : BarrierMixBlockedCertificate) :
    GateNode :=
  h.continuation

def barrierEpiRoute
    (h : BarrierEpiBlockedCertificate) :
    GateNode :=
  h.continuation

def barrierFreqRoute
    (h : BarrierFreqBlockedCertificate) :
    GateNode :=
  h.continuation

def barrierBodeRoute
    (h : BarrierBodeBlockedCertificate) :
    GateNode :=
  h.continuation

def barrierInputRoute
    (h : BarrierInputBlockedCertificate) :
    GateNode :=
  h.continuation

def barrierVarietyRoute
    (h : BarrierVarietyBlockedCertificate) :
    GateNode :=
  h.continuation

def barrierExclusionRoute
    (h : BarrierExclusionBlockedCertificate) :
    GateNode :=
  h.continuation

def mkBarrierSatBlocked
    (p : BarrierSatPayload) :
    BarrierSatBlockedCertificate :=
  { barrier := .barrierSat, payload := p, continuation := .zenoCheck }

def mkBarrierSatBreached
    (p : BarrierSatPayload) :
    BarrierSatBreachedCertificate :=
  { barrier := .barrierSat, payload := p, mode := .cE, surgery := .surgCE, reentry := .zenoCheck }

def mkBarrierCausalBlocked
    (p : BarrierCausalPayload) :
    BarrierCausalBlockedCertificate :=
  { barrier := .barrierCausal, payload := p, continuation := .compactCheck }

def mkBarrierCausalBreached
    (p : BarrierCausalPayload) :
    BarrierCausalBreachedCertificate :=
  { barrier := .barrierCausal, payload := p, mode := .cC, surgery := .surgCC,
    reentry := .compactCheck }

def mkBarrierScatBenign
    (p : BarrierScatPayload) :
    BarrierScatBenignCertificate :=
  { barrier := .barrierScat, payload := p, continuation := .compactCheck }

def mkBarrierScatPathological
    (p : BarrierScatPayload) :
    BarrierScatPathologicalCertificate :=
  { barrier := .barrierScat, payload := p, mode := .cD, surgery := .surgCDAlt,
    reentry := .compactCheck }

def mkBarrierTypeIIBlocked
    (p : BarrierTypeIIPayload) :
    BarrierTypeIIBlockedCertificate :=
  { barrier := .barrierTypeII, payload := p, continuation := .paramCheck }

def mkBarrierTypeIIBreached
    (p : BarrierTypeIIPayload) :
    BarrierTypeIIBreachedCertificate :=
  { barrier := .barrierTypeII, payload := p, mode := .sE, surgery := .surgSE,
    reentry := .paramCheck }

def mkBarrierVacBlocked
    (p : BarrierVacPayload) :
    BarrierVacBlockedCertificate :=
  { barrier := .barrierVac, payload := p, continuation := .geomCheck }

def mkBarrierVacBreached
    (p : BarrierVacPayload) :
    BarrierVacBreachedCertificate :=
  { barrier := .barrierVac, payload := p, mode := .sC, surgery := .surgSC,
    reentry := .geomCheck }

def mkBarrierCapBlocked
    (p : BarrierCapPayload) :
    BarrierCapBlockedCertificate :=
  { barrier := .barrierCap, payload := p, continuation := .stiffnessCheck }

def mkBarrierCapBreached
    (p : BarrierCapPayload) :
    BarrierCapBreachedCertificate :=
  { barrier := .barrierCap, payload := p, mode := .cD, surgery := .surgCD,
    reentry := .stiffnessCheck }

def mkBarrierGapBlocked
    (p : BarrierGapPayload) :
    BarrierGapBlockedCertificate :=
  { barrier := .barrierGap, payload := p, continuation := .topoCheck }

def mkBarrierGapStagnation
    (p : BarrierGapPayload) :
    BarrierGapStagnationCertificate :=
  { barrier := .barrierGap, payload := p, restoration := .bifurcateCheck }

def mkBarrierActionBlocked
    (p : BarrierActionPayload) :
    BarrierActionBlockedCertificate :=
  { barrier := .barrierAction, payload := p, continuation := .tameCheck }

def mkBarrierActionBreached
    (p : BarrierActionPayload) :
    BarrierActionBreachedCertificate :=
  { barrier := .barrierAction, payload := p, mode := .tE, surgery := .surgTE,
    reentry := .tameCheck }

def mkBarrierOminBlocked
    (p : BarrierOminPayload) :
    BarrierOminBlockedCertificate :=
  { barrier := .barrierOmin, payload := p, continuation := .ergoCheck }

def mkBarrierOminBreached
    (p : BarrierOminPayload) :
    BarrierOminBreachedCertificate :=
  { barrier := .barrierOmin, payload := p, mode := .tC, surgery := .surgTC,
    reentry := .ergoCheck }

def mkBarrierMixBlocked
    (p : BarrierMixPayload) :
    BarrierMixBlockedCertificate :=
  { barrier := .barrierMix, payload := p, continuation := .complexCheck }

def mkBarrierMixBreached
    (p : BarrierMixPayload) :
    BarrierMixBreachedCertificate :=
  { barrier := .barrierMix, payload := p, mode := .tD, surgery := .surgTD,
    reentry := .complexCheck }

def mkBarrierEpiBlocked
    (p : BarrierEpiPayload) :
    BarrierEpiBlockedCertificate :=
  { barrier := .barrierEpi, payload := p, continuation := .oscillateCheck }

def mkBarrierEpiBreached
    (p : BarrierEpiPayload) :
    BarrierEpiBreachedCertificate :=
  { barrier := .barrierEpi, payload := p, mode := .dC, surgery := .surgDC,
    reentry := .oscillateCheck }

def mkBarrierFreqBlocked
    (p : BarrierFreqPayload) :
    BarrierFreqBlockedCertificate :=
  { barrier := .barrierFreq, payload := p, continuation := .boundaryCheck }

def mkBarrierFreqBreached
    (p : BarrierFreqPayload) :
    BarrierFreqBreachedCertificate :=
  { barrier := .barrierFreq, payload := p, mode := .dE, surgery := .surgDE,
    reentry := .boundaryCheck }

def mkBarrierBodeBlocked
    (p : BarrierBodePayload) :
    BarrierBodeBlockedCertificate :=
  { barrier := .barrierBode, payload := p, continuation := .starveCheck }

def mkBarrierBodeBreached
    (p : BarrierBodePayload) :
    BarrierBodeBreachedCertificate :=
  { barrier := .barrierBode, payload := p, mode := .bE, surgery := .surgBE,
    reentry := .starveCheck }

def mkBarrierInputBlocked
    (p : BarrierInputPayload) :
    BarrierInputBlockedCertificate :=
  { barrier := .barrierInput, payload := p, continuation := .alignCheck }

def mkBarrierInputBreached
    (p : BarrierInputPayload) :
    BarrierInputBreachedCertificate :=
  { barrier := .barrierInput, payload := p, mode := .bD, surgery := .surgBD,
    reentry := .alignCheck }

def mkBarrierVarietyBlocked
    (p : BarrierVarietyPayload) :
    BarrierVarietyBlockedCertificate :=
  { barrier := .barrierVariety, payload := p, continuation := .lock }

def mkBarrierVarietyBreached
    (p : BarrierVarietyPayload) :
    BarrierVarietyBreachedCertificate :=
  { barrier := .barrierVariety, payload := p, mode := .bC, surgery := .surgBC,
    reentry := .lock }

def mkBarrierExclusionBlocked
    (p : BarrierExclusionPayload) :
    BarrierExclusionBlockedCertificate :=
  { barrier := .barrierExclusion, payload := p, continuation := .lock }

def mkBarrierExclusionBreached
    (p : BarrierExclusionPayload) :
    BarrierExclusionBreachedCertificate :=
  { barrier := .barrierExclusion, payload := p, mode := .dC, surgery := .surgDC,
    reentry := .lock }

theorem barrierSatRoute_spec
    (p : BarrierSatPayload) :
    barrierSatRoute (mkBarrierSatBlocked p) = .zenoCheck := rfl

theorem barrierCausalRoute_spec
    (p : BarrierCausalPayload) :
    barrierCausalRoute (mkBarrierCausalBlocked p) = .compactCheck := rfl

theorem barrierScatRoute_spec
    (p : BarrierScatPayload) :
    barrierScatRoute (mkBarrierScatBenign p) = .compactCheck := rfl

theorem barrierTypeIIRoute_spec
    (p : BarrierTypeIIPayload) :
    barrierTypeIIRoute (mkBarrierTypeIIBlocked p) = .paramCheck := rfl

theorem barrierVacRoute_spec
    (p : BarrierVacPayload) :
    barrierVacRoute (mkBarrierVacBlocked p) = .geomCheck := rfl

theorem barrierCapRoute_spec
    (p : BarrierCapPayload) :
    barrierCapRoute (mkBarrierCapBlocked p) = .stiffnessCheck := rfl

theorem barrierGapRoute_spec
    (p : BarrierGapPayload) :
    barrierGapRoute (mkBarrierGapBlocked p) = .topoCheck := rfl

theorem barrierGapRestoration_spec
    (p : BarrierGapPayload) :
    barrierGapRestoration (mkBarrierGapStagnation p) = .bifurcateCheck := rfl

theorem barrierActionRoute_spec
    (p : BarrierActionPayload) :
    barrierActionRoute (mkBarrierActionBlocked p) = .tameCheck := rfl

theorem barrierOminRoute_spec
    (p : BarrierOminPayload) :
    barrierOminRoute (mkBarrierOminBlocked p) = .ergoCheck := rfl

theorem barrierMixRoute_spec
    (p : BarrierMixPayload) :
    barrierMixRoute (mkBarrierMixBlocked p) = .complexCheck := rfl

theorem barrierEpiRoute_spec
    (p : BarrierEpiPayload) :
    barrierEpiRoute (mkBarrierEpiBlocked p) = .oscillateCheck := rfl

theorem barrierFreqRoute_spec
    (p : BarrierFreqPayload) :
    barrierFreqRoute (mkBarrierFreqBlocked p) = .boundaryCheck := rfl

theorem barrierBodeRoute_spec
    (p : BarrierBodePayload) :
    barrierBodeRoute (mkBarrierBodeBlocked p) = .starveCheck := rfl

theorem barrierInputRoute_spec
    (p : BarrierInputPayload) :
    barrierInputRoute (mkBarrierInputBlocked p) = .alignCheck := rfl

theorem barrierVarietyRoute_spec
    (p : BarrierVarietyPayload) :
    barrierVarietyRoute (mkBarrierVarietyBlocked p) = .lock := rfl

theorem barrierExclusionRoute_spec
    (p : BarrierExclusionPayload) :
    barrierExclusionRoute (mkBarrierExclusionBlocked p) = .lock := rfl

end Hypostructure.Framework
