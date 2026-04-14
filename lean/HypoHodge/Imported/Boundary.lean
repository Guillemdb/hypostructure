import HypoHodge.Algebraic.VerifiedThinInput

namespace HypoHodge.Imported
open HypoHodge.Algebraic

structure ClassicalOrigin (I : VerifiedHodgeThinInput) : Prop where
structure HodgeBridgePremises (I : VerifiedHodgeThinInput) : Prop where
structure GammaContextPremises (I : VerifiedHodgeThinInput) : Prop where
structure TannakianBridgePremises (I : VerifiedHodgeThinInput) : Prop where

structure ProducesMHS (I : VerifiedHodgeThinInput) : Prop where
structure ProducesGamma (I : VerifiedHodgeThinInput) : Prop where
structure ProducesTann (I : VerifiedHodgeThinInput) : Prop where

class ImportedHodgeAxioms (I : VerifiedHodgeThinInput) : Prop where
  classicalExtraction :
    ClassicalOrigin I
  hodgeBridgeSound :
    HodgeBridgePremises I → ProducesMHS I
  tannakianContextSound :
    GammaContextPremises I → ProducesGamma I
  tannakianBridgeSound :
    TannakianBridgePremises I → ProducesTann I

end HypoHodge.Imported
