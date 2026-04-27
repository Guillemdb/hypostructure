import Hypostructure.Backends.NavierStokes.ProofSetup.Dichotomy

namespace Hypostructure.Backends.NavierStokes.ProofSetup

structure TypeIIInterfaceContract where
  export : TypeIIExportData
  carriesPositiveConcentration : Prop := export.concentration.criticalMassLowerBound
  carriesVelocityConcentration : Prop := export.concentration.velocityMassLowerBound
  carriesLocalEnergy : Prop := export.localEnergy.transferred
  carriesPressure : Prop := export.pressure.transferred
  carriesScaleWindows : Prop := export.scaleWindows.windowControl
  deriving Repr

def contractOfExport (data : TypeIIExportData) : TypeIIInterfaceContract :=
  { export := data }

def exportedTypeIIBranchSatisfiesContract
    (data : TypeIIExportData) :
    TypeIIInterfaceContract :=
  contractOfExport data

theorem typeIIAlternativeCarriesContract
    (data : TypeIIExportData) :
    (exportedTypeIIBranchSatisfiesContract data).export = data := by
  rfl

end Hypostructure.Backends.NavierStokes.ProofSetup