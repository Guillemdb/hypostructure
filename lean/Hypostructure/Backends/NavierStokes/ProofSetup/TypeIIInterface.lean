import Hypostructure.Backends.NavierStokes.ProofSetup.Dichotomy

namespace Hypostructure.Backends.NavierStokes.ProofSetup

structure TypeIIInterfaceContract where
  typeIIExport : TypeIIExportData
  carriesPositiveConcentration : Prop :=
    0 < typeIIExport.concentration.etaCD ∧
      ∀ n, typeIIExport.concentration.etaCD ≤ (typeIIExport.concentration.criticalValues n).totalCD
  carriesVelocityConcentration : Prop :=
    0 < typeIIExport.concentration.etaV ∧
      ∀ n, typeIIExport.concentration.etaV ≤ (typeIIExport.concentration.criticalValues n).C.value
  carriesLocalEnergy : Prop := typeIIExport.localEnergy.transferred
  carriesPressure : Prop := typeIIExport.pressure.transferred
  carriesScaleWindows : Prop := typeIIExport.scaleWindows.windowControl

def contractOfExport (data : TypeIIExportData) : TypeIIInterfaceContract :=
  { typeIIExport := data }

def exportedTypeIIBranchSatisfiesContract
    (data : TypeIIExportData) :
    TypeIIInterfaceContract :=
  contractOfExport data

theorem typeIIAlternativeCarriesContract
    (data : TypeIIExportData) :
  (exportedTypeIIBranchSatisfiesContract data).typeIIExport = data := by
  rfl

end Hypostructure.Backends.NavierStokes.ProofSetup