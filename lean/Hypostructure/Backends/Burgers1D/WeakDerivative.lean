import Hypostructure.Backends.Burgers1D.StateSpace

namespace Hypostructure.Backends.Burgers1D

namespace BurgersState

/-- The mean-zero part keeps the derivative witness and subtracts the constant mean profile
from the value component. -/
noncomputable def meanZeroPart (u : BurgersState) : BurgersState :=
  u - constantState (mean u)

theorem deriv_meanZeroPart (u : BurgersState) :
    (meanZeroPart u).deriv = u.deriv := by
  simp [meanZeroPart, constantState, deriv, zeroDerivative]

theorem mean_meanZeroPart (u : BurgersState) : mean (meanZeroPart u) = 0 := by
  rw [meanZeroPart, mean_sub, mean_constantState]
  ring

theorem dissipation_meanZeroPart (u : BurgersState) :
    dissipation (meanZeroPart u) = dissipation u := by
  simp [dissipation, meanZeroPart, deriv, constantState, zeroDerivative]

theorem meanZeroEnergy_meanZeroPart (u : BurgersState) :
    meanZeroEnergy (meanZeroPart u) = meanZeroEnergy u := by
  unfold meanZeroEnergy
  rw [mean_meanZeroPart]
  simp [meanZeroPart, constantState, constantProfile, value]

theorem meanZeroPart_add_meanEquilibrium (u : BurgersState) :
    meanZeroPart u + meanEquilibrium u = u := by
  apply Prod.ext
  · ext x
    simp [meanZeroPart, meanEquilibrium, constantState, constantProfile, value]
  · ext x
    simp [meanZeroPart, meanEquilibrium, constantState, zeroDerivative, deriv]

theorem decomposes_into_mean_zero_and_equilibrium (u : BurgersState) :
    ∃ v : BurgersState, ∃ m : ℝ,
      mean v = 0 ∧ u = v + constantState m := by
  refine ⟨meanZeroPart u, mean u, mean_meanZeroPart u, ?_⟩
  rw [eq_comm]
  simpa [meanEquilibrium] using meanZeroPart_add_meanEquilibrium u

end BurgersState

end Hypostructure.Backends.Burgers1D
