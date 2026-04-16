import Mathlib.Data.Real.Basic

namespace Hypostructure.Backends.Burgers1D

/-- Physical parameters for the one-dimensional viscous Burgers backend. -/
structure BurgersParameters where
  viscosity : ℝ
  viscosity_pos : 0 < viscosity

end Hypostructure.Backends.Burgers1D
