import Physics.Prelude
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace Physics

/-- The “Riemann trend” model used in the numerics. -/
noncomputable def riemannTrend (α β : ℝ) (T : ℝ) : ℝ :=
  α * (T / (2 * Real.pi)) * Real.log T + β

end Physics

