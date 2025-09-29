import Physics.Prelude
import Physics.Field3D
import Mathlib.Data.Real.Basic

noncomputable section
namespace Physics

/-- Euclidean norm squared for a right-associated triple `ℝ × (ℝ × ℝ)`. -/
def norm2 (v : ℝ × ℝ × ℝ) : ℝ := v.1^2 + v.2.1^2 + v.2.2^2

/-- Toy line element from the paper: ds² = S² c² dt² − S⁻² ‖dx‖². -/
def ds2 (F : Field3D) (c : ℝ)
    (x : ℝ × ℝ × ℝ) (t : ℝ) (dt : ℝ) (dx : ℝ × ℝ × ℝ) : ℝ :=
  let s := F.S x t
  (s^2) * (c^2) * (dt^2) - (s⁻¹)^2 * (norm2 dx)

/-- Refractive index n(x,t) := S(x,t)^{-2}. -/
def refractiveIndex (F : Field3D) (x : ℝ × ℝ × ℝ) (t : ℝ) : ℝ :=
  1 / (F.S x t)

end Physics