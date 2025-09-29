import Physics.Prelude
import Physics.Field3D
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

noncomputable section
namespace Physics

/-- Build a field from a Newtonian potential Φ via S = 1 + Φ/c². -/
def weakFieldFromΦ (Φ : (ℝ × ℝ × ℝ) → ℝ) (c : ℝ) : Field3D :=
  { S := fun x _t => 1 + (Φ x) / (c*c) }

/-- In that weak-field model, C = -Φ/c² exactly. -/
@[simp] theorem C_eq_negPhi_over_c2
    (Φ : (ℝ × ℝ × ℝ) → ℝ) (c : ℝ) (x : ℝ × ℝ × ℝ) (t : ℝ) :
    (weakFieldFromΦ Φ c).C x t = - (Φ x) / (c*c) := by
  -- 1 - (1 + Φ/c²) = -(Φ/c²) = -(Φ)/c²
  have : (weakFieldFromΦ Φ c).C x t = - ((Φ x) / (c*c)) := by
    simp [Field3D.C, weakFieldFromΦ, sub_eq_add_neg]
  simpa [neg_div] using this

/-- Standard identifications for the gravitational mapping (from the paper). -/
def alphaFrom_c (c : ℝ) : ℝ := c*c     -- α = c²

/-- κ = -4πG / c², with π provided as an argument (pass `Real.pi` from a file that imports it). -/
def kappaFrom_pi_G_c (π G c : ℝ) : ℝ := -4 * π * G / (c*c)

end Physics