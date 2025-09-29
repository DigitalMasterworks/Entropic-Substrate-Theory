import Physics.Prelude
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace Physics

/-- A physics substrate with two real-valued fields `S` and `C`,
    required to satisfy `S(x) + C(x) = 1` for all `x`. -/
structure Field where
  S : ℝ → ℝ
  C : ℝ → ℝ
  hSC : ∀ x, S x + C x = 1

namespace Field

/-- From the conservation law, you can recover `S` from `C`. -/
theorem S_eq (f : Field) : ∀ x, f.S x = 1 - f.C x := by
  intro x
  have := f.hSC x
  linarith

/-- Symmetrically, you can recover `C` from `S`. -/
theorem C_eq (f : Field) : ∀ x, f.C x = 1 - f.S x := by
  intro x
  have := f.hSC x
  linarith

end Field
end Physics