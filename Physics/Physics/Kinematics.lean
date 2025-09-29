import Physics.Prelude
import Physics.Field3D
import Mathlib.Data.Real.Basic

noncomputable section
namespace Physics

/-- Scalar multiply an ℝ³ vector modeled as right-associated `ℝ × (ℝ × ℝ)`. -/
def smul3 (a : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a * v.1, (a * v.2.1, a * v.2.2))

/-- Kinematics on top of a field: α parameter and a gradient of C. -/
structure Kinematics (F : Field3D) where
  α     : ℝ
  gradC : (ℝ × ℝ × ℝ) → ℝ → (ℝ × ℝ × ℝ)

namespace Kinematics

/-- Acceleration law: a = α⋅∇C. -/
def a (K : Kinematics F) (x : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  smul3 K.α (K.gradC x t)

/-- Proper time scaling: dτ/dt = S(x,t). -/
def clockScale (K : Kinematics F) (x : ℝ × ℝ × ℝ) (t : ℝ) : ℝ :=
  F.S x t

@[simp] theorem accel_eq (K : Kinematics F) (x : ℝ × ℝ × ℝ) (t : ℝ) :
    K.a x t = smul3 K.α (K.gradC x t) := rfl

@[simp] theorem clock_eq_S (K : Kinematics F) (x : ℝ × ℝ × ℝ) (t : ℝ) :
    K.clockScale x t = F.S x t := rfl

end Kinematics
end Physics