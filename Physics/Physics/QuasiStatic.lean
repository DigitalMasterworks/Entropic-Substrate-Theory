import Physics.Prelude
import Physics.Field3D

noncomputable section
namespace Physics

/-- Quasi-static model: fix a time `t0`, density `ρ`, constant `κ`,
and an abstract Laplacian `Δ`, together with Poisson law `Δ C = κ ρ` at `t0`. -/
structure QuasiStatic (F : Field3D) where
  t0      : ℝ
  κ       : ℝ
  ρ       : (ℝ × ℝ × ℝ) → ℝ
  Δ       : ((ℝ × ℝ × ℝ) → ℝ) → ((ℝ × ℝ × ℝ) → ℝ)
  poisson : ∀ x, Δ (fun y => F.C y t0) x = κ * ρ x

end Physics
