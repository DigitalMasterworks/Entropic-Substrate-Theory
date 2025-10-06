-- Physics/Riemann/CriticalLine.lean
import Mathlib.Data.Complex.Basic
import Mathlib.Tactic  -- for linarith/nlinarith
namespace Physics
namespace RiemannReverse

/-- If `Im (s*(1-s))=0` then `Re(s)=1/2` or `Im(s)=0`. -/
lemma real_quadratic_slice'
  (s : ℂ) (hIm : (s * (1 - s)).im = 0) :
  s.re = (1/2 : ℝ) ∨ s.im = 0 := by
  -- write s = x + i y
  set x := s.re
  set y := s.im
  have hx : s = Complex.mk x y := by
    rcases s with ⟨x', y'⟩; simp [x, y] at *; rfl
  -- compute imaginary part: Im(s*(1-s)) = y * (1 - 2x)
  have him : (s * (1 - s)).im = y * (1 - 2*x) := by
    simp [hx, Complex.mul_def]
  -- conclude y*(1-2x)=0
  have h0 : y * (1 - 2*x) = 0 := by simpa [him] using hIm
  -- thus y=0 or 1-2x=0
  rcases mul_eq_zero.mp h0 with hY | hLin
  · right; simpa [y] using hY
  · left
    have : 2*x = 1 := by linarith
    have hx' : x = (1/2 : ℝ) := by nlinarith
    simpa [x, hx'] 

/-- If zeros of `detSpec` arise from λ = s(1−s) ∈ ℝ≥0 (self-adjoint spectrum),
    then any zero is either real or on the critical line. -/
theorem critical_line_from_selfadjoint
  (hH : SelfAdjointSymm H)
  (hZeroModel :
    ∀ s, detSpec s = 0 → ∃ λ : ℝ, 0 ≤ λ ∧ (s * (1 - s) : ℂ) = (λ : ℂ)) :
  ∀ {s}, detSpec s = 0 → s.re = (1/2 : ℝ) ∨ s.im = 0 := by
  intro s hs
  rcases hZeroModel s hs with ⟨λ, hλ, hmap⟩
  have hIm : (s * (1 - s)).im = 0 := by simpa [hmap, Complex.ofReal_im]
  exact real_quadratic_slice' s hIm

end RiemannReverse
end Physics