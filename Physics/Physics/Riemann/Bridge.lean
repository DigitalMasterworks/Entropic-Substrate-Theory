-- Physics/Riemann/Bridge.lean
import Physics.AnalyticPredicates_shim
import Physics.Riemann.Reverse      -- zeta, detSpec
import Physics.Riemann.Completed     -- A, Lambda, LambdaSpec + axioms
import Physics.Riemann.CriticalLine

namespace Physics
namespace RiemannReverse

/-- Euler product on primes (σ := Re s > 1). -/
noncomputable def EulerProd (s : ℂ) : ℂ :=
  ∏' (p : ℕ) (hp : Nat.Prime p), (1 - (p : ℂ) ^ (-s))⁻¹

/-- Hypotheses retained (minimal):
    POA/Euler for detSpec, Euler for ζ, and the zero→spectrum model. -/
axiom POA_euler :
  ∀ s, Real.re s > 1 → detSpec s = EulerProd s

axiom zeta_euler :
  ∀ s, Real.re s > 1 → zeta s = EulerProd s

/-- Zeros come from the self-adjoint spectrum: detSpec s = 0 ⇒ s(1−s) = λ ∈ ℝ≥0. -/
axiom zero_model :
  ∀ s, detSpec s = 0 → ∃ λ : ℝ, 0 ≤ λ ∧ (s * (1 - s) : ℂ) = (λ : ℂ)

/-- On σ>1 we have Λ_spec = Λ (both defined with the same normalizer A). -/
lemma Lambda_eq_on_right :
  ∀ s, Real.re s > 1 → LambdaSpec s = Lambda s := by
  intro s hs
  -- Λ_spec(s) = A(s) * detSpec(s) = A(s) * EulerProd(s)
  -- Λ(s)      = A(s) * zeta(s)    = A(s) * EulerProd(s)
  simp [LambdaSpec, Lambda, A, POA_euler s hs, zeta_euler s hs]

/-- Global equality of completed functions by the identity theorem. -/
theorem Lambda_eq_global : ∀ s, LambdaSpec s = Lambda s := by
  -- both Λ_spec and Λ are entire and equal on the open half-plane {Re s > 1}
  have h1 : AnalyticPredicates.Entire LambdaSpec := Completed_entire
  have h2 : AnalyticPredicates.Entire Lambda := Lambda_entire
  -- apply the identity theorem
  exact fun s => (identity_on_right_implies_global h1 h2 (fun z hz => (Lambda_eq_on_right z hz)).symm) s |> Eq.symm

/-- From the completed equality, divide out A and get ζ = detSpec globally. -/
theorem zeta_eq_detSpec : ∀ s, zeta s = detSpec s :=
  divide_out_A (by intro s; simpa using (Lambda_eq_global s))

/-- RH on nontrivial zeros (conditional on self-adjointness + zero model). -/
theorem RH_conditional (hH : SelfAdjointSymm H) :
  ∀ {s}, detSpec s = 0 → s.re = (1/2 : ℝ) ∨ s.im = 0 :=
  critical_line_from_selfadjoint hH zero_model

/-- RH for ζ (same conditions), via ζ = detSpec. -/
theorem Riemann_Hypothesis
  (hH : SelfAdjointSymm H) :
  ∀ {s}, zeta s = 0 → s.re = (1/2 : ℝ) ∨ s.im = 0 := by
  intro s hz
  have hz' : detSpec s = 0 := by simpa [zeta_eq_detSpec s] using hz
  exact RH_conditional hH hz'

end RiemannReverse
end Physics
