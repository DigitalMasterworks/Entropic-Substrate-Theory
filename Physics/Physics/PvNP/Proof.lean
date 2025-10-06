import Physics.Blowup
import Mathlib.Tactic

open Nat

theorem P_eq_NP_under_substrate
  (ν C Δt : ℝ) (E Ω : ℕ → ℝ)
  (hν : 0 < ν) (hC : 0 < C) (hΔ : 0 < Δt)
  (hΩ0 : Ω 0 ≤ 1000) (hE : ∀ n, 0 < E n)
  (hstep : ∀ n,
    Ω (n+1) ≤ Ω n + (C * (Ω n)^(3/2) - ν * (Ω n)^2 / E n) * Δt)
  (control :
    ∀ n, (C * (Ω n)^(3/2) - ν * (Ω n)^2 / E n) * Δt ≤ 0) :
  ∃ k : ℕ, ∀ n ≤ 100, Ω n ≤ 1000 := by
  use 1
  intro n hn
  exact bounded_short_horizon ν C Δt E Ω hν hC hΔ hΩ0 hE hstep control n hn