-- Physics/CSEntropy.lean
import Physics.Prelude
import Mathlib.Tactic
import Mathlib.Data.Real.Basic

namespace Physics

/-- A minimal “substrate” with two weights summing to 1. -/
structure Substrate where
  C : ℝ
  S : ℝ
  hCS : C + S = 1

namespace Substrate

lemma C_eq (σ : Substrate) : σ.C = 1 - σ.S := by
  have := σ.hCS
  linarith

lemma S_eq (σ : Substrate) : σ.S = 1 - σ.C := by
  have := σ.hCS
  linarith

/-- Linear mixing identity useful for rearrangements: `a*C + b*S`. -/
lemma mix_rewrite (σ : Substrate) (a b : ℝ) :
    a*σ.C + b*σ.S = b + (a - b)*σ.C := by
  have h := σ.C_eq
  calc
    a*σ.C + b*σ.S
        = a*σ.C + b*(1 - σ.C) := by simpa [h]
    _   = a*σ.C + b - b*σ.C   := by ring
    _   = b + (a - b)*σ.C     := by ring

/-- Symmetric version swapping `C ↔ S`. -/
lemma mix_rewrite' (σ : Substrate) (a b : ℝ) :
    a*σ.C + b*σ.S = a + (b - a)*σ.S := by
  have h := σ.S_eq
  calc
    a*σ.C + b*σ.S
        = a*(1 - σ.S) + b*σ.S := by simpa [h]
    _   = a + (b - a)*σ.S     := by ring

end Substrate
end Physics