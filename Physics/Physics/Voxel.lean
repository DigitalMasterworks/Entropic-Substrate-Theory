import Physics.Prelude
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

noncomputable section
namespace Physics

/-- A discrete voxel with a finite set of admissible microstates and a chosen occupied subset. -/
structure Voxel (α : Type*) [DecidableEq α] where
  microstates      : Finset α
  occupied         : Finset α
  occupied_subset  : occupied ⊆ microstates
  nonempty_states  : 0 < microstates.card

namespace Voxel

variable {α : Type*} [DecidableEq α]

/-- Normalized entropy S := (#occupied) / (#microstates). -/
def S (v : Voxel α) : ℝ :=
  (v.occupied.card : ℝ) / (v.microstates.card : ℝ)

/-- Collapse potential C := 1 - S. -/
def C (v : Voxel α) : ℝ := 1 - v.S

@[simp] theorem S_add_C (v : Voxel α) : v.S + v.C = 1 := by
  simp [S, C]

/-- 0 ≤ S. -/
theorem S_nonneg (v : Voxel α) : 0 ≤ v.S := by
  have hden : 0 < (v.microstates.card : ℝ) := by exact_mod_cast v.nonempty_states
  have hnum : 0 ≤ (v.occupied.card : ℝ) := by exact_mod_cast (Nat.zero_le _)
  simpa [S] using (div_nonneg hnum (le_of_lt hden))

/-- S ≤ 1. -/
theorem S_le_one (v : Voxel α) : v.S ≤ 1 := by
  have hden_pos : 0 < (v.microstates.card : ℝ) := by exact_mod_cast v.nonempty_states
  have hden_nonneg : 0 ≤ (v.microstates.card : ℝ) := le_of_lt hden_pos
  -- #occupied ≤ #microstates (from subset)
  have hnum_le_den_nat : v.occupied.card ≤ v.microstates.card :=
    Finset.card_mono v.occupied_subset
  have hnum_le_den : (v.occupied.card : ℝ) ≤ (v.microstates.card : ℝ) := by
    exact_mod_cast hnum_le_den_nat
  -- divide both sides by the same nonnegative denominator
  have hdiv : (v.occupied.card : ℝ) / (v.microstates.card : ℝ)
              ≤ (v.microstates.card : ℝ) / (v.microstates.card : ℝ) :=
    div_le_div_of_nonneg_right hnum_le_den hden_nonneg
  have hden_ne : (v.microstates.card : ℝ) ≠ 0 := ne_of_gt hden_pos
  have hR : (v.microstates.card : ℝ) / (v.microstates.card : ℝ) = 1 := by
    simpa using (div_self hden_ne)
  -- rewrite to `v.S ≤ 1`
  have : v.S ≤ (v.microstates.card : ℝ) / (v.microstates.card : ℝ) := by simpa [S] using hdiv
  simpa [hR] using this

/-- 0 ≤ S ≤ 1. -/
theorem S_mem_Icc (v : Voxel α) : 0 ≤ v.S ∧ v.S ≤ 1 :=
  ⟨v.S_nonneg, v.S_le_one⟩

/-- Special case: if exactly one microstate is occupied, then `S = 1 / (#microstates)`. -/
theorem S_singleton (v : Voxel α) (h : v.occupied.card = 1) :
    v.S = (1 : ℝ) / (v.microstates.card : ℝ) := by
  have h1 : (v.occupied.card : ℝ) = 1 := by exact_mod_cast h
  simp [S, h1]

end Voxel
end Physics