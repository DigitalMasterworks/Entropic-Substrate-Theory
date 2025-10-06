import Mathlib.Algebra.Ring.Rat
import Mathlib.Analysis.Matrix
import Mathlib.Tactic

namespace Physics
namespace YangMillsReverse

/-- Principal index set used in analysis. -/
def H_wilson_n2_IDX : List Nat := [0, 1, 2, 3, 4, 5, 9, 10, 11]

def det_1 : Rat := 1
theorem det_1_def : det_1 = 1 := by rfl
theorem det_1_pos : 0 < det_1 := by
  norm_num

def det_2 : Rat := 1
theorem det_2_def : det_2 = 1 := by rfl
theorem det_2_pos : 0 < det_2 := by
  norm_num

def det_3 : Rat := 1
theorem det_3_def : det_3 = 1 := by rfl
theorem det_3_pos : 0 < det_3 := by
  norm_num

def det_4 : Rat := (3 : Rat) / (4 : Rat)
theorem det_4_def : det_4 = (3 : Rat) / (4 : Rat) := by rfl
theorem det_4_pos : 0 < det_4 := by
  show 0 < (3 : Rat) / (4 : Rat)
  apply div_pos <;> norm_num

def det_5 : Rat := (9 : Rat) / (16 : Rat)
theorem det_5_def : det_5 = (9 : Rat) / (16 : Rat) := by rfl
theorem det_5_pos : 0 < det_5 := by
  show 0 < (9 : Rat) / (16 : Rat)
  apply div_pos <;> norm_num

def det_6 : Rat := (27 : Rat) / (64 : Rat)
theorem det_6_def : det_6 = (27 : Rat) / (64 : Rat) := by rfl
theorem det_6_pos : 0 < det_6 := by
  show 0 < (27 : Rat) / (64 : Rat)
  apply div_pos <;> norm_num

def det_7 : Rat := (9 : Rat) / (32 : Rat)
theorem det_7_def : det_7 = (9 : Rat) / (32 : Rat) := by rfl
theorem det_7_pos : 0 < det_7 := by
  show 0 < (9 : Rat) / (32 : Rat)
  apply div_pos <;> norm_num

def det_8 : Rat := (3 : Rat) / (16 : Rat)
theorem det_8_def : det_8 = (3 : Rat) / (16 : Rat) := by rfl
theorem det_8_pos : 0 < det_8 := by
  show 0 < (3 : Rat) / (16 : Rat)
  apply div_pos <;> norm_num

def det_9 : Rat := (1 : Rat) / (8 : Rat)
theorem det_9_def : det_9 = (1 : Rat) / (8 : Rat) := by rfl
theorem det_9_pos : 0 < det_9 := by
  show 0 < (1 : Rat) / (8 : Rat)
  apply div_pos <;> norm_num


end YangMillsReverse
end Physics
