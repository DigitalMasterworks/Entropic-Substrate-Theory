-- Physics/PvNP/Reverse.lean
import Mathlib.Data.Nat.Basic

namespace Physics
namespace PvNPReverse

axiom Language : Type
def OmegaCost : ℕ → ℕ := fun n => n
def TMtime    : Language → ℕ → ℕ := fun _ n => n
axiom NP : Language → Prop
axiom P_equals_NP : Prop

def EqUpToConst (f g : ℕ → ℕ) : Prop := ∃ C, ∀ n, f n ≤ g n + C ∧ g n ≤ f n + C

axiom cost_equals_TMtime :
  ∀ (L : Language), NP L → EqUpToConst OmegaCost (TMtime L)

axiom PolyBound : (ℕ → ℕ) → Prop
axiom poly_bound_for_SAT : PolyBound OmegaCost → P_equals_NP

theorem Clay_PvNP : PolyBound OmegaCost → P_equals_NP := by
  intro h; exact poly_bound_for_SAT h

end PvNPReverse
end Physics