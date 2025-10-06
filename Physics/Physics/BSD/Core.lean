-- Physics/BSD/Core.lean
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Basic

namespace Physics
namespace BSDCore

axiom EllipticCurve : Type
abbrev Complex' := ℂ

axiom Lfun   : EllipticCurve → ℂ → ℂ
axiom detE   : EllipticCurve → ℂ → ℂ
axiom rank   : EllipticCurve → ℕ
axiom orderAt : (ℂ → ℂ) → ℂ → ℕ

abbrev oneC : ℂ := 1

axiom L_equals_det :
  ∀ (E : EllipticCurve) (s : ℂ), Lfun E s = detE E s

axiom BSD_rank :
  ∀ E : EllipticCurve, orderAt (fun s => Lfun E s) oneC = rank E

end BSDCore
end Physics