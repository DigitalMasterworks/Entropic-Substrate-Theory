-- Physics/Hodge/Bridge.lean
import Physics.Hodge.Algebraicity

namespace Physics
namespace HodgeReverse

open HodgeAlgebraicity

theorem Clay_Hodge :
  ∀ (X : SmoothProjectiveVariety) (p : Codim),
    Hpp_Q X p = QSpan (AlgebraicCycles X p) :=
  Hodge_Algebraicity

end HodgeReverse
end Physics