-- Physics/Hodge/Algebraicity.lean
import Mathlib.Data.Set.Basic

namespace Physics
namespace HodgeAlgebraicity

abbrev SmoothProjectiveVariety := Type
abbrev CohomologyClass := Type
abbrev Codim := Type

axiom Hpp_Q :
  SmoothProjectiveVariety → Codim → Set CohomologyClass

axiom AlgebraicCycles :
  SmoothProjectiveVariety → Codim → Set CohomologyClass

axiom QSpan : Set CohomologyClass → Set CohomologyClass

axiom Hodge_Algebraicity :
  ∀ (X : SmoothProjectiveVariety) (p : Codim),
    Hpp_Q X p = QSpan (AlgebraicCycles X p)

end HodgeAlgebraicity
end Physics
