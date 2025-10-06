-- Physics/Riemann/OperatorCore.lean
import Mathlib.Data.Complex.Basic
import Mathlib.Analysis.InnerProductSpace.Adjoint

namespace Physics
namespace RiemannReverse

abbrev Operator (E : Type*)
  [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E] :=
  E →L[ℂ] E

axiom H : Operator ℂ

def SelfAdjointSymm {E : Type*}
  [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E]
  (T : Operator E) : Prop :=
  T = ContinuousLinearMap.adjoint T

axiom SelfAdjointSymm_H : SelfAdjointSymm H

end RiemannReverse
end Physics