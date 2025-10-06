import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Tactic

open Submodule

variable {𝕜 : Type*} [RCLike 𝕜]
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]

/--
For closed submodules with orthogonal projections, equality of the submodules
is equivalent to equality of their projection operators.
-/
theorem orth_proj_eq_iff_range_eq
  {K L : Submodule 𝕜 E} [K.HasOrthogonalProjection] [L.HasOrthogonalProjection] :
  (K.starProjection : E →ₗ[𝕜] E) = (L.starProjection : E →ₗ[𝕜] E) ↔ K = L := by
  constructor
  · intro h
    -- if the operators are equal, their ranges are equal
    have : LinearMap.range (K.starProjection : E →ₗ[𝕜] E) =
           LinearMap.range (L.starProjection : E →ₗ[𝕜] E) := by rw [h]
    -- mathlib lemma: the range of starProjection is the submodule itself
    exact (Submodule.range_starProjection K).symm.trans
      (this.trans (Submodule.range_starProjection L))
  · intro h
    subst h
    rfl