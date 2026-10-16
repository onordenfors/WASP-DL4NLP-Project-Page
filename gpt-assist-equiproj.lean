import Mathlib

open InnerProductSpace

variable {𝕜 E : Type*}
variable [RCLike 𝕜]
variable [NormedAddCommGroup E]
variable [InnerProductSpace 𝕜 E]

variable (K : Submodule 𝕜 E)
variable [K.HasOrthogonalProjection]

#check Submodule.orthogonalProjectionOnto

#check K.subtype

#check ContinuousLinearMap.comp

#check K.subtype.toContinuousLinearMap

noncomputable def projAmbient
    (K : Submodule 𝕜 E)
    [K.HasOrthogonalProjection] :
    E →L[𝕜] E :=
  K.subtype.toContinuousLinearMap.comp
    (Submodule.orthogonalProjectionOnto K)
