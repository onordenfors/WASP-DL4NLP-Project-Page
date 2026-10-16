import Mathlib

open scoped InnerProductSpace

variable
  {𝕜 E G : Type*}
  [RCLike 𝕜]
  [Group G]
  [NormedAddCommGroup E]
  [InnerProductSpace 𝕜 E]

def projE
    (K : Submodule 𝕜 E)
    [K.HasOrthogonalProjection] :
    E →L[𝕜] E :=
  K.subtypeL.comp (Submodule.orthogonalProjection K)

  theorem commutes_implies_invariant
    (K : Submodule 𝕜 E)
    [K.HasOrthogonalProjection]
    (U : E ≃ₗᵢ[𝕜] E)
    (hcomm :
      projE K ∘L U.toContinuousLinearMap =
      U.toContinuousLinearMap ∘L projE K) :
    ∀ x ∈ K, U x ∈ K := by
  intro x hx

  have hxproj : projE K x = x := by
    -- likely requires a mathlib lemma:
    -- orthogonalProjection_eq_self_of_mem
    sorry

  have h1 :
      projE K (U x) = U x := by
    calc
      projE K (U x)
          = U (projE K x) := by
              -- from hcomm
              sorry
      _ = U x := by simpa [hxproj]

  -- fixed points of projection are exactly points of K
  -- another standard mathlib lemma needed here
  sorry

theorem invariant_implies_commutes
    (K : Submodule 𝕜 E)
    [K.HasOrthogonalProjection]
    (U : E ≃ₗᵢ[𝕜] E)
    (hK : ∀ x ∈ K, U x ∈ K) :
    projE K ∘L U.toContinuousLinearMap =
    U.toContinuousLinearMap ∘L projE K := by
  ext x
  -- decompose x into projection + orthogonal part
  -- use U preserving orthogonality because U is a linear isometry
  -- show projection(Ux)=U(projection x)
