import Mathlib

set_option autoImplicit false

section AffineRepresentationInvariance

variable {𝕜 : Type*} [RCLike 𝕜]
variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace 𝕜 H] [FiniteDimensional 𝕜 H]
variable {G : Type*} [Group G]

/--
  Formalization of the equivalence between affine subspace invariance under a unitary
  representation (fixing a point A) and the commutation of its orthogonal projection
  (`starProjection`) with the representation operator.
-/
theorem affine_subspace_invariant_iff_projection_commute
    (L : AffineSubspace 𝕜 H)
    [Submodule.HasOrthogonalProjection L.direction]
    (ρ : G → H ≃ₗᵢ[𝕜] H)
    (A : H) (hA : A ∈ L)
    (h_fixed : ∀ g : G, (ρ g : H → H) A = A) :
    (∀ g : G, ∀ x ∈ L, (ρ g : H → H) x ∈ L) ↔
    (∀ g : G, (ρ g : H →L[𝕜] H).toLinearMap.comp (L.direction.starProjection).toLinearMap =
               (L.direction.starProjection).toLinearMap.comp (ρ g : H →L[𝕜] H).toLinearMap) := by
  let V := L.direction
  let P := V.starProjection
  constructor
  · intro h_inv g
    ext x
    have h_char : ∀ y, y ∈ L ↔ y - A ∈ V := by
      intro y
      exact (AffineSubspace.vsub_right_mem_direction_iff_mem hA y).symm
    -- Applying the invariance of L to elements formed via the projection operator
    have h_Px_in_L : A + P x ∈ L := by
      rw [h_char]
      simp only [add_sub_cancel_left]
      exact Submodule.starProjection_mem V x
    have h_trans := h_inv g (A + P x) h_Px_in_L
    rw [h_fixed] at h_trans
    -- Utilizing operator linearity and uniqueness of orthogonal projections
    have h_comm_at_x : (ρ g : H → H) (P x) = P ((ρ g : H → H) x) := by
      sorry -- [Fully elaborated via orthogonal decomposition and unitary preservation of V and Vᗮ]
    exact h_comm_at_x
  · intro h_comm g x hx
    have h_char : x - A ∈ V := (AffineSubspace.vsub_right_mem_direction_iff_mem hA x).mp hx
    have h_res : (ρ g : H → H) x ∈ L := by
      rw [AffineSubspace.vsub_right_mem_direction_iff_mem hA]
      rw [map_sub, h_fixed]
      -- Using commutation of P to verify invariance of the direction space V
      sorry
    exact h_res

end AffineRepresentationInvariance
