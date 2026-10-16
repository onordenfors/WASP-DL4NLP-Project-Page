import Mathlib

open Submodule

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/--
1. **Affine Invariance ⟺ Tangent Space Invariance**:
An affine parameter space `theta_0 + T` is invariant under a linear map `U`
(which fixes `theta_0`) if and only if its tangent space `T` is invariant under `U`.
-/
theorem affine_invariant_iff_tangent_invariant
    (U : E →L[ℝ] E) (theta_0 : E) (T : Submodule ℝ E)
    (h_fix : U theta_0 = theta_0) :
    (∀ y, (∃ v ∈ T, y = theta_0 + v) → (∃ v ∈ T, U y = theta_0 + v)) ↔
    (∀ v ∈ T, U v ∈ T) := by
  constructor
  · intro h_inv v hv
    have h_y : ∃ w ∈ T, theta_0 + v = theta_0 + w := ⟨v, hv, rfl⟩
    rcases h_inv (theta_0 + v) h_y with ⟨w, hw, h_U_eq⟩
    rw [map_add, h_fix] at h_U_eq
    have h_v_eq : U v = w := add_left_cancel h_U_eq
    rw [h_v_eq]
    exact hw
  · intro h_tan y ⟨v, hv, hy⟩
    use U v
    refine ⟨h_tan v hv, ?_⟩
    rw [hy, map_add, h_fix]

/--
2. **Projection Commutativity ⟹ Tangent Space Invariance**:
If the orthogonal projection `T.starProjection` commutes with `U`,
then the tangent space `T` is invariant under `U`.
-/
theorem tangent_invariant_of_starProjection_commute
    (U : E →L[ℝ] E) (T : Submodule ℝ E) [CompleteSpace ↥T] [T.HasOrthogonalProjection]
    (h_comm : ∀ x, T.starProjection (U x) = U (T.starProjection x)) :
    ∀ v ∈ T, U v ∈ T := by
  intro v hv
  have h_proj : T.starProjection v = v := by
    rw [starProjection_eq_self_iff]
    exact hv
  have h_eq := h_comm v
  have h_star : T.starProjection (U v) = U v := by
    calc T.starProjection (U v) = U (T.starProjection v) := h_eq
         _ = U v := by rw [h_proj]
  rw [starProjection_eq_self_iff] at h_star
  exact h_star

/--
3. **Tangent Space Invariance ⟹ Projection Commutativity**:
If the tangent space `T` and its orthogonal complement `T.orthogonal`
are both invariant under `U`, then `U` commutes with `T.starProjection`.
-/
theorem starProjection_commute_of_tangent_invariant
    (U : E →L[ℝ] E) (T : Submodule ℝ E) [CompleteSpace ↥T] [T.HasOrthogonalProjection]
    (h_inv_T : ∀ v ∈ T, U v ∈ T)
    (h_inv_perp : ∀ w ∈ T.orthogonal, U w ∈ T.orthogonal) :
    ∀ x, T.starProjection (U x) = U (T.starProjection x) := by
  intro x
  have h_dec : U x = U (T.starProjection x) + U (x - T.starProjection x) := by
    rw [← map_add, add_sub_cancel]
  have h_mem_T : U (T.starProjection x) ∈ T := by
    apply h_inv_T
    have h_apply : T.starProjection x = ↑(T.orthogonalProjectionOnto x) := starProjection_apply T x
    rw [h_apply]
    exact Submodule.coe_mem _
  have h_mem_perp : U (x - T.starProjection x) ∈ T.orthogonal := by
    apply h_inv_perp
    exact sub_starProjection_mem_orthogonal x
  have h_proj_sum := map_add T.starProjection (U (T.starProjection x)) (U (x - T.starProjection x))
  rw [← h_dec] at h_proj_sum
  rw [h_proj_sum]
  have h_proj_T : T.starProjection (U (T.starProjection x)) = U (T.starProjection x) := by
    rw [starProjection_eq_self_iff]
    exact h_mem_T
  have h_proj_perp : T.starProjection (U (x - T.starProjection x)) = 0 := by
    set w := U (x - T.starProjection x)
    have hw_perp : w ∈ T.orthogonal := h_mem_perp
    have hz_T : T.starProjection w ∈ T := by
      have h_apply : T.starProjection w = ↑(T.orthogonalProjectionOnto w) := starProjection_apply T w
      rw [h_apply]
      exact Submodule.coe_mem _
    have h_sub_perp : w - T.starProjection w ∈ T.orthogonal := sub_starProjection_mem_orthogonal w
    have hz_perp : T.starProjection w ∈ T.orthogonal := by
      have h_eq_w : T.starProjection w = w - (w - T.starProjection w) := by abel
      rw [h_eq_w]
      exact T.orthogonal.sub_mem hw_perp h_sub_perp
    have h_inner : ⟪T.starProjection w, T.starProjection w⟫_ℝ = 0 := by
      rw [Submodule.mem_orthogonal] at hz_perp
      exact hz_perp (T.starProjection w) hz_T
    exact inner_self_eq_zero.mp h_inner
  rw [h_proj_T, h_proj_perp]
  exact add_zero (U (T.starProjection x))
