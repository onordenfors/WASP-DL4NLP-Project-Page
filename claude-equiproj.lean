import Mathlib

open Submodule

namespace AffineInvariance

variable {𝕜 E : Type*} [RCLike 𝕜] [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]

/-- For a point `p` of an affine subspace `A`, `x ∈ A ↔ x - p ∈ A.direction`. -/
lemma mem_iff_sub_mem_direction (A : AffineSubspace 𝕜 E) {p : E} (hp : p ∈ A) (x : E) :
    x ∈ A ↔ x - p ∈ A.direction := by
  rw [← AffineSubspace.vsub_right_mem_direction_iff_mem hp x, vsub_eq_sub]

/-- Step 1: for a unitary `U` fixing `p ∈ A`, the affine subspace `A` is mapped onto itself
iff the tangent space `A.direction` is invariant (in both directions). -/
lemma image_eq_iff_direction_invariant (A : AffineSubspace 𝕜 E) {p : E} (hp : p ∈ A)
    (U : E ≃ₗᵢ[𝕜] E) (hU : U p = p) :
    (⇑U '' (A : Set E) = (A : Set E)) ↔ ∀ v, U v ∈ A.direction ↔ v ∈ A.direction := by
  have key := mem_iff_sub_mem_direction A hp
  constructor
  · intro h v
    constructor
    · intro hv
      -- `U v ∈ V → v ∈ V`
      have h1 : U (p + v) ∈ (A : Set E) :=
        (key _).2 (by simpa [map_add, hU] using hv)
      rw [← h] at h1
      obtain ⟨y, hy, hyeq⟩ := h1
      have hy' : y = p + v := U.injective hyeq
      subst hy'
      simpa using (key _).1 hy
    · intro hv
      -- `v ∈ V → U v ∈ V`
      have h1 : p + v ∈ A := (key _).2 (by simpa using hv)
      have h2 : U (p + v) ∈ ⇑U '' (A : Set E) := ⟨p + v, h1, rfl⟩
      rw [h] at h2
      have h3 : U (p + v) - p ∈ A.direction := (key _).1 h2
      simpa [map_add, hU] using h3
  · intro h
    ext x
    constructor
    · rintro ⟨y, hy, rfl⟩
      refine (key _).2 ?_
      have h1 : U y - p = U (y - p) := by rw [map_sub, hU]
      rw [h1, h]
      exact (key y).1 hy
    · intro hx
      refine ⟨U.symm x, ?_, U.apply_symm_apply x⟩
      refine (key _).2 ?_
      have h1 : x - p ∈ A.direction := (key x).1 hx
      have h2 : U (U.symm x - p) = x - p := by rw [map_sub, hU, U.apply_symm_apply]
      refine (h _).1 ?_
      rw [h2]
      exact h1

/-- Step 2: a unitary `U` preserves a closed subspace `V` (in both directions)
iff it commutes with the orthogonal projection onto `V`. -/
lemma invariant_iff_commutes (V : Submodule 𝕜 E) [V.HasOrthogonalProjection]
    (U : E ≃ₗᵢ[𝕜] E) :
    (∀ v, U v ∈ V ↔ v ∈ V) ↔ ∀ x, V.starProjection (U x) = U (V.starProjection x) := by
  constructor
  · intro h x
    have hmem : U (V.starProjection x) ∈ V := (h _).2 (V.starProjection_apply_mem x)
    refine Submodule.eq_starProjection_of_mem_of_inner_eq_zero hmem ?_
    intro w hw
    obtain ⟨w', rfl⟩ : ∃ w', U w' = w := ⟨U.symm w, U.apply_symm_apply w⟩
    have hw' : w' ∈ V := (h w').1 hw
    rw [← map_sub U x (V.starProjection x), LinearIsometryEquiv.inner_map_map]
    exact Submodule.starProjection_inner_eq_zero x w' hw'
  · intro h v
    constructor
    · intro hv
      -- `U v ∈ V → v ∈ V`
      have h1 : V.starProjection (U v) = U v :=
        Submodule.starProjection_eq_self_iff.mpr hv
      rw [h v] at h1
      have h2 : V.starProjection v = v := U.injective h1
      exact Submodule.starProjection_eq_self_iff.mp h2
    · intro hv
      -- `v ∈ V → U v ∈ V`
      have h1 : V.starProjection v = v := Submodule.starProjection_eq_self_iff.mpr hv
      have h2 : V.starProjection (U v) = U v := by rw [h v, h1]
      exact Submodule.starProjection_eq_self_iff.mp h2

/-- **Main theorem.** Let `A` be an affine subspace (the affine parameter space) with a fixed
element `p ∈ A` (`U p = p`), and `U` unitary. Then `A` is invariant under `U` iff the orthogonal
projection onto the tangent space `A.direction` commutes with `U`. -/
theorem affine_invariant_iff_projection_commutes
    (A : AffineSubspace 𝕜 E) [A.direction.HasOrthogonalProjection]
    {p : E} (hp : p ∈ A) (U : E ≃ₗᵢ[𝕜] E) (hU : U p = p) :
    (⇑U '' (A : Set E) = (A : Set E)) ↔
      ∀ x, A.direction.starProjection (U x) = U (A.direction.starProjection x) :=
  (image_eq_iff_direction_invariant A hp U hU).trans
    (invariant_iff_commutes A.direction U)

/-- The same statement for a unitary representation `ρ : G →* (E ≃ₗᵢ[𝕜] E)` whose
action fixes `p`: `A` is invariant under every `ρ g` iff the projection commutes with all `ρ g`. -/
theorem affine_invariant_iff_projection_commutes_rep
    {G : Type*} [Group G] (ρ : G →* (E ≃ₗᵢ[𝕜] E))
    (A : AffineSubspace 𝕜 E) [A.direction.HasOrthogonalProjection]
    {p : E} (hp : p ∈ A) (hfix : ∀ g, ρ g p = p) :
    (∀ g, ⇑(ρ g) '' (A : Set E) = (A : Set E)) ↔
      ∀ g x, A.direction.starProjection (ρ g x) = ρ g (A.direction.starProjection x) :=
  forall_congr' fun g =>
    affine_invariant_iff_projection_commutes A hp (ρ g) (hfix g)

end AffineInvariance
