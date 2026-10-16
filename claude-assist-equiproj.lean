import Mathlib

open scoped InnerProductSpace

/-- Let `L` be an affine subspace of a finite-dimensional inner product space `H`
(real or complex), let `G` be a compact group acting unitarily on `H` via `ρ`, and let
`A ∈ L` be fixed by all of `G`.  Then `L` is invariant under every `ρ g` iff the orthogonal
projection `Π` onto the direction (tangent space) of `L` commutes with every `ρ g`. -/
theorem affine_invariant_iff_commute_projection
    {𝕜 : Type*} [RCLike 𝕜]
    {H : Type*} [NormedAddCommGroup H] [InnerProductSpace 𝕜 H] [FiniteDimensional 𝕜 H]
    (L : AffineSubspace 𝕜 H)
    {G : Type*} [Group G] [TopologicalSpace G] [IsTopologicalGroup G] [CompactSpace G]
    (ρ : G →* (H ≃ₗᵢ[𝕜] H)) (A : H) (hA : A ∈ L) (hfix : ∀ g, ρ g A = A) :
    (∀ g, (ρ g) '' (L : Set H) ⊆ (L : Set H)) ↔
      (∀ g x, ρ g (L.direction.starProjection x) = L.direction.starProjection (ρ g x)) := by
  -- membership in `L` in terms of the direction, using the base point `A`
  have hmem : ∀ x, x ∈ L ↔ x - A ∈ L.direction := fun x => by
    rw [← AffineSubspace.vsub_right_mem_direction_iff_mem hA x, vsub_eq_sub]
  -- basic facts about the orthogonal projection
  have hPmem : ∀ x, L.direction.starProjection x ∈ L.direction := fun x => by
    first
    | exact Submodule.starProjection_apply_mem _ _
  have hPorth : ∀ x, ∀ w ∈ L.direction,
      inner 𝕜 (x - L.direction.starProjection x) w = 0 := fun x w hw => by
    first
    | exact Submodule.starProjection_inner_eq_zero _ _ _
    | exact Submodule.starProjection_inner_eq_zero x w hw
  have hPself : ∀ y ∈ L.direction, L.direction.starProjection y = y := fun y hy => by
    first
    | exact Submodule.starProjection_eq_self_iff.2 hy
  -- ρ g ∘ ρ g⁻¹ = id
  have hρρ : ∀ g (w : H), ρ g (ρ g⁻¹ w) = w := fun g w => by
    have h1 : ρ g * ρ g⁻¹ = 1 := by rw [← map_mul, mul_inv_cancel, map_one]
    have h2 := congrArg (fun f : H ≃ₗᵢ[𝕜] H => f w) h1
    first
    | exact h2
  have hsubA : ∀ g x, ρ g (x - A) = ρ g x - A := fun g x => by
    rw [map_sub, hfix]
  constructor
  · -- (⇒) invariance of `L` implies commutation with the projection
    intro h g
    have hmapW : ∀ g, ∀ w ∈ L.direction, ρ g w ∈ L.direction := by
      intro g w hw
      have h1 : A + w ∈ L := by
        rw [hmem, add_sub_cancel_left]; exact hw
      have h2 : ρ g (A + w) ∈ L := h g (Set.mem_image_of_mem _ h1)
      rw [hmem, map_add, hfix, add_sub_cancel_left] at h2
      exact h2
    intro x
    symm
    apply Submodule.eq_starProjection_of_mem_of_inner_eq_zero
    · exact hmapW g _ (hPmem x)
    · intro w hw
      have hw' : ρ g⁻¹ w ∈ L.direction := hmapW g⁻¹ w hw
      have h0 := hPorth x _ hw'
      have h3 : inner 𝕜 (ρ g (x - L.direction.starProjection x)) (ρ g (ρ g⁻¹ w))
          = inner 𝕜 (x - L.direction.starProjection x) (ρ g⁻¹ w) := by
        first
        | exact LinearIsometryEquiv.inner_map_map _ _ _
      rw [hρρ g w] at h3
      have e : ρ g x - ρ g (L.direction.starProjection x)
          = ρ g (x - L.direction.starProjection x) := (map_sub (ρ g) x _).symm
      rw [e, h3]
      exact h0
  · -- (⇐) commutation with the projection implies invariance of `L`
    intro hc g
    rintro _ ⟨x, hx, rfl⟩
    have h1 : x - A ∈ L.direction := (hmem x).1 hx
    have h2 : L.direction.starProjection (x - A) = x - A := hPself _ h1
    have h3 : L.direction.starProjection (ρ g (x - A)) = ρ g (x - A) := by
      rw [← hc g (x - A), h2]
    have h4 : ρ g (x - A) ∈ L.direction := by
      have h5 := hPmem (ρ g (x - A))
      rwa [h3] at h5
    rw [hsubA] at h4
    exact (hmem _).2 h4
