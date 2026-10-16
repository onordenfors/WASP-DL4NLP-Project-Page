import Mathlib

open MeasureTheory
open scoped ENNReal

noncomputable section

namespace GaussInit

/-! ### Radial measures on a finite-dimensional inner product space are isometry invariant -/

/-- A measure `volume.withDensity (fun x => φ ‖x‖)` is invariant under linear isometry
equivalences. -/
theorem map_withDensity_radial
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
    [MeasurableSpace E] [BorelSpace E]
    (φ : ℝ → ℝ≥0∞) (hφ : Measurable φ) (f : E ≃ₗᵢ[ℝ] E) :
    (volume.withDensity fun x : E => φ ‖x‖).map f
      = volume.withDensity fun x : E => φ ‖x‖ := by
  have hg : Measurable fun x : E => φ ‖x‖ := hφ.comp continuous_norm.measurable
  ext s hs
  rw [Measure.map_apply f.continuous.measurable hs,
    withDensity_apply _ (f.continuous.measurable hs), withDensity_apply _ hs]
  have := (f.measurePreserving).setLIntegral_comp_preimage hs hg
  simpa only [LinearIsometryEquiv.norm_map] using this

/-! ### Restricting an isometry to an invariant subspace -/

section Restrict

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

/-- Restriction of a linear isometry equivalence to a subspace it maps into itself,
as a linear isometry. -/
def restrictIsometry (f : V ≃ₗᵢ[ℝ] V) (W : Submodule ℝ V) (h : ∀ w ∈ W, f w ∈ W) :
    W →ₗᵢ[ℝ] W where
  toFun := fun w => ⟨f w, h w w.2⟩
  map_add' := fun x y => Subtype.ext (by simp)
  map_smul' := fun c x => Subtype.ext (by simp)
  norm_map' := fun x => f.norm_map (x : V)

omit [FiniteDimensional ℝ V] in
theorem coe_restrictIsometry (f : V ≃ₗᵢ[ℝ] V) (W : Submodule ℝ V)
    (h : ∀ w ∈ W, f w ∈ W) (w : W) :
    ((restrictIsometry f W h w : W) : V) = f w := rfl

/-- In finite dimensions, the restriction is an isometric *equivalence* of `W`. -/
def restrictEquiv (f : V ≃ₗᵢ[ℝ] V) (W : Submodule ℝ V) (h : ∀ w ∈ W, f w ∈ W) :
    W ≃ₗᵢ[ℝ] W :=
  (restrictIsometry f W h).toLinearIsometryEquiv rfl

theorem coe_restrictEquiv (f : V ≃ₗᵢ[ℝ] V) (W : Submodule ℝ V)
    (h : ∀ w ∈ W, f w ∈ W) (w : W) :
    ((restrictEquiv f W h w : W) : V) = f w := by
  first
    | exact coe_restrictIsometry f W h w

omit [FiniteDimensional ℝ V] in
/-- An isometry preserving an affine subspace `A` and fixing a point `p ∈ A`
preserves the direction of `A`. -/
theorem direction_invariant (f : V ≃ₗᵢ[ℝ] V) (A : AffineSubspace ℝ V)
    (hA : ∀ a ∈ A, f a ∈ A) {p : V} (hpA : p ∈ A) (hp : f p = p) :
    ∀ w ∈ A.direction, f w ∈ A.direction := by
  intro w hw
  obtain ⟨a, ha, rfl⟩ := (AffineSubspace.mem_direction_iff_eq_vsub_right hpA w).1 hw
  refine (AffineSubspace.mem_direction_iff_eq_vsub_right hpA _).2 ⟨f a, hA a ha, ?_⟩
  simp [vsub_eq_sub, hp]

end Restrict

/-! ### Gaussian initialization on an affine parameter space -/

section Gaussian

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
  [MeasurableSpace V] [BorelSpace V]

/-- Normalizing constant `(2πσ²)^(-d/2)` of a centered Gaussian in dimension `d`. -/
def gaussConst (d : ℕ) (σ : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal ((2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2))

/-- Radial profile `c · exp(-r² / (2σ²))` of the Gaussian density. -/
def gaussianDensity (σ : ℝ) (c : ℝ≥0∞) (r : ℝ) : ℝ≥0∞ :=
  c * ENNReal.ofReal (Real.exp (-(r ^ 2) / (2 * σ ^ 2)))

theorem measurable_gaussianDensity (σ : ℝ) (c : ℝ≥0∞) :
    Measurable (gaussianDensity σ c) := by
  unfold gaussianDensity
  exact measurable_const.mul
    (Measurable.ennreal_ofReal
      (Real.measurable_exp.comp (((measurable_id.pow_const 2).neg).div_const _)))

/-- The centered Gaussian `N(0, σ² Id)` on a subspace `W`, as a density w.r.t. the
Lebesgue (Haar) measure of the inner product space `W`. -/
def gaussianOn (W : Submodule ℝ V) (σ : ℝ) : Measure W :=
  (volume : Measure W).withDensity fun w : W =>
    gaussianDensity σ (gaussConst (Module.finrank ℝ W) σ) ‖w‖

/-- Gaussian initialization of the affine parameter space `A`, centered at `p ∈ A`:
`θ = p + w` with `w ~ N(0, σ² Id)` on the direction space of `A`. -/
def gaussInit (A : AffineSubspace ℝ V) (p : V) (σ : ℝ) : Measure V :=
  (gaussianOn A.direction σ).map (fun w : A.direction => p + (w : V))

/-- **Main theorem.** Let `ρ` be a unitary (orthogonal) representation of a group `G` on `V`,
`A` an affine parameter space invariant under `ρ`, and `p ∈ A` a fixed element.
Then the Gaussian initialization of `A` centered at `p` is `ρ`-invariant. -/
theorem gaussInit_invariant {G : Type*} [Group G] (ρ : G →* (V ≃ₗᵢ[ℝ] V))
    (A : AffineSubspace ℝ V) (hA : ∀ g : G, ∀ a ∈ A, ρ g a ∈ A)
    {p : V} (hpA : p ∈ A) (hp : ∀ g : G, ρ g p = p) (σ : ℝ) (g : G) :
    (gaussInit A p σ).map (ρ g) = gaussInit A p σ := by
  have hinv : ∀ w ∈ A.direction, ρ g w ∈ A.direction :=
    direction_invariant (ρ g) A (hA g) hpA (hp g)
  obtain ⟨T, hT⟩ : ∃ T : A.direction ≃ₗᵢ[ℝ] A.direction,
      ∀ w : A.direction, ((T w : A.direction) : V) = ρ g w :=
    ⟨restrictEquiv (ρ g) A.direction hinv, coe_restrictEquiv (ρ g) A.direction hinv⟩
  have hmeas_s : Measurable (fun w : A.direction => p + (w : V)) :=
    (continuous_const.add continuous_subtype_val).measurable
  have hcomm : (ρ g) ∘ (fun w : A.direction => p + (w : V))
      = (fun w : A.direction => p + (w : V)) ∘ T := by
    funext w
    simp only [Function.comp_apply]
    rw [hT w, map_add, hp g]
  have hTinv : (gaussianOn A.direction σ).map T = gaussianOn A.direction σ :=
    map_withDensity_radial
      (gaussianDensity σ (gaussConst (Module.finrank ℝ A.direction) σ))
      (measurable_gaussianDensity σ _) T
  unfold gaussInit
  rw [Measure.map_map (ρ g).continuous.measurable hmeas_s, hcomm,
    ← Measure.map_map hmeas_s T.continuous.measurable, hTinv]

end Gaussian

end GaussInit
