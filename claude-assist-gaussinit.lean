import Mathlib

open MeasureTheory ProbabilityTheory
open scoped ENNReal

noncomputable section

namespace GaussianInvariance

/-- Euclidean space `ℝ^p`. -/
abbrev E (p : ℕ) := EuclideanSpace ℝ (Fin p)

/-- Density of the standard Gaussian `N(0, I_p)` with respect to Lebesgue measure. -/
def gdens (p : ℕ) : E p → ℝ≥0∞ :=
  fun x => ENNReal.ofReal ((2 * Real.pi) ^ (-(p : ℝ) / 2) * Real.exp (-‖x‖ ^ 2 / 2))

/-- The standard Gaussian measure on `ℝ^p` (iid `N(0,1)` coordinates). -/
def stdGauss (p : ℕ) : Measure (E p) :=
  (volume : Measure (E p)).withDensity (gdens p)

lemma measurable_gdens (p : ℕ) : Measurable (gdens p) := by
  unfold gdens
  exact Measurable.ennreal_ofReal (Continuous.measurable (by fun_prop))

lemma gdens_map {p : ℕ} (U : E p ≃ₗᵢ[ℝ] E p) (x : E p) :
    gdens p (U x) = gdens p x := by
  simp [gdens, U.norm_map]

/-- The standard Gaussian is invariant under orthogonal maps. -/
lemma stdGauss_map_eq {p : ℕ} (U : E p ≃ₗᵢ[ℝ] E p) :
    (stdGauss p).map U = stdGauss p := by
  have hUm : Measurable U := U.continuous.measurable
  have hU : MeasurePreserving U (volume : Measure (E p)) volume := U.measurePreserving
  ext s hs
  unfold stdGauss
  rw [Measure.map_apply hUm hs, withDensity_apply _ (hUm hs), withDensity_apply _ hs,
    ← lintegral_indicator (hUm hs), ← lintegral_indicator hs]
  have h : ∀ x, (U ⁻¹' s).indicator (gdens p) x = s.indicator (gdens p) (U x) := by
    intro x
    by_cases hx : U x ∈ s
    · simp [Set.indicator, Set.mem_preimage, hx, gdens_map]
    · simp [Set.indicator, Set.mem_preimage, hx]
  refine (lintegral_congr h).trans ?_
  exact hU.lintegral_comp ((measurable_gdens p).indicator hs)

/-- If `ρg` is an isometry of `H` mapping the range of an isometric embedding `L`
into itself, then it acts on `ℝ^p` through an orthogonal map `U`: `ρg ∘ L = L ∘ U`. -/
lemma exists_orthogonal {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    {p : ℕ} (ρg : H ≃ₗᵢ[ℝ] H) (L : E p →ₗᵢ[ℝ] H)
    (h : ∀ x, ∃ y, ρg (L x) = L y) :
    ∃ U : E p ≃ₗᵢ[ℝ] E p, ∀ x, ρg (L x) = L (U x) := by
  choose f hf using h
  have hadd : ∀ x y, f (x + y) = f x + f y := by
    intro x y
    apply L.injective
    simp only [map_add, ← hf]
  have hsmul : ∀ (c : ℝ) (x : E p), f (c • x) = c • f x := by
    intro c x
    apply L.injective
    simp only [map_smul, ← hf]
  have hnorm : ∀ x, ‖f x‖ = ‖x‖ := by
    intro x
    calc ‖f x‖ = ‖L (f x)‖ := (L.norm_map _).symm
      _ = ‖ρg (L x)‖ := by rw [hf]
      _ = ‖L x‖ := ρg.norm_map _
      _ = ‖x‖ := L.norm_map _
  let Fl : E p →ₗ[ℝ] E p :=
    { toFun := f
      map_add' := hadd
      map_smul' := fun c x => hsmul c x }
  have hinj : Function.Injective Fl := by
    rw [← LinearMap.ker_eq_bot, LinearMap.ker_eq_bot']
    intro x hx
    have hx' : f x = 0 := hx
    have h0 : ‖x‖ = 0 := by rw [← hnorm x, hx', norm_zero]
    exact norm_eq_zero.mp h0
  have hsurj : Function.Surjective Fl := LinearMap.injective_iff_surjective.mp hinj
  exact ⟨{ toLinearEquiv := LinearEquiv.ofBijective Fl ⟨hinj, hsurj⟩
           norm_map' := fun x => hnorm x }, fun x => hf x⟩

/-- Main theorem: `θ = A + L c` with `c` standard Gaussian is `G`-invariant in distribution. -/
theorem gaussian_invariant_in_distribution
    {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H] [FiniteDimensional ℝ H]
    [MeasurableSpace H] [BorelSpace H]
    {G : Type*} [Group G]
    (ρ : G →* (H ≃ₗᵢ[ℝ] H))
    (p : ℕ)
    (𝓛 : AffineSubspace ℝ H) (A : H)
    (L : E p →ₗᵢ[ℝ] H) -- `L*L = id`, range `L` = tangent space
    (h𝓛 : (𝓛 : Set H) = Set.range (fun x : E p => A + L x))   -- `𝓛 = A + L ℝ^p`
    (hfix : ∀ g : G, ρ g A = A)               -- `A` is a fixed point
    (hinv : ∀ g : G, ∀ y ∈ 𝓛, ρ g y ∈ 𝓛)      -- `ρ(g) 𝓛 ⊆ 𝓛`
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (c : Ω → E p) (hcm : Measurable c) (hc : P.map c = stdGauss p) :
    ∀ g : G, P.map (fun ω => ρ g (A + L (c ω))) = P.map (fun ω => A + L (c ω)) := by
  intro g
  -- `ρ g` preserves the range of `L`
  have h : ∀ x, ∃ y, ρ g (L x) = L y := by
    intro x
    have hmem : A + L x ∈ (𝓛 : Set H) := by
      rw [h𝓛]; exact ⟨x, rfl⟩
    have h2 : ρ g (A + L x) ∈ (𝓛 : Set H) := hinv g _ hmem
    rw [h𝓛] at h2
    obtain ⟨y, hy⟩ := h2
    refine ⟨y, ?_⟩
    rw [map_add, hfix g] at hy
    exact (add_left_cancel hy).symm
  obtain ⟨U, hU⟩ := exists_orthogonal (ρ g) L h
  have hUm : Measurable U := U.continuous.measurable
  have hΦ : Measurable (fun x : E p => A + L x) :=
    (continuous_const.add L.continuous).measurable
  have hΦU : Measurable ((fun x : E p => A + L x) ∘ U) := hΦ.comp hUm
  have hfun : (fun ω => ρ g (A + L (c ω))) = ((fun x : E p => A + L x) ∘ U) ∘ c := by
    funext ω
    simp only [Function.comp_apply, map_add, hfix g, hU]
  calc P.map (fun ω => ρ g (A + L (c ω)))
      = P.map (((fun x : E p => A + L x) ∘ U) ∘ c) := by rw [hfun]
    _ = (P.map c).map ((fun x : E p => A + L x) ∘ U) := (Measure.map_map hΦU hcm).symm
    _ = ((stdGauss p).map U).map (fun x : E p => A + L x) := by
        rw [hc, Measure.map_map hΦ hUm]
    _ = (stdGauss p).map (fun x : E p => A + L x) := by rw [stdGauss_map_eq]
    _ = P.map ((fun x : E p => A + L x) ∘ c) := by
        rw [← hc]; exact Measure.map_map hΦ hcm
    _ = P.map (fun ω => A + L (c ω)) := rfl
end GaussianInvariance
