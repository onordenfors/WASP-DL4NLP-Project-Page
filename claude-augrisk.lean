import Mathlib

open MeasureTheory

namespace AugRisk

/-- Ordinary risk: expected loss under the data distribution `μ`. -/
noncomputable def risk {Θ Z : Type*} [MeasurableSpace Z]
    (ℓ : Θ → Z → ℝ) (μ : Measure Z) (θ : Θ) : ℝ :=
  ∫ z, ℓ θ z ∂μ

/-- Augmented risk: average over the group of the risk on transformed data. -/
noncomputable def augRisk (G : Type*) {Θ Z : Type*} [Group G] [Fintype G]
    [MulAction G Z] [MeasurableSpace Z]
    (ℓ : Θ → Z → ℝ) (μ : Measure Z) (θ : Θ) : ℝ :=
  (Fintype.card G : ℝ)⁻¹ * ∑ g : G, ∫ z, ℓ θ (g • z) ∂μ

/-- Group-orbit average of a function on the parameter space. -/
noncomputable def orbitAvg (G : Type*) {Θ : Type*} [Group G] [Fintype G]
    [MulAction G Θ] (f : Θ → ℝ) (θ : Θ) : ℝ :=
  (Fintype.card G : ℝ)⁻¹ * ∑ g : G, f (g • θ)

/-- The augmented risk equals the orbit average of the risk over the parameters. -/
theorem augRisk_eq_orbitAvg (G : Type*) {Θ Z : Type*} [Group G] [Fintype G]
    [MulAction G Θ] [MulAction G Z] [MeasurableSpace Z]
    (ℓ : Θ → Z → ℝ) (μ : Measure Z)
    (hinv : ∀ (g : G) (θ : Θ) (z : Z), ℓ (g • θ) (g • z) = ℓ θ z)
    (θ : Θ) :
    augRisk G ℓ μ θ = orbitAvg G (risk ℓ μ) θ := by
  unfold augRisk orbitAvg risk
  have h1 : ∀ g : G,
      ∫ z, ℓ θ (g • z) ∂μ = ∫ z, ℓ (g⁻¹ • θ) z ∂μ := by
    intro g
    have hfun : (fun z => ℓ θ (g • z)) = fun z => ℓ (g⁻¹ • θ) z := by
      funext z
      have h := hinv g⁻¹ θ (g • z)
      rw [inv_smul_smul] at h
      exact h.symm
    rw [hfun]
  rw [Finset.sum_congr rfl (fun g _ => h1 g)]
  congr 1
  exact Fintype.sum_equiv (Equiv.inv G)
    (fun g => ∫ z, ℓ (g⁻¹ • θ) z ∂μ)
    (fun g => ∫ z, ℓ (g • θ) z ∂μ)
    (fun g => rfl)

end AugRisk
