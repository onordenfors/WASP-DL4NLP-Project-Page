import Mathlib

open MeasureTheory Real BigOperators

/-- The point-wise loss duality/equivariance condition:
    Evaluating loss on transformed data g • z equals
    evaluating loss on inverse-transformed parameters g⁻¹ • theta. -/
def LossDuality (G Theta Z : Type*) [Group G] [MulAction G Theta] [MulAction G Z]
    (loss : Theta → Z → ℝ) : Prop :=
  ∀ (theta : Theta) (z : Z) (g : G),
    loss theta (g • z) = loss (g⁻¹ • theta) z

/-- Pointwise data-augmented loss. -/
noncomputable def augmentedDataLoss (G Theta Z : Type*) [Group G] [Fintype G]
    [MulAction G Z] (loss : Theta → Z → ℝ) (theta : Theta) (z : Z) : ℝ :=
  (Fintype.card G : ℝ)⁻¹ * ∑ g : G, loss theta (g • z)

/-- Expected risk R(θ) under data distribution μ. -/
noncomputable def risk (Theta Z : Type*) [MeasurableSpace Z]
    (loss : Theta → Z → ℝ) (μ : Measure Z) (theta : Theta) : ℝ :=
  ∫ z, loss theta z ∂μ

/-- Augmented expected risk: expected value of the augmented data loss. -/
noncomputable def augmentedExpectedRisk (G Theta Z : Type*) [Group G] [Fintype G]
    [MeasurableSpace Z] [MulAction G Z]
    (loss : Theta → Z → ℝ) (μ : Measure Z) (theta : Theta) : ℝ :=
  ∫ z, augmentedDataLoss G Theta Z loss theta z ∂μ

/-- Group orbit average of the expected risk over the parameter space. -/
noncomputable def parameterOrbitAverageRisk (G Theta Z : Type*) [Group G] [Fintype G]
    [MulAction G Theta] [MeasurableSpace Z]
    (loss : Theta → Z → ℝ) (μ : Measure Z) (theta : Theta) : ℝ :=
  (Fintype.card G : ℝ)⁻¹ * ∑ g : G, risk Theta Z loss μ (g • theta)

/-- Main Theorem: Under loss duality and integrability, data-augmented
    expected risk equals parameter orbit average of the expected risk. -/
theorem augmented_expected_risk_equals_parameter_orbit_average
    (G Theta Z : Type*) [Group G] [Fintype G]
    [MeasurableSpace Z] [MulAction G Z] [MulAction G Theta]
    (loss : Theta → Z → ℝ) (μ : Measure Z) [IsProbabilityMeasure μ]
    (h_duality : LossDuality G Theta Z loss)
    (h_integ : ∀ theta : Theta, Integrable (loss theta) μ)
    (theta : Theta) :
    augmentedExpectedRisk G Theta Z loss μ theta =
      parameterOrbitAverageRisk G Theta Z loss μ theta := by
  classical
  unfold augmentedExpectedRisk parameterOrbitAverageRisk
  unfold augmentedDataLoss risk
  have h_sum_integ : ∀ g : G,
      Integrable (fun z => loss theta (g • z)) μ := by
    intro g
    have h_eq : (fun z => loss theta (g • z)) =
        (fun z => loss (g⁻¹ • theta) z) := by
      ext z
      exact h_duality theta z g
    rw [h_eq]
    exact h_integ (g⁻¹ • theta)
  calc
    ∫ z, (Fintype.card G : ℝ)⁻¹ *
        ∑ g : G, loss theta (g • z) ∂μ
      _ = (Fintype.card G : ℝ)⁻¹ *
          ∫ z, ∑ g : G, loss theta (g • z) ∂μ := by
          exact integral_mul_left (Fintype.card G : ℝ)⁻¹ _
      _ = (Fintype.card G : ℝ)⁻¹ *
          ∑ g : G, ∫ z, loss theta (g • z) ∂μ := by
          congr 1
          exact integral_finset_sum Finset.univ
            (fun g _ => h_sum_integ g)
      _ = (Fintype.card G : ℝ)⁻¹ *
          ∑ g : G, ∫ z, loss (g⁻¹ • theta) z ∂μ := by
          congr 2
          apply Finset.sum_congr rfl
          intro g _
          congr 1
          ext z
          exact h_duality theta z g
      _ = (Fintype.card G : ℝ)⁻¹ *
          ∑ g : G, ∫ z, loss (g • theta) z ∂μ := by
          congr 1
          apply Finset.sum_bij (fun g _ => g⁻¹)
          · intro g _
            exact Finset.mem_univ g⁻¹
          · intro g₁ _ g₂ _ h
            exact inv_injective h
          · intro g _
            use g⁻¹
            refine ⟨Finset.mem_univ g⁻¹, inv_inv g⟩
          · intro g _
            rw [inv_inv]
