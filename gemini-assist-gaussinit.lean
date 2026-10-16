import Mathlib

set_option linter.unusedVariables false
set_option linter.style.whitespace false
set_option linter.unusedSectionVars false

section AffineInvariance

variable (H : Type*) [NormedAddCommGroup H] [NormedSpace ℝ H] [FiniteDimensional ℝ H]
  [MeasurableSpace H] [BorelSpace H]
variable (G : Type*) [Group G] [TopologicalSpace G] [CompactSpace G]

/-- An orthogonal representation of a group G on a real normed space H. -/
structure OrthogonalRepresentation (H : Type*) [NormedAddCommGroup H] [NormedSpace ℝ H]
    (G : Type*) [Group G] where
  toFun : G → (H ≃L[ℝ] H)
  map_mul' : ∀ g₁ g₂, toFun (g₁ * g₂) = toFun g₁ * toFun g₂
  map_one' : toFun (1 : G) = 1
  isometry' : ∀ g (x : H), ‖toFun g x‖ = ‖x‖

namespace OrthogonalRepresentation

variable {H G : Type*} [NormedAddCommGroup H] [NormedSpace ℝ H] [Group G]
  (ρ : OrthogonalRepresentation H G)

instance : CoeFun (OrthogonalRepresentation H G) (fun _ => G → (H ≃L[ℝ] H)) := ⟨toFun⟩

@[simp] lemma map_one (g : G) : ρ (1 : G) = 1 := ρ.map_one'

end OrthogonalRepresentation

variable (p : ℕ)
variable (μ : MeasureTheory.Measure (Fin p → ℝ))
variable (A : H)
variable (L_op : (Fin p → ℝ) →L[ℝ] H)

/--
  Theorem: If $A$ is fixed by $\rho(g)$, and the action of $\rho(g)$ on the range of $L_{\mathrm{op}}$
  corresponds to an orthogonal transformation $U_g$ on $\mathbb{R}^p$ preserving the measure $\mu$,
  then the distribution of $\theta = A + L c$ (where $c$ has law $\mu$) is invariant under $\rho(g)$.
-/
theorem invariant_gaussian_affine_subspace
    (ρ : OrthogonalRepresentation H G)
    (hA_fixed : ∀ g : G, ρ g A = A)
    (U : G → ((Fin p → ℝ) ≃L[ℝ] (Fin p → ℝ)))
    (h_comm : ∀ g (v : Fin p → ℝ), ρ g (L_op v) = L_op (U g v))
    (h_mu_inv : ∀ g, MeasureTheory.Measure.map (U g) μ = μ)
    (g : G) :
    MeasureTheory.Measure.map (fun v => ρ g (A + L_op v)) μ =
    MeasureTheory.Measure.map (fun v => A + L_op v) μ := by
  -- Step 1: Using linearity of ρ(g) and the fixed point property of A
  have h_action (v : Fin p → ℝ) : ρ g (A + L_op v) = A + L_op (U g v) := by
    have h_lin := (ρ g).map_add A (L_op v)
    rw [h_lin, hA_fixed g, h_comm]

  -- Step 2: Rewrite the pushforward measure using the algebraic identity
  have h_map_eq : MeasureTheory.Measure.map (fun v => ρ g (A + L_op v)) μ =
                  MeasureTheory.Measure.map (fun v => A + L_op (U g v)) μ := by
    congr 1
    ext v
    exact h_action v

  rw [h_map_eq]

  -- Step 3: Express the composite map as function composition
  have h_comp : (fun v => A + L_op (U g v)) = (fun w => A + L_op w) ∘ (U g) := rfl
  rw [h_comp]

  -- Step 4: Establish measurability of inner and outer functions
  have h_meas_inner : Measurable (U g) := (U g).continuous.measurable
  have h_meas_outer : Measurable (fun w => A + L_op w) :=
    (measurable_const_add A).comp L_op.measurable

  -- Step 5: Apply measure pushforward composition law (map_map) in reverse
  rw [← MeasureTheory.Measure.map_map h_meas_outer h_meas_inner]

  -- Step 6: Use the invariance of μ under the orthogonal transformation U g
  rw [h_mu_inv g]

end AffineInvariance
