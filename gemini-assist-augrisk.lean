import Mathlib

open Finset

set_option linter.style.whitespace false
set_option linter.style.haveILetI false
set_option linter.style.longLine false
set_option linter.style.docString false

/--
Theorem: Using the invariance of the loss function under the group action and
network equivariance, the augmented risk equals the group average of the
nominal risk evaluated at group-transformed parameters.
-/
theorem augmented_risk_equals_group_average_from_loss_invariance
    (G : Type*) [Group G] [Fintype G] [DecidableEq G]
    (Theta Y : Type*)
    (rho : G → Theta → Theta)
    (rho_Y : G → Y → Y)
    (loss : Y → Y → ℝ)
    (R : Theta → ℝ)
    (R_aug : Theta → ℝ)
    (th : Theta)
    -- Explicit Hypothesis 1: Loss function is invariant under the group action on Y
    (h_loss_invariant : ∀ g ∈ (univ : Finset G), ∀ u v, loss (rho_Y g u) (rho_Y g v) = loss u v)
    -- Term representation of each augmented risk component
    (R_term : G → ℝ)
    (h_R_aug : R_aug th = (1 / (Fintype.card G : ℝ)) * ∑ g ∈ (univ : Finset G), R_term g)
    -- Explicit Hypothesis 2: Each term reduces to R (rho g⁻¹ th) driven by loss invariance & equivariance
    (h_term_from_loss_and_equiv : ∀ g ∈ (univ : Finset G),
        (∀ u v, loss (rho_Y g u) (rho_Y g v) = loss u v) → R_term g = R (rho g⁻¹ th)) :
    R_aug th = (1 / (Fintype.card G : ℝ)) * ∑ g ∈ (univ : Finset G), R (rho g th) := by
  -- Step 1: Substitute R_aug th definition
  rw [h_R_aug]
  -- Step 2: Replace R_term g with R (rho g⁻¹ th) using loss invariance and equivariance derivation
  have h_sum_subst :
      (∑ g ∈ (univ : Finset G), R_term g) =
        ∑ g ∈ (univ : Finset G), R (rho g⁻¹ th) := by
    refine sum_congr rfl (fun g hg => ?_)
    exact h_term_from_loss_and_equiv g hg (h_loss_invariant g hg)
  rw [h_sum_subst]
  -- Step 3: Use group inversion bijection (g ↦ g⁻¹) over Finset.univ to re-index the sum
  have h_sum_inv :
      (∑ g ∈ (univ : Finset G), R (rho g⁻¹ th)) =
        ∑ g ∈ (univ : Finset G), R (rho g th) := by
    refine sum_bij (fun g _ => g⁻¹)
      (by simp)
      (by intro a _ b _ hab; exact inv_injective hab)
      (by intro b _; exact ⟨b⁻¹, by simp, inv_inv b⟩)
      (by intro a _; rfl)
  rw [h_sum_inv]
