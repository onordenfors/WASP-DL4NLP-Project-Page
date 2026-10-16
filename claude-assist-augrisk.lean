import Mathlib

open MeasureTheory

namespace EquivariantAugmentation

/-- Forward pass: `fwd X σ θ x m : X (m+1)` is the output after `m+1` linear layers,
with activation `σ i` applied after layer `i` (none after the last one used). -/
def fwd {K : Type*} [Field K] (X : ℕ → Type*) [∀ i, AddCommGroup (X i)]
    [∀ i, Module K (X i)]
    (σ : ∀ i, X (i + 1) → X (i + 1)) (θ : ∀ i, X i →ₗ[K] X (i + 1)) (x : X 0) :
    (n : ℕ) → X (n+1)
  | 0 => θ 0 x
  | n+1 => θ (n+1) (σ n (fwd X σ θ x n))

/-- Conjugation action of `G` on the parameters: `(g • θ)_i = ρ_{i+1}(g) ∘ θ_i ∘ ρ_i(g)⁻¹`. -/
def act {K : Type*} [Field K] {G : Type*} [Group G] (X : ℕ → Type*)
    [∀ i, AddCommGroup (X i)] [∀ i, Module K (X i)]
    (ρ : ∀ i, Representation K G (X i)) (g : G) (θ : ∀ i, X i →ₗ[K] X (i + 1)) :
    ∀ i, X i →ₗ[K] X (i+1) :=
  fun i => (ρ (i+1) g) ∘ₗ (θ i) ∘ₗ (ρ i g⁻¹)

section

variable {K : Type*} [Field K] {G : Type*} [Group G]

theorem rep_inv_apply {V : Type*} [AddCommGroup V] [Module K V]
    (ρ : Representation K G V) (g : G) (v : V) : ρ g⁻¹ (ρ g v) = v := by
  have h : ρ g⁻¹ * ρ g = 1 :=
    (map_mul ρ g⁻¹ g).symm.trans (by simp)
  calc ρ g⁻¹ (ρ g v) = (ρ g⁻¹ * ρ g) v := rfl
    _ = (1 : Module.End K V) v := by rw [h]
    _ = v := rfl

theorem rep_apply_inv {V : Type*} [AddCommGroup V] [Module K V]
    (ρ : Representation K G V) (g : G) (v : V) : ρ g (ρ g⁻¹ v) = v := by
  have h := rep_inv_apply ρ g⁻¹ v
  rwa [inv_inv] at h

variable (X : ℕ → Type*) [∀ i, AddCommGroup (X i)] [∀ i, Module K (X i)]

/-- Equivariance of the network: `f_{g•θ}(ρ_X(g) x) = ρ_Y(g) f_θ(x)`. -/
theorem fwd_act (ρ : ∀ i, Representation K G (X i)) (σ : ∀ i, X (i + 1) → X (i + 1))
    (g : G) (θ : ∀ i, X i →ₗ[K] X (i + 1)) (x : X 0) :
    ∀ m : ℕ,
      (∀ i, i < m → ∀ (h : G) (v : X (i + 1)), σ i (ρ (i + 1) h v) = ρ (i + 1) h (σ i v)) →
      fwd X σ (act X ρ g θ) (ρ 0 g x) m = ρ (m+1) g (fwd X σ θ x m) := by
  intro m
  induction m with
  | zero =>
    intro _
    have h1 : fwd X σ (act X ρ g θ) (ρ 0 g x) 0
        = ρ 1 g (θ 0 (ρ 0 g⁻¹ (ρ 0 g x))) := rfl
    refine h1.trans ?_
    rw [rep_inv_apply (ρ 0)]
    rfl
  | succ m ih =>
    intro hσ
    have ih' := ih (fun i hi => hσ i (Nat.lt_succ_of_lt hi))
    have hm := hσ m (Nat.lt_succ_self m) g (fwd X σ θ x m)
    calc fwd X σ (act X ρ g θ) (ρ 0 g x) (m+1)
        = (act X ρ g θ (m+1)) (σ m (fwd X σ (act X ρ g θ) (ρ 0 g x) m)) := rfl
      _ = (act X ρ g θ (m+1)) (σ m (ρ (m+1) g (fwd X σ θ x m))) := by rw [ih']
      _ = (act X ρ g θ (m+1)) (ρ (m+1) g (σ m (fwd X σ θ x m))) := by rw [hm]
      _ = ρ (m+1+1) g (θ (m+1) (ρ (m+1) g⁻¹ (ρ (m+1) g (σ m (fwd X σ θ x m))))) := rfl
      _ = ρ (m+1+1) g (θ (m+1) (σ m (fwd X σ θ x m))) := by rw [rep_inv_apply (ρ (m+1))]
      _ = ρ (m+1+1) g (fwd X σ θ x (m+1)) := rfl

/-- Nominal risk `R(θ) = 𝔼 ℓ(f_θ(x), y)` for a network with `n+1` linear layers. -/
noncomputable def risk (n : ℕ) [MeasurableSpace (X 0)] [MeasurableSpace (X (n + 1))]
    (μ : Measure (X 0 × X (n + 1))) (ℓ : X (n + 1) → X (n + 1) → ℝ)
    (σ : ∀ i, X (i + 1) → X (i + 1)) (θ : ∀ i, X i →ₗ[K] X (i + 1)) : ℝ :=
  ∫ p, ℓ (fwd X σ θ p.1 n) p.2 ∂μ

/-- Augmented risk `(1/|G|) ∑_g 𝔼 ℓ(f_θ(ρ_X(g) x), ρ_Y(g) y)`. -/
noncomputable def augRisk [Fintype G] (n : ℕ) [MeasurableSpace (X 0)]
    [MeasurableSpace (X (n + 1))] (ρ : ∀ i, Representation K G (X i))
    (μ : Measure (X 0 × X (n + 1))) (ℓ : X (n + 1) → X (n + 1) → ℝ)
    (σ : ∀ i, X (i + 1) → X (i + 1)) (θ : ∀ i, X i →ₗ[K] X (i + 1)) : ℝ :=
  (1 / (Fintype.card G : ℝ)) *
    ∑ g : G, ∫ p, ℓ (fwd X σ θ (ρ 0 g p.1) n) (ρ (n+1) g p.2) ∂μ

/-- Main theorem: `R^aug(θ) = (1/|G|) ∑_g R(ρ(g) θ)`. -/
theorem augRisk_eq [Fintype G] (n : ℕ) [MeasurableSpace (X 0)]
    [MeasurableSpace (X (n + 1))] (ρ : ∀ i, Representation K G (X i))
    (μ : Measure (X 0 × X (n + 1))) (ℓ : X (n + 1) → X (n + 1) → ℝ)
    (σ : ∀ i, X (i + 1) → X (i + 1))
    (hσ : ∀ i, i < n → ∀ (g : G) (v : X (i + 1)), σ i (ρ (i + 1) g v) = ρ (i + 1) g (σ i v))
    (hℓ : ∀ (g : G) (a b : X (n + 1)), ℓ (ρ (n + 1) g a) (ρ (n + 1) g b) = ℓ a b)
    (θ : ∀ i, X i →ₗ[K] X (i + 1)) :
    augRisk X n ρ μ ℓ σ θ
      = (1 / (Fintype.card G : ℝ)) * ∑ g : G, risk X n μ ℓ σ (act X ρ g θ) := by
  have key : ∀ g : G,
      (∫ p, ℓ (fwd X σ θ (ρ 0 g p.1) n) (ρ (n+1) g p.2) ∂μ)
        = risk X n μ ℓ σ (act X ρ g⁻¹ θ) := by
    intro g
    unfold risk
    congr 1
    funext p
    have h := fwd_act X ρ σ g⁻¹ θ (ρ 0 g p.1) n hσ
    rw [rep_inv_apply (ρ 0)] at h
    rw [h]
    have h2 := hℓ g (ρ (n+1) g⁻¹ (fwd X σ θ (ρ 0 g p.1) n)) p.2
    rw [rep_apply_inv (ρ (n+1))] at h2
    exact h2
  have hsum : (∑ g : G, ∫ p, ℓ (fwd X σ θ (ρ 0 g p.1) n) (ρ (n+1) g p.2) ∂μ)
      = ∑ g : G, risk X n μ ℓ σ (act X ρ g θ) := by
    calc (∑ g : G, ∫ p, ℓ (fwd X σ θ (ρ 0 g p.1) n) (ρ (n+1) g p.2) ∂μ)
        = ∑ g : G, risk X n μ ℓ σ (act X ρ g⁻¹ θ) :=
          Finset.sum_congr rfl (fun g _ => key g)
      _ = ∑ g : G, risk X n μ ℓ σ (act X ρ g θ) :=
          Fintype.sum_equiv (Equiv.inv G) _ _ (fun g => rfl)
  unfold augRisk
  exact congrArg (fun s => (1 / (Fintype.card G : ℝ)) * s) hsum

end
end EquivariantAugmentation
