import Mathlib.Algebra.Module.Equiv.Basic

/-!
# Equivariance of neural networks with equivariant activations

Setup (0-indexed version of the problem statement).

* `V 0 = X`, ..., `V (n+1) = Y`; so `L = n + 2` spaces.
* There are `n + 1` linear layers `θ i : V i →ₗ[K] V (i+1)` for `i = 0, ..., n`
  (these are the `θ_1, ..., θ_{L-1}` of the problem) and `n` activations
  `σ i : V (i+1) → V (i+1)` for `i = 0, ..., n-1`
  (these are the `σ_1, ..., σ_{L-2}` of the problem).
* The parameter space is `Θ = ∏ i, Hom(V i, V (i+1))`, a `K`-module.
* A representation of `G` on `W` is a group homomorphism `G →* Module.End K W`
  (it automatically lands in the units `GL(W)`, because `g * g⁻¹ = 1`).
  The family `ρ i : G →* Module.End K (V i)` has `ρ 0 = ρ_X`, `ρ (n+1) = ρ_Y`
  and `ρ i` for the intermediate spaces.
* The network is
  `f_θ = θ_n ∘ σ_{n-1} ∘ θ_{n-1} ∘ ⋯ ∘ σ_0 ∘ θ_0`.

We construct the representation `thetaRep ρ : G →* Module.End K Θ`,
`(g • θ) i = ρ (i+1) g ∘ θ i ∘ ρ i g⁻¹`, and prove

`f_θ (ρ_X g x) = ρ_Y g (f_{g⁻¹ • θ} x)`.
-/

namespace EquivariantNN

variable {K : Type*} [CommRing K]
variable {G : Type*} [Group G]

/-- The parameter space `Θ` of a network with `n + 1` linear layers:
the product of the spaces of linear maps `V i →ₗ V (i+1)`, `i = 0, …, n`. -/
abbrev Theta (K : Type*) [CommRing K] (V : ℕ → Type*) [∀ i, AddCommGroup (V i)]
    [∀ i, Module K (V i)] (n : ℕ) : Type _ :=
  ∀ i : Fin (n + 1), V i.val →ₗ[K] V (i.val + 1)

variable {V : ℕ → Type*} [∀ i, AddCommGroup (V i)] [∀ i, Module K (V i)]

/-- Forward pass up to (and including) the `k`-th linear layer:
`forward θ σ k _ x = θ_k (σ_{k-1} (θ_{k-1} (⋯ σ_0 (θ_0 x))))`. -/
def forward {n : ℕ} (θ : Theta K V n) (σ : ∀ i : ℕ, V (i + 1) → V (i + 1)) :
    (k : ℕ) → k < n + 1 → V 0 → V (k + 1)
  | 0, h, x => θ ⟨0, h⟩ x
  | k + 1, h, x => θ ⟨k + 1, h⟩ (σ k (forward θ σ k (Nat.lt_of_succ_lt h) x))

/-- The neural network `f_θ : X → Y` (with `X = V 0`, `Y = V (n+1)`). -/
def network {n : ℕ} (θ : Theta K V n) (σ : ∀ i : ℕ, V (i + 1) → V (i + 1)) (x : V 0) :
    V (n + 1) :=
  forward θ σ n (Nat.lt_succ_self n) x

section Action

variable (ρ : ∀ i : ℕ, G →* Module.End K (V i))

/-- Action of `g` on a single layer: `A ↦ ρ_{i+1}(g) ∘ A ∘ ρ_i(g)⁻¹`. -/
def actLayer (i : ℕ) (g : G) (A : V i →ₗ[K] V (i + 1)) : V i →ₗ[K] V (i + 1) :=
  (ρ (i + 1) g) ∘ₗ A ∘ₗ (ρ i g⁻¹)

/-- The action of `g` on `Θ`, as a linear map. -/
def actTheta {n : ℕ} (g : G) : Theta K V n →ₗ[K] Theta K V n where
  toFun θ := fun i => actLayer ρ i.val g (θ i)
  map_add' θ θ' := by
    funext i
    ext x
    simp [actLayer]
  map_smul' c θ := by
    funext i
    ext x
    simp [actLayer]

/-- The representation `ρ` of `G` on the parameter space `Θ`. -/
def thetaRep (n : ℕ) : G →* Module.End K (Theta K V n) where
  toFun g := actTheta ρ g
  map_one' := by
    ext θ i x
    simp [actTheta, actLayer]
  map_mul' g h := by
    ext θ i x
    simp [actTheta, actLayer]

theorem rep_inv_apply {W : Type*} [AddCommGroup W] [Module K W]
    (r : G →* Module.End K W) (g : G) (x : W) : r g⁻¹ (r g x) = x := by
  rw [← Module.End.mul_apply, ← map_mul, inv_mul_cancel, map_one, Module.End.one_apply]

theorem rep_apply_inv {W : Type*} [AddCommGroup W] [Module K W]
    (r : G →* Module.End K W) (g : G) (x : W) : r g (r g⁻¹ x) = x := by
  rw [← Module.End.mul_apply, ← map_mul, mul_inv_cancel, map_one, Module.End.one_apply]

variable {ρ}

/-- Layer-by-layer intertwining: running the network with the transformed parameters
`g • θ` on `ρ_X g x` gives `ρ_{k+1} g` applied to the `k`-th pre-activation of `θ` on `x`. -/
theorem forward_act {n : ℕ} (σ : ∀ i : ℕ, V (i + 1) → V (i + 1))
    (hσ : ∀ i, i < n → ∀ (g : G) (y : V (i + 1)), σ i (ρ (i + 1) g y) = ρ (i + 1) g (σ i y))
    (g : G) (θ : Theta K V n) :
    ∀ (k : ℕ) (hk : k < n + 1) (x : V 0),
      forward (thetaRep ρ n g θ) σ k hk (ρ 0 g x) = ρ (k + 1) g (forward θ σ k hk x) := by
  intro k
  induction k with
  | zero =>
    intro hk x
    simp only [forward]
    show actLayer ρ 0 g (θ ⟨0, hk⟩) (ρ 0 g x) = ρ (0 + 1) g (θ ⟨0, hk⟩ x)
    simp only [actLayer, LinearMap.comp_apply, rep_inv_apply]
  | succ k ih =>
    intro hk x
    have hkn : k < n := Nat.lt_of_succ_lt_succ hk
    simp only [forward]
    show actLayer ρ (k + 1) g (θ ⟨k + 1, hk⟩)
        (σ k (forward (thetaRep ρ n g θ) σ k (Nat.lt_of_succ_lt hk) (ρ 0 g x))) =
      ρ (k + 1 + 1) g (θ ⟨k + 1, hk⟩ (σ k (forward θ σ k (Nat.lt_of_succ_lt hk) x)))
    rw [ih (Nat.lt_of_succ_lt hk) x, hσ k hkn g]
    simp only [actLayer, LinearMap.comp_apply, rep_inv_apply]

end Action

variable {ρ : ∀ i : ℕ, G →* Module.End K (V i)}

/-- **Main theorem.** If every activation `σ i` is equivariant, then
`f_θ (ρ_X g x) = ρ_Y g (f_{ρ(g⁻¹) θ} x)`, where `ρ_X = ρ 0`, `ρ_Y = ρ (n+1)` and
`ρ(g) θ = thetaRep ρ n g θ` is the representation of `G` on `Θ`. -/
theorem network_equivariant {n : ℕ} (σ : ∀ i : ℕ, V (i + 1) → V (i + 1))
    (hσ : ∀ i, i < n → ∀ (g : G) (y : V (i + 1)), σ i (ρ (i + 1) g y) = ρ (i + 1) g (σ i y))
    (g : G) (θ : Theta K V n) (x : V 0) :
    network θ σ (ρ 0 g x) = ρ (n + 1) g (network (thetaRep ρ n g⁻¹ θ) σ x) := by
  have h := forward_act σ hσ g⁻¹ θ n (Nat.lt_succ_self n) (ρ 0 g x)
  rw [rep_inv_apply] at h
  unfold network
  rw [h, rep_apply_inv]
end EquivariantNN
